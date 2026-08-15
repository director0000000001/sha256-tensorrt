"""Landmark-warp avatar engine.

Puts a portrait photograph onto the tracked face and deforms it, per frame, so
the avatar's mouth, eyes, and brows follow the user's. It runs no neural
network: the avatar's own landmarks are found once when it loads, a triangle
mesh is built between them, and every frame each triangle is warped from where
it sits in the photograph to where the matching landmark sits on the live face.

**What this is and is not.** It is a real, working avatar engine — the face
moves, blinks, and talks with you, and it is deterministic, cheap, and carries
**no model-weight licensing risk at all**, which no neural option currently
does (see ``docs/OPEN_SOURCE_EVALUATION.md``). It is *not* photorealistic
portrait synthesis: it reshapes existing pixels rather than generating new ones,
so extreme angles stretch rather than reveal, and the avatar cannot show what
the photograph does not contain. LivePortrait remains the target for quality;
this is the engine that exists today and the fallback if the neural path is too
slow, too heavy, or legally blocked.

Design notes
------------
*   **The mesh is decimated on purpose.** MediaPipe's full tessellation is
    roughly 900 triangles, and warping each one costs a separate OpenCV call —
    tens of milliseconds, most of it per-call overhead. A curated subset of
    landmarks covering the face oval, brows, eyes, nose, and lips gives a mesh
    an order of magnitude smaller that still moves everything a viewer watches.
*   **Everything that can be precomputed is.** The portrait's landmarks and the
    triangulation are built once at load; only the blend mask depends on where
    the live face is and has to be rebuilt each frame.
*   **The avatar is composited over the camera frame**, not rendered on its own
    background, so what you see is your own scene with a different face in it.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sha256.core.avatar import AvatarDescriptor
from sha256.core.enums import DeviceKind, PixelFormat
from sha256.core.errors import AvatarLoadError
from sha256.core.types import (
    FaceTrackingResult,
    FrameMetadata,
    RenderedFrame,
    VideoFrame,
)
from sha256.utils.logging import get_logger
from sha256.utils.timing import now

__all__ = ["ENGINE_ID", "WarpAvatarEngine"]

_LOG = get_logger(__name__)

ENGINE_ID = "warp"

SOURCE_PORTRAIT_ROLE = "source_portrait"

# Extra photographs of the same face, and what makes each one relevant. All are
# optional: with none of them the engine behaves exactly as it did with one
# photograph.
#
# Every source is aligned to the base portrait's landmark layout when it loads,
# so blending them is a straight weighted average of pixels. That works because
# they are the same person under the same light: after alignment the images are
# nearly identical *except* in the region the photograph was taken to capture,
# so a global blend only changes the mouth, or the eyes, or the shading of a
# turned cheek. Nothing has to decide which pixels belong to which feature.
_VARIANT_ROLES: tuple[str, ...] = (
    # Mouth
    "source_smile",
    "source_smile_closed",
    "source_mouth_open",
    "source_mouth_o",
    "source_mouth_wide",
    "source_mouth_pressed",
    "source_frown",
    # Eyes and brows
    "source_eyes_closed",
    "source_eyes_wide",
    "source_squint",
    "source_brows_up",
    "source_brows_down",
    "source_look_left",
    "source_look_right",
    "source_look_up",
    "source_look_down",
    # Head pose
    "source_turn_left",
    "source_turn_right",
    "source_turn_left_far",
    "source_turn_right_far",
    "source_tilt_up",
    "source_tilt_down",
)

# The four roles a turn ladder replaces. An avatar with a ladder has a
# photograph at every angle, so choosing the nearest is exact where averaging
# these four was an approximation — and mixing both would undo the choosing.
_TURN_ROLES: tuple[str, ...] = (
    "source_turn_left",
    "source_turn_right",
    "source_turn_left_far",
    "source_turn_right_far",
)

# How ladder sources are keyed in the source dictionary. A prefix rather than a
# role, because there is no fixed number of them and they are chosen by angle.
_LADDER_PREFIX = "turn@"

# The role whose own mouth opening defines "fully open". Measured from the
# photograph at load rather than hard-coded, so supplying a wider one simply
# extends the range instead of needing a constant changed here.
_OPENING_REFERENCE_ROLE = "source_mouth_open"

# Yaw at which a turned photograph is fully in play. Set from the photographs
# the owner generated, which measured +21.1 and -19.6 degrees: past their own
# angle they would be extrapolating rather than contributing.
_TURN_FULL_DEGREES = 20.0

# Yaw at which the far turn photographs, if supplied, take over completely.
_TURN_FAR_DEGREES = 38.0

# The tilt photographs are deliberately not used. See _source_weights.

# Blendshape reading at which a photograph of that expression fully applies.
# One number for all of them: these are all "how much of this expression is
# happening", and the photographs were all shot at a comparable extreme.
_EXPRESSION_FULL = 0.55

# No single variant may take over completely. Each was photographed at one
# extreme of its expression, and a face driven all the way there looks like it
# is performing rather than talking.
_MAX_VARIANT_WEIGHT = 0.95

# How the resting face is learned. See :class:`_Neutral`.
#
# A blendshape floor follows a new low within about a quarter of a second and
# creeps upward over twenty, so a relaxed face is learned quickly and an
# expression held for several seconds cannot become the new normal.
_FLOOR_FALL = 0.15
_FLOOR_RISE = 0.0015

# The head-pose neutral is a slow mean — roughly a seventeen-second time
# constant at 30 fps. Holding a turn for half a minute does move it, which is
# correct: by then the turn has become how the person is sitting.
_POSE_RATE = 0.002

# Inner-lip landmarks, and the gap between them at which the mouth counts as
# open, as a fraction of face height. Measured on the avatar's own photographs:
# closed reads 0.0010, the smiling-with-teeth photo 0.0591, the open-mouth photo
# 0.0947. Below the floor no teeth can be showing; at the ceiling they fully are.
_UPPER_INNER_LIP, _LOWER_INNER_LIP = 13, 14
_FOREHEAD_LANDMARK, _CHIN_LANDMARK = 10, 152
_LIP_GAP_CLOSED = 0.012
_LIP_GAP_OPEN = 0.055

# Fallback for "fully open" when no open-mouth photograph is supplied. Measured
# from the owner's first set; a real photograph's own gap always wins.
_LIP_GAP_FULL = 0.095

# Brow blendshape above which the raised-brow photograph fully applies.
_BROW_RAISE_FULL = 0.55

# The smiling photograph's own mouthSmile reading, so "how much of a smile is
# this" is measured against the photograph rather than against 1.0.
_SMILE_REFERENCE = 0.54

# How far the eyelids have to close, as a fraction of their own resting
# opening, before the closed-eye photograph starts to apply and where it fully
# applies. Relative, not absolute: eyes differ in shape between people, and this
# is measured on the user rather than on the avatar.
#
# Measured over the owner's recorded clip, which is labelled by segment, so a
# blink, a held closure and a relaxed eye can each be summarised separately:
#
#     segment                        p50    p95    max
#     sitting still                 0.02   0.33   0.62   <- the 0.62 is one blink
#     blinking several times        0.25   0.64   0.70
#     eyes held shut for 2 seconds  0.21   0.65   0.68
#     looking left and right        0.11   0.30   0.47   <- no blink in either
#
# The third row is the point. **The blendshape read 0.66 for a blink and 0.14
# for a held closure**; the lids read 0.65 for both, because the lids are shut
# either way. That is the whole reason this signal moved to geometry.
#
# The fourth row sets the floor, and it is higher than it looks. A gaze to the
# side genuinely narrows the lids, and reaches 0.47 on a segment with no blink
# in it at all. Starting below that put a closed-eye photograph on the avatar
# through ordinary talking — measured over the whole recording, the eye
# photographs were active on 41% of frames, on a face that was doing nothing but
# speak. No single frame of that looks broken, which is exactly why it survived
# five rounds of looking at single frames.
#
# So the band starts above where ordinary life reaches and still closes fully
# inside a real blink. There is very little room between the two, and that is a
# fact about the signal rather than a choice.
_CLOSE_START, _CLOSE_FULL = 0.48, 0.66

# The same lid opening, read the other two ways. Measured on the clip generated
# to hold each expression in turn, as a fraction of the eye's own resting
# opening:
#
#     resting                 0.00 to +0.11
#     eyes held wide open    +0.38 to +0.53
#     eyes held shut         -0.63 to -0.71
#
# Wide starts above where a resting eye wanders and reaches full inside a real
# one. There is no squint band: see `_source_weights` for why a partial closure
# cannot be told from ordinary lid movement, and what would tell it.
_WIDE_START, _WIDE_FULL = 0.20, 0.45

# The camera is **never** passed through once an avatar has been placed on a
# face. Tracking drops out constantly, and every dropout used to show the user's
# real face — first for a frame, then, after a one-second grace period was
# added, for as long as the loss lasted. Both read as the picture flickering.
#
# There is no length of loss at which showing the real face becomes the right
# answer: this product exists to keep it off the call. So the avatar stays where
# it was last seen, indefinitely, and the background keeps moving. A face
# hovering over an empty chair is strange; the user's face appearing on a call
# they are hiding it from is a failure.
#
# Losses longer than this are logged, because a persistent one means something
# else is wrong and it should not be silent.
_LOG_HOLD_AFTER_SECONDS = 2.0

# A curated subset of the 478-point mesh: the face oval, both brows, both eyes,
# the nose, and the lips. Chosen so the mesh stays small enough to warp inside a
# frame budget while still carrying every feature a viewer actually watches.
# fmt: off
_FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
)
_LEFT_BROW = (46, 53, 52, 65, 55, 107)
_RIGHT_BROW = (276, 283, 282, 295, 285, 336)
# The complete eyelid ring, sixteen points per eye, in contour order — not a
# six-point sample of it, and for a reason that took four failed attempts to
# find.
#
# Six points do describe a closing lid well enough now that the photographs
# carry the lid themselves. What they cannot do is *terminate* anything. With
# only six, Delaunay leaves the eye connected to the rest of the face by long
# thin triangles that reach in from the cheek and the brow, a whole fan of them
# converging on each corner. Close the lid and that corner travels; every
# triangle in the fan sweeps with it, and the sweep is the hard-edged light
# wedge that appeared above the eye on every single blink.
#
# The ring gives those triangles somewhere to stop. Nothing reaches *through*
# the eye any more, so a lid can move without dragging the brow and the cheek
# after it. It also meshes the socket properly around the iris, which the four
# iris points alone could not do.
#
# Ordering is load-bearing: these go around the ring, so the eye opening can be
# used as a polygon the way the inner lip already is.
_LEFT_EYE = (
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246,
)
_RIGHT_EYE = (
    362, 382, 381, 380, 374, 373, 390, 249, 263,
    466, 388, 387, 386, 385, 384, 398,
)
_NOSE = (168, 6, 197, 195, 5, 4, 98, 97, 2, 326, 327)
# The complete lip contours, not a sample of them. Eight points around the
# inner lip cannot describe an opening: the triangles spanning the mouth
# stretched whatever pixels they held into long vertical streaks, which is the
# "fangs" the owner photographed.
_OUTER_LIPS = (61, 91, 84, 17, 314, 321, 291, 270, 267, 0, 37, 40)
_INNER_LIPS = (
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
)
_CHEEKS = (234, 454, 50, 280, 116, 345, 205, 425)

# The irises: a centre and four points around it, per eye. Present only in the
# 478-landmark model, which is the one B003 measured and the one this engine
# requires.
#
# In the mesh because gaze has to be *moved*, not swapped. Four gaze
# photographs cross-faded into the base put the pupil in two places at once —
# the "ghost eye" — and no amount of weighting fixes that, because a blend of
# two positions is both of them rather than a third. With the iris in the mesh
# the pupil is geometry, so it goes exactly where the user is looking, smoothly,
# at any angle, and needs no photograph at all.
_LEFT_IRIS = (468, 469, 470, 471, 472)
_RIGHT_IRIS = (473, 474, 475, 476, 477)
# fmt: on

MESH_INDICES: tuple[int, ...] = tuple(
    dict.fromkeys(
        _FACE_OVAL
        + _LEFT_BROW
        + _RIGHT_BROW
        + _LEFT_EYE
        + _RIGHT_EYE
        + _NOSE
        + _OUTER_LIPS
        + _INNER_LIPS
        + _CHEEKS
        + _LEFT_IRIS
        + _RIGHT_IRIS
    )
)

# The face oval is the first block of MESH_INDICES and stays in outline order,
# which is what lets it be filled as a polygon rather than a point cloud.
_OVAL_SLICE = slice(0, len(_FACE_OVAL))
_BROW_SLICE = slice(len(_FACE_OVAL), len(_FACE_OVAL) + len(_LEFT_BROW) + len(_RIGHT_BROW))

# Where each landmark group ended up in the mesh, looked up rather than counted.
# MESH_INDICES de-duplicates, so positions cannot be derived by adding group
# lengths without knowing which groups overlap — and two of them do.
_MESH_POSITION = {index: position for position, index in enumerate(MESH_INDICES)}


def _positions(*groups: tuple[int, ...]) -> list[int]:
    return [_MESH_POSITION[index] for group in groups for index in group]


_EYE_POSITIONS = _positions(_LEFT_EYE, _RIGHT_EYE)

# Per eye, because gaze is placed one eye at a time: each iris is positioned
# inside its own socket, so a face at an angle still gets both right.
_EYE_PAIRS = (
    (_positions(_LEFT_EYE), _positions(_LEFT_IRIS)),
    (_positions(_RIGHT_EYE), _positions(_RIGHT_IRIS)),
)
_IRIS_POSITIONS = _positions(_LEFT_IRIS, _RIGHT_IRIS)

# What an eye photograph is allowed to move — the lids *and the irises inside
# them*. The irises were left out, and on a blink that put the mesh's iris
# above the eyelid that had just closed over it: the lid travelled and the
# pupil it contains stayed at the open eye's position. The triangles between
# the two are then inside-out, which is the wedge the owner saw.
#
# Gaze is still geometry rather than photographs — nothing here chooses where
# the pupil points. This only keeps the pupil inside the eye while the eye
# itself is being changed by a photograph.
_EYE_REGION_POSITIONS = _EYE_POSITIONS + _IRIS_POSITIONS

# Each eyelid ring in contour order, so the eye opening can be tested as a
# polygon rather than as a cloud of points.
_EYE_RINGS = tuple(eye for eye, _ in _EYE_PAIRS)

# How far the pupil may travel inside its socket, as a fraction of eye width,
# and how much of the user's travel it copies.
#
# The mesh has four points around each iris, so the triangles between them and
# the lid shear as the pupil moves. Rendered across the full
# range, a small shift is clean and an extreme one turns the iris into a dark
# diamond with visible triangle edges. Vertically it is worse and asymmetric,
# because the lower lid is closer than the upper one.
#
# The limits are where the render stops being clean, and the scale keeps normal
# gaze comfortably inside them. Nothing is lost by it: gaze reads as a
# *direction* long before it reads as a magnitude, and a video call is not
# looked at from the corner of the eye.
_GAZE_SCALE = 0.55
_GAZE_LIMIT_X = 0.065
_GAZE_LIMIT_UP, _GAZE_LIMIT_DOWN = 0.035, 0.012

# How much of the space between the pupil and the nearer lid the iris ring may
# take up. See :func:`_contain_iris` for why it must take up less than all of
# it: at exactly all of it the triangles joining iris to lid have no area, and
# an affine transform fitted to three collinear points is singular.
_IRIS_OF_OPENING = 0.85
_BROW_POSITIONS = _positions(_LEFT_BROW, _RIGHT_BROW)
_MOUTH_POSITIONS = _positions(_OUTER_LIPS, _INNER_LIPS)

# Landmarks that an *expression* does not move: the face outline and the nose
# bridge. Used to place an expression photograph on the base's frame without
# letting the expression itself drag the fit.
_STABLE_POSITIONS = _positions(_FACE_OVAL, _NOSE)

# Which part of the face each photograph actually describes.
#
# This is the fix for the defect that made the avatar look like a multiple
# exposure of itself. Every source used to be averaged across the *whole* face,
# so a smile photograph changed the forehead, the ears and the hairline as well
# as the mouth, and twelve active sources meant twelve photographs fighting over
# every pixel. Sharpness measured 62 on the portrait and 17 on the blend.
#
# Confining each photograph to its own region does three things, and the third
# is the reason it matters most:
#
#   1. ghosting stops, because a region is drawn from one or two photographs
#      rather than from all of them;
#   2. cost falls, because a mouth blend touches the mouth's pixels;
#   3. **expressions stop interfering**, so a smile and a glance compose instead
#      of averaging. That is what makes a larger photograph set worth having —
#      without it, more photographs make the picture worse, which was measured.
#
# Roles absent from this table describe the whole head — a turn or a tilt really
# does change every pixel — and are applied globally, as before.
_REGION_ROLES: dict[str, tuple[str, ...]] = {
    "mouth": (
        "source_smile",
        "source_smile_closed",
        "source_mouth_open",
        "source_mouth_o",
        "source_mouth_wide",
        "source_mouth_pressed",
        "source_frown",
    ),
    "eyes": (
        "source_eyes_closed",
        "source_eyes_wide",
        "source_squint",
        "source_look_left",
        "source_look_right",
        "source_look_up",
        "source_look_down",
    ),
    "brows": (
        "source_brows_up",
        "source_brows_down",
    ),
}

_ROLE_REGION: dict[str, str] = {
    role: region for region, roles in _REGION_ROLES.items() for role in roles
}

# How far each region reaches beyond its own landmarks, and how soft its edge
# is, both as a fraction of face width. Generous on purpose: an eye photograph
# has to carry the lid and the socket shadow, not just the lash line, or the
# boundary shows as a ring around the eye. The feather is what keeps the join
# invisible — a hard edge would trade ghosting for a visible patch.
_REGION_MARGIN: dict[str, float] = {"mouth": 0.10, "eyes": 0.11, "brows": 0.09}
_REGION_FEATHER = 0.05

# How far up the forehead the mask reaches, as a fraction of the distance from
# the brows to the top of the face outline. The owner's requirement is that the
# forehead is covered — a mask stopping above the brows leaves a band of real
# skin whose tone does not match. 0.85 covers almost all of it while stopping
# just short of the hairline itself, which is structure rather than tone and is
# the one thing the colour correction cannot fix.
_FOREHEAD_REACH = 0.85

# How far down the face the jaw extension starts, as a fraction of face height
# from the chin, and how much further those points are pushed.
#
# The oval runs along the jawbone, so the avatar ended there and the user's own
# jaw carried on below it — on the owner's render, the avatar's ginger beard
# stopping in a hard curve with their darker beard continuing underneath. The
# mirror of the forehead problem, and easier, because below a jaw is neck: tone
# rather than structure, which the relighting already carries.
#
# Only the lowest quarter is moved, and by half again, so the extension tapers
# into the jawline instead of stepping away from it.
#
# **Reverted to no extension, 2026-08-15, on the owner's judgement.** They ran
# it and said the old behaviour — their own jaw showing below the avatar's — was
# better. That is worth recording rather than arguing with: an avatar chin
# stretched over a neck it does not belong to is a wrong shape, and a wrong
# shape reads worse than an honest edge. The mechanism stays here because it is
# correct for a face whose jaw the avatar's own actually fits.
_JAW_REACH = 0.25
_JAW_REACH_SCALE = 1.0

# Low-frequency relighting: the correction field is computed at 1/8 scale and
# blurred there. Lighting varies slowly across a face by definition, so the
# resolution costs nothing and the work drops by ~64x.
_CORRECTION_DOWNSCALE = 8

# The correction is a **plane** — one brightness and one gradient per colour
# channel — fitted to the masked difference by weighted least squares.
#
# It was a blurred difference field, and that was the single biggest reason the
# avatar did not look like its own photograph. The correction adds the whole
# scene-minus-avatar difference back, so everything the blur passes is taken
# from whatever is underneath the mask — and underneath is the *user's face*,
# not the room. Measured, with the fixed 21-pixel kernel it used at 1/8 scale:
#
#     overall brightness   100%   <- lighting, and correct
#     half-face gradient    97%   <- lighting, and correct
#     a brow shadow         70%   <- the user's face, and not
#     a beard edge          40%   <- the user's face, and not
#
# The avatar was being printed with the user's own brow shadows and beard —
# precisely what this product exists to hide — and no improvement to the source
# blend could survive it.
#
# Widening the blur trades one against the other: at 35% of face width the brow
# shadow fell to 27% but the half-face gradient fell to 56%, so real lighting
# stopped being corrected too. A plane has no such trade-off. Overall brightness
# and a directional gradient are *exactly* what a plane represents, so they come
# through whole; a brow shadow is curvature, which a plane cannot represent at
# all. It is also cheaper than the blur it replaces.
#
# What it cannot follow is genuinely curved illumination — a lamp pointed at the
# middle of the face. A quadratic would, at the cost of being able to represent
# a brow again, and that trade is not worth making until something demands it.
_CORRECTION_DEGREE = "plane"

# How much of the previous frame's relighting to keep. A camera hunting its
# exposure makes the scene brighten and dim continuously, and a correction that
# follows it instantly makes the avatar's face pulse against a background that
# is only drifting. Smoothing the field over time removes the pulse; it costs a
# few frames of lag on a genuine lighting change, which nobody notices.
_CORRECTION_INERTIA = 0.8

# How far *inside* the face outline the blend stops, and how soft the edge is,
# both as a fraction of face width. Eroding rather than growing keeps the
# transition on skin at both ends; a hard edge reads as a cut-out sticker, and
# too soft lets the real face show through the avatar.
# Where the inner lip ring sits inside MESH_INDICES, so the mouth opening can be
# found as a polygon rather than a cloud of points.
_INNER_LIP_SLICE = slice(
    len(_FACE_OVAL)
    + len(_LEFT_BROW)
    + len(_RIGHT_BROW)
    + len(_LEFT_EYE)
    + len(_RIGHT_EYE)
    + len(_NOSE)
    + len(_OUTER_LIPS),
    len(_FACE_OVAL)
    + len(_LEFT_BROW)
    + len(_RIGHT_BROW)
    + len(_LEFT_EYE)
    + len(_RIGHT_EYE)
    + len(_NOSE)
    + len(_OUTER_LIPS)
    + len(_INNER_LIPS),
)

# Everything the face oval stops covering once the head turns, which is one
# whole side of it. See :func:`_destination_mask` — this is the difference
# between an avatar and the user's own eye on screen.
#
# Counted at 39 degrees of yaw, live landmarks falling outside the mask: 16 of
# 32 eye points, 6 of 12 brow points, 5 of 10 iris points, 6 of 11 nose points,
# 4 of 8 cheek points. Half of each — the far half. Not a rim: a face.
_PROFILE_POSITIONS = _positions(
    _LEFT_BROW, _RIGHT_BROW, _LEFT_EYE, _RIGHT_EYE, _NOSE, _OUTER_LIPS, _CHEEKS
)

# The inner lip's upper arc, corner to corner. The mouth interior is placed
# from these and not from the whole ring, because **the upper teeth hang from
# the skull and the jaw drops away from them**. Fitted to the whole ring, the
# photograph's interior is scaled to whatever the live opening happens to be:
# open wider and the teeth grow with it, and the thin shadow above them grows
# into a band the size of a lip. That is the owner's "when the teeth show, the
# lips are lost" — the band was sitting where their upper lip should be.
#
# Anchored to the upper lip instead, the teeth arrive at the size the mouth's
# width implies and stay under the lip they belong to, and opening wider
# reveals darkness *below* them, which is what opening a jaw does.
_UPPER_INNER_LIPS = (308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78)
_UPPER_LIP_POSITIONS = _positions(_UPPER_INNER_LIPS)

_MASK_SHRINK = 0.04
_MASK_FEATHER = 0.07

# How much of the brows' movement the avatar copies. Raising the brows moves the
# landmarks a long way, and the mesh answers by stretching the eye beneath them
# — the avatar inherits the user's *proportions* where it should be inheriting
# only their motion. Halving the travel keeps the expression readable without
# the eye growing with it.
_BROW_FOLLOW = 0.5

# How much of the *eye opening* the avatar copies from the user. Almost none,
# and that is deliberate.
#
# The mesh forces the avatar's eye aperture to match the user's exactly, so the
# avatar was inheriting the shape and size of someone else's eyes and having its
# own iris stretched to fill them. On a live frame that reads as pale, flat,
# lifeless eyes — the single thing that gives the whole picture away.
#
# The eyes do not need the mesh for expression any more: blinking, squinting,
# widening and gaze all arrive as photographs now, and their geometry arrives
# with them in the blended mesh. So the target here is the avatar's *own* eye
# geometry placed rigidly on the live head, and the small remainder exists only
# so that expressions no photograph covers are not frozen out entirely.
_EYE_FOLLOW = 0.2

# How much the eyes follow when a photograph was asked for and **could not be
# delivered**, which is what happens at a turn: every eye photograph is of a
# face looking straight ahead, and past about fifteen degrees it can no longer
# be placed (see ``_CARRY_LOST``). The blend then has nothing to say about the
# eyes and they freeze at whatever the turned photograph happened to catch.
#
# The owner, testing: *"when I look sideways it closes the eye by itself while
# mine is open, and it almost never follows my eyes opening and closing then."*
# The second half is this. The mesh takes over exactly as much as the
# photograph loses, so a blink still reads as a blink at an angle — and at rest,
# where no photograph is asked for because nothing is happening, nothing
# changes at all: the avatar keeps its own eyes, which is what item 12 was for.
_EYE_FOLLOW_ALONE = 0.6

# At most one gaze photograph at a time. Two of them — a look left and a look
# up, say — put the pupil in two places and average it into a smear, which is
# most of why the eyes looked washed out. One photograph puts the pupil
# somewhere definite, and definite in roughly the right direction beats a
# precise average of two wrong ones.
_GAZE_ROLES = (
    "source_look_left",
    "source_look_right",
    "source_look_up",
    "source_look_down",
)

# Gaze is all-or-nothing, and it is the only source family that has to be.
#
# Every other photograph changes how a feature *looks* and cross-fades into the
# base cleanly. A gaze photograph changes where the pupil *is*, and a cross-fade
# of two positions renders both: at weight 0.5 the eye is a grey wash of two
# overlapping irises, and at 0.75 there is a solid pupil with a faint second one
# beside it. That is exactly what the owner reported as a ghost eye, and it was
# visible in a render across the weight range before it was fixed.
#
# Geometry cannot rescue this the way it rescues a blink: the mesh carries the
# eyelids, so a half-closed eye interpolates honestly, but the iris is not a
# landmark and a half-shifted gaze has nowhere to go. So the weight ramps from
# nothing to full across a narrow band and spends almost no time between.
#
# A jump is also the truthful behaviour. Eyes move in saccades — they are not
# capable of a slow pan — so switching is closer to life than fading ever was.
_GAZE_ON, _GAZE_FULL = 0.12, 0.25

# Source weights are rounded to this before deciding whether the previous blend
# can be reused. Fine enough that no step is visible, coarse enough that a
# still face rebuilds nothing.
_WEIGHT_QUANTUM = 0.02

# How much larger than the base a source may be before it is reduced with area
# averaging first. Below this the mesh's own resampling is milder than a second
# resize would be.
_REDUCE_THRESHOLD = 1.25

# How far the head must have moved before a region is carried onto it, in
# pixels of landmark travel. Carrying costs a resampling of the region, and a
# resampling softens it; under a pixel that is a bad trade, and with a rung
# every three degrees it would otherwise be paid on nearly every frame.
_CARRY_MINIMUM_PX = 1.0

# How badly a region may be misplaced before its photographs are worth nothing,
# as a fraction of face width. The carry is a similarity and a turned face is
# foreshortened, which a similarity cannot express, so what is left over lands
# as a second, ghostly copy of the feature. Measured across the owner's ladder,
# as RMS landmark error on a 455 px face:
#
#     yaw       mouth   eyes   brows
#     ±10°       4.0     3.0     6.6
#     ±17°       6.8     6.9    13.6
#     ±39°      19.0     7.8    20.9
#
# A full affine was tried, which *can* express foreshortening, and it is worse:
# six degrees of freedom fitted robustly to twenty points is less stable than
# four, and the brows went from 20.9 px to 23.3 px at 39°.
#
# So a frontal photograph cannot be placed on a strongly turned head, and the
# honest answer is to stop applying it rather than to apply it wrongly — this
# is what put a black wedge across the avatar's eye at 39°, where the brow
# region reaches down over the socket and was landing 21 px out. Each region
# fades by its own fit error: full weight while the miss is inside the noise
# between two photographs of the same face, nothing once it is wide enough to
# read as a doubled feature. The brows go first, which is right; they fit worst
# at every angle.
_CARRY_CLEAN = 0.008
_CARRY_LOST = 0.030


class _Neutral:
    """The user's resting face, learned while they sit in front of the camera.

    MediaPipe's blendshapes are **not zero on a relaxed face**, and its head
    pose is not zero unless the camera is exactly on axis — which no webcam
    above a monitor ever is. Driving photographs from those raw numbers meant a
    person sitting still was read as squinting hard, looking down and to the
    left, brows both raised *and* lowered, turned 19° and tilted 18°. Measured,
    on a still subject: eleven photographs blended into every frame of a neutral
    face, so the avatar stopped looking like its own portrait and started
    looking like a double exposure of eleven of them.

    The fix is to measure expression *relative to rest* rather than absolutely.
    Two kinds of signal need two estimators:

    *   **Blendshapes only rise from rest**, so their neutral is a *floor*.
        Tracked asymmetrically — follow a new low quickly, drift up slowly — so
        a relaxed face is learned in about a second while an expression held for
        several seconds cannot be mistaken for one.
    *   **Head pose is signed** and its neutral is wherever the person happens
        to sit, so it is a slow mean.

    Both are seeded from the first frame, which is what makes the very first
    frame correct rather than merely eventually correct: with the estimate equal
    to the reading, every variant weight starts at zero and the avatar starts as
    its own photograph.
    """

    __slots__ = ("_ceilings", "_floors", "_means", "_pose")

    def __init__(self) -> None:
        self._floors: dict[str, float] = {}
        self._ceilings: dict[str, float] = {}
        self._means: dict[str, float] = {}
        self._pose: dict[str, float] = {}

    def relative_expression(self, name: str, value: float) -> float:
        """``value`` above this signal's learned resting floor, never below zero."""
        floor = self._floors.get(name)
        if floor is None:
            self._floors[name] = value
            return 0.0

        rate = _FLOOR_FALL if value < floor else _FLOOR_RISE
        floor += rate * (value - floor)
        self._floors[name] = floor
        return max(value - floor, 0.0)

    def relative_drop(self, name: str, value: float) -> float:
        """How far ``value`` has fallen below its learned high, as a fraction.

        The mirror of :meth:`relative_expression`, for a signal that *falls*
        when something happens. An eye's height over its width is one: it sits
        near 0.32 open and near 0.10 shut, so the interesting direction is down.

        Tracked the same way and for the same reason, reflected: follow a new
        high quickly, so the widest the eye has been is learned in about a
        quarter of a second, and drift down slowly, so a person who blinks
        often does not teach it that half-shut is their normal eye.
        """
        ceiling = self._ceilings.get(name)
        if ceiling is None or ceiling <= 1e-6:
            self._ceilings[name] = value
            return 0.0

        rate = _FLOOR_FALL if value > ceiling else _FLOOR_RISE
        ceiling += rate * (value - ceiling)
        self._ceilings[name] = ceiling
        return max((ceiling - value) / max(ceiling, 1e-6), 0.0)

    def relative_rise(self, name: str, value: float) -> float:
        """How far ``value`` sits above its usual level, as a fraction of it.

        Against a *slow mean*, where :meth:`relative_drop` is against a ceiling,
        and the two references are different on purpose. Closing is measured
        from the most open the eye has been, because that is what a lid closes
        from. Opening wide is measured from how open the eye usually is,
        because a ceiling would absorb the very thing being measured — hold the
        eyes wide for two seconds and a fast-following high would call it the
        new normal within half of one.
        """
        rest = self._means.get(name)
        if rest is None or rest <= 1e-6:
            self._means[name] = value
            return 0.0

        # Asymmetric, and the symmetric version was visibly wrong. A plain slow
        # mean is dragged down by every blink and every glance — both narrow the
        # lids — until it sits below the eye's actual resting opening, and then
        # a face doing nothing at all reads as *wide*. Rendered, that is a
        # resting avatar staring with the whites showing.
        #
        # So it follows an opening eye at the ordinary slow rate and a closing
        # one ten times slower still. What it estimates is "how open this eye is
        # when it is open", which is the thing being compared against.
        rate = _POSE_RATE if value > rest else _POSE_RATE * 0.1
        rest += rate * (value - rest)
        self._means[name] = rest
        return max((value - rest) / max(rest, 1e-6), 0.0)

    def pose_offset(self, name: str, value: float) -> float:
        """``value`` against the resting angle, **without learning from it**.

        For a second reader of the same signal in the same frame. Calling
        :meth:`relative_pose` twice would advance the running mean twice, which
        is an invisible doubling of how fast the neutral chases the head — the
        kind of mistake that shows up weeks later as "it drifts when I hold a
        turn" and takes a day to find.
        """
        rest = self._pose.get(name)
        return 0.0 if rest is None else value - rest

    def relative_pose(self, name: str, value: float) -> float:
        """``value`` as an offset from the resting head angle. Signed."""
        rest = self._pose.get(name)
        if rest is None:
            self._pose[name] = value
            return 0.0

        rest += _POSE_RATE * (value - rest)
        self._pose[name] = rest
        return value - rest

    def copy(self) -> _Neutral:
        """An independent resting face with the same estimates.

        For measuring frames *against a fixed rest* rather than against a rest
        that keeps learning. Live, the learning is the point — the person
        settles into a new position and the avatar should follow. Offline, when
        the question is "how much of a turn is this frame, compared with the
        base portrait", every frame has to be asked from the same starting
        point, or the answer depends on what order the frames arrived in.
        """
        made = _Neutral()
        made._floors = dict(self._floors)
        made._ceilings = dict(self._ceilings)
        made._means = dict(self._means)
        made._pose = dict(self._pose)
        return made


class _Region:
    """One part of the face, and the photographs allowed to change it.

    Everything here is precomputed at load: the box the region occupies in the
    avatar's own image, a feathered mask inside that box, and each source
    photograph already cropped to it. Per frame the work is then a weighted sum
    over a small rectangle and one alpha composite — no cropping, no mask
    building, no full-face arithmetic.
    """

    __slots__ = ("box", "mask", "name", "positions", "sources")

    def __init__(
        self,
        name: str,
        box: tuple[int, int, int, int],
        mask: np.ndarray,
        sources: dict[str, np.ndarray],
        positions: list[int],
    ) -> None:
        self.name = name
        self.box = box
        self.mask = mask
        self.sources = sources
        self.positions = positions
        """Mesh positions this region owns.

        The blend moves these landmarks as well as these pixels. It has to: an
        open-mouth photograph is only useful if the mesh driving it also says
        the mouth is open, and a mesh still describing closed lips would simply
        crush it shut again.
        """


class _LoadedAvatar:
    """Everything precomputed for one avatar, held for the life of the load.

    ``sources`` maps a role to a photograph placed on the base portrait's frame
    — same position, same scale, same rotation — and ``source_points`` holds
    where each one's own landmarks landed. The two together are what every
    frame is composed from: the pixels of a photograph, and a mesh that
    describes those pixels rather than the base's.
    """

    __slots__ = (
        "blend_points",
        "cached_blend",
        "cached_weights",
        "cavity",
        "correction",
        "descriptor",
        "eye_follow",
        "full_gap",
        "gaps",
        "image",
        "interior_role",
        "ladder",
        "neutral",
        "points",
        "regions",
        "scratch",
        "source_points",
        "sources",
        "triangles",
    )

    def __init__(
        self,
        descriptor: AvatarDescriptor,
        image: np.ndarray,
        points: np.ndarray,
        triangles: np.ndarray,
        sources: dict[str, np.ndarray],
        full_gap: float = _LIP_GAP_FULL,
        source_points: dict[str, np.ndarray] | None = None,
        ladder: tuple[tuple[float, str | None], ...] = (),
        gaps: dict[str, float] | None = None,
    ) -> None:
        self.full_gap = full_gap

        # How far apart each photograph's own lips are. It is what says which
        # photographs have an *interior* — teeth, a tongue — to give, and a
        # measurement rather than a list: the owner's set separates cleanly at
        # a fifth of the way open, with three photographs above it and four
        # below.
        self.gaps = gaps if gaps is not None else {}

        # How much the eyes should follow the user this frame. Raised when an
        # eye photograph was asked for and could not be placed — see
        # :data:`_EYE_FOLLOW_ALONE`.
        self.eye_follow = _EYE_FOLLOW

        # Which photograph the mouth interior is being taken from this frame,
        # or ``None`` for "no photograph has one". Never an average of several —
        # see :func:`_paint_mouth_interior`.
        self.interior_role: str | None = None

        # The turn ladder: photographs of this face at measured angles, sorted
        # by yaw, with ``None`` marking the base portrait's own rung at zero.
        # Empty for an avatar that has only the named turn roles.
        self.ladder = ladder
        self.descriptor = descriptor
        self.image = image
        self.points = points
        self.triangles = triangles
        self.sources = sources

        # Where each source's own landmarks ended up once it was placed. They
        # differ from the base's, and the difference *is* what the photograph
        # was taken for: the open mouth, the shut eyelid, the turned head.
        self.source_points = source_points if source_points is not None else {}

        # The mesh describing the current blend. Recomputed with it, so the two
        # are always a description of the same face. Everything that has to know
        # where the blend's features actually are — the live warp, the mouth
        # interior — reads it rather than the portrait's own landmarks.
        self.blend_points = points

        # What colour the back of this mouth is, for the parts of an opening no
        # photograph reaches — see _paint_mouth_interior.
        self.cavity = _cavity_colour(image, points)

        # Where each group of photographs is allowed to act, with their pixels
        # already cropped to it.
        self.regions = _build_regions(image, points, sources, self.source_points)

        # The most recent blend and the weights that produced it.
        self.cached_weights: dict[str, int] = {}
        self.cached_blend: np.ndarray | None = None

        # Working buffer for the region composite, allocated on first use. Safe
        # to reuse: it is only rewritten when the cache misses, and a cache hit
        # returns it untouched.
        self.scratch: np.ndarray | None = None

        # The previous frame's relighting field, kept at its small working size.
        self.correction: np.ndarray | None = None

        # The driver's resting face. Held per load rather than per process so
        # that reselecting an avatar starts the estimate again, which is the
        # cheapest possible "recalibrate" for a user who has moved their chair.
        self.neutral = _Neutral()


class WarpAvatarEngine:
    """Deforms a portrait photograph to follow a tracked face."""

    def __init__(self, *, model_path: Path | str | None = None) -> None:
        # The tracker used to find the avatar's own landmarks. Only needed while
        # loading an avatar, never on the frame path.
        self._model_path = Path(model_path) if model_path is not None else None
        self._running = False
        self._loaded: dict[str, _LoadedAvatar] = {}

        # The last frame in which each avatar's face was found: where it was,
        # what it was doing, and when. Used to ride out momentary tracking
        # losses without ever showing the camera through.
        self._last_face: dict[str, tuple[np.ndarray, FaceTrackingResult, float]] = {}
        self._warned: dict[str, bool] = {}

    # --- Component ---------------------------------------------------------

    @property
    def backend_id(self) -> str:
        return ENGINE_ID

    @property
    def engine_id(self) -> str:
        return ENGINE_ID

    @property
    def device(self) -> DeviceKind:
        return DeviceKind.CPU

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        for avatar_id in list(self._loaded):
            self.unload_avatar(avatar_id)
        self._running = False

    # --- AvatarEngine ------------------------------------------------------

    def supports(self, descriptor: AvatarDescriptor) -> bool:
        return descriptor.manifest.engine.engine_id == ENGINE_ID

    def load_avatar(self, descriptor: AvatarDescriptor) -> None:
        """Read the portrait, find its landmarks, and build the mesh.

        Slow — it runs a face detection on the photograph — which is why it
        happens here and not on the first frame.
        """
        if descriptor.id in self._loaded:
            return
        if not self.supports(descriptor):
            msg = (
                f"avatar {descriptor.id!r} requires engine "
                f"{descriptor.manifest.engine.engine_id!r}, not {ENGINE_ID!r}"
            )
            raise AvatarLoadError(msg)

        full = self._read_portrait(descriptor)
        full_points = self._landmarks_of(full, descriptor.id)

        # Crop to the face before anything else. Blending sources is a
        # per-pixel cost paid every frame, and a 1024x1024 portrait is mostly
        # background and shoulders that the mesh never samples. Cropping first
        # measured 31 ms down to single digits.
        image, points = _crop_to_face(full, full_points)
        # Before triangulating, not after: an iris standing proud of its own
        # eyelid makes the triangles between them inside-out, and a
        # triangulation computed from that is wrong for every frame after.
        _contain_irises(points)
        triangles = _triangulate(points, image.shape[1], image.shape[0])
        sources, source_points, gaps, ladder = self._align_variants(
            descriptor, image, points, triangles
        )
        full_gap = gaps.get(_OPENING_REFERENCE_ROLE, _LIP_GAP_FULL)

        self._loaded[descriptor.id] = _LoadedAvatar(
            descriptor, image, points, triangles, sources, full_gap, source_points, ladder, gaps
        )
        _LOG.info(
            "Loaded avatar %s: %dx%d portrait, %d mesh points, %d triangles, "
            "%d extra source(s), %d turn rung(s) from %+.0f to %+.0f degrees: %s",
            descriptor.id,
            image.shape[1],
            image.shape[0],
            len(points),
            len(triangles),
            len(sources),
            len(ladder),
            ladder[0][0] if ladder else 0.0,
            ladder[-1][0] if ladder else 0.0,
            ", ".join(sorted(r for r in sources if not r.startswith(_LADDER_PREFIX))) or "none",
        )

    def _align_variants(
        self,
        descriptor: AvatarDescriptor,
        base_image: np.ndarray,
        base_points: np.ndarray,
        triangles: np.ndarray,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, float],
        tuple[tuple[float, str | None], ...],
    ]:
        """Bring every extra photograph onto the base portrait's frame.

        **Every photograph is placed by a similarity transform** — rotate,
        uniform scale, translate — fitted on the face outline and the nose
        bridge. Nothing is reshaped: an open mouth stays open, a closed eyelid
        stays closed, and a turned head stays turned. Each one's landmarks are
        returned alongside, moved by the same transform, so the mesh that drives
        it later describes the picture that is actually there.

        Mesh-warping onto the base's layout is what this replaced, and it failed
        twice for one reason — it forces a photograph into geometry that is not
        its own:

        *   **Expressions**, fixed first. The base has a closed mouth and open
            eyes, so the open-mouth photograph was crushed shut and left a
            sliver of teeth for the live warp to stretch, and the closed-eye
            photograph had its eyelid pulled across an open socket, which
            rendered as vertical streaks.

        *   **Poses**, fixed here, and the same defect at four times the size.
            The base is looking straight ahead, so a 35-degree face folded onto
            it: a vertical seam down the cheek, the far eye and brow drawn
            twice, the mouth smeared sideways. Measured on the owner's ladder,
            a 35-degree rung's landmarks sit 61 px from the frontal layout on
            average and 156 px at the worst of them, against a 455 px face —
            that distance *is* the fold. It is the owner's "a slight turn nearly
            loses the face", their "two photographs on top of each other", and
            their "only good at five to ten degrees", which were one defect all
            along.

        The argument for warping a pose was that the live face supplies the
        angle, so the photograph is wanted only for its shading. That is true of
        the *result* and false of the method: the live warp maps this
        photograph's mesh onto the live face, and it can only do that if the
        mesh describes the photograph. Give it a turned mesh for a turned
        picture and the warp becomes nearly an identity — which is exactly what
        should happen when the avatar is already at the angle being asked for.

        Done once, here, because it is the expensive half: each variant needs a
        face detection. A variant that cannot be read, or has no face in it, is
        skipped with a warning rather than failing the load — losing one
        expression is a smaller harm than refusing to start.
        """
        height, width = base_image.shape[:2]
        aligned: dict[str, np.ndarray] = {}
        aligned_points: dict[str, np.ndarray] = {}
        gaps: dict[str, float] = {}

        for role in _VARIANT_ROLES:
            asset = descriptor.manifest.asset_by_role(role)
            if asset is None:
                continue

            path = descriptor.asset_path(asset)
            try:
                image = self._read_image(path)
                points = self._landmarks_of(image, f"{descriptor.id}:{role}")
            except AvatarLoadError as exc:
                _LOG.warning("Avatar %s: skipping %s — %s", descriptor.id, role, exc)
                continue

            gaps[role] = _lip_gap_of(points)
            # Before either placement path: both reduce this source onto the
            # base's scale, and both do it by point-sampling.
            image, points = _reduce_to(image, points, base_points)

            placed = _place_by_similarity(image, points, base_points, width, height)
            if placed is not None:
                aligned[role], aligned_points[role] = placed
                continue

            _LOG.warning(
                "Avatar %s: %s could not be placed by similarity, using the mesh",
                descriptor.id,
                role,
            )
            aligned[role] = warp_mesh(image, points, base_points, triangles, width, height)
            aligned_points[role] = base_points

        ladder = self._align_ladder(
            descriptor, base_points, triangles, width, height, aligned, aligned_points
        )
        return aligned, aligned_points, gaps, ladder

    def _align_ladder(
        self,
        descriptor: AvatarDescriptor,
        base_points: np.ndarray,
        triangles: np.ndarray,
        width: int,
        height: int,
        aligned: dict[str, np.ndarray],
        aligned_points: dict[str, np.ndarray],
    ) -> tuple[tuple[float, str | None], ...]:
        """Every asset that carries a measured yaw, as rungs sorted by angle.

        A turn is the one thing the engine had to *guess* at. It held two turned
        photographs and averaged them against the frontal one, so every angle
        between them was a blend of two angles that were not it — and past the
        outermost photograph it was extrapolating from a shot that did not go
        that far. A clip gives one frame at every angle instead, and then the
        right answer is to *pick* rather than to mix.

        Placed by similarity, like every other photograph and for the reason set
        out in :meth:`_align_variants`: a rung keeps the geometry of the angle it
        was shot at, and carries its landmarks with it so the mesh describes the
        picture. Folding it onto the frontal layout instead is what tore every
        rung in half.

        The base portrait joins the ladder as the rung at zero, marked ``None``.
        That is what keeps a face looking straight ahead byte-identical to its
        own photograph: at zero the bracket collapses onto the portrait, and the
        weight the portrait would have had is exactly the weight it keeps by not
        being mixed with anything.
        """
        rungs: list[tuple[float, str | None]] = [(0.0, None)]

        for index, asset in enumerate(descriptor.manifest.assets):
            if asset.yaw is None:
                continue

            key = f"{_LADDER_PREFIX}{index}"
            path = descriptor.asset_path(asset)
            try:
                image = self._read_image(path)
                points = self._landmarks_of(image, f"{descriptor.id}:{key}")
            except AvatarLoadError as exc:
                _LOG.warning("Avatar %s: skipping %s — %s", descriptor.id, asset.path, exc)
                continue

            image, points = _reduce_to(image, points, base_points)
            placed = _place_by_similarity(image, points, base_points, width, height)
            if placed is None:
                _LOG.warning(
                    "Avatar %s: %s could not be placed by similarity, using the mesh",
                    descriptor.id,
                    asset.path,
                )
                aligned[key] = warp_mesh(image, points, base_points, triangles, width, height)
                aligned_points[key] = base_points
            else:
                aligned[key], aligned_points[key] = placed
            rungs.append((float(asset.yaw), key))

        return tuple(sorted(rungs))

    def unload_avatar(self, avatar_id: str) -> None:
        self._loaded.pop(avatar_id, None)
        self._last_face.pop(avatar_id, None)
        self._warned.pop(avatar_id, None)

    @property
    def loaded_avatars(self) -> Sequence[str]:
        return tuple(self._loaded)

    def estimated_vram_mb(self, avatar_id: str) -> int | None:
        """Always ``None``: this engine holds nothing on the GPU."""
        del avatar_id
        return None

    def animate(
        self, avatar_id: str, tracking: FaceTrackingResult, frame: VideoFrame
    ) -> RenderedFrame:
        avatar = self._loaded.get(avatar_id)
        if avatar is None:
            msg = f"avatar {avatar_id!r} is not loaded"
            raise AvatarLoadError(msg)

        source = frame.data
        if frame.pixel_format not in (PixelFormat.BGR24, PixelFormat.RGB24):
            msg = f"the warp engine cannot composite over {frame.pixel_format.value} frames"
            raise AvatarLoadError(msg)

        target = None
        if tracking.face_detected and tracking.landmarks is not None:
            target = _pixel_points(tracking.landmarks.points, source.shape[1], source.shape[0])

        if target is not None:
            self._last_face[avatar_id] = (target, tracking, frame.metadata.capture_timestamp)
            self._warned[avatar_id] = False
        else:
            # Tracking drops the face constantly for a frame or three, and
            # passing the camera through one of those puts the user's *real
            # face* on the call for a tenth of a second. That is a flicker to
            # watch and a failure of the one thing this product exists to do, so
            # the last known face keeps driving the avatar instead. The
            # background stays live; only the face is held.
            held = self._last_face.get(avatar_id)
            if held is None:
                # Nothing has ever been tracked, so there is no avatar to hold
                # and no real face being hidden yet.
                return self._passthrough(source, frame, tracking, avatar_id)

            target, tracking, since = held
            held_for = frame.metadata.capture_timestamp - since
            if held_for > _LOG_HOLD_AFTER_SECONDS and not self._warned.get(avatar_id, False):
                self._warned[avatar_id] = True
                _LOG.warning(
                    "Tracking has been lost for %.1fs; holding the avatar in place", held_for
                )

        output = _warp_onto(avatar, target, source, tracking)

        return RenderedFrame(
            data=output,
            pixel_format=frame.pixel_format,
            metadata=tracking.metadata,
            avatar_id=avatar_id,
            render_timestamp=now(),
            is_placeholder=False,
        )

    @staticmethod
    def _passthrough(
        source: np.ndarray, frame: VideoFrame, tracking: FaceTrackingResult, avatar_id: str
    ) -> RenderedFrame:
        return RenderedFrame(
            data=source,
            pixel_format=frame.pixel_format,
            metadata=tracking.metadata,
            avatar_id=avatar_id,
            render_timestamp=now(),
            is_placeholder=True,
        )

    # --- loading helpers ---------------------------------------------------

    def _read_portrait(self, descriptor: AvatarDescriptor) -> np.ndarray:

        asset = descriptor.manifest.asset_by_role(SOURCE_PORTRAIT_ROLE)
        if asset is None:
            msg = (
                f"avatar {descriptor.id!r} declares no {SOURCE_PORTRAIT_ROLE!r} asset; "
                f"the warp engine needs a portrait photograph"
            )
            raise AvatarLoadError(msg)

        path = descriptor.asset_path(asset)
        if not path.is_file():
            msg = f"avatar {descriptor.id!r}: portrait not found at {path}"
            raise AvatarLoadError(msg)

        return self._read_image(path)

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            msg = f"{path.name} could not be read as an image"
            raise AvatarLoadError(msg)
        loaded: np.ndarray = image
        return loaded

    def _landmarks_of(self, image: np.ndarray, avatar_id: str) -> np.ndarray:
        """Find the portrait's own landmarks, in pixels.

        The same tracker that drives the pipeline is used, so the avatar's mesh
        and the live mesh are guaranteed to be the same topology — a different
        detector here would silently pair up unrelated points.
        """
        from sha256.tracking.mediapipe_tracker import (
            DEFAULT_MODEL_RELATIVE_PATH,
            MediaPipeFaceTracker,
        )

        model_path = self._model_path or (Path("models") / DEFAULT_MODEL_RELATIVE_PATH)
        tracker = MediaPipeFaceTracker(model_path=model_path)
        tracker.start()
        try:
            result = tracker.process(
                VideoFrame(
                    data=np.ascontiguousarray(image),
                    pixel_format=PixelFormat.BGR24,
                    metadata=FrameMetadata(frame_id=0, capture_timestamp=0.0, source_id="portrait"),
                )
            )
        finally:
            tracker.stop()

        if not result.face_detected or result.landmarks is None:
            msg = (
                f"avatar {avatar_id!r}: no face was found in the portrait. "
                f"It needs a clear, front-facing photograph of one face."
            )
            raise AvatarLoadError(msg)

        points = _pixel_points(result.landmarks.points, image.shape[1], image.shape[0])
        if points is None:
            msg = f"avatar {avatar_id!r}: the portrait's landmarks are unusable"
            raise AvatarLoadError(msg)
        return points


def _pixel_points(normalised: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """The mesh subset in pixel coordinates, or ``None`` if the mesh is wrong."""
    if normalised.shape[0] <= max(MESH_INDICES):
        return None

    subset = normalised[list(MESH_INDICES), :2].astype(np.float32)
    subset[:, 0] *= width
    subset[:, 1] *= height
    return subset


def _triangulate(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Delaunay triangulation over the mesh points, as index triples.

    Indices rather than coordinates: the triangulation is computed once on the
    portrait and reused every frame against the live points, which is what makes
    the two meshes correspond.
    """
    import cv2

    # A rectangle large enough that no point sits on its edge, where Subdiv2D
    # returns triangles referencing the virtual corner vertices.
    padding = max(width, height)
    subdiv = cv2.Subdiv2D((-padding, -padding, width + 2 * padding, height + 2 * padding))
    for point in points:
        subdiv.insert((float(point[0]), float(point[1])))

    lookup = {(round(float(p[0]), 2), round(float(p[1]), 2)): i for i, p in enumerate(points)}

    triangles: list[tuple[int, int, int]] = []
    for triangle in subdiv.getTriangleList():
        corners = [
            lookup.get((round(float(triangle[i]), 2), round(float(triangle[i + 1]), 2)))
            for i in (0, 2, 4)
        ]
        if all(corner is not None for corner in corners):
            triangles.append((corners[0], corners[1], corners[2]))  # type: ignore[arg-type]

    if not triangles:
        msg = "the avatar mesh could not be triangulated"
        raise AvatarLoadError(msg)

    return _without_mouth_interior(np.array(triangles, dtype=np.int32), points)


def _without_mouth_interior(triangles: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Drop the triangles that span the mouth opening.

    Delaunay fills the mouth with long thin triangles between opposite lips.
    Warping each one stretches its own strip of the photograph independently,
    and the teeth come out as vertical stripes. The interior is painted
    separately instead, by a transform fitted to the lips as a whole, so the
    teeth keep their proportions.
    """
    import cv2

    mouth = points[_INNER_LIP_SLICE].astype(np.int32)
    centroids = points[triangles].mean(axis=1)

    keep = [
        index
        for index, centroid in enumerate(centroids)
        if cv2.pointPolygonTest(mouth, (float(centroid[0]), float(centroid[1])), False) < 0
    ]
    kept: np.ndarray = triangles[keep]
    return kept


def _warp_onto(
    avatar: _LoadedAvatar,
    target: np.ndarray,
    scene: np.ndarray,
    tracking: FaceTrackingResult,
) -> np.ndarray:
    """Warp the avatar's mesh onto ``target`` and blend it over ``scene``.

    Written around one measured fact: the cost is dominated by *per-call*
    overhead, not by pixels. An earlier version warped both the image and a mask
    for every triangle and cost 42 ms at p95, over the whole 33 ms frame budget.

    Three things follow from that, and they are the reason this reads the way it
    does rather than the obvious way:

    *   **One warp per triangle, not two.** The blend mask is built once, from
        the destination convex hull, instead of being warped alongside each
        piece.
    *   **Everything happens inside the face's bounding box.** A 1280x720 canvas
        is allocated and cleared per frame otherwise, for a face that occupies a
        fraction of it.
    *   **No boolean fancy-indexing on the hot path.** ``copyTo`` with a mask
        does the same job without building an index array per triangle.
    """

    height, width = scene.shape[:2]

    # Work in the face's bounding box, padded, and clipped to the frame.
    #
    # The box is taken from the *mask's* outline rather than from the landmarks,
    # because the mask reaches past them: below the jaw, where the avatar has to
    # continue onto the neck or its beard ends in a hard curve. Sized from the
    # landmarks instead, the mask was cut off at the box edge and the cut showed
    # as a faint rectangle under the chin.
    pad = 8
    outline = _trim_forehead(target)
    fx0 = max(int(min(target[:, 0].min(), outline[:, 0].min())) - pad, 0)
    fy0 = max(int(min(target[:, 1].min(), outline[:, 1].min())) - pad, 0)
    fx1 = min(int(max(target[:, 0].max(), outline[:, 0].max())) + pad, width)
    fy1 = min(int(max(target[:, 1].max(), outline[:, 1].max())) + pad, height)
    if fx1 - fx0 < 2 or fy1 - fy0 < 2:
        return scene.copy()

    box_w, box_h = fx1 - fx0, fy1 - fy0
    # Cast back explicitly: subtracting an int tuple promotes to float64, and
    # OpenCV's geometry functions only accept 32-bit float or int.
    local = (target - (fx0, fy0)).astype(np.float32)

    blended = _blend_sources(avatar, tracking)

    # The mesh that describes `blended`, not the base portrait's. They differ
    # exactly where an expression photograph brought its own geometry — an open
    # mouth, a closed eyelid — and warping from the base's mesh instead was what
    # crushed the teeth into a sliver and stretched the eyelids into streaks.
    source_mesh = avatar.blend_points
    canvas = warp_mesh(
        blended,
        source_mesh,
        _damp_brows(source_mesh, local, avatar.eye_follow),
        avatar.triangles,
        box_w,
        box_h,
    )

    _paint_mouth_interior(avatar, canvas, blended, local)

    alpha = _destination_mask(local, box_w, box_h)

    output = scene.copy()
    region = output[fy0:fy1, fx0:fx1].astype(np.float32)
    matched = _match_colour(avatar, canvas.astype(np.float32), region, alpha)
    blended = region * (1.0 - alpha) + matched * alpha
    output[fy0:fy1, fx0:fx1] = blended.astype(np.uint8)
    return output


def _cavity_colour(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """The colour of the back of this mouth, from the portrait's own opening.

    It stands in for cavity depth where no photograph reaches — the corners of
    a wide opening, and every frame of an avatar whose photographs all have
    their lips together.
    """
    import cv2

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points[_INNER_LIP_SLICE].astype(np.int32)], 255)

    inside = mask > 0
    if not bool(inside.any()):
        return np.zeros(3, dtype=np.float32)

    # The darker half of the opening: teeth are the bright part and would make
    # a poor stand-in for the back of a throat.
    pixels = image[inside].astype(np.float32)
    luminance = pixels.mean(axis=1)
    darker = pixels[luminance <= float(np.median(luminance))]
    cavity: np.ndarray = darker.mean(axis=0) if len(darker) else pixels.mean(axis=0)
    return cavity


def _paint_mouth_interior(
    avatar: _LoadedAvatar, canvas: np.ndarray, image: np.ndarray, target: np.ndarray
) -> None:
    """Fill the mouth opening from the photograph, in one piece.

    The mesh no longer covers this area, because triangulating across an opening
    slices whatever is behind it into strips. The mouth region is placed instead
    by a single similarity transform fitted to the lip landmarks, so the teeth
    arrive at the right size and stay one object.

    **Only the blend's own mouth interior is copied.** Placing the whole image
    and trusting the fit put the photograph's *lips* inside the live opening
    whenever the user opened wider than the photograph did — a second, smaller
    mouth inside the first, which is what the owner saw. Anything no photograph
    covers is filled with the cavity colour instead.

    **Which pixels those are is asked of the mesh, not of the portrait**, and
    getting that wrong is what emptied the mouth. The interior used to be a mask
    cut from the *base portrait* at load, and the base portrait has its lips
    together — a sliver a few pixels tall. So however wide the blend's mouth
    was, only that sliver was ever copied and the rest of the opening stayed
    flat cavity colour: no teeth on a smile, and neither teeth nor tongue on a
    wide open mouth, which is exactly what the owner reported seeing.

    **And it comes from one photograph rather than from the blend**, which is
    the difference between teeth and a ghost of them. The lips may cross-fade;
    what is behind them may not, because the photographs disagree about it — at
    the owner's widest smile the mouth was 43% a photograph with teeth and 57%
    three with their lips together, and 43% of a tooth is a translucent one.
    `_blend_sources` names the photograph, the most-parted one active this
    frame, and this places it whole. Its own inner lips are both the fit and the
    limit of what is copied, so the teeth arrive opaque and no lip ever does.

    With no such photograph in play the blend itself is used, which for a face
    with its lips together is a sliver and leaves the opening as cavity — the
    same answer as before, by the same route.
    """
    import cv2

    # The photograph the interior is coming from, and the mesh that describes
    # it: its own, so the fit maps its opening onto the live one. Falling back
    # to the blend keeps a lipsful of cavity for an avatar that has no parted
    # photograph at all.
    source = avatar.sources.get(avatar.interior_role or "")
    if source is None:
        source, source_points = image, avatar.blend_points
    else:
        source_points = avatar.source_points.get(avatar.interior_role or "", avatar.blend_points)

    # Fitted on the upper lip alone — see ``_UPPER_INNER_LIPS``. The teeth hang
    # from it; the jaw drops away from it.
    matrix: Any = cv2.estimateAffinePartial2D(
        source_points[_UPPER_LIP_POSITIONS], target[_UPPER_LIP_POSITIONS], method=cv2.LMEDS
    )[0]
    if matrix is None:
        return

    # Work in the mouth's own bounding box. Warping the whole photograph across
    # the entire face box to fill an opening a fifth of its size was most of
    # this function's cost.
    height, width = canvas.shape[:2]
    lips = target[_INNER_LIP_SLICE]
    x0 = max(int(lips[:, 0].min()) - 2, 0)
    y0 = max(int(lips[:, 1].min()) - 2, 0)
    x1 = min(int(lips[:, 0].max()) + 2, width)
    y1 = min(int(lips[:, 1].max()) + 2, height)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return

    box_w, box_h = x1 - x0, y1 - y0
    shifted = matrix.copy()
    shifted[0, 2] -= x0
    shifted[1, 2] -= y0

    opening = np.zeros((box_h, box_w), dtype=np.uint8)
    cv2.fillPoly(opening, [(lips - (x0, y0)).astype(np.int32)], 255)

    region = canvas[y0:y1, x0:x1]

    # Everything inside the live opening starts as cavity, so no part of it can
    # ever show a lip or a cheek.
    region[opening > 0] = avatar.cavity.astype(np.uint8)

    placed = cv2.warpAffine(
        source,
        shifted,
        (box_w, box_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    # Where that photograph has an interior to give: its own inner lips, moved
    # by the same fit. Drawn as a polygon rather than warped as a mask — it is
    # the same answer without a second resampling, and it follows the picture
    # wherever the picture has gone, a turned head included.
    interior = cv2.transform(source_points[_INNER_LIP_SLICE].reshape(-1, 1, 2), shifted).reshape(
        -1, 2
    )
    covered = np.zeros((box_h, box_w), dtype=np.uint8)
    cv2.fillPoly(covered, [interior.astype(np.int32)], 255)

    cv2.copyTo(placed, cv2.bitwise_and(opening, covered), region)


def _damp_brows(
    source: np.ndarray, target: np.ndarray, eye_follow: float = _EYE_FOLLOW
) -> np.ndarray:
    """Pull the brow and eye landmarks back toward the avatar's own proportions.

    The mesh maps the avatar's landmarks onto the user's, so wherever it acts at
    full strength the avatar inherits the user's *shape* and not merely their
    motion. Two places where that is visibly wrong:

    **Brows.** Raising them moves those landmarks a long way and the triangles
    between brow and eyelid answer by stretching the eye — the owner's report
    that the eyes grow when the brows go up.

    **Eyes**, and this one is worse. Forcing the avatar's eye aperture to match
    the user's stretches its iris to fill someone else's eye, which on a live
    frame reads as pale, flat, lifeless eyes. It is the single feature that gives
    the whole picture away, and it needs the mesh least: blinking, squinting,
    widening and gaze all arrive as photographs now, carrying their own geometry
    in the blended mesh. So the eyes follow the user barely at all — *while a
    photograph is doing the work*. ``eye_follow`` is raised by the caller when
    one was asked for and could not be placed, which is every eye expression at
    a turn: the mesh takes over exactly as much as the photograph loses, so a
    blink at an angle still reads as a blink rather than freezing.

    The reference for both is where a rigid fit of the whole face would put them
    — the avatar's own proportions, positioned on the live head.
    """
    import cv2

    matrix: Any = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)[0]
    if matrix is None:
        return target

    rigid = cv2.transform(source.reshape(-1, 1, 2), matrix).reshape(-1, 2)

    damped = target.copy()
    for where, follow in ((_BROW_SLICE, _BROW_FOLLOW), (_EYE_POSITIONS, eye_follow)):
        anchor = rigid[where]
        damped[where] = anchor + (damped[where] - anchor) * follow

    _place_irises(damped, target, rigid)
    return damped


def _place_irises(damped: np.ndarray, target: np.ndarray, rigid: np.ndarray) -> None:
    """Put the avatar's own iris where the user's is looking. In place.

    Gaze is a *direction*, not a shape, so what transfers is the iris's position
    **relative to its own socket** — measured on the user, applied to the avatar.
    The avatar keeps its own iris size and its own eye; only the pupil moves, and
    it moves continuously to any angle.

    Everything else about the eye is already the avatar's: ``_EYE_FOLLOW`` holds
    the aperture at the avatar's own proportions. Positioning the iris against
    ``damped`` rather than against the user's socket is what keeps the two
    consistent — an iris placed in a socket it does not belong to would sit on
    the eyelid.
    """
    for eye, iris in _EYE_PAIRS:
        user_socket = target[eye]
        user_width = float(user_socket[:, 0].max() - user_socket[:, 0].min())
        if user_width < 1.0:
            continue

        # Where the user is looking, as a fraction of their own eye's width,
        # scaled back and held inside the range the mesh can carry cleanly.
        gaze = (target[iris].mean(axis=0) - user_socket.mean(axis=0)) / user_width
        gaze = gaze * _GAZE_SCALE
        gaze[0] = min(max(float(gaze[0]), -_GAZE_LIMIT_X), _GAZE_LIMIT_X)
        gaze[1] = min(max(float(gaze[1]), -_GAZE_LIMIT_UP), _GAZE_LIMIT_DOWN)

        own_socket = damped[eye]
        own_width = float(own_socket[:, 0].max() - own_socket[:, 0].min())

        # The avatar's iris, at its own size, centred where that gaze puts it in
        # the avatar's own socket. Then held inside the lid, by the same rule
        # the source mesh is held to — see :func:`_contain_irises`.
        wanted = own_socket.mean(axis=0) + gaze * own_width
        damped[iris] = (rigid[iris] - rigid[iris].mean(axis=0)) + wanted
        _contain_iris(damped, eye, iris)


def _contain_irises(points: np.ndarray) -> None:
    """Hold both irises inside the eyelids that are supposed to contain them."""
    for eye, iris in _EYE_PAIRS:
        _contain_iris(points, eye, iris)


def _contain_iris(points: np.ndarray, eye: list[int], iris: list[int]) -> None:
    """Shrink one iris about its centre until it sits inside its lid. In place.

    **MediaPipe reports the whole iris circle, including the part behind the
    eyelid.** That is not an error in the tracker — it fits a circle to the
    visible arc — but it makes the mesh describe something the photograph does
    not contain. Measured on the owner's own photographs, in pixels of the
    avatar's own frame:

    =========================  =====  ======  =============
    photograph                   lid    iris  iris outside
    =========================  =====  ======  =============
    base portrait               29.0    37.3           8.3
    eyes wide open              38.2    39.8           1.6
    **eyes closed**             12.9    30.0          17.0
    =========================  =====  ======  =============

    So on a blink the mesh says there is an iris standing seventeen pixels proud
    of a thirteen-pixel eye. The triangles between the lid ring and the iris are
    inside-out, and warping them drags the brow and the socket down into the
    eye. **That is the hard-edged light wedge above a blinking eye**, and it is
    why the full eyelid ring alone only shrank it: the ring stopped the long
    triangles sweeping in from the cheek, and this is a different, smaller fan
    living inside the eye itself.

    The mesh warps pixels, and the pixels above a shut lid are eyelid rather
    than iris, so the honest mesh is one where the iris ends where the lid does.
    Shrinking is a stand-in for occlusion — the truthful operation would clip
    the circle, which a triangulated mesh cannot express — and it costs nothing
    visually because the part being removed is behind the lid either way.

    Applied to the source mesh and to the target by the same rule, which is the
    point: what a warp cannot tolerate is the two disagreeing.
    """
    lid = points[eye][:, 1]
    top, low = float(lid.min()), float(lid.max())

    own = points[iris]
    centre = float(own[:, 1].mean())
    reach = float(np.abs(own[:, 1] - centre).max())
    if reach <= 1e-6:
        return

    # The pupil stays where it was put — that is the gaze, and it is the one
    # thing here worth preserving. Clamping it into the opening is a safety net
    # for a lid that has shut around a pupil looking hard up or down.
    centre = min(max(centre, top), low)
    room = min(centre - top, low - centre) * _IRIS_OF_OPENING

    # Vertically only. A lid occludes an iris from above and below and leaves
    # its width alone, so squashing describes what is happening; shrinking the
    # whole disc would narrow a pupil that has not narrowed.
    points[iris, 1] = centre + (own[:, 1] - centre) * min(room / reach, 1.0)


def _reduce_to(
    image: np.ndarray, points: np.ndarray, base_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink a source to roughly the base portrait's scale, with area averaging.

    Every source is warped onto the base portrait's landmark layout, so the base
    fixes the working resolution. A source photographed much larger — a face
    filling a 4K frame against the base's 455 pixels — is therefore reduced, and
    ``warpAffine`` reduces by point-sampling however far it has to go. Sampling
    one pixel in four throws away three and aliases what is left, which is why a
    4K source measured *softer* on screen than a smaller one: 298 against 317.

    ``INTER_AREA`` averages instead, so the detail arrives as detail rather than
    as noise. Done once at load; the frame path never sees it.

    Sources already near the base's scale are left alone — resampling twice is
    worse than the mild reduction the mesh does anyway.
    """
    import cv2

    def face_width(pts: np.ndarray) -> float:
        return float(pts[:, 0].max() - pts[:, 0].min())

    ratio = face_width(points) / max(face_width(base_points), 1e-6)
    if ratio < _REDUCE_THRESHOLD:
        return image, points

    scale = 1.0 / ratio
    reduced = cv2.resize(
        image,
        (max(int(image.shape[1] * scale), 1), max(int(image.shape[0] * scale), 1)),
        interpolation=cv2.INTER_AREA,
    )
    return reduced, (points * scale).astype(np.float32)


def _crop_to_face(image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim a portrait to its face, and move the landmarks with it.

    The padding is generous — the mesh reaches past the outline at the jaw and
    brow, and the rigid backdrop layer wants some context around it — but a
    portrait is mostly shoulders and wall, and every one of those pixels would
    otherwise be blended on every frame.
    """
    height, width = image.shape[:2]
    face_width = float(points[:, 0].max() - points[:, 0].min())
    pad = max(int(face_width * 0.35), 16)

    x0 = max(int(points[:, 0].min()) - pad, 0)
    y0 = max(int(points[:, 1].min()) - pad, 0)
    x1 = min(int(points[:, 0].max()) + pad, width)
    y1 = min(int(points[:, 1].max()) + pad, height)

    cropped = np.ascontiguousarray(image[y0:y1, x0:x1])
    moved = (points - (x0, y0)).astype(np.float32)
    return cropped, moved


def _lip_gap_of(points: np.ndarray) -> float:
    """The lip gap of an already-subsetted mesh, as a fraction of face height.

    Used at load to learn each photograph's own opening, so "fully open" is
    whatever the widest photograph supplied actually shows rather than a number
    written here.
    """
    inner = points[_INNER_LIP_SLICE]
    gap = float(inner[:, 1].max() - inner[:, 1].min())
    height = float(points[:, 1].max() - points[:, 1].min())
    return gap / height if height > 1e-6 else 0.0


def _eye_opening(tracking: FaceTrackingResult) -> float:
    """How open the eyes are: lid height over lid width, the narrower eye.

    **The blendshape cannot answer this**, in the same way and for the same
    reason ``mouthSmile`` could not answer whether the teeth show.
    ``eyeBlink`` is trained on blinks, which are fast: measured on the owner's
    own recording, a blink peaks at 0.66 and *holding the eyes shut* reads
    0.14 — lower than some frames where they are plainly open. Two seconds of
    shut eyes therefore scored as an ordinary squint, and the avatar squinted
    back at a person whose eyes were closed.

    Whether the eye is shut is a question about the eyelids, so it is measured
    on the eyelids. That reading only became available this morning: it needs
    the full lid ring, and the mesh carried six points per eye until then.

    The **more open** of the two eyes, which is the same rule the blendshape
    version used and for the same reason: the photograph has both eyes shut, so
    driving it from one closing eye would shut the other one too. A wink has to
    read as no closure at all until the open eye follows.

    **Against face height, not eye width**, and that correction is worth stating
    because the first version got it wrong. The landmarks are normalised over
    the frame, so a height divided by a width carries the frame's aspect ratio
    as a factor. Live that cancels — every frame is the same shape — but the
    frame-extraction tool measures a 2160x3840 clip against a 1024x1024
    portrait, where the same eye reads 0.56 of what it should. Every frame of
    every clip then scored as a hard blink, and the squint and wide-eye
    photographs, which give way to a closure, scored zero: three eye clips
    yielded no `eyes_wide` and no `squint` at all.

    Two vertical distances divided by each other have no such factor, and the
    lip gap has been measured that way since the teeth were fixed for exactly
    this reason.
    """
    landmarks = tracking.landmarks
    if landmarks is None or landmarks.count <= max(max(_LEFT_EYE), max(_RIGHT_EYE)):
        return 0.0

    points = landmarks.points
    if landmarks.count <= max(_FOREHEAD_LANDMARK, _CHIN_LANDMARK):
        return 0.0

    face = float(points[_CHIN_LANDMARK, 1] - points[_FOREHEAD_LANDMARK, 1])
    if face <= 1e-6:
        return 0.0

    openings = [
        float(points[list(ring), 1].max() - points[list(ring), 1].min()) / face
        for ring in (_LEFT_EYE, _RIGHT_EYE)
    ]
    return max(openings)


def _lip_gap(tracking: FaceTrackingResult) -> float:
    """How far the lips are actually parted, as a fraction of face height.

    The blendshapes cannot answer this. ``mouthSmile`` reads 0.54 on the
    avatar's smiling photograph and reads just as high on a closed-mouth smile,
    so weighting the teeth-bearing photographs by it put teeth into a smile that
    had none — which is exactly what the owner saw. Whether teeth can be visible
    is a question about the lips, so it is measured on the lips.
    """
    landmarks = tracking.landmarks
    if landmarks is None or landmarks.count <= max(_LOWER_INNER_LIP, _CHIN_LANDMARK):
        return 0.0

    points = landmarks.points
    gap = abs(float(points[_UPPER_INNER_LIP][1] - points[_LOWER_INNER_LIP][1]))
    height = abs(float(points[_FOREHEAD_LANDMARK][1] - points[_CHIN_LANDMARK][1]))
    if height < 1e-6:
        return 0.0
    return gap / height


def _source_weights(
    tracking: FaceTrackingResult,
    full_gap: float = _LIP_GAP_FULL,
    neutral: _Neutral | None = None,
) -> dict[str, float]:
    """How much each extra photograph should contribute to this frame.

    Read from the tracker: the blendshapes already measure "how closed is this
    eye", the yaw was verified in B003, and the lip gap is measured from the
    landmarks. Nothing here needs a state machine, which is what keeps it from
    flickering between sources.

    Two rules make the difference between a face and a double exposure, and both
    were learned by measuring a still subject rather than by reasoning:

    *   **Everything is relative to the resting face** (:class:`_Neutral`).
        Absolute readings made a motionless person score eleven simultaneous
        expressions, because a relaxed face does not read as zero and a webcam
        is never exactly on axis.
    *   **Opposing photographs are mutually exclusive.** Looking left and
        looking right were both active on 98% of frames, and up and down on
        100%, so the engine averaged four gaze directions into eyes that pointed
        nowhere. Each pair is now one signed quantity: the stronger side wins and
        the other is zero, by construction rather than by luck.
    """
    weights: dict[str, float] = {}
    shapes = tracking.blendshapes
    neutral = neutral if neutral is not None else _Neutral()

    # Both photographs that contain teeth are driven by the lip gap. Neither can
    # contribute anything truthful while the mouth is shut, and blending them in
    # anyway is how a closed smile grew teeth.
    gap = _lip_gap(tracking)

    if shapes is not None and shapes.names:
        # How much of an *open mouth* this frame needs. At full opening the base
        # portrait must contribute nothing to the mouth: its lips are closed, and
        # a closed mouth stretched across an open one is what produced fangs.
        #
        # The reference is the *narrower* of the photograph's own gap and a
        # normal wide-open speaking mouth. Both halves of that matter and each
        # was learned the hard way:
        #
        # *   A photograph narrower than a speaking mouth cannot depict more
        #     than it contains, so it must reach full weight at its own gap.
        # *   A photograph *wider* must not push ordinary speech down the scale.
        #     The owner's replacement open-mouth photograph measured 0.229
        #     against the 0.095 of a normal opening, so dividing by it made a
        #     smile score 0.13 and the teeth stopped appearing at all — the
        #     photograph was there, and the weight that would have shown it was
        #     being measured against a yawn.
        reference = min(full_gap, _LIP_GAP_FULL)
        opening = (gap - _LIP_GAP_CLOSED) / max(reference - _LIP_GAP_CLOSED, 1e-6)
        opening = min(max(opening, 0.0), 1.0)

        # Which of the two open-mouth photographs it should come from: a smile
        # shows teeth differently from an "ah". They split the same budget
        # rather than adding to each other.
        smile = max(shapes.get("mouthSmileLeft"), shapes.get("mouthSmileRight"))
        smiling = min(smile / _SMILE_REFERENCE, 1.0)
        weights["source_smile"] = opening * smiling
        weights["source_mouth_open"] = opening * (1.0 - smiling)

        # min, not max, of the two eyes: a wink is one eye, and driving a
        # both-eyes-closed photograph from a wink closes the wrong one too.
        #
        # Measured against rest like every other expression, and it is the one
        # that most needed it. A relaxed eye reads about 0.26 on this tracker,
        # against a 0.20 threshold, so the closed-eye photograph was blended in
        # permanently at roughly 15% — a translucent shut eyelid laid over an
        # open eye, which is exactly the "two sets of eyes" the owner saw. Open
        # eyes are the floor by construction, since blinking only moves this
        # reading upward.
        # One measurement, three photographs, because an eye has one opening.
        # Named for the lids: `opening` a few lines above is the *mouth's*.
        lids = _eye_opening(tracking)
        shut = neutral.relative_drop("eyeOpen", lids)
        opened = neutral.relative_rise("eyeOpen", lids)

        def band(value: float, start: float, full: float) -> float:
            return min(max((value - start) / (full - start), 0.0), 1.0)

        closed = band(shut, _CLOSE_START, _CLOSE_FULL)
        weights["source_eyes_closed"] = closed

        def expression(*names: str) -> float:
            """How far into an expression this frame is, above the resting face."""
            above_rest = max(neutral.relative_expression(name, shapes.get(name)) for name in names)
            return min(above_rest / _EXPRESSION_FULL, 1.0)

        def opposed(positive: float, negative: float) -> tuple[float, float]:
            """One signed quantity split into two photographs, never both at once."""
            difference = positive - negative
            return max(difference, 0.0), max(-difference, 0.0)

        weights["source_brows_up"], weights["source_brows_down"] = opposed(
            expression("browInnerUp", "browOuterUpLeft", "browOuterUpRight"),
            expression("browDownLeft", "browDownRight"),
        )
        # Wide and narrowed come off the same lids, and **not** off `eyeWide`
        # and `eyeSquint`, which do not measure what their names say. Held over
        # a clip generated to show each expression in turn:
        #
        #     what the face is doing   lid opening   eyeWide   eyeSquint
        #     resting                       +0.00      0.00        0.51
        #     eyes held wide open           +0.53      0.13        0.07
        #     eyes held shut                -0.66      0.00        0.89
        #
        # `eyeWide` never rises above 0.13 even on eyes visibly at their widest,
        # so the wide-eye photograph scored 0.23 against a 0.45 floor and was
        # never once selected from three clips. And `eyeSquint` reads *highest*
        # on a shut eye: it is a second, worse closure signal wearing the name
        # of a different expression, which is why it was taking half of every
        # blink.
        #
        weights["source_eyes_wide"] = band(opened, _WIDE_START, _WIDE_FULL)

        # **The squint photograph is deliberately not used**, and this is the
        # second family to be dropped after the four gaze photographs.
        #
        # A squint is a partial closure, and partial closure is what an eye does
        # all day: glancing aside, speaking, small head movements. Measured over
        # the owner's whole recording, a squint band anywhere below a blink was
        # active on **41% of frames** — the avatar wore a quarter of a squint
        # through most of a conversation, on a face that was only talking. That
        # is invisible in any single frame and wrong in all of them, which is
        # how it survived every previous round of looking at stills.
        #
        # There is no threshold that separates it. Ordinary lid movement reaches
        # 0.47 of the eye's opening and a blink reaches 0.65, so a squint band
        # either fires constantly or never fires at all.
        #
        # What would separate it is a different measurement: a squint raises the
        # **lower** lid and tightens the cheek, where a blink lowers the upper
        # one. Both lids are in the mesh, so that is available — it is simply not
        # written yet, and a photograph used wrongly half the time is worse than
        # a photograph not used.
        weights["source_frown"] = expression("mouthFrownLeft", "mouthFrownRight")
        weights["source_mouth_pressed"] = expression("mouthPressLeft", "mouthPressRight")

        # A closed-mouth smile is a smile with the lips still together, which is
        # exactly what the gap measurement distinguishes.
        weights["source_smile_closed"] = smiling * (1.0 - opening)

        # Rounded and funnelled lips are shapes rather than openings, so they
        # are driven by their own blendshapes and not by the gap.
        weights["source_mouth_o"] = expression("mouthPucker", "mouthFunnel")
        weights["source_mouth_wide"] = expression("mouthStretchLeft", "mouthStretchRight")

        # Gaze is deliberately absent. It is handled as geometry — the iris is
        # in the mesh and is placed where the user is looking — because a
        # photograph cannot express it. Blending two gaze photographs puts the
        # pupil in two places, and blending one with the base does the same
        # against a forward-looking iris; both render as the "ghost eye" the
        # owner reported. Moving the pupil is continuous, exact at any angle,
        # and costs nothing. See _place_irises.

    pose = tracking.head_pose
    if pose is not None:
        # Relative to how this person sits, not to the camera's idea of straight
        # ahead. A camera above a monitor reads a permanent downward tilt, and
        # driving a pose photograph from that put a permanent tilt in the
        # avatar's face while the user was looking straight at them.
        yaw = neutral.relative_pose("yaw", pose.yaw)

        turn = yaw / _TURN_FULL_DEGREES
        weights["source_turn_left"] = turn
        weights["source_turn_right"] = -turn

        far = yaw / _TURN_FAR_DEGREES
        weights["source_turn_left_far"] = far
        weights["source_turn_right_far"] = -far

        # **The tilt photographs are deliberately not used**, and this is the
        # third family to be dropped after the four gaze photographs and the
        # squint. The reason is the same each time: a photograph that is wrong
        # more often than it is right costs more than it gives.
        #
        # Measured over the owner's whole recording, a tilt photograph was
        # active on **99.2% of frames** at a median weight of **0.38** — nearly
        # 40% of every frame of an ordinary conversation was a different
        # photograph mixed over the portrait. That was survivable only while
        # every source was folded onto the frontal layout, where it read as
        # shading. Now that a pose photograph brings its own geometry, the mesh
        # follows it, and the mesh it follows is wrong: the landmarks of
        # `clip_tilt_down.png` sit **47.5 px** from the base's on a 455 px face,
        # which is a tenth of a face, for a photograph only about six degrees
        # further down than the portrait. It is not a pitch axis — it is
        # another frame of the same person, differing in more than pitch.
        #
        # The cost, on the recorded clip: the avatar's mesh sits 4.2 px further
        # from the tracked face with the tilts in than with them out (11.0 px
        # against 6.8), on 99% of frames. Rendered, they are a face pulled
        # downward with hollows under the eyes, and everything is softer,
        # because cross-fading two pictures that do not agree is a blur.
        #
        # What would make a pitch photograph usable is what already made the
        # turns usable: a *measured* angle on the asset and a ladder of them
        # from a nod clip, so the weight is chosen by the angle it was shot at
        # rather than by a guessed full-scale number. `extract_avatar_frames.py`
        # already records yaw per frame; pitch is the same measurement.

    chosen = {
        role: min(max(value, 0.0), _MAX_VARIANT_WEIGHT)
        for role, value in weights.items()
        if value > 0.02
    }

    # Gaze is the one family where mixing does not average an *appearance*, it
    # averages a *position*: a look left and a look up put the pupil in two
    # places and the blend puts a ghost of it in neither. Keep the strongest and
    # drop the rest — a pupil somewhere definite, roughly in the right
    # direction, beats a precise average of two wrong ones.
    return chosen


def _place_by_similarity(
    image: np.ndarray,
    points: np.ndarray,
    base_points: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Put a photograph on the base's frame without reshaping the face in it.

    Rotation, uniform scale and translation only, fitted on the landmarks an
    expression does not move — the face outline and the nose bridge. Fitting on
    all of them would let the expression itself drag the alignment, which is
    self-defeating for a photograph taken *because* of that expression.

    Returns the placed image and its landmarks moved by the same transform, so
    that the two continue to describe each other. ``None`` when the fit fails,
    which is rare enough to be worth a warning at the call site rather than a
    silent fallback here.
    """
    import cv2

    # Typed as Any: OpenCV's stubs declare a non-optional return, but the fit
    # genuinely can fail and returns None when it does.
    matrix: Any = cv2.estimateAffinePartial2D(
        points[_STABLE_POSITIONS, :2].astype(np.float32),
        base_points[_STABLE_POSITIONS, :2].astype(np.float32),
        method=cv2.LMEDS,
    )[0]
    if matrix is None:
        return None

    placed = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_AREA if _shrinks(matrix) else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    moved = points.copy()
    moved[:, :2] = cv2.transform(
        points[:, :2].reshape(-1, 1, 2).astype(np.float32), matrix
    ).reshape(-1, 2)
    return placed, moved


def _shrinks(matrix: np.ndarray) -> bool:
    """Whether a similarity transform reduces the image.

    INTER_AREA is the correct filter for a reduction and INTER_LINEAR is not —
    bilinear samples a 2x2 neighbourhood and discards the rest, which on a
    photograph being shrunk is most of it.
    """
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    return scale < 1.0


def _build_regions(
    image: np.ndarray,
    points: np.ndarray,
    sources: dict[str, np.ndarray],
    source_points: dict[str, np.ndarray] | None = None,
) -> tuple[_Region, ...]:
    """Precompute each region's box, feathered mask, and cropped photographs.

    Regions whose photographs the avatar does not have are not built at all, so
    an avatar with only a base portrait costs nothing here.

    **Each region is sized to hold every photograph that acts on it**, not just
    the base portrait's own landmarks. The base's lips are together, so a mouth
    box cut to them stops a little below the closed lower lip — and an
    open-mouth photograph reaches far past that. Everything below the cut was
    left as base portrait, which is a chin, and the mouth interior then copied
    that chin into the live opening: **stubble inside the mouth**, under the
    tongue. The same argument applies upward for a raised brow and outward for
    a widened eye.
    """
    import cv2

    height, width = image.shape[:2]
    face_width = max(float(points[:, 0].max() - points[:, 0].min()), 1.0)
    feather = max(3, int(face_width * _REGION_FEATHER) | 1)
    source_points = source_points or {}

    built: list[_Region] = []
    for name, positions in (
        ("mouth", _MOUTH_POSITIONS),
        ("eyes", _EYE_REGION_POSITIONS),
        ("brows", _BROW_POSITIONS),
    ):
        present = [role for role in _REGION_ROLES[name] if role in sources]
        if not present:
            continue

        margin = int(face_width * _REGION_MARGIN[name])
        group = points[positions]
        extras = [source_points[role][positions] for role in present if role in source_points]

        # The box holds the base's landmarks with their margin *and* every
        # photograph's own landmarks, so nothing a photograph describes is cut
        # off at the crop.
        corners = np.concatenate([group - margin, group + margin, *extras])
        x0 = max(int(corners[:, 0].min()), 0)
        y0 = max(int(corners[:, 1].min()), 0)
        x1 = min(int(corners[:, 0].max()), width)
        y1 = min(int(corners[:, 1].max()), height)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue

        # The mask is drawn in the box's own coordinates, as the convex hull of
        # the region's landmarks grown by the margin, then blurred. Blurring
        # inside the box means the edge fades to zero at the box border, so the
        # composite can never produce a seam at the rectangle itself.
        filled = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(filled, cv2.convexHull((group[:, :2] - (x0, y0)).astype(np.int32)), 255)
        grown = cv2.dilate(
            filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin | 1, margin | 1))
        )

        # Each photograph's own hull is added *without* the margin, and this is
        # the difference between an open mouth and a doubled lip. A photograph
        # may act wherever the mesh follows it — inside its own landmarks — and
        # no further: the chin below an open mouth belongs to the face outline,
        # which no region moves, so painting the photograph's chin there lands
        # it 30 px from where the mesh says the chin is. The margin exists to
        # blur the join around the *base's* features and has no business
        # extending an opening the mesh has never seen.
        for extra in extras:
            cv2.fillConvexPoly(
                grown, cv2.convexHull((extra[:, :2] - (x0, y0)).astype(np.int32)), 255
            )
        mask = cv2.GaussianBlur(grown, (feather, feather), 0).astype(np.float32) / 255.0

        built.append(
            _Region(
                name=name,
                box=(x0, y0, x1, y1),
                mask=mask[:, :, None],
                sources={
                    role: np.ascontiguousarray(sources[role][y0:y1, x0:x1]) for role in present
                },
                positions=list(positions),
            )
        )

    return tuple(built)


def _weighted_sum(
    base: np.ndarray, sources: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    """``base`` with each source mixed in at its weight, the base keeping the rest.

    ``addWeighted`` rather than numpy arithmetic, and uint8 throughout: the
    obvious float expression allocates a buffer per source per operation, which
    measured 31 ms of a 33 ms budget when this was written.
    """
    import cv2

    total = sum(weights.values())
    if total > 1.0:
        weights = {role: weight / total for role, weight in weights.items()}
        total = 1.0

    # Scaling by (1 - total) expressed as a weighted sum, which is the one
    # overload OpenCV types accept for a scalar multiply of an image.
    out = cv2.addWeighted(base, 1.0 - total, base, 0.0, 0.0)
    for role, weight in weights.items():
        out = cv2.addWeighted(out, 1.0, sources[role], weight, 0.0)
    return out


def _shares(weights: dict[str, float]) -> tuple[dict[str, float], float]:
    """The weights as *proportions of each other*, and how much they take in all.

    Splitting the one number the caller has into two is what lets a mix of
    photographs be built once and then placed somewhere the base is not — the
    proportions describe the picture, and the total says how much of it the
    base keeps. Weights asking for more than the whole face are normalised
    down, as they always were.
    """
    total = sum(weights.values())
    if total <= 0.0:
        return {}, 0.0
    return {role: weight / total for role, weight in weights.items()}, min(total, 1.0)


def _photo_mix(sources: dict[str, np.ndarray], shares: dict[str, float]) -> np.ndarray:
    """The photographs mixed among themselves, in the proportions given.

    The base is deliberately not part of it: this is the picture the sources
    describe on their own, so it can be carried onto a head that has moved
    before anything underneath is mixed with it.
    """
    import cv2

    (first, share), *rest = shares.items()
    out: np.ndarray = cv2.addWeighted(sources[first], share, sources[first], 0.0, 0.0)
    for role, weight in rest:
        out = cv2.addWeighted(out, 1.0, sources[role], weight, 0.0)
    return out


def _point_mix(
    avatar: _LoadedAvatar, shares: dict[str, float], where: list[int] | slice
) -> np.ndarray:
    """The same mix, applied to landmarks instead of pixels.

    Each photograph brings the geometry it was taken with, in the same
    proportions as its pixels, so the mesh and the picture stay descriptions of
    one face. A source with no recorded landmarks contributes the base's, which
    is how "this photograph changes nothing about where things are" is
    expressed.
    """
    (first, share), *rest = shares.items()
    mixed: np.ndarray = avatar.source_points.get(first, avatar.points)[where] * share
    for role, weight in rest:
        mixed = mixed + avatar.source_points.get(role, avatar.points)[where] * weight
    return mixed


def _carry_matrix(
    base: np.ndarray, head: np.ndarray, where: list[int]
) -> tuple[np.ndarray, float] | None:
    """How to move a part of the base portrait onto the head the blend is wearing.

    Rotation, uniform scale and translation, fitted on that part's own
    landmarks. It exists because two different things are now true at once: the
    whole-head photographs bring their own angle, and the expression
    photographs are still cut from a face looking straight ahead. A mouth
    photograph pasted where the *frontal* mouth was would sit 35 px off the
    turned one at 17 degrees and 70 px off at 36 — a second mouth beside the
    first, which is the very defect a turned ladder was meant to remove.

    A similarity cannot express foreshortening, so **how much it fails by is
    returned with it**: the RMS distance between where the fit puts each of the
    region's landmarks and where the head actually has them. That number is the
    ghost the caller would be painting, and it is what decides whether the
    photograph is applied at all — see ``_CARRY_LOST``.
    """
    import cv2

    matrix: Any = cv2.estimateAffinePartial2D(base[where], head[where], method=cv2.LMEDS)[0]
    if matrix is None:
        return None

    carried: np.ndarray = matrix
    moved = cv2.transform(base[where].reshape(-1, 1, 2), carried).reshape(-1, 2)
    error = float(np.sqrt((np.square(moved - head[where]).sum(axis=1)).mean()))
    return carried, error


def _trust(error: float) -> float:
    """How much of a photograph survives a misfit of ``error`` face widths.

    One either side and a straight line between: inside the noise between two
    photographs of one face it is applied in full, past a visible doubling it is
    not applied at all. Linear because a fade has to be continuous — a hard
    threshold would make an expression vanish mid-sentence as the head turned.
    """
    if error <= _CARRY_CLEAN:
        return 1.0
    if error >= _CARRY_LOST:
        return 0.0
    return (_CARRY_LOST - error) / (_CARRY_LOST - _CARRY_CLEAN)


def _carry_region(
    pixels: np.ndarray,
    mask: np.ndarray,
    matrix: np.ndarray,
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]] | None:
    """Move a region's pixels and its feathered mask by ``matrix``.

    Both together, because the mask is the region's edge and an edge left
    behind would show as a patch of frontal face at the old position. The
    destination box is the source box's corners transformed and clipped, so the
    work stays proportional to the region rather than to the frame.

    ``None`` when the region lands outside the picture entirely.
    """
    import cv2

    x0, y0, x1, y1 = box
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    moved = cv2.transform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)

    dx0 = max(int(np.floor(moved[:, 0].min())), 0)
    dy0 = max(int(np.floor(moved[:, 1].min())), 0)
    dx1 = min(int(np.ceil(moved[:, 0].max())) + 1, width)
    dy1 = min(int(np.ceil(moved[:, 1].max())) + 1, height)
    if dx1 - dx0 < 2 or dy1 - dy0 < 2:
        return None

    # The same transform, expressed from the source box's own coordinates into
    # the destination box's.
    shifted = matrix.astype(np.float64).copy()
    shifted[:, 2] = matrix[:, :2] @ np.array([x0, y0], dtype=np.float64) + matrix[:, 2] - (dx0, dy0)

    size = (dx1 - dx0, dy1 - dy0)
    flags = cv2.INTER_AREA if _shrinks(matrix) else cv2.INTER_LINEAR
    placed = cv2.warpAffine(pixels, shifted, size, flags=flags, borderMode=cv2.BORDER_REPLICATE)
    # Zero outside, so the region contributes nothing where it no longer covers
    # anything — the mask is what stops a hard edge appearing at the box.
    carried = cv2.warpAffine(
        mask[:, :, 0],
        shifted,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return placed, carried[:, :, None], (dx0, dy0, dx1, dy1)


def _ladder_weights(ladder: tuple[tuple[float, str | None], ...], yaw: float) -> dict[str, float]:
    """The two rungs that bracket this angle, weighted by how near each is.

    **Selection, not blending**, which is the whole point of a dense set. Two
    photographs averaged give an angle that is neither of them, and the further
    apart they are the less the average resembles a face; a rung every three
    degrees means the two being mixed are three degrees apart, and a mix of two
    angles three degrees apart is indistinguishable from either.

    Past the outermost rung it clamps rather than extrapolating. A photograph
    cannot show more of the cheek than was pointed at the camera, and stretching
    the furthest one further was the old behaviour and is visible as smearing.

    The base portrait's rung returns nothing: it is the accumulator that
    everything else is mixed *into*, so the weight it should have is exactly the
    weight left over, and naming it here would apply it twice.
    """
    if len(ladder) < 2:
        return {}

    if yaw <= ladder[0][0]:
        first = ladder[0][1]
        return {first: 1.0} if first is not None else {}
    if yaw >= ladder[-1][0]:
        last = ladder[-1][1]
        return {last: 1.0} if last is not None else {}

    upper = bisect_left(ladder, (yaw,))
    below, above = ladder[upper - 1], ladder[upper]
    span = above[0] - below[0]
    share = (yaw - below[0]) / span if span > 1e-6 else 0.0

    # A rung at zero weight is dropped rather than carried: it would cost a
    # cache key and break "a resting face uses no sources at all", which is the
    # property that makes a still avatar byte-identical to its own photograph.
    weights: dict[str, float] = {}
    for rung, weight in ((below, 1.0 - share), (above, share)):
        if rung[1] is not None and weight > 1e-6:
            weights[rung[1]] = weight
    return weights


def _blend_sources(avatar: _LoadedAvatar, tracking: FaceTrackingResult) -> np.ndarray:
    """The source photograph for this frame: the base, plus whatever applies.

    Composed, not averaged. Each photograph is mixed into **its own region only**
    — mouth photographs into the mouth, eye photographs into the eyes — over a
    feathered mask precomputed at load. Photographs that describe the whole head,
    which is to say the turns and the tilts, are still applied across the whole
    image because that is genuinely what they change.

    The averaging version is what made the avatar look like a multiple exposure
    of itself: a smile photograph repainted the forehead and the hairline too,
    so twelve active sources meant twelve photographs contending for every
    pixel. Composing by region means a smile and a glance no longer interfere,
    which is also the property that makes a larger photograph set worth having.

    **Two frames of reference meet here**, and keeping them straight is most of
    what this function does. Every photograph was placed on the base portrait's
    frame at load, so an expression is a picture of a face looking straight
    ahead. A whole-head photograph is not: it brings its own angle, and once it
    is mixed in, the face underneath is no longer where the base's was. So the
    mesh moves with it, and a region still cut from the frontal layout is
    carried onto the moved head before it is composed. Skip that and a mouth
    photograph lands where the frontal mouth *was*, which at 36 degrees is
    70 px beside the one on screen.

    Nothing above happens on a still head, which is most frames: the carry is
    only computed when a whole-head photograph is actually mixed in.
    """
    if not avatar.sources:
        avatar.blend_points = avatar.points
        avatar.interior_role = None
        avatar.eye_follow = _EYE_FOLLOW
        return avatar.image

    weights = {
        role: weight
        for role, weight in _source_weights(tracking, avatar.full_gap, avatar.neutral).items()
        if role in avatar.sources
    }

    if avatar.ladder:
        # The ladder owns yaw entirely. Leaving the named turn roles in as well
        # would mix a chosen angle with an averaged one and undo the choosing.
        for role in _TURN_ROLES:
            weights.pop(role, None)
        pose = tracking.head_pose
        if pose is not None:
            # Read, not learn: `_source_weights` has already advanced the
            # resting angle for this frame.
            weights.update(
                _ladder_weights(avatar.ladder, avatar.neutral.pose_offset("yaw", pose.yaw))
            )

    if not weights:
        avatar.blend_points = avatar.points
        avatar.interior_role = None
        avatar.eye_follow = _EYE_FOLLOW
        return avatar.image

    # **The mouth interior comes from one photograph, never from an average.**
    # The lips may cross-fade — an opening is a continuum and the photographs
    # either side of it agree about where the mouth is. What is *behind* them
    # does not: at the owner's widest smile the mouth was 43% a photograph with
    # teeth in it and 57% three photographs with their lips together, so the
    # teeth arrived at 43% opacity, showing the inside of a closed mouth
    # through them. That is the owner's "no teeth when smiling", and it is the
    # same defect as the four gaze photographs — an average of pictures that
    # disagree is a picture of neither.
    #
    # So the interior is taken whole, from whichever photograph with its lips
    # genuinely parted has the most weight this frame. The blend still decides
    # the lips; this decides only what is seen through them.
    avatar.interior_role = max(
        (role for role in weights if avatar.gaps.get(role, 0.0) >= _LIP_GAP_OPEN),
        key=lambda role: weights[role],
        default=None,
    )

    # Expressions move far more slowly than frames arrive, so the same blend is
    # usually still correct. Rebuilding it every frame was 2.2 ms of pure
    # repetition; quantising the weights and reusing the result removes it
    # without any visible stepping, because a hundredth of a jaw is not visible.
    quantised = {role: round(weight / _WEIGHT_QUANTUM) for role, weight in weights.items()}
    if avatar.cached_blend is not None and quantised == avatar.cached_weights:
        return avatar.cached_blend

    # Whole-head photographs first, across the whole image. Everything after
    # composes on top of the result, so a turned head still gets its mouth.
    #
    # The mesh moves with them. A rung is a photograph of this face at 35
    # degrees, and it is only usable if the mesh also says 35 degrees: the live
    # warp reads this mesh as "where these pixels are", and given the frontal
    # one it would fold a turned face flat before the frame was even composed.
    whole_head = {role: weight for role, weight in weights.items() if role not in _ROLE_REGION}
    mesh = avatar.points.copy()
    if whole_head:
        blended = _weighted_sum(avatar.image, avatar.sources, whole_head)
        shares, share = _shares(whole_head)
        mesh *= 1.0 - share
        mesh += _point_mix(avatar, shares, slice(None)) * share
    else:
        # Reused rather than reallocated. This is a couple of megabytes on every
        # frame that has an expression but no head turn, which is most of them.
        if avatar.scratch is None:
            avatar.scratch = np.empty_like(avatar.image)
        np.copyto(avatar.scratch, avatar.image)
        blended = avatar.scratch

    # Whether anything below has to be carried onto a head that has moved. Only
    # a whole-head photograph moves it, so an expression on a still head takes
    # none of this and costs exactly what it did before.
    #
    # A carry is a resampling and a resampling softens, so it has to earn its
    # place: below a pixel of travel it would trade sharpness on every frame
    # for an accuracy nobody can see. The ladder makes that case common rather
    # than rare — a head is almost never at exactly its resting angle, so the
    # nearest rung is nearly always mixed in at some small weight.
    import cv2

    height, width = avatar.image.shape[:2]
    face_width = max(float(avatar.points[:, 0].max() - avatar.points[:, 0].min()), 1.0)
    travel = float(np.abs(mesh - avatar.points).max()) if whole_head else 0.0
    carrying = travel > _CARRY_MINIMUM_PX
    avatar.eye_follow = _EYE_FOLLOW

    for region in avatar.regions:
        local = {role: weights[role] for role in region.sources if role in weights}
        if not local:
            continue

        carry = None
        shares, share = _shares(local)
        box = region.box
        # The photographs mixed among themselves first, then moved onto the
        # head, and only then mixed with what is underneath. In that order the
        # thing being carried is a picture of a mouth rather than a mouth
        # already averaged with the cheek it is about to leave.
        mixed = _photo_mix(region.sources, shares)
        mask = region.mask
        if carrying:
            fitted = _carry_matrix(avatar.points, mesh, region.positions)
            if fitted is not None:
                # How much of this photograph survives being placed on a head it
                # was not taken on. Past ``_CARRY_LOST`` none of it does, and
                # the region is left as the turned head found it.
                carry, error = fitted
                trust = _trust(error / face_width)
                if region.name == "eyes":
                    # What the photographs were asked for but could not deliver,
                    # which the mesh has to cover instead.
                    lost = share * (1.0 - trust)
                    avatar.eye_follow = _EYE_FOLLOW + lost * (_EYE_FOLLOW_ALONE - _EYE_FOLLOW)
                if trust <= 0.0:
                    continue
                share *= trust
                carried = _carry_region(mixed, mask, carry, box, width, height)
                if carried is None:
                    continue
                mixed, mask, box = carried

        # Feathered so the region's own boundary never shows, and scaled by how
        # much of it this frame asked for. Float over the region's few thousand
        # pixels, not the whole face, and over a contiguous copy: the same
        # arithmetic runs several times faster than it does over a slice of a
        # wider image.
        x0, y0, x1, y1 = box
        underneath = np.ascontiguousarray(blended[y0:y1, x0:x1])
        alpha = mask * share
        blended[y0:y1, x0:x1] = (underneath * (1.0 - alpha) + mixed * alpha).astype(np.uint8)

        # The same mix, applied to the landmarks, so the mesh and the pixels stay
        # descriptions of one face — carried by the same transform as the pixels
        # were, and blended against the head's own landmarks in the same
        # proportion, so nothing jumps as a photograph fades in.
        moved = _point_mix(avatar, shares, region.positions)
        if carry is not None:
            moved = cv2.transform(moved.reshape(-1, 1, 2), carry).reshape(-1, 2)
        mesh[region.positions] = mesh[region.positions] * (1.0 - share) + moved * share

    # The lids have just moved and the irises have not — they belong to no
    # region, because a gaze is geometry rather than a photograph. So the mesh
    # can now claim an iris larger than the eye holding it, which is exactly
    # what a closing lid produces.
    _contain_irises(mesh)

    avatar.cached_weights = quantised
    avatar.cached_blend = blended
    avatar.blend_points = mesh

    result: np.ndarray = blended
    return result


def warp_mesh(
    image: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    triangles: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Warp ``image`` so its mesh lands on ``target_points``, into a new canvas.

    Used twice: at load time to bring every source photograph onto the base
    portrait's landmark layout, and every frame to bring the result onto the
    live face. One routine for both means the two can never disagree about what
    a triangle maps to.

    Written around one measured fact: the cost is dominated by *per-call*
    overhead, not by pixels. An earlier version warped both the image and a mask
    for every triangle and cost 42 ms at p95, over the whole 33 ms budget. So
    there is one warp per triangle, everything happens inside the target's
    bounding box, and no boolean fancy-indexing appears on the hot path.
    """
    import cv2

    # A rigid placement of the whole image first. The mesh paints over it; its
    # only job is that nothing is ever black, because a single-pixel gap between
    # two triangles would otherwise show the empty canvas through the face.
    canvas = _rigid_layer(image, source_points, target_points, width, height)

    for a, b, c in triangles:
        dst = target_points[[a, b, c]]

        x, y, w, h = cv2.boundingRect(dst)
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, width), min(y + h, height)
        if x1 <= x0 or y1 <= y0:
            continue

        src = source_points[[a, b, c]]
        sx, sy, sw, sh = cv2.boundingRect(src)
        if sw <= 0 or sh <= 0:
            continue

        patch = image[sy : sy + sh, sx : sx + sw]
        if patch.size == 0:
            continue

        matrix = cv2.getAffineTransform(
            (src - (sx, sy)).astype(np.float32), (dst - (x, y)).astype(np.float32)
        )

        # BORDER_REPLICATE, not the default. The source patch is the triangle's
        # bounding box, so sampling near the triangle's own edges falls just
        # outside it — and the default border is black. Those black samples land
        # inside the stamped area and show up as a mesh of thin dark cracks over
        # the whole face. Clamping to the edge pixel instead costs nothing.
        piece = cv2.warpAffine(
            patch, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

        cover = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(cover, (dst - (x, y)).astype(np.int32), 255, cv2.LINE_8)

        cv2.copyTo(
            piece[y0 - y : y1 - y, x0 - x : x1 - x],
            cover[y0 - y : y1 - y, x0 - x : x1 - x],
            canvas[y0:y1, x0:x1],
        )

    return canvas


def _rigid_layer(
    image: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """The image placed by a single similarity transform, as a backdrop.

    Rotation, uniform scale, translation. It carries no expression — the mesh
    painted over it does that. It exists so a pixel the mesh happens to miss
    shows a plausible piece of face rather than a black gap.
    """
    import cv2

    # Typed as Any deliberately: OpenCV's stub says this returns a matrix, but
    # it returns None when the fit fails, and a face at an extreme angle can
    # make it fail. Trusting the stub here would be a crash rather than a
    # dropped frame.
    matrix: Any = cv2.estimateAffinePartial2D(source_points, target_points, method=cv2.LMEDS)[0]
    if matrix is None:
        return np.zeros((height, width, 3), dtype=np.uint8)

    base: np.ndarray = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return base


def _fit_lighting_plane(difference: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Least-squares plane through the masked difference, per colour channel.

    ``difference`` arrives already multiplied by the mask and ``coverage`` is
    that mask at the same scale, which makes the arithmetic fall out neatly: the
    weighted right-hand side of the normal equations is a plain sum of
    ``difference`` against each basis image, with no division anywhere.

    Three unknowns per channel — a constant and two gradients — over a 1/8-scale
    image. That is a handful of sums and one 3x3 solve.
    """
    height, width = coverage.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    # Centred and normalised, so the three basis images are well conditioned
    # whatever the box's aspect ratio.
    xs = xs * (2.0 / max(width - 1, 1)) - 1.0
    ys = ys * (2.0 / max(height - 1, 1)) - 1.0
    basis = (np.ones_like(xs), xs, ys)

    normal = np.empty((3, 3), dtype=np.float64)
    for row in range(3):
        for column in range(row, 3):
            value = float((coverage * basis[row] * basis[column]).sum())
            normal[row, column] = normal[column, row] = value

    targets = np.stack([np.einsum("ij,ijc->c", term, difference) for term in basis]).astype(
        np.float64
    )

    try:
        # A face fully inside the box makes this well posed; a sliver of mask at
        # the frame edge may not, and a singular fit is not worth an exception.
        coefficients = np.linalg.solve(normal, targets)
    except np.linalg.LinAlgError:
        total = float(coverage.sum())
        if total < 1e-6:
            return np.zeros((height, width, difference.shape[2]), dtype=np.float32)
        mean = difference.reshape(-1, difference.shape[2]).sum(axis=0) / total
        return np.broadcast_to(mean.astype(np.float32), (height, width, difference.shape[2]))

    field: np.ndarray = np.einsum("kij,kc->ijc", np.stack(basis), coefficients.astype(np.float32))
    return field.astype(np.float32)


def _match_colour(
    loaded: _LoadedAvatar, avatar: np.ndarray, scene: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """Relight the avatar to the scene, as a smooth field rather than one number.

    The portrait was photographed under one light and the room is lit by
    another, so the two differ at the mask edge. A single mean shift — which is
    what this did first — moves the whole face by one constant and leaves the
    edge visible wherever the mismatch is not uniform, which it never is: a
    window on one side makes the difference a *gradient* across the face.

    So the correction is a gradient too. The difference between scene and avatar
    is measured at low resolution, blurred, and added back. Low frequencies
    carry lighting; high frequencies carry identity. Correcting only the low
    ones takes the avatar's illumination from the room while leaving its
    features alone, and because the field is continuous the edge matches by
    construction.

    This is what gradient-domain blending (`cv2.seamlessClone`) does properly,
    and it was measured first: 21-29 ms for a face this size, against a 33 ms
    budget with 15 ms already spent. This costs about 1 ms and removes the same
    step.

    The blur is **weighted by the mask**, so the room behind the head never
    contributes to a correction meant to describe the face.
    """
    import cv2

    height, width = avatar.shape[:2]
    small = (max(width // _CORRECTION_DOWNSCALE, 8), max(height // _CORRECTION_DOWNSCALE, 8))

    weight = alpha[:, :, 0]
    if float(weight.sum()) < 32.0:
        return avatar

    difference = cv2.resize((scene - avatar) * alpha, small, interpolation=cv2.INTER_AREA)
    coverage = cv2.resize(weight, small, interpolation=cv2.INTER_AREA)

    field = _fit_lighting_plane(difference, coverage)

    # Carry the previous field forward. The face box changes size every frame,
    # so the stored field is resized onto the current one before blending —
    # cheap at an eighth scale, and it is a smooth quantity anyway.
    previous = loaded.correction
    if previous is not None:
        resized = cv2.resize(previous, (field.shape[1], field.shape[0]))
        field = resized * _CORRECTION_INERTIA + field * (1.0 - _CORRECTION_INERTIA)
    loaded.correction = field

    correction = cv2.resize(field, (width, height), interpolation=cv2.INTER_LINEAR)

    relit: np.ndarray = np.clip(avatar + correction, 0.0, 255.0)
    return relit


def _trim_forehead(points: np.ndarray) -> np.ndarray:
    """The face outline, pulled short of the hairline and pushed past the jaw.

    **Top.** MediaPipe's oval runs *along* the hairline. Filling to it puts the
    avatar's hairline on top of the user's, and two hairlines never agree — that
    is structure, not tone, and no amount of blending fixes it. Stopping a few
    percent short keeps the whole forehead covered, which is what the owner
    asked for, while leaving the hair itself alone.

    **Bottom, and this one was missed for a week.** The oval runs *along* the
    jawbone, so the avatar stopped there and the user's own jaw continued below
    it. On a bearded face that is not subtle: the owner's render showed the
    avatar's ginger beard ending in a hard curve with their own dark beard
    carrying on underneath it. Two beards, one chin.

    The bottom is the easier half, because what is below a jaw is more jaw and
    then neck — tone rather than structure, which is exactly what the relighting
    can carry. So the lower arc is pushed *down*, past the jawbone, and the
    feather on the mask edge lands on the neck where nothing has to line up.

    Not to the collar: the avatar's own chin is a shape, and pushed far enough
    it would sit over the user's neck as a visible jaw of the wrong length.
    """
    trimmed = points[_OVAL_SLICE].copy()

    brow_top = float(points[_BROW_SLICE][:, 1].min())
    oval_top = float(trimmed[:, 1].min())
    ceiling = brow_top - (brow_top - oval_top) * _FOREHEAD_REACH
    np.maximum(trimmed[:, 1], ceiling, out=trimmed[:, 1])

    # Measured down the face rather than across it: a turned head narrows and
    # the chin does not rise, so a fraction of the width would shrink the reach
    # exactly when the jawline is most exposed.
    face_height = float(trimmed[:, 1].max() - trimmed[:, 1].min())
    chin = float(trimmed[:, 1].max())
    floor = chin - face_height * _JAW_REACH

    below = trimmed[:, 1] > floor
    trimmed[below, 1] += (trimmed[below, 1] - floor) * (_JAW_REACH_SCALE - 1.0)
    return trimmed


def _destination_mask(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Soft-edged mask following the face outline, in the local box.

    The outline, not the convex hull. A face is not convex — the hull bridges
    the gaps beside the chin and in front of the ears, and everything inside
    those bridges gets filled with whatever was behind the head in the
    *portrait*. That is the border the avatar appears to have.

    It is then eroded rather than dilated, so the replaced area stops just
    inside the real face and the blend happens over skin on both sides.

    **The nose and the mouth are added to it, and that is not a refinement —
    without them the user's own face is on screen.** The face oval is a ring of
    anatomical points around the head, and a head is only convex from the
    front. Turned 40 degrees, the nose and the lips stand *outside* that ring in
    the picture: the ring has swung round to the far side of them. So the mask
    stopped short of the profile, and the nose, the lips and the chin in front
    of it were never replaced — the owner, testing: *"when I turn my head, my
    own face appears, half of it."* They were looking at their real nose and
    mouth beside an avatar's cheek.

    Their hull rather than their outline, and only theirs: a blob over the
    middle of the face cannot bridge to an ear or to the background, which is
    what the outline rule exists to prevent. Frontal it is wholly inside the
    oval and changes nothing at all.
    """
    import cv2

    outline = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(outline, [_trim_forehead(points).astype(np.int32)], 255)

    face_width = max(float(points[:, 0].max() - points[:, 0].min()), 1.0)
    shrink = max(1, int(face_width * _MASK_SHRINK))
    feather = max(3, int(face_width * _MASK_FEATHER) | 1)

    trimmed = cv2.erode(outline, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink, shrink)))

    # The features go on **after** the erosion, at full strength. The erosion
    # exists to pull the outer boundary inside the real face so the blend
    # happens over skin on both sides; these are not a boundary, they are the
    # middle of the face, and eroding them would leave a rim of somebody's real
    # eye showing. Only the feather crosses them.
    cv2.fillConvexPoly(trimmed, cv2.convexHull(points[_PROFILE_POSITIONS].astype(np.int32)), 255)

    softened = cv2.GaussianBlur(trimmed, (feather, feather), 0)
    mask: np.ndarray = (softened.astype(np.float32) / 255.0)[:, :, None]
    return mask
