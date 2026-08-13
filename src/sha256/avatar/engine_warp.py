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

# Pitch at which the tilt photographs fully apply.
_TILT_FULL_DEGREES = 18.0

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

# Eye closure is deliberately eager. A blink is fast and reads high, but holding
# the eyes shut reads far lower, and the owner's eyes stayed open for it.
# How far a blink has to rise **above the resting eye** before the closed-eye
# photograph starts to apply, and where it fully applies. Relative, not
# absolute: a relaxed eye does not read zero, and treating its resting value as
# a partial blink kept a shut eyelid permanently ghosted over an open one.
_BLINK_START, _BLINK_FULL = 0.08, 0.50

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
_LEFT_EYE = (33, 160, 158, 133, 153, 144)
_RIGHT_EYE = (362, 385, 387, 263, 373, 380)
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

    __slots__ = ("_floors", "_pose")

    def __init__(self) -> None:
        self._floors: dict[str, float] = {}
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

    def relative_pose(self, name: str, value: float) -> float:
        """``value`` as an offset from the resting head angle. Signed."""
        rest = self._pose.get(name)
        if rest is None:
            self._pose[name] = value
            return 0.0

        rest += _POSE_RATE * (value - rest)
        self._pose[name] = rest
        return value - rest


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

    ``sources`` maps a role to a photograph already warped onto ``points`` — the
    base portrait's landmark layout — so every source shares one geometry and
    blending them is an average rather than a registration problem.
    """

    __slots__ = (
        "blend_points",
        "cached_blend",
        "cached_weights",
        "cavity",
        "correction",
        "descriptor",
        "full_gap",
        "image",
        "mouth_mask",
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
    ) -> None:
        self.full_gap = full_gap
        self.descriptor = descriptor
        self.image = image
        self.points = points
        self.triangles = triangles
        self.sources = sources

        # Where each source's own landmarks ended up once it was placed. For an
        # expression photograph these differ from the base's — that difference
        # *is* the open mouth — and for a pose photograph they are the base's,
        # because a turn is warped onto the base mesh on purpose.
        self.source_points = source_points if source_points is not None else {}

        # The mesh describing the current blend. Recomputed with it, so the two
        # are always a description of the same face.
        self.blend_points = points

        # Which pixels of the photograph are *inside* its mouth, and what colour
        # that cavity is. Both are needed when the user opens wider than the
        # photograph did — see _paint_mouth_interior.
        self.mouth_mask, self.cavity = _mouth_interior(image, points)

        # Where each group of photographs is allowed to act, with their pixels
        # already cropped to it.
        self.regions = _build_regions(image, points, sources)

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
        triangles = _triangulate(points, image.shape[1], image.shape[0])
        sources, source_points, gaps = self._align_variants(descriptor, image, points, triangles)
        full_gap = gaps.get(_OPENING_REFERENCE_ROLE, _LIP_GAP_FULL)

        self._loaded[descriptor.id] = _LoadedAvatar(
            descriptor, image, points, triangles, sources, full_gap, source_points
        )
        _LOG.info(
            "Loaded avatar %s: %dx%d portrait, %d mesh points, %d triangles, "
            "%d extra source(s): %s",
            descriptor.id,
            image.shape[1],
            image.shape[0],
            len(points),
            len(triangles),
            len(sources),
            ", ".join(sorted(sources)) or "none",
        )

    def _align_variants(
        self,
        descriptor: AvatarDescriptor,
        base_image: np.ndarray,
        base_points: np.ndarray,
        triangles: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
        """Bring every extra photograph onto the base portrait's frame.

        **Two different alignments, because two different things are being
        borrowed**, and using one for both is what made the teeth and the
        eyelids fail:

        *   **Expression photographs are placed by a similarity transform** —
            rotate, scale, translate — fitted on the landmarks an expression
            does not move. Their own geometry survives, so an open mouth stays
            open and a closed eyelid stays closed. Their landmarks are returned
            alongside, moved by the same transform, so the mesh that drives them
            later describes the picture that is actually there.

        *   **Pose photographs are warped onto the base's mesh**, as before. For
            a turn, the geometry is exactly what should *not* be borrowed: the
            live face supplies the angle, and the photograph is wanted only for
            the shading and the parts of the cheek a frontal shot cannot show.

        The previous version mesh-warped everything, which forced every source
        into the base's geometry — the base has a closed mouth and open eyes. So
        the open-mouth photograph was crushed shut, leaving a sliver of teeth
        that the live warp then stretched, and the closed-eye photograph had its
        eyelid pulled across an open eye socket, which rendered as vertical
        streaks. Both were plainly visible and neither was subtle.

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

            if role in _ROLE_REGION:
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

        return aligned, aligned_points, gaps

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
    pad = 8
    fx0 = max(int(target[:, 0].min()) - pad, 0)
    fy0 = max(int(target[:, 1].min()) - pad, 0)
    fx1 = min(int(target[:, 0].max()) + pad, width)
    fy1 = min(int(target[:, 1].max()) + pad, height)
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
        _damp_brows(source_mesh, local),
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


def _mouth_interior(image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The photograph's own mouth opening: which pixels, and their mean colour.

    The colour stands in for cavity depth when the user opens wider than the
    photograph did, which is most of the time — no still photograph is taken
    mid-yawn.
    """
    import cv2

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points[_INNER_LIP_SLICE].astype(np.int32)], 255)

    inside = mask > 0
    if not bool(inside.any()):
        return mask, np.zeros(3, dtype=np.float32)

    # The darker half of the opening: teeth are the bright part and would make
    # a poor stand-in for the back of a throat.
    pixels = image[inside].astype(np.float32)
    luminance = pixels.mean(axis=1)
    darker = pixels[luminance <= float(np.median(luminance))]
    cavity: np.ndarray = darker.mean(axis=0) if len(darker) else pixels.mean(axis=0)
    return mask, cavity


def _paint_mouth_interior(
    avatar: _LoadedAvatar, canvas: np.ndarray, image: np.ndarray, target: np.ndarray
) -> None:
    """Fill the mouth opening from the photograph, in one piece.

    The mesh no longer covers this area, because triangulating across an opening
    slices whatever is behind it into strips. The mouth region is placed instead
    by a single similarity transform fitted to the lip landmarks, so the teeth
    arrive at the right size and stay one object.

    **Only the photograph's own mouth interior is copied.** Placing the whole
    image and trusting the fit put the photograph's *lips* inside the live
    opening whenever the user opened wider than the photograph did — a second,
    smaller mouth inside the first, which is what the owner saw. Anything the
    photograph cannot cover is filled with the cavity colour, so opening wider
    reveals more darkness. That is also what it does in life: the teeth stay the
    size they are.
    """
    import cv2

    # From the blended mesh, for the same reason the warp uses it: when an
    # open-mouth photograph is mixed in, the inner lips of the picture are open,
    # and fitting from the base's closed ones would place the interior wrongly.
    matrix: Any = cv2.estimateAffinePartial2D(
        avatar.blend_points[_INNER_LIP_SLICE], target[_INNER_LIP_SLICE], method=cv2.LMEDS
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
        image,
        shifted,
        (box_w, box_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    covered = cv2.warpAffine(
        avatar.mouth_mask,
        shifted,
        (box_w, box_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    cv2.copyTo(placed, cv2.bitwise_and(opening, covered), region)


def _damp_brows(source: np.ndarray, target: np.ndarray) -> np.ndarray:
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
    in the blended mesh. So the eyes follow the user barely at all.

    The reference for both is where a rigid fit of the whole face would put them
    — the avatar's own proportions, positioned on the live head.
    """
    import cv2

    matrix: Any = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)[0]
    if matrix is None:
        return target

    rigid = cv2.transform(source.reshape(-1, 1, 2), matrix).reshape(-1, 2)

    damped = target.copy()
    for where, follow in ((_BROW_SLICE, _BROW_FOLLOW), (_EYE_POSITIONS, _EYE_FOLLOW)):
        anchor = rigid[where]
        damped[where] = anchor + (damped[where] - anchor) * follow
    return damped


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
        blink = min(shapes.get("eyeBlinkLeft"), shapes.get("eyeBlinkRight"))
        above_rest = neutral.relative_expression("eyeBlink", blink)
        closed = (above_rest - _BLINK_START) / (_BLINK_FULL - _BLINK_START)
        weights["source_eyes_closed"] = min(max(closed, 0.0), 1.0)

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
        weights["source_eyes_wide"], weights["source_squint"] = opposed(
            expression("eyeWideLeft", "eyeWideRight"),
            expression("eyeSquintLeft", "eyeSquintRight"),
        )
        weights["source_frown"] = expression("mouthFrownLeft", "mouthFrownRight")
        weights["source_mouth_pressed"] = expression("mouthPressLeft", "mouthPressRight")

        # A closed-mouth smile is a smile with the lips still together, which is
        # exactly what the gap measurement distinguishes.
        weights["source_smile_closed"] = smiling * (1.0 - opening)

        # Rounded and funnelled lips are shapes rather than openings, so they
        # are driven by their own blendshapes and not by the gap.
        weights["source_mouth_o"] = expression("mouthPucker", "mouthFunnel")
        weights["source_mouth_wide"] = expression("mouthStretchLeft", "mouthStretchRight")

        # Gaze. "In" and "out" are relative to each eye, so a look to the
        # viewer's left is the left eye looking out and the right eye looking in.
        weights["source_look_left"], weights["source_look_right"] = opposed(
            expression("eyeLookOutLeft", "eyeLookInRight"),
            expression("eyeLookOutRight", "eyeLookInLeft"),
        )
        weights["source_look_up"], weights["source_look_down"] = opposed(
            expression("eyeLookUpLeft", "eyeLookUpRight"),
            expression("eyeLookDownLeft", "eyeLookDownRight"),
        )

    pose = tracking.head_pose
    if pose is not None:
        # Relative to how this person sits, not to the camera's idea of straight
        # ahead. A camera above a monitor reads a permanent downward tilt, and
        # driving the tilt photographs from that put a permanent tilt in the
        # avatar's face while the user was looking straight at them.
        yaw = neutral.relative_pose("yaw", pose.yaw)
        pitch = neutral.relative_pose("pitch", pose.pitch)

        turn = yaw / _TURN_FULL_DEGREES
        weights["source_turn_left"] = turn
        weights["source_turn_right"] = -turn

        far = yaw / _TURN_FAR_DEGREES
        weights["source_turn_left_far"] = far
        weights["source_turn_right_far"] = -far

        tilt = pitch / _TILT_FULL_DEGREES
        weights["source_tilt_up"] = tilt
        weights["source_tilt_down"] = -tilt

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
    gazing = {role: chosen[role] for role in _GAZE_ROLES if role in chosen}
    if gazing:
        strongest = max(gazing, key=lambda role: gazing[role])
        for role in gazing:
            if role != strongest:
                del chosen[role]

        # And all-or-nothing, because a half-mixed gaze renders two pupils.
        ramped = (chosen[strongest] - _GAZE_ON) / (_GAZE_FULL - _GAZE_ON)
        ramped = min(max(ramped, 0.0), 1.0) * _MAX_VARIANT_WEIGHT
        if ramped <= 0.02:
            del chosen[strongest]
        else:
            chosen[strongest] = ramped

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
    image: np.ndarray, points: np.ndarray, sources: dict[str, np.ndarray]
) -> tuple[_Region, ...]:
    """Precompute each region's box, feathered mask, and cropped photographs.

    Regions whose photographs the avatar does not have are not built at all, so
    an avatar with only a base portrait costs nothing here.
    """
    import cv2

    height, width = image.shape[:2]
    face_width = max(float(points[:, 0].max() - points[:, 0].min()), 1.0)
    feather = max(3, int(face_width * _REGION_FEATHER) | 1)

    built: list[_Region] = []
    for name, positions in (
        ("mouth", _MOUTH_POSITIONS),
        ("eyes", _EYE_POSITIONS),
        ("brows", _BROW_POSITIONS),
    ):
        present = [role for role in _REGION_ROLES[name] if role in sources]
        if not present:
            continue

        margin = int(face_width * _REGION_MARGIN[name])
        group = points[positions]
        x0 = max(int(group[:, 0].min()) - margin, 0)
        y0 = max(int(group[:, 1].min()) - margin, 0)
        x1 = min(int(group[:, 0].max()) + margin, width)
        y1 = min(int(group[:, 1].max()) + margin, height)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue

        # The mask is drawn in the box's own coordinates, as the convex hull of
        # the region's landmarks grown by the margin, then blurred. Blurring
        # inside the box means the edge fades to zero at the box border, so the
        # composite can never produce a seam at the rectangle itself.
        local = (group[:, :2] - (x0, y0)).astype(np.int32)
        filled = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(filled, cv2.convexHull(local), 255)
        grown = cv2.dilate(
            filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin | 1, margin | 1))
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

    The variants were aligned to the base portrait's geometry at load, so within
    a region this is still a straight weighted sum of pixels — no registration,
    and no per-frame decision about which pixels are "the mouth".
    """
    if not avatar.sources:
        avatar.blend_points = avatar.points
        return avatar.image

    weights = {
        role: weight
        for role, weight in _source_weights(tracking, avatar.full_gap, avatar.neutral).items()
        if role in avatar.sources
    }
    if not weights:
        avatar.blend_points = avatar.points
        return avatar.image

    # Expressions move far more slowly than frames arrive, so the same blend is
    # usually still correct. Rebuilding it every frame was 2.2 ms of pure
    # repetition; quantising the weights and reusing the result removes it
    # without any visible stepping, because a hundredth of a jaw is not visible.
    quantised = {role: round(weight / _WEIGHT_QUANTUM) for role, weight in weights.items()}
    if avatar.cached_blend is not None and quantised == avatar.cached_weights:
        return avatar.cached_blend

    # Whole-head photographs first, across the whole image. Everything after
    # composes on top of the result, so a turned head still gets its mouth.
    whole_head = {role: weight for role, weight in weights.items() if role not in _ROLE_REGION}
    if whole_head:
        blended = _weighted_sum(avatar.image, avatar.sources, whole_head)
    else:
        # Reused rather than reallocated. This is a couple of megabytes on every
        # frame that has an expression but no head turn, which is most of them.
        if avatar.scratch is None:
            avatar.scratch = np.empty_like(avatar.image)
        np.copyto(avatar.scratch, avatar.image)
        blended = avatar.scratch

    # The mesh that describes the picture being built. It starts as the base
    # portrait's and moves with the photographs mixed in — an open-mouth
    # photograph is only useful if the mesh also says the mouth is open, and one
    # still describing closed lips would crush it shut on the very next step.
    mesh = avatar.points.copy()

    for region in avatar.regions:
        local = {role: weights[role] for role in region.sources if role in weights}
        if not local:
            continue

        x0, y0, x1, y1 = region.box
        underneath = np.ascontiguousarray(blended[y0:y1, x0:x1])
        mixed = _weighted_sum(underneath, region.sources, local)

        # Feathered so the region's own boundary never shows. Float only over
        # the region's few thousand pixels, not the whole face.
        composed = mixed * region.mask + underneath * (1.0 - region.mask)
        blended[y0:y1, x0:x1] = composed.astype(np.uint8)

        # The same weights, applied to the landmarks, so the mesh and the pixels
        # stay descriptions of one face.
        share = sum(local.values())
        scale = min(1.0 / share, 1.0) if share > 1.0 else 1.0
        moved = avatar.points[region.positions] * (1.0 - min(share, 1.0))
        for role, weight in local.items():
            # A source with no recorded landmarks contributes no geometry, which
            # is exactly what the base's own landmarks express.
            own = avatar.source_points.get(role, avatar.points)
            moved = moved + own[region.positions] * (weight * scale)
        mesh[region.positions] = moved

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
    """The face outline with its top arc pulled just short of the hairline.

    MediaPipe's oval runs *along* the hairline. Filling to it puts the avatar's
    hairline on top of the user's, and two hairlines never agree — that is
    structure, not tone, and no amount of blending fixes it.

    Stopping a few percent short keeps the whole forehead covered, which is what
    the owner asked for, while leaving the hair itself alone.
    """
    trimmed = points[_OVAL_SLICE].copy()

    brow_top = float(points[_BROW_SLICE][:, 1].min())
    oval_top = float(trimmed[:, 1].min())
    ceiling = brow_top - (brow_top - oval_top) * _FOREHEAD_REACH

    np.maximum(trimmed[:, 1], ceiling, out=trimmed[:, 1])
    return trimmed


def _destination_mask(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Soft-edged mask following the face outline, in the local box.

    The outline, not the convex hull. A face is not convex — the hull bridges
    the gaps beside the chin and in front of the ears, and everything inside
    those bridges gets filled with whatever was behind the head in the
    *portrait*. That is the border the avatar appears to have.

    It is then eroded rather than dilated, so the replaced area stops just
    inside the real face and the blend happens over skin on both sides.
    """
    import cv2

    outline = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(outline, [_trim_forehead(points).astype(np.int32)], 255)

    face_width = max(float(points[:, 0].max() - points[:, 0].min()), 1.0)
    shrink = max(1, int(face_width * _MASK_SHRINK))
    feather = max(3, int(face_width * _MASK_FEATHER) | 1)

    trimmed = cv2.erode(outline, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink, shrink)))
    softened = cv2.GaussianBlur(trimmed, (feather, feather), 0)
    mask: np.ndarray = (softened.astype(np.float32) / 255.0)[:, :, None]
    return mask
