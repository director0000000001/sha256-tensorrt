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

# The same idea on the smile axis, and the owner's own proposal after watching a
# slight smile snap straight to a photograph of a wide one: *"we already have
# the clip where he laughs — ten photographs from where the smile starts through
# to the full smile."*
#
# They are right, and it is the yaw ladder's argument word for word. One
# photograph of a smile fading in over a resting mouth is two mouths at once,
# faint, and it arrives all at the same shape: the lips never stretch, the teeth
# simply appear. A row of photographs *of the smile forming* is a smile forming.
_SMILE_PREFIX = "smile@"

# Below this much smile the ladder stays out of it, so an ordinary talking mouth
# is chosen by the shape it is making rather than by how faintly it resembles a
# smile. Measured against rest, like every other expression.
_SMILE_LADDER_START = 0.08

# And the fourth axis, which the owner ranks first of all: the lips parting.
#
# *"When I am talking it almost does not follow me at all."* Day 13 measured
# that the mouth does follow — parting gain 1.04, r 0.99 on a re-tracked render
# — and that on 65.6% of frames the photograph it followed with was
# `source_mouth_o` at a median share of 0.18 over the portrait's own *closed*
# mouth. Two thirds of a conversation rendered as a closed mouth stretched open
# by the mesh, which is what "it does not follow me" looks like from outside.
#
# **Its own axis rather than a second dimension on the smile**, and that is a
# measurement. Over the frames of `driving.mkv` where the owner is speaking, the
# mouth's widening and its parting move together at r = -0.049, and one
# direction carries 98.4% of the demand: (-0.006 widened, +1.000 parted). A
# speaking mouth parts and barely widens. See `scripts/measure_speech_plane.py`.
_SPEECH_PREFIX = "speech@"

# Below this the lips are together and the portrait is already the best
# photograph of that — it is the rung at zero, as it is on every other axis.
# Low, because it has to be: an ordinary consonant closes the lips completely
# several times a second, and a threshold high enough to be safe would take the
# mouth region out and put it back at syllable rate.
_SPEECH_LADDER_START = 0.05

# And the same foot on the width axis, which the grid gained on 2026-08-26 when
# the smile ladder was folded into it. One width step above the portrait's
# resting mouth, matching `build_speech_ladder.DEFAULT_MIN_WIDENED`: below it a
# mouth is merely resting and the portrait is its own best photograph. The
# owner's ordinary speaking widening is 0.046, which is deliberately under it —
# speech is chosen by the parting, and the width only decides *which* opening.
_SPEECH_WIDTH_START = 0.08

# **There is no hold on this axis, and that is a 2026-08-28 removal.**
#
# There were two: `_SPEECH_HOLD`, four times the other ladders' margin, added on
# 2026-08-26 because the first version of this ladder changed photograph 7.6
# times a second; and `_SPEECH_RELEASE`, added on 2026-08-27 to stop that margin
# holding a photograph the mouth had left behind. Both are gone, and the owner
# is why: *"it does not follow the mouth often when talking ... before it
# followed more often."*
#
# Measured over his five captures, choosing the nearest rung against holding one:
#
# | | changes a second | the step when it changes | parting error, p50 |
# |---|---|---|---|
# | `201806` | 7.5 → **12.0** | 0.228 → **0.141** | 0.061 → **0.034** |
# | `201814` | 0.6 → **1.8** | 0.164 → **0.106** | 0.046 → **0.029** |
# | `214327` | 3.2 → **5.7** | 0.179 → **0.116** | 0.024 → **0.022** |
# | `231503` | 2.8 → **4.8** | 0.213 → **0.147** | 0.026 → **0.013** |
#
# **The hold was making the cuts big.** Holding a rung until it is 0.15 away
# means the mouth has drifted 0.15 before it moves at all, and the rung it then
# lands on is two or three past the one next door. Every number moves the right
# way at once: it follows more often, each change is smaller, and the photograph
# is nearer the mouth being made.
#
# The flicker the margin was added for is real and is now paid in single-frame
# runs — 39 of 81 becoming 91 of 126 on `201806`. That is the trade the owner
# asked for in his own words, and it is the one his eye keeps choosing: he has
# complained that the mouth is passive three times and that it flickers once.
#
# **What actually bounds the step is the grid, not the choice.** With no hold at
# all the median step is still 0.106 to 0.147 where the rungs are 0.058 apart,
# because along the line his mouth actually travels the neighbours are two or
# three spacings away — half the grid sits at widenings he never reaches. That
# is material, not code: see the open list.

# And the third axis, which the owner ranks first: the lids closing.
#
# The eyes have been the standing complaint since day 8 — *"wrong far more often
# than they are right"* — and every fix so far has been a subtraction: the gaze
# photographs dropped, the squint dropped, two blendshapes found to measure
# something other than their names. What was left is one shut-eye photograph
# fading in over an open one, which is the same defect the smile had: while it
# is faint there are two sets of eyes, and at any weight the lid it draws is
# fully shut rather than part way.
_EYES_PREFIX = "eyes@"

# And the brows, which are the last family still driven by two named
# photographs. The owner has reported both halves of what that costs: a slight
# raise reading as surprise, because the photograph shows a full one at low
# opacity, and two brows with two more faintly above them, because the
# photograph's brows are somewhere the portrait's are not.
_BROWS_PREFIX = "brows@"
_BROWS_LADDER_START = 0.12

# How far the brows move as a fraction of their resting height above the eye
# line, for the axis to reach one.
#
# **0.62, and it is a new measurement rather than a converted one** — see
# :func:`_brow_height`, which stopped measuring from the nose bridge on
# 2026-08-24. The old 0.18 described a different numerator and does not carry
# over. Measured twice, on the two faces this axis has to agree about, and they
# agree: the furthest raised photograph in the library sits at **+0.620** of the
# portrait's resting height, and the owner's own furthest raise over the 1847
# frames of `driving.mkv` is **+0.599**. A frown reaches -0.262 in the library
# and -0.155 on the recording.
#
# So the axis reaches one where a full raise is, and the tail past it clamps:
# p95 of ordinary conversation is +0.224, which is 0.36 of the scale. That is
# the point of taking the span from a full raise rather than from a percentile —
# the alternative puts an everyday raise at 1.0, and an everyday raise reading
# as a full one is the owner's *"crazily overjoyed"* complaint exactly.
_BROW_SPAN = 0.62

# How much of a new brow reading is taken each frame. 0.4 is a time constant of
# about two and a half frames.
_BROW_SMOOTHING = 0.4

# How far the lids have to have moved before a brow reading is not believed.
# Past this the landmarks are riding a blink rather than reporting a brow.
_BROW_FREEZE_LID = 0.30
# How far the lids have to have moved before a rung is chosen at all. Below it
# the portrait's own eye is the best photograph of a resting one.
#
# It was 0.12 while the ladder's shallowest genuine rung was a fully shut eye:
# every rung from +0.122 up measured 36-44% of the resting opening, so choosing
# one early meant shutting an eye that was a fifth closed, and 0.12 was the
# guard against that. The 2026-08-21 ladder has rungs at +0.065 and +0.106 that
# really are a fraction closed — 99% and 97% of the resting opening — so the
# guard now costs what it used to save, and a quarter-closed eye has a
# photograph of a quarter-closed eye to be drawn with.
_EYES_LADDER_START = 0.05

# How much nearer another rung has to be before a ladder changes photograph.
#
# A rung is shown whole, so every change is a change of picture rather than a
# change of opacity — and every one of these measurements moves a little frame
# to frame on a face that is doing nothing. Without a hold the choice walks
# back and forth across a boundary several times a second.
#
# The hold used is half of *that ladder's own* median spacing, computed where it
# is applied: the brow ladder is fifteen rungs over two units and the eye ladder
# eleven over one, and one fixed number cannot be right for both. This is only
# the fallback, for a ladder too short to have a spacing.
_RUNG_HOLD = 0.05

# **No hold at the foot of the ladder**, and it was measured rather than
# assumed. Now that the lids arrive whole, a reading sitting on the start
# threshold turns the whole region on and off rather than fading it, so a hold
# there looked obviously right. It is not: keeping the last rung until the
# reading has fallen three fifths of the way back through the threshold more
# than doubles how long the lids stay down — the median run of a closed eye goes
# from 18 frames to 41 over the owner's recording, half a second of held blink —
# and it buys eight fewer changes a minute. The shortest run without it is three
# frames, which is a blink and not a flicker, so there was nothing to buy.

# How wide the change from one rung to the next may be, in degrees of yaw.
#
# See :func:`_ladder_weights`. A ladder cut from a real clip has gaps wherever
# the head was moving rather than stopping, and averaging two photographs eight
# degrees apart is a smeared face rather than a face at the angle between them.
# Three degrees, which is the spacing the extractor aims for, so a ladder at its
# intended density behaves exactly as it did before this existed.
_LADDER_CROSSFADE_DEGREES = 3.0

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

# How many frames the resting angle is a plain running mean of before it slows
# down to :data:`_POSE_RATE`.
#
# An estimate built from one frame should move freely and one built from a
# thousand should not, and until this was written it moved at the same crawl in
# both cases. The seed is the first frame, which is right when someone is
# looking at their screen and wrong when they are not: driven by a clip that
# opens mid-turn at 31.7°, the engine took that for straight ahead and then
# chose a 30°-turned photograph for a frontal face — a whole cheek stretched
# sideways, and half a minute before it recovered.
#
# A hundred and twenty frames is four or five seconds, and the handover is
# smooth: the warm-up's own rate has fallen to 0.008 by the end of it, four
# times the slow rate rather than a step down to it. Measured on the case this
# exists for — a session opening at 31.7 degrees — the error is 2.2 degrees by
# the end of the warm-up, against 31.4 with the slow mean alone. The very first
# frame still returns exactly zero, so an avatar still starts as its own
# photograph.
_POSE_WARMUP_FRAMES = 120

# The same argument for the eye's resting opening, and the number is its own
# because the thing it is waiting for is different.
#
# The owner: *"I switched it on with my eyes open and it was showing closed
# eyes."* Reproduced from the recording by starting the engine at six different
# moments, every one of them with the eyes open: a shut rung on **44 of the
# first 60 frames** from a standing start, 41 from another, 52 of a wide rung
# from a third.
#
# The cause is not the eye. Measured on the owner's own clip, inside a single
# "sit still and relax" instruction:
#
#   ================  ==================
#   frames            eyelid opening
#   ================  ==================
#   0-40                          0.0418
#   41-116                        0.0652
#   1757-1846                     0.0593
#   ================  ==================
#
# **A 36% climb over the first second and a half, while the person was sitting
# still** — that is the tracker converging, not a face opening its eyes. The
# resting level was seeded from frame 0 and then moved at a seventeen-second
# time constant, so it kept the first frame's answer while the readings walked
# away from it, and a quarter of a closing is enough to put a fully shut rung on
# the screen.
#
# Forty-five frames is a second and a half, which is where the climb ends.
_LEVEL_WARMUP_FRAMES = 45

# And how far back the warm-up's own mean looks, in frames.
#
# A running mean over *everything seen so far* is the right estimator when the
# thing being estimated is steady, and the wrong one here: the reading climbs
# through the whole warm-up, so an average of it is an average of a journey and
# ends the warm-up sitting well below where the face actually is — which reads
# as an eye held **wide**, the same defect mirrored. Holding the rate at a
# short window instead keeps the estimate on the reading rather than behind it.
_LEVEL_WINDOW_FRAMES = 12

# The seed counts for this many frames, so that a running mean over very few
# readings is not thrown by one of them. Without it the second frame of a
# session would move the resting angle halfway to wherever that frame happened
# to be looking, which is the same defect as the slow crawl, mirrored: a turn
# in the first second would be learned as straight ahead.
#
# The cost is honest and worth naming: a turn *held* through the opening
# seconds does still drag the estimate. Recovering from a session that opens
# turned and being deaf to a turn in the opening seconds are the same knob, and
# this is where it is set.
_POSE_PRIOR_FRAMES = 8

# How far from the resting angle a reading may be before it stops counting as
# evidence about where rest *is*, in degrees.
#
# After the warm-up the question changes. A person glancing at a second monitor
# holds a large angle for seconds at a time, and a plain mean reads that as them
# having moved their chair — the drift the owner would notice as the avatar
# slowly deciding that turned is straight ahead. So a reading's influence falls
# off with its distance: at twelve degrees it counts for half, at twenty-four
# for a fifth. Soft rather than a threshold, because someone who genuinely
# re-seats themselves at an angle must still be learned, only slowly.
_POSE_TRUST_DEGREES = 12.0

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
#
# **No longer what the smile axis is built on** — see :func:`_smiling`. Kept
# because `_source_weights` still weighs the two named smile photographs by the
# blendshape for an avatar that has no ladder.
_SMILE_REFERENCE = 0.54

# The mouth corners, and the outer eye corners the mouth is measured against.
#
# **Against the eye span rather than the face**, and in pixels rather than in
# normalised coordinates, because both are what a crop cannot move. Measured on
# 2026-08-21, cutting a rung out of its own frame and re-reading it: the mouth
# width moves 0.008 of an eye span and the lip gap 0.002, against axis spans of
# 0.19 and 0.15 — four percent and one. The same test on the brow axis moves it
# by seven percent at best and by thirty-eight at worst, which is why the brows
# still cannot be filed this way and the mouth now can.
_MOUTH_CORNERS = (61, 291)
_OUTER_EYE_CORNERS = (33, 263)

# How far the mouth widens, as a fraction of the eye span, for the widening half
# of the smile axis to reach one.
#
# Measured on the owner's own rungs: his resting mouth is 0.533 of an eye span
# across and his closed smiles are 0.635, so a closed smile widens it by 0.102.
# The axis has said since it was written that a closed smile is 0.2, so the full
# span is 0.51.
_SMILE_WIDEN_SPAN = 0.51

# And how far it widens at the broadest smile in the library, which is what
# "fully smiling" means for the two named photographs. 0.19 of an eye span,
# measured: the owner rests at 0.533 and his widest rung is 0.722.
#
# A different number from the one above and deliberately so. The *axis* is the
# ladder's coordinate and says a closed smile is 0.2; the *share* is how far
# into a smile the face is, from nothing to as far as this person goes. Using
# the axis for both capped the closed-smile photograph at 0.37 of itself.
_SMILE_WIDEN_FULL = 0.19

# And how far the lips part for the opening half to reach one. Measured on the
# same rungs: the ones where the teeth appear sat at 0.125 of an eye span and
# the widest at 0.148, and the axis says the teeth appearing is 1.0 — which with
# the widening already contributing a third gave 0.19.
#
# **Divided by 3.70 on 2026-08-26, when :func:`_lip_parting` stopped dividing by
# the eye span.** That is the map between the two denominators on the driver's
# own frames — 400 of them, median 3.70 — and it is applied to this constant and
# to `_SMILE_LIP_CLOSED` together so that the live axis does not move at all:
# the same face on the same camera reads exactly what it read yesterday. What
# changes is source material shot in a shape other than the camera's, which
# until now could not be filed correctly at all.
#
# **Two numbers already written in this file check the map, and both land within
# 3%.** A photograph is square, so the map for one is 3.70 / (16/9) = 2.08:
#
#   the teeth-showing rung      0.125 / 2.08 = 0.0601  vs  0.0591 recorded
#   the portrait's resting mouth 0.0019 / 2.08 = 0.00091 vs 0.0010 recorded
#
# The recorded pair is `_LIP_GAP_CLOSED`'s own calibration, measured on the same
# photographs against face height days before this rescale was needed.
# Two independent measurements of the same two mouths, agreeing.
_SMILE_OPEN_SPAN = 0.0514

# Below this the lips count as together and the opening contributes nothing.
#
# **Without it a face doing nothing asks for a smile**, which is how it was
# found: no mouth is ever perfectly sealed, and a tenth of a millimetre of
# parting divided by a small number is not a tenth of a millimetre any more.
# The portrait's own resting mouth measured 0.0019 on the old eye-span scale and
# its closed smiles 0.004 to 0.023, so a floor at 0.010 left a closed smile its
# widening and nothing else. `_LIP_GAP_CLOSED` is the same idea on the same
# denominator this now uses, and dropping it here was an oversight rather than a
# decision.
#
# Divided by the same 3.70 as `_SMILE_OPEN_SPAN`, and for the same reason: both
# are compared against :func:`_lip_parting`, so rescaling them together leaves
# every live reading exactly where it was.
_SMILE_LIP_CLOSED = 0.0027

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

# The iris boundary, without the centre. MediaPipe gives five points per iris
# and the first is the middle of it — see :data:`_LEFT_IRIS`.
_LEFT_IRIS_RING, _RIGHT_IRIS_RING = _LEFT_IRIS[1:], _RIGHT_IRIS[1:]
_IRIS_RING_POSITIONS = (_positions(_LEFT_IRIS_RING), _positions(_RIGHT_IRIS_RING))

# **Four more points per iris, on the circle, between the four the tracker
# gives.** Added 2026-08-24, and the reason is geometry rather than taste.
#
# The owner: *"when I look slightly to the side with my eyes the iris changes
# shape and stops being a circle — it gets partly pointed"*, and *"sometimes a
# greenish-bluish colour becomes part of the iris"*. One mechanism gives both.
#
# A tracker's four boundary points sit **on** the iris circle, so the polygon
# through them is a diamond *inscribed* in it, and an inscribed square covers
# only **2/π — 63.7% — of the circle's area**. The remaining **36%** is four
# circular segments that belong to no iris triangle at all: they are carried by
# the fan running out to the eyelid. :func:`_place_irises` then moves the
# *target's* iris while the source's stays where the portrait's is, so that fan
# is sheared by the gaze — compressed on one side, stretched on the other. The
# compressed side folds iris pixels into a corner; the stretched side pulls
# sclera in under the iris outline. A point, and a colour that is not iris.
#
# Eight points make it a regular octagon, which covers **90.0%** — the segments
# outside the mesh drop from 36% to 10%, a **3.6-fold** reduction in how much of
# the iris the shearing fan is responsible for.
#
# They are computed rather than tracked: the iris is a circle whose centre and
# radius are already known, so the bisector of each pair of neighbours is a
# point on it. Computed inside :func:`_pixel_points`, which is the one place a
# mesh is built, so the portrait and the live face get them by the same rule —
# which is the only property a warp actually requires of them.
_IRIS_ARCS_PER_EYE = 4
_LEFT_IRIS_ARCS = tuple(range(len(MESH_INDICES), len(MESH_INDICES) + _IRIS_ARCS_PER_EYE))
_RIGHT_IRIS_ARCS = tuple(
    range(len(MESH_INDICES) + _IRIS_ARCS_PER_EYE, len(MESH_INDICES) + 2 * _IRIS_ARCS_PER_EYE)
)

MESH_POINT_COUNT = len(MESH_INDICES) + 2 * _IRIS_ARCS_PER_EYE
"""Rows in a mesh: the tracked landmarks, then the computed iris arcs."""

# Per eye, because gaze is placed one eye at a time: each iris is positioned
# inside its own socket, so a face at an angle still gets both right.
_EYE_PAIRS = (
    (_positions(_LEFT_EYE), [*_positions(_LEFT_IRIS), *_LEFT_IRIS_ARCS]),
    (_positions(_RIGHT_EYE), [*_positions(_RIGHT_IRIS), *_RIGHT_IRIS_ARCS]),
)
_IRIS_POSITIONS = [
    *_positions(_LEFT_IRIS, _RIGHT_IRIS),
    *_LEFT_IRIS_ARCS,
    *_RIGHT_IRIS_ARCS,
]

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
# The triangles between the iris and the lid shear as the pupil moves, so a
# small shift is clean and an extreme one turns the iris into a dark diamond
# with visible triangle edges. Vertically it is worse and asymmetric, because
# the lower lid is closer than the upper one.
#
# **The claim that normal gaze sits comfortably inside these limits is false on
# the owner's own recording, and it was measured on 2026-08-24.** His sideways
# gaze over 1847 frames has a **median of 0.1172** of eye width, and
# ``_GAZE_LIMIT_X / _GAZE_SCALE`` puts the clamp at **0.1182** — so the iris is
# at or past the limit on about **half of an ordinary call**, not at the edge of
# what a person does. p95 is 0.2196 and the furthest is 0.2633.
#
# The limits stay where they are for now anyway, because the defect they were
# guarding against has just been attacked from the other side — see
# :data:`_IRIS_ARCS_PER_EYE`. Whether a finer iris earns a wider limit is a
# question for a measurement, not for this comment.
#
# Nothing is lost by clamping: gaze reads as a *direction* long before it reads
# as a magnitude, and a video call is not looked at from the corner of the eye.
_GAZE_SCALE = 0.55
_GAZE_LIMIT_X = 0.065
_GAZE_LIMIT_UP, _GAZE_LIMIT_DOWN = 0.035, 0.012

# How much of the space between the pupil and the nearer lid the iris ring may
# take up. See :func:`_contain_iris` for why it must take up less than all of
# it: at exactly all of it the triangles joining iris to lid have no area, and
# an affine transform fitted to three collinear points is singular.
_IRIS_OF_OPENING = 0.85

# How far in a shut eye's iris ring is tucked, as a fraction of its distance
# from the lid ring's centroid. See :func:`_tuck_iris`.
#
# Small enough that the ring is well clear of the lash line whatever the lid is
# doing — that line is the darkest thing in the picture and the whole defect was
# it being dragged flat — and large enough that the fan's own triangles keep
# some area, because the version that took the ring to nothing put a bright
# speck where the dark one had been.
_IRIS_TUCKED = 0.35

# How flat the lid ring has to get before the iris is tucked away, as a
# fraction of **this portrait's own** resting lid — see :func:`_shut`.
#
# Containment shrinks the iris to fit the lid; it never puts it anywhere else.
# Past a certain closure there is nothing left for an iris to be, and what the
# fan draws instead is the lash line squeezed flat: the owner, *"the pupils,
# very thin, low in height and long in width — a flat oval, for a very short
# time."* Rendered with the fan taken out of the mesh, all of that goes.
#
# Measured rather than chosen. The lid ring's height over its own width, on this
# avatar's own library and on 1847 frames of the recording:
#
#   ===================  ===============
#   the eye              height / width
#   ===================  ===============
#   base portrait                  0.323
#   the four wide rungs     0.403-0.448
#   every frame open        0.298-0.450
#   every shut rung         0.106-0.128
#   every frame shut        0.106-0.123
#   ===================  ===============
#
# There is nothing between 0.128 and 0.298, on either side of the warp, so the
# line goes in the middle of the empty band — and it is written against the
# portrait's own lid rather than as a number, because a fraction of *this* face
# is the only kind of threshold this file has been able to trust.
_IRIS_HIDDEN_BELOW = 0.6
_BROW_POSITIONS = _positions(_LEFT_BROW, _RIGHT_BROW)
_MOUTH_POSITIONS = _positions(_OUTER_LIPS, _INNER_LIPS)

# Landmarks that an *expression* does not move: the face outline and the nose
# bridge. Used to place an expression photograph on the base's frame without
# letting the expression itself drag the fit.
_STABLE_POSITIONS = _positions(_FACE_OVAL, _NOSE)

# The nose tip, as a position in the mesh. Which way a head is turned is
# read from where this sits between the two edges of the face — see
# :func:`_turn_bias`.
_NOSE_TIP_POSITION = _MESH_POSITION[4]

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

# The named smile photographs, which a smile ladder replaces entirely.
_SMILE_ROLES: tuple[str, ...] = ("source_smile", "source_smile_closed")

# And what the speech ladder replaces. One photograph of one shape against a row
# of photographs of every shape the mouth passes through — the smile ladder's
# argument on the axis next door.
#
# **`source_mouth_o` is here, and it was measured onto this list rather than
# reasoned onto it.** On the 240 frames of `driving.mkv` where the owner is
# speaking, it won the mouth on **69.2% of them at a median share of 0.98** —
# one photograph of rounded lips, shown almost whole, for two thirds of a
# sentence. It is driven by `mouthPucker` and `mouthFunnel`, which sit above
# `_EXPRESSION_FULL` through ordinary speech, so it is the fourth family to
# behave the way the gaze photographs, the squint and the tilts did: active
# almost always, and therefore wrong almost always. The difference is that this
# time there is something strictly better to put in its place.
#
# **`source_mouth_pressed` and `source_mouth_wide` are not here.** Pressed lips
# are a *closed* mouth, which is the ladder's foot and the portrait's own
# photograph; it wins 13.3% of speech at a share of 0.12, which is a thin wash
# on frames the ladder does not want. `source_mouth_wide` never won a frame at
# all. Neither is in the ladder's way.
#
# **`source_mouth_open` came back off this list on 2026-08-25**, once it turned
# out that the 4K material stops at a parting of 0.38 and everything wider is
# 720p. It is a 4K photograph of a mouth open 0.50, which is further than the
# sharp part of the ladder reaches, and with the weight capped at the rung's own
# parting it takes the region back exactly where the ladder runs out of
# photographs rather than competing with it where it has them.
_SPEECH_ROLES: tuple[str, ...] = ("source_mouth_o",)

# And the two an eye ladder replaces. Both, because the ladder is signed: a
# resting eye is its zero, a shut one its top, and an eye held wide is the same
# axis in the other direction. Measured on the owner's own recording, the
# wide-eye photograph fired on 4.6% of frames at a median weight of 0.11 — a
# sixth of a full stare, laid over a face that was doing nothing, which is what
# they described as looking a little mad.
_EYES_ROLES: tuple[str, ...] = ("source_eyes_closed", "source_eyes_wide")

# Both brow photographs, for the same reason as both eye ones.
_BROWS_ROLES: tuple[str, ...] = ("source_brows_up", "source_brows_down")

_ROLE_REGION: dict[str, str] = {
    role: region for region, roles in _REGION_ROLES.items() for role in roles
}


def _misplaced(prefix: str, points: np.ndarray, base: np.ndarray) -> str | None:
    """Why this rung cannot be used, or ``None`` if it can.

    A rung is a whole face, cut into by whichever region it belongs to, and the
    regions are drawn around the *portrait's* features. So a photograph whose
    brows have landed where the portrait's eyes are is not an expression of this
    face at all — it is a differently posed one that the similarity fit could
    not undo, and using it paints a brow inside the eye region.

    Two of the owner's eye rungs were exactly that, and he identified both by
    name from the screen before this check existed: their brows sat 18 px below
    the portrait's brow line, inside the socket, and appeared as a second brow
    under the real one whenever the eyes were half closed.
    """
    brow = float(points[_BROW_POSITIONS][:, 1].max())
    eye_top = float(base[_EYE_POSITIONS][:, 1].min())
    brow_line = float(base[_BROW_POSITIONS][:, 1].mean())

    if prefix == _EYES_PREFIX and brow > eye_top:
        return f"its brows sit {brow - eye_top:.0f} px inside the portrait's eye"
    if prefix == _BROWS_PREFIX:
        eye = float(points[_EYE_POSITIONS][:, 1].min())
        if eye < brow_line:
            return f"its eyes sit {brow_line - eye:.0f} px above the portrait's brow line"
    return None


def refused_as_rung(prefix: str, points: np.ndarray, base_points: np.ndarray) -> str | None:
    """Why the engine would refuse this frame as a rung, or ``None`` if it would not.

    The load-time check, offered to the tools that build ladders, because a
    builder that does not run it loses a rung *silently*. Each one takes the
    sharpest frame in each band of the movement and nothing else, so a band
    whose sharpest frame is one the engine will refuse ends up with no rung at
    all — and the ladder acquires a hole nobody was told about.

    That is not hypothetical. The owner's eye ladder had exactly two holes, at
    a tenth and a fifth of a closing, and a blink therefore had to be rendered
    by fading a **shut** photograph in over a wide-open portrait. Both pictures
    were on screen at once while it faded, which is the iris seen through a
    closed eyelid.

    ``points`` are the frame's own landmarks in its own pixels; they are placed
    on the portrait's frame here, by the same fit the loader uses, because the
    check is about where the features *land*.
    """
    import cv2

    matrix = _similarity_onto(points, base_points)
    if matrix is None:
        return "it could not be placed on the portrait at all"

    placed = points.copy()
    placed[:, :2] = cv2.transform(
        points[:, :2].reshape(-1, 1, 2).astype(np.float32), matrix
    ).reshape(-1, 2)
    return _misplaced(prefix, placed, base_points)


def _rung_widths(
    descriptor: AvatarDescriptor, ladder: tuple[tuple[float, str | None], ...]
) -> dict[str, float]:
    """Each speech rung's own width, keyed the way the ladder keys it.

    ``_align_axis`` names a rung ``speech@<index>`` after its place in the
    manifest, and skips any it could not place — so the ladder is the list of
    keys that survived, and the index inside the key is how each one finds its
    asset again. Read here rather than threaded through ``_align_axis``, which
    the smile, the eyes and the brows share and none of them wants a second
    axis.
    """
    widths: dict[str, float] = {}
    for _, key in ladder:
        if key is None:
            continue
        index = int(key.removeprefix(_SPEECH_PREFIX))
        width = descriptor.manifest.assets[index].speech_width
        if width is not None:
            widths[key] = float(width)
    return widths


def _region_of(role: str) -> str | None:
    """Which part of the face a source may act on, or ``None`` for the whole head.

    A smile rung is a mouth photograph like any other — it differs from its
    neighbours in the mouth and nowhere else — so it belongs to the mouth
    region, and the prefix is how a rung says so without needing a role name of
    its own for every step of a smile.
    """
    if role.startswith((_SMILE_PREFIX, _SPEECH_PREFIX)):
        return "mouth"
    if role.startswith(_EYES_PREFIX):
        return "eyes"
    if role.startswith(_BROWS_PREFIX):
        return "brows"
    return _ROLE_REGION.get(role)


# How far each region reaches beyond its own landmarks, and how soft its edge
# is, both as a fraction of face width. Generous on purpose: an eye photograph
# has to carry the lid and the socket shadow, not just the lash line, or the
# boundary shows as a ring around the eye. The feather is what keeps the join
# invisible — a hard edge would trade ghosting for a visible patch.
_REGION_MARGIN: dict[str, float] = {"mouth": 0.10, "eyes": 0.11, "brows": 0.09}

# Where the brow ridge is divided between the two regions that meet on it, as a
# fraction of the way from the top of the eye up to the brow line.
#
# Not halfway. The join is feathered, and a feather is 5% of face width — wider
# than the gap between an eye and the brow above it — so a line drawn in the
# middle leaves the brows painting their own brow at three quarters strength,
# and the portrait's brow shows through the remaining quarter. A quarter of the
# way up gives the brows the whole of their own feature and still leaves the
# socket, which is all the eyes need.
_RIDGE_SPLIT = 0.25
_REGION_FEATHER = 0.05

# How softly the eye's own opening is fenced off from every other region, as a
# fraction of face width. See :func:`_build_regions`.
#
# A fifteenth of a region's feather, and the reason is that the two edges are
# not the same kind of edge. A region's feather hides the join between two
# photographs of *different parts of a face*, and wants to be wide. This one
# runs along the top of the eye, three or four pixels under a lowered brow
# rung's own brow, and a wide fade here would take that brow off — which is the
# second brow the owner reported on day 10, arrived at from the other side.
_OPENING_FEATHER = 0.015

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

# How far the nose sits off the middle of the face outline, at which the near
# cheek stops being eroded and at which it keeps all of its edge.
#
# The owner, turning their head: *"the part next to my ear shows on the face —
# it does not cover me there."* The mask is eroded by :data:`_MASK_SHRINK` all
# the way round, so that the join happens over skin on both sides rather than
# over the background. Frontal that is right; the edge it pulls back from is
# the silhouette, and outside it is hair and room. Turned, the near side's edge
# runs down in front of the ear, and what lies outside *that* is more cheek —
# so the erosion is pure loss there, a band of the user's real face left showing
# inside their own avatar, at the widest and most visible part of the turn.
#
# Measured on the owner's turn clip, as the nose's offset from the middle of the
# outline over face width: 0.14 frontal, 0.42 at 10-19 degrees, 0.69 at 20-29,
# 0.89 at 30-39. So the erosion on the near side fades out from about fifteen
# degrees and is gone by thirty-five — the range where the ear comes into view.
# The far side keeps it, because that edge is still a silhouette.
_NEAR_CHEEK_START = 0.45
_NEAR_CHEEK_FULL = 0.90

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

# How much of the *lips'* shape the avatar copies from the user.
#
# **The owner found this one himself**, on 2026-08-27, watching the application:
# *"the size of the teeth changes sometimes, and the size of the lips too.
# Sometimes a thick lip, sometimes thin... I think it is trying to fit its own
# lips exactly to the size of my lips. I do not actually have big lips. Some
# girl might have botox — you would not stretch the photograph onto her lips."*
#
# He is describing this file's own sentence, three constants up: the avatar
# inherits the user's **proportions** where it should be inheriting only their
# **motion**. The brows were damped for it, then the eyes. The lips were not,
# and they followed at 1.0.
#
# Measured over his four captures, as lip thickness against face height, over
# the avatar portrait's own — `x` is how many times its own it wears, `swing` is
# p95/p05 inside the ten seconds:
#
# |  | lower lip, x | swing | at 0.3: x | swing |
# |---|---|---|---|---|
# | `201745` | 1.27 | 1.14 | **1.06** | 1.05 |
# | `201806` | 1.40 | 1.28 | **1.09** | 1.26 |
# | `201814` | 1.43 | 1.33 | **1.13** | 1.23 |
# | `214327` | 1.34 | 1.17 | **1.09** | 1.14 |
#
# **A lower lip worn 27 to 43% thicker than the avatar's own**, and swinging by
# up to a third inside ten seconds. That is *"sometimes a thick lip, sometimes
# thin"*, and it is his lip, not the avatar's.
#
# **Between the eyes' 0.2 and the brows' 0.5, and the reason is the grid.** The
# eyes can be damped almost to nothing because a photograph covers every lid
# position they have. The mouth's photographs are a grid of 67 and it has to
# move *between* rungs, so some of the mesh has to stay or the mouth stops
# between them. Measured on the same four captures, the mouth's own opening at
# each candidate — the thing that must not be flattened along with the
# thickness:
#
# |  | 1.0 | 0.3 | 0.0 |
# |---|---|---|---|
# | `201806`, speaking | 0.0397 | 0.0433 | 0.0446 |
# | `214327`, speaking | 0.0612 | 0.0643 | 0.0664 |
# | `201814`, gently | 0.0192 | **0.0127** | **0.0097** |
#
# Three of the four keep their opening or gain it — the rungs carry it, which is
# what a grid of openings is for. The fourth is the quiet one, where his mouth
# barely parts and the grid's foot is the nearest rung: there the mesh was
# supplying the difference, and damping it takes a third of the opening away.
# 0.3 is where the thickness is his no longer and that third is still mostly
# there.
_LIP_FOLLOW = 0.3

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
# How far in from the face's outline the skin sample is taken, in pixels of the
# portrait. Enough to clear the hairline on one side and the background on the
# other, both of which sit on the hull itself.
_SKIN_ERODE = 15

# The largest correction a source may be given when its channels *disagree*,
# per channel of 255. Twelve is far above the 3 to 6 a library generated in one
# sitting needs, and a mismatch that is different on each channel is a different
# *light* rather than a different exposure — dragging one onto the other would
# hide that rather than fix it.
_SKIN_SHIFT_LIMIT = 12.0

# And the largest correction when they agree, which is a far weaker claim and
# earns a far larger allowance.
#
# **Measured across the owner's whole library on 2026-08-25, 95 sources.** The
# spread between the three channels' corrections is at most **8 of 255** and its
# median is 1 to 4 — on every family, including the eleven speech rungs that
# need 36 to 40. Nothing in this library is a colour shift. They are frames of
# one face under one lighting setup, generated on different days at different
# exposures, and an exposure difference is flat by definition.
#
# The old single limit of 12 was set when every source came from one sitting.
# The speech clip does not, and the cost of clamping it was the defect the owner
# reported the day the ladder shipped: *"the beard colour differs, and that makes
# a big difference"* — the mouth region arriving 24 of 255 dark, appearing and
# vanishing as the lips parted. Sixty is chosen to clear the 40 this library
# needs with room, and to stay far below a photograph of something else.
_SKIN_SHIFT_FLAT_LIMIT = 60.0

# How close the three channels' corrections have to be to count as one exposure
# rather than one light, per channel of 255. Ten: above the 8 of the worst
# source in this library, below any real colour cast.
_SKIN_SHIFT_AGREE = 10.0

# Where a region samples its own tone: inside the mask, outside every feature
# its photographs describe. See :func:`_match_ring`.
#
# `_RING_SOLID` is how much of the mask has to be present for a pixel to count —
# well past the feather, so nothing half-faded and therefore half-portrait
# enters the estimate. `_RING_CLEARANCE` is how far the features are held off in
# pixels, because a lip's edge is a gradient and the pixel beside it is still
# lip. `_RING_MINIMUM` is the fewest pixels a median may be taken over before
# the whole-face match is left to stand on its own.
_RING_SOLID = 0.9
_RING_CLEARANCE = 9
_RING_MINIMUM = 200

# Below this the correction is invisible and is not applied at all.
_SKIN_SHIFT_FLOOR = 1.0

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

# ---------------------------------------------------------------------------
# One photograph per region, chosen rather than averaged.
#
# The owner's complaint, in their words: they smile, the teeth appear, and the
# closed mouth is still there underneath at low opacity. Both halves of that
# have the same cause. A region used to mix *every* photograph that scored
# anything — at a wide smile the mouth was a smile photograph averaged with a
# pressed-lips one and a rounded one — and whatever the mix did not claim was
# left as base portrait, whose lips are shut. An average of pictures that
# disagree is a picture of neither, and the leftover is a ghost of a closed
# mouth over an open one.
#
# The mouth *interior* was fixed this way on day 8 and it worked; this extends
# the same rule outward from the interior to the whole region.
# ---------------------------------------------------------------------------

# How much stronger a challenger must be before a region switches to it.
#
# Without it the two nearest photographs trade places whenever their scores
# cross, which on a face mid-sentence is several times a second, and a switch is
# a hard cut. A tenth is wide enough that nothing flickers and narrow enough
# that a genuine change of expression is never held back — the scores separate
# by far more than this the moment a mouth actually opens.
_SWITCH_MARGIN = 0.10

# The expression strength at which the chosen photograph is applied whole, with
# no base portrait underneath it at all.
#
# Below this it fades in against the base, because an expression is a continuum
# and a photograph appearing at full strength would pop. Above it there is
# nothing to gain from keeping a shut mouth under an open one: seven tenths of
# the way into an expression the photograph *is* the answer, and the mesh has
# already deformed it to the shape the tracker measured.
_COMMIT_FULL = 0.70

# Regions where a photograph is worth nothing until it is worth most of it.
#
# The owner: *"two brows, and two more slightly above, faint."* That is what a
# raised-brow photograph looks like while it is fading in — the base portrait's
# brows are where they rest, the photograph's are raised, and both are on
# screen at once. Every region has this while a photograph is faint, and for
# the mouth it is small enough to live with, because a mouth photograph differs
# from the base mostly in *texture*. A brow differs from the base in *position*,
# which is the one difference that reads as a second copy.
#
# So the brows keep the mesh for small raises, which is what a mesh is for, and
# the photograph only arrives once it is most of the answer. Below the floor it
# contributes nothing; above it, it ramps from nothing to whole, so there is no
# step at the boundary.
_COMMIT_FLOOR: dict[str, float] = {"brows": 0.5}

# The most a region's photograph may fade in or out in one frame.
#
# The owner, on the first version of the region change: the teeth arrive *"as
# if the frame had been cut"*. They were right, and the cause is that the
# amount was read straight off the tracker. A blendshape crosses the commit
# point in a frame or two, so a photograph that is correct for the mouth's
# final shape was appearing at full strength while the lips were still moving.
# A real smile takes about a third of a second to form and the avatar's should
# take the same, whatever the tracker does in between.
#
# **Per second, not per frame.** It was per frame until the owner asked to try
# 60 fps, which would have made every expression take twice as long to arrive
# and looked like the engine getting slower on a faster machine. A face takes
# the time it takes whatever the camera is doing.
#
# Nothing to whole in about 0.27 s: slow enough to read as a face moving, fast
# enough that a smile forming reads as a smile forming.
_EASE_PER_SECOND = 3.7

# Except the eyes, which are the one part of a face that moves faster than that.
# A blink is 100-150 ms, so a lid that took a quarter of a second to close would
# still be halfway down when the real eye was open again — not a slow blink, a
# permanently half-lidded avatar. Ten per second closes it in a tenth.
#
# The rest of the face keeps the slower rate. Nothing else moves like a lid.
_EASE_PER_SECOND_BY_REGION: dict[str, float] = {"eyes": 10.0}

# Regions whose rungs are not faded in at all, but shown from the frame they are
# chosen. **The lids, and only the lids.**
#
# Raising their rate was not enough, and the reason is arithmetic rather than
# taste: a blink is three or four frames, and a tenth of a second is three
# frames, so the fade lasted the whole blink. Measured on the owner's recording,
# frame by frame through one blink: the lids close over frames 692-693 and open
# over 702-703, and on all four of those the eye region is a mix of a shut
# photograph and the wide-open portrait underneath — 33% and 67% of the way in.
# A translucent iris under a closed eyelid, up to 131 grey levels away from the
# picture either side of it. **That is the iris the owner saw through the lid**,
# and no threshold, mask or containment could have removed it, because the
# offending pixels are a genuine average of two photographs that disagree.
#
# So the fade goes, for the eyes. It is the last of the fade that day 10's
# eighth pass removed from the *amount*: a rung already says how far shut the
# eye is, so there is nothing left for an opacity to express. What replaces it
# is the ladder itself — the rungs are a tenth of a closing apart, so the step
# from one to the next is smaller than the fade's own first frame.
#
# The mouth and the brows keep theirs. A smile takes a third of a second to
# form and its photographs are on screen while it does; nothing there is asked
# to arrive inside the time a lid takes to fall.
_ARRIVES_WHOLE: frozenset[str] = frozenset({"eyes"})

# The frame interval used when there is nothing to measure one from, and the
# range a measured one is trusted in. A dropped frame or a paused debugger must
# not let an expression jump.
_FRAME_SECONDS = 1.0 / 30.0
_FRAME_SECONDS_MAX = 0.2


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

    __slots__ = (
        "_ceilings",
        "_floors",
        "_level_seen",
        "_means",
        "_pose",
        "_pose_seen",
        "_smoothed",
    )

    def __init__(self) -> None:
        self._floors: dict[str, float] = {}
        self._ceilings: dict[str, float] = {}
        self._means: dict[str, float] = {}
        self._pose: dict[str, float] = {}
        self._smoothed: dict[str, float] = {}
        # How many readings each pose estimate is built from. What separates
        # "this is all I have seen" from "this is where they sit".
        self._pose_seen: dict[str, int] = {}
        # And the same for the levels, which need it for the same reason — see
        # :data:`_LEVEL_WARMUP_FRAMES`.
        self._level_seen: dict[str, int] = {}

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
            self._level_seen[name] = 1
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
        #
        # **Except while it is new**, and that exception is the whole of why an
        # avatar used to open a session with its eyes shut. An estimate built
        # from one frame should move freely and one built from a thousand should
        # not — the same argument :meth:`relative_pose` makes, and the same
        # remedy: a plain running mean with a prior for the first
        # :data:`_LEVEL_WARMUP_FRAMES` readings. Both directions, because the
        # reading it has to escape from can be on either side of the truth: a
        # session opened mid-blink seeds it low and everything after reads wide.
        seen = self._level_seen.get(name, 1) + 1
        self._level_seen[name] = seen
        rate = _POSE_RATE if value > rest else _POSE_RATE * 0.1
        if seen <= _LEVEL_WARMUP_FRAMES and value > rest:
            # Fast, but only upward, so the asymmetry that keeps blinks out of
            # the estimate survives the warm-up. What a new estimate has to
            # escape is a seed taken from one frame, and every way of being
            # wrong that matters here — a session opened mid-blink, a tracker
            # still converging — leaves it sitting *below* the eye it is meant
            # to describe.
            rate = max(rate, 1.0 / (min(seen, _LEVEL_WINDOW_FRAMES) + _POSE_PRIOR_FRAMES))
        rest += rate * (value - rest)
        self._means[name] = rest
        return max((value - rest) / max(rest, 1e-6), 0.0)

    def level_seen(self, name: str) -> int:
        """How many readings this level is built from. Zero if it has none."""
        return self._level_seen.get(name, 0)

    def level_of(self, name: str) -> float | None:
        """The resting level of a signal, without learning from this reading.

        The mean :meth:`relative_rise` maintains, read by a second caller in the
        same frame. Advancing it twice would double how fast the estimate chases
        the face, which is the mistake :meth:`pose_offset` exists to avoid on
        the other estimator.
        """
        return self._means.get(name)

    def floor_of(self, name: str) -> float | None:
        """The resting floor of a blendshape, without learning from a reading.

        The read-only companion to :meth:`relative_expression`, for a second
        caller in the same frame — advancing the floor twice would quietly
        double how fast it follows the face.
        """
        return self._floors.get(name)

    def smooth(self, name: str, value: float, rate: float) -> float:
        """A light low-pass on a measurement, kept per face.

        Landmark positions jitter a pixel or two between frames on a face that
        is not moving, and a ladder shown whole turns jitter into a change of
        photograph. Smoothing the *measurement* is cheaper than holding the
        choice harder, and it does not delay a real movement the way a wider
        hold does.
        """
        was = self._smoothed.get(name)
        now = value if was is None else was + rate * (value - was)
        self._smoothed[name] = now
        return now

    def track_level(self, name: str, value: float) -> float:
        """A slow mean of a signal, advanced by this reading. Warm at first.

        For a measurement that moves both ways around rest and has no preferred
        direction — where a brow sits, as against how open an eye is, which only
        ever closes from its widest. The asymmetric estimators above would drift
        toward whichever way the face happened to move most.

        **The warm-up is a 2026-08-24 fix and it was the whole of the defect.**
        This was a plain slow mean seeded from the first frame, with none of the
        acceleration :meth:`relative_pose` and :meth:`relative_rise` both have —
        so a bad first frame stayed in it for a minute. Measured on
        `driving.mkv`: frame 0 reads a brow height of **0.1489 against the
        clip's median of 0.1014**, 47% high, and at ``_POSE_RATE`` the estimate
        needs a seventeen-second time constant to walk it off. Tenth by tenth
        the brow amount came out

            -0.455  -0.378  -0.276  -0.201  -0.119  -0.279 ...

        decaying across the whole clip, with a median of **-0.202**: a person
        talking normally, brows where they keep them, reading a fifth of the way
        into a frown for the length of a call.

        The other candidate was checked and dismissed. The brow signal *is*
        skewed — brows raise much further than they lower — so a mean sits above
        a median by construction, but only by **1.7%** here, which is a fortieth
        of what was seen. It is the seed, not the skew.

        Same remedy as its two neighbours, and the same shape: for the first
        :data:`_LEVEL_WARMUP_FRAMES` readings the rate floors at a
        :data:`_LEVEL_WINDOW_FRAMES` window with a prior. **Both directions,
        unlike :meth:`relative_rise`** — that one warms upward only to protect
        an asymmetry it needs against blinks, and a plain level has no asymmetry
        to protect. A seed can be wrong on either side and this one was high.
        """
        rest = self._means.get(name)
        if rest is None:
            self._means[name] = value
            self._level_seen[name] = 1
            return value

        seen = self._level_seen.get(name, 1) + 1
        self._level_seen[name] = seen

        rate = _POSE_RATE
        if seen <= _LEVEL_WARMUP_FRAMES:
            # Windowed rather than a mean of everything seen so far, for the
            # reason :meth:`relative_rise` gives: the tracker's own reading
            # drifts 36% over the first second and a half while the face does
            # not move, so an average across the warm-up is an average of a
            # journey rather than an estimate of where it arrives.
            rate = max(rate, 1.0 / (min(seen, _LEVEL_WINDOW_FRAMES) + _POSE_PRIOR_FRAMES))

        rest += rate * (value - rest)
        self._means[name] = rest
        return rest

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
        """``value`` as an offset from the resting head angle. Signed.

        Two rates, and which one applies depends on how much the estimate has
        seen. For the first :data:`_POSE_WARMUP_FRAMES` readings it is a plain
        running mean, so a session that opens with the head somewhere other than
        straight ahead is corrected in seconds rather than in half a minute.
        After that it is the old slow mean, weighted down by how far this
        reading sits from rest — see :data:`_POSE_TRUST_DEGREES` — so that a
        glance held at a second monitor is not mistaken for a new way of
        sitting.
        """
        rest = self._pose.get(name)
        if rest is None:
            self._pose[name] = value
            self._pose_seen[name] = 1
            return 0.0

        seen = self._pose_seen.get(name, 1) + 1
        self._pose_seen[name] = seen
        offset = value - rest

        if seen <= _POSE_WARMUP_FRAMES:
            rate = max(_POSE_RATE, 1.0 / (seen + _POSE_PRIOR_FRAMES))
        else:
            rate = _POSE_RATE / (1.0 + (offset / _POSE_TRUST_DEGREES) ** 2)

        self._pose[name] = rest + rate * offset
        return offset

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
        made._pose_seen = dict(self._pose_seen)
        made._level_seen = dict(self._level_seen)
        made._smoothed = dict(self._smoothed)
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
        "applied",
        "blend_points",
        "brow_ladder",
        "cached_blend",
        "cached_weights",
        "cavity",
        "correction",
        "descriptor",
        "eye_follow",
        "eye_ladder",
        "full_gap",
        "gaps",
        "holding",
        "image",
        "interior_role",
        "ladder",
        "last_seen",
        "lid_aspect",
        "neutral",
        "points",
        "regions",
        "scratch",
        "smile_ladder",
        "source_points",
        "sources",
        "speech_ladder",
        "speech_width",
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
        smile_ladder: tuple[tuple[float, str | None], ...] = (),
        eye_ladder: tuple[tuple[float, str | None], ...] = (),
        brow_ladder: tuple[tuple[float, str | None], ...] = (),
        speech_ladder: tuple[tuple[float, str | None], ...] = (),
        speech_width: dict[str, float] | None = None,
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

        # Which photograph currently holds each region. Carried between frames
        # because it is what a challenger has to beat: without it the two
        # nearest expressions swap on every score crossing. See
        # :func:`_one_per_region`.
        self.holding: dict[str, str] = {}

        # How much of it is on screen, also carried between frames: an
        # expression arrives and leaves over a fraction of a second rather than
        # wherever the tracker put it this instant. See :func:`_eased`.
        self.applied: dict[str, float] = {}

        # When this avatar was last given a frame, so the next one can be given
        # its share of a second rather than a fixed step. ``None`` until the
        # first frame arrives.
        self.last_seen: float | None = None

        # The turn ladder: photographs of this face at measured angles, sorted
        # by yaw, with ``None`` marking the base portrait's own rung at zero.
        # Empty for an avatar that has only the named turn roles.
        self.ladder = ladder

        # Photographs of this face part-way into a smile, sorted by how far in,
        # with ``None`` marking the portrait's own resting mouth at zero.
        self.smile_ladder = smile_ladder

        # And the lids closing: zero is a resting eye, one is a shut one.
        self.eye_ladder = eye_ladder

        # And the brows: zero at rest, positive raised, negative drawn down.
        self.brow_ladder = brow_ladder

        # And the lips parting, which is speech: zero with them together, about
        # 0.5 at an ordinary vowel. It shares the mouth region with the smile
        # ladder and the two are separated by which way the mouth is moving —
        # see :func:`_variant_weights`.
        self.speech_ladder = speech_ladder

        # And how wide each of its rungs is, above the portrait's resting mouth.
        # A second axis rather than a second ladder: a parting on its own does
        # not say which sound is being made, and a rung picked on parting alone
        # had whatever shape the sharpest frame in that band happened to have.
        self.speech_width = speech_width if speech_width is not None else {}
        self.descriptor = descriptor
        self.image = image
        self.points = points
        self.triangles = triangles
        self.sources = sources

        # This face's own resting lid, per eye, as a height over a width, and
        # the mesh with either iris fan taken out of it. Together they are how a
        # frame decides there is no eye left to draw an iris in — see
        # :func:`_shut`, and :data:`_IRIS_HIDDEN_BELOW` for the measurement.
        self.lid_aspect = tuple(_lid_aspect(points, eye) for eye, _ in _EYE_PAIRS)

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
        sources, source_points, gaps, ladder, smiles, eyes, brows, speech = self._align_variants(
            descriptor, image, points, triangles
        )
        full_gap = gaps.get(_OPENING_REFERENCE_ROLE, _LIP_GAP_FULL)

        self._loaded[descriptor.id] = _LoadedAvatar(
            descriptor,
            image,
            points,
            triangles,
            sources,
            full_gap,
            source_points,
            ladder,
            gaps,
            smiles,
            eyes,
            brows,
            speech,
            _rung_widths(descriptor, speech),
        )
        _LOG.info(
            "Loaded avatar %s: %dx%d portrait, %d mesh points, %d triangles, "
            "%d extra source(s), %d turn rung(s) from %+.0f to %+.0f degrees, "
            "%d smile rung(s), %d speech rung(s): %s",
            descriptor.id,
            image.shape[1],
            image.shape[0],
            len(points),
            len(triangles),
            len(sources),
            len(ladder),
            ladder[0][0] if ladder else 0.0,
            ladder[-1][0] if ladder else 0.0,
            max(len(smiles) - 1, 0),
            max(len(speech) - 1, 0),
            ", ".join(
                sorted(
                    r
                    for r in sources
                    if not r.startswith((_LADDER_PREFIX, _SMILE_PREFIX, _SPEECH_PREFIX))
                )
            )
            or "none",
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
        tuple[tuple[float, str | None], ...],
        tuple[tuple[float, str | None], ...],
        tuple[tuple[float, str | None], ...],
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
        smiles = self._align_axis(
            descriptor, base_points, width, height, aligned, aligned_points, "smile", _SMILE_PREFIX
        )
        eyes = self._align_axis(
            descriptor, base_points, width, height, aligned, aligned_points, "eyes", _EYES_PREFIX
        )
        brows = self._align_axis(
            descriptor, base_points, width, height, aligned, aligned_points, "brows", _BROWS_PREFIX
        )
        speech = self._align_axis(
            descriptor,
            base_points,
            width,
            height,
            aligned,
            aligned_points,
            "speech",
            _SPEECH_PREFIX,
        )
        # A rung's own lip gap, like every other mouth photograph's: it is what
        # says whether this one has an interior to show through the lips. The
        # speech rungs need it most of all — a rung chosen at a wide parting is
        # the one photograph on the mouth that frame, and if nothing records
        # that it has teeth behind the lips, nothing will paint them.
        for _, key in (*smiles, *speech):
            if key is not None:
                gaps[key] = _lip_gap_of(aligned_points[key])

        # Last, once every photograph is on the portrait's frame: put them all
        # on the portrait's *exposure* as well. Each one is a frame of a
        # different generated clip and carries whatever exposure that clip had,
        # and until now nothing reconciled them — see :func:`_match_to_portrait`
        # for the measurement that found it.
        reference = _skin_median(base_image, base_points)
        for key, source in aligned.items():
            aligned[key] = _match_to_portrait(
                source, aligned_points[key], reference, key, descriptor.id
            )

        return aligned, aligned_points, gaps, ladder, smiles, eyes, brows, speech

    def _align_axis(
        self,
        descriptor: AvatarDescriptor,
        base_points: np.ndarray,
        width: int,
        height: int,
        aligned: dict[str, np.ndarray],
        aligned_points: dict[str, np.ndarray],
        field: str,
        prefix: str,
    ) -> tuple[tuple[float, str | None], ...]:
        """Every asset carrying a measured smile, as rungs sorted by how much.

        The turn ladder's argument, on the axis the owner asked for. A single
        smile photograph cross-faded over a resting mouth has two failures at
        once: while it is faint there are two mouths on screen, and whatever
        weight it is at, the *shape* it draws is the shape it was photographed
        at — so a small smile is a wide one at low opacity rather than a small
        one. The lips never stretch; the teeth simply appear.

        A row of photographs taken while the smile forms replaces both. The
        nearest is chosen, so the shape is always a shape this face actually
        made, and the neighbours are close enough that moving between them is
        the smile moving.

        The base portrait is the rung at zero, marked ``None``, exactly as it is
        for yaw: a resting mouth is the portrait's own and nothing is mixed into
        it.
        """
        rungs: list[tuple[float, str | None]] = [(0.0, None)]

        for index, asset in enumerate(descriptor.manifest.assets):
            value = getattr(asset, field)
            if value is None:
                continue

            key = f"{prefix}{index}"
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
                    "Avatar %s: %s could not be placed by similarity", descriptor.id, asset.path
                )
                continue

            wrong = _misplaced(prefix, placed[1], base_points)
            if wrong is not None:
                _LOG.warning("Avatar %s: skipping %s — %s", descriptor.id, asset.path, wrong)
                continue

            aligned[key], aligned_points[key] = placed
            rungs.append((float(value), key))

        # A ladder of one rung is the base portrait on its own, which is not a
        # ladder. Returning it as one would be enough to make callers believe
        # there is a smile ladder and stop using the named smile photographs —
        # see the same guard on the turn ladder, where it was a real defect.
        #
        # **Sorted by the amount alone.** A plain sort falls through to the key
        # when two rungs carry the same amount, and the portrait's own rung is
        # keyed ``None`` — so a grid holding any rung at exactly zero, which a
        # closed smile is, compared a string with None and the avatar would not
        # load at all.
        return tuple(sorted(rungs, key=lambda rung: rung[0])) if len(rungs) > 1 else ()

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

        # The base portrait alone is not a ladder, and saying it is has a cost:
        # a caller that finds a ladder stops using the four named turn
        # photographs, so an avatar with those and no ladder lost its turns
        # entirely. Every avatar built since has had a ladder, which is why this
        # went unseen.
        return tuple(sorted(rungs)) if len(rungs) > 1 else ()

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


def _iris_arcs(mesh: np.ndarray) -> np.ndarray:
    """The four points between each iris's four, on the iris's own outline.

    See :data:`_IRIS_ARCS_PER_EYE` for why they exist. Two properties matter
    and both are deliberate:

    *   **On the outline rather than on a fitted circle.** Each new point takes
        the radius of the two neighbours it sits between, so an iris the tracker
        reports as slightly oval keeps its shape instead of being rounded out.
    *   **Ordered by angle, not by landmark number.** The four boundary points
        are bisected pairwise going round, and which tracker index is "top" is
        not something this file should have to know or a future tracker have to
        promise.
    """
    arcs = np.zeros((2 * _IRIS_ARCS_PER_EYE, 2), dtype=np.float32)
    for eye, ring in enumerate(_IRIS_RING_POSITIONS):
        points = mesh[ring]
        centre = points.mean(axis=0)
        offsets = points - centre
        radii = np.linalg.norm(offsets, axis=1)
        angles = np.arctan2(offsets[:, 1], offsets[:, 0])
        order = np.argsort(angles)

        for step in range(_IRIS_ARCS_PER_EYE):
            first = order[step]
            second = order[(step + 1) % _IRIS_ARCS_PER_EYE]
            start, end = float(angles[first]), float(angles[second])
            if end < start:
                end += 2.0 * np.pi
            middle = (start + end) / 2.0
            radius = float(radii[first] + radii[second]) / 2.0
            arcs[eye * _IRIS_ARCS_PER_EYE + step] = centre + radius * np.array(
                [np.cos(middle), np.sin(middle)], dtype=np.float32
            )
    return arcs


def _pixel_points(normalised: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """The mesh in pixel coordinates, or ``None`` if the landmarks are wrong.

    **The one place a mesh is built**, which is what lets the iris arcs be added
    here: the portrait's mesh and the live face's both come through this
    function, so they are computed by the same rule and correspond.
    """
    if normalised.shape[0] <= max(MESH_INDICES):
        return None

    subset = normalised[list(MESH_INDICES), :2].astype(np.float32)
    subset[:, 0] *= width
    subset[:, 1] *= height
    return np.vstack([subset, _iris_arcs(subset)])


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
    # **Once, and shared with the interior.** The mouth is drawn where
    # `_damp_brows` puts it, so the opening the teeth are painted into has to be
    # the same one — see `_LIP_FOLLOW`. Painting the interior at the *driver's*
    # lips while the mesh drew the avatar's would put the teeth outside the
    # mouth holding them.
    damped = _damp_brows(source_mesh, local, avatar.eye_follow, avatar.lid_aspect)
    canvas = warp_mesh(
        blended,
        source_mesh,
        damped,
        avatar.triangles,
        box_w,
        box_h,
    )

    _paint_mouth_interior(avatar, canvas, blended, damped)

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


def _covering_interior(avatar: _LoadedAvatar, opening: float) -> str | None:
    """The narrowest photograph whose own mouth still covers this opening.

    **The closest larger mouth, not the widest one.** The widest is a shout, and
    a shout behind a smile's lips is a tongue where there should be teeth. What
    is wanted is the least amount of somebody else's mouth that answers the
    question, so the search is for the smallest gap at or above the live one and
    the widest available only when nothing covers it.

    Rungs are included and are usually the answer: the library's openings run
    from a closed mouth to 0.123 of face height in twelve steps, so the frame
    that has to borrow rarely borrows from far away.
    """
    covering = [
        (gap, role)
        for role, gap in avatar.gaps.items()
        if gap >= max(opening, _LIP_GAP_OPEN) and role in avatar.sources
    ]
    if covering:
        return min(covering)[1]
    widest = [
        (gap, role)
        for role, gap in avatar.gaps.items()
        if gap >= _LIP_GAP_OPEN and role in avatar.sources
    ]
    return max(widest)[1] if widest else None


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

    **A second photograph fills what the first cannot reach, and that is new on
    2026-08-26.** The opening on screen is drawn from the *driver's* lips, and
    the interior can only be copied as far as the *photograph's* own lips reach.
    Everything between the two was flat cavity colour. Measured over the owner's
    recording, on the 438 frames where his lips are apart, the chosen photograph
    covered less than half the opening on **39.7%** of them — and on every one
    of the 116 frames the smile ladder won, with a median coverage of **17%**.
    That is his *"the teeth show when smiling, but very badly"*: a thin band of
    real teeth and the rest a flat maroon patch.

    So where the winner stops, the library's **narrowest opening that still
    covers this frame's** is placed underneath it — the closest larger mouth
    rather than the widest one, because the widest is a shout and this may be a
    smile. Only what neither photograph reaches is left as cavity. The winner is
    painted last and is never overdrawn: it decides what is seen through the
    lips, and the backdrop decides only what would otherwise be paint.
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

    opening = np.zeros((box_h, box_w), dtype=np.uint8)
    cv2.fillPoly(opening, [(lips - (x0, y0)).astype(np.int32)], 255)

    region = canvas[y0:y1, x0:x1]

    # Everything inside the live opening starts as cavity, so no part of it can
    # ever show a lip or a cheek.
    region[opening > 0] = avatar.cavity.astype(np.uint8)

    def place(from_image: np.ndarray, from_points: np.ndarray) -> None:
        """Copy one photograph's own interior into the live opening.

        Fitted on the upper lip alone — see ``_UPPER_LIP_POSITIONS``. The teeth
        hang from it; the jaw drops away from it.
        """
        fit: Any = cv2.estimateAffinePartial2D(
            from_points[_UPPER_LIP_POSITIONS], target[_UPPER_LIP_POSITIONS], method=cv2.LMEDS
        )[0]
        if fit is None:
            return
        fit = fit.copy()
        fit[0, 2] -= x0
        fit[1, 2] -= y0

        placed = cv2.warpAffine(
            from_image,
            fit,
            (box_w, box_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        # Where that photograph has an interior to give: its own inner lips,
        # moved by the same fit. Drawn as a polygon rather than warped as a
        # mask — it is the same answer without a second resampling, and it
        # follows the picture wherever the picture has gone, a turned head
        # included.
        interior = cv2.transform(from_points[_INNER_LIP_SLICE].reshape(-1, 1, 2), fit).reshape(
            -1, 2
        )
        covered = np.zeros((box_h, box_w), dtype=np.uint8)
        cv2.fillPoly(covered, [interior.astype(np.int32)], 255)
        cv2.copyTo(placed, cv2.bitwise_and(opening, covered), region)

    backdrop = _covering_interior(avatar, _lip_gap_of(target))
    if backdrop is not None and backdrop != avatar.interior_role:
        place(avatar.sources[backdrop], avatar.source_points[backdrop])
    place(source, source_points)


def _damp_brows(
    source: np.ndarray,
    target: np.ndarray,
    eye_follow: float = _EYE_FOLLOW,
    resting: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Pull the brow, eye and lip landmarks back toward the avatar's own
    proportions.

    The mesh maps the avatar's landmarks onto the user's, so wherever it acts at
    full strength the avatar inherits the user's *shape* and not merely their
    motion. Three places where that is visibly wrong:

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

    **Lips**, and the owner named this one himself: *"it is trying to fit its own
    lips exactly to the size of my lips. I do not actually have big lips."*
    Measured on his own captures the avatar was wearing his lower lip 27 to 43%
    thicker than its own, changing thickness by up to a third inside ten
    seconds. The mouth needs the mesh more than the eyes do — its photographs
    are a grid it has to move *between* — so it keeps more of the user's motion.
    See :data:`_LIP_FOLLOW`.

    The reference for all three is where a rigid fit of the whole face would put
    them — the avatar's own proportions, positioned on the live head.
    """
    import cv2

    matrix: Any = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)[0]
    if matrix is None:
        return target

    rigid = cv2.transform(source.reshape(-1, 1, 2), matrix).reshape(-1, 2)

    damped = target.copy()
    for where, follow in (
        (_BROW_SLICE, _BROW_FOLLOW),
        (_EYE_POSITIONS, eye_follow),
        (_MOUTH_POSITIONS, _LIP_FOLLOW),
    ):
        anchor = rigid[where]
        damped[where] = anchor + (damped[where] - anchor) * follow

    _place_irises(damped, target, rigid, resting)
    return damped


def _place_irises(
    damped: np.ndarray,
    target: np.ndarray,
    rigid: np.ndarray,
    resting: tuple[float, ...] | None = None,
) -> None:
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
    for index, (eye, iris) in enumerate(_EYE_PAIRS):
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
        if resting is not None and _shut(damped, eye, resting[index]):
            _tuck_iris(damped, eye, iris)


def _contain_irises(points: np.ndarray, resting: tuple[float, ...] | None = None) -> None:
    """Hold both irises inside the eyelids that are supposed to contain them.

    ``resting`` is this portrait's own lid aspect per eye. Given it, an eye that
    has closed past having room for an iris has the ring tucked away instead —
    see :func:`_tuck_iris`. Omitted, nothing is tucked, which is what the load
    path wants: it is measuring the resting lid, so it cannot yet be compared
    against it.
    """
    for index, (eye, iris) in enumerate(_EYE_PAIRS):
        _contain_iris(points, eye, iris)
        if resting is not None and _shut(points, eye, resting[index]):
            _tuck_iris(points, eye, iris)


def _tuck_iris(points: np.ndarray, eye: Sequence[int], iris: Sequence[int]) -> None:
    """Pull a shut eye's iris ring in to the middle of the lid holding it.

    **Not removed — moved.** The first attempt at this dropped the iris fan from
    the mesh outright, on the reasoning that a shut eye has no iris to draw. It
    does not, and the render was clean where the artefact had been; but the
    triangles it took out were the only ones covering the inside of the lid, and
    what showed through the hole was the rigid backdrop layer. That layer is the
    composite placed by a single similarity transform, so it carries the
    photograph's *own* lash line a few pixels from where the mesh puts it: a
    second, thinner, darker lash line inside the shut lid. The owner marked it
    with an arrow on both eyes.

    So the fan stays and the ring is tucked instead. Scaling about the lid
    ring's own centroid is what makes the two meshes agree: it is a linear
    function of the ring, so wherever the destination ring is close to an affine
    image of the source's — which two descriptions of one shut eye are — the fan
    is close to an identity, and it paints the lid's own pixels back onto
    themselves. The triangles between the tucked ring and the lid grow rather
    than shrink, which is the other half of it: the version that clipped the
    iris to the lid outline drove them to no area at all, and an affine fitted
    to three nearly-collinear points put a bright speck where the dark one was.
    """
    ring = points[eye]
    centre = ring.mean(axis=0)
    points[iris] = centre + (points[iris] - centre) * _IRIS_TUCKED


def _lid_aspect(points: np.ndarray, eye: Sequence[int]) -> float:
    """How open this lid ring is, as its height over its own width.

    A ratio rather than a length, so it says the same thing whichever frame the
    mesh is in — the avatar's own pixels or the live face's — and whatever size
    the face is on screen.
    """
    ring = points[eye]
    width = float(ring[:, 0].max() - ring[:, 0].min())
    if width <= 1e-6:
        return 0.0
    return float(ring[:, 1].max() - ring[:, 1].min()) / width


def _shut(points: np.ndarray, eye: Sequence[int], resting: float) -> bool:
    """Whether this lid has closed past the point of having an iris in it."""
    return _lid_aspect(points, eye) < resting * _IRIS_HIDDEN_BELOW


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


def _smiling(tracking: FaceTrackingResult) -> float:
    """How wide this mouth is, against the span between the outer eye corners.

    One function because two places have to agree about it: the engine reading
    a live frame, and `build_smile_ladder.py` reading a photograph to decide
    which rung it is. A ladder whose rungs were measured differently from the
    thing they are chosen by is a ladder that picks the wrong rung.

    **Geometry, not ``mouthSmile``, and that is a correction made on
    2026-08-21.** The blendshape saturates. Measured across the owner's ten
    installed smile rungs it moves only from 0.529 to 0.798, and the top five
    sit between 0.67 and 0.80 — five photographs filed as five different smiles
    which the measurement cannot tell apart. Worse, it puts them in the wrong
    order: `smile_140.png` is filed at +1.401 and reads *less* smile than
    `smile_108.png` at +1.084. The whole upper half of the mouth axis was
    resting on a number that had stopped moving.

    The mouth's own width does not stop moving: the same rungs run 0.586 to
    0.722 of an eye span, in order. **Raw, not clamped and not rest-subtracted**
    — a resting mouth is about 0.53 and the caller takes rest off, because where
    a mouth rests is a fact about the person rather than about smiling.
    """
    landmarks = tracking.landmarks
    if landmarks is None or landmarks.count <= max(*_MOUTH_CORNERS, *_OUTER_EYE_CORNERS):
        return 0.0

    points = landmarks.points[:, :2]
    span = float(np.linalg.norm(points[_OUTER_EYE_CORNERS[0]] - points[_OUTER_EYE_CORNERS[1]]))
    if span <= 1e-6:
        return 0.0
    return float(np.linalg.norm(points[_MOUTH_CORNERS[0]] - points[_MOUTH_CORNERS[1]])) / span


def _lip_parting(tracking: FaceTrackingResult) -> float:
    """How far the lips are apart. **The same measurement as :func:`_lip_gap`**,
    under the name the smile and speech axes read it by.

    **It divided by the eye span until 2026-08-26, and that carried the frame's
    shape into the answer.** The landmarks are normalised over the frame, so a
    vertical distance over a horizontal one is multiplied by the frame's aspect
    ratio. Live it cancels — every frame is the camera's shape — and every clip
    the library was built from happened to be 16:9, so it never showed. The
    first 9:16 clip broke it: the same mouth read 3.3 times less open. Its
    frames reached 0.64 of the axis where the 16:9 take of the same sentence
    reached 2.30, and letterboxing those same frames out to 16:9 put them at
    2.14 — while the gap measured against face height did not move at all,
    0.111 either way. Filed as they read, they would have been installed as
    barely-parted rungs, and a mouth hardly open would have been answered with
    a photograph of one wide open.

    **This is the fourth time this file has had this bug.** The lids, the brows
    and the lip gap were each fixed the same way, and :func:`_eye_opening`
    states the reason in full: two vertical distances divided by each other have
    no such factor. The eye span is the steadier anchor across a re-detection
    and that is why it was reached for here; it is not worth the frame's shape.

    **The denominator moves with the jaw, and that is accepted rather than
    missed.** The chin drops as the mouth opens, so a full opening reads about a
    seventh less than proportional — measured over 400 frames of the owner's own
    recording, the map from the old reading to this one runs 3.60 where the lips
    rest and 4.13 where they are widest. The axis stays monotone and the driver
    and the sources both pass through it, which is what the comparison needs.
    The alternative was a denominator with no calibration history.
    """
    return _lip_gap(tracking)


def _eye_amount(tracking: FaceTrackingResult, neutral: _Neutral) -> float:
    """Where the lids sit against a resting eye. Signed.

    Zero at rest, positive as they close, negative when they are open wider than
    usual. **Against the resting opening rather than the widest one**, and that
    is the difference between this and the closed-eye weight it replaces: a
    ceiling that follows the most open the eye has been puts *rest* somewhere in
    the middle of its own scale, so a face doing nothing reads as part way shut.
    Rest is where a face spends its time, and it is what both directions have to
    be measured from.

    Measured on the **eyelids** rather than on ``eyeBlink``, which reads 0.14 on
    eyes held shut and had the avatar squinting back at a person whose eyes were
    closed.
    """
    lids = _eye_opening(tracking)
    rest = neutral.level_of("eyeOpen")
    if rest is None:
        # First sight of this face. Seeding it here rather than leaving it to
        # `_source_weights` keeps the two independent: whichever asks first
        # establishes rest, and the answer on that frame is zero either way.
        neutral.relative_rise("eyeOpen", lids)
        return 0.0
    if rest <= 1e-6:
        return 0.0
    if neutral.level_seen("eyeOpen") < _LEVEL_WARMUP_FRAMES:
        # **Nothing, until there is something to compare against.** The owner:
        # *"I switched it on with my eyes open and it was showing closed eyes."*
        #
        # A faster-learning rest was not enough on its own, and the measurement
        # says why: over the first second and a half the tracker's own reading
        # walks 36% while the face sits still, so an estimate chasing it is
        # always behind and the difference between the two is read as a lid
        # coming down. There is no rate that wins a race against the thing it is
        # estimating. So for as long as the estimate is young the answer is
        # zero, the eyes are the portrait's own, and the cost is that a blink in
        # the first second and a half is not shown.
        return 0.0
    return (rest - lids) / rest


def _brow_height(tracking: FaceTrackingResult) -> float:
    """How high the brows sit above the eye line, over face height.

    **Measured on the brows**, for the third time in this file and for the third
    time because a blendshape would not answer it. ``browInnerUp`` reacts to the
    eyelid as well as to the brow — over the owner's recording it reached its
    maximum on 5% of frames, most of them around blinks — so a ladder driven by
    it put a fully raised brow on the face for an instant every time the eyes
    opened. That is the doubled brow and the angry look they photographed.

    **Above the eye line, and that is the 2026-08-24 correction.** It used to be
    above the *nose bridge*, on the reasoning that a bridge cannot be moved by a
    lid — sound, and it turned out to cost far more than it bought. The brow
    ladder could not be filed at all because this reading did not survive being
    taken again on the crop that gets written: day 12 measured the same face at
    0.0650 in its source frame and 0.0825 in its crop, and every attempt to
    rebuild the ladder came out worse than the one it replaced.

    Measured on 127 frames from the six brow clips, each detected twice — once
    on the frame, once on the crop cut from it — the instability is **not in the
    brows and not in the denominator**. It is the bridge landmark itself:

    ======================  =========================
    quantity                moves, over an eye span
    ======================  =========================
    the brow points         0.0038
    the face height         0.0032
    **the nose bridge**     **0.0209**
    ======================  =========================

    Dropping it takes the axis from **11.8% of its own span to 1.9%**, with a
    bias of +0.0002, which is below the 4% the mouth axis ships at and is what
    lets a rung be filed from the photograph rather than from the clip.

    The outer eye corners are the canthi and a lid does not move them. That was
    an argument rather than a measurement, so it was measured: across six clips
    the correlation between this reading and the eyelid opening runs from +0.95
    to -0.98, **changing sign**, and the nose-bridge version changes sign with
    it. A lid reaching the brow reading would push one way every time. What the
    correlation shows is a face raising its brows as it opens its eyes on some
    clips and not on others, which is a face and not an artefact.

    Divided by the face height rather than by the span between the eye corners,
    which is steadier still at 1.0%. Both distances are vertical this way, so
    the frame's aspect ratio cancels and the landmarks can stay normalised —
    against a horizontal denominator the same face reads 0.137 in a square
    portrait and 0.367 in a 16:9 frame. Reaching 1.0% means carrying the frame's
    size into ``FaceTrackingResult``, which every tracker backend then has to
    produce, and 0.9 of a percentage point does not buy that.
    """
    landmarks = tracking.landmarks
    if landmarks is None or landmarks.count <= max(max(_LEFT_BROW), max(_RIGHT_BROW)):
        return 0.0
    if landmarks.count <= max(_FOREHEAD_LANDMARK, _CHIN_LANDMARK, *_OUTER_EYE_CORNERS):
        return 0.0

    points = landmarks.points
    face = float(points[_CHIN_LANDMARK, 1] - points[_FOREHEAD_LANDMARK, 1])
    if face <= 1e-6:
        return 0.0

    eye_line = float(np.mean(points[list(_OUTER_EYE_CORNERS), 1]))
    brows = float(np.mean(points[list(_LEFT_BROW + _RIGHT_BROW), 1]))
    return (eye_line - brows) / face


def _brow_amount(tracking: FaceTrackingResult, neutral: _Neutral) -> float:
    """Where the brows sit against a resting face. Signed.

    Zero at rest, positive raised, negative drawn down. Measured from
    :func:`_brow_height` against a slow mean of it, because rest is where a face
    keeps its brows and no two faces keep them in the same place.
    """
    measured = _brow_height(tracking)
    if measured <= 1e-6:
        return 0.0

    # Smoothed before it is compared: the brow landmarks jitter about half a
    # rung's worth on a still face, and with a rung shown whole that is a
    # change of photograph three times a second. Two and a half frames of lag,
    # which is under a tenth of a second and slower than no brow ever moved.
    height = neutral.smooth("browHeight", measured, _BROW_SMOOTHING)
    rest = neutral.track_level("browHeight", height)
    if rest <= 1e-6:
        return 0.0
    return min(max(((height - rest) / rest) / _BROW_SPAN, -1.0), 1.0)


def _smile_share(tracking: FaceTrackingResult, neutral: _Neutral) -> float:
    """How much of an open mouth is a smile rather than an "ah", from 0 to 1.

    :func:`_smiling` used to answer this directly, because a blendshape reads
    zero on a resting face and one on a full smile. A mouth's *width* does not:
    it reads 0.53 of an eye span on a face doing nothing, and handing that
    straight to the caller put a closed-smile photograph on a still face at 0.95
    — which is how the change was caught.

    So the widening above rest, on the same scale the axis uses, clamped.

    **Rest is a level here, not a floor, and that is a 2026-08-27 fix.** It was
    :meth:`_Neutral.relative_expression`, which follows a new low in a quarter
    of a second and creeps back up over twenty — a running minimum, which is the
    right estimator for a signal that only ever rises from rest and the wrong
    one for a mouth. A mouth is *narrower* than rest every time it rounds an
    "oo" or an "o", several times a sentence, so the floor parked at the
    narrowest mouth recently made and every ordinary frame after it read as a
    widening. See :meth:`_Neutral.track_level`, which the brows moved to on
    2026-08-24 for this exact reason, and the measurement in
    :func:`_widened`.
    """
    width = _smiling(tracking)
    rest = neutral.track_level("smileAmount", width)
    return min(max((width - rest) / _SMILE_WIDEN_FULL, 0.0), 1.0)


def _smile_amount(tracking: FaceTrackingResult, neutral: _Neutral) -> float:
    """How far into a smile this face is, on the axis the smile ladder is filed by.

    The smile blendshape alone will not do it. Measured across the owner's
    smiles clip it reaches 1.0 at the open smile and stays there through the
    laugh, so everything from "teeth showing" to "laughing" collapses into one
    value and a ladder built on it stops where the teeth begin — which is
    precisely where it is needed.

    So the widening carries the axis and the opening extends it:
    ``widened x (1 + parted)``. Filed against the owner's own rungs it runs 0.10
    at the faintest smile, 0.20 at a closed one, 0.23 at the widest closed one,
    0.49 as the teeth appear and 0.60 at the broadest — ordered all the way,
    which the blendshape was not.

    **Multiplied and not added, and that was learned twice.** Adding them reads
    an open mouth as a smile whatever the mouth is doing, and this library's
    broadest smile photograph then landed on 10.5% of an ordinary speaking clip
    — a man saying words, wearing a grin. The widening is what makes something a
    smile; the opening only says how far into one it has gone. A mouth that
    parts without widening is speech, and speech has its own photographs.

    Measured **above the resting face**, like every other expression here.
    Where a mouth rests is a fact about the person: the owner's sits at 0.53 of
    an eye span and someone else's will not.
    """
    above = _widened(tracking, neutral)
    if above <= 0.0:
        return 0.0

    widened = above / _SMILE_WIDEN_SPAN
    return widened * (1.0 + _speech_amount(tracking))


def _widened(tracking: FaceTrackingResult, neutral: _Neutral) -> float:
    """How much wider than rest this mouth is. **Read, not learn.**

    :func:`_smile_share` has already advanced this floor for the frame, inside
    `_source_weights`, so everything that reads it afterwards reads it. Until
    2026-08-25 they did not: `_smile_amount` called
    :meth:`_Neutral.relative_expression` a second time on the same frame and
    advanced the floor again, so it fell at ``_FLOOR_FALL`` twice — 0.15
    becoming 0.2775 — and the value the ladder chose its rung by was
    systematically smaller than the value the share was computed from. It
    picked a rung slightly too low.

    That is the hazard :meth:`_Neutral.floor_of` exists to prevent, and the head
    pose is careful about it a screen further down: *"Read, not learn:
    `_source_weights` has already advanced the resting angle for this frame."*
    The mouth was not. Now it is, and the speech ladder reads it as well, which
    is a third caller on one frame.

    The seeding path stays, for the frame where nothing has advanced it yet: an
    avatar whose tracker returned no blendshapes never reaches
    :func:`_smile_share`, and a mouth axis that silently read zero for such a
    face would be a regression rather than a fix.

    **And rest is a slow level rather than a floor, from 2026-08-27.** It was a
    running minimum — see :func:`_smile_share` for the shape of it — and the
    owner's own captures are what found it. Three moments he marked wrong on
    2026-08-27, ten seconds each, his resting mouth measuring 0.57 of an eye
    span across:

    | | the learned rest | his width axis, lips *together* | the grid's foot |
    |---|---|---|---|
    | `moment-...-201745` | **0.476** | **0.153** | open on 99% of them |
    | `moment-...-201806` | **0.409** | 0.227 | open on 88% |
    | `moment-...-201814` | 0.539 | 0.065 | closed, correctly |

    The first is the defect entire: **he is sitting with his mouth shut and the
    engine reads him as three quarters of the way into a closed smile**, so
    :data:`_SPEECH_WIDTH_START` never closes, and a widened photograph is held
    over a closed mouth — measured, one rung for **9.6 seconds**. The floor got
    there in the ordinary way: it snapped down to a rounded "oo" at
    :data:`_FLOOR_FALL` and needed twenty seconds at :data:`_FLOOR_RISE` to
    climb back, which is longer than anyone stays still.

    The third capture is the same engine with a floor that happened to land near
    his rest, and it behaves. That is the control.
    """
    width = _smiling(tracking)
    rest = neutral.level_of("smileAmount")
    if rest is None:
        rest = neutral.track_level("smileAmount", width)
    return max(width - rest, 0.0)


def _speech_amount(tracking: FaceTrackingResult) -> float:
    """How far apart the lips are, on the axis the speech ladder is filed by.

    Absolute rather than against rest, and that is the difference between this
    axis and every other one. Where a mouth *rests* is a fact about the person —
    the owner's sits at 0.53 of an eye span wide — but where lips *touch* is not:
    they touch at zero on every face, and :data:`_SMILE_LIP_CLOSED` is how much
    of a gap the landmarks still report when they do.

    So there is nothing to learn here and nothing to warm up, which is worth
    having on the one axis that moves several times a second.
    """
    return max(_lip_parting(tracking) - _SMILE_LIP_CLOSED, 0.0) / _SMILE_OPEN_SPAN


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
        smiling = _smile_share(tracking, neutral)
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


def _is_rung(role: str) -> bool:
    """Whether this source is one of a ladder rather than a named expression."""
    return role.startswith((_SMILE_PREFIX, _SPEECH_PREFIX, _EYES_PREFIX, _LADDER_PREFIX))


def _commit(weight: float, region: str = "") -> float:
    """How much of a chosen photograph is applied, against the base underneath.

    Straight line to :data:`_COMMIT_FULL` and flat at one after it. The flat
    part is the point: past it the base portrait contributes nothing to the
    region, so there is no shut mouth left under an open one and no open eye
    left under a closed lid.

    Some regions start later than others — see :data:`_COMMIT_FLOOR` — and the
    rest of the range is stretched over what is left, so the curve stays
    continuous and still reaches one.
    """
    share = min(weight / _COMMIT_FULL, 1.0)
    floor = _COMMIT_FLOOR.get(region, 0.0)
    if floor <= 0.0:
        return share
    return max(share - floor, 0.0) / (1.0 - floor)


def _eased(
    region: str, wanted: float, applied: dict[str, float], seconds: float = _FRAME_SECONDS
) -> float:
    """``wanted``, but no further than this frame's worth of change.

    The photograph is chosen instantly and *arrives* over a fraction of a
    second. Only the amount is limited: which photograph it is has already been
    decided, and a fade between two of them is the double exposure this whole
    path exists to remove.
    """
    was = applied.get(region, 0.0)
    # Clamped here rather than at the call: a dropped frame, a paused debugger
    # or a clip replayed as fast as it will go must not let an expression jump,
    # whoever is asking.
    seconds = min(max(seconds, 0.0), _FRAME_SECONDS_MAX)
    limit = _EASE_PER_SECOND_BY_REGION.get(region, _EASE_PER_SECOND) * seconds
    step = min(max(wanted - was, -limit), limit)
    now = was + step
    applied[region] = now
    return now


def _arrives_whole(region: str, role: str) -> bool:
    """Whether this photograph is shown at once rather than faded in.

    Both halves have to hold. The region has to be one that moves faster than a
    fade — see :data:`_ARRIVES_WHOLE` — and the photograph has to be a *rung*,
    because only a rung already carries how far into the movement it is. A named
    photograph on the same region still fades, which is what an avatar with no
    eye ladder falls back to.
    """
    return region in _ARRIVES_WHOLE and _is_rung(role)


def _one_per_region(
    weights: dict[str, float],
    holding: dict[str, str],
    applied: dict[str, float] | None = None,
    seconds: float = _FRAME_SECONDS,
) -> dict[str, float]:
    """Keep the strongest photograph in each region and drop the others.

    Whole-head photographs — the turns, and the ladder's rungs — pass through
    untouched. They are not in competition with anything: they describe the
    angle the head is at, and the ladder already chooses among them by that
    angle. Everything with a region competes, and exactly one wins it.

    ``holding`` is the previous frame's winner per region, and it is what stops
    the choice flickering: a challenger has to beat the incumbent by
    :data:`_SWITCH_MARGIN` rather than merely tie with it. It is mutated in
    place, so it must belong to the avatar rather than to the frame.

    The winner's weight is replaced by how much of it to apply — see
    :func:`_commit`, and :func:`_eased` for why that amount is rate limited
    rather than read straight off the tracker. What used to be "this
    photograph's share of a mix" becomes "how much of this photograph is on
    screen", which is the only question left once there is nothing to mix it
    with.
    """
    kept = {role: weight for role, weight in weights.items() if _region_of(role) is None}
    applied = {} if applied is None else applied

    contenders: dict[str, dict[str, float]] = {}
    for role, weight in weights.items():
        region = _region_of(role)
        if region is not None:
            contenders.setdefault(region, {})[role] = weight

    for region, scores in contenders.items():
        winner = max(scores, key=lambda role: scores[role])
        held = holding.get(region)
        # The incumbent keeps the region unless it is beaten by a clear margin.
        defended = held is not None and held != winner and held in scores
        if defended and scores[winner] - scores[held] < _SWITCH_MARGIN:  # type: ignore[index]
            winner = held  # type: ignore[assignment]
        holding[region] = winner

        # **A rung is shown whole.** For a named photograph the weight says how
        # much of that expression this frame has, and fading it in against the
        # base is how an expression arrives. For a rung it says no such thing:
        # the amount is already in *which rung was chosen* — the eye ladder's
        # 0.3 rung is a photograph of an eye three tenths shut — so mixing it
        # with the portrait underneath puts a wide-open eye back under a closing
        # lid at exactly the opacity the owner keeps reporting. The choosing has
        # already answered the question the fade was trying to answer.
        wanted = 1.0 if _is_rung(winner) else _commit(scores[winner], region)
        if _arrives_whole(region, winner):
            applied[region] = share = wanted
        else:
            share = _eased(region, wanted, applied, seconds)
        if share > 0.0:
            kept[winner] = share

    # A region with no photograph this frame still has to *leave*, and leaving
    # takes as many frames as arriving did. Only once it is gone does it stop
    # being a region with an incumbent — otherwise a photograph fading out would
    # have no name to be drawn under.
    #
    # Unless it is a region that arrives whole, which leaves the same way: a
    # fade out is the same double exposure as a fade in, and the owner saw both
    # — the eye ghosted on the way down and again on the way back up.
    for region in tuple(holding):
        if region in contenders:
            continue
        if _arrives_whole(region, holding[region]):
            del holding[region]
            applied.pop(region, None)
            continue
        share = _eased(region, 0.0, applied, seconds)
        if share > 0.0:
            kept[holding[region]] = share
        else:
            del holding[region]
            applied.pop(region, None)

    return kept


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

    matrix = _similarity_onto(points, base_points)
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


def _similarity_onto(points: np.ndarray, base_points: np.ndarray) -> np.ndarray | None:
    """The rotate-scale-translate that puts this face on the base's frame.

    Fitted on the landmarks an expression does not move, so that the expression
    the photograph was taken for cannot drag its own alignment. ``None`` when
    the fit fails, which a face at an extreme angle can genuinely cause.
    """
    import cv2

    # Typed as Any: OpenCV's stubs declare a non-optional return, but the fit
    # genuinely can fail and returns None when it does.
    matrix: Any = cv2.estimateAffinePartial2D(
        points[_STABLE_POSITIONS, :2].astype(np.float32),
        base_points[_STABLE_POSITIONS, :2].astype(np.float32),
        method=cv2.LMEDS,
    )[0]
    result: np.ndarray | None = matrix
    return result


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

    # **No pixel belongs to two regions.** The owner, still seeing photographs
    # mixed after each region was cut to one of them: *"especially around the
    # eyes and brows"*. Which is where the regions overlap — a brow's box is
    # grown by its margin and reaches down over the socket, an eye's reaches up
    # — so a closed eyelid and a lowered brow were both painted over the same
    # skin, in sequence, and the result is an average of two photographs after
    # all the work to stop being one.
    #
    # Each region is therefore cut back by whatever the regions before it have
    # already claimed, in the order they are built: the mouth first, then the
    # eyes, then the brows. The eyes own the socket because that is the feature
    # in it. What the brows keep is the skin above, where they are the only
    # photograph with anything to say.
    claimed = np.zeros((height, width), dtype=np.float32)

    # **Where the brows end and the eyes begin.** Giving the whole overlap to
    # the eyes was right for the socket and wrong for the ridge above it: a
    # brow rung could then not paint over the portrait's own brow, because that
    # sits inside what the eyes had claimed — so the portrait's brow stayed
    # where it was and the rung's brow appeared above it. Two brows, and a dark
    # seam between them, which the owner reported as looking angry, and which
    # happened when the eyes opened because that is when an eye rung is on.
    #
    # So the pair is divided anatomically rather than by priority: halfway
    # between the brows and the top of the eyes. Each keeps its own half, and
    # the feather crosses the line as it crosses every other edge.
    brow_line = float(points[_BROW_POSITIONS][:, 1].mean())
    eye_top = float(points[_EYE_POSITIONS][:, 1].min())
    split = eye_top + (brow_line - eye_top) * _RIDGE_SPLIT

    # The brows may reach down to the top of the eye and no further: their own
    # margin is a tenth of a face and would otherwise cover the iris, and a
    # brow photograph has nothing to say about an eye.
    brow_floor = eye_top
    rows = np.arange(height, dtype=np.float32)[:, None]

    # The portrait's own eye openings, lids and irises together. **No other
    # region may paint inside them.**
    #
    # The owner, on the third look at a shut eye: *"behind the closed-eye
    # photograph, I think the open-eye one is still there."* They were right.
    # Every other photograph in the library was taken with the eyes open — the
    # whole-head turn rungs above all, and one of those is mixed in on nearly
    # every frame because a head is never at exactly its resting angle — so
    # whatever share of the opening the eyes did not claim was *an open eye*,
    # and the mesh then squeezed it into the slit the shut lid had become.
    # Measured on this avatar, the eyes' own mask over the portrait's opening
    # fell as low as **0.333**: two thirds of an open eye at the worst pixel.
    #
    # Feathered far more tightly than the regions themselves — see
    # :data:`_OPENING_FEATHER`. A region's own feather is 5% of a face, which is
    # wider than the gap between the top of an eye and the brow above it, so
    # fading this over the same distance would reach up the ridge and clip the
    # brow rungs that hang lowest. It has to be sharp enough to stay inside the
    # eye and soft enough not to draw a line of its own.
    filled_eyes = np.zeros((height, width), dtype=np.uint8)
    for eye, iris in _EYE_PAIRS:
        hull = cv2.convexHull(points[eye + iris][:, :2].astype(np.int32))
        cv2.fillConvexPoly(filled_eyes, hull, 255)
    keep = max(3, int(face_width * _OPENING_FEATHER) | 1)
    opening = cv2.GaussianBlur(filled_eyes, (keep, keep), 0).astype(np.float32) / 255.0

    built: list[_Region] = []
    # Brows before eyes, which is the order they claim overlapping skin in.
    # Measured on the owner's own library: the brow rungs place their brows
    # anywhere from 23 px above the portrait's to 19 px below it, because that
    # is the range a brow moves through. A mask cut to a line through the ridge
    # therefore clipped the lowered rungs, and where a rung was clipped the
    # portrait's own brow showed through — the second brow, and it appeared
    # while the eyes were closing because that is when a rung is held.
    #
    # So the brows keep every pixel their photographs need, the eyes are cut at
    # the ridge line and take what is left below it.
    for name, positions in (
        ("mouth", _MOUTH_POSITIONS),
        ("brows", _BROW_POSITIONS),
        ("eyes", _EYE_REGION_POSITIONS),
    ):
        present = [role for role in sources if _region_of(role) == name]
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
        filled = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(filled, cv2.convexHull(group[:, :2].astype(np.int32)), 255)
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
        #
        for extra in extras:
            cv2.fillConvexPoly(grown, cv2.convexHull(extra[:, :2].astype(np.int32)), 255)
        if name == "eyes":
            grown = np.where(rows >= split, grown, 0).astype(np.uint8)
        elif name == "brows":
            grown = np.where(rows <= brow_floor, grown, 0).astype(np.uint8)

        mask = cv2.GaussianBlur(grown, (feather, feather), 0).astype(np.float32) / 255.0

        # Nobody but the eyes paints inside an eye. A brow photograph, a mouth
        # photograph and a turned head all have an open eye where this face's
        # shut one is, and any share of it left to them is that open eye.
        if name != "eyes":
            mask *= 1.0 - opening

        # What is left after the regions before it have taken theirs, and what
        # it takes in turn. Subtracted rather than cut, so the join stays a fade
        # — and the two can never add up to more than one photograph's worth.
        #
        np.clip(mask - claimed, 0.0, 1.0, out=mask)
        np.clip(claimed + mask, 0.0, 1.0, out=claimed)

        # **And each photograph is put on the portrait's tone over the skin this
        # region will actually show.** :func:`_match_to_portrait` has already
        # matched the whole face, which is the right sample for a turn rung
        # because a turn rung *is* the whole head. It is the wrong one for a
        # region: measured on the owner's speech rungs, a correction fitted on
        # the face median and applied in full left the mouth region 12 to 17 of
        # 255 too bright, where clamping it had left the same region 20 to 26
        # too dark. The clip is not uniformly darker than the portrait — it is
        # lit from a slightly different place, so the lower face and the whole
        # face do not want the same number.
        #
        # The sample is a **ring, drawn per photograph**: inside the region,
        # outside the portrait's own feature and outside this photograph's. Skin
        # in both pictures by construction, so nothing the photograph was taken
        # for — no cavity, no teeth, no shut lid — can reach the estimate. Per
        # photograph and not once for the region, because one ring avoiding
        # *every* rung's open mouth at once is no ring at all: on this avatar it
        # came to five pixels.
        clearance = _RING_CLEARANCE | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clearance, clearance))
        solid = mask[y0:y1, x0:x1] > _RING_SOLID
        base_patch = image[y0:y1, x0:x1]
        cropped = {}
        for role in present:
            patch = np.ascontiguousarray(sources[role][y0:y1, x0:x1])
            own = filled.copy()
            if role in source_points:
                cv2.fillConvexPoly(
                    own,
                    cv2.convexHull(source_points[role][positions][:, :2].astype(np.int32)),
                    255,
                )
            ring = solid & ~cv2.dilate(own, kernel)[y0:y1, x0:x1].astype(bool)
            if int(ring.sum()) >= _RING_MINIMUM:
                reference = np.median(base_patch[ring].reshape(-1, 3), axis=0)
                patch = _match_ring(patch, ring, reference, name, role)
            cropped[role] = patch

        built.append(
            _Region(
                name=name,
                box=(x0, y0, x1, y1),
                mask=np.ascontiguousarray(mask[y0:y1, x0:x1])[:, :, None],
                sources=cropped,
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

    # **A gap in the ladder is not a licence to average across it.** The claim
    # above — that a mix of two angles three degrees apart is indistinguishable
    # from either — is true at three degrees and false at eight, and a ladder
    # cut from a real clip has gaps: the owner's second turn clip left one of
    # eight degrees between +11 and +19, because that is where the head was
    # moving rather than stopping. Averaged across it, a cheek at +11 lies over
    # a cheek at +19 and the face reads as smeared — the owner's word was
    # *dough*, around the nose, where the two disagree most.
    #
    # So the cross-fade keeps its width in degrees rather than in rungs: it
    # happens over :data:`_LADDER_CROSSFADE_DEGREES` in the middle of the gap,
    # and everywhere else one photograph is shown whole. At a three-degree
    # spacing this is exactly the old behaviour; at eight it is the nearer rung
    # for two and a half degrees either side, then a quick change.
    if span > _LADDER_CROSSFADE_DEGREES:
        width = _LADDER_CROSSFADE_DEGREES / span
        share = min(max((share - 0.5) / width + 0.5, 0.0), 1.0)

    # A rung at zero weight is dropped rather than carried: it would cost a
    # cache key and break "a resting face uses no sources at all", which is the
    # property that makes a still avatar byte-identical to its own photograph.
    weights: dict[str, float] = {}
    for rung, weight in ((below, 1.0 - share), (above, share)):
        if rung[1] is not None and weight > 1e-6:
            weights[rung[1]] = weight
    return weights


def _nearest_smile(
    ladder: tuple[tuple[float, str | None], ...], smiling: float, held: str | None = None
) -> dict[str, float]:
    """The one rung nearest this much smile, at a weight of how much there is.

    **One, not two.** The yaw ladder crosses a short fade between neighbours
    because a head passes through every angle on its way and a jump would read
    as a flick of the head. A smile does not work that way: the rungs are
    photographs of one face at ten stages of the same movement, the shape
    changes little between neighbours, and what the owner objected to was
    *two mouths at once*. So the nearest is shown whole and the change from one
    to the next is carried by :func:`_eased`, which is a third of a second — the
    time a smile takes.

    The weight is the amount of smile rather than one, so a faint smile is a
    faint photograph of a faint smile, and an ordinary talking mouth can still
    win the region on the shape it is making.
    """
    return _nearest_rung(ladder, smiling, _SMILE_LADDER_START, held)


def _nearest_speech(
    ladder: tuple[tuple[float, str | None], ...],
    widths: dict[str, float],
    parted: float,
    widened: float,
    weight: float,
) -> dict[str, float]:
    """The rung nearest this mouth **on both axes**, at ``weight``.

    Speech is the one family where how far the movement has gone does not
    describe the movement. An "oo" and an "ah" can part the lips by the same
    amount and are not the same mouth, and the difference between them is the
    width. Filed on parting alone, the rung chosen at a given opening had
    whatever width the sharpest frame in that band happened to have — measured
    on the owner's speech clip, anything from 0.25 narrower than his resting
    mouth to 0.22 wider — and he reported the result on sight: *"in the first
    photo I am saying O, in the second A"*, with the O rendered as a wide A.

    Both axes are in the engine's own normalised units, so a unit of width and a
    unit of parting are the same size and the distance between them is plain
    Euclidean.

    **And nothing is held.** Every other ladder keeps the rung in hand until a
    challenger is clearly nearer, and this one did too, at four times their
    margin. That is what made the mouth passive: a rung held until the mouth is
    0.15 away from it is two or three rungs past its neighbour by the time it
    changes, so the mouth moves late and then jumps. See the note above the
    speech constants for the five captures that settled it.
    """
    rungs = [(amount, key) for amount, key in ladder if key is not None]
    if not rungs:
        return {}
    # **Either axis, not the distance between them.** A closed smile parts the
    # lips barely at all and is not the portrait's mouth in any other way, so
    # gating on the parting alone kept the grid out of exactly the frames the
    # smile ladder existed to cover. A plane distance was the first answer and
    # it was wrong in the other direction: an ordinary speaking mouth sits at a
    # widening of 0.046, which clears a threshold of 0.05 on its own, so the
    # grid took the region on frames where the mouth is at rest and the portrait
    # is the right photograph. That cost 10 ms a frame and bought nothing.
    if parted < _SPEECH_LADDER_START and widened < _SPEECH_WIDTH_START:
        return {}

    def distance(rung: tuple[float, str | None]) -> float:
        amount, key = rung
        return float(np.hypot(amount - parted, widths.get(str(key), 0.0) - widened))

    nearest = min(rungs, key=distance)
    key = nearest[1]
    return {} if key is None else {key: min(abs(weight), 1.0)}


def _nearest_rung(
    ladder: tuple[tuple[float, str | None], ...],
    amount: float,
    start: float,
    held: str | None = None,
    freeze: bool = False,
    weight: float | None = None,
) -> dict[str, float]:
    """The one rung nearest ``amount``, at a weight of how much there is.

    Shared by the smile, the lips and the lids, which want the same thing and
    for the same reason. **One rung, not two**: unlike yaw, where a head passes
    through every angle on its way and a jump would read as a flick, these are
    stages of one movement whose neighbours differ little — and what the owner
    objected to was two photographs on screen at once. The change from one to
    the next is carried by :func:`_eased`.

    ``weight`` splits *which rung* from *how much of it*, and only the speech
    ladder needs it. Everywhere else the amount answers both — the eye ladder's
    0.3 rung is a photograph of an eye three tenths shut, so how far into the
    movement it is and how much of it there is are one number. Speech shares the
    mouth with the smile, and the two are separated by which way the mouth is
    moving rather than by how far; scaling the *amount* would pick a rung at the
    wrong parting, so the scaling goes on the weight alone.
    """
    if len(ladder) < 2:
        return {}

    # Frozen: the measurement is not to be trusted this frame, so whatever was
    # chosen last stays and nothing new starts. Holding is not the same as
    # leaving — leaving would be a change of picture too — and starting on an
    # untrustworthy reading is the very thing being avoided.
    if freeze:
        current = next((rung for rung in ladder if rung[1] == held), None)
        if current is None or current[1] is None:
            return {}
        return {current[1]: min(abs(current[0]), 1.0)}

    if abs(amount) < start:
        return {}

    nearest = min(ladder, key=lambda rung: abs(rung[0] - amount))

    # The rung in hand keeps the region until another is *clearly* nearer.
    # Without it the tracker's own noise walks the choice back and forth across
    # a boundary several times a second, and since a rung is shown whole, each
    # crossing is a change of photograph. Half a rung of margin, so a real
    # movement still moves it immediately.
    if held is not None:
        current = next((rung for rung in ladder if rung[1] == held), None)
        if (
            current is not None
            and abs(current[0] - amount) <= abs(nearest[0] - amount) + _RUNG_HOLD
        ):
            nearest = current

    if nearest[1] is None:
        return {}
    return {nearest[1]: min(abs(amount if weight is None else weight), 1.0)}


def _carried_share(region: str, share: float, trust: float) -> float:
    """How much of a photograph is applied once the head has moved under it.

    For a mouth, fading it out as the head turns away is right: a hard threshold
    would make an expression vanish mid-sentence, and a mouth photograph differs
    from what is under it mostly in *texture*.

    **The lids are shown whole or not at all**, here as everywhere else. An eye
    photograph differs from what is under it in whether the eye is *open*, so
    any share between nought and one is a shut lid laid over an open eye — the
    same double exposure this file has now removed twice, arriving by a third
    road.

    The owner could not have described it better: *"eyes shut, head turned
    slightly, and it is still there."* Reproduced from their own recording,
    which contains the pair `driving.mkv` never does — the head past 20 degrees
    with the lids down. Over its 4006 frames the eyes region is carried on 2668
    and **1572 of those, 59%, were painted at part strength**, down to 0.62.
    Through the turn they marked it runs 1.00 at 19 degrees to **0.82 at 28**,
    so a fifth of an open eye was showing through a shut lid. Rendered three
    ways at those frames, the thin dark line inside the lid is in the scaled one
    and in neither of the others.

    Keeping it whole costs little: 0.82 of trust is a misfit of **4.8 px on a
    400 px face**, a bad number for a mouth and a small one for a lid that is
    shut. Past ``_CARRY_LOST`` — 12 px on that face — the photograph is dropped,
    as it always was.
    """
    if region in _ARRIVES_WHOLE:
        return share if trust > 0.0 else 0.0
    return share * trust


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

    if avatar.smile_ladder or avatar.speech_ladder:
        # **One grid on the mouth as of 2026-08-26, where there were two ladders
        # and a line drawn between them.** The line was `smiling`, and it was
        # asked to say whether a mouth was a smile or a word so that the two
        # families could each own their half. A mouth is not one or the other:
        # it has a width and an opening, and a smile is a corner of the same
        # plane speech moves in.
        #
        # What made this urgent is that the smile ladder's own axis was a
        # *product* of the two, `widened x (1 + parted)`, and a product cannot
        # tell a wide mouth from an open one. It answered a mouth open 0.10 with
        # a photograph whose lips were 0.018 apart: over the owner's recording
        # the chosen photograph covered a median of **17%** of the opening drawn
        # for it on **every one** of the frames the smile ladder won. That is
        # the same defect the speech family had before the grid — *"in the first
        # photo I am saying O, in the second A"* — left standing in the family
        # next door.
        #
        # So the grid takes the whole mouth and the smile ladder is not
        # consulted while it exists. The two named smile roles go with it for
        # the reason they always did: the grid holds better versions of both.
        #
        # `smiling` is gone with the line it drew. It survives in
        # :func:`_source_weights`, where the *named* photographs still split a
        # mouth between a smile and a word — an avatar with no grid still has
        # only one photograph of each.
        held = avatar.holding.get("mouth")

    if avatar.smile_ladder and not avatar.speech_ladder:
        # The smile ladder owns the smile the way the turn ladder owns yaw. The
        # two named smile roles go with it: one of them is a photograph the
        # ladder holds ten better versions of, and keeping it would put a wide
        # smile back in competition with the right-sized one.
        for role in _SMILE_ROLES:
            weights.pop(role, None)
        weights.update(
            _nearest_smile(avatar.smile_ladder, _smile_amount(tracking, avatar.neutral), held)
        )

    if avatar.speech_ladder:
        # And the speech ladder owns the parting. What goes with it is
        # `source_mouth_o` and the two named smiles, for the smile ladder's
        # reason word for word: each is one photograph of one shape, the grid
        # holds a photograph of every shape the mouth passes through, and
        # leaving them in puts one fixed mouth in competition with the
        # right-sized one at exactly the widths where the grid matters most.
        #
        # **`source_mouth_open` is deliberately not on that list** — see
        # `_SPEECH_ROLES`, which says why it came back off it on 2026-08-25.
        # Its *gap* is kept either way: `_OPENING_REFERENCE_ROLE` reads the
        # asset, not the weight, and it is still what says how far open "fully
        # open" is.
        for role in (*_SPEECH_ROLES, *_SMILE_ROLES):
            weights.pop(role, None)
        parted = _speech_amount(tracking)
        widened = _widened(tracking, avatar.neutral) / _SMILE_WIDEN_SPAN

        # **Weighed by the rung's own parting, not the driver's, once the driver
        # is past the top of the ladder.** A weight says how much of *this
        # photograph* there is, and a photograph of a mouth open 0.35 has 0.35
        # of an opening in it however wide the driver's mouth goes. Without the
        # cap the ladder's weight keeps climbing on a photograph that has
        # stopped, and it holds the region against a named photograph that
        # genuinely opens further.
        top = max(amount for amount, _ in avatar.speech_ladder)
        top_width = max(avatar.speech_width.values(), default=0.0)
        weights.update(
            _nearest_speech(
                avatar.speech_ladder,
                avatar.speech_width,
                parted,
                widened,
                # How far from rest this mouth is, on the plane the grid is
                # filed on — and capped on each axis at how far the grid
                # reaches, for the reason the parting alone was capped: a
                # photograph of a mouth open 0.35 has 0.35 of an opening in it
                # however wide the driver's goes, and past the top of the grid
                # the region belongs to a named photograph that genuinely opens
                # further.
                float(np.hypot(min(parted, top), min(widened, top_width))),
            )
        )

    # The lids, measured once: the brows are read against them as well as the
    # eye ladder being chosen by them.
    lid = (
        _eye_amount(tracking, avatar.neutral) if (avatar.eye_ladder or avatar.brow_ladder) else 0.0
    )

    if avatar.brow_ladder:
        # The brows' own ladder, and both named photographs go with it. A raised
        # brow and a lowered one are two ends of one movement, which is what the
        # signed axis says and what two separate photographs could not.
        for role in _BROWS_ROLES:
            weights.pop(role, None)
        weights.update(
            _nearest_rung(
                avatar.brow_ladder,
                _brow_amount(tracking, avatar.neutral),
                _BROWS_LADDER_START,
                avatar.holding.get("brows"),
                # **Not while the lids are moving.** Measured over the owner's
                # recording, where the brows sit correlates with the lids at
                # r=+0.24 whatever it is measured against — the landmarks ride
                # the blink, so a blink reads as a brow raise of 0.63 and puts a
                # raised brow on the avatar for three frames. That is the flash
                # they photographed. A brow does not change while an eye blinks,
                # so the choice is held until the lids are back.
                freeze=abs(lid) > _BROW_FREEZE_LID,
            )
        )

    if avatar.eye_ladder:
        # The lids' own ladder, and the shut-eye photograph goes with it for the
        # same reason the smile's did: the ladder holds ten better versions of
        # it, one for each stage of the same closing.
        for role in _EYES_ROLES:
            weights.pop(role, None)
        weights.update(
            _nearest_rung(avatar.eye_ladder, lid, _EYES_LADDER_START, avatar.holding.get("eyes"))
        )

    if not weights:
        avatar.blend_points = avatar.points
        avatar.interior_role = None
        avatar.eye_follow = _EYE_FOLLOW
        avatar.holding.clear()
        avatar.applied.clear()
        return avatar.image

    # One photograph per region from here on. Everything below still speaks of
    # mixes and shares because a whole-head photograph is still mixed across the
    # frame and the ladder still crosses two rungs three degrees apart — but
    # inside a region there is now exactly one photograph, and `share` says how
    # much of it is on screen rather than how it was averaged with its rivals.
    # How long this frame is worth. Measured from the camera's own timestamps,
    # so an expression takes the same time at 60 fps as at 30 — and clamped,
    # because a dropped frame or a paused debugger must not let one jump.
    now = tracking.metadata.capture_timestamp
    last = avatar.last_seen
    avatar.last_seen = now
    seconds = _FRAME_SECONDS if last is None else now - last

    weights = _one_per_region(weights, avatar.holding, avatar.applied, seconds)

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
    whole_head = {role: weight for role, weight in weights.items() if _region_of(role) is None}
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
                share = _carried_share(region.name, share, trust)
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
    _contain_irises(mesh, avatar.lid_aspect)

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


def _skin_median(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """The median colour of this photograph's face, per channel.

    Over the convex hull of the landmarks, eroded so the hull's own edge — hair
    on one side, background on the other — cannot reach the sample. The median
    rather than the mean because a wide-open mouth brings a black cavity with
    it, and an average would follow it while a median does not.
    """
    import cv2

    hull = cv2.convexHull(points[:, :2].astype(np.int32))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    eroded: np.ndarray = cv2.erode(mask, np.ones((_SKIN_ERODE, _SKIN_ERODE), dtype=np.uint8))
    inside = image[eroded.astype(bool)]
    if inside.size == 0:
        return np.zeros(3, dtype=np.float32)
    median: np.ndarray = np.median(inside.reshape(-1, image.shape[2]), axis=0).astype(np.float32)
    return median


def _match_to_portrait(
    image: np.ndarray, points: np.ndarray, reference: np.ndarray, role: str, avatar_id: str
) -> np.ndarray:
    """Put this photograph on the portrait's exposure.

    **Every source in a library is a frame of a different generated clip, and
    nothing was reconciling their exposures.** The engine relights the finished
    composite to the *room* — that is :func:`_match_colour` — but between the
    portrait and the photograph laid over part of it there was no step at all,
    so a region arrived carrying whatever exposure its own clip happened to
    have.

    Measured on the owner's recording of 2026-08-21: during a blink the eye
    region reads **4.9 of 255 brighter** against the cheek than it does with the
    eyes open, while the cheek itself does not move at all — a patch of another
    exposure, appearing and vanishing with the lids. The eye rungs measure 3.2
    lighter than the portrait and the ones they replaced measured 5 to 6 darker,
    so this is not new; what made it visible is that a rung now arrives whole
    rather than faded in, and on 47% of frames rather than 33%.

    A single offset per channel, not a field: what differs between two takes of
    one face under one lighting setup is the exposure, and an offset is what an
    exposure difference is. A gradient would also chase the pose, and a rung is
    a photograph of a *different pose* on purpose.

    **How far it may be dragged depends on whether the three channels agree.**
    A difference that is the same on all three is an *exposure*: the same face,
    the same light, a different day's take. A difference that is not is a
    different *light*, and quietly dragging one onto the other would hide that
    rather than fix it. So the first gets :data:`_SKIN_SHIFT_FLAT_LIMIT` and the
    second keeps :data:`_SKIN_SHIFT_LIMIT`.

    That distinction is a measurement rather than a principle. Across the
    owner's 95 sources the spread between the three channels' corrections is at
    most 8 of 255 — on the turn rungs, the eye rungs, the named roles and the
    eleven speech rungs that need 36 to 40 alike. Not one source in the library
    is a colour shift.

    It was a single limit of 12 until 2026-08-25, and what that cost was
    reported by the owner within an hour of the speech ladder shipping: *"the
    beard colour differs, and that makes a big difference."* The speech clip was
    generated on its own day, 36 of 255 darker than the portrait, and 12 of that
    was corrected — so the mouth region arrived 24 dark, and because a rung is
    shown whole, it appeared and vanished as the lips parted.
    """
    mine = _skin_median(image, points)
    offset = reference - mine
    largest = float(np.abs(offset).max())
    spread = float(offset.max() - offset.min())
    flat = spread <= _SKIN_SHIFT_AGREE
    limit = _SKIN_SHIFT_FLAT_LIMIT if flat else _SKIN_SHIFT_LIMIT
    if largest > limit:
        _LOG.warning(
            "Avatar %s: %s is %.0f of 255 from the portrait's exposure and %.0f of that is "
            "%s, further than a take of the same face should be; correcting by %.0f",
            avatar_id,
            role,
            largest,
            spread,
            "colour" if not flat else "spread across the channels",
            limit,
        )
        offset = np.clip(offset, -limit, limit)
    elif largest > _SKIN_SHIFT_LIMIT:
        # Large but flat, so it is corrected in full — said out loud, because a
        # source this far from the portrait is still worth knowing about even
        # when it is only an exposure.
        _LOG.info(
            "Avatar %s: %s is %.0f of 255 darker or lighter than the portrait, flat to "
            "within %.0f; correcting it in full",
            avatar_id,
            role,
            largest,
            spread,
        )
    if largest < _SKIN_SHIFT_FLOOR:
        # Below a value of 255 the correction is not visible and the round trip
        # through int16 is not free. Most sources land here once a library is
        # generated in one sitting.
        return image
    shifted: np.ndarray = np.clip(image.astype(np.int16) + offset.astype(np.int16), 0, 255).astype(
        np.uint8
    )
    return shifted


def _match_ring(
    patch: np.ndarray, ring: np.ndarray, reference: np.ndarray, region: str, role: str
) -> np.ndarray:
    """Put one region's copy of a photograph on the portrait's tone.

    The residual after :func:`_match_to_portrait`, fitted where it matters. That
    one samples the whole face, which is right for a whole-head photograph and
    wrong for a region: a face is not lit evenly, so two takes that differ by an
    exposure over the whole of it can still differ by something else over one
    part. Measured on the owner's speech rungs — a clip generated on its own day
    — a whole-face correction applied in full left the mouth 12 to 17 of 255 too
    bright.

    ``ring`` is skin in both pictures by construction: inside the region,
    outside every feature any of its photographs describe. So the difference
    over it is tone and nothing else, and a median over it is not moved by the
    open mouth or the shut lid that the photograph exists for.

    **Clamped by magnitude alone, and not by whether the channels agree.** The
    whole-face match asks that question because it is a real one over a whole
    face: three channels disagreeing there means one photograph was lit
    differently from the other, and a flat shift would hide it rather than fix
    it. Over one region it is not the same question. A mouth ring is a few
    hundred pixels of skin; a light cannot fall across it unevenly enough to
    matter, so three channels disagreeing there says only that this patch is a
    different colour from the skin it is being set into — which is the whole job.

    **It was clamped by the agreement test until 2026-08-26, and that is what
    the owner was looking at.** Six of the mouth rungs in this library sit 17 to
    32 of 255 from the portrait over their own ring with the channels 11 to 19
    apart, so each was judged "two lights" and corrected by 12 — leaving 5 to 20
    of 255 standing. On screen that is the pale patch around the mouth that
    appears when the lips move and vanishes when they stop, with the moustache
    washing out inside it. The magnitude limit still applies and still refuses a
    correction large enough to mean a photograph of something else.
    """
    mine = np.median(patch[ring].reshape(-1, 3), axis=0)
    offset = (reference - mine).astype(np.float32)
    largest = float(np.abs(offset).max())
    if largest < _SKIN_SHIFT_FLOOR:
        return patch

    spread = float(offset.max() - offset.min())
    limit = _SKIN_SHIFT_FLAT_LIMIT
    if largest > limit:
        _LOG.warning(
            "Region %s: %s is %.0f of 255 from the portrait over the skin around it, "
            "%.0f of that spread across the channels; correcting by %.0f",
            region,
            role,
            largest,
            spread,
            limit,
        )
        offset = np.clip(offset, -limit, limit)
    shifted: np.ndarray = np.clip(patch.astype(np.int16) + offset.astype(np.int16), 0, 255).astype(
        np.uint8
    )
    return shifted


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


def _turn_bias(points: np.ndarray) -> tuple[float, bool]:
    """How far off centre the nose sits, and which side of it is the near one.

    A turn read from the mesh itself rather than from the tracker's pose, so it
    is measured in the same picture the mask is drawn in — the two can never
    disagree about which cheek is facing the camera. The near side is simply
    the side of the nose with more face on it.

    Returns the offset as a fraction of half the outline's width, and ``True``
    when the near cheek is on the right of the picture.
    """
    oval = points[_OVAL_SLICE]
    nose = float(points[_NOSE_TIP_POSITION][0])
    left = nose - float(oval[:, 0].min())
    right = float(oval[:, 0].max()) - nose
    total = left + right
    if total <= 1e-6:
        return 0.0, True
    bias = (right - left) / total
    return abs(bias), bias > 0.0


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

    edge = _trim_forehead(points)
    outline = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(outline, [edge.astype(np.int32)], 255)

    face_width = max(float(points[:, 0].max() - points[:, 0].min()), 1.0)
    shrink = max(1, int(face_width * _MASK_SHRINK))
    feather = max(3, int(face_width * _MASK_FEATHER) | 1)

    trimmed = cv2.erode(outline, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink, shrink)))

    # **The near cheek keeps its edge once the head is turned**, because what
    # lies just outside it there is more face rather than background. See
    # :data:`_NEAR_CHEEK_START`. Never further out than the outline itself: the
    # restored piece is clipped to it, so this can only undo the erosion and
    # never reach into hair or into the room.
    bias, near_right = _turn_bias(points)
    ramp = min(max((bias - _NEAR_CHEEK_START) / (_NEAR_CHEEK_FULL - _NEAR_CHEEK_START), 0.0), 1.0)
    if ramp > 0.0:
        nose = float(points[_NOSE_TIP_POSITION][0])
        side = edge[edge[:, 0] >= nose] if near_right else edge[edge[:, 0] <= nose]
        if len(side) >= 3:
            near = np.zeros((height, width), dtype=np.uint8)
            corners = np.vstack([side, points[_PROFILE_POSITIONS]]).astype(np.int32)
            cv2.fillConvexPoly(near, cv2.convexHull(corners), 255)
            cv2.bitwise_and(near, outline, dst=near)
            keep = max(1, round(shrink * (1.0 - ramp)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (keep, keep))
            cv2.erode(near, kernel, dst=near)
            cv2.max(trimmed, near, dst=trimmed)

    # The features go on **after** the erosion, at full strength. The erosion
    # exists to pull the outer boundary inside the real face so the blend
    # happens over skin on both sides; these are not a boundary, they are the
    # middle of the face, and eroding them would leave a rim of somebody's real
    # eye showing. Only the feather crosses them.
    cv2.fillConvexPoly(trimmed, cv2.convexHull(points[_PROFILE_POSITIONS].astype(np.int32)), 255)

    softened = cv2.GaussianBlur(trimmed, (feather, feather), 0)
    mask: np.ndarray = (softened.astype(np.float32) / 255.0)[:, :, None]
    return mask
