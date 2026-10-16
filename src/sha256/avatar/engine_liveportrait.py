"""LivePortrait avatar engine: a face generated every frame, not chosen.

The warp engine picks the nearest photograph of the mouth and cuts to it, twelve
or thirteen times a second, and the owner's word for the result was *"a game
character"*. No number of photographs removes the cuts. This engine removes
them by not having photographs: one portrait is read once into LivePortrait's
feature volume, and every frame the network re-draws that face with the
driver's expression and pose. There is always a mouth between two mouths.

**Measured before it was built** (2026-09-04, RTX 5070 Ti, fp16 weights, the
three networks in one CUDA graph): **25.5 ms a frame** including the upload of
the driver crop and the download of the 512x512 result. B005's 38.4 ms was
autocast without a graph; the graph's output matches eager to 0.0000. The
budget at the camera's 30 fps is 33.2 ms.

**How a frame goes through.** The driver's face is cut from the camera frame by
a similarity fitted the way LivePortrait fits its own crop — square,
``CROP_SCALE`` (2.2) face heights, eyes level, the box shifted up an eighth so
the forehead is in it; the height is the driver's resting one scaled by his
distance, so nothing on the face can zoom the box — and reduced to 256x256.
The motion extractor reads pose and expression from
that crop. Relative to the first frames seen (the *anchor*), the difference is
applied to the portrait's own keypoints, and the warping and decoding networks
draw the portrait's face at 512x512. That face is placed back into the camera
frame by the *driver's* crop transform: the two crops are built by one rule, so
the generated face lands where the driver's was, at the driver's size and roll,
under a mask that is the portrait's own face oval and the driver's own face
oval, intersected — see :func:`_frame_mask` — and colour-matched to the room
under it (:func:`_face_colour_correction`).

**Stitching's own 2-D shift is off by default (day 4, 2026-09-05).** The
stitching network — trained to paste a re-animated face back onto a *still*
source photograph without a seam at the neck — ends by nudging every keypoint
sideways. Here the driver's own crop already re-centres the face every frame,
so that nudge fights the crop instead of correcting a paste: measured on the
owner's own clip, the avatar's nose sat 14.8 px left to 15.2 px right of his
real one (p10..p90) with the shift on, -13 px for whole seconds while his eyes
were shut or he smiled, +10 while talking — 0.88-correlated with the avatar's
apparent size, not with anything in his own motion. Three other candidates
(pinning the driver's translation onto the portrait, pinning scale, removing
the mean of the expression delta) each changed nothing; dropping the shift
alone gave ±4 px on every resting segment. See :meth:`_drive`.

**The extractor's output is smoothed, off by default (day 7, 2026-09-06).**
An EMA over the extracted expression, translation, scale and pose, in state
that persists between frames -- see :func:`_ema_`/:func:`_motion_smoothing`
and the ``motion_smoothing`` option. Answers a second live complaint: while
the owner sat still the avatar did not, his eye ratio flat at 0.314..0.320
over 120 frames against the avatar's own 0.291..0.365, with the crop itself
steady on the same frames -- the extractor's own frame-to-frame noise,
passed straight through. See :meth:`_drive`.

**The mouth and brows move only when he measurably does (day 9,
2026-09-06).** The owner's third live complaint: *"it parts its lips and
half-shuts its eyes when I simply lift my head -- it invents reactions"*. The
motion extractor reads a lifted head partly as an expression; measured on his
clip, a still mouth on a raised gaze rendered as a mouth ratio of up to 0.10
against his own 0.005. Now a per-region gate, from his own tracked mouth
ratio, mouth width and brow height against the anchor's rest, scales the
extractor's relative expression on the mouth and brow keypoints every frame
-- 1.0 when he is smiling, talking or raising his brows, 0.0 when he is not
(see the GATES section, :func:`_mouth_gate`/:func:`_brow_gate`). The eyes
never take the extractor's reading at all (day 8); the eye retargeting
network that drives them now adds exactly nothing at rest (``eye_rest``, see
:meth:`LivePortraitEngine._retarget_rest`) -- before, its "no change" answer
alone half-closed the avatar's eyes.

**What is not here yet, on purpose.** No GPU-side paste; the full landmarker
still runs for the crop where a face detector would do. Those are phase 2 of
``docs/development/ROADMAP.md``. This file exists so the owner can see the
picture first.

Weights: ``scripts/fetch_liveportrait.py`` into ``models/liveportrait``. The
network code is vendored under :mod:`sha256.avatar.liveportrait`, MIT.
"""

from __future__ import annotations

import functools
import math
import statistics
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sha256.avatar.hair import (
    HAIR_MODEL_RELATIVE_PATH,
    HairMatteWorker,
    HairSegmenter,
    hair_alpha,
    hair_box,
    head_confidence,
    skin_weight,
    union_box,
)
from sha256.core.avatar import AvatarDescriptor
from sha256.core.enums import DeviceKind, PixelFormat
from sha256.core.errors import AvatarLoadError, EngineError
from sha256.core.types import FaceTrackingResult, FrameMetadata, HeadPose, RenderedFrame, VideoFrame
from sha256.utils.logging import get_logger
from sha256.utils.timing import now

__all__ = ["ENGINE_ID", "LivePortraitEngine", "crop_matrix"]

_LOG = get_logger(__name__)

ENGINE_ID = "liveportrait"
SOURCE_PORTRAIT_ROLE = "source_portrait"

# Under ``models/``. What ``scripts/fetch_liveportrait.py`` writes.
DEFAULT_WEIGHTS_RELATIVE_PATH = Path("liveportrait") / "liveportrait"

# The crop, in LivePortrait's own terms (``CropConfig``): a 512 square holding
# CROP_SCALE face heights, shifted up by an eighth so the forehead is inside,
# reduced to the network's 256. The same rule cuts the portrait and every
# driver frame, which is what lets the generated face be pasted back with the
# driver's transform.
#
# CROP_SCALE is 2.2, not LivePortrait's 2.3, because their "face" is not ours.
# Their box comes from a landmark set that starts at the eyebrows; ours is the
# MediaPipe mesh's whole height, forehead top to chin, about a seventh
# taller -- so 2.3 of *our* heights shows the network a face about 10%
# smaller than it was trained on. Measured 2026-09-05 (day 6) on hard frames
# of the owner's clip, the same portrait and driver at each scale, the output
# read back by MediaPipe:
#
# | driver box (x rest face height)         | 1.96 | 2.07 | 2.18 | 2.30 | 2.41 |
# | mouth wide + pitch 10 (his mouth 0.549) | 0.53 | 0.51 | 0.44 | 0.01 | 0.16 |
# |   ... his eyes 0.260                    | 0.19 | 0.17 | 0.26 | 0.14 | 0.15 |
# | frown, chin down 5 (his eyes 0.283)     |      | 0.10 | 0.30 | 0.12 |      |
# |   ... his mouth shut 0.008              |      | 0.04 | 0.01 | 0.12 |      |
# | brows raised (his eyes 0.376)           |      | 0.28 | 0.32 | 0.35 |      |
#
# Cliffs, not slopes, and not the same cliff for every expression: at 2.3
# the wide-open mouth collapses (eyes shut, nose dragged into a shut mouth);
# at 2.1 the frown collapses the same way (eyes shut, lips pursed -- the
# "random reactions" the owner reported). 2.2 renders both. No earlier crop
# had held any of these frames at one stable scale (day 5's width-governed
# box shrank 5% on a turn or a wide mouth; day 4's whole-mesh box grew with
# the mouth), which is why none of this was visible before the day-6 crop
# stopped moving. Why the network is this sensitive to a 5% framing change
# is not understood; three frames, one portrait -- see
# docs/development/ROADMAP.md, day 6, for the whole-clip numbers behind it.
#
# CROP_VY_RATIO is -0.06, not LivePortrait's -0.125, and this axis has NO
# cliff (measured 2026-09-10, day 14). The ratio moves the crop's centre up
# the face by that fraction of the crop, i.e. it decides how the 512 rows
# are spent between hair and neck: at -0.125 the portrait's chin sat at row
# 489 with 22 px under it, and a jaw that opens 24-51 px (his, driving.mkv
# 986-1103) had nowhere to be followed -- the mask clipped at the array's
# edge into a hard line (see _CHIN_DROP_MAX). Swept -0.16..0.0 on the
# day-6 hard frames of both clips (wide mouth + pitch, frown, brows, toothy
# smile, turn, blink) and then whole-clip at -0.09/-0.06/-0.03/0.0, three
# portraits: every frame renders at every value, no grimace, no collapse,
# turn lag and brow gain unchanged; what moves is smooth -- the mouth and
# smile respond more as the face rises in the crop (driving talk mar gain
# 1.07 -> 1.25, smile width 0.45 -> 0.73 from -0.125 to 0.0; natural talk
# 0.93 -> 1.19) and blinks a little less (0.92 -> 0.85). -0.06 keeps the
# mouth at today's gain (talk 1.10 / natural 1.03, smile 0.51 / 1.04, blink
# 0.90) and leaves 56 px under the chin, enough for the whole open-jaw
# range once the mask's own edge margin (measured, load_avatar) is taken.
CROP_SIZE = 512
MODEL_SIZE = 256
CROP_SCALE = 2.2
CROP_VY_RATIO = -0.06

# How many tracked frames the anchor — the driver's resting expression and
# pose — is averaged over before it is frozen. LivePortrait uses its first
# frame; a call does not start with a neutral face on cue.
DEFAULT_ANCHOR_FRAMES = 15

# The split (day 16, 2026-09-12; the ROADMAP's A3). Every engine option is
# either about the PICTURE -- how the portrait is cut, masked and pasted, and
# what its own networks are allowed to do to it -- or about the PERSON in
# front of the camera: how far their readings are trusted, how their
# jitter is told from their speech, what their brow does to the
# extractor's mouth. The first set lives in the avatar's metadata.json and
# travels with the portrait; the second is a *driver profile*
# (configs/local.driver.json, written by "Calibrate me" --
# sha256.calibration) laid over whichever avatar is loaded, so a new
# person and a new portrait each get the right defaults without the
# other's. ``set_driver_options`` accepts only the driver's keys.
DRIVER_OPTION_KEYS: frozenset[str] = frozenset(
    {
        "anchor_frames",
        "expression_gates",
        "expression_gain",
        "eye_gain",
        "eye_headroom",
        "eye_scale",
        "gaze_gain",
        "gaze_down_relief",
        "gaze_rest_tracking",
        "brow_gain",
        "brow_follow",
        "brow_deadband",
        "driver_hairline",
        "brow_mouth_leak",
        "brow_mouth_leak_slopes",
        "mouth_smile_gain",
        "mouth_opening_gain",
        "mouth_lateral_gain",
        "mouth_gate_mar",
        "mouth_gate_width",
        "mouth_activity_mar",
        "mouth_activity_width",
        "mouth_rest",
        "lip_rest_rows",
        "lip_rest_tracking",
        "hold_pitch_degrees",
        "hold_yaw_degrees",
        # Day 22 evening: the turn past the knee driven harder (see _turn_boost),
        # and the clamp's own yaw cap, apart from the fade's (see _turn_cap).
        "turn_boost",
        "turn_cap_degrees",
        # Day 23: the yaw delta scaled per side (see _turn_gains) -- his right
        # turns drew 0.7-0.8 of his angle, his left ones 1.0-1.1.
        "turn_gain_left",
        "turn_gain_right",
        "depth_hold_yaw",
        "motion_smoothing",
        "size_smoothing",
        "crop_smoothing",
        "crop_smoothing_beta",
        "crop_centre_smoothing",
        "crop_centre",
        "translation_gain_x",
        "translation_gain_y",
        "scale_gain",
        "yaw_pivot_z",
        "crop_shift_x",
    }
)
PORTRAIT_OPTION_KEYS: frozenset[str] = frozenset(
    {
        "crop_scale",
        "crop_vy_ratio",
        "crop_border",
        "stitching",
        "eye_retargeting",
        "lip_retargeting",
        "lip_gain",
        "eye_expression_gain",
        "eye_mask_output",
        "retarget_zero_at_rest",
        "mouth_tilt_trim",
        "mask_jaw_shrink",
        "mask_forehead_reach",
        "mask_chin_reach",
        "chin_follow",
        "edge_match",
        "background_guard",
        "rotation_pivot",
        "pivot_z",
        # Day 17: the hair switch and its skin lever, the hairline's
        # clearance, the steep-pose mode, the blink's counters.
        "hair",
        "hair_skin",
        "hairline_clearance",
        "steep_pose",
        "blink_brow_hold",
        "blink_mouth_hold",
        "blink_mouth_lift",
        # Day 21 (2026-10-14): the skin's tone matched before the plane;
        # the forehead's moderate stretch and the roots' margin.
        "skin_match",
        "forehead_stretch_max",
        "hair_root_margin",
        # Day 22 (2026-10-15): the face's width fitted to the person's.
        "face_width_fit",
        "driver_face_width",
        # Day 23 (2026-10-16): the face's height fitted -- the brows to the person's.
        "face_height_fit",
        "driver_face_height",
        # Day 22 evening: the render's profile side fitted to his outline at a turn.
        "outline_fit",
        # Day 23: the avatar's skin carried past the drawn face's profile edge at a turn.
        "profile_fill_px",
    }
)

# The anchor as a file (day 15, 2026-09-11): everything _learn_anchor freezes,
# so a "That looked wrong" capture can carry the session's own anchor and a
# replay of it drives the portrait exactly as the session did -- until now a
# replay had to learn an anchor from the capture's own frames, and a capture
# taken with the head tilted (his of 08:10, pitch -5..-10 on all 300 frames)
# could learn none at all. The tensors are what _drive reads as the anchor;
# the scalars are the rest values the gates, the crop and the eye/brow/chin
# drives measure against. None is written as NaN and read back as None.
_ANCHOR_TENSORS = ("exp_d0", "t_d0", "scale_d0", "R_d0", "pose_d0")
_ANCHOR_SCALARS = (
    "rest_ear",
    "rest_mar",
    "rest_mouth_width",
    "rest_brow",
    "rest_chin",
    "face_height_rest",
    "z_rest",
    "centre_offset",
    "rest_pitch",
    "rest_yaw",
    # Day 22: what the forehead and the width were settled from, so a
    # capture's replay settles the same (see seed_anchor).
    "settle_hairline",
    "settle_width",
    "settle_height",
)

# What counts as a frame worth learning the anchor from (day 5, 2026-09-05).
# It used to be "the first 15 frames after load, whatever they hold" -- and
# an anchor learned while the owner was turned -12 deg away, reaching for the
# mouse, reproduced his own live report exactly: the avatar's nose sat +29 px
# from his (his own report: +27), the eyes read 26% wider than his at rest,
# and the mouth sat open a crack it should not have (MAR 0.05-0.16 against
# his 0.002). A gated anchor waits for frames that are actually frontal,
# open-eyed and closed-mouthed instead of freezing on whatever arrived first.
#
# Day 16 evening (2026-09-12), the first other face: the bounds were
# +-4/+-4/+-3 degrees around zero, which took the camera for the person's
# own eye line. The friend's head read pitch -11 sitting still in front of
# this camera (the tracker reads the owner's at -3), so every one of his
# still frames was rejected, the anchor froze only when a brow raise
# happened to pass the gate a minute later, and two calibrations measured
# nothing (recordings/calibration-20260912-185642 and -190114: 0 and 3
# frames passed in 1900 each). The pose bounds are now wide enough for a
# laptop camera below the eye line or a monitor camera above it, and what
# says "sitting still, looking at the camera" is STILLNESS: the spread of
# the pose (max - min on each axis) over the last _ANCHOR_STILL_FRAMES
# frames under _ANCHOR_STILL_SPREAD degrees. Measured on the sessions: the
# owner's still frames spread 0.1-1.3 degrees (p50-p90 over 15 frames),
# the friend's 0.2-1.4; reaching for the mouse (the day-5 case) moves
# several degrees in that half second and is refused as before. The
# anchor's rotation is relative to what it learns (R_d0), so a rest at
# pitch -11 draws the portrait frontal there.
_ANCHOR_MAX_YAW = 10.0
_ANCHOR_MAX_PITCH = 15.0
_ANCHOR_MAX_ROLL = 6.0
_ANCHOR_STILL_FRAMES = 15
_ANCHOR_STILL_SPREAD = 3.0
# A blink reads under 0.15; his own open eyes measured 0.29-0.38.
_ANCHOR_MIN_EAR = 0.20
_ANCHOR_MAX_MAR = 0.03

# The portrait's networks, ``models.yaml`` in LivePortrait's repository.
NUM_KP = 21
_APPEARANCE_PARAMS: dict[str, Any] = {
    "image_channel": 3,
    "block_expansion": 64,
    "num_down_blocks": 2,
    "max_features": 512,
    "reshape_channel": 32,
    "reshape_depth": 16,
    "num_resblocks": 6,
}
_MOTION_PARAMS: dict[str, Any] = {"num_kp": NUM_KP, "backbone": "convnextv2_tiny"}
_WARPING_PARAMS: dict[str, Any] = {
    "num_kp": NUM_KP,
    "block_expansion": 64,
    "max_features": 512,
    "num_down_blocks": 2,
    "reshape_channel": 32,
    "estimate_occlusion_map": True,
    "dense_motion_params": {
        "block_expansion": 32,
        "max_features": 1024,
        "num_blocks": 5,
        "reshape_depth": 16,
        "compress": 4,
    },
}
_DECODER_PARAMS: dict[str, Any] = {
    "upscale": 2,
    "block_expansion": 64,
    "max_features": 512,
    "num_down_blocks": 2,
}
_STITCHING_PARAMS: dict[str, Any] = {
    "input_size": 126,
    "hidden_sizes": [128, 128, 64],
    "output_size": 65,
}
# Two more networks in the same checkpoint (day 4, 2026-09-05), not loaded
# before today: eye and lip *retargeting*, LivePortrait's answer to a driver
# whose eyes or mouth do not move as far as the portrait's own. Both take the
# portrait's own keypoints (63 = NUM_KP*3, flattened) plus a handful of close
# ratios and output a keypoint delta of the same shape as stitching's own
# correction (63) -- added the same way, before stitching runs. Input sizes
# read off the checkpoint itself: 66 = 63 + 3 (the portrait's own left and
# right eye ratios, plus one driving target shared by both eyes); 65 = 63 + 2
# (the portrait's own lip ratio, plus one driving target).
_EYE_RETARGET_PARAMS: dict[str, Any] = {
    "input_size": 66,
    "hidden_sizes": [256, 256, 128, 128, 64],
    "output_size": 63,
}
_LIP_RETARGET_PARAMS: dict[str, Any] = {
    "input_size": 65,
    "hidden_sizes": [128, 128, 64],
    "output_size": 63,
}

# MediaPipe's 478-point mesh: the two anchor points the crop is fitted from —
# the midpoint of the eyes and the middle of the mouth — and the 468 face
# points (without the irises) the crop box is sized from.
_LEFT_EYE_CORNERS = (33, 133)
_RIGHT_EYE_CORNERS = (362, 263)
_MOUTH_CORNERS = (61, 291)
FACE_POINTS = 468


# ---------------------------------------------------------------------------
# The crop, as arithmetic: LivePortrait's ``_estimate_similar_transform_from_pts``
# with the anchors read off MediaPipe's mesh instead of a 68- or 203-point one.
# ---------------------------------------------------------------------------


def _anchors(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = points[list(_LEFT_EYE_CORNERS)].mean(axis=0)
    right = points[list(_RIGHT_EYE_CORNERS)].mean(axis=0)
    mouth = points[list(_MOUTH_CORNERS)].mean(axis=0)
    return (left + right) / 2.0, mouth


# The points the crop is *sized and centred* from -- eye corners, the sides
# of the face, the top of the forehead, and the nose bridge and tip. None of
# them move when the mouth opens or the face smiles; the axis direction
# (``_anchors`` above, eyes to mouth) still uses the mouth corners, because
# only the box's own extent was the problem -- see crop_matrix's docstring.
_RIGID_POINTS = (33, 133, 362, 263, 234, 454, 10, 1, 168, 6, 197, 195, 5, 4)


def _face_axes(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The face's own horizontal and vertical unit vectors, and the roll angle
    the vertical is off the image's, from the eyes-to-mouth axis."""
    eyes, mouth = _anchors(pts)
    down = mouth - eyes
    length = float(np.linalg.norm(down))
    uy = np.array([0.0, 1.0], dtype=np.float32) if length <= 1e-3 else down / length
    ux = np.array([uy[1], -uy[0]], dtype=np.float32)
    angle = math.acos(max(-1.0, min(1.0, float(ux[0]))))
    if ux[1] < 0.0:
        angle = -angle
    return ux, uy, angle


def _aligned_box(
    subset: np.ndarray, ux: np.ndarray, uy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``subset``'s centroid, and its bounding box's low/high corners measured
    along the face's own axes."""
    centre = subset.mean(axis=0)
    rotated = (subset - centre) @ np.array([ux, uy], dtype=np.float32).T
    return centre, rotated.min(axis=0), rotated.max(axis=0)


def _face_height(points: np.ndarray) -> float:
    """The whole mesh's height along the face's own vertical: forehead top to
    chin, in pixels -- what :func:`crop_matrix` sizes the crop from when it
    is not told a height, and the measure LivePortrait's own framing
    convention (face height x ``CROP_SCALE``) was tuned against.

    Read per frame it moves with the chin, so an open mouth zooms the crop
    (day 5's complaint); the engine reads it once, at rest, and afterwards
    scales it by the head pose's own distance -- see
    :meth:`LivePortraitEngine._driver_face_height`.
    """
    pts = np.asarray(points, dtype=np.float32)[:, :2]
    ux, uy, _ = _face_axes(pts)
    _, low, high = _aligned_box(pts, ux, uy)
    return float(high[1] - low[1])


def crop_matrix(
    points: np.ndarray,
    *,
    dsize: int = CROP_SIZE,
    scale: float = CROP_SCALE,
    vy_ratio: float = CROP_VY_RATIO,
    face_height: float | None = None,
    centre_offset: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The similarity that cuts a face crop, and its inverse.

    ``points`` are the mesh's face points in pixels, ``(N, 2)`` or ``(N, 3)``.
    Returns ``(to_crop, to_frame)``, both ``2x3`` float32: the first maps frame
    pixels into the ``dsize`` square, the second maps the square back.

    The square is axis-aligned to the face — the axis from the eyes down to the
    mouth is its vertical — so roll is taken out of the crop and put back by the
    inverse, which is why a driver's roll survives the round trip while the
    network never sees it.

    **Centred on the rigid points, sized by a face height given from outside
    (day 6, 2026-09-05).** The box used to be sized and centred from every
    point in the mesh, and an open mouth or a smile moves its own corners
    outward and downward, growing the box the network reads its own size
    from: the avatar's apparent size against the owner's own varied
    1.01..1.25 over a clip that mostly just talks, worst exactly when his
    mouth opened widest. The centre now comes from points expression cannot
    move (:data:`_RIGID_POINTS`: eye corners, face sides, forehead, nose).
    The size is ``face_height`` (forehead top to chin, pixels) times
    ``scale``; ``None`` measures it from these very points
    (:func:`_face_height`) -- right for a still portrait at load, and the old
    per-frame behaviour, zoom included, on a moving driver. The engine passes
    the driver's *resting* height scaled by his current distance instead
    (:meth:`LivePortraitEngine._driver_face_height`), which nothing on the
    face can move.

    Two measures that looked stable and were not, on the owner's own clip,
    each tried for a day: the rigid points' *width* (cheek to cheek) shrinks
    with cos(yaw), so a turn zoomed the crop in and the avatar lagged the
    turn by 34 px instead of 19; their *height* (forehead top to nose tip)
    moves with the nose tip's own protrusion under pitch, -13% at +14° of
    pitch, and an open mouth -- which tips the head back -- came out as a
    zoom and a lost mouth (mar gain 0.29).
    """
    cx, cy, angle, size = _crop_geometry(
        points,
        scale=scale,
        vy_ratio=vy_ratio,
        face_height=face_height,
        centre_offset=centre_offset,
    )
    return _similarity(cx, cy, angle, size, dsize=dsize)


def _crop_geometry(
    points: np.ndarray,
    *,
    scale: float = CROP_SCALE,
    vy_ratio: float = CROP_VY_RATIO,
    face_height: float | None = None,
    centre_offset: float | None = None,
    centre_weight: float = 1.0,
) -> tuple[float, float, float, float]:
    """The crop square's centre ``(cx, cy)`` in frame pixels, its roll
    ``angle`` in radians and its ``size`` in frame pixels -- the geometry
    half of :func:`crop_matrix`, split out (day 9, later) so the engine can
    smooth the centre and size between frames before building the matrices
    (see :meth:`LivePortraitEngine._smooth_crop`).

    ``centre_offset`` (day 11, 2026-09-06) moves the *vertical* centre off
    the rigid points' bounding box onto the eye line -- the four eye
    corners' mean -- plus this fixed offset in units of ``size``. The
    rigid box's own vertical centre is the midpoint between the forehead
    point and the lowest nose point, and seven of the fourteen rigid points
    are the nose: measured on the owner's own brow raise (natural.mkv
    1552-1576, and a live capture the same day), his eye corners moved
    12 px and 22 px while the nose points moved 7 px and 40 px -- so the
    box centre followed the nose, not the eyes, and the avatar's eyes
    (rendered where the crop puts them) ended 3.5-8 px off his. Under
    pitch the nose tip's own protrusion drags the box the same way. The
    eye corners are the one set nothing on the face moves.

    The offset is learned at the driver's own anchor (the median of
    :func:`_centre_offset` over the accepted frames, frozen with the
    anchor -- see :meth:`LivePortraitEngine._learn_anchor`), so that at
    rest the two rules give the same crop and the anchor's framing is
    unchanged; only what the crop follows *away* from rest changes. A
    first version took the offset from the portrait instead: on him that
    shifted the crop by a constant 5-7 px (his proportions are not the
    portrait's), the driven face then sat that far from the portrait's own
    keypoints, and blinks came out shallower (driving.mkv blink gain 0.86
    -> 0.72, the shut eyes 0.06 -> 0.09) -- the warp's framing sensitivity
    again. ``None`` is the rigid box, as before.

    ``centre_weight`` blends the two rules (0 rigid, 1 eye line). Even with
    the offset learned at his own anchor the eye-line rule cost blinks
    (0.85 -> 0.77) and tripled the toothy smile's width response on
    driving.mkv -- the corners move with the lids and a smile, and the
    warp is sensitive to a shift of a few pixels in its framing -- while
    its whole benefit is under *pitch*, where the nose tip drags the rigid
    box and the eye line does not. So the engine fades the weight in with
    his own pitch away from the anchor's (:data:`_CROP_CENTRE_PITCH`):
    rigid while he blinks, smiles and talks level, eye line while he nods.
    """
    pts = np.asarray(points, dtype=np.float32)[:, :2]
    ux, uy, angle = _face_axes(pts)
    height = _face_height(pts) if face_height is None else face_height

    centre0, low, high = _aligned_box(pts[list(_RIGID_POINTS)], ux, uy)
    centre1 = (low + high) / 2.0
    size = height * scale
    cy_face = float(centre1[1])
    if centre_offset is not None and centre_weight > 0.0:
        eye = pts[list(_LEFT_EYE_CORNERS + _RIGHT_EYE_CORNERS)].mean(axis=0)
        cy_eyeline = float((eye - centre0) @ uy) + centre_offset * size
        cy_face = (1.0 - centre_weight) * cy_face + centre_weight * cy_eyeline
    centre = centre0 + ux * centre1[0] + uy * cy_face + uy * (vy_ratio * size)
    return float(centre[0]), float(centre[1]), float(angle), float(size)


def _centre_offset(
    points: np.ndarray, *, scale: float = CROP_SCALE, vy_ratio: float = CROP_VY_RATIO
) -> float:
    """The vertical gap between the rigid box's centre and the eye line on
    these ``points`` (the driver's, on an anchor frame), in units of the
    crop size -- what :func:`_crop_geometry` adds back when it centres on
    the eye line, so that on the frames it was measured on the two rules
    give the same crop. See ``centre_offset`` there."""
    pts = np.asarray(points, dtype=np.float32)[:, :2]
    ux, uy, _ = _face_axes(pts)
    size = _face_height(pts) * scale
    centre0, low, high = _aligned_box(pts[list(_RIGID_POINTS)], ux, uy)
    eye = pts[list(_LEFT_EYE_CORNERS + _RIGHT_EYE_CORNERS)].mean(axis=0)
    return (float((low + high)[1] / 2.0) - float((eye - centre0) @ uy)) / size


def _similarity(
    cx: float, cy: float, angle: float, size: float, *, dsize: int = CROP_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """The matrix half of :func:`crop_matrix`: ``(to_crop, to_frame)`` for a
    square of ``size`` frame pixels centred on ``(cx, cy)``, rolled by
    ``angle``, mapped onto ``dsize`` pixels."""
    s = dsize / size
    c, sn = math.cos(angle), math.sin(angle)
    target = dsize / 2.0
    to_crop = np.array(
        [
            [s * c, s * sn, target - s * (c * cx + sn * cy)],
            [-s * sn, s * c, target - s * (-sn * cx + c * cy)],
        ],
        dtype=np.float32,
    )
    full = np.vstack([to_crop, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
    to_frame = np.linalg.inv(full)[:2].astype(np.float32)
    return to_crop, to_frame


# ---------------------------------------------------------------------------
# Smoothing the crop between frames (day 9, later, 2026-09-06): the "bounce"
# ---------------------------------------------------------------------------
#
# His live verdict: "the face bounces on the slightest sideways shift of my
# head, or while I talk". Measured on driving.mkv, frame to frame: while he
# talks the crop's *size* jumps 1.5 px at the median and 6.3 px at p90 (of
# ~700), its centre 0.6/2.8 px -- MediaPipe's head-pose distance z, which
# sizes the crop, reads 0.02..0.08 units differently from one frame to the
# next as his mouth moves; his own inter-ocular distance moves 0.18/0.79
# px. The extractor's own scale adds 0.1%/1.1% (p50/p90) on top, even
# sitting still. Its angles are not the problem (0.04/0.14 degrees a frame).
#
# So the crop's centre and size go through a One Euro filter (Casiez et
# al., 2012): a low-pass whose cutoff rises with speed -- nearly still, the
# cutoff is ``crop_smoothing`` Hz and the jitter is taken out; moving fast,
# the cutoff rises by ``crop_smoothing_beta`` per px/s and the face is not
# left behind. The roll angle is not filtered (0.01-0.04 degrees a frame).
# Both options live on the avatar like every other; 0.0 Hz turns it off.
_DEFAULT_CROP_SMOOTHING_HZ = 1.0
_DEFAULT_CROP_SMOOTHING_BETA = 0.02
_DEFAULT_FRAME_DT = 1.0 / 30.0


def _one_euro_alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class _OneEuro:
    """A scalar One Euro filter; ``reset()`` forgets its state so the next
    value passes through unchanged (used when the face is lost and found)."""

    __slots__ = ("beta", "d_cutoff", "dx", "min_cutoff", "x")

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        if min_cutoff <= 0.0 or beta < 0.0:
            raise ValueError(
                f"one euro filter needs min_cutoff > 0, beta >= 0, got {min_cutoff}, {beta}"
            )
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x: float | None = None
        self.dx = 0.0

    def reset(self) -> None:
        self.x = None
        self.dx = 0.0

    def __call__(self, value: float, dt: float) -> float:
        if self.x is None:
            self.x = value
            self.dx = 0.0
            return value
        a_d = _one_euro_alpha(self.d_cutoff, dt)
        raw_dx = (value - self.x) / dt
        self.dx = a_d * raw_dx + (1.0 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        a = _one_euro_alpha(cutoff, dt)
        self.x = a * value + (1.0 - a) * self.x
        return self.x


# What fills the driver crop where it reaches past the camera frame (day 9,
# night, 2026-09-06). His two "that looked wrong" captures at 23:21: sitting
# twice as close to the camera as in any clip (inter-ocular 210-220 px
# against the clips' ~110), his face was 425-475 px tall, so the crop --
# 2.2 face heights, shifted up an eighth -- was ~950 px square on a 720 px
# frame with its top ~260 px above the frame: **28% of what the motion
# extractor saw was a black band** where the forehead, hair and room should
# be. That looked like the cause -- until the clips were checked: he sits
# at the same distance in every clip (inter-ocular 196-244 px), so every
# number in the ROADMAP was measured *with* that band, and the extractor
# coped. Tried anyway on a 1.9x zoomed copy of natural.mkv (inter-ocular
# 408 px, 69% of the crop outside the frame): replicated edges made the
# extractor far *worse* than black (talk mar gain 1.12 -> 0.31, frame 1380
# mouth 0.185 -> 0.017) -- smeared edge pixels read as a face where black
# reads as nothing. So the default stays cv2.warpAffine's own black;
# ``crop_border`` ("black" | "replicate" | "reflect") is kept for
# measuring, not for use. The honest limit: at 720p the network sees the
# whole context it was trained on only when the face is under ~330 px tall
# (inter-ocular under ~160 px); at his distance it sees about 72% of it.
_CROP_BORDERS = {
    "replicate": "BORDER_REPLICATE",
    "reflect": "BORDER_REFLECT_101",
    "black": "BORDER_CONSTANT",
}


def _crop_border_mode(options: dict[str, Any], cv2: Any) -> int:
    name = str(options.get("crop_border", "black"))
    if name not in _CROP_BORDERS:
        raise ValueError(f"crop_border must be one of {sorted(_CROP_BORDERS)}, got {name!r}")
    return int(getattr(cv2, _CROP_BORDERS[name]))


_CropFilters = tuple["_OneEuro | None", "_OneEuro | None", "_OneEuro | None"]


_CROP_CENTRES = ("rigid", "eyeline")
# Degrees of pitch away from the anchor's own over which the crop centre
# fades from the rigid box (at or under the first) to the eye line (at or
# over the second) -- see _crop_geometry's centre_weight.
_CROP_CENTRE_PITCH = (3.0, 8.0)


def _crop_centre(options: dict[str, Any]) -> str:
    """What the crop's vertical centre follows, from ``options["crop_centre"]``:
    ``"rigid"`` (default) the rigid points' bounding box, ``"eyeline"`` the
    four eye corners' mean plus the portrait's own fixed offset -- see
    :func:`_crop_geometry`'s ``centre_offset`` for the measurement behind it."""
    name = str(options.get("crop_centre", "rigid"))
    if name not in _CROP_CENTRES:
        raise ValueError(f"crop_centre must be one of {_CROP_CENTRES}, got {name!r}")
    return name


def _crop_filters(options: dict[str, Any]) -> _CropFilters | None:
    """Filters for the crop's centre x, centre y and size -- ``None`` in a
    slot means that quantity passes through; ``None`` altogether when
    nothing is smoothed. The size is filtered at ``crop_smoothing`` Hz
    (default 1.0); the centre at ``crop_centre_smoothing`` Hz, **default
    0.0, off**: measured with the centre filtered at 1 Hz too
    (2026-09-06), the size flicker went away but the avatar's position
    against his own jittered *more* on the still segments (natural.mkv
    camera-still dx p90 0.63 -> 0.98 px, screen-still 1.45 -> 1.75) -- the
    filter's speed estimate is itself noisy, so its cutoff hunts and the
    centre alternates between following and lagging. ``crop_smoothing_beta``
    serves both."""
    size_hz = float(options.get("crop_smoothing", _DEFAULT_CROP_SMOOTHING_HZ))
    centre_hz = float(options.get("crop_centre_smoothing", 0.0))
    beta = float(options.get("crop_smoothing_beta", _DEFAULT_CROP_SMOOTHING_BETA))
    if size_hz < 0.0 or centre_hz < 0.0 or beta < 0.0:
        raise ValueError(
            "crop_smoothing/crop_centre_smoothing/crop_smoothing_beta must be >= 0, "
            f"got {size_hz}, {centre_hz}, {beta}"
        )
    if size_hz == 0.0 and centre_hz == 0.0:
        return None
    centre_x = _OneEuro(centre_hz, beta) if centre_hz > 0.0 else None
    centre_y = _OneEuro(centre_hz, beta) if centre_hz > 0.0 else None
    size = _OneEuro(size_hz, beta) if size_hz > 0.0 else None
    return centre_x, centre_y, size


def _frame_dt(timestamp: float | None, previous: float | None) -> float:
    """The time since the last frame, from capture timestamps; the nominal
    frame time when there is no last frame or the gap is not credible."""
    if timestamp is None or previous is None:
        return _DEFAULT_FRAME_DT
    dt = timestamp - previous
    return dt if 0.0 < dt <= 0.5 else _DEFAULT_FRAME_DT


def _transformed_box(
    to_frame: np.ndarray,
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Where a crop-space rectangle lands in the frame, clipped to it."""
    x0c, y0c, x1c, y1c = box
    corners = np.array([[x0c, y0c], [x1c, y0c], [x1c, y1c], [x0c, y1c]], dtype=np.float32)
    moved = corners @ to_frame[:, :2].T + to_frame[:, 2]
    x0 = max(int(np.floor(moved[:, 0].min())), 0)
    y0 = max(int(np.floor(moved[:, 1].min())), 0)
    x1 = min(int(np.ceil(moved[:, 0].max())) + 1, width)
    y1 = min(int(np.ceil(moved[:, 1].max())) + 1, height)
    return x0, y0, x1, y1


def _crop_box(to_frame: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    """Where the whole crop square lands in the frame, clipped to it."""
    return _transformed_box(to_frame, (0, 0, CROP_SIZE, CROP_SIZE), width, height)


# ---------------------------------------------------------------------------
# The eye/lip close ratios the retargeting networks read -- LivePortrait's own
# measure, on MediaPipe's points instead of its own landmark model. A ratio
# rather than a raw gap so it does not depend on how large the face is in the
# frame: near 0 when shut, roughly 0.2-0.3 open on the owner's own captures.
# Raw indices restated rather than imported, for the same reason as the face
# oval's above: no internals shared across engines or across modules of this
# one that do not need to be.
# ---------------------------------------------------------------------------
_L_EYE_OUTER, _L_EYE_INNER, _L_EYE_UPPER, _L_EYE_LOWER = 33, 133, 159, 145
_R_EYE_INNER, _R_EYE_OUTER, _R_EYE_UPPER, _R_EYE_LOWER = 362, 263, 386, 374
_LIP_UPPER_INNER, _LIP_LOWER_INNER, _MOUTH_L, _MOUTH_R = 13, 14, 61, 291


def _eye_ratios(points: np.ndarray) -> tuple[float, float]:
    """Left, right eyelid gap over eye width, from MediaPipe's points."""
    left = float(np.linalg.norm(points[_L_EYE_UPPER, :2] - points[_L_EYE_LOWER, :2])) / (
        float(np.linalg.norm(points[_L_EYE_INNER, :2] - points[_L_EYE_OUTER, :2])) or 1e-6
    )
    right = float(np.linalg.norm(points[_R_EYE_UPPER, :2] - points[_R_EYE_LOWER, :2])) / (
        float(np.linalg.norm(points[_R_EYE_OUTER, :2] - points[_R_EYE_INNER, :2])) or 1e-6
    )
    return left, right


def _lip_ratio(points: np.ndarray) -> float:
    """Inner-lip gap over mouth width, from MediaPipe's points."""
    return float(np.linalg.norm(points[_LIP_UPPER_INNER, :2] - points[_LIP_LOWER_INNER, :2])) / (
        float(np.linalg.norm(points[_MOUTH_R, :2] - points[_MOUTH_L, :2])) or 1e-6
    )


# The two measures the day-9 expression gates read (see _mouth_gate/_brow_gate
# and the GATES section below), each a ratio of two distances on the same face
# so it does not depend on how large the face is in the frame:
#
# * mouth width -- the mouth corners' distance over the *jaw* width at the
#   same height (points 58/288, on the face contour beside the mouth), not
#   over the inter-ocular distance measure_follow.py reports. Measured on the
#   owner's own clips (2026-09-06): mouth width over inter-ocular grows
#   +0.004 per degree of pitch (r 0.88 on natural.mkv -- the mouth sits
#   lower on the face and comes toward the camera as the head lifts, the
#   eyes do not), which at the +10 degrees of an ordinary "look up" is the
#   size of a closed smile. Over the jaw width at its own height the two
#   distances move together and the ratio holds.
# * brow -- the inner brow points (105/334) above the *eye corners'*
#   midpoint, over the midpoint's own height above the chin (152). The
#   corners, not the upper lid: the lid-to-brow gap measure_follow.py
#   reports rises by a third of itself on every blink (r -0.94 against the
#   eye ratio on the blink segment), which would open a brow gate on each
#   blink; corner-based, 0.02 over the same blinks. Over the chin distance,
#   not inter-ocular: both are vertical on the face, so a lifted head
#   shortens them together -- +0.3% of rest per degree of pitch against
#   +1.2..1.9% over inter-ocular (r 0.96 on natural.mkv), and 4% at the
#   widest of a turn against 49%. A raise still reads +34..38% at its
#   height on both clips. The residual pose dependence is what
#   _brow_gate's fades are for.
_BROW_L, _BROW_R = 105, 334
_JAW_L, _JAW_R = 58, 288
_CHIN = 152


def _eye_corner_midpoint(points: np.ndarray) -> np.ndarray:
    return points[[_L_EYE_OUTER, _L_EYE_INNER, _R_EYE_INNER, _R_EYE_OUTER], :2].mean(axis=0)


def _mouth_width_ratio(points: np.ndarray) -> float:
    """Mouth corners' distance over the jaw width at mouth height."""
    return float(np.linalg.norm(points[_MOUTH_R, :2] - points[_MOUTH_L, :2])) / (
        float(np.linalg.norm(points[_JAW_R, :2] - points[_JAW_L, :2])) or 1e-6
    )


def _mouth_open_ratio(points: np.ndarray) -> float:
    """Inner-lip gap over the jaw width at mouth height -- the opening the
    gates read (day 12).

    Not :func:`_lip_ratio`, whose denominator is the mouth's own width: that
    is LivePortrait's lip-retargeting convention and stays for the network
    and the anchor, but as an opening *gate* it punished a smile -- the
    wider the mouth, the larger the gap had to be to pass. A 4-px gap read
    0.067 at a 60-px mouth and 0.033 at a 120-px grin, so a wide closed-lip
    grin with the teeth just showing never opened the lips (his 2026-09-06
    night "smile_without_teeth"). The jaw width does not move with the
    smile; it is what :func:`_mouth_width_ratio` already divides by. At
    rest it is 2.5x the mouth width on his clips and all nine captures
    (mouth/jaw 0.39-0.43), which is where the gate thresholds' factor of
    0.4 comes from.
    """
    return float(np.linalg.norm(points[_LIP_UPPER_INNER, :2] - points[_LIP_LOWER_INNER, :2])) / (
        float(np.linalg.norm(points[_JAW_R, :2] - points[_JAW_L, :2])) or 1e-6
    )


def _brow_ratio(points: np.ndarray) -> float:
    """Inner brows' height above the eye corners' midpoint, over that
    midpoint's own height above the chin."""
    mid_y = float(_eye_corner_midpoint(points)[1])
    lift = ((mid_y - float(points[_BROW_L, 1])) + (mid_y - float(points[_BROW_R, 1]))) / 2.0
    return lift / ((float(points[_CHIN, 1]) - mid_y) or 1e-6)


def _anchor_frame_ok(pose: HeadPose | None, ear: float, mar: float, *, still: bool = True) -> bool:
    """Whether this frame is worth learning the driver's rest from.

    Facing the camera, holding still, open-eyed, closed-mouthed -- see the
    constants' own comment for why. ``pose is None`` (no head pose this
    frame) is rejected outright: an anchor needs to know it was actually
    looking at the camera, not assume it. ``still`` is :func:`_pose_still`'s
    word on the last half second (day 16): a pose that is not moving, at
    whatever angle this camera sees a resting head from.
    """
    if pose is None or not still:
        return False
    return (
        abs(pose.yaw) <= _ANCHOR_MAX_YAW
        and abs(pose.pitch) <= _ANCHOR_MAX_PITCH
        and abs(pose.roll) <= _ANCHOR_MAX_ROLL
        and ear >= _ANCHOR_MIN_EAR
        and mar <= _ANCHOR_MAX_MAR
    )


def _pose_still(recent: deque[tuple[float, float, float]], pose: HeadPose | None) -> bool:
    """Record this frame's pose in ``recent`` (a window of
    :data:`_ANCHOR_STILL_FRAMES`) and say whether the head has held still
    over it: the window full, and yaw, pitch and roll each spread (max -
    min) no more than :data:`_ANCHOR_STILL_SPREAD` degrees. A frame without
    a pose empties the window -- nothing is known about the half second
    around a lost face."""
    if pose is None:
        recent.clear()
        return False
    recent.append((float(pose.yaw), float(pose.pitch), float(pose.roll)))
    if len(recent) < recent.maxlen:  # type: ignore[operator]
        return False
    for axis in range(3):
        values = [p[axis] for p in recent]
        if max(values) - min(values) > _ANCHOR_STILL_SPREAD:
            return False
    return True


# How far past his own rest the eye target may ask the avatar to open. 1.0:
# the portrait's own open eyes are the widest the avatar goes. It was 1.2,
# and the owner's live verdict on that (2026-09-05, day 6) was "it pulls the
# eyes and raises the brows for no reason" -- the eye network's 63 outputs
# are not confined to the lids, and asking it for wider-than-the-portrait
# eyes lifts the brow keypoints with them. Wider than rest now asks for the
# portrait's own rest and nothing more; narrower still follows him.
#
# Day 9 (2026-09-06) made it the ``eye_headroom`` option, default still 1.0:
# the day-6 verdict was taken with the eye network's whole 63-output delta
# applied and with the network's own nonzero output *at rest* in it (see
# _retarget_rest); with the delta confined to EYE_KEYPOINTS (day 8) and
# zero at rest, "wider than rest" is worth measuring again.
_EYE_TARGET_HEADROOM = 1.0


def _eye_headroom(options: dict[str, Any]) -> float:
    """The ``eye_headroom`` option, validated: at least 1.0 (the portrait's
    own open eyes), a ratio of his rest the target may reach up to."""
    h = float(options.get("eye_headroom", _EYE_TARGET_HEADROOM))
    if not (1.0 <= h <= 2.0):
        raise ValueError(f"eye_headroom must be in [1, 2], got {h}")
    return h


def _eye_scale(options: dict[str, Any]) -> float:
    """The ``eye_scale`` option, validated: what the avatar's eyes are asked
    to be at *his* rest, as a fraction of the portrait's own open eyes --
    1.0 (default) is the portrait's; 0.95 asks 5% narrower. Day 9, later:
    his verdict "the eyes are slightly too big" -- the portrait's own rest
    reads 0.334 against his 0.314..0.320. Scales the whole target, so a
    blink still reaches shut."""
    s = float(options.get("eye_scale", 1.0))
    if not (0.5 <= s <= 1.2):
        raise ValueError(f"eye_scale must be in [0.5, 1.2], got {s}")
    return s


def _eye_target(
    ear: float,
    rest_ear: float | None,
    c_s_mean: float,
    headroom: float = _EYE_TARGET_HEADROOM,
    scale: float = 1.0,
) -> float:
    """The eye retargeting network's driving target, relative to the
    driver's own rest rather than his absolute openness (day 5, 2026-09-05).

    His eyes measured narrower than the portrait's at rest (his 0.30 against
    the portrait's own 0.334): a target built from his raw ratio asked the
    network to narrow the avatar's eyes for the mere fact of him sitting
    still, measured as the avatar's eyes reading 27% narrower than his over a
    whole clip. At his own rest this returns exactly ``c_s_mean`` -- no
    change asked -- and moves proportionally as his eyes open wider or
    narrower than that, capped at :data:`_EYE_TARGET_HEADROOM` above it.

    ``rest_ear`` is ``None`` until the anchor freezes (see
    :func:`_anchor_frame_ok`); until then this is simply the portrait's own
    ratio, asking for no change, the same as the anchor's own resting frame.
    """
    if rest_ear is None:
        return c_s_mean
    return c_s_mean * scale * min(ear / max(rest_ear, 1e-3), headroom)


# Day 12, his call pending ("closed_eyes", the 2026-09-06 night screenshot:
# narrowed lids, level head, faint smile). His lids drop when he looks down
# at the screen -- corr(his EAR, his iris height) -0.85 / -0.91 on the two
# clips; 33 of natural.mkv's 35 narrowed talking frames have the iris more
# than 0.05 of eye width below rest -- and the avatar's irises cannot
# follow (vertical gaze has no lever, see the GAZE section), so it shows
# the narrowing alone: eyes "closed". ``gaze_down_relief`` (default off,
# the narrowed look as now): while his iris sits below his rest by more
# than _GAZE_DOWN_RELIEF_MIN and the lids came down slowly, the eye target
# is built from his rest ratio instead, and the avatar keeps its eyes open
# looking straight. A blink still closes: the eyes under half of rest, or
# a fall of a quarter of rest within the last _BLINK_WINDOW frames, is a
# blink's speed (a blink lands in 2-3 frames), not a glance down.
_GAZE_DOWN_RELIEF_MIN = 0.03
_BLINK_EAR_FRACTION = 0.5
_BLINK_DROP_FRACTION = 0.25
_BLINK_WINDOW = 3


def _gaze_down_relief(options: dict[str, Any]) -> bool:
    """``gaze_down_relief``, default False -- see :func:`_relieved_ear`."""
    return bool(options.get("gaze_down_relief", False))


def _relieved_ear(
    ear: float,
    rest_ear: float | None,
    gaze: tuple[float, float] | None,
    rest_gaze: tuple[float, float] | None,
    recent: Sequence[float],
) -> float:
    """The eye ratio the target is built from: his own, or his rest while he
    reads the screen with his lids lowered (``gaze_down_relief``).

    ``recent`` is his ratio on the frames just before this one (up to
    :data:`_BLINK_WINDOW`). Untouched -- his own reading -- until the rest
    is known, while his iris is not below rest by :data:`_GAZE_DOWN_RELIEF_MIN`,
    while his eyes are under :data:`_BLINK_EAR_FRACTION` of rest (shut, or
    shutting), or when they fell by :data:`_BLINK_DROP_FRACTION` of rest
    within the window (a blink's speed). Otherwise no narrower than rest.
    """
    if rest_ear is None or gaze is None or rest_gaze is None:
        return ear
    if gaze[1] - rest_gaze[1] <= _GAZE_DOWN_RELIEF_MIN:
        return ear
    if ear < _BLINK_EAR_FRACTION * rest_ear:
        return ear
    if recent and max(recent) - ear > _BLINK_DROP_FRACTION * rest_ear:
        return ear
    return max(ear, rest_ear)


def _lip_target(mar: float, c_s_lip: float) -> float:
    """The lip retargeting network's driving target, never past the
    portrait's own shut lips (day 5, 2026-09-05).

    Measured: with the driver's raw ratio as the target, a shut mouth pressed
    the avatar's own lips to 38-44% of the owner's own thickness, because the
    network was asked for a ratio *tighter* than the portrait's own closed
    lips and had nowhere to put that but the lips themselves. Floored at the
    portrait's own ratio, a shut mouth asks for no more closing than the
    portrait already shows.
    """
    return max(mar, c_s_lip)


# ---------------------------------------------------------------------------
# The paste mask: the portrait's own face oval, in the crop LivePortrait
# always draws into, rather than the whole 512 square.
#
# LivePortrait ships a fixed mask template -- a soft ellipse baked for its own
# framing convention -- and it takes no account of *this* portrait's actual
# proportions or where its own hairline and jaw sit. The warp engine solved
# exactly this problem for its own composite: the face oval, pulled short of
# the hairline and left at the jaw, so two hairlines never have to agree and
# the join happens over skin. The same shape is right here for the same
# reason, so the same fractions are used -- see :func:`_oval_mask`.
#
# Raw MediaPipe landmark indices, not imported from the warp engine: the two
# engines are meant to share no internals, only the registry contract
# (AVATAR_SYSTEM.md -- an avatar names an *engine id*, never an import path).
# This is a fact about MediaPipe's 478-point topology, not about either
# engine, so it is stated again here rather than reached for across the file.
_FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
)  # fmt: skip
_BROWS = (46, 53, 52, 65, 55, 107, 276, 283, 282, 295, 285, 336)

# Same values as the warp engine's ``_MASK_SHRINK`` / ``_MASK_FEATHER`` /
# ``_FOREHEAD_REACH``: each is a fraction of a distance measured inside the
# crop itself (face width, brow-to-hairline), so the same fraction describes
# the same shape whatever the crop's pixel size is. The warp engine's jaw
# extension is not carried over -- it shipped at ``_JAW_REACH_SCALE = 1.0``,
# which the owner's own 2026-08-15 verdict made a no-op (he preferred his own
# jaw showing to a stretched avatar chin), so there is nothing to port.
# Day 19 (his screenshot of a held turn, the pale strip beside the eye):
# the oval is the convex hull of the face's rim, and at a turn the rim's
# near side projects INSIDE the face -- the hull runs straight from the
# temple to the chin and leaves the nose, the lips and the cheek in front
# of the eye outside the paste, where his own skin shows. Measured on his
# turn clip at yaw -47 (band_probe, f452): the hull's near edge 10-25 crop
# px inside the drawn face, cutting the avatar's nose off. The face's
# midline -- brow ridge, nose bridge and tip, the lips, the chin -- goes
# into the hull too: on a frontal face every one of these lies inside the
# rim and the hull is the same polygon to the pixel; at a turn the near
# side follows the profile.
_MIDLINE = (
    151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164,
    0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175,
)  # fmt: skip

# Day 20, his two presses of 04:34 (the far brow's outer end and the eye's
# corner showing at 17 and 28 degrees): the far brow's end projects 6-8 crop
# px outside the rim's hull at 28 degrees (anterior to the temple, it moves
# with the turn more than the rim does) and the paste's edge ran through
# it. So the brows go into the hull with the midline, under the same push;
# on a frontal face every brow point is inside the rim (his 894 frames).
_BROW_LINE = _BROWS

_OVAL_SHRINK = 0.04
_OVAL_FEATHER = 0.07
_OVAL_FOREHEAD_REACH = 0.85
# Day 20 (his two screenshots of 2026-09-18: his black brow's end and eye
# corner showing beside the avatar's at a held turn, his own nose tip
# beside the drawn nose -- "the nose doubled"): with the midline in the
# hull the oval reaches his profile at a turn, but the erosion then pulls
# the edge a radius (6 crop px at his face width) back inside it and the
# feather halves the paste over the next 7, so 10-15 px of his own face
# along the profile stay his (measured on day 19 past 50 degrees). So a
# midline point that lies OUTSIDE the rim's own hull -- which only a turn
# does; on a frontal face every midline point is inside the rim, proven on
# his 894 recorded frames -- is pushed outward along the ray from the
# rim's centre by this many erosion radii before the hull is taken, so the
# erosion lands the edge on the profile. Ramped over the first radius
# outside the rim, so the frame the nose first leaves the rim (about 30
# degrees on his face) moves the edge by nothing, not by a jump.
_PROFILE_PUSH = 1.0

# Day 12: the jaw band ("white line", his 2026-09-06 night screenshots). The
# portrait's beard stops short of its jaw contour; his runs along it. With
# the join 4% inside the contour it lands on the portrait's bare jaw skin,
# pasted over his jaw, which at night sits in shadow -- a pale band from ear
# to chin that a lighting *plane* cannot darken locally. So the driver's
# oval is pulled in along the jaw only: each of its points below the eye
# line moves toward the face centre by up to this fraction of face width,
# nothing at the eye line, the full amount from the nose tip down, so the
# join sits inside the beard and his own jaw shows (his 2026-08-15
# preference, over a stretched avatar chin). The forehead reach above is
# untouched. ``mask_jaw_shrink`` in ``engine_options`` sets it;
# scripts/probes/jaw_band.py measures it.
#
# The default is 0 -- the join at the contour, as before day 12 -- because
# the shrink is a *beard* setting: it only helps where his own beard runs
# the jaw and hides the join. His first look at 0.08 (2026-09-06 evening,
# clean-shaven that morning): the avatar's stubbled chin stops 8% short
# and his bare jaw and chin show under it, a ledge. 0.08 was measured
# with the beard (the day-12 table); it goes in his metadata.json when the
# beard is back, not in every user's default.
_OVAL_JAW_SHRINK = 0.0
_NOSE_TIP = 1

# How far past the mask's own edge its bounding box reaches, in crop pixels --
# enough that the feather's falloff is not itself clipped by the box the
# paste warps into.
_MASK_BOX_MARGIN = 4
# Day 20: the box's room to either side, a fraction of the portrait oval's
# own width -- at a turn the drawn face's far side reaches past the frontal
# oval's edge (his press at +28 degrees: the render's far temple 2 px inside
# the old box's edge; at 45 it is past it) and the paste warps only into
# the box. Costs the paste about that fraction of its area, twice.
_MASK_TURN_ROOM = 0.08

# Low-frequency relighting, exactly the warp engine's own technique
# (``engine_warp._match_colour`` / ``_fit_lighting_plane``), written again
# here rather than imported for the same reason the mask is: no shared
# internals between engines. See that function's docstring for why a plane
# and not a blur, and why the blur is weighted by the mask.
# Day 15 (his tilt captures of 10:57, shaved): the dark line under the
# avatar's chin when he tilts down is the portrait's own jaw shadow, pasted
# where his jaw is lit. At an eighth-scale grid the edge residual could not
# see a line ~8 px tall; at a quarter-scale one it can: on three captures
# the jaw band fell 18.8 / 18.9 / 20.4 -> 11.0 / 12.2 / 13.2 and the forehead
# 3.7 / 4.7 / 3.0 -> 2.8 / 3.8 / 2.3, the line visibly softer in the zooms
# (recordings/day15-jaw-fine-*.png). The cell counts below that mean a
# physical size (the brow-hair dilation, the zone's softening, the fill's
# reach) doubled with it; the edge blur stayed 3 cells -- 12 px now, the
# finer match being the point (5 cells, the old 24 px, gave 13.5 / 14.6 /
# 15.9).
_CORRECTION_DOWNSCALE = 4
_CORRECTION_INERTIA = 0.8

# Day 13: the forehead line -- his one complaint on the fp32 build. Measured
# on his five 02:52-02:55 captures (scripts/probes/jaw_band.py, capture
# mode): the avatar's forehead sits +25..+28 grey levels brighter than his
# in the ten pixels inside the mask's top edge, while the jaw band reads
# -6. A plane cannot fix that: it is one brightness and one gradient per
# channel, and a face that is too pale at the top edge and slightly dark at
# the bottom while right in the middle is *curvature* at the edges -- what
# the warp engine's own measurement kept the plane from ever representing,
# since the same curvature inside the face is a brow shadow or a beard, the
# user's own, which must not be printed onto the avatar. So the plane stays
# the whole-face correction, and what it leaves is added back **at the
# mask's edge alone**: the residual (difference minus plane), blurred at the
# eighth-scale it is measured at and faded from all of it at the mask's
# contour to nothing ``_EDGE_REACH`` of the box's height inside. The join is
# where the two pictures have to agree; the middle of the face keeps the
# avatar's own features. ``edge_match`` in ``engine_options`` sets how much
# of the residual is added, 0 (today's plane alone) to 1.
_EDGE_REACH = 0.2
_EDGE_BLUR = 3

# Day 13, his look of 2026-09-09 02:30 ("my eyebrows show, a dark colour
# above the AI face's brows, worse when I raise them"): the residual is
# meant to be skin against skin at the join, but the fade reaches a fifth
# of the box inside the contour, and his own brows sit 30-40 px inside it
# at rest and 15-20 on a raise -- so their darkness went into the residual
# and came out on the avatar's forehead as a smear. The features that are
# HIS and not the join -- brows, eyes, mouth -- are cut out of the residual
# before the blur (their cells count as unknown, and the blur fills them
# from the skin around); this many eighth-scale cells of dilation around
# their points covers the brow hair above MediaPipe's brow line. What the
# zone gets instead is the skin's own residual, interpolated across it: a
# second, wide blur (_EDGE_FILL_BLUR cells) of the same skin-only residual
# reaches across the zone from the forehead above and the cheek below,
# and the two are blended by the zone softened over _EDGE_EXCLUDE_SOFTEN
# cells. Two things were tried first and are recorded in his screenshots
# of 02:52 and 03:00: a hard zone came out as eighth-scale blocks once the
# field was resized up by eight, and a zone faded to the plane alone came
# out as a dark patch above the avatar's brows -- his forehead that night
# was brighter than the avatar's, the residual around the zone was
# brightening it, and the zone stayed dark.
_EDGE_EXCLUDE_DILATE = 4
_EDGE_EXCLUDE_SOFTEN = 10
_EDGE_FILL_BLUR = 30
# Day 16 evening (the first other face, captures/moment-20260912-200754):
# her brows are thick and arched, and the brow zone was the hull of the
# brows' LOWER row of landmarks (plus one inner upper point) dilated by 4
# cells -- 16 px -- which covered his thin brows and not the upper two
# thirds of hers. What was left of them went into the residual and came out
# as a dark band above the avatar's own brows ("her brow shows"), printed
# from the difference between her brow hair and the portrait's forehead
# skin. The zone now takes both rows of the brow landmarks, the whole brow.
_BROWS_UPPER = (70, 63, 105, 66, 107, 300, 293, 334, 296, 336)
_EDGE_FEATURES = (
    _BROWS[:6] + _BROWS_UPPER[:5] + _LEFT_EYE_CORNERS + (159, 145),
    _BROWS[6:] + _BROWS_UPPER[5:] + _RIGHT_EYE_CORNERS + (386, 374),
    (61, 291, 0, 17, 13, 14),
)
# Same evening, the same capture: the mask's feathered edge lies over her
# hair on both sides (long hair down the cheeks) and the match took that
# hair for the room's light -- a dark band along the avatar's cheeks, and
# a smear at the temples. The plane and the residual are matched on SKIN:
# a cell whose scene is darker than this fraction of the face's own median
# (its core, alpha over 0.9) is hair, a brow core, a lash line or a
# nostril, not skin, and counts as unknown like the feature zone. Measured
# on her three captures and his two (scratch, 2026-09-12 evening): hair
# under the edge reads 0.30-0.35 of the median at p10, her brow cells
# 0.25-0.35, the darkest skin of the core (shadow under the jaw, the eye
# sockets) 0.49-0.64 at p5 -- and what falls under 0.55 in the core is the
# eyes, the nostrils and the hair that the oval covers at the sides.
_EDGE_DARK_RATIO = 0.55


def _feature_exclusion(
    features: np.ndarray, shape: tuple[int, int], dilate: int = _EDGE_EXCLUDE_DILATE
) -> np.ndarray:
    """The cells of an eighth-scale box that are his brows, eyes or mouth
    (1.0) rather than skin (0.0): ``features`` are those points already in
    the box's own cell coordinates, in the groups of :data:`_EDGE_FEATURES`
    -- each group's convex hull filled, then all of them dilated by
    ``dilate`` cells. Hard-edged: 1.0 or 0.0 per cell (see
    :func:`_edge_residual` for how it is used and softened)."""
    import cv2

    zone: np.ndarray = np.zeros(shape, dtype=np.uint8)
    start = 0
    for group in _EDGE_FEATURES:
        pts = features[start : start + len(group)]
        start += len(group)
        cv2.fillConvexPoly(zone, cv2.convexHull(np.round(pts).astype(np.int32)), 255)
    if dilate > 0:
        size = 2 * dilate + 1
        zone = cv2.dilate(zone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return zone.astype(np.float32) / 255.0


def _edge_match(options: dict[str, Any]) -> float:
    """``edge_match`` in ``[0, 1]``, default 1.0: how much of the lighting
    plane's residual is matched at the mask's edge (see :data:`_EDGE_REACH`).
    0 is the plane alone, the correction before day 13."""
    value = float(options.get("edge_match", 1.0))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"edge_match must be in [0, 1], got {value}")
    return value


def _edge_residual(
    difference: np.ndarray,
    coverage: np.ndarray,
    plane: np.ndarray,
    *,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """What the plane leaves, at the mask's edge: the alpha-weighted
    difference less the plane, blurred over :data:`_EDGE_BLUR` cells with
    the blur weighted by the mask (so the room past the contour never
    contributes), then faded from all of it at the contour to nothing
    :data:`_EDGE_REACH` of the box's height inside. Same eighth-scale box
    as the plane, so the whole thing is a few thousand multiplies.

    ``exclude`` (:func:`_feature_exclusion`, 1.0 where his brows, eyes or
    mouth are) takes those cells out of the residual *and* out of the
    blur's weight, so nothing of his features enters the field; what the
    zone gets is the skin's residual interpolated across it -- the same
    skin-only residual blurred wide (:data:`_EDGE_FILL_BLUR` cells),
    blended in by the zone softened over :data:`_EDGE_EXCLUDE_SOFTEN`
    cells, so there is no step at the zone's edge and no patch inside it.
    """
    import cv2

    height = coverage.shape[0]
    residual = difference - plane * coverage[:, :, None]
    weight_in = coverage
    if exclude is not None:
        keep = 1.0 - exclude
        residual = residual * keep[:, :, None]
        weight_in = coverage * keep
    kernel = (_EDGE_BLUR | 1, _EDGE_BLUR | 1)
    smooth = (
        cv2.GaussianBlur(residual, kernel, 0)
        / np.maximum(cv2.GaussianBlur(weight_in, kernel, 0), 1e-3)[:, :, None]
    )
    if exclude is not None:
        wide = (_EDGE_FILL_BLUR | 1, _EDGE_FILL_BLUR | 1)
        filled = (
            cv2.GaussianBlur(residual, wide, 0)
            / np.maximum(cv2.GaussianBlur(weight_in, wide, 0), 1e-3)[:, :, None]
        )
        soften = (_EDGE_EXCLUDE_SOFTEN | 1, _EDGE_EXCLUDE_SOFTEN | 1)
        blend = cv2.GaussianBlur(exclude, soften, 0)[:, :, None]
        smooth = smooth * (1.0 - blend) + filled * blend
    inside = (coverage > 0.5).astype(np.uint8)
    distance = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
    reach = max(float(height) * _EDGE_REACH, 1.0)
    weight = np.clip(1.0 - distance / reach, 0.0, 1.0).astype(np.float32)
    faded: np.ndarray = (smooth * weight[:, :, None]).astype(np.float32)
    return faded


# ---------------------------------------------------------------------------
# The hairline (day 16, blocks B2 and C1)
# ---------------------------------------------------------------------------
#
# Where the hair starts above the brows, as a fraction of the forehead --
# the brow top to the mesh's own top point (10) along the face's own
# vertical -- so 1.0 is the mesh's top and 1.5 is half a forehead above
# it. Read by walking up from the brow top in 2-px steps at three
# columns across the forehead and calling hair the first run of samples
# whose luminance leaves the forehead skin's (the median of the first
# third) by _HAIR_CONTRAST. The importer reads a portrait with it (the
# profile); the engine reads the portrait at load and the driver at the
# anchor when ``mask_forehead_reach`` is "hairline", and takes the lower
# of the two less a clearance as the reach -- the whole forehead, on
# skin on both sides. His four portraits read 1.52-1.58 (hair above the
# mesh's top). Nothing is measured on a frame past the anchor.
_HAIR_CONTRAST = 0.30
_HAIR_RUN_PX = 6
_HAIRLINE_STEP_PX = 2.0
# Day 17, his first look: 1.8 was too short. Head down at the 13-15
# degrees he sits at, his forehead foreshortens and the hairline reads
# 1.78-1.84 of it (2.06 on the woman's portrait, whose brow sits low) --
# every live frame read "no hair" and the reach fell to the portrait's
# alone. 3.0 finds all of them (recordings/day17, the six captures).
_HAIRLINE_SEARCH = 3.0  # of the forehead height, how far up the walk goes
# The clearance under the lower hairline, an option since day 17
# (``hairline_clearance``; his "cover a bit more" -- 0.08 left a tenth of
# his forehead his own, and 0.04 was kept). Day 21: it is the clearance
# under the PERSON's hairline now (see _forehead_plan; the avatar's own
# roots have their margin), and swept on the two captures with the
# forehead band (jaw_band --capture --render, the top fifth of the mask's
# rows, grey levels; the woman's 200754 / his 105750): 0.04 -> 4.08 /
# 8.89, 0.08 -> 3.50 / 8.42, 0.12 -> 2.99 / 7.48 -- the untouched build
# read 2.70 / 7.51. At 0.04 the avatar's clean skin lay over the person's
# own hair roots; 0.12 of a forehead (six to ten pixels) is still
# "slightly below where the hair starts", his words. His eye may put it
# back.
_HAIRLINE_CLEARANCE = 0.12  # of the forehead height, kept below the hairline
_HAIRLINE_CLEARANCE_MAX = 0.3
_HAIRLINE_MODE = "hairline"
# Day 17 (2026-09-13): "hairline" is the default -- the whole forehead is
# the avatar's, on every avatar that does not say otherwise. Verified by
# replay on his two captures and baska's (recordings/day17/forehead-row-*):
# at 0.85 the top third (his) to two thirds (her taller forehead) of the
# forehead stayed the driver's own skin, shine and all; at "hairline" the
# join sits just under the hair on both.
_FOREHEAD_REACH_DEFAULT: str | float = _HAIRLINE_MODE
_FOREHEAD_REACH_MIN = 0.3
# The option's own bound (the hull is lifted linearly, 3 foreheads is 150
# px in the crop, well inside it); what a measured hairline may reach.
_FOREHEAD_REACH_MAX = 3.0
# The reach when neither hairline could be read: the old ceiling, a
# forehead and a half, not the bound above.
_HAIRLINE_CEILING = 1.6
# Fifteen frontal frames, half a second: five settled on the first frames
# after the freeze, with his head still coming to the camera.
_HAIRLINE_MIN_SAMPLES = 15
# Day 22: how far (of the forehead) the median of the last readings must
# leave the settled hairline before the forehead is planned again.
_HAIRLINE_FOLLOW = 0.15
# Past the settle, the hairline is still read on every Nth frontal frame
# for the calibration's own instrument (frame_readings["hairline"]).
_HAIRLINE_READ_EVERY = 6
# Day 17 (C1b): with the avatar's hair on, the ovals reach this much past
# the portrait's own hairline, so the face oval overlaps the hair matte and
# no strip of the driver's forehead shows between the two.
_HAIR_FOREHEAD_OVERLAP = 0.1
_HAIR_MODES = ("off", "avatar")
# The EMA over the hair mattes (see sha256.avatar.hair): half of each new
# matte, so a flicker of the segmenter is halved and a real move arrives
# within two mattes.
_HAIR_INERTIA = 0.5
# |pitch|, |yaw| within which a frame is sampled. Pitch 20 since his first
# look: he sits at -13..-15 (the camera above the screen), and the fraction
# is read along the face's own vertical, which a pitch shortens as a whole.
_HAIRLINE_FRONTAL = (20.0, 15.0)
_CHEEKS = (50, 280)
_FOREHEAD_TOP = 10


def _luminance(image: np.ndarray, *, bgr: bool = True) -> np.ndarray:
    """Rec. 601 luminance of a BGR (or RGB) uint8 picture, float32."""
    px = image.astype(np.float32)
    r, b = (px[:, :, 2], px[:, :, 0]) if bgr else (px[:, :, 0], px[:, :, 2])
    lum: np.ndarray = 0.299 * r + 0.587 * px[:, :, 1] + 0.114 * b
    return lum


def _patch_mean(lum: np.ndarray, centre: np.ndarray, radius: float) -> float:
    """Mean of the square patch of half-width ``radius`` at ``centre``; NaN
    when it falls off the picture."""
    h, w = lum.shape
    r = max(int(radius), 1)
    x, y = round(float(centre[0])), round(float(centre[1]))
    x0, x1 = max(x - r, 0), min(x + r + 1, w)
    y0, y1 = max(y - r, 0), min(y + r + 1, h)
    if x1 <= x0 or y1 <= y0:
        return math.nan
    return float(lum[y0:y1, x0:x1].mean())


def _forehead_geometry(points: np.ndarray) -> tuple[np.ndarray, float]:
    """The brow top (the brow point highest along the face's own vertical)
    and the forehead height -- from it to the mesh's top point -- in the
    picture's pixels."""
    pts = np.asarray(points, dtype=np.float32)[:FACE_POINTS, :2]
    _, uy, _ = _face_axes(pts)
    brow_pts = pts[list(_BROWS)]
    brow_top = brow_pts[np.argmin(brow_pts @ uy)]
    forehead_h = max(float((brow_top - pts[_FOREHEAD_TOP]) @ uy), 1.0)
    return brow_top, forehead_h


def _hairline_fraction(lum: np.ndarray, points: np.ndarray) -> float | None:
    """Where the hair starts above the brows on ``lum`` (a luminance
    picture), for the face at ``points`` (its pixels, the 468 face points
    or more) -- as a fraction of the forehead height; ``None`` when no
    hair is met within :data:`_HAIRLINE_SEARCH` foreheads."""
    pts = np.asarray(points, dtype=np.float32)[:FACE_POINTS, :2]
    face_w = max(float(pts[:, 0].max() - pts[:, 0].min()), 1.0)
    ux, uy, _ = _face_axes(pts)
    brow_top, forehead_h = _forehead_geometry(pts)
    centre_x = (pts[_CHEEKS[0]] + pts[_CHEEKS[1]]) / 2.0
    base = brow_top - ux * float((brow_top - centre_x) @ ux)  # above the face's middle
    steps = np.arange(0.0, _HAIRLINE_SEARCH * forehead_h, _HAIRLINE_STEP_PX)
    profile: list[float] = []
    for d in steps:
        values = [
            _patch_mean(lum, base - uy * d + ux * dx, 0.012 * face_w)
            for dx in (-0.12 * face_w, 0.0, 0.12 * face_w)
        ]
        finite = [v for v in values if math.isfinite(v)]
        profile.append(float(np.median(finite)) if finite else math.nan)
    skin_zone = [
        v
        for d, v in zip(steps, profile, strict=True)
        if 0.05 * forehead_h <= d <= 0.35 * forehead_h and math.isfinite(v)
    ]
    if not skin_zone:
        return None
    skin = float(np.median(skin_zone))
    run = 0
    for d, v in zip(steps, profile, strict=True):
        if d < 0.35 * forehead_h or not math.isfinite(v):
            run = 0
            continue
        if abs(v - skin) > _HAIR_CONTRAST * max(skin, 1.0):
            run += 1
            if run * _HAIRLINE_STEP_PX >= _HAIR_RUN_PX:
                return float((d - (run - 1) * _HAIRLINE_STEP_PX) / forehead_h)
        else:
            run = 0
    return None


# Day 22 (2026-10-15), his three presses: a pale smeared patch at the top of
# the paste on his forehead -- with the edge match off, a DARK streak in the
# same place: the avatar's own hair roots. The hairline reader above calls
# hair a 30% drop of the median over three small patches, and his portrait's
# roots begin well under that: the row's 5th percentile leaves the forehead's
# skin by 7% at 1.40 of the forehead (the reader's hairline 1.52), by 20% at
# 1.45 -- loose strands the median never sees. The roots' margin (0.15 under
# the reader's hairline, 1.37) left the mask's feather tail (alpha down to
# 0.01 at +0.04 of the reach) in them, and the edge residual then brightened
# the strands into the patch. So the portrait's clean skin now ends where
# the ROOTS begin (:func:`_roots_onset`: the first row, walking up from a
# third of the forehead, whose 5th percentile over a wide band is
# _ROOTS_CONTRAST under the skin's median for _ROOTS_RUN_PX), less
# _ROOTS_CLEARANCE -- or at the reader's hairline less the roots' margin,
# whichever is lower -- and the clean top is a SOURCE row: the stretch
# (:func:`_forehead_plan`) maps it to the join instead of taking the margin
# off after the stretch, which shrank it by the stretch. Read once at load,
# on the portrait alone (the person's side keeps the hairline reader and the
# clearance: nothing of theirs is drawn).
_ROOTS_CONTRAST = 0.07
_ROOTS_RUN_PX = 4
_ROOTS_CLEARANCE = 0.10
_ROOTS_HALF_WIDTH = 0.30  # of the forehead height, to either side of the middle


def _roots_onset(lum: np.ndarray, points: np.ndarray) -> float | None:
    """Where the hair's loose roots begin above the brows on ``lum``, as a
    fraction of the forehead height -- the first row (walking up from a
    third of the forehead, over _ROOTS_RUN_PX of rows) whose 5th percentile
    across a band _ROOTS_HALF_WIDTH wide to either side of the face's middle
    is _ROOTS_CONTRAST under the clean forehead's median; ``None`` when no
    such row is met within :data:`_HAIRLINE_SEARCH` foreheads."""
    pts = np.asarray(points, dtype=np.float32)[:FACE_POINTS, :2]
    ux, uy, _ = _face_axes(pts)
    brow_top, forehead_h = _forehead_geometry(pts)
    centre_x = (pts[_CHEEKS[0]] + pts[_CHEEKS[1]]) / 2.0
    base = brow_top - ux * float((brow_top - centre_x) @ ux)
    height, width = lum.shape[:2]
    half = max(_ROOTS_HALF_WIDTH * forehead_h, 4.0)
    offsets = np.arange(-half, half + 0.5, 1.0, dtype=np.float32)

    def row(d: float) -> np.ndarray | None:
        xs = base[0] - uy[0] * d + ux[0] * offsets
        ys = base[1] - uy[1] * d + ux[1] * offsets
        inside = (xs >= 0) & (xs <= width - 1) & (ys >= 0) & (ys <= height - 1)
        if inside.sum() < len(offsets) // 2:
            return None
        return np.asarray(lum[ys[inside].round().astype(int), xs[inside].round().astype(int)])

    skin_rows = [row(d) for d in np.arange(0.05 * forehead_h, 0.35 * forehead_h, 1.0)]
    skin_values = [r for r in skin_rows if r is not None]
    if not skin_values:
        return None
    skin = float(np.median(np.concatenate(skin_values)))
    floor = (1.0 - _ROOTS_CONTRAST) * max(skin, 1.0)
    run = 0
    for d in np.arange(0.35 * forehead_h, _HAIRLINE_SEARCH * forehead_h, 1.0):
        values = row(float(d))
        if values is None:
            run = 0
            continue
        if float(np.percentile(values, 5)) < floor:
            run += 1
            if run >= _ROOTS_RUN_PX:
                return float((d - (run - 1)) / forehead_h)
        else:
            run = 0
    return None


def _hairline_reach(
    portrait: float | None,
    driver: float | None,
    *,
    clearance: float = _HAIRLINE_CLEARANCE,
    unknown: float = _HAIRLINE_CEILING,
) -> float:
    """The forehead reach for a pair: the lower of the two hairlines less
    ``clearance``, within [_FOREHEAD_REACH_MIN, _FOREHEAD_REACH_MAX]; a
    side with no hair found within the search does not limit it (a high
    forehead), so both unknown means ``unknown`` (:data:`_HAIRLINE_CEILING`)."""
    limits = [h - clearance for h in (portrait, driver) if h is not None]
    reach = min(limits) if limits else unknown
    return float(min(max(reach, _FOREHEAD_REACH_MIN), _FOREHEAD_REACH_MAX))


# Day 21 (2026-10-14), the friend test: two foreheads taller than the
# avatar's (one bald, one with the hair pulled back). The reach above
# took the LOWER of the two hairlines less a clearance, and with the
# person's hair higher than the portrait's that was always the portrait's
# own hairline: the join sat on the avatar's hair roots -- a band of
# darkening skin (measured on his portrait: the luminance leaves the
# forehead's by 5% at 1.40 of the forehead, 10% at 1.44, 30% -- the
# reader's word for hair -- at 1.52; the woman's 10% at 1.90, 30% at
# 2.06; a tenth to a sixth of a forehead of shadow under the hairline) in
# the middle of the person's bare forehead, with their own skin above it.
# That was the stripe. His decision: the forehead should end slightly
# below where the PERSON's hair starts; the avatar's forehead may be
# stretched to reach it, moderately -- "a medium height", not a tall one
# -- and its hair is never pasted (the hair switch stays, off).
#
# So the plan at the settle (:func:`_forehead_plan`): the join the person's
# side wants is their hairline less the clearance, in crop pixels; the
# avatar's clean skin reaches its hairline less _HAIR_ROOT_MARGIN; and the
# render's rows above _STRETCH_ANCHOR foreheads over the brow line (above
# where a brow raise reaches, so the brows are never stretched) are
# stretched by what it takes for the clean skin to reach the join, within
# [1, forehead_stretch_max] -- 1.25 by default, "medium". The portrait's
# oval goes to the stretched clean skin's top, the driver's to their own
# hairline less the clearance, the paste to the lower of the two as ever.
# A person whose hairline could not be read (bald, or hair past the walk)
# gets the whole moderate stretch and no bound from their side.
#
# The stretch is the paste's alone: the render is sampled closer to the
# anchor row above it (see _paste / _paste_gpu), the drive untouched; at
# forehead_stretch_max 1.0 nothing is stretched and the join is as before
# less the roots' margin.
_FOREHEAD_STRETCH_MAX = 1.25
_FOREHEAD_STRETCH_BOUND = 2.0
_STRETCH_ANCHOR = 0.3
_HAIR_ROOT_MARGIN = 0.15
_HAIR_ROOT_MARGIN_MAX = 0.5


def _forehead_stretch_max(options: dict[str, Any]) -> float:
    """``forehead_stretch_max`` in ``[1, 2]``, default
    :data:`_FOREHEAD_STRETCH_MAX`: how far the avatar's upper forehead may
    be stretched toward the person's hairline; 1 is none."""
    value = float(options.get("forehead_stretch_max", _FOREHEAD_STRETCH_MAX))
    if not 1.0 <= value <= _FOREHEAD_STRETCH_BOUND:
        msg = f"forehead_stretch_max must be in [1, {_FOREHEAD_STRETCH_BOUND}], got {value}"
        raise ValueError(msg)
    return value


def _hair_root_margin(options: dict[str, Any]) -> float:
    """``hair_root_margin`` in ``[0, 0.5]``, default :data:`_HAIR_ROOT_MARGIN`:
    how far under the avatar's own hairline its skin counts as clean, as a
    fraction of its forehead height."""
    value = float(options.get("hair_root_margin", _HAIR_ROOT_MARGIN))
    if not 0.0 <= value <= _HAIR_ROOT_MARGIN_MAX:
        msg = f"hair_root_margin must be in [0, {_HAIR_ROOT_MARGIN_MAX}], got {value}"
        raise ValueError(msg)
    return value


def _forehead_plan(
    portrait_hairline: float | None,
    driver_hairline: float | None,
    portrait_forehead: float,
    driver_forehead: float,
    *,
    clearance: float = _HAIRLINE_CLEARANCE,
    margin: float = _HAIR_ROOT_MARGIN,
    stretch_max: float = _FOREHEAD_STRETCH_MAX,
    anchor: float = _STRETCH_ANCHOR,
    portrait_roots: float | None = None,
    roots_clearance: float = _ROOTS_CLEARANCE,
) -> tuple[float, float, float]:
    """The forehead at the settle (day 21): ``(portrait_reach, driver_reach,
    stretch)`` -- the portrait oval's reach (of its forehead), the driver
    oval's (of theirs) and the render's stretch above the anchor row -- from
    the two hairlines (fractions of each forehead; ``None`` unread), the two
    foreheads' heights in crop pixels, the clearance under the driver's
    hairline, the roots' margin under the portrait's and the stretch's cap.
    Day 22: the portrait's clean skin ends at its hairline less the margin
    or, when its roots were read (``portrait_roots``, see
    :func:`_roots_onset`), at the roots less ``roots_clearance`` -- the
    lower of the two -- and that is a SOURCE row: the stretch carries it to
    the join (``portrait_reach = anchor + stretch * (skin_top - anchor)``).
    The reaches are within [_FOREHEAD_REACH_MIN, _FOREHEAD_REACH_MAX]."""
    hairline = _HAIRLINE_CEILING if portrait_hairline is None else portrait_hairline
    skin_top = hairline - margin
    if portrait_roots is not None:
        skin_top = min(skin_top, portrait_roots - roots_clearance)
    span = max(skin_top - anchor, 0.1)
    if driver_hairline is None:
        stretch = stretch_max
        driver_reach = _FOREHEAD_REACH_MAX
    else:
        driver_reach = driver_hairline - clearance
        wanted = driver_reach * driver_forehead / max(portrait_forehead, 1.0)
        stretch = min(max((wanted - anchor) / span, 1.0), stretch_max)
    portrait_reach = anchor + stretch * span

    def bound(reach: float) -> float:
        return float(min(max(reach, _FOREHEAD_REACH_MIN), _FOREHEAD_REACH_MAX))

    return bound(portrait_reach), bound(driver_reach), float(stretch)


def _hairline_clearance(options: dict[str, Any]) -> float:
    """``hairline_clearance`` in ``[0, 0.3]``, default
    :data:`_HAIRLINE_CLEARANCE`: how far under the lower hairline the join
    stays, as a fraction of the forehead height. Smaller covers more of
    the forehead; at 0 the feather lies on the first hair."""
    value = float(options.get("hairline_clearance", _HAIRLINE_CLEARANCE))
    if not 0.0 <= value <= _HAIRLINE_CLEARANCE_MAX:
        msg = f"hairline_clearance must be in [0, {_HAIRLINE_CLEARANCE_MAX}], got {value}"
        raise ValueError(msg)
    return value


def _driver_hairline(options: dict[str, Any]) -> float | None:
    """``driver_hairline`` -- this person's hairline as a fraction of the
    forehead height, measured by Calibrate me over the still step (day 17)
    and laid over from the profile; ``None`` (the default) means the
    engine reads it live at the anchor. A number in [0.3, 3.0]."""
    raw = options.get("driver_hairline")
    if raw is None:
        return None
    value = float(raw)
    if not _FOREHEAD_REACH_MIN <= value <= _FOREHEAD_REACH_MAX:
        msg = (
            f"driver_hairline must be in [{_FOREHEAD_REACH_MIN}, {_FOREHEAD_REACH_MAX}], "
            f"got {value}"
        )
        raise ValueError(msg)
    return value


# Day 22 (2026-10-15), the width fit. The crop is sized by the face's
# HEIGHT (forehead top to chin, see crop_matrix), so in the crop the two
# faces are the same height and differ in WIDTH alone: his resting face
# reads 4% narrower than his portrait's (scripts/probes/face_width.py on
# his 2026-09-12 session: rim 0.788 of the height against the portrait's
# 0.822; the friend's 0.99, the woman's 1.00). The paste's mask is the
# portrait's oval AND the driver's, so on a narrower person the driver's
# oval cuts the render's cheeks off: the join sits on the avatar's cheek
# skin, inside its own outline, and the avatar's features sit wider in the
# face than the person's.
#
# The fit: once the anchor is frozen, the ratio of the driver's rim width
# to the portrait's (both along the face's own horizontal, in the crop,
# read on still frontal frames and taken at the median) scales the render
# and the portrait's own oval horizontally about the portrait's rim centre,
# in the paste alone -- the drive, the network, the crop are untouched
# (see _squeeze_face, _settle_forehead). Bounded moderate: a person wider
# than the portrait is widened likewise, up to the same bound.
# ``face_width_fit`` (0..1, 1 default) is how much of the measured
# difference is applied; 0 is off and byte-identical to before.
# ``driver_face_width`` (a ratio, from Calibrate me's profile) replaces the
# live reading, as driver_hairline does the hairline's.
_FACE_WIDTH_FIT = 1.0
_FACE_WIDTH_MIN = 0.85
_FACE_WIDTH_MAX = 1.15
# Frontal enough to read the width from: a turn foreshortens the rim
# (measured: 5-7% per 10 degrees of yaw on the 2D rim), a tilt the height.
_WIDTH_FRONTAL = (10.0, 8.0)
_WIDTH_MIN_SAMPLES = 8
# A scale this close to 1 is not applied at all (no resample for nothing).
_WIDTH_DEADBAND = 0.005


def _face_width_fit(options: dict[str, Any]) -> float:
    """``face_width_fit`` in ``[0, 1]``, default :data:`_FACE_WIDTH_FIT`."""
    value = float(options.get("face_width_fit", _FACE_WIDTH_FIT))
    if not 0.0 <= value <= 1.0:
        msg = f"face_width_fit must be in [0, 1], got {value}"
        raise ValueError(msg)
    return value


def _driver_face_width(options: dict[str, Any]) -> float | None:
    """``driver_face_width`` -- the person's rim width over the portrait's
    (the crop's units), measured by Calibrate me; ``None`` (the default)
    means the engine reads it live at the anchor. In [0.5, 2]."""
    raw = options.get("driver_face_width")
    if raw is None:
        return None
    value = float(raw)
    if not 0.5 <= value <= 2.0:
        msg = f"driver_face_width must be in [0.5, 2], got {value}"
        raise ValueError(msg)
    return value


def _rim_width(points: np.ndarray) -> float:
    """The face rim's extent along the face's own horizontal, in the
    points' own pixels (the 468 face points or more)."""
    pts = np.asarray(points, dtype=np.float32)[:FACE_POINTS, :2]
    ux, _, _ = _face_axes(pts)
    along = pts[list(_FACE_OVAL)] @ ux
    return max(float(along.max() - along.min()), 1.0)


def _rim_centre_x(points: np.ndarray) -> float:
    """The rim's middle column, crop pixels -- what the width fit scales about."""
    pts = np.asarray(points, dtype=np.float32)[:FACE_POINTS, 0]
    rim = pts[list(_FACE_OVAL)]
    return float((rim.max() + rim.min()) / 2.0)


def _width_scale(ratio: float | None, fit: float) -> float:
    """The horizontal scale the render and the portrait's oval take: ``fit``
    of the way from 1 to ``ratio``, within [_FACE_WIDTH_MIN, _FACE_WIDTH_MAX];
    1 for an unread ratio, an off fit, or a difference inside the dead band."""
    if ratio is None or fit <= 0.0 or not math.isfinite(ratio):
        return 1.0
    scale = min(max(1.0 + fit * (ratio - 1.0), _FACE_WIDTH_MIN), _FACE_WIDTH_MAX)
    return 1.0 if abs(scale - 1.0) < _WIDTH_DEADBAND else float(scale)


def _squeeze_points(points: np.ndarray, centre_x: float, scale: float) -> np.ndarray:
    """``points`` with their x scaled by ``scale`` about ``centre_x``."""
    out = np.array(points, dtype=np.float32, copy=True)
    out[:, 0] = centre_x + (out[:, 0] - centre_x) * scale
    return out


def _squeeze_face(
    face: np.ndarray,
    centre_x: float,
    scale: float,
    centre_y: float = 0.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    """The rendered crop (H, W, 3) with its columns scaled by ``scale``
    about ``centre_x`` -- the width fit on the CPU path -- and, day 23,
    its rows by ``scale_y`` about ``centre_y`` (the height fit), in one
    resample."""
    import cv2

    h, w = face.shape[:2]
    matrix = np.array(
        [[scale, 0.0, centre_x * (1.0 - scale)], [0.0, scale_y, centre_y * (1.0 - scale_y)]],
        dtype=np.float32,
    )
    squeezed: np.ndarray = cv2.warpAffine(
        face, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return squeezed


def _width_frame_ok(pose: HeadPose | None) -> bool:
    if pose is None:
        return False
    return abs(pose.pitch) <= _WIDTH_FRONTAL[0] and abs(pose.yaw) <= _WIDTH_FRONTAL[1]


# Day 23 (2026-10-16), the height fit -- the width fit's vertical twin. The
# crop makes the two faces the same HEIGHT (forehead top to chin), but not
# the same shape inside it: a portrait with a taller forehead puts its brows
# and eyes LOWER than the person's at the same chin. His first shaved
# avatar, capture 02:33: the avatar's nose, mouth and chin on his, its eyes
# 10 px under his, its brows 20 (scripts/probes/face_height.py: his brow
# top to chin 0.843 of the face height, the portrait's 0.805 -> x1.047;
# his own avatar x1.032, the woman's x1.017). Where the brows sit is where
# the join is: the forehead starts there, the oval's top is measured from
# there, and his own brow shows beside the avatar's when they differ.
#
# The fit: the ratio of the driver's brow-top-to-chin to the portrait's
# (along the face's own vertical, in the crop, still frontal frames, the
# median) scales the render and the portrait's points vertically about
# the portrait's CHIN ROW -- the chin stays on his, everything above
# rises -- in the paste alone, in the same resample as the width. The
# forehead's plan then reads the lifted geometry. ``face_height_fit``
# (0..1, 1 default) is how much is applied; 0 is off and byte-identical.
# ``driver_face_height`` (a ratio, from a profile) replaces the reading.
_FACE_HEIGHT_FIT = 1.0
_FACE_HEIGHT_MIN = 0.90
_FACE_HEIGHT_MAX = 1.10


def _face_height_fit(options: dict[str, Any]) -> float:
    """``face_height_fit`` in ``[0, 1]``, default :data:`_FACE_HEIGHT_FIT`."""
    value = float(options.get("face_height_fit", _FACE_HEIGHT_FIT))
    if not 0.0 <= value <= 1.0:
        msg = f"face_height_fit must be in [0, 1], got {value}"
        raise ValueError(msg)
    return value


def _driver_face_height(options: dict[str, Any]) -> float | None:
    """``driver_face_height`` -- the person's brow-top-to-chin over the
    portrait's, both of the face height, from a profile; ``None`` unset."""
    raw = options.get("driver_face_height")
    if raw is None:
        return None
    value = float(raw)
    if not 0.5 <= value <= 2.0:
        msg = f"driver_face_height must be in [0.5, 2], got {value}"
        raise ValueError(msg)
    return value


def _brow_chin(points: np.ndarray) -> float:
    """The brow top to the chin along the face's own vertical, in the
    points' own pixels (the 468 face points or more)."""
    pts = np.asarray(points, dtype=np.float32)[:FACE_POINTS, :2]
    _, uy, _ = _face_axes(pts)
    brow_top, _ = _forehead_geometry(pts)
    return max(float((pts[_CHIN] - brow_top) @ uy), 1.0)


def _chin_row(points: np.ndarray) -> float:
    """The chin's row, crop pixels -- what the height fit scales about."""
    return float(np.asarray(points, dtype=np.float32)[_CHIN, 1])


def _height_scale(ratio: float | None, fit: float) -> float:
    """The vertical scale the render and the portrait's points take:
    ``fit`` of the way from 1 to ``ratio``, within [_FACE_HEIGHT_MIN,
    _FACE_HEIGHT_MAX]; 1 for an unread ratio, an off fit, or a difference
    inside the dead band."""
    if ratio is None or fit <= 0.0 or not math.isfinite(ratio):
        return 1.0
    scale = min(max(1.0 + fit * (ratio - 1.0), _FACE_HEIGHT_MIN), _FACE_HEIGHT_MAX)
    return 1.0 if abs(scale - 1.0) < _WIDTH_DEADBAND else float(scale)


def _lift_points(points: np.ndarray, chin_y: float, scale: float) -> np.ndarray:
    """``points`` with their y scaled by ``scale`` about the row ``chin_y``."""
    out = np.array(points, dtype=np.float32, copy=True)
    out[:, 1] = chin_y + (out[:, 1] - chin_y) * scale
    return out


# Day 22 evening, his held press at a turn to his right (yaw -45) and his
# words: "on the side I turn to, my eye, brow, cheek and lip show -- both
# sides". The network draws the head less turned than his and narrower on
# the profile side (see _turn_boost), and the paste follows HIS oval: between
# the drawn face's silhouette and his own profile the guard keeps nothing
# (the render there is the photograph's background), so his skin shows,
# and inside the feather his eye shines through beside the avatar's. The
# turn boost closes most of the gap in angle; what is left is closed in
# the PASTE: on every row between the brows and the chin the drawn face's
# near edge (the first non-background, non-border column from the profile
# side) is read against the paste's own near edge (his oval with the
# day-20 room), and the render is scaled horizontally about the paste's
# far edge so the two meet -- a per-row scale within _OUTLINE_SCALE,
# smoothed over _OUTLINE_SMOOTH rows, faded to nothing outside the fitted
# rows, and eased over frames (_OUTLINE_EMA). It comes on with the turn
# (the room's own gate, nothing within 12 degrees of the rest yaw), so a
# frontal frame's paste is the one it was to the byte. Prototyped on the
# press in scripts/probes/outline_fit.py: the near-side gap read -6 px at
# the median, 19 at the brow, and the avatar's eye landed on his.
# ``outline_fit`` (portrait option, 0..1, default 1) is how much of the
# fit is applied; 0 is off.
_OUTLINE_FIT = 1.0
# The per-row scale's bounds: source x = x * scale + shift, so 0.88 widens
# the near side by at most 14% (measured on his session's left turn at
# +51: his nose stands out further than the avatar's, and a widening of
# 18% drew the avatar's nose large). The fit never NARROWS (1.0): where
# the drawn face already reaches past the paste's edge there is nothing
# of his to cover, and pulling the render in there (as the first build
# did at a moderate turn, where the drawn edge lies outside the oval's
# eroded edge) only uncovered him -- his "the face is lost sooner" of
# the second look.
_OUTLINE_SCALE = (0.88, 1.0)
# The paste's edge the drawn face is fitted to is the oval's own hull
# (alpha over this), not its half-alpha line a shrink and half a feather
# inside it.
_OUTLINE_TARGET_ALPHA = 0.05
# The nose is left alone: on its rows the profile IS the nose, and his
# nose and the avatar's differ in shape, not in turn -- fitted, the
# avatar's nose stretched into a blob (his left-turn press of 20:29) and
# sampled the background as a white sheen on the tip (his right-turn one).
# The band from under the eyes to under the nostrils reads the identity,
# ramped over this many rows on either side.
_OUTLINE_NOSE_RAMP = 12.0
_NOSE_BOTTOM = 2
_OUTLINE_SMOOTH = 21  # rows, odd
_OUTLINE_FADE_ROWS = 40.0
_OUTLINE_EMA = 0.5  # of the previous frame's map kept
_OUTLINE_BORDER = 12.0  # below this (max channel) a crop pixel is the warp's black border
_OUTLINE_MIN_SPAN = 8.0
# The drawn face's near edge is looked for within this many crop px outside
# the paste's own near edge (and nowhere past its far edge): the render is
# the face at his face, and the photograph's own edge against the warp's
# black border -- a dark grey ramp 45 px into the crop on his press --
# would otherwise read as the "drawn face" (the prototype took the largest
# blob; the engine takes the window).
_OUTLINE_WINDOW = 48


def _outline_fit(options: dict[str, Any]) -> float:
    """``outline_fit`` in ``[0, 1]``, default :data:`_OUTLINE_FIT`."""
    value = float(options.get("outline_fit", _OUTLINE_FIT))
    if not 0.0 <= value <= 1.0:
        msg = f"outline_fit must be in [0, 1], got {value}"
        raise ValueError(msg)
    return value


def _row_edges(mask: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Per row of ``mask`` (H, W): the first and the last column over
    ``threshold``, -1 where the row has none."""
    hit = mask > threshold
    any_row = hit.any(axis=1)
    first = np.where(any_row, hit.argmax(axis=1), -1)
    last = np.where(any_row, mask.shape[1] - 1 - hit[:, ::-1].argmax(axis=1), -1)
    return first.astype(np.int32), last.astype(np.int32)


def _outline_maps(
    drawn_near: np.ndarray,
    target_first: np.ndarray,
    target_last: np.ndarray,
    *,
    near_left: bool,
    first_row: int,
    last_row: int,
    weight: float,
    previous: tuple[np.ndarray, np.ndarray] | None,
    exclude: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """The per-row horizontal map ``(scale, shift)`` (each (H,) float32:
    source x = x * scale + shift, crop pixels) that takes the drawn face's
    near edge onto the paste's near edge about the paste's far edge, on
    rows ``first_row``..``last_row``, faded outside, left alone on the
    rows of ``exclude`` (the nose band, ramped over _OUTLINE_NOSE_RAMP),
    eased against ``previous`` and weighted by ``weight``; ``None`` when
    no row fits."""
    import cv2

    size = len(drawn_near)
    scale = np.ones(size, dtype=np.float32)
    shift = np.zeros(size, dtype=np.float32)
    pivots = np.zeros(size, dtype=np.float32)
    valid = np.zeros(size, dtype=bool)
    lo, hi = max(first_row, 0), min(last_row, size - 1)
    for r in range(lo, hi + 1):
        d_near = float(drawn_near[r])
        tl, tr = float(target_first[r]), float(target_last[r])
        if d_near < 0.0 or tl < 0.0 or tr - tl < _OUTLINE_MIN_SPAN:
            continue
        pivot, t_near = (tr, tl) if near_left else (tl, tr)
        if abs(pivot - t_near) < _OUTLINE_MIN_SPAN:
            continue
        s = (pivot - d_near) / (pivot - t_near)
        scale[r] = min(max(s, _OUTLINE_SCALE[0]), _OUTLINE_SCALE[1])
        pivots[r] = pivot
        valid[r] = True
    if not valid.any():
        return None
    idx = np.nonzero(valid)[0]
    first, last = int(idx[0]), int(idx[-1])
    # the unfitted rows inside take the nearest fitted row's map
    for r in range(first, last + 1):
        if valid[r]:
            shift[r] = pivots[r] - pivots[r] * scale[r]
        else:
            j = idx[np.abs(idx - r).argmin()]
            scale[r], shift[r] = scale[j], pivots[j] - pivots[j] * scale[j]
    smooth_scale = np.asarray(
        cv2.GaussianBlur(scale.reshape(-1, 1), (1, _OUTLINE_SMOOTH), 0), dtype=np.float32
    ).reshape(-1)
    smooth_shift = np.asarray(
        cv2.GaussianBlur(shift.reshape(-1, 1), (1, _OUTLINE_SMOOTH), 0), dtype=np.float32
    ).reshape(-1)
    rows = np.arange(size, dtype=np.float32)
    outside = np.where(rows < first, first - rows, np.where(rows > last, rows - last, 0.0))
    fade = np.clip(1.0 - outside / _OUTLINE_FADE_ROWS, 0.0, 1.0).astype(np.float32) * float(weight)
    if exclude is not None:
        lo_x, hi_x = float(exclude[0]), float(exclude[1])
        inside = np.where(rows < lo_x, lo_x - rows, np.where(rows > hi_x, rows - hi_x, 0.0))
        fade = fade * np.clip(inside / _OUTLINE_NOSE_RAMP, 0.0, 1.0).astype(np.float32)
    out_scale = np.asarray(1.0 + (smooth_scale - 1.0) * fade, dtype=np.float32)
    out_shift = np.asarray(smooth_shift * fade, dtype=np.float32)
    if previous is not None:
        out_scale = np.asarray(
            _OUTLINE_EMA * previous[0] + (1.0 - _OUTLINE_EMA) * out_scale, dtype=np.float32
        )
        out_shift = np.asarray(
            _OUTLINE_EMA * previous[1] + (1.0 - _OUTLINE_EMA) * out_shift, dtype=np.float32
        )
    return out_scale, out_shift


def _outline_window(
    target_first: np.ndarray, target_last: np.ndarray, near_left: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the columns ``(low, high)`` the drawn face's near edge is
    looked for in: from _OUTLINE_WINDOW outside the paste's near edge to
    its far edge; an empty window (-1, -2) where the row has no paste."""
    has = target_first >= 0
    if near_left:
        low = np.where(has, target_first - _OUTLINE_WINDOW, -1)
        high = np.where(has, target_last, -2)
    else:
        low = np.where(has, target_first, -1)
        high = np.where(has, target_last + _OUTLINE_WINDOW, -2)
    return low.astype(np.int32), high.astype(np.int32)


def _drawn_near_edge_np(
    face: np.ndarray,
    background: np.ndarray,
    near_left: bool,
    window: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Per row of the rendered crop (H, W, 3): the first column from the
    near side, within ``window`` (per-row (low, high) columns, see
    :func:`_outline_window`; None: the whole row), that is neither the
    photograph's background nor the crop's black border; -1 where none."""
    not_background = np.abs(face.astype(np.float32) - background).max(axis=2) >= _BACKGROUND_NEAR
    not_border = face.max(axis=2) > _OUTLINE_BORDER
    drawn = not_background & not_border
    if window is not None:
        cols = np.arange(face.shape[1])[None, :]
        drawn &= (cols >= window[0][:, None]) & (cols <= window[1][:, None])
    first, last = _row_edges(drawn.astype(np.float32))
    return first if near_left else last


# Day 23 (2026-10-16), his press at a turn to his left (+34): "my eye and
# eyelashes remain". Measured (scripts/probes/eye_alpha.py): the drawn
# face's far edge ends 5-10 px inside his far eye's outer corner, and his
# lashes sit in the paste's feather beyond it (alpha 0.88 at the corner,
# 0.59 five px out, 0.08 ten) -- the same on the 09-18 tree. The room
# (day 20) pastes the driver's oval only where the render is the drawn
# face, and there the render is the photograph's background: nothing of
# the avatar reaches his lashes. So at a turn the render's profile edge is
# carried outward: on every row, the ``profile_fill_px`` columns beyond
# the drawn face's edge on the profile side that are the background take
# the edge column's own colour (the avatar's cheek skin), ramped with the
# turn gate; the guard then sees drawn face there and the room pastes it,
# still inside the driver's oval. Off a turn nothing is touched.
_PROFILE_FILL_PX = 12
_PROFILE_FILL_MAX = 40
# The colour carried out is read this many columns INSIDE the drawn edge,
# and the edge's own columns are overwritten with it too: the edge column
# is the network's anti-aliased blend of skin and the photograph's light
# background -- carried out as it was it drew a pale streak down the far
# temple on his press, and left in place under the filled strip it drew a
# pale jagged line along it.
_PROFILE_FILL_INSET = 3


def _profile_fill(options: dict[str, Any]) -> int:
    """``profile_fill_px`` in ``[0, _PROFILE_FILL_MAX]``, default
    :data:`_PROFILE_FILL_PX`; 0 is off."""
    value = int(options.get("profile_fill_px", _PROFILE_FILL_PX))
    if not 0 <= value <= _PROFILE_FILL_MAX:
        msg = f"profile_fill_px must be in [0, {_PROFILE_FILL_MAX}], got {value}"
        raise ValueError(msg)
    return value


def _profile_fill_np(
    face: np.ndarray, background: np.ndarray, edge: np.ndarray, near_left: bool, width_px: int
) -> np.ndarray:
    """The rendered crop (H, W, 3) with, on every row, the ``width_px``
    background columns beyond the drawn face's edge on the profile side
    (``edge``, (H,) int, -1 where none; see :func:`_drawn_near_edge_np`)
    filled with the edge column's own colour."""
    h, w = face.shape[:2]
    cols = np.arange(w)[None, :]
    e = edge.astype(np.int64)[:, None]
    inset = _PROFILE_FILL_INSET
    if near_left:
        beyond = (cols < e) & (cols >= e - width_px)
        rim = (cols >= e) & (cols < e + inset)
        inside = e + inset
    else:
        beyond = (cols > e) & (cols <= e + width_px)
        rim = (cols <= e) & (cols > e - inset)
        inside = e - inset
    is_background = np.abs(face.astype(np.float32) - background).max(axis=2) < _BACKGROUND_NEAR
    write = (e >= 0) & ((beyond & is_background) | rim)
    src = np.where(write, np.clip(inside, 0, w - 1), cols)
    filled: np.ndarray = np.take_along_axis(face, src[:, :, None], axis=1)
    return filled


def _remap_outline(
    patch: np.ndarray, scale: np.ndarray, shift: np.ndarray, left: int, top: int
) -> np.ndarray:
    """``patch`` (h, w, float32; a crop-space map such as the guard's keep,
    whose first pixel is crop column ``left``, row ``top``) read through
    the outline fit's per-row map, so it lines up with the render the paste
    samples through the same map (see _outline_maps)."""
    import cv2

    h, w = patch.shape[:2]
    rows = np.clip(np.arange(h) + top, 0, len(scale) - 1)
    xs = np.arange(w, dtype=np.float32)[None, :] + left
    map_x = (xs * scale[rows][:, None] + shift[rows][:, None] - left).astype(np.float32)
    map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
    out: np.ndarray = cv2.remap(
        patch, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return out


def _hairline_drifted(loaded: _Portrait) -> bool:
    """Day 22: whether the median of the last :data:`_HAIRLINE_MIN_SAMPLES`
    readings (at least half of them a reading) has left the hairline the
    forehead was settled from by more than :data:`_HAIRLINE_FOLLOW`."""
    if loaded.settle_hairline is None or len(loaded.hairline_samples) < _HAIRLINE_MIN_SAMPLES:
        return False
    found = [s for s in loaded.hairline_samples if s is not None]
    if len(found) * 2 < len(loaded.hairline_samples):
        return False
    return abs(float(np.median(found)) - loaded.settle_hairline) > _HAIRLINE_FOLLOW


def _hairline_frame_ok(pose: HeadPose | None) -> bool:
    """Frontal enough to read the hairline from -- a tilt foreshortens the
    forehead, a turn puts the hair over it."""
    if pose is None:
        return False
    return abs(pose.pitch) <= _HAIRLINE_FRONTAL[0] and abs(pose.yaw) <= _HAIRLINE_FRONTAL[1]


def _forehead_reach_is_hairline(options: dict[str, Any]) -> bool:
    return bool(options.get("mask_forehead_reach", _FOREHEAD_REACH_DEFAULT) == _HAIRLINE_MODE)


def _hair_mode(options: dict[str, Any]) -> str:
    """``hair`` -- ``"off"`` (the default: the face oval alone, as before
    day 17) or ``"avatar"``: the portrait's own hair pasted over the
    driver's head from a matte of every rendered crop (see
    sha256.avatar.hair and :meth:`LivePortraitEngine.set_hair`)."""
    raw = options.get("hair", "off")
    if raw is True:
        return "avatar"
    if raw is False or raw is None:
        return "off"
    mode = str(raw)
    if mode not in _HAIR_MODES:
        msg = f"hair must be one of {_HAIR_MODES}, got {raw!r}"
        raise ValueError(msg)
    return mode


def _hair_skin(options: dict[str, Any]) -> bool:
    """``hair_skin`` -- whether the render's own face skin above the eyes
    joins the paste along with the hair (see hair.skin_weight). Default
    False: tried 2026-09-13 on his two captures, the avatar's paler temple
    skin came out as a patch over his own hair beside the hairline, worse
    than the hair showing; kept as a lever to measure against."""
    return bool(options.get("hair_skin", False))


def _forehead_reach(options: dict[str, Any]) -> float:
    """``mask_forehead_reach`` in ``[0.3, 1.6]`` or ``"hairline"`` (the
    default since day 17): how far up the forehead the ovals go, as a
    fraction of the brow-to-mesh-top distance -- 1.0 is the mesh's own
    top, above it the forehead arc is lifted past it (day 16), lower stops
    the join further down on bare skin. Day 13's forehead-line lever,
    swept with scripts/probes/jaw_band.py; both the portrait's oval and
    the driver's read it. ``"hairline"`` reads as ``_OVAL_FOREHEAD_REACH``
    (0.85) here and is settled at the anchor from the two faces' own
    hairlines (see :func:`_hairline_reach`)."""
    raw = options.get("mask_forehead_reach", _FOREHEAD_REACH_DEFAULT)
    if isinstance(raw, str):
        if raw == _HAIRLINE_MODE:
            return _OVAL_FOREHEAD_REACH
        msg = (
            f"mask_forehead_reach must be a number in [{_FOREHEAD_REACH_MIN}, "
            f"{_FOREHEAD_REACH_MAX}] or 'hairline', got {raw!r}"
        )
        raise ValueError(msg)
    value = float(raw)
    if not _FOREHEAD_REACH_MIN <= value <= _FOREHEAD_REACH_MAX:
        msg = (
            f"mask_forehead_reach must be in [{_FOREHEAD_REACH_MIN}, {_FOREHEAD_REACH_MAX}], "
            f"got {value}"
        )
        raise ValueError(msg)
    return value


def _mask_jaw_shrink(options: dict[str, Any]) -> float:
    """``mask_jaw_shrink`` in ``[0, 0.3]``, default ``_OVAL_JAW_SHRINK`` (0):
    how far the driver oval's jaw is pulled toward the face centre, as a
    fraction of face width. 0 is the join at the contour; 0.08 is the
    bearded setting measured on day 12."""
    value = float(options.get("mask_jaw_shrink", _OVAL_JAW_SHRINK))
    if not 0.0 <= value <= 0.3:
        raise ValueError(f"mask_jaw_shrink must be in [0, 0.3], got {value}")
    return value


# Day 13: the jaw under an open mouth (his "o" and wide-open captures of
# 2026-09-07 03:24 -- his own jaw showing below the avatar's). The paste
# mask is the portrait's oval intersected with his (see _frame_mask); his
# oval follows his dropped jaw, but the portrait's was built once at load
# from the portrait's own closed-mouth chin, and the intersection stops
# there while the avatar's rendered chin has dropped with his. So the
# portrait's oval is rebuilt per frame with its points below the mouth line
# let down by his chin's drop from rest, in crop pixels -- nothing at the
# mouth line, all of it at the chin (see _oval_mask's ``chin_drop``). The
# drop is the nose-tip-to-chin reach against the anchor's (see _chin_drop),
# never more than this fraction of the portrait's own face height in the
# crop, which is also how far the mask's box is extended below at load
# (see _mask_bbox's ``reach_below``): the paste warps only into that box.
_CHIN = 152
_MOUTH_CORNERS = (61, 291)
_CHIN_DROP_MAX = 0.35
# Rows the oval's hull keeps clear of the crop's last row beyond its own
# measured feather tail -- see load_avatar's ``edge_margin`` (day 14).
_MASK_EDGE_CLEARANCE = 2.0

# ...but never past the crop's own bottom row, and that is the real limit
# (measured 2026-09-09, after his "the jaw shows" captures). His portrait
# in this crop (scale 2.2, vy_ratio -0.125): face height 233 px, the chin
# at row 489 of 512 -- **22 px of crop below the chin**, because the vy
# ratio spends the crop's room above the face (256 px of hair) and leaves
# almost none under it. An oval let down past 511 is not let down at all:
# fillConvexPoly clips it at the array's edge, so the mask ends in a
# HARD row instead of a feather, and the paste draws a straight dark line
# across his neck (driving.mkv 1060, the day-13 drop: the mask's alpha
# reached the last row on 118 of 118 open-mouth frames). The same wall
# limits the render itself -- on those frames the avatar's own rendered
# chin reaches row 510. So the drop and the reach share whatever room the
# crop actually has (:meth:`LivePortraitEngine.load_avatar`), and the
# whole question of following a wide-open jaw is a CROP question first:
# more room below the chin means the vy ratio, which moves the framing
# the extractor was tuned on (the day-6 cliff) and needs its own sweep.
# Day 14 swept it (see CROP_VY_RATIO): no cliff on that axis, and at the
# kept -0.06 the chin sits at row 455 with 56 px under it -- the drop's
# cap on his portrait is ~52 px, the day-13 open-jaw maximum was 51.

# Day 13, his captures of 2026-09-09 03:28 ("my jaw shows", brows raised
# and looking down): on both, the avatar's rendered chin sits within 4 px
# of his own, and the mask's 0.5 contour 5 px ABOVE both -- the erosion
# (_OVAL_SHRINK) pulls both ovals inside their chin point and the feather
# fades the avatar out across his chin line, so the underside of his chin
# shows through it. ``mask_chin_reach`` lets BOTH ovals down at the chin
# by this fraction of the portrait's crop face height (ramped from the
# mouth line, like the drop), so the join lands under his chin on his
# neck, where the portrait's own under-chin skin is pasted and the edge
# match carries the tone. Default 0: the join at the contour, as before.
# The warp engine had the same lever (_JAW_REACH) and he turned it off in
# August -- a stretched photograph's chin; this engine renders the chin
# where his is, so it is measured again.
_CHIN_REACH_MAX = 0.2
# Day 16, block B3 ("auto"): the reach from the mask's own geometry
# rather than by hand. The ROADMAP's premise was a proportion -- the
# driver's lower face against the portrait's -- and it was measured: on
# his four portraits the network draws the chin on the portrait's own
# chin point (within 0.3-3 px, self-driven), and his nose-to-chin over
# face height differs from the portraits' by 0-4 crop px (0.446-0.454
# against 0.449-0.468); but the anchor's own rest_chin, learned live at
# a session's start, read 10 px longer than his frontal median on the
# 10:14 capture, so a reach that added the gap put the join 14-21 px
# under his chin (the hand 0.04: +7..+12; day 15's look at 0.08, +26,
# was "onto the neck"). What the hand value IS, in the mask's own
# terms, is the contour's clearance above the chin point: the erosion's
# radius plus half the feather, both of the crop face width -- 9.4 px on
# his portrait against the hand's 9.3. "auto" is that, resolved at load
# like a number; the importer writes it for every new avatar.
_CHIN_REACH_AUTO = "auto"


def _mask_chin_reach(options: dict[str, Any]) -> float:
    """``mask_chin_reach`` in ``[0, 0.2]``, default 0: how far both ovals
    reach below the chin, as a fraction of the portrait's crop face
    height (see the comment above); ``"auto"`` reads as 0 here and is
    resolved at the anchor (see :func:`_chin_reach_is_auto`)."""
    raw = options.get("mask_chin_reach", 0.0)
    if isinstance(raw, str):
        if raw == _CHIN_REACH_AUTO:
            return 0.0
        raise ValueError(
            f"mask_chin_reach must be a number in [0, {_CHIN_REACH_MAX}] or 'auto', got {raw!r}"
        )
    value = float(raw)
    if not 0.0 <= value <= _CHIN_REACH_MAX:
        raise ValueError(f"mask_chin_reach must be in [0, {_CHIN_REACH_MAX}], got {value}")
    return value


def _chin_reach_is_auto(options: dict[str, Any]) -> bool:
    return bool(options.get("mask_chin_reach", 0.0) == _CHIN_REACH_AUTO)


def _auto_chin_reach(crop_points: np.ndarray) -> float:
    """The mask contour's own clearance above the chin point, in crop
    pixels: the erosion's radius plus half the feather, both fractions of
    the portrait's crop face width (see :func:`_oval_mask`)."""
    face_width = max(float(crop_points[:, 0].max() - crop_points[:, 0].min()), 1.0)
    return float(face_width * (_OVAL_SHRINK / 2.0 + _OVAL_FEATHER / 2.0))


def _chin_follow(options: dict[str, Any]) -> float:
    """``chin_follow`` in ``[0, 1]``, default 1.0: how much of his chin's
    drop the portrait's oval is let down by (see :func:`_chin_drop`). 0 is
    the mask as before day 13 -- the join at the portrait's closed-mouth
    chin -- and what the day-13 jaw band was measured against."""
    value = float(options.get("chin_follow", 1.0))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"chin_follow must be in [0, 1], got {value}")
    return value


def _nose_chin(crop_points: np.ndarray) -> float:
    """The nose tip to the chin, down the crop, in crop pixels -- the reach
    an open mouth lengthens. The crop is rolled with the face and sized
    from his resting face height, so at rest this is one number whatever
    his roll or distance; it is not the chin's own y, which the crop's
    centring (smoothed, and offset toward the eye line under pitch) moves
    for reasons that are not his jaw."""
    return float(crop_points[_CHIN, 1] - crop_points[_NOSE_TIP, 1])


def _chin_drop(nose_chin: float, rest: float | None, limit: float) -> float:
    """How far his chin has dropped from rest, in crop pixels: this frame's
    :func:`_nose_chin` less the anchor's, never negative, never over
    ``limit`` (``chin_drop_max``, the reach the mask box was extended by
    at load). 0 until the anchor knows his rest.

    Never negative, and that was measured, not assumed: a signed drop was
    tried on 2026-09-09 (03:00) for "my jaw shows when I look down / raise
    my brows", and his capture of 03:28 (brows raised, his reach 9 px
    short of rest) showed the avatar's rendered chin had NOT risen with
    his reach -- it sat 4 px above his chin, where the portrait's own
    oval already ends -- so a mask following the reach cut the avatar's
    chin 12 px short and showed 16 px of his own. The rendered chin
    follows his reach when the mouth opens (the lip gain is ~1) and does
    not under a pitch or a brow raise (the extractor carries a fraction
    of the pitch and none of the raise into the chin); the portrait's own
    oval is the better guess for those, and the 4-7 px of his chin that
    show under it are the avatar's shorter lower face -- a proportion,
    not a mask, question."""
    if rest is None:
        return 0.0
    return float(min(max(nose_chin - rest, 0.0), limit))


@dataclass(frozen=True, slots=True)
class ChinRoom:
    """What a portrait's crop leaves under its chin (see :func:`chin_room`)."""

    crop_face_height: float
    chin_row: float
    headroom: float
    chin_reach: float
    chin_drop_max: float
    crop_inside: bool


def chin_room(
    points: np.ndarray,
    *,
    image_size: tuple[int, int] | None = None,
    scale: float = CROP_SCALE,
    vy_ratio: float = CROP_VY_RATIO,
    forehead_reach: float = _OVAL_FOREHEAD_REACH,
    chin_reach_fraction: float | str = 0.0,
) -> ChinRoom:
    """The room a portrait's crop has below its chin, in crop pixels, and
    what the mask may spend it on -- the arithmetic :meth:`load_avatar`
    does, as a function of the portrait's points alone (frame pixels, the
    468 face points or more), so the importer (day 16, block B) can read
    it before a load and say whether a photograph has the room. ``headroom``
    is the rows between the chin and the crop's last usable row (less the
    oval's own measured feather tail and :data:`_MASK_EDGE_CLEARANCE`);
    ``chin_reach`` is what ``mask_chin_reach`` takes of it (a fraction, or
    ``"auto"`` for the mask's own clearance), ``chin_drop_max``
    what the jaw's drop gets (the smaller of :data:`_CHIN_DROP_MAX` of the
    face and the rest). ``crop_inside`` (with ``image_size`` as ``(width,
    height)``) says whether the whole crop square lies inside the picture:
    outside it the crop is filled with black (the portrait's load uses
    ``warpAffine``'s constant border), which is what padding at import
    prevents."""
    face = np.asarray(points, dtype=np.float32)[:FACE_POINTS, :2]
    to_crop, to_frame = crop_matrix(face, scale=scale, vy_ratio=vy_ratio)
    crop_points = (face @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
    mask = _oval_mask(crop_points, forehead_reach=forehead_reach)
    crop_face_height = _face_height(crop_points)
    chin_row = float(crop_points[_CHIN, 1])
    tail = float(np.where((mask[:, :, 0] > 0.02).any(axis=1))[0].max()) - chin_row
    edge_margin = max(tail, 0.0) + _MASK_EDGE_CLEARANCE
    headroom = max(float(CROP_SIZE - 1) - chin_row - edge_margin, 0.0)
    wanted = (
        _auto_chin_reach(crop_points)
        if chin_reach_fraction == _CHIN_REACH_AUTO
        else float(chin_reach_fraction) * crop_face_height
    )
    reach = min(wanted, headroom)
    drop = min(_CHIN_DROP_MAX * crop_face_height, headroom - reach)
    inside = True
    if image_size is not None:
        width, height = image_size
        corners = np.array(
            [[0, 0], [CROP_SIZE, 0], [CROP_SIZE, CROP_SIZE], [0, CROP_SIZE]], dtype=np.float32
        )
        frame = corners @ to_frame[:, :2].T + to_frame[:, 2]
        inside = bool(
            frame[:, 0].min() >= 0
            and frame[:, 1].min() >= 0
            and frame[:, 0].max() <= width
            and frame[:, 1].max() <= height
        )
    return ChinRoom(crop_face_height, chin_row, headroom, reach, drop, inside)


def _oval_mask(
    crop_points: np.ndarray,
    size: int = CROP_SIZE,
    *,
    jaw_shrink: float = 0.0,
    chin_drop: float = 0.0,
    forehead_reach: float = _OVAL_FOREHEAD_REACH,
) -> np.ndarray:
    """The face oval, feathered, as it sits in a crop of ``size``.

    ``crop_points`` are the 468 face points already placed in that crop's own
    pixels -- see :func:`crop_matrix`. Used two ways, on two different faces:
    once per portrait at load, on the portrait's own points -- the generated
    face is always drawn into the portrait's own canonical crop layout, so
    that oval's position never needs recomputing -- and once per frame, on
    the *driver's* points placed the same way, so the mask that is actually
    pasted with can be shaped to whoever is in front of the camera. See
    :func:`_frame_mask`, which combines the two.

    ``jaw_shrink`` pulls the oval's lower points toward the face centre by
    that fraction of face width -- nothing at the eye line, the full amount
    from the nose tip down (see ``_OVAL_JAW_SHRINK``). The portrait's mask is
    built without it; the driver's with it.

    ``chin_drop`` lets the oval's points below the mouth line down by that
    many crop pixels -- nothing at the mouth line, all of it at the chin,
    ramped between (see ``_CHIN_DROP_MAX``); a negative value would raise
    them the same way, though :func:`_chin_drop` never produces one. The
    portrait's per-frame mask is built with it under an open mouth; the
    driver's never is, his own oval already follows his jaw.

    ``forehead_reach`` is how far up from the brows toward the hairline the
    oval's top is let go (see :func:`_forehead_reach`); both ovals use the
    same value.
    """
    import cv2

    oval = crop_points[list(_FACE_OVAL)]
    brow = crop_points[list(_BROWS)]

    # Day 19: the rim and the midline together (see _MIDLINE). Every
    # statistic below -- the rim's top, the mean the jaw shrink pulls
    # toward -- is still the rim's own, so a frontal face's hull is the
    # polygon it was to the pixel; only what the hull ENCLOSES can grow,
    # and only when a midline point leaves the rim, which is a turn.
    rim = len(oval)
    trimmed = np.concatenate([oval, crop_points[list(_MIDLINE) + list(_BROW_LINE)]]).copy()
    brow_top = float(brow[:, 1].min())
    oval_top = float(trimmed[:rim, 1].min())
    ceiling = brow_top - (brow_top - oval_top) * forehead_reach
    if forehead_reach > 1.0:
        # Day 16 (C1): past the mesh's own top the forehead arc is lifted
        # toward the ceiling -- all of it at the arc's top, none at the
        # brow line, linear between -- so the hull reaches the hairline.
        span = max(brow_top - oval_top, 1.0)
        weight = np.clip((brow_top - trimmed[:, 1]) / span, 0.0, 1.0)
        trimmed[:, 1] -= weight * (oval_top - ceiling)
    else:
        np.maximum(trimmed[:, 1], ceiling, out=trimmed[:, 1])

    if chin_drop != 0.0:
        mouth_y = float(crop_points[list(_MOUTH_CORNERS), 1].mean())
        chin_y = float(crop_points[_CHIN, 1])
        weight = np.clip((trimmed[:, 1] - mouth_y) / max(chin_y - mouth_y, 1.0), 0.0, 1.0)
        trimmed[:, 1] += weight * chin_drop

    face_width = max(float(crop_points[:, 0].max() - crop_points[:, 0].min()), 1.0)
    if jaw_shrink > 0.0:
        # The crop is rolled with the face (see _similarity), so "down" is +y
        # here whatever his head does in the frame.
        eye_y = float(crop_points[list(_LEFT_EYE_CORNERS + _RIGHT_EYE_CORNERS), 1].mean())
        nose_y = float(crop_points[_NOSE_TIP, 1])
        weight = np.clip((trimmed[:, 1] - eye_y) / max(nose_y - eye_y, 1.0), 0.0, 1.0)
        inward = trimmed[:rim].mean(axis=0) - trimmed
        inward /= np.maximum(np.linalg.norm(inward, axis=1), 1e-6)[:, None]
        trimmed = trimmed + inward * (weight * face_width * jaw_shrink)[:, None]
    shrink = max(1, int(face_width * _OVAL_SHRINK))
    feather = max(3, int(face_width * _OVAL_FEATHER) | 1)
    if _PROFILE_PUSH > 0.0 and len(trimmed) > rim:
        # Day 20: the profile's midline points out by the erosion's radius
        # (see _PROFILE_PUSH); a frontal face has none outside the rim.
        trimmed = _push_profile(trimmed, rim, ((shrink | 1) // 2) * _PROFILE_PUSH, reference=oval)

    filled = np.zeros((size, size), dtype=np.uint8)
    cv2.fillConvexPoly(filled, cv2.convexHull(trimmed.astype(np.int32)), 255)
    eroded = cv2.erode(
        filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink | 1, shrink | 1))
    )
    softened = cv2.GaussianBlur(eroded, (feather, feather), 0)
    return (softened.astype(np.float32) / 255.0)[:, :, None]


def _push_profile(
    points: np.ndarray, rim: int, radius: float, reference: np.ndarray | None = None
) -> np.ndarray:
    """``points`` with every point past index ``rim`` that lies outside the
    convex hull of the rim -- ``reference``, or the first ``rim`` points --
    moved outward, along the ray from the rim's mean, by ``radius`` --
    less, in proportion, when it is less than ``radius`` outside (see
    :data:`_PROFILE_PUSH`). Points inside the hull, and the first ``rim``
    points, are returned as they are; a ``radius`` of 0 returns the array
    itself. ``reference`` is the rim as read off the face, before the
    mask's own adjustments (the forehead ceiling, the chin drop, the jaw
    shrink), so what is pushed, and where to, depends on the face alone."""
    if radius <= 0.0 or len(points) <= rim:
        return points
    import cv2

    rim_points = points[:rim] if reference is None else reference
    hull = cv2.convexHull(np.ascontiguousarray(rim_points, dtype=np.float32))
    centre = rim_points.mean(axis=0)
    pushed = points.copy()
    for index in range(rim, len(points)):
        x, y = float(points[index, 0]), float(points[index, 1])
        outside = -cv2.pointPolygonTest(hull, (x, y), True)
        if outside <= 0.0:
            continue
        direction = points[index] - centre
        length = float(np.linalg.norm(direction))
        if length <= 1e-6:
            continue
        pushed[index] = points[index] + direction / length * min(outside, radius)
    return pushed


# Day 19 (his screenshot of a fast shake, the white streak down the cheek;
# day 18 had measured the same thing under the turn views): the paste mask
# is the two ovals, and the network draws the face where the pose puts it
# inside the crop -- where the drawn face falls short of the mask (a turn
# narrows it, a fast move lags it), the mask covers the photograph's own
# flat background and pastes it, white, over his cheek. So the photograph's
# background colour is read once at load from its border, and at every
# paste the mask is cut wherever the render is that colour AND joined to the
# box's border (a highlight on the nose is that colour too, and is not
# joined to anything). A frontal face fills its own oval, so nothing is cut
# there and the paste is the one it was; a photograph without a flat border
# (a room) gets no guard, as before.
_BACKGROUND_RING = 0.03  # of the picture's size, the border read at load
_BACKGROUND_FLAT = 12.0  # the border's spread (median absolute deviation) allowed
_BACKGROUND_NEAR = 30.0  # colour distance under which a render pixel is background
# The cut reaches this far into the drawn face's own edge: the render's
# face-to-background transition is a pixel or two of skin mixed with white,
# past the distance above yet pale on his cheek (measured on his turn clip,
# f452/f459: a thin light rim along the profile with the distance alone).
_BACKGROUND_REACH = 2
_BACKGROUND_SOFT = 5  # the cut's own feather, px (odd)
# The guard comes on with the turn -- nothing under the first, all of it
# from the second, degrees of yaw from the anchor's rest -- so a frontal
# frame's paste is the one it was to the byte: measured on his turn clip,
# a frontal frame's mask meets the render's own edge at the jaw for a pixel
# or two, and the guard would trim those; the nose leaves the rim past 30.
_BACKGROUND_YAW = (12.0, 20.0)


def _background_guard(options: dict[str, Any]) -> bool:
    """``background_guard`` (default on): whether the paste stops where the
    render shows the photograph's own background."""
    return bool(float(options.get("background_guard", 1.0)) > 0.0)


# Day 20: the room past the portrait's oval (see _turn_room) comes on over
# its own, shorter ramp: nothing under 12 degrees of yaw from the rest (a
# frontal frame's paste is the one it was, to the byte), all of it from 16
# -- his press at 17 degrees had the far brow's end 7.5 crop px past the
# portrait's oval already, and under the guard's ramp (12 to 20) the room
# there was only two thirds of the paste, his brow showing through the
# other third. The room is pasted only where the render is the drawn face,
# by the guard's keep map in full, so it needs none of the guard's own
# caution about the frontal jaw.
_ROOM_YAW = (12.0, 16.0)
# Day 22 evening: the room reaches no higher than this fraction of the
# forehead above the brow line (the far brow's end at a turn is within it).
_ROOM_BROW_MARGIN = 0.2


def _room_weight(loaded: _Portrait, head_pose: HeadPose | None) -> float:
    """How much of the room past the portrait's oval this frame may take:
    0.0 within :data:`_ROOM_YAW`'s first angle of the rest yaw, 1.0 past
    its second, linear between; 0.0 without a background or a pose."""
    if loaded.background is None or head_pose is None:
        return 0.0
    yaw = abs(float(head_pose.yaw) - (loaded.rest_yaw or 0.0))
    return float(_soft_gate(yaw, _ROOM_YAW))


def _background_weight(loaded: _Portrait, head_pose: HeadPose | None) -> float:
    """How much of the guard applies this frame: 0.0 within
    :data:`_BACKGROUND_YAW`'s first angle of the rest yaw, 1.0 past its
    second, linear between -- and, day 22, the same over the pitch from the
    rest pitch (:data:`_BACKGROUND_PITCH`: a look down draws the chin higher
    in the crop and the photograph's background can show under it); the
    larger of the two. 0.0 without a background or a pose."""
    if loaded.background is None or head_pose is None:
        return 0.0
    yaw = abs(float(head_pose.yaw) - (loaded.rest_yaw or 0.0))
    pitch = abs(float(head_pose.pitch) - (loaded.rest_pitch or 0.0))
    return float(max(_soft_gate(yaw, _BACKGROUND_YAW), _soft_gate(pitch, _BACKGROUND_PITCH)))


# Day 22: the photograph's background under the drawn chin. On a look down
# the network draws the chin higher in the crop and the rows under it are
# the photograph's white; the guard above is gated by the turn (and now the
# tilt), and the keep it cuts with wants the background JOINED to the patch's
# border -- a blob under the chin is not. So the rows below the portrait's
# own chin have a rule of their own, on every frame, with nothing gated and
# nothing downloaded: a pixel there within _BACKGROUND_NEAR of the
# background colour is background, grown by _BACKGROUND_REACH and feathered
# by _BACKGROUND_SOFT like the guard's cut -- :func:`_chin_keep` on the CPU
# path, :meth:`LivePortraitEngine._chin_keep_gpu` on the device, applied to
# the mask before it is warped (CPU) or to the warped alpha through the
# face's own sampling grid (GPU). Skin under a chin is never within 30 grey
# levels of a flat white or grey background, so nothing of the face is cut.
_BACKGROUND_PITCH = (12.0, 20.0)


def _tilt_weight(loaded: _Portrait, head_pose: HeadPose | None) -> float:
    """Day 22: how far the head is tilted from its rest pitch, as the guard
    counts it: 0.0 within :data:`_BACKGROUND_PITCH`'s first angle, 1.0 past
    its second, linear between; 0.0 without a pose."""
    if head_pose is None:
        return 0.0
    pitch = abs(float(head_pose.pitch) - (loaded.rest_pitch or 0.0))
    return float(_soft_gate(pitch, _BACKGROUND_PITCH))


def _chin_keep(face: np.ndarray, background: np.ndarray, floor: float) -> np.ndarray | None:
    """What the mask may keep of ``face`` (a BGR render patch) in the rows
    from ``floor`` (a patch row, the portrait's chin) down: 0 where the
    render is the photograph's background, 1 elsewhere and above ``floor``,
    feathered; ``None`` when no such row holds background."""
    import cv2

    height = face.shape[0]
    start = int(max(0, math.floor(floor)))
    if start >= height:
        return None
    distance = np.max(np.abs(face[start:].astype(np.float32) - background), axis=2)
    flat = (distance < _BACKGROUND_NEAR).astype(np.uint8)
    if not flat.any():
        return None
    outside = np.zeros(face.shape[:2], dtype=np.uint8)
    outside[start:] = flat
    reach = 2 * _BACKGROUND_REACH + 1
    grown = cv2.dilate(outside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach)))
    keep = cv2.GaussianBlur((1 - grown).astype(np.float32), (_BACKGROUND_SOFT, _BACKGROUND_SOFT), 0)
    return np.asarray(keep, dtype=np.float32)


def _background_colour(portrait: np.ndarray) -> np.ndarray | None:
    """The photograph's flat background colour (BGR, float32) from its
    border -- the top ring and the upper halves of the two sides, where a
    portrait's shoulders never are -- or ``None`` when that border is not
    one colour (a room behind the face)."""
    height, width = portrait.shape[:2]
    ring = max(2, round(min(height, width) * _BACKGROUND_RING))
    top = portrait[:ring].reshape(-1, 3)
    left = portrait[: height // 2, :ring].reshape(-1, 3)
    right = portrait[: height // 2, width - ring :].reshape(-1, 3)
    pixels = np.concatenate([top, left, right]).astype(np.float32)
    colour = np.median(pixels, axis=0)
    spread = np.median(np.abs(pixels - colour), axis=0).max()
    if spread > _BACKGROUND_FLAT:
        return None
    return colour.astype(np.float32)


def _background_keep(face: np.ndarray, background: np.ndarray) -> np.ndarray | None:
    """What the mask may keep of ``face`` (a BGR render patch): 0 where the
    render is the photograph's background joined to the patch's border,
    rising to 1 over :data:`_BACKGROUND_NEAR`; ``None`` when no background
    reaches the patch at all, so the caller pays nothing."""
    import cv2

    distance = np.max(np.abs(face.astype(np.float32) - background), axis=2)
    flat = (distance < _BACKGROUND_NEAR).astype(np.uint8)
    if not flat.any():
        return None
    count, labels = cv2.connectedComponents(flat, connectivity=4)
    if count <= 1:
        return None
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    joined = np.unique(border[border > 0])
    if joined.size == 0:
        return None
    outside = np.isin(labels, joined).astype(np.uint8)
    reach = 2 * _BACKGROUND_REACH + 1
    grown = cv2.dilate(outside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach)))
    keep = cv2.GaussianBlur((1 - grown).astype(np.float32), (_BACKGROUND_SOFT, _BACKGROUND_SOFT), 0)
    return np.asarray(keep, dtype=np.float32)


def _mask_bbox(
    mask: np.ndarray,
    margin: int = _MASK_BOX_MARGIN,
    *,
    reach_below: int = 0,
    room_aside: float = 0.0,
) -> tuple[int, int, int, int]:
    """Where ``mask`` is non-zero, with a small margin, as integer crop pixels.

    ``reach_below`` extends the box's bottom edge by that many more pixels
    (day 13): the per-frame mask under an open mouth reaches below the
    portrait's own oval by up to the chin drop's limit, and the paste
    warps only into this box. ``room_aside`` (day 20) widens it to either
    side by that fraction of its own width, for the drawn face's far side
    at a turn (see :data:`_MASK_TURN_ROOM`).

    This is the whole reason the mask is precomputed once at load: its box is
    a fixed rectangle inside the crop, far smaller than the crop's own 512
    square, and every frame's paste warps only into it -- not into the whole
    square, most of which is the portrait's own background or shoulders.
    """
    size = mask.shape[0]
    nonzero = np.argwhere(mask[:, :, 0] > (1.0 / 255.0))
    if nonzero.size == 0:
        return 0, 0, size, size
    y0 = max(int(nonzero[:, 0].min()) - margin, 0)
    y1 = min(int(nonzero[:, 0].max()) + 1 + margin + reach_below, size)
    x0 = max(int(nonzero[:, 1].min()) - margin, 0)
    x1 = min(int(nonzero[:, 1].max()) + 1 + margin, size)
    if room_aside > 0.0:
        room = round((x1 - x0) * room_aside)
        x0, x1 = max(x0 - room, 0), min(x1 + room, size)
    return x0, y0, x1, y1


def _turn_room(portrait: np.ndarray, driver: np.ndarray, weight: float) -> np.ndarray | None:
    """Day 20: what the paste may add beyond the portrait's static oval at a
    turn -- the driver's own oval, up to ``weight`` (see
    :func:`_room_weight`), where the portrait's oval falls short of it;
    ``None`` at weight 0, so a frontal frame's paste is the one it was.
    The portrait's oval is the frontal photograph's and keeps its
    background and shoulders out; at a turn the drawn face's far side
    reaches past that oval's edge (his press at +28 degrees: the drawn
    temple 1 px beyond his far brow's end, the oval 11 px short of it) and
    his own brow showed in the gap. What this adds is pasted only where the
    render is the drawn face, not its background -- :meth:`_paste` applies
    the guard's keep map to it in full -- so the background stays out as
    before, by the guard instead of the oval; a photograph without a flat
    background has no guard, and no room."""
    if weight <= 0.0:
        return None
    extra = np.minimum(np.maximum(weight - portrait, 0.0), driver)
    return np.asarray(extra, dtype=np.float32)


def _frame_mask(
    portrait_mask: np.ndarray,
    driver_crop_points: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    """The paste mask for one frame -- :func:`_frame_masks`'s first."""
    return _frame_masks(portrait_mask, driver_crop_points, **kwargs)[0]


def _frame_masks(
    portrait_mask: np.ndarray,
    driver_crop_points: np.ndarray,
    *,
    jaw_shrink: float = 0.0,
    portrait_crop_points: np.ndarray | None = None,
    chin_drop: float = 0.0,
    forehead_reach: float = _OVAL_FOREHEAD_REACH,
    chin_reach: float = 0.0,
    driver_forehead_reach: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The paste mask for one frame -- the portrait's oval **and** the
    driver's -- with the two ovals it was made of (day 20, for
    :func:`_turn_room`).

    ``driver_forehead_reach`` (day 17) gives the driver's oval its own
    reach, ``None`` for the same as the portrait's: with the avatar's hair
    pasted over the driver's, the driver's hairline no longer bounds the
    join and his oval is let up as far as the portrait's may go.

    ``jaw_shrink`` goes to the driver's oval alone (see :func:`_oval_mask`):
    the intersection then follows the driver's pulled-in jaw, and the
    portrait's own mask, built once at load, is unchanged -- its box still
    bounds every frame's paste.

    ``chin_drop`` goes to the portrait's oval alone (day 13): over half a
    pixel, and given ``portrait_crop_points``, the portrait's half is
    rebuilt with its jaw let down by that much (see :func:`_oval_mask`),
    so the intersection can follow the avatar's opened jaw as far as the
    driver's own oval goes; ``portrait_mask`` itself is untouched, and its
    box, extended below at load by the drop's limit, still bounds the
    paste (see :func:`_mask_bbox`).

    ``chin_reach`` (crop pixels, see :func:`_mask_chin_reach`) lets *both*
    ovals down at the chin, so the join sits under his chin rather than
    across it; the portrait's half is rebuilt whenever either it or the
    drop is over half a pixel.

    ``portrait_mask`` is the static mask computed once at load from the
    portrait's own landmarks (:func:`_oval_mask`) -- it keeps the portrait's
    own background and shoulders out, whatever the driver does. On its own it
    is not enough: it is shaped to the *portrait's* forehead-to-hairline and
    jaw-to-chin distances, and the driver's face rarely has the same
    proportions. Warped into a driver whose forehead is shorter, its top
    lands in the driver's hair rather than on skin -- structure, which colour
    correction cannot fix, seen as a band at the hairline. Warped into a
    driver whose jaw sits lower, its bottom stops short of the chin and the
    portrait's face reads as floating above the driver's own.

    A second oval, built the same way from the *driver's* crop-space
    landmarks (see :func:`_oval_mask` — the same function, a different set of
    points), has the matching problem in reverse: it would fit the driver's
    face perfectly but say nothing about where the *portrait's* own picture
    actually has content.

    The intersection is the only combination that can never show either
    defect: it is at most as large as each mask alone, so it cannot reach
    past the portrait's own background (the static mask's job) and it cannot
    reach past the driver's own hairline or jaw (the driver mask's job) --
    exactly the rule the warp engine's own composite already follows for the
    same reason (``engine_warp._destination_mask``, "the join happens over
    skin on both sides").
    """
    portrait = portrait_mask
    if abs(chin_drop + chin_reach) > 0.5 and portrait_crop_points is not None:
        portrait = _oval_mask(
            portrait_crop_points,
            size=portrait_mask.shape[0],
            chin_drop=chin_drop + chin_reach,
            forehead_reach=forehead_reach,
        )
    driver = _oval_mask(
        driver_crop_points,
        jaw_shrink=jaw_shrink,
        chin_drop=chin_reach,
        forehead_reach=forehead_reach if driver_forehead_reach is None else driver_forehead_reach,
    )
    combined: np.ndarray = np.minimum(portrait, driver)
    return combined, driver, portrait


def _fit_lighting_plane(difference: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Least-squares plane through the masked difference, per colour channel.

    Identical arithmetic to ``engine_warp._fit_lighting_plane``: three
    unknowns per channel -- a constant and two gradients -- over a downscaled
    box, which is a handful of sums and one 3x3 solve.
    """
    height, width = coverage.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
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
        coefficients = np.linalg.solve(normal, targets)
    except np.linalg.LinAlgError:
        total = float(coverage.sum())
        if total < 1e-6:
            return np.zeros((height, width, difference.shape[2]), dtype=np.float32)
        mean = difference.reshape(-1, difference.shape[2]).sum(axis=0) / total
        return np.broadcast_to(mean.astype(np.float32), (height, width, difference.shape[2]))

    field: np.ndarray = np.einsum("kij,kc->ijc", np.stack(basis), coefficients.astype(np.float32))
    return field.astype(np.float32)


# ---------------------------------------------------------------------------
# Temporal smoothing of the extracted motion (day 7, 2026-09-05)
# ---------------------------------------------------------------------------


def _motion_smoothing(options: dict[str, Any]) -> float:
    """The EMA weight given to the *previous* state in :func:`_ema_`, from
    ``options["motion_smoothing"]`` -- 0.0 (default) is off: every line in
    :meth:`LivePortraitEngine._drive` that calls :func:`_ema_` becomes a
    no-op (see that function's own docstring for why it is exactly a
    no-op, not just a small effect).

    Why this exists: a second clip of the owner's own (``recordings/
    driving.mkv``, re-recorded 2026-09-05) measured the avatar wandering
    while he sat still and did not -- his eye ratio held to 0.314..0.320
    over 120 frames, the avatar's read 0.291..0.365; his brow 0.237..0.247
    against the avatar's 0.161..0.213 -- with the crop itself steady on
    those same frames (see :func:`LivePortraitEngine._driver_face_height`).
    That is the motion extractor's own frame-to-frame noise, passed
    straight to the portrait's keypoints; nothing about *this* driver is
    moving that fast.

    Effective lag at 30 fps is about ``-1 / ln(a)`` frames: 0.5 -> 1.4
    frames, ~48 ms; 0.65 -> 2.3 frames, ~77 ms. Baked into the CUDA graph
    like every other option here (see ``eye_gain``) -- a live change needs
    the avatar reloaded, it cannot be swept without one.

    Must be in ``[0, 1)``: at 1.0 the state would never move again.
    """
    a = float(options.get("motion_smoothing", 0.0))
    if not (0.0 <= a < 1.0):
        msg = f"motion_smoothing must be in [0, 1), got {a}"
        raise ValueError(msg)
    return a


def _translation_gain(options: dict[str, Any]) -> tuple[float, float]:
    """How much of the extractor's *relative translation* (``t_now - t_d0``,
    x and y) reaches the portrait -- ``translation_gain_x`` /
    ``translation_gain_y``, each in ``[0, 1]``, default 1.0 (all of it, as
    before). At 0 the avatar's face sits exactly where the crop puts it.

    Why this exists (day 11, 2026-09-06, his "when I raise my brows the
    avatar lowers its eyes"): in LivePortrait's own pipeline the driving
    video is cropped once, so ``t`` carries where the driver's head is.
    This engine re-centres the crop on his own face every frame, so his
    head's position is already absorbed before the extractor sees the
    crop, and ``t_now - t_d0`` is left carrying only what the extractor
    *mistakes* for translation. Measured on his brow raise (natural.mkv
    1564-1600, and the same on a live capture): his face moved UP 12 px
    in the frame while the extractor's relative ``t.y`` read +0.035
    units -- 9 px in the crop, ~18 in the frame, DOWN -- and the avatar's
    eyes landed 13-17 px below his. The extractor splits "brows up" into
    "face down" (``t``) plus "brows up" (the brow keypoints' expression);
    the brow gate drops the second half and this engine kept the first.
    The same reading explained 6-8 of the 14 px the avatar's eyes still
    slid at a -12.6 degree nod after the pitch pivot (day 10), and
    reversed sign past -16 degrees. Both gains are baked at capture like
    every other option here.
    """
    gx = float(options.get("translation_gain_x", 1.0))
    gy = float(options.get("translation_gain_y", 1.0))
    for name, value in (("translation_gain_x", gx), ("translation_gain_y", gy)):
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    return gx, gy


def _scale_gain(options: dict[str, Any]) -> float:
    """How much of the extractor's *relative scale* (``scale_now /
    scale_d0``) reaches the portrait -- ``scale_gain`` in ``[0, 1]``,
    default 0.0: the portrait's own scale, always. At 1.0 the old line
    (``scale_s * scale_now / scale_d0``) is what runs, bit-exact.

    Why this exists (day 13, 2026-09-07, his "o" and wide-open captures of
    03:24, and the fp32 extractor of day 12): with the extractor running
    the function it was trained as, its scale head *responds* -- and on
    driving.mkv's wide-open frames it read 1.081 of the anchor while the
    avatar's inter-ocular distance grew to 1.087 of its rest, his own
    1.007, correlation 0.98 with his mouth ratio. In fp16 the head sat at
    1.010, which is why nobody saw this before. The crop is already sized
    from his resting face height by head-pose distance
    (:meth:`LivePortraitEngine._driver_face_height`), so a real change in
    his distance never reaches the extractor as a bigger face; what its
    relative scale carries is what it *mistakes* for one -- an open mouth
    is a taller face, read as a nearer one. The day-11 argument for
    ``translation_gain_y``, applied to ``scale_new``: the gained ratio is
    ``1 + (scale_now / scale_d0 - 1) * scale_gain``, folded into the same
    multiply, baked at capture like every other option here.
    """
    g = float(options.get("scale_gain", 0.0))
    if not (0.0 <= g <= 1.0):
        raise ValueError(f"scale_gain must be in [0, 1], got {g}")
    return g


# Day 20 (2026-09-18), his "the AI face cannot keep up with a slightly
# quick turn": 0.85 was day 9's answer to the extractor's SCALE breathing
# while he sat still, and the scale has not been applied since day 13
# (scale_gain 0) -- only the translation was left carrying the weight,
# and with it the drawn head's position lagged six frames: the drawn nose
# 20-32 px off his on the turn clip's fast return (30-45 live at 21 fps,
# the filter counts frames), the mask leading the drawn face by that much.
# Measured before the change (replay): at 0.3 the nose is 8 px off on the
# same frames and the head follows within a frame; sitting still the
# drawn nose's own frame-to-frame movement is the same (p90 0.45-0.49 px
# at the camera, 1.9 at the screen, his own head's) and avatar-minus-his
# jitter is lower (p90 0.40 -> 0.29 px); the expression (ear, mar, mouth
# width, brow: r and gain) identical on driving.mkv. His word, 04:2x.
_DEFAULT_SIZE_SMOOTHING = 0.3


def _size_smoothing(options: dict[str, Any]) -> float:
    """The EMA weight on the previous frame's extractor *scale and
    translation* (``size_smoothing``, default
    :data:`_DEFAULT_SIZE_SMOOTHING`; 0.0 off) -- applied in
    :meth:`LivePortraitEngine._drive` on top of ``motion_smoothing``, which
    stays at 0.0 for the expression. See the comment there."""
    a = float(options.get("size_smoothing", _DEFAULT_SIZE_SMOOTHING))
    if not (0.0 <= a < 1.0):
        raise ValueError(f"size_smoothing must be in [0, 1), got {a}")
    return a


# ---------------------------------------------------------------------------
# The rotation's pivot (day 10, Fable 5.1's review of Step 1, 2026-09-06)
# ---------------------------------------------------------------------------

_ROTATION_PIVOTS = ("none", "centroid", "eyes")


def _rotation_pivot(options: dict[str, Any]) -> str:
    """Where the *pitch* component of the relative rotation turns the
    portrait's canonical keypoints (``x_c_s``) about, from
    ``options["rotation_pivot"]`` -- one of :data:`_ROTATION_PIVOTS`,
    default ``"none"``.

    **The bug this answers.** ``x_c_s @ rot_new`` rotates about the
    keypoint space's own origin, not a point on the face. LivePortrait's
    own pipeline crops the driving video once for the whole clip, so a
    driver who pitches about his own neck also slides inside that fixed
    crop -- his relative translation (``t_now - t_d0``) carries the slide,
    and rotation and translation come from the same real head and agree.
    This engine re-centres the crop on the driver's own rigid points every
    frame (see :func:`_crop_geometry`), which removes his slide from the
    driver side (``t_now - t_d0`` stays near zero even at a steep pitch,
    measured on ``natural.mkv``) -- but the portrait's own keypoints still
    swing about the origin under the same relative rotation, with nothing
    left to cancel it: the whole "foreign face" at a slight downward tilt,
    day 10's fourth item.

    **Why the fix is one axis, not the whole rotation.** A first version
    (2026-09-06, kept a few hours) rotated the *whole* relative rotation
    (pitch, yaw and roll together) about a fixed point on the face. On
    ``natural.mkv`` it worked -- the eye anchor held near its rest position
    through a nod that used to slide it 29 px. On ``driving.mkv`` it broke
    head turns badly -- the turn-lag roughly doubled at every sampled
    frame. The reason is algebraic, not the network's: rotating about a
    pivot ``p`` is ``x R + p(I - R)`` -- an origin rotation plus the *same*
    translation vector ``p(I - R)`` added to every keypoint, whatever axis
    ``R`` turns on. A point chosen so pitch turns the face correctly (the
    eyes' own mean, well below the keypoint origin) is offset from the
    origin along the *other* axis too (its own depth, in this coordinate
    space) -- so the very same translation that fixes a nod adds a
    sideways shift to a turn, because ``R`` was yaw that time, not pitch.
    One point cannot serve both axes at once; the translation, split by
    axis, can.

    **The fix.** ``"none"`` (default) is today's math, unchanged,
    bit-for-bit -- :meth:`LivePortraitEngine._drive` keeps computing
    ``rel_rot @ R_s`` first and multiplying ``x_c_s`` by it in one step,
    exactly as before. ``"centroid"``/``"eyes"`` add only the *pitch*
    delta's own translation on top of the ordinary, untouched, whole-
    rotation ``x_c_s @ rel_rot`` -- so a pure yaw (``rel_pitch == 0``)
    reduces the added translation to exactly zero and is identical to
    ``"none"``, while a pure pitch reproduces the earlier, whole-rotation
    pivot exactly (verified: ``get_rotation_matrix`` returns
    ``(rot_z rot_y rot_x)^T``, so with yaw/roll at the anchor's own values
    the whole relative rotation collapses to the pitch-alone one). The
    pivot point itself -- :data:`EYE_KEYPOINTS`'s own mean, or all 21
    keypoints' -- is set once at load into ``loaded.pivot`` (see
    :meth:`LivePortraitEngine.load_avatar`), since it is computed from
    ``x_c_s``, which never changes after load; see :func:`_pivot_z` for
    the translation's own z component. Swept on the portrait alone
    (``scripts/`` scratch probe, ``recordings/day35-swing-*``, not the
    owner's face): pivoting the pitch delta on the eyes held the eye
    anchor within a couple of pixels of its own rest position through a
    +-15 degree pitch sweep while the chin still moved (a turn, not a
    slide), matched a pure-yaw ``"none"`` render exactly at every tested
    angle (0 px difference), and at a mixed point (pitch and yaw together)
    sat within 3 px of ``"none"`` on the yaw measure and within 0.1 px of
    the whole-rotation pivot on the pitch measure -- the two axes came
    apart cleanly. The centroid pivot only partly closed the pitch gap in
    every measurement so far and is kept for comparison, not expected to
    be the one used.
    """
    pivot = str(options.get("rotation_pivot", "none"))
    if pivot not in _ROTATION_PIVOTS:
        msg = f"rotation_pivot must be one of {_ROTATION_PIVOTS}, got {pivot!r}"
        raise ValueError(msg)
    return pivot


def _crop_shift_x(options: dict[str, Any]) -> float:
    """``crop_shift_x`` in crop pixels, ``[-40, 40]``, default 0: a constant
    sideways shift of the driver's crop along the face's own horizontal
    (+ toward the picture's right at zero roll). Day 16 (C2): the turn
    lag measured on day 14 -- 21 px at p90 -- was mostly a STATIC offset:
    on driving.mkv's frontal frames the avatar's eyes sit 7.5 px and its
    nose 13-14 px to the right of his, and the turns' own lag is 4-7 px
    either way once that is taken off (recordings/day16/turn-lag). A
    constant shift of the crop moves the avatar by the same amount and
    changes nothing relative (the anchor is learned in the shifted crop
    too); the calibration measures it from the avatar's eyes against his
    on the still frames and writes it to the driver profile."""
    shift = float(options.get("crop_shift_x", 0.0))
    if not (-40.0 <= shift <= 40.0):
        msg = f"crop_shift_x must be in [-40, 40] crop px, got {shift}"
        raise ValueError(msg)
    return shift


def _yaw_pivot_z(options: dict[str, Any]) -> float:
    """``yaw_pivot_z`` in ``[0, 1]``, default 0.0 (off, bit-exact): the depth
    behind the face, in keypoint units, that the *yaw* delta turns about
    -- day 10's Step 4, built on day 16 (block C2) for the turn lag open
    since day 4 (the avatar's nose 12-21 px behind his on a turn). The
    same mechanism as the pitch pivot (:func:`_rotation_pivot`), on its
    own axis: a point on the head's vertical axis behind the face, so a
    turn about it carries the face sideways the way a real head on a
    neck does, instead of spinning the keypoints about the space's
    origin. Positive z is behind the face in this space (the eyes'
    pivot sits at z = -0.113, in front). Independent of
    ``rotation_pivot``: either, both or neither."""
    z = float(options.get("yaw_pivot_z", 0.0))
    if not (0.0 <= z <= 1.0):
        msg = f"yaw_pivot_z must be in [0, 1], got {z}"
        raise ValueError(msg)
    return z


def _pivot_z(options: dict[str, Any]) -> bool:
    """Whether the pitch-pivot's own depth (z, in keypoint space) is kept
    in the translation :func:`_rotation_pivot` adds, from
    ``options["pivot_z"]`` -- default ``False`` (dropped).

    LivePortrait's own pipeline never moves a keypoint by a *translation*
    in z (``t_new``'s own ``torch.cat`` zeros it); this translation is new,
    so keeping or dropping its z component was a measurement, not a given.
    Swept on the portrait alone (day 10, second review): at a +15 degree
    pitch, dropping z left the eye anchor 1.41 px from its rest position,
    keeping it left 0.88 px -- within a pixel of each other, so the tie
    goes to the pipeline's own convention, and z is dropped by default.
    """
    return bool(options.get("pivot_z", False))


def _ema_(state: Any, new: Any, alpha: float) -> Any:
    """Blend ``new`` into ``state`` in place -- ``state = state*alpha +
    new*(1-alpha)`` -- and return it.

    In place (``mul_``/``add_``, not a fresh tensor) on purpose: this is
    what makes it capturable into a CUDA graph. The graph records the
    *operation*, not a value; every replay reads whatever ``state``'s own
    memory currently holds, blends in this call's fresh ``new``, and writes
    the result back into that same memory for the *next* replay to read --
    which is the entire mechanism the smoothing runs on. ``alpha`` is
    assumed already validated (see :func:`_motion_smoothing`, the only
    caller that reads it from an option); at ``alpha=0.0``, ``state*0.0`` is
    exactly ``0.0`` and ``0.0 + new*1.0`` is exactly ``new`` for any finite
    value, so this is bit-exact, not merely close.
    """
    state.mul_(alpha).add_(new, alpha=1.0 - alpha)
    return state


# ---------------------------------------------------------------------------
# Region-masked expression (day 8, 2026-09-06)
# ---------------------------------------------------------------------------

# Which of LivePortrait's 21 implicit keypoints move the eyes on this
# network -- measured, not guessed: the portrait driving itself, each
# keypoint's expression nudged +-0.04 on x/y/z in turn, the rendered face
# read back by MediaPipe. Keypoints 11, 13, 15, 16 (mostly their y axis)
# move the eye ratio +-0.16..0.20 each with under 0.004 leaking into the
# mouth; the same four keypoints on a second portrait
# (``my_avatar_neural3``). See docs/development/ROADMAP.md, "Day 7, later",
# for the full table (mouth: 17, 19, 20; brows: 1, 2, 3 -- not masked here).
EYE_KEYPOINTS = (11, 13, 15, 16)


def _eye_expression_gain(options: dict[str, Any]) -> float:
    """How much of the motion extractor's own reading of the eye keypoints
    (:data:`EYE_KEYPOINTS`) reaches the portrait, from
    ``options["eye_expression_gain"]`` -- multiplies ``exp_now - exp_d0`` on
    those four rows only, in :meth:`LivePortraitEngine._drive`.

    Default **0.0**: the extractor is cut out of the eyes entirely, and they
    are driven only by the eye retargeting network's own measured-EAR target
    (see :func:`_eye_target`). On the owner's own natural-posture clip
    (``recordings/natural.mkv``) his eyes never closed while the avatar's
    did -- while he looked at the screen, still, his eye ratio held
    0.292..0.381 against the avatar's 0.180..0.292; talking, his
    0.249..0.329 against the avatar's 0.131..0.299 -- tracked to those same
    four keypoints reading his gaze and his mouth motion as eye closure
    (see ROADMAP day 7's keypoint table and the frame-by-frame numbers).
    ``1.0`` is day 7 and before's behaviour: the extractor's own reading of
    those keypoints passes through unweighted, same as every other one.

    Baked into the CUDA graph like every other option here (see
    ``motion_smoothing``) -- a live change needs the avatar reloaded.

    Must be in ``[0, 1]``: there is no meaning to weighting the extractor's
    own reading negatively or beyond itself.
    """
    g = float(options.get("eye_expression_gain", 0.0))
    if not (0.0 <= g <= 1.0):
        msg = f"eye_expression_gain must be in [0, 1], got {g}"
        raise ValueError(msg)
    return g


def _eye_mask_output(options: dict[str, Any]) -> bool:
    """Whether the eye retargeting network's own delta (see ``_drive``'s
    ``eye_gain`` block) is confined to :data:`EYE_KEYPOINTS` before it is
    added to ``x_d``, from ``options["eye_mask_output"]``. Default ``True``.

    ``False`` lets the network's whole delta apply, unmasked -- the day 4
    behaviour, kept only so the two day-8 changes (this, and
    :func:`_eye_expression_gain`) can be measured apart from each other.
    """
    return bool(options.get("eye_mask_output", True))


# Day 9, later (2026-09-06): the mouth's own axes, from nudging each mouth
# keypoint on each axis on the portrait and reading the result back with
# MediaPipe (the ROADMAP has the table). The *x* of keypoints 14, 17, 20 and
# the *y* of 7 move one mouth corner against the other -- a tilt, nothing
# else of use: +-0.05 on 17's x tilts the mouth 0.06-0.12 of inter-ocular
# and changes its width 2-6%. His live verdict was "it smiles to one side,
# like a cynical smile": measured, the avatar's mouth tilt moved twice as
# far as his on the closed smile (slope 1.96) and wandered +-0.025 of
# inter-ocular while he talked with r 0.43 against his own -- not his
# asymmetry, the extractor's. ``mouth_lateral_gain`` (default 0.0) weights
# exactly those four components. The *y* of 14 is the smile itself: -0.05
# there raises both corners evenly (+0.078 / +0.089 of inter-ocular) and
# widens the mouth 27%, with the opening and the eyes untouched;
# ``mouth_smile_gain`` (default 1.0) weights it, for a smile the extractor
# spreads too far (his +15% at the top of a closed smile read as +26%).
# The *z* of 17, 19, 20 also moves the width (20's z: +32% / -58%) but with
# mixed signs -- weighting those at 0.7 made the closed smile *wider*
# (+29%), measured 2026-09-06, so they are left alone.
_MOUTH_LATERAL = ((14, 0), (17, 0), (20, 0), (7, 1))
_MOUTH_SMILE = ((14, 1),)
# The *opening*: the y of 19 and 20 (+-0.05: mouth ratio 0.24 / 0.27) and of
# 17 (0.10). Day 9, night: these three components are gated on their own,
# by the mouth *ratio* alone (_gates) -- his "that looked wrong" capture at
# 23:21:30 had him in a closed-lip grin (smile blendshapes 0.6-0.8, mouth
# ratio 0.004) and the avatar in an open, toothy one: the width gate had
# opened the whole mouth, and the extractor's reading of a strong smile
# carries parted lips with it. The lips now part only when his do.
# Day 15 (his 2026-09-11 look at the day-14 build: "on a cheerful toothy
# smile the teeth grow a little in height"): ``mouth_opening_gain`` (default
# 1.0, bit-exact) weights these three components the way mouth_smile_gain
# weights the smile's -- the lips' parting on its own, the smile's width
# and the gate untouched. Measured: the avatar's opening over his on the
# clips' smiles 1.02-1.05 at the median but 1.08-1.26 at the 90th percentile,
# and 1.76 on the frame he pressed on (his lips just parted) -- small
# openings over-shoot most, where expression_gain 1.8 multiplies a small
# reading and the network has not saturated.
_MOUTH_OPENING = ((17, 1), (19, 1), (20, 1))

# Day 15, his second capture (09:50, the toothy smile itself, with the
# session's own anchor in the file): the avatar's lips parted 1.60x his,
# and the SAME frame rendered with an anchor learned on driving.mkv parted
# them 1.06x. The two anchors differ in almost nothing but these three
# components: his live rest read (0.003, -0.007, 0.017), the clip's
# (-0.001, 0.002, 0.008), the portrait's own closed mouth (-0.000, -0.004,
# 0.003). His lips were PRESSED at that session's start -- what a person
# does looking at a screen -- and the extractor reads a pressed mouth as
# closed beyond closed, so every opening measured from there is larger by
# a constant (+0.1 of mouth ratio on both of his captures, the 08:10 talk
# frame included, which a multiplicative gain fitted by accident). With
# the anchor's opening rows taken from the portrait's own closed mouth --
# relaxed, read in the same units by the same network -- his frame came
# out 0.98-1.02x his. ``mouth_rest`` in engine_options: "portrait" does
# that; "anchor" keeps his own rest reading. **"anchor" is the default,
# and "portrait" was REVERTED on his live look the same morning:** with
# the portrait's relaxed lips as the reference, his own pressed rest reads
# as "closing" -- a slight opening did not open the avatar's mouth but
# drooped it, a closing pursed the lips, and talking stopped following. The
# reference has to be HIS posture; what goes wrong is that his posture at
# the anchor (pressed, at the session's start) is not his posture minutes
# later (relaxed), and the difference rides on every opening as a constant.
# The fix for that is a reference that follows his current neutral mouth,
# not a fixed one from either face -- see lip_rest_tracking, if built.
# Day 16: "driver" -- the rows a calibration measured on THIS person's
# relaxed mouth (``lip_rest_rows``, in the driver profile), written into
# the anchor the way "portrait" writes the portrait's: his own units, his
# own face, so a session that starts with pressed lips does not spend its
# first minute over-opening while lip_rest_tracking catches up. Without
# the rows it is "anchor".
_MOUTH_RESTS = ("anchor", "portrait", "driver")


def _mouth_rest(options: dict[str, Any]) -> str:
    value = str(options.get("mouth_rest", "anchor"))
    if value not in _MOUTH_RESTS:
        msg = f"mouth_rest must be one of {_MOUTH_RESTS}, got {value!r}"
        raise ValueError(msg)
    return value


def _rest_mouth_from_portrait(loaded: _Portrait) -> None:
    """Overwrite the anchor's lip-opening components (:data:`_MOUTH_OPENING`)
    with the portrait's own, in place -- ``exp_d0`` is a static tensor the
    graph reads, so the storage must stay the same."""
    for k, axis in _MOUTH_OPENING:
        loaded.exp_d0[0, k, axis] = loaded.exp_s[0, k, axis]


def _lip_rest_rows(options: dict[str, Any]) -> tuple[float, ...] | None:
    """``lip_rest_rows``: the driver's own resting reading on the
    :data:`_MOUTH_OPENING` rows, in that order, as a calibration measured
    it (day 16) -- what ``mouth_rest = "driver"`` writes into the anchor.
    ``None`` when absent."""
    raw = options.get("lip_rest_rows")
    if raw is None:
        return None
    try:
        rows = tuple(float(v) for v in raw)
    except (TypeError, ValueError) as exc:
        msg = f"lip_rest_rows must be {len(_MOUTH_OPENING)} numbers, got {raw!r}"
        raise ValueError(msg) from exc
    if len(rows) != len(_MOUTH_OPENING) or any(not math.isfinite(v) or abs(v) > 0.5 for v in rows):
        msg = (
            f"lip_rest_rows must be {len(_MOUTH_OPENING)} finite numbers in [-0.5, 0.5], "
            f"got {raw!r}"
        )
        raise ValueError(msg)
    return rows


def _apply_mouth_rest(loaded: _Portrait) -> bool:
    """After the anchor is frozen or seeded: overwrite its lip-opening rows
    as ``mouth_rest`` says -- the portrait's own, or the driver's measured
    rest -- and say whether anything was written. "anchor" (and "driver"
    without rows) leaves the anchor's own reading."""
    mode = _mouth_rest(loaded.options)
    if mode == "portrait":
        _rest_mouth_from_portrait(loaded)
        return True
    if mode == "driver" and loaded.lip_rest_rows is not None:
        for (k, axis), value in zip(_MOUTH_OPENING, loaded.lip_rest_rows, strict=True):
            loaded.exp_d0[0, k, axis] = value
        return True
    return False


# Day 15, the fix that held (his live look reverted "portrait" the same
# morning: a slight opening drooped, a closing pursed, talking stopped
# following -- the reference has to be HIS mouth). What goes wrong is that
# his mouth at the anchor is not his mouth minutes later: lips pressed at a
# session's start (looking at a screen), relaxed once he talks. Measured by
# seeding his 09:50 anchor (pressed) into a driving.mkv run: the toothy
# smile 1.58x his, the wide mouth 1.59x, TALK 2.20x (p90 3.6x) -- against
# 1.03 / 1.10 / 0.97 with the clip's own anchor. The reference's precision
# matters at the thousandth (0.001 on these rows is ~10% of a talking
# mouth), so it cannot be a one-time reading. ``lip_rest_tracking`` (frames,
# default 100; 0 off): once the anchor is frozen, on every frame the day-9
# gates call a resting mouth (opening and width gates both shut, the face
# not held) for more than _LIP_REST_SETTLE frames in a row, the anchor's
# lip-opening rows (_MOUTH_OPENING) move toward the extractor's reading of
# that frame by 1/frames -- a time constant of ~4-5 s live -- and nothing
# else of the anchor moves. With the same pressed anchor and tracking: 1.08
# / 1.12 / 1.12; with the clip's own anchor and tracking: 1.09 / 1.11 /
# 1.10 (a session's resting mouth reads a hair more closed than its first
# fifteen frames; +0.1 of ratio, the thousandth again). The tracked rows
# ride along in a capture's anchor, so a replay is still the session.
_LIP_REST_SETTLE = 15
_LIP_REST_GATE_SHUT = 0.02
# Day 16 (C3): the first resting frames after the anchor freezes move the
# rows this many times faster than the option's rate, for this many of
# them -- a session that starts with pressed lips (his 09:50) was over-
# opening for the 100-frame time constant's first seconds; at 4x over
# 60 frames the anchor is within 10% of the resting reading in two
# seconds instead of eight, and the slow rate takes over from there.
_LIP_REST_FAST_FACTOR = 4.0
_LIP_REST_FAST_FRAMES = 60

# Day 15, his "a strong smile with the brows raised -- look what happens to
# the teeth then". The extractor's reading of a brow raise is not confined
# to the brows: over driving.mkv's raise (1642-1760, mouth shut) the mouth
# rows move with his brow ratio at r = 0.83-0.98 -- the lower lip's y (20)
# by +0.10..+0.19 per unit of brow ratio, 17's y by +0.04..+0.06, 7's y and
# the z of 17/20 less (the two clips agree in sign there; 19's y and the
# smile row 14 did not, and are left out). Gated away while the mouth is
# shut; during a smile the mouth gate is open and the leak rides in with
# the smile's own reading -- and it points the OTHER way from an opening:
# the toothy-smile frame re-rendered with a +0.06 raise's leak added closed
# by a third (mouth ratio 0.190 -> 0.135; natural.mkv's 0.392 -> 0.266),
# the brow drive itself narrowing the smile 7% on top. So the leak his
# measured brow deviation predicts is subtracted from the relative
# expression in _drive (``mouth_leak``, written each frame in _render);
# ``brow_mouth_leak`` in engine_options scales it, 1.0 the clips' average,
# 0 off. The slopes are his face's, from two clips that differ by up to
# 1.8x -- a live capture pressed DURING the moment is what tunes them.
_BROW_MOUTH_LEAK: dict[tuple[int, int], float] = {
    (17, 1): 0.052,
    (20, 1): 0.144,
    (7, 1): 0.020,
    (17, 2): -0.014,
    (20, 2): -0.025,
}


def _brow_mouth_leak(options: dict[str, Any]) -> float:
    """``brow_mouth_leak`` in ``[0, 2]``, default 1.0: how much of the
    measured brow-to-mouth leak (:data:`_BROW_MOUTH_LEAK`) is subtracted."""
    gain = float(options.get("brow_mouth_leak", 1.0))
    if not (0.0 <= gain <= 2.0):
        msg = f"brow_mouth_leak must be in [0, 2], got {gain}"
        raise ValueError(msg)
    return gain


def _brow_mouth_leak_slopes(options: dict[str, Any]) -> dict[tuple[int, int], float]:
    """``brow_mouth_leak_slopes``: the leak table itself, as
    ``[[keypoint, axis, slope], ...]`` -- what "Calibrate me" measures on
    the driver's own brow raise (day 16). Default :data:`_BROW_MOUTH_LEAK`,
    his two clips' average. A row given here replaces the whole table, so
    a profile says every slope it wants, not only the changed ones."""
    raw = options.get("brow_mouth_leak_slopes")
    if raw is None:
        return dict(_BROW_MOUTH_LEAK)
    slopes: dict[tuple[int, int], float] = {}
    try:
        for entry in raw:
            k_raw, axis_raw, slope_raw = entry
            k, axis, slope = int(k_raw), int(axis_raw), float(slope_raw)
            in_range = 0 <= k < NUM_KP and 0 <= axis < 3
            if not (in_range and math.isfinite(slope) and abs(slope) <= 1.0):
                raise ValueError(entry)
            slopes[(k, axis)] = slope
    except (TypeError, ValueError) as exc:
        msg = (
            "brow_mouth_leak_slopes must be [[keypoint 0..20, axis 0..2, slope in [-1, 1]], "
            f"...], got {raw!r}"
        )
        raise ValueError(msg) from exc
    return slopes


def _lip_rest_tracking(options: dict[str, Any]) -> float:
    """``lip_rest_tracking`` in frames, default 100 -- the time constant the
    anchor's lip rest follows his resting mouth with; 0 is off, the anchor
    as learned (day 14's behaviour)."""
    frames = float(options.get("lip_rest_tracking", 100.0))
    if not (0.0 <= frames <= 10000.0):
        msg = f"lip_rest_tracking must be in [0, 10000] frames, got {frames}"
        raise ValueError(msg)
    return frames


# With the lateral components cut, what tilt is left is the portrait's own
# (the photograph's mouth sits 0.020 of inter-ocular higher on one side;
# his own 0.010-0.014) plus what this face's generator makes of a symmetric
# smile (+0.014 more at the median of a closed smile). ``mouth_tilt_trim``
# is a constant x offset on keypoint 17 -- +0.05 there tilted the mouth
# -0.119 of inter-ocular on the portrait (the table in the ROADMAP), so
# +0.005 is about -0.012 -- added to the driving keypoints every frame,
# rest included. In keypoint units; 0.0 off.
_TILT_TRIM_KEYPOINT = 17


def _mouth_tilt_trim(options: dict[str, Any]) -> float:
    t = float(options.get("mouth_tilt_trim", 0.0))
    if not (-0.05 <= t <= 0.05):
        raise ValueError(f"mouth_tilt_trim must be in [-0.05, 0.05], got {t}")
    return t


def _expression_gain(options: dict[str, Any]) -> float:
    """``expression_gain`` in ``[0, 3]``, default 1.0: one multiplier on the
    whole relative expression (``exp_now - exp_d0``) except the eyes' rows.

    Day 12 (2026-09-07). Until today the motion extractor ran in fp16 by a
    plain ``.half()``, and that was not a precision loss but a different
    function: on the portrait it read scale 1.86 (fp32: 1.01), t -0.073
    (-0.010) and keypoints of the wrong sign, and its relative expression
    came out about 1.8x too large and flickering (see _load_nets). Twelve
    days of gains were fitted to that. With the extractor in fp32 the
    reading is right and steady, and everything the mouth does reads
    smaller than he is used to: on driving.mkv the talk mouth gain 0.64 ->
    0.85 but the closed smile's width gain 0.73 -> 0.18, on natural.mkv
    the talk gain 0.86 -> 0.50. This is the day-6 lever, built now, so the
    amplitude can be put back by measurement instead of by the old noise.
    """
    g = float(options.get("expression_gain", 1.0))
    if not (0.0 <= g <= 3.0):
        raise ValueError(f"expression_gain must be in [0, 3], got {g}")
    return g


def _mouth_axis_gain(options: dict[str, Any], key: str, default: float) -> float:
    g = float(options.get(key, default))
    # Day 12: up to 3 -- with the extractor in fp32 the smile axis reads
    # smaller than the fp16 noise made it, and a gain past 1 is a lever now.
    if not (0.0 <= g <= 3.0):
        raise ValueError(f"{key} must be in [0, 3], got {g}")
    return g


def _region_masks(
    eye_expression_gain: float,
    eye_mask_output: bool,
    torch: Any,
    device: Any,
    *,
    mouth_lateral_gain: float = 0.0,
    mouth_smile_gain: float = 1.0,
    mouth_opening_gain: float = 1.0,
    expression_gain: float = 1.0,
) -> tuple[Any, Any]:
    """The two masks :meth:`LivePortraitEngine._drive` multiplies into the
    relative expression and the eye retargeting network's own output --
    both ``(1, NUM_KP, 3)``, built once at load and read every frame from
    static tensors on ``_Portrait`` (``rel_mask``/``eye_mask``): like every
    other option here, this is baked into the graph at capture time, and a
    live change needs the avatar reloaded.

    ``rel_mask``: ones everywhere except :data:`EYE_KEYPOINTS`' rows, which
    hold ``eye_expression_gain`` -- at ``1.0`` (today's, day-7-and-before
    behaviour) it changes nothing; at ``0.0`` the extractor's own reading of
    the eyes never reaches the portrait at all.

    ``eye_mask``: when ``eye_mask_output`` is true, zeros everywhere except
    EYE_KEYPOINTS' rows, which hold one -- confines the eye retargeting
    network's own delta to those same four keypoints, so a driven eye
    target cannot itself leak into the mouth or brows the way the raw
    extractor does. False makes it ones everywhere (the network's whole
    delta applies, unmasked) -- kept only to measure the two changes apart.

    ``eye_expression_gain`` is assumed already validated (see
    :func:`_eye_expression_gain`, the only caller that reads it from an
    option).
    """
    # Day 12: ``expression_gain`` scales the whole relative expression except
    # the eyes' rows (the retargeting network drives those). See
    # _expression_gain for why it exists at all.
    rel_mask = torch.full((1, NUM_KP, 3), expression_gain, device=device, dtype=torch.float32)
    rel_mask[0, list(EYE_KEYPOINTS), :] = eye_expression_gain
    # The mouth's own axes (day 9, later; see _MOUTH_LATERAL/_MOUTH_SMILE):
    # the tilt components and the smile component, each weighted.
    for k, axis in _MOUTH_LATERAL:
        rel_mask[0, k, axis] = mouth_lateral_gain * expression_gain
    for k, axis in _MOUTH_SMILE:
        rel_mask[0, k, axis] = mouth_smile_gain * expression_gain
    # Day 15: the lips' parting on its own (see _MOUTH_OPENING).
    for k, axis in _MOUTH_OPENING:
        rel_mask[0, k, axis] = mouth_opening_gain * expression_gain
    eye_mask = torch.zeros(1, NUM_KP, 3, device=device, dtype=torch.float32)
    if eye_mask_output:
        eye_mask[0, list(EYE_KEYPOINTS), :] = 1.0
    else:
        eye_mask.fill_(1.0)
    return rel_mask, eye_mask


# ---------------------------------------------------------------------------
# Expression gates: the mouth and the brows move only when *he* measurably
# does (day 9, 2026-09-06)
# ---------------------------------------------------------------------------
#
# The live complaint this answers, in the owner's words: "it parts its lips
# and half-shuts its eyes when I do nothing of the kind and simply lift my
# head -- it invents reactions". Measured on recordings/driving.mkv, the
# look-up segment (frames 421-538, day-8 build): his mouth ratio flat at
# 0.003-0.006, the avatar's 0.003-0.101 (0.023-0.062 over frames 469-525);
# his brow steady, the avatar's dropping 0.20 -> 0.135 (r -0.49 against
# pitch). The motion extractor reads a lifted head partly as an expression,
# and until now every keypoint but the eyes' four took that reading as it
# came.
#
# The fix is a gate, not a cut: a scalar in [0, 1] per region, multiplied
# into the relative expression on that region's keypoints every frame (see
# _drive, ``loaded.gate``). It opens from *his own tracked geometry* -- how
# far his mouth ratio / mouth width / brow height are from the anchor's rest
# -- so a real smile, a word or a raised brow pass through whole (gate 1.0),
# and a still mouth on a lifted head passes nothing (gate 0.0). Soft between
# the two thresholds so it does not click; released over a few frames so a
# reading that hovers at a threshold does not flicker. The eyes are not gated
# here: since day 8 they never take the extractor's reading at all (rel_mask)
# and are driven only by the eye retargeting network from his measured ratio.
#
# Which keypoints are which was measured by nudging each of the 21 in turn on
# both portraits (day 8): mouth 19/20 (lips, y and z), 17, 14, 7; brows 2, 3,
# 1; eyes EYE_KEYPOINTS. Everything else (nose, contour) is left to the pose.
MOUTH_KEYPOINTS = (7, 14, 17, 19, 20)
BROW_KEYPOINTS = (1, 2, 3)

# Thresholds, as (start, full): the gate is 0 up to ``start`` of deviation
# from his rest, 1 from ``full`` on, linear between. Set from his own still
# frames on both clips (2026-09-06; the day-9 ROADMAP entry has the tables):
#
# * mouth ratio (inner-lip gap over mouth width, absolute): sitting still
#   0.002-0.005, looking up 0.004-0.005, talking up to 0.24. Start 0.010 is
#   twice the still reading's own top; full 0.030 so a word already
#   half-formed passes whole. Day 12: the gap is now over the *jaw* width
#   (_mouth_open_ratio, why there), 0.4x the old reading at rest on every
#   clip and capture, so both mouth-ratio thresholds and the activity unit
#   below are the day-9 values times 0.4; replayed on both clips' every
#   segment and on all nine captures (scripts/probes/gate_replay.py), the
#   gate means moved at most 0.023 (a head turn -- the jaw foreshortens
#   differently), 0.000 on talk, smile and still. "MAR" in the gates'
#   names, rest_mar and mar_prev means this measure from here on.
# * mouth width (over jaw width, see _mouth_width_ratio; as a fraction of
#   rest): still spread +-1%; a closed smile +8% at the median, +15% at its
#   widest. Widening only -- a narrower reading is not a reaction he makes
#   on purpose often enough to pay for, and narrower is what a tilt does to
#   the measure (-5% at pitch -30).
# * brow (see _brow_ratio; as a fraction of rest): still +-1..2%; a raise
#   +34..38% at its height, a frown about -13%; 2% on a blink, 5-6% on a
#   closed smile. Either direction (a frown is real too), faded by pose
#   below.
_MOUTH_GATE_MAR = (0.004, 0.012)
_MOUTH_GATE_WIDTH = (0.03, 0.08)
_BROW_GATE = (0.08, 0.16)
# Past these the region is held at the portrait's own rather than read: 1.0
# up to the first, 0.0 from the second, linear between, on |pitch| and |yaw|
# separately, multiplied. The width measure inflates 8-10% at the widest of
# a turn (all of a closed smile's threshold), so its gate fades with yaw;
# the mouth *ratio* is one distance over another on the mouth itself and is
# not faded -- a word said turned away still moves the lips. The brow
# measure keeps +0.3% of rest per degree of pitch and +4% at a turn's
# widest, so it is faded on both, generously.
_MOUTH_GATE_YAW = (12.0, 25.0)
_BROW_GATE_PITCH = (10.0, 20.0)
_BROW_GATE_YAW = (15.0, 30.0)
# A closed gate decays by this each frame rather than dropping to 0 at once:
# 0.7 -> a tenth of the way in six frames (0.2 s at 30 fps). Opening is
# instant, so nothing he does arrives late.
_GATE_RELEASE = 0.7

# Speech is *motion*, not distance from rest (day 9, later, 2026-09-06). His
# live verdict on the distance-only gate above: "it no longer follows my
# mouth well when I talk". Measured on natural.mkv's 594 talking frames:
# the gate was shut on 20% of them and half-open on 11%, and flipped
# open/shut 49 times in 20 seconds -- every closed-lip consonant ("m", "b",
# "p") and every pursed vowel ("o", "u") shut it, and the release dragged
# the lips back to the portrait's between syllables. So a second way to
# open: the mouth is *moving*. Per frame, |change in mouth ratio| and
# |change in width| (either direction -- a pursing narrows) are summed in
# units of the constants below, gated softly, and a frame that reads as
# moving holds the gate fully open for _MOUTH_HOLD_FRAMES afterwards, so a
# pause between words does not close it. A still mouth on a lifted head
# moves nothing and stays shut, exactly as before. The units come from the
# filtered tracker's own frame-to-frame reading on his clips (the day-9
# ROADMAP entry has the table): sitting still or looking up, |dMAR| p99
# 0.0004-0.0015 and |dW| p99 0.4-0.6% of width; talking, |dMAR| p50 0.004
# (call posture) to 0.010 (scripted), p90 0.02-0.06. At these units a still
# frame's worst reading is 0.4 units on either measure -- 0.8 with both at
# their worst on one frame (gate 0.05, no hold) -- and an ordinary talking
# frame 1.0-2.4 (the hold sets from 1.25; at the call-posture median of
# 0.95 the gate alone reads 0.4, and the frames above 1.25 around it keep
# the hold going). Lip thickness was measured too and left out: it moves
# 2% a frame on a tilt with the mouth shut. Width is in units of 1.5%, not
# 1%, because a head turn changes the width-over-jaw reading 0.6% a frame
# at the median (yaw foreshortens the two unevenly).
#
# Day 9, night, from his two live "that looked wrong" captures (22 fps): with
# his mouth shut, |dMAR| read 0.0027 at p90 and 0.011 at p99, |dW| 0.7% /
# 2.0% -- three to seven times the clips' own rest jitter, and at the units
# above (0.004 / 1.5%) 17% of his shut-mouth frames read as "moving" and
# 9% set the hold. The units are now 0.010 / 2.5%: his live p99 is 1.1 / 0.8
# units (a one-frame partial open at worst, no hold), and talking still
# reads 1.9 units at p90 of the call-posture clip's |dMAR| (0.019) -- the
# hold, once set, carries the median frames between. Day 12: 0.004, the
# same unit on the over-jaw-width opening (0.4x -- see _mouth_open_ratio);
# his live shut-mouth |dMAR| p99 on that scale is 0.0043 (capture 232151).
_MOUTH_ACTIVITY_MAR = 0.004
_MOUTH_ACTIVITY_WIDTH = 0.025
_MOUTH_ACTIVITY_GATE = (0.75, 1.25)
_MOUTH_HOLD_FRAMES = 12
# A turn changes the width-over-jaw reading 0.4% of rest per degree of yaw
# (measured day 9) -- on a slow turn 0.6% a frame, on a quick one enough to
# read as "moving" and open the gate (69% of turning frames, 6 flips). The
# yaw's own change since the last frame, times this, is taken off the width
# change before it counts as motion. Also the gate now opens over two
# frames rather than one (_GATE_ATTACK): his verdict "the face bounces on
# the slightest sideways shift of my head" -- a gate snapping from 0 to 1
# in one frame moves the whole mouth region in one frame.
_WIDTH_PER_DEGREE_YAW = 0.004
_GATE_ATTACK = 0.5


def _expression_gates(options: dict[str, Any]) -> bool:
    """Whether the day-9 gates are on (``expression_gates``, default True).
    Off, both gates are 1.0 every frame -- the day-8-and-before behaviour,
    kept only to measure the change against."""
    return bool(options.get("expression_gates", True))


# How much of the extractor's brow reading passes when the brow gate is open
# at all: the ``brow_gain`` option, default 0.0 -- **the brows are held at
# the portrait's own by default.** Measured 2026-09-06 with the gate open on
# a real brow raise (his own brow measure +34..38% on both clips, mouth
# shut, head frontal): on driving.mkv the avatar's brows rose 4-8% (the
# right way, a fifth of his); on natural.mkv, the same gesture, they
# *dropped* 12-16% -- a frown for a raise. The extractor's reading of the
# brows is not even reliably the right way up from one clip to the next,
# and a brow that frowns when he raises it is exactly the invented reaction
# this day is about. Nothing here can turn a wrong-way reading right; so
# until brows are driven from his own measured reading the way the eyes
# are, they do not move. 1.0 gives the gated brows back, for measuring.
def _brow_gain(options: dict[str, Any]) -> float:
    g = float(options.get("brow_gain", 0.0))
    if not (0.0 <= g <= 1.0):
        raise ValueError(f"brow_gain must be in [0, 1], got {g}")
    return g


# Brows driven from his own measured brow height instead (day 9, later,
# 2026-09-06) -- what the eyes have had since day 8, built by hand because
# LivePortrait ships no brow retargeting network: the brow keypoints are
# moved directly, in keypoint space, by an amount proportional to how far
# his brow measure (_brow_ratio) is from the anchor's rest. Direction and
# scale were calibrated on the portrait itself, 2026-09-06, nudging each of
# the three brow keypoints in y on the portrait's own crop and reading the
# avatar's brow measure back with MediaPipe (relative to its rest, left
# and right brow separately; eye ratio and mouth width alongside):
#
# | keypoint, y      | brow  | left / right  | eye ratio | mouth width |
# |------------------|-------|---------------|-----------|-------------|
# | 2, -0.025        | +29%  | +27% / +32%   | 0.332 (=) | -0.7%       |
# | 2, -0.050        | +69%  | +73% / +64%   | 0.354     | -1.4%       |
# | 2, -0.100        | +90%  | +89% / +92%   | 0.373     | -1.7%       |
# | 2, +0.025        | -17%  | -15% / -18%   | 0.340     | +0.4%       |
# | 2, +0.050/+0.100 | -22%  | (saturates)   | 0.330     | +3% / +8%   |
# | 3, -0.050        | +31%  | +34% / +28%   | 0.337     | **+6.6%**   |
# | 1, -0.050        | -5%   | -3% / -7%     | **0.351** | +2.0%       |
# | 1, +0.050        | +16%  | +20% / +12%   | **0.299** | -0.7%       |
#
# Keypoint 2 is the brow: even on both sides, the eyes and mouth left
# alone, 11.7 units of relative brow change per unit of -y on the raise
# side (6.6 on the lowering side, saturating at a 22% drop). Keypoint 3
# widens the mouth 6.6% for the same raise; keypoint 1 is not even
# monotone and moves the eyes instead. So only keypoint 2 is used. Moves
# only when his brows do, only the way his do, faded with pose like the
# gate (_BROW_GATE_PITCH/_YAW), with a dead band for the measure's own
# +-1..2% rest jitter. ``brow_follow`` is the gain: 1.0 asks the avatar's
# brow measure to move by the same fraction of its rest as his did (his
# raise of +35% -> y -0.027 -> about +31%); 0.0 turns it off.
#
# **Raises only.** A frown cannot be told from a gaze: with the head still
# and his eyes turned up (driving.mkv 469-525) his brow measure reads -12%
# of rest, the same as his real frown (-12%), and that held for every
# measure tried (2026-09-06; brows to the eye corners, to the nose bridge
# 168, to the glabella 6, over the chin, the nose length, inter-ocular --
# gaze up read -9..-17% on all nine, frown -8..-12%). Whether MediaPipe's
# brow points ride down with an upward gaze or his brows really dip, the
# avatar cannot frown on that number without frowning every time he looks
# up -- his day-9 complaint exactly. So a lowering moves nothing, and a
# frown is not carried (stated in the ROADMAP as a known cost).
#
# The dead band on the raise side is 10% of rest, not the measure's own
# +-1% still jitter: his eyes shutting reads +5..+10% (the corner points
# move with the lids), a blink +4..+9%, a closed smile up to +5%, a turn
# +8%, a tilt +6%, ordinary talking +6% (scripted) to +13% (call posture --
# where he does seem to raise them). A real raise reads +33..+38%.
_BROW_DEADBAND = 0.10
# Keypoint-space y per unit of relative brow change (his measure / rest - 1),
# per keypoint -- negative y is up. The table above says 1 / 11.7 = 0.085
# on the portrait's own crop; driven live on his clips at 0.085 the avatar
# reached +28% (scripted) and +15% (call posture) of its rest for his +38%
# / +34% -- the response inside a real frame is weaker than on the still
# portrait -- so 0.12 (1.4x) is used.
_BROW_LIFT_Y = {2: -0.12}
# Day 17, his "when I shut my eyes the brows come down a little and the
# mouth moves as if smiling" -- measured on his calibration clip's blink
# and eyes-shut steps (scripts/probes/blink_probe.py, 18 shut frames
# against 94 open): the avatar's upper brow points sit 4.2 px lower and
# its mouth corners 1.4 px higher on the shut frames, his own not at all;
# with the eye retargeting network off (eye_gain 0) both go away (-0.4 /
# +0.2 px), with the extractor's expression off (expression_gain 0)
# neither changes, and every gate reads 0.00 on the shut frames. So it is
# the eye network's own delta on EYE_KEYPOINTS -- the lids pulled shut
# drag the warp's neighbourhood: the brows down, the mouth corners up.
# The counter is in the same tensor the brow drive writes: keypoint 2's y
# lifted by ``blink_brow_hold`` units per unit of blink depth (his eye
# ratio's shortfall from rest, 0 open, ~0.6 shut), and the smile axis
# (_MOUTH_SMILE) moved by ``blink_mouth_hold`` the same way. Both are
# portrait options. Measured at the live configuration (the profile over
# the metadata, his eye_gain 1.4): his avatar's brows -4.0 px at 0,
# -1.7 at 0.012, -1.2 at 0.015 (the corners +1.4 -> +0.3, his own +0.4),
# the zero near 0.02; the woman's -3.1 at 0, -0.2 at 0.013. Keypoint 2
# moves the corners a little with the brows, so the one lever settles
# both at the medians -- but by depth the shut end still smiled (see the
# curves below), so the mouth lever takes the shut end: -0.002 (a
# negative value lowers the corners). With the curves, his avatar by
# depth: brows -0.6 / -0.1 / +0.4 / +0.1 px, corners -0.8 / -0.5 / -0.5 /
# +0.2 (were -1.7 / -3.1 / -4.1 / -4.8 and +0.4 / +1.0 / +1.4 / +2.0).
# Each avatar's metadata says its own; these are his.
_BLINK_BROW_HOLD = 0.015
_BLINK_MOUTH_HOLD = -0.002
_BLINK_MOUTH_LIFT = 0.01
_BLINK_HOLD_MAX = 0.1


def _blink_brow_hold(options: dict[str, Any]) -> float:
    """``blink_brow_hold`` in ``[0, 0.1]``: keypoint 2's y lifted per unit
    of blink depth, against the eye network's pull on the brows."""
    value = float(options.get("blink_brow_hold", _BLINK_BROW_HOLD))
    if not 0.0 <= value <= _BLINK_HOLD_MAX:
        raise ValueError(f"blink_brow_hold must be in [0, {_BLINK_HOLD_MAX}], got {value}")
    return value


def _blink_mouth_hold(options: dict[str, Any]) -> float:
    """``blink_mouth_hold`` in ``[-0.1, 0.1]``: the smile axis moved per
    unit of blink depth, against the eye network's pull on the corners."""
    value = float(options.get("blink_mouth_hold", _BLINK_MOUTH_HOLD))
    if not -_BLINK_HOLD_MAX <= value <= _BLINK_HOLD_MAX:
        msg = f"blink_mouth_hold must be in [-{_BLINK_HOLD_MAX}, {_BLINK_HOLD_MAX}], got {value}"
        raise ValueError(msg)
    return value


def _blink_depth(ear: float, rest_ear: float | None) -> float:
    """How far his lids are shut: the eye ratio's shortfall from rest as a
    fraction of it, 0 open (or before the anchor), about 0.6 shut."""
    if rest_ear is None or rest_ear <= 1e-6:
        return 0.0
    return float(min(max(1.0 - ear / rest_ear, 0.0), 1.0))


# The pull is not linear in the depth (measured by depth bins on the same
# frames, brow hold 0): brows -1.7 px at a fifth shut, -3.1 at two fifths,
# -4.1 at three, -4.8 shut -- steep at first, then flattening -- and a
# linear counter set for the shut frames left the fast phases of a blink
# under-countered and the shut frames over (his second look: "now the
# brows go UP when I blink"). So the brow counter follows d(2 - d), the
# same saturating shape (0.36 / 0.64 / 0.84 / 0.98 at those depths).
# The corners' pull grows to the shut frames (+0.4 / +1.0 / +1.4 / +2.0)
# and keypoint 2's own pull on them saturates early, so what the brow
# lever leaves is at the shut end: the mouth counter comes in from half
# shut and is whole at 0.85.
def _blink_brow_curve(depth: float) -> float:
    return float(depth * (2.0 - depth))


def _blink_mouth_curve(depth: float) -> float:
    return float(min(max((depth - 0.5) / 0.35, 0.0), 1.0))


# His third look (the same session, by depth with his own beside): with
# the two counters the brows sit within 0.7 px at every depth, the
# corners match his at the shut end -- and sit 0.7-1.0 px UNDER his own
# (+0.3, his mouth eases as the lids shut) through the fast phases: the
# brow lever's own pull on the corners, largest where the lids are only
# starting to shut. ``blink_mouth_lift`` lifts the corners there, by a
# bump that is 0 open, largest around a third shut and gone at 0.85.
_BLINK_LIFT_END = 0.85


def _blink_lift_curve(depth: float) -> float:
    return float(max(depth * (1.0 - depth / _BLINK_LIFT_END), 0.0))


def _blink_mouth_lift(options: dict[str, Any]) -> float:
    """``blink_mouth_lift`` in ``[0, 0.1]``: the smile axis lifted through
    the blink's fast phases (see :func:`_blink_lift_curve`)."""
    value = float(options.get("blink_mouth_lift", _BLINK_MOUTH_LIFT))
    if not 0.0 <= value <= _BLINK_HOLD_MAX:
        raise ValueError(f"blink_mouth_lift must be in [0, {_BLINK_HOLD_MAX}], got {value}")
    return value


# Day 11 (2026-09-06), his "when I raise my brows the avatar lowers its
# eyes and looks down": measured on the portrait alone (the nudge probe
# with gaze read back per eye), keypoint 2 lifting the brow also drops the
# irises, +0.027 of eye width per -0.02 on its y, and closes the lids a
# little (-0.011). No keypoint undoes it: every eye keypoint's y moves the
# irises DOWN whichever way it is pushed (a V, not a line -- +0.02 and
# -0.02 on 11/13/15/16 y both read +0.01..+0.06), the other brow keypoints
# (1, 3) drop them too per unit of brow, and nothing at all moves them up.
# So the brow-up / gaze-down coupling is the network's own, and the lift
# keeps it: at his +25% raise it is ~2 px of iris in the frame. The larger
# part of what he saw -- the eyes' *position* dropping 13 px -- was the
# extractor's translation and the crop centre, fixed the same day (see
# _translation_gain and _crop_geometry's centre_offset).


# ---------------------------------------------------------------------------
# Gaze (day 11, 2026-09-06): "when I look to the side the eyes stay centred"
# ---------------------------------------------------------------------------
#
# Since day 8 the eye keypoints take nothing from the extractor, and the eye
# network only opens and shuts the lids, so the avatar's irises have always
# sat where the portrait's do. His own irises are in the tracker's 478-point
# output (468 left, 473 right): each against its eye's corner midpoint over
# the corner width, both eyes averaged -- +x right on screen, +y down, 0 is
# straight at the camera; his rest (screen posture, +0.08 on natural.mkv)
# is learned with the anchor (``rest_gaze``) and only the change from it is
# driven. Which keypoint moves the avatar's irises was measured on the
# portrait alone (the nudge probe, gaze read back per eye): keypoint 15's
# x moves BOTH irises the same way (+0.099 / +0.127 of eye width per
# +0.02, -0.112 / -0.220 per -0.02 -- the right eye further) with the lids
# within 0.03; keypoint 11's x moves mostly the left one (+0.075 / +0.015),
# and a little of it evens the two out. Vertical gaze has no lever: every
# eye keypoint's y reads the irises DOWN in both directions (see the brow
# note above), so only the horizontal offset is driven. His still-segment
# iris jitter is 0.014 (x) p90, hence the dead band; the EMA is about a
# frame of lag.
_IRIS_L, _IRIS_R = 468, 473
_GAZE_POINTS = 478
_GAZE_DEADBAND = 0.02
_GAZE_SMOOTHING = 0.6
_GAZE_BLINK_HOLD = 0.7  # of his rest eye ratio: below it, the gaze is held
_GAZE_KP_X = {15: 0.14, 11: 0.04}  # keypoint x per unit of his gaze_x


def _gaze_offset(points: np.ndarray) -> tuple[float, float] | None:
    """His iris offset ``(x, y)`` in units of eye width, both eyes averaged,
    from the tracker's full 478 points in frame pixels -- ``None`` without
    the iris points."""
    if points.shape[0] < _GAZE_POINTS:
        return None
    gx = gy = 0.0
    for iris, corners in ((_IRIS_L, _LEFT_EYE_CORNERS), (_IRIS_R, _RIGHT_EYE_CORNERS)):
        a, b = corners
        centre = (points[a, :2] + points[b, :2]) / 2.0
        width = abs(float(points[a, 0] - points[b, 0])) or 1e-6
        gx += float(points[iris, 0] - centre[0]) / width / 2.0
        gy += float(points[iris, 1] - centre[1]) / width / 2.0
    return gx, gy


def _gaze_gain(options: dict[str, Any]) -> float:
    """``gaze_gain`` in ``[0, 3]``, default 0.0 (off): 1.0 asks the avatar's
    irises to move sideways by his own offset from his rest."""
    g = float(options.get("gaze_gain", 0.0))
    if not (0.0 <= g <= 3.0):
        raise ValueError(f"gaze_gain must be in [0, 3], got {g}")
    return g


# Day 16 evening (2026-09-12, the first other face's capture of 20:07):
# the anchor learned her rest gaze while she looked at the CAMERA, as the
# status line asks; on the call she looked at the SCREEN -- the tracker read
# her irises +0.14 of an eye width to the side (the far end of her own
# left-right glance in the calibration, 0.27 wide) -- and the avatar looked
# away from the camera by the same amount, faithfully, for as long as she
# did. On the owner it never showed: his anchor happened to catch him
# looking at the screen, so "rest" WAS the screen. A resting gaze is where
# a person looks most of the time, not where they looked for the anchor's
# half second, so the rest now follows the gaze by a slow EMA -- this many
# frames as the time constant, ten seconds at 30 fps: a glance of two
# seconds moves it by 6% of the glance (under the dead band), a look held
# at the screen becomes the rest in half a minute and the avatar looks at
# the viewer again. 0 is off: the rest as the anchor learned it.
_GAZE_REST_TRACKING = 300.0


def _gaze_rest_tracking(options: dict[str, Any]) -> float:
    """``gaze_rest_tracking`` in frames, default :data:`_GAZE_REST_TRACKING`
    -- the time constant the rest gaze follows his gaze with; 0 is off."""
    frames = float(options.get("gaze_rest_tracking", _GAZE_REST_TRACKING))
    if not (0.0 <= frames <= 10000.0):
        msg = f"gaze_rest_tracking must be in [0, 10000] frames, got {frames}"
        raise ValueError(msg)
    return frames


def _brow_follow(options: dict[str, Any]) -> float:
    g = float(options.get("brow_follow", 0.0))
    if not (0.0 <= g <= 3.0):
        raise ValueError(f"brow_follow must be in [0, 3], got {g}")
    return g


# ---------------------------------------------------------------------------
# The driver's own thresholds (day 16, 2026-09-12): "Calibrate me"
# ---------------------------------------------------------------------------
#
# Every gate number above was set from HIS still frames and HIS live jitter,
# by hand, over days 9-12. A second person's mouth does not jitter by his
# 0.004 a frame and does not raise the brows by his 38%: the thresholds are
# a property of the driver, not of the portrait or the network. So each is
# an option now, read once at load into :class:`_GateThresholds`, with the
# constants as the defaults -- and the driver profile a calibration writes
# (sha256.calibration, configs/local.driver.json) lays its measured values
# over the avatar's engine_options (see DRIVER_OPTION_KEYS). Without a
# profile nothing changes: every default is the constant it replaces.


@dataclass(frozen=True, slots=True)
class _GateThresholds:
    """One driver's gate thresholds -- see :func:`_gate_thresholds`."""

    mouth_gate_mar: tuple[float, float] = _MOUTH_GATE_MAR
    mouth_gate_width: tuple[float, float] = _MOUTH_GATE_WIDTH
    activity_mar: float = _MOUTH_ACTIVITY_MAR
    activity_width: float = _MOUTH_ACTIVITY_WIDTH
    brow_deadband: float = _BROW_DEADBAND


def _threshold_pair(
    options: dict[str, Any], key: str, default: tuple[float, float], limit: float
) -> tuple[float, float]:
    raw = options.get(key, default)
    try:
        start, full = (float(v) for v in raw)
    except (TypeError, ValueError) as exc:
        msg = f"{key} must be [start, full], got {raw!r}"
        raise ValueError(msg) from exc
    if not (0.0 < start < full <= limit):
        msg = f"{key} must satisfy 0 < start < full <= {limit}, got {raw!r}"
        raise ValueError(msg)
    return start, full


def _threshold_unit(options: dict[str, Any], key: str, default: float, limit: float) -> float:
    unit = float(options.get(key, default))
    if not (0.0 < unit <= limit):
        msg = f"{key} must be in (0, {limit}], got {unit}"
        raise ValueError(msg)
    return unit


def _gate_thresholds(options: dict[str, Any]) -> _GateThresholds:
    """The gates' thresholds for this driver: ``mouth_gate_mar`` and
    ``mouth_gate_width`` as ``[start, full]`` (the distance gates, see
    :data:`_MOUTH_GATE_MAR`/:data:`_MOUTH_GATE_WIDTH`), ``mouth_activity_mar``
    and ``mouth_activity_width`` (the motion units, :data:`_MOUTH_ACTIVITY_MAR`/
    :data:`_MOUTH_ACTIVITY_WIDTH`) and ``brow_deadband`` (:data:`_BROW_DEADBAND`).
    Each defaults to its constant; validated once here, at load."""
    deadband = float(options.get("brow_deadband", _BROW_DEADBAND))
    if not (0.0 <= deadband <= 1.0):
        msg = f"brow_deadband must be in [0, 1], got {deadband}"
        raise ValueError(msg)
    return _GateThresholds(
        mouth_gate_mar=_threshold_pair(options, "mouth_gate_mar", _MOUTH_GATE_MAR, 0.2),
        mouth_gate_width=_threshold_pair(options, "mouth_gate_width", _MOUTH_GATE_WIDTH, 1.0),
        activity_mar=_threshold_unit(options, "mouth_activity_mar", _MOUTH_ACTIVITY_MAR, 0.1),
        activity_width=_threshold_unit(options, "mouth_activity_width", _MOUTH_ACTIVITY_WIDTH, 0.5),
        brow_deadband=deadband,
    )


def _brow_lift(
    brow: float, rest_brow: float | None, pose: HeadPose | None, deadband: float | None = None
) -> float:
    """How far his brows are *raised* past rest this frame, beyond the dead
    band (:data:`_BROW_DEADBAND`, or this driver's own from
    :func:`_gate_thresholds`), faded with pose -- the scalar
    :meth:`LivePortraitEngine._render` turns into a keypoint offset via
    :data:`_BROW_LIFT_Y`. Never negative: a lowering is not carried (see
    the constants' own comment for why). 0.0 until the anchor has frozen."""
    if rest_brow is None:
        return 0.0
    band = _BROW_DEADBAND if deadband is None else deadband
    lift = max(brow / (rest_brow or 1e-6) - 1.0 - band, 0.0)
    if pose is not None:
        lift *= _fade(pose.pitch, _BROW_GATE_PITCH) * _fade(pose.yaw, _BROW_GATE_YAW)
    return lift


def _soft_gate(deviation: float, thresholds: tuple[float, float]) -> float:
    """0 at or below ``start``, 1 at or above ``full``, linear between."""
    start, full = thresholds
    if full <= start:
        raise ValueError(f"gate thresholds must rise, got {thresholds}")
    return min(max((deviation - start) / (full - start), 0.0), 1.0)


def _fade(value: float, thresholds: tuple[float, float]) -> float:
    """1 at or below ``start``, 0 at or above ``full``, linear between --
    ``_soft_gate`` the other way up, on a pose angle's magnitude."""
    return 1.0 - _soft_gate(abs(value), thresholds)


def _mouth_gate(
    mar: float,
    width: float,
    rest_mar: float | None,
    rest_width: float | None,
    pose: HeadPose | None = None,
    thresholds: _GateThresholds | None = None,
) -> float:
    """How much of the extractor's mouth reading to pass this frame: the
    larger of the ratio gate (he is opening it) and the width gate (he is
    smiling; faded with yaw, see _MOUTH_GATE_YAW). 1.0 -- everything, as
    before -- until the anchor has frozen and his rest is known. With no
    head pose this frame the fade is skipped, not assumed frontal: the
    anchor itself needed a pose, so this is the rare tracker frame without
    one, and the reading is taken as it is."""
    if rest_mar is None or rest_width is None:
        return 1.0
    gate_mar = _MOUTH_GATE_MAR if thresholds is None else thresholds.mouth_gate_mar
    widening = _mouth_widening(width, rest_width, pose, thresholds)
    return max(_soft_gate(mar - rest_mar, gate_mar), widening)


def _mouth_widening(
    width: float,
    rest_width: float,
    pose: HeadPose | None = None,
    thresholds: _GateThresholds | None = None,
) -> float:
    """The width gate alone: how far his mouth is wider than at rest, 0..1
    over ``mouth_gate_width``, faded with yaw -- "he is smiling"."""
    gate_width = _MOUTH_GATE_WIDTH if thresholds is None else thresholds.mouth_gate_width
    widening = _soft_gate(width / (rest_width or 1e-6) - 1.0, gate_width)
    if pose is not None:
        widening *= _fade(pose.yaw, _MOUTH_GATE_YAW)
    return widening


# Day 16 evening (his capture of 22:43 on the new avatar woman_person, a
# toothy smile): the avatar parted its lips 1.36x his on the frame he saw
# and 1.62x on its replay, at the smile's width 1.17x -- the teeth a tall
# band where his are a thin line. ``mouth_opening_gain`` 0.5 brought the
# parting to 1.00x on that frame, but it weights talking the same way and
# would leave speech half as open. So the reduction rides on the width
# gate: the opening gate is multiplied by ``1 - (1 - mouth_smile_opening)
# * widening`` -- nothing while he talks with the mouth at its resting
# width, the full reduction on a wide smile, the width itself untouched.
# A property of the portrait (how its mouth opens on a smile), so it lives
# in the avatar's metadata; 1.0 is bit-exact with before.
def _mouth_smile_opening(options: dict[str, Any]) -> float:
    g = float(options.get("mouth_smile_opening", 1.0))
    if not (0.0 <= g <= 1.0):
        raise ValueError(f"mouth_smile_opening must be in [0, 1], got {g}")
    return g


def _mouth_activity(
    mar: float,
    width: float,
    mar_prev: float | None,
    width_prev: float | None,
    yaw_change: float = 0.0,
    thresholds: _GateThresholds | None = None,
) -> tuple[float, float]:
    """How much his mouth moved since the last frame -- ``(ratio, width)``,
    each in units of :data:`_MOUTH_ACTIVITY_MAR` / :data:`_MOUTH_ACTIVITY_WIDTH`
    (1.0 is "clearly moving"); ``(0, 0)`` with no last frame to compare.
    ``yaw_change`` (degrees since the last frame) discounts the width
    change a turn causes by itself (:data:`_WIDTH_PER_DEGREE_YAW`). The two
    are kept apart because the lips' opening is gated by the ratio's own
    motion only (see :meth:`LivePortraitEngine._gates`)."""
    if mar_prev is None or width_prev is None:
        return 0.0, 0.0
    unit_mar = _MOUTH_ACTIVITY_MAR if thresholds is None else thresholds.activity_mar
    unit_width = _MOUTH_ACTIVITY_WIDTH if thresholds is None else thresholds.activity_width
    width_change = abs(width / (width_prev or 1e-6) - 1.0)
    width_change = max(width_change - _WIDTH_PER_DEGREE_YAW * abs(yaw_change), 0.0)
    return abs(mar - mar_prev) / unit_mar, width_change / unit_width


def _brow_gate(brow: float, rest_brow: float | None, pose: HeadPose | None) -> float:
    """How much of the extractor's brow reading to pass this frame: how far
    his brows are from rest, in either direction, faded to nothing as the
    head leaves frontal (see _BROW_GATE_PITCH/_BROW_GATE_YAW). 1.0 until the
    anchor has frozen; no head pose this frame skips the fade, as
    :func:`_mouth_gate` does."""
    if rest_brow is None:
        return 1.0
    gate = _soft_gate(abs(brow / (rest_brow or 1e-6) - 1.0), _BROW_GATE)
    if pose is not None:
        gate *= _fade(pose.pitch, _BROW_GATE_PITCH) * _fade(pose.yaw, _BROW_GATE_YAW)
    return gate


def _release(gate: float, previous: float) -> float:
    """The gate actually applied: instant to open, ``_GATE_RELEASE`` per
    frame to close -- see the constant's own comment."""
    return max(gate, previous * _GATE_RELEASE)


# ---------------------------------------------------------------------------
# Holding the last face at a pose a frontal portrait cannot redraw (day 8,
# 2026-09-06)
# ---------------------------------------------------------------------------

# Degrees past which the rendered face is frozen rather than redrawn -- see
# _should_hold. Measured on recordings/natural.mkv: a "tilt your head
# slightly down" instruction reached pitch -32 degrees, far past anything a
# frontal portrait can be redrawn convincingly at; frames 560/620 show the
# face smearing, the crop's own forehead band showing.
_DEFAULT_HOLD_PITCH = 30.0
_DEFAULT_HOLD_YAW = 45.0
# Pitch 30 and yaw 45 since day 17 (were 20 / 35): under the clamp the
# network drew his -30 and -33 tilts cleanly at a cap of 30
# (recordings/day17/tiltcap-*), the eyes following his look up from under
# the brows, and his -48 turn at caps of 45 and 55 alike
# (recordings/day17/drivecap-*) -- the head plausibly turned, what went
# wrong there was the expression, which _steep_fade now calms. The day-8
# smear was the full -32 with no cap at all.
# How far below the enter threshold the pose must fall before the hold
# releases -- lower than the threshold that started it, so a pose sitting
# right at the edge does not flicker between held and rendered every other
# frame.
_HOLD_RELEASE_RATIO = 0.75


def _hold_thresholds(options: dict[str, Any]) -> tuple[float, float]:
    """(pitch, yaw) degrees past which :func:`_should_hold` freezes the
    rendered face -- ``hold_pitch_degrees``/``hold_yaw_degrees``, both must
    be positive."""
    pitch = float(options.get("hold_pitch_degrees", _DEFAULT_HOLD_PITCH))
    yaw = float(options.get("hold_yaw_degrees", _DEFAULT_HOLD_YAW))
    if pitch <= 0.0 or yaw <= 0.0:
        msg = f"hold_pitch_degrees/hold_yaw_degrees must be positive, got {pitch}/{yaw}"
        raise ValueError(msg)
    return pitch, yaw


# Day 20 (2026-09-18), his "a quick turn left-right zooms the face out":
# the crop is sized by the pose's translation z (see _driver_face_height),
# and z is a rigid fit of the whole face model -- on half a face it wanders.
# Measured on his turn clip (recordings/driving-turn.mkv, the tracker as
# the app runs it, z over the still start's median): within 10 degrees of
# yaw 0.995 (0.97-1.08 as he leans), 20-35 degrees 0.96-0.98 (nearer than
# he is, the crop up to 4% too large), 36-47 to the left 1.01 (farther),
# the profile 0.92-0.95, and the first frame found after a profile 1.012
# at -63 -- 4% farther than the frames right after, and the crop's 1-Hz
# size filter, reset by the loss, started from that frame: the avatar 3-4%
# smaller than his face for the next half second, back at the camera,
# with the photograph's background as a line along the far side (day 19).
# So past this many degrees of yaw from the rest the crop's size stops
# following z and keeps the depth he sat at -- a turn in place does not
# change the distance -- and follows z again the moment the head is back
# under. On every frame at or under the threshold the crop's geometry is
# what it was before, by construction; only the size filter's memory of a
# turn differs on the frames after one (0.0002 px past 45 frames, nothing
# past 52). 0 turns it off (the sweep of depth_probe.py).
_DEFAULT_DEPTH_HOLD_YAW = 35.0
# The depth held is read on frames within this many degrees of yaw from
# the rest, where the fit is trusted (above), as the MEDIAN of the last
# _DEPTH_TRUST_FRAMES such readings: the first version held the last
# frame's own z, and on his clip that was once a face the second look
# found for a single frame at pitch +29 with z 16% off -- the next profile
# was drawn 15% too small for six frames. His real leans move z by at
# most 1.6% in a frame (p99 0.8%), so five frames cost nothing he can see
# and one bad frame cannot reach the hold.
_DEPTH_TRUST_YAW = 20.0
_DEPTH_TRUST_FRAMES = 5


def _depth_hold_yaw(options: dict[str, Any]) -> float:
    """``depth_hold_yaw`` -- degrees of yaw from the rest past which the
    crop's depth is held (default :data:`_DEFAULT_DEPTH_HOLD_YAW`); 0 off."""
    yaw = float(options.get("depth_hold_yaw", _DEFAULT_DEPTH_HOLD_YAW))
    if yaw < 0.0:
        raise ValueError(f"depth_hold_yaw must be >= 0, got {yaw}")
    return yaw


def _held_depth(loaded: _Portrait, head_pose: HeadPose, z: float) -> float:
    """The depth the crop is sized from this frame: ``z`` itself at or under
    :func:`_depth_hold_yaw` of yaw from the rest -- and within
    :data:`_DEPTH_TRUST_YAW` remembered, the median of the last
    :data:`_DEPTH_TRUST_FRAMES` such readings kept as ``z_frontal`` -- the
    remembered depth past it; ``z`` past it when no trusted frame has been
    seen yet (found first at a turn)."""
    hold_yaw = _depth_hold_yaw(loaded.options)
    yaw = abs(float(head_pose.yaw) - (loaded.rest_yaw or 0.0))
    if hold_yaw <= 0.0 or yaw <= hold_yaw:
        if yaw <= _DEPTH_TRUST_YAW:
            loaded.z_recent.append(z)
            loaded.z_frontal = float(statistics.median(loaded.z_recent))
        return z
    return z if loaded.z_frontal is None else loaded.z_frontal


# Day 22 evening, his held press at a turn to his right (tracker yaw -45):
# the avatar came out at -37, and at his +55 at +37..40 -- the network
# turns 1:1 to about 35 degrees and saturates past it (the cap is not it:
# 45, 60 and 90 gave the same -37). With the drawn head 8-18 degrees
# short, its near-side eye and brow sit inward of his, and his own show
# through the paste's feather beside them ("my brow and half an eye"). So
# the yaw delta past _TURN_BOOST_KNEE degrees is driven harder by
# ``turn_boost`` (an option, default 0 = off, bit-identical): delta +
# boost * (|delta| - knee), then the clamp as before. Pitch and roll are
# not touched.
_TURN_BOOST_KNEE = 30.0
_TURN_BOOST_MAX = 2.0
# The boosted yaw needs a higher clamp than hold_yaw_degrees (45, which is
# also where the expression starts to fade -- day 17's smeared mouth at
# -48 -- and must stay there): measured on his session's left turn (+55),
# a delta let through to 75 breaks the picture (a near-profile with the
# portrait's hair smeared over the far cheek); at 56 (his held press at
# -45 with boost 1.0) it is whole and reads -44. ``turn_cap_degrees`` is the
# clamp's own yaw limit; None (the default) keeps it at hold_yaw_degrees.
_TURN_CAP_MAX = 90.0


def _turn_cap(options: dict[str, Any], hold_yaw: float) -> float:
    """The clamp's yaw cap in degrees: ``turn_cap_degrees`` in
    ``(0, 90]``, or ``hold_yaw`` when it is not set."""
    raw = options.get("turn_cap_degrees")
    if raw is None:
        return hold_yaw
    value = float(raw)
    if not 0.0 < value <= _TURN_CAP_MAX:
        msg = f"turn_cap_degrees must be in (0, {_TURN_CAP_MAX}], got {value}"
        raise ValueError(msg)
    return value


def _turn_boost(options: dict[str, Any]) -> float:
    """``turn_boost`` in ``[0, 2]``, default 0: how much harder the yaw
    delta past :data:`_TURN_BOOST_KNEE` degrees drives the head."""
    value = float(options.get("turn_boost", 0.0))
    if not 0.0 <= value <= _TURN_BOOST_MAX:
        msg = f"turn_boost must be in [0, {_TURN_BOOST_MAX}], got {value}"
        raise ValueError(msg)
    return value


# Day 23 (2026-10-16): the turn is not symmetric on him. His session's turn
# step through the engine, the drawn face read by MediaPipe
# (scripts/probes/turn_motion.py): to his left the avatar turns 1.0-1.1 of
# his angle up to the network's own saturation, to his right 0.7-0.8 (his
# -23 drawn -16, his -32 drawn -26) -- on the 09-18 tree the same, and on
# the shaved avatar the same, so it is his side (the extractor's reading
# of his face), not the portrait's. A constant gain per side on the yaw
# delta, before the clamp: ``turn_gain_left`` for a delta toward his left
# (yaw +), ``turn_gain_right`` for one toward his right, both 1.0 by
# default (bit-identical: the block is skipped), each within
# [_TURN_GAIN_MIN, _TURN_GAIN_MAX]. A constant gain keeps the response
# monotonic (the clamp at hold_yaw_degrees still bounds the input to the
# network's monotonic range), where the day-22 knee boost did not.
_TURN_GAIN_MIN = 0.5
_TURN_GAIN_MAX = 2.0


def _turn_gains(options: dict[str, Any]) -> tuple[float, float]:
    """``(turn_gain_left, turn_gain_right)``, each 1.0 by default."""
    gains = []
    for key in ("turn_gain_left", "turn_gain_right"):
        value = float(options.get(key, 1.0))
        if not _TURN_GAIN_MIN <= value <= _TURN_GAIN_MAX:
            msg = f"{key} must be in [{_TURN_GAIN_MIN}, {_TURN_GAIN_MAX}], got {value}"
            raise ValueError(msg)
        gains.append(value)
    return gains[0], gains[1]


_STEEP_POSE_MODES = ("hold", "clamp")
# "clamp" since day 17 (his "head down, looking to the side, the face must
# not get lost"): on natural.mkv's tilt step the hold froze 165 of 296
# frames, the avatar's eyes left down while he looked up at the camera;
# the clamp draws every frame. "hold" stays as the option, and as what
# the tests of the hold itself set.
_DEFAULT_STEEP_POSE = "clamp"
# Past the yaw cap the expression fades to rest over this many degrees
# (steep_pose = "clamp"): at a near profile the far half of the face is
# hidden and the tracker's mouth, gaze and brow readings are noise --
# measured on driving.mkv 1592 (yaw -48): a wall-eyed gaze and a smeared
# mouth corner with the expression left on. The head is still drawn at
# the cap; the blink stays (the near eye reads).
_STEEP_FADE_DEGREES = 15.0


def _steep_fade(loaded: _Portrait, head_pose: HeadPose | None) -> float:
    """How much of the expression drives this frame under the clamp: 1.0
    within the yaw cap, 0.0 :data:`_STEEP_FADE_DEGREES` past it, linear
    between; 1.0 always with ``steep_pose = "hold"`` or no pose."""
    if not loaded.steep_clamp or head_pose is None:
        return 1.0
    _, hold_yaw = _hold_thresholds(loaded.options)
    yaw = abs(head_pose.yaw - (loaded.rest_yaw or 0.0))
    return float(1.0 - min(max((yaw - hold_yaw) / _STEEP_FADE_DEGREES, 0.0), 1.0))


# The gaze fades earlier than the rest of the expression under the clamp:
# the iris offset is read against the eye corners, and past 30 degrees of
# yaw the far eye's corners crowd together and the reading over-shoots --
# a wall-eyed avatar on driving.mkv 1498/1592 (yaw +38 / -48) with the
# fade starting only at the cap. From 30 to 45 degrees the gaze goes to
# rest; the blink (a ratio of one eye) is not touched.
_GAZE_YAW_FADE = (30.0, 45.0)


def _gaze_yaw_fade(loaded: _Portrait, head_pose: HeadPose | None) -> float:
    """How much of the gaze drive reaches the eyes this frame under the
    clamp: 1.0 up to :data:`_GAZE_YAW_FADE`'s start, 0.0 from its end;
    1.0 always with ``steep_pose = "hold"`` or no pose."""
    if not loaded.steep_clamp or head_pose is None:
        return 1.0
    yaw = abs(head_pose.yaw - (loaded.rest_yaw or 0.0))
    return float(1.0 - _soft_gate(yaw, _GAZE_YAW_FADE))


def _steep_pose(options: dict[str, Any]) -> str:
    """``steep_pose`` -- what happens past ``hold_pitch_degrees`` /
    ``hold_yaw_degrees``: ``"hold"`` (day 8: the last rendered face is
    frozen and only follows him) or ``"clamp"`` (day 17, the default, his
    "looking sideways, head down": the network keeps drawing every frame,
    the head's turn applied to the portrait is capped at the threshold,
    the expression -- mouth, eyes, brows -- goes on and fades to rest past
    the yaw cap, see :func:`_steep_fade`)."""
    mode = str(options.get("steep_pose", _DEFAULT_STEEP_POSE))
    if mode not in _STEEP_POSE_MODES:
        msg = f"steep_pose must be one of {_STEEP_POSE_MODES}, got {mode!r}"
        raise ValueError(msg)
    return mode


def _should_hold(loaded: _Portrait, head_pose: HeadPose | None) -> bool:
    """Whether this frame should reuse the last rendered face rather than
    ask the network to draw a pose a frontal portrait cannot hold.

    Never with ``steep_pose = "clamp"`` (day 17): the network draws every
    frame and :meth:`LivePortraitEngine._drive` caps the turn instead.

    Hysteresis: enters the hold above ``hold_pitch_degrees``/
    ``hold_yaw_degrees`` (:func:`_hold_thresholds`, default 20.0/35.0),
    leaves it only once back under :data:`_HOLD_RELEASE_RATIO` of that --
    without this, a pose sitting right at the threshold would enter and
    leave the hold on alternating frames. ``loaded.holding`` carries the
    state between frames; ``None`` (no usable pose this frame) leaves it
    exactly as it was, same as everywhere else a missing pose is treated as
    "no new information" rather than "assume the worst".

    This is a different mechanism from the *tracking-loss* hold in
    :meth:`LivePortraitEngine.animate` (``target is None``): there, his face
    is not in frame at all and the whole crop/mask/paste is skipped, reusing
    last frame's position too. Here his face is still tracked, just too
    steeply posed to redraw -- the crop and mask are still computed fresh
    every frame (see ``animate``), only the expensive network call is
    skipped, so the held face still follows him and is still masked/pasted
    at his actual, current position.
    """
    if loaded.steep_clamp:
        return False
    if head_pose is None:
        return loaded.holding
    hold_pitch, hold_yaw = _hold_thresholds(loaded.options)
    # Day 16: measured from the anchor's own rest pose once it is known --
    # the avatar's rotation is driven relative to it, so that is the pose
    # the frontal portrait is redrawn from. The friend's head rests at
    # pitch -11 in front of this camera; measured from zero, a nod of nine
    # degrees froze his avatar. Before the anchor, from zero as before.
    pitch = abs(head_pose.pitch - (loaded.rest_pitch or 0.0))
    yaw = abs(head_pose.yaw - (loaded.rest_yaw or 0.0))
    if loaded.holding:
        holding = pitch > hold_pitch * _HOLD_RELEASE_RATIO or yaw > hold_yaw * _HOLD_RELEASE_RATIO
    else:
        holding = pitch > hold_pitch or yaw > hold_yaw
    if holding != loaded.holding:
        _LOG.info(
            "Avatar %s: %s at pose %.1f/%.1f (pitch/yaw)",
            loaded.descriptor.id,
            "holding" if holding else "released",
            head_pose.pitch,
            head_pose.yaw,
        )
    loaded.holding = holding
    return holding


def _dark_cells(
    region: np.ndarray, alpha3: np.ndarray, coverage: np.ndarray, small: tuple[int, int]
) -> np.ndarray:
    """The correction's cells whose scene is not skin: darker than
    :data:`_EDGE_DARK_RATIO` of the face's own median lightness (the
    median over the cells the mask covers whole), as 1.0 / 0.0 per cell.
    All zeros when the mask covers too few cells whole to say what the
    face's lightness is."""
    import cv2

    scene = cv2.resize(cv2.multiply(region, alpha3), small, interpolation=cv2.INTER_AREA)
    return _dark_cells_small(scene, coverage)


def _dark_cells_small(scene: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """:func:`_dark_cells` from its quarter-scale inputs: ``scene`` is the
    mask-weighted room at the correction's cell size (the resize already
    done, on the CPU or the GPU), ``coverage`` the mask at the same size."""
    lightness = scene.mean(axis=2) / np.maximum(coverage, 1e-3)
    core = coverage > 0.9
    if int(core.sum()) < 16:
        return np.zeros(coverage.shape, dtype=np.float32)
    median = float(np.median(lightness[core]))
    dark: np.ndarray = ((lightness < _EDGE_DARK_RATIO * median) & (coverage > 0.05)).astype(
        np.float32
    )
    return dark


# ---------------------------------------------------------------------------
# The skin's tone (day 21, 2026-10-14)
# ---------------------------------------------------------------------------
#
# The friend test of 2026-10-13: two faces the avatar had never met, one
# bald, one with the hair pulled back, neither calibrated -- "a stripe on
# the forehead, the skin colour differed from the avatar's". The lighting
# plane and the edge residual above match the ROOM's light: a brightness
# and a gradient per channel over the whole face, and what that leaves at
# the join, faded over a fifth of the box. Neither matches the SKIN: a
# face paler, pinker or flatter-lit than the portrait's keeps the avatar's
# own tone in its middle, and meets the person's own skin -- above the
# join on a tall forehead, at the neck, the ears -- with a step in tone.
#
# So, before the plane, the render's skin is given the person's tone. On
# the cells of the correction's own grid that are skin on BOTH sides (the
# mask's core; not the scene's dark cells, not the render's dark cells --
# its brows, eyes, nostrils -- and not his feature zone, _EDGE_FEATURES),
# the mean and the spread of each channel are read for the room and for
# the render, and the render gets, per channel, a CAST -- ``mean_room /
# mean_render``, the room's white balance over the render, multiplied
# into every pixel -- and a CONTRAST about the room's skin mean --
# ``spread_room / (cast * spread_render)`` -- both within [_SKIN_GAIN_MIN,
# _SKIN_GAIN_MAX] (a dark room must not bleach the avatar, a shine must
# not blow it out), the contrast 1 when either spread is under
# _SKIN_SPREAD_MIN (a flat picture has no spread to match). The plane
# then has only the light's gradient left to fit, on its own cells as
# before; the residual is as before. The nine numbers (the cast, the
# contrast, the room's skin mean) are read one frame in
# _SKIN_TONE_EVERY -- the reading is two sorts and a few dozen small
# kernels, 2 ms of the render thread's CPU when it ran every frame
# (measured 2026-10-14) -- and carried by an EMA (_SKIN_TONE_INERTIA per
# reading) so a camera hunting its exposure cannot make the face pulse;
# the identity until the first reading with enough skin cells, not read
# while the face is held (the mask then sits where the face was); the
# tone is APPLIED on every frame. ``skin_match`` in ``engine_options``,
# 0..1, is how much is applied: 0 is the paste before this day to the
# byte (nothing of it is computed), 1 the whole match.
#
# What it cannot do: a shine (a bald scalp, a lit forehead) is texture,
# not tone; the plane and the residual still carry that.
#
# Where it is applied, settled over five replays of the woman's capture
# of 2026-09-12 20:07. A gain and an ADDITIVE offset per channel, on
# every pixel, turned the avatar's brown irises blue-grey: the channels'
# offsets differ, and an offset that is a tone on skin is a hue on a dark
# iris. The whole tone spared on the dark pixels (each pixel's lightness
# against the render's skin) left the avatar's stubble pale over hers
# (the jaw band +5.8 -> +11.1 grey levels) and the brows pale. The
# offset's luminance everywhere and its chroma by lightness brought the
# irises back and turned her avatar's beard orange: her warmer skin's
# chroma, added whole to the beard's mid-lit hairs. What is right is the
# room's WHITE BALANCE: a cast MULTIPLIED into every pixel moves a dark
# pixel by little and a skin pixel by the whole of it, so an iris, a
# brow, a beard keep their look while the skin takes the room's tone --
# the physics of a colour cast. The contrast (a spread matched on skin,
# a fact about skin's shading and not about a brow's darkness) goes to
# each pixel by its own lightness against the render's skin (the median
# of the core cells, the dark-cell rule's own number): all of it from
# _TONE_DARK_HIGH of the skin's lightness up, nothing under
# _TONE_DARK_LOW, linear between -- the bounds above the dark-cell rule's
# 0.55, since a brown brow's lit hairs read 0.5-0.7 of the skin and took
# a contrast under 1 (her flat-lit face against the portrait's studio
# shading) as a fading. (The feature zone's geometry was tried too: grown
# and softened it left the tone off exactly the cells it had been read
# from, and the plane put it back everywhere.)
_SKIN_MATCH_DEFAULT = 1.0
_TONE_DARK_LOW = 0.7
_TONE_DARK_HIGH = 0.9
_SKIN_GAIN_MIN = 0.7
_SKIN_GAIN_MAX = 1.4
# One reading in this many frames, and what the previous reading keeps
# at each: 0.66 a reading at one in four is 0.9 a frame, a time constant
# of about ten frames.
_SKIN_TONE_EVERY = 4
_SKIN_TONE_INERTIA = 0.66
# Cells of skin on both sides under which nothing is learnt this frame
# (the previous tone stays): a face mostly hair, or mostly turned away.
_SKIN_CELLS_MIN = 24
# A spread (grey levels, either side) under which there is no spread to
# match -- a flat picture, the tests' uniform faces -- and the gain is 1;
# the means are still matched.
_SKIN_SPREAD_MIN = 2.0
# The tone before any measurement: no cast, no contrast, no room skin
# mean yet (the rows). The first measurement is taken whole (no fade from
# here), the EMA runs from then.
_TONE_IDENTITY = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)


def _skin_match(options: dict[str, Any]) -> float:
    """``skin_match`` in ``[0, 1]``, default :data:`_SKIN_MATCH_DEFAULT`:
    how much of the skin-tone match is applied to the render before the
    lighting plane (see the section comment). 0 is the paste before day
    21, to the byte."""
    value = float(options.get("skin_match", _SKIN_MATCH_DEFAULT))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"skin_match must be in [0, 1], got {value}")
    return value


def _skin_cells(
    face: np.ndarray, scene: np.ndarray, coverage: np.ndarray, zone: np.ndarray | None
) -> np.ndarray:
    """The cells that are skin on both sides, 1.0 / 0.0: the mask's core
    (coverage over 0.9) less the scene's dark cells and the render's
    (:func:`_dark_cells_small` on each) and his feature zone (``zone``,
    :func:`_feature_exclusion`; ``None`` for none). ``face`` and ``scene``
    are mask-weighted, at the cell size."""
    skin: np.ndarray = (coverage > 0.9).astype(np.float32)
    skin = skin * (1.0 - _dark_cells_small(scene, coverage))
    skin = skin * (1.0 - _dark_cells_small(face, coverage))
    if zone is not None:
        skin = skin * (1.0 - zone)
    return skin.astype(np.float32)


def _skin_tone(
    face: np.ndarray, scene: np.ndarray, coverage: np.ndarray, skin: np.ndarray
) -> np.ndarray | None:
    """The per-channel ``(cast, contrast, room skin mean)`` -- a ``(3, 3)``
    float32: the cast that gives the render's skin the room's mean, the
    contrast about that mean that gives it the room's spread, and the
    room's skin mean -- from the mask-weighted ``face`` and ``scene`` at
    the cell size over the ``skin`` cells; ``None`` under
    :data:`_SKIN_CELLS_MIN` cells."""
    count = float(skin.sum())
    if count < _SKIN_CELLS_MIN:
        return None
    weight = np.maximum(coverage, 1e-3)[:, :, None]
    w = skin[:, :, None]
    x = scene / weight
    y = face / weight
    mean_s = (x * w).sum(axis=(0, 1)) / count
    mean_f = (y * w).sum(axis=(0, 1)) / count
    var_s = (((x - mean_s) ** 2) * w).sum(axis=(0, 1)) / count
    var_f = (((y - mean_f) ** 2) * w).sum(axis=(0, 1)) / count
    cast = np.clip(mean_s / np.maximum(mean_f, 1.0), _SKIN_GAIN_MIN, _SKIN_GAIN_MAX)
    floor = _SKIN_SPREAD_MIN**2
    ratio = np.clip(
        np.sqrt(var_s) / (cast * np.sqrt(np.maximum(var_f, 1e-6))), _SKIN_GAIN_MIN, _SKIN_GAIN_MAX
    )
    contrast = np.where((var_s >= floor) & (var_f >= floor), ratio, 1.0)
    return np.stack([cast, contrast, mean_s]).astype(np.float32)


def _tone_follow(previous: np.ndarray, measured: np.ndarray | None) -> np.ndarray:
    """The tone carried to the next frame: the EMA of the measured one
    over the previous (:data:`_SKIN_TONE_INERTIA`); the measured one whole
    when the previous is still the identity (nothing measured yet, so
    nothing to fade from); the previous alone when nothing was measured."""
    if measured is None:
        return previous
    if np.array_equal(previous[:2], _TONE_IDENTITY[:2]):
        return measured.astype(np.float32)
    followed: np.ndarray = previous * _SKIN_TONE_INERTIA + measured * (1.0 - _SKIN_TONE_INERTIA)
    return followed.astype(np.float32)


def _core_median(planes: np.ndarray, coverage: np.ndarray) -> float:
    """The median lightness of the cells the mask covers whole (coverage
    over 0.9) of a mask-weighted picture at the cell size -- the dark-cell
    rule's own reference; 0.0 when fewer than 16 cells are covered whole."""
    lightness = planes.mean(axis=2) / np.maximum(coverage, 1e-3)
    core = coverage > 0.9
    if int(core.sum()) < 16:
        return 0.0
    return float(np.median(lightness[core]))


def _tone_weight(placed: np.ndarray, level: float) -> np.ndarray | None:
    """How much of the tone each pixel of the render takes, by its own
    lightness against ``level`` (the render's skin, :func:`_core_median`):
    nothing under :data:`_TONE_DARK_LOW` of it, all of it from
    :data:`_TONE_DARK_HIGH`, linear between; ``None`` (everywhere) when
    the level is unknown."""
    if level <= 0.0:
        return None
    lightness = placed.mean(axis=2) / level
    weight: np.ndarray = np.clip(
        (lightness - _TONE_DARK_LOW) / (_TONE_DARK_HIGH - _TONE_DARK_LOW), 0.0, 1.0
    )
    return weight.astype(np.float32)


def _tone_on_skin(
    placed: np.ndarray,
    cast: np.ndarray,
    contrast: np.ndarray,
    mean: np.ndarray,
    weight: np.ndarray | None,
) -> np.ndarray:
    """The render with the tone applied (see the section comment): the
    cast multiplied into every pixel, the contrast about the room's skin
    mean ``mean`` by ``weight`` (:func:`_tone_weight`; ``None`` is
    everywhere)."""
    casted = placed * cast
    deepened = (contrast - 1.0) * (casted - mean)
    if weight is not None:
        deepened = deepened * weight[:, :, None]
    toned: np.ndarray = (casted + deepened).astype(np.float32)
    return toned


def _as_numpy_tone(tone: Any) -> np.ndarray:
    """The tone as a ``(2, 3)`` numpy array -- it is one already on the CPU
    paste; the GPU paste keeps it on the device (a read waits for it)."""
    if isinstance(tone, np.ndarray):
        return tone
    out: np.ndarray = tone.detach().float().cpu().numpy()
    return out


# The frame the tone is logged on: by then the EMA (0.9 a frame) has
# taken all but a percent of the first measurement's distance.
_TONE_LOG_FRAME = 60


def _tone_applied(tone: np.ndarray, amount: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(cast, contrast, room skin mean)`` with ``amount`` (``skin_match``)
    of ``tone``: the cast and the contrast pulled toward 1 by ``1 -
    amount``; the mean as read."""
    cast = 1.0 + (tone[0] - 1.0) * amount
    contrast = 1.0 + (tone[1] - 1.0) * amount
    return cast.astype(np.float32), contrast.astype(np.float32), tone[2].astype(np.float32)


@functools.lru_cache(maxsize=1024)
def _area_weights(n: int, m: int) -> np.ndarray:
    """OpenCV's ``INTER_AREA`` downscale from ``n`` to ``m`` samples as an
    ``(m, n)`` weight matrix, so the GPU paste can shrink a box exactly the
    way ``cv2.resize`` does on the CPU (2026-10-08). The rule is
    ``computeResizeAreaTab``'s: each output cell spans ``n/m`` input
    samples; whole samples weigh ``1/cell``, the two partial ones at its
    ends their overlap over ``cell``. For a whole-number ratio this is the
    plain box average OpenCV's fast path takes. Applied as ``Wy @ image @
    Wx.T``, which is how OpenCV applies it too -- rows, then columns."""
    if m > n:
        msg = f"area weights shrink only: {n} -> {m}"
        raise ValueError(msg)
    scale = n / m
    weights = np.zeros((m, n), dtype=np.float64)
    for d in range(m):
        fs1 = d * scale
        fs2 = fs1 + scale
        cell = min(scale, n - fs1)
        s1 = math.ceil(fs1)
        s2 = math.floor(fs2)
        s2 = min(s2, n - 1)
        s1 = min(s1, s2)
        if s1 - fs1 > 1e-3:
            weights[d, s1 - 1] += (s1 - fs1) / cell
        for s in range(s1, s2):
            weights[d, s] += 1.0 / cell
        if fs2 - s2 > 1e-3:
            weights[d, s2] += min(min(fs2 - s2, 1.0), cell) / cell
    return weights.astype(np.float32)


def _face_colour_correction(
    loaded: _Portrait,
    face: np.ndarray,
    region: np.ndarray,
    alpha: np.ndarray,
    alpha3: np.ndarray,
    features: np.ndarray | None = None,
) -> np.ndarray | None:
    """The smooth field that relights the generated face to the room --
    what is *added* to the face, full-box, float32; ``None`` when the mask
    covers too little of the box to fit anything.

    ``face`` and ``region`` are both already warped into the same small paste
    box: the generated face, and the scene pixels it is about to cover. The
    portrait was rendered under LivePortrait's own studio-normalised light;
    the room is lit by whatever the camera sees, and the two differ at the
    mask's edge exactly the way the warp engine's portrait and room did.

    The field is carried between frames with the same inertia the warp engine
    uses, for the same reason: a camera hunting its exposure must not make
    the face visibly pulse against a background that is only drifting.

    Until day 11 this returned the relit face itself (``clip(face +
    field)``); returning the field lets :meth:`LivePortraitEngine._paste`
    fold the relighting into its one blend pass and saturate once at the
    end -- the same arithmetic, fewer passes over the box (see there).

    Since day 13 the plane is not the whole field: what it leaves at the
    mask's edge is matched there too (:func:`_edge_residual`, the forehead
    line), by ``edge_match`` of it -- 0 is the plane alone. ``features``
    are his brow, eye and mouth points in this box's own pixels
    (:data:`_EDGE_FEATURES`, in that order), cut out of that residual;
    ``None`` cuts nothing.
    """
    import cv2

    height, width = face.shape[:2]
    small = (max(width // _CORRECTION_DOWNSCALE, 8), max(height // _CORRECTION_DOWNSCALE, 8))

    weight = alpha[:, :, 0]
    if float(weight.sum()) < 32.0:
        return None

    difference = cv2.resize(
        cv2.multiply(cv2.subtract(region, face), alpha3), small, interpolation=cv2.INTER_AREA
    )
    coverage = cv2.resize(weight, small, interpolation=cv2.INTER_AREA)
    scene = cv2.resize(cv2.multiply(region, alpha3), small, interpolation=cv2.INTER_AREA)
    field = _correction_field(loaded, difference, coverage, scene, features, (width, height))
    correction: np.ndarray = cv2.resize(field, (width, height), interpolation=cv2.INTER_LINEAR)
    return correction


def _correction_field(
    loaded: _Portrait,
    difference: np.ndarray,
    coverage: np.ndarray,
    scene: np.ndarray,
    features: np.ndarray | None,
    box_size: tuple[int, int],
) -> np.ndarray:
    """The relighting field at the correction's own cell size, from the
    quarter-scale inputs -- :func:`_face_colour_correction`'s arithmetic
    from the resizes on, shared since 2026-10-08 by the CPU paste and the
    GPU one (:meth:`LivePortraitEngine._colour_correction_gpu`), which
    computes the three inputs on the GPU and hands them over as numpy.

    ``difference`` is the mask-weighted room-minus-face, ``coverage`` the
    mask, ``scene`` the mask-weighted room, all at the cell size;
    ``features`` his feature points in the box's own pixels; ``box_size``
    the box's ``(width, height)`` the cells were shrunk from. The field is
    carried between frames in ``loaded.correction`` as before.
    """
    import cv2

    width, height = box_size
    small = (coverage.shape[1], coverage.shape[0])
    # Day 16 evening: cells that are not skin -- hair under the edge, a
    # brow's core, lashes -- are unknown to the plane and the residual
    # alike (see _EDGE_DARK_RATIO).
    dark = _dark_cells_small(scene, coverage)
    keep = 1.0 - dark
    field = _fit_lighting_plane(difference * keep[:, :, None], coverage * keep)
    # Day 13: the plane's residual matched at the mask's edge -- see
    # _edge_residual and the forehead-line comment above it.
    edge = _edge_match(loaded.options)
    if edge > 0.0:
        exclude = dark
        if features is not None:
            cells = features / (width / small[0], height / small[1])
            exclude = np.maximum(dark, _feature_exclusion(cells.astype(np.float32), coverage.shape))
        field = field + _edge_residual(difference, coverage, field, exclude=exclude) * edge

    previous = loaded.correction
    if previous is not None:
        if not isinstance(previous, np.ndarray):
            # Left by the GPU fit (see _fit_field_gpu): the same field, on
            # the device, channels first.
            previous = previous.permute(1, 2, 0).cpu().numpy()
        resized = cv2.resize(previous, (field.shape[1], field.shape[0]))
        field = resized * _CORRECTION_INERTIA + field * (1.0 - _CORRECTION_INERTIA)
    loaded.correction = field
    return field


# ---------------------------------------------------------------------------
# One loaded portrait
# ---------------------------------------------------------------------------


class _Portrait:
    """Everything LivePortrait needs of one portrait, on the GPU, plus the
    per-frame state: the anchor, the graph, and the last face for a hold."""

    __slots__ = (
        "R_d0",
        "R_s",
        "anchor_rejected",
        "anchor_seen",
        "anchor_sum",
        "anchor_wanted",
        "background",
        "blink_brow_hold",
        "blink_mouth_hold",
        "blink_mouth_lift",
        "brow_delta",
        "c_s_eye_mean",
        "c_s_lip",
        "centre_offset",
        "centre_samples",
        "chin_drop",
        "chin_drop_max",
        "chin_reach",
        "correction",
        "crop_filters",
        "crop_points",
        "crop_shift_x",
        "descriptor",
        "driver_reach",
        "ear_recent",
        "exp_d0",
        "exp_prev",
        "exp_s",
        "eye_mask",
        "eye_rest",
        "f_s",
        "face_height_rest",
        "features",
        "forehead_reach",
        "forehead_stretch",
        "gate",
        "gate_brow_prev",
        "gate_mouth_prev",
        "gate_open_prev",
        "gaze_prev",
        "graph",
        "guard_weight",
        "hair_box",
        "hair_matte",
        "hair_on",
        "hair_seq",
        "hair_skin",
        "hairline_clearance",
        "hairline_driver",
        "hairline_last",
        "hairline_portrait",
        "hairline_samples",
        "hairline_settled",
        "height_centre",
        "height_driver",
        "height_last",
        "height_samples",
        "height_scale",
        "holding",
        "last",
        "last_dt",
        "last_timestamp",
        "leak_gain",
        "leak_slopes",
        "lip_rest",
        "lip_rest_alpha",
        "lip_rest_mask",
        "lip_rest_rows",
        "lip_rest_run",
        "lip_rest_steps",
        "mar_prev",
        "mask",
        "mask_bbox",
        "mask_points",
        "mouth_hold",
        "mouth_leak",
        "open_hold",
        "options",
        "outline_scale",
        "outline_shift",
        "pivot",
        "portrait_crop",
        "pose_d0",
        "pose_limit",
        "pose_prev",
        "pose_recent",
        "pose_samples",
        "ratios",
        "rel_mask",
        "rest_brow",
        "rest_chin",
        "rest_ear",
        "rest_gaze",
        "rest_mar",
        "rest_mouth_width",
        "rest_pitch",
        "rest_yaw",
        "roots_portrait",
        "scale_d0",
        "scale_gain",
        "scale_prev",
        "scale_s",
        "settle_hairline",
        "settle_height",
        "settle_width",
        "size_samples",
        "static_in",
        "static_out",
        "steep_clamp",
        "stretch_row",
        "t_d0",
        "t_gain",
        "t_prev",
        "t_s",
        "thresholds",
        "tone",
        "tone_frames",
        "tone_level",
        "tone_logged",
        "turn_boost",
        "turn_gain",
        "vram_mb",
        "width_centre",
        "width_driver",
        "width_last",
        "width_prev",
        "width_samples",
        "width_scale",
        "width_settled",
        "x_c_s",
        "x_s",
        "yaw_pivot",
        "yaw_prev",
        "z_frontal",
        "z_recent",
        "z_rest",
    )

    def __init__(
        self, descriptor: AvatarDescriptor, driver_options: Mapping[str, Any] | None = None
    ) -> None:
        self.descriptor = descriptor
        # The portrait's own options, with the driver's profile laid over
        # them (day 16; see DRIVER_OPTION_KEYS -- the engine's
        # set_driver_options has already refused any other key).
        self.options = dict(descriptor.manifest.engine_options)
        if driver_options:
            self.options.update(driver_options)
        self.anchor_wanted = int(self.options.get("anchor_frames", DEFAULT_ANCHOR_FRAMES))
        # Day 16: this driver's gate thresholds, brow-to-mouth leak table
        # and measured lip rest -- each the module constant unless the
        # profile set it. Validated here, once, like the rest of the options
        # load_avatar reads.
        self.thresholds: _GateThresholds = _gate_thresholds(self.options)
        self.leak_slopes: dict[tuple[int, int], float] = _brow_mouth_leak_slopes(self.options)
        self.lip_rest_rows: tuple[float, ...] | None = _lip_rest_rows(self.options)
        self.anchor_seen = 0
        # How many candidate frames failed _anchor_frame_ok while the anchor
        # was still being learned -- logged once the anchor freezes, and
        # reset alongside it. See reset_anchor and _anchor_frame_ok.
        self.anchor_rejected = 0
        # Running sums for the anchor: exp (21x3), t (3), scale (1), the pose
        # in degrees (pitch, yaw, roll), and the accepted frames' own "ear"
        # and "mar" (plain Python floats; "mar" is the gates' opening over
        # the jaw width since day 12, _mouth_open_ratio) -- averaged the same
        # way as the rest, into rest_ear/rest_mar below.
        self.anchor_sum: dict[str, Any] = {}
        # The driver's own eye/mouth openness at rest, once the anchor is
        # frozen -- the target _eye_target/_lip_target correct toward, so a
        # driver whose eyes are naturally narrower than the portrait's is not
        # asked to narrow the avatar just for sitting still. None until
        # anchor_seen reaches anchor_wanted. See _render/animate.
        self.rest_ear: float | None = None
        self.rest_mar: float | None = None
        # Day 9, 2026-09-06: his mouth width and brow height at rest (see
        # _mouth_width_ratio/_brow_ratio), learned and frozen with the
        # anchor the same way, for the expression gates (_mouth_gate/
        # _brow_gate) to measure his deviation against. None until then --
        # and until then both gates are 1.0.
        self.rest_mouth_width: float | None = None
        self.rest_brow: float | None = None
        # Day 13: his resting nose-tip-to-chin reach in crop pixels
        # (_nose_chin), learned with the anchor like rest_brow; the
        # portrait's own points in its crop, kept so its oval can be
        # rebuilt per frame with the jaw let down (_frame_mask); the
        # drop's limit in crop pixels (_CHIN_DROP_MAX of the portrait's
        # face height, set at load, the mask box's own reach below); and
        # the last frame's drop, for the record (measure_follow reads it).
        self.rest_chin: float | None = None
        self.crop_points: np.ndarray | None = None
        # Day 13, his 2026-09-09 look: his brow, eye and mouth points in
        # frame pixels (_EDGE_FEATURES order), from the last tracked frame
        # -- what the edge residual must never print (see _feature_
        # exclusion). Kept through a lost face like `last` is.
        self.features: np.ndarray | None = None
        self.chin_drop_max: float = 0.0
        self.chin_drop: float = 0.0
        # Day 13, his 03:28 captures: how far both ovals reach below the
        # chin, crop px (mask_chin_reach of the crop face height, at load).
        # Day 16: "auto" is the mask's own clearance (see _auto_chin_reach).
        self.chin_reach: float = 0.0
        # Day 16 (C1): the forehead reach in force -- the option's number,
        # or with "hairline" the default until the anchor settles it from
        # the portrait's hairline (read at load) and the driver's (sampled
        # on frontal frames until the anchor freezes).
        self.forehead_reach: float = _forehead_reach(self.options)
        # Day 16 (C2): the driver crop's constant sideways shift, crop px.
        self.crop_shift_x: float = _crop_shift_x(self.options)
        self.hairline_portrait: float | None = None
        self.hairline_samples: list[float | None] = []
        self.hairline_settled: bool = not _forehead_reach_is_hairline(self.options)
        # Day 17: the driver's hairline from the profile (Calibrate me), the
        # last frontal reading (the calibration's instrument), and the
        # clearance under the lower hairline.
        self.hairline_driver: float | None = _driver_hairline(self.options)
        self.hairline_last: float | None = None
        # Day 22: the width fit -- the profile's ratio (None: read live),
        # the still frontal readings before the settle, the last reading,
        # the scale in force (1: none), the portrait rim's middle column it
        # is taken about, and the portrait's points as the masks see them
        # (None: crop_points, unsqueezed). See _settle_forehead.
        self.width_driver: float | None = _driver_face_width(self.options)
        self.width_samples: list[float] = []
        self.width_last: float | None = None
        self.width_scale: float = 1.0
        self.width_centre: float = 0.0
        self.mask_points: np.ndarray | None = None
        self.width_settled: bool = False
        # Day 23: the height fit beside it (see _brow_chin, _height_scale):
        # the readings, the last one, the scale in force (1: none) and the
        # portrait's chin row it is taken about; settled with the width.
        self.height_driver: float | None = _driver_face_height(self.options)
        self.height_samples: list[float] = []
        self.height_last: float | None = None
        self.height_scale: float = 1.0
        self.height_centre: float = 0.0
        # Day 22 evening: the outline fit's per-row map in force (crop
        # rows; None off a turn) -- see _outline_maps.
        self.outline_scale: np.ndarray | None = None
        self.outline_shift: np.ndarray | None = None
        # Day 22: the portrait's roots (see _roots_onset), and what the
        # settle was made from -- their hairline and the width ratio -- for
        # the capture's anchor state, so a replay plans the same forehead.
        self.roots_portrait: float | None = None
        self.settle_hairline: float | None = None
        self.settle_width: float | None = None
        self.settle_height: float | None = None
        self.hairline_clearance: float = _hairline_clearance(self.options)
        # Day 21: the driver oval's own reach once the forehead is settled
        # (None: the portrait's, as before the settle), the render's stretch
        # above ``stretch_row`` (a crop row; None: no stretch) -- see
        # _forehead_plan.
        self.driver_reach: float | None = None
        self.forehead_stretch: float = 1.0
        self.stretch_row: float | None = None
        # Day 17 (C1b): the avatar's hair. ``hair_on`` is the mode in force
        # (the ``hair`` option, or set_hair's word); ``hair_matte`` the EMA
        # of the rendered crop's hair alpha, (512, 512, 1) float32 like
        # ``mask``, the paste takes the maximum of the two; ``hair_seq``
        # the worker's sequence number of the last matte folded in;
        # ``hair_box`` the crop box the hair has reached so far (only ever
        # grows, so the correction's field stays in one frame of
        # reference) -- the paste's box is its union with ``mask_bbox``.
        self.hair_on: bool = _hair_mode(self.options) == "avatar"
        self.hair_matte: np.ndarray | None = None
        self.hair_seq: int = 0
        self.hair_box: tuple[int, int, int, int] | None = None
        # How much of the render's face skin joins the paste, per row of
        # the crop -- above the eyes only (see hair.skin_weight).
        self.hair_skin: np.ndarray | None = None
        # The portrait's own 512 crop (BGR), kept for the hair's first matte
        # when the hair is switched on after the load.
        self.portrait_crop: np.ndarray | None = None
        # Day 19: the photograph's flat background colour (BGR), read at
        # load from its border, or None when the border is not flat or the
        # guard is off -- see _background_colour and _background_keep.
        self.background: np.ndarray | None = None
        # How much of the guard the last rendered frame gets (the turn's
        # weight, see _background_weight); kept for a held frame.
        self.guard_weight: float = 0.0
        # Day 11: his resting iris offset (x, y) in units of eye width,
        # learned with the anchor like rest_brow; and the gaze drive's own
        # EMA state -- see the GAZE section and _gaze_drive.
        self.rest_gaze: tuple[float, float] | None = None
        self.gaze_prev: float = 0.0
        # Day 15: the lip rest tracking (see _lip_rest_tracking) -- the
        # weight one resting frame moves the anchor's lip rows by (0 off),
        # the (1, NUM_KP, 3) mask of those rows, and how many resting frames
        # in a row the gates have called so far.
        self.lip_rest_alpha: float = 0.0
        self.lip_rest_mask: Any = None
        self.lip_rest_run: int = 0
        # Day 16 (C3): resting frames applied since the freeze (or the
        # last reset/seed) -- the fast start counts these.
        self.lip_rest_steps: int = 0
        # Day 12: his eye ratio on the last few frames, for _relieved_ear to
        # tell a blink's speed from a glance down (gaze_down_relief).
        self.ear_recent: deque[float] = deque(maxlen=_BLINK_WINDOW)
        # The gates as last applied, for _release's own decay: 0.0 to start
        # -- nothing to release from. Untouched while the rest is unknown
        # (the gates are 1.0 then, outright), so the first gated frame after
        # the anchor freezes is exactly its own reading.
        self.gate_mouth_prev: float = 0.0
        self.gate_open_prev: float = 0.0
        self.gate_brow_prev: float = 0.0
        # His mouth reading on the last gated frame (for _mouth_activity;
        # the opening over the jaw width since day 12, like rest_mar) and
        # how many more frames the mouth gate stays open because it was
        # moving (see _MOUTH_HOLD_FRAMES). None/0: nothing to compare yet.
        self.mar_prev: float | None = None
        self.width_prev: float | None = None
        self.yaw_prev: float | None = None
        self.mouth_hold: int = 0
        self.open_hold: int = 0
        # The crop's One Euro filters (centre x, centre y, size) and the
        # last frame's capture timestamp they step by -- see _crop_filters
        # and LivePortraitEngine._smooth_crop. None: smoothing off.
        self.crop_filters: _CropFilters | None = None
        self.last_timestamp: float | None = None
        # The time the last frame took (from the capture timestamps, see
        # _frame_dt): the mouth gate's motion units are per 30-fps frame, and
        # the live app ran at 22 fps on 2026-09-06 -- a frame there sees 1.4x
        # the change of a 30-fps frame for the same motion.
        self.last_dt: float = _DEFAULT_FRAME_DT
        # The portrait's own eye/lip close ratios, as plain floats -- read
        # every frame by _eye_target/_lip_target, which would otherwise force
        # a GPU sync each time to pull them back out of the `ratios` tensor.
        # Set once in load_avatar, alongside `ratios` itself.
        self.c_s_eye_mean: float = 0.0
        self.c_s_lip: float = 0.0
        # The driver's face height at rest (forehead top to chin, pixels) and
        # his distance then (the head pose's translation z, model units) --
        # medians over the anchor's accepted frames, frozen with it. Together
        # they size the driver's crop every frame afterwards, see
        # _driver_face_height. None until the anchor freezes, during which
        # the crop is sized from the frame itself (_face_height).
        self.size_samples: list[tuple[float, float]] = []
        self.face_height_rest: float | None = None
        self.z_rest: float | None = None
        # Day 20: the depth he sits at -- the median of the pose's z over
        # the last few frames within _DEPTH_TRUST_YAW of the rest -- what
        # the crop is sized from past depth_hold_yaw (see _held_depth).
        # None until such a frame; kept across a loss.
        self.z_recent: deque[float] = deque(maxlen=_DEPTH_TRUST_FRAMES)
        self.z_frontal: float | None = None
        # Day 11: the crop centre's fixed vertical offset off the eye line,
        # in units of the crop size -- collected on accepted anchor frames
        # when crop_centre is "eyeline" and frozen as their median with the
        # anchor; None (before the freeze, or with "rigid") keeps the rigid
        # box (see _crop_geometry's own docstring).
        self.centre_samples: list[tuple[float, float]] = []
        self.centre_offset: float | None = None
        self.rest_pitch: float | None = None
        # Day 16: the anchor's own yaw, frozen with it -- the extreme-pose
        # hold measures from here and from rest_pitch (see _should_hold);
        # and the pose window the anchor gate reads stillness from (see
        # _pose_still), kept across a re-centre: it is about the last half
        # second, whichever anchor is being learned.
        self.rest_yaw: float | None = None
        self.pose_recent: deque[tuple[float, float, float]] = deque(maxlen=_ANCHOR_STILL_FRAMES)
        # The TRACKER's (pitch, yaw) on accepted anchor frames, their medians
        # frozen as rest_pitch (when the crop centre does not set it) and
        # rest_yaw. The network's own pose average (pose_d0) is in other
        # degrees: on the first other face it read the anchor at yaw +4.0
        # where the tracker, which the hold and the calibration measure in,
        # read -8.5.
        self.pose_samples: list[tuple[float, float]] = []
        # The portrait's own keypoint state and appearance feature, and the
        # driver's anchor, all set once in load_avatar and read every frame
        # by _drive. Declared here, not there, so their type is known before
        # first use -- a slot with no annotation and no __init__ assignment
        # is invisible to mypy, which is what let eleven attribute typos on
        # this class go unnoticed until now. All are torch tensors once set;
        # kept as Any, matching graph/static_in/static_out below, because
        # torch is imported lazily and this module must not force it in.
        self.f_s: Any = None
        self.x_c_s: Any = None
        # Day 10, Fable's review of Step 1: where the relative rotation
        # turns the canonical keypoints about -- (1, 1, 3), a fixed point
        # on the portrait's own face (see _rotation_pivot, EYE_KEYPOINTS),
        # or None for the keypoint-space origin (the old, only, behaviour).
        # Set once in load_avatar, next to x_c_s -- computed from it, so it
        # must not move once the graph is captured.
        self.pivot: Any = None
        # Day 16 (C2): the yaw delta's own pivot, (1, 1, 3) at (0, 0, z) --
        # None when yaw_pivot_z is 0. Set once in load_avatar.
        self.yaw_pivot: Any = None
        self.x_s: Any = None
        self.exp_s: Any = None
        self.scale_s: Any = None
        self.t_s: Any = None
        self.R_s: Any = None
        self.exp_d0: Any = None
        self.scale_d0: Any = None
        self.t_d0: Any = None
        # Day 11: (1, 3) static gains on the extractor's relative
        # translation, [x, y, 0] -- see _translation_gain; 1.0 is exact.
        self.t_gain: Any = None
        # Day 13: a (1, 1) static gain on the extractor's relative scale --
        # see _scale_gain; 1.0 is the old line exactly, 0.0 (the default)
        # the portrait's own scale whatever the extractor reads.
        self.scale_gain: Any = None
        self.R_d0: Any = None
        # The anchor's own pose in degrees (3, 1), the same stack
        # _learn_anchor's own "pose" sum and pose_prev are -- R_d0 stores
        # the matrix built from it, this stores the degrees themselves, so
        # the pitch component alone can be pulled back out for the day-10
        # per-axis pivot (see _rotation_pivot, _drive's "pitch_only"-style
        # branch). Primed in load_avatar from the same reading pose_prev
        # is; re-primed alongside it in _learn_anchor; reset_anchor leaves
        # it, as it leaves R_d0.
        self.pose_d0: Any = None
        # The day-7 temporal smoothing's own EMA state (see _ema_,
        # _motion_smoothing, and _drive's own comment): the previous
        # frame's exp/t/scale, and the previous frame's pitch/yaw/roll in
        # degrees stacked the same way _learn_anchor's own "pose" sum is
        # (3, 1). Set for real in load_avatar (primed from exp_s/t_s/
        # scale_s and the portrait's own raw pose, not zero -- see the
        # comment there for why) and re-primed to the anchor's own running
        # mean on every accepted anchor frame, in _learn_anchor.
        self.exp_prev: Any = None
        self.t_prev: Any = None
        self.scale_prev: Any = None
        self.pose_prev: Any = None
        # The day-8 region masks (see _region_masks, EYE_KEYPOINTS): both
        # (1, NUM_KP, 3), set for real in load_avatar and read every frame
        # by _drive -- rel_mask multiplies the relative expression,
        # eye_mask the eye retargeting network's own output.
        self.rel_mask: Any = None
        self.eye_mask: Any = None
        # The day-9 expression gate, (1, NUM_KP, 3), multiplied into the
        # relative expression alongside rel_mask: ones everywhere except
        # MOUTH_KEYPOINTS'/BROW_KEYPOINTS' rows, which _render writes this
        # frame's _mouth_gate/_brow_gate into before every replay, the same
        # way it writes ``ratios``. Set for real in load_avatar.
        self.gate: Any = None
        # The brow offset (day 9, later; see _brow_lift/_BROW_LIFT_Y),
        # (1, NUM_KP, 3), added to the driving keypoints in _drive: zeros
        # except BROW_KEYPOINTS' y, which _render writes each frame from his
        # own measured brow height. Set for real in load_avatar.
        self.brow_delta: Any = None
        # Day 15: the brow raise's leak into the mouth rows this frame,
        # (1, NUM_KP, 3), subtracted from the relative expression in _drive
        # -- written by _render from his measured brow deviation and the
        # slopes in _BROW_MOUTH_LEAK, scaled by ``leak_gain`` (the
        # brow_mouth_leak option, read once at load).
        self.mouth_leak: Any = None
        self.leak_gain: float = 0.0
        # The eye/lip retargeting networks' own output at the portrait's own
        # ratio -- what each network says when asked for *no change*. Not
        # zero: both were trained on LivePortrait's own landmark scheme's
        # ratios, and fed MediaPipe's they answer "close a little" to a
        # target equal to the portrait's own openness (measured 2026-09-06,
        # eye network: +0.089/+0.111 on the upper lids at rest, *more* than
        # the whole difference between its rest and full-blink outputs).
        # Computed once at load (_retarget_rest) and subtracted in _drive,
        # so at rest the networks add exactly nothing. Both (1, NUM_KP, 3).
        self.eye_rest: Any = None
        self.lip_rest: Any = None
        self.graph: Any = None
        self.static_in: Any = None
        self.static_out: Any = None
        self.vram_mb: int | None = None
        # The eye/lip retargeting networks' own input, as one static (1, 5)
        # tensor read inside the graph: [portrait's left eye ratio, its right,
        # this frame's driving eye target, the portrait's lip ratio, this
        # frame's driving lip target]. The first, second and fourth never
        # change after load; _render updates the third and fifth in place,
        # the same way static_in is updated, before every replay.
        self.ratios: Any = None
        # The last generated face, where it went, and the mask it was pasted
        # with, for the frames the tracker loses: the face stays, the room
        # keeps moving, and the hold needs the same mask a fresh frame would
        # have built so a lost-tracking frame does not flash a different edge.
        # The face is a numpy picture on the CPU path and, since 2026-10-08,
        # a GPU tensor (3 x 512 x 512 float, the frame's channel order) on
        # the GPU one -- see LivePortraitEngine._paste.
        self.last: tuple[Any, np.ndarray, np.ndarray, np.ndarray | None] | None = None
        # The portrait's own oval, feathered, in the 512x512 crop the network
        # always draws into — computed once at load from the portrait's own
        # landmarks and reused every frame as one half of :func:`_frame_mask`.
        # See :func:`_oval_mask`.
        self.mask: np.ndarray | None = None
        self.mask_bbox: tuple[int, int, int, int] | None = None
        # The low-frequency relighting field, carried between frames the same
        # way the warp engine carries it — see :data:`_CORRECTION_INERTIA`.
        # A numpy field on the CPU paste; a device tensor on the GPU one
        # (2026-10-08, see _fit_field_gpu).
        self.correction: Any = None
        # Day 21: the skin's tone in force -- the cast, the contrast and the
        # room's skin mean, per channel in the frame's channel order, a
        # (3, 3) array: numpy on the CPU paste, a device tensor on the GPU
        # one; the identity until the first frame with enough skin on both
        # sides, then an EMA (see _skin_tone). ``tone_logged``: the one log
        # line, written once the EMA has settled.
        self.tone: Any = _TONE_IDENTITY.copy()
        # The render's skin level (its core median) from the last reading,
        # for the contrast's weight on every frame -- a float on the CPU
        # paste, a device scalar on the GPU one; 0.0 is "unknown"; and
        # how many frames this paste has seen, for the readings' cadence.
        self.tone_level: Any = 0.0
        self.tone_frames: int = 0
        self.tone_logged: bool = False
        # Day 8, 2026-09-06: whether the last rendered face is currently
        # being reused rather than redrawn, at a pose too steep for a
        # frontal portrait -- see _should_hold. Starts un-held; reset_anchor
        # also clears it, so a fresh anchor starts un-held too.
        self.holding: bool = False
        # Day 17: steep_pose = "clamp" -- the turn applied to the portrait
        # is capped at the hold thresholds instead of the face freezing;
        # ``pose_limit`` is the (3, 1) static tensor of degrees the graph
        # clamps against (pitch, yaw, roll -- roll unbounded).
        self.steep_clamp: bool = _steep_pose(self.options) == "clamp"
        self.pose_limit: Any = None
        # Day 22: the turn boost past the knee, a (3, 1) static tensor the
        # graph reads (0 for pitch and roll); None at a boost of 0.
        self.turn_boost: Any = None
        self.turn_gain: Any = None
        # Day 17: the blink's counter on the brows and the mouth corners
        # (see _BLINK_BROW_HOLD), read once at load.
        self.blink_brow_hold: float = _blink_brow_hold(self.options)
        self.blink_mouth_hold: float = _blink_mouth_hold(self.options)
        self.blink_mouth_lift: float = _blink_mouth_lift(self.options)


class LivePortraitEngine:
    """Drives a portrait with LivePortrait on CUDA."""

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        weights: Path | str | None = None,
        driver_options: Mapping[str, Any] | None = None,
        hair_model: Path | str | None = None,
    ) -> None:
        # The face-landmark model, to read the portrait's own face at load and
        # to name the driver's crop; the same one the tracker uses.
        self._model_path = Path(model_path) if model_path is not None else None
        self._weights = Path(weights) if weights is not None else None
        # Day 17 (C1b): the hair segmenter's model, beside the landmarker by
        # default (see sha256.avatar.hair); started the first time an
        # avatar's hair is asked for, never before.
        self._hair_model = Path(hair_model) if hair_model is not None else None
        self._hair: HairSegmenter | None = None
        self._hair_worker: HairMatteWorker | None = None
        self._hair_unavailable = False
        # set_hair's word per avatar, kept across loads: the Menu's switch
        # may be thrown before the avatar is loaded, or the avatar reloaded
        # for a profile, and the hair must not fall back to the file's say.
        self._hair_wanted: dict[str, bool] = {}
        # Day 16: the driver profile's options, laid over every avatar's own
        # at load -- see set_driver_options.
        self._driver_options: dict[str, Any] = {}
        if driver_options:
            self.set_driver_options(driver_options)
        self._running = False
        self._torch: Any = None
        self._device: Any = None
        self._nets: dict[str, Any] = {}
        self._loaded: dict[str, _Portrait] = {}
        # How long the last frame's parts took, in ms, for the offline harness
        # and the debug panel. Not a measurement of anything until read.
        self.last_timings: dict[str, float] = {}
        # 2026-10-08 ("The fast version", step 2): the paste on the GPU when
        # the engine is on CUDA (see _paste / _paste_gpu); off, the face is
        # downloaded and pasted on the CPU as before -- the probes compare
        # the two. The CUDA events time the graph without waiting for it;
        # the area matrices are the GPU paste's resize weights, per size.
        self.paste_on_gpu = True
        self.fit_on_gpu = True
        self._area_cache: dict[tuple[Any, ...], Any] = {}
        self._pinned: dict[tuple[str, str], Any] = {}
        self._model_events: tuple[Any, Any] | None = None
        self._events_recorded = False
        self._paste_wait_ms = 0.0
        self._upload_ms = 0.0
        self._paste_marks = (0.0, 0.0, 0.0)
        # Day 22: the last frame's channel order, for the GPU paste's chin keep.
        self._frame_bgr: bool = True

    # --- Component ---------------------------------------------------------

    @property
    def backend_id(self) -> str:
        return ENGINE_ID

    @property
    def engine_id(self) -> str:
        return ENGINE_ID

    @property
    def device(self) -> DeviceKind:
        return DeviceKind.CUDA

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        torch = self._import_torch()
        if not torch.cuda.is_available():
            msg = (
                "the LivePortrait engine needs an NVIDIA GPU with CUDA, and none is "
                "available here. The 'warp' engine runs without one."
            )
            raise EngineError(msg)
        self._torch = torch
        self._device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        self._nets = self._load_networks()
        self._running = True
        _LOG.info(
            "LivePortrait engine ready on %s, fp16, weights from %s",
            torch.cuda.get_device_name(0),
            self._weights_root(),
        )

    def stop(self) -> None:
        for avatar_id in list(self._loaded):
            self.unload_avatar(avatar_id)
        self._stop_hair()
        self._nets = {}
        self._running = False
        if self._torch is not None:
            self._torch.cuda.empty_cache()

    # --- the avatar's hair (day 17, C1b) ------------------------------------

    def _hair_model_path(self) -> Path:
        if self._hair_model is not None:
            return self._hair_model
        if self._model_path is not None:
            return self._model_path.parent / HAIR_MODEL_RELATIVE_PATH.name
        return Path("models") / HAIR_MODEL_RELATIVE_PATH

    def _start_hair(self) -> bool:
        """The segmenter and its thread, started once on first use; False
        (logged once) when the model is not there or MediaPipe refuses."""
        if self._hair_worker is not None:
            return True
        if self._hair_unavailable:
            return False
        try:
            segmenter = HairSegmenter(self._hair_model_path())
            segmenter.start()
        except Exception as exc:
            self._hair_unavailable = True
            _LOG.warning("The avatar's hair is not available: %s", exc)
            return False
        self._hair = segmenter
        self._hair_worker = HairMatteWorker(segmenter.segment)
        _LOG.info("Hair segmenter ready: %s", segmenter.model_path)
        return True

    def _stop_hair(self) -> None:
        if self._hair_worker is not None:
            self._hair_worker.stop()
        self._hair_worker = None
        if self._hair is not None:
            self._hair.stop()
        self._hair = None

    @property
    def hair_available(self) -> bool:
        """Whether the hair segmenter's model is there to be started."""
        return not self._hair_unavailable and (
            self._hair_worker is not None or self._hair_model_path().is_file()
        )

    def hair_enabled(self, avatar_id: str) -> bool:
        """Whether the avatar's hair is being pasted now (False for an
        avatar this engine does not hold, or whose hair could not start)."""
        loaded = self._loaded.get(avatar_id)
        return bool(loaded is not None and loaded.hair_on)

    def set_hair(self, avatar_id: str, on: bool) -> bool:
        """Paste the avatar's own hair over the driver's head (``True``), or
        the face alone (``False``) -- live, no reload: the matte lives
        outside the graph. Remembered for the avatar across loads. Returns
        what is in force now for a loaded avatar (``False`` when the
        segmenter could not start), else the wish as recorded."""
        self._hair_wanted[avatar_id] = bool(on)
        loaded = self._loaded.get(avatar_id)
        if loaded is None:
            return bool(on)
        if on and not loaded.hair_on:
            return self._enable_hair(loaded)
        if not on and loaded.hair_on:
            self._disable_hair(loaded)
        return loaded.hair_on

    @staticmethod
    def _matte_hairline(hair: np.ndarray, crop_points: np.ndarray) -> float | None:
        """The avatar's own hairline from its hair matte (day 17, his first
        look): up from the brow top along the crop's rows at three columns
        across the forehead, the first row where the hair confidence passes
        0.5 on two of the three, as a fraction of the forehead height --
        the segmenter's word, not the luminance walk's. ``None`` when no
        hair is met above the brows (a bald portrait)."""
        pts = np.asarray(crop_points, dtype=np.float32)[:FACE_POINTS, :2]
        brow_top, forehead_h = _forehead_geometry(pts)
        face_w = max(float(pts[:, 0].max() - pts[:, 0].min()), 1.0)
        centre_x = float((pts[_CHEEKS[0], 0] + pts[_CHEEKS[1], 0]) / 2.0)
        plane = hair[:, :, 0] if hair.ndim == 3 else hair
        height, width = plane.shape
        columns = [
            min(max(round(centre_x + dx), 0), width - 1)
            for dx in (-0.12 * face_w, 0.0, 0.12 * face_w)
        ]
        start = min(max(round(float(brow_top[1])), 0), height - 1)
        for row in range(start, -1, -1):
            if sum(float(plane[row, c]) > 0.5 for c in columns) >= 2:
                return float((float(brow_top[1]) - row) / forehead_h)
        return None

    @staticmethod
    def _hair_reach(loaded: _Portrait, hair: np.ndarray | None = None) -> float:
        """The forehead reach with the hair on: the avatar's own hairline --
        from its hair matte (:meth:`_matte_hairline`) when given, else the
        portrait's luminance reading, else :data:`_HAIRLINE_CEILING` --
        plus :data:`_HAIR_FOREHEAD_OVERLAP`, within the reach's bounds."""
        base = None
        if hair is not None and loaded.crop_points is not None:
            base = LivePortraitEngine._matte_hairline(hair, loaded.crop_points)
        if base is None:
            base = loaded.hairline_portrait
        if base is None:
            base = _HAIRLINE_CEILING
        reach = base + _HAIR_FOREHEAD_OVERLAP
        return float(min(max(reach, _FOREHEAD_REACH_MIN), _FOREHEAD_REACH_MAX))

    @staticmethod
    def _apply_forehead_reach(loaded: _Portrait, reach: float) -> None:
        """The portrait's mask and its box rebuilt at ``reach``."""
        if loaded.crop_points is None:
            return
        loaded.forehead_reach = reach
        points = loaded.crop_points if loaded.mask_points is None else loaded.mask_points
        loaded.mask = _oval_mask(points, forehead_reach=reach)
        loaded.mask_bbox = _mask_bbox(
            loaded.mask,
            reach_below=int(np.ceil(loaded.chin_drop_max + loaded.chin_reach)),
            room_aside=_MASK_TURN_ROOM,
        )

    def _enable_hair(self, loaded: _Portrait) -> bool:
        """The hair on for a loaded portrait: the segmenter started, the
        portrait crop's own matte as the first matte (so the first frame
        already has hair, before the worker's first word), the ovals
        raised past the portrait's hairline (see :data:`_HAIR_FOREHEAD_OVERLAP`)."""
        import cv2

        if loaded.portrait_crop is None or loaded.crop_points is None or not self._start_hair():
            loaded.hair_on = False
            return False
        assert self._hair is not None
        try:
            confidence = self._hair.segment(cv2.cvtColor(loaded.portrait_crop, cv2.COLOR_BGR2RGB))
        except Exception:
            _LOG.exception("Avatar %s: the portrait's hair could not be read", loaded.descriptor.id)
            loaded.hair_on = False
            return False
        loaded.hair_skin = (
            skin_weight(loaded.crop_points, CROP_SIZE) if _hair_skin(loaded.options) else None
        )
        loaded.hair_matte = hair_alpha(head_confidence(confidence, loaded.hair_skin))
        loaded.hair_box = hair_box(loaded.hair_matte)
        loaded.hair_seq = 0
        loaded.hair_on = True
        loaded.hairline_samples = []
        loaded.hairline_settled = True
        self._apply_forehead_reach(loaded, self._hair_reach(loaded, confidence))
        _LOG.info(
            "Avatar %s: the avatar's hair on -- forehead reach %.2f, hair box %s",
            loaded.descriptor.id,
            loaded.forehead_reach,
            loaded.hair_box,
        )
        return True

    def _disable_hair(self, loaded: _Portrait) -> None:
        """The face alone again: the matte dropped, the forehead reach back
        to the option's own (a "hairline" reach is measured again)."""
        loaded.hair_on = False
        loaded.hair_matte = None
        loaded.hair_box = None
        loaded.hair_skin = None
        loaded.hair_seq = 0
        loaded.hairline_samples = []
        loaded.hairline_settled = not _forehead_reach_is_hairline(loaded.options)
        self._apply_forehead_reach(loaded, _forehead_reach(loaded.options))
        _LOG.info("Avatar %s: the avatar's hair off", loaded.descriptor.id)

    def _update_hair(self, loaded: _Portrait, face_rgb: np.ndarray) -> None:
        """This frame's render to the worker; the newest matte, if it is
        new and this avatar's, folded into the EMA and its box grown."""
        if self._submit_hair(loaded, face_rgb):
            self._fold_hair(loaded)

    def _submit_hair(self, loaded: _Portrait, face_rgb: np.ndarray) -> bool:
        """This frame's render to the worker, starting it the first time;
        ``False`` -- and the avatar's hair switched off -- when the
        segmenter cannot start."""
        worker = self._hair_worker
        if worker is None:
            if not self._start_hair():
                loaded.hair_on = False
                return False
            worker = self._hair_worker
            assert worker is not None
        worker.submit(loaded.descriptor.id, face_rgb)
        return True

    def _fold_hair(self, loaded: _Portrait) -> None:
        """The worker's newest matte, if it is new and this avatar's, folded
        into the EMA and its box grown. On the GPU frame path (2026-10-08)
        this runs before the paste and the submit after it, from a copy of
        the face the GPU fills while the CPU builds the mask -- so the
        frame never waits for the graph on the hair's account."""
        worker = self._hair_worker
        if worker is None:
            return
        got = worker.latest()
        if got is None:
            return
        tag, seq, confidence = got
        if tag != loaded.descriptor.id or seq == loaded.hair_seq:
            return
        alpha = hair_alpha(head_confidence(confidence, loaded.hair_skin))
        if loaded.hair_matte is None:
            loaded.hair_matte = alpha
        else:
            loaded.hair_matte = loaded.hair_matte * _HAIR_INERTIA + alpha * (1.0 - _HAIR_INERTIA)
        loaded.hair_seq = seq
        loaded.hair_box = union_box(loaded.hair_box, hair_box(loaded.hair_matte))

    # --- the driver's profile (day 16) --------------------------------------

    @property
    def driver_options(self) -> dict[str, Any]:
        """The driver profile's engine options as last set -- a copy."""
        return dict(self._driver_options)

    def set_driver_options(self, options: Mapping[str, Any] | None) -> None:
        """Lay the driver's profile (``configs/local.driver.json``, written by
        "Calibrate me") over every avatar's ``engine_options`` from the next
        load on. Only :data:`DRIVER_OPTION_KEYS` are accepted: the profile
        describes the person, never the picture. An avatar already loaded
        keeps what it was loaded with -- the graph bakes its options -- so
        the caller reloads it. ``None`` or ``{}`` clears the profile."""
        merged = dict(options or {})
        unknown = sorted(set(merged) - DRIVER_OPTION_KEYS)
        if unknown:
            msg = f"not driver options (see DRIVER_OPTION_KEYS): {unknown}"
            raise ValueError(msg)
        self._driver_options = merged
        _LOG.info("Driver profile set: %s", ", ".join(sorted(merged)) or "(none)")

    # --- AvatarEngine ------------------------------------------------------

    def supports(self, descriptor: AvatarDescriptor) -> bool:
        return descriptor.manifest.engine.engine_id == ENGINE_ID

    def load_avatar(self, descriptor: AvatarDescriptor) -> None:
        if descriptor.id in self._loaded:
            return
        if not self.supports(descriptor):
            msg = (
                f"avatar {descriptor.id!r} requires engine "
                f"{descriptor.manifest.engine.engine_id!r}, not {ENGINE_ID!r}"
            )
            raise AvatarLoadError(msg)
        if not self._running:
            msg = "the LivePortrait engine is not started"
            raise AvatarLoadError(msg)

        import cv2

        torch = self._torch
        portrait = self._read_portrait(descriptor)
        points = self._landmarks_of(portrait, descriptor.id)
        loaded = _Portrait(descriptor, self._driver_options)
        scale = float(loaded.options.get("crop_scale", CROP_SCALE))
        vy = float(loaded.options.get("crop_vy_ratio", CROP_VY_RATIO))

        # Day 11: the portrait is always cut by the rigid rule; crop_centre
        # "eyeline" changes only what the *driver's* crop follows, with an
        # offset learned at his own anchor (see _crop_geometry's own
        # docstring) -- validated here, applied in animate.
        _crop_centre(loaded.options)
        _mask_jaw_shrink(loaded.options)
        _chin_follow(loaded.options)
        _edge_match(loaded.options)
        _mouth_smile_opening(loaded.options)
        if _mouth_rest(loaded.options) == "driver" and loaded.lip_rest_rows is None:
            _LOG.warning(
                "Avatar %s: mouth_rest is 'driver' but no lip_rest_rows are set -- "
                "the anchor's own lip rest is used",
                descriptor.id,
            )
        lip_rest_frames = _lip_rest_tracking(loaded.options)
        loaded.leak_gain = _brow_mouth_leak(loaded.options)
        forehead_reach = _forehead_reach(loaded.options)
        to_crop, _ = crop_matrix(points, scale=scale, vy_ratio=vy)
        crop = cv2.warpAffine(portrait, to_crop, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_LINEAR)
        image = self._to_tensor(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        # The paste mask, from the portrait's own landmarks placed in this
        # same crop -- see _oval_mask's docstring for why this is done once
        # here rather than every frame from the driver's landmarks.
        crop_points = (points @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
        loaded.crop_points = crop_points
        loaded.mask = _oval_mask(crop_points, forehead_reach=forehead_reach)
        # Day 17: the crop itself is kept for the hair's own matte (see
        # _enable_hair), and set_hair's word overrides the file's.
        loaded.portrait_crop = crop
        # Day 19: the photograph's flat background, so the paste can stop
        # where the drawn face stops (see _background_keep).
        loaded.background = (
            _background_colour(portrait) if _background_guard(loaded.options) else None
        )
        if loaded.background is None:
            _LOG.info("Avatar %s: no flat background read; the paste is not guarded", descriptor.id)
        else:
            _LOG.info(
                "Avatar %s: background colour (BGR) %s; the paste stops at the drawn face",
                descriptor.id,
                loaded.background.astype(int).tolist(),
            )
        wanted = self._hair_wanted.get(descriptor.id)
        if wanted is not None:
            loaded.hair_on = wanted
        if not loaded.hairline_settled or loaded.hair_on:
            lum = _luminance(portrait)
            loaded.hairline_portrait = _hairline_fraction(lum, points)
            loaded.roots_portrait = _roots_onset(lum, points)
        # Day 13: the box reaches below the portrait's own chin by the
        # most the per-frame mask may let the jaw down (see _chin_drop).
        # The room the crop actually has below the portrait's chin -- see
        # _CHIN_DROP_MAX's own comment for why this, and not the fraction,
        # is the limit. The reach is taken first (it is a fixed choice about
        # where the join sits); the drop gets what is left.
        room = chin_room(
            points,
            scale=scale,
            vy_ratio=vy,
            forehead_reach=forehead_reach,
            chin_reach_fraction=(
                _CHIN_REACH_AUTO
                if _chin_reach_is_auto(loaded.options)
                else _mask_chin_reach(loaded.options)
            ),
        )
        crop_face_height = room.crop_face_height
        # The oval's own edge needs a little room below its hull, and how
        # much is MEASURED on this portrait's own mask rather than argued
        # from the kernels (day 14, 2026-09-10; day 13 had argued shrink +
        # feather / 2 of the face width, ~15 px, from how erode and
        # GaussianBlur behave at an array's border -- four times too much).
        # What the mask actually does: the erosion pulls the 0.5 contour in
        # by the kernel's RADIUS (3 px at his face width, not its 9-px
        # diameter) and the feather's tail (alpha 0.02) ends 1-2 px past the
        # hull. Only a hull ON the last row defeats the erosion (nothing
        # below it to erode from) and pastes the hard line; with the hull
        # 4 rows above the last one the last row reads 0.008, at 3 rows
        # 0.04, at 2 rows 0.12, at 1 row 0.3-0.6. So the hull keeps the
        # measured tail plus two rows clear of the bottom.
        # measured -- is in chin_room, shared with the importer since day 16.
        headroom = room.headroom
        loaded.chin_reach = room.chin_reach
        loaded.chin_drop_max = room.chin_drop_max
        loaded.mask_bbox = _mask_bbox(
            loaded.mask,
            reach_below=int(np.ceil(loaded.chin_drop_max + loaded.chin_reach)),
            room_aside=_MASK_TURN_ROOM,
        )
        # Day 17: the hair asked for (the option, or set_hair before the
        # load) -- the segmenter started and the portrait's own matte read
        # now, so the first frame has hair; off with a warning if it cannot.
        if loaded.hair_on and not self._enable_hair(loaded):
            _LOG.warning(
                "Avatar %s: the avatar's hair was asked for but is not available", descriptor.id
            )
        if headroom < _CHIN_DROP_MAX * crop_face_height:
            _LOG.info(
                "Avatar %s: the crop leaves %.0f usable px below the portrait's chin -- "
                "chin reach %.0f px, chin drop capped at %.0f px (see _CHIN_DROP_MAX)",
                descriptor.id,
                headroom,
                loaded.chin_reach,
                loaded.chin_drop_max,
            )

        # The portrait's own eye/lip close ratios, for the retargeting
        # networks -- see :func:`_eye_ratios`/:func:`_lip_ratio` and the
        # ``ratios`` slot's own comment. The driving targets (positions 2, 4)
        # start at "no change from the portrait": the mean of its own two eye
        # ratios, and its own lip ratio, until the first real frame arrives.
        c_s_left, c_s_right = _eye_ratios(points)
        c_s_lip = _lip_ratio(points)
        loaded.c_s_eye_mean = (c_s_left + c_s_right) / 2.0
        loaded.c_s_lip = c_s_lip

        with torch.no_grad():
            info = self._nets["motion"](image.float())
            loaded.f_s = self._nets["appearance"](image)
            loaded.x_c_s = info["kp"].float().view(1, NUM_KP, 3)
            # Day 10, Fable's review of Step 1: a fixed point on the
            # portrait's own face to rotate about instead of the keypoint
            # space's own origin -- see _rotation_pivot's own docstring.
            # Computed once, here, from x_c_s itself, which never changes
            # after load; None (the "none" default) leaves _drive's own
            # math exactly as it always was.
            pivot_name = _rotation_pivot(loaded.options)
            if pivot_name == "centroid":
                loaded.pivot = loaded.x_c_s.mean(dim=1, keepdim=True).clone()
            elif pivot_name == "eyes":
                loaded.pivot = (
                    loaded.x_c_s[:, list(EYE_KEYPOINTS), :].mean(dim=1, keepdim=True).clone()
                )
            else:
                loaded.pivot = None
            yaw_z = _yaw_pivot_z(loaded.options)
            loaded.yaw_pivot = (
                None
                if yaw_z == 0.0
                else torch.tensor([[[0.0, 0.0, yaw_z]]], device=self._device, dtype=torch.float32)
            )
            loaded.exp_s = info["exp"].float().view(1, NUM_KP, 3)
            loaded.scale_s = info["scale"].float().view(1, 1)
            loaded.t_s = info["t"].float().view(1, 3)
            loaded.R_s = self._rotation(info)
            # Eqn. 2 of the paper, for the source: s (x_c R + exp) + t, tx and ty only.
            t_xy = torch.cat([loaded.t_s[:, :2], torch.zeros_like(loaded.t_s[:, 2:])], dim=1)
            loaded.x_s = (
                loaded.scale_s[..., None] * (loaded.x_c_s @ loaded.R_s + loaded.exp_s)
                + t_xy[:, None, :]
            )
            # Until the driver's anchor is known, the anchor is the portrait
            # itself, so the graph below renders the portrait unchanged.
            loaded.exp_d0 = loaded.exp_s.clone()
            loaded.scale_d0 = loaded.scale_s.clone()
            loaded.t_d0 = loaded.t_s.clone()
            gain_x, gain_y = _translation_gain(loaded.options)
            loaded.t_gain = torch.tensor(
                [[gain_x, gain_y, 0.0]], device=self._device, dtype=torch.float32
            )
            loaded.scale_gain = torch.tensor(
                [[_scale_gain(loaded.options)]], device=self._device, dtype=torch.float32
            )
            # Day 15: the lip rest tracking's own mask and weight.
            loaded.lip_rest_alpha = 1.0 / lip_rest_frames if lip_rest_frames > 0 else 0.0
            loaded.lip_rest_mask = torch.zeros(
                1, NUM_KP, 3, device=self._device, dtype=torch.float32
            )
            for k, axis in _MOUTH_OPENING:
                loaded.lip_rest_mask[0, k, axis] = 1.0
            loaded.R_d0 = loaded.R_s.clone()

            # The smoothing state (day 7, 2026-09-05), primed from the
            # portrait's own reading -- not zero -- for the same reason
            # exp_d0/t_d0/scale_d0 are: the graph's very first blend (during
            # _capture's warm-up, run on this very portrait crop) must be a
            # no-op, not a slow ramp up from nothing. The pose is read again
            # in degrees, the same way _learn_anchor's own "pose" sum is
            # (stacked pitch/yaw/roll, shape (3, 1)) -- R_s only stores the
            # matrix built from it, not the degrees themselves.
            loaded.exp_prev = loaded.exp_s.clone()
            loaded.scale_prev = loaded.scale_s.clone()
            loaded.t_prev = loaded.t_s.clone()
            loaded.pose_prev = torch.stack(
                [self._degrees(info[k]) for k in ("pitch", "yaw", "roll")]
            ).clone()
            # The anchor's own pose in degrees, primed the same way R_d0
            # is (both from this same portrait reading, both re-primed
            # together in _learn_anchor) -- see the slot's own comment.
            loaded.pose_d0 = loaded.pose_prev.clone()

            # The day-8 region masks -- see _region_masks and the
            # EYE_KEYPOINTS table above. Validates eye_expression_gain once,
            # here, at load; nothing reads the raw option again afterwards.
            loaded.rel_mask, loaded.eye_mask = _region_masks(
                _eye_expression_gain(loaded.options),
                _eye_mask_output(loaded.options),
                torch,
                self._device,
                mouth_lateral_gain=_mouth_axis_gain(loaded.options, "mouth_lateral_gain", 0.0),
                mouth_smile_gain=_mouth_axis_gain(loaded.options, "mouth_smile_gain", 1.0),
                mouth_opening_gain=_mouth_axis_gain(loaded.options, "mouth_opening_gain", 1.0),
                expression_gain=_expression_gain(loaded.options),
            )
            loaded.crop_filters = _crop_filters(loaded.options)

            loaded.ratios = torch.tensor(
                [[c_s_left, c_s_right, loaded.c_s_eye_mean, c_s_lip, loaded.c_s_lip]],
                device=self._device,
                dtype=torch.float32,
            )
            # Day 9: the gate starts fully open (see the slot's comment) and
            # the networks' own "no change" answers are read now, while
            # ``ratios`` still holds exactly the portrait's own targets.
            loaded.gate = torch.ones(1, NUM_KP, 3, device=self._device, dtype=torch.float32)
            loaded.brow_delta = torch.zeros(1, NUM_KP, 3, device=self._device, dtype=torch.float32)
            # Day 17: the turn's cap for steep_pose = "clamp" -- the hold
            # thresholds, in the extractor's own degrees (see _drive).
            hold_pitch, hold_yaw = _hold_thresholds(loaded.options)
            loaded.pose_limit = torch.tensor(
                [[hold_pitch], [_turn_cap(loaded.options, hold_yaw)], [1.0e6]],
                device=self._device,
                dtype=torch.float32,
            )
            boost = _turn_boost(loaded.options)
            loaded.turn_boost = (
                None
                if boost <= 0.0
                else torch.tensor([[0.0], [boost], [0.0]], device=self._device, dtype=torch.float32)
            )
            # Day 23: the per-side yaw gain (see _turn_gains); None at 1/1.
            gain_left, gain_right = _turn_gains(loaded.options)
            loaded.turn_gain = (
                None
                if gain_left == 1.0 and gain_right == 1.0
                else torch.tensor([gain_left, gain_right], device=self._device, dtype=torch.float32)
            )
            loaded.mouth_leak = torch.zeros(1, NUM_KP, 3, device=self._device, dtype=torch.float32)
            # The constant mouth tilt trim lives in the same offset tensor
            # (its own component is never written per frame) -- see
            # _mouth_tilt_trim.
            loaded.brow_delta[0, _TILT_TRIM_KEYPOINT, 0] = _mouth_tilt_trim(loaded.options)
            # Validated once, here, like eye_expression_gain; read every
            # frame afterwards without a second look.
            _eye_headroom(loaded.options)
            _eye_scale(loaded.options)
            _brow_gain(loaded.options)
            _brow_follow(loaded.options)
            _size_smoothing(loaded.options)
            loaded.eye_rest, loaded.lip_rest = self._retarget_rest(loaded)

            loaded.static_in = image.clone()
            self._capture(loaded)

        torch.cuda.synchronize()
        loaded.vram_mb = int(torch.cuda.memory_allocated() // (1024 * 1024))
        self._loaded[descriptor.id] = loaded
        _LOG.info(
            "Loaded avatar %s into LivePortrait: %dx%d portrait, crop %.2f/%+.3f, "
            "anchor over %d frames, %d MB on the GPU",
            descriptor.id,
            portrait.shape[1],
            portrait.shape[0],
            scale,
            vy,
            loaded.anchor_wanted,
            loaded.vram_mb,
        )

    def unload_avatar(self, avatar_id: str) -> None:
        loaded = self._loaded.pop(avatar_id, None)
        if loaded is None:
            return
        loaded.graph = None
        loaded.static_in = loaded.static_out = None
        if self._torch is not None:
            self._torch.cuda.empty_cache()

    @property
    def loaded_avatars(self) -> Sequence[str]:
        return tuple(self._loaded)

    def estimated_vram_mb(self, avatar_id: str) -> int | None:
        loaded = self._loaded.get(avatar_id)
        return None if loaded is None else loaded.vram_mb

    def reset_anchor(self, avatar_id: str) -> None:
        """Learn the driver's resting face again from the next frames.

        For a person who started the call mid-word or turned away: the avatar's
        rest is measured against the first frames, and this is the one knob
        that says "measure it again".

        This does not touch ``exp_d0``/``R_d0`` (the old anchor keeps driving
        the avatar, unchanged, until the new one is learned -- the first
        accepted frame after this call hard-overwrites them, same as always)
        and, for the same reason, does not touch the day-7 smoothing state
        either (``exp_prev``/``t_prev``/``scale_prev``/``pose_prev``): it
        keeps blending toward fresh readings against the *old* anchor exactly
        as it did the instant before this call, with no jump, until
        :meth:`_learn_anchor` re-primes it to the new anchor's own running
        mean on the next accepted frame -- the same moment ``exp_d0`` itself
        is next overwritten.

        This *does* clear the day-8 extreme-pose hold (:func:`_should_hold`)
        -- unlike the anchor and the smoothing state, a held face is not a
        running average of anything; it is a statement about *this* frame's
        own pose, safe to forget and re-derive from the very next frame.
        """
        loaded = self._loaded.get(avatar_id)
        if loaded is not None:
            loaded.anchor_seen = 0
            loaded.anchor_sum = {}
            loaded.anchor_rejected = 0
            loaded.rest_ear = None
            loaded.rest_mar = None
            loaded.rest_mouth_width = None
            loaded.rest_brow = None
            loaded.rest_chin = None
            loaded.rest_gaze = None
            loaded.gaze_prev = 0.0
            loaded.ear_recent.clear()
            loaded.gate_mouth_prev = 0.0
            loaded.gate_open_prev = 0.0
            loaded.gate_brow_prev = 0.0
            loaded.mouth_hold = 0
            loaded.open_hold = 0
            loaded.size_samples = []
            loaded.face_height_rest = None
            loaded.z_rest = None
            loaded.z_recent.clear()
            loaded.z_frontal = None
            loaded.centre_samples = []
            loaded.pose_samples = []
            loaded.centre_offset = None
            loaded.rest_pitch = None
            loaded.rest_yaw = None
            loaded.holding = False
            loaded.lip_rest_run = 0
            loaded.lip_rest_steps = 0
            loaded.hairline_samples = []
            loaded.width_samples = []
            loaded.height_samples = []
            loaded.width_settled = False
            loaded.driver_reach = None
            loaded.forehead_stretch = 1.0
            loaded.stretch_row = None
            # Day 17: with the hair on the reach is the portrait's own
            # hairline plus the overlap, settled at load -- nothing to
            # measure again (see _enable_hair).
            loaded.hairline_settled = loaded.hair_on or not _forehead_reach_is_hairline(
                loaded.options
            )

    def rest_status(self, avatar_id: str) -> str | None:
        """A line for the UI while the anchor is still being learned, else
        ``None`` -- both when it is already frozen and when the avatar is
        unknown to this engine."""
        loaded = self._loaded.get(avatar_id)
        if loaded is None or loaded.anchor_seen >= loaded.anchor_wanted:
            return None
        return (
            "Sit still and look at the camera to centre the avatar "
            f"({loaded.anchor_seen}/{loaded.anchor_wanted})"
        )

    def anchor_state(self, avatar_id: str) -> dict[str, np.ndarray] | None:
        """The frozen anchor as plain arrays -- what a capture writes beside
        its tracking so a replay can be the session itself (see
        :data:`_ANCHOR_TENSORS`, :meth:`seed_anchor`). ``None`` while the
        anchor is still being learned, and for an avatar this engine does
        not hold. The arrays are copies: nothing here aliases the graph's
        own tensors."""
        loaded = self._loaded.get(avatar_id)
        if loaded is None or loaded.anchor_seen < loaded.anchor_wanted or loaded.exp_d0 is None:
            return None
        state: dict[str, np.ndarray] = {}
        for name in _ANCHOR_TENSORS:
            state[name] = getattr(loaded, name).detach().float().cpu().numpy().copy()
        for name in _ANCHOR_SCALARS:
            value = getattr(loaded, name)
            state[name] = np.array(np.nan if value is None else float(value), dtype=np.float64)
        gaze = loaded.rest_gaze
        state["rest_gaze"] = np.array(
            (np.nan, np.nan) if gaze is None else (float(gaze[0]), float(gaze[1])),
            dtype=np.float64,
        )
        return state

    def seed_anchor(self, avatar_id: str, state: dict[str, Any]) -> None:
        """Make a captured anchor (:meth:`anchor_state`'s arrays) this avatar's
        own, frozen: the tensors ``_drive`` reads are overwritten in place
        (the graph keeps reading the same storage), the rest values are set,
        the smoothing state is primed to the anchor exactly as the freeze
        does, and the per-frame state (gates, holds, the blink window) is
        cleared as :meth:`reset_anchor` clears it. From the next frame the
        engine runs as the session that wrote the capture ran."""
        loaded = self._loaded.get(avatar_id)
        if loaded is None:
            msg = f"avatar {avatar_id!r} is not loaded"
            raise AvatarLoadError(msg)
        torch = self._torch
        for name in _ANCHOR_TENSORS:
            if name not in state:
                msg = f"the anchor state has no {name!r}"
                raise ValueError(msg)
            target = getattr(loaded, name)
            source = torch.as_tensor(
                np.asarray(state[name], dtype=np.float32), device=target.device
            ).to(target.dtype)
            if tuple(source.shape) != tuple(target.shape):
                msg = f"anchor {name!r} has shape {tuple(source.shape)}, not {tuple(target.shape)}"
                raise ValueError(msg)
            target.copy_(source)
        # Day 15: a captured anchor replayed under "portrait" (day 16: or
        # "driver") gets the same lip rest the session would learn now -- so
        # a capture taken before the option, or with it off, can be measured
        # with it on.
        _apply_mouth_rest(loaded)
        for name in _ANCHOR_SCALARS:
            value = float(np.asarray(state.get(name, np.nan), dtype=np.float64))
            setattr(loaded, name, None if math.isnan(value) else value)
        gaze = np.asarray(state.get("rest_gaze", (np.nan, np.nan)), dtype=np.float64).reshape(-1)
        if gaze.size < 2 or math.isnan(float(gaze[0])):
            loaded.rest_gaze = None
        else:
            loaded.rest_gaze = (float(gaze[0]), float(gaze[1]))
        # Day 22: the session's own hairline and width ratio stand in for
        # the profile's, so the forehead and the width settle at once, as
        # they stood -- unless the options say otherwise.
        if loaded.hairline_driver is None and loaded.settle_hairline is not None:
            loaded.hairline_driver = loaded.settle_hairline
        if loaded.width_driver is None and loaded.settle_width is not None:
            loaded.width_driver = loaded.settle_width
        if loaded.height_driver is None and loaded.settle_height is not None:
            loaded.height_driver = loaded.settle_height
        # The freeze's own priming of the day-7 smoothing state, and the
        # per-frame state a fresh anchor starts without (see reset_anchor).
        loaded.exp_prev.copy_(loaded.exp_d0)
        loaded.t_prev.copy_(loaded.t_d0)
        loaded.scale_prev.copy_(loaded.scale_d0)
        loaded.pose_prev.copy_(loaded.pose_d0)
        loaded.anchor_seen = loaded.anchor_wanted
        loaded.anchor_sum = {}
        loaded.anchor_rejected = 0
        loaded.gaze_prev = 0.0
        loaded.ear_recent.clear()
        loaded.gate_mouth_prev = 0.0
        loaded.gate_open_prev = 0.0
        loaded.gate_brow_prev = 0.0
        loaded.mouth_hold = 0
        loaded.open_hold = 0
        loaded.size_samples = []
        loaded.centre_samples = []
        loaded.pose_samples = []
        loaded.holding = False
        loaded.lip_rest_run = 0
        loaded.lip_rest_steps = 0
        _LOG.info(
            "Avatar %s: anchor seeded from a capture -- rest EAR %s, rest MAR %s, "
            "face height %s px at distance %s",
            avatar_id,
            "?" if loaded.rest_ear is None else f"{loaded.rest_ear:.3f}",
            "?" if loaded.rest_mar is None else f"{loaded.rest_mar:.3f}",
            "?" if loaded.face_height_rest is None else f"{loaded.face_height_rest:.0f}",
            "?" if loaded.z_rest is None else f"{loaded.z_rest:.2f}",
        )

    def frame_readings(self, avatar_id: str) -> dict[str, float] | None:
        """What the engine made of the last frame, as plain floats -- the
        calibration's instrument (day 16, ``sha256.calibration``): whether
        the anchor is frozen and the face held, the three gates as applied,
        the rest values the anchor learned (NaN while unknown), the
        extractor's own reading of every mouth keypoint's row
        (``exp_<kp>_<axis>``, what a brow raise leaks into) and the anchor's
        lip-opening rows (``anchor_<kp>_<axis>``). Two small tensors come back
        from the GPU; call it after ``animate`` on the same thread, during a
        calibration, not on every frame of a call. ``None`` for an avatar
        this engine does not hold."""
        loaded = self._loaded.get(avatar_id)
        if loaded is None:
            return None
        readings: dict[str, float] = {
            "anchor_frozen": float(loaded.anchor_seen >= loaded.anchor_wanted),
            "holding": float(loaded.holding),
            "gate_mouth": float(loaded.gate_mouth_prev),
            "gate_open": float(loaded.gate_open_prev),
            "gate_brow": float(loaded.gate_brow_prev),
            "chin_drop": float(loaded.chin_drop),
            "hair": float(loaded.hair_on),
            # Day 17: his hairline off the last frontal frame read, for the
            # profile's driver_hairline (NaN: none read, or no hair found).
            "hairline": math.nan if loaded.hairline_last is None else float(loaded.hairline_last),
        }
        for name in (
            "rest_ear",
            "rest_mar",
            "rest_mouth_width",
            "rest_brow",
            "rest_pitch",
            "rest_yaw",
            "face_height_rest",
            "z_rest",
        ):
            value = getattr(loaded, name)
            readings[name] = math.nan if value is None else float(value)
        gaze = loaded.rest_gaze
        readings["rest_gaze_x"] = math.nan if gaze is None else float(gaze[0])
        readings["rest_gaze_y"] = math.nan if gaze is None else float(gaze[1])
        # Day 21: the skin's tone in force (the cast and the contrast, the
        # mean over the channels); the forehead's plan once settled.
        tone = _as_numpy_tone(loaded.tone)
        readings["tone_cast"] = float(tone[0].mean())
        readings["tone_contrast"] = float(tone[1].mean())
        readings["forehead_stretch"] = float(loaded.forehead_stretch)
        readings["face_width"] = math.nan if loaded.width_last is None else loaded.width_last
        readings["face_width_scale"] = float(loaded.width_scale)
        readings["face_height"] = math.nan if loaded.height_last is None else loaded.height_last
        readings["face_height_scale"] = float(loaded.height_scale)
        readings["driver_reach"] = (
            math.nan if loaded.driver_reach is None else float(loaded.driver_reach)
        )
        if loaded.exp_prev is not None and loaded.exp_d0 is not None:
            exp = loaded.exp_prev.detach().float().cpu().numpy().reshape(NUM_KP, 3)
            anchor = loaded.exp_d0.detach().float().cpu().numpy().reshape(NUM_KP, 3)
            for k in MOUTH_KEYPOINTS:
                for axis in range(3):
                    readings[f"exp_{k}_{axis}"] = float(exp[k, axis])
            for k, axis in _MOUTH_OPENING:
                readings[f"anchor_{k}_{axis}"] = float(anchor[k, axis])
        return readings

    @staticmethod
    def _settle_forehead(loaded: _Portrait, driver_crop_points: np.ndarray | None = None) -> None:
        """With ``mask_forehead_reach = "hairline"``, once the anchor is frozen
        and enough frontal frames were read: the forehead's plan from the
        two hairlines (:func:`_forehead_plan`, day 21; before it
        :func:`_hairline_reach`) -- the portrait's mask and its box rebuilt
        at its reach, the driver oval's own reach and the render's stretch
        kept on ``loaded`` -- and nothing measured again until a re-centre.
        ``driver_crop_points`` are the driver's points in this frame's crop,
        for their forehead's height against the portrait's; without them the
        two are taken as equal."""
        if loaded.hairline_settled or loaded.crop_points is None:
            return
        # Day 17: the profile's word (Calibrate me) settles at once; else
        # the live frames, once enough of them.
        if loaded.hairline_driver is not None:
            driver: float | None = loaded.hairline_driver
            source = "the profile"
        else:
            if len(loaded.hairline_samples) < _HAIRLINE_MIN_SAMPLES:
                return
            found = [s for s in loaded.hairline_samples if s is not None]
            driver = (
                float(np.median(found)) if len(found) * 2 >= len(loaded.hairline_samples) else None
            )
            source = f"{len(loaded.hairline_samples)} live frames"
        # Day 22: the width fit settles with the forehead, once -- the
        # profile's ratio, else the median of the still frontal readings;
        # the portrait's points squeezed for the masks, the render in the
        # frame. A later plan of the forehead (the hairline followed) keeps
        # the width it has. Day 23: the height fit beside it (see
        # _brow_chin): the portrait's points lifted about its chin row so
        # that its brow top lands on the person's -- and the forehead's
        # plan below reads the LIFTED geometry (its brow row, its
        # forehead's height), since that is where the render's rows are.
        if not loaded.width_settled:
            if loaded.width_driver is not None:
                ratio: float | None = loaded.width_driver
                width_source = "the profile"
            elif len(loaded.width_samples) >= _WIDTH_MIN_SAMPLES:
                ratio = float(np.median(loaded.width_samples))
                width_source = f"{len(loaded.width_samples)} still frames"
            else:
                ratio = None
                width_source = f"only {len(loaded.width_samples)} still frames -- not fitted"
            if loaded.height_driver is not None:
                tall: float | None = loaded.height_driver
                height_source = "the profile"
            elif len(loaded.height_samples) >= _WIDTH_MIN_SAMPLES:
                tall = float(np.median(loaded.height_samples))
                height_source = f"{len(loaded.height_samples)} still frames"
            else:
                tall = None
                height_source = f"only {len(loaded.height_samples)} still frames -- not fitted"
            loaded.width_scale = _width_scale(ratio, _face_width_fit(loaded.options))
            loaded.width_centre = _rim_centre_x(loaded.crop_points)
            loaded.height_scale = _height_scale(tall, _face_height_fit(loaded.options))
            loaded.height_centre = _chin_row(loaded.crop_points)
            points = loaded.crop_points
            if loaded.width_scale != 1.0:
                points = _squeeze_points(points, loaded.width_centre, loaded.width_scale)
            if loaded.height_scale != 1.0:
                points = _lift_points(points, loaded.height_centre, loaded.height_scale)
            loaded.mask_points = None if points is loaded.crop_points else points
            loaded.width_samples = []
            loaded.height_samples = []
            loaded.width_settled = True
            loaded.settle_width = ratio
            loaded.settle_height = tall
            _LOG.info(
                "Avatar %s: face width -- theirs %s of the portrait's from %s, the render "
                "and the oval scaled x%.3f about column %.0f (face_width_fit %.2f); brow to "
                "chin -- theirs %s of the portrait's from %s, scaled x%.3f about the chin "
                "row %.0f (face_height_fit %.2f)",
                loaded.descriptor.id,
                "?" if ratio is None else f"{ratio:.3f}",
                width_source,
                loaded.width_scale,
                loaded.width_centre,
                _face_width_fit(loaded.options),
                "?" if tall is None else f"{tall:.3f}",
                height_source,
                loaded.height_scale,
                loaded.height_centre,
                _face_height_fit(loaded.options),
            )
        # The plan's portrait side from the lifted points alone (not the
        # squeezed ones: the width changes no row, and with the height fit
        # off the plan's inputs are what they were, to the bit).
        portrait_points = (
            loaded.crop_points
            if loaded.height_scale == 1.0
            else _lift_points(loaded.crop_points, loaded.height_centre, loaded.height_scale)
        )
        brow_top, portrait_forehead = _forehead_geometry(portrait_points)
        driver_forehead = (
            portrait_forehead
            if driver_crop_points is None
            else _forehead_geometry(driver_crop_points)[1]
        )
        # Day 23: a portrait with no hairline read (a shaved head, a very
        # high forehead) has no hair roots either -- what the roots reader
        # met there (his first shaved avatar: "roots" at 0.90 of the
        # forehead, the scalp's own shading) must not cap the clean skin.
        roots = None if loaded.hairline_portrait is None else loaded.roots_portrait
        portrait_reach, driver_reach, stretch = _forehead_plan(
            loaded.hairline_portrait,
            driver,
            portrait_forehead,
            driver_forehead,
            clearance=loaded.hairline_clearance,
            margin=_hair_root_margin(loaded.options),
            stretch_max=_forehead_stretch_max(loaded.options),
            portrait_roots=roots,
        )
        loaded.settle_hairline = driver
        LivePortraitEngine._apply_forehead_reach(loaded, portrait_reach)
        loaded.driver_reach = driver_reach
        loaded.forehead_stretch = stretch
        loaded.stretch_row = (
            float(brow_top[1]) - _STRETCH_ANCHOR * portrait_forehead if stretch > 1.0 else None
        )
        loaded.hairline_settled = True
        loaded.hairline_samples = []
        _LOG.info(
            "Avatar %s: forehead settled -- the portrait's oval to %.2f of its forehead, "
            "the driver's to %.2f of theirs, the render stretched x%.2f above the brows "
            "(portrait hairline %s, its roots %s, theirs %s from %s, foreheads %.0f / %.0f "
            "crop px, clearance %.2f, roots' margin %.2f)",
            loaded.descriptor.id,
            loaded.forehead_reach,
            driver_reach,
            stretch,
            "?" if loaded.hairline_portrait is None else f"{loaded.hairline_portrait:.2f}",
            "?" if loaded.roots_portrait is None else f"{loaded.roots_portrait:.2f}",
            "?" if driver is None else f"{driver:.2f}",
            source,
            portrait_forehead,
            driver_forehead,
            loaded.hairline_clearance,
            _hair_root_margin(loaded.options),
        )

    @staticmethod
    def _driver_face_height(loaded: _Portrait, head_pose: HeadPose | None) -> float | None:
        """How tall the driver's face is this frame, for the crop: his resting
        height scaled by how much nearer or farther the head pose says he is
        (its translation z, a rigid fit of a whole face model, which neither
        an expression nor a turn of the head can move -- measured on the
        owner's clip within -4.7..+0.6% over every segment, against -13.5%
        for the forehead-to-nose height under pitch and -20% for the cheek-
        to-cheek width under yaw). ``None`` before the anchor has frozen, or
        without a usable pose, and :func:`crop_matrix` measures the frame
        itself instead. Day 20: past ``depth_hold_yaw`` of yaw the depth is
        the last frontal frame's, not this frame's (see :func:`_held_depth`)."""
        if loaded.face_height_rest is None or loaded.z_rest is None or head_pose is None:
            return None
        z = abs(float(head_pose.translation[2]))
        if z <= 1e-6:
            return None
        z = _held_depth(loaded, head_pose, z)
        return loaded.face_height_rest * loaded.z_rest / z

    def animate(
        self, avatar_id: str, tracking: FaceTrackingResult, frame: VideoFrame
    ) -> RenderedFrame:
        loaded = self._loaded.get(avatar_id)
        if loaded is None:
            msg = f"avatar {avatar_id!r} is not loaded"
            raise AvatarLoadError(msg)
        if frame.pixel_format not in (PixelFormat.BGR24, PixelFormat.RGB24):
            msg = f"the LivePortrait engine cannot composite over {frame.pixel_format.value} frames"
            raise AvatarLoadError(msg)

        import cv2

        scene = frame.data
        height, width = scene.shape[:2]
        bgr = frame.pixel_format is PixelFormat.BGR24
        self._frame_bgr = bgr
        # 2026-10-08: on CUDA the face stays on the GPU from the graph to the
        # paste (see _paste_gpu); on the CPU -- the tests' fixtures -- the
        # frame path is the one it was, to the byte.
        on_gpu = bool(
            self.paste_on_gpu and self._device is not None and self._device.type == "cuda"
        )

        target = None
        gaze = None
        if tracking.face_detected and tracking.landmarks is not None:
            normalised = tracking.landmarks.points
            if normalised.shape[0] >= FACE_POINTS:
                target = normalised[:FACE_POINTS, :2] * (width, height)
                # Day 11: his irises, from the same points past the 468
                # the face itself uses -- see the GAZE section.
                gaze = _gaze_offset(normalised[:, :2] * (width, height))

        hair_pending = None
        if target is None:
            # The face is lost: the crop filters forget where it was, so the
            # first frame it is found again is taken as it comes, not
            # blended toward a stale position.
            for crop_filter in loaded.crop_filters or ():
                if crop_filter is not None:
                    crop_filter.reset()
            loaded.last_timestamp = None
            if loaded.last is None:
                return self._passthrough(scene, frame, tracking, avatar_id)
            face, to_frame, mask, extra = loaded.last
        else:
            started = now()
            scale = float(loaded.options.get("crop_scale", CROP_SCALE))
            vy = float(loaded.options.get("crop_vy_ratio", CROP_VY_RATIO))
            cx, cy, angle, size = _crop_geometry(
                target,
                scale=scale,
                vy_ratio=vy,
                face_height=self._driver_face_height(loaded, tracking.head_pose),
                centre_offset=loaded.centre_offset,
                centre_weight=self._centre_weight(loaded, tracking.head_pose),
            )
            cx, cy, size = self._smooth_crop(loaded, frame, cx, cy, size)
            if loaded.crop_shift_x != 0.0:
                # Day 16 (C2): a constant sideways shift, in the crop's own
                # pixels, along the face's horizontal -- see _crop_shift_x.
                shift = loaded.crop_shift_x * size / CROP_SIZE
                cx += shift * math.cos(angle)
                cy += shift * math.sin(angle)
            to_crop, to_frame = _similarity(cx, cy, angle, size)
            crop = cv2.warpAffine(
                scene,
                to_crop,
                (CROP_SIZE, CROP_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=_crop_border_mode(loaded.options, cv2),
            )
            small = cv2.resize(crop, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)
            if bgr:
                small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            cropped = now()

            # Day 8, 2026-09-06: at a pose too steep for a frontal portrait
            # to be redrawn convincingly (see _should_hold), skip the
            # network entirely and reuse the last rendered face -- its
            # expression freezes, but the crop above and the mask below are
            # still computed fresh every frame, so the held face still
            # follows him and is pasted at his actual, current position.
            # If there is no rendered face yet, render normally instead.
            # His points in the crop -- the same affine that cut the crop
            # places them inside it. Read before the render since day 13:
            # the anchor learns his resting nose-to-chin reach from them
            # (see _chin_drop); they shape the paste mask below as before.
            crop_points = (target @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
            loaded.features = target[[k for group in _EDGE_FEATURES for k in group]].astype(
                np.float32
            )
            chin_drop = 0.0
            # Day 16: the anchor gate's word on stillness, read every frame
            # (held or not) so the window is always the last half second;
            # a move while the anchor is still being learned starts the
            # learning over (see _restart_anchor).
            still = _pose_still(loaded.pose_recent, tracking.head_pose)
            if not still:
                self._restart_anchor(loaded)
            # Day 22: his rim's width over the portrait's, this frame (the
            # calibration's instrument, frame_readings["face_width"]), and
            # kept for the settle on still frontal frames (see _rim_width).
            if loaded.crop_points is not None and _width_frame_ok(tracking.head_pose):
                loaded.width_last = _rim_width(crop_points) / _rim_width(loaded.crop_points)
                loaded.height_last = _brow_chin(crop_points) / _brow_chin(loaded.crop_points)
                if still and not loaded.hairline_settled:
                    loaded.width_samples.append(loaded.width_last)
                    loaded.height_samples.append(loaded.height_last)
            if _should_hold(loaded, tracking.head_pose) and loaded.last is not None:
                face = loaded.last[0]
                # Whatever his mouth does while held is not seen; the first
                # frame after must not read the jump as speech (_gates).
                loaded.mar_prev = loaded.width_prev = None
                rendered = now()
                haired = None
            else:
                c_d_eye = sum(_eye_ratios(target)) / 2.0
                c_d_lip = _lip_ratio(target)
                # Day 12: the gates and the rest read the opening over the
                # jaw width; c_d_lip stays for the anchor and the lip net.
                c_d_open = _mouth_open_ratio(target)
                c_d_width = _mouth_width_ratio(target)
                c_d_brow = _brow_ratio(target)
                c_d_chin = _nose_chin(crop_points)
                anchor_ok = _anchor_frame_ok(tracking.head_pose, c_d_eye, c_d_lip, still=still)
                # Day 12: with gaze_down_relief on, a slow glance down keeps
                # the avatar's eyes at rest (his call, see _relieved_ear);
                # the anchor and the gaze hold below still read c_d_eye.
                target_ear = c_d_eye
                if _gaze_down_relief(loaded.options):
                    target_ear = _relieved_ear(
                        c_d_eye, loaded.rest_ear, gaze, loaded.rest_gaze, loaded.ear_recent
                    )
                loaded.ear_recent.append(c_d_eye)
                eye_target = _eye_target(
                    target_ear,
                    loaded.rest_ear,
                    loaded.c_s_eye_mean,
                    _eye_headroom(loaded.options),
                    _eye_scale(loaded.options),
                )
                lip_target = _lip_target(c_d_lip, loaded.c_s_lip)
                size_sample = None
                centre_sample = None
                pose_sample = None
                if anchor_ok and tracking.head_pose is not None:
                    z = abs(float(tracking.head_pose.translation[2]))
                    if z > 1e-6:
                        size_sample = (_face_height(target), z)
                    pose_sample = (
                        float(tracking.head_pose.pitch),
                        float(tracking.head_pose.yaw),
                    )
                    if _crop_centre(loaded.options) == "eyeline":
                        centre_sample = (
                            _centre_offset(target, scale=scale, vy_ratio=vy),
                            float(tracking.head_pose.pitch),
                        )
                gates = self._gates(loaded, c_d_open, c_d_width, c_d_brow, tracking.head_pose)
                brow_lift = _brow_follow(loaded.options) * _brow_lift(
                    c_d_brow, loaded.rest_brow, tracking.head_pose, loaded.thresholds.brow_deadband
                )
                # Day 17: under the clamp, past the yaw cap the mouth, the
                # brows and the gaze fade to rest (see _steep_fade).
                steep_fade = _steep_fade(loaded, tracking.head_pose)
                if steep_fade < 1.0:
                    gates = (gates[0] * steep_fade, gates[1] * steep_fade, gates[2] * steep_fade)
                    brow_lift *= steep_fade
                gaze_fade = steep_fade * _gaze_yaw_fade(loaded, tracking.head_pose)
                face_dev = self._render(
                    loaded,
                    small,
                    c_d_eye,
                    c_d_open,
                    eye_target,
                    lip_target,
                    anchor_ok,
                    size_sample=size_sample,
                    rest_sample=(c_d_width, c_d_brow),
                    gates=gates,
                    brow_lift=brow_lift,
                    brow_deviation=(
                        c_d_brow - loaded.rest_brow if loaded.rest_brow is not None else 0.0
                    ),
                    gaze=gaze,
                    centre_sample=centre_sample,
                    chin_sample=c_d_chin,
                    pose_sample=pose_sample,
                    steep_fade=gaze_fade,
                    on_device=True,
                )
                rendered = now()
                if on_gpu:
                    # 2026-10-08: the face stays on the GPU -- the frame's
                    # channel order, 0..255 as float -- for the GPU paste and
                    # the hold; nothing waits for the graph here (a flip,
                    # not an index list: a list would be copied up to the
                    # device, and a copy from unpinned memory waits for the
                    # stream, graph and all). The CPU sees the picture only
                    # when the hair worker needs it.
                    face = face_dev.permute(2, 0, 1)
                    face = (face.flip(0) if bgr else face).float()
                    if loaded.width_scale != 1.0 or loaded.height_scale != 1.0:
                        face = self._squeeze_face_gpu(face, loaded)
                    if loaded.hair_on:
                        # The newest matte folded in now, for this frame's
                        # paste; this frame's face to the worker after the
                        # paste, from a pinned copy the GPU fills meanwhile
                        # (see _fold_hair).
                        if self._hair_worker is None and not self._start_hair():
                            loaded.hair_on = False
                        else:
                            self._fold_hair(loaded)
                            hair_pending = self._pinned_picture("hair", face_dev)
                else:
                    face_rgb = face_dev.cpu().numpy()
                    face = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR) if bgr else face_rgb
                    if loaded.width_scale != 1.0 or loaded.height_scale != 1.0:
                        face = _squeeze_face(
                            face,
                            loaded.width_centre,
                            loaded.width_scale,
                            loaded.height_centre,
                            loaded.height_scale,
                        )
                    # Day 17 (C1b): this render to the hair segmenter's
                    # thread, and the newest matte folded in -- see
                    # _update_hair.
                    if loaded.hair_on:
                        self._update_hair(loaded, face_rgb)
                haired = now()
                # Day 13: how far his chin has dropped from rest, in crop
                # pixels -- what the portrait's oval is let down by below
                # the mouth, so the join follows the avatar's opened jaw
                # rather than the closed-mouth chin the mask was built on.
                # Read only off a rendered frame: a held face's chin has not
                # moved, so its mask must not reach past it.
                chin_drop = _chin_follow(loaded.options) * _chin_drop(
                    c_d_chin, loaded.rest_chin, loaded.chin_drop_max
                )
            # Day 16 (C1): his hairline, on frontal frames, until the anchor
            # settles the forehead reach from it and the portrait's. Day 17:
            # read on every Nth frontal frame after that too, for the
            # calibration (frame_readings["hairline"], the profile's
            # driver_hairline). Read here, after the render is launched,
            # since 2026-10-08: the reading feeds nothing the graph takes,
            # only _settle_forehead below and the calibration, and it costs
            # the whole frame's luminance -- CPU time the GPU now draws
            # through instead of waiting for.
            if _hairline_frame_ok(tracking.head_pose) and (
                not loaded.hairline_settled
                or tracking.metadata.frame_id % _HAIRLINE_READ_EVERY == 0
            ):
                reading = _hairline_fraction(_luminance(scene, bgr=bgr), target)
                loaded.hairline_last = reading
                if not loaded.hairline_settled:
                    loaded.hairline_samples.append(reading)
                elif loaded.hairline_driver is None:
                    # Day 22: the forehead FOLLOWS their hairline -- his hand
                    # pulling the hair back raised it a third of a forehead
                    # and the settled join stayed where it was, a band of
                    # his own skin above it. The last readings are kept,
                    # and when their median leaves the settled hairline by
                    # _HAIRLINE_FOLLOW the plan is made again from them.
                    loaded.hairline_samples.append(reading)
                    del loaded.hairline_samples[:-_HAIRLINE_MIN_SAMPLES]
                    if _hairline_drifted(loaded):
                        loaded.hairline_settled = False
            loaded.chin_drop = chin_drop
            if not loaded.hairline_settled and loaded.anchor_seen >= loaded.anchor_wanted:
                self._settle_forehead(loaded, crop_points)

            # The driver's own oval, in the same crop the portrait's was built
            # in -- and, under an open mouth, the portrait's own let down to
            # meet it; see _frame_mask.
            assert loaded.mask is not None
            # Day 22: at a strong turn (press 181432, yaw -40) or a look
            # down the paste reached 12 frame px under his chin -- the reach
            # under the chin (mask_chin_reach, his 0.04) drawing the avatar's
            # light chin underside over his dark one, a pale blob. The reach
            # fades out with the turn or the tilt (the guard's own gates);
            # nothing frontal. (Pulling his oval's jaw in as well, by 0.06 of
            # the face width, was tried on the press and took the join 26 px
            # up: his stubbled chin showed under the avatar's, the day-12
            # ledge. The reach alone is the blob's own height.)
            tilt = max(
                _tilt_weight(loaded, tracking.head_pose),
                _background_weight(loaded, tracking.head_pose),
            )
            mask, driver_oval, portrait_oval = _frame_masks(
                loaded.mask,
                crop_points,
                jaw_shrink=_mask_jaw_shrink(loaded.options),
                portrait_crop_points=(
                    loaded.crop_points if loaded.mask_points is None else loaded.mask_points
                ),
                chin_drop=chin_drop,
                forehead_reach=loaded.forehead_reach,
                # Day 22: the reach under the chin fades out with the turn
                # or the tilt (see above).
                chin_reach=loaded.chin_reach * (1.0 - tilt),
                # Day 17, his first look: with the avatar's hair on, his own
                # oval must not stop the paste at HIS hairline -- a band of
                # his hair showed between the avatar's forehead and its hair
                # -- so his oval reaches as high as the portrait's may.
                # Day 21: else the driver oval's own reach from the forehead's
                # plan (None before the settle: the portrait's).
                driver_forehead_reach=_FOREHEAD_REACH_MAX
                if loaded.hair_on
                else loaded.driver_reach,
            )
            masked = now()

            # Day 19: how much the paste stops at the drawn face this frame
            # -- with the turn, nothing on a frontal face (see _paste).
            loaded.guard_weight = _background_weight(loaded, tracking.head_pose)
            # Day 20: past the portrait's own oval at a turn, the driver's
            # -- pasted where the render is the drawn face (see _turn_room).
            room = _room_weight(loaded, tracking.head_pose)
            extra = _turn_room(portrait_oval, driver_oval, room)
            if extra is not None:
                # Day 22 evening, his "the avatar's hair above the forehead"
                # and the smear on the far side of a turn: since day 21 his
                # oval reaches HIS hairline, above the portrait's clean skin,
                # and at a turn the room pasted that strip -- the render's
                # hair, which the guard (background only) keeps. The room
                # is for the far brow and eye (day 20): nothing above the
                # brow line less _ROOM_BROW_MARGIN of the forehead.
                brow_top, forehead = _forehead_geometry(crop_points)
                ceiling = int(max(0, round(float(brow_top[1]) - _ROOM_BROW_MARGIN * forehead)))
                extra[:ceiling] = 0.0
            # Day 22 evening: the render's profile side fitted to his
            # outline at a turn (see _outline_maps).
            face = self._fit_outline(
                loaded, face, mask, extra, crop_points, tracking.head_pose, room, on_gpu
            )
            loaded.last = (face, to_frame, mask, extra)
            self.last_timings = {
                "crop_ms": (cropped - started) * 1000.0,
                "model_ms": (rendered - cropped) * 1000.0,
                "hair_ms": 0.0 if haired is None else (haired - rendered) * 1000.0,
                "mask_ms": (masked - (rendered if haired is None else haired)) * 1000.0,
            }

        pasted = now()
        self._paste_wait_ms = 0.0
        self._upload_ms = 0.0
        self._paste_marks = (0.0, 0.0, 0.0)
        output = self._paste(scene, face, to_frame, loaded, mask, extra=extra)
        self.last_timings["paste_ms"] = (now() - pasted) * 1000.0
        if not loaded.tone_logged and tracking.metadata.frame_id >= _TONE_LOG_FRAME:
            # Day 21: once, after the EMA has settled -- one read of the
            # six numbers (a wait for the GPU on this frame alone).
            loaded.tone_logged = True
            if _skin_match(loaded.options) > 0.0:
                tone = _as_numpy_tone(loaded.tone)
                _LOG.info(
                    "Avatar %s: skin tone matched -- cast %s, contrast %s, the room's skin %s "
                    "(skin_match %.2f)",
                    avatar_id,
                    np.round(tone[0], 3).tolist(),
                    np.round(tone[1], 3).tolist(),
                    np.round(tone[2], 0).tolist(),
                    _skin_match(loaded.options),
                )
        self.last_timings["wait_ms"] = self._paste_wait_ms
        self.last_timings["upload_ms"] = self._upload_ms
        pre, fit, post = self._paste_marks
        self.last_timings["paste_pre_ms"] = pre
        self.last_timings["paste_fit_ms"] = fit
        self.last_timings["paste_post_ms"] = post
        if hair_pending is not None:
            # The paste's last download drained the stream: the copy is
            # whole, and the worker copies it again for itself.
            self._submit_hair(loaded, hair_pending.numpy())
        if self._events_recorded:
            # The graph ran this frame and the paste's download has waited
            # for it: its time on the GPU is known now. ``launch_ms`` is
            # what the CPU spent launching it (``model_ms`` was that sum
            # before 2026-10-08, the CPU waiting the graph out).
            self._events_recorded = False
            events = self._model_events
            if events is not None:
                self.last_timings["launch_ms"] = self.last_timings.get("model_ms", 0.0)
                self.last_timings["model_ms"] = float(events[0].elapsed_time(events[1]))

        return RenderedFrame(
            data=output,
            pixel_format=frame.pixel_format,
            metadata=tracking.metadata,
            avatar_id=avatar_id,
            render_timestamp=now(),
            is_placeholder=False,
        )

    # --- the frame path ----------------------------------------------------

    @staticmethod
    def _smooth_crop(
        loaded: _Portrait, frame: VideoFrame, cx: float, cy: float, size: float
    ) -> tuple[float, float, float]:
        """The crop's centre and size through the portrait's One Euro
        filters (see :func:`_crop_filters`), stepped by the time since the
        last frame's capture timestamp; unchanged when smoothing is off."""
        timestamp = frame.metadata.capture_timestamp if frame.metadata is not None else None
        dt = _frame_dt(timestamp, loaded.last_timestamp)
        loaded.last_timestamp = timestamp
        loaded.last_dt = dt
        if loaded.crop_filters is None:
            return cx, cy, size
        fx, fy, fs = loaded.crop_filters
        return (
            cx if fx is None else fx(cx, dt),
            cy if fy is None else fy(cy, dt),
            size if fs is None else fs(size, dt),
        )

    @staticmethod
    def _gates(
        loaded: _Portrait, mar: float, width: float, brow: float, pose: HeadPose | None
    ) -> tuple[float, float, float]:
        """This frame's ``(mouth, opening, brow)`` expression gates, released
        against the last frame's (see :func:`_release`) and remembered for
        the next. All 1.0 -- and nothing remembered -- with
        ``expression_gates`` off or while the anchor is still being learned
        (his last reading is still remembered then, so the first gated frame
        has a frame to compare motion against).

        ``mouth`` weights every mouth component but the opening: the larger
        of how far his mouth is from rest (:func:`_mouth_gate`: ratio *or*
        width) and whether it is moving at all (:func:`_mouth_activity`,
        ratio and width together). ``opening`` weights the lips' own
        opening (:data:`_MOUTH_OPENING`) and knows only the *ratio*: how far
        it is from rest, and whether it is moving -- a smile, however wide,
        does not part the lips unless his part (day 9, night). Each is held
        open ``_MOUTH_HOLD_FRAMES`` past its last moving frame and opens over
        two frames (:data:`_GATE_ATTACK`). The brow gate is scaled by
        ``brow_gain`` (see :func:`_brow_gain`; 0.0 by default, so the brows
        hold) once the rest is known."""
        if not _expression_gates(loaded.options):
            return 1.0, 1.0, 1.0
        thresholds = loaded.thresholds
        yaw = pose.yaw if pose is not None else None
        yaw_change = 0.0 if yaw is None or loaded.yaw_prev is None else yaw - loaded.yaw_prev
        act_mar, act_width = _mouth_activity(
            mar, width, loaded.mar_prev, loaded.width_prev, yaw_change, thresholds
        )
        # In units of a 30-fps frame: a slower frame sees more change for
        # the same motion (see the last_dt slot).
        per_frame = _DEFAULT_FRAME_DT / max(loaded.last_dt, 1e-3)
        act_mar *= per_frame
        act_width *= per_frame
        loaded.mar_prev, loaded.width_prev, loaded.yaw_prev = mar, width, yaw
        mouth = _mouth_gate(mar, width, loaded.rest_mar, loaded.rest_mouth_width, pose, thresholds)
        brow_gate = _brow_gate(brow, loaded.rest_brow, pose)
        if loaded.rest_mar is None or loaded.rest_brow is None:
            return mouth, mouth, brow_gate
        opening = _soft_gate(mar - loaded.rest_mar, thresholds.mouth_gate_mar)

        def held(moving: float, hold_attr: str) -> float:
            if moving >= 1.0:
                setattr(loaded, hold_attr, _MOUTH_HOLD_FRAMES)
            elif getattr(loaded, hold_attr) > 0:
                setattr(loaded, hold_attr, getattr(loaded, hold_attr) - 1)
                moving = 1.0
            return moving

        moving_all = held(_soft_gate(act_mar + act_width, _MOUTH_ACTIVITY_GATE), "mouth_hold")
        moving_open = held(_soft_gate(act_mar, _MOUTH_ACTIVITY_GATE), "open_hold")
        # Open over two frames, not one (_GATE_ATTACK), so the mouth region
        # does not jump into place; the holds above keep them open after.
        mouth = min(max(mouth, moving_all), loaded.gate_mouth_prev + _GATE_ATTACK)
        opening = min(max(opening, moving_open), loaded.gate_open_prev + _GATE_ATTACK)
        brow_gate *= _brow_gain(loaded.options)
        mouth = _release(mouth, loaded.gate_mouth_prev)
        opening = _release(opening, loaded.gate_open_prev)
        brow_gate = _release(brow_gate, loaded.gate_brow_prev)
        # Day 16 evening: on a smile the lips part less (see
        # _mouth_smile_opening); the width gate says how much of a smile.
        smile_opening = _mouth_smile_opening(loaded.options)
        if smile_opening < 1.0 and loaded.rest_mouth_width is not None:
            widening = _mouth_widening(width, loaded.rest_mouth_width, pose, thresholds)
            opening *= 1.0 - (1.0 - smile_opening) * widening
        loaded.gate_mouth_prev = mouth
        loaded.gate_open_prev = opening
        loaded.gate_brow_prev = brow_gate
        return mouth, opening, brow_gate

    def _retarget_rest(self, loaded: _Portrait) -> tuple[Any, Any]:
        """The eye and lip networks' own output at the portrait's own ratios
        -- ``ratios`` as it stands at load -- see the ``eye_rest`` slot's
        comment for why this is not zero and why it is subtracted. Zeros
        (the day-8-and-before behaviour) with ``retarget_zero_at_rest``
        false, or for a network that is switched off and never runs."""
        torch = self._torch
        zeros = torch.zeros(1, NUM_KP, 3, device=self._device, dtype=torch.float32)
        eye_rest, lip_rest = zeros, zeros.clone()
        if not bool(loaded.options.get("retarget_zero_at_rest", True)):
            return eye_rest, lip_rest
        with torch.no_grad():
            if loaded.options.get("eye_retargeting", True):
                feat = torch.cat([loaded.x_s.view(1, -1), loaded.ratios[:, 0:3]], dim=1).half()
                eye_rest = self._nets["eye"](feat).float().view(1, NUM_KP, 3).clone()
            if loaded.options.get("lip_retargeting", False):
                feat = torch.cat([loaded.x_s.view(1, -1), loaded.ratios[:, 3:5]], dim=1).half()
                lip_rest = self._nets["lip"](feat).float().view(1, NUM_KP, 3).clone()
        return eye_rest, lip_rest

    def _render(
        self,
        loaded: _Portrait,
        driver_rgb: np.ndarray,
        ear: float,
        mar: float,
        eye_target: float,
        lip_target: float,
        anchor_ok: bool = True,
        size_sample: tuple[float, float] | None = None,
        rest_sample: tuple[float, float] | None = None,
        gates: tuple[float, float, float] = (1.0, 1.0, 1.0),
        brow_lift: float = 0.0,
        brow_deviation: float = 0.0,
        gaze: tuple[float, float] | None = None,
        centre_sample: tuple[float, float] | None = None,
        chin_sample: float | None = None,
        pose_sample: tuple[float, float] | None = None,
        steep_fade: float = 1.0,
        on_device: bool = False,
    ) -> Any:
        """The portrait's face with this driver's expression, 512x512 RGB --
        a numpy picture, or with ``on_device`` the same ``uint8`` picture as
        the tensor it is on the network's device (2026-10-08), so the frame
        path need not wait for the GPU here: the download, and the wait for
        the graph with it, happen in the paste, after the CPU has built the
        mask meanwhile.

        ``ear``/``mar`` are the driver's own raw readings -- what the
        anchor's rest is learned from, unrelated to the portrait; ``mar`` is
        the gates' opening over the jaw width (:func:`_mouth_open_ratio`,
        day 12), not the lip network's ratio. ``eye_target``/
        ``lip_target`` are what actually drives the retargeting networks
        (see :func:`_eye_target`/:func:`_lip_target`): relative to the
        driver's rest once it is known, so the two pairs diverge as soon as
        the anchor freezes and must not be conflated. ``size_sample`` is the
        driver's ``(face height in pixels, head-pose distance)`` this frame,
        collected on accepted anchor frames for :meth:`_driver_face_height`;
        ``None`` records nothing. ``rest_sample`` is his ``(mouth width,
        brow)`` reading this frame (:func:`_mouth_width_ratio`/
        :func:`_brow_ratio`), averaged into ``rest_mouth_width``/``rest_brow``
        on accepted anchor frames the same way ``ear``/``mar`` are; ``None``
        records nothing. ``gates`` is this frame's ``(mouth, brow)`` from
        :meth:`_gates`, written into ``loaded.gate`` before the replay;
        ``brow_lift`` his brows' signed relative deviation from rest
        (:func:`_brow_lift`, already scaled by ``brow_follow``), written
        into ``loaded.brow_delta`` through :data:`_BROW_LIFT_Y` the same way.
        ``chin_sample`` is his nose-tip-to-chin reach this frame in crop
        pixels (:func:`_nose_chin`, day 13), averaged into ``rest_chin`` on
        accepted anchor frames like ``rest_sample`` is; ``None`` records
        nothing.
        """
        torch = self._torch
        image = self._to_tensor(driver_rgb)
        with torch.no_grad():
            if loaded.anchor_seen < loaded.anchor_wanted:
                if anchor_ok:
                    self._learn_anchor(
                        loaded,
                        image,
                        ear,
                        mar,
                        size_sample,
                        rest_sample,
                        gaze_sample=gaze,
                        centre_sample=centre_sample,
                        chin_sample=chin_sample,
                        pose_sample=pose_sample,
                    )
                else:
                    loaded.anchor_rejected += 1
            # Positions 0, 1, 3 were set once at load and never move; see the
            # ``ratios`` slot's own comment.
            loaded.ratios[0, 2] = eye_target
            loaded.ratios[0, 4] = lip_target
            loaded.gate[0, list(MOUTH_KEYPOINTS), :] = gates[0]
            for k, axis in _MOUTH_OPENING:
                loaded.gate[0, k, axis] = gates[1]
            loaded.gate[0, list(BROW_KEYPOINTS), :] = gates[2]
            # Day 17: the blink's pull on the brows and the mouth corners
            # countered in the same rows (see _BLINK_BROW_HOLD).
            depth = _blink_depth(ear, loaded.rest_ear)
            brow_hold = loaded.blink_brow_hold * _blink_brow_curve(depth)
            mouth_hold = loaded.blink_mouth_hold * _blink_mouth_curve(
                depth
            ) + loaded.blink_mouth_lift * _blink_lift_curve(depth)
            for k, y_per_unit in _BROW_LIFT_Y.items():
                loaded.brow_delta[0, k, 1] = brow_lift * y_per_unit - brow_hold
            for k, axis in _MOUTH_SMILE:
                loaded.brow_delta[0, k, axis] = -mouth_hold
            # Day 15: what his brow deviation leaks into the mouth rows,
            # for _drive to subtract (see _BROW_MOUTH_LEAK). ``brow_deviation``
            # is his brow ratio less the anchor's rest, unscaled and not
            # dead-banded -- the leak follows the reading, not the drive.
            for (k, axis), slope in loaded.leak_slopes.items():
                loaded.mouth_leak[0, k, axis] = slope * loaded.leak_gain * brow_deviation
            # Day 11: the gaze drive, on the eye keypoints' x rows of the
            # same offset tensor, written fresh every frame (see the GAZE
            # section and _gaze_drive).
            # Day 17: the gaze fades with the turn (``steep_fade`` here is
            # the expression's fade times the gaze's own earlier one, see
            # _steep_fade and _gaze_yaw_fade); the drive's own state goes on.
            gaze_x = self._gaze_drive(loaded, gaze, ear) * steep_fade
            for k, x_per_unit in _GAZE_KP_X.items():
                loaded.brow_delta[0, k, 0] = gaze_x * x_per_unit
            loaded.static_in.copy_(image)
            # The graph's own time on the GPU, read after the frame's last
            # download (see animate) -- the CPU no longer waits here.
            events = self._model_events
            if events is None and self._device.type == "cuda":
                events = self._model_events = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
            if events is not None:
                events[0].record()
            loaded.graph.replay()
            if events is not None:
                events[1].record()
                self._events_recorded = True
            self._track_lip_rest(loaded, gates)
            out = loaded.static_out[0].permute(1, 2, 0).float().clamp_(0.0, 1.0).mul_(255.0)
            picture = out.to(torch.uint8)
            if on_device:
                return picture
            face: np.ndarray = picture.cpu().numpy()
        return face

    @staticmethod
    def _restart_anchor(loaded: _Portrait) -> None:
        """The head moved while the anchor was still being learned: the
        frames accepted so far were at a pose the person has left, and
        averaging them with the next still pose would put the anchor
        between the two. The accumulation starts over; nothing else (the
        hold, the gates, the smoothing state) is touched, and a frozen
        anchor is never restarted -- see :meth:`reset_anchor` for that."""
        if loaded.anchor_seen == 0 or loaded.anchor_seen >= loaded.anchor_wanted:
            return
        loaded.anchor_seen = 0
        loaded.anchor_sum = {}
        loaded.size_samples = []
        loaded.centre_samples = []
        loaded.pose_samples = []

    @staticmethod
    def _track_lip_rest(loaded: _Portrait, gates: tuple[float, float, float]) -> None:
        """After a replay: if the anchor is frozen and this frame's gates say
        the mouth is at rest -- for more than :data:`_LIP_REST_SETTLE` frames
        in a row -- move the anchor's lip-opening rows toward the extractor's
        reading of this frame (``exp_prev``, which the replay just wrote) by
        ``lip_rest_alpha``. One masked in-place op on the static tensor the
        graph reads; nothing else of the anchor moves. See
        :func:`_lip_rest_tracking`."""
        if loaded.lip_rest_alpha <= 0.0 or loaded.lip_rest_mask is None:
            return
        if loaded.anchor_seen < loaded.anchor_wanted or loaded.holding:
            loaded.lip_rest_run = 0
            return
        at_rest = gates[0] < _LIP_REST_GATE_SHUT and gates[1] < _LIP_REST_GATE_SHUT
        loaded.lip_rest_run = loaded.lip_rest_run + 1 if at_rest else 0
        if loaded.lip_rest_run > _LIP_REST_SETTLE:
            fast = loaded.lip_rest_steps < _LIP_REST_FAST_FRAMES
            alpha = min(loaded.lip_rest_alpha * (_LIP_REST_FAST_FACTOR if fast else 1.0), 1.0)
            loaded.exp_d0.add_(
                (loaded.exp_prev - loaded.exp_d0) * loaded.lip_rest_mask, alpha=alpha
            )
            loaded.lip_rest_steps += 1

    @staticmethod
    def _centre_weight(loaded: _Portrait, head_pose: HeadPose | None) -> float:
        """How far the crop centre sits toward the eye line this frame: the
        fade of his pitch away from the anchor's own over
        :data:`_CROP_CENTRE_PITCH` -- 0 (the rigid box) without a pose or
        before the anchor knows his resting pitch. See
        :func:`_crop_geometry`'s ``centre_weight``."""
        if head_pose is None or loaded.rest_pitch is None:
            return 0.0
        return _soft_gate(abs(head_pose.pitch - loaded.rest_pitch), _CROP_CENTRE_PITCH)

    @staticmethod
    def _gaze_drive(
        loaded: _Portrait, gaze: tuple[float, float] | None, ear: float | None = None
    ) -> float:
        """His horizontal gaze offset from his own rest, dead-banded,
        smoothed and gained -- in units of eye width, what the eye
        keypoints' x is moved by (see the GAZE section). Zero, and the
        smoothing state reset, until the anchor knows his rest or while the
        gain is off. While his lids are shut (``ear`` under
        :data:`_GAZE_BLINK_HOLD` of his rest) the last value is held: the
        tracker's iris estimate has nothing to see then and jumps, and a
        jump on keypoint 15's x opens the lids a little -- measured as a
        shallower blink (driving.mkv blink gain 0.85 -> 0.77) before this
        hold existed.

        Once the anchor is frozen the rest itself follows his gaze by
        :func:`_gaze_rest_tracking` (day 16 evening): where he looks most
        of the time is his rest, and the avatar looks at the viewer while
        he looks at the screen; a glance still passes through."""
        gain = _gaze_gain(loaded.options)
        if gaze is None or loaded.rest_gaze is None or gain == 0.0:
            loaded.gaze_prev = 0.0
            return 0.0
        rest_ear = loaded.rest_ear
        if ear is not None and rest_ear is not None and ear < _GAZE_BLINK_HOLD * rest_ear:
            return gain * loaded.gaze_prev
        frames = _gaze_rest_tracking(loaded.options)
        if frames > 0.0 and loaded.anchor_seen >= loaded.anchor_wanted and not loaded.holding:
            alpha = 1.0 / frames
            rx, ry = loaded.rest_gaze
            loaded.rest_gaze = (rx + (gaze[0] - rx) * alpha, ry + (gaze[1] - ry) * alpha)
        dx = gaze[0] - loaded.rest_gaze[0]
        dx = math.copysign(max(abs(dx) - _GAZE_DEADBAND, 0.0), dx)
        a = _GAZE_SMOOTHING
        loaded.gaze_prev = loaded.gaze_prev * a + dx * (1.0 - a)
        return gain * loaded.gaze_prev

    def _learn_anchor(
        self,
        loaded: _Portrait,
        image: Any,
        ear: float,
        mar: float,
        size_sample: tuple[float, float] | None = None,
        rest_sample: tuple[float, float] | None = None,
        gaze_sample: tuple[float, float] | None = None,
        centre_sample: tuple[float, float] | None = None,
        chin_sample: float | None = None,
        pose_sample: tuple[float, float] | None = None,
    ) -> None:
        """Average the driver's first *accepted* frames into the anchor, then
        freeze it -- see :func:`_anchor_frame_ok` for what "accepted" means.

        Expression, translation and scale are averaged as they are; the pose is
        averaged in degrees and rebuilt into a rotation, because a mean of
        rotation matrices is not one. ``ear``/``mar`` are averaged the same
        plain way, into :attr:`_Portrait.rest_ear`/``rest_mar``. The
        ``size_sample`` pairs (face height, head-pose distance) are only
        collected, and their *medians* become ``face_height_rest``/``z_rest``
        at the freeze -- medians because the tracker's first frame or two
        read a face 10% shorter than the settled ones on the owner's clip,
        and a mean would carry that. Until the freeze the crop is sized
        from each frame itself, which at rest is the same height, so the
        anchor's frames and the frames it serves are cut alike.

        The day-7 smoothing state (``exp_prev``/``t_prev``/``scale_prev``/
        ``pose_prev``, see :func:`_ema_`) is *also* hard-copied to this same
        running mean on every accepted frame, while the anchor is still
        being learned -- not left to drift on whatever a smoothed graph
        replay happened to blend it to during that window. Once the anchor
        freezes, this stops (this method is not called again for this
        avatar until :meth:`LivePortraitEngine.reset_anchor`), and the state
        evolves purely by EMA from that clean starting point.
        """
        torch = self._torch
        info = self._nets["motion"](image.float())
        sums = loaded.anchor_sum
        parts = {
            "exp": info["exp"].float().view(1, NUM_KP, 3),
            "t": info["t"].float().view(1, 3),
            "scale": info["scale"].float().view(1, 1),
            "pose": torch.stack([self._degrees(info[k]) for k in ("pitch", "yaw", "roll")]),
        }
        for key, value in parts.items():
            sums[key] = value if key not in sums else sums[key] + value
        sums["ear"] = ear if "ear" not in sums else sums["ear"] + ear
        sums["mar"] = mar if "mar" not in sums else sums["mar"] + mar
        if rest_sample is not None:
            width, brow = rest_sample
            sums["width"] = width if "width" not in sums else sums["width"] + width
            sums["brow"] = brow if "brow" not in sums else sums["brow"] + brow
        if gaze_sample is not None:
            gx, gy = gaze_sample
            sums["gaze_x"] = gx if "gaze_x" not in sums else sums["gaze_x"] + gx
            sums["gaze_y"] = gy if "gaze_y" not in sums else sums["gaze_y"] + gy
        if chin_sample is not None:
            sums["chin"] = chin_sample if "chin" not in sums else sums["chin"] + chin_sample
        if size_sample is not None:
            loaded.size_samples.append(size_sample)
        if centre_sample is not None:
            loaded.centre_samples.append(centre_sample)
        if pose_sample is not None:
            loaded.pose_samples.append(pose_sample)
        loaded.anchor_seen += 1
        n = float(loaded.anchor_seen)
        loaded.exp_d0.copy_(sums["exp"] / n)
        loaded.t_d0.copy_(sums["t"] / n)
        loaded.scale_d0.copy_(sums["scale"] / n)
        pitch, yaw, roll = (sums["pose"] / n).unbind(0)
        from sha256.avatar.liveportrait.camera import get_rotation_matrix

        loaded.R_d0.copy_(
            get_rotation_matrix(pitch, yaw, roll)  # type: ignore[no-untyped-call]
        )
        loaded.rest_ear = sums["ear"] / n
        loaded.rest_mar = sums["mar"] / n
        # The gates' own rest (day 9) freezes with the anchor: known only
        # once every accepted frame carried a rest_sample, else left None
        # (both gates then stay 1.0 -- see _gates).
        if "width" in sums and "brow" in sums:
            loaded.rest_mouth_width = sums["width"] / n
            loaded.rest_brow = sums["brow"] / n
        if "gaze_x" in sums:
            loaded.rest_gaze = (sums["gaze_x"] / n, sums["gaze_y"] / n)
        # Day 13: his resting nose-to-chin reach, the jaw drop's zero.
        if "chin" in sums:
            loaded.rest_chin = sums["chin"] / n
        loaded.exp_prev.copy_(sums["exp"] / n)
        loaded.t_prev.copy_(sums["t"] / n)
        loaded.scale_prev.copy_(sums["scale"] / n)
        loaded.pose_prev.copy_(sums["pose"] / n)
        loaded.pose_d0.copy_(sums["pose"] / n)
        # Day 15: the lips' rest is the portrait's, not his pressed one (see
        # _MOUTH_RESTS; day 16: or the driver's own measured rest); the
        # smoothing state follows the anchor it primes to.
        if _apply_mouth_rest(loaded):
            loaded.exp_prev.copy_(loaded.exp_d0)
        if loaded.anchor_seen == loaded.anchor_wanted:
            if loaded.size_samples:
                loaded.face_height_rest = float(
                    statistics.median(h for h, _ in loaded.size_samples)
                )
                loaded.z_rest = float(statistics.median(z for _, z in loaded.size_samples))
            if loaded.centre_samples:
                # Day 11: from here the crop follows his eye line under
                # pitch, by the offset that makes it the rigid box's own
                # crop at rest, faded in by his pitch away from the rest
                # pitch learned here (see _centre_weight).
                loaded.centre_offset = float(statistics.median(o for o, _ in loaded.centre_samples))
                loaded.rest_pitch = float(statistics.median(p for _, p in loaded.centre_samples))
            # Day 16: the anchor's own pose by the TRACKER, for the hold and
            # the calibration to measure from (see pose_samples).
            if loaded.pose_samples:
                loaded.rest_yaw = float(statistics.median(y for _, y in loaded.pose_samples))
                if loaded.rest_pitch is None:
                    loaded.rest_pitch = float(statistics.median(p for p, _ in loaded.pose_samples))
            _LOG.info(
                "Avatar %s: anchor frozen over %d frames (%d rejected) -- "
                "pose %.1f/%.1f/%.1f (pitch/yaw/roll), rest EAR %.3f, rest MAR %.3f, "
                "rest mouth width %s, rest brow %s, nose-chin %s px, "
                "face height %s px at distance %s",
                loaded.descriptor.id,
                loaded.anchor_seen,
                loaded.anchor_rejected,
                float(pitch.item()),
                float(yaw.item()),
                float(roll.item()),
                loaded.rest_ear,
                loaded.rest_mar,
                "?" if loaded.rest_mouth_width is None else f"{loaded.rest_mouth_width:.3f}",
                "?" if loaded.rest_brow is None else f"{loaded.rest_brow:.3f}",
                "?" if loaded.rest_chin is None else f"{loaded.rest_chin:.1f}",
                "?" if loaded.face_height_rest is None else f"{loaded.face_height_rest:.0f}",
                "?" if loaded.z_rest is None else f"{loaded.z_rest:.2f}",
            )

    def _drive(self, loaded: _Portrait, image: Any) -> Any:
        """Motion in, the portrait's face out. Tensor arithmetic only, so it
        captures into a graph; every reference it reads is a static tensor."""
        torch = self._torch
        info = self._nets["motion"](image.float())
        exp = info["exp"].float().view(1, NUM_KP, 3)
        scale = info["scale"].float().view(1, 1)
        t = info["t"].float().view(1, 3)
        pose = torch.stack([self._degrees(info[k]) for k in ("pitch", "yaw", "roll")])

        # Temporal smoothing of the extractor's own output (day 7,
        # 2026-09-05, see _motion_smoothing's docstring for the live
        # complaint this answers): an EMA over exp/t/scale/pose, in state
        # tensors that persist between graph replays (_Portrait's own
        # *_prev slots). At the default motion_smoothing=0.0 every _ema_
        # call below is exactly a no-op -- state becomes `new` bit-for-bit
        # -- so the unsmoothed path is unchanged; a nonzero value is baked
        # into the graph at capture time like every other option here, and
        # needs the avatar reloaded to change.
        a = _motion_smoothing(loaded.options)
        exp_now = _ema_(loaded.exp_prev, exp, a)
        pose_now = _ema_(loaded.pose_prev, pose, a)
        # Day 17 (his "looking sideways, head down"): with steep_pose =
        # "clamp" the turn applied to the portrait is capped at the hold
        # thresholds, relative to the anchor's own pose -- the network keeps
        # drawing, the expression goes on, and the head is drawn at the
        # steepest angle a frontal portrait can still carry while his own
        # goes further. The EMA state above keeps the real reading.
        if loaded.turn_boost is not None:
            # Day 22: the yaw past the knee driven harder (see _turn_boost).
            delta = pose_now - loaded.pose_d0
            past = torch.clamp(delta.abs() - _TURN_BOOST_KNEE, min=0.0)
            pose_now = loaded.pose_d0 + delta + loaded.turn_boost * past * delta.sign()
        if loaded.turn_gain is not None:
            # Day 23: the yaw delta scaled by the side it turns to (see
            # _turn_gains); pitch and roll untouched.
            # The extractor's yaw runs the other way from the tracker's: a
            # delta under 0 here is a turn to HIS LEFT (tracker yaw +),
            # measured on his session (turn_motion.py with one gain set).
            yaw_delta = pose_now[1:2] - loaded.pose_d0[1:2]
            gain = torch.where(yaw_delta < 0.0, loaded.turn_gain[0], loaded.turn_gain[1])
            pose_now = torch.cat(
                [pose_now[0:1], loaded.pose_d0[1:2] + yaw_delta * gain, pose_now[2:3]], dim=0
            )
        if loaded.steep_clamp and loaded.pose_limit is not None:
            pose_now = loaded.pose_d0 + torch.maximum(
                torch.minimum(pose_now - loaded.pose_d0, loaded.pose_limit), -loaded.pose_limit
            )
        # Day 9, later: the extractor's scale and translation are smoothed
        # harder than its expression and pose (size_smoothing, the same
        # _ema_ in the same state). Sitting still, the extractor's scale
        # read 0.1%/1.1% (p50/p90) differently frame to frame -- a face
        # that breathes in size for no reason ("bounce") -- hence 0.85, a
        # fifth of each frame. Day 20: the scale is unused since day 13
        # and 0.85 on the translation alone lagged the drawn head six
        # frames behind a quick turn; 0.3 now (see _DEFAULT_SIZE_SMOOTHING).
        a_size = max(a, _size_smoothing(loaded.options))
        t_now = _ema_(loaded.t_prev, t, a_size)
        scale_now = _ema_(loaded.scale_prev, scale, a_size)
        pitch_s, yaw_s, roll_s = pose_now.unbind(0)
        from sha256.avatar.liveportrait.camera import get_rotation_matrix

        rot_d = get_rotation_matrix(pitch_s, yaw_s, roll_s)  # type: ignore[no-untyped-call]

        # Relative motion, as LivePortrait's pipeline does it: the driver's
        # change of pose and expression since the anchor, applied to the
        # portrait's own pose and expression.
        rel_rot = rot_d @ loaded.R_d0.permute(0, 2, 1)
        # Day 8, 2026-09-06: rel_mask weights the eye keypoints
        # (EYE_KEYPOINTS) of the relative expression by eye_expression_gain
        # -- 1.0 (every other keypoint, always) is a no-op; at the default
        # 0.0 the extractor's own reading of the eyes never reaches the
        # portrait, and they are driven only by the eye retargeting network
        # below. See _region_masks.
        #
        # Day 9, 2026-09-06: ``gate`` weights MOUTH_KEYPOINTS/BROW_KEYPOINTS
        # by this frame's own _mouth_gate/_brow_gate, written in by _render
        # before the replay -- 1.0 while he is measurably smiling, talking
        # or raising his brows, 0.0 while his mouth and brows are at rest
        # whatever the extractor made of his lifted head. See the GATES
        # section above for the live complaint and the numbers.
        # Day 15: less the brow raise's leak into the mouth rows (mouth_leak,
        # zeros without a raise -- see _BROW_MOUTH_LEAK).
        delta = (
            loaded.exp_s
            + (exp_now - loaded.exp_d0 - loaded.mouth_leak) * loaded.rel_mask * loaded.gate
        )
        # Day 13: scale_gain is (1, 1) -- at 1.0 the ratio is exact and this
        # is the old scale_s * (scale_now / scale_d0); at the default 0.0 the
        # portrait keeps its own scale whatever the extractor reads into an
        # open mouth (see _scale_gain for the measurement behind it).
        scale_new = loaded.scale_s * (1.0 + (scale_now / loaded.scale_d0 - 1.0) * loaded.scale_gain)
        # Day 11: t_gain is [gx, gy, 0] -- at the default 1.0 the product is
        # exact, so this line is bit-for-bit what it was (see
        # _translation_gain for the brow-raise measurement behind it).
        t_new = loaded.t_s + (t_now - loaded.t_d0) * loaded.t_gain
        t_new = torch.cat([t_new[:, :2], torch.zeros_like(t_new[:, 2:])], dim=1)
        # Day 10, Fable's review of Step 1, and the second review after a
        # whole-rotation pivot broke head turns: rotating x_c_s about the
        # keypoint space's own origin (the "none" branch, today's math,
        # kept bit-for-bit -- x_c_s @ (rel_rot @ R_s) in one step, exactly
        # as before floating point's own associativity was a concern) turns
        # a driver's pitch into a slide, not a turn, once the crop no
        # longer carries his own translation (see _rotation_pivot). A
        # pivot adds a fixed translation instead -- but the SAME
        # translation for every axis the relative rotation carries, which
        # is what broke yaw the first time (see _rotation_pivot's own
        # docstring). Only the pitch delta's own translation is added here,
        # on top of the ordinary, untouched, whole-rotation x_c_s @ rel_rot
        # -- a pure yaw (rel_pitch == 0) makes it exactly zero, a pure
        # pitch reproduces the whole-rotation pivot exactly.
        # Day 16 (C2): the yaw delta's own translation about a point behind
        # the face (see _yaw_pivot_z), the same construction as the pitch
        # pivot's: exactly zero on a pure pitch, so the two never fight.
        t_yaw = None
        if loaded.yaw_pivot is not None:
            rel_yaw = yaw_s - loaded.pose_d0[1]
            zero_yaw = torch.zeros_like(rel_yaw)
            rot_y = get_rotation_matrix(zero_yaw, rel_yaw, zero_yaw)  # type: ignore[no-untyped-call]
            eye_y = torch.eye(3, dtype=rot_y.dtype, device=rot_y.device)[None]
            t_yaw = loaded.yaw_pivot @ (eye_y - rot_y)
            t_yaw = torch.cat([t_yaw[..., :2], torch.zeros_like(t_yaw[..., 2:])], dim=-1)
        if _rotation_pivot(loaded.options) == "none":
            if t_yaw is None:
                x_c_rotated = loaded.x_c_s @ (rel_rot @ loaded.R_s)
            else:
                x_c_rotated = (loaded.x_c_s @ rel_rot + t_yaw) @ loaded.R_s
        else:
            rel_pitch = pitch_s - loaded.pose_d0[0]
            zero_deg = torch.zeros_like(rel_pitch)
            rot_p = get_rotation_matrix(rel_pitch, zero_deg, zero_deg)  # type: ignore[no-untyped-call]
            eye3 = torch.eye(3, dtype=rot_p.dtype, device=rot_p.device)[None]
            t_p = loaded.pivot @ (eye3 - rot_p)
            if not _pivot_z(loaded.options):
                t_p = torch.cat([t_p[..., :2], torch.zeros_like(t_p[..., 2:])], dim=-1)
            if t_yaw is not None:
                t_p = t_p + t_yaw
            x_c_rotated = (loaded.x_c_s @ rel_rot + t_p) @ loaded.R_s
        x_d = scale_new[..., None] * (x_c_rotated + delta) + t_new[:, None, :]

        # Eye/lip retargeting (day 4, 2026-09-05): the driver's relative
        # expression above already carries some of a blink or an open mouth,
        # but measured against his own tracked reading it was only 36% of a
        # blink and 73% of a mouth opening -- these two networks correct
        # toward his own close ratio (see the ``ratios`` slot) instead of
        # leaving the shortfall.
        #
        # **The gains are small, and that is a real finding, not a guess.**
        # Both networks were trained on LivePortrait's own landmark scheme's
        # close-ratio numbers, not MediaPipe's; feeding them MediaPipe's ratio
        # directly (structurally the same input, a different numeric scale)
        # makes their output far stronger than intended, so gain 1.0 overshoots
        # badly. Calibrated 2026-09-05 on the owner's own clip, against his own
        # tracked reading, target = his raw ratio (day 4, before _eye_target/
        # _lip_target existed):
        #
        # | eye_gain | ear gain | r    |     | lip_gain | mar gain | r    |
        # |----------|----------|------|     |----------|----------|------|
        # | 0 (off)  | 0.42     | 0.78 |     | 0        | 0.75     | 0.99 |
        # | 0.02     | 0.53     | 0.82 |     | 0.05     | 0.81     | 0.99 |
        # | 0.04     | 0.64     | 0.85 |     | 0.10     | 0.87     | 0.99 |
        # | **0.06** | **0.70** | 0.86 |     | 0.15     | 0.94     | 0.99 |
        # | 0.07     | 0.70     | 0.86 |     | 0.20     | 0.99     | 0.99 |
        # | 0.08     | 0.69     | 0.85 |     | 0.30     | 1.10     | 0.99 |
        #
        # Eye rises smoothly to 0.06-0.07 and then plateaus -- more gain does
        # not buy more closing, it only leaves less budget before the same
        # overshoot a gain-1.0 probe hit. 0.70 is the ceiling this adaptation
        # reaches for the eye axis, not a stopping point chosen early.
        #
        # **Lip's own table above stopped being the right one the day after it
        # was measured.** Day 5 (2026-09-06) gave the lip target a floor at
        # the portrait's own closed ratio (:func:`_lip_target`) -- gain 0.2's
        # raw-target mar gain of 0.99 had come with the avatar's own lips
        # pressed to 38-44% of the owner's thickness, because nothing had
        # stopped the network asking for *less* gap than the portrait's own
        # shut mouth already has. Re-measured with the floor and with Day 5's
        # rigid-point crop (which also, unasked, raised the *unretargeted*
        # mar gain from 0.75 to 0.94 -- a more stable crop tracks the mouth
        # better even with no lip network at all):
        #
        # | lip_gain | mar gain | r    | lip thickness (upper/lower, avatar/his) |
        # |----------|----------|------|------------------------------------------|
        # | **0**    | **0.94** | 0.98 | **0.77 / 0.78**                          |
        # | 0.05     | 0.99     | 0.98 | 0.74 / 0.77                              |
        # | 0.10     | 1.02     | 0.98 | 0.73 / 0.75                              |
        # | 0.20     | 1.09     | 0.98 | 0.68 / 0.70                              |
        #
        # The floored network still only ever *presses*, never opens -- every
        # gain above 0 buys a little more mar gain at a real thickness cost,
        # and 0 already clears both the mar-gain and the thickness bar. So
        # ``lip_retargeting`` is off by default (see ``my_avatar_neural``'s
        # own ``engine_options``): the network earned its place for the eye
        # axis and did not for the lip one, once the crop itself stopped
        # being the reason the mouth was mistracked. The option and the gain
        # stay wired in -- a different avatar's own proportions, or a driver
        # whose relative motion undershoots more than this one's does, may
        # still need it.
        if loaded.options.get("eye_retargeting", True):
            eye_gain = float(loaded.options.get("eye_gain", 0.06))
            feat = torch.cat([loaded.x_s.view(1, -1), loaded.ratios[:, 0:3]], dim=1).half()
            # eye_mask (day 8, 2026-09-06): confines this network's own
            # delta to EYE_KEYPOINTS by default (eye_mask_output=True), so a
            # driven eye target cannot itself leak into the mouth or brows
            # the way the raw extractor's relative expression did. See
            # _region_masks.
            #
            # eye_rest (day 9, 2026-09-06): the network's own answer at the
            # portrait's own ratio, subtracted so that a target of "no
            # change" moves nothing -- see the slot's comment. Measured
            # before it: at rest the network closed the upper lids by more
            # than a whole blink's worth of its own differential output,
            # which is why no eye_gain ever reached the day-8 blink target
            # and why the avatar's eyes read 0.24 at rest against the
            # portrait's own 0.334.
            eye_out = self._nets["eye"](feat).float().view(1, NUM_KP, 3)
            x_d = x_d + eye_gain * (eye_out - loaded.eye_rest) * loaded.eye_mask
        if loaded.options.get("lip_retargeting", False):
            lip_gain = float(loaded.options.get("lip_gain", 0.0))
            feat = torch.cat([loaded.x_s.view(1, -1), loaded.ratios[:, 3:5]], dim=1).half()
            lip_out = self._nets["lip"](feat).float().view(1, NUM_KP, 3)
            x_d = x_d + lip_gain * (lip_out - loaded.lip_rest)
        # The brows, from his own measured height (day 9, later; see
        # _brow_lift and the brow_delta slot): a hand-built retargeting in
        # keypoint space, zero when his brows are at rest -- and, in the
        # same tensor, the constant mouth tilt trim (_mouth_tilt_trim).
        x_d = x_d + loaded.brow_delta

        # Off by default: measured 2026-09-05 on the owner's own clip, this
        # module's own 2-D shift is the whole of "the avatar sits to the side
        # of my face" (see the module docstring). The stitching correction
        # itself (the first NUM_KP*3 outputs) still runs when enabled -- it is
        # the shift, the last two, that is the problem, but the option is one
        # switch because nobody has needed the two apart.
        if loaded.options.get("stitching", False):
            feat = torch.cat([loaded.x_s.view(1, -1), x_d.view(1, -1)], dim=1).half()
            stitch = self._nets["stitching"](feat).float()
            x_d = x_d + stitch[:, : NUM_KP * 3].view(1, NUM_KP, 3)
            shift = stitch[:, NUM_KP * 3 : NUM_KP * 3 + 2].view(1, 1, 2)
            x_d = torch.cat([x_d[..., :2] + shift, x_d[..., 2:]], dim=-1)

        warped = self._nets["warping"](
            loaded.f_s, kp_source=loaded.x_s.half(), kp_driving=x_d.half()
        )["out"]
        return self._nets["decoder"](feature=warped)

    def _capture(self, loaded: _Portrait) -> None:
        """Warm the path up on a side stream, then record it as one graph."""
        torch = self._torch
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._drive(loaded, loaded.static_in)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loaded.static_out = self._drive(loaded, loaded.static_in)
        loaded.graph = graph

    def _paste(
        self,
        scene: np.ndarray,
        face: Any,
        to_frame: np.ndarray,
        loaded: _Portrait,
        mask: np.ndarray,
        extra: np.ndarray | None = None,
    ) -> np.ndarray:
        """The generated face into the frame, under its own oval, colour-matched.

        ``face`` is the 512x512 picture in the frame's channel order: a
        numpy ``uint8`` one, pasted here on the CPU as it always was, or --
        since 2026-10-08, the frame path on CUDA -- a ``3 x 512 x 512``
        float tensor still on the GPU, pasted by :meth:`_paste_gpu` with
        the same arithmetic; this function then only dispatches.

        ``mask`` is this frame's own paste mask -- the portrait's oval
        intersected with the driver's, see :func:`_frame_mask` -- not
        ``loaded.mask``, which is the portrait's half alone and no longer
        enough on its own (see that function's docstring for why).
        ``extra`` (day 20, see :func:`_turn_room`) is what may be added past
        the portrait's oval at a turn, pasted only where the render is the
        drawn face; ``None`` on a frontal frame.

        Bounded to ``loaded.mask_bbox``, the *portrait* mask's own box in the
        crop -- not the whole 512 square LivePortrait draws into, most of
        which is the crop's own margin around the face. That box still
        bounds every frame's mask because the intersection can only be
        smaller than the portrait's half, never larger. Measured
        2026-09-04, this machine, `driving-speech.mkv` frames 300-700: paste
        fell from 22.3 ms p50 (the whole square) to 7.57 ms p50 / 8.10 ms p90
        (the oval's own box, about 13% of the square's area on this avatar)
        -- also colour-matched now, at no extra line in this function, since
        the box shrinking is what made the lighting-plane fit affordable at
        every frame. Frames outside the box are byte-identical to the scene;
        see TestPaste in tests/unit/test_engine_liveportrait.py.
        """
        if not isinstance(face, np.ndarray):
            return self._paste_gpu(scene, face, to_frame, loaded, mask, extra=extra)

        import cv2

        assert loaded.mask_bbox is not None
        height, width = scene.shape[:2]
        # Day 17 (C1b): with the avatar's hair on, the paste covers the face
        # mask OR the hair matte, in a box grown to the hair's own; the
        # colour match is still fitted on the FACE mask alone (skin against
        # skin at the join, as day 13 built it), and its plane -- the room's
        # light -- is what the hair takes too. Off, every line below is what
        # it was: one mask, one box.
        box = loaded.mask_bbox
        hair = loaded.hair_on and loaded.hair_matte is not None
        if hair:
            grown = union_box(box, loaded.hair_box)
            assert grown is not None
            box = grown
        bx0, by0, bx1, by1 = box
        x0, y0, x1, y1 = _transformed_box(to_frame, box, width, height)
        output = scene.copy()
        if x1 - x0 < 2 or y1 - y0 < 2:
            return output

        # The same affine, re-based twice over: once so its source reads from
        # the mask box's own corner rather than the crop's (0, 0), and once so
        # its destination writes into this small output rather than the whole
        # frame.
        shifted = to_frame.copy()
        shifted[:, 2] = (
            to_frame[:, :2] @ np.array([bx0, by0], dtype=np.float32) + to_frame[:, 2] - (x0, y0)
        )
        size = (x1 - x0, y1 - y0)

        face_patch = face[by0:by1, bx0:bx1]
        mask_patch = mask[by0:by1, bx0:bx1, 0]
        # Day 19: the paste stops where the drawn face stops -- the mask
        # cut wherever the render shows the photograph's own background
        # (see _background_keep), by the turn's weight (see
        # _background_weight): nothing on a frontal frame, whose paste is
        # the one it was to the byte.
        weight = loaded.guard_weight
        keep = None
        if loaded.background is not None and (weight > 0.0 or extra is not None):
            keep = _background_keep(face_patch, loaded.background)
            if keep is not None and loaded.outline_scale is not None:
                assert loaded.outline_shift is not None
                # Day 22 evening: the keep through the outline fit's map,
                # as the render is (see _outline_maps).
                keep = _remap_outline(keep, loaded.outline_scale, loaded.outline_shift, bx0, by0)
        if keep is not None and weight > 0.0:
            mask_patch = mask_patch * (1.0 - weight * (1.0 - keep))
        if extra is not None:
            # Day 20: the room past the portrait's oval, the drawn face only.
            extra_patch = extra[by0:by1, bx0:bx1, 0]
            mask_patch = mask_patch + (extra_patch if keep is None else extra_patch * keep)
        if loaded.background is not None and loaded.crop_points is not None:
            # Day 22: the background under the drawn chin, every frame.
            chin_keep = _chin_keep(
                face_patch, loaded.background, float(loaded.crop_points[_CHIN, 1]) - by0
            )
            if chin_keep is not None:
                mask_patch = mask_patch * chin_keep

        placed = self._warp_face(face_patch, shifted, size, loaded, by0, bx0)
        mask = cv2.warpAffine(mask_patch, shifted, size, flags=cv2.INTER_LINEAR)[:, :, None]

        region = output[y0:y1, x0:x1].astype(np.float32)
        fit3 = cv2.merge((mask[:, :, 0], mask[:, :, 0], mask[:, :, 0]))
        alpha3 = fit3
        if hair:
            assert loaded.hair_matte is not None
            matte = loaded.hair_matte
            matte_patch = matte[by0:by1, bx0:bx1, 0] if matte.ndim == 3 else matte[by0:by1, bx0:bx1]
            blend = cv2.warpAffine(
                np.maximum(mask_patch, matte_patch), shifted, size, flags=cv2.INTER_LINEAR
            )
            alpha3 = cv2.merge((blend, blend, blend))
        features = None if loaded.features is None else loaded.features - (x0, y0)
        # Day 21: the render's skin given the room's tone before the plane
        # (see _skin_tone); at skin_match 0 nothing here runs and the
        # paste is the one before this day, to the byte.
        amount = _skin_match(loaded.options)
        if amount > 0.0:
            measure = loaded.tone_frames % _SKIN_TONE_EVERY == 0 and not loaded.holding
            loaded.tone_frames += 1
            if measure:
                small = (
                    max(size[0] // _CORRECTION_DOWNSCALE, 8),
                    max(size[1] // _CORRECTION_DOWNSCALE, 8),
                )
                face_small = cv2.resize(
                    cv2.multiply(placed, fit3), small, interpolation=cv2.INTER_AREA
                )
                scene_small = cv2.resize(
                    cv2.multiply(region, fit3), small, interpolation=cv2.INTER_AREA
                )
                coverage = cv2.resize(mask[:, :, 0], small, interpolation=cv2.INTER_AREA)
                zone = None
                if features is not None:
                    cells = features / (size[0] / small[0], size[1] / small[1])
                    zone = _feature_exclusion(cells.astype(np.float32), coverage.shape)
                skin = _skin_cells(face_small, scene_small, coverage, zone)
                loaded.tone = _tone_follow(
                    _as_numpy_tone(loaded.tone), _skin_tone(face_small, scene_small, coverage, skin)
                )
                loaded.tone_level = _core_median(face_small, coverage)
            cast, contrast, mean = _tone_applied(_as_numpy_tone(loaded.tone), amount)
            level = loaded.tone_level
            if not isinstance(level, float):
                level = float(level)
            placed = _tone_on_skin(placed, cast, contrast, mean, _tone_weight(placed, level))
        correction = _face_colour_correction(loaded, placed, region, mask, fit3, features)

        # Day 11 (fps): one fused blend. region + alpha * (face + correction
        # - region) is the same arithmetic as before -- (1 - alpha) * region
        # + alpha * clip(face + correction) -- less the clip on the relit
        # face, which the saturating cast at the end now does once for the
        # whole result; the two differ only where a correction would push a
        # pixel past 255 or under 0 and alpha is under 1 (the mask's edge),
        # by the overshoot times (1 - alpha). Four passes over the box
        # instead of eight, and no np.clip.
        delta = cv2.subtract(placed, region)
        if correction is not None:
            delta = cv2.add(delta, correction)
        output[y0:y1, x0:x1] = cv2.add(region, cv2.multiply(delta, alpha3), dtype=cv2.CV_8U)
        return output

    def _fit_outline(
        self,
        loaded: _Portrait,
        face: Any,
        mask: np.ndarray,
        extra: np.ndarray | None,
        crop_points: np.ndarray,
        head_pose: HeadPose | None,
        room: float,
        on_gpu: bool,
    ) -> Any:
        """Day 22 evening: the per-row map that fits the drawn face's near
        edge to the paste's at a turn, kept on ``loaded`` for the paste
        (see _outline_maps); cleared off a turn, so a frontal frame's
        paste is untouched. Day 23: before the map, the render's profile
        edge carried outward over the background (see _profile_fill); the
        face returned is the one to paste (the same object when nothing
        was done)."""
        weight = room * _outline_fit(loaded.options)
        fill_px = int(round(_profile_fill(loaded.options) * room))
        if (weight <= 0.0 and fill_px <= 0) or loaded.background is None or head_pose is None:
            loaded.outline_scale = None
            loaded.outline_shift = None
            return face
        near_left = float(head_pose.yaw) - (loaded.rest_yaw or 0.0) < 0.0
        paste = mask[:, :, 0] if extra is None else np.maximum(mask[:, :, 0], extra[:, :, 0])
        target_first, target_last = _row_edges(paste, _OUTLINE_TARGET_ALPHA)
        window = _outline_window(target_first, target_last, near_left)
        if on_gpu:
            drawn_near = self._drawn_near_edge_gpu(face, loaded.background, near_left, window)
            if fill_px > 0:
                face = self._profile_fill_gpu(
                    face, loaded.background, drawn_near, near_left, fill_px
                )
        else:
            drawn_near = _drawn_near_edge_np(np.asarray(face), loaded.background, near_left, window)
            if fill_px > 0:
                face = _profile_fill_np(
                    np.asarray(face), loaded.background, drawn_near, near_left, fill_px
                )
        if weight <= 0.0:
            loaded.outline_scale = None
            loaded.outline_shift = None
            return face
        # Day 23: the band starts at the eyes' top, not the brows' -- fitted
        # from the brows, his press at a turn to his right drew the brow
        # smeared sideways (the brow row wanted 19 px of widening, the eye
        # row 6); the fade above the first row still eases the eyes' scale
        # out over _OUTLINE_FADE_ROWS, so the brow takes little of it.
        brow_row = int(crop_points[[159, 386], 1].min()) - 4
        chin_row = int(crop_points[_CHIN, 1])
        eye_row = float(crop_points[list(_LEFT_EYE_CORNERS + _RIGHT_EYE_CORNERS), 1].mean())
        nose_band = (int(eye_row) + 4, int(crop_points[_NOSE_BOTTOM, 1]) + 4)
        previous = (
            None
            if loaded.outline_scale is None or loaded.outline_shift is None
            else (loaded.outline_scale, loaded.outline_shift)
        )
        maps = _outline_maps(
            drawn_near,
            target_first,
            target_last,
            near_left=near_left,
            first_row=brow_row,
            last_row=chin_row,
            weight=weight,
            previous=previous,
            exclude=nose_band,
        )
        if maps is None:
            loaded.outline_scale = None
            loaded.outline_shift = None
        else:
            loaded.outline_scale, loaded.outline_shift = maps
        return face

    def _drawn_near_edge_gpu(
        self,
        face: Any,
        background: np.ndarray,
        near_left: bool,
        window: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """:func:`_drawn_near_edge_np` on the device: ``face`` (3, H, W)
        float in the frame's channel order; one (H,) download."""
        torch = self._torch
        colour = torch.as_tensor(
            np.ascontiguousarray(
                background if self._frame_bgr else background[::-1], dtype=np.float32
            ),
            device=face.device,
        )[:, None, None]
        not_background = (face - colour).abs().amax(0) >= _BACKGROUND_NEAR
        not_border = face.amax(0) > _OUTLINE_BORDER
        drawn = not_background & not_border
        if window is not None:
            cols = torch.arange(int(face.shape[2]), device=face.device)[None, :]
            low = torch.as_tensor(window[0], device=face.device)[:, None]
            high = torch.as_tensor(window[1], device=face.device)[:, None]
            drawn = drawn & (cols >= low) & (cols <= high)
        width = int(drawn.shape[1])
        if near_left:
            first = drawn.float().argmax(dim=1)
        else:
            first = width - 1 - drawn.flip(1).float().argmax(dim=1)
        found = drawn.any(dim=1)
        edge = torch.where(found, first, torch.full_like(first, -1))
        return np.asarray(edge.cpu().numpy(), dtype=np.int32)

    def _profile_fill_gpu(
        self, face: Any, background: np.ndarray, edge: np.ndarray, near_left: bool, width_px: int
    ) -> Any:
        """:func:`_profile_fill_np` on the device: ``face`` (3, H, W) float
        in the frame's channel order; the edge uploaded (one (H,) tensor)."""
        torch = self._torch
        _, h, w = face.shape
        colour = torch.as_tensor(
            np.ascontiguousarray(
                background if self._frame_bgr else background[::-1], dtype=np.float32
            ),
            device=face.device,
        )[:, None, None]
        cols = torch.arange(w, device=face.device)[None, :]
        e = torch.as_tensor(edge.astype(np.int64), device=face.device)[:, None]
        inset = _PROFILE_FILL_INSET
        if near_left:
            beyond = (cols < e) & (cols >= e - width_px)
            rim = (cols >= e) & (cols < e + inset)
            inside = e + inset
        else:
            beyond = (cols > e) & (cols <= e + width_px)
            rim = (cols <= e) & (cols > e - inset)
            inside = e - inset
        is_background = (face - colour).abs().amax(0) < _BACKGROUND_NEAR
        write = (e >= 0) & ((beyond & is_background) | rim)
        src = torch.where(write, inside.clamp(0, w - 1), cols).expand(3, h, w)
        return face.gather(2, src)

    def _chin_keep_gpu(self, face: Any, background: np.ndarray, floor: float, bgr: bool) -> Any:
        """:func:`_chin_keep` on the device: ``face`` is (3, H, W) float in
        the frame's channel order, ``background`` the photograph's BGR
        colour; ``None`` when ``floor`` is past the patch. Grown by a max
        pool, feathered by a box blur of the guard's width -- no download."""
        import torch.nn.functional as functional

        torch = self._torch
        height = int(face.shape[1])
        start = int(max(0, math.floor(floor)))
        if start >= height:
            return None
        colour = torch.as_tensor(
            np.ascontiguousarray(background if bgr else background[::-1], dtype=np.float32),
            device=face.device,
        )[:, None, None]
        flat = (face[:, start:] - colour).abs().amax(0) < _BACKGROUND_NEAR
        if not bool(flat.any()):
            return None
        outside = torch.zeros((1, 1, height, int(face.shape[2])), device=face.device)
        outside[0, 0, start:] = flat.float()
        reach = 2 * _BACKGROUND_REACH + 1
        grown = functional.max_pool2d(outside, reach, stride=1, padding=reach // 2)
        soft = functional.avg_pool2d(
            1.0 - grown,
            _BACKGROUND_SOFT,
            stride=1,
            padding=_BACKGROUND_SOFT // 2,
            count_include_pad=False,
        )
        return soft[0, 0]

    def _squeeze_face_gpu(self, face: Any, loaded: _Portrait) -> Any:
        """Day 22: the rendered face (3, H, W) on the device, its columns
        scaled by ``loaded.width_scale`` about ``loaded.width_centre`` --
        the width fit, before the hair, the hold and the paste (see
        :func:`_squeeze_face` for the CPU picture). A bilinear resample of
        the render alone; the drive is not touched."""
        import torch.nn.functional as functional

        torch = self._torch
        _, h, w = face.shape
        scale = float(loaded.width_scale)
        tall = float(loaded.height_scale)
        # align_corners=True: pixel x is at normalised 2x/(w-1) - 1.
        cx = loaded.width_centre * 2.0 / max(w - 1, 1) - 1.0
        cy = loaded.height_centre * 2.0 / max(h - 1, 1) - 1.0
        theta = torch.tensor(
            [
                [1.0 / scale, 0.0, cx * (1.0 - 1.0 / scale)],
                [0.0, 1.0 / tall, cy * (1.0 - 1.0 / tall)],
            ],
            device=face.device,
            dtype=torch.float32,
        )[None]
        grid = functional.affine_grid(theta, [1, 3, int(h), int(w)], align_corners=True)
        return functional.grid_sample(
            face[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )[0]

    @staticmethod
    def _warp_face(
        face_patch: np.ndarray,
        shifted: np.ndarray,
        size: tuple[int, int],
        loaded: _Portrait,
        patch_top: int,
        patch_left: int = 0,
    ) -> np.ndarray:
        """The face patch warped into the paste box, float32: ``warpAffine``
        as ever -- or, with the forehead stretched (day 21,
        ``loaded.forehead_stretch`` over 1 and a ``stretch_row``), a remap
        by the same affine's inverse with the rows above the anchor row
        sampled closer to it, so the render above it is drawn
        ``forehead_stretch`` times taller; nothing below the row moves.
        ``patch_top`` is the patch's first crop row (the anchor row is a
        crop row)."""
        import cv2

        stretch = loaded.forehead_stretch
        row = loaded.stretch_row
        outline = loaded.outline_scale is not None and loaded.outline_shift is not None
        if (stretch <= 1.0 or row is None) and not outline:
            warped: np.ndarray = cv2.warpAffine(
                face_patch, shifted, size, flags=cv2.INTER_LINEAR
            ).astype(np.float32)
            return warped
        inverse = cv2.invertAffineTransform(shifted.astype(np.float64))
        xs, ys = np.meshgrid(
            np.arange(size[0], dtype=np.float32), np.arange(size[1], dtype=np.float32)
        )
        source_x = (inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2]).astype(np.float32)
        source_y = (inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2]).astype(np.float32)
        if stretch > 1.0 and row is not None:
            anchor = float(row - patch_top)
            above = source_y < anchor
            source_y = np.where(above, anchor - (anchor - source_y) / stretch, source_y).astype(
                np.float32
            )
        if outline:
            # Day 22 evening: the per-row horizontal map (crop coordinates;
            # see _outline_maps) -- the row read from the source row.
            assert loaded.outline_scale is not None and loaded.outline_shift is not None
            rows = np.clip(np.rint(source_y + patch_top), 0, len(loaded.outline_scale) - 1)
            rows = rows.astype(np.int32)
            scale_map = loaded.outline_scale[rows]
            shift_map = loaded.outline_shift[rows]
            source_x = ((source_x + patch_left) * scale_map + shift_map - patch_left).astype(
                np.float32
            )
        remapped: np.ndarray = cv2.remap(
            face_patch, source_x, source_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        ).astype(np.float32)
        return remapped

    def _paste_gpu(
        self,
        scene: np.ndarray,
        face: Any,
        to_frame: np.ndarray,
        loaded: _Portrait,
        mask: np.ndarray,
        extra: np.ndarray | None = None,
    ) -> np.ndarray:
        """:meth:`_paste` with the face still on the GPU (2026-10-08, "The
        fast version", step 2) -- the same steps in the same order, the
        full-box arithmetic on the GPU, the small arithmetic where it was.

        What moved: the three ``warpAffine``s become one ``grid_sample`` over
        the face, the mask and the hair blend as channels of one tensor,
        sampled where the inverse of the same affine says, at pixel centres
        the way OpenCV counts them (``align_corners=True`` on integer
        coordinates); the mask-weighted difference, the mask and the
        mask-weighted room are shrunk to the correction's cells with
        OpenCV's own area weights (:func:`_area_weights`) and *those* small
        arrays are downloaded -- a few kilobytes -- for the plane fit, the
        edge residual and the inertia, which run on the CPU unchanged
        (:func:`_correction_field`); the field comes back up and is widened
        bilinearly as ``cv2.resize`` would; the one blend and the saturating
        cast run on the GPU; the box alone is downloaded and written into a
        copy of the scene. What stays on the CPU besides the fit: the mask's
        own adjustments (the guard's keep, the turn's room), on the small
        crop-box patch, as before -- the keep needs the face patch and gets
        it downloaded on the frames the guard is on, a turn.

        Why it is not byte-identical: OpenCV's warp quantises the sampling
        position to 1/32 of a pixel and its area resize sums in another
        order, so a pixel at a gradient lands a grey level off now and then.
        The bar (the plan): at most 2 at p99 of the pixels that differ,
        under 0.5 mean over the frame, nothing outside the box.

        The first download (the small arrays) is where the CPU waits for
        the graph; ``_paste_wait_ms`` says how long.
        """
        import cv2
        import torch.nn.functional as functional

        torch = self._torch
        device = face.device
        started = now()
        assert loaded.mask_bbox is not None
        height, width = scene.shape[:2]
        box = loaded.mask_bbox
        hair = loaded.hair_on and loaded.hair_matte is not None
        if hair:
            grown = union_box(box, loaded.hair_box)
            assert grown is not None
            box = grown
        bx0, by0, bx1, by1 = box
        x0, y0, x1, y1 = _transformed_box(to_frame, box, width, height)
        output = scene.copy()
        if x1 - x0 < 2 or y1 - y0 < 2:
            return output

        shifted = to_frame.copy()
        shifted[:, 2] = (
            to_frame[:, :2] @ np.array([bx0, by0], dtype=np.float32) + to_frame[:, 2] - (x0, y0)
        )
        w, h = x1 - x0, y1 - y0
        pw, ph = bx1 - bx0, by1 - by0

        # The mask's patch and its adjustments, as the CPU paste makes them.
        mask_patch = mask[by0:by1, bx0:bx1, 0]
        weight = loaded.guard_weight
        keep = None
        if loaded.background is not None and (weight > 0.0 or extra is not None):
            face_patch = face[:, by0:by1, bx0:bx1].to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            keep = _background_keep(np.ascontiguousarray(face_patch), loaded.background)
            if keep is not None and loaded.outline_scale is not None:
                assert loaded.outline_shift is not None
                keep = _remap_outline(keep, loaded.outline_scale, loaded.outline_shift, bx0, by0)
        if keep is not None and weight > 0.0:
            mask_patch = mask_patch * (1.0 - weight * (1.0 - keep))
        if extra is not None:
            extra_patch = extra[by0:by1, bx0:bx1, 0]
            mask_patch = mask_patch + (extra_patch if keep is None else extra_patch * keep)
        # The mask -- and the hair blend -- warped on the CPU, by the very
        # call the CPU paste makes: float, unrounded, so the alpha the GPU
        # blends with is the CPU paste's to the bit, and its cell-size
        # resize, its sum and its distance transform can all be had here,
        # before anything waits for the graph.
        size = (w, h)
        alpha_np = cv2.warpAffine(mask_patch, shifted, size, flags=cv2.INTER_LINEAR)
        blend_np = None
        if hair:
            assert loaded.hair_matte is not None
            matte = loaded.hair_matte
            matte_patch = matte[by0:by1, bx0:bx1, 0] if matte.ndim == 3 else matte[by0:by1, bx0:bx1]
            blend_np = cv2.warpAffine(
                np.maximum(mask_patch, matte_patch), shifted, size, flags=cv2.INTER_LINEAR
            )

        # The face warped on the GPU: where each output pixel reads in the
        # patch is the inverse of the same affine, as warpAffine inverts
        # it, on integer pixel positions.
        inverse = cv2.invertAffineTransform(shifted.astype(np.float64))
        xs = torch.arange(w, device=device, dtype=torch.float32)
        ys = torch.arange(h, device=device, dtype=torch.float32)
        sx = (
            float(inverse[0, 0]) * xs[None, :]
            + float(inverse[0, 1]) * ys[:, None]
            + float(inverse[0, 2])
        )
        sy = (
            float(inverse[1, 0]) * xs[None, :]
            + float(inverse[1, 1]) * ys[:, None]
            + float(inverse[1, 2])
        )
        if loaded.forehead_stretch > 1.0 and loaded.stretch_row is not None:
            # Day 21: the rows above the anchor row sampled closer to it
            # (see _warp_face) -- the forehead drawn taller.
            anchor = float(loaded.stretch_row - by0)
            sy = torch.where(sy < anchor, anchor - (anchor - sy) / loaded.forehead_stretch, sy)
        if loaded.outline_scale is not None and loaded.outline_shift is not None:
            # Day 22 evening: the per-row horizontal map (see _outline_maps).
            scale_rows = self._upload("outline_scale", loaded.outline_scale, device)
            shift_rows = self._upload("outline_shift", loaded.outline_shift, device)
            rows = torch.clamp(torch.round(sy + by0), 0, scale_rows.shape[0] - 1).long()
            sx = (sx + bx0) * scale_rows[rows] + shift_rows[rows] - bx0
        grid = torch.stack(
            [sx * (2.0 / max(pw - 1, 1)) - 1.0, sy * (2.0 / max(ph - 1, 1)) - 1.0], dim=-1
        )[None]
        placed = functional.grid_sample(
            face[None, :, by0:by1, bx0:bx1],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0].round_()  # the uint8 warp rounds
        alpha = self._upload("alpha", alpha_np, device)[None]
        blend = alpha if blend_np is None else self._upload("blend", blend_np, device)[None]
        if loaded.background is not None and loaded.crop_points is not None:
            # Day 22: the background under the drawn chin, every frame -- the
            # keep made on the device and sampled by the face's own grid.
            chin_keep = self._chin_keep_gpu(
                face[:, by0:by1, bx0:bx1],
                loaded.background,
                float(loaded.crop_points[_CHIN, 1]) - by0,
                self._frame_bgr,
            )
            if chin_keep is not None:
                sampled = functional.grid_sample(
                    chin_keep[None, None],
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )[0]
                alpha = alpha * sampled
                blend = blend * sampled if blend is not alpha else alpha

        region = self._upload("region", output[y0:y1, x0:x1], device).permute(2, 0, 1).float()
        enqueued = now()
        # Day 21: the skin's tone, on the device (see _skin_tone_gpu) --
        # read one frame in _SKIN_TONE_EVERY, applied on every one.
        amount = _skin_match(loaded.options)
        if amount > 0.0:
            measure = loaded.tone_frames % _SKIN_TONE_EVERY == 0 and not loaded.holding
            loaded.tone_frames += 1
            placed = self._skin_tone_gpu(
                loaded, placed, region, alpha, alpha_np, amount, x0, y0, w, h, measure=measure
            )
        correction = self._colour_correction_gpu(
            loaded, placed, region, alpha, alpha_np, x0, y0, w, h
        )
        fitted = now()

        # The one fused blend of the CPU paste (day 11), and its saturating
        # cast to bytes.
        delta = placed - region
        if correction is not None:
            delta = delta + correction
        result = (region + delta * blend).round_().clamp_(0.0, 255.0).to(torch.uint8)
        output[y0:y1, x0:x1] = result.permute(1, 2, 0).cpu().numpy()
        # For last_timings: the CPU before the wait, the fit after it, the
        # blend and the download.
        self._paste_marks = (
            (enqueued - started) * 1000.0,
            (fitted - enqueued) * 1000.0 - self._paste_wait_ms,
            (now() - fitted) * 1000.0,
        )
        return output

    def _colour_correction_gpu(
        self,
        loaded: _Portrait,
        placed: Any,
        region: Any,
        alpha: Any,
        alpha_np: np.ndarray,
        x0: int,
        y0: int,
        w: int,
        h: int,
    ) -> Any:
        """:func:`_face_colour_correction` for the GPU paste. The mask's
        side of it is the CPU paste's own, computed here from ``alpha_np``
        (the CPU-warped mask) before anything waits: the full-box sum and
        its "too little mask" answer, the cell-size coverage by the same
        ``cv2.resize``, the distance transform for the edge's fade, his
        features' cells. The face's side -- the mask-weighted difference
        and room, shrunk to the cells with OpenCV's area weights -- is on
        the GPU, and so is the fit itself (:meth:`_fit_field_gpu`), so the
        CPU enqueues the whole correction and waits for the GPU only once,
        at the paste's last download. ``None`` when the mask covers too
        little of the box, as the CPU one says. With ``fit_on_gpu`` off,
        or cells too few for the widest blur, the fit runs on the CPU from
        downloaded planes (:func:`_correction_field`) -- a wait."""
        import cv2
        import torch.nn.functional as functional

        torch = self._torch
        device = alpha.device
        if float(alpha_np.sum()) < 32.0:
            return None
        small = (max(w // _CORRECTION_DOWNSCALE, 8), max(h // _CORRECTION_DOWNSCALE, 8))
        sw, sh = small
        coverage_np = cv2.resize(alpha_np, small, interpolation=cv2.INTER_AREA)
        features = None if loaded.features is None else loaded.features - (x0, y0)
        edge = _edge_match(loaded.options)

        rows = self._area_matrix("area_rows", h, sh, device)
        columns = self._area_matrix("area_columns", w, sw, device)
        difference = (region - placed) * alpha
        scene = region * alpha
        stacked = torch.cat([difference, scene], dim=0)
        shrunk = torch.matmul(torch.matmul(rows, stacked), columns.T)

        gpu_fit = self.fit_on_gpu and min(sh, sw) > (_EDGE_FILL_BLUR | 1) // 2 + 1
        if gpu_fit:
            exclude_t = None
            weight_t = None
            if edge > 0.0:
                if features is not None:
                    cells = features / (w / sw, h / sh)
                    zone = _feature_exclusion(cells.astype(np.float32), (sh, sw))
                    exclude_t = self._upload("exclude", zone, device)
                inside = (coverage_np > 0.5).astype(np.uint8)
                distance = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
                reach = max(float(sh) * _EDGE_REACH, 1.0)
                weight_np = np.clip(1.0 - distance / reach, 0.0, 1.0).astype(np.float32)
                weight_t = self._upload("edge_weight", weight_np, device)
            coverage_t = self._upload("coverage", coverage_np, device)
            field_t = self._fit_field_gpu(loaded, shrunk, coverage_t, exclude_t, weight_t, edge)
        else:
            waited = now()
            planes = shrunk.cpu().numpy()
            self._paste_wait_ms = (now() - waited) * 1000.0
            difference_small = np.ascontiguousarray(planes[:3].transpose(1, 2, 0))
            scene_small = np.ascontiguousarray(planes[3:6].transpose(1, 2, 0))
            field = _correction_field(
                loaded, difference_small, coverage_np, scene_small, features, (w, h)
            )
            field_t = self._upload("field", field, device).permute(2, 0, 1)
        return functional.interpolate(
            field_t[None], size=(h, w), mode="bilinear", align_corners=False
        )[0]

    def _fit_field_gpu(
        self,
        loaded: _Portrait,
        shrunk: Any,
        coverage: Any,
        exclude: Any,
        weight: Any,
        edge: float,
    ) -> Any:
        """:func:`_correction_field` on the GPU (2026-10-08): the dark
        cells, the plane, the edge residual and the inertia as tensor
        arithmetic on the cell planes already there, nothing of it waiting
        for the GPU -- the CPU enqueues it and goes on to the paste's last
        download. ``shrunk`` is the mask-weighted difference and room at
        the cell size (6 planes); ``coverage``, ``exclude`` and ``weight``
        come from the CPU, the CPU paste's own values. The median of the
        core cells is read without a boolean index (a sort, and a gather by
        a device-side count); the 3x3 plane solve is the closed form, with
        numpy's singular-matrix fallback chosen by the determinant; the
        blurs are the same Gaussian kernels as banded matrices over the
        reflect-101 border (against ``cv2.GaussianBlur``: 2e-7). The field
        is kept on the device for the next frame's inertia."""
        import torch.nn.functional as functional

        torch = self._torch
        device = shrunk.device
        difference = shrunk[:3]
        scene = shrunk[3:6]
        sh, sw = coverage.shape

        dark = self._dark_cells_gpu(scene, coverage)
        keep = 1.0 - dark

        # _fit_lighting_plane: the sums, and the 3x3 solve in closed form
        # (the adjugate over the determinant), in float64 on the GPU; a
        # zero determinant takes numpy's fallback -- the mean where there
        # is any mask, zeros where there is none.
        basis, pairs = self._plane_basis(sh, sw, device)
        weighted = coverage * keep
        target = difference * keep
        n = (pairs * weighted).double().sum(dim=(2, 3))
        targets = (basis[:, None] * target[None]).double().sum(dim=(2, 3))
        total = weighted.double().sum()
        channel_sums = target.double().sum(dim=(1, 2))
        c00 = n[1, 1] * n[2, 2] - n[1, 2] * n[2, 1]
        c01 = n[1, 2] * n[2, 0] - n[1, 0] * n[2, 2]
        c02 = n[1, 0] * n[2, 1] - n[1, 1] * n[2, 0]
        det = n[0, 0] * c00 + n[0, 1] * c01 + n[0, 2] * c02
        adjugate = torch.stack(
            [
                torch.stack(
                    [
                        c00,
                        n[0, 2] * n[2, 1] - n[0, 1] * n[2, 2],
                        n[0, 1] * n[1, 2] - n[0, 2] * n[1, 1],
                    ]
                ),
                torch.stack(
                    [
                        c01,
                        n[0, 0] * n[2, 2] - n[0, 2] * n[2, 0],
                        n[0, 2] * n[1, 0] - n[0, 0] * n[1, 2],
                    ]
                ),
                torch.stack(
                    [
                        c02,
                        n[0, 1] * n[2, 0] - n[0, 0] * n[2, 1],
                        n[0, 0] * n[1, 1] - n[0, 1] * n[1, 0],
                    ]
                ),
            ]
        )
        safe_det = torch.where(det != 0.0, det, torch.ones_like(det))
        solution = torch.matmul(adjugate, targets) / safe_det
        fallback = torch.zeros_like(solution)
        fallback[0] = torch.where(
            total >= 1e-6, channel_sums / total.clamp_min(1e-6), torch.zeros_like(channel_sums)
        )
        coefficients = torch.where(det != 0.0, solution, fallback).float()
        field = torch.einsum("kij,kc->cij", basis, coefficients)

        # _edge_residual, with the exclusion always present as there (the
        # dark cells at least); the residual and its weight blurred
        # together, as four planes, narrow and wide.
        if edge > 0.0:
            assert weight is not None
            zone = dark if exclude is None else torch.maximum(dark, exclude)
            keep_zone = 1.0 - zone
            residual = (difference - field * coverage) * keep_zone
            planes = torch.cat([residual, (coverage * keep_zone)[None]], dim=0)
            narrow = self._blur(planes, _EDGE_BLUR | 1)
            wide = self._blur(planes, _EDGE_FILL_BLUR | 1)
            smooth = narrow[:3] / narrow[3:4].clamp_min(1e-3)
            filled = wide[:3] / wide[3:4].clamp_min(1e-3)
            soften = self._blur(zone[None], _EDGE_EXCLUDE_SOFTEN | 1)
            smooth = smooth * (1.0 - soften) + filled * soften
            field = field + smooth * weight * edge

        previous = loaded.correction
        if previous is not None:
            if isinstance(previous, np.ndarray):
                previous = self._upload("previous", previous, device).permute(2, 0, 1)
            if tuple(previous.shape[1:]) != (sh, sw):
                previous = functional.interpolate(
                    previous[None], size=(sh, sw), mode="bilinear", align_corners=False
                )[0]
            field = previous * _CORRECTION_INERTIA + field * (1.0 - _CORRECTION_INERTIA)
        loaded.correction = field
        return field

    def _tone_identity(self, device: Any) -> Any:
        """:data:`_TONE_IDENTITY` on ``device``, kept."""
        key = ("tone_identity", str(device))
        identity = self._area_cache.get(key)
        if identity is None:
            identity = self._upload("tone_identity", _TONE_IDENTITY, device)
            self._area_cache[key] = identity
        return identity

    def _dark_cells_gpu(self, scene: Any, coverage: Any) -> Any:
        """:func:`_dark_cells_small` on the device (see :meth:`_core_median_gpu`
        for the median)."""
        return self._dark_cells_and_level_gpu(scene, coverage)[0]

    def _dark_cells_and_level_gpu(self, scene: Any, coverage: Any) -> tuple[Any, Any]:
        """:func:`_dark_cells_small` and :func:`_core_median` on the device,
        nothing of it waiting: the median of the core cells is numpy's --
        the two middle values of the sorted core, averaged -- gathered by
        device-side indices (a plain ``tensor[index_tensor]`` would read
        the index back to the host and wait for the GPU). The level is 0
        under 16 core cells, as the CPU's."""
        torch = self._torch
        lightness = scene.mean(dim=0) / coverage.clamp_min(1e-3)
        core = coverage > 0.9
        count = core.sum()
        ordered = torch.where(core, lightness, torch.full_like(lightness, math.inf))
        ordered = ordered.flatten().sort().values
        middle = torch.stack([torch.clamp(count - 1, min=0) // 2, count // 2])
        lower, upper = ordered.index_select(0, middle)
        median = (lower + upper) * 0.5
        dark = ((lightness < _EDGE_DARK_RATIO * median) & (coverage > 0.05)).float()
        dark = torch.where(count >= 16, dark, torch.zeros_like(dark))
        level = torch.where(count >= 16, median, torch.zeros_like(median))
        return dark, level

    def _skin_tone_gpu(
        self,
        loaded: _Portrait,
        placed: Any,
        region: Any,
        alpha: Any,
        alpha_np: np.ndarray,
        amount: float,
        x0: int,
        y0: int,
        w: int,
        h: int,
        measure: bool = True,
    ) -> Any:
        """:func:`_skin_cells`, :func:`_skin_tone`, :func:`_tone_follow` and
        :func:`_tone_applied` for the GPU paste (day 21): with ``measure``,
        the mask-weighted render and room shrunk to the correction's cells
        with the same area weights the correction uses, the skin cells and
        the nine numbers as tensor sums, the EMA state and the render's
        skin level kept on the device -- nothing of it waiting for the GPU;
        the "too few cells" rule a device-side ``where`` (the previous tone
        stays). Without it, the tone in force applied alone. Returns the
        render with the tone applied."""
        import cv2

        torch = self._torch
        device = alpha.device
        previous = loaded.tone
        if isinstance(previous, np.ndarray):
            previous = self._upload("tone", previous, device)
        level = loaded.tone_level
        if isinstance(level, float):
            level = torch.tensor(level, device=device)  # once: "unknown", before a reading
        if measure:
            small = (max(w // _CORRECTION_DOWNSCALE, 8), max(h // _CORRECTION_DOWNSCALE, 8))
            sw, sh = small
            coverage = self._upload(
                "tone_coverage", cv2.resize(alpha_np, small, interpolation=cv2.INTER_AREA), device
            )
            zone = None
            if loaded.features is not None:
                cells = (loaded.features - (x0, y0)) / (w / sw, h / sh)
                zone = self._upload(
                    "tone_zone", _feature_exclusion(cells.astype(np.float32), (sh, sw)), device
                )
            rows = self._area_matrix("area_rows", h, sh, device)
            columns = self._area_matrix("area_columns", w, sw, device)
            stacked = torch.cat([placed * alpha, region * alpha], dim=0)
            shrunk = torch.matmul(torch.matmul(rows, stacked), columns.T)
            face_s = shrunk[:3]
            scene_s = shrunk[3:6]
            skin = (coverage > 0.9).float()
            skin = skin * (1.0 - self._dark_cells_gpu(scene_s, coverage))
            dark_face, level = self._dark_cells_and_level_gpu(face_s, coverage)
            skin = skin * (1.0 - dark_face)
            if zone is not None:
                skin = skin * (1.0 - zone)
            count = skin.sum()
            safe = count.clamp_min(1.0)
            weight = coverage.clamp_min(1e-3)
            x = scene_s / weight
            y = face_s / weight
            mean_s = (x * skin).sum(dim=(1, 2)) / safe
            mean_f = (y * skin).sum(dim=(1, 2)) / safe
            var_s = (((x - mean_s[:, None, None]) ** 2) * skin).sum(dim=(1, 2)) / safe
            var_f = (((y - mean_f[:, None, None]) ** 2) * skin).sum(dim=(1, 2)) / safe
            cast = (mean_s / mean_f.clamp_min(1.0)).clamp(_SKIN_GAIN_MIN, _SKIN_GAIN_MAX)
            ratio = (var_s.sqrt() / (cast * var_f.clamp_min(1e-6).sqrt())).clamp(
                _SKIN_GAIN_MIN, _SKIN_GAIN_MAX
            )
            floor = _SKIN_SPREAD_MIN**2
            contrast = torch.where(
                (var_s >= floor) & (var_f >= floor), ratio, torch.ones_like(ratio)
            )
            measured = torch.stack([cast, contrast, mean_s])
            # _tone_follow: whole on the first measurement (the previous
            # still the identity), the EMA from then -- the inertia itself
            # a device-side choice, so nothing here waits.
            identity = self._tone_identity(device)
            started = (previous[:2] != identity[:2]).any()
            inertia = torch.where(started, _SKIN_TONE_INERTIA, 0.0)
            followed = previous * inertia + measured * (1.0 - inertia)
            previous = torch.where(count >= _SKIN_CELLS_MIN, followed, previous)
            loaded.tone = previous
            loaded.tone_level = level
        cast_applied = 1.0 + (previous[0] - 1.0) * amount
        contrast_applied = 1.0 + (previous[1] - 1.0) * amount
        mean = previous[2]
        # _tone_on_skin: the cast into every pixel, the contrast about the
        # room's skin mean by each pixel's lightness against the render's
        # skin (_tone_weight); an unknown level (0) is everywhere, as on
        # the CPU.
        casted = placed * cast_applied[:, None, None]
        deepened = (contrast_applied - 1.0)[:, None, None] * (casted - mean[:, None, None])
        lightness = placed.mean(dim=0) / level.clamp_min(1e-3)
        weight = ((lightness - _TONE_DARK_LOW) / (_TONE_DARK_HIGH - _TONE_DARK_LOW)).clamp(0.0, 1.0)
        weight = torch.where(level > 0.0, weight, torch.ones_like(weight))
        placed = casted + deepened * weight[None]
        return placed

    def _blur(self, planes: Any, size: int) -> Any:
        """``cv2.GaussianBlur(planes, (size, size), 0)`` on the GPU, for
        ``(C, H, W)`` planes: the same kernel (``getGaussianKernel``) with
        the reflect-101 border, as two banded matrices -- rows, then
        columns -- so it is two small matrix products, not a convolution
        whose plan cuDNN would search for anew at every new box size."""
        _count, height, width = planes.shape
        along_width = self._blur_matrix(width, size, planes.device)
        along_height = self._blur_matrix(height, size, planes.device)
        rows = self._torch.matmul(planes, along_width.T)
        return self._torch.matmul(along_height, rows)

    def _blur_matrix(self, n: int, size: int, device: Any) -> Any:
        """The 1-D Gaussian of ``size`` taps over ``n`` samples with OpenCV's
        default border (reflect-101) as an ``(n, n)`` matrix, kept per size."""
        key = ("blur", n, size, str(device))
        matrix = self._area_cache.get(key)
        if matrix is None:
            import cv2

            taps = cv2.getGaussianKernel(size, 0).astype(np.float64).ravel()
            radius = size // 2
            weights = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for t in range(size):
                    j = i + t - radius
                    # reflect-101: ... 2 1 | 0 1 2 ... n-2 n-1 | n-2 n-3 ...
                    while j < 0 or j >= n:
                        j = -j if j < 0 else 2 * n - 2 - j
                    weights[i, j] += taps[t]
            matrix = self._upload("blur_matrix", weights.astype(np.float32), device)
            self._area_cache[key] = matrix
        return matrix

    def _plane_basis(self, height: int, width: int, device: Any) -> tuple[Any, Any]:
        """:func:`_fit_lighting_plane`'s ``(1, x, y)`` basis for a cell
        grid -- the same values -- and its pairwise products for the normal
        matrix, kept on the device per size."""
        key = ("basis", height, width, str(device))
        cached = self._area_cache.get(key)
        if cached is None:
            ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
            xs = xs * (2.0 / max(width - 1, 1)) - 1.0
            ys = ys * (2.0 / max(height - 1, 1)) - 1.0
            stacked = np.stack((np.ones_like(xs), xs, ys)).astype(np.float32)
            basis = self._upload("basis", stacked, device)
            cached = (basis, basis[:, None] * basis[None, :])
            self._area_cache[key] = cached
        return cached

    def _upload(self, name: str, array: np.ndarray, device: Any) -> Any:
        """``array`` onto ``device`` without waiting for the GPU: through
        pinned memory, since a copy from ordinary (pageable) memory makes
        the driver drain the stream first -- the whole graph -- before it
        moves a byte, which is the wait the GPU paste exists to avoid. The
        pinned block comes from torch's caching host allocator, which does
        not hand it out again until the transfer has read it (a staging
        buffer of our own, reused by name, was overwritten under a pending
        transfer when two uploads shared a name in one frame -- the blur
        matrices -- and corrupted what the cache kept). ``name`` is kept
        for the timing's sake only."""
        del name
        torch = self._torch
        started = now()
        source = torch.from_numpy(np.ascontiguousarray(array))
        if getattr(device, "type", str(device)) != "cuda":
            # No device to move to (the CPU tests): a copy of its own.
            moved = source.clone()
        else:
            moved = source.pin_memory().to(device, non_blocking=True)
        self._upload_ms += (now() - started) * 1000.0
        return moved

    def _pinned_picture(self, name: str, picture: Any) -> Any:
        """``picture`` (a device tensor) copied down into pinned memory
        without waiting: whole once the stream next drains, which every
        frame's paste does at its last download. The block is the host
        allocator's, fresh per call and recycled by it."""
        del name
        torch = self._torch
        staging = torch.empty(picture.shape, dtype=picture.dtype, device="cpu", pin_memory=True)
        staging.copy_(picture, non_blocking=True)
        return staging

    def _area_matrix(self, name: str, n: int, m: int, device: Any) -> Any:
        """:func:`_area_weights` as a tensor on ``device``, kept per size:
        the box's size changes a little with the face's, so a few dozen
        sizes recur for a whole session. A new size goes up through the
        pinned staging under ``name`` (see _upload), so the frame that
        first meets it does not wait for the graph on its account."""
        key = ("area", n, m, str(device))
        matrix = self._area_cache.get(key)
        if matrix is None:
            if len(self._area_cache) > 1024:
                self._area_cache.clear()
            matrix = self._upload(name, _area_weights(n, m), device)
            self._area_cache[key] = matrix
        return matrix

    @staticmethod
    def _passthrough(
        scene: np.ndarray, frame: VideoFrame, tracking: FaceTrackingResult, avatar_id: str
    ) -> RenderedFrame:
        return RenderedFrame(
            data=scene,
            pixel_format=frame.pixel_format,
            metadata=tracking.metadata,
            avatar_id=avatar_id,
            render_timestamp=now(),
            is_placeholder=True,
            is_passthrough=True,
        )

    # --- tensors -----------------------------------------------------------

    def _to_tensor(self, rgb: np.ndarray) -> Any:
        """A 256x256 RGB uint8 image as the network's ``1x3xHxW`` half in 0..1."""
        torch = self._torch
        if rgb.shape[:2] != (MODEL_SIZE, MODEL_SIZE):
            import cv2

            rgb = cv2.resize(rgb, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to(self._device)
        return tensor.permute(2, 0, 1)[None].half().div_(255.0)

    def _degrees(self, pred: Any) -> Any:
        from sha256.avatar.liveportrait.camera import headpose_pred_to_degree

        return headpose_pred_to_degree(pred.float())  # type: ignore[no-untyped-call]

    def _rotation(self, info: dict[str, Any]) -> Any:
        from sha256.avatar.liveportrait.camera import get_rotation_matrix

        return get_rotation_matrix(  # type: ignore[no-untyped-call]
            self._degrees(info["pitch"]), self._degrees(info["yaw"]), self._degrees(info["roll"])
        )

    # --- loading -----------------------------------------------------------

    @staticmethod
    def _import_torch() -> Any:
        try:
            import torch
        except ImportError as exc:
            msg = (
                "the LivePortrait engine needs PyTorch with CUDA: "
                "uv pip install torch --index-url https://download.pytorch.org/whl/cu128"
            )
            raise EngineError(msg) from exc
        return torch

    def _weights_root(self) -> Path:
        return self._weights or (Path("models") / DEFAULT_WEIGHTS_RELATIVE_PATH)

    def _load_networks(self) -> dict[str, Any]:
        from sha256.avatar.liveportrait.appearance_feature_extractor import (
            AppearanceFeatureExtractor,
        )
        from sha256.avatar.liveportrait.motion_extractor import MotionExtractor
        from sha256.avatar.liveportrait.spade_generator import SPADEDecoder
        from sha256.avatar.liveportrait.stitching_retargeting_network import (
            StitchingRetargetingNetwork,
        )
        from sha256.avatar.liveportrait.warping_network import WarpingNetwork

        torch = self._torch
        root = self._weights_root()
        base = root / "base_models"
        files = {
            "appearance": base / "appearance_feature_extractor.pth",
            "motion": base / "motion_extractor.pth",
            "warping": base / "warping_module.pth",
            "decoder": base / "spade_generator.pth",
            "stitching": root / "retargeting_models" / "stitching_retargeting_module.pth",
        }
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            msg = (
                "LivePortrait weights are missing: "
                + ", ".join(missing)
                + ". Fetch them once with: python scripts/fetch_liveportrait.py"
            )
            raise EngineError(msg)

        def build(net: Any, path: Path, key: str | None = None, dtype: Any = None) -> Any:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if key is not None:
                state = state[key]
            state = {name.replace("module.", ""): value for name, value in state.items()}
            net.load_state_dict(state)
            return net.to(self._device, dtype or torch.float16).eval()

        # Day 12 (2026-09-07), his "the smile trembles": the motion extractor
        # runs in fp32, the other three in fp16. In fp16 its reading of a
        # *held* closed-lip smile (driving.mkv 806-834, his mouth and the
        # camera still to the pixel) flipped between three levels -- full
        # smile, half, neutral -- from one frame to the next, and every head
        # with it: frame-to-frame |change| p90, fp16 against fp32: mouth
        # keypoint 14 y 0.0030 vs 0.0002, yaw 0.64 vs 0.07 degrees, scale
        # 0.022 vs 0.003, t x 0.0034 vs 0.0003. The day-7 EMA on these
        # outputs (motion_smoothing, rejected) was treating this. It costs
        # nothing: 5.1 ms fp16, 5.0 ms fp32 eager on this machine (cuDNN's
        # convolutions run TF32 either way); the image it reads is cast on
        # the way in. bf16 was nearly as good (0.0004 / 0.09 / 0.004).
        return {
            "appearance": build(
                AppearanceFeatureExtractor(**_APPEARANCE_PARAMS),  # type: ignore[no-untyped-call]
                files["appearance"],
            ),
            "motion": build(
                MotionExtractor(**_MOTION_PARAMS),  # type: ignore[no-untyped-call]
                files["motion"],
                dtype=torch.float32,
            ),
            "warping": build(
                WarpingNetwork(**_WARPING_PARAMS),  # type: ignore[no-untyped-call]
                files["warping"],
            ),
            "decoder": build(
                SPADEDecoder(**_DECODER_PARAMS),  # type: ignore[no-untyped-call]
                files["decoder"],
            ),
            "stitching": build(
                StitchingRetargetingNetwork(  # type: ignore[no-untyped-call]
                    **_STITCHING_PARAMS
                ),
                files["stitching"],
                key="retarget_shoulder",
            ),
            # Same checkpoint as "stitching" -- it holds three networks under
            # three keys, not one file each.
            "eye": build(
                StitchingRetargetingNetwork(  # type: ignore[no-untyped-call]
                    **_EYE_RETARGET_PARAMS
                ),
                files["stitching"],
                key="retarget_eye",
            ),
            "lip": build(
                StitchingRetargetingNetwork(  # type: ignore[no-untyped-call]
                    **_LIP_RETARGET_PARAMS
                ),
                files["stitching"],
                key="retarget_mouth",
            ),
        }

    def _read_portrait(self, descriptor: AvatarDescriptor) -> np.ndarray:
        import cv2

        asset = descriptor.manifest.asset_by_role(SOURCE_PORTRAIT_ROLE)
        if asset is None:
            msg = (
                f"avatar {descriptor.id!r} declares no {SOURCE_PORTRAIT_ROLE!r} asset; "
                f"the LivePortrait engine needs a portrait photograph"
            )
            raise AvatarLoadError(msg)
        path = descriptor.asset_path(asset)
        if not path.is_file():
            msg = f"avatar {descriptor.id!r}: portrait not found at {path}"
            raise AvatarLoadError(msg)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            msg = f"{path.name} could not be read as an image"
            raise AvatarLoadError(msg)
        loaded: np.ndarray = image
        return loaded

    def _landmarks_of(self, image: np.ndarray, avatar_id: str) -> np.ndarray:
        """The portrait's face points in pixels, from the same landmarker the
        tracker uses, in IMAGE mode so the reading carries no state."""
        from sha256.tracking.mediapipe_tracker import (
            DEFAULT_MODEL_RELATIVE_PATH,
            MediaPipeFaceTracker,
        )

        model_path = self._model_path or (Path("models") / DEFAULT_MODEL_RELATIVE_PATH)
        tracker = MediaPipeFaceTracker(model_path=model_path, running_mode="image")
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
        points = result.landmarks.points
        if points.shape[0] < FACE_POINTS:
            msg = f"avatar {avatar_id!r}: the portrait's landmarks are unusable"
            raise AvatarLoadError(msg)
        height, width = image.shape[:2]
        pixels: np.ndarray = points[:FACE_POINTS, :2] * (width, height)
        return pixels.astype(np.float32)
