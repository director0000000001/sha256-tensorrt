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

import math
import statistics
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

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
CROP_SIZE = 512
MODEL_SIZE = 256
CROP_SCALE = 2.2
CROP_VY_RATIO = -0.125

# How many tracked frames the anchor — the driver's resting expression and
# pose — is averaged over before it is frozen. LivePortrait uses its first
# frame; a call does not start with a neutral face on cue.
DEFAULT_ANCHOR_FRAMES = 15

# What counts as a frame worth learning the anchor from (day 5, 2026-09-05).
# It used to be "the first 15 frames after load, whatever they hold" -- and
# an anchor learned while the owner was turned -12 deg away, reaching for the
# mouse, reproduced his own live report exactly: the avatar's nose sat +29 px
# from his (his own report: +27), the eyes read 26% wider than his at rest,
# and the mouth sat open a crack it should not have (MAR 0.05-0.16 against
# his 0.002). A gated anchor waits for frames that are actually frontal,
# open-eyed and closed-mouthed instead of freezing on whatever arrived first.
_ANCHOR_MAX_YAW = 4.0
_ANCHOR_MAX_PITCH = 4.0
_ANCHOR_MAX_ROLL = 3.0
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


def _anchor_frame_ok(pose: HeadPose | None, ear: float, mar: float) -> bool:
    """Whether this frame is worth learning the driver's rest from.

    Frontal, open-eyed, closed-mouthed -- see the constants' own comment for
    why. ``pose is None`` (no head pose this frame) is rejected outright: an
    anchor needs to know it was actually looking at the camera, not assume it.
    """
    if pose is None:
        return False
    return (
        abs(pose.yaw) <= _ANCHOR_MAX_YAW
        and abs(pose.pitch) <= _ANCHOR_MAX_PITCH
        and abs(pose.roll) <= _ANCHOR_MAX_ROLL
        and ear >= _ANCHOR_MIN_EAR
        and mar <= _ANCHOR_MAX_MAR
    )


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
_OVAL_SHRINK = 0.04
_OVAL_FEATHER = 0.07
_OVAL_FOREHEAD_REACH = 0.85

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

# Low-frequency relighting, exactly the warp engine's own technique
# (``engine_warp._match_colour`` / ``_fit_lighting_plane``), written again
# here rather than imported for the same reason the mask is: no shared
# internals between engines. See that function's docstring for why a plane
# and not a blur, and why the blur is weighted by the mask.
_CORRECTION_DOWNSCALE = 8
_CORRECTION_INERTIA = 0.8


def _mask_jaw_shrink(options: dict[str, Any]) -> float:
    """``mask_jaw_shrink`` in ``[0, 0.3]``, default ``_OVAL_JAW_SHRINK`` (0):
    how far the driver oval's jaw is pulled toward the face centre, as a
    fraction of face width. 0 is the join at the contour; 0.08 is the
    bearded setting measured on day 12."""
    value = float(options.get("mask_jaw_shrink", _OVAL_JAW_SHRINK))
    if not 0.0 <= value <= 0.3:
        raise ValueError(f"mask_jaw_shrink must be in [0, 0.3], got {value}")
    return value


def _oval_mask(
    crop_points: np.ndarray, size: int = CROP_SIZE, *, jaw_shrink: float = 0.0
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
    """
    import cv2

    oval = crop_points[list(_FACE_OVAL)]
    brow = crop_points[list(_BROWS)]

    trimmed = oval.copy()
    brow_top = float(brow[:, 1].min())
    oval_top = float(trimmed[:, 1].min())
    ceiling = brow_top - (brow_top - oval_top) * _OVAL_FOREHEAD_REACH
    np.maximum(trimmed[:, 1], ceiling, out=trimmed[:, 1])

    face_width = max(float(crop_points[:, 0].max() - crop_points[:, 0].min()), 1.0)
    if jaw_shrink > 0.0:
        # The crop is rolled with the face (see _similarity), so "down" is +y
        # here whatever his head does in the frame.
        eye_y = float(crop_points[list(_LEFT_EYE_CORNERS + _RIGHT_EYE_CORNERS), 1].mean())
        nose_y = float(crop_points[_NOSE_TIP, 1])
        weight = np.clip((trimmed[:, 1] - eye_y) / max(nose_y - eye_y, 1.0), 0.0, 1.0)
        inward = trimmed.mean(axis=0) - trimmed
        inward /= np.maximum(np.linalg.norm(inward, axis=1), 1e-6)[:, None]
        trimmed = trimmed + inward * (weight * face_width * jaw_shrink)[:, None]
    shrink = max(1, int(face_width * _OVAL_SHRINK))
    feather = max(3, int(face_width * _OVAL_FEATHER) | 1)

    filled = np.zeros((size, size), dtype=np.uint8)
    cv2.fillConvexPoly(filled, cv2.convexHull(trimmed.astype(np.int32)), 255)
    eroded = cv2.erode(
        filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink | 1, shrink | 1))
    )
    softened = cv2.GaussianBlur(eroded, (feather, feather), 0)
    return (softened.astype(np.float32) / 255.0)[:, :, None]


def _mask_bbox(mask: np.ndarray, margin: int = _MASK_BOX_MARGIN) -> tuple[int, int, int, int]:
    """Where ``mask`` is non-zero, with a small margin, as integer crop pixels.

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
    y1 = min(int(nonzero[:, 0].max()) + 1 + margin, size)
    x0 = max(int(nonzero[:, 1].min()) - margin, 0)
    x1 = min(int(nonzero[:, 1].max()) + 1 + margin, size)
    return x0, y0, x1, y1


def _frame_mask(
    portrait_mask: np.ndarray, driver_crop_points: np.ndarray, *, jaw_shrink: float = 0.0
) -> np.ndarray:
    """The paste mask for one frame: the portrait's oval **and** the driver's.

    ``jaw_shrink`` goes to the driver's oval alone (see :func:`_oval_mask`):
    the intersection then follows the driver's pulled-in jaw, and the
    portrait's own mask, built once at load, is unchanged -- its box still
    bounds every frame's paste.

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
    combined: np.ndarray = np.minimum(
        portrait_mask, _oval_mask(driver_crop_points, jaw_shrink=jaw_shrink)
    )
    return combined


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


def _size_smoothing(options: dict[str, Any]) -> float:
    """The EMA weight on the previous frame's extractor *scale and
    translation* (``size_smoothing``, default 0.85; 0.0 off) -- applied in
    :meth:`LivePortraitEngine._drive` on top of ``motion_smoothing``, which
    stays at 0.0 for the expression. See the comment there."""
    a = float(options.get("size_smoothing", 0.85))
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
_MOUTH_OPENING = ((17, 1), (19, 1), (20, 1))

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


def _mouth_axis_gain(options: dict[str, Any], key: str, default: float) -> float:
    g = float(options.get(key, default))
    if not (0.0 <= g <= 1.0):
        raise ValueError(f"{key} must be in [0, 1], got {g}")
    return g


def _region_masks(
    eye_expression_gain: float,
    eye_mask_output: bool,
    torch: Any,
    device: Any,
    *,
    mouth_lateral_gain: float = 0.0,
    mouth_smile_gain: float = 1.0,
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
    rel_mask = torch.ones(1, NUM_KP, 3, device=device, dtype=torch.float32)
    rel_mask[0, list(EYE_KEYPOINTS), :] = eye_expression_gain
    # The mouth's own axes (day 9, later; see _MOUTH_LATERAL/_MOUTH_SMILE):
    # the tilt components and the smile component, each weighted.
    for k, axis in _MOUTH_LATERAL:
        rel_mask[0, k, axis] = mouth_lateral_gain
    for k, axis in _MOUTH_SMILE:
        rel_mask[0, k, axis] = mouth_smile_gain
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


def _brow_follow(options: dict[str, Any]) -> float:
    g = float(options.get("brow_follow", 0.0))
    if not (0.0 <= g <= 3.0):
        raise ValueError(f"brow_follow must be in [0, 3], got {g}")
    return g


def _brow_lift(brow: float, rest_brow: float | None, pose: HeadPose | None) -> float:
    """How far his brows are *raised* past rest this frame, beyond
    :data:`_BROW_DEADBAND`, faded with pose -- the scalar
    :meth:`LivePortraitEngine._render` turns into a keypoint offset via
    :data:`_BROW_LIFT_Y`. Never negative: a lowering is not carried (see
    the constants' own comment for why). 0.0 until the anchor has frozen."""
    if rest_brow is None:
        return 0.0
    lift = max(brow / (rest_brow or 1e-6) - 1.0 - _BROW_DEADBAND, 0.0)
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
    widening = _soft_gate(width / (rest_width or 1e-6) - 1.0, _MOUTH_GATE_WIDTH)
    if pose is not None:
        widening *= _fade(pose.yaw, _MOUTH_GATE_YAW)
    return max(_soft_gate(mar - rest_mar, _MOUTH_GATE_MAR), widening)


def _mouth_activity(
    mar: float,
    width: float,
    mar_prev: float | None,
    width_prev: float | None,
    yaw_change: float = 0.0,
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
    width_change = abs(width / (width_prev or 1e-6) - 1.0)
    width_change = max(width_change - _WIDTH_PER_DEGREE_YAW * abs(yaw_change), 0.0)
    return abs(mar - mar_prev) / _MOUTH_ACTIVITY_MAR, width_change / _MOUTH_ACTIVITY_WIDTH


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
_DEFAULT_HOLD_PITCH = 20.0
_DEFAULT_HOLD_YAW = 35.0
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


def _should_hold(loaded: _Portrait, head_pose: HeadPose | None) -> bool:
    """Whether this frame should reuse the last rendered face rather than
    ask the network to draw a pose a frontal portrait cannot hold.

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
    if head_pose is None:
        return loaded.holding
    hold_pitch, hold_yaw = _hold_thresholds(loaded.options)
    if loaded.holding:
        holding = (
            abs(head_pose.pitch) > hold_pitch * _HOLD_RELEASE_RATIO
            or abs(head_pose.yaw) > hold_yaw * _HOLD_RELEASE_RATIO
        )
    else:
        holding = abs(head_pose.pitch) > hold_pitch or abs(head_pose.yaw) > hold_yaw
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


def _face_colour_correction(
    loaded: _Portrait,
    face: np.ndarray,
    region: np.ndarray,
    alpha: np.ndarray,
    alpha3: np.ndarray,
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
    field = _fit_lighting_plane(difference, coverage)

    previous = loaded.correction
    if previous is not None:
        resized = cv2.resize(previous, (field.shape[1], field.shape[0]))
        field = resized * _CORRECTION_INERTIA + field * (1.0 - _CORRECTION_INERTIA)
    loaded.correction = field

    correction: np.ndarray = cv2.resize(field, (width, height), interpolation=cv2.INTER_LINEAR)
    return correction


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
        "brow_delta",
        "c_s_eye_mean",
        "c_s_lip",
        "centre_offset",
        "centre_samples",
        "correction",
        "crop_filters",
        "descriptor",
        "ear_recent",
        "exp_d0",
        "exp_prev",
        "exp_s",
        "eye_mask",
        "eye_rest",
        "f_s",
        "face_height_rest",
        "gate",
        "gate_brow_prev",
        "gate_mouth_prev",
        "gate_open_prev",
        "gaze_prev",
        "graph",
        "holding",
        "last",
        "last_dt",
        "last_timestamp",
        "lip_rest",
        "mar_prev",
        "mask",
        "mask_bbox",
        "mouth_hold",
        "open_hold",
        "options",
        "pivot",
        "pose_d0",
        "pose_prev",
        "ratios",
        "rel_mask",
        "rest_brow",
        "rest_ear",
        "rest_gaze",
        "rest_mar",
        "rest_mouth_width",
        "rest_pitch",
        "scale_d0",
        "scale_prev",
        "scale_s",
        "size_samples",
        "static_in",
        "static_out",
        "t_d0",
        "t_gain",
        "t_prev",
        "t_s",
        "vram_mb",
        "width_prev",
        "x_c_s",
        "x_s",
        "yaw_prev",
        "z_rest",
    )

    def __init__(self, descriptor: AvatarDescriptor) -> None:
        self.descriptor = descriptor
        self.options = dict(descriptor.manifest.engine_options)
        self.anchor_wanted = int(self.options.get("anchor_frames", DEFAULT_ANCHOR_FRAMES))
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
        # Day 11: his resting iris offset (x, y) in units of eye width,
        # learned with the anchor like rest_brow; and the gaze drive's own
        # EMA state -- see the GAZE section and _gaze_drive.
        self.rest_gaze: tuple[float, float] | None = None
        self.gaze_prev: float = 0.0
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
        # Day 11: the crop centre's fixed vertical offset off the eye line,
        # in units of the crop size -- collected on accepted anchor frames
        # when crop_centre is "eyeline" and frozen as their median with the
        # anchor; None (before the freeze, or with "rigid") keeps the rigid
        # box (see _crop_geometry's own docstring).
        self.centre_samples: list[tuple[float, float]] = []
        self.centre_offset: float | None = None
        self.rest_pitch: float | None = None
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
        self.last: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        # The portrait's own oval, feathered, in the 512x512 crop the network
        # always draws into — computed once at load from the portrait's own
        # landmarks and reused every frame as one half of :func:`_frame_mask`.
        # See :func:`_oval_mask`.
        self.mask: np.ndarray | None = None
        self.mask_bbox: tuple[int, int, int, int] | None = None
        # The low-frequency relighting field, carried between frames the same
        # way the warp engine carries it — see :data:`_CORRECTION_INERTIA`.
        self.correction: np.ndarray | None = None
        # Day 8, 2026-09-06: whether the last rendered face is currently
        # being reused rather than redrawn, at a pose too steep for a
        # frontal portrait -- see _should_hold. Starts un-held; reset_anchor
        # also clears it, so a fresh anchor starts un-held too.
        self.holding: bool = False


class LivePortraitEngine:
    """Drives a portrait with LivePortrait on CUDA."""

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        weights: Path | str | None = None,
    ) -> None:
        # The face-landmark model, to read the portrait's own face at load and
        # to name the driver's crop; the same one the tracker uses.
        self._model_path = Path(model_path) if model_path is not None else None
        self._weights = Path(weights) if weights is not None else None
        self._running = False
        self._torch: Any = None
        self._device: Any = None
        self._nets: dict[str, Any] = {}
        self._loaded: dict[str, _Portrait] = {}
        # How long the last frame's parts took, in ms, for the offline harness
        # and the debug panel. Not a measurement of anything until read.
        self.last_timings: dict[str, float] = {}

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
        self._nets = {}
        self._running = False
        if self._torch is not None:
            self._torch.cuda.empty_cache()

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
        loaded = _Portrait(descriptor)
        scale = float(loaded.options.get("crop_scale", CROP_SCALE))
        vy = float(loaded.options.get("crop_vy_ratio", CROP_VY_RATIO))

        # Day 11: the portrait is always cut by the rigid rule; crop_centre
        # "eyeline" changes only what the *driver's* crop follows, with an
        # offset learned at his own anchor (see _crop_geometry's own
        # docstring) -- validated here, applied in animate.
        _crop_centre(loaded.options)
        _mask_jaw_shrink(loaded.options)
        to_crop, _ = crop_matrix(points, scale=scale, vy_ratio=vy)
        crop = cv2.warpAffine(portrait, to_crop, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_LINEAR)
        image = self._to_tensor(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        # The paste mask, from the portrait's own landmarks placed in this
        # same crop -- see _oval_mask's docstring for why this is done once
        # here rather than every frame from the driver's landmarks.
        crop_points = points @ to_crop[:, :2].T + to_crop[:, 2]
        loaded.mask = _oval_mask(crop_points.astype(np.float32))
        loaded.mask_bbox = _mask_bbox(loaded.mask)

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
            info = self._nets["motion"](image)
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
            loaded.centre_samples = []
            loaded.centre_offset = None
            loaded.rest_pitch = None
            loaded.holding = False

    def rest_status(self, avatar_id: str) -> str | None:
        """A line for the UI while the anchor is still being learned, else
        ``None`` -- both when it is already frozen and when the avatar is
        unknown to this engine."""
        loaded = self._loaded.get(avatar_id)
        if loaded is None or loaded.anchor_seen >= loaded.anchor_wanted:
            return None
        return (
            "Look straight at the camera to centre the avatar "
            f"({loaded.anchor_seen}/{loaded.anchor_wanted})"
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
        itself instead."""
        if loaded.face_height_rest is None or loaded.z_rest is None or head_pose is None:
            return None
        z = abs(float(head_pose.translation[2]))
        if z <= 1e-6:
            return None
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

        target = None
        gaze = None
        if tracking.face_detected and tracking.landmarks is not None:
            normalised = tracking.landmarks.points
            if normalised.shape[0] >= FACE_POINTS:
                target = normalised[:FACE_POINTS, :2] * (width, height)
                # Day 11: his irises, from the same points past the 468
                # the face itself uses -- see the GAZE section.
                gaze = _gaze_offset(normalised[:, :2] * (width, height))

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
            face, to_frame, mask = loaded.last
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
            if _should_hold(loaded, tracking.head_pose) and loaded.last is not None:
                face, _, _ = loaded.last
                # Whatever his mouth does while held is not seen; the first
                # frame after must not read the jump as speech (_gates).
                loaded.mar_prev = loaded.width_prev = None
                rendered = now()
            else:
                c_d_eye = sum(_eye_ratios(target)) / 2.0
                c_d_lip = _lip_ratio(target)
                # Day 12: the gates and the rest read the opening over the
                # jaw width; c_d_lip stays for the anchor and the lip net.
                c_d_open = _mouth_open_ratio(target)
                c_d_width = _mouth_width_ratio(target)
                c_d_brow = _brow_ratio(target)
                anchor_ok = _anchor_frame_ok(tracking.head_pose, c_d_eye, c_d_lip)
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
                if anchor_ok and tracking.head_pose is not None:
                    z = abs(float(tracking.head_pose.translation[2]))
                    if z > 1e-6:
                        size_sample = (_face_height(target), z)
                    if _crop_centre(loaded.options) == "eyeline":
                        centre_sample = (
                            _centre_offset(target, scale=scale, vy_ratio=vy),
                            float(tracking.head_pose.pitch),
                        )
                gates = self._gates(loaded, c_d_open, c_d_width, c_d_brow, tracking.head_pose)
                brow_lift = _brow_follow(loaded.options) * _brow_lift(
                    c_d_brow, loaded.rest_brow, tracking.head_pose
                )
                face_rgb = self._render(
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
                    gaze=gaze,
                    centre_sample=centre_sample,
                )
                face = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR) if bgr else face_rgb
                rendered = now()

            # The driver's own oval, in the same crop the portrait's was built
            # in -- see _frame_mask. `target` is already the full 468-point
            # set in frame pixels; the same affine that cut the crop places
            # them inside it.
            assert loaded.mask is not None
            crop_points = (target @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
            mask = _frame_mask(
                loaded.mask, crop_points, jaw_shrink=_mask_jaw_shrink(loaded.options)
            )
            masked = now()

            loaded.last = (face, to_frame, mask)
            self.last_timings = {
                "crop_ms": (cropped - started) * 1000.0,
                "model_ms": (rendered - cropped) * 1000.0,
                "mask_ms": (masked - rendered) * 1000.0,
            }

        pasted = now()
        output = self._paste(scene, face, to_frame, loaded, mask)
        self.last_timings["paste_ms"] = (now() - pasted) * 1000.0

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
        yaw = pose.yaw if pose is not None else None
        yaw_change = 0.0 if yaw is None or loaded.yaw_prev is None else yaw - loaded.yaw_prev
        act_mar, act_width = _mouth_activity(
            mar, width, loaded.mar_prev, loaded.width_prev, yaw_change
        )
        # In units of a 30-fps frame: a slower frame sees more change for
        # the same motion (see the last_dt slot).
        per_frame = _DEFAULT_FRAME_DT / max(loaded.last_dt, 1e-3)
        act_mar *= per_frame
        act_width *= per_frame
        loaded.mar_prev, loaded.width_prev, loaded.yaw_prev = mar, width, yaw
        mouth = _mouth_gate(mar, width, loaded.rest_mar, loaded.rest_mouth_width, pose)
        brow_gate = _brow_gate(brow, loaded.rest_brow, pose)
        if loaded.rest_mar is None or loaded.rest_brow is None:
            return mouth, mouth, brow_gate
        opening = _soft_gate(mar - loaded.rest_mar, _MOUTH_GATE_MAR)

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
        gaze: tuple[float, float] | None = None,
        centre_sample: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """The portrait's face with this driver's expression, 512x512 RGB.

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
            for k, y_per_unit in _BROW_LIFT_Y.items():
                loaded.brow_delta[0, k, 1] = brow_lift * y_per_unit
            # Day 11: the gaze drive, on the eye keypoints' x rows of the
            # same offset tensor, written fresh every frame (see the GAZE
            # section and _gaze_drive).
            gaze_x = self._gaze_drive(loaded, gaze, ear)
            for k, x_per_unit in _GAZE_KP_X.items():
                loaded.brow_delta[0, k, 0] = gaze_x * x_per_unit
            loaded.static_in.copy_(image)
            loaded.graph.replay()
            out = loaded.static_out[0].permute(1, 2, 0).float().clamp_(0.0, 1.0).mul_(255.0)
            face: np.ndarray = out.to(torch.uint8).cpu().numpy()
        return face

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
        hold existed."""
        gain = _gaze_gain(loaded.options)
        if gaze is None or loaded.rest_gaze is None or gain == 0.0:
            loaded.gaze_prev = 0.0
            return 0.0
        rest_ear = loaded.rest_ear
        if ear is not None and rest_ear is not None and ear < _GAZE_BLINK_HOLD * rest_ear:
            return gain * loaded.gaze_prev
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
        info = self._nets["motion"](image)
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
        if size_sample is not None:
            loaded.size_samples.append(size_sample)
        if centre_sample is not None:
            loaded.centre_samples.append(centre_sample)
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
        loaded.exp_prev.copy_(sums["exp"] / n)
        loaded.t_prev.copy_(sums["t"] / n)
        loaded.scale_prev.copy_(sums["scale"] / n)
        loaded.pose_prev.copy_(sums["pose"] / n)
        loaded.pose_d0.copy_(sums["pose"] / n)
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
            _LOG.info(
                "Avatar %s: anchor frozen over %d frames (%d rejected) -- "
                "pose %.1f/%.1f/%.1f (pitch/yaw/roll), rest EAR %.3f, rest MAR %.3f, "
                "rest mouth width %s, rest brow %s, face height %s px at distance %s",
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
                "?" if loaded.face_height_rest is None else f"{loaded.face_height_rest:.0f}",
                "?" if loaded.z_rest is None else f"{loaded.z_rest:.2f}",
            )

    def _drive(self, loaded: _Portrait, image: Any) -> Any:
        """Motion in, the portrait's face out. Tensor arithmetic only, so it
        captures into a graph; every reference it reads is a static tensor."""
        torch = self._torch
        info = self._nets["motion"](image)
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
        # Day 9, later: the extractor's scale and translation are smoothed
        # harder than its expression and pose (size_smoothing, default
        # 0.85 -- a fifth of each frame's fresh reading), the same _ema_
        # in the same state. Sitting still, the extractor's scale reads
        # 0.1%/1.1% (p50/p90) differently frame to frame -- a face that
        # breathes in size for no reason ("bounce"); neither should change
        # quickly in a crop that already follows him.
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
        delta = loaded.exp_s + (exp_now - loaded.exp_d0) * loaded.rel_mask * loaded.gate
        scale_new = loaded.scale_s * (scale_now / loaded.scale_d0)
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
        if _rotation_pivot(loaded.options) == "none":
            x_c_rotated = loaded.x_c_s @ (rel_rot @ loaded.R_s)
        else:
            rel_pitch = pitch_s - loaded.pose_d0[0]
            zero_deg = torch.zeros_like(rel_pitch)
            rot_p = get_rotation_matrix(rel_pitch, zero_deg, zero_deg)  # type: ignore[no-untyped-call]
            eye3 = torch.eye(3, dtype=rot_p.dtype, device=rot_p.device)[None]
            t_p = loaded.pivot @ (eye3 - rot_p)
            if not _pivot_z(loaded.options):
                t_p = torch.cat([t_p[..., :2], torch.zeros_like(t_p[..., 2:])], dim=-1)
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
        face: np.ndarray,
        to_frame: np.ndarray,
        loaded: _Portrait,
        mask: np.ndarray,
    ) -> np.ndarray:
        """The generated face into the frame, under its own oval, colour-matched.

        ``mask`` is this frame's own paste mask -- the portrait's oval
        intersected with the driver's, see :func:`_frame_mask` -- not
        ``loaded.mask``, which is the portrait's half alone and no longer
        enough on its own (see that function's docstring for why).

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
        import cv2

        assert loaded.mask_bbox is not None
        height, width = scene.shape[:2]
        bx0, by0, bx1, by1 = loaded.mask_bbox
        x0, y0, x1, y1 = _transformed_box(to_frame, loaded.mask_bbox, width, height)
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

        placed = cv2.warpAffine(face_patch, shifted, size, flags=cv2.INTER_LINEAR).astype(
            np.float32
        )
        mask = cv2.warpAffine(mask_patch, shifted, size, flags=cv2.INTER_LINEAR)[:, :, None]

        region = output[y0:y1, x0:x1].astype(np.float32)
        alpha3 = cv2.merge((mask[:, :, 0], mask[:, :, 0], mask[:, :, 0]))
        correction = _face_colour_correction(loaded, placed, region, mask, alpha3)

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

        def build(net: Any, path: Path, key: str | None = None) -> Any:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if key is not None:
                state = state[key]
            state = {name.replace("module.", ""): value for name, value in state.items()}
            net.load_state_dict(state)
            return net.to(self._device).half().eval()

        return {
            "appearance": build(
                AppearanceFeatureExtractor(**_APPEARANCE_PARAMS),  # type: ignore[no-untyped-call]
                files["appearance"],
            ),
            "motion": build(
                MotionExtractor(**_MOTION_PARAMS),  # type: ignore[no-untyped-call]
                files["motion"],
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
