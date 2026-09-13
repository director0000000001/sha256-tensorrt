"""The avatar's hair (day 17, 2026-09-13, block C1b): a matte of the hair in
the network's own rendered crop, so the portrait's hair can be pasted over
the driver's head along with the face.

The network draws the whole 512x512 crop every frame -- the face, the hair
and the portrait's own background -- and what has hidden all but the face
until now is the paste mask, a face oval. To paste the hair the mask has to
know where the hair IS in *this frame's* render, since the hair turns with
the head; a matte of the still portrait would be a few pixels off on every
turn. So the rendered crop is segmented, every frame it can be, by
MediaPipe's multiclass selfie segmenter (``selfie_multiclass_256x256``,
Apache-2.0; classes background / hair / body skin / face skin / clothes /
accessories), and the hair class's confidence becomes an alpha the engine
takes the maximum of with its face mask.

The segmenter costs 35-65 ms a frame on this machine's CPU (measured
2026-09-13, both of MediaPipe's segmenters, image and video modes), more
than the whole frame has, so it runs on its own thread one or two frames
behind (:class:`HairMatteWorker`): the frame path submits the latest render
and takes whatever matte is newest, and an EMA over the mattes hides the
segmenter's own frame-to-frame flicker. Hair moves slowly against a 30 fps
camera; a matte one or two frames old is not seen.

What the alpha is shaped by (:func:`hair_alpha`): the confidence is
curved -- nothing under :data:`_MASK_LOW`, everything over :data:`_MASK_HIGH`
-- so the segmenter's faint, wide "maybe" around a beard or an ear does not
paste a ghost of the portrait there; feathered by a few pixels; and faded
to nothing at the crop's own border, because a hairstyle the crop cuts
(long hair past the crop's bottom, tall hair past its top) must end in a
fade and not in a straight line across the driver's picture.

Nothing here touches the GPU or the graph. The engine reads only
:func:`hair_alpha`'s output, a ``(512, 512, 1)`` float32 like its own mask.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from sha256.utils.logging import get_logger

__all__ = [
    "FACE_SKIN_CLASS",
    "HAIR_CLASS",
    "HAIR_MODEL_RELATIVE_PATH",
    "MODEL_INPUT",
    "HairMatteWorker",
    "HairSegmenter",
    "hair_alpha",
    "hair_box",
    "head_confidence",
    "skin_weight",
    "union_box",
]

_LOG = get_logger(__name__)

# Beside the face landmarker under models/ (see scripts/fetch_selfie_segmenter.py).
HAIR_MODEL_RELATIVE_PATH = Path("downloads") / "selfie_multiclass_256x256.tflite"
MODEL_INPUT = 256
HAIR_CLASS = 1
FACE_SKIN_CLASS = 3

# The confidence curve: the segmenter's output between these two reads as
# 0..1, linear. Measured on his portrait's crop (2026-09-13): the curls read
# 0.9-1.0, the background 0.0, and the beard and sideburns a wide 0.1-0.4
# smear that must not paste.
_MASK_LOW = 0.25
_MASK_HIGH = 0.75
# The curved alpha is pulled in by this much before the feather: at the
# hair's outer edge the render's own pixels are hair mixed with the
# portrait's background (white on his, grey on the woman's), and a soft
# alpha over that mix pastes a pale rim around the curls (seen on the
# first replay, 2026-09-13). Inside by three pixels the render is hair.
_ERODE_PX = 3
_FEATHER_PX = 5  # at the crop's 512, odd
_BORDER_PX = 24  # the fade at the crop's own edges
_BOX_MARGIN = 16  # around the hair's own box, crop px
_BOX_THRESHOLD = 0.02
# The render's face skin can join the paste ABOVE the eyes (see
# :func:`skin_weight`; the engine's ``hair_skin`` option, off): the idea was
# the forehead, the temples and a parting, where the driver's own hair sits
# under the avatar's hairline and shows between the hair matte and the face
# oval. Tried 2026-09-13 on his captures: the avatar's paler temple skin
# came out as a patch over his own hair, plainly worse than the hair
# showing (recordings/day17/head-row2-*.png). Left as a lever.
_SKIN_FULL_ABOVE = 0.15  # of the face height above the eye line: weight 1
_SKIN_NONE_BELOW = 0.05  # of the face height below the eye line: weight 0


class HairSegmenter:
    """MediaPipe's multiclass selfie segmenter, image mode, hair class out.

    ``segment`` returns the hair confidence as float32 ``(H, W)`` in 0..1 at
    the picture's own size (the model's 256 output resized back up). Not
    thread-safe on its own; :class:`HairMatteWorker` owns one.
    """

    def __init__(self, model_path: Path | str) -> None:
        self._model_path = Path(model_path)
        self._segmenter: Any = None
        self._mp: Any = None

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def is_running(self) -> bool:
        return self._segmenter is not None

    def start(self) -> None:
        if self._segmenter is not None:
            return
        if not self._model_path.is_file():
            msg = (
                f"the hair segmenter model is missing at {self._model_path}; fetch it once "
                "with: python scripts/fetch_selfie_segmenter.py"
            )
            raise FileNotFoundError(msg)
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)
        self._mp = mp

    def stop(self) -> None:
        if self._segmenter is not None:
            self._segmenter.close()
        self._segmenter = None

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        """The hair and the face-skin confidence of an RGB uint8 picture,
        ``(H, W, 2)`` float32 at the picture's own size -- hair first."""
        import cv2

        if self._segmenter is None:
            msg = "the hair segmenter is not started"
            raise RuntimeError(msg)
        height, width = rgb.shape[:2]
        small = rgb
        if (height, width) != (MODEL_INPUT, MODEL_INPUT):
            small = cv2.resize(rgb, (MODEL_INPUT, MODEL_INPUT), interpolation=cv2.INTER_AREA)
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(small)
        )
        result = self._segmenter.segment(image)
        masks = result.confidence_masks or []
        out = np.zeros((height, width, 2), dtype=np.float32)
        for column, index in enumerate((HAIR_CLASS, FACE_SKIN_CLASS)):
            if len(masks) <= index:
                continue
            plane = np.asarray(masks[index].numpy_view(), dtype=np.float32)
            if plane.ndim == 3:
                plane = plane[:, :, 0]
            if plane.shape != (height, width):
                plane = cv2.resize(plane, (width, height), interpolation=cv2.INTER_LINEAR)
            out[:, :, column] = plane
        return out


class HairMatteWorker:
    """One thread that keeps the newest matte of the newest picture.

    ``submit`` replaces whatever picture is waiting (a frame the segmenter
    never got to is simply skipped -- the next one is newer); ``latest``
    is the last matte made, tagged with the submitter's own tag and a
    sequence number, so a caller can tell a new matte from the one it
    already has and an avatar's matte from another's. ``segment`` is any
    callable from an RGB picture to a float32 confidence map; a failure
    inside it is logged once per streak and the picture dropped.
    """

    def __init__(self, segment: Callable[[np.ndarray], np.ndarray]) -> None:
        self._segment = segment
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._pending: tuple[Any, np.ndarray] | None = None
        self._latest: tuple[Any, int, np.ndarray] | None = None
        self._seq = 0
        self._stopping = False
        self._failed = False
        self._thread = threading.Thread(target=self._run, name="hair-matte", daemon=True)
        self._thread.start()

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def submit(self, tag: Any, rgb: np.ndarray) -> None:
        with self._wake:
            if self._stopping:
                return
            self._pending = (tag, np.array(rgb, copy=True))
            self._wake.notify()

    def latest(self) -> tuple[Any, int, np.ndarray] | None:
        with self._lock:
            return self._latest

    def stop(self, timeout: float = 2.0) -> None:
        with self._wake:
            self._stopping = True
            self._pending = None
            self._wake.notify_all()
        self._thread.join(timeout)

    def _run(self) -> None:
        while True:
            with self._wake:
                while self._pending is None and not self._stopping:
                    self._wake.wait()
                if self._stopping:
                    return
                assert self._pending is not None
                tag, rgb = self._pending
                self._pending = None
            try:
                matte = self._segment(rgb)
            except Exception:
                if not self._failed:
                    _LOG.exception("The hair segmenter failed; the last matte stays")
                self._failed = True
                continue
            self._failed = False
            with self._lock:
                self._seq += 1
                self._latest = (tag, self._seq, matte)


def skin_weight(
    crop_points: np.ndarray,
    size: int,
    *,
    full_above: float = _SKIN_FULL_ABOVE,
    none_below: float = _SKIN_NONE_BELOW,
) -> np.ndarray:
    """How much of the render's face-skin confidence joins the paste per
    row, ``(size, size)`` float32: 1 from ``full_above`` of the face
    height above the eye corners' line up, 0 from ``none_below`` below it
    down, linear between. ``crop_points`` are the portrait's 468 points in
    the crop's own pixels."""
    pts = np.asarray(crop_points, dtype=np.float32)
    eye_y = float(pts[[33, 133, 362, 263], 1].mean())
    face_h = max(float(pts[:, 1].max() - pts[:, 1].min()), 1.0)
    top = eye_y - full_above * face_h
    bottom = eye_y + none_below * face_h
    rows = np.arange(size, dtype=np.float32)
    weight = np.clip((bottom - rows) / max(bottom - top, 1.0), 0.0, 1.0).astype(np.float32)
    return np.repeat(weight[:, None], size, axis=1)


def head_confidence(confidence: np.ndarray, skin: np.ndarray | None = None) -> np.ndarray:
    """One confidence map for the paste from the segmenter's ``(H, W, 2)``
    (hair, face skin): the hair, or the face skin where ``skin`` (see
    :func:`skin_weight`) lets it in -- whichever is higher. A plain
    ``(H, W)`` map is taken as hair alone."""
    conf = np.asarray(confidence, dtype=np.float32)
    if conf.ndim == 2:
        return conf
    hair = conf[:, :, 0]
    if conf.shape[2] < 2 or skin is None:
        return np.ascontiguousarray(hair)
    head: np.ndarray = np.maximum(hair, conf[:, :, 1] * skin).astype(np.float32)
    return head


def hair_alpha(
    confidence: np.ndarray,
    *,
    low: float = _MASK_LOW,
    high: float = _MASK_HIGH,
    erode: int = _ERODE_PX,
    feather: int = _FEATHER_PX,
    border: int = _BORDER_PX,
) -> np.ndarray:
    """The paste alpha of a confidence map, ``(H, W, 1)`` float32: the
    confidence curved between ``low`` and ``high``, pulled in by
    ``erode`` px, feathered by ``feather`` px, and faded to nothing over
    the last ``border`` px at each edge of the picture (see the module
    docstring)."""
    import cv2

    conf = np.asarray(confidence, dtype=np.float32)
    if conf.ndim == 3:
        conf = conf[:, :, 0]
    height, width = conf.shape
    span = max(high - low, 1e-6)
    alpha = np.clip((conf - low) / span, 0.0, 1.0).astype(np.float32)
    if erode > 0:
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
        alpha = np.asarray(cv2.erode(alpha, element), dtype=np.float32)
    if feather > 1:
        kernel = feather | 1
        alpha = np.asarray(cv2.GaussianBlur(alpha, (kernel, kernel), 0), dtype=np.float32)
    if border > 0:
        ramp_y = np.minimum(np.arange(height, dtype=np.float32), np.arange(height)[::-1])
        ramp_x = np.minimum(np.arange(width, dtype=np.float32), np.arange(width)[::-1])
        fade = np.minimum(
            np.clip(ramp_y / float(border), 0.0, 1.0)[:, None],
            np.clip(ramp_x / float(border), 0.0, 1.0)[None, :],
        )
        alpha = (alpha * fade).astype(np.float32)
    return np.ascontiguousarray(alpha, dtype=np.float32)[:, :, None]


def hair_box(
    alpha: np.ndarray, *, threshold: float = _BOX_THRESHOLD, margin: int = _BOX_MARGIN
) -> tuple[int, int, int, int] | None:
    """Where ``alpha`` is over ``threshold``, with ``margin`` around it, as
    ``(x0, y0, x1, y1)`` in the picture's own pixels; ``None`` for no hair."""
    plane = alpha[:, :, 0] if alpha.ndim == 3 else alpha
    height, width = plane.shape
    rows = np.where((plane > threshold).any(axis=1))[0]
    if rows.size == 0:
        return None
    cols = np.where((plane > threshold).any(axis=0))[0]
    return (
        max(int(cols[0]) - margin, 0),
        max(int(rows[0]) - margin, 0),
        min(int(cols[-1]) + 1 + margin, width),
        min(int(rows[-1]) + 1 + margin, height),
    )


def union_box(
    a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None
) -> tuple[int, int, int, int] | None:
    """The box around both, or whichever is there."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])
