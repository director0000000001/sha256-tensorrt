"""Face tracking via MediaPipe Face Landmarker.

Adapts Google's Face Landmarker task into the :class:`~sha256.core.interfaces.FaceTracker`
contract. Nothing above this module knows MediaPipe exists: the vendor result is
translated into :class:`~sha256.core.types.FaceTrackingResult` here, at the edge.

Why this backend
----------------
It is the only mainstream open-source tracker that emits **blendshapes**
directly, which is what an expression-driven avatar engine consumes. Apache-2.0
covers both the code and the weights. See ``docs/OPEN_SOURCE_EVALUATION.md``.

Measured behaviour on the reference machine (B003)
--------------------------------------------------
*   **CPU only.** The Windows Python wheel is built without GPU support — asking
    for the GPU delegate fails at graph construction with "GPU processing is
    disabled in build flags". :meth:`start` reports that plainly instead of
    pretending, and ``delegate="auto"`` therefore resolves to CPU on Windows.
*   Building the landmarker takes seconds, which is why it happens in
    :meth:`start` and never on the frame path.

Conventions this adapter is responsible for
-------------------------------------------
*   MediaPipe landmarks are already normalised to ``[0, 1]``, matching
    :class:`~sha256.core.types.LandmarkSet`. Its ``z`` is source-relative with
    negative toward the camera — also what the contract asks for.
*   Head pose comes from the facial transformation matrix, re-expressed in the
    project's documented pitch/yaw/roll convention (see :class:`HeadPose`). The
    signs were verified against landmark geometry on real frames rather than
    assumed; see ``_head_pose_from_matrix``.
*   MediaPipe emits **52 blendshape scores**, of which the first is ``_neutral``
    and the other 51 use ARKit names. That is *not* the ARKit 52 (ARKit's
    ``tongueOut`` has no MediaPipe equivalent), so the scheme is reported as
    ``mediapipe_52``. A consumer that wants ARKit semantics must remap by name.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from sha256.core.enums import PixelFormat
from sha256.core.errors import TrackingError
from sha256.core.types import (
    BlendshapeSet,
    FaceTrackingResult,
    FrameMetadata,
    HeadPose,
    LandmarkSet,
    VideoFrame,
)
from sha256.utils.logging import get_logger
from sha256.utils.timing import now

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["MediaPipeFaceTracker"]

_LOG = get_logger(__name__)

LANDMARK_SCHEME = "mediapipe_478"
BLENDSHAPE_SCHEME = "mediapipe_52"

DEFAULT_MODEL_RELATIVE_PATH = Path("downloads") / "face_landmarker.task"

# The lips, in MediaPipe's own numbering: the outer ring and the inner one.
#
# **They are filtered separately from the rest of the face, and the reason is
# measured.** The owner, 2026-09-03: *"at fast speech the photographs cannot
# follow the mouth; when I speak slowly they roughly match."* That is the
# signature of a low-pass filter, which is speed-selective by construction, and
# the numbers say the same. On the speech window of `recordings/driving.mkv`,
# the same 320 frames through this tracker twice, filtered at 0.35 and not
# filtered at all: on the 17% of frames where his mouth moves 0.10 of the speech
# axis or more between frames, the filtered reading sits **0.090 away at p50,
# 0.231 at p90 and 0.309 at worst** — six to nine cells of the speech grid — and
# it is one frame late. The engine was measured the same evening and is not
# late at all: the rung it chooses is nearest the mouth it was handed, same
# frame, offset zero on three of his captures.
#
# **What the filter is actually for is somewhere else.** It exists for landmark
# jitter around the mask edge and near the ladders' feet, which is a property of
# a face at rest, and the lips are not what it steadies. Measured on 400 quiet
# frames as second differences in pixels — a slow real movement does not count,
# only the frame-to-frame rattle — leaving the lips unfiltered moves the whole
# face's shake from 0.045 to 0.046 at p50 and 1.095 to 1.118 at p99, while the
# mouth's lag goes to **zero**. The rattle it adds is on the lip contour alone
# (1.036 to 1.348 px at p99), and the mouth region is drawn from a photograph
# rather than from that contour.
#
# The face-height reference the parting is measured against — the forehead and
# chin landmarks — stays filtered on purpose: it is a denominator, and a steady
# one is worth more than a fast one.
#
# The same two rings the renderer knows as `_OUTER_LIPS` and `_INNER_LIPS`. They
# are a fact about the 478-point mesh rather than about the avatar, which is why
# they are stated here rather than imported across the layer.
MOUTH_LANDMARKS: tuple[int, ...] = (
    # outer ring
    61, 91, 84, 17, 314, 321, 291, 270, 267, 0, 37, 40,
    # inner ring
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
)  # fmt: skip

Delegate = Literal["cpu", "gpu", "auto"]

# ``"video"`` is the frame path. ``"image"`` is for still photographs — the
# avatar's own library at load — and it is not a convenience. VIDEO mode carries
# state from one frame into the next: measured 2026-09-04, one still frame
# handed to it ten times drifts 0.8 px mean and 5.7 px at worst, and a frame
# read after a different one lands 3.3 px from where it reads on its own. So a
# landmarker in VIDEO mode cannot be shared between photographs, and one in
# IMAGE mode can: each is read as if it were the first frame ever seen, which is
# exactly what a fresh VIDEO landmarker gave each photograph before, to 0.000 px.
RunningMode = Literal["video", "image"]

# Day 19 (the client's "the face is lost when the head turns"): the second
# look in a crop. The VIDEO landmarker tracks a face it has from frame to
# frame, and once it has let go it must DETECT again -- and its detector, a
# short-range one that sees the whole 720p frame scaled to 128 pixels, finds
# a face turned well past 30 degrees only when the face is large and near
# the front. So a head that went to 60 degrees and came back is found again
# only near 30 (his live look of 2026-09-15: 143 of 300 frames with no face,
# the face back only "coming round"). On a frame the VIDEO landmarker finds
# nothing, an IMAGE-mode landmarker reads a crop around the last face --
# _REDETECT_SCALE times the face's height, the frame's own aspect so the
# task's camera model still holds -- where the face fills the picture and
# the detector has its chance. Wider after _REDETECT_WIDEN_FRAMES lost
# frames (a head that moved). The points and the pose are mapped back into
# the frame (see _crop_to_frame_points / _pose_from_crop). Frames the VIDEO
# landmarker does see are untouched: the crop is only ever read where there
# would otherwise be no face at all.
_REDETECT_SCALE = 1.8
_REDETECT_SCALE_WIDE = 2.6
_REDETECT_WIDEN_FRAMES = 30
# The task's own perspective camera: a vertical field of view of 63 degrees,
# the principal point at the picture's centre (mediapipe face_geometry). Its
# focal length in pixels is FOCAL_PER_HEIGHT times the picture's height. A
# crop puts its own centre where the face is, so the same pixels are read
# along a different ray. A pure perspective fit would shift the turn by the
# whole angle between the two rays; MEASURED 2026-09-17 on the owner's clip
# (scripts/probes/redetect_probe.py, the same frame read again moved 160
# and 320 px sideways and 120 px up and down): the fitted yaw moves 0.23 of
# that angle at the median (0.20 on a frontal face, up to 0.69 at 50
# degrees of yaw), the pitch 0.27 -- the task's solver is only weakly
# perspective. So a quarter of the ray angle is undone, not all of it: at a
# face 320 px off the frame's centre the fit is off by 7-20 degrees raw and
# 0-13 after, and a face within 160 px of the centre is within 3 degrees
# either way. Depth scales nearly with the picture the face
# is measured against (7.6% off raw at 50 degrees, 1.7% after; see
# _DEPTH_EXPONENT).
_FOCAL_PER_HEIGHT = 0.5 / math.tan(math.radians(63.0 / 2.0))
_RAY_SHARE = 0.25
# Depth, by the same measurement: the crop reads the face nearer by the
# picture scale to the power _DEPTH_EXPONENT, not the scale itself (a face
# read at 2.09 times its size came out 2.01 times nearer, at 1.03 times
# 1.03; the exponent fits both within 2%).
_DEPTH_EXPONENT = 0.92
FACE_MESH_POINTS = 468  # the mesh proper; 469-477 are the irises


class MediaPipeFaceTracker:
    """MediaPipe Face Landmarker behind the :class:`FaceTracker` protocol."""

    def __init__(
        self,
        *,
        model_path: Path | str,
        delegate: Delegate = "auto",
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        output_blendshapes: bool = True,
        smoothing: float = 0.0,
        mouth_smoothing: float = 0.0,
        running_mode: RunningMode = "video",
        redetect_in_crop: bool = False,
        redetect_confidence: float = 0.3,
    ) -> None:
        if not 0.0 <= smoothing < 1.0:
            msg = f"smoothing must be in [0, 1), got {smoothing}"
            raise ValueError(msg)
        if not 0.0 <= mouth_smoothing < 1.0:
            msg = f"mouth_smoothing must be in [0, 1), got {mouth_smoothing}"
            raise ValueError(msg)
        if not 0.0 <= redetect_confidence <= 1.0:
            msg = f"redetect_confidence must be in [0, 1], got {redetect_confidence}"
            raise ValueError(msg)

        self._model_path = Path(model_path)
        self._delegate = delegate
        self._num_faces = num_faces
        self._min_detection_confidence = min_detection_confidence
        self._min_presence_confidence = min_presence_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._output_blendshapes = output_blendshapes
        self._smoothing = smoothing
        self._mouth_smoothing = mouth_smoothing
        self._running_mode = running_mode
        self._mouth_rows = np.asarray(MOUTH_LANDMARKS, dtype=np.intp)

        # Day 19: the second look in a crop around the last face (see
        # _REDETECT_SCALE). Only on the frame path: a still has no last face.
        self._redetect_wanted = bool(redetect_in_crop) and running_mode == "video"
        self._redetect_confidence = float(redetect_confidence)
        self._crop_landmarker: Any = None
        # The last face's centre and height in the frame's pixels, and how
        # many frames in a row the VIDEO landmarker has found nothing.
        self._last_face: tuple[float, float, float] | None = None
        self._lost_run = 0

        self._landmarker: Any = None
        self._running = False
        self._active_delegate = ""

        # The mediapipe module, held after start() so the frame path never does
        # an import lookup to build an Image. ``_cv2`` is the optional fast path
        # for colour conversion; None means the numpy fallback.
        self._mp: Any = None
        self._cv2: Any = None
        self._cv2_conversions: dict[PixelFormat, int] = {}

        # detect_for_video demands strictly increasing timestamps for the life of
        # the landmarker. Frame timestamps are monotonic but two frames can land
        # in the same millisecond, so the last value issued is tracked and
        # nudged forward rather than trusted.
        self._last_timestamp_ms = -1

        # Blendshape names are identical on every frame. Built once and shared
        # by reference, so a per-frame result costs one array, not 52 strings.
        self._blendshape_names: tuple[str, ...] = ()

        # Smoothing state, only used when smoothing > 0.
        self._smoothed_points: NDArray[np.float32] | None = None
        self._smoothed_weights: NDArray[np.float32] | None = None
        self._smoothed_pose: tuple[float, float, float] | None = None

    # --- Component ---------------------------------------------------------

    @property
    def backend_id(self) -> str:
        return "mediapipe"

    @property
    def landmark_scheme(self) -> str:
        return LANDMARK_SCHEME

    @property
    def provides_blendshapes(self) -> bool:
        return self._output_blendshapes

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def active_delegate(self) -> str:
        """Delegate actually in use once started: ``"cpu"`` or ``"gpu"``."""
        return self._active_delegate

    def start(self) -> None:
        if self._running:
            return

        if not self._model_path.is_file():
            msg = (
                f"MediaPipe model bundle not found at {self._model_path}. "
                f"Fetch it once with: python scripts/fetch_face_landmarker.py"
            )
            raise TrackingError(msg)

        mp, vision, base_options_cls = _import_mediapipe()
        delegate_enum = base_options_cls.Delegate

        requested = self._delegate
        # "auto" means CPU on Windows: the published wheel has no GPU delegate
        # compiled in, so trying GPU first would only cost a failed graph build.
        candidates: tuple[str, ...] = ("cpu",) if requested == "auto" else (requested,)

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                self._landmarker = self._create_landmarker(
                    vision, base_options_cls, delegate_enum, candidate
                )
            except Exception as exc:
                last_error = exc
                _LOG.warning("MediaPipe %s delegate unavailable: %s", candidate, exc)
                continue
            self._active_delegate = candidate
            break
        else:
            msg = (
                f"could not create the MediaPipe Face Landmarker with delegate "
                f"{requested!r}: {last_error}"
            )
            raise TrackingError(msg) from last_error

        if self._redetect_wanted:
            # Built here, not on the first lost frame: creating a task takes
            # a good fraction of a second, and the frame path must not stall
            # the moment the face is lost.
            self._crop_landmarker = self._create_landmarker(
                vision,
                base_options_cls,
                delegate_enum,
                self._active_delegate,
                running_mode="image",
                detection_confidence=self._redetect_confidence,
            )

        self._mp = mp
        self._cv2, self._cv2_conversions = _try_import_cv2()
        self._running = True
        self._last_timestamp_ms = -1
        self._last_face = None
        self._lost_run = 0
        self._clear_smoothing()

        _LOG.info(
            "MediaPipe Face Landmarker ready: %s delegate, %d face(s), model %s",
            self._active_delegate,
            self._num_faces,
            self._model_path.name,
        )

    def _create_landmarker(
        self,
        vision: Any,
        base_options_cls: Any,
        delegate_enum: Any,
        delegate: str,
        *,
        running_mode: RunningMode | None = None,
        detection_confidence: float | None = None,
    ) -> Any:
        mode = self._running_mode if running_mode is None else running_mode
        detection = (
            self._min_detection_confidence if detection_confidence is None else detection_confidence
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options_cls(
                model_asset_path=str(self._model_path),
                delegate=delegate_enum.GPU if delegate == "gpu" else delegate_enum.CPU,
            ),
            # VIDEO rather than IMAGE on the frame path: the task then carries
            # tracking state between frames instead of re-detecting from scratch
            # every time. IMAGE for stills — see :data:`RunningMode` — and for
            # the second look in a crop (day 19), which is a still each time.
            # LIVE_STREAM would deliver results through a callback, which does
            # not fit a synchronous `process()` and buys nothing here.
            running_mode=(
                vision.RunningMode.IMAGE if mode == "image" else vision.RunningMode.VIDEO
            ),
            num_faces=self._num_faces,
            min_face_detection_confidence=detection,
            min_face_presence_confidence=self._min_presence_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
            output_face_blendshapes=self._output_blendshapes,
            # Always on: it is the only source of head pose, and measurement
            # shows it costs nothing worth saving (B003).
            output_facial_transformation_matrixes=True,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def stop(self) -> None:
        landmarker, self._landmarker = self._landmarker, None
        crop_landmarker, self._crop_landmarker = self._crop_landmarker, None
        self._running = False
        self._active_delegate = ""
        self._mp = None
        self._cv2 = None
        self._cv2_conversions = {}
        self._last_face = None
        self._lost_run = 0
        self._clear_smoothing()
        for task in (landmarker, crop_landmarker):
            if task is not None:
                try:
                    task.close()
                except Exception:
                    _LOG.exception("Error closing the MediaPipe landmarker")

    def reset(self) -> None:
        """Drop this adapter's temporal state.

        MediaPipe's own inter-frame priors live inside the task graph and cannot
        be cleared without rebuilding it, which takes seconds. Rebuilding here
        would turn a cheap call into a stall, so ``reset`` clears the smoothing
        filters this adapter owns and leaves the graph intact. A genuine cold
        start is ``stop()`` followed by ``start()``.
        """
        self._clear_smoothing()

    # --- FaceTracker -------------------------------------------------------

    def process(self, frame: VideoFrame) -> FaceTrackingResult:
        started = now()

        if self._landmarker is None:
            return FaceTrackingResult.no_face(frame.metadata, tracker_id=self.backend_id)

        rgb = self._frame_rgb(frame)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        try:
            if self._running_mode == "image":
                # No timestamp and no state: every still is the first one.
                result = self._landmarker.detect(image)
            else:
                timestamp_ms = self._next_timestamp_ms(frame.metadata)
                result = self._landmarker.detect_for_video(image, timestamp_ms)
        except Exception as exc:
            msg = f"MediaPipe failed on frame {frame.metadata.frame_id}: {exc}"
            raise TrackingError(msg) from exc

        if not result.face_landmarks:
            # Day 19: the second look, in a crop around the last face.
            found = self._redetect(rgb, frame.metadata, started) if self._crop_landmarker else None
            self._lost_run += 1
            if found is not None:
                return found
            # Losing the face is an expected condition, not an error: the engine
            # is responsible for holding or easing to neutral.
            self._clear_smoothing()
            return FaceTrackingResult(
                metadata=frame.metadata,
                face_detected=False,
                tracker_id=self.backend_id,
                processing_time=now() - started,
            )

        self._lost_run = 0
        adapted = self._adapt(result, frame.metadata, started)
        self._remember_face(adapted, frame.width, frame.height)
        return adapted

    def _redetect(
        self, rgb: np.ndarray, metadata: FrameMetadata, started: float
    ) -> FaceTrackingResult | None:
        """The second look (day 19): the IMAGE landmarker on a crop around
        the last face, its result mapped back into the frame. ``None`` when
        there is no last face, the crop would be the whole frame (the VIDEO
        landmarker's own try) or the crop holds no face either."""
        if self._last_face is None:
            return None
        height, width = rgb.shape[:2]
        scale = (
            _REDETECT_SCALE_WIDE if self._lost_run >= _REDETECT_WIDEN_FRAMES else _REDETECT_SCALE
        )
        window = _redetect_window(self._last_face, scale, width, height)
        if window is None:
            return None
        x0, y0, w, h = window
        patch = np.ascontiguousarray(rgb[y0 : y0 + h, x0 : x0 + w])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=patch)
        try:
            result = self._crop_landmarker.detect(image)
        except Exception as exc:
            msg = f"MediaPipe failed on the crop of frame {metadata.frame_id}: {exc}"
            raise TrackingError(msg) from exc
        if not result.face_landmarks:
            return None
        adapted = self._adapt(result, metadata, started, crop=(x0, y0, w, h, width, height))
        self._remember_face(adapted, width, height)
        return adapted

    def _remember_face(self, result: FaceTrackingResult, width: int, height: int) -> None:
        """Where the face is, for the next second look: its centre and its
        height in the frame's pixels (the height, not the width -- a turn
        narrows the face, it does not shorten it)."""
        if result.landmarks is None:
            return
        points = result.landmarks.points[:FACE_MESH_POINTS, :2]
        xs = points[:, 0] * width
        ys = points[:, 1] * height
        self._last_face = (
            float((xs.min() + xs.max()) / 2.0),
            float((ys.min() + ys.max()) / 2.0),
            float(ys.max() - ys.min()),
        )

    # --- conversion --------------------------------------------------------

    def _to_mediapipe_image(self, frame: VideoFrame) -> Any:
        """Wrap the frame as an SRGB ``mp.Image`` -- see :meth:`_frame_rgb`."""
        mp = self._mp
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=self._frame_rgb(frame))

    def _frame_rgb(self, frame: VideoFrame) -> np.ndarray:
        """The frame as contiguous RGB, what an SRGB ``mp.Image`` wraps.

        MediaPipe wants contiguous RGB. The camera delivers BGR (OpenCV's native
        layout, kept unconverted at capture so the cost lands on whoever actually
        needs a different one — that is here).

        **OpenCV does this conversion when it is available**, and the numpy path
        is the fallback. That is not a stylistic preference: on the reference
        machine, at 1280x720, reversing the channel axis in numpy costs
        **2.90 ms** against **0.45 ms** for ``cv2.cvtColor`` (B003, 50 repeats on
        one frame). At 33 ms per frame, 2.4 ms is 7% of the whole budget spent on
        rearranging bytes. Reversing an axis in numpy yields a non-contiguous
        view, and making it contiguous again is a full-frame gather.

        The fallback stays because the tracker must not *require* OpenCV. A unit
        test asserts both paths produce identical pixels.
        """
        data = frame.data
        fmt = frame.pixel_format

        code = self._cv2_conversions.get(fmt)
        if code is not None:
            converted: np.ndarray = self._cv2.cvtColor(data, code)
            return converted

        if fmt is PixelFormat.RGB24:
            rgb = data if data.flags["C_CONTIGUOUS"] else np.ascontiguousarray(data)
        elif fmt is PixelFormat.BGR24:
            rgb = np.ascontiguousarray(data[:, :, ::-1])
        elif fmt is PixelFormat.RGBA32:
            rgb = np.ascontiguousarray(data[:, :, :3])
        elif fmt is PixelFormat.BGRA32:
            rgb = np.ascontiguousarray(data[:, :, 2::-1])
        elif fmt is PixelFormat.GRAY8:
            rgb = np.ascontiguousarray(np.repeat(data.reshape(*data.shape[:2], 1), 3, axis=2))
        else:
            msg = f"MediaPipe tracker cannot consume {fmt.value} frames"
            raise TrackingError(msg)

        return rgb

    def _next_timestamp_ms(self, metadata: FrameMetadata) -> int:
        """A strictly increasing millisecond stamp for ``detect_for_video``."""
        candidate = int(metadata.capture_timestamp * 1000.0)
        if candidate <= self._last_timestamp_ms:
            candidate = self._last_timestamp_ms + 1
        self._last_timestamp_ms = candidate
        return candidate

    def _adapt(
        self,
        result: Any,
        metadata: FrameMetadata,
        started: float,
        *,
        crop: tuple[int, int, int, int, int, int] | None = None,
    ) -> FaceTrackingResult:
        """Translate one MediaPipe result into the neutral contract type.
        ``crop`` -- ``(x0, y0, w, h, width, height)`` -- says the result was
        read off that window of a ``width x height`` frame (day 19's second
        look) and is mapped back into the frame before anything else."""
        landmarks = result.face_landmarks[0]
        points = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)

        pose = self._head_pose(result)
        if crop is not None:
            points = _crop_to_frame_points(points, crop)
            pose = None if pose is None else _pose_from_crop(pose, crop)
        blendshapes = self._blendshapes(result)

        if self._smoothing > 0.0 or self._mouth_smoothing > 0.0:
            points, pose, blendshapes = self._smooth(points, pose, blendshapes)

        extras: dict[str, float] = {}
        if crop is not None:
            extras["redetected"] = 1.0
        if blendshapes is not None and blendshapes.names:
            # The model's own "no expression" score. Cheap to carry and the only
            # graded quality-ish signal MediaPipe emits.
            extras["neutral"] = blendshapes.get("_neutral")

        return FaceTrackingResult(
            metadata=metadata,
            face_detected=True,
            landmarks=LandmarkSet(points=points, scheme=LANDMARK_SCHEME),
            head_pose=pose,
            blendshapes=blendshapes,
            # MediaPipe reports no per-face score — FaceLandmarkerResult carries
            # landmarks, blendshapes, and matrices only. Detection is therefore
            # binary: a result exists precisely because the model's internal
            # presence score cleared min_face_presence_confidence. Reporting 1.0
            # states "tracked", not "certain"; nothing should build a graded
            # quality heuristic on a number the model never produced.
            confidence=1.0,
            tracker_id=self.backend_id,
            processing_time=now() - started,
            extras=extras,
        )

    def _head_pose(self, result: Any) -> HeadPose | None:
        matrices = result.facial_transformation_matrixes
        if not matrices:
            return None
        return _head_pose_from_matrix(np.asarray(matrices[0], dtype=np.float32))

    def _blendshapes(self, result: Any) -> BlendshapeSet | None:
        categories = result.face_blendshapes
        if not categories:
            return None
        scores = categories[0]

        if len(self._blendshape_names) != len(scores):
            self._blendshape_names = tuple(category.category_name for category in scores)

        weights = np.fromiter(
            (category.score for category in scores), dtype=np.float32, count=len(scores)
        )
        return BlendshapeSet(
            names=self._blendshape_names, weights=weights, scheme=BLENDSHAPE_SCHEME
        )

    # --- smoothing ---------------------------------------------------------

    def _smooth(
        self,
        points: NDArray[np.float32],
        pose: HeadPose | None,
        blendshapes: BlendshapeSet | None,
    ) -> tuple[NDArray[np.float32], HeadPose | None, BlendshapeSet | None]:
        """Exponential moving average over landmarks, pose, and expression.

        Steadier output at the cost of latency: a factor of ``a`` means the
        output is still ``a`` of the way toward the previous frame's value. It is
        off by default, and the trade-off has to be measured rather than tuned by
        eye — see ``docs/benchmarks/PERFORMANCE_PLAN.md``.

        **The lips have their own factor**, because the two halves of the
        trade-off live on different landmarks: the shimmer this filter exists
        for is the mask edge and the ladders near rest, and the speed it costs
        is spent on the mouth. See :data:`MOUTH_LANDMARKS` for the measurement
        that separated them. With one factor for both the behaviour is exactly
        what it was.
        """
        alpha = self._smoothing
        mouth_alpha = self._mouth_smoothing

        if self._smoothed_points is not None and self._smoothed_points.shape == points.shape:
            previous = self._smoothed_points
            blended = alpha * previous + (1.0 - alpha) * points
            if mouth_alpha != alpha:
                rows = self._mouth_rows
                blended[rows] = mouth_alpha * previous[rows] + (1.0 - mouth_alpha) * points[rows]
            points = blended
        self._smoothed_points = points

        if pose is not None:
            current = (pose.pitch, pose.yaw, pose.roll)
            if self._smoothed_pose is not None:
                current = tuple(  # type: ignore[assignment]
                    alpha * previous + (1.0 - alpha) * value
                    for previous, value in zip(self._smoothed_pose, current, strict=True)
                )
            self._smoothed_pose = current
            pose = HeadPose(
                pitch=current[0],
                yaw=current[1],
                roll=current[2],
                translation=pose.translation,
                confidence=pose.confidence,
            )

        if blendshapes is not None:
            weights = blendshapes.weights
            if self._smoothed_weights is not None and self._smoothed_weights.shape == weights.shape:
                weights = alpha * self._smoothed_weights + (1.0 - alpha) * weights
            self._smoothed_weights = weights
            blendshapes = BlendshapeSet(
                names=blendshapes.names, weights=weights, scheme=blendshapes.scheme
            )

        return points, pose, blendshapes

    def _clear_smoothing(self) -> None:
        self._smoothed_points = None
        self._smoothed_weights = None
        self._smoothed_pose = None


# ---------------------------------------------------------------------------
# Pure helpers — no MediaPipe import, so they are testable without the wheel
# ---------------------------------------------------------------------------


def _head_pose_from_matrix(matrix: NDArray[np.float32]) -> HeadPose:
    """Convert MediaPipe's 4x4 facial transformation matrix into a HeadPose.

    MediaPipe returns the transform that places the canonical face model into a
    camera-anchored space: the upper-left 3x3 is rotation (possibly with scale),
    and the last column is translation in centimetre-scale model units.

    That space was **measured**, not assumed, to be right-handed with **X to the
    image right, Y up, and Z toward the viewer** (B003): rotating a real frame
    counter-clockwise by a known angle moves the extracted Z rotation by exactly
    that angle, which is only true in that frame.

    The angles come out as a ``Rz * Ry * Rx`` decomposition and are then
    re-signed into the convention on :class:`~sha256.core.types.HeadPose` —
    pitch positive looking *up*, yaw positive turning to the subject's *left*,
    roll positive tilting to the subject's *right*. In the frame above:

    * a positive rotation about +X turns the face's forward direction downward,
      so **pitch is negated**;
    * a positive rotation about +Y turns it toward the image right, which is the
      subject's left, so **yaw is taken as is**;
    * a positive rotation about +Z tips the top of the head toward the image
      left, which is the subject's right, so **roll is taken as is**.

    Two independent measurements confirm this, both in
    ``docs/benchmarks/B003-face-tracking.md``: applying known image rotations and
    mirroring to a real frame, and comparing every reported angle against the
    face normal computed from the landmarks — a separate model output, so the
    agreement is evidence rather than a tautology.
    """
    rotation = matrix[:3, :3].astype(np.float64)

    # Divide out scale so the arctangents see a pure rotation. A degenerate
    # matrix (all-zero, as a stub or a failed frame) yields a neutral pose
    # rather than a NaN that would poison everything downstream.
    scale = np.linalg.norm(rotation, axis=0)
    if not np.all(np.isfinite(scale)) or np.any(scale < 1e-8):
        return HeadPose()
    rotation = rotation / scale

    sy = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    if sy < 1e-6:
        # Gimbal lock: yaw near +-90 degrees leaves roll and pitch degenerate.
        x_angle = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        y_angle = math.atan2(-float(rotation[2, 0]), sy)
        z_angle = 0.0
    else:
        x_angle = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        y_angle = math.atan2(-float(rotation[2, 0]), sy)
        z_angle = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))

    translation = matrix[:3, 3]

    return HeadPose(
        pitch=-math.degrees(x_angle),
        yaw=math.degrees(y_angle),
        roll=math.degrees(z_angle),
        translation=(
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
        ),
        confidence=1.0,
    )


def _try_import_cv2() -> tuple[Any, dict[PixelFormat, int]]:
    """OpenCV and its colour-conversion codes, or ``(None, {})`` if absent.

    Optional on purpose. OpenCV is five times faster at this than numpy (B003),
    but making it mandatory would put a heavyweight dependency behind a tracker
    that otherwise needs only numpy — and the pipeline can already feed this
    tracker RGB frames that need no conversion at all.

    ``RGB24`` is deliberately absent from the map: it needs no conversion, so it
    falls through to the zero-copy path.
    """
    try:
        import cv2
    except ImportError:
        return None, {}

    return cv2, {
        PixelFormat.BGR24: cv2.COLOR_BGR2RGB,
        PixelFormat.BGRA32: cv2.COLOR_BGRA2RGB,
        PixelFormat.RGBA32: cv2.COLOR_RGBA2RGB,
        PixelFormat.GRAY8: cv2.COLOR_GRAY2RGB,
    }


def _import_mediapipe() -> tuple[Any, Any, Any]:
    """Import MediaPipe, turning an absent wheel into a project error.

    Imported here rather than at module scope so that the import cost (about
    0.3 s, measured in B003) is paid at ``start()`` by whoever selected this
    backend, and so the failure message names the extra to install.

    Returns ``(mediapipe, vision_tasks, BaseOptions)``.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
    except ImportError as exc:
        msg = (
            "the MediaPipe tracker requires the 'tracking' extra. "
            "Install it with: uv pip install mediapipe --no-deps "
            "(plus absl-py, certifi, matplotlib)"
        )
        raise TrackingError(msg) from exc

    return mp, vision, BaseOptions


# ---------------------------------------------------------------------------
# Day 19: the second look in a crop -- pure helpers, testable without the wheel
# ---------------------------------------------------------------------------


def _redetect_window(
    last_face: tuple[float, float, float], scale: float, width: int, height: int
) -> tuple[int, int, int, int] | None:
    """The crop to look for the face in: ``scale`` times the last face's
    height tall, the frame's own aspect ratio, centred on the last face and
    shifted (never shrunk) to lie inside the frame. ``(x0, y0, w, h)`` in
    pixels, or ``None`` when the crop would be the whole frame or more --
    that look is the VIDEO landmarker's own -- or the last face is too small
    to mean anything."""
    cx, cy, face_height = last_face
    if face_height < 16.0 or width <= 0 or height <= 0:
        return None
    h = scale * face_height
    w = h * width / height
    if h >= height or w >= width:
        return None
    x0 = min(max(cx - w / 2.0, 0.0), width - w)
    y0 = min(max(cy - h / 2.0, 0.0), height - h)
    return round(x0), round(y0), round(w), round(h)


def _crop_to_frame_points(
    points: NDArray[np.float32], crop: tuple[int, int, int, int, int, int]
) -> NDArray[np.float32]:
    """Landmarks read off a crop, as the frame's own normalised landmarks.
    ``z`` is scaled like ``x`` (MediaPipe's z shares x's scale)."""
    x0, y0, w, h, width, height = crop
    out = points.copy()
    out[:, 0] = (x0 + points[:, 0] * w) / width
    out[:, 1] = (y0 + points[:, 1] * h) / height
    out[:, 2] = points[:, 2] * (w / width)
    return out


def _pose_from_crop(pose: HeadPose, crop: tuple[int, int, int, int, int, int]) -> HeadPose:
    """The head pose fitted on a crop, as the frame's own.

    The task fits its face model through a perspective camera whose axis
    runs through the picture's centre (:data:`_FOCAL_PER_HEIGHT`). The crop's
    centre is on the face, the frame's is not, so the crop's fit explains the
    same pixels along a different ray and its turn is off by a share of the
    angle between the rays (:data:`_RAY_SHARE`, measured): a face to the
    right of the frame's centre that looks at the lens is turned toward the
    frame's centre (a negative yaw, the subject's right) in the frame and
    straight ahead in the crop. Depth scales with the picture the face is
    measured against: the face fills ``height / h`` more of the crop, so it
    is read that much nearer.
    """
    x0, y0, w, h, width, height = crop
    focal = _FOCAL_PER_HEIGHT * height
    # The crop's centre, in the frame's pixels, against the frame's centre.
    dx = (x0 + w / 2.0) - width / 2.0
    dy = (y0 + h / 2.0) - height / 2.0
    # The crop's fit sees the face on its axis; it is not. Yaw: positive to
    # the subject's left = toward the image right, so a face right of centre
    # looking at the lens has a negative yaw in the frame. Pitch: positive
    # looks up, so a face below the centre looking at the lens looks up.
    yaw_shift = _RAY_SHARE * math.degrees(math.atan2(dx, focal))
    pitch_shift = _RAY_SHARE * math.degrees(math.atan2(dy, focal))
    depth_scale = (height / h) ** _DEPTH_EXPONENT
    tx, ty, tz = pose.translation
    return HeadPose(
        pitch=pose.pitch + pitch_shift,
        yaw=pose.yaw - yaw_shift,
        roll=pose.roll,
        translation=(tx * depth_scale, ty * depth_scale, tz * depth_scale),
        confidence=pose.confidence,
    )
