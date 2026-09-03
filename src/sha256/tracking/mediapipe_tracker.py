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
    ) -> None:
        if not 0.0 <= smoothing < 1.0:
            msg = f"smoothing must be in [0, 1), got {smoothing}"
            raise ValueError(msg)
        if not 0.0 <= mouth_smoothing < 1.0:
            msg = f"mouth_smoothing must be in [0, 1), got {mouth_smoothing}"
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
        self._mouth_rows = np.asarray(MOUTH_LANDMARKS, dtype=np.intp)

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

        self._mp = mp
        self._cv2, self._cv2_conversions = _try_import_cv2()
        self._running = True
        self._last_timestamp_ms = -1
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
    ) -> Any:
        options = vision.FaceLandmarkerOptions(
            base_options=base_options_cls(
                model_asset_path=str(self._model_path),
                delegate=delegate_enum.GPU if delegate == "gpu" else delegate_enum.CPU,
            ),
            # VIDEO rather than IMAGE: the task then carries tracking state
            # between frames instead of re-detecting from scratch every time.
            # LIVE_STREAM would deliver results through a callback, which does
            # not fit a synchronous `process()` and buys nothing here.
            running_mode=vision.RunningMode.VIDEO,
            num_faces=self._num_faces,
            min_face_detection_confidence=self._min_detection_confidence,
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
        self._running = False
        self._active_delegate = ""
        self._mp = None
        self._cv2 = None
        self._cv2_conversions = {}
        self._clear_smoothing()
        if landmarker is not None:
            try:
                landmarker.close()
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

        image = self._to_mediapipe_image(frame)
        timestamp_ms = self._next_timestamp_ms(frame.metadata)

        try:
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        except Exception as exc:
            msg = f"MediaPipe failed on frame {frame.metadata.frame_id}: {exc}"
            raise TrackingError(msg) from exc

        if not result.face_landmarks:
            # Losing the face is an expected condition, not an error: the engine
            # is responsible for holding or easing to neutral.
            self._clear_smoothing()
            return FaceTrackingResult(
                metadata=frame.metadata,
                face_detected=False,
                tracker_id=self.backend_id,
                processing_time=now() - started,
            )

        return self._adapt(result, frame.metadata, started)

    # --- conversion --------------------------------------------------------

    def _to_mediapipe_image(self, frame: VideoFrame) -> Any:
        """Wrap the frame as an SRGB ``mp.Image``.

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
            rgb = self._cv2.cvtColor(data, code)
            mp = self._mp
            return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

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

        mp = self._mp
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def _next_timestamp_ms(self, metadata: FrameMetadata) -> int:
        """A strictly increasing millisecond stamp for ``detect_for_video``."""
        candidate = int(metadata.capture_timestamp * 1000.0)
        if candidate <= self._last_timestamp_ms:
            candidate = self._last_timestamp_ms + 1
        self._last_timestamp_ms = candidate
        return candidate

    def _adapt(self, result: Any, metadata: FrameMetadata, started: float) -> FaceTrackingResult:
        """Translate one MediaPipe result into the neutral contract type."""
        landmarks = result.face_landmarks[0]
        points = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)

        pose = self._head_pose(result)
        blendshapes = self._blendshapes(result)

        if self._smoothing > 0.0 or self._mouth_smoothing > 0.0:
            points, pose, blendshapes = self._smooth(points, pose, blendshapes)

        extras: dict[str, float] = {}
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
