"""Tests for the MediaPipe face tracker adapter.

MediaPipe itself is faked. That is deliberate: what needs testing here is the
*translation* — pixel format handling, timestamp discipline, the neutral-type
conversion, and what happens when no face is found. Those are the parts that
break on a version bump, and none of them needs a real neural network to
exercise. The real model is measured separately in ``scripts/benchmark_tracking.py``.

The pose helper is tested against a real rotation matrix, because sign errors
there are the classic axis-flip bug and would otherwise only surface as an
avatar that leans the wrong way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from sha256.core.enums import PixelFormat
from sha256.core.errors import TrackingError
from sha256.core.interfaces import FaceTracker
from sha256.core.types import FrameMetadata, VideoFrame
from sha256.tracking.mediapipe_tracker import (
    BLENDSHAPE_SCHEME,
    LANDMARK_SCHEME,
    MOUTH_LANDMARKS,
    MediaPipeFaceTracker,
    _head_pose_from_matrix,
)

# --- fake MediaPipe --------------------------------------------------------


@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float


@dataclass
class FakeCategory:
    category_name: str
    score: float


@dataclass
class FakeResult:
    face_landmarks: list[list[FakeLandmark]] = field(default_factory=list)
    face_blendshapes: list[list[FakeCategory]] = field(default_factory=list)
    facial_transformation_matrixes: list[np.ndarray] = field(default_factory=list)


class FakeLandmarker:
    """Stands in for the MediaPipe task object."""

    def __init__(self, *, detects: bool = True, raises: bool = False) -> None:
        self.detects = detects
        self.raises = raises
        self.closed = False
        self.timestamps: list[int] = []
        self.images: list[object] = []
        self.stills = 0
        # Every landmark sits at this coordinate. Settable so a test can move
        # the "face" between frames and watch what the filters do.
        self.position = 0.5
        # Set by FakeFaceLandmarker.create_from_options, so a test can read
        # back what running mode (and anything else) it was built with.
        self.options: object | None = None

    def _result(self) -> FakeResult:
        if self.raises:
            msg = "inference exploded"
            raise RuntimeError(msg)
        if not self.detects:
            return FakeResult()

        landmarks = [FakeLandmark(self.position, self.position, self.position) for _ in range(478)]
        blendshapes = [
            FakeCategory("_neutral", 0.9),
            FakeCategory("jawOpen", 0.25),
            FakeCategory("eyeBlinkLeft", 0.1),
        ]
        return FakeResult(
            face_landmarks=[landmarks],
            face_blendshapes=[blendshapes],
            facial_transformation_matrixes=[np.eye(4, dtype=np.float32)],
        )

    def detect_for_video(self, image: object, timestamp_ms: int) -> FakeResult:
        self.images.append(image)
        if self.timestamps and timestamp_ms <= self.timestamps[-1]:
            # The real task raises on a non-increasing timestamp; so does this,
            # otherwise the adapter's guard would be untested.
            msg = f"timestamp {timestamp_ms} is not greater than {self.timestamps[-1]}"
            raise ValueError(msg)
        self.timestamps.append(timestamp_ms)
        return self._result()

    def detect(self, image: object) -> FakeResult:
        # IMAGE mode: no timestamp, no monotonic guard — every still is read
        # as if it were the first frame ever seen. See RunningMode in
        # sha256/tracking/mediapipe_tracker.py.
        self.images.append(image)
        self.stills += 1
        return self._result()

    def close(self) -> None:
        self.closed = True


class FakeImage:
    def __init__(self, *, image_format: object, data: np.ndarray) -> None:
        self.image_format = image_format
        self.data = data


@pytest.fixture
def fake_mediapipe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Replace the MediaPipe import with fakes. Returns a control dict."""
    control: dict[str, object] = {"detects": True, "raises": False, "gpu_available": False}
    created: list[FakeLandmarker] = []

    class FakeDelegate:
        CPU = "CPU"
        GPU = "GPU"

    class FakeBaseOptions:
        Delegate = FakeDelegate

        def __init__(self, *, model_asset_path: str, delegate: str) -> None:
            self.model_asset_path = model_asset_path
            self.delegate = delegate

    class FakeOptions:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeFaceLandmarker:
        @staticmethod
        def create_from_options(options: FakeOptions) -> FakeLandmarker:
            base = options.kwargs["base_options"]
            if base.delegate == FakeDelegate.GPU and not control["gpu_available"]:  # type: ignore[attr-defined]
                msg = "GPU processing is disabled in build flags"
                raise NotImplementedError(msg)
            landmarker = FakeLandmarker(
                detects=bool(control["detects"]), raises=bool(control["raises"])
            )
            landmarker.options = options
            created.append(landmarker)
            return landmarker

    class FakeVision:
        FaceLandmarker = FakeFaceLandmarker
        FaceLandmarkerOptions = FakeOptions

        class RunningMode:
            VIDEO = "VIDEO"
            IMAGE = "IMAGE"

    class FakeMp:
        Image = FakeImage

        class ImageFormat:
            SRGB = "SRGB"

    monkeypatch.setattr(
        "sha256.tracking.mediapipe_tracker._import_mediapipe",
        lambda: (FakeMp, FakeVision, FakeBaseOptions),
    )

    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"not a real model")

    control["created"] = created
    control["model_path"] = model
    return control


def make_tracker(control: dict[str, object], **kwargs: object) -> MediaPipeFaceTracker:
    return MediaPipeFaceTracker(model_path=control["model_path"], **kwargs)  # type: ignore[arg-type]


def make_frame(
    *,
    frame_id: int = 0,
    timestamp: float = 100.0,
    pixel_format: PixelFormat = PixelFormat.BGR24,
    width: int = 8,
    height: int = 6,
) -> VideoFrame:
    channels = pixel_format.channels
    shape = (height, width) if channels == 1 else (height, width, channels)
    data = np.arange(int(np.prod(shape)), dtype=np.uint8).reshape(shape)
    return VideoFrame(
        data=data,
        pixel_format=pixel_format,
        metadata=FrameMetadata(frame_id=frame_id, capture_timestamp=timestamp),
    )


# --- tests -----------------------------------------------------------------


class TestProtocolConformance:
    def test_satisfies_the_face_tracker_protocol(self, fake_mediapipe):
        assert isinstance(make_tracker(fake_mediapipe), FaceTracker)

    def test_declares_its_schemes(self, fake_mediapipe):
        tracker = make_tracker(fake_mediapipe)
        assert tracker.backend_id == "mediapipe"
        assert tracker.landmark_scheme == LANDMARK_SCHEME
        assert tracker.provides_blendshapes

    def test_blendshapes_are_not_claimed_when_disabled(self, fake_mediapipe):
        assert not make_tracker(fake_mediapipe, output_blendshapes=False).provides_blendshapes

    def test_rejects_an_out_of_range_smoothing_factor(self, fake_mediapipe):
        with pytest.raises(ValueError, match="smoothing"):
            make_tracker(fake_mediapipe, smoothing=1.0)


class TestStart:
    def test_starts_on_the_cpu_delegate(self, fake_mediapipe):
        tracker = make_tracker(fake_mediapipe)
        tracker.start()

        assert tracker.is_running
        assert tracker.active_delegate == "cpu"

    def test_auto_does_not_attempt_gpu_on_windows(self, fake_mediapipe):
        """The wheel has no GPU delegate; trying it would only cost a failure."""
        tracker = make_tracker(fake_mediapipe, delegate="auto")
        tracker.start()

        assert tracker.active_delegate == "cpu"

    def test_an_explicit_gpu_request_fails_loudly(self, fake_mediapipe):
        tracker = make_tracker(fake_mediapipe, delegate="gpu")

        with pytest.raises(TrackingError, match="GPU processing is disabled"):
            tracker.start()

        assert not tracker.is_running

    def test_gpu_is_used_when_it_does_work(self, fake_mediapipe):
        """Guards against hard-coding CPU: a wheel with GPU support must be used."""
        fake_mediapipe["gpu_available"] = True
        tracker = make_tracker(fake_mediapipe, delegate="gpu")
        tracker.start()

        assert tracker.active_delegate == "gpu"

    def test_a_missing_model_names_the_fetch_script(self, fake_mediapipe, tmp_path):
        tracker = MediaPipeFaceTracker(model_path=tmp_path / "absent.task")

        with pytest.raises(TrackingError, match="fetch_face_landmarker"):
            tracker.start()

    def test_starting_twice_is_idempotent(self, fake_mediapipe):
        tracker = make_tracker(fake_mediapipe)
        tracker.start()
        tracker.start()

        assert len(fake_mediapipe["created"]) == 1

    def test_stop_closes_the_landmarker(self, fake_mediapipe):
        tracker = make_tracker(fake_mediapipe)
        tracker.start()
        tracker.stop()

        assert fake_mediapipe["created"][0].closed
        assert not tracker.is_running

    def test_stopping_a_stopped_tracker_is_safe(self, fake_mediapipe):
        make_tracker(fake_mediapipe).stop()


class TestRunningMode:
    """VIDEO is the frame path; IMAGE is for stills read with no shared state
    — see :data:`sha256.tracking.mediapipe_tracker.RunningMode`."""

    def test_video_is_the_default(self, fake_mediapipe):
        tracker = make_tracker(fake_mediapipe)
        tracker.start()

        options = fake_mediapipe["created"][0].options
        assert options.kwargs["running_mode"] == "VIDEO"

    def test_image_mode_calls_detect_with_no_timestamp(self, fake_mediapipe):
        tracker = make_tracker(fake_mediapipe, running_mode="image")
        tracker.start()
        tracker.process(make_frame())

        landmarker = fake_mediapipe["created"][0]
        assert landmarker.options.kwargs["running_mode"] == "IMAGE"
        assert landmarker.stills == 1
        assert landmarker.timestamps == []

    def test_image_mode_has_no_monotonic_guard(self, fake_mediapipe):
        """Two stills at the same timestamp both detect — the guard that
        rejects a non-increasing timestamp only exists on the VIDEO path."""
        tracker = make_tracker(fake_mediapipe, running_mode="image")
        tracker.start()

        first = tracker.process(make_frame(frame_id=0, timestamp=100.0))
        second = tracker.process(make_frame(frame_id=1, timestamp=100.0))

        assert first.face_detected
        assert second.face_detected


class TestProcess:
    def test_produces_a_complete_tracking_result(self, fake_mediapipe):
        tracker = make_tracker(fake_mediapipe)
        tracker.start()

        result = tracker.process(make_frame(frame_id=7))

        assert result.face_detected
        assert result.tracker_id == "mediapipe"
        assert result.metadata.frame_id == 7
        assert result.landmarks is not None
        assert result.landmarks.count == 478
        assert result.landmarks.scheme == LANDMARK_SCHEME
        assert result.blendshapes is not None
        assert result.blendshapes.scheme == BLENDSHAPE_SCHEME
        assert result.head_pose is not None
        assert result.processing_time > 0.0

    def test_blendshape_names_and_weights_line_up(self, fake_mediapipe):
        tracker = make_tracker(fake_mediapipe)
        tracker.start()

        blendshapes = tracker.process(make_frame()).blendshapes

        assert blendshapes is not None
        assert blendshapes.get("jawOpen") == pytest.approx(0.25)
        assert blendshapes.get("_neutral") == pytest.approx(0.9)

    def test_the_neutral_score_is_carried_as_a_diagnostic(self, fake_mediapipe):
        tracker = make_tracker(fake_mediapipe)
        tracker.start()

        assert tracker.process(make_frame()).extras["neutral"] == pytest.approx(0.9)

    def test_no_face_is_a_normal_result_not_an_error(self, fake_mediapipe):
        """Losing the face happens constantly; the engine handles it."""
        fake_mediapipe["detects"] = False
        tracker = make_tracker(fake_mediapipe)
        tracker.start()

        result = tracker.process(make_frame())

        assert not result.face_detected
        assert result.landmarks is None
        assert result.tracker_id == "mediapipe"

    def test_processing_before_start_returns_an_empty_result(self, fake_mediapipe):
        result = make_tracker(fake_mediapipe).process(make_frame())

        assert not result.face_detected

    def test_a_vendor_failure_becomes_a_project_error(self, fake_mediapipe):
        """A MediaPipe exception must not escape into UI code untranslated."""
        fake_mediapipe["raises"] = True
        tracker = make_tracker(fake_mediapipe)
        tracker.start()

        with pytest.raises(TrackingError, match="frame 3"):
            tracker.process(make_frame(frame_id=3))


class TestTimestamps:
    def test_timestamps_strictly_increase(self, fake_mediapipe):
        tracker = make_tracker(fake_mediapipe)
        tracker.start()

        for index in range(5):
            tracker.process(make_frame(frame_id=index, timestamp=100.0 + index * 0.033))

        stamps = fake_mediapipe["created"][0].timestamps
        assert stamps == sorted(stamps)
        assert len(set(stamps)) == len(stamps)

    def test_frames_inside_one_millisecond_are_nudged_apart(self, fake_mediapipe):
        """Two frames can share a millisecond; MediaPipe rejects a repeat."""
        tracker = make_tracker(fake_mediapipe)
        tracker.start()

        for index in range(3):
            tracker.process(make_frame(frame_id=index, timestamp=100.0))

        assert fake_mediapipe["created"][0].timestamps == [100_000, 100_001, 100_002]


class TestPixelFormats:
    def test_bgr_is_converted_to_rgb(self, fake_mediapipe):
        tracker = make_tracker(fake_mediapipe)
        tracker.start()

        frame = make_frame(pixel_format=PixelFormat.BGR24)
        tracker.process(frame)

        delivered = fake_mediapipe["created"][0].images[0].data
        assert np.array_equal(delivered, frame.data[:, :, ::-1])
        assert delivered.flags["C_CONTIGUOUS"]

    def test_rgb_is_passed_through_untouched(self, fake_mediapipe):
        tracker = make_tracker(fake_mediapipe)
        tracker.start()

        frame = make_frame(pixel_format=PixelFormat.RGB24)
        tracker.process(frame)

        assert np.array_equal(fake_mediapipe["created"][0].images[0].data, frame.data)

    @pytest.mark.parametrize(
        "pixel_format",
        [PixelFormat.RGBA32, PixelFormat.BGRA32, PixelFormat.GRAY8],
    )
    def test_other_supported_formats_arrive_as_three_channels(self, fake_mediapipe, pixel_format):
        tracker = make_tracker(fake_mediapipe)
        tracker.start()

        tracker.process(make_frame(pixel_format=pixel_format))

        delivered = fake_mediapipe["created"][0].images[0].data
        assert delivered.shape == (6, 8, 3)
        assert delivered.flags["C_CONTIGUOUS"]

    @pytest.mark.parametrize(
        "pixel_format",
        [PixelFormat.BGR24, PixelFormat.RGB24, PixelFormat.RGBA32, PixelFormat.BGRA32],
    )
    def test_the_numpy_fallback_matches_the_opencv_path(
        self, fake_mediapipe, monkeypatch, pixel_format
    ):
        """The two conversion paths must produce identical pixels.

        OpenCV is used when present because it is five times faster (B003), but
        a tracker that saw different pixels depending on which library happened
        to be installed would be a silent source of divergent tracking.
        """
        frame = make_frame(pixel_format=pixel_format)

        with_opencv = make_tracker(fake_mediapipe)
        with_opencv.start()
        with_opencv.process(frame)
        opencv_pixels = fake_mediapipe["created"][0].images[0].data

        monkeypatch.setattr("sha256.tracking.mediapipe_tracker._try_import_cv2", lambda: (None, {}))
        with_numpy = make_tracker(fake_mediapipe)
        with_numpy.start()
        with_numpy.process(frame)
        numpy_pixels = fake_mediapipe["created"][1].images[0].data

        assert np.array_equal(opencv_pixels, numpy_pixels)

    def test_an_unsupported_format_is_refused_clearly(self, fake_mediapipe):
        tracker = make_tracker(fake_mediapipe)
        tracker.start()

        frame = VideoFrame(
            data=np.zeros((6, 8), dtype=np.uint8),
            pixel_format=PixelFormat.NV12,
            metadata=FrameMetadata(frame_id=0, capture_timestamp=1.0),
        )

        with pytest.raises(TrackingError, match="nv12"):
            tracker.process(frame)


class TestSmoothing:
    def test_smoothing_is_off_by_default(self, fake_mediapipe):
        """Without smoothing the tracker must report exactly what it was given."""
        tracker = make_tracker(fake_mediapipe)
        tracker.start()

        tracker.process(make_frame(frame_id=0, timestamp=100.0))
        fake_mediapipe["created"][0].position = 1.0
        result = tracker.process(make_frame(frame_id=1, timestamp=100.1))

        assert result.landmarks is not None
        assert result.landmarks.points == pytest.approx(1.0)

    def test_smoothing_blends_toward_the_previous_frame(self, fake_mediapipe):
        tracker = make_tracker(fake_mediapipe, smoothing=0.5)
        tracker.start()

        fake_mediapipe["created"][0].position = 0.0
        tracker.process(make_frame(frame_id=0, timestamp=100.0))

        fake_mediapipe["created"][0].position = 1.0
        result = tracker.process(make_frame(frame_id=1, timestamp=100.1))

        # Half way between where the face was and where it now is — everywhere
        # except the lips, which have their own factor and default to none of it.
        assert result.landmarks is not None
        rest = [i for i in range(478) if i not in set(MOUTH_LANDMARKS)]
        assert result.landmarks.points[rest] == pytest.approx(0.5)

    def test_smoothing_also_applies_to_expression(self, fake_mediapipe):
        tracker = make_tracker(fake_mediapipe, smoothing=0.5)
        tracker.start()

        tracker.process(make_frame(frame_id=0, timestamp=100.0))
        result = tracker.process(make_frame(frame_id=1, timestamp=100.1))

        # A steady input must survive the filter unchanged: an EMA that shifts a
        # constant signal is wrong however smooth it looks.
        assert result.blendshapes is not None
        assert result.blendshapes.get("jawOpen") == pytest.approx(0.25)

    def test_losing_the_face_clears_the_filter(self, fake_mediapipe):
        tracker = make_tracker(fake_mediapipe, smoothing=0.9)
        tracker.start()
        tracker.process(make_frame())

        fake_mediapipe["created"][0].detects = False
        tracker.process(make_frame(frame_id=1, timestamp=100.1))

        # A stale filter would drag the first re-acquired frame toward wherever
        # the face used to be, which reads on screen as the avatar sliding in.
        assert tracker._smoothed_points is None

    def test_reset_clears_temporal_state_without_rebuilding_the_graph(self, fake_mediapipe):
        tracker = make_tracker(fake_mediapipe, smoothing=0.5)
        tracker.start()
        tracker.process(make_frame())

        tracker.reset()

        assert tracker._smoothed_points is None
        assert tracker.is_running
        assert len(fake_mediapipe["created"]) == 1


class TestTheLipsAreFilteredOnTheirOwn:
    """`mouth_smoothing`, and the owner's report that produced it.

    2026-09-03, after testing: *"at fast speech the photographs cannot follow
    the mouth; when I speak slowly they roughly match."* A low-pass filter is
    speed-selective by construction, which is exactly that shape, and the
    measurement agreed — at 0.35 the mouth reading is 0.090 from the unfiltered
    one at p50 on the frames where he moves fast, 0.309 at worst, and one frame
    late, while the engine choosing the photograph was measured the same evening
    and is not late at all.

    The filter is not wrong; it is aimed at the wrong landmarks. What it steadies
    is the mask edge and the ladders near rest, which is a still face, and the
    lips are not what it steadies there.
    """

    def _moved(self, fake_mediapipe, **kwargs):
        tracker = make_tracker(fake_mediapipe, **kwargs)
        tracker.start()
        fake_mediapipe["created"][0].position = 0.0
        tracker.process(make_frame(frame_id=0, timestamp=100.0))
        fake_mediapipe["created"][0].position = 1.0
        result = tracker.process(make_frame(frame_id=1, timestamp=100.1))
        assert result.landmarks is not None
        return result.landmarks.points

    def test_the_lips_arrive_while_the_face_is_still_catching_up(self, fake_mediapipe):
        points = self._moved(fake_mediapipe, smoothing=0.5, mouth_smoothing=0.0)

        lips = list(MOUTH_LANDMARKS)
        rest = [i for i in range(len(points)) if i not in set(lips)]
        assert points[lips] == pytest.approx(1.0)
        assert points[rest] == pytest.approx(0.5)

    def test_one_factor_for_both_is_the_filter_it_replaced(self, fake_mediapipe):
        """The old behaviour has to remain reachable, or the change cannot be
        undone by configuration alone."""
        points = self._moved(fake_mediapipe, smoothing=0.5, mouth_smoothing=0.5)

        assert points == pytest.approx(0.5)

    def test_the_lips_can_be_the_filtered_ones(self, fake_mediapipe):
        """Filtering only the lips is the same code path backwards, and it also
        proves the filter runs at all when `smoothing` itself is zero."""
        points = self._moved(fake_mediapipe, smoothing=0.0, mouth_smoothing=0.5)

        lips = list(MOUTH_LANDMARKS)
        rest = [i for i in range(len(points)) if i not in set(lips)]
        assert points[lips] == pytest.approx(0.5)
        assert points[rest] == pytest.approx(1.0)

    def test_a_factor_outside_the_range_is_refused(self, fake_mediapipe):
        with pytest.raises(ValueError, match="mouth_smoothing"):
            make_tracker(fake_mediapipe, mouth_smoothing=1.0)

    def test_the_lips_named_here_are_the_ones_the_renderer_reads(self):
        """The two rings are a fact about the mesh, and they are written down in
        two places — here and `engine_warp` — because the tracker must not import
        across the layer. This is what keeps them from drifting apart."""
        from sha256.avatar.engine_warp import _INNER_LIPS, _OUTER_LIPS

        assert set(MOUTH_LANDMARKS) == set(_OUTER_LIPS) | set(_INNER_LIPS)


class TestHeadPoseConversion:
    """The convention on HeadPose: pitch up, yaw to the subject's left, roll to
    the subject's right, all positive."""

    def test_identity_is_a_neutral_pose(self):
        pose = _head_pose_from_matrix(np.eye(4, dtype=np.float32))

        assert (pose.pitch, pose.yaw, pose.roll) == (0.0, 0.0, 0.0)

    def test_translation_is_carried_through(self):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = (1.5, -2.0, 30.0)

        assert _head_pose_from_matrix(matrix).translation == (1.5, -2.0, 30.0)

    @pytest.mark.parametrize(
        ("axis", "angle_degrees"),
        [("pitch", 20.0), ("yaw", 25.0), ("roll", 15.0)],
    )
    def test_each_axis_recovers_the_angle_it_was_built_from(self, axis, angle_degrees):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = _rotation(axis, math.radians(angle_degrees))

        pose = _head_pose_from_matrix(matrix)

        assert getattr(pose, axis) == pytest.approx(angle_degrees, abs=1e-3)
        # The other two axes must stay at zero, or the decomposition is leaking
        # rotation between axes — the bug that makes an avatar drift as it turns.
        for other in {"pitch", "yaw", "roll"} - {axis}:
            assert getattr(pose, other) == pytest.approx(0.0, abs=1e-3)

    def test_scale_in_the_matrix_does_not_change_the_angles(self):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = _rotation("yaw", math.radians(30.0)) * 4.0

        assert _head_pose_from_matrix(matrix).yaw == pytest.approx(30.0, abs=1e-3)

    def test_a_degenerate_matrix_yields_neutral_rather_than_nan(self):
        """A NaN pose would propagate silently into every downstream value."""
        pose = _head_pose_from_matrix(np.zeros((4, 4), dtype=np.float32))

        assert (pose.pitch, pose.yaw, pose.roll) == (0.0, 0.0, 0.0)


def _rotation(axis: str, radians: float) -> np.ndarray:
    """A camera-space rotation that *should* read back as ``+radians`` on ``axis``.

    MediaPipe's matrix space is right-handed with X to the image right, Y up, and
    Z toward the viewer — measured in B003, not assumed. In that frame:

    * a **positive pitch** (looking up) is a *negative* rotation about +X, since
      a positive one swings the face's forward direction downward;
    * a **positive yaw** (toward the subject's left) is a positive rotation
      about +Y, which turns the face toward the image right;
    * a **positive roll** (toward the subject's right) is a positive rotation
      about +Z, which tips the top of the head toward the image left.

    Writing the expected physical rotation here, rather than mirroring whatever
    the implementation does, is what makes these tests capable of failing if the
    convention is broken later.
    """
    if axis == "pitch":
        cos, sin = math.cos(-radians), math.sin(-radians)
        return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]], dtype=np.float32)
    cos, sin = math.cos(radians), math.sin(radians)
    if axis == "yaw":
        return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]], dtype=np.float32)
    return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)
