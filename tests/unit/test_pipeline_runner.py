"""Tests for the real-time loop.

The loop is the first threaded component in the project, so the things tested
here are the ones that go wrong quietly: a failure on a background thread that
nobody sees, a frame published while a reader is mid-read, and the "drop, never
queue" rule silently becoming "queue everything".

A fake controller stands in for the real pipeline — the loop's behaviour does
not depend on what a frame contains.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from sha256.core.enums import PixelFormat
from sha256.core.types import (
    BlendshapeSet,
    FaceTrackingResult,
    FrameMetadata,
    HeadPose,
    LandmarkSet,
    RenderedFrame,
    VideoFrame,
)
from sha256.pipeline.runner import PipelineRunner


class FakeController:
    """A controller that produces frames on demand, or misbehaves on request."""

    def __init__(self, *, fail_after: int | None = None, produce_none: bool = False) -> None:
        self.started = False
        self.stopped = False
        self.start_error: Exception | None = None
        self.frames = 0
        self.fail_after = fail_after
        self._produce_none = produce_none
        self._lock = threading.Lock()
        # Set by a test that wants particular tracking published with a frame;
        # left ``None``, the controller reports no face, as it always did.
        self.tracking: FaceTrackingResult | None = None

    def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def process_once(self) -> RenderedFrame | None:
        with self._lock:
            self.frames += 1
            count = self.frames

        if self.fail_after is not None and count > self.fail_after:
            msg = "camera exploded"
            raise RuntimeError(msg)
        if self._produce_none:
            return None

        metadata = FrameMetadata(frame_id=count, capture_timestamp=float(count))
        return RenderedFrame(
            data=np.zeros((4, 4, 3), dtype=np.uint8),
            pixel_format=PixelFormat.RGB24,
            metadata=metadata,
            render_timestamp=float(count),
        )

    @property
    def last_source_frame(self) -> VideoFrame | None:
        # Not the rendered frame's zeros: a capture must be able to tell the
        # camera frame from the frame on screen.
        return VideoFrame(
            data=np.full((4, 4, 3), 200, dtype=np.uint8),
            pixel_format=PixelFormat.BGR24,
            metadata=FrameMetadata(frame_id=self.frames, capture_timestamp=0.0),
        )

    @property
    def last_tracking(self) -> FaceTrackingResult | None:
        if self.tracking is not None:
            return self.tracking
        return FaceTrackingResult.no_face(
            FrameMetadata(frame_id=self.frames, capture_timestamp=0.0)
        )


def wait_until(predicate, timeout: float = 2.0) -> bool:
    """Poll until ``predicate`` holds. Threads make sleeps unavoidable; keep them short."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def runner_factory():
    """Build runners and guarantee they are stopped, even if a test fails."""
    built: list[PipelineRunner] = []

    def build(controller: FakeController, **kwargs) -> PipelineRunner:
        runner = PipelineRunner(controller, **kwargs)  # type: ignore[arg-type]
        built.append(runner)
        return runner

    yield build

    for runner in built:
        runner.stop()


class TestLifecycle:
    def test_start_starts_the_controller_then_produces_frames(self, runner_factory):
        controller = FakeController()
        runner = runner_factory(controller)

        runner.start()

        assert controller.started
        assert wait_until(lambda: runner.latest() is not None)
        assert runner.is_running

    def test_stop_stops_the_thread_and_the_controller(self, runner_factory):
        controller = FakeController()
        runner = runner_factory(controller)
        runner.start()
        wait_until(lambda: runner.latest() is not None)

        runner.stop()

        assert not runner.is_running
        assert controller.stopped

    def test_starting_twice_is_idempotent(self, runner_factory):
        controller = FakeController()
        runner = runner_factory(controller)

        runner.start()
        runner.start()

        assert runner.is_running

    def test_stopping_a_stopped_runner_is_safe(self, runner_factory):
        runner_factory(FakeController()).stop()

    def test_a_component_that_cannot_start_raises_to_the_caller(self, runner_factory):
        """The UI has to be able to show this; a background thread cannot."""
        controller = FakeController()
        controller.start_error = RuntimeError("camera in use by another application")
        runner = runner_factory(controller)

        with pytest.raises(RuntimeError, match="camera in use"):
            runner.start()

        assert not runner.is_running


class TestFailureHandling:
    def test_a_failing_loop_stops_and_records_why(self, runner_factory):
        controller = FakeController(fail_after=2)
        runner = runner_factory(controller)

        runner.start()

        assert wait_until(lambda: runner.last_error is not None)
        assert "camera exploded" in (runner.last_error or "")
        assert wait_until(lambda: not runner.is_running), "the loop must not spin on a failure"

    def test_the_error_callback_fires(self, runner_factory):
        seen: list[str] = []
        runner = runner_factory(FakeController(fail_after=1), on_error=seen.append)

        runner.start()

        assert wait_until(lambda: bool(seen))
        assert "camera exploded" in seen[0]

    def test_restarting_after_fixing_the_fault_clears_the_error(self, runner_factory):
        """A recovered pipeline must not keep reporting the old failure."""
        controller = FakeController(fail_after=1)
        runner = runner_factory(controller)
        runner.start()
        assert wait_until(lambda: runner.last_error is not None)

        runner.stop()
        controller.fail_after = None  # the camera came back
        runner.start()

        assert wait_until(lambda: runner.latest() is not None)
        assert runner.last_error is None


class TestFrameDelivery:
    def test_the_snapshot_carries_frame_source_and_tracking_together(self, runner_factory):
        """Reading them separately would let a consumer mix two frames."""
        runner = runner_factory(FakeController())
        runner.start()

        assert wait_until(lambda: runner.latest() is not None)
        snapshot = runner.latest()

        assert snapshot is not None
        assert snapshot.frame is not None
        assert snapshot.source is not None
        assert snapshot.tracking is not None
        assert snapshot.frame_id == snapshot.frame.metadata.frame_id

    def test_only_the_newest_frame_is_kept(self, runner_factory):
        """Drop, never queue: a slow reader must not build a backlog."""
        runner = runner_factory(FakeController())
        runner.start()

        assert wait_until(lambda: runner.stats["produced"] > 20)
        first = runner.latest()
        assert first is not None

        assert wait_until(lambda: runner.stats["produced"] > first.frame_id + 10)
        second = runner.latest()

        assert second is not None
        # A queue would have handed back the very next frame in order.
        assert second.frame_id > first.frame_id + 1

    def test_uncollected_frames_are_counted_not_hidden(self, runner_factory):
        runner = runner_factory(FakeController())
        runner.start()

        assert wait_until(lambda: runner.stats["produced"] > 30)
        runner.stop()

        stats = runner.stats
        assert stats["uncollected"] > 0
        assert stats["produced"] >= stats["collected"] + stats["uncollected"]

    def test_collecting_the_same_frame_twice_counts_once(self, runner_factory):
        runner = runner_factory(FakeController())
        runner.start()
        assert wait_until(lambda: runner.latest() is not None)
        runner.stop()

        # Collect whatever the loop published between the wait and the stop, so
        # the counter below is measuring re-reads and not a genuinely new frame.
        runner.latest()
        before = runner.stats["collected"]

        runner.latest()
        runner.latest()

        assert runner.stats["collected"] == before

    def test_a_source_that_yields_nothing_does_not_spin_hot(self, runner_factory):
        """A paused device must cost a sleeping thread, not a busy core."""
        controller = FakeController(produce_none=True)
        runner = runner_factory(controller)

        runner.start()
        time.sleep(0.15)
        runner.stop()

        # Unthrottled this loop would run millions of times; the idle wait keeps
        # it near 150ms / 2ms.
        assert controller.frames < 500, f"loop ran {controller.frames} times in 150ms"
        assert runner.latest() is None


class TestThroughput:
    def test_fps_is_reported_once_a_window_has_elapsed(self, runner_factory):
        runner = runner_factory(FakeController())
        runner.start()

        assert wait_until(lambda: runner.fps > 0.0, timeout=3.0)

    def test_fps_is_zero_before_anything_runs(self, runner_factory):
        assert runner_factory(FakeController()).fps == 0.0


def tracked(frame_id: int, *, face: bool) -> FaceTrackingResult:
    """One frame of tracking the way the real tracker reports it."""
    metadata = FrameMetadata(frame_id=frame_id, capture_timestamp=frame_id / 30.0)
    if not face:
        return FaceTrackingResult.no_face(metadata)
    return FaceTrackingResult(
        metadata=metadata,
        face_detected=True,
        landmarks=LandmarkSet(
            points=np.full((478, 3), 0.5, dtype=np.float32), scheme="mediapipe_478"
        ),
        head_pose=HeadPose(yaw=1.0),
        blendshapes=BlendshapeSet(
            names=("jawOpen", "mouthPucker"), weights=np.array([0.1, 0.2], dtype=np.float32)
        ),
    )


class TestCapture:
    """The "That looked wrong" button, and the frames it must not choke on."""

    def _publish(self, runner: PipelineRunner, controller: FakeController, faces: list[bool]):
        for index, face in enumerate(faces):
            controller.tracking = tracked(index, face=face)
            frame = controller.process_once()
            assert frame is not None
            runner._publish(frame)

    def test_a_frame_without_a_face_does_not_lose_the_capture(self, runner_factory, tmp_path):
        """Frames 3 and 7 have no face: a tracking dropout inside the ten
        seconds, which is the very moment the button exists for."""
        controller = FakeController()
        runner = runner_factory(controller)
        self._publish(runner, controller, [i not in (3, 7) for i in range(10)])

        target = runner.capture(tmp_path)

        data = np.load(target / "tracking.npz", allow_pickle=False)
        assert data["landmarks"].shape == (10, 478, 3)
        assert data["detected"].tolist() == [i not in (3, 7) for i in range(10)]
        assert not data["detected"][3] and not data["detected"][7]
        assert data["weights"].shape == (10, 2)
        assert data["pose"].shape == (10, 3)
        assert data["timestamps"].shape == (10,)

    def test_a_capture_holds_the_screen_frame_and_the_camera_frame(self, runner_factory, tmp_path):
        """Day 12: the neural engine reads pixels, so a complaint can only be
        replayed against the camera frame of the same tick as the frame that
        was on screen. Both are written, and they are not the same picture."""
        import cv2

        controller = FakeController()
        runner = runner_factory(controller)
        self._publish(runner, controller, [True] * 3)

        target = runner.capture(tmp_path)

        seen = cv2.imread(str(target / "seen.png"))
        camera = cv2.imread(str(target / "camera.png"))
        assert seen is not None and camera is not None
        assert seen.shape == camera.shape == (4, 4, 3)
        assert not seen.any()  # the rendered frame: zeros
        assert (camera == 200).all()  # the source frame: what the camera gave

    def test_ten_seconds_with_no_face_at_all_is_still_written(self, runner_factory, tmp_path):
        """A loss lasting the whole window is a report in its own right: the
        file holds no points and ``detected`` says so on every row."""
        controller = FakeController()
        runner = runner_factory(controller)
        self._publish(runner, controller, [False] * 10)

        target = runner.capture(tmp_path)

        data = np.load(target / "tracking.npz", allow_pickle=False)
        assert data["landmarks"].shape == (10, 0, 3)
        assert not data["detected"].any()
