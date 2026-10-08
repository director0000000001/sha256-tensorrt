"""The two-thread loop (2026-10-08, "The fast version", step 1).

What goes wrong quietly with two threads: a frame published with another
frame's tracking, a slow render thread turning the mailbox into a queue, a
failure on one thread leaving the other running, a stop that waits on an
empty mailbox forever. A fake controller with the two halves stands in for
the pipeline; the loop's behaviour does not depend on what a frame contains.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from sha256.core.enums import PixelFormat
from sha256.core.types import FaceTrackingResult, FrameMetadata, RenderedFrame, VideoFrame
from sha256.pipeline.controller import TrackedFrame
from sha256.pipeline.runner import PipelineRunner, PipelineSnapshot
from tests.unit.test_pipeline_runner import FakeController, wait_until


class TwoHalfController:
    """A controller with the two halves. The capture half numbers its frames;
    the render half can be slow, or fail, on request."""

    def __init__(
        self,
        *,
        render_seconds: float = 0.0,
        capture_seconds: float = 0.0,
        fail_capture_after: int | None = None,
        fail_render_after: int | None = None,
        capture_nothing: bool = False,
    ) -> None:
        self.started = False
        self.stopped = False
        self.render_seconds = render_seconds
        self.capture_seconds = capture_seconds
        self.fail_capture_after = fail_capture_after
        self.fail_render_after = fail_render_after
        self.capture_nothing = capture_nothing
        self.rendered_ids: list[int] = []
        # How many frames the capture half was ahead of each frame rendered,
        # read as the render starts: the lag a queue would let grow.
        self.gaps: list[int] = []
        self.failures_published: list[str] = []
        self._captured = 0
        self._rendered = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def capture_and_track(self) -> TrackedFrame | None:
        if self.capture_nothing:
            return None
        with self._lock:
            self._captured += 1
            count = self._captured
        if self.fail_capture_after is not None and count > self.fail_capture_after:
            msg = "camera exploded"
            raise RuntimeError(msg)
        if self.capture_seconds:
            time.sleep(self.capture_seconds)
        metadata = FrameMetadata(frame_id=count, capture_timestamp=float(count))
        frame = VideoFrame(
            data=np.full((4, 4, 3), count % 256, dtype=np.uint8),
            pixel_format=PixelFormat.BGR24,
            metadata=metadata,
        )
        return TrackedFrame(frame, FaceTrackingResult.no_face(metadata))

    def render_tracked(self, tracked: TrackedFrame) -> RenderedFrame | None:
        with self._lock:
            self._rendered += 1
            count = self._rendered
            self.gaps.append(self._captured - tracked.frame.metadata.frame_id)
        if self.fail_render_after is not None and count > self.fail_render_after:
            msg = "gpu exploded"
            raise RuntimeError(msg)
        if self.render_seconds:
            time.sleep(self.render_seconds)
        self.rendered_ids.append(tracked.frame.metadata.frame_id)
        return RenderedFrame(
            data=tracked.frame.data,
            pixel_format=PixelFormat.BGR24,
            metadata=tracked.frame.metadata,
            render_timestamp=float(tracked.frame.metadata.frame_id) + 0.5,
        )

    def process_once(self) -> RenderedFrame | None:
        tracked = self.capture_and_track()
        return None if tracked is None else self.render_tracked(tracked)

    def publish_failure(self, reason: str) -> bool:
        self.failures_published.append(reason)
        return True

    @property
    def last_source_frame(self) -> VideoFrame | None:
        return None

    @property
    def last_tracking(self) -> FaceTrackingResult | None:
        return None


@pytest.fixture
def runner_factory():
    built: list[PipelineRunner] = []

    def build(controller, **kwargs) -> PipelineRunner:
        runner = PipelineRunner(controller, **kwargs)  # type: ignore[arg-type]
        built.append(runner)
        return runner

    yield build

    for runner in built:
        runner.stop()


def _thread_names() -> set[str]:
    return {t.name for t in threading.enumerate()}


class TestTwoThreads:
    def test_the_loop_runs_on_two_threads_when_the_controller_has_the_halves(self, runner_factory):
        controller = TwoHalfController()
        runner = runner_factory(controller)
        assert runner.overlapping
        runner.start()
        assert wait_until(lambda: runner.stats["produced"] >= 5)
        names = _thread_names()
        assert "sha256-capture" in names
        assert "sha256-render" in names
        assert "sha256-pipeline" not in names

    def test_every_snapshot_pairs_the_frame_with_its_own_tracking(self, runner_factory):
        controller = TwoHalfController(render_seconds=0.004)
        runner = runner_factory(controller)
        seen: list[PipelineSnapshot] = []
        runner.add_observer(seen.append)
        runner.start()
        assert wait_until(lambda: len(seen) >= 20)
        runner.stop()
        for snapshot in seen:
            assert snapshot.frame is not None
            assert snapshot.tracking is not None
            assert snapshot.source is not None
            assert snapshot.tracking.metadata.frame_id == snapshot.frame.metadata.frame_id
            assert snapshot.source.metadata.frame_id == snapshot.frame.metadata.frame_id
            assert snapshot.frame_id == snapshot.frame.metadata.frame_id

    def test_a_slow_render_drops_at_the_mailbox_and_never_queues(self, runner_factory):
        # Capture is instant, the render takes 10 ms: most tracked frames
        # are overwritten before the render thread takes them. What it
        # renders is always the newest, in order, and the drops are counted.
        controller = TwoHalfController(render_seconds=0.01)
        runner = runner_factory(controller)
        runner.start()
        assert wait_until(lambda: runner.stats["produced"] >= 10)
        runner.stop()
        stats = runner.stats
        assert stats["dropped_at_capture"] > 0
        ids = list(controller.rendered_ids)
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)
        assert stats["produced"] == len(ids)

    def test_a_render_slower_than_capture_does_not_fall_behind(self, runner_factory):
        # The failure the mailbox exists to prevent: with a queue, the frame
        # rendered would be ever older than the frame captured. Here the
        # gap between the newest captured frame and the one being rendered
        # stays within a frame or two for the whole run.
        controller = TwoHalfController(render_seconds=0.005, capture_seconds=0.001)
        runner = runner_factory(controller)
        runner.start()
        assert wait_until(lambda: runner.stats["produced"] >= 60)
        runner.stop()
        gaps = controller.gaps
        # A render takes five captures' worth of time, so the frame taken is
        # a handful behind the newest -- and as far behind at the end as at
        # the start. A queue would make the last gaps the largest.
        assert max(gaps) <= 12
        assert max(gaps[-20:]) <= max(gaps[:20]) + 3

    def test_a_failing_capture_half_stops_both_threads_and_reports(self, runner_factory):
        errors: list[str] = []
        controller = TwoHalfController(fail_capture_after=3)
        runner = runner_factory(controller, on_error=errors.append)
        runner.start()
        assert wait_until(lambda: not runner.is_running)
        assert runner.last_error == "camera exploded"
        assert errors == ["camera exploded"]
        assert controller.failures_published == ["camera exploded"]
        assert "sha256-render" not in _thread_names()

    def test_a_failing_render_half_stops_both_threads_and_reports(self, runner_factory):
        errors: list[str] = []
        controller = TwoHalfController(fail_render_after=2)
        runner = runner_factory(controller, on_error=errors.append)
        runner.start()
        assert wait_until(lambda: not runner.is_running)
        assert runner.last_error == "gpu exploded"
        assert errors == ["gpu exploded"]
        assert "sha256-capture" not in _thread_names()

    def test_stop_returns_promptly_with_an_empty_mailbox(self, runner_factory):
        controller = TwoHalfController(capture_nothing=True)
        runner = runner_factory(controller)
        runner.start()
        time.sleep(0.05)
        started = time.monotonic()
        runner.stop()
        assert time.monotonic() - started < 1.0
        assert not runner.is_running
        assert controller.stopped

    def test_restarting_resets_the_mailbox_and_the_drop_count(self, runner_factory):
        controller = TwoHalfController(render_seconds=0.01)
        runner = runner_factory(controller)
        runner.start()
        assert wait_until(lambda: runner.stats["dropped_at_capture"] > 0)
        runner.stop()
        # Nothing captured on the second run, so the count can only be the
        # reset one.
        controller.capture_nothing = True
        runner.start()
        time.sleep(0.05)
        assert runner.is_running
        assert runner.stats["dropped_at_capture"] == 0


class TestSingleThreadStillThere:
    def test_overlap_off_keeps_the_single_thread(self, runner_factory):
        controller = TwoHalfController()
        runner = runner_factory(controller, overlap=False)
        assert not runner.overlapping
        runner.start()
        assert wait_until(lambda: runner.stats["produced"] >= 5)
        names = _thread_names()
        assert "sha256-pipeline" in names
        assert "sha256-render" not in names
        assert runner.stats["dropped_at_capture"] == 0

    def test_a_controller_without_the_halves_runs_the_single_thread(self, runner_factory):
        controller = FakeController()
        runner = runner_factory(controller)
        assert not runner.overlapping
        runner.start()
        assert wait_until(lambda: runner.stats["produced"] >= 5)
        assert "sha256-pipeline" in _thread_names()
