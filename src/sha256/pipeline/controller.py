"""Pipeline lifecycle control.

**Scope:** this is not the real-time loop. It owns component lifecycle
(start/stop ordering, failure roll-back) and executes the stages for one
frame -- as one call (:meth:`PipelineController.process_once`) or as the two
halves the loop runs on two threads (:meth:`capture_and_track`, then
:meth:`render_tracked`; see ``pipeline/runner.py``, "The fast version",
2026-10-08). The threads, the frame hand-over and the pacing live in the
runner; this class knows nothing of them, which is what keeps the single
call -- the path the tests, the probes and the replays use -- the same code
as the two halves.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from sha256.avatar.manager import AvatarManager
from sha256.core.enums import PipelineStage, PipelineState
from sha256.core.errors import VirtualCameraError
from sha256.core.interfaces import (
    CameraProvider,
    FaceTracker,
    PerformanceMonitor,
    Renderer,
    VirtualCameraSink,
)
from sha256.core.types import FaceTrackingResult, RenderedFrame, VideoFrame
from sha256.pipeline.conditioning import TURNED_AWAY_DEGREES, OutputConditioner
from sha256.utils.logging import get_logger

__all__ = ["PipelineComponents", "PipelineController", "TrackedFrame"]

_LOG = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class TrackedFrame:
    """A camera frame with its own tracking -- what the first half of a
    frame hands the second (2026-10-08). The two travel together so a
    render can never pair a frame with another frame's landmarks."""

    frame: VideoFrame
    tracking: FaceTrackingResult


@dataclass(slots=True)
class PipelineComponents:
    """The swappable parts the controller drives.

    Grouped into one object so wiring is constructed and validated in a single
    place, and so a test can substitute any subset without a long argument list.
    """

    camera: CameraProvider
    tracker: FaceTracker
    renderer: Renderer
    virtual_camera: VirtualCameraSink
    avatars: AvatarManager
    monitor: PerformanceMonitor
    # Day 15: the last stage before the sink -- the output quality and the
    # black screen (see sha256.pipeline.conditioning). Defaulted so the
    # wiring that predates it, tests included, is unchanged.
    conditioner: OutputConditioner = field(default_factory=OutputConditioner)


class PipelineController:
    """Starts, stops, and single-steps the processing components."""

    def __init__(self, components: PipelineComponents, *, publish_output: bool = False) -> None:
        self._components = components
        self._publish_output = publish_output
        self._state = PipelineState.STOPPED
        self._last_error: str | None = None
        self._last_frame: RenderedFrame | None = None
        self._last_source: VideoFrame | None = None
        self._last_tracking: FaceTrackingResult | None = None
        # Day 19: the head's turn (the larger of |yaw| and |pitch|) on the
        # last frame with a face, for the conditioner's turned-away loss.
        self._last_turn = 0.0

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state is PipelineState.RUNNING

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def last_frame(self) -> RenderedFrame | None:
        """Most recent rendered frame, for the preview widget."""
        return self._last_frame

    @property
    def last_source_frame(self) -> VideoFrame | None:
        """Most recent *camera* frame, for a diagnostic preview.

        The buffer is **borrowed**, following the rule on :class:`VideoFrame`: a
        capture backend may reuse it for the next frame. Anything that needs the
        pixels beyond the current tick must copy them.
        """
        return self._last_source

    @property
    def last_tracking(self) -> FaceTrackingResult | None:
        """Most recent tracking result, for overlays and diagnostics.

        Exposed so the preview can draw landmarks without running the tracker a
        second time, and so "is it tracking my face right now" is answerable
        from the UI rather than from a log file.
        """
        return self._last_tracking

    @property
    def components(self) -> PipelineComponents:
        return self._components

    @property
    def publish_output(self) -> bool:
        """Whether rendered frames are reaching the virtual camera."""
        return self._publish_output

    def set_publish_output(self, enabled: bool) -> None:
        """Turn publishing on or off, including while the pipeline runs.

        Publishing is a decision made *during* a call — "put me on the virtual
        camera now" — so it cannot be fixed at construction. The sink's lifecycle
        follows the flag rather than the pipeline's: a stopped sink exposes no
        device, which is what keeps sha256 out of Zoom's camera list while the
        owner is only looking at the preview.

        Raises:
            VirtualCameraError: the sink is unavailable or refused to start. The
                flag stays off in that case, so a failure leaves the pipeline
                running and un-published rather than in a half-on state.
        """
        if enabled == self._publish_output:
            return

        if not self.is_running:
            self._publish_output = enabled
            return

        sink = self._components.virtual_camera
        if enabled:
            self._start_sink(sink)
        else:
            sink.stop()
        self._publish_output = enabled

    @staticmethod
    def _start_sink(sink: VirtualCameraSink) -> None:
        if not sink.is_available:
            msg = f"virtual camera backend {sink.backend_id!r} is not available"
            raise VirtualCameraError(msg)
        sink.start()

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start components in dependency order, rolling back on failure.

        Order matters: the sink is started last so it never advertises a device
        that has no frames behind it, and it is stopped first for the same
        reason.
        """
        if self._state in (PipelineState.RUNNING, PipelineState.STARTING):
            return

        self._state = PipelineState.STARTING
        self._last_error = None
        self._components.conditioner.reset()
        started: list[object] = []

        try:
            for component in (
                self._components.camera,
                self._components.tracker,
                self._components.renderer,
            ):
                component.start()
                started.append(component)

            if self._publish_output:
                sink = self._components.virtual_camera
                self._start_sink(sink)
                started.append(sink)

            self._state = PipelineState.RUNNING
            _LOG.info("Pipeline started")
        except Exception as exc:
            # Partial start leaves a camera handle or GPU context open, which
            # would block the next attempt. Unwind before reporting.
            self._rollback(started)
            self._state = PipelineState.FAILED
            self._last_error = str(exc)
            _LOG.exception("Pipeline failed to start")
            raise

    @staticmethod
    def _rollback(started: list[object]) -> None:
        for component in reversed(started):
            try:
                component.stop()  # type: ignore[attr-defined]
            except Exception:
                _LOG.exception("Error while rolling back component %r", component)

    def stop(self) -> None:
        """Stop everything. Never raises — a failing stop must not strand state."""
        if self._state is PipelineState.STOPPED:
            return

        self._state = PipelineState.STOPPING
        for component in (
            self._components.virtual_camera,
            self._components.renderer,
            self._components.tracker,
            self._components.camera,
        ):
            try:
                component.stop()
            except Exception:
                _LOG.exception("Error stopping %r", component)

        self._state = PipelineState.STOPPED
        _LOG.info("Pipeline stopped")

    # --- one frame, in one call or in two halves ---------------------------

    def process_once(self) -> RenderedFrame | None:
        """Run one frame through the stages, or return ``None`` if none is ready.

        The single-thread path: the tests, the probes and the replays. It is
        the two halves below, one after the other, so what they do per frame
        is the same code the loop runs on two threads.
        """
        tracked = self.capture_and_track()
        if tracked is None:
            return None
        return self.render_tracked(tracked)

    def capture_and_track(self) -> TrackedFrame | None:
        """The first half: the camera's next frame and its tracking, or
        ``None`` when no frame is ready. CPU only -- the half the loop runs
        on its capture thread while the GPU draws the frame before."""
        if not self.is_running:
            return None

        monitor = self._components.monitor

        with monitor.measure(PipelineStage.CAPTURE):
            frame = self._components.camera.read()
        if frame is None:
            return None
        monitor.record_frame(PipelineStage.CAPTURE)
        self._last_source = frame

        with monitor.measure(PipelineStage.TRACKING):
            tracking = self._components.tracker.process(frame)
        monitor.record_frame(PipelineStage.TRACKING)
        self._last_tracking = tracking
        return TrackedFrame(frame, tracking)

    def render_tracked(self, tracked: TrackedFrame) -> RenderedFrame | None:
        """The second half: the avatar drawn over ``tracked``'s frame, the
        output conditioned and sent. ``None`` only if the pipeline stopped
        between the halves."""
        if not self.is_running:
            return None

        monitor = self._components.monitor
        frame = tracked.frame
        tracking = tracked.tracking

        engine = self._components.avatars.engine
        active = self._components.avatars.active

        passthrough = False
        if engine is None or active is None:
            # No avatar selected is a normal idle state, not an error: the sink
            # keeps emitting a valid signal so downstream consumers stay happy.
            rendered = self._components.renderer.render_placeholder(
                frame.metadata, "No avatar selected"
            )
        else:
            with monitor.measure(PipelineStage.ANIMATION):
                animated = engine.animate(active.id, tracking, frame)
            # The engine's passthrough is the camera's own picture (no face to
            # draw yet) -- the one frame that must never reach the sink.
            passthrough = animated.is_passthrough
            with monitor.measure(PipelineStage.RENDERING):
                rendered = self._components.renderer.render(animated)

        # Day 19: how far the head was turned on the last frame with a face,
        # so a loss can be told apart -- turned away past what the tracker
        # reads, or a face in front of the camera it failed on.
        if tracking.face_detected and tracking.head_pose is not None:
            pose = tracking.head_pose
            self._last_turn = max(abs(float(pose.yaw)), abs(float(pose.pitch)))
        turned_away = self._last_turn >= TURNED_AWAY_DEGREES

        # Day 15: the quality, and the black screen -- what goes out is also
        # what the preview shows, so what he sees is what the other side sees.
        rendered = self._components.conditioner.condition(
            rendered,
            face_detected=tracking.face_detected,
            passthrough=passthrough,
            turned_away=turned_away,
        )
        monitor.record_frame(PipelineStage.RENDERING)

        if self._publish_output and self._components.virtual_camera.is_running:
            with monitor.measure(PipelineStage.OUTPUT):
                self._components.virtual_camera.send(rendered)
            monitor.record_frame(PipelineStage.OUTPUT)

        monitor.record_duration(PipelineStage.END_TO_END, rendered.pipeline_latency)
        self._last_frame = rendered
        return rendered

    def publish_failure(self, reason: str) -> bool:
        """Put the black screen on the virtual camera, once, because the loop
        is stopping on an error (see ``PipelineRunner._run``): a camera whose
        battery died is what the other side should see, not the last avatar
        frame frozen for the rest of the call. Nothing to do -- and ``False``
        -- while nothing is being published, or before a first frame has
        told the sink's size."""
        sink = self._components.virtual_camera
        last = self._last_frame
        if not self._publish_output or not sink.is_running or last is None:
            return False
        black = replace(
            last,
            data=self._components.conditioner.failure_frame_for(last.width, last.height),
            avatar_id=None,
            is_placeholder=True,
        )
        sink.send(black)
        self._last_frame = black
        _LOG.warning("Black screen published on a pipeline failure: %s", reason)
        return True
