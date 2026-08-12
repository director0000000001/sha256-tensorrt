"""Pipeline lifecycle control.

**Scope:** this is not the real-time loop. It owns component lifecycle
(start/stop ordering, failure roll-back) and can execute a *single* pass
through the stages for testing and diagnostics. There is no capture thread, no
frame queue, and no back-pressure handling yet — those are the next milestone
and are designed in ``docs/architecture/SYSTEM_ARCHITECTURE.md``.

Separating lifecycle from the loop now means the threading work later changes
one class rather than rewiring the application.
"""

from __future__ import annotations

from dataclasses import dataclass

from sha256.avatar.manager import AvatarManager
from sha256.core.enums import PipelineStage, PipelineState
from sha256.core.interfaces import (
    CameraProvider,
    FaceTracker,
    PerformanceMonitor,
    Renderer,
    VirtualCameraSink,
)
from sha256.core.types import FaceTrackingResult, RenderedFrame, VideoFrame
from sha256.utils.logging import get_logger

__all__ = ["PipelineComponents", "PipelineController"]

_LOG = get_logger(__name__)


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
                if not sink.is_available:
                    msg = f"virtual camera backend {sink.backend_id!r} is not available"
                    raise RuntimeError(msg)
                sink.start()
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

    # --- single step -------------------------------------------------------

    def process_once(self) -> RenderedFrame | None:
        """Run one frame through the stages, or return ``None`` if none is ready.

        Diagnostic and test entry point. The production path will call this
        logic from a dedicated thread with proper frame pacing.
        """
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

        engine = self._components.avatars.engine
        active = self._components.avatars.active

        if engine is None or active is None:
            # No avatar selected is a normal idle state, not an error: the sink
            # keeps emitting a valid signal so downstream consumers stay happy.
            rendered = self._components.renderer.render_placeholder(
                frame.metadata, "No avatar selected"
            )
        else:
            with monitor.measure(PipelineStage.ANIMATION):
                animated = engine.animate(active.id, tracking)
            with monitor.measure(PipelineStage.RENDERING):
                rendered = self._components.renderer.render(animated)

        monitor.record_frame(PipelineStage.RENDERING)

        if self._publish_output and self._components.virtual_camera.is_running:
            with monitor.measure(PipelineStage.OUTPUT):
                self._components.virtual_camera.send(rendered)
            monitor.record_frame(PipelineStage.OUTPUT)

        monitor.record_duration(PipelineStage.END_TO_END, rendered.pipeline_latency)
        self._last_frame = rendered
        return rendered
