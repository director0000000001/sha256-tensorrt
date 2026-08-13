"""End-to-end tests over the placeholder pipeline.

These assert that the *contracts* compose: a frame produced by a camera reaches
a tracker, an engine, a renderer, and a sink, carrying its identity the whole
way. When real backends replace the placeholders, these tests are what proves
the substitution worked.
"""

from __future__ import annotations

import time

import pytest

from sha256.app.bootstrap import ApplicationContext, build_application
from sha256.config.paths import AppPaths
from sha256.config.schema import Settings
from sha256.core.enums import PipelineState
from sha256.core.errors import VirtualCameraError
from sha256.core.interfaces import (
    AvatarEngine,
    CameraProvider,
    FaceTracker,
    PerformanceMonitor,
    Renderer,
    VirtualCameraSink,
)
from sha256.virtual_camera.null import NullVirtualCameraSink
from tests.conftest import write_avatar

pytestmark = pytest.mark.integration


class TestProtocolConformance:
    """Placeholders must satisfy the protocols structurally, not just by name."""

    def test_components_conform(self, app_context: ApplicationContext):
        components = app_context.controller.components

        assert isinstance(components.camera, CameraProvider)
        assert isinstance(components.tracker, FaceTracker)
        assert isinstance(components.renderer, Renderer)
        assert isinstance(components.virtual_camera, VirtualCameraSink)
        assert isinstance(app_context.monitor, PerformanceMonitor)

    def test_engine_conforms(self, app_context: ApplicationContext):
        assert isinstance(app_context.avatars.engine, AvatarEngine)


class TestLifecycle:
    def test_start_and_stop_transition_state(self, app_context: ApplicationContext):
        controller = app_context.controller
        observed = [controller.state]

        controller.start()
        observed.append(controller.state)

        controller.stop()
        observed.append(controller.state)

        assert observed == [
            PipelineState.STOPPED,
            PipelineState.RUNNING,
            PipelineState.STOPPED,
        ]

    def test_starting_twice_is_idempotent(self, app_context: ApplicationContext):
        app_context.controller.start()
        app_context.controller.start()
        assert app_context.controller.is_running

    def test_stopping_a_stopped_pipeline_is_safe(self, app_context: ApplicationContext):
        app_context.controller.stop()

    def test_components_are_started_and_stopped(self, app_context: ApplicationContext):
        components = app_context.controller.components

        app_context.controller.start()
        running = (components.camera.is_running, components.tracker.is_running)

        app_context.controller.stop()
        stopped = (components.camera.is_running, components.tracker.is_running)

        assert running == (True, True)
        assert stopped == (False, False)


class TestFrameFlow:
    def test_frame_metadata_survives_the_whole_pipeline(self, app_context: ApplicationContext):
        """Identity must be preserved end to end, or latency cannot be measured."""
        app_context.avatars.select("avatar_001")
        app_context.controller.start()

        frame = _step_until_frame(app_context)

        assert frame.avatar_id == "avatar_001"
        assert frame.metadata.source_id == "null-camera"
        assert frame.render_timestamp >= frame.metadata.capture_timestamp

    def test_output_matches_the_configured_resolution(self, app_context: ApplicationContext):
        app_context.avatars.select("avatar_001")
        app_context.controller.start()

        frame = _step_until_frame(app_context)

        assert (frame.width, frame.height) == (
            app_context.settings.rendering.output_width,
            app_context.settings.rendering.output_height,
        )

    def test_no_avatar_yields_a_placeholder_frame(self, app_context: ApplicationContext):
        """Idle is a normal state — the stream must stay valid, not stop."""
        app_context.avatars.deselect()
        app_context.controller.start()

        frame = _step_until_frame(app_context)

        assert frame.is_placeholder
        assert frame.avatar_id is None

    def test_stopped_pipeline_produces_nothing(self, app_context: ApplicationContext):
        assert app_context.controller.process_once() is None

    def test_switching_avatars_changes_the_output(self, app_context: ApplicationContext):
        app_context.controller.start()

        app_context.avatars.select("avatar_001")
        assert _step_until_frame(app_context).avatar_id == "avatar_001"

        app_context.avatars.select("avatar_002")
        assert _step_until_frame(app_context).avatar_id == "avatar_002"


class TestMetricsIntegration:
    def test_stepping_frames_populates_metrics(self, app_context: ApplicationContext):
        app_context.avatars.select("avatar_001")
        app_context.controller.start()

        for _ in range(3):
            _step_until_frame(app_context)

        snapshot = app_context.monitor.snapshot()
        assert snapshot.duration("capture").count >= 1
        assert snapshot.duration("tracking").count >= 1
        assert snapshot.duration("end_to_end").count >= 1
        assert snapshot.frame_totals["capture"] >= 3


class TestVirtualCameraPublishing:
    def test_frames_reach_the_sink_when_publishing_is_enabled(self, app_paths: AppPaths):
        write_avatar(app_paths.avatars, "avatar_001")
        settings = Settings(_env_file=None, virtual_camera={"auto_start": True})
        context = build_application(settings=settings, paths=app_paths, setup_logging=False)

        try:
            # No configure() call here on purpose: wiring the sink is the
            # composition root's job, and it not doing it was a real bug —
            # auto_start could not work at all.
            sink = context.controller.components.virtual_camera
            context.avatars.select("avatar_001")
            context.controller.start()

            _step_until_frame(context)

            assert sink.frames_sent >= 1  # type: ignore[attr-defined]
        finally:
            context.shutdown()

    def test_publishing_is_off_until_it_is_asked_for(self, app_context: ApplicationContext):
        """Appearing in every application's camera list unasked is not acceptable."""
        controller = app_context.controller
        controller.start()

        assert not controller.publish_output
        assert not controller.components.virtual_camera.is_running

    def test_publishing_can_be_switched_on_while_the_pipeline_runs(
        self, app_context: ApplicationContext
    ):
        """The owner turns the camera on mid-call; it cannot be a startup-only flag."""
        controller = app_context.controller
        controller.start()

        controller.set_publish_output(True)

        sink = controller.components.virtual_camera
        assert controller.publish_output
        assert sink.is_running

        _step_until_frame(app_context)
        assert sink.frames_sent >= 1  # type: ignore[attr-defined]

    def test_switching_publishing_off_releases_the_device(self, app_context: ApplicationContext):
        """A stopped sink exposes no device — that is what keeps sha256 out of Zoom's list."""
        controller = app_context.controller
        sink = controller.components.virtual_camera
        controller.start()
        controller.set_publish_output(True)

        controller.set_publish_output(False)

        assert not controller.publish_output
        assert not sink.is_running
        assert sink.device_name is None

    def test_an_unavailable_sink_leaves_publishing_off(
        self, app_context: ApplicationContext, monkeypatch
    ):
        """A failed switch-on must not leave the controller believing it published."""
        controller = app_context.controller
        monkeypatch.setattr(NullVirtualCameraSink, "is_available", property(lambda self: False))
        controller.start()

        with pytest.raises(VirtualCameraError, match="not available"):
            controller.set_publish_output(True)

        assert not controller.publish_output
        assert not controller.components.virtual_camera.is_running


class TestTrackingContract:
    def test_tracker_output_is_shaped_as_declared(self, app_context: ApplicationContext):
        components = app_context.controller.components
        components.camera.start()
        components.tracker.start()

        frame = None
        while frame is None:
            frame = components.camera.read()

        result = components.tracker.process(frame)

        assert result.face_detected
        assert result.metadata.frame_id == frame.metadata.frame_id
        assert result.landmarks is not None
        assert result.landmarks.scheme == components.tracker.landmark_scheme
        assert result.blendshapes is not None
        assert components.tracker.provides_blendshapes


def _step_until_frame(context: ApplicationContext, timeout: float = 2.0):
    """Step the pipeline until a frame appears, or fail after ``timeout``.

    The placeholder camera rate-limits to its target FPS exactly as a real
    device would, so a step can legitimately return ``None`` when the next
    frame is not yet due. Polling with a short sleep rather than spinning keeps
    the suite from burning a core to wait 33 ms.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = context.controller.process_once()
        if frame is not None:
            return frame
        time.sleep(0.002)
    pytest.fail(f"pipeline produced no frame within {timeout}s")
