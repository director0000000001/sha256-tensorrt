"""The controller's one frame, end to end through fakes (day 15): what goes to
the sink is what the conditioner made of the render, the preview gets the
same frame, and a failing loop leaves the black screen on the camera."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sha256.core.enums import PixelFormat
from sha256.core.types import (
    FaceTrackingResult,
    FrameMetadata,
    LandmarkSet,
    RenderedFrame,
    VideoFrame,
)
from sha256.performance.monitor import NullPerformanceMonitor
from sha256.pipeline.conditioning import OutputConditioner
from sha256.pipeline.controller import PipelineComponents, PipelineController
from sha256.rendering.null import NullRenderer
from sha256.virtual_camera.null import NullVirtualCameraSink


class Camera:
    def __init__(self, width: int = 128, height: int = 72) -> None:
        self.data = np.full((height, width, 3), 200, dtype=np.uint8)
        self.index = 0

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def read(self) -> VideoFrame:
        self.index += 1
        return VideoFrame(
            data=self.data,
            pixel_format=PixelFormat.BGR24,
            metadata=FrameMetadata(frame_id=self.index, capture_timestamp=float(self.index)),
        )


class Tracker:
    def __init__(self) -> None:
        self.face = True

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def process(self, frame: VideoFrame) -> FaceTrackingResult:
        if not self.face:
            return FaceTrackingResult.no_face(frame.metadata)
        return FaceTrackingResult(
            metadata=frame.metadata,
            face_detected=True,
            landmarks=LandmarkSet(
                points=np.zeros((478, 3), dtype=np.float32), scheme="mediapipe_478"
            ),
        )


class Engine:
    """Draws a mid-grey face over a tracked frame; hands the camera's own
    picture back, flagged, when there is none -- the neural engine's way."""

    def animate(self, avatar_id: str, tracking: FaceTrackingResult, frame: VideoFrame):
        if not tracking.face_detected:
            return RenderedFrame(
                data=frame.data,
                pixel_format=frame.pixel_format,
                metadata=frame.metadata,
                avatar_id=avatar_id,
                render_timestamp=frame.metadata.capture_timestamp + 0.01,
                is_placeholder=True,
                is_passthrough=True,
            )
        data = np.full_like(frame.data, 120)
        return RenderedFrame(
            data=data,
            pixel_format=frame.pixel_format,
            metadata=frame.metadata,
            avatar_id=avatar_id,
            render_timestamp=frame.metadata.capture_timestamp + 0.01,
        )


class Clock:
    t = 10.0

    def __call__(self) -> float:
        return self.t


def build(*, conditioner: OutputConditioner, publish: bool = True):
    camera, tracker = Camera(), Tracker()
    sink = NullVirtualCameraSink()
    sink.configure(128, 72, 30.0)
    components = PipelineComponents(
        camera=camera,
        tracker=tracker,
        renderer=NullRenderer(width=128, height=72),
        virtual_camera=sink,
        avatars=SimpleNamespace(engine=Engine(), active=SimpleNamespace(id="me")),
        monitor=NullPerformanceMonitor(),
        conditioner=conditioner,
    )
    controller = PipelineController(components, publish_output=publish)  # type: ignore[arg-type]
    controller.start()
    return controller, tracker, sink


class TestOneFrame:
    def test_the_avatar_reaches_the_sink_and_the_preview_at_the_chosen_quality(self):
        clock = Clock()
        controller, _, sink = build(conditioner=OutputConditioner(quality="360p", clock=clock))
        rendered = controller.process_once()
        assert rendered is not None and sink.frames_sent == 1
        assert sink.last_frame is rendered and controller.last_frame is rendered
        assert rendered.avatar_id == "me" and not rendered.is_placeholder
        assert rendered.data.shape == (72, 128, 3)
        controller.stop()

    def test_the_cameras_own_picture_never_reaches_the_sink(self):
        clock = Clock()
        controller, tracker, sink = build(conditioner=OutputConditioner(clock=clock))
        tracker.face = False
        rendered = controller.process_once()
        assert rendered is not None
        assert rendered.is_placeholder and rendered.avatar_id is None
        assert (rendered.data == 0).mean() > 0.9  # black, not the camera's 200s
        assert sink.last_frame is rendered
        assert controller.components.conditioner.blackout is not None
        controller.stop()

    def test_the_manual_switch_blacks_the_sink_and_the_preview(self):
        clock = Clock()
        conditioner = OutputConditioner(clock=clock)
        controller, _, sink = build(conditioner=conditioner)
        conditioner.set_manual(True)
        rendered = controller.process_once()
        assert rendered is not None and rendered.is_placeholder
        assert (sink.last_frame.data == 0).mean() > 0.9
        conditioner.set_manual(False)
        rendered = controller.process_once()
        assert rendered is not None and not rendered.is_placeholder
        controller.stop()

    def test_a_start_forgets_an_automatic_black_screen(self):
        clock = Clock()
        conditioner = OutputConditioner(clock=clock)
        controller, tracker, _ = build(conditioner=conditioner)
        tracker.face = False
        controller.process_once()
        assert conditioner.blackout is not None
        controller.stop()
        tracker.face = True
        controller.start()
        assert conditioner.blackout is None
        rendered = controller.process_once()
        assert rendered is not None and not rendered.is_placeholder
        controller.stop()


class TestPublishFailure:
    def test_the_black_screen_goes_out_once_when_the_loop_dies(self):
        clock = Clock()
        controller, _, sink = build(conditioner=OutputConditioner(clock=clock))
        controller.process_once()
        assert sink.frames_sent == 1
        assert controller.publish_failure("camera exploded") is True
        assert sink.frames_sent == 2
        assert sink.last_frame.is_placeholder and (sink.last_frame.data == 0).mean() > 0.9
        assert sink.last_frame.data.shape == (72, 128, 3)
        assert controller.last_frame is sink.last_frame
        controller.stop()

    def test_nothing_to_publish_without_a_sink_or_a_first_frame(self):
        clock = Clock()
        controller, _, sink = build(conditioner=OutputConditioner(clock=clock), publish=False)
        assert controller.publish_failure("x") is False
        assert sink.frames_sent == 0
        controller.stop()
        controller2, _, sink2 = build(conditioner=OutputConditioner(clock=clock))
        assert controller2.publish_failure("x") is False  # no frame yet: size unknown
        assert sink2.frames_sent == 0
        controller2.stop()
