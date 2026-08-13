"""Virtual camera sinks.

The pyvirtualcam sink is tested against a fake ``pyvirtualcam`` module rather
than a real device. Opening the real one advertises a camera to every
application on the machine, and a test suite that makes a webcam appear in Zoom
for a few milliseconds is a test suite nobody will run. The device itself is
exercised by ``scripts/benchmark_virtual_camera.py`` and by the application.

What is worth asserting here is everything that goes wrong *around* the device:
the format the pipeline actually hands over, the size mismatch, the frame that
arrives after stop. Those are the paths a real device makes hardest to reach.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from sha256.core.enums import PixelFormat
from sha256.core.errors import VirtualCameraError
from sha256.core.types import FrameMetadata, RenderedFrame
from sha256.virtual_camera.devices import (
    SUPPORTED_DEVICES,
    registered_capture_filters,
    supported_devices_present,
)
from sha256.virtual_camera.null import NullVirtualCameraSink
from sha256.virtual_camera.pyvirtualcam_sink import PyVirtualCameraSink


def make_frame(
    width: int = 8,
    height: int = 4,
    pixel_format: PixelFormat = PixelFormat.BGR24,
    fill: tuple[int, int, int] = (10, 20, 30),
) -> RenderedFrame:
    data = np.zeros((height, width, 3), dtype=np.uint8)
    data[:, :] = fill
    return RenderedFrame(
        data=data,
        pixel_format=pixel_format,
        metadata=FrameMetadata(frame_id=1, capture_timestamp=0.0, source_id="test"),
    )


class FakeCamera:
    """Stands in for ``pyvirtualcam.Camera``, including its shape check."""

    def __init__(self, *, width, height, fps, fmt, device, backend, print_fps):
        self.width = width
        self.height = height
        self.fps = fps
        self.fmt = fmt
        self.backend = backend or "obs"
        self.device = device or "OBS Virtual Camera"
        self.sent: list[np.ndarray] = []
        self.closed = False

    def send(self, frame: np.ndarray) -> None:
        expected = (self.height, self.width, 3)
        if frame.shape != expected:
            raise ValueError(f"unexpected frame shape: {frame.shape} != {expected}")
        self.sent.append(frame.copy())

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_pyvirtualcam(monkeypatch):
    """Install a fake ``pyvirtualcam`` module for the duration of one test."""
    module = types.ModuleType("pyvirtualcam")
    module.Camera = FakeCamera  # type: ignore[attr-defined]
    module.PixelFormat = types.SimpleNamespace(BGR="24BG", RGB="raw ")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyvirtualcam", module)
    return module


@pytest.fixture
def sink(fake_pyvirtualcam, monkeypatch):
    """A started sink bound to the fake device."""
    monkeypatch.setattr(
        "sha256.virtual_camera.pyvirtualcam_sink.supported_devices_present", lambda: ("obs",)
    )
    instance = PyVirtualCameraSink()
    instance.configure(8, 4, 30.0)
    instance.start()
    yield instance
    instance.stop()


class TestNullSink:
    def test_refuses_to_start_before_being_configured(self):
        """The ordering rule the real backends inherit, enforced on the cheap one."""
        with pytest.raises(VirtualCameraError, match="configure"):
            NullVirtualCameraSink().start()

    def test_counts_frames_and_ignores_them_while_stopped(self):
        instance = NullVirtualCameraSink()
        instance.configure(8, 4, 30.0)

        instance.send(make_frame())
        assert instance.frames_sent == 0

        instance.start()
        instance.send(make_frame())
        assert instance.frames_sent == 1


class TestDeviceDetection:
    def test_reports_names_without_opening_anything(self):
        """Whatever this machine has, the scan must not raise or hang."""
        assert isinstance(registered_capture_filters(), tuple)

    def test_supported_devices_are_a_subset_of_what_is_registered(self):
        installed = set(registered_capture_filters())
        for backend in supported_devices_present():
            assert SUPPORTED_DEVICES[backend] in installed


class TestPyVirtualCameraSinkLifecycle:
    def test_refuses_to_start_before_being_configured(self, fake_pyvirtualcam):
        with pytest.raises(VirtualCameraError, match="configure"):
            PyVirtualCameraSink().start()

    @pytest.mark.parametrize(
        ("width", "height", "fps"),
        [(0, 4, 30.0), (8, -1, 30.0), (8, 4, 0.0)],
    )
    def test_rejects_a_format_no_device_could_honour(self, width, height, fps):
        with pytest.raises(VirtualCameraError):
            PyVirtualCameraSink().configure(width, height, fps)

    def test_format_cannot_change_while_running(self, sink):
        with pytest.raises(VirtualCameraError, match="cannot change while running"):
            sink.configure(16, 8, 30.0)

    def test_reports_the_device_name_the_os_gave_it_not_the_one_we_asked_for(self, sink):
        assert sink.device_name == "OBS Virtual Camera"

    def test_device_name_is_none_while_stopped(self, sink):
        sink.stop()
        assert sink.device_name is None

    def test_stop_closes_the_device_and_is_idempotent(self, sink):
        camera = sink._camera
        sink.stop()
        sink.stop()
        assert camera.closed
        assert not sink.is_running

    def test_start_failure_names_what_to_install(self, fake_pyvirtualcam, monkeypatch):
        monkeypatch.setattr(
            "sha256.virtual_camera.pyvirtualcam_sink.supported_devices_present", lambda: ()
        )

        def explode(**kwargs):
            raise RuntimeError("no device")

        monkeypatch.setattr(fake_pyvirtualcam, "Camera", explode)

        instance = PyVirtualCameraSink()
        instance.configure(8, 4, 30.0)
        with pytest.raises(VirtualCameraError, match="OBS Studio"):
            instance.start()
        assert not instance.is_running

    def test_unavailable_when_no_device_is_installed(self, monkeypatch):
        monkeypatch.setattr(
            "sha256.virtual_camera.pyvirtualcam_sink.supported_devices_present", lambda: ()
        )
        assert not PyVirtualCameraSink().is_available


class TestPyVirtualCameraSinkSending:
    def test_publishes_a_bgr_frame_unchanged(self, sink):
        sink.send(make_frame(fill=(10, 20, 30)))

        assert sink.frames_sent == 1
        assert sink.frames_dropped == 0
        np.testing.assert_array_equal(sink._camera.sent[0][0, 0], [10, 20, 30])

    def test_converts_an_rgb_frame_rather_than_publishing_a_blue_face(self, sink):
        """The renderer preserves its input format, so both really do arrive."""
        sink.send(make_frame(pixel_format=PixelFormat.RGB24, fill=(10, 20, 30)))

        assert sink.frames_sent == 1
        np.testing.assert_array_equal(sink._camera.sent[0][0, 0], [30, 20, 10])

    def test_drops_a_frame_of_the_wrong_size(self, sink):
        sink.send(make_frame(width=16, height=8))

        assert sink.frames_sent == 0
        assert sink.frames_dropped == 1
        assert "16x8" in sink.last_drop_reason

    def test_drops_a_format_it_cannot_publish(self, sink):
        frame = make_frame()
        unsendable = RenderedFrame(
            data=frame.data,
            pixel_format=PixelFormat.GRAY8,
            metadata=frame.metadata,
        )

        sink.send(unsendable)

        assert sink.frames_sent == 0
        assert "gray8" in sink.last_drop_reason

    def test_a_frame_the_device_rejects_is_dropped_not_raised(self, sink):
        """The caller is the capture thread; a throwing sink would stop the pipeline."""

        def explode(frame):
            raise RuntimeError("device went away")

        sink._camera.send = explode

        sink.send(make_frame())

        assert sink.frames_dropped == 1
        assert "device went away" in sink.last_drop_reason

    def test_sending_while_stopped_does_nothing(self, sink):
        sink.stop()
        sink.send(make_frame())

        assert sink.frames_sent == 0
        assert sink.frames_dropped == 0

    def test_counters_reset_on_restart(self, sink):
        sink.send(make_frame())
        sink.send(make_frame(width=16, height=8))
        assert (sink.frames_sent, sink.frames_dropped) == (1, 1)

        sink.stop()
        sink.start()

        assert (sink.frames_sent, sink.frames_dropped) == (0, 0)
        assert sink.last_drop_reason is None
