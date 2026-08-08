"""Tests for the OpenCV capture backend.

No camera is required. The device is faked so that error handling, format
negotiation, and frame construction are all tested — those are the parts most
likely to break, and the parts hardest to test with real hardware because a real
camera rarely misbehaves on demand.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest

from sha256.camera.devices import CameraDeviceInfo
from sha256.camera.opencv_capture import OpenCVCameraProvider
from sha256.core.enums import PixelFormat
from sha256.core.errors import CameraError
from sha256.core.interfaces import CameraProvider


class FakeCapture:
    """Stands in for ``cv2.VideoCapture``.

    Models a camera that supports exactly **one** resolution and silently
    ignores requests for anything else — which is how real UVC devices behave.
    An obliging fake that accepts every resolution would hide the bug this
    backend most needs to avoid: trusting the request instead of the result.
    """

    def __init__(
        self,
        *,
        opened: bool = True,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        fail_after: int | None = None,
        never_delivers: bool = False,
    ) -> None:
        self._opened = opened
        self._supported_width = width
        self._supported_height = height
        self._props: dict[int, float] = {
            cv2.CAP_PROP_FRAME_WIDTH: float(width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(height),
            cv2.CAP_PROP_FPS: fps,
        }
        self._fail_after = fail_after
        self._never_delivers = never_delivers
        self.reads = 0
        self.released = False
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the cv2 API
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.reads += 1
        if self._never_delivers:
            return False, None
        if self._fail_after is not None and self.reads > self._fail_after:
            return False, None
        # Frames always come back at the one resolution the device supports.
        return True, np.full(
            (self._supported_height, self._supported_width, 3), 128, dtype=np.uint8
        )

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            return False  # unsupported mode: request accepted, silently ignored
        self._props[prop] = value
        return True

    def get(self, prop: int) -> float:
        return self._props.get(prop, 0.0)

    def release(self) -> None:
        self.released = True
        self._opened = False


@pytest.fixture
def fake_capture(monkeypatch: pytest.MonkeyPatch):
    """Patch cv2.VideoCapture. Returns a list of every instance created."""
    created: list[FakeCapture] = []
    config: dict[str, Any] = {}

    def factory(index: int, backend: int) -> FakeCapture:
        del index, backend
        capture = FakeCapture(**config)
        created.append(capture)
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    return created, config


class TestProtocolConformance:
    def test_satisfies_the_camera_provider_protocol(self):
        assert isinstance(OpenCVCameraProvider(), CameraProvider)

    def test_backend_id(self):
        assert OpenCVCameraProvider().backend_id == "opencv"


class TestStart:
    def test_opens_and_reports_negotiated_format(self, fake_capture):
        created, config = fake_capture
        config.update(width=1920, height=1080, fps=60.0)

        provider = OpenCVCameraProvider(width=1920, height=1080, target_fps=60.0)
        provider.start()

        assert provider.is_running
        assert (provider.actual_width, provider.actual_height) == (1920, 1080)
        assert len(created) == 1

    def test_reports_what_the_camera_gave_not_what_was_asked(self, fake_capture):
        """Cameras snap to their nearest supported mode; report the truth."""
        _, config = fake_capture
        config.update(width=640, height=480)

        provider = OpenCVCameraProvider(width=1920, height=1080)
        provider.start()

        assert (provider.actual_width, provider.actual_height) == (640, 480)

    def test_requests_mjpeg_before_resolution(self, fake_capture):
        """Order matters: some drivers reset resolution when the codec changes."""
        created, _ = fake_capture

        OpenCVCameraProvider(use_mjpeg=True).start()

        props = [prop for prop, _ in created[0].set_calls]
        assert cv2.CAP_PROP_FOURCC in props
        assert props.index(cv2.CAP_PROP_FOURCC) < props.index(cv2.CAP_PROP_FRAME_WIDTH)

    def test_mjpeg_can_be_disabled(self, fake_capture):
        created, _ = fake_capture

        OpenCVCameraProvider(use_mjpeg=False).start()

        assert cv2.CAP_PROP_FOURCC not in [prop for prop, _ in created[0].set_calls]

    def test_requests_a_shallow_driver_buffer(self, fake_capture):
        """A deep queue means every frame read is already stale."""
        created, _ = fake_capture

        OpenCVCameraProvider().start()

        assert (cv2.CAP_PROP_BUFFERSIZE, 1) in created[0].set_calls

    def test_starting_twice_is_idempotent(self, fake_capture):
        created, _ = fake_capture
        provider = OpenCVCameraProvider()

        provider.start()
        provider.start()

        assert len(created) == 1


class TestStartFailures:
    def test_device_that_will_not_open_raises(self, fake_capture):
        _, config = fake_capture
        config.update(opened=False)

        with pytest.raises(CameraError, match="could not open camera"):
            OpenCVCameraProvider(device_index=7).start()

    def test_error_names_the_index_and_backends_tried(self, fake_capture):
        _, config = fake_capture
        config.update(opened=False)

        with pytest.raises(CameraError) as exc_info:
            OpenCVCameraProvider(device_index=3).start()

        message = str(exc_info.value)
        assert "3" in message
        assert "MSMF" in message and "DirectShow" in message

    def test_device_that_opens_but_never_delivers_raises(self, fake_capture):
        """The classic symptom of a camera held by another application."""
        created, config = fake_capture
        config.update(never_delivers=True)

        with pytest.raises(CameraError, match="delivered no frames"):
            OpenCVCameraProvider().start()

        assert created[0].released, "the device must be released before raising"


class TestRead:
    def test_returns_a_bgr_frame(self, fake_capture):
        _, config = fake_capture
        config.update(width=320, height=240)

        provider = OpenCVCameraProvider(width=320, height=240)
        provider.start()
        frame = provider.read()

        assert frame is not None
        assert frame.pixel_format is PixelFormat.BGR24
        assert frame.resolution == (320, 240)

    def test_frame_ids_increment_from_zero(self, fake_capture):
        provider = OpenCVCameraProvider()
        provider.start()

        ids = [provider.read().metadata.frame_id for _ in range(3)]  # type: ignore[union-attr]

        assert ids == [0, 1, 2]

    def test_timestamps_are_monotonic(self, fake_capture):
        provider = OpenCVCameraProvider()
        provider.start()

        first = provider.read()
        second = provider.read()

        assert first is not None and second is not None
        assert second.metadata.capture_timestamp >= first.metadata.capture_timestamp

    def test_reading_before_start_returns_none(self):
        assert OpenCVCameraProvider().read() is None

    def test_a_single_failed_read_returns_none(self, fake_capture):
        """One dropped frame is a USB hiccup, not a broken device."""
        _, config = fake_capture
        config.update(fail_after=1)

        provider = OpenCVCameraProvider()
        provider.start()  # consumes one read for verification

        assert provider.read() is None

    def test_sustained_failure_raises(self, fake_capture):
        _, config = fake_capture
        config.update(fail_after=1)

        provider = OpenCVCameraProvider()
        provider.start()

        with pytest.raises(CameraError, match="stopped delivering"):
            for _ in range(40):
                provider.read()

    def test_failure_counter_resets_on_success(self, fake_capture):
        provider = OpenCVCameraProvider()
        provider.start()

        for _ in range(50):
            assert provider.read() is not None


class TestStop:
    def test_releases_the_device(self, fake_capture):
        created, _ = fake_capture
        provider = OpenCVCameraProvider()
        provider.start()

        provider.stop()

        assert created[0].released
        assert not provider.is_running

    def test_stopping_a_stopped_provider_is_safe(self):
        OpenCVCameraProvider().stop()


class TestDeviceInfo:
    def test_describe_includes_what_is_known(self):
        info = CameraDeviceInfo(
            device_id="usb#vid_054c",
            display_name="Sony ZV-E10",
            index=0,
            max_resolution=(1920, 1080),
            max_frame_rate=30.0,
        )
        assert info.describe() == "Sony ZV-E10 — 1920x1080 — 30fps"

    def test_describe_degrades_when_details_are_missing(self):
        info = CameraDeviceInfo(device_id="x", display_name="Unknown Camera", index=1)
        assert info.describe() == "Unknown Camera"
