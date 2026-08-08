"""Webcam capture via OpenCV.

Targets UVC devices, which is what the Sony ZV-E10 presents as in USB-streaming
mode — the same path as any ordinary webcam, so nothing here is camera-specific.

Design notes
------------
*   **Frames are emitted as BGR24**, OpenCV's native layout, with no conversion.
    Converting here would cost a full-frame pass on every frame for the benefit
    of whichever consumer happens to want RGB. Consumers that need a different
    layout convert in their own adapter — the same rule every backend follows.

*   **Requested settings are not assumed to be honoured.** Cameras silently snap
    to the nearest supported mode, or ignore the request entirely. Everything is
    read back after opening and the *actual* values are what get reported.

*   **MJPEG is requested by default.** Most UVC cameras are limited to a few fps
    at higher resolutions on raw YUY2, and only reach 30 fps with MJPEG. This is
    the single setting most likely to decide whether 1080p30 works at all.

*   ``read()`` returns a freshly allocated array, so unlike the placeholder
    backend these frames own their buffer and need no ``.copy()``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import cv2
import numpy as np

from sha256.camera.devices import CameraDeviceInfo, enumerate_cameras
from sha256.core.enums import PixelFormat
from sha256.core.errors import CameraError
from sha256.core.types import FrameMetadata, VideoFrame
from sha256.utils.logging import get_logger
from sha256.utils.timing import now

__all__ = ["OpenCVCameraProvider"]

_LOG = get_logger(__name__)

# Media Foundation first: it is the modern Windows path and generally reports
# formats more honestly. DirectShow is the fallback because some devices and
# some driver versions still only work there.
_BACKEND_PREFERENCE: tuple[tuple[int, str], ...] = (
    (cv2.CAP_MSMF, "MSMF"),
    (cv2.CAP_DSHOW, "DirectShow"),
)

# OpenCV 5 exposes this on the class; the module-level alias still exists but is
# absent from the type stubs.
_MJPEG_FOURCC = cv2.VideoWriter.fourcc(*"MJPG")


class OpenCVCameraProvider:
    """Captures frames from a physical camera."""

    def __init__(
        self,
        *,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: float = 30.0,
        use_mjpeg: bool = True,
        source_id: str | None = None,
    ) -> None:
        self._requested_index = device_index
        self._requested_width = width
        self._requested_height = height
        self._requested_fps = target_fps
        self._use_mjpeg = use_mjpeg
        self._source_id = source_id or f"camera:{device_index}"

        self._capture: cv2.VideoCapture | None = None
        self._backend_name = ""
        self._frame_id = 0
        self._consecutive_failures = 0

        # Negotiated values, filled in by start(). Zero until then.
        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0.0

    # --- Component ---------------------------------------------------------

    @property
    def backend_id(self) -> str:
        return "opencv"

    @property
    def is_running(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def device_index(self) -> int:
        return self._requested_index

    @property
    def backend_name(self) -> str:
        """Which OS capture API actually opened the device."""
        return self._backend_name

    def start(self) -> None:
        if self.is_running:
            return

        capture, backend_name = self._open_device()
        self._capture = capture
        self._backend_name = backend_name

        self._configure(capture)
        self._read_back_actuals(capture)
        self._verify_first_frame(capture)

        self._frame_id = 0
        self._consecutive_failures = 0

        _LOG.info(
            "Camera %d opened via %s: %dx%d @ %.1f fps (requested %dx%d @ %.1f)",
            self._requested_index,
            backend_name,
            self.actual_width,
            self.actual_height,
            self.actual_fps,
            self._requested_width,
            self._requested_height,
            self._requested_fps,
        )

        if (self.actual_width, self.actual_height) != (
            self._requested_width,
            self._requested_height,
        ):
            _LOG.warning(
                "Camera negotiated %dx%d instead of the requested %dx%d",
                self.actual_width,
                self.actual_height,
                self._requested_width,
                self._requested_height,
            )

    def _open_device(self) -> tuple[cv2.VideoCapture, str]:
        """Try each OS backend in turn. Raises if none opens the device."""
        attempts: list[str] = []

        for backend_flag, backend_name in _BACKEND_PREFERENCE:
            capture = cv2.VideoCapture(self._requested_index, backend_flag)
            if capture.isOpened():
                return capture, backend_name
            capture.release()
            attempts.append(backend_name)
            _LOG.debug("Camera %d did not open via %s", self._requested_index, backend_name)

        available = enumerate_cameras()
        detail = (
            ", ".join(f"[{d.index}] {d.display_name}" for d in available)
            if available
            else "no cameras detected"
        )
        msg = (
            f"could not open camera index {self._requested_index} "
            f"(tried {', '.join(attempts)}). Available: {detail}"
        )
        raise CameraError(msg)

    def _configure(self, capture: cv2.VideoCapture) -> None:
        """Request format. Order matters — FOURCC before resolution."""
        if self._use_mjpeg:
            # Set before resolution: on several drivers, changing the codec
            # afterwards resets the resolution back to a default.
            capture.set(cv2.CAP_PROP_FOURCC, _MJPEG_FOURCC)

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
        capture.set(cv2.CAP_PROP_FPS, self._requested_fps)

        # A small driver-side buffer keeps latency down: a deep queue means the
        # frame just read is already several frames old.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _read_back_actuals(self, capture: cv2.VideoCapture) -> None:
        self.actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = float(capture.get(cv2.CAP_PROP_FPS))

    def _verify_first_frame(self, capture: cv2.VideoCapture) -> None:
        """Pull one frame before declaring success.

        ``isOpened()`` can be true for a device that then never delivers — a
        camera held by another application is the usual cause. Failing here
        gives a clear message instead of an empty preview.
        """
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            self._capture = None
            msg = (
                f"camera {self._requested_index} opened but delivered no frames. "
                f"It may be in use by another application."
            )
            raise CameraError(msg)

        # Trust the delivered frame over the driver's own property reporting,
        # which is not always accurate.
        self.actual_height, self.actual_width = frame.shape[:2]

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._backend_name = ""

    # --- CameraProvider ----------------------------------------------------

    def enumerate_devices(self) -> Sequence[CameraDeviceInfo]:
        return enumerate_cameras()

    def read(self) -> VideoFrame | None:
        """Grab the next frame, or ``None`` if this one failed.

        A single failed read is normal — a USB hiccup, a mode change. Several in
        a row means the device is gone, which is an error worth raising.
        """
        capture = self._capture
        if capture is None:
            return None

        captured_at = now()
        ok, data = capture.read()

        if not ok or data is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 30:
                msg = (
                    f"camera {self._requested_index} stopped delivering frames "
                    f"({self._consecutive_failures} consecutive failures)"
                )
                raise CameraError(msg)
            return None

        self._consecutive_failures = 0

        metadata = FrameMetadata(
            frame_id=self._frame_id,
            capture_timestamp=captured_at,
            source_id=self._source_id,
        )
        self._frame_id += 1

        return VideoFrame(
            data=np.asarray(data, dtype=np.uint8),
            pixel_format=PixelFormat.BGR24,
            metadata=metadata,
        )

    def frames(self) -> Iterator[VideoFrame]:
        while self.is_running:
            frame = self.read()
            if frame is not None:
                yield frame
