"""Virtual camera output through pyvirtualcam.

**This backend is for development, not for distribution.** pyvirtualcam is
GPL-2.0 — verified from the installed package's own metadata, which carries
``License :: OSI Approved :: GNU General Public License v2 (GPLv2)`` and a full
GPL-2 text. Linking it into a distributed sha256 would place the whole work
under the GPL, which would oblige us to hand the client the source, which would
let them remove the licence server. The commercial protection and this library
cannot both survive, so it lives in the optional ``vcam-gpl`` extra and is never
installed by default. See ``docs/decisions/ADR-0005-virtual-camera-strategy.md``.

What it buys today is the whole point of day 5: the avatar becomes visible to
Meet, Zoom, Discord and OBS on this machine, so every quality decision from here
is judged where the product actually lives instead of inside our own preview.

**It writes to an existing device; it does not create one.** On Windows that
device comes from an OBS Studio installation (or UnityCapture). This is not an
implementation detail we can hide — it is the reason the shipping backend has to
be a different one.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from sha256.core.enums import PixelFormat
from sha256.core.errors import VirtualCameraError
from sha256.core.types import RenderedFrame
from sha256.utils.logging import get_logger
from sha256.virtual_camera.devices import (
    SUPPORTED_DEVICES,
    registered_capture_filters,
    supported_devices_present,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["PyVirtualCameraSink"]

_LOG = get_logger(__name__)

# Frames the pipeline can hand us. Everything else is a wiring mistake rather
# than a runtime condition, and is dropped loudly instead of published wrongly —
# publishing a BGR buffer down an RGB device is the blue-face bug the renderer's
# docstring already records once.
_SENDABLE = (PixelFormat.BGR24, PixelFormat.RGB24)

# Drops are logged on the first occurrence and then geometrically, so a sink
# that fails on every one of 30 frames a second cannot fill the log file while
# still making the first failure impossible to miss.
_DROP_LOG_STEPS = (1, 10, 100, 1000, 10000)


class PyVirtualCameraSink:
    """Publishes rendered frames to an existing Windows virtual camera device.

    The sink is configured in BGR because that is what the pipeline actually
    produces — the camera captures BGR, the warp engine composites in it, and
    the renderer preserves whatever it was handed. RGB frames (the idle
    placeholder is one) are converted; anything else is dropped and counted.
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        backend: str | None = None,
    ) -> None:
        """
        Args:
            device: Exact OS device name to bind to, e.g. ``"OBS Virtual
                Camera"``. ``None`` takes the first one that works. The name
                cannot be chosen by us — the device belongs to whichever product
                installed it — so the configured display name is deliberately
                not passed here.
            backend: pyvirtualcam backend id (``"obs"``, ``"unitycapture"``).
                ``None`` tries every one it has, and reports what each said if
                they all fail, which is a better diagnostic than our own guess.
        """
        self._device = device
        self._backend = backend
        self._camera: Any | None = None
        self._running = False
        self._configured = False
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self.frames_sent = 0
        self.frames_dropped = 0
        self.last_drop_reason: str | None = None

    @property
    def backend_id(self) -> str:
        return "pyvirtualcam"

    @property
    def is_available(self) -> bool:
        """Whether both the library and a device it can drive are present.

        Deliberately does not import pyvirtualcam: ``find_spec`` locates it
        without executing it, keeping the registry's lazy-import promise intact
        for a UI that asks this question on every status refresh.
        """
        if importlib.util.find_spec("pyvirtualcam") is None:
            return False
        return bool(supported_devices_present())

    @property
    def device_name(self) -> str | None:
        """The name the OS actually exposes, read back from the open device.

        Reported rather than configured. Asking for ``"sha256 Virtual Camera"``
        and being given ``"OBS Virtual Camera"`` is exactly the kind of quiet
        mismatch that makes a support conversation impossible.
        """
        if self._camera is None:
            return None
        name: str = self._camera.device
        return name

    @property
    def is_running(self) -> bool:
        return self._running

    def configure(self, width: int, height: int, fps: float) -> None:
        if self._running:
            msg = "virtual camera format cannot change while running"
            raise VirtualCameraError(msg)
        if width <= 0 or height <= 0:
            msg = f"virtual camera needs a positive frame size, got {width}x{height}"
            raise VirtualCameraError(msg)
        if fps <= 0:
            msg = f"virtual camera needs a positive frame rate, got {fps}"
            raise VirtualCameraError(msg)
        self._width = width
        self._height = height
        self._fps = fps
        self._configured = True

    def start(self) -> None:
        if self._running:
            return
        if not self._configured:
            msg = "configure() must be called before start()"
            raise VirtualCameraError(msg)

        try:
            import pyvirtualcam
        except ImportError as exc:  # pragma: no cover - depends on the install
            msg = (
                "pyvirtualcam is not installed. It is an optional extra because "
                "it is GPL-2.0: uv pip install pyvirtualcam"
            )
            raise VirtualCameraError(msg) from exc

        try:
            self._camera = pyvirtualcam.Camera(
                width=self._width,
                height=self._height,
                fps=self._fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
                device=self._device,
                backend=self._backend,
                print_fps=False,
            )
        except Exception as exc:
            raise VirtualCameraError(_start_failure_message(exc)) from exc

        self._running = True
        self.frames_sent = 0
        self.frames_dropped = 0
        self.last_drop_reason = None
        _LOG.info(
            "Virtual camera publishing to %r (%s backend) at %dx%d @ %.2f fps",
            self._camera.device,
            self._camera.backend,
            self._width,
            self._height,
            self._fps,
        )

    def stop(self) -> None:
        """Release the device. Never raises — a failed close must not strand the pipeline."""
        camera, self._camera = self._camera, None
        self._running = False
        if camera is None:
            return
        try:
            camera.close()
        except Exception:
            _LOG.exception("Error closing the virtual camera device")

    def send(self, frame: RenderedFrame) -> None:
        """Publish one frame, or drop it.

        Never raises. The caller is the capture thread: a sink that throws would
        stop the pipeline, and losing the preview because the virtual camera hit
        a bad frame is a worse outcome than losing that frame.
        """
        if not self._running or self._camera is None:
            return

        payload = self._payload(frame)
        if payload is None:
            return

        try:
            self._camera.send(payload)
        except Exception as exc:
            self._drop(f"device rejected the frame: {exc}")
            return

        self.frames_sent += 1

    # --- internals ---------------------------------------------------------

    def _payload(self, frame: RenderedFrame) -> NDArray[np.uint8] | None:
        """The frame as a contiguous BGR buffer of the configured size, or ``None``."""
        data = frame.data

        if data.shape[:2] != (self._height, self._width):
            self._drop(
                f"frame is {data.shape[1]}x{data.shape[0]}, device is {self._width}x{self._height}"
            )
            return None

        if frame.pixel_format not in _SENDABLE:
            self._drop(f"cannot publish {frame.pixel_format.value} frames")
            return None

        if data.dtype != np.uint8 or data.ndim != 3 or data.shape[2] != 3:
            self._drop(f"expected 3-channel uint8, got {data.shape} of {data.dtype}")
            return None

        if frame.pixel_format is PixelFormat.RGB24:
            return _swap_red_and_blue(data)
        return data if data.flags.c_contiguous else np.ascontiguousarray(data)

    def _drop(self, reason: str) -> None:
        self.frames_dropped += 1
        self.last_drop_reason = reason
        if self.frames_dropped in _DROP_LOG_STEPS:
            _LOG.warning(
                "Virtual camera dropped a frame (%d so far): %s", self.frames_dropped, reason
            )


def _swap_red_and_blue(data: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """RGB to BGR, using OpenCV when it is there.

    The numpy fallback is a reversed-stride view that has to be materialised,
    which costs several times what OpenCV's conversion does. OpenCV is present
    in every real configuration — the camera and the warp engine both need it —
    so the fallback exists only to keep this module importable without it.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - cv2 is installed on the target
        return np.ascontiguousarray(data[:, :, ::-1])
    return cast("NDArray[np.uint8]", cv2.cvtColor(data, cv2.COLOR_RGB2BGR))


def _start_failure_message(exc: Exception) -> str:
    """Turn pyvirtualcam's per-backend error list into something actionable.

    pyvirtualcam reports what each backend it tried complained about, which is
    accurate but says nothing about what to do. The two causes worth naming are
    the only two that occur in practice: no virtual camera is installed at all,
    or OBS Studio is running and holding the single device it provides.
    """
    if not supported_devices_present():
        installed = registered_capture_filters()
        expected = ", ".join(sorted(SUPPORTED_DEVICES.values()))
        found = ", ".join(installed) if installed else "none"
        return (
            f"No virtual camera device is installed. pyvirtualcam writes to an "
            f"existing device and cannot create one; it needs one of: {expected}. "
            f"Installing OBS Studio once registers it — OBS does not have to be "
            f"running. Capture filters found on this machine: {found}. ({exc})"
        )
    return (
        f"A virtual camera device exists but could not be opened. It provides a "
        f"single instance, so this usually means OBS Studio is running with its "
        f"virtual camera started — stop it there and try again. ({exc})"
    )
