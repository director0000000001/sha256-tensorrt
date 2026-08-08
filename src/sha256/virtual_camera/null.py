"""Virtual camera sink that publishes nowhere.

Counts frames and records the last one so the pipeline and its tests have a
real sink to write to. Nothing is exposed to the operating system.

No virtual camera is implemented yet — deliberately. The Windows options each
carry a licensing or deployment consequence that has to be decided before code
is written, not after; see
``docs/decisions/ADR-0005-virtual-camera-strategy.md``.
"""

from __future__ import annotations

from sha256.core.errors import VirtualCameraError
from sha256.core.types import RenderedFrame

__all__ = ["NullVirtualCameraSink"]


class NullVirtualCameraSink:
    """Accepts frames and discards them, tracking counters for diagnostics."""

    def __init__(self, *, device_name: str = "SHA-256 Virtual Camera (null)") -> None:
        self._device_name = device_name
        self._running = False
        self._configured = False
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self.frames_sent = 0
        self.last_frame: RenderedFrame | None = None

    @property
    def backend_id(self) -> str:
        return "null"

    @property
    def is_available(self) -> bool:
        """Always available — it depends on nothing."""
        return True

    @property
    def device_name(self) -> str | None:
        return self._device_name if self._running else None

    @property
    def is_running(self) -> bool:
        return self._running

    def configure(self, width: int, height: int, fps: float) -> None:
        if self._running:
            msg = "virtual camera format cannot change while running"
            raise VirtualCameraError(msg)
        self._width = width
        self._height = height
        self._fps = fps
        self._configured = True

    def start(self) -> None:
        if self._running:
            return
        if not self._configured:
            # A real sink must know its format before the OS advertises the
            # device; enforcing it here keeps that ordering bug out of the
            # pipeline rather than surfacing it only on the real backend.
            msg = "configure() must be called before start()"
            raise VirtualCameraError(msg)
        self._running = True
        self.frames_sent = 0

    def stop(self) -> None:
        self._running = False

    def send(self, frame: RenderedFrame) -> None:
        if not self._running:
            return
        self.frames_sent += 1
        self.last_frame = frame
