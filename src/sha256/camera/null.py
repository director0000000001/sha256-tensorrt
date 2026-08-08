"""Synthetic camera provider.

Not a stub that raises — a working :class:`~sha256.core.interfaces.CameraProvider`
that generates frames procedurally. That matters for three reasons:

*   the application shell runs and is demonstrable on a machine with no webcam;
*   integration tests exercise the real wiring deterministically, with no
    hardware and no flakiness;
*   when a real backend lands, any bug is immediately bisectable — swap to
    ``null`` and see whether the problem follows the camera.

Real capture backends (Media Foundation, OpenCV/DirectShow) are a later
milestone; see docs/architecture/SYSTEM_ARCHITECTURE.md.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from sha256.core.enums import PixelFormat
from sha256.core.types import FrameMetadata, VideoFrame
from sha256.utils.timing import now

__all__ = ["NullCameraProvider", "SyntheticCameraSpec"]


@dataclass(frozen=True, slots=True)
class SyntheticCameraSpec:
    """A fake device entry, satisfying the ``CameraSpec`` protocol."""

    device_id: str
    display_name: str


class NullCameraProvider:
    """Generates a moving gradient at a fixed resolution.

    The pattern animates so that a stalled pipeline is visually obvious — a
    frozen static image and a working-but-idle one look identical, which hides
    exactly the bug this is meant to help find.
    """

    def __init__(
        self,
        *,
        width: int = 1280,
        height: int = 720,
        target_fps: float = 30.0,
        source_id: str = "null-camera",
        device_index: int = 0,
        use_mjpeg: bool = True,
    ) -> None:
        # device_index and use_mjpeg are accepted and ignored: every camera
        # backend is constructed from the same settings, and filtering kwargs
        # per backend in the composition root would be worse than two unused
        # parameters here.
        del device_index, use_mjpeg
        self._width = width
        self._height = height
        self._target_fps = target_fps
        self._source_id = source_id
        self._running = False
        self._frame_id = 0
        self._started_at = 0.0
        self._last_frame_at = 0.0

        # Precomputed coordinate ramps: recomputing these per frame would make
        # the placeholder more expensive than a real camera read.
        self._x_ramp = np.linspace(0, 255, width, dtype=np.float32)
        self._y_ramp = np.linspace(0, 255, height, dtype=np.float32).reshape(-1, 1)
        self._buffer = np.zeros((height, width, 3), dtype=np.uint8)

    @property
    def backend_id(self) -> str:
        return "null"

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._frame_id = 0
        self._started_at = now()
        self._last_frame_at = 0.0

    def stop(self) -> None:
        self._running = False

    def enumerate_devices(self) -> Sequence[SyntheticCameraSpec]:
        return (SyntheticCameraSpec(device_id="null:0", display_name="Synthetic Camera"),)

    def read(self) -> VideoFrame | None:
        """Return a frame, or ``None`` when the next one is not yet due.

        Rate limiting lives here on purpose: it mimics how a real capture device
        behaves, so the consuming loop is written against realistic timing from
        the start rather than being tuned later against a source that never
        blocks.
        """
        if not self._running:
            return None

        current = now()
        interval = 1.0 / self._target_fps
        if self._last_frame_at and (current - self._last_frame_at) < interval:
            return None

        self._last_frame_at = current
        elapsed = current - self._started_at
        phase = (math.sin(elapsed * 1.5) + 1.0) * 0.5 * 255.0

        self._buffer[:, :, 0] = self._x_ramp.astype(np.uint8)
        self._buffer[:, :, 1] = self._y_ramp.astype(np.uint8)
        self._buffer[:, :, 2] = np.uint8(phase)

        metadata = FrameMetadata(
            frame_id=self._frame_id,
            capture_timestamp=current,
            source_id=self._source_id,
        )
        self._frame_id += 1

        # The buffer is reused across calls — this is the borrowed-buffer
        # convention documented on VideoFrame.
        return VideoFrame(data=self._buffer, pixel_format=PixelFormat.RGB24, metadata=metadata)

    def frames(self) -> Iterator[VideoFrame]:
        while self._running:
            frame = self.read()
            if frame is not None:
                yield frame
