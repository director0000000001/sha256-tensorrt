"""Basic CPU compositor.

Letterboxes engine output into the configured delivery resolution and can draw
a placeholder frame for idle and error states.

Sufficient for the foundation and for tests; a GPU compositing path is a later
decision recorded in docs/decisions/.
"""

from __future__ import annotations

import numpy as np

from sha256.core.enums import PixelFormat
from sha256.core.types import FrameMetadata, RenderedFrame
from sha256.utils.timing import now

__all__ = ["NullRenderer"]


class NullRenderer:
    """Fits a source frame into the output canvas, preserving aspect ratio."""

    def __init__(self, *, width: int = 1280, height: int = 720) -> None:
        self._width = width
        self._height = height
        self._running = False
        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)

    @property
    def backend_id(self) -> str:
        return "null"

    @property
    def output_format(self) -> str:
        return PixelFormat.RGB24.value

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def configure(self, width: int, height: int) -> None:
        if (width, height) == (self._width, self._height):
            return
        self._width = width
        self._height = height
        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)

    def render(self, source: RenderedFrame) -> RenderedFrame:
        self._canvas.fill(16)
        self._blit_centred(source.data)
        return RenderedFrame(
            data=self._canvas,
            pixel_format=PixelFormat.RGB24,
            metadata=source.metadata,
            avatar_id=source.avatar_id,
            render_timestamp=now(),
            is_placeholder=source.is_placeholder,
        )

    def render_placeholder(self, metadata: FrameMetadata, message: str) -> RenderedFrame:
        """Solid frame used when no avatar is active.

        ``message`` is accepted but not drawn: text rendering needs a font stack
        this layer does not have yet, and drawing nothing is better than
        pretending. The UI shows the message; the frame keeps the stream alive.
        """
        del message
        self._canvas.fill(16)
        return RenderedFrame(
            data=self._canvas,
            pixel_format=PixelFormat.RGB24,
            metadata=metadata,
            avatar_id=None,
            render_timestamp=now(),
            is_placeholder=True,
        )

    def _blit_centred(self, source: np.ndarray) -> None:
        """Copy ``source`` into the canvas centre, cropping if it is larger.

        No scaling: resampling belongs in a real renderer where the filter
        choice is a considered quality decision, not an incidental one.
        """
        src_h, src_w = source.shape[:2]
        copy_h = min(src_h, self._height)
        copy_w = min(src_w, self._width)

        src_y = (src_h - copy_h) // 2
        src_x = (src_w - copy_w) // 2
        dst_y = (self._height - copy_h) // 2
        dst_x = (self._width - copy_w) // 2

        self._canvas[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = source[
            src_y : src_y + copy_h, src_x : src_x + copy_w
        ]
