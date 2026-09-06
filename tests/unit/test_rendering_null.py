"""Tests for the CPU compositor.

One thing is worth testing here and it is about ownership, not pixels. The
renderer's output is read by the preview on the UI thread and written to disk
by the capture button, while the pipeline thread is already rendering the next
frame. A frame that has been handed out must therefore never be written again
— the version that reused one canvas painted the preview flat grey for a frame
whenever the two threads overlapped.
"""

from __future__ import annotations

import numpy as np

from sha256.core.enums import PixelFormat
from sha256.core.types import FrameMetadata, RenderedFrame
from sha256.rendering.null import NullRenderer


def source_frame(frame_id: int, value: int) -> RenderedFrame:
    return RenderedFrame(
        data=np.full((8, 12, 3), value, dtype=np.uint8),
        pixel_format=PixelFormat.BGR24,
        metadata=FrameMetadata(frame_id=frame_id, capture_timestamp=float(frame_id)),
    )


class TestOwnership:
    def test_a_second_render_does_not_mutate_the_first_frame(self):
        renderer = NullRenderer(width=12, height=8)

        first = renderer.render(source_frame(1, 200)).data
        before = first.copy()
        renderer.render(source_frame(2, 50))

        assert np.array_equal(first, before)

    def test_consecutive_frames_do_not_share_memory(self):
        renderer = NullRenderer(width=12, height=8)

        first = renderer.render(source_frame(1, 200)).data
        second = renderer.render(source_frame(2, 50)).data

        assert not np.shares_memory(first, second)

    def test_the_placeholder_does_not_share_memory_with_a_frame(self):
        renderer = NullRenderer(width=12, height=8)
        metadata = FrameMetadata(frame_id=3, capture_timestamp=3.0)

        frame = renderer.render(source_frame(1, 200)).data
        placeholder = renderer.render_placeholder(metadata, "idle").data
        renderer.render(source_frame(2, 50))

        assert not np.shares_memory(frame, placeholder)
        assert int(placeholder.max()) == 16

    def test_a_smaller_source_is_letterboxed_on_grey(self):
        """The canvas is fresh memory each frame, so the border around a
        picture smaller than the output has to be painted, not assumed."""
        renderer = NullRenderer(width=12, height=8)
        small = RenderedFrame(
            data=np.full((4, 6, 3), 200, dtype=np.uint8),
            pixel_format=PixelFormat.BGR24,
            metadata=FrameMetadata(frame_id=1, capture_timestamp=1.0),
        )

        out = renderer.render(small).data

        assert out.shape == (8, 12, 3)
        assert int(out[2:6, 3:9].min()) == 200
        border = np.ones((8, 12), dtype=bool)
        border[2:6, 3:9] = False
        assert int(out[border].min()) == 16 and int(out[border].max()) == 16
