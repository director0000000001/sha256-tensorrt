"""Tests for the preview widget.

Pixel handling is what is worth testing here. A preview that shows a frame with
its colour channels swapped looks *almost* right, which is exactly the kind of
bug that survives a glance and ships.
"""

from __future__ import annotations

import numpy as np
import pytest

from sha256.core.enums import PixelFormat
from sha256.core.types import BlendshapeSet, FaceTrackingResult, FrameMetadata, LandmarkSet

pytestmark = pytest.mark.gui

pytest.importorskip("PySide6.QtWidgets", reason="PySide6 is required for UI tests")


@pytest.fixture
def preview(qtbot):
    from sha256.ui.preview import PreviewWidget

    widget = PreviewWidget()
    qtbot.addWidget(widget)
    widget.resize(320, 180)
    return widget


def make_frame(colour: tuple[int, int, int], size: tuple[int, int] = (48, 64)) -> np.ndarray:
    data = np.zeros((*size, 3), dtype=np.uint8)
    data[:, :] = colour
    return data


def tracking_with_landmarks(count: int = 60) -> FaceTrackingResult:
    points = np.tile(np.array([0.5, 0.5, 0.0], dtype=np.float32), (count, 1))
    return FaceTrackingResult(
        metadata=FrameMetadata(frame_id=1, capture_timestamp=1.0),
        face_detected=True,
        landmarks=LandmarkSet(points=points, scheme="mediapipe_478"),
        blendshapes=BlendshapeSet.empty(),
        confidence=1.0,
    )


class TestIdleState:
    def test_starts_with_a_prompt_and_no_image(self, preview):
        assert preview.pixmap().isNull()
        assert preview.text()

    def test_clearing_returns_to_the_prompt(self, preview):
        preview.show_frame(make_frame((10, 20, 30)), PixelFormat.RGB24)
        preview.clear_frame()

        assert preview.pixmap().isNull()


class TestFrameDisplay:
    def test_displays_an_rgb_frame(self, preview):
        preview.show_frame(make_frame((10, 20, 30)), PixelFormat.RGB24)

        assert not preview.pixmap().isNull()

    def test_rgb_and_bgr_of_the_same_colour_look_the_same(self, preview):
        """The format field must actually be honoured, not assumed."""
        from PySide6.QtGui import QColor

        preview.set_mirror(False)
        preview.set_show_landmarks(False)

        preview.show_frame(make_frame((200, 40, 10)), PixelFormat.RGB24)
        as_rgb = preview.pixmap().toImage().pixelColor(5, 5)

        preview.show_frame(make_frame((10, 40, 200)), PixelFormat.BGR24)
        as_bgr = preview.pixmap().toImage().pixelColor(5, 5)

        assert as_rgb == QColor(200, 40, 10)
        assert as_bgr == QColor(200, 40, 10)

    def test_aspect_ratio_is_preserved(self, preview):
        preview.show_frame(make_frame((0, 0, 0), size=(360, 640)), PixelFormat.RGB24)
        pixmap = preview.pixmap()

        source_ratio = 640 / 360
        assert pixmap.width() / pixmap.height() == pytest.approx(source_ratio, abs=0.02)

    def test_an_unsupported_format_is_ignored_rather_than_crashing(self, preview):
        preview.show_frame(np.zeros((8, 8), dtype=np.uint8), PixelFormat.GRAY8)

        assert preview.pixmap().isNull()

    def test_the_frame_survives_the_source_buffer_being_overwritten(self, preview):
        """Capture backends reuse buffers; the preview must have copied."""
        from PySide6.QtGui import QColor

        preview.set_mirror(False)
        preview.set_show_landmarks(False)
        data = make_frame((200, 40, 10))

        preview.show_frame(data, PixelFormat.RGB24)
        data[:, :] = (0, 0, 0)

        assert preview.pixmap().toImage().pixelColor(5, 5) == QColor(200, 40, 10)


class TestOverlay:
    def _green_pixels(self, preview) -> int:
        image = preview.pixmap().toImage()
        count = 0
        for y in range(image.height()):
            for x in range(image.width()):
                colour = image.pixelColor(x, y)
                if colour.green() > colour.red() + 25 and colour.green() > colour.blue() + 25:
                    count += 1
        return count

    def test_landmarks_are_drawn_when_enabled(self, preview):
        preview.set_show_landmarks(True)
        preview.show_frame(make_frame((0, 0, 0)), PixelFormat.RGB24, tracking_with_landmarks())

        assert self._green_pixels(preview) > 0

    def test_landmarks_are_not_drawn_when_disabled(self, preview):
        preview.set_show_landmarks(False)
        preview.show_frame(make_frame((0, 0, 0)), PixelFormat.RGB24, tracking_with_landmarks())

        assert self._green_pixels(preview) == 0

    def test_nothing_is_drawn_when_no_face_is_detected(self, preview):
        preview.set_show_landmarks(True)
        result = FaceTrackingResult.no_face(FrameMetadata(frame_id=1, capture_timestamp=1.0))

        preview.show_frame(make_frame((0, 0, 0)), PixelFormat.RGB24, result)

        assert self._green_pixels(preview) == 0

    def test_mirroring_moves_the_overlay_with_the_image(self, preview):
        """A mirrored image with an unmirrored overlay puts the mesh beside the face."""
        points = np.tile(np.array([0.2, 0.5, 0.0], dtype=np.float32), (40, 1))
        tracking = FaceTrackingResult(
            metadata=FrameMetadata(frame_id=1, capture_timestamp=1.0),
            face_detected=True,
            landmarks=LandmarkSet(points=points, scheme="mediapipe_478"),
            confidence=1.0,
        )

        preview.set_show_landmarks(True)
        preview.set_mirror(True)
        preview.show_frame(make_frame((0, 0, 0), size=(180, 320)), PixelFormat.RGB24, tracking)

        image = preview.pixmap().toImage()
        columns = [
            x
            for x in range(image.width())
            for y in range(image.height())
            if image.pixelColor(x, y).green() > image.pixelColor(x, y).red() + 25
        ]

        assert columns, "the overlay was not drawn at all"
        # Landmarks at x=0.2 must appear at 0.8 of the width once mirrored.
        assert min(columns) > image.width() * 0.6
