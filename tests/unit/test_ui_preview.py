"""Tests for the preview widget.

Two things are worth testing here and both are about pixels.

Colour handling: a preview that shows a frame with its channels swapped looks
*almost* right, which is exactly the kind of bug that survives a glance.

Transparency: the widget must paint the video and nothing else. It is sized to
expand, so any background of its own would cover the window's background image
— which is precisely the bug this design replaced. The tests render it over a
known colour and assert that colour still shows where the video does not.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sha256.core.enums import PixelFormat
from sha256.core.types import BlendshapeSet, FaceTrackingResult, FrameMetadata, LandmarkSet

pytestmark = pytest.mark.gui

pytest.importorskip("PySide6.QtWidgets", reason="PySide6 is required for UI tests")

BACKDROP = (255, 0, 255)  # magenta: never produced by the frames used here


@pytest.fixture
def preview(qtbot):
    from sha256.ui.preview import PreviewWidget

    widget = PreviewWidget()
    qtbot.addWidget(widget)
    widget.resize(320, 180)
    return widget


def render(widget: Any) -> np.ndarray:
    """Paint the widget over a magenta backdrop and return it as an array."""
    from PySide6.QtGui import QColor, QPixmap

    target = QPixmap(widget.size())
    target.fill(QColor(*BACKDROP))
    widget.render(target)

    image = target.toImage().convertToFormat(target.toImage().format())
    array = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine() // 4, 4
    )
    # Qt hands back BGRA here; flip to RGB so assertions read naturally.
    return array[:, : image.width(), :3][:, :, ::-1].copy()


def backdrop_fraction(pixels: np.ndarray) -> float:
    """Share of the widget still showing what was behind it."""
    matches = np.all(np.abs(pixels.astype(np.int16) - np.array(BACKDROP)) < 12, axis=2)
    return float(matches.mean())


def green_pixels(pixels: np.ndarray) -> int:
    red, green, blue = (pixels[:, :, i].astype(np.int16) for i in range(3))
    return int((((green - red) > 25) & ((green - blue) > 25)).sum())


def amber_pixels(pixels: np.ndarray) -> int:
    red, green, blue = (pixels[:, :, i].astype(np.int16) for i in range(3))
    return int((((red - blue) > 120) & ((green - blue) > 80)).sum())


def make_frame(colour: tuple[int, int, int], size: tuple[int, int] = (48, 64)) -> np.ndarray:
    data = np.zeros((*size, 3), dtype=np.uint8)
    data[:, :] = colour
    return data


def tracking_at(x: float = 0.5, count: int = 60) -> FaceTrackingResult:
    points = np.tile(np.array([x, 0.5, 0.0], dtype=np.float32), (count, 1))
    return FaceTrackingResult(
        metadata=FrameMetadata(frame_id=1, capture_timestamp=1.0),
        face_detected=True,
        landmarks=LandmarkSet(points=points, scheme="mediapipe_478"),
        blendshapes=BlendshapeSet.empty(),
        confidence=1.0,
    )


class TestTransparency:
    def test_nothing_is_painted_before_a_frame_arrives(self, preview):
        """An opaque idle state would hide the window background image."""
        assert not preview.has_frame
        assert backdrop_fraction(render(preview)) == pytest.approx(1.0, abs=0.01)

    def test_clearing_returns_to_painting_nothing(self, preview):
        preview.show_frame(make_frame((10, 20, 30)), PixelFormat.RGB24)
        preview.clear_frame()

        assert not preview.has_frame
        assert backdrop_fraction(render(preview)) == pytest.approx(1.0, abs=0.01)

    def test_the_area_around_the_video_stays_see_through(self, preview):
        """A 4:3 frame in a 16:9 widget must leave the sides transparent."""
        preview.set_show_landmarks(False)
        preview.show_frame(make_frame((10, 200, 30), size=(120, 160)), PixelFormat.RGB24)

        # 320x180 holding a 4:3 video leaves about a quarter of the width bare.
        assert backdrop_fraction(render(preview)) > 0.15


class TestFrameDisplay:
    def test_displays_a_frame(self, preview):
        preview.set_show_landmarks(False)
        preview.show_frame(make_frame((10, 20, 30)), PixelFormat.RGB24)

        assert preview.has_frame
        assert backdrop_fraction(render(preview)) < 0.9

    def test_rgb_and_bgr_of_the_same_colour_look_the_same(self, preview):
        """The format field must actually be honoured, not assumed."""
        preview.set_mirror(False)
        preview.set_show_landmarks(False)

        preview.show_frame(make_frame((200, 40, 10)), PixelFormat.RGB24)
        as_rgb = render(preview)[90, 160]

        preview.show_frame(make_frame((10, 40, 200)), PixelFormat.BGR24)
        as_bgr = render(preview)[90, 160]

        assert abs(int(as_rgb[0]) - 200) < 12 and abs(int(as_rgb[2]) - 10) < 12
        assert np.abs(as_rgb.astype(np.int16) - as_bgr.astype(np.int16)).max() < 12

    def test_aspect_ratio_is_preserved(self, preview):
        preview.show_frame(make_frame((0, 0, 0), size=(360, 640)), PixelFormat.RGB24)
        rect = preview.video_rect()

        assert rect.width() / rect.height() == pytest.approx(640 / 360, abs=0.02)
        assert rect.width() <= preview.width() and rect.height() <= preview.height()

    def test_the_video_is_centred(self, preview):
        preview.show_frame(make_frame((0, 0, 0), size=(120, 160)), PixelFormat.RGB24)
        rect = preview.video_rect()

        assert abs(rect.center().x() - preview.width() // 2) <= 1
        assert abs(rect.center().y() - preview.height() // 2) <= 1

    def test_an_unsupported_format_is_ignored_rather_than_crashing(self, preview):
        preview.show_frame(np.zeros((8, 8), dtype=np.uint8), PixelFormat.GRAY8)

        assert not preview.has_frame

    def test_the_frame_survives_the_source_buffer_being_overwritten(self, preview):
        """Capture backends reuse buffers; the preview must have copied."""
        preview.set_mirror(False)
        preview.set_show_landmarks(False)
        data = make_frame((200, 40, 10))

        preview.show_frame(data, PixelFormat.RGB24)
        data[:, :] = (0, 0, 0)

        assert abs(int(render(preview)[90, 160][0]) - 200) < 12


class TestOverlay:
    def test_landmarks_are_drawn_when_enabled(self, preview):
        preview.set_show_landmarks(True)
        preview.show_frame(make_frame((0, 0, 0)), PixelFormat.RGB24, tracking_at())

        assert green_pixels(render(preview)) > 0

    def test_landmarks_are_not_drawn_when_disabled(self, preview):
        preview.set_show_landmarks(False)
        preview.show_frame(make_frame((0, 0, 0)), PixelFormat.RGB24, tracking_at())

        assert green_pixels(render(preview)) == 0

    def test_nothing_is_drawn_when_no_face_is_detected(self, preview):
        preview.set_show_landmarks(True)
        result = FaceTrackingResult.no_face(FrameMetadata(frame_id=1, capture_timestamp=1.0))

        preview.show_frame(make_frame((0, 0, 0)), PixelFormat.RGB24, result)

        assert green_pixels(render(preview)) == 0

    def test_mirroring_moves_the_overlay_with_the_image(self, preview):
        """A mirrored image with an unmirrored overlay puts the mesh beside the face."""
        preview.set_show_landmarks(True)
        preview.set_mirror(True)
        preview.show_frame(
            make_frame((0, 0, 0), size=(180, 320)), PixelFormat.RGB24, tracking_at(x=0.2)
        )

        pixels = render(preview)
        red, green, blue = (pixels[:, :, i].astype(np.int16) for i in range(3))
        columns = np.where((((green - red) > 25) & ((green - blue) > 25)).any(axis=0))[0]

        assert columns.size, "the overlay was not drawn at all"
        # Landmarks at x=0.2 must appear at 0.8 across the video once mirrored.
        rect = preview.video_rect()
        assert columns.min() > rect.left() + rect.width() * 0.6


class TestMirrorOverride:
    """Day 15: the black screen carries words; mirrored words read
    backwards, so one frame can be shown unmirrored whatever the setting."""

    def _frame(self):
        data = np.zeros((8, 16, 3), dtype=np.uint8)
        data[:, :4] = 255  # bright on the LEFT of the source
        return data

    def test_the_setting_mirrors_by_default(self, preview):
        preview.set_mirror(True)
        preview.show_frame(self._frame(), PixelFormat.RGB24)
        image = preview._pixmap.toImage()
        assert image.pixelColor(15, 4).red() == 255  # moved to the right
        assert image.pixelColor(0, 4).red() == 0

    def test_a_frame_can_be_shown_unmirrored(self, preview):
        preview.set_mirror(True)
        preview.show_frame(self._frame(), PixelFormat.RGB24, mirror=False)
        image = preview._pixmap.toImage()
        assert image.pixelColor(0, 4).red() == 255  # stays on the left
        assert image.pixelColor(15, 4).red() == 0


class TestHeadGuide:
    """Day 21: the head guide's oval and line drawn over the frame, in the
    fit's colour -- green with the head placed, amber until."""

    def test_a_good_fit_draws_green_and_a_poor_one_amber(self, preview):
        from sha256.ui.head_guide import GuideFit, GuideState, HeadGuide

        preview.set_show_landmarks(False)
        frame = make_frame((40, 40, 40), size=(180, 320))
        guide = HeadGuide()

        preview.show_frame(frame, PixelFormat.RGB24)
        plain = render(preview)
        assert green_pixels(plain) == 0 and amber_pixels(plain) == 0

        preview.show_frame(
            frame, PixelFormat.RGB24, guide=(guide, GuideFit(GuideState.GOOD, "Good. Hold still."))
        )
        good = render(preview)
        assert green_pixels(good) > 200 and amber_pixels(good) == 0

        preview.show_frame(
            frame, PixelFormat.RGB24, guide=(guide, GuideFit(GuideState.CLOSER, "Come closer"))
        )
        waiting = render(preview)
        assert amber_pixels(waiting) > 200 and green_pixels(waiting) == 0

    def test_the_oval_sits_where_the_guide_says(self, preview):
        """The oval's centre and size are fractions of the frame's height:
        on a 320x180 frame, centred at (160, 83), 79 wide and 112 tall."""
        from sha256.ui.head_guide import GuideFit, GuideState, HeadGuide

        preview.set_show_landmarks(False)
        frame = make_frame((40, 40, 40), size=(180, 320))
        preview.show_frame(
            frame, PixelFormat.RGB24, guide=(HeadGuide(), GuideFit(GuideState.GOOD, "x"))
        )
        pixels = render(preview)
        red, green, blue = (pixels[:, :, i].astype(np.int16) for i in range(3))
        oval = ((green - red) > 25) & ((green - blue) > 25)
        rows = np.where(oval.any(axis=1))[0]
        cols = np.where(oval.any(axis=0))[0]
        assert abs(rows.min() - (83 - 56)) <= 4 and abs(cols.min() - (160 - 40)) <= 4
        assert abs(cols.max() - (160 + 40)) <= 4
