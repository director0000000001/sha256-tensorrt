"""Tests for the window background.

The point of these is that decoration can never take the application down. A
background image is the least important thing in the program and sits on the
path to everything important, so every way it can be wrong is tested.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.gui

pytest.importorskip("PySide6.QtWidgets", reason="PySide6 is required for UI tests")


@pytest.fixture
def panel_factory(qtbot):
    from sha256.ui.background import BackgroundPanel

    def build(**kwargs):
        panel = BackgroundPanel(**kwargs)
        qtbot.addWidget(panel)
        panel.resize(320, 180)
        return panel

    return build


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    from PySide6.QtGui import QColor, QPixmap

    pixmap = QPixmap(64, 48)
    pixmap.fill(QColor(200, 60, 40))
    path = tmp_path / "background.png"
    assert pixmap.save(str(path))
    return path


class TestWithoutAnImage:
    def test_no_image_configured_is_normal(self, panel_factory):
        panel = panel_factory()

        assert not panel.has_image

    def test_a_missing_file_falls_back_instead_of_failing(self, panel_factory, tmp_path):
        panel = panel_factory(image_path=tmp_path / "absent.jpg")

        assert not panel.has_image

    def test_a_file_that_is_not_an_image_falls_back(self, panel_factory, tmp_path):
        broken = tmp_path / "background.jpg"
        broken.write_text("this is not a JPEG", encoding="utf-8")

        panel = panel_factory(image_path=broken)

        assert not panel.has_image

    def test_it_still_paints_without_an_image(self, panel_factory):
        """A panel that throws in paintEvent takes the window with it."""
        from PySide6.QtGui import QPixmap

        panel = panel_factory()
        target = QPixmap(panel.size())
        panel.render(target)

        assert not target.isNull()


class TestWithAnImage:
    def test_a_valid_image_is_used(self, panel_factory, image_file):
        panel = panel_factory(image_path=image_file)

        assert panel.has_image

    def test_the_image_is_visible_when_undimmed(self, panel_factory, image_file):
        from PySide6.QtGui import QPixmap

        panel = panel_factory(image_path=image_file, dim=0.0)
        target = QPixmap(panel.size())
        panel.render(target)

        colour = target.toImage().pixelColor(160, 90)
        assert colour.red() > colour.blue(), "the image should dominate, not the fallback colour"

    def test_dimming_darkens_the_image(self, panel_factory, image_file):
        from PySide6.QtGui import QPixmap

        def brightness(dim: float) -> int:
            panel = panel_factory(image_path=image_file, dim=dim)
            target = QPixmap(panel.size())
            panel.render(target)
            return target.toImage().pixelColor(160, 90).red()

        assert brightness(0.8) < brightness(0.0)

    def test_the_image_can_be_replaced_at_runtime(self, panel_factory, image_file):
        panel = panel_factory()

        assert panel.set_image(image_file)
        assert panel.has_image

        assert not panel.set_image(None)
        assert not panel.has_image

    def test_a_smaller_image_is_scaled_to_cover_the_panel(self, panel_factory, image_file):
        """A 64x48 image behind a 320x180 window must not leave bars."""
        from PySide6.QtGui import QPixmap

        panel = panel_factory(image_path=image_file, dim=0.0)
        target = QPixmap(panel.size())
        panel.render(target)

        image = target.toImage()
        corners = [
            image.pixelColor(2, 2),
            image.pixelColor(image.width() - 3, 2),
            image.pixelColor(2, image.height() - 3),
            image.pixelColor(image.width() - 3, image.height() - 3),
        ]
        assert all(c.red() > c.blue() for c in corners)


class TestDimClamping:
    @pytest.mark.parametrize("dim", [-5.0, 0.0, 0.5, 1.0, 5.0])
    def test_any_value_is_survivable(self, panel_factory, image_file, dim):
        from PySide6.QtGui import QPixmap

        panel = panel_factory(image_path=image_file, dim=dim)
        target = QPixmap(panel.size())
        panel.render(target)

        assert not target.isNull()
