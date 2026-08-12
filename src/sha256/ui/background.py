"""Window background.

Holds the session controls and paints a photograph behind them.

Two rules shape this:

*   **Decoration must never break the application.** A missing, corrupt, or
    unreadable image falls back to the plain colour and logs it. Nobody should
    lose a video call because a background file was moved.
*   **Controls have to stay readable over an image nobody has seen.** The photo
    is darkened by a configurable amount before anything is drawn on top,
    because a light photo and light text is a real failure mode and "pick a
    darker picture" is not a fix that survives contact with a user.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPixmap
from PySide6.QtWidgets import QWidget

from sha256.utils.logging import get_logger

__all__ = ["BackgroundPanel"]

_LOG = get_logger(__name__)


class BackgroundPanel(QWidget):
    """A panel that paints a scaled, dimmed background image behind its children."""

    def __init__(
        self,
        *,
        image_path: Path | None = None,
        dim: float = 0.55,
        colour: str = "#101014",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._colour = QColor(colour)
        self._dim = min(max(dim, 0.0), 1.0)
        self._pixmap: QPixmap | None = None
        self._scaled: QPixmap | None = None

        if image_path is not None:
            self.set_image(image_path)

    # --- configuration -----------------------------------------------------

    @property
    def has_image(self) -> bool:
        return self._pixmap is not None

    def set_image(self, path: Path | None) -> bool:
        """Load a background image. Returns whether it is now in use."""
        self._scaled = None

        if path is None:
            self._pixmap = None
            self.update()
            return False

        if not path.is_file():
            _LOG.warning("Background image not found, using the plain background: %s", path)
            self._pixmap = None
            self.update()
            return False

        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            _LOG.warning("Background image could not be read as an image: %s", path)
            self._pixmap = None
            self.update()
            return False

        self._pixmap = pixmap
        _LOG.info("Background image loaded: %s (%dx%d)", path.name, pixmap.width(), pixmap.height())
        self.update()
        return True

    def set_dim(self, dim: float) -> None:
        self._dim = min(max(dim, 0.0), 1.0)
        self.update()

    # --- painting ----------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), self._colour)

            if self._pixmap is not None:
                painter.drawPixmap(0, 0, self._cover_pixmap())
                if self._dim > 0.0:
                    scrim = QColor(0, 0, 0)
                    scrim.setAlphaF(self._dim)
                    painter.fillRect(self.rect(), scrim)
        finally:
            painter.end()

        super().paintEvent(event)

    def _cover_pixmap(self) -> QPixmap:
        """The image scaled to cover the panel, cropped to centre.

        Cached: rescaling a 1920x1080 photo on every repaint would cost more
        than everything else the window does, and the window repaints whenever
        the preview updates.
        """
        assert self._pixmap is not None

        if self._scaled is None or self._scaled.size() != self.size():
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            # Crop the overhang so the visible part stays centred rather than
            # anchored to the top-left, which lops the head off a portrait.
            x = max((scaled.width() - self.width()) // 2, 0)
            y = max((scaled.height() - self.height()) // 2, 0)
            self._scaled = scaled.copy(x, y, self.width(), self.height())

        return self._scaled
