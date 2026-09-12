"""Small widgets the window is built from.

:class:`ElidedLabel` exists because of one measurement (day 16, 2026-09-12):
the bar along the bottom of the window set the window's minimum width, and a
plain ``QLabel`` on it added the whole width of whatever it said. With the
black screen on, the status line alone widened the minimum by 500 px, and a
long error at Start by more -- the window then opened wider than the screen
and could not be dragged smaller. A label that asks for no width of its own
and cuts what does not fit ends that.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

__all__ = ["ElidedLabel"]


class ElidedLabel(QLabel):
    """A one-line label that never widens its window.

    It asks the layout for the room its text would take (``sizeHint``) but
    insists on none (``minimumSizeHint`` is zero wide); what does not fit is
    cut with an ellipsis, and the full text goes to the tooltip so a cut
    message can still be read. ``text()`` returns the full text, whatever
    is painted.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def text(self) -> str:
        return self._full

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        self._full = text
        self._fit()

    def _fit(self) -> None:
        width = self.contentsRect().width()
        shown = self._full
        if width > 0 and self._full:
            shown = self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, width)
        QLabel.setText(self, shown)
        self.setToolTip(self._full if shown != self._full else "")

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._fit()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        margins = self.contentsMargins()
        width = self.fontMetrics().horizontalAdvance(self._full) + margins.left() + margins.right()
        return QSize(width + 2, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return QSize(0, super().minimumSizeHint().height())
