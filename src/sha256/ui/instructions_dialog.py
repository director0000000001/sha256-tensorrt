"""Menu -> Instructions: the whole text of :mod:`sha256.app.instructions`
in a window a person can read, scroll, select and copy from (day 16
night, his ask)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextEdit, QVBoxLayout, QWidget

from sha256.app.instructions import instructions_text

__all__ = ["InstructionsDialog"]


class InstructionsDialog(QDialog):
    """Read, then close."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Instructions")
        self.resize(820, 700)

        layout = QVBoxLayout(self)
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        # The text is laid out in columns; a fixed-width face keeps them.
        fixed: QFont = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self._text.setFont(fixed)
        self._text.setPlainText(instructions_text())
        self._text.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        layout.addWidget(self._text, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    @property
    def text(self) -> str:
        return self._text.toPlainText()
