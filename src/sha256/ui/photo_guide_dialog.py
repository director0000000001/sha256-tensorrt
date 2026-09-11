"""The window that opens before "Add avatar" chooses a file: what the
photograph must be, and the prompts that make one -- selectable, so a
prompt can be copied into an image generator (day 16, his ask)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from sha256.avatar.photo_guide import PARTED_LIPS_PROMPT, PORTRAIT_PROMPT, photo_guide_text

__all__ = ["PhotoGuideDialog"]


class PhotoGuideDialog(QDialog):
    """Read, copy a prompt, then choose the photograph -- or cancel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add avatar -- the photograph")
        self.resize(760, 640)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Read what the photograph must be. The two prompts below can be copied "
            "into an image generator as they are."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setPlainText(photo_guide_text())
        self._text.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        layout.addWidget(self._text, 1)

        copies = QHBoxLayout()
        self._copy_portrait = QPushButton("Copy prompt 1 (the portrait)")
        self._copy_portrait.clicked.connect(lambda: self._copy(PORTRAIT_PROMPT))
        copies.addWidget(self._copy_portrait)
        self._copy_parted = QPushButton("Copy prompt 2 (lips parted)")
        self._copy_parted.clicked.connect(lambda: self._copy(PARTED_LIPS_PROMPT))
        copies.addWidget(self._copy_parted)
        copies.addStretch(1)
        self._copied = QLabel("")
        copies.addWidget(self._copied)
        layout.addLayout(copies)

        buttons = QDialogButtonBox()
        choose = buttons.addButton(
            "Choose the photograph...", QDialogButtonBox.ButtonRole.AcceptRole
        )
        choose.setDefault(True)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _copy(self, text: str) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
            self._copied.setText("Copied.")

    @property
    def text(self) -> str:
        return self._text.toPlainText()
