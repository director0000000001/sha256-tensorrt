"""The window that opens before "Add avatar" chooses a file: what the
photograph must be, and the prompts that make one -- selectable, so a
prompt can be copied into an image generator (day 16, his ask). Day 23,
his idea: a button reads the person's skin colour from the camera and
puts its words and code into every prompt in place of "[your skin color]".

The reading: while the pipeline runs, the window hands in the runner's
``latest`` and the head guide; on the press the dialog polls the newest
snapshot thirty times a second and reads the frames whose head sits in
the guide (close enough, centred, facing the camera -- the same gate as
Calibrate me's placing phase), until :data:`FRAMES_WANTED` of them gave
a reading or :data:`READ_TIMEOUT_SECONDS` passed. The person is told
what is happening on one line under the buttons.
"""

from __future__ import annotations

from collections.abc import Callable
from time import monotonic
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QTimer
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

from sha256.avatar.photo_guide import PROMPTS, photo_guide_text
from sha256.avatar.skin_tone import SkinReading, SkinSampler, fill_prompt
from sha256.pipeline.runner import _as_bgr
from sha256.ui.head_guide import HeadGuide, fit_head

__all__ = ["READ_TIMEOUT_SECONDS", "PhotoGuideDialog"]

READ_TIMEOUT_SECONDS = 10.0
_TICK_MS = 33

_NO_CAMERA = "Start the camera first (Start), then press again."
_NOTHING_READ = (
    "No steady face seen -- sit in the oval, face the camera in good light, and press again."
)


class PhotoGuideDialog(QDialog):
    """Read, copy a prompt (filled with the skin reading when there is
    one), then choose the photograph -- or cancel."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        frames: Callable[[], Any | None] | None = None,
        guide: HeadGuide | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add avatar -- the photograph")
        self.resize(760, 680)
        self._frames = frames
        self._guide = guide
        self._sampler: SkinSampler | None = None
        self._reading: SkinReading | None = None
        self._deadline = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._tick)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Read what the photograph must be. The prompts below can be copied into "
            "Gemini or ChatGPT as they are; add the SHAVED photograph (prompt 2) as the avatar."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setPlainText(photo_guide_text())
        self._text.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        layout.addWidget(self._text, 1)

        skin = QHBoxLayout()
        self._read_button = QPushButton("Read my skin from the camera")
        self._read_button.setToolTip(
            "With the camera running: look at the camera in the light your calls will "
            "have and hold still for a second. The words and the colour code replace "
            "[your skin color] in every prompt."
        )
        self._read_button.clicked.connect(self._on_read)
        skin.addWidget(self._read_button)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        skin.addWidget(self._status, 1)
        layout.addLayout(skin)

        copies = QHBoxLayout()
        self._copy_buttons: list[QPushButton] = []
        for number, (_, prompt) in enumerate(PROMPTS, start=1):
            label = ("Copy prompt 1 (the face)", "Copy prompt 2 (shaved)", "Copy prompt 3 (lips)")[
                number - 1
            ]
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, text=prompt: self._copy(text))
            copies.addWidget(button)
            self._copy_buttons.append(button)
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

    # --- the prompts -------------------------------------------------------

    def _copy(self, text: str) -> None:
        """The prompt to the clipboard, filled with the reading when there
        is one (day 23)."""
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(fill_prompt(text, self._reading))
            self._copied.setText("Copied." if self._reading is None else "Copied, with your skin.")

    @property
    def text(self) -> str:
        return self._text.toPlainText()

    @property
    def reading(self) -> SkinReading | None:
        """The skin reading taken in this window, for the import's
        comparison; ``None`` when none was taken."""
        return self._reading

    # --- the skin reading --------------------------------------------------

    def _on_read(self) -> None:
        if self._frames is None:
            self._status.setText(_NO_CAMERA)
            return
        self._sampler = SkinSampler()
        self._deadline = monotonic() + READ_TIMEOUT_SECONDS
        self._status.setText("Reading... look at the camera and hold still.")
        self._read_button.setEnabled(False)
        self._timer.start()

    def _tick(self) -> None:
        sampler = self._sampler
        if sampler is None or self._frames is None:
            self._timer.stop()
            return
        snapshot = self._frames()
        if snapshot is not None and snapshot.source is not None:
            tracking = snapshot.tracking
            if fit_head(tracking, self._guide).good and tracking.landmarks is not None:
                source = snapshot.source
                frame = _as_bgr(source.data, source.pixel_format)
                if frame is not None:
                    points = np.asarray(tracking.landmarks.points, dtype=np.float32)[:, :2].copy()
                    points[:, 0] *= frame.shape[1]
                    points[:, 1] *= frame.shape[0]
                    if sampler.add(frame, points):
                        self._status.setText(
                            f"Reading... {sampler.count} of {sampler.wanted}, hold still."
                        )
        if sampler.done:
            self._finish(sampler.reading())
        elif monotonic() >= self._deadline:
            self._finish(sampler.reading() if sampler.count >= sampler.wanted // 2 else None)

    def _finish(self, reading: SkinReading | None) -> None:
        self._timer.stop()
        self._sampler = None
        self._read_button.setEnabled(True)
        if reading is None:
            self._status.setText(_NOTHING_READ)
            return
        self._reading = reading
        self._text.setPlainText(photo_guide_text(reading))
        self._status.setText(
            f"Your skin on the camera: {reading.describe()} ({reading.frames} frames). "
            "The prompts above now say it."
        )
