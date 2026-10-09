"""The start-up window (2026-10-09): what the application is doing before
its main window exists.

The start takes about twenty seconds (the neural networks onto the graphics
card, the avatar) and, the first time on a machine, minutes: the fast
engine is built for that card once (see ``sha256.avatar.liveportrait_tensorrt``).
Without a window a person sees nothing for that time and double-clicks
again. This one is small, frameless, on top, and says in a sentence what is
happening; :meth:`StartupSplash.say` is the ``progress`` hook
``sha256.app.bootstrap.build_application`` takes, called on the main thread,
so it pumps the event loop itself -- there is no other loop running yet.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QLabel, QProgressBar, QVBoxLayout, QWidget

__all__ = ["StartupSplash"]

_STYLE = """
QWidget#StartupSplash { background: #101014; border: 1px solid #2a2a33; }
QLabel#SplashTitle { color: #e8e8ec; font-size: 22px; font-weight: 600; }
QLabel#SplashMessage { color: #b8b8c0; font-size: 14px; }
QProgressBar { background: #1a1a20; border: none; height: 6px; }
QProgressBar::chunk { background: #6ee7b7; }
"""


class StartupSplash(QWidget):
    """A sentence and an indeterminate bar, centred on the screen."""

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setObjectName("StartupSplash")
        self.setFixedWidth(540)
        self.setStyleSheet(_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(12)
        title = QLabel("sha256")
        title.setObjectName("SplashTitle")
        self._message = QLabel("Starting")
        self._message.setObjectName("SplashMessage")
        self._message.setWordWrap(True)
        bar = QProgressBar()
        bar.setRange(0, 0)
        bar.setTextVisible(False)
        layout.addWidget(title)
        layout.addWidget(self._message)
        layout.addWidget(bar)

    @classmethod
    def open(cls) -> StartupSplash:
        """The Qt application (created here when there is none yet -- the
        main window reuses it), the splash shown and painted once."""
        if QApplication.instance() is None:
            QApplication(sys.argv)
        splash = cls()
        splash.adjustSize()
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            splash.move(area.center() - splash.rect().center())
        splash.show()
        QApplication.processEvents()
        return splash

    @property
    def message(self) -> str:
        return self._message.text()

    def say(self, text: str) -> None:
        """Show ``text`` and let the window repaint: the caller is the
        start itself, on this thread, between two of its steps."""
        self._message.setText(text)
        QApplication.processEvents()
