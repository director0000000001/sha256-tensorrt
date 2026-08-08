"""Qt application setup.

Isolated from :mod:`sha256.__main__` so that everything except the UI can be
imported and tested in a headless environment.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QApplication

from sha256.utils.logging import get_logger

if TYPE_CHECKING:
    from sha256.app.bootstrap import ApplicationContext

__all__ = ["run_ui"]

_LOG = get_logger(__name__)


def run_ui(context: ApplicationContext) -> int:
    """Create the Qt application, show the main window, and run the event loop."""
    from sha256.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("sha256")
    app.setOrganizationName("sha256")
    app.setApplicationVersion(context.settings.environment)

    window = MainWindow(context)
    window.show()

    _LOG.info("UI started")
    return app.exec()
