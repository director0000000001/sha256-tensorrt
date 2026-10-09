"""The start-up window (2026-10-09): shown before the main window, told
what the start is doing."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gui


def test_the_splash_shows_what_it_is_told(qtbot):
    pytest.importorskip("PySide6.QtWidgets")
    from sha256.ui.splash import StartupSplash

    splash = StartupSplash.open()
    qtbot.addWidget(splash)
    try:
        assert splash.isVisible()
        assert splash.message == "Starting"
        splash.say("Preparing the fast engine for this computer. 12 s")
        assert splash.message == "Preparing the fast engine for this computer. 12 s"
    finally:
        splash.close()
    assert not splash.isVisible()
