"""The black screen's hotkey (day 15): the spec parser, and -- on Windows,
with a window -- a posted WM_HOTKEY reaching the callback through Qt's native
event filter, which is the whole mechanism without a key being pressed."""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6.QtWidgets", reason="PySide6 is required for UI tests")

from sha256.ui.hotkey import WM_HOTKEY, GlobalHotkey, parse_hotkey


class TestParse:
    def test_letters_digits_and_function_keys_with_modifiers(self):
        assert parse_hotkey("ctrl+alt+b") == (0x0002 | 0x0001, ord("B"))
        assert parse_hotkey("Ctrl + Shift + 7") == (0x0002 | 0x0004, ord("7"))
        assert parse_hotkey("win+f12") == (0x0008, 0x70 + 11)
        assert parse_hotkey("control+alt+F1") == (0x0003, 0x70)

    @pytest.mark.parametrize("bad", ["", "b", "ctrl+", "ctrl+esc", "meta+b", "ctrl+f25", "+ +"])
    def test_the_rest_is_refused(self, bad):
        with pytest.raises(ValueError):
            parse_hotkey(bad)


@pytest.mark.gui
class TestRegistration:
    def test_a_bad_spec_does_not_bind_and_says_why(self, qtbot):
        from PySide6.QtWidgets import QWidget

        widget = QWidget()
        qtbot.addWidget(widget)
        hotkey = GlobalHotkey(widget, "ctrl+esc", lambda: None)
        assert hotkey.register() is False
        assert not hotkey.is_registered
        assert hotkey.error is not None
        hotkey.unregister()  # a no-op, not an error

    @pytest.mark.skipif(sys.platform != "win32", reason="RegisterHotKey is Windows")
    def test_a_posted_hotkey_message_reaches_the_callback(self, qtbot):
        import ctypes

        from PySide6.QtWidgets import QApplication, QWidget

        if QApplication.platformName() != "windows":
            # The suite runs on Qt's offscreen platform, whose windows have no
            # Win32 handle to bind a key to; the same check on a real window
            # is scripts-level (day 15: bound, fired, unbound).
            pytest.skip(f"needs the windows platform plugin, not {QApplication.platformName()!r}")
        widget = QWidget()
        qtbot.addWidget(widget)
        widget.show()
        fired: list[int] = []
        # A combination no application is likely to hold.
        hotkey = GlobalHotkey(widget, "ctrl+alt+shift+f11", lambda: fired.append(1))
        if not hotkey.register():
            pytest.skip(f"the key is not free on this machine: {hotkey.error}")
        try:
            from ctypes import wintypes

            post = ctypes.windll.user32.PostMessageW
            post.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            handle = int(widget.winId())
            post(handle, WM_HOTKEY, hotkey._id, 0)
            post(handle, WM_HOTKEY, hotkey._id + 1, 0)  # not ours
            qtbot.waitUntil(lambda: len(fired) >= 1, timeout=2000)
            QApplication.processEvents()
        finally:
            hotkey.unregister()
        assert fired == [1]
        assert not hotkey.is_registered
