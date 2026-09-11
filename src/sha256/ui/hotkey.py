"""A hotkey that works while another application is in front.

During a call the meeting app has the keyboard, and the one moment the black
screen is needed is exactly when sha256 is behind it. A Qt shortcut stops at
the window's edge; this registers the key with Windows itself
(``RegisterHotKey``) and hears the ``WM_HOTKEY`` message through Qt's native
event filter, so it fires whatever is in front. Windows only: elsewhere
:meth:`GlobalHotkey.register` says so once and does nothing, and the button
in the window is the way.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QAbstractNativeEventFilter, QCoreApplication

from sha256.utils.logging import get_logger

_LOG = get_logger(__name__)

WM_HOTKEY = 0x0312
_MODIFIERS = {"ctrl": 0x0002, "control": 0x0002, "alt": 0x0001, "shift": 0x0004, "win": 0x0008}
_MOD_NOREPEAT = 0x4000
_VK_F1 = 0x70


def parse_hotkey(spec: str) -> tuple[int, int]:
    """``"ctrl+alt+b"`` -> ``(modifier flags, virtual-key code)`` as
    ``RegisterHotKey`` wants them. A letter, a digit or F1-F24, with at least
    one modifier -- a bare key would steal it from every application."""
    parts = [part.strip().lower() for part in spec.split("+") if part.strip()]
    if not parts:
        msg = f"empty hotkey {spec!r}"
        raise ValueError(msg)
    key = parts[-1]
    modifiers = 0
    for part in parts[:-1]:
        if part not in _MODIFIERS:
            msg = f"unknown modifier {part!r} in hotkey {spec!r}"
            raise ValueError(msg)
        modifiers |= _MODIFIERS[part]
    if modifiers == 0:
        msg = f"hotkey {spec!r} needs a modifier (ctrl, alt, shift or win)"
        raise ValueError(msg)
    if len(key) == 1 and key.isalnum():
        return modifiers, ord(key.upper())
    if key.startswith("f") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
        return modifiers, _VK_F1 + int(key[1:]) - 1
    msg = f"unsupported key {key!r} in hotkey {spec!r} (a letter, a digit or F1-F24)"
    raise ValueError(msg)


def _user32() -> Any:
    """``user32`` with the argument types declared: a window handle is a
    64-bit pointer, and ctypes passes a bare Python int as a 32-bit C int --
    which truncates the handle and gets "invalid window handle" (1400) back,
    as the first run of this module did."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
    user32.RegisterHotKey.restype = wintypes.BOOL
    user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.UnregisterHotKey.restype = wintypes.BOOL
    return user32


class GlobalHotkey(QAbstractNativeEventFilter):
    """``callback`` runs on the GUI thread whenever ``spec`` is pressed,
    anywhere on the desktop, between :meth:`register` and :meth:`unregister`.
    ``window`` supplies the native handle the key is bound to."""

    _next_id = 0xA256

    def __init__(self, window: Any, spec: str, callback: Callable[[], None]) -> None:
        super().__init__()
        self._window = window
        self._spec = spec
        self._callback = callback
        self._id = GlobalHotkey._next_id
        GlobalHotkey._next_id += 1
        self._registered = False
        self._error: str | None = None

    @property
    def spec(self) -> str:
        return self._spec

    @property
    def is_registered(self) -> bool:
        return self._registered

    @property
    def error(self) -> str | None:
        """Why the key is not bound, for a tooltip; ``None`` while it is."""
        return self._error

    def register(self) -> bool:
        if self._registered:
            return True
        if sys.platform != "win32":
            self._error = "global hotkeys are Windows-only; use the button"
            _LOG.info("Global hotkey %s not bound: %s", self._spec, self._error)
            return False
        try:
            modifiers, key = parse_hotkey(self._spec)
        except ValueError as exc:
            self._error = str(exc)
            _LOG.warning("Global hotkey not bound: %s", exc)
            return False

        import ctypes

        user32 = _user32()
        handle = int(self._window.winId())
        if not user32.RegisterHotKey(handle, self._id, modifiers | _MOD_NOREPEAT, key):
            code = ctypes.GetLastError()
            self._error = f"Windows refused {self._spec} (error {code}); another app may hold it"
            _LOG.warning("Global hotkey not bound: %s", self._error)
            return False
        app = QCoreApplication.instance()
        if app is not None:
            app.installNativeEventFilter(self)
        self._registered = True
        self._error = None
        _LOG.info("Global hotkey %s bound", self._spec)
        return True

    def unregister(self) -> None:
        if not self._registered:
            return
        try:
            _user32().UnregisterHotKey(int(self._window.winId()), self._id)
        except Exception:
            _LOG.exception("Global hotkey %s could not be unbound", self._spec)
        app = QCoreApplication.instance()
        if app is not None:
            app.removeNativeEventFilter(self)
        self._registered = False

    def nativeEventFilter(self, event_type: Any, message: Any) -> Any:  # noqa: N802 - Qt naming
        if self._registered and bytes(event_type) in (
            b"windows_generic_MSG",
            b"windows_dispatcher_MSG",
        ):
            import ctypes.wintypes

            msg = ctypes.wintypes.MSG.from_address(int(message))
            if msg.message == WM_HOTKEY and int(msg.wParam) == self._id:
                try:
                    self._callback()
                except Exception:
                    _LOG.exception("Global hotkey %s callback failed", self._spec)
                return True, 0
        return False, 0
