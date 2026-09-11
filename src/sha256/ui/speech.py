"""A voice for the calibration's instructions (day 16, his ask: "either voice
or more time" -- both).

Qt's own text-to-speech (``PySide6.QtTextToSpeech``, the SAPI / WinRT
engines on Windows) reads each instruction aloud as it appears, so the
person can look at the camera instead of the screen. Optional in every
way: a build without the module, a machine without a voice, or an engine
that refuses, all leave :attr:`Speaker.available` false and ``say`` a
no-op -- the read-aloud time on screen is what carries the instruction
then.
"""

from __future__ import annotations

import contextlib
from typing import Any

from sha256.utils.logging import get_logger

__all__ = ["Speaker"]

_LOG = get_logger(__name__)


class Speaker:
    """One text-to-speech engine, or nothing."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._tts: Any = None
        if not enabled:
            return
        try:
            from PySide6.QtTextToSpeech import QTextToSpeech

            tts = QTextToSpeech()
            # An engine that could not load reports an error state; treat
            # it as no voice rather than a voice that never speaks.
            if tts.state() == QTextToSpeech.State.Error:
                _LOG.info("Text-to-speech engine in error state; no voice")
                return
            self._tts = tts
        except Exception as exc:  # no module, no engine, no device
            _LOG.info("No text-to-speech available: %s", exc)

    @property
    def available(self) -> bool:
        return self._tts is not None

    def say(self, text: str) -> None:
        """Speak ``text``, cutting off whatever was still being said."""
        if self._tts is None or not text:
            return
        try:
            self._tts.stop()
            self._tts.say(text)
        except Exception as exc:
            _LOG.info("Text-to-speech failed: %s", exc)
            self._tts = None

    def stop(self) -> None:
        if self._tts is not None:
            with contextlib.suppress(Exception):
                self._tts.stop()
