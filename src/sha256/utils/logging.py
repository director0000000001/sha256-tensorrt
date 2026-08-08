"""Logging setup.

Built on the standard library only — no extra dependency for something this
well served by ``logging``.

Two properties matter for sha256 specifically:

1.  **Nothing leaves the machine.** Handlers write to the console and to a
    rotating local file. There is no network handler, and adding one would be a
    privacy regression (see ``docs/PRIVACY.md``).
2.  **The real-time loop must not log per frame.** File I/O in the capture path
    causes frame drops. Per-frame diagnostics belong in the performance
    monitor, which aggregates; logging is for lifecycle events and errors.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

__all__ = ["configure_logging", "get_logger"]

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_STANDARD_RECORD_FIELDS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}

_configured = False


class JsonFormatter(logging.Formatter):
    """Render records as one JSON object per line.

    Useful when logs are being collected by tooling during benchmarking. Any
    extra keyword passed via ``extra={...}`` is merged into the object, which is
    how structured context reaches the output without a third-party library.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, _DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_FIELDS
        }
        if extras:
            payload["context"] = extras
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    *,
    level: str = "INFO",
    log_dir: Path | None = None,
    json_format: bool = False,
    console: bool = True,
) -> None:
    """Configure the root logger. Idempotent — repeated calls reconfigure cleanly.

    Args:
        level: Threshold name, e.g. ``"DEBUG"``.
        log_dir: Directory for the rotating log file. ``None`` disables file
            logging entirely, which is the right default for tests.
        json_format: Emit JSON lines instead of human-readable text.
        console: Also write to stderr.
    """
    global _configured

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(level.upper())
    formatter: logging.Formatter = (
        JsonFormatter() if json_format else logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)
    )

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        root.addHandler(stream)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Bounded on purpose: an unattended real-time app can log for days, and
        # an unbounded file is both a disk risk and a privacy one.
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "sha256.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Qt is chatty at DEBUG and drowns out application logs.
    logging.getLogger("PySide6").setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger.

    Prefer ``get_logger(__name__)`` so the module path shows up in output.
    """
    return logging.getLogger(name)


def is_configured() -> bool:
    """Whether :func:`configure_logging` has run in this process."""
    return _configured
