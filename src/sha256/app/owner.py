"""The owner's passphrase, and what it unlocks (day 15; removed 2026-10-09
on his ask; back on 2026-10-16 on his ask, "temporarily": a copy of the
application went missing on a friend's computer, and Add avatar / Delete
avatar must sit behind a Login again).

This is a gate in the window, not a vault: the passphrase is salted and
stretched (PBKDF2-HMAC-SHA256) and kept in ``configs/local.owner.json``
-- gitignored with the rest of ``configs/local.*``, so it never reaches
the repository; the portable build copies the file in, so the package
carries the gate. The window never SETS a passphrase (the first user of a
copy without the file would otherwise choose one): it is set on the
owner's machine with ``sha256 --set-owner``, and a copy without the file
has no Login at all. Anyone with the machine's files could still add a
directory to the library by hand; it keeps the button from his friends,
which is what was asked.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path

from sha256.utils.logging import get_logger

_LOG = get_logger(__name__)

OWNER_FILENAME = "local.owner.json"
_ITERATIONS = 200_000
MIN_PASSPHRASE_LENGTH = 4


class OwnerAccess:
    """One passphrase, one unlocked flag for the running session."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._unlocked = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def is_configured(self) -> bool:
        return self._read() is not None

    @property
    def is_unlocked(self) -> bool:
        return self._unlocked

    def set_passphrase(self, passphrase: str) -> None:
        """Store a new passphrase and unlock. Refuses a short one."""
        text = passphrase.strip()
        if len(text) < MIN_PASSPHRASE_LENGTH:
            msg = f"the passphrase needs at least {MIN_PASSPHRASE_LENGTH} characters"
            raise ValueError(msg)
        salt = os.urandom(16)
        key = _derive(text, salt)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"salt": salt.hex(), "key": key.hex(), "iterations": _ITERATIONS}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        self._unlocked = True
        _LOG.info("Owner passphrase set (%s)", self._path.name)

    def unlock(self, passphrase: str) -> bool:
        """``True`` and unlocked when ``passphrase`` is the stored one."""
        record = self._read()
        if record is None:
            return False
        try:
            salt = bytes.fromhex(str(record["salt"]))
            expected = bytes.fromhex(str(record["key"]))
            raw = record.get("iterations", _ITERATIONS)
            iterations = raw if isinstance(raw, int) else _ITERATIONS
        except (KeyError, ValueError, TypeError):
            return False
        key = _derive(passphrase.strip(), salt, iterations)
        if hmac.compare_digest(key, expected):
            self._unlocked = True
            _LOG.info("Owner unlocked")
            return True
        _LOG.warning("Owner passphrase refused")
        return False

    def lock(self) -> None:
        self._unlocked = False

    def _read(self) -> dict[str, object] | None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) and "salt" in data and "key" in data else None


def _derive(passphrase: str, salt: bytes, iterations: int = _ITERATIONS) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, iterations)
