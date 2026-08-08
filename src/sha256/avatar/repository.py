"""Filesystem-backed avatar library.

Layout — one directory per avatar, named by its id::

    avatars/
      avatar_001/
        metadata.json      <- the manifest; the only required file
        preview.png
        model/             <- weights, gitignored
        config/

Why directory-per-avatar rather than a central index file:

*   **Adding an avatar is a file copy.** No central registry to edit, no merge
    conflict when two avatars are added independently, which is what "adding an
    avatar must not require modifying core code" means in practice.
*   **A corrupt avatar is isolated.** One unparseable manifest cannot take the
    library down; it is reported and skipped.
*   **Deleting is a directory removal**, with no dangling index entry.

The cost is a directory scan at startup. For dozens of avatars that is a few
milliseconds, and discovery reads only manifests — never weights.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from sha256.core.avatar import AvatarDescriptor, AvatarManifest
from sha256.core.errors import AvatarManifestError, AvatarNotFoundError
from sha256.utils.logging import get_logger

__all__ = ["MANIFEST_FILENAME", "FilesystemAvatarRepository", "InMemoryAvatarRepository"]

_LOG = get_logger(__name__)

MANIFEST_FILENAME = "metadata.json"


class FilesystemAvatarRepository:
    """Discovers avatars by scanning a library directory."""

    def __init__(self, library_path: Path) -> None:
        self._library_path = library_path
        self._descriptors: dict[str, AvatarDescriptor] = {}
        self._errors: list[tuple[str, str]] = []
        self._scanned = False

    @property
    def library_path(self) -> Path:
        return self._library_path

    def discover(self, *, force: bool = False) -> Sequence[AvatarDescriptor]:
        """Scan the library. Results are cached until ``force=True``.

        Never raises for a bad avatar: problems are collected in :meth:`errors`
        so a single malformed manifest cannot hide a healthy library.
        """
        if self._scanned and not force:
            return tuple(self._descriptors.values())

        self._descriptors.clear()
        self._errors.clear()

        if not self._library_path.is_dir():
            _LOG.warning("Avatar library not found at %s", self._library_path)
            self._scanned = True
            return ()

        for entry in sorted(self._library_path.iterdir()):
            if not entry.is_dir() or entry.name.startswith((".", "_")):
                continue
            self._load_entry(entry)

        self._scanned = True
        _LOG.info(
            "Discovered %d avatar(s) in %s (%d error(s))",
            len(self._descriptors),
            self._library_path,
            len(self._errors),
        )
        return tuple(self._descriptors.values())

    def _load_entry(self, directory: Path) -> None:
        manifest_path = directory / MANIFEST_FILENAME
        if not manifest_path.is_file():
            self._errors.append((str(directory), f"missing {MANIFEST_FILENAME}"))
            return

        try:
            manifest = self._read_manifest(manifest_path)
        except AvatarManifestError as exc:
            self._errors.append((str(manifest_path), str(exc)))
            _LOG.warning("Skipping avatar at %s: %s", directory, exc)
            return

        # The id is a directory name elsewhere in the system (caches, exports),
        # so a mismatch would produce two plausible identities for one avatar.
        if manifest.id != directory.name:
            reason = f"manifest id {manifest.id!r} does not match directory name {directory.name!r}"
            self._errors.append((str(manifest_path), reason))
            _LOG.warning("Skipping avatar at %s: %s", directory, reason)
            return

        if manifest.id in self._descriptors:
            self._errors.append((str(manifest_path), f"duplicate avatar id {manifest.id!r}"))
            return

        self._descriptors[manifest.id] = AvatarDescriptor(
            manifest=manifest, root=directory.resolve()
        )

    @staticmethod
    def _read_manifest(path: Path) -> AvatarManifest:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            msg = f"invalid JSON: {exc}"
            raise AvatarManifestError(msg) from exc
        except OSError as exc:
            msg = f"unreadable: {exc}"
            raise AvatarManifestError(msg) from exc

        try:
            return AvatarManifest.model_validate(raw)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
            )
            msg = f"invalid manifest: {details}"
            raise AvatarManifestError(msg) from exc

    def get(self, avatar_id: str) -> AvatarDescriptor:
        if not self._scanned:
            self.discover()
        try:
            return self._descriptors[avatar_id]
        except KeyError:
            raise AvatarNotFoundError(avatar_id) from None

    def exists(self, avatar_id: str) -> bool:
        if not self._scanned:
            self.discover()
        return avatar_id in self._descriptors

    def errors(self) -> Sequence[tuple[str, str]]:
        """``(location, reason)`` for every entry that failed to load."""
        return tuple(self._errors)


class InMemoryAvatarRepository:
    """Repository backed by a fixed list. For tests and fixtures."""

    def __init__(self, descriptors: Sequence[AvatarDescriptor] = ()) -> None:
        self._descriptors = {d.id: d for d in descriptors}

    def discover(self, *, force: bool = False) -> Sequence[AvatarDescriptor]:
        del force  # nothing to re-scan; the contents are fixed at construction
        return tuple(self._descriptors.values())

    def get(self, avatar_id: str) -> AvatarDescriptor:
        try:
            return self._descriptors[avatar_id]
        except KeyError:
            raise AvatarNotFoundError(avatar_id) from None

    def exists(self, avatar_id: str) -> bool:
        return avatar_id in self._descriptors

    def errors(self) -> Sequence[tuple[str, str]]:
        return ()

    def add(self, descriptor: AvatarDescriptor) -> None:
        self._descriptors[descriptor.id] = descriptor
