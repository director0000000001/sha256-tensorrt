"""Filesystem layout resolution.

sha256 runs in two shapes and they need different roots:

*   **Development** — run from a git checkout. Everything (avatars, models,
    logs, configs) lives inside the repository so a developer can see and edit
    it, and so a clean checkout is a clean state.
*   **Installed** — run from a packaged build. Program files are read-only on
    Windows, so user data must live under ``%LOCALAPPDATA%``.

Resolving this in one place means no module ever hard-codes a path, and the
switch between the two is a single decision made at startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from platformdirs import PlatformDirs

__all__ = ["AppPaths", "find_repo_root"]

_APP_NAME = "sha256"
_APP_AUTHOR = "sha256"

# A checkout is identified by these markers rather than by ``.git``: a source
# tarball or an exported build tree is still "development", and CI checkouts
# occasionally lack a git directory.
_REPO_MARKERS = ("pyproject.toml", "src")


def find_repo_root(start: Path | None = None) -> Path | None:
    """Walk upwards looking for a repository checkout. ``None`` if not in one."""
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in _REPO_MARKERS):
            return candidate
    return None


@dataclass(frozen=True, slots=True)
class AppPaths:
    """Resolved locations for everything sha256 reads or writes."""

    root: Path
    """Base directory: the repo root in development, user data dir when installed."""

    avatars: Path
    """Avatar library root — one subdirectory per avatar."""

    models: Path
    """Shared model artifacts (downloads, converted engines). Never in Git."""

    configs: Path
    """Layered configuration files."""

    logs: Path
    cache: Path
    is_development: bool

    @property
    def captures(self) -> Path:
        """Where "that looked wrong" writes what it saved.

        Under ``root`` rather than the cache: a capture is something the user
        made deliberately and will want to find again, and a cache is a place
        things are allowed to vanish from.
        """
        return self.root / "captures"

    @classmethod
    def resolve(cls, *, force_development: bool | None = None) -> AppPaths:
        """Detect the run shape and build the matching layout.

        Args:
            force_development: Override detection. Useful in tests.
        """
        repo_root = find_repo_root()
        development = repo_root is not None if force_development is None else force_development

        if development and repo_root is not None:
            return cls(
                root=repo_root,
                avatars=repo_root / "avatars",
                models=repo_root / "models",
                configs=repo_root / "configs",
                logs=repo_root / "logs",
                cache=repo_root / ".cache",
                is_development=True,
            )

        dirs = PlatformDirs(appname=_APP_NAME, appauthor=_APP_AUTHOR, roaming=False)
        data_root = Path(dirs.user_data_dir)
        return cls(
            root=data_root,
            avatars=data_root / "avatars",
            models=data_root / "models",
            configs=Path(dirs.user_config_dir),
            logs=Path(dirs.user_log_dir),
            cache=Path(dirs.user_cache_dir),
            is_development=False,
        )

    def ensure(self) -> None:
        """Create every directory that is written to.

        ``configs`` is not created here — a missing config directory is a valid
        state that falls back to built-in defaults.
        """
        for directory in (self.avatars, self.models, self.logs, self.cache):
            directory.mkdir(parents=True, exist_ok=True)

    def resolve_relative(self, value: Path | str) -> Path:
        """Interpret a possibly-relative configured path against :attr:`root`."""
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (self.root / candidate)
