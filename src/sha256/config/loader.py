"""Configuration loading with layered precedence.

Lowest to highest priority:

1.  built-in defaults declared on the models in :mod:`sha256.config.schema`
2.  ``configs/default.toml``    — committed, the project's baseline
3.  ``configs/local.toml``      — gitignored, per-machine overrides
4.  environment variables / ``.env`` — handled by pydantic-settings
5.  explicit overrides passed in code (CLI flags, tests)

TOML is used rather than YAML because ``tomllib`` is in the standard library —
one less dependency for a file format that is easier to get right by hand.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sha256.config.paths import AppPaths
from sha256.config.schema import Settings
from sha256.core.errors import ConfigurationError

__all__ = ["load_settings", "merge_deep"]

DEFAULT_CONFIG_NAME = "default.toml"
LOCAL_CONFIG_NAME = "local.toml"


def merge_deep(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base``, returning a new dict.

    Nested tables merge key-by-key rather than replacing wholesale, so a local
    override of one camera setting does not silently reset the others.
    """
    result = dict(base)
    for key, value in overlay.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = merge_deep(existing, value)
        else:
            result[key] = value
    return result


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        msg = f"{path} is not valid TOML: {exc}"
        raise ConfigurationError(msg) from exc
    except OSError as exc:
        msg = f"could not read {path}: {exc}"
        raise ConfigurationError(msg) from exc


def load_settings(
    paths: AppPaths | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
    use_files: bool = True,
) -> Settings:
    """Build the validated :class:`Settings` for this run.

    Args:
        paths: Resolved app paths. Resolved automatically when omitted.
        overrides: Highest-priority values, e.g. parsed CLI flags.
        use_files: Skip the TOML layers. Tests use this for isolation from
            whatever happens to be on the developer's disk.

    Raises:
        ConfigurationError: If a file is malformed or a value fails validation.
    """
    paths = paths or AppPaths.resolve()

    file_data: dict[str, Any] = {}
    if use_files:
        file_data = merge_deep(
            _read_toml(paths.configs / DEFAULT_CONFIG_NAME),
            _read_toml(paths.configs / LOCAL_CONFIG_NAME),
        )

    merged = merge_deep(file_data, overrides or {})

    try:
        # pydantic-settings applies env vars and .env on top of what is passed in.
        return Settings(**merged)
    except ValidationError as exc:
        msg = f"invalid configuration:\n{_format_validation_error(exc)}"
        raise ConfigurationError(msg) from exc


def _format_validation_error(exc: ValidationError) -> str:
    """Render a validation failure as actionable lines rather than a stack trace."""
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)
