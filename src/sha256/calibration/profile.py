"""The driver profile file: ``configs/local.driver.json``.

The profile in force on this machine -- gitignored with the rest of
``configs/local.*``. It carries the driver options "Calibrate me" measured
(see ``DRIVER_OPTION_KEYS`` in the engine for what a profile may set) and,
for the record and for the offline measurement, the rest values, the
jitter and the response fits they came from. The engine reads only
``engine_options``; everything else is for a person or a script reading
the file. Since day 16's evening a profile carries the person's ``name``
and lives, one file each, in ``configs/drivers/`` (see :mod:`.roster`);
``local.driver.json`` is a copy of the chosen person's.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sha256.avatar.engine_liveportrait import DRIVER_OPTION_KEYS
from sha256.calibration.measure import CalibrationResult
from sha256.utils.logging import get_logger

__all__ = ["PROFILE_FILENAME", "PROFILE_VERSION", "DriverProfile", "driver_options_from"]

_LOG = get_logger(__name__)

PROFILE_FILENAME = "local.driver.json"
PROFILE_VERSION = 1


def _finite_only(values: dict[str, Any]) -> dict[str, Any]:
    """Drop NaN scalars: JSON has no NaN and a reader should not meet one."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, float) and not math.isfinite(value):
            continue
        out[key] = value
    return out


class DriverProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    profile_version: int = PROFILE_VERSION
    measured_at: str = ""
    # The person, as they typed it at the end of Calibrate me; "" for a
    # profile from before names existed (see roster.adopt_unnamed).
    name: str = ""
    avatar: str | None = None
    session: str | None = None
    engine_options: dict[str, Any] = Field(default_factory=dict)
    rest: dict[str, Any] = Field(default_factory=dict)
    jitter: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, dict[str, float]] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @classmethod
    def from_result(
        cls,
        result: CalibrationResult,
        *,
        avatar: str | None,
        session: str | None,
        measured_at: datetime | None = None,
        name: str = "",
    ) -> DriverProfile:
        stamp = (measured_at or datetime.now().astimezone()).isoformat(timespec="seconds")
        return cls(
            measured_at=stamp,
            name=name,
            avatar=avatar,
            session=session,
            engine_options={k: v for k, v in result.options.items() if k in DRIVER_OPTION_KEYS},
            rest=_finite_only(result.rest),
            jitter=_finite_only(result.jitter),
            response={name: _finite_only(fit) for name, fit in result.response.items()},
            notes=list(result.notes),
        )

    def overlay(self) -> dict[str, Any]:
        """The options the engine may take from this profile -- the driver's
        keys only; anything else in the file is said once and left."""
        unknown = sorted(set(self.engine_options) - DRIVER_OPTION_KEYS)
        if unknown:
            _LOG.warning("Driver profile: ignoring options that are not the driver's: %s", unknown)
        return {k: v for k, v in self.engine_options.items() if k in DRIVER_OPTION_KEYS}

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        _LOG.info("Driver profile written to %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> DriverProfile | None:
        """The profile at ``path``, or ``None`` when there is none -- and
        ``None``, with a warning, for a file that cannot be read: a damaged
        profile must not stop the application."""
        path = Path(path)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            profile = cls.model_validate(data)
        except (OSError, ValueError, ValidationError) as exc:
            _LOG.warning("Driver profile %s not read: %s", path, exc)
            return None
        return profile


def driver_options_from(path: Path) -> dict[str, Any]:
    """The overlay of the profile at ``path``; ``{}`` without one."""
    profile = DriverProfile.load(path)
    return profile.overlay() if profile is not None else {}
