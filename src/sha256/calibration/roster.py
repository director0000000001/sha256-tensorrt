"""The people who sit at this camera: one driver profile each, by name.

Day 16 evening (2026-09-12), the owner's ask after the first other face
was calibrated: "after every calibration, a field where the person writes
their name; when they sit down to use the app they choose their name in
the menu, and the avatar is fitted to their face". Until then one file,
``configs/local.driver.json``, held whoever calibrated last, and the
friend's run silently replaced the owner's.

``configs/drivers/<name>.json`` holds each named profile.
``configs/local.driver.json`` stays what it was -- the profile in force,
the one the start lays over the engine (``apply_driver_profile``) -- and is
now a copy of the chosen person's file; choosing nobody removes it. A
``local.driver.json`` from before names existed is adopted into the roster
under a name of its own the first time the window looks, so nothing
measured is lost.
"""

from __future__ import annotations

import re
from pathlib import Path

from sha256.calibration.profile import PROFILE_FILENAME, DriverProfile
from sha256.utils.logging import get_logger

__all__ = [
    "DRIVERS_DIRNAME",
    "UNNAMED_PREFIX",
    "activate",
    "active_profile",
    "adopt_unnamed",
    "drivers_dir",
    "find_profile",
    "list_profiles",
    "named_path",
    "profile_slug",
    "save_named",
]

_LOG = get_logger(__name__)

DRIVERS_DIRNAME = "drivers"
UNNAMED_PREFIX = "Unnamed"
_SLUG_MAX = 80


def profile_slug(name: str) -> str:
    """A file name from a person's name: letters and digits of any script
    kept, every run of anything else one hyphen, lower-cased and bounded;
    ``"person"`` when nothing is left."""
    cleaned = re.sub(r"[^\w]+", "-", name.strip()).strip("-").lower()
    return cleaned[:_SLUG_MAX].strip("-") or "person"


def drivers_dir(configs: Path) -> Path:
    return Path(configs) / DRIVERS_DIRNAME


def named_path(configs: Path, name: str) -> Path:
    return drivers_dir(configs) / f"{profile_slug(name)}.json"


def list_profiles(configs: Path) -> list[DriverProfile]:
    """Every readable named profile in the roster, by name."""
    folder = drivers_dir(configs)
    if not folder.is_dir():
        return []
    found: list[DriverProfile] = []
    for path in sorted(folder.glob("*.json")):
        profile = DriverProfile.load(path)
        if profile is not None and profile.name.strip():
            found.append(profile)
    found.sort(key=lambda p: p.name.casefold())
    return found


def find_profile(configs: Path, name: str) -> DriverProfile | None:
    """The roster's profile under ``name``, or ``None``."""
    return DriverProfile.load(named_path(configs, name))


def save_named(profile: DriverProfile, configs: Path) -> Path:
    """Write ``profile`` into the roster under its own name."""
    if not profile.name.strip():
        msg = "a roster profile needs a name"
        raise ValueError(msg)
    return profile.save(named_path(configs, profile.name))


def activate(profile: DriverProfile | None, configs: Path) -> Path:
    """Make ``profile`` the one in force -- a copy at ``local.driver.json``,
    what the start reads -- or none: the file removed."""
    path = Path(configs) / PROFILE_FILENAME
    if profile is None:
        path.unlink(missing_ok=True)
        _LOG.info("No driver profile in force")
    else:
        profile.save(path)
        _LOG.info("Driver profile in force: %s", profile.name or "(unnamed)")
    return path


def active_profile(configs: Path) -> DriverProfile | None:
    """The profile in force on this machine, or ``None``."""
    return DriverProfile.load(Path(configs) / PROFILE_FILENAME)


def adopt_unnamed(configs: Path) -> DriverProfile | None:
    """A ``local.driver.json`` without a name (from before names existed, or
    written by hand) is given one -- "Unnamed (when it was measured)" --
    saved into the roster, and rewritten in place with the name, so the
    menu can show it and a new calibration does not lose it. ``None`` when
    there was nothing to adopt."""
    profile = active_profile(configs)
    if profile is None or profile.name.strip():
        return None
    when = profile.measured_at[:16].replace("T", " ").strip()
    profile.name = f"{UNNAMED_PREFIX} ({when})" if when else UNNAMED_PREFIX
    save_named(profile, configs)
    profile.save(Path(configs) / PROFILE_FILENAME)
    _LOG.info("Driver profile without a name adopted into the roster as %r", profile.name)
    return profile
