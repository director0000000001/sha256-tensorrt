"""Avatar data contracts.

An avatar is described by a **manifest** (``metadata.json`` on disk) that the
application validates before touching a single byte of model weight. The
manifest is the only thing the core application knows about an avatar; engines
receive it and decide what to do with the asset references inside.

This is what satisfies the "adding an avatar must not require modifying core
application code" requirement: a new avatar is a new directory with a valid
manifest, discovered at runtime.

These are Pydantic models rather than plain dataclasses because they cross an
I/O boundary — they are parsed from untrusted-ish JSON on disk and must fail
loudly and legibly. Per-frame types stay in :mod:`sha256.core.types`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "AVATAR_ID_PATTERN",
    "SUPPORTED_MANIFEST_VERSIONS",
    "AssetRef",
    "AvatarDescriptor",
    "AvatarManifest",
    "EngineRequirement",
    "PathSafetyProbe",
    "RenderingParameters",
]

AVATAR_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{2,63}$"
"""Avatar ids are lowercase slugs.

Constrained because the id is used as a directory name, a metric label, and a
config key. Rejecting anything else at parse time removes an entire class of
path-traversal and quoting bugs.
"""

SUPPORTED_MANIFEST_VERSIONS: frozenset[int] = frozenset({1})
"""Manifest schema versions this build can read.

An avatar declaring a newer version is reported as unsupported rather than
being parsed on a best-effort basis — silently ignoring fields it depends on
would render it incorrectly. See ``docs/architecture/AVATAR_SYSTEM.md``.
"""

AvatarIdStr = Annotated[str, Field(pattern=AVATAR_ID_PATTERN)]


class PathSafetyProbe:
    """Minimal path-safety probe that does not depend on the host OS rules.

    ``pathlib`` resolves differently on Windows and POSIX, and an avatar pack
    authored on one must be judged the same way on the other. This inspects the
    raw string instead of trusting platform semantics.
    """

    __slots__ = ("escapes", "has_drive", "is_absolute")

    def __init__(self, raw: str) -> None:
        normalised = raw.replace("\\", "/")
        parts = [p for p in normalised.split("/") if p not in ("", ".")]
        self.is_absolute = normalised.startswith("/")
        self.has_drive = bool(re.match(r"^[a-zA-Z]:", normalised))
        self.escapes = ".." in parts

    @property
    def unsafe(self) -> bool:
        return self.is_absolute or self.has_drive or self.escapes


class AssetRef(BaseModel):
    """Reference to a file belonging to an avatar.

    Paths are always **relative to the avatar directory**. Absolute paths and
    parent traversal are rejected so that an avatar pack — which may eventually
    be downloadable — cannot reference arbitrary locations on disk.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(description="Path relative to the avatar directory.")
    role: str = Field(
        description="What the asset is for, e.g. 'source_portrait', 'weights', 'lut'."
    )
    sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
        description="Optional lowercase hex digest for integrity verification.",
    )
    size_bytes: int | None = Field(default=None, ge=0)
    optional: bool = Field(
        default=False,
        description="When true, a missing asset degrades the avatar instead of failing it.",
    )
    yaw: float | None = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description=(
            "Head yaw in degrees this asset was measured at, if it is one of a "
            "series covering an angle. Assets that carry it are chosen by "
            "nearest angle rather than by role, which is what lets a library "
            "hold many turns instead of two."
        ),
    )
    smile: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description=(
            "How far into a smile this asset was measured: 0 at a resting mouth, "
            "about 0.2 at a closed smile, 1 as the teeth appear and 2 at a laugh. "
            "The same idea as "
            "`yaw` on another axis: a series carrying it is chosen by nearest "
            "amount, so a smile grows through photographs of itself instead of "
            "one photograph fading in over a resting mouth."
        ),
    )

    eyes: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description=(
            "Where this asset's eyelids sit against a resting eye: 0 at rest, "
            "positive as they close and 1 fully shut, negative when they are "
            "open wider than usual. The third axis after `yaw` and `smile`, and "
            "the one the owner cares about most. Signed because both directions "
            "had the same defect — one photograph of fully shut eyes, and one of "
            "fully wide ones, each faded in by however far the lids had moved, "
            "so a resting face wore a sixth of a stare."
        ),
    )

    brows: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description=(
            "Where this asset's brows sit against a resting face: 0 at rest, "
            "positive raised, negative drawn down. Signed like `eyes`, and for "
            "the same reason — a raised-brow photograph faded in over a resting "
            "one puts two brows on screen, the second slightly above the first."
        ),
    )

    @field_validator("path")
    @classmethod
    def _reject_unsafe_paths(cls, value: str) -> str:
        if PathSafetyProbe(value).unsafe:
            msg = f"asset path must stay inside the avatar directory: {value!r}"
            raise ValueError(msg)
        return value


class EngineRequirement(BaseModel):
    """Which avatar engine an avatar needs, and on what terms.

    The engine is named by a **registry id**, not an import path, so the core
    never imports an engine module to read an avatar manifest.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    engine_id: str = Field(description="Registry id of the required engine backend.")
    min_engine_version: str | None = Field(
        default=None, description="Lowest compatible engine version (semver-ish)."
    )
    estimated_vram_mb: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Author's estimate of VRAM this avatar needs once resident. "
            "Used as a planning hint by the residency cache; the manager "
            "replaces it with measured values once loads have been observed."
        ),
    )
    device_preference: Literal["auto", "cuda", "cpu"] = "auto"


class RenderingParameters(BaseModel):
    """Per-avatar rendering knobs.

    Intentionally generic. Engine-specific tuning lives in
    :attr:`AvatarManifest.engine_options`, which is passed through opaquely so
    a new engine can accept new options without a schema change here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_width: int = Field(default=512, gt=0, le=4096)
    output_height: int = Field(default=512, gt=0, le=4096)
    background: Literal["transparent", "solid", "image"] = "solid"
    background_color: str = Field(default="#101014", pattern=r"^#[0-9a-fA-F]{6}$")
    mirror: bool = Field(default=False, description="Horizontally flip the output.")
    expression_scale: float = Field(
        default=1.0, gt=0.0, le=3.0, description="Multiplier applied to expression intensity."
    )
    head_motion_scale: float = Field(default=1.0, gt=0.0, le=3.0)


class AvatarManifest(BaseModel):
    """Contents of an avatar's ``metadata.json``.

    ``extra="forbid"`` is deliberate: a typo in a hand-written manifest should
    be an error at load time, not a silently ignored field that leaves the
    avatar subtly misconfigured.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: int = Field(description="Schema version of this manifest file.")
    id: AvatarIdStr = Field(description="Stable unique id; must match the directory name.")
    name: str = Field(min_length=1, max_length=128, description="Human-readable display name.")
    version: str = Field(
        default="1.0.0",
        pattern=r"^\d+\.\d+\.\d+$",
        description="Version of this avatar's content, independent of the schema version.",
    )

    description: str = Field(default="", max_length=1024)
    tags: tuple[str, ...] = Field(default=())

    preview: str | None = Field(
        default=None, description="Relative path to a preview image for the gallery."
    )
    assets: tuple[AssetRef, ...] = Field(default=())
    engine: EngineRequirement
    rendering: RenderingParameters = Field(default_factory=RenderingParameters)

    engine_options: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Opaque engine-specific settings. The core never interprets these; "
            "they are handed to the engine named in `engine.engine_id`."
        ),
    )

    fictional: bool = Field(
        default=True,
        description=(
            "Asserts the avatar is a fictional, AI-generated identity and not a "
            "likeness of a real person. See docs/PRIVACY.md."
        ),
    )

    @field_validator("manifest_version")
    @classmethod
    def _known_version(cls, value: int) -> int:
        if value not in SUPPORTED_MANIFEST_VERSIONS:
            supported = ", ".join(str(v) for v in sorted(SUPPORTED_MANIFEST_VERSIONS))
            msg = f"unsupported manifest_version {value}; this build supports {supported}"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _preview_is_safe(self) -> AvatarManifest:
        if self.preview is not None and PathSafetyProbe(self.preview).unsafe:
            msg = f"preview path must stay inside the avatar directory: {self.preview!r}"
            raise ValueError(msg)
        return self

    def asset_by_role(self, role: str) -> AssetRef | None:
        """First asset declared with ``role``, or ``None``."""
        return next((a for a in self.assets if a.role == role), None)


class AvatarDescriptor(BaseModel):
    """A discovered avatar: its manifest plus where it lives.

    This is what the UI gallery and the :class:`AvatarManager` pass around. It
    carries no model weights and is cheap to build for every avatar in the
    library, which is what keeps discovery fast with dozens of avatars.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    manifest: AvatarManifest
    root: Path = Field(description="Absolute path to the avatar's directory.")

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def preview_path(self) -> Path | None:
        """Absolute path to the preview image, if the manifest declares one."""
        if self.manifest.preview is None:
            return None
        return self.root / self.manifest.preview

    def asset_path(self, ref: AssetRef) -> Path:
        """Absolute path for an asset reference."""
        return self.root / ref.path
