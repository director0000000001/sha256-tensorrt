"""Shared test fixtures.

Every fixture builds an isolated application: temporary paths, no config files
read from disk, and no logging reconfiguration. Tests must not depend on the
developer's `.env`, `configs/local.toml`, or avatar library.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Qt must run headless: CI has no display, and a real window stealing focus
# during a local test run is its own kind of failure. Set before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from sha256.app.bootstrap import ApplicationContext, build_application
from sha256.config.paths import AppPaths
from sha256.config.schema import Settings
from sha256.core.avatar import AvatarDescriptor, AvatarManifest
from sha256.core.enums import PixelFormat
from sha256.core.types import FrameMetadata, VideoFrame


def make_manifest_dict(avatar_id: str = "avatar_test", **overrides: Any) -> dict[str, Any]:
    """A minimal valid manifest, as a dict, for building fixtures."""
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "id": avatar_id,
        "name": f"Test {avatar_id}",
        "version": "1.0.0",
        "engine": {"engine_id": "null", "estimated_vram_mb": 0},
    }
    manifest.update(overrides)
    return manifest


def write_avatar(library: Path, avatar_id: str, **overrides: Any) -> Path:
    """Create a valid avatar directory inside ``library``."""
    directory = library / avatar_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata.json").write_text(
        json.dumps(make_manifest_dict(avatar_id, **overrides)), encoding="utf-8"
    )
    return directory


@pytest.fixture
def avatar_library(tmp_path: Path) -> Path:
    """An avatar library containing two valid avatars."""
    library = tmp_path / "avatars"
    library.mkdir()
    write_avatar(library, "avatar_001")
    write_avatar(library, "avatar_002")
    return library


@pytest.fixture
def app_paths(tmp_path: Path) -> AppPaths:
    """Paths rooted in a temporary directory, isolated from the real install."""
    paths = AppPaths(
        root=tmp_path,
        avatars=tmp_path / "avatars",
        models=tmp_path / "models",
        configs=tmp_path / "configs",
        logs=tmp_path / "logs",
        cache=tmp_path / "cache",
        is_development=True,
    )
    paths.ensure()
    return paths


@pytest.fixture
def settings() -> Settings:
    """Default settings with no file or environment influence.

    ``_env_file=None`` matters: without it pydantic-settings would read the
    developer's `.env` and the suite would pass or fail depending on their
    machine.
    """
    return Settings(_env_file=None)


@pytest.fixture
def app_context(app_paths: AppPaths, settings: Settings) -> Iterator[ApplicationContext]:
    """A fully wired application using placeholder backends."""
    write_avatar(app_paths.avatars, "avatar_001")
    write_avatar(app_paths.avatars, "avatar_002")

    context = build_application(settings=settings, paths=app_paths, setup_logging=False)
    try:
        yield context
    finally:
        context.shutdown()


@pytest.fixture
def sample_descriptor(tmp_path: Path) -> AvatarDescriptor:
    """A descriptor whose directory exists but declares no assets."""
    root = tmp_path / "avatar_sample"
    root.mkdir()
    manifest = AvatarManifest.model_validate(make_manifest_dict("avatar_sample"))
    return AvatarDescriptor(manifest=manifest, root=root)


@pytest.fixture
def sample_frame() -> VideoFrame:
    """A small RGB frame with deterministic content."""
    data = np.zeros((16, 24, 3), dtype=np.uint8)
    data[:, :, 0] = 128
    return VideoFrame(
        data=data,
        pixel_format=PixelFormat.RGB24,
        metadata=FrameMetadata(frame_id=1, capture_timestamp=100.0, source_id="test"),
    )
