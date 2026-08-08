"""Tests for avatar discovery.

The central property: one broken avatar must never take down the library. Every
failure mode is checked for *isolation*, not just for being detected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sha256.avatar.repository import FilesystemAvatarRepository
from sha256.core.errors import AvatarNotFoundError
from tests.conftest import make_manifest_dict, write_avatar


class TestDiscovery:
    def test_finds_every_valid_avatar(self, avatar_library: Path):
        repository = FilesystemAvatarRepository(avatar_library)
        found = repository.discover()

        assert {d.id for d in found} == {"avatar_001", "avatar_002"}
        assert repository.errors() == ()

    def test_missing_library_is_not_fatal(self, tmp_path: Path):
        """A first run has no library yet; that is empty, not broken."""
        repository = FilesystemAvatarRepository(tmp_path / "does_not_exist")
        assert repository.discover() == ()

    def test_results_are_cached_until_forced(self, avatar_library: Path):
        repository = FilesystemAvatarRepository(avatar_library)
        assert len(repository.discover()) == 2

        write_avatar(avatar_library, "avatar_003")
        assert len(repository.discover()) == 2, "cached result should be reused"
        assert len(repository.discover(force=True)) == 3

    def test_ignores_hidden_and_private_directories(self, avatar_library: Path):
        write_avatar(avatar_library, "avatar_004")
        (avatar_library / "avatar_004").rename(avatar_library / "_scratch")
        (avatar_library / ".git").mkdir()

        repository = FilesystemAvatarRepository(avatar_library)
        assert {d.id for d in repository.discover()} == {"avatar_001", "avatar_002"}

    def test_ignores_loose_files(self, avatar_library: Path):
        (avatar_library / "README.md").write_text("notes", encoding="utf-8")
        repository = FilesystemAvatarRepository(avatar_library)
        assert len(repository.discover()) == 2


class TestFailureIsolation:
    def test_malformed_json_does_not_hide_valid_avatars(self, avatar_library: Path):
        broken = avatar_library / "avatar_broken"
        broken.mkdir()
        (broken / "metadata.json").write_text("{ not json", encoding="utf-8")

        repository = FilesystemAvatarRepository(avatar_library)
        found = repository.discover()

        assert {d.id for d in found} == {"avatar_001", "avatar_002"}
        assert len(repository.errors()) == 1
        assert "invalid JSON" in repository.errors()[0][1]

    def test_missing_manifest_is_reported(self, avatar_library: Path):
        (avatar_library / "avatar_empty").mkdir()

        repository = FilesystemAvatarRepository(avatar_library)
        repository.discover()

        assert any("missing metadata.json" in reason for _, reason in repository.errors())

    def test_id_must_match_directory_name(self, avatar_library: Path):
        """Two plausible identities for one avatar would corrupt caches and exports."""
        directory = avatar_library / "avatar_005"
        directory.mkdir()
        (directory / "metadata.json").write_text(
            json.dumps(make_manifest_dict("avatar_999")), encoding="utf-8"
        )

        repository = FilesystemAvatarRepository(avatar_library)
        repository.discover()

        assert not repository.exists("avatar_999")
        assert any("does not match directory name" in reason for _, reason in repository.errors())

    def test_schema_violation_is_reported_with_detail(self, avatar_library: Path):
        directory = avatar_library / "avatar_006"
        directory.mkdir()
        payload = make_manifest_dict("avatar_006")
        del payload["engine"]
        (directory / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")

        repository = FilesystemAvatarRepository(avatar_library)
        repository.discover()

        reasons = [reason for _, reason in repository.errors()]
        assert any("invalid manifest" in reason and "engine" in reason for reason in reasons)


class TestLookup:
    def test_get_returns_descriptor_with_absolute_root(self, avatar_library: Path):
        repository = FilesystemAvatarRepository(avatar_library)
        descriptor = repository.get("avatar_001")

        assert descriptor.name == "Test avatar_001"
        assert descriptor.root.is_absolute()

    def test_get_unknown_id_raises(self, avatar_library: Path):
        repository = FilesystemAvatarRepository(avatar_library)
        with pytest.raises(AvatarNotFoundError, match="avatar_nope"):
            repository.get("avatar_nope")

    def test_lookup_triggers_discovery_lazily(self, avatar_library: Path):
        repository = FilesystemAvatarRepository(avatar_library)
        assert repository.exists("avatar_001")
