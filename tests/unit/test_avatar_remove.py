"""The owner's delete (day 17, his ask): ``remove_avatar`` takes an avatar's
directory out of the library, and nothing else."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sha256.avatar.importer import AvatarImportError, remove_avatar
from tests.conftest import write_avatar


class TestRemove:
    def test_removes_the_directory_whole(self, avatar_library: Path):
        target = avatar_library / "avatar_001"
        assert target.is_dir()
        removed = remove_avatar(avatar_library, "avatar_001")
        assert removed == target.resolve()
        assert not target.exists()
        assert (avatar_library / "avatar_002").is_dir()

    def test_refuses_an_unknown_id(self, avatar_library: Path):
        with pytest.raises(AvatarImportError, match="no avatar"):
            remove_avatar(avatar_library, "nobody")

    def test_refuses_a_path_instead_of_an_id(self, avatar_library: Path, tmp_path: Path):
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        (outside / "metadata.json").write_text("{}", encoding="utf-8")
        for bad in ("../elsewhere", "..", ".", "avatar_001/..", "avatar_001\\x"):
            with pytest.raises(AvatarImportError):
                remove_avatar(avatar_library, bad)
        assert outside.is_dir() and (avatar_library / "avatar_001").is_dir()

    def test_refuses_a_directory_without_a_manifest_naming_it(self, avatar_library: Path):
        loose = avatar_library / "loose"
        loose.mkdir()
        with pytest.raises(AvatarImportError, match=r"metadata\.json"):
            remove_avatar(avatar_library, "loose")
        assert loose.is_dir()
        wrong = avatar_library / "wrong"
        write_avatar(avatar_library, "wrong")
        manifest = wrong / "metadata.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["id"] = "other"
        manifest.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(AvatarImportError, match="names"):
            remove_avatar(avatar_library, "wrong")
        assert wrong.is_dir()
