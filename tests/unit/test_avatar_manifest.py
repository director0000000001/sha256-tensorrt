"""Tests for avatar manifest parsing and its safety rules.

The manifest is a trust boundary: avatar packs may eventually be downloadable,
so path-escape and version rules are tested as security properties, not
conveniences.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sha256.core.avatar import AssetRef, AvatarManifest, PathSafetyProbe
from tests.conftest import make_manifest_dict


class TestManifestVersioning:
    def test_accepts_supported_version(self):
        manifest = AvatarManifest.model_validate(make_manifest_dict())
        assert manifest.manifest_version == 1

    def test_rejects_future_version(self):
        """A newer manifest must be refused, not parsed best-effort."""
        with pytest.raises(ValidationError, match="unsupported manifest_version"):
            AvatarManifest.model_validate(make_manifest_dict(manifest_version=99))


class TestManifestIdRules:
    @pytest.mark.parametrize("avatar_id", ["avatar_001", "abc", "a-b_c-123"])
    def test_accepts_valid_slugs(self, avatar_id: str):
        assert AvatarManifest.model_validate(make_manifest_dict(avatar_id)).id == avatar_id

    @pytest.mark.parametrize(
        "avatar_id",
        [
            "Avatar_001",  # uppercase
            "ab",  # too short
            "../escape",  # path traversal
            "with space",
            "sym#bol",
            "",
        ],
    )
    def test_rejects_invalid_ids(self, avatar_id: str):
        with pytest.raises(ValidationError):
            AvatarManifest.model_validate(make_manifest_dict(avatar_id))


class TestPathSafety:
    @pytest.mark.parametrize(
        "path",
        [
            "../../../windows/system32/config/sam",
            "/etc/passwd",
            "C:/Windows/System32",
            "model\\..\\..\\secrets.txt",
        ],
    )
    def test_asset_paths_cannot_escape_the_avatar_directory(self, path: str):
        with pytest.raises(ValidationError, match="stay inside"):
            AssetRef(path=path, role="weights")

    @pytest.mark.parametrize("path", ["model/weights.onnx", "./config/tuning.json", "preview.png"])
    def test_accepts_contained_paths(self, path: str):
        assert AssetRef(path=path, role="weights").path == path

    def test_preview_path_is_also_checked(self):
        with pytest.raises(ValidationError, match="stay inside"):
            AvatarManifest.model_validate(make_manifest_dict(preview="../../evil.png"))

    def test_probe_judges_windows_and_posix_alike(self):
        """Separator handling must not depend on the host OS."""
        assert PathSafetyProbe("model\\..\\x").escapes
        assert PathSafetyProbe("model/../x").escapes
        assert not PathSafetyProbe("model/sub/x").escapes


class TestManifestStrictness:
    def test_unknown_fields_are_rejected(self):
        """A typo should fail loudly rather than be silently ignored."""
        with pytest.raises(ValidationError):
            AvatarManifest.model_validate(make_manifest_dict(renderring={}))

    def test_engine_requirement_is_mandatory(self):
        payload = make_manifest_dict()
        del payload["engine"]
        with pytest.raises(ValidationError):
            AvatarManifest.model_validate(payload)


class TestManifestHelpers:
    def test_asset_lookup_by_role(self):
        manifest = AvatarManifest.model_validate(
            make_manifest_dict(
                assets=[
                    {"path": "model/w.onnx", "role": "weights"},
                    {"path": "src.png", "role": "source_portrait"},
                ]
            )
        )
        found = manifest.asset_by_role("source_portrait")
        assert found is not None
        assert found.path == "src.png"

    def test_asset_lookup_returns_none_when_absent(self):
        manifest = AvatarManifest.model_validate(make_manifest_dict())
        assert manifest.asset_by_role("weights") is None

    def test_checksum_must_be_hex_digest(self):
        with pytest.raises(ValidationError):
            AssetRef(path="model/w.onnx", role="weights", sha256="not-a-digest")

    def test_defaults_assert_fictional_identity(self):
        assert AvatarManifest.model_validate(make_manifest_dict()).fictional is True
