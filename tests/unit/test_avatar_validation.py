"""Tests for avatar asset validation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from sha256.avatar.validation import compute_sha256, validate_avatar
from sha256.core.avatar import AvatarDescriptor, AvatarManifest
from tests.conftest import make_manifest_dict


def _descriptor(root: Path, **overrides: object) -> AvatarDescriptor:
    manifest = AvatarManifest.model_validate(make_manifest_dict("avatar_test", **overrides))
    return AvatarDescriptor(manifest=manifest, root=root)


class TestPresence:
    def test_avatar_with_no_assets_is_valid_but_warned(self, sample_descriptor: AvatarDescriptor):
        report = validate_avatar(sample_descriptor)

        assert report.is_valid
        assert any("no assets" in warning for warning in report.warnings)

    def test_missing_directory_is_an_error(self, tmp_path: Path):
        report = validate_avatar(_descriptor(tmp_path / "gone"))

        assert not report.is_valid
        assert "does not exist" in report.errors[0]

    def test_missing_required_asset_is_an_error(self, tmp_path: Path):
        descriptor = _descriptor(tmp_path, assets=[{"path": "model/w.onnx", "role": "weights"}])
        report = validate_avatar(descriptor)

        assert not report.is_valid
        assert any("asset missing" in error for error in report.errors)

    def test_missing_optional_asset_is_only_a_warning(self, tmp_path: Path):
        descriptor = _descriptor(
            tmp_path, assets=[{"path": "extra.bin", "role": "lut", "optional": True}]
        )
        report = validate_avatar(descriptor)

        assert report.is_valid
        assert any("asset missing" in warning for warning in report.warnings)

    def test_empty_asset_is_an_error(self, tmp_path: Path):
        (tmp_path / "model").mkdir()
        (tmp_path / "model" / "w.onnx").touch()

        report = validate_avatar(
            _descriptor(tmp_path, assets=[{"path": "model/w.onnx", "role": "weights"}])
        )

        assert not report.is_valid
        assert any("is empty" in error for error in report.errors)

    def test_missing_preview_is_only_a_warning(self, tmp_path: Path):
        report = validate_avatar(_descriptor(tmp_path, preview="preview.png"))

        assert report.is_valid
        assert any("preview image is missing" in warning for warning in report.warnings)


class TestIntegrity:
    def test_size_mismatch_is_caught_without_hashing(self, tmp_path: Path):
        """A truncated download is detectable for free; report it either way."""
        (tmp_path / "w.bin").write_bytes(b"12345")

        report = validate_avatar(
            _descriptor(
                tmp_path,
                assets=[{"path": "w.bin", "role": "weights", "size_bytes": 999}],
            )
        )

        assert not report.is_valid
        assert any("size mismatch" in error for error in report.errors)

    def test_checksum_verification_passes_for_intact_files(self, tmp_path: Path):
        payload = b"model-weights"
        (tmp_path / "w.bin").write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()

        report = validate_avatar(
            _descriptor(tmp_path, assets=[{"path": "w.bin", "role": "weights", "sha256": digest}]),
            verify_checksums=True,
        )

        assert report.is_valid
        assert report.checked_integrity

    def test_checksum_mismatch_is_an_error(self, tmp_path: Path):
        (tmp_path / "w.bin").write_bytes(b"tampered")

        report = validate_avatar(
            _descriptor(
                tmp_path,
                assets=[{"path": "w.bin", "role": "weights", "sha256": "0" * 64}],
            ),
            verify_checksums=True,
        )

        assert not report.is_valid
        assert any("checksum mismatch" in error for error in report.errors)

    def test_checksums_are_skipped_when_not_requested(self, tmp_path: Path):
        (tmp_path / "w.bin").write_bytes(b"tampered")

        report = validate_avatar(
            _descriptor(
                tmp_path,
                assets=[{"path": "w.bin", "role": "weights", "sha256": "0" * 64}],
            ),
            verify_checksums=False,
        )

        assert report.is_valid
        assert not report.checked_integrity


class TestHashing:
    def test_streamed_digest_matches_hashlib(self, tmp_path: Path):
        payload = b"x" * (2 * 1024 * 1024 + 17)  # spans several chunks
        target = tmp_path / "big.bin"
        target.write_bytes(payload)

        assert compute_sha256(target) == hashlib.sha256(payload).hexdigest()


class TestReport:
    def test_describe_summarises_a_clean_avatar(self, tmp_path: Path):
        (tmp_path / "w.bin").write_bytes(b"data")
        (tmp_path / "preview.png").write_bytes(b"img")

        report = validate_avatar(
            _descriptor(
                tmp_path,
                preview="preview.png",
                assets=[{"path": "w.bin", "role": "weights"}],
            )
        )

        assert report.describe() == "avatar_test: ok"
