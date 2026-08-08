"""Avatar asset validation.

Validation runs *before* an avatar is loaded, so a missing or corrupted file
produces a clear message in the gallery rather than a crash halfway through a
GPU allocation.

Three levels, escalating in cost:

1.  **Manifest** — schema-valid. Already enforced at parse time by the
    repository, so a descriptor that exists is manifest-valid by construction.
2.  **Presence** — every non-optional asset exists and is non-empty. Cheap
    enough to run on the whole library at startup.
3.  **Integrity** — declared SHA-256 digests match. Expensive (reads every
    byte), so it is opt-in and reserved for imports and explicit checks.

The digest is what makes downloadable avatar packs safe to support later: an
avatar whose weights do not match its manifest is rejected rather than loaded.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from sha256.core.avatar import AssetRef, AvatarDescriptor

__all__ = ["AvatarValidationReport", "validate_avatar"]

_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class AvatarValidationReport:
    """Outcome of validating one avatar."""

    avatar_id: str
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checked_integrity: bool = False

    @property
    def is_valid(self) -> bool:
        """No blocking errors. Warnings do not prevent loading."""
        return not self.errors

    def describe(self) -> str:
        if self.is_valid and not self.warnings:
            return f"{self.avatar_id}: ok"
        parts = [f"{self.avatar_id}:"]
        parts.extend(f"  error: {message}" for message in self.errors)
        parts.extend(f"  warning: {message}" for message in self.warnings)
        return "\n".join(parts)


@dataclass(slots=True)
class _Accumulator:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_avatar(
    descriptor: AvatarDescriptor, *, verify_checksums: bool = False
) -> AvatarValidationReport:
    """Check that an avatar's declared assets are present and, optionally, intact.

    Args:
        descriptor: The avatar to check.
        verify_checksums: Hash every asset that declares a digest. Slow.
    """
    acc = _Accumulator()

    if not descriptor.root.is_dir():
        return AvatarValidationReport(
            avatar_id=descriptor.id,
            errors=(f"avatar directory does not exist: {descriptor.root}",),
        )

    _check_preview(descriptor, acc)

    for ref in descriptor.manifest.assets:
        _check_asset(descriptor.asset_path(ref), ref, acc, verify_checksums=verify_checksums)

    if not descriptor.manifest.assets:
        acc.warnings.append("manifest declares no assets; the engine may have nothing to load")

    return AvatarValidationReport(
        avatar_id=descriptor.id,
        errors=tuple(acc.errors),
        warnings=tuple(acc.warnings),
        checked_integrity=verify_checksums,
    )


def _check_preview(descriptor: AvatarDescriptor, acc: _Accumulator) -> None:
    preview = descriptor.preview_path
    if preview is None:
        # A warning, not an error: an avatar without a preview still works, it
        # just shows a generic tile in the gallery.
        acc.warnings.append("no preview image declared")
    elif not preview.is_file():
        acc.warnings.append(f"preview image is missing: {descriptor.manifest.preview}")


def _check_asset(path: Path, ref: AssetRef, acc: _Accumulator, *, verify_checksums: bool) -> None:
    if not path.is_file():
        message = f"asset missing: {ref.path} (role={ref.role})"
        (acc.warnings if ref.optional else acc.errors).append(message)
        return

    size = path.stat().st_size
    if size == 0:
        acc.errors.append(f"asset is empty: {ref.path}")
        return

    # A size mismatch is a strong, near-free signal of a truncated download —
    # worth reporting even when full hashing is switched off.
    if ref.size_bytes is not None and size != ref.size_bytes:
        acc.errors.append(
            f"asset size mismatch for {ref.path}: expected {ref.size_bytes} bytes, found {size}"
        )
        return

    if verify_checksums and ref.sha256 is not None:
        digest = compute_sha256(path)
        if digest != ref.sha256:
            acc.errors.append(
                f"checksum mismatch for {ref.path}: expected {ref.sha256}, computed {digest}"
            )


def compute_sha256(path: Path) -> str:
    """Streamed SHA-256 of a file, as lowercase hex.

    Streamed because avatar weights can be hundreds of megabytes and reading
    one into memory to hash it would spike RSS for no reason.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
