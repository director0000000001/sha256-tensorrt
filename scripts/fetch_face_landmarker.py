"""Fetch the MediaPipe Face Landmarker model bundle.

The application never downloads anything (`models.allow_downloads = false`, see
docs/PRIVACY.md). This is a developer/setup step, run once per machine.

The bundle is *not* committed — `models/downloads/` and `*.task` are gitignored,
because model files are large and redistribution terms vary. What *is* committed
is `models/metadata/face_landmarker.json`: the URL, the version, the size, and
the SHA-256 of exactly what was downloaded. That record is what makes a build
reproducible without putting binaries in Git.

Usage::

    python scripts/fetch_face_landmarker.py           # download if absent
    python scripts/fetch_face_landmarker.py --force   # re-download
    python scripts/fetch_face_landmarker.py --verify  # check the existing file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

# Version "1" rather than "latest": a floating pointer would silently change the
# model under a benchmark, and B003's numbers have to mean something later.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_VERSION = "float16/1"

# Apache-2.0 per the MediaPipe model card. Recorded here so the licence travels
# with the artifact rather than living only in an evaluation document.
MODEL_LICENSE = "Apache-2.0"
MODEL_CARD = (
    "https://storage.googleapis.com/mediapipe-assets/"
    "MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "models" / "downloads" / "face_landmarker.task"
METADATA = REPO_ROOT / "models" / "metadata" / "face_landmarker.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recorded_checksum() -> str | None:
    """The SHA-256 from a previous run, if this machine has one."""
    if not METADATA.exists():
        return None
    try:
        return str(json.loads(METADATA.read_text(encoding="utf-8"))["sha256"])
    except (ValueError, KeyError):
        return None


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")

    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        if response.status != 200:
            msg = f"HTTP {response.status} from {url}"
            raise RuntimeError(msg)
        partial.write_bytes(response.read())

    # Rename only after a complete read, so an interrupted download can never
    # be mistaken for a valid model file on the next run.
    partial.replace(target)


def write_metadata(target: Path, checksum: str) -> None:
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "name": "face_landmarker",
        "task": "MediaPipe Face Landmarker (478 landmarks, 52 blendshapes)",
        "url": MODEL_URL,
        "version": MODEL_VERSION,
        "license": MODEL_LICENSE,
        "model_card": MODEL_CARD,
        "sha256": checksum,
        "bytes": target.stat().st_size,
        "downloaded_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "local_path": str(target.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    METADATA.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {METADATA.relative_to(REPO_ROOT)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    parser.add_argument("--verify", action="store_true", help="Only check the existing file.")
    args = parser.parse_args(argv)

    expected = recorded_checksum()

    if args.verify:
        if not TARGET.exists():
            print(f"MISSING: {TARGET}")
            return 1
        actual = sha256_of(TARGET)
        if expected is None:
            print(f"No recorded checksum to compare against. File is sha256:{actual}")
            return 1
        if actual != expected:
            print(f"MISMATCH\n  expected {expected}\n  actual   {actual}")
            return 1
        print(f"OK — {TARGET.name} matches the recorded checksum ({actual[:16]}...)")
        return 0

    if TARGET.exists() and not args.force:
        print(f"Already present: {TARGET.relative_to(REPO_ROOT)}")
        actual = sha256_of(TARGET)
        if expected is not None and actual != expected:
            print(f"WARNING: checksum differs from the recorded value ({expected[:16]}...)")
            return 1
        if expected is None:
            write_metadata(TARGET, actual)
        return 0

    download(MODEL_URL, TARGET)
    actual = sha256_of(TARGET)

    if expected is not None and actual != expected:
        print(
            f"WARNING: the downloaded file does not match the recorded checksum.\n"
            f"  recorded {expected}\n  downloaded {actual}\n"
            f"The upstream artifact changed. Re-benchmark before trusting old numbers."
        )

    size_mb = TARGET.stat().st_size / (1024 * 1024)
    print(f"Saved {TARGET.relative_to(REPO_ROOT)} ({size_mb:.2f} MiB)")
    print(f"sha256:{actual}")
    write_metadata(TARGET, actual)
    return 0


if __name__ == "__main__":
    sys.exit(main())
