"""Fetch MediaPipe's multiclass selfie segmenter -- the avatar's hair (day 17).

The application never downloads anything (`models.allow_downloads = false`, see
docs/PRIVACY.md). This is a developer/setup step, run once per machine, like
``fetch_face_landmarker.py``: the model lands beside the face landmarker under
``models/downloads/`` (gitignored) and what is committed is the record under
``models/metadata/`` -- the URL, the version, the size and the SHA-256 of
exactly what was downloaded.

The model segments a picture into background / hair / body skin / face skin /
clothes / accessories at 256x256; the engine reads its hair class off every
rendered crop when Menu -> Avatar's hair is on (see ``sha256.avatar.hair``).
Without the file the switch simply stays off and says so.

Usage::

    python scripts/fetch_selfie_segmenter.py           # download if absent
    python scripts/fetch_selfie_segmenter.py --force   # re-download
    python scripts/fetch_selfie_segmenter.py --verify  # check the existing file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

# "latest" is the only version Google publishes for this model; the checksum
# recorded on the first download is what pins it for every later machine.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
MODEL_VERSION = "float32/latest"
MODEL_LICENSE = "Apache-2.0"
MODEL_CARD = (
    "https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "models" / "downloads" / "selfie_multiclass_256x256.tflite"
METADATA = REPO_ROOT / "models" / "metadata" / "selfie_multiclass.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recorded_checksum() -> str | None:
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
    partial.replace(target)


def write_metadata(target: Path, checksum: str) -> None:
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "name": "selfie_multiclass_256x256",
        "task": (
            "MediaPipe Image Segmenter, multiclass selfie "
            "(background, hair, body skin, face skin, clothes, accessories)"
        ),
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
        print(f"OK -- {TARGET.name} matches the recorded checksum ({actual[:16]}...)")
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
            f"The upstream 'latest' may have moved; review before trusting it."
        )
        return 1
    write_metadata(TARGET, actual)
    print(f"OK -- {TARGET.relative_to(REPO_ROOT)} ({TARGET.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
