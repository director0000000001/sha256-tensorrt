"""Save one frame every N seconds of a clip, into the avatar's own folder.

The selective extractor (``extract_avatar_frames.py``) answers "which single
frame is the best open mouth" and throws the rest away. This one throws nothing
away: it walks the clip at a fixed interval and writes every frame it lands on,
so the whole take is on disk as stills and can be looked through, re-judged, or
fed to something that has not been written yet.

**What this does not do, and it matters.** A frame in this folder is not a
source the avatar uses. The engine loads exactly what the avatar's
``metadata.json`` lists under ``assets`` and nothing else, each entry naming the
role it plays or the yaw it was measured at — the renderer chooses between them
by that. A folder of unlabelled frames has neither, and a tenth of a second
apart most of them are the same face anyway. Two things have to happen before
any of this is used by the application:

1.  something has to say *what each frame is* — which is the classification
    ``extract_avatar_frames.py`` performs, with the renderer's own selector;
2.  the chosen ones have to be listed in ``metadata.json``.

Dumping frames is still worth doing before that: it is the raw material, it
costs nothing but disk, and a clip that has been deleted has to be paid for
again.

Usage::

    python scripts/dump_clip_frames.py clip.mp4 another.mp4
    python scripts/dump_clip_frames.py clip.mp4 --interval 0.05
    python scripts/dump_clip_frames.py clip.mp4 --format png     # lossless, ~8x larger
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AVATAR = REPO_ROOT / "avatars" / "my_avatar"

# A tenth of a second. At 24 fps that is every second or third frame — dense
# enough that a held expression contributes several frames and a movement is
# sampled all the way through, without writing the whole file out.
DEFAULT_INTERVAL = 0.1

# JPEG at 95 rather than PNG, and the reason is size: these clips are 4K, where
# one PNG frame is 8-10 MB and one clip alone would be most of a gigabyte. At 95
# the difference is invisible against what the warp does to the pixels
# afterwards. `--format png` is there for when it is not.
DEFAULT_QUALITY = 95


def slug(name: str) -> str:
    """A folder name from a file name, keeping it readable and path-safe."""
    kept = [character if character.isalnum() else "-" for character in name.lower()]
    return "-".join(part for part in "".join(kept).split("-") if part)


def write_image(path: Path, image: np.ndarray, quality: int) -> None:
    """Encode in memory and write the bytes.

    ``cv2.imwrite`` goes through a narrow-string path on Windows and fails
    silently on anything outside the system codepage. Encoding first and letting
    pathlib write the bytes has no such limit.
    """
    import cv2

    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if path.suffix == ".jpg" else []
    ok, buffer = cv2.imencode(path.suffix, image, params)
    if not ok:
        raise RuntimeError(f"Could not encode {path.name}")
    path.write_bytes(buffer.tobytes())


def dump(video: Path, out: Path, interval: float, quality: int, suffix: str) -> dict[str, object]:
    """Write one frame per ``interval`` seconds of ``video`` into ``out``."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"{video.name} reports no frame rate; cannot sample by time.")

    out.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, object]] = []
    index = 0
    # Sequential read rather than seeking: seeking by timestamp lands on the
    # nearest keyframe in some containers, which would silently repeat frames.
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        moment = index / fps
        wanted = round(moment / interval) * interval
        # Take the frame nearest each multiple of the interval, once each.
        if abs(moment - wanted) <= 0.5 / fps and (
            not written or wanted > float(written[-1]["time_seconds"]) + interval / 2
        ):
            name = f"{len(written):04d}_{round(wanted * 1000):05d}ms{suffix}"
            write_image(out / name, frame, quality)
            written.append({"file": name, "time_seconds": round(wanted, 3), "frame": index})
        index += 1
    capture.release()

    (out / "frames.json").write_text(
        json.dumps(
            {
                "source": str(video),
                "fps": round(fps, 3),
                "frames_in_clip": count,
                "resolution": [width, height],
                "interval_seconds": interval,
                "frames": written,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    size = sum((out / str(row["file"])).stat().st_size for row in written)
    return {
        "video": video.name,
        "written": len(written),
        "fps": fps,
        "resolution": (width, height),
        "megabytes": size / 1e6,
        "out": out,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path, nargs="+", help="Clips to sample.")
    parser.add_argument(
        "--avatar",
        type=Path,
        default=DEFAULT_AVATAR,
        help="Avatar folder; frames go in its `frames/<clip>/` subfolder.",
    )
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY)
    parser.add_argument(
        "--name",
        action="append",
        default=[],
        help="Folder name for each clip, in order. Defaults to the file name.",
    )
    args = parser.parse_args(argv)

    missing = [video for video in args.videos if not video.is_file()]
    if missing:
        for video in missing:
            print(f"Not found: {video}")
        return 1

    suffix = f".{args.format}"
    results = []
    for position, video in enumerate(args.videos):
        folder = args.name[position] if position < len(args.name) else slug(video.stem)
        out = args.avatar / "frames" / folder
        print(f"{video.name} -> {out}")
        results.append(dump(video, out, args.interval, args.quality, suffix))

    print(f"\n{'clip':<28}{'frames':>8}{'fps':>7}{'resolution':>14}{'MB':>9}")
    for row in results:
        width, height = row["resolution"]  # type: ignore[misc]
        print(
            f"{row['video']!s:<28}{row['written']:>8}{row['fps']:>7.0f}"
            f"{f'{width}x{height}':>14}{row['megabytes']:>9.0f}"
        )
    total = sum(int(row["written"]) for row in results)
    megabytes = sum(float(row["megabytes"]) for row in results)
    print(f"\n{total} frames, {megabytes:.0f} MB, under {args.avatar / 'frames'}")
    print(
        "\nThese are raw frames, not avatar sources: the engine loads only what\n"
        "metadata.json lists. Classify them with extract_avatar_frames.py before\n"
        "any of them can be used."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
