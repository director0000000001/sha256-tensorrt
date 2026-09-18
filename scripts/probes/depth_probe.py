"""The crop's depth through a turn (day 20, his "a quick turn left-right
zooms the face out"): a clip through the tracker and the engine as the app
runs them, and per frame the pose's z, the depth the crop was sized from,
the crop's raw and filtered size, and -- read by a second, unfiltered
tracker on the output and a third on the camera frame, as measure_follow
reads them -- the avatar's face height against his.

``--sweep depth_hold_yaw=0,35`` compares the hold off against on: per
window (the first frame found after each loss, and each return under the
threshold; or ``--windows``) the crop's size against where it settles,
frame by frame; and the proof that no frame at or under the threshold
changed -- the raw size to the byte on every such frame, the filtered size
on every such frame more than ``--settle`` frames after a held one, and the
output pixels on the frontal ``--check`` frames within the GPU's own
run-to-run noise (day 19's rule: none over 8 grey levels).

Usage::

    uv run python scripts/probes/depth_probe.py --clip recordings/driving-turn.mkv \\
        --sweep depth_hold_yaw=0,35

The CSVs are numbers only (no picture of his face).
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "probes"))

from capture_replay import parse_options, parse_sweep  # noqa: E402

from sha256.avatar.engine_liveportrait import (  # noqa: E402
    CROP_SIZE,
    DEFAULT_WEIGHTS_RELATIVE_PATH,
    FACE_POINTS,
    LivePortraitEngine,
    _face_height,
)
from sha256.avatar.repository import FilesystemAvatarRepository  # noqa: E402
from sha256.calibration.profile import PROFILE_FILENAME, DriverProfile  # noqa: E402
from sha256.config.loader import load_settings  # noqa: E402
from sha256.config.paths import AppPaths  # noqa: E402
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FrameMetadata, VideoFrame  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)

COLUMNS = (
    "frame",
    "seen",
    "rel_yaw",
    "pitch",
    "z",
    "z_used",
    "raw_size",
    "size",
    "his_height",
    "avatar_height",
    "ratio",
)


def make_tracker(settings, models: Path, *, filtered: bool) -> MediaPipeFaceTracker:
    return MediaPipeFaceTracker(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        smoothing=settings.tracking.smoothing if filtered else 0.0,
        mouth_smoothing=settings.tracking.mouth_smoothing if filtered else 0.0,
        min_detection_confidence=settings.tracking.min_detection_confidence,
        min_presence_confidence=settings.tracking.min_presence_confidence,
        min_tracking_confidence=settings.tracking.min_tracking_confidence,
        redetect_in_crop=settings.tracking.redetect_in_crop if filtered else False,
    )


def face_height_px(tracking, width: int, height: int) -> float:
    if not tracking.face_detected or tracking.landmarks is None:
        return math.nan
    pts = tracking.landmarks.points[:FACE_POINTS, :2] * (width, height)
    return float(_face_height(pts))


def run(
    clip: Path,
    avatar_id: str,
    options: dict[str, Any],
    engine: LivePortraitEngine,
    avatar,
    settings,
    models: Path,
    last: int,
    check: set[int],
) -> tuple[list[dict[str, float]], dict[int, np.ndarray]]:
    import cv2

    avatar.manifest.engine_options.clear()
    avatar.manifest.engine_options.update(options)
    engine.unload_avatar(avatar_id)
    engine.load_avatar(avatar)
    tracker = make_tracker(settings, models, filtered=True)
    truth = make_tracker(settings, models, filtered=False)
    reader = make_tracker(settings, models, filtered=False)
    for t in (tracker, truth, reader):
        t.start()
    # The raw (unfiltered) face height the crop is sized from, hooked off
    # the engine's own sizing -- None before the anchor, or without a pose.
    raw: dict[str, float | None] = {"height": None}
    original = LivePortraitEngine._driver_face_height

    def hooked(loaded, head_pose):
        value = original(loaded, head_pose)
        raw["height"] = value
        return value

    LivePortraitEngine._driver_face_height = staticmethod(hooked)  # type: ignore[method-assign]
    rows: list[dict[str, float]] = []
    outputs: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(clip))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    index = 0
    try:
        while index <= last:
            ok, image = capture.read()
            if not ok:
                break
            height, width = image.shape[:2]
            frame = VideoFrame(
                data=np.ascontiguousarray(image),
                pixel_format=PixelFormat.BGR24,
                metadata=FrameMetadata(frame_id=index, capture_timestamp=index / fps),
            )
            raw["height"] = None
            tracking = tracker.process(frame)
            out = engine.animate(avatar_id, tracking, frame)
            loaded = engine._loaded[avatar_id]
            row: dict[str, float] = dict.fromkeys(COLUMNS, math.nan)
            row["frame"] = index
            row["seen"] = float(tracking.face_detected)
            pose = tracking.head_pose
            if tracking.face_detected and pose is not None:
                row["rel_yaw"] = float(pose.yaw) - (loaded.rest_yaw or 0.0)
                row["pitch"] = float(pose.pitch)
                row["z"] = abs(float(pose.translation[2]))
                if raw["height"] is not None and loaded.face_height_rest and loaded.z_rest:
                    scale = float(loaded.options.get("crop_scale", 2.2))
                    row["raw_size"] = raw["height"] * scale
                    row["z_used"] = loaded.face_height_rest * loaded.z_rest / raw["height"]
                if loaded.last is not None:
                    to_frame = loaded.last[1]
                    row["size"] = float(math.hypot(to_frame[0, 0], to_frame[1, 0]) * CROP_SIZE)
                his = truth.process(frame)
                row["his_height"] = face_height_px(his, width, height)
                seen = reader.process(
                    VideoFrame(
                        data=np.ascontiguousarray(out.data),
                        pixel_format=PixelFormat.BGR24,
                        metadata=FrameMetadata(frame_id=index, capture_timestamp=index / fps),
                    )
                )
                row["avatar_height"] = face_height_px(seen, width, height)
                if row["his_height"] > 0:
                    row["ratio"] = row["avatar_height"] / row["his_height"]
            if index in check:
                outputs[index] = np.array(out.data, copy=True)
            rows.append(row)
            index += 1
    finally:
        capture.release()
        for t in (tracker, truth, reader):
            t.stop()
        LivePortraitEngine._driver_face_height = staticmethod(original)  # type: ignore[method-assign]
    return rows, outputs


def auto_windows(rows: list[dict[str, float]], threshold: float) -> list[tuple[int, str]]:
    """The first frame found after a loss of 5+ frames, and the first frame
    back at or under the threshold after 5+ frames past it."""
    found: list[tuple[int, str]] = []
    lost = 0
    past = 0
    for row in rows:
        seen = row["seen"] > 0
        if not seen:
            lost += 1
            continue
        if lost >= 5:
            found.append((int(row["frame"]), f"found after {lost} lost frames"))
        lost = 0
        if threshold > 0 and abs(row["rel_yaw"]) > threshold:
            past += 1
            continue
        if past >= 5:
            found.append((int(row["frame"]), f"back under {threshold:.0f} after {past} frames"))
        past = 0
    return found


def settled(rows: list[dict[str, float]], start: int, span: int, key: str) -> float:
    values = [
        r[key]
        for r in rows[start + span : start + 2 * span]
        if r["seen"] > 0 and not math.isnan(r[key])
    ]
    return statistics.median(values) if values else math.nan


def describe_window(
    label: str, runs: dict[str, list[dict[str, float]]], start: int, span: int
) -> None:
    print(f"\n  window f{start} ({label}), size and avatar/his against where they settle:")
    print("    frame  yaw  " + "  ".join(f"{name:>22}" for name in runs))
    sub = "size%  raw%  ratio%"
    print("    " + " " * 11 + "  ".join(f"{sub:>22}" for _ in runs))
    bases = {
        name: (settled(rows, start, span, "size"), settled(rows, start, span, "ratio"))
        for name, rows in runs.items()
    }
    for offset in range(-3, span):
        i = start + offset
        parts = []
        yaw = math.nan
        for name, rows in runs.items():
            if i < 0 or i >= len(rows) or rows[i]["seen"] == 0:
                parts.append(f"{'lost':>22}")
                continue
            r = rows[i]
            yaw = r["rel_yaw"]
            base_size, base_ratio = bases[name]
            size = 100.0 * (r["size"] / base_size - 1.0) if base_size else math.nan
            rawp = 100.0 * (r["raw_size"] / base_size - 1.0) if base_size else math.nan
            ratio = 100.0 * (r["ratio"] / base_ratio - 1.0) if base_ratio else math.nan
            parts.append(f"{size:+6.1f} {rawp:+6.1f} {ratio:+7.1f}")
        print(f"    f{i:<5d}{yaw:+5.0f}  " + "  ".join(parts))
    for name, rows in runs.items():
        base_size, _ = bases[name]
        if not base_size:
            continue
        window = [
            100.0 * (r["size"] / base_size - 1.0)
            for r in rows[start : start + span]
            if r["seen"] > 0 and not math.isnan(r["size"])
        ]
        if window:
            under = sum(1 for v in window if v < -1.0)
            print(
                f"    {name}: size min {min(window):+.1f}% max {max(window):+.1f}%, "
                f"{under} frames more than 1% small"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--last", type=int, default=10**9)
    parser.add_argument("--avatar", default="my_avatar_neural")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--sweep", default=None, metavar="KEY=v1,v2,...")
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--windows", default=None, help="Window starts, comma-separated.")
    parser.add_argument("--span", type=int, default=30, help="Frames per window.")
    parser.add_argument("--threshold", type=float, default=35.0)
    parser.add_argument("--settle", type=int, default=45)
    parser.add_argument("--check", default="30,60,90,110,338,499,900,1150")
    parser.add_argument("--out-prefix", type=Path, default=Path("recordings/day20/depth"))
    args = parser.parse_args()

    overrides = parse_options(args.option)
    sweep_key, sweep_values = parse_sweep(args.sweep)
    check = {int(v) for v in args.check.split(",") if v.strip()}

    paths = AppPaths.resolve()
    settings = load_settings(paths)
    models = paths.resolve_relative(settings.models.root_path)
    repo = FilesystemAvatarRepository(paths.resolve_relative(settings.avatars.library_path))
    profile = None if args.no_profile else DriverProfile.load(paths.configs / PROFILE_FILENAME)
    driver_options = profile.overlay() if profile else {}
    engine = LivePortraitEngine(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        weights=models / DEFAULT_WEIGHTS_RELATIVE_PATH,
        driver_options=driver_options,
    )
    engine.start()
    avatar = repo.get(args.avatar)
    base = copy.deepcopy(avatar.manifest.engine_options)
    print(f"{args.clip}: avatar {args.avatar}, profile {'none' if not profile else 'yes'}")
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    runs: dict[str, list[dict[str, float]]] = {}
    pictures: dict[str, dict[int, np.ndarray]] = {}
    try:
        for value in sweep_values:
            options: dict[str, Any] = dict(base)
            options.update(overrides)
            if sweep_key is not None:
                options[sweep_key] = value
            name = "base" if sweep_key is None else f"{sweep_key}={value}"
            rows, outputs = run(
                args.clip, args.avatar, options, engine, avatar, settings, models, args.last, check
            )
            runs[name] = rows
            pictures[name] = outputs
            tag = "" if sweep_key is None else f"-{sweep_key}{value}"
            out = args.out_prefix.parent / f"{args.out_prefix.name}-{args.clip.stem}{tag}.csv"
            with out.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=COLUMNS)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: f"{row[k]:.4f}" for k in COLUMNS})
            loaded = engine._loaded[args.avatar]
            lost = sum(1 for r in rows if r["seen"] == 0)
            z_rest = "?" if loaded.z_rest is None else f"{loaded.z_rest:.2f}"
            rest_h = "?" if loaded.face_height_rest is None else f"{loaded.face_height_rest:.0f}"
            print(
                f"  {name}: {len(rows)} frames, {lost} lost; rest yaw "
                f"{(loaded.rest_yaw or 0.0):+.1f}, z_rest {z_rest}, face height rest {rest_h}"
                f" -> {out}"
            )
    finally:
        engine.stop()

    first = next(iter(runs.values()))
    if args.windows:
        windows = [(int(v), "asked") for v in args.windows.split(",") if v.strip()]
    else:
        windows = auto_windows(first, args.threshold)
    for start, label in windows:
        describe_window(label, runs, start, args.span)

    if len(runs) >= 2:
        names = list(runs)
        a, b = runs[names[0]], runs[names[1]]
        limit = f"{args.threshold:.0f}"
        print(f"\n  {names[0]} against {names[1]}, frames at or under {limit} degrees:")
        raw_diff = []
        size_diff = []
        since_past = 10**9
        for ra, rb in zip(a, b, strict=True):
            if ra["seen"] == 0:
                continue
            if abs(ra["rel_yaw"]) > args.threshold:
                since_past = 0
                continue
            since_past += 1
            both_nan = math.isnan(ra["raw_size"]) and math.isnan(rb["raw_size"])
            if not both_nan and ra["raw_size"] != rb["raw_size"]:
                raw_diff.append(int(ra["frame"]))
            if since_past > args.settle and ra["size"] != rb["size"]:
                size_diff.append(int(ra["frame"]))
        print(f"    raw crop size differs on {len(raw_diff)} frames {raw_diff[:8]}")
        print(
            f"    filtered size differs on {len(size_diff)} frames more than {args.settle} "
            f"frames after a held one {size_diff[:8]}"
        )
        pa, pb = pictures[names[0]], pictures[names[1]]
        for index in sorted(check):
            if index in pa and index in pb:
                diff = np.abs(pa[index].astype(np.int16) - pb[index].astype(np.int16))
                row = a[index]
                print(
                    f"    f{index} (yaw {row['rel_yaw']:+.0f}): output max diff {int(diff.max())} "
                    f"levels, {int((diff > 8).any(axis=2).sum())} px over 8"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
