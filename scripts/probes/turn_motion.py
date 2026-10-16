"""A recorded session's turn, frame by frame, through a given tree's
engine: his tracked yaw against the drawn face's yaw, whether the frame
was held, and how far the paste reaches toward his near rim -- so two
builds (today's and the 09-18 push in a worktree) can be compared on the
same movement, not one held frame (day 23, his words: "on 09-18 the face
followed me further when I turned").

The frames are the recording's own (clip.mkv) with its own tracking
(tracking.npz, the app's filtered tracker as it ran); the engine starts
at ``--first`` minus a still lead so the anchor is learned as live, the
output read by MediaPipe every ``--every`` frames.

Usage::

    uv run python scripts/probes/turn_motion.py --src src \\
        --recording recordings/calibration-20261015-190122 --first 7000 --last 7330
    uv run python scripts/probes/turn_motion.py --src <worktree>/src --recording ... --first ...

Nothing is written unless ``--save-dir`` is given; what is written is his
face -- delete after looking.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

RIM = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--lead", type=int, default=60, help="Still frames before --first.")
    parser.add_argument("--every", type=int, default=3)
    parser.add_argument("--avatar", default="my_avatar_neural")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--save-dir", type=Path, default=None)
    args = parser.parse_args()
    sys.path.insert(0, str(args.src.resolve()))
    logging.basicConfig(level=logging.WARNING)

    import cv2

    from sha256.avatar.engine_liveportrait import (
        DEFAULT_WEIGHTS_RELATIVE_PATH,
        LivePortraitEngine,
    )
    from sha256.avatar.repository import FilesystemAvatarRepository
    from sha256.config.loader import load_settings
    from sha256.config.paths import AppPaths
    from sha256.core.enums import PixelFormat
    from sha256.core.types import (
        BlendshapeSet,
        FaceTrackingResult,
        FrameMetadata,
        HeadPose,
        LandmarkSet,
        VideoFrame,
    )
    from sha256.tracking.mediapipe_tracker import DEFAULT_MODEL_RELATIVE_PATH, MediaPipeFaceTracker

    data = np.load(args.recording / "tracking.npz", allow_pickle=False)
    names = tuple(str(n) for n in data["names"])
    timestamps = data["timestamps"] if "timestamps" in data.files else np.arange(len(data["pose"]))

    def tracking_at(index: int) -> FaceTrackingResult:
        yaw, pitch, roll = (float(v) for v in data["pose"][index][:3])
        return FaceTrackingResult(
            metadata=FrameMetadata(frame_id=index, capture_timestamp=float(timestamps[index])),
            face_detected=bool(data["detected"][index]),
            landmarks=LandmarkSet(
                points=data["landmarks"][index].astype(np.float32), scheme="mediapipe_478"
            ),
            head_pose=HeadPose(yaw=yaw, pitch=pitch, roll=roll),
            blendshapes=BlendshapeSet(
                names=names,
                weights=data["weights"][index].astype(np.float32),
                scheme="mediapipe_52",
            ),
            confidence=1.0,
        )

    overrides: dict[str, Any] = {}
    for item in args.option:
        key, _, value = item.partition("=")
        overrides[key] = json.loads(value)

    paths = AppPaths.resolve()
    settings = load_settings(paths)
    models = paths.resolve_relative(settings.models.root_path)
    repository = FilesystemAvatarRepository(paths.resolve_relative(settings.avatars.library_path))
    engine = LivePortraitEngine(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        weights=models / DEFAULT_WEIGHTS_RELATIVE_PATH,
    )
    engine.start()
    avatar = repository.get(args.avatar)
    avatar.manifest.engine_options.update(overrides)
    engine.load_avatar(avatar)
    reader = MediaPipeFaceTracker(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        smoothing=0.0,
        mouth_smoothing=0.0,
        running_mode="image",
    )
    reader.start()

    clip = cv2.VideoCapture(str(args.recording / "clip.mkv"))
    start = max(args.first - args.lead, 0)
    clip.set(cv2.CAP_PROP_POS_FRAMES, start)
    print(
        f"{args.recording.name} frames {args.first}..{args.last} / {args.src} overrides {overrides}"
    )
    print(
        f"{'frame':>5} {'his yaw':>8} {'drawn':>6} {'ratio':>6} {'pitch his/drawn':>16} "
        f"{'gap brow/eye':>12} {'top':>5} {'pasted':>7} {'held':>4}"
    )
    rows: list[tuple[float, float]] = []
    try:
        for index in range(start, args.last + 1):
            ok, image = clip.read()
            if not ok:
                break
            frame = VideoFrame(
                data=image,
                pixel_format=PixelFormat.BGR24,
                metadata=FrameMetadata(frame_id=index, capture_timestamp=float(timestamps[index])),
            )
            tracking = tracking_at(index)
            out = engine.animate(args.avatar, tracking, frame)
            if index < args.first or (index - args.first) % args.every:
                continue
            rendered = np.ascontiguousarray(out.data)
            if out.pixel_format is PixelFormat.RGB24:
                rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            if args.save_dir is not None:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.save_dir / f"{index:05d}.png"), rendered)
            result = reader.process(
                VideoFrame(
                    data=rendered,
                    pixel_format=PixelFormat.BGR24,
                    metadata=FrameMetadata(frame_id=index, capture_timestamp=0.0),
                )
            )
            his_yaw = float(data["pose"][index][0])
            his_pitch = float(data["pose"][index][1])
            drawn_yaw = drawn_pitch = float("nan")
            if result.face_detected and result.head_pose is not None:
                drawn_yaw = float(result.head_pose.yaw)
                drawn_pitch = float(result.head_pose.pitch)
            h, w = image.shape[:2]
            his = data["landmarks"][index][:468, :2] * (w, h)
            eye_row = int(his[[159, 145, 386, 374], 1].mean())
            brow_row = int(his[[105, 334], 1].mean())
            diff = np.abs(rendered.astype(np.int16) - image.astype(np.int16)).max(axis=2) > 8
            rim = his[list(RIM)]
            gaps = []
            for row in (brow_row, eye_row):
                cols = np.where(diff[row])[0]
                near = rim[np.abs(rim[:, 1] - row) < 12]
                if len(cols) and len(near):
                    if his_yaw < 0.0:
                        gaps.append(float(cols.min() - near[:, 0].min()))
                    else:
                        gaps.append(float(near[:, 0].max() - cols.max()))
                else:
                    gaps.append(float("nan"))
            gap_brow, gap = gaps
            top = int(np.where(diff.any(axis=1))[0].min()) if diff.any() else -1
            held = getattr(engine._loaded[args.avatar], "holding", False)
            ratio = drawn_yaw / his_yaw if abs(his_yaw) > 5.0 else float("nan")
            if np.isfinite(ratio):
                rows.append((abs(his_yaw), ratio))
            print(
                f"{index:5d} {his_yaw:+8.1f} {drawn_yaw:+6.1f} {ratio:6.2f} "
                f"{his_pitch:+7.1f}/{drawn_pitch:+6.1f} {gap_brow:+6.0f}/{gap:+5.0f} "
                f"{top - brow_row:+5.0f} {int(diff.sum()):7d} {'held' if held else '':>4}"
            )
    finally:
        reader.stop()
        engine.stop()
        clip.release()
    if rows:
        arr = np.array(rows)
        for lo, hi in ((5, 20), (20, 30), (30, 40), (40, 90)):
            sel = arr[(arr[:, 0] >= lo) & (arr[:, 0] < hi)]
            if len(sel):
                median = float(np.median(sel[:, 1]))
                print(f"  |yaw| {lo:2d}..{hi:2d}: drawn/his median {median:.2f} (n={len(sel)})")


if __name__ == "__main__":
    main()
