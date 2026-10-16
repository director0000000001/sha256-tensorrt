"""One turned capture re-rendered by a given tree's engine, and what the
output does at the turn: the drawn face's yaw against his, where the
paste's near edge sits against his face's rim, and the eye row -- so two
builds (today's and the 09-18 push, in a worktree) can be read side by
side on the same press (day 23, his words: "on 09-18 the face followed me
further when I turned").

The engine is loaded from ``--src`` (a tree's ``src`` directory; its own
``models``/``avatars`` resolve from there), the anchor learned from the
capture's own frame (the old tree knows nothing of today's settle keys),
the capture's last frame driven 300 times as jaw_band.py does.

Usage::

    uv run python scripts/probes/turn_compare.py --src src --capture captures/moment-20261015-191804
    uv run python scripts/probes/turn_compare.py --src <worktree>/src --capture ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--avatar", default="my_avatar_neural")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Learn the anchor from the capture's frame (a turned frame never freezes one).",
    )
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

    def tracking_at(data: Any, index: int) -> FaceTrackingResult:
        names = tuple(str(n) for n in data["names"])
        yaw, pitch, roll = (float(v) for v in data["pose"][index])
        return FaceTrackingResult(
            metadata=FrameMetadata(
                frame_id=index, capture_timestamp=float(data["timestamps"][index])
            ),
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

    import json

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

    camera = cv2.imread(str(args.capture / "camera.png"))
    data = np.load(args.capture / "tracking.npz", allow_pickle=False)
    detected = data["detected"]
    last = int(np.where(detected)[0].max())
    prefix = "anchor_"
    state = {
        name[len(prefix) :]: data[name]
        for name in data.files
        if name.startswith(prefix) and name != "anchor_avatar"
    }
    if not args.no_seed and "exp_d0" in state:
        engine.seed_anchor(args.avatar, state)
    try:
        for index in range(last + 1):
            frame = VideoFrame(
                data=camera,
                pixel_format=PixelFormat.BGR24,
                metadata=FrameMetadata(
                    frame_id=index, capture_timestamp=float(data["timestamps"][index])
                ),
            )
            out = engine.animate(args.avatar, tracking_at(data, last), frame)
    finally:
        engine.stop()
    rendered = np.ascontiguousarray(out.data)
    if out.pixel_format is PixelFormat.RGB24:
        rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
    if args.save is not None:
        cv2.imwrite(str(args.save), rendered)

    reader = MediaPipeFaceTracker(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        smoothing=0.0,
        mouth_smoothing=0.0,
        running_mode="image",
    )
    reader.start()
    try:
        result = reader.process(
            VideoFrame(
                data=rendered,
                pixel_format=PixelFormat.BGR24,
                metadata=FrameMetadata(frame_id=0, capture_timestamp=0.0),
            )
        )
    finally:
        reader.stop()
    h, w = camera.shape[:2]
    his = data["landmarks"][last][:468, :2] * (w, h)
    his_yaw, his_pitch = (float(v) for v in data["pose"][last][:2])
    drawn_yaw = drawn_pitch = float("nan")
    drawn_eye = float("nan")
    if result.face_detected and result.head_pose is not None and result.landmarks is not None:
        drawn_yaw, drawn_pitch = float(result.head_pose.yaw), float(result.head_pose.pitch)
        pts = result.landmarks.points[:, :2] * (w, h)
        drawn_eye = float(pts[[159, 145, 386, 374], 1].mean())
    his_eye = float(his[[159, 145, 386, 374], 1].mean())
    # The pasted region: where the output differs from the camera frame.
    diff = np.abs(rendered.astype(np.int16) - camera.astype(np.int16)).max(axis=2) > 8
    rim = his[
        [
            10,
            338,
            297,
            332,
            284,
            251,
            389,
            356,
            454,
            323,
            361,
            288,
            397,
            365,
            379,
            378,
            400,
            377,
            152,
            148,
            176,
            149,
            150,
            136,
            172,
            58,
            132,
            93,
            234,
            127,
            162,
            21,
            54,
            103,
            67,
            109,
        ]
    ]
    rows = {
        "brow": int(his[[105, 334], 1].mean()),
        "eye": int(his_eye),
        "nose": int(his[2, 1]),
        "mouth": int(his[[13, 14], 1].mean()),
    }
    near_left = his_yaw < 0.0  # yaw + = his left (the image's right on the unmirrored frame)
    print(
        f"{args.capture.name} / {args.src}: his yaw {his_yaw:+.1f} pitch {his_pitch:+.1f}; "
        f"drawn yaw {drawn_yaw:+.1f} pitch {drawn_pitch:+.1f}; eye row his {his_eye:.0f} "
        f"drawn {drawn_eye:.0f}; pasted {int(diff.sum())} px"
    )
    print(
        "  row    paste L..R   his rim L..R   near-side gap (rim - paste, px; + = his skin shows)"
    )
    for name, row in rows.items():
        cols = np.where(diff[row])[0]
        if len(cols) == 0:
            print(f"  {name:5}  (nothing pasted on this row)")
            continue
        near = rim[np.abs(rim[:, 1] - row) < 12]
        if len(near) == 0:
            continue
        rim_l, rim_r = float(near[:, 0].min()), float(near[:, 0].max())
        pl, pr = int(cols.min()), int(cols.max())
        gap = (pl - rim_l) if near_left else (rim_r - pr)
        print(f"  {name:5}  {pl:4d}..{pr:4d}    {rim_l:5.0f}..{rim_r:5.0f}    {gap:+6.0f}")


if __name__ == "__main__":
    main()
