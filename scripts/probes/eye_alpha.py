"""At a turned press, how much of the avatar covers the person's own
eyes: the paste's alpha at his eye corners and lashes, both eyes, and
where the drawn eyes landed against his -- for a given tree (``--src``)
and options, so a build and its switches can be read side by side on
the same press (day 23, his words: "when I turn left my eye and lashes
remain; when I turn right the brow and eye stretch").

The capture's last frame is driven with the session's anchor (or, with
``--no-seed``, one learned from the frame), 300 times as jaw_band.py
does; the pasted mask is the engine's own (``loaded.last``), warped into
the frame by the same affine the paste used.

Usage::

    uv run python scripts/probes/eye_alpha.py --src src --capture captures/moment-...
    uv run python scripts/probes/eye_alpha.py --src <worktree>/src --capture ... \
        --option outline_fit=0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# MediaPipe: the image-left eye's outer/inner corners and lower lid, the
# image-right eye's, and the brows' tops.
LEFT_EYE = {"outer": 33, "inner": 133, "lower lid": 145, "lash row": 23}
RIGHT_EYE = {"outer": 263, "inner": 362, "lower lid": 374, "lash row": 253}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--avatar", default="my_avatar_neural")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--no-seed", action="store_true")
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--dump", type=Path, default=None, help="alpha, pasted, his points as npz")
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

    data = np.load(args.capture / "tracking.npz", allow_pickle=False)
    names = tuple(str(n) for n in data["names"])

    def tracking_at(index: int) -> FaceTrackingResult:
        yaw, pitch, roll = (float(v) for v in data["pose"][index][:3])
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
    h, w = camera.shape[:2]
    last = int(np.where(data["detected"])[0].max())
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
            out = engine.animate(args.avatar, tracking_at(last), frame)
        kept = engine._loaded[args.avatar].last
    finally:
        engine.stop()
    rendered = np.ascontiguousarray(out.data)
    if out.pixel_format is PixelFormat.RGB24:
        rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
    if args.save is not None:
        cv2.imwrite(str(args.save), rendered)
    if kept is None:
        raise SystemExit("nothing rendered")
    _, to_frame, mask, *rest = kept
    alpha = cv2.warpAffine(mask[:, :, 0], to_frame, (w, h), flags=cv2.INTER_LINEAR)
    extra = rest[0] if rest and isinstance(rest[0], np.ndarray) and rest[0].ndim == 3 else None
    if extra is not None and extra.shape[:2] == mask.shape[:2]:
        alpha = np.maximum(alpha, cv2.warpAffine(extra[:, :, 0], to_frame, (w, h)))
    # Where the output differs from the camera: the paste as it landed.
    pasted = np.abs(rendered.astype(np.int16) - camera.astype(np.int16)).max(axis=2) > 8

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
    his = data["landmarks"][last][:, :2] * (w, h)
    if args.dump is not None:
        np.savez_compressed(args.dump, alpha=alpha, pasted=pasted, his=his, rendered=rendered)
    drawn = (
        result.landmarks.points[:, :2] * (w, h)
        if result.face_detected and result.landmarks is not None
        else None
    )
    his_yaw = float(data["pose"][last][0])
    drawn_yaw = float(result.head_pose.yaw) if result.head_pose is not None else float("nan")
    print(
        f"{args.capture.name} / {args.src} {overrides}: his yaw {his_yaw:+.1f}, "
        f"drawn {drawn_yaw:+.1f}"
    )
    print(f"{'eye / point':>24} {'his x,y':>10} {'alpha':>6} {'pasted':>7} {'drawn dx,dy':>12}")
    for side, points in (("image-left eye", LEFT_EYE), ("image-right eye", RIGHT_EYE)):
        for name, idx in points.items():
            x, y = his[idx]
            xi, yi = round(float(x)), round(float(y))
            a = float(alpha[yi, xi]) if 0 <= yi < h and 0 <= xi < w else float("nan")
            p = bool(pasted[yi, xi]) if 0 <= yi < h and 0 <= xi < w else False
            d = ""
            if drawn is not None:
                d = f"{drawn[idx, 0] - x:+5.0f},{drawn[idx, 1] - y:+4.0f}"
            print(f"{side + ' ' + name:>24} {xi:5d},{yi:4d} {a:6.2f} {p!s:>7} {d:>12}")


if __name__ == "__main__":
    main()
