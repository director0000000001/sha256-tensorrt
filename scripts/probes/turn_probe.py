"""The steep poses (day 17, his "looking sideways, head down"): a clip's
turn or tilt segment through the engine as the app runs it (the filtered
tracker, the driver profile laid over the avatar), with stills at the
angles that matter and a side-by-side video of the segment.

His "That looked wrong" captures carry a picture of the press's own frame
only, and he presses once he is back at the camera -- so the turn itself
is read off his recorded clips: the calibration session's turn step, the
tilt step of natural.mkv, driving.mkv's turns. The engine starts at the
clip's first frame (the anchor learns from the still start) and the
stills are taken inside ``--first``..``--last``: the first frame past
each of ``--yaw-marks`` and ``--pitch-marks`` in each direction, the
farthest turn each way and the deepest tilt, each as ``camera | output``
head crops with the pose and whether the face was held written on it,
stacked into one sheet per run. ``--option`` / ``--sweep`` as in
capture_replay.py; ``--video`` also writes the segment as an mp4.

Usage::

    uv run python scripts/probes/turn_probe.py \\
        --clip recordings/calibration-20260912-004157/clip.mkv --first 735 --last 831 \\
        --sweep steep_pose=hold,clamp
    uv run python scripts/probes/turn_probe.py --clip recordings/natural.mkv \\
        --first 417 --last 712 --pitch-marks 15,20,25,30 \\
        --sweep hold_pitch_degrees=20,28 --option steep_pose=clamp

Everything written is his face; delete after looking.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "probes"))

from capture_replay import parse_options, parse_sweep  # noqa: E402

from sha256.avatar.engine_liveportrait import (  # noqa: E402
    DEFAULT_WEIGHTS_RELATIVE_PATH,
    FACE_POINTS,
    LivePortraitEngine,
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

_CROP_SCALE = 1.9  # of the face width, the head crop's side


def head_crop(image: np.ndarray, points: np.ndarray | None, size: int = 420) -> np.ndarray:
    """A square around the face (or the picture's centre), resized to ``size``."""
    import cv2

    height, width = image.shape[:2]
    if points is None:
        cx, cy, side = width / 2.0, height / 2.0, height
    else:
        pts = points[:FACE_POINTS]
        cx = float((pts[:, 0].min() + pts[:, 0].max()) / 2.0)
        cy = float((pts[:, 1].min() + pts[:, 1].max()) / 2.0) - 0.1 * float(
            pts[:, 1].max() - pts[:, 1].min()
        )
        side = _CROP_SCALE * max(float(pts[:, 0].max() - pts[:, 0].min()), 1.0)
    half = int(side / 2.0)
    x0, y0 = int(cx) - half, int(cy) - half
    x1, y1 = x0 + 2 * half, y0 + 2 * half
    padded = cv2.copyMakeBorder(
        image,
        max(-y0, 0),
        max(y1 - height, 0),
        max(-x0, 0),
        max(x1 - width, 0),
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    ox, oy = max(-x0, 0), max(-y0, 0)
    crop = padded[y0 + oy : y1 + oy, x0 + ox : x1 + ox]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def labelled(camera: np.ndarray, output: np.ndarray, text: str) -> np.ndarray:
    import cv2

    row = np.hstack([camera, output])
    cv2.putText(row, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return row


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True, help="The segment's first frame.")
    parser.add_argument("--last", type=int, required=True, help="The segment's last frame.")
    parser.add_argument("--avatar", default="my_avatar_neural")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--sweep", default=None, metavar="KEY=v1,v2,...")
    parser.add_argument("--no-profile", action="store_true", help="Ignore the machine's profile.")
    parser.add_argument("--yaw-marks", default="20,30,40,50,60")
    parser.add_argument("--pitch-marks", default="15,20,25,30")
    parser.add_argument("--video", action="store_true", help="Also write the segment as an mp4.")
    parser.add_argument("--out-prefix", type=Path, default=Path("recordings/day17/turn"))
    args = parser.parse_args()

    yaw_marks = [float(v) for v in args.yaw_marks.split(",") if v.strip()]
    pitch_marks = [float(v) for v in args.pitch_marks.split(",") if v.strip()]
    overrides = parse_options(args.option)
    sweep_key, sweep_values = parse_sweep(args.sweep)

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

    try:
        for value in sweep_values:
            options: dict[str, Any] = dict(base)
            options.update(overrides)
            if sweep_key is not None:
                options[sweep_key] = value
            tag = "" if sweep_key is None else f"-{sweep_key}{value}"
            avatar.manifest.engine_options.clear()
            avatar.manifest.engine_options.update(options)
            engine.unload_avatar(args.avatar)
            engine.load_avatar(avatar)
            tracker = MediaPipeFaceTracker(
                model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
                smoothing=settings.tracking.smoothing,
                mouth_smoothing=settings.tracking.mouth_smoothing,
                min_detection_confidence=settings.tracking.min_detection_confidence,
                min_presence_confidence=settings.tracking.min_presence_confidence,
                min_tracking_confidence=settings.tracking.min_tracking_confidence,
            )
            tracker.start()
            capture = cv2.VideoCapture(str(args.clip))
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
            writer = None
            prefix = args.out_prefix.parent / f"{args.out_prefix.name}-{args.clip.stem}{tag}"
            prefix.parent.mkdir(parents=True, exist_ok=True)
            stills: dict[str, tuple[float, np.ndarray]] = {}  # key -> (score, row)
            crossed: set[str] = set()
            held_frames = 0
            lost_frames = 0
            index = 0
            try:
                while index <= args.last:
                    ok, image = capture.read()
                    if not ok:
                        break
                    frame = VideoFrame(
                        data=np.ascontiguousarray(image),
                        pixel_format=PixelFormat.BGR24,
                        metadata=FrameMetadata(frame_id=index, capture_timestamp=index / fps),
                    )
                    tracking = tracker.process(frame)
                    out = engine.animate(args.avatar, tracking, frame)
                    if index >= args.first:
                        loaded = engine._loaded[args.avatar]
                        pose = tracking.head_pose
                        yaw = 0.0 if pose is None else float(pose.yaw)
                        pitch = 0.0 if pose is None else float(pose.pitch)
                        rel_yaw = yaw - (loaded.rest_yaw or 0.0)
                        rel_pitch = pitch - (loaded.rest_pitch or 0.0)
                        state = (
                            "LOST"
                            if not tracking.face_detected
                            else ("held" if loaded.holding else "drawn")
                        )
                        held_frames += loaded.holding
                        lost_frames += not tracking.face_detected
                        points = None
                        if tracking.landmarks is not None:
                            points = tracking.landmarks.points[:, :2] * (
                                image.shape[1],
                                image.shape[0],
                            )
                        text = f"f{index} yaw {rel_yaw:+.0f} pitch {rel_pitch:+.0f} {state}"
                        # Which stills this frame is: a mark's first crossing,
                        # or a new farthest turn / deepest tilt.
                        wanted: list[tuple[str, float]] = []
                        for mark in yaw_marks:
                            for sign, name in ((1.0, "right"), (-1.0, "left")):
                                key = f"yaw {name} {mark:.0f}"
                                if key not in crossed and sign * rel_yaw >= mark:
                                    crossed.add(key)
                                    wanted.append((key, 0.0))
                        for mark in pitch_marks:
                            key = f"pitch down {mark:.0f}"
                            if key not in crossed and -rel_pitch >= mark:
                                crossed.add(key)
                                wanted.append((key, 0.0))
                        for key, score in (
                            ("yaw max right", rel_yaw),
                            ("yaw max left", -rel_yaw),
                            ("pitch max down", -rel_pitch),
                        ):
                            best = stills.get(key)
                            if best is None or score > best[0]:
                                wanted.append((key, score))
                        if wanted:
                            row = labelled(
                                head_crop(image, points),
                                head_crop(np.ascontiguousarray(out.data), points),
                                text,
                            )
                            for key, score in wanted:
                                stills[key] = (score, row)
                        if args.video:
                            composite = np.ascontiguousarray(out.data)
                            small = np.hstack(
                                [
                                    cv2.resize(image, None, fx=0.5, fy=0.5),
                                    cv2.resize(composite, None, fx=0.5, fy=0.5),
                                ]
                            )
                            if writer is None:
                                writer = cv2.VideoWriter(
                                    f"{prefix}.mp4",
                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                    fps,
                                    (small.shape[1], small.shape[0]),
                                )
                            writer.write(small)
                    index += 1
            finally:
                capture.release()
                if writer is not None:
                    writer.release()
                tracker.stop()
            order = [k for k in stills if k.startswith("yaw right")] + ["yaw max right"]
            order += [k for k in stills if k.startswith("yaw left")] + ["yaw max left"]
            order += [k for k in stills if k.startswith("pitch down")] + ["pitch max down"]
            rows = [stills[k][1] for k in order if k in stills]
            label = "base build" if sweep_key is None else f"{sweep_key}={value}"
            print(
                f"  {label}: frames {args.first}-{args.last}, held {held_frames}, "
                f"lost {lost_frames}; stills {len(rows)} -> {prefix}-sheet.png"
                + (f", video {prefix}.mp4" if args.video else "")
            )
            if rows:
                cv2.imwrite(f"{prefix}-sheet.png", np.vstack(rows))
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
