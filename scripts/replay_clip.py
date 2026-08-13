"""Replay a recorded clip through the engine, and render the moments that matter.

The companion to ``record_driving_clip.py``, and the reason it exists. Quality
work on this engine has cost a round trip per idea: change something, ask the
owner to run the application, wait for them to describe what they saw. This
replaces that with a command.

Two things it does that a live run cannot:

*   **The input is identical every time.** The same face, the same light, the
    same movements, so a difference between two runs is the engine and nothing
    else. Live, the room and the person are variables.
*   **It finds its own moments.** The prompts during recording were a guide and
    the owner could not keep up with them, so the behaviour is not where the
    labels say. It does not need to be: the tracker's own signals say where the
    widest smile and the deepest blink are, and those frames are what a contact
    sheet should show.

Output is a sheet of before/after-able crops, one row per moment, plus optional
full frames. Run it, look at it, change something, run it again.

Usage::

    python scripts/replay_clip.py
    python scripts/replay_clip.py --out reports/after-iris-fix.png
    python scripts/replay_clip.py --region eyes --scale 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sha256.avatar.engine_warp import WarpAvatarEngine  # noqa: E402
from sha256.avatar.repository import FilesystemAvatarRepository  # noqa: E402
from sha256.config.loader import load_settings  # noqa: E402
from sha256.config.paths import AppPaths  # noqa: E402
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FrameMetadata, VideoFrame  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)

# What to look for, and how to score a frame for it. Each becomes one row.
MOMENTS: tuple[tuple[str, str, str], ...] = (
    ("resting", "still", "min"),
    ("looking left", "gaze_x", "min"),
    ("looking right", "gaze_x", "max"),
    ("blinking", "blink", "max"),
    ("smiling", "smile", "max"),
    ("mouth open", "jaw", "max"),
    ("brows raised", "brow", "max"),
    ("head turned", "yaw", "max"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default=str(REPO_ROOT / "recordings" / "driving.mkv"))
    parser.add_argument("--out", default=str(REPO_ROOT / "recordings" / "replay.png"))
    parser.add_argument("--avatar", default=None, help="Avatar id; defaults to the configured one.")
    parser.add_argument(
        "--region",
        default="face",
        choices=("face", "eyes", "mouth"),
        help="Which part to crop into the sheet.",
    )
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()

    import cv2

    clip = Path(args.clip)
    if not clip.exists():
        print(f"No clip at {clip}. Record one with scripts/record_driving_clip.py")
        return 2

    paths = AppPaths.resolve()
    settings = load_settings(paths)
    models = paths.resolve_relative(settings.models.root_path)

    tracker = MediaPipeFaceTracker(model_path=models / DEFAULT_MODEL_RELATIVE_PATH)
    tracker.start()

    engine = WarpAvatarEngine(model_path=models / DEFAULT_MODEL_RELATIVE_PATH)
    engine.start()
    repository = FilesystemAvatarRepository(paths.resolve_relative(settings.avatars.library_path))
    avatar_id = args.avatar or settings.avatars.default_avatar_id or "my_avatar"
    engine.load_avatar(repository.get(avatar_id))

    # Pass one: track everything and score every frame. Cheap enough (a few
    # milliseconds each) that scoring twice would be the wrong optimisation.
    print(f"scanning {clip.name} ...")
    capture = cv2.VideoCapture(str(clip))
    scores: list[dict[str, float]] = []
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        result = tracker.process(
            VideoFrame(
                data=np.ascontiguousarray(image),
                pixel_format=PixelFormat.BGR24,
                metadata=FrameMetadata(frame_id=index, capture_timestamp=index / 30.0),
            )
        )
        index += 1
        if not result.face_detected or result.landmarks is None or result.blendshapes is None:
            scores.append({})
            continue

        shapes = result.blendshapes
        points = result.landmarks.points
        socket = points[[33, 133, 160, 158, 153, 144], :2]
        iris = points[468:473, :2].mean(axis=0)
        width = float(socket[:, 0].max() - socket[:, 0].min()) or 1.0
        gaze = (iris - socket.mean(axis=0)) / width
        blink = min(shapes.get("eyeBlinkLeft"), shapes.get("eyeBlinkRight"))
        smile = max(shapes.get("mouthSmileLeft"), shapes.get("mouthSmileRight"))
        jaw = float(shapes.get("jawOpen"))
        brow = float(shapes.get("browInnerUp"))
        yaw = float(result.head_pose.yaw) if result.head_pose else 0.0

        scores.append(
            {
                "gaze_x": float(gaze[0]),
                "blink": float(blink),
                "smile": float(smile),
                "jaw": jaw,
                "brow": brow,
                "yaw": yaw,
                # "Nothing happening": the frame that best represents a face at
                # rest, which is the baseline every other row is judged against.
                "still": float(blink + smile + jaw + brow + abs(gaze[0]) * 4 + abs(yaw) / 20.0),
            }
        )
    capture.release()

    chosen: list[tuple[str, int]] = []
    for label, key, how in MOMENTS:
        candidates = [(s[key], i) for i, s in enumerate(scores) if key in s]
        if not candidates:
            continue
        value, frame = max(candidates) if how == "max" else min(candidates)
        chosen.append((f"{label}  ({key} {value:+.2f}, frame {frame})", frame))

    # Pass two: render only the frames chosen. The engine is stateful — it
    # learns the resting face — so it is fed from the beginning of the clip
    # rather than seeked to, exactly as it would run live.
    wanted = {frame for _, frame in chosen}
    rendered: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(clip))
    index = 0
    print(f"rendering {len(wanted)} moment(s) ...")
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frame = VideoFrame(
            data=np.ascontiguousarray(image),
            pixel_format=PixelFormat.BGR24,
            metadata=FrameMetadata(frame_id=index, capture_timestamp=index / 30.0),
        )
        result = tracker.process(frame)
        output = engine.animate(avatar_id, result, frame)
        if index in wanted and result.landmarks is not None:
            # The live landmarks travel with the frame, so a crop can follow the
            # face instead of a fixed rectangle. Rows then line up between runs
            # even though the head has moved.
            points = result.landmarks.points[:, :2].copy()
            points[:, 0] *= image.shape[1]
            points[:, 1] *= image.shape[0]
            rendered[index] = (output.data.copy(), points)
        index += 1
    capture.release()

    rows = []
    for label, frame in chosen:
        found = rendered.get(frame)
        if found is None:
            continue
        image, points = found
        crop = _crop(image, points, args.region)
        crop = cv2.resize(
            crop,
            (crop.shape[1] * args.scale, crop.shape[0] * args.scale),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.putText(crop, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        rows.append(crop)

    if not rows:
        print("Nothing rendered.")
        return 1

    widest = max(row.shape[1] for row in rows)
    padded = [
        cv2.copyMakeBorder(
            row, 0, 4, 0, widest - row.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        for row in rows
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(padded))
    print(f"wrote {out}")
    return 0


# Landmarks bounding each crop, and how much margin to leave around them.
_CROPS: dict[str, tuple[list[int], float]] = {
    "face": ([10, 152, 234, 454], 0.15),
    "eyes": ([33, 263, 133, 362, 159, 145, 386, 374, 105, 334], 0.55),
    "mouth": ([61, 291, 0, 17, 13, 14], 0.65),
}


def _crop(image: np.ndarray, points: np.ndarray, region: str) -> np.ndarray:
    """The part of the rendered frame worth looking at, following the face.

    Bounded by the live landmarks rather than by a fixed rectangle, so the same
    row frames the same feature between runs even though the head has moved.
    """
    height, width = image.shape[:2]
    indices, margin = _CROPS[region]
    if points.shape[0] <= max(indices):
        return image

    block = points[indices]
    x0, x1 = float(block[:, 0].min()), float(block[:, 0].max())
    y0, y1 = float(block[:, 1].min()), float(block[:, 1].max())
    pad_x, pad_y = (x1 - x0) * margin, (y1 - y0) * margin

    return image[
        max(int(y0 - pad_y), 0) : min(int(y1 + pad_y), height),
        max(int(x0 - pad_x), 0) : min(int(x1 + pad_x), width),
    ]


if __name__ == "__main__":
    raise SystemExit(main())
