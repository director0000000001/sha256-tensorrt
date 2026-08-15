"""Replay a moment the owner marked as wrong.

The companion to the "That looked wrong" button, and the reason it exists.

Every eye fix so far has been verified against one recorded clip, and the owner
keeps seeing something that clip does not contain — a different room, a
different distance, a moment that does not happen on cue. A capture is those
exact seconds: the tracking that drove the engine, frame by frame, and the one
picture that was on screen when the button was pressed.

There is no video in a capture, and none is needed. Landmarks, blendshapes and
head pose fully determine what the engine did with them, so the avatar can be
re-rendered from the readings alone — over a plain background, because what is
being examined is the avatar and not the room.

Usage::

    python scripts/replay_capture.py                     # the newest capture
    python scripts/replay_capture.py captures/moment-...
    python scripts/replay_capture.py --region eyes --scale 3
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
from sha256.core.types import (  # noqa: E402
    BlendshapeSet,
    FaceTrackingResult,
    FrameMetadata,
    HeadPose,
    LandmarkSet,
    VideoFrame,
)
from sha256.tracking.mediapipe_tracker import DEFAULT_MODEL_RELATIVE_PATH  # noqa: E402

# Which landmarks bound each crop, and how much room to leave around them.
_CROPS: dict[str, tuple[list[int], float]] = {
    "face": ([10, 152, 234, 454], 0.15),
    "eyes": ([33, 263, 133, 362, 159, 145, 386, 374, 105, 334], 0.55),
    "mouth": ([61, 291, 0, 17, 13, 14], 0.65),
}

# The frame size the readings were captured at. Landmarks are normalised, so
# this only sets how large the avatar is rendered — matching the camera keeps
# the face the size the engine actually saw.
_WIDTH, _HEIGHT = 1280, 720


def newest_capture(root: Path) -> Path | None:
    moments = sorted(root.glob("moment-*"), reverse=True)
    return moments[0] if moments else None


def tracking_at(data: dict[str, np.ndarray], index: int) -> FaceTrackingResult:
    """One frame's readings, rebuilt as the engine's own type."""
    names = tuple(str(n) for n in data["names"])
    yaw, pitch, roll = (float(v) for v in data["pose"][index])
    return FaceTrackingResult(
        metadata=FrameMetadata(frame_id=index, capture_timestamp=float(data["timestamps"][index])),
        face_detected=bool(data["detected"][index]),
        landmarks=LandmarkSet(
            points=data["landmarks"][index].astype(np.float32), scheme="mediapipe_478"
        ),
        head_pose=HeadPose(yaw=yaw, pitch=pitch, roll=roll),
        blendshapes=BlendshapeSet(
            names=names, weights=data["weights"][index].astype(np.float32), scheme="mediapipe_52"
        ),
        confidence=1.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", nargs="?", type=Path, help="A captures/moment-* directory.")
    parser.add_argument("--avatar", default=None)
    parser.add_argument("--region", default="face", choices=tuple(_CROPS))
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--every", type=int, default=15, help="Render one frame in this many.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import cv2

    paths = AppPaths.resolve()
    settings = load_settings(paths)

    capture = args.capture or newest_capture(paths.captures)
    if capture is None or not (capture / "tracking.npz").is_file():
        print(
            f"No capture found. Press 'That looked wrong' in the app; they land in {paths.captures}"
        )
        return 2

    data = dict(np.load(capture / "tracking.npz", allow_pickle=False))
    count = len(data["timestamps"])
    print(f"{capture.name}: {count} frames of tracking")

    models = paths.resolve_relative(settings.models.root_path)
    engine = WarpAvatarEngine(model_path=models / DEFAULT_MODEL_RELATIVE_PATH)
    engine.start()
    repository = FilesystemAvatarRepository(paths.resolve_relative(settings.avatars.library_path))
    avatar_id = args.avatar or settings.avatars.default_avatar_id or "my_avatar"
    engine.load_avatar(repository.get(avatar_id))

    # A plain background rather than the room: the room was not captured, and
    # the question is what the avatar looks like anyway.
    backdrop = np.full((_HEIGHT, _WIDTH, 3), 60, dtype=np.uint8)

    rows: list[np.ndarray] = []
    for index in range(count):
        result = tracking_at(data, index)
        frame = VideoFrame(
            data=backdrop.copy(),
            pixel_format=PixelFormat.BGR24,
            metadata=result.metadata,
        )
        # Every frame is fed through, because the engine learns the resting face
        # and skipping frames would hand it a history it never had.
        output = engine.animate(avatar_id, result, frame)
        if index % args.every or not result.face_detected:
            continue

        points = result.landmarks.points[:, :2].copy() * (_WIDTH, _HEIGHT)
        indices, margin = _CROPS[args.region]
        block = points[indices]
        x0, x1 = float(block[:, 0].min()), float(block[:, 0].max())
        y0, y1 = float(block[:, 1].min()), float(block[:, 1].max())
        pad_x, pad_y = (x1 - x0) * margin, (y1 - y0) * margin
        crop = output.data[
            max(int(y0 - pad_y), 0) : min(int(y1 + pad_y), _HEIGHT),
            max(int(x0 - pad_x), 0) : min(int(x1 + pad_x), _WIDTH),
        ]
        if crop.size == 0:
            continue
        crop = cv2.resize(
            crop,
            (crop.shape[1] * args.scale, crop.shape[0] * args.scale),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.putText(crop, f"frame {index}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        rows.append(crop)

    if not rows:
        print("Nothing rendered — no frame in the capture had a face in it.")
        return 1

    widest = max(row.shape[1] for row in rows)
    sheet = np.vstack(
        [
            cv2.copyMakeBorder(
                row, 0, 4, 0, widest - row.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            for row in rows
        ]
    )
    out = args.out or capture / f"replay-{args.region}.png"
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out}  ({len(rows)} frame(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
