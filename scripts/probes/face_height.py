"""How far the eyes (and the brows) sit above the chin, against the face
height -- the driver's against the portrait's (day 23, the height fit).

The crop is sized by the face HEIGHT (forehead top to chin, ``_face_height``),
so in crop space the two faces have the same height; what differs is how
that height is shared between the forehead and the rest. A portrait with a
taller forehead puts its eyes and brows LOWER than the person's at the same
chin (his capture 2026-10-16 02:33: the brows 20 px under his, the eyes 10,
the chin on his). The ratio printed here -- the person's eye-to-chin over
the portrait's, both as fractions of the face height -- is what the paste
has to stretch the render by, about the chin, for the eyes to land.

Usage::

    uv run python scripts/probes/face_height.py --portrait avatars/baldnew2/portrait.png \
        --capture captures/moment-20261016-023317
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sha256.avatar.engine_liveportrait import (  # noqa: E402
    _HAIRLINE_FRONTAL,
    FACE_POINTS,
    _anchors,
    _face_axes,
    _face_height,
)

CHIN = 152
BROWS = (105, 334)


def height_index(points: np.ndarray) -> tuple[float, float]:
    """(eye-to-chin / face height, brow-to-chin / face height), along the
    face's own vertical."""
    pts = np.asarray(points, dtype=np.float32)[:FACE_POINTS, :2]
    _, uy, _ = _face_axes(pts)
    eyes, _ = _anchors(pts)
    chin = pts[CHIN]
    brow = pts[list(BROWS)].mean(axis=0)
    height = _face_height(pts)
    return float((chin - eyes) @ uy) / height, float((chin - brow) @ uy) / height


def _frame_size(directory: Path) -> tuple[int, int]:
    import json

    sidecar = directory / "clip.json"
    if sidecar.exists():
        meta = json.loads(sidecar.read_text())
        return int(meta["width"]), int(meta["height"])
    import cv2

    image = cv2.imread(str(directory / "camera.png"))
    return int(image.shape[1]), int(image.shape[0])


def from_npz(directory: Path) -> tuple[float, float, int]:
    data = np.load(directory / "tracking.npz")
    w, h = _frame_size(directory)
    marks = data["landmarks"][:, :, :2] * np.array([w, h], dtype=np.float32)
    pose = data["pose"]
    detected = data["detected"].astype(bool)
    rows = []
    for i in range(len(marks)):
        if not detected[i]:
            continue
        # A capture's pose is (yaw, pitch, roll); a recording's too.
        yaw, pitch = float(pose[i][0]), float(pose[i][1])
        if abs(pitch) > _HAIRLINE_FRONTAL[0] or abs(yaw) > _HAIRLINE_FRONTAL[1]:
            continue
        rows.append(height_index(marks[i]))
    if not rows:
        return float("nan"), float("nan"), 0
    arr = np.array(rows)
    return float(np.median(arr[:, 0])), float(np.median(arr[:, 1])), len(rows)


def from_picture(path: Path) -> tuple[float, float]:
    import cv2

    from sha256.config.loader import load_settings
    from sha256.config.paths import AppPaths
    from sha256.core.enums import PixelFormat
    from sha256.core.types import FrameMetadata, VideoFrame
    from sha256.tracking.mediapipe_tracker import DEFAULT_MODEL_RELATIVE_PATH, MediaPipeFaceTracker

    paths = AppPaths.resolve()
    settings = load_settings(paths)
    models = paths.resolve_relative(settings.models.root_path)
    reader = MediaPipeFaceTracker(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        smoothing=0.0,
        mouth_smoothing=0.0,
        min_detection_confidence=settings.tracking.min_detection_confidence,
        min_presence_confidence=settings.tracking.min_presence_confidence,
        min_tracking_confidence=settings.tracking.min_tracking_confidence,
        running_mode="image",
    )
    reader.start()
    image = cv2.imread(str(path))
    h, w = image.shape[:2]
    frame = VideoFrame(
        data=np.ascontiguousarray(image),
        pixel_format=PixelFormat.BGR24,
        metadata=FrameMetadata(frame_id=0, capture_timestamp=0.0),
    )
    result = reader.process(frame)
    reader.stop()
    assert result.face_detected and result.landmarks is not None, path
    pts = result.landmarks.points.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    return height_index(pts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portrait", type=Path, action="append", default=[])
    parser.add_argument("--recording", type=Path, action="append", default=[])
    parser.add_argument("--capture", type=Path, action="append", default=[])
    parser.add_argument("--picture", type=Path, action="append", default=[])
    args = parser.parse_args()

    portraits = []
    for path in args.portrait:
        eye, brow = from_picture(path)
        portraits.append((path, eye, brow))
        print(f"portrait {path}: eye-chin {eye:.4f}  brow-chin {brow:.4f} of the face height")
    for path in args.picture:
        eye, brow = from_picture(path)
        print(f"picture  {path}: eye-chin {eye:.4f}  brow-chin {brow:.4f}")
    for directory in [*args.recording, *args.capture]:
        eye, brow, n = from_npz(directory)
        print(f"frames   {directory}: eye-chin {eye:.4f}  brow-chin {brow:.4f}  ({n} still frames)")
        for path, p_eye, p_brow in portraits:
            print(
                f"    against {path.parent.name}: eyes x{eye / p_eye:.4f}  "
                f"brows x{brow / p_brow:.4f}  (the render's stretch about the chin)"
            )


if __name__ == "__main__":
    main()
