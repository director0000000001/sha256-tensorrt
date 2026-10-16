"""How wide a face is against its height -- the driver's against the
portrait's (day 22, the width fit).

The crop is sized by the face HEIGHT (forehead top to chin, ``_face_height``),
so in crop space the two faces have the same height and differ only in
width: the ratio printed here is what the paste would have to squeeze (or
widen) the render by for the avatar's rim to land on the person's.

Width = the face rim's extent along the face's own horizontal (``_face_axes``),
read on frontal, still frames only (the hairline reader's own bounds).

Usage::

    uv run python scripts/probes/face_width.py --portrait avatars/my_avatar_neural/portrait.png \
        --recording recordings/calibration-20260912-194044 --capture captures/moment-20260912-200754
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sha256.avatar.engine_liveportrait import (  # noqa: E402
    _FACE_OVAL,
    _HAIRLINE_FRONTAL,
    FACE_POINTS,
    _face_axes,
    _face_height,
)

RIM = list(_FACE_OVAL)
SIDES = (234, 454)


def width_index(points: np.ndarray) -> tuple[float, float]:
    """(rim width / face height, cheek-side width / face height)."""
    pts = np.asarray(points, dtype=np.float32)[:FACE_POINTS, :2]
    ux, _, _ = _face_axes(pts)
    along = pts[RIM] @ ux
    height = _face_height(pts)
    sides = float(abs((pts[SIDES[1]] - pts[SIDES[0]]) @ ux))
    return float(along.max() - along.min()) / height, sides / height


def _frame_size(directory: Path) -> tuple[int, int]:
    """(width, height) of the frames the landmarks were normalised by."""
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
        pitch, yaw = float(pose[i][0]), float(pose[i][1])
        if abs(pitch) > _HAIRLINE_FRONTAL[0] or abs(yaw) > _HAIRLINE_FRONTAL[1]:
            continue
        rows.append(width_index(marks[i]))
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
    return width_index(pts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portrait", type=Path, action="append", default=[])
    parser.add_argument("--recording", type=Path, action="append", default=[])
    parser.add_argument("--capture", type=Path, action="append", default=[])
    parser.add_argument("--picture", type=Path, action="append", default=[])
    args = parser.parse_args()
    portraits = []
    for p in args.portrait:
        rim, sides = from_picture(p)
        portraits.append((p, rim, sides))
        print(f"portrait {p}: rim {rim:.4f}  sides {sides:.4f}")
    for p in args.picture:
        rim, sides = from_picture(p)
        print(f"picture  {p}: rim {rim:.4f}  sides {sides:.4f}")
        for pp, prim, psides in portraits:
            print(f"    vs {pp.parent.name}: rim x{rim / prim:.3f}  sides x{sides / psides:.3f}")
    for d in args.recording + args.capture:
        rim, sides, n = from_npz(d)
        print(f"frames   {d}: rim {rim:.4f}  sides {sides:.4f}  ({n} frontal frames)")
        for pp, prim, psides in portraits:
            print(f"    vs {pp.parent.name}: rim x{rim / prim:.3f}  sides x{sides / psides:.3f}")


if __name__ == "__main__":
    main()
