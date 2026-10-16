"""Where the drawn face's brows, eyes, nose, mouth and chin land against
the person's own, on a "That looked wrong" capture re-rendered by this
build (day 23, the height fit).

The capture's camera.png is driven with its own last frame's tracking
(the way jaw_band.py --render does), the output read by MediaPipe, and
the rows of the two faces printed side by side -- the avatar's minus the
person's, frame pixels, positive = the avatar's lower. Options override
the engine's (``--option face_height_fit=0``); a capture older than the
height fit carries no settle for it, so give the ratio by hand
(``--option driver_face_height=1.047``, from face_height.py).

Usage::

    uv run python scripts/probes/face_rows.py --capture captures/moment-20261016-023317 \
        --avatar baldnew2 --option driver_face_height=1.047 --save out.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "probes"))

from jaw_band import capture_anchor, parse_options, start_engine, tracking_at  # noqa: E402

from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FrameMetadata, VideoFrame  # noqa: E402

ROWS = {
    "brow": (105, 334),
    "eye": (159, 145, 386, 374),
    "nose": (2,),
    "mouth": (13, 14),
    "chin": (152,),
}


def rows_of(image: np.ndarray) -> dict[str, float] | None:
    from sha256.config.loader import load_settings
    from sha256.config.paths import AppPaths
    from sha256.tracking.mediapipe_tracker import DEFAULT_MODEL_RELATIVE_PATH, MediaPipeFaceTracker

    paths = AppPaths.resolve()
    settings = load_settings(paths)
    models = paths.resolve_relative(settings.models.root_path)
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
                data=np.ascontiguousarray(image),
                pixel_format=PixelFormat.BGR24,
                metadata=FrameMetadata(frame_id=0, capture_timestamp=0.0),
            )
        )
    finally:
        reader.stop()
    if not result.face_detected or result.landmarks is None:
        return None
    pts = result.landmarks.points.copy()
    pts[:, 1] *= image.shape[0]
    return {name: float(pts[list(idx), 1].mean()) for name, idx in ROWS.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--avatar", required=True)
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--save", type=Path, default=None, help="Write the rendered frame here.")
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Learn the anchor from the capture's frame instead of seeding the session's.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    for noisy in ("sha256.avatar.hair", "sha256.tracking"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    import cv2

    camera = cv2.imread(str(args.capture / "camera.png"))
    data = np.load(args.capture / "tracking.npz", allow_pickle=False)
    detected = data["detected"]
    last = int(np.where(detected)[0].max())
    overrides = parse_options(args.option)
    engine = start_engine(args.avatar, overrides)
    try:
        anchor = None if args.no_seed else capture_anchor(data)
        if anchor is not None:
            engine.seed_anchor(args.avatar, anchor)
        for index in range(last + 1):
            frame = VideoFrame(
                data=camera,
                pixel_format=PixelFormat.BGR24,
                metadata=FrameMetadata(
                    frame_id=index, capture_timestamp=float(data["timestamps"][index])
                ),
            )
            out = engine.animate(args.avatar, tracking_at(data, last), frame)
        readings = engine.frame_readings(args.avatar) or {}
    finally:
        engine.stop()
    rendered = np.ascontiguousarray(out.data)
    if out.pixel_format is PixelFormat.RGB24:
        rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
    if args.save is not None:
        cv2.imwrite(str(args.save), rendered)
    his = rows_of(camera)
    drawn = rows_of(rendered)
    if his is None or drawn is None:
        raise SystemExit("no face read on one of the two pictures")
    print(
        f"{args.capture.name} / {args.avatar}, overrides {overrides}: "
        f"face_height_scale {readings.get('face_height_scale', float('nan')):.3f}, "
        f"face_width_scale {readings.get('face_width_scale', float('nan')):.3f}, "
        f"forehead_stretch {readings.get('forehead_stretch', float('nan')):.2f}"
    )
    print(f"{'row':>6} {'his':>7} {'avatar':>7} {'diff':>6}")
    for name in ROWS:
        print(f"{name:>6} {his[name]:7.1f} {drawn[name]:7.1f} {drawn[name] - his[name]:+6.1f}")


if __name__ == "__main__":
    main()
