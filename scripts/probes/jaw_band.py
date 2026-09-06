"""The jaw band ("white line"), measured: mean |output - scene| in the 10 px
inside the paste mask's edge, jaw half, per frame -- the metric day 11's
night asked for before the jaw shrink was kept.

Two inputs. A **clip**: every frame is tracked and rendered as the app would
(the same tracker settings and engine options; ``--option`` overrides one for
the run, so ``--option mask_jaw_shrink=0`` is the day-11 mask), and the mask
measured is the one the engine actually pasted with (``loaded.last``). A
**capture** directory (day 12's ``seen.png`` + ``camera.png`` +
``tracking.npz``): the two pictures are compared under the driver's own oval
built from the capture's last tracked frame -- the portrait's half of the
mask is not stored, but at the jaw it is the larger of the two, so the
driver's oval is the edge that was drawn there; ``--jaw-shrink`` says which.

The band: pixels where the pasted alpha is over 0.5, within ``--band`` px of
the 0.5 contour (distance transform). "jaw half" is the band's rows below the
mask's vertical middle. Two numbers per frame, both in grey levels (0-255)
over the jaw half: ``abs`` is mean |output - scene| (the mismatch the mask
edge shows), ``step`` is mean(output) - mean(scene), signed -- a pale band
over a shadowed jaw is a positive step. p50 / p90 over frames.

Numbers are comparable only between runs of this script on the same frames.

Usage::

    uv run python scripts/probes/jaw_band.py --clip recordings/natural.mkv
    uv run python scripts/probes/jaw_band.py --clip recordings/natural.mkv \\
        --option mask_jaw_shrink=0 --stills 925,968
    uv run python scripts/probes/jaw_band.py --capture captures/moment-20260907-010203
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sha256.avatar.engine_liveportrait import (  # noqa: E402
    DEFAULT_WEIGHTS_RELATIVE_PATH,
    FACE_POINTS,
    LivePortraitEngine,
    _oval_mask,
    crop_matrix,
)
from sha256.avatar.repository import FilesystemAvatarRepository  # noqa: E402
from sha256.config.loader import load_settings  # noqa: E402
from sha256.config.paths import AppPaths  # noqa: E402
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FrameMetadata, VideoFrame  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)


def band_numbers(
    output: np.ndarray, scene: np.ndarray, alpha: np.ndarray, band_px: int
) -> dict[str, float] | None:
    """``abs`` and ``step`` over the jaw half of the band, plus the band's
    pixel count; ``None`` when there is no mask in the frame."""
    import cv2

    inside = (alpha > 0.5).astype(np.uint8)
    if not inside.any():
        return None
    distance = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
    band = (distance > 0.0) & (distance <= band_px)
    rows = np.where(inside.any(axis=1))[0]
    middle = (int(rows.min()) + int(rows.max())) // 2
    jaw = band.copy()
    jaw[:middle, :] = False
    if not jaw.any():
        return None
    grey_out = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grey_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY).astype(np.float32)
    difference = grey_out - grey_scene
    return {
        "abs": float(np.abs(difference[jaw]).mean()),
        "step": float(difference[jaw].mean()),
        "whole_abs": float(np.abs(difference[band]).mean()),
        "n": float(jaw.sum()),
    }


def face_zoom(
    scene: np.ndarray, output: np.ndarray, alpha: np.ndarray, margin: int = 40
) -> np.ndarray:
    """Camera and output side by side, cut to the mask's box, at native size."""
    ys, xs = np.nonzero(alpha > 0.05)
    height, width = scene.shape[:2]
    y0, y1 = max(int(ys.min()) - margin, 0), min(int(ys.max()) + margin, height)
    x0, x1 = max(int(xs.min()) - margin, 0), min(int(xs.max()) + margin, width)
    return np.hstack([scene[y0:y1, x0:x1], output[y0:y1, x0:x1]])


def _percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return float("nan")
    return float(ordered[min(int(q * len(ordered)), len(ordered) - 1)])


def report(name: str, values: list[float]) -> str:
    ordered = sorted(values)
    if not ordered:
        return f"  {name:<10} n/a"
    return (
        f"  {name:<10} p50 {_percentile(ordered, 0.5):7.2f}   "
        f"p90 {_percentile(ordered, 0.9):7.2f}   max {ordered[-1]:7.2f}   n={len(ordered)}"
    )


def run_clip(args: argparse.Namespace) -> int:
    import cv2

    overrides: dict[str, object] = {}
    for item in args.option:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--option expects KEY=VALUE, got {item!r}")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw

    paths = AppPaths.resolve()
    settings = load_settings(paths)
    models = paths.resolve_relative(settings.models.root_path)
    repository = FilesystemAvatarRepository(paths.resolve_relative(settings.avatars.library_path))

    tracker = MediaPipeFaceTracker(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        smoothing=settings.tracking.smoothing,
        mouth_smoothing=settings.tracking.mouth_smoothing,
        min_detection_confidence=settings.tracking.min_detection_confidence,
        min_presence_confidence=settings.tracking.min_presence_confidence,
        min_tracking_confidence=settings.tracking.min_tracking_confidence,
    )
    tracker.start()
    engine = LivePortraitEngine(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        weights=models / DEFAULT_WEIGHTS_RELATIVE_PATH,
    )
    engine.start()
    avatar = repository.get(args.avatar)
    avatar.manifest.engine_options.update(overrides)
    engine.load_avatar(avatar)
    shrink = engine._loaded[args.avatar].options.get("mask_jaw_shrink", "default")
    print(
        f"clip {args.clip}, avatar {args.avatar}, mask_jaw_shrink {shrink}, overrides {overrides}"
    )

    stills = {int(s) for s in args.stills.split(",") if s.strip()}
    capture = cv2.VideoCapture(str(args.clip))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    last = total - 1 if args.last is None else min(args.last, total - 1)
    numbers: dict[str, list[float]] = {"abs": [], "step": [], "whole_abs": []}
    index = 0
    try:
        while index <= last:
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
                state = engine._loaded[args.avatar].last
                if state is not None:
                    _, to_frame, mask = state
                    height, width = image.shape[:2]
                    alpha = cv2.warpAffine(
                        mask[:, :, 0], to_frame, (width, height), flags=cv2.INTER_LINEAR
                    )
                    values = band_numbers(out.data, image, alpha, args.band)
                    if values is not None:
                        for key in numbers:
                            numbers[key].append(values[key])
                    if index in stills:
                        still = args.out_prefix.parent / (
                            f"{args.out_prefix.name}-{args.clip.stem}-s{shrink}-f{index}.png"
                        )
                        still.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(still), face_zoom(image, out.data, alpha))
                        print(f"  still written: {still}")
            index += 1
            if index % 300 == 0:
                print(f"  ... {index} frames", flush=True)
    finally:
        capture.release()
        engine.stop()
        tracker.stop()

    print(f"\njaw band, frames {args.first}-{last}, {args.band} px inside the edge (grey levels):")
    print(report("abs", numbers["abs"]))
    print(report("step", numbers["step"]))
    print(report("whole abs", numbers["whole_abs"]))
    print("\nStills are the owner's face. Delete them when they have been looked at.")
    return 0


def run_capture(args: argparse.Namespace) -> int:
    import cv2

    seen = cv2.imread(str(args.capture / "seen.png"))
    camera = cv2.imread(str(args.capture / "camera.png"))
    if seen is None or camera is None:
        raise SystemExit(f"{args.capture} needs both seen.png and camera.png (day 12 captures)")
    data = np.load(args.capture / "tracking.npz", allow_pickle=False)
    detected = data["detected"]
    if not detected.any():
        raise SystemExit(f"{args.capture}: no face in the capture")
    last = int(np.where(detected)[0].max())
    height, width = camera.shape[:2]
    points = data["landmarks"][last][:FACE_POINTS, :2] * (width, height)
    to_crop, to_frame = crop_matrix(points.astype(np.float32))
    crop_points = (points @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
    mask = _oval_mask(crop_points, jaw_shrink=args.jaw_shrink)
    alpha = cv2.warpAffine(mask[:, :, 0], to_frame, (width, height), flags=cv2.INTER_LINEAR)
    values = band_numbers(seen, camera, alpha, args.band)
    if values is None:
        raise SystemExit(f"{args.capture}: the oval covers nothing")
    print(
        f"{args.capture.name}: frame {last} of {len(detected)}, driver oval with jaw shrink "
        f"{args.jaw_shrink}, {args.band} px inside the edge (grey levels):"
    )
    print(
        f"  abs {values['abs']:7.2f}   step {values['step']:+7.2f}   "
        f"whole abs {values['whole_abs']:7.2f}   n={int(values['n'])}"
    )
    if args.stills:
        still = args.out_prefix.parent / f"{args.out_prefix.name}-{args.capture.name}.png"
        still.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(still), face_zoom(camera, seen, alpha))
        print(f"  still written: {still}  (the owner's face; delete after)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--clip", type=Path, help="A driving clip: tracked and rendered here.")
    source.add_argument("--capture", type=Path, help="A day-12 capture directory.")
    parser.add_argument("--avatar", default="my_avatar_neural")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=None)
    parser.add_argument("--band", type=int, default=10, help="Band width inside the edge, px.")
    parser.add_argument(
        "--jaw-shrink",
        type=float,
        default=0.0,
        help="Capture mode: the driver oval's jaw shrink to measure under (0 = the join at "
        "the contour, the default; 0.08 = the bearded setting).",
    )
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Clip mode: override one engine option for this run (JSON value); repeatable.",
    )
    parser.add_argument(
        "--stills",
        default="",
        help="Clip mode: frame indices to write as camera|output face zooms. "
        "Capture mode: any value writes the capture's zoom.",
    )
    parser.add_argument("--out-prefix", type=Path, default=Path("recordings/day12-jaw"))
    args = parser.parse_args()
    return run_capture(args) if args.capture is not None else run_clip(args)


if __name__ == "__main__":
    raise SystemExit(main())
