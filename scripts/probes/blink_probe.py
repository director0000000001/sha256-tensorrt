"""The blink (day 17, his "when I shut my eyes the mouth moves a little, as
if smiling, and the brows come down"): a clip's blink and eyes-shut steps
through the engine as the app runs them, his readings and the avatar's on
every frame, and the avatar's brow and mouth on the shut frames against
the open ones -- per ``--sweep`` value, so the mechanism can be split.

What is measured on the avatar (the composite read by MediaPipe in image
mode, the same instrument the calibration uses) and on him (the app's
tracker): the eye ratio; the brow's height -- the upper brow points' mean
row above the nose tip, in frame pixels (a dip is a smaller number); the
mouth corners' row above the nose tip (a smile lifts them); the mouth
width over the jaw; the opening. ``shut`` frames are his eye ratio under
0.7 of his rest (the gaze drive's own blink hold), ``open`` over 0.9 of
it, both inside ``--first``..``--last``. The engine's gates on the shut
frames are printed too (``frame_readings``). A still of the deepest shut
frame, ``camera | output``, per run.

Usage::

    uv run python scripts/probes/blink_probe.py \\
        --clip recordings/calibration-20260912-004157/clip.mkv --first 338 --last 580 \\
        --sweep eye_gain=1.4,0
    uv run python scripts/probes/blink_probe.py ... --sweep expression_gain=1.497,0

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
from turn_probe import head_crop, labelled  # noqa: E402

from sha256.avatar.engine_liveportrait import (  # noqa: E402
    _BROWS_UPPER,
    _MOUTH_CORNERS,
    _NOSE_TIP,
    DEFAULT_WEIGHTS_RELATIVE_PATH,
    DRIVER_OPTION_KEYS,
    FACE_POINTS,
    LivePortraitEngine,
    _eye_ratios,
    _mouth_open_ratio,
    _mouth_width_ratio,
)
from sha256.avatar.repository import FilesystemAvatarRepository  # noqa: E402
from sha256.calibration.profile import PROFILE_FILENAME, DriverProfile  # noqa: E402
from sha256.config.loader import load_settings  # noqa: E402
from sha256.config.paths import AppPaths  # noqa: E402
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FaceTrackingResult, FrameMetadata, VideoFrame  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)

_SHUT = 0.7
_OPEN = 0.9


def face_measures(points: np.ndarray) -> dict[str, float]:
    """The measures of one face from its 468+ points in frame pixels."""
    pts = points[:FACE_POINTS]
    nose_y = float(pts[_NOSE_TIP, 1])
    return {
        "ear": sum(_eye_ratios(pts)) / 2.0,
        "brow_up": nose_y - float(pts[list(_BROWS_UPPER), 1].mean()),
        "corner_up": nose_y - float(pts[list(_MOUTH_CORNERS), 1].mean()),
        "width": _mouth_width_ratio(pts),
        "opening": _mouth_open_ratio(pts),
    }


def read_composite(reader: MediaPipeFaceTracker, image: np.ndarray) -> dict[str, float] | None:
    got: FaceTrackingResult = reader.process(
        VideoFrame(
            data=np.ascontiguousarray(image),
            pixel_format=PixelFormat.BGR24,
            metadata=FrameMetadata(frame_id=0, capture_timestamp=0.0),
        )
    )
    if not got.face_detected or got.landmarks is None:
        return None
    h, w = image.shape[:2]
    return face_measures(got.landmarks.points[:, :2] * (w, h))


def summarise(label: str, rows: list[dict[str, Any]]) -> None:
    rest = [r for r in rows if r["his"] is not None]
    if not rest:
        print(f"  {label}: no frames")
        return
    rest_ear = float(np.median([r["his"]["ear"] for r in rest]))
    open_ = [r for r in rest if r["his"]["ear"] >= _OPEN * rest_ear and r["avatar"]]
    shut = [r for r in rest if r["his"]["ear"] < _SHUT * rest_ear and r["avatar"]]
    if not open_ or not shut:
        print(f"  {label}: open {len(open_)} shut {len(shut)} -- not enough of one")
        return

    def med(frames: list[dict[str, Any]], who: str, key: str) -> float:
        return float(np.median([f[who][key] for f in frames]))

    parts = []
    for key, unit in (("brow_up", "px"), ("corner_up", "px"), ("width", ""), ("opening", "")):
        d_av = med(shut, "avatar", key) - med(open_, "avatar", key)
        d_his = med(shut, "his", key) - med(open_, "his", key)
        parts.append(f"{key} avatar {d_av:+.2f}{unit} (his {d_his:+.2f}{unit})")
    # By how deep the lids are: the blink's own fast phases live between
    # open and shut, and a counter that is linear in depth may over- or
    # under-shoot there while the medians above read fine.
    bins = ((0.9, 1.5), (0.7, 0.9), (0.5, 0.7), (0.3, 0.5), (0.0, 0.3))
    base_brow = med(open_, "avatar", "brow_up")
    base_corner = med(open_, "avatar", "corner_up")
    per_bin = []
    for lo, hi in bins:
        frames = [r for r in rest if lo <= r["his"]["ear"] / rest_ear < hi and r["avatar"]]
        if not frames:
            continue
        brows = med(frames, "avatar", "brow_up") - base_brow
        corners = med(frames, "avatar", "corner_up") - base_corner
        his_brows = med(frames, "his", "brow_up") - med(open_, "his", "brow_up")
        his_corners = med(frames, "his", "corner_up") - med(open_, "his", "corner_up")
        per_bin.append(
            f"[{lo:.1f},{hi:.1f}) n{len(frames)} brows {brows:+.1f} (his {his_brows:+.1f})"
            f" corners {corners:+.1f} (his {his_corners:+.1f})"
        )
    print("    by depth (his ratio / rest): " + "; ".join(per_bin))
    gates = [r["engine"] for r in shut if r["engine"]]
    gate = ""
    if gates:
        gate = "; gates on shut: " + ", ".join(
            f"{k} p50 {np.median([g[k] for g in gates]):.2f} max {max(g[k] for g in gates):.2f}"
            for k in ("gate_mouth", "gate_open", "gate_brow")
        )
    print(
        f"  {label}: open {len(open_)} shut {len(shut)} frames; shut - open: "
        + ", ".join(parts)
        + f"; avatar ear shut {med(shut, 'avatar', 'ear'):.3f}"
        + f" open {med(open_, 'avatar', 'ear'):.3f}"
        + gate
    )


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--avatar", default="my_avatar_neural")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--sweep", default=None, metavar="KEY=v1,v2,...")
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--out-prefix", type=Path, default=Path("recordings/day17/blink"))
    args = parser.parse_args()

    overrides = parse_options(args.option)
    sweep_key, sweep_values = parse_sweep(args.sweep)
    paths = AppPaths.resolve()
    settings = load_settings(paths)
    models = paths.resolve_relative(settings.models.root_path)
    repo = FilesystemAvatarRepository(paths.resolve_relative(settings.avatars.library_path))
    profile = None if args.no_profile else DriverProfile.load(paths.configs / PROFILE_FILENAME)
    overlay = profile.overlay() if profile else {}
    engine = LivePortraitEngine(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        weights=models / DEFAULT_WEIGHTS_RELATIVE_PATH,
    )
    engine.start()
    avatar = repo.get(args.avatar)
    base = copy.deepcopy(avatar.manifest.engine_options)
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
    print(f"{args.clip}: avatar {args.avatar}, profile {'none' if not profile else 'yes'}")
    try:
        for value in sweep_values:
            options: dict[str, Any] = dict(base)
            options.update(overrides)
            if sweep_key is not None:
                options[sweep_key] = value
            # The profile lies over the avatar's options at load, as live; a
            # swept or overridden driver key wins over the profile's own,
            # the avatar's own driver keys do not (they never do live).
            driver = dict(overlay)
            driver.update({k: v for k, v in overrides.items() if k in DRIVER_OPTION_KEYS})
            if sweep_key in DRIVER_OPTION_KEYS:
                driver[sweep_key] = value
            engine.set_driver_options(driver)
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
            rows: list[dict[str, Any]] = []
            deepest: tuple[float, np.ndarray] | None = None
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
                        his = None
                        points = None
                        if tracking.face_detected and tracking.landmarks is not None:
                            points = tracking.landmarks.points[:, :2] * (
                                image.shape[1],
                                image.shape[0],
                            )
                            his = face_measures(points)
                        composite = np.ascontiguousarray(out.data)
                        readings = engine.frame_readings(args.avatar) or {}
                        rows.append(
                            {
                                "frame": index,
                                "his": his,
                                "avatar": read_composite(reader, composite),
                                "engine": {
                                    k: float(readings.get(k, float("nan")))
                                    for k in ("gate_mouth", "gate_open", "gate_brow")
                                },
                            }
                        )
                        if his is not None and (deepest is None or his["ear"] < deepest[0]):
                            deepest = (
                                his["ear"],
                                labelled(
                                    head_crop(image, points),
                                    head_crop(composite, points),
                                    f"f{index} ear {his['ear']:.2f}",
                                ),
                            )
                    index += 1
            finally:
                capture.release()
                tracker.stop()
            label = "base build" if sweep_key is None else f"{sweep_key}={value}"
            summarise(label, rows)
            if deepest is not None:
                tag = "" if sweep_key is None else f"-{sweep_key}{value}"
                prefix = args.out_prefix.parent / f"{args.out_prefix.name}-{args.clip.stem}{tag}"
                prefix.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f"{prefix}-shut.png", deepest[1])
    finally:
        engine.stop()
        reader.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
