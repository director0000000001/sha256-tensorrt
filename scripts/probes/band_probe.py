"""The pasted background (day 19, his two screenshots): how much of the
paste mask, frame by frame, lands on the render's BACKGROUND rather than
on the drawn face -- the pale band along the far cheek on a held turn, the
white streak on a fast shake.

The mask is the portrait's oval and the driver's, and the network draws
the face where the pose puts it inside the crop; where the drawn face
falls short of the mask (a turn narrows it on the far side, a fast move
lags it), the mask covers the photograph's own flat background and pastes
it. This probe runs a clip through the app's own tracker and engine, hooks
the paste, and on the chosen frames counts the mask's pixels (alpha over
``--alpha``) whose render pixel is within ``--tolerance`` of the
photograph's background colour (the median of the portrait crop's four
corners), prints them with the head's yaw, and writes a zoom of the mask's
box: the render with those pixels tinted red | the paste as it goes out.

Frames: ``--frames`` by number, and/or ``--fastest N`` for the N frames of
the segment with the largest yaw change from the frame before.

Usage::

    uv run python scripts/probes/band_probe.py --clip recordings/driving-turn.mkv \\
        --first 770 --last 889 --fastest 3 --frames 298,452,459,550,557,1042

Everything written is his face; delete after looking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sha256.avatar.engine_liveportrait import (  # noqa: E402
    DEFAULT_WEIGHTS_RELATIVE_PATH,
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


def background_colour(crop: np.ndarray, corner: int = 32) -> np.ndarray:
    """The photograph's flat background: the median of the crop's corners."""
    h, w = crop.shape[:2]
    patches = [
        crop[:corner, :corner],
        crop[:corner, w - corner :],
        crop[h - corner :, :corner],
        crop[h - corner :, w - corner :],
    ]
    return np.median(np.concatenate([p.reshape(-1, 3) for p in patches]), axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--avatar", default="my_avatar_neural")
    parser.add_argument("--frames", default="")
    parser.add_argument("--fastest", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=25.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--out-prefix", type=Path, default=Path("recordings/day19/band"))
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Also write each chosen frame's whole output as it would go out (PNG, "
        "lossless) -- for a byte comparison between two runs.",
    )
    args = parser.parse_args()

    import cv2

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "probes"))
    from capture_replay import parse_options

    paths = AppPaths.resolve()
    settings = load_settings(paths)
    models = paths.resolve_relative(settings.models.root_path)
    repo = FilesystemAvatarRepository(paths.resolve_relative(settings.avatars.library_path))
    profile = None if args.no_profile else DriverProfile.load(paths.configs / PROFILE_FILENAME)
    engine = LivePortraitEngine(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        weights=models / DEFAULT_WEIGHTS_RELATIVE_PATH,
        driver_options=profile.overlay() if profile else {},
    )
    engine.start()
    avatar = repo.get(args.avatar)
    avatar.manifest.engine_options.update(parse_options(args.option))
    engine.load_avatar(avatar)
    loaded = engine._loaded[args.avatar]
    tracker = MediaPipeFaceTracker(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        smoothing=settings.tracking.smoothing,
        mouth_smoothing=settings.tracking.mouth_smoothing,
        min_detection_confidence=settings.tracking.min_detection_confidence,
        min_presence_confidence=settings.tracking.min_presence_confidence,
        min_tracking_confidence=settings.tracking.min_tracking_confidence,
        redetect_in_crop=settings.tracking.redetect_in_crop,
    )
    tracker.start()

    # The paste, hooked: the render, the mask and the affine of every frame.
    captured: dict[str, Any] = {}
    original_paste = engine._paste

    def hooked(
        scene: np.ndarray,
        face: np.ndarray,
        to_frame: np.ndarray,
        port: Any,
        mask: np.ndarray,
        extra: np.ndarray | None = None,
    ):
        captured.update(face=face, mask=mask, to_frame=to_frame, extra=extra)
        return original_paste(scene, face, to_frame, port, mask, extra=extra)

    engine._paste = hooked  # type: ignore[method-assign]

    wanted = {int(v) for v in args.frames.split(",") if v.strip()}
    capture = cv2.VideoCapture(str(args.clip))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    assert loaded.portrait_crop is not None and loaded.mask is not None
    # The engine's own reading of the photograph's border (day 19), or the
    # crop's corners when the guard is off for this avatar.
    bg = (
        loaded.background
        if loaded.background is not None
        else background_colour(loaded.portrait_crop)
    )
    print(
        f"{args.clip}: avatar {args.avatar}, background colour (BGR) {bg.astype(int).tolist()}"
        f"{'' if loaded.background is not None else ' (from the crop corners; no guard)'}"
    )

    # One pass for the yaw (to pick the fastest frames), one for the frames.
    yaws: dict[int, float] = {}
    if args.fastest > 0:
        reader = MediaPipeFaceTracker(
            model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
            smoothing=settings.tracking.smoothing,
            mouth_smoothing=settings.tracking.mouth_smoothing,
            min_detection_confidence=settings.tracking.min_detection_confidence,
            min_presence_confidence=settings.tracking.min_presence_confidence,
            min_tracking_confidence=settings.tracking.min_tracking_confidence,
        )
        reader.start()
        index = 0
        while index <= args.last:
            ok, image = capture.read()
            if not ok:
                break
            frame = VideoFrame(
                data=np.ascontiguousarray(image),
                pixel_format=PixelFormat.BGR24,
                metadata=FrameMetadata(frame_id=index, capture_timestamp=index / fps),
            )
            result = reader.process(frame)
            if index >= args.first and result.face_detected and result.head_pose is not None:
                yaws[index] = float(result.head_pose.yaw)
            index += 1
        reader.stop()
        capture.release()
        steps = sorted(
            ((abs(yaws[i] - yaws[i - 1]), i) for i in yaws if i - 1 in yaws), reverse=True
        )
        for step, i in steps[: args.fastest]:
            wanted.add(i)
            print(f"  fastest: f{i} yaw {yaws[i]:+.0f}, {step:.1f} degrees from the frame before")
        capture = cv2.VideoCapture(str(args.clip))

    index = 0
    rows: list[tuple[int, float, int, str]] = []
    try:
        while index <= max(wanted, default=args.last):
            ok, image = capture.read()
            if not ok:
                break
            frame = VideoFrame(
                data=np.ascontiguousarray(image),
                pixel_format=PixelFormat.BGR24,
                metadata=FrameMetadata(frame_id=index, capture_timestamp=index / fps),
            )
            tracking = tracker.process(frame)
            captured.clear()
            out = engine.animate(args.avatar, tracking, frame)
            if index in wanted:
                pose = tracking.head_pose
                yaw = 0.0 if pose is None else float(pose.yaw)
                if args.dump:
                    cv2.imwrite(
                        f"{args.out_prefix}-dump-f{index}.png", np.ascontiguousarray(out.data)
                    )
                if "face" not in captured:
                    rows.append((index, yaw, -1, "no paste (face lost)"))
                    index += 1
                    continue
                face, mask = captured["face"], captured["mask"]
                near = np.max(np.abs(face.astype(np.float32) - bg), axis=2) < args.tolerance
                covered = mask[:, :, 0] > args.alpha
                band = near & covered
                count = int(band.sum())
                ys, xs = np.nonzero(band)
                where = "" if count == 0 else f"box x {xs.min()}-{xs.max()} y {ys.min()}-{ys.max()}"
                rows.append((index, yaw, count, where))
                # The zoom: the mask's box in the crop, the render with the
                # band tinted red | the mask | the paste as it goes out,
                # cut from the frame at the same box.
                assert loaded.mask_bbox is not None
                bx0, by0, bx1, by1 = loaded.mask_bbox
                tinted = face.copy()
                tinted[band] = (0, 0, 255)
                alpha = (mask[:, :, 0] * 255).astype(np.uint8)
                alpha3 = cv2.merge((alpha, alpha, alpha))
                to_frame = captured["to_frame"]
                to_crop = cv2.invertAffineTransform(to_frame)
                pasted = cv2.warpAffine(
                    np.ascontiguousarray(out.data), to_crop, (face.shape[1], face.shape[0])
                )
                zoom = np.hstack(
                    [tinted[by0:by1, bx0:bx1], alpha3[by0:by1, bx0:bx1], pasted[by0:by1, bx0:bx1]]
                )
                zoom = cv2.resize(zoom, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
                cv2.putText(
                    zoom,
                    f"f{index} yaw {yaw:+.0f} band {count} px",
                    (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.imwrite(f"{args.out_prefix}-f{index}.png", zoom)
            index += 1
    finally:
        capture.release()
        tracker.stop()
        engine.stop()
    print(f"  mask pixels within {args.tolerance:.0f} of the background, alpha > {args.alpha}:")
    for index, yaw, count, where in sorted(rows):
        print(f"    f{index} yaw {yaw:+5.0f}: {count:5d} px  {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
