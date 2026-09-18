"""The tracker's reach on a turned head (day 19, the client's "the face is
lost when the head turns"), and the second look in a crop that is meant to
extend it -- see ``MediaPipeFaceTracker`` (``redetect_in_crop``).

Three measurements on one clip, all read off the frames themselves:

1. **Where the tracker lets go and where it finds the face again**, the
   app's own VIDEO tracker over the clip twice -- the second look off and
   on -- with every loss and every find printed with its yaw and length.
2. **The detector's reach**: on every frame (or every ``--step``-th) of the
   segment a fresh IMAGE-mode read of the whole frame -- what the VIDEO
   landmarker has to do once it has let go -- against the same read of the
   crop the second look would take, binned by the head's yaw. The bin where
   the whole frame stops finding the face and the crop still does is what
   the crop buys.
3. **The crop's pose, checked** (``--check-frames``): on the chosen frames
   the crop's landmarks and pose are mapped back into the frame and set
   against the whole frame's own read (pixels, degrees, depth) -- and the
   whole frame is read again shifted sideways and downward by known
   amounts, to confirm that the fitted turn moves by the angle between the
   rays -- ``_pose_from_crop`` undoes a measured share of it, not all: the
   task's fit is only weakly perspective (0.23-0.35 of the angle on his
   clips).

Usage::

    uv run python scripts/probes/redetect_probe.py --clip recordings/driving.mkv \\
        --first 1342 --last 1641 --check-frames 60,1400,1440,1560,1600
    uv run python scripts/probes/redetect_probe.py --clip recordings/driving-turn.mkv

Everything printed is numbers; nothing is written unless ``--out-prefix`` is
given, and then the check frames' crops are his face -- delete after.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sha256.config.loader import load_settings  # noqa: E402
from sha256.config.paths import AppPaths  # noqa: E402
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FrameMetadata, VideoFrame  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    _FOCAL_PER_HEIGHT,
    _REDETECT_SCALE,
    DEFAULT_MODEL_RELATIVE_PATH,
    FACE_MESH_POINTS,
    MediaPipeFaceTracker,
    _crop_to_frame_points,
    _pose_from_crop,
    _redetect_window,
)


def frame_of(image: np.ndarray, index: int, fps: float) -> VideoFrame:
    return VideoFrame(
        data=np.ascontiguousarray(image),
        pixel_format=PixelFormat.BGR24,
        metadata=FrameMetadata(frame_id=index, capture_timestamp=index / fps),
    )


def face_box(points: np.ndarray, width: int, height: int) -> tuple[float, float, float]:
    """The tracker's own notion of the last face: centre and height in px."""
    pts = points[:FACE_MESH_POINTS, :2]
    xs, ys = pts[:, 0] * width, pts[:, 1] * height
    return (
        float((xs.min() + xs.max()) / 2.0),
        float((ys.min() + ys.max()) / 2.0),
        float(ys.max() - ys.min()),
    )


def shifted(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """The picture moved by (dx, dy) pixels over a flat grey, same size."""
    out = np.full_like(image, 128)
    h, w = image.shape[:2]
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    out[sy0 + dy : sy1 + dy, sx0 + dx : sx1 + dx] = image[sy0:sy1, sx0:sx1]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=None)
    parser.add_argument("--step", type=int, default=2, help="Every n-th frame for the reach.")
    parser.add_argument("--bin", type=float, default=5.0, help="Yaw bin width, degrees.")
    parser.add_argument("--check-frames", default="", help="Frames for the pose check.")
    parser.add_argument("--redetect-confidence", type=float, default=0.3)
    parser.add_argument(
        "--shrink",
        type=float,
        default=1.0,
        help="Scale the picture down by this factor onto a grey canvas of the same size: "
        "a face further from the camera than the clip's, without a new clip.",
    )
    parser.add_argument("--out-prefix", type=Path, default=None)
    args = parser.parse_args()

    import cv2

    def shrunk(image: np.ndarray) -> np.ndarray:
        if args.shrink >= 1.0:
            return image
        h, w = image.shape[:2]
        small = cv2.resize(
            image, None, fx=args.shrink, fy=args.shrink, interpolation=cv2.INTER_AREA
        )
        out = np.full_like(image, 128)
        y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
        out[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small
        return out

    paths = AppPaths.resolve()
    settings = load_settings(paths)
    models = paths.resolve_relative(settings.models.root_path)
    model = models / DEFAULT_MODEL_RELATIVE_PATH
    tracking = settings.tracking

    def video_tracker(redetect: bool) -> MediaPipeFaceTracker:
        return MediaPipeFaceTracker(
            model_path=model,
            smoothing=tracking.smoothing,
            mouth_smoothing=tracking.mouth_smoothing,
            min_detection_confidence=tracking.min_detection_confidence,
            min_presence_confidence=tracking.min_presence_confidence,
            min_tracking_confidence=tracking.min_tracking_confidence,
            redetect_in_crop=redetect,
            redetect_confidence=args.redetect_confidence,
        )

    def still_tracker(detection: float) -> MediaPipeFaceTracker:
        return MediaPipeFaceTracker(
            model_path=model,
            running_mode="image",
            min_detection_confidence=detection,
            min_presence_confidence=tracking.min_presence_confidence,
            min_tracking_confidence=tracking.min_tracking_confidence,
        )

    check = [int(v) for v in args.check_frames.split(",") if v.strip()]
    capture = cv2.VideoCapture(str(args.clip))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    # Only the segment and the check frames are kept: a 720p clip is 2.7 MB
    # a frame, and the whole of driving.mkv would be 5 GB.
    frames: dict[int, np.ndarray] = {}
    index = 0
    while True:
        ok, image = capture.read()
        if not ok or (args.last is not None and index > args.last):
            break
        if index >= args.first or index in check:
            frames[index] = shrunk(image)
        index += 1
    capture.release()
    last = index - 1 if args.last is None else min(args.last, index - 1)
    height, width = next(iter(frames.values())).shape[:2]
    print(f"{args.clip}: {index} frames read, {width}x{height} at {fps:.1f} fps")

    # --- 1. the VIDEO tracker, the second look off and on --------------------
    yaw_by_frame: dict[int, float] = {}
    box_by_frame: dict[int, tuple[float, float, float]] = {}
    for redetect in (False, True):
        tracker = video_tracker(redetect)
        tracker.start()
        events: list[str] = []
        last_seen: tuple[int, float] | None = None
        lost_run = 0
        lost_total = 0
        redetected = 0
        cost_ms: list[float] = []
        try:
            for index in range(args.first, last + 1):
                started = time.perf_counter()
                result = tracker.process(frame_of(frames[index], index, fps))
                cost_ms.append((time.perf_counter() - started) * 1000.0)
                if index < args.first:
                    continue
                if not result.face_detected:
                    if lost_run == 0:
                        where = (
                            "before the segment"
                            if last_seen is None
                            else f"f{last_seen[0]} yaw {last_seen[1]:+.0f}"
                        )
                        events.append(f"lost after {where}")
                    lost_run += 1
                    lost_total += 1
                    continue
                pose = result.head_pose
                yaw = 0.0 if pose is None else float(pose.yaw)
                via = " (in the crop)" if result.extras.get("redetected") else ""
                redetected += bool(result.extras.get("redetected"))
                if lost_run > 0:
                    events[-1] += (
                        f", found at f{index} yaw {yaw:+.0f}{via} after {lost_run} frames "
                        f"({lost_run / fps:.1f} s)"
                    )
                lost_run = 0
                last_seen = (index, yaw)
                if not redetect and result.landmarks is not None:
                    yaw_by_frame[index] = yaw
                    box_by_frame[index] = face_box(result.landmarks.points, width, height)
        finally:
            tracker.stop()
        if lost_run > 0:
            events[-1] += f", still lost at the end ({lost_run} frames)"
        label = "second look ON " if redetect else "second look OFF"
        print(
            f"  {label}: frames {args.first}-{last}, lost {lost_total}, "
            f"found in the crop {redetected}, tracker {np.median(cost_ms):.1f} ms/frame "
            f"(p95 {np.percentile(cost_ms, 95):.1f})"
        )
        for event in events:
            print(f"    the tracker: {event}")

    # --- 2. the detector's reach: whole frame vs the crop, by yaw -----------
    whole = still_tracker(tracking.min_detection_confidence)
    crop_reader = still_tracker(args.redetect_confidence)
    whole.start()
    crop_reader.start()
    # frames, found on the whole frame, frames with a crop, found in the crop
    bins: dict[float, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    try:
        previous_box: tuple[float, float, float] | None = None
        for index in range(args.first, last + 1, args.step):
            box = box_by_frame.get(index, previous_box)
            if box is None:
                continue
            previous_box = box
            yaw = yaw_by_frame.get(index)
            # A frame the VIDEO tracker lost has no yaw: those frames -- the
            # ones that matter -- are counted in their own row.
            key = math.inf if yaw is None else math.floor(yaw / args.bin) * args.bin
            bins[key][0] += 1
            image = frames[index]
            bins[key][1] += whole.process(frame_of(image, index, fps)).face_detected
            window = _redetect_window(box, _REDETECT_SCALE, width, height)
            if window is None:
                continue  # the crop would be the whole frame: the face is that large
            bins[key][2] += 1
            x0, y0, w, h = window
            patch = image[y0 : y0 + h, x0 : x0 + w]
            bins[key][3] += crop_reader.process(frame_of(patch, index, fps)).face_detected
        print("  the detector's reach (a fresh read, no tracking state), by the VIDEO yaw:")
        print("    yaw bin   frames   whole frame   the crop (of the frames small enough for one)")
        for key in sorted(bins):
            n, found_whole, with_crop, found_crop = bins[key]
            crop_text = (
                "   (none)"
                if with_crop == 0
                else f"{100.0 * found_crop / with_crop:6.0f}% of {with_crop}"
            )
            row = (
                "  lost by VIDEO" if key == math.inf else f"    {key:+5.0f}..{key + args.bin:+4.0f}"
            )
            print(f"{row}  {n:6d}   {100.0 * found_whole / n:6.0f}%      {crop_text}")
    finally:
        whole.stop()
        crop_reader.stop()

    # --- 3. the crop's pose, checked against the whole frame's ---------------
    if not check:
        return 0
    whole = still_tracker(tracking.min_detection_confidence)
    crop_reader = still_tracker(args.redetect_confidence)
    whole.start()
    crop_reader.start()
    focal = _FOCAL_PER_HEIGHT * height
    shifts = ((-320, 0), (-160, 0), (160, 0), (320, 0), (0, -120), (0, 120))
    slopes_yaw: list[float] = []
    slopes_pitch: list[float] = []
    try:
        print("  the crop's pose against the whole frame's (a fresh read of each):")
        for index in check:
            if index > last:
                continue
            image = frames[index]
            base = whole.process(frame_of(image, index, fps))
            if not base.face_detected or base.landmarks is None or base.head_pose is None:
                print(f"    f{index}: the whole frame has no face")
                continue
            box = face_box(base.landmarks.points, width, height)
            window = _redetect_window(box, _REDETECT_SCALE, width, height)
            if window is None:
                print(f"    f{index}: the crop would be the whole frame")
                continue
            x0, y0, w, h = window
            patch = image[y0 : y0 + h, x0 : x0 + w]
            read = crop_reader.process(frame_of(patch, index, fps))
            p0 = base.head_pose
            if not read.face_detected or read.landmarks is None or read.head_pose is None:
                print(f"    f{index}: yaw {p0.yaw:+.1f} -- the crop has no face")
                continue
            crop = (x0, y0, w, h, width, height)
            mapped = _crop_to_frame_points(read.landmarks.points, crop)
            raw = read.head_pose
            fixed = _pose_from_crop(raw, crop)
            px = np.hypot(
                (mapped[:FACE_MESH_POINTS, 0] - base.landmarks.points[:FACE_MESH_POINTS, 0])
                * width,
                (mapped[:FACE_MESH_POINTS, 1] - base.landmarks.points[:FACE_MESH_POINTS, 1])
                * height,
            )
            z0, zc, zf = (
                abs(p0.translation[2]),
                abs(raw.translation[2]),
                abs(fixed.translation[2]),
            )
            print(
                f"    f{index}: whole yaw {p0.yaw:+.1f} pitch {p0.pitch:+.1f} z {z0:.1f} | "
                f"crop raw yaw {raw.yaw:+.1f} pitch {raw.pitch:+.1f} z {zc:.1f} | "
                f"mapped yaw {fixed.yaw:+.1f} pitch {fixed.pitch:+.1f} z {zf:.1f} | "
                f"points {np.median(px):.1f} px median, {px.max():.1f} max | "
                f"crop {w}x{h} at ({x0},{y0})"
            )
            if args.out_prefix is not None:
                args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f"{args.out_prefix}-f{index}-crop.png", patch)
            # The shift check: the same pixels moved sideways / downward.
            for dx, dy in shifts:
                moved = whole.process(frame_of(shifted(image, dx, dy), index, fps))
                if not moved.face_detected or moved.head_pose is None:
                    print(f"      shifted ({dx:+d},{dy:+d}): no face")
                    continue
                m = moved.head_pose
                if dx:
                    expect = -math.degrees(math.atan2(dx, focal))
                    slopes_yaw.append((m.yaw - p0.yaw) / expect)
                    print(
                        f"      shifted ({dx:+d},{dy:+d}): yaw {m.yaw:+.1f} "
                        f"(moved {m.yaw - p0.yaw:+.1f}, expected {expect:+.1f}), "
                        f"pitch {m.pitch:+.1f}, z {abs(m.translation[2]):.1f}"
                    )
                else:
                    expect = math.degrees(math.atan2(dy, focal))
                    slopes_pitch.append((m.pitch - p0.pitch) / expect)
                    print(
                        f"      shifted ({dx:+d},{dy:+d}): pitch {m.pitch:+.1f} "
                        f"(moved {m.pitch - p0.pitch:+.1f}, expected {expect:+.1f}), "
                        f"yaw {m.yaw:+.1f}, z {abs(m.translation[2]):.1f}"
                    )
        if slopes_yaw:
            print(
                f"  the shift check: yaw moves {np.median(slopes_yaw):.2f} of the ray angle "
                f"(median; {min(slopes_yaw):.2f}..{max(slopes_yaw):.2f}), pitch "
                f"{np.median(slopes_pitch):.2f} ({min(slopes_pitch):.2f}..{max(slopes_pitch):.2f})"
                " -- 1.00 is what _pose_from_crop assumes"
            )
    finally:
        whole.stop()
        crop_reader.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
