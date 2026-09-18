"""The paste's edge against his profile at a turn (day 20, step 3 of the
turn plan; his two screenshots of 2026-09-18: his brow's end and eye corner
beside the avatar's, his nose tip beside the drawn nose).

A clip through the app's own tracker and engine; on the chosen frames the
driver's crop points and the paste are hooked, and the frame's mask is
rebuilt under each value of ``_PROFILE_PUSH`` (``--push``, erosion radii;
0 is the mask as it was before day 20). The profile is the midline points
that lie outside the rim's own hull that frame (``--outside`` crop px or
more); from each, along the ray from the rim's centre inward, the probe
reads how far in the pasted mask reaches half and nine tenths -- the strip
of his own face along the profile, in crop pixels. Frames under
``--frontal-under`` degrees prove the push moves nothing: the mask must be
the same bytes at every push value. ``--zoom`` writes, per frame, the paste
around the profile under each push value side by side (his face; delete
after looking).

Usage::

    uv run python scripts/probes/profile_probe.py --clip recordings/driving-turn.mkv \\
        --frames 298,452,459,553,697,804,1042 --frontal 30,60,90,110,338,499,900 --zoom
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "probes"))

from capture_replay import parse_options  # noqa: E402

from sha256.avatar import engine_liveportrait as engine_module  # noqa: E402
from sha256.avatar.engine_liveportrait import (  # noqa: E402
    _FACE_OVAL,
    _MIDLINE,
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


def profile_points(crop_points: np.ndarray, outside: float) -> list[tuple[int, float]]:
    """The midline points at least ``outside`` crop px outside the rim's
    hull, as (index, how far outside)."""
    import cv2

    rim = np.ascontiguousarray(crop_points[list(_FACE_OVAL)], dtype=np.float32)
    hull = cv2.convexHull(rim)
    found = []
    for index in _MIDLINE:
        x, y = float(crop_points[index, 0]), float(crop_points[index, 1])
        distance = -cv2.pointPolygonTest(hull, (x, y), True)
        if distance >= outside:
            found.append((index, distance))
    return found


def reach_inward(
    mask: np.ndarray, point: np.ndarray, centre: np.ndarray, levels=(0.5, 0.9), limit: int = 60
) -> list[float | None]:
    """Along the ray from ``point`` toward ``centre``, the distance at which
    the mask first reaches each level; None when it never does."""
    direction = centre - point
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return [None for _ in levels]
    direction = direction / length
    out: list[float | None] = [None for _ in levels]
    height, width = mask.shape[:2]
    for step in range(0, limit * 2):
        t = step / 2.0
        x, y = point + direction * t
        xi, yi = round(float(x)), round(float(y))
        if not (0 <= xi < width and 0 <= yi < height):
            break
        value = float(mask[yi, xi])
        for k, level in enumerate(levels):
            if out[k] is None and value >= level:
                out[k] = t
    return out


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--avatar", default="my_avatar_neural")
    parser.add_argument("--frames", default="298,452,459,553,697,804,1042")
    parser.add_argument("--frontal", default="30,60,90,110,338,499,900")
    parser.add_argument("--frontal-under", type=float, default=12.0)
    parser.add_argument("--push", default="0,1,2")
    parser.add_argument("--outside", type=float, default=2.0)
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--zoom", action="store_true")
    parser.add_argument("--out-prefix", type=Path, default=Path("recordings/day20/profile"))
    args = parser.parse_args()

    wanted = {int(v) for v in args.frames.split(",") if v.strip()}
    frontal = {int(v) for v in args.frontal.split(",") if v.strip()}
    pushes = [float(v) for v in args.push.split(",") if v.strip()]
    default_push = engine_module._PROFILE_PUSH

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

    # The frame's mask as the engine builds it, hooked with its inputs, so
    # it can be rebuilt under each push value; the paste, for the zooms.
    captured: dict[str, Any] = {}
    original_mask = engine_module._frame_masks
    original_paste = engine._paste

    def hooked_mask(portrait_mask, driver_crop_points, **kwargs):
        captured.update(portrait_mask=portrait_mask, crop_points=driver_crop_points, kw=kwargs)
        return original_mask(portrait_mask, driver_crop_points, **kwargs)

    def hooked_paste(scene, face, to_frame, port, mask, extra=None):
        captured.update(scene=scene, face=face, to_frame=to_frame, extra=extra)
        return original_paste(scene, face, to_frame, port, mask, extra=extra)

    engine_module._frame_masks = hooked_mask  # type: ignore[assignment]
    engine._paste = hooked_paste  # type: ignore[method-assign]

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.clip))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    print(f"{args.clip}: avatar {args.avatar}, push values {pushes} (erosion radii)")
    rows: list[str] = []
    identical = differing = 0
    index = 0
    try:
        while index <= max(wanted | frontal):
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
            engine.animate(args.avatar, tracking, frame)
            if index not in wanted and index not in frontal:
                index += 1
                continue
            pose = tracking.head_pose
            if "crop_points" not in captured or pose is None:
                rows.append(f"  f{index}: no mask (face lost)")
                index += 1
                continue
            rel_yaw = float(pose.yaw) - (loaded.rest_yaw or 0.0)
            crop_points = captured["crop_points"]
            masks: dict[float, np.ndarray] = {}
            for push in pushes:
                engine_module._PROFILE_PUSH = push
                masks[push] = original_mask(
                    captured["portrait_mask"], crop_points, **captured["kw"]
                )[0][:, :, 0]
            engine_module._PROFILE_PUSH = default_push
            if index in frontal or abs(rel_yaw) < args.frontal_under:
                same = all(np.array_equal(masks[pushes[0]], m) for m in masks.values())
                identical += same
                differing += not same
                rows.append(
                    f"  f{index} yaw {rel_yaw:+5.0f}: frontal, the mask "
                    f"{'identical' if same else 'DIFFERS'} across the push values"
                )
                if index not in wanted:
                    index += 1
                    continue
            points = profile_points(crop_points, args.outside)
            rim_centre = crop_points[list(_FACE_OVAL)].mean(axis=0)
            line = f"  f{index} yaw {rel_yaw:+5.0f}: {len(points)} profile points"
            if points:
                near, far = min(d for _, d in points), max(d for _, d in points)
                line += f" (outside the rim by {near:.0f}-{far:.0f} px)"
            for push in pushes:
                halves, nines = [], []
                for point_index, _ in points:
                    half, nine = reach_inward(masks[push], crop_points[point_index], rim_centre)
                    if half is not None:
                        halves.append(half)
                    if nine is not None:
                        nines.append(nine)
                if halves:
                    line += (
                        f" | push {push:g}: half at {statistics.median(halves):.1f} px in "
                        f"(max {max(halves):.1f}), 0.9 at "
                        f"{statistics.median(nines):.1f} (max {max(nines):.1f})"
                        if nines
                        else f" | push {push:g}: half at {statistics.median(halves):.1f} px in"
                    )
                else:
                    line += f" | push {push:g}: no profile point reaches the mask"
            rows.append(line)
            if args.zoom and points and "face" in captured:
                panels = []
                to_frame = captured["to_frame"]
                to_crop = cv2.invertAffineTransform(to_frame)
                xs = [crop_points[i, 0] for i, _ in points]
                ys = [crop_points[i, 1] for i, _ in points]
                x0 = max(int(min(xs)) - 50, 0)
                x1 = min(int(max(xs)) + 50, 512)
                y0 = max(int(min(ys)) - 40, 0)
                y1 = min(int(max(ys)) + 40, 512)
                for push in pushes:
                    mask3 = masks[push][:, :, None]
                    pasted_frame = original_paste(
                        captured["scene"].copy(), captured["face"], to_frame, loaded, mask3
                    )
                    pasted = cv2.warpAffine(np.ascontiguousarray(pasted_frame), to_crop, (512, 512))
                    panel = pasted[y0:y1, x0:x1].copy()
                    half = (masks[push][y0:y1, x0:x1] >= 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    panel = cv2.resize(panel, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
                    cv2.drawContours(panel, [c * 3 for c in contours], -1, (0, 0, 255), 1)
                    for i, _ in points:
                        px, py = crop_points[i]
                        cv2.circle(
                            panel, (int((px - x0) * 3), int((py - y0) * 3)), 3, (0, 255, 0), -1
                        )
                    cv2.putText(
                        panel,
                        f"push {push:g}",
                        (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    panels.append(panel)
                sheet = np.hstack(panels)
                cv2.putText(
                    sheet,
                    f"f{index} yaw {rel_yaw:+.0f}",
                    (6, sheet.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imwrite(f"{args.out_prefix}-f{index}.png", sheet)
            index += 1
    finally:
        capture.release()
        tracker.stop()
        engine.stop()
        engine_module._frame_masks = original_mask  # type: ignore[assignment]
        engine_module._PROFILE_PUSH = default_push
    for row in rows:
        print(row)
    print(f"  frontal frames: {identical} identical across the push values, {differing} differing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
