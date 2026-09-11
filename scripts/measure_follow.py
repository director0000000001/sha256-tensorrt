"""Does the neural face sit where his face is, and does it follow him?

**Three trackers, on purpose.** The driver frame is tracked twice: once
through the app's own filter (the reading that actually drives the engine --
smoothing and mouth-smoothing included, exactly as in the running app) and
once unfiltered (the truth of where his face and expression really are, this
frame, with nothing smoothed away). The engine's own output is tracked a
third time, so the avatar's eyes/nose/mouth can be read the same way his are
and compared on the same frame -- not against a proxy, against what a face
tracker sees when it looks at the avatar.

**Anchors, not raw landmarks.** ``eye``/``nose``/``mouth``/``chin`` are each
built from a few stable points (eye-corner mean, nose tip, mouth-corner mean,
chin) rather than compared point-by-point, because the avatar is a different
face: its own nose is not at his tracker's nose-tip index by pixel identity,
only by anatomical role. Position is reported as avatar-minus-him, in pixels,
+x toward the image's right and +y down.

**Expression following** compares his geometric measure (eye-aspect ratio,
mouth-aspect ratio, mouth width, brow-to-eye gap -- all normalised by his own
inter-ocular distance, or the avatar's own for the avatar's reading) against
the avatar's reading of the same measure on the same frame, over the segment
of the clip built to move that one thing (closing the eyes, opening the
mouth, smiling, raising the brows -- see the clip's own sidecar). ``r`` is the
correlation, ``gain`` the slope of a straight line through (his, avatar's):
1.0 means the avatar's own tracked reading moves exactly as far as his did;
lower means it under-shoots.

**Numbers from this script are comparable only to another run of this script
on the same clip** -- before a change against after, never across clips or
against a number with different units.

Usage::

    uv run python scripts/measure_follow.py --clip recordings/driving.mkv
    uv run python scripts/measure_follow.py --clip recordings/driving.mkv \\
        --stills 700,1590
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sha256.avatar.engine_liveportrait import (  # noqa: E402
    CROP_SIZE,
    DEFAULT_WEIGHTS_RELATIVE_PATH,
    LivePortraitEngine,
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

# MediaPipe's 478-point scheme. Eye corners/lids give the eye-aspect ratio and
# the eye anchor; the inner lip pair and mouth corners give the mouth-aspect
# ratio, mouth width and the mouth anchor; the brow points give the brow gap;
# the outer lip contours against the inner give each lip's own thickness --
# added 2026-09-05, "thin angry lips" (a lip target that pressed the avatar's
# own lips past the portrait's own closed ones -- see engine_liveportrait's
# _lip_target).
L_EYE_OUTER, L_EYE_INNER, L_EYE_UPPER, L_EYE_LOWER = 33, 133, 159, 145
R_EYE_INNER, R_EYE_OUTER, R_EYE_UPPER, R_EYE_LOWER = 362, 263, 386, 374
LIP_UPPER_INNER, LIP_LOWER_INNER, MOUTH_L, MOUTH_R = 13, 14, 61, 291
LIP_UPPER_OUTER, LIP_LOWER_OUTER = 0, 17
BROW_L, BROW_R = 105, 334
JAW_L, JAW_R = 58, 288
NOSE, CHIN, FOREHEAD = 1, 152, 10
L_IRIS, R_IRIS = 468, 473
FACE_POINTS = 468

POSITION_ANCHORS = ("eye", "nose", "mouth", "chin", "forehead", "all")


def measures(pts: np.ndarray) -> dict[str, float]:
    """Geometric expression measures, all normalised by inter-ocular distance."""
    io = float(np.linalg.norm(pts[R_EYE_OUTER, :2] - pts[L_EYE_OUTER, :2])) or 1e-6
    ear_l = float(np.linalg.norm(pts[L_EYE_UPPER, :2] - pts[L_EYE_LOWER, :2])) / (
        float(np.linalg.norm(pts[L_EYE_INNER, :2] - pts[L_EYE_OUTER, :2])) or 1e-6
    )
    ear_r = float(np.linalg.norm(pts[R_EYE_UPPER, :2] - pts[R_EYE_LOWER, :2])) / (
        float(np.linalg.norm(pts[R_EYE_OUTER, :2] - pts[R_EYE_INNER, :2])) or 1e-6
    )
    mouth_w = float(np.linalg.norm(pts[MOUTH_R, :2] - pts[MOUTH_L, :2]))
    mar = float(np.linalg.norm(pts[LIP_UPPER_INNER, :2] - pts[LIP_LOWER_INNER, :2])) / (
        mouth_w or 1e-6
    )
    lip_upper = float(np.linalg.norm(pts[LIP_UPPER_OUTER, :2] - pts[LIP_UPPER_INNER, :2])) / io
    lip_lower = float(np.linalg.norm(pts[LIP_LOWER_OUTER, :2] - pts[LIP_LOWER_INNER, :2])) / io
    brow = (
        float(
            ((pts[L_EYE_UPPER, 1] - pts[BROW_L, 1]) + (pts[R_EYE_UPPER, 1] - pts[BROW_R, 1])) / 2.0
        )
        / io
    )
    # The day-9 gate measures, exactly as the engine reads them (see
    # engine_liveportrait._mouth_width_ratio/_brow_ratio and their comment):
    # mouth width over the jaw width at its height (pitch-stable where
    # mouth_w over inter-ocular is not), and the brows' height above the eye
    # corners' midpoint over that midpoint's height above the chin (moved
    # by neither a blink nor, much, a pitch).
    eye_mid_y = float(
        (pts[L_EYE_OUTER, 1] + pts[L_EYE_INNER, 1] + pts[R_EYE_INNER, 1] + pts[R_EYE_OUTER, 1])
        / 4.0
    )
    mouth_wj = mouth_w / (float(np.linalg.norm(pts[JAW_R, :2] - pts[JAW_L, :2])) or 1e-6)
    # Mouth tilt (day 9, later -- "a side smile"): the two mouth corners'
    # height difference over inter-ocular, signed (+: the right corner, his
    # right = screen left, sits higher). 0 for a level mouth.
    mouth_tilt = float(pts[MOUTH_L, 1] - pts[MOUTH_R, 1]) / io
    brow_c = (((eye_mid_y - float(pts[BROW_L, 1])) + (eye_mid_y - float(pts[BROW_R, 1]))) / 2.0) / (
        (float(pts[CHIN, 1]) - eye_mid_y) or 1e-6
    )
    # Gaze (day 11, 2026-09-06): each iris centre (MediaPipe 468 left, 473
    # right -- present only with the 478-point output) minus its own eye's
    # corner midpoint, over that eye's corner width; the two eyes averaged.
    # +x = toward the image's right, +y = down; 0 = looking straight at
    # the camera. NaN without iris points.
    gaze_x = gaze_y = float("nan")
    if pts.shape[0] > R_IRIS:
        gx, gy = [], []
        eyes = ((L_IRIS, L_EYE_OUTER, L_EYE_INNER), (R_IRIS, R_EYE_OUTER, R_EYE_INNER))
        for iris, outer, inner in eyes:
            centre = (pts[outer, :2] + pts[inner, :2]) / 2.0
            width = abs(float(pts[outer, 0] - pts[inner, 0])) or 1e-6
            gx.append(float(pts[iris, 0] - centre[0]) / width)
            gy.append(float(pts[iris, 1] - centre[1]) / width)
        gaze_x, gaze_y = sum(gx) / 2.0, sum(gy) / 2.0
    return {
        "gaze_x": gaze_x,
        "gaze_y": gaze_y,
        "ear_l": ear_l,
        "ear_r": ear_r,
        "ear": (ear_l + ear_r) / 2.0,
        "mar": mar,
        "mouth_w": mouth_w / io,
        "mouth_wj": mouth_wj,
        "mouth_tilt": mouth_tilt,
        "lip_upper": lip_upper,
        "lip_lower": lip_lower,
        "brow": brow,
        "brow_c": brow_c,
        "io": io,
    }


def anchors(pts_px: np.ndarray) -> dict[str, np.ndarray]:
    eye_c = (
        pts_px[L_EYE_OUTER, :2]
        + pts_px[L_EYE_INNER, :2]
        + pts_px[R_EYE_INNER, :2]
        + pts_px[R_EYE_OUTER, :2]
    ) / 4.0
    mouth_c = (pts_px[MOUTH_L, :2] + pts_px[MOUTH_R, :2]) / 2.0
    return {
        "eye": eye_c,
        "nose": pts_px[NOSE, :2],
        "mouth": mouth_c,
        "chin": pts_px[CHIN, :2],
        "forehead": pts_px[FOREHEAD, :2],
        "all": pts_px[:FACE_POINTS, :2].mean(axis=0),
    }


def make_tracker(settings, models: Path, *, filtered: bool) -> MediaPipeFaceTracker:
    return MediaPipeFaceTracker(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        smoothing=settings.tracking.smoothing if filtered else 0.0,
        mouth_smoothing=settings.tracking.mouth_smoothing if filtered else 0.0,
        min_detection_confidence=settings.tracking.min_detection_confidence,
        min_presence_confidence=settings.tracking.min_presence_confidence,
        min_tracking_confidence=settings.tracking.min_tracking_confidence,
    )


def parse_options(items: list[str]) -> dict[str, object]:
    """``--option key=value`` pairs into engine_options overrides: the value is
    read as JSON (``0.12``, ``true``, ``"text"``), else kept as the string."""
    overrides: dict[str, object] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--option expects KEY=VALUE, got {item!r}")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    return overrides


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q * 100.0))


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True, help="Driving clip to measure.")
    parser.add_argument("--avatar", default="my_avatar_neural")
    parser.add_argument(
        "--csv", type=Path, default=None, help="Defaults to '<clip stem>-follow.csv' beside it."
    )
    parser.add_argument(
        "--stills",
        default="",
        help="Comma-separated frame indices to write as overlay stills "
        "('<clip stem>-follow-f<N>.png' beside the clip): his tracked points "
        "in red, his anchors in green, the avatar's own in magenta.",
    )
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one of the avatar's engine_options for this run (the value is "
        "read as JSON, else kept as a string); repeatable. The avatar's own "
        "metadata.json is not touched -- for sweeps.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="Stop after this frame index (the clip is always read from frame 0, so "
        "the anchor is learned exactly as in a full run) -- for a quicker look at "
        "one part of a clip.",
    )
    args = parser.parse_args()
    stills = {int(s) for s in args.stills.split(",") if s.strip()}
    csv_out = args.csv or args.clip.with_name(f"{args.clip.stem}-follow.csv")
    overrides = parse_options(args.option)
    sidecar = args.clip.with_suffix(".json")

    paths = AppPaths.resolve()
    settings = load_settings(paths)
    models = paths.resolve_relative(settings.models.root_path)
    repository = FilesystemAvatarRepository(paths.resolve_relative(settings.avatars.library_path))
    segments = (
        json.loads(sidecar.read_text(encoding="utf-8"))["segments"] if sidecar.is_file() else []
    )

    def segment_of(index: int) -> str:
        for seg in segments:
            if seg["first_frame"] <= index <= seg["last_frame"]:
                return str(seg["instruction"])[:28]
        return "?"

    driver_filtered = make_tracker(settings, models, filtered=True)
    driver_raw = make_tracker(settings, models, filtered=False)
    output_raw = make_tracker(settings, models, filtered=False)
    for t in (driver_filtered, driver_raw, output_raw):
        t.start()

    neural = LivePortraitEngine(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        weights=models / DEFAULT_WEIGHTS_RELATIVE_PATH,
    )
    neural.start()
    avatar = repository.get(args.avatar)
    avatar.manifest.engine_options.update(overrides)
    if overrides:
        print(f"engine_options overridden for this run: {overrides}")
    neural.load_avatar(avatar)

    capture = cv2.VideoCapture(str(args.clip))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    rows: list[dict[str, float | int | str]] = []
    index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok or (args.last is not None and index > args.last):
                break
            h, w = image.shape[:2]
            frame = VideoFrame(
                data=np.ascontiguousarray(image),
                pixel_format=PixelFormat.BGR24,
                metadata=FrameMetadata(frame_id=index, capture_timestamp=index / fps),
            )
            filt = driver_filtered.process(frame)
            raw = driver_raw.process(frame)
            out = neural.animate(args.avatar, filt, frame)
            out_frame = VideoFrame(
                data=np.ascontiguousarray(out.data),
                pixel_format=PixelFormat.BGR24,
                metadata=FrameMetadata(frame_id=index, capture_timestamp=index / fps),
            )
            got = output_raw.process(out_frame)

            row: dict[str, float | int | str] = {"frame": index, "segment": segment_of(index)}
            # The day-9 gates as applied this frame (engine state, read for
            # the record only): how much of the extractor's mouth/brow
            # reading reached the portrait.
            portrait = neural._loaded.get(args.avatar)
            if portrait is not None:
                row["gate_mouth"] = portrait.gate_mouth_prev
                row["gate_brow"] = portrait.gate_brow_prev
                row["gate_open"] = portrait.gate_open_prev
                # Day 10 (4): whether this frame's face came from the network
                # or is the extreme-pose hold's frozen reuse of the last one
                # (see _should_hold) -- its hysteresis means a pitch bin near
                # the threshold mixes held and rendered frames depending on
                # whether it was reached on the way in or the way out, so a
                # "not held" filter needs this, not just a pitch cutoff.
                row["held"] = int(portrait.holding)
                # Day 13: the extractor's own relative scale as _drive uses
                # it (the size-smoothed reading over the anchor's), before
                # scale_gain -- what an open mouth is mistaken for.
                if portrait.scale_prev is not None and portrait.scale_d0 is not None:
                    row["rel_scale"] = float(portrait.scale_prev / portrait.scale_d0)
                # Day 13: how far the portrait's oval was let down at the
                # chin this frame, crop px (see the engine's _chin_drop).
                row["chin_drop"] = portrait.chin_drop
                # Day 10 (4): the crop's own size in frame pixels this frame --
                # `to_frame`'s linear part is a rotation scaled by size/CROP_SIZE
                # (see _similarity), so its first column's norm times CROP_SIZE
                # is the crop's edge length in the *source* frame. Read after
                # animate() has run, so it is this frame's crop, not the last
                # one's -- unchanged while a face is lost (the hold keeps the
                # last crop's own `to_frame`, which is exactly what a held
                # frame renders with).
                if portrait.last is not None:
                    to_frame = portrait.last[1]
                    row["crop_px"] = float(CROP_SIZE * np.linalg.norm(to_frame[:, 0]))
            if raw.face_detected and raw.landmarks is not None and raw.head_pose is not None:
                rp = raw.landmarks.points.copy()
                rp[:, 0] *= w
                rp[:, 1] *= h
                row.update(
                    yaw=raw.head_pose.yaw, pitch=raw.head_pose.pitch, roll=raw.head_pose.roll
                )
                rm = measures(rp)
                ra = anchors(rp)
                for k, v in rm.items():
                    row[f"d_{k}"] = v
                if got.face_detected and got.landmarks is not None:
                    op = got.landmarks.points.copy()
                    op[:, 0] *= w
                    op[:, 1] *= h
                    om = measures(op)
                    oa = anchors(op)
                    for k, v in om.items():
                        row[f"o_{k}"] = v
                    # Day 10 (4): the same instrument (this script's own
                    # tracker) reading the avatar's own pose, so it is
                    # comparable to his `pitch`/`yaw`/`roll` above without a
                    # cross-tracker offset -- MediaPipe on him, MediaPipe on
                    # the avatar.
                    if got.head_pose is not None:
                        row["o_yaw"] = got.head_pose.yaw
                        row["o_pitch"] = got.head_pose.pitch
                        row["o_roll"] = got.head_pose.roll
                    for k in POSITION_ANCHORS:
                        row[f"dx_{k}"] = float(oa[k][0] - ra[k][0])
                        row[f"dy_{k}"] = float(oa[k][1] - ra[k][1])
                    row["scale"] = om["io"] / rm["io"]
                    if index in stills:
                        overlay = out.data.copy()
                        for x, y in rp[:FACE_POINTS, :2]:
                            cv2.circle(overlay, (int(x), int(y)), 1, (0, 0, 255), -1)
                        for k in ("eye", "nose", "mouth", "chin"):
                            cv2.circle(overlay, tuple(int(v) for v in ra[k]), 6, (0, 255, 0), 2)
                            cv2.circle(overlay, tuple(int(v) for v in oa[k]), 6, (255, 0, 255), 2)
                        cv2.putText(
                            overlay,
                            f"frame {index} yaw {raw.head_pose.yaw:+.1f}  green=him magenta=avatar",
                            (12, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        still_path = args.clip.with_name(f"{args.clip.stem}-follow-f{index}.png")
                        cv2.imwrite(str(still_path), overlay)
                else:
                    row["output_lost"] = 1
            else:
                row["driver_lost"] = 1
            rows.append(row)
            index += 1
            if index % 300 == 0:
                print(f"  ... {index} frames", flush=True)
    finally:
        capture.release()
        neural.stop()
        for t in (driver_filtered, driver_raw, output_raw):
            t.stop()

    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with csv_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    good = [r for r in rows if "dx_eye" in r]
    print(
        f"\n{len(rows)} frames, {len(good)} with both faces tracked, "
        f"{sum(1 for r in rows if 'output_lost' in r)} output lost, "
        f"{sum(1 for r in rows if 'driver_lost' in r)} driver lost"
    )

    def col(rs: list[dict[str, float | int | str]], k: str) -> list[float]:
        return [float(r[k]) for r in rs if k in r]

    print("\nPOSITION  (avatar minus him, pixels; +x = right on screen, +y = down)")
    print(
        f"  {'anchor':<9} {'dx p50':>8} {'dx p10':>8} {'dx p90':>8}   "
        f"{'dy p50':>8} {'dy p10':>8} {'dy p90':>8}"
    )
    for k in POSITION_ANCHORS:
        dx, dy = col(good, f"dx_{k}"), col(good, f"dy_{k}")
        print(
            f"  {k:<9} {pct(dx, 0.5):8.1f} {pct(dx, 0.1):8.1f} {pct(dx, 0.9):8.1f}   "
            f"{pct(dy, 0.5):8.1f} {pct(dy, 0.1):8.1f} {pct(dy, 0.9):8.1f}"
        )
    sc = col(good, "scale")
    print(
        f"  scale (avatar inter-ocular / his): "
        f"p50 {pct(sc, 0.5):.3f}  p10 {pct(sc, 0.1):.3f}  p90 {pct(sc, 0.9):.3f}"
    )

    if segments:
        print("\nPOSITION BY SEGMENT  (dx/dy of nose, p50 | of eye centre, p50)")
        for seg in segments:
            name = str(seg["instruction"])[:28]
            rs = [r for r in good if r["segment"] == name]
            if not rs:
                continue
            nose_dx, nose_dy = pct(col(rs, "dx_nose"), 0.5), pct(col(rs, "dy_nose"), 0.5)
            eye_dx, eye_dy = pct(col(rs, "dx_eye"), 0.5), pct(col(rs, "dy_eye"), 0.5)
            yaw_lo, yaw_hi = pct(col(rs, "yaw"), 0.1), pct(col(rs, "yaw"), 0.9)
            print(
                f"  {name:<30} n={len(rs):4d}  nose dx {nose_dx:6.1f} dy {nose_dy:6.1f}"
                f"   eye dx {eye_dx:6.1f} dy {eye_dy:6.1f}"
                f"   yaw {yaw_lo:6.1f}..{yaw_hi:6.1f}"
            )

    print("\nPOSITION BY YAW  (nose dx p50; his yaw in degrees, + = his left = screen right)")
    bins = [(-90, -15), (-15, -8), (-8, -3), (-3, 3), (3, 8), (8, 15), (15, 90)]
    for lo, hi in bins:
        rs = [r for r in good if lo <= float(r["yaw"]) < hi]
        if not rs:
            continue
        nose_dx = pct(col(rs, "dx_nose"), 0.5)
        eye_dx = pct(col(rs, "dx_eye"), 0.5)
        mouth_dx = pct(col(rs, "dx_mouth"), 0.5)
        chin_dx = pct(col(rs, "dx_chin"), 0.5)
        nose_dy = pct(col(rs, "dy_nose"), 0.5)
        print(
            f"  yaw [{lo:>3},{hi:>3})  n={len(rs):4d}  nose dx {nose_dx:6.1f}  "
            f"eye dx {eye_dx:6.1f}  mouth dx {mouth_dx:6.1f}  "
            f"chin dx {chin_dx:6.1f}   nose dy {nose_dy:6.1f}"
        )

    print("\nEXPRESSION FOLLOWING  (his reading vs the avatar's, same frame)")
    print(
        f"  {'measure':<9} {'r':>6} {'gain':>6}   {'his min':>8} {'his max':>8}   "
        f"{'avatar min':>10} {'avatar max':>10}   where"
    )
    checks = [
        ("ear", "Blink normally, several tim|Close your eyes for two seco"),
        ("mar", "Open your mouth wide, then c|TALK. Say anything, normally"),
        ("mouth_w", "Smile with your mouth closed|Smile widely, showing your t"),
        ("brow", "Raise your eyebrows, then fr"),
    ]
    for k, where in checks:
        names = where.split("|")
        rs = [r for r in good if r["segment"] in names]
        d, o = col(rs, f"d_{k}"), col(rs, f"o_{k}")
        if len(d) < 3:
            continue
        d_a, o_a = np.asarray(d), np.asarray(o)
        r_ = float(np.corrcoef(d_a, o_a)[0, 1])
        gain = float(np.polyfit(d_a, o_a, 1)[0])
        print(
            f"  {k:<9} {r_:6.2f} {gain:6.2f}   {d_a.min():8.3f} {d_a.max():8.3f}   "
            f"{o_a.min():10.3f} {o_a.max():10.3f}   {where}"
        )
    d, o = col(good, "d_ear"), col(good, "o_ear")
    if d:
        print(f"  ear whole clip: r {np.corrcoef(d, o)[0, 1]:.2f}")
    d, o = col(good, "d_mar"), col(good, "o_mar")
    if d:
        print(f"  mar whole clip: r {np.corrcoef(d, o)[0, 1]:.2f}")

    print("\nLIP THICKNESS  (avatar / his, median -- 1.0 is a match, low is thin)")
    for name in (
        "SIT STILL. Look straight at ",  # [:28] of the sidecar's own text
        "SIT STILL again. Relax.",
        "Smile with your mouth closed",
    ):
        rs = [r for r in good if r["segment"] == name]
        if not rs:
            continue
        for k in ("lip_upper", "lip_lower"):
            d, o = col(rs, f"d_{k}"), col(rs, f"o_{k}")
            if not d:
                continue
            ratio = float(np.median(np.asarray(o) / np.maximum(np.asarray(d), 1e-6)))
            print(f"  {name:<30} {k:<10} avatar/his {ratio:.2f}")

    print(f"\nCSV: {csv_out}")
    if stills:
        print(f"Overlays: {args.clip.with_name(args.clip.stem + '-follow-f*.png')}  (his face)")
    print("This is the owner's face. Delete the outputs when they have been looked at.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
