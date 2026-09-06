"""Render a driving clip with the LivePortrait engine, beside the warp engine.

**What it is for.** The owner judges by eye, and the neural engine's whole case
is the picture. This puts three panels in one video — the camera, the
photograph engine, the neural engine — frame for frame on the same clip, so
the comparison is of the two engines and nothing else. It also prints what the
neural engine costs per frame, split into the crop, the network and the paste,
because the budget at 30 fps is 33.2 ms and the tracker takes 8 of them.

**The seam number.** Alongside timings, it measures how strong an edge the
paste mask itself leaves in the output: the mask, warped into the frame, gives
a band where it is neither fully off nor fully on (0.05 < alpha < 0.95); inside
a margin of that band closest to alpha 0.5 — the boundary itself, not the
whole feather — it takes the output's own gradient magnitude (Sobel). A mask
that fades into skin on both sides leaves a smooth boundary and a low number; a
mask edge sitting on hair or on a hard jaw line leaves a step and a high one.
The band's rows are split into thirds by their own extent so the top
(hairline) and bottom (jaw) can be read apart from the whole. This is a
gradient in intensity units, not a physical distance — only comparable to
itself, before a change against after, same clip, same frames.

**Privacy.** The clip is the owner's real face and so is the output. Both stay
under ``recordings/``, which is gitignored; delete the output when it has been
looked at.

Usage::

    uv run python scripts/render_liveportrait.py --clip recordings/driving.mkv
    uv run python scripts/render_liveportrait.py --clip recordings/driving-speech.mkv \\
        --first 300 --last 600 --no-warp
    uv run python scripts/render_liveportrait.py --clip recordings/driving.mkv \\
        --stills 320,450,600,680
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sha256.avatar.engine_liveportrait import (  # noqa: E402
    DEFAULT_WEIGHTS_RELATIVE_PATH,
    LivePortraitEngine,
)
from sha256.avatar.engine_warp import WarpAvatarEngine  # noqa: E402
from sha256.avatar.repository import FilesystemAvatarRepository  # noqa: E402
from sha256.config.loader import load_settings  # noqa: E402
from sha256.config.paths import AppPaths  # noqa: E402
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FrameMetadata, VideoFrame  # noqa: E402
from sha256.performance.metrics import _percentile  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)

PANEL_WIDTH = 640

# How close to the mask's own 0.5 crossing a pixel has to sit to count as part
# of the boundary rather than the whole feather -- see the seam paragraph in
# this file's docstring.
SEAM_EDGE_MARGIN = 0.15


def summary(name: str, values: list[float]) -> str:
    ordered = sorted(values)
    if not ordered:
        return f"  {name:<14} n/a"
    return (
        f"  {name:<14} p50 {_percentile(ordered, 0.50):6.2f} ms   "
        f"p90 {_percentile(ordered, 0.90):6.2f}   max {ordered[-1]:6.2f}"
    )


def seam_summary(name: str, values: list[float]) -> str:
    ordered = sorted(v for v in values if not math.isnan(v))
    if not ordered:
        return f"  {name:<14} n/a"
    return (
        f"  {name:<14} p50 {_percentile(ordered, 0.50):7.2f}   "
        f"p90 {_percentile(ordered, 0.90):7.2f}   max {ordered[-1]:7.2f}   n={len(ordered)}"
    )


def _seam_edge_values(
    output: np.ndarray, scene_shape: tuple[int, int], mask: np.ndarray, to_frame: np.ndarray
) -> dict[str, float]:
    """The mask-edge gradient for one frame, split whole / hairline / jaw.

    ``mask`` is the crop-space paste mask (``(size, size, 1)`` float32) and
    ``to_frame`` the affine that placed it -- exactly what ``_paste`` itself
    used, so this measures the boundary that was actually drawn, not a
    recomputed one.
    """
    import cv2

    height, width = scene_shape
    alpha = cv2.warpAffine(mask[:, :, 0], to_frame, (width, height), flags=cv2.INTER_LINEAR)
    band = (alpha > 0.05) & (alpha < 0.95)
    if not band.any():
        return {}
    rows = np.where(band.any(axis=1))[0]
    top, bottom = int(rows.min()), int(rows.max())
    third = max(1, (bottom - top + 1) // 3)

    grey = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    boundary = band & (np.abs(alpha - 0.5) < SEAM_EDGE_MARGIN)
    values: dict[str, float] = {}
    if boundary.any():
        values["whole"] = float(gradient[boundary].mean())
    hairline = boundary.copy()
    hairline[top + third :, :] = False
    if hairline.any():
        values["hairline"] = float(gradient[hairline].mean())
    jaw = boundary.copy()
    jaw[: bottom - third + 1, :] = False
    if jaw.any():
        values["jaw"] = float(gradient[jaw].mean())
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip",
        type=Path,
        required=True,
        help="Driving clip. No default -- day 4 (2026-09-05) lost a run to the "
        "old default silently being the speech clip, not the fuller one intended.",
    )
    parser.add_argument("--avatar", default="my_avatar_neural")
    parser.add_argument("--warp-avatar", default="my_avatar")
    parser.add_argument("--no-warp", action="store_true", help="Skip the photograph engine.")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=None)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "recordings" / "neural-side-by-side.mp4"
    )
    parser.add_argument(
        "--stills",
        default="",
        help="Comma-separated frame indices to also write as '<out stem>-f<N>.png' beside --out.",
    )
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one of the neural avatar's engine_options for this run (the "
        "value is read as JSON, else kept as a string); repeatable. metadata.json "
        "is not touched.",
    )
    args = parser.parse_args()
    stills = {int(s) for s in args.stills.split(",") if s.strip()}
    overrides: dict[str, object] = {}
    for item in args.option:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--option expects KEY=VALUE, got {item!r}")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw

    import cv2

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

    neural = LivePortraitEngine(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        weights=models / DEFAULT_WEIGHTS_RELATIVE_PATH,
    )
    neural.start()
    started = time.perf_counter()
    avatar = repository.get(args.avatar)
    avatar.manifest.engine_options.update(overrides)
    if overrides:
        print(f"engine_options overridden for this run: {overrides}")
    neural.load_avatar(avatar)
    print(f"neural avatar loaded in {time.perf_counter() - started:.1f} s")

    warp: WarpAvatarEngine | None = None
    if not args.no_warp:
        warp = WarpAvatarEngine(model_path=models / DEFAULT_MODEL_RELATIVE_PATH)
        warp.start()
        started = time.perf_counter()
        warp.load_avatar(repository.get(args.warp_avatar))
        print(f"warp avatar loaded in {time.perf_counter() - started:.1f} s")

    capture = cv2.VideoCapture(str(args.clip))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    last = total - 1 if args.last is None else min(args.last, total - 1)
    writer = None
    timings: dict[str, list[float]] = {
        "tracker": [],
        "neural": [],
        "crop": [],
        "model": [],
        "paste": [],
        "mask": [],
        "warp": [],
    }
    seam: dict[str, list[float]] = {"whole": [], "hairline": [], "jaw": []}

    index = 0
    written = 0
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
            t0 = time.perf_counter()
            tracking = tracker.process(frame)
            t1 = time.perf_counter()
            neural_out = neural.animate(args.avatar, tracking, frame)
            t2 = time.perf_counter()
            warp_out = warp.animate(args.warp_avatar, tracking, frame) if warp is not None else None
            t3 = time.perf_counter()

            if index >= args.first:
                timings["tracker"].append((t1 - t0) * 1000.0)
                timings["neural"].append((t2 - t1) * 1000.0)
                for key in ("crop", "model", "paste", "mask"):
                    value = neural.last_timings.get(f"{key}_ms")
                    if value is not None:
                        timings[key].append(value)
                if warp is not None:
                    timings["warp"].append((t3 - t2) * 1000.0)

                # The seam number: the mask the engine actually pasted with
                # this frame, warped into it -- ``loaded.last`` is
                # (face, to_frame, mask), see _frame_mask.
                loaded_avatar = neural._loaded.get(args.avatar)
                state = loaded_avatar.last if loaded_avatar is not None else None
                if state is not None:
                    _, to_frame, frame_mask = state
                    edge = _seam_edge_values(neural_out.data, image.shape[:2], frame_mask, to_frame)
                    for key, value in edge.items():
                        seam[key].append(value)

                panels = [image]
                if warp_out is not None:
                    panels.append(warp_out.data)
                panels.append(neural_out.data)
                height = int(PANEL_WIDTH * image.shape[0] / image.shape[1])
                sheet = np.hstack([cv2.resize(p, (PANEL_WIDTH, height)) for p in panels])
                labels = ["camera"] + (["photographs"] if warp_out is not None else []) + ["neural"]
                for i, label in enumerate(labels):
                    cv2.putText(
                        sheet,
                        label,
                        (i * PANEL_WIDTH + 12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    sheet,
                    f"frame {index}",
                    (12, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
                if writer is None:
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(args.out),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (sheet.shape[1], sheet.shape[0]),
                    )
                writer.write(sheet)
                written += 1

                if index in stills:
                    still_path = args.out.parent / f"{args.out.stem}-f{index}.png"
                    cv2.imwrite(str(still_path), sheet)
                    print(f"  still written: {still_path}")
            index += 1
            if index % 200 == 0:
                print(f"  ... {index} frames", flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        neural.stop()
        if warp is not None:
            warp.stop()
        tracker.stop()

    print(f"\n{written} frames written to {args.out}")
    print("per frame, this machine, this run:")
    for name in ("tracker", "neural", "crop", "model", "paste", "mask", "warp"):
        if timings[name]:
            print(summary(name, timings[name]))
    total_path = [a + b for a, b in zip(timings["tracker"], timings["neural"], strict=True)]
    over = 100.0 * sum(1 for t in total_path if t > 33.2) / max(len(total_path), 1)
    print(summary("tracker+neural", total_path) + f"   over 33.2 ms: {over:.1f}%")

    print("\nseam (mask-edge gradient, output's own Sobel magnitude, unitless):")
    for name in ("whole", "hairline", "jaw"):
        if seam[name]:
            print(seam_summary(name, seam[name]))
    print("\nThis is the owner's face. Delete the output when it has been looked at.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
