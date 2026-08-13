"""B004 — what the virtual camera costs the frame budget.

The pipeline had roughly 15 ms of headroom at the end of day 4 (B003 plus the
warp engine). Publishing spends some of it, and this measures how much, because
a sink that costs 12 ms turns a working 30 fps pipeline into a stuttering one
and would not announce itself — the preview would still look fine.

What it measures
----------------
1.  **``send()`` per frame at the delivery resolution**, against the real
    device, separately for the two pixel formats the pipeline actually produces:

    * **BGR** — what the camera captures and the warp engine composites in, so
      the sink passes the buffer straight through;
    * **RGB** — what the idle placeholder is labelled, which the sink has to
      convert first. Both are measured because the renderer preserves whatever
      format it was handed, so both really do arrive.

2.  **``is_available``**, because the window polls it while the checkbox is
    disabled and a slow answer would be paid on the UI thread.

3.  Optionally the **whole pipeline with publishing on** (``--pipeline``),
    which is the only number that answers "does it still hold 30 fps". That one
    needs the real camera and a face in front of it.

Opening the device makes a virtual camera appear to every application on this
machine for as long as the benchmark runs. That is unavoidable — it is the
thing being measured — but it is brief, and nothing else can be running on the
device at the time. Close OBS Studio first if it has its virtual camera started:
it provides a single instance.

Usage::

    python scripts/benchmark_virtual_camera.py                 # send() only
    python scripts/benchmark_virtual_camera.py --frames 2000
    python scripts/benchmark_virtual_camera.py --pipeline      # + full pipeline
    python scripts/benchmark_virtual_camera.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sha256.app.bootstrap import build_application  # noqa: E402
from sha256.config.loader import load_settings  # noqa: E402
from sha256.config.paths import AppPaths  # noqa: E402
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.metrics import MetricSummary  # noqa: E402
from sha256.core.types import FrameMetadata, RenderedFrame  # noqa: E402
from sha256.performance.metrics import RollingWindow  # noqa: E402
from sha256.virtual_camera.devices import registered_capture_filters  # noqa: E402
from sha256.virtual_camera.pyvirtualcam_sink import PyVirtualCameraSink  # noqa: E402

# The first sends pay for the shared-memory mapping and the first NV12
# conversion buffer. Averaging them in would overstate the steady-state cost;
# dropping them silently would hide a slow first frame that a user does see.
WARMUP_FRAMES = 30


def make_frame(width: int, height: int, pixel_format: PixelFormat) -> RenderedFrame:
    """A frame of real, varied pixels.

    Not a constant fill: the conversion to NV12 inside the backend is data
    independent, but a compressible buffer can still flatter a memory copy, and
    a benchmark that measures memcpy on zeroes is measuring nothing.
    """
    rng = np.random.default_rng(seed=20260813)
    data = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return RenderedFrame(
        data=np.ascontiguousarray(data),
        pixel_format=pixel_format,
        metadata=FrameMetadata(frame_id=0, capture_timestamp=0.0, source_id="benchmark"),
    )


def measure_send(
    sink: PyVirtualCameraSink,
    frame: RenderedFrame,
    frames: int,
) -> MetricSummary:
    window = RollingWindow("send", capacity=frames)

    for index in range(frames + WARMUP_FRAMES):
        start = time.perf_counter()
        sink.send(frame)
        duration = time.perf_counter() - start
        if index >= WARMUP_FRAMES:
            window.add(duration)

    return window.summarise()


def measure_is_available(sink: PyVirtualCameraSink, repeats: int) -> float:
    """Mean milliseconds per call. The window asks this on a timer."""
    start = time.perf_counter()
    for _ in range(repeats):
        sink.is_available  # noqa: B018 - the property is the work being timed
    return (time.perf_counter() - start) * 1000.0 / repeats


def format_row(label: str, summary: MetricSummary) -> str:
    if summary.count == 0:
        return f"| {label} | — | — | — | — | — | 0 |"
    return (
        f"| {label} "
        f"| {summary.mean * 1000:.3f} "
        f"| {summary.p50 * 1000:.3f} "
        f"| {summary.p95 * 1000:.3f} "
        f"| {summary.p99 * 1000:.3f} "
        f"| {summary.maximum * 1000:.3f} "
        f"| {summary.count} |"
    )


def measure_pipeline(seconds: float) -> dict[str, Any]:
    """Full pipeline with publishing on, using the configured backends."""
    paths = AppPaths.resolve()
    settings = load_settings(paths)
    context = build_application(paths=paths, setup_logging=False)

    try:
        controller = context.controller
        controller.start()
        controller.set_publish_output(True)

        deadline = time.perf_counter() + seconds
        frames = 0
        while time.perf_counter() < deadline:
            if controller.process_once() is not None:
                frames += 1

        elapsed = seconds
        sink = controller.components.virtual_camera
        snapshot = context.monitor.snapshot()
        return {
            "camera_backend": settings.camera.backend,
            "engine_backend": settings.engine.backend,
            "avatar": context.avatars.active_id,
            "seconds": elapsed,
            "frames": frames,
            "fps": frames / elapsed,
            "frames_published": getattr(sink, "frames_sent", 0),
            "frames_dropped": getattr(sink, "frames_dropped", 0),
            "output_ms_mean": snapshot.duration("output").mean * 1000.0,
            "output_ms_p95": snapshot.duration("output").p95 * 1000.0,
            "end_to_end_ms_p95": snapshot.duration("end_to_end").p95 * 1000.0,
        }
    finally:
        context.shutdown()


def run(args: argparse.Namespace) -> int:
    paths = AppPaths.resolve()
    settings = load_settings(paths)
    width = settings.rendering.output_width
    height = settings.rendering.output_height
    fps = settings.rendering.target_fps

    print("=" * 72)
    print("B004 — virtual camera output cost")
    print("=" * 72)
    print(f"resolution : {width}x{height} @ {fps:.2f} fps")
    print(f"filters    : {', '.join(registered_capture_filters()) or 'none registered'}")
    print()

    sink = PyVirtualCameraSink(device=settings.virtual_camera.device)
    if not sink.is_available:
        print("No virtual camera device is installed. Install OBS Studio once.")
        return 2

    sink.configure(width, height, fps)
    try:
        sink.start()
    except Exception as exc:
        print(f"Could not open the device: {exc}")
        return 2

    results: dict[str, Any] = {
        "width": width,
        "height": height,
        "fps": fps,
        "device": sink.device_name,
        "frames": args.frames,
    }

    try:
        print(f"device     : {sink.device_name!r}")
        print(f"warmup     : {WARMUP_FRAMES} frames, discarded")
        print()

        rows = []
        for label, pixel_format in (
            ("send, BGR (no conversion)", PixelFormat.BGR24),
            ("send, RGB (converted)", PixelFormat.RGB24),
        ):
            frame = make_frame(width, height, pixel_format)
            summary = measure_send(sink, frame, args.frames)
            rows.append((label, summary))
            results[pixel_format.value] = {
                "mean_ms": summary.mean * 1000.0,
                "p50_ms": summary.p50 * 1000.0,
                "p95_ms": summary.p95 * 1000.0,
                "p99_ms": summary.p99 * 1000.0,
                "max_ms": summary.maximum * 1000.0,
                "count": summary.count,
            }

        print("| stage | mean ms | p50 | p95 | p99 | max | n |")
        print("|---|---|---|---|---|---|---|")
        for label, summary in rows:
            print(format_row(label, summary))
        print()

        assert sink.frames_dropped == 0, f"sink dropped frames: {sink.last_drop_reason}"

        available_ms = measure_is_available(sink, repeats=args.frames)
        results["is_available_ms"] = available_ms
        print(f"is_available(): {available_ms:.3f} ms per call, {args.frames} calls")
        print()
    finally:
        sink.stop()

    if args.pipeline:
        print("-" * 72)
        print("Full pipeline with publishing on")
        print("-" * 72)
        print(">>> Sit in front of the camera. <<<")
        pipeline = measure_pipeline(args.seconds)
        results["pipeline"] = pipeline
        for key, value in pipeline.items():
            formatted = f"{value:.2f}" if isinstance(value, float) else value
            print(f"{key:>20} : {formatted}")
        print()

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=1000, help="Frames per format.")
    parser.add_argument("--pipeline", action="store_true", help="Also run the full pipeline.")
    parser.add_argument("--seconds", type=float, default=20.0, help="Pipeline run length.")
    parser.add_argument("--json", type=str, default=None, help="Write results as JSON.")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
