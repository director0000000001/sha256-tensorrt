"""B003 — measure the MediaPipe face tracker on real camera frames.

Run it with a face in front of the camera. The script says so and reports the
detection rate, because a tracking benchmark taken on an empty room measures the
detector giving up, which is a different and much cheaper operation.

What it measures
----------------
1.  **Per-frame tracking cost** — the number the frame budget cares about: what
    ``FaceTracker.process()`` costs, start to finish, through the same code the
    pipeline runs.
2.  **A breakdown of that cost** — colour conversion, MediaPipe inference, and
    the adaptation into sha256's neutral types — so that if the total is too
    large it is obvious which part to attack.
3.  **Detection rate**, since a fast tracker that loses the face is not fast.
4.  **Head-pose sign verification**, two ways, because an axis-flip otherwise
    only surfaces later as an avatar that leans the wrong way:

    * against the **face normal** computed from the landmarks, which is a
      separate model output from the pose matrix and works on a still head;
    * against **known transforms** of a real frame — a rotation of a known angle
      is a pure roll, and a mirror must flip yaw and roll while leaving pitch.

    An earlier version compared 2-D landmark proxies (nose against eye midpoint,
    chin-to-eye against eye-to-forehead). Those were dropped: they are confounded
    by the *other* axes, and with a head tilted 30 degrees they confidently
    reported the opposite of the truth. See docs/benchmarks/B003-face-tracking.md.

Usage::

    python scripts/benchmark_tracking.py                  # 600 frames, default config
    python scripts/benchmark_tracking.py --seconds 30     # sustained run
    python scripts/benchmark_tracking.py --delegate gpu   # record the GPU result
    python scripts/benchmark_tracking.py --json out.json  # machine-readable copy
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sha256.camera.opencv_capture import OpenCVCameraProvider  # noqa: E402
from sha256.config.loader import load_settings  # noqa: E402
from sha256.config.paths import AppPaths  # noqa: E402
from sha256.core.metrics import MetricSummary  # noqa: E402
from sha256.core.types import FrameMetadata, VideoFrame  # noqa: E402
from sha256.performance.metrics import RollingWindow  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)
from sha256.utils.timing import now  # noqa: E402

# Warm-up frames are discarded rather than averaged in. The first inferences pay
# for lazy allocation and XNNPACK setup; including them would understate steady
# state, and hiding them silently would overstate it. Both numbers are reported.
WARMUP_FRAMES = 30


class Timings:
    """Named sample sets, summarised with the application's own statistics."""

    def __init__(self, capacity: int) -> None:
        self._windows: dict[str, RollingWindow] = {}
        self._capacity = capacity

    def add(self, name: str, seconds: float) -> None:
        window = self._windows.get(name)
        if window is None:
            window = RollingWindow(name, capacity=self._capacity)
            self._windows[name] = window
        window.add(seconds)

    def summary(self, name: str) -> MetricSummary:
        window = self._windows.get(name)
        return window.summarise() if window else MetricSummary.empty(name)

    def names(self) -> list[str]:
        return list(self._windows)


# Canonical MediaPipe face-mesh indices used by the geometric cross-check.
# Named rather than inlined so the check is auditable against the published
# topology instead of being a set of magic numbers.
EYE_OUTER_A, EYE_OUTER_B = 33, 263
FOREHEAD_TOP, CHIN_BOTTOM = 10, 152


def face_normal_signals(points: np.ndarray) -> dict[str, float]:
    """Absolute orientation signals from the landmark cloud's 3-D structure.

    The landmarks and the transformation matrix are *separate* outputs of the
    model, so checking one against the other is evidence rather than a
    tautology. Unlike the movement-based check, this one works on a perfectly
    still head, which is what makes the pose verification repeatable.

    Landmarks arrive as x right, y **down**, z negative toward the camera. Both
    y and z are negated to reach the same right-handed frame the pose matrix
    uses: X right, Y up, Z toward the viewer.

    Returns the face normal's ``x`` and ``y`` components — pointing out of the
    face — and the eye-line angle:

    * ``normal_y`` > 0 => the face points upward => pitch should be positive;
    * ``normal_x`` > 0 => the face points toward the image right, which is the
      subject's left => yaw should be positive;
    * ``eye_line`` > 0 => the image-right eye is higher, so the subject is
      tilted toward their own right => roll should be positive.
    """

    def to_frame(index: int) -> np.ndarray:
        x, y, z = points[index]
        return np.array([float(x), -float(y), -float(z)])

    eye_a, eye_b = to_frame(EYE_OUTER_A), to_frame(EYE_OUTER_B)
    image_left, image_right = (eye_a, eye_b) if eye_a[0] < eye_b[0] else (eye_b, eye_a)

    across = image_right - image_left
    upward = to_frame(FOREHEAD_TOP) - to_frame(CHIN_BOTTOM)

    normal = np.cross(across, upward)
    norm = float(np.linalg.norm(normal))
    if norm > 1e-9:
        normal = normal / norm

    return {
        "normal_x": float(normal[0]),
        "normal_y": float(normal[1]),
        "eye_line": math.degrees(math.atan2(float(across[1]), float(across[0]))),
    }


def sign_agreement(
    reported: list[float], expected: list[float], *, angle_deadzone: float, signal_deadzone: float
) -> dict[str, Any]:
    """Fraction of frames where a reported angle has the sign the geometry implies.

    Frames where either value sits inside its dead zone are skipped: a face that
    is square to the camera has no meaningful sign on that axis, and counting
    coin flips would bury a real inversion.
    """
    pairs = [
        (r, e)
        for r, e in zip(reported, expected, strict=True)
        if abs(r) >= angle_deadzone and abs(e) >= signal_deadzone
    ]
    if not pairs:
        return {"samples": 0, "agree_pct": None}

    agree = sum(1 for r, e in pairs if (r > 0) == (e > 0))
    return {"samples": len(pairs), "agree_pct": 100.0 * agree / len(pairs)}


def transformed(frame: VideoFrame, data: np.ndarray, offset: int) -> VideoFrame:
    """A frame carrying different pixels but a fresh, later timestamp."""
    return VideoFrame(
        data=np.ascontiguousarray(data),
        pixel_format=frame.pixel_format,
        metadata=FrameMetadata(
            frame_id=frame.metadata.frame_id + offset,
            capture_timestamp=frame.metadata.capture_timestamp + 0.05 * offset,
            source_id="synthetic",
        ),
    )


def synthetic_pose_checks(build_tracker: Callable[[], Any], frame: VideoFrame) -> dict[str, Any]:
    """Verify the pose convention by transforming a real frame in known ways.

    The head-movement check needs the subject to actually move. This one does
    not, and it is stronger where it applies, because the ground truth is exact
    rather than inferred:

    * **Rotating the image** is a pure roll of known size. Rotating the picture
      counter-clockwise puts the top of the head toward the image left, and the
      image left is the subject's *right* in an unmirrored camera view — so a
      correct roll convention reports a change of ``+angle``. A slope near -1
      means the axis is inverted; near +1 means it is right.
    * **Mirroring the image** swaps the subject's left and right. Yaw and roll
      must change sign; pitch must not. This catches an axis that is inverted
      *and* one that is silently coupled to another.

    Both are run through the tracker's normal public path, so they test the
    shipped conversion rather than a reimplementation of it.

    **Each transform gets a fresh tracker.** In VIDEO running mode MediaPipe
    carries a prior from the previous frame, and feeding it an abruptly rotated
    image makes it lose the face often enough to put holes in the results. A
    fresh tracker sees each transform as a first frame, which is what the check
    is asking about anyway.
    """
    try:
        import cv2
    except ImportError:
        return {"available": False}

    def pose_of(pixels: np.ndarray, index: int) -> Any:
        tracker = build_tracker()
        tracker.start()
        try:
            result = tracker.process(transformed(frame, pixels, index))
        finally:
            tracker.stop()
        return result.head_pose if result.face_detected else None

    baseline = pose_of(frame.data, 1)
    if baseline is None:
        return {"available": False}

    height, width = frame.data.shape[:2]
    centre = (width / 2.0, height / 2.0)

    rotations: dict[float, float] = {}
    for index, applied in enumerate((-20.0, -10.0, 10.0, 20.0), start=2):
        matrix = cv2.getRotationMatrix2D(centre, applied, 1.0)
        pose = pose_of(cv2.warpAffine(frame.data, matrix, (width, height)), index)
        if pose is not None:
            rotations[applied] = pose.roll - baseline.roll

    slope = None
    if len(rotations) >= 2:
        applied_values = list(rotations)
        observed = [rotations[a] for a in applied_values]
        mean_applied = sum(applied_values) / len(applied_values)
        mean_observed = sum(observed) / len(observed)
        denominator = sum((a - mean_applied) ** 2 for a in applied_values)
        if denominator > 0:
            slope = (
                sum(
                    (a - mean_applied) * (o - mean_observed)
                    for a, o in zip(applied_values, observed, strict=True)
                )
                / denominator
            )

    mirrored = pose_of(cv2.flip(frame.data, 1), 10)
    mirror: dict[str, Any] = {"detected": mirrored is not None}
    if mirrored is not None:
        mirror.update(
            {
                "yaw": (baseline.yaw, mirrored.yaw),
                "roll": (baseline.roll, mirrored.roll),
                "pitch": (baseline.pitch, mirrored.pitch),
            }
        )

    return {
        "available": True,
        "baseline": {
            "pitch": baseline.pitch,
            "yaw": baseline.yaw,
            "roll": baseline.roll,
        },
        "rotation_response": rotations,
        "roll_slope": slope,
        "mirror": mirror,
    }


def measure_conversion_cost(frame: VideoFrame, repeats: int = 50) -> dict[str, float]:
    """Compare the two ways to hand a BGR frame to MediaPipe.

    The tracker uses numpy so that it does not require OpenCV; this measures
    what that choice costs against ``cv2.cvtColor`` on the same frame.
    """
    data = frame.data

    start = time.perf_counter()
    for _ in range(repeats):
        np.ascontiguousarray(data[:, :, ::-1])
    numpy_ms = (time.perf_counter() - start) * 1000.0 / repeats

    try:
        import cv2
    except ImportError:
        return {"numpy_ms": numpy_ms, "cv2_ms": float("nan")}

    start = time.perf_counter()
    for _ in range(repeats):
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    cv2_ms = (time.perf_counter() - start) * 1000.0 / repeats

    return {"numpy_ms": numpy_ms, "cv2_ms": cv2_ms}


def format_row(label: str, summary: MetricSummary) -> str:
    if summary.count == 0:
        return f"| {label} | — | — | — | — | — | 0 |"
    return (
        f"| {label} "
        f"| {summary.mean * 1000:.2f} "
        f"| {summary.p50 * 1000:.2f} "
        f"| {summary.p95 * 1000:.2f} "
        f"| {summary.p99 * 1000:.2f} "
        f"| {summary.maximum * 1000:.2f} "
        f"| {summary.count} |"
    )


def run(args: argparse.Namespace) -> int:
    paths = AppPaths.resolve()
    settings = load_settings(paths)

    model_path = (
        Path(args.model)
        if args.model
        else paths.resolve_relative(settings.models.root_path) / DEFAULT_MODEL_RELATIVE_PATH
    )

    print("=" * 72)
    print("B003 — MediaPipe Face Landmarker on real camera frames")
    print("=" * 72)
    print(f"model     : {model_path}")
    print(f"delegate  : {args.delegate}")
    print(f"camera    : index {settings.camera.device_index} via the opencv backend")
    print()
    print(">>> Sit in front of the camera and move your head: nod, turn, tilt. <<<")
    print(">>> The pose check needs real movement to have anything to verify.  <<<")
    print()

    camera = OpenCVCameraProvider(
        device_index=settings.camera.device_index,
        width=settings.camera.width,
        height=settings.camera.height,
        target_fps=settings.camera.target_fps,
        use_mjpeg=settings.camera.use_mjpeg,
    )

    def build_tracker() -> MediaPipeFaceTracker:
        return MediaPipeFaceTracker(
            model_path=model_path,
            delegate=args.delegate,
            num_faces=settings.tracking.num_faces,
            min_detection_confidence=settings.tracking.min_detection_confidence,
            min_presence_confidence=settings.tracking.min_presence_confidence,
            min_tracking_confidence=settings.tracking.min_tracking_confidence,
            smoothing=args.smoothing,
        )

    tracker = build_tracker()

    camera.start()
    print(f"camera opened: {camera.actual_width}x{camera.actual_height} via {camera.backend_name}")

    start_cost = time.perf_counter()
    try:
        tracker.start()
    except Exception as exc:
        camera.stop()
        print(f"\nTRACKER FAILED TO START with delegate={args.delegate!r}:\n{exc}")
        return 1
    startup_seconds = time.perf_counter() - start_cost
    # Read now, not at reporting time: stop() clears it, and the report is
    # written after the tracker has been shut down.
    active_delegate = tracker.active_delegate
    print(f"tracker ready in {startup_seconds:.2f} s on the {active_delegate} delegate")
    print()

    capacity = max(args.frames, 10_000)
    timings = Timings(capacity)
    warmup = Timings(capacity)

    detected = 0
    processed = 0
    read_failures = 0
    landmark_counts: set[int] = set()
    blendshape_names: tuple[str, ...] = ()
    reported: dict[str, list[float]] = {"pitch": [], "yaw": [], "roll": []}
    normals: dict[str, list[float]] = {"pitch": [], "yaw": [], "roll": []}
    capture_stamps: list[float] = []
    conversion_costs: dict[str, float] = {}
    last_good_frame: VideoFrame | None = None
    synthetic: dict[str, Any] = {"available": False}

    deadline = time.perf_counter() + args.seconds if args.seconds else None
    target_frames = args.frames + WARMUP_FRAMES

    try:
        while True:
            if deadline is not None:
                if time.perf_counter() >= deadline:
                    break
            elif processed >= target_frames:
                break

            frame = camera.read()
            if frame is None:
                read_failures += 1
                continue

            capture_stamps.append(frame.metadata.capture_timestamp)

            # The number that matters: the whole public call, exactly as the
            # pipeline invokes it.
            started = now()
            result = tracker.process(frame)
            total = now() - started

            bucket = warmup if processed < WARMUP_FRAMES else timings
            bucket.add("process_total", total)

            if args.breakdown:
                # Re-runs the same internal steps on the same frame rather than
                # leaving timing hooks in production code for the sake of one
                # benchmark. It processes every frame twice, so it is off by
                # default: the honest per-frame and capture-rate numbers come
                # from a run without it.
                t0 = now()
                image = tracker._to_mediapipe_image(frame)
                convert = now() - t0

                t0 = now()
                raw = tracker._landmarker.detect_for_video(
                    image,
                    tracker._next_timestamp_ms(frame.metadata),
                )
                infer = now() - t0

                t0 = now()
                if raw.face_landmarks:
                    tracker._adapt(raw, frame.metadata, now())
                adapt = now() - t0

                bucket.add("convert", convert)
                bucket.add("inference", infer)
                bucket.add("adapt", adapt)

            processed += 1

            if not result.face_detected:
                continue
            detected += 1

            if processed <= WARMUP_FRAMES:
                continue

            if result.landmarks is not None:
                points = result.landmarks.points
                landmark_counts.add(points.shape[0])
                if result.head_pose is not None and points.shape[0] >= 468:
                    reported["pitch"].append(result.head_pose.pitch)
                    reported["yaw"].append(result.head_pose.yaw)
                    reported["roll"].append(result.head_pose.roll)

                    signals = face_normal_signals(points)
                    normals["pitch"].append(signals["normal_y"])
                    normals["yaw"].append(signals["normal_x"])
                    normals["roll"].append(signals["eye_line"])

            if result.blendshapes is not None and not blendshape_names:
                blendshape_names = result.blendshapes.names

            if not conversion_costs:
                conversion_costs = measure_conversion_cost(frame)

            last_good_frame = frame

            if processed % 60 == 0:
                rate = 100.0 * detected / processed
                print(f"  {processed} frames, {rate:.0f}% with a face...")

        # After the timed loop, so the synthetic frames cannot contaminate the
        # measurements above.
        if last_good_frame is not None:
            synthetic = synthetic_pose_checks(build_tracker, last_good_frame)
    finally:
        tracker.stop()
        camera.stop()

    measured = processed - WARMUP_FRAMES
    if measured <= 0:
        print("\nNot enough frames captured to report anything.")
        return 1

    capture_fps = 0.0
    if len(capture_stamps) > 1:
        span = capture_stamps[-1] - capture_stamps[0]
        capture_fps = (len(capture_stamps) - 1) / span if span > 0 else 0.0

    detection_rate = 100.0 * detected / processed

    print()
    print("-" * 72)
    print(f"Frames captured     : {processed} ({WARMUP_FRAMES} warm-up discarded)")
    print(f"Failed camera reads : {read_failures}")
    print(f"Capture rate        : {capture_fps:.2f} fps")
    print(f"Face detected       : {detected}/{processed}  ({detection_rate:.1f}%)")
    print(f"Landmarks per face  : {sorted(landmark_counts) or 'n/a'}")
    print(f"Blendshapes         : {len(blendshape_names)}")
    print(f"Startup cost        : {startup_seconds * 1000:.0f} ms (excluded from all timings)")
    print()

    print("| stage | mean ms | p50 | p95 | p99 | max | n |")
    print("|---|---|---|---|---|---|---|")
    print(format_row("process() total", timings.summary("process_total")))
    if args.breakdown:
        print(format_row("  colour convert", timings.summary("convert")))
        print(format_row("  mediapipe infer", timings.summary("inference")))
        print(format_row("  adapt to types", timings.summary("adapt")))
        print("(breakdown mode processes every frame twice; capture rate above is not")
        print(" representative of a normal run)")
    print()
    print("Warm-up frames, for comparison:")
    print(format_row("process() total", warmup.summary("process_total")))
    print()

    if conversion_costs:
        cv2_ms = conversion_costs.get("cv2_ms", float("nan"))
        print(
            f"BGR->RGB conversion: numpy {conversion_costs['numpy_ms']:.2f} ms "
            f"vs cv2.cvtColor {cv2_ms:.2f} ms (50 repeats, same frame)"
        )
        print()

    print("Head-pose sign agreement with the face normal")
    print("(absolute check — works on a still head, since the landmarks are a")
    print(" separate model output from the pose matrix)")
    print()
    print("| axis | frames | agree % |")
    print("|---|---|---|")
    sign_checks: dict[str, Any] = {}
    for axis, signal_deadzone in (("pitch", 0.02), ("yaw", 0.02), ("roll", 1.0)):
        check = sign_agreement(
            reported[axis],
            normals[axis],
            angle_deadzone=1.0,
            signal_deadzone=signal_deadzone,
        )
        sign_checks[axis] = check
        if check["agree_pct"] is None:
            print(f"| {axis} | 0 | too close to neutral |")
        else:
            print(f"| {axis} | {check['samples']} | {check['agree_pct']:.1f}% |")
    print()

    print("Agreement near 100% means the sign convention is right; near 0% means")
    print("that axis is inverted.")
    print()
    print("| axis | resting angle |")
    print("|---|---|")
    for axis in ("pitch", "yaw", "roll"):
        values = reported[axis]
        resting = f"{statistics.median(values):+.1f} deg" if values else "—"
        print(f"| {axis} | {resting} |")
    print()

    if synthetic.get("available"):
        print("Synthetic check — a real frame transformed in known ways")
        print()
        base = synthetic["baseline"]
        print(
            f"baseline pose: pitch {base['pitch']:+.1f}  "
            f"yaw {base['yaw']:+.1f}  roll {base['roll']:+.1f}"
        )
        print()
        print("| image rotated by | roll change |")
        print("|---|---|")
        for applied, observed in synthetic["rotation_response"].items():
            print(f"| {applied:+.0f} deg | {observed:+.1f} deg |")
        slope = synthetic["roll_slope"]
        if slope is not None:
            verdict = "correct" if slope > 0.5 else "INVERTED" if slope < -0.5 else "inconclusive"
            print(f"\nroll response slope: {slope:+.2f}  ->  roll sign is {verdict}")
        print()

        mirror = synthetic["mirror"]
        if "yaw" in mirror:
            print("| axis | original | mirrored | expected |")
            print("|---|---|---|---|")
            for axis, expectation in (
                ("yaw", "sign flips"),
                ("roll", "sign flips"),
                ("pitch", "unchanged"),
            ):
                original, flipped = mirror[axis]
                print(f"| {axis} | {original:+.1f} | {flipped:+.1f} | {expectation} |")
        print()

    budget_ms = 1000.0 / settings.camera.target_fps
    tracking_p95_ms = timings.summary("process_total").p95 * 1000.0
    print(
        f"Frame budget at {settings.camera.target_fps:.0f} fps: {budget_ms:.1f} ms — "
        f"tracking p95 uses {100.0 * tracking_p95_ms / budget_ms:.0f}% of it."
    )

    if args.json:
        payload = {
            "frames_captured": processed,
            "warmup_frames": WARMUP_FRAMES,
            "failed_reads": read_failures,
            "capture_fps": capture_fps,
            "detection_rate_pct": detection_rate,
            "landmark_counts": sorted(landmark_counts),
            "blendshape_count": len(blendshape_names),
            "blendshape_names": list(blendshape_names),
            "startup_seconds": startup_seconds,
            "delegate": active_delegate,
            "smoothing": args.smoothing,
            "resolution": [camera.actual_width, camera.actual_height],
            "conversion_ms": conversion_costs,
            "pose_sign_checks": sign_checks,
            "synthetic_pose_checks": synthetic,
            "timings_ms": {
                name: {
                    "mean": timings.summary(name).mean * 1000,
                    "p50": timings.summary(name).p50 * 1000,
                    "p95": timings.summary(name).p95 * 1000,
                    "p99": timings.summary(name).p99 * 1000,
                    "min": timings.summary(name).minimum * 1000,
                    "max": timings.summary(name).maximum * 1000,
                    "count": timings.summary(name).count,
                }
                for name in timings.names()
            },
        }
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        # default=float: numpy scalars leak in from the landmark maths, and a
        # benchmark that computes everything and then fails to write it out is
        # the most annoying possible failure.
        target.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
        print(f"\nWrote {target}")

    if detection_rate < 50.0:
        print(
            "\nWARNING: a face was found in under half the frames. These timings "
            "describe a detector that is mostly failing, not a tracker at work."
        )
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=600, help="Frames to measure.")
    parser.add_argument("--seconds", type=float, default=0.0, help="Run for a duration instead.")
    parser.add_argument("--delegate", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Also time convert/infer/adapt separately. Processes each frame twice.",
    )
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--model", type=str, default="", help="Override the model bundle path.")
    parser.add_argument("--json", type=str, default="", help="Also write results as JSON.")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
