"""The quality measure: one clip, fixed frames, numbers that carry their conditions.

Every quality claim made about this engine since day 8 has been a number taken
by hand, in passing, from a script that was deleted afterwards. Twice on day 11
one of those reached a commit message and had to be corrected the next day —
once because the frame cost was taken while the machine was busy with something
else, and once because a figure measured for one region was reported for
another. Neither was a lie, and neither was catchable by rereading the diff.

So this is the committed version, and it is built around the two rules that
would have caught both:

*   **Nothing is reported without its conditions.** Which commit, which clip,
    which avatar library, which frames, and how fast the machine was running
    while the numbers were taken. `docs/benchmarks/PERFORMANCE_PLAN.md` section
    7 lists what a result must state; this prints all of it, every run.
*   **Two runs are compared by the tool, not by eye.** ``--baseline`` refuses
    the comparison outright when the clip, the library or the frame range
    differ, and reports the frame cost as *not comparable* when the machine's
    own speed moved between the two runs.

**Machine speed is measured rather than assumed.** A fixed OpenCV workload is
timed before and after the clip. On this machine, idle, it runs 0.395 ms and
repeats within about 1%; with 24 other processes competing for the CPU it runs
0.650 ms. That probe is the difference between "21.4 ms" and "21.4 ms on a
machine that was running at two thirds speed".

**And it found something on its first use.** A full-clip run warms this CPU by
about half — 0.392 ms before, 0.590 ms after, reproducibly, and still 0.60 ms
after twenty seconds of rest. It is the measurement slowing the machine it is
measuring. That is not noise to be filtered out: it is the same slowdown a real
call produces after a minute, and it is why the frame cost is reported for the
first and last tenth of the clip separately rather than as one average. What is
refused is a *comparison* between two runs that began in different states — see
:func:`compare`.

**What it measures**, on every frame in the chosen range:

*   **warp residual** — the mean pixels left after a rigid fit of the blended
    mesh onto the tracked face. It is how hard the warp has to work, which is
    how much the picture is being stretched.
*   **near-eye sharpness** — Laplacian variance over the eye nearer the camera,
    on the *rendered* frame. The whole-face measure cannot judge a turn: a
    turned face shows less detail whether or not it is sharp.
*   **face sharpness** — the whole-face variance, for continuity with the
    numbers already written in the backlog.
*   **frame cost** — wall time of ``animate`` alone, mean and percentiles.
*   **photographs** — which source was on screen, on how many frames, and at
    what share.

Reported for all frames, for frontal ones and for turned ones separately,
because a change that helps a turn and costs nothing frontally is invisible in
a single average — which is exactly what the turn fix on day 8 nearly was.

**What it deliberately does not do.** It does not combine these into one score:
a residual in pixels and a Laplacian variance do not add up to anything, and a
weighted total would look like a measurement while being an opinion. And it
renders no pictures — `scripts/replay_clip.py` is the contact sheet, and the two
answer different questions. Look at both.

Usage::

    python scripts/measure_quality.py --label "before the lid fix"
    python scripts/measure_quality.py --segment "turn your head"
    python scripts/measure_quality.py --baseline recordings/quality/before.json
    python scripts/measure_quality.py --compare before.json after.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from extract_avatar_frames import near_eye_sharpness, sharpness_of  # noqa: E402

from sha256.avatar.engine_warp import (  # noqa: E402
    _BROWS_PREFIX,
    _EYES_PREFIX,
    _LADDER_PREFIX,
    _SMILE_PREFIX,
    _WEIGHT_QUANTUM,
    WarpAvatarEngine,
    _pixel_points,
    _region_of,
)
from sha256.avatar.repository import FilesystemAvatarRepository  # noqa: E402
from sha256.config.loader import load_settings  # noqa: E402
from sha256.config.paths import AppPaths  # noqa: E402
from sha256.core.avatar import AvatarDescriptor  # noqa: E402
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FaceTrackingResult, FrameMetadata, VideoFrame  # noqa: E402
from sha256.performance.metrics import _percentile  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)

# The report's own version. Bumped when a field changes meaning, so that a
# baseline written by an older tool is refused rather than silently misread.
REPORT_FORMAT = 1

# Where a run is written unless ``--out`` says otherwise. Under `recordings/`
# because that directory is gitignored: CONTRIBUTING forbids committing result
# files, and what does get committed is the summary a human writes into
# PROJECT_STATUS from these numbers.
DEFAULT_OUT_DIR = REPO_ROOT / "recordings" / "quality"

# Frames discarded before anything is measured.
#
# The engine is stateful in a way that matters here: it estimates the driver's
# resting face from the frames it has seen, and until that estimate settles the
# expressions it chooses are wrong — backlog 27 is an entire defect made of the
# first second of a run. One second at 30 fps, and the recorded clip opens with
# a "sit still" segment, so the discarded frames carry the least anyway.
DEFAULT_WARMUP_FRAMES = 30

# Where "frontal" stops and "turned" begins, in degrees of yaw. Ten because
# that is the angle the turn fix was reported at (backlog 20), and keeping it
# means today's numbers can be read next to that table.
TURNED_DEGREES = 10.0

# The mesh points the residual is measured over: the face mesh without the two
# iris fans. The irises are positioned by gaze logic rather than carried by the
# warp — see ``_place_irises`` — so including them would count another
# subsystem's decisions as warp error.
FACE_MESH_POINTS = 468

# --- the machine-speed probe -----------------------------------------------
#
# A fixed workload, chosen to look like the real one: a face-sized OpenCV warp,
# which is what ``animate`` spends most of its time doing. Measured on this
# machine (i7-14700KF, OpenCV using 28 threads):
#
#   idle                    median 0.395 ms, run to run within ~1%, spread 1.03-1.06
#   24 competing processes  median 0.650 ms (+64%), spread 1.16
#
# The spread inside one probe barely moves under steady load — a busy machine is
# *uniformly* slower rather than more erratic — which is why the number that
# actually catches a loaded run is the median compared against the baseline's,
# not the spread.
_PROBE_PIXELS = 400
_PROBE_REPEATS = 40
_PROBE_SAMPLES = 12
_PROBE_WARMUP = 2

# How far apart two runs may *start* before their frame costs stop being
# comparable. Five percent, against the 2.9% spread of the starting probe over
# four runs on this machine (0.381, 0.386, 0.386, 0.392 ms) — just outside its
# own noise, and far inside the 64% a machine with other work on it produces.
MACHINE_DRIFT_LIMIT = 0.05

# How erratic one probe may be — slowest sample over fastest — before the run is
# called noisy. Idle measures 1.03 to 1.06.
MACHINE_NOISE_LIMIT = 1.25

# How differently two runs may warm the machine before their frame costs stop
# being comparable, as a difference in fractions. Fifteen points: a full-clip
# run on this machine warms it by about 50% every time — two such runs are
# comparable — while a two-second run barely warms it at all, and its frame cost
# is not the same measurement as a full clip's.
MACHINE_WARMING_DIFFERENCE = 0.15

# The conditions that must match before two runs may be compared at all: the
# ones that define *what was measured*. A different clip or a different frame
# range means the two runs asked different questions, and the arrow between
# their answers says nothing.
#
# The photograph library is deliberately **not** in this list, and that is a
# correction. It was, on the reasoning that a rebuilt ladder moves every number
# — which is true, and is exactly why the comparison has to be allowed: most of
# the work on this engine *is* rebuilding a ladder, and a tool that refuses to
# measure it cannot report the days it matters most. A library change is
# announced instead, at the top of the comparison, so nobody reads the arrow as
# a code change.
COMPARABLE: tuple[tuple[str, str], ...] = (
    ("clip", "fingerprint"),
    ("clip", "frames"),
    ("sampling", "first"),
    ("sampling", "last"),
    ("sampling", "warmup"),
)

# And the conditions that must match before the *frame cost* may be compared.
# The geometry and detail numbers are deterministic — the same frames through
# the same code give the same pixels on any machine — so they survive a
# comparison across machines. A duration does not.
COMPARABLE_TIMINGS: tuple[tuple[str, str], ...] = (
    ("machine", "platform"),
    ("machine", "processor"),
    ("machine", "cv2_threads"),
    ("libraries", "cv2"),
)


@dataclass(frozen=True)
class Spread:
    """One measurement over many frames, with its unit and its sample count.

    A separate type from :class:`sha256.core.metrics.MetricSummary` for one
    reason: that one documents itself as seconds and offers ``mean_ms``, and
    most of what is measured here is pixels and Laplacian variance. Carrying the
    unit explicitly is what stops a number being read as the wrong thing, which
    is the mistake this whole tool exists to prevent.
    """

    name: str
    unit: str
    count: int
    mean: float
    minimum: float
    maximum: float
    p50: float
    p95: float
    p99: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "count": self.count,
            "mean": round(self.mean, 4),
            "min": round(self.minimum, 4),
            "max": round(self.maximum, 4),
            "p50": round(self.p50, 4),
            "p95": round(self.p95, 4),
            "p99": round(self.p99, 4),
        }


@dataclass(frozen=True)
class Frame:
    """Everything measured on one rendered frame."""

    index: int
    yaw: float
    residual_px: float | None
    sharpness_face: float
    sharpness_near_eye: float
    face_width_px: float
    animate_ms: float
    base_mesh: bool
    roles: dict[str, float] = field(default_factory=dict)


def spread_of(name: str, unit: str, values: list[float]) -> Spread:
    """Summarise a column of per-frame values.

    Nearest-rank percentiles, using the application's own percentile function
    rather than a second implementation: a p95 printed here and a p95 in the
    debug panel then mean the same thing, and every value reported is one that
    actually occurred.
    """
    if not values:
        return Spread(name, unit, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ordered = sorted(values)
    return Spread(
        name=name,
        unit=unit,
        count=len(ordered),
        mean=sum(ordered) / len(ordered),
        minimum=ordered[0],
        maximum=ordered[-1],
        p50=_percentile(ordered, 0.50),
        p95=_percentile(ordered, 0.95),
        p99=_percentile(ordered, 0.99),
    )


def machine_speed() -> dict[str, float]:
    """Time a fixed workload, so the frame cost can be read in context.

    Returns the fastest and the median sample, and the spread between fastest
    and slowest. The fastest is what this machine can do when nothing interrupts
    it; the median is what it was actually doing during this run.
    """
    import cv2

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (_PROBE_PIXELS, _PROBE_PIXELS, 3), dtype=np.uint8)
    matrix = np.array([[1.02, 0.03, 1.5], [-0.03, 1.02, -2.0]], dtype=np.float32)

    samples: list[float] = []
    for sample in range(_PROBE_SAMPLES + _PROBE_WARMUP):
        started = time.perf_counter()
        for _ in range(_PROBE_REPEATS):
            warped = cv2.warpAffine(
                image, matrix, (_PROBE_PIXELS, _PROBE_PIXELS), flags=cv2.INTER_LINEAR
            )
            float(warped.mean())
        elapsed = (time.perf_counter() - started) * 1000.0 / _PROBE_REPEATS
        # The first samples pay for OpenCV's thread pool and the allocation,
        # which is a startup cost rather than this machine's speed.
        if sample >= _PROBE_WARMUP:
            samples.append(elapsed)

    ordered = sorted(samples)
    return {
        "fastest_ms": round(ordered[0], 4),
        "median_ms": round(_percentile(ordered, 0.50), 4),
        "spread": round(ordered[-1] / ordered[0], 3),
    }


def _command(*argv: str) -> str | None:
    """Run a short command and return its output, or ``None`` if it cannot.

    ``None`` means *not measured*, which the report keeps distinct from a
    measured value — CONTRIBUTING rule 6.
    """
    if shutil.which(argv[0]) is None:
        return None
    try:
        finished = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if finished.returncode != 0:
        return None
    return finished.stdout.strip()


def code_state() -> dict[str, Any]:
    """Which commit produced these numbers, and whether it was the whole story.

    A dirty tree is recorded rather than refused: measuring an uncommitted
    change is the normal case — it is how a change is judged before it is
    committed. What must not happen is the number reaching a commit message
    without anyone knowing it came from a tree that no longer exists.
    """
    commit = _command("git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD")
    status = _command("git", "-C", str(REPO_ROOT), "status", "--porcelain")
    changed = [line for line in (status or "").splitlines() if line.strip()]
    return {
        "commit": commit,
        "dirty": bool(changed) if status is not None else None,
        "changed_files": len(changed) if status is not None else None,
    }


def gpu_state() -> dict[str, str | None]:
    """The GPU and its driver, if there is one to ask.

    Neither the warp nor the tracker uses the GPU today, so this is context
    rather than a measurement — but the benchmark rules ask every result to name
    the driver, and the neural refiner will make it load-bearing.
    """
    output = _command("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader")
    if not output:
        return {"name": None, "driver": None}
    fields = [value.strip() for value in output.splitlines()[0].split(",")]
    if len(fields) < 2:
        return {"name": None, "driver": None}
    return {"name": fields[0], "driver": fields[1]}


def library_versions() -> dict[str, str | None]:
    """What was actually imported, by version. A different OpenCV is a different
    warp, and a different MediaPipe is a different set of landmarks."""
    import cv2

    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cv2": cv2.__version__,
        "mediapipe": None,
    }
    try:
        import mediapipe
    except ImportError:
        return versions
    versions["mediapipe"] = getattr(mediapipe, "__version__", None)
    return versions


def clip_fingerprint(path: Path) -> str:
    """A cheap identity for a video file: its size, and both of its ends.

    The recorded clip is 834 MB and hashing all of it would cost seconds on
    every run. The head and the tail plus the exact byte count identify a file
    well enough for this purpose — what it has to catch is *a different clip*,
    not a crafted collision.
    """
    digest = hashlib.sha256()
    size = path.stat().st_size
    digest.update(str(size).encode())
    window = 4 * 1024 * 1024
    with path.open("rb") as handle:
        digest.update(handle.read(window))
        if size > window:
            handle.seek(max(size - window, 0))
            digest.update(handle.read(window))
    return digest.hexdigest()[:12]


def library_fingerprint(descriptor: AvatarDescriptor) -> str:
    """An identity for the photograph library, which is half of what is measured.

    A rebuilt ladder changes every number in this report while the code and the
    clip stay identical — that is the whole of day 9 — so a comparison that
    ignores the library is comparing two different avatars.

    Built from the manifest's bytes plus each asset's name, size and
    modification time. Not from the pixels: the library is 2.1 GB across 86
    files and hashing it would cost more than the measurement. The consequence
    is stated rather than hidden — a file replaced by one of exactly the same
    size at the same timestamp reads as unchanged, and a byte-identical file
    recopied reads as changed. The second direction is the safe one.
    """
    digest = hashlib.sha256()
    digest.update((descriptor.root / "metadata.json").read_bytes())
    for asset in sorted(descriptor.manifest.assets, key=lambda ref: ref.path):
        path = descriptor.asset_path(asset)
        if not path.is_file():
            digest.update(f"{asset.path}:missing".encode())
            continue
        stat = path.stat()
        digest.update(f"{asset.path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:12]


def clip_segments(clip: Path) -> list[dict[str, Any]]:
    """The instructions the driving clip was recorded against, if it has any.

    `record_driving_clip.py` writes a sidecar naming what the owner was asked to
    do over which frames. It is what makes ``--segment "turn your head"``
    possible, and a measurement of the turn alone is worth far more than the
    same number diluted across a clip that is 97% frontal.
    """
    sidecar = clip.with_suffix(".json")
    if not sidecar.is_file():
        return []
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    segments = payload.get("segments")
    return segments if isinstance(segments, list) else []


def resolve_range(
    args: argparse.Namespace, frames: int, segments: list[dict[str, Any]]
) -> tuple[int, int] | str:
    """The first and last frame to measure, or a sentence saying why not.

    ``--warmup`` wins over everything: no frame before it is measured whatever
    else was asked for, because the engine has not settled yet and a number
    taken there describes the warm-up rather than the change.
    """
    first, last = 0, frames - 1
    if args.segment:
        wanted = args.segment.lower()
        matches = [
            segment for segment in segments if wanted in str(segment.get("instruction", "")).lower()
        ]
        if not matches:
            listed = "\n".join(f"    {seg.get('instruction')}" for seg in segments)
            return f"No segment matches {args.segment!r}. This clip has:\n{listed}"
        if len(matches) > 1:
            listed = "\n".join(f"    {seg.get('instruction')}" for seg in matches)
            return f"{args.segment!r} matches more than one segment:\n{listed}"
        first = int(matches[0]["first_frame"])
        last = int(matches[0]["last_frame"])
    if args.first is not None:
        first = max(first, args.first)
    if args.last is not None:
        last = min(last, args.last)
    first = max(first, args.warmup)
    if last < first:
        return f"Nothing left to measure: frames {first} to {last}."
    return first, last


def loaded_avatar(engine: WarpAvatarEngine, avatar_id: str) -> Any:
    """The engine's own state for this avatar, reached into on purpose.

    Three of the numbers here — the residual, which photograph was on screen,
    and whether any was — exist only inside the engine, and it publishes none of
    them. Instrumenting the engine to hand them out would put diagnostic code on
    the hot path for the sake of a tool that runs offline; monkeypatching it
    would mean the tool measures something the application does not do. So it
    reads the state the frame just left behind, and the reach-in is written down
    here rather than scattered.

    What it depends on, so that a rename breaks loudly instead of quietly: the
    private ``_loaded`` map, and on the avatar ``blend_points``, ``points``,
    ``holding`` and ``cached_weights``.
    """
    return engine._loaded[avatar_id]


def rigid_residual(source: np.ndarray, target: np.ndarray) -> float | None:
    """How far the blended mesh sits from the tracked face after a rigid fit.

    Rotation, uniform scale and translation are free — the head being nearer or
    turned is not warp error. What is left after taking them out is the part the
    warp has to close by *deforming*, and deforming is what stretches a
    photograph. Fitted the same way the engine fits its own backdrop layer
    (``_rigid_layer``), so the number describes the same placement the renderer
    made rather than a second opinion about it.

    ``None`` when the fit fails, which an extreme angle genuinely can cause.
    """
    import cv2

    points = min(FACE_MESH_POINTS, len(source), len(target))
    if points < 3:
        return None
    origin = np.ascontiguousarray(source[:points, :2], dtype=np.float32)
    destination = np.ascontiguousarray(target[:points, :2], dtype=np.float32)
    # Typed as Any deliberately: OpenCV's stub says this returns a matrix, but
    # it returns None when the fit fails.
    matrix: Any = cv2.estimateAffinePartial2D(origin, destination, method=cv2.LMEDS)[0]
    if matrix is None:
        return None
    moved = cv2.transform(origin.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    return float(np.linalg.norm(moved - destination, axis=1).mean())


def measure_clip(
    clip: Path,
    engine: WarpAvatarEngine,
    tracker: MediaPipeFaceTracker,
    avatar_id: str,
    fps: float,
    first: int,
    last: int,
) -> tuple[list[Frame], dict[str, int]]:
    """Render the clip and measure the chosen frames.

    Every frame from the start of the file is fed to the tracker and the engine,
    including the ones that are not measured. Both are stateful — the tracker
    carries a prior, the engine learns the resting face and eases expressions in
    over time — so seeking to frame 1342 would measure a different engine from
    the one that runs live.
    """
    import cv2

    measured: list[Frame] = []
    counts = {"read": 0, "measured": 0, "no_face": 0, "passthrough": 0, "fit_failed": 0}

    capture = cv2.VideoCapture(str(clip))
    try:
        index = 0
        while index <= last:
            ok, image = capture.read()
            if not ok:
                break
            counts["read"] += 1
            frame = VideoFrame(
                data=np.ascontiguousarray(image),
                pixel_format=PixelFormat.BGR24,
                metadata=FrameMetadata(frame_id=index, capture_timestamp=index / fps),
            )
            result = tracker.process(frame)

            started = time.perf_counter()
            output = engine.animate(avatar_id, result, frame)
            animate_ms = (time.perf_counter() - started) * 1000.0

            if index >= first:
                taken = _measure_frame(
                    engine,
                    avatar_id,
                    index,
                    result,
                    output.data,
                    output.is_placeholder,
                    animate_ms,
                    image.shape[1],
                    image.shape[0],
                )
                if taken is None:
                    counts["passthrough" if output.is_placeholder else "no_face"] += 1
                else:
                    if taken.residual_px is None:
                        counts["fit_failed"] += 1
                    measured.append(taken)
                    counts["measured"] += 1

            index += 1
            if index % 200 == 0:
                print(f"  ... {index} frames", flush=True)
    finally:
        capture.release()

    return measured, counts


def _measure_frame(
    engine: WarpAvatarEngine,
    avatar_id: str,
    index: int,
    result: FaceTrackingResult,
    rendered: np.ndarray,
    is_placeholder: bool,
    animate_ms: float,
    width: int,
    height: int,
) -> Frame | None:
    """One frame's numbers, or ``None`` if nothing was drawn on it."""
    if is_placeholder or result.landmarks is None:
        return None
    target = _pixel_points(result.landmarks.points, width, height)
    if target is None:
        return None

    avatar = loaded_avatar(engine, avatar_id)

    # Whether any photograph was mixed in at all. The engine sets the blend's
    # mesh back to the portrait's own when no source scored anything, so this
    # identity check is the honest reading of "nothing but the base portrait was
    # on screen" — and on a frame like that there are no weights to report.
    base_mesh = avatar.blend_points is avatar.points
    roles: dict[str, float] = {}
    if not base_mesh:
        roles = {
            role: round(quantised * _WEIGHT_QUANTUM, 3)
            for role, quantised in avatar.cached_weights.items()
            if quantised
        }

    pose = result.head_pose
    return Frame(
        index=index,
        yaw=float(pose.yaw) if pose is not None else 0.0,
        residual_px=rigid_residual(avatar.blend_points, target),
        sharpness_face=sharpness_of(rendered, result.landmarks.points),
        sharpness_near_eye=near_eye_sharpness(rendered, result.landmarks.points),
        face_width_px=float(target[:, 0].max() - target[:, 0].min()),
        animate_ms=animate_ms,
        base_mesh=base_mesh,
        roles=roles,
    )


def _by_angle(frames: list[Frame]) -> dict[str, list[Frame]]:
    """The three groups every measurement is reported over."""
    return {
        "all": frames,
        "frontal": [frame for frame in frames if abs(frame.yaw) <= TURNED_DEGREES],
        "turned": [frame for frame in frames if abs(frame.yaw) > TURNED_DEGREES],
    }


def _metric(name: str, unit: str, frames: list[Frame], read: Any) -> dict[str, Any]:
    """One measurement, summarised for all frames, frontal ones and turned ones."""
    groups = _by_angle(frames)
    return {
        group: spread_of(
            name, unit, [value for value in map(read, members) if value is not None]
        ).as_dict()
        for group, members in groups.items()
    }


def rung_labels(descriptor: AvatarDescriptor) -> dict[str, dict[str, Any]]:
    """What each ladder rung actually is, keyed by the name the engine gives it.

    The engine names a rung after its **position in the manifest** —
    ``_align_axis`` builds ``f"{prefix}{index}"`` — so ``eyes@78`` is the
    seventy-eighth asset, not an eye 78% shut. A table of those keys is
    unreadable at best and quietly misleading at worst, which is the failure
    this tool exists to stop. Every rung is therefore reported with the axis
    value it was measured at and the file it came from.
    """
    axes = {
        _LADDER_PREFIX: ("yaw", "degrees"),
        _SMILE_PREFIX: ("smile", "amount"),
        _EYES_PREFIX: ("eyes", "amount"),
        _BROWS_PREFIX: ("brows", "amount"),
    }
    labels: dict[str, dict[str, Any]] = {}
    for index, asset in enumerate(descriptor.manifest.assets):
        for prefix, (attribute, unit) in axes.items():
            value = getattr(asset, attribute, None)
            if value is None:
                continue
            labels[f"{prefix}{index}"] = {
                "axis": attribute,
                "unit": unit,
                "value": float(value),
                "file": asset.path,
            }
    return labels


def photographs(frames: list[Frame], labels: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Which sources were on screen, on how many frames, and at what share.

    The table day 8 built by hand and item 4 of the code list asks to keep: it
    is how a photograph that is active on 84% of an ordinary conversation was
    found, and how the next one will be.
    """
    shares: dict[str, list[float]] = {}
    for frame in frames:
        for role, share in frame.roles.items():
            shares.setdefault(role, []).append(share)

    rows = []
    for role, values in shares.items():
        ordered = sorted(values)
        rung = labels.get(role)
        rows.append(
            {
                "role": role,
                "region": _region_of(role) or "whole head",
                "frames": len(values),
                "percent": round(100.0 * len(values) / max(len(frames), 1), 1),
                "median_share": round(_percentile(ordered, 0.50), 3),
                "rung": None if rung is None else f"{rung['axis']} {rung['value']:+.3f}",
                "file": None if rung is None else rung["file"],
            }
        )
    return sorted(rows, key=lambda row: row["frames"], reverse=True)


def machine_verdict(before: dict[str, float], after: dict[str, float]) -> dict[str, Any]:
    """What the machine was doing while the frame cost was taken.

    Geometry and detail are unaffected by a busy machine — the same frames
    through the same code give the same pixels, and two runs an hour apart
    produced identical residuals to the last decimal — so only the durations are
    in question here.

    **The run warms the machine, and this had to be measured before it could be
    interpreted.** The first version of this treated any drift between the two
    probes as contention and refused the frame cost. It refused every full-clip
    run, always by about half. Measured directly: probe, then sixty seconds of
    many-threaded OpenCV warping, then probe again — 0.392 ms to 0.572 ms, and
    still 0.601 ms after twenty seconds of rest. The measurement is slowing the
    machine it is measuring, and twenty seconds is not enough to undo it.

    So the drift is *reported* rather than refused: it is the same slowdown a
    real call produces after a minute, which makes it part of the answer rather
    than noise in it. What is refused is a *comparison* between two runs that
    started from different machine states — see :func:`compare`.
    """
    warmed = (after["median_ms"] - before["median_ms"]) / max(before["median_ms"], 1e-9)
    erratic = max(before["spread"], after["spread"]) > MACHINE_NOISE_LIMIT
    notes = []
    if warmed > MACHINE_DRIFT_LIMIT:
        notes.append(
            f"the machine ran {warmed * 100:.0f}% slower by the end of the run — "
            "sustained load, not another program"
        )
    elif warmed < -MACHINE_DRIFT_LIMIT:
        notes.append(
            f"the machine ran {-warmed * 100:.0f}% faster by the end of the run, "
            "so something else was competing at the start"
        )
    if erratic:
        notes.append(
            f"the probe varied by {max(before['spread'], after['spread']):.2f}x within "
            f"itself (idle is 1.03-1.06), so something else was running"
        )
    return {
        "before": before,
        "after": after,
        "warmed": round(warmed, 4),
        "erratic": erratic,
        "notes": notes,
    }


def frame_cost_drift(frames: list[Frame]) -> dict[str, Any]:
    """The frame cost at the start of the run against the frame cost at the end.

    The machine-speed probe says the machine slows under sustained load; this
    says whether ``animate`` slowed with it. It is the only part of this report
    that measures the engine *over time*, and scenario S5 of the performance
    plan — thirty minutes sustained, for exactly this reason — has never been
    run. A tenth of the clip at each end: long enough to average, short enough
    that the two ends are genuinely different thermal states.
    """
    tenth = max(len(frames) // 10, 1)
    start = [frame.animate_ms for frame in frames[:tenth]]
    end = [frame.animate_ms for frame in frames[-tenth:]]
    return {
        "frames_each_end": tenth,
        "start": spread_of("animate, first tenth", "ms", start).as_dict(),
        "end": spread_of("animate, last tenth", "ms", end).as_dict(),
    }


def build_report(
    args: argparse.Namespace,
    descriptor: AvatarDescriptor,
    clip: Path,
    clip_facts: dict[str, Any],
    first: int,
    last: int,
    frames: list[Frame],
    counts: dict[str, int],
    machine: dict[str, Any],
) -> dict[str, Any]:
    """Everything measured plus everything needed to read it, as one document."""
    import cv2

    return {
        "tool": "measure_quality",
        "format": REPORT_FORMAT,
        "run": {
            "at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "label": args.label,
        },
        "conditions": {
            "code": code_state(),
            "clip": {
                "path": str(clip),
                "fingerprint": clip_fingerprint(clip),
                **clip_facts,
            },
            "avatar": {
                "id": descriptor.id,
                "manifest_version": descriptor.manifest.version,
                "assets": len(descriptor.manifest.assets),
                "fingerprint": library_fingerprint(descriptor),
            },
            "sampling": {
                "first": first,
                "last": last,
                "warmup": args.warmup,
                "segment": args.segment,
            },
            "machine": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "logical_cpus": None if (count := __import__("os").cpu_count()) is None else count,
                "cv2_threads": cv2.getNumThreads(),
                "speed": machine,
            },
            "libraries": library_versions(),
            "gpu": gpu_state(),
        },
        "counts": counts,
        "metrics": {
            "residual_px": _metric("warp residual", "px", frames, lambda f: f.residual_px),
            "sharpness_near_eye": _metric(
                "near-eye sharpness", "laplacian variance", frames, lambda f: f.sharpness_near_eye
            ),
            "sharpness_face": _metric(
                "face sharpness", "laplacian variance", frames, lambda f: f.sharpness_face
            ),
            "face_width_px": _metric("face width", "px", frames, lambda f: f.face_width_px),
            "animate_ms": _metric("animate", "ms", frames, lambda f: f.animate_ms),
        },
        "frame_cost_drift": frame_cost_drift(frames),
        "base_mesh_frames": sum(1 for frame in frames if frame.base_mesh),
        "photographs": photographs(frames, rung_labels(descriptor)),
    }


# --- printing ---------------------------------------------------------------

_COLUMNS = f"{'':<30}{'count':>7}{'mean':>10}{'p50':>10}{'p95':>10}{'min':>10}{'max':>10}"


def _row(label: str, entry: dict[str, Any], places: int = 2) -> str:
    return (
        f"  {label:<28}{entry['count']:>7}"
        f"{entry['mean']:>10.{places}f}{entry['p50']:>10.{places}f}"
        f"{entry['p95']:>10.{places}f}{entry['min']:>10.{places}f}{entry['max']:>10.{places}f}"
    )


def print_report(report: dict[str, Any]) -> None:
    """The whole result, in the order it should be read: conditions first."""
    conditions = report["conditions"]
    code = conditions["code"]
    clip = conditions["clip"]
    avatar = conditions["avatar"]
    sampling = conditions["sampling"]
    machine = conditions["machine"]
    libraries = conditions["libraries"]
    gpu = conditions["gpu"]
    speed = machine["speed"]

    print()
    print(f"quality measure - {report['run']['at']}")
    if report["run"]["label"]:
        print(f"  {report['run']['label']}")
    print()

    print("conditions")
    dirty = "clean" if code["dirty"] is False else f"DIRTY, {code['changed_files']} files"
    print(f"  code            {code['commit'] or 'n/a'} ({dirty})")
    print(
        f"  clip            {Path(clip['path']).name}  {clip['frames']} frames  "
        f"{clip['width']}x{clip['height']} @{clip['fps']:.1f}  [{clip['fingerprint']}]"
    )
    print(
        f"  avatar          {avatar['id']}  v{avatar['manifest_version']}  "
        f"{avatar['assets']} assets  [{avatar['fingerprint']}]"
    )
    segment = f"  segment {sampling['segment']!r}" if sampling["segment"] else ""
    print(
        f"  measured        frames {sampling['first']}-{sampling['last']}, "
        f"warm-up {sampling['warmup']}{segment}"
    )
    counts = report["counts"]
    print(
        f"                  {counts['measured']} measured, {counts['no_face']} without a face, "
        f"{counts['passthrough']} passed through, {counts['fit_failed']} unfittable"
    )
    print(
        f"  machine         {machine['platform']}, {machine['logical_cpus']} logical CPUs, "
        f"OpenCV using {machine['cv2_threads']}"
    )
    print(
        f"  libraries       python {libraries['python']}, numpy {libraries['numpy']}, "
        f"cv2 {libraries['cv2']}, mediapipe {libraries['mediapipe'] or 'n/a'}"
    )
    print(f"  gpu             {gpu['name'] or 'n/a'}, driver {gpu['driver'] or 'n/a'}")
    print(
        f"  machine speed   {speed['before']['median_ms']:.3f} ms at the start, "
        f"{speed['after']['median_ms']:.3f} ms at the end ({speed['warmed'] * 100:+.0f}%)"
    )
    for note in speed["notes"]:
        print(f"                  {note}")

    print()
    print("geometry and detail - the same numbers on any machine")
    print(_COLUMNS)
    for label, key, places in (
        ("warp residual", "residual_px", 2),
        ("near-eye sharpness", "sharpness_near_eye", 0),
        ("face sharpness", "sharpness_face", 0),
        ("face width", "face_width_px", 1),
    ):
        for group in ("all", "frontal", "turned"):
            entry = report["metrics"][key][group]
            if entry["count"]:
                print(_row(f"{label}, {group}", entry, places))

    print()
    print("frame cost - this machine, this run")
    print(_COLUMNS)
    for group in ("all", "frontal", "turned"):
        entry = report["metrics"]["animate_ms"][group]
        if entry["count"]:
            print(_row(f"animate, {group}", entry, 2))
    drift = report["frame_cost_drift"]
    print(_row(f"first {drift['frames_each_end']} frames", drift["start"], 2))
    print(_row(f"last {drift['frames_each_end']} frames", drift["end"], 2))
    if speed["erratic"]:
        print("  ^ something else was running: " + "; ".join(speed["notes"]))

    base_mesh = report["base_mesh_frames"]
    print()
    print("photographs on screen")
    print(
        f"  {'role':<14}{'region':>11}{'frames':>8}{'%':>7}{'median share':>14}"
        f"  {'what it is':<18}{'file'}"
    )
    for row in report["photographs"]:
        print(
            f"  {row['role']:<14}{row['region']:>11}{row['frames']:>8}"
            f"{row['percent']:>7.1f}{row['median_share']:>14.2f}"
            f"  {row['rung'] or '':<18}{row['file'] or ''}"
        )
    if base_mesh:
        print(
            f"  {'(none at all)':<14}{'':>11}{base_mesh:>8}"
            f"{100.0 * base_mesh / max(counts['measured'], 1):>7.1f}"
        )


# --- comparing --------------------------------------------------------------


def _dig(report: dict[str, Any], path: tuple[str, str]) -> Any:
    return report.get("conditions", {}).get(path[0], {}).get(path[1])


def compare(baseline: dict[str, Any], current: dict[str, Any]) -> int:
    """Print one run against another, or refuse to.

    The refusal is the point. Two numbers side by side look like a comparison
    whatever produced them, and the arrow between a run on the old library and a
    run on the new one is not a result — it is two different measurements
    subtracted.
    """
    print()
    print("comparison")
    print(f"  before   {baseline['run']['at']}  {baseline['run']['label'] or ''}")
    print(f"  after    {current['run']['at']}  {current['run']['label'] or ''}")

    if baseline.get("format") != current.get("format"):
        print()
        print(
            f"  REFUSED: the baseline was written by report format "
            f"{baseline.get('format')} and this one is {current.get('format')}. "
            "Re-measure the baseline."
        )
        return 2

    differences = [
        (path, _dig(baseline, path), _dig(current, path))
        for path in COMPARABLE
        if _dig(baseline, path) != _dig(current, path)
    ]
    if differences:
        print()
        print("  REFUSED: these two runs did not measure the same thing.")
        for (group, key), was, now in differences:
            print(f"    {group}.{key:<14} {was}  ->  {now}")
        print()
        print("  Re-measure the baseline under the same conditions, or compare a")
        print("  different pair. Two numbers are not a comparison by themselves.")
        return 2

    before_code = baseline["conditions"]["code"]
    after_code = current["conditions"]["code"]
    print(
        f"  code     {before_code['commit']} -> {after_code['commit']}"
        f"{'  (after is DIRTY)' if after_code['dirty'] else ''}"
    )

    was_library = _dig(baseline, ("avatar", "fingerprint"))
    now_library = _dig(current, ("avatar", "fingerprint"))
    if was_library != now_library:
        print(f"  library  {was_library} -> {now_library}  <- THE PHOTOGRAPHS CHANGED")
        print("           Everything below is two different avatars, not two versions")
        print("           of one engine. Read it as such.")

    # Whether the two frame costs may be subtracted from one another. Three
    # things have to hold, and each of them has been the reason a number was
    # wrong at least once.
    #
    # The comparison is *start against start*. Both runs warm the machine by
    # roughly half — that is measured, and it is the same slowdown a real call
    # produces — so the state that says whether they began alike is the probe
    # taken before each one, not the one taken after.
    was_speed = baseline["conditions"]["machine"]["speed"]
    now_speed = current["conditions"]["machine"]["speed"]
    timings_ok = True
    reasons = []

    for which, speed in (("the baseline run", was_speed), ("this run", now_speed)):
        if speed["erratic"]:
            timings_ok = False
            reasons.append(f"{which}'s probe was erratic, so something else was running")

    for path in COMPARABLE_TIMINGS:
        if _dig(baseline, path) != _dig(current, path):
            timings_ok = False
            reasons.append(f"{path[0]}.{path[1]} differs")

    started = abs(now_speed["before"]["median_ms"] - was_speed["before"]["median_ms"]) / max(
        was_speed["before"]["median_ms"], 1e-9
    )
    if started > MACHINE_DRIFT_LIMIT:
        timings_ok = False
        reasons.append(
            f"the two runs started on a machine {started * 100:.0f}% apart in speed "
            f"({was_speed['before']['median_ms']:.3f} ms then "
            f"{now_speed['before']['median_ms']:.3f} ms)"
        )

    # And they must have warmed alike. A run that slowed by half and one that
    # slowed by a tenth spent their frames in different thermal states, however
    # they began.
    warmed = abs(now_speed["warmed"] - was_speed["warmed"])
    if warmed > MACHINE_WARMING_DIFFERENCE:
        timings_ok = False
        reasons.append(
            f"one run warmed {was_speed['warmed'] * 100:.0f}% and the other "
            f"{now_speed['warmed'] * 100:.0f}%"
        )

    print()
    print(f"  {'':<30}{'before':>12}{'after':>12}{'change':>12}")
    for label, key, group, places in (
        ("warp residual, all", "residual_px", "all", 2),
        ("warp residual, frontal", "residual_px", "frontal", 2),
        ("warp residual, turned", "residual_px", "turned", 2),
        ("near-eye sharpness, all", "sharpness_near_eye", "all", 0),
        ("near-eye sharpness, turned", "sharpness_near_eye", "turned", 0),
        ("face sharpness, all", "sharpness_face", "all", 0),
    ):
        was = baseline["metrics"][key][group]
        now = current["metrics"][key][group]
        if not was["count"] or not now["count"]:
            continue
        change = now["mean"] - was["mean"]
        print(
            f"  {label:<30}{was['mean']:>12.{places}f}{now['mean']:>12.{places}f}"
            f"{change:>+12.{places}f}"
        )

    for label, group in (("animate mean", "all"), ("animate p95", "all")):
        was = baseline["metrics"]["animate_ms"][group]
        now = current["metrics"]["animate_ms"][group]
        if not was["count"] or not now["count"]:
            continue
        field_name = "mean" if label.endswith("mean") else "p95"
        if timings_ok:
            change = f"{now[field_name] - was[field_name]:>+12.2f}"
        else:
            change = f"{'not comparable':>12}"
        print(f"  {label:<30}{was[field_name]:>12.2f}{now[field_name]:>12.2f}{change}")

    if not timings_ok:
        print()
        print("  The frame cost was not compared: " + "; ".join(reasons) + ".")
        print("  The geometry and detail numbers above are unaffected - they do not")
        print("  depend on how busy the machine was.")

    moved = _photograph_changes(baseline, current)
    if moved:
        print()
        print(f"  {'photographs that moved':<30}{'before':>12}{'after':>12}{'change':>12}")
        for role, was_percent, now_percent in moved:
            print(
                f"  {role:<30}{was_percent:>11.1f}%{now_percent:>11.1f}%"
                f"{now_percent - was_percent:>+11.1f}%"
            )
    return 0


def _photograph_changes(
    baseline: dict[str, Any], current: dict[str, Any]
) -> list[tuple[str, float, float]]:
    """Sources whose share of the clip moved by more than a percent."""
    was = {row["role"]: row["percent"] for row in baseline["photographs"]}
    now = {row["role"]: row["percent"] for row in current["photographs"]}
    moved = [
        (role, was.get(role, 0.0), now.get(role, 0.0))
        for role in sorted(set(was) | set(now))
        if abs(now.get(role, 0.0) - was.get(role, 0.0)) > 1.0
    ]
    return sorted(moved, key=lambda row: abs(row[2] - row[1]), reverse=True)


def read_report(path: Path) -> dict[str, Any] | str:
    """A report from disk, or a sentence saying why it could not be read."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return f"Could not read {path}."
    except json.JSONDecodeError as error:
        return f"{path} is not a report this tool wrote: {error}."
    if not isinstance(payload, dict) or payload.get("tool") != "measure_quality":
        return f"{path} is not a report this tool wrote."
    return payload


# --- entry point ------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip",
        type=Path,
        default=REPO_ROOT / "recordings" / "driving.mkv",
        help="The clip to measure. The same one every time, or the numbers mean nothing.",
    )
    parser.add_argument("--avatar", default=None, help="Avatar id; defaults to the configured one.")
    parser.add_argument(
        "--segment",
        default=None,
        help=(
            "Measure only the frames of the recorded segment whose instruction contains "
            "this text — 'turn your head', 'blink'. NOT TRUSTWORTHY on driving.mkv: its "
            "map is off by up to 450 frames, see backlog 31. Name frames instead."
        ),
    )
    parser.add_argument("--first", type=int, default=None, help="First frame to measure.")
    parser.add_argument("--last", type=int, default=None, help="Last frame to measure.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP_FRAMES,
        help=(
            "Frames to render before measuring anything, so the engine's estimate of the "
            "driver's resting face has settled."
        ),
    )
    parser.add_argument(
        "--label",
        default="",
        help="What this run is, in words. It travels with the numbers into the report.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Where to write the JSON report.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Measure, then compare against this earlier report.",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        nargs=2,
        default=None,
        metavar=("BEFORE", "AFTER"),
        help="Compare two existing reports and measure nothing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.compare:
        reports = [read_report(path) for path in args.compare]
        for report in reports:
            if isinstance(report, str):
                print(report)
                return 1
        before, after = reports
        assert isinstance(before, dict) and isinstance(after, dict)
        return compare(before, after)

    import cv2

    clip = args.clip
    if not clip.is_file():
        print(f"No clip at {clip}. Record one with scripts/record_driving_clip.py")
        return 1

    baseline: dict[str, Any] | None = None
    if args.baseline is not None:
        # Read it now rather than after the run: an unreadable baseline should
        # cost a second, not a minute of rendering.
        found = read_report(args.baseline)
        if isinstance(found, str):
            print(found)
            return 1
        baseline = found

    paths = AppPaths.resolve()
    settings = load_settings(paths)
    models = paths.resolve_relative(settings.models.root_path)
    model_path = models / DEFAULT_MODEL_RELATIVE_PATH
    if not model_path.is_file():
        print(f"Model bundle missing: {model_path}")
        print("Fetch it with: python scripts/fetch_face_landmarker.py")
        return 1

    repository = FilesystemAvatarRepository(paths.resolve_relative(settings.avatars.library_path))
    avatar_id = args.avatar or settings.avatars.default_avatar_id or "my_avatar"
    descriptor = repository.get(avatar_id)

    capture = cv2.VideoCapture(str(clip))
    clip_facts = {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)) or 30.0,
    }
    capture.release()

    chosen = resolve_range(args, clip_facts["frames"], clip_segments(clip))
    if isinstance(chosen, str):
        print(chosen)
        return 1
    first, last = chosen

    print(f"measuring {clip.name}, frames {first}-{last} ...", flush=True)
    speed_before = machine_speed()

    tracker = MediaPipeFaceTracker(model_path=model_path)
    tracker.start()
    engine = WarpAvatarEngine(model_path=model_path)
    engine.start()
    try:
        engine.load_avatar(descriptor)
        frames, counts = measure_clip(
            clip, engine, tracker, avatar_id, clip_facts["fps"], first, last
        )
    finally:
        engine.stop()
        tracker.stop()

    speed_after = machine_speed()

    if not frames:
        print("Nothing was measured: no frame in the chosen range rendered an avatar.")
        return 1

    report = build_report(
        args,
        descriptor,
        clip,
        clip_facts,
        first,
        last,
        frames,
        counts,
        machine_verdict(speed_before, speed_after),
    )

    out = args.out or DEFAULT_OUT_DIR / f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_report(report)
    print()
    print(f"written to {out}")

    if baseline is not None:
        return compare(baseline, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
