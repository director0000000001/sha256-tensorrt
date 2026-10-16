"""The live "Calibrate me" run, on the running pipeline.

The pipeline keeps running as it does on a call -- the same tracker, the
same engine, the same frame rate -- and the session listens to every
published frame through :meth:`PipelineRunner.add_observer`. On the
pipeline thread it takes his readings from the frame's tracking, the
engine's readings for that frame (``frame_readings``), and hands two
copies of the frame on: the camera frame to a writer thread that encodes
the session's clip, the composited frame to a second tracker on its own
thread that reads the avatar's face back -- so neither the encoder nor the
second tracker costs the pipeline a frame.

**It records its own session.** ``<directory>/clip.mkv`` (the camera
frames, lossless FFV1 like the recorder's clips), ``clip.json`` (the
segment map ``scripts/measure_follow.py`` reads, one segment per step,
plus what the session was), ``tracking.npz`` (every written frame's
landmarks, blendshapes, pose, distance and timestamp, the capture's
format) -- so the same face can be re-measured offline after any change,
without the person present (:mod:`sha256.calibration.offline`). The clip
and the tracking are aligned frame for frame: a camera frame the writer
had no room for is dropped together with its tracking row, and counted.

**Day 22 (2026-10-15).** Every frame carries whether the head sat in the
window's green oval (``his["guide_ok"]``, and ``guide_ok`` in
``tracking.npz``), so the measurement can drop the frames out of it; the
clock never pauses for it. And a step whose movement was too small to
measure -- by measure.py's own thresholds, read off the step's own frames
as its window ends -- is asked once more, right after it ("Once more:
..."), once per step at most: the step list grows, and ``steps``,
``current`` and ``finished`` follow it.

Privacy: this is the person's face. The directory is under ``recordings/``
(gitignored), it is written only after they said yes in the window, and
:meth:`cancel` removes it.
"""

from __future__ import annotations

import json
import math
import queue
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sha256.calibration.measure import (
    BLINK_MIN_RANGE,
    BROW_MIN_RANGE,
    GAZE_MIN_RANGE,
    TALK_MIN_RANGE,
)
from sha256.calibration.samples import Sample, avatar_measures, his_measures
from sha256.calibration.script import (
    STEPS,
    CalibrationStep,
    StepPosition,
    repeat_of,
    step_at,
    total_seconds,
)
from sha256.core.enums import PixelFormat
from sha256.core.interfaces import FaceTracker
from sha256.core.types import FaceTrackingResult, FrameMetadata, VideoFrame
from sha256.pipeline.runner import PipelineRunner, PipelineSnapshot, _as_bgr
from sha256.utils.logging import get_logger
from sha256.utils.timing import now

__all__ = ["CalibrationSession", "SessionFiles", "step_movement"]

_LOG = get_logger(__name__)

# The read phase's own kind: never one of the measured steps.
READY_KIND = "ready"
CLIP_NAME = "clip.mkv"
SIDECAR_NAME = "clip.json"
TRACKING_NAME = "tracking.npz"
# Frames the writer may fall behind by before frames are dropped: three
# seconds at 30 fps, about 250 MB of 720p frames at the worst.
_WRITER_QUEUE = 90

# Day 22: a step whose movement was too small to measure is asked once
# more, right after it. What each kind's movement is read on, and the
# smallest movement that counts -- measure.py's own thresholds for what a
# real movement is on the step made for it, so a step that passes here is
# one the measurement can use. "rest", "turn" and "lead" are never asked
# again: nothing is wanted from them but stillness or a recording.
#   eyes:  the eye ratio's fall from the rest (rest median - p5)
#   mouth: the opening's rise over the rest (p95 - rest median)
#   smile: the mouth width's rise over the rest (p95 / rest median - 1)
#   brows: the brow's rise over the rest (p95 / rest median - 1)
#   gaze:  the glance's spread (p95 - p5 of the iris offset)
# Measured against the still step's own median, not the window's p5, so
# a person who closed their eyes on "one" and held them shut -- a window
# with no change in it -- still reads as a full movement.
_EYES, _MOUTH, _SMILE, _BROWS, _GAZE = "ear", "opening", "width", "brow", "gaze_x"
# A smile's width has no threshold in measure.py (the smile is reported,
# not fitted): +5% of the resting width, about a third of a closed smile
# on the four sessions of 2026-09-12 (+15..22%).
SMILE_MIN_RANGE = 0.05
RETRY_MEASURES: dict[str, tuple[str, float]] = {
    "blink": (_EYES, BLINK_MIN_RANGE),
    "eyes_half": (_EYES, BLINK_MIN_RANGE),
    "eyes_shut": (_EYES, BLINK_MIN_RANGE),
    "open_small": (_MOUTH, TALK_MIN_RANGE),
    "open": (_MOUTH, TALK_MIN_RANGE),
    "read": (_MOUTH, TALK_MIN_RANGE),
    "talk": (_MOUTH, TALK_MIN_RANGE),
    "laugh": (_MOUTH, TALK_MIN_RANGE),
    "smile_small": (_SMILE, SMILE_MIN_RANGE),
    "smile_closed": (_SMILE, SMILE_MIN_RANGE),
    "smile_teeth": (_SMILE, SMILE_MIN_RANGE),
    "brows_half": (_BROWS, BROW_MIN_RANGE),
    "brows": (_BROWS, BROW_MIN_RANGE),
    "gaze": (_GAZE, GAZE_MIN_RANGE),
}
# The steps asked for a LITTLE of the movement are held to half the
# threshold: measure.py's are set on the full movement ("a real raise is
# +34..38%"), and a person doing exactly what "a little" asks must not be
# told to do it once more.
_HALF_LEVEL_KINDS = frozenset({"eyes_half", "open_small", "smile_small", "brows_half"})
_HALF_LEVEL_FACTOR = 0.5
# Fewer frames of a face in the oval than this, and the window saw no
# movement at all: asked again too.
RETRY_MIN_FRAMES = 10


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def step_movement(
    kind: str, readings: list[dict[str, float]], rest: dict[str, float]
) -> tuple[float, float] | None:
    """How far ``readings`` (his, one dict per frame of the measured
    window) moved for a step of ``kind``, and the threshold it is held to;
    ``None`` for a kind that is never asked again. ``rest`` holds the
    still step's medians (empty before it: the window's own p95 / p5
    stand in). Too few frames read as no movement at all."""
    measure = RETRY_MEASURES.get(kind)
    if measure is None:
        return None
    key, threshold = measure
    if kind in _HALF_LEVEL_KINDS:
        threshold *= _HALF_LEVEL_FACTOR
    values = [v for v in (r.get(key, math.nan) for r in readings) if math.isfinite(v)]
    if len(values) < RETRY_MIN_FRAMES:
        return 0.0, threshold
    ref = rest.get(key, math.nan)
    if key == _EYES:
        top = ref if math.isfinite(ref) else _percentile(values, 95)
        return top - _percentile(values, 5), threshold
    if key == _MOUTH:
        low = ref if math.isfinite(ref) else _percentile(values, 5)
        return _percentile(values, 95) - low, threshold
    if key in (_SMILE, _BROWS):
        low = ref if math.isfinite(ref) and ref > 0.0 else _percentile(values, 5)
        return _percentile(values, 95) / max(low, 1e-6) - 1.0, threshold
    return _percentile(values, 95) - _percentile(values, 5), threshold


@dataclass(frozen=True, slots=True)
class SessionFiles:
    directory: Path
    clip: Path | None
    sidecar: Path | None
    tracking: Path
    frames_written: int
    frames_dropped: int


class _ClipWriter(threading.Thread):
    """Encodes camera frames handed to it, in order, on its own thread."""

    def __init__(self, path: Path, fps: float) -> None:
        super().__init__(name="sha256-calibration-clip", daemon=True)
        self._path = path
        self._fps = fps
        self._queue: queue.Queue[tuple[np.ndarray, PixelFormat] | None] = queue.Queue(
            maxsize=_WRITER_QUEUE
        )
        self._writer: Any = None
        self.codec = ""
        self.size: tuple[int, int] | None = None
        self.written = 0
        self.dropped = 0
        self.error: str | None = None

    def offer(self, data: np.ndarray, pixel_format: PixelFormat) -> bool:
        """Queue a frame (already copied). ``False``, and counted, when the
        writer is too far behind."""
        try:
            self._queue.put_nowait((data, pixel_format))
        except queue.Full:
            self.dropped += 1
            return False
        return True

    def close(self) -> None:
        self._queue.put(None)
        self.join(timeout=30.0)

    def run(self) -> None:
        import cv2

        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                data, pixel_format = item
                bgr = _as_bgr(data, pixel_format)
                if bgr is None:
                    continue
                if self._writer is None:
                    height, width = bgr.shape[:2]
                    self.size = (width, height)
                    self._writer = cv2.VideoWriter(
                        str(self._path), cv2.VideoWriter.fourcc(*"FFV1"), self._fps, (width, height)
                    )
                    self.codec = "FFV1 (lossless)"
                    if not self._writer.isOpened():
                        self._writer = cv2.VideoWriter(
                            str(self._path),
                            cv2.VideoWriter.fourcc(*"MJPG"),
                            self._fps,
                            (width, height),
                        )
                        self.codec = "MJPG (fallback)"
                    if not self._writer.isOpened():
                        self.error = "no video writer could be opened"
                        self._writer = None
                        break
                self._writer.write(bgr)
                self.written += 1
        except Exception as exc:  # the session must still finish
            self.error = str(exc)
            _LOG.exception("The calibration clip writer failed")
        finally:
            if self._writer is not None:
                self._writer.release()


class _OutputReader(threading.Thread):
    """Reads the avatar's face back from the composited frames -- the newest
    one each time it is free, on its own thread."""

    def __init__(self, tracker: FaceTracker) -> None:
        super().__init__(name="sha256-calibration-output", daemon=True)
        self._tracker = tracker
        self._pending: tuple[int, float, np.ndarray, PixelFormat] | None = None
        self._condition = threading.Condition()
        self._closing = False
        self.results: dict[int, dict[str, float] | None] = {}
        self.read = 0
        self.skipped = 0

    def offer(self, frame_id: int, timestamp: float, data: np.ndarray, fmt: PixelFormat) -> None:
        with self._condition:
            if self._pending is not None:
                self.skipped += 1
            self._pending = (frame_id, timestamp, data, fmt)
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify()
        self.join(timeout=10.0)

    def run(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._pending is None and not self._closing:
                        self._condition.wait(0.5)
                    if self._pending is None:
                        break
                    frame_id, timestamp, data, fmt = self._pending
                    self._pending = None
                frame = VideoFrame(
                    data=data,
                    pixel_format=fmt,
                    metadata=FrameMetadata(frame_id=frame_id, capture_timestamp=timestamp),
                )
                try:
                    result = self._tracker.process(frame)
                except Exception:
                    _LOG.exception("The output reader failed on frame %d", frame_id)
                    continue
                height, width = data.shape[:2]
                measures = avatar_measures(result, width, height)
                self.results[frame_id] = measures if measures is not None else {}
                self.read += 1
        finally:
            with self._condition:
                self._closing = True


class CalibrationSession:
    """One guided run, from :meth:`start` to :meth:`finish` (or
    :meth:`cancel`), driven by the window's timer through :meth:`current`."""

    def __init__(
        self,
        runner: PipelineRunner,
        *,
        engine: Any,
        avatar_id: str,
        directory: Path,
        steps: tuple[CalibrationStep, ...] = STEPS,
        output_tracker_factory: Callable[[], FaceTracker] | None = None,
        record_clip: bool = True,
        fps: float = 30.0,
        clock: Callable[[], float] = now,
        guide: Callable[[FaceTrackingResult], bool] | None = None,
        retry: bool = True,
    ) -> None:
        self._runner = runner
        self._engine = engine
        self._avatar_id = avatar_id
        self._directory = Path(directory)
        # Day 22: the list grows by a step's repeat when its movement was
        # too small (see _judge); the window reads it through `steps`.
        self._steps: list[CalibrationStep] = list(steps)
        self._repeat: list[bool] = [False] * len(self._steps)
        # Day 22: whether the head sits in the window's green oval, per
        # frame (the window hands in the head guide's fit); None: no guide,
        # every frame counts as in it.
        self._guide = guide
        self._retry = retry
        self._judged = 0  # the steps whose measured window was checked
        self._windows: dict[int, list[dict[str, float]]] = {}
        self._rest_readings: list[dict[str, float]] = []
        self._repeats: list[str] = []
        self._guide_rows: list[bool] = []  # per written frame, beside _rows
        self._factory = output_tracker_factory
        self._record_clip = record_clip
        self._fps = fps
        self._clock = clock
        self._lock = threading.RLock()
        self._samples: list[Sample] = []
        self._rows: list[FaceTrackingResult] = []  # the written frames' tracking
        self._segments: dict[str, dict[str, Any]] = {}
        self._started: float | None = None
        self._step_index = -1
        self._phase: tuple[int, bool] | None = None
        self._writer: _ClipWriter | None = None
        self._reader: _OutputReader | None = None
        self._tracker: FaceTracker | None = None
        self._observed = 0
        self._done = False

    # --- state for the window ---------------------------------------------

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def steps(self) -> tuple[CalibrationStep, ...]:
        """The steps as they stand now -- with any repeat appended (day 22)."""
        with self._lock:
            return tuple(self._steps)

    @property
    def repeats(self) -> list[str]:
        """The kinds asked once more, in order (day 22)."""
        with self._lock:
            return list(self._repeats)

    @property
    def started(self) -> bool:
        return self._started is not None

    @property
    def elapsed(self) -> float:
        return 0.0 if self._started is None else self._clock() - self._started

    @property
    def finished(self) -> bool:
        if self._started is None:
            return False
        elapsed = self.elapsed
        with self._lock:
            self._settle(elapsed)
            return elapsed >= total_seconds(tuple(self._steps))

    @property
    def frames_observed(self) -> int:
        return self._observed

    def current(self) -> StepPosition | None:
        """Where the clock is now (see :func:`step_at`), ``None`` before the
        start and after the end. A measured window that has just ended is
        judged first (day 22), so the window shows a repeat the moment it
        is due, whichever of the two threads reaches the clock first."""
        if self._started is None:
            return None
        elapsed = self.elapsed
        with self._lock:
            self._settle(elapsed)
            return step_at(tuple(self._steps), elapsed)

    @property
    def samples(self) -> list[Sample]:
        with self._lock:
            return list(self._samples)

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._started is not None:
            return
        self._directory.mkdir(parents=True, exist_ok=True)
        if self._record_clip:
            self._writer = _ClipWriter(self._directory / CLIP_NAME, self._fps)
            self._writer.start()
        if self._factory is not None:
            self._tracker = self._factory()
            self._tracker.start()
            self._reader = _OutputReader(self._tracker)
            self._reader.start()
        self._started = self._clock()
        self._runner.add_observer(self._observe)
        _LOG.info(
            "Calibration started: %d steps, %.0f s, avatar %s, into %s",
            len(self._steps),
            total_seconds(tuple(self._steps)),
            self._avatar_id,
            self._directory,
        )

    # --- the repeat of a too-small step (day 22) ---------------------------

    def _settle(self, elapsed: float) -> None:
        """Judge every measured window that has ended by ``elapsed`` and
        not been judged yet, in order. Called with the lock held."""
        while self._judged < len(self._steps):
            end = total_seconds(tuple(self._steps[: self._judged + 1]))
            if elapsed < end:
                return
            index = self._judged
            self._judged += 1
            self._judge(index)

    def _rest_medians(self) -> dict[str, float]:
        """His still step's medians, the reference a movement is read
        from; empty before the still step."""
        rest: dict[str, float] = {}
        for key in (_EYES, _MOUTH, _SMILE, _BROWS):
            values = [r[key] for r in self._rest_readings if math.isfinite(r.get(key, math.nan))]
            if values:
                rest[key] = float(np.median(values))
        return rest

    def _judge(self, index: int) -> None:
        """A step's measured window is over: if its movement was too small
        to measure (see :func:`step_movement`), the same step is asked once
        more, right after it -- once per step, never for a repeat."""
        step = self._steps[index]
        readings = self._windows.pop(index, [])
        if not self._retry or not step.measured or self._repeat[index]:
            return
        judged = step_movement(step.kind, readings, self._rest_medians())
        if judged is None:
            return
        movement, threshold = judged
        if movement >= threshold:
            return
        self._steps.insert(index + 1, repeat_of(step))
        self._repeat.insert(index + 1, True)
        self._repeats.append(step.kind)
        _LOG.info(
            "Calibration step %s moved %.3f (under %.3f, on %d frames in the oval): asked once "
            "more -- now %d steps, %.0f s",
            step.kind,
            movement,
            threshold,
            len(readings),
            len(self._steps),
            total_seconds(tuple(self._steps)),
        )

    # --- the frames --------------------------------------------------------

    def _enter_phase(self, index: int, step: CalibrationStep, reading: bool) -> None:
        """The clock moved into a new phase: a step's read phase or its
        measured one. The anchor is learned from the still step's measured
        frames and nothing earlier -- reset as they start, so the read
        phase (the person looking at the instruction) is not in it."""
        if index != self._step_index:
            self._step_index = index
            _LOG.info("Calibration step %d/%d: %s", index + 1, len(self._steps), step.instruction)
        if step.kind == "rest" and not reading:
            reset = getattr(self._engine, "reset_anchor", None)
            if reset is not None:
                reset(self._avatar_id)

    def _in_guide(self, tracking: FaceTrackingResult) -> bool:
        if self._guide is None:
            return True
        try:
            return bool(self._guide(tracking))
        except Exception:
            _LOG.exception("The head guide failed on a calibration frame")
            return True

    def _observe(self, snapshot: PipelineSnapshot) -> None:
        """On the pipeline thread, once per published frame."""
        if self._started is None or snapshot.frame is None or snapshot.tracking is None:
            return
        elapsed = self._clock() - self._started
        with self._lock:
            self._settle(elapsed)
            placed = step_at(tuple(self._steps), elapsed)
        if placed is None:
            self._done = True
            return
        index, step, _, reading, _phase = placed
        if (index, reading) != self._phase:
            self._phase = (index, reading)
            self._enter_phase(index, step, reading)
        # The read and count phases: the instruction is on screen, nothing
        # measured -- their frames carry their own kind, and the sidecar
        # their own segment.
        kind = READY_KIND if reading else step.kind
        instruction = f"Get ready: {step.instruction}" if reading else step.instruction
        tracking = snapshot.tracking
        source = snapshot.source
        frame = snapshot.frame
        height, width = frame.data.shape[:2]
        his = his_measures(tracking, width, height)
        # Day 22: the head in the green oval or not, on every frame; the
        # measurement can drop the frames out of it. Only on a frame with
        # readings -- an empty `his` is how a lost face is told.
        in_guide = self._in_guide(tracking)
        if his:
            his["guide_ok"] = 1.0 if in_guide else 0.0
        readings: dict[str, float] = {}
        read = getattr(self._engine, "frame_readings", None)
        if read is not None:
            got = read(self._avatar_id)
            if got:
                readings = dict(got)
        sample = Sample(
            frame_id=frame.metadata.frame_id,
            timestamp=frame.metadata.capture_timestamp,
            step=kind,
            face=bool(tracking.face_detected and his),
            his=his,
            engine=readings,
        )
        with self._lock:
            self._samples.append(sample)
            self._observed += 1
            if not reading and his and in_guide:
                self._windows.setdefault(index, []).append(his)
                if step.kind == "rest":
                    self._rest_readings.append(his)
            # The source buffer is borrowed: copied before it is queued.
            queued = (
                self._writer is not None
                and source is not None
                and self._writer.offer(source.data.copy(), source.pixel_format)
            )
            if queued:
                clip_index = len(self._rows)
                self._rows.append(tracking)
                self._guide_rows.append(in_guide)
                # By the step's place, not its kind: a repeat (day 22) is a
                # segment of its own, after the first try's.
                segment = self._segments.setdefault(
                    f"{index}:{kind}",
                    {
                        "instruction": instruction,
                        "kind": kind,
                        "first_frame": clip_index,
                        "last_frame": clip_index,
                    },
                )
                segment["last_frame"] = clip_index
        if self._reader is not None:
            self._reader.offer(
                frame.metadata.frame_id,
                frame.metadata.capture_timestamp,
                frame.data.copy(),
                frame.pixel_format,
            )

    def _stop_listening(self) -> None:
        self._runner.remove_observer(self._observe)
        if self._writer is not None:
            self._writer.close()
        if self._reader is not None:
            self._reader.close()
        if self._tracker is not None:
            try:
                self._tracker.stop()
            except Exception:
                _LOG.exception("The output tracker did not stop cleanly")
            self._tracker = None

    def finish(self) -> SessionFiles:
        """Stop listening, join the threads, merge the avatar's readings into
        the samples, and write the session's files."""
        if self._started is None:
            msg = "the calibration was never started"
            raise RuntimeError(msg)
        self._stop_listening()
        with self._lock:
            if self._reader is not None:
                for sample in self._samples:
                    got = self._reader.results.get(sample.frame_id)
                    if got is not None:
                        sample.avatar = got
            rows = list(self._rows)
            guide_rows = list(self._guide_rows)
            segments = sorted(self._segments.values(), key=lambda s: int(s["first_frame"]))
        tracking_path = self._directory / TRACKING_NAME
        _write_tracking(tracking_path, rows, self._anchor_state(), guide_rows)
        clip_path = sidecar_path = None
        written = dropped = 0
        if self._writer is not None:
            written, dropped = self._writer.written, self._writer.dropped
            if self._writer.error:
                _LOG.warning("The session's clip is incomplete: %s", self._writer.error)
            if written:
                clip_path = self._directory / CLIP_NAME
                sidecar_path = self._directory / SIDECAR_NAME
                size = self._writer.size or (0, 0)
                sidecar_path.write_text(
                    json.dumps(
                        {
                            "video": CLIP_NAME,
                            "width": size[0],
                            "height": size[1],
                            "fps": self._fps,
                            "frames": written,
                            "codec": self._writer.codec,
                            "segments": segments,
                            "dropped": dropped,
                            "avatar": self._avatar_id,
                            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "what": (
                                "a Calibrate me session -- the person's face, this machine only"
                            ),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        _LOG.info(
            "Calibration finished: %d frames observed, %d written to the clip (%d dropped), "
            "avatar read on %d",
            self._observed,
            written,
            dropped,
            self._reader.read if self._reader is not None else 0,
        )
        return SessionFiles(
            self._directory, clip_path, sidecar_path, tracking_path, written, dropped
        )

    def cancel(self) -> None:
        """Stop, and remove everything written -- the person changed their mind."""
        if self._started is None:
            return
        self._stop_listening()
        shutil.rmtree(self._directory, ignore_errors=True)
        _LOG.info("Calibration cancelled; %s removed", self._directory)

    def _anchor_state(self) -> dict[str, np.ndarray] | None:
        snapshot = getattr(self._engine, "anchor_state", None)
        if snapshot is None:
            return None
        try:
            state = snapshot(self._avatar_id)
        except Exception:
            _LOG.exception("The anchor could not be read for the session")
            return None
        return dict(state) if state else None


def _write_tracking(
    path: Path,
    rows: list[FaceTrackingResult],
    anchor: dict[str, np.ndarray] | None,
    guide_ok: list[bool] | None = None,
) -> None:
    """The capture's own format (see ``PipelineRunner.capture``), plus the
    head pose's translation per frame, which the capture does not keep,
    and (day 22) whether the head sat in the green oval on each frame."""
    names = next((r.blendshapes.names for r in rows if r.blendshapes is not None), ())
    count = next((r.landmarks.count for r in rows if r.landmarks is not None), 0)
    width = len(names)
    arrays: dict[str, Any] = {
        "landmarks": np.stack(
            [
                r.landmarks.points
                if r.landmarks is not None
                else np.zeros((count, 3), dtype=np.float32)
                for r in rows
            ]
        )
        if rows
        else np.zeros((0, count, 3), dtype=np.float32),
        "weights": np.stack(
            [
                r.blendshapes.weights
                if r.blendshapes is not None
                else np.zeros(width, dtype=np.float32)
                for r in rows
            ]
        )
        if rows
        else np.zeros((0, width), dtype=np.float32),
        "pose": np.array(
            [
                (r.head_pose.yaw, r.head_pose.pitch, r.head_pose.roll)
                if r.head_pose is not None
                else (0.0, 0.0, 0.0)
                for r in rows
            ],
            dtype=np.float32,
        ).reshape(-1, 3),
        "translation": np.array(
            [
                tuple(r.head_pose.translation) if r.head_pose is not None else (0.0, 0.0, 0.0)
                for r in rows
            ],
            dtype=np.float32,
        ).reshape(-1, 3),
        "timestamps": np.array([r.metadata.capture_timestamp for r in rows], dtype=np.float64),
        "detected": np.array([r.face_detected for r in rows], dtype=bool),
        "guide_ok": np.array(
            guide_ok if guide_ok is not None and len(guide_ok) == len(rows) else [True] * len(rows),
            dtype=bool,
        ),
        "names": np.array(names),
    }
    arrays.update({f"anchor_{key}": value for key, value in (anchor or {}).items()})
    np.savez_compressed(path, **arrays)
