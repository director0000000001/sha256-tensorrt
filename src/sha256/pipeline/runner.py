"""The real-time loop.

Until now the pipeline could only be single-stepped. This is the loop the
architecture document specifies (``SYSTEM_ARCHITECTURE.md`` §6), built to the
decisions recorded there:

*   **One capture thread**, not a stage-per-thread pipeline. Handing a frame
    between threads costs a queue hop and a synchronisation point each way; at
    30 fps the stages fit inside 33 ms, so a single thread is both simpler and
    lower-latency. B003 measured the stages at about 8 ms total, which is what
    makes that choice safe rather than merely hopeful.
*   **Drop, never queue.** Only the newest frame is published. An unbounded
    queue turns a throughput problem into an ever-growing latency problem — the
    failure where the avatar falls further behind the user the longer the call
    runs. Frames the consumer never collected are counted, not kept.
*   **The UI never blocks on the pipeline.** Readers take the latest frame and
    move on; they cannot apply back-pressure, and a slow or absent reader cannot
    stall capture.

Deliberately free of Qt. The loop is application logic, and keeping it that way
means it can be driven by a benchmark, a test, or a headless run — not only by a
window.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sha256.core.enums import PixelFormat
from sha256.core.types import FaceTrackingResult, RenderedFrame, VideoFrame
from sha256.pipeline.controller import PipelineController
from sha256.utils.logging import get_logger
from sha256.utils.timing import now

__all__ = ["PipelineRunner", "PipelineSnapshot"]

_LOG = get_logger(__name__)

# How many frames of tracking to keep for :meth:`PipelineRunner.capture` — ten
# seconds at 30 fps. Long enough that a person can notice something, decide it
# was wrong, and reach the button; short enough to be a couple of megabytes.
_HISTORY_FRAMES = 300

# How long to wait after a tick that produced nothing. A camera read normally
# blocks until the next frame, so this only matters for sources that return
# immediately (the synthetic backend, a paused device) — where spinning would
# burn a core for no reason.
_IDLE_SLEEP_SECONDS = 0.002


def _as_bgr(data: Any, pixel_format: PixelFormat) -> Any | None:
    """``data`` as the BGR that ``cv2.imwrite`` expects, or ``None`` for a
    layout with no cheap conversion (NV12 is never seen past the camera)."""
    import cv2

    if pixel_format is PixelFormat.BGR24:
        return data
    conversions = {
        PixelFormat.RGB24: cv2.COLOR_RGB2BGR,
        PixelFormat.RGBA32: cv2.COLOR_RGBA2BGR,
        PixelFormat.BGRA32: cv2.COLOR_BGRA2BGR,
        PixelFormat.GRAY8: cv2.COLOR_GRAY2BGR,
    }
    code = conversions.get(pixel_format)
    return None if code is None else cv2.cvtColor(data, code)


class PipelineSnapshot:
    """The newest frame and its context, taken as one consistent set.

    Reading the rendered frame, the source frame, and the tracking result as
    three separate properties would let a consumer pick up pieces of two
    different frames and draw landmarks in the wrong place. They are handed over
    together instead.
    """

    __slots__ = ("frame", "frame_id", "source", "tracking")

    def __init__(
        self,
        frame: RenderedFrame | None,
        source: VideoFrame | None,
        tracking: FaceTrackingResult | None,
        frame_id: int,
    ) -> None:
        self.frame = frame
        self.source = source
        self.tracking = tracking
        self.frame_id = frame_id


class PipelineRunner:
    """Drives :class:`PipelineController` continuously on its own thread."""

    def __init__(
        self,
        controller: PipelineController,
        *,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._controller = controller
        self._on_error = on_error

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # The last few seconds of tracking, kept so that "that looked wrong" can
        # become a file. See :meth:`capture`.
        self._history: deque[FaceTrackingResult] = deque(maxlen=_HISTORY_FRAMES)

        self._latest: PipelineSnapshot | None = None
        self._latest_collected = True
        self._produced = 0
        self._collected = 0
        self._uncollected = 0
        self._last_error: str | None = None
        self._fps = 0.0
        self._fps_window_started = 0.0
        self._fps_window_frames = 0

    # --- lifecycle ---------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_error(self) -> str | None:
        """Message from the failure that stopped the loop, if one did."""
        with self._lock:
            return self._last_error

    def start(self) -> None:
        """Start the components, then the loop. Idempotent."""
        if self.is_running:
            return

        with self._lock:
            self._last_error = None
            self._latest = None
            self._latest_collected = True
            self._produced = 0
            self._collected = 0
            self._uncollected = 0
            self._fps = 0.0
            self._fps_window_started = now()
            self._fps_window_frames = 0

        # Before the thread: a component that fails to start must raise to the
        # caller, where the UI can show it, rather than dying quietly on a
        # background thread.
        self._controller.start()

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="sha256-pipeline", daemon=True)
        self._thread.start()
        _LOG.info("Pipeline loop started")

    def stop(self, *, timeout: float = 2.0) -> None:
        """Stop the loop and the components. Never raises."""
        self._stop_event.set()

        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                # The thread is blocked inside a driver call. Say so plainly
                # rather than pretending the stop was clean.
                _LOG.warning("Pipeline thread did not stop within %.1fs", timeout)

        self._controller.stop()
        _LOG.info("Pipeline loop stopped")

    # --- consumption -------------------------------------------------------

    def latest(self) -> PipelineSnapshot | None:
        """The newest completed frame, or ``None`` if none is ready.

        Counts frames that were produced and never collected: that number is the
        difference between "the pipeline is slow" and "the consumer is slow",
        which look identical in a frame-rate graph.
        """
        with self._lock:
            snapshot = self._latest
            if snapshot is not None and not self._latest_collected:
                self._latest_collected = True
                self._collected += 1
            return snapshot

    def capture(self, directory: Path) -> Path:
        """Write the last few seconds of tracking, and the frame on screen now.

        **The point of this is that "the eyes look wrong" is not a bug report
        and cannot be fixed.** Five rounds of eye work have each been verified
        against a clip recorded on one particular day, and the owner keeps
        seeing something the clip does not contain — a different room, a
        different distance, a moment that never happens on cue. So they press a
        button when it looks wrong instead, and the exact seconds that produced
        it arrive as a file.

        **Tracking, not video.** Landmarks, blendshapes and head pose are a few
        kilobytes a second against a hundred megabytes for the frames, and for
        the photograph engine they fully determine what it drew. Two frames are
        written beside them, both from the tick of the press: ``seen.png``, the
        composited output that was on screen, and ``camera.png``, the camera
        frame it was made from. The second one **is a photograph of the
        person**, and it is written knowingly: the neural engine reads raw
        pixels, so its output cannot be replayed from landmarks, and a jaw band
        or a lost smile can only be measured as output against scene at the
        same instant. Day 11's four complaints had to be guessed at from old
        clips because this file did not exist. The window says which file is
        which when the button is pressed.
        """
        import numpy as np

        with self._lock:
            history = list(self._history)
            snapshot = self._latest

        if not history:
            msg = "nothing has been tracked yet"
            raise RuntimeError(msg)

        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = directory / f"moment-{stamp}"
        target.mkdir(exist_ok=True)

        names = next(
            (r.blendshapes.names for r in history if r.blendshapes is not None),
            (),
        )
        width = len(names)
        # A frame without a face has no landmarks, and its placeholder has to be
        # the shape of a real reading or ``np.stack`` refuses the whole capture.
        # It was ``(1, 3)`` against a real ``(478, 3)``, so any tracking loss in
        # the ten seconds — the very moment the button exists for — turned the
        # press into a warning and the moment was lost. The count comes from the
        # first frame that has one, the way ``width`` does above; ``detected``
        # below says which rows are placeholders, so nothing is invented. With
        # no face in any frame the capture is still written, holding no points
        # at all: a ten-second loss is itself a report worth having on disk.
        count = next((r.landmarks.count for r in history if r.landmarks is not None), 0)
        if count == 0:
            _LOG.warning(
                "No face in any of the last %d frame(s); the capture holds no landmarks",
                len(history),
            )
        np.savez_compressed(
            target / "tracking.npz",
            landmarks=np.stack(
                [
                    r.landmarks.points
                    if r.landmarks is not None
                    else np.zeros((count, 3), dtype=np.float32)
                    for r in history
                ]
            ),
            weights=np.stack(
                [
                    r.blendshapes.weights
                    if r.blendshapes is not None
                    else np.zeros(width, dtype=np.float32)
                    for r in history
                ]
            ),
            pose=np.array(
                [
                    (r.head_pose.yaw, r.head_pose.pitch, r.head_pose.roll)
                    if r.head_pose is not None
                    else (0.0, 0.0, 0.0)
                    for r in history
                ],
                dtype=np.float32,
            ),
            timestamps=np.array([r.metadata.capture_timestamp for r in history], dtype=np.float64),
            detected=np.array([r.face_detected for r in history], dtype=bool),
            names=np.array(names),
        )

        if snapshot is not None:
            import cv2

            # One snapshot, so the two pictures are the same tick: the frame on
            # screen and the camera frame it was composited from.
            for name, picture in (("seen.png", snapshot.frame), ("camera.png", snapshot.source)):
                if picture is None:
                    continue
                bgr = _as_bgr(picture.data, picture.pixel_format)
                if bgr is None:
                    _LOG.warning("%s not written: cannot convert %s", name, picture.pixel_format)
                    continue
                cv2.imwrite(str(target / name), bgr)

        _LOG.info("Captured %d frame(s) of tracking to %s", len(history), target)
        return target

    @property
    def fps(self) -> float:
        """Frames produced per second, over a short rolling window."""
        with self._lock:
            return self._fps

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "produced": self._produced,
                "collected": self._collected,
                "uncollected": self._uncollected,
            }

    # --- the loop ----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._controller.process_once()
            except Exception as exc:
                # A raising pipeline must not take the process with it, and must
                # not spin on the same failure either.
                message = str(exc)
                _LOG.exception("Pipeline loop failed")
                with self._lock:
                    self._last_error = message
                if self._on_error is not None:
                    self._on_error(message)
                return

            if frame is None:
                self._stop_event.wait(_IDLE_SLEEP_SECONDS)
                continue

            self._publish(frame)

    def _publish(self, frame: RenderedFrame) -> None:
        snapshot = PipelineSnapshot(
            frame=frame,
            source=self._controller.last_source_frame,
            tracking=self._controller.last_tracking,
            frame_id=frame.metadata.frame_id,
        )

        with self._lock:
            if snapshot.tracking is not None:
                self._history.append(snapshot.tracking)
            # Overwriting a frame nobody read is the "drop, never queue" rule
            # doing its job. Counted rather than hidden: a consumer that never
            # keeps up is a real problem, just not the pipeline's.
            if self._latest is not None and not self._latest_collected:
                self._uncollected += 1
            self._latest = snapshot
            self._latest_collected = False
            self._produced += 1
            self._update_fps_locked()

    def _update_fps_locked(self) -> None:
        """Rolling throughput. Caller holds the lock."""
        self._fps_window_frames += 1
        elapsed = now() - self._fps_window_started
        if elapsed >= 0.5:
            self._fps = self._fps_window_frames / elapsed
            self._fps_window_started = now()
            self._fps_window_frames = 0
