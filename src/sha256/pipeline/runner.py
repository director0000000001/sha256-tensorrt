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

**Two threads since 2026-10-08 ("The fast version", step 1).** The neural
engine changed the arithmetic above: the GPU draws a face in about 26 ms and
the tracker reads the next frame in about 13 ms on the CPU, and on one thread
the two wait for each other -- 42-55 ms a frame, 18-23 fps on the owner's
machine. So the loop is now two halves of :class:`PipelineController` on two
threads: the *capture* thread reads the camera and tracks the face
(:meth:`capture_and_track`), the *render* thread draws, conditions and sends
(:meth:`render_tracked`). Between them is a **mailbox of one**: the capture
thread overwrites whatever the render thread has not taken yet (counted as
``dropped_at_capture``), so the render thread always starts on the newest
tracked frame and nothing queues. The frame and its tracking travel together
(:class:`TrackedFrame`) and the snapshot is built from that pair, never from a
"last" the other thread may have moved on. What it costs is one frame of
latency: the render of frame N overlaps the tracking of N+1. ``overlap=False``
keeps the single thread, and a controller without the two halves (a test
double) gets it regardless.

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
from sha256.pipeline.controller import PipelineController, TrackedFrame
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
        overlap: bool = True,
    ) -> None:
        self._controller = controller
        self._on_error = on_error
        # Two threads (the module docstring) when asked and when the
        # controller has the two halves; a test double with only
        # ``process_once`` runs the single thread it always did.
        self._overlap = overlap and all(
            callable(getattr(controller, name, None))
            for name in ("capture_and_track", "render_tracked")
        )

        self._thread: threading.Thread | None = None
        self._render_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        # The mailbox of one between the two threads, and its own condition
        # so a reader of ``latest`` never waits on a frame hand-over.
        self._hand = threading.Condition()
        self._mailbox: TrackedFrame | None = None
        self._dropped_at_capture = 0

        # The last few seconds of tracking, kept so that "that looked wrong" can
        # become a file. See :meth:`capture`.
        self._history: deque[FaceTrackingResult] = deque(maxlen=_HISTORY_FRAMES)
        # Day 16: who wants every published frame as it happens -- see
        # :meth:`add_observer`.
        self._observers: list[Callable[[PipelineSnapshot], None]] = []

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
        return any(t is not None and t.is_alive() for t in (self._thread, self._render_thread))

    @property
    def overlapping(self) -> bool:
        """Whether the loop runs its two halves on two threads."""
        return self._overlap

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
        with self._hand:
            self._mailbox = None
            self._dropped_at_capture = 0

        # Before the thread: a component that fails to start must raise to the
        # caller, where the UI can show it, rather than dying quietly on a
        # background thread.
        self._controller.start()

        self._stop_event.clear()
        if self._overlap:
            self._thread = threading.Thread(
                target=self._run_capture, name="sha256-capture", daemon=True
            )
            self._render_thread = threading.Thread(
                target=self._run_render, name="sha256-render", daemon=True
            )
            self._render_thread.start()
            self._thread.start()
            _LOG.info("Pipeline loop started on two threads (capture + render)")
        else:
            self._thread = threading.Thread(target=self._run, name="sha256-pipeline", daemon=True)
            self._thread.start()
            _LOG.info("Pipeline loop started")

    def stop(self, *, timeout: float = 2.0) -> None:
        """Stop the loop and the components. Never raises."""
        self._stop_event.set()
        with self._hand:
            self._hand.notify_all()

        threads = (self._thread, self._render_thread)
        self._thread = self._render_thread = None
        for thread in threads:
            if thread is not None and thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    # The thread is blocked inside a driver call. Say so plainly
                    # rather than pretending the stop was clean.
                    _LOG.warning("%s did not stop within %.1fs", thread.name, timeout)

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

    def add_observer(self, observer: Callable[[PipelineSnapshot], None]) -> None:
        """Call ``observer`` with every snapshot as it is published, **on the
        pipeline thread**, after the frame is in :meth:`latest`.

        For a consumer that must see every frame rather than the newest one
        -- "Calibrate me" (day 16) records the whole session's frames and
        tracking, and a poll of :meth:`latest` would miss the frames between
        polls. The rules that follow from running on the pipeline thread:
        be quick (a slow observer lowers the frame rate for everyone), copy
        the source frame's buffer before keeping it (it is borrowed, see
        :class:`VideoFrame`), and do not raise -- an exception is logged and
        swallowed, the loop must not die for a listener."""
        with self._lock:
            if observer not in self._observers:
                self._observers.append(observer)

    def remove_observer(self, observer: Callable[[PipelineSnapshot], None]) -> None:
        """Stop calling ``observer``. A call already in progress on the
        pipeline thread finishes; none starts after this returns."""
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)

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

        **The anchor too (day 15).** The neural engine drives the portrait
        relative to a resting face it learned from the first frames of the
        session, and a replay that has to learn its own from the capture's
        frames is a different session -- or none, when the ten seconds were
        all head-down (his capture of 2026-09-11 08:10). So the engine's
        frozen anchor goes in beside the tracking, as ``anchor_*`` arrays
        (see ``LivePortraitEngine.anchor_state``); an engine without one
        (the photograph engine) writes nothing extra.
        """
        import numpy as np

        with self._lock:
            history = list(self._history)
            snapshot = self._latest
        anchor = self._anchor_state()

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
            **{f"anchor_{key}": value for key, value in (anchor or {}).items()},
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

        _LOG.info(
            "Captured %d frame(s) of tracking to %s (%s)",
            len(history),
            target,
            "with the avatar's anchor" if anchor else "no anchor to carry",
        )
        return target

    def _anchor_state(self) -> dict[str, Any] | None:
        """The active avatar's frozen anchor, from an engine that has one --
        ``anchor_state`` is the neural engine's, not part of the engine
        protocol, so this asks and takes ``None`` for an answer from any
        engine, controller or test double without it. The avatar's id rides
        along so a replay loads the same one."""
        import numpy as np

        avatars = getattr(getattr(self._controller, "components", None), "avatars", None)
        engine = getattr(avatars, "engine", None)
        avatar_id = getattr(avatars, "active_id", None)
        snapshot = getattr(engine, "anchor_state", None)
        if engine is None or avatar_id is None or snapshot is None:
            return None
        try:
            state = snapshot(avatar_id)
        except Exception as exc:
            _LOG.warning("The avatar's anchor could not be read for the capture: %s", exc)
            return None
        if not state:
            return None
        state = dict(state)
        state["avatar"] = np.array(str(avatar_id))
        return state

    @property
    def fps(self) -> float:
        """Frames produced per second, over a short rolling window."""
        with self._lock:
            return self._fps

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            produced, collected, uncollected = self._produced, self._collected, self._uncollected
        with self._hand:
            dropped = self._dropped_at_capture
        return {
            "produced": produced,
            "collected": collected,
            "uncollected": uncollected,
            # Tracked frames the render thread never got to, because a newer
            # one arrived first (two-thread loop only; always 0 on one).
            "dropped_at_capture": dropped,
        }

    # --- the loop ----------------------------------------------------------

    def _run(self) -> None:
        """The single thread: one frame, all stages, then the next."""
        while not self._stop_event.is_set():
            try:
                frame = self._controller.process_once()
            except Exception as exc:
                self._fail(exc)
                return

            if frame is None:
                self._stop_event.wait(_IDLE_SLEEP_SECONDS)
                continue

            self._publish(frame, self._controller.last_source_frame, self._controller.last_tracking)

    def _run_capture(self) -> None:
        """The capture thread: read, track, into the mailbox -- newest wins."""
        while not self._stop_event.is_set():
            try:
                tracked = self._controller.capture_and_track()
            except Exception as exc:
                self._fail(exc)
                return

            if tracked is None:
                self._stop_event.wait(_IDLE_SLEEP_SECONDS)
                continue

            with self._hand:
                if self._mailbox is not None:
                    self._dropped_at_capture += 1
                self._mailbox = tracked
                self._hand.notify()

    def _run_render(self) -> None:
        """The render thread: the newest tracked frame drawn, conditioned,
        sent and published; nothing queues behind it."""
        while not self._stop_event.is_set():
            with self._hand:
                if self._mailbox is None:
                    self._hand.wait(0.1)
                tracked, self._mailbox = self._mailbox, None
            if tracked is None:
                continue

            try:
                frame = self._controller.render_tracked(tracked)
            except Exception as exc:
                self._fail(exc)
                return

            if frame is not None:
                self._publish(frame, tracked.frame, tracked.tracking)

    def _fail(self, exc: Exception) -> None:
        """The loop stops on an error, on either thread: the other side sees
        the black screen (day 15), the window hears why, both threads end.
        A raising pipeline must not take the process with it, and must not
        spin on the same failure either."""
        message = str(exc)
        _LOG.exception("Pipeline loop failed")
        # Day 15: the other side sees a dead camera, not the last avatar
        # frame frozen for the rest of the call. Asked for, not assumed --
        # a controller without it (a test double) simply has nothing to
        # publish.
        publish_failure = getattr(self._controller, "publish_failure", None)
        if publish_failure is not None:
            try:
                publish_failure(message)
            except Exception:
                _LOG.exception("The black screen could not be published either")
        with self._lock:
            self._last_error = message
        # The other thread ends too: the loop is one loop, on two threads.
        self._stop_event.set()
        with self._hand:
            self._hand.notify_all()
        if self._on_error is not None:
            self._on_error(message)

    def _publish(
        self,
        frame: RenderedFrame,
        source: VideoFrame | None = None,
        tracking: FaceTrackingResult | None = None,
    ) -> None:
        if source is None and tracking is None:
            # The single-thread path, and the tests that publish by hand:
            # the controller's "last" is this frame's own.
            source = self._controller.last_source_frame
            tracking = self._controller.last_tracking
        snapshot = PipelineSnapshot(
            frame=frame,
            source=source,
            tracking=tracking,
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
            observers = list(self._observers)

        # Outside the lock: an observer may take a while, and readers must
        # not wait on it; the copy above is what makes add/remove safe.
        for observer in observers:
            try:
                observer(snapshot)
            except Exception:
                _LOG.exception("A frame observer failed; the loop goes on")

    def _update_fps_locked(self) -> None:
        """Rolling throughput. Caller holds the lock."""
        self._fps_window_frames += 1
        elapsed = now() - self._fps_window_started
        if elapsed >= 0.5:
            self._fps = self._fps_window_frames / elapsed
            self._fps_window_started = now()
            self._fps_window_frames = 0
