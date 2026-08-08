"""Performance monitor implementations.

:class:`InProcessPerformanceMonitor` is the real one used by the application.
:class:`NullPerformanceMonitor` satisfies the protocol while doing nothing, so
monitoring can be switched off without any call site growing an ``if``.

Thread safety
-------------
Metrics are written from the capture/processing thread and read from the Qt UI
thread. A single lock guards all state. It is held only for the duration of a
``deque.append`` or a snapshot build — never across user code — so contention
stays negligible at video frame rates.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext

from sha256.core.enums import PipelineStage
from sha256.core.metrics import PerformanceSnapshot
from sha256.performance.metrics import FrameCounter, RollingWindow
from sha256.utils.timing import now

__all__ = ["InProcessPerformanceMonitor", "NullPerformanceMonitor"]


class InProcessPerformanceMonitor:
    """Collects timings, throughput, and gauges in memory.

    Nothing is written to disk or sent anywhere. Metrics exist for the local
    debug panel and for benchmark runs that explicitly export them.
    """

    def __init__(self, *, window_size: int = 240, enabled: bool = True) -> None:
        self._enabled = enabled
        self._window_size = window_size
        self._lock = threading.Lock()
        self._durations: dict[str, RollingWindow] = {}
        self._counters: dict[str, FrameCounter] = {}
        self._gauges: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    # --- recording ---------------------------------------------------------

    def measure(self, stage: PipelineStage) -> AbstractContextManager[None]:
        if not self._enabled:
            return nullcontext()
        return self._measure(stage)

    @contextmanager
    def _measure(self, stage: PipelineStage) -> Iterator[None]:
        start = now()
        try:
            yield
        finally:
            # Recorded in `finally` so a failing stage still contributes its
            # timing. Otherwise the metrics look healthiest exactly when the
            # pipeline is breaking.
            self.record_duration(stage, now() - start)

    def record_duration(self, stage: PipelineStage, seconds: float) -> None:
        if not self._enabled:
            return
        key = stage.value
        with self._lock:
            window = self._durations.get(key)
            if window is None:
                window = RollingWindow(key, capacity=self._window_size)
                self._durations[key] = window
            window.add(seconds)

    def record_frame(self, stage: PipelineStage) -> None:
        if not self._enabled:
            return
        timestamp = now()
        with self._lock:
            self._counter(stage.value).record(timestamp)

    def record_dropped(self, stage: PipelineStage, count: int = 1) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._counter(stage.value).record_dropped(count)

    def record_value(self, name: str, value: float) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._gauges[name] = value

    def _counter(self, key: str) -> FrameCounter:
        """Fetch or create a counter. Caller must hold the lock."""
        counter = self._counters.get(key)
        if counter is None:
            counter = FrameCounter(capacity=self._window_size)
            self._counters[key] = counter
        return counter

    # --- reading -----------------------------------------------------------

    def snapshot(self) -> PerformanceSnapshot:
        with self._lock:
            durations = {key: window.summarise() for key, window in self._durations.items()}
            fps = {key: counter.fps for key, counter in self._counters.items()}
            totals = {key: counter.total for key, counter in self._counters.items()}
            dropped = {key: counter.dropped for key, counter in self._counters.items()}
            gauges = dict(self._gauges)

        return PerformanceSnapshot(
            durations=durations,
            fps=fps,
            frame_totals=totals,
            dropped=dropped,
            gauges=gauges,
            captured_at=now(),
        )

    def reset(self) -> None:
        with self._lock:
            self._durations.clear()
            self._counters.clear()
            self._gauges.clear()


class NullPerformanceMonitor:
    """No-op monitor. Every method is a cheap return."""

    @property
    def enabled(self) -> bool:
        return False

    def measure(self, stage: PipelineStage) -> AbstractContextManager[None]:
        del stage
        return nullcontext()

    def record_duration(self, stage: PipelineStage, seconds: float) -> None: ...

    def record_frame(self, stage: PipelineStage) -> None: ...

    def record_dropped(self, stage: PipelineStage, count: int = 1) -> None: ...

    def record_value(self, name: str, value: float) -> None: ...

    def snapshot(self) -> PerformanceSnapshot:
        return PerformanceSnapshot(captured_at=now())

    def reset(self) -> None: ...
