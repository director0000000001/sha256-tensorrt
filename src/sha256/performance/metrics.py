"""Metric collection primitives.

The result types (:class:`MetricSummary`, :class:`PerformanceSnapshot`) live in
:mod:`sha256.core.metrics` because the monitor protocol names them; this module
holds the machinery that produces them.

One deliberate choice worth stating: **rolling windows, not cumulative
averages.** A lifetime mean hides the thing that matters in a real-time system
— that the last few seconds got worse. Every statistic here describes a bounded
recent window.
"""

from __future__ import annotations

from collections import deque

from sha256.core.metrics import MetricSummary, PerformanceSnapshot

__all__ = ["FrameCounter", "MetricSummary", "PerformanceSnapshot", "RollingWindow"]


class RollingWindow:
    """Fixed-capacity sample buffer with percentile queries.

    Percentiles sort on read rather than maintaining an ordered structure on
    write. Writes happen on the capture thread every frame and must stay cheap;
    reads happen when the debug panel refreshes, a few times a second, where the
    sort of a few hundred floats is irrelevant.
    """

    __slots__ = ("_capacity", "_name", "_samples")

    def __init__(self, name: str, capacity: int = 240) -> None:
        if capacity < 1:
            msg = "capacity must be at least 1"
            raise ValueError(msg)
        self._name = name
        self._capacity = capacity
        self._samples: deque[float] = deque(maxlen=capacity)

    @property
    def name(self) -> str:
        return self._name

    @property
    def capacity(self) -> int:
        return self._capacity

    def add(self, value: float) -> None:
        self._samples.append(value)

    def clear(self) -> None:
        self._samples.clear()

    def __len__(self) -> int:
        return len(self._samples)

    def summarise(self) -> MetricSummary:
        if not self._samples:
            return MetricSummary.empty(self._name)

        ordered = sorted(self._samples)
        count = len(ordered)
        return MetricSummary(
            name=self._name,
            count=count,
            mean=sum(ordered) / count,
            minimum=ordered[0],
            maximum=ordered[-1],
            p50=_percentile(ordered, 0.50),
            p95=_percentile(ordered, 0.95),
            p99=_percentile(ordered, 0.99),
        )


def _percentile(ordered: list[float], fraction: float) -> float:
    """Nearest-rank percentile over a pre-sorted list.

    Nearest-rank rather than interpolated: every reported value is one that
    actually occurred, which matters when the number is used to justify a
    performance claim.
    """
    if not ordered:
        return 0.0
    return ordered[round(fraction * (len(ordered) - 1))]


class FrameCounter:
    """Counts frames and derives throughput over a rolling time window."""

    __slots__ = ("_dropped", "_timestamps", "_total")

    def __init__(self, capacity: int = 240) -> None:
        self._timestamps: deque[float] = deque(maxlen=capacity)
        self._total = 0
        self._dropped = 0

    def record(self, timestamp: float) -> None:
        self._timestamps.append(timestamp)
        self._total += 1

    def record_dropped(self, count: int = 1) -> None:
        self._dropped += count

    @property
    def total(self) -> int:
        return self._total

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def fps(self) -> float:
        """Frames per second across the retained window.

        Measured from timestamps rather than counted per wall-clock second, so
        the value stays meaningful when the window covers a fraction of a second
        or several seconds.
        """
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0.0:
            return 0.0
        return (len(self._timestamps) - 1) / span

    def clear(self) -> None:
        self._timestamps.clear()
        self._total = 0
        self._dropped = 0
