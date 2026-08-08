"""Performance measurement result types.

These live in :mod:`sha256.core` rather than in :mod:`sha256.performance` for
the same reason the avatar manifest does: they are *data contracts* crossing a
component boundary, and the :class:`~sha256.core.interfaces.PerformanceMonitor`
protocol needs to name them in its signature.

The collection machinery that produces them — rolling windows, counters,
locking — stays in :mod:`sha256.performance`, which is implementation and is
replaceable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["MetricSummary", "PerformanceSnapshot"]


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Statistics over one rolling window. All durations are in seconds.

    Percentiles are reported alongside the mean because smooth video is
    governed by the slow frames: a 30 fps mean with a p99 of 80 ms stutters
    visibly, and the mean alone cannot show that.
    """

    name: str
    count: int
    mean: float
    minimum: float
    maximum: float
    p50: float
    p95: float
    p99: float

    @property
    def mean_ms(self) -> float:
        return self.mean * 1000.0

    @property
    def p95_ms(self) -> float:
        return self.p95 * 1000.0

    @property
    def p99_ms(self) -> float:
        return self.p99 * 1000.0

    @classmethod
    def empty(cls, name: str) -> MetricSummary:
        return cls(
            name=name, count=0, mean=0.0, minimum=0.0, maximum=0.0, p50=0.0, p95=0.0, p99=0.0
        )


@dataclass(frozen=True, slots=True)
class PerformanceSnapshot:
    """Point-in-time copy of every metric.

    Immutable so the UI thread can read it without holding a lock against the
    capture thread that produced it.
    """

    durations: dict[str, MetricSummary] = field(default_factory=dict)
    fps: dict[str, float] = field(default_factory=dict)
    frame_totals: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    gauges: dict[str, float] = field(default_factory=dict)
    captured_at: float = 0.0

    def duration(self, stage: str) -> MetricSummary:
        """Summary for ``stage``, or an empty one if it was never measured."""
        return self.durations.get(stage, MetricSummary.empty(stage))

    def is_empty(self) -> bool:
        return not (self.durations or self.fps or self.gauges)
