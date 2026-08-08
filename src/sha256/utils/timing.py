"""Clock helpers.

Everything in the pipeline measures time with :func:`time.perf_counter`, a
monotonic high-resolution clock. Wall-clock time (``time.time``) can step
backwards across NTP corrections and DST changes, which would produce negative
latencies and corrupt rolling statistics. Wall-clock is used only for log
records meant for humans.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["Stopwatch", "elapsed", "now"]


def now() -> float:
    """Current monotonic time in seconds. The one clock the pipeline uses."""
    return time.perf_counter()


@contextmanager
def elapsed(sink: list[float]) -> Iterator[None]:
    """Append the duration of the enclosed block to ``sink``."""
    start = now()
    try:
        yield
    finally:
        sink.append(now() - start)


class Stopwatch:
    """Reusable interval timer.

    Kept as a small explicit object rather than a decorator so that timing can
    span a loop iteration and be read at several points within it.
    """

    __slots__ = ("_start",)

    def __init__(self) -> None:
        self._start = now()

    def reset(self) -> None:
        self._start = now()

    def read(self) -> float:
        """Seconds since construction or the last :meth:`reset`."""
        return now() - self._start

    def lap(self) -> float:
        """Read the elapsed time and reset in one operation."""
        current = now()
        delta = current - self._start
        self._start = current
        return delta
