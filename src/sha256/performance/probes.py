"""System resource probes.

Only the *contract* is defined here, plus a probe that honestly reports "I do
not know". CPU, GPU, and VRAM sampling need dependencies (``psutil``,
``pynvml``) that are not warranted yet, and inventing numbers would be worse
than having none — see rule 12 in the project brief.

When a real probe is added it registers here and the debug panel picks it up
with no other change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["NullSystemProbe", "SystemProbe", "SystemSample"]


@dataclass(frozen=True, slots=True)
class SystemSample:
    """One reading of host resource usage.

    Every field is optional. ``None`` means *not measured*, which is a different
    statement from ``0.0`` meaning *measured as zero* — a distinction the UI
    reflects by showing "n/a" rather than a misleading number.
    """

    cpu_percent: float | None = None
    process_memory_mb: float | None = None
    gpu_utilisation_percent: float | None = None
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None
    gpu_name: str | None = None

    @property
    def vram_free_mb(self) -> float | None:
        if self.vram_used_mb is None or self.vram_total_mb is None:
            return None
        return self.vram_total_mb - self.vram_used_mb

    @property
    def has_gpu_data(self) -> bool:
        return self.vram_total_mb is not None


@runtime_checkable
class SystemProbe(Protocol):
    """Samples host CPU/GPU/memory usage."""

    @property
    def is_available(self) -> bool:
        """Whether this probe can actually read anything on this machine."""
        ...

    def sample(self) -> SystemSample:
        """Take a reading. Must not raise; report ``None`` fields instead."""
        ...


class NullSystemProbe:
    """Reports nothing. The honest default until a real probe is implemented."""

    @property
    def is_available(self) -> bool:
        return False

    def sample(self) -> SystemSample:
        return SystemSample()
