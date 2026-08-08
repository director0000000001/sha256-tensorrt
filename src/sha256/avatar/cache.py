"""Residency policy for loaded avatars.

The problem this solves: avatar models can be large, VRAM is finite (16 GB on
the target RTX 5070 Ti, shared with the desktop compositor and whatever
video-call application is consuming the virtual camera), and a user switching
between avatars should not pay a full cold load every time.

The policy is deliberately simple and explicit:

*   The **active** avatar is pinned and never evicted. Evicting the avatar
    currently on camera is never the right answer.
*   Other loaded avatars are kept warm in **least-recently-used** order.
*   A load is admitted if it fits within either the count limit or the VRAM
    budget; otherwise LRU entries are evicted until it does.
*   If it still does not fit after evicting everything evictable, the load is
    **refused** rather than attempted. An out-of-memory failure mid-load can
    leave a driver in a bad state; refusing is recoverable and explainable.

VRAM figures are *estimates* until an engine reports measured values. The cache
records which it is using, because a policy driven by an author's guess deserves
less confidence than one driven by measurement.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from sha256.utils.logging import get_logger

__all__ = ["ResidencyCache", "ResidencyDecision", "ResidentEntry"]

_LOG = get_logger(__name__)


@dataclass(slots=True)
class ResidentEntry:
    """Bookkeeping for one resident avatar."""

    avatar_id: str
    vram_mb: int
    """Footprint in MiB. 0 means unknown or CPU-resident."""

    measured: bool = False
    """True when :attr:`vram_mb` came from the engine rather than the manifest."""

    is_active: bool = False


@dataclass(frozen=True, slots=True)
class ResidencyDecision:
    """What the cache decided about admitting an avatar."""

    admitted: bool
    evict: tuple[str, ...] = ()
    reason: str = ""

    @property
    def requires_eviction(self) -> bool:
        return bool(self.evict)


class ResidencyCache:
    """Tracks resident avatars and decides what to evict.

    Holds no engine reference and frees nothing itself. It answers "what should
    be resident?"; the :class:`AvatarManager` performs the actual load and
    unload calls. Keeping the policy free of side effects makes it directly
    unit-testable, which matters because eviction bugs otherwise only appear
    under memory pressure.
    """

    def __init__(self, *, max_resident: int = 2, vram_budget_mb: int = 0) -> None:
        if max_resident < 1:
            msg = "max_resident must be at least 1"
            raise ValueError(msg)
        self._max_resident = max_resident
        self._vram_budget_mb = vram_budget_mb
        # Ordered most-recently-used last.
        self._entries: OrderedDict[str, ResidentEntry] = OrderedDict()

    @property
    def max_resident(self) -> int:
        return self._max_resident

    @property
    def vram_budget_mb(self) -> int:
        return self._vram_budget_mb

    @property
    def used_vram_mb(self) -> int:
        return sum(entry.vram_mb for entry in self._entries.values())

    def resident_ids(self) -> tuple[str, ...]:
        """Resident avatars, least-recently-used first."""
        return tuple(self._entries)

    def is_resident(self, avatar_id: str) -> bool:
        return avatar_id in self._entries

    def entry(self, avatar_id: str) -> ResidentEntry | None:
        return self._entries.get(avatar_id)

    # --- policy ------------------------------------------------------------

    def plan_admission(self, avatar_id: str, estimated_vram_mb: int) -> ResidencyDecision:
        """Decide whether ``avatar_id`` can become resident, and what to evict.

        Pure: it does not mutate the cache. Call :meth:`admit` once the load has
        actually succeeded.
        """
        if avatar_id in self._entries:
            return ResidencyDecision(admitted=True, reason="already resident")

        evictable = [entry.avatar_id for entry in self._entries.values() if not entry.is_active]
        to_evict: list[str] = []

        # Count pressure.
        projected_count = len(self._entries) + 1
        while projected_count > self._max_resident and evictable:
            to_evict.append(evictable.pop(0))
            projected_count -= 1

        if projected_count > self._max_resident:
            return ResidencyDecision(
                admitted=False,
                reason=(
                    f"cannot admit {avatar_id!r}: {len(self._entries)} resident, "
                    f"limit {self._max_resident}, and no avatar is evictable"
                ),
            )

        # VRAM pressure. A budget of 0 means unbounded — the honest reading when
        # nothing has been measured yet.
        if self._vram_budget_mb > 0 and estimated_vram_mb > 0:
            projected = self._projected_vram(to_evict) + estimated_vram_mb
            while projected > self._vram_budget_mb and evictable:
                victim = evictable.pop(0)
                to_evict.append(victim)
                projected = self._projected_vram(to_evict) + estimated_vram_mb

            if projected > self._vram_budget_mb:
                return ResidencyDecision(
                    admitted=False,
                    reason=(
                        f"cannot admit {avatar_id!r}: needs ~{estimated_vram_mb} MiB, "
                        f"only ~{self._vram_budget_mb - self._projected_vram(to_evict)} MiB "
                        f"of the {self._vram_budget_mb} MiB budget is reclaimable"
                    ),
                )

        return ResidencyDecision(admitted=True, evict=tuple(to_evict))

    def _projected_vram(self, evicting: list[str]) -> int:
        pending = set(evicting)
        return sum(
            entry.vram_mb for avatar_id, entry in self._entries.items() if avatar_id not in pending
        )

    # --- mutation ----------------------------------------------------------

    def admit(self, avatar_id: str, vram_mb: int, *, measured: bool = False) -> None:
        """Record a successful load."""
        self._entries[avatar_id] = ResidentEntry(
            avatar_id=avatar_id, vram_mb=max(0, vram_mb), measured=measured
        )
        self._entries.move_to_end(avatar_id)

    def evict(self, avatar_id: str) -> None:
        """Record an unload."""
        self._entries.pop(avatar_id, None)

    def touch(self, avatar_id: str) -> None:
        """Mark an avatar as recently used, protecting it from imminent eviction."""
        if avatar_id in self._entries:
            self._entries.move_to_end(avatar_id)

    def set_active(self, avatar_id: str | None) -> None:
        """Pin ``avatar_id`` as active (and unpin any previous one)."""
        for entry in self._entries.values():
            entry.is_active = entry.avatar_id == avatar_id
        if avatar_id is not None:
            self.touch(avatar_id)

    def update_measurement(self, avatar_id: str, vram_mb: int) -> None:
        """Replace an estimate with an engine-reported measurement."""
        entry = self._entries.get(avatar_id)
        if entry is None:
            return
        if not entry.measured or entry.vram_mb != vram_mb:
            _LOG.debug(
                "VRAM for %s: %d MiB (%s) -> %d MiB (measured)",
                avatar_id,
                entry.vram_mb,
                "measured" if entry.measured else "estimated",
                vram_mb,
            )
        entry.vram_mb = max(0, vram_mb)
        entry.measured = True

    def clear(self) -> None:
        self._entries.clear()
