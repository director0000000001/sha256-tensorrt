"""Tests for the VRAM residency policy.

Eviction bugs normally surface only under real memory pressure, on real
hardware, at the worst moment. The cache is deliberately side-effect free so
that pressure can be simulated exactly here instead.
"""

from __future__ import annotations

import pytest

from sha256.avatar.cache import ResidencyCache


@pytest.fixture
def cache() -> ResidencyCache:
    return ResidencyCache(max_resident=2, vram_budget_mb=4096)


class TestConstruction:
    def test_requires_at_least_one_slot(self):
        with pytest.raises(ValueError, match="at least 1"):
            ResidencyCache(max_resident=0)


class TestCountPressure:
    def test_admits_while_under_the_limit(self, cache: ResidencyCache):
        decision = cache.plan_admission("a", 1000)
        assert decision.admitted
        assert decision.evict == ()

    def test_evicts_least_recently_used_when_full(self, cache: ResidencyCache):
        cache.admit("a", 1000)
        cache.admit("b", 1000)

        decision = cache.plan_admission("c", 1000)

        assert decision.admitted
        assert decision.evict == ("a",), "oldest entry should be chosen"

    def test_touch_protects_from_eviction(self, cache: ResidencyCache):
        cache.admit("a", 1000)
        cache.admit("b", 1000)
        cache.touch("a")

        assert cache.plan_admission("c", 1000).evict == ("b",)

    def test_already_resident_is_a_no_op(self, cache: ResidencyCache):
        cache.admit("a", 1000)
        decision = cache.plan_admission("a", 1000)

        assert decision.admitted
        assert decision.evict == ()


class TestActivePinning:
    def test_active_avatar_is_never_evicted(self, cache: ResidencyCache):
        """Evicting the avatar currently on camera is never the right answer."""
        cache.admit("a", 1000)
        cache.admit("b", 1000)
        cache.set_active("a")

        assert cache.plan_admission("c", 1000).evict == ("b",)

    def test_refuses_when_only_the_active_avatar_could_be_evicted(self):
        cache = ResidencyCache(max_resident=1, vram_budget_mb=4096)
        cache.admit("a", 1000)
        cache.set_active("a")

        decision = cache.plan_admission("b", 1000)

        assert not decision.admitted
        assert "no avatar is evictable" in decision.reason

    def test_setting_active_moves_the_pin(self, cache: ResidencyCache):
        cache.admit("a", 1000)
        cache.admit("b", 1000)
        cache.set_active("a")
        cache.set_active("b")

        entry_a = cache.entry("a")
        entry_b = cache.entry("b")
        assert entry_a is not None and not entry_a.is_active
        assert entry_b is not None and entry_b.is_active


class TestVramPressure:
    def test_evicts_until_the_budget_fits(self):
        cache = ResidencyCache(max_resident=8, vram_budget_mb=4096)
        cache.admit("a", 2000)
        cache.admit("b", 1500)

        decision = cache.plan_admission("c", 3000)

        assert decision.admitted
        assert set(decision.evict) == {"a", "b"}

    def test_refuses_a_load_that_cannot_fit(self):
        """Refusing beats attempting: a mid-load OOM can wedge the driver."""
        cache = ResidencyCache(max_resident=8, vram_budget_mb=4096)
        cache.admit("a", 1000)
        cache.set_active("a")

        decision = cache.plan_admission("huge", 8000)

        assert not decision.admitted
        assert "8000" in decision.reason or "budget" in decision.reason

    def test_zero_budget_means_unbounded(self):
        """No budget configured means nothing measured — not a budget of zero."""
        cache = ResidencyCache(max_resident=8, vram_budget_mb=0)
        cache.admit("a", 99_999)

        assert cache.plan_admission("b", 99_999).admitted

    def test_unknown_estimate_does_not_trigger_vram_eviction(self):
        cache = ResidencyCache(max_resident=8, vram_budget_mb=1024)
        cache.admit("a", 1000)

        decision = cache.plan_admission("b", 0)

        assert decision.admitted
        assert decision.evict == ()


class TestAccounting:
    def test_used_vram_sums_entries(self, cache: ResidencyCache):
        cache.admit("a", 1000)
        cache.admit("b", 512)
        assert cache.used_vram_mb == 1512

    def test_eviction_frees_accounting(self, cache: ResidencyCache):
        cache.admit("a", 1000)
        cache.evict("a")

        assert cache.used_vram_mb == 0
        assert not cache.is_resident("a")

    def test_measurement_replaces_the_estimate(self, cache: ResidencyCache):
        cache.admit("a", 1000)
        cache.update_measurement("a", 2750)

        entry = cache.entry("a")
        assert entry is not None
        assert entry.vram_mb == 2750
        assert entry.measured is True

    def test_measuring_an_absent_avatar_is_ignored(self, cache: ResidencyCache):
        cache.update_measurement("ghost", 500)
        assert cache.entry("ghost") is None

    def test_resident_ids_are_ordered_least_recently_used_first(self, cache: ResidencyCache):
        cache.admit("a", 1)
        cache.admit("b", 1)
        cache.touch("a")

        assert cache.resident_ids() == ("b", "a")
