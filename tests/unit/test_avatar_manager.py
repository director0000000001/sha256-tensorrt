"""Tests for avatar lifecycle coordination."""

from __future__ import annotations

from pathlib import Path

import pytest

from sha256.avatar.cache import ResidencyCache
from sha256.avatar.engine_null import NullAvatarEngine
from sha256.avatar.manager import AvatarManager, AvatarStatus
from sha256.avatar.repository import FilesystemAvatarRepository
from sha256.core.enums import AvatarState
from sha256.core.errors import (
    AvatarLoadError,
    AvatarNotFoundError,
    InsufficientResourcesError,
)
from tests.conftest import write_avatar


@pytest.fixture
def manager(avatar_library: Path) -> AvatarManager:
    engine = NullAvatarEngine()
    engine.start()
    manager = AvatarManager(
        FilesystemAvatarRepository(avatar_library),
        ResidencyCache(max_resident=2, vram_budget_mb=4096),
        engine=engine,
    )
    manager.refresh()
    return manager


class TestDiscovery:
    def test_lists_discovered_avatars(self, manager: AvatarManager):
        assert {a.id for a in manager.list_avatars()} == {"avatar_001", "avatar_002"}

    def test_new_avatars_start_as_discovered(self, manager: AvatarManager):
        assert manager.state_of("avatar_001") is AvatarState.DISCOVERED

    def test_refresh_drops_deleted_avatars(self, manager: AvatarManager, avatar_library: Path):
        manager.select("avatar_001")
        for child in (avatar_library / "avatar_001").iterdir():
            child.unlink()
        (avatar_library / "avatar_001").rmdir()

        manager.refresh()

        assert manager.active_id is None
        assert manager.state_of("avatar_001") is AvatarState.DISCOVERED


class TestLoading:
    def test_load_makes_the_avatar_resident(self, manager: AvatarManager):
        manager.load("avatar_001")

        assert manager.state_of("avatar_001") is AvatarState.LOADED
        assert manager.engine is not None
        assert "avatar_001" in manager.engine.loaded_avatars

    def test_loading_twice_is_idempotent(self, manager: AvatarManager):
        manager.load("avatar_001")
        manager.load("avatar_001")

        assert manager.engine is not None
        assert list(manager.engine.loaded_avatars) == ["avatar_001"]

    def test_unknown_avatar_raises(self, manager: AvatarManager):
        with pytest.raises(AvatarNotFoundError):
            manager.load("avatar_missing")

    def test_load_without_engine_raises(self, avatar_library: Path):
        manager = AvatarManager(
            FilesystemAvatarRepository(avatar_library), ResidencyCache(), engine=None
        )
        with pytest.raises(AvatarLoadError, match="no avatar engine"):
            manager.load("avatar_001")

    def test_engine_mismatch_is_refused(self, manager: AvatarManager, avatar_library: Path):
        """A neural avatar must not silently fall back to the placeholder engine."""
        write_avatar(
            avatar_library,
            "avatar_007",
            engine={"engine_id": "some_future_engine", "estimated_vram_mb": 0},
        )
        manager.refresh()

        with pytest.raises(AvatarLoadError, match="cannot drive"):
            manager.load("avatar_007")
        assert manager.state_of("avatar_007") is AvatarState.ERROR

    def test_refusal_when_over_budget(self, avatar_library: Path):
        write_avatar(
            avatar_library,
            "avatar_big",
            engine={"engine_id": "null", "estimated_vram_mb": 90_000},
        )
        engine = NullAvatarEngine()
        engine.start()
        manager = AvatarManager(
            FilesystemAvatarRepository(avatar_library),
            ResidencyCache(max_resident=2, vram_budget_mb=2048),
            engine=engine,
        )
        manager.refresh()

        with pytest.raises(InsufficientResourcesError):
            manager.load("avatar_big")
        assert manager.state_of("avatar_big") is AvatarState.ERROR


class TestSelection:
    def test_select_loads_and_activates(self, manager: AvatarManager):
        descriptor = manager.select("avatar_001")

        assert descriptor.id == "avatar_001"
        assert manager.active_id == "avatar_001"
        assert manager.state_of("avatar_001") is AvatarState.ACTIVE

    def test_switching_keeps_the_previous_avatar_warm(self, manager: AvatarManager):
        """Switching back should be instant, so the old avatar stays resident."""
        manager.select("avatar_001")
        manager.select("avatar_002")

        assert manager.active_id == "avatar_002"
        assert manager.state_of("avatar_001") is AvatarState.LOADED
        assert manager.engine is not None
        assert set(manager.engine.loaded_avatars) == {"avatar_001", "avatar_002"}

    def test_deselect_leaves_the_avatar_resident(self, manager: AvatarManager):
        manager.select("avatar_001")
        manager.deselect()

        assert manager.active_id is None
        assert manager.state_of("avatar_001") is AvatarState.LOADED

    def test_active_returns_the_descriptor(self, manager: AvatarManager):
        manager.select("avatar_002")
        active = manager.active

        assert active is not None
        assert active.id == "avatar_002"


class TestUnloading:
    def test_unload_clears_state_and_engine(self, manager: AvatarManager):
        manager.select("avatar_001")
        manager.unload("avatar_001")

        assert manager.active_id is None
        assert manager.state_of("avatar_001") is AvatarState.DISCOVERED
        assert manager.engine is not None
        assert list(manager.engine.loaded_avatars) == []

    def test_unloading_an_unloaded_avatar_is_safe(self, manager: AvatarManager):
        manager.unload("avatar_001")

    def test_unload_all_clears_everything(self, manager: AvatarManager):
        manager.select("avatar_001")
        manager.load("avatar_002")
        manager.unload_all()

        assert manager.engine is not None
        assert list(manager.engine.loaded_avatars) == []


class TestObservers:
    def test_observers_receive_state_changes(self, manager: AvatarManager):
        seen: list[AvatarStatus] = []
        manager.add_observer(seen.append)

        manager.select("avatar_001")

        assert [s.state for s in seen][-1] is AvatarState.ACTIVE
        assert all(s.avatar_id == "avatar_001" for s in seen)

    def test_a_failing_observer_does_not_break_the_load(self, manager: AvatarManager):
        """Observers are usually UI code; their bugs must not abort a good load."""

        def broken(_: AvatarStatus) -> None:
            raise RuntimeError("observer blew up")

        manager.add_observer(broken)
        manager.select("avatar_001")

        assert manager.active_id == "avatar_001"

    def test_observers_can_be_removed(self, manager: AvatarManager):
        seen: list[AvatarStatus] = []
        manager.add_observer(seen.append)
        manager.remove_observer(seen.append)

        manager.select("avatar_001")
        assert seen == []


class TestStatuses:
    def test_statuses_cover_every_avatar(self, manager: AvatarManager):
        statuses = manager.statuses()
        assert {s.avatar_id for s in statuses} == {"avatar_001", "avatar_002"}
        assert all(s.is_usable for s in statuses)
