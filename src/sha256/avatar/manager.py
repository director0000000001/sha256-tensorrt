"""Avatar lifecycle coordination.

:class:`AvatarManager` is the single place that knows how to get from "an
avatar id the user clicked" to "an avatar the engine can drive". It composes
three collaborators that each do one thing:

* :class:`~sha256.core.interfaces.AvatarRepository` — what avatars exist
* :mod:`sha256.avatar.validation` — are this avatar's files usable
* :class:`~sha256.avatar.cache.ResidencyCache` — what may stay in memory

The pipeline never touches any of them directly. It asks the manager for the
active avatar and gets an :class:`~sha256.core.avatar.AvatarDescriptor`, which
is what keeps the processing loop independent of how avatars are stored.

Threading
---------
Loading is slow (disk, and eventually GPU transfers) and must not run on the Qt
UI thread. This class does not create threads — it is synchronous and
thread-agnostic on purpose, so the caller decides. State changes are announced
through :meth:`add_observer` so a UI can show a loading indicator regardless of
which thread the load happened on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sha256.avatar.cache import ResidencyCache
from sha256.avatar.validation import AvatarValidationReport, validate_avatar
from sha256.core.avatar import AvatarDescriptor
from sha256.core.enums import AvatarState
from sha256.core.errors import (
    AvatarAssetError,
    AvatarLoadError,
    AvatarNotFoundError,
    InsufficientResourcesError,
)
from sha256.core.interfaces import AvatarEngine, AvatarRepository
from sha256.utils.logging import get_logger

__all__ = ["AvatarManager", "AvatarStatus"]

_LOG = get_logger(__name__)

StateObserver = Callable[["AvatarStatus"], None]


@dataclass(frozen=True, slots=True)
class AvatarStatus:
    """Everything the UI needs to render one avatar tile."""

    avatar_id: str
    name: str
    state: AvatarState
    error: str | None = None
    vram_mb: int | None = None

    @property
    def is_usable(self) -> bool:
        return self.state is not AvatarState.ERROR


class AvatarManager:
    """Owns avatar discovery, selection, and residency."""

    def __init__(
        self,
        repository: AvatarRepository,
        cache: ResidencyCache,
        *,
        engine: AvatarEngine | None = None,
        verify_checksums: bool = False,
    ) -> None:
        self._repository = repository
        self._cache = cache
        self._engine = engine
        self._verify_checksums = verify_checksums

        self._states: dict[str, AvatarState] = {}
        self._errors: dict[str, str] = {}
        self._active_id: str | None = None
        self._observers: list[StateObserver] = []

    # --- wiring ------------------------------------------------------------

    def attach_engine(self, engine: AvatarEngine | None) -> None:
        """Set or replace the engine.

        Replacing one unloads everything: avatars resident in the old engine's
        memory are not resident in the new one, and leaving stale cache entries
        would make the residency accounting silently wrong.
        """
        if engine is self._engine:
            return
        if self._engine is not None:
            self.unload_all()
        self._engine = engine

    @property
    def engine(self) -> AvatarEngine | None:
        return self._engine

    def add_observer(self, observer: StateObserver) -> None:
        """Register a callback fired on every avatar state change."""
        self._observers.append(observer)

    def remove_observer(self, observer: StateObserver) -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    # --- discovery ---------------------------------------------------------

    def refresh(self) -> Sequence[AvatarDescriptor]:
        """Re-scan storage and reconcile state.

        ``force=True`` is essential here rather than incidental: the repository
        caches, so without it this would only ever return what was found at
        startup and an avatar added while running would stay invisible.

        Avatars that disappeared are dropped; unknown ones start as DISCOVERED.
        Already-loaded avatars keep their state, so a refresh does not disturb a
        running session.
        """
        descriptors = self._repository.discover(force=True)
        known = {d.id for d in descriptors}

        for avatar_id in list(self._states):
            if avatar_id not in known:
                self._states.pop(avatar_id, None)
                self._errors.pop(avatar_id, None)
                self._cache.evict(avatar_id)
                if self._active_id == avatar_id:
                    self._active_id = None

        for descriptor in descriptors:
            self._states.setdefault(descriptor.id, AvatarState.DISCOVERED)

        for location, reason in self._repository.errors():
            _LOG.warning("Invalid avatar entry %s: %s", location, reason)

        return descriptors

    def list_avatars(self) -> Sequence[AvatarDescriptor]:
        """Every valid avatar. Triggers discovery on first call."""
        if not self._states:
            return self.refresh()
        return self._repository.discover()

    def statuses(self) -> Sequence[AvatarStatus]:
        """UI-facing status for every known avatar."""
        return tuple(self._status_for(d) for d in self.list_avatars())

    def _status_for(self, descriptor: AvatarDescriptor) -> AvatarStatus:
        entry = self._cache.entry(descriptor.id)
        return AvatarStatus(
            avatar_id=descriptor.id,
            name=descriptor.name,
            state=self._states.get(descriptor.id, AvatarState.DISCOVERED),
            error=self._errors.get(descriptor.id),
            vram_mb=entry.vram_mb if entry else None,
        )

    def validate(self, avatar_id: str) -> AvatarValidationReport:
        """Check an avatar's assets without loading it."""
        descriptor = self._repository.get(avatar_id)
        return validate_avatar(descriptor, verify_checksums=self._verify_checksums)

    # --- state -------------------------------------------------------------

    @property
    def active_id(self) -> str | None:
        return self._active_id

    @property
    def active(self) -> AvatarDescriptor | None:
        """The descriptor the pipeline should be driving, if any."""
        if self._active_id is None:
            return None
        try:
            return self._repository.get(self._active_id)
        except AvatarNotFoundError:
            self._active_id = None
            return None

    def state_of(self, avatar_id: str) -> AvatarState:
        return self._states.get(avatar_id, AvatarState.DISCOVERED)

    def _set_state(self, avatar_id: str, state: AvatarState, error: str | None = None) -> None:
        self._states[avatar_id] = state
        if error is None:
            self._errors.pop(avatar_id, None)
        else:
            self._errors[avatar_id] = error
        self._notify(avatar_id)

    def _notify(self, avatar_id: str) -> None:
        try:
            descriptor = self._repository.get(avatar_id)
        except AvatarNotFoundError:
            return
        status = self._status_for(descriptor)
        for observer in list(self._observers):
            # An observer is usually UI code. A failure there must not abort a
            # load that has already succeeded.
            try:
                observer(status)
            except Exception:
                _LOG.exception("Avatar state observer failed for %s", avatar_id)

    # --- loading -----------------------------------------------------------

    def load(self, avatar_id: str) -> AvatarDescriptor:
        """Make an avatar resident. Slow — never call from the UI thread.

        Raises:
            AvatarNotFoundError: Unknown id.
            AvatarAssetError: Validation found blocking problems.
            InsufficientResourcesError: Residency policy refused admission.
            AvatarLoadError: The engine failed to load it.
        """
        descriptor = self._repository.get(avatar_id)

        if self._cache.is_resident(avatar_id):
            self._cache.touch(avatar_id)
            return descriptor

        if self._engine is None:
            msg = "no avatar engine is attached"
            raise AvatarLoadError(msg)

        report = validate_avatar(descriptor, verify_checksums=self._verify_checksums)
        if not report.is_valid:
            reason = "; ".join(report.errors)
            self._set_state(avatar_id, AvatarState.ERROR, reason)
            raise AvatarAssetError(f"avatar {avatar_id!r} failed validation: {reason}")
        for warning in report.warnings:
            _LOG.warning("Avatar %s: %s", avatar_id, warning)

        if not self._engine.supports(descriptor):
            reason = (
                f"engine {self._engine.engine_id!r} cannot drive avatar {avatar_id!r} "
                f"(requires {descriptor.manifest.engine.engine_id!r})"
            )
            self._set_state(avatar_id, AvatarState.ERROR, reason)
            raise AvatarLoadError(reason)

        estimate = descriptor.manifest.engine.estimated_vram_mb or 0
        decision = self._cache.plan_admission(avatar_id, estimate)
        if not decision.admitted:
            self._set_state(avatar_id, AvatarState.ERROR, decision.reason)
            raise InsufficientResourcesError(decision.reason)

        for victim in decision.evict:
            _LOG.info("Evicting avatar %s to make room for %s", victim, avatar_id)
            self.unload(victim)

        self._set_state(avatar_id, AvatarState.LOADING)
        try:
            self._engine.load_avatar(descriptor)
        except Exception as exc:
            self._set_state(avatar_id, AvatarState.ERROR, str(exc))
            if isinstance(exc, AvatarLoadError):
                raise
            raise AvatarLoadError(f"engine failed to load {avatar_id!r}: {exc}") from exc

        self._cache.admit(avatar_id, estimate)
        measured = self._engine.estimated_vram_mb(avatar_id)
        if measured is not None:
            self._cache.update_measurement(avatar_id, measured)

        self._set_state(avatar_id, AvatarState.LOADED)
        _LOG.info("Avatar %s loaded", avatar_id)
        return descriptor

    def unload(self, avatar_id: str) -> None:
        """Release an avatar. Safe to call when it is not loaded."""
        if self._engine is not None:
            self._engine.unload_avatar(avatar_id)
        self._cache.evict(avatar_id)

        if self._active_id == avatar_id:
            self._active_id = None
            self._cache.set_active(None)

        if self._states.get(avatar_id) in (AvatarState.LOADED, AvatarState.ACTIVE):
            self._set_state(avatar_id, AvatarState.DISCOVERED)

    def unload_all(self) -> None:
        for avatar_id in self._cache.resident_ids():
            self.unload(avatar_id)

    # --- selection ---------------------------------------------------------

    def select(self, avatar_id: str) -> AvatarDescriptor:
        """Load if needed, then make this the avatar the pipeline drives."""
        descriptor = self.load(avatar_id)

        previous = self._active_id
        self._active_id = avatar_id
        self._cache.set_active(avatar_id)

        if previous is not None and previous != avatar_id and self._cache.is_resident(previous):
            # Kept warm rather than unloaded — switching back should be instant,
            # and the residency cache will evict it if something needs the room.
            self._set_state(previous, AvatarState.LOADED)

        self._set_state(avatar_id, AvatarState.ACTIVE)
        _LOG.info("Avatar %s is now active", avatar_id)
        return descriptor

    def deselect(self) -> None:
        """Clear the active avatar, leaving it resident."""
        if self._active_id is None:
            return
        previous = self._active_id
        self._active_id = None
        self._cache.set_active(None)
        if self._cache.is_resident(previous):
            self._set_state(previous, AvatarState.LOADED)
