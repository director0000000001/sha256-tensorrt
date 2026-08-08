"""Component registry.

Backends register themselves under a short id; configuration names that id; the
composition root looks it up. Nothing in the application imports a backend
module directly, which is the mechanism behind the "not tightly coupled to one
specific AI model" requirement.

Why a registry rather than importing by dotted path from config:

*   An import path in a config file is executable input — a supply-chain hazard
    once avatar packs or plugins become downloadable.
*   Registration is lazy by *factory*, so declaring a backend costs nothing
    until it is actually selected. A registered MediaPipe backend does not
    import MediaPipe unless the user picks it.
*   Available ids can be listed, which makes "unknown backend" a helpful error
    and lets the UI populate a dropdown honestly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from sha256.core.errors import ComponentNotRegisteredError

__all__ = ["ComponentRegistry", "Registration", "Registries"]


@dataclass(frozen=True, slots=True)
class Registration[T]:
    """One registered backend."""

    backend_id: str
    factory: Callable[..., T]
    description: str = ""
    requires: tuple[str, ...] = ()
    """Import names this backend needs, e.g. ``("mediapipe",)``.

    Declared rather than discovered so availability can be reported without
    importing — the point of the lazy design.
    """

    experimental: bool = False


class ComponentRegistry[T]:
    """Named factories for one component category."""

    def __init__(self, category: str) -> None:
        self._category = category
        self._items: dict[str, Registration[T]] = {}

    @property
    def category(self) -> str:
        return self._category

    def register(
        self,
        backend_id: str,
        factory: Callable[..., T],
        *,
        description: str = "",
        requires: Iterable[str] = (),
        experimental: bool = False,
        replace: bool = False,
    ) -> None:
        """Register ``factory`` under ``backend_id``.

        Args:
            replace: Permit overwriting an existing id. Off by default so a
                duplicate registration is a loud error rather than a silent
                shadowing that is painful to debug.
        """
        if backend_id in self._items and not replace:
            msg = f"{self._category} backend {backend_id!r} is already registered"
            raise ValueError(msg)
        self._items[backend_id] = Registration(
            backend_id=backend_id,
            factory=factory,
            description=description,
            requires=tuple(requires),
            experimental=experimental,
        )

    def create(self, backend_id: str, **kwargs: Any) -> T:
        """Instantiate the backend registered as ``backend_id``."""
        registration = self.get(backend_id)
        return registration.factory(**kwargs)

    def get(self, backend_id: str) -> Registration[T]:
        try:
            return self._items[backend_id]
        except KeyError:
            raise ComponentNotRegisteredError(
                self._category, backend_id, list(self._items)
            ) from None

    def available(self) -> list[str]:
        """Registered ids, sorted. Says nothing about whether deps are installed."""
        return sorted(self._items)

    def describe(self) -> list[Registration[T]]:
        """All registrations, for UI listings and diagnostics."""
        return [self._items[key] for key in self.available()]

    def __contains__(self, backend_id: object) -> bool:
        return backend_id in self._items

    def __len__(self) -> int:
        return len(self._items)


@dataclass(slots=True)
class Registries:
    """The complete set of registries for one application instance.

    Held on the application context rather than in module-level globals so tests
    can build an isolated set, and so two instances in one process (a benchmark
    harness, for example) cannot interfere with each other.
    """

    camera: ComponentRegistry[Any] = field(default_factory=lambda: ComponentRegistry("camera"))
    tracker: ComponentRegistry[Any] = field(default_factory=lambda: ComponentRegistry("tracker"))
    engine: ComponentRegistry[Any] = field(default_factory=lambda: ComponentRegistry("engine"))
    renderer: ComponentRegistry[Any] = field(default_factory=lambda: ComponentRegistry("renderer"))
    virtual_camera: ComponentRegistry[Any] = field(
        default_factory=lambda: ComponentRegistry("virtual_camera")
    )

    def summary(self) -> dict[str, list[str]]:
        """Category -> available ids. Used by the debug panel and diagnostics."""
        return {
            registry.category: registry.available()
            for registry in (
                self.camera,
                self.tracker,
                self.engine,
                self.renderer,
                self.virtual_camera,
            )
        }
