"""Tests for the component registry.

The registry is what makes backends swappable, so its failure modes — duplicate
ids, unknown ids — are tested as behaviour rather than left implicit.
"""

from __future__ import annotations

import pytest

from sha256.app.registry import ComponentRegistry, Registries
from sha256.core.errors import ComponentNotRegisteredError


class Dummy:
    def __init__(self, value: int = 0) -> None:
        self.value = value


@pytest.fixture
def registry() -> ComponentRegistry[Dummy]:
    return ComponentRegistry("dummy")


class TestRegistration:
    def test_registered_backend_can_be_created(self, registry: ComponentRegistry[Dummy]):
        registry.register("test", Dummy)
        instance = registry.create("test", value=7)

        assert isinstance(instance, Dummy)
        assert instance.value == 7

    def test_duplicate_id_is_rejected(self, registry: ComponentRegistry[Dummy]):
        """Silent shadowing would be painful to debug; fail loudly instead."""
        registry.register("test", Dummy)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("test", Dummy)

    def test_replacement_is_allowed_when_explicit(self, registry: ComponentRegistry[Dummy]):
        registry.register("test", Dummy)
        registry.register("test", Dummy, replace=True, description="second")

        assert registry.get("test").description == "second"

    def test_unknown_id_lists_the_alternatives(self, registry: ComponentRegistry[Dummy]):
        registry.register("alpha", Dummy)
        registry.register("beta", Dummy)

        with pytest.raises(ComponentNotRegisteredError) as exc_info:
            registry.create("gamma")

        message = str(exc_info.value)
        assert "gamma" in message
        assert "alpha" in message and "beta" in message

    def test_empty_registry_reports_none_available(self, registry: ComponentRegistry[Dummy]):
        with pytest.raises(ComponentNotRegisteredError, match="<none>"):
            registry.create("anything")


class TestIntrospection:
    def test_available_ids_are_sorted(self, registry: ComponentRegistry[Dummy]):
        registry.register("zebra", Dummy)
        registry.register("alpha", Dummy)

        assert registry.available() == ["alpha", "zebra"]

    def test_membership_and_length(self, registry: ComponentRegistry[Dummy]):
        registry.register("test", Dummy)

        assert "test" in registry
        assert "absent" not in registry
        assert len(registry) == 1

    def test_requirements_are_declared_not_imported(self, registry: ComponentRegistry[Dummy]):
        """Declaring deps lets availability be reported without importing them."""
        registry.register("heavy", Dummy, requires=["mediapipe", "numpy"])
        assert registry.get("heavy").requires == ("mediapipe", "numpy")


class TestRegistries:
    def test_builtin_registries_are_independent(self):
        registries = Registries()
        registries.camera.register("test", Dummy)

        assert "test" in registries.camera
        assert "test" not in registries.tracker

    def test_summary_covers_every_category(self):
        registries = Registries()
        summary = registries.summary()

        assert set(summary) == {"camera", "tracker", "engine", "renderer", "virtual_camera"}

    def test_two_instances_do_not_share_state(self):
        """Module-level globals would let a test leak into the next one."""
        first, second = Registries(), Registries()
        first.camera.register("test", Dummy)

        assert "test" not in second.camera
