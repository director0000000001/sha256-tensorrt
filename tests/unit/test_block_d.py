"""Block D (day 16): the switch's settings, the portable layout of a frozen
build, the GPU check's fallback to the photograph engine, and the window
stopping on the owner's word."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sha256.app.bootstrap import _start_engine_with_fallback, build_application
from sha256.app.kill_switch import KillSwitchState
from sha256.config.loader import load_settings
from sha256.config.paths import AppPaths
from sha256.config.schema import Settings
from sha256.core.errors import EngineError

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestSettings:
    def test_the_switch_is_off_by_default_and_the_committed_file_says_so(self):
        settings = Settings(_env_file=None)
        assert settings.kill_switch.url == ""
        assert settings.kill_switch.check_hours == 1.0
        paths = AppPaths.resolve()
        loaded = load_settings(paths)
        assert loaded.kill_switch.url == ""

    def test_the_section_is_validated(self):
        with pytest.raises(ValueError):
            Settings(_env_file=None, kill_switch={"check_hours": 0.0})


class TestPortableLayout:
    def test_a_frozen_build_with_models_beside_it_is_its_own_root(self, tmp_path, monkeypatch):
        exe = tmp_path / "bundle" / "sha256.exe"
        exe.parent.mkdir()
        exe.write_bytes(b"")
        (exe.parent / "models").mkdir()
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "executable", str(exe))

        paths = AppPaths.resolve()

        assert paths.root == exe.parent.resolve()
        assert paths.models == exe.parent.resolve() / "models"
        assert paths.avatars == exe.parent.resolve() / "avatars"
        assert paths.configs == exe.parent.resolve() / "configs"
        assert paths.captures == exe.parent.resolve() / "captures"
        assert not paths.is_development

    def test_a_frozen_build_without_models_beside_it_is_installed(self, tmp_path, monkeypatch):
        exe = tmp_path / "sha256.exe"
        exe.write_bytes(b"")
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "executable", str(exe))
        assert AppPaths.portable_root() is None

    def test_unfrozen_is_never_portable(self, monkeypatch):
        monkeypatch.delattr(sys, "frozen", raising=False)
        assert AppPaths.portable_root() is None


class _Engine:
    def __init__(self, name: str, *, fails: bool = False) -> None:
        self.name = name
        self.fails = fails
        self.started = False

    def start(self) -> None:
        if self.fails:
            raise EngineError("the LivePortrait engine needs an NVIDIA GPU with CUDA")
        self.started = True


class _Registry:
    def __init__(self, engines: dict[str, _Engine]) -> None:
        self.created: list[str] = []
        self._engines = engines

    def create(self, name: str, **kwargs):
        self.created.append(name)
        return self._engines[name]


class TestGpuFallback:
    def _settings(self, backend: str) -> Settings:
        return Settings(_env_file=None, engine={"backend": backend})

    def test_a_neural_engine_that_cannot_start_gives_way_to_the_photograph_engine(
        self, app_paths, monkeypatch
    ):
        monkeypatch.setattr("sha256.app.bootstrap._gpu_memory_mb", lambda: None)
        neural, warp = _Engine("liveportrait", fails=True), _Engine("warp")
        registries = SimpleNamespace(engine=_Registry({"liveportrait": neural, "warp": warp}))

        engine, notes = _start_engine_with_fallback(
            registries, self._settings("liveportrait"), app_paths
        )

        assert engine is warp and warp.started and not neural.started
        assert registries.engine.created == ["liveportrait", "warp"]
        assert notes and "no usable NVIDIA GPU" in notes[0]

    def test_a_small_card_is_noted_and_the_neural_engine_kept(self, app_paths, monkeypatch):
        monkeypatch.setattr("sha256.app.bootstrap._gpu_memory_mb", lambda: 4096)
        neural = _Engine("liveportrait")
        registries = SimpleNamespace(engine=_Registry({"liveportrait": neural}))
        engine, notes = _start_engine_with_fallback(
            registries, self._settings("liveportrait"), app_paths
        )
        assert engine is neural and neural.started
        assert notes and "4096 MB" in notes[0]

    def test_another_engine_that_fails_still_raises(self, app_paths, monkeypatch):
        monkeypatch.setattr("sha256.app.bootstrap._gpu_memory_mb", lambda: None)
        registries = SimpleNamespace(engine=_Registry({"warp": _Engine("warp", fails=True)}))
        with pytest.raises(EngineError):
            _start_engine_with_fallback(registries, self._settings("warp"), app_paths)

    def test_the_context_carries_the_switch_and_no_notes_on_the_null_engine(
        self, app_paths, settings
    ):
        context = build_application(settings=settings, paths=app_paths, setup_logging=False)
        try:
            assert context.kill_switch is not None and not context.kill_switch.configured
            assert context.startup_notes == ()
        finally:
            context.shutdown()


@pytest.mark.gui
class TestWindowSwitch:
    def test_the_owners_word_stops_the_pipeline_and_disables_start(self, qtbot, app_context):
        pytest.importorskip("PySide6.QtWidgets")
        from sha256.ui import main_window as module

        window = module.MainWindow(app_context)
        qtbot.addWidget(window)
        told: list[str] = []
        module.QMessageBox.information = staticmethod(lambda *a, **k: told.append(a[2]))
        window._on_start()
        assert window._runner.is_running

        window.switched_off.emit("Back on Monday.")
        qtbot.waitUntil(lambda: window._switched_off, timeout=2000)

        assert not window._runner.is_running
        assert not window._primary_button.isEnabled()
        assert told == ["Back on Monday."]
        assert window._status_line.text() == "Back on Monday."
        window._on_switched_off("again")  # once only
        assert told == ["Back on Monday."]

    def test_startup_notes_are_shown_once(self, qtbot, app_context):
        pytest.importorskip("PySide6.QtWidgets")
        from dataclasses import replace

        from sha256.ui import main_window as module

        context = replace(app_context, startup_notes=("The photograph engine is running.",))
        window = module.MainWindow(context)
        qtbot.addWidget(window)
        told: list[str] = []
        module.QMessageBox.information = staticmethod(lambda *a, **k: told.append(a[2]))
        window._show_startup_notes()
        assert told == ["The photograph engine is running."]
        _ = KillSwitchState  # the switch's own type is what the signal carries the word from
