"""Tests for configuration loading, layering, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from sha256.config.loader import load_settings, merge_deep
from sha256.config.paths import AppPaths, find_repo_root
from sha256.config.schema import Settings
from sha256.core.errors import ConfigurationError


class TestMergeDeep:
    def test_nested_tables_merge_key_by_key(self):
        """A local override of one key must not reset its siblings."""
        base = {"camera": {"width": 1280, "height": 720}}
        overlay = {"camera": {"width": 1920}}

        assert merge_deep(base, overlay) == {"camera": {"width": 1920, "height": 720}}

    def test_scalars_are_replaced(self):
        assert merge_deep({"a": 1}, {"a": 2}) == {"a": 2}

    def test_inputs_are_not_mutated(self):
        base = {"a": {"b": 1}}
        merge_deep(base, {"a": {"b": 2}})
        assert base == {"a": {"b": 1}}


class TestLoading:
    def test_defaults_load_without_any_files(self, app_paths: AppPaths):
        settings = load_settings(app_paths, use_files=False)

        assert settings.camera.backend == "null"
        assert settings.camera.target_fps == 30.0

    def test_toml_layers_apply_in_order(self, app_paths: AppPaths):
        app_paths.configs.mkdir(parents=True, exist_ok=True)
        (app_paths.configs / "default.toml").write_text(
            "[camera]\nwidth = 1920\nheight = 1080\n", encoding="utf-8"
        )
        (app_paths.configs / "local.toml").write_text("[camera]\nwidth = 800\n", encoding="utf-8")

        settings = load_settings(app_paths)

        assert settings.camera.width == 800, "local overrides default"
        assert settings.camera.height == 1080, "unrelated default survives"

    def test_explicit_overrides_win(self, app_paths: AppPaths):
        app_paths.configs.mkdir(parents=True, exist_ok=True)
        (app_paths.configs / "default.toml").write_text(
            "[camera]\nwidth = 1920\n", encoding="utf-8"
        )

        settings = load_settings(app_paths, overrides={"camera": {"width": 640}})
        assert settings.camera.width == 640

    def test_malformed_toml_is_reported_clearly(self, app_paths: AppPaths):
        app_paths.configs.mkdir(parents=True, exist_ok=True)
        (app_paths.configs / "default.toml").write_text("[camera\nwidth =", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="not valid TOML"):
            load_settings(app_paths)

    def test_invalid_value_names_the_field(self, app_paths: AppPaths):
        with pytest.raises(ConfigurationError, match=r"camera\.target_fps"):
            load_settings(app_paths, use_files=False, overrides={"camera": {"target_fps": -5}})


class TestValidation:
    def test_unknown_top_level_key_is_rejected(self):
        with pytest.raises(Exception, match="extra"):
            Settings(_env_file=None, nonsense=True)  # type: ignore[call-arg]

    def test_fps_mismatch_produces_a_warning_not_an_error(self):
        settings = Settings(
            _env_file=None,
            camera={"target_fps": 30.0},
            rendering={"target_fps": 60.0},
        )
        assert any("exceeds" in warning for warning in settings.warnings)

    def test_consistent_configuration_has_no_warnings(self):
        assert Settings(_env_file=None).warnings == ()

    def test_settings_are_immutable(self, settings: Settings):
        with pytest.raises(Exception, match=r"frozen|immutable"):
            settings.log_level = "DEBUG"  # type: ignore[misc]


class TestEnvironmentVariables:
    def test_nested_values_use_the_double_underscore_delimiter(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("SHA256_CAMERA__TARGET_FPS", "60")
        monkeypatch.setenv("SHA256_LOG_LEVEL", "DEBUG")

        settings = Settings(_env_file=None)

        assert settings.camera.target_fps == 60.0
        assert settings.log_level == "DEBUG"


class TestPaths:
    def test_repo_root_is_found_from_inside_the_package(self):
        root = find_repo_root()
        assert root is not None
        assert (root / "pyproject.toml").is_file()

    def test_development_layout_lives_in_the_checkout(self):
        paths = AppPaths.resolve()
        assert paths.is_development
        assert paths.avatars == paths.root / "avatars"

    def test_relative_paths_resolve_against_the_root(self, app_paths: AppPaths):
        assert app_paths.resolve_relative("avatars") == app_paths.root / "avatars"

    def test_absolute_paths_are_left_alone(self, app_paths: AppPaths, tmp_path: Path):
        target = tmp_path / "elsewhere"
        assert app_paths.resolve_relative(target) == target

    def test_ensure_creates_writable_directories(self, tmp_path: Path):
        paths = AppPaths(
            root=tmp_path,
            avatars=tmp_path / "a",
            models=tmp_path / "m",
            configs=tmp_path / "c",
            logs=tmp_path / "l",
            cache=tmp_path / "k",
            is_development=True,
        )
        paths.ensure()

        assert paths.avatars.is_dir()
        assert paths.logs.is_dir()
        assert not paths.configs.exists(), "a missing config dir is a valid state"


class TestOutputSection:
    """Day 15: what leaves the pipeline for the virtual camera."""

    def test_defaults(self):
        output = Settings(_env_file=None).output
        assert output.quality == "native"
        assert output.failure_text == "Camera battery died"
        assert output.auto_blackout is True
        assert (output.lost_face_seconds, output.min_blackout_seconds, output.recovery_seconds) == (
            1.5,
            3.0,
            1.0,
        )
        assert output.blackout_hotkey == "ctrl+alt+b"

    def test_the_quality_is_one_of_the_conditioners_names(self):
        from sha256.pipeline.conditioning import QUALITY_NAMES

        for name in QUALITY_NAMES:
            assert Settings(_env_file=None, output={"quality": name}).output.quality == name
        with pytest.raises(Exception, match="quality"):
            Settings(_env_file=None, output={"quality": "4k"})

    def test_the_committed_default_config_carries_the_section(self, app_paths: AppPaths):
        settings = load_settings(app_paths)
        assert settings.output.failure_text == "Camera battery died"
