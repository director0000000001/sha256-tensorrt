"""The driver profile at start (day 16): laid over an engine that takes
one, before the first avatar loads; ignored by an engine that does not;
never fatal."""

from __future__ import annotations

import json
from pathlib import Path

from sha256.app.bootstrap import apply_driver_profile, build_application


class TakesProfile:
    def __init__(self, *, refuse: bool = False) -> None:
        self.given: list[dict] = []
        self.refuse = refuse

    def set_driver_options(self, options: dict) -> None:
        if self.refuse:
            raise ValueError("not driver options")
        self.given.append(dict(options))


def write_profile(path: Path, options: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"measured_at": "2026-09-12T10:00:00", "engine_options": options}),
        encoding="utf-8",
    )
    return path


class TestApply:
    def test_the_profile_reaches_an_engine_that_takes_one(self, tmp_path: Path):
        path = write_profile(
            tmp_path / "local.driver.json", {"expression_gain": 1.6, "crop_scale": 2.5}
        )
        engine = TakesProfile()

        profile = apply_driver_profile(engine, path)

        assert profile is not None and profile.measured_at == "2026-09-12T10:00:00"
        assert engine.given == [{"expression_gain": 1.6}]  # the portrait's key dropped

    def test_no_file_no_setter_or_a_refusal_give_none(self, tmp_path: Path):
        assert apply_driver_profile(TakesProfile(), tmp_path / "none.json") is None
        path = write_profile(tmp_path / "p.json", {"expression_gain": 1.6})
        assert apply_driver_profile(object(), path) is None
        assert apply_driver_profile(TakesProfile(refuse=True), path) is None

    def test_build_application_records_no_profile_with_the_null_engine(self, app_paths, settings):
        write_profile(app_paths.configs / "local.driver.json", {"expression_gain": 1.6})
        context = build_application(settings=settings, paths=app_paths, setup_logging=False)
        try:
            assert context.driver_profile is None  # the null engine takes none
        finally:
            context.shutdown()
