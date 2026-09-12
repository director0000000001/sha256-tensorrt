"""The roster (day 16 evening): named profiles under ``configs/drivers/``,
the one in force at ``local.driver.json``, and a file from before names
existed adopted rather than lost."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sha256.calibration import roster
from sha256.calibration.profile import PROFILE_FILENAME, DriverProfile


def named(name: str, **options: object) -> DriverProfile:
    return DriverProfile(measured_at="2026-09-12T18:58:11+04:00", name=name, engine_options=options)


class TestSlug:
    @pytest.mark.parametrize(
        ("name", "slug"),
        [
            ("Giorgi Beridze", "giorgi-beridze"),
            ("  Ana   K.  ", "ana-k"),
            ("გიორგი ბერიძე", "გიორგი-ბერიძე"),
            ("---", "person"),
            ("", "person"),
            ("Unnamed (2026-09-12 00:42)", "unnamed-2026-09-12-00-42"),
        ],
    )
    def test_a_file_name_from_a_persons_name(self, name: str, slug: str):
        assert roster.profile_slug(name) == slug

    def test_bounded(self):
        assert len(roster.profile_slug("x" * 500)) == 80


class TestRoster:
    def test_saved_listed_by_name_and_found(self, tmp_path: Path):
        configs = tmp_path / "configs"
        roster.save_named(named("Zura", gaze_gain=0.7), configs)
        roster.save_named(named("ana"), configs)
        roster.save_named(named("Giorgi"), configs)

        assert [p.name for p in roster.list_profiles(configs)] == ["ana", "Giorgi", "Zura"]
        found = roster.find_profile(configs, "Zura")
        assert found is not None and found.engine_options == {"gaze_gain": 0.7}
        assert roster.find_profile(configs, "nobody") is None
        assert roster.named_path(configs, "Zura") == configs / "drivers" / "zura.json"

    def test_a_nameless_profile_cannot_join(self, tmp_path: Path):
        with pytest.raises(ValueError, match="name"):
            roster.save_named(named("  "), tmp_path)

    def test_files_without_a_name_or_unreadable_are_not_listed(self, tmp_path: Path):
        folder = roster.drivers_dir(tmp_path)
        folder.mkdir(parents=True)
        (folder / "old.json").write_text(json.dumps({"engine_options": {}}), encoding="utf-8")
        (folder / "bad.json").write_text("{not json", encoding="utf-8")
        assert roster.list_profiles(tmp_path) == []
        assert roster.list_profiles(tmp_path / "missing") == []

    def test_activate_writes_the_file_in_force_and_none_removes_it(self, tmp_path: Path):
        path = tmp_path / PROFILE_FILENAME

        assert roster.activate(named("Zura", gaze_gain=0.7), tmp_path) == path
        active = roster.active_profile(tmp_path)
        assert active is not None and active.name == "Zura"
        assert json.loads(path.read_text(encoding="utf-8"))["engine_options"] == {"gaze_gain": 0.7}

        roster.activate(None, tmp_path)
        assert not path.exists() and roster.active_profile(tmp_path) is None
        roster.activate(None, tmp_path)  # twice is fine


class TestAdoptUnnamed:
    def test_a_file_from_before_names_is_named_and_kept(self, tmp_path: Path):
        path = tmp_path / PROFILE_FILENAME
        path.write_text(
            json.dumps(
                {"measured_at": "2026-09-12T00:42:46+04:00", "engine_options": {"eye_gain": 1.4}}
            ),
            encoding="utf-8",
        )

        adopted = roster.adopt_unnamed(tmp_path)

        assert adopted is not None and adopted.name == "Unnamed (2026-09-12 00:42)"
        assert [p.name for p in roster.list_profiles(tmp_path)] == ["Unnamed (2026-09-12 00:42)"]
        again = roster.active_profile(tmp_path)
        assert again is not None and again.name == adopted.name
        assert again.engine_options == {"eye_gain": 1.4}

    def test_nothing_to_adopt(self, tmp_path: Path):
        assert roster.adopt_unnamed(tmp_path) is None
        roster.activate(named("Zura"), tmp_path)
        assert roster.adopt_unnamed(tmp_path) is None
        assert roster.list_profiles(tmp_path) == []  # a named file in force is not copied

    def test_without_a_measured_at(self, tmp_path: Path):
        (tmp_path / PROFILE_FILENAME).write_text(
            json.dumps({"engine_options": {}}), encoding="utf-8"
        )
        adopted = roster.adopt_unnamed(tmp_path)
        assert adopted is not None and adopted.name == roster.UNNAMED_PREFIX
