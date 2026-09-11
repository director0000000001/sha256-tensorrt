"""The driver profile file: written from a result, read back, and only the
driver's options handed to the engine."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from sha256.calibration.measure import CalibrationResult
from sha256.calibration.profile import PROFILE_VERSION, DriverProfile, driver_options_from


def a_result() -> CalibrationResult:
    result = CalibrationResult()
    result.rest = {"ear": 0.31, "opening": 0.004, "gaze_y": math.nan}
    result.jitter = {"opening_change_p99": 0.0041}
    result.response = {"talk_opening": {"slope": 1.1, "r": 0.9, "n": 200.0, "range": 0.05}}
    result.options = {
        "expression_gain": 1.64,
        "mouth_gate_mar": [0.0045, 0.0135],
        "lip_rest_rows": [-0.001, 0.002, 0.008],
        "mouth_rest": "driver",
        "mask_forehead_reach": 0.5,  # not the driver's: must not reach the file's overlay
    }
    result.notes = ["blink: slope 0.98, within 5% -- eye_gain kept"]
    return result


class TestRoundTrip:
    def test_written_and_read_back(self, tmp_path: Path):
        when = datetime(2026, 9, 12, 10, 30, 0).astimezone()
        profile = DriverProfile.from_result(
            a_result(),
            avatar="my_avatar_neural",
            session="recordings/calibration-x",
            measured_at=when,
        )
        path = profile.save(tmp_path / "configs" / "local.driver.json")

        again = DriverProfile.load(path)

        assert again is not None
        assert again.profile_version == PROFILE_VERSION
        assert again.avatar == "my_avatar_neural" and again.session == "recordings/calibration-x"
        assert again.measured_at.startswith("2026-09-12T10:30:00")
        assert again.engine_options["expression_gain"] == 1.64
        assert again.engine_options["lip_rest_rows"] == [-0.001, 0.002, 0.008]
        assert "mask_forehead_reach" not in again.engine_options
        assert "gaze_y" not in again.rest  # NaN dropped
        assert again.response["talk_opening"]["slope"] == 1.1
        assert again.notes == a_result().notes

    def test_the_overlay_is_the_drivers_keys_only(self, tmp_path: Path):
        path = tmp_path / "local.driver.json"
        path.write_text(
            json.dumps(
                {
                    "engine_options": {
                        "expression_gain": 2.0,
                        "crop_scale": 2.5,
                        "mask_jaw_shrink": 0.1,
                    }
                }
            ),
            encoding="utf-8",
        )
        assert driver_options_from(path) == {"expression_gain": 2.0}

    def test_no_file_and_a_damaged_file_give_nothing(self, tmp_path: Path):
        assert DriverProfile.load(tmp_path / "missing.json") is None
        assert driver_options_from(tmp_path / "missing.json") == {}
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert DriverProfile.load(bad) is None
        wrong = tmp_path / "wrong.json"
        wrong.write_text(json.dumps({"engine_options": "not a dict"}), encoding="utf-8")
        assert DriverProfile.load(wrong) is None

    def test_unknown_fields_in_the_file_are_ignored(self, tmp_path: Path):
        path = tmp_path / "p.json"
        path.write_text(json.dumps({"engine_options": {}, "someday": 1}), encoding="utf-8")
        assert DriverProfile.load(path) is not None
