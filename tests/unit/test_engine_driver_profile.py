"""The engine's side of "Calibrate me" (day 16): the gate thresholds, the
leak table, the driver's lip rest and the brow dead band as options; the
driver profile laid over an avatar's options at load; the per-frame
readings the calibration measures from. On the CPU fixtures of
``test_engine_liveportrait``."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sha256.avatar.engine_liveportrait import (
    _BROW_DEADBAND,
    _BROW_MOUTH_LEAK,
    _MOUTH_ACTIVITY_MAR,
    _MOUTH_ACTIVITY_WIDTH,
    _MOUTH_GATE_MAR,
    _MOUTH_GATE_WIDTH,
    _MOUTH_OPENING,
    DRIVER_OPTION_KEYS,
    MODEL_SIZE,
    MOUTH_KEYPOINTS,
    NUM_KP,
    PORTRAIT_OPTION_KEYS,
    LivePortraitEngine,
    _brow_lift,
    _brow_mouth_leak_slopes,
    _gate_thresholds,
    _GateThresholds,
    _lip_rest_rows,
    _mouth_activity,
    _mouth_gate,
    _mouth_rest,
    _Portrait,
)
from sha256.core.types import HeadPose
from tests.unit.test_engine_liveportrait import TestAnchorGatingInRender, descriptor

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pose(**kwargs) -> HeadPose:
    return HeadPose(**kwargs)


class TestTheSplit:
    def test_driver_and_portrait_keys_do_not_overlap(self):
        assert not (DRIVER_OPTION_KEYS & PORTRAIT_OPTION_KEYS)
        for key in ("expression_gain", "mouth_gate_mar", "lip_rest_rows", "brow_mouth_leak_slopes"):
            assert key in DRIVER_OPTION_KEYS
        for key in ("crop_scale", "mask_forehead_reach", "rotation_pivot"):
            assert key in PORTRAIT_OPTION_KEYS

    def test_his_avatars_every_option_belongs_to_one_side(self):
        manifest = REPO_ROOT / "avatars" / "my_avatar_neural" / "metadata.json"
        if not manifest.is_file():
            pytest.skip("the owner's avatar is not here")
        options = json.loads(manifest.read_text(encoding="utf-8"))["engine_options"]
        unknown = set(options) - DRIVER_OPTION_KEYS - PORTRAIT_OPTION_KEYS
        assert not unknown


class TestThresholdOptions:
    def test_defaults_are_the_constants(self):
        t = _gate_thresholds({})
        assert t == _GateThresholds()
        assert t.mouth_gate_mar == _MOUTH_GATE_MAR and t.mouth_gate_width == _MOUTH_GATE_WIDTH
        assert t.activity_mar == _MOUTH_ACTIVITY_MAR and t.activity_width == _MOUTH_ACTIVITY_WIDTH
        assert t.brow_deadband == _BROW_DEADBAND

    def test_the_profile_values_are_read_and_validated(self):
        t = _gate_thresholds(
            {
                "mouth_gate_mar": [0.005, 0.015],
                "mouth_gate_width": (0.02, 0.07),
                "mouth_activity_mar": 0.006,
                "mouth_activity_width": 0.03,
                "brow_deadband": 0.12,
            }
        )
        assert t.mouth_gate_mar == (0.005, 0.015) and t.mouth_gate_width == (0.02, 0.07)
        assert t.activity_mar == 0.006 and t.activity_width == 0.03 and t.brow_deadband == 0.12
        for bad in (
            {"mouth_gate_mar": [0.015, 0.005]},
            {"mouth_gate_mar": [0.0, 0.01]},
            {"mouth_gate_mar": 0.01},
            {"mouth_gate_width": [0.1, 2.0]},
            {"mouth_activity_mar": 0.0},
            {"mouth_activity_width": -1},
            {"brow_deadband": 1.5},
        ):
            with pytest.raises(ValueError):
                _gate_thresholds(bad)

    def test_the_gate_functions_take_the_drivers_thresholds(self):
        wide = _GateThresholds(mouth_gate_mar=(0.010, 0.030), activity_mar=0.008)
        # The same opening: past "full" on the defaults, only started on the wider ones.
        assert _mouth_gate(0.004 + 0.012, 0.40, 0.004, 0.40) == 1.0
        assert _mouth_gate(0.004 + 0.012, 0.40, 0.004, 0.40, None, wide) == pytest.approx(0.1)
        # The same motion: two units on the default, one on the wider.
        assert _mouth_activity(0.012, 0.40, 0.004, 0.40)[0] == pytest.approx(2.0)
        assert _mouth_activity(0.012, 0.40, 0.004, 0.40, 0.0, wide)[0] == pytest.approx(1.0)

    def test_brow_lift_takes_its_dead_band(self):
        assert _brow_lift(0.20 * 1.30, 0.20, _pose()) == pytest.approx(0.30 - _BROW_DEADBAND)
        assert _brow_lift(0.20 * 1.30, 0.20, _pose(), 0.20) == pytest.approx(0.10)
        assert _brow_lift(0.20 * 1.15, 0.20, _pose(), 0.20) == 0.0


class TestLeakAndLipRestOptions:
    def test_the_leak_table_defaults_and_reads_the_profiles(self):
        assert _brow_mouth_leak_slopes({}) == _BROW_MOUTH_LEAK
        got = _brow_mouth_leak_slopes({"brow_mouth_leak_slopes": [[20, 1, 0.2], [17, 2, -0.01]]})
        assert got == {(20, 1): 0.2, (17, 2): -0.01}
        for bad in ([[21, 0, 0.1]], [[1, 3, 0.1]], [[1, 1, 2.0]], [[1, 1]], "x", [[1, 1, "n"]]):
            with pytest.raises(ValueError, match="brow_mouth_leak_slopes"):
                _brow_mouth_leak_slopes({"brow_mouth_leak_slopes": bad})

    def test_lip_rest_rows_and_the_driver_mode(self):
        assert _lip_rest_rows({}) is None
        assert _lip_rest_rows({"lip_rest_rows": [-0.001, 0.002, 0.008]}) == (-0.001, 0.002, 0.008)
        for bad in ([0.1, 0.2], [0.1, 0.2, 0.9], "abc", [0.1, None, 0.2]):
            with pytest.raises(ValueError, match="lip_rest_rows"):
                _lip_rest_rows({"lip_rest_rows": bad})
        assert _mouth_rest({"mouth_rest": "driver"}) == "driver"

    def test_render_writes_the_drivers_own_leak_table(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = TestAnchorGatingInRender()._engine_and_loaded(
            tmp_path, torch=torch, anchor_wanted=1
        )
        loaded.leak_gain = 1.0
        loaded.leak_slopes = {(20, 1): 0.30, (14, 1): -0.05}
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)

        engine._render(loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=False, brow_deviation=0.1)

        assert loaded.mouth_leak[0, 20, 1] == pytest.approx(0.03)
        assert loaded.mouth_leak[0, 14, 1] == pytest.approx(-0.005)
        assert loaded.mouth_leak[0, 17, 1] == 0.0  # not in this driver's table

    def test_the_driver_rest_goes_into_the_anchor_at_the_freeze_and_on_a_seed(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = TestAnchorGatingInRender()._engine_and_loaded(
            tmp_path, torch=torch, anchor_wanted=2
        )
        loaded.options = {"mouth_rest": "driver", "lip_rest_rows": [-0.001, 0.002, 0.008]}
        loaded.lip_rest_rows = _lip_rest_rows(loaded.options)
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)

        for _ in range(2):
            engine._render(loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=True)

        for (k, axis), value in zip(_MOUTH_OPENING, (-0.001, 0.002, 0.008), strict=True):
            assert loaded.exp_d0[0, k, axis] == pytest.approx(value), (k, axis)
        assert loaded.exp_d0[0, 14, 1] == 0.0
        assert torch.equal(loaded.exp_prev, loaded.exp_d0)

        state = {
            "exp_d0": np.full((1, NUM_KP, 3), 0.01, dtype=np.float32),
            "t_d0": np.zeros((1, 3), dtype=np.float32),
            "scale_d0": np.ones((1, 1), dtype=np.float32),
            "R_d0": np.eye(3, dtype=np.float32)[None],
            "pose_d0": np.zeros((3, 1), dtype=np.float32),
        }
        engine.seed_anchor(loaded.descriptor.id, state)
        assert loaded.exp_d0[0, 20, 1] == pytest.approx(0.008)
        assert loaded.exp_d0[0, 3, 0] == pytest.approx(0.01)

    def test_driver_without_rows_keeps_the_anchors_own(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = TestAnchorGatingInRender()._engine_and_loaded(
            tmp_path, torch=torch, anchor_wanted=1
        )
        loaded.options = {"mouth_rest": "driver"}
        loaded.lip_rest_rows = None
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
        engine._render(loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=True)
        assert float(loaded.exp_d0.abs().max()) == 0.0


class TestDriverOptionsOnTheEngine:
    def test_the_portrait_takes_the_profile_over_its_own_options(self, tmp_path):
        d = descriptor(tmp_path)
        d.manifest.engine_options.update({"expression_gain": 1.8, "crop_scale": 2.2})
        loaded = _Portrait(d, {"expression_gain": 1.6, "mouth_activity_mar": 0.006})
        assert loaded.options["expression_gain"] == 1.6
        assert loaded.options["crop_scale"] == 2.2
        assert loaded.thresholds.activity_mar == 0.006
        assert loaded.leak_slopes == _BROW_MOUTH_LEAK and loaded.lip_rest_rows is None
        plain = _Portrait(d)
        assert plain.options["expression_gain"] == 1.8 and plain.thresholds == _GateThresholds()

    def test_set_driver_options_accepts_the_drivers_keys_only(self):
        engine = LivePortraitEngine(driver_options={"expression_gain": 2.0})
        assert engine.driver_options == {"expression_gain": 2.0}
        engine.driver_options["expression_gain"] = 9.0  # a copy
        assert engine.driver_options == {"expression_gain": 2.0}
        with pytest.raises(ValueError, match="crop_scale"):
            engine.set_driver_options({"expression_gain": 1.0, "crop_scale": 2.5})
        assert engine.driver_options == {"expression_gain": 2.0}  # untouched by the refusal
        engine.set_driver_options(None)
        assert engine.driver_options == {}

    def test_frame_readings_carry_the_gates_the_rest_and_the_mouth_rows(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = TestAnchorGatingInRender()._engine_and_loaded(
            tmp_path, torch=torch, anchor_wanted=1
        )
        engine._loaded[loaded.descriptor.id] = loaded
        assert engine.frame_readings("nobody") is None

        before = engine.frame_readings(loaded.descriptor.id)
        assert before["anchor_frozen"] == 0.0 and math.isnan(before["rest_ear"])

        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
        engine._render(
            loaded, driver_rgb, 0.31, 0.004, 0.3, 0.0, anchor_ok=True, rest_sample=(0.4, 0.2)
        )
        loaded.exp_prev[0, 20, 1] = 0.017
        loaded.gate_open_prev = 0.25

        after = engine.frame_readings(loaded.descriptor.id)
        assert after["anchor_frozen"] == 1.0 and after["holding"] == 0.0
        assert after["rest_ear"] == pytest.approx(0.31) and after["rest_mar"] == pytest.approx(
            0.004
        )
        assert after["rest_mouth_width"] == 0.4 and after["rest_brow"] == 0.2
        assert after["gate_open"] == 0.25
        assert after["exp_20_1"] == pytest.approx(0.017)
        assert after["anchor_20_1"] == 0.0
        for k in MOUTH_KEYPOINTS:
            for axis in range(3):
                assert f"exp_{k}_{axis}" in after
