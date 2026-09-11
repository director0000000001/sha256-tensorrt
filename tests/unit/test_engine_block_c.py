"""Block C (day 16): the yaw pivot behind the face (C2) and the lip rest
tracking's faster first approach (C3), on the CPU fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from sha256.avatar.engine_liveportrait import (
    _LIP_REST_FAST_FACTOR,
    _LIP_REST_FAST_FRAMES,
    _MOUTH_OPENING,
    DRIVER_OPTION_KEYS,
    MODEL_SIZE,
    NUM_KP,
    _crop_shift_x,
    _yaw_pivot_z,
)
from tests.unit.test_engine_liveportrait import TestAnchorGatingInRender as _RenderFixture
from tests.unit.test_engine_liveportrait import TestDriveNetworkOptions as _DriveFixture
from tests.unit.test_engine_liveportrait import TestLipRestTracking as _LipRestFixture
from tests.unit.test_engine_liveportrait import TestRotationPivot as _PivotFixture


class TestYawPivotOption:
    def test_reads_validates_and_is_the_drivers(self):
        assert _yaw_pivot_z({}) == 0.0
        assert _yaw_pivot_z({"yaw_pivot_z": 0.2}) == 0.2
        with pytest.raises(ValueError, match="yaw_pivot_z"):
            _yaw_pivot_z({"yaw_pivot_z": -0.1})
        with pytest.raises(ValueError, match="yaw_pivot_z"):
            _yaw_pivot_z({"yaw_pivot_z": 1.5})
        assert "yaw_pivot_z" in DRIVER_OPTION_KEYS


class TestCropShift:
    def test_reads_validates_and_is_the_drivers(self, tmp_path):
        from tests.unit.test_engine_liveportrait import descriptor

        assert _crop_shift_x({}) == 0.0
        assert _crop_shift_x({"crop_shift_x": -5.8}) == -5.8
        with pytest.raises(ValueError, match="crop_shift_x"):
            _crop_shift_x({"crop_shift_x": 41.0})
        assert "crop_shift_x" in DRIVER_OPTION_KEYS
        from sha256.avatar.engine_liveportrait import _Portrait

        d = descriptor(tmp_path)
        assert _Portrait(d, {"crop_shift_x": -5.8}).crop_shift_x == -5.8


class TestYawPivotInDrive:
    """The same fixture the rotation pivot's own tests use: a motion net
    that reports a chosen yaw, the warping call recorded."""

    def _keypoints(self, tmp_path, torch, *, yaw_bin: int | None, yaw_pivot_z: float, pivot: str):
        calls: dict[str, list] = {}
        engine, loaded = _DriveFixture()._engine_and_loaded(tmp_path, torch=torch, calls=calls)
        engine._nets["motion"] = _PivotFixture()._motion_with_pose(torch, yaw_bin=yaw_bin)
        loaded.options = {
            "size_smoothing": 0.0,
            "eye_retargeting": False,
            "rotation_pivot": pivot,
            "yaw_pivot_z": yaw_pivot_z,
        }
        loaded.x_c_s = torch.rand(1, NUM_KP, 3)
        loaded.rel_mask = torch.ones(1, NUM_KP, 3)
        loaded.gate = torch.ones(1, NUM_KP, 3)
        loaded.pivot = loaded.x_c_s[:, :4, :].mean(dim=1, keepdim=True).clone()
        loaded.yaw_pivot = (
            None
            if yaw_pivot_z == 0.0
            else torch.tensor([[[0.0, 0.0, yaw_pivot_z]]], dtype=torch.float32)
        )
        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        return calls["warping"][-1][1]["kp_driving"].float().clone()

    def test_zero_is_the_old_line_and_a_turn_moves_the_face_sideways(self, tmp_path):
        torch = pytest.importorskip("torch")
        torch.manual_seed(0)
        for pivot in ("none", "eyes"):
            torch.manual_seed(1)
            plain = self._keypoints(tmp_path, torch, yaw_bin=50, yaw_pivot_z=0.0, pivot=pivot)
            torch.manual_seed(1)
            same = self._keypoints(tmp_path, torch, yaw_bin=50, yaw_pivot_z=0.0, pivot=pivot)
            assert torch.equal(plain, same)
            torch.manual_seed(1)
            turned = self._keypoints(tmp_path, torch, yaw_bin=50, yaw_pivot_z=0.3, pivot=pivot)
            diff = turned - plain
            # One rigid translation of every keypoint, along x, no z.
            # (fp16 keypoints: a few thousandths of rounding)
            assert float(diff[..., 0].std()) < 3e-3 and float(diff[..., 0].abs().mean()) > 0.05
            assert float(diff[..., 1].abs().max()) < 3e-3
            assert float(diff[..., 2].abs().max()) < 1e-5

    def test_a_pure_pitch_or_no_turn_adds_nothing(self, tmp_path):
        torch = pytest.importorskip("torch")
        torch.manual_seed(2)
        plain = self._keypoints(tmp_path, torch, yaw_bin=None, yaw_pivot_z=0.0, pivot="eyes")
        torch.manual_seed(2)
        with_pivot = self._keypoints(tmp_path, torch, yaw_bin=None, yaw_pivot_z=0.3, pivot="eyes")
        assert float((with_pivot - plain).abs().max()) < 1e-5


class TestFastLipRestStart:
    def test_the_first_resting_frames_move_faster_then_settle_to_the_rate(self, tmp_path):
        torch = pytest.importorskip("torch")
        helper = _LipRestFixture()
        engine, loaded, driver_rgb = helper._fixture(tmp_path, torch, frames=100.0)
        loaded.lip_rest_steps = 0  # the fixture skips the fast start; this test wants it
        helper._rest(engine, loaded, driver_rgb, 16)  # the settle, then one step
        first = float(loaded.exp_d0[0, 20, 1])
        assert first == pytest.approx(0.1 * _LIP_REST_FAST_FACTOR / 100.0)
        assert loaded.lip_rest_steps == 1

        helper._rest(engine, loaded, driver_rgb, _LIP_REST_FAST_FRAMES - 1)
        assert loaded.lip_rest_steps == _LIP_REST_FAST_FRAMES
        before = float(loaded.exp_d0[0, 20, 1])
        helper._rest(engine, loaded, driver_rgb, 1)
        after = float(loaded.exp_d0[0, 20, 1])
        # Past the fast frames: the option's own rate, 1/100 of the gap.
        assert after - before == pytest.approx((0.1 - before) / 100.0, rel=1e-3)
        for k, axis in _MOUTH_OPENING:
            assert 0.0 < float(loaded.exp_d0[0, k, axis]) < 0.1

    def test_a_reset_starts_the_fast_frames_again(self, tmp_path):
        torch = pytest.importorskip("torch")
        helper = _LipRestFixture()
        engine, loaded, driver_rgb = helper._fixture(tmp_path, torch, frames=100.0)
        loaded.lip_rest_steps = 0
        helper._rest(engine, loaded, driver_rgb, 20)
        assert loaded.lip_rest_steps == 5
        engine.reset_anchor(loaded.descriptor.id)
        assert loaded.lip_rest_steps == 0 and loaded.lip_rest_run == 0

    def test_the_fixture_is_the_cpu_render(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = _RenderFixture()._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
        engine._render(loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=True)
        assert loaded.lip_rest_steps == 0  # alpha 0 on this fixture: nothing counts
