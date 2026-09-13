"""The steep poses (day 17, his "looking sideways, head down"): with
``steep_pose = "clamp"`` the face is never held, and the turn applied to
the portrait is capped at the hold thresholds while the reading itself is
kept -- on the CPU fixture, the networks faked."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sha256.avatar.engine_liveportrait import (
    NUM_KP,
    _gaze_yaw_fade,
    _Portrait,
    _should_hold,
    _steep_fade,
    _steep_pose,
)
from sha256.core.types import HeadPose
from tests.unit.test_engine_liveportrait import TestDriveNetworkOptions as _DriveFixture
from tests.unit.test_engine_liveportrait import descriptor


class TestOption:
    def test_clamp_by_default_hold_by_choice(self):
        assert _steep_pose({}) == "clamp"
        assert _steep_pose({"steep_pose": "hold"}) == "hold"
        with pytest.raises(ValueError, match="steep_pose"):
            _steep_pose({"steep_pose": "freeze"})

    def test_the_portrait_reads_it_once(self, tmp_path):
        assert _Portrait(descriptor(tmp_path)).steep_clamp
        with_hold = descriptor(tmp_path)
        with_hold.manifest.engine_options["steep_pose"] = "hold"
        assert not _Portrait(with_hold).steep_clamp

    def test_the_fade_past_the_yaw_cap(self, tmp_path):
        loaded = _Portrait(descriptor(tmp_path))
        loaded.rest_yaw = 0.0
        assert _steep_fade(loaded, None) == 1.0
        assert _steep_fade(loaded, HeadPose(yaw=30.0)) == 1.0
        assert _steep_fade(loaded, HeadPose(yaw=45.0)) == 1.0
        assert _steep_fade(loaded, HeadPose(yaw=-52.5)) == pytest.approx(0.5)
        assert _steep_fade(loaded, HeadPose(yaw=60.0)) == 0.0
        loaded.rest_yaw = 20.0  # relative to the rest, like the hold
        assert _steep_fade(loaded, HeadPose(yaw=65.0)) == 1.0
        loaded.steep_clamp = False
        assert _steep_fade(loaded, HeadPose(yaw=80.0)) == 1.0

    def test_the_gaze_fades_earlier_than_the_rest(self, tmp_path):
        loaded = _Portrait(descriptor(tmp_path))
        loaded.rest_yaw = 0.0
        assert _gaze_yaw_fade(loaded, None) == 1.0
        assert _gaze_yaw_fade(loaded, HeadPose(yaw=25.0)) == 1.0
        assert _gaze_yaw_fade(loaded, HeadPose(yaw=30.0)) == 1.0
        assert _gaze_yaw_fade(loaded, HeadPose(yaw=-37.5)) == pytest.approx(0.5)
        assert _gaze_yaw_fade(loaded, HeadPose(yaw=45.0)) == 0.0
        assert _steep_fade(loaded, HeadPose(yaw=45.0)) == 1.0  # the mouth still drives
        loaded.steep_clamp = False
        assert _gaze_yaw_fade(loaded, HeadPose(yaw=80.0)) == 1.0


class TestNoHold:
    def test_the_face_is_never_held_under_the_clamp(self, tmp_path):
        loaded = _Portrait(descriptor(tmp_path))
        loaded.rest_pitch = 0.0
        loaded.rest_yaw = 0.0
        loaded.steep_clamp = False
        assert _should_hold(loaded, HeadPose(pitch=0.0, yaw=80.0))
        loaded.holding = False
        loaded.steep_clamp = True
        assert not _should_hold(loaded, HeadPose(pitch=0.0, yaw=80.0))
        assert not _should_hold(loaded, HeadPose(pitch=-40.0, yaw=0.0))
        assert not loaded.holding


def _yaw_logits(torch, degrees: float):
    """One-hot pose logits that decode to ``degrees`` (bin * 3 - 97.5)."""
    logits = torch.zeros(1, 66)
    logits[0, round((degrees + 97.5) / 3.0)] = 1000.0
    return logits


class TestClampInDrive:
    def _fixture(self, tmp_path, torch, calls: dict, yaw_degrees: float):
        engine, loaded = _DriveFixture()._engine_and_loaded(tmp_path, torch=torch, calls=calls)
        loaded.options = {"rotation_pivot": "none"}

        def motion(*args, **kwargs):
            calls.setdefault("motion", []).append((args, kwargs))
            return {
                "exp": torch.zeros(1, NUM_KP * 3),
                "scale": torch.ones(1, 1),
                "t": torch.zeros(1, 3),
                "pitch": torch.zeros(1, 66),
                "yaw": _yaw_logits(torch, yaw_degrees),
                "roll": torch.zeros(1, 66),
            }

        engine._nets["motion"] = motion
        # One keypoint on the x axis: its x after a yaw of theta is cos(theta).
        loaded.x_c_s = torch.zeros(1, NUM_KP, 3)
        loaded.x_c_s[0, 0, 0] = 1.0
        loaded.pose_limit = torch.tensor([[20.0], [35.0], [1.0e6]])
        return engine, loaded

    def test_the_turn_is_capped_and_the_reading_is_kept(self, tmp_path):
        torch = pytest.importorskip("torch")
        calls: dict[str, list] = {}
        engine, loaded = self._fixture(tmp_path, torch, calls, 58.5)

        loaded.steep_clamp = False
        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        free = calls["warping"][-1][1]["kp_driving"].float()[0, 0, 0].item()
        assert abs(free) == pytest.approx(math.cos(math.radians(58.5)), abs=0.02)
        assert loaded.pose_prev[1].item() == pytest.approx(58.5, abs=0.1)

        loaded.steep_clamp = True
        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        capped = calls["warping"][-1][1]["kp_driving"].float()[0, 0, 0].item()
        assert abs(capped) == pytest.approx(math.cos(math.radians(35.0)), abs=0.02)
        # The EMA state is the real reading, not the capped one.
        assert loaded.pose_prev[1].item() == pytest.approx(58.5, abs=0.1)

    def test_inside_the_threshold_nothing_changes(self, tmp_path):
        torch = pytest.importorskip("torch")
        calls: dict[str, list] = {}
        engine, loaded = self._fixture(tmp_path, torch, calls, 19.5)
        loaded.steep_clamp = False
        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        free = calls["warping"][-1][1]["kp_driving"].float().clone()
        loaded.steep_clamp = True
        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        capped = calls["warping"][-1][1]["kp_driving"].float()
        assert np.allclose(free.numpy(), capped.numpy(), atol=1e-6)

    def test_the_cap_is_relative_to_the_anchors_pose(self, tmp_path):
        torch = pytest.importorskip("torch")
        calls: dict[str, list] = {}
        engine, loaded = self._fixture(tmp_path, torch, calls, 58.5)
        loaded.steep_clamp = True
        loaded.pose_d0 = torch.tensor([[0.0], [30.0], [0.0]])  # the anchor sat at yaw 30
        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        capped = calls["warping"][-1][1]["kp_driving"].float()[0, 0, 0].item()
        # 58.5 - 30 = 28.5 within the 35 cap: the whole turn is drawn.
        assert abs(capped) == pytest.approx(math.cos(math.radians(58.5)), abs=0.02)
