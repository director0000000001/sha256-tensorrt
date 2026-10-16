"""Day 23 (2026-10-16): the per-side turn gain (``turn_gain_left`` /
``turn_gain_right``, 1.0 = off, bit-identical). The extractor's yaw runs
the other way from the tracker's: a delta under 0 is a turn to his left."""

from __future__ import annotations

import numpy as np
import pytest

from sha256.avatar.engine_liveportrait import (
    _drawn_near_edge_np,
    _profile_fill,
    _profile_fill_np,
    _turn_gains,
)


class TestTurnGains:
    def test_defaults_and_bounds(self):
        assert _turn_gains({}) == (1.0, 1.0)
        assert _turn_gains({"turn_gain_right": 1.3}) == (1.0, 1.3)
        assert _turn_gains({"turn_gain_left": 0.8, "turn_gain_right": 1.5}) == (0.8, 1.5)
        with pytest.raises(ValueError, match="turn_gain_right"):
            _turn_gains({"turn_gain_right": 2.5})
        with pytest.raises(ValueError, match="turn_gain_left"):
            _turn_gains({"turn_gain_left": 0.2})

    def test_the_drive_applies_the_gain_to_the_side_it_turns_to(self):
        """The tensor arithmetic of ``_drive``'s gain block, as written there:
        an extractor yaw delta under 0 (his left) takes the left gain, one
        over 0 (his right) the right one; pitch and roll untouched."""
        torch = pytest.importorskip("torch")
        pose_d0 = torch.tensor([[2.0], [-1.0], [0.5]])
        gains = torch.tensor([1.0, 1.3])  # left, right

        def drive(pose_now):
            yaw_delta = pose_now[1:2] - pose_d0[1:2]
            gain = torch.where(yaw_delta < 0.0, gains[0], gains[1])
            return torch.cat([pose_now[0:1], pose_d0[1:2] + yaw_delta * gain, pose_now[2:3]], dim=0)

        left = drive(torch.tensor([[5.0], [-21.0], [1.0]]))
        assert left.tolist() == [[5.0], [-21.0], [1.0]]  # -20 x 1.0
        right = drive(torch.tensor([[5.0], [19.0], [1.0]]))
        assert right[1, 0].item() == pytest.approx(-1.0 + 20.0 * 1.3)
        assert right[0, 0].item() == 5.0 and right[2, 0].item() == 1.0


def _render(background=(200, 200, 200), block=(20, 40)) -> np.ndarray:
    """A 64x64 render: the background, and a drawn block on columns
    ``block`` whose colour changes column by column (so the edge column's
    own colour is distinct)."""
    face = np.empty((64, 64, 3), dtype=np.uint8)
    face[:] = background
    lo, hi = block
    for c in range(lo, hi + 1):
        face[:, c] = (60 + c, 40, 30)
    return face


class TestProfileFill:
    def test_option_default_and_bounds(self):
        assert _profile_fill({}) == 12
        assert _profile_fill({"profile_fill_px": 0}) == 0
        with pytest.raises(ValueError, match="profile_fill_px"):
            _profile_fill({"profile_fill_px": 100})

    @pytest.mark.parametrize("near_left", [False, True])
    def test_the_background_beyond_the_edge_takes_the_edge_colour(self, near_left):
        face = _render()
        background = np.array([200.0, 200.0, 200.0], dtype=np.float32)
        edge = _drawn_near_edge_np(face, background, near_left)
        expect = 20 if near_left else 40
        assert int(edge[10]) == expect
        out = _profile_fill_np(face, background, edge, near_left, 6)
        assert out.shape == face.shape
        # the six background columns beyond the edge AND the edge's own
        # three columns take the colour three columns inside the edge
        filled = range(14, 23) if near_left else range(38, 47)
        inside = expect + 3 if near_left else expect - 3  # _PROFILE_FILL_INSET
        for c in filled:
            np.testing.assert_array_equal(out[:, c], face[:, inside])
        untouched = 13 if near_left else 47
        np.testing.assert_array_equal(out[:, untouched], face[:, untouched])
        # the drawn face inside the rim is never written
        np.testing.assert_array_equal(out[:, 23:38], face[:, 23:38])

    def test_a_row_without_a_drawn_edge_is_left_alone(self):
        face = _render()
        face[5] = (200, 200, 200)  # row 5: background only
        background = np.array([200.0, 200.0, 200.0], dtype=np.float32)
        edge = _drawn_near_edge_np(face, background, False)
        assert int(edge[5]) == -1
        out = _profile_fill_np(face, background, edge, False, 6)
        np.testing.assert_array_equal(out[5], face[5])

    def test_the_gpu_fill_is_the_cpu_one(self, tmp_path):
        torch = pytest.importorskip("torch")
        from tests.unit.test_engine_hairline import _RenderFixture

        engine, _ = _RenderFixture()._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        face = _render()
        background = np.array([200.0, 200.0, 200.0], dtype=np.float32)
        edge = _drawn_near_edge_np(face, background, False)
        cpu = _profile_fill_np(face, background, edge, False, 6)
        dev = torch.from_numpy(face).permute(2, 0, 1).float()
        engine._frame_bgr = True
        gpu = engine._profile_fill_gpu(dev, background, edge, False, 6)
        np.testing.assert_array_equal(gpu.permute(1, 2, 0).numpy().astype(np.uint8), cpu)
