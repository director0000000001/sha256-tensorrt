"""The blink's counter (day 17, his "when I shut my eyes the brows come down
a little and the mouth moves as if smiling"): the eye network's pull on the
brows and the mouth corners countered in the brow drive's own rows, by the
blink's depth -- on the CPU fixture."""

from __future__ import annotations

import numpy as np
import pytest

from sha256.avatar.engine_liveportrait import (
    _BROW_LIFT_Y,
    _MOUTH_SMILE,
    MODEL_SIZE,
    _blink_brow_curve,
    _blink_brow_hold,
    _blink_depth,
    _blink_lift_curve,
    _blink_mouth_curve,
    _blink_mouth_hold,
    _blink_mouth_lift,
    _Portrait,
)
from tests.unit.test_engine_liveportrait import TestAnchorGatingInRender as _RenderFixture
from tests.unit.test_engine_liveportrait import descriptor


class TestOptions:
    def test_defaults_and_bounds(self):
        assert _blink_brow_hold({}) >= 0.0
        assert _blink_brow_hold({"blink_brow_hold": 0.02}) == 0.02
        assert _blink_mouth_hold({"blink_mouth_hold": -0.01}) == -0.01
        with pytest.raises(ValueError, match="blink_brow_hold"):
            _blink_brow_hold({"blink_brow_hold": -0.01})
        with pytest.raises(ValueError, match="blink_brow_hold"):
            _blink_brow_hold({"blink_brow_hold": 0.5})
        with pytest.raises(ValueError, match="blink_mouth_hold"):
            _blink_mouth_hold({"blink_mouth_hold": 0.5})

    def test_the_portrait_reads_them_once(self, tmp_path):
        d = descriptor(tmp_path)
        d.manifest.engine_options["blink_brow_hold"] = 0.02
        d.manifest.engine_options["blink_mouth_hold"] = -0.01
        loaded = _Portrait(d)
        assert loaded.blink_brow_hold == 0.02 and loaded.blink_mouth_hold == -0.01


class TestDepth:
    def test_the_curves(self):
        assert _blink_brow_curve(0.0) == 0.0 and _blink_brow_curve(1.0) == 1.0
        assert _blink_brow_curve(0.2) == pytest.approx(0.36)
        assert _blink_brow_curve(0.6) == pytest.approx(0.84)
        assert _blink_mouth_curve(0.5) == 0.0 and _blink_mouth_curve(0.85) == 1.0
        assert _blink_mouth_curve(0.675) == pytest.approx(0.5)
        assert _blink_mouth_curve(0.2) == 0.0

    def test_the_lift_bump(self):
        assert _blink_lift_curve(0.0) == 0.0
        assert _blink_lift_curve(0.85) == 0.0 and _blink_lift_curve(1.0) == 0.0
        assert _blink_lift_curve(0.2) == pytest.approx(0.2 * (1 - 0.2 / 0.85))
        assert _blink_lift_curve(0.425) > _blink_lift_curve(0.2) > _blink_lift_curve(0.7)
        assert _blink_mouth_lift({}) >= 0.0
        assert _blink_mouth_lift({"blink_mouth_lift": 0.02}) == 0.02
        with pytest.raises(ValueError, match="blink_mouth_lift"):
            _blink_mouth_lift({"blink_mouth_lift": -0.01})

    def test_shortfall_from_rest(self):
        assert _blink_depth(0.3, None) == 0.0
        assert _blink_depth(0.3, 0.3) == 0.0
        assert _blink_depth(0.35, 0.3) == 0.0  # wider than rest is not a blink
        assert _blink_depth(0.12, 0.3) == pytest.approx(0.6)
        assert _blink_depth(0.0, 0.3) == 1.0


class TestCounterInRender:
    def _fixture(self, tmp_path, torch):
        engine, loaded = _RenderFixture()._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        loaded.blink_brow_hold = 0.02
        loaded.blink_mouth_hold = -0.01
        loaded.blink_mouth_lift = 0.0
        loaded.rest_ear = 0.3
        loaded.anchor_seen = loaded.anchor_wanted  # frozen: no learning in the way
        return engine, loaded

    def test_shut_eyes_lift_the_brow_row_and_move_the_smile_axis(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._fixture(tmp_path, torch)
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)

        engine._render(loaded, driver_rgb, 0.12, 0.0, 0.3, 0.0, anchor_ok=False)  # depth 0.6
        for k, _ in _BROW_LIFT_Y.items():
            expected = -0.02 * _blink_brow_curve(0.6)
            assert loaded.brow_delta[0, k, 1].item() == pytest.approx(expected)
        for k, axis in _MOUTH_SMILE:
            assert loaded.brow_delta[0, k, axis].item() == pytest.approx(
                0.01 * _blink_mouth_curve(0.6)
            )

        engine._render(loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=False)  # open
        for k, _ in _BROW_LIFT_Y.items():
            assert loaded.brow_delta[0, k, 1].item() == 0.0
        for k, axis in _MOUTH_SMILE:
            assert loaded.brow_delta[0, k, axis].item() == 0.0

    def test_a_raise_and_a_blink_add_up(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._fixture(tmp_path, torch)
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
        engine._render(loaded, driver_rgb, 0.12, 0.0, 0.3, 0.0, anchor_ok=False, brow_lift=0.5)
        (k, y_per_unit), *_ = _BROW_LIFT_Y.items()
        assert loaded.brow_delta[0, k, 1].item() == pytest.approx(
            0.5 * y_per_unit - 0.02 * _blink_brow_curve(0.6)
        )

    def test_the_lift_adds_to_the_smile_axis(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._fixture(tmp_path, torch)
        loaded.blink_mouth_lift = 0.015
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
        engine._render(loaded, driver_rgb, 0.12, 0.0, 0.3, 0.0, anchor_ok=False)  # depth 0.6
        expected = -(-0.01 * _blink_mouth_curve(0.6) + 0.015 * _blink_lift_curve(0.6))
        for k, axis in _MOUTH_SMILE:
            assert loaded.brow_delta[0, k, axis].item() == pytest.approx(expected)

    def test_nothing_before_the_anchor_knows_the_rest(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._fixture(tmp_path, torch)
        loaded.rest_ear = None
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
        engine._render(loaded, driver_rgb, 0.05, 0.0, 0.3, 0.0, anchor_ok=False)
        for k, _ in _BROW_LIFT_Y.items():
            assert loaded.brow_delta[0, k, 1].item() == 0.0
        for k, axis in _MOUTH_SMILE:
            assert loaded.brow_delta[0, k, axis].item() == 0.0
