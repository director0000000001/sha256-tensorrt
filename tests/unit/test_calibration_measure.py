"""The calibration's measurement (day 16): synthetic sessions with known
gains, jitter and leak slopes go in, and the driver options that come out
must reproduce them by the rules the hand values were set with."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sha256.avatar.engine_liveportrait import (
    _BROW_MOUTH_LEAK,
    _MOUTH_ACTIVITY_MAR,
    _MOUTH_OPENING,
)
from sha256.calibration.measure import (
    DEAD_ZONE,
    GAIN_LIMITS,
    CalibrationResult,
    calibrate,
    describe,
    fit_line,
)
from sha256.calibration.samples import Sample
from sha256.calibration.script import STEPS

REST = {"ear": 0.32, "opening": 0.005, "width": 0.40, "brow": 0.20, "gaze_x": 0.0}
HIS_CURRENT = {
    "expression_gain": 1.8,
    "eye_gain": 1.0,
    "brow_follow": 1.0,
    "gaze_gain": 1.0,
    "brow_mouth_leak": 1.0,
}


def synthetic_samples(
    *,
    talk_gain: float = 1.0,
    blink_gain: float = 1.0,
    brow_gain: float = 1.0,
    gaze_gain: float = 1.0,
    leak: dict[tuple[int, int], float] | None = None,
    jitter: float = 0.002,
    with_avatar: bool = True,
    with_engine: bool = True,
    anchor_after: int = 15,
    fps: float = 30.0,
    seed: int = 0,
    eye_offset_px: float = 0.0,
) -> list[Sample]:
    """A whole run of STEPS with his face moving as each step asks and the
    avatar answering by the given gains; the engine's rows leak by ``leak``
    (default: the engine's own table)."""
    rng = np.random.default_rng(seed)
    leak = dict(_BROW_MOUTH_LEAK) if leak is None else leak
    samples: list[Sample] = []
    frame = 0
    for step in STEPS:
        n = round(step.seconds * fps)
        for i in range(n):
            phase = i / max(n - 1, 1)
            his = dict(REST)
            his.update({"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "z": 50.0, "face_height": 300.0})
            his["opening"] += float(rng.uniform(-jitter, jitter))
            his["width"] *= 1.0 + float(rng.uniform(-0.004, 0.004))
            his["brow"] *= 1.0 + float(rng.uniform(-0.01, 0.01))
            his["gaze_x"] += float(rng.uniform(-0.005, 0.005))
            his["gaze_y"] = 0.0
            his["lip"] = 0.01
            his["eye_x"], his["eye_y"] = 640.0, 300.0
            if step.kind == "blink":
                his["ear"] = 0.32 - 0.27 * max(math.sin(phase * math.pi * 3), 0.0)
            elif step.kind == "eyes_shut":
                his["ear"] = 0.05 if 0.2 < phase < 0.8 else 0.32
            elif step.kind == "open":
                his["opening"] = 0.005 + 0.12 * abs(math.sin(phase * math.pi * 2))
            elif step.kind == "talk":
                his["opening"] = 0.005 + float(rng.uniform(0.0, 0.06))
                # Talking moves the brows a little (his: +6..13%): the dead
                # band's confound, about 0.10 at the 90th percentile.
                his["brow"] *= 1.0 + float(rng.uniform(-0.12, 0.12))
            elif step.kind == "smile_closed":
                his["width"] = 0.40 * (1.0 + 0.15 * math.sin(phase * math.pi))
            elif step.kind == "smile_teeth":
                his["width"] = 0.40 * (1.0 + 0.22 * math.sin(phase * math.pi))
                his["opening"] = 0.005 + 0.03 * math.sin(phase * math.pi)
            elif step.kind == "brows":
                his["brow"] = 0.20 * (1.0 + 0.40 * abs(math.sin(phase * math.pi * 2)))
            elif step.kind == "gaze":
                his["gaze_x"] = 0.15 * math.sin(phase * math.pi * 2)
            elif step.kind == "turn":
                his["yaw"] = 30.0 * math.sin(phase * math.pi * 2)
            avatar = None
            if with_avatar:
                rise = his["brow"] / 0.20 - 1.0
                gx = his["gaze_x"]
                avatar = {
                    "ear": 0.33 + blink_gain * (his["ear"] - 0.32) + float(rng.normal(0, 0.002)),
                    "opening": 0.010
                    + talk_gain * (his["opening"] - 0.005)
                    + float(rng.normal(0, 0.0005)),
                    "width": 0.41 * (1.0 + 0.5 * (his["width"] / 0.40 - 1.0)),
                    "brow": 0.21 * (1.0 + brow_gain * rise) + float(rng.normal(0, 0.0005)),
                    "gaze_x": gaze_gain * gx + float(rng.normal(0, 0.002)),
                    "gaze_y": 0.0,
                    "lip": 0.01,
                    "eye_x": 640.0 + eye_offset_px,
                    "eye_y": 300.0,
                    "face_height": 300.0,
                    "yaw": 0.0,
                    "pitch": 0.0,
                    "roll": 0.0,
                    "z": 50.0,
                }
            engine: dict[str, float] = {}
            if with_engine:
                frozen = frame >= anchor_after
                moving = his["opening"] > 0.011 or his["width"] > 0.42
                engine = {
                    "anchor_frozen": 1.0 if frozen else 0.0,
                    "holding": 0.0,
                    "gate_mouth": 1.0 if moving else 0.0,
                    "gate_open": 1.0 if his["opening"] > 0.011 else 0.0,
                    "gate_brow": 0.0,
                    "rest_ear": 0.32,
                    "rest_mar": 0.005,
                    "rest_mouth_width": 0.40,
                    "rest_brow": 0.20,
                    "face_height_rest": 300.0,
                    "z_rest": 50.0,
                }
                deviation = his["brow"] - 0.20
                base = {(17, 1): -0.001, (19, 1): 0.002, (20, 1): 0.008}
                for k in (7, 14, 17, 19, 20):
                    for axis in range(3):
                        value = base.get((k, axis), 0.0) + leak.get((k, axis), 0.0) * deviation
                        engine[f"exp_{k}_{axis}"] = value + float(rng.normal(0, 0.0003))
                for k, axis in _MOUTH_OPENING:
                    engine[f"anchor_{k}_{axis}"] = base[(k, axis)] + 0.009  # pressed lips
            samples.append(
                Sample(
                    frame_id=frame,
                    timestamp=frame / fps,
                    step=step.kind,
                    face=True,
                    his=his,
                    engine=engine,
                    avatar=avatar,
                )
            )
            frame += 1
    return samples


def change_of(result: CalibrationResult, key: str):
    return next((c for c in result.changes if c.key == key), None)


class TestFit:
    def test_a_straight_line_is_read_back(self):
        xs = np.linspace(0, 1, 50)
        fit = fit_line(xs, 0.8 * xs + 0.1)
        assert fit.slope == pytest.approx(0.8)
        assert fit.intercept == pytest.approx(0.1)
        assert fit.r == pytest.approx(1.0)
        assert fit.n == 50 and fit.trusted()

    def test_too_few_or_flat_points_are_not_trusted(self):
        assert not fit_line([0, 1], [0, 1]).trusted()
        flat = fit_line([0.0] * 40, np.linspace(0, 1, 40))
        assert math.isnan(flat.slope) and not flat.trusted()
        nans = fit_line([math.nan] * 40, [1.0] * 40)
        assert nans.n == 0


class TestRestAndJitter:
    def test_the_rest_is_his_median_over_the_still_steps(self):
        result = calibrate(synthetic_samples(), HIS_CURRENT)
        assert result.ok
        assert result.rest["ear"] == pytest.approx(0.32, abs=0.005)
        assert result.rest["opening"] == pytest.approx(0.005, abs=0.001)
        assert result.rest["width"] == pytest.approx(0.40, abs=0.005)
        assert result.rest["brow"] == pytest.approx(0.20, abs=0.005)
        assert result.rest["anchor_ear"] == 0.32  # the engine's own
        assert result.frames["rest"] == 150 and result.frames["talk"] == 210

    def test_the_activity_unit_follows_the_still_jitter_by_the_day_12_rule(self):
        low = calibrate(synthetic_samples(jitter=0.001), HIS_CURRENT)
        high = calibrate(synthetic_samples(jitter=0.004), HIS_CURRENT)
        # Uniform +-j noise: |change| p99 ~ 1.9 j; the unit is p99 / 1.1 --
        # but never under the day-12 live floor (a still step reads less
        # jitter than a call).
        assert low.options["mouth_activity_mar"] == _MOUTH_ACTIVITY_MAR
        assert high.options["mouth_activity_mar"] == pytest.approx(0.0069, abs=0.001)
        assert high.options["mouth_activity_mar"] > low.options["mouth_activity_mar"]
        start, full = high.options["mouth_gate_mar"]
        assert full == pytest.approx(3 * start, rel=1e-3)
        # Twice the still top (rest 0.005 + the noise's own 95th percentile):
        # the low case reads the rule, the high one meets the bound.
        assert low.options["mouth_gate_mar"][0] == pytest.approx(
            2 * low.jitter["opening_top_p95"], rel=0.05
        )
        assert high.jitter["opening_top_p95"] == pytest.approx(0.005 + 0.95 * 0.004, abs=0.0005)
        assert start == 0.012
        w_start, w_full = high.options["mouth_gate_width"]
        assert w_full == pytest.approx(w_start + 0.05)
        # Three times the within-segment spread of +-0.4%: below the floor of 0.02.
        assert w_start == 0.02
        assert change_of(high, "mouth_activity_mar") is not None

    def test_his_own_jitter_reproduces_his_hand_unit(self):
        """His live still-mouth jitter read 0.0043 at p99 (day 12) and the
        unit was set to 0.004: uniform noise whose p99 lands there gives
        the unit back within a few percent."""
        result = calibrate(synthetic_samples(jitter=0.0043 / 1.9), HIS_CURRENT)
        assert result.options["mouth_activity_mar"] == pytest.approx(_MOUTH_ACTIVITY_MAR, rel=0.15)

    def test_the_brow_dead_band_is_the_confound_steps_p90(self):
        result = calibrate(synthetic_samples(), HIS_CURRENT)
        # Talking moves the synthetic brows +-12%: the band lands near his
        # hand value, and the brow gain stays where it was.
        assert result.options["brow_deadband"] == pytest.approx(0.10, abs=0.02)
        assert result.jitter["brow_confound_p90"] == pytest.approx(0.10, abs=0.02)

    def test_a_band_that_moves_corrects_the_brow_gain_with_it(self):
        current = dict(HIS_CURRENT, brow_deadband=0.05)
        result = calibrate(synthetic_samples(), current)
        # The band goes 0.05 -> ~0.10 on the same raise (+40% at p95):
        # the lift shrinks from 0.35 to 0.30, the gain grows by the ratio.
        band = result.options["brow_deadband"]
        expected = (0.40 - 0.05) / (0.40 - band)
        assert result.options["brow_follow"] == pytest.approx(expected, abs=0.06)

    def test_a_resting_width_that_drifts_past_the_gate_is_said(self):
        samples = synthetic_samples()
        for s in samples:
            if s.step == "rest_end":
                s.his["width"] *= 1.04
        result = calibrate(samples, HIS_CURRENT)
        assert result.jitter["width_drift"] == pytest.approx(0.04, abs=0.005)
        assert any("drifted" in note for note in result.notes)
        assert result.options["mouth_gate_width"][0] == 0.02  # the spread, not the drift

    def test_no_frozen_anchor_means_no_profile(self):
        result = calibrate(synthetic_samples(anchor_after=10_000), HIS_CURRENT)
        assert not result.ok
        assert any("still frames" in note for note in result.notes)
        assert not result.options


class TestGains:
    def test_gains_that_already_read_one_are_kept(self):
        result = calibrate(synthetic_samples(), HIS_CURRENT)
        for key in ("expression_gain", "eye_gain", "brow_follow", "gaze_gain"):
            assert result.options[key] == HIS_CURRENT[key], key
            assert change_of(result, key) is None, key
        assert result.response["talk_opening"]["slope"] == pytest.approx(1.0, abs=DEAD_ZONE)
        assert result.response["blink"]["slope"] == pytest.approx(1.0, abs=DEAD_ZONE)
        assert result.response["brow_raise"]["slope"] == pytest.approx(1.0, abs=0.05)
        assert result.response["gaze"]["slope"] == pytest.approx(1.0, abs=0.05)
        assert result.response["brow_raise"]["range"] == pytest.approx(0.40, abs=0.03)

    def test_a_weak_talking_mouth_raises_expression_gain_by_the_inverse(self):
        result = calibrate(synthetic_samples(talk_gain=0.8), HIS_CURRENT)
        assert result.options["expression_gain"] == pytest.approx(1.8 / 0.8, abs=0.05)
        change = change_of(result, "expression_gain")
        assert change is not None and change.before == 1.8

    def test_one_calibration_moves_a_gain_by_a_bounded_factor(self):
        _, _, change_lo, change_hi = GAIN_LIMITS["expression_gain"]
        strong = calibrate(synthetic_samples(talk_gain=2.5), HIS_CURRENT)
        assert strong.options["expression_gain"] == pytest.approx(1.8 * change_lo, abs=0.01)
        weak = calibrate(synthetic_samples(talk_gain=0.4), HIS_CURRENT)
        assert weak.options["expression_gain"] == pytest.approx(1.8 * change_hi, abs=0.01)

    def test_blink_brow_and_gaze_gains_are_corrected_the_same_way(self):
        result = calibrate(
            synthetic_samples(blink_gain=0.8, brow_gain=0.5, gaze_gain=2.0), HIS_CURRENT
        )
        assert result.options["eye_gain"] == pytest.approx(1.25, abs=0.05)
        assert result.options["brow_follow"] == pytest.approx(2.0, abs=0.15)
        assert result.options["gaze_gain"] == pytest.approx(
            GAIN_LIMITS["gaze_gain"][2], abs=0.05
        )  # 1/2 would be 0.5; the change floor is 0.6

    def test_a_drive_that_is_off_is_reported_not_switched_on(self):
        current = dict(HIS_CURRENT, brow_follow=0.0, gaze_gain=0.0)
        result = calibrate(synthetic_samples(), current)
        assert "brow_follow" not in result.options and "gaze_gain" not in result.options
        assert any("brow_follow is off" in n for n in result.notes)

    def test_without_the_avatars_reading_no_gain_is_touched(self):
        result = calibrate(synthetic_samples(with_avatar=False), HIS_CURRENT)
        assert result.ok
        assert "expression_gain" not in result.options
        assert "mouth_activity_mar" in result.options
        assert any("no gains measured" in n for n in result.notes)

    def test_the_smile_is_reported_and_not_written(self):
        result = calibrate(synthetic_samples(), HIS_CURRENT)
        assert result.response["smile_width"]["slope"] == pytest.approx(0.5, abs=0.1)
        assert "mouth_smile_gain" not in result.options


class TestSidewaysOffset:
    def test_the_avatars_eyes_off_his_become_a_crop_shift(self):
        result = calibrate(synthetic_samples(eye_offset_px=7.5), HIS_CURRENT)
        assert result.rest["avatar_eye_dx"] == pytest.approx(7.5, abs=0.1)
        # 512 crop px over 2.2 face heights of 300 px, the other way.
        assert result.options["crop_shift_x"] == pytest.approx(-7.5 * 512 / (2.2 * 300), abs=0.1)
        assert change_of(result, "crop_shift_x") is not None

    def test_a_small_offset_keeps_the_crop_where_it_is(self):
        result = calibrate(synthetic_samples(eye_offset_px=1.0), HIS_CURRENT)
        assert result.options["crop_shift_x"] == 0.0
        assert change_of(result, "crop_shift_x") is None
        assert any("sideways offset" in n for n in result.notes)

    def test_an_existing_shift_is_corrected_from(self):
        current = dict(HIS_CURRENT, crop_shift_x=-3.0)
        result = calibrate(synthetic_samples(eye_offset_px=4.0), current)
        assert result.options["crop_shift_x"] == pytest.approx(-3.0 - 4.0 * 512 / 660, abs=0.1)


class TestLeakAndLipRest:
    def test_a_clear_leak_slope_replaces_the_tables_row(self):
        leak = dict(_BROW_MOUTH_LEAK)
        leak[(20, 1)] = 0.30
        leak[(7, 1)] = 0.0  # no leak on this row: its fit is noise
        result = calibrate(synthetic_samples(leak=leak), HIS_CURRENT)
        rows = {(k, a): v for k, a, v in result.options["brow_mouth_leak_slopes"]}
        assert rows[(20, 1)] == pytest.approx(0.30, abs=0.02)
        assert rows[(17, 1)] == pytest.approx(_BROW_MOUTH_LEAK[(17, 1)], abs=0.02)
        assert rows[(7, 1)] == _BROW_MOUTH_LEAK[(7, 1)]  # kept, not fitted
        change = change_of(result, "brow_mouth_leak_slopes")
        assert change is not None and any(k == 20 for k, _, _ in change.after)

    def test_his_own_table_comes_back_within_a_few_percent(self):
        result = calibrate(synthetic_samples(), HIS_CURRENT)
        rows = {(k, a): v for k, a, v in result.options["brow_mouth_leak_slopes"]}
        for key, slope in _BROW_MOUTH_LEAK.items():
            assert rows[key] == pytest.approx(slope, abs=0.01), key

    def test_the_lip_rest_is_the_relaxed_rows_not_the_anchors(self):
        result = calibrate(synthetic_samples(), HIS_CURRENT)
        assert result.options["mouth_rest"] == "driver"
        rows = result.options["lip_rest_rows"]
        assert rows == pytest.approx([-0.001, 0.002, 0.008], abs=0.001)
        anchor = result.rest["anchor_lip_rows"]
        assert anchor == pytest.approx([0.008, 0.011, 0.017], abs=0.001)
        assert change_of(result, "lip_rest_rows") is not None
        assert change_of(result, "mouth_rest") is not None

    def test_without_engine_readings_the_driver_side_is_all_there_is(self):
        result = calibrate(synthetic_samples(with_engine=False), HIS_CURRENT)
        assert result.ok
        assert "mouth_activity_mar" in result.options
        assert "brow_mouth_leak_slopes" not in result.options
        assert "lip_rest_rows" not in result.options
        assert "expression_gain" in result.options


class TestDescribe:
    def test_the_report_names_the_changes(self):
        result = calibrate(synthetic_samples(talk_gain=0.8), HIS_CURRENT)
        text = describe(result, current=HIS_CURRENT)
        assert "Talking mouth: avatar 0.8" in text
        assert "expression_gain: 1.8 -> 2.25" in text
        assert "At rest" in text

    def test_a_failed_run_says_so(self):
        result = calibrate([], HIS_CURRENT)
        assert describe(result).startswith("Nothing measured")
