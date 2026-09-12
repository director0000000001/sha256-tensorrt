"""The calibration's steps and the mapping of a clip's segments onto them."""

from __future__ import annotations

import pytest

from sha256.calibration.script import (
    LEAD_IN,
    READ_SECONDS,
    STEPS,
    step_at,
    step_for_instruction,
    total_seconds,
)


class TestSteps:
    def test_the_script_covers_every_measurement(self):
        kinds = [step.kind for step in STEPS]
        assert kinds[0] == "lead" and kinds[1] == "rest" and kinds[-1] == "rest_end"
        for needed in (
            "blink",
            "eyes_shut",
            "open",
            "talk",
            "smile_teeth",
            "brows",
            "gaze",
            "turn",
        ):
            assert needed in kinds
        assert not LEAD_IN.measured and all(s.measured for s in STEPS[1:])
        # Every measured step opens with a read phase -- the still step too,
        # since day 16's evening (the anchor is reset as its measured phase
        # starts); the lead-in does not.
        assert STEPS[0].read_seconds == 0.0
        assert all(s.read_seconds == READ_SECONDS for s in STEPS[1:])
        assert 75.0 <= total_seconds(STEPS) <= 95.0

    def test_step_at_walks_the_clock_through_the_read_phases(self):
        first = step_at(STEPS, -1.0)
        assert first is not None
        assert (first.step_index, first.step, first.remaining, first.reading) == (
            0,
            STEPS[0],
            STEPS[0].seconds,
            False,
        )
        reading_rest = step_at(STEPS, LEAD_IN.seconds + 0.5)
        assert reading_rest is not None and reading_rest.step.kind == "rest"
        assert reading_rest.reading
        rest = step_at(STEPS, LEAD_IN.seconds + READ_SECONDS + 0.5)
        assert rest is not None and (rest.step_index, rest.step.kind, rest.reading) == (
            1,
            "rest",
            False,
        )
        assert rest.remaining == pytest.approx(STEPS[1].seconds - 0.5)
        # The blink step: first its read phase, then the movement.
        blink_start = LEAD_IN.seconds + STEPS[1].total
        reading = step_at(STEPS, blink_start + 1.0)
        assert reading is not None and reading.step.kind == "blink" and reading.reading
        assert reading.remaining == pytest.approx(READ_SECONDS - 1.0)
        moving = step_at(STEPS, blink_start + READ_SECONDS + 0.5)
        assert moving is not None and moving.step.kind == "blink" and not moving.reading
        assert moving.remaining == pytest.approx(STEPS[2].seconds - 0.5)
        assert step_at(STEPS, total_seconds(STEPS)) is None
        last = step_at(STEPS, total_seconds(STEPS) - 0.1)
        assert last is not None and last.step.kind == "rest_end" and not last.reading


class TestInstructions:
    @pytest.mark.parametrize(
        ("instruction", "kind"),
        [
            ("SIT STILL. Look straight at the camera. Relax your face.", "rest"),
            ("SIT STILL again. Relax.", "rest_end"),
            ("Look slowly LEFT, then back to centre.", "gaze"),
            ("Look UP, then DOWN, then centre.", "other"),
            ("Blink normally, several times.", "blink"),
            ("Close your eyes for two seconds, then open.", "eyes_shut"),
            ("Smile with your mouth closed.", "smile_closed"),
            ("Smile widely, showing your teeth.", "smile_teeth"),
            ("Open your mouth wide, then close it.", "open"),
            ("TALK. Say anything, normally, as if on a call.", "talk"),
            ("Turn your head slowly LEFT, then back.", "turn"),
            ("Raise your eyebrows, then frown.", "brows"),
            ("Now look at the SCREEN, as on a call. Stay still, relaxed. Do nothing.", "other"),
            ("Tilt your head SLIGHTLY down, hold two seconds, come back. Three times.", "other"),
            ("Smile, then relax.", "smile_closed"),
            ("Get ready: sit straight, face the camera, relax.", "lead"),
            ("Get ready: Blink normally, several times.", "lead"),
            ("Get ready: SIT STILL. Look straight at the camera. Relax your face.", "lead"),
            ("Get ready: SIT STILL again. Relax.", "lead"),
        ],
    )
    def test_the_recorders_scripts_find_their_steps(self, instruction, kind):
        assert step_for_instruction(instruction) == kind

    def test_the_calibrations_own_wording_maps_to_itself(self):
        for step in STEPS:
            assert step_for_instruction(step.instruction) == step.kind, step.instruction
