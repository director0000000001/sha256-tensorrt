"""The calibration's steps and the mapping of a clip's segments onto them."""

from __future__ import annotations

import pytest

from sha256.calibration.script import (
    COUNT_SECONDS,
    LEAD_IN,
    PHASE_COUNT,
    PHASE_MEASURE,
    PHASE_READ,
    READ_SECONDS,
    READING_TEXT,
    REPEAT_PREFIX,
    STEPS,
    CalibrationStep,
    StepPosition,
    repeat_of,
    step_at,
    step_for_instruction,
    total_seconds,
)


class TestSteps:
    def test_the_script_is_the_owners_order(self):
        """Day 22: the kinds, in his order -- the contract for the
        measurement code."""
        assert [step.kind for step in STEPS] == [
            "lead",
            "rest",
            "blink",
            "eyes_half",
            "eyes_shut",
            "open_small",
            "open",
            "read",
            "smile_small",
            "smile_closed",
            "smile_teeth",
            "laugh",
            "brows_half",
            "brows",
            "gaze",
            "turn",
            "rest_end",
        ]
        seconds = {step.kind: step.seconds for step in STEPS}
        # day 22 evening: his first session asked for 5 s more on every movement
        assert seconds["rest"] == 11.0 and seconds["read"] == 25.0 and seconds["laugh"] == 10.0
        assert seconds["turn"] == 11.0 and seconds["rest_end"] == 9.0

    def test_every_measured_step_reads_then_counts_three_then_measures(self):
        assert not LEAD_IN.measured and all(s.measured for s in STEPS[1:])
        assert (LEAD_IN.read_seconds, LEAD_IN.count_seconds) == (0.0, 0.0)
        assert READ_SECONDS == 3.0 and COUNT_SECONDS == 3.0
        for step in STEPS[1:]:
            assert step.read_seconds == READ_SECONDS and step.count_seconds == COUNT_SECONDS
            assert step.instruction.startswith("On three")
            assert step.total == pytest.approx(6.0 + step.seconds)
        assert total_seconds(STEPS) == pytest.approx(266.0)

    def test_the_reading_text_is_his(self):
        assert READING_TEXT.startswith("Oh, hello. My name is Emma, and I am at home.")
        assert READING_TEXT.endswith("Oh, okay, I am done now.")
        assert "Over the ocean, the moon moves slowly." in READING_TEXT

    def test_a_repeat_says_so_and_keeps_the_step(self):
        blink = STEPS[2]
        again = repeat_of(blink)
        assert again.instruction == REPEAT_PREFIX + blink.instruction
        assert (again.kind, again.seconds, again.read_seconds, again.count_seconds) == (
            blink.kind,
            blink.seconds,
            blink.read_seconds,
            blink.count_seconds,
        )
        assert step_for_instruction(again.instruction) == "blink"

    def test_step_at_walks_the_clock_through_the_three_phases(self):
        first = step_at(STEPS, -1.0)
        assert first is not None
        assert first == StepPosition(0, STEPS[0], STEPS[0].seconds, False, PHASE_MEASURE)
        rest_start = LEAD_IN.seconds
        reading = step_at(STEPS, rest_start + 0.5)
        assert reading is not None and reading.step.kind == "rest"
        assert reading.reading and reading.phase == PHASE_READ
        assert reading.remaining == pytest.approx(READ_SECONDS - 0.5)
        # The count: still unmeasured ("reading" for the session), its own
        # phase for the window, the seconds left the number shown.
        counting = step_at(STEPS, rest_start + READ_SECONDS + 0.2)
        assert counting is not None and counting.step.kind == "rest"
        assert counting.reading and counting.phase == PHASE_COUNT
        assert counting.remaining == pytest.approx(COUNT_SECONDS - 0.2)
        last_count = step_at(STEPS, rest_start + READ_SECONDS + COUNT_SECONDS - 0.5)
        assert last_count is not None and last_count.phase == PHASE_COUNT
        assert last_count.remaining == pytest.approx(0.5)
        rest = step_at(STEPS, rest_start + READ_SECONDS + COUNT_SECONDS + 0.5)
        assert rest is not None
        assert (rest.step_index, rest.step.kind, rest.reading, rest.phase) == (
            1,
            "rest",
            False,
            PHASE_MEASURE,
        )
        assert rest.remaining == pytest.approx(STEPS[1].seconds - 0.5)
        blink_start = LEAD_IN.seconds + STEPS[1].total
        moving = step_at(STEPS, blink_start + READ_SECONDS + COUNT_SECONDS + 0.5)
        assert moving is not None and moving.step.kind == "blink" and not moving.reading
        assert moving.remaining == pytest.approx(STEPS[2].seconds - 0.5)
        assert step_at(STEPS, total_seconds(STEPS)) is None
        last = step_at(STEPS, total_seconds(STEPS) - 0.1)
        assert last is not None and last.step.kind == "rest_end" and not last.reading

    def test_a_step_without_phases_is_measured_at_once(self):
        steps = (CalibrationStep("rest", 1.0, "SIT STILL", read_seconds=0.0, count_seconds=0.0),)
        placed = step_at(steps, 0.1)
        assert placed is not None and placed.phase == PHASE_MEASURE and not placed.reading


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
            ("Open your mouth wide, then close it. Twice.", "open"),
            ("TALK. Say anything, normally, as if on a call.", "talk"),
            ("Turn your head slowly LEFT, then back.", "turn"),
            ("Turn your head slowly LEFT, back, then RIGHT, back.", "turn"),
            ("Raise your eyebrows, then frown.", "brows"),
            ("Raise your eyebrows high, hold, relax. Twice.", "brows"),
            ("Eyes only: look LEFT, then RIGHT, then at the camera. Head still.", "gaze"),
            ("Now look at the SCREEN, as on a call. Stay still, relaxed. Do nothing.", "other"),
            ("Tilt your head SLIGHTLY down, hold two seconds, come back. Three times.", "other"),
            ("Smile, then relax.", "smile_closed"),
            ("Get ready: sit straight, face the camera, relax.", "lead"),
            ("Get ready: Blink normally, several times.", "lead"),
            ("Get ready: SIT STILL. Look straight at the camera. Relax your face.", "lead"),
            ("Get ready: SIT STILL again. Relax.", "lead"),
            ("Get ready: On three: read the text aloud, at your own pace.", "lead"),
            ("Once more: On three: smile just a LITTLE, lips closed, and hold.", "smile_small"),
        ],
    )
    def test_the_recorders_scripts_find_their_steps(self, instruction, kind):
        assert step_for_instruction(instruction) == kind

    @pytest.mark.parametrize("step", STEPS, ids=[step.kind for step in STEPS])
    def test_the_calibrations_own_wording_maps_to_itself(self, step):
        assert step_for_instruction(step.instruction) == step.kind
        assert step_for_instruction(repeat_of(step).instruction) == step.kind
