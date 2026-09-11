"""The guided steps of "Calibrate me", and how a clip's segments map to them.

Each step exists for one measurement (see :mod:`sha256.calibration.measure`):

* ``rest`` -- the anchor is learned here, then the still frames after it
  give the rest values and the jitter the gates' thresholds are set from.
* ``blink`` / ``eyes_shut`` -- the blink gain (``eye_gain``).
* ``open`` / ``talk`` -- the talking mouth's gain (``expression_gain``).
* ``smile_closed`` / ``smile_teeth`` -- the smile's width, reported.
* ``brows`` -- the brow drive's gain (``brow_follow``) and what the raise
  leaks into the extractor's mouth rows (``brow_mouth_leak_slopes``).
* ``gaze`` -- the iris drive's gain (``gaze_gain``).
* ``turn`` -- recorded for the offline measurement (the turn lag), not
  measured here.
* ``rest_end`` -- more still frames, and the relaxed lip rest.

The wording repeats ``scripts/record_driving_clip.py`` where the movement
is the same, so a recorded calibration session and the owner's own clips
read the same way in ``measure_follow.py``; the keyword table below is how
either kind of clip's segment finds its step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

__all__ = [
    "LEAD_IN",
    "READ_SECONDS",
    "STEPS",
    "CalibrationStep",
    "StepPosition",
    "step_at",
    "step_for_instruction",
    "total_seconds",
]


# His first run (2026-09-12 00:41): "you give me very little time -- I have
# to read and then do the movement". Every movement step now opens with a
# read phase of this many seconds, the instruction on screen (and spoken,
# where a voice exists) and nothing measured; the movement's own seconds
# start after it.
READ_SECONDS = 4.0


@dataclass(frozen=True, slots=True)
class CalibrationStep:
    kind: str
    seconds: float
    instruction: str
    measured: bool = True
    read_seconds: float = 0.0

    @property
    def total(self) -> float:
        return self.read_seconds + self.seconds


class StepPosition(NamedTuple):
    """Where the clock is: the step, the seconds left in the current phase,
    and whether that phase is the read phase (nothing measured yet)."""

    step_index: int
    step: CalibrationStep
    remaining: float
    reading: bool


LEAD_IN = CalibrationStep(
    "lead", 3.0, "Get ready: sit straight, face the camera, relax.", measured=False
)

# About 85 seconds with the read phases. The still step is longest: the
# anchor takes about a second of frontal frames and the jitter wants three
# more; the brows step is long enough for the leak's thirty shut-lips frames.
STEPS: tuple[CalibrationStep, ...] = (
    LEAD_IN,
    CalibrationStep("rest", 5.0, "SIT STILL. Look straight at the camera. Relax your face."),
    CalibrationStep("blink", 3.0, "Blink normally, several times.", read_seconds=READ_SECONDS),
    CalibrationStep(
        "eyes_shut", 3.0, "Close your eyes for two seconds, then open.", read_seconds=READ_SECONDS
    ),
    CalibrationStep(
        "open", 3.0, "Open your mouth wide, then close it. Twice.", read_seconds=READ_SECONDS
    ),
    CalibrationStep(
        "talk", 7.0, "TALK. Say anything, normally, as if on a call.", read_seconds=READ_SECONDS
    ),
    CalibrationStep(
        "smile_closed", 3.0, "Smile with your mouth closed.", read_seconds=READ_SECONDS
    ),
    CalibrationStep(
        "smile_teeth", 3.0, "Smile widely, showing your teeth.", read_seconds=READ_SECONDS
    ),
    CalibrationStep(
        "brows", 5.0, "Raise your eyebrows high, hold, relax. Twice.", read_seconds=READ_SECONDS
    ),
    CalibrationStep(
        "gaze",
        4.0,
        "Eyes only: look LEFT, then RIGHT, then at the camera. Head still.",
        read_seconds=READ_SECONDS,
    ),
    CalibrationStep(
        "turn",
        5.0,
        "Turn your head slowly LEFT, back, then RIGHT, back.",
        read_seconds=READ_SECONDS,
    ),
    CalibrationStep("rest_end", 3.0, "SIT STILL again. Relax.", read_seconds=READ_SECONDS),
)

# In order: the first keyword found in the instruction (lower-cased) names
# the step, so "sit still again" must come before "sit still" and "teeth"
# before "smile". Covers this script, the recorder's "full" and "natural"
# scripts, and their older spellings.
_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("sit still again", "rest_end"),
    ("still again", "rest_end"),
    ("sit still", "rest"),
    ("get ready", "lead"),
    ("close your eyes", "eyes_shut"),
    ("blink", "blink"),
    ("mouth wide", "open"),
    ("talk", "talk"),
    ("teeth", "smile_teeth"),
    ("mouth closed", "smile_closed"),
    ("smile", "smile_closed"),
    ("eyebrows", "brows"),
    ("eyes only", "gaze"),
    ("look slowly left", "gaze"),
    ("look slowly right", "gaze"),
    ("turn your head", "turn"),
)


def step_for_instruction(instruction: str) -> str:
    """The step kind a segment's instruction belongs to, or ``"other"`` for
    a movement no step measures (a look up, a tilt, a screen posture)."""
    text = instruction.lower()
    for keyword, kind in _KEYWORDS:
        if keyword in text:
            return kind
    return "other"


def total_seconds(steps: tuple[CalibrationStep, ...] = STEPS) -> float:
    return float(sum(step.total for step in steps))


def step_at(steps: tuple[CalibrationStep, ...], elapsed: float) -> StepPosition | None:
    """Where ``elapsed`` seconds from the start falls: the step, the seconds
    left in its current phase, and whether that is the read phase; ``None``
    once every step is over. Before the start (a negative ``elapsed``) it
    is the first step, whole."""
    at = 0.0
    for index, step in enumerate(steps):
        read_end = at + step.read_seconds
        end = read_end + step.seconds
        if step.read_seconds > 0.0 and elapsed < read_end:
            return StepPosition(index, step, read_end - max(elapsed, at), True)
        if elapsed < end:
            return StepPosition(index, step, end - max(elapsed, read_end), False)
        at = end
    return None
