"""The guided steps of "Calibrate me", and how a clip's segments map to them.

Each step exists for one measurement (see :mod:`sha256.calibration.measure`):

* ``rest`` -- the anchor is learned here, then the still frames after it
  give the rest values and the jitter the gates' thresholds are set from.
* ``blink`` / ``eyes_half`` / ``eyes_shut`` -- the blink gain (``eye_gain``),
  and (day 22) the eye's response at a second level, halfway.
* ``open_small`` / ``open`` / ``read`` / ``laugh`` -- the talking mouth's
  gain (``expression_gain``). ``read`` (day 22) replaces ``talk``: a fixed
  text read aloud (:data:`READING_TEXT`), full of "o" and "m" -- the
  mouth's round and shut shapes; ``open_small`` and ``laugh`` are the
  opening's other levels.
* ``smile_small`` / ``smile_closed`` / ``smile_teeth`` -- the smile's width
  at three levels, reported.
* ``brows_half`` / ``brows`` -- the brow drive's gain (``brow_follow``) and
  what the raise leaks into the extractor's mouth rows
  (``brow_mouth_leak_slopes``).
* ``gaze`` -- the iris drive's gain (``gaze_gain``).
* ``turn`` -- recorded for the offline measurement (the turn lag), not
  measured here.
* ``rest_end`` -- more still frames, and the relaxed lip rest.

Day 22 (2026-10-15), the owner's order: SLOW and dead simple, so anyone
follows it -- not "laugh" but "on the count of three, laugh". Every
measured step has three phases: the instruction (``read_seconds``, on
screen and spoken), a visible count "3", "2", "1" (``count_seconds``, one a
second, spoken where a voice exists), then the measured window. Only the
window is measured; the first two are "reading" (see :class:`StepPosition`).

The keyword table below still reads ``scripts/record_driving_clip.py``'s
wordings and this script's older ones, so a recorded calibration session
and the owner's own clips read the same way in ``measure_follow.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

__all__ = [
    "COUNT_SECONDS",
    "LEAD_IN",
    "PHASE_COUNT",
    "PHASE_MEASURE",
    "PHASE_READ",
    "READING_TEXT",
    "READ_SECONDS",
    "REPEAT_PREFIX",
    "STEPS",
    "CalibrationStep",
    "StepPosition",
    "repeat_of",
    "step_at",
    "step_for_instruction",
    "total_seconds",
]


# His first run (2026-09-12 00:41): "you give me very little time -- I have
# to read and then do the movement". Every movement step opens with an
# instruction phase of this many seconds, the instruction on screen (and
# spoken, where a voice exists) and nothing measured. Day 22: three
# seconds, since the count of three that follows gives the time to get
# ready.
READ_SECONDS = 3.0
# Day 22: then "3", "2", "1", one a second, drawn large -- the movement
# starts on the end of the count, as everyone knows from a camera's timer.
COUNT_SECONDS = 3.0

PHASE_READ = "read"
PHASE_COUNT = "count"
PHASE_MEASURE = "measure"

# Day 22: the repeat of a step whose movement was too small to measure
# (see CalibrationSession) says so first.
REPEAT_PREFIX = "Once more: "

# Day 22, the read step's text: twenty seconds of reading at an ordinary
# pace, full of round "o" and "oo" (the mouth's O, for its shape) and shut
# "m" (the lips together) between open vowels.
READING_TEXT = (
    "Oh, hello. My name is Emma, and I am at home. Open the window, Mom. "
    "Over the ocean, the moon moves slowly. Oh no, my old boat is gone. Amen. "
    "Every morning I make mango jam and more tea. Oh, okay, I am done now."
)


@dataclass(frozen=True, slots=True)
class CalibrationStep:
    kind: str
    seconds: float
    instruction: str
    measured: bool = True
    read_seconds: float = READ_SECONDS
    count_seconds: float = COUNT_SECONDS

    @property
    def total(self) -> float:
        return self.read_seconds + self.count_seconds + self.seconds


class StepPosition(NamedTuple):
    """Where the clock is: the step, the seconds left in the current phase,
    whether that phase is unmeasured (the instruction or the count -- both
    "reading" for the session), and which phase it is: :data:`PHASE_READ`,
    :data:`PHASE_COUNT` or :data:`PHASE_MEASURE` (``""`` from a caller that
    did not say; then ``reading`` tells the two apart)."""

    step_index: int
    step: CalibrationStep
    remaining: float
    reading: bool
    phase: str = ""


def _step(kind: str, seconds: float, instruction: str) -> CalibrationStep:
    return CalibrationStep(kind, seconds, instruction)


LEAD_IN = CalibrationStep(
    "lead",
    3.0,
    "Get ready: sit straight, face the camera, relax.",
    measured=False,
    read_seconds=0.0,
    count_seconds=0.0,
)

# Day 22 evening, his first session on this script: "the turn and some
# tasks give me little time -- add 5 seconds to each". Every measured
# window is 5 s longer than the morning's (the turn's own frames ended
# with his head still at -39 degrees, not back at the camera).
# Day 22: the owner's script, slow. The still step is the longest of the
# short ones: the anchor takes about a second of still frames and the
# jitter wants three more; the brows step is long enough for the leak's
# thirty shut-lips frames. The anchor is reset when the still step's
# measured window starts, not when its instruction appears (day 16
# evening: the friend was being told what to do during the still frames).
STEPS: tuple[CalibrationStep, ...] = (
    LEAD_IN,
    _step("rest", 11.0, "On three: SIT STILL, look at the camera, relax your face."),
    _step("blink", 9.0, "On three: blink normally, several times."),
    _step("eyes_half", 9.0, "On three: close your eyes HALFWAY and hold them there."),
    _step("eyes_shut", 9.0, "On three: close your eyes and keep them closed."),
    _step("open_small", 9.0, "On three: open your mouth a LITTLE, as if saying 'oh', and hold."),
    _step("open", 9.0, "On three: open your mouth WIDE, hold, then close. Twice."),
    _step("read", 25.0, "On three: read the text aloud, at your own pace."),
    _step("smile_small", 9.0, "On three: smile just a LITTLE, lips closed, and hold."),
    _step("smile_closed", 9.0, "On three: smile with your mouth closed, and hold."),
    _step("smile_teeth", 9.0, "On three: smile WIDE, showing your teeth, and hold."),
    _step("laugh", 10.0, "On three: LAUGH, as if at a good joke."),
    _step("brows_half", 9.0, "On three: raise your eyebrows a LITTLE and hold."),
    _step("brows", 10.0, "On three: raise your eyebrows HIGH, hold, relax. Twice."),
    _step(
        "gaze", 10.0, "On three, eyes only: look LEFT, then RIGHT, then at the camera. Head still."
    ),
    _step(
        "turn", 11.0, "On three: turn your head slowly LEFT, back to the camera, then RIGHT, back."
    ),
    _step("rest_end", 9.0, "On three: SIT STILL again. Relax."),
)

# In order: the first keyword found in the instruction (lower-cased) names
# the step, so "get ready" (a read phase, nothing measured) must come before
# every movement it names, "sit still again" before "sit still", "halfway"
# before "close your eyes", the "a little" levels before their full
# movement, "teeth" before "smile". Covers this script, its day-16
# wordings, the recorder's "full" and "natural" scripts, and their older
# spellings.
_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("get ready", "lead"),
    ("sit still again", "rest_end"),
    ("still again", "rest_end"),
    ("sit still", "rest"),
    ("halfway", "eyes_half"),
    ("close your eyes", "eyes_shut"),
    ("blink", "blink"),
    ("mouth a little", "open_small"),
    ("mouth wide", "open"),
    ("read the text", "read"),
    ("talk", "talk"),
    ("laugh", "laugh"),
    ("teeth", "smile_teeth"),
    ("smile just a little", "smile_small"),
    ("mouth closed", "smile_closed"),
    ("smile", "smile_closed"),
    ("eyebrows a little", "brows_half"),
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


def repeat_of(step: CalibrationStep) -> CalibrationStep:
    """The same step once more, its instruction saying so."""
    return CalibrationStep(
        step.kind,
        step.seconds,
        REPEAT_PREFIX + step.instruction,
        measured=step.measured,
        read_seconds=step.read_seconds,
        count_seconds=step.count_seconds,
    )


def total_seconds(steps: tuple[CalibrationStep, ...] = STEPS) -> float:
    return float(sum(step.total for step in steps))


def step_at(steps: tuple[CalibrationStep, ...], elapsed: float) -> StepPosition | None:
    """Where ``elapsed`` seconds from the start falls: the step, the seconds
    left in its current phase, whether that phase is unmeasured, and which
    phase it is; ``None`` once every step is over. Before the start (a
    negative ``elapsed``) it is the first step, whole."""
    at = 0.0
    for index, step in enumerate(steps):
        read_end = at + step.read_seconds
        count_end = read_end + step.count_seconds
        end = count_end + step.seconds
        if step.read_seconds > 0.0 and elapsed < read_end:
            return StepPosition(index, step, read_end - max(elapsed, at), True, PHASE_READ)
        if step.count_seconds > 0.0 and elapsed < count_end:
            return StepPosition(index, step, count_end - max(elapsed, read_end), True, PHASE_COUNT)
        if elapsed < end:
            return StepPosition(index, step, end - max(elapsed, count_end), False, PHASE_MEASURE)
        at = end
    return None
