"""From a calibration's samples to the driver's options.

Every number here reproduces a hand measurement recorded in the ROADMAP,
with the rule that produced the hand value written beside it:

* **The gates' units** (days 9-12). The motion units ``mouth_activity_mar``
  / ``mouth_activity_width`` were set so that his live still-mouth jitter's
  99th percentile read about 1.1 units (0.0043 -> 0.004; 2.0% at 22 fps ->
  2.5%): ``unit = p99 / 1.1``. The distance gate ``mouth_gate_mar`` starts
  at twice the still reading's own top (the 95th percentile of his
  resting opening itself, drift included) and is full at three times its
  start (0.004 / 0.012); ``mouth_gate_width`` starts at three times the
  still spread within a still segment (+-1% -> 0.03) and is full 0.05
  later (0.08). The brow drive's dead band is the brow change a blink, a
  shut eye, a smile or a word makes (+9..13% -> 0.10): the 90th
  percentile over those steps -- a person raises the brows while talking
  now and then, and that is a raise, not a confound. Measured back on the
  clip the hand values came from (driving.mkv): 0.0044 / 0.031 / 0.079.
* **The gains** (days 8-12). Each was set by measuring the avatar's own
  tracked reading against his on the segment made for it and moving the
  gain until the response read 1.0: the talking mouth and the blink by
  the slope of a straight line through the pairs (``expression_gain``,
  ``eye_gain``); the brow raise and the sideways glance by the peaks --
  the avatar's rise at the 95th percentile over his, the avatar's glance
  range over his (``brow_follow``, ``gaze_gain``) -- because both drives
  have a dead band and the hand values were set so the avatar's whole
  raise matched his whole raise (+38% -> +34%, day 9), not the part past
  the band. The same here; the new gain is the old one over the measured
  response, kept when the response is within 5% of 1.0, and
  never moved by more than a bounded factor in one calibration -- the
  responses are close to linear (day 15's sweep) but not exactly, so a
  second calibration refines what the first got near.
* **The brow-to-mouth leak** (day 15): the extractor's reading of each
  mouth row regressed on his brow deviation over the raise, lips shut (his
  opening within twice the gate's start of rest; the width is not asked,
  a raise purses and widens the mouth a little and day 15 measured over
  the whole segment) --
  the slope replaces the table's row when the fit is clear, else the row
  keeps the value it had.
* **The sideways offset** (day 16, C2; withdrawn the same evening): the
  avatar's eyes against his on the still frames, in frame pixels, is
  reported. It was turned into ``crop_shift_x`` until the first other
  face showed the cost: the shift moves the whole rendered face, outline
  included, and the paste's join then sits inside the driver's cheek on
  one side. The two outlines coincide at zero shift; the eye offset is
  the driver's resting yaw against a frontal render, not a misplacement.
* **The lip rest** (day 15): the extractor's lip-opening rows while he is
  measurably at rest, so the next session's anchor can start from his
  relaxed mouth rather than the pressed one a session tends to begin with.

What is *not* written: the smile's width gain (``mouth_smile_gain``) --
measured and reported, because the width response is not linear and the
owner set his by eye (day 9) -- and anything about the picture.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np

from sha256.avatar.engine_liveportrait import (
    _BROW_DEADBAND,
    _MOUTH_ACTIVITY_GATE,
    _MOUTH_GATE_MAR,
    _MOUTH_GATE_WIDTH,
    _MOUTH_OPENING,
    _brow_mouth_leak_slopes,
    _gate_thresholds,
)
from sha256.calibration.samples import Sample

__all__ = ["CalibrationResult", "Change", "Fit", "calibrate", "fit_line"]

# A fit is trusted with this many paired frames, this much correlation.
FIT_MIN_FRAMES = 30
FIT_MIN_R = 0.7
LEAK_MIN_R = 0.6
# A slope within this of 1.0 leaves the gain alone.
DEAD_ZONE = 0.05
# Per gain: the absolute range the option allows, and how far one
# calibration may move it (as a factor of the current value).
GAIN_LIMITS: dict[str, tuple[float, float, float, float]] = {
    "expression_gain": (0.5, 3.0, 0.65, 1.5),
    "eye_gain": (0.3, 2.0, 0.7, 1.4),
    "brow_follow": (0.3, 3.0, 0.6, 2.0),
    "gaze_gain": (0.3, 3.0, 0.6, 2.0),
}
# The still-jitter rule and the bounds the thresholds are kept inside.
JITTER_UNITS = 1.1
GATE_MAR_START_FACTOR = 2.0
GATE_MAR_FULL_FACTOR = 3.0
GATE_WIDTH_START_FACTOR = 3.0
GATE_WIDTH_FULL_STEP = 0.05
# The floors of the motion units are the day-12 LIVE values: a deliberate
# still step reads less jitter than a person sitting "still" on a call
# (his first live run: 0.0008 against 0.0043 on the day-9 captures), and
# a unit set from it would call a call's rest "talking"; the calibration
# can only raise them. The dead band is capped at 0.15: past it a real
# raise loses too much (see the brow gain's consistency term below).
THRESHOLD_BOUNDS: dict[str, tuple[float, float]] = {
    "mouth_activity_mar": (0.004, 0.02),
    "mouth_activity_width": (0.025, 0.06),
    "mouth_gate_mar_start": (0.003, 0.012),
    "mouth_gate_width_start": (0.02, 0.06),
    "brow_deadband": (0.05, 0.15),
}
# Frames of the first still step dropped after the anchor freezes: the
# tracker and the person both settle (his first live run: the width
# spread read 4.2% with them, the closed smile's gate would have started
# at 0.06).
REST_WARMUP_FRAMES = 30
# The lips count as shut for the leak within this opening of rest, at least.
LEAK_SHUT_MIN = 0.008
BROW_CONFOUND_PERCENTILE = 90
STILL_PERCENTILE = 95
# Frames of rest wanted before the rest values and the jitter are trusted.
REST_MIN_FRAMES = 20
JITTER_MIN_FRAMES = 40
CONFOUND_MIN_FRAMES = 60
# Day 17: still frames on which the engine read a hairline, for the
# profile's driver_hairline (the engine reads one frame in six once the
# anchor has settled the reach).
HAIRLINE_MIN_FRAMES = 5
# His readings that count as a real movement on the step made for it.
TALK_MIN_RANGE = 0.02  # opening over jaw width, p95 - p5
BLINK_MIN_RANGE = 0.08  # eye ratio
BROW_MIN_RANGE = 0.15  # his rise at p95 -- a real raise is +34..38%
GAZE_MIN_RANGE = 0.10  # of eye width, p95 - p5 of his glance
LEAK_MIN_RANGE = 0.04  # brow ratio deviation, p95 - p5
# Frames of a resting mouth wanted for the lip rest rows.
LIP_REST_MIN_FRAMES = 30
_GATE_SHUT = 0.02
_BLINK_HOLD = 0.7  # of his rest eye ratio, as the gaze drive's own hold
_FRONTAL_PITCH = 10.0
_FRONTAL_YAW = 15.0
_NOMINAL_DT = 1.0 / 30.0

CONFOUND_STEPS = ("blink", "eyes_shut", "open", "talk", "smile_closed", "smile_teeth", "gaze")
REST_STEPS = ("rest", "rest_end")


@dataclass(frozen=True, slots=True)
class Fit:
    """A straight line through ``(his, avatar's)`` pairs -- ``slope`` is the
    gain, ``r`` the correlation, ``n`` the pairs, ``his_range`` how far his
    own reading moved (p95 - p5)."""

    slope: float
    intercept: float
    r: float
    n: int
    his_range: float

    def trusted(self, min_r: float = FIT_MIN_R, min_range: float = 0.0) -> bool:
        return (
            self.n >= FIT_MIN_FRAMES
            and math.isfinite(self.slope)
            and abs(self.r) >= min_r
            and self.his_range >= min_range
        )

    def as_dict(self) -> dict[str, float]:
        return {"slope": self.slope, "r": self.r, "n": float(self.n), "range": self.his_range}


def fit_line(xs: Iterable[float], ys: Iterable[float]) -> Fit:
    x = np.asarray(list(xs), dtype=np.float64)
    y = np.asarray(list(ys), dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = int(x.size)
    if n < 3 or float(x.std()) < 1e-9 or float(y.std()) < 1e-9:
        rng = float(np.percentile(x, 95) - np.percentile(x, 5)) if n else 0.0
        return Fit(math.nan, math.nan, 0.0, n, rng)
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return Fit(
        float(slope),
        float(intercept),
        r,
        n,
        float(np.percentile(x, 95) - np.percentile(x, 5)),
    )


def _peak_ratio(his: Sequence[float], avatar: Sequence[float], q: float) -> Fit:
    """The avatar's ``q``-th percentile over his -- a whole movement against
    a whole movement -- as a :class:`Fit` whose ``slope`` is that ratio,
    ``r`` the pairs' correlation and ``his_range`` his own peak."""
    line = fit_line(his, avatar)
    if line.n < 3:
        return line
    his_peak, av_peak = _p(list(his), q), _p(list(avatar), q)
    ratio = av_peak / his_peak if abs(his_peak) > 1e-6 else math.nan
    return Fit(ratio, 0.0, line.r, line.n, his_peak)


def _range_ratio(his: Sequence[float], avatar: Sequence[float]) -> Fit:
    """The avatar's p95-p5 range over his, as :func:`_peak_ratio` does it."""
    line = fit_line(his, avatar)
    if line.n < 3:
        return line
    his_range = _p(list(his), 95) - _p(list(his), 5)
    av_range = _p(list(avatar), 95) - _p(list(avatar), 5)
    ratio = av_range / his_range if his_range > 1e-6 else math.nan
    return Fit(ratio, 0.0, line.r, line.n, his_range)


@dataclass(frozen=True, slots=True)
class Change:
    key: str
    before: Any
    after: Any
    why: str


@dataclass(slots=True)
class CalibrationResult:
    rest: dict[str, float] = field(default_factory=dict)
    jitter: dict[str, float] = field(default_factory=dict)
    response: dict[str, dict[str, float]] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    changes: list[Change] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    frames: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether a profile is worth writing: the rest was measured."""
        return "opening" in self.rest


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _p(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else math.nan


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else math.nan


def _finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(v)]


def _usable(samples: Iterable[Sample], steps: Sequence[str]) -> list[Sample]:
    """Frames of ``steps`` with his face tracked, the anchor frozen and the
    face not held -- what every measurement below is made on."""
    return [
        s
        for s in samples
        if s.step in steps and s.face and s.his and s.anchor_frozen and not s.holding
    ]


def _with_avatar(samples: Iterable[Sample]) -> list[Sample]:
    return [s for s in samples if s.avatar]


def _anchor_pose(rest: Sequence[Sample], first_kind: str) -> tuple[float, float]:
    """(pitch, yaw) the still frames are measured against: the anchor's own,
    as the engine reported it on the last still frame that carried a
    reading (``rest_pitch``/``rest_yaw``, day 16), else the medians of the
    first still step's frames; 0.0 for an axis nothing says anything about."""
    last = next((s.engine for s in reversed(rest) if s.engine), {})
    pitch, yaw = last.get("rest_pitch", math.nan), last.get("rest_yaw", math.nan)
    first = [s for s in rest if s.step == first_kind] or list(rest)
    if not math.isfinite(pitch):
        pitch = _median(_finite(s.his.get("pitch", math.nan) for s in first))
    if not math.isfinite(yaw):
        yaw = _median(_finite(s.his.get("yaw", math.nan) for s in first))
    return (pitch if math.isfinite(pitch) else 0.0, yaw if math.isfinite(yaw) else 0.0)


def _frontal(sample: Sample, rest_pitch: float, rest_yaw: float = 0.0) -> bool:
    """Facing the camera the way this person does at rest -- both angles
    measured from their own still pose (day 16: a head that rests at pitch
    -11 in front of a camera is frontal there)."""
    pitch, yaw = sample.his.get("pitch", math.nan), sample.his.get("yaw", math.nan)
    if not (math.isfinite(pitch) and math.isfinite(yaw)):
        return True
    return abs(pitch - rest_pitch) < _FRONTAL_PITCH and abs(yaw - rest_yaw) < _FRONTAL_YAW


def _per_frame_changes(samples: Sequence[Sample], key: str, *, relative: bool) -> list[float]:
    """|change of ``key``| between consecutive frames, in units of a 30-fps
    frame (the engine's own ``per_frame`` scaling); ``relative`` reads the
    change as a fraction of the previous value, as the width gate does."""
    out: list[float] = []
    for prev, cur in pairwise(samples):
        if cur.frame_id != prev.frame_id + 1:
            continue
        dt = cur.timestamp - prev.timestamp
        if not (1e-3 <= dt <= 0.2):
            continue
        a, b = prev.his.get(key, math.nan), cur.his.get(key, math.nan)
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        change = abs(b / (a or 1e-6) - 1.0) if relative else abs(b - a)
        out.append(change * (_NOMINAL_DT / dt))
    return out


def _adjust(
    result: CalibrationResult,
    current: Mapping[str, Any],
    key: str,
    default: float,
    fit: Fit,
    *,
    min_range: float,
    what: str,
    extra_factor: float = 1.0,
) -> None:
    """The gain ``key`` from a response fit: kept within the dead zone,
    else the current value over the slope (times ``extra_factor``, a
    correction the caller knows about), bounded."""
    result.response[what] = fit.as_dict()
    if not fit.trusted(min_range=min_range):
        result.notes.append(
            f"{what}: not trusted (n {fit.n}, r {fit.r:.2f}, his range {fit.his_range:.3f}) "
            f"-- {key} left as it is"
        )
        return
    before = float(current.get(key, default))
    if before <= 0.0:
        result.notes.append(f"{what}: {key} is off ({before}); measured slope {fit.slope:.2f}")
        return
    if fit.slope <= 0.05:
        result.notes.append(f"{what}: slope {fit.slope:.2f} too small to correct from")
        return
    wanted = extra_factor / fit.slope
    if abs(wanted - 1.0) < DEAD_ZONE:
        result.notes.append(f"{what}: slope {fit.slope:.2f}, within {DEAD_ZONE:.0%} -- {key} kept")
        result.options[key] = before
        return
    lo, hi, change_lo, change_hi = GAIN_LIMITS[key]
    factor = _clamp(wanted, change_lo, change_hi)
    after = round(_clamp(before * factor, lo, hi), 3)
    result.options[key] = after
    if after != before:
        result.changes.append(
            Change(
                key,
                before,
                after,
                f"{what} measured {fit.slope:.2f}x his (r {fit.r:.2f}, n {fit.n})",
            )
        )


def calibrate(
    samples: Sequence[Sample], current: Mapping[str, Any] | None = None
) -> CalibrationResult:
    """The samples of one calibration into a result: the rest, the jitter,
    the response fits, and the driver options that follow. ``current`` is
    the option set the session ran with (the avatar's ``engine_options``
    with any earlier profile over them) -- the gains are corrected from it,
    and the leak table's rows it names are the ones re-measured."""
    current = dict(current or {})
    result = CalibrationResult()
    steps = sorted({s.step for s in samples})
    for kind in steps:
        result.frames[kind] = sum(1 for s in samples if s.step == kind and s.face and s.his)
    if not samples:
        result.notes.append("no frames at all")
        return result

    # --- rest --------------------------------------------------------------
    rest_all = _usable(samples, REST_STEPS)
    first_kind = REST_STEPS[0]
    # Day 16 evening: only frames facing the camera as the anchor did count
    # as still. The friend's turn ran into his last still step (yaw +44 at
    # its median, the tracker's eye ratio reading 1.07 there) and the
    # frames between the turn and the hold put his still jitter at 0.047
    # per frame -- every gate at its ceiling. The reference is the anchor's
    # own pose from the engine's readings, else the first still step's.
    ref_pitch, ref_yaw = _anchor_pose(rest_all, first_kind)
    rest_all = [s for s in rest_all if _frontal(s, ref_pitch, ref_yaw)]
    warm = [s for s in rest_all if s.step == first_kind][:REST_WARMUP_FRAMES]
    rest = (
        [s for s in rest_all if s not in warm]
        if len(rest_all) - len(warm) >= REST_MIN_FRAMES
        else rest_all
    )
    if len(rest) < REST_MIN_FRAMES:
        result.notes.append(
            f"only {len(rest)} still frames with the anchor frozen -- the rest could not be "
            "measured; sit still and frontal at the start"
        )
        return result
    for key in (
        "ear",
        "opening",
        "width",
        "brow",
        "gaze_x",
        "gaze_y",
        "pitch",
        "yaw",
        "roll",
        "face_height",
        "z",
    ):
        result.rest[key] = _median(_finite(s.his.get(key, math.nan) for s in rest))
    last_engine = next((s.engine for s in reversed(rest) if s.engine), {})
    for key in (
        "rest_ear",
        "rest_mar",
        "rest_mouth_width",
        "rest_brow",
        "rest_pitch",
        "rest_yaw",
        "face_height_rest",
        "z_rest",
    ):
        value = last_engine.get(key, math.nan)
        if math.isfinite(value):
            result.rest[f"anchor_{key[5:] if key.startswith('rest_') else key}"] = value
    rest_opening, rest_width, rest_brow = (
        result.rest["opening"],
        result.rest["width"],
        result.rest["brow"],
    )
    rest_pitch = result.rest["pitch"] if math.isfinite(result.rest["pitch"]) else 0.0
    rest_yaw = result.rest["yaw"] if math.isfinite(result.rest["yaw"]) else 0.0

    # Day 17: his hairline, as the engine read it off the still frames
    # (frame_readings["hairline"], a fraction of the forehead height) --
    # the profile's driver_hairline, so the forehead join lands under HIS
    # hair whatever the portrait's, from the first frame of every session.
    hairline = _finite(s.engine.get("hairline", math.nan) for s in rest if s.engine)
    if len(hairline) >= HAIRLINE_MIN_FRAMES:
        result.rest["hairline"] = _median(hairline)
        after = round(result.rest["hairline"], 3)
        before = current.get("driver_hairline")
        result.options["driver_hairline"] = after
        if before != after:
            result.changes.append(
                Change(
                    "driver_hairline", before, after, f"his hairline over {len(hairline)} frames"
                )
            )
    else:
        result.notes.append(
            f"only {len(hairline)} still frames read a hairline -- driver_hairline not set"
        )

    # --- jitter -> the gates' thresholds -------------------------------------
    thresholds = _gate_thresholds(current)
    d_open = _per_frame_changes(rest, "opening", relative=False)
    d_width = _per_frame_changes(rest, "width", relative=True)
    # The opening's still top: his resting opening itself, drift between
    # the two still segments included (the gate compares to the frozen
    # rest). The width's still spread: each still segment against its own
    # median, so a resting mouth that sits a little wider a minute later
    # (his does, +2..4%) is reported as drift rather than folded into the
    # gate that a closed smile (+8%) has to pass.
    open_top = [abs(s.his["opening"]) for s in rest]
    width_spread: list[float] = []
    for kind in REST_STEPS:
        segment = [s for s in rest if s.step == kind]
        if segment:
            centre = _median([s.his["width"] for s in segment])
            width_spread.extend(abs(s.his["width"] / (centre or 1e-6) - 1.0) for s in segment)
    ends = {kind: [s for s in rest if s.step == kind] for kind in REST_STEPS}
    if all(ends.values()):
        first, last = ends[REST_STEPS[0]], ends[REST_STEPS[-1]]
        result.jitter["width_drift"] = (
            _median([s.his["width"] for s in last])
            / (_median([s.his["width"] for s in first]) or 1e-6)
            - 1.0
        )
        result.jitter["opening_drift"] = _median([s.his["opening"] for s in last]) - _median(
            [s.his["opening"] for s in first]
        )
    result.jitter.update(
        {
            "opening_change_p99": _p(d_open, 99),
            "width_change_p99": _p(d_width, 99),
            "opening_top_p95": _p(open_top, STILL_PERCENTILE),
            "width_spread_p95": _p(width_spread, STILL_PERCENTILE),
            "still_frames": float(len(rest)),
        }
    )
    if len(d_open) >= JITTER_MIN_FRAMES:
        lo, hi = THRESHOLD_BOUNDS["mouth_activity_mar"]
        result.options["mouth_activity_mar"] = round(
            _clamp(_p(d_open, 99) / JITTER_UNITS, lo, hi), 4
        )
        lo, hi = THRESHOLD_BOUNDS["mouth_activity_width"]
        result.options["mouth_activity_width"] = round(
            _clamp(_p(d_width, 99) / JITTER_UNITS, lo, hi), 4
        )
        lo, hi = THRESHOLD_BOUNDS["mouth_gate_mar_start"]
        top = _p(open_top, STILL_PERCENTILE)
        start = round(_clamp(GATE_MAR_START_FACTOR * top, lo, hi), 4)
        result.options["mouth_gate_mar"] = [start, round(GATE_MAR_FULL_FACTOR * start, 4)]
        lo, hi = THRESHOLD_BOUNDS["mouth_gate_width_start"]
        spread = _p(width_spread, STILL_PERCENTILE)
        start = round(_clamp(GATE_WIDTH_START_FACTOR * spread, lo, hi), 4)
        result.options["mouth_gate_width"] = [start, round(start + GATE_WIDTH_FULL_STEP, 4)]
        drift = result.jitter.get("width_drift", 0.0)
        if drift > start:
            result.notes.append(
                f"resting mouth width drifted {drift:+.1%} between the two still steps, past "
                f"the width gate's start ({start:.3f}) -- the gate may sit part-open at rest "
                "later in a session (no rest-width tracking exists yet)"
            )
        # Talking must still read as motion at these units.
        talk = _usable(samples, ("talk",))
        talk_changes = _per_frame_changes(talk, "opening", relative=False)
        if talk_changes:
            unit = result.options["mouth_activity_mar"]
            p75 = _p(talk_changes, 75)
            result.jitter["talk_change_p75"] = p75
            if p75 / unit < _MOUTH_ACTIVITY_GATE[1]:
                result.notes.append(
                    f"talking moves the mouth {p75 / unit:.2f} units at p75 -- close to the "
                    "rest jitter; the still frames set the unit, speech may read as quieter"
                )
    else:
        result.notes.append(
            f"only {len(d_open)} consecutive still frames -- the gates' thresholds kept"
        )

    confound = [s for s in _usable(samples, CONFOUND_STEPS) if _frontal(s, rest_pitch, rest_yaw)]
    brow_dev = [abs(s.his["brow"] / (rest_brow or 1e-6) - 1.0) for s in confound]
    result.jitter["brow_confound_p90"] = _p(brow_dev, BROW_CONFOUND_PERCENTILE)
    if len(brow_dev) >= CONFOUND_MIN_FRAMES:
        lo, hi = THRESHOLD_BOUNDS["brow_deadband"]
        result.options["brow_deadband"] = round(
            _clamp(_p(brow_dev, BROW_CONFOUND_PERCENTILE), lo, hi), 3
        )
    else:
        result.notes.append(
            f"only {len(brow_dev)} frames of blinks/smiles/talk -- brow_deadband kept"
        )
    for key, before in (
        ("mouth_activity_mar", thresholds.activity_mar),
        ("mouth_activity_width", thresholds.activity_width),
        ("mouth_gate_mar", list(thresholds.mouth_gate_mar)),
        ("mouth_gate_width", list(thresholds.mouth_gate_width)),
        ("brow_deadband", thresholds.brow_deadband),
    ):
        if key in result.options and result.options[key] != before:
            result.changes.append(Change(key, before, result.options[key], "his still frames"))

    # --- the avatar's response ----------------------------------------------
    rest_av = _with_avatar(rest)
    if len(rest_av) < REST_MIN_FRAMES:
        result.notes.append(
            f"the avatar's face was read on {len(rest_av)} still frames -- no gains measured"
        )
    else:
        # The sideways offset of the avatar's eyes against his, at rest --
        # reported, not corrected (day 16 evening). It used to set
        # crop_shift_x, and on the first other face that put the paste's
        # join 13 px inside her cheek: the avatar is drawn frontal at rest
        # whatever the head's own resting yaw, so on a head resting at yaw
        # -7 the eyes sit 11-22 px toward the turn from a frontal face's
        # while the two face OUTLINES coincide within 1 crop px (measured
        # on her three captures and his two, scratch script of 2026-09-12
        # evening; on his own head, yaw -2, the eyes read +2..+4). The
        # join follows the outline, so the outline is what stays aligned.
        dx = _median(
            _finite(
                s.avatar["eye_x"] - s.his["eye_x"]
                for s in rest_av
                if s.avatar and "eye_x" in s.avatar and "eye_x" in s.his
            )
        )
        if math.isfinite(dx):
            result.rest["avatar_eye_dx"] = dx
            yaw = result.rest.get("yaw", math.nan)
            result.notes.append(
                f"the avatar's eyes sat {dx:+.1f} px from his at rest (his head at yaw "
                f"{yaw:+.1f}) -- a frontal face on a turned head; the outline is aligned, "
                "crop_shift_x not touched"
            )
        av_rest = {
            key: _median(_finite(s.avatar[key] for s in rest_av if s.avatar))
            for key in ("ear", "opening", "width", "brow", "gaze_x")
        }
        result.rest.update({f"avatar_{k}": v for k, v in av_rest.items()})

        talk = _with_avatar(_usable(samples, ("talk", "open")))
        fit = fit_line(
            (s.his["opening"] - rest_opening for s in talk),
            (s.avatar["opening"] - av_rest["opening"] for s in talk if s.avatar),
        )
        _adjust(
            result,
            current,
            "expression_gain",
            1.0,
            fit,
            min_range=TALK_MIN_RANGE,
            what="talk_opening",
        )

        blink = _with_avatar(_usable(samples, ("blink", "eyes_shut")))
        fit = fit_line((s.his["ear"] for s in blink), (s.avatar["ear"] for s in blink if s.avatar))
        _adjust(result, current, "eye_gain", 0.06, fit, min_range=BLINK_MIN_RANGE, what="blink")

        # The brow drive has a dead band, so its response is the peak of the
        # avatar's rise over the peak of his -- the whole raise against the
        # whole raise, the way day 9 read it (+38% -> +34%); the fit's r
        # says whether the avatar followed the raise's shape at all.
        brows = _with_avatar(_usable(samples, ("brows",)))
        his_rise = [s.his["brow"] / (rest_brow or 1e-6) - 1.0 for s in brows]
        av_rise = [s.avatar["brow"] / (av_rest["brow"] or 1e-6) - 1.0 for s in brows if s.avatar]
        fit = _peak_ratio(his_rise, av_rise, STILL_PERCENTILE)
        # The brow drive lifts by (his rise - the dead band): if this
        # calibration moves the band, the same gain would raise the avatar
        # less or more at the same peak -- the gain is corrected for it so
        # the measured peak ratio still comes out at 1.0.
        band_now = thresholds.brow_deadband
        band_new = float(result.options.get("brow_deadband", band_now))
        consistency = 1.0
        if fit.his_range > band_new + 0.02 and fit.his_range > band_now + 0.02:
            consistency = (fit.his_range - band_now) / (fit.his_range - band_new)
        _adjust(
            result,
            current,
            "brow_follow",
            0.0,
            fit,
            min_range=BROW_MIN_RANGE,
            what="brow_raise",
            extra_factor=consistency,
        )

        rest_ear = result.rest["ear"]
        gaze = [
            s
            for s in _with_avatar(_usable(samples, ("gaze",)))
            if s.his["ear"] >= _BLINK_HOLD * rest_ear
            and math.isfinite(s.his["gaze_x"])
            and s.avatar is not None
            and math.isfinite(s.avatar.get("gaze_x", math.nan))
        ]
        rest_gaze = result.rest["gaze_x"]
        # The gaze drive has a dead band too (_GAZE_DEADBAND): the range of
        # the avatar's glance over the range of his, left to right.
        fit = _range_ratio(
            [s.his["gaze_x"] - rest_gaze for s in gaze],
            [s.avatar["gaze_x"] - av_rest["gaze_x"] for s in gaze if s.avatar],
        )
        _adjust(result, current, "gaze_gain", 0.0, fit, min_range=GAZE_MIN_RANGE, what="gaze")

        smile = _with_avatar(_usable(samples, ("smile_closed", "smile_teeth")))
        fit = fit_line(
            (s.his["width"] / (rest_width or 1e-6) - 1.0 for s in smile),
            (s.avatar["width"] / (av_rest["width"] or 1e-6) - 1.0 for s in smile if s.avatar),
        )
        result.response["smile_width"] = fit.as_dict()

    # --- the brow-to-mouth leak ---------------------------------------------
    gate_mar_start = thresholds.mouth_gate_mar[0]
    shut = [
        s
        for s in _usable(samples, ("brows",))
        if s.engine
        and abs(s.his["opening"] - rest_opening) < max(2.0 * gate_mar_start, LEAK_SHUT_MIN)
    ]
    table = _brow_mouth_leak_slopes(current)
    if shut:
        deviation = [s.his["brow"] - rest_brow for s in shut]
        accepted = 0
        rows: dict[tuple[int, int], float] = {}
        for (k, axis), kept in table.items():
            key = f"exp_{k}_{axis}"
            fit = fit_line(deviation, (s.engine.get(key, math.nan) for s in shut))
            result.response[f"leak_{k}_{axis}"] = fit.as_dict()
            if fit.trusted(min_r=LEAK_MIN_R, min_range=LEAK_MIN_RANGE):
                rows[(k, axis)] = round(fit.slope, 4)
                accepted += 1
            else:
                rows[(k, axis)] = kept
        if accepted:
            result.options["brow_mouth_leak_slopes"] = [[k, a, v] for (k, a), v in rows.items()]
            changed = {ka: v for ka, v in rows.items() if abs(v - table[ka]) > 1e-4}
            if changed:
                result.changes.append(
                    Change(
                        "brow_mouth_leak_slopes",
                        [[k, a, table[(k, a)]] for (k, a) in changed],
                        [[k, a, v] for (k, a), v in changed.items()],
                        f"{accepted} of {len(table)} rows fitted on his brow raise, mouth shut",
                    )
                )
        else:
            result.notes.append("brow raise: no leak row fitted clearly -- the table kept")
    else:
        result.notes.append(
            "brow raise: no shut-lips frames with engine readings -- leak not measured"
        )

    # --- the lip rest -------------------------------------------------------
    resting = [
        s
        for s in rest
        if s.engine
        and s.engine.get("gate_open", 1.0) < _GATE_SHUT
        and s.engine.get("gate_mouth", 1.0) < _GATE_SHUT
        and all(math.isfinite(s.engine.get(f"exp_{k}_{a}", math.nan)) for k, a in _MOUTH_OPENING)
    ]
    if len(resting) >= LIP_REST_MIN_FRAMES:
        rows_rest = [
            round(_median([s.engine[f"exp_{k}_{a}"] for s in resting]), 4)
            for k, a in _MOUTH_OPENING
        ]
        anchor_rows = [
            _median(_finite(s.engine.get(f"anchor_{k}_{a}", math.nan) for s in resting))
            for k, a in _MOUTH_OPENING
        ]
        result.options["lip_rest_rows"] = rows_rest
        result.options["mouth_rest"] = "driver"
        result.rest["lip_rest_rows"] = rows_rest  # type: ignore[assignment]
        result.rest["anchor_lip_rows"] = anchor_rows  # type: ignore[assignment]
        before_rows = current.get("lip_rest_rows")
        if before_rows is None or any(
            abs(float(a) - b) > 5e-4 for a, b in zip(before_rows, rows_rest, strict=False)
        ):
            result.changes.append(
                Change(
                    "lip_rest_rows",
                    before_rows,
                    rows_rest,
                    f"his relaxed mouth over {len(resting)} still frames "
                    f"(the anchor's own: {[round(v, 4) for v in anchor_rows]})",
                )
            )
        if current.get("mouth_rest", "anchor") != "driver":
            result.changes.append(
                Change(
                    "mouth_rest", current.get("mouth_rest", "anchor"), "driver", "the rows above"
                )
            )
    else:
        result.notes.append(
            f"only {len(resting)} still frames with the mouth gates shut -- lip rest not measured"
        )
    return result


def describe(result: CalibrationResult, *, current: Mapping[str, Any] | None = None) -> str:
    """The result as a few readable lines -- the window's dialog and the
    session's ``report.txt``."""
    current = dict(current or {})
    lines: list[str] = []
    if not result.ok:
        lines.append("Nothing measured.")
        lines.extend(f"  - {note}" for note in result.notes)
        return "\n".join(lines)
    rest = result.rest
    lines.append(
        "At rest: eyes {ear:.3f}, mouth opening {opening:.4f}, mouth width {width:.3f}, "
        "brow {brow:.3f}".format(**rest)
    )
    if "lip_rest_rows" in rest:
        lines.append(f"Relaxed lips (extractor rows): {rest['lip_rest_rows']}")
    j = result.jitter
    if "opening_change_p99" in j:
        lines.append(
            f"Still jitter per frame (p99): opening {j['opening_change_p99']:.4f}, "
            f"width {j['width_change_p99']:.3%}; still top opening "
            f"{j.get('opening_top_p95', math.nan):.4f}, width spread "
            f"{j.get('width_spread_p95', math.nan):.1%}; brow confound "
            f"{j.get('brow_confound_p90', math.nan):.1%}"
        )
    for what, label in (
        ("talk_opening", "Talking mouth"),
        ("blink", "Blink"),
        ("brow_raise", "Brow raise"),
        ("gaze", "Sideways glance"),
        ("smile_width", "Smile width"),
    ):
        fit = result.response.get(what)
        if fit:
            lines.append(
                f"{label}: avatar {fit['slope']:.2f}x his "
                f"(r {fit['r']:.2f}, {int(fit['n'])} frames)"
            )
    if result.changes:
        lines.append("Changed:")
        for change in result.changes:
            lines.append(f"  - {change.key}: {change.before} -> {change.after}  ({change.why})")
    else:
        lines.append("Nothing changed: the profile already matches this face.")
    kept = sorted(k for k in result.options if all(c.key != k for c in result.changes))
    if kept:
        lines.append(f"Confirmed: {', '.join(kept)}")
    if result.notes:
        lines.append("Notes:")
        lines.extend(f"  - {note}" for note in result.notes)
    return "\n".join(lines)


_ = (_BROW_DEADBAND, _MOUTH_GATE_MAR, _MOUTH_GATE_WIDTH)  # the constants the rules above cite
