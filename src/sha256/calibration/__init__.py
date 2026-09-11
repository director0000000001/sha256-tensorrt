""" "Calibrate me" (day 16, 2026-09-12): the driver's own profile.

Fifteen days of tuning the neural engine were done on one face -- the
owner's -- by hand: how far his mouth jitters at rest, how much his brow
raise leaks into the extractor's mouth, the gain that brings his talking
mouth to the size of the avatar's, his relaxed lips. Five other people are
going to sit in front of their own cameras, and none of those numbers is
theirs. This package measures them in under a minute, with the same
instruments the engine and ``scripts/measure_follow.py`` use, and writes a
*driver profile* -- ``configs/local.driver.json`` -- that the engine lays
over whichever avatar is loaded (see ``DRIVER_OPTION_KEYS`` in the engine:
the profile describes the person, the avatar's ``metadata.json`` the
picture).

The parts:

* :mod:`.script` -- the guided steps (sit still, blink, open the mouth,
  talk, smile, raise the brows, glance, turn) and how a recorded clip's
  segments map onto them.
* :mod:`.samples` -- one frame's readings: his (the tracker's), the
  engine's (gates, the extractor's mouth rows) and the avatar's (a second
  tracker on the composite).
* :mod:`.measure` -- the samples into thresholds, slopes and gains, with
  the fits' own quality beside each, and the driver options to write.
* :mod:`.profile` -- the file.
* :mod:`.session` -- the live run in the window: it records its own
  session (the camera frames as a clip, the tracking, the segment map)
  under ``recordings/`` so the same face can be re-measured offline after
  any change, without the person present.
* :mod:`.offline` -- a recorded clip (a calibration session's, or one of
  the owner's ``scripts/record_driving_clip.py`` clips) through the same
  measurement.

Privacy: a session is the person's face. It is written under a gitignored
directory on this machine only, with their consent asked in the window,
and is theirs to delete.
"""

from sha256.calibration.measure import CalibrationResult, calibrate
from sha256.calibration.profile import PROFILE_FILENAME, DriverProfile, driver_options_from
from sha256.calibration.script import STEPS, CalibrationStep, step_at, step_for_instruction
from sha256.calibration.session import CalibrationSession, SessionFiles

__all__ = [
    "PROFILE_FILENAME",
    "STEPS",
    "CalibrationResult",
    "CalibrationSession",
    "CalibrationStep",
    "DriverProfile",
    "SessionFiles",
    "calibrate",
    "driver_options_from",
    "step_at",
    "step_for_instruction",
]
