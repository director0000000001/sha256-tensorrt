"""One frame of a calibration: what his face read, what the engine made of
it, and what the avatar's face read afterwards.

The same instruments as the engine's own -- ``_eye_ratios``,
``_mouth_open_ratio`` (the day-12 opening over the jaw width, what the
gates read), ``_mouth_width_ratio``, ``_brow_ratio``, ``_gaze_offset`` --
so a threshold measured here is in the units the gate compares against,
not a proxy for them. The avatar's reading is the second tracker's, on the
composited output, exactly as ``scripts/measure_follow.py`` reads it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sha256.avatar.engine_liveportrait import (
    FACE_POINTS,
    _brow_ratio,
    _eye_ratios,
    _face_height,
    _gaze_offset,
    _lip_ratio,
    _mouth_open_ratio,
    _mouth_width_ratio,
)
from sha256.core.types import FaceTrackingResult

__all__ = ["Sample", "avatar_measures", "his_measures"]


@dataclass(slots=True)
class Sample:
    frame_id: int
    timestamp: float
    step: str
    face: bool
    his: dict[str, float] = field(default_factory=dict)
    engine: dict[str, float] = field(default_factory=dict)
    # None: the output was not read this frame; {}: read, no face found.
    avatar: dict[str, float] | None = None

    @property
    def anchor_frozen(self) -> bool:
        """True once the engine's anchor is frozen -- and for an engine that
        reports nothing (no readings), taken as frozen: the readings that
        need the anchor are absent anyway."""
        value = self.engine.get("anchor_frozen")
        return True if value is None else value >= 1.0

    @property
    def holding(self) -> bool:
        return self.engine.get("holding", 0.0) >= 1.0


def _measures(tracking: FaceTrackingResult, width: int, height: int) -> dict[str, float]:
    assert tracking.landmarks is not None
    points = tracking.landmarks.points[:, :2].astype(np.float64) * (float(width), float(height))
    face = points[:FACE_POINTS]
    left, right = _eye_ratios(face)
    eye_centre = face[[33, 133, 362, 263]].mean(axis=0)
    out = {
        "eye_x": float(eye_centre[0]),
        "eye_y": float(eye_centre[1]),
        "nose_x": float(face[1, 0]),
        "nose_y": float(face[1, 1]),
        "ear": (left + right) / 2.0,
        "opening": _mouth_open_ratio(face),
        "width": _mouth_width_ratio(face),
        "brow": _brow_ratio(face),
        "lip": _lip_ratio(face),
        "face_height": _face_height(face),
    }
    gaze = _gaze_offset(points)
    out["gaze_x"], out["gaze_y"] = (math.nan, math.nan) if gaze is None else gaze
    pose = tracking.head_pose
    if pose is None:
        out["yaw"] = out["pitch"] = out["roll"] = out["z"] = math.nan
    else:
        out["yaw"], out["pitch"], out["roll"] = float(pose.yaw), float(pose.pitch), float(pose.roll)
        out["z"] = abs(float(pose.translation[2]))
    return out


def his_measures(tracking: FaceTrackingResult, width: int, height: int) -> dict[str, float]:
    """His readings this frame, in the engine's own units; ``{}`` without a
    face or with fewer points than a face."""
    if (
        not tracking.face_detected
        or tracking.landmarks is None
        or tracking.landmarks.count < FACE_POINTS
    ):
        return {}
    return _measures(tracking, width, height)


def avatar_measures(
    tracking: FaceTrackingResult, width: int, height: int
) -> dict[str, float] | None:
    """The avatar's readings on the composited output -- the same measures --
    or ``None`` when the second tracker found no face there."""
    if (
        not tracking.face_detected
        or tracking.landmarks is None
        or tracking.landmarks.count < FACE_POINTS
    ):
        return None
    return _measures(tracking, width, height)
