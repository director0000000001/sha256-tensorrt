"""The head guide (day 21, 2026-10-14): where the head should be in the
picture during "Calibrate me", drawn as an oval in the preview -- green
once the face sits in it at the right distance and facing the camera,
amber, with one short line of what to do, until then. The session's clock
does not start before it has been green for a second (see MainWindow), so
the measurements are taken from where every good session had the face.

His ask of 2026-10-14, after the friend test: "outline in green where the
person should put their head, so they do not sit too far from the camera
or too close; it must be a simple process for everyone".

The target is read off the four good sessions of 2026-09-12 (his, his
friend's two, the woman's; 894 to 1948 frames each, recordings/calibration-*):
the face's mesh -- MediaPipe's 468 points, top to chin -- stood 0.43 to
0.55 of the frame's height, centred 0.50 across and 0.42 to 0.51 down. So
the target is a mesh 0.48 of the frame's height, allowed 0.80 to 1.25 of
that (0.38 to 0.60 of the frame: farther reads "come closer", nearer
"move back"), centred within 0.12 of (0.50, 0.46), and frontal within 15
degrees of yaw and 20 of pitch -- the bounds the hairline reading itself
uses, and how he sits (pitch -13 to -15, the camera above the screen).
The oval drawn is a HEAD, not the mesh: taller by the hair and the chin's
underside, wider by the ears -- 0.62 by 0.44 of the frame's height, so it
keeps its shape whatever the frame's aspect.

Nothing here touches Qt: the fit is a pure function of the tracking, so
it is tested without a window, and the preview only draws what it is
handed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sha256.core.types import FaceTrackingResult

__all__ = ["GuideFit", "GuideState", "HeadGuide", "fit_head"]

# MediaPipe's face mesh proper; the irises past it do not bound the face.
_FACE_POINTS = 468


@dataclass(frozen=True, slots=True)
class HeadGuide:
    """The target, every number a fraction of the frame (see the module);
    the angles in degrees."""

    centre: tuple[float, float] = (0.5, 0.46)
    face_height: float = 0.48
    height_band: tuple[float, float] = (0.80, 1.25)
    centre_slack: float = 0.12
    yaw_limit: float = 15.0
    pitch_limit: float = 20.0
    oval_height: float = 0.62
    oval_width: float = 0.44


class GuideState(Enum):
    NO_FACE = "no_face"
    CLOSER = "closer"
    BACK = "back"
    CENTRE = "centre"
    FACE_CAMERA = "face_camera"
    GOOD = "good"


_MESSAGES: dict[GuideState, str] = {
    GuideState.NO_FACE: "Put your head in the oval",
    GuideState.CLOSER: "Come closer",
    GuideState.BACK: "Move back",
    GuideState.CENTRE: "Put your head in the oval",
    GuideState.FACE_CAMERA: "Face the camera",
    GuideState.GOOD: "Good. Hold still.",
}


@dataclass(frozen=True, slots=True)
class GuideFit:
    """Where the face is against the guide, and the one line to say."""

    state: GuideState
    message: str

    @property
    def good(self) -> bool:
        return self.state is GuideState.GOOD


def _answer(state: GuideState) -> GuideFit:
    return GuideFit(state, _MESSAGES[state])


def fit_head(tracking: FaceTrackingResult | None, guide: HeadGuide | None = None) -> GuideFit:
    """The face against ``guide``: the distance first (the mesh's height
    against the target's), then the centre, then the pose -- the first
    thing wrong is the one line said, since one instruction at a time is
    what a person can follow."""
    guide = HeadGuide() if guide is None else guide
    if tracking is None or not tracking.face_detected or tracking.landmarks is None:
        return _answer(GuideState.NO_FACE)
    points = tracking.landmarks.points
    if points.shape[0] < _FACE_POINTS:
        return _answer(GuideState.NO_FACE)
    face = points[:_FACE_POINTS, :2]
    low = face.min(axis=0)
    high = face.max(axis=0)
    height = float(high[1] - low[1])
    if height <= 0.0:
        return _answer(GuideState.NO_FACE)
    ratio = height / guide.face_height
    if ratio < guide.height_band[0]:
        return _answer(GuideState.CLOSER)
    if ratio > guide.height_band[1]:
        return _answer(GuideState.BACK)
    centre_x = float(low[0] + high[0]) / 2.0
    centre_y = float(low[1] + high[1]) / 2.0
    if (
        abs(centre_x - guide.centre[0]) > guide.centre_slack
        or abs(centre_y - guide.centre[1]) > guide.centre_slack
    ):
        return _answer(GuideState.CENTRE)
    pose = tracking.head_pose
    if pose is not None and (
        abs(pose.yaw) > guide.yaw_limit or abs(pose.pitch) > guide.pitch_limit
    ):
        return _answer(GuideState.FACE_CAMERA)
    return _answer(GuideState.GOOD)
