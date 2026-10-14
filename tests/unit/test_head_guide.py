"""The head guide (day 21): the fit of a tracked face against the target --
the distance first, then the centre, then the pose -- as a pure function."""

from __future__ import annotations

import numpy as np

from sha256.core.types import FaceTrackingResult, FrameMetadata, HeadPose, LandmarkSet
from sha256.ui.head_guide import GuideFit, GuideState, HeadGuide, fit_head


def mesh(
    *,
    height: float = 0.48,
    centre: tuple[float, float] = (0.5, 0.46),
    pose: HeadPose | None = None,
    count: int = 478,
    face: bool = True,
) -> FaceTrackingResult:
    """A face whose 468 mesh points span ``height`` of the frame around
    ``centre`` (the width 0.45 of the height, as a real mesh reads)."""
    metadata = FrameMetadata(frame_id=1, capture_timestamp=1.0)
    if not face:
        return FaceTrackingResult.no_face(metadata)
    rng = np.random.default_rng(0)
    points = rng.uniform(-0.5, 0.5, size=(count, 3)).astype(np.float32)
    points[:, 0] = points[:, 0] * height * 0.45 + centre[0]
    points[:, 1] = points[:, 1] * height + centre[1]
    # pin the extremes so the span is exact
    points[0, 1] = centre[1] - height / 2.0
    points[1, 1] = centre[1] + height / 2.0
    points[2, 0] = centre[0] - height * 0.45 / 2.0
    points[3, 0] = centre[0] + height * 0.45 / 2.0
    return FaceTrackingResult(
        metadata=metadata,
        face_detected=True,
        landmarks=LandmarkSet(points=points, scheme="mediapipe_478"),
        head_pose=pose,
        confidence=1.0,
    )


class TestFit:
    def test_no_face_no_landmarks_or_too_few_points(self):
        assert fit_head(None).state is GuideState.NO_FACE
        assert fit_head(mesh(face=False)).state is GuideState.NO_FACE
        assert fit_head(mesh(count=68)).state is GuideState.NO_FACE
        fit = fit_head(None)
        assert isinstance(fit, GuideFit) and not fit.good and "oval" in fit.message

    def test_the_target_distance_centred_and_frontal_is_good(self):
        fit = fit_head(mesh(pose=HeadPose(pitch=-14.0, yaw=3.0)))
        assert fit.state is GuideState.GOOD and fit.good
        assert fit.message == "Good. Hold still."
        # His own session: 0.547 of the frame, centred 0.50 x 0.51 (2026-09-12)
        assert fit_head(mesh(height=0.547, centre=(0.5, 0.51))).good
        # The woman's: 0.435, centred 0.57 x 0.42
        assert fit_head(mesh(height=0.435, centre=(0.57, 0.42))).good

    def test_too_far_says_come_closer_and_too_near_move_back(self):
        guide = HeadGuide()
        far = fit_head(mesh(height=guide.face_height * 0.7))
        assert far.state is GuideState.CLOSER and far.message == "Come closer"
        near = fit_head(mesh(height=guide.face_height * 1.4))
        assert near.state is GuideState.BACK and near.message == "Move back"
        # the band's edges, inside
        assert fit_head(mesh(height=guide.face_height * 0.81)).good
        assert fit_head(mesh(height=guide.face_height * 1.24)).good

    def test_off_centre_says_put_the_head_in_the_oval(self):
        fit = fit_head(mesh(centre=(0.72, 0.46)))
        assert fit.state is GuideState.CENTRE and "oval" in fit.message
        assert fit_head(mesh(centre=(0.5, 0.66))).state is GuideState.CENTRE
        assert fit_head(mesh(centre=(0.6, 0.5))).good  # within the slack

    def test_turned_or_tilted_says_face_the_camera(self):
        turned = fit_head(mesh(pose=HeadPose(yaw=25.0)))
        assert turned.state is GuideState.FACE_CAMERA and turned.message == "Face the camera"
        assert fit_head(mesh(pose=HeadPose(pitch=-30.0))).state is GuideState.FACE_CAMERA
        assert fit_head(mesh(pose=HeadPose(pitch=-15.0, yaw=-12.0))).good

    def test_the_distance_is_said_before_the_centre_and_the_pose(self):
        fit = fit_head(mesh(height=0.3, centre=(0.8, 0.8), pose=HeadPose(yaw=40.0)))
        assert fit.state is GuideState.CLOSER
        fit = fit_head(mesh(centre=(0.8, 0.46), pose=HeadPose(yaw=40.0)))
        assert fit.state is GuideState.CENTRE

    def test_a_guide_of_its_own(self):
        guide = HeadGuide(face_height=0.3, centre=(0.4, 0.4), centre_slack=0.05)
        assert fit_head(mesh(height=0.3, centre=(0.4, 0.4)), guide).good
        assert fit_head(mesh(height=0.48, centre=(0.4, 0.4)), guide).state is GuideState.BACK
        assert fit_head(mesh(height=0.3, centre=(0.5, 0.4)), guide).state is GuideState.CENTRE
