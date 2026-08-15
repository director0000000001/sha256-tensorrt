"""Tests for the clip-to-sources extractor.

It had no tests, and it failed in the way untested selection code always fails:
silently and completely. A clip of a man turning his head end to end reported
**0 of 22 roles covered** and wrote nothing, twice, while every frame it needed
sat in the file. Both causes are here, because both are invisible from the
outside — the tool reports what survived, and "nothing survived" reads like a
bad clip rather than a bad filter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import extract_avatar_frames as extractor  # noqa: E402

from sha256.core.types import (  # noqa: E402
    BlendshapeSet,
    FaceTrackingResult,
    FrameMetadata,
    HeadPose,
    LandmarkSet,
)

LANDMARK_COUNT = 478

# Enough of MediaPipe's set that the selector has something to read. The values
# are a plausible relaxed face: not zero, which is the entire reason a resting
# baseline has to exist at all.
RESTING = {
    "eyeBlinkLeft": 0.18,
    "eyeBlinkRight": 0.17,
    "eyeSquintLeft": 0.22,
    "eyeSquintRight": 0.21,
    "browInnerUp": 0.14,
    "browDownLeft": 0.09,
    "browDownRight": 0.08,
    "jawOpen": 0.05,
    "mouthSmileLeft": 0.06,
    "mouthSmileRight": 0.06,
    "mouthPressLeft": 0.07,
    "mouthPressRight": 0.07,
}


def tracking(*, yaw: float = 0.0, pitch: float = 0.0, **shapes: float) -> FaceTrackingResult:
    readings = dict(RESTING)
    readings.update(shapes)
    names = tuple(readings)

    points = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)
    points[:, 0] = 0.5
    points[:, 1] = 0.5
    # A face tall enough that the lip gap reads as closed rather than as a
    # division by nothing.
    points[10, 1] = 0.2
    points[152, 1] = 0.8
    points[13, 1] = 0.50
    points[14, 1] = 0.51

    return FaceTrackingResult(
        metadata=FrameMetadata(frame_id=0, capture_timestamp=0.0),
        face_detected=True,
        landmarks=LandmarkSet(points=points, scheme="mediapipe_478"),
        head_pose=HeadPose(yaw=yaw, pitch=pitch, roll=0.0),
        blendshapes=BlendshapeSet(
            names=names,
            weights=np.array([readings[n] for n in names], dtype=np.float32),
            scheme="mediapipe_52",
        ),
        confidence=1.0,
    )


class TestRestingFace:
    """Without a shared baseline the selector answers "nothing is happening".

    ``_source_weights`` measures against rest, and given no rest it seeds one
    from the frame in front of it — so every frame becomes its own baseline and
    every role driven by a blendshape or by head pose scores exactly zero. Only
    the two driven by the lip gap survive, which is why the speaking clip
    reported 2 of 22 and the head-turn clip reported 0.
    """

    def test_a_turned_frame_scores_a_turn(self):
        base = tracking(yaw=1.0)
        turned = tracking(yaw=-35.0)

        weights = extractor._source_weights(turned, neutral=extractor.resting_face(base).copy())

        assert weights.get("source_turn_right", 0.0) > extractor.MIN_ROLE_SCORE

    def test_only_the_lip_gap_survives_a_frame_judged_against_itself(self):
        """The bug, stated as a test, and the shape of what it left behind.

        Against no baseline, a 35-degree turn scores no turn at all. What does
        survive is exactly the family that never consults the resting face: the
        teeth-bearing roles, which are driven by the lip gap measured off the
        landmarks. Two of twenty-two — which is what the speaking clip reported,
        and it was written down as a property of speaking clips.
        """
        turned = tracking(yaw=-35.0)

        weights = extractor._source_weights(turned)

        scored = {role for role, weight in weights.items() if weight > 0.01}
        assert scored <= {"source_mouth_open", "source_smile_closed"}
        assert "source_turn_right" not in scored

    def test_the_baseline_does_not_learn_from_the_frames_it_judges(self):
        """A held turn must not become "how this person sits".

        The resting face adapts, which is right live and wrong here: a clip that
        is mostly a turn would teach it that a turned head is neutral, and the
        frames the clip exists for would score lowest of all.
        """
        neutral = extractor.resting_face(tracking(yaw=1.0))
        turned = tracking(yaw=-35.0)

        first = extractor._source_weights(turned, neutral=neutral.copy())
        for _ in range(200):
            extractor._source_weights(turned, neutral=neutral.copy())
        last = extractor._source_weights(turned, neutral=neutral.copy())

        assert first == last


class TestSharpnessFloor:
    """A turn clip judged against its own still frames throws itself away.

    Laplacian variance over the face falls about eighteenfold across a turn on
    a clip where nothing goes out of focus: a turned face presents a smooth
    cheek where a frontal one presents two eyes, two brows and a hairline, and
    the codec starves a moving shot. So the floor has to come from frames at a
    similar angle.
    """

    @staticmethod
    def candidate(index: int, yaw: float, sharpness: float) -> extractor.Candidate:
        return extractor.Candidate(
            index=index,
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            tracking=tracking(yaw=yaw),
            face_width=900.0,
            sharpness=sharpness,
            identity_drift=0.0,
        )

    def test_a_turned_frame_is_judged_against_turned_frames(self):
        still = [self.candidate(i, 0.0, 200.0) for i in range(5)]
        turned = [self.candidate(5 + i, -35.0 - i, 20.0) for i in range(5)]

        floor = extractor.sharpness_floor(still + turned, turned[0])

        assert floor == pytest.approx(20.0)

    def test_a_still_frame_is_still_judged_against_still_frames(self):
        still = [self.candidate(i, 0.0, 200.0) for i in range(5)]
        turned = [self.candidate(5 + i, -35.0 - i, 20.0) for i in range(5)]

        floor = extractor.sharpness_floor(still + turned, still[0])

        assert floor == pytest.approx(200.0)

    def test_the_whole_of_a_turn_survives(self):
        """The end-to-end shape of the failure: 135 of 192 frames discarded."""
        still = [self.candidate(i, 0.0, 200.0) for i in range(5)]
        turned = [self.candidate(5 + i, -20.0 - i * 3, 20.0) for i in range(10)]
        rejected: dict[str, int] = {"no face": 0}

        kept = extractor.usable(still + turned, min_face=380.0, rejected=rejected)

        assert len(kept) == 15
        assert rejected.get("motion blurred", 0) == 0

    def test_a_far_turn_is_not_mistaken_for_a_different_face(self):
        """Rotation moves the identity ratios on its own.

        They are ratios of distances across the face, and a turned face is a
        projection: horizontal spans foreshorten, and not together, because the
        nose stands out of the plane the cheeks lie in. Against a flat tolerance
        that discarded 41 frames of one clip, and the frames it discarded were
        the far turns the clip was generated for.
        """
        turned = extractor.Candidate(
            index=0,
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            tracking=tracking(yaw=-45.0),
            face_width=900.0,
            sharpness=100.0,
            identity_drift=0.169,
        )
        rejected: dict[str, int] = {"no face": 0}

        kept = extractor.usable([turned], min_face=380.0, rejected=rejected)

        assert kept == [turned]
        assert rejected.get("identity drifted", 0) == 0

    def test_a_frontal_frame_of_someone_else_still_goes(self):
        """The allowance grows with the angle; at zero it is unchanged."""
        stranger = extractor.Candidate(
            index=0,
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            tracking=tracking(yaw=0.0),
            face_width=900.0,
            sharpness=100.0,
            identity_drift=0.169,
        )
        rejected: dict[str, int] = {"no face": 0}

        assert extractor.usable([stranger], min_face=380.0, rejected=rejected) == []
        assert rejected["identity drifted"] == 1

    def test_a_genuinely_soft_frame_at_the_same_angle_still_goes(self):
        sharp = [self.candidate(i, -30.0, 200.0) for i in range(5)]
        soft = self.candidate(9, -30.0, 20.0)
        rejected: dict[str, int] = {"no face": 0}

        kept = extractor.usable([*sharp, soft], min_face=380.0, rejected=rejected)

        assert soft not in kept
        assert rejected["motion blurred"] == 1


class TestTurnLadder:
    """One frame per band of angle, and the sharpest one in each."""

    @staticmethod
    def candidate(index: int, yaw: float, sharpness: float) -> extractor.Candidate:
        return extractor.Candidate(
            index=index,
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            tracking=tracking(yaw=yaw),
            face_width=900.0,
            sharpness=sharpness,
            identity_drift=0.0,
        )

    def test_one_rung_per_band(self):
        candidates = [self.candidate(i, -30.0 + i, 100.0) for i in range(31)]

        rungs = extractor.turn_ladder(candidates, step=3.0)

        assert len(rungs) == 11

    def test_the_sharpest_frame_in_the_band_wins(self):
        candidates = [
            self.candidate(0, 10.0, 50.0),
            self.candidate(1, 10.2, 200.0),
            self.candidate(2, 10.4, 90.0),
        ]

        rungs = extractor.turn_ladder(candidates, step=3.0)

        assert [c.index for c in rungs] == [1]

    def test_rungs_come_out_in_order_of_angle(self):
        """The engine bisects them, so out of order is silently wrong."""
        candidates = [self.candidate(i, yaw, 100.0) for i, yaw in enumerate([20.0, -30.0, 0.0])]

        rungs = extractor.turn_ladder(candidates, step=3.0)

        angles = [extractor.yaw_of(c) for c in rungs]
        assert angles == sorted(angles)
