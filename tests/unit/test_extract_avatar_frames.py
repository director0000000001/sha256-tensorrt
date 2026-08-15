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
    """Blur is measured in time, because blur is a thing that happens in time.

    Laplacian variance over a face measures how much detail the face is
    *showing*, not how sharp it is. It falls eighteenfold across a turn where
    nothing goes out of focus, and a neutral face scores a fraction of an
    open-mouthed one in the same clip — teeth, and a dark cavity. Judged against
    the sharpest frame in the whole clip, each of those becomes a reason to
    discard the frames the clip was generated for: 135 of 192 on the turn clip,
    163 of 192 on the eye clip.

    The frames next to a frame show almost the same face doing almost the same
    thing, so they are the only fair thing to compare it with.
    """

    @staticmethod
    def candidate(index: int, sharpness: float, yaw: float = 0.0) -> extractor.Candidate:
        return extractor.Candidate(
            index=index,
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            tracking=tracking(yaw=yaw),
            face_width=900.0,
            sharpness=sharpness,
            identity_drift=0.0,
        )

    def test_a_frame_is_judged_against_its_neighbours_in_time(self):
        frames = [self.candidate(i, 200.0) for i in range(5)]
        frames += [self.candidate(5 + i, 20.0) for i in range(40)]

        # Far from the sharp opening, the floor comes from the soft plateau.
        assert extractor.sharpness_floor(frames, frames[30]) == pytest.approx(20.0)
        # Right beside it, the sharp frames are still the reference.
        assert extractor.sharpness_floor(frames, frames[6]) == pytest.approx(200.0)

    def test_a_held_expression_survives_whatever_it_scores(self):
        """Two seconds of neutral beside two of an open mouth.

        The open mouth scores ten times what the neutral face does and neither
        is out of focus. Both are what the clip is for.
        """
        frames = [self.candidate(i, 20.0) for i in range(48)]
        frames += [self.candidate(48 + i, 200.0) for i in range(48)]
        rejected: dict[str, int] = {"no face": 0}

        kept = extractor.usable(frames, min_face=380.0, rejected=rejected)

        # Only the neutral frames within one window of the step are judged
        # against it; the rest of the plateau is compared with itself.
        assert len(kept) >= 88
        assert kept[0].index == 0

    def test_one_blurred_frame_inside_a_plateau_goes(self):
        held = [self.candidate(i, 200.0) for i in range(20)]
        held[10] = self.candidate(10, 20.0)
        rejected: dict[str, int] = {"no face": 0}

        kept = extractor.usable(held, min_face=380.0, rejected=rejected)

        assert [c.index for c in kept] == [i for i in range(20) if i != 10]
        assert rejected["motion blurred"] == 1

    def test_the_whole_of_a_slow_turn_survives(self):
        """The turn clip: detail ramps down with the angle, nothing is blurred."""
        ramp = [self.candidate(i, 200.0 - i * 4.0, yaw=-float(i)) for i in range(40)]
        rejected: dict[str, int] = {"no face": 0}

        kept = extractor.usable(ramp, min_face=380.0, rejected=rejected)

        assert len(kept) == 40
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

    def test_a_genuinely_soft_frame_beside_sharp_ones_still_goes(self):
        """The filter still has to do its job, or it is not a filter."""
        sharp = [self.candidate(i, 200.0) for i in range(5)]
        soft = self.candidate(5, 20.0)
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


class TestEyeSharpness:
    """The blur measure that can judge a turn, because the whole-face one cannot.

    A Laplacian counts the detail a face is *showing*, and a turned face shows
    far less of it — one eye goes behind the nose and a smooth cheek fills the
    frame. Measured on frames sharp enough to count beard hairs, the median runs
    179 at 0-8 degrees of yaw against 9 at 20-35, so a blurred turn and a sharp
    one score alike. That is how a ladder of thirty motion-blurred frames was
    built and shipped. The eye nearer the camera keeps its lashes and its brow
    hairs to well past 40 degrees, so it can be compared across a whole turn.
    """

    @staticmethod
    def candidate(image: np.ndarray) -> extractor.Candidate:
        """A frame with one eye over the top-left quarter and one over the rest."""
        points = np.full((LANDMARK_COUNT, 3), 0.5, dtype=np.float32)
        blocks = (
            (list(extractor._LEFT_EYE) + list(extractor._LEFT_BROW), 0.05),
            (list(extractor._RIGHT_EYE) + list(extractor._RIGHT_BROW), 0.55),
        )
        for group, origin in blocks:
            for offset, index in enumerate(group):
                points[index, 0] = origin + 0.30 * (offset % 2)
                points[index, 1] = origin + 0.30 * ((offset // 2) % 2)

        base = tracking()
        return extractor.Candidate(
            index=0,
            image=image,
            tracking=FaceTrackingResult(
                metadata=base.metadata,
                face_detected=True,
                landmarks=LandmarkSet(points=points, scheme="mediapipe_478"),
                head_pose=base.head_pose,
                blendshapes=base.blendshapes,
                confidence=1.0,
            ),
            face_width=900.0,
            sharpness=0.0,
            identity_drift=0.0,
        )

    @staticmethod
    def noise(size: int = 240) -> np.ndarray:
        return np.random.default_rng(7).integers(0, 255, (size, size, 3), dtype=np.uint8)

    def test_a_blurred_frame_scores_far_below_a_sharp_one(self):
        cv2 = pytest.importorskip("cv2")
        sharp = self.noise()

        blurred = extractor.eye_sharpness(self.candidate(cv2.GaussianBlur(sharp, (9, 9), 0)))

        assert blurred < 0.25 * extractor.eye_sharpness(self.candidate(sharp))

    def test_it_reads_the_eye_that_still_has_detail(self):
        """Whichever way the head is pointing, one eye is the near one."""
        pytest.importorskip("cv2")
        image = self.noise()
        # The far half of the face, flattened: at a turn it carries nothing.
        image[image.shape[0] // 2 :, image.shape[1] // 2 :] = 128

        assert extractor.eye_sharpness(self.candidate(image)) > 1000.0
