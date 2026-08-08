"""Placeholder face tracker.

Emits a synthetic, slowly-oscillating face so the downstream contract can be
exercised before any real tracker is integrated. It reports
``face_detected=True`` with a deliberately modest confidence, and it never
touches the frame pixels.

This is *not* face tracking. It exists so that:

*   the ``FaceTracker`` protocol has a reference implementation to test against;
*   engine and renderer work can proceed against realistic-shaped data;
*   the animation path can be validated independently of tracker accuracy.

MediaPipe Face Landmarker and OpenSeeFace are the candidates under evaluation
(docs/OPEN_SOURCE_EVALUATION.md). Neither is integrated yet.
"""

from __future__ import annotations

import math

import numpy as np

from sha256.core.types import (
    BlendshapeSet,
    FaceTrackingResult,
    HeadPose,
    LandmarkSet,
    VideoFrame,
)
from sha256.utils.timing import now

__all__ = ["NullFaceTracker"]

# A small ARKit-named subset. Full 52-name schemes come from the real trackers;
# using a subset here keeps the placeholder honest about what it is.
_PLACEHOLDER_BLENDSHAPES: tuple[str, ...] = (
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "jawOpen",
    "mouthSmileLeft",
    "mouthSmileRight",
    "browInnerUp",
)

_LANDMARK_COUNT = 68


class NullFaceTracker:
    """Synthetic tracker producing a gently animated neutral face."""

    def __init__(self, *, animate: bool = True, confidence: float = 0.5) -> None:
        self._animate = animate
        self._confidence = confidence
        self._running = False
        self._origin = now()

        # A ring of points standing in for a face outline. Fixed, then modulated
        # per frame, so the shape is stable but not frozen.
        angles = np.linspace(0.0, 2.0 * math.pi, _LANDMARK_COUNT, endpoint=False)
        self._base_points = np.stack(
            [
                0.5 + 0.18 * np.cos(angles),
                0.5 + 0.24 * np.sin(angles),
                np.zeros(_LANDMARK_COUNT),
            ],
            axis=1,
        ).astype(np.float32)

    @property
    def backend_id(self) -> str:
        return "null"

    @property
    def landmark_scheme(self) -> str:
        return "null_68"

    @property
    def provides_blendshapes(self) -> bool:
        return True

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True
        self._origin = now()

    def stop(self) -> None:
        self._running = False

    def reset(self) -> None:
        self._origin = now()

    def process(self, frame: VideoFrame) -> FaceTrackingResult:
        started = now()
        phase = (started - self._origin) if self._animate else 0.0

        yaw = 12.0 * math.sin(phase * 0.6)
        pitch = 6.0 * math.sin(phase * 0.4 + 1.0)
        roll = 3.0 * math.sin(phase * 0.3)

        # A blink roughly every four seconds, so eye-driven code paths get
        # exercised rather than sitting at a constant value.
        blink = max(0.0, math.sin(phase * math.pi / 2.0) ** 32)
        jaw = 0.5 * (1.0 + math.sin(phase * 1.1)) * 0.3

        weights = np.array(
            [blink, blink, jaw, 0.2, 0.2, 0.1 * (1.0 + math.sin(phase * 0.7))],
            dtype=np.float32,
        )

        points = self._base_points.copy()
        points[:, 0] += np.float32(0.01 * math.sin(phase * 0.6))

        return FaceTrackingResult(
            metadata=frame.metadata,
            face_detected=True,
            landmarks=LandmarkSet(points=points, scheme=self.landmark_scheme),
            head_pose=HeadPose(pitch=pitch, yaw=yaw, roll=roll, confidence=self._confidence),
            blendshapes=BlendshapeSet(
                names=_PLACEHOLDER_BLENDSHAPES, weights=weights, scheme="arkit_subset"
            ),
            confidence=self._confidence,
            tracker_id=self.backend_id,
            processing_time=now() - started,
        )
