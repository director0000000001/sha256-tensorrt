"""Stable data structures that flow through the sha256 pipeline.

This module is the contract described in Phase 6 of the project brief:

    FaceTracker -> FaceTrackingResult -> AvatarEngine -> RenderedFrame

Nothing here imports MediaPipe, LivePortrait, PyTorch, OpenCV, or Qt. A backend
adapter converts *into* these types at its own boundary, which is what lets a
tracker or engine be swapped without touching the rest of the application.

Design notes
------------
*   These are ``dataclass`` types, not Pydantic models. They are constructed on
    every frame, so validation cost matters; Pydantic is reserved for the I/O
    boundary (config files, avatar manifests) where validation is worth paying
    for. See ``docs/decisions/ADR-0003-data-model-strategy.md``.
*   Frame payloads are ``numpy`` arrays. The array is *borrowed*, not copied —
    see :class:`VideoFrame` for the ownership rule.
*   All timestamps are monotonic seconds from :func:`sha256.utils.timing.now`.
    Never wall-clock: wall-clock can jump backwards and corrupt latency maths.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Self

import numpy as np

from sha256.core.enums import PixelFormat, TrackingQuality

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "BlendshapeSet",
    "FaceTrackingResult",
    "FrameMetadata",
    "HeadPose",
    "LandmarkSet",
    "RenderedFrame",
    "VideoFrame",
]


# ---------------------------------------------------------------------------
# Frame metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    """Identity and timing information that travels with a frame.

    The same ``frame_id`` is carried from capture through to virtual-camera
    output, which is what makes true end-to-end latency measurable rather than
    estimated.
    """

    frame_id: int
    """Monotonically increasing counter assigned by the capture source."""

    capture_timestamp: float
    """Monotonic seconds at which the frame left the camera."""

    source_id: str = "unknown"
    """Identifier of the producing device or backend."""

    sequence_gap: int = 0
    """Frames the source reports as skipped immediately before this one.

    Non-zero means the *source* dropped frames (USB bandwidth, driver stall),
    which is a different failure from the pipeline dropping them under load.
    """

    def age(self, now: float) -> float:
        """Seconds elapsed since capture, given a monotonic ``now``."""
        return now - self.capture_timestamp


# ---------------------------------------------------------------------------
# Video frames
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """A single image buffer plus its metadata.

    Ownership
    ---------
    ``data`` is **borrowed**. A capture backend may reuse the underlying buffer
    for the next frame, so any consumer that needs the pixels beyond the current
    pipeline tick must call :meth:`copy` first. This convention avoids a
    per-frame allocation on the hot path while keeping the hazard explicit.
    """

    data: NDArray[np.uint8]
    pixel_format: PixelFormat
    metadata: FrameMetadata

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def resolution(self) -> tuple[int, int]:
        """``(width, height)`` in pixels."""
        return self.width, self.height

    def copy(self) -> Self:
        """Return a frame that owns its pixel buffer."""
        return replace(self, data=self.data.copy())


@dataclass(frozen=True, slots=True)
class RenderedFrame:
    """Output of the renderer, ready for the virtual-camera sink.

    Separate from :class:`VideoFrame` on purpose: a rendered frame carries the
    provenance needed for diagnostics (which avatar produced it, how long the
    pipeline took) that an input frame has no notion of.
    """

    data: NDArray[np.uint8]
    pixel_format: PixelFormat
    metadata: FrameMetadata

    avatar_id: str | None = None
    """Avatar that produced this frame; ``None`` for placeholder/idle output."""

    render_timestamp: float = 0.0
    """Monotonic seconds at which rendering completed."""

    is_placeholder: bool = False
    """True when this is a fallback frame (no avatar loaded, tracking lost)."""

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def pipeline_latency(self) -> float:
        """Seconds from camera capture to completed render."""
        return self.render_timestamp - self.metadata.capture_timestamp


# ---------------------------------------------------------------------------
# Tracking output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LandmarkSet:
    """Facial landmarks in a backend-neutral form.

    Coordinates are **normalised**: ``x`` and ``y`` in ``[0, 1]`` relative to the
    source frame, ``z`` in source-relative depth units where negative is toward
    the camera. Normalising at the adapter boundary means downstream code does
    not need to know the capture resolution, and a 68-point tracker and a
    478-point tracker differ only in ``count``.
    """

    points: NDArray[np.float32]
    """Array of shape ``(N, 3)`` holding normalised ``(x, y, z)`` per point."""

    scheme: str
    """Landmark topology identifier, e.g. ``"mediapipe_478"`` or ``"osf_66"``.

    Consumers that need specific indices must check this rather than assuming a
    layout. Two trackers with different schemes are not index-compatible.
    """

    visibility: NDArray[np.float32] | None = None
    """Optional per-point visibility/occlusion score in ``[0, 1]``."""

    @property
    def count(self) -> int:
        return int(self.points.shape[0])

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            msg = f"landmark points must have shape (N, 3), got {self.points.shape}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class HeadPose:
    """Rigid head pose relative to the camera.

    Rotation is expressed in degrees using an intrinsic Tait-Bryan convention
    with the camera looking down ``+z``:

    * ``pitch`` — nodding, positive looks up
    * ``yaw``   — shaking, positive turns to the subject's left
    * ``roll``  — tilting, positive tilts to the subject's right

    Trackers that emit quaternions or rotation matrices convert in their adapter.
    A single documented convention here prevents the axis-flip bugs that
    otherwise appear when swapping trackers.
    """

    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0

    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Head centre offset ``(x, y, z)``; ``x``/``y`` normalised, ``z`` relative."""

    confidence: float = 0.0
    """Tracker confidence in this pose estimate, in ``[0, 1]``."""

    @property
    def rotation(self) -> tuple[float, float, float]:
        return self.pitch, self.yaw, self.roll


@dataclass(frozen=True, slots=True)
class BlendshapeSet:
    """Named expression coefficients, normally in ``[0, 1]``.

    Stored as parallel ``names``/``weights`` sequences rather than a dict: the
    engine reads them as a vector every frame, and the name list is constant for
    a given tracker so it can be shared across frames without copying.

    ``scheme`` identifies the naming convention (e.g. ``"arkit_52"``). An engine
    that needs a different convention performs an explicit, documented remap —
    it never assumes ordering.
    """

    names: tuple[str, ...]
    weights: NDArray[np.float32]
    scheme: str = "arkit_52"

    def __post_init__(self) -> None:
        if len(self.names) != int(self.weights.shape[0]):
            msg = (
                f"blendshape names ({len(self.names)}) and weights "
                f"({self.weights.shape[0]}) must be the same length"
            )
            raise ValueError(msg)

    def as_mapping(self) -> dict[str, float]:
        """Return a name -> weight dict. Convenience for UI/debug, not hot-path."""
        return dict(zip(self.names, (float(w) for w in self.weights), strict=True))

    def get(self, name: str, default: float = 0.0) -> float:
        """Look up a single coefficient by name."""
        try:
            return float(self.weights[self.names.index(name)])
        except ValueError:
            return default

    @classmethod
    def empty(cls, scheme: str = "arkit_52") -> Self:
        """A set with no coefficients — used by neutral/placeholder trackers."""
        return cls(names=(), weights=np.zeros(0, dtype=np.float32), scheme=scheme)


@dataclass(frozen=True, slots=True)
class FaceTrackingResult:
    """Everything a tracker reports for one frame.

    This is the single hand-off point between tracking and animation. An engine
    consumes *only* this type, which is what allows MediaPipe, OpenSeeFace, or a
    future tracker to be interchanged.

    A result with ``face_detected=False`` is still a valid result and is
    delivered normally — losing the face is an expected condition the engine must
    handle (hold last pose, ease to neutral), not an error to raise.
    """

    metadata: FrameMetadata
    face_detected: bool

    landmarks: LandmarkSet | None = None
    head_pose: HeadPose | None = None
    blendshapes: BlendshapeSet | None = None

    confidence: float = 0.0
    """Overall tracker confidence in ``[0, 1]``."""

    tracker_id: str = "unknown"
    """Backend that produced this result, for diagnostics and metric labelling."""

    processing_time: float = 0.0
    """Seconds spent inside the tracker for this frame."""

    extras: dict[str, float] = field(default_factory=dict)
    """Backend-specific scalars (gaze angles, per-eye openness, ...).

    Deliberately untyped and optional: it is an escape hatch for diagnostics and
    experiments. Core pipeline logic must never *require* a key here — anything
    that becomes load-bearing gets promoted to a real field.
    """

    @property
    def quality(self) -> TrackingQuality:
        """Coarse bucket of :attr:`confidence` for display purposes."""
        if not self.face_detected:
            return TrackingQuality.NONE
        if self.confidence >= 0.75:
            return TrackingQuality.GOOD
        if self.confidence >= 0.45:
            return TrackingQuality.FAIR
        return TrackingQuality.POOR

    @classmethod
    def no_face(cls, metadata: FrameMetadata, tracker_id: str = "unknown") -> Self:
        """Construct the canonical 'nothing detected' result."""
        return cls(metadata=metadata, face_detected=False, tracker_id=tracker_id)
