"""Stable contracts: data types, component protocols, enums, errors.

This package is the dependency floor. It must never import a feature package,
a UI toolkit, or a third-party AI/vision library — only the standard library,
``numpy``, and ``pydantic``. If that rule is ever broken, the decoupling the
whole architecture rests on is gone.
"""

from __future__ import annotations

from sha256.core.enums import (
    AvatarState,
    DeviceKind,
    PipelineStage,
    PipelineState,
    PixelFormat,
    TrackingQuality,
)
from sha256.core.types import (
    BlendshapeSet,
    FaceTrackingResult,
    FrameMetadata,
    HeadPose,
    LandmarkSet,
    RenderedFrame,
    VideoFrame,
)

__all__ = [
    "AvatarState",
    "BlendshapeSet",
    "DeviceKind",
    "FaceTrackingResult",
    "FrameMetadata",
    "HeadPose",
    "LandmarkSet",
    "PipelineStage",
    "PipelineState",
    "PixelFormat",
    "RenderedFrame",
    "TrackingQuality",
    "VideoFrame",
]
