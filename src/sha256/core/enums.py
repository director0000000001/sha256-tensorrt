"""Stable enumerations shared across the SHA-256 pipeline.

These names are part of the internal contract. Adding members is a
backwards-compatible change; renaming or removing them is not.
"""

from __future__ import annotations

from enum import StrEnum


class PixelFormat(StrEnum):
    """Pixel layout of a :class:`~sha256.core.types.VideoFrame` buffer.

    Kept deliberately small. Backends convert into one of these at their own
    boundary so the rest of the pipeline never inspects backend-native formats.
    """

    RGB24 = "rgb24"
    BGR24 = "bgr24"
    RGBA32 = "rgba32"
    BGRA32 = "bgra32"
    NV12 = "nv12"
    GRAY8 = "gray8"

    @property
    def channels(self) -> int:
        """Number of interleaved channels, or 0 when planar/undefined."""
        return _PIXEL_FORMAT_CHANNELS[self]

    @property
    def is_planar(self) -> bool:
        """True when the format is not a simple interleaved channel layout."""
        return self is PixelFormat.NV12


_PIXEL_FORMAT_CHANNELS: dict[PixelFormat, int] = {
    PixelFormat.RGB24: 3,
    PixelFormat.BGR24: 3,
    PixelFormat.RGBA32: 4,
    PixelFormat.BGRA32: 4,
    PixelFormat.NV12: 0,
    PixelFormat.GRAY8: 1,
}


class PipelineStage(StrEnum):
    """Named stages of the real-time loop.

    Used as the key space for performance metrics so timing data can be
    collected without any stage knowing about the monitor implementation.
    """

    CAPTURE = "capture"
    TRACKING = "tracking"
    ANIMATION = "animation"
    RENDERING = "rendering"
    OUTPUT = "output"
    END_TO_END = "end_to_end"


class PipelineState(StrEnum):
    """Lifecycle state of the processing pipeline as a whole."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


class TrackingQuality(StrEnum):
    """Coarse, human-readable summary of tracking health.

    Derived from confidence and continuity by the tracker adapter. The UI shows
    this; automated logic should prefer the numeric confidence value.
    """

    NONE = "none"
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"


class AvatarState(StrEnum):
    """Residency state of an avatar within the :class:`AvatarManager`."""

    DISCOVERED = "discovered"
    """Manifest read and validated; no weights touched."""

    LOADING = "loading"
    UNLOADING = "unloading"

    LOADED = "loaded"
    """Resident in memory/VRAM and ready to drive frames."""

    ACTIVE = "active"
    """Loaded and currently selected by the pipeline."""

    ERROR = "error"
    """Validation or load failed; see the recorded error for the reason."""


class DeviceKind(StrEnum):
    """Compute placement for an engine or renderer."""

    CPU = "cpu"
    CUDA = "cuda"
    DIRECTML = "directml"
    AUTO = "auto"
