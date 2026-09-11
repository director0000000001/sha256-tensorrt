"""Application settings schema.

Every value is validated at startup. Configuration errors surface as one clear
message before any hardware is touched, rather than as an exception three
layers into the capture loop.

Backends are named by **registry id** (a string), never by import path. That is
deliberate: it keeps the configuration file free of Python internals, lets an
unavailable backend be reported as a friendly "not installed" message, and
means swapping tracker or engine is a one-line config change.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "AvatarSettings",
    "CameraSettings",
    "EngineSettings",
    "GpuSettings",
    "OutputSettings",
    "PerformanceSettings",
    "RenderingSettings",
    "Settings",
    "TrackingSettings",
    "UiSettings",
    "VirtualCameraSettings",
]


class _Section(BaseModel):
    """Base for settings sections: strict, immutable after load."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CameraSettings(_Section):
    backend: str = Field(default="null", description="Registry id of the camera backend.")
    device_index: int = Field(default=0, ge=0)
    device_id: str | None = Field(
        default=None, description="Explicit device id; takes precedence over the index."
    )
    width: int = Field(default=1280, gt=0, le=7680)
    height: int = Field(default=720, gt=0, le=4320)
    target_fps: float = Field(default=30.0, gt=0, le=240)
    mirror: bool = Field(default=True, description="Mirror the preview, as users expect.")
    use_mjpeg: bool = Field(
        default=True,
        description=(
            "Request MJPEG from the camera. Most UVC devices only reach 30 fps at "
            "higher resolutions with MJPEG; raw formats cap out far lower. Turn off "
            "only if a specific camera misbehaves with it."
        ),
    )


class TrackingSettings(_Section):
    backend: str = Field(default="null")
    min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_presence_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Threshold for a face still being present in a tracked frame, as "
            "opposed to the initial detection. MediaPipe separates the two."
        ),
    )
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    smoothing: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description=(
            "Temporal smoothing factor. Higher is steadier but adds latency; "
            "the trade-off must be measured, not guessed."
        ),
    )
    mouth_smoothing: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description=(
            "The same factor for the lips alone. They carry the fastest movement "
            "on a face and none of the shimmer the filter exists for, so they are "
            "filtered separately — see MOUTH_LANDMARKS in the MediaPipe tracker "
            "for the measurement. Set it equal to `smoothing` for one filter over "
            "the whole face."
        ),
    )
    num_faces: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Faces to track. One is the product case; more only costs time.",
    )
    delegate: Literal["auto", "cpu", "gpu"] = Field(
        default="auto",
        description=(
            "Inference device for the tracker. 'auto' resolves to CPU on "
            "Windows: the published MediaPipe wheel is built without a GPU "
            "delegate (measured — see docs/benchmarks/B003-face-tracking.md)."
        ),
    )
    model_path: str | None = Field(
        default=None,
        description=(
            "Model bundle location, relative to the models root unless "
            "absolute. Defaults to downloads/face_landmarker.task, which "
            "scripts/fetch_face_landmarker.py writes."
        ),
    )


class EngineSettings(_Section):
    backend: str = Field(default="null")
    device: Literal["auto", "cuda", "cpu"] = "auto"
    precision: Literal["fp32", "fp16", "auto"] = "auto"


class RenderingSettings(_Section):
    backend: str = Field(default="null")
    output_width: int = Field(default=1280, gt=0, le=3840)
    output_height: int = Field(default=720, gt=0, le=2160)
    target_fps: float = Field(default=30.0, gt=0, le=240)


class VirtualCameraSettings(_Section):
    backend: str = Field(default="null")
    device_name: str = Field(
        default="sha256 Virtual Camera",
        description=(
            "Name for a device sha256 creates itself. Backends that write to "
            "someone else's device — pyvirtualcam does — cannot honour it and "
            "report the real name instead."
        ),
    )
    device: str | None = Field(
        default=None,
        description=(
            "Exact OS device to bind to, e.g. 'OBS Virtual Camera'. None takes "
            "the first that works, which is right unless a machine has several."
        ),
    )
    auto_start: bool = Field(
        default=False,
        description="Whether starting the pipeline also publishes to the virtual camera.",
    )


class OutputSettings(_Section):
    """What leaves the pipeline for the virtual camera, after the avatar is
    rendered -- see ``sha256.pipeline.conditioning``."""

    quality: Literal["native", "1080p", "720p", "480p", "360p"] = Field(
        default="native",
        description=(
            "The published picture is scaled down to this height and back up to "
            "the camera's own size, so the stream's format never changes and the "
            "picture looks like a cheaper webcam's. 'native' leaves it alone; a "
            "height at or above the camera's own is native too."
        ),
    )
    failure_text: str = Field(
        default="Camera battery died",
        max_length=64,
        description="The line on the black screen that replaces the avatar.",
    )
    auto_blackout: bool = Field(
        default=True,
        description=(
            "Black the screen on the pipeline's own failures: the camera's own "
            "picture about to go out, or the face lost for lost_face_seconds. "
            "Off keeps the black screen for the button and the hotkey alone."
        ),
    )
    lost_face_seconds: float = Field(default=1.5, ge=0.0, le=30.0)
    min_blackout_seconds: float = Field(
        default=3.0,
        ge=0.0,
        le=60.0,
        description="An automatic black screen stays at least this long.",
    )
    recovery_seconds: float = Field(
        default=1.0,
        ge=0.0,
        le=30.0,
        description="...and lifts only after the face has been tracked this long.",
    )
    blackout_hotkey: str = Field(
        default="ctrl+alt+b",
        description=(
            "Toggles the black screen from any application (Windows). Modifiers "
            "ctrl / alt / shift / win plus a letter, a digit or F1-F24."
        ),
    )


class GpuSettings(_Section):
    vram_budget_mb: int = Field(
        default=6144,
        ge=0,
        description=(
            "Soft VRAM ceiling for resident avatars. A budget rather than a hard "
            "limit: it drives eviction decisions, it does not cap allocation."
        ),
    )
    allow_fallback_to_cpu: bool = Field(default=True)


class AvatarSettings(_Section):
    library_path: str = Field(
        default="avatars", description="Avatar library root; relative to the app root."
    )
    default_avatar_id: str | None = Field(default=None)
    max_resident: int = Field(
        default=2,
        ge=1,
        le=16,
        description="How many avatars may stay loaded at once (active plus warm spares).",
    )
    verify_checksums: bool = Field(
        default=False,
        description=(
            "Hash every declared asset during discovery. Correct but slow on a "
            "large library, so it is opt-in and always on for imports."
        ),
    )


class PerformanceSettings(_Section):
    enabled: bool = Field(default=True)
    window_size: int = Field(
        default=240,
        ge=10,
        le=10_000,
        description="Samples retained per metric — 240 is about 8s at 30fps.",
    )
    sample_system_metrics: bool = Field(
        default=False,
        description="Poll CPU/GPU/VRAM. Off by default: the probe itself costs time.",
    )
    system_sample_interval: float = Field(default=1.0, gt=0.0, le=60.0)


class UiSettings(_Section):
    background_image: str | None = Field(
        default=None,
        description=(
            "Background image for the window, relative to the app root unless "
            "absolute. A missing file falls back to the plain background rather "
            "than failing to start — an unreadable decoration must never stop "
            "the application."
        ),
    )
    background_dim: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description=(
            "How much to darken the background image, 0 = untouched, 1 = black. "
            "Controls sit on top of it and have to stay readable over a photo "
            "whose brightness nobody can predict."
        ),
    )
    background_color: str = Field(
        default="#101014",
        pattern=r"^#[0-9a-fA-F]{6}$",
        description="Window background when no image is configured.",
    )


class ModelSettings(_Section):
    root_path: str = Field(
        default="models", description="Model artifact root, relative to the app root."
    )
    allow_downloads: bool = Field(
        default=False,
        description=(
            "Whether the app may fetch model files over the network. Off by "
            "default; sha256 makes no network calls unless explicitly enabled. "
            "See docs/PRIVACY.md."
        ),
    )


class Settings(BaseSettings):
    """Root settings object.

    Environment variables use the ``SHA256_`` prefix with ``__`` as the nesting
    delimiter, e.g. ``SHA256_CAMERA__TARGET_FPS=60``.
    """

    model_config = SettingsConfigDict(
        env_prefix="SHA256_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        frozen=True,
    )

    environment: Literal["development", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = False

    developer_mode: bool = Field(
        default=False,
        description=(
            "Unlocks owner-only operations: avatar import, library mutation, raw "
            "frame inspection. Gated so a normal run has no code path that can "
            "modify the avatar library."
        ),
    )

    camera: CameraSettings = Field(default_factory=CameraSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)
    rendering: RenderingSettings = Field(default_factory=RenderingSettings)
    virtual_camera: VirtualCameraSettings = Field(default_factory=VirtualCameraSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    gpu: GpuSettings = Field(default_factory=GpuSettings)
    avatars: AvatarSettings = Field(default_factory=AvatarSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    ui: UiSettings = Field(default_factory=UiSettings)

    _warnings: tuple[str, ...] = PrivateAttr(default=())

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        found: list[str] = []

        if self.rendering.target_fps > self.camera.target_fps:
            # Not fatal — a renderer may legitimately interpolate — but it is
            # almost always a misconfiguration, so make it visible.
            found.append(
                f"rendering.target_fps ({self.rendering.target_fps}) exceeds "
                f"camera.target_fps ({self.camera.target_fps}); output cannot "
                f"contain more unique frames than the camera supplies."
            )

        if self.avatars.max_resident > 1 and self.gpu.vram_budget_mb == 0:
            found.append(
                "avatars.max_resident > 1 with gpu.vram_budget_mb = 0; the "
                "residency cache cannot make eviction decisions without a budget."
            )

        # PrivateAttr assignment is permitted on a frozen model.
        self._warnings = tuple(found)
        return self

    @property
    def warnings(self) -> tuple[str, ...]:
        """Non-fatal configuration concerns raised during validation."""
        return self._warnings
