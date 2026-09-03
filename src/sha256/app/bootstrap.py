"""Composition root.

The single place where concrete implementations are chosen and wired. Every
other module depends on protocols only; this one does the ``import`` and the
``registry.create(...)``.

Keeping it in one file has a practical payoff: to find out what sha256 is
actually running, you read this file, and nothing else can quietly introduce a
dependency on a specific tracker or engine behind your back.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sha256.app.registry import Registries
from sha256.avatar.cache import ResidencyCache
from sha256.avatar.engine_null import NullAvatarEngine
from sha256.avatar.engine_warp import WarpAvatarEngine
from sha256.avatar.manager import AvatarManager
from sha256.avatar.repository import FilesystemAvatarRepository
from sha256.camera.null import NullCameraProvider
from sha256.camera.opencv_capture import OpenCVCameraProvider
from sha256.config.loader import load_settings
from sha256.config.paths import AppPaths
from sha256.config.schema import Settings
from sha256.core.interfaces import (
    AvatarEngine,
    CameraProvider,
    FaceTracker,
    PerformanceMonitor,
    Renderer,
    VirtualCameraSink,
)
from sha256.performance.monitor import InProcessPerformanceMonitor, NullPerformanceMonitor
from sha256.pipeline.controller import PipelineComponents, PipelineController
from sha256.rendering.null import NullRenderer
from sha256.tracking.mediapipe_tracker import (
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)
from sha256.tracking.null import NullFaceTracker
from sha256.utils.logging import configure_logging, get_logger
from sha256.virtual_camera.null import NullVirtualCameraSink
from sha256.virtual_camera.pyvirtualcam_sink import PyVirtualCameraSink

__all__ = ["ApplicationContext", "build_application", "register_builtin_backends"]

_LOG = get_logger(__name__)


def register_builtin_backends(registries: Registries) -> None:
    """Register every backend this build ships with.

    Only placeholders exist today. When a real backend lands it is added here
    with its ``requires`` list; nothing else in the codebase changes, and the
    module is not imported unless the backend is selected.
    """
    registries.camera.register(
        "null",
        NullCameraProvider,
        description="Synthetic frame generator (no hardware required).",
    )
    registries.camera.register(
        "opencv",
        OpenCVCameraProvider,
        description="Physical camera capture via OpenCV (Media Foundation / DirectShow).",
        requires=("cv2",),
    )
    registries.tracker.register(
        "null",
        NullFaceTracker,
        description="Synthetic tracking data. Not face tracking.",
    )
    registries.tracker.register(
        "mediapipe",
        MediaPipeFaceTracker,
        description="MediaPipe Face Landmarker — 478 landmarks, 52 blendshapes, CPU.",
        requires=("mediapipe",),
    )
    registries.engine.register(
        "null",
        NullAvatarEngine,
        description="Procedural placeholder face. Runs no model.",
    )
    registries.engine.register(
        "warp",
        WarpAvatarEngine,
        description="Warps a portrait photograph onto the tracked face. No model weights.",
        requires=("cv2",),
    )
    registries.renderer.register(
        "null",
        NullRenderer,
        description="CPU letterbox compositor.",
    )
    registries.virtual_camera.register(
        "null",
        NullVirtualCameraSink,
        description="Discards frames. Exposes nothing to the OS.",
    )
    registries.virtual_camera.register(
        "pyvirtualcam",
        PyVirtualCameraSink,
        description=(
            "Publishes to an existing virtual camera device (OBS). "
            "GPL-2.0 — development only, never distributed."
        ),
        requires=("pyvirtualcam",),
    )


@dataclass(slots=True)
class ApplicationContext:
    """Everything a running sha256 instance owns.

    Passed to the UI rather than reached for through globals, so a second
    instance (a benchmark harness, a test) is fully isolated.
    """

    settings: Settings
    paths: AppPaths
    registries: Registries
    monitor: PerformanceMonitor
    avatars: AvatarManager
    controller: PipelineController

    def shutdown(self) -> None:
        """Release everything, in the safe order. Idempotent."""
        self.controller.stop()
        self.avatars.unload_all()
        engine = self.avatars.engine
        if engine is not None:
            engine.stop()


def build_application(
    *,
    settings: Settings | None = None,
    paths: AppPaths | None = None,
    setup_logging: bool = True,
) -> ApplicationContext:
    """Construct a fully wired application.

    Args:
        settings: Pre-built settings. Loaded from files/env when omitted.
        paths: Pre-resolved paths. Resolved when omitted.
        setup_logging: Configure the root logger. Tests pass ``False`` to keep
            pytest's own capture intact.
    """
    paths = paths or AppPaths.resolve()
    paths.ensure()
    settings = settings or load_settings(paths)

    if setup_logging:
        configure_logging(
            level=settings.log_level,
            log_dir=paths.logs,
            json_format=settings.log_json,
        )

    for warning in settings.warnings:
        _LOG.warning("Configuration: %s", warning)

    registries = Registries()
    register_builtin_backends(registries)

    monitor: PerformanceMonitor = (
        InProcessPerformanceMonitor(
            window_size=settings.performance.window_size,
            enabled=True,
        )
        if settings.performance.enabled
        else NullPerformanceMonitor()
    )

    camera: CameraProvider = registries.camera.create(
        settings.camera.backend,
        device_index=settings.camera.device_index,
        width=settings.camera.width,
        height=settings.camera.height,
        target_fps=settings.camera.target_fps,
        use_mjpeg=settings.camera.use_mjpeg,
    )
    tracker: FaceTracker = _create_tracker(registries, settings, paths)
    renderer: Renderer = registries.renderer.create(
        settings.rendering.backend,
        width=settings.rendering.output_width,
        height=settings.rendering.output_height,
    )
    virtual_camera: VirtualCameraSink = _create_virtual_camera(registries, settings)
    engine: AvatarEngine = _create_engine(registries, settings, paths)
    engine.start()

    avatars = _build_avatar_manager(settings, paths, engine)

    controller = PipelineController(
        PipelineComponents(
            camera=camera,
            tracker=tracker,
            renderer=renderer,
            virtual_camera=virtual_camera,
            avatars=avatars,
            monitor=monitor,
        ),
        publish_output=settings.virtual_camera.auto_start,
    )

    _LOG.info(
        "sha256 wired: camera=%s tracker=%s engine=%s renderer=%s vcam=%s",
        settings.camera.backend,
        settings.tracking.backend,
        settings.engine.backend,
        settings.rendering.backend,
        settings.virtual_camera.backend,
    )

    return ApplicationContext(
        settings=settings,
        paths=paths,
        registries=registries,
        monitor=monitor,
        avatars=avatars,
        controller=controller,
    )


def _create_virtual_camera(registries: Registries, settings: Settings) -> VirtualCameraSink:
    """Build the output sink and declare its format.

    Configured here, not at pipeline start, because the output format is a
    property of the wiring rather than of a run. Nothing was calling
    ``configure()`` at all, which made ``virtual_camera.auto_start = true``
    fail on the very check that exists to catch that ordering mistake — the
    placeholder sink caught the bug exactly as designed, and this is the fix.

    The two backends take different arguments for the same reason the trackers
    do: a sink that *creates* a device can be told what to call it, and a sink
    that writes to somebody else's device can only be told which one to pick.
    """
    virtual_camera = settings.virtual_camera

    if virtual_camera.backend == "pyvirtualcam":
        sink: VirtualCameraSink = registries.virtual_camera.create(
            "pyvirtualcam", device=virtual_camera.device
        )
    else:
        sink = registries.virtual_camera.create(
            virtual_camera.backend, device_name=virtual_camera.device_name
        )

    sink.configure(
        settings.rendering.output_width,
        settings.rendering.output_height,
        settings.rendering.target_fps,
    )
    return sink


def _create_engine(registries: Registries, settings: Settings, paths: AppPaths) -> AvatarEngine:
    """Build the configured avatar engine.

    The warp engine needs the face-landmark model to read the avatar's own
    landmarks when it loads a portrait — the same model the tracker uses, so the
    two meshes are the same topology by construction.
    """
    if settings.engine.backend != "warp":
        placeholder: AvatarEngine = registries.engine.create(settings.engine.backend)
        return placeholder

    models_root = paths.resolve_relative(settings.models.root_path)
    engine: AvatarEngine = registries.engine.create(
        "warp", model_path=models_root / DEFAULT_MODEL_RELATIVE_PATH
    )
    return engine


def _create_tracker(registries: Registries, settings: Settings, paths: AppPaths) -> FaceTracker:
    """Build the configured face tracker.

    Camera backends share one constructor signature, so the camera is created
    inline above. Trackers do not: MediaPipe needs a model bundle resolved
    against the models root, and the placeholder needs nothing at all. Resolving
    that difference is exactly the composition root's job — it is the one place
    allowed to know which backend is which, and keeping it here is what stops
    that knowledge leaking into the pipeline or the UI.
    """
    tracking = settings.tracking

    if tracking.backend != "mediapipe":
        placeholder: FaceTracker = registries.tracker.create(tracking.backend)
        return placeholder

    models_root = paths.resolve_relative(settings.models.root_path)
    configured = tracking.model_path
    model_path = (
        models_root / DEFAULT_MODEL_RELATIVE_PATH
        if configured is None
        else (Path(configured) if Path(configured).is_absolute() else models_root / configured)
    )

    tracker: FaceTracker = registries.tracker.create(
        "mediapipe",
        model_path=model_path,
        delegate=tracking.delegate,
        num_faces=tracking.num_faces,
        min_detection_confidence=tracking.min_detection_confidence,
        min_presence_confidence=tracking.min_presence_confidence,
        min_tracking_confidence=tracking.min_tracking_confidence,
        smoothing=tracking.smoothing,
        mouth_smoothing=tracking.mouth_smoothing,
    )
    return tracker


def _build_avatar_manager(
    settings: Settings, paths: AppPaths, engine: AvatarEngine
) -> AvatarManager:
    library_path: Path = paths.resolve_relative(settings.avatars.library_path)
    repository = FilesystemAvatarRepository(library_path)
    cache = ResidencyCache(
        max_resident=settings.avatars.max_resident,
        vram_budget_mb=settings.gpu.vram_budget_mb,
    )
    manager = AvatarManager(
        repository,
        cache,
        engine=engine,
        verify_checksums=settings.avatars.verify_checksums,
    )
    manager.refresh()

    default_id = settings.avatars.default_avatar_id
    if default_id and repository.exists(default_id):
        try:
            manager.select(default_id)
        except Exception:
            _LOG.exception("Could not select default avatar %s", default_id)

    return manager
