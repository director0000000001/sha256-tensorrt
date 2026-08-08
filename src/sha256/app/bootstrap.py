"""Composition root.

The single place where concrete implementations are chosen and wired. Every
other module depends on protocols only; this one does the ``import`` and the
``registry.create(...)``.

Keeping it in one file has a practical payoff: to find out what SHA-256 is
actually running, you read this file, and nothing else can quietly introduce a
dependency on a specific tracker or engine behind your back.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sha256.app.registry import Registries
from sha256.avatar.cache import ResidencyCache
from sha256.avatar.engine_null import NullAvatarEngine
from sha256.avatar.manager import AvatarManager
from sha256.avatar.repository import FilesystemAvatarRepository
from sha256.camera.null import NullCameraProvider
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
from sha256.tracking.null import NullFaceTracker
from sha256.utils.logging import configure_logging, get_logger
from sha256.virtual_camera.null import NullVirtualCameraSink

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
    registries.tracker.register(
        "null",
        NullFaceTracker,
        description="Synthetic tracking data. Not face tracking.",
    )
    registries.engine.register(
        "null",
        NullAvatarEngine,
        description="Procedural placeholder face. Runs no model.",
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


@dataclass(slots=True)
class ApplicationContext:
    """Everything a running SHA-256 instance owns.

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
        width=settings.camera.width,
        height=settings.camera.height,
        target_fps=settings.camera.target_fps,
    )
    tracker: FaceTracker = registries.tracker.create(settings.tracking.backend)
    renderer: Renderer = registries.renderer.create(
        settings.rendering.backend,
        width=settings.rendering.output_width,
        height=settings.rendering.output_height,
    )
    virtual_camera: VirtualCameraSink = registries.virtual_camera.create(
        settings.virtual_camera.backend,
        device_name=settings.virtual_camera.device_name,
    )
    engine: AvatarEngine = registries.engine.create(settings.engine.backend)
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
        "SHA-256 wired: camera=%s tracker=%s engine=%s renderer=%s vcam=%s",
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
