"""Replaceable component contracts.

Every swappable part of sha256 is declared here as a :class:`typing.Protocol`.
Protocols rather than abstract base classes, because:

* a backend adapter does not have to import (and therefore depend on) sha256
  types just to be usable — structural typing is enough;
* test doubles are trivial to write;
* no metaclass or MRO surprises when a backend already inherits from a
  vendor base class.

The one rule that makes the whole design work: **no method here mentions a
vendor type.** Parameters and returns are drawn from :mod:`sha256.core.types`
and :mod:`sha256.core.avatar` only. A backend translates at its own edge.

None of these are implemented for real yet. The foundation ships ``Null*``
placeholders (see each feature package) so that the application shell runs, the
wiring is exercised, and tests have something concrete to assert against.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from sha256.core.avatar import AvatarDescriptor
from sha256.core.enums import DeviceKind, PipelineStage
from sha256.core.metrics import PerformanceSnapshot
from sha256.core.types import (
    FaceTrackingResult,
    FrameMetadata,
    RenderedFrame,
    VideoFrame,
)

__all__ = [
    "AvatarEngine",
    "AvatarRepository",
    "CameraProvider",
    "CameraSpec",
    "Component",
    "FaceTracker",
    "PerformanceMonitor",
    "Renderer",
    "VirtualCameraSink",
]


@runtime_checkable
class Component(Protocol):
    """Common lifecycle shared by every pipeline component.

    ``start``/``stop`` are separate from construction so that a component can be
    created, inspected, and wired while the pipeline is stopped — acquiring a
    camera handle or a CUDA context in ``__init__`` would make the object graph
    impossible to build without hardware.

    Implementations must be idempotent: calling ``start`` twice or ``stop`` on a
    stopped component is a no-op, never an error.
    """

    @property
    def backend_id(self) -> str:
        """Registry id of this implementation, e.g. ``"null"`` or ``"mediapipe"``."""
        ...

    @property
    def is_running(self) -> bool: ...

    def start(self) -> None:
        """Acquire resources and become ready to process."""
        ...

    def stop(self) -> None:
        """Release all resources. Must not raise for an already-stopped component."""
        ...


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


class CameraSpec(Protocol):
    """Description of an enumerable capture device."""

    @property
    def device_id(self) -> str: ...

    @property
    def display_name(self) -> str: ...


@runtime_checkable
class CameraProvider(Component, Protocol):
    """Source of :class:`VideoFrame` objects.

    Implementations may wrap a physical webcam, a video file, or a synthetic
    generator. The pipeline cannot tell the difference, which is what makes the
    system testable without hardware and demoable without a camera.
    """

    def enumerate_devices(self) -> Sequence[CameraSpec]:
        """List available capture devices. Safe to call while stopped."""
        ...

    def read(self) -> VideoFrame | None:
        """Return the next frame, or ``None`` if none is available right now.

        Must not block indefinitely. Returning ``None`` is normal back-pressure,
        not an error; raise :class:`~sha256.core.errors.CameraError` only when
        the device is genuinely broken.
        """
        ...

    def frames(self) -> Iterator[VideoFrame]:
        """Iterate frames until stopped. Convenience over repeated :meth:`read`."""
        ...


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


@runtime_checkable
class FaceTracker(Component, Protocol):
    """Turns a :class:`VideoFrame` into a :class:`FaceTrackingResult`.

    Candidate implementations (MediaPipe Face Landmarker, OpenSeeFace, ...) are
    compared in ``docs/OPEN_SOURCE_EVALUATION.md``. None is integrated yet.
    """

    @property
    def landmark_scheme(self) -> str:
        """Topology this tracker emits, e.g. ``"mediapipe_478"``.

        Exposed so a consumer can refuse an incompatible pairing at wiring time
        rather than producing silently wrong animation at frame time.
        """
        ...

    @property
    def provides_blendshapes(self) -> bool:
        """Whether results carry a populated :class:`BlendshapeSet`.

        Trackers that only emit landmarks need a separate solver stage before an
        expression-driven engine can use them.
        """
        ...

    def process(self, frame: VideoFrame) -> FaceTrackingResult:
        """Track a single frame. Must always return a result, never ``None``."""
        ...

    def reset(self) -> None:
        """Drop temporal state (smoothing filters, previous-frame priors)."""
        ...


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


@runtime_checkable
class AvatarEngine(Component, Protocol):
    """Drives a loaded avatar from tracking data.

    The engine owns the expensive resources: model weights, CUDA context,
    warmed-up graphs. Because those are per-avatar, loading and unloading are
    explicit operations coordinated by the :class:`AvatarManager` rather than
    side effects of ``start``/``stop``.
    """

    @property
    def engine_id(self) -> str:
        """Id matched against :attr:`EngineRequirement.engine_id` in a manifest."""
        ...

    @property
    def device(self) -> DeviceKind: ...

    def supports(self, descriptor: AvatarDescriptor) -> bool:
        """Whether this engine can drive the given avatar.

        Checked before loading so an incompatible avatar surfaces as a clear UI
        message instead of a mid-load crash.
        """
        ...

    def load_avatar(self, descriptor: AvatarDescriptor) -> None:
        """Make an avatar resident and ready. May be slow; never call on the UI thread."""
        ...

    def unload_avatar(self, avatar_id: str) -> None:
        """Release the avatar's resources. No-op if it is not loaded."""
        ...

    @property
    def loaded_avatars(self) -> Sequence[str]: ...

    def estimated_vram_mb(self, avatar_id: str) -> int | None:
        """Measured VRAM footprint once loaded, else ``None``.

        The residency policy prefers this over the manifest's author estimate.
        """
        ...

    def animate(self, avatar_id: str, tracking: FaceTrackingResult) -> RenderedFrame:
        """Produce one animated frame for ``avatar_id`` from ``tracking``."""
        ...


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@runtime_checkable
class Renderer(Component, Protocol):
    """Composites engine output into the final delivered frame.

    Kept separate from :class:`AvatarEngine` because compositing concerns
    (background, letterboxing, output resolution, overlays) are shared across
    every engine. A neural engine and a 3D engine differ upstream of here but
    both need the same framing work done afterwards.
    """

    @property
    def output_format(self) -> str:
        """Pixel format id this renderer emits, matching a ``PixelFormat`` value."""
        ...

    def configure(self, width: int, height: int) -> None:
        """Set output dimensions. Valid while stopped or running."""
        ...

    def render(self, source: RenderedFrame) -> RenderedFrame:
        """Composite ``source`` into a delivery-ready frame."""
        ...

    def render_placeholder(self, metadata: FrameMetadata, message: str) -> RenderedFrame:
        """Produce a frame for idle/error states.

        Required so the virtual camera can keep a valid signal alive while no
        avatar is loaded — consumers like Zoom handle a static frame far better
        than a stream that stops.
        """
        ...


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@runtime_checkable
class VirtualCameraSink(Component, Protocol):
    """Publishes frames so other applications can consume them as a webcam.

    Named a *sink* rather than a *provider* because from sha256's perspective
    it consumes frames; it is only a provider from the operating system's side.
    Windows approaches are compared in
    ``docs/decisions/ADR-0005-virtual-camera-strategy.md``.
    """

    @property
    def is_available(self) -> bool:
        """Whether the backing virtual-camera device exists on this machine.

        Checked before ``start`` so the UI can explain what to install rather
        than failing with a driver-level error.
        """
        ...

    @property
    def device_name(self) -> str | None:
        """Name the OS exposes to consumers, once started."""
        ...

    def configure(self, width: int, height: int, fps: float) -> None:
        """Declare the output format. Must be called before :meth:`start`."""
        ...

    def send(self, frame: RenderedFrame) -> None:
        """Publish a frame. Should drop rather than block if the sink is behind."""
        ...


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


@runtime_checkable
class PerformanceMonitor(Protocol):
    """Collects timing and resource metrics.

    Not a :class:`Component`: monitoring outlives any single pipeline run, and
    making it startable would invite the mistake of losing metrics exactly when
    a component fails to start.

    Implementations must be cheap enough to call on every frame and safe to call
    from the capture thread.
    """

    @property
    def enabled(self) -> bool: ...

    def measure(self, stage: PipelineStage) -> AbstractContextManager[None]:
        """Time a block and record it against ``stage``.

        Usage::

            with monitor.measure(PipelineStage.TRACKING):
                result = tracker.process(frame)
        """
        ...

    def record_duration(self, stage: PipelineStage, seconds: float) -> None:
        """Record a duration measured elsewhere (e.g. reported by a backend)."""
        ...

    def record_frame(self, stage: PipelineStage) -> None:
        """Count one frame through ``stage``, for throughput/FPS."""
        ...

    def record_dropped(self, stage: PipelineStage, count: int = 1) -> None:
        """Count frames deliberately discarded at ``stage``."""
        ...

    def record_value(self, name: str, value: float) -> None:
        """Record an arbitrary gauge, e.g. ``"gpu.vram_used_mb"``."""
        ...

    def snapshot(self) -> PerformanceSnapshot:
        """Return a point-in-time copy of all metrics.

        Immutable, so a UI thread can read it without locking against the
        capture thread that produced it.
        """
        ...

    def reset(self) -> None:
        """Discard all accumulated samples."""
        ...


# ---------------------------------------------------------------------------
# Avatar library
# ---------------------------------------------------------------------------


@runtime_checkable
class AvatarRepository(Protocol):
    """Read access to the avatar library.

    Deliberately read-only. Creating and modifying avatars is an owner/developer
    operation handled by a separate, gated interface — see
    ``docs/architecture/AVATAR_SYSTEM.md`` — so the normal running application
    has no code path that can mutate the library.
    """

    def discover(self, *, force: bool = False) -> Sequence[AvatarDescriptor]:
        """Return every valid avatar in the library.

        Implementations may cache: discovery is called often (the gallery, the
        manager, diagnostics) and re-reading every manifest each time would be
        wasteful. Pass ``force=True`` to re-scan storage, which is what picks up
        an avatar added while the application is running.

        Invalid avatars are reported through :meth:`errors` rather than raising,
        so one malformed manifest cannot hide an otherwise healthy library.
        """
        ...

    def get(self, avatar_id: str) -> AvatarDescriptor:
        """Return one avatar. Raises ``AvatarNotFoundError`` if absent."""
        ...

    def exists(self, avatar_id: str) -> bool: ...

    def errors(self) -> Sequence[tuple[str, str]]:
        """``(location, reason)`` pairs for entries that failed to load."""
        ...
