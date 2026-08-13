"""Placeholder avatar engine.

Renders a simple procedural face driven by real tracking data — head pose moves
it, blendshapes open the mouth and close the eyes. It performs no inference and
loads no weights.

The point is not the visuals. It is that the *contract* around loading,
unloading, VRAM accounting, and per-frame animation is exercised end-to-end
before a real engine (LivePortrait or an alternative) is chosen, so integrating
one becomes a matter of implementing a known-good interface.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sha256.core.avatar import AvatarDescriptor
from sha256.core.enums import DeviceKind, PixelFormat
from sha256.core.errors import AvatarLoadError
from sha256.core.types import FaceTrackingResult, RenderedFrame, VideoFrame
from sha256.utils.logging import get_logger
from sha256.utils.timing import now

__all__ = ["NullAvatarEngine"]

_LOG = get_logger(__name__)

ENGINE_ID = "null"


class NullAvatarEngine:
    """A working engine that draws a schematic face instead of running a model."""

    def __init__(self, *, width: int = 512, height: int = 512) -> None:
        self._width = width
        self._height = height
        self._running = False
        self._loaded: dict[str, AvatarDescriptor] = {}
        self._buffers: dict[str, np.ndarray] = {}

    # --- Component ---------------------------------------------------------

    @property
    def backend_id(self) -> str:
        return ENGINE_ID

    @property
    def engine_id(self) -> str:
        return ENGINE_ID

    @property
    def device(self) -> DeviceKind:
        return DeviceKind.CPU

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        for avatar_id in list(self._loaded):
            self.unload_avatar(avatar_id)
        self._running = False

    # --- AvatarEngine ------------------------------------------------------

    def supports(self, descriptor: AvatarDescriptor) -> bool:
        """Accept avatars that ask for this engine.

        The placeholder deliberately does not accept everything: an avatar built
        for a neural engine must not silently fall back to a schematic face and
        look like a rendering bug.
        """
        return descriptor.manifest.engine.engine_id == ENGINE_ID

    def load_avatar(self, descriptor: AvatarDescriptor) -> None:
        if descriptor.id in self._loaded:
            return
        if not self.supports(descriptor):
            msg = (
                f"avatar {descriptor.id!r} requires engine "
                f"{descriptor.manifest.engine.engine_id!r}, not {ENGINE_ID!r}"
            )
            raise AvatarLoadError(msg)

        self._loaded[descriptor.id] = descriptor
        rendering = descriptor.manifest.rendering
        self._buffers[descriptor.id] = np.zeros(
            (rendering.output_height, rendering.output_width, 3), dtype=np.uint8
        )
        _LOG.info("Loaded avatar %s into null engine", descriptor.id)

    def unload_avatar(self, avatar_id: str) -> None:
        self._loaded.pop(avatar_id, None)
        self._buffers.pop(avatar_id, None)

    @property
    def loaded_avatars(self) -> Sequence[str]:
        return tuple(self._loaded)

    def estimated_vram_mb(self, avatar_id: str) -> int | None:
        """Always ``None``: this engine holds nothing on the GPU.

        Reporting 0 would be wrong in a subtler way — it would read as a
        measured value rather than an absent one, and the residency policy
        distinguishes the two.
        """
        return None

    def animate(
        self, avatar_id: str, tracking: FaceTrackingResult, frame: VideoFrame
    ) -> RenderedFrame:
        del frame  # this engine synthesises from tracking alone
        descriptor = self._loaded.get(avatar_id)
        if descriptor is None:
            msg = f"avatar {avatar_id!r} is not loaded"
            raise AvatarLoadError(msg)

        buffer = self._buffers[avatar_id]
        buffer.fill(0)
        height, width = buffer.shape[:2]

        pose = tracking.head_pose
        rendering = descriptor.manifest.rendering
        yaw = (pose.yaw if pose else 0.0) * rendering.head_motion_scale
        pitch = (pose.pitch if pose else 0.0) * rendering.head_motion_scale

        # Map head rotation to a translation of the drawn face. A real engine
        # would re-synthesise; shifting is enough to prove the data arrived.
        offset_x = int(np.clip(yaw / 45.0, -1.0, 1.0) * width * 0.15)
        offset_y = int(np.clip(-pitch / 45.0, -1.0, 1.0) * height * 0.15)
        centre_x = width // 2 + offset_x
        centre_y = height // 2 + offset_y

        self._draw_face(buffer, centre_x, centre_y, tracking, rendering.expression_scale)

        return RenderedFrame(
            data=buffer,
            pixel_format=PixelFormat.RGB24,
            metadata=tracking.metadata,
            avatar_id=avatar_id,
            render_timestamp=now(),
            is_placeholder=True,
        )

    # --- drawing helpers ---------------------------------------------------

    def _draw_face(
        self,
        buffer: np.ndarray,
        centre_x: int,
        centre_y: int,
        tracking: FaceTrackingResult,
        expression_scale: float,
    ) -> None:
        height, width = buffer.shape[:2]
        radius = int(min(width, height) * 0.3)

        yy, xx = np.ogrid[:height, :width]
        head = (xx - centre_x) ** 2 + (yy - centre_y) ** 2 <= radius**2
        buffer[head] = (90, 110, 150)

        shapes = tracking.blendshapes
        blink_left = shapes.get("eyeBlinkLeft") if shapes else 0.0
        blink_right = shapes.get("eyeBlinkRight") if shapes else 0.0
        jaw_open = (shapes.get("jawOpen") if shapes else 0.0) * expression_scale

        eye_offset = radius // 3
        eye_radius = max(2, radius // 8)
        eye_y = centre_y - eye_offset // 2
        self._draw_eye(buffer, centre_x - eye_offset, eye_y, eye_radius, blink_left)
        self._draw_eye(buffer, centre_x + eye_offset, eye_y, eye_radius, blink_right)

        mouth_half_width = radius // 3
        mouth_half_height = max(1, int(radius * 0.05 + radius * 0.25 * jaw_open))
        mouth_y = centre_y + radius // 3
        y0 = max(0, mouth_y - mouth_half_height)
        y1 = min(height, mouth_y + mouth_half_height)
        x0 = max(0, centre_x - mouth_half_width)
        x1 = min(width, centre_x + mouth_half_width)
        buffer[y0:y1, x0:x1] = (30, 20, 40)

    @staticmethod
    def _draw_eye(
        buffer: np.ndarray, centre_x: int, centre_y: int, radius: int, blink: float
    ) -> None:
        height, width = buffer.shape[:2]
        open_amount = max(0.05, 1.0 - float(blink))
        half_height = max(1, int(radius * open_amount))
        y0 = max(0, centre_y - half_height)
        y1 = min(height, centre_y + half_height)
        x0 = max(0, centre_x - radius)
        x1 = min(width, centre_x + radius)
        buffer[y0:y1, x0:x1] = (235, 235, 245)
