"""Landmark-warp avatar engine.

Puts a portrait photograph onto the tracked face and deforms it, per frame, so
the avatar's mouth, eyes, and brows follow the user's. It runs no neural
network: the avatar's own landmarks are found once when it loads, a triangle
mesh is built between them, and every frame each triangle is warped from where
it sits in the photograph to where the matching landmark sits on the live face.

**What this is and is not.** It is a real, working avatar engine — the face
moves, blinks, and talks with you, and it is deterministic, cheap, and carries
**no model-weight licensing risk at all**, which no neural option currently
does (see ``docs/OPEN_SOURCE_EVALUATION.md``). It is *not* photorealistic
portrait synthesis: it reshapes existing pixels rather than generating new ones,
so extreme angles stretch rather than reveal, and the avatar cannot show what
the photograph does not contain. LivePortrait remains the target for quality;
this is the engine that exists today and the fallback if the neural path is too
slow, too heavy, or legally blocked.

Design notes
------------
*   **The mesh is decimated on purpose.** MediaPipe's full tessellation is
    roughly 900 triangles, and warping each one costs a separate OpenCV call —
    tens of milliseconds, most of it per-call overhead. A curated subset of
    landmarks covering the face oval, brows, eyes, nose, and lips gives a mesh
    an order of magnitude smaller that still moves everything a viewer watches.
*   **Everything that can be precomputed is.** The portrait's landmarks and the
    triangulation are built once at load; only the blend mask depends on where
    the live face is and has to be rebuilt each frame.
*   **The avatar is composited over the camera frame**, not rendered on its own
    background, so what you see is your own scene with a different face in it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from sha256.core.avatar import AvatarDescriptor
from sha256.core.enums import DeviceKind, PixelFormat
from sha256.core.errors import AvatarLoadError
from sha256.core.types import (
    FaceTrackingResult,
    FrameMetadata,
    RenderedFrame,
    VideoFrame,
)
from sha256.utils.logging import get_logger
from sha256.utils.timing import now

__all__ = ["ENGINE_ID", "WarpAvatarEngine"]

_LOG = get_logger(__name__)

ENGINE_ID = "warp"

SOURCE_PORTRAIT_ROLE = "source_portrait"

# A curated subset of the 478-point mesh: the face oval, both brows, both eyes,
# the nose, and the lips. Chosen so the mesh stays small enough to warp inside a
# frame budget while still carrying every feature a viewer actually watches.
# fmt: off
_FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
)
_LEFT_BROW = (46, 53, 52, 65, 55, 107)
_RIGHT_BROW = (276, 283, 282, 295, 285, 336)
_LEFT_EYE = (33, 160, 158, 133, 153, 144)
_RIGHT_EYE = (362, 385, 387, 263, 373, 380)
_NOSE = (168, 6, 197, 195, 5, 4, 98, 97, 2, 326, 327)
_OUTER_LIPS = (61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181)
_INNER_LIPS = (78, 81, 13, 311, 308, 402, 14, 178)
_CHEEKS = (234, 454, 50, 280, 116, 345, 205, 425)
# fmt: on

MESH_INDICES: tuple[int, ...] = tuple(
    dict.fromkeys(
        _FACE_OVAL
        + _LEFT_BROW
        + _RIGHT_BROW
        + _LEFT_EYE
        + _RIGHT_EYE
        + _NOSE
        + _OUTER_LIPS
        + _INNER_LIPS
        + _CHEEKS
    )
)

# How far outside the landmark hull the blend mask reaches, and how soft its
# edge is, both as a fraction of face width. A hard edge reads as a cut-out
# sticker; too soft and the real face shows through the avatar.
_MASK_GROW = 0.06
_MASK_FEATHER = 0.05


class _LoadedAvatar:
    """Everything precomputed for one avatar, held for the life of the load."""

    __slots__ = ("descriptor", "image", "points", "triangles")

    def __init__(
        self,
        descriptor: AvatarDescriptor,
        image: np.ndarray,
        points: np.ndarray,
        triangles: np.ndarray,
    ) -> None:
        self.descriptor = descriptor
        self.image = image
        self.points = points
        self.triangles = triangles


class WarpAvatarEngine:
    """Deforms a portrait photograph to follow a tracked face."""

    def __init__(self, *, model_path: Path | str | None = None) -> None:
        # The tracker used to find the avatar's own landmarks. Only needed while
        # loading an avatar, never on the frame path.
        self._model_path = Path(model_path) if model_path is not None else None
        self._running = False
        self._loaded: dict[str, _LoadedAvatar] = {}

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
        return descriptor.manifest.engine.engine_id == ENGINE_ID

    def load_avatar(self, descriptor: AvatarDescriptor) -> None:
        """Read the portrait, find its landmarks, and build the mesh.

        Slow — it runs a face detection on the photograph — which is why it
        happens here and not on the first frame.
        """
        if descriptor.id in self._loaded:
            return
        if not self.supports(descriptor):
            msg = (
                f"avatar {descriptor.id!r} requires engine "
                f"{descriptor.manifest.engine.engine_id!r}, not {ENGINE_ID!r}"
            )
            raise AvatarLoadError(msg)

        image = self._read_portrait(descriptor)
        points = self._landmarks_of(image, descriptor.id)
        triangles = _triangulate(points, image.shape[1], image.shape[0])

        self._loaded[descriptor.id] = _LoadedAvatar(descriptor, image, points, triangles)
        _LOG.info(
            "Loaded avatar %s: %dx%d portrait, %d mesh points, %d triangles",
            descriptor.id,
            image.shape[1],
            image.shape[0],
            len(points),
            len(triangles),
        )

    def unload_avatar(self, avatar_id: str) -> None:
        self._loaded.pop(avatar_id, None)

    @property
    def loaded_avatars(self) -> Sequence[str]:
        return tuple(self._loaded)

    def estimated_vram_mb(self, avatar_id: str) -> int | None:
        """Always ``None``: this engine holds nothing on the GPU."""
        del avatar_id
        return None

    def animate(
        self, avatar_id: str, tracking: FaceTrackingResult, frame: VideoFrame
    ) -> RenderedFrame:
        avatar = self._loaded.get(avatar_id)
        if avatar is None:
            msg = f"avatar {avatar_id!r} is not loaded"
            raise AvatarLoadError(msg)

        source = frame.data
        if frame.pixel_format not in (PixelFormat.BGR24, PixelFormat.RGB24):
            msg = f"the warp engine cannot composite over {frame.pixel_format.value} frames"
            raise AvatarLoadError(msg)

        # Losing the face is normal and constant. Passing the scene through
        # unchanged is the least surprising thing to do with a frame that has
        # nowhere to put an avatar.
        if not tracking.face_detected or tracking.landmarks is None:
            return self._passthrough(source, frame, tracking, avatar_id)

        target = _pixel_points(tracking.landmarks.points, source.shape[1], source.shape[0])
        if target is None:
            return self._passthrough(source, frame, tracking, avatar_id)

        output = _warp_onto(avatar, target, source)

        return RenderedFrame(
            data=output,
            pixel_format=frame.pixel_format,
            metadata=tracking.metadata,
            avatar_id=avatar_id,
            render_timestamp=now(),
            is_placeholder=False,
        )

    @staticmethod
    def _passthrough(
        source: np.ndarray, frame: VideoFrame, tracking: FaceTrackingResult, avatar_id: str
    ) -> RenderedFrame:
        return RenderedFrame(
            data=source,
            pixel_format=frame.pixel_format,
            metadata=tracking.metadata,
            avatar_id=avatar_id,
            render_timestamp=now(),
            is_placeholder=True,
        )

    # --- loading helpers ---------------------------------------------------

    def _read_portrait(self, descriptor: AvatarDescriptor) -> np.ndarray:
        import cv2

        asset = descriptor.manifest.asset_by_role(SOURCE_PORTRAIT_ROLE)
        if asset is None:
            msg = (
                f"avatar {descriptor.id!r} declares no {SOURCE_PORTRAIT_ROLE!r} asset; "
                f"the warp engine needs a portrait photograph"
            )
            raise AvatarLoadError(msg)

        path = descriptor.asset_path(asset)
        if not path.is_file():
            msg = f"avatar {descriptor.id!r}: portrait not found at {path}"
            raise AvatarLoadError(msg)

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            msg = f"avatar {descriptor.id!r}: {path.name} could not be read as an image"
            raise AvatarLoadError(msg)
        return image

    def _landmarks_of(self, image: np.ndarray, avatar_id: str) -> np.ndarray:
        """Find the portrait's own landmarks, in pixels.

        The same tracker that drives the pipeline is used, so the avatar's mesh
        and the live mesh are guaranteed to be the same topology — a different
        detector here would silently pair up unrelated points.
        """
        from sha256.tracking.mediapipe_tracker import (
            DEFAULT_MODEL_RELATIVE_PATH,
            MediaPipeFaceTracker,
        )

        model_path = self._model_path or (Path("models") / DEFAULT_MODEL_RELATIVE_PATH)
        tracker = MediaPipeFaceTracker(model_path=model_path)
        tracker.start()
        try:
            result = tracker.process(
                VideoFrame(
                    data=np.ascontiguousarray(image),
                    pixel_format=PixelFormat.BGR24,
                    metadata=FrameMetadata(frame_id=0, capture_timestamp=0.0, source_id="portrait"),
                )
            )
        finally:
            tracker.stop()

        if not result.face_detected or result.landmarks is None:
            msg = (
                f"avatar {avatar_id!r}: no face was found in the portrait. "
                f"It needs a clear, front-facing photograph of one face."
            )
            raise AvatarLoadError(msg)

        points = _pixel_points(result.landmarks.points, image.shape[1], image.shape[0])
        if points is None:
            msg = f"avatar {avatar_id!r}: the portrait's landmarks are unusable"
            raise AvatarLoadError(msg)
        return points


def _pixel_points(normalised: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """The mesh subset in pixel coordinates, or ``None`` if the mesh is wrong."""
    if normalised.shape[0] <= max(MESH_INDICES):
        return None

    subset = normalised[list(MESH_INDICES), :2].astype(np.float32)
    subset[:, 0] *= width
    subset[:, 1] *= height
    return subset


def _triangulate(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Delaunay triangulation over the mesh points, as index triples.

    Indices rather than coordinates: the triangulation is computed once on the
    portrait and reused every frame against the live points, which is what makes
    the two meshes correspond.
    """
    import cv2

    # A rectangle large enough that no point sits on its edge, where Subdiv2D
    # returns triangles referencing the virtual corner vertices.
    padding = max(width, height)
    subdiv = cv2.Subdiv2D((-padding, -padding, width + 2 * padding, height + 2 * padding))
    for point in points:
        subdiv.insert((float(point[0]), float(point[1])))

    lookup = {(round(float(p[0]), 2), round(float(p[1]), 2)): i for i, p in enumerate(points)}

    triangles: list[tuple[int, int, int]] = []
    for triangle in subdiv.getTriangleList():
        corners = [
            lookup.get((round(float(triangle[i]), 2), round(float(triangle[i + 1]), 2)))
            for i in (0, 2, 4)
        ]
        if all(corner is not None for corner in corners):
            triangles.append((corners[0], corners[1], corners[2]))  # type: ignore[arg-type]

    if not triangles:
        msg = "the avatar mesh could not be triangulated"
        raise AvatarLoadError(msg)
    return np.array(triangles, dtype=np.int32)


def _warp_onto(avatar: _LoadedAvatar, target: np.ndarray, scene: np.ndarray) -> np.ndarray:
    """Warp the avatar's mesh onto ``target`` and blend it over ``scene``.

    Written around one measured fact: the cost is dominated by *per-call*
    overhead, not by pixels. An earlier version warped both the image and a mask
    for every triangle and cost 42 ms at p95, over the whole 33 ms frame budget.

    Three things follow from that, and they are the reason this reads the way it
    does rather than the obvious way:

    *   **One warp per triangle, not two.** The blend mask is built once, from
        the destination convex hull, instead of being warped alongside each
        piece.
    *   **Everything happens inside the face's bounding box.** A 1280x720 canvas
        is allocated and cleared per frame otherwise, for a face that occupies a
        fraction of it.
    *   **No boolean fancy-indexing on the hot path.** ``copyTo`` with a mask
        does the same job without building an index array per triangle.
    """
    import cv2

    height, width = scene.shape[:2]

    # Work in the face's bounding box, padded, and clipped to the frame.
    pad = 8
    fx0 = max(int(target[:, 0].min()) - pad, 0)
    fy0 = max(int(target[:, 1].min()) - pad, 0)
    fx1 = min(int(target[:, 0].max()) + pad, width)
    fy1 = min(int(target[:, 1].max()) + pad, height)
    if fx1 - fx0 < 2 or fy1 - fy0 < 2:
        return scene.copy()

    box_w, box_h = fx1 - fx0, fy1 - fy0
    # Cast back explicitly: subtracting an int tuple promotes to float64, and
    # OpenCV's geometry functions only accept 32-bit float or int.
    local = (target - (fx0, fy0)).astype(np.float32)

    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    source_points = avatar.points

    for a, b, c in avatar.triangles:
        dst = local[[a, b, c]]

        x, y, w, h = cv2.boundingRect(dst)
        # Clip to the canvas; a triangle partly outside is normal at the edges.
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, box_w), min(y + h, box_h)
        if x1 <= x0 or y1 <= y0:
            continue

        src = source_points[[a, b, c]]
        sx, sy, sw, sh = cv2.boundingRect(src)
        if sw <= 0 or sh <= 0:
            continue

        matrix = cv2.getAffineTransform(
            (src - (sx, sy)).astype(np.float32), (dst - (x, y)).astype(np.float32)
        )
        patch = avatar.image[sy : sy + sh, sx : sx + sw]
        if patch.size == 0:
            continue

        piece = cv2.warpAffine(patch, matrix, (w, h), flags=cv2.INTER_LINEAR)

        cover = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(cover, (dst - (x, y)).astype(np.int32), 255, cv2.LINE_8)

        cv2.copyTo(
            piece[y0 - y : y1 - y, x0 - x : x1 - x],
            cover[y0 - y : y1 - y, x0 - x : x1 - x],
            canvas[y0:y1, x0:x1],
        )

    alpha = _destination_mask(local, box_w, box_h)

    output = scene.copy()
    region = output[fy0:fy1, fx0:fx1].astype(np.float32)
    blended = region * (1.0 - alpha) + canvas.astype(np.float32) * alpha
    output[fy0:fy1, fx0:fx1] = blended.astype(np.uint8)
    return output


def _destination_mask(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Soft-edged mask over the live face hull, in the local box."""
    import cv2

    hull = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(points.astype(np.int32)), 255)

    face_width = max(float(points[:, 0].max() - points[:, 0].min()), 1.0)
    grow = max(1, int(face_width * _MASK_GROW))
    feather = max(3, int(face_width * _MASK_FEATHER) | 1)

    grown = cv2.dilate(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    softened = cv2.GaussianBlur(grown, (feather, feather), 0)
    mask: np.ndarray = (softened.astype(np.float32) / 255.0)[:, :, None]
    return mask
