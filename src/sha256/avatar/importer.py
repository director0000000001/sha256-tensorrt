"""Adding an avatar from a photograph, inside the application (day 15).

His ask: new faces are added from the app, by him alone -- a photograph in,
an avatar directory out. The directory is what the library already
understands (``avatars/README.md``): ``metadata.json`` plus the portrait,
the manifest copied from an existing neural avatar so every engine option
measured on this machine carries over, and the photograph re-encoded as
``portrait.png`` beside it.

**What is checked before anything is written.** Fifteen days of measurement
say which photographs work: one face, frontal (the anchor and the crop are
built for a frontal portrait), eyes open, the mouth shut or barely parted
(a portrait that grins grins at rest -- day 6's portrait 2), the face
large enough to feed a 512 crop, and room below the chin -- the crop needs
about a third of a face height under it, or the paste mask clips into a
hard line at the picture's edge (day 14's wall). The hard failures refuse
the import; the soft ones are reported and the import goes ahead, since
his eye on the result is the last word.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from sha256.core.avatar import AvatarDescriptor, AvatarManifest
from sha256.core.enums import PixelFormat
from sha256.core.errors import Sha256Error
from sha256.core.types import FrameMetadata, VideoFrame
from sha256.utils.logging import get_logger

_LOG = get_logger(__name__)

PORTRAIT_FILENAME = "portrait.png"
NEURAL_ENGINE_ID = "liveportrait"

# MediaPipe indices the checks read (see engine_liveportrait's own table).
_FOREHEAD_TOP, _CHIN = 10, 152
_L_EYE = (33, 133, 159, 145)
_R_EYE = (362, 263, 386, 374)
_LIPS = (13, 14, 61, 291)

# Limits, from the days that found them.
MAX_FRONTAL_DEGREES = 12.0  # hard: the anchor is a frontal face
SOFT_FRONTAL_DEGREES = 5.0
MIN_FACE_HEIGHT_PX = 200.0  # hard: the crop is 2.2 face heights into 512
SOFT_FACE_HEIGHT_PX = 350.0
MIN_EYE_RATIO = 0.15  # hard: shut eyes cannot be the rest
GRIN_MOUTH_RATIO = 0.08  # soft: a parted-lips portrait is fine, a grin is not
MIN_ROOM_BELOW_CHIN = 0.35  # soft: of the face height (day 14)


class AvatarImportError(Sha256Error):
    """The photograph cannot be an avatar, and why."""


@dataclass(frozen=True)
class PortraitReport:
    """What one look at the photograph found."""

    width: int
    height: int
    face_found: bool
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    face_height_px: float = 0.0
    room_below_chin: float = 0.0
    eye_ratio: float = 0.0
    mouth_ratio: float = 0.0
    errors: tuple[str, ...] = field(default=())
    warnings: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return not self.errors

    def describe(self) -> str:
        if not self.face_found:
            return "no face found"
        lines = [
            f"face {self.face_height_px:.0f} px tall in a {self.width}x{self.height} picture, "
            f"pose {self.pitch:+.1f} / {self.yaw:+.1f} / {self.roll:+.1f} deg "
            "(pitch / yaw / roll), "
            f"eyes {self.eye_ratio:.2f}, mouth {self.mouth_ratio:.3f}, "
            f"room below the chin {self.room_below_chin:.2f} of the face height",
        ]
        lines.extend(f"error: {e}" for e in self.errors)
        lines.extend(f"note: {w}" for w in self.warnings)
        return "\n".join(lines)


class Inspector(Protocol):
    def __call__(self, photo: Path, model_path: Path) -> PortraitReport: ...


def _ratio(points: np.ndarray, indices: tuple[int, int, int, int]) -> float:
    outer, inner, upper, lower = indices
    width = float(np.linalg.norm(points[inner, :2] - points[outer, :2])) or 1e-6
    return float(np.linalg.norm(points[upper, :2] - points[lower, :2])) / width


def inspect_portrait(photo: Path, model_path: Path) -> PortraitReport:
    """Look at the photograph the way the engine will: MediaPipe in image
    mode, the same landmarks the engine crops and masks by."""
    import cv2

    from sha256.tracking.mediapipe_tracker import MediaPipeFaceTracker

    image = cv2.imread(str(photo), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"{photo.name} could not be read as an image"
        raise AvatarImportError(msg)
    height, width = image.shape[:2]
    tracker = MediaPipeFaceTracker(
        model_path=model_path, smoothing=0.0, mouth_smoothing=0.0, running_mode="image"
    )
    tracker.start()
    try:
        result = tracker.process(
            VideoFrame(
                data=np.ascontiguousarray(image),
                pixel_format=PixelFormat.BGR24,
                metadata=FrameMetadata(frame_id=0, capture_timestamp=0.0),
            )
        )
    finally:
        tracker.stop()
    if not result.face_detected or result.landmarks is None:
        return PortraitReport(
            width=width, height=height, face_found=False, errors=("no face found in the picture",)
        )
    points = result.landmarks.points.copy()
    points[:, 0] *= width
    points[:, 1] *= height
    face_height = float(points[_CHIN, 1] - points[_FOREHEAD_TOP, 1])
    room = float(height - points[_CHIN, 1]) / max(face_height, 1.0)
    eye = (_ratio(points, _L_EYE) + _ratio(points, _R_EYE)) / 2.0
    lip_gap = float(np.linalg.norm(points[_LIPS[0], :2] - points[_LIPS[1], :2]))
    mouth_width = float(np.linalg.norm(points[_LIPS[3], :2] - points[_LIPS[2], :2])) or 1e-6
    mouth = lip_gap / mouth_width
    pose = result.head_pose
    yaw, pitch, roll = (pose.yaw, pose.pitch, pose.roll) if pose is not None else (0.0, 0.0, 0.0)

    errors: list[str] = []
    warnings: list[str] = []
    if abs(yaw) > MAX_FRONTAL_DEGREES or abs(pitch) > MAX_FRONTAL_DEGREES:
        errors.append(
            f"the face must look straight at the camera (turned {yaw:+.0f} deg, "
            f"tilted {pitch:+.0f} deg; up to {MAX_FRONTAL_DEGREES:.0f} is fine)"
        )
    elif abs(yaw) > SOFT_FRONTAL_DEGREES or abs(pitch) > SOFT_FRONTAL_DEGREES:
        warnings.append("the face is a little off straight; a frontal photograph centres better")
    if face_height < MIN_FACE_HEIGHT_PX:
        errors.append(
            f"the face is too small ({face_height:.0f} px tall; {MIN_FACE_HEIGHT_PX:.0f} at least, "
            f"{SOFT_FACE_HEIGHT_PX:.0f} or more is good)"
        )
    elif face_height < SOFT_FACE_HEIGHT_PX:
        warnings.append(
            f"the face is small ({face_height:.0f} px tall); a closer photograph is sharper"
        )
    if eye < MIN_EYE_RATIO:
        errors.append("the eyes must be open")
    if mouth > GRIN_MOUTH_RATIO:
        warnings.append(
            "the mouth is open; an avatar made from it will hold that at rest -- "
            "closed or barely parted lips work best"
        )
    if room < MIN_ROOM_BELOW_CHIN:
        warnings.append(
            f"little room below the chin ({room:.2f} of the face height; about "
            f"{MIN_ROOM_BELOW_CHIN:.2f} is needed) -- an open jaw may hit the picture's edge"
        )
    return PortraitReport(
        width=width,
        height=height,
        face_found=True,
        yaw=float(yaw),
        pitch=float(pitch),
        roll=float(roll),
        face_height_px=face_height,
        room_below_chin=room,
        eye_ratio=float(eye),
        mouth_ratio=float(mouth),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def slug_for(name: str, library: Path) -> str:
    """An avatar id from a display name: lowercase, letters, digits and
    underscores, at least three characters, unique in ``library``."""
    base = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "avatar"
    if len(base) < 3:
        base = f"{base}_avatar"
    base = base[:56]
    candidate = base
    counter = 2
    while (library / candidate).exists():
        candidate = f"{base}_{counter}"
        counter += 1
    return candidate


def neural_template(
    descriptors: Iterable[AvatarDescriptor], active_id: str | None = None
) -> AvatarManifest | None:
    """The manifest a new avatar copies its engine and options from: the
    active neural avatar if there is one, else the first neural avatar in
    the library, else ``None`` (the caller falls back to bare defaults)."""
    neural = [d for d in descriptors if d.manifest.engine.engine_id == NEURAL_ENGINE_ID]
    if not neural:
        return None
    for descriptor in neural:
        if descriptor.id == active_id:
            return descriptor.manifest
    return neural[0].manifest


def import_portrait(
    library: Path,
    photo: Path,
    *,
    name: str,
    model_path: Path,
    template: AvatarManifest | None = None,
    inspect: Inspector = inspect_portrait,
) -> tuple[AvatarDescriptor, PortraitReport]:
    """Make ``photo`` an avatar named ``name`` in ``library``: check it,
    create the directory, write the portrait and a manifest copied from
    ``template`` (engine, rendering, every engine option). Returns the new
    descriptor and the report; raises :class:`AvatarImportError` on a hard
    failure with nothing written."""
    import cv2

    photo = Path(photo)
    if not photo.is_file():
        msg = f"{photo} is not a file"
        raise AvatarImportError(msg)
    if not name.strip():
        msg = "the avatar needs a name"
        raise AvatarImportError(msg)
    report = inspect(photo, model_path)
    if not report.ok:
        msg = "the photograph cannot be an avatar:\n" + "\n".join(report.errors)
        raise AvatarImportError(msg)

    library = Path(library)
    library.mkdir(parents=True, exist_ok=True)
    avatar_id = slug_for(name, library)
    directory = library / avatar_id
    directory.mkdir()
    try:
        image = cv2.imread(str(photo), cv2.IMREAD_COLOR)
        if image is None or not cv2.imwrite(str(directory / PORTRAIT_FILENAME), image):
            msg = f"{photo.name} could not be written as {PORTRAIT_FILENAME}"
            raise AvatarImportError(msg)
        manifest = _manifest_for(avatar_id, name.strip(), template, report)
        (directory / "metadata.json").write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        # Nothing half-made in the library: one bad import must not leave a
        # directory the scan then reports as broken every start.
        for child in directory.iterdir():
            child.unlink()
        directory.rmdir()
        raise
    _LOG.info("Avatar %s added from %s (%s)", avatar_id, photo.name, report.describe())
    return AvatarDescriptor(manifest=manifest, root=directory), report


def _manifest_for(
    avatar_id: str, name: str, template: AvatarManifest | None, report: PortraitReport
) -> AvatarManifest:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    data: dict[str, Any] = {
        "manifest_version": 1,
        "id": avatar_id,
        "name": name,
        "version": "1.0.0",
        "description": (
            f"Added from the application on {stamp}: {report.describe().splitlines()[0]}."
        ),
        "tags": ["owner-added", "neural"],
        "preview": None,
        "assets": [{"path": PORTRAIT_FILENAME, "role": "source_portrait", "optional": False}],
        "engine": (
            template.engine.model_dump(mode="json")
            if template is not None
            else {
                "engine_id": NEURAL_ENGINE_ID,
                "estimated_vram_mb": 4500,
                "device_preference": "cuda",
            }
        ),
        "rendering": (
            template.rendering.model_dump(mode="json")
            if template is not None
            else {"output_width": 1280, "output_height": 720}
        ),
        "engine_options": dict(template.engine_options) if template is not None else {},
        "fictional": template.fictional if template is not None else True,
    }
    return AvatarManifest.model_validate(data)
