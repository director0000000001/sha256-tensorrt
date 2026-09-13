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

**What is done to the picture (day 16, block B).** The photograph is
*prepared* before it becomes the portrait: a face taller than the
standard is scaled down to it (INTER_AREA -- the engine's crop then
resamples a picture that is already near its own 512, instead of
skipping through a 4000-px photo with a bilinear warp), and the picture
is padded with its own edge rows until the engine's crop square lies
inside it and the chin has :data:`STANDARD_ROOM_BELOW_CHIN` of a face
height under it (his four portraits had 0.23-0.27; the crop at
``CROP_VY_RATIO`` -0.06 wants 0.24 and the mask's reach under the chin a
little more). The original is kept beside it. Then the picture is *read*
(:func:`portrait_profile`): the chin's tone against the cheek's (a beard
or stubble on the portrait -- reported; the jaw shrink is a decision
about the pair, day 12), where the hair starts above the brows (a
hairline lower than the mask's forehead reach would paste the
portrait's hair over the person's forehead -- the reach is lowered to
stay on skin, with the day-13 floor above the brows), and the lips.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from sha256.avatar.engine_liveportrait import (
    _HAIRLINE_CLEARANCE,
    _OVAL_FEATHER,
    _OVAL_FOREHEAD_REACH,
    _OVAL_SHRINK,
    CROP_SCALE,
    CROP_VY_RATIO,
    FACE_POINTS,
    ChinRoom,
    _face_height,
    _forehead_geometry,
    _hairline_fraction,
    _luminance,
    _patch_mean,
    chin_room,
    crop_matrix,
)
from sha256.core.avatar import AvatarDescriptor, AvatarManifest
from sha256.core.enums import PixelFormat
from sha256.core.errors import Sha256Error
from sha256.core.types import FrameMetadata, VideoFrame
from sha256.utils.logging import get_logger

_LOG = get_logger(__name__)

PORTRAIT_FILENAME = "portrait.png"
ORIGINAL_STEM = "original"
PROFILE_FILENAME = "portrait_profile.json"
NEURAL_ENGINE_ID = "liveportrait"

# The prepared portrait (day 16): the face brought DOWN to this height when
# it is taller (never up), and the room the picture is padded to under
# the chin. 450 px is about twice what the 512 crop resamples from (2.2
# face heights into 512: 233 px of face), so the crop is a mild
# downscale rather than an upscale; 0.40 covers the crop's own 0.24 under
# the chin at CROP_VY_RATIO plus the mask's reach and drop.
STANDARD_FACE_HEIGHT_PX = 450.0
STANDARD_ROOM_BELOW_CHIN = 0.40
_PAD_MARGIN_PX = 8

# The portrait profile's own thresholds (day 16). The chin's darkness
# against the cheek's: a bare chin reads within a few percent of the
# cheek, stubble 8-20% darker, a beard more. The hairline is the engine's
# own reader (_hairline_fraction), shared with the anchor-time read.
STUBBLE_DARKNESS = 0.08
BEARD_DARKNESS = 0.20
# The mask's edge softness (erosion + half the feather, of the face
# width) plus the brow hair's own height, as the floor the forehead reach
# must keep above the brow points -- day 13's lesson at 0.65.
_BROW_HAIR_HEIGHT = 0.03  # of the face height
MIN_FOREHEAD_REACH = 0.30
PARTED_MOUTH_RATIO = 0.02

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
    # The face's points in the picture's pixels (478 x 3), for the
    # preparation and the profile; None from an inspector that has none.
    points: np.ndarray | None = field(default=None, compare=False, repr=False)

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
        points=points,
    )


# ---------------------------------------------------------------------------
# B1: the picture prepared
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Preparation:
    """What :func:`prepare_portrait` did to the picture."""

    scale: float
    padding: tuple[int, int, int, int]  # top, bottom, left, right, output pixels
    width: int
    height: int
    face_height_px: float
    room_below_chin: float
    room: ChinRoom

    @property
    def changed(self) -> bool:
        return self.scale < 1.0 or any(self.padding)

    def describe(self) -> str:
        parts: list[str] = []
        if self.scale < 1.0:
            parts.append(f"scaled to {self.scale:.2f} (face {self.face_height_px:.0f} px)")
        top, bottom, left, right = self.padding
        if any(self.padding):
            parts.append(
                f"padded {top}/{bottom}/{left}/{right} px (top/bottom/left/right) to "
                f"{self.width}x{self.height}, room below the chin now "
                f"{self.room_below_chin:.2f} of the face height"
            )
        parts.append(
            f"the crop keeps {self.room.headroom:.0f} px under the chin "
            f"(jaw drop up to {self.room.chin_drop_max:.0f} px)"
        )
        return "; ".join(parts)


def prepare_portrait(
    image: np.ndarray,
    points: np.ndarray,
    *,
    face_height: float = STANDARD_FACE_HEIGHT_PX,
    room_below: float = STANDARD_ROOM_BELOW_CHIN,
    crop_scale: float = CROP_SCALE,
    vy_ratio: float = CROP_VY_RATIO,
) -> tuple[np.ndarray, np.ndarray, Preparation]:
    """The picture as the engine wants it: the face no taller than
    ``face_height`` (scaled down with INTER_AREA, never up), and the
    picture padded with its own edge rows until the engine's crop square
    lies inside it and the chin has ``room_below`` of a face height under
    it. Returns the prepared picture, the points moved with it, and what
    was done."""
    import cv2

    pts = np.asarray(points, dtype=np.float64)[:, :2].copy()
    out = image
    height_now = _face_height(pts[:FACE_POINTS])
    scale = 1.0
    if height_now > face_height:
        scale = face_height / height_now
        size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
        out = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        pts *= scale
    face = pts[:FACE_POINTS].astype(np.float32)
    face_h = _face_height(face)
    _, to_frame = crop_matrix(face, scale=crop_scale, vy_ratio=vy_ratio)
    corners = np.array([[0, 0], [512, 0], [512, 512], [0, 512]], dtype=np.float32)
    frame = corners @ to_frame[:, :2].T + to_frame[:, 2]
    h, w = out.shape[:2]
    top = max(0, math.ceil(-float(frame[:, 1].min())) + _PAD_MARGIN_PX)
    left = max(0, math.ceil(-float(frame[:, 0].min())) + _PAD_MARGIN_PX)
    right = max(0, math.ceil(float(frame[:, 0].max()) - w) + _PAD_MARGIN_PX)
    bottom = max(0, math.ceil(float(frame[:, 1].max()) - h) + _PAD_MARGIN_PX)
    chin_need = math.ceil(room_below * face_h - (h - float(face[_CHIN, 1])))
    bottom = max(bottom, chin_need, 0)
    # A padding wanted only because of the margin is not worth a copy.
    top, left, right, bottom = (0 if v <= _PAD_MARGIN_PX else v for v in (top, left, right, bottom))
    if any((top, bottom, left, right)):
        out = cv2.copyMakeBorder(out, top, bottom, left, right, cv2.BORDER_REPLICATE)
        pts += (left, top)
    face = pts[:FACE_POINTS].astype(np.float32)
    room = chin_room(
        face, image_size=(out.shape[1], out.shape[0]), scale=crop_scale, vy_ratio=vy_ratio
    )
    prepared = Preparation(
        scale=scale,
        padding=(top, bottom, left, right),
        width=int(out.shape[1]),
        height=int(out.shape[0]),
        face_height_px=float(face_h),
        room_below_chin=float(out.shape[0] - face[_CHIN, 1]) / max(float(face_h), 1.0),
        room=room,
    )
    return out, pts, prepared


# ---------------------------------------------------------------------------
# B2: the picture read
# ---------------------------------------------------------------------------

_CHEEKS = (50, 280)
_LOWER_LIP_OUTER = 17


@dataclass(frozen=True)
class PortraitProfile:
    """What the picture itself says about the face it holds."""

    chin_darkness: float
    beard: str
    hairline: float | None
    forehead_reach: float
    reach_floor: float
    lips: str
    mouth_ratio: float

    def engine_options(self) -> dict[str, Any]:
        """The options the profile sets on a new avatar: the chin reach on
        auto, and the forehead reach on "hairline" -- the engine reads this
        portrait's hairline itself at load (the same reader as
        ``hairline`` here) and the driver's at the anchor, and joins the
        whole forehead under the lower of the two (day 17); the jaw shrink
        is not set here (the pair decides it -- day 12: 0.08 hid a bearded
        driver's jaw under a bare portrait and put a ledge under a
        stubbled portrait on a shaved one)."""
        # The chin reach is the mask's own clearance, resolved by the engine
        # ("auto", day 16 B3 -- his hand 0.04 to a tenth of a pixel).
        return {"mask_chin_reach": "auto", "mask_forehead_reach": "hairline"}

    def describe(self) -> str:
        hair = (
            "hair above the mesh's forehead"
            if self.hairline is None
            else f"hairline at {self.hairline:.2f} of the forehead"
        )
        return (
            f"{self.beard} (chin {self.chin_darkness:+.0%} against the cheek), {hair} "
            f"(mask reach {self.forehead_reach:.2f}, floor {self.reach_floor:.2f}), "
            f"lips {self.lips} ({self.mouth_ratio:.3f})"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "chin_darkness": round(self.chin_darkness, 4),
            "beard": self.beard,
            "hairline": None if self.hairline is None else round(self.hairline, 3),
            "forehead_reach": round(self.forehead_reach, 3),
            "reach_floor": round(self.reach_floor, 3),
            "lips": self.lips,
            "mouth_ratio": round(self.mouth_ratio, 4),
        }


def portrait_profile(
    image: np.ndarray, points: np.ndarray, *, default_reach: float = _OVAL_FOREHEAD_REACH
) -> PortraitProfile:
    """Read the picture: the chin's tone against the cheeks', where the hair
    starts above the brows along the face's own vertical, and the lips'
    state -- and the forehead reach that keeps the mask on skin."""
    pts = np.asarray(points, dtype=np.float32)[:FACE_POINTS, :2]
    lum = _luminance(image)
    face_h = max(_face_height(pts), 1.0)
    face_w = max(float(pts[:, 0].max() - pts[:, 0].min()), 1.0)
    # The chin: between the lower lip and the chin point, the lower half
    # of that stretch; the cheeks: MediaPipe's own cheek points.
    lip, chin = pts[_LOWER_LIP_OUTER], pts[_CHIN]
    chin_centre = lip + (chin - lip) * 0.62
    chin_value = _patch_mean(lum, chin_centre, 0.05 * face_w)
    cheek_values = [_patch_mean(lum, pts[i], 0.04 * face_w) for i in _CHEEKS]
    cheek_value = float(np.nanmean(cheek_values)) if cheek_values else math.nan
    darkness = (
        1.0 - chin_value / cheek_value
        if math.isfinite(chin_value) and math.isfinite(cheek_value) and cheek_value > 1.0
        else 0.0
    )
    beard = (
        "beard"
        if darkness >= BEARD_DARKNESS
        else "stubble"
        if darkness >= STUBBLE_DARKNESS
        else "bare"
    )

    hairline = _hairline_fraction(lum, pts)
    _, forehead_h = _forehead_geometry(pts)
    floor = (
        (_OVAL_SHRINK + _OVAL_FEATHER / 2.0) * face_w + _BROW_HAIR_HEIGHT * face_h
    ) / forehead_h
    floor = min(max(floor, MIN_FOREHEAD_REACH), default_reach)
    reach = default_reach
    if hairline is not None and hairline - _HAIRLINE_CLEARANCE < default_reach:
        reach = max(hairline - _HAIRLINE_CLEARANCE, floor)

    mouth = _ratio(pts, (_LIPS[2], _LIPS[3], _LIPS[0], _LIPS[1]))
    lips = (
        "open" if mouth > GRIN_MOUTH_RATIO else "parted" if mouth > PARTED_MOUTH_RATIO else "closed"
    )
    return PortraitProfile(
        chin_darkness=float(darkness),
        beard=beard,
        hairline=hairline,
        forehead_reach=float(reach),
        reach_floor=float(floor),
        lips=lips,
        mouth_ratio=float(mouth),
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
    prepare: bool = True,
) -> tuple[AvatarDescriptor, PortraitReport]:
    """Make ``photo`` an avatar named ``name`` in ``library``: check it,
    create the directory, write the portrait -- prepared (day 16: scaled
    to the standard, padded for the crop and the chin) with the original
    kept beside it -- its profile, and a manifest copied from ``template``
    (engine, rendering, every engine option) with the profile's own
    options over it. Returns the new descriptor and the report, its
    warnings and description carrying what was done; raises
    :class:`AvatarImportError` on a hard failure with nothing written."""
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
        if image is None:
            msg = f"{photo.name} could not be read as an image"
            raise AvatarImportError(msg)
        profile: PortraitProfile | None = None
        extra_options: dict[str, Any] = {}
        assets: list[dict[str, Any]] = [
            {"path": PORTRAIT_FILENAME, "role": "source_portrait", "optional": False}
        ]
        if prepare and report.points is not None:
            prepared, points, preparation = prepare_portrait(image, report.points)
            profile = portrait_profile(prepared, points)
            extra_options = profile.engine_options()
            notes = [w for w in report.warnings if "room below the chin" not in w]
            notes.append("prepared: " + preparation.describe())
            notes.append("portrait: " + profile.describe())
            report = replace(
                report,
                width=preparation.width,
                height=preparation.height,
                face_height_px=preparation.face_height_px,
                room_below_chin=preparation.room_below_chin,
                warnings=tuple(notes),
                points=points,
            )
            original = directory / f"{ORIGINAL_STEM}{photo.suffix.lower() or '.png'}"
            shutil.copyfile(photo, original)
            assets.append({"path": original.name, "role": "source_original", "optional": True})
            (directory / PROFILE_FILENAME).write_text(
                json.dumps(
                    {
                        "profile": profile.as_dict(),
                        "preparation": {
                            "scale": round(preparation.scale, 4),
                            "padding": list(preparation.padding),
                            "size": [preparation.width, preparation.height],
                            "face_height_px": round(preparation.face_height_px, 1),
                            "room_below_chin": round(preparation.room_below_chin, 3),
                            "crop_headroom_px": round(preparation.room.headroom, 1),
                            "chin_drop_max_px": round(preparation.room.chin_drop_max, 1),
                        },
                        "engine_options": extra_options,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            image = prepared
        if not cv2.imwrite(str(directory / PORTRAIT_FILENAME), image):
            msg = f"{photo.name} could not be written as {PORTRAIT_FILENAME}"
            raise AvatarImportError(msg)
        manifest = _manifest_for(
            avatar_id, name.strip(), template, report, assets=assets, extra_options=extra_options
        )
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
    avatar_id: str,
    name: str,
    template: AvatarManifest | None,
    report: PortraitReport,
    *,
    assets: list[dict[str, Any]] | None = None,
    extra_options: dict[str, Any] | None = None,
) -> AvatarManifest:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    options = dict(template.engine_options) if template is not None else {}
    options.update(extra_options or {})
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
        "assets": assets
        or [{"path": PORTRAIT_FILENAME, "role": "source_portrait", "optional": False}],
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
        "engine_options": options,
        "fictional": template.fictional if template is not None else True,
    }
    return AvatarManifest.model_validate(data)
