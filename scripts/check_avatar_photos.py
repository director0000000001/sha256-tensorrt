"""Check generated avatar photographs against the base portrait.

The avatar photo set only works if every photograph is recognisably the *same
person*, framed the same way, under the same light. An image generator asked for
"the same person, smiling" will often return a convincing photograph of someone
slightly different, and that is invisible until the engine switches source
mid-call and the avatar's face changes.

This measures the things that would make a variant unusable, before it is used:

*   **Identity** — proportions that do not change between two photographs of one
    person: eye spacing against face width, nose length against face height,
    mouth and jaw width. A generator that drifted will move these.
*   **Framing** — how large the face is in the frame and where it sits. A
    variant shot closer makes the avatar jump when the engine switches to it.
*   **Pose** — measured yaw, pitch, roll. This is also how the turned photos are
    identified, so left and right do not have to be got right by hand.
*   **Lighting** — mean skin colour in the cheeks. A variant lit differently
    reads as a flicker.

**What this is not.** These are geometric ratios, not face recognition. They
catch a generator that reshaped the face; they cannot certify that two
photographs are the same person. Under a large head turn, foreshortening changes
the horizontal ratios legitimately — the tool says so where it applies.

Usage::

    python scripts/check_avatar_photos.py base.png variant1.png variant2.png
    python scripts/check_avatar_photos.py avatars/my_avatar/portrait.png new/*.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FrameMetadata, VideoFrame  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)

# Canonical MediaPipe indices. Named so the measurements below can be audited
# against the published topology rather than trusted.
EYE_OUTER_LEFT, EYE_OUTER_RIGHT = 33, 263
EYE_INNER_LEFT, EYE_INNER_RIGHT = 133, 362
FOREHEAD_TOP, CHIN_BOTTOM = 10, 152
NOSE_BRIDGE, NOSE_TIP = 168, 4
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
JAW_LEFT, JAW_RIGHT = 234, 454
CHEEK_LEFT, CHEEK_RIGHT = 50, 280

# Ratios beyond this fraction of the base value are called out. Chosen so that
# ordinary landmark noise between two photographs stays quiet: repeated
# detections of one image move these by well under a percent.
IDENTITY_TOLERANCE = 0.06

# Below this the face carries too few pixels to warp without going soft. The
# base portrait puts about 450 px across the face; half of that still holds up
# on a 720p call.
MINIMUM_FACE_PIXELS = 220

# Blendshape total above which the mouth counts as open or smiling, and mouth
# measurements stop being comparable with the neutral base.
EXPRESSION_THRESHOLD = 0.20

# Above this yaw the horizontal ratios are compressed by foreshortening rather
# than by the generator changing the face, and saying otherwise would be wrong.
YAW_FORESHORTENING_DEGREES = 12.0


@dataclass(frozen=True)
class PhotoReport:
    path: Path
    detected: bool
    width: int = 0
    height: int = 0
    identity: dict[str, float] | None = None
    framing: dict[str, float] | None = None
    pose: dict[str, float] | None = None
    skin: tuple[float, float, float] | None = None
    expression: float = 0.0
    """How open or smiling the mouth is, from the tracker's own blendshapes."""


def measure(path: Path, tracker: MediaPipeFaceTracker, index: int) -> PhotoReport:
    """Everything measurable about one photograph."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return PhotoReport(path=path, detected=False)

    height, width = image.shape[:2]
    result = tracker.process(
        VideoFrame(
            data=np.ascontiguousarray(image),
            pixel_format=PixelFormat.BGR24,
            # Timestamps must keep increasing across photographs, which is why
            # the caller's index is threaded through rather than reset here.
            metadata=FrameMetadata(frame_id=index, capture_timestamp=float(index)),
        )
    )
    if not result.face_detected or result.landmarks is None:
        return PhotoReport(path=path, detected=False, width=width, height=height)

    points = result.landmarks.points.copy()
    points[:, 0] *= width
    points[:, 1] *= height

    def distance(a: int, b: int) -> float:
        return float(np.linalg.norm(points[a, :2] - points[b, :2]))

    face_width = max(distance(JAW_LEFT, JAW_RIGHT), 1e-6)
    face_height = max(distance(FOREHEAD_TOP, CHIN_BOTTOM), 1e-6)

    identity = {
        "eye_spacing / face_width": distance(EYE_INNER_LEFT, EYE_INNER_RIGHT) / face_width,
        "eye_span / face_width": distance(EYE_OUTER_LEFT, EYE_OUTER_RIGHT) / face_width,
        "mouth_width / face_width": distance(MOUTH_LEFT, MOUTH_RIGHT) / face_width,
        "nose_length / face_height": distance(NOSE_BRIDGE, NOSE_TIP) / face_height,
        "face_width / face_height": face_width / face_height,
    }

    framing = {
        "face_width / image_width": face_width / width,
        "face_centre_x": float(points[:, 0].mean()) / width,
        "face_centre_y": float(points[:, 1].mean()) / height,
    }

    pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    if result.head_pose is not None:
        pose = {
            "yaw": result.head_pose.yaw,
            "pitch": result.head_pose.pitch,
            "roll": result.head_pose.roll,
        }

    shapes = result.blendshapes
    expression = 0.0
    if shapes is not None:
        expression = max(
            shapes.get("jawOpen"),
            shapes.get("mouthSmileLeft"),
            shapes.get("mouthSmileRight"),
        )

    return PhotoReport(
        path=path,
        detected=True,
        width=width,
        height=height,
        identity=identity,
        framing=framing,
        pose=pose,
        skin=_skin_colour(image, points),
        expression=expression,
    )


def _skin_colour(image: np.ndarray, points: np.ndarray) -> tuple[float, float, float]:
    """Mean colour of both cheeks — a stand-in for how the face is lit."""
    height, width = image.shape[:2]
    patch = max(int(np.linalg.norm(points[JAW_LEFT, :2] - points[JAW_RIGHT, :2]) * 0.06), 3)

    samples: list[np.ndarray] = []
    for index in (CHEEK_LEFT, CHEEK_RIGHT):
        x, y = int(points[index, 0]), int(points[index, 1])
        x0, y0 = max(x - patch, 0), max(y - patch, 0)
        x1, y1 = min(x + patch, width), min(y + patch, height)
        if x1 > x0 and y1 > y0:
            samples.append(image[y0:y1, x0:x1].reshape(-1, 3))

    if not samples:
        return (0.0, 0.0, 0.0)
    mean = np.concatenate(samples).mean(axis=0)
    return (float(mean[2]), float(mean[1]), float(mean[0]))  # as R, G, B


def compare(base: PhotoReport, variant: PhotoReport) -> int:
    """Print one variant against the base. Returns the number of concerns."""
    print(f"\n{'=' * 72}")
    print(f"{variant.path.name}")
    print("=" * 72)

    if not variant.detected:
        print("  NO FACE FOUND — unusable. Needs a clear, single, front-lit face.")
        return 1

    assert variant.identity and variant.framing and variant.pose and variant.skin
    assert base.identity and base.framing and base.skin

    concerns = 0
    yaw = variant.pose["yaw"]
    turned = abs(yaw) >= YAW_FORESHORTENING_DEGREES

    print(f"  size   : {variant.width}x{variant.height}", end="")
    if (variant.width, variant.height) != (base.width, base.height):
        print(f"   (base is {base.width}x{base.height})")
    else:
        print()

    print(
        f"  pose   : yaw {yaw:+.1f}  pitch {variant.pose['pitch']:+.1f}  "
        f"roll {variant.pose['roll']:+.1f}"
        + (f"   -> turned to the {'left' if yaw > 0 else 'right'}" if turned else "")
    )

    expressive = variant.expression > EXPRESSION_THRESHOLD
    if expressive:
        print(f"  mouth  : open or smiling ({variant.expression:.2f}) — mouth ratios excused")

    print("\n  identity")
    for name, value in variant.identity.items():
        reference = base.identity[name]
        drift = (value - reference) / reference if reference else 0.0

        # Two honest reasons a ratio moves without the face having changed:
        # a turned head foreshortens everything the turn compresses, and an
        # open or smiling mouth is *supposed* to be a different shape. Flagging
        # either would train the reader to ignore the flags.
        excuse = ""
        if turned and ("width" in name or "spacing" in name or "span" in name or "nose" in name):
            excuse = "foreshortened by the turn"
        elif expressive and "mouth" in name:
            excuse = "the expression, which is the point of this photo"

        flag = ""
        if abs(drift) > IDENTITY_TOLERANCE:
            if excuse:
                flag = f"  (expected: {excuse})"
            else:
                flag = "  <-- DRIFTED"
                concerns += 1

        print(f"    {name:<30} {value:6.3f}  vs {reference:6.3f}  {drift:+6.1%}{flag}")

    # Framing is reported, not judged. The engine finds landmarks in each source
    # separately and warps from those, so a variant framed differently costs
    # nothing — this was flagged as a fault in the first version of this tool
    # and it was wrong. What does matter is having enough pixels on the face.
    print("\n  framing (informational — the engine warps from landmarks, not from the frame)")
    for name, value in variant.framing.items():
        print(f"    {name:<30} {value:6.3f}  vs {base.framing[name]:6.3f}")

    face_pixels = variant.framing["face_width / image_width"] * variant.width
    base_pixels = base.framing["face_width / image_width"] * base.width
    detail = face_pixels / base_pixels if base_pixels else 1.0
    print(f"    {'face width in pixels':<30} {face_pixels:6.0f}  vs {base_pixels:6.0f}", end="")
    if face_pixels < MINIMUM_FACE_PIXELS:
        print(f"  <-- TOO SMALL (want {MINIMUM_FACE_PIXELS}+)")
        concerns += 1
    elif detail < 0.75:
        print(f"  ({detail:.0%} of the base's detail — softer, still usable)")
    else:
        print()

    difference = max(abs(a - b) for a, b in zip(variant.skin, base.skin, strict=True))
    flag = "  <-- LIT DIFFERENTLY" if difference > 25.0 else ""
    print(
        f"\n  skin   : RGB({variant.skin[0]:.0f}, {variant.skin[1]:.0f}, "
        f"{variant.skin[2]:.0f})  vs base RGB({base.skin[0]:.0f}, "
        f"{base.skin[1]:.0f}, {base.skin[2]:.0f}){flag}"
    )
    if difference > 25.0:
        concerns += 1

    print()
    if concerns == 0:
        print("  VERDICT: usable.")
    else:
        print(f"  VERDICT: {concerns} concern(s) — regenerate, changing only what is flagged.")
    return concerns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="The reference portrait.")
    parser.add_argument("variants", type=Path, nargs="*", help="Generated variants to check.")
    parser.add_argument("--model", type=str, default="", help="Override the model bundle path.")
    args = parser.parse_args(argv)

    model_path = (
        Path(args.model) if args.model else REPO_ROOT / "models" / DEFAULT_MODEL_RELATIVE_PATH
    )
    if not model_path.is_file():
        print(f"Model bundle missing: {model_path}")
        print("Fetch it with: python scripts/fetch_face_landmarker.py")
        return 1

    tracker = MediaPipeFaceTracker(model_path=model_path)
    tracker.start()
    try:
        base = measure(args.base, tracker, index=0)
        if not base.detected:
            print(f"No face found in the base photo {args.base} — nothing to compare against.")
            return 1

        print(f"base   : {args.base.name}  ({base.width}x{base.height})")
        assert base.pose is not None
        print(
            f"         yaw {base.pose['yaw']:+.1f}  pitch {base.pose['pitch']:+.1f}  "
            f"roll {base.pose['roll']:+.1f}"
        )

        if not args.variants:
            print("\nNo variants given. Pass the generated photos to compare them.")
            return 0

        concerns = 0
        for index, path in enumerate(args.variants, start=1):
            concerns += compare(base, measure(path, tracker, index=index))
    finally:
        tracker.stop()

    print(f"\n{'=' * 72}")
    if concerns == 0:
        print("All variants usable.")
    else:
        print(f"{concerns} concern(s) across the set.")
    print("Reminder: these are geometric proxies, not face recognition. They catch a")
    print("generator that reshaped the face; they cannot prove two photos are one person.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
