"""Pick avatar source photographs out of a video clip.

A clip of one face is worth more than any number of separately generated stills,
because the identity cannot drift between frames of the same take. What it needs
is a way to find, among a few hundred frames, the best example of each thing the
engine knows how to ask for.

**The engine's own selector does the classifying.** ``_source_weights`` already
answers "how much of the open-mouth photograph do I want for this frame", from
blendshapes and head pose. Run over a video it answers "which frame is the best
example of an open mouth" — the same question from the other end. Using one
function for both means the library can never be classified by rules the
renderer disagrees with.

Frames are rejected before they are scored, on the three things that make a
frame unusable no matter what expression it caught:

*   **no face**, or one too small to carry detail;
*   **motion blur** — a frame taken mid-movement is soft, and softness is the
    one flaw the warp cannot recover from;
*   **identity drift** — measured against the base portrait, in case the
    generator lost the person partway through.

Usage::

    python scripts/extract_avatar_frames.py clip.mp4 --out extracted/
    python scripts/extract_avatar_frames.py clip.mp4 --out extracted/ --min-face 420
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sha256.avatar.engine_warp import _VARIANT_ROLES, _source_weights  # noqa: E402
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FaceTrackingResult, FrameMetadata, VideoFrame  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)

# The role every avatar must have, and the one a clip cannot supply: it is the
# reference every other source is registered against.
BASE_ROLE = "source_portrait"

# A frame has to be at least this good an example of a role before it is taken.
# Below it the "best" frame is only the least bad one, and a photograph that
# barely shows the expression is worse than not having it — the engine would
# blend it in whenever the expression appeared.
MIN_ROLE_SCORE = 0.45

# Sharpness is the variance of the Laplacian over the face. It has no absolute
# meaning across cameras, so the threshold is relative: a frame must reach this
# fraction of the sharpest frame in its own clip.
MIN_SHARPNESS_FRACTION = 0.55

# How far a structural proportion may move from the base portrait before the
# frame is treated as a different face. Matches the photo checker.
IDENTITY_TOLERANCE = 0.08


@dataclass(frozen=True)
class Candidate:
    index: int
    image: np.ndarray
    tracking: FaceTrackingResult
    face_width: float
    sharpness: float
    identity_drift: float


def identity_ratios(points: np.ndarray, width: int, height: int) -> dict[str, float]:
    """Proportions that do not change between photographs of one person.

    Measured against the eye span rather than face height: the chin drops when
    the mouth opens, and half these frames are mid-sentence.
    """
    scaled = points[:, :2] * (width, height)

    def distance(a: int, b: int) -> float:
        return float(np.linalg.norm(scaled[a] - scaled[b]))

    eye_span = max(distance(33, 263), 1e-6)
    return {
        "eye_spacing": distance(133, 362) / eye_span,
        "face_width": distance(234, 454) / eye_span,
        "nose_width": distance(98, 327) / eye_span,
    }


def sharpness_of(image: np.ndarray, points: np.ndarray) -> float:
    """Laplacian variance over the face only.

    Over the whole frame a sharp background would hide a soft face, which is
    exactly the frame this is meant to reject.
    """
    import cv2

    height, width = image.shape[:2]
    x0 = max(int(points[:, 0].min() * width), 0)
    x1 = min(int(points[:, 0].max() * width), width)
    y0 = max(int(points[:, 1].min() * height), 0)
    y1 = min(int(points[:, 1].max() * height), height)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0

    face = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(face, cv2.CV_64F).var())


def read_candidates(
    video: Path, tracker: MediaPipeFaceTracker, base_ratios: dict[str, float]
) -> tuple[list[Candidate], dict[str, int]]:
    """Every frame with a usable face, and a tally of why the others went."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        msg = f"could not open {video}"
        raise SystemExit(msg)

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    candidates: list[Candidate] = []
    rejected = {"no face": 0}
    index = 0

    while True:
        ok, image = capture.read()
        if not ok:
            break

        result = tracker.process(
            VideoFrame(
                data=np.ascontiguousarray(image),
                pixel_format=PixelFormat.BGR24,
                metadata=FrameMetadata(frame_id=index, capture_timestamp=index / fps),
            )
        )
        index += 1

        if not result.face_detected or result.landmarks is None:
            rejected["no face"] += 1
            continue

        points = result.landmarks.points
        height, width = image.shape[:2]
        ratios = identity_ratios(points, width, height)
        drift = max(
            abs(value - base_ratios[name]) / base_ratios[name] for name, value in ratios.items()
        )

        candidates.append(
            Candidate(
                index=index - 1,
                image=image,
                tracking=result,
                face_width=float(points[:, 0].max() - points[:, 0].min()) * width,
                sharpness=sharpness_of(image, points),
                identity_drift=drift,
            )
        )

    capture.release()
    return candidates, rejected


def usable(
    candidates: list[Candidate], min_face: float, rejected: dict[str, int]
) -> list[Candidate]:
    """Drop the frames no expression could redeem."""
    if not candidates:
        return []

    sharpest = max(c.sharpness for c in candidates)
    floor = sharpest * MIN_SHARPNESS_FRACTION

    kept: list[Candidate] = []
    for candidate in candidates:
        if candidate.face_width < min_face:
            rejected["face too small"] = rejected.get("face too small", 0) + 1
        elif candidate.sharpness < floor:
            rejected["motion blurred"] = rejected.get("motion blurred", 0) + 1
        elif candidate.identity_drift > IDENTITY_TOLERANCE:
            rejected["identity drifted"] = rejected.get("identity drifted", 0) + 1
        else:
            kept.append(candidate)
    return kept


def best_per_role(candidates: list[Candidate]) -> dict[str, tuple[Candidate, float]]:
    """The clearest example of each role the clip contains.

    Ties on score are broken by sharpness. Two frames a fifth of a second apart
    show the same expression, and there is no reason to take the softer one.
    """
    chosen: dict[str, tuple[Candidate, float]] = {}

    for candidate in candidates:
        for role, score in _source_weights(candidate.tracking).items():
            if score < MIN_ROLE_SCORE:
                continue
            current = chosen.get(role)
            if current is None:
                chosen[role] = (candidate, score)
                continue

            held, held_score = current
            better_score = score > held_score + 0.05
            similar_but_sharper = (
                abs(score - held_score) <= 0.05 and candidate.sharpness > held.sharpness
            )
            if better_score or similar_but_sharper:
                chosen[role] = (candidate, score)

    return chosen


def main(argv: list[str] | None = None) -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="The clip to pull frames from.")
    parser.add_argument(
        "--base",
        type=Path,
        default=REPO_ROOT / "avatars" / "my_avatar" / "portrait.png",
        help="The portrait every frame is checked against.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Where to write the frames.")
    parser.add_argument(
        "--min-face",
        type=float,
        default=380.0,
        help="Reject frames whose face is narrower than this, in pixels.",
    )
    args = parser.parse_args(argv)

    model_path = REPO_ROOT / "models" / DEFAULT_MODEL_RELATIVE_PATH
    if not model_path.is_file():
        print(f"Model bundle missing: {model_path}")
        print("Fetch it with: python scripts/fetch_face_landmarker.py")
        return 1

    base_image = cv2.imread(str(args.base), cv2.IMREAD_COLOR)
    if base_image is None:
        print(f"Could not read the base portrait: {args.base}")
        return 1

    tracker = MediaPipeFaceTracker(model_path=model_path)
    tracker.start()
    try:
        base = tracker.process(
            VideoFrame(
                data=np.ascontiguousarray(base_image),
                pixel_format=PixelFormat.BGR24,
                metadata=FrameMetadata(frame_id=0, capture_timestamp=0.0),
            )
        )
        if not base.face_detected or base.landmarks is None:
            print(f"No face found in the base portrait {args.base}.")
            return 1

        base_ratios = identity_ratios(
            base.landmarks.points, base_image.shape[1], base_image.shape[0]
        )
        base_face = (
            float(base.landmarks.points[:, 0].max() - base.landmarks.points[:, 0].min())
            * base_image.shape[1]
        )
    finally:
        tracker.stop()

    # A fresh tracker for the clip: the base portrait is an unrelated still, and
    # the tracker carries a prior between frames.
    tracker = MediaPipeFaceTracker(model_path=model_path)
    tracker.start()
    try:
        candidates, rejected = read_candidates(args.video, tracker, base_ratios)
    finally:
        tracker.stop()

    total = len(candidates) + rejected["no face"]
    print(f"{args.video.name}: {total} frames")
    print(f"base portrait face width: {base_face:.0f} px")
    print()

    kept = usable(candidates, args.min_face, rejected)
    for reason, count in sorted(rejected.items()):
        if count:
            print(f"  rejected, {reason:<18} {count}")
    print(f"  usable                    {len(kept)}")

    if not kept:
        print("\nNothing usable. Lower --min-face, or generate a sharper clip.")
        return 1

    widths = [c.face_width for c in kept]
    print(
        f"\n  face width: min {min(widths):.0f}  median {float(np.median(widths)):.0f}  "
        f"max {max(widths):.0f} px"
    )

    chosen = best_per_role(kept)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'role':<26}{'frame':>7}{'score':>8}{'face px':>9}")
    for role in _VARIANT_ROLES:
        picked = chosen.get(role)
        if picked is None:
            print(f"{role:<26}{'—':>7}{'':>8}{'':>9}   not in this clip")
            continue
        candidate, score = picked
        name = role.replace("source_", "portrait_") + ".png"
        cv2.imwrite(str(args.out / name), candidate.image)
        print(f"{role:<26}{candidate.index:>7}{score:>8.2f}{candidate.face_width:>9.0f}")

    missing = [role for role in _VARIANT_ROLES if role not in chosen]
    print(f"\n{len(chosen)} of {len(_VARIANT_ROLES)} roles covered by this clip.")
    if missing:
        print("Not covered — another clip, or the existing photographs, must supply:")
        print("  " + ", ".join(r.replace("source_", "") for r in missing))

    print(f"\nWritten to {args.out}")
    print("Check them before use:")
    print(f"  python scripts/check_avatar_photos.py {args.base} {args.out}/*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
