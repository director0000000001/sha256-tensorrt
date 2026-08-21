"""Build a ladder of the smile forming, out of dumped frames, and install it.

The owner's own proposal, after watching a slight smile snap straight to a
photograph of a wide one: *"we already have the clip where he laughs — ten
photographs from where the smile starts through to the full smile."*

They are right, and it is the turn ladder's argument on another axis. One smile
photograph faded in over a resting mouth fails twice at once:

*   while it is faint there are two mouths on screen, the resting one and the
    smiling one, and the second is a ghost over the first;
*   and whatever weight it is at, the *shape* it draws is the shape it was
    photographed at. A small smile becomes a wide smile at low opacity rather
    than a small smile — the lips never stretch, the teeth simply appear.

A row of photographs taken while the smile forms answers both, because the
nearest one is always a shape this face actually made.

The axis is :func:`sha256.avatar.engine_warp._smiling`, the engine's own
measure, so the value a rung is filed under is the value it will be chosen by.
Frames are grouped into bands of it and the sharpest in each band wins.

Usage::

    python scripts/build_smile_ladder.py avatars/my_avatar/frames/video-09
    python scripts/build_smile_ladder.py .../video-09 .../video-10 --install
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from extract_avatar_frames import (  # noqa: E402
    Candidate,
    crop_around_face,
    sharpness_of,
    timestamp_of,
)

from sha256.avatar.engine_warp import (  # noqa: E402
    _SMILE_LIP_CLOSED,
    _SMILE_OPEN_SPAN,
    _SMILE_WIDEN_SPAN,
    _lip_gap,
    _lip_parting,
    _smiling,
)
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FrameMetadata, VideoFrame  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)

# One rung per this much of the axis, which runs 0 at rest to about 2 at a
# laugh — so a fifth gives the ten photographs the owner asked for, and it is
# fine enough that the step from one to the next is a smile moving rather than
# a smile changing.
DEFAULT_STEP = 0.2

# Below this a face is not smiling, it is resting, and the portrait is already
# the best photograph of that.
DEFAULT_MIN_SMILE = 0.05

# A rung has to be a *frontal* photograph: the mouth is being cut out of it and
# placed on the avatar's own frontal layout, so a head turned while smiling
# would bring the wrong mouth shape and the wrong position at once.
DEFAULT_MAX_YAW = 12.0

DEFAULT_MIN_FACE = 380.0


@dataclass(frozen=True)
class Step:
    path: Path
    smile: float
    gap: float
    opening: float
    sharpness: float
    candidate: Candidate


def measure(
    path: Path, tracker: MediaPipeFaceTracker, timestamp: float = 0.0
) -> tuple[Step, float] | None:
    """One frame's smile, mouth opening and sharpness, with its yaw."""
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        return None

    result = tracker.process(
        VideoFrame(
            data=np.ascontiguousarray(image),
            pixel_format=PixelFormat.BGR24,
            metadata=FrameMetadata(frame_id=0, capture_timestamp=timestamp),
        )
    )
    if not result.face_detected or result.landmarks is None or result.head_pose is None:
        return None

    points = result.landmarks.points
    candidate = Candidate(
        index=0,
        image=image,
        tracking=result,
        face_width=float(points[:, 0].max() - points[:, 0].min()) * image.shape[1],
        sharpness=sharpness_of(image, points),
        identity_drift=0.0,
    )
    step = Step(
        path=path,
        smile=_smiling(result),
        gap=_lip_gap(result),
        opening=_lip_parting(result),
        sharpness=candidate.sharpness,
        candidate=candidate,
    )
    return step, float(result.head_pose.yaw)


def amounts(steps: list[Step], rest: float | None = None) -> list[tuple[Step, float]]:
    """Each frame with its place on the ladder's axis.

    The engine's own arithmetic, and it has to stay that way: the number a rung
    is filed under is the number it will be chosen by. See
    :func:`sha256.avatar.engine_warp._smile_amount` — how much wider the mouth
    is than it rests, **times** how far the lips have come apart, each on its
    own measured span. Adding them instead reads speech as a smile.

    **The blendshape is gone from both sides as of 2026-08-21.** Across the ten
    rungs it had already filed it moved only from 0.529 to 0.798 and put them in
    the wrong order — `smile_140.png` was filed above `smile_108.png` and shows
    less smile. Mouth width against the eye span does not saturate: the same ten
    run 0.586 to 0.722, in order.

    ``rest`` is the mouth's resting width, and it comes from the portrait when
    one is given: the avatar's resting mouth is the zero the engine will
    subtract. The narrowest frame in the clip is the fallback.
    """
    if rest is None:
        rest = min(step.smile for step in steps)
    placed = []
    for step in steps:
        widened = max(step.smile - rest, 0.0) / _SMILE_WIDEN_SPAN
        parted = max(step.opening - _SMILE_LIP_CLOSED, 0.0) / _SMILE_OPEN_SPAN
        placed.append((step, widened * (1.0 + parted)))
    return placed


def best_per_band(placed: list[tuple[Step, float]], step_size: float) -> list[tuple[Step, float]]:
    """The sharpest frame in each band of the axis, ordered along it."""
    held: dict[int, tuple[Step, float]] = {}
    for step, amount in placed:
        band = round(amount / step_size)
        if band not in held or step.sharpness > held[band][0].sharpness:
            held[band] = (step, amount)
    return sorted(held.values(), key=lambda pair: pair[1])


def rung_name(amount: float) -> str:
    return f"smile_{round(amount * 100):03d}.png"


def install(chosen: list[tuple[Step, float]], avatar: Path) -> None:
    """Write the rungs in and replace the manifest's smile entries.

    The two named smile photographs go as well. The ladder holds better
    versions of both — a closed smile is its low end and an open one its high —
    and leaving them in would put a single wide smile back in competition with
    the right-sized rung.
    """
    import cv2

    manifest = avatar / "metadata.json"
    shutil.copy2(manifest, manifest.with_suffix(".json.bak"))
    meta = json.loads(manifest.read_text(encoding="utf-8"))

    retired = avatar / "superseded-smiles"
    retired.mkdir(exist_ok=True)
    for old in avatar.glob("smile_*.png"):
        shutil.move(str(old), str(retired / old.name))

    before = len(meta["assets"])
    meta["assets"] = [
        asset
        for asset in meta["assets"]
        if asset.get("smile") is None
        and asset.get("role") not in ("source_smile", "source_smile_closed")
    ]
    print(f"\nretired {before - len(meta['assets'])} manifest entrie(s)")

    for step, amount in chosen:
        name = rung_name(amount)
        cv2.imwrite(str(avatar / name), crop_around_face(step.candidate))
        meta["assets"].append(
            {
                "path": name,
                "role": "source_smile_step",
                "optional": True,
                "smile": round(amount, 3),
            }
        )

    manifest.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"installed {len(chosen)} rung(s); manifest now holds {len(meta['assets'])} assets")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--avatar", type=Path, default=REPO_ROOT / "avatars" / "my_avatar")
    parser.add_argument("--step", type=float, default=DEFAULT_STEP)
    parser.add_argument("--min-smile", type=float, default=DEFAULT_MIN_SMILE)
    parser.add_argument("--max-yaw", type=float, default=DEFAULT_MAX_YAW)
    parser.add_argument("--min-face", type=float, default=DEFAULT_MIN_FACE)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)

    model = REPO_ROOT / "models" / DEFAULT_MODEL_RELATIVE_PATH
    if not model.is_file():
        print(f"Model bundle missing: {model}")
        return 1

    tracker = MediaPipeFaceTracker(model_path=model)
    tracker.start()
    try:
        found: list[Step] = []
        for folder in args.folders:
            frames = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
            measured = [
                measure(path, tracker, timestamp_of(path, index))
                for index, path in enumerate(frames)
            ]
            usable = [
                step
                for step, yaw in (m for m in measured if m is not None)
                if step.smile >= args.min_smile
                and abs(yaw) <= args.max_yaw
                and step.candidate.face_width >= args.min_face
            ]
            print(
                f"{folder.name:<12} {len(frames):>4} frames  {len(usable):>4} smiling and frontal"
            )
            found.extend(usable)

        if not found:
            print("\nNothing smiling in any of those folders.")
            return 1

        chosen = best_per_band(amounts(found), args.step)
        print(f"\n{len(chosen)} rung(s), one per {args.step:g} of the axis")
        print(f"{'amount':>7}{'smile':>7}{'lip gap':>9}{'sharp':>8}  frame")
        for step, amount in chosen:
            print(
                f"{amount:>7.2f}{step.smile:>7.2f}{step.gap:>9.3f}{step.sharpness:>8.0f}  "
                f"{step.path.parent.name}/{step.path.name}"
            )

        if args.install:
            install(chosen, args.avatar)
        else:
            print("\nNothing written. Pass --install to replace the avatar's smiles.")
    finally:
        tracker.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
