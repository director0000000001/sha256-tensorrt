"""Build a ladder of the brows moving, out of dumped frames, and install it.

The last family still driven by two named photographs, and the owner has
reported both halves of what that costs:

*   *"a slight raise takes on a surprised face — the brows do not actually go
    up."* Because the photograph shows a **full** raise, faded in by how far the
    brows moved: a small raise is a big one at low opacity, which is a surprised
    face, and the brows themselves never move.
*   *"two brows, and two more slightly above, faint."* Because the photograph's
    brows are somewhere the portrait's are not, so while it is faint both are on
    screen.

The axis is **where the brows actually are**: their height above the nose
bridge, over face height, against a resting face. Signed — zero at rest,
positive raised, negative drawn down.

Not ``browInnerUp``, which was the first attempt and had the failure this file
keeps meeting: it reacts to the eyelid as well as to the brow, reached its
maximum on 5% of the owner's recording, and put a fully raised brow on the
avatar for an instant every time their eyes opened. Measured on the brows clip,
the geometric height runs 0.0646 at rest, 0.0797 raised and 0.0570 drawn down,
monotonically through the movement.

Usage::

    python scripts/build_brow_ladder.py avatars/my_avatar/frames/video-08
    python scripts/build_brow_ladder.py .../video-08 --install
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
    portrait_tracking,
    sharpness_of,
    timestamp_of,
)

from sha256.avatar.engine_warp import _BROW_SPAN, _brow_height  # noqa: E402
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FrameMetadata, VideoFrame  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)

# One rung per this much of the axis, which runs -1 to +1.
DEFAULT_STEP = 0.15

# Below this much movement either way the brows are at rest, and the portrait is
# already the best photograph of that.
DEFAULT_MIN_MOVED = 0.10

DEFAULT_MAX_YAW = 12.0
DEFAULT_MIN_FACE = 380.0


@dataclass(frozen=True)
class Step:
    path: Path
    height: float
    sharpness: float
    candidate: Candidate


def measure(
    path: Path, tracker: MediaPipeFaceTracker, timestamp: float = 0.0
) -> tuple[Step, float] | None:
    """One frame's brow blendshapes and sharpness, with its yaw."""
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
        height=_brow_height(result),
        sharpness=candidate.sharpness,
        candidate=candidate,
    )
    return step, float(result.head_pose.yaw)


def amounts(steps: list[Step], rest: float | None = None) -> list[tuple[Step, float]]:
    """Each frame with where its brows sit against rest. Signed.

    ``(height - rest) / rest``, scaled so a full raise reaches one — the same
    arithmetic the engine performs against its own slow mean.

    **Rest comes from the base portrait**, for the reason set out at length in
    `build_eye_ladder.amounts`: the median of the clip is only rest if the clip
    is mostly at rest, and a brow clip is built out of held brow positions. The
    engine compares the driver's brow against the driver's rest with the rung's
    amount against the avatar's, and the avatar's rest is the portrait.

    Measured after the timestamp fix, the rungs installed under the old rule are
    filed 0.563 out on an axis of 0 to 1, and not even in order:
    `brows_u088.png` is filed at +0.880 and is a resting brow.

    ``rest`` falls back to the median when no portrait is available.
    """
    if rest is None:
        rest = float(np.median([step.height for step in steps]))
    if rest <= 1e-6:
        return [(step, 0.0) for step in steps]
    return [
        (step, min(max(((step.height - rest) / rest) / _BROW_SPAN, -1.0), 1.0)) for step in steps
    ]


def best_per_band(placed: list[tuple[Step, float]], step_size: float) -> list[tuple[Step, float]]:
    """The sharpest frame in each band of the axis, ordered along it."""
    held: dict[int, tuple[Step, float]] = {}
    for step, amount in placed:
        band = round(amount / step_size)
        if band not in held or step.sharpness > held[band][0].sharpness:
            held[band] = (step, amount)
    return sorted(held.values(), key=lambda pair: pair[1])


def rung_name(amount: float) -> str:
    side = "d" if amount < 0 else "u"
    return f"brows_{side}{round(abs(amount) * 100):03d}.png"


def install(
    chosen: list[tuple[Step, float]],
    avatar: Path,
    tracker: MediaPipeFaceTracker,
    rest: float,
) -> None:
    """Write the rungs in and replace the manifest's brow entries.

    Both named photographs go with them. A raised brow and a lowered one are two
    ends of one movement, which is what the signed axis says and what two
    separate photographs could never say.

    Filed from the **sequence** measurement, not from the crop that is written.
    That was tried on 2026-08-21 and made the brow ladder worse — 0.442 mean
    error to 0.545 — because MediaPipe's brow landmarks are not repeatable
    across a re-detection: the same face reads 0.0650 in its source frame,
    0.0831 in the crop cut from it and 0.0798 in a crop framed like the
    portrait, on an axis whose whole span is 0.039. The reading inside one
    continuous clip is far steadier than any reading taken again afterwards,
    and the engine never re-measures a rung anyway — it reads the manifest.
    """

    import cv2

    manifest = avatar / "metadata.json"
    shutil.copy2(manifest, manifest.with_suffix(".json.bak"))
    meta = json.loads(manifest.read_text(encoding="utf-8"))

    retired = avatar / "superseded-brows"
    retired.mkdir(exist_ok=True)
    for old in avatar.glob("brows_*.png"):
        shutil.move(str(old), str(retired / old.name))

    before = len(meta["assets"])
    meta["assets"] = [
        asset
        for asset in meta["assets"]
        if asset.get("brows") is None
        and asset.get("role") not in ("source_brows_up", "source_brows_down")
    ]
    print(f"\nretired {before - len(meta['assets'])} manifest entrie(s)")

    for step, amount in chosen:
        name = rung_name(amount)
        cv2.imwrite(str(avatar / name), crop_around_face(step.candidate))
        meta["assets"].append(
            {
                "path": name,
                "role": "source_brows_step",
                "optional": True,
                "brows": round(amount, 3),
            }
        )

    manifest.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"installed {len(chosen)} rung(s); manifest now holds {len(meta['assets'])} assets")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--avatar", type=Path, default=REPO_ROOT / "avatars" / "my_avatar")
    parser.add_argument(
        "--base",
        type=Path,
        default=None,
        help="The portrait the ladder's zero is measured from; defaults to the avatar's.",
    )
    parser.add_argument("--step", type=float, default=DEFAULT_STEP)
    parser.add_argument("--min-moved", type=float, default=DEFAULT_MIN_MOVED)
    parser.add_argument("--max-yaw", type=float, default=DEFAULT_MAX_YAW)
    parser.add_argument("--min-face", type=float, default=DEFAULT_MIN_FACE)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)

    model = REPO_ROOT / "models" / DEFAULT_MODEL_RELATIVE_PATH
    if not model.is_file():
        print(f"Model bundle missing: {model}")
        return 1

    base_path = args.base or args.avatar / "portrait.png"

    # A tracker of its own for the portrait: it is an unrelated still, and the
    # tracker carries a prior between the frames it is given.
    portrait = MediaPipeFaceTracker(model_path=model)
    portrait.start()
    try:
        base = portrait_tracking(base_path, portrait)
    finally:
        portrait.stop()
    if base is None:
        print(f"No face found in the base portrait {base_path}.")
        return 1

    # The ladder's zero — see :func:`amounts`.
    resting = _brow_height(base[0])
    print(f"resting brow, from {base_path.name}: {resting:.4f}")

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
                if abs(yaw) <= args.max_yaw and step.candidate.face_width >= args.min_face
            ]
            print(f"{folder.name:<12} {len(frames):>4} frames  {len(usable):>4} frontal")
            found.extend(usable)

        if not found:
            print("\nNothing usable in any of those folders.")
            return 1

        placed = [
            (step, amount)
            for step, amount in amounts(found, resting)
            if abs(amount) >= args.min_moved
        ]
        if not placed:
            print("\nNo frame in these folders has the brows anywhere but at rest.")
            return 1

        chosen = best_per_band(placed, args.step)
        print(f"\n{len(chosen)} rung(s), one per {args.step:g} of the axis")
        print(f"{'brows':>7}{'height':>9}{'sharp':>8}  frame")
        for step, amount in chosen:
            print(
                f"{amount:>7.2f}{step.height:>9.4f}{step.sharpness:>8.0f}  "
                f"{step.path.parent.name}/{step.path.name}"
            )

        if args.install:
            install(chosen, args.avatar, tracker, resting)
        else:
            print("\nNothing written. Pass --install to replace the avatar's brows.")
    finally:
        tracker.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
