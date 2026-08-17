"""Build a ladder of the lids closing, out of dumped frames, and install it.

The eyes are the owner's oldest complaint and the one they rank first: *"wrong
far more often than they are right"*. Every fix so far has been a subtraction —
the four gaze photographs dropped, the squint dropped, two blendshapes found to
measure something other than their names — and what was left is a single
shut-eye photograph fading in over an open one. That has the smile's two
failures exactly:

*   while it is faint there are two sets of eyes, an open one and a shut one
    over it, which is the ghost the owner keeps reporting;
*   and at any weight the lid it draws is *fully shut*, so a half-closed eye is
    a shut one at half opacity rather than a half-closed one.

A row of photographs of the lid on its way down answers both, and the same row
continued the other way answers the third: measured on the owner's recording,
the wide-eye photograph fired on 4.6% of frames at a median weight of 0.11 — a
sixth of a full stare over a face doing nothing, which they described as looking
a little mad.

The axis is the eyelid opening against a **resting** eye, signed: zero at rest,
positive as the lids close, negative when they open wider. Rest is the median of
the clip rather than its widest frame — a clip containing an eyes-wide hold has
its widest frame at the wide end, and measuring from there puts a resting eye a
third of the way shut on its own scale.

Usage::

    python scripts/build_eye_ladder.py avatars/my_avatar/frames/video-07
    python scripts/build_eye_ladder.py .../video-07 .../video-08 --install
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
)

from sha256.avatar.engine_warp import _eye_opening  # noqa: E402
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FrameMetadata, VideoFrame  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)

# One rung per this much of the closing. A tenth is ten photographs from open to
# shut, finer than the lid moves between two frames of a blink.
DEFAULT_STEP = 0.1

# Below this much movement either way the eye is at rest, and the portrait is
# already the best photograph of that.
DEFAULT_MIN_MOVED = 0.08

# A rung has to be frontal: the lid is cut out of it and placed on the avatar's
# own frontal layout.
DEFAULT_MAX_YAW = 12.0
DEFAULT_MIN_FACE = 380.0


@dataclass(frozen=True)
class Step:
    path: Path
    opening: float
    sharpness: float
    candidate: Candidate


def measure(path: Path, tracker: MediaPipeFaceTracker) -> tuple[Step, float] | None:
    """One frame's eyelid opening and sharpness, with its yaw."""
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        return None

    result = tracker.process(
        VideoFrame(
            data=np.ascontiguousarray(image),
            pixel_format=PixelFormat.BGR24,
            metadata=FrameMetadata(frame_id=0, capture_timestamp=0.0),
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
        opening=_eye_opening(result),
        sharpness=candidate.sharpness,
        candidate=candidate,
    )
    return step, float(result.head_pose.yaw)


def amounts(steps: list[Step]) -> list[tuple[Step, float]]:
    """Each frame with where its lids sit against a resting eye. Signed.

    ``(rest - opening) / rest``: zero at rest, positive as the lids close,
    negative when they are open wider than usual — the same quantity the engine
    computes from the resting level it learns.

    **Rest is the median, not the widest.** A clip that contains an eyes-wide
    hold has its widest frame at the wide end, and measuring everything from
    there puts a resting eye a third of the way shut on its own scale. The
    median is where a face spends its time, which is what rest means.
    """
    rest = float(np.median([step.opening for step in steps]))
    if rest <= 1e-6:
        return [(step, 0.0) for step in steps]
    return [(step, (rest - step.opening) / rest) for step in steps]


def best_per_band(placed: list[tuple[Step, float]], step_size: float) -> list[tuple[Step, float]]:
    """The sharpest frame in each band of closing, ordered along it."""
    held: dict[int, tuple[Step, float]] = {}
    for step, amount in placed:
        band = round(amount / step_size)
        if band not in held or step.sharpness > held[band][0].sharpness:
            held[band] = (step, amount)
    return sorted(held.values(), key=lambda pair: pair[1])


def rung_name(amount: float) -> str:
    side = "w" if amount < 0 else "c"
    return f"eyes_{side}{round(abs(amount) * 100):03d}.png"


def install(chosen: list[tuple[Step, float]], avatar: Path) -> None:
    """Write the rungs in and replace the manifest's eye entries.

    ``source_eyes_closed`` and ``source_eyes_wide`` go with them: the ladder's
    two ends are better versions of both, measured rather than named, and
    leaving them in would put a fully shut eye back in competition with a
    half-closed one and a full stare with a slightly widened one.
    """
    import cv2

    manifest = avatar / "metadata.json"
    shutil.copy2(manifest, manifest.with_suffix(".json.bak"))
    meta = json.loads(manifest.read_text(encoding="utf-8"))

    retired = avatar / "superseded-eyes"
    retired.mkdir(exist_ok=True)
    for old in avatar.glob("eyes_*.png"):
        shutil.move(str(old), str(retired / old.name))

    before = len(meta["assets"])
    meta["assets"] = [
        asset
        for asset in meta["assets"]
        if asset.get("eyes") is None
        and asset.get("role") not in ("source_eyes_closed", "source_eyes_wide")
    ]
    print(f"\nretired {before - len(meta['assets'])} manifest entrie(s)")

    for step, amount in chosen:
        name = rung_name(amount)
        cv2.imwrite(str(avatar / name), crop_around_face(step.candidate))
        meta["assets"].append(
            {
                "path": name,
                "role": "source_eyes_step",
                "optional": True,
                "eyes": round(amount, 3),
            }
        )

    manifest.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"installed {len(chosen)} rung(s); manifest now holds {len(meta['assets'])} assets")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--avatar", type=Path, default=REPO_ROOT / "avatars" / "my_avatar")
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

    tracker = MediaPipeFaceTracker(model_path=model)
    tracker.start()
    try:
        found: list[Step] = []
        for folder in args.folders:
            frames = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
            measured = [measure(path, tracker) for path in frames]
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
            (step, amount) for step, amount in amounts(found) if abs(amount) >= args.min_moved
        ]
        if not placed:
            print("\nNo frame in these folders has the eyes further shut than rest.")
            return 1

        chosen = best_per_band(placed, args.step)
        print(f"\n{len(chosen)} rung(s), one per {args.step:g} of the closing")
        print(f"{'closed':>7}{'lid':>9}{'sharp':>8}  frame")
        for step, amount in chosen:
            print(
                f"{amount:>7.2f}{step.opening:>9.3f}{step.sharpness:>8.0f}  "
                f"{step.path.parent.name}/{step.path.name}"
            )

        if args.install:
            install(chosen, args.avatar)
        else:
            print("\nNothing written. Pass --install to replace the avatar's eyes.")
    finally:
        tracker.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
