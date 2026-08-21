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
    portrait_tracking,
    sharpness_of,
    timestamp_of,
)

from sha256.avatar.engine_warp import (  # noqa: E402
    _EYES_PREFIX,
    _eye_opening,
    _pixel_points,
    refused_as_rung,
)
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FrameMetadata, VideoFrame  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)

# One rung per this much of the closing. A twentieth: the clip it is built from
# holds the lids at every stage of a closing rather than blinking through it, so
# the bands that used to be empty are the ones the owner's first report is
# about. A tenth was right for a clip that had two usable frames in it.
DEFAULT_STEP = 0.05

# Below this much movement either way the eye is at rest, and the portrait is
# already the best photograph of that. Down from 0.08 with the step, and for the
# same reason: at 0.08 the shallowest rung of the 2026-08-21 ladder — a lid a
# fifteenth of the way down, and a genuine photograph of one — is thrown away
# before the bands are formed.
DEFAULT_MIN_MOVED = 0.03

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
    refused: str | None = None


def measure(
    path: Path,
    tracker: MediaPipeFaceTracker,
    base_points: np.ndarray | None = None,
    timestamp: float = 0.0,
) -> tuple[Step, float] | None:
    """One frame's eyelid opening and sharpness, with its yaw.

    And whether the engine would refuse it. That question is asked here rather
    than at install because the answer decides which frame wins its band — see
    :func:`best_per_band`.
    """
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
    refused = None
    if base_points is not None:
        mesh = _pixel_points(points, image.shape[1], image.shape[0])
        refused = (
            "its landmarks could not be read"
            if mesh is None
            else refused_as_rung(_EYES_PREFIX, mesh, base_points)
        )

    step = Step(
        path=path,
        opening=_eye_opening(result),
        sharpness=candidate.sharpness,
        candidate=candidate,
        refused=refused,
    )
    return step, float(result.head_pose.yaw)


def amounts(steps: list[Step], rest: float | None = None) -> list[tuple[Step, float]]:
    """Each frame with where its lids sit against a resting eye. Signed.

    ``(rest - opening) / rest``: zero at rest, positive as the lids close,
    negative when they are open wider than usual — the same quantity the engine
    computes from the resting level it learns.

    **Rest comes from the base portrait**, and that is a correction. It used to
    be the median frame of the clip, on the reasoning that the median is where a
    face spends its time. That holds only for a clip that is mostly at rest, and
    an eye clip is the one kind that never is: the take that built the current
    ladder holds the eyes shut for four of its eight seconds, so its median *is*
    a shut eye. Measured on it — true rest 0.0538, median 0.0356 — which files
    every genuinely shut frame at about +0.45 instead of +0.65, and files the
    open frames as though the eyes were wide.

    The portrait is the right zero for a second reason, not just a safer one.
    The engine picks a rung by comparing the driver's closure *against the
    driver's own rest* with the rung's amount *against the avatar's*. The
    avatar's rest is the portrait's face. Anything else answers a different
    question from the one the rung is asked.

    ``rest`` falls back to the median when no portrait is available, with the
    caveat above.
    """
    if rest is None:
        rest = float(np.median([step.opening for step in steps]))
    if rest <= 1e-6:
        return [(step, 0.0) for step in steps]
    return [(step, (rest - step.opening) / rest) for step in steps]


def best_per_band(placed: list[tuple[Step, float]], step_size: float) -> list[tuple[Step, float]]:
    """The sharpest **usable** frame in each band of closing, ordered along it.

    Sharpest alone was the rule until a band was lost by it. The engine refuses,
    at load, a rung whose features land in another region — and both of the
    owner's shallow rungs were refused that way, so the ladder ran from a wide
    open eye straight to a shut one with nothing between. A blink then had to be
    rendered by fading the shut photograph in over the open portrait, which puts
    both pictures on screen together: the iris seen through a closed eyelid.

    So a frame the engine will not accept is not a candidate for its band, and
    the band goes to the sharpest one that is left. A softer rung that can be
    placed is worth more than a sharp one that is thrown away.
    """
    held: dict[int, tuple[Step, float]] = {}
    for step, amount in placed:
        if step.refused is not None:
            continue
        band = round(amount / step_size)
        if band not in held or step.sharpness > held[band][0].sharpness:
            held[band] = (step, amount)
    return sorted(held.values(), key=lambda pair: pair[1])


def _by_band(
    placed: list[tuple[Step, float]], step_size: float
) -> dict[int, list[tuple[Step, float]]]:
    """Every frame grouped by its band, refused ones included. For reporting."""
    grouped: dict[int, list[tuple[Step, float]]] = {}
    for step, amount in placed:
        grouped.setdefault(round(amount / step_size), []).append((step, amount))
    return grouped


def rung_name(amount: float) -> str:
    side = "w" if amount < 0 else "c"
    return f"eyes_{side}{round(abs(amount) * 100):03d}.png"


def install(
    chosen: list[tuple[Step, float]],
    avatar: Path,
    tracker: MediaPipeFaceTracker,
    rest: float,
) -> None:
    """Write the rungs in and replace the manifest's eye entries.

    ``source_eyes_closed`` and ``source_eyes_wide`` go with them: the ladder's
    two ends are better versions of both, measured rather than named, and
    leaving them in would put a fully shut eye back in competition with a
    half-closed one and a full stare with a slightly widened one.

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
    parser.add_argument(
        "--base",
        type=Path,
        default=None,
        help="The portrait each rung is checked against. Defaults to the avatar's own.",
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

    # A tracker of its own for the portrait: it is an unrelated still and the
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
    base_result, base_points = base

    # The ladder's zero, and it comes from the avatar's own resting face rather
    # than from the middle of the clip — see :func:`amounts`.
    resting = _eye_opening(base_result)
    print(f"resting eye, from {base_path.name}: {resting:.4f}")

    tracker = MediaPipeFaceTracker(model_path=model)
    tracker.start()
    try:
        found: list[Step] = []
        for folder in args.folders:
            frames = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
            measured = [
                measure(path, tracker, base_points, timestamp_of(path, index))
                for index, path in enumerate(frames)
            ]
            usable = [
                step
                for step, yaw in (m for m in measured if m is not None)
                if abs(yaw) <= args.max_yaw and step.candidate.face_width >= args.min_face
            ]
            refused = sum(1 for step in usable if step.refused is not None)
            print(
                f"{folder.name:<12} {len(frames):>4} frames  {len(usable):>4} frontal"
                f"  {refused:>4} the engine would refuse"
            )
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
            print("\nNo frame in these folders has the eyes further shut than rest.")
            return 1

        chosen = best_per_band(placed, args.step)
        if not chosen:
            print("\nEvery frame with the eyes moved is one the engine would refuse.")
            return 1

        print(f"\n{len(chosen)} rung(s), one per {args.step:g} of the closing")
        print(f"{'closed':>7}{'lid':>9}{'sharp':>8}  frame")
        for step, amount in chosen:
            print(
                f"{amount:>7.2f}{step.opening:>9.3f}{step.sharpness:>8.0f}  "
                f"{step.path.parent.name}/{step.path.name}"
            )

        # What sharpness alone would have taken, and why it was passed over.
        # Silently is how the ladder lost two rungs the first time.
        taken = {id(step) for step, _ in chosen}
        for band, group in sorted(_by_band(placed, args.step).items()):
            top = max(group, key=lambda pair: pair[0].sharpness)
            if top[0].refused is None or id(top[0]) in taken:
                continue
            print(
                f"\npassed over at {band * args.step:+.2f}: "
                f"{top[0].path.parent.name}/{top[0].path.name} "
                f"(sharpness {top[0].sharpness:.0f}) — {top[0].refused}"
            )

        if args.install:
            install(chosen, args.avatar, tracker, resting)
        else:
            print("\nNothing written. Pass --install to replace the avatar's eyes.")
    finally:
        tracker.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
