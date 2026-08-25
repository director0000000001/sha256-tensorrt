"""Build a ladder of the lips parting, out of dumped frames, and install it.

The owner's first complaint, and the one day 13 diagnosed against the obvious
reading: *"when I am talking it almost does not follow me at all."* The mouth
**does** follow — lip parting gain 1.04, r 0.99, measured by re-tracking the
render. What is wrong is which photograph it follows with. Measured on the same
clip, `source_mouth_o` won the mouth region on **65.6% of frames at a median
share of 0.18**: two thirds of a conversation rendered as the portrait's own
*closed* mouth with one rounded-lips photograph washed over it at a fifth, and
the mesh pulling the closed mouth open. A stretched closed mouth, not a
speaking one.

That is the smile's old failure and the eyes' old failure, for the third time,
and it has the same answer: a row of photographs of the movement, and the
nearest one shown whole.

**The axis is lip parting alone, and that is a measurement rather than a
choice.** Speech was expected to need a grid — an eye is open or shut, but a
mouth's width and its parting move independently, and the engine computes both.
`scripts/measure_speech_plane.py` measured what the owner's mouth actually does
over the 240 frames of `driving.mkv` where he is speaking, in the engine's own
normalised units:

    widened   p5/50/95   0.000 / 0.046 / 0.118
    parted    p5/50/95   0.000 / 0.202 / 0.947

They move together at r = -0.049, and one direction carries **98.4%** of the
demand: (-0.006 widened, +1.000 parted). Speech parts the lips and barely widens
the mouth. So: one ladder, on parting, and no grid.

**Two clips, on purpose.** The same tool measured that the `speech` clip covers
every band the driver visits with no holes — worst nearest-photograph distance
0.066, against 0.567 for the whole installed library — but it is 720p with a 389
px face, where `video-10` is 4K with 1009 against a portrait of 455. `video-10`
reaches only to parted 0.6. So the sharpest frame per band wins, which gives the
4K clip the bottom of the ladder and the 720p clip the top, and the top is the
part that did not exist at all before.

Usage::

    python scripts/build_speech_ladder.py avatars/my_avatar/frames/speech
    python scripts/build_speech_ladder.py .../speech .../video-10 --install
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
    identity_ratios,
    portrait_tracking,
    sharpness_of,
    timestamp_of,
)

from sha256.avatar.engine_warp import (  # noqa: E402
    _SMILE_LIP_CLOSED,
    _SMILE_OPEN_SPAN,
    _SPEECH_PREFIX,
    _lip_gap,
    _lip_parting,
    _smiling,
    refused_as_rung,
)
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FrameMetadata, VideoFrame  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)

# One rung per this much of the axis. A tenth, which gives fifteen rungs over
# the range below — the eye ladder's order of magnitude, and fine enough that
# the step from one rung to the next is a mouth moving rather than a mouth
# changing.
DEFAULT_STEP = 0.1

# Below this the lips are together and the portrait is already the best
# photograph of that. It is the ladder's own foot: the portrait is the rung at
# zero, exactly as it is for yaw, the smile and the lids.
DEFAULT_MIN_PARTED = 0.05

# And the top. The owner's widest opening over the whole of `driving.mkv` is
# 1.21 and his widest while speaking is 1.03; the `speech` clip goes on to 2.3.
# Rungs past the demand are not wrong, they are unused — and every one costs a
# face detection at load, on a startup that already takes 21 seconds. So: the
# driver's own maximum with a quarter of margin. Raise it for a driver who opens
# their mouth wider than this one does.
DEFAULT_MAX_PARTED = 1.5

# A rung has to be frontal and large enough to cut a mouth out of: it is placed
# on the avatar's own frontal layout. Both are the other builders' thresholds.
DEFAULT_MAX_YAW = 12.0
DEFAULT_MIN_FACE = 380.0

# How far a frame's proportions may drift from the portrait's before it is a
# photograph of somebody else. `extract_avatar_frames.IDENTITY_TOLERANCE`, and
# it matters more here than on the other ladders: the `speech` clip's worst
# frame measures 0.095 against it.
DEFAULT_MAX_DRIFT = 0.08

ROLE = "source_speech_step"


@dataclass(frozen=True)
class Step:
    path: Path
    parted: float
    width: float
    gap: float
    sharpness: float
    candidate: Candidate
    drift: float
    refused: str | None = None


def measure(
    path: Path,
    tracker: MediaPipeFaceTracker,
    base_points: np.ndarray,
    base_ratios: dict[str, float],
    timestamp: float = 0.0,
) -> tuple[Step, float] | None:
    """One frame's lip parting, mouth width and sharpness, with its yaw.

    And whether the engine would refuse it, asked here rather than at install
    because the answer decides which frame wins its band — a band whose sharpest
    frame is one the loader will drop ends up with no rung at all, and the
    ladder acquires a hole nobody was told about. That is not hypothetical: it
    is what `refused_as_rung` was written for, after the eye ladder lost two.
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
    height, width = image.shape[:2]
    ratios = identity_ratios(points, width, height)
    drift = max(
        abs(ratios[name] - base_ratios[name]) / max(base_ratios[name], 1e-6) for name in base_ratios
    )
    candidate = Candidate(
        index=0,
        image=image,
        tracking=result,
        face_width=float(points[:, 0].max() - points[:, 0].min()) * width,
        sharpness=sharpness_of(image, points),
        identity_drift=drift,
    )
    step = Step(
        path=path,
        parted=max(_lip_parting(result) - _SMILE_LIP_CLOSED, 0.0) / _SMILE_OPEN_SPAN,
        width=_smiling(result),
        gap=_lip_gap(result),
        sharpness=candidate.sharpness,
        candidate=candidate,
        drift=drift,
        refused=refused_as_rung(_SPEECH_PREFIX, points * (width, height, 1.0), base_points),
    )
    return step, float(result.head_pose.yaw)


def best_per_band(steps: list[Step], step_size: float) -> list[Step]:
    """The sharpest frame in each band of the axis, ordered along it.

    A frame the engine would refuse never wins a band, however sharp it is: the
    band would then be empty at load and nothing would say so.
    """
    held: dict[int, Step] = {}
    for step in steps:
        if step.refused is not None:
            continue
        band = round(step.parted / step_size)
        if band not in held or step.sharpness > held[band].sharpness:
            held[band] = step
    return sorted(held.values(), key=lambda step: step.parted)


def rung_name(parted: float) -> str:
    return f"speech_{round(parted * 100):03d}.png"


def install(chosen: list[Step], avatar: Path) -> None:
    """Write the rungs in and replace the manifest's speech entries.

    **The named mouth photographs stay.** The smile ladder retired
    `source_smile` and `source_smile_closed` because it held ten better versions
    of both. This ladder holds better versions of one thing only — the *opening*
    — and `source_mouth_o`, `source_mouth_wide` and `source_mouth_pressed` are
    shapes rather than openings, driven by their own blendshapes and carrying
    what the parting axis cannot. Whether they still earn their place is a
    measurement, not a guess, and it is the engine's to make frame by frame:
    `_one_per_region` already keeps exactly one photograph on the mouth.
    """
    import cv2

    manifest = avatar / "metadata.json"
    shutil.copy2(manifest, manifest.with_suffix(".json.bak"))
    meta = json.loads(manifest.read_text(encoding="utf-8"))

    retired = avatar / "superseded-speech"
    retired.mkdir(exist_ok=True)
    for old in avatar.glob("speech_*.png"):
        shutil.move(str(old), str(retired / old.name))

    before = len(meta["assets"])
    meta["assets"] = [asset for asset in meta["assets"] if asset.get("speech") is None]
    if before != len(meta["assets"]):
        print(f"\nretired {before - len(meta['assets'])} manifest entrie(s)")

    for step in chosen:
        name = rung_name(step.parted)
        cv2.imwrite(str(avatar / name), crop_around_face(step.candidate))
        meta["assets"].append(
            {
                "path": name,
                "role": ROLE,
                "optional": True,
                "speech": round(step.parted, 3),
            }
        )

    manifest.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"installed {len(chosen)} rung(s); manifest now holds {len(meta['assets'])} assets")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--avatar", type=Path, default=REPO_ROOT / "avatars" / "my_avatar")
    parser.add_argument("--step", type=float, default=DEFAULT_STEP)
    parser.add_argument("--min-parted", type=float, default=DEFAULT_MIN_PARTED)
    parser.add_argument("--max-parted", type=float, default=DEFAULT_MAX_PARTED)
    parser.add_argument("--max-yaw", type=float, default=DEFAULT_MAX_YAW)
    parser.add_argument("--min-face", type=float, default=DEFAULT_MIN_FACE)
    parser.add_argument("--max-drift", type=float, default=DEFAULT_MAX_DRIFT)
    parser.add_argument("--install", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    import cv2

    args = build_parser().parse_args(argv)

    model = REPO_ROOT / "models" / DEFAULT_MODEL_RELATIVE_PATH
    if not model.is_file():
        print(f"Model bundle missing: {model}")
        return 1

    tracker = MediaPipeFaceTracker(model_path=model)
    tracker.start()
    try:
        portrait_path = args.avatar / "portrait.png"
        portrait = portrait_tracking(portrait_path, tracker)
        portrait_image = cv2.imread(str(portrait_path))
        if portrait is None or portrait_image is None:
            print(f"No face in {portrait_path}.")
            return 1
        base_result, base_points = portrait
        base_ratios = identity_ratios(
            base_result.landmarks.points, portrait_image.shape[1], portrait_image.shape[0]
        )
        print(f"portrait: resting mouth {_smiling(base_result):.4f} of an eye span")

        found: list[Step] = []
        for folder in args.folders:
            frames = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
            measured = [
                measure(path, tracker, base_points, base_ratios, timestamp_of(path, index))
                for index, path in enumerate(frames)
            ]
            usable = [
                step
                for step, yaw in (m for m in measured if m is not None)
                if args.min_parted <= step.parted <= args.max_parted
                and abs(yaw) <= args.max_yaw
                and step.candidate.face_width >= args.min_face
                and step.drift <= args.max_drift
            ]
            refused = sum(1 for step in usable if step.refused is not None)
            print(
                f"{folder.name:<12} {len(frames):>4} frames  {len(usable):>4} parted and frontal"
                + (f"  ({refused} the engine would refuse)" if refused else "")
            )
            found.extend(usable)
    finally:
        tracker.stop()

    if not found:
        print("\nNo parted lips in any of those folders.")
        return 1

    chosen = best_per_band(found, args.step)
    print(f"\n{len(chosen)} rung(s), one per {args.step:g} of the axis")
    print(f"{'parted':>7}{'width':>7}{'lip gap':>9}{'sharp':>7}{'face px':>9}{'drift':>7}  frame")
    for step in chosen:
        print(
            f"{step.parted:>7.2f}{step.width:>7.3f}{step.gap:>9.3f}{step.sharpness:>7.0f}"
            f"{step.candidate.face_width:>9.0f}{step.drift:>7.3f}  "
            f"{step.path.parent.name}/{step.path.name}"
        )

    # Between the lowest rung and the highest, and no further. The band at zero
    # is the portrait's own — the foot of this ladder is a closed mouth, and the
    # portrait is already the best photograph of one — and a band past the last
    # rung is the top of the range rather than a hole in it. What matters is a
    # gap *inside* the ladder, because that is a parting with nothing to render
    # it but a neighbour two bands away.
    filled = {round(step.parted / args.step) for step in chosen}
    holes = sorted(set(range(min(filled), max(filled) + 1)) - filled)
    if holes:
        named = ", ".join(f"{band * args.step:.1f}" for band in holes)
        print(f"\n{len(holes)} empty band(s) inside the ladder: {named}")
    else:
        print("\nNo holes: every band between the lowest rung and the highest has one.")

    if args.install:
        install(chosen, args.avatar)
    else:
        print("\nNothing written. Pass --install to give the avatar a speech ladder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
