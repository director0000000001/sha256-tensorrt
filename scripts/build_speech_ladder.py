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

**It is a grid on two axes, and the one-axis version was wrong.** The first
build of this ladder filed rungs by lip parting alone, on a measurement that
said speech was one-dimensional: over the frames of `driving.mkv` where the
owner speaks, the widening carried 1.6% of the demand and the parting 98.4%.

**That measurement was taken through a clamp and the clamp hid half the
movement.** ``_smiling`` above rest is floored at zero, so every mouth *narrower*
than rest — which is what an "oo" and an "o" are — reads as exactly the same
value as a resting one. Measured again without it, on the speech clip: **20% of
its frames are narrower than the portrait's resting mouth**, and within a single
band of parting the width runs from 0.25 narrower to 0.22 wider.

So a rung chosen by parting alone has an essentially arbitrary shape, and the
owner reported precisely that within the hour of the first build: *"in the first
photo I am saying O, in the second A"* — with an O rendered as a wide open A.

Both axes are now filed and both are matched, on the engine's own units:

*   ``speech`` — the lips parted, over ``_SMILE_OPEN_SPAN``.
*   ``speech_width`` — the mouth's width against the portrait's rest, over
    ``_SMILE_WIDEN_SPAN``, clamped at zero on both sides so the driver's floor
    and the portrait's rest mean the same thing.

**Two clips, on purpose, and the bigger face wins a cell.** The `speech` clip
covers every band of parting the driver visits, but it is 720p with a 389 px
face where `video-10` is 4K with 1009, against a portrait of 455. A Laplacian
variance cannot compare those, so face size decides and sharpness only breaks
ties — see :func:`best_per_cell`.

Usage::

    python scripts/build_speech_ladder.py avatars/my_avatar/frames/video-10
    python scripts/build_speech_ladder.py .../video-10 .../speech --install
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    _SMILE_WIDEN_SPAN,
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

# **Unless the mouth is wide**, and that is what makes this a grid of the whole
# mouth rather than of speech. A closed smile parts the lips barely at all and
# is not the portrait's mouth in any other way: measured on this avatar, the
# widest closed smile in the library sits 0.28 above the portrait's resting
# width with the lips 0.003 apart. Filed by parting alone it fell under the foot
# and the grid never held it, so a smile had to be a second family with its own
# axis — and that axis was a *product* of the two, which cannot tell a wide
# mouth from an open one. It put a photograph with its lips nearly together on a
# mouth open six times as far: on the owner's recording the chosen photograph
# covered a median of 17% of the opening drawn for it, and he reported it as
# teeth that show "very badly".
#
# One width step away from rest — **either way, since 2026-08-28**. It was
# "above rest", on a reading clamped at zero, so every mouth narrower than rest
# was filed as though it were resting: a rounded "oo" and a mouth doing nothing
# had the same coordinate. Thirteen of the 67 rungs this built sat at exactly
# 0.00 with a pucker running from 0.001 to 0.911.
DEFAULT_MIN_WIDENED = 0.08

# And the top. The owner's widest opening while speaking is around 1.2 and the
# clips go on past 2.
#
# **Raised from 1.5 to 2.5 on 2026-08-26, because rungs past the demand stopped
# being unused.** `_covering_interior` borrows the *interior* of the narrowest
# photograph whose own mouth still covers the live opening, so the widest mouths
# in the library are what a frame wider than the grid falls back on. Cut at 1.5,
# the widest opening the avatar owned was 0.080 of face height and every frame
# past it showed a flat cavity below a band of teeth — which is what the owner
# reported on the wide open mouth. The cost is the same as it always was: a face
# detection each at load, on a startup that is already the worst thing here.
DEFAULT_MAX_PARTED = 2.5

# A rung has to be frontal and large enough to cut a mouth out of: it is placed
# on the avatar's own frontal layout. Both are the other builders' thresholds.
DEFAULT_MAX_YAW = 12.0
DEFAULT_MIN_FACE = 380.0

# How far a frame's proportions may drift from the portrait's before it is a
# photograph of somebody else. `extract_avatar_frames.IDENTITY_TOLERANCE`, and
# it matters more here than on the other ladders: the `speech` clip's worst
# frame measures 0.095 against it.
DEFAULT_MAX_DRIFT = 0.08

# How close to the widest face in a band a candidate has to be before its
# sharpness is looked at. Nine tenths: enough that two frames of one clip are
# always compared on sharpness, and far too little for a 720p clip to reach past
# a 4K one. See :func:`best_per_band`.
FACE_TIE = 0.9

# One cell per this much of the width axis. Measured on the owner's speech clip,
# the widening above his portrait's resting mouth runs 0 to 0.22 once it is
# clamped, so three cells cover it — enough to tell an "oo" from an "ah", and few
# enough that the grid does not spend a face detection at load on a shape the
# mouth barely makes.
DEFAULT_WIDTH_STEP = 0.08

# The fewest frames a cell needs before it becomes a rung.
#
# **One, since 2026-08-30, and the two it replaces could not rebuild the grid
# that is installed.** The rule was *"one frame is a moment of the clip rather
# than a shape the mouth makes"*, written when a rung was a 5.7 MB 4K PNG and
# the avatar took 24 s to load. Rungs are written at the portrait's scale now,
# and day 17 already measured what the rule costs: **105 of `video (14)`'s 227
# usable frames**.
#
# What it costs is worse than that, because the step came down with the grid.
# Run as committed over `video (14)` at the step the installed grid was built
# with — 0.035 of parting by 0.04 of width — this file produced **11 rungs**
# where the library holds **143** from that clip. At that spacing nearly every
# frame is alone in its cell, so a minimum of two throws away nearly everything.
#
# **That matters now rather than in principle.** The owner's remaining
# complaint is that the mouth *"makes a fast jump and goes straight to the A"*,
# and the answer to it is a clip whose frames are exactly the ones a fast
# opening passes through — one to a cell, every one of them.
CELL_MINIMUM = 1

ROLE = "source_speech_step"


@dataclass(frozen=True)
class Step:
    path: Path
    parted: float
    widened: float
    width: float
    gap: float
    sharpness: float
    candidate: Candidate
    drift: float
    refused: str | None = None


# **The nose flares when the mouth widens, so it cannot vouch for the identity
# of a mouth photograph.** `identity_ratios` holds three proportions that do not
# change between photographs of one person, and on a broad smile two of them
# hold: measured over the 98 frames of `video-09` where the mouth is 0.20 wider
# than the portrait's rest, the median drift is **0.008 of the face width and
# 0.047 of the eye spacing — against 0.152 of the nose width**. Judged on all
# three at a tolerance of 0.08, *every one of those 98 frames was refused*, and
# the grid could not hold a single wide smile. That is why a smile had to be a
# second family, and it is the reason it could stay one.
#
# The same lesson as the brow anchor, which was moved off the nose bridge on
# 2026-08-24 for moving with the thing it was measuring.
_MOVED_BY_THE_MOUTH = ("nose_width",)


def measure(
    path: Path,
    tracker: MediaPipeFaceTracker,
    base_points: np.ndarray,
    base_ratios: dict[str, float],
    base_width: float,
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
        abs(ratios[name] - base_ratios[name]) / max(base_ratios[name], 1e-6)
        for name in base_ratios
        if name not in _MOVED_BY_THE_MOUTH
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
        widened=(_smiling(result) - base_width) / _SMILE_WIDEN_SPAN,
        width=_smiling(result),
        gap=_lip_gap(result),
        sharpness=candidate.sharpness,
        candidate=candidate,
        drift=drift,
        refused=refused_as_rung(_SPEECH_PREFIX, points * (width, height, 1.0), base_points),
    )
    return step, float(result.head_pose.yaw)


def best_per_cell(steps: list[Step], step_size: float, width_step: float) -> list[Step]:
    """The best frame in each cell of the grid, ordered along the parting axis.

    **A cell, not a band**: one band of parting and one of width. Filing by
    parting alone left the width of the chosen rung to whatever the sharpest
    frame in the band happened to have, and on this clip that is anything from
    0.25 narrower than the resting mouth to 0.22 wider.

    **The bigger face wins, and sharpness only breaks ties within one clip.**
    The other builders rank by sharpness alone, which is right when every
    candidate comes from clips of one resolution and wrong the moment they do
    not: a Laplacian variance is a property of the pixels a face was drawn with,
    so a 389 px face and a 1009 px face cannot be compared by it at all.

    Ranking the first ladder by sharpness alone put eleven rungs from a 720p
    clip into a library whose every other source is 4K, and the owner named both
    halves of what that cost within the hour: *"the beard colour differs, and
    that makes a big difference"*, and *"the lips are also a little thicker when
    you open your mouth slightly"*. Both are one cause — a 389 px face scaled up
    to the portrait's 455 is soft, and its beard smears into the skin.

    So: a candidate has to be within :data:`FACE_TIE` of the largest face in its
    cell before its sharpness is consulted at all. Where a 4K clip has any frame
    in a cell, that cell is 4K.

    A frame the engine would refuse never wins a cell, however good it is: the
    cell would then be empty at load and nothing would say so.

    Cells holding fewer than :data:`CELL_MINIMUM` frames are dropped. A cell with
    one frame in it is a moment of the clip rather than a shape the mouth makes,
    and installing it spends a face detection at load on a rung that will win
    almost nothing.
    """
    cells: dict[tuple[int, int], list[Step]] = {}
    for step in steps:
        if step.refused is not None:
            continue
        key = (round(step.parted / step_size), round(step.widened / width_step))
        cells.setdefault(key, []).append(step)

    held = []
    for candidates in cells.values():
        if len(candidates) < CELL_MINIMUM:
            continue
        widest = max(step.candidate.face_width for step in candidates)
        close = [s for s in candidates if s.candidate.face_width >= widest * FACE_TIE]
        held.append(max(close, key=lambda step: step.sharpness))
    return sorted(held, key=lambda step: (step.parted, step.widened))


def rung_name(parted: float, widened: float) -> str:
    """A name that says where the rung sits, and no two of them the same.

    **Three decimals, not two, and that is a 2026-08-28 fix.** The name was
    ``round(x * 100)`` on both axes, which is coarser than the grid: at a step
    of 0.035 by 0.04 two rungs from different cells rounded to the same name and
    the second overwrote the first. It cost three rungs of a 268-rung build —
    silently, because the manifest still listed both and one of them then
    described a photograph it was not.
    """
    return f"speech_{round(parted * 1000):04d}_{round(widened * 1000):+05d}.png"


# The face width a rung is written at, as a multiple of the portrait's own.
#
# **One, and it is not a compromise.** `engine_warp._reduce_to` already shrinks
# every source to exactly the portrait's face width at load, with `INTER_AREA`,
# because "every source is warped onto the base portrait's landmark layout, so
# the base fixes the working resolution". A rung written at 4K is therefore
# decoded, reduced, and the reduction thrown away again on the next run.
#
# Measured on the grid as it stood: each rung is a **2160x2673 PNG of 5.7 MB**,
# 67 of them for 273 MB, and the avatar takes **24 seconds to load**. The
# portrait's face is 455 px. Writing the rung at that scale is the same
# operation the engine performs anyway, done once, and it is what makes a grid
# of two hundred rungs cost less than a grid of sixty-seven did.
WRITE_FACE_SCALE = 1.0


def _written(candidate: Any, portrait_face: float) -> np.ndarray:
    """The rung as it goes to disk: cropped, and reduced to the portrait's scale.

    Not resized *up*: a source smaller than the portrait is already the day-15
    complaint — *"the beard colour differs"* — and stretching it here would only
    hide where that came from.
    """
    import cv2

    image = crop_around_face(candidate)
    wanted = portrait_face * WRITE_FACE_SCALE
    ratio = candidate.face_width / max(wanted, 1e-6)
    if ratio < 1.0:
        return image
    scale = 1.0 / ratio
    reduced: np.ndarray = cv2.resize(
        image,
        (max(int(image.shape[1] * scale), 1), max(int(image.shape[0] * scale), 1)),
        interpolation=cv2.INTER_AREA,
    )
    return reduced


def install(chosen: list[Step], avatar: Path, portrait_face: float) -> None:
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
        name = rung_name(step.parted, step.widened)
        cv2.imwrite(str(avatar / name), _written(step.candidate, portrait_face))
        meta["assets"].append(
            {
                "path": name,
                "role": ROLE,
                "optional": True,
                "speech": round(step.parted, 3),
                "speech_width": round(step.widened, 3),
            }
        )

    manifest.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"installed {len(chosen)} rung(s); manifest now holds {len(meta['assets'])} assets")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--avatar", type=Path, default=REPO_ROOT / "avatars" / "my_avatar")
    parser.add_argument("--step", type=float, default=DEFAULT_STEP)
    parser.add_argument("--width-step", type=float, default=DEFAULT_WIDTH_STEP)
    parser.add_argument("--min-parted", type=float, default=DEFAULT_MIN_PARTED)
    parser.add_argument("--min-widened", type=float, default=DEFAULT_MIN_WIDENED)
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
        rest = _smiling(base_result)
        print(f"portrait: resting mouth {rest:.4f} of an eye span")

        found: list[Step] = []
        for folder in args.folders:
            # **A fresh tracker per folder**, because each folder is its own
            # clip and its timestamps start again at zero.
            # :func:`timestamp_of` reads the millisecond out of the file name,
            # so handing a second folder to the same tracker runs its clock
            # *backwards* — and MediaPipe in VIDEO mode reasons about how far a
            # face can have moved from the gap between timestamps.
            #
            # Measured, on this ladder: `video-10/0057_02850ms.jpg` reads a
            # parting of 0.35 when it follows the speech folder and 0.51 when it
            # is measured on its own. Two thirds of a band, on the frame that
            # then wins one. The other four builders take several folders the
            # same way and have the same fault; the installed eye and smile
            # ladders were both built from two.
            tracker.stop()
            tracker.start()
            frames = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
            measured = [
                measure(path, tracker, base_points, base_ratios, rest, timestamp_of(path, index))
                for index, path in enumerate(frames)
            ]
            usable = [
                step
                for step, yaw in (m for m in measured if m is not None)
                if (args.min_parted <= step.parted or args.min_widened <= abs(step.widened))
                and step.parted <= args.max_parted
                and abs(yaw) <= args.max_yaw
                and step.candidate.face_width >= args.min_face
                and step.drift <= args.max_drift
            ]
            refused = sum(1 for step in usable if step.refused is not None)
            print(
                f"{folder.name:<12} {len(frames):>4} frames  {len(usable):>4} moved and frontal"
                + (f"  ({refused} the engine would refuse)" if refused else "")
            )
            found.extend(usable)
    finally:
        tracker.stop()

    if not found:
        print("\nNo parted lips in any of those folders.")
        return 1

    chosen = best_per_cell(found, args.step, args.width_step)
    print(
        f"\n{len(chosen)} rung(s), one per {args.step:g} of parting by {args.width_step:g} of width"
    )
    print(f"{'parted':>7}{'widened':>9}{'lip gap':>9}{'sharp':>7}{'face px':>9}{'drift':>7}  frame")
    for step in chosen:
        print(
            f"{step.parted:>7.2f}{step.widened:>9.3f}{step.gap:>9.3f}{step.sharpness:>7.0f}"
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
        install(
            chosen,
            args.avatar,
            float(base_points[:, 0].max() - base_points[:, 0].min()),
        )
    else:
        print("\nNothing written. Pass --install to give the avatar a speech ladder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
