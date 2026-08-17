"""Build the avatar's turn ladder out of dumped frames, and install it.

``extract_avatar_frames.py`` walks a *video* and rejects a great deal on the
way: too small a face, identity ratios that drifted, a frame softer than its
neighbours. Every one of those filters is right for picking the best open mouth
and wrong at a deep profile, where a face is narrower, its measured proportions
genuinely change, and the near eye — the thing sharpness is measured on — is
foreshortened to almost nothing. The result was a ladder that stopped at 43
degrees while the clip it came from held a sharp profile at 54, and the avatar
stretched a 32-degree photograph across a 40-degree head. The owner saw the
nose smear into dough and said so three times.

So this reads the frames themselves, from ``dump_clip_frames.py``, and asks one
question of each: **what angle is this face at, and is it the sharpest thing at
that angle?** One rung per band of yaw, the best in each band, and nothing
thrown away for being at an angle where faces are hard.

What it does *not* do is judge across bands. A profile is legitimately less
sharp on the near eye than a frontal frame, so comparing them is comparing a
foreshortened lash line with a square one. Each band's number is printed
instead, and a band whose best is poor is visible rather than silently missing.

Usage::

    python scripts/build_turn_ladder.py avatars/my_avatar/frames/video-06
    python scripts/build_turn_ladder.py .../video-06 .../video-05 --install
    python scripts/build_turn_ladder.py .../video-06 --step 2 --install
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
    eye_sharpness,
    sharpness_of,
)

from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FrameMetadata, VideoFrame  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)

# One rung per this many degrees of yaw, matching the engine's own spacing.
DEFAULT_STEP = 3.0

# A face narrower than this is too small to carry detail, and is the one filter
# worth keeping: everything else the video pass rejects on is a property of
# being turned rather than of being bad.
DEFAULT_MIN_FACE = 380.0

# Where the ladder ends. Past this the tracker's own landmarks stop being
# trustworthy — it reports a face at 70 degrees with the far half invented.
DEFAULT_MAX_YAW = 62.0

# What counts as a sharp rung, against the sharpest in the set, and how far a
# soft one has to be from every sharp one before it is worth having.
#
# The owner, after the first ladder built this way: turning one direction got
# *worse*. It had, and the reason is that a soft rung two degrees from a sharp
# one is pure loss — stretching a sharp photograph by two degrees costs almost
# nothing, and a smeared one costs everything. Eight degrees away it is the
# other way round: that is where a stretched photograph starts to tear, and a
# soft photograph at the right angle beats a sharp one at the wrong angle.
#
# So soft rungs are kept only where they are the only thing there. Measured on
# the owner's set, this drops the 1-4% rungs at 35 and 38 degrees, which sat
# beside a 95% rung at 32, and keeps the ones from 42 outward.
DEFAULT_SHARP_FRACTION = 0.15
DEFAULT_SOFT_GAP = 8.0


@dataclass(frozen=True)
class Rung:
    path: Path
    yaw: float
    detail: float
    face_width: float
    candidate: Candidate


def measure(path: Path, tracker: MediaPipeFaceTracker) -> Rung | None:
    """Angle and sharpness for one frame, or ``None`` if it holds no face."""
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
    width = float(points[:, 0].max() - points[:, 0].min()) * image.shape[1]
    candidate = Candidate(
        index=0,
        image=image,
        tracking=result,
        face_width=width,
        sharpness=sharpness_of(image, points),
        identity_drift=0.0,
    )
    return Rung(
        path=path,
        yaw=float(result.head_pose.yaw),
        detail=eye_sharpness(candidate),
        face_width=width,
        candidate=candidate,
    )


def rung_name(yaw: float) -> str:
    """The file a rung at this angle is written to. Also its identity."""
    return f"turn_{'m' if yaw < 0 else 'p'}{abs(yaw):04.1f}".replace(".", "_") + ".png"


def best_per_band(rungs: list[Rung], step: float) -> list[Rung]:
    """The sharpest frame in each band of yaw, ordered by angle.

    Deduplicated by the name each would be written to as well as by band: two
    frames a tenth of a degree apart can fall either side of a band boundary
    and then land on the same file, which would leave the manifest naming one
    picture twice and counting it as two rungs.
    """
    held: dict[int, Rung] = {}
    for rung in rungs:
        band = round(rung.yaw / step)
        if band not in held or rung.detail > held[band].detail:
            held[band] = rung

    by_name: dict[str, Rung] = {}
    for rung in held.values():
        name = rung_name(rung.yaw)
        if name not in by_name or rung.detail > by_name[name].detail:
            by_name[name] = rung
    return sorted(by_name.values(), key=lambda rung: rung.yaw)


def drop_needless_soft(chosen: list[Rung], sharp_fraction: float, gap: float) -> list[Rung]:
    """Keep every sharp rung, and a soft one only where nothing sharp is near.

    A soft rung beside a sharp one is pure loss: stretching a sharp photograph
    two degrees costs nothing a viewer can see, and a smeared photograph costs
    everything. Far from any sharp rung the trade reverses, because that is
    where stretching starts to tear. See :data:`DEFAULT_SOFT_GAP`.
    """
    if not chosen:
        return chosen

    sharpest = max(rung.detail for rung in chosen)
    floor = sharpest * sharp_fraction
    sharp = [rung for rung in chosen if rung.detail >= floor]
    if not sharp:
        return chosen

    kept = list(sharp)
    kept += [
        rung
        for rung in chosen
        if rung.detail < floor and min(abs(rung.yaw - other.yaw) for other in sharp) > gap
    ]
    return sorted(kept, key=lambda rung: rung.yaw)


def install(chosen: list[Rung], avatar: Path, tracker: MediaPipeFaceTracker) -> None:
    """Write the rungs into the avatar and replace the manifest's turn entries.

    **The old rungs go first.** A rung is chosen by the angle on its manifest
    entry, so one left behind competes with the new ones on equal terms — and
    the ones being replaced are the blurred ladder that started all of this.

    **Every rung is read back before it is kept.** The engine finds each source's
    own landmarks at load and drops what it cannot read, silently but for a line
    in the log — and cropping to the face is exactly what makes a deep profile
    unreadable: the tracker loses a face it had found comfortably in the whole
    frame. Four of the sharpest profile rungs went that way on the first run.
    So a crop that cannot be read is written out whole instead, which costs a
    larger file and keeps the angle.
    """
    import cv2

    manifest = avatar / "metadata.json"
    shutil.copy2(manifest, manifest.with_suffix(".json.bak"))
    meta = json.loads(manifest.read_text(encoding="utf-8"))

    retired = avatar / "superseded-ladder"
    retired.mkdir(exist_ok=True)
    moved = 0
    for old in avatar.glob("turn_*.png"):
        shutil.move(str(old), str(retired / old.name))
        moved += 1
    before = len(meta["assets"])
    meta["assets"] = [a for a in meta["assets"] if a.get("role") != "source_turn"]
    print(f"\nretired {moved} file(s), {before - len(meta['assets'])} manifest entrie(s)")

    whole = 0
    for rung in chosen:
        name = rung_name(rung.yaw)
        destination = avatar / name
        cv2.imwrite(str(destination), crop_around_face(rung.candidate))
        if measure(destination, tracker) is None:
            cv2.imwrite(str(destination), rung.candidate.image)
            whole += 1
        meta["assets"].append(
            {"path": name, "role": "source_turn", "optional": True, "yaw": round(rung.yaw, 1)}
        )
    if whole:
        print(f"{whole} rung(s) kept whole rather than cropped, to stay readable")

    manifest.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"installed {len(chosen)} rung(s); manifest now holds {len(meta['assets'])} assets")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+", help="Folders of dumped frames.")
    parser.add_argument("--avatar", type=Path, default=REPO_ROOT / "avatars" / "my_avatar")
    parser.add_argument("--step", type=float, default=DEFAULT_STEP)
    parser.add_argument("--min-face", type=float, default=DEFAULT_MIN_FACE)
    parser.add_argument("--max-yaw", type=float, default=DEFAULT_MAX_YAW)
    parser.add_argument("--sharp-fraction", type=float, default=DEFAULT_SHARP_FRACTION)
    parser.add_argument("--soft-gap", type=float, default=DEFAULT_SOFT_GAP)
    parser.add_argument("--install", action="store_true", help="Write them into the avatar.")
    args = parser.parse_args(argv)

    model = REPO_ROOT / "models" / DEFAULT_MODEL_RELATIVE_PATH
    if not model.is_file():
        print(f"Model bundle missing: {model}")
        return 1

    tracker = MediaPipeFaceTracker(model_path=model)
    tracker.start()
    try:
        found: list[Rung] = []
        for folder in args.folders:
            frames = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
            measured = [measure(path, tracker) for path in frames]
            kept = [rung for rung in measured if rung is not None]
            usable = [
                rung
                for rung in kept
                if rung.face_width >= args.min_face and abs(rung.yaw) <= args.max_yaw
            ]
            print(
                f"{folder.name:<12} {len(frames):>4} frames  "
                f"{len(frames) - len(kept):>3} no face  "
                f"{len(kept) - len(usable):>3} too small or too far round  "
                f"{len(usable):>4} usable"
            )
            if usable:
                print(
                    f"{'':<12} yaw {min(r.yaw for r in usable):+6.1f} .. "
                    f"{max(r.yaw for r in usable):+6.1f}"
                )
            found.extend(usable)

        if not found:
            print("\nNothing usable in any of those folders.")
            return 1

        every = best_per_band(found, args.step)
        chosen = drop_needless_soft(every, args.sharp_fraction, args.soft_gap)
        dropped = len(every) - len(chosen)
        if dropped:
            print(
                f"\ndropped {dropped} soft rung(s) within {args.soft_gap:g} degrees of a "
                f"sharp one — stretching the sharp one that far is the better trade"
            )
        sharpest = max(rung.detail for rung in chosen)
        print(f"\n{len(chosen)} rung(s), one per {args.step:g} degrees")
        print(f"{'yaw':>7}{'detail':>9}{'face px':>9}  frame")
        for rung in chosen:
            print(
                f"{rung.yaw:>7.1f}{rung.detail / sharpest:>8.0%}{rung.face_width:>9.0f}  "
                f"{rung.path.parent.name}/{rung.path.name}"
            )

        # Inside the tracker's lifetime: every rung is read back after it is
        # written, and the tracker is what reads it.
        if args.install:
            install(chosen, args.avatar, tracker)
        else:
            print("\nNothing written. Pass --install to replace the avatar's ladder.")
    finally:
        tracker.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
