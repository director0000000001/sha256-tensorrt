"""Measure every rung a ladder installed against what the manifest claims it is.

**Why this is a committed tool rather than a scratch script.** Three ladders were
built out of numbers that did not describe their photographs, and each time it
was found by hand, afterwards, by measuring the rungs one at a time:

    eyes    2026-08-21   6 rungs filed 0.12..0.56 were one photograph of a shut eye
    brows   2026-08-21   5 of 7 "raised" rungs are photographs of LOWERED brows
    mouth   2026-08-21   smile_140 (+1.401) shows LESS smile than smile_108

Every one of those was installed, shipped, and only then checked. A builder
chooses a frame by a reading taken during the clip; what ends up on disk is a
crop of that frame, and nothing verified that the crop still reads the same. So
this asks one question of an installed ladder -- *does each photograph show what
its entry says it shows* -- and answers it in the only way that counts, by
re-detecting the file that was written.

It is also the gate. ``--max-error`` exits non-zero when the filing is worse
than a threshold, so a bad ladder fails a check instead of reaching the owner.

``--refile`` rewrites the manifest from the measurements, which is only sound
when the axis survives a re-detection. The brow axis did not until 2026-08-24 --
its reading came off the nose bridge, which moves more across a crop than the
brows do -- and re-filing it then made the ladder worse, 0.442 to 0.545. On the
eye-line axis it moves 1.9% of its span, and re-filing is the whole point:
a photograph can be measured, so a ladder can be corrected without a new clip.

``--refuse`` retires named rungs whatever they measure. It was written for a
defect it then failed to fix, and both halves of that are worth keeping.

The owner, 2026-08-24: *"when I raise my brows all the way the brow disappears
completely -- it looks like they are shaved."* **Reproduced**, which is itself a
first: frames 510-513 of `driving.mkv`, his brows plainly raised and the avatar
with none. And seemingly localised, by compositing the brow region at full
weight -- from the portrait through `brows@81` the brow is a solid dark bar, and
on `brows@82` and `brows@83` it is gone.

**Four hypotheses, all wrong.** The brow is inside the region's box (240 px
against a box starting at 227). The mask does not fade over it (0.77 to 1.00 at
each rung's own brow). Its darkness at the landmarks matches the portrait's
(82.9 against 86.5). And "brow mass" -- the one that looked decisive -- separated
the two rungs cleanly under one parameterisation and then gave three different
answers under three others: sized from each photograph's own brow-to-eye gap it
accused the *lowered* rungs instead, because the raised rungs have wider-open
eyes, so a wider band, so more forehead inside it. A measurement that finds
whatever it is pointed at is not a measurement, and none of it shipped.

**Then the two rungs were refused and the render did not change.** Same frames,
same amounts, same missing brow with the ladder ending at +0.002. So they were
never the cause; they were put back, and the defect is still open. The remaining
candidate, unmeasured: this avatar's brows are light ginger, the owner's room is
dim and warm, and ``_match_colour`` fits the avatar to that light -- which would
cost brow-against-skin contrast everywhere, not only at a raise.

Kept because the option is sound and the record is the point: two renders
agreeing localised nothing here, and the check that would catch this
automatically is still owed.

Usage::

    uv run python scripts/check_ladder.py brows
    uv run python scripts/check_ladder.py brows --max-error 0.15
    uv run python scripts/check_ladder.py brows --refile
    uv run python scripts/check_ladder.py brows --refile --refuse brows_u089.png
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from extract_avatar_frames import sharpness_of  # noqa: E402

from sha256.avatar.engine_warp import _BROW_SPAN, _brow_height  # noqa: E402
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FaceTrackingResult, FrameMetadata, VideoFrame  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)

# Rungs closer together than this are the same rung. A ladder is chosen from by
# nearest amount, so two photographs a fiftieth apart are one photograph filed
# twice and the second only costs a load and a cache slot. The brow ladder had
# five inside 0.04 of each other on 2026-08-24.
DEFAULT_BAND = 0.05


@dataclass(frozen=True)
class Family:
    """One expression axis, and everything needed to read it off a still."""

    key: str
    """The manifest field holding the amount."""

    role: str
    prefix: str
    measure: Callable[[FaceTrackingResult], float]
    span: float
    """The engine's own full scale for this axis.

    Taken from the engine rather than recomputed from the library, and that is
    deliberate: the engine divides the *driver's* movement by this constant, so
    a rung filed against anything else is filed on a different scale from the
    one it will be chosen by. Recomputing it here would also mean every existing
    rung silently re-scaled the first time a new clip reached further than the
    old ones did.
    """

    retired_dir: str


FAMILIES = {
    "brows": Family(
        key="brows",
        role="source_brows_step",
        prefix="brows_",
        measure=_brow_height,
        span=_BROW_SPAN,
        retired_dir="superseded-brows",
    ),
}


def rung_name(prefix: str, amount: float) -> str:
    side = "u" if amount >= 0 else "d"
    return f"{prefix}{side}{round(abs(amount) * 100):03d}.png"


def read_still(path: Path, family: Family, model: Path) -> tuple[float, float] | None:
    """One photograph's raw reading and its sharpness, or ``None``.

    A tracker of its own per still, because a re-detection is exactly what is
    being tested: the tracker runs in video mode and carries a prior between the
    frames it is given, and reusing one would let the previous rung's face
    answer for this one. That is the defect that built the eye ladder out of the
    tracker's own lag on 2026-08-21.
    """
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        return None

    tracker = MediaPipeFaceTracker(model_path=model)
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
        return None
    return family.measure(result), sharpness_of(image, result.landmarks.points)


@dataclass
class Rung:
    path: str
    filed: float
    raw: float
    sharpness: float
    measured: float = 0.0

    @property
    def error(self) -> float:
        return abs(self.filed - self.measured)

    @property
    def side_is_wrong(self) -> bool:
        """Filed above rest while showing a face below it, or the reverse.

        Counted separately from the error because it is a different failure: a
        rung 0.3 out is a coarse ladder, and a rung on the wrong side of rest is
        an avatar doing the opposite of what the owner is doing.
        """
        return self.filed * self.measured < 0 and min(abs(self.filed), abs(self.measured)) > 0.02


def inversions(rungs: list[Rung]) -> tuple[int, int]:
    """Pairs the manifest orders one way and the photographs the other."""
    order = sorted(rungs, key=lambda r: r.filed)
    bad = sum(
        1
        for a in range(len(order))
        for b in range(a + 1, len(order))
        if order[a].measured > order[b].measured
    )
    return bad, len(order) * (len(order) - 1) // 2


def keep_one_per_band(rungs: list[Rung], band: float) -> tuple[list[Rung], list[Rung]]:
    """The sharpest rung in each band, and everything the band displaced.

    Sharpness decides, as it does in every builder: the amounts inside a band
    are equal by definition of the band, so the only thing left to choose on is
    which photograph the warp can use.
    """
    kept: dict[int, Rung] = {}
    for rung in sorted(rungs, key=lambda r: r.measured):
        slot = round(rung.measured / band)
        best = kept.get(slot)
        if best is None or rung.sharpness > best.sharpness:
            kept[slot] = rung
    chosen = sorted(kept.values(), key=lambda r: r.measured)
    # By identity: two rungs that measure the same and are equally sharp are
    # still two files, and `in` on a dataclass compares fields.
    dropped = [r for r in rungs if not any(r is c for c in chosen)]
    return chosen, dropped


def retire_into(directory: Path, source: Path) -> Path:
    """Move a file out of the way without ever overwriting what is already there.

    ``superseded-brows/`` already holds a ``brows_d093.png`` from an earlier
    rebuild, and the names a ladder generates repeat by construction -- they are
    the amount, and two ladders built from the same clips land on the same
    amounts. ``shutil.move`` onto an existing file copies over it. That is how
    ``brows_d100.png`` was lost on 2026-08-21: the photograph recovered in its
    place was a different one, and it had to be dropped rather than shipped
    under a number that did not describe it.
    """
    directory.mkdir(exist_ok=True)
    target = directory / source.name
    attempt = 2
    while target.exists():
        target = directory / f"{source.stem}-{attempt}{source.suffix}"
        attempt += 1
    shutil.move(str(source), str(target))
    return target


def refile(avatar: Path, family: Family, kept: list[Rung], dropped: list[Rung]) -> None:
    """Write the measured amounts back, rename to match, retire the duplicates.

    The rename matters and is not tidiness. The owner reads these names off the
    screen -- *"eyes_c014, eyes_c017, brows_d100, one of those is being drawn"*
    -- so a file called ``brows_u057`` holding a photograph of lowered brows
    costs an evening every time it is looked at.
    """
    manifest = avatar / "metadata.json"
    shutil.copy2(manifest, manifest.with_suffix(".json.bak"))
    meta = json.loads(manifest.read_text(encoding="utf-8"))

    retired = avatar / family.retired_dir
    for rung in dropped:
        source = avatar / rung.path
        if source.is_file():
            landed = retire_into(retired, source)
            print(f"  retired {source.name} -> {retired.name}/{landed.name}")
    print(f"retired {len(dropped)} duplicate rung(s)")

    meta["assets"] = [a for a in meta["assets"] if a.get(family.key) is None]

    # Renamed through a temporary name first: a re-filed ladder can want a name
    # another rung still holds, and a straight move would overwrite it. Losing a
    # photograph that way is how `brows_d100.png` was lost on 2026-08-21.
    staged: list[tuple[Path, str, float]] = []
    for index, rung in enumerate(kept):
        source = avatar / rung.path
        staging = avatar / f".refile-{index}.png"
        if source.is_file():
            shutil.move(str(source), str(staging))
        staged.append((staging, rung_name(family.prefix, rung.measured), rung.measured))

    for staging, name, amount in staged:
        destination = avatar / name
        if destination.exists():
            # Two kept rungs cannot share a name -- they are one per band, and a
            # band is wider than the rounding. If it happens anyway, stopping
            # loses nothing and overwriting loses a photograph.
            msg = f"two rungs both want the name {name}; refusing to overwrite"
            raise RuntimeError(msg)
        if staging.is_file():
            shutil.move(str(staging), str(destination))
        meta["assets"].append(
            {
                "path": name,
                "role": family.role,
                "optional": True,
                family.key: round(amount, 3),
            }
        )

    manifest.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"re-filed {len(kept)} rung(s); manifest now holds {len(meta['assets'])} assets")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=sorted(FAMILIES))
    parser.add_argument("--avatar", type=Path, default=REPO_ROOT / "avatars" / "my_avatar")
    parser.add_argument("--base", type=Path, default=None)
    parser.add_argument("--band", type=float, default=DEFAULT_BAND)
    parser.add_argument(
        "--max-error",
        type=float,
        default=None,
        help="Exit non-zero if the mean error is worse than this.",
    )
    parser.add_argument("--refile", action="store_true")
    parser.add_argument(
        "--refuse",
        nargs="+",
        default=(),
        metavar="RUNG",
        help=(
            "Retire these rungs by name, whatever they measure. For a rung "
            "refused on the evidence of a render rather than of a number -- see "
            "the note in the module docstring."
        ),
    )
    args = parser.parse_args(argv)

    family = FAMILIES[args.family]
    model = REPO_ROOT / "models" / DEFAULT_MODEL_RELATIVE_PATH
    if not model.is_file():
        print(f"Model bundle missing: {model}")
        return 1

    base_path = args.base or args.avatar / "portrait.png"
    base = read_still(base_path, family, model)
    if base is None:
        print(f"No face found in {base_path}.")
        return 1
    rest, _ = base
    if rest <= 1e-6:
        print(f"{base_path.name} reads {rest:.4f} at rest, which cannot be divided by.")
        return 1
    print(f"rest, from {base_path.name}: {rest:.4f}\n")

    meta = json.loads((args.avatar / "metadata.json").read_text(encoding="utf-8"))
    entries = sorted(
        (a for a in meta["assets"] if a.get(family.key) is not None),
        key=lambda a: a[family.key],
    )
    if not entries:
        print(f"The manifest holds no {args.family} rungs.")
        return 1

    rungs: list[Rung] = []
    for entry in entries:
        got = read_still(args.avatar / entry["path"], family, model)
        if got is None:
            print(f"{entry['path']:<18} no face found -- not measured")
            continue
        raw, sharp = got
        rungs.append(Rung(entry["path"], float(entry[family.key]), raw, sharp))

    if not rungs:
        print("Nothing could be measured.")
        return 1

    relative = [(r.raw - rest) / rest for r in rungs]
    for rung, rel in zip(rungs, relative, strict=True):
        rung.measured = max(-1.0, min(1.0, rel / family.span))

    # The library's own reach against the constant the engine divides by. They
    # describe the same thing -- how far a face goes -- measured on the avatar
    # and on the driver, so a wide disagreement means one of the two was
    # measured on the wrong face and every rung is on the wrong scale.
    reach = max(abs(min(relative)), abs(max(relative)))
    if reach > family.span * 1.15:
        print(
            f"WARNING: a rung reaches {reach:.3f} against a span of {family.span:.3f}, "
            f"so {sum(1 for r in rungs if abs(r.measured) >= 1.0)} rung(s) clamp at 1.0 "
            f"and are no longer distinguishable.\n"
        )

    print(f"{'rung':<18}{'filed':>8}{'measured':>10}{'error':>8}{'sharp':>9}  shows")
    for rung in rungs:
        shows = (
            "raised" if rung.measured > 0.02 else "lowered" if rung.measured < -0.02 else "at rest"
        )
        flag = "  <- wrong side of rest" if rung.side_is_wrong else ""
        print(
            f"{rung.path:<18}{rung.filed:>8.3f}{rung.measured:>10.3f}"
            f"{rung.error:>8.3f}{rung.sharpness:>9.0f}  {shows}{flag}"
        )

    mean_error = sum(r.error for r in rungs) / len(rungs)
    wrong = sum(1 for r in rungs if r.side_is_wrong)
    bad_pairs, pairs = inversions(rungs)

    print(f"\nmean error, filed against measured: {mean_error:.3f}")
    print(f"rungs on the wrong side of rest:    {wrong} of {len(rungs)}")
    print(f"out-of-order pairs:                 {bad_pairs} of {pairs}")
    print(
        f"the library reaches:                {min(relative):+.3f} .. {max(relative):+.3f}"
        f"   (the axis spans {family.span:.3f})"
    )

    refused = [r for r in rungs if r.path in set(args.refuse)]
    missing = set(args.refuse) - {r.path for r in rungs}
    if missing:
        print(f"\n--refuse names rungs the manifest does not hold: {sorted(missing)}")
        return 1
    if refused:
        print(f"\nrefused by name, {len(refused)} rung(s):")
        for rung in refused:
            print(f"  {rung.measured:>+7.3f}  {rung.path}")

    usable = [r for r in rungs if not any(r is x for x in refused)]
    if not usable:
        print("\nEvery rung was refused; nothing would be left of the ladder.")
        return 1

    kept, dropped = keep_one_per_band(usable, args.band)
    dropped = dropped + refused
    print(f"\none rung per {args.band:g} of the axis leaves {len(kept)} of {len(rungs)}:")
    for rung in kept:
        print(f"  {rung.measured:>+7.3f}  {rung.path}")
    if dropped:
        print(f"displaced, as duplicates of a band already filled: {len(dropped)}")
        for rung in dropped:
            print(f"  {rung.measured:>+7.3f}  {rung.path}")

    # The holes, which is what a re-filed ladder is honest about and a
    # mis-filed one hides: a gap wider than a band is a range of the owner's
    # face with no photograph of it.
    gaps = [
        (a.measured, b.measured)
        for a, b in pairwise(kept)
        if b.measured - a.measured > args.band * 2
    ]
    if gaps:
        print("\nholes wider than two bands -- these need a clip, not a change:")
        for low, high in gaps:
            print(f"  {low:+.3f} .. {high:+.3f}   ({high - low:.3f} wide)")

    if args.refile:
        refile(args.avatar, family, kept, dropped)

    if args.max_error is not None and mean_error > args.max_error:
        print(f"\nFAIL: mean error {mean_error:.3f} is worse than {args.max_error:.3f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
