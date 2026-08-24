"""How much of the warp's error any refiner could remove, and how much it could not.

**The question this answers.** The first refiner took the held-out error from
9.11 to 8.76. Whether that is a poor result or most of what is available cannot
be read off the number, because *the number has no denominator*: some of the
9.11 is damage the warp did, and some of it is content that was never in the
source frame at all. Warping a face with its eyes open onto a mesh whose lids
are shut cannot produce a lid -- there is no lid in the input. No network reading
that input can invent the right one, and scoring it as though it should is how a
week gets spent on a ceiling.

**Three numbers bracket it**, and all three are measured on the same frames:

*   **The self-warp.** Frame B warped onto B's *own* mesh. Source and target
    geometry are identical, so nothing is stretched and nothing is missing: what
    is left is the warp machinery's own loss -- resampling, triangle edges,
    the border replication. **This is the floor no arrangement of source frames
    can go below.**

*   **One source.** What the engine does and what the refiner is trained on:
    frame A warped onto B. This is the 9.11.

*   **The consensus of many sources.** Every available source warped onto B, and
    the per-pixel median taken across them. Where several independent warps
    agree, they agree because the content was in all of them; where they
    disagree, at least some of them are artefacts, and the median throws those
    out. A refiner sees **one** warped image, so it has strictly less to work
    with than this median does -- which makes the consensus error an optimistic
    estimate of the best a refiner could do.

Read as: **consensus is roughly the ceiling, and (one source - consensus) is the
headroom a refiner is competing for.** If the headroom is a tenth of the error,
the refiner is close to done at 8.76 and the effort belongs elsewhere. If it is
most of it, the first refiner has barely started.

Usage::

    uv run python scripts/measure_refiner_ceiling.py
    uv run python scripts/measure_refiner_ceiling.py --stride 20 --clip video-07
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from make_training_pairs import (  # noqa: E402
    DEFAULT_MIN_FACE,
    DEFAULT_MIN_SHARPNESS,
    DEFAULT_OFFSETS,
    DEFAULT_SIZE,
    Frame,
    build_pair,
    read_clip,
)

from sha256.tracking.mediapipe_tracker import DEFAULT_MODEL_RELATIVE_PATH  # noqa: E402


def scores(prediction: np.ndarray, truth: np.ndarray, inside: np.ndarray) -> tuple[float, float]:
    """Mean absolute error over the face and over its worst 1%, in 0-255."""
    error = np.abs(prediction.astype(np.float32) - truth.astype(np.float32)).mean(axis=2)
    values = error[inside]
    return float(values.mean()), float(np.percentile(values, 99))


def tone_matched(prediction: np.ndarray, truth: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """``prediction`` with its per-channel mean shifted onto ``truth``'s.

    The crudest possible version of what ``_match_colour`` does in the engine,
    and it is here because the pairs do not go through ``_match_colour`` at all:
    ``build_pair`` stops at ``warp_mesh``, while ``_warp_onto`` fits a lighting
    plane before compositing. If a large part of the error is a flat exposure
    difference between two frames of a generated clip, then the refiner was
    being asked to learn a correction the engine already applies, and every
    number measured on those pairs is measuring the wrong thing.
    """
    shifted = prediction.astype(np.float32).copy()
    for channel in range(3):
        offset = truth[..., channel][inside].mean() - prediction[..., channel][inside].mean()
        shifted[..., channel] += offset
    return np.clip(shifted, 0.0, 255.0).astype(np.uint8)


def split_mask(mask: np.ndarray, ring: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """The mask's core and its outer ring, as two boolean maps.

    Asked because the floor turned out not to move when the source frame moved
    closer, which is not what "the source is missing content" predicts. The
    other candidate is the mask's own edge: it runs through hair, ears and the
    jaw's silhouette, where two frames of the same face differ by a strand and
    no warp of any kind can agree with the other. If the floor lives in the
    ring, it is a coverage fact rather than a refiner ceiling.
    """
    import cv2

    solid = (mask > 127).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring * 2 + 1, ring * 2 + 1))
    core = cv2.erode(solid, kernel) > 0
    return core, (solid > 0) & ~core


def measure_target(
    target: Frame, sources: list[Frame], size: int
) -> tuple[float, float, float, float, float, float, int, float, float, float, float] | None:
    """Self-warp, single-source and consensus scores for one target frame."""
    itself = build_pair(target, target, size)
    if itself is None:
        return None
    self_warped, truth, mask, _, _ = itself
    inside = mask > 127
    if not bool(inside.any()):
        return None
    self_mean, self_worst = scores(self_warped, truth, inside)

    warped: list[np.ndarray] = []
    singles: list[tuple[float, float]] = []
    toned: list[tuple[float, float]] = []
    for source in sources:
        built = build_pair(source, target, size)
        if built is None:
            continue
        image = built[0]
        warped.append(image)
        singles.append(scores(image, truth, inside))
        toned.append(scores(tone_matched(image, truth, inside), truth, inside))

    if len(warped) < 3:
        return None

    # The per-pixel median across every source's warp. Median rather than mean:
    # a mean is dragged by one badly sheared triangle, and the whole point is to
    # find what the sources *agree* on.
    consensus = np.median(np.stack(warped).astype(np.float32), axis=0).astype(np.uint8)
    consensus_mean, consensus_worst = scores(consensus, truth, inside)

    core, ring = split_mask(mask)
    core_mean = scores(consensus, truth, core)[0] if bool(core.any()) else float("nan")
    ring_mean = scores(consensus, truth, ring)[0] if bool(ring.any()) else float("nan")

    return (
        self_mean,
        self_worst,
        statistics.mean(s[0] for s in singles),
        statistics.mean(s[1] for s in singles),
        consensus_mean,
        consensus_worst,
        len(warped),
        core_mean,
        ring_mean,
        statistics.mean(t[0] for t in toned),
        statistics.mean(t[1] for t in toned),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames", type=Path, default=REPO_ROOT / "avatars" / "my_avatar" / "frames"
    )
    parser.add_argument("--clip", default=None, help="Only this clip; default is all of them.")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--min-face", type=float, default=DEFAULT_MIN_FACE)
    parser.add_argument("--min-sharpness", type=float, default=DEFAULT_MIN_SHARPNESS)
    parser.add_argument("--offsets", type=int, nargs="+", default=list(DEFAULT_OFFSETS))
    args = parser.parse_args(argv)

    model = REPO_ROOT / "models" / DEFAULT_MODEL_RELATIVE_PATH
    if not model.is_file():
        print(f"Model bundle missing: {model}")
        return 1

    folders = sorted(p for p in args.frames.iterdir() if p.is_dir())
    if args.clip:
        folders = [p for p in folders if p.name == args.clip]
    if not folders:
        print("No clip folders to measure.")
        return 1

    rows: list[tuple[float, ...]] = []
    for folder in folders:
        frames = read_clip(folder, model, args.min_face, args.min_sharpness)
        if len(frames) < 4:
            print(f"{folder.name:<14} {len(frames):>4} usable -- too few", flush=True)
            continue
        by_index = {frame.index: frame for frame in frames}

        measured = 0
        for target in frames[:: args.stride]:
            sources = [
                by_index[target.index + offset]
                for offset in args.offsets
                if target.index + offset in by_index
            ]
            if len(sources) < 3:
                continue
            result = measure_target(target, sources, args.size)
            if result is None:
                continue
            rows.append(result)
            measured += 1
        print(f"{folder.name:<14} {len(frames):>4} usable  {measured:>4} targets", flush=True)

    if not rows:
        print("\nNothing measured.")
        return 1

    columns = list(zip(*rows, strict=True))
    self_mean, self_worst = statistics.mean(columns[0]), statistics.mean(columns[1])
    one_mean, one_worst = statistics.mean(columns[2]), statistics.mean(columns[3])
    many_mean, many_worst = statistics.mean(columns[4]), statistics.mean(columns[5])
    per_target = statistics.mean(columns[6])

    print(
        f"\n{len(rows)} target frames, {per_target:.1f} sources each, "
        f"{args.size}x{args.size}, error in 0-255 inside the mask\n"
    )
    print(f"{'':<34}{'mean':>9}{'worst 1%':>11}")
    print(f"{'the warp machinery alone (self-warp)':<34}{self_mean:>9.2f}{self_worst:>11.2f}")
    print(f"{'one source, which is what ships':<34}{one_mean:>9.2f}{one_worst:>11.2f}")
    tone_mean, tone_worst = statistics.mean(columns[9]), statistics.mean(columns[10])
    print(f"{'one source, exposure removed':<34}{tone_mean:>9.2f}{tone_worst:>11.2f}")
    print(f"{'the consensus of every source':<34}{many_mean:>9.2f}{many_worst:>11.2f}")

    # The split the whole question was asked for. The consensus is an optimistic
    # ceiling, so the headroom below is optimistic too -- a refiner reading one
    # image should be expected to reach less of it, not more.
    headroom = one_mean - many_mean
    floor = many_mean
    print(f"\n{'':<34}{'mean':>9}{'worst 1%':>11}")
    print(f"{'headroom a refiner competes for':<34}{headroom:>9.2f}{one_worst - many_worst:>11.2f}")
    print(f"{'floor no single-input refiner beats':<34}{floor:>9.2f}{many_worst:>11.2f}")
    share = 100.0 * headroom / one_mean if one_mean > 1e-9 else float("nan")
    print(f"\nthe headroom is {share:.0f}% of the error the engine makes today.")

    # And where the floor is. The consensus error, split into the face's core
    # and the ring just inside the mask's edge, because the edge runs through
    # hair, ears and the jaw's silhouette -- where two frames of one face differ
    # by a strand and no warp can agree with any other.
    core = statistics.mean(columns[7])
    ring = statistics.mean(columns[8])
    print("\nwhere the floor sits, consensus error by part of the mask:")
    print(f"{'the face core':<34}{core:>9.2f}")
    print(f"{'the ring inside the mask edge':<34}{ring:>9.2f}")
    if ring > core * 1.5:
        print(
            "\nMost of the floor is at the mask's edge, not on the face. That is a\n"
            "coverage fact rather than a refiner ceiling -- read the core number as\n"
            "the one a refiner is really working against."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
