"""How two output frames differ, pixel by pixel -- the paste tolerance of
"The fast version" (ROADMAP, 2026-10-08) read as numbers.

For each pair of PNGs (the same frame from two runs) it prints the pixels
that differ at all, the largest difference, the 99th percentile over the
pixels that differ and over the whole frame, and the mean absolute
difference -- all in grey levels (0..255), per frame. The plan's bar for a
paste moved to the GPU: at most 2 at p99, at most 0.5 mean, and nothing
outside the paste box.

Usage::

    uv run python scripts/probes/still_diff.py a-f450.png b-f450.png
    uv run python scripts/probes/still_diff.py --prefix-a recordings/budget/a \\
        --prefix-b recordings/budget/b --frames 330,450,600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def compare(a_path: Path, b_path: Path) -> dict[str, float]:
    import cv2

    a = cv2.imread(str(a_path))
    b = cv2.imread(str(b_path))
    if a is None or b is None:
        raise SystemExit(f"cannot read {a_path} or {b_path}")
    if a.shape != b.shape:
        raise SystemExit(f"shapes differ: {a.shape} vs {b.shape}")
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    per_pixel = diff.max(axis=2)
    differing = per_pixel > 0
    count = int(differing.sum())
    ys, xs = np.nonzero(differing)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if count else None
    return {
        "pixels_differing": count,
        "fraction_differing": count / per_pixel.size,
        "max": float(per_pixel.max()),
        "p99_of_differing": float(np.percentile(per_pixel[differing], 99)) if count else 0.0,
        "p99_whole": float(np.percentile(per_pixel, 99)),
        "mean_abs_whole": float(diff.mean()),
        "mean_abs_differing": float(diff[differing].mean()) if count else 0.0,
        "box": box,  # type: ignore[dict-item]
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pair", nargs="*", type=Path, help="Two PNGs.")
    parser.add_argument("--prefix-a", type=Path, default=None)
    parser.add_argument("--prefix-b", type=Path, default=None)
    parser.add_argument(
        "--frames", default="", help="Frame ids, with --prefix-a/-b: <prefix>-f<N>.png"
    )
    args = parser.parse_args()

    pairs: list[tuple[Path, Path]] = []
    if len(args.pair) == 2:
        pairs.append((args.pair[0], args.pair[1]))
    if args.prefix_a is not None and args.prefix_b is not None:
        for raw in args.frames.split(","):
            if raw.strip():
                n = int(raw)
                pairs.append(
                    (
                        Path(f"{args.prefix_a}-f{n}.png"),
                        Path(f"{args.prefix_b}-f{n}.png"),
                    )
                )
    if not pairs:
        parser.error("give two PNGs, or --prefix-a/--prefix-b with --frames")

    worst_p99 = 0.0
    worst_mean = 0.0
    for a_path, b_path in pairs:
        r = compare(a_path, b_path)
        worst_p99 = max(worst_p99, r["p99_of_differing"])
        worst_mean = max(worst_mean, r["mean_abs_whole"])
        print(
            f"{a_path.name} vs {b_path.name}: {r['pixels_differing']} px differ "
            f"({100 * r['fraction_differing']:.2f}%), max {r['max']:.0f}, "
            f"p99 of differing {r['p99_of_differing']:.1f}, p99 whole {r['p99_whole']:.1f}, "
            f"mean |d| whole {r['mean_abs_whole']:.3f}, "
            f"mean |d| differing {r['mean_abs_differing']:.2f}, box {r['box']}"
        )
    print(f"worst p99 of differing: {worst_p99:.1f}; worst mean |d| whole: {worst_mean:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
