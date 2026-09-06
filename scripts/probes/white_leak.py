"""Near-white pixels on the avatar half of a side-by-side still that are not
near-white on the camera half at the same place -- the portrait's white
background showing through the paste mask. Threshold 200 (the room has
nothing that bright except the ceiling strip, which the camera half cancels)."""

import glob
import sys

import cv2
import numpy as np

thr = int(sys.argv[2]) if len(sys.argv) > 2 else 200
for f in sorted(glob.glob(sys.argv[1])):  # noqa: PTH207 -- a shell pattern
    im = cv2.imread(f)
    h, w = im.shape[:2]
    half = w // 2
    left, right = im[:, :half], im[:, half : half * 2]
    wl = left.min(axis=2) > thr
    wr = right.min(axis=2) > thr
    leak = wr & ~wl
    ys, xs = np.nonzero(leak)
    box = "" if not len(xs) else f" box x {xs.min()}-{xs.max()} y {ys.min()}-{ys.max()}"
    print(
        f"{f.split('/')[-1]:>34}: leak px {int(leak.sum()):5d} "
        f"(avatar half {int(wr.sum())}, camera half {int(wl.sum())}){box}"
    )
