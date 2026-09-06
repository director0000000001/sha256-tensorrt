"""Per-frame ratios of one capture's 300 tracked frames: his EAR, MAR,
mouth width, brow, gaze x/y and pose -- what the engine would have read."""

import sys

import numpy as np

from sha256.avatar.engine_liveportrait import (
    _brow_ratio,
    _eye_ratios,
    _gaze_offset,
    _lip_ratio,
    _mouth_open_ratio,
    _mouth_width_ratio,
)

W, H = 1280, 720
d = sys.argv[1]
z = np.load(d + "/tracking.npz", allow_pickle=True)
lm, pose, det, ts = z["landmarks"], z["pose"], z["detected"], z["timestamps"]
rows = []
for i in range(len(det)):
    if not det[i]:
        continue
    p = lm[i][:, :2] * (W, H)
    left, right = _eye_ratios(p)
    g = _gaze_offset(p)
    rows.append(
        (
            ts[i] - ts[0],
            (left + right) / 2,
            left,
            right,
            _lip_ratio(p),
            _mouth_open_ratio(p),
            _mouth_width_ratio(p),
            _brow_ratio(p),
            g[0] if g else np.nan,
            g[1] if g else np.nan,
            *pose[i],
        )
    )
a = np.array(rows)
names = [
    "t",
    "ear",
    "earL",
    "earR",
    "mar",
    "opn",
    "width",
    "brow",
    "gaze_x",
    "gaze_y",
    "yaw",
    "pitch",
    "roll",
]  # opn: the day-12 gate opening, over the jaw width  # npz pose rows are (yaw, pitch, roll)
print("n", len(a))
for j, n in enumerate(names[1:], 1):
    print(
        f"{n:>7}: min {a[:, j].min():7.3f} p10 {np.percentile(a[:, j], 10):7.3f} "
        f"med {np.median(a[:, j]):7.3f} p90 {np.percentile(a[:, j], 90):7.3f} "
        f"max {a[:, j].max():7.3f}"
    )
print("\nlast 4 seconds, every 5th frame:")
print(" ".join(f"{n:>6}" for n in names))
for row in a[a[:, 0] > a[-1, 0] - 4][::5]:
    print(" ".join(f"{v:6.2f}" for v in row))
print(
    "\nframes with ear < 0.7*median:", int((a[:, 1] < 0.7 * np.median(a[:, 1])).sum()), "of", len(a)
)
print("frames with mar > median+0.03:", int((a[:, 4] > np.median(a[:, 4]) + 0.03).sum()))
print("frames with opening > median+0.012:", int((a[:, 5] > np.median(a[:, 5]) + 0.012).sum()))
print("frames with width > median+0.05:", int((a[:, 6] > np.median(a[:, 6]) + 0.05).sum()))
print("corr(ear, gaze_y):", np.corrcoef(a[:, 1], a[:, 9])[0, 1])
