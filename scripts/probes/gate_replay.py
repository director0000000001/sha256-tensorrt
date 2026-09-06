"""Day 11, item 3 (day 12: the gates read _mouth_open_ratio, the opening
over the jaw width; the anchor still reads _lip_ratio): the engine's own
gate logic replayed on his LIVE tracking
(captures/moment-*/tracking.npz -- 300 frames of landmarks + pose, no
imagery), the day-9-night method. The clips cannot show live gate
flicker (their jitter is 3-7x lower); this can.

Per capture: rest from the first accepted frames, then
LivePortraitEngine._gates() per frame on a bare _Portrait with the real
avatar's options, with the capture's own frame timing (last_dt). Reports
flips/second of the mouth and opening gates (a flip = crossing 0.5), the
share of "mouth moving but lips held" frames (|dMAR| > activity unit while
opening < 0.8) and "mouth shut but lips driven" frames (MAR < rest + 0.0024
while opening > 0.2) -- the two numbers a threshold change trades against
each other -- for: the gates as they are; an EMA on the gate scalars
(preview of a gate_smoothing option); the opening thresholds lowered.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

import sha256.avatar.engine_liveportrait as eng
from sha256.avatar.engine_liveportrait import (
    LivePortraitEngine,
    _anchor_frame_ok,
    _brow_ratio,
    _eye_ratios,
    _lip_ratio,
    _mouth_open_ratio,
    _mouth_width_ratio,
    _Portrait,
)
from sha256.avatar.repository import FilesystemAvatarRepository
from sha256.config.loader import load_settings
from sha256.config.paths import AppPaths
from sha256.core.types import HeadPose

W, H = 1280, 720


def replay(descriptor, z, *, ema: float = 0.0, gate_mar=None, activity_mar=None):
    saved = (eng._MOUTH_GATE_MAR, eng._MOUTH_ACTIVITY_MAR)
    if gate_mar is not None:
        eng._MOUTH_GATE_MAR = gate_mar
    if activity_mar is not None:
        eng._MOUTH_ACTIVITY_MAR = activity_mar
    try:
        lm, pose, det, ts = z["landmarks"], z["pose"], z["detected"], z["timestamps"]
        loaded = _Portrait(descriptor)
        # rest: the first 30 accepted frames, as the anchor would take them
        rest = []
        for i in range(len(det)):
            if not det[i]:
                continue
            p = lm[i][:, :2] * (W, H)
            ear = sum(_eye_ratios(p)) / 2
            mar = _lip_ratio(p)  # the anchor's own measure, over the mouth width
            hp = HeadPose(
                yaw=float(pose[i][0]), pitch=float(pose[i][1]), roll=float(pose[i][2])
            )  # npz pose rows are (yaw, pitch, roll)
            if _anchor_frame_ok(hp, ear, mar):
                rest.append((_mouth_open_ratio(p), _mouth_width_ratio(p), _brow_ratio(p)))
            if len(rest) >= 30:
                break
        if len(rest) < 10:
            return None
        loaded.rest_mar = float(np.mean([r[0] for r in rest]))
        loaded.rest_mouth_width = float(np.mean([r[1] for r in rest]))
        loaded.rest_brow = float(np.mean([r[2] for r in rest]))
        mouth_g, open_g, mars, dts = [], [], [], []
        sm_m = sm_o = None
        for i in range(len(det)):
            if not det[i]:
                continue
            p = lm[i][:, :2] * (W, H)
            mar = _mouth_open_ratio(p)  # day 12: the gates read the opening over the jaw width
            width = _mouth_width_ratio(p)
            brow = _brow_ratio(p)
            hp = HeadPose(
                yaw=float(pose[i][0]), pitch=float(pose[i][1]), roll=float(pose[i][2])
            )  # npz pose rows are (yaw, pitch, roll)
            if i > 0:
                loaded.last_dt = float(min(max(ts[i] - ts[i - 1], 1e-3), 0.2))
                dts.append(loaded.last_dt)
            m, o, _ = LivePortraitEngine._gates(loaded, mar, width, brow, hp)
            if ema > 0.0:
                sm_m = m if sm_m is None else sm_m * ema + m * (1 - ema)
                sm_o = o if sm_o is None else sm_o * ema + o * (1 - ema)
                m, o = sm_m, sm_o
            mouth_g.append(m)
            open_g.append(o)
            mars.append(mar)
        mouth_g, open_g, mars = np.array(mouth_g), np.array(open_g), np.array(mars)
        seconds = float(ts[-1] - ts[0]) or 1.0
        dmar = np.abs(np.diff(mars))
        moving = dmar > eng._MOUTH_ACTIVITY_MAR
        shut = mars[1:] < loaded.rest_mar + 0.0024  # day 12 scale: day 11's 0.006 x 0.4

        def flips(g):
            return int(np.sum(np.abs(np.diff((g > 0.5).astype(int))))) / seconds

        return {
            "n": len(mars),
            "fps": len(mars) / seconds,
            "rest_mar": loaded.rest_mar,
            "dmar_p90": float(np.percentile(dmar, 90)),
            "dmar_p99": float(np.percentile(dmar, 99)),
            "mouth_flips": flips(mouth_g),
            "open_flips": flips(open_g),
            "moving_held": float(np.mean(open_g[1:][moving] < 0.8))
            if moving.any()
            else float("nan"),
            "n_moving": int(moving.sum()),
            "shut_driven": float(np.mean(open_g[1:][shut] > 0.2)) if shut.any() else float("nan"),
            "n_shut": int(shut.sum()),
            "open_mean": float(open_g.mean()),
        }
    finally:
        eng._MOUTH_GATE_MAR, eng._MOUTH_ACTIVITY_MAR = saved


def main() -> None:
    paths = AppPaths.resolve()
    settings = load_settings(paths)
    repo = FilesystemAvatarRepository(paths.resolve_relative(settings.avatars.library_path))
    descriptor = repo.get("my_avatar_neural")
    dirs = sys.argv[1:] or sorted(
        glob.glob("captures/moment-20260906-*"),  # noqa: PTH207 -- a shell pattern
        key=os.path.getmtime,
    )
    variants = [
        ("as is", {}),
        ("gate EMA 0.5", {"ema": 0.5}),
        ("gate EMA 0.7", {"ema": 0.7}),
        # day 12: thresholds are on the over-jaw-width opening (0.4x day 11's)
        ("opening 0.0024/0.008", {"gate_mar": (0.0024, 0.008)}),
        ("opening 0.0032/0.010", {"gate_mar": (0.0032, 0.010)}),
        ("activity 0.0028", {"activity_mar": 0.0028}),
        ("0.0032/0.010 + EMA 0.5", {"gate_mar": (0.0032, 0.010), "ema": 0.5}),
    ]
    for d in dirs:
        z = np.load(os.path.join(d, "tracking.npz"), allow_pickle=True)  # noqa: PTH118
        print(f"\n== {d}")
        first = True
        for name, kw in variants:
            r = replay(descriptor, z, **kw)
            if r is None:
                print("   (no usable rest frames)")
                break
            if first:
                print(
                    f"   {r['n']} frames at {r['fps']:.1f} fps, rest MAR {r['rest_mar']:.4f}, "
                    f"live |dMAR| p90 {r['dmar_p90']:.4f} p99 {r['dmar_p99']:.4f}; "
                    f"moving frames {r['n_moving']}, shut frames {r['n_shut']}"
                )
                print(
                    f"   {'variant':>22} | flips/s mouth  opening | "
                    "moving-but-held  shut-but-driven | opening mean"
                )
                first = False
            print(
                f"   {name:>22} | {r['mouth_flips']:6.2f} {r['open_flips']:8.2f} | "
                f"{100 * r['moving_held']:14.0f}% {100 * r['shut_driven']:15.0f}% | "
                f"{r['open_mean']:.2f}"
            )


if __name__ == "__main__":
    main()
