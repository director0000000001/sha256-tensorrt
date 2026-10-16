"""The render fitted to HIS outline, row by row -- a prototype on a capture
(day 22 evening, his "my eye, brow, cheek and lip show on the side I turn
to").

At a turn the network draws the head less turned than his and narrower on
the profile side, so between the drawn face's silhouette and his own
profile the paste shows his skin (the guard keeps only the drawn face).
Here the rendered crop is warped horizontally, row by row, so the drawn
face's extent on each row lands on the paste's own extent (his oval, with
the day-20 room): for a row with the drawn face on [dl, dr] and the paste
on [tl, tr], output column x reads the render at dl + (x - tl) * (dr - dl)
/ (tr - tl); the per-row scale and shift are smoothed over rows and the
scale is bounded. The result is pasted by the engine's own CPU paste and
written beside the untouched composite, with a zoom on the eyes.

Usage::

    uv run python scripts/probes/outline_fit.py --capture captures/moment-20261015-191804 \\
        --out-prefix recordings/day22-outline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "probes"))

from capture_replay import SETTLE_FRAMES, parse_options  # noqa: E402
from jaw_band import capture_anchor, tracking_at  # noqa: E402

from sha256.avatar.engine_liveportrait import (  # noqa: E402
    _BACKGROUND_NEAR,
    DEFAULT_WEIGHTS_RELATIVE_PATH,
    LivePortraitEngine,
)
from sha256.avatar.repository import FilesystemAvatarRepository  # noqa: E402
from sha256.config.loader import load_settings  # noqa: E402
from sha256.config.paths import AppPaths  # noqa: E402
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FrameMetadata, VideoFrame  # noqa: E402
from sha256.tracking.mediapipe_tracker import DEFAULT_MODEL_RELATIVE_PATH  # noqa: E402

SCALE_MIN, SCALE_MAX = 0.75, 1.35
SMOOTH_ROWS = 21  # odd


def row_extents(mask: np.ndarray, threshold: float) -> np.ndarray:
    """(H, 2) first/last column where ``mask`` > threshold per row; -1 where none."""
    hit = mask > threshold
    out = np.full((mask.shape[0], 2), -1, dtype=np.int32)
    rows = np.nonzero(hit.any(axis=1))[0]
    for r in rows:
        cols = np.nonzero(hit[r])[0]
        out[r] = (cols[0], cols[-1])
    return out


def fit_maps(
    drawn: np.ndarray,
    target: np.ndarray,
    size: int,
    near_left: bool,
    first_row: int,
    last_row: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """``map_x`` (H, W) float32 for cv2.remap (map_y the identity) that
    takes the drawn face's NEAR edge of each row (the profile side: the
    crop's left when ``near_left``) onto the target's near edge, scaling
    about the target's far edge, on rows ``first_row``..``last_row`` (the
    brows to the chin), fading to identity outside; rows with no target or
    no drawn face read straight. Returns the maps and a few numbers."""
    import cv2

    scale = np.ones(size, dtype=np.float32)
    shift = np.zeros(size, dtype=np.float32)
    valid = np.zeros(size, dtype=bool)
    gaps = []
    for r in range(max(first_row, 0), min(last_row, size - 1) + 1):
        dl, dr = drawn[r]
        tl, tr = target[r]
        if dl < 0 or tl < 0 or dr - dl < 8 or tr - tl < 8:
            continue
        pivot = float(tr if near_left else tl)
        d_near = float(dl if near_left else dr)
        t_near = float(tl if near_left else tr)
        if abs(pivot - t_near) < 8.0:
            continue
        s = (pivot - d_near) / (pivot - t_near)  # output x -> source pivot + (x - pivot) * s
        scale[r] = min(max(s, SCALE_MIN), SCALE_MAX)
        shift[r] = pivot - pivot * scale[r]
        valid[r] = True
        gaps.append(t_near - d_near)
    stats = {"rows": float(valid.sum())}
    if gaps:
        stats["gap_px_median"] = float(np.median(gaps))
        stats["gap_px_max"] = float(np.max(np.abs(gaps)))
    if valid.any():
        # fill the gaps from the nearest fitted row, then smooth
        idx = np.nonzero(valid)[0]
        first, last = idx[0], idx[-1]
        for r in range(size):
            if not valid[r]:
                j = idx[np.abs(idx - r).argmin()]
                scale[r], shift[r] = scale[j], shift[j]
        scale = cv2.GaussianBlur(scale.reshape(-1, 1), (1, SMOOTH_ROWS), 0).reshape(-1)
        shift = cv2.GaussianBlur(shift.reshape(-1, 1), (1, SMOOTH_ROWS), 0).reshape(-1)
        # fade to identity above the first and below the last fitted row
        for r in range(size):
            if r < first or r > last:
                d = min(abs(r - first), abs(r - last))
                w = max(0.0, 1.0 - d / 40.0)
                scale[r] = 1.0 + (scale[r] - 1.0) * w
                shift[r] = shift[r] * w
        stats["scale_min"] = float(scale[valid].min())
        stats["scale_max"] = float(scale[valid].max())
        stats["scale_median"] = float(np.median(scale[valid]))
    xs = np.arange(size, dtype=np.float32)[None, :]
    map_x = (xs * scale[:, None] + shift[:, None]).astype(np.float32)
    map_y = np.repeat(np.arange(size, dtype=np.float32)[:, None], size, axis=1)
    return map_x, map_y, stats


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out-prefix", type=Path, default=Path("recordings/day22-outline"))
    args = parser.parse_args()
    overrides = parse_options(args.option)

    data = np.load(args.capture / "tracking.npz")
    camera = cv2.imread(str(args.capture / "camera.png"))
    detected = data["detected"].astype(bool)
    last = int(np.nonzero(detected)[0][-1])
    anchor = capture_anchor(data)
    avatar_id = str(data["anchor_avatar"]) if "anchor_avatar" in data.files else "my_avatar_neural"

    paths = AppPaths.resolve()
    settings = load_settings(paths)
    models = paths.resolve_relative(settings.models.root_path)
    repo = FilesystemAvatarRepository(paths.resolve_relative(settings.avatars.library_path))
    engine = LivePortraitEngine(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        weights=models / DEFAULT_WEIGHTS_RELATIVE_PATH,
    )
    engine.start()
    avatar = repo.get(avatar_id)
    avatar.manifest.engine_options.update(overrides)
    frame = VideoFrame(
        data=camera,
        pixel_format=PixelFormat.BGR24,
        metadata=FrameMetadata(frame_id=last, capture_timestamp=float(data["timestamps"][last])),
    )
    reading = tracking_at(data, last)
    try:
        engine.load_avatar(avatar)
        if anchor is not None:
            engine.seed_anchor(avatar_id, anchor)
        out: Any = None
        for _ in range(SETTLE_FRAMES):
            out = engine.animate(avatar_id, reading, frame)
        composite = np.ascontiguousarray(out.data)
        loaded = engine._loaded[avatar_id]
        face, to_frame, mask, extra = loaded.last
        face_np = (
            face.detach().permute(1, 2, 0).cpu().numpy() if hasattr(face, "detach") else face
        )
        face_u8 = np.clip(face_np, 0, 255).astype(np.uint8)
        size = face_u8.shape[0]
        assert loaded.background is not None, "no flat background: nothing to fit against"
        not_background = (
            np.abs(face_u8.astype(np.float32) - loaded.background).max(axis=2) >= _BACKGROUND_NEAR
        )
        not_border = face_u8.max(axis=2) > 12  # the warp's black outside the photograph
        drawn_mask = (not_background & not_border).astype(np.float32)
        # the drawn face is the largest connected blob (the hair's stray
        # pixels and the shoulders are not the face)
        count, labels, stats_cc, _ = cv2.connectedComponentsWithStats(
            drawn_mask.astype(np.uint8), connectivity=4
        )
        if count > 2:
            biggest = 1 + int(np.argmax(stats_cc[1:, cv2.CC_STAT_AREA]))
            drawn_mask = (labels == biggest).astype(np.float32)
        paste = mask[:, :, 0] if extra is None else np.maximum(mask[:, :, 0], extra[:, :, 0])
        drawn = row_extents(drawn_mask, 0.5)
        target = row_extents(paste, 0.5)
        # his points in the crop: the inverse of the paste's affine
        h, w = camera.shape[:2]
        pts = reading.landmarks.points[:, :2] * np.array([w, h], dtype=np.float32)
        inv = cv2.invertAffineTransform(to_frame.astype(np.float64))
        crop_pts = pts @ inv[:, :2].T + inv[:, 2]
        brow_row = int(crop_pts[[46, 53, 52, 65, 55, 107, 276, 283, 282, 295, 285, 336], 1].min())
        chin_row = int(crop_pts[152, 1])
        near_left = float(reading.head_pose.yaw) < 0.0  # his right: the profile on the crop's left
        map_x, map_y, stats = fit_maps(drawn, target, size, near_left, brow_row, chin_row)
        stats["brow_row"], stats["chin_row"], stats["near_left"] = brow_row, chin_row, near_left
        print(f"{args.capture.name}: fitted rows {stats}")
        fitted = cv2.remap(face_u8, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        scene = np.ascontiguousarray(camera)
        before = engine._paste(scene, face_u8, to_frame, loaded, mask, extra=extra)
        after = engine._paste(scene, fitted, to_frame, loaded, mask, extra=extra)
        prefix = args.out_prefix.parent / f"{args.out_prefix.name}-{args.capture.name}"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{prefix}-before.png", before)
        cv2.imwrite(f"{prefix}-after.png", after)
        cv2.imwrite(f"{prefix}-gpu.png", composite)
        # a diagnostic of the two outlines on the crop
        diag = cv2.cvtColor(face_u8, cv2.COLOR_BGR2GRAY)
        diag = cv2.cvtColor(diag, cv2.COLOR_GRAY2BGR)
        for r in range(0, size, 2):
            if drawn[r, 0] >= 0:
                cv2.circle(diag, (int(drawn[r, 0]), r), 1, (0, 0, 255), -1)
                cv2.circle(diag, (int(drawn[r, 1]), r), 1, (0, 0, 255), -1)
            if target[r, 0] >= 0:
                cv2.circle(diag, (int(target[r, 0]), r), 1, (0, 255, 0), -1)
                cv2.circle(diag, (int(target[r, 1]), r), 1, (0, 255, 0), -1)
        cv2.imwrite(f"{prefix}-outlines.png", diag)
        print(f"    stills: {prefix}-before/after/gpu/outlines.png (his face; delete after)")
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
