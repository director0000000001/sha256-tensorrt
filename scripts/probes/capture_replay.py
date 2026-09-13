"""Replay a "That looked wrong" capture as the session itself, and read what
the avatar's mouth, eyes and brows did against his -- day 15's tool.

A capture (``PipelineRunner.capture``) holds the last seconds of tracking,
``camera.png`` (his face at the press), ``seen.png`` (what was on screen)
and, since day 15, the engine's own frozen anchor. The neural engine reads
pixels, so only the press's own frame can be re-rendered: the last tracked
frame's readings are fed with ``camera.png`` as the picture, enough times
for the engine's filters to settle, and the composite is read back by
MediaPipe in image mode -- his readings on ``camera.png``, the avatar's on
the replay (and on ``seen.png``, so a replay of the base build can be
checked against what he actually saw). With the anchor seeded the replay
*is* the session; an older capture learns an anchor from that one frame
if it is frontal enough, else runs against the portrait's own rest, and
the output says which.

``--option KEY=VALUE`` overrides engine options for the run; ``--sweep
KEY=v1,v2,...`` reloads the avatar per value in this one process (a load
is ~90 s, a reload a few seconds) and prints one row per value, so a knob
can be measured on the very frame he pressed on. Stills (his face) go
under ``--out-prefix``; delete them once looked at.

Usage::

    uv run python scripts/probes/capture_replay.py --capture captures/moment-20260911-081054
    uv run python scripts/probes/capture_replay.py --capture captures/moment-... \\
        --sweep mouth_opening_gain=1.0,0.8,0.6 --stills
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "probes"))

from jaw_band import capture_anchor, tracking_at  # noqa: E402
from measure_follow import anchors, measures  # noqa: E402

from sha256.avatar.engine_liveportrait import (  # noqa: E402
    _BROWS_UPPER,
    _MOUTH_CORNERS,
    _NOSE_TIP,
    DEFAULT_WEIGHTS_RELATIVE_PATH,
    FACE_POINTS,
    LivePortraitEngine,
    _mouth_open_ratio,
)
from sha256.avatar.repository import FilesystemAvatarRepository  # noqa: E402
from sha256.config.loader import load_settings  # noqa: E402
from sha256.config.paths import AppPaths  # noqa: E402
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FrameMetadata, VideoFrame  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)

# Frames of the press's own reading fed before the one that is measured:
# the crop's One-Euro, the size EMA (0.85 a frame) and the gates' holds all
# settle well inside this.
SETTLE_FRAMES = 60


def parse_options(items: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--option expects KEY=VALUE, got {item!r}")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    return overrides


def parse_sweep(spec: str | None) -> tuple[str | None, list[object]]:
    if not spec:
        return None, [None]
    key, sep, raw = spec.partition("=")
    if not sep or not key:
        raise SystemExit(f"--sweep expects KEY=v1,v2,..., got {spec!r}")
    values: list[object] = []
    for item in raw.split(","):
        try:
            values.append(json.loads(item))
        except json.JSONDecodeError:
            values.append(item)
    return key, values


def read_face(tracker: MediaPipeFaceTracker, image: np.ndarray) -> dict[str, float] | None:
    got = tracker.process(
        VideoFrame(
            data=np.ascontiguousarray(image),
            pixel_format=PixelFormat.BGR24,
            metadata=FrameMetadata(frame_id=0, capture_timestamp=0.0),
        )
    )
    if not got.face_detected or got.landmarks is None:
        return None
    h, w = image.shape[:2]
    pts = got.landmarks.points.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    m = measures(pts)
    a = anchors(pts)
    m["opening"] = _mouth_open_ratio(pts[:FACE_POINTS, :2])
    m["mouth_x"], m["mouth_y"] = float(a["mouth"][0]), float(a["mouth"][1])
    m["chin_y"] = float(a["chin"][1])
    m["eye_y"] = float(a["eye"][1])
    # Day 17 (the blink): the upper brow points' and the mouth corners'
    # height above the nose tip, frame px -- what blink_probe.py reads.
    nose_y = float(pts[_NOSE_TIP, 1])
    m["brow_up"] = nose_y - float(pts[list(_BROWS_UPPER), 1].mean())
    m["corner_up"] = nose_y - float(pts[list(_MOUTH_CORNERS), 1].mean())
    return m


def row_of(name: str, m: dict[str, float] | None, his: dict[str, float] | None) -> str:
    if m is None:
        return f"  {name:<28} no face found"
    line = (
        f"  {name:<28} mar {m['mar']:.3f}  open {m['opening']:.3f}  width {m['mouth_w']:.3f}  "
        f"ear {m['ear']:.3f}  brow {m['brow']:.3f}  lips {m['lip_upper']:.3f}/{m['lip_lower']:.3f}"
        f"  brow_up {m['brow_up']:.1f}  corner_up {m['corner_up']:.1f}"
    )
    if his is not None and his is not m:
        mar_ratio = m["mar"] / max(his["mar"], 1e-3)
        open_ratio = m["opening"] / max(his["opening"], 1e-3)
        width_ratio = m["mouth_w"] / his["mouth_w"]
        line += (
            f"  | /his: mar {mar_ratio:.2f} open {open_ratio:.2f} width {width_ratio:.2f}"
            f"  dy mouth {m['mouth_y'] - his['mouth_y']:+.1f}"
            f" chin {m['chin_y'] - his['chin_y']:+.1f}"
        )
    return line


def mouth_zoom(camera: np.ndarray, out: np.ndarray, his: dict[str, float] | None) -> np.ndarray:
    """His mouth and the avatar's, side by side at 2x, centred on his mouth."""
    import cv2

    h, w = camera.shape[:2]
    cx, cy = (int(his["mouth_x"]), int(his["mouth_y"])) if his else (w // 2, h // 2)
    x0, y0 = max(cx - 160, 0), max(cy - 110, 0)
    tiles = []
    for label, image in (("him", camera), ("avatar", out)):
        tile = image[y0 : y0 + 220, x0 : x0 + 320].copy()
        tile = cv2.resize(tile, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        cv2.putText(tile, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(tile)
    return np.hstack(tiles)


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--avatar", default=None, help="Defaults to the capture's own avatar.")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--sweep", default=None, metavar="KEY=v1,v2,...")
    parser.add_argument("--stills", action="store_true", help="Write the composite + a mouth zoom.")
    parser.add_argument("--out-prefix", type=Path, default=Path("recordings/day15-replay"))
    args = parser.parse_args()

    camera = cv2.imread(str(args.capture / "camera.png"))
    seen = cv2.imread(str(args.capture / "seen.png"))
    if camera is None:
        raise SystemExit(f"{args.capture} has no camera.png")
    data = np.load(args.capture / "tracking.npz", allow_pickle=False)
    detected = data["detected"]
    if not detected.any():
        raise SystemExit(f"{args.capture}: no face in the capture")
    last = int(np.where(detected)[0].max())
    anchor = capture_anchor(data)
    avatar_id = args.avatar or (
        str(data["anchor_avatar"]) if "anchor_avatar" in data.files else "my_avatar_neural"
    )
    overrides = parse_options(args.option)
    sweep_key, sweep_values = parse_sweep(args.sweep)

    paths = AppPaths.resolve()
    settings = load_settings(paths)
    models = paths.resolve_relative(settings.models.root_path)
    repo = FilesystemAvatarRepository(paths.resolve_relative(settings.avatars.library_path))
    reader = MediaPipeFaceTracker(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        smoothing=0.0,
        mouth_smoothing=0.0,
        min_detection_confidence=settings.tracking.min_detection_confidence,
        min_presence_confidence=settings.tracking.min_presence_confidence,
        min_tracking_confidence=settings.tracking.min_tracking_confidence,
        running_mode="image",
    )
    reader.start()
    his = read_face(reader, camera)
    anchor_note = "the session's own (from the capture)" if anchor else "NOT in the capture"
    print(
        f"{args.capture.name}: frame {last} of {len(detected)}, avatar {avatar_id}, "
        f"anchor {anchor_note}"
    )
    print(row_of("him (camera.png)", his, his))
    if seen is not None:
        print(row_of("what he saw (seen.png)", read_face(reader, seen), his))

    engine = LivePortraitEngine(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        weights=models / DEFAULT_WEIGHTS_RELATIVE_PATH,
    )
    engine.start()
    avatar = repo.get(avatar_id)
    base = copy.deepcopy(avatar.manifest.engine_options)
    frame = VideoFrame(
        data=camera,
        pixel_format=PixelFormat.BGR24,
        metadata=FrameMetadata(frame_id=last, capture_timestamp=float(data["timestamps"][last])),
    )
    reading = tracking_at(data, last)
    try:
        for value in sweep_values:
            options: dict[str, Any] = dict(base)
            options.update(overrides)
            if sweep_key is not None:
                options[sweep_key] = value
            avatar.manifest.engine_options.clear()
            avatar.manifest.engine_options.update(options)
            engine.unload_avatar(avatar_id)
            engine.load_avatar(avatar)
            if anchor is not None:
                engine.seed_anchor(avatar_id, anchor)
            out = None
            for _ in range(SETTLE_FRAMES):
                out = engine.animate(avatar_id, reading, frame)
            assert out is not None
            composite = np.ascontiguousarray(out.data)
            loaded = engine._loaded[avatar_id]
            label = "replay, base build" if sweep_key is None else f"replay, {sweep_key}={value}"
            if anchor is None:
                label += (
                    " (anchor learned)"
                    if loaded.anchor_seen >= loaded.anchor_wanted
                    else " (NO anchor: portrait rest)"
                )
            print(row_of(label, read_face(reader, composite), his))
            timings = ", ".join(f"{k} {v:.1f}" for k, v in sorted(engine.last_timings.items()))
            print(f"    last frame, ms: {timings}")
            if args.stills:
                tag = "" if sweep_key is None else f"-{sweep_key}{value}"
                prefix = args.out_prefix.parent / f"{args.out_prefix.name}-{args.capture.name}{tag}"
                prefix.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f"{prefix}.png", composite)
                cv2.imwrite(f"{prefix}-mouth.png", mouth_zoom(camera, composite, his))
                print(f"    stills: {prefix}.png, {prefix}-mouth.png (his face; delete after)")
    finally:
        engine.stop()
        reader.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
