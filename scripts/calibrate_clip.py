"""Run "Calibrate me" on a recorded clip -- the owner's own guided clips, or
a calibration session's -- and say what profile it would write.

Day 16's own bar (ROADMAP, "The road to the finish", A2): the calibration
must measure, from his face, what fifteen days set by hand. His hand values
live in ``avatars/my_avatar_neural/metadata.json``; his face is in
``recordings/driving.mkv`` (the recorder's guided script, the same
movements under the calibration's step names -- see
``sha256.calibration.script.step_for_instruction``). This script runs the
clip through the same instruments the live run uses (the app's filtered
tracker, the engine, a second tracker on the output), prints the
measurement and a table of the driver options the calibration would set
against the ones the avatar runs with today, and -- with ``--write`` --
writes the profile.

Usage::

    uv run python scripts/calibrate_clip.py --clip recordings/driving.mkv
    uv run python scripts/calibrate_clip.py --clip recordings/natural.mkv --last 900
    uv run python scripts/calibrate_clip.py --session recordings/calibration-20260912-101010
    uv run python scripts/calibrate_clip.py --clip recordings/driving.mkv --write
    uv run python scripts/calibrate_clip.py --samples recordings/day16/driving-samples.json

``--samples-out`` keeps the per-frame samples as JSON and ``--samples`` re-reads
them: the measurement's rules can be changed and re-run in seconds, on the
same frames, with no GPU. ``--option KEY=VALUE`` overrides an engine option for
the run (the avatar's file is not touched); ``--no-profile`` runs the avatar's
own options without the machine's existing profile laid over them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sha256.avatar.engine_liveportrait import (  # noqa: E402
    DEFAULT_WEIGHTS_RELATIVE_PATH,
    DRIVER_OPTION_KEYS,
    LivePortraitEngine,
)
from sha256.avatar.repository import FilesystemAvatarRepository  # noqa: E402
from sha256.calibration.measure import calibrate, describe  # noqa: E402
from sha256.calibration.offline import samples_from_clip  # noqa: E402
from sha256.calibration.profile import PROFILE_FILENAME, DriverProfile  # noqa: E402
from sha256.calibration.samples import Sample  # noqa: E402
from sha256.calibration.session import CLIP_NAME  # noqa: E402
from sha256.config.loader import load_settings  # noqa: E402
from sha256.config.paths import AppPaths  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)


def parse_options(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--option expects KEY=VALUE, got {item!r}")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    return overrides


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    return str(value)


def _deviation(before: Any, after: Any) -> str:
    """How far ``after`` is from ``before``, as a percentage where both are
    numbers (or lists of numbers, the largest)."""
    try:
        if isinstance(before, list) and isinstance(after, list):
            devs = [abs(float(a) / float(b) - 1.0) for a, b in zip(before, after, strict=True) if b]
            return f"{max(devs):.0%}" if devs else "-"
        if isinstance(before, int | float) and isinstance(after, int | float) and before:
            return f"{abs(float(after) / float(before) - 1.0):.0%}"
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return "-"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--clip", type=Path, help="A recorded clip with its .json sidecar beside it."
    )
    source.add_argument("--session", type=Path, help="A Calibrate me session directory.")
    source.add_argument("--samples", type=Path, help="Samples saved by --samples-out.")
    parser.add_argument(
        "--avatar", default=None, help="Defaults to the session's, else the config's."
    )
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--no-profile", action="store_true", help="Ignore the machine's profile.")
    parser.add_argument("--last", type=int, default=None, help="Stop after this frame index.")
    parser.add_argument("--write", action="store_true", help="Write configs/local.driver.json.")
    parser.add_argument(
        "--samples-out", type=Path, default=None, help="Also dump the per-frame samples as JSON."
    )
    args = parser.parse_args()

    paths = AppPaths.resolve()
    settings = load_settings(paths)
    if args.samples is not None:
        return measure_saved(args, paths, settings)
    clip = args.clip if args.clip is not None else args.session / CLIP_NAME
    if not clip.is_file():
        raise SystemExit(f"{clip} is not there")
    models = paths.resolve_relative(settings.models.root_path)
    repository = FilesystemAvatarRepository(paths.resolve_relative(settings.avatars.library_path))
    avatar_id = args.avatar
    if avatar_id is None and args.session is not None:
        sidecar = args.session / "clip.json"
        if sidecar.is_file():
            avatar_id = json.loads(sidecar.read_text(encoding="utf-8")).get("avatar")
    avatar_id = avatar_id or settings.avatars.default_avatar_id or "my_avatar_neural"

    profile_path = paths.configs / PROFILE_FILENAME
    profile = None if args.no_profile else DriverProfile.load(profile_path)
    driver_options = profile.overlay() if profile else {}
    overrides = parse_options(args.option)
    driver_overrides = {k: v for k, v in overrides.items() if k in DRIVER_OPTION_KEYS}
    manifest_overrides = {k: v for k, v in overrides.items() if k not in DRIVER_OPTION_KEYS}
    driver_options.update(driver_overrides)

    def tracker(*, filtered: bool) -> MediaPipeFaceTracker:
        return MediaPipeFaceTracker(
            model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
            smoothing=settings.tracking.smoothing if filtered else 0.0,
            mouth_smoothing=settings.tracking.mouth_smoothing if filtered else 0.0,
            min_detection_confidence=settings.tracking.min_detection_confidence,
            min_presence_confidence=settings.tracking.min_presence_confidence,
            min_tracking_confidence=settings.tracking.min_tracking_confidence,
        )

    driver = tracker(filtered=True)
    output = tracker(filtered=False)
    driver.start()
    output.start()
    engine = LivePortraitEngine(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        weights=models / DEFAULT_WEIGHTS_RELATIVE_PATH,
        driver_options=driver_options,
    )
    engine.start()
    avatar = repository.get(avatar_id)
    avatar.manifest.engine_options.update(manifest_overrides)
    engine.load_avatar(avatar)
    current = dict(avatar.manifest.engine_options)
    current.update(driver_options)
    print(f"{clip}: avatar {avatar_id}, profile {'none' if not profile else profile_path.name}")
    if driver_options:
        print(f"  driver options in force: {driver_options}")
    try:
        samples = samples_from_clip(
            clip,
            tracker=driver,
            engine=engine,
            avatar_id=avatar_id,
            output_tracker=output,
            last=args.last,
            on_progress=lambda n: print(f"  ... {n} frames", flush=True),
        )
    finally:
        engine.stop()
        driver.stop()
        output.stop()

    if args.samples_out is not None:
        args.samples_out.write_text(
            json.dumps([sample_record(s) for s in samples]), encoding="utf-8"
        )
    report(samples, current, args, paths, avatar_id, clip)
    return 0


def sample_record(s: Sample) -> dict[str, Any]:
    return {
        "frame": s.frame_id,
        "timestamp": s.timestamp,
        "step": s.step,
        "face": s.face,
        "his": s.his,
        "engine": s.engine,
        "avatar": s.avatar,
    }


def measure_saved(args: argparse.Namespace, paths: AppPaths, settings: Any) -> int:
    """The saved samples through the measurement alone: the avatar's options
    as they stand now, no engine, no GPU."""
    records = json.loads(args.samples.read_text(encoding="utf-8"))
    samples = [
        Sample(
            frame_id=int(r["frame"]),
            timestamp=float(r.get("timestamp", r["frame"] / 30.0)),
            step=str(r["step"]),
            face=bool(r["face"]),
            his=dict(r["his"]),
            engine=dict(r["engine"]),
            avatar=None if r["avatar"] is None else dict(r["avatar"]),
        )
        for r in records
    ]
    repository = FilesystemAvatarRepository(paths.resolve_relative(settings.avatars.library_path))
    avatar_id = args.avatar or settings.avatars.default_avatar_id or "my_avatar_neural"
    current = dict(repository.get(avatar_id).manifest.engine_options)
    if not args.no_profile:
        profile = DriverProfile.load(paths.configs / PROFILE_FILENAME)
        if profile:
            current.update(profile.overlay())
    current.update(parse_options(args.option))
    print(f"{args.samples}: {len(samples)} saved samples, avatar {avatar_id}")
    report(samples, current, args, paths, avatar_id, args.samples)
    return 0


def report(
    samples: list[Sample],
    current: dict[str, Any],
    args: argparse.Namespace,
    paths: AppPaths,
    avatar_id: str,
    source: Path,
) -> None:
    profile_path = paths.configs / PROFILE_FILENAME
    result = calibrate(samples, current)
    print()
    print("Frames per step:", ", ".join(f"{k} {v}" for k, v in sorted(result.frames.items())))
    print(describe(result, current=current))
    print()
    print("THE PROFILE AGAINST WHAT THE AVATAR RUNS WITH NOW")
    print(f"  {'option':<26} {'now':>28} {'calibrated':>28} {'off by':>7}")
    for key in sorted(result.options):
        before = current.get(key, "(default)")
        after = result.options[key]
        print(f"  {key:<26} {_fmt(before):>28} {_fmt(after):>28} {_deviation(before, after):>7}")
    if args.write:
        new = DriverProfile.from_result(result, avatar=avatar_id, session=str(source.parent))
        new.save(profile_path)
        print(f"\nWritten: {profile_path}")
    else:
        print("\n(not written; pass --write to save the profile)")


if __name__ == "__main__":
    raise SystemExit(main())
