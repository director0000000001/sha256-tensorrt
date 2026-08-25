"""Does the mouth need a grid, a ladder, or a new clip? Measured rather than argued.

Day 13 diagnosed the owner's first complaint — *"when I am talking it almost
does not follow me at all"* — and found the obvious reading wrong. The mouth's
*amount* and *timing* are right (lip parting gain 1.04, r 0.99). What is wrong
is the *content*: on two thirds of the clip the mouth on screen is 82% the
portrait's own closed mouth with one rounded-lips photograph washed over it.

Three questions were left open, and this tool answers all three on one pass.

**One: is speech two-dimensional?** An eye is open or shut, so its ladder has
one axis. Speech was assumed to have two — the mouth's width and the lips'
parting move independently, and the engine already computes both. If that is
true, a ladder is the wrong shape and it needs a grid.

**Two: does the material already exist?** If the `speech` clip on disk already
visits every part of the plane the driver visits, this is a builder and can be
done today. If it does not, it is a clip, which means money and a wait.

**Three: what does the engine's own scalar throw away?** The smile ladder is
chosen by ``widened x (1 + parted)`` — see
:func:`sha256.avatar.engine_warp._smile_amount`. Two frames with the same
scalar can be very different mouths. How different is measurable: band the
frames by the scalar and see how far apart the ones sharing a band sit.

**The plane is the engine's own, not a new one.** Both axes are what the engine
computes and normalised by the spans it normalises them with, so a unit here is
a unit there:

*   ``widened`` — :func:`_smiling`, the mouth's width against the outer eye
    corners, above the resting mouth, over ``_SMILE_WIDEN_SPAN``. The driver's
    rest is the floor :class:`_Neutral` learns; a source's rest is the
    portrait's own mouth, which is the zero the engine subtracts for it.
*   ``parted`` — :func:`_lip_parting`, the inner lips' gap on the same
    denominator, less ``_SMILE_LIP_CLOSED``, over ``_SMILE_OPEN_SPAN``.

**The driving clip's own segment map is not used, and that is deliberate.**
`recordings/driving.json` names which frames were recorded against which
instruction, and on this clip those names are wrong: the content lags the
prompts by up to 450 frames, so the eight seconds labelled *TALK* hold a smile
and the speech sits under *"Turn your head slowly RIGHT"*. Measuring speech
demand through that map measures a smile. Until a clip is recorded whose map
survives being checked, the window is given by hand with ``--speech``, and the
default is the one verified by eye on `driving.mkv` — frames 1560 to 1799.

Usage::

    python scripts/measure_speech_plane.py --supply "C:/.../speech.mp4"
    python scripts/measure_speech_plane.py --speech 1560 1799 --step 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from extract_avatar_frames import (  # noqa: E402
    portrait_tracking,
    sharpness_of,
    timestamp_of,
)

from sha256.avatar.engine_warp import (  # noqa: E402
    _FLOOR_FALL,
    _FLOOR_RISE,
    _SMILE_LIP_CLOSED,
    _SMILE_OPEN_SPAN,
    _SMILE_WIDEN_SPAN,
    _lip_parting,
    _smiling,
)
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FaceTrackingResult, FrameMetadata, VideoFrame  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)

# The frames of `driving.mkv` that actually hold speech, read off the pictures
# rather than off the sidecar. Eight seconds, which is what the TALK step asks
# for — the recording is complete, it is only mislabelled.
DEFAULT_SPEECH = (1560, 1799)

# One rung per this much of the parting axis. A tenth: the axis runs to about
# 1.2 on this driver, so it gives twelve rungs, which is the eye ladder's order
# of magnitude and the size the owner asked the smile ladder for.
DEFAULT_STEP = 0.1

# A rung has to be frontal and large enough to be cut from: the mouth is placed
# on the avatar's own frontal layout. Both thresholds are the ladder builders'.
DEFAULT_MAX_YAW = 12.0
DEFAULT_MIN_FACE = 380.0

# Which installed photographs can be chosen for a mouth. The turn, eye and brow
# ladders act on other regions and are not competing for this frame.
MOUTH_ROLES = frozenset(
    {
        "source_portrait",
        "source_mouth_open",
        "source_mouth_o",
        "source_mouth_wide",
        "source_mouth_pressed",
        "source_frown",
        "source_smile_step",
    }
)


@dataclass(frozen=True)
class Seen:
    """One frame, measured. ``name`` is what to call it in a table."""

    name: str
    width: float
    parting: float
    yaw: float
    face_px: float
    sharpness: float

    def usable(self, max_yaw: float, min_face: float) -> bool:
        return abs(self.yaw) <= max_yaw and self.face_px >= min_face


def place(width_above: float, parting: float) -> tuple[float, float]:
    """A mouth's point on the plane, in the engine's own units."""
    return (
        max(width_above, 0.0) / _SMILE_WIDEN_SPAN,
        max(parting - _SMILE_LIP_CLOSED, 0.0) / _SMILE_OPEN_SPAN,
    )


def read_frame(result: FaceTrackingResult, image: np.ndarray, name: str) -> Seen | None:
    if not result.face_detected or result.landmarks is None or result.head_pose is None:
        return None
    points = result.landmarks.points
    return Seen(
        name=name,
        width=_smiling(result),
        parting=_lip_parting(result),
        yaw=float(result.head_pose.yaw),
        face_px=float(points[:, 0].max() - points[:, 0].min()) * image.shape[1],
        sharpness=sharpness_of(image, points),
    )


def track_video(path: Path, tracker: MediaPipeFaceTracker) -> list[Seen]:
    """Every frame of a video, in order. The tracker carries state, so order matters."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    seen: list[Seen] = []
    index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            result = tracker.process(
                VideoFrame(
                    data=np.ascontiguousarray(image),
                    pixel_format=PixelFormat.BGR24,
                    metadata=FrameMetadata(frame_id=index, capture_timestamp=index / fps),
                )
            )
            taken = read_frame(result, image, str(index))
            if taken is not None:
                seen.append(taken)
            index += 1
    finally:
        capture.release()
    return seen


def track_folder(path: Path, tracker: MediaPipeFaceTracker) -> list[Seen]:
    """Every dumped frame in a folder, with the timestamps the tracker needs."""
    import cv2

    seen: list[Seen] = []
    frames = sorted(path.glob("*.jpg")) + sorted(path.glob("*.png"))
    for index, frame_path in enumerate(frames):
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        result = tracker.process(
            VideoFrame(
                data=np.ascontiguousarray(image),
                pixel_format=PixelFormat.BGR24,
                metadata=FrameMetadata(
                    frame_id=index, capture_timestamp=timestamp_of(frame_path, index)
                ),
            )
        )
        taken = read_frame(result, image, frame_path.name)
        if taken is not None:
            seen.append(taken)
    return seen


def track_source(path: Path, tracker: MediaPipeFaceTracker) -> list[Seen]:
    tracker.stop()
    tracker.start()
    return track_folder(path, tracker) if path.is_dir() else track_video(path, tracker)


def installed_mouths(avatar: Path, tracker: MediaPipeFaceTracker) -> list[Seen]:
    """The photographs the engine can already put on a mouth."""
    import cv2

    meta = json.loads((avatar / "metadata.json").read_text(encoding="utf-8"))
    seen: list[Seen] = []
    for asset in meta["assets"]:
        if asset.get("role") not in MOUTH_ROLES:
            continue
        path = avatar / asset["path"]
        got = portrait_tracking(path, tracker)
        image = cv2.imread(str(path))
        if got is None or image is None:
            print(f"  ! could not read {asset['path']}")
            continue
        taken = read_frame(got[0], image, asset["path"])
        if taken is not None:
            seen.append(taken)
    return seen


def driver_points(seen: list[Seen]) -> np.ndarray:
    """The driver's frames on the plane, with the floor the engine actually keeps.

    **Advanced twice a frame, because the engine advances it twice a frame.**
    ``_smile_share`` reads it inside ``_source_weights`` and ``_smile_amount``
    reads it again to choose the rung, both through
    ``_Neutral.relative_expression`` and both learning from the reading. The
    same file's :meth:`_Neutral.floor_of` exists precisely so a second caller
    does not do that, and the head pose is careful about it — *"Read, not
    learn: `_source_weights` has already advanced the resting angle for this
    frame"*. The smile floor is not. Reproduced here rather than corrected, so
    that what is measured is the engine that ships.
    """
    floor: float | None = None
    points = []
    for frame in seen:
        value = frame.width
        if floor is None:
            floor, above = value, 0.0
        else:
            for _ in range(2):
                rate = _FLOOR_FALL if value < floor else _FLOOR_RISE
                floor += rate * (value - floor)
            above = max(value - floor, 0.0)
        points.append(place(above, frame.parting))
    return np.array(points, dtype=float)


def source_points(seen: list[Seen], rest: float) -> np.ndarray:
    """A library's frames on the plane. ``rest`` is the portrait's own mouth."""
    if not seen:
        return np.zeros((0, 2), dtype=float)
    return np.array([place(frame.width - rest, frame.parting) for frame in seen], dtype=float)


def nearest(demand: np.ndarray, supply: np.ndarray) -> np.ndarray:
    """How far each wanted mouth is from the closest photograph available."""
    return np.linalg.norm(demand[:, None, :] - supply[None, :, :], axis=2).min(axis=1)


def spread(points: np.ndarray) -> float:
    return float(np.linalg.norm(points - points.mean(axis=0), axis=1).mean())


def scalar_spread(points: np.ndarray, bands: int = 12) -> float:
    """How far apart frames sit when the engine's one number cannot tell them apart."""
    scalar = points[:, 0] * (1.0 + points[:, 1])
    held = [
        spread(points[band]) for band in np.array_split(np.argsort(scalar), bands) if len(band) >= 3
    ]
    return float(np.mean(held)) if held else 0.0


def dimensions(points: np.ndarray) -> tuple[float, float, tuple[float, float]]:
    """How much of the demand lies on one axis, and which one."""
    centred = points - points.mean(axis=0)
    _, values, directions = np.linalg.svd(centred, full_matrices=False)
    share = values**2 / float((values**2).sum())
    return float(share[0]), float(share[1]), (float(directions[0][0]), float(directions[0][1]))


def percentiles(values: np.ndarray) -> str:
    return (
        f"p5/50/95 {np.percentile(values, 5):.3f}/"
        f"{np.percentile(values, 50):.3f}/{np.percentile(values, 95):.3f}"
        f"  max {values.max():.3f}"
    )


def report_bands(
    demand: np.ndarray,
    supply: list[Seen],
    rest: float,
    step: float,
    max_yaw: float,
    min_face: float,
) -> list[int]:
    """The ladder this clip could fill, band by band, against what the driver asks for."""
    usable = [frame for frame in supply if frame.usable(max_yaw, min_face)]
    points = source_points(usable, rest)
    top = max(demand[:, 1].max(), points[:, 1].max() if len(points) else 0.0)

    header = f"{'band (parted)':>15} {'demand':>8} {'frames':>7} {'sharpest':>9}"
    print(f"\n{header} {'its widened':>12}")
    holes = []
    for index in range(int(np.ceil(top / step))):
        low, high = index * step, (index + 1) * step
        wanted = int(((demand[:, 1] >= low) & (demand[:, 1] < high)).sum())
        inside = [
            (frame, point)
            for frame, point in zip(usable, points, strict=True)
            if low <= point[1] < high
        ]
        if wanted == 0 and not inside:
            continue
        if wanted > 0 and not inside:
            holes.append(index)
        best = max(inside, key=lambda pair: pair[0].sharpness) if inside else None
        share = f"{100 * wanted / len(demand):6.1f}%"
        if best is None:
            print(f"{low:6.2f}-{high:<8.2f} {share} {len(inside):>7} {'--':>9} {'--':>12}")
        else:
            print(
                f"{low:6.2f}-{high:<8.2f} {share} {len(inside):>7} "
                f"{best[0].sharpness:>9.0f} {best[1][0]:>12.3f}"
            )
    return holes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, default=REPO_ROOT / "recordings" / "driving.mkv")
    parser.add_argument("--avatar", type=Path, default=REPO_ROOT / "avatars" / "my_avatar")
    parser.add_argument(
        "--supply",
        type=Path,
        nargs="*",
        default=[],
        help="Clips to test as rung material: a video file, or a folder of dumped frames.",
    )
    parser.add_argument(
        "--speech",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        default=DEFAULT_SPEECH,
        help="The frames of the driving clip that are speech. Not read from the sidecar.",
    )
    parser.add_argument("--step", type=float, default=DEFAULT_STEP)
    parser.add_argument("--max-yaw", type=float, default=DEFAULT_MAX_YAW)
    parser.add_argument("--min-face", type=float, default=DEFAULT_MIN_FACE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    model = REPO_ROOT / "models" / DEFAULT_MODEL_RELATIVE_PATH
    if not model.is_file():
        print(f"Model bundle missing: {model}")
        return 1
    if not args.clip.is_file():
        print(f"No clip at {args.clip}. Record one with scripts/record_driving_clip.py")
        return 1

    tracker = MediaPipeFaceTracker(model_path=model)
    tracker.start()
    try:
        portrait = portrait_tracking(args.avatar / "portrait.png", tracker)
        if portrait is None:
            print(f"No face in {args.avatar / 'portrait.png'}.")
            return 1
        rest = _smiling(portrait[0])

        library = installed_mouths(args.avatar, tracker)
        driver = track_source(args.clip, tracker)
        supplies = [(path.name, track_source(path, tracker)) for path in args.supply]
    finally:
        tracker.stop()

    first, last = args.speech
    speech = [frame for frame in driver if first <= int(frame.name) <= last]
    if not speech:
        print(f"No tracked frames between {first} and {last} in {args.clip.name}.")
        return 1

    print("\nconditions")
    print(f"  clip            {args.clip.name}, {len(driver)} frames tracked")
    print(f"  speech window   {first}-{last}, {len(speech)} frames, named by hand")
    print(f"  avatar          {args.avatar.name}, resting mouth {rest:.4f} of an eye span")
    print(f"  library         {len(library)} photographs the engine may put on a mouth")
    print(f"  gates           |yaw| <= {args.max_yaw:g}, face >= {args.min_face:g} px")

    # The whole clip has to be walked so the floor is the one the engine would
    # hold by then; only the speech window is reported.
    points = driver_points(driver)
    indices = np.array([int(frame.name) for frame in driver])
    demand = points[(indices >= first) & (indices <= last)]

    print("\nwhat speech asks the mouth for")
    print(f"  widened   {percentiles(demand[:, 0])}")
    print(f"  parted    {percentiles(demand[:, 1])}")
    print(f"  the two move together at r = {np.corrcoef(demand[:, 0], demand[:, 1])[0, 1]:+.3f}")
    first_share, second_share, direction = dimensions(demand)
    print(
        f"  one direction carries {first_share * 100:.1f}% of it, the other "
        f"{second_share * 100:.1f}%; that direction is "
        f"({direction[0]:+.3f} widened, {direction[1]:+.3f} parted)"
    )

    whole, banded = spread(demand), scalar_spread(demand)
    print("\nwhat the engine's own scalar keeps")
    print(f"  the demand is {whole:.3f} wide about its centre")
    print(f"  frames the scalar cannot tell apart still sit {banded:.3f} apart")
    print(f"  so widened x (1 + parted) removes {100 * (1 - banded / whole):.0f}% of it")

    print("\nhow far the nearest photograph is, on the speech frames")
    print(f"  {'the installed library':<34} {'sources':>7} {'med':>7} {'p90':>7} {'worst':>7}")
    candidates: list[tuple[str, list[Seen]]] = [("the installed library", library), *supplies]
    for name, seen in candidates:
        usable = [frame for frame in seen if frame.usable(args.max_yaw, args.min_face)]
        if not usable:
            print(f"  {name:<34} {0:>7}  nothing passes the gates")
            continue
        distances = nearest(demand, source_points(usable, rest))
        print(
            f"  {name:<34} {len(usable):>7} {np.median(distances):>7.3f} "
            f"{np.percentile(distances, 90):>7.3f} {distances.max():>7.3f}"
        )

    for name, seen in supplies:
        print(f"\nthe ladder {name} could fill")
        holes = report_bands(demand, seen, rest, args.step, args.max_yaw, args.min_face)
        if holes:
            named = ", ".join(f"{index * args.step:.1f}" for index in holes)
            print(f"  {len(holes)} band(s) the driver visits and this clip cannot fill: {named}")
        else:
            print("  no holes: every band the driver visits, this clip can fill.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
