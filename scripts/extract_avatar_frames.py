"""Pick avatar source photographs out of a video clip.

A clip of one face is worth more than any number of separately generated stills,
because the identity cannot drift between frames of the same take. What it needs
is a way to find, among a few hundred frames, the best example of each thing the
engine knows how to ask for.

**The engine's own selector does the classifying.** ``_source_weights`` already
answers "how much of the open-mouth photograph do I want for this frame", from
blendshapes and head pose. Run over a video it answers "which frame is the best
example of an open mouth" — the same question from the other end. Using one
function for both means the library can never be classified by rules the
renderer disagrees with.

Frames are rejected before they are scored, on the three things that make a
frame unusable no matter what expression it caught:

*   **no face**, or one too small to carry detail;
*   **motion blur** — a frame taken mid-movement is soft, and softness is the
    one flaw the warp cannot recover from;
*   **identity drift** — measured against the base portrait, in case the
    generator lost the person partway through.

**A turn clip is different, and ``--ladder`` is for it.** One best frame per
role is the right answer for an expression: there is one open mouth. A turn is
not a role, it is an axis, and the engine used to be handed two points on it and
left to average everything in between. ``--ladder`` writes the sharpest frame in
each band of yaw, each carrying the angle it was measured at, so the engine can
select the nearest instead of averaging two that are neither.

Usage::

    python scripts/extract_avatar_frames.py clip.mp4 --out extracted/
    python scripts/extract_avatar_frames.py clip.mp4 --out extracted/ --min-face 420
    python scripts/extract_avatar_frames.py turns.mp4 --out extracted/ --ladder
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

from sha256.avatar.engine_warp import (  # noqa: E402
    _LEFT_BROW,
    _LEFT_EYE,
    _RIGHT_BROW,
    _RIGHT_EYE,
    _VARIANT_ROLES,
    _Neutral,
    _source_weights,
)
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FaceTrackingResult, FrameMetadata, VideoFrame  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)

# The role every avatar must have, and the one a clip cannot supply: it is the
# reference every other source is registered against.
BASE_ROLE = "source_portrait"

# A frame has to be at least this good an example of a role before it is taken.
# Below it the "best" frame is only the least bad one, and a photograph that
# barely shows the expression is worse than not having it — the engine would
# blend it in whenever the expression appeared.
MIN_ROLE_SCORE = 0.45

# How much a frame is penalised for the *other* expressions it is also doing.
#
# Every source is confined to its own region, but a region reaches past its own
# landmarks — the eye region has to carry the lid and the socket shadow, so it
# takes in the lower brow as well. A photograph that is doing two things at once
# brings the second one with it.
#
# Seen: the clearest "eyes closed" frame in the eye clip was the one from the
# *frown*, because a hard scowl squeezes the lids further shut than resting them
# does. The avatar then blinked with a glare. Half weight is enough to prefer a
# calm frame without rejecting the genuine peak of an expression, which always
# carries a little of its neighbours.
_INTERFERENCE = 0.5

# Sharpness is the variance of the Laplacian over the face. It has no absolute
# meaning across cameras, so the threshold is relative: a frame must reach this
# fraction of the sharpest frame **at a similar head angle**.
MIN_SHARPNESS_FRACTION = 0.55

# How many frames either side to look for the frame this one is judged against.
#
# **Laplacian variance over a face is not a focus measure**, and this is the
# third time that has cost a clip. It measures how much high-contrast detail the
# crop happens to contain, which depends on what the face is doing:
#
#   * a turned face shows a smooth cheek where a frontal one shows two eyes, two
#     brows and a hairline — measured, median 179 at 0-8 degrees of yaw against
#     9 at 20-35, on frames sharp enough to count beard hairs;
#   * an open mouth adds teeth and a dark cavity, and a wide eye adds sclera, so
#     a neutral face scores a fraction of an expressive one in the same clip;
#   * and the codec starves a moving shot of bits wherever the movement is.
#
# Judged against the sharpest frame in the whole clip, each of those becomes a
# reason to throw away exactly the frames the clip was generated for: 135 of 192
# on the turn clip, 163 of 192 on the eye clip.
#
# **Motion blur is a phenomenon in time**, so it is measured in time. A frame is
# blurred if it is much softer than the frames around it — a held expression
# sits in a plateau where every neighbour scores alike and all of them pass,
# while the frames during a movement dip below their own neighbours and go.
# Nothing in that depends on what the face happens to be doing.
#
# Eight frames either side is a third of a second at 24 fps: longer than any
# blurred transition, shorter than the two-second holds the clips are built
# around.
SHARPNESS_WINDOW_FRAMES = 8

# How far a structural proportion may move from the base portrait before the
# frame is treated as a different face, at zero yaw, and how much more is
# allowed per degree of turn. See :func:`identity_tolerance`.
IDENTITY_TOLERANCE = 0.08
IDENTITY_TOLERANCE_PER_DEGREE = 0.003

# The turn ladder: one frame per this many degrees of yaw.
#
# The engine used to be given two turned photographs and a frontal one, and it
# *blended* them, so every angle in between was an average of two angles that
# were not it. Given a frame at each angle it can select the nearest instead,
# which is exact where blending was approximate — and only then does a dense set
# help rather than hurt.
#
# Three degrees because that is finer than the alignment error the mesh warp
# leaves behind, so a finer ladder would be measuring its own noise.
LADDER_STEP_DEGREES = 3.0

# How much of the sharpest rung's near-eye detail a rung must keep before it is
# called blurred. Measured on the owner's first ladder, whose turned rungs came
# out at **4%** of the near-frontal one — a clip generated with the head moving
# continuously, so every frame during the turn is smeared. A sharp turn loses
# some of this measure too, because a lash line at 40 degrees is foreshortened,
# but nothing like that much; a quarter leaves room for the honest loss and
# still catches a smear.
LADDER_MIN_DETAIL = 0.25

# How much room to leave around the face when writing a frame out, as a fraction
# of face width. The engine crops to the face itself when it loads a source, so
# everything beyond this is read from disk and thrown away — and at 4K that is
# most of a ten megabyte file. Generous enough that the mesh and the rigid
# backdrop layer never want a pixel that was cut off.
WRITE_MARGIN = 0.6


@dataclass(frozen=True)
class Candidate:
    index: int
    image: np.ndarray
    tracking: FaceTrackingResult
    face_width: float
    sharpness: float
    identity_drift: float


def identity_ratios(points: np.ndarray, width: int, height: int) -> dict[str, float]:
    """Proportions that do not change between photographs of one person.

    Measured against the eye span rather than face height: the chin drops when
    the mouth opens, and half these frames are mid-sentence.
    """
    scaled = points[:, :2] * (width, height)

    def distance(a: int, b: int) -> float:
        return float(np.linalg.norm(scaled[a] - scaled[b]))

    eye_span = max(distance(33, 263), 1e-6)
    return {
        "eye_spacing": distance(133, 362) / eye_span,
        "face_width": distance(234, 454) / eye_span,
        "nose_width": distance(98, 327) / eye_span,
    }


def sharpness_of(image: np.ndarray, points: np.ndarray) -> float:
    """Laplacian variance over the face only.

    Over the whole frame a sharp background would hide a soft face, which is
    exactly the frame this is meant to reject.
    """
    import cv2

    height, width = image.shape[:2]
    x0 = max(int(points[:, 0].min() * width), 0)
    x1 = min(int(points[:, 0].max() * width), width)
    y0 = max(int(points[:, 1].min() * height), 0)
    y1 = min(int(points[:, 1].max() * height), height)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0

    face = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(face, cv2.CV_64F).var())


def eye_sharpness(candidate: Candidate) -> float:
    """Sharpness measured on the eye and brow **nearer the camera**.

    The whole-face measure cannot judge a turn, and this is not a subtlety: a
    Laplacian counts the detail a face is *showing*, and a turned face shows
    much less of it — one eye goes behind the nose, and the cheek that fills
    the frame is smooth skin. Measured on frames sharp enough to count beard
    hairs, the median runs 179 at 0-8 degrees of yaw against 9 at 20-35. A
    blurred turn and a sharp one therefore score alike, which is how a ladder
    of thirty motion-blurred frames was built and shipped.

    The near eye does not have that problem. It stays fully visible and fully
    detailed to well past 40 degrees — lashes, brow hairs, the iris edge — so
    its sharpness is comparable across the whole turn. Taking the better of the
    two eyes always measures the near one, whichever way the head is pointing.
    """
    import cv2

    points = candidate.tracking.landmarks
    if points is None:
        return 0.0

    height, width = candidate.image.shape[:2]
    scaled = points.points[:, :2] * (width, height)
    grey = cv2.cvtColor(candidate.image, cv2.COLOR_BGR2GRAY)

    best = 0.0
    for eye, brow in ((_LEFT_EYE, _LEFT_BROW), (_RIGHT_EYE, _RIGHT_BROW)):
        group = scaled[list(eye) + list(brow)]
        pad = (group[:, 0].max() - group[:, 0].min()) * 0.25
        x0 = max(int(group[:, 0].min() - pad), 0)
        x1 = min(int(group[:, 0].max() + pad), width)
        y0 = max(int(group[:, 1].min() - pad), 0)
        y1 = min(int(group[:, 1].max() + pad), height)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        best = max(best, float(cv2.Laplacian(grey[y0:y1, x0:x1], cv2.CV_64F).var()))
    return best


def read_candidates(
    video: Path, tracker: MediaPipeFaceTracker, base_ratios: dict[str, float]
) -> tuple[list[Candidate], dict[str, int]]:
    """Every frame with a usable face, and a tally of why the others went."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        msg = f"could not open {video}"
        raise SystemExit(msg)

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    candidates: list[Candidate] = []
    rejected = {"no face": 0}
    index = 0

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
        index += 1

        if not result.face_detected or result.landmarks is None:
            rejected["no face"] += 1
            continue

        points = result.landmarks.points
        height, width = image.shape[:2]
        ratios = identity_ratios(points, width, height)
        drift = max(
            abs(value - base_ratios[name]) / base_ratios[name] for name, value in ratios.items()
        )

        candidates.append(
            Candidate(
                index=index - 1,
                image=image,
                tracking=result,
                face_width=float(points[:, 0].max() - points[:, 0].min()) * width,
                sharpness=sharpness_of(image, points),
                identity_drift=drift,
            )
        )

    capture.release()
    return candidates, rejected


def yaw_of(candidate: Candidate) -> float:
    pose = candidate.tracking.head_pose
    return float(pose.yaw) if pose is not None else 0.0


def sharpness_floor(candidates: list[Candidate], candidate: Candidate) -> float:
    """The best sharpness achieved within a third of a second of this frame.

    See ``SHARPNESS_WINDOW_FRAMES``: Laplacian variance over a face measures how
    much detail the face is showing, not how sharp it is, so the only fair
    comparison is with frames of the same face doing nearly the same thing —
    which, at 24 fps, is the frames next to it.
    """
    here = candidate.index
    nearby = [
        other.sharpness
        for other in candidates
        if abs(other.index - here) <= SHARPNESS_WINDOW_FRAMES
    ]
    return max(nearby) if nearby else candidate.sharpness


def identity_tolerance(yaw: float) -> float:
    """How far the proportions may move at this head angle.

    They are ratios of distances across the face, and a turned face is a
    *projection*: horizontal spans foreshorten, and they do not foreshorten
    together, because the nose stands out of the plane the cheeks lie in. So
    rotation moves these numbers on its own, with no change of identity at all.

    Measured on the owner's head-turn clip, drift against the base portrait:

        yaw   1 degree   0.014
        yaw  13 degrees  0.039
        yaw  27 degrees  0.022
        yaw  39 degrees  0.094
        yaw  45 degrees  0.169

    Against a flat 0.08 that discarded 41 frames of one 192-frame clip, and the
    frames it discarded were the far turns — the ones the clip was generated
    for. The allowance grows with the angle, which keeps the check meaningful
    where it can discriminate and stops it deleting the far end of every turn.

    It is weaker at 45 degrees than at zero, and that is honest: at that angle
    nothing here can tell a different face from the same one in profile.
    """
    return IDENTITY_TOLERANCE + IDENTITY_TOLERANCE_PER_DEGREE * abs(yaw)


def usable(
    candidates: list[Candidate], min_face: float, rejected: dict[str, int]
) -> list[Candidate]:
    """Drop the frames no expression could redeem."""
    if not candidates:
        return []

    kept: list[Candidate] = []
    for candidate in candidates:
        floor = sharpness_floor(candidates, candidate) * MIN_SHARPNESS_FRACTION
        if candidate.face_width < min_face:
            rejected["face too small"] = rejected.get("face too small", 0) + 1
        elif candidate.sharpness < floor:
            rejected["motion blurred"] = rejected.get("motion blurred", 0) + 1
        elif candidate.identity_drift > identity_tolerance(yaw_of(candidate)):
            rejected["identity drifted"] = rejected.get("identity drifted", 0) + 1
        else:
            kept.append(candidate)
    return kept


def turn_ladder(candidates: list[Candidate], step: float) -> list[Candidate]:
    """The sharpest frame in each band of yaw, ordered by angle.

    Not "every frame": frames a fifth of a second apart at the same angle are
    the same photograph twice, and each one the engine loads costs a face
    detection at load and memory for its lifetime. One per band keeps the
    coverage and drops the duplicates.
    """
    best: dict[int, Candidate] = {}
    for candidate in candidates:
        band = round(yaw_of(candidate) / step)
        held = best.get(band)
        if held is None or candidate.sharpness > held.sharpness:
            best[band] = candidate
    return [best[band] for band in sorted(best)]


def crop_around_face(candidate: Candidate) -> np.ndarray:
    """The frame trimmed to the face and a margin.

    Written frames are avatar sources, and the engine crops to the face when it
    loads one. Keeping the rest costs disk and load time for pixels no part of
    the pipeline ever reads — at 4K, most of the file.
    """
    points = candidate.tracking.landmarks
    assert points is not None

    height, width = candidate.image.shape[:2]
    scaled = points.points[:, :2] * (width, height)
    margin = candidate.face_width * WRITE_MARGIN

    x0 = max(int(scaled[:, 0].min() - margin), 0)
    y0 = max(int(scaled[:, 1].min() - margin), 0)
    x1 = min(int(scaled[:, 0].max() + margin), width)
    y1 = min(int(scaled[:, 1].max() + margin), height)

    cropped: np.ndarray = np.ascontiguousarray(candidate.image[y0:y1, x0:x1])
    return cropped


def resting_face(base: FaceTrackingResult) -> _Neutral:
    """The resting face every frame of the clip is measured against.

    **This is the whole classifier**, and getting it wrong silently costs the
    clip. `_source_weights` measures expression and head pose *relative to
    rest*, and given no rest it makes one from the frame in front of it — which
    reads as "nothing is happening" for every frame, because the frame is its
    own baseline. Every role driven by a blendshape or by head pose then scores
    exactly zero, and only the two driven by the lip gap survive, because the
    gap is measured off the landmarks directly.

    That is why a clip of a man turning his head end to end reported **0 of 22**
    roles, and why the speaking clip before it reported 2 — the two that do not
    go through here.

    The right baseline is the base portrait: it is the face every source is
    registered against, and it does not move.
    """
    neutral = _Neutral()
    # One pass seeds a floor for every blendshape and a rest for every pose
    # angle that the selector reads.
    _source_weights(base, neutral=neutral)
    return neutral


def best_per_role(
    candidates: list[Candidate], neutral: _Neutral
) -> dict[str, tuple[Candidate, float]]:
    """The clearest example of each role the clip contains.

    Ties on score are broken by sharpness. Two frames a fifth of a second apart
    show the same expression, and there is no reason to take the softer one.

    Each frame gets its own copy of the resting face, so a frame is scored the
    same whatever order the clip is read in. Sharing one would let it learn from
    the clip — and a clip that is a held turn would teach it that a turned head
    is how this person sits.
    """
    chosen: dict[str, tuple[Candidate, float]] = {}
    ranked: dict[str, float] = {}

    for candidate in candidates:
        weights = _source_weights(candidate.tracking, neutral=neutral.copy())
        for role, score in weights.items():
            if score < MIN_ROLE_SCORE:
                continue

            # **The clearest example of a role is the frame that is most that
            # role and least anything else.** Taking the highest score alone
            # picked, for "eyes closed", the frame from the frown: a hard scowl
            # squeezes the lids further shut than resting them does, so it won
            # on closure while carrying furrowed brows into the eye region — the
            # avatar blinked with a glare.
            interference = sum(other for name, other in weights.items() if name != role)
            rank = score - _INTERFERENCE * interference

            current = chosen.get(role)
            if current is None:
                better = True
            elif abs(rank - ranked[role]) <= 0.02:
                # Two frames a fifth of a second apart show the same thing, and
                # there is no reason to take the softer one.
                better = candidate.sharpness > current[0].sharpness
            else:
                better = rank > ranked[role]

            if better:
                chosen[role], ranked[role] = (candidate, score), rank

    return chosen


def write_ladder(kept: list[Candidate], args: argparse.Namespace) -> list[dict[str, object]]:
    """Write one frame per band of yaw, and the manifest entries for them.

    Returns one row per rung — angle, frame, and the near-eye detail it kept —
    so a caller that is deciding whether a clip is worth paying for again can
    read the same numbers this prints. See ``--report``.
    """
    import cv2

    rungs = turn_ladder(kept, args.ladder_step)
    print(f"\nturn ladder: {len(rungs)} rung(s), one per {args.ladder_step:g} degrees")
    print(f"{'file':<30}{'yaw':>8}{'frame':>7}{'face px':>9}{'near eye':>10}")

    detail = {candidate.index: eye_sharpness(candidate) for candidate in rungs}
    sharpest = max(detail.values(), default=0.0)

    entries: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for candidate in rungs:
        yaw = yaw_of(candidate)
        name = f"turn_{'m' if yaw < 0 else 'p'}{abs(yaw):04.1f}".replace(".", "_") + ".png"
        cv2.imwrite(str(args.out / name), crop_around_face(candidate))
        entries.append(
            {"path": name, "role": "source_turn", "optional": True, "yaw": round(yaw, 1)}
        )
        share = detail[candidate.index] / sharpest if sharpest > 0 else 1.0
        mark = "  <- blurred" if share < LADDER_MIN_DETAIL else ""
        print(
            f"{name:<30}{yaw:>8.1f}{candidate.index:>7}{candidate.face_width:>9.0f}"
            f"{share:>9.0%}{mark}"
        )
        rows.append(
            {
                "path": name,
                "yaw": round(yaw, 1),
                "frame": candidate.index,
                "face_px": round(candidate.face_width),
                "detail": round(share, 3),
                "blurred": share < LADDER_MIN_DETAIL,
            }
        )

    floor = sharpest * LADDER_MIN_DETAIL
    soft = [index for index, value in detail.items() if sharpest > 0 and value < floor]
    if soft:
        print(
            f"\n{len(soft)} of {len(rungs)} rungs are motion blurred — "
            f"under {LADDER_MIN_DETAIL:.0%} of the sharpest rung's near-eye detail."
        )
        print("The clip was generated with the head moving continuously, so every frame")
        print("during the turn is smeared and the per-frame blur floor cannot see it:")
        print("it compares each frame with the third of a second around it, and all of")
        print("those are blurred alike. Geometry cannot recover this — the avatar will")
        print("be sharp facing forward and soft at every angle.")
        print("Regenerate the clip with the head **pausing at each angle**, or turning")
        print("slowly enough that a frame is still, and extract again.")

    manifest = args.out / "ladder_assets.json"
    manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"\nManifest entries: {manifest}")
    print("Paste them into the avatar's metadata.json 'assets' list.")
    return rows


def write_report(path: Path | None, payload: dict[str, object]) -> None:
    """Write the JSON summary, if one was asked for.

    ``error`` is set instead of the measurements when the run failed for a
    reason that has nothing to do with the clip — a missing model bundle, an
    unreadable portrait. A caller that regenerates on failure must be able to
    tell those apart from a bad clip, or a broken checkout spends money.
    """
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="The clip to pull frames from.")
    parser.add_argument(
        "--base",
        type=Path,
        default=REPO_ROOT / "avatars" / "my_avatar" / "portrait.png",
        help="The portrait every frame is checked against.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Where to write the frames.")
    parser.add_argument(
        "--min-face",
        type=float,
        default=380.0,
        help="Reject frames whose face is narrower than this, in pixels.",
    )
    parser.add_argument(
        "--ladder",
        action="store_true",
        help=(
            "Also write a turn ladder: the sharpest frame in each band of yaw, "
            "with the manifest entries to paste in. For a head-turn clip this "
            "is the point of the clip — the engine selects the nearest angle "
            "instead of averaging two that are not it."
        ),
    )
    parser.add_argument(
        "--ladder-step",
        type=float,
        default=LADDER_STEP_DEGREES,
        help="Degrees of yaw per ladder rung.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Also write everything printed here as JSON, for a caller that has to "
            "decide whether the clip is usable without a human reading the output. "
            "scripts/generate_avatar_clips.py uses it to judge a clip it just paid for."
        ),
    )
    args = parser.parse_args(argv)

    model_path = REPO_ROOT / "models" / DEFAULT_MODEL_RELATIVE_PATH
    if not model_path.is_file():
        print(f"Model bundle missing: {model_path}")
        print("Fetch it with: python scripts/fetch_face_landmarker.py")
        write_report(args.report, {"error": f"model bundle missing: {model_path}"})
        return 1

    base_image = cv2.imread(str(args.base), cv2.IMREAD_COLOR)
    if base_image is None:
        print(f"Could not read the base portrait: {args.base}")
        write_report(args.report, {"error": f"unreadable base portrait: {args.base}"})
        return 1

    tracker = MediaPipeFaceTracker(model_path=model_path)
    tracker.start()
    try:
        base = tracker.process(
            VideoFrame(
                data=np.ascontiguousarray(base_image),
                pixel_format=PixelFormat.BGR24,
                metadata=FrameMetadata(frame_id=0, capture_timestamp=0.0),
            )
        )
        if not base.face_detected or base.landmarks is None:
            print(f"No face found in the base portrait {args.base}.")
            write_report(args.report, {"error": f"no face in the base portrait: {args.base}"})
            return 1

        base_ratios = identity_ratios(
            base.landmarks.points, base_image.shape[1], base_image.shape[0]
        )
        base_face = (
            float(base.landmarks.points[:, 0].max() - base.landmarks.points[:, 0].min())
            * base_image.shape[1]
        )
    finally:
        tracker.stop()

    # A fresh tracker for the clip: the base portrait is an unrelated still, and
    # the tracker carries a prior between frames.
    tracker = MediaPipeFaceTracker(model_path=model_path)
    tracker.start()
    try:
        candidates, rejected = read_candidates(args.video, tracker, base_ratios)
    finally:
        tracker.stop()

    total = len(candidates) + rejected["no face"]
    print(f"{args.video.name}: {total} frames")
    print(f"base portrait face width: {base_face:.0f} px")
    print()

    kept = usable(candidates, args.min_face, rejected)
    for reason, count in sorted(rejected.items()):
        if count:
            print(f"  rejected, {reason:<18} {count}")
    print(f"  usable                    {len(kept)}")

    if not kept:
        print("\nNothing usable. Lower --min-face, or generate a sharper clip.")
        write_report(
            args.report,
            {
                "video": str(args.video),
                "frames": total,
                "usable": 0,
                "rejected": dict(rejected),
                "roles": {"covered": [], "missing": list(_VARIANT_ROLES)},
                "ladder": None,
            },
        )
        return 1

    widths = [c.face_width for c in kept]
    print(
        f"\n  face width: min {min(widths):.0f}  median {float(np.median(widths)):.0f}  "
        f"max {max(widths):.0f} px"
    )

    chosen = best_per_role(kept, resting_face(base))
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'role':<26}{'frame':>7}{'score':>8}{'face px':>9}")
    for role in _VARIANT_ROLES:
        picked = chosen.get(role)
        if picked is None:
            print(f"{role:<26}{'—':>7}{'':>8}{'':>9}   not in this clip")
            continue
        candidate, score = picked
        name = role.replace("source_", "portrait_") + ".png"
        cropped = crop_around_face(candidate)
        cv2.imwrite(str(args.out / name), cropped)
        print(
            f"{role:<26}{candidate.index:>7}{score:>8.2f}{candidate.face_width:>9.0f}"
            f"{cropped.shape[1]:>7}x{cropped.shape[0]}"
        )

    missing = [role for role in _VARIANT_ROLES if role not in chosen]
    print(f"\n{len(chosen)} of {len(_VARIANT_ROLES)} roles covered by this clip.")
    if missing:
        print("Not covered — another clip, or the existing photographs, must supply:")
        print("  " + ", ".join(r.replace("source_", "") for r in missing))

    ladder: dict[str, object] | None = None
    if args.ladder:
        rungs = write_ladder(kept, args)
        yaws = [float(row["yaw"]) for row in rungs]  # type: ignore[arg-type]
        ladder = {
            "step_degrees": args.ladder_step,
            "min_detail_fraction": LADDER_MIN_DETAIL,
            "rungs": rungs,
            "blurred": sum(1 for row in rungs if row["blurred"]),
            "yaw_min": min(yaws, default=0.0),
            "yaw_max": max(yaws, default=0.0),
        }

    write_report(
        args.report,
        {
            "video": str(args.video),
            "frames": total,
            "usable": len(kept),
            "rejected": dict(rejected),
            "face_width": {
                "min": round(min(widths), 1),
                "median": round(float(np.median(widths)), 1),
                "max": round(max(widths), 1),
            },
            "roles": {
                "covered": [role for role in _VARIANT_ROLES if role in chosen],
                "missing": missing,
                "known": list(_VARIANT_ROLES),
            },
            "ladder": ladder,
        },
    )

    print(f"\nWritten to {args.out}")
    print("Check them before use:")
    print(f"  python scripts/check_avatar_photos.py {args.base} {args.out}/*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
