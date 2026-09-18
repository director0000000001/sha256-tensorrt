"""Record a guided clip of the owner's face, to drive the engine offline.

**Why this exists.** Quality defects in this engine are motion defects — a
ghosted pupil while the eye moves, a flicker, a shape that only breaks at an
angle. A screenshot cannot show them, and describing them costs a round trip per
attempt: the owner looks, reports, the engine changes, the owner looks again. An
hour per idea.

With one recorded clip the loop closes. The same forty seconds of real face can
be replayed through the engine as many times as needed, before and after any
change, without the owner present and without the camera. Twenty variations in
the time one round trip used to take.

**It records the camera, not the avatar.** The avatar is what the engine
produces and changes with every fix; the camera frames are the *input*, and they
are what makes a comparison fair — the same face, the same light, the same
movements, every time.

The prompts are timed and their frame ranges are written alongside, so the
replay can jump straight to "the blink" or "the smile" instead of hunting.

**Privacy.** This is the owner's real face. It is written under ``recordings/``,
which is gitignored, it never leaves the machine, and it should be deleted when
the work that needed it is finished. Nothing here uploads anything.

Usage::

    python scripts/record_driving_clip.py
    python scripts/record_driving_clip.py --out recordings/second-try.mkv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sha256.camera.opencv_capture import OpenCVCameraProvider  # noqa: E402
from sha256.config.loader import load_settings  # noqa: E402
from sha256.config.paths import AppPaths  # noqa: E402

# Each step is (seconds, instruction). Ordered so the calibration comes first —
# the engine learns the resting face in about a second and everything after is
# measured against it, so a clip that opens mid-expression teaches it the wrong
# neutral and every later judgement inherits that.
STEPS: tuple[tuple[float, str], ...] = (
    (4.0, "SIT STILL. Look straight at the camera. Relax your face."),
    (5.0, "Look slowly LEFT, then back to centre."),
    (5.0, "Look slowly RIGHT, then back to centre."),
    (4.0, "Look UP, then DOWN, then centre."),
    (4.0, "Blink normally, several times."),
    (3.0, "Close your eyes for two seconds, then open."),
    (4.0, "Smile with your mouth closed."),
    (4.0, "Smile widely, showing your teeth."),
    (4.0, "Open your mouth wide, then close it."),
    (8.0, "TALK. Say anything, normally, as if on a call."),
    (5.0, "Turn your head slowly LEFT, then back."),
    (5.0, "Turn your head slowly RIGHT, then back."),
    (4.0, "Raise your eyebrows, then frown."),
    (3.0, "SIT STILL again. Relax."),
)

# The speech script, added 2026-09-03 for one question the long clip cannot
# answer: **the mouth at speed.**
#
# The owner tested the tracker's lip filter and said it was worse — *"the mouth
# barely follows, it has closed up, it does not move freely"* — while every
# number measured on `driving.mkv` said it was better. That clip holds **eight
# seconds of ordinary talking** and no fast speech at all, so it cannot show the
# thing being argued about. A clip that can is one command, and after it every
# setting is answerable offline instead of costing him another round trip.
#
# The still four seconds at the front are not optional in either script: the
# engine learns the resting face in about a second and everything after is
# measured against it.
SPEECH_STEPS: tuple[tuple[float, str], ...] = (
    (4.0, "SIT STILL. Look straight at the camera. Relax your face."),
    (8.0, "TALK normally, as if on a call."),
    (8.0, "TALK FAST. As fast as you can while still being understood."),
    (6.0, 'Say "make, make, make" quickly, over and over.'),
    (6.0, "COUNT fast, out loud: one two three four five six..."),
    (4.0, "TALK FAST again, a different sentence."),
    (3.0, "SIT STILL again. Relax."),
)

# The natural script, added 2026-09-05 (day 6) for the two things the owner
# sees live that the guided clip never holds: *a face simply kept still* (the
# avatar "tries expressions on its own") and *the head tilted slightly down*
# (the avatar turns sulky). Both happen in a call posture -- eyes on the
# screen, a little below the camera -- which the guided clip's "look straight
# at the camera" instructions never produce. The still four seconds at the
# front stay: the anchor is only learned from frontal frames (see
# ``_anchor_frame_ok``), and without them there is no anchor at all.
NATURAL_STEPS: tuple[tuple[float, str], ...] = (
    (4.0, "SIT STILL. Look straight at the camera. Relax your face."),
    (10.0, "Now look at the SCREEN, as on a call. Stay still, relaxed. Do nothing."),
    (10.0, "Tilt your head SLIGHTLY down, hold two seconds, come back. Three times."),
    (20.0, "TALK normally, as if on a call. Move the way you naturally would."),
    (5.0, "Smile, then relax."),
    (5.0, "Raise your eyebrows, then frown, then relax."),
    (3.0, "SIT STILL again. Relax."),
)

# The turn script, added 2026-09-17 (day 19) for the client's "the face is
# lost when the head turns". The guided clip turns the head "slowly LEFT,
# then back" in five seconds and never reaches the angle where the tracker
# lets go; his live look of 2026-09-15 did (143 of 300 frames with no face,
# 22 seconds of black screen) and carried no picture of the turn itself.
# This clip holds the three things that have to be measured: how far round
# the tracker follows (slow full turns), how long a held profile stays lost
# and at what angle the face is found again on the way back (held
# profiles), and whether an ordinary glance is ever lost at all (quick
# glances). The still four seconds at the front stay, as in every script.
TURN_STEPS: tuple[tuple[float, str], ...] = (
    (4.0, "SIT STILL. Look straight at the camera. Relax your face."),
    (6.0, "Turn your head SLOWLY all the way LEFT, to a full profile, then slowly back."),
    (6.0, "Turn your head SLOWLY all the way RIGHT, to a full profile, then slowly back."),
    (5.0, "Turn LEFT to a full profile and HOLD it there. Come back on the next line."),
    (5.0, "Turn RIGHT to a full profile and HOLD it there. Come back on the next line."),
    (4.0, "Look at the camera. Quick GLANCE left and back, right and back, twice."),
    (6.0, "Turn HALF way left, hold two seconds, back. Then half way right, hold, back."),
    (3.0, "SIT STILL again. Relax."),
)

SCRIPTS: dict[str, tuple[tuple[float, str], ...]] = {
    "full": STEPS,
    "speech": SPEECH_STEPS,
    "natural": NATURAL_STEPS,
    "turn": TURN_STEPS,
}

LEAD_IN = 5.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=None,
        help="Where to write the clip. Gitignored. Defaults to the script's own name.",
    )
    parser.add_argument(
        "--script",
        choices=sorted(SCRIPTS),
        default="full",
        help="'full' is the 62-second guided clip; 'speech' is 39 seconds of the mouth at "
        "speed; 'natural' is 57 seconds of a call posture: still, head slightly down, talk; "
        "'turn' is 39 seconds of the head turned: slow full turns, held profiles, glances.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing clip of the same name. Without it, an existing clip is "
        "kept -- on 2026-09-05 a re-run silently replaced the clip every measurement of "
        "the day had been made on.",
    )
    args = parser.parse_args()
    steps = SCRIPTS[args.script]

    import cv2

    paths = AppPaths.resolve()
    settings = load_settings(paths)
    camera = OpenCVCameraProvider(
        device_index=settings.camera.device_index,
        width=settings.camera.width,
        height=settings.camera.height,
        target_fps=settings.camera.target_fps,
        use_mjpeg=settings.camera.use_mjpeg,
    )

    default_name = "driving.mkv" if args.script == "full" else f"driving-{args.script}.mkv"
    out = Path(args.out) if args.out else REPO_ROOT / "recordings" / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.overwrite:
        print(f"{out} already exists. Pass --out <new name> to keep it, or --overwrite.")
        return 2

    total = sum(seconds for seconds, _ in steps)
    print("=" * 68)
    print("Recording a driving clip")
    print("=" * 68)
    print(f"output   : {out}")
    print(f"length   : {total:.0f} seconds")
    print()
    print("Before you start:")
    print("  * Light on your FACE, from the front. A shadow across the face")
    print("    makes the tracker put eyebrows along the shadow's edge.")
    print("  * Camera squarely in front of you, at about eye height.")
    print("  * Close Discord and anything else that uses the camera.")
    print()
    print("Read each instruction and do it at a normal, unhurried pace.")
    print()

    camera.start()
    try:
        first = None
        for _ in range(30):
            first = camera.read()
            if first is not None:
                break
        if first is None:
            print("The camera returned no frames.")
            return 2

        height, width = first.data.shape[:2]
        fps = settings.camera.target_fps

        # FFV1 is lossless: compression artifacts would move the landmarks
        # slightly, and a comparison between two engine versions has to differ
        # only by the engine. MJPG is the fallback if the codec is unavailable.
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
        codec = "FFV1 (lossless)"
        if not writer.isOpened():
            fallback = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(out), fallback, fps, (width, height))
            codec = "MJPG (fallback)"
        if not writer.isOpened():
            print("Could not open a video writer.")
            return 2

        print(f"codec    : {codec}")
        print(f"frames   : {width}x{height} @ {fps:.0f}")
        print()
        for remaining in range(int(LEAD_IN), 0, -1):
            print(f"  starting in {remaining}...", flush=True)
            deadline = time.perf_counter() + 1.0
            while time.perf_counter() < deadline:
                camera.read()

        print()
        segments = []
        frames = 0
        for index, (seconds, instruction) in enumerate(steps, start=1):
            print(f"[{index:>2}/{len(steps)}]  {instruction}", flush=True)
            started = frames
            deadline = time.perf_counter() + seconds
            while time.perf_counter() < deadline:
                frame = camera.read()
                if frame is None:
                    continue
                writer.write(frame.data)
                frames += 1
            segments.append(
                {
                    "instruction": instruction,
                    "first_frame": started,
                    "last_frame": max(frames - 1, started),
                }
            )

        writer.release()
    finally:
        camera.stop()

    labels = out.with_suffix(".json")
    labels.write_text(
        json.dumps(
            {
                "video": out.name,
                "width": width,
                "height": height,
                "fps": fps,
                "frames": frames,
                "codec": codec,
                "segments": segments,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Wrote {frames} frames to {out}")
    print(f"Wrote the segment map to {labels}")
    print()
    print("This is your real face. It stays on this machine, it is gitignored,")
    print("and it should be deleted when the work that needed it is done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
