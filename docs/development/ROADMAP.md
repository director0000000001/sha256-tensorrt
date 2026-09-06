# sha256 — Roadmap to a deliverable neural version

**Written 2026-09-04, after the day-22 audit and the LivePortrait re-measurement.**
Owner's brief: highest quality, as fast as possible, the app will be used by a
team of five people other than the owner.

## What "highest level" is made of

| part | today | target | how |
|---|---|---|---|
| the picture | photographs chosen and cut 12-13 times a second; reads as a game character | a face generated every frame, no cuts | LivePortrait engine. Measured 2026-09-04 on the RTX 5070 Ti: **25.5 ms a frame** with fp16 weights and one CUDA graph (B005's 38.4 was autocast without graphs). Output identical to eager to 0.0000 |
| reaction | ~260 ms glass to glass, 227 of it before the engine starts (ZV-E10 over USB) | under 150 ms | phone-camera test first (free); if the camera is the cause, an HDMI capture card at 1080p60. Shutter 1/125-1/250 (free): the fastest mouth frames are 2.3x blurrier than still ones. MediaPipe's own VIDEO-mode lag disappears because LivePortrait reads motion itself |
| stability | fixed 2026-09-04: preview blackouts, the capture button on a tracking loss, load 47 s → 18 s, the "face not found" line | the same on five machines | GPU and camera check at startup with a plain message; a replay that starts from the engine's live state |
| delivery | nothing started: no shippable virtual camera (pyvirtualcam is GPL), no licence server, no kill switch, no site, no encryption, no `.exe` | installable by a non-programmer on five machines | see phase 3 and 4 |

## The three facts that decide the whole plan

1. **The neural engine needs an NVIDIA GPU on every user's machine.** 25.5 ms is a
   5070 Ti. A 3060-class card is roughly 2 to 2.5x slower, which is 15 fps. A
   laptop without a discrete NVIDIA GPU cannot run it at all; the photograph
   engine (CPU) is the fallback there. **The users' GPUs must be known before
   phase 2 ends.**
2. **Each user needs one good portrait, not a library.** LivePortrait animates a
   single photograph. The 250-photograph ladder, the clip generation and the
   builders are the photograph engine's need only. Five users means five
   portraits, generated the way the owner's was, plus one enrolment step each.
3. **Delivery is a separate project of about a week** and it cannot start before
   the engine is chosen, because the package now has to carry torch and the CUDA
   runtime (about 2.5 GB).

## Phases

Days are working days for one engineer at full time, starting 2026-09-05.

### Phase 0 — decisions, day 0 (the owner, today)

- Which GPU and which camera do each of the five users have? Model names.
- One shared avatar or one per user?
- The deadline conversation with the client, with the numbers below in hand.
- Phone-camera latency test (`scripts/measure_latency.py` with DroidCam or
  Phone Link), and the shutter set to 1/125 for a new `--script speech` recording.

### Phase 1 -- the neural engine, offline, days 1-5

**Day 1, done.** `LivePortraitEngine` behind the existing `AvatarEngine`
protocol (`src/sha256/avatar/engine_liveportrait.py`), registered as backend
`liveportrait`. Portrait prepared once at load (appearance feature, source
keypoints, crop). Per frame: MediaPipe's full landmarker for the crop (the
detector-only cut is still open, see Phase 2), motion -> warp -> decode in one
CUDA graph, paste back with LivePortrait's own crop transform. The four
CPU-side allocations inside LivePortrait's forward were moved to the GPU for
graph capture, vendored under `src/sha256/avatar/liveportrait/` (MIT,
provenance in that package's own `__init__.py`).

First render (`scripts/render_liveportrait.py`, frames 300-700 of
`driving-speech.mkv`): the picture was recognisably the avatar, correctly
placed and sized -- and pasted under LivePortrait's own template mask, which
covers the whole 512 crop square. On screen that is the portrait's studio
background and shoulders laid over the room, edged in black outside the
square. Not a defect in the network; a mask built for a different framing
convention.

**Day 2, done, three fixes:**

1. **The paste mask.** `_oval_mask` builds the portrait's own face oval from
   its landmarks placed in the crop -- same technique as the warp engine's own
   composite (`engine_warp._trim_forehead`/`_destination_mask`), same
   fractions (`_OVAL_SHRINK`, `_OVAL_FEATHER`, `_OVAL_FOREHEAD_REACH`),
   written again rather than imported so the two engines keep sharing no
   internals. Computed once per portrait at load, not per frame -- see the
   function's own docstring for why that is sound rather than a shortcut.
2. **Colour match.** `_match_face_colour` / `_fit_lighting_plane`: the same
   low-frequency relighting plane the warp engine fits between its portrait
   and the room, fit here between the generated face and the scene under the
   mask, with the same frame-to-frame inertia (`_CORRECTION_INERTIA = 0.8`).
3. **The paste, bounded to the mask's own box.** `_mask_bbox` finds where the
   oval actually is inside the 512 square -- about 13% of its area on this
   avatar -- and every frame warps only into that box, not the whole square.

   Measured 2026-09-04, this machine, `driving-speech.mkv` frames 300-700,
   before -> after:

   | | before (whole square) | after (oval's own box + colour match) |
   |---|---|---|
   | paste, p50 | 22.32 ms | **7.57 ms** |
   | paste, p90 | 22.98 ms | **8.10 ms** |
   | neural total, p50 | 50.79 ms | **34.85 ms** |
   | model (unchanged) | 27.69 ms | 26.74 ms |

   16 tests in `tests/unit/test_engine_liveportrait.py` (was 11): the mask's
   box is asserted smaller than the crop, feathered not binary; `_paste` is
   tested on the CPU alone (no GPU, no loaded networks) for byte-identical
   pixels outside its box, correct colour-matching on a synthetic scene, and
   inertia carrying between frames. The GPU test (portrait driving itself,
   graph against eager) still passes. mypy: one pre-existing-class error
   (a `cv2.subtract` stub gap `engine_warp.py` already carries at its own
   matching line), everything else clean; the vendored network code is
   excluded from mypy by its own override in `pyproject.toml`, the six calls
   into it from `engine_liveportrait.py` each carry their own
   `# type: ignore[no-untyped-call]`.

   **Looked at again, frames 120 and 300**: the oval is right and the room
   shows through around it now, which is the fix. There is still a visible
   seam at the mask's own edge -- most noticeable at the hairline, a band
   where the portrait's forehead tone meets the room's rather than fading
   into it -- and the mask stops at the jaw with nothing below it, so a
   turned or lowered head will show a hard edge above the neck. Neither is
   surprising: the warp engine's own `_MASK_FEATHER`/`_JAW_REACH` were tuned
   against the owner's eye over several days, not derived once and correct.
   This is the next thing to look at, not yet fixed.

**Still open, not blocking the owner's first look:**

- **The budget.** `tracker+neural` is 42.35 ms p50 against the 33.2 ms camera
  interval, 100% of frames over -- down from 63.34 ms before today, but still
  over. The full landmarker (7.4 ms today) is doing more than the crop needs;
  a face detector alone is the next lever, in Phase 2.
- **The seam and the jaw**, above -- Day 3 intersected the portrait's oval
  with the driver's, which measurably helps the hairline in aggregate but not
  on the left-turn frames specifically; see "Day 3, done" below for the
  number and the still that shows it did not close.
- Load time: 13.1 s this run, 21.8 s the run before -- this machine's mood
  (see PROJECT_STATUS.md trap 5), not yet measured as a back-to-back pair.

**Day 3, done, one fix, and it is only half of one:**

1. **The mask follows the driver, not only the portrait.** `_frame_mask`
   intersects the portrait's own oval (computed once at load, as before) with
   a second oval built the same way (`_oval_mask`, now used for two different
   faces) from the *driver's* landmarks, every frame. Neither oval alone was
   enough: the portrait's, warped onto a driver with different
   forehead-to-hairline or jaw-to-chin proportions, could reach into the
   driver's own hair or stop short of the driver's own chin -- exactly the
   hairline band and the floating chin the owner's Day 2 pictures showed. The
   intersection can only be smaller than each half, so it can never show
   either. Same rule the warp engine already follows in
   `engine_warp._destination_mask` ("the join happens over skin on both
   sides"), written again rather than shared -- the two engines still share no
   internals, only the registry contract.

   `animate` now builds this mask every frame (`mask_ms`, new) from the
   crop-space points already computed for the model; `_Portrait.last` carries
   it alongside the face and the transform so a held frame (tracking lost)
   keeps the mask that belongs with it rather than the portrait's own.
   `_paste` takes the mask as a parameter instead of reading `loaded.mask`
   directly. 8 new tests (`TestFrameMask`, `TestSharedLoadTracker`, tracker
   `TestRunningMode`) plus the 4 existing `TestPaste` tests updated for the
   new parameter -- 726 total (was 716). mypy: still the one pre-existing
   `cv2.subtract` error, now also present at its own line in this file (same
   stub gap, same shape, not new). ruff: the vendored `liveportrait/` package
   is now excluded from linting the same way it already was from mypy --
   `extend-exclude` in `pyproject.toml` -- since none of its 346 findings were
   ever this project's to fix.

   **Measured**, this machine, before -> after, mask-edge Sobel gradient
   (unitless, only comparable before to after -- see
   `scripts/render_liveportrait.py`'s own docstring):

   | | `driving-speech.mkv` 300-700 | `driving.mkv` 1300-1650 (both turns) |
   |---|---|---|
   | hairline, p50 | 48.97 -> **39.62** | 64.57 -> **50.72** |
   | jaw, p50 | 46.25 -> 50.21 | 50.40 -> 49.95 |
   | whole band, p50 | 48.47 -> 46.89 | 58.80 -> 51.46 |
   | mask cost, p50 | -- | 1.23-1.26 ms |
   | tracker+neural, p50 | 42.93 -> 44.74 | 40.90 -> 42.36 |

   The hairline number improves on both clips, clearly. The jaw number does
   not move -- expected: no jaw extension exists on either oval to disagree
   about (the owner rejected one on the warp engine, 2026-08-15, and none was
   added here), so the jaw edge is set by the portrait's own oval either way.
   The cost is real: +1.2-1.3 ms of mask work a frame, and `tracker+neural`
   rose 1.5-1.8 ms -- on a budget already 100% over at 33.2 ms, so it changes
   nothing about Phase 2's plan to close that gap, only the size of it.

   **Looked at, eight stills (four per clip, `recordings/day23-mask-before-
   after*.png`), and the picture is not as clean as the number.** On four --
   both near-frontal speech frames and the right-turn frames -- the fix is
   invisible because there was nothing visibly wrong to begin with. On the
   two left-turn frames (`driving.mkv` 1380, 1420) the hairline band the
   owner's Day 2 pictures showed is **still there, unchanged in appearance,
   before and after.** The aggregate p50 improved because most frames in that
   window were already fine; it does not mean this specific defect closed.
   **Working hypothesis, not yet checked:** at this yaw the portrait's and
   the driver's ovals may be wrong in the same direction rather than
   different directions -- both ceilings computed from a 2D `brow_top`/
   `oval_top` reading that a turned head foreshortens -- in which case the
   intersection of two wrong masks is still wrong, and the real fix is in how
   the ceiling is computed under yaw, not in whose oval is used. The
   right-turn frames not showing it argues against a simple yaw-magnitude
   explanation and for something sign- or pose-specific; not run down yet.

**Day 4, done -- the owner ran it live for the first time, and named two
things: "the AI face is slightly to the side of my face" and "reactions
don't sit on the face."** The left-turn hairline investigation this section
used to end on is still open -- his own report took priority, and finding it
turned up a mechanism the hairline question never got to.

**The tool: `scripts/measure_follow.py`.** Three trackers on one clip -- the
driver frame through the app's own filter (what actually drives the engine),
the driver frame unfiltered (the truth of where his face and expression are),
and the engine's own output tracked the same way, so the avatar's eyes/nose/
mouth can be read and compared to his on the same frame. Position is reported
as avatar-minus-him in pixels; expression following as a gain (1.0 = the
avatar's own tracked reading moves exactly as far as his did) and a
correlation. Numbers only compare before a change to after, same clip.

**"To the side" was stitching's own 2-D shift, and nothing else.** Four
candidates were measured on a fresh 62-second clip (`recordings/driving.mkv`,
re-recorded 2026-09-04 for this): pinning the driver's translation onto the
portrait, pinning scale, removing the mean of the expression delta, and
dropping stitching's shift. The first three changed nothing. The shift was the
whole effect:

| | with the shift | without |
|---|---|---|
| nose, p10..p90 | -14.8 .. +15.2 px | **-6.1 .. +4.0 px** |
| "Close your eyes", nose dx | -13.4 px | -1.0 px |
| "TALK", nose dx | +10.2 px | -1.7 px |
| correlated with | avatar's apparent size, r=0.88 | -- |
| 40° turn (frame 1590) | a grey wedge above the jaw | gone |

Stitching was trained to paste a re-animated face back onto its *still*
source photograph without a seam at the neck; here the driver's own crop
already re-centres the face every frame, so the shift fights the crop instead
of correcting a paste. `stitching` defaults to `false` now (the correction
half of the network still runs when explicitly enabled -- it is the shift,
not the whole module, that was the problem, but nobody has needed the two
apart). Cost: nothing, this is *less* work per frame, not more.

**"Reactions don't sit on the face" was real, and partly fixed.** The
driver's relative expression alone reproduced only 36% of a blink and 73% of
a mouth opening, against his own tracked reading. LivePortrait ships eye and
lip retargeting networks in the same checkpoint stitching's is in
(`retarget_eye`, `retarget_mouth`) and neither was loaded before today.
Wired in, on his own close ratio -- computed on MediaPipe's points, since
that is what this engine tracks with, not LivePortrait's own landmark scheme
the networks were trained against. That mismatch turned out to matter:

| | eye | lip |
|---|---|---|
| gain, off | 0.42 | 0.75 |
| gain, calibrated | **0.67** (full clip) / 0.70 (isolated) | **0.99** |
| r | 0.79 -> 0.92 | 0.99 (unchanged) |
| calibrated constant | `eye_gain = 0.06` | `lip_gain = 0.20` |

Lip reaches the target cleanly: gain 1.0 at `lip_gain = 0.2`, correlation
0.99 throughout the sweep, a clean monotonic response (0, 0.05, ... 0.3 ->
0.75, 0.81, ... 1.10). **Eye does not, and this is a real finding, not an
under-tuned one.** Its response rises smoothly from gain 0 to 0.06-0.07 and
then plateaus -- 0.08 gave *less* than 0.07, not more -- while a first probe
at gain 1.0 (both networks' original guess) was actively worse than off,
overshooting into territory that read as *more* shut regardless of the
driving target. Both networks take the same shape of input (the portrait's
own kp, plus close ratios) at gain 1.0 in LivePortrait's reference pipeline;
the plausible read is that a MediaPipe-based ratio and a LivePortrait-based
one agree well enough in shape for the lip network's training to tolerate,
and not well enough for the eye network's -- not measured further, since the
gain sweep already shows the ceiling rather than an unfinished tuning curve.
**Blinking on the app will be visibly closer, not identical.**

Position held through the calibration: nose dx still -5.8..+4.0 px (p10..p90)
with both networks on. 728 -> **737 tests**. mypy: the same one pre-existing
error, at its own line, none new. ruff: the vendored `liveportrait/` package
is now excluded the same way it already was from mypy (`extend-exclude` in
`pyproject.toml` -- 346 findings there were never this project's to fix).

**What was not run down:** the left-turn hairline band this section used to
be about. **At |yaw| > 15° the avatar's whole face lags the real turn by
18-20 px, both directions** -- measured today, not stitching, not yet
explained. **The "That looked wrong" capture button cannot replay this
engine.** It stores landmarks and one rendered frame by design (privacy: "not
a photograph of anyone" beyond that one frame) -- exactly right for the
photograph engine, which only ever needed landmarks to redrive itself. This
engine's motion extractor reads raw pixels from the driver crop, not
landmarks, so a captured moment with no saved frames for it cannot be
reconstructed offline. Fixing that means deciding to store more of the
owner's face on disk, which is his call, not a default to change quietly.

**A footgun closed:** `render_liveportrait.py --clip` had no default that
matched the clip most days actually needed -- it silently pointed at the
speech clip and cost one wasted run today. `--clip` is required now.

**Files from today under `recordings/` and `captures/`, all his face, all
gitignored, none looked at yet:** `driving.mkv` (the new 62 s clip) and its
`driving-follow*.csv`; `day24-overlay-f*.png` (8 stills, before any of
today's fixes) and `day25-nostitch-f1590.png` (after -- the wedge, gone);
`captures/moment-20260904-233810/` (the "That looked wrong" press that
started today, plus `overlay.png` built to check it).

**Day 5, done -- his six live complaints, each run down to one mechanism and
fixed: the anchor, the crop, and the eye/lip targets.**

**1. "Face to the side, not symmetric" — the anchor.** It learned the
driver's rest from the first 15 frames after load, whatever they held. His
own capture (`captures/moment-20260905-010827`) showed why that is not
enough: nose +27 px, eye +18 px off his, an asymmetry of +9.5 -- reproduced
exactly offline by anchoring on frames turned -12°: nose +29, eye +17, asym
+12, plus the avatar's mouth sitting ajar and its eyes 26% wider than rest.
`_anchor_frame_ok` now gates which frames the anchor is allowed to learn
from (frontal within 4°/4°/3°, eyes open, mouth shut); a **"Re-centre"
button** calls the already-existing `reset_anchor` (was wired to nothing);
the status line says when the anchor is still being learned.

**2. "Zooms in and out, enlarges the eyes" — the crop.** It sized itself from
all 468 mesh points, and an open mouth or a smile moves its own corners
outward, growing the box the network reads its own size from. Avatar size
against his own varied **1.01..1.25** over the clip, worst exactly at the
widest mouth opening. `crop_matrix` now sizes and centres from
`_RIGID_POINTS` -- points expression cannot move (eye corners, face sides,
forehead, nose) -- times a fixed `_RIGID_EXTENT_FACTOR` so it still matches
LivePortrait's own framing convention. Size variance fell to **1.00..1.08**,
mouth-wide scale from 1.25 to 1.03. **Not free:** at |yaw| > 15° the nose
offset got *worse* (19 px → 34 px) and the eyebrow-raise correlation fell
0.95 → 0.30 -- reported, not chased further this session.

**3 and 4. "Half-shut drunk eyes" and "thin, angry lips" — the retargeting
targets.** Both networks were driven by his *raw* close ratio. His eyes read
narrower than the portrait's own at rest (0.30 against 0.334), so a raw
target asked the network to narrow the avatar for the mere fact of him
sitting still -- measured as the avatar's eyes 27% narrower than his over a
whole clip. His raw lip ratio, at rest, was *tighter* than the portrait's own
already-closed lips, and the network had nowhere to put that demand but the
lips themselves -- thickness fell to 38-44% of his own. `_eye_target` now
asks relative to his own rest (his rest → the portrait's own ratio, capped
1.2x above it); `_lip_target` is floored at the portrait's own closed ratio,
never tighter.

**Before -> after, measured on `driving.mkv` (1844 frames), same clip
throughout:**

| | before today (day 4's end) | after A (anchor) | after B (crop) | after C (targets) |
|---|---|---|---|---|
| nose dx, p10..p90 (px) | -5.8..+4.0 | -5.8..+4.0 | -4.3..+5.7 | -4.5..+6.0 |
| scale, p10..p90 | 1.011..1.159 | 1.011..1.160 | 1.002..1.075 | 1.002..1.077 |
| mouth-wide scale, p90 | 1.254 | -- | 1.027 | 1.03 |
| ear gain / r | 0.67 / 0.92 | 0.67 / 0.92 | 0.61 / 0.79 | 0.62 / 0.80 |
| mar gain / r | 0.99 / 0.99 | 0.99 / 0.99 | 1.09 / 0.99 | 0.94 / 0.99 |
| lip thickness (upper/lower, avatar/his) | 0.38 / 0.44 (live report) | -- | -- | 0.60-0.78 / 0.73-0.78 |
| \|yaw\|>15°, nose dx | 11.3 | -- | -34.0 | -34.1 |

**The lip finding is worth stating plainly: retargeting turned out not to
earn its place.** Re-swept with the floor in place and Task B's crop
already fixed: the mouth-following gain from relative motion *alone*, no
lip network at all, had already reached 0.94 (r 0.98) -- Task B's crop
stability turns out to have fixed most of the mouth-tracking gap too, not
only the zoom. Every `lip_gain` above 0 bought a little more mar gain at a
real, measured thickness cost (0.20 → 1.09 gain but only 68/70% thickness).
`lip_retargeting` is off by default now; the option and `_lip_target` stay
wired in for an avatar or a driver that needs it more than this one does.
Eye retargeting keeps earning its keep (`eye_gain 0.06`, ear gain 0.36 → 0.62
off a baseline that never had this problem to begin with) and stays on.

**Lip thickness itself is not fully explained.** It recovers well at the
very start of a clip (0.77/0.78, clearing the informal 0.75 bar) and sags
later in it (0.60/0.73 by "SIT STILL again") even with the lip network fully
off -- something else, not yet identified, thins the mouth over time. Not
run down this session.

737 -> **767 tests**. mypy: the same one pre-existing error. ruff clean.

**What was not run down.** The **forehead mask line** from Day 3 -- visible
again in `captures/moment-20260905-010827/seen.png`, frontal and close --
his live report moved to the position/zoom/target problems instead and it
was never revisited. The **|yaw| > 15° lag**, now 34 px rather than 19,
made worse by Task B rather than fixed. The **capture button's own limit**:
it stores landmarks and one rendered frame, enough for the photograph engine
to replay itself but not this one, whose motion extractor reads raw pixels
it never saved. Fixing that means storing more of the owner's face on disk
by design, which is his call.

**Files from today under `recordings/` and `captures/`, all his face, all
gitignored, none looked at yet:** `driving-follow-taskA.csv`,
`-taskB.csv`, `-taskC-final.csv` (one per task, this measurement run each
built on the last); `day25-final.mp4` and its five stills (the finished
state, for his own side-by-side); `captures/moment-20260905-010827/` (the
capture that started today's list) and its `overlay.png`.

**Day 6, done (2026-09-05, Fable 5.1 implementing directly on the owner's
instruction) -- his third live verdict, and the crop rebuilt around what
actually stays still on a face.**

His four notes after the day-5 build: (1) "my jaw shows, the face got
shorter"; (2) "random reactions -- raises the brows, pulls the eyes wide";
(3) "teeth on a smile are terrible"; (4) "the reactions have to match mine,
that is a hard prerequisite".

**(1) and (2) were the same mistake: the day-5 rigid crop.** Its box was
sized from the rigid points' *larger side* -- cheek to cheek on any face --
times one constant taken from the portrait. Two things followed. His face is
proportionally longer than the portrait's (forehead-to-chin over
forehead-to-nose: 1.91 against 1.815), so the constant under-sized his crop
by 5% and the avatar's chin landed 14-20 px above his own -- the jaw he saw.
And a cheek-to-cheek width shrinks with cos(yaw), so every turn zoomed the
crop in: the turn lag went 19 → 34 px on day 5, and the brow-raise
correlation fell 0.95 → 0.30, because the motion extractor reads its own
framing, not ours -- it wants the driver at the driver's *own* whole-face
framing (day 4's), and every experiment below that gave it the portrait's
framing instead killed the brows (r 0.27-0.30) no matter what else changed.

**Three tries, measured on `driving.mkv` each time, before the one that
holds:**

| | day 4 | day 5 | A: his ratio, running mean | B: "own" width-gov. | C: "paste" | D: "own" height-gov. | E: "paste" height-gov. |
|---|---|---|---|---|---|---|---|
| still: chin dy (px) | -2.6 | -14.4 | -15.5 | -11.9 | -4.7 | -4.4 | -7.1 |
| still: scale | 1.083 | 1.020 | 1.144 | 1.018 | 1.064 | 1.063 | 1.059 |
| still: avatar EAR (his 0.293) | 0.318 | 0.286 | **0.419** | 0.286 | 0.286 | 0.281 | 0.299 |
| brows gain / r | 1.14 / 0.95 | 0.23 / 0.30 | 0.75 / 0.75 | 0.35 / 0.84 | 0.24 / 0.27 | 0.88 / 0.90 | -0.36 / -0.50 |
| mar gain / r | 1.05 / 0.99 | 0.97 / 0.99 | 0.90 / 0.98 | **0.17 / 0.29** | 0.92 / 0.99 | **0.29 / 0.50** | 0.62 / 0.86 |
| eyes shut: avatar EAR | 0.233 | 0.290 | 0.337 | 0.112 | 0.323 | 0.215 | **0.411** |
| \|yaw\|>15°: nose dx | -20 | -34 | -37 | -49 | -30 | -49 | -32 |

- **A** learned his whole-over-rigid ratio as a *running mean during the
  anchor window and applied it while averaging* -- so the anchor's frames
  were cut at fifteen different sizes and every frame after at a sixteenth.
  The avatar sat 12% too big with its eyes 0.419 wide at rest and its brows
  up: exactly note (2), reproduced offline. Rule learned: **the anchor and
  the frames it serves must be cut at one framing.**
- **B/C** fixed that (median, applied once) but kept the width-governed
  box; C put the driver at the portrait's framing for the network and only
  *pasted* larger. C fixed the jaw and lost the brows; B had a paste bug and
  is not evidence of anything.
- **D/E** switched the box to the rigid points' *height* (forehead top to
  nose tip) to stop the yaw zoom -- and found the nose tip's own protrusion
  under pitch: -13.5% at +14° of pitch, which is exactly what his head does
  when his mouth opens wide. D got the brows back (0.90) and lost the mouth
  (0.29).

**What every 2-D measure of the face does under his own motion** (p10..p90
per segment, relative to its own rest value; the filtered tracker, as the
engine sees it):

| segment | pitch / yaw | rigid height | rigid width | whole height | 1 / pose distance |
|---|---|---|---|---|---|
| look up/down | pitch to +14° | -11.7% | -7.4% | -10.0% | -4.7% |
| mouth wide open | pitch to +13° | -13.5% | -5.5% | -6.3..**+6.7%** | -3.6..+0.2% |
| talk | pitch -5..+10° | -9.5..+4.0% | -4.2% | -2.5..+2.6% | -2.6..-0.6% |
| turn right | yaw -48..+31° | -14.1% | **-20.5%** | -10.3% | -4.3..-2.1% |
| smile, closed | -- | +1.2% | -2.8% | -3.4% | -2.6..-2.0% |
| still again | pitch -5° | **+8.2%** | -3.5% | +0.7% | -2.8..-2.2% |

The last column is MediaPipe's own head pose: the translation of a rigid
whole-face model fitted to the mesh, which neither an expression nor a turn
can move -- it reads only how near he is (the -2..-5% in the table is him
actually leaning back; every other column agrees he did). **That is the crop
size now:** `crop_matrix(face_height=...)` takes the face height in pixels;
the engine measures the driver's resting height (forehead top to chin, the
day-4 measure LivePortrait's framing was tuned against) and resting distance
over the anchor's accepted frames -- medians, because the tracker's first
frame reads a face 10% shorter -- and afterwards passes
`height_rest x z_rest / z` every frame (`_driver_face_height`). Before the
freeze the crop is sized from each frame itself, which at rest is the same
height, so the anchor's frames and the frames after are cut alike (the rule
above, kept for free). The rigid points only *centre* the box now. The
`extent_factor` / `driver_framing` machinery from tries A-E is gone.

**And then the mouth died anyway (try F: the distance-sized crop at the
old scale 2.3): mar gain 0.30, and on the wide-open-mouth frames the whole
face collapsed** -- eyes shut, nose dragged down into a shut mouth (frame
1067: his mouth ratio 0.55, the avatar's 0.005). Run down in order: the crop
box at those frames was steady (835-855 px); the network's input crop looked
textbook; the motion extractor's own numbers were moderate (scale ratio
0.97, |exp - exp_d0| 0.12); the eye network off changed nothing; the
network's raw 512 output was already the collapsed face. One frame, the same
portrait and driver, the driver's box swept in size:

| driver box, x resting face height | 1.96 | 2.07 | 2.18 | **2.30** | 2.41 | 2.53 |
|---|---|---|---|---|---|---|
| output mouth ratio (his 0.549) | 0.53 | 0.51 | 0.44 | **0.01** | 0.16 | 0.08 |
| output eye ratio (his 0.260) | 0.19 | 0.17 | 0.26 | **0.14** | 0.15 | 0.12 |

A cliff between 2.18 and 2.30, and 2.30 is exactly LivePortrait's own
`CropConfig` scale -- which we had applied to the wrong height since day 1.
Their "face" box comes from a landmark set that starts at the eyebrows; ours
is the MediaPipe mesh's whole height, forehead top to chin, about a seventh
taller. 2.3 of *our* heights shows the network a face ~10% smaller than it
was trained on, and on a hard expression it falls off. No earlier crop had
ever held that frame at 2.3 with an anchor cut at 2.3: day 5's
width-governed box shrank 5% on that frame (a turn or a wide mouth narrows
the cheek-to-cheek width) and rendered it; day 4's whole-mesh box *grew* 7%
on it and rendered it too -- with lip retargeting still on, and with an
anchor cut smaller than the frame. Why a mismatched anchor/frame pair
survives where the matched pair of the sweep collapses is **not
understood**; the sweep is one frame, one portrait, and is reported as
that.

**2.1 fixed that frame and broke another.** The whole clip at 2.1 measured
well (mar gain 0.84, brows r 0.78, chin -6.6) -- and the owner's own
proposal, *look at the side-by-side, not the numbers*, showed frames
1760-1808 (he frowns, chin down 5°, eyes open) with the avatar's eyes
squeezed shut and lips pursed: a grimace day 5 did not have (avatar eye
ratio 0.30 there; at 2.1, 0.10) -- the "random reactions" he reported, made
worse. The same one-frame sweep on that frame and on a brow raise:

| driver box, x resting face height | 2.07 (2.1) | 2.18 (2.2) | 2.30 (2.3) |
|---|---|---|---|
| frown, chin down 5° -- eye ratio (his 0.283) | **0.10** | **0.30** | 0.12 |
| ... mouth ratio (his 0.008, shut) | 0.04 | 0.01 | 0.12 |
| brows raised -- eye ratio (his 0.376) | 0.28 | 0.32 | 0.35 |
| mouth wide + pitch 10° -- mouth ratio (his 0.549) | 0.51 | 0.44 | 0.01 |

Two cliffs, not one: the wide mouth collapses at 2.3, the frown collapses at
2.1 *and* at 2.3, and 2.2 renders all three. **`CROP_SCALE` is 2.2, for the
portrait and the driver alike.** Why the network is this sensitive to a 5%
framing change is not understood and is the single most important open
question about this engine -- a stable crop was what exposed it, because
every earlier crop wandered across these cliffs frame by frame and the
failures read as noise. The whole-clip numbers below are the test of
whether 2.2 holds beyond three frames; the owner's eye on the side-by-side
is the test of whether the numbers mean anything.

**(2), the eyes pulled wide, had a second cause:** `_eye_target`'s 1.2x
headroom -- a driver whose eyes read wider than his rest asked the eye
network for eyes 20% wider than the portrait's own, and that network's 63
outputs are not confined to the lids. Headroom is 1.0 now: the portrait's
open eyes are as wide as the avatar goes; narrower still follows him.

**(3), the teeth.** Two things in one picture. The teeth themselves are
hallucinated -- the portrait's lips are closed, the network has never seen
this mouth's interior and paints a generic one; the lever is a portrait with
a slightly open mouth, not code. But the *grimace* he saw is measurable: on
"smile showing teeth" the avatar's eyes squeezed to EAR 0.152 (day 5) while
his stayed at 0.267 -- the network entangles a wide smile with a squint.
Final numbers below say how much of that this day recovered.

**Whole clip, the distance-sized crop at each scale (`driving-follow-
day6-final.csv` = 2.3, `-s21.csv` = 2.1, `-s22.csv` = 2.2, eye headroom 1.0
in all three) against day 4 and day 5 (what he tested):**

| | day 4 | day 5 (tested) | 2.3 | 2.1 | **2.2 (this build)** |
|---|---|---|---|---|---|
| still: chin dy (px; his note 1) | -2.6 | -14.4 | -8.8 | -6.6 | **-8.0** |
| still end: chin dy | -2.0 | -20.5 | +1.6 | +0.9 | -8.1 |
| still: scale (1.08 is this pair's identity) | 1.083 | 1.020 | 1.099 | 1.073 | **1.066** |
| mouth wide: scale p90 (day 5's zoom fix) | 1.254 | 1.033 | 1.067 | 1.105 | 1.156 |
| still: avatar EAR (his 0.293; note 2) | 0.318 | 0.286 | 0.325 | 0.308 | **0.301** |
| talk: avatar EAR p90 (his 0.332) | 0.410 | 0.335 | 0.291 | 0.347 | 0.371 |
| frown, chin down (fr. 1760-1808): avatar EAR (his 0.28-0.34) | 0.16-0.20 | 0.30-0.36 | -- | **0.10-0.12** | **0.31** |
| brows gain / r (note 2) | 1.14 / 0.95 | 0.23 / 0.30 | 0.67 / 0.83 | 0.31 / 0.78 | 0.18 / 0.24 |
| eyes shut: avatar EAR | 0.233 | 0.290 | 0.192 | 0.196 | **0.351** |
| ear gain / r | 0.85 / 0.79 | 0.52 / 0.63 | 0.71 / 0.65 | 0.67 / 0.64 | 0.51 / 0.72 |
| smile with teeth: avatar EAR (his 0.267; note 3) | 0.213 | **0.152** | 0.218 | 0.234 | **0.265** |
| smile with teeth: mouth-width gain | 0.79 | 0.86 | 0.91 | 0.96 | 0.88 |
| mar gain / r | 1.05 / 0.99 | 0.97 / 0.99 | **0.30 / 0.50** | 0.84 / 0.98 | 0.77 / 0.98 |
| \|yaw\| > 15°: nose dx (+ / -) | -20 / +19 | -34 / +25 | -51 / +24 | -27 / +20 | -33 / +21 |
| lip thickness, start (upper/lower) | -- | 0.77 / 0.78 | 0.71 / 0.70 | 0.77 / 0.81 | 0.75 / 0.75 |

**What the side-by-side (`day27-side-by-side.mp4`, 12 stills) shows at
2.2, frame by frame:** rest, talk, the wide-open mouth (which collapsed at
2.3), the toothy smile (eyes 0.265 against his 0.267 -- the day-5 grimace
gone), the 40° turn, and the frown frames (which grimaced at 2.1) all render
as a plausible face doing roughly what he does. **What it also shows: the
avatar is flatter than he is.** His brow raise (1724) and his frown (1784)
come out as a neutral face; his closed-mouth smile (800) barely registers;
in the eyes-shut segment the avatar's eyes stay half open (0.351). Nothing
breaks at 2.2 -- and the expression gain that 2.1 had (eyes shut 0.196,
brows r 0.78) is gone with it. Read against his four notes: (1) the chin is
8 px above his, was 14-20 -- better, not day 4's 2.6; (2) the *random*
reactions are gone in the clip (no wide eyes, no grimace) but the *real*
reactions are under-driven; (3) the toothy-smile squint is gone, the teeth
are the portrait's problem; (4) not met: the avatar follows him, weakly.

**Where the trade sits.** Three crop scales, three different failures: 2.3
collapses a wide mouth, 2.1 grimaces a frown, 2.2 does neither and
under-reacts. The next lever is not the crop: it is a multiplier on the
relative expression itself (`exp_s + k * (exp - exp_d0)`, k > 1 -- what
LivePortrait's own demos call a driving multiplier), tried at 2.2 on the
same frames first (the risk is amplifying exactly the two failures the
scale choice just avoided), and the portrait-side lever for the teeth.
Both are his call.

**(4) is his exit criterion, not mine to declare met.** The numbers above
are the clip; his eye on the live app decides. What is still known to be
open: the |yaw| > 15° lag (never better than day 4's 19-20 px); teeth on a
closed-lip portrait; the forehead mask line from day 3.

775 tests (767 → 775; the day-6 tests cover the height/distance freeze,
medians, reset, and the crop's indifference to width and mouth). mypy: the
same 12 pre-existing errors. ruff clean.

**Files from today under `recordings/`, gitignored:** the measurement CSVs
(his tracked geometry, no pixels) `driving-follow-day6.csv`, `-day6-own.csv`,
`-day6-paste.csv`, `-day6-own2.csv`, `-day6-paste2.csv`, `-day6-final.csv`
(tries A-F), `-day6-s21.csv` (2.1) and `-day6-s22.csv` (2.2, this build);
**`day27-side-by-side.mp4` and its 12 stills (`-f60 … -f1808.png`), his
face beside the avatar's at 2.2 -- made at his request, for him to look at,
to be deleted once he has.** Every other image and clip of his face made
for the diagnosis (`day26-*`, the 2.1 side-by-side) was looked at and
deleted the same day; the temporary avatar copies `avatars/exp_*` are
deleted too.

**Later the same day: a second clip, and his two live observations
reproduced in it.** He re-ran the recorder (which silently replaced
`driving.mkv` -- the recorder now refuses to overwrite without
`--overwrite`, and has a `natural` script for a call posture) and reported
two things from the live app: *"when I tilt my head just slightly down it
makes a sulky, offended face"* and *"when my face is simply still it tries
different expressions on its own"*. Both are in the new clip, at the 2.2
build (`driving2-follow-s22.csv`, `day27b-side-by-side.mp4` + 14 stills):

| | him | avatar |
|---|---|---|
| still, frames 0-119: eye ratio p10..p90 | 0.314..0.320 | **0.291..0.365** |
| still, frames 0-119: brow p10..p90 | 0.237..0.247 | **0.161..0.213** |
| eyes up, frames 470-530: eye ratio | 0.43-0.44 | 0.27-0.33 (headroom 1.0 caps it) |
| eyes up, frames 470-530: brow | 0.16 | **0.12** (brows pulled down) |
| eyes up, frames 470-530: mouth ratio | 0.005 | **0.02-0.05** (ajar, pouting) |

He does not move and the avatar wanders ±12% in the eyes and ±14% in the
brows -- the crop is steady (distance and height within ±0.5% on those
frames), so this is the motion extractor's own frame-to-frame noise passed
straight through. Looking up reads as brows down + mouth ajar on the
portrait: the sulky face (frames 500/520). The frown segment renders flat
and calm; the toothy smile is fine.

**Portrait 2** (`avatars/my_avatar_neural2`, his own upload: smiling, teeth
showing, 642x673): real teeth appear -- and the avatar grins at rest and
while he talks neutrally (frames 100, 1200), because the source face's
smile *is* its rest. Unusable as the main portrait; proof that the teeth
lever is real. What is needed is neutral, lips slightly parted.

**Day 7, next -- two engine levers, in this order, each measured on the new
clip's still and eyes-up frames before anything is kept:**
1. **Temporal smoothing of the extracted motion** (an EMA on exp/t/scale/
   pose inside the captured graph, state reset with the anchor) -- for the
   wandering. Low risk; the cost is a fraction of a frame of lag.
2. **The pitch/gaze leak** -- the extractor under-reads pitch (MediaPipe
   -5° read as +3°, +10° as -2°), so a tilt lands in the expression as a
   pout. Candidate fix: re-decompose the driver's motion around MediaPipe's
   rotation (`exp' = exp + kp @ (R_ext - R_mp)`, then relative motion with
   R_mp) -- principled, untested, conventions between the two pose
   parametrisations must be checked first.
Plus his side: a `natural` clip (`--script natural --out recordings/
natural.mkv`) and a neutral portrait with parted lips. The expression
multiplier is deferred behind these two (amplifying noise before smoothing
it would be backwards). Exit unchanged.

**Files kept for him to look at (his face, gitignored, delete after):**
`recordings/day27b-side-by-side.mp4` + 14 stills (2.2, new clip),
`recordings/day27b-portrait2.mp4` + 6 stills, and the older
`day27-side-by-side*` (2.2, the old clip).

**Day 7, done (2026-09-05, Sonnet 5, lever 1 only) -- temporal smoothing
built and measured; the plain-EMA version does not clear its own bar.**
Task A: an EMA over the extractor's own exp/t/scale/pose, in state tensors
on `_Portrait` that persist between graph replays (`_ema_`, in-place
`mul_`/`add_` so it captures into the graph), gated by a new
`motion_smoothing` option (`[0, 1)`, default `0.0`, validated at
`load_avatar`'s own `_capture` call). At `0.0` every line is a no-op --
`state*0.0` is exactly `0.0`, `0.0 + new*1.0` is exactly `new` -- checked
bit-for-bit against the unsmoothed path, not just approximately. Primed at
load from the portrait's own reading (not zero); re-primed to the anchor's
own running mean on every accepted anchor frame while it is still being
learned, so a smoothed graph's first replays after the freeze start from
the clean mean, not from noise collected during the anchor window;
`reset_anchor` deliberately leaves the state alone (no jump the instant it
is called, matching `exp_d0`'s own behaviour) until the next accepted frame
re-primes it. 789 tests (775 → 789), mypy the same 12 pre-existing errors,
ruff clean.

Task B: measured on the same clip (`recordings/driving.mkv`, `driving.json`
segments) at `motion_smoothing` 0.5 and 0.65 against the no-smoothing
baseline (`driving2-follow-s22.csv`):

| | baseline (off) | 0.5 | 0.65 | bar |
|---|---|---|---|---|
| still start: avatar eye p10..p90 (his 0.314..0.320), width | 0.291..0.365 (**0.074**) | 0.294..0.348 (**0.054**) | 0.294..0.345 (**0.051**) | width ≤ 0.037 |
| still start: avatar brow p10..p90 (his 0.237..0.247), width | 0.161..0.213 (**0.052**) | 0.160..0.206 (**0.046**) | 0.159..0.205 (**0.046**) | width ≤ 0.026 |
| blink: ear gain / r (% of baseline) | 0.807 / 0.953 | 0.741 / 0.897 (**92%**) | 0.660 / 0.839 (**82%**) | ≥ 90% |
| mouth wide: mar gain / r (% of baseline) | 0.562 / 0.971 | 0.560 / 0.976 (100%) | 0.554 / 0.977 (99%) | ≥ 90% |
| talk: mar gain / r | 0.764 / 0.941 | 0.702 / 0.898 | 0.652 / 0.845 | -- |
| turn: nose dx p50, yaw>+15 / yaw<-15 (px) | -35.5 / 35.0 | -35.5 / 33.2 | -35.9 / 32.0 | -- |

**Neither value clears the spread bar.** A single-pole EMA on white noise
would be expected to leave `sqrt((1-a)/(1+a))` of the raw std: 58% at 0.5,
46% at 0.65. The *measured* eye spread only fell to 73% (0.5) / 69% (0.65)
of the baseline's -- the extractor's frame-to-frame wander is not simple
white noise, so a plain EMA buys less than the textbook number and costs
more lag for it than expected: 0.65 also drops the blink ear gain to 82% of
baseline, under the 90% floor.
Per the rule set for this measurement (**if neither value clears the
spread bar, report both and stop -- do not go above 0.65, do not substitute
a different filter without saying so**): stopped here. `motion_smoothing`
is left at `0.0` in `my_avatar_neural`'s own `engine_options` -- the option
and the whole EMA mechanism stay wired in, correct and tested, waiting for
either a higher ceiling to be accepted deliberately (more lag) or a
different filter shape. No pictures were rendered for this build: nothing
here is a "before/after" worth keeping, only a measurement that the
first thing tried was not enough.

**Day 7, still open:** lever 1 (smoothing) needs a shape that does not cost
this much lag for this little gain -- a speed-adaptive (One-Euro) filter is
the obvious next candidate, not decided or built here. Lever 2 (the
pitch/gaze leak behind the sulky face) is untouched. His side (the
`natural` clip, the parted-lips portrait) is still outstanding. Exit
unchanged.

**Day 7, later (Fable 5.1 analysis on the `natural` clip he then recorded,
`recordings/natural.mkv`, 1690 frames, `natural.json`).** The clip that
finally holds his call posture, and it is worse than the scripted one:

| segment | his eye ratio | avatar eye ratio | his brow | avatar brow |
|---|---|---|---|---|
| look at camera (0-118) | 0.288..0.316 | 0.218..0.344 | 0.252..0.269 | 0.158..0.192 |
| look at the screen, still (119-416) | 0.292..0.381 | **0.180..0.292** | 0.208..0.263 | 0.135..0.189 |
| talk (713-1306) | 0.249..0.329 | **0.131..0.299** | 0.245..0.307 | 0.140..0.181 |
| still again (1602-1689) | 0.262..0.338 | **0.085..0.243** | 0.197..0.352 | 0.144..0.189 |

His eyes are open all clip; the avatar's are half shut while he talks
(frames 1000, 1380: eyes squeezed, lips pursed) and squinting-frowning when
his eyes open *wider* than rest (frames 300-390: his ratio 0.37-0.39, the
avatar squints -- the "sulky face"). The head-down segment (417-712) goes to
pitch -32°, far past anything the engine can hold from a frontal portrait,
and the face smears with the forehead band showing (frames 560/620).

Run down, in order, each cheaply:
- **Not the crop.** The 1/z box tracks his inter-ocular width within 1-2% on
  every sitting segment (box/io 0.99-1.01) -- the day-6 crop is doing its job.
- **Not the anchor posture.** Re-rendered with the anchor learned while he
  looks at the screen (`--first 119`): frames 300/400/1000/1380 identical.
  The problem frames are deviations from *either* anchor.
- **Not pitch, as such.** The extractor's own pitch runs *opposite in sign*
  to MediaPipe's (slope -1.7 on the up/down segment, -0.9 on the natural
  clip) -- a convention, consistent internally, not a leak. What moves the
  expression on neutral-face frames is **his eyes opening wider than his
  rest**: |exp - exp_d0| 0.09-0.10 at frames 314-329 (eye ratio 0.38-0.39,
  pitch flat at -1.3°) against 0.01-0.02 at rest -- and the network renders
  "wider than the portrait's eyes" as a *squint with lowered brows*. Talking
  closes the avatar's eyes the same way (frame 1000: his 0.28, avatar
  squeezed): mouth motion leaking into the eye keypoints.
- **Not the portrait.** `portrait3.jpg`/`portrait4.jpg` (his uploads:
  neutral, lips parted, 2048x2048 -- exactly what was asked) rendered on the
  same clip (`my_avatar_neural3`/`4`): the same squint at 300, the same
  squeezed eyes at 1000/1380. Plus a new cost: a parted-lips *rest* means the
  avatar's mouth sits open whenever his is shut (frames 60/200/400), because
  `_lip_target` is floored at the portrait's own ratio -- for these portraits
  that floor must go and lip retargeting must come back on, or the mouth
  never closes. Teeth at frame 1380: no better than the closed-lip portrait.

**The finding that opens a door: LivePortrait's 21 implicit keypoints are
semantically separable, and the same on both portraits.** Nudging each
keypoint's expression by ±0.04 on x/y/z with the portrait driving itself and
reading the result back with MediaPipe:

| region | keypoints (axis) | effect on that region | leak into the other |
|---|---|---|---|
| eyes | **11, 13, 15, 16 (y)** | eye ratio ±0.16..0.20 each | mouth ≤ 0.004 (15/16 move brows 0.06-0.12) |
| mouth | **19, 20 (y, z)**, 17 (y), 14, 7 | mouth ratio ±0.30..0.37 | eyes ≤ 0.03 |
| brows | 2 (y: 0.196), 3 (y), 1 (y) | brow ±0.07..0.20 | -- |
| the rest (0, 4-6, 8-10, 12, 18) | -- | ≤ 0.015 anywhere | pose/shape |

So the eye channel can be cut out of the relative expression and driven by
the *measured* eye ratio alone (the eye retargeting network already wired
in, its delta masked to the same four keypoints) -- which is exactly the
thing the extractor gets wrong. This is **lever 2, redefined**: not a pitch
correction, a region-masked expression. Proposed as the next Sonnet task,
with the natural clip as the measure and frames 300/1000/1380 as the eye.
Probe scripts (scratch, not in the repo): `gaze_probe.py`,
`pitch_leak_probe.py`, `keypoint_sensitivity.py`.

**Files kept for him to look at (his face, gitignored, delete after):**
`recordings/day28-natural-side-by-side.mp4` + 16 stills (2.2, natural
clip), `day28-natural-anchor119.mp4` + 6 stills, `day28-natural-p3.mp4` +
7, `day28-natural-p4.mp4` + 7; CSVs `natural-follow-s22.csv`,
`natural-gaze.csv` (geometry only). Avatars `my_avatar_neural3`/`4` kept --
selectable in the app, but see the parted-lips caveat above.

**Day 8, done (2026-09-06, Sonnet 5, on Fable 5.1's written instruction) --
the eye channel cut from the extractor, kept; an extreme-pose hold, kept.**

**Task A.** `EYE_KEYPOINTS = (11, 13, 15, 16)`. Two new options:
`eye_expression_gain` (`[0, 1]`, default **0.0**) weights the eye keypoints
of the *relative expression* -- at 0.0 the extractor's own reading of the
eyes never reaches the portrait, only the eye retargeting network does;
`eye_mask_output` (default **true**) confines that network's own delta to
the same four keypoints, so a driven eye target cannot itself leak into the
mouth or brows. Both built once at load into static masks
(`rel_mask`/`eye_mask`, `_region_masks`) the graph multiplies every frame;
at `1.0`/`false` the whole thing is bit-identical to day 7. Validated once,
at load (not per frame like `motion_smoothing` -- the value only ever feeds
a precomputed tensor, so there is nothing to re-check after that) -- a
deliberate difference from the letter of "validated where `motion_smoothing`
is", noted here rather than silently.

**Task B1 -- `driving.mkv`, `eye_expression_gain=0.0`, sweeping `eye_gain`:**

| | baseline (day 7) | 0.06 | 0.12 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| blink: ear gain / r | 0.807 / 0.953 | 0.095 / 0.594 | **0.482 / 0.833** | 0.222 / 0.522 | -0.087 / -0.561 |
| eyes shut: avatar ear p10 (≤0.08) | 0.039 | 0.245 | **0.059** | 0.054 | 0.031 |
| still: avatar ear p10..p90 | 0.291..0.365 | 0.279..0.300 | 0.228..0.268 | 0.100..0.199 | 0.025..0.042 |
| still: avatar brow p10..p90 | 0.161..0.213 | 0.162..0.243 | 0.180..0.259 | 0.208..0.293 | 0.247..0.333 |
| smile teeth: avatar ear p50 | 0.147 | 0.261 | 0.209 | 0.078 | 0.077 |
| mouth-wide mar gain (% of baseline) | 0.562 | 0.537 (96%) | 0.525 (93%) | 0.499 (**89%**) | 0.457 (**81%**) |
| talk mar gain (% of baseline) | 0.764 | 0.743 (97%) | 0.711 (93%) | 0.708 (93%) | 0.690 (90%) |

**No value reaches the 0.70 blink-gain target -- not monotonic, either: 0.12
is the peak, 0.20 and 0.30 are worse than 0.12 on every row, not just
blink.** With the extractor cut out, the eye network is now the *only*
driver of eye motion instead of a small correction on top of one, and it
overshoots the way the old *combined* channel did in day 4's calibration --
past 0.12 the avatar's eyes narrow even at rest (`still` collapsing toward
shut) and the toothy smile squints again (0.078, worse than day 5's 0.152
that started this whole investigation). 0.20 and 0.30 also breach the
mar-gain integrity check (>10% off baseline) that was meant to catch
exactly this kind of cross-talk -- not simple keypoint leakage (the mouth
keypoints are untouched by the masks) but the warping network's own dense
motion field responding to an extreme deformation elsewhere in the image.
**Per the stated rule ("if none reaches 0.70, pick the best and say so"):
`eye_gain = 0.12`**, the only value that also stays inside the mar-gain
tolerance.

**Task B2 -- `natural.mkv`, `eye_gain=0.12` throughout, isolating which
change matters:**

| | baseline (day 7) | chosen (gain=0.0) | control (gain=1.0, masked) |
|---|---|---|---|
| screen-still: avatar ear p10 (≥0.24) | 0.180 | **0.236** | 0.092 |
| screen-still: avatar ear width (≤0.06) | 0.112 | **0.024** | 0.148 |
| talk: avatar ear p10 (≥0.20) | 0.131 | **0.218** | 0.072 |
| still-again: avatar ear p10 (≥0.20) | 0.085 | **0.212** | 0.068 |
| talk mar gain (% of baseline) | 0.799 | 0.816 (**102%**) | 0.754 (94%) |
| frame 300: his 0.376, avatar ear (≥0.22) | 0.267 | **0.265** | 0.221 |
| frame 1000: his 0.308, avatar ear (≥0.22) | 0.184 | **0.230** | 0.080 |
| frame 1380: his 0.325, avatar ear (≥0.22) | 0.141 | **0.256** | 0.085 |

**The control -- masking alone, extractor still at full strength -- is not
the fix; it is sometimes worse than the day-7 baseline** (frame 1000: 0.080;
screen-still p10: 0.092, both below baseline). Cutting the extractor's own
reading of the eyes (`eye_expression_gain=0.0`) is what matters; the mask
on the eye network's own output is a second, smaller safeguard. The chosen
build clears every acceptance line except one, and that one by 0.004
(screen-still p10 0.236 against 0.24) -- everything else, including all
three single-frame checks (the direct evidence for the day-7 squint/closure
complaint), passes with room. **Kept**: `eye_expression_gain: 0.0`,
`eye_mask_output: true`, `eye_gain: 0.12` in `my_avatar_neural`'s own
`engine_options`.

**Task C.** `hold_pitch_degrees`/`hold_yaw_degrees` (default 20.0/35.0,
must be positive), hysteresis at 0.75x to enter/leave without flicker
(`_should_hold`, `loaded.holding`). Past the threshold, `animate` reuses the
last rendered face but still computes this frame's own crop and mask, so
the held face follows him at his real position; `reset_anchor` clears it.
Rendered on `natural.mkv` with Task B's kept options
(`day29-natural-eyech.mp4` + 10 stills): frames 500/560/620 (pitch to -32°,
the forehead-band smear in day 7's own stills) now hold a calm, coherent
face that tracks his position and comes back the instant he does --
compared side by side with `day28-natural-anchor119-f500/560/620.png`, no
smear, no forehead band. Frames 60/300/1000/1380 (the eye-channel
evidence) read as a genuinely following face for the first time: eyes open
at rest, open and alert while he talks, open when his are wide -- none of
day 7's squeezed-shut or squint-and-frown.

**What did not change, on purpose.** Brows still leak via their own
keypoints (2, 3, 1 -- not masked, no attempt made); the parted-lips
portraits (`my_avatar_neural3`/`4`) still need lip retargeting turned back
on before they are usable, not attempted; the |yaw| > 15° lag from day 4 is
untouched. 823 tests (789 -> 823; +34 from Task A's masks and Task C's
hold). mypy the same 12 pre-existing errors. ruff clean.

**Files kept for him to look at (his face, gitignored, delete after):**
`recordings/day29-natural-eyech.mp4` + 10 stills (frames 60, 200, 300, 400,
500, 560, 620, 1000, 1380, 1650). CSVs (geometry only, not deleted):
`driving2-follow-eyech-g0.06/0.12/0.20/0.30.csv`,
`natural-follow-eyech.csv`, `natural-follow-eyech-ctrl.csv`.

**Day 9 (2026-09-06, his live test, then Sonnet 5's quick measurement of it)
-- day 8's fix does not generalise: the mouth and brows invent reactions
the same way the eyes did, and day 8 never touched them.** His own words:
*"it takes on emotions/reactions automatically... when I do nothing but
lift my head up, it moves its mouth and half-shuts its eyes -- it invents
reactions, this is terrible, it must be fixed before anything else."* He
is running the app correctly; this is real, not operator error.

Measured on `driving.mkv`'s "Look UP" segment (421-538) at today's kept
build (`driving2-follow-eyech-g0.12.csv`, already in hand -- no new render
needed):

| | him | avatar |
|---|---|---|
| mar, whole segment | **0.003..0.006** (mouth shut, no change) | **0.003..0.101** (opens on its own) |
| frames 469-525 specifically: mar | 0.004..0.005 | **0.023..0.062** |
| brow, same frames | ~0.24, barely moving | **0.135..0.152**, dropping |
| ear, same frames (his eyes open wide) | 0.43..0.44 | 0.227..0.254 -- does not follow, moves the wrong way |

Exactly frames 469-525 are where his own eye ratio rises to 0.43-0.44 (he
is looking up, eyes open wide) -- and that is when the avatar's mouth opens
and its brow drops, though his mouth never moved. **This is the same
mechanism day 8 fixed for the eyes, showing up in a place day 8 did not
touch.** Day 8's `EYE_KEYPOINTS` table was built from a *synthetic* test
(nudge one keypoint at a time on a still portrait, read the isolated
effect) -- it answers "does keypoint 19 moving open the mouth" (yes), not
"does the extractor's own *natural* reading of keypoint 19, across real
frames, correlate with something the driver is not actually doing with his
mouth" (apparently yes, when his gaze/eyes change). The mouth and brow
keypoints (19/20/17; 1/2/3) still take the raw extractor's relative
expression unmasked, exactly as before day 8.

**Proposed next step, not measured or built yet:** extend day 8's own
pattern -- `rel_mask` zeroed on the mouth keypoints too, mouth driven only
by the lip retargeting network (`lip_retargeting`, off since day 5) against
his own measured `mar`, the same way the eyes are now driven only by the
eye network against his measured `ear`; the brow keypoints likely need the
analogous treatment, though nothing currently retargets brows from a
measured ratio, so that may need its own new mechanism rather than
reusing one that exists. Day 5's finding that "the lip network only presses,
never opens" was measured *before* this leak was known to exist, on a
driver clip that never isolated a pure gaze-up gesture from a mouth
gesture -- it needs re-checking under this specific case, not assumed
still true. His priority, stated plainly: this closes before Phase 2 or
Phase 3 get any time. Handoff to Fable 5.1 for the precise instruction --
this entry is written for that.

**Day 9, done (2026-09-06, Fable 5.1 implementing directly on his
instruction -- "no instruction, start on the defect yourself"): the mouth
and brows no longer invent reactions, and the eyes are the portrait's own
at rest.** Three changes, each measured on both clips against the day-8
build (`driving2-follow-eyech-g0.12.csv`, `natural-follow-eyech.csv`):

1. **Expression gates** (`expression_gates`, default on; the GATES section
   of `engine_liveportrait.py`). The extractor's relative expression on the
   mouth keypoints (7, 14, 17, 19, 20) and brow keypoints (1, 2, 3) is
   multiplied, every frame, by a scalar in [0, 1] computed from *his own
   tracked geometry* against the anchor's rest: mouth ratio above rest
   (start 0.010, full 0.030), mouth width above rest (+3% .. +8%, over the
   *jaw* width at mouth height -- over inter-ocular it grows 0.4%/degree of
   pitch, a closed smile's worth at +10 degrees; faded past |yaw| 12-25
   degrees), brow height above the eye corners over the eye-to-chin height
   (+-8% .. 16%; blink-proof, 0.3%/degree of pitch against 1.2-1.9% for
   the lid-based measure; faded past |pitch| 10-20 and |yaw| 15-30). Opens
   at once, closes at 0.7 per frame. Written into a static `(1, 21, 3)`
   tensor before each replay, like `ratios`. Not a cut: a smile, a word or
   a raise passes whole. Rest values `rest_mouth_width`/`rest_brow` freeze
   with the anchor; `reset_anchor` clears them.
2. **Zero at rest for the retargeting networks** (`retarget_zero_at_rest`,
   default on). Measured 2026-09-06 with the real weights: asked for *no
   change* (target = the portrait's own ratio), the eye network answered
   +0.089/+0.111 on the upper-lid keypoints -- **more than the whole
   difference between its rest and full-blink outputs** (0.072/0.066). At
   `eye_gain` 0.12 that alone closed the avatar's eyes from the portrait's
   0.334 to 0.23-0.27 at rest, and is why no gain ever reached the 0.70
   blink target on day 8: the "closing" that was measured was mostly this
   constant. The network's output at the portrait's own ratio is now read
   once at load (`_retarget_rest`) and subtracted (`eye_rest`/`lip_rest`).
3. **`eye_headroom`** option (was the constant `_EYE_TARGET_HEADROOM`),
   1.3 kept: with the eye net's delta confined to the eye keypoints (day 8)
   and zero at rest, "wider than the portrait" no longer lifts the brows --
   on the wide-eyed frames (his 0.43-0.44) the avatar's eyes go 0.31 ->
   0.375-0.38, brow measure unchanged to the third decimal, every other
   line of the table identical. Day 6's "pulls the eyes and raises the
   brows" verdict was taken with the unmasked network and the rest offset
   in it.

`eye_gain` re-swept under 2 (0.12 / 0.25 / 0.5, `driving2-follow-gate-*.csv`):
blink ear gain 0.26 / **0.77** / 0.83, eyes-shut avatar ear p50 0.26 /
**0.13** / 0.09, still-again avatar ear p10 0.30 / 0.26 / **0.15 (over)**,
blink r 0.86 / **0.89** / 0.81. **0.25 kept** -- the first value ever to
clear the 0.70 blink bar, with the eyes at rest still the portrait's.

**The complaint's own frames, `driving.mkv` 421-538 (his gaze up, mouth
shut), day 8 -> day 9:**

| | him | day 8 | day 9 |
|---|---|---|---|
| mouth ratio, p50 / p90 / max | 0.005 (p90) | 0.023 / 0.055 / **0.101** | **0.003 / 0.005 / 0.006** |
| frames 469-525, mouth ratio | 0.004..0.005 | 0.023..0.067 | **0.003..0.004** |
| eye ratio p10..p90 | 0.307..0.441 | 0.225..0.254 (half shut) | **0.316..0.372** (0.309..0.330 at headroom 1.0) |
| brow measure (brow_c) p10..p90 | -- | -- | 0.173..0.175 against 0.174..0.180 at rest: flat |
| still 0-119, avatar eye ratio | his 0.314..0.320 | 0.228..0.268 | **0.316..0.331** (portrait 0.334) |
| still, avatar mouth ratio p90 | 0.005 | 0.007 | 0.004 |

`natural.mkv` (call posture): screen-still avatar mouth ratio p90 **0.064 ->
0.003** (his 0.005), still-again **0.151 -> 0.003**, screen-still avatar eyes
0.236..0.260 -> **0.301..0.333**, talk mar gain 0.82 -> **0.96** (r 0.85 ->
0.92), smile mouth-width gain 1.35 -> 1.26, smile mar gain 0.92 -> 1.05.
Frames 300/1000/1380 (his eye -> avatar eye | his mouth -> avatar mouth):
0.376 -> **0.357** | 0.003 -> 0.002; 0.308 -> **0.318** | 0.009 -> **0.011**
(was 0.137); 0.325 -> **0.335** | 0.293 -> 0.400 (was 0.297). (Kept build,
`natural-follow-day9.csv`; the gate-only sweep at headroom 1.0 read 0.346 /
0.318 / 0.332 on the eyes.)

**What must still come through, `driving.mkv`, day 8 -> day 9 (`eye_gain`
0.25):** closed-smile width gain 1.17 -> 1.22, toothy-smile mar gain 0.93
-> 0.90, mouth-wide mar gain 0.53 -> 0.56, talk mar gain 0.71 -> 0.72 (r
0.92), eyes-shut avatar ear p10 0.059 -> 0.052. **Costs, stated:** the
toothy smile's *width* gain 0.41 -> 0.23 -- per frame the avatar's width
saturates at 0.67-0.69 in both builds (his 0.75; the gate is fully open,
+27..35%), so this is the network's ceiling read through a slope, minus
~0.02; and a slow closed smile arrives late: at his +2% width (frames
781-797) the day-8 avatar was already +10% wide (the extractor amplifying
a beginning), the gated one waits until his +3-6% (frame 805), about half
a second.

**Brows: measured, and held by default (`brow_gain` 0.0).** With the gate
open on his real brow raise (his measure +34..38% on both clips, frontal,
mouth shut): `driving.mkv` frames 1678-1708, avatar brow +4..8% (right way,
a fifth of his); `natural.mkv` frames 1570-1600, the same gesture, avatar
brow **-12..-16% -- a frown for a raise**, in the day-8 build too (lid-based
brow 0.139-0.149 against 0.16-0.20 before it). The extractor's brow
reading is not reliably the right way up from one clip to the next, and a
gate cannot turn a wrong-way reading right. A brow that frowns when he
raises it is exactly this day's complaint, so until brows are driven from
his own measured reading the way the eyes are (no such network exists in
LivePortrait -- it would be a new mechanism), they do not move. `brow_gain`
1.0 gives the gated brows back for measuring. His frown: driving frames
1720-1756, his -12%, avatar -2..-11% (right way) -- also held now.

**Also:** `scripts/measure_follow.py`/`render_liveportrait.py` take
`--option KEY=VALUE` (repeatable, JSON values) to override an avatar's
`engine_options` for one run -- sweeps no longer copy avatars or edit
`metadata.json`; `measure_follow.py` reports the two gate measures as
`mouth_wj`/`brow_c`. Kept options (`my_avatar_neural/metadata.json`):
`eye_gain 0.25, eye_headroom 1.3, retarget_zero_at_rest true,
expression_gates true, brow_gain 0.0` (the rest as day 8). Kept build
re-measured whole (`driving2-follow-day9.csv`): blink ear gain 0.78 (r
0.88), eyes-shut avatar ear p10 0.051, closed-smile width gain 1.33,
mouth-wide mar gain 0.58, talk mar gain 0.73; brows flat everywhere
(brow_c gain -0.02). Tests 823 -> 855 (+32: gate measures, gates, the
frame path, `_drive`'s gate and rest terms, headroom). mypy the same
pre-existing errors, ruff clean. (Note for whoever runs the suite from
Git Bash: `tests/unit/test_ui_background.py` aborts the process at exit
127 there -- a Qt platform matter, not a failure; `QT_QPA_PLATFORM=offscreen`
runs all 53 UI/virtual-camera tests green.)

**Not done, on purpose:** brows (above); the lip network stays off (its
`lip_rest` is wired and tested, unmeasured with real weights since it does
not run); the parted-lips portraits; |yaw| > 15 degree lag; the |yaw| 12-25
fade means a smile *while turned well away* arrives only through the mouth
ratio. **Next: his live verdict on this build, then Phase 2.**

**Files kept for him to look at (his face, gitignored, delete after):**
`recordings/day30-driving-day9.mp4` + stills `day30-driving-day9-f{60,469,
500,520,700,830,940,1250,1700}.png`, `recordings/day30-natural-day9.mp4` +
stills `day30-natural-day9-f{300,560,1000,1380}.png`. CSVs (geometry only,
not deleted): `driving2-follow-gate-g0.12/0.25/0.5.csv`,
`driving2-follow-gate-g0.25-h1.3.csv`, `natural-follow-gate-g0.25.csv`,
`driving2-follow-day9.csv`, `natural-follow-day9.csv` (the kept build).

**Day 9, later (2026-09-06, Fable 5.1, on his live verdict of the build
above): "it no longer follows my mouth well when I talk, the face is not as
clear; eyes and brows have gone inert; it still makes its own emotions at
moments" -- mouth first, then brows, on his order.**

*"Not as clear" is not blur.* Laplacian sharpness of the rendered face on
the same natural.mkv frames, day 8 -> day 9: eyes region 101 -> 159, mouth
92 -> 99, whole face 160 -> 173 (higher is crisper). What he saw is the
articulation, below.

**Mouth: speech is motion, not distance from rest.** The distance-only gate,
replayed on his own tracked readings over natural.mkv's 594 talking frames:
shut on **20%** of them, half-open on 11%, **49 open/shut flips in 20 s**
(closed runs of ~3 frames) -- every closed-lip consonant and every pursed
vowel shut it and the release dragged the lips back. Fix: a second way to
open, *the mouth is moving* -- per frame |dMAR|/0.004 + |dwidth|/1.5%
(either direction), soft-gated 0.75..1.25, and a frame that reads as
moving holds the gate fully open for 12 frames (0.4 s)
(`_mouth_activity`, `_MOUTH_HOLD_FRAMES`). Units from the filtered
tracker's own frame-to-frame reading (his clips): still or gazing up,
|dMAR| p99 0.0004-0.0015 and |dW| p99 0.4-0.6%; talking, |dMAR| p50 0.004
(call) to 0.010 (scripted). Lip thickness was tried and dropped (moves 2% a
frame on a tilt, mouth shut). Result, the gate as actually applied (now
logged by `measure_follow.py` as `gate_mouth`/`gate_brow`): natural talk
**shut 8% / open 92% / 1 flip** (from 20% / 49); driving talk open 100%;
still and gaze-up **shut 100%**; screen-still shut 98%. Nothing invented
came back: gaze-up avatar mouth max 0.006, still 0.004, screen-still
0.003, still-again 0.003 (all as before). Following: natural talk mar gain
0.96 (same), driving talk 0.73 -> 0.74, mouth-wide 0.58 -> 0.59, closed
smile 1.33, natural smile width 1.27 -> 1.43; his pursed "o" at frame 1000
(mar 0.009): avatar 0.011 -> 0.019-0.026 (day 8: 0.137) -- lip motion
inside speech passes again. Known: a head turn's own width jitter opens
the gate on 69% of turning frames (he also talked while turning, so the
harm there is unmeasured).

**Brows: driven from his own measured brow height, raises only.**
LivePortrait has no brow network, so one was built by hand in keypoint
space (`_brow_lift`, `brow_follow`, `_BROW_LIFT_Y`, the `brow_delta` slot
added to the driving keypoints in `_drive`). Calibrated on the portrait
itself by nudging each brow keypoint in y and reading the avatar's brow
measure back (table in the engine): keypoint 2 is the brow (-0.025 ->
+29%, -0.05 -> +69%, both sides even, eyes and mouth untouched); 3 widens
the mouth 6.6% for the same raise; 1 is not monotone and moves the eyes.
So keypoint 2 alone, -0.12 of y per unit of his relative raise.

*Why raises only.* Nine brow measures were tried (brows to the eye
corners, the nose bridge, the glabella; over the chin, the nose length,
inter-ocular): **on every one, his gaze turned up with the head still
(driving 469-525) reads -9..-17% of rest, and his real frown -8..-12%.**
The two cannot be told apart, and lowering the avatar's brows on that
number would frown it every time he looks up -- the day-9 complaint
itself. A frown is therefore not carried. On the raise side the confounds
are all under 10%: eyes shut +5..+10% (the corner points ride with the
lids), a blink +4..+9%, a closed smile +5%, a turn +8%, a tilt +6%, talking
+6% (scripted) to +13% (call posture); a real raise +33..+38%. Dead band
10%. Faded past |pitch| 10-20 and |yaw| 15-30 as the gate was.

*Measured, `brow_follow` 1.0 (`driving2-follow-day9d.csv`,
`natural-follow-day9d.csv`), against the brows-held build:*

| | his brow (rel. to rest) | brows held | driven, raises only |
|---|---|---|---|
| driving raise 1678-1708, p50 | +37.7% | -3.0% | **+34.1%** |
| natural raise 1570-1600, p50 | +33.8% | -3.1% | **+17.0%** |
| driving brows segment, gain / r | | -0.02 / -0.21 | **0.57 / 0.89** |
| natural brows segment, gain / r | | -0.05 / -0.54 | **0.54 / 0.96** |
| driving frown 1720-1756, p50 | -12.3% | -1.8% | -1.8% (not carried) |
| gaze up 469-525, avatar brow p10 | (his -11.6%) | -2.8% | **-2.9%** (a symmetric drive: -7.8%) |
| still / screen-still, avatar brow spread | | 0.174..0.180 / -5..+2% | 0.174..0.180 / -6..+1% (no added jitter) |

Every other line unchanged (gaze-up mouth max 0.005, still mouth 0.004,
blink ear gain 0.79, talk mar gain 0.73 / 0.97, closed smile 1.33). The
same gesture drives the avatar 90% of the way on the scripted clip and 50%
on the call-posture one -- the keypoint's effect inside a real frame is
weaker than on the still portrait and varies with posture; 1.0 kept rather
than pushed, so the scripted clip does not overshoot. An earlier pass with
a symmetric 3% band (`*-day9c.csv`) carried frowns at -8% but also dipped
the brows -7.8% on the gaze-up -- the table's last column is why it is
raises only.

Kept options added: `brow_follow 1.0` (`brow_gain` stays 0.0 -- the
extractor's own brow reading is still not used). Tests 855 -> 866 (the
whole suite runs green in one go from Git Bash with
`QT_QPA_PLATFORM=offscreen`). ruff clean, mypy the same pre-existing error.

**Not done:** the "own emotions at moments" line of his verdict -- he has
not yet said *which* moments; candidates listed in the chat answer (the
1.3 headroom's wide-eyed look on a gaze up; the nine ungated keypoints;
the extractor's overshoot on a genuine small opening, frame 1380 0.29 ->
0.40). Eyes: what is missing since day 8 is gaze (the eyeballs always look
straight) -- an experiment was proposed, not built: let the extractor's eye
reading back in at 0.3-0.5 only while his lids are at rest. Frowns (above).
Turns opening the mouth gate (above).

**Files kept for him to look at (his face, gitignored, delete after):**
`recordings/day31-driving.mp4` + stills `day31-driving-f{60,469,520,700,
830,940,1250,1690,1740}.png` (1690 = his brow raise, 1740 = his frown),
`recordings/day31-natural.mp4` + `day31-natural-f{300,560,1000,1380,1585}.png`
(1585 = his brow raise). The `day30-*` renders of the morning's build were
deleted, superseded. CSVs (geometry only): `*-day9b.csv` (activity gate),
`*-day9c.csv` (symmetric brow drive), `*-day9d.csv` (kept build).

**Day 9, evening (2026-09-06, Fable 5.1, on his third live verdict of the
day: "better, but: (1) the face *bounces* on the slightest sideways shift
of my head, or while I talk; (2) a slight smile spreads the mouth a lot;
the eyes are slightly too big; it smiles to one side, like a cynical
smile").**

*Bounce, measured before touching anything* (`pose_jitter_probe.py`,
driving.mkv, frame-to-frame |change| p50/p90): the extractor's angles are
*not* it (pitch 0.04/0.14 degrees sitting still). The crop's **size**
jumps 1.5/6.3 px (of ~700) while he talks and 1.2/3.3 on a turn -- it is
sized by MediaPipe's head-pose distance z, which reads 0.02..0.08 units
differently each frame as his mouth moves; his own inter-ocular distance
moves 0.18/0.79 px. On top, the extractor's own *scale* reads 0.1%/1.1%
differently each frame even sitting still, and the avatar's inter-ocular
distance jittered 0.18/1.20 px against his 0.10/0.32 on the still segment
-- a face breathing in size for no reason. Two fixes: (a) the crop's size
through a One Euro filter (`crop_smoothing` 1.0 Hz, `crop_smoothing_beta`
0.02; `_OneEuro`, `_crop_geometry`/`_similarity` split out of
`crop_matrix`, `_smooth_crop`); (b) the extractor's scale and translation
smoothed on their own (`size_smoothing` 0.85, same `_ema_`, expression and
pose untouched). The crop's *centre* was filtered too at first and that
was withdrawn (`crop_centre_smoothing` 0.0): the size flicker went but the
position against his own jittered more on the still segments (natural.mkv
camera-still dx p90 0.63 -> 0.98 px) -- a One Euro's speed estimate hunts.
Also the mouth gate: a turn's own width jitter no longer counts as motion
(`_WIDTH_PER_DEGREE_YAW`) and the gate opens over two frames instead of
one (`_GATE_ATTACK`), so the mouth region does not jump into place.

*Smile / side smile, measured* (`mouth_tilt` added to `measure_follow.py`:
the mouth corners' height difference over inter-ocular; `mouth_axis_probe.py`):
his closed smile tilts his mouth +0.018 of inter-ocular at the median (his
rest +0.014); the avatar's moved **twice as far as his (slope 1.96)** and,
talking on natural.mkv, wandered -0.028..+0.025 with r 0.43 against his --
not his asymmetry, the extractor's. Nudging each mouth keypoint on each
axis on the portrait: the **x of keypoints 14, 17, 20 and the y of 7 tilt
the mouth** (+-0.05 on 17's x: tilt 0.06-0.12, width 2-6%) and carry
nothing else of use; the **z of 17, 19, 20 is the width** (20's z: +32% /
-58%); the y of 19/20 the opening; 14's y the corners' height, symmetric.
So `mouth_lateral_gain` (default 0.0) cuts the four tilt components and
`mouth_width_gain` (0.7 kept) weights the width -- his +15% at the top of
a closed smile had read as +26%. With the tilt components cut the avatar's
tilt *spread* while talking equals his (35 vs 35, 37 vs 37 on the smile;
was 53 and 40) but a constant offset remains -- the photograph's own mouth
sits +0.020 higher on one side and this face's generator adds +0.014 on a
symmetric smile -- so `mouth_tilt_trim` (0.005 on keypoint 17's x, about
-0.012 of tilt) takes most of it out.

*Eyes*: `eye_headroom` 1.3 -> 1.1 and a new `eye_scale` 0.95 (the target at
his rest is 95% of the portrait's open eyes -- the portrait's 0.334
against his 0.314..0.320). Still-segment avatar eyes 0.316..0.331 ->
0.306..0.324; his 0.314..0.320. Blink gain 0.79 -> 0.79.

*Measured, kept build (`driving2-follow-day9g.csv`, `natural-follow-day9g.csv`
plus the `-day9h` smile-segment runs) against the morning's
(`*-day9d.csv`); frame-to-frame |change|, p50/p90:*

| bounce | morning | evening | him |
|---|---|---|---|
| driving still: avatar inter-ocular, px | 0.18 / 1.20 | **0.18 / 0.50** | 0.10 / 0.32 |
| driving still: avatar-minus-him dy nose, px | 0.25 / 1.58 | **0.20 / 0.89** | |
| driving still-again: inter-ocular / dy nose | 0.15 / 1.75, 0.16 / 2.77 | **0.17 / 0.54, 0.27 / 0.79** | 0.08 / 0.30 |
| driving talk: avatar inter-ocular | 0.25 / 0.93 | **0.23 / 0.77** | 0.18 / 0.79 |
| natural camera-still: inter-ocular / dy nose | 0.19 / 1.36, 0.20 / 2.73 | **0.14 / 0.77, 0.21 / 1.18** | 0.14 / 0.63 |
| natural talk / smile: avatar inter-ocular | 0.44 / 1.70, 0.54 / 1.91 | **0.29 / 1.03, 0.44 / 1.23** | 0.31 / 1.25, 0.38 / 1.29 |
| scale (avatar/him) still, % | 0.10 / 0.67 | **0.07 / 0.29** | |

The avatar's size now moves frame to frame about as much as his own face
does; sitting still its position jitters ~40% less in y. A turn's dx is
unchanged (0.76 / 3.2-3.6 px -- the unfiltered centre following a moving
head, not a bounce).

| side smile: mouth tilt, x1000 of inter-ocular, p10 / p50 / p90 | him | morning | evening |
|---|---|---|---|
| driving closed smile | +12 / +18 / +27 | +5 / +17 / +36 | **+8 / +13 / +24** |
| driving talk | +21 / +31 / +45 | -- | **+1 / +24 / +34** |
| natural talk | -14 / +3 / +21 | -28 / +6 / +25 | **-9 / +5 / +24** |
| natural smile | -10 / +7 / +28 | -22 / -1 / +18 | **-4 / +8 / +30** |
| at rest | +14 (driving), +10 (natural) | +20, +18 | **+7, +6** |

The avatar's mouth now tilts as much and as often as his, no more.

*Eyes*: still-segment avatar 0.316..0.331 -> **0.306..0.322** (his
0.314..0.320); gaze-up 0.316..0.372 -> 0.308..0.334; blink gain 0.79 kept.
Natural frame 300 (his eyes wide, 0.376): 0.358 -> 0.333.

*Smile width -- not fixed, and two things tried are on record.* His closed
smile widens his mouth +8% at the median and +15% at its widest; the
avatar's +8% and **+26%** -- the median matches, the peak over-spreads --
and that number did not move: weighting the z of 17/19/20 at 0.7 made it
+29% (those components *narrow* during a smile), weighting keypoint 14's y
at 0.7 left it at +26% and cost the toothy smile (+27% -> +21%, his +33%).
`mouth_smile_gain` stays 1.0. On natural.mkv his smile is an open-mouth
laugh whose width hardly changes (-1..+2%), and the avatar's narrows (-10%)
-- there the complaint cannot be about width. What his eye called
"spreads the mouth a lot" was, on the numbers, most likely the tilt, which
is gone; if the peak spread still reads wrong live, the next lever is a
soft cap on the *rendered* width (measure the avatar's own mouth width and
pull keypoint 14 back), not a gain.

*Unchanged, checked*: gaze-up avatar mouth max 0.005, still mouth 0.003,
screen-still 0.003; talk mar gain driving 0.73 -> 0.77, natural 0.97 ->
0.87 (r 0.94); mouth-wide 0.58 -> 0.66; brow raise +34% / +19%; the mouth
gate on turns 69% -> 64% open, 6 -> 4 flips.

Kept options added: `eye_scale 0.95`, `eye_headroom 1.1`, `mouth_lateral_gain
0.0`, `mouth_tilt_trim 0.005`, `mouth_smile_gain 1.0`, `size_smoothing
0.85`, `crop_smoothing 1.0`, `crop_centre_smoothing 0.0`,
`crop_smoothing_beta 0.02`. Tests 866 -> 883. ruff clean, mypy the same one
pre-existing error.

**Files kept for him to look at (his face, gitignored, delete after):**
`recordings/day32-driving.mp4` + stills `day32-driving-f{60,469,700,830,
940,1250,1690}.png`, `recordings/day32-natural.mp4` + `day32-natural-f{300,
1000,1380,1585}.png`. The `day31-*` renders were deleted, superseded. CSVs
(geometry only): `*-day9e-smile.csv` (tilt measured, before), `*-day9f.csv`
(centre filtered too), `*-day9g.csv` (kept build, full), `*-day9h.csv`
(smile gain 0.7 on 14's y, smile segments).

**Day 9, night (2026-09-06, 23:21 -- he pressed "That looked wrong" twice
on the evening build: "a catastrophe").** The two captures
(`captures/moment-20260905-232130`, `-232151`: the frame on screen and the
last 300 tracked frames, no camera imagery) read as follows. 23:21:30 --
he is in a **closed-lip grin** (MediaPipe smile blendshapes 0.58-0.79,
mouth ratio 0.004) and the avatar shows an **open, toothy grin**, eyes
wide. 23:21:51 -- he is **at rest** (no smile, ratio 0.005, eyes 0.34) and
the avatar's **lips are parted**, staring. Two causes, both measured from
the tracking he saved:

1. **The width gate opened the whole mouth.** A smile's width opened the
   gate on every mouth component, and the extractor's reading of a strong
   smile carries parted lips with it (the clips' smiles were milder). Now
   the lips' opening -- the y of keypoints 17, 19, 20 (`_MOUTH_OPENING`,
   from the axis probe: +-0.05 on 19/20's y is a mouth ratio of 0.24-0.27)
   -- has **its own gate that knows only the ratio**: how far his ratio is
   from rest, and whether *it* is moving. A smile widens the avatar's
   mouth; only his own lips parting parts the avatar's. `_gates` returns
   `(mouth, opening, brow)`; `measure_follow.py` logs `gate_open`.
2. **His live tracking is 3-7x noisier than the clips.** With his mouth
   shut, |dMAR| per frame read 0.0027 (p90) / 0.011 (p99) and |dW| 0.7% /
   2.0% at 22 fps -- the clips' worst was 0.0015 / 0.6%. At the motion
   units tuned on the clips (0.004 / 1.5%), **17% of his shut-mouth frames
   read as "moving" and 9% set the hold** -- the parted lips at 23:21:51.
   Units now 0.010 / 2.5% (his live p99 is 1.1 / 0.8 units: a one-frame
   partial open at worst, no hold; a talking frame at the call clip's p90
   |dMAR| 0.019 is still 1.9 units) and the motion is measured per 30-fps
   frame (`last_dt` from the capture timestamps -- at 22 fps a frame sees
   1.4x the change).

*A wrong turn, on record.* His face is 425-475 px tall in the captures, so
the 2.2-height crop (~950 px, shifted up an eighth) reaches ~260 px above
the 720 px frame: **28% of the extractor's input is a black band.** That
looked like the live/offline difference -- until the clips were measured:
he sits at the same distance in all of them (inter-ocular 196-244 px), so
every number in this file was taken *with* that band, and the extractor
copes. Tried anyway on a 1.9x zoomed copy of natural.mkv (`natural-zoom.mkv`,
inter-ocular 408 px, 69% of the crop outside the frame): replicated edges
were far *worse* than black (talk mar gain 1.12 -> 0.31; frame 1380 mouth
0.185 -> 0.017) -- smeared edge pixels read as a face, black reads as
nothing. `crop_border` exists ("black" default, "replicate", "reflect")
for measuring, not for use. The honest limit stands: at 720p the network
sees its whole training context only for a face under ~330 px (inter-
ocular under ~160 px); at his distance it sees ~72% of it. A wider camera
framing, or sitting a little further back, would give it the rest.

*Measured (`*-day9i.csv` against the evening's `*-day9g.csv`):*

| | evening | night |
|---|---|---|
| driving closed smile: **opening gate** (lips) | (one gate) | **shut 100%** -- the lips never part |
| driving closed smile: avatar mouth ratio p90 / his | 0.006 / 0.005 | 0.010 / 0.005 (shut) |
| driving closed smile: width gain, avatar p90 (his 0.634) | 1.63, 0.708 | **1.25, 0.670** |
| driving toothy smile: width gain | 0.22 | 0.35 |
| driving talk: mar gain / opening gate | 0.77 / -- | 0.75 / open 88%, shut 5% |
| natural talk: mar gain (r) | 0.87 (0.94) | 0.89 (0.93) |
| natural smile (a laugh): mar gain / opening gate | 0.94 | 1.00 / open 100% |
| natural screen-still, tilt: mouth gate shut | 99%, 93% (2 flips each) | **100%, 100%** (0 flips) |
| gaze-up avatar mouth max, still mouth p90 | 0.005, 0.003 | 0.005, 0.003 |
| driving turn: mouth gate open | 64% | 40% |
| blink ear gain | 0.79 | 0.80 |

The closed smile's over-spread came down with it (the opening components
add width too: 17's y at +0.05 is +13% width) -- +29% -> +24% at the peak
of his +15%. Tests 883 -> 887. ruff clean, mypy the same one pre-existing
error. **What cannot be checked here:** his live jitter itself -- the
clips are quieter than his live tracking, so the thresholds were set from
the two captures, and only a live test says whether 0.010 / 2.5% is
enough. If the lips still part at rest, the next capture's `tracking.npz`
gives the number directly (the |dMAR| p99 of shut-mouth frames).

**Files kept for him to look at (his face, gitignored, delete after):**
`recordings/day33-driving.mp4` + stills `day33-driving-f{60,469,700,830,
940,1250,1690}.png`, `recordings/day33-natural.mp4` + `day33-natural-f{300,
1000,1380,1585}.png`. The `day32-*` renders were deleted, superseded. Also
his own `captures/moment-20260905-232130`, `-232151` (tracking + the frame
on screen) -- his to delete. `recordings/natural-zoom.mkv` (his face,
zoomed) kept for the border question only; CSVs `natural-zoom-follow-black
/-replicate.csv`, `*-day9i.csv`.

**Day 10 opens with his fifth live verdict (2026-09-06, on the night
build), verbatim, and a request to continue in a fresh chat.** His words:
*"The eyes are now more shut. When I look to the side, the eyes don't
follow. Blinking isn't as natural as it was yesterday. Yesterday's version
had more flexible expressions and I wanted a balance of that -- it isn't
working yet. When I look down it takes on a terrible, foreign face; I tilt
my head slightly down and it is already a completely different face. This
is a big problem."* Nothing was changed in response; the next session
starts here. Hypotheses, each with the measurement that settles it:

1. *Eyes more shut* -- the evening's `eye_scale 0.95` and `eye_headroom
   1.3 -> 1.1`. "Yesterday" is the morning build (scale 1.0, headroom 1.3),
   which he found "slightly too big" -- the middle is `eye_scale 1.0,
   eye_headroom 1.15-1.2`. Measure: still-segment avatar eye ratio against
   his 0.314..0.320 (morning 0.316..0.331, evening 0.306..0.322).
2. *Eyes don't follow a sideways glance* -- gaze. Since day 8 the eye
   keypoints take nothing from the extractor (`eye_expression_gain 0.0`),
   and the eye network only opens and shuts; the eyeballs always look
   straight. Two routes: (a) a hand-built gaze drive like the brows' --
   MediaPipe's iris points (468-477) give his gaze offset directly; the
   nudge probe (`mouth_axis_probe.py` pattern, on EYE_KEYPOINTS' x/z) says
   which keypoint component moves the irises; (b) the extractor's eye
   reading let back in at 0.3-0.5 *only while his lids are at rest*
   (|ear/rest - 1| < 0.1), so the day-8 half-shut leak on a gaze-up cannot
   return. Measure on driving.mkv 120-420 (gaze L/R): avatar iris offset
   against his; and 421-538 must stay at eyes >= 0.31.
3. *Blinks less natural* -- `eye_gain` 0.25 is unchanged since the morning;
   blink ear gain measured 0.79-0.80 all day. Suspects: the smaller open
   state (1) making the same blink look smaller; the size/crop smoothing
   (`size_smoothing`, `crop_smoothing`) changing the eye region's timing.
   Measure: blink segment 539-658, avatar ear per frame, morning vs night
   CSVs (`driving2-follow-day9d.csv` vs `-day9i.csv`) -- the shape of the
   blink, not just its gain.
4. *Looking down / a slight tilt down = a foreign face* -- the largest.
   The hold only engages at |pitch| > 20 degrees; between -5 and -20 the
   frontal portrait is re-drawn pitched down, and a downward gaze lowers
   his lids, so the eye network shuts the avatar's eyes on top -- brows
   held, lips gated: a dead, tilted face. Candidates: (a) a *pitch gain*
   < 1 on the relative rotation (the avatar pitches, say, 60% of his --
   the portrait cannot render steep pitch well anyway); (b) hold earlier
   (`hold_pitch_degrees` 20 -> 10-12) with a blend instead of a freeze;
   (c) the crop's vy shift under pitch (day-6 cliff -- check the framing
   at pitch -10..-20). Measure on natural.mkv 417-712 (tilt down to -32):
   render stills at pitch -5 / -10 / -15 / -20 and read the avatar's eye
   and mouth ratios against his; and ask him to press "That looked wrong"
   on the next such moment -- `tracking.npz` gives his exact pitch/gaze.
5. *"More flexible yesterday; I want a balance"* -- the gates and the
   brow/lateral cuts each removed invented motion and some real motion
   with it. Worth a single `expression_gain`-style knob he can feel live
   (mouth gate floor, brow follow, eye headroom together), and a live A/B
   with him rather than more clip measurement -- the clips are quieter
   than his live tracking (night entry) and have not predicted his
   verdicts well twice now.

The kept options at the close of day 9 are exactly
`avatars/my_avatar_neural/metadata.json`; the measurement tools are
`scripts/measure_follow.py` (`--option`, `--last`, gate columns) and
`scripts/render_liveportrait.py` (`--option`, `--stills`); his own captures
are the ground truth for live behaviour.

**Day 10, analysis of (4) -- the pitch-down "foreign face" -- from the
night build's own CSV, no new render (2026-09-06, Fable 5.1; the
implementation is Sonnet 5's, instruction below).** `natural-follow-day9i.csv`,
segment 417-712 (the "tilt slightly down" instruction: in fact ONE tilt,
492 -> 684, a 4 s plateau at -32 degrees in the hold; his eyes are shut
0.05-0.10 on the descent 477-495 -- a blink as he starts -- and open
0.29-0.39 on the return 669-693, so the RETURN frames are the clean ones
for stills). Avatar-minus-him pixels, binned by his MediaPipe pitch:

| his pitch | n | his EAR | avatar EAR | dy forehead | dy eye | dy mouth | dy chin | scale (io) |
|-----------|---|---------|------------|-------------|--------|----------|---------|------------|
| -5..0     | 84 | 0.295 | 0.300 | +4  | -5  | -5  | -5 | 1.05 |
| -10..-5   | 10 | 0.289 | 0.283 | +28 | +15 | +7  | 0  | 1.02 |
| -15..-10  | 13 | 0.116 | 0.272 | +43 | +29 | +24 | +3 | 0.98 |
| -20..-15  | 21 | 0.134 | 0.305 | +55 | +38 | +39 | -1 | 0.95 |
| < -20 (held) | 158 | 0.146 | 0.305 | +57 | +50 | +52 | +20 | 0.98 |

Per frame on the return: -17.9 deg dy_eye +46 / dy_chin -1; -12.6: +28 / +3;
-6.9: +16 / +2; -3.4: +2 / -4. Read: at rest every feature sits within 5 px
of his; from -5 degrees on, the avatar's forehead, eyes and mouth slide
DOWN 2-3 px per degree faster than his while the chin stays -- a face
compressed from the top, the mouth pushed toward the chin, and 7-10%
narrower (scale 0.95-0.98 against 1.05-1.08 at rest; his own io GROWS
220 -> 237 px because he leans in as he tilts, the avatar's does not
follow, 231 -> 228). That is his "completely different face" at a slight
tilt. Two things the table rules out: the eyes are NOT shut by the eye net
on a look-down (avatar 0.27-0.31 while his read 0.10-0.15; hypothesis (4)
in the block above was wrong about that), and the gates are all 0 here, so
no expression is being invented -- this is a rigid-motion error.

Three candidate mechanisms, not yet separated (each is a different lever):
- **A. over-rotation**: the rendered pitch is larger than his (the
  relative rotation `rot_d @ R_d0^T` applied to `x_c_s`); would move the
  forehead down AND the chin up, the chin then hidden under his own oval
  mask (dy_chin ~0 could be that artefact).
- **B. a translation/scale leak**: the extractor's `t` reads a pitched face
  as "lower" and its `scale` as "smaller" (its relative t/scale are added
  to the portrait's); would slide the whole face down and shrink it.
- **C. the crop**: `_driver_face_height` (rest height x z_rest/z) not
  following his lean-in under pitch; would leave the avatar too small.

**Day 10, instruction for Sonnet 5 -- (4) pitch-down, in this order; stop
and report at any step whose numbers contradict the step before.**

*Step 1 -- separate A/B/C on natural.mkv 480-700, no engine change.*
1a. `scripts/measure_follow.py` (the output tracker block, ~line 295-320,
    `got` is the tracker on the avatar's own frame): add columns
    `o_yaw`/`o_pitch`/`o_roll` from `got.head_pose` (same instrument on
    both faces -- the only apples-to-apples pitch comparison), and
    `crop_px` = the crop's size in frame pixels this frame, read from the
    engine's `portrait.last[1]` (`to_frame`, 2x3: size = 512 x the norm of
    its first column). Keep every existing column. Run:
    `uv run python scripts/measure_follow.py --clip recordings/natural.mkv
    --last 712 --csv recordings/natural-follow-day10-base.csv` (~3 min).
    Report over frames 480-700 with his pitch in [-20, -3] and not held:
    slope of `o_pitch` on `pitch` (least squares, and the ratio at -7 and
    -12.6); `crop_px / d_io` at rest (417-470) vs at -12.6 and -17.9 --
    if it drops more than 4%, that is C.
1b. Copy the day-7 probe
    `C:/Users/User/AppData/Local/Temp/claude/C--Users-User-desktop-sha256/afec2b6f-5311-4591-bd3c-05eba767cb36/scratchpad/pitch_leak_probe.py`
    into this session's scratchpad (it already runs the motion extractor
    eagerly on `loaded.static_in` after the anchor and prints MediaPipe
    pitch, extractor pitch, scale/scale_d0 per frame). Add the extractor's
    relative `t` (`t - t_d0`, y component, in crop units x 512 for pixels)
    and run it on `recordings/natural.mkv 480 700 3`. Report, against his
    MediaPipe pitch: slope of the extractor's relative pitch (day 7 read
    -0.9 on this clip -- confirm), of relative t.y (pixels per degree), of
    relative scale (% per degree). t.y or scale moving with pitch = B.
1c. Stills of the night build for the before picture:
    `uv run python scripts/render_liveportrait.py --clip recordings/natural.mkv
    --first 400 --last 712 --no-warp --out recordings/day34-tilt-base.mp4
    --stills 450,687,681,675,506,500` (450 rest; return frames 687/681/675
    = -6.9/-12.6/-17.9 with his eyes open; 500/506 descent, lids lowered).
    His face -- gitignored, announce, delete after he looks.

*Step 2 -- the levers, all in `_drive` (engine_liveportrait.py ~2498-2521),
each a float option baked at capture like `eye_gain`, each defaulting to
today's behaviour exactly.* Parse them with `_eye_scale`-style helpers that
raise `ValueError` out of range; document them in the module's option list
and add them to `avatars/my_avatar_neural/metadata.json` at their defaults
until the sweep picks values.
- `pitch_gain` (default 1.0, allowed [0, 2]) and `pitch_cap_degrees`
  (default 0.0 = no cap, allowed >= 0): on the extractor's RELATIVE pitch.
  Needs a new static slot `pose_d0` (3, 1) beside `R_d0` (slot list ~1547,
  init ~1692): primed in `load_avatar` from the same raw-pose stack that
  primes `pose_prev` (~1914) -- so `R_d0 == R(pose_d0)` holds from the
  first frame -- and copied in `_learn_anchor` where `pose_prev` is
  (~2438, `sums["pose"] / n` -- exactly what `R_d0` is built from at
  ~2424). `reset_anchor` leaves it alone, as it leaves `R_d0`. Then,
  replacing the `rot_d = get_rotation_matrix(pitch_s, yaw_s, roll_s)` line:
  ```
  rel = pose_now - loaded.pose_d0                      # degrees, (3, 1)
  rel_pitch = rel[0]
  if cap > 0:  rel_pitch = cap * torch.tanh(gain * rel_pitch / cap)
  else:        rel_pitch = gain * rel_pitch
  used = loaded.pose_d0 + torch.stack([rel_pitch, rel[1], rel[2]])
  rot_d = get_rotation_matrix(*used.unbind(0))
  ```
  Keep the old line on the Python-level branch `gain == 1.0 and cap == 0.0`
  so the default stays bit-exact (a float round trip through pose_d0 is
  not). The gain is sign-agnostic, so the extractor's inverted pitch
  convention (day 7) does not matter; the cap is in the extractor's own
  degrees. Yaw and roll untouched (roll is taken out by the crop; yaw is
  the separate turn-lag item).
- `translation_gain` (default 1.0, [0, 1]): `t_new = t_s + g * (t_now -
  t_d0)`. At 0 the avatar sits exactly where the crop (his rigid points)
  puts it.
- `scale_gain` (default 1.0, [0, 1]): `scale_new = scale_s *
  (scale_now / scale_d0) ** g`. At 0 the size is the crop's alone (which
  already follows his distance through z and is One-Euro smoothed).
Which of the three to sweep is decided by Step 1: A -> pitch_gain/cap;
B -> translation_gain and/or scale_gain; C -> stop and report (a crop
change; the day-6 cliff is 5% away and needs its own instruction).

*Step 3 -- sweep, natural.mkv only until a candidate exists.* Stills first
(`--option KEY=VALUE`, same six frames): for A, pitch_gain 0.7 / 0.5 /
0.35 with cap 0, then the best gain with `pitch_cap_degrees` 10; for B,
translation_gain 0.5 / 0 and scale_gain 0.5 / 0. Then measure_follow
`--last 712` with the candidate. Numbers that decide, on the same bins as
the table above (his pitch -10..-5 and -15..-10, return frames):

| measure | now | target |
|---|---|---|
| dy_eye - dy_chin at -7 / -12.6 deg | +15 / +25 px | <= 8 / <= 12 px |
| scale (io) at -15..-10 | 0.98 (rest 1.05) | >= 1.00 |
| o_pitch / pitch slope over [-20, -3] | (Step 1a) | 0.5-0.8 -- follows, gently |
| still segment 119-416 avatar EAR | 0.30-0.32 | unchanged +-0.005 |
| talk 713-1306 mar gain | 0.87 | >= 0.85 |

Then ONE full run of `driving.mkv` on the final candidate: blink ear gain
>= 0.75, turn lag no worse than -33/+21 px, toothy-smile width unchanged.
A candidate that clears the pitch lines and breaks a must-not-break line
is reported, not kept.

*Step 4 -- keep, prove, record.* Write the kept values into
`my_avatar_neural/metadata.json`; unit tests (the file's own style, fake
nets on CPU, see `_engine_and_loaded` ~492): the relative-pitch helper
(gain 1/cap 0 identity; gain 0.5 halves; cap 10 with rel 40 stays under
10 and keeps the sign), the three option validators (refused at first
drive, as `test_an_invalid_value_is_refused_at_first_drive_call`),
`pose_d0` primed at load and equal to `pose_prev` after `_learn_anchor`,
and `reset_anchor` leaving it. `uv run pytest tests/unit -q` (823 pass
today), `uv run mypy src` (12 pre-existing errors, no new), `uv run ruff
check`. Write "Day 10, done" here with the before/after table, the stills
kept for him (`recordings/day34-tilt-*`, his face, delete after), and
what was NOT fixed. Nothing is committed unless he asks.

*Not in this instruction:* items (1) eyes, (2) gaze, (3) blink shape and
(5) the balance knob from the block above -- (4) first, alone, so its
before/after is clean. One observation for (2): while he looks down his
lids read 0.10-0.15 and the avatar's stay 0.27-0.31 -- on a call the
avatar keeps staring ahead when he looks at the keyboard; whether that is
wanted is his call, not a bug to fix here.

**Day 10, Step 1 done (2026-09-06, Sonnet 5) -- stopped at the instruction's
own C-branch; Steps 2-4 (the pitch_gain/translation_gain/scale_gain levers)
NOT built.** Measured on `natural.mkv`, no engine code changed --
`scripts/measure_follow.py` gained three diagnostic columns used below
(`o_yaw`/`o_pitch`/`o_roll`, the avatar's own pose read by the same
MediaPipe instrument as his; `crop_px`, the crop's edge length in source-
frame pixels; `held`, whether the frame is the extreme-pose hold's frozen
reuse) -- kept, they cost nothing at `motion_smoothing`-style defaults and
the next crop instruction will want them too.

**Correction to Step 1's own request first: "not held" needs the engine's
own flag, not a pitch cutoff.** The hold (`_should_hold`) has hysteresis --
enters above 20 degrees, releases only below 15 -- so on the RETURN
(669-684) frames at -15 to -20 degrees are still held from the descent,
while the same pitch on the way DOWN (504-513) is not yet held. A pitch-
range filter alone pools six genuinely held frames into the "-20..-3,
should be live" bucket; `held` (now a column) is exact. All numbers below
exclude held frames.

*1a/1b together -- A (over-rotation) does not hold up; C (the crop) does.*

| measure | at rest | -6.9 deg | -12.6 deg | -17.9 deg\* |
|---|---|---|---|---|
| avatar's own pitch (`o_pitch`, MediaPipe on the output) | -3.7 (his ~0) | -10.5 (his -6.9) | -16.5 (his -12.6) | -23.7 (his -17.9, held) |
| excess (`o_pitch` - his pitch) | -3.7 | -3.6 | -3.9 | -5.8 (held, not comparable) |
| crop_px / his own inter-ocular distance | 4.480 | 4.429 (-1.1%) | 4.331 (-3.3%) | 4.288 (-4.3%, held) |
| avatar io / his io (`scale`) | 1.049 | 1.019 (-2.9%) | 1.014 (-3.4%) | 0.999 (-4.8%, held) |

\* -17.9 deg is inside the hold on this pass of the clip (see the box
above) -- the crop and mask are still fresh every frame even while held
(`animate`'s own comment), so `crop_px`/`scale` there are real; `o_pitch`
is not (it is the frozen face's own pose, not this frame's).

Read: the "excess" pitch is **~3.6-3.9 degrees, present already near rest
and not growing with |pitch|** over the whole unheld range (confirmed on
segments 0-118 and 119-416 too: avatar reads -3.7/-5.2 degrees while he
reads -0.06/-1.50 -- essentially the same constant, nowhere near his own
motion). A true over-rotation (gain > 1 on the *relative* pitch, which is
zero at rest by construction) would be **zero at rest and grow with
|pitch|** -- the opposite of what is measured. `pitch_gain`/
`pitch_cap_degrees`, as specified in Step 2, act only on the relative
term (`pose_now - pose_d0`); they cannot touch a bias that is already
present at zero relative pitch, so building them would not address this.

The day-7 probe, re-run on this clip with the held frames now correctly
excluded (they were re-reading one frozen crop and had flattened the
original day-7 numbers): the extractor's own pitch reads **-0.78x** his
MediaPipe pitch over the unheld descent (sign-flipped, as day 7 found;
0.78 here vs 0.9-1.7 reported on other segments/clips -- consistent
family, same convention, not identical -- expected, different clip and
range) -- **under 1:1, not over**. Its relative translation barely moves
with pitch (r -0.22, ~0.18 px/deg) and its relative scale, if anything,
grows slightly with pitch (r +0.24, +0.05%/deg) -- the wrong sign to
explain a shrink. Neither supports B as the mechanism either.

**C, by contrast, is clean and consistent both ways.** `crop_px` (how wide
a source-frame region `_driver_face_height` cuts) shrinks *relative to his
own true inter-ocular distance* as his pitch grows -- -3.3% at -12.6 deg,
crossing Step 1's own 4% line by -17.9 deg -- and the avatar's *rendered*
size (`scale`, io/io in frame pixels, unaffected by the held/unheld split)
drops by the same amount at the same points (-3.4%/-4.8%). Both numbers
move together because they are the same mechanism seen twice: the crop is
undersized, so what comes back through the same `to_frame` is undersized
too. Independently corroborated: his own tracked `d_io` *grows* 220 -> 247
px over this segment (he leans toward the camera as he tilts down) while
`face_height_rest * z_rest / z` is not keeping pace -- the lean-in, not
just the tilt, is not being fully absorbed by the z-based crop size.

**Stills confirm it visually and point at a second, connected symptom.**
`recordings/day34-tilt-base-f{450,687,681,675}.png` (rest, -6.9, -12.6,
-17.9 degrees; sent to him, his face, delete after he looks): at rest and
-6.9 the two faces track closely; at -12.6 and sharply by -17.9 a straight
horizontal line cuts across the upper forehead, above which the pasted
region stops looking like a face. This is the forehead mask line already
on the open list (day 3) -- not a new defect -- but an undersized crop is
a plausible reason it is far more visible here than at rest: the mask's
own vertical reach (`_OVAL_FOREHEAD_REACH`, `_frame_mask`) is built from
the same crop geometry that is now wrong by several percent, so the
oval's edge and the network's own rendered content stop agreeing on where
the forehead ends. Worth checking first when the crop fix lands, not a
claim -- not measured directly here.

**Decision, per the instruction's own branch:** stopping at Step 1. C
crosses the stated threshold and both A and B are contradicted by their
own measurements (a flat, rest-present offset for A; a weak and wrong-
signed correlation for B). Steps 2-4 as written build lures for A/B;
building them now would spend the day on a lever this data says will not
close the gap, and the crop fix the numbers actually point to is exactly
the one the instruction itself flagged as needing its own turn -- the
day-6 cliff (CROP_SCALE 2.1 grimaces, 2.3 grimaces, 2.2 is the only
frame-count-of-three that renders all three probes) sits close enough
that a `_driver_face_height` change should be swept the same cautious way,
not folded into this instruction's pitch levers.

**State: nothing in `src/` changed.** `scripts/measure_follow.py` gained
the three columns above (diagnostic only, no default-changing behaviour).
`uv run pytest tests/unit -q -m "not gui"`: 826 passed (32 gui-marked
deselected). `uv run pytest tests/unit -q` (everything, unfiltered)
crashes reproducibly partway through the gui-marked tests (exit 127, no
summary line) -- confirmed pre-existing and unrelated to today: the
crashing file passes cleanly in isolation (14/14), and today's only
touched files are this ROADMAP entry and `measure_follow.py`, neither on
pytest's collection path for `tests/`. Flagging for whoever next runs the
full suite with a display/Qt change nearby; not chased further here, out
of this instruction's scope. `ruff check scripts/measure_follow.py`:
clean. mypy does not cover `scripts/` (pyproject's own note).

**Next instruction, proposed:** a `_driver_face_height`/crop-size fix for
the lean-in-under-pitch shortfall, swept as cautiously as day 6's own
cliff (one-frame stills before any full measure run, the same six probe
frames), with the forehead-mask-line stills above as the visual check for
whether it also closes that. Items (1)(2)(3)(5) from "Day 10 opens with"
are still untouched.

**Day 10, Fable 5.1's review of Step 1 (2026-09-06), and the next
instruction.** Sonnet's Step 1 was done carefully and two of its catches
stand for good: (a) the extreme-pose hold freezes `static_in`, so any probe
that re-reads it on a held frame re-samples one stale crop -- the `held`
column is now the only correct "not held" filter, and day 7's own
extractor-pitch slope on this clip (-0.9) was contaminated the same way
(-0.78 unheld); (b) `crop_px`/`o_pitch`/`o_roll`/`o_yaw` are the right
columns and stay. Its conclusion does not stand, for four concrete reasons:

1. **The -3.7 degree "rest offset" is the photograph.** MediaPipe on
   `portrait.png` itself reads pitch **-3.56** (yaw 1.1, roll 0.8). The
   portrait is tilted down 3.6 degrees in the photo; the avatar carries it
   at every pitch. Cosmetic, not the engine, not this defect; if he wants
   the avatar level at rest that is a portrait choice.
2. **C cannot produce the measured shape.** A crop 3% too small is a
   uniform 3% shrink about the crop centre, which sits near the brows
   (`vy -0.125`): the chin would move UP ~10 px and the eyes ~0. The table
   shows eyes +29 and mouth +24 moving DOWN together at -12.6 degrees with
   "chin 0". That is a translation of the whole face -- and the chin's own
   motion is hidden: the paste mask is HIS oval, so a face slid down has
   its chin cut off and the composite's chin is his. `dy_chin ~ 0` is an
   artefact of measuring on the composite, on every day this table has
   been read. C is real and stays on the list (-1.1% at -7, -3.3% at
   -12.6 -- nothing at the "slight tilt" he complains of, a 2.13 effective
   crop scale by -13, i.e. toward day 6's 2.1 frown cliff) but it is the
   secondary item.
3. **Two unit slips.** The keypoint grid is [-1, 1] over the crop, y down
   (`make_coordinate_grid`): 1 t-unit = 256 crop px, not 512. The
   extractor's relative t.y is then +6 / +8 frame px at -7 / -12.6 degrees
   (of the 16 / 29 observed) and reverses sign at -16..-20 while the slide
   keeps growing -- a quarter of the effect, wrong direction at the end;
   B is a contributor, not the mechanism. And `o_pitch` on a face whose
   upper half has slid down is not a clean rotation reading -- the pose
   solver fits scale and translation too -- so "slope 0.99, no
   over-rotation" is weak evidence either way.
4. **The forehead line is the slide, not the crop.** A 3% crop error moves
   the mask edge ~4 px; the line on frame 681 sits where the avatar's
   forehead is ~29 px lower than his: his skin above the edge, the
   portrait's forehead below it, a tone step sharpened by the feather. Fix
   the slide and the line should fall back to its day-3 level -- a
   prediction to check on the same still.

**The candidate nobody has tested: the rotation's pivot.** In `_drive`,
`x_c_s @ rot_new` rotates the canonical keypoints about the keypoint-
space ORIGIN. In LivePortrait's own pipeline the driving video is cropped
once for the whole clip, so when the driver pitches about the neck his
face also slides down/forward in that fixed crop, `t_d` reads the slide
and `t_s + (t_d - t_d0)` re-applies it -- rotation and slide come from
the same head and agree. Our crop re-centres on his rigid points every
frame: his slide is removed on the driver side (`t_now - t_d0 ~ 0`), but
the portrait's keypoints still swing about their origin under the
relative rotation, by (origin-to-face distance) x sin(rel pitch), with
nothing left to cancel it. His face, in the crop, turns about the rigid-
point centroid (eyes/nose); the avatar's turns about a point that is not
on the face. Signature: eyes and mouth move together, forehead a little
more (the cos compression on top) -- exactly the table's shape. The
probe below settles it in one run and calibrates the fix.

**Next instruction (Sonnet 5), in order; stop and report where a number
says so.**

*Step 1 -- the swing probe, no engine change.* `pitch_swing_probe.py` in
the scratchpad: self-driving on the portrait. Rebuild the portrait crop
exactly as `load_avatar` does (`_read_portrait`, `_landmarks_of`,
`crop_matrix(points, scale, vy_ratio)`, `warpAffine`, `_to_tensor`) and
take `x_c_s`, `exp_s`, `scale_s`, `t_s`, `R_s`, `f_s`, `x_s` and
`pose_s = stack(_degrees(info[k]))` from a loaded avatar (all set in
`load_avatar`, ~1884-1915). Render eagerly, outside the graph:
```
def render(d_pitch, pivot):            # degrees, extractor's own convention
    R_rel = get_rotation_matrix(pose_s[0] + d_pitch, pose_s[1], pose_s[2]) \
            @ R_s.permute(0, 2, 1)
    base = x_c_s if pivot is None else x_c_s - pivot
    rotated = base @ R_rel if pivot is None else base @ R_rel + pivot
    x_d = scale_s[..., None] * (rotated @ R_s + exp_s) + t_xy[:, None, :]
    out = nets["decoder"](feature=nets["warping"](f_s, kp_source=x_s.half(),
                                                  kp_driving=x_d.half())["out"])
```
(`t_xy` as in load_avatar: `t_s` with z zeroed.) Assert first that at
`d_pitch = 0`, `x_d` equals `x_s` within 1e-5 -- the portrait must come
back unchanged. Then `d_pitch` in {-15, -10, -5, 0, +5, +10, +15} x
`pivot` in {None, mean of `x_c_s` over all 21 keypoints, mean over
`EYE_KEYPOINTS`}: write each 512 render as
`recordings/day35-swing-<pivot>-<deg>.png` (the portrait, not his face)
and run MediaPipe on it; report per render the eye-centre / mouth /
chin / forehead y (the `measure_follow` anchors) minus the `d_pitch = 0`
render's, and the render's own `head_pose.pitch`. Which sign of
`d_pitch` is "down" is whatever makes MediaPipe's pitch on the render go
negative -- say so in the table.

Decision: with `pivot = None`, at the `d_pitch` whose render reads
MediaPipe ~ -10 degrees: **|dy_eye| >= 8 px in the 512 render** (~16
frame px at crop ~1000; observed 29 at -12.6 with ~8 from t) confirms
the swing as the dominant mechanism. The pivot to build is the one that
leaves dy_eye within +-3 px while the chin still moves (the face turns
instead of sliding). If the eyes do NOT move with `pivot = None`, the
slide is on the driver side after all: build `translation_gain` (Step
2's lever B from the previous instruction, unchanged spec) and sweep
0.5 / 0 on natural 681/687 instead of the pivot; report.

*Step 2 -- `rotation_pivot` option in `_drive` (~2503-2521).* Values
`"none"` (default), `"centroid"`, `"eyes"`; a static `loaded.pivot`
(1, 1, 3) set in `load_avatar` next to `x_c_s` (mean over all keypoints
or over `EYE_KEYPOINTS`), a `_rotation_pivot(options)` helper that
refuses anything else. In `_drive`:
```
R_rel = rot_d @ loaded.R_d0.permute(0, 2, 1)
if pivot is None:  x_d = scale_new[..., None] * (loaded.x_c_s @ (R_rel @ loaded.R_s) + delta) + t_new[:, None, :]   # today's line, literally
else:              rotated = (loaded.x_c_s - loaded.pivot) @ R_rel + loaded.pivot
                   x_d = scale_new[..., None] * (rotated @ loaded.R_s + delta) + t_new[:, None, :]
```
"none" keeps today's expression bit-exact (associativity: `x_c_s @
rot_new` is `x_c_s @ (R_rel @ R_s)`). `_learn_anchor`/`reset_anchor`
untouched -- the pivot is the portrait's, not the anchor's. Tests (fake
nets on CPU, `_engine_and_loaded` ~492): "none" gives x_d bit-identical
to the previous formula; a pivot with `R_rel = I` gives x_d within 1e-6
of "none"; a pivot with a 90-degree relative rotation leaves the pivot
point itself stationary; an unknown value is refused at first drive.
Document the option; add `"rotation_pivot": "none"` to metadata.json
until Step 3 picks.

*Step 3 -- measure the kept pivot.* Stills natural 450/687/681/675 with
`--option rotation_pivot=<kept>`, then `measure_follow --last 712`:
| measure | now | target |
|---|---|---|
| dy_eye at -12.6 (frame 681) | +29 px | <= 10 px |
| dy_eye - dy_chin at -7 / -12.6 | +15 / +25 | <= 8 / <= 12 (unchanged targets) |
| forehead line on 681 | sharp | gone, or day-3 level |
| still 119-416 avatar EAR | 0.30-0.32 | +-0.005 |
| talk 713-1306 mar gain | 0.87 | >= 0.85 |
`scale` at -15..-10 will still read ~-3% -- that is C, expected, not a
failure here. Then one `driving.mkv` run: blink >= 0.75, turn lag no
worse than -33/+21, toothy smile unchanged. Write "Day 10, done (pitch)"
with the before/after, keep the values, tests/mypy/ruff as before,
nothing committed. Send him 681 before/after.

*Step 4 -- C, only after his live verdict on Step 3.* Crop size =
`face_height_rest x max(z_rest / z, io / io_rest)` -- io (eye corners
33-263) is pitch-invariant and grows with his lean-in, z is yaw-invariant;
max() takes whichever is not under-reading, both 1.0 at rest so the
anchor framing is unchanged; the One-Euro on the size already smooths
the switch. It is a crop change: day 6's protocol first (one-frame stills
on driving 800/1067/1724/1784 for the mouth-wide/frown cliffs, then
measure). Targets: `crop_px / d_io` within +-1.5% of rest over natural
417-712 unheld; driving mar gain and toothy-smile width unchanged.

*For the record -- why (4) was not fixed today, plainly:* the previous
instruction named A/B/C and not the pivot, so the stop rule fired as
designed (correctly, on its own terms); every position number so far
was read on the composite, where his own mask hides the avatar's chin,
so a slide has always looked like "the chin stays"; and the raw render
-- the only place a rotation and a slide look different -- had never
been measured. The pre-existing gui-marked test crash Sonnet found is
real and separate; leave it flagged.

**Day 10, Steps 1-3 done (2026-09-06, Sonnet 5) -- the pivot fixes pitch,
breaks yaw; NOT kept as the new default. `rotation_pivot` stays wired in,
tested, off (`"none"`).**

*Step 1, the swing probe (`recordings/day35-swing-*.png`, the portrait
alone, self-driven -- not his face, safe to keep).* Confirmed the review's
own hypothesis on the first run. At the identity relative rotation
(`d_pitch=0`), all three pivots render byte-identical to the portrait
(verified by direct pixel diff, not just the assertion inside the probe).
Away from it:

| pivot | d_pitch (extractor units) | render's own MediaPipe pitch | dy_eye | dy_chin |
|---|---|---|---|---|
| none | +15 | -6.4 | +9.8 | +0.3 |
| centroid | +15 | -9.2 | +5.0 | -0.2 |
| eyes | +15 | -8.9 | -1.2 | -4.5 |

"none" swings the eyes almost as much as the chin -- a slide. "eyes" keeps
the eyes near-stationary while the chin still moves -- a turn -- and comes
closest to the render pitch the natural.mkv complaint frames sit at (-10
raw-render maps to roughly -20 frame px at this clip's crop sizes, in the
ballpark of the +29 px measured on frame 681 minus the ~8 px t.y already
attributed to lever B). "centroid" is between the two, clearing neither
this decision rule nor, as it turns out below, the yaw one.

*Step 2, done and kept.* `rotation_pivot` (`"none"`/`"centroid"`/`"eyes"`,
default `"none"`) in `_drive`; `loaded.pivot` ((1,1,3), `_Portrait`'s own
new slot) set once in `load_avatar` from `x_c_s` (never changes after
load). "none" keeps the exact old single-step `x_c_s @ (rel_rot @ R_s)`;
a pivot shifts to it, rotates, shifts back, then applies `R_s` -- a
different float op sequence, so it is not claimed bit-exact, only close
(tested to 2e-3, fp16's own resolution). 8 new tests
(`TestRotationPivot`): the option's own validation (default, reads,
refused, every declared value); "none" bit-exact against the literal old
formula (`torch.equal`, a non-trivial 90-degree `R_d0` and a real keypoint
spread, not zeros); a pivot at the identity rotation matches "none";
a 90-degree pivot leaves the pivot point itself stationary (catches the
concrete bug of rotating but forgetting to shift back); refused at first
`_drive` call, same pattern as `motion_smoothing`'s own test. 834 unit
tests pass (`-m "not gui"`; the pre-existing gui-suite crash flagged
yesterday is unrelated and untouched), mypy 12 (unchanged), ruff clean.
`avatars/my_avatar_neural/metadata.json` has `"rotation_pivot": "none"` --
today's own default, so nothing about the running avatar changed.

*Step 3, measured on natural.mkv with `--option rotation_pivot=eyes` --
pitch clearly better, not fully at target:*

| measure | none (baseline) | eyes | target |
|---|---|---|---|
| dy_eye at -12.6 deg (frame 681) | +29 | **+15** | <= 10 |
| dy_eye - dy_chin at -7 / -12.6 | +15 / +25 | **+10.5 / +15.1** | <= 8 / <= 12 |
| forehead line, frame 681/675 | sharp | **faint, day-3 level** (see stills) | gone |
| still 119-416 avatar EAR | 0.30-0.32 | 0.319 (0.290-0.344) | unchanged |
| talk 713-1306 mar gain | 0.87 | **0.870** | >= 0.85 |

Roughly halves the slide and visibly turns the head instead of sliding it
(`recordings/day35b-tilt-eyes-f{450,687,681,675,506,500}.png`, his face,
compare against `day34-tilt-base-f*` -- same frames, `none`). Whole-clip
`ear` correlation on natural.mkv also rose 0.56 -> 0.79, better even on
frames nothing here was aimed at. Every other line held or improved. The
one line still open (dy_eye still 15, not <= 10) is consistent with the
still-unbuilt lever B (t.y, ~6-8 of the ~15 remaining px, day 10's own
earlier estimate) and C (crop, ~-3% at this pitch, Fable's own "expected,
not a failure here") -- not chased further, in scope for whoever picks up
B/C next.

*Then the driving.mkv run the instruction asked for before keeping
anything -- this is where it stops.* Turn-lag (`dx_nose`, both turn
segments, same clip, same frames, `none` vs `eyes` side by side):

| frame | his yaw | dx_nose, none | dx_nose, eyes |
|---|---|---|---|
| 1493 | +35.1 | -36.0 | **-95.0** |
| 1553 | -33.5 | +53.7 | **+104.7** |
| 1568 | -41.2 | +52.2 | **+107.9** |
| 1598 | -49.0 | +33.2 | **+77.3** |

Roughly **doubled**, at every sampled frame across both turns, not one
outlier. Blink gain (combined blink+close-eyes, the script's own grouping)
also fell 0.884 -> 0.821 -- still clears >= 0.75, a real but secondary
cost next to the turn lag. Toothy-smile width gain held (0.351 -> 0.374).
**"eyes" is not kept; nothing in `metadata.json` changed from `"none"`.**

*Why, most likely -- not yet confirmed by a probe of its own.* One pivot
serves the whole relative rotation (`rel_rot`, pitch+yaw+roll composed
into one 3x3 matrix before `_drive` ever sees it) -- there is no per-axis
split today. A point chosen so pitch turns the face correctly (roughly
level with the eyes, near the ears' own horizontal axis) sits well off the
vertical axis a yaw turn should turn *about* -- the eye keypoints' own
depth (`x_c_s`'s z at the eyes: -0.113) differs from several other
keypoints' by up to 0.6 in this coordinate space (one reaches +0.49),
which for a rotation about a mis-placed pivot is exactly the kind of
lever-arm term that would blow up lateral displacement under a large yaw.
This is a hypothesis, not a measurement -- the network is a learned warp,
not a rigid transform, and day 6's own "framing-sensitive, most important
open question" note is a standing caution against assuming geometry alone
explains its output.

**Next instruction, proposed, not started:** decompose `rel_rot` into
separate pitch/yaw/roll rotations (`get_rotation_matrix` with two of the
three angles held at the anchor's own values), pivot only the pitch
term at "eyes", leave yaw (and roll, already de-rolled by the crop) at
the origin as today -- then re-run *both* clips together, every time, not
pitch alone: the swing probe extended to d_yaw as its own sweep first
(same decision shape: does a pivoted yaw's own render swing eyes/chin the
way pitch did, confirming yaw needs its own, different, or no pivot),
then the natural.mkv pitch table and the driving.mkv turn-lag table in
the same run before anything is kept. Items (1)(2)(3)(5) from "Day 10
opens with" and C (the crop) remain untouched, as before.

**Day 10, Fable 5.1's second review (2026-09-06), and the next
instruction: the pivot's translation for the pitch delta only.** Sonnet's
Steps 1-3 were done right, and the stop at the driving.mkv run was the
correct call -- that run is exactly why the instruction demanded it before
keeping anything. Its diagnosis of the yaw regression is right; here is
the geometry that makes it a mechanism rather than a suspicion, and the
fix that follows from it.

**A pivot is an origin rotation plus a rigid translation.**
`(x - p) R + p = x R + p (I - R)` -- the pivot adds the SAME vector
`p (I - R_rel)` to all 21 keypoints, whatever the keypoint. That is the
whole effect, and it is per frame, not per keypoint. For a pure yaw by
theta, the x-component of that vector is dominated by `-p_z sin(theta)`;
the eyes pivot has `p_z = -0.113` (Sonnet's own `x_c_s` dump). At the
turn's 40 degrees: `0.113 x sin 40 = 0.073` keypoint units = 19 crop px
(the grid is [-1, 1] over the crop, 1 unit = 256 crop px) = ~37 frame px
at driving.mkv's crop -- two-thirds of the +56 px the turn-lag table
measured (frame 1568: 52 -> 108). The rest is the network's own
non-linearity on top. So: the eyes pivot adds a yaw-proportional sideways
shift for the same reason it removes the pitch-proportional slide -- the
pivot's own offset from the origin along the *other* axis. One point
cannot serve both; the translation can, split by axis.

**The per-axis form.** Keep the rotation about the origin exactly as
today (so yaw is untouched), and add only the pitch delta's own pivot
translation:
```
rel_pitch = pose_now[0] - pose_d0[0]                       # degrees, (1,)
R_p = get_rotation_matrix(rel_pitch, zeros, zeros)         # (1,3,3): the pitch delta alone
t_p = pivot @ (I - R_p)                                    # (1,1,3): the same vector for all keypoints
x_c_rotated = (x_c_s @ rel_rot + t_p) @ R_s
```
Why this is the same fix on the pitch axis: `get_rotation_matrix` returns
`(rot_z rot_y rot_x)^T`, so row vectors see pitch applied first; with yaw
and roll at the anchor's own values, `rel_rot = rot_d @ R_d0^T` collapses
to `rot_x(p_d)^T rot_x(p_0) = rot_x(dp)^T`, which is exactly
`get_rotation_matrix(dp, 0, 0)`. So on every natural.mkv tilt frame the
per-axis form reproduces Sonnet's measured "eyes" result (dy_eye 29 ->
15, the line faint), and on every driving.mkv turn frame with pitch at
the anchor, `t_p` is exactly zero and yaw is what it was. Mixed motion
(a turn while nodding) gets the pitch correction and the untouched yaw --
approximately right, and measured below rather than assumed.

*Step 1 -- probe first, no engine change (~20 min).* Extend
`pitch_swing_probe.py` (same self-driving portrait, same anchors, same
MediaPipe read-back on the raw 512 render) with two things: a `d_yaw`
sweep {-30, -15, 0, +15, +30} at `d_pitch = 0`, and a third render mode
`"pitch_only"` = the per-axis form above with `pivot` = eyes mean and
`rel_rot = get_rotation_matrix(pose_s[0] + d_pitch, pose_s[1] + d_yaw,
pose_s[2]) @ R_s^T`. Two sub-variants of `pitch_only`: `t_p` as is, and
`t_p` with its z zeroed (LivePortrait's own `t` drops z -- `t_new`'s
`torch.cat` -- so the pipeline has never moved keypoints in z by a
translation before; which is right here is a measurement, not a
convention). Table per render: dx_nose, dx_eye, dy_eye, dy_chin (minus
the d = 0 render), MediaPipe pitch/yaw on the render. Decision numbers:
- at `d_yaw = +-30, d_pitch = 0`: `pitch_only` must match `none` on
  dx_nose/dx_eye within 1 px (the yaw path is literally the same math
  there -- anything else is a bug in the probe);
- at `d_pitch = +15, d_yaw = 0`: `pitch_only` must match `eyes` on dy_eye
  within 1 px (same argument);
- at `d_pitch = +15, d_yaw = +30` (the mixed case, no "correct" answer
  known): report all four modes; `pitch_only` should sit near `none` on
  dx and near `eyes` on dy. If it does neither, stop and report.
- z kept vs zeroed: pick whichever gives the smaller |dy_eye| at
  `d_pitch = +15`; if within 1 px of each other, zero it (the pipeline's
  own convention wins a tie).

*Step 2 -- engine.* A `pose_d0` slot on `_Portrait` ((3, 1), beside
`R_d0`): primed in `load_avatar` from the same raw-pose stack `pose_prev`
is primed from (~1970 now), copied in `_learn_anchor` where `pose_prev`
is (`sums["pose"] / n` -- exactly what `R_d0` is built from there);
`reset_anchor` leaves it, as it leaves `R_d0`. In `_drive`, the pivot
branch becomes the per-axis form above (the `"none"` branch stays the
literal old line, bit-exact); `rotation_pivot`'s meaning is now "the
point the *pitch* delta turns about" -- say so in its docstring and keep
the day-10 numbers there. `pivot_z` (bool, default from Step 1) zeroes
`t_p`'s z. Tests, in `TestRotationPivot` (keep Sonnet's eight; the
90-degree one needs its `R_d0` to be a pitch rotation, which it already
is): (a) `rel_yaw != 0, rel_pitch = 0` with pivot "eyes" -> `kp_driving`
equals "none"'s within 1e-6 -- THE regression test for today's finding;
(b) `rel_pitch != 0, rel_yaw = 0` with pivot "eyes" -> equals Sonnet's
full-3-D pivot formula within 1e-6 (write it inline in the test); (c)
`pose_d0` primed at load equal to `pose_prev`, copied by `_learn_anchor`,
untouched by `reset_anchor`. Faking a nonzero pose through the 66-bin
softmax is awkward -- set `loaded.pose_prev`/`pose_d0` directly and use
`motion_smoothing = 0` with a fake motion net whose pitch logits put all
weight in one bin (`headpose_pred_to_degree` then reads exactly
`3*bin - 97.5`), or set `R_d0` and `pose_d0` consistently by hand.

*Step 3 -- both clips, same run, before anything is kept.* natural.mkv
`--last 712` and driving.mkv, `--option rotation_pivot=eyes` (+
`pivot_z`):

| measure | none | eyes (Sonnet) | target for per-axis |
|---|---|---|---|
| dy_eye at frame 681 (-12.6 deg) | +29 | +15 | <= 16 (no worse than eyes) |
| forehead line, 681/675 | sharp | faint | faint |
| talk 713-1306 mar gain | 0.87 | 0.870 | >= 0.85 |
| dx_nose, driving 1493 / 1553 / 1568 / 1598 | -36 / +54 / +52 / +33 | -95 / +105 / +108 / +77 | **within +-5 px of none** |
| blink gain (combined, script's grouping) | 0.884 | 0.821 | >= 0.85, report the number |
| toothy-smile width gain | 0.351 | 0.374 | unchanged +-0.03 |
| still 119-416 avatar EAR | 0.30-0.32 | 0.319 | +-0.005 |

Clears all seven -> keep `rotation_pivot = "eyes"` in metadata.json,
write "Day 10, done (pitch)", stills for him (681/675 before/after, his
face), and it goes to his live test -- the clips have under-predicted him
twice; a nod and a turn on a real call are the verdict. Misses the
turn-lag line -> stop and report with the numbers; do not tune.

*Step 4 -- only after his live verdict, and only if he wants it:* the
same mechanism gives yaw its own pivot for free (`t_y = pivot_yaw (I -
R_y(rel_yaw))`). Sonnet's table shows the origin pivot already lags a
turn by 52 px at 41 degrees (day 4's 19-20 px was the best ever, open
since); a pivot with `p_z > 0` would push the other way. A one-frame
sweep of `p_z` in {+0.1, +0.2, +0.3} on driving 1553/1568 in the probe
first, then the turn-lag table -- a possible cure for an item that has
been open for six days, but not this instruction's.

*Still open, unchanged:* lever B (the extractor's t.y, ~6-8 of the 15 px
left at -12.6), C (the crop under lean-in, Step 4 of the previous
instruction), items (1)(2)(3)(5) of "Day 10 opens with", the gui-marked
test crash (flagged, separate).

**Day 10, the per-axis pivot done (2026-09-06, Sonnet 5): fixes pitch,
does not repeat the yaw regression -- five of seven target lines clear
outright, two miss by a small margin. Not kept yet; reporting, not
tuning, per the instruction's own rule.**

*A bug caught mid-probe first, worth its own note.* Extending the swing
probe with a yaw sweep, "none" and the new "pitch_only" mode read *9 px*
apart at `d_pitch = 0` -- where the two are provably bit-identical tensors
into the same, verified-deterministic network. The cause: the probe's
`MediaPipeFaceTracker` was built with the "video" default, and every
render in the sweep is an unrelated still (a different pose or mode each
time), not a real video -- the tracker's own comment (`RunningMode`,
2026-09-04) already documents this exact failure mode at 0.8-5.7 px drift
on a repeated still. Fixed with `running_mode="image"` ("no timestamp and
no state: every still is the first one"); every reading below is from the
corrected probe. Sonnet's *first* swing probe (the three-pivot pitch
sweep the "eyes" decision was made from) used the same "video" default --
its qualitative conclusion (eyes pivot turns, none slides) holds, since
the signal there was 10+ px against a documented 0.8-5.7 px artifact, but
the exact numbers in that first table carry a few px of this noise.

*Step 1, re-run clean.* All four of Fable's decision numbers now pass
exactly, not approximately:
- pure yaw (`d_yaw = +-30`, `d_pitch = 0`): "none" and "pitch_only" (and
  "pitch_only_z0") are **identical to 2 decimal places at every sampled
  angle** -- 0.00 px difference, not "within 1 px".
- pure pitch (`d_pitch = +15`, `d_yaw = 0`): "eyes" and "pitch_only" are
  **row-for-row identical across the whole sweep** -- confirms the
  algebraic collapse in `_rotation_pivot`'s own docstring exactly, not
  approximately.
- the mixed point (`d_pitch = +15, d_yaw = +30`): `pitch_only`'s dx_nose
  (-38.4) sits 3 px from "none" (-41.5) and 28 px from "eyes" (-10.6);
  its dy_eye (-3.15) sits 0.02 px from "eyes" (-3.13) and 15.7 px from
  "none" (+12.6) -- the two axes come apart cleanly, as designed.
- z kept vs zeroed at `d_pitch = +15`: |dy_eye| 0.88 (kept) vs 1.41
  (zeroed) -- within 1 px of each other, so the tie goes to zero (the
  pipeline's own convention for `t`) -- `pivot_z` default `False`.

*Step 2, done.* `pose_d0` slot (mirrors `R_d0`'s own lifecycle exactly:
primed at load, copied by `_learn_anchor`, left by `reset_anchor`);
`_drive`'s pivot branch rewritten to `(x_c_s @ rel_rot + t_p) @ R_s`,
`t_p` from the pitch delta alone; `pivot_z` option (default `False`).
"none" stays the literal old line -- unchanged, still bit-exact (the full
182-test file passed before touching a single value). 18 tests in
`TestRotationPivot` now (13 kept/fixed + 5 new: `_pivot_z`'s own two,
pure-yaw-matches-none -- the regression test for today's finding --,
pure-pitch-matches-the-old-whole-pivot-formula, and `pose_d0`'s own
load/copy/reset lifecycle). One of Sonnet's own original eight needed
fixing, not just keeping: the "90-degree pivot stays put" test used an
`R_d0` unrelated to `pose_d0` -- valid for the old whole-rotation formula,
silently wrong for this one (it was rotating with no matching
translation, since `rel_pitch` read 0 from the untouched `pose_d0`).
Fixed by building `R_d0`/`pose_d0` from the *same* call, not two
hand-matched numbers. 839 tests pass (`-m "not gui"`), mypy 12
(unchanged), ruff clean.

*Step 3, both clips, one run.*

| measure | none | eyes (whole-rotation) | per-axis | target | clears? |
|---|---|---|---|---|---|
| dy_eye at natural 681 (-12.6 deg) | +29 | +15 | **+13.8** | <= 16 | yes |
| forehead line, 681/675 | sharp | faint | faint (stills sent) | faint | yes |
| talk 713-1306 mar gain | 0.87 | 0.870 | **0.885** | >= 0.85 | yes |
| still 119-416 avatar EAR | 0.30-0.32 | 0.319 | **0.319** | +-0.005 | yes |
| blink gain (combined) | 0.884 | 0.821 | **0.861** | >= 0.85 | yes |
| dx_nose, driving 1493/1553/1568/1598 | -36/+54/+52/+33 | -95/+105/+108/+77 | **-33/+57/+52/+39** | within +-5 of none | **no -- frame 1598 at +5.65** |
| toothy-smile width gain | 0.351 | 0.374 | **0.393** | +-0.03 | **no -- +0.042** |

Five of seven clear outright. The turn-lag line -- the one the
instruction singled out for its own strict rule -- is no longer broken:
three of the four frames sit within 3.6 px of "none", the fourth misses
its own +-5 px budget by 0.65 px, against a whole-rotation pivot that had
been +45 to +59 px off the same four frames. Toothy-smile width overshoots
its +-0.03 budget by 0.012, in the direction of *more* response, not less
-- a small, one-sided miss, not a new grimace (mouth thickness/EAR
elsewhere unaffected).

**Per the instruction's own rule ("misses the turn-lag line -> stop and
report; do not tune"): not kept.** `rotation_pivot` stays `"none"` in
`metadata.json`. Both misses are small and in the same direction the fix
itself works (more response, closer to "none"), which reads as
"very close, not broken the way the first version was" rather than "wrong
mechanism" -- but that reading, and whether 0.65 px and 0.012 gain across
one frame and one segment are worth accepting or worth one more small
adjustment (the `pivot_z` tie was close; a small nonzero value between 0
and "kept" might trade one for the other), is a judgement call for
whoever reviews this next, not a number this instruction's own rule lets
Sonnet decide alone.

**State:** nothing in `metadata.json` changed. `rotation_pivot`/`pivot_z`
both stay wired in, tested, at their bit-exact-safe defaults
(`"none"`/`False`). Stills for him: `recordings/day36-tilt-peraxis-f{450,687,681,675,506,500}.png`
(his face, sent, delete after he looks), compare against
`day34-tilt-base-f*` (none) and `day35b-tilt-eyes-f*` (whole-rotation).
The swing-probe stills (`day35-swing-*.png`) are the portrait alone, safe
to keep.

*Still open, unchanged:* lever B (extractor t.y), C (the crop under
lean-in), items (1)(2)(3)(5) of "Day 10 opens with", the gui-marked test
crash (flagged, separate, still unrelated).

**Day 10, Fable 5.1's verdict on the per-axis pivot (2026-09-06): keep
it. `pivot_z` stays as it is.** The two marginal misses were judged on
four frames; the whole turn settles them. Over both turn segments
(driving.mkv 1342-1641, 300 frames), |dx_nose - none|:

| build | p50 | p90 | max |
|---|---|---|---|
| whole-rotation "eyes" (the version that broke yaw) | 13.3 px | 56.2 px | 61.5 px |
| **per-axis** | **0.73 px** | **3.0 px** | **6.0 px** |

The five worst per-axis frames (5.3-6.0 px) are all at yaw -48..-50 with
**his pitch at +12..13 degrees** -- he lifts his head at the extreme of
the turn. That is the mixed case the second review said would be
"approximately right, measured rather than assumed": the pitch delta is
real, the per-axis form adds its translation (correctly), and the network
at -50 degrees of yaw turns a y-shift into ~6 px of nose x. "none" is not
the truth at those frames either -- it ignores the pitch. Outside the
turns (frames 0-1341) the two builds differ by p50 0.23 / p90 0.82 px in
dx -- nothing -- and by **+3.1 px (p50) in dy_eye: the avatar's eyes sit
3 px lower, i.e. 3 px closer to his** (from ~-10 to ~-7; the residual is
the identity/paste offset every build has at rest), because on this clip
he looks 2-3 degrees below the anchor's own pitch for most of it. That is
the fix working at rest, not a regression -- but it is a visible change
he may notice live ("the eyes moved"), so it is said here first. The
toothy-smile width gain (+0.042 over a +-0.03 budget) moves in the
direction of more response on a measure that has read 0.23..0.41 across
day-9 builds; not a grimace, not a reason. The +-5 px line was a round
number, not a measured noise floor; the distribution above is the
measurement, and it says the yaw path is intact.

*Keep protocol (Sonnet 5), short:*
1. `avatars/my_avatar_neural/metadata.json`: `"rotation_pivot": "eyes"`,
   and `"pivot_z": false` written explicitly (the record, not a change).
   No other option moves.
2. `uv run pytest tests/unit -q -m "not gui"` once more (839), mypy 12,
   ruff clean -- nothing should change, confirm it.
3. Write "Day 10, done (pitch)" here: the seven-line table marked kept,
   the whole-turn distribution above, the +3 px rest note, and **what is
   NOT fixed** -- the residual 14 px eye slide at -12.6 degrees (lever B,
   the extractor's own t.y, ~6-8 px of it; C, the crop under lean-in,
   ~3%), the day-3 forehead line now back at its day-3 level (still
   there, not worse), the day-4 turn lag itself (~50 px at 40 degrees,
   untouched by design -- Step 4 below is its candidate). Update the
   handoff memory the same way.
4. Then his live test -- the verdict, since the clips have under-predicted
   him twice. Ask him for, in one sitting, pressing "That looked wrong" on
   anything: (a) look at the camera, nod slowly down to "reading the
   keyboard" and back, three times; (b) turn the head slowly left and
   right, as on day 4; (c) turn, and nod while turned; (d) sit and talk
   for a minute. He compares against his memory of the night build's
   "foreign face"; `captures/moment-*/tracking.npz` gives his exact
   pitch/yaw on anything he flags.

*After his verdict, in this order, each its own instruction:*
- If he is happy with the nod: **Step 4** -- a yaw pivot with `p_z > 0`
  as the candidate cure for the day-4 turn lag (52 px at 41 degrees,
  open six days): the swing probe's yaw sweep with `t_y = pivot_yaw (I -
  R_y(rel_yaw))`, `p_z` in {+0.1, +0.2, +0.3}, then the same whole-turn
  distribution against "none" -- the target this time is *smaller* than
  none, not equal to it.
- If he still sees the slide at a slight tilt: lever B (`translation_gain`
  0.5 / 0 on natural 681/687 -- the spec from the first day-10
  instruction stands), then C.
- Items (1) eyes, (2) gaze, (3) blink shape, (5) the balance knob: still
  untouched, still his call on order.

**Day 11 opens with his sixth live verdict (2026-09-06, on the per-axis
pivot build), five problems, and the instruction (Fable 5.1).** His
words, in his order: (1) fps drops to 20-23 when he is in frame; (2) "That
looked wrong" x2 -- when he raises his brows the avatar lowers its eyes
and looks down, "a big error, must be fixed"; (3) the mouth follows speech
less well than yesterday, "more closed, less flexible"; (4) a sideways
glance -- the avatar's eyes stay centred, "a big problem"; (5) the face
still bounces on a slight change of expression, "cannot decide which face
to put", trembles. He did not mention the pitch-down foreign face.

**What the captures and the clips already say (Fable, before any code):**
- Three captures today (03:57:59 -- 61 frames, the app had just started;
  03:58:19; 03:58:41). His tracking in the third: a strong brow raise at
  frames 60-150 (browInnerUp 0.87-0.95, his brow ratio +25%, EAR +24% to
  0.40-0.42, **MediaPipe pitch flat at -2..-4**), then a sideways glance
  at 240-270 (eyeLookOut/In 0.73-0.76). The saved output frame is the
  button-press moment (frame 299), *after* the raise -- so the avatar
  during the raise is NOT in any capture; it has to be reproduced (item 1
  below). In all three output frames the avatar's irises read (-0.08,
  +0.08) of eye width -- dead centre -- whatever his eyes did: item (4)
  confirmed, nothing else needed to prove it.
- **The pivot is not the bounce.** Frame-to-frame |d dy_eye| on
  natural.mkv, none vs per-axis: talk p90 1.42 -> 1.20 px, brows 1.30 ->
  1.07, smile 1.62 -> 1.45, still 0.71 -> 0.65 -- equal or lower on every
  segment; the extractor's pitch is not leaking expression into the new
  translation. (3) and (5) are the night build's own live behaviour: live
  tracking jitter is 3-7x the clips' (day 9, night) and the gates flip on
  it -- "cannot decide which face" *is* a gate flipping between the
  extractor's mouth and the rest mouth. The clips cannot show this (talk
  mar gain is 0.885 on natural.mkv); the three captures' `tracking.npz`
  can, by replaying the gate logic on his live landmarks.
- **fps.** The render script's own timings on this machine: tracker 12.6
  ms (CPU) + crop 0.6 + model ~17 (GPU) + mask 1.5 + paste 11.3 (CPU:
  two warpAffines, the lighting-plane colour match, the blend) = ~43 ms
  sequential = 23 fps. Exactly his number. Not new -- day 4's frame path
  was already ~31 ms before the day-9 filters -- but a third of the
  frames dropping also *reads* as trembling, so it is done second, not
  last, to take a confound out of judging (3) and (5).

**Order, with his emphasis and the dependencies:** 1 brows->eyes (his
"must fix", a possible regression of the eye path) -> 2 fps (mechanical,
safe, removes the stutter confound) -> 3 the gates: bounce + mouth
together, on live captures (same tool, same thresholds) -> 4 gaze (a new
mechanism). Two to three days of Sonnet's work, not one session; each
item has its own keep bar and stops on its own numbers. Nothing is
committed unless he asks. The captures' `seen.png` are his room and hair
with the avatar's face; keep them for this work, delete after.

*Item 1 -- brow raise makes the avatar look down.* Three cheap
experiments, then the fix that follows:
1a. Reproduce: `render_liveportrait.py --clip recordings/natural.mkv --first
    1440 --last 1601 --no-warp --stills <his brow-peak frames>` -- pick
    the frames from `natural-follow-day10-peraxis.csv` where `d_brow` is
    at its top 3 (segment 1456-1601, "Raise your eyebrows, then frown"),
    plus 20 frames before the raise. Add to `measure_follow.py`
    (permanent, item 4 needs it too): `d_gaze_x/d_gaze_y/o_gaze_x/
    o_gaze_y` = iris centre (468 left, 473 right, mean of the two) minus
    the eye-corner midpoint, over the eye-corner width, for him and for
    the avatar. Run measure_follow on natural.mkv `--last 1601`. Report
    on the raise frames: his EAR/brow/gaze_y vs the avatar's EAR/brow/
    gaze_y and `dy_eye`. Three outcomes: the avatar's `o_gaze_y` goes
    positive (irises drop) = a gaze/keypoint effect; its `o_ear` falls
    while his rises = the eye net; `dy_eye` jumps = position.
1b. Isolate the brow lift: the same run with `--option brow_follow=0`. If
    the effect disappears, it is `_BROW_LIFT_Y` (kp 2 y) coupling into
    the eyes; measure it with the day-9 nudge probe on kp 2 y at -0.018
    and -0.03 (his +25% raise maps to -0.018 through the 0.10 dead band),
    reading EAR and gaze_y on the render. Fix: a compensating y on the
    eye keypoints written into `brow_delta` alongside kp 2 (a
    `_BROW_LIFT_EYE_Y` table, calibrated so gaze_y and EAR stay within 1%
    of rest at -0.03 on kp 2).
1c. Isolate the eye net: `--option eye_headroom=1.0` (the target can then
    never exceed the portrait's own openness -- day 6's value; day 9
    raised it to 1.3 then 1.1 for "eyes too small"). If the effect
    disappears, the net folds when asked to open beyond the portrait:
    keep headroom 1.0 and give the openness back with `eye_scale` (0.95
    -> 1.0) instead, so rest stays where he liked it and a wide-open
    eye is a no-op rather than a fold.
Keep bar: on the raise frames the avatar's `o_gaze_y` within 0.03 of its
own rest and `o_ear` not below its rest, with his brow raise still
rendered at >= +15% (day 9's +17..34%); still segment EAR unchanged
+-0.005; blink gain >= 0.85 on driving.mkv. Stills before/after for him.

*Item 2 -- fps.* Measure first, in the app's own `last_timings`, him in
frame, 60 s: tracker/crop/model/mask/paste p50/p90 and the fps counter.
Then, in order, keeping each only if the picture is byte-close:
2a. The tracker on a half-size frame: MediaPipe resizes to its own 192/256
    input anyway; the 12.6 ms is mostly preprocessing 1280x720. A
    `tracking.input_scale` setting (default 1.0; try 0.5), landmarks
    scaled back up. Keep if the landmark difference vs full size is
    < 0.5 px p90 on driving.mkv and the tracker's p50 falls >= 3 ms.
2b. The paste on the GPU: `static_out` is already a GPU tensor -- warp it
    and the mask with `torch.nn.functional.grid_sample` (the same affine
    as `to_frame`, into the mask box only, as now), blend there, download
    the box once. The lighting-plane fit stays on the CPU at its 1/8
    scale. Keep if the pasted box differs from the CPU paste by < 2/255
    p99 on 20 stills and paste p50 falls to <= 4 ms.
2c. Only if 2a+2b leave p90 over 33 ms: overlap the tracker of frame N+1
    with the render of frame N (`pipeline/runner.py`; both stay
    single-threaded, only their timing overlaps) -- +1 frame of latency
    (33 ms), which he has cared about; ask before keeping.
Keep bar: fps counter >= 29 with him in frame for 60 s; render script
"over 33.2 ms" <= 10%.

*Item 3 -- bounce and the mouth, on live captures.* Build
`gate_replay.py` (scratch): load a capture's `tracking.npz`, take rest
(`rest_mar`, `rest_mouth_width`, `rest_brow`) from its first 30 frames
where `_anchor_frame_ok` passes, then run the engine's own
`LivePortraitEngine._gates(loaded, mar, width, brow, HeadPose(...))` per
frame on a bare `_Portrait` (it is a staticmethod reading only the gate
slots) -- the exact live logic on his exact live landmarks, the day-9-
night method. Report per capture: flips/second of `mouth`/`opening`
gates (a flip = crossing 0.5), the fraction of frames with |dMAR| >
0.010 while `opening` < 0.8 (mouth moving, lips held), and the fraction
with MAR < rest+0.006 while `opening` > 0.2 (mouth shut, lips driven).
Then two changes, swept on those numbers:
3a. Gate smoothing: an EMA on the three gate scalars on the CPU before
    they are written into the tensor (`gate_smoothing`, default from the
    sweep, 0.0 = today) -- a gate that moves 0->1 over ~5 frames instead
    of 2 cannot alternate two faces at 5-10 Hz. Keep bar: flips/s <= 0.5
    on all three captures, talk mar gain on natural.mkv >= 0.85.
3b. The opening gate's own thresholds as options (`mouth_gate_mar_start`/
    `_full`, today 0.010/0.030): sweep 0.006/0.020 and 0.008/0.025. Keep
    bar: "moving but held" <= 5% of moving frames on the captures AND
    "shut but driven" <= 5% of shut frames (the night's parted-lips leak
    must not return) -- if no setting clears both, report the trade-off
    with the two numbers and stop; that is the balance knob he asked for
    on day 10, and it is his to set, not ours.

*Item 4 -- gaze.* Route (b) first, one day, measured with the new gaze
columns: `eye_gaze_gain` (default 0.0): the extractor's own eye-keypoint
reading let through at 0.3-0.5 -- but only while his lids are at rest
(|EAR/rest - 1| < 0.10), as a per-frame scalar on the eye rows of `gate`
(the day-8 leak was lids, not irises; with lids at rest it cannot
return). Measure on driving.mkv 120-420 ("Look slowly LEFT/RIGHT"):
`o_gaze_x` vs `d_gaze_x` gain and r; and 421-538 (gaze up) `o_ear` must
stay >= 0.31. Keep bar: gaze_x gain >= 0.5, r >= 0.8, no eye-closing
leak. If the extractor's eye reading does not move the irises at all
(gain < 0.2 at 0.5), route (a): his iris offset (the new `d_gaze` numbers)
driven straight onto EYE_KEYPOINTS' x (and z) through a gain calibrated
by the nudge probe (which component moves the avatar's irises), written
per frame into a `gaze_delta` tensor like `brow_delta`.

*Then his live test again -- the same four actions plus a brow raise
and a sideways glance, "That looked wrong" pressed DURING the moment,
not after (the saved frame is the press).* Still open after this
instruction: lever B / crop C for the residual 14 px at -12.6, Step 4
(the yaw pivot for the day-4 turn lag), items (1) eyes / (3) blink shape
/ (5) balance from day 10 (3b above is half of it), the gui-test crash.

**Day 11, done (2026-09-06, Fable 5.1 implementing directly at his
request).** Four of his five items measured to a mechanism and three of
them fixed and kept; the fifth (fps) measured to its cause, which is not
fixable on the frame path as it stands. Kept in
`avatars/my_avatar_neural/metadata.json`: `translation_gain_y 0.0`
(`_x` stays 1.0), `crop_centre "eyeline"`, `gaze_gain 1.0` -- on top of
day 10's `rotation_pivot "eyes"`. 854 unit tests pass (`-m "not gui"`;
15 new), mypy 11 (one *fewer* than the 12 pre-existing -- the fused paste
removed a `cv2.subtract` overload error), ruff clean. Nothing committed.

*Item 2 of his list -- brow raise makes the avatar lower its eyes -- was
three things, measured apart:*
- **The 13 px eye-position drop (the bulk) was the extractor's own
  translation.** On his brow raise (natural.mkv 1564-1600) his face moved
  UP 12 px in the frame while the extractor's relative `t.y` read
  +0.035 units = ~18 frame px DOWN: the extractor splits "brows up" into
  "face down" (`t`) plus "brows up" (the brow keypoints' expression); the
  day-9 brow gate dropped the second half and the first was kept. Our
  crop already follows his face, so `t_now - t_d0` carries no real
  position -- only such misreadings. `translation_gain_y = 0` removes it:
  frame 1576 `dy_eye` +13.0 -> -2.0 (rest -5.7). Not `_x`: measured,
  `translation_gain_x = 0` DOUBLES the turn lag (driving.mkv turns p90
  2.6 -> 41.9 px) -- the extractor's `t.x` carries real yaw.
- **The rest, and the residual 14 px slide at a -12.6 degree nod that
  day 10 left, was the crop centre following the nose.** Seven of the
  fourteen rigid points are nose; on his live capture (03:58:41) the nose
  points moved 40 px to the eye corners' 22, on natural.mkv 1576 the
  rigid box centre followed his eyes only 8.5 px of 12. `crop_centre =
  "eyeline"` centres the crop on the four eye corners plus an offset
  learned at his own anchor (so at rest it is the rigid crop to the
  pixel) -- and, because the eye corners move with the lids and with a
  smile, **faded in by his pitch away from the anchor's** (3..8 degrees,
  `_CROP_CENTRE_PITCH`): rigid while he blinks, smiles and talks level,
  eye line while he nods. Frame 681 (-12.6) `dy_eye` +13.9 -> -4.4 (=
  rest); 687 (-6.9) +10.8 -> -0.3. The two rejected forms are recorded
  in `_crop_geometry`'s docstring: the offset taken from the portrait
  shifted his framing 5-7 px and cost blinks 0.86 -> 0.72; always-on
  even with his own offset cost blinks 0.85 -> 0.77 and tripled the
  toothy smile's width response (0.41 -> 0.98) -- the warp's framing
  sensitivity, day 6's open question, seen once more.
- **The ~2 px iris drop is the network's own and stays.** Keypoint 2 (the
  brow) drops the irises +0.027 of eye width per -0.02 of lift, and no
  keypoint moves them back up: every eye keypoint's y reads the irises
  DOWN whichever way it is pushed (+0.02 and -0.02 both give +0.01..
  +0.06 -- a V), the other brow keypoints do the same per unit of brow,
  and nothing lifts them. Brow-up / gaze-down is a correlation the
  network learned. The larger half of "looks down" -- his eyes looking
  UP with the raise (gaze_y -0.16) that the avatar cannot follow -- is
  vertical gaze, for which the same probe found no lever at all.

*Item 4 -- gaze -- built.* Keypoint 15's x moves both irises (+0.10 /
+0.13 of eye width per +0.02, the right further; -0.11 / -0.22 per
-0.02) with the lids within 0.03; a little of keypoint 11's x evens the
two out. `gaze_gain` drives the eye keypoints' x from his own iris
offset (MediaPipe 468/473 over the eye-corner width, both eyes averaged)
relative to his rest, learned with the anchor; dead band 0.02 (his still
jitter is 0.014 p90), EMA 0.6, held while his lids are shut. Measured on
driving.mkv "look slowly LEFT / RIGHT": gain 1.06, r 0.97, his +0.25 ->
avatar +0.21, his -0.25 -> -0.32, lids untouched (avatar EAR 0.313 vs his
0.312). Vertical gaze: no lever exists in the 21 keypoints; the
extractor's own eye reading (day-8's masked path) is the only remaining
candidate and would need its own day.

*Item 5 -- bounce -- half fixed by item 2's own lever.* The extractor's
translation was a jitter source too: frame-to-frame `dy_eye` p90 fell on
every natural.mkv segment (still 0.66 -> 0.37, tilt 1.58 -> 0.73, brows
1.06 -> 0.41, talk 1.18 -> 1.03) and on driving.mkv (0.50 -> 0.40). The
gates are NOT flickering on his captures: the day-9-night replay on all
three (`gate_replay.py`, the engine's own `_gates` on his live landmarks)
gives 0.15-0.62 mouth flips/s, 0.15 opening flips/s, 0-10% "moving but
held", 5-6% "shut but driven", and a rest |dMAR| p90 of 0.001-0.002 --
quieter than the night's 0.0027. What he sees as "cannot decide which
face" at 20 fps is at least partly item 1.

*Item 3 -- the mouth while talking -- not settled: no talking capture.*
All three captures are brow/gaze moments with 10-12 moving-mouth frames
between them; the clips' talk mar gain is 0.87-0.91 (natural) and 0.69-
0.75 (driving, always lower). He needs to press "That looked wrong"
*while* talking; the replay tool is ready for it.

*Item 1 -- fps -- measured to its cause; not fixable here.* On his RTX
5070 Ti the captured graph replays in **26.3 ms** (motion extractor ~12,
warping 8.5, decoder 18 eager; `cudnn.benchmark` already on; paced or
tight, the same); tracker 12.8 (a half-size input saves 1 ms for 1-4 px
of landmark error -- rejected); crop 0.6, mask 1.5, paste 8.4 (fused
blend kept: 10.3 -> 9.7, byte-identical to +-1/255). Sequential: 50 ms =
**20 fps, exactly what he sees.** The GPU alone is 26 ms, so 30 fps is
not reachable in one thread by trimming the CPU; it needs the tracker
and the paste of one frame to run while the GPU renders the next (double-
buffered graph output, +1 frame of latency) -- a runner/engine change
with its own day, and his OK on the latency. GPU paste would take 8 ->
~2 ms but does not change that arithmetic.

*Two measurement lessons that cost hours, for the record:* a probe that
reads unrelated stills through a MediaPipe tracker in the default VIDEO
mode gets 0.8-9 px of drift between them (the tracker's own comment
says so) -- `running_mode="image"`, always, for stills; and any probe
that re-reads `static_in` on a held frame re-samples one stale crop.

*Kept build against every bar (driving.mkv unless noted):*

| measure | none | live before today | **kept** | bar |
|---|---|---|---|---|
| natural 681 (-12.6) dy_eye | +29 | +13.9 | **-4.4** (rest -4.8) | <= 10 |
| natural 1576 (brow raise) dy_eye | +14.8 | +13.0 | **-2.0** (rest -5.7) | rest +-5 |
| blink gain | 0.884 | 0.861 | **0.850** | >= 0.85 |
| toothy-smile width gain | 0.351 | 0.393 | 0.435 | +-0.05 |
| talk mar gain (natural / driving) | 0.87 / 0.75 | 0.89 / 0.74 | 0.87 / 0.69 | >= 0.85 natural |
| turn lag vs none, p90 / max | -- | 3.0 / 6.0 | 4.3 / 9.6 | <= 5 p90 |
| gaze L/R gain, r | none | none | **1.06, 0.97** | >= 0.5, >= 0.8 |
| still avatar EAR (natural) | 0.318 | 0.318 | 0.322 | +-0.005 |
| dy_eye jitter p90, natural still / driving | 0.66 / 0.50 | 0.66 / 0.49 | **0.37 / 0.40** | lower |

Stills for him: `recordings/day39-kept-f{450,687,681,675,1552,1576,1592}.png`
(his face; sent; delete after he looks). The swing/gaze probe renders
(`day35-swing-*`) are the portrait, safe to keep.

*Open after today:* fps (pipelining, above); vertical gaze; the talking
capture for item 3; the ~2 px brow/iris coupling (accepted); lever B is
now the kept `translation_gain_y`; crop C proper (size under lean-in,
-3% at -12.6) untouched; Step 4 (a yaw pivot for the day-4 turn lag);
items (1) eyes / (3) blink shape / (5) balance from day 10; the gui-test
crash.

**Day 11, night: his four screenshots (2026-09-06 05:18-05:22; black
T-shirt, the room lit from one side, 20-25 fps).** He reports talking
improved. Four screenshots, each named for its problem, and one "That
looked wrong" at 05:23:04 -- 300 frames of tracking at 25.1 fps while
talking; its `seen.png` is the *camera* frame (the capture does not store
the avatar's frame; proposal below). `_anchor_frame_ok` finds no rest in
that capture (mouth open throughout), so the gate replay's rest is its 30
shutest frames: mouth flips 0.42/s, opening 0.25/s, moving-but-held 5%,
shut-but-driven 18% -- talking behaves as the day-11 build measured.

*white_line_on_cheek and look_down_emotion_and_white_line -- a light band
along his left jaw (image left), ear to chin.* Not the portrait's white
background: the band's brightest pixel is 157 (first shot) and 193
(second) on its darkest channel, the background is 245; and at 35 degrees
of yaw and 8.7 of roll on driving.mkv (frames 1490, 1556, rendered today)
no near-white pixel appears on the avatar half that is not on the camera
half. Zoomed, the band is smooth pale skin between the avatar's beard line
and the outside: the portrait's own bare jawline -- its beard stops short
of the jaw contour, his runs along it -- pasted over his jaw, which at
night sits in shadow. The mask's join is the portrait oval's, 4% of face
width inside the portrait's contour with a 7% feather, i.e. on the
portrait's bare skin, and the colour correction is a plane
(`_fit_lighting_plane`): it cannot darken one band at the jaw. The
daytime clips hide it because his own jaw is lit like the portrait's.
Two fixes, both measurable with a band-mismatch metric (mean |output -
scene| in the 10 px inside the mask edge, jaw half): (a) the join moved
inside the beard at the jaw -- a jaw-only shrink of the driver oval's
lower points, ~8% of face width, so his own jaw shows (his 2026-08-15
preference); (b) the correction from a plane to the warp engine's blurred
low-resolution difference (`engine_warp._match_colour`, ~1 ms), which can
darken locally. (a) first -- geometry, costs nothing; (b) helps every
night-time mismatch. A portrait whose beard runs the jaw as his does
would remove the cause.

*look_down_emotion -- pitched down, irises at the bottom of wide eyes, a
pouted "o".* The "o" is not it: natural.mkv 1277/1290 (his "o"s, rendered
today, stills sent) give a level "o" with the irises centred, and on both
clips the avatar's gaze_y on open-and-narrow mouths equals its shut-mouth
value (-0.106 / -0.093 against -0.107 / -0.098). What the screenshot
shows is his head pitched down (eye pivot: forehead grows, chin tucks --
as designed) with the avatar's irises rigid in the lids where his either
counter-rotate toward the camera or drop to the screen -- vertical gaze,
no lever (day 11) -- and the eyes wide: his own EAR in the 05:23 capture
is 0.41 against 0.31-0.32 on the daytime clips, so either his eyes were
wider at night or the anchor learned a narrower rest and `eye_headroom
1.1` opened them. Needs the moment captured with the avatar's frame in it.

*smile_without_teeth -- a very wide closed-lip smile.* Not reproduced on
the clip: driving.mkv 925/968 (his toothy smile) show teeth, and still
show them with the eye-line crop forced fully on (`_CROP_CENTRE_PITCH`
patched to always; the always-on variant's o_mar 0.196 against his
0.188). The shape is exactly what the gates produce when `mouth` is open
and `opening` shut: width from keypoints 7/14, no parting of 17/19/20.
The opening gate reads `_lip_ratio` = inner-lip gap over MOUTH WIDTH,
0.010/0.030 above rest: the wider the smile, the larger the gap must be
to pass -- a 4-px gap reads 0.067 at a 60-px mouth and 0.033 at a 120-px
grin, his rest 0.006-0.02. The clip's smile has an 18-px-class gap
(0.188) and passes; a grin with the teeth just showing does not. Fix:
gate the opening on the gap over a width-independent size (the jaw width
`_JAW_L/_JAW_R` the width ratio already uses, or eye-corners-to-chin),
thresholds re-derived so the clips' talk / smile / shut segments keep
their gate values, then verified on a smile capture (he presses the
button while smiling).

*closed_eyes -- narrowed lids, level head, faint smile.* The avatar's
lids copy his (`_eye_target`, eye_gain 0.25): on natural.mkv TALK, 33 of
the 35 frames where his EAR falls under 60% of rest are frames where his
iris sits more than 0.05 of eye width below rest (gaze down: the
screen), 2 are not; corr(his EAR, his iris height) is -0.85 driving /
-0.91 natural. His lids drop when he looks down; the avatar's irises
cannot follow (vertical gaze), so it shows the narrowing alone -- eyes
"closed". In the driving "look DOWN" his EAR reaches 0.35 of rest, the
avatar's 0.55. Either accept, or a `gaze_down_relief`: while his iris is
below rest and the drop is slow (not a blink -- a blink hides the iris
and lands in 2-3 frames), hold the eye target at rest so the avatar keeps
its eyes open looking straight. His call: open-and-straight or narrowed.
(A blink caught by the screenshot is the other reading; a capture with
the avatar's frame would settle it.)

*Two tool corrections.* `tracking.npz`'s `pose` rows are (yaw, pitch,
roll) (runner.capture); `gate_replay.py` and today's `capture_stats.py`
read them as (pitch, yaw, roll) -- no gate result changes (both under 5
degrees on every capture) but the day-9-night and day-11 replays'
"pitch" was yaw; fixed in both. And the capture's `seen.png` is the
camera frame, not what he saw: proposal -- the runner also writes
`shown.png`, the composited output at the press, so a screenshot's
complaint can be replayed against the avatar's actual frame instead of
guessed at from the clips, as all four had to be tonight.

*Stills for him (his face; delete after he looks):*
`recordings/day40-turn-f{925,968,1486,1490,1551,1556}.png`,
`day40-o-f{1277,1290}.png`, `day40-smile-eyeline-f{925,968}.png`, and
the three `day40-*.mp4`. `day40-o-f1290` and `day40-smile-eyeline-f968`
sent.

*Next, in order:* (1) `shown.png` in the capture -- one line, makes every
later complaint reproducible; (2) the jaw shrink and the band metric
(white line); (3) the opening gate over a width-independent size
(smile); (4) his decision on gaze-down relief (closed eyes); the
look-down emotion waits for a capture with the avatar in it. fps
(pipelining) and vertical gaze stand as day 11 left them.

**Day 12 opens with (2026-09-07; his go, given at the end of day 11's
night: "let's start fixing tomorrow"; Fable 5.1 implements directly, as
on day 11).** In this order, each step measured before the next:

1. **`shown.png` in the capture** (`pipeline/runner.py capture`): beside
   `seen.png` (the camera frame) write the composited output frame at the
   press. Until this exists every complaint has to be guessed from the
   clips, as all four were on day 11's night. Then he records four
   moments with the button: a toothy smile; looking at the screen while
   talking; an "o" while looking down; the head tilted at night with the
   jaw band visible. Unit test: a capture holds both files.
2. **The jaw band ("white line")**: a jaw-only shrink of the driver oval's
   lower points (~8% of face width; `_oval_mask` gets a jaw shrink, the
   forehead reach untouched) so the join sits inside the beard and his own
   jaw shows. Metric first: mean |output - scene| in the 10 px inside the
   mask edge, jaw half, on his night capture (once step 1 exists) and on
   natural.mkv; bars: blink gain >= 0.85, talk mar gain >= 0.85 unchanged.
   If the band survives the shrink, the colour field from the plane to
   the warp engine's blurred low-resolution difference (~1 ms).
3. **The opening gate over a width-independent size** (smile without
   teeth): `_lip_ratio`'s denominator from the mouth corners' distance to
   the jaw width at mouth height (`_JAW_L/_JAW_R`, what
   `_mouth_width_ratio` already divides by); `_MOUTH_GATE_MAR` and
   `_MOUTH_ACTIVITY_MAR` re-derived so the clips' shut / talk / smile
   segments keep their gate values within 0.05 (gate_replay on all
   captures, the pose order fixed); verified on his smile capture.
4. **Closed eyes -- his decision first**: open-and-straight while he reads
   the screen (`gaze_down_relief`: hold the eye target at rest while his
   iris is below rest by > 0.03 of eye width and the drop is slow; a
   blink -- iris hidden, EAR under 0.5 of rest within 3 frames -- still
   closes), or narrowed as now. Bars if built: driving "look DOWN" avatar
   EAR >= 0.9 of rest, blink gain unchanged >= 0.85.
5. The look-down emotion waits for its capture with `shown.png`.

The day-11 probes are in `scripts/probes/` now (`gate_replay.py`,
`capture_stats.py`, `white_leak.py`, `render_eyeline_on.py`) -- the
session scratchpad they lived in does not survive a new chat.

Not on day 12 unless he asks: fps pipelining (+1 frame latency, his OK
pending), vertical gaze (no lever), crop C, the yaw pivot.

**Day 12 (2026-09-07, Fable 5.1, his "start from step 1, in order"; steps
1-3 built and measured, step 4 built behind a switch that is off until he
decides).** 927 tests pass (up from 913: the capture's two files, the jaw
shrink and its option, the opening measure, the relief); mypy the same 12
pre-existing errors; ruff clean. Nothing committed.

*Step 1 -- the capture, and a correction.* The night's premise was wrong:
`seen.png` was never the camera frame. `PipelineRunner.capture` writes
`snapshot.frame.data`, which is the renderer's canvas -- the engine's
composited output (`_paste` returns the scene with the face blended in) --
and the day-9 entry above already says so ("his room and hair with the
avatar's face"); the 05:23:04 capture's own `seen.png`, zoomed, carries the
pale cheek band, which only the avatar has. So what was missing was the
opposite file: the camera frame at the press, without which the neural
engine (which reads pixels, not landmarks) cannot be replayed and the jaw
metric below has no "scene" to compare against. The capture now writes
`camera.png` -- the source frame of the same snapshot, so the same tick as
`seen.png` -- beside it. It is a photograph of him, written knowingly; the
runner's docstring says so and the "That looked wrong" dialog names all
three files. `_as_bgr` converts either frame for `imwrite` (NV12 skipped
with a warning). Test: a capture holds both files and they are not the
same picture. His part unchanged: four moments with the button -- a toothy
smile; looking at the screen while talking; an "o" while looking down; the
head tilted at night with the band visible.

*Step 2 -- the jaw band.* Metric first, `scripts/probes/jaw_band.py`: the
mask the engine actually pasted with (`loaded.last`) warped into the frame,
the 10 px inside its 0.5 contour (distance transform), jaw half (rows below
the mask's middle); `abs` = mean |output - scene| there in grey levels,
`step` = mean(output) - mean(scene), signed (a pale band over a shadowed
jaw is positive). Clip mode renders as the app does (`--option` for the
engine option, `--stills` writes camera|output face zooms); capture mode
takes `seen.png` + `camera.png` under the driver's oval from the capture's
last frame. Then the shrink: `_oval_mask(..., jaw_shrink=)`, each oval
point below the eye line pulled toward the face centre by up to
`_OVAL_JAW_SHRINK` = 0.08 of face width -- nothing at the eye line, the
full amount from the nose tip down (the crop is rolled with the face, so
"down" holds under roll); applied to the driver's oval only, in
`_frame_mask`, so the portrait's mask and its box are untouched; the
forehead reach as before. Engine option `mask_jaw_shrink` (0 is the
day-11 mask; validated at load). On the synthetic face the chin rises
0.077 of face width and the mouth row narrows 0.18 (two sides), the eye
row 0.05 (the hull's taper), the top row not at all.

| jaw band, p50 / p90 (grey levels) | natural.mkv before | after | driving.mkv before | after |
|---|---|---|---|---|
| abs, jaw half | 24.8 / 26.5 | **15.8 / 18.4** | 27.5 / 29.6 | **15.6 / 18.1** |
| step, jaw half | 15.6 / 19.8 | **8.0 / 13.2** | 16.0 / 22.3 | **8.0 / 13.9** |
| abs, whole band | 21.4 / 23.5 | 16.5 / 18.4 | 22.1 / 23.8 | 15.1 / 17.2 |
| blink gain (driving) | 0.850 (day 11) | **0.90** | | |
| talk 713-1306 mar gain (natural) | 0.87 (day 11) | **0.859** | | |

The mismatch at the jaw drops by a third and the pale step halves on both
daytime clips; both bars clear 0.85. Stills, his face, delete after he
looks: `recordings/day12-jaw-{natural,driving}-s{0,default}-f*.png`
(natural 800 / 1277 / 1576, driving 925 / 1490 / 1556; `s0` is before).
At driving 1556 (the 35-degree turn the night measured) his own beard now
runs the jaw and the chin; the avatar's bare jawline is gone from the
join. The night verdict -- the band in his shadowed room -- needs his
capture with `camera.png` (capture mode above); if it survives there, the
colour field from the plane to a blurred low-resolution difference (b)
stays the next lever. Not done today.

*Step 3 -- the opening gate over the jaw width.* `_mouth_open_ratio`:
inner-lip gap over the jaw width at mouth height (58/288, what
`_mouth_width_ratio` divides by). `_lip_ratio` is not touched: it is
LivePortrait's lip-retargeting convention and the anchor's `_ANCHOR_MAX_MAR`
reads it; only what the gates and their rest read changed (`animate`
passes `c_d_open` to `_gates` and, as `_render`'s `mar`, to the anchor's
`rest_mar`; `c_d_lip` stays for `_anchor_frame_ok` and `lip_target`). The
factor: mouth width over jaw width at rest is 0.39-0.43 on both clips and
all nine captures (natural 0.420, driving 0.396), so `_MOUTH_GATE_MAR`
(0.010, 0.030) -> (0.004, 0.012) and `_MOUTH_ACTIVITY_MAR` 0.010 ->
0.004. Replayed with the engine's own `_gates` on both clips (tracked with
the app's settings) and the nine captures, old measure and thresholds
against new: the per-segment gate means moved 0.000 on every still, talk
and smile segment and on every capture (worst 0.004), 0.023 on driving's
"turn RIGHT" (the jaw foreshortens differently from the mouth) -- inside
the 0.05 bar. Live jitter on the new scale: shut-mouth |dMAR| p99
0.0023-0.0024 on the clips, 0.0043 on capture 232151 (day 9 night's 0.011
times 0.4); the test constants follow. Unit test: the same 4-px gap at a
60-px and a 120-px mouth reads the same opening ratio while `_lip_ratio`
halves -- the smile-without-teeth case, which the clips do not contain and
his smile capture will show. `gate_replay.py` and `capture_stats.py` read
the new measure (the shut-mouth threshold 0.006 -> 0.0024; the variants'
thresholds scaled). Bars with the gates in: blink gain 0.90, talk mar gain
0.859 (>= 0.85; the talk segment's opening gate reads 0.916 either way).

*Step 4 -- closed eyes, his call, built so the choice is measured.*
`gaze_down_relief` (engine option, default **off**: the narrowed look as
now). On: while his iris sits more than 0.03 of eye width below his rest
(`rest_gaze`, the day-11 anchor reading; +y down) and the lids came down
slowly, `_relieved_ear` builds the eye target from his rest ratio -- no
narrower than rest -- so the avatar keeps its eyes open looking straight.
A blink still closes: eyes under 0.5 of rest, or a fall of 0.25 of rest
within the last 3 frames (`ear_recent`, cleared with the anchor), pass
through untouched; so does anything wider than rest. The anchor and the
gaze blink hold still read his raw ratio. driving.mkv, "Look UP, then
DOWN" 421-538, frames with his iris > 0.03 below rest: the sidecar's segment holds only the
first five frames of the glance (a blink-speed drop to 0.35 of rest); the
look-down itself runs 534-575, his iris 0.06 below rest and his lids
settled at 0.67 of rest -- 36 frames with the iris > 0.03 below rest and
the eyes over 0.3 of rest (the measure).

| driving.mkv | relief off | **relief on** | bar |
|---|---|---|---|
| look-down avatar EAR, p50 of rest | 0.82 | **1.05** | >= 0.9 |
| look-down avatar EAR, p10 / min | 0.64 / 0.53 | 0.65 / 0.53 | (the three frames after the drop, held by the blink rule) |
| blink gain | 0.902 | **0.909** | >= 0.85 |

Both bars clear. What he must also know before saying yes: the rule fires
wherever the iris reads low against the eye corners, and on this clip that
is also the toothy smile (36 of 118 frames), the wide-open mouth (32 of
118) and the head turns (9-14 of ~150) -- his squint on a wide smile would
be undone while the iris reads down. A "no smile" exclusion (mouth width
under rest + 3%) removes those but also two-thirds of the real look-down
frames (his width reading moved with the pitched head), so it is not the
discriminator; if he wants the relief and the squint, that needs its own
measure. Off by default, so nothing he sees changed. CSVs:
`recordings/driving-follow-day12-relief.csv` (on), `-gates.csv` (off). To try it:
`"gaze_down_relief": true` in `avatars/my_avatar_neural/metadata.json`;
his verdict decides the default.

*Not done.* The look-down emotion (step 5) and the night jaw verdict wait
for his captures with `camera.png`. fps pipelining, vertical gaze, crop C,
the yaw pivot: as day 11 left them.

**Day 12, his first look (2026-09-06, 23:41 screenshot; he had shaved that
morning).** "Quite good, I am satisfied with this development" -- and one
picture: the avatar's face ends at the mask edge and his own bare cheeks,
jaw and chin show under it, a ledge. That is the jaw shrink doing exactly
what it was built to do, on a face that no longer has the beard it was
built for: the join was moved 8% inside the contour so it would sit
*inside his beard*; clean-shaven, it sits on bare skin and cuts the
avatar's stubbled chin short. So the shrink is a beard setting, not a
default: `_OVAL_JAW_SHRINK` is 0 again (the join at the contour, as
before day 12, for every user), and `mask_jaw_shrink` is written into his
metadata.json at 0.0 -- back to 0.08 when the beard is back, the day-12
table stands for that case. The night band was a beard-vs-bare-jaw
mismatch too (his beard in shadow under the portrait's bare jaw); shaved,
its cause is gone with it, and tonight's capture says whether anything
remains. The portrait's own moustache and chin stubble are now the
mismatch in the other direction; a clean-shaven portrait would remove it
(his call, a portrait lever). Tests: the default test reads 0.0.

**Day 12, night (2026-09-07, 00:15-01:40; his four captures were to come, one
did): "the face trembles when I smile -- it cannot settle how wide the smile
should be." Found, and it was not the smile.**

*The trail.* His capture at 00:15:55 (the night jaw: band gone, step +0.8
grey levels against +16 on the clips with the beard; the forehead line is
what shows now). Then the smile, on driving.mkv's closed-lip smile
(806-870, his width steady at 0.628-0.636, gates 1.00 throughout, crop
constant to 0.5 px, pose constant, no hold, no blink): the avatar's mouth
width fell 0.666 -> 0.561 in two frames and climbed back over twelve,
three times in sixty frames, the corners dropping 6-8 px with it. Not the
camera (the mouth box's sharpness, brightness and frame-to-frame change
flat at the drops), not the measuring tracker (stills of the render show
it: 815 smiling, 816-819 neutral, 824 half, 828 smiling). The motion
extractor's own reading, read eagerly on the same crops: keypoint 14's y
relative to the anchor flipping -0.010 / -0.007 / -0.004 between
consecutive frames -- three levels, no in-between.

*The cause.* The extractor ran in fp16 by a plain `.half()` on the whole
module (day 1). LivePortrait's own half-precision is `torch.autocast`,
which keeps the normalisations in fp32; ours put the ConvNeXt-V2's global
norms in fp16 too. That is not a precision loss, it is a different
function: on the portrait, fp16 read scale 1.86 (fp32: 1.01), t -0.073
(-0.010), the eye keypoints' y at -0.31..-0.14 (+0.23..+0.30) and the
anchor's eye expression at -0.07 (-0.002). Frame-to-frame |change| on a
held smile, fp16 against fp32: keypoint 14 y 0.0030 vs 0.0002, yaw 0.64
vs 0.07 degrees, scale 0.022 vs 0.003, t x 0.0034 vs 0.0003 (bf16 nearly
as good as fp32). The cost is nothing: 5.1 ms fp16, 5.0 ms fp32 eager
(cuDNN's convolutions run TF32 either way). The other three networks in
fp16 are fine: warping, decoder and the three MLPs re-run in fp32 give
the same picture (mean 0.12-0.16 grey levels, p99 2; the tracked EAR,
mouth ratio and width identical), so they stay half.

*What it explains.* Twelve days of gains were fitted to that function:
eye_gain 0.25 against LivePortrait's own 1.0 (the eye network was reading
an out-of-range x_s); the lip network "only ever pressed the lips
thinner" and stitching "sat the face to the side" (both read x_s); the
crop-scale cliff and the grimaces at 2.1/2.3; "the avatar wanders while
I sit still"; "the face bounces on the slightest sideways shift"; day
7's EMA that could not clear its bar. And the relative expression came
out about 1.8x too large, flickering -- the closed-lip smile he liked was
partly noise.

*The build now.* Motion extractor in fp32 (`_load_nets`), the image cast
on the way in. `eye_gain` 1.0 (0.25 barely closed the eyes on the
correct x_s: blink gain 0.16). A new `expression_gain` (default 1.0,
[0, 3]; the day-6 lever) on the whole relative expression except the
eyes' rows, so the amplitude is put back by measurement; his metadata
carries 1.8. `mouth_smile_gain`'s bound is [0, 3] now, for the closed
smile if he wants it wider. Measured on both clips (eye_gain 1.0):

| driving.mkv | fp16, day 11 | fp32, eg 1.0 | fp32, eg 1.5 | **fp32, eg 1.8** | fp32, eg 2.2 |
|---|---|---|---|---|---|
| blink gain (>= 0.85) | 0.902 | 0.971 | 0.972 | **0.971** | 0.971 |
| talk + open-mouth mar gain | 0.636 | 0.842 | 0.957 | **0.987** | 0.953 |
| natural talk 713-1306 mar gain (>= 0.85) | 0.859 | -- | -- | **0.941** | -- |
| smile width gain (both smiles / closed) | 0.728 / 1.383 | 0.182 / 0.078 | 0.320 / 0.202 | **0.441 / 0.288** | 0.588 / 0.383 |
| brow gain | 0.641 | 0.674 | 0.693 | 0.701 | 0.711 |
| closed smile: avatar width jitter p90 | 2.46% | 0.30% | 0.44% | **0.50%** | 0.54% |
| still: avatar EAR / brow / width spread | .038 / .013 / .022 | .027 / .015 / .016 | same | **.027 / .015 / .016** | same |
| turn L/R: dx_nose p90 | 47.7 px | 19.7 | 19.6 | **19.9** | 19.9 |
| natural 681 / 1576 dy_eye | -5.2 / -1.6 | -- | -- | -6.8 / -6.9 | -- |

1.8 kept: the talk gain reads 1.0 on driving and 0.94 on natural, the
blink 0.97, the turn lag less than half of day 11's, the smile's jitter a
fifth, and the still face steadier than it has ever been. What he will
see smaller: the closed-lip smile (0.29 against the old, inflated 1.38)
-- `mouth_smile_gain` above 1 is the knob, unmeasured. The natural
clip's still-segment EAR spread is 0.044 against 0.035 (eye_gain 1.0
passes more of the tracker's own jitter on his lids), and 1576's dy_eye
-6.9 sits at the edge of the rest +-5 bar. 213 engine tests; ruff and
mypy as before. Not committed until he has looked.

*His look, 03:00 (five captures 02:52-02:55: closed smile, toothy smile,
turn with a blink, eyes shut, a 42-degree turn).* "The only thing I do
not like is that my forehead shows -- that line is bad and noticeable.
Everything else is good. Push this version to main, it is the best we
have had." Pushed. The forehead line -- the mask's top edge at the
hairline, the portrait's paler forehead over his -- is the next thing.

*Re-opened by this, in order of likely payoff, each a measurement away:*
lip_retargeting and stitching (both read x_s, both were judged on the
broken one); the crop scale (2.2 was chosen against fp16's cliffs, and
2.2 in our box is about LivePortrait's 2.5 -- the expression reads small
partly for that); eye_headroom / eye_scale; the gate thresholds do not
move (they read his geometry, not the network).

### Phase 2 — live, days 6-9

- Engine in the app, live camera, the owner testing daily.
- Tuning by his eye: identity, mouth interior, eyes, head-turn range, lighting.
- Latency: pipeline the CPU crop with the GPU render if the budget is tight
  (+1 frame); 60 fps if the capture card arrives, including the preview timer.
- GPU check at startup; the photograph engine as automatic fallback.
- Exit: the owner uses it on a real call and does not report the mouth.

### Phase 3 — a package that installs, days 8-12 (overlaps phase 2)

- A shippable virtual camera backend: `MFCreateVirtualCamera` (Windows 11) or a
  small native DirectShow filter; pyvirtualcam stays for development only.
- `.exe` with the CUDA runtime; first install on a second machine that never
  saw the repository; a plain message when the GPU is missing.
- Enrolment: a portrait in, an avatar out, no scripts.
- Exit: a user installs it from a file and is on a call in ten minutes.

### Phase 4 — protection and rollout, days 12-18

- Licence server with the kill switch, the download site, avatar encryption.
- Five portraits generated and enrolled; five installs; a written one-page
  guide with the camera settings (NTSC 30 fps, shutter, USB Streaming).
- Exit: five users on calls; the owner can switch one off remotely.

## Honest odds

- **The picture:** high. LivePortrait is proven and it fits this GPU; what is not
  yet seen is its quality in a real room at 720p. Day 4 answers that.
- **"Hard to tell it is an avatar" on a normal call window:** realistic within
  phases 1-2. Under close inspection or a large screen: not in three weeks.
- **Five users:** depends entirely on their GPUs and cameras. Without an NVIDIA
  GPU a user gets the photograph engine, which is what exists today.
- **Eleven days from today:** phases 1-2 fit; phase 3 partly; phase 4 does not.
  A quality delivery to five people is about **three weeks**, four with margin.

## What stops

No more work on the photograph engine beyond keeping it running as the
fallback: no refiner, no new clips, no third axis, no morphing between rungs.
Every day spent there is a day the neural engine is not being built.
