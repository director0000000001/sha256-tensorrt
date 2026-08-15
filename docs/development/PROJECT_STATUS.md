# sha256 — Current Status

**Living document.** Update it at the end of each working day. For the frozen
record of the foundation milestone see `FOUNDATION_STATUS.md`.

**Last updated:** 2026-08-15 · **Day 8 — and the plan changed today**

---

## The plan is now one month, and the target is quality

**Decided by the owner, 2026-08-15, after running the application themselves.**
Their words: the eyes are wrong far more often than they are right, a slight
turn nearly loses the face, and the quality only holds within five to ten
degrees of straight ahead. They also observed, correctly, that day 1 to day 2
improved a great deal and day 3 barely moved.

That observation is the reason for the change. The first two days removed
outright defects; what is left is the method's ceiling. Every expression axis
examined since has turned out not to survive being blended: the four gaze
photographs were dropped, the squint photograph was dropped, and two of
MediaPipe's eye blendshapes were found to measure something other than their
names. That is a pattern, not a run of bad luck.

**So: four weeks, and a neural refiner is the plan's centre.** Not a face
swapper — the licence-clean, entirely-ours version: a small network trained on
the owner's own generated clips to correct what the warp gets wrong. Its input
is an already-posed, already-correct face needing artefacts removed, which is a
far smaller problem than synthesising one, and B001 measured that shape at
**1.83 ms fp32 / 1.33 ms fp16** — it fits.

| Week | |
|---|---|
| **1** | The broken turn sources, the brows, and 20 fps. A repeatable quality measure. The training-pair generator. |
| **2** | First refiner. First honest comparison. |
| **3** | Iterate: architecture, data, temporal stability. |
| **4** | Integrate, optimise, `.exe` and the kill switch, deliver. |

**`.exe` at the end, by the owner's choice** — everything gets its proper time
and nothing is rushed.

**The odds, given plainly rather than promised:** roughly one in five that this
reaches "high level"; roughly two in three that it ends up markedly better than
today; about one in four that the network does not earn its place and the
improved warp is what ships. The largest risk is the gap between studio clips
and a real room, and it is smallest here because the network never sees the
camera — only the avatar, which is already normalised.

**A commitment made with it:** a fixed clip, fixed frames, one number, reported
before and after every change. Several times today the owner was told a defect
was found and still saw the same thing. A measure they can watch is worth more
than another assurance.

---

## Day 8

**Done today**, in order:

1. **The wedge above a blinking eye** (backlog 15). The eyelid ring was only
   half the cause; the other half is that MediaPipe reports the whole iris
   circle including the part hidden behind the lid, so the mesh described a
   thirty-pixel iris inside a thirteen-pixel eye and the triangles between them
   were inside-out. Costs 1.18 ms of mean frame time.
2. **The extractor was throwing away the clips it was built for.** A head-turn
   clip reported 0 of 22 roles while every frame it needed sat in the file: no
   resting face to measure against, and a blur floor taken from the still
   frontal frames. Now 7 of 22 from that clip, and the ladder below.
3. **The eyes never actually closed** (the owner's standing complaint). A squint
   took half of every blink, and a *held* closure read 0.14 on `eyeBlink` — so
   two seconds of shut eyes rendered as a squint. Closure is measured on the
   eyelids now, which only became possible once the lid ring existed.
4. **A turn is chosen rather than averaged** (backlog 17). Thirty photographs
   at measured angles from −47° to +56°, one per three degrees; the engine takes
   the two that bracket the head and clamps past the ends. Seven times the turn
   coverage for 0.30 ms.

**Still open and visible:** the far eye at large yaw, which is item 13's limit
(an iris dragged through the mesh rather than translated as a rigid disc).

### Waiting on the owner. Nothing below starts without these.

| # | | Why it blocks |
|---|---|---|
| 1 | **Choose a video generator.** Sora is gone — the app shut down 2026-04-26 and the API goes on 2026-09-24. | See the checklist below. Candidates still live: Runway, Kling, Luma, Pika, Veo, Hailuo. |
| 2 | ~~The scope decision~~ — **made 2026-08-15, see below.** | Settled. |
| 3 | ~~Regenerate the two far-turn photographs at ~40°~~ — **done 2026-08-15.** | The owner generated a head-turn clip; 30 frames from −47° to +56° replaced the pair entirely. See backlog 17. |
| 4 | An avatar whose **hair** resembles the owner's. | The mask stops at the hairline, so the hair on screen is always theirs. Matching it is free; replacing it needs segmentation the frame budget cannot afford. |

### The scope decision — made 2026-08-15

The owner's answer, and it removes most of what was threatening the schedule:

- **No licence system and no payments.** The application is free to use. There
  is nothing to check, no accounts, no entitlements.
- **A kill switch is still wanted** — the owner wants to be able to turn the
  application off when they choose.
- **The download site they will build themselves**, separately. Not our work.
- **Avatar encryption: dropped.**

What that leaves is far smaller than the two days a licence server was costed
at. A kill switch with no licensing behind it is a static file on any host —
GitHub raw, a plain web page, anything — that the application reads and obeys.
No server to write, nothing to keep running, nothing to administer.

Three things have to be right, and they are design decisions rather than code:

1. **It must fail open.** If the check cannot be reached, the application keeps
   working. A network blip that ends a live call is a worse failure than a
   delayed shutdown, and the owner can wait an hour for the switch to take.
2. **It must not block the UI.** The check belongs off the frame path entirely,
   at startup and then occasionally, never in the loop.
3. **It is a courtesy switch, not enforcement.** Anyone technical can block the
   address or patch the binary. It stops an ordinary user; it does not stop a
   determined one, and it should not be sold to the client as though it does.

Remaining before delivery: the kill switch, `.exe` packaging, and quality.

### Buying a video generator — what to check, in this order

The owner is paying for one, so this is what decides whether the money buys
anything usable. Ordered by what kills the plan outright.

1. **Image-to-video, starting from `portrait.png`.** The single most important
   one and the easiest to overlook. A tool that only generates from a text
   prompt produces a *different person*, and every photograph, the manifest and
   the tuning already built around this identity become worthless. The avatar
   must be driven from the portrait that already exists.
2. **No watermark at the tier being bought.** Not "removable" — absent. A mark
   across the face makes a frame useless as a source, and removing one is
   normally a licence breach in exactly the commercial case this is for.
3. **Commercial use, output owned by the user.** The shape to look for is
   OpenAI's: *"you own the Output; we assign to you all our right, title, and
   interest, if any"*.
4. **1080p or better**, and the face filling much of the frame. Frames are cut
   from this and the current portrait is 1024×1024 — a small face in a 720p clip
   is a downgrade, not an upgrade.
5. **Length and identity stability.** One long clip beats several short ones:
   every restart is a chance for the face to drift. Whatever arrives, every
   frame goes through `scripts/check_avatar_photos.py` before it is used.

### Queued here, in order

1. ~~The wedge above a blinking eye~~ (backlog 15) — **fixed 2026-08-15.** The
   eyelid ring was half of it; the other half was that MediaPipe reports the
   whole iris circle including the part behind the lid, so on a blink the mesh
   claimed a 30-pixel iris inside a 13-pixel eye. Rendered before and after on
   the recorded clip. Costs 1.18 ms of the ~2.5 ms headroom.
2. ~~`INTER_AREA` before the mesh warp~~ (backlog 9) — **done**, and it moved
   the rendered number by one part in three hundred. See backlog 16.
3. **A 4K source renders *softer* than a small one** (backlog 16). Open,
   measured, not diagnosed. Until it is, more 4K clips buy coverage but not
   sharpness.
4. **If a video arrives:** extract frames, run every one through
   `scripts/check_avatar_photos.py`, and then the real prize — switch the engine
   from *blending* sources to *selecting* the nearest. Dense coverage is what
   makes that possible, and it removes cross-fade ghosting by construction
   rather than by clamping.

### Housekeeping

`recordings/` holds the owner's real face. Gitignored, local only, **delete it
when the quality work that needs it is done.**

---

## Where the project is

| Day | Work | State |
|---|---|---|
| 1 | GPU and toolchain verification | ✅ Done — B001 |
| 2 | Real camera capture | ✅ Done — B002 |
| 3 | Face tracking (MediaPipe) | ✅ Done — B003 |
| 4 | Avatar engine — first moving avatar | ✅ Done — landmark warp, 22 source roles |
| 5 | Virtual camera | ✅ Done — B004. Visible in Meet, Zoom, Discord, OBS |
| 6 | Quality: neutral baseline, eyes, teeth | ✅ Done — backlog 8, 9, 10 |
| **7** | **Neural engine evaluated and closed; gaze as geometry; replay harness** | **✅ Done — B005** |
| **8** | **Quality: the wedge above a blinking eye (backlog 15)** | **✅ Done** |
| 9–10 | Quality iteration | |
| 11–12 | UI, avatar gallery, kill switch, `.exe` | |
| 13–14 | Bugs, buffer | |

**Day 7 in one line:** LivePortrait was measured and rejected on speed (B005),
the licence question that had blocked it since the evaluation turned out to be
answered in our favour, gaze became geometry instead of photographs, and the
recording/replay pair now lets a quality idea be tested in a command instead of
an hour of the owner's time.

**The full set arrived on day 5.** All 22 photograph roles now have a file and
20 are wired into the manifest. Two are held back: the far-turn pair measured
17–20° of yaw, which the ordinary turn photographs already cover, and they are
swapped relative to their filenames. They need regenerating at roughly 40°.

**Day 6 — three quality defects found and fixed**, all of them the same mistake:
a signal compared against a fixed threshold when it should have been compared
against *this* face, or *this* photograph, or confined to *its own* region.

1. **A motionless face scored eleven simultaneous expressions**, because
   expression was measured absolutely and opposing photographs were averaged
   together. Now **0 sources at rest, byte-identical to the portrait**.
2. **A shut eyelid was ghosted over an open eye on every frame** — `eyeBlink`
   was the one signal still measured absolutely, and a relaxed eye reads 0.26
   against a 0.20 threshold. That was the owner's "two sets of eyes".
3. **Every photograph repainted the whole face.** Each is now confined to its
   own region. Under expression, sharpness went from **20 to 52** on the same
   weights; the base portrait measures 62.

Also fixed: the teeth stopped appearing because the replacement open-mouth
photograph is a yawn (lip gap 0.229 against 0.095), and the opening was being
scored against it. See `AVATAR_QUALITY_BACKLOG.md` items 8 and 9.

**Not yet re-measured end to end:** the camera dropped off the bus during the
final run (`no cameras detected`). The engine numbers above are from the
avatar's own photographs and do not need it; the pipeline figure does.

---

## Verified facts — do not re-derive these

**Hardware** (`scripts/check_gpu.py`)
- RTX 5070 Ti, 16303 MiB, compute capability 12.0 (`sm_120`), driver 610.62
- i7-14700KF, 32 GB, Windows 11

**Toolchain** (B001)
- `torch 2.11.0+cu128` includes `sm_120` kernels and a real kernel launch
  succeeds. The Blackwell support problem reported through 2025 is resolved.
- `onnxruntime-gpu 1.28.0` exposes **both** CUDA and TensorRT providers.
- A portrait-shaped conv stack at 256×256 runs in 1.83 ms fp32 / 1.33 ms fp16.

**Camera** (B002)
- Sony ZV-E10 over USB Streaming: **1280×720 @ 29.77 fps**, median interval
  33.19 ms, jitter 4.4 ms.
- **DirectShow is required.** Media Foundation cannot open this device.
- **NTSC/PAL gates the frame rate.** PAL offers only 25 fps; NTSC gives 30.
  Global camera setting, separate from the recording format.
- 720p is a limit of the camera's USB Streaming, not the cable. Accepted —
  the engine works at 256×256 internally.

**Face tracking** (B003)
- MediaPipe Face Landmarker `1.0.0`: **6.30 ms mean / 6.93 ms p95** per frame
  over 1747 frames, 100% detection, 478 landmarks + 52 blendshapes.
- **No GPU delegate exists on the Windows Python wheel** — it is compiled out.
  Tracking is CPU/XNNPACK, which leaves the GPU entirely free for the engine.
- MediaPipe emits **no confidence score**. Detection is binary.
- Colour conversion must use OpenCV, not numpy: 0.45 ms vs 2.90 ms per frame.
- Head-pose signs are verified, not assumed. Two of three were wrong initially.
- **Full pipeline with the real tracker: 29.50 fps sustained**, tracking 6.37 ms
  inside the application versus 6.30 ms standalone — integration costs nothing
  measurable.

**Avatar engine** (day 4)
- **Landmark-warp engine**: the portrait is deformed per frame onto the tracked
  face. No neural network, no model weights, **no licensing risk**.
- **13.8 ms mean / 17.7 ms p95** in ordinary use; 15.2 / 19.2 in the worst case
  where the expression changes every single frame — six sources, and 97 mesh
  points as the mesh then stood. It now carries **139 points and 229 triangles**
  (the full lip contours, both irises, and since day 8 the full eyelid rings),
  and with the owner's 20 sources costs **20.42 ms mean / 22.51 ms p95** over
  1817 frames of the recorded clip.
- First version cost 42 ms at p95 — over budget. The fix was one warp per
  triangle instead of two, and working inside the face's bounding box. A second
  round cached the source blend against weights quantised to 0.02 and painted
  the mouth interior inside the mouth's own box rather than the whole face's.
- **Multi-source: 22 photograph roles**, selected and blended per frame from
  blendshapes the tracker already measures and the head pose B003 verified. No
  thresholds, no state machine; every role is optional. Six supplied so far.
- **Seam fixed** by low-frequency relighting rather than Poisson blending, which
  measured 21–29 ms and did not fit. The forehead is now covered.
- Not photorealistic synthesis. LivePortrait remains the quality target and is
  still blocked on the weight-licence question.
- Five quality problems were observed on real use and recorded with causes in
  `AVATAR_QUALITY_BACKLOG.md`. One is fixed, three became supply problems that
  the 22-role set answers, one is engine-side and done. **A periodic flicker had
  three causes addressed on day 4; the owner is verifying whether a fourth
  remains.**

**Virtual camera** (day 5, B004)
- `pyvirtualcam 0.15.0` behind `VirtualCameraSink`, writing to the **OBS Virtual
  Camera** device. OBS must be *installed*; it does not need to be running.
- **1.44 ms mean / 1.54 ms p95** per 720p BGR frame. RGB placeholder frames cost
  0.65 ms more because they are converted first.
- Full pipeline sustained **29.65 fps** with publishing on, 593 frames, **zero
  dropped by the sink** — measured without an avatar loaded, so it does not yet
  include the warp engine's 13.8 ms.
- **GPL-2.0, verified from the installed package's own metadata. It cannot
  ship** — copyleft would oblige giving the client source, and source is all
  anyone needs to remove the licence server's kill switch. The committed default
  stays `null`; this machine selects it in gitignored `configs/local.toml`.
- Publishing is a **checkbox in the window**, off until asked for, and it can be
  switched during a call.

**Budget** — every line from one run, 596 frames, 20 sources (B004)
```
33.6 ms  frame interval the camera actually delivers (29.77 fps)
 6.9 ms  face tracking (p95 8.1), CPU
22.1 ms  avatar animation (p95 25.2), CPU  <- 6.8 of it is the source blend
 0.3 ms  rendering (p95 0.4)
 1.8 ms  virtual camera output (p95 2.7)
-------
31.1 ms  spent, ~2.5 ms headroom
```

**The blend cache stopped working.** It is keyed on weights quantised to 0.02;
with six sources it almost always hit, with 12.1 active per frame it hits 2.3%
of the time and the blend is rebuilt on 97.7% of frames. Capping the source
count was measured and rejected — the weight beyond the top six averages 0.59
of the total, so cheap caps change the picture and safe caps save little. The
lever is the cache and the per-pass cost. **First task for day 6.**

---

## Not yet measured

- **Camera and USB latency.** Everything upstream of `read()` returning —
  sensor, in-camera processing, USB transfer, driver buffering. Needs an
  external reference (film a millisecond timer, compare).
- **The virtual camera with a consumer attached.** Every B004 figure is sha256
  writing into the queue with nothing reading from it. Worth re-running with
  Meet open.
- **Tracker robustness** in poor light, motion blur, or occlusion (scenario S7).
  100% detection so far is one cooperative subject in a well-lit room.
- Sustained behaviour beyond one minute.

---

## Environment

Already installed in `.venv` beyond the declared dependencies:

```
torch 2.11.0+cu128 · torchvision · onnxruntime-gpu 1.28.0
opencv-python-headless 5.0.0.93
mediapipe 1.0.0 (installed --no-deps) + absl-py · certifi · matplotlib
pyvirtualcam 0.15.0          <- GPL-2.0, development only, never distributed
```

`pyvirtualcam` also needs **OBS Studio installed** for its device to exist. It
is present on this machine and the DirectShow filter is registered.

`opencv-python-headless` is deliberate — the normal `opencv-python` wheel
bundles its own Qt and conflicts with PySide6.

**MediaPipe must be installed with `--no-deps`.** It declares a hard dependency
on `opencv-contrib-python`, which would replace the headless build and
reintroduce exactly that Qt conflict. Its three genuinely required imports are
installed separately:

```powershell
uv pip install mediapipe --no-deps
uv pip install absl-py certifi matplotlib
python scripts/fetch_face_landmarker.py    # model bundle, ~3.6 MiB, not in Git
```

`configs/local.toml` (gitignored) selects the real camera on this machine.

---

## Working agreements

- **Verify, do not assume.** Every claim in this repository traces to a
  measurement or a primary source. Package availability is checked with
  `uv pip install --dry-run`; licences are read from the project's own files.
- **No performance number without its conditions** — hardware, sample count,
  percentiles. See `docs/benchmarks/PERFORMANCE_PLAN.md`.
- **Benchmarks are numbered** `B00N-topic.md` in `docs/benchmarks/`.
- Run after every change: `pytest` · `ruff check` · `ruff format` · `mypy`.
  All four are currently clean; 368 tests pass.
- Commit messages explain *why*. Push to `origin main` (private repo).

---

## Open decisions

| Decision | State |
|---|---|
| **Neural engine (LivePortrait)** | ❌ **Closed, day 6 — B005.** Licence is fine; the frame rate is not. 38.4 ms model-only against a 33.3 ms budget |
| Project licence | `LICENSE` is a restrictive placeholder. **Not GPL** — settled by the kill-switch requirement, see ADR-0005 |
| Virtual camera — development | ✅ **Settled** — pyvirtualcam, day 5, B004 |
| Virtual camera — shipping | ❌ **Open, due at the day 7 review.** Needs a native component; pyvirtualcam cannot ship |
| Is Windows 10 in scope? | Open. `MFCreateVirtualCamera` is Windows 11 only, and it is the licence-clean option |
| LivePortrait weight licence | ✅ **Resolved, day 6** — the model card declares `license: mit`. Commercial use permitted; a written confirmation request is still open upstream |
| InsightFace replacement | ✅ **Moot** — never downloaded. MediaPipe already does that job, Apache-2.0 |

---

## Scope watch

The owner has added, beyond the original pipeline: a licence server with a
kill switch, a download website behind an unguessable path, avatar encryption,
and `.exe` packaging. Each is small; together they exceed 14 days.

**A scope review was promised for day 7.** Staged payment was recommended as
the primary protection, with the kill switch as backup.
