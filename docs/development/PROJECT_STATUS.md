# sha256 — Current Status

**Living document.** Update it at the end of each working day. For the frozen
record of the foundation milestone see `FOUNDATION_STATUS.md`.

**Last updated:** 2026-08-14 · **End of day 7 of a 14-day plan**

---

## Tomorrow — day 8

### Waiting on the owner. Nothing below starts without these.

| # | | Why it blocks |
|---|---|---|
| 1 | **Choose a video generator.** Sora is gone — the app shut down 2026-04-26 and the API goes on 2026-09-24. | The whole video→frames plan depends on which tool, and its licence and watermark have to be checked *before* time is spent generating. A visible watermark on the face makes the frames useless, and Sora's was deliberately unremovable. Candidates still live: Runway, Kling, Luma, Pika, Veo, Hailuo. |
| 2 | **The scope decision.** Promised for day 7 and now overdue. | Licence server, download site, avatar encryption, `.exe` packaging and a shippable virtual camera total 6–7 days. Seven remain, before any quality work. Something has to come off, and that is not an engineering call. |
| 3 | Regenerate the two **far-turn** photographs at ~40°. | The current pair measures 17–20° and is swapped relative to its filenames. |
| 4 | An avatar whose **hair** resembles the owner's. | The mask stops at the hairline, so the hair on screen is always theirs. Matching it is free; replacing it needs segmentation the frame budget cannot afford. |

### Queued here, in order

1. **The wedge above a blinking eye** (backlog 15). Diagnosed: six landmarks per
   eye leave it anchored to the cheek by long triangles that sweep as the lid
   closes. The fix is the full eyelid ring — sixteen points per eye — and it
   invalidates ten tests that encode the six-point layout. A morning's work with
   its own verification, not a tweak.
2. **`INTER_AREA` before the mesh warp** (backlog 9). The source is reduced about
   3× onto the face and bilinear discards most of it. The remaining softness.
3. **If a video arrives:** extract frames, run every one through
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
| 8–10 | Quality iteration | |
| 11–12 | UI, avatar gallery, licence server | |
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
  where the expression changes every single frame. 97 mesh points, 156
  triangles, at a face the size one occupies at desk distance.
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
