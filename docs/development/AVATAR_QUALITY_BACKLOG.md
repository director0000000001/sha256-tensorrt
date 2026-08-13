# sha256 — Current Status

**Living document.** Update it at the end of each working day. For the frozen
record of the foundation milestone see `FOUNDATION_STATUS.md`.

**Last updated:** 2026-08-13 · **Day 3 of a 14-day plan**

---

## Where the project is

| Day | Work | State |
|---|---|---|
| 1 | GPU and toolchain verification | ✅ Done — B001 |
| 2 | Real camera capture | ✅ Done — B002 |
| 3 | Face tracking (MediaPipe) | ✅ Done — B003 |
| 4 | Avatar engine — first moving avatar | ✅ Done — landmark warp |
| **5** | **Virtual camera, or avatar quality** | **⬅️ owner's call — see the backlog** |
| 6–10 | Quality iteration | |
| 11–12 | UI, avatar gallery, licence server | |
| 13–14 | Bugs, buffer | |

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
- 11.5 ms mean / 15.5 ms p95 per animated frame; 97 mesh points, 156 triangles.
- Full pipeline holds **29.6 fps** with the avatar composited.
- First version cost 42 ms at p95 — over budget. The fix was one warp per
  triangle instead of two, and working inside the face's bounding box.
- Not photorealistic synthesis. LivePortrait remains the quality target and is
  still blocked on the weight-licence question.
- **Multi-source avatars**: five extra photographs of one identity, aligned at
  load and blended per frame from blendshapes and yaw. 14.85 ms mean / 17.94 ms
  p95. Fixes teeth on a smile, improves blinking and head rotation.
- **Seam fixed** by low-frequency relighting rather than Poisson blending, which
  measured 21-29 ms and did not fit. The forehead is now covered.
- **Five quality problems observed on real use** — forehead seam, no teeth when
  smiling, eyes do not close, eyes grow when opened wide, and rotation past
  ~15° losing alignment. Recorded with causes and candidate fixes in
  `AVATAR_QUALITY_BACKLOG.md`. Three of the five come down to one limit: the
  engine can only show what the source photograph contains.

**Budget**
```
33.3 ms  frame budget at 30 fps
 6.3 ms  face tracking (p95 6.9 ms), CPU
11.5 ms  avatar warp (p95 15.5 ms), CPU
 0.3 ms  rendering
-------
~15 ms   headroom
```

---

## Not yet measured

- **Camera and USB latency.** Everything upstream of `read()` returning —
  sensor, in-camera processing, USB transfer, driver buffering. Needs an
  external reference (film a millisecond timer, compare).
- Anything about avatar quality. No real engine exists yet.
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
```

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
  All four are currently clean; 309 tests pass.
- Commit messages explain *why*. Push to `origin main` (private repo).

---

## Open decisions

| Decision | State |
|---|---|
| Project licence | `LICENSE` is a restrictive placeholder |
| Virtual camera approach | ADR-0005 — owner chose a **licence server with instant on/off** over the offline option |
| LivePortrait weight licence | **UNCLEAR** — unresolved, blocks commercial use |
| InsightFace replacement | Required — its weights are non-commercial research only |

---

## Scope watch

The owner has added, beyond the original pipeline: a licence server with a
kill switch, a download website behind an unguessable path, avatar encryption,
and `.exe` packaging. Each is small; together they exceed 14 days.

**A scope review was promised for day 7.** Staged payment was recommended as
the primary protection, with the kill switch as backup.
