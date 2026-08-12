# sha256 — Current Status

**Living document.** Update it at the end of each working day. For the frozen
record of the foundation milestone see `FOUNDATION_STATUS.md`.

**Last updated:** 2026-08-13 · **Day 2 of a 14-day plan**

---

## Where the project is

| Day | Work | State |
|---|---|---|
| 1 | GPU and toolchain verification | ✅ Done — B001 |
| 2 | Real camera capture | ✅ Done — B002 |
| **3** | **Face tracking (MediaPipe)** | **⬅️ next** |
| 4 | Tracking → avatar engine, first moving avatar | |
| 5 | Virtual camera | |
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

**Budget**
```
33.3 ms  frame budget at 30 fps
 2.0 ms  spent by the pipeline machinery (placeholders)
-------
~31 ms   available for MediaPipe + avatar engine
```

---

## Not yet measured

- **Camera and USB latency.** Everything upstream of `read()` returning —
  sensor, in-camera processing, USB transfer, driver buffering. Needs an
  external reference (film a millisecond timer, compare).
- Anything about avatar quality. No real tracker or engine exists yet.
- Sustained behaviour beyond a few seconds.

---

## Environment

Already installed in `.venv` beyond the declared dependencies:

```
torch 2.11.0+cu128 · torchvision · onnxruntime-gpu 1.28.0
opencv-python-headless 5.0.0.93
```

`opencv-python-headless` is deliberate — the normal `opencv-python` wheel
bundles its own Qt and conflicts with PySide6.

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
  All four are currently clean; 202 tests pass.
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
