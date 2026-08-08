# sha256 — Performance Plan

**Status:** the measurement infrastructure is implemented and tested. **No
benchmark has been run. No performance number is claimed anywhere in this
repository.**

This document defines how performance will be measured, what will be measured,
and what would count as success — established *before* any implementation, so
that targets are set by requirements rather than retrofitted to whatever the
first implementation happens to achieve.

---

## 1. Why this document exists first

Two failure modes it is meant to prevent:

1. **Optimising without evidence.** Building a GPU compositing path, or adopting
   TensorRT, before measuring whether the simpler option is sufficient. Every
   such choice adds complexity permanently in exchange for a speedup nobody has
   demonstrated is needed.
2. **Declaring success against a moved goalpost.** Measuring first and setting
   the target afterwards produces a number that describes the implementation
   rather than the requirement.

Project rule 11 — *benchmark before claiming optimisation* — and rule 12 — *do
not make unsupported claims* — are the reasons this file is written now and
contains no results.

---

## 2. What actually matters

For a real-time avatar in a video call, in priority order:

1. **End-to-end latency** — camera photon to virtual camera frame. Above roughly
   100 ms, the avatar is visibly out of sync with the user's voice, and no
   amount of frame rate fixes it.
2. **Frame-time consistency** — p95 and p99, not the mean. A 30 fps average with
   a p99 of 80 ms stutters visibly. Smoothness is governed by the slow frames.
3. **Sustained throughput** — 30 fps held for the length of a meeting, not for
   a ten-second benchmark.
4. **Resource headroom** — the machine is simultaneously running a browser, a
   video-call application, and a desktop compositor. A pipeline that saturates
   the GPU makes everything else stutter, which the user experiences as sha256
   being broken.

Peak frame rate in isolation is the least interesting number and the easiest to
report misleadingly.

---

## 3. Metrics

All collected through `PerformanceMonitor` (`src/sha256/performance/`).
Implemented and unit-tested.

### Per-stage durations

| Stage | Measures |
|---|---|
| `capture` | Camera read |
| `tracking` | Face tracker inference |
| `animation` | Avatar engine inference |
| `rendering` | Compositing |
| `output` | Virtual camera write |
| `end_to_end` | Capture timestamp → render complete |

Reported as mean, min, max, p50, p95, p99 over a rolling window (default 240
samples ≈ 8 s at 30 fps).

**Rolling, not cumulative.** A lifetime average hides the thing that matters —
that the last few seconds got worse.

**Percentiles are nearest-rank**, so every reported value is one that actually
occurred. Interpolated percentiles invent numbers, which is unacceptable when
the number is used to justify a claim.

### Throughput and loss

| Metric | Meaning |
|---|---|
| FPS per stage | Derived from timestamps across the window, not counted per wall-clock second |
| Frames processed | Cumulative per stage |
| **Frames dropped** | Deliberately discarded under load |
| **Source sequence gaps** | Frames the *camera* dropped — carried on `FrameMetadata.sequence_gap` |

The last two are deliberately separate. A USB bandwidth problem and a pipeline
overload problem look identical in a frame-rate graph and have completely
different fixes.

### Resource usage — **not yet implemented**

| Metric | Status |
|---|---|
| CPU utilisation | Interface only (`SystemProbe`) |
| Process memory | Interface only |
| GPU utilisation | Interface only |
| VRAM used / total | Interface only |

`NullSystemProbe` returns `None` for every field. **`None` means "not
measured"** and is displayed as `n/a`; `0.0` would read as a measurement. That
distinction is enforced in code and tested.

Implementing real probes needs `psutil` and `pynvml`. Deliberately deferred:
nothing currently produces numbers worth sampling, and the probe itself costs
time on the measured path.

---

## 4. Measurement methodology

### The rules

1. **Monotonic clocks only.** `time.perf_counter()` throughout. Wall-clock can
   step backwards across NTP corrections, producing negative latencies that
   silently corrupt rolling statistics.
2. **Frame identity end to end.** The same `frame_id` and `capture_timestamp`
   travel from capture to output. End-to-end latency is therefore *measured*,
   not inferred by summing stage averages — a sum that omits queueing delay and
   is always optimistic.
3. **Measure in `finally`.** A stage that raises still records its timing.
   Otherwise metrics look healthiest exactly when the pipeline is breaking.
4. **Discard warm-up.** First-run CUDA context creation, kernel autotuning, and
   memory allocation are one-time costs. Report them separately as startup cost;
   excluding them silently would overstate steady-state performance, and
   including them would understate it.
5. **Report the distribution, not a single number.** Mean, p95, p99, and the
   sample count. A number without a count is not a measurement.
6. **State the conditions.** Resolution, frame rate, avatar, backends, driver
   version, and what else was running.

### Reference machine

Verified via `nvidia-smi` and `scripts/check_gpu.py`:

| | |
|---|---|
| CPU | Intel Core i7-14700KF |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| VRAM | 16303 MiB |
| Compute capability | 12.0 (`sm_120`, Blackwell) |
| Driver | 610.62 |
| RAM | 32 GB |
| OS | Windows 11 Pro |
| Python | 3.12 |

Results from any other machine must say so. A figure without its hardware is
not a result.

### Scenarios to run

| # | Scenario | Purpose |
|---|---|---|
| S1 | Idle pipeline, no avatar | Baseline overhead of the machinery itself |
| S2 | Single avatar, 720p30 | The primary target case |
| S3 | Single avatar, 1080p30 | Resolution scaling |
| S4 | Avatar switch under load | Load stall duration, eviction cost |
| S5 | Sustained 30 minutes | Thermal throttling, leaks, drift |
| S6 | Concurrent video call | The realistic case — contended GPU |
| S7 | Degraded input (low light, motion) | Tracker robustness and its cost |

S6 is the one most likely to be skipped and the one that most resembles actual
use.

---

## 5. Targets

Set from requirements, not from measurements. **None has been validated.**

### Latency budget at 720p30

33.3 ms per frame. A provisional allocation:

| Stage | Budget | Notes |
|---|---|---|
| Capture | ≤ 5 ms | Largely driver-bound |
| Tracking | ≤ 8 ms | Unknown; MediaPipe on Windows Python may be CPU-only |
| Animation | ≤ 15 ms | The dominant cost and the largest unknown |
| Rendering | ≤ 3 ms | CPU compositing may suffice |
| Output | ≤ 2 ms | Format conversion plus the write |

**This allocation is a hypothesis.** Its purpose is to make it obvious *which*
stage broke the budget when the total is missed — not to predict what any stage
will cost. Expect it to be redistributed once real numbers exist.

### Success criteria

| Metric | Target | Rationale |
|---|---|---|
| End-to-end p95 | < 80 ms | Below the threshold where lip-sync drift is noticeable |
| End-to-end p99 | < 120 ms | Occasional spikes tolerable; sustained ones are not |
| Sustained FPS | ≥ 30 for 30 min | A meeting, not a demo |
| Dropped frames | < 1% | Occasional drops are invisible; systematic drops are not |
| VRAM, one avatar | < 4 GB | Leaves headroom on a 16 GB card for everything else |
| Avatar switch | < 3 s | Above this, warm caching stops feeling useful |
| Idle CPU (S1) | < 5% | The machinery must not cost anything when doing nothing |

**Fallback position:** if 30 fps proves unreachable at acceptable quality, 24 fps
with consistent frame times is a better product than 30 fps with visible
stutter. Consistency outranks peak rate.

---

## 6. What gets benchmarked, and when

Deliberately ordered so that each measurement informs the next decision:

| Order | Benchmark | Decides |
|---|---|---|
| 1 | Camera capture backends (Media Foundation vs OpenCV/DirectShow) | Which capture path to build on |
| 2 | MediaPipe Face Landmarker: CPU vs GPU delegate, resolutions | Whether GPU tracking is needed, or even available on Windows Python |
| 3 | OpenSeeFace, for comparison | Whether the backup is viable, and what the solver would cost |
| 4 | LivePortrait via ONNX Runtime CUDA | Whether the neural path is real-time on this GPU |
| 5 | Same, via TensorRT | Whether TensorRT's complexity and proprietary license are earned |
| 6 | CPU vs GPU compositing | Whether a GPU renderer is needed at all |
| 7 | Virtual camera write cost | Whether output needs its own thread |
| 8 | 3D + blendshape path | The backup's real cost profile |

Nothing before step 1 is measurable, because no real backend exists.

---

## 7. Reporting

Results go in `docs/benchmarks/results/` (gitignored; large and machine-specific)
with a committed summary per benchmark.

Every report must state:

- Date, commit SHA, hardware, driver, OS build, Python version
- Backends and versions under test
- Scenario, resolution, frame rate, avatar
- What else was running
- Sample count, mean, p50, p95, p99, min, max
- Warm-up handling
- Known caveats

A number without these is not a result and must not be quoted.

### Language rules

| Do not write | Write instead |
|---|---|
| "sha256 runs at 60 fps" | "S2 measured p50 16.2 ms / p95 21.4 ms over 5400 frames on the reference machine" |
| "TensorRT is faster" | "TensorRT reduced animation p95 from X to Y under S2" |
| "Optimised the pipeline" | "Change Z moved end-to-end p95 from X to Y" |

Every claim carries its measurement, or it is not made.

---

## 8. Current state

| Capability | Status |
|---|---|
| Duration timing per stage | Implemented, tested |
| Rolling windows with percentiles | Implemented, tested |
| FPS from timestamps | Implemented, tested |
| Dropped-frame counting | Implemented, tested |
| Source sequence gaps | Field exists; nothing populates it |
| End-to-end latency | Implemented via frame identity |
| Snapshot for UI/export | Implemented, tested |
| CPU / GPU / VRAM sampling | **Interface only** |
| Benchmark harness | **Not built** |
| Results | **None** |

The debug panel displays `n/a` for every unmeasured metric and states plainly
that resource sampling is not implemented. That is the intended behaviour, not a
placeholder to be filled with plausible-looking values.

---

## See also

- `docs/architecture/SYSTEM_ARCHITECTURE.md` — threading design to be validated here
- `docs/OPEN_SOURCE_EVALUATION.md` — third-party claims to be reproduced
- `scripts/check_gpu.py` — verify GPU capability before trusting any GPU result
