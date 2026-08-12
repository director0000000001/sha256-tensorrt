# B002 — Camera capture and pipeline baseline

**Date:** 2026-08-09
**Purpose:** First measurement with real hardware. Establishes what the camera
actually delivers and how much of the frame budget the pipeline machinery costs
before any AI component exists.

---

## Setup

| | |
|---|---|
| Camera | Sony ZV-E10, USB Streaming mode |
| Connection | USB-C, USB 3.0 root hub |
| Camera mode | NTSC, XAVC S HD 60p |
| Capture backend | `opencv` via **DirectShow** |
| Host | i7-14700KF, RTX 5070 Ti, Windows 11 |

**Note on the backend:** Media Foundation (MSMF) was tried first and failed to
open the device; DirectShow succeeded. The fallback chain in
`OpenCVCameraProvider._open_device` is therefore load-bearing on this hardware,
not a theoretical safety net.

---

## Result 1 — Camera output

The camera negotiates **1280×720** over USB Streaming and will not provide 1080p
regardless of what is requested. This appears to be a limitation of the camera's
USB Streaming implementation, not of the cable or port — the device enumerates
on a USB 3.0 root hub, and only two formats are advertised.

**720p is adequate.** Face tracking at this resolution is unconstrained for a
subject at desk distance, and the avatar engine works at 256×256 internally, so
input resolution above 720p would be discarded almost immediately.

### Frame rate, 6-second run

| | |
|---|---|
| Frames captured | 179 |
| Failed reads | 0 |
| **Achieved rate** | **29.77 fps** |
| Frame interval, median | 33.19 ms |
| Frame interval, min | 24.37 ms |
| Frame interval, p95 | 40.94 ms |
| Frame interval, max | 64.79 ms |
| Jitter (stdev) | 4.38 ms |

A 30 fps target needs a frame every 33.3 ms. The median of 33.19 ms is exactly
on target.

**The NTSC/PAL setting matters.** In PAL the camera offered only 25 fps; after
switching to NTSC it offers 30. This is a global camera setting, separate from
the recording format, and it is the single change that took capture from 25 to
30 fps.

The 64.79 ms maximum is one interval of roughly double length — a single
skipped frame in 179. Not investigated further at this stage; worth watching if
it becomes frequent under load.

---

## Result 2 — Pipeline cost, placeholders only

Full path exercised: camera → tracker → avatar engine → renderer, measured
through the application's own `PerformanceMonitor`.

| Stage | mean | p95 | max | n |
|---|---|---|---|---|
| capture | 31.97 ms | 42.49 ms | 77.29 ms | 176 |
| tracking | 0.04 ms | 0.05 ms | 0.11 ms | 176 |
| animation | 1.85 ms | 2.09 ms | 2.97 ms | 176 |
| rendering | 0.16 ms | 0.19 ms | 0.52 ms | 176 |
| **end-to-end** | **34.05 ms** | **44.48 ms** | 80.39 ms | 176 |

Effective throughput: **29.33 fps**.

### Reading the capture number correctly

**31.97 ms for `capture` is almost entirely waiting, not work.** `read()` blocks
until the camera delivers the next frame, and at 30 fps that is a 33.3 ms
interval. The number measures the frame period, not the cost of capturing.

The figure that matters is what the pipeline *spends*:

```
tracking + animation + rendering  =  ~2.05 ms
frame budget at 30 fps            =  33.3 ms
------------------------------------------------
headroom for real AI components   =  ~31 ms
```

**About 31 ms is available for MediaPipe plus the avatar engine.** For context,
B001 measured a portrait-shaped conv workload at 1.83 ms fp32 — the real
LivePortrait model is expected to be roughly an order of magnitude heavier, so
this headroom is meaningful rather than merely comfortable.

---

## What this does not measure

- **Camera and USB latency.** `pipeline_latency` starts from the timestamp the
  application assigns when `read()` returns, so it excludes everything upstream:
  sensor exposure, in-camera processing, USB transfer, driver buffering. That
  delay is real and unmeasured. Measuring it needs an external reference — the
  standard method is filming a running millisecond timer and comparing.
- **Anything about avatar quality.** No real tracker or engine is integrated.
- **Sustained behaviour.** Six seconds is enough to establish a rate, not enough
  to reveal thermal throttling, memory growth, or drift.

---

## Verdict

**Camera capture is solved and is not a bottleneck.** 30 fps is sustained, frame
pacing is even, and the pipeline machinery costs about 2 ms of a 33 ms budget.

Two findings worth carrying forward:

1. DirectShow is required on this hardware; MSMF does not open the device.
2. The camera is fixed at 720p over USB Streaming. Accepted rather than worked
   around, because 720p exceeds what the pipeline needs.

## Next

**B003 — MediaPipe Face Landmarker**, measured on real frames from this camera:
per-frame cost, CPU versus GPU delegate availability on Windows Python, and
blendshape output quality.

## Reproduce

```powershell
# configs/local.toml selects the opencv backend
uv run python -m sha256 --check
```
