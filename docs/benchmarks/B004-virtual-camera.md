# B004 — Virtual camera output

**Date:** 2026-08-13 · **Day 5**
**Question:** what does publishing to a virtual camera cost the frame budget,
and does the pipeline still hold 30 fps with it on?

Everything here was measured on the reference machine with
`scripts/benchmark_virtual_camera.py`. Reproduce with:

```powershell
.venv\Scripts\python.exe scripts\benchmark_virtual_camera.py --frames 1000
.venv\Scripts\python.exe scripts\benchmark_virtual_camera.py --pipeline --seconds 20
```

---

## Conditions

| | |
|---|---|
| Machine | i7-14700KF, 32 GB, RTX 5070 Ti, Windows 11 Pro 26200 |
| Backend | `pyvirtualcam 0.15.0`, `obs` backend |
| Device | `OBS Virtual Camera` (obs-virtualcam-module64.dll) |
| Format sent | BGR, converted to NV12 inside the backend |
| Resolution | 1280×720 @ 30.00 fps |
| Warm-up | 30 frames, discarded |
| Samples | 1000 per format |
| Consumer | **none attached** — see the caveat below |

The device was verified present by reading the DirectShow capture-filter
category from the registry, not by opening it. OBS Studio was installed but not
running.

---

## 1. `send()` per frame

| stage | mean ms | p50 | p95 | p99 | max | n |
|---|---|---|---|---|---|---|
| send, BGR (no conversion) | 1.437 | 1.417 | 1.540 | 1.910 | 2.504 | 1000 |
| send, RGB (converted) | 2.086 | 1.990 | 2.720 | 3.333 | 3.631 | 1000 |

**BGR is the steady-state path.** The camera captures BGR, the warp engine
composites in BGR, and the renderer preserves whatever format it was handed, so
a running avatar costs **1.44 ms mean / 1.54 ms p95**.

**RGB costs 0.65 ms more**, and that path is real rather than hypothetical: the
idle placeholder frame is labelled RGB24, so it is what publishes while no
avatar is loaded. The conversion uses OpenCV; the numpy fallback in the sink was
not benchmarked because OpenCV is present in every real configuration.

Both are dominated by the same work — a 2.7 MB buffer converted to NV12 and
copied into shared memory. Nothing here is avoidable without changing the format
the pipeline composites in.

## 2. `is_available()`

**0.039 ms per call**, 1000 calls. The window polls this on a 500 ms timer while
the checkbox is disabled, so it costs about 0.008% of one UI thread. Cheap
because it is a registry read and an `importlib.util.find_spec`, and it opens
nothing.

## 3. Full pipeline with publishing on

Real Sony ZV-E10, real MediaPipe tracker, real virtual camera, 20 seconds.

| | |
|---|---|
| Frames | 593 |
| **Sustained** | **29.65 fps** |
| Frames published | 593 |
| Frames dropped by the sink | **0** |
| Output stage | 2.03 ms mean, 2.26 ms p95 |
| End-to-end p95 | 38.43 ms |

**⚠ No avatar was loaded** — `avatars/my_avatar/portrait.png` is missing, so the
warp engine had nothing to drive and the run published placeholder frames. That
is why the output stage shows the RGB figure (2.03) rather than the BGR one, and
it means **this run does not include the 13.8 ms the warp engine costs**.

The end-to-end p95 of 38.43 ms is measured from capture timestamp to completed
render and exceeds the 33.3 ms frame interval; sustained throughput was
nonetheless 29.65 fps, which says the pipeline is pipelined against the camera's
own cadence rather than falling behind. It is a latency figure, not a throughput
one, and it is the one to watch when the avatar is added back.

**This number has to be re-measured once the portrait exists.** It is the only
claim in this document that is not yet the real thing.

## 4. Round trip — does anything else actually see it?

Performance figures say nothing about whether the picture arrives. A solid
BGR (32, 200, 64) frame was published while a **DirectShow consumer** — the same
interface Discord, Zoom and OBS use — opened each capture device in turn and
read a frame back.

| DirectShow index | Device | Read back |
|---|---|---|
| 0 | Sony ZV-E10 | the real room, as expected |
| 1 | OBS Virtual Camera | **BGR (32, 201, 64)** |

Maximum channel error **1**, which is NV12 chroma rounding inside the backend
and not a colour-order mistake — a swapped R and B would read (64, 200, 32).
75 frames published, none dropped.

So the pixels sha256 renders reach an unrelated application unchanged. That is
the day's actual deliverable, and it is now verified rather than assumed.

Worth knowing: the consumer negotiated **640×480** by default even though the
device was configured at 1280×720. The OBS filter offers several resolutions and
the consumer picks; sha256 does not control that end.

---

## Budget

Using the measured parts, with the day-4 warp figure carried over:

```
33.3 ms  frame budget at 30 fps
 6.3 ms  face tracking (p95 6.9), CPU — B003
13.8 ms  avatar warp (p95 17.7), CPU — day 4
 0.3 ms  rendering
 1.4 ms  virtual camera output (p95 1.5) — this benchmark
-------
21.8 ms  spent, ~11.5 ms headroom
```

At p95 throughout the sum is 26.4 ms, still inside 33.3. Publishing spends
about **4% of the frame budget**, which was the thing worth knowing before
committing to a backend.

---

## Caveats, stated rather than discovered later

1. **No consumer was attached during the timings.** The round trip in section 4
   proves a consumer *can* read our frames, but every number in sections 1–3 was
   taken with nothing reading from the queue. The OBS backend is designed so a
   reader cannot block the writer; that is a design claim, not a measurement.
   Worth re-running with Meet open.
2. **No avatar in the pipeline run**, as above.
3. **`pyvirtualcam` is GPL-2.0 and cannot ship.** These numbers characterise the
   *development* backend. A native `MFCreateVirtualCamera` component would have
   a different cost profile, though the dominant term — one 720p frame converted
   and copied — is the same work whoever does it, so 1.4 ms is a reasonable
   order-of-magnitude expectation for the shipping path too.
4. **One machine, one device.** UnityCapture is not installed here and was not
   measured.
