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
| Consumer | **none attached** during the timings — see section 4 and the caveats |
| Avatar | `my_avatar`, 20 source photographs (section 3) |

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

Real Sony ZV-E10, real MediaPipe tracker, real virtual camera, real avatar with
**20 source photographs**, face present and tracked. 20 seconds, 596 frames.

| Stage | mean ms | p95 ms |
|---|---|---|
| capture (blocked waiting for the camera) | 2.11 | 9.44 |
| tracking | 6.93 | 8.14 |
| **animation** | **22.06** | **25.16** |
| rendering | 0.33 | 0.41 |
| output (virtual camera) | 1.82 | 2.70 |
| end-to-end (capture → rendered) | 31.56 | 38.82 |

| | |
|---|---|
| **Sustained** | **29.80 fps** |
| Frames published | 596 |
| Frames dropped by the sink | **0** |

Throughput is intact — the camera supplies 29.77 fps and the pipeline keeps up
with it, dropping nothing. **The headroom is not.** Work excluding the
capture wait is 31.14 ms mean against a 33.6 ms frame interval: about 2.5 ms of
margin, where day 4 had roughly 11.5 ms. At p95 the stages sum past the frame
interval, which is the condition that produces occasional hitches under any
additional system load — and a video call is additional system load.

### Where it went: the source blend stopped being cached

The animation stage was 13.8 ms with 6 source photographs and is 22.06 ms with
20. Instrumented over 596 real frames:

| | |
|---|---|
| Sources active per frame | **12.1 mean** (min 8, max 15) |
| Blend cache hit rate | **2.3%** |
| Blend cost | 6.75 ms mean, 8.04 ms p95 |

The blend costs one full-image pass per active source, and its result is cached
against weights quantised to 0.02. With six sources that cache almost always
hit and the blend was effectively free. With twelve independent dials, at least
one crosses a quantisation step nearly every frame, so the cache hits 2.3% of
the time and the full blend is rebuilt on 97.7% of frames.

**Capping the number of sources is not the answer**, which is worth recording
because it is the obvious fix. Keeping only the strongest N and measuring the
difference against the full blend, over 239 frames that had ≥8 active sources:

| Keep | mean pixel difference | worst pixel difference | blend ms |
|---|---|---|---|
| 4 | 5.66 | 56 | 2.70 |
| 6 | 2.77 | 29 | 3.65 |
| 8 | 1.46 | 18 | 4.62 |
| 10 | 0.80 | 10 | 5.68 |
| all (12.1) | — | — | 6.28 |

The weight beyond the top six averages **0.59** of the total — it is genuinely
spread across many photographs rather than concentrated in a few, so the cheap
caps change the picture and the safe caps save little. The lever is the cache
and the per-pass cost, not the source count.

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

All measured in the same run, with 20 source photographs:

```
33.6 ms  frame interval the camera actually delivers (29.77 fps)
 6.9 ms  face tracking (p95 8.1), CPU
22.1 ms  avatar animation (p95 25.2), CPU  <- of which 6.8 ms is the source blend
 0.3 ms  rendering (p95 0.4)
 1.8 ms  virtual camera output (p95 2.7)
-------
31.1 ms  spent, ~2.5 ms headroom
```

Publishing spends about **5% of the frame budget** — it was never the problem,
and that was the thing worth knowing before committing to a backend. The
animation stage is now 71% of it.

---

## Caveats, stated rather than discovered later

1. **No consumer was attached during the timings.** The round trip in section 4
   proves a consumer *can* read our frames, but every number in sections 1–3 was
   taken with nothing reading from the queue. The OBS backend is designed so a
   reader cannot block the writer; that is a design claim, not a measurement.
   Worth re-running with Meet open.
2. **One face, one distance, one room.** The animation cost depends on how large
   the face is in frame; a subject sitting closer costs more. Measured at
   ordinary desk distance.
3. **`pyvirtualcam` is GPL-2.0 and cannot ship.** These numbers characterise the
   *development* backend. A native `MFCreateVirtualCamera` component would have
   a different cost profile, though the dominant term — one 720p frame converted
   and copied — is the same work whoever does it, so 1.4 ms is a reasonable
   order-of-magnitude expectation for the shipping path too.
4. **One machine, one device.** UnityCapture is not installed here and was not
   measured.
