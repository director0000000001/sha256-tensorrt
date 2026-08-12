# B003 — MediaPipe Face Landmarker on real camera frames

**Date:** 2026-08-13
**Purpose:** Establish what the first real AI component costs, on this machine,
on frames from this camera — and settle the two questions
`docs/OPEN_SOURCE_EVALUATION.md` left open: whether a GPU delegate exists on
Windows Python, and what a frame actually costs.

This is the first measurement of a component that does real work. B001 measured
the GPU's capability and B002 measured the camera; both found headroom. This one
spends some of it.

---

## Setup

| | |
|---|---|
| Tracker | MediaPipe Face Landmarker, `mediapipe 1.0.0` |
| Model bundle | `face_landmarker.task`, float16 v1, 3.58 MiB |
| Model SHA-256 | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |
| Model licence | Apache-2.0 (code and weights) |
| Running mode | `VIDEO` (tracking carried between frames) |
| Camera | Sony ZV-E10, 1280×720, DirectShow, NTSC |
| Host | i7-14700KF, RTX 5070 Ti, 32 GB, Windows 11 Pro, Python 3.12 |
| Subject | One face, seated at desk distance, ordinary room lighting |

The model bundle is not committed. `models/metadata/face_landmarker.json`
records its URL, version, size, and checksum;
`scripts/fetch_face_landmarker.py` fetches and verifies it.

---

## Result 1 — There is no GPU delegate on Windows Python

Asking for it fails at graph construction, before any frame:

```
could not create the MediaPipe Face Landmarker with delegate 'gpu':
ValidatedGraphConfig Initialization failed.
ImageCloneCalculator: GPU processing is disabled in build flags
```

**The published Windows wheel is built without GPU support.** This is not a
driver problem, a CUDA problem, or something a newer NVIDIA release fixes — the
capability is compiled out of the wheel. The `BaseOptions.Delegate.GPU` enum
value exists, which is exactly the trap: the option is selectable and fails only
when the graph is built.

Consequences, all now settled rather than assumed:

- Face tracking runs on the **CPU**, through XNNPACK, and will keep doing so
  unless the project builds MediaPipe from source — which is a large amount of
  work for a component that, as measured below, is not the bottleneck.
- `tracking.delegate = "auto"` therefore resolves to CPU on Windows rather than
  attempting GPU first. Trying costs a failed graph build and buys nothing.
- An explicit `delegate = "gpu"` still fails loudly, with the message above. It
  is not silently downgraded: a user who asks for the GPU should be told it does
  not exist, not left believing they got it.

**This is good news for the frame budget**, not bad. Tracking on the CPU leaves
the GPU entirely free for the avatar engine, which is the component that will
actually need it.

---

## Result 2 — Per-frame cost

Sustained run: **60 seconds, 1777 frames captured**, first 30 discarded as
warm-up, 1747 measured. Zero failed camera reads. Capture held **29.61 fps**
throughout, matching B002.

Timing is of `FaceTracker.process()` — the whole public call the pipeline makes,
including colour conversion and the translation into sha256's own types, not
just model inference.

| | mean | p50 | p95 | p99 | min | max | n |
|---|---|---|---|---|---|---|---|
| `process()` | **6.30 ms** | 6.26 | **6.93** | 7.32 | 5.63 | 8.06 | 1747 |

**The distribution is tight.** p99 is 16% above the median and the worst frame in
a minute was 8.06 ms. Nothing here threatens frame pacing; compare B002, where
the camera's own delivery jitter was 4.4 ms.

### Where the time goes

Separate 300-frame run with `--breakdown`, which times the internal steps
individually:

| stage | mean | p50 | p95 | p99 | max | n |
|---|---|---|---|---|---|---|
| `process()` total | 6.14 ms | 6.09 | 6.76 | 7.06 | 7.19 | 300 |
| — colour convert + wrap | 0.95 ms | 0.94 | 1.10 | 1.15 | 1.29 | 300 |
| — MediaPipe inference | 4.90 ms | 4.87 | 5.24 | 5.55 | 6.24 | 300 |
| — adapt to neutral types | 0.16 ms | 0.15 | 0.17 | 0.20 | 0.21 | 300 |

**Inference is 80% of it.** The architecture's own overhead — converting 478
landmarks, 52 blendshapes, and a 4×4 matrix into `FaceTrackingResult` — is
0.16 ms, about 2.5%. The neutral-type boundary that makes the tracker
replaceable costs approximately nothing, which is the answer to the obvious
question about whether the abstraction is affordable.

### Startup

Building the landmarker takes **0.26–0.35 s** across runs, plus ~0.31 s to import
MediaPipe the first time. Paid once, in `start()`, never on the frame path. It is
excluded from every number above and reported here separately, per
`PERFORMANCE_PLAN.md` rule 4.

---

## Result 3 — Colour conversion was worth 2.4 ms

The camera delivers BGR (OpenCV's native layout, deliberately unconverted at
capture). MediaPipe wants RGB. The first implementation reversed the channel
axis in numpy, on the reasonable-sounding grounds that the tracker should not
require OpenCV.

Measured on the same 1280×720 frame, 50 repeats:

| method | per frame |
|---|---|
| `np.ascontiguousarray(data[:, :, ::-1])` | **2.90 ms** |
| `cv2.cvtColor(data, COLOR_BGR2RGB)` | **0.45 ms** |

Reversing a numpy axis produces a non-contiguous view, and making it contiguous
again is a full-frame gather. OpenCV's version is a vectorised copy.

**2.4 ms is 7% of the entire 33 ms frame budget, spent rearranging bytes.** The
tracker now uses OpenCV when it is importable and keeps the numpy path as a
fallback, so it still does not *require* OpenCV. A unit test asserts the two
paths produce identical pixels, because a tracker that saw different pixels
depending on which library happened to be installed would be a silent source of
divergent behaviour.

End to end this moved `process()` from a mean of **8.77 ms** (n=90) to
**6.30 ms** (n=1747) on the same machine and camera. The sample counts differ, so
the micro-benchmark above is the direct evidence and this is corroboration.

---

## Result 4 — What the tracker emits

| | |
|---|---|
| Landmarks | **478**, every frame, normalised to `[0, 1]` |
| Blendshapes | **52** |
| Facial transform matrix | 4×4, every frame |
| Detection rate | **100%** (1777 of 1777 frames) |

The 52 blendshape names are ARKit-style, alphabetically ordered, and the **first
is `_neutral`** — a model output rather than an expression. So MediaPipe's set is
`_neutral` plus 51 ARKit names, and it is **not** the ARKit 52 (ARKit's
`tongueOut` has no MediaPipe equivalent).

The adapter therefore reports the scheme as `mediapipe_52`, not `arkit_52`. An
engine that wants ARKit semantics must remap by name. Labelling it `arkit_52`
would have been the kind of small lie that produces a subtly wrong mouth three
weeks later.

100% detection is on a cooperative subject in decent light. It says nothing about
poor light, motion blur, or partial occlusion — scenario S7 in the performance
plan, not yet run.

---

## Result 5 — The head-pose convention, and why the obvious check was wrong

`HeadPose` documents one convention: pitch positive looking **up**, yaw positive
turning to the subject's **left**, roll positive tilting to the subject's
**right**. MediaPipe supplies a 4×4 matrix and no statement of which frame it is
expressed in. Getting a sign wrong here produces an avatar that leans the wrong
way — a bug that is invisible in tests and obvious on a video call.

**Two of the three signs were wrong in the first implementation.** They were
caught by measurement, not review.

### The check that failed

The first attempt compared the reported angles against 2-D landmark proxies:
the nose's offset from the eye midpoint for yaw, the chin-to-eye against
eye-to-forehead ratio for pitch, which eye sits lower for roll.

It reported **3.5% agreement on yaw and 5.7% on pitch** — while the more careful
checks below reported 100% on the same frames. The proxies were wrong, not the
tracker. With the subject's head tilted about 30°, a "nose versus eye midpoint"
offset is dominated by *roll*, not yaw; the pitch ratio is contaminated the same
way. **Each proxy silently measures the other axes.** They have been removed from
the benchmark rather than left in to be misread later.

### Check 1 — known transforms of a real frame

Rotating the image is a pure roll of exactly known size. Mirroring swaps the
subject's left and right, so yaw and roll must invert and pitch must not.

| image rotated by | reported roll change |
|---|---|
| −20° | −19.8° |
| −10° | −10.0° |
| +10° | +10.5° |
| +20° | +20.4° |

Response slope **+1.01**. Before the fix this was **−1.00** — an exact
inversion.

| axis | original | mirrored | expected |
|---|---|---|---|
| yaw | −0.3° | −0.1° | flips |
| roll | +3.9° | −4.1° | flips |
| pitch | −27.2° | −28.2° | unchanged |

No axis leaks into another.

### Check 2 — against the face normal, on every frame

The face normal computed from three landmarks is an absolute 3-D statement of
where the face points, and the landmarks are a **separate model output** from the
transformation matrix — so agreement between them is evidence, not a tautology.
Unlike a movement-based check it works on a perfectly still head, which is what
makes it repeatable in an automated benchmark.

Over the 60-second run:

| axis | frames compared | sign agreement |
|---|---|---|
| pitch | 1747 | **100.0%** |
| yaw | 1109 | **100.0%** |
| roll | 1745 | **100.0%** |

### The frame, now established

MediaPipe's matrix is right-handed with **X to the image right, Y up, Z toward
the viewer**. That is not read from documentation — it is the only frame in which
a counter-clockwise image rotation produces a matching positive rotation about Z,
which is what the rotation table above measures. From it:

| axis | raw rotation | reported as | why |
|---|---|---|---|
| pitch | about +X | **negated** | positive +X swings the face's forward direction downward |
| yaw | about +Y | as is | positive +Y turns the face toward the image right, the subject's left |
| roll | about +Z | as is | positive +Z tips the top of the head toward the image left, the subject's right |

The unit tests now build rotation matrices for the *physical* pose they describe
and assert the tracker recovers them, so a future change that breaks the
convention fails the suite instead of shipping.

---

## Result 6 — MediaPipe reports no confidence

`FaceLandmarkerResult` carries exactly three fields: landmarks, blendshapes, and
transformation matrices. **There is no per-face score.** Detection is binary — a
result exists precisely because the model's internal presence score cleared
`min_face_presence_confidence`.

`FaceTrackingResult.confidence` is therefore reported as `1.0` when a face is
tracked, and the adapter says plainly in its own docstring that this means
"tracked", not "certain". The alternative — inventing a graded number — would put
a fabricated value into a field the UI displays and the engine may come to trust.

A real confidence signal, if one is wanted, has to be derived (landmark
stability across frames, for instance) and measured. That is quality work, not
integration work.

The model's `_neutral` blendshape score is carried in `extras` as a diagnostic,
since it is a genuine model output.

---

## Budget impact

```
33.3 ms   frame budget at 30 fps
 2.0 ms   pipeline machinery (B002, placeholders)
 6.3 ms   face tracking, measured here (p95 6.9 ms)
--------
~25 ms    left for the avatar engine
```

Tracking uses **21% of the frame budget at p95**, on the CPU, leaving the GPU
untouched. B002's estimate of ~31 ms for "MediaPipe plus the avatar engine" was
close: MediaPipe took 6.3 of it.

For context, B001 measured a portrait-shaped conv stack at 1.83 ms fp32 and noted
the real LivePortrait model is realistically an order of magnitude heavier —
15–20 ms. Against 25 ms remaining, that fits, with less room than before but
still fits. **This is the number to hold onto when B004 measures the engine.**

---

## What this does not measure

- **Robustness.** One subject, one room, good light, cooperative pose. Poor
  light, motion blur, glasses, partial occlusion, and off-axis angles are
  scenario S7 and are not covered by 100% detection here.
- **Multiple faces.** `num_faces = 1`. Nothing is known about the cost of more.
- **Smoothing.** The EMA filter is implemented and unit-tested but measured only
  at `smoothing = 0`. What it costs in latency, and whether it is worth it, is a
  quality question for later.
- **Contention.** Nothing else was running. Scenario S6 — a real video call
  competing for the machine — remains the most realistic and least tested case.
- **Camera and USB latency.** Still unmeasured, still needs an external
  reference. Unchanged from B002.
- **Anything beyond one minute.** Sixty seconds shows no drift; it is not the
  30-minute sustained run of scenario S5.

---

## Operational notes

Two things cost time during this benchmark and are worth recording:

1. **The camera intermittently refuses to open.** It enumerates correctly
   (`[0] ZV-E10`) and then fails on both MSMF and DirectShow, for a minute or
   two, before working again. Since B002 established that MSMF cannot open this
   device at all, the failed MSMF attempt is a candidate cause for the
   DirectShow attempt immediately after it failing too. Not investigated. If it
   becomes frequent, the fix is a configurable backend preference so the dead
   MSMF attempt can be skipped entirely.
2. **In `VIDEO` mode, an abrupt change between frames can lose the face.** Feeding
   a 20°-rotated frame straight after an unrotated one sometimes returned no
   detection, where a freshly created landmarker handled the same image fine.
   The prior helps in normal use and hurts on discontinuities. The benchmark's
   synthetic checks now build a fresh tracker per transform for that reason.

---

## Verdict

**MediaPipe Face Landmarker is adopted as the face tracker.** It costs 6.3 ms of
a 33 ms budget on the CPU, holds 30 fps without touching the GPU, detects
reliably in normal conditions, and emits exactly the blendshapes an
expression-driven engine needs — under Apache-2.0 on both code and weights, which
remains the cleanest licensing position of any component in this project.

The GPU-delegate question is closed: there is no GPU delegate, and it is not
needed.

OpenSeeFace stays on the shelf as a backup. Nothing measured here argues for
building the landmark→blendshape solver it would require.

## Next

**B004 — the avatar engine.** The remaining ~25 ms is the whole question, and it
is the largest unknown left in the schedule.

## Reproduce

```powershell
python scripts/fetch_face_landmarker.py            # once, ~3.6 MiB
python scripts/benchmark_tracking.py --seconds 60  # sit in front of the camera
python scripts/benchmark_tracking.py --breakdown   # per-stage decomposition
python scripts/benchmark_tracking.py --delegate gpu  # reproduce the GPU failure
```

Raw results are written with `--json` into `docs/benchmarks/results/`
(gitignored).
