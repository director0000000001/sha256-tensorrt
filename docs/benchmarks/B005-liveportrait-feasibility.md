# B005 — Can LivePortrait run in real time on this machine?

**Date:** 2026-08-14 · **Day 6**
**Question:** the warp engine has a ceiling and the owner asked for the fastest
route to a genuinely good result. LivePortrait is this project's documented
primary candidate (`OPEN_SOURCE_EVALUATION.md` B1/B2). Does it fit?

**Answer: no, not on this machine within this project's remaining time.**
Measured, not estimated. The reasons are specific and are recorded below so this
does not have to be re-discovered.

---

## 1. The licence question is resolved, and favourably

This was the blocker recorded as **UNCLEAR** since the evaluation. It is no
longer.

| | |
|---|---|
| LivePortrait code | **MIT** — LICENSE file, "Copyright (c) 2024 Kuaishou Visual Generation and Interaction Center" |
| LivePortrait **weights** | **MIT** — the model repository's own card declares `license: mit` in its metadata |
| FasterLivePortrait (code + ONNX models) | **MIT** — declared on its model repository |
| InsightFace `buffalo_l` | **Non-commercial research only.** Unchanged, and unambiguous |

The evaluation recorded that "a separate license statement for the released
checkpoints was not found". It exists: it is the `license: mit` field in the
model card's frontmatter. That is an explicit licence declaration attached to
the distribution of the weights themselves.

**What is still not confirmed:** a GitHub issue asking the team to confirm this
in writing (`KwaiVGI/LivePortrait#548`) is open and unanswered. So the position
is "declared MIT on the official distribution", not "confirmed on request".
Materially better than unknown; not a signed statement.

**InsightFace is avoidable and was avoided.** LivePortrait uses it only for face
detection and cropping. sha256 already tracks faces with MediaPipe Face
Landmarker, Apache-2.0 on code *and* weights. `scripts/fetch_liveportrait.py`
therefore downloads only the generator stack and never requests
`insightface/**` — omission rather than later deletion.

So the licence was never what stopped this. Performance was.

## 2. PyTorch: 38.4 ms per frame, model only

RTX 5070 Ti, torch 2.11.0+cu128, 256×256 input, 100 samples, 20 warm-up frames
discarded, `torch.cuda.synchronize()` around every measurement.

The per-frame path is `motion_extractor → warping_module → spade_generator`.
`appearance_feature_extractor` runs once per avatar and is excluded.

| stage | fp32 mean | fp16 mean |
|---|---|---|
| motion_extractor | 5.54 | 5.86 |
| warping_module | 23.41 | 11.94 |
| spade_generator | 32.95 | 19.58 |
| **per frame** | **63.45** | **39.85** |
| | 15.8 fps | 25.1 fps |

Peak VRAM 681 MB in fp16 — never the constraint.

Optimisations, same conditions:

| configuration | mean ms | fps |
|---|---|---|
| fp16 baseline | 41.10 | 24.3 |
| fp16 + `cudnn.benchmark` | **38.43** | **26.0** |
| fp16 + `cudnn.benchmark` + `channels_last` | 41.75 | 24.0 |
| `torch.compile` | **unavailable** — inductor lowering fails on Windows |

`channels_last` does not help and cannot be applied to the warping module at
all: it carries 3-D convolutions, and `channels_last` is a rank-4 layout.

**38.4 ms is model inference alone**, against a 33.3 ms budget that must also
hold capture, tracking (6.9 ms), cropping, compositing and the virtual camera
(1.8 ms). The authors' 12.8 ms figure is an RTX 4090 and excludes everything
around the model; it is not reachable here.

## 3. The ONNX route is blocked by a plugin, not by speed

FasterLivePortrait exists precisely to solve this, and it fuses warping and
SPADE — 31.5 of the 38.4 ms — into one graph. It could not be measured.

| blocker | detail |
|---|---|
| `warping_spade-fix.onnx` will not load | `INVALID_GRAPH`. It carries a **custom `grid_sample_3d` node**, which is why a `grid_sample_3d_plugin.dll` ships beside it. Only TensorRT can execute it |
| CUDA execution provider will not start | `onnxruntime_providers_cuda.dll` depends on `cublasLt64_13.dll` (CUDA **13**); this machine has CUDA 12.8 from torch. Downgrading to `onnxruntime-gpu==1.22.0` moved the requirement to `cublasLt64_12.dll` and still failed to load it, even with torch's own bundled copy on the DLL search path |
| TensorRT execution provider will not start | same missing CUDA runtime, plus TensorRT itself is not installed |
| FasterLivePortrait's own prebuilt runtime | ships as `onnxruntime_gpu-1.17.0-**cp310**-win_amd64.whl`. This project is **Python 3.12** |

### ⚠ Correction to B001

B001 recorded: *"`onnxruntime-gpu 1.28.0` exposes **both** CUDA and TensorRT
providers."*

That is true and misleading. `get_available_providers()` lists them, but
**neither can create a session on this machine** — the provider DLLs cannot load
their CUDA dependencies. Listing a provider is not the same as having one.
B001's own conclusions do not depend on this, but the sentence should not be
relied on.

## 4. What making it work would actually cost

Not "install a package". A matching CUDA toolkit and cuDNN system-wide, plus
TensorRT for that exact version, plus a `grid_sample_3d` plugin built against
that TensorRT, plus engine builds per model — and quite possibly Python 3.10 to
use the runtime its authors ship.

Days of environment work with an uncertain end, on **day 6 of 14**, with the
licence server, download site, avatar encryption and `.exe` packaging still
outstanding.

**Decision: stopped.** The weights are kept (`models/liveportrait*`, gitignored,
1.2 GB) and `scripts/fetch_liveportrait.py` is committed, so the route can be
reopened without repeating any of this — most usefully on a machine where
TensorRT is already working, or after this delivery.

## 5. What this means

The project brief already said photorealistic neural synthesis "is not
achievable with current real-time open-source technology under the licence
constraints". This benchmark is that statement, measured, on this hardware, with
the licence part now removed from it: **the licence is fine and the frame rate
is not.**

The warp engine is what ships. Its remaining quality work is item 9 in
`AVATAR_QUALITY_BACKLOG.md` — region-limited blending — which is also what makes
additional photographs help instead of interfere.
