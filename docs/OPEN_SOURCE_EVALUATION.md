# SHA-256 — Open-Source Technology Evaluation

**Date of research:** August 2026
**Status:** research complete; **no candidate has been integrated or benchmarked
by this project.**

---

## How to read this document

Every performance figure below is either **claimed by the project's own authors**
or **reported by third parties**. None has been reproduced on the SHA-256
reference machine. They are recorded as leads for benchmarking, not as facts
about what SHA-256 will achieve. Where a number appears, its source is named.

**Licensing is reported at four separate levels**, because conflating them is
the single most common and most expensive mistake in this space:

| Level | Question |
|---|---|
| Source code | May I use and modify the code? |
| Pretrained weights | May I use the shipped model files? |
| Third-party deps | What do the things it pulls in require of me? |
| Commercial use | Can this ship in a product that makes money? |

Where a license could not be established from a primary source, it is marked
**UNCLEAR** and left unresolved. Guessing would be worse than not knowing.

### What was verified, and how

| Verified | Method |
|---|---|
| GPU: RTX 5070 Ti, driver 610.62, 16303 MiB, compute capability 12.0 | `nvidia-smi` on the reference machine |
| `torch 2.11.0+cu128`, `2.9.0+cu129`, `2.13.0+cu130` resolve for Python 3.12/Windows | `uv pip install --dry-run` against the PyTorch indexes |
| `mediapipe 1.0.0`, `onnxruntime-gpu 1.28.0`, `pyvirtualcam 0.15.0`, `opencv-python 5.0.0.93` resolve for Python 3.12/Windows | `uv pip install --dry-run` |
| Licenses cited below | Fetched from each project's repository or model card |

**Not verified:** any runtime behaviour, any latency, any quality claim. Nothing
was installed and run.

---

## 1. Requirements this is measured against

| # | Requirement | Consequence |
|---|---|---|
| R1 | Real-time on one consumer GPU | Rules out anything that cannot approach ~30 fps locally |
| R2 | Local-first, no cloud in the loop | Rules out API-only services entirely |
| R3 | Windows 10/11 | Linux-only tooling is a hard blocker |
| R4 | Python integration | A C++-only library needs a binding written and maintained |
| R5 | Replaceable behind an interface | Anything demanding architectural surrender is disqualified |
| R6 | Licensing that does not foreclose the project's options | Non-commercial weights and copyleft are decision-forcing |
| R7 | Fits in 16 GB VRAM alongside a video call | Large diffusion models are out |
| R8 | Actively maintained | An abandoned dependency becomes the maintainer's problem |

---

# Part A — Face tracking

## A1. MediaPipe Face Landmarker

| | |
|---|---|
| Source | `google-ai-edge/mediapipe` |
| Code license | **Apache-2.0** |
| Weights license | **Apache-2.0** (stated on the model card and the task documentation) |
| Python (3.12/Win) | **Verified available** — `mediapipe 1.0.0` resolves |
| Output | 478 3-D landmarks, **52 blendshape scores**, 4×4 facial transform matrix |
| Maintenance | Active; Google-maintained, documentation updated 2026 |

**Why it leads.** It is the only mainstream open-source tracker that emits
**blendshapes directly**. That matters more than landmark count: an
expression-driven avatar engine consumes blendshapes, and a tracker that emits
only landmarks needs an additional solver stage — one that does not exist, whose
quality is unknown, and which would add its own latency.

The 52 coefficients follow the ARKit naming convention, which is also what VRM
"Perfect Sync" avatars and most 3D avatar tooling expect. That alignment removes
a whole translation layer.

Apache-2.0 on **both code and weights** is the cleanest licensing position of
any candidate here, at either level.

**Concerns.**

- **GPU acceleration on Windows Python is UNCLEAR.** MediaPipe's GPU delegate is
  well documented for Android and web; whether the Windows Python wheel exposes
  a usable GPU delegate could not be confirmed from primary sources. Assume
  CPU/XNNPACK inference until measured. On an i7-14700KF this may well be
  sufficient — but that is an assumption to test, not a finding.
- **No latency benchmarks are published** in the official documentation. The
  model is described as optimised for mobile GPU inference; what that means on
  this hardware is unknown.
- Google has deprecated MediaPipe solutions before. The task API is the current
  supported surface, but this is a maintenance risk to watch.

**Suitable for SHA-256:** yes — the strongest candidate.

## A2. OpenSeeFace

| | |
|---|---|
| Source | `emilianavt/OpenSeeFace` |
| Code license | **BSD-2-Clause** |
| Weights license | **BSD-2-Clause** — the project states this covers code *and* models |
| Runtime | ONNX Runtime (models converted from PyTorch specifically because Windows CPU inference in PyTorch was slow) |
| Output | **66 landmarks**, gaze, blink. **No blendshapes.** |
| Claimed performance | 30–60 fps single face on CPU; model 0 ≈ 68 fps (low accuracy), model 3 ≈ 44 fps (highest accuracy); under one CPU core at 30 fps — *authors' figures* |
| Maintenance | Active; long-standing, widely used in the VTuber ecosystem |

**Why it is the right backup.** It was built for exactly this use case and has
years of real-world use behind it. The authors specifically claim robustness in
"low light, high noise, low resolution" and stable landmark positions —
properties that matter more for a video call than peak accuracy in good
conditions. Their own comparison notes greater mouth-pose variety than MediaPipe
but potentially weaker eye-region tracking.

BSD-2-Clause on code and models is permissive and unambiguous.

**The blocking gap: no blendshapes.** Using it as a real backup requires a
landmark→blendshape solver. That is genuine work of unknown quality, and it must
not be underestimated when treating this as a drop-in alternative. It is a
backup that requires building something first.

**Suitable for SHA-256:** yes as a backup, with the solver caveat stated
plainly.

## A3. Other options considered

| Option | Verdict |
|---|---|
| **dlib / 68-point classic** | Mature and permissive, but weaker under motion and poor light than either candidate above, and no blendshapes. No advantage. |
| **NVIDIA Maxine / Broadcast** | Excellent tracking and hardware-matched, but **proprietary and not open source**. Fails the project's premise. Worth noting it exists as a commercial fallback. |
| **InsightFace-based trackers** | Code MIT, but **pretrained models are non-commercial research only** (see B1). Disqualified for anything with a commercial future. |
| **3DDFA / SPIGA and similar research trackers** | Strong published accuracy, but research-grade packaging, uncertain Windows/Python story, and per-project weight licensing. Not worth the integration risk when MediaPipe is available. |

---

# Part B — Avatar / portrait animation

## B1. LivePortrait

| | |
|---|---|
| Source | `KwaiVGI/LivePortrait` (Kuaishou) |
| Code license | **MIT** — verified from the LICENSE file: "Copyright (c) 2024 Kuaishou Visual Generation and Interaction Center" |
| Weights license | **UNCLEAR** — the repository is MIT-licensed, but a *separate, explicit* license statement for the released checkpoints was not found. Do not assume MIT covers the weights. |
| Third-party constraint | **Depends on InsightFace `buffalo_l` detection models.** InsightFace code is MIT; its **pretrained models are "available for non-commercial research purposes only"** (verbatim from the InsightFace repository). |
| Commercial use | Possible **only** after removing and replacing InsightFace's detection models — the LivePortrait documentation states this explicitly. |
| Claimed performance | ~12.8 ms per frame on an **RTX 4090** with PyTorch — *authors' figure, model inference only* |
| Maintenance | Last significant update January 2025; not abandoned, but not fast-moving |

**Why it leads on capability.** It is a stitching-and-retargeting approach rather
than a diffusion model, which is precisely why it can run in real time at all.
It is driven by a **video/face source**, matching SHA-256's webcam-driven
pipeline. Quality on portraits is widely regarded as the best available in the
open-source real-time class.

**The licensing situation, stated precisely.** Three distinct facts:

1. LivePortrait's **code** is MIT. Unambiguous.
2. LivePortrait's **weights** have no separate license statement that could be
   located. **UNCLEAR.**
3. The **InsightFace models it depends on** are non-commercial research only.
   This is unambiguous and it is a real constraint.

Point 3 is a solvable engineering problem — InsightFace is used for face
detection and can be replaced with a permissively-licensed detector. Point 2 is
a legal question that must be resolved before any commercial deployment, and it
is not resolvable by reading code.

**Concerns.**

- The 12.8 ms figure is **model inference on an RTX 4090**, excluding pre- and
  post-processing. The RTX 5070 Ti is a lower tier. Real end-to-end cost will be
  substantially higher. Do not plan against 12.8 ms.
- Operates at 256×256 internally. Upscaling to a call-appropriate resolution is
  additional cost and a quality question.
- VRAM footprint on this hardware is unmeasured.

**Suitable for SHA-256:** yes as the primary candidate, **conditional on
resolving the weight licensing** and replacing InsightFace.

## B2. FasterLivePortrait

| | |
|---|---|
| Source | `warmshao/FasterLivePortrait` |
| Code license | **MIT** |
| Weights license | **Inherited** — "models are subject to their respective licenses", i.e. LivePortrait's and InsightFace's. Same unresolved questions as B1. |
| Runtime | **TensorRT** (recommended) or ONNX Runtime GPU |
| Claimed performance | **30+ fps on an RTX 3090 including pre- and post-processing** — *author's figure* |
| Windows | **Yes** — an installation-free Windows package is provided |
| Real-time webcam mode | **Yes** — explicitly supported |
| Maintenance | Updates through 2025 |

**Why this is the practical integration path.** The 30+ fps figure is the more
useful number precisely because it **includes pre- and post-processing**, which
is the honest measure. An RTX 3090 is roughly comparable to an RTX 5070 Ti for
this class of work, which makes it a plausible starting point rather than
aspirational.

More importantly it already provides ONNX and TensorRT paths, a Windows package,
and a webcam mode — meaning the integration work is adapter-writing rather than
research.

It does **not** improve the licensing position. It inherits every question from
B1.

**Suitable for SHA-256:** yes — this is how LivePortrait should be adopted if
LivePortrait is adopted.

## B3. Ditto (`antgroup/ditto-talkinghead`)

| | |
|---|---|
| Code license | **Apache-2.0** |
| Weights license | **UNCLEAR** — no separate statement located for the HuggingFace checkpoints |
| Driving signal | **Audio-driven** |
| Claimed performance | "Realtime"; tested on an **A100**; prebuilt TensorRT models target Ampere-plus |
| Windows | **UNCLEAR** — tested environment is CentOS 7.2; no explicit Windows support stated |

**Verdict: not suitable as a primary candidate**, for a structural reason rather
than a quality one. Ditto is **audio-driven** — it synthesises a talking head
from speech. SHA-256 is **motion-driven**: it reproduces the user's actual head
pose and expressions from a webcam. These solve different problems. An
audio-driven model cannot reproduce the user looking away, raising an eyebrow,
or reacting silently.

Apache-2.0 code is attractive, and it is worth remembering if an audio-driven
mode ever becomes a feature. It does not compete for the current role.

## B4. Rendering approach comparison

This is the more consequential decision, and it is not purely a model choice.

| | **Neural portrait animation** (LivePortrait class) | **3D avatar + blendshapes** (VRM/glTF) |
|---|---|---|
| Visual quality | Photorealistic; matches a source portrait | Stylised; quality is the artist's, not the model's |
| Latency | Per-frame inference — the dominant cost | Rasterisation — sub-millisecond class |
| Determinism | Output varies with model/version | Fully deterministic |
| VRAM | Hundreds of MB to GB per avatar | Tens of MB |
| Avatar authoring | One portrait image | Full 3D asset production |
| Weight licensing risk | **Real and unresolved** | **None** — no weights involved |
| Failure mode | Artifacts, identity drift | Geometry is always correct |
| Python real-time rendering on Windows | Model handles it | **Weak** — no strong Python VRM renderer; realistically needs Godot/Unity/Unreal or a custom GL/Vulkan renderer |

**These are complementary, not competing.** The architecture already supports
both: `AvatarEngine` is per-avatar, selected by manifest. An avatar can be
neural, another can be 3D, and the pipeline does not care.

The honest position: **neural gives the quality this project wants; 3D gives the
predictability, VRAM economy, and clean licensing it may eventually need.** The
recommendation below reflects that.

---

# Part C — Virtual camera on Windows

This is the area with the **least satisfying options** and the one with the
largest bearing on SHA-256's own license.

## C1. pyvirtualcam

| | |
|---|---|
| Source | `letmaik/pyvirtualcam` |
| License | **GPL-2.0** — verified from the repository |
| Python (3.12/Win) | **Verified available** — `pyvirtualcam 0.15.0` resolves |
| Mechanism | Writes to an **existing** virtual camera. It does not create one. |
| Windows backends | OBS virtual camera (requires OBS installed and started once), or UnityCapture |

**The problem is not technical, it is legal.** pyvirtualcam is GPL-2.0. Linking
it into a distributed application has copyleft implications for the whole work.
For a project whose own license is still undecided, adopting it would make that
decision by default.

It is also not self-contained: the user must install OBS, start the virtual
camera once, and stop it. That is a poor first-run experience for a consumer
application, and it makes SHA-256 dependent on another application's install
state.

**Verdict:** excellent for **development and prototyping**, behind the
`VirtualCameraSink` abstraction and in an optional extra (`vcam-gpl`) that makes
the constraint visible. Not a shipping answer unless SHA-256 chooses to be GPL.

## C2. OBS virtual camera directly

OBS has included a virtual camera since version 26.0. OBS itself is GPL-2.0.
Using it as an *external application* (rather than linking its code) is a
different legal posture from linking pyvirtualcam — but it still requires the
user to install and run OBS.

Also noted: OBS provides a **single camera instance**, so frames cannot be sent
in, mixed, and sent out again through the same device.

## C3. Windows 11 Media Foundation virtual camera — `MFCreateVirtualCamera`

| | |
|---|---|
| Availability | **Windows 11 build 22000+** — not available on Windows 10 |
| Mechanism | Native OS API; creates a real software camera source |
| Formats | RGB32 and NV12 (NV12 typically preferred; conversion via the Media Foundation Video Processor MFT) |
| Lifetime | Session-scoped or system-scoped (persists across reboots) |
| Access scope | Current user or all users |
| Python binding | **None first-class.** Requires a COM/Media Foundation source, realistically a small C++/WinRT or .NET component. |

**This is the architecturally correct answer** and the only one with no
third-party licensing entanglement — it is an operating system API. Microsoft's
own samples (`smourier/VCamSample`, and .NET variants) demonstrate the approach.

**The costs are real:** it needs a native component built and maintained, it is
Windows 11 only, and registering a system-scoped camera touches installation and
permissions. That is meaningfully more work than `pip install pyvirtualcam`.

**Verdict:** the recommended target for a shipping product; **not** the
recommended first implementation.

## C4. UnityCapture

Supports RGBA with transparency, which is genuinely interesting for an avatar
with an alpha channel. Requires separate installation. License and current
maintenance status **not verified** for this evaluation.

---

# Part D — Inference stack

| Technology | License | Windows/Py 3.12 | Assessment |
|---|---|---|---|
| **PyTorch** | BSD-3-Clause | **Verified**: `2.11.0+cu128`, `2.9.0+cu129`, `2.13.0+cu130` | Best for development, conversion, and experimentation. Heaviest deployment footprint (multi-GB). |
| **ONNX Runtime** | **MIT** | **Verified**: `onnxruntime-gpu 1.28.0` | Best deployment runtime: far smaller, faster cold start, CUDA and TensorRT execution providers, no framework lock-in. |
| **TensorRT** | **NVIDIA proprietary EULA** — *not* open source; redistribution is restricted | Available via NVIDIA | Fastest option. Engines are hardware- and version-specific and must be built on the target machine, which complicates distribution. |
| **DirectML** | MIT (ORT provider) | Available | Vendor-neutral GPU path. Slower than CUDA on NVIDIA, but the fallback if AMD/Intel support is ever wanted. |

## The Blackwell / `sm_120` issue — resolved

The reference GPU reports **compute capability 12.0** (`sm_120`). Through 2025
there was a well-documented period where stable PyTorch builds shipped no
`sm_120` kernels, producing the confusing failure mode of `cuda.is_available()`
returning `True` followed by a crash at the first kernel launch.

**That period is over.** CUDA 12.8, 12.9, and 13.0 builds of PyTorch are now
available for Python 3.12 on Windows — verified by dependency resolution against
the official PyTorch indexes.

**However:** availability of a wheel is not proof that it works on this machine.
`scripts/check_gpu.py` exists precisely for this — it inspects
`torch.cuda.get_arch_list()` for the actual `sm_` target and **launches a real
kernel**, because import-time checks do not prove execution works. Run it before
trusting any GPU claim.

---

# Recommendations

Provisional. Each is a starting point for benchmarking, not a commitment — and
the architecture is built so that being wrong here is cheap.

### Primary face tracker — **MediaPipe Face Landmarker**

Apache-2.0 on **both code and weights**, the only candidate emitting blendshapes
directly, ARKit-compatible naming that aligns with 3D avatar tooling, verified
available for Python 3.12 on Windows, and actively maintained by Google. The
open question is GPU acceleration on Windows Python; the i7-14700KF makes CPU
inference a plausible fallback, but that must be measured.

### Backup face tracker — **OpenSeeFace**

BSD-2-Clause on code and models, ONNX-based (so it composes with the recommended
inference stack), and specifically built for robustness in poor conditions —
which is what a real video call is. **Adopting it requires building a
landmark→blendshape solver first.** That caveat is not a footnote.

### Primary avatar engine — **LivePortrait, integrated via FasterLivePortrait**

Best available real-time open-source portrait animation, motion-driven (which
matches this pipeline), with an existing Windows package, ONNX/TensorRT paths,
and a webcam mode — so integration is adapter work, not research.

**Two conditions before any commercial use:**
1. Replace the InsightFace detection models (their weights are non-commercial
   research only — unambiguous).
2. Resolve the licensing of LivePortrait's own weights, which is **UNCLEAR** and
   cannot be settled by reading code.

### Backup avatar engine — **3D avatar + blendshapes**

Not merely a fallback: it is the risk hedge. No model weights means **no weight
licensing risk at all**, VRAM measured in tens of megabytes, fully deterministic
output, and latency in the rasterisation class rather than the inference class.
It costs 3D asset production, and Python real-time 3D rendering on Windows is
genuinely weak — it would likely mean an embedded engine or a custom renderer.

Pairing these two covers the realistic failure modes: if the neural path is too
slow, too heavy, or legally blocked, the 3D path still ships.

### Inference stack — **ONNX Runtime primary, PyTorch for development, TensorRT only after measurement**

MIT-licensed, deployment-sized, with CUDA and TensorRT execution providers
behind one API. PyTorch stays a development dependency for conversion and
experimentation rather than a shipping one. TensorRT is an optimisation to reach
for **after** benchmarking shows ONNX Runtime is insufficient — its engines are
hardware- and version-specific, its license is proprietary, and adopting it
early would trade flexibility for a speedup nobody has demonstrated is needed.

### Rendering — **CPU compositing now; GPU only if measurement demands it**

Compositing at 1280×720 may well be cheap enough on this CPU. Building a GPU
compositing path before measuring the CPU one would be optimising without
evidence.

### Virtual camera — **prototype with pyvirtualcam, target Media Foundation**

Develop against pyvirtualcam behind the `VirtualCameraSink` abstraction, in the
`vcam-gpl` optional extra so its GPL-2.0 status stays visible. Target a native
`MFCreateVirtualCamera` implementation for anything distributed: it is the only
option with no third-party licensing entanglement and no dependency on the user
installing OBS.

This is the decision most likely to need revisiting, and the one with the
largest bearing on SHA-256's own license.

---

# Licensing summary

| Component | Code | Weights | Commercial use |
|---|---|---|---|
| MediaPipe Face Landmarker | Apache-2.0 | **Apache-2.0** | Yes |
| OpenSeeFace | BSD-2-Clause | **BSD-2-Clause** | Yes |
| LivePortrait | MIT | **UNCLEAR** | Only after replacing InsightFace; weights question unresolved |
| FasterLivePortrait | MIT | Inherited (**UNCLEAR**) | Same as LivePortrait |
| InsightFace | MIT | **Non-commercial research only** | **No** |
| Ditto | Apache-2.0 | **UNCLEAR** | Unresolved |
| PyTorch | BSD-3-Clause | n/a | Yes |
| ONNX Runtime | MIT | n/a | Yes |
| TensorRT | **NVIDIA proprietary** | n/a | Per NVIDIA EULA |
| pyvirtualcam | **GPL-2.0** | n/a | Copyleft obligations on distribution |
| OBS | GPL-2.0 | n/a | Copyleft; external-application use differs from linking |
| PySide6 / Qt | **LGPL-3.0** or commercial Qt license | n/a | Yes, with LGPL relinking obligations |

**Three items require a decision before SHA-256 can be distributed
commercially:** LivePortrait's weight license (UNCLEAR), the InsightFace
replacement (required), and the virtual camera approach (GPL vs native).

**This is an engineering assessment, not legal advice.** Anything with money
attached should be reviewed by someone qualified to do so.

---

## Sources

- [MediaPipe Face Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
- [MediaPipe BlazeFace model card](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf)
- [OpenSeeFace](https://github.com/emilianavt/OpenSeeFace)
- [LivePortrait](https://github.com/KwaiVGI/LivePortrait) · [LICENSE](https://raw.githubusercontent.com/KwaiVGI/LivePortrait/main/LICENSE)
- [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait)
- [InsightFace](https://github.com/deepinsight/insightface)
- [Ditto](https://github.com/antgroup/ditto-talkinghead)
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam)
- [MFCreateVirtualCamera](https://learn.microsoft.com/en-us/windows/win32/api/mfvirtualcamera/nf-mfvirtualcamera-mfcreatevirtualcamera) · [VCamSample](https://github.com/smourier/VCamSample)
- [PyTorch get-started](https://pytorch.org/get-started/locally/)
