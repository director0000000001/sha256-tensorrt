# B006 — TensorRT for the two pixel networks: measured, outside the frame path

**Date:** 2026-10-09 · **Branch:** `tensorrt` (a copy of the repo at 468b5ba, made for this one experiment)
**Question:** "Phase 2 -- TensorRT" of *The fast version* (ROADMAP): the warping module
and the SPADE decoder compiled to a TensorRT fp16 engine, the motion extractor and every
line of `_drive` left in torch. B005 (2026-08-14) stopped this route on three blockers --
the `grid_sample_3d` plugin, the CUDA runtime DLLs, a Python 3.10 wheel. Are they still
there, what does the engine give in milliseconds, and what does it do to the pixels?

**Answer: the blockers are gone, the model's GPU time falls from 25.8 to 14.3 ms a frame
(1.81x; the two networks alone 23.5 to 12.1 ms, 1.94x), the expression is untouched to the
byte, and the drawn pixels differ by fp16 rounding -- less than torch's own fp16 differs
from fp32.** Everything below is measured on this machine; the estimates for other GPUs are
marked as such. Nothing in `engine_liveportrait.py` was changed; stage 2 (integration) is a
separate decision.

The instrument is `scripts/probes/tensorrt_probe.py` (export, build, check, clip, bench,
diff, kp). Its JSONs are under `recordings/budget/trt/` (gitignored), the ONNX files and
engines under `models/tensorrt/` (gitignored). The stills of his face that the clip runs
wrote there were read through `still_diff` and deleted.

---

## 1. The environment, and B005's three blockers today

| | |
|---|---|
| machine | i7-14700KF, RTX 5070 Ti 16 GB, driver 610.88, Windows 11 Pro 10.0.26200 |
| Python / torch | 3.12.13; torch 2.11.0+cu128, cuDNN 9.19 (installed outside the lock, as before) |
| TensorRT | **11.3.0.99**, the `tensorrt-cu12` wheels: `tensorrt-cu12-bindings` (cp312 win_amd64, PyPI) and `tensorrt-cu12-libs` (win_amd64, from `https://pypi.nvidia.com`; the PyPI entry is a 15 KB shim) |
| exporter side | onnx 1.23.2, onnxscript 0.7.2, onnx-ir 1.0.0, ml-dtypes 0.6.0 |
| install | `uv pip install "tensorrt-cu12==11.3.0.99" "onnx==1.23.2" "onnxscript==0.7.2" --extra-index-url https://pypi.nvidia.com` -- 4 min 47 s, a 2.1 GiB download |
| not installed, not needed | a CUDA toolkit (no `nvcc` on this machine), MSVC, cmake, onnxruntime's providers, any plugin |

Why 11.3 and not 11.4: 11.4.0.106 was released on 2026-10-08 but its `tensorrt-cu12-libs`
had no Windows wheel on NVIDIA's index that day (11.3.0.99 has one). The cu12 variant is
built against CUDA 12.9 and declared compatible with CUDA 12.x; it loaded and ran in the same
process as torch's cu128 DLLs with nothing else installed. Rejected on the way:
`torch-tensorrt` -- 2.14 wants torch 2.14 and CUDA 13, and 2.11.0+cu128 pins TensorRT
10.15, which has no 3-D GridSample.

| B005's blocker (2026-08-14) | today |
|---|---|
| `warping_spade-fix.onnx` carries a custom `grid_sample_3d` node; only a plugin DLL built against the installed TensorRT could run it | **gone.** TensorRT 11.2.1 "extends GridSample from 2D-only to 3D (rank-5 input) with FP32, FP16, and BF16; linear, nearest, and cubic interpolation; align_corners 0/1; and padding modes zeros, border, and reflection". Our own ONNX export (opset 20) carries the two 5-D `GridSample` nodes as standard ops; the parser took both as `GRID_SAMPLE` layers. No plugin, no compiler |
| onnxruntime's CUDA provider wanted `cublasLt64_13.dll`; TensorRT not installed | **not in the path any more.** onnxruntime is not used at all. TensorRT 11 needs neither cuBLAS nor cuDNN; its wheel's DLLs loaded beside torch's |
| FasterLivePortrait's runtime ships as a cp310 wheel | **irrelevant.** The ONNX is exported from this repo's own vendored modules and weights with torch 2.11 on Python 3.12 |

One real risk stays, from the 11.3.0 release notes: *"Creating an execution context for a
GridSample FP16 network may fail on Windows with NVIDIA RTX 5080 (CUDA error 700)"*, fixed
in 11.4. It did **not** occur here (RTX 5070 Ti, 610.88): the fp16 engine's context was
created and ran 601-frame passes five times. It is not a proof for other RTX 50 cards; the
probe carries the fallback (`export --grid-sample-fp32`, the two grid_samples in fp32), unused.

## 2. Export and build

The two networks are exported as **one fused module** (`f_s`, `kp_source`, `kp_driving` in,
the 3x512x512 picture out -- `_drive`'s last two lines), with the engine's own fp16 weights,
batch 1, the fixed shapes the engine uses: `f_s` (1, 32, 16, 64, 64), keypoints (1, 21, 3).

| file | exporter | nodes | size | export time |
|---|---|---|---|---|
| `fused-fp16.onnx` | dynamo (`torch.export` + onnxscript), opset 20 | 377 (2 GridSample, 90 Conv, 16 InstanceNormalization) | 204.0 MB | 5.0 s |
| `fused-fp32.onnx` (precision reference) | torchscript, opset 20 | 801 | 526.0 MB | 3.1 s |
| `warping-fp16.onnx`, `spade-fp16.onnx` (for the split) | dynamo | 111, 266 | 92.6, 111.4 MB | 3 s each |

Two exporter quirks, both worked around in the probe: the dynamo exporter prints a check
mark and dies under the Windows console's cp1252 (`PYTHONUTF8=1` fixes it); the torchscript
exporter cannot export the fp16 modules ("tensor does not have a device") but exports fp32.

TensorRT 11 builds every network **strongly typed** -- the ONNX file's own precision, op by
op, which is also what torch does. The TensorRT 10 recipe "fp32 ONNX + the FP16 builder
flag" no longer exists: 11.3 has no weakly typed network flag and no `BuilderFlag.FP16`.
So the fp16 engine is the fp16 export, and the fp32 engine is the fp32 export.

| engine | parse | build (this GPU) | layers | plan on disk | activations | weights |
|---|---|---|---|---|---|---|
| **`fused-fp16.plan`** | 0.2 s | **44.9 s** | 634 | 328.1 MB | 189 MB | 326 MB |
| `fused-fp32.plan` | 0.5 s | 49.1 s | 1523 | 410.5 MB | 428 MB | 409 MB |
| `warping-fp16.plan` | 0.1 s | 21.1 s | 134 | 94.4 MB | | |
| `spade-fp16.plan` | 0.1 s | 26.4 s | 500 | 233.8 MB | | |

Loading an engine (deserialise + execution context) takes 0.19-0.21 s. Engines are specific
to the GPU's compute capability and the TensorRT version: on a user's machine this is a
45-second first run, or a prebuilt plan per GPU family (section 7).

The TensorRT enqueue (`execute_async_v3` on torch's current stream) **captures into
torch's CUDA graph**; replay equals eager to the bit. So the engine's own `_capture` recorded
its graph around the TensorRT engine exactly as it does around the torch modules, with the
probe swapping the two `_nets` entries in its own process before the avatar loaded. The
graph the application replays is then: motion extractor and `_drive` in torch, one TensorRT
enqueue, the uint8 conversion.

## 3. The proofs: the expression untouched, the runs reproducible

All on `recordings/driving.mkv` frames 300-900 (601 frames, his face), through the
application's own `PipelineController`, `--deterministic` (cuDNN's benchmark off, as in the
2026-10-08 proofs).

| what | result |
|---|---|
| the torch pass with the probe's keypoint-recording wrapper, against `det-gpufit3.json` (the reference hashes of the unchanged code) | **601 of 601 frames byte-identical** -- the wrapper changes nothing |
| `kp_driving` handed to the warping network, torch pass against TensorRT pass (fp16, 1x21x3, every frame) | **601 of 601 byte-identical, largest difference 0** -- the drive path is the same path |
| the torch pass run twice | 601 of 601 identical |
| the TensorRT pass run twice | 601 of 601 identical -- the engine is deterministic run to run |
| the composite, torch against TensorRT | 601 of 601 differ (the pixels; section 5) |

## 4. Milliseconds

### 4.1 The two networks alone (600 recorded keypoint sets, CUDA events, `cudnn.benchmark` on)

The bench feeds both implementations the very keypoints the torch pass recorded per frame,
with the avatar's own `f_s` and `x_s`.

| the two networks | p50 | p90 | p99 |
|---|---|---|---|
| torch fp16, eager | 25.96 | 27.21 | 28.05 |
| **torch fp16, one CUDA graph (the application's)** | **23.47** | **24.21** | 24.69 |
| torch warping alone / SPADE alone, eager | 8.78 / 16.87 | 9.38 / 17.50 | |
| torch fp32, eager (precision reference) | 50.97 | 54.35 | 56.82 |
| TensorRT fp16, eager | 12.74 | 13.43 | 13.93 |
| **TensorRT fp16, inside a CUDA graph** | **12.09** | **12.55** | 13.24 |
| TensorRT warping alone / SPADE alone, eager | 5.34 / 7.32 | 5.43 / 7.35 | |
| TensorRT fp32, inside a CUDA graph | 36.69 | 37.10 | 37.47 |

**1.94x** on the two networks. The SPADE decoder gains most (16.9 to 7.3 ms, 2.3x); the
warping module 8.8 to 5.3 (1.6x). TensorRT fp32 is 37 ms -- the fp16 engine is the only
one that fits.

### 4.2 The whole frame through the application (one thread, `asap` pacing, hair off, 571 frames after a 30-frame warm-up)

`model` is the graph's own GPU time from the engine's CUDA events; `launch` the CPU's cost of
launching it; `paste` includes the one wait for the graph; `total` is `process_once`.

| ms per frame | torch (today) p50 / p90 / p99 | **TensorRT fp16** p50 / p90 / p99 |
|---|---|---|
| tracker (CPU) | 11.02 / 12.27 / 12.50 | 11.11 / 12.16 / 12.39 |
| crop | 0.48 / 0.51 | 0.48 / 0.51 |
| launch | 2.32 / 2.47 | 2.05 / 2.17 |
| **model (the graph on the GPU)** | **25.81 / 26.67 / 27.23** | **14.26 / 14.88 / 15.38** |
| mask | 2.39 / 16.24 | 2.35 / 12.33 |
| paste, with the wait for the graph | 24.30 / 25.39 | 12.92 / 13.70 |
| animate (CPU, the engine's whole call) | 29.71 / 30.58 / 31.44 | 18.09 / 19.71 / 28.29 |
| **frame, one thread** | **43.22 / 44.99 / 46.22** | **31.90 / 33.90 / 41.88** |

The graph: 25.8 to 14.3 ms, **1.81x**; what stays is the motion extractor and the drive
arithmetic, about 2.2 ms, and they are the same kernels. The torch figure reproduces the
2026-10-08 measurement (26.0). Deterministic mode (cuDNN benchmark off) reads 26.37 and
14.24. The fp32 engine through the same path: 39.30 ms.

### 4.3 The application's two threads at a 30 fps camera (`--mode threads --pace realtime`)

| | torch | **TensorRT fp16** |
|---|---|---|
| frames published of 600 | 450 (150 dropped at capture) | **506** (94 dropped) |
| frame rate, the 2026-10-08 formula (publish interval after warm-up) | 26.6 fps (p50 37.4 ms, p90 39.7) | **30.1 fps** (p50 33.1, p90 46.4) |
| frames published a second over the whole clip, start-up included | 22.7 | 25.6 |
| render thread per frame (`animate`) | 34.3 p50 / 36.4 p90 | **21.5 p50** / 33.8 p90 |
| render frames over 33.3 ms | 355 of 420 | 74 of 476 |
| latency, capture to render | 66.1 p50 / 79.9 p90 | **37.7 p50** / 50.0 p90 |

Two readings of this table. The torch row is 26.6 where the same code measured 29.4 on
2026-10-08: this session's tracker ran at its slow mode (12.9 ms, the bimodal tracker of the
ROADMAP) and the probe's own bookkeeping on the render thread (a synchronise and the
keypoint copy per frame) is on top -- the two rows here are like for like, and the
difference between them is the engine. And what is left over 33 ms with TensorRT is not the
model: the 74 slow frames are the **hairline read** (`_luminance` over the whole 1280x720,
every 6th frontal frame, 17-20 ms; `mask` p90 21 ms in the table). It was moved behind the
GPU's wait on 2026-10-08 and hid there while the graph took 26 ms; at 14 ms it sticks out.
Stage 2 should move it to the capture thread or make it cheaper -- then the render thread
is 21 ms with 12 of margin.

## 5. The pixels

Grey levels, 0..255, `still_diff`'s numbers: the pixels that differ, the largest difference,
the 99th percentile over the differing pixels and over the whole image, the mean over the
whole image. Deterministic passes, so the torch side is the application's own output to the
byte (section 3).

### 5.1 The 512x512 drawn face, every frame (the bench, 601 frames)

| face, against ... | max over frames (per-frame max, p50) | p99 whole face | mean abs, p50 (max) | pixels differing, p50 of 262,144 |
|---|---|---|---|---|
| **TensorRT fp16 vs the application's torch fp16** | **38 (11)** | **1** | **0.094 (0.115)** | **54,028 (20.6%)**, max 59,753 |
| torch fp32 vs torch fp16 -- fp16 rounding's own footprint | 38 (15) | 2 | 0.155 (0.217) | 60,820 (23.2%) |
| TensorRT fp32 vs torch fp32 | 4 (1) | 1 | 0.007 (0.008) | 5,311 (2.0%) |
| TensorRT fp16 vs torch fp32 | 66 (17) | 2 | 0.180 (0.242) | 72,144 (27.5%) |

Read together: the fp32 engine reproduces torch's fp32 arithmetic to 4 levels at most and
0.007 mean -- the network TensorRT runs is the same network. The fp16 engine differs from
torch's fp16 by *less* than torch's own fp16 differs from fp32 (0.094 against 0.155 mean;
20.6 against 23.2% of pixels). What differs is where each implementation rounds fp16 --
kernel order, fusion -- not what it computes.

### 5.2 Nine stills across the clip (five from the 2026-10-08 set, four at the clip's largest keypoint moves)

Frames 314, 470, 627, 767 are the four largest frame-to-frame keypoint displacements in
the recorded `kp_driving` (314 is the anchor settling; 470, 627, 767 are turns and speech);
330-340 is the pose furthest from the clip's median.

| frame | face: px differ | max | p99 of differing | mean abs | composite: px differ (of 921,600) | max | p99 of differing | mean abs |
|---|---|---|---|---|---|---|---|---|
| 314 | 48,800 (18.6%) | 3 | 1 | 0.075 | 37,910 (4.11%) | **2** | 1 | 0.017 |
| 330 | 50,298 (19.2%) | 19 | 2 | 0.084 | 42,765 (4.64%) | 3 | 2 | 0.020 |
| 450 | 53,775 (20.5%) | 8 | 2 | 0.091 | 44,835 (4.86%) | 3 | 2 | 0.022 |
| 470 | 56,030 (21.4%) | 27 | 2 | 0.107 | 49,243 (5.34%) | **6** | 2 | 0.027 |
| 600 | 53,083 (20.3%) | 7 | 2 | 0.094 | 47,503 (5.15%) | 5 | 2 | 0.025 |
| 627 | 54,092 (20.6%) | 27 | 2 | 0.099 | 46,082 (5.00%) | 4 | 2 | 0.023 |
| 700 | 53,556 (20.4%) | 12 | 2 | 0.092 | 45,297 (4.92%) | 4 | 2 | 0.022 |
| 767 | 54,839 (20.9%) | 15 | 2 | 0.095 | 47,132 (5.11%) | 4 | 2 | 0.024 |
| 850 | 52,277 (19.9%) | 12 | 2 | 0.089 | 43,039 (4.67%) | 4 | 2 | 0.020 |

Where the face's large differences are: in frame 330 the histogram of per-pixel differences
is 211,846 equal, 49,521 by one level, 538 by two, 161 by three, 50 by four, 21 by five and
**7 pixels by six or more**; those seven (3 in frame 700, 6 in frame 850) all sit at rows
5-7, column 14 of the 512 square -- the generator's saturated border artefact in the crop's
margin, on a local gradient of 141-223 levels -- outside the paste oval. In the composite
nothing differs outside the paste box, the largest difference is 6, the 99th percentile of
the differing pixels is 2, the mean over the frame 0.017-0.027.

For scale, from the 2026-10-08 work: the GPU paste against the CPU paste differed on 17-24
pixels a frame by one level; the application's *own* run-to-run difference with
`cudnn.benchmark` on was about 18,000 pixels a frame by 1-3 levels (2% of the frame). The
engine's difference is 4-5% of the frame, almost all by one level, at most 6. The plan's
tolerance for the paste (at most 2 at p99, 0.5 mean) is met on every still; the count of
pixels touched is of a different order than the paste's, because every pixel of the face is
drawn by a different fp16 kernel schedule.

The fp32 engine's stills, for completeness: face mean 0.148-0.190 (max 15-33), composite
6.4-7.0% of pixels, max 6-11, p99 of differing 2-4, mean 0.035-0.044 -- further from the
application than the fp16 engine is, as 5.1 says.

## 6. What it costs: the package, the first run, the memory

Measured in this venv (`site-packages` 5,828,812,308 bytes before, 9,323,540,650 after:
**+3.49 GB** for TensorRT and the exporter side).

| what | size |
|---|---|
| `tensorrt_libs/` in all | 3,465 MB |
| of which `nvinfer_11.dll` -- the runtime, the one DLL a prebuilt engine needs | 420 MB |
| `nvinfer_plugin_11.dll` (not used by these engines), `nvonnxparser_11.dll` (build only) | 58 MB, 2.3 MB |
| `nvinfer_builder_resource_*_11.dll` -- build only, one per GPU family: sm120 (RTX 50) / sm89 (RTX 40) / sm86 (RTX 30) / sm75 / sm80 / sm90 / sm100 / ptx | 402 / 268 / 255 / 169 / 270 / 678 / 437 / 506 MB |
| `tensorrt_bindings`, onnx + onnxscript + onnx-ir + ml-dtypes | 3.2 MB, 29 MB |
| the fp16 ONNX (needed only to build) / the fp16 engine | 204 MB / 328 MB |
| the engine at run time on the GPU | 189 MB activations + 326 MB weights (the torch fp16 weights of the two networks, 210 MB, can be dropped once the engine is loaded; the torch fallback would reload them) |
| the engine build on this GPU | 45 s, once per GPU family and TensorRT version |

So the package grows by about **0.75 GB** if engines are shipped prebuilt per GPU family
(`nvinfer_11.dll` + one 328 MB plan each for sm120, sm89, sm86 -- a further 0.66 GB for the
three), or by about **1.0 GB** if the engine is built on the user's machine at first run
(`nvinfer_11.dll`, `nvonnxparser_11.dll`, the fp16 ONNX, and the one builder-resource DLL
for the GPU families the build supports; 45 s on a 5070 Ti, longer on a slower card, and a
new way for a user's machine to fail). The 2026-10-08 plan's "about 1 GB" holds.

## 7. What it means for a weaker card (estimates, marked as such)

On this machine the model was 25.8 of the render thread's ~31 ms (the ROADMAP's "~85%");
TensorRT makes it 14.3, and the render thread 21.5 ms, 12 under the camera's 33.3.

| GPU | today's code, model (ROADMAP's estimate) | with TensorRT fp16 (the same scaling) | render thread, two threads |
|---|---|---|---|
| RTX 5070 Ti (this machine, measured) | 25.8 | **14.3** | **21.5 ms, 30 fps** |
| RTX 5060 Ti class (Blackwell, half the SMs, half the bandwidth) | ~45-52 | ~26-29 | ~33-36 ms: 28-30 fps |
| RTX 4060 Ti 16 GB (the ROADMAP's 2-2.5x) | 52-65 | ~29-36 | ~36-43 ms: 23-28 fps |
| RTX 3060 12 GB (the written minimum) | ~65-80 | ~36-45 | ~43-52 ms: 19-23 fps |

The scaling assumes TensorRT's gain (1.81x on the graph) carries over; on Ampere and Ada it
usually does for convolution-bound fp16 work, but the hairline read and the tracker are CPU
costs that do not scale with the GPU. **Step 5's measurement on a real 4060 Ti / 5060 Ti
replaces every row but the first.** The one thing TensorRT certainly changes for such a card
is which side is slower: with the model at ~30 ms the CPU's mask, paste prep and the hairline
read are no longer hidden, so stage 2's CPU housekeeping matters more there than here.

## 8. Recommendation

Stage 2 is worth doing, with the **fused fp16 strongly-typed engine** and these terms:

1. The torch path stays as it is and is the fallback: no engine for this GPU and TensorRT
   version, a failed build, the 11.3 context-creation issue on another RTX 50, an older
   GPU. The engine is loaded lazily after `_load_networks`, swapped into the same two
   `_nets` entries the probe swapped, and the engine's `_capture` records the graph as it
   does today -- the probe already proved that path.
2. The proof of the expression is the keypoint byte-identity (section 3), not the frame
   hash: the frame hashes cannot match across torch and TensorRT by design.
3. The hairline read moves off the render thread (section 4.3); otherwise a sixth of the
   frames still miss the 33 ms budget on this machine.
4. Prefer TensorRT 11.4's Windows wheel when NVIDIA publishes it (the GridSample fp16 fix);
   the engine must then be rebuilt -- plans are tied to the TensorRT version.
5. Ship prebuilt plans per GPU family (sm120, sm89, sm86) rather than build on first run,
   unless the 0.66 GB matters more than the first-run minute and its failure modes.

The pixel change this buys is fp16 rounding, measured: at most 6 grey levels in the
composite on nine stills, 99% of the differing pixels within 2, a mean of 0.02 over the
frame; and it is the only change -- the keypoints are the same bytes. His own look at the
live picture is still the last word, as it was for the GPU paste.

## Appendix: how to repeat it

```
uv pip install "tensorrt-cu12==11.3.0.99" "onnx==1.23.2" "onnxscript==0.7.2" --extra-index-url https://pypi.nvidia.com
set PYTHONUTF8=1
python scripts/probes/tensorrt_probe.py export --parts fused --precision fp16 --exporter dynamo
python scripts/probes/tensorrt_probe.py build --onnx models/tensorrt/fused-fp16.onnx
python scripts/probes/tensorrt_probe.py check --plan models/tensorrt/fused-fp16.plan
python scripts/probes/tensorrt_probe.py clip --nets torch --deterministic --tag torch-det --compare recordings/budget/det-gpufit3.json --stills 330,450,600,700,850
python scripts/probes/tensorrt_probe.py clip --nets trt --plan models/tensorrt/fused-fp16.plan --deterministic --tag trt16-det --compare recordings/budget/trt/torch-det.json --stills 330,450,600,700,850
python scripts/probes/tensorrt_probe.py kp recordings/budget/trt/torch-det-kp.npy recordings/budget/trt/trt16-det-kp.npy
python scripts/probes/tensorrt_probe.py clip --nets torch --tag torch-time
python scripts/probes/tensorrt_probe.py clip --nets trt --plan models/tensorrt/fused-fp16.plan --tag trt16-time
python scripts/probes/tensorrt_probe.py bench --kp recordings/budget/trt/torch-det-kp.npy --plan models/tensorrt/fused-fp16.plan --plan models/tensorrt/fused-fp32.plan
python scripts/probes/tensorrt_probe.py diff --a torch-det --b trt16-det --frames 330,450,600,700,850
python scripts/probes/tensorrt_probe.py clip --nets trt --plan models/tensorrt/fused-fp16.plan --mode threads --pace realtime --tag trt16-rt
```
