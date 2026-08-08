# B001 — GPU baseline and toolchain viability

**Date:** 2026-08-08
**Commit:** 0478630
**Purpose:** De-risk the largest unknown before building anything — whether the
RTX 50-series (Blackwell, `sm_120`) can run the intended inference stack at all.

This is the first measurement taken by this project. It is a *capability* check,
not a LivePortrait benchmark.

---

## Machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti |
| VRAM | 16303 MiB |
| Compute capability | 12.0 (`sm_120`, Blackwell) |
| Driver | 610.62 |
| CPU | Intel Core i7-14700KF |
| OS | Windows 11 Pro |
| Python | 3.12.13 |

## Software under test

| | |
|---|---|
| PyTorch | 2.11.0+cu128 (CUDA 12.8 build) |
| ONNX Runtime | onnxruntime-gpu 1.28.0 |

---

## Result 1 — `sm_120` kernel support

```
built-in arch list : sm_75, sm_80, sm_86, sm_90, sm_100, sm_120
GPU matmul smoke test: OK
```

**`sm_120` is present, and a real kernel launched and completed.**

This matters because the failure mode here is silent: a wheel built without
`sm_120` still imports, still reports `cuda.is_available() == True`, and only
fails at the first kernel launch. The smoke test in `scripts/check_gpu.py`
exists specifically to catch that, and it passed.

The 2025-era reports of Blackwell being unsupported by PyTorch are resolved.

## Result 2 — Execution providers

```
ONNX Runtime 1.28.0
  providers: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider
```

Both **CUDA** and **TensorRT** execution providers are available. TensorRT is
the path FasterLivePortrait recommends, so the fastest documented route is open
without additional toolchain work.

## Result 3 — Throughput on a portrait-shaped workload

A convolution stack (2 downsampling convs, 6 residual-width convs at 256 channels,
2 transposed convs) run on a 1×3×256×256 input — the resolution and operation
type LivePortrait works at. 20 warm-up iterations discarded, 100 measured.

| Precision | Median | p95 | Implied rate |
|---|---|---|---|
| fp32 | **1.83 ms** | 2.00 ms | 545 fps |
| fp16 | **1.33 ms** | 1.47 ms | 754 fps |

Peak VRAM for the test: **67 MiB**.

---

## Interpretation

**What this proves:**

- The toolchain works end to end on this hardware. The single largest schedule
  risk is cleared.
- The GPU has substantial headroom at 256×256. Against a 33.3 ms budget for
  30 fps, this workload consumes under 2 ms.
- VRAM is not a constraint: 16 GB against a test footprint of 67 MiB, and the
  configured budget of 6144 MiB is generous.
- TensorRT is available, so the optimisation path exists if needed.

**What this does NOT prove:**

- **This is not LivePortrait.** The real system runs several networks —
  appearance extraction, motion extraction, warping, generation — plus stitching
  and retargeting, and pre/post-processing (crop, warp, paste-back, colour
  conversion). It is realistically an order of magnitude heavier than this test.
- Nothing here says anything about **visual quality** or identity stability.
- No end-to-end latency has been measured, because no pipeline exists yet.

**Honest extrapolation:** if the real model is ~10× this workload, it lands
around 15–20 ms and 30 fps is comfortable. If it is ~20×, it is marginal at
fp32 and needs fp16 or TensorRT. Which of those is true is unknown until the
actual weights are run — that is benchmark B002.

For reference, the FasterLivePortrait authors claim 30+ fps on an RTX 3090 with
TensorRT *including* pre- and post-processing. This GPU is in a comparable or
better class for this work, and the same runtime is available here. That claim
is therefore plausible on this machine — but it remains their measurement, not
ours.

---

## Verdict

**Green light.** Proceed with the LivePortrait/ONNX path as the primary avatar
engine. The 3D-avatar fallback is not needed on toolchain grounds.

## Next

**B002 — real LivePortrait weights**, measuring per-stage cost and end-to-end
latency at the target output resolution.

## Reproduce

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install onnxruntime-gpu
uv run python scripts/check_gpu.py
```
