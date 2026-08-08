# SHA-256 — Foundation Status

**Date:** 2026-08-08 · **Version:** 0.1.0 · **Status:** foundation complete

This is the verification record for the foundation milestone. Everything below
marked *verified* was executed on the reference machine; the commands are
included so any claim can be re-checked.

---

## 1. Verification results

| # | Check | Command | Result |
|---|---|---|---|
| 1 | Repository structure | — | 44 source files, 15 test files, 21 documents |
| 2 | Environment | `uv sync --all-groups` | Python 3.12.13, all dependencies resolved |
| 3 | Application shell launches | Real Qt event loop, offscreen | **Launched, produced frames, exited 0** |
| 4 | Tests | `uv run pytest` | **180 passed**, 0 failed |
| 5 | Imports | `uv run python scripts/verify_imports.py` | **38/38 modules import cleanly** |
| 6 | Lint | `uv run ruff check .` | **Clean** |
| 7 | Formatting | `uv run ruff format --check .` | **80 files formatted** |
| 8 | Types | `uv run mypy` | **Clean — 59 files, strict mode** |
| 9 | Wiring smoke test | `uv run python -m sha256 --check` | **Wired; 2 avatars discovered** |
| 10 | GPU report | `uv run python scripts/check_gpu.py` | RTX 5070 Ti, 16303 MiB, cc 12.0, driver 610.62 |
| 11 | Documentation | — | All required documents present |
| 12 | Git | `git status` | Clean tree, all work committed |

### Launch evidence

The shell was started under a real `QApplication.exec()` loop, not merely
constructed in a test:

```
window visible      : True
window title        : SHA-256 — foundation build
avatar entries      : 3          (none + 2 sample avatars)
camera entries      : 1
after start         : running
frame produced      : id=0 1280x720 avatar=None
after stop          : stopped
event loop exit code: 0
```

### Hardware verified

`nvidia-smi` on the reference machine:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti |
| VRAM | 16303 MiB |
| Compute capability | 12.0 (`sm_120`, Blackwell) |
| Driver | 610.62 |

### Dependency availability verified

By dependency resolution (`uv pip install --dry-run`) for Python 3.12 on
Windows — not installed, not executed:

`mediapipe 1.0.0` · `onnxruntime-gpu 1.28.0` · `pyvirtualcam 0.15.0` ·
`opencv-python 5.0.0.93` · `torch 2.11.0+cu128` / `2.9.0+cu129` / `2.13.0+cu130`

This settles the `sm_120` question: CUDA 12.8+ PyTorch builds exist for this
platform. The 2025-era reports of Blackwell being unsupported are stale.

---

## 2. What was built

### Core contracts — `src/sha256/core/`

The dependency floor. Imports only the standard library, `numpy`, and
`pydantic`. Nothing here knows what MediaPipe or LivePortrait are.

- Seven component protocols covering every replaceable part.
- Vendor-neutral per-frame data models with a documented rotation convention, a
  documented buffer-ownership rule, and mandatory `scheme` identifiers so two
  incompatible trackers cannot be silently mixed.
- Avatar manifest schema with strict validation and path-traversal rejection.
- Metric result types, exception hierarchy, enumerations.

### Application — `src/sha256/app/`, `src/sha256/pipeline/`

- Component registry: backends selected by config string, instantiated lazily,
  with no import paths in configuration.
- Composition root — the single place concrete backends are chosen.
- `PipelineController`: ordered start with roll-back on partial failure, safe
  stop, single-frame stepping.

### Avatar system — `src/sha256/avatar/`

- Filesystem library, one directory per avatar, per-avatar manifest validation.
- Three validation levels: manifest, asset presence, optional SHA-256 integrity.
- Residency cache: LRU eviction, active-avatar pinning, VRAM budget that refuses
  rather than attempts an oversized load.
- `AvatarManager` coordinating discovery, loading, selection, and observers.

### Placeholder backends

Working implementations rather than stubs that raise — so the application runs
end to end, and any future bug is bisectable by switching a backend to `null`.

### Supporting systems

Performance metrics (rolling windows, nearest-rank percentiles,
timestamp-derived FPS, thread-safe snapshots), layered configuration, path
resolution, logging with rotation and no network handlers, and a PySide6 shell.

---

## 3. What was explicitly not built

Per the milestone scope. Each is absent by decision, not oversight.

| Component | State |
|---|---|
| Face tracking | Not implemented. Placeholder emits synthetic data. |
| Avatar animation / inference | Not implemented. Placeholder draws a schematic face. |
| Camera capture | Not implemented. Placeholder generates frames. |
| Virtual camera output | Not implemented. Placeholder discards frames. |
| Real-time loop | Not implemented. Single-step only. |
| CPU / GPU / VRAM sampling | Interface only; reports "not measured". |
| Benchmarks | None run. **No performance number is claimed anywhere.** |
| Avatar gallery UI | Not implemented. Combo box placeholder. |
| Avatar import tooling | Not implemented. Repository is read-only by design. |

---

## 4. Decisions taken

Recorded in `docs/decisions/`:

| ADR | Decision |
|---|---|
| 0001 | Python 3.12 + uv, ruff, mypy strict, pytest |
| 0002 | Protocols + registry for replaceable backends |
| 0003 | Dataclasses on the hot path, Pydantic at boundaries |
| 0004 | Directory-per-avatar library layout |
| 0005 | **Virtual camera — open** |
| 0006 | PySide6 for the shell |

### Provisional technology recommendations

From `docs/OPEN_SOURCE_EVALUATION.md`. Starting points for benchmarking, not
commitments.

| Role | Recommendation |
|---|---|
| Face tracking (primary) | MediaPipe Face Landmarker — Apache-2.0 code **and** weights; only candidate emitting blendshapes directly |
| Face tracking (backup) | OpenSeeFace — BSD-2-Clause; **requires building a landmark→blendshape solver** |
| Avatar engine (primary) | LivePortrait via FasterLivePortrait — best real-time quality; **licensing conditions apply** |
| Avatar engine (backup) | 3D avatar + blendshapes — no weights, therefore no weight-licensing risk |
| Inference | ONNX Runtime primary; PyTorch for development; TensorRT only after measurement |
| Rendering | CPU compositing until measurement justifies GPU |
| Virtual camera | pyvirtualcam for development; native Media Foundation for distribution |

---

## 5. Deviations from the original brief

Each was a deliberate architectural judgement, documented where it was made.

| Change | Reason |
|---|---|
| `src/sha256/` package rather than loose `src/app`, `src/camera`, … | Makes the project a real installable package; loose top-level directories collide on `sys.path` |
| Added `core/` | The proposal had no home for shared contracts; without one the dependency rule would be unenforceable |
| Added `pipeline/` | Orchestration is neither a component nor UI |
| `avatars/` at repo root, not under `assets/` | User data with its own lifecycle and gitignore rules; `assets/` is application chrome |
| `VirtualCameraSink`, not `VirtualCameraProvider` | From SHA-256's side it consumes frames; the name should match the behaviour |
| Metric result types in `core/`, collection in `performance/` | The monitor protocol must name its return type |

---

## 6. Issues found and fixed during verification

Recorded because they are the evidence that the checks are doing work:

1. **`AvatarManager.refresh()` never re-scanned.** It was documented as
   re-scanning, but the repository caches and `refresh()` called `discover()`
   without forcing. An avatar added while running would have stayed invisible.
   Found by `mypy --strict` flagging the private-attribute workaround the tests
   had grown to compensate. Fixed by adding `force` to the protocol.
2. **`PerformanceMonitor.snapshot()` returned `object`.** An attempt to keep the
   protocol free of a concrete type made it unusable without casts. Fixed by
   moving the result types into `core/metrics.py`.
3. **Settings warnings used undeclared attribute injection**, which is fragile
   under Pydantic v2. Replaced with a declared `PrivateAttr`.
4. **An integration test busy-looped** faster than the placeholder camera's
   frame interval, so a frame was never due. Fixed to poll with a sleep.

---

## 7. Known limitations

- **`LICENSE` is a restrictive placeholder.** No project license has been
  chosen. This is a decision awaiting the owner, and it interacts with ADR-0005.
- **LivePortrait's model-weight license is UNCLEAR**, and its InsightFace
  dependency ships weights restricted to non-commercial research. Both must be
  resolved before commercial use.
- **The virtual camera approach is unresolved** and bears on the license choice.
- **The residency policy has never evicted a real model.** It is unit-tested
  against simulated pressure only.
- **Avatar loading runs on the UI thread** (`TODO(threading)` marks the site).
  Harmless with the placeholder engine; would freeze the window with a real one.
- **MediaPipe GPU acceleration on Windows Python is unverified.** Assume CPU
  inference until measured.
- **The renderer does not scale images** — it letterboxes and crops.
- **No performance data exists.** The debug panel shows `n/a`, deliberately.

---

## 8. Repository at a glance

```
44 source files      3,737 lines
15 test files        1,274 lines   180 tests
21 documents         2,351 lines
```

```
sha256/
├── src/sha256/
│   ├── core/            contracts — the dependency floor
│   ├── app/             composition root + registry
│   ├── pipeline/        lifecycle control
│   ├── avatar/          library, residency, engines
│   ├── camera/ tracking/ rendering/ virtual_camera/
│   ├── performance/     metric collection
│   ├── config/          schema, loader, paths
│   ├── ui/              PySide6 shell
│   └── utils/           logging, timing
├── tests/{unit,integration,fixtures}/
├── avatars/             2 sample avatars
├── models/ configs/ scripts/ assets/ logs/
├── docs/{architecture,decisions,benchmarks,development}/
└── .github/workflows/
```

---

## 9. Suggested next milestone

Not started — awaiting explicit assignment. Ordered so each step yields a
measurement that informs the next:

1. **Camera capture.** The only stage with no dependency on an AI decision, and
   it makes everything downstream testable against real input.
2. **The real-time loop.** Threading, frame pacing, drop policy — validating the
   design in `SYSTEM_ARCHITECTURE.md` §6.
3. **MediaPipe adapter**, then benchmark it. Settles the CPU-versus-GPU question
   that the evaluation could not.
4. **Virtual camera** (ADR-0005), which needs the license decision first.
5. **Avatar engine**, which is the largest unknown and benefits most from
   having everything else measured first.

---

## 10. Stop condition

The foundation is complete and verified. Per the milestone scope, work stops
here.

**Not started, deliberately:** the face-tracking pipeline, MediaPipe
integration, LivePortrait integration, virtual camera implementation.
