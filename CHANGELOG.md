# Changelog

All notable changes to sha256 are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions below `1.0.0` carry no API stability guarantee.

---

## [Unreleased]

Nothing yet. The next milestone will be assigned explicitly.

---

## [0.1.0] — 2026-08-08

Project foundation. **No avatar functionality works yet** — this release is the
structure the real pipeline will be built into.

### Added

**Core contracts** (`sha256.core`)
- Component protocols: `CameraProvider`, `FaceTracker`, `AvatarEngine`,
  `Renderer`, `VirtualCameraSink`, `PerformanceMonitor`, `AvatarRepository`.
- Vendor-neutral data models: `VideoFrame`, `FrameMetadata`, `LandmarkSet`,
  `HeadPose`, `BlendshapeSet`, `FaceTrackingResult`, `RenderedFrame`.
- Avatar manifest schema with strict validation and path-traversal rejection.
- Metric result types, exception hierarchy, shared enumerations.

**Application**
- Component registry: backends selected by config string, instantiated lazily.
- Composition root wiring the full object graph.
- `PipelineController` for component lifecycle with roll-back on partial start,
  plus single-frame stepping. **No real-time loop.**

**Avatar system**
- Filesystem library, one directory per avatar, discovered at startup.
- Three-level validation: manifest, asset presence, optional SHA-256 integrity.
- Residency cache: LRU eviction, active-avatar pinning, VRAM budget that refuses
  rather than attempts an oversized load.
- `AvatarManager` coordinating discovery, loading, selection, and observers.

**Placeholder backends** — working implementations, so the app runs end to end
and the wiring is testable without hardware or model weights.
- Synthetic camera, synthetic tracker, procedural engine, CPU letterbox
  renderer, discarding virtual-camera sink.

**Performance**
- Rolling-window metrics with nearest-rank percentiles, timestamp-derived FPS,
  dropped-frame counting, thread-safe snapshots.
- System resource probe interface reporting "not measured" rather than zero.

**Configuration and infrastructure**
- Layered configuration: defaults → TOML → local TOML → env/`.env` → overrides.
- Development/installed path resolution.
- Stdlib logging with rotation and optional JSON output. No network handlers.
- PySide6 shell: avatar selector, camera selector, Start/Stop, developer panel.
- `uv` + Python 3.12, ruff, mypy (strict), pytest, pre-commit, GitHub Actions CI
  on Windows with a gitleaks history scan.
- `scripts/check_gpu.py` — reports real GPU capability and launches an actual
  kernel, because import-time checks do not prove execution works.

**Documentation**
- System architecture, avatar system, open-source evaluation, security, privacy,
  performance plan, six ADRs, contributing guide, foundation status.

### Verified

- 180 tests pass.
- `mypy --strict` clean across 59 source files.
- `ruff check` and `ruff format --check` clean.
- `python -m sha256 --check` wires the application and discovers both sample
  avatars.
- GPU report confirms RTX 5070 Ti, 16303 MiB, compute capability 12.0, driver
  610.62.
- Package availability for Python 3.12 on Windows confirmed by dependency
  resolution for `mediapipe`, `onnxruntime-gpu`, `pyvirtualcam`,
  `opencv-python`, and CUDA 12.8/12.9/13.0 builds of PyTorch.

### Not included — deliberately

Face tracking, avatar animation, camera capture, virtual camera output, the
real-time processing loop, CPU/GPU/VRAM sampling, benchmark results, and the
avatar gallery UI. No performance figure is claimed anywhere in this release.

### Known limitations

- `LICENSE` is a restrictive placeholder; the project license is undecided.
- The virtual camera approach is unresolved (ADR-0005) and bears on that choice.
- LivePortrait's model-weight license is **unclear**, and its InsightFace
  dependency ships non-commercial-only weights. See
  `docs/OPEN_SOURCE_EVALUATION.md`.
- The residency policy is unit-tested against simulated pressure but has never
  evicted a real model.
- Avatar loading still runs on the UI thread (marked `TODO(threading)`).

