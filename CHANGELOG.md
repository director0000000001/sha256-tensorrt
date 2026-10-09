# Changelog

All notable changes to sha256 are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions below `1.0.0` carry no API stability guarantee.

---

## [Unreleased]

### Changed

- **Add avatar and Delete avatar without a login (2026-10-09, the owner's
  ask).** Both entries are in the Menu for whoever uses the application; the
  owner's passphrase (`sha256.app.owner`, Menu -> Login) is removed, with its
  tests. An avatar stays a folder under `avatars/` on the user's own computer;
  nothing is uploaded. A `configs/local.owner.json` left from before is no
  longer read.
- **The frame rate (2026-10-08, "The fast version").** The real-time loop runs
  on two threads -- the camera and the tracker on one, the avatar, the output
  conditioning and the virtual camera on the other, a mailbox of one between
  them (`PipelineRunner`, `PipelineController.capture_and_track` /
  `render_tracked`; `performance.overlap_tracking = false` restores the single
  thread). The neural engine keeps the generated face on the GPU from the
  graph to the paste: the face is warped, colour-matched and blended there,
  with the same arithmetic as the CPU paste (17-24 pixels a frame differ, by
  one grey level), and the CPU waits for the GPU once per frame. On the
  owner's machine under a 30 fps camera: 18.8 to 29.4 fps; one more frame of
  latency, about seven milliseconds measured on a clip. The face, the eyes,
  the brows and the mouth are untouched -- proven byte-identical by replay
  (`scripts/probes/frame_budget.py`, `--compare`).

### Added

- **Virtual camera output.** `PyVirtualCameraSink` publishes rendered frames to
  an existing Windows virtual camera device, so the avatar is visible in Meet,
  Zoom, Discord and OBS rather than only in sha256's own preview. Verified by
  round trip: a DirectShow consumer reads back the exact pixels published
  (`docs/benchmarks/B004-virtual-camera.md`).
- Publishing is a checkbox in the window and can be switched during a call —
  `PipelineController.set_publish_output()`. It is off until asked for.
- `sha256.virtual_camera.devices` reports which virtual camera devices are
  registered, by reading the DirectShow capture category rather than by opening
  a device, so the UI can say what to install without briefly advertising a
  camera to every application on the machine.
- `virtual_camera.device` setting: bind to one named device when a machine has
  several.

### Fixed

- The virtual camera sink was never configured by the composition root, so
  `virtual_camera.auto_start = true` failed on the very check that exists to
  catch that ordering mistake. Sinks are now configured where they are wired.
- `docs/development/AVATAR_QUALITY_BACKLOG.md` had been overwritten by a copy of
  `PROJECT_STATUS.md`. Recovered, and brought up to date.

### Notes

- `pyvirtualcam` is **GPL-2.0** and stays in the optional `vcam-gpl` extra. It is
  a development backend and cannot be distributed; the shipping decision is due
  at the day 7 scope review (ADR-0005).

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

