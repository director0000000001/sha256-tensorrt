# Changelog

All notable changes to sha256 are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions below `1.0.0` carry no API stability guarantee.

---

## [Unreleased]

### Added

- **The skin's tone matched (2026-10-14, after the friend test).** Before the
  lighting plane, the neural engine gives the rendered face the person's own
  skin tone: on the cells that are skin on both sides (the mask's core, less
  the dark cells of either picture and the person's brows, eyes and mouth) the
  mean and the spread of each channel are read for the room and for the
  render, and the render gets, per channel, a CAST -- the room's mean over
  the render's, the room's white balance multiplied into every pixel, which
  moves a dark pixel by little and a skin pixel by the whole of it, so the
  irises, the brows and a beard keep their look while the skin takes the
  room's tone -- and a CONTRAST about the room's skin mean that gives the
  skin the room's spread, applied to each pixel by its own lightness against
  the render's skin (none under 0.7 of it, all from 0.9), both within
  0.7..1.4. (A gain and an additive offset on every pixel had turned the
  woman's avatar's brown irises blue-grey; the tone spared on dark pixels had
  left the avatar's stubble pale over hers; an additive chroma had turned the
  beard orange: five replays of her capture.) The plane then fits the light's
  gradient as before, the edge residual as before. Read one frame in four
  (the reading is 2 ms of the render thread's CPU, the apply 0.2), carried by
  an EMA, held with a held face. `skin_match` in `engine_options` (0..1,
  default 1; 0 is the paste before this day, to the byte -- proven 601/601
  on the owner's clip). On the GPU paste it is tensor sums on the device,
  nothing waiting; `tests/unit/test_engine_skin_tone.py`.
- **The forehead to the person's own hairline (2026-10-14).** The join sits
  slightly below where the PERSON's hair starts, and the avatar's upper
  forehead (above 0.3 of a forehead over the brows, past any brow raise) is
  drawn taller to reach it -- moderately, `forehead_stretch_max` 1.25 by
  default, "a medium height" -- its own hair roots' shadow kept out of the
  join by `hair_root_margin` (0.15 of its forehead). A person whose hairline
  cannot be read (bald, or hair past the walk) gets the whole moderate stretch
  and no bound from their side. The stretch is the paste's alone (the render
  sampled closer to the anchor row above it, on the CPU and the GPU alike);
  the drive is untouched. `_forehead_plan`, settled at the anchor with the
  driver's own forehead height against the portrait's;
  `tests/unit/test_engine_forehead_stretch.py`.
- **The head guide in Calibrate me (2026-10-14).** An oval on the preview
  where the head should be -- the distance of every good session (the face
  0.38..0.60 of the frame's height), centred, facing the camera -- amber with
  one line of what to do ("Come closer", "Move back", "Put your head in the
  oval", "Face the camera"), green with "Good. Hold still."; the session's
  clock starts once it has been green for a second (`sha256.ui.head_guide`,
  the preview's `guide`, the window's placing phase). The person is told the
  one thing to fix, so the still step and everything after it are measured
  from the right place. `tests/unit/test_head_guide.py` and the UI tests.

### Changed

- **The forehead join's rule.** Until today the reach took the LOWER of the
  two hairlines less a clearance, so a person with a taller forehead than the
  portrait's got the join on the avatar's hair roots, in the middle of their
  bare forehead -- the stripe of the friend test. Now the driver's oval goes
  to their own hairline less `hairline_clearance`, the portrait's to its
  clean skin (see above), and the paste to the lower of the two as ever. The
  engine's log says both reaches and the stretch at the settle.
  `hairline_clearance` defaults to 0.12 of a forehead now (was 0.04): under
  the person's own hair roots, swept on two captures with the forehead band
  (0.04 / 0.08 / 0.12 -> 4.1 / 3.5 / 3.0 grey levels on the woman's, 8.9 /
  8.4 / 7.5 on his; the untouched build 2.7 / 7.5).
- **`scripts/probes/jaw_band.py`** reads the engine's four-part `last` again
  (the turn's room of day 20 had broken its capture re-render).

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

