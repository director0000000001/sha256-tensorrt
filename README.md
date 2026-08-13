# sha256

A local-first, real-time AI avatar application for Windows.

sha256 captures a webcam feed, tracks the user's face, drives a fictional
AI-generated avatar with that motion, and publishes the result as a virtual
camera that Google Meet, Zoom, Discord, or OBS can consume like any other
webcam.

```
Webcam → Face Tracking → Facial Motion / Expression Data
       → Avatar Animation → Rendering → Virtual Camera
       → Meet / Zoom / Discord / OBS
```

Webcam frames, facial data, and avatar assets stay on the local machine. There
is no cloud service in the real-time path, and the application makes no network
requests at all in its current form. See [`docs/PRIVACY.md`](docs/PRIVACY.md).

---

## Current status: the pipeline runs end to end

The whole path described above works: a camera frame is tracked, an avatar is
warped onto the face, and the result is published as a webcam other applications
can select. Quality is the open work, not connectivity.

| Area | Status |
|---|---|
| Repository structure, tooling, CI | Implemented |
| Component contracts (`sha256.core`) | Implemented |
| Data models (landmarks, pose, expression, frames) | Implemented |
| Configuration, logging, paths | Implemented |
| Avatar library: discovery, validation, residency, selection | Implemented |
| Performance metric collection | Implemented |
| Application shell (PySide6) | One screen, live preview |
| Camera capture | Implemented — DirectShow via OpenCV (B002) |
| Face tracking | Implemented — MediaPipe Face Landmarker (B003) |
| Avatar animation | Implemented — landmark warp, 22 photograph roles |
| Real-time processing loop | Implemented — one capture thread, newest frame wins |
| Virtual camera output | **Development backend only** — see below (B004) |
| **Photorealistic synthesis** | **Not implemented.** Blocked on model-weight licences |

**The virtual camera cannot ship as built.** The working backend is
`pyvirtualcam`, which is GPL-2.0 and needs OBS Studio installed for its device
to exist. It is an optional extra, never a default, and a licence-clean
replacement is an open decision — see
[`docs/decisions/ADR-0005-virtual-camera-strategy.md`](docs/decisions/ADR-0005-virtual-camera-strategy.md).

Every performance number in this repository traces to a numbered benchmark in
[`docs/benchmarks/`](docs/benchmarks/), reported with its hardware, sample count
and percentiles. Nothing is estimated.

---

## Target architecture

Every replaceable part is defined as a protocol in
[`src/sha256/core/interfaces.py`](src/sha256/core/interfaces.py), and backends
are selected by a registry id in configuration. No application code imports
MediaPipe, LivePortrait, PyTorch, or any other vendor library — adapters
translate into the neutral types in
[`src/sha256/core/types.py`](src/sha256/core/types.py) at their own boundary.

```
CameraProvider ─► VideoFrame ─► FaceTracker ─► FaceTrackingResult
                                                     │
                                        AvatarManager ▼
                     AvatarRepository ──► AvatarEngine ─► RenderedFrame
                                                     │
                                              Renderer ▼
                                          VirtualCameraSink ─► OS
```

Full design: [`docs/architecture/SYSTEM_ARCHITECTURE.md`](docs/architecture/SYSTEM_ARCHITECTURE.md)
and [`docs/architecture/AVATAR_SYSTEM.md`](docs/architecture/AVATAR_SYSTEM.md).

The technology evaluation behind the (provisional) backend choices is in
[`docs/OPEN_SOURCE_EVALUATION.md`](docs/OPEN_SOURCE_EVALUATION.md).

---

## Hardware target

Development and benchmarking target a single reference machine:

| | |
|---|---|
| CPU | Intel Core i7-14700KF |
| GPU | NVIDIA GeForce RTX 5070 Ti — 16 GB VRAM, Blackwell, compute capability 12.0 (`sm_120`) |
| RAM | 32 GB |
| OS | Windows 10 / 11 |
| Python | 3.12 |

The `sm_120` compute capability matters: it needs a CUDA 12.8-or-newer build of
any GPU framework. Verify with `python scripts/check_gpu.py` before assuming a
wheel works.

---

## Setting up a development environment

Requires [uv](https://docs.astral.sh/uv/) and Git. uv provisions the Python
interpreter itself, so no system Python is needed.

```powershell
cd sha256

uv sync --all-groups          # creates .venv and installs runtime + dev deps
```

Optional per-machine settings:

```powershell
Copy-Item .env.example .env   # then edit
```

Configuration is layered, lowest priority first: built-in defaults →
`configs/default.toml` → `configs/local.toml` (gitignored) → environment
variables / `.env` → command-line overrides.

### Optional extras

Heavy and license-encumbered dependencies are **not** installed by default:

```powershell
uv sync --extra tracking      # MediaPipe            (Apache-2.0)
uv sync --extra onnx          # ONNX Runtime GPU     (MIT)
uv sync --extra vcam-gpl      # pyvirtualcam         (GPL-2.0 — read the ADR first)
```

Read [`docs/OPEN_SOURCE_EVALUATION.md`](docs/OPEN_SOURCE_EVALUATION.md) before
enabling any of these. Licensing differs between a project's source code, its
pretrained weights, and its dependencies, and at least one candidate restricts
its weights to non-commercial use.

---

## Running

```powershell
uv run sha256                 # launch the application shell
uv run sha256 --check         # wire everything, print diagnostics, exit (no UI)
uv run sha256 --version
```

`--check` is the fastest way to confirm an environment is sound; it needs no
display and is what CI runs.

The shell shows an avatar selector, a camera selector, Start/Stop controls, and
a developer panel. "Step one frame" pushes a single frame through the
placeholder stages — useful for confirming the wiring, not for seeing an avatar.

---

## Running tests and checks

```powershell
uv run pytest                             # all tests
uv run pytest tests/unit -q               # unit tests only
uv run pytest --cov=sha256                # with coverage

uv run ruff check .                       # lint
uv run ruff format --check .              # formatting
uv run mypy                               # type checking (strict)
```

All four run in CI on Windows. See
[`docs/development/CONTRIBUTING.md`](docs/development/CONTRIBUTING.md).

---

## Adding an avatar

Create a directory under `avatars/` named after the avatar id, containing a
`metadata.json` manifest:

```
avatars/avatar_003/
  metadata.json
  preview.png
  model/
```

No application code changes. The library is scanned at startup, manifests are
validated individually, and a malformed avatar is reported without affecting the
others. Schema: [`src/sha256/core/avatar.py`](src/sha256/core/avatar.py).
Design: [`docs/architecture/AVATAR_SYSTEM.md`](docs/architecture/AVATAR_SYSTEM.md).

Avatar model weights are **not** tracked in Git — size, and redistribution
terms that vary per model.

---

## Current limitations

Beyond the unimplemented components listed above:

- **No real-time loop.** Frames advance one at a time on the UI thread. Threading,
  frame pacing, and back-pressure are the next milestone.
- **No performance data.** The monitor collects metrics correctly, but nothing
  has been benchmarked, so no FPS or latency figure is claimed.
- **No CPU/GPU/VRAM sampling.** The probe interface exists and reports "not
  measured" rather than a fabricated value.
- **VRAM budgeting is unproven.** The residency policy is implemented and unit
  tested, but its estimates come from manifests; it has never evicted a real model.
- **The renderer does not scale images.** It letterboxes and crops. Resampling
  is a quality decision deferred to the real renderer.
- **No project license chosen.** `LICENSE` is a restrictive placeholder — see
  the file for why, and for the third-party constraints that bear on the choice.
- **Windows only in practice.** Nothing is deliberately non-portable, but only
  Windows is targeted, tested, or supported.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/architecture/SYSTEM_ARCHITECTURE.md`](docs/architecture/SYSTEM_ARCHITECTURE.md) | Layers, contracts, threading plan, data flow |
| [`docs/architecture/AVATAR_SYSTEM.md`](docs/architecture/AVATAR_SYSTEM.md) | Avatar representation, storage, loading, VRAM management |
| [`docs/OPEN_SOURCE_EVALUATION.md`](docs/OPEN_SOURCE_EVALUATION.md) | Technology evaluation and recommendations |
| [`docs/SECURITY.md`](docs/SECURITY.md) | Threat model and security posture |
| [`docs/PRIVACY.md`](docs/PRIVACY.md) | What is processed, stored, and never transmitted |
| [`docs/benchmarks/PERFORMANCE_PLAN.md`](docs/benchmarks/PERFORMANCE_PLAN.md) | How performance will be measured |
| [`docs/decisions/`](docs/decisions/) | Architecture decision records |
| [`docs/development/`](docs/development/) | Setup, conventions, foundation status |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history |

---

## A note on the avatars

Every avatar in sha256 is fictional and AI-generated. The project is not
intended for impersonating real people, and manifests carry an explicit
`fictional` assertion. See [`docs/PRIVACY.md`](docs/PRIVACY.md).
