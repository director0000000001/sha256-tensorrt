# ADR-0001 — Python version and toolchain

**Status:** Accepted
**Date:** 2026-08-08

## Context

sha256 is Python-first on Windows, and will eventually depend on the CV/AI
ecosystem — MediaPipe, ONNX Runtime, possibly PyTorch. Those wheels are the
binding constraint on Python version: they lag new releases, and a version with
no wheel means building from source on Windows, which is not viable.

## Options

**Python version**

- **3.14** — newest; already installed on the development machine. Wheel
  coverage for the AI stack is unproven and historically lags by a year or more.
- **3.13** — free-threading is interesting for a multi-stage pipeline, but the
  ecosystem is still catching up.
- **3.12** — mature, broad wheel coverage, and supports the language features
  this codebase actually uses (PEP 695 generics, `StrEnum`, `tomllib`).
- **3.11 or older** — no benefit; loses `tomllib` ergonomics and PEP 695.

**Dependency management**

- **pip + requirements.txt** — universal, but no lockfile, no interpreter
  management, slow.
- **Poetry** — mature, but slower and does not provision interpreters.
- **uv** — very fast, provisions the interpreter itself, produces a hash-pinned
  lockfile, and is already installed here.

**Lint / format / types**

- ruff (replacing black + isort + flake8) and mypy.

## Decision

**Python 3.12**, pinned as `>=3.12,<3.13`, managed by **uv**. **ruff** for
linting and formatting, **mypy --strict** for type checking, **pytest** for
tests.

The upper bound is deliberate. Without it, `uv sync` on a machine with a newer
interpreter would silently resolve to a version whose AI wheels may not exist —
producing a failure far from its cause.

Verified during this work: `mediapipe 1.0.0`, `onnxruntime-gpu 1.28.0`,
`pyvirtualcam 0.15.0`, `opencv-python 5.0.0.93`, and CUDA 12.8/12.9/13.0 builds
of PyTorch all resolve for Python 3.12 on Windows.

## Consequences

**Good**

- The AI stack is available today, verified rather than assumed.
- uv provisions the interpreter, so setup is `uv sync` with no system Python.
- The lockfile pins the full transitive tree with hashes; CI installs `--frozen`.
- One tool (ruff) replaces three, and runs fast enough for pre-commit.

**Bad**

- Not on the newest Python; no free-threading experiments without changing this.
- uv is younger than pip and Poetry. Mitigated by uv being a drop-in producer of
  standard virtual environments — the escape hatch is `pip install -e .`.
- `mypy --strict` costs real effort on new code. Accepted deliberately: the
  contracts in `core/` are the thing most worth type-checking, and strictness
  there is what caught a genuine caching bug in `AvatarManager.refresh()` during
  the foundation work.

**Note:** on this machine, generated console-script executables (`sha256.exe`)
are blocked by an Application Control policy. `python -m sha256` works and is
what the documentation uses.
