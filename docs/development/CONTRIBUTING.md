# Contributing to sha256

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Git. uv provisions Python itself.

```powershell
cd sha256
uv sync --all-groups
uv run pre-commit install
```

Verify:

```powershell
uv run python -m sha256 --check    # wires everything, prints diagnostics
uv run pytest                      # 180 tests
```

> On some Windows machines an Application Control policy blocks generated
> console-script executables (`sha256.exe`). Use `python -m sha256`, which the
> documentation and CI both use.

## The checks CI runs

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run python -m sha256 --check
```

Run them before pushing. `pre-commit` covers lint and formatting; mypy is
CI-only because it needs the full environment and is slow enough to make
committing painful.

---

## Architectural rules

These are not style preferences. Each protects a specific property of the
design.

### 1. `core` imports nothing from above or beside it

`sha256.core` may import the standard library, `numpy`, and `pydantic` — nothing
else. No Qt, no vendor AI library, no feature package.

*Why:* `core` is the dependency floor that makes every backend replaceable.
Once it imports MediaPipe, MediaPipe is no longer replaceable.

### 2. No protocol signature mentions a vendor type

Parameters and returns come from `core.types` and `core.avatar` only. A backend
translates at its own edge.

*Why:* this is the actual mechanism behind "not coupled to one specific AI
model". Without it, the protocols are decoration.

### 3. Backends are reached through the registry

Only `app/bootstrap.py` imports a concrete backend. Everything else goes through
`ComponentRegistry`.

*Why:* keeps imports lazy, keeps import paths out of config files (executable
input is a supply-chain hazard), and makes the set of backends auditable in one
place.

### 4. Validate at boundaries, trust inside

Anything read from disk or the environment is validated by Pydantic with
`extra="forbid"`. Per-frame types are plain dataclasses.

*Why:* ADR-0003. Validation belongs where input is untrusted, not on the hot
path.

### 5. Never log per frame

Frame-rate diagnostics go through `PerformanceMonitor`, which aggregates.

*Why:* file I/O in the capture path causes dropped frames, and a per-frame log
would be a record of the user's facial movements on disk.

### 6. Do not report a number you did not measure

`None` means "not measured" and displays as `n/a`. `0.0` means "measured as
zero". Never substitute one for the other.

*Why:* project rules 11 and 12. A fabricated zero is worse than a blank,
because it looks like data.

### 7. Fail the individual avatar, never the library

A malformed manifest, missing asset, or failed load marks that avatar `ERROR`
and leaves the rest working.

*Why:* a user with twelve avatars and one broken one should see eleven working
avatars, not an application that will not start.

---

## Adding a backend

Worked example — a new face tracker:

1. **Write the adapter** in `src/sha256/tracking/<name>.py`. It must satisfy
   `FaceTracker` and convert vendor output into `FaceTrackingResult` inside its
   own module. Translate vendor exceptions into `TrackingError`.
2. **Register it** in `register_builtin_backends()`, declaring `requires` so
   availability can be reported without importing.
3. **Add its dependency** as an optional extra in `pyproject.toml` — never a
   runtime dependency.
4. **Test the adapter** against real vendor output, and add it to the protocol
   conformance test in `tests/integration/test_pipeline_wiring.py`.
5. **Document the licensing** in `docs/OPEN_SOURCE_EVALUATION.md` — separately
   for source code, pretrained weights, and third-party dependencies. If a
   license is unclear, write **UNCLEAR**. Do not guess.

No core code changes. If a backend requires one, that is a signal the contract
is wrong — fix the contract deliberately rather than working around it.

---

## Testing

| Location | Scope |
|---|---|
| `tests/unit/` | One module, no I/O beyond `tmp_path` |
| `tests/integration/` | Contracts composing across modules |
| `tests/fixtures/` | Shared data |

Conventions:

- Test names state the **property**, not the method:
  `test_active_avatar_is_never_evicted`, not `test_plan_admission_3`.
- A non-obvious test carries a one-line docstring explaining *why the property
  matters*.
- Fixtures build isolated applications — `_env_file=None` on `Settings`, so a
  developer's `.env` cannot change the result.
- Failure modes are tested as first-class behaviour, especially path traversal
  and manifest corruption. Those are security properties.
- Markers: `gui`, `integration`, `slow`, `requires_gpu`.

Timing-sensitive tests must poll with a sleep, not spin — the placeholder camera
rate-limits exactly like a real device, and a tight loop will out-run it. See
`_step_until_frame`.

---

## Commits

Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.

The body should explain **why**, not what — the diff already shows what.

Never commit: model weights, avatar payloads, `.env`, credentials, or benchmark
result files. `.gitignore`, `check-added-large-files`, `detect-private-key`, and
gitleaks all guard this, but they are backstops rather than the plan.

---

## When to write an ADR

Write one in `docs/decisions/` when a decision is expensive to reverse, forces a
trade-off, or will otherwise look arbitrary later — adding or replacing a major
dependency, changing a `core` contract, altering the threading model, or
anything with licensing consequences.

An ADR that lists no downsides was not analysed. Record what the choice costs.

---

## Performance claims

Do not make one without a measurement, and do not report a measurement without
its conditions: hardware, driver, resolution, backends, sample count, and
percentiles. See `docs/benchmarks/PERFORMANCE_PLAN.md`, including the table of
phrasings to avoid.
