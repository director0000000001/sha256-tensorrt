# ADR-0003 — Dataclasses on the hot path, Pydantic at the boundary

**Status:** Accepted
**Date:** 2026-08-08

## Context

Two kinds of data with opposite requirements:

- **Per-frame data** — `VideoFrame`, `FaceTrackingResult`, `RenderedFrame`.
  Constructed 30+ times a second inside the latency budget. Produced by code we
  control.
- **Boundary data** — configuration, avatar manifests. Read once, from files
  that may be hand-edited or (eventually) downloaded. Must fail loudly and
  legibly when wrong.

## Options

- **Pydantic everywhere.** Consistent, excellent errors — but validation runs on
  every construction. On the hot path that is per-frame cost for data that was
  produced internally and cannot be malformed in the ways validation catches.
- **Dataclasses everywhere.** Zero overhead, but boundary data gets no
  validation, and a typo in a manifest becomes a confusing `AttributeError` at
  some later point rather than a clear message at load time.
- **Split by role.** Two idioms in one codebase.

## Decision

**Split by role.**

| Location | Mechanism | Why |
|---|---|---|
| `core/types.py` | `@dataclass(frozen=True, slots=True)` | Per-frame; no validation cost |
| `core/metrics.py` | `@dataclass(frozen=True, slots=True)` | Produced internally |
| `core/avatar.py` | Pydantic, `extra="forbid"` | Parsed from JSON on disk |
| `config/schema.py` | pydantic-settings, `extra="forbid"` | Parsed from TOML and env |

`frozen=True` prevents a consumer from mutating a frame another stage still
holds. `slots=True` removes the per-instance `__dict__` — meaningful when
thousands of these are created per minute.

Hot-path types are not validation-free: `LandmarkSet` rejects a wrong-shaped
array and `BlendshapeSet` rejects mismatched name/weight lengths. Both are cheap
shape checks that catch adapter bugs at the point of construction, where they
are diagnosable.

## Consequences

**Good**

- No validation cost where the latency budget lives.
- Clear, field-level error messages where input is untrusted.
- `extra="forbid"` turns a manifest typo into a load-time error instead of a
  silently ignored field.
- Pydantic's schema generation is available for manifest documentation and
  future tooling.

**Bad**

- Two idioms to learn. Mitigated by the rule being simple: *crosses an I/O
  boundary → Pydantic; flows through the pipeline → dataclass.*
- Dataclass types trust their producers. A buggy adapter can emit nonsense that
  no validation catches. Adapter tests are the mitigation.
- `frozen=True` means updates go through `dataclasses.replace`, which allocates.
  Acceptable, because frames are rarely modified after construction — and the
  hazard it prevents (mutating a frame another stage is reading) is worse.

## Related

`VideoFrame.data` is a **borrowed** buffer: a capture backend may reuse it for
the next frame. Consumers needing the pixels beyond the current tick must call
`.copy()`. This avoids a per-frame allocation on the hot path while keeping the
hazard explicit and documented on the type itself.
