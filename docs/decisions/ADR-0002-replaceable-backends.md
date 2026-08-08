# ADR-0002 — Protocols and a registry for replaceable backends

**Status:** Accepted
**Date:** 2026-08-08

## Context

The hard requirement: SHA-256 must not be tightly coupled to any specific AI
model. MediaPipe, LivePortrait, and any future tracker or engine must be
replaceable.

This is not hypothetical. `docs/OPEN_SOURCE_EVALUATION.md` concludes that the
leading avatar engine has **unresolved weight licensing** and a dependency whose
models are non-commercial-only. There is a realistic chance the primary engine
has to be swapped for legal reasons rather than technical ones — which is
exactly the case where a tightly coupled design is most expensive.

## Options

**Abstraction mechanism**

- **Abstract base classes.** Familiar, explicit. But a backend must import
  SHA-256 to subclass, and metaclass conflicts arise when it already inherits
  from a vendor base class — which several candidates do.
- **`typing.Protocol`.** Structural typing: a backend satisfies the contract
  without importing anything. Trivial test doubles. No MRO surprises.
- **Duck typing with no contract.** Fastest to write, no way to verify, and the
  compatibility question moves to runtime on a user's machine.

**Backend selection**

- **Direct imports in the composition root.** Simple, but every backend is
  imported at startup — including heavy optional ones.
- **Dotted import path in config.** Flexible, but a config file naming a Python
  module to import is executable input. That is a real supply-chain hazard once
  avatar packs or plugins are downloadable.
- **Registry of factories keyed by short id.** One indirection; ids can be
  listed; imports stay lazy.
- **Entry-point auto-discovery.** Convenient, but loads arbitrary installed
  packages — a supply-chain risk that buys nothing while all backends ship
  in-tree.

## Decision

**Protocols** for the contracts (`src/sha256/core/interfaces.py`) and an
explicit **registry of factories** keyed by short string id
(`src/sha256/app/registry.py`).

Backends are named in configuration by registry id. The composition root
(`app/bootstrap.py`) is the only place that imports a concrete backend.

The load-bearing constraint on the protocols: **no method signature mentions a
vendor type.** Parameters and returns come only from `core.types` and
`core.avatar`. A backend translates at its own edge. Without that rule the
protocols would be decoration.

## Consequences

**Good**

- Swapping the tracker is one config line plus one adapter file.
- Registration stores a factory, so a registered MediaPipe backend does not
  import MediaPipe until selected. Startup stays fast; an uninstalled optional
  dependency is not an error.
- Configuration contains no import paths, so it cannot cause code execution.
- Available ids can be enumerated, making "unknown backend" a helpful error and
  letting the UI list backends honestly.
- `isinstance` works against `@runtime_checkable` protocols, which the
  integration tests use to assert conformance structurally.

**Bad**

- One layer of indirection between config and instance. Finding what runs means
  reading `register_builtin_backends`.
- Protocols check shape, not behaviour. A backend can satisfy the type and still
  be wrong — the integration tests exist to cover that gap.
- `runtime_checkable` only verifies method *presence*, not signatures. Real
  verification comes from mypy plus tests.
- Adding a backend means editing `register_builtin_backends`. Accepted: an
  explicit list is exactly what makes the plugin surface auditable.

## Notes

`AvatarRepository` is deliberately **read-only**. Mutation is an owner-only
operation handled elsewhere, so the running application has no code path that
can modify the avatar library.

`PerformanceMonitor` is deliberately **not** a `Component`. Monitoring outlives
any single pipeline run, and making it startable would invite losing metrics
exactly when a component fails to start.
