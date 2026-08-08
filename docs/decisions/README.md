# Architecture Decision Records

Each ADR records one decision: the context that forced it, the options weighed,
what was chosen, and what it costs.

The point is not ceremony. It is that six months from now, someone — possibly
the author — will look at a choice that seems obviously wrong and need to know
whether it was considered and rejected for a reason, or simply never thought
about. Those two situations need very different responses.

## Format

```
# ADR-NNNN — Title

**Status:** Proposed | Accepted | Superseded by ADR-XXXX
**Date:** YYYY-MM-DD

## Context      what forced a decision
## Options      what was actually considered
## Decision     what was chosen
## Consequences what it costs, including the bad parts
```

A decision that lists no downsides was not analysed.

## Records

| ADR | Title | Status |
|---|---|---|
| [0001](ADR-0001-python-tooling.md) | Python version and toolchain | Accepted |
| [0002](ADR-0002-replaceable-backends.md) | Protocols and a registry for replaceable backends | Accepted |
| [0003](ADR-0003-data-model-strategy.md) | Dataclasses on the hot path, Pydantic at the boundary | Accepted |
| [0004](ADR-0004-avatar-storage.md) | Directory-per-avatar library layout | Accepted |
| [0005](ADR-0005-virtual-camera-strategy.md) | Virtual camera approach | **Proposed — open** |
| [0006](ADR-0006-ui-framework.md) | PySide6 for the desktop shell | Accepted |
