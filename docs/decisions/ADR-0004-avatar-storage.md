# ADR-0004 — Directory-per-avatar library layout

**Status:** Accepted
**Date:** 2026-08-08

## Context

sha256 must support many avatars — dozens eventually — with per-avatar models,
configuration, previews, and versions. Adding one must not require changing
application code, and the layout has to survive avatars being added, removed,
updated, and (later) distributed as downloadable packs.

## Options

- **Central index** (`avatars/library.json` listing every avatar). One read to
  discover everything.
- **Directory per avatar**, each with its own manifest. One scan to discover.
- **Database** (SQLite). Queryable, transactional.

## Decision

**Directory per avatar**, named exactly by the manifest `id`, each containing a
`metadata.json`.

## Rationale

| | Directory-per-avatar | Central index | SQLite |
|---|---|---|---|
| Add an avatar | Copy a directory | Edit a shared file | Import via tooling |
| Two added independently | No conflict | **Merge conflict** | Needs migration handling |
| One corrupt manifest | That avatar skipped | **Whole library may fail** | Constrained by schema |
| Delete | Remove directory | Remove files **and** index entry | Delete row + files |
| Inspect by hand | Open a JSON file | Open a JSON file | Needs a tool |
| Discovery cost (50 avatars) | One scan, ~50 small reads | One read | One query |

The central index wins only on discovery cost, and only marginally — discovery
reads manifests, never weights. Every other row is a failure mode that occurs in
ordinary use, and the directory layout wins all of them.

SQLite adds transactional integrity that this workload does not need, and makes
the library opaque to inspection. For a user-managed asset library, being able
to look at it in a file manager has real value.

## Supporting rules

- **`id` must equal the directory name.** Otherwise one avatar has two plausible
  identities and anything keyed by id — caches, exports, metric labels — becomes
  ambiguous.
- **Ids are constrained** to `^[a-z0-9][a-z0-9_-]{2,63}$`. The id becomes a
  directory name, so this closes path traversal via the id itself.
- **Asset paths are relative and contained.** Absolute paths, drive letters, and
  `..` are rejected at parse time — checked on the raw string, because
  `pathlib` resolves differently per OS and a pack authored on one must be
  judged identically on the other.
- **Discovery never raises.** Failures are collected and reported; one bad
  manifest cannot hide a healthy library.
- **Manifests are tracked in Git; weights are not.** Manifests are small and
  reviewable and define the layout. Weights are large and their redistribution
  terms vary per model.

## Consequences

**Good**

- Adding an avatar is a file copy with no code change and no index edit.
- Failures are isolated to the individual avatar.
- The library is inspectable and diffable.
- Scales to dozens without restructuring — nothing is linear in library size
  except a startup scan of small JSON files.
- Maps naturally onto a future downloadable pack format: a pack is a directory.

**Bad**

- No cross-avatar queries without scanning. Not needed today; if it becomes
  needed, a derived index can be cached without changing the storage format.
- A filesystem scan on every forced refresh. Cheap at this scale, and results
  are cached between refreshes.
- Nothing enforces that the directory contents match the manifest beyond
  declared assets. Validation covers what is declared; undeclared files are
  ignored.
