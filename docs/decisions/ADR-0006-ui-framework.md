# ADR-0006 — PySide6 for the desktop shell

**Status:** Accepted
**Date:** 2026-08-08

## Context

SHA-256 needs a Windows desktop UI: avatar gallery with image previews, live
video preview, controls, and a diagnostics panel. It must not interfere with a
real-time pipeline running alongside it.

## Options

- **PySide6** — official Qt for Python bindings, **LGPL-3.0** (or a commercial
  Qt license). Mature, complete, good image and video primitives.
- **PyQt6** — also Qt, but **GPL-3.0 or a paid Riverbank commercial license**.
  Technically comparable; strictly worse licensing for a project whose own
  license is undecided.
- **Tkinter** — in the standard library, no dependency. Weak image handling, no
  video primitives, dated appearance. A gallery of previews would be painful.
- **Dear PyGui / imgui** — fast, GPU-rendered, good for debug tooling. Immediate-
  mode fits a debug panel well but fits a consumer application poorly, and the
  ecosystem is thinner.
- **Web UI (Electron / pywebview / Tauri)** — best styling story, but adds a
  browser runtime, an IPC boundary between the UI and the pipeline, and a
  frame-marshalling cost for live preview. Wrong shape for a local real-time
  application.

## Decision

**PySide6.**

The deciding factors, in order:

1. **Licensing.** LGPL-3.0 permits use in a proprietary application provided
   relinking is possible — usually satisfied by dynamic linking. PyQt6's GPL-3.0
   would impose copyleft on the whole application. With SHA-256's own license
   still open (`LICENSE` is a restrictive placeholder), PySide6 preserves
   options that PyQt6 would foreclose.
2. **Live video preview.** Displaying frames at 30 fps without fighting the
   toolkit is a first-class requirement. Qt handles it; Tkinter does not.
3. **Threading model.** Qt's signal/slot mechanism has well-defined cross-thread
   semantics, which matters when a capture thread must hand frames to the UI
   thread. This is the part most likely to go wrong, and Qt has a real answer.
4. **Gallery UI.** Dozens of avatar tiles with previews, states, and errors is
   ordinary Qt work.

## Consequences

**Good**

- Mature toolkit with the primitives this application actually needs.
- LGPL keeps the project's licensing options open.
- Native Windows appearance without styling work.
- `pytest-qt` gives real UI tests; the shell tests run headless via
  `QT_QPA_PLATFORM=offscreen` and pass in CI.

**Bad**

- **Large dependency** (~150 MB installed across `pyside6-essentials` and
  `-addons`). By far the biggest item in a deliberately small runtime tree. If
  distribution size becomes a problem, trimming unused Qt modules is the first
  lever.
- **LGPL relinking obligations** on distribution: dynamic linking must be
  preserved, and a fully-static bundled build would need care. This is a
  packaging constraint to remember, not a blocker.
- **Incomplete type stubs.** `mypy` needs `ignore_missing_imports` and
  `follow_untyped_imports` for `PySide6.*`. UI code is therefore less
  type-checked than the rest of the codebase — an argument for keeping logic out
  of widgets, which the architecture already does.
- **Qt idioms leak into UI code** (`closeEvent` naming, parent ownership). Kept
  contained by keeping the UI thin: `MainWindow` calls into the controller and
  the avatar manager and renders their state; it owns no pipeline logic.

## Notes

The foundation shell is deliberately unstyled — no stylesheet, no custom
painting, no icons. The goal of this milestone is correct structure; visual
design would lock in decisions meant to change.

The structural rules being established: panels are independent widgets, the
window owns no pipeline logic, and nothing slow runs on the UI thread. The one
place that currently violates the last rule — avatar loading in
`_on_avatar_changed` — is marked with `TODO(threading)`, because with a real
engine it would freeze the window.
