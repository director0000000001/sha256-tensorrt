# ADR-0005 — Virtual camera approach

**Status:** **Proposed — the shipping decision is open**
**Date:** 2026-08-08

This ADR is deliberately unresolved. It records the constraints so the decision
is made with them in view rather than by accident, when someone reaches for the
easiest `pip install`.

## Context

sha256's output must appear to Windows as a webcam so Meet, Zoom, Discord, and
OBS can consume it. This is the **only** part of the system where the obvious
technical choice carries a licensing consequence for the whole project — and
sha256's own license is not yet decided (`LICENSE` is a restrictive
placeholder).

## Options

### A. pyvirtualcam

- License: **GPL-2.0** (verified from the repository)
- Verified available: `pyvirtualcam 0.15.0` for Python 3.12 on Windows
- Writes to an **existing** virtual camera; it does not create one
- Windows backends: OBS virtual camera (requires OBS installed and started once)
  or UnityCapture

Trivial to integrate. But linking a GPL-2.0 library into a distributed
application has copyleft implications for the whole work — adopting it would
decide sha256's license by default. It also requires the user to install OBS,
which is a poor first-run experience and makes sha256 dependent on another
application's install state.

### B. OBS virtual camera, used externally

OBS has shipped a virtual camera since 26.0. Driving it as a separate
application is a different legal posture from linking pyvirtualcam, but still
requires the user to install and run OBS. OBS also exposes a **single** camera
instance, so frames cannot be sent in, mixed, and sent out through the same
device.

### C. Windows Media Foundation — `MFCreateVirtualCamera`

- **Native OS API.** No third-party license involved at all.
- **Windows 11 build 22000+.** Not available on Windows 10.
- Provides RGB32 and NV12; NV12 typically preferred, converted via the Media
  Foundation Video Processor MFT.
- Session-scoped or system-scoped lifetime; per-user or all-users access.
- **No first-class Python binding.** Needs a COM/Media Foundation source —
  realistically a small C++/WinRT or .NET component. Microsoft publishes samples
  (`smourier/VCamSample`).

Architecturally correct, and the only option with no licensing entanglement. It
costs a native component to build and maintain, drops Windows 10, and touches
installation and permissions for system-scoped registration.

### D. Custom DirectShow filter

Works on Windows 10 and 11, but DirectShow is legacy, modern applications
increasingly prefer Media Foundation, and it requires `regsvr32` registration
with administrator rights. Highest effort, declining relevance.

## Decision (provisional)

**Develop against option A behind the `VirtualCameraSink` abstraction; target
option C for anything distributed.**

Specifically:

- `pyvirtualcam` stays in an optional extra named **`vcam-gpl`** — the name
  makes the constraint visible at the point of installation, and it is not
  installed by default.
- All output goes through `VirtualCameraSink` (`core/interfaces.py`), so the
  backend is swappable like any other component.
- **No virtual camera is implemented in the foundation build.** The `null` sink
  counts frames and discards them.

The abstraction is what makes deferring this safe: the decision can be made
later without the rest of the pipeline caring.

## Open questions

1. **What license will sha256 adopt?** This determines whether option A is even
   permissible for distribution.
2. **Is Windows 10 in scope?** Option C is Windows 11 only. Dropping Windows 10
   would be a product decision, not a technical one.
3. **Who builds and maintains the native component?** It is C++/WinRT or .NET —
   outside this project's Python-first premise.
4. **Is transparency needed?** UnityCapture supports RGBA; Media Foundation
   virtual cameras conventionally do not.

## Consequences

**Good**

- The foundation ships without committing to a GPL dependency.
- Development can proceed with the easy option without contaminating the
  distributable.
- The abstraction means the decision is reversible.

**Bad**

- A significant unknown remains at the end of the pipeline — the last stage is
  the least resolved.
- Option C is materially more work than a `pip install`, and there is a standing
  temptation to take the easy path and inherit GPL by inattention. Naming the
  extra `vcam-gpl` exists to make that inattention harder.
- Developing against A and shipping C means the shipping path gets less
  exercise, so the switch must happen early enough to be tested properly.
