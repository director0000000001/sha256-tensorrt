"""sha256 — a local-first, real-time AI avatar application for Windows.

Pipeline shape (target)::

    Webcam -> FaceTracker -> FaceTrackingResult -> AvatarEngine
           -> RenderedFrame -> Renderer -> VirtualCameraSink -> Meet/Zoom/OBS

Current status: **foundation only**. Every backend is a placeholder. See
``docs/development/FOUNDATION_STATUS.md`` for exactly what does and does not
work today.

Import guidance: prefer :mod:`sha256.core.interfaces` and
:mod:`sha256.core.types` — they are the stable contracts. Anything under a
feature package is an implementation detail and may be replaced.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
