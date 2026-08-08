"""Exception hierarchy for sha256.

Every error raised across a module boundary derives from :class:`Sha256Error`,
so the application shell can catch one type and degrade gracefully instead of
letting a backend-specific exception escape into the UI thread.

Backend adapters are expected to translate third-party exceptions (a MediaPipe
error, an ONNX Runtime error, a Win32 HRESULT) into these types at their own
boundary. That translation is what keeps the core independent of any vendor.
"""

from __future__ import annotations


class Sha256Error(Exception):
    """Base class for all sha256 errors."""


# --- configuration ---------------------------------------------------------


class ConfigurationError(Sha256Error):
    """Configuration was missing, malformed, or internally inconsistent."""


class ComponentNotRegisteredError(ConfigurationError):
    """A backend was requested by id but nothing is registered under that id."""

    def __init__(self, category: str, backend_id: str, available: list[str]) -> None:
        self.category = category
        self.backend_id = backend_id
        self.available = available
        known = ", ".join(sorted(available)) or "<none>"
        super().__init__(f"No {category} backend registered as {backend_id!r}. Available: {known}.")


# --- device / capture ------------------------------------------------------


class DeviceError(Sha256Error):
    """A hardware or OS-level device could not be used."""


class CameraError(DeviceError):
    """Camera capture failed to open, read, or configure."""


class VirtualCameraError(DeviceError):
    """The virtual camera sink could not be created or written to."""


# --- pipeline components ---------------------------------------------------


class TrackingError(Sha256Error):
    """The face tracker failed on a frame or during initialisation."""


class EngineError(Sha256Error):
    """The avatar engine failed to initialise or produce animation output."""


class RenderingError(Sha256Error):
    """The renderer failed to produce a frame."""


# --- avatar library --------------------------------------------------------


class AvatarError(Sha256Error):
    """Base class for avatar library problems."""


class AvatarNotFoundError(AvatarError):
    """No avatar exists with the requested id."""

    def __init__(self, avatar_id: str) -> None:
        self.avatar_id = avatar_id
        super().__init__(f"No avatar found with id {avatar_id!r}.")


class AvatarManifestError(AvatarError):
    """An avatar manifest was unreadable, invalid, or of an unsupported version."""


class AvatarAssetError(AvatarError):
    """A referenced avatar asset is missing, unreadable, or failed integrity checks."""


class AvatarLoadError(AvatarError):
    """An avatar passed validation but could not be made resident."""


class InsufficientResourcesError(AvatarError):
    """Loading was refused because it would exceed the configured VRAM budget."""
