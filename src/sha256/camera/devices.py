"""Camera device enumeration.

Uses Qt's ``QMediaDevices`` rather than adding a dependency: PySide6 is already
required for the UI, and OpenCV cannot report device *names* at all — it only
knows integer indices. A user picking between "Integrated Webcam" and "Sony
ZV-E10" needs names.

The index mapping caveat
------------------------
Qt and OpenCV enumerate independently. On Windows both walk Media Foundation's
device list, so position N in Qt's list is *normally* index N in OpenCV — but
that is an observed convention, not a guarantee. Where it matters, the capture
backend verifies by opening the device and comparing the reported resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

from sha256.utils.logging import get_logger

__all__ = ["CameraDeviceInfo", "enumerate_cameras"]

_LOG = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CameraDeviceInfo:
    """A capture device the OS is advertising.

    Satisfies the ``CameraSpec`` protocol from :mod:`sha256.core.interfaces`.
    """

    device_id: str
    display_name: str
    index: int
    """Position in the enumeration — what OpenCV is opened with."""

    is_default: bool = False
    max_resolution: tuple[int, int] | None = None
    max_frame_rate: float | None = None

    def describe(self) -> str:
        parts = [self.display_name]
        if self.max_resolution:
            width, height = self.max_resolution
            parts.append(f"{width}x{height}")
        if self.max_frame_rate:
            parts.append(f"{self.max_frame_rate:.0f}fps")
        return " — ".join(parts)


def enumerate_cameras() -> tuple[CameraDeviceInfo, ...]:
    """List capture devices. Returns an empty tuple if none or if Qt is absent.

    Never raises: enumeration runs during UI construction and on a background
    refresh, and a driver hiccup there must not take the application down.
    """
    try:
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtMultimedia import QMediaDevices
    except ImportError:
        _LOG.warning("PySide6.QtMultimedia unavailable; cannot enumerate cameras")
        return ()

    # Qt's multimedia plugin needs an application object before it will
    # initialise. In the UI there always is one; in a script, a benchmark, or a
    # test there may not be, and without this the call warns and can misbehave.
    if QCoreApplication.instance() is None:
        QCoreApplication([])

    try:
        devices = QMediaDevices.videoInputs()
    except Exception:
        _LOG.exception("Camera enumeration failed")
        return ()

    found: list[CameraDeviceInfo] = []
    for index, device in enumerate(devices):
        best_resolution: tuple[int, int] | None = None
        best_rate: float | None = None

        for video_format in device.videoFormats():
            size = video_format.resolution()
            resolution = (size.width(), size.height())
            if best_resolution is None or resolution > best_resolution:
                best_resolution = resolution
            rate = float(video_format.maxFrameRate())
            if best_rate is None or rate > best_rate:
                best_rate = rate

        found.append(
            CameraDeviceInfo(
                device_id=bytes(device.id().data()).decode(errors="replace"),
                display_name=device.description() or f"Camera {index}",
                index=index,
                is_default=device.isDefault(),
                max_resolution=best_resolution,
                max_frame_rate=best_rate,
            )
        )

    _LOG.info("Enumerated %d camera device(s)", len(found))
    return tuple(found)
