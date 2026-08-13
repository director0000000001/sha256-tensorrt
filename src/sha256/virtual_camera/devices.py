"""Which virtual-camera devices already exist on this machine.

:attr:`~sha256.core.interfaces.VirtualCameraSink.is_available` has to answer
"will ``start()`` work" *before* ``start()`` is called, so the UI can name what
to install instead of surfacing a driver-level failure from inside the capture
thread.

Opening the device to find out is not an option: opening it advertises a camera
to every application on the system, and a probe that momentarily makes a camera
appear in Zoom's device menu is worse than the error it was meant to prevent.

Reading the registry costs nothing and has no side effects. Windows lists
DirectShow capture filters under one well-known category key, and a software
virtual camera is a filter registered there like any other — this is the same
list the video-call applications read when they populate their camera menus.

Verified on the reference machine (2026-08-13): that key holds exactly one
entry, ``OBS Virtual Camera``, whose ``InprocServer32`` points at
``obs-virtualcam-module64.dll`` inside the OBS installation. Physical UVC
cameras do **not** appear here — they are enumerated through the kernel
streaming proxy instead — so what this returns is, in practice, the list of
software cameras and nothing else.
"""

from __future__ import annotations

import sys

from sha256.utils.logging import get_logger

__all__ = [
    "SUPPORTED_DEVICES",
    "registered_capture_filters",
    "supported_devices_present",
]

_LOG = get_logger(__name__)

# The DirectShow "video capture sources" category. A fixed GUID defined by
# Windows, not by any application.
_CAPTURE_CATEGORY_KEY = r"CLSID\{860BB310-5D01-11d0-BD3B-00A0C911CE86}\Instance"

SUPPORTED_DEVICES: dict[str, str] = {
    "obs": "OBS Virtual Camera",
    "unitycapture": "Unity Video Capture",
}
"""pyvirtualcam backend name -> the device name Windows exposes for it.

The OBS entry is **verified** against this machine's registry. The UnityCapture
entry is taken from pyvirtualcam's documentation and has not been observed here,
because UnityCapture is not installed — it is listed so that a machine which
does have it is recognised, not as a measured fact.
"""


def registered_capture_filters() -> tuple[str, ...]:
    """Friendly names of every DirectShow capture filter registered here.

    Returns an empty tuple on a non-Windows platform, or when the registry
    cannot be read. "Could not tell" and "nothing installed" deliberately look
    the same to callers: both mean the sink must not claim to be available.
    """
    if sys.platform != "win32":
        return ()

    import winreg

    names: list[str] = []
    try:
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, _CAPTURE_CATEGORY_KEY) as category:
            subkey_count = winreg.QueryInfoKey(category)[0]
            for index in range(subkey_count):
                clsid = winreg.EnumKey(category, index)
                try:
                    with winreg.OpenKey(category, clsid) as entry:
                        value = winreg.QueryValueEx(entry, "FriendlyName")[0]
                except OSError:
                    # A filter without a FriendlyName is not one a user could
                    # pick from a menu anyway. Skip it rather than fail the scan.
                    continue
                if isinstance(value, str):
                    names.append(value)
    except OSError as exc:
        # An absent category key means no DirectShow capture filter has ever
        # been registered. That is a normal state, not an error worth raising.
        _LOG.debug("Could not enumerate DirectShow capture filters: %s", exc)
        return ()

    return tuple(names)


def supported_devices_present() -> tuple[str, ...]:
    """pyvirtualcam backend names whose device is installed on this machine.

    Empty means publishing cannot work yet, whatever else is installed —
    pyvirtualcam writes to an existing virtual camera, it does not create one.
    """
    installed = set(registered_capture_filters())
    return tuple(backend for backend, device in SUPPORTED_DEVICES.items() if device in installed)
