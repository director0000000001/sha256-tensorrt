# -*- mode: python -*-
"""PyInstaller spec for the portable build (day 16, D4).

    uv run pyinstaller packaging/sha256.spec --noconfirm

Produces ``dist/sha256/`` with ``sha256.exe`` and its libraries; the
``scripts/build_exe.py`` wrapper runs this and then lays ``avatars/``,
``models/`` and ``configs/`` beside the executable (the portable layout
``AppPaths.portable_root`` looks for), writes the guide in, and zips it.

What is *not* in the executable, on purpose: the model weights (1.2 GB of
LivePortrait plus the face landmarker; fetched by the scripts, copied by the
wrapper), the avatars (owner data), the configuration (the machine's own).
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 -- PyInstaller's global
SRC = ROOT / "src"

hidden = []
for package in ("sha256", "mediapipe", "cv2", "pyvirtualcam", "torch", "torchvision"):
    with contextlib.suppress(Exception):  # a package not installed in this environment
        hidden += collect_submodules(package)

datas = []
for package in ("mediapipe",):
    with contextlib.suppress(Exception):
        datas += collect_data_files(package)
# The window's own assets (background image, if any) travel next to the exe
# via the wrapper, not inside the bundle.

block_cipher = None

a = Analysis(  # noqa: F821
    [str(SRC / "sha256" / "__main__.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib.tests", "IPython", "pytest"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="sha256",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="sha256",
)
