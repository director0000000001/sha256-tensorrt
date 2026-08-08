"""Report what GPU acceleration is actually available on this machine.

Run before assuming any CUDA wheel works::

    uv run python scripts/check_gpu.py

Why this exists: the sha256 reference machine has an RTX 50-series (Blackwell)
GPU with compute capability 12.0 / ``sm_120``. Frameworks need a CUDA 12.8-or-
newer build to emit kernels for it, and a wheel built for older architectures
will import fine, report ``cuda.is_available() == True``, and then fail at the
first kernel launch. That failure mode is confusing enough to be worth a
dedicated check.

This script imports nothing heavy unless it is already installed. Absent
frameworks are reported as absent — never guessed at.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

MINIMUM_CUDA_FOR_BLACKWELL = (12, 8)


def _heading(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def check_driver() -> tuple[str, str] | None:
    """Query ``nvidia-smi``. Returns ``(gpu_name, compute_capability)`` if found."""
    _heading("NVIDIA driver")

    if shutil.which("nvidia-smi") is None:
        print("  nvidia-smi not found — no NVIDIA driver visible.")
        return None

    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"  nvidia-smi failed: {exc}")
        return None

    first: tuple[str, str] | None = None
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        name, driver, memory, compute = fields
        print(f"  {name}")
        print(f"    driver             : {driver}")
        print(f"    total memory       : {memory}")
        print(f"    compute capability : {compute}")
        if first is None:
            first = (name, compute)

    return first


def check_torch() -> None:
    _heading("PyTorch")
    try:
        import torch
    except ImportError:
        print("  not installed (optional — see pyproject optional-dependencies)")
        return

    print(f"  version            : {torch.__version__}")
    print(f"  built for CUDA     : {torch.version.cuda or 'CPU-only build'}")
    print(f"  cuda.is_available  : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        return

    print(f"  device             : {torch.cuda.get_device_name(0)}")
    capability = torch.cuda.get_device_capability(0)
    print(f"  device capability  : sm_{capability[0]}{capability[1]}")

    supported = torch.cuda.get_arch_list()
    print(f"  built-in arch list : {', '.join(supported)}")

    # The check that matters: a wheel can report CUDA available and still have
    # no kernels for this specific architecture.
    wanted = f"sm_{capability[0]}{capability[1]}"
    if wanted not in supported:
        print(f"  !! this build has NO kernels for {wanted}")
        print("     Expect failures at the first kernel launch, not at import.")
        print("     Install a CUDA 12.8+ build of PyTorch.")
    else:
        print(f"  OK: this build includes {wanted} kernels")

    # Actually launch a kernel. A build with no kernels for this architecture
    # still imports cleanly and still reports cuda.is_available() == True — it
    # only fails when something is executed. An import-time check would miss it.
    try:
        sample = torch.randn(256, 256, device="cuda")
        checksum = (sample @ sample).sum().item()
    except Exception as exc:
        print(f"  !! GPU matmul smoke test FAILED: {type(exc).__name__}: {exc}")
    else:
        print(f"  GPU matmul smoke test: OK (checksum {checksum:.2f})")


def check_onnxruntime() -> None:
    _heading("ONNX Runtime")
    try:
        import onnxruntime
    except ImportError:
        print("  not installed (optional)")
        return

    print(f"  version            : {onnxruntime.__version__}")
    providers = onnxruntime.get_available_providers()
    print(f"  providers          : {', '.join(providers)}")

    if "CUDAExecutionProvider" not in providers:
        print("  note: CPU-only. Install onnxruntime-gpu for CUDA support.")
    if "TensorrtExecutionProvider" in providers:
        print("  TensorRT provider is present.")


def main() -> int:
    print("sha256 GPU capability report")
    print(f"Python {sys.version.split()[0]} on {sys.platform}")

    driver = check_driver()
    check_torch()
    check_onnxruntime()

    _heading("Summary")
    if driver is None:
        print("  No NVIDIA GPU detected. sha256 will need a CPU fallback path.")
        return 0

    name, compute = driver
    print(f"  GPU: {name} (compute {compute})")
    try:
        major, minor = (int(part) for part in compute.split("."))
    except ValueError:
        print("  Could not parse the compute capability; check manually.")
        return 0

    if (major, minor) >= (12, 0):
        required = ".".join(str(part) for part in MINIMUM_CUDA_FOR_BLACKWELL)
        print(f"  Blackwell-class GPU: requires CUDA {required}+ builds of any framework.")
        print("  Verify with this script after installing torch or onnxruntime-gpu.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
