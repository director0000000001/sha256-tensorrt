"""Build the TensorRT engine for LivePortrait's two pixel networks on this
GPU -- Phase 2 of "The fast version" (ROADMAP), measured in
docs/benchmarks/B006-tensorrt.md.

Exports the warping module and the SPADE decoder to one fp16 ONNX graph if
that file is missing (the export is the same on every machine), then builds
the engine for *this* GPU family and TensorRT version under
``models/tensorrt/`` and loads it once. The application picks it up at the
next start when ``engine.pixel_backend`` is ``auto`` (the default) or
``tensorrt``. About a minute on an RTX 5070 Ti (export 5 s, build 45 s).

Needs the ``tensorrt`` package (and ``onnx`` + ``onnxscript`` for the
export)::

    uv pip install "tensorrt-cu12==11.3.0.99" --extra-index-url https://pypi.nvidia.com
    uv pip install "onnx==1.23.2" "onnxscript==0.7.2"

Usage::

    uv run python scripts/build_tensorrt_engine.py            # build if missing
    uv run python scripts/build_tensorrt_engine.py --force    # rebuild
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sha256.avatar import liveportrait_tensorrt as trt_backend  # noqa: E402
from sha256.avatar.engine_liveportrait import DEFAULT_WEIGHTS_RELATIVE_PATH  # noqa: E402
from sha256.config.loader import load_settings  # noqa: E402
from sha256.config.paths import AppPaths  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--force", action="store_true", help="Rebuild an existing engine.")
    parser.add_argument(
        "--force-export", action="store_true", help="Re-export the ONNX file as well."
    )
    args = parser.parse_args()

    version = trt_backend.tensorrt_version()
    if version is None:
        print(
            "the tensorrt package is not installed in this environment:\n"
            '  uv pip install "tensorrt-cu12==11.3.0.99" --extra-index-url https://pypi.nvidia.com'
        )
        return 1
    import torch

    if not torch.cuda.is_available():
        print("no CUDA GPU is available; the engine is built on the GPU it will run on")
        return 1

    paths = AppPaths.resolve()
    settings = load_settings(paths)
    models = paths.resolve_relative(settings.models.root_path)
    engines = models / trt_backend.TENSORRT_RELATIVE_DIR
    capability = tuple(torch.cuda.get_device_capability(0))
    gpu = torch.cuda.get_device_name(0)
    onnx_path = engines / trt_backend.ONNX_NAME
    plan = engines / trt_backend.engine_name((capability[0], capability[1]), version)
    print(f"GPU {gpu} (sm{capability[0]}{capability[1]}), TensorRT {version}")
    print(f"engine: {plan}")

    if plan.is_file() and not args.force:
        print(f"already built ({plan.stat().st_size / 1e6:.1f} MB); --force rebuilds it")
    else:
        if not onnx_path.is_file() or args.force_export:
            try:
                import onnx  # noqa: F401
                import onnxscript  # noqa: F401
            except ImportError:
                print(
                    "the export needs onnx and onnxscript:\n"
                    '  uv pip install "onnx==1.23.2" "onnxscript==0.7.2"'
                )
                return 1
            print(f"exporting {onnx_path.name} ...")
            meta = trt_backend.export_onnx(models / DEFAULT_WEIGHTS_RELATIVE_PATH, onnx_path)
            print(f"  {meta['size_bytes'] / 1e6:.1f} MB in {meta['export_s']:.1f} s")
        else:
            print(f"ONNX present: {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
        print(f"building {plan.name} (a minute or so) ...")
        meta = trt_backend.build_engine(onnx_path, plan)
        print(
            f"  {meta['size_bytes'] / 1e6:.1f} MB, parse {meta['parse_s']:.1f} s, "
            f"build {meta['build_s']:.1f} s, {meta['layers']} layers"
        )

    started = time.perf_counter()
    engine = trt_backend.PixelEngine(plan, torch.device("cuda"))
    f_s = torch.zeros(trt_backend.F_S_SHAPE, dtype=torch.float16, device="cuda")
    kp = torch.zeros(trt_backend.KP_SHAPE, dtype=torch.float16, device="cuda")
    out = engine.run(f_s=f_s, kp_source=kp, kp_driving=kp)["image"]
    torch.cuda.synchronize()
    print(
        f"loaded and run once in {time.perf_counter() - started:.2f} s: output "
        f"{tuple(out.shape)} {out.dtype}, {engine.device_memory_bytes // (1024 * 1024)} MB "
        "of activations"
    )
    engine.close()
    print("the application will use it with engine.pixel_backend = auto (the default)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
