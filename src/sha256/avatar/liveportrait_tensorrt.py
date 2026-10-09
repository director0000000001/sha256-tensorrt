"""LivePortrait's two pixel networks on TensorRT -- stage 2 of "Phase 2 --
TensorRT" (ROADMAP, "The fast version"; measured first in
docs/benchmarks/B006-tensorrt.md, 2026-10-09).

The warping module and the SPADE decoder -- the networks that draw the
pixels -- run as one TensorRT fp16 engine. The motion extractor and every
line of ``_drive`` (the keypoints, the expression) stay in torch, bit for
bit: :class:`LivePortraitEngine` swaps only its ``_nets["warping"]`` and
``_nets["decoder"]`` for the two modules :func:`pixel_nets` returns, before
the avatar loads, and records its CUDA graph around the TensorRT enqueue as
it does around the torch modules. B006 proved the swap on the clip: the
keypoints handed to the warping network byte-identical over 601 frames, the
drawn pixels different by fp16 rounding (at most 6 grey levels in the
composite, mean 0.02), the graph's GPU time 25.8 to 14.3 ms a frame.

An engine (a ``.plan``) is tied to the GPU's compute capability and the
TensorRT version it was built with, so it is named by both and built once
per machine -- ``scripts/build_tensorrt_engine.py``, 45 s on an RTX 5070 Ti
-- from an ONNX export of the vendored networks (:func:`export_onnx`,
opset 20, fp16 weights straight from the checkpoint). Without the
``tensorrt`` package or without a plan for this GPU the engine keeps its
torch networks: ``engine.pixel_backend = "auto"`` (the default) falls back
and says so in the log; ``"tensorrt"`` refuses to start without it;
``"torch"`` never looks.

Stage 3 (2026-10-09, the portable build): the engine is built on the
user's machine at the first start. The package carries the ONNX export
(the same on every machine) and TensorRT's builder resource for the GPU
families it supports; when ``auto`` finds no plan for this GPU but the
export and the builder are there, :func:`resolve_pixel_backend` answers
``"build"`` and the engine builds the plan on a worker thread while the
start-up window counts the seconds (:func:`build_engine_watched`). A build
that fails leaves a record beside the plan (:func:`failed_marker`) so the
next start does not spend the minutes again -- delete it to retry -- and
the torch networks draw the face as before.

Install (this project's venv, outside the lock like torch itself)::

    uv pip install "tensorrt-cu12==11.3.0.99" --extra-index-url https://pypi.nvidia.com
    uv pip install "onnx==1.23.2" "onnxscript==0.7.2"   # the export only
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from sha256.utils.logging import get_logger

__all__ = [
    "ENGINE_STEM",
    "ONNX_NAME",
    "PIXEL_BACKENDS",
    "TENSORRT_RELATIVE_DIR",
    "PixelEngine",
    "build_engine",
    "build_engine_watched",
    "builder_resource_name",
    "engine_name",
    "export_onnx",
    "failed_marker",
    "has_builder",
    "libs_dir",
    "pixel_nets",
    "record_failure",
    "resolve_pixel_backend",
    "tensorrt_version",
]

_LOG = get_logger(__name__)

#: Where the engines live under the models root: ``models/tensorrt/``.
TENSORRT_RELATIVE_DIR = Path("tensorrt")
ENGINE_STEM = "liveportrait-pixels-fp16"
#: The export is the same on every machine (the checkpoint's weights, fp16).
ONNX_NAME = f"{ENGINE_STEM}.onnx"
PIXEL_BACKENDS = ("auto", "torch", "tensorrt")

# The shapes the engine feeds the two networks, batch 1 (see _drive).
F_S_SHAPE = (1, 32, 16, 64, 64)
KP_SHAPE = (1, 21, 3)
IMAGE_SHAPE = (1, 3, 512, 512)
WORKSPACE_BYTES = 4 << 30


def tensorrt_version() -> str | None:
    """The installed TensorRT's version, or ``None`` when the package is not
    importable (the usual state of a machine without the extra)."""
    try:
        import tensorrt as trt
    except Exception as exc:  # not installed, or its DLLs cannot be loaded here
        _LOG.info("The tensorrt package could not be imported: %s: %s", type(exc).__name__, exc)
        return None
    return str(trt.__version__)


def engine_name(compute_capability: tuple[int, int], version: str) -> str:
    """``liveportrait-pixels-fp16-sm120-trt11.3.0.99.plan``: a plan serves
    one GPU family and one TensorRT version."""
    major, minor = compute_capability
    return f"{ENGINE_STEM}-sm{major}{minor}-trt{version}.plan"


def builder_resource_name(compute_capability: tuple[int, int], version: str) -> str:
    """``nvinfer_builder_resource_sm120_11.dll``: the builder tactics for
    one GPU family, one DLL each, named by the library's major version.
    The runtime (``nvinfer_11.dll``) loads any engine; *building* one needs
    the resource for the GPU it runs on, so a package ships one per family
    it supports (see ``packaging/sha256.spec``)."""
    major, minor = compute_capability
    return f"nvinfer_builder_resource_sm{major}{minor}_{version.split('.')[0]}.dll"


def libs_dir() -> Path | None:
    """The folder of TensorRT's DLLs (the ``tensorrt_libs`` package, in a
    venv or in the frozen build's ``_internal``), or ``None`` without it.
    Found by its spec, not imported: importing it loads every DLL."""
    try:
        spec = importlib.util.find_spec("tensorrt_libs")
    except Exception:
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def has_builder(compute_capability: tuple[int, int], version: str, libs: Path | None) -> bool:
    """Whether an engine can be built *here*: the builder resource for this
    GPU family is in ``libs``."""
    if libs is None:
        return False
    return (libs / builder_resource_name(compute_capability, version)).is_file()


def failed_marker(plan: Path) -> Path:
    """Beside the plan: the record of a build that failed on this machine
    (:func:`record_failure`), so the next start does not spend the minutes
    again. Delete it to try again."""
    return plan.with_name(plan.name + ".failed.json")


def record_failure(plan: Path, error: BaseException, onnx: Path | None) -> Path:
    """Write the failed marker for ``plan``: what went wrong, when, from
    which export."""
    marker = failed_marker(plan)
    marker.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "plan": plan.name,
        "onnx": onnx.name if onnx is not None else None,
        "error": f"{type(error).__name__}: {error}",
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Delete this file to let the application try the build again.",
    }
    marker.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return marker


def resolve_pixel_backend(
    wanted: str,
    *,
    version: str | None,
    plan: Path | None,
    onnx: Path | None = None,
    builder: bool = False,
    failed: Path | None = None,
) -> tuple[str, str]:
    """Which networks draw the face, and why: ``("tensorrt", reason)``,
    ``("build", reason)`` -- no engine yet, but the export ``onnx`` and the
    ``builder`` for this GPU are here and no ``failed`` record stands in the
    way, so the caller builds one now -- or ``("torch", reason)``. Pure, so
    it is tested without a GPU. ``wanted`` is the setting; ``version`` the
    installed TensorRT or ``None``; ``plan`` the engine file for this GPU
    and version, or ``None``.

    Raises ``RuntimeError`` when ``"tensorrt"`` was asked for outright and is
    not there -- the caller turns that into its own error."""
    if wanted not in PIXEL_BACKENDS:
        msg = f"pixel_backend must be one of {PIXEL_BACKENDS}, not {wanted!r}"
        raise ValueError(msg)
    if wanted == "torch":
        return "torch", "pixel_backend = torch"
    if version is None:
        reason = "the tensorrt package is not installed"
    elif plan is None or not plan.is_file():
        failed_here = failed is not None and failed.is_file()
        export_here = onnx is not None and onnx.is_file()
        if plan is not None and onnx is not None and export_here and builder and not failed_here:
            return (
                "build",
                f"no engine for this GPU and TensorRT {version} yet; built from {onnx.name}",
            )
        reason = (
            f"no engine for this GPU and TensorRT {version}"
            + (f" at {plan}" if plan is not None else "")
            + "; build one with scripts/build_tensorrt_engine.py"
        )
        if failed_here and failed is not None:
            reason += f" (the last build here failed -- see {failed.name}; delete it to try again)"
        elif not export_here:
            reason += " (no ONNX export to build from)"
        elif not builder:
            reason += " (no builder for this GPU family shipped)"
    else:
        return "tensorrt", f"TensorRT {version}, {plan.name}"
    if wanted == "tensorrt":
        raise RuntimeError(reason)
    return "torch", reason


def _torch_dtype(trt: Any, trt_dtype: Any) -> Any:
    import torch

    table = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
    }
    if hasattr(trt.DataType, "BF16"):
        table[trt.DataType.BF16] = torch.bfloat16
    return table[trt_dtype]


class PixelEngine:
    """A built engine run from torch: the output buffers allocated once on
    the device, the inputs bound by address per call, one
    ``execute_async_v3`` on torch's *current* stream -- which is what lets
    the call capture into the engine's CUDA graph (B006: the replay equals
    the eager run to the bit).

    After the capture the context is not touched at replay time -- the graph
    holds the kernels -- but it owns the activation memory those kernels
    use, so this object must outlive every graph recorded through it."""

    def __init__(self, plan: Path, device: Any) -> None:
        import tensorrt as trt
        import torch

        self.plan = Path(plan)
        self.version = str(trt.__version__)
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        started = time.perf_counter()
        self.engine = self._runtime.deserialize_cuda_engine(self.plan.read_bytes())
        if self.engine is None:
            msg = f"TensorRT could not deserialise {self.plan}"
            raise RuntimeError(msg)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            # TensorRT 11.3's known issue: an FP16 GridSample network may fail
            # here on Windows with some RTX 50 cards (CUDA error 700).
            msg = f"TensorRT could not create an execution context for {self.plan}"
            raise RuntimeError(msg)
        self.load_s = time.perf_counter() - started
        self.inputs: dict[str, tuple[tuple[int, ...], Any]] = {}
        self.outputs: dict[str, Any] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = _torch_dtype(trt, self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                buffer = torch.empty(shape, dtype=dtype, device=device)
                self.outputs[name] = buffer
                self.context.set_tensor_address(name, buffer.data_ptr())
            else:
                self.inputs[name] = (shape, dtype)
        self.device_memory_bytes = int(self.engine.device_memory_size_v2)
        self._held: dict[str, Any] = {}

    def run(self, **tensors: Any) -> dict[str, Any]:
        """Bind ``tensors`` by name and enqueue; returns the output buffers
        (the same tensors every call -- clone what must outlive the next)."""
        import torch

        for name, tensor in tensors.items():
            shape, dtype = self.inputs[name]
            if tuple(tensor.shape) != shape:
                msg = f"{name}: got {tuple(tensor.shape)}, the engine wants {shape}"
                raise ValueError(msg)
            if tensor.dtype != dtype:
                tensor = tensor.to(dtype)
            tensor = tensor.contiguous()
            self._held[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            msg = "TensorRT execute_async_v3 failed"
            raise RuntimeError(msg)
        return self.outputs

    def close(self) -> None:
        self._held.clear()
        self.outputs.clear()
        self.context = None
        self.engine = None


class PixelRunner(Protocol):
    """What :func:`pixel_nets` needs of an engine: :meth:`PixelEngine.run`."""

    def run(self, **tensors: Any) -> dict[str, Any]: ...


def pixel_nets(engine: PixelRunner) -> tuple[Any, Any]:
    """The two ``_nets`` entries ``_drive`` calls -- ``warping(f_s,
    kp_source=..., kp_driving=...)["out"]`` and ``decoder(feature=...)`` --
    backed by the fused engine: the first runs it and hands the picture on
    as ``"out"``, the second passes it through. The picture is cloned so
    each avatar's graph owns its output rather than the engine's one
    buffer."""
    import torch

    class TensorRTWarping(torch.nn.Module):
        def forward(self, feature_3d: Any, kp_driving: Any, kp_source: Any) -> dict[str, Any]:
            image = engine.run(f_s=feature_3d, kp_source=kp_source, kp_driving=kp_driving)["image"]
            return {"out": image.clone(), "occlusion_map": None, "deformation": None}

    class TensorRTDecoder(torch.nn.Module):
        def forward(self, feature: Any) -> Any:
            return feature

    return TensorRTWarping(), TensorRTDecoder()


# --- the export and the build ------------------------------------------------


def export_onnx(weights_root: Path, target: Path) -> dict[str, Any]:
    """The warping module and the SPADE decoder, the checkpoint's weights in
    fp16, as one ONNX graph at opset 20 (the dynamo exporter: the torchscript
    one cannot export these in fp16). Needs ``onnx`` and ``onnxscript``."""
    import os

    import torch

    from sha256.avatar.engine_liveportrait import _DECODER_PARAMS, _WARPING_PARAMS
    from sha256.avatar.liveportrait.spade_generator import SPADEDecoder
    from sha256.avatar.liveportrait.warping_network import WarpingNetwork

    # The exporter prints a check mark; Windows' cp1252 console cannot.
    os.environ.setdefault("PYTHONUTF8", "1")
    device = torch.device("cuda")
    base = Path(weights_root) / "base_models"

    def build(net: Any, path: Path) -> Any:
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = {name.replace("module.", ""): value for name, value in state.items()}
        net.load_state_dict(state)
        return net.to(device, torch.float16).eval()

    warping = build(WarpingNetwork(**_WARPING_PARAMS), base / "warping_module.pth")  # type: ignore[no-untyped-call]
    decoder = build(SPADEDecoder(**_DECODER_PARAMS), base / "spade_generator.pth")  # type: ignore[no-untyped-call]

    class Fused(torch.nn.Module):
        """``_drive``'s last two lines as one module. The networks are
        registered as submodules, not closed over: the exporter then keeps
        their weights as the parameters they are (B006's 204 MB graph of 377
        nodes), where a closure made constants of them (265 MB)."""

        def __init__(self) -> None:
            super().__init__()
            self.warping = warping
            self.decoder = decoder

        def forward(self, f_s: Any, kp_source: Any, kp_driving: Any) -> Any:
            out = self.warping(f_s, kp_driving=kp_driving, kp_source=kp_source)["out"]
            return self.decoder(feature=out)

    inputs = (
        torch.randn(F_S_SHAPE, device=device, dtype=torch.float16),
        torch.randn(KP_SHAPE, device=device, dtype=torch.float16) * 0.1,
        torch.randn(KP_SHAPE, device=device, dtype=torch.float16) * 0.1,
    )
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with torch.no_grad():
        program = torch.onnx.export(
            Fused().eval(),
            inputs,
            dynamo=True,
            opset_version=20,
            input_names=["f_s", "kp_source", "kp_driving"],
            output_names=["image"],
            external_data=False,
        )
    assert program is not None
    program.save(str(target), external_data=False)
    elapsed = time.perf_counter() - started
    meta: dict[str, Any] = {
        "onnx": target.name,
        "size_bytes": target.stat().st_size,
        "export_s": elapsed,
        "torch": torch.__version__,
        "opset": 20,
    }
    Path(str(target) + ".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    _LOG.info("exported %s: %.1f MB in %.1f s", target.name, meta["size_bytes"] / 1e6, elapsed)
    return meta


def build_engine(onnx_path: Path, target: Path) -> dict[str, Any]:
    """An ONNX file to a TensorRT engine for this GPU: strongly typed (the
    default since TensorRT 11, so fp16 stays fp16 op by op, as in torch),
    4 GB of workspace, the build timed. Minutes at most; 45 s on a 5070 Ti."""
    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    started = time.perf_counter()
    if not parser.parse_from_file(str(onnx_path)):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        msg = f"TensorRT could not parse {onnx_path}: {errors}"
        raise RuntimeError(msg)
    parsed = time.perf_counter() - started
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - started
    if serialized is None:
        msg = f"TensorRT could not build an engine from {onnx_path}"
        raise RuntimeError(msg)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Written beside and renamed into place: a start killed mid-write must
    # not leave a half plan that the next start would try to load.
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(bytes(serialized))
    partial.replace(target)
    meta: dict[str, Any] = {
        "onnx": Path(onnx_path).name,
        "plan": target.name,
        "tensorrt": str(trt.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "layers": network.num_layers,
        "parse_s": parsed,
        "build_s": elapsed,
        "size_bytes": target.stat().st_size,
    }
    Path(str(target) + ".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    _LOG.info(
        "built %s: %.1f MB in %.1f s (TensorRT %s, %s)",
        target.name,
        meta["size_bytes"] / 1e6,
        elapsed,
        meta["tensorrt"],
        meta["gpu"],
    )
    return meta


def build_engine_watched(
    onnx_path: Path,
    target: Path,
    progress: Callable[[float], None] | None = None,
    *,
    poll_s: float = 0.5,
    build: Callable[[Path, Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """:func:`build_engine` on a worker thread, the caller's ``progress``
    told the elapsed seconds every ``poll_s`` from the calling thread -- the
    start-up window pumps its events there. The builder gives no progress
    of its own; the seconds are the whole story. An error on the worker is
    raised here. ``build`` replaces the builder in tests."""
    builder = build_engine if build is None else build
    result: dict[str, Any] = {}
    errors: list[BaseException] = []

    def work() -> None:
        try:
            result.update(builder(onnx_path, target))
        except BaseException as exc:  # carried to the caller's thread
            errors.append(exc)

    thread = threading.Thread(target=work, name="tensorrt-build", daemon=True)
    started = time.perf_counter()
    thread.start()
    while thread.is_alive():
        thread.join(poll_s)
        if progress is not None:
            progress(time.perf_counter() - started)
    if errors:
        raise errors[0]
    return result
