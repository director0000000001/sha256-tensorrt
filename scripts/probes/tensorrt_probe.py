"""TensorRT for LivePortrait's two pixel networks, measured outside the
application's frame path -- stage 1 of "Phase 2 -- TensorRT" (ROADMAP, "The
fast version"), written up in docs/benchmarks/B006-tensorrt.md.

The rule: the motion extractor and every line of ``_drive`` -- the
keypoints, the expression -- stay in torch, bit for bit. Only the warping
module and the SPADE decoder, the networks that draw the pixels, are
compiled. Nothing in ``engine_liveportrait.py`` changes: this probe swaps the
two entries of the engine's ``_nets`` *in its own process* before the avatar
loads, so the engine's own ``_capture`` records its CUDA graph around the
TensorRT engine exactly as it records it around the torch modules.

Sub-commands, in the order they are used::

    export   the two networks (fused, or one of them) to ONNX, opset 20
    build    an ONNX file to a TensorRT engine (``.plan``), the build timed
    check    an engine against torch on the avatar's own features
    clip     the clip through the application's PipelineController with the
             torch networks or the TensorRT engine: hashes, timings, the
             keypoints the warping network was given, stills
    bench    the two networks alone on the recorded keypoints: torch's CUDA
             graph against TensorRT, p50/p90, and the drawn face compared
    diff     two runs' stills (face and composite) through still_diff
    kp       two runs' recorded keypoints compared to the byte

Stills are his face; they go under recordings/budget/trt/ and are deleted
once read. Engines and ONNX files go under models/tensorrt/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frame_budget import (  # noqa: E402
    ENGINE_KEYS,
    ClipCamera,
    TimedEngine,
    TimedTracker,
    _Avatars,
    frame_hash,
    print_table,
    summarise,
)
from still_diff import compare as still_compare  # noqa: E402

from sha256.avatar.engine_liveportrait import (  # noqa: E402
    _DECODER_PARAMS,
    _WARPING_PARAMS,
    DEFAULT_WEIGHTS_RELATIVE_PATH,
    NUM_KP,
    LivePortraitEngine,
)

MODELS_DIR = REPO_ROOT / "models" / "tensorrt"
OUT_DIR = REPO_ROOT / "recordings" / "budget" / "trt"

# The shapes the engine feeds the two networks, batch 1 (see _drive).
F_S_SHAPE = (1, 32, 16, 64, 64)
KP_SHAPE = (1, NUM_KP, 3)
WARPED_SHAPE = (1, 256, 64, 64)
IMAGE_SHAPE = (1, 3, 512, 512)


# --- the networks, as the engine builds them -------------------------------


def load_pixel_nets(precision: str, device: Any) -> tuple[Any, Any]:
    """The warping module and the SPADE decoder with the engine's own
    hyper-parameters and weights -- the same ``build`` as
    ``LivePortraitEngine._load_networks``, in fp16 (the engine's) or fp32
    (the checkpoint's own)."""
    import torch

    from sha256.avatar.liveportrait.spade_generator import SPADEDecoder
    from sha256.avatar.liveportrait.warping_network import WarpingNetwork

    dtype = torch.float16 if precision == "fp16" else torch.float32
    base = REPO_ROOT / "models" / DEFAULT_WEIGHTS_RELATIVE_PATH / "base_models"

    def build(net: Any, path: Path) -> Any:
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = {name.replace("module.", ""): value for name, value in state.items()}
        net.load_state_dict(state)
        return net.to(device, dtype).eval()

    warping = build(WarpingNetwork(**_WARPING_PARAMS), base / "warping_module.pth")
    decoder = build(SPADEDecoder(**_DECODER_PARAMS), base / "spade_generator.pth")
    return warping, decoder


def fused_module(warping: Any, decoder: Any) -> Any:
    import torch

    class Fused(torch.nn.Module):
        """``_drive``'s last two lines as one module."""

        def __init__(self) -> None:
            super().__init__()
            self.warping = warping
            self.decoder = decoder

        def forward(self, f_s: Any, kp_source: Any, kp_driving: Any) -> Any:
            out = self.warping(f_s, kp_driving=kp_driving, kp_source=kp_source)["out"]
            return self.decoder(feature=out)

    return Fused().eval()


def warping_module(warping: Any) -> Any:
    import torch

    class WarpOnly(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.warping = warping

        def forward(self, f_s: Any, kp_source: Any, kp_driving: Any) -> Any:
            return self.warping(f_s, kp_driving=kp_driving, kp_source=kp_source)["out"]

    return WarpOnly().eval()


def decoder_module(decoder: Any) -> Any:
    import torch

    class SpadeOnly(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = decoder

        def forward(self, feature: Any) -> Any:
            return self.decoder(feature=feature)

    return SpadeOnly().eval()


# --- export ------------------------------------------------------------------


def cmd_export(args: argparse.Namespace) -> int:
    import torch

    device = torch.device("cuda")
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    warping, decoder = load_pixel_nets(args.precision, device)
    if args.parts == "fused":
        module = fused_module(warping, decoder)
        inputs = (
            torch.randn(F_S_SHAPE, device=device, dtype=dtype),
            torch.randn(KP_SHAPE, device=device, dtype=dtype) * 0.1,
            torch.randn(KP_SHAPE, device=device, dtype=dtype) * 0.1,
        )
        names = ["f_s", "kp_source", "kp_driving"]
        outputs = ["image"]
    elif args.parts == "warping":
        module = warping_module(warping)
        inputs = (
            torch.randn(F_S_SHAPE, device=device, dtype=dtype),
            torch.randn(KP_SHAPE, device=device, dtype=dtype) * 0.1,
            torch.randn(KP_SHAPE, device=device, dtype=dtype) * 0.1,
        )
        names = ["f_s", "kp_source", "kp_driving"]
        outputs = ["out"]
    else:
        module = decoder_module(decoder)
        inputs = (torch.randn(WARPED_SHAPE, device=device, dtype=dtype),)
        names = ["feature"]
        outputs = ["image"]

    if args.grid_sample_fp32:
        # The fallback for TensorRT 11.3's known issue (an FP16 GridSample
        # network failing to create its execution context on Windows with
        # RTX 50): the two grid_sample calls in fp32, everything else as is.
        import torch.nn.functional as functional

        original = functional.grid_sample

        def grid_sample_fp32(inp: Any, grid: Any, *a: Any, **k: Any) -> Any:
            return original(inp.float(), grid.float(), *a, **k).to(inp.dtype)

        functional.grid_sample = grid_sample_fp32

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "-gs32" if args.grid_sample_fp32 else ""
    target = MODELS_DIR / f"{args.parts}-{args.precision}{suffix}.onnx"
    exporter = None
    started = time.perf_counter()
    with torch.no_grad():
        reference = module(*inputs)
    errors: list[str] = []
    for dynamo in (True, False) if args.exporter == "auto" else (args.exporter == "dynamo",):
        try:
            print(f"export: {'dynamo' if dynamo else 'torchscript'} exporter, opset {args.opset}")
            if dynamo:
                program = torch.onnx.export(
                    module,
                    inputs,
                    dynamo=True,
                    opset_version=args.opset,
                    input_names=names,
                    output_names=outputs,
                    external_data=False,
                )
                assert program is not None
                program.save(str(target), external_data=False)
            else:
                torch.onnx.export(
                    module,
                    inputs,
                    str(target),
                    dynamo=False,
                    opset_version=args.opset,
                    input_names=names,
                    output_names=outputs,
                    do_constant_folding=True,
                )
            exporter = "dynamo" if dynamo else "torchscript"
            break
        except Exception as exc:
            errors.append(f"{'dynamo' if dynamo else 'torchscript'}: {type(exc).__name__}: {exc}")
            print(f"  failed: {errors[-1][:800]}")
    if exporter is None:
        print("export failed with every exporter")
        return 1
    elapsed = time.perf_counter() - started

    import onnx

    model = onnx.load(str(target), load_external_data=True)
    onnx.checker.check_model(model, full_check=False)
    ops: dict[str, int] = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    grid = [n for n in model.graph.node if n.op_type == "GridSample"]
    opsets = {(imp.domain or "ai.onnx"): imp.version for imp in model.opset_import}
    print(
        f"exported {target.name}: {target.stat().st_size / 1e6:.1f} MB in {elapsed:.1f} s, "
        f"{exporter} exporter, opsets {opsets}, {len(model.graph.node)} nodes"
    )
    print(f"  ops: {dict(sorted(ops.items(), key=lambda kv: -kv[1]))}")
    print(f"  GridSample nodes: {len(grid)}")
    for n in grid:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        print(f"    {n.name}: {attrs}")
    meta = {
        "file": target.name,
        "parts": args.parts,
        "precision": args.precision,
        "grid_sample_fp32": args.grid_sample_fp32,
        "exporter": exporter,
        "exporter_errors": errors,
        "opset": opsets,
        "nodes": len(model.graph.node),
        "ops": ops,
        "size_bytes": target.stat().st_size,
        "export_s": elapsed,
        "torch": torch.__version__,
        "onnx": onnx.__version__,
    }
    Path(str(target) + ".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    del reference
    return 0


# --- TensorRT ----------------------------------------------------------------


def _trt() -> Any:
    import tensorrt as trt

    return trt


def _torch_dtype(trt_dtype: Any) -> Any:
    import torch

    trt = _trt()
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


def cmd_build(args: argparse.Namespace) -> int:
    trt = _trt()
    import torch

    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 0
    typed = "strongly typed"
    if args.fp16_flag:
        # The weakly typed recipe of TensorRT 10: fp32 ONNX, TensorRT picks
        # fp16 where it may. Since 11.0 strongly typed is the default; the
        # weakly typed network needs its own creation flag if one exists.
        names = [n for n in dir(trt.NetworkDefinitionCreationFlag) if not n.startswith("_")]
        print(f"network creation flags available: {names}")
        if "WEAKLY_TYPED" in names:
            flags |= 1 << int(trt.NetworkDefinitionCreationFlag.WEAKLY_TYPED)
            typed = "weakly typed + FP16 flag"
        elif "STRONGLY_TYPED" in names and not args.fp16_flag_strong:
            typed = "weakly typed (default of this version) + FP16 flag"
        else:
            print("no weakly typed flag in this TensorRT; the FP16 flag would be ignored")
            return 1
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    started = time.perf_counter()
    if not parser.parse_from_file(str(args.onnx)):
        for i in range(parser.num_errors):
            print(f"parser error: {parser.get_error(i)}")
        return 1
    parsed = time.perf_counter() - started
    print(
        f"parsed {args.onnx.name} in {parsed:.1f} s: {network.num_layers} layers, "
        f"{network.num_inputs} inputs, {network.num_outputs} outputs ({typed})"
    )
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"  input {t.name}: {tuple(t.shape)} {t.dtype}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"  output {t.name}: {tuple(t.shape)} {t.dtype}")
    layer_types: dict[str, int] = {}
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        key = str(layer.type).split(".")[-1]
        layer_types[key] = layer_types.get(key, 0) + 1
    print(f"  layer types: {dict(sorted(layer_types.items(), key=lambda kv: -kv[1]))}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb << 30)
    if args.fp16_flag:
        if not hasattr(trt.BuilderFlag, "FP16"):
            print(
                "this TensorRT has no BuilderFlag.FP16: every network is strongly typed, "
                "the precision is the ONNX file's own (11.x)"
            )
            return 1
        config.set_flag(trt.BuilderFlag.FP16)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if args.builder_level is not None:
        config.builder_optimization_level = args.builder_level
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - started
    if serialized is None:
        print("build failed")
        return 1
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = args.out or MODELS_DIR / (
        args.onnx.stem + ("-fp16flag" if args.fp16_flag else "") + ".plan"
    )
    target.write_bytes(bytes(serialized))
    size = target.stat().st_size
    print(
        f"built {target.name}: {size / 1e6:.1f} MB in {elapsed:.1f} s "
        f"(TensorRT {trt.__version__}, {torch.cuda.get_device_name(0)})"
    )
    meta = {
        "onnx": args.onnx.name,
        "plan": target.name,
        "typed": typed,
        "fp16_flag": args.fp16_flag,
        "builder_optimization_level": args.builder_level,
        "workspace_gb": args.workspace_gb,
        "parse_s": parsed,
        "build_s": elapsed,
        "size_bytes": size,
        "tensorrt": trt.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "driver": _driver_version(),
        "layers_parsed": network.num_layers,
        "layer_types": layer_types,
    }
    Path(str(target) + ".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return 0


def _driver_version() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return out.stdout.strip()
    except Exception:
        return "?"


class TrtEngine:
    """A built engine run from torch: static output buffers on the device,
    inputs by address, one ``execute_async_v3`` on torch's current stream --
    which is what lets it capture into the engine's CUDA graph."""

    def __init__(self, plan: Path, device: Any) -> None:
        import torch

        trt = _trt()
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        started = time.perf_counter()
        self.engine = self.runtime.deserialize_cuda_engine(plan.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not deserialise {plan}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(
                f"could not create an execution context for {plan} "
                "(TensorRT 11.3's known FP16 GridSample issue on Windows / RTX 50?)"
            )
        self.load_s = time.perf_counter() - started
        self.inputs: dict[str, tuple[tuple[int, ...], Any]] = {}
        self.outputs: dict[str, Any] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = _torch_dtype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                buffer = torch.empty(shape, dtype=dtype, device=device)
                self.outputs[name] = buffer
                self.context.set_tensor_address(name, buffer.data_ptr())
            else:
                self.inputs[name] = (shape, dtype)
        self.device_memory_bytes = int(self.engine.device_memory_size_v2)
        self.casts = 0
        self._held: dict[str, Any] = {}

    def run(self, **tensors: Any) -> dict[str, Any]:
        import torch

        for name, tensor in tensors.items():
            shape, dtype = self.inputs[name]
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name}: got {tuple(tensor.shape)}, the engine wants {shape}")
            if tensor.dtype != dtype:
                # An fp32 engine fed the application's fp16 tensors (the
                # precision reference): the cast is one more kernel on the
                # stream, inside the graph like everything else.
                tensor = tensor.to(dtype)
                self.casts += 1
            tensor = tensor.contiguous()
            self._held[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("execute_async_v3 failed")
        return self.outputs


def trt_as_nets(engine: TrtEngine, kp_buffer: Any, parts: str) -> tuple[Any, Any]:
    """The two ``_nets`` entries the engine's ``_drive`` calls, backed by the
    TensorRT engine. ``kp_buffer`` records the keypoints the warping network
    is given (``kp_driving``, the expression) -- the copy captures into the
    graph with everything else."""
    import torch

    if parts == "fused":

        class Warping(torch.nn.Module):
            def forward(self, feature_3d: Any, kp_driving: Any, kp_source: Any) -> Any:
                kp_buffer.copy_(kp_driving)
                image = engine.run(
                    f_s=feature_3d.contiguous(),
                    kp_source=kp_source.contiguous(),
                    kp_driving=kp_driving.contiguous(),
                )["image"]
                return {"out": image, "occlusion_map": None, "deformation": None}

        class Decoder(torch.nn.Module):
            def forward(self, feature: Any) -> Any:
                # The fused engine already drew the picture.
                return feature

        return Warping(), Decoder()
    raise ValueError(parts)


def torch_recording_nets(warping: Any, kp_buffer: Any) -> Any:
    """The torch warping module with the same keypoint recording, so the torch
    run's graph differs from the application's by one copy and nothing
    else -- proven by its hashes against the reference run."""
    import torch

    class Recording(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = warping

        def forward(self, feature_3d: Any, kp_driving: Any, kp_source: Any) -> Any:
            kp_buffer.copy_(kp_driving)
            return self.inner(feature_3d, kp_driving=kp_driving, kp_source=kp_source)

    return Recording()


# --- the engine and the avatar, as frame_budget builds them -------------------


def make_engine() -> tuple[Any, Any, Any, Any]:
    from sha256.avatar.repository import FilesystemAvatarRepository
    from sha256.config.loader import load_settings
    from sha256.config.paths import AppPaths
    from sha256.tracking.mediapipe_tracker import (
        DEFAULT_MODEL_RELATIVE_PATH,
        MediaPipeFaceTracker,
    )

    paths = AppPaths.resolve()
    settings = load_settings(paths)
    models = paths.resolve_relative(settings.models.root_path)
    repository = FilesystemAvatarRepository(paths.resolve_relative(settings.avatars.library_path))
    tracker = MediaPipeFaceTracker(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        smoothing=settings.tracking.smoothing,
        mouth_smoothing=settings.tracking.mouth_smoothing,
        min_detection_confidence=settings.tracking.min_detection_confidence,
        min_presence_confidence=settings.tracking.min_presence_confidence,
        min_tracking_confidence=settings.tracking.min_tracking_confidence,
        redetect_in_crop=settings.tracking.redetect_in_crop,
    )
    engine = LivePortraitEngine(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        weights=models / DEFAULT_WEIGHTS_RELATIVE_PATH,
        # The probe swaps the networks itself (install_nets); the engine's
        # own stage-2 backend (B006, "pixel_backend") stays out of the way.
        pixel_backend="torch",
    )
    return engine, tracker, repository, settings


def set_deterministic() -> None:
    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def install_nets(engine: LivePortraitEngine, args: argparse.Namespace) -> tuple[Any, Any]:
    """Swap the engine's two pixel networks for the probe's, before the
    avatar loads. Returns (kp_buffer, trt_engine or None)."""
    import torch

    kp_buffer = torch.zeros(KP_SHAPE, dtype=torch.float16, device=engine._device)
    trt_engine = None
    if args.nets == "trt":
        trt_engine = TrtEngine(args.plan, engine._device)
        warping, decoder = trt_as_nets(trt_engine, kp_buffer, "fused")
        engine._nets["warping"] = warping
        engine._nets["decoder"] = decoder
        print(
            f"nets: TensorRT {trt_engine.trt.__version__}, {args.plan.name}, loaded in "
            f"{trt_engine.load_s:.2f} s, {trt_engine.device_memory_bytes / 1e6:.0f} MB activations"
        )
    else:
        engine._nets["warping"] = torch_recording_nets(engine._nets["warping"], kp_buffer)
        print("nets: torch (the application's), the keypoints recorded")
    return kp_buffer, trt_engine


def cmd_check(args: argparse.Namespace) -> int:
    """The engine against torch on the avatar's own features: the portrait
    drawn with its own keypoints, and with a nudged copy."""
    import torch

    engine, _tracker, repository, _settings = make_engine()
    engine.start()
    if args.deterministic:
        set_deterministic()
    descriptor = repository.get(args.avatar)
    engine.load_avatar(descriptor)
    loaded = engine._loaded[descriptor.id]
    f_s = loaded.f_s.contiguous()
    x_s = loaded.x_s.half().contiguous()
    fused = fused_module(engine._nets["warping"], engine._nets["decoder"])
    trt_engine = TrtEngine(args.plan, engine._device)
    print(
        f"engine {args.plan.name}: loaded in {trt_engine.load_s:.2f} s, "
        f"{trt_engine.device_memory_bytes / 1e6:.0f} MB device memory, "
        f"inputs {trt_engine.inputs}, outputs "
        f"{ {k: (tuple(v.shape), str(v.dtype)) for k, v in trt_engine.outputs.items()} }"
    )
    torch.manual_seed(0)
    cases = {
        "portrait itself": x_s,
        "nudged 0.02": (x_s.float() + 0.02 * torch.randn_like(x_s.float())).half(),
    }
    with torch.no_grad():
        for name, kp_d in cases.items():
            a = fused(f_s, x_s, kp_d.contiguous())
            b = trt_engine.run(f_s=f_s, kp_source=x_s, kp_driving=kp_d.contiguous())["image"]
            torch.cuda.synchronize()
            a8 = a[0].float().clamp(0, 1).mul(255).to(torch.uint8)
            b8 = b[0].float().clamp(0, 1).mul(255).to(torch.uint8)
            d = (a8.int() - b8.int()).abs()
            print(
                f"{name}: torch fp16 vs TensorRT on the 512 face -- max {int(d.max())}, "
                f"mean {float(d.float().mean()):.4f}, pixels differing "
                f"{int((d.amax(0) > 0).sum())} of {512 * 512}, "
                f"raw |diff| max {float((a.float() - b.float()).abs().max()):.5f}"
            )
    if trt_engine.casts:
        print(f"  (inputs cast to the engine's dtype: {trt_engine.casts} casts)")
    # Does the enqueue capture into a CUDA graph?
    kp_static = x_s.clone()
    with torch.no_grad():
        for _ in range(3):
            trt_engine.run(f_s=f_s, kp_source=x_s, kp_driving=kp_static)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                out = trt_engine.run(f_s=f_s, kp_source=x_s, kp_driving=kp_static)["image"]
            graph.replay()
            torch.cuda.synchronize()
            eager = trt_engine.run(f_s=f_s, kp_source=x_s, kp_driving=kp_static)["image"].clone()
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            same = bool(torch.equal(out, eager))
            print(f"CUDA graph capture of the TensorRT enqueue: yes; replay == eager: {same}")
        except Exception as exc:
            print(f"CUDA graph capture of the TensorRT enqueue: NO -- {type(exc).__name__}: {exc}")
    engine.stop()
    return 0


# --- the clip through the application -------------------------------------------


def cmd_clip(args: argparse.Namespace) -> int:
    import cv2
    import torch

    from sha256.core.types import RenderedFrame
    from sha256.performance.monitor import InProcessPerformanceMonitor
    from sha256.pipeline.controller import PipelineComponents, PipelineController
    from sha256.rendering.null import NullRenderer
    from sha256.utils.timing import now
    from sha256.virtual_camera.null import NullVirtualCameraSink

    stills = {int(s) for s in args.stills.split(",") if s.strip()}
    engine, tracker, repository, _settings = make_engine()
    engine.start()
    if args.deterministic:
        set_deterministic()
        print("deterministic: cudnn.benchmark off, cudnn.deterministic on")
    kp_buffer, trt_engine = install_nets(engine, args)
    descriptor = repository.get(args.avatar)
    started = time.perf_counter()
    engine.load_avatar(descriptor)
    print(f"avatar {args.avatar} loaded in {time.perf_counter() - started:.1f} s (graph captured)")
    loaded = engine._loaded[descriptor.id]

    camera = ClipCamera(args.clip, first=args.first, last=args.last, pace=args.pace, fps=None)
    timed_tracker = TimedTracker(tracker)
    timed_engine = TimedEngine(engine)
    monitor = InProcessPerformanceMonitor(window_size=200_000)
    sink = NullVirtualCameraSink()
    sink.configure(1280, 720, 30.0)
    components = PipelineComponents(
        camera=camera,  # type: ignore[arg-type]
        tracker=timed_tracker,  # type: ignore[arg-type]
        renderer=NullRenderer(width=1280, height=720),
        virtual_camera=sink,
        avatars=_Avatars(timed_engine, descriptor),  # type: ignore[arg-type]
        monitor=monitor,
    )
    controller = PipelineController(components, publish_output=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    rows: list[dict[str, Any]] = []
    kps: list[np.ndarray] = []
    kp_frames: list[int] = []

    def record(frame: RenderedFrame, total_ms: float | None) -> None:
        torch.cuda.synchronize()
        rendered = "launch_ms" in engine.last_timings
        row: dict[str, Any] = {
            "frame_id": frame.metadata.frame_id,
            "hash": frame_hash(frame),
            "tracker": timed_tracker.ms,
            "animate": timed_engine.ms,
            "total": total_ms,
            "latency_ms": frame.pipeline_latency * 1000.0,
            "published_at": now(),
            "rendered": rendered,
            "passthrough": bool(frame.is_passthrough),
        }
        for key in ENGINE_KEYS:
            value = engine.last_timings.get(f"{key}_ms")
            row[key] = None if value is None else float(value)
        rows.append(row)
        if rendered:
            kps.append(kp_buffer.cpu().numpy().copy())
            kp_frames.append(frame.metadata.frame_id)
        if frame.metadata.frame_id in stills:
            composite = OUT_DIR / f"{tag}-f{frame.metadata.frame_id}-composite.png"
            cv2.imwrite(str(composite), np.ascontiguousarray(frame.data))
            face = loaded.last[0] if loaded.last is not None else None
            if face is not None and not isinstance(face, np.ndarray):
                picture = face.round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                cv2.imwrite(str(OUT_DIR / f"{tag}-f{frame.metadata.frame_id}-face.png"), picture)
            print(f"still written: {composite.name} and its face (his face; delete when read)")

    wall_started = time.perf_counter()
    runner_stats = None
    if args.mode == "once":
        controller.start()
        try:
            while not camera.exhausted:
                started = time.perf_counter()
                frame = controller.process_once()
                total_ms = (time.perf_counter() - started) * 1000.0
                if frame is None:
                    continue
                record(frame, total_ms)
        finally:
            controller.stop()
    else:
        # The application's own two threads (frame_budget's --mode threads):
        # the camera and the tracker on one, the render on the other.
        import threading

        from sha256.pipeline.runner import PipelineRunner, PipelineSnapshot

        done = threading.Event()

        def observer(snapshot: PipelineSnapshot) -> None:
            if snapshot.frame is None:
                return
            record(snapshot.frame, None)
            camera.release(snapshot.frame_id)
            if snapshot.frame_id >= camera.last:
                done.set()

        runner = PipelineRunner(controller)
        runner.add_observer(observer)
        runner.start()
        try:
            last_count = -1
            quiet_since = time.perf_counter()
            while not done.is_set():
                time.sleep(0.05)
                if runner.last_error:
                    print(f"pipeline error: {runner.last_error}")
                    break
                if len(rows) != last_count:
                    last_count = len(rows)
                    quiet_since = time.perf_counter()
                elif camera.exhausted and time.perf_counter() - quiet_since > 2.0:
                    break
                elif time.perf_counter() - quiet_since > 30.0:
                    print("no frame published for 30 s; giving up")
                    break
        finally:
            runner.stop()
        runner_stats = str(runner.stats)
        print(f"runner stats: {runner_stats}; source frames skipped: {camera.skipped}")
    wall = time.perf_counter() - wall_started
    engine.stop()

    measured = rows[args.warmup :] if len(rows) > args.warmup else rows
    print(
        f"\n{tag} ({args.mode} / {args.pace}): {len(rows)} frames published, "
        f"{len(measured)} measured, "
        f"{sum(1 for r in rows if r['rendered'])} rendered (the rest held), wall {wall:.1f} s"
    )
    if len(measured) >= 2:
        intervals = np.diff([r["published_at"] for r in measured]) * 1000.0
        print(
            f"  publish interval ms: p50 {np.percentile(intervals, 50):.2f}  "
            f"p90 {np.percentile(intervals, 90):.2f}  -> {1000.0 / np.mean(intervals):.1f} fps"
        )
    keys = ["tracker", "animate", *ENGINE_KEYS, "latency_ms"]
    if args.mode == "once":
        keys.append("total")
    print_table("per frame, ms", summarise(measured, keys))

    payload = {
        "clip": str(args.clip),
        "first": args.first,
        "last": camera.last,
        "mode": args.mode,
        "pace": args.pace,
        "hair": False,
        "runner_stats": runner_stats,
        "source_frames_skipped": camera.skipped,
        "avatar": args.avatar,
        "nets": args.nets,
        "plan": None if args.plan is None else args.plan.name,
        "deterministic": args.deterministic,
        "tensorrt": None if trt_engine is None else trt_engine.trt.__version__,
        "torch": torch.__version__,
        "rows": rows,
    }
    out = OUT_DIR / f"{tag}.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    np.save(OUT_DIR / f"{tag}-kp.npy", np.stack(kps) if kps else np.zeros((0, *KP_SHAPE)))
    np.save(OUT_DIR / f"{tag}-kp-frames.npy", np.array(kp_frames, dtype=np.int64))
    print(f"rows written: {out}; keypoints: {tag}-kp.npy ({len(kps)} frames)")

    if args.compare is not None:
        return compare_hashes(out, args.compare)
    return 0


def compare_hashes(mine_path: Path, theirs_path: Path) -> int:
    mine_rows = json.loads(mine_path.read_text(encoding="utf-8"))["rows"]
    theirs_rows = json.loads(theirs_path.read_text(encoding="utf-8"))["rows"]
    mine = {r["frame_id"]: r["hash"] for r in mine_rows}
    theirs = {r["frame_id"]: r["hash"] for r in theirs_rows}
    shared = sorted(set(mine) & set(theirs))
    same = [f for f in shared if mine[f] == theirs[f]]
    differ = [f for f in shared if mine[f] != theirs[f]]
    print(
        f"\ncompare {mine_path.name} against {theirs_path.name}: {len(shared)} frames in both, "
        f"{len(same)} identical, {len(differ)} differ"
    )
    if differ:
        print(f"  differing frames: {differ[:20]}{' ...' if len(differ) > 20 else ''}")
    return 0 if not differ else 2


def cmd_hashes(args: argparse.Namespace) -> int:
    return compare_hashes(args.a, args.b)


def cmd_kp(args: argparse.Namespace) -> int:
    a = np.load(args.a)
    b = np.load(args.b)
    fa = np.load(args.a.with_name(args.a.stem + "-frames.npy"))
    fb = np.load(args.b.with_name(args.b.stem + "-frames.npy"))
    print(
        f"{args.a.name}: {a.shape} {a.dtype} ({len(fa)} frames); {args.b.name}: {b.shape} {b.dtype}"
    )
    if a.shape != b.shape or not np.array_equal(fa, fb):
        print("the two runs rendered different frames")
        return 2
    same = sum(1 for i in range(len(a)) if a[i].tobytes() == b[i].tobytes())
    worst = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if len(a) else 0.0
    print(
        f"keypoints handed to the warping network: {same} of {len(a)} frames byte-identical, "
        f"largest |difference| {worst:.3g}"
    )
    return 0 if same == len(a) else 2


# --- the two networks alone --------------------------------------------------------


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 90)),
        float(np.percentile(arr, 99)),
    )


def face_diff(reference: list[Any], pictures: list[Any]) -> dict[str, Any]:
    """still_diff's numbers over a run: per frame the pixels that differ,
    the largest difference, the 99th percentile over the face and the mean,
    then the median and the worst over frames."""
    import torch

    per_max, per_mean, per_count, per_p99 = [], [], [], []
    for a, b in zip(reference, pictures, strict=True):
        d = (a.int() - b.int()).abs()
        per_pixel = d.amax(0)
        per_max.append(int(per_pixel.max()))
        per_mean.append(float(d.float().mean()))
        per_count.append(int((per_pixel > 0).sum()))
        per_p99.append(float(torch.quantile(per_pixel.float().flatten(), 0.99)))
    return {
        "frames": len(pictures),
        "max_of_max": max(per_max),
        "p50_of_max": float(np.percentile(per_max, 50)),
        "p99_whole_p50": float(np.percentile(per_p99, 50)),
        "p99_whole_max": max(per_p99),
        "mean_abs_p50": float(np.percentile(per_mean, 50)),
        "mean_abs_max": max(per_mean),
        "pixels_differing_p50": float(np.percentile(per_count, 50)),
        "pixels_differing_max": max(per_count),
        "fraction_differing_p50": float(np.percentile(per_count, 50)) / (512 * 512),
    }


def cmd_bench(args: argparse.Namespace) -> int:
    import torch

    engine, _tracker, repository, _settings = make_engine()
    engine.start()  # cudnn.benchmark on: the application's own setting
    if args.deterministic:
        set_deterministic()
    descriptor = repository.get(args.avatar)
    engine.load_avatar(descriptor)
    loaded = engine._loaded[descriptor.id]
    f_s = loaded.f_s.contiguous()
    x_s = loaded.x_s.half().contiguous()
    kps = torch.from_numpy(np.load(args.kp)).to(engine._device)
    if args.frames:
        kps = kps[: args.frames]
    n = int(kps.shape[0])
    print(f"{n} recorded keypoint sets from {args.kp.name}; f_s {tuple(f_s.shape)} {f_s.dtype}")

    warping, decoder = engine._nets["warping"], engine._nets["decoder"]
    fused = fused_module(warping, decoder)
    kp_static = x_s.clone()
    results: dict[str, Any] = {}
    faces: dict[str, Any] = {}

    def time_loop(name: str, fn: Any, keep: bool) -> None:
        times: list[float] = []
        pictures = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            for _ in range(10):
                kp_static.copy_(kps[0])
                fn()
            torch.cuda.synchronize()
            for i in range(n):
                kp_static.copy_(kps[i])
                torch.cuda.synchronize()
                start.record()
                out = fn()
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
                if keep:
                    pictures.append(out[0].float().clamp(0, 1).mul(255).to(torch.uint8).clone())
        p50, p90, p99 = _percentiles(times)
        results[name] = {"p50": p50, "p90": p90, "p99": p99, "mean": float(np.mean(times))}
        print(f"  {name:<44} p50 {p50:7.2f}  p90 {p90:7.2f}  p99 {p99:7.2f} ms")
        if keep:
            faces[name] = pictures

    print("\nthe two networks alone, ms on the GPU (CUDA events), per frame:")
    # torch, eager
    time_loop("torch fp16, eager", lambda: fused(f_s, x_s, kp_static), keep=False)
    # torch, one CUDA graph -- the application's way
    with torch.no_grad():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fused(f_s, x_s, kp_static)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch_out = fused(f_s, x_s, kp_static)

    def torch_graph() -> Any:
        graph.replay()
        return torch_out

    time_loop("torch fp16, CUDA graph (the application's)", torch_graph, keep=True)

    # the separate torch networks, for the split
    with torch.no_grad():
        warped_static = warping(f_s, kp_driving=kp_static, kp_source=x_s)["out"].clone()
    time_loop(
        "  torch warping alone, eager",
        lambda: warping(f_s, kp_driving=kp_static, kp_source=x_s)["out"],
        keep=False,
    )
    time_loop("  torch spade alone, eager", lambda: decoder(feature=warped_static), keep=False)

    # The checkpoint's own fp32 arithmetic on the very same fp16 inputs: the
    # precision reference, so the fp16 rounding's own footprint is known
    # before TensorRT's difference is read.
    warping32, decoder32 = load_pixel_nets("fp32", engine._device)
    fused32 = fused_module(warping32, decoder32)
    f_s32 = f_s.float()
    x_s32 = x_s.float()
    time_loop(
        "torch fp32, eager (precision reference)",
        lambda: fused32(f_s32, x_s32, kp_static.float()),
        keep=True,
    )

    for plan in args.plan:
        trt_engine = TrtEngine(plan, engine._device)
        label = plan.stem
        if "f_s" in trt_engine.inputs and "image" in trt_engine.outputs:

            def trt_eager(e: TrtEngine = trt_engine) -> Any:
                return e.run(f_s=f_s, kp_source=x_s, kp_driving=kp_static)["image"]

            time_loop(f"TensorRT {label}, eager", trt_eager, keep=True)
            # inside a CUDA graph, as the engine would replay it
            try:
                with torch.no_grad():
                    for _ in range(3):
                        trt_eager()
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        trt_out = trt_eager()

                def trt_graph(g: Any = g, out: Any = trt_out) -> Any:
                    g.replay()
                    return out

                time_loop(f"TensorRT {label}, CUDA graph", trt_graph, keep=True)
            except Exception as exc:
                print(f"  TensorRT {label} does not capture into a CUDA graph: {exc}")
        elif "f_s" in trt_engine.inputs and "out" in trt_engine.outputs:
            time_loop(
                f"  TensorRT {label} (warping alone), eager",
                lambda e=trt_engine: e.run(f_s=f_s, kp_source=x_s, kp_driving=kp_static)["out"],
                keep=False,
            )
        elif "feature" in trt_engine.inputs:
            time_loop(
                f"  TensorRT {label} (spade alone), eager",
                lambda e=trt_engine: e.run(feature=warped_static)["image"],
                keep=False,
            )
        results[f"{label} load_s"] = trt_engine.load_s
        results[f"{label} device_memory_mb"] = trt_engine.device_memory_bytes / 1e6
        del trt_engine

    # the drawn face, every variant against the application's torch fp16
    # graph and against the fp32 reference, every frame
    diffs: dict[str, Any] = {}
    app = "torch fp16, CUDA graph (the application's)"
    ref32 = "torch fp32, eager (precision reference)"
    for base in (app, ref32):
        reference = faces.get(base)
        if reference is None:
            continue
        print(f"\nthe 512x512 face against [{base}], grey levels, over all frames:")
        for name, pictures in faces.items():
            if name == base:
                continue
            summary = face_diff(reference, pictures)
            diffs[f"{name} | vs | {base}"] = summary
            print(
                f"  {name}: max diff {summary['max_of_max']} (p50 over frames "
                f"{summary['p50_of_max']:.0f}), p99 whole-face p50 {summary['p99_whole_p50']:.1f}, "
                f"mean |d| p50 {summary['mean_abs_p50']:.3f} "
                f"(max {summary['mean_abs_max']:.3f}), pixels differing p50 "
                f"{summary['pixels_differing_p50']:.0f} of {512 * 512} "
                f"({100 * summary['fraction_differing_p50']:.1f}%), "
                f"max {summary['pixels_differing_max']}"
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.tag}.json"
    out.write_text(
        json.dumps(
            {
                "kp": args.kp.name,
                "frames": n,
                "deterministic": args.deterministic,
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "timings_ms": results,
                "face_diff": diffs,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nbench written: {out}")
    engine.stop()
    return 0


# --- stills --------------------------------------------------------------------


def cmd_diff(args: argparse.Namespace) -> int:
    frames = [int(s) for s in args.frames.split(",") if s.strip()]
    rows = []
    for kind in ("face", "composite"):
        print(f"\n{kind}: {args.a} against {args.b}")
        for n in frames:
            a = OUT_DIR / f"{args.a}-f{n}-{kind}.png"
            b = OUT_DIR / f"{args.b}-f{n}-{kind}.png"
            if not a.is_file() or not b.is_file():
                print(f"  f{n}: missing {a.name if not a.is_file() else b.name}")
                continue
            r = still_compare(a, b)
            rows.append({"kind": kind, "frame": n, **r})
            print(
                f"  f{n}: {r['pixels_differing']} px differ "
                f"({100 * r['fraction_differing']:.2f}%), "
                f"max {r['max']:.0f}, p99 of differing {r['p99_of_differing']:.1f}, "
                f"p99 whole {r['p99_whole']:.1f}, mean |d| whole {r['mean_abs_whole']:.3f}, "
                f"mean |d| differing {r['mean_abs_differing']:.2f}, box {r['box']}"
            )
    out = OUT_DIR / f"diff-{args.a}-vs-{args.b}.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\ndiff written: {out}")
    return 0


def cmd_env(_args: argparse.Namespace) -> int:
    import torch

    print(f"python {sys.version.split()[0]} {platform.platform()}")
    print(
        f"torch {torch.__version__}, cuda {torch.version.cuda}, "
        f"cudnn {torch.backends.cudnn.version()}"
    )
    print(f"gpu {torch.cuda.get_device_name(0)}, driver {_driver_version()}")
    for mod in ("tensorrt", "onnx", "onnxscript", "onnxruntime"):
        try:
            m = __import__(mod)
            print(f"{mod} {m.__version__}  {Path(m.__file__).parent}")
        except Exception as exc:
            print(f"{mod}: {type(exc).__name__}: {exc}")
    site = Path(torch.__file__).parent.parent
    for name in (
        "tensorrt",
        "tensorrt_libs",
        "tensorrt_bindings",
        "onnx",
        "onnxscript",
        "onnx_ir",
        "ml_dtypes",
    ):
        for folder in site.glob(f"{name}*"):
            if folder.is_dir():
                size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
                print(f"  {folder.name}: {size / 1e6:.1f} MB")
    return 0


# --- main -------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("env", help="Versions and sizes of what is installed.")
    p.set_defaults(fn=cmd_env)

    p = sub.add_parser("export", help="The networks to ONNX.")
    p.add_argument("--parts", choices=("fused", "warping", "spade"), default="fused")
    p.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    p.add_argument("--opset", type=int, default=20)
    p.add_argument("--exporter", choices=("auto", "dynamo", "torchscript"), default="auto")
    p.add_argument("--grid-sample-fp32", action="store_true")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("build", help="An ONNX file to a TensorRT engine.")
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fp16-flag", action="store_true", help="Weakly typed + the FP16 flag.")
    p.add_argument("--fp16-flag-strong", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--workspace-gb", type=int, default=4)
    p.add_argument("--builder-level", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("check", help="An engine against torch on the avatar's features.")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--avatar", default="my_avatar_neural")
    p.add_argument("--deterministic", action="store_true")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("clip", help="The clip through the application's pipeline.")
    p.add_argument("--clip", type=Path, default=REPO_ROOT / "recordings" / "driving.mkv")
    p.add_argument("--avatar", default="my_avatar_neural")
    p.add_argument("--first", type=int, default=300)
    p.add_argument("--last", type=int, default=900)
    p.add_argument("--nets", choices=("torch", "trt"), default="torch")
    p.add_argument("--plan", type=Path, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--stills", default="")
    p.add_argument("--tag", required=True, help="Names the JSON, the keypoints and the stills.")
    p.add_argument("--compare", type=Path, default=None, help="A frame_budget/clip JSON.")
    p.add_argument("--mode", choices=("once", "threads"), default="once")
    p.add_argument("--pace", choices=("asap", "lockstep", "realtime"), default="asap")
    p.set_defaults(fn=cmd_clip)

    p = sub.add_parser("hashes", help="Two clip JSONs' hashes, frame by frame.")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.set_defaults(fn=cmd_hashes)

    p = sub.add_parser("kp", help="Two runs' recorded keypoints, to the byte.")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.set_defaults(fn=cmd_kp)

    p = sub.add_parser("bench", help="The two networks alone on recorded keypoints.")
    p.add_argument("--kp", type=Path, required=True)
    p.add_argument("--plan", type=Path, action="append", default=[])
    p.add_argument("--avatar", default="my_avatar_neural")
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--tag", default="bench")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("diff", help="Two runs' stills through still_diff.")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--frames", required=True)
    p.set_defaults(fn=cmd_diff)

    args = parser.parse_args()
    if args.command == "clip" and args.nets == "trt" and args.plan is None:
        parser.error("--nets trt needs --plan")
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
