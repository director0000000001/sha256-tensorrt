"""Stage 3 of the fast version (2026-10-09): the TensorRT engine built on
the user's machine at the first start -- the decision, the builder check,
the record of a failed build, the watched build and the progress hook.
Nothing here needs a GPU or the tensorrt package."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from sha256.avatar.engine_liveportrait import LivePortraitEngine
from sha256.avatar.liveportrait_tensorrt import (
    build_engine_watched,
    builder_resource_name,
    failed_marker,
    has_builder,
    record_failure,
    resolve_pixel_backend,
)

VERSION = "11.3.0.99"


class TestBuilderResource:
    def test_named_by_the_gpu_family_and_the_major_version(self):
        assert builder_resource_name((12, 0), VERSION) == "nvinfer_builder_resource_sm120_11.dll"
        assert builder_resource_name((8, 9), VERSION) == "nvinfer_builder_resource_sm89_11.dll"
        assert builder_resource_name((8, 6), "12.0.0.1") == "nvinfer_builder_resource_sm86_12.dll"

    def test_has_builder_looks_for_the_family_in_the_libs_folder(self, tmp_path):
        assert not has_builder((8, 9), VERSION, None)
        assert not has_builder((8, 9), VERSION, tmp_path)
        (tmp_path / "nvinfer_builder_resource_sm89_11.dll").write_bytes(b"\0")
        assert has_builder((8, 9), VERSION, tmp_path)
        assert not has_builder((8, 6), VERSION, tmp_path)


def _plan(folder: Path) -> Path:
    return folder / f"liveportrait-pixels-fp16-sm89-trt{VERSION}.plan"


def _onnx(folder: Path) -> Path:
    onnx = folder / "liveportrait-pixels-fp16.onnx"
    onnx.write_bytes(b"\0")
    return onnx


class TestFirstStartDecision:
    def test_the_export_and_a_builder_without_a_plan_means_build(self, tmp_path):
        plan, onnx = _plan(tmp_path), _onnx(tmp_path)
        backend, reason = resolve_pixel_backend(
            "auto", version=VERSION, plan=plan, onnx=onnx, builder=True, failed=failed_marker(plan)
        )
        assert backend == "build"
        assert onnx.name in reason and VERSION in reason

    def test_without_the_export_it_is_torch_and_says_so(self, tmp_path):
        plan = _plan(tmp_path)
        backend, reason = resolve_pixel_backend(
            "auto",
            version=VERSION,
            plan=plan,
            onnx=tmp_path / "missing.onnx",
            builder=True,
            failed=failed_marker(plan),
        )
        assert backend == "torch"
        assert "no ONNX export" in reason

    def test_without_a_builder_for_this_family_it_is_torch_and_says_so(self, tmp_path):
        plan, onnx = _plan(tmp_path), _onnx(tmp_path)
        backend, reason = resolve_pixel_backend(
            "auto", version=VERSION, plan=plan, onnx=onnx, builder=False, failed=failed_marker(plan)
        )
        assert backend == "torch"
        assert "no builder" in reason

    def test_a_failed_record_stops_the_retry_and_names_itself(self, tmp_path):
        plan, onnx = _plan(tmp_path), _onnx(tmp_path)
        marker = record_failure(plan, RuntimeError("boom"), onnx)
        backend, reason = resolve_pixel_backend(
            "auto", version=VERSION, plan=plan, onnx=onnx, builder=True, failed=marker
        )
        assert backend == "torch"
        assert "failed" in reason and marker.name in reason

    def test_a_plan_already_there_is_used_whatever_else_is(self, tmp_path):
        plan, onnx = _plan(tmp_path), _onnx(tmp_path)
        plan.write_bytes(b"\0")
        marker = record_failure(plan, RuntimeError("an old failure"), onnx)
        backend, _ = resolve_pixel_backend(
            "auto", version=VERSION, plan=plan, onnx=onnx, builder=True, failed=marker
        )
        assert backend == "tensorrt"

    def test_tensorrt_insisted_on_builds_too_and_raises_without_a_builder(self, tmp_path):
        plan, onnx = _plan(tmp_path), _onnx(tmp_path)
        backend, _ = resolve_pixel_backend(
            "tensorrt", version=VERSION, plan=plan, onnx=onnx, builder=True
        )
        assert backend == "build"
        with pytest.raises(RuntimeError, match="no builder"):
            resolve_pixel_backend("tensorrt", version=VERSION, plan=plan, onnx=onnx, builder=False)

    def test_the_old_call_without_the_new_arguments_still_answers_torch(self, tmp_path):
        backend, reason = resolve_pixel_backend("auto", version=VERSION, plan=_plan(tmp_path))
        assert backend == "torch"
        assert "build_tensorrt_engine" in reason


class TestFailureRecord:
    def test_written_beside_the_plan_with_the_error_and_the_export(self, tmp_path):
        plan, onnx = _plan(tmp_path), _onnx(tmp_path)
        marker = record_failure(plan, RuntimeError("the builder said no"), onnx)
        assert marker == failed_marker(plan) == plan.with_name(plan.name + ".failed.json")
        record = json.loads(marker.read_text(encoding="utf-8"))
        assert record["error"] == "RuntimeError: the builder said no"
        assert record["plan"] == plan.name and record["onnx"] == onnx.name
        assert "Delete this file" in record["note"]


class TestBuildWatched:
    def test_the_seconds_reach_the_caller_on_its_own_thread_and_the_result_comes_back(
        self, tmp_path
    ):
        seen: list[float] = []
        threads: set[int] = set()
        workers: set[int] = set()

        def slow_build(onnx: Path, target: Path) -> dict[str, float]:
            workers.add(threading.get_ident())
            time.sleep(0.25)
            target.write_bytes(b"plan")
            return {"build_s": 0.25}

        def progress(seconds: float) -> None:
            seen.append(seconds)
            threads.add(threading.get_ident())

        result = build_engine_watched(
            tmp_path / "x.onnx", tmp_path / "x.plan", progress, poll_s=0.05, build=slow_build
        )

        assert result == {"build_s": 0.25}
        assert (tmp_path / "x.plan").read_bytes() == b"plan"
        assert seen and seen == sorted(seen) and seen[-1] >= 0.2
        assert threads == {threading.get_ident()}
        assert workers and workers != threads

    def test_an_error_on_the_worker_is_raised_here(self, tmp_path):
        def bad_build(onnx: Path, target: Path) -> dict[str, float]:
            raise RuntimeError("no tactics for this card")

        with pytest.raises(RuntimeError, match="no tactics"):
            build_engine_watched(
                tmp_path / "x.onnx", tmp_path / "x.plan", None, poll_s=0.01, build=bad_build
            )
        assert not (tmp_path / "x.plan").exists()


class TestProgressHook:
    def test_the_engine_tells_the_hook_and_is_silent_without_one(self):
        said: list[str] = []
        LivePortraitEngine(progress=said.append)._say("Loading the neural networks")
        assert said == ["Loading the neural networks"]
        LivePortraitEngine()._say("nothing to hear this")
