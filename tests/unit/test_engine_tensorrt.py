"""Phase 2 of "The fast version" (2026-10-09, B006): the two pixel networks
on TensorRT -- the choice of backend, the engine's naming, the two modules
that stand in for the networks, the setting; and the hairline window, the
one exact CPU saving that came with it. Nothing here needs a GPU or the
tensorrt package: the engine itself is proven on the clip by
``frame_budget.py --pixel-backend`` (hashes against the probe's run)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sha256.avatar.engine_liveportrait import (
    LivePortraitEngine,
    _hairline_fraction,
    _hairline_window,
    _luminance,
)
from sha256.avatar.liveportrait_tensorrt import (
    ENGINE_STEM,
    PIXEL_BACKENDS,
    engine_name,
    pixel_nets,
    resolve_pixel_backend,
)
from sha256.config.schema import Settings
from tests.unit.test_portrait_preparation import picture_with_face


class TestEngineName:
    def test_names_the_gpu_family_and_the_tensorrt_version(self):
        assert engine_name((12, 0), "11.3.0.99") == f"{ENGINE_STEM}-sm120-trt11.3.0.99.plan"
        assert engine_name((8, 9), "11.4.0.106") == f"{ENGINE_STEM}-sm89-trt11.4.0.106.plan"


class TestResolvePixelBackend:
    def test_torch_never_looks(self):
        backend, _ = resolve_pixel_backend("torch", version="11.3.0.99", plan=Path("x.plan"))
        assert backend == "torch"

    def test_auto_without_the_package_is_torch_and_says_so(self):
        backend, reason = resolve_pixel_backend("auto", version=None, plan=None)
        assert backend == "torch"
        assert "not installed" in reason

    def test_auto_without_an_engine_is_torch_and_names_the_build(self, tmp_path):
        plan = tmp_path / "missing.plan"
        backend, reason = resolve_pixel_backend("auto", version="11.3.0.99", plan=plan)
        assert backend == "torch"
        assert "build_tensorrt_engine" in reason and "11.3.0.99" in reason

    def test_auto_with_an_engine_is_tensorrt(self, tmp_path):
        plan = tmp_path / "there.plan"
        plan.write_bytes(b"\0")
        backend, reason = resolve_pixel_backend("auto", version="11.3.0.99", plan=plan)
        assert backend == "tensorrt"
        assert "there.plan" in reason

    def test_tensorrt_insists(self, tmp_path):
        with pytest.raises(RuntimeError, match="not installed"):
            resolve_pixel_backend("tensorrt", version=None, plan=None)
        with pytest.raises(RuntimeError, match="no engine"):
            resolve_pixel_backend("tensorrt", version="11.3.0.99", plan=tmp_path / "no.plan")

    def test_an_unknown_setting_is_refused(self):
        with pytest.raises(ValueError, match="pixel_backend"):
            resolve_pixel_backend("fast", version=None, plan=None)


class TestPixelNets:
    """The two ``_nets`` entries ``_drive`` calls, backed by a fake engine."""

    def test_the_warping_entry_runs_the_engine_and_the_decoder_passes_through(self):
        torch = pytest.importorskip("torch")
        calls: list[dict[str, object]] = []
        picture = torch.full((1, 3, 512, 512), 0.5, dtype=torch.float16)

        class FakeEngine:
            def run(self, **tensors):
                calls.append(tensors)
                return {"image": picture}

        warping, decoder = pixel_nets(FakeEngine())
        f_s = torch.zeros(1, 32, 16, 64, 64, dtype=torch.float16)
        kp_s = torch.zeros(1, 21, 3, dtype=torch.float16)
        kp_d = torch.ones(1, 21, 3, dtype=torch.float16)
        out = warping(f_s, kp_driving=kp_d, kp_source=kp_s)
        assert calls == [{"f_s": f_s, "kp_source": kp_s, "kp_driving": kp_d}]
        assert torch.equal(out["out"], picture)
        # Cloned: each avatar's graph owns its picture, not the engine's buffer.
        assert out["out"].data_ptr() != picture.data_ptr()
        assert out["occlusion_map"] is None and out["deformation"] is None
        assert decoder(feature=out["out"]) is out["out"]


class TestEngineConstruction:
    def test_scripts_and_tests_get_torch_unless_they_ask(self):
        engine = LivePortraitEngine()
        assert engine.pixel_backend == "torch"
        assert engine._pixel_backend_wanted == "torch"

    def test_the_setting_is_validated_at_construction(self):
        with pytest.raises(ValueError, match="pixel_backend"):
            LivePortraitEngine(pixel_backend="fast")
        for wanted in PIXEL_BACKENDS:
            assert LivePortraitEngine(pixel_backend=wanted)._pixel_backend_wanted == wanted


class TestSetting:
    def test_auto_is_the_default_and_the_values_are_the_three(self):
        assert Settings().engine.pixel_backend == "auto"
        assert Settings(engine={"pixel_backend": "tensorrt"}).engine.pixel_backend == "tensorrt"
        with pytest.raises(ValueError):
            Settings(engine={"pixel_backend": "fast"})


class TestMedianOfFew:
    def test_equals_numpy_median_for_one_to_three_values(self):
        import itertools
        import math

        from sha256.avatar.engine_liveportrait import _median_of_few

        rng = np.random.default_rng(7)
        for n in (1, 2, 3):
            for _ in range(200):
                values = [float(v) for v in rng.uniform(0.0, 255.0, size=n)]
                assert _median_of_few(values) == float(np.median(values))
        for ordering in itertools.permutations([3.0, 1.0, 2.0]):
            assert _median_of_few(list(ordering)) == 2.0
        assert _median_of_few([0.1, 0.2]) == float(np.median([0.1, 0.2]))
        assert math.isclose(_median_of_few([0.1, 0.2]), 0.15)


class TestHairlineWindow:
    """The hairline read from the window it touches equals the read from the
    whole frame, to the bit -- the saving is exact (2026-10-09)."""

    @staticmethod
    def _both(
        image: np.ndarray, pts: np.ndarray
    ) -> tuple[float | None, float | None, tuple[int, int, int, int] | None]:
        whole = _hairline_fraction(_luminance(image), pts)
        window = _hairline_window(pts, image.shape[:2])
        if window is None:
            return whole, None, None
        x0, y0, x1, y1 = window
        windowed = _hairline_fraction(
            _luminance(image[y0:y1, x0:x1]),
            pts,
            origin=(x0, y0),
            bounds=(image.shape[0], image.shape[1]),
        )
        return whole, windowed, window

    def test_a_face_in_the_frame_reads_the_same_from_a_small_window(self):
        image, pts = picture_with_face((1280, 720), (640.0, 420.0), 300.0)
        whole, windowed, window = self._both(image, pts)
        assert whole is not None and windowed == whole
        assert window is not None
        x0, y0, x1, y1 = window
        assert (x1 - x0) * (y1 - y0) < 0.1 * 1280 * 720
        assert 0 <= x0 < x1 <= 1280 and 0 <= y0 < y1 <= 720

    def test_patches_clamped_at_the_frames_top_read_the_same(self):
        # The brows near the top: the search runs off the picture, patches
        # are cut at row 0 or fall off entirely, as they would on the whole.
        image, pts = picture_with_face((1280, 720), (640.0, 170.0), 300.0)
        whole, windowed, window = self._both(image, pts)
        assert windowed == whole
        if window is not None:
            assert window[1] == 0

    def test_a_face_at_the_left_edge_reads_the_same(self):
        image, pts = picture_with_face((1280, 720), (40.0, 400.0), 300.0)
        whole, windowed, _ = self._both(image, pts)
        assert windowed == whole

    def test_every_patch_off_the_picture_is_no_window_and_no_reading(self):
        image, pts = picture_with_face((1280, 720), (640.0, 420.0), 300.0)
        far = pts.copy()
        far[:, 1] -= 2000.0  # the whole face above the frame
        assert _hairline_window(far, image.shape[:2]) is None
        assert _hairline_fraction(_luminance(image), far) is None
