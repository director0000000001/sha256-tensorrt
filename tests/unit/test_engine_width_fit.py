"""The width fit (day 22, 2026-10-15): the person's rim width over the
portrait's, read on still frontal frames, scales the render and the
portrait's oval horizontally in the paste alone -- the drive untouched,
``face_width_fit`` 0 the old picture to the byte."""

from __future__ import annotations

import numpy as np
import pytest

from sha256.avatar.engine_liveportrait import (
    _FACE_WIDTH_MAX,
    _FACE_WIDTH_MIN,
    _OVAL_FOREHEAD_REACH,
    _WIDTH_DEADBAND,
    _WIDTH_MIN_SAMPLES,
    CROP_SIZE,
    _driver_face_width,
    _face_width_fit,
    _oval_mask,
    _rim_centre_x,
    _rim_width,
    _squeeze_face,
    _squeeze_points,
    _width_scale,
)
from tests.unit.test_engine_hairline import _RenderFixture, synthetic_face


class TestOptions:
    def test_defaults_and_bounds(self):
        assert _face_width_fit({}) == 1.0
        assert _face_width_fit({"face_width_fit": 0.0}) == 0.0
        with pytest.raises(ValueError, match="face_width_fit"):
            _face_width_fit({"face_width_fit": 1.5})
        assert _driver_face_width({}) is None
        assert _driver_face_width({"driver_face_width": 0.96}) == pytest.approx(0.96)
        with pytest.raises(ValueError, match="driver_face_width"):
            _driver_face_width({"driver_face_width": 3.0})


class TestTheScale:
    def test_unread_or_off_is_one(self):
        assert _width_scale(None, 1.0) == 1.0
        assert _width_scale(0.9, 0.0) == 1.0
        assert _width_scale(float("nan"), 1.0) == 1.0

    def test_the_ratio_whole_and_half(self):
        assert _width_scale(0.96, 1.0) == pytest.approx(0.96)
        assert _width_scale(0.96, 0.5) == pytest.approx(0.98)
        assert _width_scale(1.06, 1.0) == pytest.approx(1.06)

    def test_bounded_moderate(self):
        assert _width_scale(0.5, 1.0) == _FACE_WIDTH_MIN
        assert _width_scale(1.5, 1.0) == _FACE_WIDTH_MAX

    def test_a_difference_inside_the_dead_band_is_not_applied(self):
        assert _width_scale(1.0 + _WIDTH_DEADBAND / 2.0, 1.0) == 1.0
        assert _width_scale(1.0 - _WIDTH_DEADBAND / 2.0, 1.0) == 1.0


class TestTheGeometry:
    def test_the_rim_width_is_the_faces_own_horizontal_extent(self):
        pts = synthetic_face((256.0, 300.0), 233.0)[:468, :2]
        wide = pts.copy()
        wide[:, 0] = 256.0 + (wide[:, 0] - 256.0) * 1.1
        assert _rim_width(wide) == pytest.approx(_rim_width(pts) * 1.1, rel=1e-4)

    def test_the_points_are_scaled_about_the_centre_column(self):
        pts = synthetic_face((256.0, 300.0), 233.0)[:468, :2]
        centre = _rim_centre_x(pts)
        out = _squeeze_points(pts, centre, 0.9)
        np.testing.assert_allclose(out[:, 1], pts[:, 1])
        np.testing.assert_allclose(out[:, 0] - centre, (pts[:, 0] - centre) * 0.9, atol=1e-4)
        assert _rim_centre_x(out) == pytest.approx(centre, abs=1e-3)
        assert _rim_width(out) == pytest.approx(_rim_width(pts) * 0.9, rel=1e-4)


def _striped_columns(period: int = 8) -> np.ndarray:
    cols = (np.arange(CROP_SIZE) // period) % 2
    stripes = np.broadcast_to(cols[None, :], (CROP_SIZE, CROP_SIZE))
    face = np.where(stripes, 220, 60).astype(np.uint8)
    return np.repeat(face[:, :, None], 3, axis=2)


class TestTheSqueeze:
    def test_the_cpu_squeeze_narrows_the_picture_about_the_centre(self):
        face = _striped_columns()
        out = _squeeze_face(face, 256.0, 0.8)
        assert out.shape == face.shape
        # the centre column reads the same; a column 100 px out reads what
        # was 125 px out; the far sides are empty (nothing drawn there)
        row = 256
        assert int(out[row, 256, 0]) == int(face[row, 256, 0])
        assert int(out[row, 256 + 80, 0]) == int(face[row, 256 + 100, 0])
        assert out[row, 505:, 0].max() == 0

    def test_the_gpu_squeeze_is_the_cpu_one_within_a_grey_level(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = _RenderFixture()._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        loaded.width_scale = 0.9
        loaded.width_centre = 250.0
        face = _striped_columns(period=16)
        cpu = _squeeze_face(face, 250.0, 0.9)
        dev = torch.from_numpy(face).permute(2, 0, 1).float()
        gpu = engine._squeeze_face_gpu(dev, loaded).round_().permute(1, 2, 0).numpy()
        diff = np.abs(gpu.astype(np.int16) - cpu.astype(np.int16))
        # the two samplers agree everywhere but on the stripes' very
        # edges, where a half-pixel of phase is a stripe's whole contrast
        assert np.percentile(diff, 99) <= 1
        assert (diff > 1).mean() < 0.02


class TestTheSettle:
    def _loaded(self, tmp_path, torch, options):
        engine, loaded = _RenderFixture()._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        crop = synthetic_face((256.0, 300.0), 233.0)[:468, :2]
        loaded.crop_points = crop
        loaded.options = {"mask_forehead_reach": "hairline", **options}
        loaded.hairline_settled = False
        loaded.hairline_portrait = 1.5
        loaded.hairline_driver = 1.2
        loaded.mask = _oval_mask(crop, forehead_reach=_OVAL_FOREHEAD_REACH)
        loaded.chin_drop_max = 40.0
        loaded.chin_reach = 0.0
        return engine, loaded

    @staticmethod
    def _mask_width(mask: np.ndarray, row: int) -> int:
        return int((mask[row, :, 0] > 0.5).sum())

    def test_the_still_frames_median_squeezes_the_oval(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._loaded(tmp_path, torch, {})
        loaded.width_samples = [0.95, 0.96, 0.94, 0.95, 0.97, 0.95, 0.93, 0.95, 1.2]
        plain_engine, plain = self._loaded(tmp_path, torch, {"face_width_fit": 0.0})
        plain_engine._settle_forehead(plain)

        engine._settle_forehead(loaded)

        assert loaded.width_scale == pytest.approx(0.95)
        assert loaded.mask_points is not None and loaded.width_samples == []
        assert loaded.mask is not None and plain.mask is not None
        row = 300
        assert self._mask_width(loaded.mask, row) == pytest.approx(
            self._mask_width(plain.mask, row) * 0.95, abs=3
        )
        # the fit off: the old mask, the points untouched
        assert plain.width_scale == 1.0 and plain.mask_points is None

    def test_the_profiles_ratio_beats_the_live_readings(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._loaded(tmp_path, torch, {"driver_face_width": 0.9})
        loaded.width_driver = 0.9
        loaded.width_samples = [1.0] * 20
        engine._settle_forehead(loaded)
        assert loaded.width_scale == pytest.approx(0.9)

    def test_too_few_readings_fit_nothing(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._loaded(tmp_path, torch, {})
        loaded.width_samples = [0.9] * (_WIDTH_MIN_SAMPLES - 1)
        engine._settle_forehead(loaded)
        assert loaded.width_scale == 1.0 and loaded.mask_points is None

    def test_the_fit_is_in_the_readings(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._loaded(tmp_path, torch, {})
        loaded.width_samples = [0.95] * _WIDTH_MIN_SAMPLES
        engine._settle_forehead(loaded)
        engine._loaded = {loaded.descriptor.id: loaded}
        readings = engine.frame_readings(loaded.descriptor.id)
        assert readings is not None
        assert readings["face_width_scale"] == pytest.approx(0.95)
        assert np.isnan(readings["face_width"])
