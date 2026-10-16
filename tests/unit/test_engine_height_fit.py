"""The height fit (day 23, 2026-10-16): the person's brow-top-to-chin over
the portrait's, read on still frontal frames, scales the render and the
portrait's points vertically about the portrait's chin row in the paste
alone -- the drive untouched, ``face_height_fit`` 0 the old picture to the
byte. And the shaved head: a portrait with no hairline read has no roots
either, so the roots reader's find there does not cap the forehead."""

from __future__ import annotations

import numpy as np
import pytest

from sha256.avatar.engine_liveportrait import (
    _CHIN,
    _FACE_HEIGHT_MAX,
    _FACE_HEIGHT_MIN,
    _OVAL_FOREHEAD_REACH,
    _WIDTH_DEADBAND,
    _WIDTH_MIN_SAMPLES,
    CROP_SIZE,
    _brow_chin,
    _chin_row,
    _driver_face_height,
    _face_height_fit,
    _forehead_geometry,
    _height_scale,
    _lift_points,
    _oval_mask,
    _squeeze_face,
)
from tests.unit.test_engine_hairline import _RenderFixture, synthetic_face


class TestOptions:
    def test_defaults_and_bounds(self):
        assert _face_height_fit({}) == 1.0
        assert _face_height_fit({"face_height_fit": 0.5}) == 0.5
        with pytest.raises(ValueError, match="face_height_fit"):
            _face_height_fit({"face_height_fit": 1.5})
        assert _driver_face_height({}) is None
        assert _driver_face_height({"driver_face_height": 1.05}) == 1.05
        with pytest.raises(ValueError, match="driver_face_height"):
            _driver_face_height({"driver_face_height": 3.0})


class TestTheScale:
    def test_unread_or_off_is_one(self):
        assert _height_scale(None, 1.0) == 1.0
        assert _height_scale(1.05, 0.0) == 1.0
        assert _height_scale(float("nan"), 1.0) == 1.0

    def test_the_ratio_whole_and_half(self):
        assert _height_scale(1.047, 1.0) == pytest.approx(1.047)
        assert _height_scale(1.04, 0.5) == pytest.approx(1.02)

    def test_bounded_moderate(self):
        assert _height_scale(1.5, 1.0) == _FACE_HEIGHT_MAX
        assert _height_scale(0.5, 1.0) == _FACE_HEIGHT_MIN

    def test_a_difference_inside_the_dead_band_is_not_applied(self):
        assert _height_scale(1.0 + _WIDTH_DEADBAND / 2.0, 1.0) == 1.0


class TestTheGeometry:
    def test_the_brow_to_chin_is_along_the_faces_own_vertical(self):
        pts = synthetic_face((256.0, 300.0), 233.0)[:468, :2]
        chin = _chin_row(pts)
        tall = _lift_points(pts, chin, 1.1)
        assert _brow_chin(tall) == pytest.approx(_brow_chin(pts) * 1.1, rel=1e-3)

    def test_the_points_are_lifted_about_the_chin_row(self):
        pts = synthetic_face((256.0, 300.0), 233.0)[:468, :2]
        chin = _chin_row(pts)
        out = _lift_points(pts, chin, 1.05)
        np.testing.assert_allclose(out[:, 0], pts[:, 0])
        np.testing.assert_allclose(out[:, 1] - chin, (pts[:, 1] - chin) * 1.05, atol=1e-4)
        assert out[_CHIN, 1] == pytest.approx(chin)
        # the brow top rises, the forehead grows with it
        brow_before, forehead_before = _forehead_geometry(pts)
        brow_after, forehead_after = _forehead_geometry(out)
        assert brow_after[1] < brow_before[1]
        assert forehead_after == pytest.approx(forehead_before * 1.05, rel=1e-3)


def _striped_rows(period: int = 8) -> np.ndarray:
    rows = (np.arange(CROP_SIZE) // period) % 2
    stripes = np.broadcast_to(rows[:, None], (CROP_SIZE, CROP_SIZE))
    face = np.where(stripes, 220, 60).astype(np.uint8)
    return np.repeat(face[:, :, None], 3, axis=2)


class TestTheLift:
    def test_the_cpu_lift_stretches_the_picture_about_the_chin_row(self):
        face = _striped_rows()
        out = _squeeze_face(face, 256.0, 1.0, 400.0, 1.25)
        assert out.shape == face.shape
        # the chin row reads the same; a row 100 px above it reads what
        # was 80 px above; the width is untouched
        col = 256
        assert int(out[400, col, 0]) == int(face[400, col, 0])
        assert int(out[400 - 100, col, 0]) == int(face[400 - 80, col, 0])
        np.testing.assert_array_equal(out[400, :, 0], face[400, :, 0])

    def test_the_gpu_lift_is_the_cpu_one_within_a_grey_level(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = _RenderFixture()._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        loaded.width_scale = 0.95
        loaded.width_centre = 250.0
        loaded.height_scale = 1.08
        loaded.height_centre = 390.0
        face = _striped_rows(period=16)
        cpu = _squeeze_face(face, 250.0, 0.95, 390.0, 1.08)
        dev = torch.from_numpy(face).permute(2, 0, 1).float()
        gpu = engine._squeeze_face_gpu(dev, loaded).round_().permute(1, 2, 0).numpy()
        diff = np.abs(gpu.astype(np.int16) - cpu.astype(np.int16))
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
    def _mask_top(mask: np.ndarray) -> int:
        rows = np.where((mask[:, :, 0] > 0.5).any(axis=1))[0]
        return int(rows.min())

    def test_the_still_frames_median_lifts_the_oval(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._loaded(tmp_path, torch, {})
        loaded.height_samples = [1.05, 1.06, 1.04, 1.05, 1.07, 1.05, 1.03, 1.05, 1.3]
        plain_engine, plain = self._loaded(tmp_path, torch, {"face_height_fit": 0.0})
        plain_engine._settle_forehead(plain)

        engine._settle_forehead(loaded)

        assert loaded.height_scale == pytest.approx(1.05)
        assert loaded.height_centre == pytest.approx(_chin_row(loaded.crop_points))
        assert loaded.mask_points is not None and loaded.height_samples == []
        assert loaded.settle_height == pytest.approx(1.05)
        # the lifted points: the chin where it was, the brow top higher
        assert loaded.mask_points[_CHIN, 1] == pytest.approx(loaded.crop_points[_CHIN, 1])
        brow_lifted = _forehead_geometry(loaded.mask_points)[0][1]
        brow_plain = _forehead_geometry(loaded.crop_points)[0][1]
        assert brow_lifted < brow_plain
        # the forehead's plan read the lifted brow: the oval's top is higher
        # than with the fit off
        assert loaded.mask is not None and plain.mask is not None
        assert self._mask_top(loaded.mask) < self._mask_top(plain.mask)
        # the fit off: the old mask, the points untouched
        assert plain.height_scale == 1.0 and plain.mask_points is None

    def test_the_profiles_ratio_beats_the_live_readings(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._loaded(tmp_path, torch, {"driver_face_height": 1.04})
        loaded.height_driver = 1.04
        loaded.height_samples = [1.0] * 20
        engine._settle_forehead(loaded)
        assert loaded.height_scale == pytest.approx(1.04)

    def test_too_few_readings_fit_nothing(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._loaded(tmp_path, torch, {})
        loaded.height_samples = [1.05] * (_WIDTH_MIN_SAMPLES - 1)
        engine._settle_forehead(loaded)
        assert loaded.height_scale == 1.0 and loaded.mask_points is None

    def test_the_width_and_the_height_compose(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._loaded(tmp_path, torch, {})
        loaded.width_samples = [0.95] * _WIDTH_MIN_SAMPLES
        loaded.height_samples = [1.05] * _WIDTH_MIN_SAMPLES
        engine._settle_forehead(loaded)
        assert loaded.width_scale == pytest.approx(0.95)
        assert loaded.height_scale == pytest.approx(1.05)
        assert loaded.mask_points is not None
        crop = loaded.crop_points
        np.testing.assert_allclose(
            loaded.mask_points[:, 0] - loaded.width_centre,
            (crop[:, 0] - loaded.width_centre) * 0.95,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            loaded.mask_points[:, 1] - loaded.height_centre,
            (crop[:, 1] - loaded.height_centre) * 1.05,
            atol=1e-3,
        )

    def test_the_fit_is_in_the_readings(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._loaded(tmp_path, torch, {})
        loaded.height_samples = [1.05] * _WIDTH_MIN_SAMPLES
        engine._settle_forehead(loaded)
        engine._loaded = {loaded.descriptor.id: loaded}
        readings = engine.frame_readings(loaded.descriptor.id)
        assert readings is not None
        assert readings["face_height_scale"] == pytest.approx(1.05)
        assert np.isnan(readings["face_height"])

    def test_a_shaved_portrait_ignores_what_the_roots_reader_met(self, tmp_path):
        """Day 23: his first shaved avatar read "roots" at 0.90 of the
        forehead with no hairline at all -- the plan must take the whole
        clean forehead, not stop at the scalp's shading."""
        torch = pytest.importorskip("torch")
        engine, loaded = self._loaded(tmp_path, torch, {})
        loaded.hairline_portrait = None
        loaded.roots_portrait = 0.9
        loaded.hairline_driver = 1.7
        engine._settle_forehead(loaded)
        assert loaded.forehead_reach > 1.2
        assert loaded.forehead_stretch == pytest.approx(1.0, abs=0.3)

        haired_engine, haired = self._loaded(tmp_path, torch, {})
        haired.hairline_portrait = 1.5
        haired.roots_portrait = 0.9
        haired.hairline_driver = 1.7
        haired_engine._settle_forehead(haired)
        # with a hairline read, the roots still cap the clean skin (day 22)
        assert haired.forehead_reach < loaded.forehead_reach
