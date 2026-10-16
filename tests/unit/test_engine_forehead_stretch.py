"""The forehead (day 21, 2026-10-14): the plan from the two hairlines --
the join slightly below the person's own, the avatar's clean skin reaching
it by a moderate stretch -- and the stretch in the paste's warp, on the CPU
and on the GPU."""

from __future__ import annotations

import numpy as np
import pytest

from sha256.avatar.engine_liveportrait import (
    _FOREHEAD_REACH_MAX,
    _FOREHEAD_STRETCH_MAX,
    _HAIR_ROOT_MARGIN,
    _HAIRLINE_CEILING,
    _HAIRLINE_CLEARANCE,
    _STRETCH_ANCHOR,
    CROP_SIZE,
    LivePortraitEngine,
    _forehead_plan,
    _forehead_stretch_max,
    _hair_root_margin,
    _mask_bbox,
    _oval_mask,
    _Portrait,
    _transformed_box,
    crop_matrix,
)
from tests.unit.test_engine_liveportrait import descriptor, face_points


class TestOptions:
    def test_defaults_and_bounds(self):
        assert _forehead_stretch_max({}) == _FOREHEAD_STRETCH_MAX == 1.25
        assert _forehead_stretch_max({"forehead_stretch_max": 1.0}) == 1.0
        assert _hair_root_margin({}) == _HAIR_ROOT_MARGIN == 0.15
        assert _hair_root_margin({"hair_root_margin": 0}) == 0.0
        for bad in (0.9, 2.5):
            with pytest.raises(ValueError, match="forehead_stretch_max"):
                _forehead_stretch_max({"forehead_stretch_max": bad})
        for bad in (-0.1, 0.6):
            with pytest.raises(ValueError, match="hair_root_margin"):
                _hair_root_margin({"hair_root_margin": bad})


class TestPlan:
    """``_forehead_plan`` -- the numbers of the day's own faces."""

    def test_his_face_the_join_just_under_his_hair_with_a_small_stretch(self):
        # his hairline 1.6 of his forehead, the portrait's 1.52, foreheads alike
        portrait, driver, stretch = _forehead_plan(1.52, 1.6, 50.0, 50.0)
        assert driver == pytest.approx(1.6 - _HAIRLINE_CLEARANCE)
        # the avatar's clean skin reaches exactly the join: both ovals meet there
        assert portrait == pytest.approx(driver, abs=1e-6)
        # day 22: the clean skin (the hairline less the margin) is a source
        # row the stretch carries to the join
        wanted = (1.6 - _HAIRLINE_CLEARANCE) - _STRETCH_ANCHOR
        assert stretch == pytest.approx(wanted / (1.52 - _HAIR_ROOT_MARGIN - _STRETCH_ANCHOR))
        assert 1.0 < stretch < _FOREHEAD_STRETCH_MAX

    def test_a_taller_forehead_gets_the_cap_and_the_join_on_the_avatars_clean_skin(self):
        # the woman's: hairline 1.7 of a forehead a third taller than the portrait's
        portrait, driver, stretch = _forehead_plan(1.52, 1.7, 60.0, 80.0)
        assert stretch == _FOREHEAD_STRETCH_MAX
        assert driver == pytest.approx(1.7 - _HAIRLINE_CLEARANCE)
        assert portrait == pytest.approx(
            _STRETCH_ANCHOR + _FOREHEAD_STRETCH_MAX * (1.52 - _HAIR_ROOT_MARGIN - _STRETCH_ANCHOR)
        )
        # in crop pixels the portrait's oval stops under the driver's hairline
        assert portrait * 60.0 < driver * 80.0

    def test_a_smaller_forehead_than_the_portraits_takes_no_stretch(self):
        portrait, driver, stretch = _forehead_plan(1.52, 1.1, 50.0, 50.0)
        assert stretch == 1.0
        assert driver == pytest.approx(1.1 - _HAIRLINE_CLEARANCE)
        assert portrait == pytest.approx(1.52 - _HAIR_ROOT_MARGIN)
        assert driver < portrait  # the driver's own hairline bounds the join

    def test_an_unread_hairline_gets_the_whole_moderate_stretch_and_no_bound(self):
        portrait, driver, stretch = _forehead_plan(1.52, None, 50.0, 50.0)
        assert stretch == _FOREHEAD_STRETCH_MAX and driver == _FOREHEAD_REACH_MAX
        assert portrait == pytest.approx(
            _STRETCH_ANCHOR + _FOREHEAD_STRETCH_MAX * (1.52 - _HAIR_ROOT_MARGIN - _STRETCH_ANCHOR)
        )

    def test_an_unread_portrait_hairline_counts_as_the_ceiling(self):
        portrait, _, stretch = _forehead_plan(None, 1.3, 50.0, 50.0, stretch_max=1.0)
        assert stretch == 1.0
        assert portrait == pytest.approx(_HAIRLINE_CEILING - _HAIR_ROOT_MARGIN)

    def test_the_stretch_off_is_the_old_join_less_the_margin(self):
        portrait, driver, stretch = _forehead_plan(
            1.52, 1.6, 50.0, 50.0, stretch_max=1.0, margin=0.0
        )
        assert stretch == 1.0 and portrait == pytest.approx(1.52)
        assert driver == pytest.approx(1.6 - _HAIRLINE_CLEARANCE)

    def test_the_reaches_stay_within_the_bounds(self):
        portrait, driver, _ = _forehead_plan(3.0, 3.0, 50.0, 200.0, stretch_max=2.0)
        assert portrait <= _FOREHEAD_REACH_MAX and driver <= _FOREHEAD_REACH_MAX
        portrait, driver, _ = _forehead_plan(0.35, 0.3, 50.0, 50.0, margin=0.5)
        assert portrait >= 0.3 and driver >= 0.3


def _loaded_portrait(tmp_path, *, stretch: float, row: float | None) -> _Portrait:
    pts = face_points()
    to_crop, _ = crop_matrix(pts)
    crop_points = (pts @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
    loaded = _Portrait(descriptor(tmp_path))
    loaded.options = {"skin_match": 0.0, "edge_match": 0.0}
    loaded.crop_points = crop_points
    # the oval lifted to 1.4 foreheads, so the paste reaches the stretched rows
    loaded.mask = _oval_mask(crop_points, forehead_reach=1.4)
    loaded.mask_bbox = _mask_bbox(loaded.mask)
    loaded.forehead_stretch = stretch
    loaded.stretch_row = row
    return loaded


def _striped_face(period: int = 8) -> np.ndarray:
    """Horizontal stripes over the whole crop: a ruler for the stretch."""
    rows = (np.arange(CROP_SIZE) // period) % 2
    stripes = np.broadcast_to(rows[:, None], (CROP_SIZE, CROP_SIZE))
    face = np.where(stripes, 220, 60).astype(np.uint8)
    return np.repeat(face[:, :, None], 3, axis=2)


class TestTheStretchInThePaste:
    """``_warp_face`` through ``_paste`` on the CPU: nothing moves below the
    anchor row; above it the render is drawn taller by the stretch."""

    def test_no_stretch_is_the_affine_warp_to_the_byte(self, tmp_path):
        pts = face_points()
        _, to_frame = crop_matrix(pts)
        scene = np.full((720, 1280, 3), 90, dtype=np.uint8)
        face = _striped_face()
        plain = _loaded_portrait(tmp_path, stretch=1.0, row=100.0)
        assert plain.mask is not None
        one = LivePortraitEngine()._paste(scene, face, to_frame, plain, plain.mask)
        none = _loaded_portrait(tmp_path, stretch=1.25, row=None)
        assert none.mask is not None
        two = LivePortraitEngine()._paste(scene, face, to_frame, none, none.mask)
        np.testing.assert_array_equal(one, two)

    def test_the_rows_above_the_anchor_are_drawn_taller_and_the_rest_the_same(self, tmp_path):
        pts = face_points()
        _, to_frame = crop_matrix(pts)
        scene = np.full((720, 1280, 3), 90, dtype=np.uint8)
        face = _striped_face()
        plain = _loaded_portrait(tmp_path, stretch=1.0, row=None)
        assert plain.mask is not None and plain.crop_points is not None
        # a row well inside the oval, so the stretched part above it has
        # room for several stripes under the mask's core
        brow_row = float(plain.crop_points[:, 1].min()) + 110.0
        stretched = _loaded_portrait(tmp_path, stretch=1.25, row=brow_row)
        assert stretched.mask is not None

        one = LivePortraitEngine()._paste(scene, face, to_frame, plain, plain.mask)
        two = LivePortraitEngine()._paste(scene, face, to_frame, stretched, stretched.mask)

        # the anchor row in the frame (the affine is a similarity: rows map to rows)
        frame_row = round(float(to_frame[1, 1] * brow_row + to_frame[1, 2]))
        x0, y0, x1, y1 = _transformed_box(to_frame, plain.mask_bbox, 1280, 720)
        assert y0 < frame_row < y1
        # below the anchor: the same picture -- the remap's float sampling
        # against warpAffine's fixed-point, and the plane fitted over a box
        # whose top rows changed, are a grey level or two
        below = np.abs(
            one[frame_row + 2 :].astype(np.int16) - two[frame_row + 2 :].astype(np.int16)
        )
        assert below.max() <= 2
        # above it: the stripes' period grows by the stretch, read down one column
        column = (x0 + x1) // 2
        above_one = one[y0 + 12 : frame_row - 2, column, 0].astype(np.int16)
        above_two = two[y0 + 12 : frame_row - 2, column, 0].astype(np.int16)
        assert np.abs(above_one - above_two).max() > 100  # something moved

        def period(signal: np.ndarray) -> float:
            # the stripes' period as the mean distance between upward
            # crossings of the middle grey
            mid = (int(signal.max()) + int(signal.min())) / 2.0
            up = np.where((signal[:-1] < mid) & (signal[1:] >= mid))[0]
            assert len(up) >= 3
            return float(np.diff(up).mean())

        scale = float(to_frame[1, 1])
        assert period(above_one) == pytest.approx(16.0 * scale, rel=0.15)  # a dark and a light band
        assert period(above_two) == pytest.approx(16.0 * scale * 1.25, rel=0.15)


def _cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


@pytest.mark.skipif(not _cuda(), reason="needs CUDA")
class TestTheStretchOnTheGpu:
    def test_the_gpu_paste_stretches_as_the_cpu_one(self, tmp_path):
        import torch

        pts = face_points()
        _, to_frame = crop_matrix(pts)
        scene = np.full((720, 1280, 3), 90, dtype=np.uint8)
        face = _striped_face(12)
        loaded = _loaded_portrait(tmp_path, stretch=1.25, row=None)
        assert loaded.mask is not None and loaded.crop_points is not None
        loaded.stretch_row = float(loaded.crop_points[:, 1].min()) + 40.0

        cpu = LivePortraitEngine()._paste(scene, face, to_frame, loaded, loaded.mask)

        engine = LivePortraitEngine()
        engine._torch = torch
        engine.fit_on_gpu = True
        device = torch.device("cuda")
        face_t = torch.from_numpy(face).permute(2, 0, 1).float().to(device)
        gpu = engine._paste(scene, face_t, to_frame, loaded, loaded.mask)
        torch.cuda.synchronize()

        x0, y0, x1, y1 = _transformed_box(to_frame, loaded.mask_bbox, 1280, 720)
        box_cpu = cpu[y0:y1, x0:x1].astype(np.int16)
        box_gpu = gpu[y0:y1, x0:x1].astype(np.int16)
        # the stripes' edges land within a pixel: the two warps differ by
        # their sampling arithmetic, so most pixels agree and a few at the
        # edges are a grey level or a stripe apart
        differing = np.abs(box_cpu - box_gpu).max(axis=2) > 2
        assert differing.mean() < 0.05


class TestSettleWithTheDriversForehead:
    """The settle given the driver's crop points: their forehead's height
    against the portrait's decides the stretch."""

    def test_a_taller_driver_forehead_settles_a_stretch_and_its_row(self, tmp_path):
        torch = pytest.importorskip("torch")
        from sha256.avatar.engine_liveportrait import _forehead_geometry
        from tests.unit.test_engine_liveportrait import TestAnchorGatingInRender
        from tests.unit.test_portrait_preparation import synthetic_face

        engine, loaded = TestAnchorGatingInRender()._engine_and_loaded(
            tmp_path, torch=torch, anchor_wanted=1
        )
        crop = synthetic_face((256.0, 300.0), 233.0)[:468, :2]
        loaded.crop_points = crop
        loaded.options = {"mask_forehead_reach": "hairline"}
        loaded.hairline_settled = False
        loaded.hairline_portrait = 1.52
        loaded.hairline_driver = 1.7
        loaded.mask = _oval_mask(crop)
        loaded.chin_drop_max = 40.0
        loaded.chin_reach = 0.0
        # the driver's face: the same mesh with the forehead a third taller
        brow_top, forehead = _forehead_geometry(crop)
        driver = crop.copy()
        above = driver[:, 1] < brow_top[1]
        driver[above, 1] = brow_top[1] - (brow_top[1] - driver[above, 1]) * 1.33

        engine._settle_forehead(loaded, driver)

        assert loaded.hairline_settled
        assert loaded.forehead_stretch == pytest.approx(_FOREHEAD_STRETCH_MAX)
        assert loaded.stretch_row == pytest.approx(brow_top[1] - _STRETCH_ANCHOR * forehead)
        assert loaded.driver_reach == pytest.approx(1.7 - _HAIRLINE_CLEARANCE)
        assert loaded.forehead_reach == pytest.approx(
            _STRETCH_ANCHOR + _FOREHEAD_STRETCH_MAX * (1.52 - _HAIR_ROOT_MARGIN - _STRETCH_ANCHOR)
        )
        # a re-centre forgets the plan
        engine._loaded[loaded.descriptor.id] = loaded
        engine.reset_anchor(loaded.descriptor.id)
        assert loaded.driver_reach is None and loaded.forehead_stretch == 1.0
        assert loaded.stretch_row is None and not loaded.hairline_settled
