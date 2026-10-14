"""The forehead to the hairline (day 16, C1): the hairline read off a
picture, the reach for a pair from the two hairlines, the oval lifted past
the mesh's top, and the anchor settling the reach on the CPU fixture."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sha256.avatar.engine_liveportrait import (
    _FOREHEAD_REACH_MAX,
    _FOREHEAD_REACH_MIN,
    _HAIR_ROOT_MARGIN,
    _HAIRLINE_CEILING,
    _HAIRLINE_CLEARANCE,
    _HAIRLINE_MIN_SAMPLES,
    _OVAL_FOREHEAD_REACH,
    CROP_SIZE,
    MODEL_SIZE,
    LivePortraitEngine,
    _driver_hairline,
    _forehead_geometry,
    _forehead_reach,
    _forehead_reach_is_hairline,
    _frame_mask,
    _hairline_clearance,
    _hairline_fraction,
    _hairline_frame_ok,
    _hairline_reach,
    _luminance,
    _oval_mask,
    _Portrait,
)
from sha256.core.types import HeadPose
from tests.unit.test_engine_liveportrait import TestAnchorGatingInRender as _RenderFixture
from tests.unit.test_engine_liveportrait import descriptor
from tests.unit.test_portrait_preparation import picture_with_face, synthetic_face


class TestOption:
    def test_numbers_past_the_mesh_top_and_the_hairline_mode(self):
        assert _forehead_reach({}) == _OVAL_FOREHEAD_REACH
        assert _forehead_reach({"mask_forehead_reach": 1.4}) == 1.4
        assert _forehead_reach({"mask_forehead_reach": "hairline"}) == _OVAL_FOREHEAD_REACH
        assert _forehead_reach_is_hairline({"mask_forehead_reach": "hairline"})
        assert not _forehead_reach_is_hairline({"mask_forehead_reach": 0.85})
        for bad in (
            {"mask_forehead_reach": 0.1},
            {"mask_forehead_reach": 3.5},
            {"mask_forehead_reach": "up"},
        ):
            with pytest.raises(ValueError, match="mask_forehead_reach"):
                _forehead_reach(bad)
        assert _forehead_reach({"mask_forehead_reach": 2.0}) == 2.0  # day 17: up to 3.0

    def test_a_frame_is_read_only_near_frontal(self):
        assert _hairline_frame_ok(HeadPose(pitch=3.0, yaw=-8.0))
        assert _hairline_frame_ok(HeadPose(pitch=-15.0))  # day 17: how he sits
        assert not _hairline_frame_ok(HeadPose(pitch=25.0))
        assert not _hairline_frame_ok(HeadPose(yaw=30.0))
        assert not _hairline_frame_ok(None)

    def test_the_clearance_and_the_drivers_hairline_options(self):
        assert _hairline_clearance({}) == _HAIRLINE_CLEARANCE
        assert _hairline_clearance({"hairline_clearance": 0.0}) == 0.0
        with pytest.raises(ValueError, match="hairline_clearance"):
            _hairline_clearance({"hairline_clearance": 0.5})
        assert _driver_hairline({}) is None
        assert _driver_hairline({"driver_hairline": 1.2}) == 1.2
        with pytest.raises(ValueError, match="driver_hairline"):
            _driver_hairline({"driver_hairline": 0.1})


class TestHairline:
    def test_the_hairline_is_read_where_the_hair_starts(self):
        image, pts = picture_with_face((1024, 1024), (512.0, 560.0), 400.0)
        fraction = _hairline_fraction(_luminance(image), pts)
        assert fraction is not None and 0.5 < fraction < 0.7
        brow_top, forehead_h = _forehead_geometry(pts)
        assert forehead_h == pytest.approx(0.2 * 400.0, rel=0.05)
        assert brow_top[1] == pytest.approx(560.0 - 0.3 * 400.0, abs=2.0)

    def test_no_hair_within_the_search_is_none(self):
        import cv2

        image, pts = picture_with_face((1024, 1024), (512.0, 560.0), 400.0)
        cv2.rectangle(image, (0, 0), (1024, 420), (200, 200, 200), -1)
        assert _hairline_fraction(_luminance(image), pts) is None

    def test_the_pairs_reach_is_the_lower_hairline_less_the_clearance(self):
        assert _hairline_reach(1.5, 0.9) == pytest.approx(0.9 - _HAIRLINE_CLEARANCE)
        assert _hairline_reach(0.6, None) == pytest.approx(0.6 - _HAIRLINE_CLEARANCE)
        assert _hairline_reach(None, None) == _HAIRLINE_CEILING  # nothing read: not the bound
        assert _hairline_reach(3.0, 3.0) == pytest.approx(3.0 - _HAIRLINE_CLEARANCE)
        assert _hairline_reach(5.0, 5.0) == _FOREHEAD_REACH_MAX
        assert _hairline_reach(0.2, 0.2) == _FOREHEAD_REACH_MIN
        assert _hairline_reach(1.5, 1.2, clearance=0.1) == pytest.approx(1.1)

    def test_the_search_reaches_a_hairline_two_foreheads_up(self):
        """Day 17, his first look: his live frames read 1.78-1.84 of the
        forehead (head down), the woman's portrait 2.06 -- all "no hair"
        under the old 1.8 search."""
        import cv2

        image, pts = picture_with_face((1024, 1024), (512.0, 560.0), 400.0)
        brow_top, forehead_h = _forehead_geometry(pts)
        # Paint the skin band up to 2.2 foreheads, hair above it.
        skin_top = int(brow_top[1] - 2.2 * forehead_h)
        cv2.rectangle(image, (0, skin_top), (1024, int(brow_top[1])), (200, 200, 200), -1)
        cv2.rectangle(image, (0, 0), (1024, skin_top), (30, 30, 30), -1)
        fraction = _hairline_fraction(_luminance(image), pts)
        assert fraction is not None and 2.1 < fraction < 2.35


class TestOvalPastTheTop:
    def test_reach_past_one_lifts_the_forehead_arc_and_below_it_is_unchanged(self):
        crop = synthetic_face((256.0, 300.0), 233.0)[:468, :2]
        at_one = _oval_mask(crop, forehead_reach=1.0)
        default = _oval_mask(crop, forehead_reach=_OVAL_FOREHEAD_REACH)
        lifted = _oval_mask(crop, forehead_reach=1.4)

        def top_row(mask: np.ndarray) -> int:
            rows = np.where((mask[:, :, 0] > 0.5).any(axis=1))[0]
            return int(rows.min())

        assert top_row(default) > top_row(at_one) > top_row(lifted)
        _, forehead_h = _forehead_geometry(crop)
        # The lift is 0.4 of a forehead, within the feather.
        assert top_row(at_one) - top_row(lifted) == pytest.approx(0.4 * forehead_h, abs=6)
        # From the nose down nothing moved (the feather reaches ~12 px past the brows).
        nose_row = int(crop[1, 1])
        assert np.array_equal(at_one[nose_row:], lifted[nose_row:])
        assert 0 <= top_row(lifted) < CROP_SIZE


class TestSettleAtTheAnchor:
    def test_the_anchor_settles_the_reach_from_both_hairlines(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = _RenderFixture()._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        crop = synthetic_face((256.0, 300.0), 233.0)[:468, :2]
        loaded.crop_points = crop
        loaded.options = {"mask_forehead_reach": "hairline"}
        loaded.hairline_settled = False
        loaded.hairline_portrait = 1.5
        loaded.mask = _oval_mask(crop, forehead_reach=_OVAL_FOREHEAD_REACH)
        loaded.chin_drop_max = 40.0
        loaded.chin_reach = 0.0
        loaded.hairline_samples = [0.70, 0.75, None, 0.72, 0.73, 0.71] * 3  # 18, 3 unread
        before = loaded.mask.copy()

        engine._settle_forehead(loaded)

        assert loaded.hairline_settled
        # Day 21 (_forehead_plan): the driver's oval to their hairline less
        # the clearance; the portrait's to its own clean skin -- its
        # hairline less the roots' margin -- with no stretch, since the
        # driver's hairline is the lower.
        assert loaded.driver_reach == pytest.approx(0.72 - _HAIRLINE_CLEARANCE, abs=0.01)
        assert loaded.forehead_reach == pytest.approx(1.5 - _HAIR_ROOT_MARGIN, abs=0.01)
        assert loaded.forehead_stretch == 1.0 and loaded.stretch_row is None
        assert not np.array_equal(before, loaded.mask)
        assert loaded.mask_bbox is not None and loaded.hairline_samples == []

    def test_the_profiles_hairline_settles_at_once_and_the_clearance_is_the_options(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = _RenderFixture()._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        crop = synthetic_face((256.0, 300.0), 233.0)[:468, :2]
        loaded.crop_points = crop
        loaded.hairline_settled = False
        loaded.hairline_portrait = 1.5
        loaded.hairline_driver = 1.2  # Calibrate me's word: a smaller forehead
        loaded.hairline_clearance = 0.1
        loaded.mask = _oval_mask(crop, forehead_reach=_OVAL_FOREHEAD_REACH)
        loaded.chin_drop_max = 40.0
        loaded.chin_reach = 0.0
        loaded.hairline_samples = []  # no live frames needed

        engine._settle_forehead(loaded)

        assert loaded.hairline_settled
        assert loaded.driver_reach == pytest.approx(1.2 - 0.1)
        assert loaded.forehead_reach == pytest.approx(1.5 - _HAIR_ROOT_MARGIN)
        plain = _Portrait(descriptor(tmp_path))
        assert plain.hairline_driver is None and plain.hairline_clearance == _HAIRLINE_CLEARANCE
        with_profile = descriptor(tmp_path)
        with_profile.manifest.engine_options["driver_hairline"] = 1.3
        with_profile.manifest.engine_options["hairline_clearance"] = 0.02
        got = _Portrait(with_profile)
        assert got.hairline_driver == 1.3 and got.hairline_clearance == 0.02

    def test_the_last_reading_is_in_the_frame_readings(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = _RenderFixture()._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        engine._loaded[loaded.descriptor.id] = loaded
        readings = engine.frame_readings(loaded.descriptor.id)
        assert readings is not None and math.isnan(readings["hairline"])
        loaded.hairline_last = 1.7
        assert engine.frame_readings(loaded.descriptor.id)["hairline"] == 1.7

    def test_with_the_hair_on_the_drivers_oval_reaches_as_high_as_the_portraits(self):
        crop = synthetic_face((256.0, 300.0), 233.0)[:468, :2]
        portrait = _oval_mask(crop, forehead_reach=2.0)

        def top_row(mask: np.ndarray) -> int:
            return int(np.where((mask[:, :, 0] > 0.5).any(axis=1))[0].min())

        same = _frame_mask(portrait, crop, forehead_reach=2.0)
        capped = _frame_mask(portrait, crop, forehead_reach=0.85)
        freed = _frame_mask(portrait, crop, forehead_reach=0.85, driver_forehead_reach=3.0)
        assert top_row(capped) > top_row(same)
        assert top_row(freed) == top_row(same)  # the portrait's oval is the only limit now

    def test_the_avatars_hairline_off_its_own_matte(self):
        crop = synthetic_face((256.0, 300.0), 233.0)[:468, :2]
        brow_top, forehead_h = _forehead_geometry(crop)
        hair = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        hair_row = int(brow_top[1] - 1.3 * forehead_h)
        hair[:hair_row, :] = 0.9
        got = LivePortraitEngine._matte_hairline(hair, crop)
        assert got is not None and got == pytest.approx(1.3, abs=0.05)
        assert LivePortraitEngine._matte_hairline(np.zeros((CROP_SIZE, CROP_SIZE)), crop) is None
        stacked = np.stack([hair, np.zeros_like(hair)], axis=-1)
        assert LivePortraitEngine._matte_hairline(stacked, crop) == pytest.approx(got)

    def test_too_few_samples_wait_and_a_number_never_settles(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = _RenderFixture()._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        crop = synthetic_face((256.0, 300.0), 233.0)[:468, :2]
        loaded.crop_points = crop
        loaded.hairline_settled = False
        loaded.hairline_samples = [0.9] * (_HAIRLINE_MIN_SAMPLES - 1)
        engine._settle_forehead(loaded)
        assert not loaded.hairline_settled
        # Day 17: "hairline" is the default -- a plain portrait measures; a
        # number never does.
        plain = _Portrait(descriptor(tmp_path))
        assert not plain.hairline_settled and plain.forehead_reach == _OVAL_FOREHEAD_REACH
        number = descriptor(tmp_path)
        number.manifest.engine_options["mask_forehead_reach"] = 0.85
        assert _Portrait(number).hairline_settled
        hair = descriptor(tmp_path)
        hair.manifest.engine_options["mask_forehead_reach"] = "hairline"
        assert not _Portrait(hair).hairline_settled

    def test_a_reset_measures_again(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = _RenderFixture()._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        loaded.options = {"mask_forehead_reach": "hairline"}
        loaded.hairline_settled = True
        loaded.hairline_samples = [0.9]
        engine._loaded[loaded.descriptor.id] = loaded
        engine.reset_anchor(loaded.descriptor.id)
        assert not loaded.hairline_settled and loaded.hairline_samples == []
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
        engine._render(loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=True)  # still runs
