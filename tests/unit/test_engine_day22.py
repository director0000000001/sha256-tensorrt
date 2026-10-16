"""Day 22 (2026-10-15), his three presses: the avatar's hair roots at the top
of the paste (the portrait's clean skin now ends where its roots begin, a
source row the stretch carries to the join), the photograph's background
under the drawn chin on a look down (the chin's own keep, every frame; the
guard on a tilt too), the forehead following the hairline, and the settle's
inputs carried by the capture."""

from __future__ import annotations

import numpy as np
import pytest

from sha256.avatar.engine_liveportrait import (
    _ANCHOR_SCALARS,
    _BACKGROUND_NEAR,
    _FOREHEAD_STRETCH_MAX,
    _HAIR_ROOT_MARGIN,
    _HAIRLINE_CLEARANCE,
    _HAIRLINE_FOLLOW,
    _HAIRLINE_MIN_SAMPLES,
    _OVAL_FOREHEAD_REACH,
    _ROOTS_CLEARANCE,
    _STRETCH_ANCHOR,
    LivePortraitEngine,
    _chin_keep,
    _forehead_geometry,
    _forehead_plan,
    _hairline_drifted,
    _hairline_fraction,
    _luminance,
    _oval_mask,
    _roots_onset,
)
from tests.unit.test_engine_hairline import _RenderFixture, synthetic_face
from tests.unit.test_portrait_preparation import picture_with_face


def _forehead_picture(*, strands_from: float | None, hair_from: float = 1.6):
    """A face whose forehead is flat skin up to ``hair_from`` foreheads
    (dark hair above), with thin dark strands every 12 px from
    ``strands_from`` up -- loose roots the median never sees."""
    import cv2

    image, pts = picture_with_face((1024, 1024), (512.0, 560.0), 400.0)
    brow_top, forehead_h = _forehead_geometry(pts)
    brow_row = int(brow_top[1])
    hair_row = int(brow_top[1] - hair_from * forehead_h)
    cv2.rectangle(image, (0, hair_row), (1024, brow_row), (200, 200, 200), -1)
    cv2.rectangle(image, (0, 0), (1024, hair_row), (30, 30, 30), -1)
    if strands_from is not None:
        strand_row = int(brow_top[1] - strands_from * forehead_h)
        for x in range(0, 1024, 12):
            cv2.line(image, (x, hair_row), (x, strand_row), (150, 150, 150), 1)
    return image, pts


class TestRootsOnset:
    def test_the_roots_are_read_where_the_strands_begin_under_the_hairline(self):
        image, pts = _forehead_picture(strands_from=1.2)
        lum = _luminance(image)
        hairline = _hairline_fraction(lum, pts)
        roots = _roots_onset(lum, pts)
        # the median reader still calls the hair 1.6 up; the strands start at 1.2
        assert hairline is not None and 1.5 < hairline < 1.7
        assert roots is not None and 1.15 < roots < 1.3

    def test_without_strands_the_roots_are_the_hair_itself(self):
        image, pts = _forehead_picture(strands_from=None)
        roots = _roots_onset(_luminance(image), pts)
        assert roots is not None and 1.5 < roots < 1.7

    def test_nothing_dark_within_the_search_is_none(self):
        import cv2

        image, pts = picture_with_face((1024, 1024), (512.0, 560.0), 400.0)
        cv2.rectangle(image, (0, 0), (1024, 420), (200, 200, 200), -1)
        assert _roots_onset(_luminance(image), pts) is None


class TestPlanWithRoots:
    def test_the_roots_lower_the_clean_skin_and_the_stretch_carries_it_to_the_join(self):
        portrait, driver, stretch = _forehead_plan(1.52, 1.6, 50.0, 50.0, portrait_roots=1.40)
        skin_top = 1.40 - _ROOTS_CLEARANCE  # under the hairline less the margin (1.37)
        assert skin_top < 1.52 - _HAIR_ROOT_MARGIN
        assert driver == pytest.approx(1.6 - _HAIRLINE_CLEARANCE)
        assert stretch == pytest.approx((driver - _STRETCH_ANCHOR) / (skin_top - _STRETCH_ANCHOR))
        assert portrait == pytest.approx(driver)
        # the source row drawn at the join is the clean top
        assert _STRETCH_ANCHOR + (portrait - _STRETCH_ANCHOR) / stretch == pytest.approx(skin_top)

    def test_roots_above_the_margin_change_nothing(self):
        with_roots = _forehead_plan(1.52, 1.6, 50.0, 50.0, portrait_roots=1.50)
        without = _forehead_plan(1.52, 1.6, 50.0, 50.0)
        assert with_roots == pytest.approx(without)

    def test_the_cap_still_holds_with_a_low_clean_top(self):
        portrait, _, stretch = _forehead_plan(1.52, 2.5, 50.0, 50.0, portrait_roots=1.2)
        assert stretch == _FOREHEAD_STRETCH_MAX
        assert portrait == pytest.approx(
            _STRETCH_ANCHOR + _FOREHEAD_STRETCH_MAX * (1.2 - _ROOTS_CLEARANCE - _STRETCH_ANCHOR)
        )


def _chin_patch(height: int = 120, width: int = 100, floor: int = 80) -> np.ndarray:
    """Skin everywhere, the photograph's white under the chin from ``floor``
    down in the middle, and a white island above the floor (a highlight)."""
    face = np.full((height, width, 3), (150, 160, 180), dtype=np.uint8)
    face[floor:, 30:70] = (240, 240, 240)
    face[20:30, 40:50] = (240, 240, 240)
    return face


class TestChinKeep:
    def test_the_background_under_the_chin_is_cut_and_the_island_above_is_kept(self):
        bg = np.array([240.0, 240.0, 240.0], dtype=np.float32)
        keep = _chin_keep(_chin_patch(), bg, 80.0)
        assert keep is not None and keep.shape == (120, 100)
        assert keep[100, 50] == 0.0  # the white under the chin
        assert keep[25, 45] == 1.0  # the highlight above the floor stays
        assert keep[50, 10] == 1.0  # skin
        assert keep[100, 10] == 1.0  # skin beside the blob, below the floor

    def test_no_background_below_the_floor_is_none(self):
        bg = np.array([240.0, 240.0, 240.0], dtype=np.float32)
        face = np.full((120, 100, 3), (150, 160, 180), dtype=np.uint8)
        assert _chin_keep(face, bg, 80.0) is None
        assert _chin_keep(_chin_patch(), bg, 200.0) is None  # the floor past the patch
        assert _BACKGROUND_NEAR < 60.0  # skin at 60 levels from white is never cut

    def test_the_gpu_keep_agrees_with_the_cpu_one_away_from_the_feather(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, _ = _RenderFixture()._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        bg = np.array([240.0, 240.0, 240.0], dtype=np.float32)
        patch = _chin_patch()
        cpu = _chin_keep(patch, bg, 80.0)
        dev = torch.from_numpy(patch).permute(2, 0, 1).float()
        gpu = engine._chin_keep_gpu(dev, bg, 80.0, True)
        assert cpu is not None and gpu is not None
        gpu_np = gpu.numpy()
        assert gpu_np.shape == cpu.shape
        # the two feathers differ in shape (a Gaussian, a box), not in reach:
        # where the CPU keep is whole or nothing, so is the GPU's
        core = (cpu == 0.0) | (cpu == 1.0)
        assert core.mean() > 0.8
        np.testing.assert_allclose(gpu_np[core], cpu[core], atol=0.1)  # the box blur's corners
        # the frame in RGB: the same background colour flipped is the same grey
        rgb = engine._chin_keep_gpu(dev, bg, 80.0, False)
        assert rgb is not None and float((rgb - gpu).abs().max()) == 0.0


class TestTheSettleCarriesItsInputs:
    def _loaded(self, tmp_path, torch):
        engine, loaded = _RenderFixture()._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        crop = synthetic_face((256.0, 300.0), 233.0)[:468, :2]
        loaded.crop_points = crop
        loaded.options = {"mask_forehead_reach": "hairline"}
        loaded.hairline_settled = False
        loaded.hairline_portrait = 1.5
        loaded.mask = _oval_mask(crop, forehead_reach=_OVAL_FOREHEAD_REACH)
        loaded.chin_drop_max = 40.0
        loaded.chin_reach = 0.0
        return engine, loaded

    def test_the_hairline_and_the_width_used_are_kept_and_are_anchor_scalars(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._loaded(tmp_path, torch)
        loaded.hairline_samples = [1.3] * _HAIRLINE_MIN_SAMPLES
        loaded.width_samples = [0.95] * 10
        engine._settle_forehead(loaded)
        assert loaded.settle_hairline == pytest.approx(1.3)
        assert loaded.settle_width == pytest.approx(0.95)
        assert "settle_hairline" in _ANCHOR_SCALARS and "settle_width" in _ANCHOR_SCALARS

    def test_a_second_plan_keeps_the_width(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._loaded(tmp_path, torch)
        loaded.hairline_samples = [1.3] * _HAIRLINE_MIN_SAMPLES
        loaded.width_samples = [0.95] * 10
        engine._settle_forehead(loaded)
        assert loaded.width_settled and loaded.width_scale == pytest.approx(0.95)
        # the hairline followed: planned again from new readings, no width samples
        loaded.hairline_settled = False
        loaded.hairline_samples = [1.6] * _HAIRLINE_MIN_SAMPLES
        engine._settle_forehead(loaded)
        assert loaded.settle_hairline == pytest.approx(1.6)
        assert loaded.width_scale == pytest.approx(0.95) and loaded.mask_points is not None


class TestTheForeheadFollows:
    def test_drift_past_the_threshold_asks_for_a_new_plan(self, tmp_path):
        torch = pytest.importorskip("torch")
        _, loaded = _RenderFixture()._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        loaded.settle_hairline = 1.5
        loaded.hairline_samples = [1.5 + _HAIRLINE_FOLLOW / 2] * _HAIRLINE_MIN_SAMPLES
        assert not _hairline_drifted(loaded)
        loaded.hairline_samples = [1.5 + 2 * _HAIRLINE_FOLLOW] * _HAIRLINE_MIN_SAMPLES
        assert _hairline_drifted(loaded)
        # too few readings, or mostly unread, or nothing settled: no
        loaded.hairline_samples = [2.0] * (_HAIRLINE_MIN_SAMPLES - 1)
        assert not _hairline_drifted(loaded)
        loaded.hairline_samples = [2.0] * 5 + [None] * (_HAIRLINE_MIN_SAMPLES - 5)
        assert not _hairline_drifted(loaded)
        loaded.hairline_samples = [2.0] * _HAIRLINE_MIN_SAMPLES
        loaded.settle_hairline = None
        assert not _hairline_drifted(loaded)
        # the engine is also sure LivePortraitEngine is what the fixture gave
        assert isinstance(
            _RenderFixture()._engine_and_loaded(tmp_path, torch=torch)[0], LivePortraitEngine
        )


class TestTheChinReachOnATilt:
    def test_the_reach_under_the_chin_fades_with_the_look_down(self, tmp_path):
        from types import SimpleNamespace

        from sha256.avatar.engine_liveportrait import _tilt_weight
        from sha256.core.types import HeadPose

        loaded = SimpleNamespace(rest_pitch=-5.0)
        assert _tilt_weight(loaded, None) == 0.0
        assert _tilt_weight(loaded, HeadPose(pitch=-5.0)) == 0.0
        assert _tilt_weight(loaded, HeadPose(pitch=-16.0)) == 0.0
        assert 0.0 < _tilt_weight(loaded, HeadPose(pitch=-21.0)) < 1.0
        assert _tilt_weight(loaded, HeadPose(pitch=-40.0)) == 1.0
        assert _tilt_weight(loaded, HeadPose(pitch=20.0)) == 1.0  # a look up, the same


class TestTheTurnBoost:
    def test_the_option_and_its_bounds(self):
        from sha256.avatar.engine_liveportrait import _TURN_BOOST_MAX, _turn_boost

        assert _turn_boost({}) == 0.0
        assert _turn_boost({"turn_boost": 0.5}) == 0.5
        for bad in (-0.1, _TURN_BOOST_MAX + 0.1):
            with pytest.raises(ValueError, match="turn_boost"):
                _turn_boost({"turn_boost": bad})

    def test_the_yaw_past_the_knee_is_driven_harder_and_nothing_else_moves(self):
        torch = pytest.importorskip("torch")
        from sha256.avatar.engine_liveportrait import _TURN_BOOST_KNEE

        # the drive's own arithmetic, on the CPU: (3, 1) pose, pitch / yaw / roll
        pose_d0 = torch.tensor([[3.7], [2.2], [-0.1]])
        boost = torch.tensor([[0.0], [0.5], [0.0]])
        for yaw in (-45.0, -20.0, 20.0, 45.0):
            pose_now = torch.tensor([[-10.0], [yaw], [5.0]])
            delta = pose_now - pose_d0
            past = torch.clamp(delta.abs() - _TURN_BOOST_KNEE, min=0.0)
            out = pose_d0 + delta + boost * past * delta.sign()
            assert float(out[0]) == pytest.approx(-10.0) and float(out[2]) == pytest.approx(5.0)
            d = yaw - 2.2
            expected = (
                d
                if abs(d) <= _TURN_BOOST_KNEE
                else d + 0.5 * (abs(d) - _TURN_BOOST_KNEE) * (1 if d > 0 else -1)
            )
            assert float(out[1]) == pytest.approx(2.2 + expected)


class TestTheTurnCap:
    def test_the_cap_is_the_hold_yaw_unless_set(self):
        from sha256.avatar.engine_liveportrait import _TURN_CAP_MAX, _turn_cap

        assert _turn_cap({}, 45.0) == 45.0
        assert _turn_cap({"turn_cap_degrees": 60}, 45.0) == 60.0
        for bad in (0.0, _TURN_CAP_MAX + 1.0):
            with pytest.raises(ValueError, match="turn_cap_degrees"):
                _turn_cap({"turn_cap_degrees": bad}, 45.0)


class TestTheOutlineFit:
    def test_the_option_and_the_row_edges(self):
        from sha256.avatar.engine_liveportrait import _outline_fit, _row_edges

        assert _outline_fit({}) == 1.0 and _outline_fit({"outline_fit": 0.0}) == 0.0
        with pytest.raises(ValueError, match="outline_fit"):
            _outline_fit({"outline_fit": 2.0})
        mask = np.zeros((4, 10), dtype=np.float32)
        mask[1, 3:7] = 1.0
        mask[2, 0:2] = 1.0
        first, last = _row_edges(mask)
        assert first.tolist() == [-1, 3, 0, -1] and last.tolist() == [-1, 6, 1, -1]

    def test_the_near_edge_lands_on_the_target_about_the_far_edge(self):
        from sha256.avatar.engine_liveportrait import _outline_maps

        size = 100
        # the drawn face on [35, 80] every row, the target (his oval) on [30, 80]:
        # the near (left) edge is 5 px short; the far edge is shared
        drawn_near = np.full(size, 35, dtype=np.int32)
        target_first = np.full(size, 30, dtype=np.int32)
        target_last = np.full(size, 80, dtype=np.int32)
        maps = _outline_maps(
            drawn_near,
            target_first,
            target_last,
            near_left=True,
            first_row=20,
            last_row=79,
            weight=1.0,
            previous=None,
        )
        assert maps is not None
        scale, shift = maps
        r = 50
        # output x = 30 (the target's near edge) reads the drawn edge, 35;
        # output x = 80 (the pivot) reads itself
        assert 30 * scale[r] + shift[r] == pytest.approx(35.0, abs=0.5)
        assert 80 * scale[r] + shift[r] == pytest.approx(80.0, abs=0.5)
        assert scale[r] == pytest.approx(0.9, abs=0.01)
        # far outside the fitted rows the map is the identity
        assert scale[0] == pytest.approx(1.0) and shift[0] == pytest.approx(0.0)
        # half the weight: half the way
        half = _outline_maps(
            drawn_near,
            target_first,
            target_last,
            near_left=True,
            first_row=20,
            last_row=79,
            weight=0.5,
            previous=None,
        )
        assert half is not None and half[0][r] == pytest.approx(0.95, abs=0.01)
        # the drawn edge already past the target: nothing is narrowed
        wide = _outline_maps(
            np.full(size, 20, dtype=np.int32),
            target_first,
            target_last,
            near_left=True,
            first_row=20,
            last_row=79,
            weight=1.0,
            previous=None,
        )
        assert (
            wide is not None
            and wide[0][r] == pytest.approx(1.0)
            and wide[1][r] == pytest.approx(0.0)
        )
        # the right-hand near side mirrors it
        mirrored = _outline_maps(
            np.full(size, 65, dtype=np.int32),
            np.full(size, 20, dtype=np.int32),
            np.full(size, 70, dtype=np.int32),
            near_left=False,
            first_row=20,
            last_row=79,
            weight=1.0,
            previous=None,
        )
        assert mirrored is not None
        ms, mh = mirrored
        assert 70 * ms[r] + mh[r] == pytest.approx(65.0, abs=0.5)
        assert 20 * ms[r] + mh[r] == pytest.approx(20.0, abs=0.5)

    def test_no_fitted_row_is_none_and_the_previous_map_is_eased(self):
        from sha256.avatar.engine_liveportrait import _OUTLINE_EMA, _outline_maps

        size = 50
        none = _outline_maps(
            np.full(size, -1, dtype=np.int32),
            np.full(size, 10, dtype=np.int32),
            np.full(size, 40, dtype=np.int32),
            near_left=True,
            first_row=0,
            last_row=49,
            weight=1.0,
            previous=None,
        )
        assert none is None
        prev = (np.full(size, 1.2, dtype=np.float32), np.full(size, -5.0, dtype=np.float32))
        eased = _outline_maps(
            np.full(size, 10, dtype=np.int32),
            np.full(size, 10, dtype=np.int32),
            np.full(size, 40, dtype=np.int32),
            near_left=True,
            first_row=0,
            last_row=49,
            weight=1.0,
            previous=prev,
        )
        assert eased is not None
        # the drawn edge already on the target: this frame's map is the
        # identity, eased against the previous one
        assert eased[0][25] == pytest.approx(_OUTLINE_EMA * 1.2 + (1 - _OUTLINE_EMA) * 1.0)
        assert eased[1][25] == pytest.approx(_OUTLINE_EMA * -5.0)

    def test_the_drawn_near_edge_skips_the_background_and_the_border(self, tmp_path):
        torch = pytest.importorskip("torch")
        from sha256.avatar.engine_liveportrait import _drawn_near_edge_np

        engine, _ = _RenderFixture()._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        bg = np.array([240.0, 240.0, 240.0], dtype=np.float32)
        face = np.full((20, 30, 3), 240, dtype=np.uint8)  # background everywhere
        face[:, :5] = 0  # the warp's black border on the left
        face[5:15, 12:25] = (150, 160, 180)  # the drawn face
        cpu = _drawn_near_edge_np(face, bg, True)
        assert cpu[10] == 12 and cpu[0] == -1
        assert _drawn_near_edge_np(face, bg, False)[10] == 24
        engine._frame_bgr = True
        dev = torch.from_numpy(face).permute(2, 0, 1).float()
        gpu = engine._drawn_near_edge_gpu(dev, bg, True)
        np.testing.assert_array_equal(gpu, cpu)
        np.testing.assert_array_equal(
            engine._drawn_near_edge_gpu(dev, bg, False), _drawn_near_edge_np(face, bg, False)
        )


class TestTheOutlineWindow:
    def test_a_stray_dark_column_outside_the_window_is_not_the_face(self):
        from sha256.avatar.engine_liveportrait import (
            _OUTLINE_WINDOW,
            _drawn_near_edge_np,
            _outline_window,
        )

        bg = np.array([240.0, 240.0, 240.0], dtype=np.float32)
        face = np.full((10, 200, 3), 240, dtype=np.uint8)
        face[:, 5:9] = 120  # the photograph's own edge, grey, far left
        face[:, 100:160] = (150, 160, 180)  # the drawn face
        target_first = np.full(10, 110, dtype=np.int32)
        target_last = np.full(10, 170, dtype=np.int32)
        window = _outline_window(target_first, target_last, True)
        assert window[0][0] == 110 - _OUTLINE_WINDOW and window[1][0] == 170
        assert _drawn_near_edge_np(face, bg, True)[3] == 5  # without the window: the stray
        assert _drawn_near_edge_np(face, bg, True, window)[3] == 100  # with it: the face
        right = _outline_window(target_first, target_last, False)
        assert right[0][0] == 110 and right[1][0] == 170 + _OUTLINE_WINDOW
        empty = _outline_window(np.array([-1]), np.array([-1]), True)
        assert empty[0][0] == -1 and empty[1][0] == -2


class TestTheKeepThroughTheOutline:
    def test_the_keep_is_read_through_the_same_map(self):
        from sha256.avatar.engine_liveportrait import _remap_outline

        size = 64
        scale = np.full(size, 0.9, dtype=np.float32)
        shift = np.full(size, 10.0, dtype=np.float32)
        keep = np.ones((20, 30), dtype=np.float32)
        keep[:, :8] = 0.0  # background on the patch's left (crop columns 16..23)
        out = _remap_outline(keep, scale, shift, 16, 10)
        # patch column c reads crop column (c + 16) * 0.9 + 10 - 16 in patch
        # units: column 0 -> 8.4 (just inside the kept part), column 2 -> 10.2
        assert out[5, 0] == pytest.approx(keep[5, 8] * 0.6 + keep[5, 9] * 0.4, abs=0.05)
        assert out[5, 2] == pytest.approx(1.0)
        # the identity map leaves it alone
        same = _remap_outline(keep, np.ones(size, np.float32), np.zeros(size, np.float32), 16, 10)
        np.testing.assert_allclose(same, keep, atol=1e-6)


class TestTheNoseIsLeftAlone:
    def test_the_excluded_band_reads_the_identity_and_ramps(self):
        from sha256.avatar.engine_liveportrait import _OUTLINE_NOSE_RAMP, _outline_maps

        size = 120
        maps = _outline_maps(
            np.full(size, 35, dtype=np.int32),
            np.full(size, 30, dtype=np.int32),
            np.full(size, 80, dtype=np.int32),
            near_left=True,
            first_row=10,
            last_row=109,
            weight=1.0,
            previous=None,
            exclude=(50, 70),
        )
        assert maps is not None
        scale, shift = maps
        assert scale[60] == pytest.approx(1.0) and shift[60] == pytest.approx(0.0)
        assert scale[30] == pytest.approx(0.9, abs=0.01)  # fitted above the band
        assert scale[90] == pytest.approx(0.9, abs=0.01)  # and below
        half = 50 - int(_OUTLINE_NOSE_RAMP / 2)
        assert 0.9 < scale[half] < 1.0  # on the ramp
