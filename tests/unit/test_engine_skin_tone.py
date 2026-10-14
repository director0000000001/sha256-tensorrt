"""The skin's tone (day 21, 2026-10-14): the cells that are skin on both
sides, the cast and the contrast per channel, the EMA that carries them,
where the tone lands on a pixel (the cast everywhere, the contrast by
lightness), the CPU paste with and without the match, and the GPU helper
against the numpy one."""

from __future__ import annotations

import numpy as np
import pytest

from sha256.avatar.engine_liveportrait import (
    _EDGE_FEATURES,
    _SKIN_CELLS_MIN,
    _SKIN_GAIN_MAX,
    _SKIN_MATCH_DEFAULT,
    _SKIN_TONE_INERTIA,
    _TONE_DARK_HIGH,
    _TONE_DARK_LOW,
    _TONE_IDENTITY,
    CROP_SIZE,
    LivePortraitEngine,
    _as_numpy_tone,
    _core_median,
    _mask_bbox,
    _oval_mask,
    _Portrait,
    _skin_cells,
    _skin_match,
    _skin_tone,
    _tone_applied,
    _tone_follow,
    _tone_on_skin,
    _tone_weight,
    _transformed_box,
    crop_matrix,
)
from tests.unit.test_engine_liveportrait import descriptor, face_points


def planes(
    scene_mean: tuple[float, float, float],
    scene_spread: float,
    face_mean: tuple[float, float, float],
    face_spread: float,
    size: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mask-weighted cell planes: a core of whole coverage, the same texture
    on both sides scaled to each spread -- so the spreads' ratio is exact."""
    rng = np.random.default_rng(1)
    coverage = np.zeros((size, size), dtype=np.float32)
    coverage[2:-2, 2:-2] = 1.0
    texture = rng.standard_normal((size, size, 1)).astype(np.float32)
    scene = np.asarray(scene_mean, np.float32) + texture * scene_spread
    face = np.asarray(face_mean, np.float32) + texture * face_spread
    return face * coverage[:, :, None], scene * coverage[:, :, None], coverage


class TestOption:
    def test_default_and_bounds(self):
        assert _skin_match({}) == _SKIN_MATCH_DEFAULT == 1.0
        assert _skin_match({"skin_match": 0.5}) == 0.5
        assert _skin_match({"skin_match": 0}) == 0.0
        for bad in (-0.1, 1.5):
            with pytest.raises(ValueError, match="skin_match"):
                _skin_match({"skin_match": bad})


class TestSkinCells:
    def test_the_core_less_the_dark_cells_on_either_side_and_the_zone(self):
        face, scene, coverage = planes((120, 120, 120), 0.0, (150, 150, 150), 0.0)
        scene[4:8, 4:8] = 20.0  # hair under the mask: dark on the room's side
        face[10:14, 10:14] = 30.0  # the render's brow: dark on its side
        zone = np.zeros_like(coverage)
        zone[16:20, 16:20] = 1.0  # his feature zone

        skin = _skin_cells(face, scene, coverage, zone)

        assert skin.dtype == np.float32
        assert skin[0, 0] == 0.0  # outside the core
        assert skin[2, 2] == 1.0
        assert skin[5, 5] == 0.0 and skin[11, 11] == 0.0 and skin[17, 17] == 0.0
        assert skin.sum() == 20 * 20 - 3 * 16

    def test_no_zone_is_the_core_less_the_dark(self):
        face, scene, coverage = planes((120, 120, 120), 0.0, (150, 150, 150), 0.0)
        assert _skin_cells(face, scene, coverage, None).sum() == 20 * 20

    def test_the_core_median_is_the_dark_rules_level(self):
        coverage = np.ones((8, 8), dtype=np.float32)
        cells = np.full((8, 8, 3), 120.0, dtype=np.float32)
        cells[0, 0] = 10.0
        assert _core_median(cells, coverage) == 120.0
        assert _core_median(cells, np.zeros_like(coverage)) == 0.0


class TestSkinTone:
    def test_the_cast_meets_the_means_and_the_contrast_the_spreads(self):
        face, scene, coverage = planes((120, 130, 140), 10.0, (150, 150, 150), 8.0)
        skin = _skin_cells(face, scene, coverage, None)

        tone = _skin_tone(face, scene, coverage, skin)

        assert tone is not None and tone.shape == (3, 3) and tone.dtype == np.float32
        cast, contrast, mean = tone
        np.testing.assert_allclose(cast, np.array([120, 130, 140]) / 150.0, atol=1e-3)
        np.testing.assert_allclose(mean, [120, 130, 140], atol=0.5)
        # the contrast makes the spread the room's: cast * contrast * 8 = 10
        # where it is not capped
        expected = np.minimum(10.0 / (cast * 8.0), _SKIN_GAIN_MAX)
        np.testing.assert_allclose(contrast, expected, atol=1e-3)
        # applied to the render's skin (weight 1), it has the room's mean and spread
        weight = np.maximum(coverage, 1e-3)[:, :, None]
        matched = _tone_on_skin(face / weight, cast, contrast, mean, None)
        core = coverage > 0.9
        for c in range(3):
            assert abs(matched[core, c].mean() - mean[c]) < 0.5
            if contrast[c] < _SKIN_GAIN_MAX:
                assert abs(matched[core, c].std() - (scene / weight)[core, c].std()) < 0.1

    def test_the_cast_and_the_contrast_are_capped(self):
        face, scene, coverage = planes((110, 110, 110), 10.0, (150, 150, 150), 4.0)
        skin = _skin_cells(face, scene, coverage, None)
        tone = _skin_tone(face, scene, coverage, skin)
        assert tone is not None
        np.testing.assert_allclose(tone[0], 110.0 / 150.0, atol=5e-3)  # within the cap
        np.testing.assert_allclose(tone[1], _SKIN_GAIN_MAX)  # 10 / (0.733 * 4) = 3.4: capped
        dark = planes((90, 90, 90), 10.0, (150, 150, 150), 10.0)
        tone = _skin_tone(*dark, _skin_cells(*dark, None))
        assert tone is not None
        np.testing.assert_allclose(tone[0], 0.7)  # 0.6: capped
        bright = planes((250, 250, 250), 10.0, (100, 100, 100), 10.0)
        tone = _skin_tone(*bright, _skin_cells(*bright, None))
        assert tone is not None
        np.testing.assert_allclose(tone[0], _SKIN_GAIN_MAX)  # 2.5: capped

    def test_flat_pictures_cast_the_mean_and_keep_the_contrast(self):
        face, scene, coverage = planes((120, 120, 120), 0.0, (150, 150, 150), 0.0)
        skin = _skin_cells(face, scene, coverage, None)
        tone = _skin_tone(face, scene, coverage, skin)
        assert tone is not None
        np.testing.assert_allclose(tone[0], 0.8, atol=1e-3)
        np.testing.assert_allclose(tone[1], 1.0)
        np.testing.assert_allclose(tone[2], 120.0, atol=1e-3)

    def test_too_few_cells_measure_nothing(self):
        face, scene, coverage = planes((100, 100, 100), 10.0, (150, 150, 150), 10.0)
        skin = _skin_cells(face, scene, coverage, None)
        skin[:, 3:] = 0.0  # a sliver of skin, one column of the core
        assert skin.sum() < _SKIN_CELLS_MIN
        assert _skin_tone(face, scene, coverage, skin) is None


class TestFollowAndApply:
    def test_the_first_measurement_is_taken_whole_then_the_ema(self):
        measured = np.array(
            [[1.2, 1.1, 1.0], [1.3, 1.0, 0.9], [150.0, 140.0, 130.0]], dtype=np.float32
        )
        first = _tone_follow(_TONE_IDENTITY.copy(), measured)
        np.testing.assert_array_equal(first, measured)
        again = np.array(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [150.0, 140.0, 130.0]], dtype=np.float32
        )
        second = _tone_follow(first, again)
        expected = measured * _SKIN_TONE_INERTIA + again * (1.0 - _SKIN_TONE_INERTIA)
        np.testing.assert_allclose(second, expected, atol=1e-5)
        assert second.dtype == np.float32

    def test_nothing_measured_keeps_the_previous(self):
        tone = np.array([[1.2, 1.1, 1.0], [1.3, 1.0, 0.9], [150.0, 140.0, 130.0]], dtype=np.float32)
        assert _tone_follow(tone, None) is tone

    def test_the_amount_pulls_toward_the_identity(self):
        tone = np.array([[1.4, 1.0, 0.8], [0.8, 1.0, 1.2], [150.0, 140.0, 130.0]], dtype=np.float32)
        cast, contrast, mean = _tone_applied(tone, 0.5)
        np.testing.assert_allclose(cast, [1.2, 1.0, 0.9])
        np.testing.assert_allclose(contrast, [0.9, 1.0, 1.1])
        np.testing.assert_allclose(mean, tone[2])
        cast, contrast, _ = _tone_applied(tone, 0.0)
        np.testing.assert_allclose(cast, 1.0)
        np.testing.assert_allclose(contrast, 1.0)
        cast, contrast, mean = _tone_applied(tone, 1.0)
        np.testing.assert_allclose(cast, tone[0])
        assert cast.dtype == contrast.dtype == mean.dtype == np.float32


class TestToneOnAPixel:
    """Where the tone lands: the cast on every pixel, the contrast by each
    pixel's lightness against the render's skin -- an iris keeps its look."""

    def test_the_weight_follows_each_pixels_lightness_against_the_skins(self):
        placed = np.full((16, 12, 3), 100.0, dtype=np.float32)
        placed[4:8, 4:8] = 30.0  # an iris: under the low bound
        placed[10:12, 4:8] = 100.0 * (_TONE_DARK_LOW + _TONE_DARK_HIGH) / 2.0  # halfway
        weight = _tone_weight(placed, 100.0)
        assert weight is not None and weight.shape == (16, 12) and weight.dtype == np.float32
        assert weight[1, 1] == 1.0 and weight[5, 5] == 0.0
        assert weight[10, 5] == pytest.approx(0.5, abs=1e-3)
        assert _tone_weight(placed, 0.0) is None  # no level known: everywhere

    def test_the_cast_moves_a_dark_pixel_little_and_the_contrast_not_at_all(self):
        placed = np.full((16, 12, 3), 100.0, dtype=np.float32)
        placed[4:8, 4:8] = 30.0
        weight = _tone_weight(placed, 100.0)
        cast = np.array([1.2, 1.0, 0.9], dtype=np.float32)
        contrast = np.array([1.3, 1.0, 0.8], dtype=np.float32)
        mean = np.array([120.0, 100.0, 90.0], dtype=np.float32)  # the room's skin
        toned = _tone_on_skin(placed, cast, contrast, mean, weight)
        # skin: cast to the room's mean, and at the mean the contrast adds nothing
        np.testing.assert_allclose(toned[1, 1], [120.0, 100.0, 90.0], atol=1e-3)
        # the iris: the cast alone -- six, zero and three levels, its hue kept
        np.testing.assert_allclose(toned[5, 5], [36.0, 30.0, 27.0], atol=1e-3)
        everywhere = _tone_on_skin(placed, cast, contrast, mean, None)
        # with the contrast on the iris too: pulled away from the room's mean
        np.testing.assert_allclose(
            everywhere[5, 5],
            [36.0 + 0.3 * (36.0 - 120.0), 30.0, 27.0 - 0.2 * (27.0 - 90.0)],
            atol=1e-3,
        )


def _loaded_portrait(tmp_path, options: dict | None = None) -> _Portrait:
    pts = face_points()
    to_crop, _ = crop_matrix(pts)
    crop_points = (pts @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
    loaded = _Portrait(descriptor(tmp_path))
    loaded.options = dict(options or {})
    loaded.mask = _oval_mask(crop_points)
    loaded.mask_bbox = _mask_bbox(loaded.mask)
    return loaded


def _blocks(shape: tuple[int, int], period: int, depth: float, rng_seed: int = 3) -> np.ndarray:
    """A coarse random block texture, +-depth, that survives the warp."""
    rng = np.random.default_rng(rng_seed)
    rows = -(-shape[0] // period)
    cols = -(-shape[1] // period)
    coarse = rng.choice([-depth, depth], size=(rows, cols)).astype(np.float32)
    fine = np.repeat(np.repeat(coarse, period, axis=0), period, axis=1)
    return fine[: shape[0], : shape[1]]


class TestPasteWithTheTone:
    """The CPU paste (``_paste``), pure numpy/cv2."""

    def test_at_zero_nothing_is_learnt(self, tmp_path):
        pts = face_points()
        _, to_frame = crop_matrix(pts)
        loaded = _loaded_portrait(tmp_path, {"skin_match": 0.0})
        assert loaded.mask is not None
        scene = np.full((720, 1280, 3), 60, dtype=np.uint8)
        face = np.full((CROP_SIZE, CROP_SIZE, 3), 210, dtype=np.uint8)

        LivePortraitEngine()._paste(scene, face, to_frame, loaded, loaded.mask)

        np.testing.assert_array_equal(loaded.tone, _TONE_IDENTITY)

    def test_a_flat_face_on_a_flat_room_lands_on_the_room(self, tmp_path):
        """What the plane alone did already -- still true with the tone,
        whose cast is capped and leaves the plane the rest."""
        pts = face_points()
        _, to_frame = crop_matrix(pts)
        loaded = _loaded_portrait(tmp_path)
        assert loaded.mask is not None and loaded.mask_bbox is not None
        scene = np.full((720, 1280, 3), 60, dtype=np.uint8)
        face = np.full((CROP_SIZE, CROP_SIZE, 3), 210, dtype=np.uint8)

        output = LivePortraitEngine()._paste(scene, face, to_frame, loaded, loaded.mask)

        tone = _as_numpy_tone(loaded.tone)
        np.testing.assert_allclose(tone[0], 0.7)  # 60 / 210: capped
        np.testing.assert_allclose(tone[1], 1.0)  # flat: no spread to match
        bx0, by0, bx1, by1 = loaded.mask_bbox
        centre = np.array([[(bx0 + bx1) / 2.0, (by0 + by1) / 2.0]], dtype=np.float32)
        frame_xy = centre @ to_frame[:, :2].T + to_frame[:, 2]
        cx, cy = int(frame_xy[0, 0]), int(frame_xy[0, 1])
        assert abs(int(output[cy, cx, 0]) - 60) < 15

    def test_a_flatter_face_takes_the_rooms_spread(self, tmp_path):
        """The thing the plane cannot do: a render with a third of the
        room's texture contrast is pasted with more of it (the contrast,
        capped at 1.4), so the two sides of the join read alike."""
        pts = face_points()
        _, to_frame = crop_matrix(pts)
        scene = np.clip(120.0 + _blocks((720, 1280), 12, 24.0), 0, 255).astype(np.uint8)
        scene = np.repeat(scene[:, :, None], 3, axis=2)
        face = np.clip(120.0 + _blocks((CROP_SIZE, CROP_SIZE), 12, 8.0, 5), 0, 255)
        face = np.repeat(face.astype(np.uint8)[:, :, None], 3, axis=2)

        spreads = {}
        for amount in (0.0, 1.0):
            loaded = _loaded_portrait(tmp_path, {"skin_match": amount, "edge_match": 0.0})
            assert loaded.mask is not None and loaded.mask_bbox is not None
            engine = LivePortraitEngine()
            for _ in range(3):
                output = engine._paste(scene, face, to_frame, loaded, loaded.mask)
            x0, y0, x1, y1 = _transformed_box(to_frame, loaded.mask_bbox, 1280, 720)
            qx, qy = (x1 - x0) // 4, (y1 - y0) // 4
            core = output[y0 + qy : y1 - qy, x0 + qx : x1 - qx, 0].astype(np.float32)
            spreads[amount] = float(core.std())
            if amount == 1.0:
                tone = _as_numpy_tone(loaded.tone)
                np.testing.assert_allclose(tone[0], 1.0, atol=0.05)
                np.testing.assert_allclose(tone[1], _SKIN_GAIN_MAX, atol=0.05)

        assert spreads[1.0] > spreads[0.0] * 1.3

    def test_the_tone_is_read_one_frame_in_four_and_applied_on_every_one(self, tmp_path):
        from sha256.avatar.engine_liveportrait import _SKIN_TONE_EVERY

        pts = face_points()
        _, to_frame = crop_matrix(pts)
        loaded = _loaded_portrait(tmp_path)
        assert loaded.mask is not None
        face = np.full((CROP_SIZE, CROP_SIZE, 3), 210, dtype=np.uint8)
        engine = LivePortraitEngine()
        tones = []
        for frame in range(2 * _SKIN_TONE_EVERY + 1):
            scene = np.full((720, 1280, 3), 60 + 10 * frame, dtype=np.uint8)  # the room drifting
            engine._paste(scene, face, to_frame, loaded, loaded.mask)
            tones.append(_as_numpy_tone(loaded.tone).copy())
        # read on frames 0, 4 and 8; the frames between keep the last reading
        for frame in range(1, _SKIN_TONE_EVERY):
            np.testing.assert_array_equal(tones[frame], tones[0])
        assert not np.array_equal(tones[_SKIN_TONE_EVERY], tones[0])
        np.testing.assert_array_equal(tones[_SKIN_TONE_EVERY + 1], tones[_SKIN_TONE_EVERY])
        assert not np.array_equal(tones[2 * _SKIN_TONE_EVERY], tones[_SKIN_TONE_EVERY])
        assert loaded.tone_frames == 2 * _SKIN_TONE_EVERY + 1
        assert float(loaded.tone_level) > 0.0

    def test_the_tone_does_not_move_while_the_face_is_held(self, tmp_path):
        pts = face_points()
        _, to_frame = crop_matrix(pts)
        loaded = _loaded_portrait(tmp_path)
        assert loaded.mask is not None
        scene = np.full((720, 1280, 3), 60, dtype=np.uint8)
        face = np.full((CROP_SIZE, CROP_SIZE, 3), 210, dtype=np.uint8)
        engine = LivePortraitEngine()
        engine._paste(scene, face, to_frame, loaded, loaded.mask)
        before = _as_numpy_tone(loaded.tone).copy()

        loaded.holding = True
        engine._paste(np.full_like(scene, 200), face, to_frame, loaded, loaded.mask)

        np.testing.assert_array_equal(_as_numpy_tone(loaded.tone), before)

    def test_the_paste_keeps_an_iris_its_look(self, tmp_path):
        """A face with a dark brown patch where his eyes are, pasted onto a
        cooler room: the skin takes the room's cast whole, the iris by
        little -- red stays far over blue."""
        pts = face_points()
        to_crop, to_frame = crop_matrix(pts)
        crop_points = (pts @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
        loaded = _loaded_portrait(tmp_path, {"edge_match": 0.0})
        assert loaded.mask is not None
        loaded.crop_points = crop_points
        loaded.features = pts[[k for group in _EDGE_FEATURES for k in group]].astype(np.float32)

        scene = np.zeros((720, 1280, 3), dtype=np.uint8)
        scene[:, :] = (120, 110, 100)  # a cooler room (BGR)
        face = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
        face[:, :] = (100, 115, 135)  # warmer skin
        eye = crop_points[list(_EDGE_FEATURES[0])]
        ex, ey = (round(float(eye[:, 0].mean())), round(float(eye[:, 1].mean())))
        face[ey - 4 : ey + 4, ex - 6 : ex + 6] = (10, 40, 80)  # a dark brown iris

        output = LivePortraitEngine()._paste(scene, face, to_frame, loaded, loaded.mask)

        eye_frame = np.array([[ex, ey]], np.float32) @ to_frame[:, :2].T + to_frame[:, 2]
        fx, fy = int(eye_frame[0, 0]), int(eye_frame[0, 1])
        iris = output[fy, fx].astype(int)
        assert iris[2] > 2.5 * max(iris[0], 1)  # still brown: red far over blue
        cheek = output[fy + 60, fx].astype(int)
        # the skin: the room's own balance now (its red twenty under its blue)
        assert abs((cheek[2] - cheek[0]) - (100 - 120)) < 12


def _cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


@pytest.mark.skipif(not _cuda(), reason="needs CUDA")
class TestToneOnTheGpu:
    """``_skin_tone_gpu`` against the numpy functions on the same box."""

    def test_the_gpu_tone_is_the_numpy_one(self, tmp_path):
        import cv2
        import torch

        from sha256.avatar.engine_liveportrait import _CORRECTION_DOWNSCALE, _feature_exclusion

        h, w = 160, 128
        rng = np.random.default_rng(7)
        region_np = np.clip(
            np.array([110, 120, 130], np.float32) + rng.standard_normal((h, w, 3)) * 12.0, 0, 255
        ).astype(np.float32)
        placed_np = np.clip(
            np.array([150, 140, 150], np.float32) + rng.standard_normal((h, w, 3)) * 6.0, 0, 255
        ).astype(np.float32)
        # a shading both pictures share, the room's twice the render's, so the
        # cells (4 x 4 pixels, where pixel noise averages out) have a spread
        ramp = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None, None]
        region_np = np.clip(region_np + ramp * 16.0, 0, 255).astype(np.float32)
        placed_np = np.clip(placed_np + ramp * 8.0, 0, 255).astype(np.float32)
        placed_np[52:68, 42:58] = (30.0, 40.0, 50.0)  # a dark patch: an iris, a brow
        alpha_np = np.zeros((h, w), dtype=np.float32)
        alpha_np[12:-12, 10:-10] = 1.0
        alpha_np[12:20, :] = 0.5  # a feathered top
        # His feature points in the box: _feature_exclusion takes the groups'
        # lengths from _EDGE_FEATURES, so exactly that many, a small zone.
        n = sum(len(g) for g in _EDGE_FEATURES)
        features = np.tile(np.array([[50.0, 60.0]], np.float32), (n, 1)) + rng.uniform(
            -6, 6, size=(n, 2)
        ).astype(np.float32)

        loaded = _Portrait(descriptor(tmp_path))
        loaded.options = {}
        loaded.features = features
        loaded.holding = False

        engine = LivePortraitEngine()
        engine._torch = torch
        device = torch.device("cuda")
        placed = torch.from_numpy(placed_np).permute(2, 0, 1).contiguous().to(device)
        region = torch.from_numpy(region_np).permute(2, 0, 1).contiguous().to(device)
        alpha = torch.from_numpy(alpha_np).to(device)[None]

        out = engine._skin_tone_gpu(
            loaded, placed, region, alpha, alpha_np, 1.0, 0, 0, w, h, measure=True
        )
        torch.cuda.synchronize()
        tone_gpu = _as_numpy_tone(loaded.tone)

        # The numpy side, as the CPU paste does it.
        small = (max(w // _CORRECTION_DOWNSCALE, 8), max(h // _CORRECTION_DOWNSCALE, 8))
        alpha3 = np.repeat(alpha_np[:, :, None], 3, axis=2)
        face_small = cv2.resize(placed_np * alpha3, small, interpolation=cv2.INTER_AREA)
        scene_small = cv2.resize(region_np * alpha3, small, interpolation=cv2.INTER_AREA)
        coverage = cv2.resize(alpha_np, small, interpolation=cv2.INTER_AREA)
        cells = features / (w / small[0], h / small[1])
        zone = _feature_exclusion(cells.astype(np.float32), coverage.shape)
        skin = _skin_cells(face_small, scene_small, coverage, zone)
        measured = _skin_tone(face_small, scene_small, coverage, skin)
        assert measured is not None
        tone_cpu = _tone_follow(_TONE_IDENTITY.copy(), measured)

        np.testing.assert_allclose(tone_gpu, tone_cpu, atol=2e-2)
        assert float(loaded.tone_level) == pytest.approx(
            _core_median(face_small, coverage), abs=0.5
        )
        cast, contrast, mean = _tone_applied(tone_cpu, 1.0)
        weight = _tone_weight(placed_np, _core_median(face_small, coverage))
        assert weight is not None
        expected = _tone_on_skin(placed_np, cast, contrast, mean, weight)
        np.testing.assert_allclose(out.permute(1, 2, 0).cpu().numpy(), expected, atol=0.6)
        # the dark patch: the cast alone (its hue kept); the skin: the contrast too
        np.testing.assert_allclose(expected[60, 50], placed_np[60, 50] * cast, atol=0.05)
        skin_box = (slice(120, 150), slice(90, 110))
        assert np.abs(expected[skin_box] - placed_np[skin_box] * cast).max() > 0.5

        # A frame without a reading: the tone in force applied, nothing moved.
        again = engine._skin_tone_gpu(
            loaded, placed, region, alpha, alpha_np, 1.0, 0, 0, w, h, measure=False
        )
        torch.cuda.synchronize()
        np.testing.assert_allclose(_as_numpy_tone(loaded.tone), tone_gpu)
        np.testing.assert_allclose(again.cpu().numpy(), out.cpu().numpy())
        # A second reading: the EMA, on the device, still the numpy rule.
        engine._skin_tone_gpu(
            loaded, placed, region, alpha, alpha_np, 1.0, 0, 0, w, h, measure=True
        )
        torch.cuda.synchronize()
        np.testing.assert_allclose(
            _as_numpy_tone(loaded.tone), _tone_follow(tone_cpu, measured), atol=2e-2
        )

    def test_a_held_face_keeps_the_tone_on_the_device(self, tmp_path):
        import torch

        h, w = 64, 64
        loaded = _Portrait(descriptor(tmp_path))
        loaded.options = {}
        loaded.features = None
        loaded.holding = True
        engine = LivePortraitEngine()
        engine._torch = torch
        device = torch.device("cuda")
        placed = torch.full((3, h, w), 200.0, device=device)
        region = torch.full((3, h, w), 100.0, device=device)
        alpha_np = np.ones((h, w), dtype=np.float32)
        alpha = torch.from_numpy(alpha_np).to(device)[None]

        # the paste reads nothing on a held frame: measure=False (see _paste_gpu)
        out = engine._skin_tone_gpu(
            loaded, placed, region, alpha, alpha_np, 1.0, 0, 0, w, h, measure=False
        )
        torch.cuda.synchronize()

        np.testing.assert_array_equal(_as_numpy_tone(loaded.tone), _TONE_IDENTITY)
        assert float(out.mean()) == 200.0
