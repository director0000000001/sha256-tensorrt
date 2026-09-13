"""The avatar's hair (day 17, C1b): the matte's alpha and box, the worker
thread, the paste under a hair matte, and the engine's switch on the CPU
fixture -- no segmenter model, no GPU."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sha256.avatar.engine_liveportrait import (
    _FOREHEAD_REACH_MAX,
    _HAIR_FOREHEAD_OVERLAP,
    _OVAL_FOREHEAD_REACH,
    CROP_SIZE,
    LivePortraitEngine,
    _forehead_reach,
    _forehead_reach_is_hairline,
    _hair_mode,
    _mask_bbox,
    _oval_mask,
    _Portrait,
    _transformed_box,
    crop_matrix,
)
from sha256.avatar.hair import (
    HairMatteWorker,
    hair_alpha,
    hair_box,
    head_confidence,
    skin_weight,
    union_box,
)
from tests.unit.test_engine_liveportrait import TestAnchorGatingInRender as _RenderFixture
from tests.unit.test_engine_liveportrait import descriptor, face_points


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class TestOption:
    def test_off_avatar_and_booleans(self):
        assert _hair_mode({}) == "off"
        assert _hair_mode({"hair": "avatar"}) == "avatar"
        assert _hair_mode({"hair": True}) == "avatar"
        assert _hair_mode({"hair": False}) == "off"
        assert _hair_mode({"hair": None}) == "off"
        with pytest.raises(ValueError, match="hair"):
            _hair_mode({"hair": "wig"})

    def test_the_forehead_reaches_the_hairline_by_default_since_day_17(self):
        assert _forehead_reach_is_hairline({})
        assert _forehead_reach({}) == _OVAL_FOREHEAD_REACH  # until the anchor settles it
        assert not _forehead_reach_is_hairline({"mask_forehead_reach": 0.85})


class TestAlpha:
    def test_the_curve_keeps_the_hair_and_drops_the_smear(self):
        conf = np.zeros((128, 128), dtype=np.float32)
        conf[20:60, 30:90] = 0.95  # hair
        conf[70:110, 30:90] = 0.2  # the beard's "maybe"
        alpha = hair_alpha(conf, erode=0, feather=0, border=0)
        assert alpha.shape == (128, 128, 1) and alpha.dtype == np.float32
        assert alpha[40, 60, 0] == pytest.approx(1.0)
        assert alpha[90, 60, 0] == 0.0
        assert alpha[5, 5, 0] == 0.0

    def test_the_erosion_pulls_the_edge_in_before_the_feather(self):
        conf = np.zeros((128, 128), dtype=np.float32)
        conf[20:60, 30:90] = 1.0
        eroded = hair_alpha(conf, erode=3, feather=0, border=0)
        assert eroded[20, 60, 0] == 0.0 and eroded[22, 60, 0] == 0.0
        assert eroded[23, 60, 0] == 1.0 and eroded[40, 60, 0] == 1.0
        assert eroded[40, 30, 0] == 0.0 and eroded[40, 33, 0] == 1.0

    def test_the_feather_softens_the_edge(self):
        conf = np.zeros((128, 128), dtype=np.float32)
        conf[20:60, 30:90] = 1.0
        hard = hair_alpha(conf, erode=0, feather=0, border=0)
        soft = hair_alpha(conf, erode=0, feather=5, border=0)
        assert hard[19, 60, 0] == 0.0 and hard[20, 60, 0] == 1.0
        assert 0.0 < soft[19, 60, 0] < 1.0 and 0.0 < soft[20, 60, 0] < 1.0
        assert soft[40, 60, 0] == pytest.approx(1.0)

    def test_the_face_skin_joins_above_the_eyes_only(self):
        pts = face_points()
        to_crop, _ = crop_matrix(pts)
        crop_points = (pts @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
        weight = skin_weight(crop_points, CROP_SIZE)
        eye_y = float(crop_points[[33, 133, 362, 263], 1].mean())
        face_h = float(crop_points[:, 1].max() - crop_points[:, 1].min())
        assert weight.shape == (CROP_SIZE, CROP_SIZE)
        assert weight[int(eye_y - 0.3 * face_h), 10] == 1.0
        assert weight[int(eye_y + 0.2 * face_h), 500] == 0.0
        assert 0.0 < weight[int(eye_y), 256] < 1.0
        conf = np.zeros((CROP_SIZE, CROP_SIZE, 2), dtype=np.float32)
        conf[:, :, 1] = 1.0  # skin everywhere, no hair
        head = head_confidence(conf, weight)
        assert head[int(eye_y - 0.3 * face_h), 10] == 1.0
        assert head[int(eye_y + 0.2 * face_h), 10] == 0.0
        conf[:, :, 0] = 0.6  # hair everywhere wins below the eyes
        assert head_confidence(conf, weight)[int(eye_y + 0.2 * face_h), 10] == pytest.approx(0.6)
        plain = np.full((8, 8), 0.4, dtype=np.float32)
        assert np.array_equal(head_confidence(plain, None), plain)

    def test_hair_the_picture_cuts_fades_at_the_border(self):
        conf = np.ones((128, 128), dtype=np.float32)
        alpha = hair_alpha(conf, erode=0, feather=0, border=16)
        assert alpha[0, 64, 0] == 0.0 and alpha[64, 0, 0] == 0.0
        assert alpha[127, 64, 0] == 0.0 and alpha[64, 127, 0] == 0.0
        assert alpha[8, 64, 0] == pytest.approx(0.5)
        assert alpha[16, 64, 0] == pytest.approx(1.0)
        assert alpha[64, 64, 0] == pytest.approx(1.0)


class TestBoxes:
    def test_the_box_is_around_the_hair_with_a_margin(self):
        alpha = np.zeros((128, 128, 1), dtype=np.float32)
        alpha[20:60, 30:90, 0] = 1.0
        assert hair_box(alpha, margin=4) == (26, 16, 94, 64)
        assert hair_box(alpha, margin=100) == (0, 0, 128, 128)
        assert hair_box(np.zeros((128, 128, 1), dtype=np.float32)) is None

    def test_union(self):
        assert union_box(None, None) is None
        assert union_box((1, 2, 3, 4), None) == (1, 2, 3, 4)
        assert union_box(None, (1, 2, 3, 4)) == (1, 2, 3, 4)
        assert union_box((10, 10, 20, 20), (5, 15, 15, 30)) == (5, 10, 20, 30)


class TestWorker:
    def test_the_newest_picture_gets_the_newest_matte(self):
        seen: list[float] = []

        def segment(rgb):
            seen.append(float(rgb[0, 0, 0]))
            return np.full(rgb.shape[:2], float(rgb[0, 0, 0]) / 255.0, dtype=np.float32)

        worker = HairMatteWorker(segment)
        try:
            assert worker.latest() is None
            worker.submit("a", np.full((8, 8, 3), 51, dtype=np.uint8))
            assert _wait_for(lambda: worker.latest() is not None)
            tag, seq, matte = worker.latest()
            assert tag == "a" and seq == 1 and matte[0, 0] == pytest.approx(0.2)
            worker.submit("b", np.full((8, 8, 3), 102, dtype=np.uint8))
            assert _wait_for(lambda: worker.latest()[1] == 2)
            tag, seq, matte = worker.latest()
            assert tag == "b" and matte[0, 0] == pytest.approx(0.4)
        finally:
            worker.stop()
        assert not worker.is_alive

    def test_a_failing_segmenter_drops_the_picture_and_goes_on(self):
        calls = {"n": 0}

        def segment(rgb):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return np.zeros(rgb.shape[:2], dtype=np.float32)

        worker = HairMatteWorker(segment)
        try:
            worker.submit("a", np.zeros((4, 4, 3), dtype=np.uint8))
            assert _wait_for(lambda: calls["n"] >= 1)
            assert worker.latest() is None
            worker.submit("a", np.zeros((4, 4, 3), dtype=np.uint8))
            assert _wait_for(lambda: worker.latest() is not None)
            assert worker.latest()[1] == 1
        finally:
            worker.stop()


class TestPasteWithHair:
    """``_paste`` with a hair matte: numpy/cv2 only, like TestPaste."""

    def _loaded(self, tmp_path) -> tuple[_Portrait, np.ndarray]:
        pts = face_points()
        to_crop, to_frame = crop_matrix(pts)
        crop_points = (pts @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
        loaded = _Portrait(descriptor(tmp_path))
        loaded.mask = _oval_mask(crop_points)
        loaded.mask_bbox = _mask_bbox(loaded.mask)
        return loaded, to_frame

    def _hair_above_the_face(self, loaded: _Portrait) -> np.ndarray:
        assert loaded.mask_bbox is not None
        x0, y0, x1, _ = loaded.mask_bbox
        matte = np.zeros((CROP_SIZE, CROP_SIZE, 1), dtype=np.float32)
        top = max(y0 - 60, 0)
        matte[top : y0 + 10, x0 + 10 : x1 - 10, 0] = 1.0  # a block over the forehead's top
        return matte

    def test_off_is_the_plain_paste_and_on_reaches_past_the_face_box(self, tmp_path):
        loaded, to_frame = self._loaded(tmp_path)
        assert loaded.mask is not None and loaded.mask_bbox is not None
        scene = np.full((720, 1280, 3), 50, dtype=np.uint8)
        # A light face, dark hair: the colour match (fitted on the face
        # alone) pulls the face to the scene's own grey and takes the hair
        # the same way -- so the hair lands well under the scene's 50.
        face = np.full((CROP_SIZE, CROP_SIZE, 3), 255, dtype=np.uint8)
        engine = LivePortraitEngine()
        loaded.hair_matte = self._hair_above_the_face(loaded)
        loaded.hair_box = hair_box(loaded.hair_matte)
        face[loaded.hair_matte[:, :, 0] > 0.5] = 0

        loaded.hair_on = False
        plain = engine._paste(scene, face, to_frame, loaded, loaded.mask)
        loaded.correction = None
        loaded.correction = None
        off = engine._paste(scene, face, to_frame, loaded, loaded.mask)
        assert np.array_equal(off, plain)

        loaded.hair_on = True
        loaded.correction = None
        on = engine._paste(scene, face, to_frame, loaded, loaded.mask)
        fx0, fy0, fx1, _ = _transformed_box(to_frame, loaded.mask_bbox, 1280, 720)
        # Above the face box: untouched without the hair, painted with it.
        above = (slice(max(fy0 - 30, 0), fy0 - 5), slice(fx0 + 40, fx1 - 40))
        assert np.array_equal(plain[above], scene[above])
        assert not np.array_equal(on[above], scene[above])
        assert float(np.abs(on[above].astype(int) - scene[above].astype(int)).mean()) > 20.0
        # Outside the union of the two boxes: byte-identical either way.
        hx0, hy0, hx1, hy1 = _transformed_box(
            to_frame, union_box(loaded.mask_bbox, loaded.hair_box), 1280, 720
        )
        outside = np.ones((720, 1280), dtype=bool)
        outside[hy0:hy1, hx0:hx1] = False
        assert np.array_equal(on[outside], scene[outside])

    def test_the_hair_is_not_pasted_without_a_matte(self, tmp_path):
        loaded, to_frame = self._loaded(tmp_path)
        assert loaded.mask is not None
        scene = np.full((720, 1280, 3), 50, dtype=np.uint8)
        face = np.full((CROP_SIZE, CROP_SIZE, 3), 200, dtype=np.uint8)
        engine = LivePortraitEngine()
        plain = engine._paste(scene, face, to_frame, loaded, loaded.mask)
        loaded.hair_on = True
        loaded.hair_matte = None
        loaded.correction = None
        assert np.array_equal(engine._paste(scene, face, to_frame, loaded, loaded.mask), plain)


class _FakeSegmenter:
    """Hair on the top third; ``gate`` cleared holds the worker inside
    ``segment`` so a test can say exactly when a matte arrives."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = value
        self.calls = 0
        self.model_path = "fake"
        self.gate = threading.Event()
        self.gate.set()

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        self.gate.wait()
        self.calls += 1
        height = rgb.shape[0]
        conf = np.zeros((height, rgb.shape[1], 2), dtype=np.float32)
        conf[: height // 3, :, 0] = self.value  # the top third is hair
        conf[height // 3 : 2 * height // 3, :, 1] = self.value  # the middle third is skin
        return conf

    def stop(self) -> None:
        pass


class TestEngineSwitch:
    def _fixture(
        self, tmp_path: Path, torch: Any
    ) -> tuple[LivePortraitEngine, _Portrait, _FakeSegmenter]:
        engine, loaded = _RenderFixture()._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        pts = face_points()
        to_crop, _ = crop_matrix(pts)
        loaded.crop_points = (pts @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
        loaded.mask = _oval_mask(loaded.crop_points)
        loaded.mask_bbox = _mask_bbox(loaded.mask)
        loaded.portrait_crop = np.full((CROP_SIZE, CROP_SIZE, 3), 120, dtype=np.uint8)
        engine._loaded[loaded.descriptor.id] = loaded
        fake = _FakeSegmenter()

        def start_hair() -> bool:
            if engine._hair_worker is None:
                engine._hair = fake
                engine._hair_worker = HairMatteWorker(fake.segment)
            return True

        engine._start_hair = start_hair
        return engine, loaded, fake

    def test_on_reads_the_portraits_matte_and_lifts_the_forehead(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded, fake = self._fixture(tmp_path, torch)
        avatar_id = loaded.descriptor.id
        assert not engine.hair_enabled(avatar_id)
        before = loaded.forehead_reach
        try:
            assert engine.set_hair(avatar_id, True)
            assert engine.hair_enabled(avatar_id) and loaded.hair_on
            assert fake.calls == 1
            assert loaded.hair_matte is not None and loaded.hair_matte.shape == (
                CROP_SIZE,
                CROP_SIZE,
                1,
            )
            assert loaded.hair_box is not None and loaded.hair_box[1] == 0
            assert loaded.hair_skin is None  # the skin lever is off by default
            # The reach from the avatar's own matte (the fake's top third),
            # overlapped -- see _matte_hairline.
            hairline = engine._matte_hairline(
                fake.segment(loaded.portrait_crop)[:, :, 0], loaded.crop_points
            )
            assert hairline is not None
            assert loaded.forehead_reach == pytest.approx(
                min(max(hairline + _HAIR_FOREHEAD_OVERLAP, 0.3), _FOREHEAD_REACH_MAX)
            )
            assert loaded.hairline_settled
            assert engine.frame_readings(avatar_id)["hair"] == 1.0

            assert not engine.set_hair(avatar_id, False)
            assert not loaded.hair_on and loaded.hair_matte is None and loaded.hair_box is None
            assert loaded.forehead_reach == before
            assert engine.frame_readings(avatar_id)["hair"] == 0.0
        finally:
            engine._stop_hair()

    def test_the_wish_is_kept_for_an_avatar_not_yet_loaded(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, _, _ = self._fixture(tmp_path, torch)
        try:
            assert engine.set_hair("someone_else", True)
            assert engine._hair_wanted == {"someone_else": True}
            assert not engine.hair_enabled("someone_else")
        finally:
            engine._stop_hair()

    def test_a_new_matte_is_folded_in_and_the_box_only_grows(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded, fake = self._fixture(tmp_path, torch)
        avatar_id = loaded.descriptor.id
        try:
            assert engine.set_hair(avatar_id, True)
            assert loaded.hair_matte is not None and engine._hair_worker is not None
            first = loaded.hair_matte.copy()
            box = loaded.hair_box
            fake.value = 0.0  # the segmenter now sees no hair at all
            fake.gate.clear()
            render = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
            engine._update_hair(loaded, render)  # submitted; the worker is held
            assert loaded.hair_seq == 0 and np.array_equal(loaded.hair_matte, first)
            fake.gate.set()
            assert _wait_for(lambda: engine._hair_worker.latest() is not None)
            fake.gate.clear()
            engine._update_hair(loaded, render)  # the first matte folded, the next held
            assert loaded.hair_seq == 1
            # Half-way between the portrait's matte and an empty one.
            assert loaded.hair_matte[10, 256, 0] == pytest.approx(first[10, 256, 0] * 0.5)
            assert loaded.hair_box == box
            fake.gate.set()
        finally:
            engine._stop_hair()

    def test_the_skin_lever_joins_the_renders_skin_above_the_eyes(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded, _ = self._fixture(tmp_path, torch)
        loaded.options = {"hair_skin": True}
        try:
            assert engine.set_hair(loaded.descriptor.id, True)
            assert loaded.hair_skin is not None and loaded.hair_matte is not None
            # The fake's skin is the middle third; only the rows above the
            # eyes take it, so the matte reaches below the hair's own third.
            assert loaded.hair_matte[CROP_SIZE // 3 + 8, 256, 0] > 0.0
        finally:
            engine._stop_hair()

    def test_a_segmenter_that_cannot_start_leaves_the_hair_off(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded, _ = self._fixture(tmp_path, torch)
        engine._start_hair = lambda: False
        assert not engine.set_hair(loaded.descriptor.id, True)
        assert not loaded.hair_on and loaded.hair_matte is None

    def test_reset_anchor_keeps_the_hairs_reach(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded, _ = self._fixture(tmp_path, torch)
        avatar_id = loaded.descriptor.id
        try:
            assert engine.set_hair(avatar_id, True)
            engine.reset_anchor(avatar_id)
            assert loaded.hairline_settled and loaded.hair_on
        finally:
            engine._stop_hair()
