"""Tests for the LivePortrait engine.

Three kinds. The crop arithmetic is pure numpy and is tested exactly: the crop
and its inverse must round-trip, the eye-to-mouth axis must come out vertical
in the crop (roll removed), and the box must be the size the scale says.
``_drive``'s control flow (which of the five networks run, with what) is
tested on the CPU with fakes standing in for the networks -- real weights are
not needed to ask "was the stitching network called". Everything that needs
the real weights or a GPU is marked ``requires_gpu`` and skipped without them.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sha256.avatar.engine_liveportrait import (
    _BROW_DEADBAND,
    _BROW_GATE,
    _BROW_GATE_PITCH,
    _BROW_GATE_YAW,
    _BROW_LIFT_Y,
    _BROW_MOUTH_LEAK,
    _CROP_CENTRES,
    _DEFAULT_CROP_SMOOTHING_HZ,
    _DEFAULT_FRAME_DT,
    _EDGE_FEATURES,
    _GATE_ATTACK,
    _GATE_RELEASE,
    _MASK_EDGE_CLEARANCE,
    _MOUTH_ACTIVITY_GATE,
    _MOUTH_ACTIVITY_MAR,
    _MOUTH_ACTIVITY_WIDTH,
    _MOUTH_GATE_MAR,
    _MOUTH_GATE_WIDTH,
    _MOUTH_GATE_YAW,
    _MOUTH_HOLD_FRAMES,
    _MOUTH_LATERAL,
    _MOUTH_OPENING,
    _MOUTH_SMILE,
    _RIGID_POINTS,
    _ROTATION_PIVOTS,
    _WIDTH_PER_DEGREE_YAW,
    BROW_KEYPOINTS,
    CROP_SCALE,
    CROP_SIZE,
    CROP_VY_RATIO,
    ENGINE_ID,
    EYE_KEYPOINTS,
    FACE_POINTS,
    MODEL_SIZE,
    MOUTH_KEYPOINTS,
    NUM_KP,
    LivePortraitEngine,
    _anchor_frame_ok,
    _brow_follow,
    _brow_gain,
    _brow_gate,
    _brow_lift,
    _brow_mouth_leak,
    _brow_ratio,
    _centre_offset,
    _chin_drop,
    _chin_follow,
    _crop_border_mode,
    _crop_centre,
    _crop_filters,
    _crop_geometry,
    _edge_match,
    _edge_residual,
    _ema_,
    _expression_gates,
    _eye_expression_gain,
    _eye_headroom,
    _eye_mask_output,
    _eye_ratios,
    _eye_scale,
    _eye_target,
    _face_height,
    _fade,
    _feature_exclusion,
    _fit_lighting_plane,
    _forehead_reach,
    _frame_dt,
    _frame_mask,
    _gaze_down_relief,
    _gaze_gain,
    _gaze_offset,
    _hold_thresholds,
    _lip_ratio,
    _lip_rest_tracking,
    _lip_target,
    _mask_bbox,
    _mask_chin_reach,
    _mask_jaw_shrink,
    _motion_smoothing,
    _mouth_activity,
    _mouth_axis_gain,
    _mouth_gate,
    _mouth_open_ratio,
    _mouth_rest,
    _mouth_tilt_trim,
    _mouth_width_ratio,
    _nose_chin,
    _OneEuro,
    _oval_mask,
    _pivot_z,
    _Portrait,
    _region_masks,
    _release,
    _relieved_ear,
    _rotation_pivot,
    _scale_gain,
    _should_hold,
    _similarity,
    _size_smoothing,
    _soft_gate,
    _transformed_box,
    _translation_gain,
    crop_matrix,
)
from sha256.avatar.engine_liveportrait import (
    _CHIN as _CHIN_INDEX,
)
from sha256.core.avatar import AvatarDescriptor, AvatarManifest
from sha256.core.enums import PixelFormat
from sha256.core.errors import AvatarLoadError
from sha256.core.interfaces import AvatarEngine
from sha256.core.types import (
    FaceTrackingResult,
    FrameMetadata,
    HeadPose,
    LandmarkSet,
    VideoFrame,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def face_points(*, centre=(640.0, 360.0), width=300.0, roll_degrees=0.0) -> np.ndarray:
    """A synthetic face: 468 points on an ellipse, with the eye corners, the
    mouth corners and the rigid crop-sizing points (see ``_RIGID_POINTS``) at
    the places the crop reads them from, rolled as asked.

    The rigid points are placed on purpose, not scattered with the rest: face
    sides at +-0.5 width, forehead top and nose tip 0.5 width apart, the nose
    bridge between them -- a deliberate box the crop-sizing tests can predict
    the extent of exactly, independent of the random cloud around it.
    """
    rng = np.random.default_rng(1)
    angles = rng.uniform(0.0, 2.0 * math.pi, FACE_POINTS)
    radii = np.sqrt(rng.uniform(0.0, 1.0, FACE_POINTS))
    pts = np.stack([np.cos(angles) * radii * width / 2, np.sin(angles) * radii * width * 0.65], 1)
    # anchors: eyes a quarter up, mouth a third down, corners spread
    pts[33], pts[133] = (-0.35 * width, -0.15 * width), (-0.12 * width, -0.15 * width)
    pts[362], pts[263] = (0.12 * width, -0.15 * width), (0.35 * width, -0.15 * width)
    pts[61], pts[291] = (-0.2 * width, 0.28 * width), (0.2 * width, 0.28 * width)
    # the rigid points: face sides, forehead top, nose bridge and tip
    pts[234], pts[454] = (-0.5 * width, 0.0), (0.5 * width, 0.0)
    pts[10] = (0.0, -0.45 * width)
    pts[1] = (0.0, 0.05 * width)
    for index, y in zip((168, 6, 197, 195, 5, 4), np.linspace(-0.4, 0.0, 6) * width, strict=True):
        pts[index] = (0.0, y)
    theta = math.radians(roll_degrees)
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    return (pts @ rot.T + centre).astype(np.float32)


def long_face_points(*, width: float = 300.0) -> np.ndarray:
    """:func:`face_points` with its random cloud held inside the rigid box
    (|y| <= 0.45 widths) and the chin (152) dropped to 0.65 widths below the
    centre, so the face height is known exactly: 1.1 widths, forehead top at
    -0.45 to the chin at +0.65 -- taller than the 1.0 the face is wide, the
    shape of face (the owner's) a width-governed crop under-sized."""
    pts = face_points(width=width)
    pts[:, 1] = np.clip(pts[:, 1], 360.0 - 0.45 * width, 360.0 + 0.45 * width)
    pts[152] = (640.0, 360.0 + 0.65 * width)
    return pts


class TestCrop:
    def test_the_crop_and_its_inverse_round_trip(self):
        pts = face_points(roll_degrees=12.0)
        to_crop, to_frame = crop_matrix(pts)

        moved = pts @ to_crop[:, :2].T + to_crop[:, 2]
        back = moved @ to_frame[:, :2].T + to_frame[:, 2]

        assert np.abs(back - pts).max() < 1e-2

    def test_the_eye_to_mouth_axis_is_vertical_in_the_crop(self):
        for roll in (0.0, 12.0, -25.0):
            pts = face_points(roll_degrees=roll)
            to_crop, _ = crop_matrix(pts)
            moved = pts @ to_crop[:, :2].T + to_crop[:, 2]
            eyes = (moved[[33, 133]].mean(0) + moved[[362, 263]].mean(0)) / 2
            mouth = moved[[61, 291]].mean(0)
            assert abs(eyes[0] - mouth[0]) < 1e-2, f"roll {roll}: axis is not vertical"
            assert mouth[1] > eyes[1], "the mouth is below the eyes"

    def test_the_face_fills_the_share_the_scale_says(self):
        """With no ``face_height`` given, the whole mesh's height fills
        ``CROP_SIZE / CROP_SCALE`` -- LivePortrait's own framing, which is
        what a still portrait is cut with at load. The rigid points only
        place the box; :func:`_face_height` sizes it."""
        pts = long_face_points()
        to_crop, _ = crop_matrix(pts)
        moved = pts @ to_crop[:, :2].T + to_crop[:, 2]
        extent = float((moved.max(0) - moved.min(0)).max())
        assert extent == pytest.approx(CROP_SIZE / CROP_SCALE, rel=1e-3)

    def test_the_face_height_is_forehead_top_to_chin(self):
        """``long_face_points``: 1.1 widths tall, along the face's vertical --
        the height, whatever the face's width (1.0 here) is doing."""
        assert _face_height(long_face_points()) == pytest.approx(1.1 * 300.0, rel=1e-3)
        assert _face_height(long_face_points(width=200.0)) == pytest.approx(220.0, rel=1e-3)

    def test_a_given_face_height_ignores_the_width(self):
        """A yaw narrows the face's 2-D width by cos(yaw); sized from a given
        height the crop does not care. (Day 5's width-governed rigid box
        zoomed in on every turn: the avatar's lag on a turn went 19 → 34 px.)"""
        pts = long_face_points()
        height = _face_height(pts)
        turned = pts.copy()
        turned[:, 0] = 640.0 + (turned[:, 0] - 640.0) * math.cos(math.radians(25.0))

        to_crop_front, _ = crop_matrix(pts, face_height=height)
        to_crop_turned, _ = crop_matrix(turned, face_height=height)

        assert to_crop_turned[0, 0] == pytest.approx(to_crop_front[0, 0], rel=1e-4)

    def test_a_given_face_height_equal_to_the_faces_own_changes_nothing(self):
        """Passing the height measured on a face gives that face's own crop
        -- which is why the anchor's frames (cut from themselves) and the
        frames after the freeze (cut from the resting height) match at rest."""
        pts = long_face_points()
        own, _ = crop_matrix(pts)
        given, _ = crop_matrix(pts, face_height=_face_height(pts))
        np.testing.assert_allclose(given, own, rtol=1e-5, atol=1e-3)

    def test_a_crop_is_centred_left_to_right(self):
        pts = face_points()
        to_crop, _ = crop_matrix(pts)
        moved = pts[list(_RIGID_POINTS)] @ to_crop[:, :2].T + to_crop[:, 2]
        centre_x = (moved[:, 0].max() + moved[:, 0].min()) / 2
        assert centre_x == pytest.approx(CROP_SIZE / 2, abs=1e-2)

    def test_an_open_mouth_does_not_change_the_crop_scale(self):
        """The whole point of sizing from rigid points: a mouth corner (and
        the chin dropping with it, the same "mouth wide open" gesture) is
        free to move without the crop's scale factor -- ``to_crop[0, 0]`` --
        changing at all, *given* the height learned at rest. Measured on the
        owner's own clip before this: the avatar's apparent size reached
        1.25x his at the widest mouth opening in ``driving.mkv``."""
        pts = long_face_points()
        height = _face_height(pts)
        moved = pts.copy()
        moved[61, 1] += 0.2 * 300.0
        moved[291, 1] += 0.2 * 300.0
        moved[152, 1] += 0.2 * 300.0

        to_crop_before, _ = crop_matrix(pts, face_height=height)
        to_crop_after, _ = crop_matrix(moved, face_height=height)

        assert to_crop_after[0, 0] == pytest.approx(to_crop_before[0, 0], abs=1e-3)

    def test_without_a_given_height_the_open_mouth_would_zoom(self):
        """Why the engine passes the resting height rather than reading it
        per frame: read per frame, it grows with the chin and the crop zooms
        out -- exactly the day-5 zoom complaint."""
        pts = long_face_points()
        moved = pts.copy()
        moved[152, 1] += 0.2 * 300.0

        to_crop_before, _ = crop_matrix(pts)
        to_crop_after, _ = crop_matrix(moved)

        assert to_crop_after[0, 0] == pytest.approx(to_crop_before[0, 0] * 1.1 / 1.3, rel=1e-3)


class TestCropCentre:
    """``crop_centre`` (day 11): the crop's vertical centre on the eye line
    instead of the rigid points' bounding box -- see ``_crop_geometry``'s
    ``centre_offset`` for the brow-raise measurement behind it."""

    def test_default_and_validation(self):
        assert _crop_centre({}) == "rigid"
        for name in _CROP_CENTRES:
            assert _crop_centre({"crop_centre": name}) == name
        with pytest.raises(ValueError, match="crop_centre"):
            _crop_centre({"crop_centre": "nose"})

    def test_the_offset_reproduces_the_rigid_centre_on_the_same_points(self):
        """Measured on the portrait, added back on the portrait: the two
        rules give the same crop, so the anchor's framing is untouched."""
        pts = face_points(roll_degrees=8.0)
        rigid = _crop_geometry(pts)
        eyeline = _crop_geometry(pts, centre_offset=_centre_offset(pts))
        assert eyeline[0] == pytest.approx(rigid[0], abs=1e-3)
        assert eyeline[1] == pytest.approx(rigid[1], abs=1e-3)
        assert eyeline[2] == rigid[2]
        assert eyeline[3] == rigid[3]

    def test_the_nose_moving_drags_the_rigid_centre_but_not_the_eyeline(self):
        """The rigid box's vertical centre is the midpoint of the forehead
        point and the lowest nose point; the eye line does not know the
        nose exists -- the brow-raise case, where his nose points moved
        7-40 px and his eye corners far less."""
        pts = face_points()
        offset = _centre_offset(pts)
        nosed = pts.copy()
        for index in (1, 168, 6, 197, 195, 5, 4):
            nosed[index, 1] += 20.0
        rigid_before, rigid_after = _crop_geometry(pts), _crop_geometry(nosed)
        eye_before = _crop_geometry(pts, centre_offset=offset)
        eye_after = _crop_geometry(nosed, centre_offset=offset)
        assert rigid_after[1] - rigid_before[1] > 5.0
        assert abs(eye_after[1] - eye_before[1]) < 0.5

    def test_the_offset_is_learned_with_the_anchor_and_forgotten_by_reset(self, tmp_path):
        """Collected on accepted anchor frames, frozen as the median at the
        freeze (so at rest the eye-line crop is the rigid one), cleared by
        ``reset_anchor`` with the rest of the anchor."""
        torch = pytest.importorskip("torch")
        engine, loaded = TestAnchorGatingInRender()._engine_and_loaded(
            tmp_path, torch=torch, anchor_wanted=3
        )
        image = torch.zeros(1, 3, MODEL_SIZE, MODEL_SIZE)
        engine._learn_anchor(loaded, image, 0.3, 0.0, centre_sample=(0.10, -1.0))
        engine._learn_anchor(loaded, image, 0.3, 0.0, centre_sample=(0.30, -3.0))
        assert loaded.centre_offset is None  # still collecting
        engine._learn_anchor(loaded, image, 0.3, 0.0, centre_sample=(0.12, -2.0))
        assert loaded.centre_offset == pytest.approx(0.12)  # median, not mean 0.173
        assert loaded.rest_pitch == pytest.approx(-2.0)
        # The weight: rigid at rest, the eye line once he has nodded past
        # the fade (_CROP_CENTRE_PITCH), half way between in between.
        weight = LivePortraitEngine._centre_weight
        assert weight(loaded, HeadPose(pitch=-2.0, yaw=0.0, roll=0.0)) == 0.0
        assert weight(loaded, HeadPose(pitch=-4.0, yaw=0.0, roll=0.0)) == 0.0
        assert weight(loaded, HeadPose(pitch=-7.5, yaw=0.0, roll=0.0)) == pytest.approx(0.5)
        assert weight(loaded, HeadPose(pitch=-14.0, yaw=0.0, roll=0.0)) == 1.0
        assert weight(loaded, None) == 0.0
        engine.reset_anchor(loaded.descriptor.id)
        assert loaded.centre_offset is None
        assert loaded.rest_pitch is None
        assert loaded.centre_samples == []

    def test_the_eye_corners_moving_carry_the_eyeline_centre_with_them(self):
        pts = face_points()
        offset = _centre_offset(pts)
        lifted = pts.copy()
        for index in (33, 133, 362, 263):
            lifted[index, 1] -= 10.0
        eye_before = _crop_geometry(pts, centre_offset=offset)
        eye_after = _crop_geometry(lifted, centre_offset=offset)
        assert eye_after[1] - eye_before[1] == pytest.approx(-10.0, abs=0.5)


class TestMask:
    """The paste mask: the portrait's own oval, and the box it is warped in."""

    def test_the_mask_bbox_is_far_smaller_than_the_whole_crop(self):
        """This is the whole reason the mask is precomputed at load: the box
        the paste warps into should be a small fraction of the 512 square,
        not the square itself."""
        pts = face_points()
        to_crop, _ = crop_matrix(pts)
        crop_points = (pts @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)

        mask = _oval_mask(crop_points)
        bbox = _mask_bbox(mask)

        assert mask.shape == (CROP_SIZE, CROP_SIZE, 1)
        x0, y0, x1, y1 = bbox
        area_fraction = (x1 - x0) * (y1 - y0) / float(CROP_SIZE * CROP_SIZE)
        assert 0.0 < area_fraction < 0.35, f"mask box covers {area_fraction:.2%} of the crop"

    def test_the_mask_is_feathered_not_binary(self):
        pts = face_points()
        to_crop, _ = crop_matrix(pts)
        crop_points = (pts @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
        mask = _oval_mask(crop_points)

        values = mask[mask[:, :, 0] > 0]
        # A feathered edge means some pixels sit strictly between 0 and 1, not
        # only at the two ends -- a hard-edged mask would fail this.
        between = ((values > 0.02) & (values < 0.98)).sum()
        assert between > 0

    def test_a_jaw_shrink_pulls_the_jaw_in_and_leaves_the_forehead_alone(self):
        """Day 12, the jaw band: with a jaw shrink the mask's bottom rises and
        its sides at mouth height move in, each by about the asked fraction
        of face width; the top row does not move, and at the eye line the
        narrowing is a fraction of the jaw's. Zero is the day-11 mask
        exactly."""
        pts = face_points()
        to_crop, _ = crop_matrix(pts)
        crop_points = (pts @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
        face_width = float(crop_points[:, 0].max() - crop_points[:, 0].min())

        plain = _oval_mask(crop_points)[:, :, 0] > 0.5
        shrunk = _oval_mask(crop_points, jaw_shrink=0.08)[:, :, 0] > 0.5

        def rows(mask: np.ndarray) -> tuple[int, int]:
            filled = np.where(mask.any(axis=1))[0]
            return int(filled.min()), int(filled.max())

        def width_at(mask: np.ndarray, row: int) -> int:
            xs = np.where(mask[row])[0]
            return int(xs.max() - xs.min()) if len(xs) else 0

        assert rows(shrunk)[0] == rows(plain)[0]
        rise = (rows(plain)[1] - rows(shrunk)[1]) / face_width
        assert 0.05 < rise < 0.11, f"the chin rose by {rise:.3f} of face width"

        mouth_row = int(crop_points[61, 1])
        narrowing = (width_at(plain, mouth_row) - width_at(shrunk, mouth_row)) / face_width
        assert 0.12 < narrowing < 0.24, f"the jaw narrowed by {narrowing:.3f} of face width"

        eye_row = int(crop_points[33, 1])
        at_eyes = (width_at(plain, eye_row) - width_at(shrunk, eye_row)) / face_width
        assert at_eyes < narrowing / 2.5, f"the eye line narrowed by {at_eyes:.3f}"

        assert np.array_equal(_oval_mask(crop_points, jaw_shrink=0.0), _oval_mask(crop_points))


class TestMaskJawShrink:
    """``mask_jaw_shrink``: the option that sets the day-12 jaw shrink."""

    def test_default_and_override(self):
        assert _mask_jaw_shrink({}) == 0.0  # the join at the contour; 0.08 is the bearded setting
        assert _mask_jaw_shrink({"mask_jaw_shrink": 0}) == 0.0
        assert _mask_jaw_shrink({"mask_jaw_shrink": 0.12}) == 0.12

    def test_out_of_range_is_refused(self):
        with pytest.raises(ValueError, match="mask_jaw_shrink"):
            _mask_jaw_shrink({"mask_jaw_shrink": 0.5})
        with pytest.raises(ValueError, match="mask_jaw_shrink"):
            _mask_jaw_shrink({"mask_jaw_shrink": -0.01})


class TestFrameMask:
    """``_frame_mask``: the portrait's oval and the driver's, intersected."""

    def _portrait_mask_and_points(self) -> tuple[np.ndarray, np.ndarray]:
        pts = face_points()
        to_crop, _ = crop_matrix(pts)
        crop_points = (pts @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
        return _oval_mask(crop_points), crop_points

    def test_never_exceeds_either_half(self):
        """The combined mask can only be as large as each half alone -- it is
        the whole reason the intersection can never show the portrait's own
        background or the driver's own hair/neck."""
        portrait_mask, portrait_points = self._portrait_mask_and_points()
        centre_y = float(portrait_points[:, 1].mean())
        driver_points = portrait_points.copy()
        driver_points[:, 1] = centre_y + (driver_points[:, 1] - centre_y) * 0.75
        driver_mask = _oval_mask(driver_points)

        combined = _frame_mask(portrait_mask, driver_points)

        assert np.all(combined <= portrait_mask + 1e-6)
        assert np.all(combined <= driver_mask + 1e-6)

    def test_a_driver_matching_the_portrait_leaves_the_mask_unchanged(self):
        """The everyday case: the combined mask equals the static one exactly
        when the driver's own oval is the same shape as the portrait's."""
        portrait_mask, portrait_points = self._portrait_mask_and_points()

        combined = _frame_mask(portrait_mask, portrait_points)

        assert np.array_equal(combined, portrait_mask)

    def test_the_jaw_shrink_goes_to_the_driver_half_alone(self):
        """Day 12: the same driver as the portrait, with a jaw shrink -- the
        combined mask is the driver's pulled-in oval at the jaw and the
        portrait's own everywhere the shrink does not reach (the top row)."""
        portrait_mask, portrait_points = self._portrait_mask_and_points()

        combined = _frame_mask(portrait_mask, portrait_points, jaw_shrink=0.08)

        expected = np.minimum(portrait_mask, _oval_mask(portrait_points, jaw_shrink=0.08))
        assert np.array_equal(combined, expected)
        assert np.all(combined <= portrait_mask + 1e-6)
        top = [
            int(np.where((m[:, :, 0] > 0.5).any(axis=1))[0].min())
            for m in (combined, portrait_mask)
        ]
        assert top[0] == top[1]
        assert (combined < portrait_mask - 0.5).any()

    def test_a_shorter_driver_forehead_pulls_the_ceiling_down(self):
        """The hairline defect, directly: a driver whose face is shorter than
        the portrait's (forehead-to-chin compressed toward the centre) must
        not let the combined mask reach as high as the portrait's own oval
        does -- reaching that high would be painting the portrait's forehead
        tone onto the driver's hair."""
        portrait_mask, portrait_points = self._portrait_mask_and_points()
        centre_y = float(portrait_points[:, 1].mean())
        driver_points = portrait_points.copy()
        driver_points[:, 1] = centre_y + (driver_points[:, 1] - centre_y) * 0.75

        combined = _frame_mask(portrait_mask, driver_points)

        def first_row_over_half(mask: np.ndarray) -> int:
            rows = np.where((mask[:, :, 0] > 0.5).any(axis=1))[0]
            return int(rows.min())

        assert first_row_over_half(combined) > first_row_over_half(portrait_mask)

    @staticmethod
    def _rows_over_half(mask: np.ndarray) -> tuple[int, int]:
        rows = np.where((mask[:, :, 0] > 0.5).any(axis=1))[0]
        return int(rows.min()), int(rows.max())

    def test_a_chin_drop_lets_the_oval_down_at_the_chin_and_leaves_the_top_alone(self):
        """Day 13: the portrait's oval with its jaw let down 30 px reaches
        30 px lower at the bottom and no higher at the top; at the eye
        line, well above where the ramp starts, it is the same mask."""
        portrait_mask, portrait_points = self._portrait_mask_and_points()

        dropped = _oval_mask(portrait_points, chin_drop=30.0)

        top0, bottom0 = self._rows_over_half(portrait_mask)
        top1, bottom1 = self._rows_over_half(dropped)
        assert top1 == top0
        assert bottom1 == pytest.approx(bottom0 + 30, abs=2)
        eye_y = int(portrait_points[[33, 133, 362, 263], 1].mean())
        assert np.array_equal(portrait_mask[eye_y], dropped[eye_y])

    def test_the_chin_drop_goes_to_the_portrait_half_alone(self):
        """Day 13, the complaint itself: the same driver as the portrait
        with his mouth open (every point below the mouth line 30 px lower).
        Without the drop the combined mask stops at the portrait's own
        closed-mouth chin, his jaw showing under it; with it, the mask
        follows his jaw down -- and still never past his own oval."""
        portrait_mask, portrait_points = self._portrait_mask_and_points()
        driver_points = portrait_points.copy()
        mouth_y = float(portrait_points[[61, 291], 1].mean())
        driver_points[driver_points[:, 1] > mouth_y, 1] += 30.0

        closed = _frame_mask(portrait_mask, driver_points)
        opened = _frame_mask(
            portrait_mask, driver_points, portrait_crop_points=portrait_points, chin_drop=30.0
        )

        assert self._rows_over_half(closed)[1] == pytest.approx(
            self._rows_over_half(portrait_mask)[1], abs=1
        )
        assert self._rows_over_half(opened)[1] >= self._rows_over_half(closed)[1] + 25
        assert self._rows_over_half(opened)[0] == self._rows_over_half(closed)[0]
        assert np.all(opened <= _oval_mask(driver_points) + 1e-6)

    def test_a_drop_under_half_a_pixel_or_without_the_portrait_points_is_the_static_mask(self):
        portrait_mask, portrait_points = self._portrait_mask_and_points()

        tiny = _frame_mask(
            portrait_mask, portrait_points, portrait_crop_points=portrait_points, chin_drop=0.4
        )
        no_points = _frame_mask(portrait_mask, portrait_points, chin_drop=30.0)

        assert np.array_equal(tiny, portrait_mask)
        assert np.array_equal(no_points, portrait_mask)


class TestForeheadLine:
    """Day 13, the forehead line: the ovals' reach up the forehead
    (``mask_forehead_reach``) and the plane's residual matched at the
    mask's edge (``edge_match`` / ``_edge_residual``)."""

    def test_forehead_reach_default_and_validation(self):
        assert _forehead_reach({}) == 0.85
        assert _forehead_reach({"mask_forehead_reach": 0.6}) == 0.6
        with pytest.raises(ValueError, match="mask_forehead_reach"):
            _forehead_reach({"mask_forehead_reach": 1.2})
        with pytest.raises(ValueError, match="mask_forehead_reach"):
            _forehead_reach({"mask_forehead_reach": 0.1})

    def test_a_lower_reach_stops_the_oval_lower_on_the_forehead_and_nowhere_else(self):
        pts = face_points()
        to_crop, _ = crop_matrix(pts)
        crop_points = (pts @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)

        high = _oval_mask(crop_points, forehead_reach=0.85)
        low = _oval_mask(crop_points, forehead_reach=0.55)

        def rows(mask):
            r = np.where((mask[:, :, 0] > 0.5).any(axis=1))[0]
            return int(r.min()), int(r.max())

        assert rows(low)[0] > rows(high)[0]
        assert rows(low)[1] == rows(high)[1]
        assert np.all(low <= high + 1e-6)

    def test_edge_match_default_and_validation(self):
        assert _edge_match({}) == 1.0
        assert _edge_match({"edge_match": 0.0}) == 0.0
        with pytest.raises(ValueError, match="edge_match"):
            _edge_match({"edge_match": 2.0})

    def test_a_difference_the_plane_fits_leaves_no_residual(self):
        """A uniform difference under a full mask is exactly a plane, so what
        is matched at the edge is nothing -- the plane alone, as before."""
        coverage = np.ones((40, 32), dtype=np.float32)
        difference = np.full((40, 32, 3), 12.0, dtype=np.float32)
        plane = _fit_lighting_plane(difference, coverage)

        residual = _edge_residual(difference, coverage, plane)

        assert np.abs(residual).max() < 1e-3

    def test_a_pale_top_edge_is_darkened_at_the_top_edge_and_left_alone_in_the_middle(self):
        """The forehead line itself: the scene darker than the face along
        the mask's top rows only. The plane spreads that over the whole
        face; the residual puts the rest back at the top edge and fades to
        nothing by the middle -- and the bottom edge, which the plane
        over-corrected the other way, gets its own sign back."""
        # The mask sits inside its box with a margin, as the paste box does
        # (the contour is where the distance transform finds its zeros).
        coverage = np.zeros((40, 32), dtype=np.float32)
        coverage[2:38, 2:30] = 1.0
        difference = np.zeros((40, 32, 3), dtype=np.float32)
        difference[2:5, 2:30] = -30.0  # the scene 30 levels darker than the face at the top
        plane = _fit_lighting_plane(difference, coverage)

        residual = _edge_residual(difference, coverage, plane)
        total = plane + residual
        top, middle, bottom = 2, 20, 37

        assert total[top, 2:30].mean() < plane[top, 2:30].mean() - 5.0  # darker still at the top
        assert abs(residual[middle - 2 : middle + 2, 2:30].mean()) < 1.0  # nothing in the middle
        assert total[top, 2:30].mean() < total[middle, 2:30].mean()
        # The bottom edge, which the plane pulled the wrong way, is put back
        # closer to its own (zero) difference.
        assert abs(total[bottom, 2:30].mean()) < abs(plane[bottom, 2:30].mean())

    @staticmethod
    def _features(left=(10.0, 12.0), right=(30.0, 12.0), mouth=(20.0, 30.0)) -> np.ndarray:
        """His brow/eye and mouth points in cell coordinates, each group a
        small cluster around its centre, in the ``_EDGE_FEATURES`` order."""
        rng = np.random.default_rng(3)
        groups = []
        for centre, group in zip((left, right, mouth), _EDGE_FEATURES, strict=True):
            groups.append(np.asarray(centre) + rng.uniform(-1.5, 1.5, (len(group), 2)))
        return np.concatenate(groups).astype(np.float32)

    def test_feature_exclusion_marks_his_brows_eyes_and_mouth_not_the_forehead(self):
        zone = _feature_exclusion(self._features(), (40, 40))

        assert zone[12, 10] == 1.0 and zone[12, 30] == 1.0 and zone[30, 20] == 1.0
        assert zone[9, 10] == 1.0  # dilated above the brow points, where the hair is
        assert zone[3, 20] == 0.0 and zone[20, 20] == 0.0  # the forehead, the nose

    def test_an_excluded_dark_feature_is_not_printed_into_the_residual(self):
        """His 2026-09-09 looks: a dark blob (his raised brow) just inside
        the top edge went into the residual and came out on the avatar's
        forehead; cut to the plane alone it came out as a dark patch. Cut
        out, the blob's cells read as the skin beside them at the same
        rows; the forehead line beside it is matched as before."""
        coverage = np.zeros((40, 40), dtype=np.float32)
        coverage[2:38, 2:38] = 1.0
        difference = np.zeros((40, 40, 3), dtype=np.float32)
        difference[2:5, 2:38] = -30.0  # the forehead line along the top edge
        difference[5:8, 9:15] = -80.0  # his brow, darker still, under it
        plane = _fit_lighting_plane(difference, coverage)
        exclude = np.zeros((40, 40), dtype=np.float32)
        exclude[5:8, 9:15] = 1.0

        plain = _edge_residual(difference, coverage, plane)
        cut = _edge_residual(difference, coverage, plane, exclude=exclude)

        assert plain[6, 12].mean() < cut[6, 12].mean() - 10.0  # the blob no longer prints
        assert abs(cut[6, 12].mean() - cut[6, 25].mean()) < 8.0  # it reads as the skin beside
        assert abs(cut[3, 30].mean() - plain[3, 30].mean()) < 1.0  # the line beside it as before


class TestChinDrop:
    """Day 13: his chin's drop from rest, in crop pixels (``_nose_chin`` /
    ``_chin_drop``), the mask box's reach below it, and the ``chin_follow``
    option -- see the module's "jaw under an open mouth" comment."""

    def test_nose_chin_is_the_reach_down_the_crop(self):
        pts = np.zeros((FACE_POINTS, 2), dtype=np.float32)
        pts[1] = (100.0, 200.0)
        pts[152] = (130.0, 260.0)
        assert _nose_chin(pts) == pytest.approx(60.0)

    def test_chin_drop_is_zero_without_a_rest_never_negative_and_capped(self):
        """Never negative -- see ``_chin_drop`` for the 2026-09-09 capture
        that measured a signed drop cutting the avatar's chin short."""
        assert _chin_drop(70.0, None, 50.0) == 0.0
        assert _chin_drop(55.0, 60.0, 50.0) == 0.0
        assert _chin_drop(75.0, 60.0, 50.0) == pytest.approx(15.0)
        assert _chin_drop(200.0, 60.0, 50.0) == pytest.approx(50.0)

    def test_the_drop_and_the_reach_share_the_crops_room_below_the_chin(self):
        """The cap that matters is the crop's own bottom row, not the
        fraction: a mask let down past it is clipped hard by the array's
        edge and pastes a straight line (see ``_CHIN_DROP_MAX``). The
        reach is taken first, the drop gets the rest, and neither can put
        the mask outside the crop."""
        pts = face_points()
        to_crop, _ = crop_matrix(pts)
        crop_points = (pts @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
        chin_row = float(crop_points[_CHIN_INDEX, 1])
        headroom = max(float(CROP_SIZE - 1) - chin_row, 0.0)
        face_height = _face_height(crop_points)

        reach = min(0.08 * face_height, headroom)
        drop_max = min(0.35 * face_height, headroom - reach)

        assert reach + drop_max <= headroom + 1e-6
        assert chin_row + reach + drop_max <= CROP_SIZE - 1 + 1e-6
        # And the mask built at the full budget still ends inside the crop.
        mask = _oval_mask(crop_points, chin_drop=reach + drop_max)
        assert int(np.where((mask[:, :, 0] > 0.0).any(axis=1))[0].max()) < CROP_SIZE - 1

    def test_the_hull_keeps_its_measured_tail_clear_of_the_last_row(self):
        """Day 14: the margin the hull keeps above the crop's last row is
        the mask's own feather tail (alpha 0.02 past the hull's chin) plus
        two rows, measured at load -- not the day-13 argument from kernel
        sizes, which came to four times as much. At that margin the last
        row stays clear; a hull on the last row is the hard line."""
        pts = face_points()
        to_crop, _ = crop_matrix(pts)
        crop_points = (pts @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
        chin_row = float(crop_points[_CHIN_INDEX, 1])
        rest = _oval_mask(crop_points)
        tail = float(np.where((rest[:, :, 0] > 0.02).any(axis=1))[0].max()) - chin_row
        assert -1.0 <= tail <= 6.0  # a few pixels, not the 15 the kernels suggested
        margin = max(tail, 0.0) + _MASK_EDGE_CLEARANCE

        clear = _oval_mask(crop_points, chin_drop=(CROP_SIZE - 1) - margin - chin_row)
        assert float(clear[CROP_SIZE - 1, :, 0].max()) < 0.01
        # A hull past the edge is clipped into a band the erosion cannot
        # reach from below: the hard line (12 rows past, so the band is
        # wider than the erosion kernel even on this ellipse's pointed chin).
        past_the_edge = _oval_mask(crop_points, chin_drop=(CROP_SIZE - 1) - chin_row + 12.0)
        assert float(past_the_edge[CROP_SIZE - 1, :, 0].max()) > 0.3

    def test_the_vy_ratio_moves_the_face_up_the_crop_by_its_share_of_the_crop(self):
        """Day 14: ``crop_vy_ratio`` decides how the crop's rows are spent
        between hair and neck -- the chin rises by the ratio's difference
        times the crop size, which is the room the open jaw gets. The
        default is -0.06 (see ``CROP_VY_RATIO``), 33 rows more under the
        chin than LivePortrait's own -0.125."""
        pts = face_points()

        def chin_row(vy: float) -> float:
            to_crop, _ = crop_matrix(pts, vy_ratio=vy)
            return float((pts @ to_crop[:, :2].T + to_crop[:, 2])[_CHIN_INDEX, 1])

        assert CROP_VY_RATIO == -0.06
        rise = chin_row(-0.125) - chin_row(CROP_VY_RATIO)
        assert rise == pytest.approx(0.065 * CROP_SIZE, abs=1.0)
        assert chin_row(0.0) - chin_row(CROP_VY_RATIO) == pytest.approx(-0.06 * CROP_SIZE, abs=1.0)

    def test_mask_chin_reach_default_and_validation(self):
        assert _mask_chin_reach({}) == 0.0
        assert _mask_chin_reach({"mask_chin_reach": 0.08}) == 0.08
        with pytest.raises(ValueError, match="mask_chin_reach"):
            _mask_chin_reach({"mask_chin_reach": 0.5})

    def test_a_chin_reach_lets_both_ovals_down_so_the_join_is_under_the_chin(self):
        """His 03:28 captures: the mask's contour sat above his chin and the
        feather faded the avatar out across it. With a reach, the combined
        mask (the portrait's half AND the driver's) ends that much lower;
        without one it is exactly the mask it was."""
        pts = face_points()
        to_crop, _ = crop_matrix(pts)
        crop_points = (pts @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
        portrait = _oval_mask(crop_points)

        plain = _frame_mask(portrait, crop_points, portrait_crop_points=crop_points)
        reached = _frame_mask(
            portrait, crop_points, portrait_crop_points=crop_points, chin_reach=24.0
        )

        def bottom(mask):
            return int(np.where((mask[:, :, 0] > 0.5).any(axis=1))[0].max())

        assert np.array_equal(plain, portrait)
        assert bottom(reached) == pytest.approx(bottom(plain) + 24, abs=2)

    def test_the_oval_itself_would_rise_for_a_negative_drop(self):
        pts = face_points()
        to_crop, _ = crop_matrix(pts)
        crop_points = (pts @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
        rest = _oval_mask(crop_points)
        risen = _oval_mask(crop_points, chin_drop=-30.0)

        def rows(mask):
            r = np.where((mask[:, :, 0] > 0.5).any(axis=1))[0]
            return int(r.min()), int(r.max())

        assert rows(risen)[0] == rows(rest)[0]
        assert rows(risen)[1] == pytest.approx(rows(rest)[1] - 30, abs=2)
        combined = _frame_mask(rest, crop_points, portrait_crop_points=crop_points, chin_drop=-30.0)
        assert rows(combined)[1] == rows(risen)[1]

    def test_the_mask_box_reaches_below_by_what_it_is_told(self):
        mask = np.zeros((512, 512, 1), dtype=np.float32)
        mask[100:200, 100:200] = 1.0
        assert _mask_bbox(mask) == (96, 96, 204, 204)
        assert _mask_bbox(mask, reach_below=40) == (96, 96, 204, 244)
        assert _mask_bbox(mask, reach_below=1000) == (96, 96, 204, 512)

    def test_chin_follow_default_and_validation(self):
        assert _chin_follow({}) == 1.0
        assert _chin_follow({"chin_follow": 0.0}) == 0.0
        with pytest.raises(ValueError, match="chin_follow"):
            _chin_follow({"chin_follow": 1.5})

    def test_the_anchor_learns_the_resting_reach(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = TestGatesInTheFramePath()._fixture(tmp_path, torch, anchor_wanted=2)
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
        assert loaded.rest_chin is None

        for reach in (118.0, 122.0):
            engine._render(
                loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=True, chin_sample=reach
            )

        assert loaded.anchor_seen == 2
        assert loaded.rest_chin == pytest.approx(120.0)
        engine.reset_anchor(loaded.descriptor.id)
        assert loaded.rest_chin is None


class TestAnchorSnapshot:
    """``anchor_state`` / ``seed_anchor`` (day 15): the frozen anchor as
    arrays a capture can carry, and an engine seeded from them running as
    the session that wrote them. On :class:`TestAnchorGatingInRender`'s
    CPU fixture (a faked motion net, a no-op graph)."""

    @staticmethod
    def _fresh(tmp_path, torch, anchor_wanted: int = 2):
        engine, loaded = TestAnchorGatingInRender()._engine_and_loaded(
            tmp_path, torch=torch, anchor_wanted=anchor_wanted
        )
        # The round trip is the subject here, not the lip rest (see
        # TestMouthRest): his own reading travels as it is.
        loaded.options = {"mouth_rest": "anchor"}
        return engine, loaded

    def _frozen(self, tmp_path, torch):
        engine, loaded = self._fresh(tmp_path, torch, anchor_wanted=2)
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
        samples = ((0.30, 0.01, 0.40, 0.20, 118.0), (0.32, 0.03, 0.42, 0.22, 122.0))
        for ear, mar, width, brow, chin in samples:
            engine._render(
                loaded,
                driver_rgb,
                ear,
                mar,
                ear,
                mar,
                anchor_ok=True,
                rest_sample=(width, brow),
                chin_sample=chin,
            )
        assert loaded.anchor_seen == 2
        # Values the fake motion net cannot produce, so the tensors are seen
        # to travel and not just their zeros.
        loaded.exp_d0.fill_(0.25)
        loaded.t_d0.copy_(torch.tensor([[0.1, -0.2, 0.0]]))
        loaded.scale_d0.fill_(1.05)
        loaded.pose_d0.copy_(torch.tensor([[3.0], [-2.0], [1.0]]))
        return engine, loaded

    def test_nothing_before_the_freeze_and_nothing_for_an_unknown_avatar(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._fresh(tmp_path, torch, anchor_wanted=2)
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
        assert engine.anchor_state(loaded.descriptor.id) is None
        assert engine.anchor_state("nobody") is None

        engine._render(loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=True)
        assert engine.anchor_state(loaded.descriptor.id) is None  # one of two
        engine._render(loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=True)
        assert engine.anchor_state(loaded.descriptor.id) is not None

    def test_the_state_is_plain_copies_with_the_rest_values_and_nan_for_unknowns(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._frozen(tmp_path, torch)

        state = engine.anchor_state(loaded.descriptor.id)

        assert state is not None
        assert all(isinstance(v, np.ndarray) for v in state.values())
        assert state["exp_d0"].shape == (1, NUM_KP, 3)
        assert float(state["exp_d0"].mean()) == pytest.approx(0.25)
        assert state["t_d0"].shape == (1, 3) and state["scale_d0"].shape == (1, 1)
        assert state["R_d0"].shape == (1, 3, 3) and state["pose_d0"].shape == (3, 1)
        assert float(state["rest_ear"]) == pytest.approx(0.31)
        assert float(state["rest_mar"]) == pytest.approx(0.02)
        assert float(state["rest_mouth_width"]) == pytest.approx(0.41)
        assert float(state["rest_brow"]) == pytest.approx(0.21)
        assert float(state["rest_chin"]) == pytest.approx(120.0)
        # No size or centre samples in this fixture: unknown, written as NaN.
        assert np.isnan(float(state["face_height_rest"]))
        assert np.isnan(float(state["z_rest"]))
        assert np.isnan(state["rest_gaze"]).all()
        # A copy: writing into it does not reach the engine's own tensor.
        state["exp_d0"][...] = 9.0
        assert float(loaded.exp_d0.mean()) == pytest.approx(0.25)

    def test_seeding_makes_a_fresh_engine_the_session_that_wrote_it(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._frozen(tmp_path, torch)
        state = engine.anchor_state(loaded.descriptor.id)
        fresh, fresh_loaded = self._fresh(tmp_path, torch, anchor_wanted=15)
        fresh_loaded.holding = True
        fresh_loaded.gate_mouth_prev = 0.7
        fresh_loaded.ear_recent.append(0.3)
        assert fresh.rest_status(fresh_loaded.descriptor.id) is not None

        fresh.seed_anchor(fresh_loaded.descriptor.id, state)

        assert fresh.rest_status(fresh_loaded.descriptor.id) is None
        assert fresh_loaded.anchor_seen == fresh_loaded.anchor_wanted == 15
        for name in ("exp_d0", "t_d0", "scale_d0", "R_d0", "pose_d0"):
            assert torch.equal(getattr(fresh_loaded, name), getattr(loaded, name)), name
        assert fresh_loaded.rest_ear == pytest.approx(0.31)
        assert fresh_loaded.rest_mar == pytest.approx(0.02)
        assert fresh_loaded.rest_mouth_width == pytest.approx(0.41)
        assert fresh_loaded.rest_brow == pytest.approx(0.21)
        assert fresh_loaded.rest_chin == pytest.approx(120.0)
        assert fresh_loaded.face_height_rest is None and fresh_loaded.z_rest is None
        assert fresh_loaded.rest_gaze is None
        # The smoothing state primed to the anchor as the freeze primes it;
        # the per-frame state cleared as reset_anchor clears it.
        assert torch.equal(fresh_loaded.exp_prev, loaded.exp_d0)
        assert torch.equal(fresh_loaded.t_prev, loaded.t_d0)
        assert torch.equal(fresh_loaded.scale_prev, loaded.scale_d0)
        assert torch.equal(fresh_loaded.pose_prev, loaded.pose_d0)
        assert fresh_loaded.holding is False
        assert fresh_loaded.gate_mouth_prev == 0.0
        assert len(fresh_loaded.ear_recent) == 0
        # And the seeded engine reports the same state back.
        again = fresh.anchor_state(fresh_loaded.descriptor.id)
        assert again is not None
        for key, value in state.items():
            np.testing.assert_allclose(again[key], value, equal_nan=True)

    def test_it_survives_the_file_a_capture_writes(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._frozen(tmp_path, torch)
        state = engine.anchor_state(loaded.descriptor.id)
        assert state is not None
        np.savez_compressed(
            tmp_path / "tracking.npz", **{f"anchor_{k}": v for k, v in state.items()}
        )
        data = np.load(tmp_path / "tracking.npz", allow_pickle=False)
        back = {k[len("anchor_") :]: data[k] for k in data.files}

        fresh, fresh_loaded = self._fresh(tmp_path, torch)
        fresh.seed_anchor(fresh_loaded.descriptor.id, back)

        assert torch.equal(fresh_loaded.exp_d0, loaded.exp_d0)
        assert torch.equal(fresh_loaded.R_d0, loaded.R_d0)
        assert fresh_loaded.rest_mar == pytest.approx(0.02)
        assert fresh_loaded.centre_offset is None

    def test_seeding_refuses_a_wrong_shape_a_missing_tensor_and_an_unknown_avatar(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._frozen(tmp_path, torch)
        state = engine.anchor_state(loaded.descriptor.id)
        assert state is not None

        with pytest.raises(AvatarLoadError):
            engine.seed_anchor("nobody", state)
        bad = dict(state)
        bad["exp_d0"] = np.zeros((1, 2, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="exp_d0"):
            engine.seed_anchor(loaded.descriptor.id, bad)
        missing = dict(state)
        del missing["R_d0"]
        with pytest.raises(ValueError, match="R_d0"):
            engine.seed_anchor(loaded.descriptor.id, missing)


class TestMouthRest:
    """``mouth_rest`` (day 15): the anchor's lip-opening components come from
    his own rest reading ("anchor", the default) or from the portrait's own
    closed mouth ("portrait") -- his pressed lips at a session's start had
    every opening 1.6x his and the portrait's relaxed ones put that frame at
    1.0x, but live "portrait" drooped a slight opening and pursed a closing,
    so it is an option, not the rule. On the CPU fixture with a faked
    motion net."""

    @staticmethod
    def _fixture(tmp_path, torch, mouth_rest: str | None):
        engine, loaded = TestAnchorGatingInRender()._engine_and_loaded(
            tmp_path, torch=torch, anchor_wanted=2
        )
        if mouth_rest is not None:
            loaded.options = {"mouth_rest": mouth_rest}
        # The portrait's own reading: a value on every mouth row the fake
        # motion net (all zeros) can never produce.
        loaded.exp_s = torch.zeros(1, NUM_KP, 3)
        for k, axis in _MOUTH_OPENING:
            loaded.exp_s[0, k, axis] = 0.05
        loaded.exp_s[0, 14, 1] = 0.02
        return engine, loaded

    def test_reads_and_validates(self):
        assert _mouth_rest({}) == "anchor"
        assert _mouth_rest({"mouth_rest": "portrait"}) == "portrait"
        with pytest.raises(ValueError, match="mouth_rest"):
            _mouth_rest({"mouth_rest": "clip"})

    def test_portrait_puts_the_portraits_lip_rest_into_the_anchor(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._fixture(tmp_path, torch, "portrait")
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)

        for _ in range(2):
            engine._render(loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=True)

        assert loaded.anchor_seen == 2
        for k, axis in _MOUTH_OPENING:
            assert loaded.exp_d0[0, k, axis] == pytest.approx(0.05), (k, axis)
        assert loaded.exp_d0[0, 14, 1] == 0.0  # the smile row stays his own (the fake's mean)
        assert loaded.exp_d0[0, 3, 0] == 0.0
        assert torch.equal(loaded.exp_prev, loaded.exp_d0)

    def test_anchor_keeps_his_own_reading(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._fixture(tmp_path, torch, None)  # the default
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)

        for _ in range(2):
            engine._render(loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=True)

        for k, axis in _MOUTH_OPENING:
            assert loaded.exp_d0[0, k, axis] == 0.0, (k, axis)

    def test_a_seeded_anchor_gets_the_same_treatment(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._fixture(tmp_path, torch, "portrait")
        state = {
            "exp_d0": np.full((1, NUM_KP, 3), 0.01, dtype=np.float32),
            "t_d0": np.zeros((1, 3), dtype=np.float32),
            "scale_d0": np.ones((1, 1), dtype=np.float32),
            "R_d0": np.eye(3, dtype=np.float32)[None],
            "pose_d0": np.zeros((3, 1), dtype=np.float32),
        }

        engine.seed_anchor(loaded.descriptor.id, state)

        for k, axis in _MOUTH_OPENING:
            assert loaded.exp_d0[0, k, axis] == pytest.approx(0.05), (k, axis)
        assert loaded.exp_d0[0, 14, 1] == pytest.approx(0.01)
        assert torch.equal(loaded.exp_prev, loaded.exp_d0)

        loaded.options = {"mouth_rest": "anchor"}
        engine.seed_anchor(loaded.descriptor.id, state)
        for k, axis in _MOUTH_OPENING:
            assert loaded.exp_d0[0, k, axis] == pytest.approx(0.01), (k, axis)


class TestLipRestTracking:
    """``lip_rest_tracking`` (day 15): once the anchor is frozen, frames the
    gates call a resting mouth -- after a settle -- pull the anchor's
    lip-opening rows toward the extractor's reading; an open or moving
    mouth stops it and starts the settle again; 0 is exact. On the CPU
    fixture (a faked motion net, a no-op graph): ``exp_prev`` stands for
    the reading the replay would have written."""

    def _fixture(self, tmp_path, torch, frames: float = 2.0):
        engine, loaded = TestAnchorGatingInRender()._engine_and_loaded(
            tmp_path, torch=torch, anchor_wanted=1
        )
        loaded.lip_rest_alpha = 1.0 / frames if frames > 0 else 0.0
        loaded.lip_rest_mask = torch.zeros(1, NUM_KP, 3)
        for k, axis in _MOUTH_OPENING:
            loaded.lip_rest_mask[0, k, axis] = 1.0
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
        engine._render(loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=True)  # frozen
        assert loaded.anchor_seen == loaded.anchor_wanted
        # The reading a resting frame would leave: 0.1 on every row.
        loaded.exp_prev.fill_(0.1)
        return engine, loaded, driver_rgb

    @staticmethod
    def _rest(engine, loaded, driver_rgb, n: int, gates=(0.0, 0.0, 0.0)):
        for _ in range(n):
            engine._render(loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=False, gates=gates)

    def test_reads_and_validates(self):
        assert _lip_rest_tracking({}) == 100.0
        assert _lip_rest_tracking({"lip_rest_tracking": 0}) == 0.0
        with pytest.raises(ValueError, match="lip_rest_tracking"):
            _lip_rest_tracking({"lip_rest_tracking": -1})

    def test_resting_frames_move_the_lip_rows_only_after_the_settle(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded, driver_rgb = self._fixture(tmp_path, torch, frames=2.0)

        self._rest(engine, loaded, driver_rgb, 15)
        assert loaded.lip_rest_run == 15
        assert float(loaded.exp_d0.abs().max()) == 0.0  # nothing yet

        self._rest(engine, loaded, driver_rgb, 1)
        for k, axis in _MOUTH_OPENING:
            assert loaded.exp_d0[0, k, axis] == pytest.approx(0.05), (k, axis)
        self._rest(engine, loaded, driver_rgb, 1)
        for k, axis in _MOUTH_OPENING:
            assert loaded.exp_d0[0, k, axis] == pytest.approx(0.075), (k, axis)
        # Only those three components: the smile row, the brows, everything
        # else stays the anchor's own.
        untouched = loaded.exp_d0.clone()
        for k, axis in _MOUTH_OPENING:
            untouched[0, k, axis] = 0.0
        assert float(untouched.abs().max()) == 0.0

    def test_an_open_or_moving_mouth_stops_it_and_restarts_the_settle(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded, driver_rgb = self._fixture(tmp_path, torch, frames=2.0)
        self._rest(engine, loaded, driver_rgb, 16)
        after_one = float(loaded.exp_d0[0, 19, 1])
        assert after_one == pytest.approx(0.05)

        self._rest(engine, loaded, driver_rgb, 1, gates=(1.0, 0.0, 0.0))  # the width gate open
        assert loaded.lip_rest_run == 0
        self._rest(engine, loaded, driver_rgb, 1, gates=(0.0, 1.0, 0.0))  # the opening gate open
        assert loaded.lip_rest_run == 0
        assert float(loaded.exp_d0[0, 19, 1]) == pytest.approx(after_one)

        self._rest(engine, loaded, driver_rgb, 15)
        assert float(loaded.exp_d0[0, 19, 1]) == pytest.approx(after_one)
        self._rest(engine, loaded, driver_rgb, 1)
        assert float(loaded.exp_d0[0, 19, 1]) == pytest.approx(0.075)

    def test_a_held_face_and_an_unfrozen_anchor_do_not_count(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded, driver_rgb = self._fixture(tmp_path, torch, frames=2.0)
        loaded.holding = True
        self._rest(engine, loaded, driver_rgb, 20)
        assert loaded.lip_rest_run == 0 and float(loaded.exp_d0.abs().max()) == 0.0
        loaded.holding = False
        engine.reset_anchor(loaded.descriptor.id)  # learning again: nothing moves
        self._rest(engine, loaded, driver_rgb, 20)
        assert float(loaded.exp_d0.abs().max()) == 0.0

    def test_off_is_exact(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded, driver_rgb = self._fixture(tmp_path, torch, frames=0.0)
        self._rest(engine, loaded, driver_rgb, 40)
        assert float(loaded.exp_d0.abs().max()) == 0.0

    def test_seeding_and_resetting_forget_the_count(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded, driver_rgb = self._fixture(tmp_path, torch, frames=2.0)
        self._rest(engine, loaded, driver_rgb, 10)
        assert loaded.lip_rest_run == 10
        engine.reset_anchor(loaded.descriptor.id)
        assert loaded.lip_rest_run == 0
        self._rest(engine, loaded, driver_rgb, 0)
        state = {
            "exp_d0": np.zeros((1, NUM_KP, 3), dtype=np.float32),
            "t_d0": np.zeros((1, 3), dtype=np.float32),
            "scale_d0": np.ones((1, 1), dtype=np.float32),
            "R_d0": np.eye(3, dtype=np.float32)[None],
            "pose_d0": np.zeros((3, 1), dtype=np.float32),
        }
        loaded.lip_rest_run = 7
        engine.seed_anchor(loaded.descriptor.id, state)
        assert loaded.lip_rest_run == 0


class TestBrowMouthLeak:
    """``brow_mouth_leak`` (day 15): the extractor reads a brow raise into
    the mouth rows too, and a toothy smile with the brows up closed by a
    third for it; the leak his brow deviation predicts is written by
    ``_render`` and subtracted in ``_drive``."""

    def test_reads_and_validates(self):
        assert _brow_mouth_leak({}) == 1.0
        assert _brow_mouth_leak({"brow_mouth_leak": 0.0}) == 0.0
        with pytest.raises(ValueError, match="brow_mouth_leak"):
            _brow_mouth_leak({"brow_mouth_leak": 2.5})

    def test_render_writes_the_leak_from_his_brow_deviation(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = TestAnchorGatingInRender()._engine_and_loaded(
            tmp_path, torch=torch, anchor_wanted=1
        )
        loaded.leak_gain = 1.0
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)

        engine._render(loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=False, brow_deviation=0.05)

        for (k, axis), slope in _BROW_MOUTH_LEAK.items():
            assert loaded.mouth_leak[0, k, axis] == pytest.approx(slope * 0.05), (k, axis)
        rest = loaded.mouth_leak.clone()
        for k, axis in _BROW_MOUTH_LEAK:
            rest[0, k, axis] = 0.0
        assert float(rest.abs().max()) == 0.0
        # No deviation, no leak; half the gain, half the leak.
        engine._render(loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=False, brow_deviation=0.0)
        assert float(loaded.mouth_leak.abs().max()) == 0.0
        loaded.leak_gain = 0.5
        engine._render(loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=False, brow_deviation=0.05)
        assert loaded.mouth_leak[0, 20, 1] == pytest.approx(_BROW_MOUTH_LEAK[(20, 1)] * 0.025)

    def test_drive_subtracts_it_from_the_relative_expression(self, tmp_path):
        torch = pytest.importorskip("torch")
        calls: dict[str, list] = {}
        engine, loaded = TestDriveNetworkOptions()._engine_and_loaded(
            tmp_path, torch=torch, calls=calls
        )
        engine._nets["motion"] = lambda image: {
            "exp": torch.zeros(1, NUM_KP * 3),
            "scale": torch.ones(1, 1),
            "t": torch.zeros(1, 3),
            "pitch": torch.zeros(1, 66),
            "yaw": torch.zeros(1, 66),
            "roll": torch.zeros(1, 66),
        }
        loaded.options = {"size_smoothing": 0.0, "eye_retargeting": False}
        loaded.x_c_s = torch.ones(1, NUM_KP, 3)
        loaded.rel_mask = torch.ones(1, NUM_KP, 3)
        loaded.gate = torch.ones(1, NUM_KP, 3)

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        without = calls["warping"][-1][1]["kp_driving"].float().clone()
        loaded.mouth_leak[0, 20, 1] = 0.01
        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        with_leak = calls["warping"][-1][1]["kp_driving"].float().clone()

        diff = with_leak - without
        assert diff[0, 20, 1] == pytest.approx(-0.01, abs=5e-4)  # fp16 keypoints
        diff[0, 20, 1] = 0.0
        assert float(diff.abs().max()) < 5e-4


class TestEyeLipRatios:
    """``_eye_ratios``/``_lip_ratio``: LivePortrait's own close-ratio measure,
    on MediaPipe's points -- the eye/lip retargeting networks' own input."""

    def test_eye_ratio_is_the_lid_gap_over_the_eye_width(self):
        points = np.zeros((FACE_POINTS + 10, 3), dtype=np.float32)
        points[33] = (0.0, 0.5, 0.0)  # left outer
        points[133] = (1.0, 0.5, 0.0)  # left inner
        points[159] = (0.5, 0.35, 0.0)  # left upper lid
        points[145] = (0.5, 0.65, 0.0)  # left lower lid
        points[362] = (0.0, 0.5, 0.0)  # right inner
        points[263] = (1.0, 0.5, 0.0)  # right outer
        points[386] = (0.5, 0.4, 0.0)  # right upper lid
        points[374] = (0.5, 0.6, 0.0)  # right lower lid

        left, right = _eye_ratios(points)

        assert left == pytest.approx(0.30, rel=1e-4)
        assert right == pytest.approx(0.20, rel=1e-4)

    def test_eye_ratio_is_zero_when_the_lid_is_shut(self):
        points = np.zeros((FACE_POINTS + 10, 3), dtype=np.float32)
        points[33] = (0.0, 0.5, 0.0)
        points[133] = (1.0, 0.5, 0.0)
        points[159] = points[145] = (0.5, 0.5, 0.0)

        left, _ = _eye_ratios(points)

        assert left == pytest.approx(0.0, abs=1e-6)

    def test_lip_ratio_is_the_inner_gap_over_the_mouth_width(self):
        points = np.zeros((FACE_POINTS + 10, 3), dtype=np.float32)
        points[61] = (0.0, 0.5, 0.0)
        points[291] = (1.0, 0.5, 0.0)
        points[13] = (0.5, 0.4, 0.0)
        points[14] = (0.5, 0.65, 0.0)

        assert _lip_ratio(points) == pytest.approx(0.25, rel=1e-4)

    def test_lip_ratio_is_zero_when_the_mouth_is_shut(self):
        points = np.zeros((FACE_POINTS + 10, 3), dtype=np.float32)
        points[61] = (0.0, 0.5, 0.0)
        points[291] = (1.0, 0.5, 0.0)
        points[13] = points[14] = (0.5, 0.5, 0.0)

        assert _lip_ratio(points) == pytest.approx(0.0, abs=1e-6)


class TestPaste:
    """``_paste``: pure numpy/cv2, needs no GPU, no loaded networks."""

    def _loaded_portrait(self, tmp_path: Path) -> _Portrait:
        pts = face_points()
        to_crop, _ = crop_matrix(pts)
        crop_points = (pts @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
        loaded = _Portrait(descriptor(tmp_path))
        loaded.mask = _oval_mask(crop_points)
        loaded.mask_bbox = _mask_bbox(loaded.mask)
        return loaded

    def test_pixels_outside_the_paste_box_are_byte_identical(self, tmp_path):
        pts = face_points()
        _, to_frame = crop_matrix(pts)
        loaded = self._loaded_portrait(tmp_path)
        assert loaded.mask is not None and loaded.mask_bbox is not None

        scene = np.full((720, 1280, 3), 50, dtype=np.uint8)
        face = np.full((CROP_SIZE, CROP_SIZE, 3), 200, dtype=np.uint8)

        engine = LivePortraitEngine()
        output = engine._paste(scene, face, to_frame, loaded, loaded.mask)

        x0, y0, x1, y1 = _transformed_box(to_frame, loaded.mask_bbox, 1280, 720)
        outside = np.ones((720, 1280), dtype=bool)
        outside[y0:y1, x0:x1] = False
        assert np.array_equal(output[outside], scene[outside])

    def test_a_uniform_face_is_colour_matched_to_the_scene(self, tmp_path):
        """A constant-colour face pasted under full mask weight, colour-matched
        against a constant-colour scene, should land close to the scene's own
        colour deep inside the mask -- proof the lighting-plane correction is
        actually being applied, not skipped."""
        pts = face_points()
        _, to_frame = crop_matrix(pts)
        loaded = self._loaded_portrait(tmp_path)
        assert loaded.mask is not None and loaded.mask_bbox is not None

        scene = np.full((720, 1280, 3), 60, dtype=np.uint8)
        face = np.full((CROP_SIZE, CROP_SIZE, 3), 210, dtype=np.uint8)

        engine = LivePortraitEngine()
        output = engine._paste(scene, face, to_frame, loaded, loaded.mask)

        # The mask's own centre, in crop coordinates, carried into the frame.
        bx0, by0, bx1, by1 = loaded.mask_bbox
        centre_crop = np.array([[(bx0 + bx1) / 2.0, (by0 + by1) / 2.0]], dtype=np.float32)
        centre_frame = centre_crop @ to_frame[:, :2].T + to_frame[:, 2]
        cx, cy = int(centre_frame[0, 0]), int(centre_frame[0, 1])

        assert abs(int(output[cy, cx, 0]) - 60) < 15

    def test_correction_carries_between_frames_with_inertia(self, tmp_path):
        pts = face_points()
        _, to_frame = crop_matrix(pts)
        loaded = self._loaded_portrait(tmp_path)
        assert loaded.mask is not None
        engine = LivePortraitEngine()

        scene = np.full((720, 1280, 3), 60, dtype=np.uint8)
        face = np.full((CROP_SIZE, CROP_SIZE, 3), 210, dtype=np.uint8)

        engine._paste(scene, face, to_frame, loaded, loaded.mask)
        first = loaded.correction
        assert first is not None

        engine._paste(scene, face, to_frame, loaded, loaded.mask)
        second = loaded.correction
        assert second is not None
        # Both frames see the same constant scene and face, so the field
        # should have already converged rather than drifted.
        assert np.abs(first - second).max() < 1.0


def descriptor(tmp_path: Path, engine_id: str = ENGINE_ID) -> AvatarDescriptor:
    root = tmp_path / "avatar_lp"
    root.mkdir(exist_ok=True)
    manifest = AvatarManifest.model_validate(
        {
            "manifest_version": 1,
            "id": "avatar_lp",
            "name": "lp",
            "version": "1.0.0",
            "assets": [{"path": "portrait.png", "role": "source_portrait", "optional": False}],
            "engine": {"engine_id": engine_id, "estimated_vram_mb": 4500},
        }
    )
    return AvatarDescriptor(manifest=manifest, root=root)


class TestDriveNetworkOptions:
    """``_drive``'s control flow around the ``stitching``/``eye_retargeting``/
    ``lip_retargeting`` options, on the CPU.

    The five networks are faked -- constant tensors of the right shape, no
    real weights -- so this asks only "which networks ran, with what shaped
    input", not "is the picture right"; the picture is what
    :class:`TestOnTheGpu` is for. Stitching is off by default since day 4
    (2026-09-05): see the module docstring and :meth:`LivePortraitEngine._drive`
    for why its own 2-D shift was measured to be the whole of "the avatar sits
    to the side of my face". Eye and lip retargeting are on by default from
    the same day, correcting toward the driver's own close ratio -- see
    :data:`_Portrait.ratios`.
    """

    def _engine_and_loaded(self, tmp_path: Path, *, torch, calls: dict[str, list]):
        def counting(name: str, result):
            def net(*args, **kwargs):
                calls.setdefault(name, []).append((args, kwargs))
                return result

            return net

        engine = LivePortraitEngine()
        engine._torch = torch
        engine._device = torch.device("cpu")
        engine._nets = {
            "motion": counting(
                "motion",
                {
                    "exp": torch.zeros(1, NUM_KP * 3),
                    "scale": torch.ones(1, 1),
                    "t": torch.zeros(1, 3),
                    "pitch": torch.zeros(1, 66),
                    "yaw": torch.zeros(1, 66),
                    "roll": torch.zeros(1, 66),
                },
            ),
            "stitching": counting("stitching", torch.zeros(1, NUM_KP * 3 + 2)),
            "eye": counting("eye", torch.zeros(1, NUM_KP * 3)),
            "lip": counting("lip", torch.zeros(1, NUM_KP * 3)),
            "warping": counting("warping", {"out": torch.zeros(1, 1, 1, 1)}),
            "decoder": counting("decoder", torch.zeros(1, 1, 1, 1)),
        }

        loaded = _Portrait(descriptor(tmp_path))
        eye = torch.eye(3)[None]
        zeros_kp = torch.zeros(1, NUM_KP, 3)
        loaded.x_c_s = zeros_kp
        loaded.x_s = zeros_kp
        loaded.exp_s = zeros_kp
        loaded.exp_d0 = zeros_kp
        loaded.scale_s = torch.ones(1, 1)
        loaded.scale_d0 = torch.ones(1, 1)
        loaded.t_s = torch.zeros(1, 3)
        loaded.t_d0 = torch.zeros(1, 3)
        loaded.t_gain = torch.tensor([[1.0, 1.0, 0.0]])
        loaded.scale_gain = torch.zeros(1, 1)
        loaded.R_s = eye
        loaded.R_d0 = eye
        loaded.f_s = torch.zeros(1, 1, 1, 1, 1)
        # Five distinct values so a test can confirm which ones the eye/lip
        # nets' input slices actually carry.
        loaded.ratios = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        # The day-8 region masks, built the same way load_avatar builds them
        # for the default options ({}): none of this class's own tests
        # inspect x_d's values, only which networks ran with what input, so
        # any real tensors of the right shape would do -- these are exactly
        # what production code would build, for honesty's sake.
        loaded.rel_mask, loaded.eye_mask = _region_masks(0.0, True, torch, torch.device("cpu"))
        # The day-9 gate (fully open) and the networks' rest terms (zero):
        # both no-ops here, as load_avatar's own start state is.
        loaded.gate = torch.ones(1, NUM_KP, 3)
        loaded.brow_delta = torch.zeros(1, NUM_KP, 3)
        loaded.mouth_leak = torch.zeros(1, NUM_KP, 3)
        loaded.eye_rest = torch.zeros(1, NUM_KP, 3)
        loaded.lip_rest = torch.zeros(1, NUM_KP, 3)
        # The day-7 smoothing state: primed the same way load_avatar primes
        # it (from the "_s"/"_d0" values above, all zero/identity here), so
        # every test in this class runs with motion_smoothing's default
        # (0.0) exactly as if the state did not exist -- see TestMotionSmoothing
        # for the smoothing behaviour itself.
        loaded.exp_prev = zeros_kp.clone()
        loaded.t_prev = torch.zeros(1, 3)
        loaded.scale_prev = torch.ones(1, 1)
        loaded.pose_prev = torch.zeros(3, 1)
        loaded.pose_d0 = torch.zeros(3, 1)
        return engine, loaded

    def test_off_by_default_the_stitching_network_is_not_called(self, tmp_path):
        torch = pytest.importorskip("torch")
        calls: dict[str, list] = {}
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, calls=calls)
        loaded.options = {}

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))

        assert "stitching" not in calls

    def test_explicitly_enabled_the_stitching_network_is_called_once(self, tmp_path):
        torch = pytest.importorskip("torch")
        calls: dict[str, list] = {}
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, calls=calls)
        loaded.options = {"stitching": True}

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))

        assert len(calls.get("stitching", [])) == 1

    def test_eye_retargeting_runs_by_default(self, tmp_path):
        torch = pytest.importorskip("torch")
        calls: dict[str, list] = {}
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, calls=calls)
        loaded.options = {}

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))

        assert len(calls.get("eye", [])) == 1

    def test_lip_retargeting_is_off_by_default(self, tmp_path):
        """Day 5, 2026-09-05: with the lip target floored at the portrait's
        own closed ratio (see :func:`_lip_target`'s docstring) and the
        rigid-point crop in place, the *un*retargeted mar gain measured 0.94
        -- every gain above 0 only pressed the lips thinner for a mar gain
        that was not needed. See the module's own calibration table in
        :meth:`_drive`."""
        torch = pytest.importorskip("torch")
        calls: dict[str, list] = {}
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, calls=calls)
        loaded.options = {}

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))

        assert "lip" not in calls

    def test_the_eye_network_gets_its_own_three_ratios(self, tmp_path):
        torch = pytest.importorskip("torch")
        calls: dict[str, list] = {}
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, calls=calls)
        loaded.options = {}

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))

        (feat,), _ = calls["eye"][0]
        assert feat.shape == (1, NUM_KP * 3 + 3)
        assert feat[0, -3:].float().tolist() == pytest.approx([0.1, 0.2, 0.3], abs=1e-3)

    def test_the_lip_network_gets_its_own_two_ratios_when_enabled(self, tmp_path):
        torch = pytest.importorskip("torch")
        calls: dict[str, list] = {}
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, calls=calls)
        loaded.options = {"lip_retargeting": True}

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))

        (feat,), _ = calls["lip"][0]
        assert feat.shape == (1, NUM_KP * 3 + 2)
        assert feat[0, -2:].float().tolist() == pytest.approx([0.4, 0.5], abs=1e-3)

    def test_disabled_eye_retargeting_is_not_called(self, tmp_path):
        torch = pytest.importorskip("torch")
        calls: dict[str, list] = {}
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, calls=calls)
        loaded.options = {"eye_retargeting": False}

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))

        assert "eye" not in calls
        assert "lip" not in calls


class TestRotationPivot:
    """``_rotation_pivot``/the pivot branch in ``_drive``: day 10, Fable's
    review of Step 1 -- rotating the portrait's canonical keypoints about a
    fixed point on its own face instead of the keypoint space's own origin,
    so a driver's pitch turns the avatar's face rather than sliding it. See
    ``_rotation_pivot``'s own docstring for the mechanism and the numbers.
    """

    # --- the pure function, in isolation ------------------------------------

    def test_default_is_none(self):
        assert _rotation_pivot({}) == "none"

    def test_reads_the_option(self):
        assert _rotation_pivot({"rotation_pivot": "eyes"}) == "eyes"
        assert _rotation_pivot({"rotation_pivot": "centroid"}) == "centroid"

    def test_an_unknown_value_is_refused(self):
        with pytest.raises(ValueError, match="rotation_pivot"):
            _rotation_pivot({"rotation_pivot": "chin"})

    def test_every_declared_pivot_is_accepted(self):
        for pivot in _ROTATION_PIVOTS:
            assert _rotation_pivot({"rotation_pivot": pivot}) == pivot

    def test_pivot_z_default_is_false(self):
        assert _pivot_z({}) is False

    def test_pivot_z_reads_the_option(self):
        assert _pivot_z({"pivot_z": True}) is True
        assert _pivot_z({"pivot_z": False}) is False

    # --- through _drive, with fakes standing in for the five networks ------

    def _engine_and_loaded(self, tmp_path: Path, *, torch, calls: dict[str, list] | None = None):
        calls = {} if calls is None else calls

        def counting(name: str, result):
            def net(*args, **kwargs):
                calls.setdefault(name, []).append((args, kwargs))
                return result

            return net

        engine = LivePortraitEngine()
        engine._torch = torch
        engine._device = torch.device("cpu")
        engine._nets = {
            "motion": counting(
                "motion",
                {
                    "exp": torch.zeros(1, NUM_KP * 3),
                    "scale": torch.ones(1, 1),
                    "t": torch.zeros(1, 3),
                    "pitch": torch.zeros(1, 66),
                    "yaw": torch.zeros(1, 66),
                    "roll": torch.zeros(1, 66),
                },
            ),
            "stitching": counting("stitching", torch.zeros(1, NUM_KP * 3 + 2)),
            "eye": counting("eye", torch.zeros(1, NUM_KP * 3)),
            "lip": counting("lip", torch.zeros(1, NUM_KP * 3)),
            "warping": counting("warping", {"out": torch.zeros(1, 1, 1, 1)}),
            "decoder": counting("decoder", torch.zeros(1, 1, 1, 1)),
        }

        loaded = _Portrait(descriptor(tmp_path))
        eye = torch.eye(3)[None]
        zeros_kp = torch.zeros(1, NUM_KP, 3)
        loaded.x_c_s = zeros_kp
        loaded.x_s = zeros_kp
        loaded.exp_s = zeros_kp
        loaded.exp_d0 = zeros_kp
        loaded.scale_s = torch.ones(1, 1)
        loaded.scale_d0 = torch.ones(1, 1)
        loaded.t_s = torch.zeros(1, 3)
        loaded.t_d0 = torch.zeros(1, 3)
        loaded.t_gain = torch.tensor([[1.0, 1.0, 0.0]])
        loaded.scale_gain = torch.zeros(1, 1)
        loaded.R_s = eye
        loaded.R_d0 = eye
        loaded.f_s = torch.zeros(1, 1, 1, 1, 1)
        loaded.ratios = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        loaded.rel_mask, loaded.eye_mask = _region_masks(0.0, True, torch, torch.device("cpu"))
        loaded.gate = torch.ones(1, NUM_KP, 3)
        loaded.brow_delta = torch.zeros(1, NUM_KP, 3)
        loaded.mouth_leak = torch.zeros(1, NUM_KP, 3)
        loaded.eye_rest = torch.zeros(1, NUM_KP, 3)
        loaded.lip_rest = torch.zeros(1, NUM_KP, 3)
        loaded.exp_prev = zeros_kp.clone()
        loaded.t_prev = torch.zeros(1, 3)
        loaded.scale_prev = torch.ones(1, 1)
        loaded.pose_prev = torch.zeros(3, 1)
        loaded.pose_d0 = torch.zeros(3, 1)
        loaded.pivot = None
        return engine, loaded

    @staticmethod
    def _one_hot_pose_bin(torch, bin_index: int | None):
        """One of the 66 pose bins as an exact one-hot logit tensor -- a
        1000-wide gap so softmax rounds to exactly 1.0/0.0 in float32, no
        approximation -- ``headpose_pred_to_degree`` then reads exactly
        ``bin_index * 3 - 97.5`` degrees. ``None`` gives the all-zero
        logits every other fixture in this file uses, which read exactly
        0 degrees the same way (verified 2026-09-06, the module's own
        softmax formula: a uniform distribution's own weighted mean)."""
        if bin_index is None:
            return torch.zeros(1, 66)
        logits = torch.full((1, 66), -1000.0)
        logits[0, bin_index] = 1000.0
        return logits

    def _motion_with_pose(self, torch, *, pitch_bin: int | None = None, yaw_bin: int | None = None):
        """A fake motion net with a fixed, exact pitch/yaw (roll stays 0) --
        for the per-axis tests below, which need a real, known-degree
        rotation on one axis and exactly zero on the other."""

        def net(image):
            return {
                "exp": torch.zeros(1, NUM_KP * 3),
                "scale": torch.ones(1, 1),
                "t": torch.zeros(1, 3),
                "pitch": self._one_hot_pose_bin(torch, pitch_bin),
                "yaw": self._one_hot_pose_bin(torch, yaw_bin),
                "roll": torch.zeros(1, 66),
            }

        return net

    def _motion_with_pitch_sequence(self, torch, pitch_degrees: list[float]):
        """A fake motion net that returns the next pitch (as an exact
        one-hot bin -- ``degrees`` must each land on an integer bin,
        ``(d + 97.5) / 3``) from ``pitch_degrees`` each call, yaw/roll/exp/
        scale/t fixed at zero/one -- for the anchor running-mean test,
        which needs the mean to actually move between calls."""
        bins = iter(round((d + 97.5) / 3.0) for d in pitch_degrees)

        def net(image):
            return {
                "exp": torch.zeros(1, NUM_KP * 3),
                "scale": torch.ones(1, 1),
                "t": torch.zeros(1, 3),
                "pitch": self._one_hot_pose_bin(torch, next(bins)),
                "yaw": torch.zeros(1, 66),
                "roll": torch.zeros(1, 66),
            }

        return net

    def test_none_reproduces_the_old_single_step_formula_bit_for_bit(self, tmp_path):
        """The refactor that split ``rot_new = (rot_d @ R_d0^T) @ R_s`` into
        a named ``rel_rot`` and a branch must not change the "none" path's
        own arithmetic -- same tensors, same order of operations, so the
        two are compared with ``torch.equal``, not ``allclose``. A
        non-trivial rotation and a non-zero keypoint spread, so
        associativity actually has something to get wrong: with zeros
        everywhere the old and a broken formula could agree by accident.
        """
        torch = pytest.importorskip("torch")
        calls: dict[str, list] = {}
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, calls=calls)
        # A real 90-degree rotation (Y into Z) as the anchor's own R_d0;
        # rot_d stays identity (the fake motion net's pitch/yaw/roll are
        # all-zero, and headpose_pred_to_degree(zeros) is exactly 0 -- see
        # the module's own softmax formula), so rel_rot = R_d0^T, itself a
        # genuine 90-degree rotation, not a near-identity numerical fluke.
        r_d0 = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]])
        loaded.R_d0 = r_d0
        loaded.x_c_s = torch.linspace(-1.0, 1.0, NUM_KP * 3).view(1, NUM_KP, 3)
        loaded.options = {}

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        got = calls["warping"][-1][1]["kp_driving"]

        rot_d = torch.eye(3)[None]
        rot_new = (rot_d @ loaded.R_d0.permute(0, 2, 1)) @ loaded.R_s
        expected = (loaded.x_c_s @ rot_new).half()  # delta/scale/t/eye-net are all no-ops here
        assert torch.equal(got, expected)

    def test_a_pivot_at_the_identity_rotation_matches_none(self, tmp_path):
        """At ``rel_rot = I`` (rest, or any frame back at the anchor), a
        pivot changes nothing algebraically -- ``(x - p) @ I + p == x`` --
        but it is a *different* sequence of floating-point operations, so
        this is checked with a tolerance, not ``torch.equal``, and is a
        distinct claim from the "none" test above. The tolerance is fp16's
        own resolution (``kp_driving`` is cast with ``.half()`` before
        either path reaches this test), not float32's -- a tighter one
        would fail on a rounding-boundary value with nothing wrong."""
        torch = pytest.importorskip("torch")
        calls: dict[str, list] = {}
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, calls=calls)
        loaded.x_c_s = torch.linspace(-1.0, 1.0, NUM_KP * 3).view(1, NUM_KP, 3)
        loaded.pivot = loaded.x_c_s.mean(dim=1, keepdim=True).clone()
        loaded.options = {"rotation_pivot": "centroid"}

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        pivoted = calls["warping"][-1][1]["kp_driving"].float()

        calls.clear()
        loaded.options = {"rotation_pivot": "none"}
        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        none = calls["warping"][-1][1]["kp_driving"].float()

        assert torch.allclose(pivoted, none, atol=2e-3)

    def test_a_ninety_degree_pitch_pivot_leaves_the_pivot_point_stationary(self, tmp_path):
        """The point at the pivot must map to itself under a pure-pitch
        relative rotation -- catches the concrete, plausible bug of
        rotating about the pivot but forgetting to shift back afterwards
        (in which case this keypoint would land elsewhere). Since day 10's
        second review, the formula only touches the *pitch* component
        (see ``_rotation_pivot``'s own docstring), so ``R_d0``/``pose_d0``
        must agree on the same 90-degree pitch -- built from one call, not
        two matching-by-hand numbers -- for "a pure pitch" to be true;
        ``pitch_s`` stays 0 (the fixture's own fake motion net), so
        ``rel_pitch = 0 - 90 = -90`` degrees. ``pivot_z`` is set to keep
        the pivot's own z here on purpose -- this test is about the
        shift-back algebra in full 3-D, not about :func:`_pivot_z`'s own
        default (covered by its own tests above)."""
        torch = pytest.importorskip("torch")
        from sha256.avatar.liveportrait.camera import get_rotation_matrix

        calls: dict[str, list] = {}
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, calls=calls)
        ninety, zero = torch.tensor([90.0]), torch.zeros(1)
        loaded.R_d0 = get_rotation_matrix(ninety, zero, zero)  # type: ignore[no-untyped-call]
        loaded.pose_d0 = torch.tensor([[90.0], [0.0], [0.0]])
        pivot = torch.tensor([[[0.3, -0.2, 0.5]]])
        x_c_s = torch.linspace(-1.0, 1.0, NUM_KP * 3).view(1, NUM_KP, 3).clone()
        x_c_s[0, 5, :] = pivot[0, 0, :]
        loaded.x_c_s = x_c_s
        loaded.pivot = pivot
        loaded.options = {"rotation_pivot": "eyes", "pivot_z": True}

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        got = calls["warping"][-1][1]["kp_driving"].float()

        assert torch.allclose(got[0, 5, :], pivot[0, 0, :], atol=1e-3)

    def test_a_pure_yaw_with_the_eyes_pivot_matches_none(self, tmp_path):
        """THE regression test for the day-10 finding: a pivot that adds
        only the pitch delta's own translation must leave a pure yaw turn
        identical to "none" -- exactly what broke (the turn-lag roughly
        doubled) when the first version rotated the whole relative
        rotation about a fixed point instead. ``rel_pitch`` is 0 here
        (the anchor and this frame both read exactly 0 degrees of pitch),
        so the added translation is exactly zero algebraically; the two
        branches still differ at the last few bits from computing
        ``x_c_s @ rel_rot @ R_s`` in a different grouping, so this is
        checked close, not bit-exact -- fp16's own resolution, the same
        tolerance as the "identity" test above, not float32's."""
        torch = pytest.importorskip("torch")
        calls: dict[str, list] = {}
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, calls=calls)
        engine._nets["motion"] = self._motion_with_pose(torch, yaw_bin=50)  # 52.5 degrees
        loaded.x_c_s = torch.linspace(-1.0, 1.0, NUM_KP * 3).view(1, NUM_KP, 3)
        loaded.pivot = loaded.x_c_s[:, list(EYE_KEYPOINTS), :].mean(dim=1, keepdim=True)
        loaded.options = {"rotation_pivot": "eyes"}

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        eyes = calls["warping"][-1][1]["kp_driving"].float()

        calls.clear()
        loaded.options = {"rotation_pivot": "none"}
        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        none = calls["warping"][-1][1]["kp_driving"].float()

        assert torch.allclose(eyes, none, atol=2e-3)

    def test_a_pure_pitch_with_the_eyes_pivot_matches_the_old_whole_rotation_formula(
        self, tmp_path
    ):
        """At ``rel_yaw = rel_roll = 0``, the per-axis translation
        reproduces the earlier, whole-rotation pivot exactly (see
        ``_rotation_pivot``'s own docstring for why) -- the old formula is
        written out independently here, not imported from anywhere, so
        this cannot pass by both sides sharing the same bug. ``pivot_z``
        is set to keep the pivot's own z -- the old, pre-day-10-second-
        review formula never dropped it, so reproducing it exactly means
        not dropping it here either; :func:`_pivot_z`'s own default is a
        separate, later claim with its own tests above."""
        torch = pytest.importorskip("torch")
        from sha256.avatar.liveportrait.camera import get_rotation_matrix

        calls: dict[str, list] = {}
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, calls=calls)
        engine._nets["motion"] = self._motion_with_pose(torch, pitch_bin=40)  # 22.5 degrees
        loaded.x_c_s = torch.linspace(-1.0, 1.0, NUM_KP * 3).view(1, NUM_KP, 3)
        pivot = loaded.x_c_s[:, list(EYE_KEYPOINTS), :].mean(dim=1, keepdim=True)
        loaded.pivot = pivot
        loaded.options = {"rotation_pivot": "eyes", "pivot_z": True}

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        got = calls["warping"][-1][1]["kp_driving"].float()

        p, zero = torch.tensor([22.5]), torch.zeros(1)
        rel_rot = get_rotation_matrix(p, zero, zero) @ loaded.R_d0.permute(0, 2, 1)  # type: ignore[no-untyped-call]
        old_rotated = (loaded.x_c_s - pivot) @ rel_rot + pivot
        expected = old_rotated @ loaded.R_s  # delta/scale/t/eye-net are all no-ops here

        assert torch.allclose(got, expected, atol=2e-3)

    def test_pose_d0_is_primed_at_load_copied_by_the_anchor_and_left_by_reset(self, tmp_path):
        """``pose_d0`` mirrors ``R_d0``'s own lifecycle exactly: both come
        from the same portrait reading at load, both are hard-overwritten
        together on every accepted anchor frame (:meth:`LivePortraitEngine._learn_anchor`),
        and :meth:`LivePortraitEngine.reset_anchor` leaves both alone (the
        old anchor keeps driving until the next one freezes -- see that
        method's own docstring)."""
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch)
        # At load: pose_d0 == pose_prev, both the portrait's own reading
        # (zero, this fixture) -- see load_avatar's own comment for why
        # the two are primed identically.
        assert torch.equal(loaded.pose_d0, loaded.pose_prev)

        engine._nets["motion"] = self._motion_with_pitch_sequence(torch, [22.5, -37.5])
        image = torch.zeros(1, 3, MODEL_SIZE, MODEL_SIZE)

        engine._learn_anchor(loaded, image, ear=0.3, mar=0.0)
        assert torch.allclose(loaded.pose_d0, torch.tensor([[22.5], [0.0], [0.0]]), atol=1e-3)

        engine._learn_anchor(loaded, image, ear=0.3, mar=0.0)
        # mean(22.5, -37.5) == -7.5
        assert torch.allclose(loaded.pose_d0, torch.tensor([[-7.5], [0.0], [0.0]]), atol=1e-3)

        before = loaded.pose_d0.clone()
        engine.reset_anchor(loaded.descriptor.id)
        assert torch.equal(loaded.pose_d0, before)

    def test_an_unknown_value_is_refused_at_first_drive_call(self, tmp_path):
        """``load_avatar``'s own ``_capture`` calls ``_drive`` eagerly before
        the graph exists -- this is what "at load/first use" means in
        practice, the same as every other option validated inside
        ``_drive`` (see ``TestMotionSmoothing``'s own version of this
        test)."""
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch)
        loaded.options = {"rotation_pivot": "chin"}

        with pytest.raises(ValueError, match="rotation_pivot"):
            engine._drive(loaded, torch.zeros(1, 3, 256, 256))


class TestMotionSmoothing:
    """``_motion_smoothing``/``_ema_``: the day-7 EMA over the extractor's
    own output -- see ``_drive``'s own comment and ``docs/development/
    ROADMAP.md``'s "second clip" section for the live complaint it answers
    (while sitting still, the avatar tried different expressions on its
    own)."""

    # --- the two pure functions, in isolation -------------------------------

    def test_default_is_off(self):
        assert _motion_smoothing({}) == pytest.approx(0.0)

    def test_reads_the_option(self):
        assert _motion_smoothing({"motion_smoothing": 0.65}) == pytest.approx(0.65)

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5, -1.0])
    def test_out_of_range_is_refused(self, bad):
        with pytest.raises(ValueError, match="motion_smoothing"):
            _motion_smoothing({"motion_smoothing": bad})

    def test_ema_blends_toward_the_new_value(self):
        torch = pytest.importorskip("torch")
        state = torch.tensor([1.0, 2.0])
        _ema_(state, torch.tensor([3.0, 4.0]), 0.25)
        assert torch.allclose(
            state, torch.tensor([0.25 * 1.0 + 0.75 * 3.0, 0.25 * 2.0 + 0.75 * 4.0])
        )

    def test_ema_zero_alpha_replaces_the_state_exactly(self):
        """Bit-exact, not merely close: ``state*0.0`` is exactly ``0.0`` and
        ``0.0 + new*1.0`` is exactly ``new`` for any finite value -- what
        makes ``motion_smoothing=0.0`` a true no-op rather than a very small
        effect."""
        torch = pytest.importorskip("torch")
        state = torch.tensor([9.0, -9.0])
        _ema_(state, torch.tensor([1.5, -1.5]), 0.0)
        assert torch.equal(state, torch.tensor([1.5, -1.5]))

    def test_ema_updates_in_place_and_returns_the_same_tensor(self):
        """In place, not a fresh allocation -- what makes it capturable into
        the CUDA graph: every replay reads and writes the same tensor's own
        memory, not a new Python object each call."""
        torch = pytest.importorskip("torch")
        state = torch.tensor([1.0])
        result = _ema_(state, torch.tensor([2.0]), 0.5)
        assert result is state

    # --- through _drive, with fakes standing in for the five networks ------

    def _engine_and_loaded(self, tmp_path: Path, *, torch, calls: dict[str, list] | None = None):
        calls = {} if calls is None else calls

        def counting(name: str, result):
            def net(*args, **kwargs):
                calls.setdefault(name, []).append((args, kwargs))
                return result

            return net

        engine = LivePortraitEngine()
        engine._torch = torch
        engine._device = torch.device("cpu")
        engine._nets = {
            "motion": counting(
                "motion",
                {
                    "exp": torch.zeros(1, NUM_KP * 3),
                    "scale": torch.ones(1, 1),
                    "t": torch.zeros(1, 3),
                    "pitch": torch.zeros(1, 66),
                    "yaw": torch.zeros(1, 66),
                    "roll": torch.zeros(1, 66),
                },
            ),
            "stitching": counting("stitching", torch.zeros(1, NUM_KP * 3 + 2)),
            "eye": counting("eye", torch.zeros(1, NUM_KP * 3)),
            "lip": counting("lip", torch.zeros(1, NUM_KP * 3)),
            "warping": counting("warping", {"out": torch.zeros(1, 1, 1, 1)}),
            "decoder": counting("decoder", torch.zeros(1, 1, 1, 1)),
        }

        loaded = _Portrait(descriptor(tmp_path))
        eye = torch.eye(3)[None]
        zeros_kp = torch.zeros(1, NUM_KP, 3)
        loaded.x_c_s = zeros_kp
        loaded.x_s = zeros_kp
        loaded.exp_s = zeros_kp
        loaded.exp_d0 = zeros_kp
        loaded.scale_s = torch.ones(1, 1)
        loaded.scale_d0 = torch.ones(1, 1)
        loaded.t_s = torch.zeros(1, 3)
        loaded.t_d0 = torch.zeros(1, 3)
        loaded.t_gain = torch.tensor([[1.0, 1.0, 0.0]])
        # Day 13: scale_gain 1.0 here, the old line exactly -- these tests
        # are about the smoothing state, with the extractor's scale passing
        # through to x_d as it did before the gain existed.
        loaded.scale_gain = torch.ones(1, 1)
        loaded.R_s = eye
        loaded.R_d0 = eye
        loaded.f_s = torch.zeros(1, 1, 1, 1, 1)
        loaded.ratios = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        loaded.rel_mask, loaded.eye_mask = _region_masks(
            1.0, False, torch, torch.device("cpu"), mouth_lateral_gain=1.0
        )
        loaded.gate = torch.ones(1, NUM_KP, 3)
        loaded.brow_delta = torch.zeros(1, NUM_KP, 3)
        loaded.mouth_leak = torch.zeros(1, NUM_KP, 3)
        loaded.eye_rest = torch.zeros(1, NUM_KP, 3)
        loaded.lip_rest = torch.zeros(1, NUM_KP, 3)
        loaded.exp_prev = zeros_kp.clone()
        loaded.t_prev = torch.zeros(1, 3)
        loaded.scale_prev = torch.ones(1, 1)
        loaded.pose_prev = torch.zeros(3, 1)
        loaded.pose_d0 = torch.zeros(3, 1)
        return engine, loaded

    @staticmethod
    def _motion_returning(torch, *, exp_value: float, scale_value: float):
        """A fake motion net returning a fixed exp/scale (t and pose held at
        zero) -- enough to move ``exp_prev``/``scale_prev`` without the
        pose-in-degrees softmax machinery complicating the arithmetic."""

        def net(image):
            return {
                "exp": torch.full((1, NUM_KP * 3), exp_value),
                "scale": torch.tensor([[scale_value]]),
                "t": torch.zeros(1, 3),
                "pitch": torch.zeros(1, 66),
                "yaw": torch.zeros(1, 66),
                "roll": torch.zeros(1, 66),
            }

        return net

    def test_zero_matches_the_unsmoothed_path_across_two_different_frames(self, tmp_path):
        """a=0: exp_prev/scale_prev become exactly this frame's own reading
        every call, so ``x_d`` for two different consecutive frames is
        exactly ``scale * exp`` each time -- no memory of the previous frame
        leaks through. (``x_c_s``/``exp_s``/``exp_d0``/``t``/``R`` are all
        zero or identity in this fixture, so ``x_d`` reduces to exactly that
        product; see ``_engine_and_loaded``.)"""
        torch = pytest.importorskip("torch")
        calls: dict[str, list] = {}
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, calls=calls)
        # size_smoothing (day 9, later) blends scale/t on its own, default
        # 0.85 -- off here too, so this is the wholly unsmoothed path.
        loaded.options = {"motion_smoothing": 0.0, "size_smoothing": 0.0}

        for exp_value, scale_value in ((0.4, 1.1), (-0.2, 0.9)):
            engine._nets["motion"] = self._motion_returning(
                torch, exp_value=exp_value, scale_value=scale_value
            )
            engine._drive(loaded, torch.zeros(1, 3, 256, 256))
            expected = (scale_value * torch.full((1, NUM_KP, 3), exp_value)).half()
            got = calls["warping"][-1][1]["kp_driving"]
            assert torch.allclose(got.float(), expected.float(), atol=1e-3), (
                f"exp={exp_value}, scale={scale_value}: smoothing leaked across frames"
            )

    def test_nonzero_blends_two_consecutive_frames(self, tmp_path):
        """First frame at ``motion_smoothing=0`` primes ``exp_prev`` to
        exactly A (see the previous test); switching to 0.5 for the second
        frame (B) then gives ``exp_prev = 0.5*A + 0.5*B`` -- the plain EMA
        formula, checked end to end through ``_drive`` rather than only on
        ``_ema_`` directly."""
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch)
        loaded.options = {"motion_smoothing": 0.0}
        a_exp, b_exp = 0.4, -0.2

        engine._nets["motion"] = self._motion_returning(torch, exp_value=a_exp, scale_value=1.0)
        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        assert torch.allclose(loaded.exp_prev, torch.full((1, NUM_KP, 3), a_exp))

        loaded.options = {"motion_smoothing": 0.5}
        engine._nets["motion"] = self._motion_returning(torch, exp_value=b_exp, scale_value=1.0)
        engine._drive(loaded, torch.zeros(1, 3, 256, 256))

        expected = 0.5 * torch.full((1, NUM_KP, 3), a_exp) + 0.5 * torch.full((1, NUM_KP, 3), b_exp)
        assert torch.allclose(loaded.exp_prev, expected)

    def test_an_invalid_value_is_refused_at_first_drive_call(self, tmp_path):
        """``load_avatar``'s own ``_capture`` calls ``_drive`` eagerly before
        the graph exists -- this is what "at load/first use" means in
        practice; there is no separate validation step to bypass."""
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch)
        loaded.options = {"motion_smoothing": 1.0}

        with pytest.raises(ValueError, match="motion_smoothing"):
            engine._drive(loaded, torch.zeros(1, 3, 256, 256))


class TestRegionMasks:
    """``_eye_expression_gain``/``_eye_mask_output``/``_region_masks``: the
    day-8 masks that confine the extractor's own reading of the eyes, and
    the eye retargeting network's own output, to :data:`EYE_KEYPOINTS` --
    see the module's own keypoint table and ``docs/development/ROADMAP.md``,
    "Day 7, later"."""

    # --- the three pure functions -------------------------------------------

    def test_eye_expression_gain_default_is_zero(self):
        assert _eye_expression_gain({}) == pytest.approx(0.0)

    def test_eye_expression_gain_reads_the_option(self):
        assert _eye_expression_gain({"eye_expression_gain": 0.4}) == pytest.approx(0.4)

    @pytest.mark.parametrize("bad", [-0.1, 1.1, -1.0, 2.0])
    def test_eye_expression_gain_out_of_range_is_refused(self, bad):
        with pytest.raises(ValueError, match="eye_expression_gain"):
            _eye_expression_gain({"eye_expression_gain": bad})

    def test_eye_mask_output_default_is_true(self):
        assert _eye_mask_output({}) is True

    def test_eye_mask_output_reads_the_option(self):
        assert _eye_mask_output({"eye_mask_output": False}) is False

    def test_rel_mask_holds_the_gain_only_on_the_eye_keypoints(self):
        torch = pytest.importorskip("torch")
        rel_mask, _ = _region_masks(0.0, True, torch, torch.device("cpu"), mouth_lateral_gain=1.0)
        assert rel_mask.shape == (1, NUM_KP, 3)
        for k in range(NUM_KP):
            expected = 0.0 if k in EYE_KEYPOINTS else 1.0
            assert torch.allclose(rel_mask[0, k], torch.full((3,), expected)), k

    def test_rel_mask_cuts_the_mouths_lateral_components_by_default(self):
        """Day 9, later: the x of 14/17/20 and the y of 7 tilt the mouth
        (see _MOUTH_LATERAL); at the default mouth_lateral_gain 0.0 they
        are zero, everything else of the mouth stays one."""
        torch = pytest.importorskip("torch")
        rel_mask, _ = _region_masks(1.0, True, torch, torch.device("cpu"))
        for k, axis in _MOUTH_LATERAL:
            assert rel_mask[0, k, axis] == 0.0, (k, axis)
        for k in MOUTH_KEYPOINTS:
            for axis in range(3):
                if (k, axis) not in _MOUTH_LATERAL:
                    assert rel_mask[0, k, axis] == 1.0, (k, axis)

    def test_rel_mask_smile_gain_weights_the_corners_rise_only(self):
        """Keypoint 14's y is the smile (both corners up, +27% width per
        -0.05 on the portrait); nothing else of the mouth is touched by
        the smile gain -- the opening (19/20 y) least of all."""
        torch = pytest.importorskip("torch")
        rel_mask, _ = _region_masks(
            1.0, True, torch, torch.device("cpu"), mouth_lateral_gain=1.0, mouth_smile_gain=0.7
        )
        assert _MOUTH_SMILE == ((14, 1),)
        for k, axis in _MOUTH_SMILE:
            assert rel_mask[0, k, axis] == pytest.approx(0.7), (k, axis)
        for k in MOUTH_KEYPOINTS:
            for axis in range(3):
                if (k, axis) not in _MOUTH_SMILE:
                    assert rel_mask[0, k, axis] == 1.0, (k, axis)

    def test_rel_mask_opening_gain_weights_the_lips_parting_only(self):
        """Day 15: the y of 17 / 19 / 20 is the lips' parting (see
        _MOUTH_OPENING); mouth_opening_gain weights exactly those, and
        nothing of the smile, the lateral components or the rest of the
        mouth -- and it multiplies with expression_gain like the others."""
        torch = pytest.importorskip("torch")
        rel_mask, _ = _region_masks(
            1.0,
            True,
            torch,
            torch.device("cpu"),
            mouth_lateral_gain=1.0,
            mouth_smile_gain=1.0,
            mouth_opening_gain=0.6,
            expression_gain=2.0,
        )
        assert _MOUTH_OPENING == ((17, 1), (19, 1), (20, 1))
        assert not set(_MOUTH_OPENING) & (set(_MOUTH_SMILE) | set(_MOUTH_LATERAL))
        for k, axis in _MOUTH_OPENING:
            assert rel_mask[0, k, axis] == pytest.approx(1.2), (k, axis)
        for k in MOUTH_KEYPOINTS:
            for axis in range(3):
                if (k, axis) not in _MOUTH_OPENING:
                    assert rel_mask[0, k, axis] == pytest.approx(2.0), (k, axis)

    def test_mouth_axis_gain_reads_and_validates(self):
        assert _mouth_axis_gain({}, "mouth_lateral_gain", 0.0) == 0.0
        assert _mouth_axis_gain({}, "mouth_opening_gain", 1.0) == 1.0
        assert _mouth_axis_gain({"mouth_opening_gain": 0.6}, "mouth_opening_gain", 1.0) == 0.6
        assert _mouth_axis_gain({"mouth_smile_gain": 0.8}, "mouth_smile_gain", 1.0) == 0.8
        assert (
            _mouth_axis_gain({"mouth_smile_gain": 1.5}, "mouth_smile_gain", 1.0) == 1.5
        )  # day 12: up to 3
        with pytest.raises(ValueError, match="mouth_smile_gain"):
            _mouth_axis_gain({"mouth_smile_gain": 3.5}, "mouth_smile_gain", 1.0)

    def test_rel_mask_at_gain_one_is_uniformly_ones(self):
        """1.0 is a no-op wherever it appears in the mask, eye keypoints
        included -- day 7 and before's behaviour."""
        torch = pytest.importorskip("torch")
        rel_mask, _ = _region_masks(
            1.0, True, torch, torch.device("cpu"), mouth_lateral_gain=1.0, mouth_smile_gain=1.0
        )
        assert torch.allclose(rel_mask, torch.ones(1, NUM_KP, 3))

    def test_eye_mask_confines_to_the_eye_keypoints_by_default(self):
        torch = pytest.importorskip("torch")
        _, eye_mask = _region_masks(0.0, True, torch, torch.device("cpu"))
        for k in range(NUM_KP):
            expected = 1.0 if k in EYE_KEYPOINTS else 0.0
            assert torch.allclose(eye_mask[0, k], torch.full((3,), expected)), k

    def test_eye_mask_output_false_is_uniformly_ones(self):
        torch = pytest.importorskip("torch")
        _, eye_mask = _region_masks(0.0, False, torch, torch.device("cpu"))
        assert torch.allclose(eye_mask, torch.ones(1, NUM_KP, 3))

    # --- through _drive, with fakes standing in for the five networks ------

    def _engine_and_loaded(
        self,
        tmp_path: Path,
        *,
        torch,
        exp,
        eye_expression_gain: float,
        eye_mask_output: bool,
        eye_delta_value: float = 0.0,
        eye_gain: float = 1.0,
    ):
        calls: dict[str, list] = {}

        def counting(name: str, result):
            def net(*args, **kwargs):
                calls.setdefault(name, []).append((args, kwargs))
                return result

            return net

        engine = LivePortraitEngine()
        engine._torch = torch
        engine._device = torch.device("cpu")
        engine._nets = {
            "motion": lambda image: {
                "exp": exp,
                "scale": torch.ones(1, 1),
                "t": torch.zeros(1, 3),
                "pitch": torch.zeros(1, 66),
                "yaw": torch.zeros(1, 66),
                "roll": torch.zeros(1, 66),
            },
            "eye": lambda feat: torch.full((1, NUM_KP * 3), eye_delta_value),
            "warping": counting("warping", {"out": torch.zeros(1, 1, 1, 1)}),
            "decoder": counting("decoder", torch.zeros(1, 1, 1, 1)),
        }

        loaded = _Portrait(descriptor(tmp_path))
        zeros_kp = torch.zeros(1, NUM_KP, 3)
        loaded.x_c_s = zeros_kp
        loaded.x_s = zeros_kp
        loaded.exp_s = zeros_kp
        loaded.exp_d0 = zeros_kp
        loaded.exp_prev = zeros_kp.clone()
        loaded.scale_s = torch.ones(1, 1)
        loaded.scale_d0 = torch.ones(1, 1)
        loaded.scale_prev = torch.ones(1, 1)
        loaded.t_s = torch.zeros(1, 3)
        loaded.t_d0 = torch.zeros(1, 3)
        loaded.t_gain = torch.tensor([[1.0, 1.0, 0.0]])
        loaded.scale_gain = torch.zeros(1, 1)
        loaded.t_prev = torch.zeros(1, 3)
        loaded.pose_prev = torch.zeros(3, 1)
        loaded.pose_d0 = torch.zeros(3, 1)
        eye = torch.eye(3)[None]
        loaded.R_s = eye
        loaded.R_d0 = eye
        loaded.f_s = torch.zeros(1, 1, 1, 1, 1)
        loaded.ratios = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        # eye_retargeting defaults True, lip_retargeting/stitching default
        # False -- exactly the terms this class means to isolate.
        loaded.options = {"motion_smoothing": 0.0, "eye_gain": eye_gain}
        # mouth_lateral_gain 1.0: this class isolates the eye terms and the
        # gate, so the mouth's own axis weights (day 9, later) are no-ops.
        loaded.rel_mask, loaded.eye_mask = _region_masks(
            eye_expression_gain,
            eye_mask_output,
            torch,
            torch.device("cpu"),
            mouth_lateral_gain=1.0,
        )
        loaded.gate = torch.ones(1, NUM_KP, 3)
        loaded.brow_delta = torch.zeros(1, NUM_KP, 3)
        loaded.mouth_leak = torch.zeros(1, NUM_KP, 3)
        loaded.eye_rest = torch.zeros(1, NUM_KP, 3)
        loaded.lip_rest = torch.zeros(1, NUM_KP, 3)
        return engine, loaded, calls

    @staticmethod
    def _kp_driving(calls):
        (_, kwargs) = calls["warping"][0]
        return kwargs["kp_driving"].float()

    def test_gain_zero_removes_the_extractors_own_reading_of_the_eyes(self, tmp_path):
        """``exp`` differs from ``exp_d0`` (all zero) on the eye keypoints
        and on keypoint 19; at ``eye_expression_gain=0.0`` the eye
        keypoints in ``kp_driving`` stay at the rest pose while keypoint
        19's difference passes through untouched -- with no eye-net
        contribution (``eye_delta_value=0.0``) to confound the reading."""
        torch = pytest.importorskip("torch")
        exp = torch.zeros(1, NUM_KP, 3)
        for k in (*EYE_KEYPOINTS, 19):
            exp[0, k, :] = 0.5
        engine, loaded, calls = self._engine_and_loaded(
            tmp_path,
            torch=torch,
            exp=exp,
            eye_expression_gain=0.0,
            eye_mask_output=True,
            eye_delta_value=0.0,
        )

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        kp_driving = self._kp_driving(calls)

        for k in EYE_KEYPOINTS:
            assert torch.allclose(kp_driving[0, k], torch.zeros(3)), k
        assert torch.allclose(kp_driving[0, 19], torch.full((3,), 0.5))

    def test_gain_one_and_mask_output_false_matches_todays_computation(self, tmp_path):
        """At ``eye_expression_gain=1.0`` (``rel_mask`` all ones) and
        ``eye_mask_output=False`` (``eye_mask`` all ones), both day-8 masks
        are no-ops and ``kp_driving`` equals the pre-day-8 formula exactly:
        ``exp*1 + eye_gain*eye_net_output*1`` at every keypoint."""
        torch = pytest.importorskip("torch")
        exp = torch.full((1, NUM_KP, 3), 0.2)
        engine, loaded, calls = self._engine_and_loaded(
            tmp_path,
            torch=torch,
            exp=exp,
            eye_expression_gain=1.0,
            eye_mask_output=False,
            eye_delta_value=0.3,
            eye_gain=1.0,
        )

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        kp_driving = self._kp_driving(calls)

        assert torch.allclose(kp_driving, torch.full((1, NUM_KP, 3), 0.5), atol=1e-5)

    def test_eye_mask_output_confines_a_nonzero_eye_delta_to_the_eye_keypoints(self, tmp_path):
        torch = pytest.importorskip("torch")
        # 0.7, not 0.5: not exactly representable in fp16, on purpose -- the
        # assertions below tolerate that (atol=1e-3, fp16's own precision),
        # the same margin TestMotionSmoothing's own .half()-touched
        # comparisons use, rather than picking a value that hides it.
        exp = torch.zeros(1, NUM_KP, 3)

        engine_masked, loaded_masked, calls_masked = self._engine_and_loaded(
            tmp_path,
            torch=torch,
            exp=exp,
            eye_expression_gain=1.0,
            eye_mask_output=True,
            eye_delta_value=0.7,
            eye_gain=1.0,
        )
        engine_masked._drive(loaded_masked, torch.zeros(1, 3, 256, 256))
        masked = self._kp_driving(calls_masked)

        # A separate _Portrait built on the same tmp_path (descriptor()'s own
        # directory is exist_ok=True) -- the two loaded portraits are
        # independent Python objects, nothing about them is shared.
        engine_open, loaded_open, calls_open = self._engine_and_loaded(
            tmp_path,
            torch=torch,
            exp=exp,
            eye_expression_gain=1.0,
            eye_mask_output=False,
            eye_delta_value=0.7,
            eye_gain=1.0,
        )
        engine_open._drive(loaded_open, torch.zeros(1, 3, 256, 256))
        unmasked = self._kp_driving(calls_open)

        for k in range(NUM_KP):
            if k in EYE_KEYPOINTS:
                assert torch.allclose(masked[0, k], torch.full((3,), 0.7), atol=1e-3), k
            else:
                assert torch.allclose(masked[0, k], torch.zeros(3), atol=1e-3), k
            assert torch.allclose(unmasked[0, k], torch.full((3,), 0.7), atol=1e-3), k


class TestAnchorFrameOk:
    """``_anchor_frame_ok``: frontal, open-eyed, closed-mouthed, and a known
    pose -- see the constants' own comment in the module for the
    2026-09-05 reproduction that motivated each threshold."""

    @pytest.mark.parametrize(
        ("yaw", "pitch", "roll", "ear", "mar", "expected"),
        [
            (0.0, 0.0, 0.0, 0.30, 0.01, True),
            (4.0, 4.0, 3.0, 0.20, 0.03, True),  # exactly at every threshold
            (4.1, 0.0, 0.0, 0.30, 0.01, False),  # yaw just over
            (0.0, 4.1, 0.0, 0.30, 0.01, False),  # pitch just over
            (0.0, 0.0, 3.1, 0.30, 0.01, False),  # roll just over
            (0.0, 0.0, 0.0, 0.19, 0.01, False),  # eyes too narrow (a blink)
            (0.0, 0.0, 0.0, 0.30, 0.031, False),  # mouth too open
            (-12.0, 0.0, 0.0, 0.30, 0.01, False),  # the reproduction itself
        ],
    )
    def test_the_gate(self, yaw, pitch, roll, ear, mar, expected):
        pose = HeadPose(yaw=yaw, pitch=pitch, roll=roll)
        assert _anchor_frame_ok(pose, ear, mar) is expected

    def test_no_pose_is_rejected(self):
        assert _anchor_frame_ok(None, 0.30, 0.01) is False


class _FakeGraph:
    def replay(self) -> None:
        pass


class TestAnchorGatingInRender:
    """``_render``'s anchor bookkeeping, on the CPU: only ``motion`` is faked
    -- ``graph.replay()`` is a no-op standing in for the whole captured
    ``_drive``, which this is not testing (:class:`TestDriveNetworkOptions`
    and :class:`TestOnTheGpu` do)."""

    def _engine_and_loaded(self, tmp_path: Path, *, torch, anchor_wanted: int = 3):
        engine = LivePortraitEngine()
        engine._torch = torch
        engine._device = torch.device("cpu")
        engine._nets = {
            "motion": lambda image: {
                "exp": torch.zeros(1, NUM_KP * 3),
                "scale": torch.ones(1, 1),
                "t": torch.zeros(1, 3),
                "pitch": torch.zeros(1, 66),
                "yaw": torch.zeros(1, 66),
                "roll": torch.zeros(1, 66),
            }
        }

        loaded = _Portrait(descriptor(tmp_path))
        loaded.exp_s = torch.zeros(1, NUM_KP, 3)
        loaded.exp_d0 = torch.zeros(1, NUM_KP, 3)
        loaded.scale_s = torch.ones(1, 1)
        loaded.scale_d0 = torch.ones(1, 1)
        loaded.t_s = torch.zeros(1, 3)
        loaded.t_d0 = torch.zeros(1, 3)
        loaded.t_gain = torch.tensor([[1.0, 1.0, 0.0]])
        loaded.scale_gain = torch.zeros(1, 1)
        loaded.R_s = torch.eye(3)[None]
        loaded.R_d0 = torch.eye(3)[None]
        loaded.ratios = torch.zeros(1, 5)
        # _render writes this frame's gates and brow offset into these
        # before every replay.
        loaded.gate = torch.ones(1, NUM_KP, 3)
        loaded.brow_delta = torch.zeros(1, NUM_KP, 3)
        loaded.mouth_leak = torch.zeros(1, NUM_KP, 3)
        # The day-7 smoothing state: _render's own path (through the fake
        # graph below) never touches these, but _learn_anchor now primes
        # them on every accepted anchor frame (see TestMotionSmoothing), so
        # they must exist as real tensors even where the anchor is not this
        # test's own concern.
        loaded.exp_prev = torch.zeros(1, NUM_KP, 3)
        loaded.t_prev = torch.zeros(1, 3)
        loaded.scale_prev = torch.ones(1, 1)
        loaded.pose_prev = torch.zeros(3, 1)
        loaded.pose_d0 = torch.zeros(3, 1)
        loaded.graph = _FakeGraph()
        loaded.static_in = torch.zeros(1, 3, MODEL_SIZE, MODEL_SIZE)
        loaded.static_out = torch.zeros(1, 3, CROP_SIZE, CROP_SIZE)
        loaded.anchor_wanted = anchor_wanted
        engine._loaded[loaded.descriptor.id] = loaded
        return engine, loaded

    @staticmethod
    def _render(engine, loaded, driver_rgb, ear, mar, *, anchor_ok):
        """``_render`` with the eye/lip targets set equal to the raw ratios --
        what actually reaches the retargeting networks is not this class's
        concern (:class:`TestEyeLipTargets` and :class:`TestDriveNetworkOptions`
        cover that); only the anchor bookkeeping, which reads ``ear``/``mar``,
        is."""
        return engine._render(loaded, driver_rgb, ear, mar, ear, mar, anchor_ok=anchor_ok)

    def test_rejected_frames_do_not_advance_the_anchor(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch)
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)

        for _ in range(3):
            self._render(engine, loaded, driver_rgb, 0.30, 0.0, anchor_ok=False)

        assert loaded.anchor_seen == 0
        assert loaded.anchor_rejected == 3
        assert loaded.rest_ear is None
        assert loaded.rest_mar is None

    def test_accepted_frames_freeze_the_anchor_and_the_rest_reading(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=3)
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)

        for ear, mar in ((0.30, 0.01), (0.32, 0.02), (0.28, 0.00)):
            self._render(engine, loaded, driver_rgb, ear, mar, anchor_ok=True)

        assert loaded.anchor_seen == 3
        assert loaded.anchor_rejected == 0
        assert loaded.rest_ear == pytest.approx((0.30 + 0.32 + 0.28) / 3, abs=1e-4)
        assert loaded.rest_mar == pytest.approx((0.01 + 0.02 + 0.00) / 3, abs=1e-4)

    def test_a_mix_only_the_accepted_ones_count(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=2)
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)

        self._render(engine, loaded, driver_rgb, 0.30, 0.01, anchor_ok=False)
        self._render(engine, loaded, driver_rgb, 0.30, 0.01, anchor_ok=True)
        self._render(engine, loaded, driver_rgb, 0.30, 0.01, anchor_ok=False)
        self._render(engine, loaded, driver_rgb, 0.30, 0.01, anchor_ok=True)

        assert loaded.anchor_seen == 2
        assert loaded.anchor_rejected == 2

    def test_reset_anchor_clears_the_count_and_the_rest_reading(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=2)
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
        self._render(engine, loaded, driver_rgb, 0.30, 0.01, anchor_ok=True)
        self._render(engine, loaded, driver_rgb, 0.30, 0.01, anchor_ok=False)

        engine.reset_anchor(loaded.descriptor.id)

        assert loaded.anchor_seen == 0
        assert loaded.anchor_rejected == 0
        assert loaded.rest_ear is None
        assert loaded.rest_mar is None

    def test_rest_status_reports_progress_then_none(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=2)
        avatar_id = loaded.descriptor.id
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)

        prefix = "Look straight at the camera to centre the avatar"
        assert engine.rest_status(avatar_id) == f"{prefix} (0/2)"
        self._render(engine, loaded, driver_rgb, 0.30, 0.01, anchor_ok=True)
        assert engine.rest_status(avatar_id) == f"{prefix} (1/2)"
        self._render(engine, loaded, driver_rgb, 0.30, 0.01, anchor_ok=True)
        assert engine.rest_status(avatar_id) is None

    def _feed(self, engine, loaded, samples, *, anchor_ok=True):
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
        for sample in samples:
            engine._render(
                loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=anchor_ok, size_sample=sample
            )

    def test_the_resting_face_height_and_distance_freeze_with_the_anchor(self, tmp_path):
        """Collected on accepted frames only, frozen as *medians* (the
        tracker's first frame read the owner's face 10% shorter than the
        settled ones -- a mean would carry that), and ignored afterwards."""
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=3)

        self._feed(engine, loaded, [(900.0, 9.0)], anchor_ok=False)  # rejected: not counted
        self._feed(engine, loaded, [(335.0, 41.0), (369.0, 40.0)])
        assert loaded.face_height_rest is None  # still collecting
        assert loaded.anchor_seen == 2

        self._feed(engine, loaded, [(371.0, 40.4)])  # third accepted frame: the freeze
        assert loaded.face_height_rest == pytest.approx(369.0)  # median, not mean 358.3
        assert loaded.z_rest == pytest.approx(40.4)
        assert engine.rest_status(loaded.descriptor.id) is None

        self._feed(engine, loaded, [(900.0, 9.0)])  # frozen: a later frame changes nothing
        assert loaded.face_height_rest == pytest.approx(369.0)
        assert loaded.z_rest == pytest.approx(40.4)

    def test_the_driver_face_height_follows_his_distance_once_known(self, tmp_path):
        """The crop's face height after the freeze: the resting height scaled
        by rest distance over current distance -- twice as far, half as tall.
        None before the freeze, and without a usable pose."""
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        near = HeadPose(pitch=0.0, yaw=0.0, roll=0.0, translation=(0.0, 0.0, -40.0))
        far = HeadPose(pitch=0.0, yaw=0.0, roll=0.0, translation=(0.0, 0.0, -80.0))
        assert engine._driver_face_height(loaded, near) is None

        self._feed(engine, loaded, [(369.0, 40.0)])

        assert engine._driver_face_height(loaded, near) == pytest.approx(369.0)
        assert engine._driver_face_height(loaded, far) == pytest.approx(184.5)
        assert engine._driver_face_height(loaded, None) is None
        no_depth = HeadPose(pitch=0.0, yaw=0.0, roll=0.0, translation=(0.0, 0.0, 0.0))
        assert engine._driver_face_height(loaded, no_depth) is None

    def test_reset_anchor_forgets_the_face_height_and_distance(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        self._feed(engine, loaded, [(369.0, 40.0)])
        assert loaded.face_height_rest is not None

        engine.reset_anchor(loaded.descriptor.id)

        assert loaded.face_height_rest is None
        assert loaded.z_rest is None
        assert loaded.size_samples == []

    def test_a_missing_size_sample_leaves_the_height_unknown(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)

        self._render(engine, loaded, driver_rgb, 0.30, 0.01, anchor_ok=True)

        assert loaded.anchor_seen == 1
        assert loaded.face_height_rest is None
        assert engine.rest_status(loaded.descriptor.id) is None

    @staticmethod
    def _motion_with_exp(torch, exp_values):
        """A fake motion net that returns the next value from ``exp_values``
        (an iterator) as ``exp`` each call, everything else fixed at zero/
        one -- for the day-7 smoothing-state priming tests below, which need
        the anchor's *running mean* to actually move between frames."""
        values = iter(exp_values)

        def net(image):
            return {
                "exp": torch.full((1, NUM_KP * 3), next(values)),
                "scale": torch.ones(1, 1),
                "t": torch.zeros(1, 3),
                "pitch": torch.zeros(1, 66),
                "yaw": torch.zeros(1, 66),
                "roll": torch.zeros(1, 66),
            }

        return net

    def test_learn_anchor_primes_the_smoothing_state_to_the_running_mean(self, tmp_path):
        """The day-7 smoothing state tracks the anchor's own running mean
        while it is still being learned -- not zero, and not whatever a
        smoothed graph replay happened to leave it at -- so that once the
        anchor freezes, the state's very first evolution by EMA starts from
        the same clean mean ``exp_d0`` itself was just set to."""
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=2)
        loaded.options = {
            "mouth_rest": "anchor"
        }  # the running mean itself, not the portrait's lip rest
        engine._nets["motion"] = self._motion_with_exp(torch, [0.2, 0.6])
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)

        self._render(engine, loaded, driver_rgb, 0.30, 0.01, anchor_ok=True)
        assert torch.allclose(loaded.exp_prev, torch.full((1, NUM_KP, 3), 0.2))

        self._render(engine, loaded, driver_rgb, 0.30, 0.01, anchor_ok=True)
        assert torch.allclose(loaded.exp_prev, torch.full((1, NUM_KP, 3), 0.4))  # mean(0.2, 0.6)

    def test_reset_anchor_leaves_the_smoothing_state_until_the_next_anchor_frame(self, tmp_path):
        """Documents the choice made in ``reset_anchor``'s own docstring: no
        jump at the instant ``reset_anchor`` is called (matching ``exp_d0``'s
        own behaviour, which this mirrors) -- the *next* accepted anchor
        frame re-primes the state to the new anchor, exactly like ``exp_d0``
        is hard-overwritten there too."""
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch, anchor_wanted=1)
        loaded.options = {
            "mouth_rest": "anchor"
        }  # the running mean itself, not the portrait's lip rest
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)

        engine._nets["motion"] = self._motion_with_exp(torch, [0.7])
        self._render(engine, loaded, driver_rgb, 0.30, 0.01, anchor_ok=True)
        primed = loaded.exp_prev.clone()
        assert torch.allclose(primed, torch.full((1, NUM_KP, 3), 0.7))

        engine.reset_anchor(loaded.descriptor.id)

        assert torch.allclose(loaded.exp_prev, primed)  # unchanged by reset itself

        engine._nets["motion"] = self._motion_with_exp(torch, [0.1])
        self._render(engine, loaded, driver_rgb, 0.30, 0.01, anchor_ok=True)
        assert torch.allclose(loaded.exp_prev, torch.full((1, NUM_KP, 3), 0.1))

    def test_ratios_carry_the_target_not_the_raw_reading(self, tmp_path):
        """``_render``'s third/fourth args (eye/lip target) are what reach the
        network, not the first/fourth (raw ear/mar) -- the two diverge as
        soon as the anchor is known, see :func:`_eye_target`."""
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch)
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)

        engine._render(loaded, driver_rgb, 0.10, 0.20, 0.77, 0.88, anchor_ok=False)

        assert loaded.ratios[0, 2].item() == pytest.approx(0.77, abs=1e-4)
        assert loaded.ratios[0, 4].item() == pytest.approx(0.88, abs=1e-4)

    def test_rest_status_is_none_for_an_unknown_avatar(self):
        assert LivePortraitEngine().rest_status("absent") is None


class TestShouldHold:
    """``_should_hold``/``_hold_thresholds``: freeze the rendered face past
    a pose a frontal portrait cannot redraw -- day 8, 2026-09-06. Pure
    function, no torch needed."""

    def _loaded(self, tmp_path: Path, **options):
        loaded = _Portrait(descriptor(tmp_path))
        loaded.options.update(options)
        return loaded

    def test_default_thresholds(self):
        assert _hold_thresholds({}) == (20.0, 35.0)

    def test_reads_the_options(self):
        assert _hold_thresholds({"hold_pitch_degrees": 10.0, "hold_yaw_degrees": 15.0}) == (
            10.0,
            15.0,
        )

    @pytest.mark.parametrize(
        "options",
        [
            {"hold_pitch_degrees": 0.0},
            {"hold_pitch_degrees": -5.0},
            {"hold_yaw_degrees": 0.0},
            {"hold_yaw_degrees": -1.0},
        ],
    )
    def test_nonpositive_thresholds_are_refused(self, options):
        with pytest.raises(ValueError, match="hold_pitch_degrees/hold_yaw_degrees"):
            _hold_thresholds(options)

    def test_enters_above_the_pitch_threshold(self, tmp_path):
        loaded = self._loaded(tmp_path)
        assert _should_hold(loaded, HeadPose(pitch=-25.0, yaw=0.0)) is True
        assert loaded.holding is True

    def test_does_not_enter_below_the_pitch_threshold(self, tmp_path):
        loaded = self._loaded(tmp_path)
        assert _should_hold(loaded, HeadPose(pitch=-19.0, yaw=0.0)) is False

    def test_stays_held_within_the_release_band_then_releases(self, tmp_path):
        """20 degrees x 0.75 = 15: 16 still holds, 14 releases."""
        loaded = self._loaded(tmp_path)
        assert _should_hold(loaded, HeadPose(pitch=-25.0, yaw=0.0)) is True
        assert _should_hold(loaded, HeadPose(pitch=-16.0, yaw=0.0)) is True
        assert _should_hold(loaded, HeadPose(pitch=-14.0, yaw=0.0)) is False

    def test_yaw_enters_stays_and_releases(self, tmp_path):
        """35 degrees x 0.75 = 26.25: 27 still holds, 25 releases."""
        loaded = self._loaded(tmp_path)
        assert _should_hold(loaded, HeadPose(pitch=0.0, yaw=36.0)) is True
        assert _should_hold(loaded, HeadPose(pitch=0.0, yaw=27.0)) is True
        assert _should_hold(loaded, HeadPose(pitch=0.0, yaw=25.0)) is False

    def test_no_pose_keeps_the_current_state(self, tmp_path):
        loaded = self._loaded(tmp_path)
        loaded.holding = True
        assert _should_hold(loaded, None) is True
        loaded.holding = False
        assert _should_hold(loaded, None) is False

    def test_custom_thresholds_are_honoured(self, tmp_path):
        loaded = self._loaded(tmp_path, hold_pitch_degrees=10.0, hold_yaw_degrees=15.0)
        assert _should_hold(loaded, HeadPose(pitch=11.0, yaw=0.0)) is True
        assert _should_hold(loaded, HeadPose(pitch=0.0, yaw=0.0)) is False

    def test_nonpositive_thresholds_raise_through_should_hold(self, tmp_path):
        loaded = self._loaded(tmp_path, hold_pitch_degrees=0.0)
        with pytest.raises(ValueError, match="hold_pitch_degrees"):
            _should_hold(loaded, HeadPose(pitch=0.0, yaw=0.0))


class _CountingGraph:
    """Stands in for the captured graph in :class:`TestExtremePoseHold`:
    ``replay()`` does nothing to ``static_out`` (the fixed rest pose stays
    the output every time) but counts its own calls, which is what tells a
    held frame apart from a rendered one -- ``_render`` runs iff this
    increments."""

    def __init__(self) -> None:
        self.calls = 0

    def replay(self) -> None:
        self.calls += 1


class TestExtremePoseHold:
    """The day-8 extreme-pose hold, through the whole of ``animate()`` on
    the CPU -- ``_should_hold``'s own unit tests (:class:`TestShouldHold`)
    cover the hysteresis; this covers ``animate`` actually using it: the
    network is not called again, the crop and mask still are."""

    _WIDTH, _HEIGHT = 1280, 720

    def _engine_and_loaded(self, tmp_path: Path, *, torch, anchor_wanted: int = 100):
        engine = LivePortraitEngine()
        engine._torch = torch
        engine._device = torch.device("cpu")
        engine._nets = {
            "motion": lambda image: {
                "exp": torch.zeros(1, NUM_KP * 3),
                "scale": torch.ones(1, 1),
                "t": torch.zeros(1, 3),
                "pitch": torch.zeros(1, 66),
                "yaw": torch.zeros(1, 66),
                "roll": torch.zeros(1, 66),
            }
        }

        loaded = _Portrait(descriptor(tmp_path))
        loaded.exp_s = torch.zeros(1, NUM_KP, 3)
        loaded.exp_d0 = torch.zeros(1, NUM_KP, 3)
        loaded.exp_prev = torch.zeros(1, NUM_KP, 3)
        loaded.scale_s = torch.ones(1, 1)
        loaded.scale_d0 = torch.ones(1, 1)
        loaded.scale_prev = torch.ones(1, 1)
        loaded.t_s = torch.zeros(1, 3)
        loaded.t_d0 = torch.zeros(1, 3)
        loaded.t_gain = torch.tensor([[1.0, 1.0, 0.0]])
        loaded.scale_gain = torch.zeros(1, 1)
        loaded.t_prev = torch.zeros(1, 3)
        loaded.pose_prev = torch.zeros(3, 1)
        loaded.pose_d0 = torch.zeros(3, 1)
        loaded.R_s = torch.eye(3)[None]
        loaded.R_d0 = torch.eye(3)[None]
        loaded.ratios = torch.zeros(1, 5)
        loaded.rel_mask = torch.ones(1, NUM_KP, 3)
        loaded.eye_mask = torch.zeros(1, NUM_KP, 3)
        loaded.gate = torch.ones(1, NUM_KP, 3)
        loaded.brow_delta = torch.zeros(1, NUM_KP, 3)
        loaded.mouth_leak = torch.zeros(1, NUM_KP, 3)
        loaded.graph = _CountingGraph()
        loaded.static_in = torch.zeros(1, 3, MODEL_SIZE, MODEL_SIZE)
        loaded.static_out = torch.zeros(1, 3, CROP_SIZE, CROP_SIZE)
        loaded.anchor_wanted = anchor_wanted

        pts = face_points()
        to_crop, _ = crop_matrix(pts)
        crop_points = (pts @ to_crop[:, :2].T + to_crop[:, 2]).astype(np.float32)
        loaded.mask = _oval_mask(crop_points)
        loaded.mask_bbox = _mask_bbox(loaded.mask)

        engine._loaded[loaded.descriptor.id] = loaded
        return engine, loaded

    def _frame(self, pts: np.ndarray, *, pitch: float, yaw: float, roll: float = 0.0):
        scene = np.zeros((self._HEIGHT, self._WIDTH, 3), dtype=np.uint8)
        frame = VideoFrame(
            scene, PixelFormat.BGR24, FrameMetadata(frame_id=0, capture_timestamp=0.0)
        )
        normalised = np.zeros((pts.shape[0], 3), dtype=np.float32)
        normalised[:, 0] = pts[:, 0] / self._WIDTH
        normalised[:, 1] = pts[:, 1] / self._HEIGHT
        tracking = FaceTrackingResult(
            metadata=frame.metadata,
            face_detected=True,
            landmarks=LandmarkSet(points=normalised, scheme="mediapipe_478"),
            head_pose=HeadPose(pitch=pitch, yaw=yaw, roll=roll, translation=(0.0, 0.0, -40.0)),
        )
        return frame, tracking

    def test_extreme_pitch_holds_the_previous_face(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch)
        pts = face_points()

        frame1, tracking1 = self._frame(pts, pitch=0.0, yaw=0.0)
        engine.animate(loaded.descriptor.id, tracking1, frame1)
        assert loaded.graph.calls == 1
        face_before, to_frame_before, _ = loaded.last

        moved = pts.copy()
        moved[:, 1] += 40.0  # his face really did move, so the crop differs
        frame2, tracking2 = self._frame(moved, pitch=-25.0, yaw=0.0)
        engine.animate(loaded.descriptor.id, tracking2, frame2)

        assert loaded.graph.calls == 1  # the network was not asked again
        assert loaded.holding is True
        face_after, to_frame_after, _ = loaded.last
        assert np.array_equal(face_after, face_before)
        assert not np.allclose(to_frame_after, to_frame_before)

    def test_extreme_yaw_holds_the_previous_face(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch)
        pts = face_points()

        frame1, tracking1 = self._frame(pts, pitch=0.0, yaw=0.0)
        engine.animate(loaded.descriptor.id, tracking1, frame1)
        assert loaded.graph.calls == 1

        frame2, tracking2 = self._frame(pts, pitch=0.0, yaw=40.0)
        engine.animate(loaded.descriptor.id, tracking2, frame2)

        assert loaded.graph.calls == 1
        assert loaded.holding is True

    def test_a_moderate_pose_renders_normally(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch)
        pts = face_points()

        frame1, tracking1 = self._frame(pts, pitch=0.0, yaw=0.0)
        engine.animate(loaded.descriptor.id, tracking1, frame1)
        frame2, tracking2 = self._frame(pts, pitch=-10.0, yaw=0.0)
        engine.animate(loaded.descriptor.id, tracking2, frame2)

        assert loaded.graph.calls == 2
        assert loaded.holding is False

    def test_holding_releases_and_renders_again(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch)
        pts = face_points()

        frame1, tracking1 = self._frame(pts, pitch=0.0, yaw=0.0)
        engine.animate(loaded.descriptor.id, tracking1, frame1)
        frame2, tracking2 = self._frame(pts, pitch=-25.0, yaw=0.0)
        engine.animate(loaded.descriptor.id, tracking2, frame2)  # enters the hold
        assert loaded.graph.calls == 1
        assert loaded.holding is True

        frame3, tracking3 = self._frame(pts, pitch=0.0, yaw=0.0)
        engine.animate(loaded.descriptor.id, tracking3, frame3)  # well under the release band

        assert loaded.graph.calls == 2
        assert loaded.holding is False

    def test_a_hold_with_no_previous_face_renders_normally(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch)
        pts = face_points()

        frame1, tracking1 = self._frame(pts, pitch=-25.0, yaw=0.0)
        engine.animate(loaded.descriptor.id, tracking1, frame1)

        assert loaded.graph.calls == 1
        assert loaded.last is not None

    def test_reset_anchor_clears_holding(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._engine_and_loaded(tmp_path, torch=torch)
        loaded.holding = True

        engine.reset_anchor(loaded.descriptor.id)

        assert loaded.holding is False


def _pose(pitch: float = 0.0, yaw: float = 0.0) -> HeadPose:
    return HeadPose(pitch=pitch, yaw=yaw, roll=0.0, translation=(0.0, 0.0, -40.0))


def _act(*args, **kwargs) -> float:
    """``_mouth_activity``'s two terms (ratio, width) summed -- the whole
    mouth's motion, which is what the tests below reason about."""
    return sum(_mouth_activity(*args, **kwargs))


class TestGateMeasures:
    """``_mouth_width_ratio``/``_brow_ratio``: the two readings the day-9
    gates measure his deviation from rest with -- see their own comment in
    the engine for why these denominators and not the inter-ocular one."""

    @staticmethod
    def _points() -> np.ndarray:
        pts = face_points()
        # face_points places the eye corners on y = 315 and the mouth corners
        # 120 px apart; the jaw, brows and chin are placed here on purpose.
        pts[58], pts[288] = (640.0 - 200.0, 444.0), (640.0 + 200.0, 444.0)
        pts[105], pts[334] = (640.0 - 60.0, 285.0), (640.0 + 60.0, 285.0)
        pts[152] = (640.0, 555.0)
        return pts

    def test_mouth_width_ratio_is_the_mouth_over_the_jaw_at_its_height(self):
        assert _mouth_width_ratio(self._points()) == pytest.approx(120.0 / 400.0, rel=1e-5)

    def test_mouth_open_ratio_is_the_lip_gap_over_the_jaw_not_the_mouth(self):
        """Day 12 (smile without teeth): the same 4-px gap must read the
        same whether the mouth is 60 px or 120 px wide -- the old ratio over
        the mouth's own width halved on a grin, and the opening gate never
        passed a wide closed-lip smile with the teeth just showing."""
        pts = self._points()
        pts[13], pts[14] = (640.0, 442.0), (640.0, 446.0)  # a 4-px inner-lip gap
        assert _mouth_open_ratio(pts) == pytest.approx(4.0 / 400.0, rel=1e-5)

        grin = pts.copy()
        grin[61], grin[291] = (640.0 - 120.0, 444.0), (640.0 + 120.0, 444.0)
        assert _mouth_open_ratio(grin) == pytest.approx(_mouth_open_ratio(pts), rel=1e-6)
        assert _lip_ratio(grin) == pytest.approx(_lip_ratio(pts) / 2.0, rel=1e-6)

        twice = (pts - (640.0, 360.0)) * 2.0 + (640.0, 360.0)
        assert _mouth_open_ratio(twice) == pytest.approx(_mouth_open_ratio(pts), rel=1e-6)

    def test_brow_ratio_is_the_brow_lift_over_the_eye_to_chin_height(self):
        # brows 30 px above the eye corners (315 - 285), chin 240 px below them
        assert _brow_ratio(self._points()) == pytest.approx(30.0 / 240.0, rel=1e-5)

    def test_both_ratios_ignore_the_size_of_the_face(self):
        pts = self._points()
        twice = (pts - (640.0, 360.0)) * 2.0 + (640.0, 360.0)
        assert _mouth_width_ratio(twice) == pytest.approx(_mouth_width_ratio(pts), rel=1e-6)
        assert _brow_ratio(twice) == pytest.approx(_brow_ratio(pts), rel=1e-6)

    def test_brow_ratio_ignores_the_lids(self):
        """A blink moves the upper lids (159/386), not this measure -- the
        lid-based gap measure_follow.py reports rose by a third on each
        blink, which would have opened the brow gate on every one."""
        pts = self._points()
        shut = pts.copy()
        shut[159, 1] += 20.0
        shut[386, 1] += 20.0
        assert _brow_ratio(shut) == pytest.approx(_brow_ratio(pts), rel=1e-6)


class TestGates:
    """``_soft_gate``/``_fade``/``_mouth_gate``/``_brow_gate``/``_release``:
    the day-9 expression gates as pure functions of his own readings -- see
    the GATES section of the engine for the live complaint and the numbers
    the thresholds came from."""

    def test_soft_gate_is_zero_to_start_one_from_full_and_linear_between(self):
        assert _soft_gate(-1.0, (0.01, 0.03)) == 0.0
        assert _soft_gate(0.01, (0.01, 0.03)) == 0.0
        assert _soft_gate(0.02, (0.01, 0.03)) == pytest.approx(0.5)
        assert _soft_gate(0.03, (0.01, 0.03)) == 1.0
        assert _soft_gate(9.0, (0.01, 0.03)) == 1.0

    def test_soft_gate_rejects_thresholds_that_do_not_rise(self):
        with pytest.raises(ValueError, match="rise"):
            _soft_gate(0.0, (0.03, 0.01))

    def test_fade_is_the_gate_upside_down_on_the_magnitude(self):
        assert _fade(0.0, (10.0, 20.0)) == 1.0
        assert _fade(-15.0, (10.0, 20.0)) == pytest.approx(0.5)
        assert _fade(25.0, (10.0, 20.0)) == 0.0

    def test_expression_gates_default_on_and_read_the_option(self):
        assert _expression_gates({}) is True
        assert _expression_gates({"expression_gates": False}) is False

    def test_mouth_gate_is_fully_open_until_his_rest_is_known(self):
        assert _mouth_gate(0.004, 0.40, None, 0.40) == 1.0
        assert _mouth_gate(0.004, 0.40, 0.004, None) == 1.0

    def test_mouth_gate_is_shut_at_his_own_rest_and_on_a_still_frames_jitter(self):
        assert _mouth_gate(0.004, 0.40, 0.004, 0.40) == 0.0
        # his still frames: opening +0.0012 at most (day 9's +0.003 over
        # the mouth width, 0.4x over the jaw width), width +-1% of rest
        assert _mouth_gate(0.0052, 0.40 * 1.01, 0.004, 0.40) == 0.0
        assert _mouth_gate(0.004, 0.40 * 0.99, 0.004, 0.40) == 0.0

    def test_mouth_gate_opens_with_an_opening_mouth(self):
        start, full = _MOUTH_GATE_MAR
        assert _mouth_gate(0.004 + full, 0.40, 0.004, 0.40) == pytest.approx(1.0)
        assert _mouth_gate(0.004 + (start + full) / 2, 0.40, 0.004, 0.40) == pytest.approx(0.5)

    def test_mouth_gate_opens_with_a_smile_and_not_with_a_narrower_mouth(self):
        _, full = _MOUTH_GATE_WIDTH
        assert _mouth_gate(0.004, 0.40 * (1.0 + full), 0.004, 0.40) == pytest.approx(1.0)
        assert _mouth_gate(0.004, 0.40 * 1.5, 0.004, 0.40) == 1.0
        # a tilt narrows the measure 5%; pursed lips more -- neither opens it
        assert _mouth_gate(0.004, 0.40 * 0.80, 0.004, 0.40) == 0.0

    def test_mouth_gate_width_fades_with_yaw_but_the_ratio_does_not(self):
        _, full = _MOUTH_GATE_WIDTH
        _, turned = _MOUTH_GATE_YAW
        smile = 0.40 * (1.0 + full)
        assert _mouth_gate(0.004, smile, 0.004, 0.40, _pose(yaw=turned)) == 0.0
        assert _mouth_gate(0.004, smile, 0.004, 0.40, _pose(yaw=-turned)) == 0.0
        assert _mouth_gate(0.004, smile, 0.004, 0.40, _pose(yaw=0.0)) == pytest.approx(1.0)
        assert _mouth_gate(0.004 + _MOUTH_GATE_MAR[1], 0.40, 0.004, 0.40, _pose(yaw=turned)) == 1.0

    def test_brow_gate_is_fully_open_until_his_rest_is_known(self):
        assert _brow_gate(0.20, None, _pose()) == 1.0

    def test_brow_gate_is_shut_at_rest_and_opens_either_way(self):
        _, full = _BROW_GATE
        assert _brow_gate(0.20, 0.20, _pose()) == 0.0
        assert _brow_gate(0.20 * 1.02, 0.20, _pose()) == 0.0  # a blink's worth
        assert _brow_gate(0.20 * (1.0 + full), 0.20, _pose()) == pytest.approx(1.0)  # raised
        assert _brow_gate(0.20 * (1.0 - full), 0.20, _pose()) == pytest.approx(1.0)  # frown
        assert _brow_gate(0.20 * 1.5, 0.20, _pose()) == 1.0

    def test_brow_gate_fades_as_the_head_leaves_frontal(self):
        _, full = _BROW_GATE
        raised = 0.20 * (1.0 + full)
        assert _brow_gate(raised, 0.20, _pose(pitch=_BROW_GATE_PITCH[1])) == 0.0
        assert _brow_gate(raised, 0.20, _pose(pitch=-_BROW_GATE_PITCH[1])) == 0.0
        assert _brow_gate(raised, 0.20, _pose(yaw=_BROW_GATE_YAW[1])) == 0.0
        half_pitch = sum(_BROW_GATE_PITCH) / 2.0
        assert _brow_gate(raised, 0.20, _pose(pitch=half_pitch)) == pytest.approx(0.5)
        # no pose this frame: the fade is skipped, not assumed frontal
        assert _brow_gate(raised, 0.20, None) == pytest.approx(1.0)

    def test_mouth_activity_is_zero_with_nothing_to_compare(self):
        assert _act(0.1, 0.4, None, 0.4) == 0.0
        assert _act(0.1, 0.4, 0.004, None) == 0.0

    def test_mouth_activity_counts_ratio_and_width_change_either_way_in_units(self):
        assert _act(0.004, 0.40, 0.004, 0.40) == 0.0
        assert _act(0.004 + _MOUTH_ACTIVITY_MAR, 0.40, 0.004, 0.40) == pytest.approx(1.0)
        assert _act(0.004 - _MOUTH_ACTIVITY_MAR, 0.40, 0.004, 0.40) == pytest.approx(1.0)
        narrower = 0.40 * (1.0 - _MOUTH_ACTIVITY_WIDTH)  # a pursing counts too
        assert _act(0.004, narrower, 0.004, 0.40) == pytest.approx(1.0)
        both = _act(0.004 + _MOUTH_ACTIVITY_MAR, narrower, 0.004, 0.40)
        assert both == pytest.approx(2.0)

    def test_a_still_frames_own_jitter_does_not_read_as_moving(self):
        """His worst still-frame readings (|dMAR| 0.0006, |dW| 0.6% -- the
        p99 of each on his clips, the opening over the jaw width since day
        12: 0.4x day 9's 0.0015) stay under the activity gate's start on
        their own, and both together on one frame open it a twentieth with
        no hold."""
        assert _soft_gate(_act(0.004 + 0.0006, 0.40, 0.004, 0.40), _MOUTH_ACTIVITY_GATE) == 0.0
        assert _soft_gate(_act(0.004, 0.40 * 1.006, 0.004, 0.40), _MOUTH_ACTIVITY_GATE) == 0.0
        both = _soft_gate(_act(0.004 + 0.0006, 0.40 * 1.006, 0.004, 0.40), _MOUTH_ACTIVITY_GATE)
        assert 0.0 <= both < 0.1
        # his live shut-mouth jitter at p99 (|dMAR| 0.0043 on the day-12
        # measure, capture 232151 -- day 9 night's 0.011 over the mouth
        # width; |dW| 2%): each alone a partial open at most, never the hold
        assert _soft_gate(_act(0.004 + 0.0043, 0.40, 0.004, 0.40), _MOUTH_ACTIVITY_GATE) < 1.0
        assert _soft_gate(_act(0.004, 0.40 * 1.02, 0.004, 0.40), _MOUTH_ACTIVITY_GATE) < 0.2
        # a talking frame at the scripted clip's p90 of |dMAR|, 0.008 (day
        # 9's 0.02 over the mouth width): moving
        assert _soft_gate(_act(0.012, 0.40, 0.004, 0.40), _MOUTH_ACTIVITY_GATE) == 1.0

    def test_mouth_activity_discounts_the_width_change_a_turn_makes(self):
        """A 3-degree yaw step moves the width-over-jaw reading 1.2% by
        itself (_WIDTH_PER_DEGREE_YAW); that much is taken off before the
        change counts as motion."""
        turned = 0.40 * (1.0 + 5.0 * _WIDTH_PER_DEGREE_YAW)  # a 5-degree step: +2%
        assert _act(0.004, turned, 0.004, 0.40) > 0.5
        assert _act(0.004, turned, 0.004, 0.40, yaw_change=5.0) == pytest.approx(0.0)
        assert _act(0.004, turned, 0.004, 0.40, yaw_change=-5.0) == pytest.approx(0.0)
        # a real widening on top of the turn still counts
        widened = turned * (1.0 + _MOUTH_ACTIVITY_WIDTH)
        assert _act(0.004, widened, 0.004, 0.40, yaw_change=5.0) > 0.9

    def test_release_opens_at_once_and_closes_by_the_constant_each_frame(self):
        assert _release(1.0, 0.0) == 1.0
        assert _release(0.0, 1.0) == pytest.approx(_GATE_RELEASE)
        assert _release(0.0, _GATE_RELEASE) == pytest.approx(_GATE_RELEASE**2)
        assert _release(0.9, 1.0) == 0.9


class TestGatesInTheFramePath:
    """``LivePortraitEngine._gates`` and the gate's own place in ``_render``
    /``_learn_anchor``, on the CPU with :class:`TestAnchorGatingInRender`'s
    own fixture (a no-op graph; ``_render`` writes ``loaded.gate`` before
    every replay, which is what is read back here)."""

    @staticmethod
    def _fixture(tmp_path, torch, anchor_wanted: int = 2):
        return TestAnchorGatingInRender()._engine_and_loaded(
            tmp_path, torch=torch, anchor_wanted=anchor_wanted
        )

    def test_render_writes_the_gates_into_the_mouth_and_brow_rows_only(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._fixture(tmp_path, torch)
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)

        engine._render(
            loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=False, gates=(0.25, 0.75, 0.5)
        )

        for k in range(NUM_KP):
            for axis in range(3):
                if (k, axis) in _MOUTH_OPENING:
                    expected = 0.75  # the lips' opening has its own gate
                elif k in MOUTH_KEYPOINTS:
                    expected = 0.25
                elif k in BROW_KEYPOINTS:
                    expected = 0.5
                else:
                    expected = 1.0
                assert loaded.gate[0, k, axis] == pytest.approx(expected), (k, axis)

    def test_the_gates_rest_freezes_with_the_anchor(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._fixture(tmp_path, torch, anchor_wanted=2)
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
        assert loaded.rest_mouth_width is None and loaded.rest_brow is None

        for width, brow in ((0.40, 0.20), (0.42, 0.22)):
            engine._render(
                loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=True, rest_sample=(width, brow)
            )

        assert loaded.anchor_seen == 2
        assert loaded.rest_mouth_width == pytest.approx(0.41)
        assert loaded.rest_brow == pytest.approx(0.21)

    def test_without_a_rest_sample_the_gates_rest_stays_unknown(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._fixture(tmp_path, torch, anchor_wanted=1)
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)

        engine._render(loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=True)

        assert loaded.rest_ear == pytest.approx(0.3)
        assert loaded.rest_mouth_width is None and loaded.rest_brow is None
        assert engine._gates(loaded, 0.2, 0.9, 0.9, _pose()) == (1.0, 1.0, 1.0)

    def test_gates_are_open_until_the_rest_is_known_then_read_and_released(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._fixture(tmp_path, torch)
        loaded.options = {"brow_gain": 1.0}
        assert engine._gates(loaded, 0.004, 0.40, 0.20, _pose()) == (1.0, 1.0, 1.0)
        assert (loaded.gate_mouth_prev, loaded.gate_open_prev, loaded.gate_brow_prev) == (
            0.0,
            0.0,
            0.0,
        )

        loaded.rest_mar, loaded.rest_mouth_width, loaded.rest_brow = 0.004, 0.40, 0.20
        # a word: the mouth opens over two frames (_GATE_ATTACK), the brows stay
        assert engine._gates(loaded, 0.10, 0.40, 0.20, _pose()) == pytest.approx(
            (_GATE_ATTACK, _GATE_ATTACK, 0.0)
        )
        # back to rest: getting there is itself motion, so fully open and held
        assert engine._gates(loaded, 0.004, 0.40, 0.20, _pose()) == (1.0, 1.0, 0.0)
        assert loaded.mouth_hold == _MOUTH_HOLD_FRAMES == loaded.open_hold
        # with the hold spent (its own test above), a still frame at rest
        # decays the gate rather than dropping it
        loaded.mouth_hold = loaded.open_hold = 0
        assert engine._gates(loaded, 0.004, 0.40, 0.20, _pose()) == pytest.approx(
            (_GATE_RELEASE, _GATE_RELEASE, 0.0)
        )
        assert engine._gates(loaded, 0.004, 0.40, 0.20, _pose()) == pytest.approx(
            (_GATE_RELEASE**2, _GATE_RELEASE**2, 0.0)
        )
        assert loaded.gate_mouth_prev == pytest.approx(_GATE_RELEASE**2)

    def test_gates_off_pass_everything(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._fixture(tmp_path, torch)
        loaded.options = {"expression_gates": False}
        loaded.rest_mar, loaded.rest_mouth_width, loaded.rest_brow = 0.004, 0.40, 0.20

        assert engine._gates(loaded, 0.004, 0.40, 0.20, _pose(pitch=15.0)) == (1.0, 1.0, 1.0)

    def test_a_moving_mouth_opens_the_gate_and_holds_it_through_a_pause(self, tmp_path):
        """A closed-lip consonant mid-word: his mouth ratio is back at rest,
        but it *moved* to get there, so the gate stays open -- for
        ``_MOUTH_HOLD_FRAMES`` frames, then releases as usual."""
        torch = pytest.importorskip("torch")
        engine, loaded = self._fixture(tmp_path, torch)
        loaded.rest_mar, loaded.rest_mouth_width, loaded.rest_brow = 0.004, 0.40, 0.20
        # a first frame at rest: nothing to compare, nothing moving
        assert engine._gates(loaded, 0.004, 0.40, 0.20, _pose())[:2] == (0.0, 0.0)
        # a vowel: the lips open (both gates, over two frames -- _GATE_ATTACK)
        assert engine._gates(loaded, 0.06, 0.40, 0.20, _pose())[:2] == pytest.approx(
            (_GATE_ATTACK, _GATE_ATTACK)
        )
        assert loaded.mouth_hold == _MOUTH_HOLD_FRAMES == loaded.open_hold
        # shut again ("m"): the ratio is back at rest, but it *moved* to get
        # there -- both gates fully open, both holds set again
        assert engine._gates(loaded, 0.004, 0.40, 0.20, _pose())[:2] == (1.0, 1.0)
        assert loaded.mouth_hold == _MOUTH_HOLD_FRAMES == loaded.open_hold
        # then perfectly still: held open for the hold, counting down
        for expected_hold in range(_MOUTH_HOLD_FRAMES - 1, -1, -1):
            assert engine._gates(loaded, 0.004, 0.40, 0.20, _pose())[:2] == (1.0, 1.0)
            assert loaded.mouth_hold == expected_hold == loaded.open_hold
        # the hold is spent: the ordinary release takes over
        assert engine._gates(loaded, 0.004, 0.40, 0.20, _pose())[:2] == pytest.approx(
            (_GATE_RELEASE, _GATE_RELEASE)
        )

    def test_a_smile_opens_the_mouth_but_not_the_lips(self, tmp_path):
        """Day 9, night: his closed-lip grin (width +20%, ratio at rest) must
        widen the avatar's mouth and part nothing -- the opening gate knows
        only the ratio. A word (ratio moving) opens both."""
        torch = pytest.importorskip("torch")
        engine, loaded = self._fixture(tmp_path, torch)
        loaded.rest_mar, loaded.rest_mouth_width, loaded.rest_brow = 0.004, 0.40, 0.20
        engine._gates(loaded, 0.004, 0.40, 0.20, _pose())
        # the smile forms: width up 20% in one frame (moving, and past the width gate)
        mouth, opening, _ = engine._gates(loaded, 0.004, 0.40 * 1.2, 0.20, _pose())
        assert mouth == pytest.approx(_GATE_ATTACK)
        assert opening == 0.0
        mouth, opening, _ = engine._gates(loaded, 0.004, 0.40 * 1.2, 0.20, _pose())
        assert mouth == 1.0
        assert opening == 0.0
        # the hold was set by the first (moving) frame and is counting down
        assert loaded.mouth_hold == _MOUTH_HOLD_FRAMES - 1 and loaded.open_hold == 0
        # now a word: the ratio moves, the lips may part
        mouth, opening, _ = engine._gates(loaded, 0.06, 0.40 * 1.2, 0.20, _pose())
        assert mouth == 1.0
        assert opening == pytest.approx(_GATE_ATTACK)
        assert loaded.open_hold == _MOUTH_HOLD_FRAMES

    def test_mouth_activity_keeps_the_ratio_and_width_terms_apart(self):
        assert _mouth_activity(0.004 + _MOUTH_ACTIVITY_MAR, 0.40, 0.004, 0.40) == pytest.approx(
            (1.0, 0.0)
        )
        assert _mouth_activity(
            0.004, 0.40 * (1 + _MOUTH_ACTIVITY_WIDTH), 0.004, 0.40
        ) == pytest.approx((0.0, 1.0))
        assert _mouth_activity(0.1, 0.4, None, 0.4) == (0.0, 0.0)

    def test_the_first_frame_after_a_hold_does_not_read_as_speech(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._fixture(tmp_path, torch)
        loaded.rest_mar, loaded.rest_mouth_width, loaded.rest_brow = 0.004, 0.40, 0.20
        engine._gates(loaded, 0.004, 0.40, 0.20, _pose())
        # animate's hold branch forgets the last reading ...
        loaded.mar_prev = loaded.width_prev = None
        # ... so a mouth that changed meanwhile is not a jump
        assert engine._gates(loaded, 0.006, 0.40, 0.20, _pose())[0] == 0.0

    def test_brows_hold_by_default_and_follow_only_with_brow_gain(self, tmp_path):
        """``brow_gain`` defaults to 0.0 -- see its own comment for the
        natural.mkv measurement (a raise rendered as a frown) behind it. The
        rest being unknown still passes everything, as for the mouth."""
        torch = pytest.importorskip("torch")
        engine, loaded = self._fixture(tmp_path, torch)
        raised = 0.20 * 1.5
        assert engine._gates(loaded, 0.004, 0.40, raised, _pose()) == (1.0, 1.0, 1.0)

        loaded.rest_mar, loaded.rest_mouth_width, loaded.rest_brow = 0.004, 0.40, 0.20
        assert engine._gates(loaded, 0.004, 0.40, raised, _pose()) == (0.0, 0.0, 0.0)
        loaded.options = {"brow_gain": 0.5}
        assert engine._gates(loaded, 0.004, 0.40, raised, _pose()) == (0.0, 0.0, 0.5)
        loaded.options = {"brow_gain": 1.0}
        assert engine._gates(loaded, 0.004, 0.40, raised, _pose()) == (0.0, 0.0, 1.0)

    def test_brow_gain_is_validated(self):
        assert _brow_gain({}) == 0.0
        assert _brow_gain({"brow_gain": 1.0}) == 1.0
        with pytest.raises(ValueError, match="brow_gain"):
            _brow_gain({"brow_gain": 1.5})

    def test_reset_anchor_forgets_the_gates_rest_and_release(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = self._fixture(tmp_path, torch)
        loaded.options = {"brow_gain": 1.0}
        loaded.rest_mar, loaded.rest_mouth_width, loaded.rest_brow = 0.004, 0.40, 0.20
        for _ in range(2):  # the mouth gate opens over two frames (_GATE_ATTACK)
            engine._gates(loaded, 0.10, 0.40, 0.20 * 1.5, _pose())
        assert (loaded.gate_mouth_prev, loaded.gate_open_prev, loaded.gate_brow_prev) == (
            1.0,
            1.0,
            1.0,
        )

        engine.reset_anchor(loaded.descriptor.id)

        assert loaded.rest_mouth_width is None and loaded.rest_brow is None
        assert (loaded.gate_mouth_prev, loaded.gate_open_prev, loaded.gate_brow_prev) == (
            0.0,
            0.0,
            0.0,
        )
        assert loaded.mouth_hold == 0 == loaded.open_hold


class TestGateAndRestInDrive:
    """The gate and the networks' rest terms inside ``_drive`` itself, on the
    CPU with :class:`TestRegionMasks`' own fixture -- ``kp_driving`` as
    handed to the warping network is what is read back."""

    @staticmethod
    def _fixture(tmp_path, torch, *, exp, eye_delta_value: float):
        return TestRegionMasks()._engine_and_loaded(
            tmp_path,
            torch=torch,
            exp=exp,
            eye_expression_gain=1.0,
            eye_mask_output=True,
            eye_delta_value=eye_delta_value,
            eye_gain=1.0,
        )

    def test_a_shut_gate_holds_the_mouth_and_brows_at_rest_and_nothing_else(self, tmp_path):
        torch = pytest.importorskip("torch")
        exp = torch.full((1, NUM_KP, 3), 0.5)
        engine, loaded, calls = self._fixture(tmp_path, torch, exp=exp, eye_delta_value=0.0)
        loaded.gate[0, list(MOUTH_KEYPOINTS), :] = 0.0
        loaded.gate[0, list(BROW_KEYPOINTS), :] = 0.0

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        kp_driving = TestRegionMasks._kp_driving(calls)

        for k in range(NUM_KP):
            expected = 0.0 if k in MOUTH_KEYPOINTS or k in BROW_KEYPOINTS else 0.5
            assert torch.allclose(kp_driving[0, k], torch.full((3,), expected), atol=1e-3), k

    def test_a_half_open_gate_passes_half_the_reading(self, tmp_path):
        torch = pytest.importorskip("torch")
        exp = torch.full((1, NUM_KP, 3), 0.5)
        engine, loaded, calls = self._fixture(tmp_path, torch, exp=exp, eye_delta_value=0.0)
        loaded.gate[0, list(MOUTH_KEYPOINTS), :] = 0.5

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        kp_driving = TestRegionMasks._kp_driving(calls)

        for k in MOUTH_KEYPOINTS:
            assert torch.allclose(kp_driving[0, k], torch.full((3,), 0.25), atol=1e-3), k

    def test_the_eye_networks_rest_answer_is_subtracted(self, tmp_path):
        """With the network answering 0.7 to every target and ``eye_rest``
        holding that same 0.7, a frame at rest moves no keypoint at all --
        before day 9 the 0.7 would have closed the eyes."""
        torch = pytest.importorskip("torch")
        exp = torch.zeros(1, NUM_KP, 3)
        engine, loaded, calls = self._fixture(tmp_path, torch, exp=exp, eye_delta_value=0.7)
        loaded.eye_rest = torch.full((1, NUM_KP, 3), 0.7)

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        kp_driving = TestRegionMasks._kp_driving(calls)

        assert torch.allclose(kp_driving, torch.zeros(1, NUM_KP, 3), atol=1e-3)

    def test_the_lip_networks_rest_answer_is_subtracted_when_it_runs(self, tmp_path):
        torch = pytest.importorskip("torch")
        exp = torch.zeros(1, NUM_KP, 3)
        engine, loaded, calls = self._fixture(tmp_path, torch, exp=exp, eye_delta_value=0.0)
        engine._nets["lip"] = lambda feat: torch.full((1, NUM_KP * 3), 0.4)
        loaded.options.update(lip_retargeting=True, lip_gain=1.0)

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        unsubtracted = TestRegionMasks._kp_driving(calls)
        assert torch.allclose(unsubtracted, torch.full((1, NUM_KP, 3), 0.4), atol=1e-3)

        calls.clear()
        loaded.lip_rest = torch.full((1, NUM_KP, 3), 0.4)
        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        assert torch.allclose(
            TestRegionMasks._kp_driving(calls), torch.zeros(1, NUM_KP, 3), atol=1e-3
        )


class TestBrowFollow:
    """The brows driven from his own measured height (``_brow_lift``,
    ``brow_follow``, ``_BROW_LIFT_Y``, the ``brow_delta`` slot) -- see the
    engine's own comment above ``_brow_follow`` for why this exists instead
    of the extractor's brow reading."""

    def test_brow_follow_default_off_and_validated(self):
        assert _brow_follow({}) == 0.0
        assert _brow_follow({"brow_follow": 1.0}) == 1.0
        with pytest.raises(ValueError, match="brow_follow"):
            _brow_follow({"brow_follow": -0.1})

    def test_brow_lift_is_zero_until_rest_is_known_and_inside_the_dead_band(self):
        assert _brow_lift(0.30, None, _pose()) == 0.0
        assert _brow_lift(0.20, 0.20, _pose()) == 0.0
        assert _brow_lift(0.20 * (1.0 + _BROW_DEADBAND), 0.20, _pose()) == pytest.approx(
            0.0, abs=1e-9
        )
        # his own confounds, all under the band: eyes shut +10%, a blink +9%,
        # a closed smile +5%, a turn +8%, a tilt +6%
        for confound in (1.10, 1.09, 1.05, 1.08, 1.06):
            assert _brow_lift(0.20 * confound, 0.20, _pose()) == pytest.approx(0.0, abs=1e-9), (
                confound
            )

    def test_brow_lift_carries_a_raise_past_the_band_and_never_a_lowering(self):
        assert _brow_lift(0.20 * 1.35, 0.20, _pose()) == pytest.approx(0.35 - _BROW_DEADBAND)
        # a gaze up with the head still reads -12%, the same as his frown:
        # neither moves the brows
        assert _brow_lift(0.20 * 0.88, 0.20, _pose()) == 0.0
        assert _brow_lift(0.20 * 0.50, 0.20, _pose()) == 0.0

    def test_brow_lift_fades_with_pose_and_skips_the_fade_without_one(self):
        raised = 0.20 * 1.35
        assert _brow_lift(raised, 0.20, _pose(pitch=_BROW_GATE_PITCH[1])) == 0.0
        assert _brow_lift(raised, 0.20, _pose(yaw=-_BROW_GATE_YAW[1])) == 0.0
        half = sum(_BROW_GATE_PITCH) / 2.0
        assert _brow_lift(raised, 0.20, _pose(pitch=half)) == pytest.approx(
            (0.35 - _BROW_DEADBAND) / 2.0
        )
        assert _brow_lift(raised, 0.20, None) == pytest.approx(0.35 - _BROW_DEADBAND)

    def test_render_writes_the_lift_into_the_brow_keypoints_y_only(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = TestAnchorGatingInRender()._engine_and_loaded(tmp_path, torch=torch)
        driver_rgb = np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)

        engine._render(loaded, driver_rgb, 0.3, 0.0, 0.3, 0.0, anchor_ok=False, brow_lift=0.5)

        for k in range(NUM_KP):
            expected = torch.zeros(3)
            if k in _BROW_LIFT_Y:
                expected[1] = 0.5 * _BROW_LIFT_Y[k]
            assert torch.allclose(loaded.brow_delta[0, k], expected), k
        # only brow keypoints are driven, and at least one is
        assert _BROW_LIFT_Y and set(_BROW_LIFT_Y) <= set(BROW_KEYPOINTS)
        assert all(y < 0.0 for y in _BROW_LIFT_Y.values())  # a raise is -y

    def test_drive_adds_the_brow_offset_to_the_driving_keypoints(self, tmp_path):
        torch = pytest.importorskip("torch")
        exp = torch.zeros(1, NUM_KP, 3)
        engine, loaded, calls = TestRegionMasks()._engine_and_loaded(
            tmp_path,
            torch=torch,
            exp=exp,
            eye_expression_gain=1.0,
            eye_mask_output=True,
            eye_delta_value=0.0,
        )
        loaded.brow_delta[0, 2, 1] = -0.04

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        kp_driving = TestRegionMasks._kp_driving(calls)

        expected = torch.zeros(1, NUM_KP, 3)
        expected[0, 2, 1] = -0.04
        assert torch.allclose(kp_driving, expected, atol=1e-3)


class TestCropSmoothing:
    """The One Euro filter on the crop's centre and size (day 9, later --
    the "bounce"), and the geometry/matrix split of ``crop_matrix`` it
    needed."""

    def test_geometry_and_similarity_compose_to_crop_matrix(self):
        for pts in (face_points(), long_face_points(), face_points(roll_degrees=12.0)):
            expected_to, expected_from = crop_matrix(pts)
            cx, cy, angle, size = _crop_geometry(pts)
            to_crop, to_frame = _similarity(cx, cy, angle, size)
            assert np.allclose(to_crop, expected_to, atol=1e-4)
            assert np.allclose(to_frame, expected_from, atol=1e-4)

    def test_one_euro_passes_the_first_value_and_a_constant_unchanged(self):
        f = _OneEuro(1.0, 0.02)
        assert f(100.0, 1 / 30) == 100.0
        for _ in range(20):
            assert f(100.0, 1 / 30) == pytest.approx(100.0)

    def test_one_euro_takes_out_jitter_and_follows_a_step(self):
        rng = np.random.default_rng(3)
        noisy = 300.0 + rng.normal(0.0, 2.0, 300)
        f = _OneEuro(1.0, 0.02)
        filtered = np.array([f(v, 1 / 30) for v in noisy])
        assert np.std(np.diff(filtered[30:])) < 0.35 * np.std(np.diff(noisy[30:]))
        # a 40 px step: within 5% of the way after a third of a second, and
        # the speed-raised cutoff means it is not left far behind meanwhile
        f = _OneEuro(1.0, 0.02)
        f(300.0, 1 / 30)
        values = [f(340.0, 1 / 30) for _ in range(10)]
        assert values[0] > 305.0
        assert values[-1] == pytest.approx(340.0, abs=2.0)

    def test_one_euro_reset_forgets_where_it_was(self):
        f = _OneEuro(1.0, 0.02)
        f(100.0, 1 / 30)
        f.reset()
        assert f(500.0, 1 / 30) == 500.0

    def test_one_euro_rejects_a_zero_cutoff(self):
        with pytest.raises(ValueError, match="min_cutoff"):
            _OneEuro(0.0, 0.02)

    def test_crop_filters_from_the_options(self):
        """By default only the size is filtered; the centre passes through
        (see _crop_filters for the measurement behind that)."""
        assert _crop_filters({"crop_smoothing": 0.0}) is None
        filters = _crop_filters({})
        assert filters is not None and len(filters) == 3
        assert filters[0] is None and filters[1] is None
        assert filters[2] is not None and filters[2].min_cutoff == _DEFAULT_CROP_SMOOTHING_HZ
        with_centre = _crop_filters({"crop_centre_smoothing": 2.0})
        assert with_centre is not None and with_centre[0] is not None
        assert with_centre[0].min_cutoff == 2.0
        assert _crop_filters({"crop_smoothing": 0.0, "crop_centre_smoothing": 1.0})[2] is None
        with pytest.raises(ValueError, match="crop_smoothing"):
            _crop_filters({"crop_smoothing": -1.0})

    def test_crop_border_defaults_to_black_and_validates(self):
        """Black, as cv2 has it -- see the comment above _CROP_BORDERS for
        the zoomed-clip measurement that put replicate back in its box."""
        cv2 = pytest.importorskip("cv2")
        assert _crop_border_mode({}, cv2) == cv2.BORDER_CONSTANT
        assert _crop_border_mode({"crop_border": "replicate"}, cv2) == cv2.BORDER_REPLICATE
        assert _crop_border_mode({"crop_border": "reflect"}, cv2) == cv2.BORDER_REFLECT_101
        with pytest.raises(ValueError, match="crop_border"):
            _crop_border_mode({"crop_border": "mirror"}, cv2)

    def test_the_mouth_gates_motion_is_measured_per_30fps_frame(self, tmp_path):
        """At 15 fps a frame sees twice the change of a 30-fps frame for the
        same motion; the activity is scaled back so the thresholds mean the
        same thing at any frame rate."""
        torch = pytest.importorskip("torch")
        engine, loaded = TestAnchorGatingInRender()._engine_and_loaded(tmp_path, torch=torch)
        loaded.rest_mar, loaded.rest_mouth_width, loaded.rest_brow = 0.004, 0.40, 0.20
        engine._gates(loaded, 0.004, 0.40, 0.20, _pose())
        # a change worth exactly one 30-fps unit on the ratio ...
        one_unit = 0.004 + _MOUTH_ACTIVITY_MAR
        loaded.last_dt = 2 * _DEFAULT_FRAME_DT  # ... seen over a 15-fps frame
        assert engine._gates(loaded, one_unit, 0.40, 0.20, _pose())[0] == 0.0
        assert loaded.mouth_hold == 0
        # the same change over a 30-fps frame reads as moving
        loaded.last_dt = _DEFAULT_FRAME_DT
        engine._gates(loaded, 0.004, 0.40, 0.20, _pose())  # back, over two frames
        loaded.mouth_hold = 0
        assert engine._gates(loaded, one_unit, 0.40, 0.20, _pose())[0] > 0.0

    def test_mouth_tilt_trim_reads_and_validates(self):
        assert _mouth_tilt_trim({}) == 0.0
        assert _mouth_tilt_trim({"mouth_tilt_trim": 0.005}) == 0.005
        with pytest.raises(ValueError, match="mouth_tilt_trim"):
            _mouth_tilt_trim({"mouth_tilt_trim": 0.1})

    def test_frame_dt_uses_credible_timestamp_gaps_only(self):
        assert _frame_dt(None, None) == _DEFAULT_FRAME_DT
        assert _frame_dt(1.0, None) == _DEFAULT_FRAME_DT
        assert _frame_dt(1.05, 1.0) == pytest.approx(0.05)
        assert _frame_dt(1.0, 1.0) == _DEFAULT_FRAME_DT  # no gap
        assert _frame_dt(3.0, 1.0) == _DEFAULT_FRAME_DT  # a two-second hole

    def test_smooth_crop_is_a_no_op_without_filters_and_steadies_with_them(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = TestAnchorGatingInRender()._engine_and_loaded(tmp_path, torch=torch)
        frame = VideoFrame(
            np.zeros((8, 8, 3), dtype=np.uint8),
            PixelFormat.BGR24,
            FrameMetadata(frame_id=0, capture_timestamp=0.0),
        )
        assert loaded.crop_filters is None
        assert engine._smooth_crop(loaded, frame, 640.0, 360.0, 700.0) == (640.0, 360.0, 700.0)

        loaded.crop_filters = _crop_filters({})
        rng = np.random.default_rng(5)
        sizes = 700.0 + rng.normal(0.0, 3.0, 120)
        xs = 640.0 + rng.normal(0.0, 1.0, 120)
        out, out_x = [], []
        for i, (size, x) in enumerate(zip(sizes, xs, strict=True)):
            frame = VideoFrame(
                np.zeros((8, 8, 3), dtype=np.uint8),
                PixelFormat.BGR24,
                FrameMetadata(frame_id=i, capture_timestamp=i / 30),
            )
            got = engine._smooth_crop(loaded, frame, float(x), 360.0, float(size))
            out.append(got[2])
            out_x.append(got[0])
        assert np.std(np.diff(out[30:])) < 0.35 * np.std(np.diff(sizes[30:]))
        # the centre is not filtered by default: it passes through exactly
        assert np.allclose(out_x, xs)
        assert loaded.last_timestamp == pytest.approx(119 / 30)


class TestSizeSmoothing:
    def test_size_smoothing_default_and_validation(self):
        assert _size_smoothing({}) == 0.85
        assert _size_smoothing({"size_smoothing": 0.0}) == 0.0
        with pytest.raises(ValueError, match="size_smoothing"):
            _size_smoothing({"size_smoothing": 1.0})

    def test_drive_blends_the_extractors_translation_by_size_smoothing(self, tmp_path):
        """With everything else zero, ``kp_driving`` is the (smoothed)
        translation broadcast to every keypoint: at size_smoothing 0.5 a
        fresh t of (1, 0, 0) against a previous 0 lands at 0.5."""
        torch = pytest.importorskip("torch")
        calls: dict[str, list] = {}
        engine, loaded = TestDriveNetworkOptions()._engine_and_loaded(
            tmp_path, torch=torch, calls=calls
        )
        engine._nets["motion"] = lambda image: {
            "exp": torch.zeros(1, NUM_KP * 3),
            "scale": torch.ones(1, 1),
            "t": torch.tensor([[1.0, 0.0, 0.0]]),
            "pitch": torch.zeros(1, 66),
            "yaw": torch.zeros(1, 66),
            "roll": torch.zeros(1, 66),
        }
        loaded.options = {"size_smoothing": 0.5, "eye_retargeting": False}

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        kp_driving = calls["warping"][0][1]["kp_driving"].float()

        assert torch.allclose(kp_driving[0, :, 0], torch.full((NUM_KP,), 0.5), atol=1e-3)
        assert torch.allclose(loaded.t_prev, torch.tensor([[0.5, 0.0, 0.0]]), atol=1e-6)


class TestTranslationGain:
    """``translation_gain_x``/``_y`` (day 11): how much of the extractor's
    relative translation reaches the portrait -- see ``_translation_gain``
    for the brow-raise measurement behind it."""

    def test_default_and_validation(self):
        assert _translation_gain({}) == (1.0, 1.0)
        assert _translation_gain({"translation_gain_y": 0.0}) == (1.0, 0.0)
        assert _translation_gain({"translation_gain_x": 0.5, "translation_gain_y": 0.25}) == (
            0.5,
            0.25,
        )
        with pytest.raises(ValueError, match="translation_gain_y"):
            _translation_gain({"translation_gain_y": 1.5})
        with pytest.raises(ValueError, match="translation_gain_x"):
            _translation_gain({"translation_gain_x": -0.1})

    @pytest.mark.parametrize(
        ("gain", "expected_x", "expected_y"),
        [((1.0, 1.0), 1.0, 0.5), ((0.0, 0.0), 0.0, 0.0), ((1.0, 0.0), 1.0, 0.0)],
    )
    def test_drive_scales_the_relative_translation_per_axis(
        self, tmp_path, gain, expected_x, expected_y
    ):
        """With everything else zero and no size smoothing, ``kp_driving``
        is the gained translation broadcast to every keypoint."""
        torch = pytest.importorskip("torch")
        calls: dict[str, list] = {}
        engine, loaded = TestDriveNetworkOptions()._engine_and_loaded(
            tmp_path, torch=torch, calls=calls
        )
        engine._nets["motion"] = lambda image: {
            "exp": torch.zeros(1, NUM_KP * 3),
            "scale": torch.ones(1, 1),
            "t": torch.tensor([[1.0, 0.5, 0.0]]),
            "pitch": torch.zeros(1, 66),
            "yaw": torch.zeros(1, 66),
            "roll": torch.zeros(1, 66),
        }
        loaded.options = {"size_smoothing": 0.0, "eye_retargeting": False}
        loaded.t_gain = torch.tensor([[gain[0], gain[1], 0.0]])

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        kp_driving = calls["warping"][0][1]["kp_driving"].float()

        assert torch.allclose(kp_driving[0, :, 0], torch.full((NUM_KP,), expected_x), atol=1e-3)
        assert torch.allclose(kp_driving[0, :, 1], torch.full((NUM_KP,), expected_y), atol=1e-3)


class TestScaleGain:
    """``scale_gain`` (day 13): how much of the extractor's relative scale
    reaches the portrait -- see ``_scale_gain`` for the open-mouth
    measurement behind it. Default 0: the portrait's own scale, always."""

    def test_default_and_validation(self):
        assert _scale_gain({}) == 0.0
        assert _scale_gain({"scale_gain": 1.0}) == 1.0
        assert _scale_gain({"scale_gain": 0.25}) == 0.25
        with pytest.raises(ValueError, match="scale_gain"):
            _scale_gain({"scale_gain": 1.5})
        with pytest.raises(ValueError, match="scale_gain"):
            _scale_gain({"scale_gain": -0.1})

    @pytest.mark.parametrize(
        ("gain", "expected"),
        [(1.0, 1.5), (0.0, 1.0), (0.5, 1.25)],
    )
    def test_drive_gains_the_relative_scale(self, tmp_path, gain, expected):
        """The extractor reads scale 1.5 against an anchor of 1.0 and a
        portrait of 1.0; a unit canonical keypoint lands at ``1 + 0.5 *
        gain`` -- at 1.0 exactly the old ``scale_s * scale_now / scale_d0``,
        at 0 the portrait's own scale whatever the extractor read."""
        torch = pytest.importorskip("torch")
        calls: dict[str, list] = {}
        engine, loaded = TestDriveNetworkOptions()._engine_and_loaded(
            tmp_path, torch=torch, calls=calls
        )
        engine._nets["motion"] = lambda image: {
            "exp": torch.zeros(1, NUM_KP * 3),
            "scale": torch.full((1, 1), 1.5),
            "t": torch.zeros(1, 3),
            "pitch": torch.zeros(1, 66),
            "yaw": torch.zeros(1, 66),
            "roll": torch.zeros(1, 66),
        }
        loaded.options = {"size_smoothing": 0.0, "eye_retargeting": False}
        loaded.x_c_s = torch.ones(1, NUM_KP, 3)
        loaded.scale_gain = torch.tensor([[gain]])

        engine._drive(loaded, torch.zeros(1, 3, 256, 256))
        kp_driving = calls["warping"][0][1]["kp_driving"].float()

        assert torch.allclose(kp_driving, torch.full((1, NUM_KP, 3), expected), atol=1e-5)

    def test_load_avatar_bakes_the_gain_from_the_options(self):
        """What load_avatar puts into the (1, 1) tensor is the validated
        option, so a metadata.json value of 1.0 restores the old line."""
        torch = pytest.importorskip("torch")
        for options, expected in (({}, 0.0), ({"scale_gain": 1.0}, 1.0)):
            tensor = torch.tensor([[_scale_gain(options)]], dtype=torch.float32)
            assert tensor.shape == (1, 1)
            assert float(tensor) == expected


class TestGaze:
    """The day-11 gaze drive: his iris offset (``_gaze_offset``) relative to
    his rest, dead-banded and smoothed (``_gaze_drive``), onto the eye
    keypoints' x -- see the GAZE section of the module."""

    def test_gaze_offset_needs_the_iris_points(self):
        assert _gaze_offset(face_points()) is None  # 468 points, no irises

    def test_gaze_offset_is_zero_with_the_irises_centred_and_moves_with_them(self):
        pts = np.zeros((478, 2), dtype=np.float32)
        pts[33], pts[133] = (100.0, 200.0), (140.0, 200.0)  # left eye, 40 wide
        pts[362], pts[263] = (200.0, 200.0), (240.0, 200.0)  # right eye, 40 wide
        pts[468], pts[473] = (120.0, 200.0), (220.0, 200.0)  # irises at the centres
        assert _gaze_offset(pts) == pytest.approx((0.0, 0.0))
        pts[468, 0] += 8.0  # both irises 8 px right = 0.2 of the width
        pts[473, 0] += 8.0
        pts[468, 1] -= 4.0  # and 4 px up
        pts[473, 1] -= 4.0
        assert _gaze_offset(pts) == pytest.approx((0.2, -0.1))

    def test_gaze_gain_default_and_validation(self):
        assert _gaze_gain({}) == 0.0
        assert _gaze_gain({"gaze_gain": 1.5}) == 1.5
        with pytest.raises(ValueError, match="gaze_gain"):
            _gaze_gain({"gaze_gain": 3.5})

    def _portrait(self, tmp_path, options):
        loaded = _Portrait(descriptor(tmp_path))
        loaded.options = options
        return loaded

    def test_drive_is_zero_until_the_rest_is_known_or_while_off(self, tmp_path):
        loaded = self._portrait(tmp_path, {"gaze_gain": 1.0})
        assert LivePortraitEngine._gaze_drive(loaded, (0.3, 0.0)) == 0.0
        loaded.rest_gaze = (0.0, 0.0)
        loaded.options = {}
        assert LivePortraitEngine._gaze_drive(loaded, (0.3, 0.0)) == 0.0

    def test_drive_dead_bands_smooths_and_gains_the_offset_from_rest(self, tmp_path):
        loaded = self._portrait(tmp_path, {"gaze_gain": 2.0})
        loaded.rest_gaze = (0.08, -0.09)
        # Inside the dead band: nothing.
        assert LivePortraitEngine._gaze_drive(loaded, (0.09, -0.09)) == 0.0
        # A 0.12 step past rest: the dead band takes 0.02, the EMA passes
        # 40% of the rest on the first frame, the gain doubles it.
        first = LivePortraitEngine._gaze_drive(loaded, (0.20, -0.09))
        assert first == pytest.approx(2.0 * 0.4 * 0.10)
        # Held there, it converges toward the full 2.0 * 0.10.
        for _ in range(30):
            last = LivePortraitEngine._gaze_drive(loaded, (0.20, -0.09))
        assert last == pytest.approx(0.20, abs=1e-3)
        # A look the other way is signed.
        for _ in range(30):
            last = LivePortraitEngine._gaze_drive(loaded, (-0.04, -0.09))
        assert last == pytest.approx(-0.20, abs=1e-3)

    def test_the_anchor_learns_his_rest_gaze_and_reset_forgets_it(self, tmp_path):
        torch = pytest.importorskip("torch")
        engine, loaded = TestAnchorGatingInRender()._engine_and_loaded(
            tmp_path, torch=torch, anchor_wanted=2
        )
        image = torch.zeros(1, 3, MODEL_SIZE, MODEL_SIZE)
        engine._learn_anchor(loaded, image, 0.3, 0.0, gaze_sample=(0.10, -0.08))
        engine._learn_anchor(loaded, image, 0.3, 0.0, gaze_sample=(0.06, -0.10))
        assert loaded.rest_gaze == pytest.approx((0.08, -0.09))
        engine.reset_anchor(loaded.descriptor.id)
        assert loaded.rest_gaze is None
        assert loaded.gaze_prev == 0.0


class TestEyeScale:
    def test_eye_scale_default_and_validation(self):
        assert _eye_scale({}) == 1.0
        assert _eye_scale({"eye_scale": 0.95}) == 0.95
        with pytest.raises(ValueError, match="eye_scale"):
            _eye_scale({"eye_scale": 0.3})

    def test_eye_target_scales_the_whole_target_and_still_shuts(self):
        assert _eye_target(0.30, 0.30, 0.334, 1.1, 0.95) == pytest.approx(0.334 * 0.95, rel=1e-4)
        assert _eye_target(0.60, 0.30, 0.334, 1.1, 0.95) == pytest.approx(
            0.334 * 1.1 * 0.95, rel=1e-4
        )
        assert _eye_target(0.0, 0.30, 0.334, 1.1, 0.95) == pytest.approx(0.0, abs=1e-6)


class TestEyeHeadroom:
    def test_eye_headroom_default_is_the_portraits_own_rest(self):
        assert _eye_headroom({}) == 1.0

    def test_eye_headroom_reads_and_validates_the_option(self):
        assert _eye_headroom({"eye_headroom": 1.3}) == 1.3
        with pytest.raises(ValueError, match="eye_headroom"):
            _eye_headroom({"eye_headroom": 0.9})
        with pytest.raises(ValueError, match="eye_headroom"):
            _eye_headroom({"eye_headroom": 2.5})

    def test_eye_target_with_headroom_opens_past_the_portraits_rest_and_stops_there(self):
        assert _eye_target(0.39, 0.30, 0.334, 1.3) == pytest.approx(0.334 * 1.3, rel=1e-4)
        assert _eye_target(0.60, 0.30, 0.334, 1.3) == pytest.approx(0.334 * 1.3, rel=1e-4)
        assert _eye_target(0.30, 0.30, 0.334, 1.3) == pytest.approx(0.334, rel=1e-4)


class TestEyeLipTargets:
    """``_eye_target``/``_lip_target``: the retargeting networks' driving
    target, relative to the driver's own rest -- see their own docstrings
    for the live measurements that motivated each rule."""

    def test_eye_target_before_the_anchor_is_known_is_the_portraits_own(self):
        assert _eye_target(0.10, None, 0.334) == pytest.approx(0.334)
        assert _eye_target(0.30, None, 0.334) == pytest.approx(0.334)

    def test_eye_target_at_rest_asks_for_no_change(self):
        assert _eye_target(0.30, 0.30, 0.334) == pytest.approx(0.334, rel=1e-4)

    def test_eye_target_scales_with_how_far_from_rest_he_is(self):
        assert _eye_target(0.0, 0.30, 0.334) == pytest.approx(0.0, abs=1e-6)

    def test_eye_target_never_asks_wider_than_the_portraits_own_rest(self):
        # 2x rest would be 2x c_s_mean uncapped; the headroom (1.0, day 6)
        # stops at the portrait's own open eyes -- wider asked the eye network
        # for brow raises the owner did not make.
        assert _eye_target(0.60, 0.30, 0.334) == pytest.approx(0.334, rel=1e-4)
        assert _eye_target(0.31, 0.30, 0.334) == pytest.approx(0.334, rel=1e-4)

    def test_lip_target_never_asks_past_the_portraits_own_closed_lips(self):
        assert _lip_target(0.001, 0.005) == pytest.approx(0.005)

    def test_lip_target_follows_the_driver_once_past_the_floor(self):
        assert _lip_target(0.2, 0.005) == pytest.approx(0.2)


class TestGazeDownRelief:
    """Day 12, ``gaze_down_relief`` (off by default, his call): while he
    reads the screen with lowered lids the eye target is built from his
    rest, so the avatar -- whose irises cannot follow -- keeps its eyes
    open; a blink still closes. See ``_relieved_ear``."""

    REST, DOWN = 0.30, (0.0, 0.08)

    def test_option_default_off(self):
        assert _gaze_down_relief({}) is False
        assert _gaze_down_relief({"gaze_down_relief": True}) is True

    def test_untouched_until_the_rest_is_known(self):
        assert _relieved_ear(0.20, None, (0.0, 0.2), self.DOWN, []) == 0.20
        assert _relieved_ear(0.20, self.REST, None, self.DOWN, []) == 0.20
        assert _relieved_ear(0.20, self.REST, (0.0, 0.2), None, []) == 0.20

    def test_a_slow_glance_down_with_lowered_lids_reads_as_rest(self):
        # his iris 0.06 of eye width below his rest, lids at 2/3 of rest,
        # having come down over a dozen frames -- the last three of them
        recent = [0.24, 0.22, 0.21]
        assert _relieved_ear(0.20, self.REST, (0.0, 0.14), self.DOWN, recent) == self.REST

    def test_looking_straight_or_up_is_not_relieved(self):
        assert _relieved_ear(0.20, self.REST, (0.0, 0.08), self.DOWN, []) == 0.20
        assert _relieved_ear(0.20, self.REST, (0.0, 0.10), self.DOWN, []) == 0.20  # +0.02
        assert _relieved_ear(0.20, self.REST, (0.0, 0.00), self.DOWN, []) == 0.20

    def test_a_blink_still_closes(self):
        # shut: under half of rest, wherever the iris reads
        assert _relieved_ear(0.10, self.REST, (0.0, 0.2), self.DOWN, [0.30, 0.30, 0.30]) == 0.10
        # shutting fast: a quarter of rest gone within three frames
        assert _relieved_ear(0.20, self.REST, (0.0, 0.2), self.DOWN, [0.30, 0.30, 0.29]) == 0.20
        # the same lids reached slowly are relieved
        assert _relieved_ear(0.20, self.REST, (0.0, 0.2), self.DOWN, [0.26, 0.24, 0.22]) == 0.30

    def test_wider_than_rest_is_left_alone(self):
        assert _relieved_ear(0.34, self.REST, (0.0, 0.2), self.DOWN, [0.33]) == 0.34


class TestProtocolConformance:
    def test_satisfies_the_avatar_engine_protocol(self):
        assert isinstance(LivePortraitEngine(), AvatarEngine)

    def test_reports_its_identity(self):
        built = LivePortraitEngine()
        assert built.engine_id == ENGINE_ID
        assert built.backend_id == ENGINE_ID

    def test_accepts_only_its_own_avatars(self, tmp_path):
        built = LivePortraitEngine()
        assert built.supports(descriptor(tmp_path))
        assert not built.supports(descriptor(tmp_path, engine_id="warp"))

    def test_loading_another_engines_avatar_is_refused(self, tmp_path):
        built = LivePortraitEngine()
        with pytest.raises(AvatarLoadError, match="requires engine"):
            built.load_avatar(descriptor(tmp_path, engine_id="warp"))

    def test_animating_an_unloaded_avatar_is_an_error(self, tmp_path):
        from sha256.core.enums import PixelFormat
        from sha256.core.types import FaceTrackingResult, FrameMetadata, VideoFrame

        built = LivePortraitEngine()
        metadata = FrameMetadata(frame_id=0, capture_timestamp=0.0)
        frame = VideoFrame(np.zeros((8, 8, 3), np.uint8), PixelFormat.BGR24, metadata)
        with pytest.raises(AvatarLoadError, match="not loaded"):
            built.animate("absent", FaceTrackingResult.no_face(metadata), frame)

    def test_vram_is_unmeasured_before_a_load(self):
        assert LivePortraitEngine().estimated_vram_mb("anything") is None


def _cuda_and_weights() -> bool:
    try:
        import torch
    except ImportError:
        return False
    weights = REPO_ROOT / "models" / "liveportrait" / "liveportrait" / "base_models"
    portrait = REPO_ROOT / "avatars" / "my_avatar_neural" / "portrait.png"
    return torch.cuda.is_available() and weights.is_dir() and portrait.is_file()


@pytest.mark.requires_gpu
@pytest.mark.slow
@pytest.mark.skipif(not _cuda_and_weights(), reason="needs CUDA, the weights and the portrait")
class TestOnTheGpu:
    def test_the_portrait_drives_itself(self):
        """The portrait as its own driver: the face comes back where it was,
        the graph agrees with eager execution, and the frame is otherwise
        untouched outside the crop."""
        import cv2

        from sha256.avatar.repository import FilesystemAvatarRepository
        from sha256.core.enums import PixelFormat
        from sha256.core.types import FrameMetadata, VideoFrame
        from sha256.tracking.mediapipe_tracker import (
            DEFAULT_MODEL_RELATIVE_PATH,
            MediaPipeFaceTracker,
        )

        models = REPO_ROOT / "models"
        engine = LivePortraitEngine(
            model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
            weights=models / "liveportrait" / "liveportrait",
        )
        engine.start()
        try:
            desc = FilesystemAvatarRepository(REPO_ROOT / "avatars").get("my_avatar_neural")
            engine.load_avatar(desc)
            assert engine.estimated_vram_mb("my_avatar_neural") is not None

            image = cv2.imread(str(desc.root / "portrait.png"))
            tracker = MediaPipeFaceTracker(
                model_path=models / DEFAULT_MODEL_RELATIVE_PATH, running_mode="image"
            )
            tracker.start()
            frame = VideoFrame(
                image, PixelFormat.BGR24, FrameMetadata(frame_id=0, capture_timestamp=0.0)
            )
            tracking = tracker.process(frame)
            tracker.stop()

            out = engine.animate("my_avatar_neural", tracking, frame)
            assert out.data.shape == image.shape
            assert not out.is_placeholder
            # The face is re-drawn, not copied: it differs from the photograph a
            # little everywhere under the mask and is not a blank.
            diff = np.abs(out.data.astype(np.int16) - image.astype(np.int16))
            assert 0 < float(diff.mean()) < 40.0

            # graph against eager, on the same driver
            loaded: Any = engine._loaded["my_avatar_neural"]
            with engine._torch.no_grad():
                eager = engine._drive(loaded, loaded.static_in).float()
                loaded.graph.replay()
                engine._torch.cuda.synchronize()
                gap = (loaded.static_out.float() - eager).abs().max().item()
            assert gap < 1e-3
        finally:
            engine.stop()

    def test_eye_retargeting_closes_and_opens_the_output_eyes(self):
        """Forcing the driving eye target closed, then open, on the portrait
        driving itself: the rendered face's own tracked eyes follow -- proof
        the eye retargeting network actually does something, not just that
        it is wired in and inert. ``_render`` is called directly (not
        ``animate``) so the forced value is not immediately overwritten by
        the driver's own real reading.

        A relative comparison, not fixed absolute numbers: at the calibrated
        gain (0.06 -- see the module docstring's table, chosen for how well it
        followed the owner's own blinking over the whole clip, not for how far
        a single isolated frame can be pushed) a shut target only nudges this
        one static frame, it does not slam the eyes shut the way gain 1.0 did
        in the probe that showed 1.0 was too strong to begin with.
        """
        import cv2

        from sha256.avatar.repository import FilesystemAvatarRepository
        from sha256.core.enums import PixelFormat
        from sha256.core.types import FrameMetadata, VideoFrame
        from sha256.tracking.mediapipe_tracker import (
            DEFAULT_MODEL_RELATIVE_PATH,
            MediaPipeFaceTracker,
        )

        models = REPO_ROOT / "models"
        engine = LivePortraitEngine(
            model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
            weights=models / "liveportrait" / "liveportrait",
        )
        engine.start()
        try:
            desc = FilesystemAvatarRepository(REPO_ROOT / "avatars").get("my_avatar_neural")
            engine.load_avatar(desc)
            loaded: Any = engine._loaded["my_avatar_neural"]

            image = cv2.imread(str(desc.root / "portrait.png"))
            model_tracker = MediaPipeFaceTracker(
                model_path=models / DEFAULT_MODEL_RELATIVE_PATH, running_mode="image"
            )
            model_tracker.start()
            frame = VideoFrame(
                image, PixelFormat.BGR24, FrameMetadata(frame_id=0, capture_timestamp=0.0)
            )
            tracking = model_tracker.process(frame)
            model_tracker.stop()
            assert tracking.landmarks is not None

            height, width = image.shape[:2]
            target = tracking.landmarks.points[:FACE_POINTS, :2] * (width, height)
            scale = float(loaded.options.get("crop_scale", CROP_SCALE))
            vy = float(loaded.options.get("crop_vy_ratio", CROP_VY_RATIO))
            to_crop, _ = crop_matrix(target, scale=scale, vy_ratio=vy)
            crop = cv2.warpAffine(image, to_crop, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_LINEAR)
            small = cv2.cvtColor(
                cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
            )

            c_s_lip = float(loaded.ratios[0, 3].item())
            c_s_left = float(loaded.ratios[0, 0].item())

            def rendered_ear(c_d_eye: float) -> float:
                # Forced straight through as both the raw reading (for the
                # still-running anchor) and the target (what the network
                # sees) -- this test drives the target directly, bypassing
                # _eye_target's own rest-relative logic on purpose.
                face_rgb = engine._render(loaded, small, c_d_eye, c_s_lip, c_d_eye, c_s_lip)
                face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
                still = MediaPipeFaceTracker(
                    model_path=models / DEFAULT_MODEL_RELATIVE_PATH, running_mode="image"
                )
                still.start()
                result = still.process(
                    VideoFrame(
                        face_bgr,
                        PixelFormat.BGR24,
                        FrameMetadata(frame_id=0, capture_timestamp=0.0),
                    )
                )
                still.stop()
                assert result.landmarks is not None
                pixels = result.landmarks.points[:, :2] * (CROP_SIZE, CROP_SIZE)
                left, right = _eye_ratios(pixels)
                return (left + right) / 2.0

            shut = rendered_ear(0.02)
            open_ = rendered_ear(c_s_left)
            assert shut < open_ - 0.03, f"shut={shut:.3f} open={open_:.3f}: no separation"
        finally:
            engine.stop()
