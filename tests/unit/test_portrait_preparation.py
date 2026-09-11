"""Block B (day 16): the portrait prepared at import -- scaled to the
standard, padded for the crop and the chin -- and read for its profile;
the engine's chin room as a function of the points alone. On a synthetic
face whose key points sit where a face's do, and on the owner's own
portrait when it is on this machine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sha256.avatar.engine_liveportrait import (
    _CHIN_DROP_MAX,
    _OVAL_FOREHEAD_REACH,
    CROP_SIZE,
    chin_room,
)
from sha256.avatar.importer import (
    ORIGINAL_STEM,
    PORTRAIT_FILENAME,
    PROFILE_FILENAME,
    STANDARD_FACE_HEIGHT_PX,
    STANDARD_ROOM_BELOW_CHIN,
    PortraitReport,
    import_portrait,
    inspect_portrait,
    portrait_profile,
    prepare_portrait,
)
from tests.unit.test_avatar_importer import good_report, neural_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = REPO_ROOT / "models" / "downloads" / "face_landmarker.task"
PORTRAIT = REPO_ROOT / "avatars" / "my_avatar_neural" / "portrait.png"


def synthetic_face(
    centre: tuple[float, float], face_height: float, *, width_ratio: float = 0.72
) -> np.ndarray:
    """478 points of a face ``face_height`` tall (forehead top to chin) at
    ``centre``: the oval on an ellipse, the indices the engine and the
    importer read placed where a face has them."""
    cx, cy = centre
    h = face_height
    w = face_height * width_ratio
    pts = np.zeros((478, 3), dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * np.pi, 478, endpoint=False)
    pts[:, 0] = cx + 0.5 * w * np.cos(angles)
    pts[:, 1] = cy + 0.5 * h * np.sin(angles)

    def put(index: int, x: float, y: float) -> None:
        pts[index, 0] = cx + x * w
        pts[index, 1] = cy + y * h

    put(10, 0.0, -0.5)  # forehead top
    put(152, 0.0, 0.5)  # chin
    for i in (234,):
        put(i, -0.5, 0.05)
    for i in (454,):
        put(i, 0.5, 0.05)
    put(33, -0.32, -0.18)
    put(133, -0.12, -0.18)
    put(159, -0.22, -0.21)
    put(145, -0.22, -0.15)
    put(362, 0.12, -0.18)
    put(263, 0.32, -0.18)
    put(386, 0.22, -0.21)
    put(374, 0.22, -0.15)
    put(1, 0.0, 0.05)  # nose tip
    for i, y in ((168, -0.16), (6, -0.12), (197, -0.05), (195, -0.02), (5, 0.02), (4, 0.04)):
        put(i, 0.0, y)
    put(61, -0.2, 0.28)
    put(291, 0.2, 0.28)
    put(13, 0.0, 0.278)
    put(14, 0.0, 0.282)
    put(0, 0.0, 0.24)
    put(17, 0.0, 0.33)
    put(58, -0.42, 0.28)
    put(288, 0.42, 0.28)
    put(50, -0.3, 0.05)
    put(280, 0.3, 0.05)
    for i in (46, 53, 52, 65, 55):
        put(i, -0.25 + 0.05 * ((i * 7) % 5), -0.3)
    for i in (107, 276, 283, 282, 295, 285, 336):
        put(i, 0.25 - 0.05 * ((i * 7) % 5), -0.3)
    put(468, -0.22, -0.18)
    put(473, 0.22, -0.18)
    return pts


def picture_with_face(
    size: tuple[int, int], centre: tuple[float, float], face_height: float
) -> tuple[np.ndarray, np.ndarray]:
    """A plain picture with a skin-toned face ellipse, dark hair above the
    forehead and a slightly darker chin -- enough for the profile to read."""
    import cv2

    width, height = size
    image = np.full((height, width, 3), (200, 200, 200), dtype=np.uint8)
    pts = synthetic_face(centre, face_height)
    cx, cy = centre
    axes = (int(0.36 * face_height), int(0.5 * face_height))
    cv2.ellipse(image, (int(cx), int(cy)), axes, 0, 0, 360, (150, 170, 210), -1)  # skin
    # Hair: a dark band from 0.6 of the forehead up.
    brow_y = cy - 0.3 * face_height
    top_y = cy - 0.5 * face_height
    hair_y = int(brow_y - 0.6 * (brow_y - top_y))
    cv2.rectangle(image, (0, 0), (width, hair_y), (30, 30, 40), -1)
    return image, pts


class TestChinRoom:
    def test_the_room_is_the_crops_own_and_the_drop_takes_what_the_reach_leaves(self):
        pts = synthetic_face((512.0, 512.0), 400.0)
        room = chin_room(pts, image_size=(1024, 1024))
        assert 0 < room.chin_row < CROP_SIZE
        assert room.headroom > 0
        assert room.chin_reach == 0.0
        assert room.chin_drop_max == pytest.approx(
            min(_CHIN_DROP_MAX * room.crop_face_height, room.headroom)
        )
        with_reach = chin_room(pts, image_size=(1024, 1024), chin_reach_fraction=0.04)
        assert with_reach.chin_reach == pytest.approx(0.04 * room.crop_face_height)
        assert with_reach.chin_drop_max <= room.chin_drop_max
        assert chin_room(pts, image_size=(600, 600)).crop_inside is False
        roomy = synthetic_face((1024.0, 1100.0), 400.0)
        assert chin_room(roomy, image_size=(2048, 2048)).crop_inside is True


class TestPreparation:
    def test_a_tall_face_is_scaled_down_and_a_small_one_left(self):
        image, pts = picture_with_face((1600, 1600), (800.0, 800.0), 900.0)
        _, moved, prep = prepare_portrait(image, pts)
        assert prep.scale == pytest.approx(STANDARD_FACE_HEIGHT_PX / 900.0, rel=0.01)
        assert prep.face_height_px == pytest.approx(STANDARD_FACE_HEIGHT_PX, rel=0.02)
        assert moved[152, 1] - moved[10, 1] == pytest.approx(prep.face_height_px, rel=0.02)
        small, small_pts = picture_with_face((1024, 1024), (512.0, 512.0), 300.0)
        _, _, kept = prepare_portrait(small, small_pts)
        assert kept.scale == 1.0

    def test_a_chin_at_the_bottom_edge_gets_its_room_and_the_crop_fits(self):
        # The chin 10 px above the picture's bottom edge.
        image, pts = picture_with_face((1024, 700), (512.0, 490.0), 400.0)
        assert 700 - pts[152, 1] < 0.1 * 400
        out, moved, prep = prepare_portrait(image, pts)
        assert prep.changed and prep.padding[1] > 0
        assert prep.room_below_chin >= STANDARD_ROOM_BELOW_CHIN - 1e-6
        assert prep.room.crop_inside
        assert prep.room.chin_drop_max >= 50.0
        assert out.shape[0] == 700 + prep.padding[1] + prep.padding[0]
        # The points moved with the padding: the chin is where it was, plus the top pad.
        assert moved[152, 1] == pytest.approx(pts[152, 1] + prep.padding[0])
        assert moved[152, 0] == pytest.approx(pts[152, 0] + prep.padding[2])
        # The padding is the picture's own edge, not black.
        assert (out[-1] == out[-prep.padding[1] - 1]).all()
        assert "padded" in prep.describe() and "jaw drop up to" in prep.describe()

    def test_a_picture_with_room_everywhere_is_left_alone(self):
        image, pts = picture_with_face((2400, 2400), (1200.0, 1100.0), 400.0)
        out, _, prep = prepare_portrait(image, pts)
        assert not prep.changed and out is image
        assert prep.room.crop_inside


class TestProfile:
    def test_the_hairline_lowers_the_forehead_reach_and_the_chin_reads_bare(self):
        image, pts = picture_with_face((1024, 1024), (512.0, 560.0), 400.0)
        profile = portrait_profile(image, pts)
        assert profile.hairline is not None and 0.5 < profile.hairline < 0.7
        assert profile.forehead_reach < _OVAL_FOREHEAD_REACH
        assert profile.forehead_reach >= profile.reach_floor
        assert profile.engine_options() == {
            "mask_chin_reach": "auto",
            "mask_forehead_reach": round(profile.forehead_reach, 3),
        }
        assert profile.beard == "bare" and abs(profile.chin_darkness) < 0.05
        assert profile.lips == "closed"
        assert "hairline" in profile.describe()

    def test_a_dark_chin_reads_as_stubble_or_beard_and_hair_far_up_keeps_the_reach(self):
        import cv2

        image, pts = picture_with_face((1024, 1024), (512.0, 560.0), 400.0)
        # No hair band: paint the top back to skin-ish grey.
        cv2.rectangle(image, (0, 0), (1024, 400), (200, 200, 200), -1)
        cx, cy = 512, 560
        cv2.rectangle(image, (cx - 60, cy + 120), (cx + 60, cy + 200), (60, 70, 90), -1)
        profile = portrait_profile(image, pts)
        assert profile.beard in ("stubble", "beard") and profile.chin_darkness > 0.08
        assert profile.forehead_reach == _OVAL_FOREHEAD_REACH
        assert profile.engine_options() == {"mask_chin_reach": "auto"}


class TestImportWithPreparation:
    def test_the_import_prepares_keeps_the_original_and_writes_the_profile(self, tmp_path):
        import cv2

        image, pts = picture_with_face((1024, 700), (512.0, 490.0), 400.0)
        photo = tmp_path / "friend.jpg"
        cv2.imwrite(str(photo), image)

        def inspect(path: Path, model_path: Path) -> PortraitReport:
            return good_report(
                width=1024,
                height=700,
                room_below_chin=0.03,
                points=pts,
                warnings=("little room below the chin (0.03 of the face height)",),
            )

        descriptor, report = import_portrait(
            tmp_path / "avatars",
            photo,
            name="Friend",
            model_path=tmp_path / "m",
            template=neural_manifest(),
            inspect=inspect,
        )
        root = descriptor.root
        portrait = cv2.imread(str(root / PORTRAIT_FILENAME))
        assert portrait.shape[0] > 700  # padded below the chin
        assert (root / f"{ORIGINAL_STEM}.jpg").is_file()
        profile = json.loads((root / PROFILE_FILENAME).read_text(encoding="utf-8"))
        assert profile["preparation"]["padding"][1] > 0
        assert profile["preparation"]["chin_drop_max_px"] >= 50
        assert profile["profile"]["beard"] == "bare"
        assert report.room_below_chin >= STANDARD_ROOM_BELOW_CHIN - 1e-6
        assert not any("room below the chin" in w and "little" in w for w in report.warnings)
        assert any(w.startswith("prepared:") for w in report.warnings)
        assert any(w.startswith("portrait:") for w in report.warnings)
        roles = {a.role: a for a in descriptor.manifest.assets}
        assert roles["source_original"].optional and roles["source_original"].path == "original.jpg"
        assert "mask_forehead_reach" in descriptor.manifest.engine_options
        assert descriptor.manifest.engine_options["mask_chin_reach"] == "auto"
        assert descriptor.manifest.engine_options["expression_gain"] == 1.8  # the template's

    def test_without_points_the_picture_is_written_as_it_is(self, tmp_path):
        import cv2

        image, _ = picture_with_face((300, 300), (150.0, 150.0), 100.0)
        photo = tmp_path / "plain.png"
        cv2.imwrite(str(photo), image)
        descriptor, _ = import_portrait(
            tmp_path / "avatars",
            photo,
            name="Plain",
            model_path=tmp_path / "m",
            template=neural_manifest(),
            inspect=lambda p, m: good_report(),
        )
        assert cv2.imread(str(descriptor.root / PORTRAIT_FILENAME)).shape == (300, 300, 3)
        assert not (descriptor.root / PROFILE_FILENAME).exists()
        assert len(descriptor.manifest.assets) == 1


@pytest.mark.skipif(
    not (MODEL.is_file() and PORTRAIT.is_file()),
    reason="needs the MediaPipe model and the owner's portrait on this machine",
)
class TestOnHisPortrait:
    def test_his_portrait_gains_its_room_and_reads_sensibly(self):
        import cv2

        report = inspect_portrait(PORTRAIT, MODEL)
        assert report.points is not None and report.room_below_chin < STANDARD_ROOM_BELOW_CHIN
        image = cv2.imread(str(PORTRAIT))
        out, pts, prep = prepare_portrait(image, report.points)
        assert prep.scale < 1.0 and prep.padding[1] > 0
        assert prep.room_below_chin >= STANDARD_ROOM_BELOW_CHIN - 1e-6
        assert prep.room.crop_inside and prep.room.chin_drop_max >= 50.0
        profile = portrait_profile(out, pts)
        assert profile.lips == "closed"
        assert 0.0 <= profile.forehead_reach <= _OVAL_FOREHEAD_REACH
