"""Adding an avatar from a photograph (day 15): the checks, the directory,
the manifest copied from the neural template -- and nothing half-made when
a check fails. The photograph's inspection is faked here (MediaPipe needs
its model file); one test runs the real inspector on the owner's portrait
when both the model and the portrait are on this machine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sha256.avatar.importer import (
    PORTRAIT_FILENAME,
    AvatarImportError,
    PortraitReport,
    import_portrait,
    inspect_portrait,
    neural_template,
    slug_for,
)
from sha256.core.avatar import AvatarDescriptor, AvatarManifest
from tests.conftest import make_manifest_dict

REPO_ROOT = Path(__file__).resolve().parents[2]


def good_report(**overrides) -> PortraitReport:
    fields = {
        "width": 1024,
        "height": 1024,
        "face_found": True,
        "yaw": 1.0,
        "pitch": -2.0,
        "roll": 0.5,
        "face_height_px": 420.0,
        "room_below_chin": 0.5,
        "eye_ratio": 0.3,
        "mouth_ratio": 0.01,
    }
    fields.update(overrides)
    return PortraitReport(**fields)


def photo(tmp_path: Path) -> Path:
    import cv2

    path = tmp_path / "me.jpg"
    image = np.full((64, 48, 3), (30, 90, 200), dtype=np.uint8)
    cv2.imwrite(str(path), image)
    return path


def neural_manifest() -> AvatarManifest:
    return AvatarManifest.model_validate(
        make_manifest_dict(
            "my_neural",
            engine={
                "engine_id": "liveportrait",
                "estimated_vram_mb": 4500,
                "device_preference": "cuda",
            },
            rendering={"output_width": 1280, "output_height": 720},
            engine_options={"crop_vy_ratio": -0.06, "expression_gain": 1.8, "mouth_rest": "anchor"},
            assets=[{"path": "portrait.png", "role": "source_portrait"}],
        )
    )


class TestSlug:
    def test_lowercase_letters_digits_and_underscores(self, tmp_path):
        assert slug_for("Giorgi's Face #2", tmp_path) == "giorgi_s_face_2"
        assert slug_for("  ", tmp_path) == "avatar"
        assert slug_for("Al", tmp_path) == "al_avatar"

    def test_unique_in_the_library(self, tmp_path):
        (tmp_path / "giorgi").mkdir()
        assert slug_for("Giorgi", tmp_path) == "giorgi_2"
        (tmp_path / "giorgi_2").mkdir()
        assert slug_for("Giorgi", tmp_path) == "giorgi_3"


class TestTemplate:
    def test_the_active_neural_avatar_else_the_first_else_none(self, tmp_path):
        null = AvatarDescriptor(
            manifest=AvatarManifest.model_validate(make_manifest_dict("nul")), root=tmp_path
        )
        a = AvatarDescriptor(manifest=neural_manifest(), root=tmp_path)
        b_manifest = neural_manifest().model_copy(
            update={"id": "other", "engine_options": {"x": 1}}
        )
        b = AvatarDescriptor(manifest=b_manifest, root=tmp_path)
        assert neural_template([null], None) is None
        assert neural_template([null, a, b], None) is a.manifest
        assert neural_template([null, a, b], "other") is b.manifest
        assert neural_template([null, a, b], "nul") is a.manifest


class TestImport:
    def test_writes_the_portrait_and_a_manifest_copied_from_the_template(self, tmp_path):
        library = tmp_path / "avatars"
        seen: list[Path] = []

        def inspect(path: Path, model_path: Path) -> PortraitReport:
            seen.append(path)
            return good_report(warnings=("the face is small (300 px tall)",))

        descriptor, report = import_portrait(
            library,
            photo(tmp_path),
            name="Giorgi",
            model_path=tmp_path / "model.task",
            template=neural_manifest(),
            inspect=inspect,
        )

        assert seen == [tmp_path / "me.jpg"]
        assert descriptor.id == "giorgi" and descriptor.root == library / "giorgi"
        assert (library / "giorgi" / PORTRAIT_FILENAME).is_file()
        manifest = json.loads((library / "giorgi" / "metadata.json").read_text(encoding="utf-8"))
        assert manifest["id"] == "giorgi" and manifest["name"] == "Giorgi"
        assert manifest["engine"]["engine_id"] == "liveportrait"
        assert manifest["engine_options"] == {
            "crop_vy_ratio": -0.06,
            "expression_gain": 1.8,
            "mouth_rest": "anchor",
        }
        assert manifest["rendering"]["output_width"] == 1280
        (asset,) = manifest["assets"]
        assert (asset["path"], asset["role"], asset["optional"]) == (
            "portrait.png",
            "source_portrait",
            False,
        )
        assert "Added from the application" in manifest["description"]
        assert report.warnings == ("the face is small (300 px tall)",)
        # The written manifest reads back as a valid one.
        AvatarManifest.model_validate(manifest)

    def test_without_a_template_the_engine_is_neural_with_bare_defaults(self, tmp_path):
        descriptor, _ = import_portrait(
            tmp_path / "avatars",
            photo(tmp_path),
            name="Nino",
            model_path=tmp_path / "m",
            template=None,
            inspect=lambda p, m: good_report(),
        )
        assert descriptor.manifest.engine.engine_id == "liveportrait"
        assert descriptor.manifest.engine_options == {}

    def test_a_hard_failure_writes_nothing(self, tmp_path):
        library = tmp_path / "avatars"
        with pytest.raises(AvatarImportError, match="straight at the camera"):
            import_portrait(
                library,
                photo(tmp_path),
                name="Turned",
                model_path=tmp_path / "m",
                template=neural_manifest(),
                inspect=lambda p, m: good_report(
                    yaw=30.0, errors=("the face must look straight at the camera",)
                ),
            )
        assert not (library / "turned").exists()

    def test_a_missing_photograph_or_an_empty_name_is_refused(self, tmp_path):
        with pytest.raises(AvatarImportError, match="not a file"):
            import_portrait(tmp_path, tmp_path / "nope.png", name="x", model_path=tmp_path)
        with pytest.raises(AvatarImportError, match="needs a name"):
            import_portrait(tmp_path, photo(tmp_path), name="   ", model_path=tmp_path)

    def test_two_imports_of_the_same_name_get_distinct_ids(self, tmp_path):
        library = tmp_path / "avatars"
        first, _ = import_portrait(
            library,
            photo(tmp_path),
            name="Same",
            model_path=tmp_path,
            inspect=lambda p, m: good_report(),
        )
        second, _ = import_portrait(
            library,
            photo(tmp_path),
            name="Same",
            model_path=tmp_path,
            inspect=lambda p, m: good_report(),
        )
        assert (first.id, second.id) == ("same", "same_2")


class TestReport:
    def test_describe_says_what_was_found(self):
        report = good_report(errors=("the eyes must be open",), warnings=("the mouth is open",))
        text = report.describe()
        assert "face 420 px tall" in text and "error: the eyes must be open" in text
        assert "note: the mouth is open" in text and not report.ok
        assert PortraitReport(width=8, height=8, face_found=False).describe() == "no face found"
        assert "skin" not in text  # none read

    def test_describe_says_the_photographs_skin_when_read(self):
        """Day 23: the photograph's skin colour, for the comparison with
        the person's camera reading."""
        from sha256.avatar.skin_tone import SkinReading

        report = good_report(skin=SkinReading((110.0, 150.0, 198.0)))
        assert "skin light-medium skin with a warm, golden undertone, sRGB #C6966E" in (
            report.describe()
        )


@pytest.mark.skipif(
    not (REPO_ROOT / "models" / "mediapipe" / "face_landmarker.task").is_file()
    or not (REPO_ROOT / "avatars" / "my_avatar_neural" / "portrait.png").is_file(),
    reason="needs the MediaPipe model and the owner's portrait on this machine",
)
class TestRealInspector:
    def test_the_owners_portrait_passes(self):
        from sha256.tracking.mediapipe_tracker import DEFAULT_MODEL_RELATIVE_PATH

        report = inspect_portrait(
            REPO_ROOT / "avatars" / "my_avatar_neural" / "portrait.png",
            REPO_ROOT / "models" / DEFAULT_MODEL_RELATIVE_PATH,
        )
        assert report.face_found and report.ok, report.describe()
        assert report.face_height_px > 300
        assert abs(report.yaw) < 5 and abs(report.pitch) < 8
        assert report.mouth_ratio < 0.05
