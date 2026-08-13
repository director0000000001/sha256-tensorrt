"""Tests for the landmark-warp avatar engine.

No real photograph and no real tracker: a synthetic portrait and a synthetic
landmark set exercise the geometry, which is where this engine can be wrong in
ways that are hard to see. The visual quality question is answered by looking at
it, not by asserting on pixels.

What matters here is that it fails clearly when an avatar is unusable — a
missing portrait, a portrait with no face in it, the wrong engine id — because
those are the errors a user will actually hit, and each one has an obvious cause
that the message should name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sha256.avatar.engine_warp import MESH_INDICES, WarpAvatarEngine
from sha256.core.avatar import AvatarDescriptor, AvatarManifest
from sha256.core.enums import PixelFormat
from sha256.core.errors import AvatarLoadError
from sha256.core.interfaces import AvatarEngine
from sha256.core.types import FaceTrackingResult, FrameMetadata, LandmarkSet, VideoFrame

cv2 = pytest.importorskip("cv2", reason="OpenCV is required for the warp engine")

LANDMARK_COUNT = 478


def make_descriptor(
    tmp_path: Path,
    *,
    engine_id: str = "warp",
    portrait: str | None = "portrait.png",
    write_image: bool = True,
    image: np.ndarray | None = None,
) -> AvatarDescriptor:
    root = tmp_path / "avatar_test"
    root.mkdir(exist_ok=True)

    assets: list[dict[str, Any]] = []
    if portrait is not None:
        assets.append({"path": portrait, "role": "source_portrait"})
        if write_image:
            data = image if image is not None else np.full((256, 256, 3), 90, dtype=np.uint8)
            cv2.imwrite(str(root / portrait), data)

    manifest = AvatarManifest.model_validate(
        json.loads(
            json.dumps(
                {
                    "manifest_version": 1,
                    "id": "avatar_test",
                    "name": "Test",
                    "assets": assets,
                    "engine": {"engine_id": engine_id},
                }
            )
        )
    )
    return AvatarDescriptor(manifest=manifest, root=root)


def synthetic_landmarks(*, centre: tuple[float, float] = (0.5, 0.5), scale: float = 0.2):
    """A ring of landmarks standing in for a face, normalised."""
    angles = np.linspace(0.0, 2.0 * np.pi, LANDMARK_COUNT, endpoint=False)
    points = np.stack(
        [
            centre[0] + scale * np.cos(angles),
            centre[1] + scale * 1.3 * np.sin(angles),
            np.zeros(LANDMARK_COUNT),
        ],
        axis=1,
    ).astype(np.float32)
    return points


def tracking_from(points: np.ndarray | None) -> FaceTrackingResult:
    metadata = FrameMetadata(frame_id=1, capture_timestamp=1.0)
    if points is None:
        return FaceTrackingResult.no_face(metadata, tracker_id="test")
    return FaceTrackingResult(
        metadata=metadata,
        face_detected=True,
        landmarks=LandmarkSet(points=points, scheme="mediapipe_478"),
        confidence=1.0,
        tracker_id="test",
    )


def camera_frame(width: int = 320, height: int = 240) -> VideoFrame:
    data = np.full((height, width, 3), 30, dtype=np.uint8)
    return VideoFrame(
        data=data,
        pixel_format=PixelFormat.BGR24,
        metadata=FrameMetadata(frame_id=1, capture_timestamp=1.0),
    )


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch):
    """An engine whose portrait landmark detection is stubbed.

    Running the real detector would make these tests need the model bundle and a
    photograph with a real face in it. The geometry under test is the same
    either way.
    """
    built = WarpAvatarEngine()

    def fake_landmarks(image: np.ndarray, avatar_id: str) -> np.ndarray:
        del avatar_id
        height, width = image.shape[:2]
        points = synthetic_landmarks()
        subset = points[list(MESH_INDICES), :2].copy()
        subset[:, 0] *= width
        subset[:, 1] *= height
        result: np.ndarray = subset.astype(np.float32)
        return result

    monkeypatch.setattr(built, "_landmarks_of", fake_landmarks)
    built.start()
    return built


class TestProtocolConformance:
    def test_satisfies_the_avatar_engine_protocol(self):
        assert isinstance(WarpAvatarEngine(), AvatarEngine)

    def test_reports_its_identity(self):
        built = WarpAvatarEngine()
        assert built.engine_id == "warp"
        assert built.backend_id == "warp"

    def test_holds_nothing_on_the_gpu(self):
        """None means "not measured"; zero would read as a measurement."""
        assert WarpAvatarEngine().estimated_vram_mb("anything") is None


class TestLoading:
    def test_loads_an_avatar_and_builds_a_mesh(self, engine, tmp_path):
        engine.load_avatar(make_descriptor(tmp_path))

        assert engine.loaded_avatars == ("avatar_test",)

    def test_loading_twice_is_idempotent(self, engine, tmp_path):
        descriptor = make_descriptor(tmp_path)
        engine.load_avatar(descriptor)
        engine.load_avatar(descriptor)

        assert engine.loaded_avatars == ("avatar_test",)

    def test_unloading_releases_it(self, engine, tmp_path):
        engine.load_avatar(make_descriptor(tmp_path))
        engine.unload_avatar("avatar_test")

        assert engine.loaded_avatars == ()

    def test_stop_unloads_everything(self, engine, tmp_path):
        engine.load_avatar(make_descriptor(tmp_path))
        engine.stop()

        assert engine.loaded_avatars == ()

    def test_an_avatar_for_another_engine_is_refused(self, engine, tmp_path):
        """A neural avatar must not silently render as a warped photo."""
        with pytest.raises(AvatarLoadError, match="requires engine"):
            engine.load_avatar(make_descriptor(tmp_path, engine_id="liveportrait"))

    def test_an_avatar_with_no_portrait_says_so(self, engine, tmp_path):
        with pytest.raises(AvatarLoadError, match="source_portrait"):
            engine.load_avatar(make_descriptor(tmp_path, portrait=None))

    def test_a_missing_portrait_file_names_the_path(self, engine, tmp_path):
        with pytest.raises(AvatarLoadError, match="portrait not found"):
            engine.load_avatar(make_descriptor(tmp_path, write_image=False))

    def test_a_file_that_is_not_an_image_says_so(self, engine, tmp_path):
        descriptor = make_descriptor(tmp_path, write_image=False)
        (descriptor.root / "portrait.png").write_text("not an image", encoding="utf-8")

        with pytest.raises(AvatarLoadError, match="could not be read as an image"):
            engine.load_avatar(descriptor)

    def test_a_portrait_with_no_face_explains_what_is_needed(self, tmp_path, monkeypatch):
        """The most likely user error, so the message has to be actionable."""
        built = WarpAvatarEngine()
        built.start()

        def no_face(image: np.ndarray, avatar_id: str) -> np.ndarray:
            del image
            msg = (
                f"avatar {avatar_id!r}: no face was found in the portrait. "
                f"It needs a clear, front-facing photograph of one face."
            )
            raise AvatarLoadError(msg)

        monkeypatch.setattr(built, "_landmarks_of", no_face)

        with pytest.raises(AvatarLoadError, match="front-facing"):
            built.load_avatar(make_descriptor(tmp_path))


class TestAnimation:
    def test_animating_an_unloaded_avatar_is_an_error(self, engine):
        with pytest.raises(AvatarLoadError, match="not loaded"):
            engine.animate("absent", tracking_from(synthetic_landmarks()), camera_frame())

    def test_the_avatar_is_composited_over_the_camera_frame(self, engine, tmp_path):
        engine.load_avatar(
            make_descriptor(tmp_path, image=np.full((256, 256, 3), 200, dtype=np.uint8))
        )
        frame = camera_frame()

        result = engine.animate("avatar_test", tracking_from(synthetic_landmarks()), frame)

        assert not result.is_placeholder
        assert result.avatar_id == "avatar_test"
        assert result.data.shape == frame.data.shape
        # The bright portrait must have replaced part of the dark scene.
        assert int(result.data.max()) > int(frame.data.max())

    def test_the_scene_outside_the_face_is_untouched(self, engine, tmp_path):
        engine.load_avatar(
            make_descriptor(tmp_path, image=np.full((256, 256, 3), 200, dtype=np.uint8))
        )
        frame = camera_frame()

        result = engine.animate("avatar_test", tracking_from(synthetic_landmarks()), frame)

        assert np.array_equal(result.data[0, 0], frame.data[0, 0])
        assert np.array_equal(result.data[-1, -1], frame.data[-1, -1])

    def test_losing_the_face_passes_the_scene_through(self, engine, tmp_path):
        """Expected constantly; the stream must not stop or flash."""
        engine.load_avatar(make_descriptor(tmp_path))
        frame = camera_frame()

        result = engine.animate("avatar_test", tracking_from(None), frame)

        assert result.is_placeholder
        assert np.array_equal(result.data, frame.data)

    def test_the_frame_identity_is_preserved(self, engine, tmp_path):
        """End-to-end latency is measured from it; losing it loses the metric."""
        engine.load_avatar(make_descriptor(tmp_path))
        frame = camera_frame()

        result = engine.animate("avatar_test", tracking_from(synthetic_landmarks()), frame)

        assert result.metadata.frame_id == frame.metadata.frame_id
        assert result.metadata.capture_timestamp == frame.metadata.capture_timestamp

    def test_a_face_partly_outside_the_frame_does_not_crash(self, engine, tmp_path):
        """Walking out of shot must not take the pipeline with it."""
        engine.load_avatar(make_descriptor(tmp_path))

        for centre in ((0.02, 0.5), (0.98, 0.5), (0.5, 0.02), (0.5, 0.98)):
            result = engine.animate(
                "avatar_test",
                tracking_from(synthetic_landmarks(centre=centre)),
                camera_frame(),
            )
            assert result.data.shape == (240, 320, 3)

    def test_a_tiny_face_does_not_crash(self, engine, tmp_path):
        engine.load_avatar(make_descriptor(tmp_path))

        result = engine.animate(
            "avatar_test", tracking_from(synthetic_landmarks(scale=0.002)), camera_frame()
        )

        assert result.data.shape == (240, 320, 3)

    def test_rgb_frames_are_accepted_too(self, engine, tmp_path):
        engine.load_avatar(make_descriptor(tmp_path))
        frame = camera_frame()
        rgb = VideoFrame(data=frame.data, pixel_format=PixelFormat.RGB24, metadata=frame.metadata)

        result = engine.animate("avatar_test", tracking_from(synthetic_landmarks()), rgb)

        assert result.pixel_format is PixelFormat.RGB24

    def test_a_format_it_cannot_composite_over_is_refused(self, engine, tmp_path):
        engine.load_avatar(make_descriptor(tmp_path))
        frame = VideoFrame(
            data=np.zeros((240, 320, 3), dtype=np.uint8),
            pixel_format=PixelFormat.RGBA32,
            metadata=FrameMetadata(frame_id=1, capture_timestamp=1.0),
        )

        with pytest.raises(AvatarLoadError, match="cannot composite"):
            engine.animate("avatar_test", tracking_from(synthetic_landmarks()), frame)
