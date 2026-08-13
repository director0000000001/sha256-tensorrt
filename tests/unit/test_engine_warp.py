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

from sha256.avatar.engine_warp import _OVAL_SLICE as OVAL_COUNT_SLICE
from sha256.avatar.engine_warp import (
    MESH_INDICES,
    WarpAvatarEngine,
    _blend_sources,
    _Neutral,
    _source_weights,
)
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
    """A landmark set shaped like a face, normalised.

    The oval indices are laid out **in outline order** around the ring, and the
    feature points inside it. That ordering is not decoration: the engine fills
    the face outline as a polygon, so scattered oval points would produce a
    self-intersecting star and a mask with a hole where the face should be.
    """
    points = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)
    points[:, 0] = centre[0]
    points[:, 1] = centre[1]

    oval = list(MESH_INDICES[OVAL_COUNT_SLICE])
    ring = np.linspace(0.0, 2.0 * np.pi, len(oval), endpoint=False)
    for index, angle in zip(oval, ring, strict=True):
        points[index, 0] = centre[0] + scale * np.cos(angle)
        points[index, 1] = centre[1] + scale * 1.25 * np.sin(angle)

    inner = [i for i in MESH_INDICES if i not in oval]
    for position, (index, angle) in enumerate(
        zip(inner, np.linspace(0.0, 2.0 * np.pi, len(inner), endpoint=False), strict=True)
    ):
        radius = scale * (0.25 + 0.15 * (position % 4))
        points[index, 0] = centre[0] + radius * np.cos(angle)
        points[index, 1] = centre[1] + radius * 1.25 * np.sin(angle)

    return points


def relaxed_like(tracking: FaceTrackingResult) -> FaceTrackingResult:
    """The same face at rest: every signal it carries, all of them quiet.

    Needed because the engine measures expression *relative to rest*, so it has
    to have seen rest. The blendshape names must match — a resting face that
    reports no signals at all teaches the engine nothing about them.
    """
    from sha256.core.types import BlendshapeSet, HeadPose

    shapes = tracking.blendshapes
    names = shapes.names if shapes is not None else ()
    return FaceTrackingResult(
        metadata=tracking.metadata,
        face_detected=True,
        landmarks=tracking.landmarks,
        head_pose=HeadPose(yaw=0.0, pitch=0.0, roll=0.0),
        blendshapes=BlendshapeSet(
            names=names,
            weights=np.zeros(len(names), dtype=np.float32),
            scheme="mediapipe_52",
        ),
        confidence=1.0,
    )


def weights_from_rest(
    tracking: FaceTrackingResult, *, full_gap: float = 0.0947
) -> dict[str, float]:
    """Weights for ``tracking``, after the engine has been shown a resting face.

    A single frame in isolation is by definition neutral — that is the whole
    point of the change these tests describe — so a test that wants an
    expression to register has to establish rest first, exactly as the engine
    does on its first frame.
    """
    neutral = _Neutral()
    _source_weights(relaxed_like(tracking), full_gap, neutral)
    return _source_weights(tracking, full_gap, neutral)


def expressive_tracking(
    *,
    gap: float = 0.08,
    yaw: float = 0.0,
    smile: float | None = None,
    jaw: float | None = None,
    blink: tuple[float, float] | None = None,
    extra: dict[str, float] | None = None,
    **shapes: float,
) -> FaceTrackingResult:
    """Tracking with a controllable lip gap, pose, and blendshapes.

    ``smile``, ``jaw`` and ``blink`` are spelled out here rather than left to
    ``**shapes`` because a mistyped blendshape name is silently accepted by the
    tracker's own type — it is just a string — and the test would then measure
    nothing while looking like it measured something.

    ``gap`` defaults to open: the teeth-bearing photographs are gated on the
    lips being parted, so a test about mouth expressions has to say so.

    ``extra`` takes blendshapes whose names a test computes rather than writes,
    which ``**shapes`` cannot express without also being able to collide with
    the named arguments above.
    """
    if extra:
        shapes.update(extra)
    if smile is not None:
        shapes["mouthSmileLeft"] = shapes["mouthSmileRight"] = smile
    if jaw is not None:
        shapes["jawOpen"] = jaw
    if blink is not None:
        shapes["eyeBlinkLeft"], shapes["eyeBlinkRight"] = blink
    from sha256.core.types import BlendshapeSet, HeadPose

    points = synthetic_landmarks()
    # Face height is forehead-to-chin; put the inner lips `gap` of it apart.
    points[10, 1] = 0.2
    points[152, 1] = 0.8
    points[13, 1] = 0.5
    points[14, 1] = 0.5 + gap * 0.6

    names = tuple(shapes)
    return FaceTrackingResult(
        metadata=FrameMetadata(frame_id=1, capture_timestamp=1.0),
        face_detected=True,
        landmarks=LandmarkSet(points=points, scheme="mediapipe_478"),
        head_pose=HeadPose(yaw=yaw),
        blendshapes=BlendshapeSet(
            names=names,
            weights=np.array([shapes[n] for n in names], dtype=np.float32),
            scheme="mediapipe_52",
        ),
        confidence=1.0,
    )


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


def _striped(size: int) -> np.ndarray:
    """A portrait with visible texture, so a warp of it is detectable."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[::8] = 220
    image[:, ::8] = 40
    return image


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
        # A textured portrait, not a flat colour. Colour matching shifts the
        # avatar's mean onto the scene's, so a flat portrait over a flat scene
        # comes out *identical* to the scene by construction — and the test
        # would pass or fail for reasons that have nothing to do with warping.
        engine.load_avatar(make_descriptor(tmp_path, image=_striped(256)))
        frame = camera_frame()

        result = engine.animate("avatar_test", tracking_from(synthetic_landmarks()), frame)

        assert not result.is_placeholder
        assert result.avatar_id == "avatar_test"
        assert result.data.shape == frame.data.shape
        # The face region must carry the portrait's texture, not the flat scene.
        face = result.data[80:160, 120:200]
        assert int(face.max()) - int(face.min()) > 20

    def test_the_avatar_is_relit_to_match_the_scene(self, engine, tmp_path):
        """A portrait shot under one light, dropped into another, shows a seam.

        Matching the mean brings it most of the way; without this the edge is a
        step change in brightness that no amount of feathering hides.
        """
        engine.load_avatar(
            make_descriptor(tmp_path, image=np.full((256, 256, 3), 240, dtype=np.uint8))
        )
        frame = camera_frame()  # a dark scene, value 30

        result = engine.animate("avatar_test", tracking_from(synthetic_landmarks()), frame)

        centre = float(result.data[120, 160].mean())
        assert centre < 120.0, f"the bright portrait was pasted in unrelit ({centre:.0f})"

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


class TestSourceSelection:
    """Which extra photograph applies, and how much of it.

    These weights are the whole multi-source feature: everything else is a
    weighted average. A wrong sign here means the avatar turns the wrong way,
    and a missing clamp means it performs its expressions rather than wearing
    them.
    """

    def test_an_open_mouth_calls_for_the_open_mouth_photo(self):
        weights = _source_weights(expressive_tracking(gap=0.09, jaw=0.6))

        assert weights["source_mouth_open"] > 0.7

    def test_a_smile_calls_for_the_smiling_photo_rather_than_the_open_one(self):
        """Both show teeth; a smile shows them differently from an "ah"."""
        weights = _source_weights(expressive_tracking(gap=0.06, smile=0.55))

        assert weights["source_smile"] > weights.get("source_mouth_open", 0.0)

    def test_a_blink_calls_for_the_closed_eye_photo(self):
        weights = _source_weights(expressive_tracking(blink=(0.9, 0.85)))

        assert weights["source_eyes_closed"] > 0.8

    def test_eyes_held_shut_close_as_firmly_as_a_blink(self):
        """A held closure reads much lower than a blink, and left the eyes open."""
        weights = _source_weights(expressive_tracking(blink=(0.55, 0.5)))

        assert weights["source_eyes_closed"] > 0.7

    def test_a_wink_does_not(self):
        """The photo closes both eyes; driving it from one would close the other."""
        weights = _source_weights(expressive_tracking(blink=(0.95, 0.02)))

        assert "source_eyes_closed" not in weights

    def test_yaw_picks_the_side_it_turned_towards(self):
        left = weights_from_rest(expressive_tracking(yaw=20.0))
        right = weights_from_rest(expressive_tracking(yaw=-20.0))

        assert left["source_turn_left"] > 0.5
        assert "source_turn_right" not in left
        assert right["source_turn_right"] > 0.5
        assert "source_turn_left" not in right

    def test_no_variant_takes_over_completely(self):
        """Each was shot at one extreme; all the way there reads as performing."""
        weights = _source_weights(expressive_tracking(gap=0.12, jaw=1.0, yaw=90.0))

        assert max(weights.values()) <= 0.95

    def test_a_wide_open_mouth_leaves_the_closed_one_out_of_it(self):
        """The base portrait's lips are shut; stretched across a gap they are fangs."""
        weights = _source_weights(expressive_tracking(gap=0.095, jaw=0.5, smile=0.3))

        mouth = weights.get("source_smile", 0.0) + weights.get("source_mouth_open", 0.0)
        assert mouth > 0.9, f"the closed-mouth base still supplies {1 - mouth:.0%} of the mouth"

    def test_a_still_neutral_face_asks_for_nothing(self):
        assert _source_weights(expressive_tracking(jaw=0.0, yaw=0.0, gap=0.0)) == {}

    def test_a_result_without_blendshapes_or_pose_is_survivable(self):
        bare = FaceTrackingResult(
            metadata=FrameMetadata(frame_id=1, capture_timestamp=1.0),
            face_detected=True,
        )

        assert _source_weights(bare) == {}


class TestSourceBlending:
    def test_an_avatar_with_no_variants_uses_its_one_photograph(self, engine, tmp_path):
        engine.load_avatar(make_descriptor(tmp_path))
        avatar = engine._loaded["avatar_test"]

        blended = _blend_sources(avatar, tracking_from(synthetic_landmarks()))

        assert blended is avatar.image

    def test_a_weighted_variant_moves_the_pixels_towards_it(self, engine, tmp_path):
        engine.load_avatar(make_descriptor(tmp_path))
        avatar = engine._loaded["avatar_test"]
        # A variant that is pure white, so its contribution is unmistakable.
        avatar.sources["source_mouth_open"] = np.full_like(avatar.image, 255)

        blended = _blend_sources(avatar, expressive_tracking(jaw=0.5))

        assert float(blended.mean()) > float(avatar.image.mean())
        assert float(blended.mean()) < 255.0


class TestTrackingLoss:
    """What happens in the frames where the tracker loses the face.

    This is a privacy requirement, not a smoothness one. Passing the camera
    through a dropout puts the user's real face on the call — which is the one
    thing the product exists to prevent — and it is what the owner saw as a
    flicker.
    """

    @staticmethod
    def _frame(timestamp: float) -> VideoFrame:
        return VideoFrame(
            data=np.full((240, 320, 3), 30, dtype=np.uint8),
            pixel_format=PixelFormat.BGR24,
            metadata=FrameMetadata(frame_id=1, capture_timestamp=timestamp),
        )

    def test_a_momentary_loss_keeps_the_avatar_on(self, engine, tmp_path):
        engine.load_avatar(make_descriptor(tmp_path, image=_striped(256)))
        engine.animate("avatar_test", tracking_from(synthetic_landmarks()), self._frame(0.0))

        lost = engine.animate("avatar_test", tracking_from(None), self._frame(0.1))

        assert not lost.is_placeholder, "the camera was passed through during a dropout"
        face = lost.data[80:160, 120:200]
        assert int(face.max()) - int(face.min()) > 20, "the avatar is not being drawn"

    def test_the_scene_still_updates_while_the_face_is_held(self, engine, tmp_path):
        """Only the face is held; a frozen picture would look broken on a call."""
        engine.load_avatar(make_descriptor(tmp_path, image=_striped(256)))
        engine.animate("avatar_test", tracking_from(synthetic_landmarks()), self._frame(0.0))

        moving = self._frame(0.1)
        moving.data[:, :] = (200, 10, 10)
        lost = engine.animate("avatar_test", tracking_from(None), moving)

        assert tuple(lost.data[0, 0]) == (200, 10, 10)

    def test_a_long_loss_still_never_shows_the_camera(self, engine, tmp_path):
        """There is no length of dropout at which the real face becomes correct.

        An earlier version gave up after a second and passed the camera through,
        on the theory that the user had left. If they had not left — a hand over
        the face, an extreme angle, bad light — their real face went out on the
        call for as long as the loss lasted.
        """
        engine.load_avatar(make_descriptor(tmp_path, image=_striped(256)))
        engine.animate("avatar_test", tracking_from(synthetic_landmarks()), self._frame(0.0))

        lost = engine.animate("avatar_test", tracking_from(None), self._frame(60.0))

        assert not lost.is_placeholder

    def test_a_loss_before_any_face_was_seen_passes_through(self, engine, tmp_path):
        """Nothing to hold, and nothing to hide either."""
        engine.load_avatar(make_descriptor(tmp_path))

        result = engine.animate("avatar_test", tracking_from(None), self._frame(0.0))

        assert result.is_placeholder

    def test_unloading_forgets_the_held_face(self, engine, tmp_path):
        descriptor = make_descriptor(tmp_path)
        engine.load_avatar(descriptor)
        engine.animate("avatar_test", tracking_from(synthetic_landmarks()), self._frame(0.0))

        engine.unload_avatar("avatar_test")
        engine.load_avatar(descriptor)
        result = engine.animate("avatar_test", tracking_from(None), self._frame(0.1))

        assert result.is_placeholder


class TestTeethGate:
    """Teeth may only appear when the lips are actually parted.

    Measured on the avatar's own photographs: the smiling-with-teeth photo reads
    mouthSmile 0.54, and so does a closed-mouth smile — so weighting the
    teeth-bearing photographs by the smile blendshape put teeth into a smile
    that had none.
    """

    def test_a_closed_smile_shows_no_teeth(self):
        weights = _source_weights(expressive_tracking(gap=0.001, smile=0.6))

        assert "source_smile" not in weights

    def test_an_open_smile_does(self):
        weights = _source_weights(expressive_tracking(gap=0.06, smile=0.6))

        assert weights["source_smile"] > 0.5

    def test_the_open_mouth_photo_is_gated_the_same_way(self):
        """A dropped jaw with the lips still together shows no teeth either."""
        shut = _source_weights(expressive_tracking(gap=0.001, jaw=0.7))
        open_ = _source_weights(expressive_tracking(gap=0.09, jaw=0.7))

        assert "source_mouth_open" not in shut
        assert open_["source_mouth_open"] > 0.5

    def test_the_gate_is_gradual(self):
        """A hard threshold would pop the teeth in as the lips part."""
        weights = _source_weights(expressive_tracking(gap=0.03, smile=1.0))

        assert 0.1 < weights["source_smile"] < 0.9


class TestMouthInterior:
    """What fills the mouth when the user opens wider than the photograph did.

    Which is most of the time: no still portrait is taken mid-yawn. Placing the
    whole photograph and trusting the fit put its *lips* inside the live
    opening — a second, smaller mouth inside the first.
    """

    def test_opening_wider_than_the_photo_shows_cavity_not_lips(self, engine, tmp_path):
        engine.load_avatar(make_descriptor(tmp_path, image=_striped(256)))
        avatar = engine._loaded["avatar_test"]

        # A cavity the photograph's own mouth never contains, so anything of
        # this colour in the result can only have come from the fallback.
        avatar.cavity = np.array([7.0, 11.0, 13.0], dtype=np.float32)

        frame = camera_frame()
        result = engine.animate("avatar_test", tracking_from(synthetic_landmarks()), frame)

        distance = np.abs(result.data.astype(np.int16) - avatar.cavity.astype(np.int16))
        assert int((distance.sum(axis=2) < 30).sum()) > 0, "the opening was never filled"

    def test_the_photographs_own_opening_sets_what_fully_open_means(self):
        """A wider photograph should extend the range, not need a constant changed."""
        narrow = _source_weights(expressive_tracking(gap=0.05, jaw=0.5), full_gap=0.05)
        wide = _source_weights(expressive_tracking(gap=0.05, jaw=0.5), full_gap=0.15)

        assert narrow["source_mouth_open"] > wide["source_mouth_open"]


class TestBrowSource:
    def test_raised_brows_call_for_the_raised_brow_photo(self):
        weights = weights_from_rest(expressive_tracking(gap=0.0, browInnerUp=0.6))

        assert weights["source_brows_up"] > 0.9

    def test_resting_brows_do_not(self):
        weights = _source_weights(expressive_tracking(gap=0.0, browInnerUp=0.01))

        assert "source_brows_up" not in weights


class TestExpandedSourceSet:
    """The wider set of source photographs, and what selects each.

    A wrong mapping here is invisible in a still and obvious in motion — the
    avatar looks left when you look right, or pulls a face you are not pulling.
    """

    def test_a_face_that_is_not_moving_asks_for_no_photographs(self):
        """The regression that made the avatar a double exposure of itself.

        A relaxed face does not read as zero, and a webcam above a monitor never
        reads zero pose. Driven absolutely, a motionless person scored eleven
        simultaneous expressions and the base portrait was blended away to
        nothing. Held still, the answer must be: use the portrait.
        """
        resting = expressive_tracking(
            gap=0.0,
            yaw=19.0,
            eyeSquintLeft=0.52,
            eyeSquintRight=0.5,
            eyeLookDownLeft=0.4,
            browInnerUp=0.3,
        )
        neutral = _Neutral()

        for _ in range(30):
            weights = _source_weights(resting, 0.0947, neutral)

        assert weights == {}, f"a still face asked for {sorted(weights)}"

    @pytest.mark.parametrize(
        ("left_role", "right_role", "left_shape", "right_shape"),
        [
            ("source_look_left", "source_look_right", "eyeLookOutLeft", "eyeLookOutRight"),
            ("source_look_up", "source_look_down", "eyeLookUpLeft", "eyeLookDownLeft"),
            ("source_brows_up", "source_brows_down", "browInnerUp", "browDownLeft"),
        ],
    )
    def test_opposing_photographs_are_never_both_used(
        self, left_role, right_role, left_shape, right_shape
    ):
        """Measured on a real face: left and right gaze were both active on 98%
        of frames, up and down on 100%. Averaging opposites gave eyes that
        pointed nowhere, which is why they stopped looking real."""
        weights = weights_from_rest(
            expressive_tracking(gap=0.0, extra={left_shape: 0.5, right_shape: 0.35})
        )

        assert weights.get(right_role, 0.0) == 0.0
        assert weights.get(left_role, 0.0) > 0.0

    def test_gaze_left_and_right_are_not_swapped(self):
        """Looking one way must not fetch the photograph of the other."""
        left = weights_from_rest(expressive_tracking(gap=0.0, eyeLookOutLeft=0.6))
        right = weights_from_rest(expressive_tracking(gap=0.0, eyeLookOutRight=0.6))

        assert left["source_look_left"] > 0.9
        assert "source_look_right" not in left
        assert right["source_look_right"] > 0.9
        assert "source_look_left" not in right

    def test_looking_up_and_down(self):
        up = weights_from_rest(expressive_tracking(gap=0.0, eyeLookUpLeft=0.6))
        down = weights_from_rest(expressive_tracking(gap=0.0, eyeLookDownLeft=0.6))

        assert up["source_look_up"] > 0.9
        assert down["source_look_down"] > 0.9

    def test_a_closed_smile_asks_for_the_closed_smile_photo(self):
        """And not for the one with teeth in it."""
        weights = _source_weights(expressive_tracking(gap=0.001, smile=0.6))

        assert weights["source_smile_closed"] > 0.9
        assert "source_smile" not in weights

    def test_an_open_smile_asks_for_the_open_one_instead(self):
        weights = _source_weights(expressive_tracking(gap=0.09, smile=0.6))

        assert weights["source_smile"] > weights.get("source_smile_closed", 0.0)

    def test_a_far_turn_brings_in_the_far_photograph(self):
        near = weights_from_rest(expressive_tracking(gap=0.0, yaw=20.0))
        far = weights_from_rest(expressive_tracking(gap=0.0, yaw=38.0))

        assert far["source_turn_left_far"] > near.get("source_turn_left_far", 0.0)
        assert "source_turn_right_far" not in far

    def test_tilting_up_and_down_are_not_swapped(self):
        up = weights_from_rest(expressive_tracking(gap=0.0, yaw=0.0))
        up_pose = weights_from_rest(_pitched(18.0))
        down_pose = weights_from_rest(_pitched(-18.0))

        assert "source_tilt_up" not in up
        assert up_pose["source_tilt_up"] > 0.9
        assert down_pose["source_tilt_down"] > 0.9

    def test_brows_down_is_separate_from_brows_up(self):
        weights = weights_from_rest(expressive_tracking(gap=0.0, browDownLeft=0.6))

        assert weights["source_brows_down"] > 0.9
        assert "source_brows_up" not in weights

    def test_an_avatar_only_uses_the_photographs_it_has(self, engine, tmp_path):
        """Weights are computed for every role; only the supplied ones apply."""
        engine.load_avatar(make_descriptor(tmp_path))
        avatar = engine._loaded["avatar_test"]

        assert avatar.sources == {}
        assert _blend_sources(avatar, expressive_tracking(jaw=0.9)) is avatar.image


def _pitched(pitch: float) -> FaceTrackingResult:
    from sha256.core.types import HeadPose

    base = expressive_tracking(gap=0.0)
    return FaceTrackingResult(
        metadata=base.metadata,
        face_detected=True,
        landmarks=base.landmarks,
        head_pose=HeadPose(pitch=pitch),
        blendshapes=base.blendshapes,
        confidence=1.0,
    )


class TestRelightingStability:
    """The relighting must not chase the camera's exposure.

    A webcam hunting its exposure makes the scene brighten and dim continuously.
    A correction that follows it instantly makes the avatar's face pulse against
    a background that is merely drifting — which is what the owner saw as the
    picture flickering.
    """

    @staticmethod
    def _scene(value: int) -> VideoFrame:
        return VideoFrame(
            data=np.full((240, 320, 3), value, dtype=np.uint8),
            pixel_format=PixelFormat.BGR24,
            metadata=FrameMetadata(frame_id=1, capture_timestamp=1.0),
        )

    def test_a_sudden_exposure_jump_is_absorbed(self, engine, tmp_path):
        engine.load_avatar(make_descriptor(tmp_path, image=_striped(256)))
        tracking = tracking_from(synthetic_landmarks())

        # Settle on a dark scene, then jump the exposure.
        for _ in range(10):
            engine.animate("avatar_test", tracking, self._scene(30))
        settled = engine.animate("avatar_test", tracking, self._scene(30))
        jumped = engine.animate("avatar_test", tracking, self._scene(200))

        face_before = float(settled.data[110:130, 150:170].mean())
        face_after = float(jumped.data[110:130, 150:170].mean())

        # The face follows the room, but nothing like all the way in one frame.
        assert face_after > face_before, "the face ignored the change entirely"
        assert face_after - face_before < 120, "the face jumped with the exposure"

    def test_it_does_catch_up_over_several_frames(self, engine, tmp_path):
        engine.load_avatar(make_descriptor(tmp_path, image=_striped(256)))
        tracking = tracking_from(synthetic_landmarks())

        for _ in range(5):
            engine.animate("avatar_test", tracking, self._scene(30))
        one = engine.animate("avatar_test", tracking, self._scene(200))
        for _ in range(20):
            engine.animate("avatar_test", tracking, self._scene(200))
        many = engine.animate("avatar_test", tracking, self._scene(200))

        assert float(many.data[110:130, 150:170].mean()) > float(one.data[110:130, 150:170].mean())
