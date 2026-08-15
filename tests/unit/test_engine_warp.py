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
    _build_regions,
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


def set_eye_opening(points: np.ndarray, left: float, right: float) -> None:
    """Place both eyelid rings at a given openness, 1.0 open and 0.0 shut.

    Eye closure is measured on the **eyelids**, not on ``eyeBlink``: the
    blendshape is trained on blinks and reads 0.14 on eyes held shut, which had
    the avatar squinting back at a person whose eyes were closed. So a test that
    wants a closed eye has to close one.
    """
    from sha256.avatar.engine_warp import _LEFT_EYE, _RIGHT_EYE

    for ring, openness, x in ((_LEFT_EYE, left, 0.42), (_RIGHT_EYE, right, 0.58)):
        angles = np.linspace(0.0, 2.0 * np.pi, len(ring), endpoint=False)
        # A lid ring 0.06 wide and, fully open, a third of that tall.
        points[list(ring), 0] = x + 0.03 * np.cos(angles)
        points[list(ring), 1] = 0.44 + 0.01 * openness * np.sin(angles)


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

    # Open eyes are part of "at rest", and closure is measured on the lids, so
    # a resting frame that kept the expression's shut eyelids would teach the
    # engine that shut eyes are this person's normal.
    landmarks = tracking.landmarks
    if landmarks is not None:
        from sha256.core.types import LandmarkSet as _LandmarkSet

        points = landmarks.points.copy()
        set_eye_opening(points, 1.0, 1.0)
        landmarks = _LandmarkSet(points=points, scheme=landmarks.scheme)

    return FaceTrackingResult(
        metadata=tracking.metadata,
        face_detected=True,
        landmarks=landmarks,
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
    # `blink` closes the eyelids as well as raising the blendshape. Closure is
    # measured on the lids now, so a test asking for a blink has to produce one.
    left, right = blink if blink is not None else (0.0, 0.0)
    set_eye_opening(points, 1.0 - left, 1.0 - right)

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


class TestMeshLayout:
    """The slices into the mesh are counted, so changing a group can move them.

    ``_OVAL_SLICE``, ``_BROW_SLICE`` and ``_INNER_LIP_SLICE`` are built by adding
    up the lengths of the groups that come before them. That is only correct
    while none of those groups share a landmark, and it fails *silently*: the
    face outline would still be filled as a polygon, just over the wrong points.
    Enlarging the eyes from six landmarks to the full lid ring is exactly the
    kind of edit that would have found out the hard way.
    """

    def test_no_landmark_appears_twice(self):
        assert len(MESH_INDICES) == len(set(MESH_INDICES))

    @pytest.mark.parametrize(
        ("slice_name", "group_names"),
        [
            ("_OVAL_SLICE", ("_FACE_OVAL",)),
            ("_BROW_SLICE", ("_LEFT_BROW", "_RIGHT_BROW")),
            ("_INNER_LIP_SLICE", ("_INNER_LIPS",)),
        ],
    )
    def test_each_slice_lands_on_the_group_it_names(self, slice_name, group_names):
        from sha256.avatar import engine_warp

        where = getattr(engine_warp, slice_name)
        expected = tuple(index for name in group_names for index in getattr(engine_warp, name))
        assert tuple(MESH_INDICES[where]) == expected

    def test_each_eye_carries_its_whole_lid(self):
        """Six points leave the eye anchored to the cheek by long triangles that
        sweep as the lid closes — the wedge above a blink. The ring terminates
        them."""
        from sha256.avatar.engine_warp import _EYE_RINGS

        assert [len(ring) for ring in _EYE_RINGS] == [16, 16]


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
        weights = weights_from_rest(expressive_tracking(blink=(0.9, 0.85)))

        assert weights["source_eyes_closed"] > 0.8

    def test_eyes_held_shut_close_as_firmly_as_a_blink(self):
        """Shut lids are shut lids, whatever ``eyeBlink`` happens to say.

        The blendshape is trained on blinks, which are fast. Measured on the
        owner's recording it peaks at 0.66 on a blink and reads **0.14** on two
        seconds of held closure — lower than plenty of frames with the eyes
        plainly open. So the avatar squinted back at a person whose eyes were
        closed. Closure is measured on the eyelids now, and the eyelids are in
        the same place either way.
        """
        blinking = weights_from_rest(
            expressive_tracking(blink=(0.95, 0.95), eyeBlinkLeft=0.66, eyeBlinkRight=0.66)
        )
        holding = weights_from_rest(
            expressive_tracking(blink=(0.95, 0.95), eyeBlinkLeft=0.14, eyeBlinkRight=0.14)
        )

        assert blinking["source_eyes_closed"] > 0.9
        assert holding["source_eyes_closed"] == pytest.approx(blinking["source_eyes_closed"])

    def test_a_closing_eye_does_not_also_squint(self):
        """One eye, one opening, and the shut photograph must win it.

        MediaPipe reports a squint *and* a blink on a blink, because narrowing
        is how closing begins: measured on the owner's recording, `eyeSquint`
        ran 0.83-0.94 at the moment `eyeBlink` peaked at 0.66. Both then shared
        the eye region's budget, so the closed-eye photograph applied at half
        strength and the eye never actually shut. That is the owner's "the eyes
        still work very badly".
        """
        weights = weights_from_rest(
            expressive_tracking(blink=(0.95, 0.95), eyeSquintLeft=0.9, eyeSquintRight=0.9)
        )

        assert weights["source_eyes_closed"] > 0.9
        assert weights.get("source_squint", 0.0) < 0.05

    def test_ordinary_lid_movement_asks_for_no_eye_photograph(self):
        """The one that had survived five rounds of looking at single frames.

        A glance aside narrows the lids by up to 47% of their opening with no
        blink involved. Measured over the owner's whole recording, an eye
        photograph was active on **41% of frames** while they were only talking
        — a quarter of a squint worn through most of a conversation. No frame of
        that looks broken, which is why every still render passed.
        """
        for narrowing in (0.15, 0.30, 0.45):
            weights = weights_from_rest(expressive_tracking(blink=(narrowing, narrowing)))

            assert not [
                role
                for role, weight in weights.items()
                if role.startswith(("source_eyes", "source_squint")) and weight > 0.01
            ], f"a lid {narrowing:.0%} narrowed asked for {sorted(weights)}"

    def test_the_squint_photograph_is_not_used_at_all(self):
        """A partial closure cannot be told from ordinary lid movement.

        Ordinary life reaches 0.47 of the eye's opening and a blink reaches
        0.65, so a squint band either fires constantly or never. What would
        separate it is the *lower* lid, which a squint raises and a blink does
        not — available in the mesh, and not written yet.
        """
        for narrowing in (0.2, 0.4, 0.6, 0.9):
            weights = weights_from_rest(expressive_tracking(blink=(narrowing, narrowing)))
            assert "source_squint" not in weights

    def test_a_wide_eye_asks_for_the_wide_photograph(self):
        """``eyeWide`` never rises above 0.13 on eyes visibly at their widest,
        so the wide photograph scored 0.23 against a 0.45 floor and was never
        once selected from three clips generated to contain it."""
        weights = weights_from_rest(expressive_tracking(blink=(-0.5, -0.5)))

        assert weights["source_eyes_wide"] > 0.8
        assert weights.get("source_eyes_closed", 0.0) == 0.0
        assert weights.get("source_squint", 0.0) == 0.0

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
        # Regions hold their own crop of each photograph, cut at load, so one
        # injected afterwards has to be re-cut. Loading a real variant is not an
        # option here: alignment runs face detection, and a white square has no
        # face in it.
        avatar.regions = _build_regions(avatar.image, avatar.points, avatar.sources)

        blended = _blend_sources(avatar, expressive_tracking(jaw=0.5))

        assert float(blended.mean()) > float(avatar.image.mean())
        assert float(blended.mean()) < 255.0

    def test_a_mouth_photograph_leaves_the_forehead_alone(self, engine, tmp_path):
        """The defect that made the avatar a multiple exposure of itself.

        Every photograph used to be averaged across the whole face, so a smile
        repainted the forehead and the hairline too. Confining each to its own
        region is what stops expressions interfering — and what makes a larger
        photograph set worth having at all.
        """
        engine.load_avatar(make_descriptor(tmp_path))
        avatar = engine._loaded["avatar_test"]
        avatar.sources["source_mouth_open"] = np.full_like(avatar.image, 255)
        avatar.regions = _build_regions(avatar.image, avatar.points, avatar.sources)

        blended = _blend_sources(avatar, expressive_tracking(jaw=0.5))

        # Everything outside the mouth's own box is untouched. Asserted against
        # the box rather than against "the forehead", because the synthetic
        # landmarks here are a ring of points and do not put a mouth where a
        # real face has one — the invariant is what is being tested, not the
        # anatomy of the fixture.
        mouth = next(region for region in avatar.regions if region.name == "mouth")
        x0, y0, x1, y1 = mouth.box

        untouched = blended.copy()
        untouched[y0:y1, x0:x1] = avatar.image[y0:y1, x0:x1]
        np.testing.assert_array_equal(untouched, avatar.image)

        assert float(blended.mean()) > float(avatar.image.mean())


class TestEyeGeometry:
    """The avatar keeps its own eyes.

    The mesh maps the avatar's landmarks onto the user's, so at full strength
    the avatar inherits the *shape* of the user's eye aperture and has its own
    iris stretched to fill it. Live, that reads as pale flat eyes and is the one
    feature that gives the whole picture away.
    """

    @staticmethod
    def mesh_looking(dx: float, dy: float = 0.0) -> np.ndarray:
        """A mesh with eye sockets of a known size and irises at a known gaze offset.

        The socket is built as a ring of however many points the mesh declares,
        rather than a fixed list of corners: the engine carries the full eyelid
        contour and the count is allowed to change again. Its extremes are the
        same either way — a full width across and 0.6 of that tall — so what the
        assertions measure does not move with the point count.
        """
        from sha256.avatar.engine_warp import _EYE_PAIRS

        points: np.ndarray = (
            synthetic_landmarks()[list(MESH_INDICES), :2].astype(np.float32) * 400.0
        )
        for index, (socket, iris) in enumerate(_EYE_PAIRS):
            centre = np.array([150.0 + 100.0 * index, 180.0], dtype=np.float32)
            width = 40.0
            around_lid = np.linspace(0.0, 2.0 * np.pi, len(socket), endpoint=False)
            ring = np.stack([-0.5 * np.cos(around_lid), -0.3 * np.sin(around_lid)], axis=1).astype(
                np.float32
            )
            points[socket] = centre + ring * width

            pupil = centre + np.array([dx, dy], dtype=np.float32) * width
            around = np.array(
                [(0.0, 0.0), (0.2, 0.0), (0.0, -0.2), (-0.2, 0.0), (0.0, 0.2)], dtype=np.float32
            )
            points[iris] = pupil + around[: len(iris)] * width
        return points

    @pytest.mark.parametrize(("dx", "dy"), [(0.10, 0.0), (-0.10, 0.0), (0.0, 0.04), (0.0, -0.06)])
    def test_the_pupil_goes_where_the_user_is_looking(self, dx, dy):
        from sha256.avatar.engine_warp import _EYE_PAIRS, _GAZE_SCALE, _damp_brows

        source = self.mesh_looking(0.0, 0.0)
        target = self.mesh_looking(dx, dy)

        damped = _damp_brows(source, target)

        for socket, iris in _EYE_PAIRS:
            width = float(damped[socket][:, 0].max() - damped[socket][:, 0].min())
            offset = (damped[iris].mean(axis=0) - damped[socket].mean(axis=0)) / width
            assert offset[0] == pytest.approx(dx * _GAZE_SCALE, abs=0.02)
            assert offset[1] == pytest.approx(dy * _GAZE_SCALE, abs=0.02)

    @pytest.mark.parametrize(("dx", "dy"), [(0.6, 0.0), (-0.6, 0.0), (0.0, 0.5), (0.0, -0.5)])
    def test_an_extreme_gaze_is_held_inside_what_the_mesh_can_carry(self, dx, dy):
        """Far enough out, the triangles around the iris shear it into a dark
        diamond with visible edges. Rendered across the range, that is where it
        stops being clean, so that is where the pupil stops."""
        from sha256.avatar.engine_warp import (
            _EYE_PAIRS,
            _GAZE_LIMIT_DOWN,
            _GAZE_LIMIT_UP,
            _GAZE_LIMIT_X,
            _damp_brows,
        )

        damped = _damp_brows(self.mesh_looking(0.0), self.mesh_looking(dx, dy))

        for socket, iris in _EYE_PAIRS:
            width = float(damped[socket][:, 0].max() - damped[socket][:, 0].min())
            offset = (damped[iris].mean(axis=0) - damped[socket].mean(axis=0)) / width
            assert abs(offset[0]) <= _GAZE_LIMIT_X + 0.01
            assert -_GAZE_LIMIT_UP - 0.01 <= offset[1] <= _GAZE_LIMIT_DOWN + 0.01
            # And it still points the right way.
            assert np.sign(offset[0]) == np.sign(dx) or dx == 0

    def test_the_pupil_keeps_the_avatars_own_size(self):
        """Gaze is a direction. Nothing about the user's iris should transfer."""
        from sha256.avatar.engine_warp import _EYE_PAIRS, _damp_brows

        source = self.mesh_looking(0.0)
        target = self.mesh_looking(0.3)
        # A user whose iris is half again the size of the avatar's.
        for _, iris in _EYE_PAIRS:
            centre = target[iris].mean(axis=0)
            target[iris] = centre + (target[iris] - centre) * 1.5

        damped = _damp_brows(source, target)

        for _, iris in _EYE_PAIRS:

            def spread(points: np.ndarray) -> float:
                return float(np.abs(points - points.mean(axis=0)).max())

            assert spread(damped[iris]) == pytest.approx(spread(source[iris]), rel=0.05)

    def test_looking_straight_ahead_leaves_the_pupil_centred(self):
        from sha256.avatar.engine_warp import _EYE_PAIRS, _damp_brows

        source = self.mesh_looking(0.0)
        damped = _damp_brows(source, self.mesh_looking(0.0))

        for socket, iris in _EYE_PAIRS:
            width = float(damped[socket][:, 0].max() - damped[socket][:, 0].min())
            offset = (damped[iris].mean(axis=0) - damped[socket].mean(axis=0)) / width
            assert abs(offset[0]) < 0.03
            assert abs(offset[1]) < 0.03

    def test_the_avatar_does_not_inherit_the_users_eye_shape(self):
        from sha256.avatar.engine_warp import _EYE_POSITIONS, _damp_brows

        source = synthetic_landmarks()[list(MESH_INDICES)][:, :2].astype(np.float32) * 500.0
        target = source.copy()
        # A user whose eyes open half again as wide as the avatar's.
        centre = target[_EYE_POSITIONS].mean(axis=0)
        target[_EYE_POSITIONS] = centre + (target[_EYE_POSITIONS] - centre) * 1.5

        damped = _damp_brows(source, target)

        asked = np.abs(target[_EYE_POSITIONS] - source[_EYE_POSITIONS]).sum()
        given = np.abs(damped[_EYE_POSITIONS] - source[_EYE_POSITIONS]).sum()
        assert given < asked * 0.5, "the avatar followed the user's eye shape"

    def test_no_photograph_is_used_for_gaze(self):
        """Gaze is geometry now. A photograph cannot express it.

        Two gaze photographs blended put the pupil in two places; one blended
        with the base does the same against a forward-looking iris. Both render
        as the ghost eye the owner reported, and no weighting fixes it, because
        a blend of two positions is both of them rather than a third.
        """
        weights = weights_from_rest(
            expressive_tracking(
                gap=0.0,
                extra={
                    "eyeLookOutLeft": 0.8,
                    "eyeLookInRight": 0.78,
                    "eyeLookUpLeft": 0.6,
                    "eyeLookUpRight": 0.58,
                },
            )
        )

        assert not [role for role in weights if role.startswith("source_look_")]


class TestTurnLadder:
    """A photograph at every angle, and the nearest one chosen rather than mixed.

    The engine held two turned photographs and averaged them against the
    frontal one, so every angle in between was a blend of two angles that were
    not it, and past the outermost photograph it extrapolated from a shot that
    did not go that far. A clip supplies a frame every three degrees; the right
    answer then is to pick.
    """

    # Rung zero is the base portrait, which is the accumulator every other
    # source is mixed into, so it is named `None` and never appears in the
    # weights: the share it should have is the share left over.
    LADDER = ((-30.0, "turn@a"), (-10.0, "turn@b"), (0.0, None), (10.0, "turn@c"))

    def test_straight_ahead_uses_the_portrait_and_nothing_else(self):
        from sha256.avatar.engine_warp import _ladder_weights

        assert _ladder_weights(self.LADDER, 0.0) == {}

    def test_between_two_rungs_it_mixes_only_those_two(self):
        from sha256.avatar.engine_warp import _ladder_weights

        weights = _ladder_weights(self.LADDER, -25.0)

        assert set(weights) == {"turn@a", "turn@b"}
        assert weights["turn@a"] == pytest.approx(0.75)
        assert weights["turn@b"] == pytest.approx(0.25)

    def test_on_a_rung_it_is_that_rung(self):
        from sha256.avatar.engine_warp import _ladder_weights

        weights = _ladder_weights(self.LADDER, -10.0)

        assert weights["turn@b"] == pytest.approx(1.0)
        assert weights.get("turn@a", 0.0) == pytest.approx(0.0)

    def test_past_the_end_it_clamps_rather_than_extrapolating(self):
        """A photograph cannot show more cheek than was pointed at the camera."""
        from sha256.avatar.engine_warp import _ladder_weights

        assert _ladder_weights(self.LADDER, -80.0) == {"turn@a": 1.0}
        assert _ladder_weights(self.LADDER, 80.0) == {"turn@c": 1.0}

    def test_a_lone_portrait_is_not_a_ladder(self):
        from sha256.avatar.engine_warp import _ladder_weights

        assert _ladder_weights(((0.0, None),), 20.0) == {}

    def test_near_the_portrait_the_portrait_keeps_most_of_it(self):
        """The rungs either side of zero must not push the portrait out.

        The share the portrait keeps is whatever the rungs do not take, so a
        rung that reached full weight at a small angle would replace the
        avatar's own photograph with a video frame on a face that is barely
        turned.
        """
        from sha256.avatar.engine_warp import _ladder_weights

        weights = _ladder_weights(self.LADDER, 2.0)

        assert sum(weights.values()) == pytest.approx(0.2)


class TestPosePhotographsKeepTheirGeometry:
    """A turned photograph brings its angle with it, and the mesh follows.

    The rungs used to be mesh-warped onto the frontal portrait's layout, which
    at 35 degrees is not a correction but a fold: a seam down the face, the far
    eye and brow drawn twice, the mouth smeared sideways. What each rung holds
    now is the picture as it was shot, and its own landmarks alongside — so the
    live warp runs from a turned mesh onto a turned face and barely deforms
    anything.

    A translation stands in for a turn here. It is the part of a turn these
    tests can assert exactly, and the thing being tested is the bookkeeping:
    that pixels and landmarks make the same journey.
    """

    SHIFT = 40.0

    def _turned(self, avatar, shift: float, fill: int = 120, squeeze: float = 1.0):
        """Give the avatar one rung: the same face, ``shift`` px to the right.

        ``squeeze`` narrows it as well, which is the part of a turn a similarity
        cannot follow — a real turn foreshortens, and that leftover is what
        decides whether a frontal photograph can still be placed.
        """
        avatar.sources["turn@a"] = np.full_like(avatar.image, fill)
        sideways = np.array([shift, 0.0], dtype=np.float32)
        moved = avatar.points + sideways
        if squeeze != 1.0:
            centre = float(moved[:, 0].mean())
            moved[:, 0] = centre + (moved[:, 0] - centre) * squeeze
        avatar.source_points["turn@a"] = moved
        avatar.ladder = ((0.0, None), (30.0, "turn@a"))

    def _weight(self, avatar, tracking) -> float:
        from sha256.avatar.engine_warp import _ladder_weights

        # Read-only: `_blend_sources` has already advanced the resting angle.
        yaw = avatar.neutral.pose_offset("yaw", tracking.head_pose.yaw)
        return _ladder_weights(avatar.ladder, yaw).get("turn@a", 0.0)

    def _turn_from_rest(self, avatar, tracking):
        """Blend ``tracking`` after the engine has been shown a face at rest.

        A single frame in isolation *is* the resting angle — the neutral takes
        its first reading as straight ahead, which is what stops a camera above
        a monitor from putting a permanent tilt in the avatar — so a test that
        wants a turn to register has to establish rest first.

        The resting frame carries the same signals as the one under test, all of
        them quiet: a frame that reports no blendshapes at all teaches the
        engine nothing about them, and an expression measured against nothing
        scores zero.
        """
        _blend_sources(avatar, relaxed_like(tracking))
        return _blend_sources(avatar, tracking)

    def test_the_mesh_moves_with_the_photograph(self, engine, tmp_path):
        engine.load_avatar(make_descriptor(tmp_path))
        avatar = engine._loaded["avatar_test"]
        self._turned(avatar, self.SHIFT)

        tracking = expressive_tracking(gap=0.0, yaw=30.0)
        self._turn_from_rest(avatar, tracking)

        weight = self._weight(avatar, tracking)
        assert weight > 0.5, "the fixture should put the rung in play"
        moved = avatar.blend_points - avatar.points
        assert float(moved[:, 0].mean()) == pytest.approx(self.SHIFT * weight, abs=0.5)
        assert float(np.abs(moved[:, 1]).max()) < 0.5

    def test_the_eyes_follow_the_user_when_no_photograph_can_be_placed(self, engine, tmp_path):
        """The owner: "it almost never follows my eyes opening and closing" at a turn.

        Every eye photograph is of a face looking straight ahead, so past a
        certain angle none of them can be placed and the blend has nothing to
        say about the eyes — they freeze at whatever the turned photograph
        caught. The mesh takes over exactly as much as the photograph loses.
        """
        from sha256.avatar.engine_warp import _EYE_FOLLOW, _EYE_FOLLOW_ALONE

        engine.load_avatar(make_descriptor(tmp_path))
        avatar = engine._loaded["avatar_test"]
        # Narrowed as well as moved: a similarity can follow a shift exactly, so
        # a rung that is only shifted is one a photograph *can* be placed on.
        self._turned(avatar, self.SHIFT, squeeze=0.7)
        avatar.sources["source_eyes_closed"] = np.full_like(avatar.image, 255)
        avatar.regions = _build_regions(avatar.image, avatar.points, avatar.sources)

        # Eyes shut and the head turned: a photograph is wanted and cannot be
        # placed on a head it was not taken on.
        shut = expressive_tracking(gap=0.0, yaw=30.0, blink=(0.9, 0.9))
        self._turn_from_rest(avatar, shut)

        assert avatar.eye_follow > _EYE_FOLLOW
        assert avatar.eye_follow <= _EYE_FOLLOW_ALONE

    def test_a_resting_face_keeps_the_avatars_own_eyes(self, engine, tmp_path):
        """Nothing is asked for, so nothing is taken over — item 12 still holds."""
        from sha256.avatar.engine_warp import _EYE_FOLLOW

        engine.load_avatar(make_descriptor(tmp_path))
        avatar = engine._loaded["avatar_test"]
        self._turned(avatar, self.SHIFT)

        self._turn_from_rest(avatar, expressive_tracking(gap=0.0, yaw=30.0))

        assert avatar.eye_follow == _EYE_FOLLOW

    def test_a_still_head_leaves_the_mesh_where_the_portrait_has_it(self, engine, tmp_path):
        """The property that makes a resting avatar identical to its own photograph."""
        engine.load_avatar(make_descriptor(tmp_path))
        avatar = engine._loaded["avatar_test"]
        self._turned(avatar, self.SHIFT)

        _blend_sources(avatar, expressive_tracking(gap=0.0, yaw=0.0))

        assert avatar.blend_points is avatar.points

    def test_an_expression_lands_on_the_turned_head_not_the_frontal_one(self, engine, tmp_path):
        """The defect this fix would otherwise have introduced.

        Expression photographs are cut from a face looking straight ahead. Pasted
        at the frontal mouth's coordinates over a head that has turned, they land
        beside the mouth on screen — 70 px beside it at 36 degrees on the owner's
        avatar, which is the "two photographs on top of each other" all over
        again. So a region is carried onto the head before it is composed.
        """
        engine.load_avatar(make_descriptor(tmp_path))
        avatar = engine._loaded["avatar_test"]
        self._turned(avatar, self.SHIFT)
        avatar.sources["source_mouth_open"] = np.full_like(avatar.image, 255)
        avatar.regions = _build_regions(avatar.image, avatar.points, avatar.sources)

        tracking = expressive_tracking(gap=0.09, jaw=0.6, yaw=30.0)
        blended = self._turn_from_rest(avatar, tracking)

        # Where the mouth photograph landed: it is the only white thing in a
        # picture of grey, so the centre of the excess brightness is its centre.
        # Weighted rather than thresholded, because it arrives through a
        # feathered mask at the weight the jaw asked for and so never reaches
        # 255 anywhere.
        excess = np.clip(blended.mean(axis=2).astype(np.float64) - 130.0, 0.0, None)
        assert excess.sum() > 0.0, "the mouth photograph did not appear at all"
        columns = excess.sum(axis=0)
        centre = float((columns * np.arange(columns.size)).sum() / columns.sum())

        mouth = next(region for region in avatar.regions if region.name == "mouth")
        x0, _, x1, _ = mouth.box
        frontal = (x0 + x1) / 2.0
        expected = frontal + self.SHIFT * self._weight(avatar, tracking)
        assert centre == pytest.approx(expected, abs=4.0)
        assert abs(centre - frontal) > 8.0, "it stayed at the frontal mouth"


class TestRelighting:
    """The avatar must take the room's light and none of the user's face.

    The correction adds the whole scene-minus-avatar difference back, and what
    is under the mask is the user's face — so whatever the correction can
    represent, it takes from the person the product exists to hide. It used to
    be a blurred field, which could represent a brow shadow and did: the avatar
    was printed with the user's own brows and beard, and no improvement to the
    source blend survived it.

    A plane can represent a brightness and a gradient. It cannot represent a
    brow.
    """

    SIZE = 128

    def correction_for(self, pattern: np.ndarray) -> np.ndarray:
        from sha256.avatar.engine_warp import _match_colour

        holder = type("Holder", (), {"correction": None})()
        avatar = np.full((self.SIZE, self.SIZE, 3), 180.0, dtype=np.float32)
        alpha = np.ones((self.SIZE, self.SIZE, 1), dtype=np.float32)
        scene = avatar + 40.0 * pattern[:, :, None]
        return _match_colour(holder, avatar.copy(), scene, alpha) - avatar

    def transferred(self, pattern: np.ndarray) -> float:
        """Fraction of ``pattern`` the correction copied onto the avatar."""
        correction = self.correction_for(pattern)
        return float((correction[:, :, 0] * pattern).sum()) / float((pattern**2).sum()) / 40.0

    def test_a_uniform_light_change_is_followed_completely(self):
        assert self.transferred(np.ones((self.SIZE, self.SIZE), dtype=np.float32)) > 0.95

    @pytest.mark.parametrize("axis", [0, 1])
    def test_a_light_gradient_is_followed_completely(self, axis):
        """A window on one side. This is what a single mean shift could not do."""
        ramp = np.linspace(-1.0, 1.0, self.SIZE, dtype=np.float32)
        pattern = np.broadcast_to(ramp if axis else ramp[:, None], (self.SIZE, self.SIZE))

        assert self.transferred(np.ascontiguousarray(pattern)) > 0.95

    def test_a_brow_shadow_is_mostly_rejected(self):
        """The user's face must not print through onto the avatar's."""
        pattern = np.zeros((self.SIZE, self.SIZE), dtype=np.float32)
        pattern[int(self.SIZE * 0.28) : int(self.SIZE * 0.36)] = -1.0

        # A plane cannot reject it entirely — an off-centre band has a linear
        # component — but the blurred field it replaced transferred 52%.
        assert self.transferred(pattern) < 0.35

    def test_fine_detail_is_not_transferred_at_all(self):
        stripes = np.sign(np.sin(np.arange(self.SIZE, dtype=np.float32) * 1.4))
        pattern = np.broadcast_to(stripes, (self.SIZE, self.SIZE))

        assert abs(self.transferred(np.ascontiguousarray(pattern))) < 0.05


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

    def test_the_blends_own_teeth_reach_the_live_opening(self, engine, tmp_path):
        """The owner's report: a smile with no teeth, and an open mouth with none.

        The interior was a mask cut from the *base portrait*, and the base
        portrait's lips are together — a sliver a few pixels tall. So however
        wide the blend's mouth was, only that sliver was ever copied and the
        rest of the opening stayed flat cavity colour. It is asked of the
        blended mesh now, which is the thing that actually describes the blend.
        """
        from sha256.avatar.engine_warp import _INNER_LIP_SLICE, _paint_mouth_interior

        engine.load_avatar(make_descriptor(tmp_path))
        avatar = engine._loaded["avatar_test"]
        avatar.cavity = np.array([7.0, 11.0, 13.0], dtype=np.float32)
        count = avatar.points[_INNER_LIP_SLICE].shape[0]

        # A blend whose mouth is open, with something unmistakable inside it.
        teeth = (240, 250, 255)
        blend = np.full_like(avatar.image, 90)
        avatar.blend_points = avatar.points.copy()
        avatar.blend_points[_INNER_LIP_SLICE] = _ring(count, (128, 170), (30, 18))
        cv2.fillPoly(blend, [avatar.blend_points[_INNER_LIP_SLICE].astype(np.int32)], teeth)

        # The user, opening a little wider than the photograph did.
        target = avatar.points.copy()
        target[_INNER_LIP_SLICE] = _ring(count, (128, 170), (34, 22))
        canvas = np.zeros((256, 256, 3), dtype=np.uint8)
        _paint_mouth_interior(avatar, canvas, blend, target)

        opening = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.fillPoly(opening, [target[_INNER_LIP_SLICE].astype(np.int32)], 255)
        inside = canvas[opening > 0].astype(np.int16)
        assert inside.size > 0, "the fixture drew no opening"

        near_teeth = (np.abs(inside - np.array(teeth, dtype=np.int16)).sum(axis=1) < 60).mean()
        assert near_teeth > 0.5, f"only {near_teeth:.0%} of the opening came from the blend"

    def test_the_interior_is_taken_from_one_photograph_not_an_average(self, engine, tmp_path):
        """43% of a tooth is a translucent one.

        At the owner's widest smile the mouth was 43% a photograph with teeth
        in it and 57% three photographs with their lips together, and what
        arrived was teeth with a closed mouth showing through them. The lips
        may cross-fade; what is behind them may not.
        """
        engine.load_avatar(make_descriptor(tmp_path))
        avatar = engine._loaded["avatar_test"]
        for role in ("source_smile", "source_smile_closed", "source_mouth_open"):
            avatar.sources[role] = np.full_like(avatar.image, 128)
        # Measured at load in the real thing. The owner's own set separates
        # exactly here: three photographs parted, four with the lips together.
        avatar.gaps = {
            "source_smile": 0.075,
            "source_smile_closed": 0.015,
            "source_mouth_open": 0.199,
        }

        _blend_sources(avatar, expressive_tracking(gap=0.062, smile=1.0))
        assert avatar.interior_role == "source_smile"

        _blend_sources(avatar, expressive_tracking(gap=0.095, jaw=0.9))
        assert avatar.interior_role == "source_mouth_open"

    def test_a_mouth_with_its_lips_together_offers_no_interior(self, engine, tmp_path):
        """Which is what leaves a wide opening as cavity, rather than as lips."""
        engine.load_avatar(make_descriptor(tmp_path))
        avatar = engine._loaded["avatar_test"]
        avatar.sources["source_smile_closed"] = np.full_like(avatar.image, 128)
        avatar.gaps = {"source_smile_closed": 0.015}

        _blend_sources(avatar, expressive_tracking(gap=0.001, smile=1.0))

        assert avatar.interior_role is None

    def test_a_closed_blend_still_shows_cavity_rather_than_lips(self, engine, tmp_path):
        """The other half of it, and the reason the mask exists at all.

        With no open-mouth photograph mixed in, the blend's lips are together
        and there is no interior to give. Opening wider then has to reveal
        darkness, not a stretched picture of a closed mouth.
        """
        from sha256.avatar.engine_warp import _INNER_LIP_SLICE, _paint_mouth_interior

        engine.load_avatar(make_descriptor(tmp_path))
        avatar = engine._loaded["avatar_test"]
        avatar.cavity = np.array([7.0, 11.0, 13.0], dtype=np.float32)
        count = avatar.points[_INNER_LIP_SLICE].shape[0]

        blend = np.full_like(avatar.image, 200)
        avatar.blend_points = avatar.points.copy()
        avatar.blend_points[_INNER_LIP_SLICE] = _ring(count, (128, 170), (30, 0.5))

        target = avatar.points.copy()
        target[_INNER_LIP_SLICE] = _ring(count, (128, 170), (34, 22))
        canvas = np.zeros((256, 256, 3), dtype=np.uint8)
        _paint_mouth_interior(avatar, canvas, blend, target)

        opening = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.fillPoly(opening, [target[_INNER_LIP_SLICE].astype(np.int32)], 255)
        inside = canvas[opening > 0].astype(np.int16)

        cavity = np.abs(inside - avatar.cavity.astype(np.int16)).sum(axis=1) < 30
        assert cavity.mean() > 0.8, f"only {cavity.mean():.0%} of the opening was cavity"

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

    def test_open_eyes_never_ghost_the_closed_eye_photograph(self):
        """The "two sets of eyes" the owner photographed.

        A relaxed eye reads about 0.26 on this tracker. Measured against a fixed
        0.20 threshold that scored a permanent 15% blink, so a shut eyelid was
        laid over an open eye on every frame at partial opacity.
        """
        resting = expressive_tracking(gap=0.0, blink=(0.26, 0.25))
        neutral = _Neutral()

        for _ in range(30):
            weights = _source_weights(resting, 0.0947, neutral)

        assert "source_eyes_closed" not in weights

    def test_a_real_blink_still_closes_the_eyes(self):
        """The other half: relative must not mean insensitive."""
        neutral = _Neutral()
        for _ in range(30):
            _source_weights(expressive_tracking(gap=0.0, blink=(0.26, 0.25)), 0.0947, neutral)

        blinked = _source_weights(expressive_tracking(gap=0.0, blink=(0.95, 0.93)), 0.0947, neutral)

        assert blinked["source_eyes_closed"] > 0.9

    def test_a_wide_open_mouth_photograph_does_not_hide_the_teeth(self):
        """A photograph of a yawn must not make ordinary speech score as closed.

        The owner's replacement open-mouth photograph measured a lip gap of
        0.229 against 0.095 for a normal opening. Dividing by it scored a smile
        at 0.13 and the teeth stopped appearing — the photograph was loaded and
        the weight that would have shown it was measured against a yawn.
        """
        smiling = expressive_tracking(gap=0.06, smile=0.6)

        against_a_yawn = weights_from_rest(smiling, full_gap=0.2287)
        against_normal = weights_from_rest(smiling, full_gap=0.095)

        assert against_a_yawn["source_smile"] == pytest.approx(against_normal["source_smile"])
        assert against_a_yawn["source_smile"] > 0.5

    def test_a_narrow_photograph_still_reaches_full_weight_at_its_own_gap(self):
        """The other half: a photograph cannot depict more than it contains."""
        narrow = weights_from_rest(expressive_tracking(gap=0.05, jaw=0.5), full_gap=0.05)

        assert narrow["source_mouth_open"] > 0.9

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

    def test_gaze_asks_for_no_photograph_at_all(self):
        """It is geometry now — see TestEyeGeometry. The four gaze photographs
        are no longer used, and an avatar that has them is not penalised for it.
        """
        for shape in ("eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookDownLeft"):
            weights = weights_from_rest(expressive_tracking(gap=0.0, extra={shape: 0.6}))
            assert not [role for role in weights if role.startswith("source_look_")]

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

    def test_a_tilted_head_asks_for_no_photograph_at_all(self):
        """The third family dropped, after the gaze photographs and the squint.

        A tilt photograph was active on 99.2% of the owner's recording at a
        median weight of 0.38, and its landmarks sit a tenth of a face away
        from the portrait's — so once a pose photograph began carrying its own
        geometry, it moved the mesh 4.2 px further from the tracked face on
        nearly every frame. An avatar that has the photographs is not penalised
        for having them; they are simply never asked for.
        """
        for pitch in (18.0, -18.0, 40.0, -40.0):
            weights = weights_from_rest(_pitched(pitch))
            assert not [role for role in weights if role.startswith("source_tilt_")]

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


def _ring(count: int, centre: tuple[float, float], radii: tuple[float, float]) -> np.ndarray:
    """``count`` points around an ellipse, in contour order.

    Contour order because the lips are filled as a polygon, not tested as a
    point cloud — scattered points would describe a star.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack(
        (centre[0] + radii[0] * np.cos(angles), centre[1] + radii[1] * np.sin(angles)),
        axis=1,
    ).astype(np.float32)


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
