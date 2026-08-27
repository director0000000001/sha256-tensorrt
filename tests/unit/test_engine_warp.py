"""Tests for the landmark-warp avatar engine.

No real photograph and no real tracker: a synthetic portrait and a synthetic
landmark set exercise the geometry, which is where this engine can be wrong in
ways that are hard to see. The visual quality question is answered by looking at
it, not by asserting on pixels.

What matters here is that it fails clearly when an avatar is unusable â€” a
missing portrait, a portrait with no face in it, the wrong engine id â€” because
those are the errors a user will actually hit, and each one has an obvious cause
that the message should name.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from sha256.avatar.engine_warp import (
    _BROW_POSITIONS,
    _BROWS_LADDER_START,
    _BROWS_PREFIX,
    _COMMIT_FULL,
    _EASE_PER_SECOND,
    _EYE_PAIRS,
    _EYE_POSITIONS,
    _EYES_LADDER_START,
    _EYES_PREFIX,
    _FRAME_SECONDS,
    _IRIS_POSITIONS,
    _LADDER_CROSSFADE_DEGREES,
    _LEVEL_WARMUP_FRAMES,
    _LIP_FOLLOW,
    _MASK_FEATHER,
    _NEAR_CHEEK_START,
    _NOSE_TIP_POSITION,
    _POSE_RATE,
    _POSE_WARMUP_FRAMES,
    _PROFILE_POSITIONS,
    _ROLE_REGION,
    _SMILE_LADDER_START,
    _SMILE_REFERENCE,
    _SMILE_WIDEN_SPAN,
    _SPEECH_HOLD,
    _SPEECH_LADDER_START,
    _SPEECH_RELEASE,
    _SPEECH_WIDTH_START,
    _STABLE_POSITIONS,
    _SWITCH_MARGIN,
    MESH_INDICES,
    MESH_POINT_COUNT,
    WarpAvatarEngine,
    _blend_sources,
    _brow_amount,
    _brow_height,
    _build_regions,
    _carried_share,
    _commit,
    _contain_irises,
    _destination_mask,
    _eased,
    _eye_amount,
    _eye_opening,
    _ladder_weights,
    _lid_aspect,
    _lip_parting,
    _misplaced,
    _nearest_rung,
    _nearest_smile,
    _nearest_speech,
    _Neutral,
    _one_per_region,
    _pixel_points,
    _shut,
    _smile_amount,
    _smile_share,
    _source_weights,
    _speech_amount,
    _trim_forehead,
    _turn_bias,
    _widened,
    refused_as_rung,
)
from sha256.avatar.engine_warp import _OVAL_SLICE as OVAL_COUNT_SLICE
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


def with_brow_anatomy(points: np.ndarray) -> np.ndarray:
    """The synthetic face, with the brows put above the eyes.

    ``synthetic_landmarks`` scatters every inner feature around a ring, which is
    enough for anything that only needs *a* face â€” and useless for a question
    about what sits above what. Anything testing where the brows end and the
    eyes begin has to say where they are.
    """
    from sha256.avatar.engine_warp import (
        _LEFT_BROW,
        _LEFT_EYE,
        _LEFT_IRIS,
        _RIGHT_BROW,
        _RIGHT_EYE,
        _RIGHT_IRIS,
    )

    moved = points.copy()
    # The brows get a slight arc rather than a flat line: a convex hull of
    # collinear points has no area, so a flat brow would leave the region's mask
    # empty and every question about what it covers unanswerable.
    for position, index in enumerate(_LEFT_BROW + _RIGHT_BROW):
        moved[index, 1] = 0.36 + 0.004 * (position % 3)
    for index in _LEFT_EYE + _RIGHT_EYE:
        moved[index, 1] = 0.46
    for index in _LEFT_IRIS + _RIGHT_IRIS:
        moved[index, 1] = 0.46
    return moved


def relaxed_like(tracking: FaceTrackingResult) -> FaceTrackingResult:
    """The same face at rest: every signal it carries, all of them quiet.

    Needed because the engine measures expression *relative to rest*, so it has
    to have seen rest. The blendshape names must match â€” a resting face that
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
        # And a resting mouth, for the same reason and since 2026-08-21: the
        # smile is measured from the mouth's own width now, so a resting frame
        # that kept the expression's widened corners would teach the engine that
        # this person's face is always smiling.
        span = float(abs(points[263, 0] - points[33, 0])) or 0.6
        points[61, :2] = (0.5 - span * 0.53 / 2, points[61, 1])
        points[291, :2] = (0.5 + span * 0.53 / 2, points[291, 1])
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

    A single frame in isolation is by definition neutral â€” that is the whole
    point of the change these tests describe â€” so a test that wants an
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
    tracker's own type â€” it is just a string â€” and the test would then measure
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

    # The smile axis is geometry now, so a test that asks for a smile has to
    # produce one: the mouth corners move apart against the span between the
    # outer eye corners. Resting is 0.53 of that span and the owner's widest
    # rung is 0.72, both measured on his own photographs.
    #
    # **After `set_eye_opening`, which moves the eye corners.** Placing the
    # mouth first and assuming a span read 3.18 where it should have read 0.53,
    # and every test about an open mouth then thought it was a smile.
    # `set_eye_opening` leaves the outer corners a tenth of a frame apart, which
    # is not a face: the lip parting is measured against this span and would
    # read six times what a real one does. Two fifths, so that a parting reads
    # about one and a half times the gap over face height, as it does on the
    # owner's own photographs.
    points[33, 0] = 0.3
    points[263, 0] = 0.7
    span = float(abs(points[263, 0] - points[33, 0]))
    reach = 0.53 + 0.19 * min(max((smile or 0.0) / _SMILE_REFERENCE, 0.0), 1.0)
    points[61, :2] = (0.5 - span * reach / 2, 0.6)
    points[291, :2] = (0.5 + span * reach / 2, 0.6)

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
        # Through `_pixel_points` rather than by hand, so the fixture builds a
        # mesh the same way the engine does -- including the computed iris arcs,
        # which a hand-built subset would silently leave off.
        mesh = _pixel_points(synthetic_landmarks(), width, height)
        assert mesh is not None
        result: np.ndarray = mesh
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
        sweep as the lid closes â€” the wedge above a blink. The ring terminates
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
        # comes out *identical* to the scene by construction â€” and the test
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
        weights = weights_from_rest(expressive_tracking(gap=0.06, smile=0.55))

        assert weights["source_smile"] > weights.get("source_mouth_open", 0.0)

    def test_a_blink_calls_for_the_closed_eye_photo(self):
        weights = weights_from_rest(expressive_tracking(blink=(0.9, 0.85)))

        assert weights["source_eyes_closed"] > 0.8

    def test_eyes_held_shut_close_as_firmly_as_a_blink(self):
        """Shut lids are shut lids, whatever ``eyeBlink`` happens to say.

        The blendshape is trained on blinks, which are fast. Measured on the
        owner's recording it peaks at 0.66 on a blink and reads **0.14** on two
        seconds of held closure â€” lower than plenty of frames with the eyes
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
        â€” a quarter of a squint worn through most of a conversation. No frame of
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
        not â€” available in the mesh, and not written yet.
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
        weights = weights_from_rest(expressive_tracking(gap=0.095, jaw=0.5, smile=0.3))

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
        region is what stops expressions interfering â€” and what makes a larger
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
        # real face has one â€” the invariant is what is being tested, not the
        # anatomy of the fixture.
        mouth = next(region for region in avatar.regions if region.name == "mouth")
        x0, y0, x1, y1 = mouth.box

        untouched = blended.copy()
        untouched[y0:y1, x0:x1] = avatar.image[y0:y1, x0:x1]
        np.testing.assert_array_equal(untouched, avatar.image)

        assert float(blended.mean()) > float(avatar.image.mean())


class TestTheRestingAngle:
    """Where the engine thinks straight ahead is.

    It is seeded from the first frame, which is right when someone opens the
    application looking at their screen and wrong when they do not. Driven by a
    clip that opens mid-turn at 31.7 degrees, the engine took that for straight
    ahead and then chose a 30-degree-turned photograph for a frontal face: a
    whole cheek stretched sideways, for half a minute.
    """

    def feed(self, neutral: _Neutral, angle: float, frames: int) -> float:
        for _ in range(frames):
            neutral.relative_pose("yaw", angle)
        return neutral.pose_offset("yaw", angle)

    def test_the_first_frame_is_its_own_resting_angle(self):
        """The property that makes an avatar start as its own photograph."""
        neutral = _Neutral()

        assert neutral.relative_pose("yaw", 31.7) == 0.0

    def test_one_frame_does_not_move_the_estimate_far(self):
        """A turn in the opening second must not be learned as straight ahead."""
        neutral = _Neutral()
        neutral.relative_pose("yaw", 0.0)

        neutral.relative_pose("yaw", 30.0)

        assert neutral.pose_offset("yaw", 30.0) > 25.0

    def test_a_session_that_opens_turned_is_corrected_within_the_warm_up(self):
        """The defect this exists for, stated as a number."""
        neutral = _Neutral()
        neutral.relative_pose("yaw", 31.7)

        settled = self.feed(neutral, 0.0, _POSE_WARMUP_FRAMES)

        assert abs(settled) < 3.0, "still thinks a turned head was straight ahead"

    def test_the_old_slow_mean_would_not_have_managed_that(self):
        """Guards the fix rather than the framework.

        At the slow rate alone, the same number of frames leaves most of the
        error in place â€” which is what the owner would have seen as a cheek
        stretched sideways until well after they had stopped caring why.
        """
        left_over = 31.7 * (1.0 - _POSE_RATE) ** _POSE_WARMUP_FRAMES

        assert left_over > 20.0

    def test_a_held_turn_after_the_warm_up_is_not_a_new_way_of_sitting(self):
        neutral = _Neutral()
        self.feed(neutral, 0.0, _POSE_WARMUP_FRAMES * 2)

        self.feed(neutral, 30.0, 90)

        # Three seconds of looking at a second monitor. The estimate may creep;
        # it may not follow.
        assert neutral.pose_offset("yaw", 30.0) > 27.0

    def test_someone_who_genuinely_moves_their_chair_is_still_learned(self):
        """Soft rather than a threshold: slowly is not never."""
        neutral = _Neutral()
        self.feed(neutral, 0.0, _POSE_WARMUP_FRAMES * 2)

        settled = self.feed(neutral, 20.0, 6000)

        assert abs(settled) < 5.0

    def test_reading_the_offset_does_not_teach_it(self):
        neutral = _Neutral()
        neutral.relative_pose("yaw", 0.0)

        for _ in range(500):
            neutral.pose_offset("yaw", 40.0)

        assert neutral.pose_offset("yaw", 40.0) == pytest.approx(40.0)


class TestTheRestingBrowLevel:
    """Where the engine thinks a brow sits when nothing is happening.

    ``track_level`` was a plain slow mean seeded from the first frame, with none
    of the acceleration :meth:`relative_pose` and :meth:`relative_rise` both
    have. Measured on `driving.mkv`: frame 0 reads a brow height of **0.1489
    against the clip's median of 0.1014**, 47% high, and at ``_POSE_RATE`` alone
    that takes a seventeen-second time constant to walk off. The brow amount had
    a median of **-0.202** over the whole recording -- a person talking normally
    reading a fifth of the way into a frown -- and **-0.393 over the first ten
    seconds**, which is the part of a call anybody actually judges.
    """

    SEED = 0.1489
    TRUE = 0.1014

    def feed(self, neutral: _Neutral, value: float, frames: int) -> float:
        for _ in range(frames):
            rest = neutral.track_level("browHeight", value)
        return rest

    def test_the_first_reading_is_its_own_rest(self):
        """The property that makes an avatar start as its own photograph."""
        neutral = _Neutral()

        assert neutral.track_level("browHeight", self.SEED) == self.SEED

    def test_a_bad_first_frame_is_walked_off_inside_the_warm_up(self):
        """The defect this exists for, stated as a number."""
        neutral = _Neutral()
        neutral.track_level("browHeight", self.SEED)

        settled = self.feed(neutral, self.TRUE, _LEVEL_WARMUP_FRAMES)

        left = (settled - self.TRUE) / (self.SEED - self.TRUE)
        assert left < 0.15, "the first frame is still most of the resting estimate"

    def test_the_old_slow_mean_would_not_have_managed_that(self):
        """Guards the fix rather than the framework.

        At the slow rate alone the same number of frames leaves almost all of
        the error in place, which is a mild frown on a face doing nothing for
        as long as the owner keeps looking.
        """
        left = (1.0 - _POSE_RATE) ** _LEVEL_WARMUP_FRAMES

        assert left > 0.85

    def test_it_warms_in_both_directions(self):
        """The difference from :meth:`relative_rise`, which warms upward only.

        That one is protecting an asymmetry it needs against blinks. A plain
        level has no asymmetry to protect, and a seed can be wrong on either
        side -- the one that was measured was high.
        """
        neutral = _Neutral()
        neutral.track_level("browHeight", self.TRUE)

        settled = self.feed(neutral, self.SEED, _LEVEL_WARMUP_FRAMES)

        left = (self.SEED - settled) / (self.SEED - self.TRUE)
        assert left < 0.15, "a seed below the face is not corrected the way one above it is"

    def test_a_held_brow_after_the_warm_up_is_not_a_new_resting_face(self):
        """Otherwise raising the brows for two seconds teaches it that they rest
        there, and lowering them back reads as a frown."""
        neutral = _Neutral()
        self.feed(neutral, self.TRUE, _LEVEL_WARMUP_FRAMES * 2)

        settled = self.feed(neutral, self.SEED, 60)

        held = (self.SEED - settled) / (self.SEED - self.TRUE)
        assert held > 0.8, "two seconds of raised brows moved where they rest"


class TestOnePhotographPerRegion:
    """The owner's report, in their words: they smile, the teeth appear, and the
    closed mouth is still there underneath at low opacity.

    Both halves of that were one cause. A region mixed every photograph that
    scored anything, and whatever the mix did not claim was left as base
    portrait â€” whose lips are shut. So a wide smile was a smile photograph
    averaged with a pressed-lips one and a rounded one, over a closed mouth.
    """

    def test_only_the_strongest_photograph_in_a_region_survives(self):
        kept = _one_per_region(
            {
                "source_smile": 0.8,
                "source_smile_closed": 0.5,
                "source_mouth_pressed": 0.3,
                "source_mouth_o": 0.2,
            },
            {},
        )

        assert set(kept) == {"source_smile"}

    def test_a_committed_expression_leaves_no_base_portrait_underneath(self):
        """The teeth case. Past the commit point the shut mouth is gone, not faint.

        Held for a few frames first, because the amount is now rate limited â€”
        see :class:`TestExpressionsArriveOverTime`. What matters here is where
        it settles, not how long it takes to get there.
        """
        holding: dict[str, str] = {}
        applied: dict[str, float] = {}
        for _ in range(12):
            kept = _one_per_region(
                {"source_smile": 0.8, "source_smile_closed": 0.5}, holding, applied
            )

        assert kept["source_smile"] == 1.0

    def test_a_faint_expression_still_fades_in_rather_than_popping(self):
        kept = _one_per_region({"source_mouth_o": _COMMIT_FULL / 2}, {})

        assert 0.0 < kept["source_mouth_o"] < 1.0

    def test_every_region_keeps_its_own_winner(self):
        kept = _one_per_region(
            {
                "source_smile": 0.8,
                "source_mouth_o": 0.4,
                "source_eyes_closed": 0.6,
                "source_eyes_wide": 0.1,
                "source_brows_up": 0.5,
            },
            {},
        )

        assert set(kept) == {"source_smile", "source_eyes_closed", "source_brows_up"}

    def test_whole_head_photographs_are_not_in_the_competition(self):
        """A rung is not a rival of a mouth: it says what angle the head is at."""
        kept = _one_per_region(
            {"turn@4": 0.7, "turn@5": 0.3, "source_turn_left": 0.4, "source_smile": 0.9}, {}
        )

        assert kept["turn@4"] == 0.7
        assert kept["turn@5"] == 0.3
        assert kept["source_turn_left"] == 0.4

    def test_the_incumbent_holds_the_region_against_a_narrow_challenge(self):
        """Without this the two nearest expressions swap several times a second."""
        holding = {"mouth": "source_mouth_o"}

        kept = _one_per_region(
            {"source_mouth_o": 0.50, "source_mouth_pressed": 0.50 + _SWITCH_MARGIN / 2}, holding
        )

        assert set(kept) == {"source_mouth_o"}
        assert holding["mouth"] == "source_mouth_o"

    def test_a_clear_challenger_takes_the_region(self):
        holding = {"mouth": "source_mouth_o"}

        kept = _one_per_region(
            {"source_mouth_o": 0.50, "source_mouth_pressed": 0.50 + _SWITCH_MARGIN * 2}, holding
        )

        assert set(kept) == {"source_mouth_pressed"}
        assert holding["mouth"] == "source_mouth_pressed"

    def test_a_region_with_nothing_this_frame_forgets_its_incumbent(self):
        """So the next expression to arrive starts unopposed, not out-argued by
        whatever was last on screen seconds ago."""
        holding = {"mouth": "source_smile"}

        _one_per_region({"source_brows_up": 0.4}, holding)

        assert "mouth" not in holding

    def test_the_commit_curve_is_continuous_up_to_full(self):
        assert _commit(0.0) == 0.0
        assert _commit(_COMMIT_FULL) == 1.0
        assert _commit(_COMMIT_FULL * 2) == 1.0
        assert _commit(_COMMIT_FULL / 4) < _commit(_COMMIT_FULL / 2)

    def test_a_blend_ends_with_one_mouth_photograph_and_remembers_it(self, engine, tmp_path):
        """The same rule, through the real composite rather than the arithmetic."""
        engine.load_avatar(make_descriptor(tmp_path))
        avatar = engine._loaded["avatar_test"]
        for role in ("source_mouth_open", "source_smile", "source_smile_closed"):
            avatar.sources[role] = np.full_like(avatar.image, 255)
        avatar.regions = _build_regions(avatar.image, avatar.points, avatar.sources)

        _blend_sources(avatar, expressive_tracking(jaw=0.5, smile=0.5))

        mouth = [role for role in avatar.cached_weights if _ROLE_REGION.get(role) == "mouth"]
        assert len(mouth) == 1, f"more than one photograph in the mouth: {mouth}"
        assert avatar.holding["mouth"] == mouth[0]


class TestTheLadderDoesNotAverageAcrossAGap:
    """The owner, on turning their head: *dough*, around the nose.

    A ladder cut from a real clip is not evenly spaced â€” the second turn clip
    left eight degrees between +11 and +19, because that is where the head was
    moving rather than stopping. Averaged across that gap, a cheek at +11 lies
    over a cheek at +19 and the face reads as smeared.
    """

    TIGHT = ((0.0, None), (3.0, "a"), (6.0, "b"))
    WIDE = ((0.0, None), (11.0, "a"), (19.0, "b"))

    def test_a_ladder_at_its_intended_spacing_is_unchanged(self):
        """Three degrees apart, the mix is the old proportional one."""
        weights = _ladder_weights(self.TIGHT, 4.0)

        assert weights["a"] == pytest.approx(2 / 3)
        assert weights["b"] == pytest.approx(1 / 3)

    def test_across_a_wide_gap_one_photograph_is_shown_whole(self):
        weights = _ladder_weights(self.WIDE, 12.5)

        assert weights == {"a": 1.0}

    def test_the_change_still_happens_and_is_centred_in_the_gap(self):
        middle = _ladder_weights(self.WIDE, 15.0)

        assert middle["a"] == pytest.approx(0.5)
        assert middle["b"] == pytest.approx(0.5)

    def test_the_change_is_no_wider_than_the_crossfade(self):
        below = _ladder_weights(self.WIDE, 15.0 - _LADDER_CROSSFADE_DEGREES)
        above = _ladder_weights(self.WIDE, 15.0 + _LADDER_CROSSFADE_DEGREES)

        assert below == {"a": 1.0}
        assert above == {"b": 1.0}

    def test_nothing_is_ever_extrapolated_past_the_ends(self):
        assert _ladder_weights(self.WIDE, 90.0) == {"b": 1.0}
        assert _ladder_weights(self.WIDE, -90.0) == {}


class TestExpressionsArriveOverTime:
    """The owner, on the first version of the region change: the teeth arrive
    *"as if the frame had been cut"*.

    They were right. The amount was read straight off the tracker, and a
    blendshape crosses the commit point in a frame or two â€” so a photograph
    correct for the mouth's final shape was appearing whole while the lips were
    still moving. A real smile takes about a third of a second to form.
    """

    def run(
        self,
        weights: dict[str, float],
        frames: int,
        holding: dict[str, str] | None = None,
        applied: dict[str, float] | None = None,
    ) -> tuple[list[dict[str, float]], dict[str, str], dict[str, float]]:
        holding = {} if holding is None else holding
        applied = {} if applied is None else applied
        seen = []
        for _ in range(frames):
            seen.append(_one_per_region(dict(weights), holding, applied))
        return seen, holding, applied

    def test_a_smile_does_not_arrive_in_one_frame(self):
        seen, _, _ = self.run({"source_smile": 1.0}, 1)

        assert seen[-1]["source_smile"] == pytest.approx(_EASE_PER_SECOND * _FRAME_SECONDS)

    def test_it_is_whole_within_a_third_of_a_second(self):
        frames = round(0.34 / _FRAME_SECONDS)
        seen, _, _ = self.run({"source_smile": 1.0}, frames)

        assert seen[-1]["source_smile"] == 1.0

    def test_it_rises_a_step_at_a_time(self):
        seen, _, _ = self.run({"source_smile": 1.0}, 4)
        shares = [frame["source_smile"] for frame in seen]

        assert shares == sorted(shares)
        step = _EASE_PER_SECOND * _FRAME_SECONDS
        assert all(b - a == pytest.approx(step) for a, b in pairwise(shares))

    def test_an_expression_that_stops_fades_rather_than_vanishing(self):
        _, holding, applied = self.run({"source_smile": 1.0}, 20)

        gone, _, _ = self.run({}, 1, holding, applied)

        assert gone[-1]["source_smile"] == pytest.approx(1.0 - _EASE_PER_SECOND * _FRAME_SECONDS)

    def test_and_is_forgotten_once_it_has_gone(self):
        _, holding, applied = self.run({"source_smile": 1.0}, 20)

        self.run({}, 20, holding, applied)

        assert holding == {} and applied == {}

    def test_switching_photograph_does_not_restart_the_fade(self):
        """The content changes at once; the amount carries over. Restarting it
        would drop the region back to base portrait mid-expression."""
        _, holding, applied = self.run({"source_mouth_o": 1.0}, 20)

        after, holding, _ = self.run({"source_mouth_open": 1.0}, 1, holding, applied)

        assert holding["mouth"] == "source_mouth_open"
        assert after[-1]["source_mouth_open"] == 1.0

    def test_a_blink_still_closes_the_eye(self):
        """The limit must not outlast the fastest thing a face does."""
        seen, _, _ = self.run({"source_eyes_closed": 1.0}, 6)

        assert seen[-1]["source_eyes_closed"] > 0.7


class TestTheMaskAtALargeTurn:
    """The owner, turning their head: *"the part next to my ear shows on the
    face â€” it does not cover me there."*

    The mask is eroded all the way round so the join happens over skin on both
    sides rather than over the background. Frontal that is right. Turned, the
    near side's edge runs down in front of the ear and what lies outside it is
    more cheek, so the erosion is pure loss: a band of the user's own face left
    showing inside their avatar, at the most visible part of the turn.
    """

    def turned(self, points: np.ndarray, by: float) -> np.ndarray:
        """The same mesh with the nose moved off centre, which is what a turn
        looks like to the mask â€” it reads the picture, not the tracker.

        ``by`` runs from -1 (nose against the left edge of the face, so the near
        cheek is the right one) through 0 to +1.
        """
        moved = points.copy()
        oval = points[OVAL_COUNT_SLICE][:, 0]
        low, high = float(oval.min()), float(oval.max())
        middle = (low + high) / 2.0
        moved[_NOSE_TIP_POSITION, 0] = middle + (high - middle) * by
        return moved

    def uncovered(self, points: np.ndarray, size: int = 320) -> float:
        """How much of the face the mask leaves showing."""
        mask = _destination_mask(points, size, size)[:, :, 0]
        face = np.zeros((size, size), np.uint8)
        cv2.fillPoly(face, [points[OVAL_COUNT_SLICE].astype(np.int32)], 255)
        return float((mask[face > 0] < 0.5).mean())

    def points(self) -> np.ndarray:
        from sha256.avatar.engine_warp import _pixel_points

        mesh = _pixel_points(synthetic_landmarks(scale=0.3), 320, 320)
        assert mesh is not None
        return mesh

    def test_a_face_looking_straight_ahead_is_left_exactly_as_it_was(self):
        """The erosion still protects the silhouette where it is a silhouette."""
        points = self.turned(self.points(), 0.0)

        assert _turn_bias(points)[0] < _NEAR_CHEEK_START
        assert self.uncovered(points) == pytest.approx(self.uncovered(self.turned(points, 0.2)))

    def test_a_turned_face_is_covered_where_it_was_not(self):
        points = self.points()

        frontal = self.uncovered(self.turned(points, 0.0))
        turned = self.uncovered(self.turned(points, 1.0))

        assert turned < frontal / 2.0, "the near cheek is still being eroded"

    def test_it_never_reaches_outside_the_face(self):
        """Undoing the erosion, never extending past the outline into hair or
        into the room â€” the restored piece is clipped to the outline first."""
        points = self.turned(self.points(), 1.0)
        size = 320

        mask = _destination_mask(points, size, size)[:, :, 0]
        outline = np.zeros((size, size), np.uint8)
        cv2.fillPoly(outline, [_trim_forehead(points).astype(np.int32)], 255)
        # The nose and lips are filled at full strength too, and always were â€”
        # what is being tested here is that the near-side restoration adds
        # nothing beyond them.
        cv2.fillConvexPoly(
            outline, cv2.convexHull(points[_PROFILE_POSITIONS].astype(np.int32)), 255
        )
        feather = max(3, int(320 * 0.3 * _MASK_FEATHER) | 1)
        allowed = cv2.dilate(outline, np.ones((feather * 2 + 1, feather * 2 + 1), np.uint8))

        assert float(mask[allowed == 0].max(initial=0.0)) == 0.0

    def test_which_cheek_is_the_near_one(self):
        points = self.points()

        assert _turn_bias(self.turned(points, 1.0))[1] is False
        assert _turn_bias(self.turned(points, -1.0))[1] is True


class TestRegionsDoNotOverlap:
    """The owner, after each region was cut to one photograph: *"sometimes the
    photographs still mix into each other, especially around the eyes and
    brows."*

    Which is where the regions overlap. A brow's box is grown by its margin and
    reaches down over the socket, an eye's reaches up, so a closed eyelid and a
    lowered brow were painted over the same skin in sequence â€” an average of two
    photographs after all the work to stop being one.
    """

    def regions(self, engine: Any, tmp_path: Path) -> tuple[Any, tuple[Any, ...]]:
        engine.load_avatar(make_descriptor(tmp_path))
        avatar = engine._loaded["avatar_test"]
        for role in ("source_eyes_closed", "source_brows_up", "source_mouth_open"):
            avatar.sources[role] = np.full_like(avatar.image, 255)
        points = _pixel_points(
            with_brow_anatomy(synthetic_landmarks()), avatar.image.shape[1], avatar.image.shape[0]
        )
        assert points is not None
        avatar.points = points
        return avatar, _build_regions(avatar.image, points, avatar.sources)

    def full(self, region: Any, shape: tuple[int, ...]) -> np.ndarray:
        """One region's mask back in the frame's own coordinates."""
        canvas = np.zeros(shape[:2], dtype=np.float32)
        x0, y0, x1, y1 = region.box
        canvas[y0:y1, x0:x1] = region.mask[:, :, 0]
        return canvas

    def test_no_pixel_is_claimed_by_two_regions(self, engine, tmp_path):
        avatar, regions = self.regions(engine, tmp_path)

        masks = [self.full(region, avatar.image.shape) for region in regions]
        total = np.sum(masks, axis=0)

        assert float(total.max()) <= 1.0 + 1e-6

    def test_the_eyes_keep_the_socket_and_the_brows_give_it_up(self, engine, tmp_path):
        avatar, regions = self.regions(engine, tmp_path)
        by_name = {region.name: self.full(region, avatar.image.shape) for region in regions}

        # The irises, which are unambiguously inside the socket. The eye ring's
        # topmost points sit on the brow ridge itself, and which side of the
        # line those fall on is what the anatomical split decides â€” see
        # :class:`TestTheBrowsAndTheEyesDivideTheRidge`.
        at_eyes = avatar.points[_IRIS_POSITIONS].astype(int)
        eyes = np.array([by_name["eyes"][y, x] for x, y in at_eyes])
        brows = np.array([by_name["brows"][y, x] for x, y in at_eyes])

        assert float(eyes.mean()) > 0.9, "the eyes lost their own socket"
        assert float(brows.max()) < 0.1, "a brow photograph is still painted over the eyes"

    def test_nothing_but_the_eyes_paints_inside_an_eye(self, engine, tmp_path):
        """The owner, on a shut eye: *"behind the closed-eye photograph, I think
        the open-eye one is still there."*

        Every other photograph in the library was taken with the eyes open â€” the
        whole-head turn rungs above all, and one of those is mixed in on nearly
        every frame because a head is never at exactly its resting angle. So any
        share of the opening the eyes do not hold is an open eye, and the mesh
        squeezes it into the slit a shut lid has become. Measured on the owner's
        avatar before this: the eyes held as little as **0.333** of their own
        opening, and 0.608 of their own iris.
        """
        avatar, regions = self.regions(engine, tmp_path)
        by_name = {region.name: self.full(region, avatar.image.shape) for region in regions}

        # The irises rather than the whole opening: the fence fades over the
        # opening's own outline, and what has to be clean is the part of the eye
        # a pupil is actually seen through.
        for _, iris in _EYE_PAIRS:
            inside = np.zeros(avatar.image.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(
                inside, cv2.convexHull(avatar.points[iris][:, :2].astype(np.int32)), 255
            )
            # Eroded by the fence's own width, which became necessary when the
            # iris gained four more points on 2026-08-24. The mesh used to
            # describe it as a diamond *inscribed* in the circle, so its hull
            # stopped well short of the lid; the octagon's vertices sit on the
            # outline itself, and on this avatar the iris is wider than the lid
            # opening is tall -- so the diagonal vertices now land in the fence
            # that fades along the eye's own edge. Six pixels of the hull fell
            # below the bar and every one of them was on its boundary.
            #
            # What the assertion is about is unchanged and is what the docstring
            # already says: the part of the eye a pupil is seen *through*. An
            # outline is a blend edge by construction.
            inside = cv2.erode(inside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            sel = inside > 0
            assert bool(sel.any()), "the erosion left no iris to test"

            others = sum(mask[sel] for name, mask in by_name.items() if name != "eyes")
            assert float(np.max(others)) < 0.05, "another photograph reaches inside the eye"
            assert float(by_name["eyes"][sel].min()) > 0.9, "the eyes do not hold their own iris"


class TestTheSmileLadder:
    """The owner's own proposal: *"we already have the clip where he laughs â€”
    ten photographs from where the smile starts through to the full smile."*

    One smile photograph faded in over a resting mouth fails twice at once.
    While it is faint there are two mouths on screen; and at any weight the
    shape it draws is the shape it was photographed at, so a small smile is a
    wide one at low opacity. The lips never stretch, the teeth simply appear â€”
    which is exactly what the owner described.
    """

    LADDER = ((0.0, None), (0.2, "smile@a"), (0.6, "smile@b"), (1.5, "smile@c"))

    def test_a_resting_mouth_uses_no_rung_at_all(self):
        assert _nearest_smile(self.LADDER, 0.0) == {}

    def test_a_faint_smile_is_below_the_ladder_and_left_to_the_mouth_shapes(self):
        """So an ordinary talking mouth is chosen by the shape it is making."""
        assert _nearest_smile(self.LADDER, _SMILE_LADDER_START / 2) == {}

    def test_the_nearest_rung_wins_and_only_it(self):
        """One photograph, not two: the rungs are stages of one movement and
        what the owner objected to was two mouths at once."""
        assert set(_nearest_smile(self.LADDER, 0.5)) == {"smile@b"}
        assert set(_nearest_smile(self.LADDER, 0.25)) == {"smile@a"}

    def test_the_rung_is_weighted_by_how_much_smile_there_is(self):
        """A faint smile is a faint photograph of a faint smile, so a speech
        shape can still win the mouth."""
        assert _nearest_smile(self.LADDER, 0.3)["smile@a"] == pytest.approx(0.3)

    def test_past_the_end_it_holds_the_widest_rather_than_inventing_one(self):
        assert set(_nearest_smile(self.LADDER, 3.0)) == {"smile@c"}

    def test_a_ladder_of_one_rung_is_not_a_ladder(self):
        assert _nearest_smile(((0.0, None),), 1.0) == {}

    def test_the_axis_keeps_rising_after_the_blendshape_saturates(self):
        """The smile blendshape reaches 1.0 at the open smile and stays there
        through the laugh, so the opening has to carry the rest â€” otherwise the
        ladder stops exactly where the teeth begin."""
        neutral = _Neutral()
        # A resting face first: the floor is learned from what a face does when
        # it is doing nothing, exactly as every other expression here is.
        _smile_amount(expressive_tracking(gap=0.002, smile=0.0), neutral)

        closed = _smile_amount(expressive_tracking(gap=0.002, smile=1.0), neutral)
        wide = _smile_amount(expressive_tracking(gap=0.090, smile=1.0), neutral)

        assert wide > closed > 0.0

    def test_a_mouth_open_without_a_smile_stays_off_the_ladder(self):
        neutral = _Neutral()
        _smile_amount(expressive_tracking(gap=0.002, smile=0.0), neutral)

        assert _smile_amount(expressive_tracking(gap=0.090, jaw=0.9), neutral) < _SMILE_LADDER_START


class TestTheEyeLadder:
    """The oldest complaint, and the one the owner ranks first.

    What was left after the gaze photographs and the squint were dropped is a
    single shut-eye photograph fading in over an open one: two sets of eyes
    while it is faint, and a lid that is fully shut at any weight, so a
    half-closed eye was a shut one at half opacity.
    """

    LADDER = ((0.0, None), (0.2, "eyes@a"), (0.5, "eyes@b"), (0.7, "eyes@c"))

    def test_an_open_eye_uses_no_rung(self):
        assert _nearest_rung(self.LADDER, 0.0, _EYES_LADDER_START) == {}

    def test_the_lid_moves_through_the_rungs_as_it_closes(self):
        assert set(_nearest_rung(self.LADDER, 0.22, _EYES_LADDER_START)) == {"eyes@a"}
        assert set(_nearest_rung(self.LADDER, 0.48, _EYES_LADDER_START)) == {"eyes@b"}
        assert set(_nearest_rung(self.LADDER, 0.69, _EYES_LADDER_START)) == {"eyes@c"}

    def test_never_two_rungs_at_once(self):
        """Which is the whole complaint: an open eye and a shut one together."""
        for amount in (0.15, 0.3, 0.45, 0.6, 0.75):
            assert len(_nearest_rung(self.LADDER, amount, _EYES_LADDER_START)) <= 1

    def settled(self) -> _Neutral:
        """A resting eye the estimate has had time to learn â€” see
        :data:`_LEVEL_WARMUP_FRAMES`."""
        neutral = _Neutral()
        for _ in range(_LEVEL_WARMUP_FRAMES + 1):
            _eye_amount(expressive_tracking(blink=(0.0, 0.0)), neutral)
            neutral.relative_rise("eyeOpen", _eye_opening(expressive_tracking(blink=(0.0, 0.0))))
        return neutral

    def test_the_amount_rises_as_the_lids_come_down(self):
        neutral = self.settled()

        half = _eye_amount(expressive_tracking(blink=(0.5, 0.5)), neutral)
        shut = _eye_amount(expressive_tracking(blink=(1.0, 1.0)), neutral)

        assert shut > half > 0.0

    def test_the_lids_say_nothing_until_the_resting_eye_is_learned(self):
        """The owner: *"I switched it on with my eyes open and it was showing
        closed eyes."*

        Measured on their own recording, inside one "sit still and relax"
        instruction: the tracker's eyelid reading climbs **36%** over the first
        second and a half while the face does not move. An estimate chasing that
        is always behind it, and the gap between the two reads as a lid coming
        down â€” a shut rung on 44 of the first 60 frames. There is no rate that
        wins a race against the thing it is estimating, so the lids say nothing
        until the estimate has stopped being new.
        """
        neutral = _Neutral()
        shut = expressive_tracking(blink=(1.0, 1.0))

        early = [_eye_amount(shut, neutral) for _ in range(_LEVEL_WARMUP_FRAMES)]

        assert early == [0.0] * _LEVEL_WARMUP_FRAMES

    def test_and_speaks_once_it_is_not(self):
        assert _eye_amount(expressive_tracking(blink=(1.0, 1.0)), self.settled()) > 0.0

    def test_a_blink_is_faster_than_the_rest_of_the_face(self):
        """A blink is 100-150 ms â€” three or four frames. At the mouth's rate the
        lid would still be half shut when the eye was open again."""
        eyes: dict[str, float] = {}
        mouth: dict[str, float] = {}
        for _ in range(3):
            _eased("eyes", 1.0, eyes)
            _eased("mouth", 1.0, mouth)

        assert eyes["eyes"] == pytest.approx(1.0)
        assert mouth["mouth"] < 0.5


class TestTheLidAxisIsSigned:
    """The owner: *"sometimes I do not widen my eyes at all and the screen
    does, as if I were mad. It is a bit funny."*

    Measured on their own recording, the wide-eye photograph fired on 4.6% of
    frames at a median weight of 0.11 â€” a sixth of a full stare over a face
    doing nothing. The same shape of defect as the squint, which was dropped for
    it, and the tilts, which were dropped for it.
    """

    LADDER = (
        (-0.30, "eyes@w"),
        (-0.15, "eyes@v"),
        (0.0, None),
        (0.20, "eyes@a"),
        (0.50, "eyes@b"),
    )

    def test_a_resting_eye_uses_the_portrait(self):
        assert _nearest_rung(self.LADDER, 0.0, _EYES_LADDER_START) == {}

    def test_and_so_does_a_flicker_of_one(self):
        """The lid measure moves a few percent frame to frame on a face doing
        nothing. That is not an expression."""
        assert _nearest_rung(self.LADDER, _EYES_LADDER_START / 2, _EYES_LADDER_START) == {}
        assert _nearest_rung(self.LADDER, -_EYES_LADDER_START / 2, _EYES_LADDER_START) == {}

    def test_a_widened_eye_is_its_own_photograph_rather_than_a_faded_stare(self):
        assert set(_nearest_rung(self.LADDER, -0.16, _EYES_LADDER_START)) == {"eyes@v"}
        assert set(_nearest_rung(self.LADDER, -0.31, _EYES_LADDER_START)) == {"eyes@w"}

    def test_a_closing_eye_still_walks_the_other_way(self):
        assert set(_nearest_rung(self.LADDER, 0.22, _EYES_LADDER_START)) == {"eyes@a"}
        assert set(_nearest_rung(self.LADDER, 0.55, _EYES_LADDER_START)) == {"eyes@b"}

    def test_the_amount_is_measured_from_rest_not_from_the_widest(self):
        """A ceiling that follows the most open the eye has been puts rest a
        third of the way shut on its own scale, and a face doing nothing then
        wears a part-closed lid."""
        neutral = _Neutral()
        _eye_amount(expressive_tracking(blink=(0.0, 0.0)), neutral)

        assert _eye_amount(expressive_tracking(blink=(0.0, 0.0)), neutral) == pytest.approx(
            0.0, abs=0.02
        )

    def test_the_rung_in_hand_is_kept_until_another_is_clearly_nearer(self):
        """Every change is a change of photograph, so a boundary crossed by
        noise is a visible flicker."""
        just_past = 0.11  # a hair nearer 0.20 than the -0.15 rung it holds
        assert set(_nearest_rung(self.LADDER, just_past, 0.05, "eyes@a")) == {"eyes@a"}

    def test_but_a_real_movement_moves_it_at_once(self):
        assert set(_nearest_rung(self.LADDER, 0.50, 0.05, "eyes@v")) == {"eyes@b"}


class TestTheBrowsAndTheEyesDivideTheRidge:
    """The owner, on a screenshot taken while opening their eyes: two brows, a
    dark seam between them, and an angry face for an instant.

    Giving the whole overlap to the eyes was right for the socket and wrong for
    the ridge above it. A brow rung could not paint over the *portrait's* brow,
    because that sits inside what the eyes had claimed â€” so the portrait's brow
    stayed and the rung's appeared above it. It happened when the eyes opened
    because that is when an eye rung is on at all.
    """

    def masks(self, engine: Any, tmp_path: Path) -> tuple[Any, dict[str, np.ndarray]]:
        engine.load_avatar(make_descriptor(tmp_path))
        avatar = engine._loaded["avatar_test"]
        for role in ("source_eyes_closed", "source_brows_up"):
            avatar.sources[role] = np.full_like(avatar.image, 255)
        points = _pixel_points(
            with_brow_anatomy(synthetic_landmarks()), avatar.image.shape[1], avatar.image.shape[0]
        )
        assert points is not None
        avatar.points = points
        regions = _build_regions(avatar.image, points, avatar.sources)
        full = {}
        for region in regions:
            canvas = np.zeros(avatar.image.shape[:2], dtype=np.float32)
            x0, y0, x1, y1 = region.box
            canvas[y0:y1, x0:x1] = region.mask[:, :, 0]
            full[region.name] = canvas
        return avatar, full

    def test_the_brows_reach_the_brow_line(self, engine, tmp_path):
        """Which is what lets a brow rung cover the portrait's own brow."""
        avatar, full = self.masks(engine, tmp_path)

        at_brows = avatar.points[_BROW_POSITIONS].astype(int)
        brows = np.array([full["brows"][y, x] for x, y in at_brows])

        assert float(brows.mean()) > 0.9, "the brows cannot paint over their own brow"

    def test_and_the_eyes_do_not(self, engine, tmp_path):
        avatar, full = self.masks(engine, tmp_path)

        at_brows = avatar.points[_BROW_POSITIONS].astype(int)
        eyes = np.array([full["eyes"][y, x] for x, y in at_brows])

        assert float(eyes.max()) < 0.5, "an eye photograph still covers the brow"


class TestTheBrowAxisIsSigned:
    """The last family that was two named photographs, and the owner reported
    both halves of what that cost.

    *"A slight raise takes on a surprised face â€” the brows do not actually go
    up"*, because the photograph shows a full raise faded in by how far they
    moved. And *"two brows, and two more slightly above, faint"*, because the
    photograph's brows are somewhere the portrait's are not.
    """

    LADDER = (
        (-0.60, "brows@d"),
        (-0.20, "brows@e"),
        (0.0, None),
        (0.25, "brows@a"),
        (0.80, "brows@b"),
    )

    def test_resting_brows_use_the_portrait(self):
        assert _nearest_rung(self.LADDER, 0.0, _BROWS_LADDER_START) == {}

    def test_a_small_raise_is_a_photograph_of_a_small_raise(self):
        """Not a big one at low opacity, which is what read as surprise."""
        assert set(_nearest_rung(self.LADDER, 0.22, _BROWS_LADDER_START)) == {"brows@a"}

    def test_a_big_raise_reaches_the_big_photograph(self):
        assert set(_nearest_rung(self.LADDER, 0.85, _BROWS_LADDER_START)) == {"brows@b"}

    def test_lowered_brows_are_the_same_axis_the_other_way(self):
        assert set(_nearest_rung(self.LADDER, -0.25, _BROWS_LADDER_START)) == {"brows@e"}
        assert set(_nearest_rung(self.LADDER, -0.70, _BROWS_LADDER_START)) == {"brows@d"}

    def test_raised_and_lowered_can_never_both_be_on_screen(self):
        """Which two named photographs could not promise, and one signed axis
        does by construction."""
        for amount in (-0.9, -0.5, -0.2, 0.2, 0.5, 0.9):
            assert len(_nearest_rung(self.LADDER, amount, _BROWS_LADDER_START)) <= 1

    def moved(self, by: float) -> FaceTrackingResult:
        """A face with its brows shifted, in normalised landmark coordinates.

        Measured on the brows themselves now, not on ``browInnerUp`` â€” which
        reacts to the eyelid as well, reached its maximum on 5% of the owner's
        recording, and so put a fully raised brow on the avatar for an instant
        every time their eyes opened.
        """
        from sha256.avatar.engine_warp import _LEFT_BROW, _RIGHT_BROW

        points = with_brow_anatomy(synthetic_landmarks())
        points[10, 1], points[152, 1] = 0.2, 0.8  # forehead and chin
        points[168, 1] = 0.5  # the nose bridge the brows are measured against
        for index in _LEFT_BROW + _RIGHT_BROW:
            points[index, 1] = 0.36 - by
        return tracking_from(points)

    def test_the_amount_is_signed_around_a_resting_face(self):
        neutral = _Neutral()
        _brow_amount(self.moved(0.0), neutral)  # learns where they rest

        raised = _brow_amount(self.moved(0.02), neutral)
        lowered = _brow_amount(self.moved(-0.02), neutral)

        assert raised > 0.0 > lowered

    def test_a_blendshape_is_not_consulted_at_all(self):
        """So a tracker that spikes a brow blendshape on a blink cannot move the
        avatar's brows."""
        neutral = _Neutral()
        _brow_amount(self.moved(0.0), neutral)

        assert _brow_amount(self.moved(0.0), neutral) == pytest.approx(0.0, abs=0.02)


class TestTheBrowReadingSurvivesBeingTakenAgain:
    """Why the brow ladder could not be filed until 2026-08-24, in two tests.

    The reading used to be taken from the **nose bridge**, and the bridge is the
    part that moves: over 127 frames from the six brow clips, each detected once
    on the frame and once on the crop cut from it, the brow points shifted 0.0038
    of an eye span and the bridge 0.0209. Every rebuilt ladder came out worse
    than the one it replaced because a rung was filed from a number that changed
    when the photograph was written.

    Both tests below are about what a *re-detection* may not change. A rung is
    measured in one frame and re-read in another, and the two have to agree, or
    the ladder is a row of labels that do not describe their photographs.
    """

    def face(self, *, brow: float = 0.36, bridge: float = 0.5, stretch: float = 1.0):
        from sha256.avatar.engine_warp import _LEFT_BROW, _RIGHT_BROW

        points = with_brow_anatomy(synthetic_landmarks())
        points[10, 1], points[152, 1] = 0.2, 0.8  # forehead and chin
        points[168, 1] = bridge
        for index in _LEFT_BROW + _RIGHT_BROW:
            points[index, 1] = brow
        # A different frame shape, in normalised coordinates: x is normalised by
        # the width and y by the height, so a 16:9 frame and a square one place
        # the same face at different x for the same pixels.
        points[:, 0] *= stretch
        return tracking_from(points)

    def test_the_nose_bridge_cannot_move_the_reading(self):
        """It moved it by more than the brows did, which is the whole defect."""
        high = _brow_height(self.face(bridge=0.42))
        low = _brow_height(self.face(bridge=0.58))

        assert high == pytest.approx(low, abs=1e-6)

    def test_the_brows_still_move_it(self):
        """A guard on the test above: a reading nothing can move is not a fix."""
        assert _brow_height(self.face(brow=0.30)) > _brow_height(self.face(brow=0.40))

    def test_a_change_of_aspect_ratio_cannot_move_it(self):
        """Which is why the denominator is the face height and not the eye span.

        The eye span is steadier across a re-detection -- 1.0% of the axis
        against 1.9% -- and it is horizontal, so dividing a vertical distance by
        it carries the frame's shape into the answer: the same face reads 0.137
        in a square portrait and 0.367 in a 16:9 frame. Rungs are cropped to
        whatever shape the face needs and the portrait is square, so that
        difference would land straight in the ladder.
        """
        square = _brow_height(self.face())
        widescreen = _brow_height(self.face(stretch=16 / 9))

        assert square == pytest.approx(widescreen, abs=1e-6)


class TestTheSpeechAxisSurvivesTheFrameShape:
    """The same defect as the two tests above, in the last measure that had it.

    :func:`_lip_parting` divided the lips' gap by the eye span until 2026-08-26,
    which multiplied the reading by the frame's aspect ratio. Live it cancelled,
    and every clip the library was built from happened to be 16:9, so nothing
    saw it. The first 9:16 source clip broke it: the same sentence reached 0.64
    of the axis where the 16:9 take reached 2.30, and the grid would have
    installed a wide open mouth as a barely parted rung — answering a mouth that
    is hardly open with a photograph of one wide open.

    The lids and the brows were fixed the same way and hold the same guard.
    """

    def face(self, *, gap: float = 0.10, stretch: float = 1.0):
        from sha256.avatar.engine_warp import _LOWER_INNER_LIP, _UPPER_INNER_LIP

        points = synthetic_landmarks()
        points[10, 1], points[152, 1] = 0.2, 0.8  # forehead and chin
        points[_UPPER_INNER_LIP, 1] = 0.55 - gap / 2
        points[_LOWER_INNER_LIP, 1] = 0.55 + gap / 2
        # As in the brow tests: x is normalised by the frame's width and y by
        # its height, so stretching x is what a wider frame does to a face.
        points[:, 0] *= stretch
        return tracking_from(points)

    def test_a_change_of_aspect_ratio_cannot_move_it(self):
        square = _lip_parting(self.face())
        widescreen = _lip_parting(self.face(stretch=16 / 9))

        assert square == pytest.approx(widescreen, abs=1e-6)

    def test_the_lips_still_move_it(self):
        """A guard on the test above: a reading nothing can move is not a fix."""
        assert _lip_parting(self.face(gap=0.16)) > _lip_parting(self.face(gap=0.04))

    def test_the_axis_a_rung_is_filed_on_carries_both(self):
        """:func:`_speech_amount` is the reading a rung is filed by, over two constants.

        The constants were rescaled with the fix so that the live axis did not
        move. Asserting on them here would test the rescale; what has to hold is
        that the shape of the frame is gone and the mouth is still in.
        """
        assert _speech_amount(self.face(gap=0.16, stretch=16 / 9)) == pytest.approx(
            _speech_amount(self.face(gap=0.16)), abs=1e-6
        )
        assert _speech_amount(self.face(gap=0.16)) > _speech_amount(self.face(gap=0.04)) > 0.0


class TestTheBrowsHoldStillThroughABlink:
    """The owner, twice: two brows and an angry face, *"for a very short time"*,
    while the eyes open.

    The brow landmarks ride the blink. Measured over their recording, where the
    brows sit correlates with the lids at r=+0.24 against the nose bridge and
    -0.22 against the forehead â€” neither anchor escapes it â€” and a blink reads
    as a brow raise of 0.63, which is most of the way up a ladder that is shown
    whole. Three frames of a raised brow, every blink.
    """

    LADDER = ((-0.5, "brows@d"), (0.0, None), (0.3, "brows@a"), (0.9, "brows@b"))

    def test_a_blink_does_not_move_the_brows(self):
        frozen = _nearest_rung(self.LADDER, 0.9, 0.12, "brows@a", freeze=True)

        assert set(frozen) == {"brows@a"}, "the blink raised the brows"

    def test_and_they_do_not_leave_either(self):
        """Leaving would be a change of picture too, and the brows have not
        moved â€” only the eyes have."""
        assert _nearest_rung(self.LADDER, 0.0, 0.12, "brows@a", freeze=True) != {}

    def test_once_the_eyes_are_back_the_brows_move_again(self):
        assert set(_nearest_rung(self.LADDER, 0.9, 0.12, "brows@a")) == {"brows@b"}

    def test_a_frozen_ladder_with_nothing_in_hand_stays_empty(self):
        assert _nearest_rung(self.LADDER, 0.9, 0.12, None, freeze=True) == {}


class TestARungWhoseFeaturesLandInAnotherRegion:
    """The owner picked the files off the screen: *"eyes_c014, eyes_c017,
    brows_d100 â€” one of those is being drawn."*

    Two of them were. A rung is a whole face, cut into by whichever region it
    belongs to, and the regions are drawn around the *portrait's* features. Both
    of those eye rungs had their brows 18 px below the portrait's brow line â€”
    inside the socket â€” because the head was pitched in those frames and the
    similarity fit could not undo it. Painted in the eye region, that is a
    second brow under the real one, exactly where he saw it.
    """

    def face(self, brow: float, eye: float) -> np.ndarray:
        points = np.zeros((MESH_POINT_COUNT, 2), dtype=np.float32)
        points[_BROW_POSITIONS, 1] = brow
        points[_EYE_POSITIONS, 1] = eye
        return points

    def test_an_eye_rung_may_not_bring_a_brow_into_the_socket(self):
        base = self.face(brow=100.0, eye=140.0)
        rung = self.face(brow=150.0, eye=170.0)

        assert _misplaced(_EYES_PREFIX, rung, base) is not None

    def test_an_ordinary_eye_rung_passes(self):
        base = self.face(brow=100.0, eye=140.0)
        rung = self.face(brow=95.0, eye=145.0)

        assert _misplaced(_EYES_PREFIX, rung, base) is None

    def test_a_brow_rung_may_not_bring_an_eye_above_the_brow_line(self):
        base = self.face(brow=100.0, eye=140.0)
        rung = self.face(brow=60.0, eye=80.0)

        assert _misplaced(_BROWS_PREFIX, rung, base) is not None

    def test_and_the_reason_is_said_in_pixels(self):
        """So a rung dropped at load can be found and replaced rather than
        wondered about."""
        base = self.face(brow=100.0, eye=140.0)
        rung = self.face(brow=150.0, eye=170.0)

        assert "10 px" in str(_misplaced(_EYES_PREFIX, rung, base))


class TestAShutEyesIrisIsTuckedOutOfTheWay:
    """The owner: *"the pupils, very thin, low in height and long in width â€” a
    flat oval, for a very short time."*

    Containment shrinks the iris to fit the lid and never puts it anywhere else,
    so at a shut lid the fan is a wide flat sliver lying along the lash line â€”
    and the lash line is the darkest thing in the picture.

    **Taking the fan out of the mesh was the first answer and it was wrong.**
    The render was clean where the artefact had been, but those were the only
    triangles covering the inside of the lid, and the rigid backdrop showed
    through the hole: the composite placed by one similarity transform carries
    its own lash line a few pixels off the mesh's, so a second, thinner, darker
    line appeared inside the shut lid. The owner marked it with an arrow on both
    eyes. Measured after: 27 to 252 pixels of every shut eye were painted by no
    triangle at all.

    So the ring is tucked in toward the middle of the lid instead. The fan stays,
    the hole never opens, and what it paints is the lid's own pixels.

    The line between open and shut is drawn against this portrait's own resting
    lid rather than as a number, and there is room for it: measured over the
    library and 1847 frames, an open lid's height is 0.298 to 0.450 of its own
    width and a shut one's is 0.106 to 0.128. Nothing in between.
    """

    def mesh(self, left: float, right: float) -> np.ndarray:
        """A mesh whose two lid rings have the given height over width."""
        points = np.zeros((MESH_POINT_COUNT, 2), dtype=np.float32)
        for (eye, iris), aspect in zip(_EYE_PAIRS, (left, right), strict=True):
            ring = points[eye]
            ring[:, 0] = np.linspace(0.0, 100.0, len(eye))
            ring[: len(eye) // 2, 1] = 0.0
            ring[len(eye) // 2 :, 1] = 100.0 * aspect
            points[eye] = ring
            points[iris, 0] = np.linspace(35.0, 65.0, len(iris))
            points[iris, 1] = 50.0 * aspect
        return points

    def spread(self, points: np.ndarray, iris: list[int]) -> float:
        ring = points[iris]
        return float(np.linalg.norm(ring - ring.mean(axis=0), axis=1).max())

    def test_an_open_lid_measures_about_a_third_of_its_own_width(self):
        assert _lid_aspect(self.mesh(0.32, 0.32), _EYE_PAIRS[0][0]) == pytest.approx(0.32)

    def test_an_open_eye_is_not_shut(self):
        assert not _shut(self.mesh(0.32, 0.32), _EYE_PAIRS[0][0], 0.32)

    def test_a_closed_one_is(self):
        assert _shut(self.mesh(0.11, 0.11), _EYE_PAIRS[0][0], 0.32)

    def test_a_lid_with_no_width_at_all_counts_as_shut(self):
        """Rather than dividing by it."""
        points = np.zeros((MESH_POINT_COUNT, 2), dtype=np.float32)

        assert _shut(points, _EYE_PAIRS[0][0], 0.32)

    def test_the_rule_is_a_ratio_so_scale_does_not_enter_it(self):
        """A face filling the frame and a face across the room answer it the
        same way."""
        far = self.mesh(0.32, 0.32)

        assert not _shut(far * 7.0, _EYE_PAIRS[0][0], 0.32)

    def test_a_shut_eye_has_its_iris_pulled_in(self):
        points = self.mesh(0.11, 0.11)
        before = self.spread(points, _EYE_PAIRS[0][1])

        _contain_irises(points, (0.32, 0.32))

        assert self.spread(points, _EYE_PAIRS[0][1]) < before * 0.5

    def test_an_open_one_keeps_its_iris_where_it_was(self):
        points = self.mesh(0.32, 0.32)
        before = self.spread(points, _EYE_PAIRS[0][1])

        _contain_irises(points, (0.32, 0.32))

        assert self.spread(points, _EYE_PAIRS[0][1]) == pytest.approx(before, rel=0.01)

    def test_only_the_shut_eye_of_the_two(self):
        """A wink is a face doing two different things at once."""
        points = self.mesh(0.11, 0.32)
        before = [self.spread(points, iris) for _, iris in _EYE_PAIRS]

        _contain_irises(points, (0.32, 0.32))

        after = [self.spread(points, iris) for _, iris in _EYE_PAIRS]
        assert after[0] < before[0] * 0.5
        assert after[1] == pytest.approx(before[1], rel=0.01)

    def test_the_tucked_ring_ends_up_inside_the_lid_holding_it(self):
        points = self.mesh(0.11, 0.11)

        _contain_irises(points, (0.32, 0.32))

        eye, iris = _EYE_PAIRS[0]
        ring, tucked = points[eye], points[iris]
        assert tucked[:, 0].min() > ring[:, 0].min()
        assert tucked[:, 0].max() < ring[:, 0].max()
        assert tucked[:, 1].min() > ring[:, 1].min()
        assert tucked[:, 1].max() < ring[:, 1].max()

    def test_it_keeps_some_size_rather_than_collapsing_to_a_point(self):
        """The version that clipped the iris to the lid outline drove the fan's
        triangles to no area, and an affine fitted to three nearly-collinear
        points put a bright speck where the dark one had been."""
        points = self.mesh(0.11, 0.11)

        _contain_irises(points, (0.32, 0.32))

        assert self.spread(points, _EYE_PAIRS[0][1]) > 1.0

    def test_without_a_resting_lid_nothing_is_tucked(self):
        """Which is what the load path wants: it is measuring the resting lid,
        so it cannot yet be compared against it."""
        points = self.mesh(0.11, 0.11)
        before = self.spread(points, _EYE_PAIRS[0][1])

        _contain_irises(points)

        assert self.spread(points, _EYE_PAIRS[0][1]) == pytest.approx(before, rel=0.01)


class TestALidIsNotFadedOutByATurn:
    """The owner: *"eyes shut, head turned slightly, and it is still there."*

    Every eye photograph is of a face looking straight ahead, so at a turn the
    carry has to place one on a head it was not taken on and `_trust` says how
    well it managed. Scaling the photograph by that is right for a mouth, which
    differs from what is under it mostly in texture, and wrong for a lid, which
    differs from what is under it in whether the eye is **open**.

    Reproduced from the owner's own recording, which contains the pair
    `driving.mkv` never does â€” the head past 20 degrees with the lids down. The
    eye rung was applied at 1.00 at 19 degrees falling to 0.82 at 28, and over
    the whole recording on 1572 of the 2668 frames the region was carried at
    all. A shut lid at four fifths of an opacity leaves a fifth of an open eye
    showing, which is the thin dark line inside the lid that they marked with an
    arrow.
    """

    def test_a_mouth_photograph_still_fades_as_the_head_turns_from_it(self):
        assert _carried_share("mouth", 1.0, 0.4) == pytest.approx(0.4)

    def test_a_lid_does_not(self):
        assert _carried_share("eyes", 1.0, 0.4) == 1.0

    def test_nor_at_the_worst_fit_that_is_still_a_fit(self):
        assert _carried_share("eyes", 1.0, 0.01) == 1.0

    def test_and_is_gone_once_there_is_no_fit_at_all(self):
        """Past ``_CARRY_LOST``, where it always was."""
        assert _carried_share("eyes", 1.0, 0.0) == 0.0

    def test_the_region_s_own_share_is_still_what_is_being_scaled(self):
        """A photograph half way through arriving does not become whole because
        the head turned."""
        assert _carried_share("eyes", 0.5, 0.4) == pytest.approx(0.5)


class TestTheBuildersCanAskTheSameQuestion:
    """A refusal at load is a rung lost, and the builder that chose it never
    hears about it.

    Each ladder builder takes the sharpest frame in each band of the movement
    and nothing else, so a band whose sharpest frame is one the engine will
    refuse ends up empty. The owner's eye ladder had two such holes, at a tenth
    and a fifth of a closing, and a blink therefore had to be drawn by fading a
    shut photograph in over a wide-open portrait â€” which is the iris seen
    through a closed eyelid. So the check is offered to the builders, and they
    ask it before choosing rather than finding out afterwards.
    """

    def face(self, *, brow: float, eye: float, shift: float = 0.0) -> np.ndarray:
        """A face whose stable landmarks are spread enough to fit a transform."""
        points = np.zeros((MESH_POINT_COUNT, 2), dtype=np.float32)
        spread = np.linspace(-60.0, 60.0, len(_STABLE_POSITIONS), dtype=np.float32)
        points[_STABLE_POSITIONS, 0] = spread
        points[_STABLE_POSITIONS, 1] = np.abs(spread) + shift
        points[_BROW_POSITIONS, 1] = brow + shift
        points[_EYE_POSITIONS, 1] = eye + shift
        return points

    def test_a_frame_that_would_be_refused_says_so_before_it_is_chosen(self):
        base = self.face(brow=100.0, eye=140.0)
        rung = self.face(brow=150.0, eye=170.0)

        assert refused_as_rung(_EYES_PREFIX, rung, base) is not None

    def test_an_ordinary_one_does_not(self):
        base = self.face(brow=100.0, eye=140.0)
        rung = self.face(brow=95.0, eye=145.0)

        assert refused_as_rung(_EYES_PREFIX, rung, base) is None

    def test_it_is_asked_of_the_frame_where_the_placement_puts_it(self):
        """Not where it was photographed. The whole point of the check is where
        the features *land* once the photograph is on the portrait's frame, so a
        rung that is merely offset is fine and one that is posed differently is
        not."""
        base = self.face(brow=100.0, eye=140.0)
        offset = self.face(brow=95.0, eye=145.0, shift=400.0)

        assert refused_as_rung(_EYES_PREFIX, offset, base) is None


class TestARungIsShownWhole:
    """The owner, after the eye ladder went in: *"while it closes, the open eye
    is still there over the closed one, at low opacity."*

    Right again, and the fade was the cause. For a named photograph the weight
    says how much of that expression this frame has, and fading it in against
    the portrait is how an expression arrives. For a rung it says no such thing:
    the amount is already in *which rung was chosen*, so mixing a
    three-tenths-shut eye with a wide-open portrait puts the wide-open one back
    underneath at exactly the opacity being reported.
    """

    def settle(self, weights: dict[str, float], frames: int = 20) -> dict[str, float]:
        holding: dict[str, str] = {}
        applied: dict[str, float] = {}
        kept: dict[str, float] = {}
        for _ in range(frames):
            kept = _one_per_region(dict(weights), holding, applied)
        return kept

    def test_a_rung_leaves_nothing_of_the_portrait_underneath(self):
        assert self.settle({"eyes@a": 0.3})["eyes@a"] == 1.0

    def test_even_a_faint_one(self):
        """A tenth-shut eye is a photograph of a tenth-shut eye, not a tenth of
        a photograph of a shut one."""
        assert self.settle({"eyes@a": 0.12})["eyes@a"] == 1.0

    def test_a_smile_rung_the_same(self):
        assert self.settle({"smile@b": 0.4})["smile@b"] == 1.0

    def test_a_named_photograph_still_fades_in(self):
        """Because for one of those the weight is the only thing that says how
        far into the expression this frame is."""
        assert self.settle({"source_mouth_o": 0.3})["source_mouth_o"] < 1.0


class TestTheLidsArriveWhole:
    """*"At a blink, a small piece of the iris shows through the closed eyelid."*

    Shown whole answered how much of a rung is applied once it has settled; this
    answers the frames on the way there. The lids ease at ten a second, so a rung
    took three frames to arrive â€” and a blink is three or four frames, so the
    fade lasted the whole blink and every one of them was a mix of a shut
    photograph and the wide-open portrait underneath it.

    Rendered frame by frame on the owner's recording, one blink: the closing
    frames sit at 33% and 67% of the way in, and so do the opening ones. Four
    frames of a translucent iris under a closed lid, per blink.
    """

    def frames(
        self, weights: dict[str, float], count: int
    ) -> tuple[list[dict[str, float]], dict[str, str], dict[str, float]]:
        holding: dict[str, str] = {}
        applied: dict[str, float] = {}
        seen = [_one_per_region(dict(weights), holding, applied) for _ in range(count)]
        return seen, holding, applied

    def test_an_eye_rung_is_whole_on_its_very_first_frame(self):
        seen, _, _ = self.frames({"eyes@shut": 0.6}, 1)

        assert seen[0]["eyes@shut"] == 1.0

    def test_and_gone_on_the_first_frame_it_is_not_wanted(self):
        """A fade out is the same double exposure as a fade in, and the owner
        saw the eye ghost on the way back up as well as on the way down."""
        _, holding, applied = self.frames({"eyes@shut": 0.6}, 3)

        gone = _one_per_region({}, holding, applied)

        assert gone == {} and holding == {} and applied == {}

    def test_a_smile_rung_still_arrives_over_a_third_of_a_second(self):
        """Nothing else on a face moves inside the time a lid takes to fall, and
        a smile that appeared in one frame would read as a cut."""
        seen, _, _ = self.frames({"smile@wide": 0.6}, 1)

        assert seen[0]["smile@wide"] < 1.0

    def test_a_named_eye_photograph_still_fades(self):
        """Which is what an avatar with no eye ladder falls back to: for one of
        those the weight is the only thing saying how far shut the eye is."""
        seen, _, _ = self.frames({"source_eyes_closed": 0.6}, 1)

        assert seen[0]["source_eyes_closed"] < 1.0


class TestTheEasingIsMeasuredInSeconds:
    """Asked to try 60 fps, and it would have made every expression take twice
    as long â€” the engine appearing to get slower on a faster machine."""

    def test_the_same_expression_takes_the_same_time_at_any_rate(self):
        slow: dict[str, float] = {}
        fast: dict[str, float] = {}
        for _ in range(15):
            _eased("mouth", 1.0, slow, 1 / 30)
        for _ in range(30):
            _eased("mouth", 1.0, fast, 1 / 60)

        assert slow["mouth"] == pytest.approx(fast["mouth"])

    def test_a_long_gap_between_frames_does_not_let_it_jump(self):
        applied: dict[str, float] = {}

        _eased("mouth", 1.0, applied, 5.0)

        assert applied["mouth"] < 1.0


class TestTheBrowsDoNotDouble:
    """The owner: *"two brows, and two more slightly above, faint."*

    A raised-brow photograph fading in over the portrait puts the base's brows
    and the photograph's on screen together. Every region has that while a
    photograph is faint; for the mouth it is a texture difference and survivable,
    for a brow it is a *position* difference, which reads as a second copy.
    """

    def test_a_small_raise_does_not_bring_the_photograph_in_at_all(self):
        kept = _one_per_region({"source_brows_up": 0.25}, {}, {})

        assert kept == {}

    def test_a_real_raise_still_reaches_the_photograph_in_full(self):
        holding: dict[str, str] = {}
        applied: dict[str, float] = {}
        for _ in range(20):
            kept = _one_per_region({"source_brows_up": 1.0}, holding, applied)

        assert kept["source_brows_up"] == 1.0

    def test_the_mouth_keeps_its_own_curve(self):
        """Only the brows start late. A mouth photograph differs from the base
        in texture, and fading one in is what an expression arriving looks
        like."""
        kept = _one_per_region({"source_mouth_o": 0.25}, {}, {})

        assert kept["source_mouth_o"] > 0.0


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
        same either way â€” a full width across and 0.6 of that tall â€” so what the
        assertions measure does not move with the point count.
        """
        from sha256.avatar.engine_warp import _EYE_PAIRS

        built = _pixel_points(synthetic_landmarks(), 400, 400)
        assert built is not None
        points: np.ndarray = built
        for index, (socket, iris) in enumerate(_EYE_PAIRS):
            centre = np.array([150.0 + 100.0 * index, 180.0], dtype=np.float32)
            width = 40.0
            around_lid = np.linspace(0.0, 2.0 * np.pi, len(socket), endpoint=False)
            ring = np.stack([-0.5 * np.cos(around_lid), -0.3 * np.sin(around_lid)], axis=1).astype(
                np.float32
            )
            points[socket] = centre + ring * width

            pupil = centre + np.array([dx, dy], dtype=np.float32) * width
            # The iris the same way as the socket, and for the same reason: it
            # is a centre followed by however many boundary points the mesh
            # declares. It was four and is eight -- see `_IRIS_ARCS_PER_EYE` --
            # and a fixed list of five silently stopped describing it.
            points[iris[0]] = pupil
            around_iris = np.linspace(0.0, 2.0 * np.pi, len(iris) - 1, endpoint=False)
            circle = np.stack([np.cos(around_iris), np.sin(around_iris)], axis=1).astype(np.float32)
            points[iris[1:]] = pupil + circle * 0.2 * width
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
        from sha256.avatar.engine_warp import _damp_brows

        built = _pixel_points(synthetic_landmarks(), 500, 500)
        assert built is not None
        source = built
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
        """Never a third rung, whatever the spacing.

        Measured at the middle of the gap, which is the only place two rungs
        this far apart are both on screen at all â€” twenty degrees is far wider
        than the ladder's own three, and across a gap that wide the nearer rung
        is shown whole. See :class:`TestTheLadderDoesNotAverageAcrossAGap`.
        """
        from sha256.avatar.engine_warp import _ladder_weights

        weights = _ladder_weights(self.LADDER, -20.0)

        assert set(weights) == {"turn@a", "turn@b"}
        assert weights["turn@a"] == pytest.approx(0.5)
        assert weights["turn@b"] == pytest.approx(0.5)

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

        The rule is stricter than it was: with the nearest rung ten degrees
        away, two degrees of turn now leaves the portrait *untouched* rather
        than mixing a fifth of a ten-degree photograph into it.
        """
        from sha256.avatar.engine_warp import _ladder_weights

        assert _ladder_weights(self.LADDER, 2.0) == {}
        assert sum(_ladder_weights(self.LADDER, 9.0).values()) <= 1.0


class TestPosePhotographsKeepTheirGeometry:
    """A turned photograph brings its angle with it, and the mesh follows.

    The rungs used to be mesh-warped onto the frontal portrait's layout, which
    at 35 degrees is not a correction but a fold: a seam down the face, the far
    eye and brow drawn twice, the mouth smeared sideways. What each rung holds
    now is the picture as it was shot, and its own landmarks alongside â€” so the
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
        cannot follow â€” a real turn foreshortens, and that leftover is what
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

    def _turn_from_rest(self, avatar, tracking) -> np.ndarray:
        """Blend ``tracking`` after the engine has been shown a face at rest.

        A single frame in isolation *is* the resting angle â€” the neutral takes
        its first reading as straight ahead, which is what stops a camera above
        a monitor from putting a permanent tilt in the avatar â€” so a test that
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
        say about the eyes â€” they freeze at whatever the turned photograph
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
        """Nothing is asked for, so nothing is taken over â€” item 12 still holds."""
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
        beside the mouth on screen â€” 70 px beside it at 36 degrees on the owner's
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
    is under the mask is the user's face â€” so whatever the correction can
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

        # A plane cannot reject it entirely â€” an off-centre band has a linear
        # component â€” but the blurred field it replaced transferred 52%.
        assert self.transferred(pattern) < 0.35

    def test_fine_detail_is_not_transferred_at_all(self):
        stripes = np.sign(np.sin(np.arange(self.SIZE, dtype=np.float32) * 1.4))
        pattern = np.broadcast_to(stripes, (self.SIZE, self.SIZE))

        assert abs(self.transferred(np.ascontiguousarray(pattern))) < 0.05


class TestTrackingLoss:
    """What happens in the frames where the tracker loses the face.

    This is a privacy requirement, not a smoothness one. Passing the camera
    through a dropout puts the user's real face on the call â€” which is the one
    thing the product exists to prevent â€” and it is what the owner saw as a
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
        on the theory that the user had left. If they had not left â€” a hand over
        the face, an extreme angle, bad light â€” their real face went out on the
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
    mouthSmile 0.54, and so does a closed-mouth smile â€” so weighting the
    teeth-bearing photographs by the smile blendshape put teeth into a smile
    that had none.
    """

    def test_a_closed_smile_shows_no_teeth(self):
        weights = weights_from_rest(expressive_tracking(gap=0.001, smile=0.6))

        assert "source_smile" not in weights

    def test_an_open_smile_does(self):
        weights = weights_from_rest(expressive_tracking(gap=0.06, smile=0.6))

        assert weights["source_smile"] > 0.5

    def test_the_open_mouth_photo_is_gated_the_same_way(self):
        """A dropped jaw with the lips still together shows no teeth either."""
        shut = _source_weights(expressive_tracking(gap=0.001, jaw=0.7))
        open_ = _source_weights(expressive_tracking(gap=0.09, jaw=0.7))

        assert "source_mouth_open" not in shut
        assert open_["source_mouth_open"] > 0.5

    def test_the_gate_is_gradual(self):
        """A hard threshold would pop the teeth in as the lips part."""
        weights = weights_from_rest(expressive_tracking(gap=0.03, smile=1.0))

        assert 0.1 < weights["source_smile"] < 0.9


class TestMouthInterior:
    """What fills the mouth when the user opens wider than the photograph did.

    Which is most of the time: no still portrait is taken mid-yawn. Placing the
    whole photograph and trusting the fit put its *lips* inside the live
    opening â€” a second, smaller mouth inside the first.
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
        portrait's lips are together â€” a sliver a few pixels tall. So however
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

        # Rest first, as the engine sees it: the smile is measured from the
        # mouth's own width against rest now, so a single frame in isolation is
        # by definition a face that is not smiling.
        smiling = expressive_tracking(gap=0.062, smile=1.0)
        _blend_sources(avatar, relaxed_like(smiling))
        _blend_sources(avatar, smiling)
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

    A wrong mapping here is invisible in a still and obvious in motion â€” the
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
        at 0.13 and the teeth stopped appearing â€” the photograph was loaded and
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
        """It is geometry now â€” see TestEyeGeometry. The four gaze photographs
        are no longer used, and an avatar that has them is not penalised for it.
        """
        for shape in ("eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookDownLeft"):
            weights = weights_from_rest(expressive_tracking(gap=0.0, extra={shape: 0.6}))
            assert not [role for role in weights if role.startswith("source_look_")]

    def test_a_closed_smile_asks_for_the_closed_smile_photo(self):
        """And not for the one with teeth in it."""
        weights = weights_from_rest(expressive_tracking(gap=0.001, smile=0.6))

        assert weights["source_smile_closed"] > 0.9
        assert "source_smile" not in weights

    def test_an_open_smile_asks_for_the_open_one_instead(self):
        weights = weights_from_rest(expressive_tracking(gap=0.09, smile=0.6))

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
        from the portrait's â€” so once a pose photograph began carrying its own
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
    point cloud â€” scattered points would describe a star.
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
    a background that is merely drifting â€” which is what the owner saw as the
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


class TestTheSpeechGridLetsGoOfAMouthTheFaceHasLeft:
    """The owner, on the running application: *"when I am talking it does not
    follow my mouth at all."*

    Measured on the eight seconds of `driving.mkv` where he speaks, the margin
    alone froze the photograph for up to 800 ms while his own lips travelled
    0.46 of the parting axis — nearly half the way from closed to teeth showing.
    A margin asks whether a challenger is clearly nearer, and while a mouth
    walks away one rung at a time the answer stays no on every single step.
    """

    # Rungs a fifth of the axis apart, which is the grid's own spacing near the
    # foot of it, and no width on any of them so the distance is the parting.
    LADDER = ((0.2, "speech@a"), (0.4, "speech@b"), (0.6, "speech@c"), (0.8, "speech@d"))
    WIDTHS: ClassVar[dict[str, float]] = dict.fromkeys(
        ("speech@a", "speech@b", "speech@c", "speech@d"), 0.0
    )

    def chosen(self, parted: float, held: str | None) -> set[str]:
        return set(_nearest_speech(self.LADDER, self.WIDTHS, parted, 0.0, parted, held))

    def test_the_rung_in_hand_still_survives_a_boundary(self):
        """The margin's own job, and it is not being taken away: a mouth that has
        barely moved must not change photograph, or every crossing is a flicker."""
        assert self.chosen(0.31, "speech@a") == {"speech@a"}

    def test_but_it_is_let_go_once_it_is_no_longer_this_mouth(self):
        """Past the release the incumbent is not a candidate at all, however
        little the challenger beats it by."""
        assert self.chosen(0.55, "speech@a") == {"speech@c"}

    def test_the_release_is_what_decides_it_rather_than_the_margin(self):
        """At this parting the nearest rung beats the incumbent by less than the
        margin, so the margin alone would have kept it."""
        parted = 0.2 + _SPEECH_RELEASE + 0.02
        distances = sorted(abs(amount - parted) for amount, _ in self.LADDER)
        assert distances[0] > distances[1] - _SPEECH_HOLD, "the margin would have released it"
        assert self.chosen(parted, "speech@a") != {"speech@a"}

    def test_a_mouth_walking_away_one_step_at_a_time_is_still_followed(self):
        """The failure as it was found. Each frame moves the lips too little for
        any challenger to clear the margin; the release is what keeps the
        photograph from sitting still through the whole movement."""
        held: str | None = "speech@a"
        parted = 0.2
        while parted < 0.8:
            parted += 0.02
            got = self.chosen(parted, held)
            held = next(iter(got), None)
            assert held is not None
            amount = next(a for a, key in self.LADDER if key == held)
            assert abs(amount - parted) <= _SPEECH_RELEASE, (
                f"at a parting of {parted:.2f} the mouth on screen is {held}"
            )

    def test_and_a_mouth_at_rest_is_still_the_portrait_own(self):
        assert self.chosen(_SPEECH_LADDER_START / 2, "speech@a") == set()


class TestARoundedMouthDoesNotBecomeTheRestingWidth:
    """Found in the owner's own captures on 2026-08-27. He sits with his mouth
    shut, and the engine reads him as three quarters of the way into a closed
    smile — so the grid's width foot never closes and a widened photograph is
    held over a closed mouth for **9.6 seconds**.

    Rest was learned as a *floor*: down to a new low in a quarter of a second,
    back up over twenty. That is right for a signal that only rises from rest,
    and a mouth is not one. It narrows every time it rounds an "oo", several
    times a sentence, and the floor parked there.
    """

    REST = 0.53  # of the eye span, the owner's own resting mouth
    ROUNDED = 0.40  # and an "oo", which is narrower than rest, not wider

    def mouth(self, reach: float) -> Any:
        """One frame of a face at rest, with the mouth this wide."""
        tracking = expressive_tracking(gap=0.001)
        points = tracking.landmarks.points
        span = float(abs(points[263, 0] - points[33, 0]))
        points[61, :2] = (0.5 - span * reach / 2, 0.6)
        points[291, :2] = (0.5 + span * reach / 2, 0.6)
        return tracking

    def show(self, neutral: _Neutral, reach: float, frames: int) -> float:
        """``frames`` of this mouth, read the way the engine reads them: the
        share advances the estimate, the axis reads it after."""
        widened = 0.0
        for _ in range(frames):
            tracking = self.mouth(reach)
            _smile_share(tracking, neutral)
            widened = _widened(tracking, neutral) / _SMILE_WIDEN_SPAN
        return widened

    def test_a_face_at_rest_reads_as_resting(self):
        neutral = _Neutral()
        assert self.show(neutral, self.REST, 90) < _SPEECH_WIDTH_START

    def test_and_still_does_after_the_mouth_has_rounded_an_oo(self):
        """The defect entire. Half a second of an "oo" and then rest again —
        which is every second sentence — and the mouth that follows must not
        read as a widening."""
        neutral = _Neutral()
        self.show(neutral, self.REST, 90)
        self.show(neutral, self.ROUNDED, 15)

        after = self.show(neutral, self.REST, 3)
        assert after < _SPEECH_WIDTH_START, (
            f"a resting mouth reads {after:.3f} of a widening after an 'oo', "
            f"which holds the grid's width foot open at {_SPEECH_WIDTH_START}"
        )

    def test_but_a_real_smile_is_still_a_widening(self):
        """The estimate is slow in both directions, so it must not learn away a
        smile either. The owner's widest rung is 0.72 of the eye span."""
        neutral = _Neutral()
        self.show(neutral, self.REST, 90)

        assert self.show(neutral, 0.72, 30) > _SPEECH_WIDTH_START


class TestTheAvatarKeepsItsOwnLips:
    """The owner, watching the application on 2026-08-27: *"the size of the
    teeth changes sometimes, and the size of the lips too. Sometimes a thick
    lip, sometimes thin ... I think it is trying to fit its own lips exactly to
    the size of my lips. I do not actually have big lips. Some girl might have
    botox — you would not stretch the photograph onto her lips."*

    He had read the mesh correctly. It maps the avatar's landmarks onto the
    driver's, so the lips on screen were his outline at his thickness, changing
    as his changed. Measured over his four captures the avatar wore his lower
    lip 27 to 43% thicker than its own.
    """

    OUTER_UPPER, INNER_UPPER, INNER_LOWER, OUTER_LOWER = 0, 13, 14, 17

    def mesh(self, *, lip: float, opening: float) -> np.ndarray:
        """A mesh whose lips are ``lip`` thick and ``opening`` apart, in pixels.

        Everything else is left where :func:`synthetic_landmarks` put it, so a
        rigid fit between two of these is an identity and what the damping does
        to the lips is the only thing measured.
        """
        seat = {original: row for row, original in enumerate(MESH_INDICES)}
        built = _pixel_points(synthetic_landmarks(), 400, 400)
        assert built is not None
        points: np.ndarray = built
        middle = 260.0
        for index, offset in (
            (self.OUTER_UPPER, -opening / 2 - lip),
            (self.INNER_UPPER, -opening / 2),
            (self.INNER_LOWER, opening / 2),
            (self.OUTER_LOWER, opening / 2 + lip),
        ):
            points[seat[index], 1] = middle + offset
        return points

    def measured(self, points: np.ndarray) -> tuple[float, float]:
        """That mesh's lip thickness and mouth opening, read back."""
        seat = {original: row for row, original in enumerate(MESH_INDICES)}
        lip = float(points[seat[self.OUTER_LOWER], 1] - points[seat[self.INNER_LOWER], 1])
        opening = float(points[seat[self.INNER_LOWER], 1] - points[seat[self.INNER_UPPER], 1])
        return lip, opening

    def test_a_thick_lipped_driver_does_not_thicken_the_avatar(self):
        """The defect as he described it. The avatar's lip is 10 px, the
        driver's is 30, and what is drawn must be the avatar's."""
        from sha256.avatar.engine_warp import _damp_brows

        avatar = self.mesh(lip=10.0, opening=4.0)
        driver = self.mesh(lip=30.0, opening=4.0)

        lip, _ = self.measured(_damp_brows(avatar, driver))

        assert lip == pytest.approx(10.0 + _LIP_FOLLOW * 20.0, abs=1.0)
        assert lip < 20.0, "the avatar is still wearing more than half of his lip"

    def test_and_a_thin_lipped_one_does_not_thin_it(self):
        """The other direction, which is the one that matters for someone whose
        lips are fuller than the driver's."""
        from sha256.avatar.engine_warp import _damp_brows

        avatar = self.mesh(lip=30.0, opening=4.0)
        driver = self.mesh(lip=10.0, opening=4.0)

        lip, _ = self.measured(_damp_brows(avatar, driver))

        assert lip == pytest.approx(30.0 - _LIP_FOLLOW * 20.0, abs=1.0)

    def test_but_the_photograph_own_opening_survives_the_damping(self):
        """**The thing this must not cost.** The opening arrives in the chosen
        rung, and the rung is what ``source`` is: an open mouth damped toward an
        open mouth stays open. If this ever fails, the mouth has been damped
        shut and the speech grid is drawing nothing."""
        from sha256.avatar.engine_warp import _damp_brows

        rung = self.mesh(lip=10.0, opening=40.0)
        driver = self.mesh(lip=10.0, opening=46.0)

        _, opening = self.measured(_damp_brows(rung, driver))

        assert opening > 35.0, f"the mouth closed to {opening:.0f} px"
        assert opening == pytest.approx(40.0 + _LIP_FOLLOW * 6.0, abs=1.0)
