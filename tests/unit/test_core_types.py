"""Tests for the pipeline data contracts.

These types are the interop surface between every replaceable component, so
their invariants are tested directly rather than incidentally through the
components that use them.
"""

from __future__ import annotations

import numpy as np
import pytest

from sha256.core.enums import PixelFormat, TrackingQuality
from sha256.core.types import (
    BlendshapeSet,
    FaceTrackingResult,
    FrameMetadata,
    LandmarkSet,
    RenderedFrame,
    VideoFrame,
)


class TestFrameMetadata:
    def test_age_uses_supplied_clock(self):
        metadata = FrameMetadata(frame_id=1, capture_timestamp=10.0)
        assert metadata.age(10.5) == pytest.approx(0.5)

    def test_sequence_gap_defaults_to_zero(self):
        assert FrameMetadata(frame_id=1, capture_timestamp=0.0).sequence_gap == 0


class TestVideoFrame:
    def test_dimensions_come_from_the_array(self, sample_frame: VideoFrame):
        assert sample_frame.height == 16
        assert sample_frame.width == 24
        assert sample_frame.resolution == (24, 16)

    def test_copy_detaches_the_buffer(self, sample_frame: VideoFrame):
        """The borrowed-buffer convention is only safe if copy() truly detaches."""
        copied = sample_frame.copy()
        copied.data[0, 0, 0] = 255

        assert sample_frame.data[0, 0, 0] == 128
        assert copied.metadata == sample_frame.metadata

    def test_frames_are_immutable(self, sample_frame: VideoFrame):
        with pytest.raises(AttributeError):
            sample_frame.pixel_format = PixelFormat.BGR24  # type: ignore[misc]


class TestLandmarkSet:
    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
            LandmarkSet(points=np.zeros((10, 2), dtype=np.float32), scheme="bad")

    def test_count_reflects_point_total(self):
        landmarks = LandmarkSet(points=np.zeros((68, 3), dtype=np.float32), scheme="test_68")
        assert landmarks.count == 68


class TestBlendshapeSet:
    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            BlendshapeSet(names=("a", "b"), weights=np.zeros(3, dtype=np.float32))

    def test_lookup_by_name(self):
        shapes = BlendshapeSet(
            names=("jawOpen", "eyeBlinkLeft"), weights=np.array([0.75, 0.1], dtype=np.float32)
        )
        assert shapes.get("jawOpen") == pytest.approx(0.75)

    def test_missing_name_returns_default(self):
        shapes = BlendshapeSet.empty()
        assert shapes.get("nope", default=0.5) == pytest.approx(0.5)

    def test_as_mapping_round_trips(self):
        shapes = BlendshapeSet(names=("a", "b"), weights=np.array([0.25, 0.5], dtype=np.float32))
        assert shapes.as_mapping() == pytest.approx({"a": 0.25, "b": 0.5})


class TestFaceTrackingResult:
    def test_no_face_helper_is_a_valid_result(self):
        metadata = FrameMetadata(frame_id=3, capture_timestamp=1.0)
        result = FaceTrackingResult.no_face(metadata, tracker_id="test")

        assert result.face_detected is False
        assert result.quality is TrackingQuality.NONE
        assert result.landmarks is None

    @pytest.mark.parametrize(
        ("confidence", "expected"),
        [
            (0.9, TrackingQuality.GOOD),
            (0.75, TrackingQuality.GOOD),
            (0.5, TrackingQuality.FAIR),
            (0.1, TrackingQuality.POOR),
        ],
    )
    def test_quality_buckets(self, confidence: float, expected: TrackingQuality):
        result = FaceTrackingResult(
            metadata=FrameMetadata(frame_id=1, capture_timestamp=0.0),
            face_detected=True,
            confidence=confidence,
        )
        assert result.quality is expected

    def test_extras_do_not_leak_between_instances(self):
        """A shared mutable default would silently couple every result."""
        metadata = FrameMetadata(frame_id=1, capture_timestamp=0.0)
        first = FaceTrackingResult(metadata=metadata, face_detected=True)
        second = FaceTrackingResult(metadata=metadata, face_detected=True)

        first.extras["gaze_x"] = 1.0
        assert second.extras == {}


class TestRenderedFrame:
    def test_pipeline_latency_spans_capture_to_render(self):
        frame = RenderedFrame(
            data=np.zeros((4, 4, 3), dtype=np.uint8),
            pixel_format=PixelFormat.RGB24,
            metadata=FrameMetadata(frame_id=1, capture_timestamp=10.0),
            render_timestamp=10.025,
        )
        assert frame.pipeline_latency == pytest.approx(0.025)


class TestPixelFormat:
    def test_channel_counts(self):
        assert PixelFormat.RGB24.channels == 3
        assert PixelFormat.RGBA32.channels == 4
        assert PixelFormat.GRAY8.channels == 1

    def test_planar_formats_are_flagged(self):
        assert PixelFormat.NV12.is_planar
        assert not PixelFormat.RGB24.is_planar
