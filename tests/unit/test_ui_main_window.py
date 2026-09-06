"""Tests for the main window's status line.

The line has one job while the pipeline runs: say when the face is not being
found, because on a loss the avatar is held in place and a frozen face over a
moving background reads as the avatar having vanished. A version of
``_refresh_status`` set that text and then cleared it on the next line, so the
message never reached the screen; this is the test that would have caught it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sha256.core.enums import PixelFormat
from sha256.core.types import FaceTrackingResult, FrameMetadata, LandmarkSet, RenderedFrame
from sha256.pipeline.runner import PipelineSnapshot

pytestmark = pytest.mark.gui

pytest.importorskip("PySide6.QtWidgets", reason="PySide6 is required for UI tests")


class StillRunner:
    """A runner standing still on one snapshot, so the status can be read off it."""

    def __init__(self, tracking: FaceTrackingResult) -> None:
        metadata = tracking.metadata
        frame = RenderedFrame(
            data=np.zeros((4, 4, 3), dtype=np.uint8),
            pixel_format=PixelFormat.BGR24,
            metadata=metadata,
        )
        self._snapshot = PipelineSnapshot(frame, None, tracking, metadata.frame_id)

    is_running = True
    last_error = None
    fps = 30.0

    def latest(self) -> PipelineSnapshot:
        return self._snapshot

    def stop(self) -> None:
        return None


def tracking(*, face: bool) -> FaceTrackingResult:
    metadata = FrameMetadata(frame_id=7, capture_timestamp=7.0)
    if not face:
        return FaceTrackingResult.no_face(metadata)
    return FaceTrackingResult(
        metadata=metadata,
        face_detected=True,
        landmarks=LandmarkSet(points=np.zeros((478, 3), dtype=np.float32), scheme="mediapipe_478"),
    )


@pytest.fixture
def window(qtbot, app_context: Any):
    from sha256.ui.main_window import MainWindow

    widget = MainWindow(app_context)
    qtbot.addWidget(widget)
    return widget


class TestStatusLine:
    def test_a_lost_face_is_said_on_the_status_line(self, window):
        window._runner = StillRunner(tracking(face=False))

        window._refresh_status()

        assert "Face not found" in window._status_line.text()

    def test_a_tracked_face_leaves_the_line_empty(self, window):
        window._runner = StillRunner(tracking(face=True))

        window._refresh_status()

        assert window._status_line.text() == ""

    def test_a_learning_rest_is_said_on_the_status_line(self, window):
        """The null engine has no rest to learn -- rest_status is patched
        directly so this exercises the wiring, not a specific engine."""
        window._runner = StillRunner(tracking(face=True))
        window._context.avatars.select("avatar_001")
        window._context.avatars.engine.rest_status = lambda avatar_id: "centring (3/15)"

        window._refresh_status()

        assert window._status_line.text() == "centring (3/15)"


class TestRecentre:
    """The "Re-centre" button: AvatarEngine.reset_anchor, added 2026-09-05
    after an anchor learned mid-turn reproduced the owner's own "the avatar
    is to the side" report exactly."""

    def test_recentre_calls_reset_anchor_on_the_active_avatar(self, window):
        window._context.avatars.select("avatar_001")
        calls: list[str] = []
        window._context.avatars.engine.reset_anchor = calls.append

        window._on_recentre()

        assert calls == ["avatar_001"]

    def test_recentre_with_no_active_avatar_does_nothing(self, window):
        window._on_recentre()  # must not raise; nothing selected yet
