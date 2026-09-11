"""The last stage before the sink (day 15): the output quality and the black
screen -- see ``sha256.pipeline.conditioning``.

The clock is a number the test moves by hand, so the hysteresis is checked in
seconds that never have to pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from sha256.core.enums import PixelFormat
from sha256.core.types import FrameMetadata, RenderedFrame
from sha256.pipeline.conditioning import (
    DEFAULT_FAILURE_TEXT,
    QUALITY_NAMES,
    OutputConditioner,
    apply_quality,
    failure_frame,
)


def checkerboard(height: int = 720, width: int = 1280, cell: int = 2) -> np.ndarray:
    """Fine detail that any down-and-up scaling must lose."""
    rows = (np.arange(height) // cell) % 2
    cols = (np.arange(width) // cell) % 2
    board = (rows[:, None] ^ cols[None, :]).astype(np.uint8) * 255
    return np.repeat(board[:, :, None], 3, axis=2)


def frame(data: np.ndarray, **kwargs) -> RenderedFrame:
    return RenderedFrame(
        data=data,
        pixel_format=PixelFormat.BGR24,
        metadata=FrameMetadata(frame_id=1, capture_timestamp=1.0),
        avatar_id="me",
        render_timestamp=1.02,
        **kwargs,
    )


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def portrait_like(height: int = 720, width: int = 1280) -> np.ndarray:
    """A soft gradient with a few hard edges and thin lines -- enough like a
    lit face and its hair for a sharpness measure to mean something."""
    y, x = np.mgrid[0:height, 0:width]
    base = (120 + 60 * np.sin(x / 90.0) * np.cos(y / 70.0)).astype(np.uint8)
    picture = np.repeat(base[:, :, None], 3, axis=2)
    picture[height // 3 : height // 2, width // 4 : width // 2] = (40, 60, 200)
    picture[:, ::16] = 250
    picture[::24, :] = 30
    return picture


def sharpness(picture: np.ndarray) -> float:
    import cv2

    return float(cv2.Laplacian(cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


class TestQuality:
    def test_native_returns_the_same_array(self):
        data = checkerboard()
        assert apply_quality(data, "native") is data

    @pytest.mark.parametrize("quality", ["1080p", "720p", "480p", "360p"])
    def test_every_other_level_keeps_the_size_and_is_a_new_picture(self, quality):
        data = portrait_like()
        out = apply_quality(data, quality)
        assert out.shape == data.shape and out.dtype == np.uint8
        assert out is not data
        assert np.abs(out.astype(int) - data.astype(int)).mean() > 0.2

    @pytest.mark.parametrize("quality", ["480p", "360p"])
    def test_a_lower_quality_loses_the_fine_detail(self, quality):
        data = checkerboard()
        out = apply_quality(data, quality)
        # A two-pixel checkerboard averages to grey once it has been through
        # a smaller picture; the original is pure black and white.
        assert 60 < out.mean() < 200
        assert np.abs(out.astype(int) - data.astype(int)).mean() > 40

    def test_each_level_down_is_less_sharp_than_the_one_above(self):
        """His first look at a plain down-and-up scale: "nothing changed".
        The ladder has to be visible, step by step, on a face-like picture."""
        data = portrait_like()
        names = ("native", "1080p", "720p", "480p", "360p")
        levels = [sharpness(apply_quality(data, q)) for q in names]
        assert levels == sorted(levels, reverse=True), levels
        assert levels[-1] < 0.5 * levels[0]

    def test_a_frame_smaller_than_the_level_is_not_scaled_up(self):
        data = portrait_like(360, 640)
        out = apply_quality(data, "720p")
        assert out.shape == data.shape

    def test_an_unknown_quality_is_refused(self):
        with pytest.raises(ValueError, match="unknown output quality"):
            apply_quality(checkerboard(64, 64), "4k")
        with pytest.raises(ValueError, match="unknown output quality"):
            OutputConditioner(quality="best")

    def test_the_names_are_the_ones_the_settings_and_the_window_use(self):
        assert QUALITY_NAMES == ("native", "1080p", "720p", "480p", "360p")


class TestFailureFrame:
    def test_black_with_the_text_in_the_middle_and_cached(self):
        a = failure_frame(640, 360, "Camera battery died")
        assert a.shape == (360, 640, 3) and a.dtype == np.uint8
        # Mostly black, some grey text, nothing white or coloured.
        assert (a == 0).mean() > 0.9
        assert a.max() > 100
        assert a.max() < 220
        lit = np.argwhere(a.max(axis=2) > 0)
        assert 100 < lit[:, 0].mean() < 260  # around the middle row
        assert 200 < lit[:, 1].mean() < 440  # around the middle column
        assert failure_frame(640, 360, "Camera battery died") is a
        assert failure_frame(640, 360, "Other words") is not a

    def test_no_text_is_plain_black(self):
        assert not failure_frame(32, 16, "").any()


class TestBlackScreen:
    def _conditioner(self, clock: Clock, **kwargs) -> OutputConditioner:
        return OutputConditioner(clock=clock, **kwargs)

    def test_the_avatar_goes_out_as_it_is_at_native_quality(self):
        clock = Clock()
        c = self._conditioner(clock)
        data = checkerboard(72, 128)
        out = c.condition(frame(data), face_detected=True, passthrough=False)
        assert out.data is data
        assert out.avatar_id == "me" and not out.is_placeholder
        assert c.blackout is None

    def test_the_quality_is_applied_on_the_way_out(self):
        clock = Clock()
        c = self._conditioner(clock, quality="360p")
        data = checkerboard()
        out = c.condition(frame(data), face_detected=True, passthrough=False)
        assert out.data is not data and out.data.shape == data.shape
        c.set_quality("native")
        assert c.condition(frame(data), face_detected=True, passthrough=False).data is data

    def test_the_cameras_own_picture_is_blacked_at_once(self):
        clock = Clock()
        c = self._conditioner(clock)
        data = checkerboard(72, 128)
        out = c.condition(frame(data, is_passthrough=True), face_detected=False, passthrough=True)
        assert not out.data.any() or out.data.max() < 220  # black, maybe the text
        assert (out.data == 0).mean() > 0.9
        assert out.is_placeholder and out.avatar_id is None
        assert out.data.shape == data.shape
        assert c.blackout is not None and "camera" in c.blackout

    def test_a_short_loss_passes_and_a_long_one_blacks(self):
        clock = Clock()
        c = self._conditioner(clock, lost_face_seconds=1.5)
        data = checkerboard(72, 128)
        for dt in (0.0, 0.5, 1.0, 1.4):
            clock.t = 100.0 + dt
            out = c.condition(frame(data), face_detected=False, passthrough=False)
            assert out.data is data, dt
            assert c.blackout is None
        clock.t = 101.5
        out = c.condition(frame(data), face_detected=False, passthrough=False)
        assert (out.data == 0).mean() > 0.9
        assert c.blackout == "face lost"

    def test_an_automatic_black_screen_holds_then_lifts_after_a_steady_face(self):
        clock = Clock()
        c = self._conditioner(clock, min_blackout_seconds=3.0, recovery_seconds=1.0)
        data = checkerboard(72, 128)
        c.condition(frame(data), face_detected=False, passthrough=True)  # black at t=100
        # The face is back at t=100.2, tracked steadily from then on: the
        # screen still holds until 103 (the minimum) -- and past the minimum
        # it lifts once the face has been steady for a second.
        for t in (100.2, 101.0, 102.0, 102.9):
            clock.t = t
            out = c.condition(frame(data), face_detected=True, passthrough=False)
            assert out.is_placeholder, t
        clock.t = 103.0
        out = c.condition(frame(data), face_detected=True, passthrough=False)
        assert not out.is_placeholder and out.data is data
        assert c.blackout is None

    def test_the_face_must_be_steady_for_the_recovery_time(self):
        clock = Clock()
        c = self._conditioner(clock, min_blackout_seconds=3.0, recovery_seconds=1.0)
        data = checkerboard(72, 128)
        c.condition(frame(data), face_detected=False, passthrough=True)  # black at t=100
        clock.t = 105.0  # well past the minimum, but the face has only just come back
        assert c.condition(frame(data), face_detected=True, passthrough=False).is_placeholder
        clock.t = 105.5
        assert c.condition(frame(data), face_detected=True, passthrough=False).is_placeholder
        clock.t = 105.8
        # ...and a blink of loss starts the count again: back at 106.5, the
        # face must hold until 107.5.
        assert c.condition(frame(data), face_detected=False, passthrough=False).is_placeholder
        clock.t = 106.5
        assert c.condition(frame(data), face_detected=True, passthrough=False).is_placeholder
        clock.t = 106.9
        assert c.condition(frame(data), face_detected=True, passthrough=False).is_placeholder
        clock.t = 107.5
        assert not c.condition(frame(data), face_detected=True, passthrough=False).is_placeholder

    def test_the_manual_switch_blacks_at_once_and_returns_at_once(self):
        clock = Clock()
        c = self._conditioner(clock)
        data = checkerboard(72, 128)
        c.set_manual(True)
        assert c.manual and c.blackout == "manual"
        out = c.condition(frame(data), face_detected=True, passthrough=False)
        assert out.is_placeholder and (out.data == 0).mean() > 0.9
        c.set_manual(False)
        out = c.condition(frame(data), face_detected=True, passthrough=False)
        assert out.data is data and c.blackout is None

    def test_the_manual_switch_outranks_an_automatic_one_and_survives_a_reset(self):
        clock = Clock()
        c = self._conditioner(clock)
        data = checkerboard(72, 128)
        c.set_manual(True)
        c.condition(frame(data), face_detected=False, passthrough=True)
        assert c.blackout == "manual"
        c.reset()
        assert c.manual and c.blackout == "manual"

    def test_reset_forgets_an_automatic_black_screen(self):
        clock = Clock()
        c = self._conditioner(clock)
        data = checkerboard(72, 128)
        c.condition(frame(data), face_detected=False, passthrough=True)
        assert c.blackout is not None
        c.reset()
        assert c.blackout is None
        assert c.condition(frame(data), face_detected=True, passthrough=False).data is data

    def test_auto_off_leaves_even_the_cameras_picture_alone(self):
        """A developer's setting: the passthrough goes out as it is."""
        clock = Clock()
        c = self._conditioner(clock, auto_blackout=False)
        data = checkerboard(72, 128)
        out = c.condition(frame(data, is_passthrough=True), face_detected=False, passthrough=True)
        assert out.data is data and c.blackout is None
        c.set_manual(True)
        assert c.condition(frame(data), face_detected=True, passthrough=False).is_placeholder

    def test_the_text_and_the_frame_it_makes(self):
        c = OutputConditioner(failure_text="Camera disconnected")
        assert c.failure_text == "Camera disconnected"
        assert c.failure_frame_for(64, 32).shape == (32, 64, 3)
        assert OutputConditioner().failure_text == DEFAULT_FAILURE_TEXT
