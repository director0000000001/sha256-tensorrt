"""Day 23 (2026-10-16), his idea: the person's skin colour read from the
camera and said in the avatar's prompts -- the reader, its words, the
sampler, the fill, the comparison."""

from __future__ import annotations

import numpy as np
import pytest

from sha256.avatar.photo_guide import (
    BALD_PROMPT,
    PARTED_LIPS_PROMPT,
    PORTRAIT_PROMPT,
    PROMPTS,
    photo_guide_text,
)
from sha256.avatar.skin_tone import (
    FRAMES_WANTED,
    SKIN_PLACEHOLDER,
    SkinReading,
    SkinSampler,
    compare,
    fill_prompt,
    read_skin,
)

# sRGB skins, as (B, G, R) the way a frame holds them.
LIGHT = (152.0, 194.0, 229.0)
TAN = (110.0, 150.0, 198.0)
DEEP = (40.0, 60.0, 90.0)
PINK = (180.0, 190.0, 235.0)


def face_points(width: int = 640, height: int = 480) -> np.ndarray:
    """478 points of a face 200 px wide and 240 tall, centred, in pixels:
    the cheeks at the face's sides, the forehead's middle above."""
    rng = np.random.default_rng(3)
    pts = np.zeros((478, 3), dtype=np.float32)
    pts[:, 0] = rng.uniform(220.0, 420.0, 478)
    pts[:, 1] = rng.uniform(120.0, 360.0, 478)
    pts[50] = (260.0, 260.0, 0.0)
    pts[280] = (380.0, 260.0, 0.0)
    pts[151] = (320.0, 150.0, 0.0)
    return pts


def frame_of(colour: tuple[float, float, float], width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = np.asarray(colour, dtype=np.uint8)
    return frame


class TestWords:
    @pytest.mark.parametrize(
        ("bgr", "tone"),
        [(LIGHT, "light"), (TAN, "light-medium"), (DEEP, "deep brown")],
    )
    def test_the_lightness_band_from_the_ita_angle(self, bgr, tone):
        assert SkinReading(bgr).tone == tone

    def test_the_undertone_from_the_hue(self):
        assert SkinReading(LIGHT).undertone == "warm, golden"
        assert SkinReading(PINK).undertone == "cool, pinkish"

    def test_the_code_is_srgb_from_the_bgr(self):
        reading = SkinReading(LIGHT)
        assert reading.hex == "#E5C298"
        assert reading.rgb == (229.0, 194.0, 152.0)
        assert reading.describe() == "light skin with a warm, golden undertone, sRGB #E5C298"
        assert reading.as_dict()["hex"] == "#E5C298"

    def test_the_ita_angle_is_light_for_a_pale_skin_and_negative_for_a_deep_one(self):
        assert SkinReading(LIGHT).ita > 41.0
        assert SkinReading(DEEP).ita < -30.0

    def test_distance_is_zero_to_itself_and_grows_with_the_difference(self):
        light = SkinReading(LIGHT)
        assert light.distance(light) == 0.0
        assert light.distance(SkinReading(TAN)) < light.distance(SkinReading(DEEP))


class TestReader:
    def test_a_flat_skin_reads_as_itself(self):
        assert read_skin(frame_of(TAN), face_points()) == TAN

    def test_dark_cells_and_highlights_are_dropped(self):
        """A brow's edge inside a cheek patch (dark) and a highlight on the
        forehead (bright) do not move the reading."""
        frame = frame_of(TAN)
        frame[255:265, 250:270] = (10, 10, 10)  # a dark streak in the left cheek's patch
        frame[145:150, 315:325] = (250, 250, 250)  # a highlight on the forehead
        assert read_skin(frame, face_points()) == TAN

    def test_too_few_points_or_a_patch_off_the_picture_gives_none(self):
        assert read_skin(frame_of(TAN), face_points()[:100]) is None
        pts = face_points()
        pts[:, 0] -= 1000.0  # the whole face left of the picture
        assert read_skin(frame_of(TAN), pts) is None


class TestSampler:
    def test_the_median_over_the_wanted_frames(self):
        sampler = SkinSampler(wanted=5)
        pts = face_points()
        colours = [TAN, LIGHT, TAN, TAN, DEEP, LIGHT]
        added = [sampler.add(frame_of(c), pts) for c in colours]
        assert added == [True, True, True, True, True, False]  # the sixth: done
        assert sampler.done and sampler.count == 5
        reading = sampler.reading()
        assert reading is not None and reading.bgr == TAN and reading.frames == 5

    def test_a_frame_without_skin_is_not_counted(self):
        sampler = SkinSampler()
        assert sampler.wanted == FRAMES_WANTED
        assert not sampler.add(frame_of(TAN), face_points()[:10])
        assert sampler.count == 0 and sampler.reading() is None


class TestPrompts:
    def test_every_prompt_carries_the_placeholder(self):
        for prompt in (PORTRAIT_PROMPT, BALD_PROMPT, PARTED_LIPS_PROMPT):
            assert SKIN_PLACEHOLDER in prompt
        assert [p for _, p in PROMPTS] == [PORTRAIT_PROMPT, BALD_PROMPT, PARTED_LIPS_PROMPT]

    def test_the_fill_replaces_every_placeholder_with_the_words(self):
        filled = fill_prompt(BALD_PROMPT, SkinReading(LIGHT))
        assert SKIN_PLACEHOLDER not in filled
        assert "light skin with a warm, golden undertone, sRGB #E5C298" in filled
        assert fill_prompt(BALD_PROMPT, None) == BALD_PROMPT

    def test_the_bald_prompt_shaves_and_keeps_the_face(self):
        assert "completely shaved" in BALD_PROMPT and "no hairline" in BALD_PROMPT
        assert "Eyebrows, eyelashes and any facial hair stay" in BALD_PROMPT
        assert "the shaved head" in PARTED_LIPS_PROMPT

    def test_the_guide_text_names_the_shaved_one_as_the_avatar_and_fills_on_a_reading(self):
        text = photo_guide_text()
        assert "Add the SHAVED one (prompt 2)" in text
        assert text.count(SKIN_PLACEHOLDER) >= 4  # the requirements, the three prompts, the note
        filled = photo_guide_text(SkinReading(TAN))
        assert SKIN_PLACEHOLDER in filled  # the requirements' and the note's mention stay
        assert filled.count("light-medium skin with") == 3


class TestCompare:
    def test_close_noticeable_and_far(self):
        camera = SkinReading(TAN)
        assert "close" in compare(camera, SkinReading(TAN))
        nearby = SkinReading((TAN[0] + 30.0, TAN[1] + 30.0, TAN[2] + 30.0))
        assert 8.0 < camera.distance(nearby) <= 16.0
        assert "noticeably different" in compare(camera, nearby)
        far = compare(camera, SkinReading(DEEP))
        assert "far" in far and "Generating the photograph again" in far
        assert camera.hex in far and SkinReading(DEEP).hex in far
