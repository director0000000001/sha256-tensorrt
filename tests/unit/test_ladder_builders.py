"""Tests for the two things the ladder builders got wrong, and both were silent.

Neither defect could be seen in the output. The builders printed a clean,
ordered, plausible table of rungs both times, and the photographs behind the
numbers were something else entirely — six rungs filed from a tenth to half a
closing that were all a shut eye, and a brow rung filed at +0.88 that is a
resting brow.

1.  **The tracker was told every still happened at the same moment.** It runs in
    VIDEO mode and carries state between frames, and the adapter turns repeated
    timestamps into one millisecond apart — so it expected a face that had
    barely moved and reported its own lag instead of the eyelids. Measured: a
    smooth ramp from 0.014 to 0.038 across sixteen frames whose eyes were all
    wide open at 0.050.
2.  **The ladder's zero came from the middle of the clip.** Only correct for a
    clip that is mostly at rest, which an eye clip never is: the take that built
    the old ladder holds the eyes shut for four of its eight seconds, so its
    median was a shut eye and everything was filed against one.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_brow_ladder as brow_ladder  # noqa: E402
import build_eye_ladder as eye_ladder  # noqa: E402
import extract_avatar_frames as extractor  # noqa: E402


class TestWhenEachFrameHappened:
    """`dump_clip_frames.py` writes the millisecond into every filename. Reading
    it back is the whole fix, and nothing else in the pipeline had to change."""

    def test_the_millisecond_is_read_from_the_name(self):
        assert extractor.timestamp_of(Path("0150_06240ms.png"), 0) == pytest.approx(6.24)

    def test_a_folder_assembled_by_hand_still_gets_spaced_timestamps(self):
        """The fallback is the index, not a constant. A constant is the defect."""
        first = extractor.timestamp_of(Path("a.png"), 0)
        second = extractor.timestamp_of(Path("b.png"), 1)

        assert second > first

    def test_consecutive_frames_never_share_a_timestamp(self):
        """The property that was violated. Every still arrived at the same
        instant, so the tracker was asked to believe a face had moved from wide
        open to shut in one millisecond, and sensibly refused to."""
        names = [Path(f"{index:04d}_{index * 40:05d}ms.png") for index in range(10)]

        stamps = [extractor.timestamp_of(path, index) for index, path in enumerate(names)]

        assert stamps == sorted(stamps)
        assert len(set(stamps)) == len(stamps)

    def test_the_spacing_is_the_clip_s_own(self):
        """Forty milliseconds apart at 24 fps, not one. The gap is what the
        tracker reasons about."""
        gap = extractor.timestamp_of(Path("0001_00040ms.png"), 1) - extractor.timestamp_of(
            Path("0000_00000ms.png"), 0
        )

        assert gap == pytest.approx(0.04)


def eye_step(opening: float) -> Any:
    return eye_ladder.Step(
        path=Path(f"{opening}.png"), opening=opening, sharpness=100.0, candidate=None
    )


def brow_step(height: float) -> Any:
    return brow_ladder.Step(
        path=Path(f"{height}.png"), height=height, sharpness=100.0, candidate=None
    )


class TestWhereTheLadderPutsItsZero:
    """Rest is the avatar's own resting face, which is the portrait."""

    # A clip like the one that built the old ladder: a moment at rest, and then
    # the eyes shut for most of it.
    MOSTLY_SHUT: ClassVar[list[float]] = [
        0.0538,
        0.0530,
        0.0200,
        0.0195,
        0.0190,
        0.0188,
        0.0185,
        0.0182,
    ]

    def test_the_portrait_decides_what_a_resting_eye_is(self):
        """Given the portrait's 0.0524, a shut frame is filed near two thirds of
        a closing — which is what it is."""
        steps = [eye_step(opening) for opening in self.MOSTLY_SHUT]

        placed = {step.opening: amount for step, amount in eye_ladder.amounts(steps, 0.0524)}

        assert placed[0.0538] == pytest.approx(-0.027, abs=0.01)
        assert placed[0.0182] == pytest.approx(0.653, abs=0.01)

    def test_the_median_of_a_shut_eyed_clip_calls_a_shut_eye_rest(self):
        """The old rule, kept as the fallback and tested so the cost of using it
        is written down: the same shut frame is filed at less than half a
        closing, and the genuinely open frames read as eyes held wide."""
        steps = [eye_step(opening) for opening in self.MOSTLY_SHUT]

        placed = {step.opening: amount for step, amount in eye_ladder.amounts(steps)}

        assert placed[0.0182] < 0.06
        assert placed[0.0538] < -0.8

    def test_a_brow_ladder_takes_its_zero_from_the_portrait_too(self):
        steps = [brow_step(height) for height in (0.0751, 0.0700, 0.0912)]

        placed = {step.height: amount for step, amount in brow_ladder.amounts(steps, 0.0751)}

        assert placed[0.0751] == pytest.approx(0.0, abs=1e-6)
        assert placed[0.0912] > 0.0
        assert placed[0.0700] < 0.0

    def test_an_eye_that_never_moved_is_not_divided_by_zero(self):
        placed = eye_ladder.amounts([eye_step(0.0)], rest=0.0)

        assert placed[0][1] == 0.0
