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
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_brow_ladder as brow_ladder  # noqa: E402
import build_eye_ladder as eye_ladder  # noqa: E402
import build_speech_ladder as speech_ladder  # noqa: E402
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


def speech_step(parted: float, widened: float, sharpness: float = 100.0, face: float = 1000.0):
    """One measured frame, with only the fields the cell rule reads filled in."""
    return speech_ladder.Step(
        path=Path(f"{parted:.3f}_{widened:.3f}.jpg"),
        parted=parted,
        widened=widened,
        width=0.5,
        gap=0.05,
        sharpness=sharpness,
        candidate=SimpleNamespace(face_width=face),
        drift=0.0,
    )


class TestAFrameAloneInItsCellIsStillAShapeTheMouthMakes:
    """`CELL_MINIMUM`, and the two it used to be could not rebuild the grid that
    is installed.

    The rule was *"one frame is a moment of the clip rather than a shape the
    mouth makes"*, written when a rung was a 5.7 MB 4K PNG and the avatar took
    24 s to load. Rungs are written at the portrait's scale now, and the step
    came down with the grid: run over `video (14)` at the spacing the installed
    library was built with, the builder produced **11 rungs** where the library
    holds **143** from that clip. At that spacing nearly every frame is alone.

    It matters because of what the owner asked for next — *"when I say 'make'
    the mouth makes a fast jump and goes straight to the A"* — whose answer is a
    clip whose useful frames are exactly the ones a fast opening passes through,
    one to a cell and every one of them.
    """

    def test_a_cell_with_one_frame_in_it_becomes_a_rung(self):
        chosen = speech_ladder.best_per_cell(
            [speech_step(0.10, 0.0), speech_step(0.50, 0.0), speech_step(1.20, 0.0)], 0.035, 0.04
        )
        assert [round(step.parted, 2) for step in chosen] == [0.10, 0.50, 1.20]

    def test_two_frames_in_one_cell_still_give_one_rung(self):
        """A cell is one rung however many frames fall in it — that is what
        makes this a grid rather than a dump of the clip."""
        chosen = speech_ladder.best_per_cell(
            [speech_step(0.100, 0.0, sharpness=10.0), speech_step(0.101, 0.0, sharpness=90.0)],
            0.035,
            0.04,
        )
        assert len(chosen) == 1
        assert chosen[0].sharpness == 90.0

    def test_the_bigger_face_wins_a_cell_before_sharpness_is_read(self):
        chosen = speech_ladder.best_per_cell(
            [
                speech_step(0.100, 0.0, sharpness=900.0, face=400.0),
                speech_step(0.101, 0.0, sharpness=10.0, face=1200.0),
            ],
            0.035,
            0.04,
        )
        assert [step.candidate.face_width for step in chosen] == [1200.0]

    def test_a_frame_the_engine_would_refuse_never_wins_a_cell(self):
        refused = speech_step(0.10, 0.0)
        chosen = speech_ladder.best_per_cell(
            [speech_ladder.Step(**{**refused.__dict__, "refused": "its brows are in its eyes"})],
            0.035,
            0.04,
        )
        assert chosen == []


class TestTheCellIsCoarserWhereTheMouthIsWideOpen:
    """`WIDE_ABOVE` and `WIDE_STEP`, measured on the owner's own captures.

    155 of the 221 rungs three speech folders build sit above a parting of 0.70
    and 34 sit in the band he complained about. Coarsening the cell above the
    line takes the grid from 253 rungs to 174 — fewer than were installed the
    day before — for a twentieth of a cell of error at the wide end and *no*
    change at all in the middle.
    """

    def test_two_wide_frames_a_fine_cell_apart_become_one_rung(self):
        chosen = speech_ladder.best_per_cell(
            [speech_step(1.19, 0.0, sharpness=10.0), speech_step(1.22, 0.0, sharpness=90.0)],
            0.035,
            0.04,
        )

        assert len(chosen) == 1
        assert chosen[0].sharpness == 90.0

    def test_the_same_two_near_the_foot_stay_two_rungs(self):
        """The coarsening is the wide end's alone: 0.10 and 0.14 of parting are
        a mouth barely open and a mouth speaking, and the day-19 clip was bought
        to tell them apart."""
        chosen = speech_ladder.best_per_cell(
            [speech_step(0.10, 0.0), speech_step(0.14, 0.0)], 0.035, 0.04
        )

        assert [round(step.parted, 2) for step in chosen] == [0.10, 0.14]

    def test_a_coarse_cell_never_collides_with_a_fine_one(self):
        """0.70 over a step of 0.035 and 1.40 over one of 0.07 are both index
        twenty. Tagged by which grid they belong to, they are two cells."""
        chosen = speech_ladder.best_per_cell(
            [speech_step(0.70, 0.0), speech_step(1.40, 0.0)], 0.035, 0.04
        )

        assert [round(step.parted, 2) for step in chosen] == [0.70, 1.40]

    def test_the_uniform_grid_is_still_reachable(self):
        chosen = speech_ladder.best_per_cell(
            [speech_step(1.19, 0.0), speech_step(1.22, 0.0)], 0.035, 0.04, wide_step=0.035
        )

        assert [round(step.parted, 2) for step in chosen] == [1.19, 1.22]
