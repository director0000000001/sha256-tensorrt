"""Tests for the clip generator.

The tool spends money, so the parts worth testing are the ones that decide
whether to spend it: what counts as a usable take, what the ledger says has
been spent, and the two guards that stop a run — the budget, and a probe that
failed. Nothing here touches the network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_turn_ladder as ladder  # noqa: E402
import generate_avatar_clips as gen  # noqa: E402

from sha256.avatar.engine_warp import _VARIANT_ROLES  # noqa: E402

CATALOGUE = REPO_ROOT / "configs" / "avatar_clips.toml"


def ladder_report(
    rungs: list[tuple[float, bool]], usable: int = 120, frames: int = 200
) -> dict[str, Any]:
    """A report shaped like the extractor's, from (yaw, blurred) pairs."""
    rows = [
        {
            "path": f"turn_{i}.png",
            "yaw": yaw,
            "frame": i,
            "face_px": 500,
            "detail": 0.1 if blurred else 0.9,
            "blurred": blurred,
        }
        for i, (yaw, blurred) in enumerate(rungs)
    ]
    yaws = [yaw for yaw, _ in rungs]
    return {
        "video": "clip.mp4",
        "frames": frames,
        "usable": usable,
        "rejected": {},
        "roles": {"covered": [], "missing": list(_VARIANT_ROLES), "known": list(_VARIANT_ROLES)},
        "ladder": {
            "step_degrees": 3.0,
            "min_detail_fraction": 0.25,
            "rungs": rows,
            "blurred": sum(1 for _, blurred in rungs if blurred),
            "yaw_min": min(yaws, default=0.0),
            "yaw_max": max(yaws, default=0.0),
        },
    }


def roles_report(covered: list[str], usable: int = 120, frames: int = 200) -> dict[str, Any]:
    return {
        "video": "clip.mp4",
        "frames": frames,
        "usable": usable,
        "rejected": {},
        "roles": {
            "covered": list(covered),
            "missing": [r for r in _VARIANT_ROLES if r not in covered],
            "known": list(_VARIANT_ROLES),
        },
        "ladder": None,
    }


@pytest.fixture(scope="module")
def catalogue() -> Any:
    return gen.load_catalogue(CATALOGUE)


class TestCatalogue:
    def test_it_loads(self, catalogue):
        api, clips = catalogue
        assert clips
        assert api.duration_seconds > 0

    def test_ids_and_order_are_unique(self, catalogue):
        _, clips = catalogue
        assert len({clip.id for clip in clips}) == len(clips)
        assert len({clip.order for clip in clips}) == len(clips)

    def test_every_prompt_carries_the_shared_block(self, catalogue):
        _, clips = catalogue
        for clip in clips:
            assert "Static locked-off camera" in clip.prompt
            assert clip.prompt.count("Static locked-off camera") == 1

    def test_a_turn_clip_states_which_way_it_turns(self, catalogue):
        """The failure that cost two clips: nobody could tell which way it went."""
        _, clips = catalogue
        for clip in clips:
            if clip.is_ladder:
                assert clip.expect_yaw_sign in {"positive", "negative"}

    def test_a_turn_prompt_asks_for_stops_and_for_slowness(self, catalogue):
        """A sweep is what produced a ladder of smears; a snap is what looked rushed."""
        _, clips = catalogue
        for clip in clips:
            if clip.is_ladder:
                assert "complete stop" in clip.prompt
                assert "slow" in clip.prompt

    def test_wanted_roles_are_roles_the_engine_has(self, catalogue):
        _, clips = catalogue
        for clip in clips:
            for role in clip.want_roles:
                assert role in _VARIANT_ROLES, f"{clip.id} wants unknown role {role}"

    def test_the_tilt_clip_is_parked(self, catalogue):
        """The engine has no pitch ladder, so its frames would sit unused."""
        _, clips = catalogue
        assert next(clip for clip in clips if clip.id == "tilts").parked

    def test_the_negative_prompt_is_one_line(self):
        negative = gen.negative_prompt(CATALOGUE)
        assert "\n" not in negative
        assert "motion blur" in negative


class TestJudgingALadder:
    def clip(self, **kwargs: Any) -> gen.Clip:
        defaults: dict[str, Any] = {
            "id": "turn",
            "order": 1,
            "title": "t",
            "why": "",
            "check": "ladder",
            "prompt": "p",
            "min_rungs": 5,
            "min_yaw_span": 25.0,
            "max_blurred_fraction": 0.34,
            "expect_yaw_sign": "positive",
        }
        return gen.Clip(**{**defaults, **kwargs})

    def test_a_stepped_clip_passes(self):
        rungs = [(float(i * 8), False) for i in range(1, 8)]
        assert gen.judge(self.clip(), ladder_report(rungs)).ok

    def test_a_continuous_sweep_fails_on_blur(self):
        """The first turn clip: every frame smeared, and nothing caught it."""
        rungs = [(float(i * 8), True) for i in range(1, 8)]
        verdict = gen.judge(self.clip(), ladder_report(rungs))
        assert not verdict.ok
        assert any("motion blurred" in reason for reason in verdict.reasons)

    def test_turning_the_wrong_way_fails_and_says_so(self):
        """The second turn clip: it obeyed the other reading of 'left'."""
        rungs = [(float(-i * 8), False) for i in range(1, 8)]
        verdict = gen.judge(self.clip(), ladder_report(rungs))
        assert not verdict.ok
        assert any("wrong way" in reason for reason in verdict.reasons)

    def test_a_turn_that_stops_short_fails_on_reach(self):
        rungs = [(float(i * 2), False) for i in range(1, 8)]
        verdict = gen.judge(self.clip(), ladder_report(rungs))
        assert not verdict.ok
        assert any("reach" in reason for reason in verdict.reasons)

    def test_too_few_sharp_rungs_fails_even_when_the_reach_is_right(self):
        rungs = [(10.0, False), (50.0, False)] + [(float(i * 8), True) for i in range(3, 8)]
        verdict = gen.judge(self.clip(), ladder_report(rungs))
        assert not verdict.ok
        assert any("sharp rungs" in reason for reason in verdict.reasons)

    def test_blurred_rungs_on_the_wrong_side_do_not_count_as_reach(self):
        """Reach is measured on frames that are both sharp and on the right side."""
        rungs = [(float(i * 8), False) for i in range(1, 8)] + [(-60.0, True)]
        verdict = gen.judge(self.clip(expect_yaw_sign="negative"), ladder_report(rungs))
        assert not verdict.ok


class TestJudgingRoles:
    def clip(self, **kwargs: Any) -> gen.Clip:
        defaults: dict[str, Any] = {
            "id": "eyes",
            "order": 1,
            "title": "t",
            "why": "",
            "check": "roles",
            "prompt": "p",
            "want_roles": ("source_brows_up", "source_eyes_wide", "source_eyes_closed"),
            "min_roles": 2,
        }
        return gen.Clip(**{**defaults, **kwargs})

    def test_enough_roles_passes(self):
        report = roles_report(["source_brows_up", "source_eyes_wide"])
        assert gen.judge(self.clip(), report).ok

    def test_too_few_roles_names_what_is_missing(self):
        verdict = gen.judge(self.clip(), roles_report(["source_brows_up"]))
        assert not verdict.ok
        assert any("eyes_wide" in reason for reason in verdict.reasons)

    def test_a_role_the_engine_does_not_have_stops_the_run(self):
        """A typo in the catalogue must not buy another take of the same clip."""
        verdict = gen.judge(self.clip(want_roles=("source_wink",)), roles_report([]))
        assert verdict.fatal

    def test_an_extractor_error_is_fatal_rather_than_a_bad_clip(self):
        verdict = gen.judge(self.clip(), {"error": "model bundle missing"})
        assert not verdict.ok
        assert verdict.fatal

    def test_a_clip_with_no_usable_frames_is_a_bad_clip(self):
        verdict = gen.judge(self.clip(), roles_report([], usable=0))
        assert not verdict.ok
        assert not verdict.fatal


class TestLedger:
    def test_it_survives_a_crash_between_attempts(self, tmp_path):
        ledger = gen.Ledger.load(tmp_path / "ledger.json")
        ledger.record(clip="a", tier="final", model="m", usd=3.2, ok=True)
        assert gen.Ledger.load(tmp_path / "ledger.json").total_usd == pytest.approx(3.2)

    def test_it_totals_every_attempt_paid_or_wasted(self, tmp_path):
        ledger = gen.Ledger.load(tmp_path / "ledger.json")
        ledger.record(clip="a", tier="probe", model="m", usd=1.2, ok=False)
        ledger.record(clip="a", tier="final", model="m", usd=3.2, ok=True)
        assert ledger.total_usd == pytest.approx(4.4)

    def test_only_a_usable_final_take_counts_as_passed(self, tmp_path):
        ledger = gen.Ledger.load(tmp_path / "ledger.json")
        ledger.record(clip="a", tier="probe", model="m", usd=1.2, ok=True)
        assert not ledger.passed("a")
        ledger.record(clip="a", tier="final", model="m", usd=3.2, ok=False)
        assert not ledger.passed("a")
        ledger.record(clip="a", tier="final", model="m", usd=3.2, ok=True)
        assert ledger.passed("a")

    def test_attempts_are_counted_per_tier(self, tmp_path):
        ledger = gen.Ledger.load(tmp_path / "ledger.json")
        ledger.record(clip="a", tier="probe", model="m", usd=1.2, ok=False)
        ledger.record(clip="a", tier="probe", model="m", usd=1.2, ok=True)
        assert ledger.attempts("a", "probe") == 2
        assert ledger.attempts("a", "final") == 0


class TestTheRunGuards:
    """The two ways a run stops before it spends more."""

    @pytest.fixture
    def api(self) -> gen.Api:
        return gen.Api(
            model_final="final-model",
            model_probe="probe-model",
            usd_per_second_final=0.40,
            usd_per_second_probe=0.15,
            duration_seconds=8,
            resolution="1080p",
            aspect_ratio="16:9",
            person_generation="allow_adult",
            poll_seconds=0.0,
            timeout_seconds=1.0,
            drop_fields=(),
        )

    @pytest.fixture
    def clip(self) -> gen.Clip:
        return gen.Clip(
            id="turn",
            order=1,
            title="t",
            why="",
            check="ladder",
            prompt="p",
            probe_first=True,
            expect_yaw_sign="positive",
        )

    def args(self, tmp_path: Path, **kwargs: Any) -> argparse.Namespace:
        defaults: dict[str, Any] = {
            "out": tmp_path,
            "portrait": tmp_path / "portrait.png",
            "budget": 50.0,
            "max_attempts": 2,
            "no_probe": False,
            "force": False,
        }
        return argparse.Namespace(**{**defaults, **kwargs})

    @pytest.fixture
    def calls(self, monkeypatch: Any) -> list[tuple[str, str]]:
        """Record what would have been generated, and generate nothing."""
        seen: list[tuple[str, str]] = []

        def fake_generate(
            client: Any,
            api: Any,
            model: str,
            prompt: str,
            negative: str,
            portrait: Path,
            dest: Path,
        ) -> None:
            seen.append((model, str(dest)))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"video")

        monkeypatch.setattr(gen, "generate", fake_generate)
        return seen

    def measures(self, monkeypatch: Any, *verdicts: gen.Verdict) -> None:
        remaining = list(verdicts)

        def fake_measure(
            video: Path, work: Path, clip: gen.Clip, portrait: Path
        ) -> tuple[gen.Verdict, dict[str, Any]]:
            return remaining.pop(0), {"usable": 10}

        monkeypatch.setattr(gen, "measure", fake_measure)

    def test_a_failed_probe_never_buys_the_expensive_take(
        self, monkeypatch, tmp_path, api, clip, calls
    ):
        """The whole point of the probe: the prompt is wrong, so stop here."""
        self.measures(
            monkeypatch,
            gen.Verdict(False, reasons=("wrong way",)),
            gen.Verdict(False, reasons=("wrong way",)),
        )
        ledger = gen.Ledger.load(tmp_path / "ledger.json")

        outcome = gen.run_clip(clip, api, "neg", object(), ledger, self.args(tmp_path))

        assert outcome == "failed"
        assert [model for model, _ in calls] == ["probe-model", "probe-model"]
        assert ledger.total_usd == pytest.approx(2.4)

    def test_a_passing_probe_goes_on_to_the_final_take(
        self, monkeypatch, tmp_path, api, clip, calls
    ):
        self.measures(monkeypatch, gen.Verdict(True), gen.Verdict(True))
        ledger = gen.Ledger.load(tmp_path / "ledger.json")

        outcome = gen.run_clip(clip, api, "neg", object(), ledger, self.args(tmp_path))

        assert outcome == "passed"
        assert [model for model, _ in calls] == ["probe-model", "final-model"]

    def test_the_budget_stops_the_run_before_the_request(
        self, monkeypatch, tmp_path, api, clip, calls
    ):
        self.measures(monkeypatch, gen.Verdict(True))
        ledger = gen.Ledger.load(tmp_path / "ledger.json")
        ledger.record(clip="earlier", tier="final", model="m", usd=49.5, ok=True)

        outcome = gen.run_clip(clip, api, "neg", object(), ledger, self.args(tmp_path))

        assert outcome == "budget"
        assert calls == []

    def test_a_clip_that_already_passed_is_not_paid_for_twice(
        self, monkeypatch, tmp_path, api, clip, calls
    ):
        ledger = gen.Ledger.load(tmp_path / "ledger.json")
        ledger.record(clip="turn", tier="final", model="m", usd=3.2, ok=True)

        outcome = gen.run_clip(clip, api, "neg", object(), ledger, self.args(tmp_path))

        assert outcome == "skipped"
        assert calls == []

    def test_a_fatal_verdict_stops_instead_of_regenerating(
        self, monkeypatch, tmp_path, api, clip, calls
    ):
        """A broken checkout must not read as a bad clip."""
        self.measures(monkeypatch, gen.Verdict(False, reasons=("no model",), fatal=True))
        ledger = gen.Ledger.load(tmp_path / "ledger.json")

        outcome = gen.run_clip(clip, api, "neg", object(), ledger, self.args(tmp_path))

        assert outcome == "fatal"
        assert len(calls) == 1

    def test_no_probe_goes_straight_to_the_final_model(
        self, monkeypatch, tmp_path, api, clip, calls
    ):
        self.measures(monkeypatch, gen.Verdict(True))
        ledger = gen.Ledger.load(tmp_path / "ledger.json")

        outcome = gen.run_clip(
            clip, api, "neg", object(), ledger, self.args(tmp_path, no_probe=True)
        )

        assert outcome == "passed"
        assert [model for model, _ in calls] == ["final-model"]


class TestBuildingTheLadderFromFrames:
    """The ladder stopped at 43 degrees while the clip it came from held a sharp
    profile at 54, because the video pass rejects deep-profile frames on filters
    that are right for picking an open mouth: too small a face, identity ratios
    that drifted, a frame softer than the sharpest in the clip. Reading the
    dumped frames directly is what gets the profile back.
    """

    def rung(self, yaw: float, detail: float) -> Any:
        return ladder.Rung(
            path=Path(f"{yaw}.jpg"), yaw=yaw, detail=detail, face_width=900.0, candidate=None
        )

    def test_one_rung_per_band_and_the_sharpest_wins(self):
        chosen = ladder.best_per_band(
            [self.rung(9.5, 1.0), self.rung(10.0, 5.0), self.rung(20.0, 2.0)], 3.0
        )

        assert [r.yaw for r in chosen] == [10.0, 20.0]

    def test_two_bands_that_would_write_the_same_file_become_one(self):
        """A tenth of a degree either side of a boundary lands on one name, and
        the manifest would then hold one picture twice."""
        chosen = ladder.best_per_band([self.rung(31.51, 1.0), self.rung(31.49, 9.0)], 3.0)

        assert len(chosen) == 1
        assert chosen[0].detail == 9.0

    def test_a_profile_is_not_dropped_for_being_less_sharp_than_a_frontal_frame(self):
        """No judgement across bands: the near eye is foreshortened at a profile,
        so comparing it with a square-on lash line compares nothing."""
        chosen = ladder.best_per_band([self.rung(0.0, 500.0), self.rung(55.0, 6.0)], 3.0)

        assert [r.yaw for r in chosen] == [0.0, 55.0]

    def test_the_name_carries_the_angle_and_its_sign(self):
        assert ladder.rung_name(-43.04) == "turn_m43_0.png"
        assert ladder.rung_name(6.0) == "turn_p06_0.png"
