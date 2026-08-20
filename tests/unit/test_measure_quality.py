"""Tests for the quality measure.

The tool's whole claim is that a number it prints can be trusted in a commit
message, so what is tested here is mostly *refusal*: that it will not compare
two runs that measured different things, will not quote a frame cost taken on a
machine whose speed moved, and will not print a ladder rung under a name that
looks like a measurement and is actually an array index.

The measurement itself is covered where it can be covered without a clip and an
avatar library: the residual's defining property is that a rigid move of the
face costs nothing and a deformation costs something, and that is checkable in
three lines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import measure_quality as quality  # noqa: E402

from sha256.core.avatar import AssetRef, AvatarDescriptor, AvatarManifest  # noqa: E402


def spread(mean: float, p95: float | None = None) -> dict[str, object]:
    """One metric's summary, as the report stores it."""
    return {
        "unit": "px",
        "count": 100,
        "mean": mean,
        "min": mean,
        "max": mean,
        "p50": mean,
        "p95": mean if p95 is None else p95,
        "p99": mean,
    }


def report(
    *,
    clip: str = "aaaaaaaaaaaa",
    library: str = "bbbbbbbbbbbb",
    first: int = 30,
    last: int = 1846,
    residual: float = 6.83,
    animate: float = 21.4,
    speed_before: float = 0.395,
    speed_after: float = 0.396,
    label: str = "",
) -> dict[str, object]:
    """A complete report, with the parts a comparison reads made adjustable."""
    machine_speed = quality.machine_verdict(
        {"median_ms": speed_before, "fastest_ms": speed_before, "spread": 1.04},
        {"median_ms": speed_after, "fastest_ms": speed_after, "spread": 1.04},
    )
    return {
        "tool": "measure_quality",
        "format": quality.REPORT_FORMAT,
        "run": {"at": "2026-08-20T10:00:00Z", "label": label},
        "conditions": {
            "code": {"commit": "abc1234", "dirty": False, "changed_files": 0},
            "clip": {
                "path": "recordings/driving.mkv",
                "fingerprint": clip,
                "frames": 1847,
                "width": 1280,
                "height": 720,
                "fps": 30.0,
            },
            "avatar": {
                "id": "my_avatar",
                "manifest_version": "1.4.0",
                "assets": 86,
                "fingerprint": library,
            },
            "sampling": {"first": first, "last": last, "warmup": 30, "segment": None},
            "machine": {
                "platform": "Windows-11",
                "processor": "Intel64",
                "logical_cpus": 28,
                "cv2_threads": 28,
                "speed": machine_speed,
            },
            "libraries": {
                "python": "3.12.13",
                "numpy": "2.5.1",
                "cv2": "5.0.0",
                "mediapipe": "1.0.0",
            },
            "gpu": {"name": "RTX 5070 Ti", "driver": "610.62"},
        },
        "counts": {"read": 1847, "measured": 1817, "no_face": 0, "passthrough": 0, "fit_failed": 0},
        "metrics": {
            "residual_px": {
                "all": spread(residual),
                "frontal": spread(residual),
                "turned": spread(residual + 0.5),
            },
            "sharpness_near_eye": {
                "all": spread(185.0),
                "frontal": spread(190.0),
                "turned": spread(58.0),
            },
            "sharpness_face": {
                "all": spread(140.0),
                "frontal": spread(140.0),
                "turned": spread(58.0),
            },
            "face_width_px": {
                "all": spread(300.0),
                "frontal": spread(300.0),
                "turned": spread(300.0),
            },
            "animate_ms": {
                "all": spread(animate, animate + 2.0),
                "frontal": spread(animate, animate + 2.0),
                "turned": spread(animate, animate + 2.0),
            },
        },
        "base_mesh_frames": 0,
        "photographs": [
            {
                "role": "source_mouth_o",
                "region": "mouth",
                "frames": 1556,
                "percent": 84.2,
                "median_share": 0.44,
                "rung": None,
                "file": None,
            }
        ],
    }


class TestRefusingToCompare:
    """Two numbers side by side look like a comparison whatever produced them."""

    def test_a_rebuilt_library_is_announced_rather_than_refused(self, capsys):
        """Most of the work on this engine *is* rebuilding a ladder, so a tool
        that refused to measure that could not report the days it matters most.
        What it must not do is let the arrow read as a code change."""
        code = quality.compare(report(), report(library="cccccccccccc"))
        printed = capsys.readouterr().out

        assert code == 0
        assert "THE PHOTOGRAPHS CHANGED" in printed
        assert "two different avatars" in printed

    def test_a_different_clip_refuses_the_comparison(self, capsys):
        code = quality.compare(report(), report(clip="cccccccccccc"))

        assert code == 2
        assert "clip.fingerprint" in capsys.readouterr().out

    def test_a_different_frame_range_refuses_the_comparison(self, capsys):
        """Measuring the turn segment against the whole clip compares a defect
        against an average that is 97% frontal."""
        code = quality.compare(report(), report(first=1342, last=1489))

        assert code == 2
        assert "sampling.first" in capsys.readouterr().out

    def test_an_older_report_format_refuses_the_comparison(self, capsys):
        stale = report()
        stale["format"] = quality.REPORT_FORMAT - 1

        assert quality.compare(stale, report()) == 2
        assert "format" in capsys.readouterr().out

    def test_the_refusal_names_what_differed(self, capsys):
        """A refusal that does not say why is one the reader will work around."""
        quality.compare(report(), report(clip="cccccccccccc"))

        assert "clip.fingerprint" in capsys.readouterr().out


class TestQuotingTheFrameCost:
    """The mistake this exists to prevent: a duration taken on a busy machine.

    The rule is not "the machine must not change" — a full-clip run warms this
    CPU by about half, every time, and refusing that would refuse every run. It
    is that two runs must have *begun* in the same state and warmed alike.
    """

    def test_two_runs_that_began_alike_compare_the_frame_cost(self, capsys):
        code = quality.compare(
            report(animate=21.5, speed_before=0.392, speed_after=0.590),
            report(animate=20.1, speed_before=0.392, speed_after=0.590),
        )
        printed = capsys.readouterr().out

        assert code == 0
        assert "not comparable" not in printed
        assert "-1.40" in printed

    def test_warming_under_the_run_s_own_load_is_reported_not_refused(self, capsys):
        """Measured: probe, sixty seconds of many-threaded warping, probe again —
        0.392 ms to 0.572 ms. Treating that as contention refused every full-clip
        run, always by about half, which is how it was found."""
        quality.compare(
            report(animate=21.5, speed_before=0.392, speed_after=0.590),
            report(animate=21.9, speed_before=0.392, speed_after=0.590),
        )

        assert "not comparable" not in capsys.readouterr().out

    def test_two_runs_that_warmed_differently_withhold_the_frame_cost(self, capsys):
        """A two-second run barely warms the machine and a full clip halves its
        speed. Their frame costs are not the same measurement."""
        short_run = report(animate=21.5, speed_before=0.392, speed_after=0.396)
        full_run = report(animate=33.0, speed_before=0.392, speed_after=0.590)

        code = quality.compare(short_run, full_run)
        printed = capsys.readouterr().out

        assert code == 0
        assert "not comparable" in printed
        assert "33.00" in printed

    def test_two_runs_that_began_on_different_machines_withhold_the_frame_cost(self, capsys):
        """Day 11's frame-cost figure was taken while the machine was busy with
        something else, and had to be corrected the next day. A run that starts
        60% slow started on a machine that was already doing something."""
        rested = report(animate=21.5, speed_before=0.392, speed_after=0.590)
        already_busy = report(animate=33.0, speed_before=0.620, speed_after=0.930)

        quality.compare(rested, already_busy)

        assert "started on a machine" in capsys.readouterr().out

    def test_geometry_is_compared_even_when_the_frame_cost_is_not(self, capsys):
        """A busy machine does not move a pixel — two runs an hour apart gave
        identical residuals. Refusing the geometry too would throw away the half
        of the measurement that is still valid."""
        loaded = report(residual=6.10, animate=41.0, speed_before=0.62, speed_after=0.93)

        quality.compare(report(residual=6.83), loaded)
        printed = capsys.readouterr().out

        assert "warp residual, all" in printed
        assert "-0.73" in printed

    def test_a_machine_that_did_not_change_is_reported_without_a_note(self):
        verdict = quality.machine_verdict(
            {"median_ms": 0.395, "fastest_ms": 0.390, "spread": 1.04},
            {"median_ms": 0.398, "fastest_ms": 0.392, "spread": 1.05},
        )

        assert verdict["notes"] == []
        assert verdict["erratic"] is False

    def test_warming_is_reported_as_the_load_rather_than_as_interference(self):
        verdict = quality.machine_verdict(
            {"median_ms": 0.392, "fastest_ms": 0.386, "spread": 1.04},
            {"median_ms": 0.590, "fastest_ms": 0.558, "spread": 1.13},
        )

        assert verdict["erratic"] is False
        assert "sustained load, not another program" in verdict["notes"][0]

    def test_a_machine_that_got_faster_names_what_that_means(self):
        """Faster at the end means something else had the CPU at the start, which
        is the one direction that cannot be this run's own heat."""
        verdict = quality.machine_verdict(
            {"median_ms": 0.620, "fastest_ms": 0.600, "spread": 1.04},
            {"median_ms": 0.398, "fastest_ms": 0.392, "spread": 1.05},
        )

        assert "competing at the start" in verdict["notes"][0]

    def test_an_erratic_probe_is_reported_even_without_drift(self):
        """A machine can average the right speed while stalling repeatedly, and a
        p95 taken through that is not this engine's p95."""
        verdict = quality.machine_verdict(
            {"median_ms": 0.395, "fastest_ms": 0.390, "spread": 1.04},
            {"median_ms": 0.396, "fastest_ms": 0.300, "spread": 2.10},
        )

        assert verdict["erratic"] is True
        assert any("within itself" in note for note in verdict["notes"])

    def test_the_frame_cost_is_reported_for_each_end_of_the_run(self):
        """One average over a run that warms by half is two measurements added
        together. Scenario S5 of the performance plan exists for this and has
        never been run."""
        frames = [
            quality.Frame(
                index=index,
                yaw=0.0,
                residual_px=6.8,
                sharpness_face=140.0,
                sharpness_near_eye=185.0,
                face_width_px=300.0,
                animate_ms=20.0 if index < 50 else 30.0,
                base_mesh=False,
            )
            for index in range(100)
        ]

        drift = quality.frame_cost_drift(frames)

        assert drift["start"]["mean"] == 20.0
        assert drift["end"]["mean"] == 30.0


class TestTheResidual:
    """What the number means, checked against its own definition."""

    def test_a_face_that_only_moved_rigidly_costs_the_warp_nothing(self):
        """Rotation, scale and translation are the head being nearer or turned,
        not warp error. If they registered, every frame of a moving head would
        read as a defect."""
        rng = np.random.default_rng(0)
        mesh = rng.uniform(100.0, 400.0, (quality.FACE_MESH_POINTS, 2)).astype(np.float32)
        angle = np.deg2rad(12.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32
        )
        moved = (mesh @ rotation.T) * 1.4 + (60.0, -25.0)

        assert quality.rigid_residual(mesh, moved) == pytest.approx(0.0, abs=1e-3)

    def test_a_face_that_was_deformed_costs_the_warp_something(self):
        rng = np.random.default_rng(0)
        mesh = rng.uniform(100.0, 400.0, (quality.FACE_MESH_POINTS, 2)).astype(np.float32)
        deformed = mesh.copy()
        deformed[:100, 1] += 8.0

        residual = quality.rigid_residual(mesh, deformed)

        assert residual is not None
        assert residual > 1.0

    def test_the_iris_fans_are_left_out_of_it(self):
        """The irises are positioned by gaze logic rather than carried by the warp,
        so counting them would report another subsystem's decisions as warp error."""
        rng = np.random.default_rng(0)
        mesh = rng.uniform(100.0, 400.0, (478, 2)).astype(np.float32)
        with_moved_irises = mesh.copy()
        with_moved_irises[quality.FACE_MESH_POINTS :, :] += 40.0

        assert quality.rigid_residual(mesh, with_moved_irises) == pytest.approx(0.0, abs=1e-3)


class TestNamingWhatIsOnScreen:
    """A rung's name is its position in the manifest, which reads like a value."""

    @staticmethod
    def descriptor() -> AvatarDescriptor:
        assets = [
            AssetRef(path="portrait.png", role="source_portrait"),
            AssetRef(path="eyes_w023.png", role="source_eyes_step", eyes=-0.228),
            AssetRef(path="turn_p02.png", role="source_turn_step", yaw=2.1),
        ]
        manifest = AvatarManifest(
            manifest_version=1,
            id="my_avatar",
            name="My Avatar",
            version="1.4.0",
            assets=assets,
            engine={"engine_id": "warp", "estimated_vram_mb": 0},
        )
        return AvatarDescriptor(manifest=manifest, root=Path("/tmp/my_avatar"))

    def test_a_rung_reports_the_amount_it_was_measured_at(self):
        """`eyes@1` is the first asset, not an eye 1% shut. Read as a value it
        would have said the opposite of the truth: this rung is an eye held
        *wider* than resting."""
        labels = quality.rung_labels(self.descriptor())

        assert labels["eyes@1"]["value"] == pytest.approx(-0.228)
        assert labels["eyes@1"]["file"] == "eyes_w023.png"
        assert labels["turn@2"]["axis"] == "yaw"

    def test_a_photograph_with_no_axis_is_not_given_one(self):
        """CONTRIBUTING rule 6: a blank is a statement, an invented zero is not."""
        labels = quality.rung_labels(self.descriptor())

        assert "turn@0" not in labels
        assert "eyes@0" not in labels

    def test_the_table_carries_each_rung_s_amount(self):
        frames = [
            quality.Frame(
                index=index,
                yaw=0.0,
                residual_px=6.8,
                sharpness_face=140.0,
                sharpness_near_eye=185.0,
                face_width_px=300.0,
                animate_ms=21.0,
                base_mesh=False,
                roles={"eyes@1": 1.0},
            )
            for index in range(4)
        ]

        rows = quality.photographs(frames, quality.rung_labels(self.descriptor()))

        assert rows[0]["role"] == "eyes@1"
        assert rows[0]["percent"] == 100.0
        assert rows[0]["rung"] == "eyes -0.228"


class TestChoosingTheFrames:
    """Which frames were measured is a condition, not a detail."""

    @staticmethod
    def args(**overrides: object) -> argparse.Namespace:

        values = {
            "segment": None,
            "first": None,
            "last": None,
            "warmup": quality.DEFAULT_WARMUP_FRAMES,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    SEGMENTS: ClassVar[list[dict[str, object]]] = [
        {"instruction": "SIT STILL. Relax.", "first_frame": 0, "last_frame": 116},
        {
            "instruction": "Turn your head slowly LEFT, then back.",
            "first_frame": 1342,
            "last_frame": 1489,
        },
        {
            "instruction": "Turn your head slowly RIGHT, then back.",
            "first_frame": 1490,
            "last_frame": 1637,
        },
    ]

    def test_the_warm_up_frames_are_never_measured(self):
        """The engine estimates the driver's resting face from the frames it has
        seen. Backlog 27 is an entire defect made of the first second of a run."""
        assert quality.resolve_range(self.args(first=0), 1847, []) == (30, 1846)

    def test_a_segment_is_measured_by_its_instruction(self):
        chosen = quality.resolve_range(self.args(segment="left"), 1847, self.SEGMENTS)

        assert chosen == (1342, 1489)

    def test_an_ambiguous_segment_is_refused_rather_than_guessed(self):
        chosen = quality.resolve_range(self.args(segment="turn your head"), 1847, self.SEGMENTS)

        assert isinstance(chosen, str)
        assert "more than one" in chosen

    def test_a_segment_that_matches_nothing_lists_the_ones_that_exist(self):
        chosen = quality.resolve_range(self.args(segment="wink"), 1847, self.SEGMENTS)

        assert isinstance(chosen, str)
        assert "SIT STILL" in chosen

    def test_an_empty_range_is_refused(self):
        chosen = quality.resolve_range(self.args(first=100, last=50), 1847, [])

        assert isinstance(chosen, str)


class TestReportsOnDisk:
    def test_a_file_this_tool_did_not_write_is_refused(self, tmp_path):
        """`--baseline` pointed at the wrong JSON must say so, not compare fields
        that happen to be missing and print zeroes."""
        stranger = tmp_path / "other.json"
        stranger.write_text(json.dumps({"tool": "something_else"}), encoding="utf-8")

        assert isinstance(quality.read_report(stranger), str)

    def test_a_file_that_is_not_json_is_refused(self, tmp_path):
        broken = tmp_path / "broken.json"
        broken.write_text("not json at all", encoding="utf-8")

        assert isinstance(quality.read_report(broken), str)

    def test_a_report_this_tool_wrote_is_read_back(self, tmp_path):
        written = tmp_path / "run.json"
        written.write_text(json.dumps(report()), encoding="utf-8")

        assert quality.read_report(written)["tool"] == "measure_quality"


class TestSummarising:
    def test_every_percentile_is_a_value_that_occurred(self):
        """Nearest-rank, like the application's own statistics. An interpolated
        percentile invents a number, which is unacceptable in a figure used to
        justify a claim."""
        values = [1.0, 2.0, 3.0, 4.0, 100.0]

        summary = quality.spread_of("test", "px", values)

        assert summary.p95 in values
        assert summary.p50 in values
        assert summary.maximum == 100.0
        assert summary.count == 5

    def test_nothing_measured_is_reported_as_nothing_measured(self):
        """A count of zero next to a mean of zero says "measured as zero", which
        is why the count is printed beside every number."""
        summary = quality.spread_of("test", "px", [])

        assert summary.count == 0
