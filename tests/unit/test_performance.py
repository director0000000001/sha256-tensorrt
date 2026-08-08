"""Tests for metric collection.

Percentiles and rolling behaviour are asserted precisely, because these numbers
will later be used to justify performance claims — a quietly wrong statistic
would be worse than none.
"""

from __future__ import annotations

import pytest

from sha256.core.enums import PipelineStage
from sha256.performance.metrics import FrameCounter, MetricSummary, RollingWindow
from sha256.performance.monitor import InProcessPerformanceMonitor, NullPerformanceMonitor
from sha256.performance.probes import NullSystemProbe, SystemSample


class TestRollingWindow:
    def test_empty_window_summarises_to_zeros(self):
        summary = RollingWindow("test").summarise()
        assert summary.count == 0
        assert summary.mean == 0.0

    def test_discards_oldest_beyond_capacity(self):
        window = RollingWindow("test", capacity=3)
        for value in (1.0, 2.0, 3.0, 4.0):
            window.add(value)

        summary = window.summarise()
        assert summary.count == 3
        assert summary.minimum == 2.0
        assert summary.maximum == 4.0

    def test_percentiles_return_observed_values(self):
        """Nearest-rank: every reported percentile actually occurred."""
        window = RollingWindow("test", capacity=100)
        for value in range(1, 101):
            window.add(float(value))

        summary = window.summarise()
        assert summary.p50 in {50.0, 51.0}
        assert summary.p95 in {95.0, 96.0}
        assert summary.p99 in {99.0, 100.0}

    def test_rejects_zero_capacity(self):
        with pytest.raises(ValueError, match="at least 1"):
            RollingWindow("test", capacity=0)

    def test_millisecond_helpers(self):
        summary = MetricSummary(
            name="t",
            count=1,
            mean=0.016,
            minimum=0.016,
            maximum=0.016,
            p50=0.016,
            p95=0.016,
            p99=0.016,
        )
        assert summary.mean_ms == pytest.approx(16.0)


class TestFrameCounter:
    def test_fps_is_derived_from_timestamps(self):
        counter = FrameCounter()
        for index in range(11):
            counter.record(index * 0.1)

        assert counter.fps == pytest.approx(10.0)
        assert counter.total == 11

    def test_single_sample_has_no_rate(self):
        counter = FrameCounter()
        counter.record(1.0)
        assert counter.fps == 0.0

    def test_identical_timestamps_do_not_divide_by_zero(self):
        counter = FrameCounter()
        counter.record(5.0)
        counter.record(5.0)
        assert counter.fps == 0.0

    def test_dropped_frames_are_counted_separately(self):
        counter = FrameCounter()
        counter.record(0.0)
        counter.record_dropped(3)

        assert counter.total == 1
        assert counter.dropped == 3


class TestInProcessMonitor:
    def test_measure_records_a_duration(self):
        monitor = InProcessPerformanceMonitor()
        with monitor.measure(PipelineStage.TRACKING):
            pass

        assert monitor.snapshot().duration("tracking").count == 1

    def test_duration_is_recorded_even_when_the_block_raises(self):
        """Metrics must not look healthiest exactly when the pipeline breaks."""
        monitor = InProcessPerformanceMonitor()

        with pytest.raises(RuntimeError), monitor.measure(PipelineStage.ANIMATION):
            raise RuntimeError("boom")

        assert monitor.snapshot().duration("animation").count == 1

    def test_disabled_monitor_records_nothing(self):
        monitor = InProcessPerformanceMonitor(enabled=False)
        with monitor.measure(PipelineStage.CAPTURE):
            pass
        monitor.record_frame(PipelineStage.CAPTURE)

        assert monitor.snapshot().is_empty()

    def test_frames_and_drops_are_tracked(self):
        monitor = InProcessPerformanceMonitor()
        monitor.record_frame(PipelineStage.CAPTURE)
        monitor.record_frame(PipelineStage.CAPTURE)
        monitor.record_dropped(PipelineStage.CAPTURE, 2)

        snapshot = monitor.snapshot()
        assert snapshot.frame_totals["capture"] == 2
        assert snapshot.dropped["capture"] == 2

    def test_gauges_hold_the_latest_value(self):
        monitor = InProcessPerformanceMonitor()
        monitor.record_value("gpu.vram_used_mb", 1024.0)
        monitor.record_value("gpu.vram_used_mb", 2048.0)

        assert monitor.snapshot().gauges["gpu.vram_used_mb"] == 2048.0

    def test_reset_clears_everything(self):
        monitor = InProcessPerformanceMonitor()
        monitor.record_duration(PipelineStage.RENDERING, 0.01)
        monitor.reset()

        assert monitor.snapshot().is_empty()

    def test_snapshot_is_detached_from_later_writes(self):
        monitor = InProcessPerformanceMonitor()
        monitor.record_duration(PipelineStage.CAPTURE, 0.01)
        snapshot = monitor.snapshot()

        monitor.record_duration(PipelineStage.CAPTURE, 0.02)

        assert snapshot.duration("capture").count == 1


class TestNullMonitor:
    def test_satisfies_the_protocol_without_recording(self):
        monitor = NullPerformanceMonitor()
        with monitor.measure(PipelineStage.CAPTURE):
            pass
        monitor.record_frame(PipelineStage.CAPTURE)
        monitor.record_value("x", 1.0)

        assert monitor.enabled is False
        assert monitor.snapshot().is_empty()


class TestSystemProbe:
    def test_null_probe_reports_nothing_rather_than_zero(self):
        """`None` means not measured; `0.0` would read as a measurement."""
        sample = NullSystemProbe().sample()

        assert sample.vram_used_mb is None
        assert sample.has_gpu_data is False
        assert NullSystemProbe().is_available is False

    def test_free_vram_needs_both_endpoints(self):
        assert SystemSample(vram_used_mb=1000.0).vram_free_mb is None
        assert SystemSample(vram_used_mb=1000.0, vram_total_mb=16384.0).vram_free_mb == 15384.0
