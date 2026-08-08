"""Developer/debug panel.

Shows what is wired, what was discovered, and what has been measured. It reads
only from the performance monitor and the managers — it never computes a metric
itself, so what is displayed here is exactly what the pipeline recorded.

Metrics show ``n/a`` until something has actually been measured. Displaying a
zero would read as a measurement, and this project does not report performance
numbers it has not taken.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sha256.core.enums import PipelineStage

if TYPE_CHECKING:
    from sha256.app.bootstrap import ApplicationContext

__all__ = ["DebugPanel"]


class DebugPanel(QWidget):
    """Read-only diagnostics view."""

    def __init__(self, context: ApplicationContext) -> None:
        super().__init__()
        self._context = context

        layout = QVBoxLayout(self)

        layout.addWidget(self._build_wiring_box())
        layout.addWidget(self._build_metrics_box())

        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        layout.addWidget(refresh)
        layout.addStretch(1)

        self.refresh()

    def _build_wiring_box(self) -> QWidget:
        box = QGroupBox("Wiring")
        layout = QVBoxLayout(box)
        self._wiring_label = QLabel()
        self._wiring_label.setWordWrap(True)
        layout.addWidget(self._wiring_label)
        return box

    def _build_metrics_box(self) -> QWidget:
        box = QGroupBox("Metrics")
        layout = QVBoxLayout(box)
        self._metrics_view = QPlainTextEdit()
        self._metrics_view.setReadOnly(True)
        self._metrics_view.setMinimumHeight(260)
        layout.addWidget(self._metrics_view)
        return box

    def refresh(self) -> None:
        """Re-read state and redraw. Cheap enough to call on a timer."""
        settings = self._context.settings
        paths = self._context.paths

        self._wiring_label.setText(
            "\n".join(
                (
                    f"camera        : {settings.camera.backend}",
                    f"tracker       : {settings.tracking.backend}",
                    f"engine        : {settings.engine.backend}",
                    f"renderer      : {settings.rendering.backend}",
                    f"virtual camera: {settings.virtual_camera.backend}",
                    "",
                    f"avatar library: {paths.avatars}",
                    f"developer mode: {settings.developer_mode}",
                    f"VRAM budget   : {settings.gpu.vram_budget_mb} MiB "
                    f"(max {settings.avatars.max_resident} resident)",
                )
            )
        )
        self._metrics_view.setPlainText(self._format_metrics())

    def _format_metrics(self) -> str:
        snapshot = self._context.monitor.snapshot()
        if snapshot.is_empty():
            return (
                "No measurements yet.\n\n"
                "Start the pipeline and step a frame to populate these counters."
            )

        lines: list[str] = ["stage             fps      mean      p95      p99   frames  dropped"]
        lines.append("-" * 70)

        for stage in PipelineStage:
            key = stage.value
            summary = snapshot.duration(key)
            fps = snapshot.fps.get(key)
            total = snapshot.frame_totals.get(key, 0)
            dropped = snapshot.dropped.get(key, 0)

            if summary.count == 0 and fps is None:
                continue

            lines.append(
                f"{key:<16}"
                f"{_fmt(fps, '{:.1f}'):>6}"
                f"{_fmt(summary.mean_ms if summary.count else None, '{:.2f}ms'):>11}"
                f"{_fmt(summary.p95_ms if summary.count else None, '{:.2f}ms'):>10}"
                f"{_fmt(summary.p99_ms if summary.count else None, '{:.2f}ms'):>10}"
                f"{total:>8}{dropped:>9}"
            )

        if snapshot.gauges:
            lines.append("")
            lines.append("gauges:")
            lines.extend(f"  {name}: {value:g}" for name, value in sorted(snapshot.gauges.items()))

        lines.append("")
        lines.append("CPU/GPU/VRAM sampling is not implemented — see docs/benchmarks/.")
        return "\n".join(lines)


def _fmt(value: float | None, spec: str) -> str:
    """Format a value, or ``n/a`` when it was never measured."""
    return "n/a" if value is None else spec.format(value)
