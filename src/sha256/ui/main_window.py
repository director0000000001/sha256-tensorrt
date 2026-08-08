"""Main application window.

Intentionally plain. The goal of this milestone is correct structure, not
visual design: no stylesheet, no custom painting, no icons.

Structural rules being established here, which the real UI will inherit:

*   **The window owns no pipeline logic.** It calls into
    :class:`~sha256.pipeline.controller.PipelineController` and
    :class:`~sha256.avatar.manager.AvatarManager` and renders their state.
*   **Panels are independent widgets.** Each takes the context and can be
    tested, replaced, or moved without touching the others.
*   **Nothing slow runs on this thread.** Avatar loading is slow; today the
    placeholder engine makes it instant, and the call site is marked so the
    threading work lands in one obvious place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sha256.core.enums import PipelineState
from sha256.core.errors import Sha256Error
from sha256.ui.panels.debug_panel import DebugPanel
from sha256.utils.logging import get_logger

if TYPE_CHECKING:
    from sha256.app.bootstrap import ApplicationContext

__all__ = ["MainWindow"]

_LOG = get_logger(__name__)

_STATUS_REFRESH_MS = 500


class MainWindow(QMainWindow):
    """Shell window: selectors, start/stop, and a developer panel."""

    def __init__(self, context: ApplicationContext) -> None:
        super().__init__()
        self._context = context

        self.setWindowTitle("sha256 — foundation build")
        self.resize(900, 620)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_session_tab(), "Session")
        self._debug_panel = DebugPanel(context)
        self._tabs.addTab(self._debug_panel, "Developer")
        self.setCentralWidget(self._tabs)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(_STATUS_REFRESH_MS)
        self._refresh_timer.timeout.connect(self._refresh_status)
        self._refresh_timer.start()

        self._populate_avatars()
        self._populate_cameras()
        self._refresh_status()

    # --- construction ------------------------------------------------------

    def _build_session_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        layout.addWidget(self._build_notice())
        layout.addWidget(self._build_selectors())
        layout.addWidget(self._build_controls())
        layout.addStretch(1)
        return page

    def _build_notice(self) -> QWidget:
        notice = QLabel(
            "Foundation build — no face tracking, no avatar inference, and no "
            "virtual camera are implemented yet.\nAll backends are placeholders; "
            "Start exercises the wiring only."
        )
        notice.setWordWrap(True)
        notice.setTextFormat(Qt.TextFormat.PlainText)
        return notice

    def _build_selectors(self) -> QWidget:
        box = QGroupBox("Sources")
        layout = QVBoxLayout(box)

        avatar_row = QHBoxLayout()
        avatar_row.addWidget(QLabel("Avatar:"))
        self._avatar_combo = QComboBox()
        self._avatar_combo.setMinimumWidth(280)
        self._avatar_combo.currentIndexChanged.connect(self._on_avatar_changed)
        avatar_row.addWidget(self._avatar_combo, 1)
        layout.addLayout(avatar_row)

        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("Camera:"))
        self._camera_combo = QComboBox()
        self._camera_combo.setMinimumWidth(280)
        camera_row.addWidget(self._camera_combo, 1)
        layout.addLayout(camera_row)

        # Placeholder for the avatar gallery this becomes later. Deliberately a
        # combo box today — a gallery needs previews, loading states, and
        # background loading, which are the next milestone's work.
        hint = QLabel("Avatar gallery with previews replaces this selector in a later milestone.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        return box

    def _build_controls(self) -> QWidget:
        box = QGroupBox("Pipeline")
        layout = QHBoxLayout(box)

        self._start_button = QPushButton("Start")
        self._start_button.clicked.connect(self._on_start)
        layout.addWidget(self._start_button)

        self._stop_button = QPushButton("Stop")
        self._stop_button.clicked.connect(self._on_stop)
        layout.addWidget(self._stop_button)

        self._step_button = QPushButton("Step one frame")
        self._step_button.setToolTip(
            "Run a single frame through the placeholder stages. "
            "The continuous real-time loop is a later milestone."
        )
        self._step_button.clicked.connect(self._on_step)
        layout.addWidget(self._step_button)

        layout.addStretch(1)

        self._state_label = QLabel()
        layout.addWidget(self._state_label)
        return box

    # --- population --------------------------------------------------------

    def _populate_avatars(self) -> None:
        self._avatar_combo.blockSignals(True)
        self._avatar_combo.clear()
        self._avatar_combo.addItem("— none —", userData=None)

        for status in self._context.avatars.statuses():
            label = f"{status.name}  [{status.avatar_id}]"
            if not status.is_usable:
                label += "  (error)"
            self._avatar_combo.addItem(label, userData=status.avatar_id)

        active = self._context.avatars.active_id
        if active is not None:
            index = self._avatar_combo.findData(active)
            if index >= 0:
                self._avatar_combo.setCurrentIndex(index)

        self._avatar_combo.blockSignals(False)

    def _populate_cameras(self) -> None:
        self._camera_combo.clear()
        camera = self._context.controller.components.camera
        for spec in camera.enumerate_devices():
            self._camera_combo.addItem(spec.display_name, userData=spec.device_id)
        if self._camera_combo.count() == 0:
            self._camera_combo.addItem("No devices found", userData=None)

    # --- actions -----------------------------------------------------------

    def _on_avatar_changed(self, index: int) -> None:
        avatar_id = self._avatar_combo.itemData(index)
        manager = self._context.avatars

        if avatar_id is None:
            manager.deselect()
            return

        # TODO(threading): move to a worker thread once a real engine is wired.
        # A neural engine load takes seconds and would freeze this window.
        try:
            manager.select(avatar_id)
        except Sha256Error as exc:
            _LOG.warning("Could not select avatar %s: %s", avatar_id, exc)
            QMessageBox.warning(self, "Avatar unavailable", str(exc))
            self._avatar_combo.setCurrentIndex(0)

    def _on_start(self) -> None:
        try:
            self._context.controller.start()
        except Exception as exc:
            _LOG.exception("Pipeline start failed")
            QMessageBox.critical(self, "Could not start", str(exc))
        self._refresh_status()

    def _on_stop(self) -> None:
        self._context.controller.stop()
        self._refresh_status()

    def _on_step(self) -> None:
        controller = self._context.controller
        if not controller.is_running:
            QMessageBox.information(self, "Not running", "Start the pipeline first.")
            return

        frame = controller.process_once()
        if frame is None:
            self._status_bar.showMessage("No frame ready yet — try again.", 2000)
        else:
            self._status_bar.showMessage(
                f"Frame {frame.metadata.frame_id}: "
                f"{frame.width}x{frame.height}, "
                f"latency {frame.pipeline_latency * 1000:.1f} ms",
                3000,
            )
        self._debug_panel.refresh()

    # --- status ------------------------------------------------------------

    def _refresh_status(self) -> None:
        state = self._context.controller.state
        self._state_label.setText(f"State: {state.value}")

        running = state is PipelineState.RUNNING
        self._start_button.setEnabled(not running)
        self._stop_button.setEnabled(running)
        self._step_button.setEnabled(running)

        active = self._context.avatars.active_id or "none"
        message = f"Avatar: {active}   |   Pipeline: {state.value}"
        if state is PipelineState.FAILED and self._context.controller.last_error:
            message += f"   |   {self._context.controller.last_error}"
        self._status_bar.showMessage(message)

    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        """Stop the pipeline before the window goes away."""
        self._refresh_timer.stop()
        self._context.controller.stop()
        super().closeEvent(event)  # type: ignore[arg-type]
