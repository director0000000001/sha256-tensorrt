"""Main application window.

Structural rules, which the finished UI inherits:

*   **The window owns no pipeline logic.** It drives
    :class:`~sha256.pipeline.runner.PipelineRunner` and
    :class:`~sha256.avatar.manager.AvatarManager` and displays their state.
*   **Nothing slow runs on this thread.** The pipeline runs on its own thread;
    this one only paints the newest frame it finds.
*   **One screen, few controls.** What a user needs is a picture, a camera, an
    avatar, and a way to start. Everything diagnostic lives behind the developer
    panel, which appears only when ``developer_mode`` is enabled in
    configuration — so the normal window cannot accumulate debug widgets by
    accident.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
from sha256.pipeline.runner import PipelineRunner
from sha256.ui.panels.debug_panel import DebugPanel
from sha256.ui.preview import PreviewWidget
from sha256.utils.logging import get_logger

if TYPE_CHECKING:
    from sha256.app.bootstrap import ApplicationContext

__all__ = ["MainWindow"]

_LOG = get_logger(__name__)

# Painting faster than the pipeline produces frames only repaints the same
# picture. 33 ms matches a 30 fps source; the preview simply shows whatever the
# newest frame is when it wakes.
_PREVIEW_INTERVAL_MS = 33
_STATUS_INTERVAL_MS = 500

_VIEW_CAMERA = "Camera + tracking"
_VIEW_OUTPUT = "Avatar output"


class MainWindow(QMainWindow):
    """One screen: a live preview, the two selectors, and Start/Stop."""

    def __init__(self, context: ApplicationContext) -> None:
        super().__init__()
        self._context = context
        self._runner = PipelineRunner(context.controller)

        self.setWindowTitle("sha256")
        self.resize(1080, 760)

        self._preview = PreviewWidget()
        self._debug_panel = DebugPanel(context)

        session = self._build_session_tab()
        if context.settings.developer_mode:
            tabs = QTabWidget()
            tabs.addTab(session, "Session")
            tabs.addTab(self._debug_panel, "Developer")
            self.setCentralWidget(tabs)
        else:
            self.setCentralWidget(session)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(_PREVIEW_INTERVAL_MS)
        self._preview_timer.timeout.connect(self._refresh_preview)

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(_STATUS_INTERVAL_MS)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start()

        self._populate_avatars()
        self._populate_cameras()
        self._refresh_status()

    # --- construction ------------------------------------------------------

    def _build_session_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        layout.addWidget(self._preview, 1)
        layout.addLayout(self._build_selector_row())
        layout.addLayout(self._build_control_row())
        return page

    def _build_selector_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        row.addWidget(QLabel("Camera:"))
        self._camera_combo = QComboBox()
        self._camera_combo.setMinimumWidth(220)
        row.addWidget(self._camera_combo, 1)

        row.addSpacing(16)

        row.addWidget(QLabel("Avatar:"))
        self._avatar_combo = QComboBox()
        self._avatar_combo.setMinimumWidth(220)
        self._avatar_combo.currentIndexChanged.connect(self._on_avatar_changed)
        row.addWidget(self._avatar_combo, 1)

        return row

    def _build_control_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        self._start_button = QPushButton("Start")
        self._start_button.setMinimumWidth(110)
        self._start_button.clicked.connect(self._on_start)
        row.addWidget(self._start_button)

        self._stop_button = QPushButton("Stop")
        self._stop_button.setMinimumWidth(110)
        self._stop_button.clicked.connect(self._on_stop)
        row.addWidget(self._stop_button)

        row.addSpacing(16)

        self._view_combo = QComboBox()
        self._view_combo.addItems([_VIEW_CAMERA, _VIEW_OUTPUT])
        row.addWidget(QLabel("View:"))
        row.addWidget(self._view_combo)

        self._landmarks_check = QCheckBox("Show tracking")
        self._landmarks_check.setChecked(True)
        self._landmarks_check.toggled.connect(self._preview.set_show_landmarks)
        row.addWidget(self._landmarks_check)

        self._mirror_check = QCheckBox("Mirror")
        self._mirror_check.setChecked(self._context.settings.camera.mirror)
        self._mirror_check.setToolTip("Affects this preview only, not the published output.")
        self._mirror_check.toggled.connect(self._preview.set_mirror)
        self._preview.set_mirror(self._mirror_check.isChecked())
        row.addWidget(self._mirror_check)

        row.addStretch(1)

        self._fps_label = QLabel()
        self._fps_label.setTextFormat(Qt.TextFormat.PlainText)
        row.addWidget(self._fps_label)

        return row

    # --- population --------------------------------------------------------

    def _populate_avatars(self) -> None:
        self._avatar_combo.blockSignals(True)
        self._avatar_combo.clear()
        self._avatar_combo.addItem("— none —", userData=None)

        for status in self._context.avatars.statuses():
            label = status.name if status.is_usable else f"{status.name}  (error)"
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
            self._runner.start()
        except Exception as exc:
            _LOG.exception("Pipeline start failed")
            QMessageBox.critical(self, "Could not start", str(exc))
            self._refresh_status()
            return

        self._preview_timer.start()
        self._refresh_status()

    def _on_stop(self) -> None:
        self._preview_timer.stop()
        self._runner.stop()
        self._preview.clear_frame()
        self._refresh_status()

    # --- display -----------------------------------------------------------

    def _refresh_preview(self) -> None:
        snapshot = self._runner.latest()
        if snapshot is None:
            return

        if self._view_combo.currentText() == _VIEW_CAMERA and snapshot.source is not None:
            self._preview.show_frame(
                snapshot.source.data, snapshot.source.pixel_format, snapshot.tracking
            )
        elif snapshot.frame is not None:
            # The rendered frame is already composited; landmarks would be drawn
            # against a picture they do not correspond to.
            self._preview.show_frame(snapshot.frame.data, snapshot.frame.pixel_format, None)

    def _refresh_status(self) -> None:
        running = self._runner.is_running
        self._start_button.setEnabled(not running)
        self._stop_button.setEnabled(running)

        error = self._runner.last_error
        if error is not None:
            self._preview_timer.stop()
            self._fps_label.setText("")
            self._status_bar.showMessage(f"Pipeline stopped: {error}")
            return

        if running:
            self._fps_label.setText(f"{self._runner.fps:.1f} fps")
        else:
            self._fps_label.setText("")

        state = self._context.controller.state
        avatar = self._context.avatars.active_id or "none"
        tracker = self._context.controller.components.tracker.backend_id
        self._status_bar.showMessage(
            f"Pipeline: {state.value}   |   Tracker: {tracker}   |   Avatar: {avatar}"
        )

        if self._context.settings.developer_mode and state is PipelineState.RUNNING:
            self._debug_panel.refresh()

    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        """Stop the pipeline before the window goes away."""
        self._preview_timer.stop()
        self._status_timer.stop()
        self._runner.stop()
        super().closeEvent(event)  # type: ignore[arg-type]
