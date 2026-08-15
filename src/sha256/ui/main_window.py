"""Main application window.

Structural rules, which the finished UI inherits:

*   **The window owns no pipeline logic.** It drives
    :class:`~sha256.pipeline.runner.PipelineRunner` and
    :class:`~sha256.avatar.manager.AvatarManager` and displays their state.
*   **Nothing slow runs on this thread.** The pipeline runs on its own thread;
    this one only paints the newest frame it finds.
*   **The background image is the window.** Everything on top of it is either
    the video or a translucent control surface — no opaque panel is allowed to
    cover it, which is the mistake the first version made.

The shape is deliberately spare: before Start the window is the background
image and a single control bar along the bottom, nothing else. No title, no
tagline, no centred panel. Running swaps the empty area for the video.
Diagnostics live behind the developer panel, which appears only when
``developer_mode`` is enabled in configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sha256.core.enums import PipelineState
from sha256.core.errors import Sha256Error
from sha256.pipeline.runner import PipelineRunner
from sha256.ui.background import BackgroundPanel
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

# The avatar view is first, and therefore the default. The tracking view exists
# to answer "is it following me"; the avatar view is the product, and defaulting
# to the diagnostic made the app look like it was doing nothing.
_VIEW_OUTPUT = "Avatar"
_VIEW_CAMERA = "Tracking (no avatar)"

# Everything is translucent so the background photograph stays visible through
# it. Surfaces are dark enough that light text on them is readable over a photo
# of unknown brightness — which the window cannot know in advance.
_STYLE = """
QLabel { color: #f2f2f6; background: transparent; }
QCheckBox { color: #e2e2ea; background: transparent; }

#StatusLine { color: #fca5a5; font-size: 13px; }
#Fps { color: #6ee7b7; font-size: 14px; font-weight: 600; }

#Bar {
    background-color: rgba(12, 12, 18, 165);
    border: 1px solid rgba(255, 255, 255, 28);
    border-radius: 12px;
}

QPushButton {
    color: #f2f2f6;
    background-color: rgba(38, 38, 50, 200);
    border: 1px solid rgba(255, 255, 255, 45);
    border-radius: 8px;
    padding: 8px 20px;
    font-size: 14px;
}
QPushButton:hover { background-color: rgba(58, 58, 76, 225); }
QPushButton:disabled { color: #74747f; background-color: rgba(30, 30, 38, 110); }

#PrimaryButton {
    font-weight: 600;
    color: #06281c;
    background-color: rgba(110, 231, 183, 235);
    border: 1px solid rgba(255, 255, 255, 70);
}
#PrimaryButton:hover { background-color: rgba(140, 245, 205, 255); }
#PrimaryButton:disabled { color: #5a6b64; background-color: rgba(110, 231, 183, 70); }

QComboBox {
    color: #f2f2f6;
    background-color: rgba(38, 38, 50, 200);
    border: 1px solid rgba(255, 255, 255, 45);
    border-radius: 8px;
    padding: 7px 12px;
    font-size: 13px;
}
QComboBox:hover { background-color: rgba(58, 58, 76, 225); }
QComboBox QAbstractItemView {
    color: #f2f2f6;
    background-color: #1b1b24;
    border: 1px solid rgba(255, 255, 255, 40);
    selection-background-color: #38384a;
    outline: none;
}
"""


class MainWindow(QMainWindow):
    """Start screen and running screen, over a background image."""

    def __init__(self, context: ApplicationContext) -> None:
        super().__init__()
        self._context = context
        self._runner = PipelineRunner(context.controller)

        self.setWindowTitle("sha256")
        self.resize(1120, 780)

        self._preview = PreviewWidget()
        self._debug_panel = DebugPanel(context)

        session = self._build_session_page()
        if context.settings.developer_mode:
            tabs = QTabWidget()
            tabs.addTab(session, "Session")
            tabs.addTab(self._debug_panel, "Developer")
            self.setCentralWidget(tabs)
        else:
            self.setCentralWidget(session)

        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(_PREVIEW_INTERVAL_MS)
        self._preview_timer.timeout.connect(self._refresh_preview)

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(_STATUS_INTERVAL_MS)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start()

        self._populate_avatars()
        self._populate_cameras()
        self._refresh_publish_state()
        self._refresh_status()

    # --- construction ------------------------------------------------------

    def _build_session_page(self) -> QWidget:
        ui = self._context.settings.ui
        image = (
            self._context.paths.resolve_relative(ui.background_image)
            if ui.background_image
            else None
        )
        page = BackgroundPanel(image_path=image, dim=ui.background_dim, colour=ui.background_color)
        page.setStyleSheet(_STYLE)

        layout = QVBoxLayout(page)
        layout.setSpacing(14)
        layout.setContentsMargins(28, 24, 28, 22)

        self._stack = QStackedWidget()
        self._stack.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, on=True)
        self._stack.addWidget(self._build_start_screen())
        self._stack.addWidget(self._preview)
        layout.addWidget(self._stack, 1)

        layout.addWidget(self._build_bar())
        return page

    def _build_start_screen(self) -> QWidget:
        """The idle screen: nothing at all.

        Deliberately empty. Start and Stop live in the bar below, so before the
        pipeline runs the window is the background image and nothing else.
        """
        screen = QWidget()
        screen.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, on=True)
        return screen

    def _build_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("Bar")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        # Start and Stop first, at the bottom left, because that is where the
        # eye goes and they are the only two controls that always matter.
        self._primary_button = QPushButton("Start")
        self._primary_button.setObjectName("PrimaryButton")
        self._primary_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._primary_button.clicked.connect(self._on_start)
        layout.addWidget(self._primary_button)

        self._stop_button = QPushButton("Stop")
        self._stop_button.clicked.connect(self._on_stop)
        layout.addWidget(self._stop_button)

        layout.addSpacing(18)

        layout.addWidget(QLabel("Camera"))
        self._camera_combo = QComboBox()
        self._camera_combo.setMinimumWidth(130)
        layout.addWidget(self._camera_combo)

        layout.addSpacing(10)

        layout.addWidget(QLabel("Avatar"))
        self._avatar_combo = QComboBox()
        self._avatar_combo.setMinimumWidth(130)
        self._avatar_combo.currentIndexChanged.connect(self._on_avatar_changed)
        layout.addWidget(self._avatar_combo)

        layout.addSpacing(10)

        # No label: the two entries name themselves.
        self._view_combo = QComboBox()
        self._view_combo.addItems([_VIEW_OUTPUT, _VIEW_CAMERA])
        layout.addWidget(self._view_combo)

        self._landmarks_check = QCheckBox("Dots")
        self._landmarks_check.setToolTip("Draw the tracked landmarks. Tracking view only.")
        self._landmarks_check.setChecked(True)
        self._landmarks_check.toggled.connect(self._preview.set_show_landmarks)
        layout.addWidget(self._landmarks_check)

        self._mirror_check = QCheckBox("Mirror")
        self._mirror_check.setChecked(self._context.settings.camera.mirror)
        self._mirror_check.setToolTip("Affects this preview only, not the published output.")
        self._mirror_check.toggled.connect(self._preview.set_mirror)
        self._preview.set_mirror(self._mirror_check.isChecked())
        layout.addWidget(self._mirror_check)

        layout.addSpacing(10)

        # The one control that decides whether anything outside this window can
        # see the avatar. It is a checkbox rather than an always-on behaviour
        # because appearing in every application's camera list unasked is not a
        # thing a user should have to discover after the fact.
        self._publish_check = QCheckBox("Virtual camera")
        self._publish_check.setChecked(self._context.controller.publish_output)
        self._publish_check.toggled.connect(self._on_publish_toggled)
        layout.addWidget(self._publish_check)

        layout.addSpacing(10)

        # "That looked wrong." Saves the last ten seconds of tracking and the
        # frame on screen, so a complaint about the avatar can be replayed
        # instead of reproduced from a description.
        self._capture_button = QPushButton("That looked wrong")
        self._capture_button.setToolTip(
            "Save the last ten seconds of tracking and the frame on screen, "
            "so the moment can be replayed."
        )
        self._capture_button.clicked.connect(self._on_capture)
        layout.addWidget(self._capture_button)

        layout.addStretch(1)

        # Empty unless something went wrong, so the bar stays quiet in normal
        # use but a failure is never silent.
        self._status_line = QLabel()
        self._status_line.setObjectName("StatusLine")
        layout.addWidget(self._status_line)

        self._fps_label = QLabel()
        self._fps_label.setObjectName("Fps")
        layout.addWidget(self._fps_label)

        return bar

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

    def _on_capture(self) -> None:
        """Save the moment that looked wrong, and say where it went.

        Where it went is not a detail. What is written is tracking — landmarks
        and expression readings, not a photograph — plus the one frame that was
        on screen, and the person pressing the button is entitled to know which
        of those is which and where both landed.
        """
        try:
            target = self._runner.capture(self._context.paths.captures)
        except Exception as exc:
            _LOG.warning("Capture failed: %s", exc)
            QMessageBox.warning(self, "That looked wrong", str(exc))
            return

        QMessageBox.information(
            self,
            "That looked wrong",
            "Saved the last ten seconds.\n\n"
            f"{target}\n\n"
            "tracking.npz — landmark and expression readings, no imagery\n"
            "seen.png — the frame that was on screen",
        )

    def _on_publish_toggled(self, enabled: bool) -> None:
        try:
            self._context.controller.set_publish_output(enabled)
        except Sha256Error as exc:
            # The sink refused. Say why — "no virtual camera installed" and "OBS
            # is holding the device" need different actions from the user — and
            # put the checkbox back where the controller actually is.
            _LOG.warning(
                "Virtual camera could not be switched %s: %s", "on" if enabled else "off", exc
            )
            QMessageBox.warning(self, "Virtual camera", str(exc))
            self._publish_check.blockSignals(True)
            self._publish_check.setChecked(self._context.controller.publish_output)
            self._publish_check.blockSignals(False)
        self._refresh_publish_state()

    def _refresh_publish_state(self) -> None:
        """Reflect what the sink can actually do, in the tooltip and the enabling."""
        sink = self._context.controller.components.virtual_camera

        if not sink.is_available:
            self._publish_check.setEnabled(False)
            self._publish_check.setToolTip(
                "No virtual camera device on this machine. Installing OBS Studio "
                "once registers one — OBS itself does not have to be running."
            )
            return

        self._publish_check.setEnabled(True)
        device = sink.device_name
        self._publish_check.setToolTip(
            f"Other applications see this as {device!r}."
            if device
            else "Show the avatar in Meet, Zoom, Discord and OBS as a webcam."
        )

    def _on_start(self) -> None:
        try:
            self._runner.start()
        except Exception as exc:
            _LOG.exception("Pipeline start failed")
            QMessageBox.critical(self, "Could not start", str(exc))
            self._refresh_status()
            return

        self._preview_timer.start()
        self._stack.setCurrentIndex(1)
        self._refresh_publish_state()
        self._refresh_status()

    def _on_stop(self) -> None:
        self._preview_timer.stop()
        self._runner.stop()
        self._preview.clear_frame()
        self._stack.setCurrentIndex(0)
        self._refresh_publish_state()
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
        self._stop_button.setEnabled(running)
        self._primary_button.setEnabled(not running)

        error = self._runner.last_error
        if error is not None:
            self._preview_timer.stop()
            self._fps_label.setText("")
            self._stack.setCurrentIndex(0)
            self._status_line.setText(f"Stopped: {error}")
            return

        self._fps_label.setText(f"{self._runner.fps:.1f} fps" if running else "")

        # Say when the face is not being found. On a loss the avatar is held
        # exactly where it was — the camera is never passed through, whatever
        # happens — and a frozen face over a moving background reads as the
        # avatar having vanished. It has not; the tracker simply cannot see
        # enough of a face turned that far. Saying so turns a defect into a
        # limit the user can work with.
        snapshot = self._runner.latest() if running else None
        tracking = snapshot.tracking if snapshot is not None else None
        if running and tracking is not None and not tracking.face_detected:
            self._status_line.setText("Face not found — holding the avatar")
        else:
            self._status_line.setText("")
        self._status_line.setText("")

        if not self._publish_check.isEnabled():
            # Poll only for the one transition that matters — the owner installs
            # a virtual camera while sha256 is open. Once it is available nothing
            # can change it except a toggle, which refreshes on its own.
            self._refresh_publish_state()

        if (
            self._context.settings.developer_mode
            and self._context.controller.state is PipelineState.RUNNING
        ):
            self._debug_panel.refresh()

    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        """Stop the pipeline before the window goes away."""
        self._preview_timer.stop()
        self._status_timer.stop()
        self._runner.stop()
        super().closeEvent(event)  # type: ignore[arg-type]
