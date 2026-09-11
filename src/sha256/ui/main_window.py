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

import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sha256.app.owner import OWNER_FILENAME, OwnerAccess
from sha256.avatar.importer import AvatarImportError, import_portrait, neural_template
from sha256.calibration import PROFILE_FILENAME, STEPS, CalibrationSession, DriverProfile
from sha256.calibration.measure import calibrate, describe
from sha256.calibration.script import total_seconds
from sha256.core.enums import PipelineState
from sha256.core.errors import Sha256Error
from sha256.pipeline.conditioning import QUALITY_NAMES, OutputConditioner
from sha256.pipeline.runner import PipelineRunner
from sha256.tracking.mediapipe_tracker import DEFAULT_MODEL_RELATIVE_PATH, MediaPipeFaceTracker
from sha256.ui.background import BackgroundPanel
from sha256.ui.hotkey import GlobalHotkey
from sha256.ui.panels.debug_panel import DebugPanel
from sha256.ui.photo_guide_dialog import PhotoGuideDialog
from sha256.ui.preview import PreviewWidget
from sha256.ui.speech import Speaker
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
# The calibration's instruction and its countdown are redrawn this often.
_CALIBRATION_TICK_MS = 100

# The avatar view is first, and therefore the default. The tracking view exists
# to answer "is it following me"; the avatar view is the product, and defaulting
# to the diagnostic made the app look like it was doing nothing.
_VIEW_OUTPUT = "Avatar"
_VIEW_CAMERA = "Tracking (no avatar)"


def _quality_label(name: str) -> str:
    """The combo's entry for a conditioner quality name: "Native", "480p"."""
    return "Native" if name == "native" else name


def _quality_name(label: str) -> str:
    return "native" if label == "Native" else label


# Everything is translucent so the background photograph stays visible through
# it. Surfaces are dark enough that light text on them is readable over a photo
# of unknown brightness — which the window cannot know in advance.
_STYLE = """
QLabel { color: #f2f2f6; background: transparent; }
QCheckBox { color: #e2e2ea; background: transparent; }

#StatusLine { color: #fca5a5; font-size: 13px; }
#Instruction {
    color: #ffffff;
    font-size: 26px;
    font-weight: 600;
    background-color: rgba(12, 12, 18, 200);
    border: 1px solid rgba(255, 255, 255, 40);
    border-radius: 12px;
    padding: 14px 24px;
}
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

#BlackoutButton:checked {
    color: #fff1f1;
    background-color: rgba(176, 44, 44, 230);
    border: 1px solid rgba(255, 180, 180, 120);
}
#BlackoutButton:checked:hover { background-color: rgba(200, 60, 60, 245); }

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

    # Day 16 (D1): the kill switch reports from its own thread; the
    # signal carries its word onto this one.
    switched_off = Signal(str)

    def __init__(self, context: ApplicationContext) -> None:
        super().__init__()
        self._context = context
        self._runner = PipelineRunner(context.controller)
        # Day 15: his passphrase, and what it unlocks (see sha256.app.owner).
        self._owner = OwnerAccess(context.paths.configs / OWNER_FILENAME)

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

        # Day 16: "Calibrate me" -- the running session, if one is on, and
        # the timer that keeps its instruction and countdown on screen.
        self._calibration: CalibrationSession | None = None
        # The instructions read aloud where a voice exists (see ui.speech);
        # made when a calibration starts, replaced in tests.
        self._speaker: Speaker | None = None
        self._speaker_factory: Callable[[], Speaker] = Speaker
        self._spoken_step = -1
        self._calibration_timer = QTimer(self)
        self._calibration_timer.setInterval(_CALIBRATION_TICK_MS)
        self._calibration_timer.timeout.connect(self._tick_calibration)

        self._populate_avatars()
        self._populate_cameras()
        self._refresh_publish_state()
        self._refresh_owner_state()

        # The black screen's hotkey works with the meeting app in front --
        # that is the whole point of it (see sha256.ui.hotkey). Bound now,
        # released in closeEvent; if Windows refuses the key the button's
        # tooltip says so and the button still works.
        self._hotkey = GlobalHotkey(
            self, context.settings.output.blackout_hotkey, self._toggle_blackout
        )
        self._hotkey.register()
        self._refresh_blackout_tooltip()
        # Day 16 (D1/D2): the courtesy switch, and what the start had to say.
        self._switched_off = False
        self.switched_off.connect(self._on_switched_off)
        switch = context.kill_switch
        if switch is not None and switch.configured:
            switch._on_state = lambda state: (
                self.switched_off.emit(state.message) if not state.allowed else None
            )
            switch.start()
        if context.startup_notes:
            QTimer.singleShot(0, self._show_startup_notes)
        self._refresh_status()

    @property
    def _conditioner(self) -> OutputConditioner:
        return self._context.controller.components.conditioner

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

        # Day 16: the calibration's instruction, over the video and above
        # the bar, large enough to read from where a person sits. Hidden
        # except while a calibration runs.
        self._instruction = QLabel()
        self._instruction.setObjectName("Instruction")
        self._instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._instruction.setWordWrap(True)
        self._instruction.hide()
        layout.addWidget(self._instruction)

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

        # Day 15, his ask: the published picture scaled down and back up, so
        # the other side sees a cheaper webcam rather than the avatar's small
        # defects. Native is the camera's own size; the stream's format never
        # changes whatever is chosen (see sha256.pipeline.conditioning).
        self._quality_combo = QComboBox()
        self._quality_combo.addItems([_quality_label(name) for name in QUALITY_NAMES])
        self._quality_combo.setCurrentText(_quality_label(self._conditioner.quality))
        self._quality_combo.setToolTip(
            "Quality of the published picture. Lower hides small defects of the "
            "avatar, like a cheaper webcam would. Native is the camera's own size."
        )
        self._quality_combo.currentTextChanged.connect(self._on_quality_changed)
        layout.addWidget(self._quality_combo)

        # Day 15, his ask: the black screen. Pressed, the virtual camera shows
        # a black frame with the failure text instead of the avatar -- as if
        # the camera's battery died -- until pressed again. The pipeline also
        # switches it on by itself when it cannot work (see the conditioner);
        # the status line says which.
        self._blackout_button = QPushButton(self._conditioner.failure_text)
        self._blackout_button.setObjectName("BlackoutButton")
        self._blackout_button.setCheckable(True)
        self._blackout_button.setChecked(self._conditioner.manual)
        self._blackout_button.toggled.connect(self._on_blackout_toggled)
        layout.addWidget(self._blackout_button)

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

        # An engine that learns a resting face from the first frames after
        # load gets it wrong if the owner was not sitting still and frontal
        # right then (day 5, 2026-09-05: a call reached for the mouse and the
        # avatar sat visibly off-centre for the rest of the session). This is
        # the one knob that says "measure it again" -- see AvatarEngine.reset_anchor.
        self._recentre_button = QPushButton("Re-centre")
        self._recentre_button.setToolTip(
            "Learn the resting face again — look straight at the camera for a second."
        )
        self._recentre_button.clicked.connect(self._on_recentre)
        layout.addWidget(self._recentre_button)

        # Day 16: "Calibrate me" -- a guided three-quarters of a minute
        # measured by the engine's own instruments into a driver profile
        # (see sha256.calibration), and a recording of the session for the
        # offline measurement. For anyone, not the owner alone: each of the
        # five runs it once at their own camera.
        self._calibrate_button = QPushButton("Calibrate me")
        self._calibrate_button.setToolTip(
            f"A guided {total_seconds(STEPS):.0f} seconds -- sit still, blink, talk, smile, "
            "raise the brows, glance, turn -- measured into your own profile, so the "
            "avatar follows your face and not the owner's. Records the session on this "
            "computer; press again to stop."
        )
        self._calibrate_button.clicked.connect(self._on_calibrate)
        layout.addWidget(self._calibrate_button)

        layout.addSpacing(10)

        # Day 15, his ask: new faces are added from the app, by him alone.
        # "Login" takes his passphrase (and sets it the first time); only
        # then does "Add avatar" appear -- a photograph in, an avatar out
        # (see sha256.avatar.importer for what is checked on the way).
        self._login_button = QPushButton("Login")
        self._login_button.setToolTip("The owner's passphrase; unlocks adding avatars.")
        self._login_button.clicked.connect(self._on_login)
        layout.addWidget(self._login_button)

        self._add_avatar_button = QPushButton("Add avatar")
        self._add_avatar_button.setToolTip(
            "Make an avatar from a photograph: one face, looking straight at the "
            "camera, eyes open, lips closed or barely parted, room below the chin."
        )
        self._add_avatar_button.clicked.connect(self._on_add_avatar)
        self._add_avatar_button.setVisible(False)
        layout.addWidget(self._add_avatar_button)

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
        and expression readings, not a photograph — plus the frame that was on
        screen and, since day 12, the camera frame it came from, which is a
        photograph of the person pressing the button. They are entitled to know
        which file is which and where all of them landed.
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
            "tracking.npz — landmark and expression readings and the avatar's "
            "resting anchor, no imagery\n"
            "seen.png — the frame that was on screen (the avatar)\n"
            "camera.png — the camera frame at the same instant (your face)",
        )

    def _refresh_owner_state(self) -> None:
        unlocked = self._owner.is_unlocked
        self._add_avatar_button.setVisible(unlocked)
        self._login_button.setText("Owner" if unlocked else "Login")
        self._login_button.setEnabled(not unlocked)

    def _ask_passphrase(self, title: str, prompt: str) -> str | None:
        text, ok = QInputDialog.getText(self, title, prompt, QLineEdit.EchoMode.Password)
        return text if ok else None

    def _on_login(self) -> None:
        """His passphrase -- set the first time, asked for after."""
        if not self._owner.is_configured:
            first = self._ask_passphrase(
                "Set the owner passphrase",
                "No passphrase is set yet. Choose one (at least 4 characters):",
            )
            if first is None:
                return
            again = self._ask_passphrase("Set the owner passphrase", "Type it once more:")
            if again is None:
                return
            if first != again:
                QMessageBox.warning(self, "Owner", "The two entries differ; nothing was set.")
                return
            try:
                self._owner.set_passphrase(first)
            except ValueError as exc:
                QMessageBox.warning(self, "Owner", str(exc))
                return
            QMessageBox.information(
                self, "Owner", f"Passphrase set and kept in {self._owner.path.name}."
            )
        else:
            entered = self._ask_passphrase("Login", "The owner's passphrase:")
            if entered is None:
                return
            if not self._owner.unlock(entered):
                QMessageBox.warning(self, "Login", "That is not the passphrase.")
                return
        self._refresh_owner_state()

    def _tracker_model_path(self) -> Path:
        settings = self._context.settings
        configured = settings.tracking.model_path
        if configured:
            return self._context.paths.resolve_relative(configured)
        models_root = self._context.paths.resolve_relative(settings.models.root_path)
        return models_root / DEFAULT_MODEL_RELATIVE_PATH

    def _on_add_avatar(self) -> None:
        """A photograph in, an avatar in the library out -- checked first.
        The guide comes first (day 16, his ask): what the photograph must be
        and the prompts that make one, then the file dialog."""
        if not self._owner.is_unlocked:
            return
        if PhotoGuideDialog(self).exec() != int(PhotoGuideDialog.DialogCode.Accepted):
            return
        photo, _ = QFileDialog.getOpenFileName(
            self, "Choose a photograph", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if not photo:
            return
        default = Path(photo).stem.replace("_", " ").replace("-", " ").strip() or "New avatar"
        name, ok = QInputDialog.getText(self, "Add avatar", "Name for the avatar:", text=default)
        if not ok or not name.strip():
            return
        manager = self._context.avatars
        template = neural_template(manager.list_avatars(), manager.active_id)
        try:
            descriptor, report = import_portrait(
                self._context.paths.avatars,
                Path(photo),
                name=name,
                model_path=self._tracker_model_path(),
                template=template,
            )
        except (AvatarImportError, Sha256Error) as exc:
            _LOG.warning("Avatar not added: %s", exc)
            QMessageBox.warning(self, "Add avatar", str(exc))
            return
        except Exception as exc:
            _LOG.exception("Avatar import failed")
            QMessageBox.critical(self, "Add avatar", f"The photograph could not be added: {exc}")
            return
        manager.refresh()
        self._populate_avatars()
        index = self._avatar_combo.findData(descriptor.id)
        if index >= 0:
            self._avatar_combo.setCurrentIndex(index)  # loads it, like picking it by hand
        QMessageBox.information(
            self,
            "Add avatar",
            f"{descriptor.manifest.name} added as {descriptor.root}\n\n{report.describe()}",
        )

    # --- Calibrate me (day 16) ---------------------------------------------

    def _calibration_engine(self) -> Any | None:
        """The engine, when it can be calibrated -- one with the neural
        engine's per-frame readings; the photograph engine has none."""
        engine = self._context.avatars.engine
        return engine if engine is not None and hasattr(engine, "frame_readings") else None

    def _output_tracker(self) -> MediaPipeFaceTracker:
        """An unfiltered tracker for reading the avatar's face back off the
        composited output, the way scripts/measure_follow.py does."""
        tracking = self._context.settings.tracking
        return MediaPipeFaceTracker(
            model_path=self._tracker_model_path(),
            min_detection_confidence=tracking.min_detection_confidence,
            min_presence_confidence=tracking.min_presence_confidence,
            min_tracking_confidence=tracking.min_tracking_confidence,
            smoothing=0.0,
            mouth_smoothing=0.0,
        )

    def _on_calibrate(self) -> None:
        """Start a calibration, or stop the one running."""
        if self._calibration is not None:
            self._cancel_calibration()
            return
        engine = self._calibration_engine()
        active = self._context.avatars.active_id
        if not self._runner.is_running:
            QMessageBox.information(
                self, "Calibrate me", "Press Start first, so the avatar is running on your face."
            )
            return
        if engine is None or active is None:
            QMessageBox.information(
                self,
                "Calibrate me",
                "Calibrate me needs the neural engine and an avatar selected.",
            )
            return
        recordings = self._context.paths.root / "recordings"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        directory = recordings / f"calibration-{stamp}"
        answer = QMessageBox.question(
            self,
            "Calibrate me",
            f"About {total_seconds(STEPS):.0f} seconds of instructions: sit still, blink, "
            "open your mouth, talk, smile, raise your eyebrows, glance, turn your head.\n\n"
            "The session is RECORDED -- your camera frames and their tracking -- into\n"
            f"{directory}\n"
            "on this computer only, so your face can be re-measured later without you. "
            "You can delete that folder at any time.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        session = CalibrationSession(
            self._runner,
            engine=engine,
            avatar_id=active,
            directory=directory,
            output_tracker_factory=self._output_tracker,
            fps=self._context.settings.camera.target_fps,
        )
        try:
            session.start()
        except Exception as exc:
            _LOG.exception("Calibration could not start")
            QMessageBox.critical(self, "Calibrate me", f"Could not start: {exc}")
            session.cancel()
            return
        self._calibration = session
        self._spoken_step = -1
        if self._speaker is None:
            self._speaker = self._speaker_factory()
        self._calibrate_button.setText("Stop calibrating")
        self._instruction.show()
        self._calibration_timer.start()
        self._tick_calibration()

    def _tick_calibration(self) -> None:
        session = self._calibration
        if session is None:
            return
        if session.finished:
            self._finish_calibration()
            return
        placed = session.current()
        if placed is None:
            return
        index, step, left, reading = placed
        if index != self._spoken_step:
            self._spoken_step = index
            if self._speaker is not None:
                self._speaker.say(step.instruction)
        phase = "Get ready..." if reading else "NOW"
        self._instruction.setText(
            f"{index + 1} / {len(session.steps)}\n{step.instruction}\n{phase}  {left:.0f}"
        )

    def _end_calibration_ui(self) -> None:
        self._calibration = None
        self._calibration_timer.stop()
        if self._speaker is not None:
            self._speaker.stop()
        self._instruction.hide()
        self._calibrate_button.setText("Calibrate me")
        self._refresh_status()

    def _cancel_calibration(self) -> None:
        """Stop, and remove what was recorded -- nothing of the person stays."""
        session = self._calibration
        self._end_calibration_ui()
        if session is not None:
            session.cancel()

    def _current_driver_options(self, avatar_id: str, engine: Any) -> dict[str, Any]:
        """What the avatar runs with now: its own options with the engine's
        driver profile over them -- the gains are corrected from these."""
        current: dict[str, Any] = {}
        try:
            for candidate in self._context.avatars.list_avatars():
                if candidate.id == avatar_id:
                    current.update(candidate.manifest.engine_options)
        except Sha256Error:
            pass
        current.update(getattr(engine, "driver_options", {}) or {})
        return current

    def _finish_calibration(self) -> None:
        """The steps are over: measure, write the profile, apply it."""
        session = self._calibration
        if session is None:
            return
        self._calibration_timer.stop()
        self._instruction.setText("Measuring...")
        engine = self._calibration_engine()
        active = self._context.avatars.active_id
        try:
            files = session.finish()
        except Exception as exc:
            _LOG.exception("Calibration could not finish")
            QMessageBox.critical(self, "Calibrate me", f"The session could not be saved: {exc}")
            self._end_calibration_ui()
            return
        if engine is None or active is None:
            self._end_calibration_ui()
            return
        current = self._current_driver_options(active, engine)
        result = calibrate(session.samples, current)
        report = describe(result, current=current)
        (files.directory / "report.txt").write_text(report + "\n", encoding="utf-8")
        where = (
            f"The session ({files.frames_written} frames of your face) is at\n"
            f"{files.directory}\n-- this computer only; delete it when it is no longer needed."
        )
        if not result.ok:
            self._end_calibration_ui()
            QMessageBox.warning(
                self, "Calibrate me", f"Nothing could be measured.\n\n{report}\n\n{where}"
            )
            return
        path = self._context.paths.configs / PROFILE_FILENAME
        try:
            session_name = str(files.directory.relative_to(self._context.paths.root))
        except ValueError:
            session_name = str(files.directory)
        profile = DriverProfile.from_result(result, avatar=active, session=session_name)
        if path.is_file():
            # The last person's profile is kept beside the new one.
            previous = DriverProfile.load(path)
            stamp = (previous.measured_at if previous else "").replace(":", "") or "previous"
            shutil.copy(path, path.with_name(f"local.driver.{stamp}.json"))
        profile.save(path)
        profile.save(files.directory / "profile.json")
        self._end_calibration_ui()
        applied = self._apply_driver_profile(engine, active, profile)
        QMessageBox.information(
            self,
            "Calibrate me",
            f"Your profile is written to {path}\n\n{report}\n\n{where}"
            + ("" if applied else "\n\nIt was not applied; restart to use it."),
        )

    def _apply_driver_profile(self, engine: Any, avatar_id: str, profile: DriverProfile) -> bool:
        """Lay the profile over the engine and reload the avatar under it --
        the graph bakes the options at load. The pipeline is stopped for the
        reload (a frame must not meet an unloaded avatar) and started again."""
        try:
            engine.set_driver_options(profile.overlay())
        except ValueError as exc:
            _LOG.warning("The profile was refused by the engine: %s", exc)
            return False
        was_running = self._runner.is_running
        if was_running:
            self._preview_timer.stop()
            self._runner.stop()
        manager = self._context.avatars
        ok = True
        try:
            manager.unload(avatar_id)
            manager.select(avatar_id)
        except Sha256Error as exc:
            _LOG.warning("The avatar could not be reloaded with the profile: %s", exc)
            QMessageBox.warning(self, "Calibrate me", f"The avatar did not reload: {exc}")
            ok = False
        if was_running:
            try:
                self._runner.start()
                self._preview_timer.start()
            except Exception as exc:
                _LOG.exception("The pipeline did not restart after the calibration")
                QMessageBox.critical(self, "Could not start", str(exc))
                ok = False
        self._refresh_status()
        return ok

    def _show_startup_notes(self) -> None:
        QMessageBox.information(self, "sha256", "\n\n".join(self._context.startup_notes))

    def _on_switched_off(self, message: str) -> None:
        """The owner's switch says stop: the pipeline stops, Start is
        disabled, and the message is shown once."""
        if self._switched_off:
            return
        self._switched_off = True
        if self._calibration is not None:
            self._cancel_calibration()
        self._on_stop()
        self._primary_button.setEnabled(False)
        self._status_line.setText(message)
        QMessageBox.information(self, "sha256", message)

    def _on_recentre(self) -> None:
        """Learn the resting face again, for whichever avatar is active.

        A no-op for an engine (or an avatar) with no learned rest — see
        AvatarEngine.reset_anchor — so this is safe to press at any time,
        including before anything is loaded.
        """
        engine = self._context.avatars.engine
        active = self._context.avatars.active_id
        if engine is not None and active is not None:
            engine.reset_anchor(active)

    def _on_quality_changed(self, label: str) -> None:
        try:
            self._conditioner.set_quality(_quality_name(label))
        except ValueError as exc:
            _LOG.warning("Output quality not changed: %s", exc)

    def _on_blackout_toggled(self, on: bool) -> None:
        self._conditioner.set_manual(on)
        self._refresh_status()

    def _toggle_blackout(self) -> None:
        """The hotkey: the same as pressing the button, so the two never
        disagree about what is on the virtual camera."""
        self._blackout_button.toggle()

    def _refresh_blackout_tooltip(self) -> None:
        text = self._conditioner.failure_text
        lines = [
            f"Black out the virtual camera with the words “{text}”, as if the "
            "camera's battery died. Press again to come back.",
        ]
        if self._hotkey.is_registered:
            lines.append(f"Hotkey {self._hotkey.spec}, from any application.")
        elif self._hotkey.error:
            lines.append(f"Hotkey not available: {self._hotkey.error}.")
        self._blackout_button.setToolTip("\n".join(lines))

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
            # against a picture they do not correspond to. The black screen
            # carries words, and mirrored words read backwards: that frame is
            # shown as the other side sees it, whatever the mirror is set to.
            unmirrored = self._conditioner.blackout is not None
            self._preview.show_frame(
                snapshot.frame.data,
                snapshot.frame.pixel_format,
                None,
                mirror=False if unmirrored else None,
            )

    def _refresh_status(self) -> None:
        running = self._runner.is_running
        self._stop_button.setEnabled(running)
        self._primary_button.setEnabled(not running and not self._switched_off)

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
            # Once a face is found, an engine still learning its rest (see
            # AvatarEngine.rest_status) gets to say so — the owner then knows
            # to sit still and frontal rather than wonder why the avatar looks
            # off, for the second or so that takes.
            rest = None
            engine = self._context.avatars.engine
            active = self._context.avatars.active_id
            if running and engine is not None and active is not None:
                rest = engine.rest_status(active)
            if self._calibration is not None:
                rest = "Calibrating -- follow the instruction above"
            self._status_line.setText(rest or "")

        # The black screen outranks the rest of the line: while it is on, the
        # one thing to know is that the other side is not seeing the avatar,
        # and whether that was his hand or the pipeline's.
        blackout = self._conditioner.blackout
        if blackout is not None and (running or blackout == "manual"):
            why = "by the button" if blackout == "manual" else blackout
            self._status_line.setText(
                f"BLACK SCREEN — the other side sees “{self._conditioner.failure_text}” ({why})"
            )
        if self._blackout_button.isChecked() != self._conditioner.manual:
            self._blackout_button.blockSignals(True)
            self._blackout_button.setChecked(self._conditioner.manual)
            self._blackout_button.blockSignals(False)

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
        self._hotkey.unregister()
        if self._context.kill_switch is not None:
            self._context.kill_switch.stop()
        if self._calibration is not None:
            # A half-done recording of the person is not kept.
            self._cancel_calibration()
        self._runner.stop()
        super().closeEvent(event)  # type: ignore[arg-type]
