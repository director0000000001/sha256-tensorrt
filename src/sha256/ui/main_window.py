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

The bar holds what a call needs at one click -- Start, Stop, the camera, the
avatar, the virtual camera, the black screen -- and a **Menu** with the rest
(day 16, 2026-09-12, his report: with fifteen controls in a row the bar set
a minimum width wider than the screen, so the window opened with its bottom
off the screen and could not be dragged smaller). The bar's minimum is the
window's minimum: nothing on it may ask for more width than it must.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QGuiApplication
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
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sha256.app.owner import OWNER_FILENAME, OwnerAccess
from sha256.avatar.importer import AvatarImportError, import_portrait, neural_template
from sha256.calibration import STEPS, CalibrationSession, DriverProfile, roster
from sha256.calibration.measure import calibrate, describe
from sha256.calibration.script import total_seconds
from sha256.core.enums import PipelineState
from sha256.core.errors import Sha256Error
from sha256.pipeline.conditioning import QUALITY_NAMES, OutputConditioner
from sha256.pipeline.runner import PipelineRunner
from sha256.tracking.mediapipe_tracker import DEFAULT_MODEL_RELATIVE_PATH, MediaPipeFaceTracker
from sha256.ui.background import BackgroundPanel
from sha256.ui.hotkey import GlobalHotkey
from sha256.ui.instructions_dialog import InstructionsDialog
from sha256.ui.panels.debug_panel import DebugPanel
from sha256.ui.photo_guide_dialog import PhotoGuideDialog
from sha256.ui.preview import PreviewWidget
from sha256.ui.speech import Speaker
from sha256.ui.widgets import ElidedLabel
from sha256.utils.logging import get_logger

if TYPE_CHECKING:
    from sha256.app.bootstrap import ApplicationContext

__all__ = ["MainWindow", "opening_geometry"]

_LOG = get_logger(__name__)

# The size the window opens at when the screen has the room. When it has
# not, the window takes what fits less the frame: the title bar above and
# the border below are the window manager's, not the window's, and Windows
# does not move a window up to keep its bottom on the screen -- it opened
# 780 tall on a screen with less than that free, and the bar was below the
# edge. The taskbar is already outside a screen's availableGeometry.
_PREFERRED_SIZE = QSize(1200, 780)
_FRAME_TOP = 36
_FRAME_MARGIN = QSize(40, 60)
_SMALLEST_OPENING = QSize(320, 240)
# A combo asks for room for this many characters, not for its longest
# entry: a camera called "Sony ZV-E10 (USB Streaming)" widens the list, not
# the bar.
_COMBO_CHARS = 8


def opening_geometry(available: QRect, preferred: QSize = _PREFERRED_SIZE) -> QRect:
    """Where the window opens and how large, given the screen's free area:
    the preferred size when it fits, else as large as fits under the frame;
    centred either way, and never above the title bar's room."""
    width = min(preferred.width(), available.width() - _FRAME_MARGIN.width())
    height = min(preferred.height(), available.height() - _FRAME_MARGIN.height())
    width = max(width, _SMALLEST_OPENING.width())
    height = max(height, _SMALLEST_OPENING.height())
    x = available.x() + max((available.width() - width) // 2, 0)
    spare = available.height() - height - _FRAME_MARGIN.height()
    y = available.y() + _FRAME_TOP + max(spare // 2, 0)
    return QRect(x, y, width, height)


def _compact(combo: QComboBox) -> None:
    """A combo that asks for a fixed width, whatever its longest entry. The
    list still opens at the entries' full width, and the chosen entry's full
    name is the tooltip, since the box cuts what does not fit."""
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(_COMBO_CHARS)
    combo.currentTextChanged.connect(combo.setToolTip)


# Painting faster than the pipeline produces frames only repaints the same
# picture. 33 ms matches a 30 fps source; the preview simply shows whatever the
# newest frame is when it wakes.
_PREVIEW_INTERVAL_MS = 33
_STATUS_INTERVAL_MS = 500
# The calibration's instruction and its countdown are redrawn this often.
_CALIBRATION_TICK_MS = 100
# The Person menu's entry for no profile at all: the avatar's own values.
_NO_PERSON = "Nobody (the avatar's own values)"

# The avatar view is first, and therefore the default. The tracking view exists
# to answer "is it following me"; the avatar view is the product, and defaulting
# to the diagnostic made the app look like it was doing nothing.
_VIEW_OUTPUT = "Avatar"
_VIEW_CAMERA = "Tracking (no avatar)"


def _quality_label(name: str) -> str:
    """The menu's entry for a conditioner quality name: "Native", "480p"."""
    return "Native" if name == "native" else name


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

QPushButton, QToolButton {
    color: #f2f2f6;
    background-color: rgba(38, 38, 50, 200);
    border: 1px solid rgba(255, 255, 255, 45);
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 14px;
}
QPushButton:hover, QToolButton:hover, QToolButton:pressed, QToolButton:open {
    background-color: rgba(58, 58, 76, 225);
}
QPushButton:disabled { color: #74747f; background-color: rgba(30, 30, 38, 110); }
QToolButton::menu-indicator { image: none; width: 0px; }

QMenu {
    color: #f2f2f6;
    background-color: #1b1b24;
    border: 1px solid rgba(255, 255, 255, 40);
    padding: 6px 0px;
    font-size: 14px;
}
QMenu::item { padding: 7px 28px 7px 12px; background: transparent; }
QMenu::item:selected { background-color: #38384a; }
QMenu::item:disabled { color: #74747f; }
QMenu::separator { height: 1px; background: rgba(255, 255, 255, 40); margin: 6px 10px; }
QMenu::indicator { width: 14px; height: 14px; margin-left: 6px; }
QMenu::right-arrow { margin-right: 8px; }

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
        # Day 16 evening, his ask: the people at this camera, by name (see
        # sha256.calibration.roster). The file in force is the truth --
        # the start laid it over the engine -- and one from before names
        # existed is adopted into the roster so the menu can show it.
        configs = context.paths.configs
        self._active_profile: DriverProfile | None = roster.adopt_unnamed(
            configs
        ) or roster.active_profile(configs)

        self.setWindowTitle("sha256")
        screen = QGuiApplication.primaryScreen()
        available = (
            screen.availableGeometry()
            if screen is not None
            else QRect(QPoint(0, 0), _PREFERRED_SIZE + _FRAME_MARGIN)
        )
        self.setGeometry(opening_geometry(available))

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
        layout.setSpacing(8)

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
        _compact(self._camera_combo)
        layout.addWidget(self._camera_combo)

        layout.addSpacing(6)

        layout.addWidget(QLabel("Avatar"))
        self._avatar_combo = QComboBox()
        _compact(self._avatar_combo)
        self._avatar_combo.currentIndexChanged.connect(self._on_avatar_changed)
        layout.addWidget(self._avatar_combo)

        layout.addSpacing(10)

        # The one control that decides whether anything outside this window can
        # see the avatar. It is a checkbox rather than an always-on behaviour
        # because appearing in every application's camera list unasked is not a
        # thing a user should have to discover after the fact.
        self._publish_check = QCheckBox("Virtual camera")
        self._publish_check.setChecked(self._context.controller.publish_output)
        self._publish_check.toggled.connect(self._on_publish_toggled)
        layout.addWidget(self._publish_check)

        # Day 15, his ask: the black screen. Pressed, the virtual camera shows
        # a black frame with the failure text instead of the avatar -- as if
        # the camera's battery died -- until pressed again. The pipeline also
        # switches it on by itself when it cannot work (see the conditioner);
        # the status line says which. On the bar, not in the menu: it is the
        # one control that must be a single click away in a bad moment.
        self._blackout_button = QPushButton(self._conditioner.failure_text)
        self._blackout_button.setObjectName("BlackoutButton")
        self._blackout_button.setCheckable(True)
        self._blackout_button.setChecked(self._conditioner.manual)
        self._blackout_button.toggled.connect(self._on_blackout_toggled)
        layout.addWidget(self._blackout_button)

        # Empty unless something went wrong, so the bar stays quiet in normal
        # use but a failure is never silent. It takes the room that is left
        # and cuts what does not fit: a long message must never widen the
        # window (see sha256.ui.widgets).
        self._status_line = ElidedLabel()
        self._status_line.setObjectName("StatusLine")
        layout.addWidget(self._status_line, 1)

        self._fps_label = QLabel()
        self._fps_label.setObjectName("Fps")
        layout.addWidget(self._fps_label)

        layout.addSpacing(6)
        layout.addWidget(self._build_menu_button())

        return bar

    def _build_menu_button(self) -> QToolButton:
        """The Menu: everything that is not needed at one click on a call.

        Day 16, his ask ("a dropdown menu"): the view, the dots, the mirror,
        the quality, "That looked wrong", "Re-centre", "Calibrate me", and
        the owner's Login / Add avatar. Each is a QAction, so the window's
        handlers and the tests drive the same objects the menu shows.
        """
        button = QToolButton()
        button.setObjectName("MenuButton")
        button.setText("Menu ▾")
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        menu = QMenu(button)
        menu.setToolTipsVisible(True)
        button.setMenu(menu)
        self._menu_button = button
        self._menu = menu

        # The view: the avatar (the product) or the camera with its tracking
        # (the answer to "is it following me"). Exclusive, the avatar first.
        view = menu.addMenu("View")
        self._view_group = QActionGroup(view)
        self._view_group.setExclusive(True)
        self._view_avatar_action = self._exclusive(view, self._view_group, _VIEW_OUTPUT, "avatar")
        self._view_camera_action = self._exclusive(view, self._view_group, _VIEW_CAMERA, "camera")
        self._view_avatar_action.setChecked(True)

        self._landmarks_action = QAction("Dots", menu)
        self._landmarks_action.setCheckable(True)
        self._landmarks_action.setChecked(True)
        self._landmarks_action.setToolTip("Draw the tracked landmarks. Tracking view only.")
        self._landmarks_action.toggled.connect(self._preview.set_show_landmarks)
        menu.addAction(self._landmarks_action)

        self._mirror_action = QAction("Mirror", menu)
        self._mirror_action.setCheckable(True)
        self._mirror_action.setChecked(self._context.settings.camera.mirror)
        self._mirror_action.setToolTip("Affects this preview only, not the published output.")
        self._mirror_action.toggled.connect(self._preview.set_mirror)
        self._preview.set_mirror(self._mirror_action.isChecked())
        menu.addAction(self._mirror_action)

        # Day 15, his ask: the published picture scaled down and back up, so
        # the other side sees a cheaper webcam rather than the avatar's small
        # defects. Native is the camera's own size; the stream's format never
        # changes whatever is chosen (see sha256.pipeline.conditioning).
        quality = menu.addMenu("Quality")
        quality.setToolTipsVisible(True)
        quality.menuAction().setToolTip(
            "Quality of the published picture. Lower hides small defects of the "
            "avatar, like a cheaper webcam would. Native is the camera's own size."
        )
        self._quality_group = QActionGroup(quality)
        self._quality_group.setExclusive(True)
        self._quality_actions: dict[str, QAction] = {}
        for name in QUALITY_NAMES:
            action = self._exclusive(quality, self._quality_group, _quality_label(name), name)
            self._quality_actions[name] = action
        self._quality_actions[self._conditioner.quality].setChecked(True)
        self._quality_group.triggered.connect(
            lambda action: self._on_quality_changed(str(action.data()))
        )

        menu.addSeparator()

        # "That looked wrong." Saves the last ten seconds of tracking and the
        # frame on screen, so a complaint about the avatar can be replayed
        # instead of reproduced from a description.
        self._capture_action = QAction("That looked wrong", menu)
        self._capture_action.setToolTip(
            "Save the last ten seconds of tracking and the frame on screen, "
            "so the moment can be replayed."
        )
        self._capture_action.triggered.connect(self._on_capture)
        menu.addAction(self._capture_action)

        # An engine that learns a resting face from the first frames after
        # load gets it wrong if the owner was not sitting still and frontal
        # right then (day 5, 2026-09-05: a call reached for the mouse and the
        # avatar sat visibly off-centre for the rest of the session). This is
        # the one knob that says "measure it again" -- see AvatarEngine.reset_anchor.
        self._recentre_action = QAction("Re-centre", menu)
        self._recentre_action.setToolTip(
            "Learn the resting face again -- look straight at the camera for a second."
        )
        self._recentre_action.triggered.connect(self._on_recentre)
        menu.addAction(self._recentre_action)

        # Day 16 evening, his ask: who is at the camera. One entry per named
        # profile (Calibrate me asks the name at its end) and one for none
        # -- the avatar's own values. Choosing one lays it over the engine
        # and reloads the avatar, as the end of a calibration does.
        self._person_menu = menu.addMenu("Person")
        self._person_menu.setToolTipsVisible(True)
        self._person_menu.menuAction().setToolTip(
            "Whose profile the avatar follows. Calibrate me adds you to this list; "
            "choose your name when you sit down."
        )
        self._person_group = QActionGroup(self._person_menu)
        self._person_group.setExclusive(True)
        self._person_group.triggered.connect(
            lambda action: self._on_person_chosen(str(action.data()) or None)
        )
        self._rebuild_person_menu()

        # Day 16: "Calibrate me" -- a guided minute and a half measured by
        # the engine's own instruments into a driver profile (see
        # sha256.calibration), and a recording of the session for the
        # offline measurement. For anyone, not the owner alone: each of the
        # five runs it once at their own camera.
        self._calibrate_action = QAction("Calibrate me", menu)
        self._calibrate_action.setToolTip(
            f"A guided {total_seconds(STEPS):.0f} seconds -- sit still, blink, talk, smile, "
            "raise the brows, glance, turn -- measured into your own profile, so the "
            "avatar follows your face and not the owner's. Records the session on this "
            "computer; choose again to stop."
        )
        self._calibrate_action.triggered.connect(self._on_calibrate)
        menu.addAction(self._calibrate_action)

        menu.addSeparator()

        # Day 15, his ask: new faces are added from the app, by him alone.
        # "Login" takes his passphrase (and sets it the first time); only
        # then does "Add avatar" appear -- a photograph in, an avatar out
        # (see sha256.avatar.importer for what is checked on the way).
        self._login_action = QAction("Login", menu)
        self._login_action.setToolTip("The owner's passphrase; unlocks adding avatars.")
        self._login_action.triggered.connect(self._on_login)
        menu.addAction(self._login_action)

        self._add_avatar_action = QAction("Add avatar", menu)
        self._add_avatar_action.setToolTip(
            "Make an avatar from a photograph: one face, looking straight at the "
            "camera, eyes open, lips closed or barely parted, room below the chin."
        )
        self._add_avatar_action.triggered.connect(self._on_add_avatar)
        self._add_avatar_action.setVisible(False)
        menu.addAction(self._add_avatar_action)

        menu.addSeparator()

        # Day 16 night, his ask: the whole of it in text, inside the app --
        # what the computer and the camera must be, the first run, every
        # control, the calls (see sha256.app.instructions).
        self._instructions_action = QAction("Instructions", menu)
        self._instructions_action.setToolTip(
            "What you need, the first run, every button, the calls, and what to do "
            "when something goes wrong."
        )
        self._instructions_action.triggered.connect(self._on_instructions)
        menu.addAction(self._instructions_action)

        return button

    @staticmethod
    def _exclusive(menu: QMenu, group: QActionGroup, text: str, data: str) -> QAction:
        """One checkable entry of an exclusive choice, carrying its value."""
        action = QAction(text, menu)
        action.setCheckable(True)
        action.setData(data)
        group.addAction(action)
        menu.addAction(action)
        return action

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
        self._add_avatar_action.setVisible(unlocked)
        self._login_action.setText("Owner" if unlocked else "Login")
        self._login_action.setEnabled(not unlocked)

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

    def _on_instructions(self) -> None:
        InstructionsDialog(self).exec()

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
        self._calibrate_action.setText("Stop calibrating")
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
        self._calibrate_action.setText("Calibrate me")
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
        configs = self._context.paths.configs
        try:
            session_name = str(files.directory.relative_to(self._context.paths.root))
        except ValueError:
            session_name = str(files.directory)
        # Day 16 evening, his ask: the profile is the person's, by name.
        # A file in force from before names existed is kept under a name
        # of its own; the same person calibrating again keeps, for what
        # this run could not measure, what their earlier run did.
        self._end_calibration_ui()
        name = self._ask_person_name()
        roster.adopt_unnamed(configs)
        earlier = roster.find_profile(configs, name)
        profile = DriverProfile.from_result(result, avatar=active, session=session_name, name=name)
        if earlier is not None:
            profile.engine_options = {**earlier.overlay(), **profile.engine_options}
        path = roster.save_named(profile, configs)
        profile.save(files.directory / "profile.json")
        applied = self._use_profile(profile)
        QMessageBox.information(
            self,
            "Calibrate me",
            f"The profile of {name} is saved ({path}) and in use now -- "
            f"Menu → Person shows it, choose it when you sit down.\n\n{report}\n\n{where}"
            + ("" if applied else "\n\nIt was not applied; restart to use it."),
        )

    def _ask_person_name(self) -> str:
        """Whose profile this is -- typed at the end of a calibration. Left
        blank or cancelled, the profile is still kept, under the moment."""
        text, ok = QInputDialog.getText(
            self,
            "Calibrate me",
            "Your name, first and last. The profile is kept under it, and you "
            "choose it from Menu → Person when you sit down:",
            QLineEdit.EchoMode.Normal,
            "",
        )
        name = text.strip() if ok else ""
        return name or f"Person {time.strftime('%Y-%m-%d %H:%M')}"

    # --- Person (day 16 evening) --------------------------------------------

    def _rebuild_person_menu(self) -> None:
        """The roster into the Person submenu: Nobody, then every named
        profile, the one in force checked (added even if its roster file
        is gone, so the check always shows the truth)."""
        menu, group = self._person_menu, self._person_group
        for action in list(group.actions()):
            group.removeAction(action)
            menu.removeAction(action)
        active = self._active_profile.name if self._active_profile is not None else ""
        nobody = self._exclusive(menu, group, _NO_PERSON, "")
        nobody.setToolTip("No profile: the avatar runs with its own values.")
        nobody.setChecked(not active)
        names: list[str] = []
        for profile in roster.list_profiles(self._context.paths.configs):
            names.append(profile.name)
            action = self._exclusive(menu, group, profile.name, profile.name)
            action.setToolTip(f"Measured {profile.measured_at[:16].replace('T', ' ')}")
            action.setChecked(profile.name == active)
        if active and active not in names:
            action = self._exclusive(menu, group, active, active)
            action.setChecked(True)

    def _on_person_chosen(self, name: str | None) -> None:
        """A Person entry chosen: that profile (or none) becomes the one in
        force, on the engine and in the file the start reads."""
        if self._calibration is not None:
            QMessageBox.information(self, "Person", "Finish or stop the calibration first.")
            self._rebuild_person_menu()
            return
        configs = self._context.paths.configs
        profile = None
        if name is not None:
            profile = roster.find_profile(configs, name)
            if profile is None and self._active_profile is not None:
                active = self._active_profile
                profile = active if active.name == name else None
            if profile is None:
                QMessageBox.warning(
                    self,
                    "Person",
                    f"The profile of {name} is missing from {roster.drivers_dir(configs)}.",
                )
                self._rebuild_person_menu()
                return
        self._use_profile(profile)

    def _use_profile(self, profile: DriverProfile | None) -> bool:
        """Make ``profile`` (or none) the one in force: the file the start
        reads, the engine, the avatar reloaded under it. ``True`` when the
        engine took it and the avatar runs with it now."""
        roster.activate(profile, self._context.paths.configs)
        self._active_profile = profile
        self._context.driver_profile = profile
        self._rebuild_person_menu()
        engine = self._context.avatars.engine
        if engine is None or not hasattr(engine, "set_driver_options"):
            # The photograph engine takes no profile; the file waits for
            # the neural one.
            self._refresh_status()
            return True
        overlay = {} if profile is None else profile.overlay()
        return self._apply_driver_overlay(engine, self._context.avatars.active_id, overlay)

    def _apply_driver_overlay(
        self, engine: Any, avatar_id: str | None, overlay: dict[str, Any]
    ) -> bool:
        """Lay the driver options over the engine and reload the avatar
        under them -- the graph bakes the options at load. The pipeline is
        stopped for the reload (a frame must not meet an unloaded avatar)
        and started again. Without an avatar selected, the engine keeps
        the options for the next load."""
        try:
            engine.set_driver_options(overlay)
        except ValueError as exc:
            _LOG.warning("The profile was refused by the engine: %s", exc)
            return False
        if avatar_id is None:
            self._refresh_status()
            return True
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
            QMessageBox.warning(self, "Person", f"The avatar did not reload: {exc}")
            ok = False
        if was_running:
            try:
                self._runner.start()
                self._preview_timer.start()
            except Exception as exc:
                _LOG.exception("The pipeline did not restart under the profile")
                QMessageBox.critical(self, "Could not start", str(exc))
                ok = False
        self._refresh_status()
        return ok

    def _profile_note(self) -> str:
        """What the status line says when nothing else needs saying: whose
        profile the avatar follows -- his own answer to "did it work"."""
        if self._active_profile is None or not self._active_profile.name:
            return ""
        return f"Profile: {self._active_profile.name}"

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

    def _on_quality_changed(self, name: str) -> None:
        try:
            self._conditioner.set_quality(name)
        except ValueError as exc:
            _LOG.warning("Output quality not changed: %s", exc)
            self._quality_actions[self._conditioner.quality].setChecked(True)

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

        if self._view_camera_action.isChecked() and snapshot.source is not None:
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
            self._status_line.setText(rest or self._profile_note())

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
