"""Tests for the main window's status line.

The line has one job while the pipeline runs: say when the face is not being
found, because on a loss the avatar is held in place and a frozen face over a
moving background reads as the avatar having vanished. A version of
``_refresh_status`` set that text and then cleared it on the next line, so the
message never reached the screen; this is the test that would have caught it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sha256.core.enums import PixelFormat
from sha256.core.types import (
    FaceTrackingResult,
    FrameMetadata,
    LandmarkSet,
    RenderedFrame,
    VideoFrame,
)
from sha256.pipeline.runner import PipelineSnapshot

pytestmark = pytest.mark.gui

pytest.importorskip("PySide6.QtWidgets", reason="PySide6 is required for UI tests")


class StillRunner:
    """A runner standing still on one snapshot, so the status can be read off it."""

    def __init__(self, tracking: FaceTrackingResult, *, with_source: bool = False) -> None:
        metadata = tracking.metadata
        frame = RenderedFrame(
            data=np.zeros((4, 4, 3), dtype=np.uint8),
            pixel_format=PixelFormat.BGR24,
            metadata=metadata,
        )
        source = (
            VideoFrame(
                data=np.zeros((4, 4, 3), dtype=np.uint8),
                pixel_format=PixelFormat.BGR24,
                metadata=metadata,
            )
            if with_source
            else None
        )
        self._snapshot = PipelineSnapshot(frame, source, tracking, metadata.frame_id)

    is_running = True
    last_error = None
    fps = 30.0

    def latest(self) -> PipelineSnapshot:
        return self._snapshot

    def stop(self) -> None:
        return None


def tracking(*, face: bool) -> FaceTrackingResult:
    metadata = FrameMetadata(frame_id=7, capture_timestamp=7.0)
    if not face:
        return FaceTrackingResult.no_face(metadata)
    return FaceTrackingResult(
        metadata=metadata,
        face_detected=True,
        landmarks=LandmarkSet(points=np.zeros((478, 3), dtype=np.float32), scheme="mediapipe_478"),
    )


@pytest.fixture
def window(qtbot, app_context: Any):
    from sha256.ui.main_window import MainWindow

    widget = MainWindow(app_context)
    qtbot.addWidget(widget)
    return widget


class TestStatusLine:
    def test_a_lost_face_is_said_on_the_status_line(self, window):
        window._runner = StillRunner(tracking(face=False))

        window._refresh_status()

        assert "Face not found" in window._status_line.text()

    def test_a_tracked_face_leaves_the_line_empty(self, window):
        window._runner = StillRunner(tracking(face=True))

        window._refresh_status()

        assert window._status_line.text() == ""

    def test_a_learning_rest_is_said_on_the_status_line(self, window):
        """The null engine has no rest to learn -- rest_status is patched
        directly so this exercises the wiring, not a specific engine."""
        window._runner = StillRunner(tracking(face=True))
        window._context.avatars.select("avatar_001")
        window._context.avatars.engine.rest_status = lambda avatar_id: "centring (3/15)"

        window._refresh_status()

        assert window._status_line.text() == "centring (3/15)"


class TestRecentre:
    """The "Re-centre" button: AvatarEngine.reset_anchor, added 2026-09-05
    after an anchor learned mid-turn reproduced the owner's own "the avatar
    is to the side" report exactly."""

    def test_recentre_calls_reset_anchor_on_the_active_avatar(self, window):
        window._context.avatars.select("avatar_001")
        calls: list[str] = []
        window._context.avatars.engine.reset_anchor = calls.append

        window._on_recentre()

        assert calls == ["avatar_001"]

    def test_recentre_with_no_active_avatar_does_nothing(self, window):
        window._on_recentre()  # must not raise; nothing selected yet


class TestOutputControls:
    """Day 15: the quality combo and the black-screen button drive the
    controller's conditioner, and the status line says when the other side
    is looking at a black screen."""

    def test_the_quality_entries_set_the_published_quality(self, window):
        conditioner = window._context.controller.components.conditioner
        assert window._quality_actions["native"].isChecked()
        assert window._quality_actions["native"].text() == "Native"
        assert conditioner.quality == "native"

        window._quality_actions["480p"].trigger()

        assert conditioner.quality == "480p"
        assert window._quality_actions["480p"].isChecked()
        assert not window._quality_actions["native"].isChecked()
        window._quality_actions["native"].trigger()
        assert conditioner.quality == "native"

    def test_the_button_switches_the_black_screen_and_the_line_says_so(self, window):
        conditioner = window._context.controller.components.conditioner
        assert window._blackout_button.text() == "Camera battery died"
        assert not conditioner.manual

        window._blackout_button.setChecked(True)

        assert conditioner.manual and conditioner.blackout == "manual"
        assert "BLACK SCREEN" in window._status_line.text()
        assert "Camera battery died" in window._status_line.text()

        window._blackout_button.setChecked(False)

        assert not conditioner.manual
        window._refresh_status()
        assert "BLACK SCREEN" not in window._status_line.text()

    def test_the_hotkey_is_the_button(self, window):
        conditioner = window._context.controller.components.conditioner
        window._toggle_blackout()
        assert conditioner.manual and window._blackout_button.isChecked()
        window._toggle_blackout()
        assert not conditioner.manual and not window._blackout_button.isChecked()

    def test_an_automatic_black_screen_is_said_while_running(self, window):
        conditioner = window._context.controller.components.conditioner
        window._runner = StillRunner(tracking(face=False))
        # What the pipeline would do on the camera's own picture going out.
        black = RenderedFrame(
            data=np.zeros((4, 4, 3), dtype=np.uint8),
            pixel_format=PixelFormat.BGR24,
            metadata=FrameMetadata(frame_id=1, capture_timestamp=1.0),
            is_passthrough=True,
        )
        conditioner.condition(black, face_detected=False, passthrough=True)

        window._refresh_status()

        assert "BLACK SCREEN" in window._status_line.text()
        assert "camera" in window._status_line.text()
        conditioner.reset()


class TestBlackScreenUnmirrored:
    """Day 15: with the mirror on, the black screen's words read backwards
    in the preview -- so that frame goes to the preview unmirrored."""

    def test_the_black_screen_frame_is_shown_unmirrored(self, window):
        conditioner = window._context.controller.components.conditioner
        shown: list[dict] = []
        window._preview.show_frame = lambda data, fmt, tracking=None, **kw: shown.append(kw)
        window._runner = StillRunner(tracking(face=True))

        window._refresh_preview()
        assert shown[-1] == {"mirror": None, "guide": None}

        conditioner.set_manual(True)
        window._refresh_preview()
        assert shown[-1] == {"mirror": False, "guide": None}
        conditioner.set_manual(False)


@pytest.fixture
def guide_accepted(monkeypatch):
    """The photo guide (day 16) answered "choose the photograph"."""
    from sha256.ui import main_window as module

    monkeypatch.setattr(
        module.PhotoGuideDialog,
        "exec",
        lambda self: int(module.PhotoGuideDialog.DialogCode.Accepted),
    )


class TestAddAndDeleteAvatar:
    """Day 15: the import is checked and the library re-read. Day 16: the
    photo guide comes first. 2026-10-09, his ask: no Login any more -- Add
    avatar and Delete avatar are in the menu for whoever uses the app."""

    def test_add_and_delete_are_there_without_a_login(self, window):
        entries = [action.text() for action in window._menu.actions() if not action.isSeparator()]
        assert "Add avatar" in entries and "Delete avatar" in entries
        assert "Login" not in entries and "Owner" not in entries
        assert window._add_avatar_action.isVisible()
        assert window._delete_avatar_action.isVisible()
        assert not hasattr(window, "_owner") and not hasattr(window, "_login_action")

    def test_deleting_an_avatar_asks_removes_and_refills_the_bar(self, window, monkeypatch):
        """Day 17, his ask: the avatar chosen in the bar is removed after a
        question, with the pipeline stopped."""
        from sha256.ui import main_window as module

        library = window._context.paths.avatars
        index = window._avatar_combo.findData("avatar_001")
        assert index >= 0
        window._avatar_combo.setCurrentIndex(index)
        assert (library / "avatar_001").is_dir()

        asked: list[str] = []
        monkeypatch.setattr(
            module.QMessageBox,
            "question",
            lambda *a, **k: (asked.append(a[2]), module.QMessageBox.StandardButton.No)[1],
        )
        monkeypatch.setattr(module.QMessageBox, "information", lambda *a, **k: None)
        window._on_delete_avatar()
        assert (asked and "avatar_001" in asked[0]) or "Avatar" in asked[0]
        assert (library / "avatar_001").is_dir()  # No leaves it

        monkeypatch.setattr(
            module.QMessageBox, "question", lambda *a, **k: module.QMessageBox.StandardButton.Yes
        )
        window._on_delete_avatar()
        assert not (library / "avatar_001").exists()
        assert window._avatar_combo.findData("avatar_001") < 0
        assert window._context.avatars.active_id != "avatar_001"

    def test_delete_wants_an_avatar_chosen(self, window, monkeypatch):
        from sha256.ui import main_window as module

        library = window._context.paths.avatars
        window._avatar_combo.setCurrentIndex(0)  # none
        told: list[str] = []
        monkeypatch.setattr(module.QMessageBox, "information", lambda *a, **k: told.append(a[2]))
        window._on_delete_avatar()
        assert told and "Choose" in told[0]
        assert (library / "avatar_001").is_dir()

    def test_the_guide_can_be_declined(self, window, monkeypatch):
        from sha256.ui import main_window as module

        monkeypatch.setattr(
            module.PhotoGuideDialog,
            "exec",
            lambda self: int(module.PhotoGuideDialog.DialogCode.Rejected),
        )
        opened: list[int] = []
        monkeypatch.setattr(module.QFileDialog, "getOpenFileName", lambda *a, **k: opened.append(1))
        window._on_add_avatar()
        assert not opened

    def test_the_guide_carries_the_requirements_and_both_prompts(self, qtbot):
        from sha256.avatar.photo_guide import PARTED_LIPS_PROMPT, PORTRAIT_PROMPT
        from sha256.ui.photo_guide_dialog import PhotoGuideDialog

        dialog = PhotoGuideDialog()
        qtbot.addWidget(dialog)
        text = dialog.text
        assert "STRAIGHT at the camera" in text and "BELOW THE CHIN" in text
        assert PORTRAIT_PROMPT.strip() in text and PARTED_LIPS_PROMPT.strip() in text
        dialog._copy(PORTRAIT_PROMPT)
        assert dialog._copied.text() == "Copied."

    def test_adding_an_avatar_imports_refreshes_and_selects_it(
        self, window, monkeypatch, tmp_path, guide_accepted
    ):
        from sha256.ui import main_window as module

        library = window._context.paths.avatars
        calls: list[dict] = []

        def fake_import(lib, photo, *, name, model_path, template):
            calls.append({"lib": lib, "photo": photo, "name": name, "template": template})
            from tests.conftest import write_avatar

            root = write_avatar(lib, "nino")
            from sha256.avatar.importer import PortraitReport
            from sha256.core.avatar import AvatarDescriptor, AvatarManifest
            from tests.conftest import make_manifest_dict

            manifest = AvatarManifest.model_validate(make_manifest_dict("nino", name="Nino"))
            return AvatarDescriptor(manifest=manifest, root=root), PortraitReport(
                width=1, height=1, face_found=True, face_height_px=400.0
            )

        monkeypatch.setattr(module, "import_portrait", fake_import)
        monkeypatch.setattr(
            module.QFileDialog, "getOpenFileName", lambda *a, **k: (str(tmp_path / "nino.jpg"), "")
        )
        monkeypatch.setattr(module.QInputDialog, "getText", lambda *a, **k: ("Nino", True))
        monkeypatch.setattr(module.QMessageBox, "information", lambda *a, **k: None)

        window._on_add_avatar()

        assert calls and calls[0]["lib"] == library and calls[0]["name"] == "Nino"
        assert calls[0]["template"] is None  # the test library has no neural avatar
        assert window._avatar_combo.findData("nino") >= 0
        assert window._avatar_combo.currentData() == "nino"
        assert window._context.avatars.active_id == "nino"

    def test_a_refused_photograph_is_reported_and_nothing_changes(
        self, window, monkeypatch, tmp_path, guide_accepted
    ):
        from sha256.avatar.importer import AvatarImportError
        from sha256.ui import main_window as module

        warned: list[str] = []

        def refuse(*a, **k):
            raise AvatarImportError("the photograph cannot be an avatar:\nno face found")

        monkeypatch.setattr(module, "import_portrait", refuse)
        monkeypatch.setattr(
            module.QFileDialog, "getOpenFileName", lambda *a, **k: (str(tmp_path / "x.jpg"), "")
        )
        monkeypatch.setattr(module.QInputDialog, "getText", lambda *a, **k: ("X", True))
        monkeypatch.setattr(module.QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
        before = window._avatar_combo.count()

        window._on_add_avatar()

        assert warned and "no face found" in warned[0]
        assert window._avatar_combo.count() == before


class TestTheBarAndTheMenu:
    """Day 16, his report: fifteen controls in a row set a minimum width
    wider than the screen, so the window opened with its bar below the
    bottom edge and could not be dragged smaller. The bar keeps the six a
    call needs; the rest live in the Menu; the status line takes what is
    left and cuts what does not fit; the window opens inside the screen."""

    @staticmethod
    def _bar_texts(window) -> list[str]:
        layout = window._primary_button.parentWidget().layout()
        texts: list[str] = []
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if widget is not None:
                texts.append(widget.text() if hasattr(widget, "text") else widget.currentText())
        return texts

    def test_the_bar_holds_six_controls_and_the_menu(self, window):
        from sha256.pipeline.conditioning import QUALITY_NAMES

        texts = self._bar_texts(window)
        assert texts[:2] == ["Start", "Stop"]
        assert "Virtual camera" in texts and "Camera battery died" in texts
        assert texts[-1].startswith("Menu")
        moved = (
            "Dots",
            "Mirror",
            "Avatar's hair",
            "That looked wrong",
            "Re-centre",
            "Person",
            "Calibrate me",
            "Add avatar",
            "Delete avatar",
            "Instructions",
        )
        assert not any(text in texts for text in moved)

        entries = [action.text() for action in window._menu.actions() if not action.isSeparator()]
        assert entries == ["View", "Dots", "Mirror", "Avatar's hair", "Quality", *moved[3:]]
        assert [a.text() for a in window._view_group.actions()] == [
            "Avatar",
            "Tracking (no avatar)",
        ]
        assert window._view_avatar_action.isChecked()
        expected = ["Native" if n == "native" else n for n in QUALITY_NAMES]
        assert [a.text() for a in window._quality_group.actions()] == expected
        assert window._menu_button.menu() is window._menu

    def test_the_view_entries_choose_what_the_preview_shows(self, window):
        shown: list[object] = []
        window._preview.show_frame = lambda data, fmt, tracking=None, **kw: shown.append(tracking)
        result = tracking(face=True)
        window._runner = StillRunner(result, with_source=True)

        window._refresh_preview()
        assert shown[-1] is None  # the avatar view: the composited frame, no dots on it

        window._view_camera_action.trigger()
        window._refresh_preview()
        assert shown[-1] is result  # the tracking view: the camera frame with its tracking
        assert not window._view_avatar_action.isChecked()

    def test_dots_and_mirror_reach_the_preview(self, window):
        assert window._landmarks_action.isChecked() and window._preview._show_landmarks
        window._landmarks_action.setChecked(False)
        assert not window._preview._show_landmarks

        configured = window._context.settings.camera.mirror
        assert window._mirror_action.isChecked() == configured == window._preview._mirror
        window._mirror_action.toggle()
        assert window._preview._mirror == window._mirror_action.isChecked() == (not configured)

    def test_a_long_status_never_widens_the_window(self, window):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        window.show()
        QApplication.processEvents()
        before = window.minimumSize().width()
        assert before > 0

        text = "BLACK SCREEN — the other side sees “Camera battery died” (by the button) " * 3
        window._status_line.setText(text)
        QApplication.processEvents()

        assert window.minimumSize().width() == before
        assert window._status_line.text() == text
        window.close()

    def test_the_status_line_cuts_what_does_not_fit_and_keeps_the_rest_in_the_tooltip(self, qtbot):
        from PySide6.QtWidgets import QLabel

        from sha256.ui.widgets import ElidedLabel

        label = ElidedLabel()
        qtbot.addWidget(label)
        label.show()  # a hidden widget is not told about its resizes
        label.resize(120, 24)
        text = "Face not found — holding the avatar, and then some more words"
        label.setText(text)

        assert label.text() == text
        painted = QLabel.text(label)
        assert painted != text and painted.endswith("…")
        assert label.toolTip() == text
        assert label.minimumSizeHint().width() == 0
        assert label.sizeHint().width() > label.fontMetrics().horizontalAdvance(text)

        label.resize(label.sizeHint().width() + 20, 24)
        assert QLabel.text(label) == text and label.toolTip() == ""

        label.setText("")
        assert QLabel.text(label) == "" and label.toolTip() == ""

    def test_the_window_opens_inside_the_screen(self):
        from PySide6.QtCore import QRect

        from sha256.ui.main_window import opening_geometry

        roomy = opening_geometry(QRect(0, 0, 2560, 1392))
        assert (roomy.width(), roomy.height()) == (1200, 780)
        assert roomy.x() == (2560 - 1200) // 2 and roomy.y() >= 36

        # A 1080p laptop at 150 %, less the taskbar: 1280 x 660 free.
        small = opening_geometry(QRect(0, 0, 1280, 660))
        assert (small.width(), small.height()) == (1200, 600)
        assert small.y() >= 36 and small.y() + small.height() <= 660 - 20

        # A second screen to the right keeps its offset.
        second = opening_geometry(QRect(2560, 0, 1280, 660))
        assert second.x() >= 2560 and second.x() + second.width() <= 2560 + 1280

    def test_the_window_is_built_at_the_opening_size(self, window):
        from PySide6.QtGui import QGuiApplication

        from sha256.ui.main_window import opening_geometry

        available = QGuiApplication.primaryScreen().availableGeometry()
        assert window.size() == opening_geometry(available).size()


class TestPerson:
    """Menu → Person (day 16 evening, his ask): the people at this camera by
    name, the one in force checked, chosen with a click."""

    @staticmethod
    def _roster(app_context, *names: str):
        from sha256.calibration import DriverProfile, roster

        for i, name in enumerate(names):
            profile = DriverProfile(
                measured_at=f"2026-09-12T1{i}:00:00",
                name=name,
                engine_options={"gaze_gain": 0.5 + i},
            )
            roster.save_named(profile, app_context.paths.configs)

    @staticmethod
    def _window(qtbot, app_context):
        from sha256.ui.main_window import MainWindow

        widget = MainWindow(app_context)
        qtbot.addWidget(widget)
        return widget

    def test_the_submenu_lists_nobody_and_the_roster_by_name(self, qtbot, app_context):
        from sha256.ui.main_window import _NO_PERSON

        self._roster(app_context, "Zura", "ana")
        window = self._window(qtbot, app_context)

        actions = window._person_group.actions()
        assert [a.text() for a in actions] == [_NO_PERSON, "ana", "Zura"]
        assert [a.isChecked() for a in actions] == [True, False, False]
        assert window._person_menu.title() == "Person"
        window._refresh_status()
        assert window._status_line.text() == ""

    def test_choosing_a_person_puts_the_profile_in_force_and_nobody_removes_it(
        self, qtbot, app_context
    ):
        import json

        self._roster(app_context, "Zura", "ana")
        window = self._window(qtbot, app_context)
        path = app_context.paths.configs / "local.driver.json"

        window._on_person_chosen("Zura")

        assert json.loads(path.read_text(encoding="utf-8"))["name"] == "Zura"
        assert window._active_profile is not None and window._active_profile.name == "Zura"
        assert app_context.driver_profile is window._active_profile
        checked = [a.text() for a in window._person_group.actions() if a.isChecked()]
        assert checked == ["Zura"]
        window._refresh_status()
        assert window._status_line.text() == "Profile: Zura"

        window._on_person_chosen(None)

        assert not path.exists() and window._active_profile is None
        checked = [a.text() for a in window._person_group.actions() if a.isChecked()]
        assert checked[0].startswith("Nobody")
        window._refresh_status()
        assert window._status_line.text() == ""

    def test_an_unnamed_profile_in_force_is_adopted_at_start(self, qtbot, app_context):
        import json

        path = app_context.paths.configs / "local.driver.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"measured_at": "2026-09-12T00:42:46+04:00", "engine_options": {}}),
            encoding="utf-8",
        )

        window = self._window(qtbot, app_context)

        assert json.loads(path.read_text(encoding="utf-8"))["name"] == "Unnamed (2026-09-12 00:42)"
        assert (app_context.paths.configs / "drivers" / "unnamed-2026-09-12-00-42.json").is_file()
        checked = [a.text() for a in window._person_group.actions() if a.isChecked()]
        assert checked == ["Unnamed (2026-09-12 00:42)"]

    def test_a_chosen_profile_reaches_an_engine_that_takes_one_and_the_avatar_reloads(
        self, qtbot, app_context
    ):
        self._roster(app_context, "Zura")
        window = self._window(qtbot, app_context)
        engine = app_context.avatars.engine
        given: list[dict] = []
        engine.set_driver_options = given.append
        app_context.avatars.select("avatar_001")
        window._on_start()
        try:
            window._on_person_chosen("Zura")
            assert given[-1] == {"gaze_gain": 0.5}
            assert app_context.avatars.active_id == "avatar_001" and window._runner.is_running
            window._on_person_chosen(None)
            assert given[-1] == {}
        finally:
            window._on_stop()

    def test_a_missing_roster_file_is_said_and_nothing_changes(
        self, qtbot, app_context, monkeypatch
    ):
        from sha256.ui import main_window as module

        warned: list[str] = []
        monkeypatch.setattr(module.QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
        window = self._window(qtbot, app_context)

        window._on_person_chosen("Nobody Here")

        assert warned and "Nobody Here" in warned[0]
        assert window._active_profile is None


class TestInstructions:
    """Menu -> Instructions (day 16 night): the last entry, opening the text."""

    def test_the_entry_opens_the_dialog_with_the_text(self, window, monkeypatch):
        from sha256.ui import instructions_dialog as module

        opened: list[str] = []

        def fake_exec(self):
            opened.append(self.text)
            return 0

        monkeypatch.setattr(module.InstructionsDialog, "exec", fake_exec)
        window._on_instructions()
        assert opened and "3. THE BAR" in opened[0] and "Calibrate me" in opened[0]
        assert window._instructions_action.text() == "Instructions"


class TestHeadGuideInThePreview:
    """Day 21: while Calibrate me is on, every preview frame carries the
    head guide and its fit, read off the snapshot's tracking."""

    def test_the_guide_rides_with_the_frame_while_calibrating(self, window):
        from sha256.ui.head_guide import GuideFit, GuideState, HeadGuide

        shown: list[dict] = []
        window._preview.show_frame = lambda data, fmt, tracking=None, **kw: shown.append(kw)
        window._runner = StillRunner(tracking(face=True))

        window._refresh_preview()
        assert shown[-1]["guide"] is None and window._guide_fit is None

        window._calibration = object()  # a session, as far as the preview cares
        try:
            window._refresh_preview()
            guide, fit = shown[-1]["guide"]
            assert isinstance(guide, HeadGuide) and isinstance(fit, GuideFit)
            # all-zero landmarks: no face to speak of
            assert fit.state is GuideState.NO_FACE and window._guide_fit is fit
        finally:
            window._calibration = None
