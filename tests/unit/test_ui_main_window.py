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
from sha256.core.types import FaceTrackingResult, FrameMetadata, LandmarkSet, RenderedFrame
from sha256.pipeline.runner import PipelineSnapshot

pytestmark = pytest.mark.gui

pytest.importorskip("PySide6.QtWidgets", reason="PySide6 is required for UI tests")


class StillRunner:
    """A runner standing still on one snapshot, so the status can be read off it."""

    def __init__(self, tracking: FaceTrackingResult) -> None:
        metadata = tracking.metadata
        frame = RenderedFrame(
            data=np.zeros((4, 4, 3), dtype=np.uint8),
            pixel_format=PixelFormat.BGR24,
            metadata=metadata,
        )
        self._snapshot = PipelineSnapshot(frame, None, tracking, metadata.frame_id)

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

    def test_the_quality_combo_sets_the_published_quality(self, window):
        conditioner = window._context.controller.components.conditioner
        assert window._quality_combo.currentText() == "Native"
        assert conditioner.quality == "native"

        window._quality_combo.setCurrentText("480p")

        assert conditioner.quality == "480p"
        window._quality_combo.setCurrentText("Native")
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
        assert shown[-1] == {"mirror": None}

        conditioner.set_manual(True)
        window._refresh_preview()
        assert shown[-1] == {"mirror": False}
        conditioner.set_manual(False)


class TestOwnerAndAddAvatar:
    """Day 15: "Add avatar" appears only behind the owner's passphrase; the
    passphrase is set on the first login; the import is checked and the
    library re-read."""

    def test_add_avatar_is_hidden_until_the_owner_logs_in(self, window, monkeypatch):
        from sha256.ui import main_window as module

        assert not window._add_avatar_button.isVisible() or window._add_avatar_button.isHidden()
        assert window._login_button.text() == "Login"
        assert not window._owner.is_configured

        answers = iter([("open sesame", True), ("open sesame", True)])
        monkeypatch.setattr(module.QInputDialog, "getText", lambda *a, **k: next(answers))
        monkeypatch.setattr(module.QMessageBox, "information", lambda *a, **k: None)

        window._on_login()

        assert window._owner.is_configured and window._owner.is_unlocked
        assert window._login_button.text() == "Owner" and not window._login_button.isEnabled()
        assert not window._add_avatar_button.isHidden()

    def test_a_wrong_passphrase_unlocks_nothing(self, window, monkeypatch):
        from sha256.ui import main_window as module

        window._owner.set_passphrase("open sesame")
        window._owner.lock()
        window._refresh_owner_state()
        warned: list[str] = []
        monkeypatch.setattr(module.QInputDialog, "getText", lambda *a, **k: ("nope", True))
        monkeypatch.setattr(module.QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))

        window._on_login()

        assert not window._owner.is_unlocked and window._add_avatar_button.isHidden()
        assert warned and "not the passphrase" in warned[0]

    def test_adding_an_avatar_imports_refreshes_and_selects_it(self, window, monkeypatch, tmp_path):
        from sha256.ui import main_window as module

        window._owner.set_passphrase("open sesame")
        window._refresh_owner_state()
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
        self, window, monkeypatch, tmp_path
    ):
        from sha256.avatar.importer import AvatarImportError
        from sha256.ui import main_window as module

        window._owner.set_passphrase("open sesame")
        window._refresh_owner_state()
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

    def test_locked_the_button_does_nothing(self, window, monkeypatch):
        from sha256.ui import main_window as module

        opened: list[int] = []
        monkeypatch.setattr(module.QFileDialog, "getOpenFileName", lambda *a, **k: opened.append(1))
        window._on_add_avatar()
        assert not opened
