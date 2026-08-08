"""Smoke tests for the application shell.

Deliberately shallow: the UI is a placeholder and asserting on its layout would
lock in decisions that are meant to change. What is worth testing now is that
the window constructs against a real context, that the controls are wired to
the controller and the avatar manager, and that closing it stops the pipeline.

Requires an offscreen Qt platform on CI; ``conftest`` sets ``QT_QPA_PLATFORM``.
"""

from __future__ import annotations

import pytest

from sha256.app.bootstrap import ApplicationContext
from sha256.core.enums import PipelineState

pytestmark = [pytest.mark.gui, pytest.mark.integration]

pytest.importorskip("PySide6.QtWidgets", reason="PySide6 is required for UI tests")


@pytest.fixture
def window(qtbot, app_context: ApplicationContext):
    from sha256.ui.main_window import MainWindow

    widget = MainWindow(app_context)
    qtbot.addWidget(widget)
    return widget


class TestConstruction:
    def test_window_builds_with_a_real_context(self, window):
        assert "sha256" in window.windowTitle()

    def test_avatar_selector_lists_discovered_avatars(self, window):
        labels = [window._avatar_combo.itemData(i) for i in range(window._avatar_combo.count())]
        assert None in labels, "a 'none' entry should always be offered"
        assert "avatar_001" in labels
        assert "avatar_002" in labels

    def test_camera_selector_lists_enumerated_devices(self, window):
        assert window._camera_combo.count() >= 1


class TestControls:
    def test_start_and_stop_drive_the_controller(self, window, app_context: ApplicationContext):
        window._on_start()
        after_start = app_context.controller.state

        window._on_stop()
        after_stop = app_context.controller.state

        assert (after_start, after_stop) == (PipelineState.RUNNING, PipelineState.STOPPED)

    def test_buttons_reflect_pipeline_state(self, window):
        window._on_stop()
        assert window._start_button.isEnabled()
        assert not window._stop_button.isEnabled()

        window._on_start()
        assert not window._start_button.isEnabled()
        assert window._stop_button.isEnabled()

    def test_selecting_an_avatar_activates_it(self, window, app_context: ApplicationContext):
        index = window._avatar_combo.findData("avatar_002")
        window._avatar_combo.setCurrentIndex(index)

        assert app_context.avatars.active_id == "avatar_002"

    def test_selecting_none_deselects(self, window, app_context: ApplicationContext):
        window._avatar_combo.setCurrentIndex(window._avatar_combo.findData("avatar_001"))
        window._avatar_combo.setCurrentIndex(window._avatar_combo.findData(None))

        assert app_context.avatars.active_id is None


class TestShutdown:
    def test_closing_stops_the_pipeline(self, window, app_context: ApplicationContext):
        window._on_start()
        window.close()

        assert app_context.controller.state is PipelineState.STOPPED


class TestDebugPanel:
    def test_panel_reports_no_measurements_before_any_run(self, window):
        """Absent metrics must read as absent, not as zero."""
        window._debug_panel.refresh()
        text = window._debug_panel._metrics_view.toPlainText()

        assert "No measurements yet" in text

    def test_panel_shows_wiring(self, window):
        text = window._debug_panel._wiring_label.text()

        assert "camera" in text
        assert "VRAM budget" in text
