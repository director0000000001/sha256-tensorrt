"""The window's "Calibrate me" (day 16): refused without a running pipeline
or a calibratable engine; asks first; shows the instruction while running;
writes and applies the profile at the end; a second press cancels and
removes the recording."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from sha256.calibration.script import STEPS
from sha256.calibration.session import SessionFiles
from tests.unit.test_calibration_measure import synthetic_samples

pytestmark = pytest.mark.gui

pytest.importorskip("PySide6.QtWidgets", reason="PySide6 is required for UI tests")


class FakeSession:
    """Stands in for CalibrationSession: finished at once, with a full set
    of synthetic samples."""

    instances: ClassVar[list[FakeSession]] = []

    def __init__(self, runner, *, engine, avatar_id, directory, **kwargs) -> None:
        self.engine = engine
        self.avatar_id = avatar_id
        self.directory = Path(directory)
        self.kwargs = kwargs
        self.steps = STEPS
        self.started_ = False
        self.cancelled = False
        self.finish_calls = 0
        self.finished_flag = False
        FakeSession.instances.append(self)

    def start(self) -> None:
        self.started_ = True
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def started(self) -> bool:
        return self.started_

    @property
    def finished(self) -> bool:
        return self.finished_flag

    def current(self):
        from sha256.calibration.script import StepPosition

        return StepPosition(2, STEPS[2], 1.5, True)

    @property
    def samples(self):
        return synthetic_samples(talk_gain=0.8)

    def finish(self) -> SessionFiles:
        self.finish_calls += 1
        tracking = self.directory / "tracking.npz"
        tracking.write_bytes(b"")
        return SessionFiles(self.directory, self.directory / "clip.mkv", None, tracking, 120, 0)

    def cancel(self) -> None:
        self.cancelled = True
        import shutil

        shutil.rmtree(self.directory, ignore_errors=True)


@pytest.fixture
def window(qtbot, app_context: Any):
    from sha256.ui.main_window import MainWindow

    widget = MainWindow(app_context)
    qtbot.addWidget(widget)
    return widget


class FakeSpeaker:
    instances: ClassVar[list[FakeSpeaker]] = []

    def __init__(self) -> None:
        self.said: list[str] = []
        self.stopped = False
        FakeSpeaker.instances.append(self)

    available = True

    def say(self, text: str) -> None:
        self.said.append(text)

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def calibratable(window, monkeypatch):
    """The null engine, given the neural engine's calibration surface; the
    name asked at the end answered "Giorgi Test"."""
    from sha256.ui import main_window as module

    engine = window._context.avatars.engine
    given: list[dict] = []
    engine.frame_readings = lambda avatar_id: {"anchor_frozen": 1.0}
    engine.set_driver_options = given.append
    engine.driver_options = {}
    window._context.avatars.select("avatar_001")
    FakeSpeaker.instances.clear()
    window._speaker_factory = FakeSpeaker
    monkeypatch.setattr(module.QInputDialog, "getText", lambda *a, **k: ("Giorgi Test", True))
    return given


class TestRefusals:
    def test_not_running_is_said_and_nothing_starts(self, window, monkeypatch):
        from sha256.ui import main_window as module

        said: list[str] = []
        monkeypatch.setattr(module.QMessageBox, "information", lambda *a, **k: said.append(a[2]))
        window._on_calibrate()
        assert said and "Start" in said[0] and window._calibration is None

    def test_the_photograph_engine_cannot_be_calibrated(self, window, monkeypatch):
        from sha256.ui import main_window as module

        said: list[str] = []
        monkeypatch.setattr(module.QMessageBox, "information", lambda *a, **k: said.append(a[2]))
        window._context.avatars.select("avatar_001")
        window._on_start()
        try:
            window._on_calibrate()
        finally:
            window._on_stop()
        assert said and "neural engine" in said[0] and window._calibration is None

    def test_saying_no_starts_nothing(self, window, calibratable, monkeypatch):
        from sha256.ui import main_window as module

        FakeSession.instances.clear()
        monkeypatch.setattr(module, "CalibrationSession", FakeSession)
        monkeypatch.setattr(
            module.QMessageBox, "question", lambda *a, **k: module.QMessageBox.StandardButton.No
        )
        window._on_start()
        try:
            window._on_calibrate()
        finally:
            window._on_stop()
        assert not FakeSession.instances and window._calibration is None


def good_fit(*args, **kwargs):
    from sha256.ui.head_guide import GuideFit, GuideState

    return GuideFit(GuideState.GOOD, "Good. Hold still.")


def far_fit(*args, **kwargs):
    from sha256.ui.head_guide import GuideFit, GuideState

    return GuideFit(GuideState.CLOSER, "Come closer")


class TestTheRun:
    def _start(self, window, monkeypatch, *, place: bool = True):
        """Press Calibrate me; with ``place``, sit in the green oval for the
        second the placing phase wants, so the session's clock starts."""
        from sha256.ui import main_window as module

        FakeSession.instances.clear()
        monkeypatch.setattr(module, "CalibrationSession", FakeSession)
        monkeypatch.setattr(
            module.QMessageBox, "question", lambda *a, **k: module.QMessageBox.StandardButton.Yes
        )
        window._on_start()
        window._on_calibrate()
        if place:
            # The preview reads the fit off the runner's snapshot each tick
            # (_refresh_preview); here it is set by hand, as a good one.
            for _ in range(module._PLACING_GOOD_TICKS):
                window._guide_fit = good_fit()
                window._tick_calibration()
        return FakeSession.instances[-1]

    def test_the_clock_waits_for_the_head_in_the_green_oval(
        self, window, calibratable, monkeypatch
    ):
        """Day 21: the session is made at the press but started only once
        the guide has read the head as placed for a second; the guide is
        drawn meanwhile, and the one line of what to do is shown and said."""
        from sha256.ui import main_window as module

        session = self._start(window, monkeypatch, place=False)
        try:
            assert window._calibration is session and not session.started
            assert window._calibrate_action.text() == "Stop calibrating"
            assert module._PLACING_TEXT in window._instruction.text()
            assert FakeSpeaker.instances[-1].said == [module._PLACING_TEXT]

            window._guide_fit = far_fit()
            window._tick_calibration()
            assert not session.started
            assert "Come closer" in window._instruction.text()

            for _ in range(module._PLACING_GOOD_TICKS - 1):
                window._guide_fit = good_fit()
                window._tick_calibration()
            assert not session.started  # not a second yet
            assert "Starting" in window._instruction.text()
            # a bad frame starts the second over
            window._guide_fit = far_fit()
            window._tick_calibration()
            assert not session.started and window._placing_good == 0

            for _ in range(module._PLACING_GOOD_TICKS):
                window._guide_fit = good_fit()
                window._tick_calibration()
            assert session.started
            window._tick_calibration()
            assert STEPS[2].instruction in window._instruction.text()
            # the guide is gone with the session
            window._cancel_calibration()
            assert window._guide_fit is None and window._placing_good == 0
        finally:
            window._cancel_calibration()
            window._on_stop()

    def test_the_instruction_shows_and_the_button_becomes_stop(
        self, window, calibratable, monkeypatch
    ):
        session = self._start(window, monkeypatch)
        try:
            assert session.started and window._calibration is session
            assert session.avatar_id == "avatar_001"
            assert (
                "recordings" in str(session.directory) and "calibration-" in session.directory.name
            )
            assert window._calibrate_action.text() == "Stop calibrating"
            assert not window._instruction.isHidden()
            assert STEPS[2].instruction in window._instruction.text()
            assert "Get ready" in window._instruction.text()
            # Spoken once per step, not once per tick -- after the placing
            # line (day 21), said once at the press.
            window._tick_calibration()
            assert FakeSpeaker.instances and FakeSpeaker.instances[-1].said == [
                "Put your head in the green oval",
                STEPS[2].instruction,
            ]
            window._refresh_status()
            assert "Calibrating" in window._status_line.text()
        finally:
            window._cancel_calibration()
            window._on_stop()

    def test_a_second_press_cancels_and_removes_the_recording(
        self, window, calibratable, monkeypatch
    ):
        session = self._start(window, monkeypatch)
        assert session.directory.is_dir()

        window._on_calibrate()

        assert session.cancelled and not session.directory.exists()
        assert FakeSpeaker.instances[-1].stopped
        assert window._calibration is None and window._calibrate_action.text() == "Calibrate me"
        assert window._instruction.isHidden()
        window._on_stop()

    def test_the_end_writes_the_profile_applies_it_and_restarts(
        self, window, calibratable, monkeypatch
    ):
        from sha256.ui import main_window as module

        told: list[str] = []
        monkeypatch.setattr(module.QMessageBox, "information", lambda *a, **k: told.append(a[2]))
        session = self._start(window, monkeypatch)
        session.finished_flag = True

        window._tick_calibration()

        assert session.finish_calls == 1
        configs = window._context.paths.configs
        path = configs / "local.driver.json"
        assert path.is_file()
        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["name"] == "Giorgi Test"
        # The roster holds the same profile under the person's name.
        named = configs / "drivers" / "giorgi-test.json"
        assert named.is_file() and json.loads(named.read_text(encoding="utf-8")) == written
        assert window._active_profile is not None and window._active_profile.name == "Giorgi Test"
        checked = [a.text() for a in window._person_group.actions() if a.isChecked()]
        assert checked == ["Giorgi Test"]
        assert written["avatar"] == "avatar_001"
        assert written["engine_options"]["expression_gain"] == pytest.approx(1.25, abs=0.05)
        assert written["session"] == str(session.directory.relative_to(window._context.paths.root))
        assert (session.directory / "profile.json").is_file()
        assert (session.directory / "report.txt").is_file()
        assert (
            calibratable
            and calibratable[-1]["expression_gain"] == written["engine_options"]["expression_gain"]
        )
        # Reloaded under the profile, and running again.
        assert window._context.avatars.active_id == "avatar_001"
        assert window._runner.is_running and window._calibration is None
        assert window._calibrate_action.text() == "Calibrate me" and window._instruction.isHidden()
        assert told and "Giorgi Test" in told[-1] and "expression_gain" in told[-1]
        window._refresh_status()
        assert window._status_line.text() == "Profile: Giorgi Test"
        window._on_stop()

    def test_an_earlier_unnamed_profile_joins_the_roster(self, window, calibratable, monkeypatch):
        from sha256.ui import main_window as module

        monkeypatch.setattr(module.QMessageBox, "information", lambda *a, **k: None)
        path = window._context.paths.configs / "local.driver.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"measured_at": "2026-09-11T09:00:00", "engine_options": {"eye_gain": 0.5}}),
            encoding="utf-8",
        )
        session = self._start(window, monkeypatch)
        session.finished_flag = True

        window._tick_calibration()

        kept = path.parent / "drivers" / "unnamed-2026-09-11-09-00.json"
        assert kept.is_file()
        earlier = json.loads(kept.read_text(encoding="utf-8"))
        assert earlier["engine_options"] == {"eye_gain": 0.5}
        assert earlier["name"] == "Unnamed (2026-09-11 09:00)"
        assert json.loads(path.read_text(encoding="utf-8"))["name"] == "Giorgi Test"
        names = [a.text() for a in window._person_group.actions()]
        assert names[1:] == ["Giorgi Test", "Unnamed (2026-09-11 09:00)"]
        window._on_stop()

    def test_the_same_person_again_keeps_what_this_run_did_not_measure(
        self, window, calibratable, monkeypatch
    ):
        from sha256.calibration import roster
        from sha256.ui import main_window as module

        monkeypatch.setattr(module.QMessageBox, "information", lambda *a, **k: None)
        configs = window._context.paths.configs
        earlier = module.DriverProfile(
            measured_at="2026-09-11T09:00:00",
            name="Giorgi Test",
            engine_options={"gaze_gain": 0.7, "expression_gain": 9.0},
        )
        roster.save_named(earlier, configs)
        session = self._start(window, monkeypatch)
        session.finished_flag = True

        window._tick_calibration()

        written = json.loads((configs / "local.driver.json").read_text(encoding="utf-8"))
        assert written["engine_options"]["gaze_gain"] == 0.7  # not measured by this run: kept
        assert written["engine_options"]["expression_gain"] != 9.0  # measured: replaced
        window._on_stop()

    def test_a_blank_name_still_keeps_the_profile_under_the_moment(
        self, window, calibratable, monkeypatch
    ):
        from sha256.ui import main_window as module

        monkeypatch.setattr(module.QMessageBox, "information", lambda *a, **k: None)
        monkeypatch.setattr(module.QInputDialog, "getText", lambda *a, **k: ("", False))
        session = self._start(window, monkeypatch)
        session.finished_flag = True

        window._tick_calibration()

        written = json.loads(
            (window._context.paths.configs / "local.driver.json").read_text(encoding="utf-8")
        )
        assert written["name"].startswith("Person 20")
        window._on_stop()

    def test_closing_the_window_cancels_a_run(self, window, calibratable, monkeypatch):
        session = self._start(window, monkeypatch)
        window.close()
        assert session.cancelled and not session.directory.exists()
