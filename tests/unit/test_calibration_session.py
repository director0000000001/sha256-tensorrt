"""The live session (day 16): it listens to the runner, places each frame
in its step by the clock, resets the anchor on entering the still step,
reads the engine and the avatar's face, records the clip and its tracking
aligned frame for frame, and removes everything on a cancel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sha256.calibration.script import CalibrationStep
from sha256.calibration.session import (
    CLIP_NAME,
    READY_KIND,
    SIDECAR_NAME,
    TRACKING_NAME,
    CalibrationSession,
)
from sha256.core.enums import PixelFormat
from sha256.core.types import (
    BlendshapeSet,
    FaceTrackingResult,
    FrameMetadata,
    HeadPose,
    LandmarkSet,
    RenderedFrame,
    VideoFrame,
)
from sha256.pipeline.runner import PipelineRunner
from tests.unit.test_pipeline_runner import FakeController

STEPS = (
    CalibrationStep("lead", 0.2, "Get ready", measured=False),
    CalibrationStep("rest", 0.5, "SIT STILL"),
    CalibrationStep("talk", 0.5, "TALK"),
    CalibrationStep("rest_end", 0.3, "SIT STILL again"),
)
FPS = 30.0


def face_points(seed: int) -> np.ndarray:
    """A plausible 478-point face in normalised coordinates: random, so
    every ratio has a non-zero denominator."""
    rng = np.random.default_rng(seed)
    return (0.3 + 0.4 * rng.random((478, 3))).astype(np.float32)


def tracked(frame_id: int, timestamp: float, *, face: bool = True) -> FaceTrackingResult:
    metadata = FrameMetadata(frame_id=frame_id, capture_timestamp=timestamp)
    if not face:
        return FaceTrackingResult.no_face(metadata)
    return FaceTrackingResult(
        metadata=metadata,
        face_detected=True,
        landmarks=LandmarkSet(points=face_points(frame_id), scheme="mediapipe_478"),
        head_pose=HeadPose(yaw=1.0, pitch=-2.0, translation=(0.1, 0.2, 48.0)),
        blendshapes=BlendshapeSet(names=("jawOpen",), weights=np.array([0.1], dtype=np.float32)),
    )


class BiggerController(FakeController):
    """The runner's fake, with frames a video codec accepts."""

    def process_once(self) -> RenderedFrame | None:
        frame = super().process_once()
        assert frame is not None
        return RenderedFrame(
            data=np.full((48, 64, 3), 30, dtype=np.uint8),
            pixel_format=PixelFormat.BGR24,
            metadata=frame.metadata,
        )

    @property
    def last_source_frame(self) -> VideoFrame | None:
        return VideoFrame(
            data=np.full((48, 64, 3), self.frames % 255, dtype=np.uint8),
            pixel_format=PixelFormat.BGR24,
            metadata=FrameMetadata(frame_id=self.frames, capture_timestamp=0.0),
        )


class FakeEngine:
    def __init__(self) -> None:
        self.resets: list[str] = []
        self.readings = 0

    def reset_anchor(self, avatar_id: str) -> None:
        self.resets.append(avatar_id)

    def frame_readings(self, avatar_id: str) -> dict[str, float]:
        self.readings += 1
        return {"anchor_frozen": 1.0, "holding": 0.0, "gate_open": 0.0, "exp_20_1": 0.008}

    def anchor_state(self, avatar_id: str) -> dict[str, np.ndarray]:
        return {"exp_d0": np.zeros((1, 21, 3), dtype=np.float32), "rest_ear": np.array(0.3)}


class FakeOutputTracker:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.seen: list[int] = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def process(self, frame: VideoFrame) -> FaceTrackingResult:
        self.seen.append(frame.metadata.frame_id)
        return tracked(frame.metadata.frame_id, frame.metadata.capture_timestamp)


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def parts(tmp_path: Path):
    controller = BiggerController()
    runner = PipelineRunner(controller)  # type: ignore[arg-type]
    engine = FakeEngine()
    output = FakeOutputTracker()
    clock = Clock()
    session = CalibrationSession(
        runner,
        engine=engine,
        avatar_id="me",
        directory=tmp_path / "recordings" / "calibration-test",
        steps=STEPS,
        output_tracker_factory=lambda: output,
        fps=FPS,
        clock=clock,
    )
    return controller, runner, engine, output, clock, session


def run_frames(controller, runner, clock, count: int, *, faces: bool = True) -> None:
    for i in range(count):
        clock.t += 1.0 / FPS
        controller.tracking = tracked(i, clock.t, face=faces)
        frame = controller.process_once()
        runner._publish(frame)


class TestReadPhase:
    def test_read_phase_frames_are_ready_not_the_step(self, tmp_path):
        steps = (
            CalibrationStep("rest", 0.3, "SIT STILL"),
            CalibrationStep("talk", 0.4, "TALK", read_seconds=0.3),
        )
        controller = BiggerController()
        runner = PipelineRunner(controller)  # type: ignore[arg-type]
        clock = Clock()
        session = CalibrationSession(
            runner,
            engine=FakeEngine(),
            avatar_id="me",
            directory=tmp_path / "r",
            steps=steps,
            clock=clock,
        )
        session.start()
        run_frames(controller, runner, clock, 30)  # 1.0 s
        kinds = [s.step for s in session.samples]
        assert kinds[:9] == ["rest"] * 9
        assert READY_KIND in kinds and kinds.index(READY_KIND) == 9
        assert kinds[-3:] == ["talk"] * 3
        files = session.finish()
        sidecar = json.loads((files.directory / SIDECAR_NAME).read_text(encoding="utf-8"))
        kinds_in_sidecar = [s["kind"] for s in sidecar["segments"]]
        assert kinds_in_sidecar == ["rest", READY_KIND, "talk"]
        assert sidecar["segments"][1]["instruction"] == "Get ready: TALK"

    def test_the_anchor_is_reset_as_the_still_steps_measured_phase_starts(self, tmp_path):
        """Not while the instruction is being read: the read phase's frames
        are the person looking at the screen, not at the camera."""
        steps = (
            CalibrationStep("rest", 0.3, "SIT STILL", read_seconds=0.3),
            CalibrationStep("talk", 0.3, "TALK"),
        )
        controller = BiggerController()
        runner = PipelineRunner(controller)  # type: ignore[arg-type]
        clock = Clock()
        engine = FakeEngine()
        session = CalibrationSession(
            runner,
            engine=engine,
            avatar_id="me",
            directory=tmp_path / "r",
            steps=steps,
            clock=clock,
        )
        session.start()
        run_frames(controller, runner, clock, 5)  # 0.17 s: reading
        assert engine.resets == []
        run_frames(controller, runner, clock, 10)  # 0.5 s: the measured phase
        assert engine.resets == ["me"]
        run_frames(controller, runner, clock, 10)  # 0.83 s: talking
        assert engine.resets == ["me"]
        session.cancel()


class TestSession:
    def test_frames_land_in_their_steps_and_the_anchor_is_reset_once(self, parts):
        controller, runner, engine, output, clock, session = parts
        session.start()
        assert session.started and not session.finished and session.current()[1].kind == "lead"

        run_frames(controller, runner, clock, 45)  # 1.5 s = the whole script
        clock.t += 0.001  # past the last step's end, not on it

        assert session.finished
        kinds = [s.step for s in session.samples]
        assert kinds[0] == "lead" and "rest" in kinds and "talk" in kinds and "rest_end" in kinds
        assert kinds == sorted(kinds, key=[st.kind for st in STEPS].index)
        assert engine.resets == ["me"]
        assert engine.readings == len(session.samples) == 45
        assert all(s.face and s.his and s.engine for s in session.samples)
        assert output.started

    def test_finish_writes_the_clip_the_sidecar_and_the_tracking_aligned(self, parts):
        controller, runner, engine, output, clock, session = parts
        session.start()
        run_frames(controller, runner, clock, 45)

        files = session.finish()

        assert files.clip == files.directory / CLIP_NAME and files.clip.is_file()
        assert files.frames_written == 45 and files.frames_dropped == 0
        sidecar = json.loads((files.directory / SIDECAR_NAME).read_text(encoding="utf-8"))
        assert sidecar["frames"] == 45 and sidecar["avatar"] == "me"
        assert (sidecar["width"], sidecar["height"]) == (64, 48)
        segments = {s["kind"]: s for s in sidecar["segments"]}
        assert set(segments) == {"lead", "rest", "talk", "rest_end"}
        assert segments["lead"]["first_frame"] == 0
        assert segments["rest_end"]["last_frame"] == 44
        assert all(
            segments[a]["last_frame"] + 1 == segments[b]["first_frame"]
            for a, b in (("lead", "rest"), ("rest", "talk"), ("talk", "rest_end"))
        )
        data = np.load(files.directory / TRACKING_NAME, allow_pickle=False)
        assert data["landmarks"].shape == (45, 478, 3)
        assert data["translation"].shape == (45, 3) and data["translation"][0, 2] == 48.0
        assert data["detected"].all() and data["timestamps"].shape == (45,)
        assert data["anchor_exp_d0"].shape == (1, 21, 3)
        assert output.stopped
        # The output reader saw frames and its readings were merged in.
        assert output.seen and any(s.avatar for s in session.samples)
        with pytest.raises(RuntimeError):
            CalibrationSession(
                runner, engine=engine, avatar_id="me", directory=files.directory
            ).finish()

    def test_a_lost_face_is_a_sample_without_readings(self, parts):
        controller, runner, _engine, _output, clock, session = parts
        session.start()
        run_frames(controller, runner, clock, 10, faces=False)
        assert all(not s.face and not s.his for s in session.samples)
        files = session.finish()
        data = np.load(files.directory / TRACKING_NAME, allow_pickle=False)
        assert not data["detected"].any() and data["landmarks"].shape == (10, 0, 3)

    def test_cancel_removes_the_recording(self, parts):
        controller, runner, _engine, output, clock, session = parts
        session.start()
        run_frames(controller, runner, clock, 12)
        assert session.directory.is_dir()

        session.cancel()

        assert not session.directory.exists()
        assert output.stopped
        # Nothing arrives after: the observer is gone.
        before = len(session.samples)
        run_frames(controller, runner, clock, 3)
        assert len(session.samples) == before

    def test_start_twice_is_once(self, parts):
        _controller, _runner, _engine, _output, clock, session = parts
        session.start()
        first = session.elapsed
        clock.t += 0.1
        session.start()
        assert session.elapsed == pytest.approx(first + 0.1)
        session.cancel()

    def test_without_a_clip_or_an_output_tracker_only_the_tracking_is_written(self, tmp_path):
        controller = BiggerController()
        runner = PipelineRunner(controller)  # type: ignore[arg-type]
        clock = Clock()
        session = CalibrationSession(
            runner,
            engine=object(),  # no reset_anchor, no readings, no anchor
            avatar_id="me",
            directory=tmp_path / "s",
            steps=STEPS,
            record_clip=False,
            clock=clock,
        )
        session.start()
        run_frames(controller, runner, clock, 20)
        files = session.finish()
        assert files.clip is None and files.sidecar is None and files.frames_written == 0
        assert files.tracking.is_file()
        assert all(not s.engine and s.avatar is None for s in session.samples)
        assert all(s.anchor_frozen for s in session.samples)  # no readings: taken as frozen
