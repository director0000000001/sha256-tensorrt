"""A recorded clip through the calibration's measurement: the sidecar's
segments name the steps, the anchor is reset on the still step, and every
frame comes out as a sample with his, the engine's and the avatar's
readings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sha256.calibration.offline import samples_from_clip, steps_from_sidecar
from sha256.core.types import RenderedFrame, VideoFrame
from tests.unit.test_calibration_session import FakeEngine, FakeOutputTracker


class AnimatingEngine(FakeEngine):
    def animate(self, avatar_id: str, tracking, frame: VideoFrame) -> RenderedFrame:
        return RenderedFrame(
            data=frame.data, pixel_format=frame.pixel_format, metadata=frame.metadata
        )


class Tracker(FakeOutputTracker):
    pass


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "driving.mkv"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 30.0, (64, 48))
    assert writer.isOpened()
    for i in range(30):
        writer.write(np.full((48, 64, 3), i * 5, dtype=np.uint8))
    writer.release()
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "video": path.name,
                "fps": 30.0,
                "frames": 30,
                "segments": [
                    {
                        "instruction": "SIT STILL. Look straight.",
                        "first_frame": 0,
                        "last_frame": 14,
                    },
                    {"instruction": "TALK. Say anything.", "first_frame": 15, "last_frame": 24},
                    {"instruction": "Look UP, then DOWN.", "first_frame": 25, "last_frame": 29},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


class TestOffline:
    def test_the_sidecar_names_the_steps(self, clip: Path):
        segments, kind_of = steps_from_sidecar(clip.with_suffix(".json"))
        assert len(segments) == 3
        assert [kind_of(i) for i in (0, 14, 15, 24, 25, 29, 40)] == [
            "rest",
            "rest",
            "talk",
            "talk",
            "other",
            "other",
            "other",
        ]

    def test_every_frame_becomes_a_sample(self, clip: Path):
        engine = AnimatingEngine()
        driver, output = Tracker(), Tracker()
        seen: list[int] = []

        samples = samples_from_clip(
            clip,
            tracker=driver,
            engine=engine,
            avatar_id="me",
            output_tracker=output,
            on_progress=seen.append,
        )

        assert len(samples) == 30
        assert [s.step for s in samples[:15]] == ["rest"] * 15
        assert [s.step for s in samples[15:25]] == ["talk"] * 10
        assert all(s.face and s.his and s.engine and s.avatar for s in samples)
        assert engine.resets == ["me"] and engine.readings == 30
        assert samples[7].timestamp == pytest.approx(7 / 30.0)
        assert len(output.seen) == 30
        assert seen == []  # progress every 100 frames; the clip is shorter

    def test_last_stops_early_and_no_output_tracker_leaves_avatar_unread(self, clip: Path):
        samples = samples_from_clip(
            clip, tracker=Tracker(), engine=AnimatingEngine(), avatar_id="me", last=9
        )
        assert len(samples) == 10 and all(s.avatar is None for s in samples)

    def test_a_missing_clip_is_an_error(self, tmp_path: Path):
        with pytest.raises(RuntimeError):
            samples_from_clip(
                tmp_path / "none.mkv",
                tracker=Tracker(),
                engine=AnimatingEngine(),
                avatar_id="me",
                kind_of=lambda i: "rest",
            )
