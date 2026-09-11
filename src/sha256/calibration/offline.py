"""A recorded clip through the same measurement as the live run.

Two uses. A calibration session's own ``clip.mkv`` can be re-measured
after any change to the engine without the person present -- that is what
the session records itself for. And the owner's guided clips
(``recordings/driving.mkv`` and the rest, from
``scripts/record_driving_clip.py``) are the same movements under other
names, so his hand-set values of day 15 can be checked against what the
calibration would have set from the same frames (the ROADMAP's own bar
for A2). The sidecar's segments name the steps through
:func:`sha256.calibration.script.step_for_instruction`.

The frames go through the app's own filtered tracker (what drives the
engine live), the engine, and an unfiltered second tracker on the output
-- ``scripts/measure_follow.py``'s arrangement -- and come out as the
:class:`Sample` list :func:`sha256.calibration.measure.calibrate` reads.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from sha256.calibration.samples import Sample, avatar_measures, his_measures
from sha256.calibration.script import step_for_instruction
from sha256.core.enums import PixelFormat
from sha256.core.interfaces import FaceTracker
from sha256.core.types import FrameMetadata, VideoFrame
from sha256.utils.logging import get_logger

__all__ = ["samples_from_clip", "steps_from_sidecar"]

_LOG = get_logger(__name__)


def steps_from_sidecar(sidecar: Path) -> tuple[list[dict[str, Any]], Callable[[int], str]]:
    """The sidecar's segments, and a function from frame index to step
    kind (``"other"`` outside every segment)."""
    data = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    segments: list[dict[str, Any]] = list(data.get("segments", []))
    table = [
        (int(s["first_frame"]), int(s["last_frame"]), step_for_instruction(str(s["instruction"])))
        for s in segments
    ]

    def kind_of(index: int) -> str:
        for first, last, kind in table:
            if first <= index <= last:
                return kind
        return "other"

    return segments, kind_of


def samples_from_clip(
    clip: Path,
    *,
    tracker: FaceTracker,
    engine: Any,
    avatar_id: str,
    output_tracker: FaceTracker | None = None,
    kind_of: Callable[[int], str] | None = None,
    last: int | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> list[Sample]:
    """Every frame of ``clip`` as a :class:`Sample`. ``kind_of`` maps a frame
    index to its step; without it the clip's ``.json`` sidecar is read.
    The engine's anchor is reset on entering a ``rest`` step, as the live
    session does. ``last`` stops after that frame index."""
    import cv2

    clip = Path(clip)
    if kind_of is None:
        _, kind_of = steps_from_sidecar(clip.with_suffix(".json"))
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        msg = f"{clip} could not be opened"
        raise RuntimeError(msg)
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    read = getattr(engine, "frame_readings", None)
    reset = getattr(engine, "reset_anchor", None)
    samples: list[Sample] = []
    previous = None
    index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok or (last is not None and index > last):
                break
            kind = kind_of(index)
            if kind != previous:
                if kind == "rest" and reset is not None:
                    reset(avatar_id)
                previous = kind
            height, width = image.shape[:2]
            frame = VideoFrame(
                data=np.ascontiguousarray(image),
                pixel_format=PixelFormat.BGR24,
                metadata=FrameMetadata(frame_id=index, capture_timestamp=index / fps),
            )
            tracking = tracker.process(frame)
            out = engine.animate(avatar_id, tracking, frame)
            readings = dict(read(avatar_id) or {}) if read is not None else {}
            his = his_measures(tracking, width, height)
            sample = Sample(
                frame_id=index,
                timestamp=index / fps,
                step=kind,
                face=bool(tracking.face_detected and his),
                his=his,
                engine=readings,
            )
            if output_tracker is not None:
                got = output_tracker.process(
                    VideoFrame(
                        data=np.ascontiguousarray(out.data),
                        pixel_format=out.pixel_format,
                        metadata=frame.metadata,
                    )
                )
                measures = avatar_measures(got, width, height)
                sample.avatar = measures if measures is not None else {}
            samples.append(sample)
            index += 1
            if on_progress is not None and index % 100 == 0:
                on_progress(index)
    finally:
        capture.release()
    _LOG.info("%s: %d frames measured", clip.name, len(samples))
    return samples
