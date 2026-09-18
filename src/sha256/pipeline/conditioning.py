"""What goes out to the virtual camera after the avatar is rendered.

Two things the other side of a call must never see: a defect of the avatar,
and the owner's own face. His words (2026-09-09, 2026-09-11): the app is
shown by his friends to their boss, and a failure in front of the boss must
look like the internet or the camera, not like the app. Two levers, both on
the one frame that leaves the pipeline, in :class:`OutputConditioner`:

**Quality.** The published picture is scaled down to a chosen height, put
through a JPEG at a quality to match, and scaled straight back up to the
camera's own size, so the virtual camera's format never changes -- Zoom and
Meet see one stream -- and the picture looks like a cheaper webcam's on a
thinner line: at 480p the mask's edge, a stubble mismatch and the small
defects of a generated face are below what the picture can carry. The JPEG
is not decoration: a plain down-and-up scale of a softly lit face is nearly
invisible (his first look, 2026-09-11: "I set 360p and honestly nothing
changed"; measured, the Laplacian sharpness fell 38 -> 6 but the mean
difference was under one grey level), while the block artefacts of a
compressed small picture are what everyone recognises as a bad camera or a
bad connection. "native" is the camera's own size, untouched.

**The black screen.** A black frame with a line of text -- "Camera battery
died" -- in place of the avatar, by hand (a button, a hotkey) and on its own
when the engine cannot work: when the frame that would go out is his own
face (the engine's passthrough, before a face is first found or after it is
lost with nothing rendered yet), or when the face has been lost long enough
that a frozen avatar over a moving room would give the game away. A dead
battery cuts at once and does not come back in a second, so the screen
appears immediately, stays at least a few seconds, and the avatar returns
only after the face has been tracked steadily for a moment -- hysteresis,
never a flicker. The manual switch overrides all of it and comes back the
instant it is released: that is his call, not a guess.

The preview shows the same frame the sink gets, so what he sees is what the
other side sees; the window says why the screen is black.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Literal

import numpy as np

from sha256.core.types import RenderedFrame
from sha256.utils.logging import get_logger
from sha256.utils.timing import now

_LOG = get_logger(__name__)

OutputQuality = Literal["native", "1080p", "720p", "480p", "360p"]

# Per level: the height the picture is taken down to before it is scaled
# back (never up -- a frame already smaller than the level keeps its size),
# and the JPEG quality it is passed through at that size. ``native`` touches
# nothing. The qualities were chosen on a real frame of his (recordings/
# day15-quality-sheet.png): 480p/50 reads as an ordinary webcam, 360p/40 as
# a poor connection, and 1080p/90 and 720p/65 as the same camera on a
# slightly thinner line (on a 720p source neither resizes; the JPEG alone
# is the step, 3.5-3.8 ms a frame; 480p 2.6 ms, 360p 1.6 ms).
QUALITY_LEVELS: dict[str, tuple[int | None, int | None]] = {
    "native": (None, None),
    "1080p": (1080, 90),
    "720p": (720, 65),
    "480p": (480, 50),
    "360p": (360, 40),
}
QUALITY_NAMES: tuple[str, ...] = tuple(QUALITY_LEVELS)

DEFAULT_FAILURE_TEXT = "Camera battery died"

# Day 19 (the client's "the face is lost when the head turns"): a face lost
# with the head turned this far -- yaw or pitch, read off the last frame the
# tracker saw a face on -- is the head past what a webcam tracker reads
# (MediaPipe lets go around 60 degrees of yaw; his live look of 2026-09-15
# lost 143 of 300 frames that way and blacked the call for 22 seconds), not
# a face in front of the camera the tracker failed on. Such a loss holds the
# last frame that went out for DEFAULT_LOST_FACE_SECONDS_TURNED before the
# black screen: a look at someone in the room, not a dead camera.
TURNED_AWAY_DEGREES = 25.0
DEFAULT_LOST_FACE_SECONDS_TURNED = 8.0

# The black frames, one per size and text: built once, read many times, and
# never written to by anyone (the sink copies what it sends).
_FAILURE_FRAMES: dict[tuple[int, int, str], np.ndarray] = {}


def apply_quality(data: np.ndarray, quality: str) -> np.ndarray:
    """``data`` at ``quality``: scaled down to the level's height (INTER_AREA,
    only when the frame is taller), passed through a JPEG at the level's
    quality, and scaled back up to its own size (INTER_LINEAR) -- the
    cheap-webcam look; or ``data`` itself, the same object, at native."""
    if quality not in QUALITY_LEVELS:
        msg = f"unknown output quality {quality!r}; one of {', '.join(QUALITY_NAMES)}"
        raise ValueError(msg)
    target, jpeg = QUALITY_LEVELS[quality]
    if target is None:
        return data

    import cv2

    height, width = data.shape[:2]
    out = data
    if target < height:
        small_width = max(1, round(width * target / height))
        out = cv2.resize(out, (small_width, target), interpolation=cv2.INTER_AREA)
    if jpeg is not None:
        encoded, buffer = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg)])
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR) if encoded else None
        if decoded is not None:
            out = decoded
    if out.shape[:2] != (height, width):
        out = cv2.resize(out, (width, height), interpolation=cv2.INTER_LINEAR)
    restored: np.ndarray = out
    return restored


def failure_frame(width: int, height: int, text: str) -> np.ndarray:
    """A black ``height x width x 3`` frame with ``text`` centred in a quiet
    grey -- what a camera that died would leave on the other side. Cached per
    size and text; the array is shared, so callers treat it as read-only."""
    key = (int(width), int(height), text)
    cached = _FAILURE_FRAMES.get(key)
    if cached is not None:
        return cached

    canvas = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    if text:
        import cv2

        scale = max(height / 720.0, 0.3) * 1.1
        thickness = max(1, round(scale * 2))
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        x = max((width - text_width) // 2, 0)
        y = (height + text_height) // 2
        cv2.putText(
            canvas,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (176, 176, 176),
            thickness,
            cv2.LINE_AA,
        )
    _FAILURE_FRAMES[key] = canvas
    return canvas


class OutputConditioner:
    """The last stage before the sink: the quality, and the black screen.

    ``quality`` is one of :data:`QUALITY_NAMES`. ``failure_text`` is the line
    on the black screen. ``auto_blackout`` off keeps the screen for the
    manual switch alone -- a developer's setting; on (the default) the
    engine's passthrough blacks the screen at once and a face lost for
    ``lost_face_seconds`` blacks it too; an automatic black screen stays
    ``min_blackout_seconds`` at least and lifts only after the face has been
    tracked for ``recovery_seconds`` without a break. ``clock`` is monotonic
    seconds, replaceable for tests.

    The manual switch and the quality are set from the UI thread and read on
    the pipeline thread; both are single attribute writes, which is all the
    synchronisation a bool and a short string need.
    """

    def __init__(
        self,
        *,
        quality: str = "native",
        failure_text: str = DEFAULT_FAILURE_TEXT,
        auto_blackout: bool = True,
        lost_face_seconds: float = 1.5,
        lost_face_seconds_turned: float = DEFAULT_LOST_FACE_SECONDS_TURNED,
        min_blackout_seconds: float = 3.0,
        recovery_seconds: float = 1.0,
        clock: Callable[[], float] = now,
    ) -> None:
        self.set_quality(quality)
        self._text = failure_text
        self._auto = auto_blackout
        self._lost_after = float(lost_face_seconds)
        self._lost_after_turned = float(lost_face_seconds_turned)
        self._min_black = float(min_blackout_seconds)
        self._recover_after = float(recovery_seconds)
        self._clock = clock
        self._manual = False
        self._lost_since: float | None = None
        self._stable_since: float | None = None
        self._black_since: float | None = None
        self._reason: str | None = None
        # Day 19: the turned-away loss -- decided on the loss's first frame
        # and kept for its length, and the last frame that went out with a
        # face, which stands in for the avatar while it lasts.
        self._lost_turned = False
        self._held: RenderedFrame | None = None

    # --- the switches --------------------------------------------------------

    @property
    def quality(self) -> str:
        return self._quality

    def set_quality(self, quality: str) -> None:
        if quality not in QUALITY_LEVELS:
            msg = f"unknown output quality {quality!r}; one of {', '.join(QUALITY_NAMES)}"
            raise ValueError(msg)
        self._quality = quality

    @property
    def failure_text(self) -> str:
        return self._text

    @property
    def manual(self) -> bool:
        """Whether the black screen is switched on by hand."""
        return self._manual

    def set_manual(self, on: bool) -> None:
        if on != self._manual:
            _LOG.info("Black screen switched %s by hand", "on" if on else "off")
        self._manual = bool(on)
        if not on:
            self._reason = None

    @property
    def auto_blackout(self) -> bool:
        return self._auto

    @property
    def blackout(self) -> str | None:
        """Why the screen is black right now -- ``"manual"``, or the automatic
        trigger's own words -- or ``None`` while the avatar is going out."""
        if self._manual:
            return "manual"
        return self._reason if self._black_since is not None else None

    @property
    def holding(self) -> str | None:
        """Why the last frame is standing in for the avatar right now -- the
        face lost while the head was turned away (see
        :data:`TURNED_AWAY_DEGREES`) -- or ``None`` while the avatar itself,
        or the black screen, is going out."""
        if self._manual or self._black_since is not None or not self._auto:
            return None
        if self._lost_since is not None and self._lost_turned and self._held is not None:
            return "face lost, turned away"
        return None

    def reset(self) -> None:
        """Forget the automatic state (a pipeline start). The manual switch is
        his and stays where he left it."""
        self._lost_since = None
        self._stable_since = None
        self._black_since = None
        self._lost_turned = False
        self._held = None
        if not self._manual:
            self._reason = None

    # --- the frame -----------------------------------------------------------

    def failure_frame_for(self, width: int, height: int) -> np.ndarray:
        return failure_frame(width, height, self._text)

    def condition(
        self,
        frame: RenderedFrame,
        *,
        face_detected: bool,
        passthrough: bool,
        turned_away: bool = False,
    ) -> RenderedFrame:
        """The frame that goes out for ``frame``: the black screen when the
        manual switch is on or an automatic trigger holds, else ``frame`` at
        the chosen quality. ``passthrough`` says the frame would show the
        camera's own picture (the engine had no face to draw).

        ``turned_away`` says the last face the tracker saw was turned past
        :data:`TURNED_AWAY_DEGREES` (day 19). A face lost that way is not the
        tracker failing on a face in front of it but the head past what any
        webcam tracker reads -- a glance at someone in the room -- so the
        last frame that went out with a face stays on the screen for
        ``lost_face_seconds_turned`` before the black screen, instead of the
        engine's own frame (the last face pasted over the live picture) for
        ``lost_face_seconds``. Read on the loss's first frame; the flag on
        the frames after is ignored.
        """
        if self._blackout_now(
            face_detected=face_detected, passthrough=passthrough, turned_away=turned_away
        ):
            return replace(
                frame,
                data=self.failure_frame_for(frame.width, frame.height),
                avatar_id=None,
                is_placeholder=True,
            )
        held = self._held
        if self.holding is not None and held is not None:
            return replace(frame, data=held.data, avatar_id=held.avatar_id, is_placeholder=False)
        data = apply_quality(frame.data, self._quality)
        out = frame if data is frame.data else replace(frame, data=data)
        if face_detected and not passthrough and not frame.is_placeholder:
            self._remember(out)
        return out

    def _remember(self, out: RenderedFrame) -> None:
        """Keep a copy of the frame that just went out. A copy, not the
        frame: its pixels are the camera's buffer with the face pasted on,
        and a capture backend may reuse that buffer for the next frame (the
        rule on ``VideoFrame``). One buffer, refilled -- a 720p copy is well
        under a millisecond."""
        buffer = None if self._held is None else self._held.data
        if buffer is None or buffer.shape != out.data.shape or buffer.dtype != out.data.dtype:
            buffer = np.empty_like(out.data)
        np.copyto(buffer, out.data)
        self._held = replace(out, data=buffer)

    def _blackout_now(
        self, *, face_detected: bool, passthrough: bool, turned_away: bool = False
    ) -> bool:
        if self._manual:
            return True
        if not self._auto:
            self._reason = None
            return False
        t = self._clock()
        if face_detected and not passthrough:
            self._lost_since = None
            self._lost_turned = False
            if self._stable_since is None:
                self._stable_since = t
        else:
            self._stable_since = None
            if self._lost_since is None:
                self._lost_since = t
                self._lost_turned = bool(turned_away) and not passthrough
                if self._lost_turned:
                    _LOG.info(
                        "Face lost while turned away: the last frame is held for %.1f s",
                        self._lost_after_turned,
                    )

        if self._black_since is None:
            patience = self._lost_after_turned if self._lost_turned else self._lost_after
            lost_long = self._lost_since is not None and t - self._lost_since >= patience
            if passthrough or lost_long:
                self._black_since = t
                self._reason = (
                    "the camera's own picture would have gone out"
                    if passthrough
                    else ("face lost, turned away" if self._lost_turned else "face lost")
                )
                _LOG.info("Black screen on: %s", self._reason)
                return True
            return False

        held = t - self._black_since
        stable = self._stable_since is not None and t - self._stable_since >= self._recover_after
        if held >= self._min_black and stable:
            _LOG.info("Black screen off: the face is back")
            self._black_since = None
            self._reason = None
            return False
        return True
