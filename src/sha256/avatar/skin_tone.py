"""The person's skin colour, read from the camera, for the avatar's prompt
(day 23, 2026-10-16, his idea: "one button that finds the person's skin
colour; the prompts say [your skin color] and that is replaced by it").

Why words and not only a code. An image generator follows words ("a
light-medium skin with a warm, golden undertone") far more faithfully
than a hex code, so the reading is said both ways: the lightness band
from the ITA angle (the Individual Typology Angle of dermatology --
``atan((L* - 50) / b*)`` in CIELAB, the bands of Chardon et al.), the
undertone from the hue angle in the same space, and the sRGB code after
them for the generators that read one.

What is read. The flat skin only: both cheeks (the importer's own
points) and the middle of the forehead, each a patch of a twentieth of
the face's width; in a patch the dark cells (hair, a brow's edge, a
shadow: under :data:`_DARK` of the patch's median lightness -- day 16's
rule for the edge match) and the brightest twentieth (a highlight) are
dropped, and the per-channel median is the frame's reading. A reading
is the median over :data:`FRAMES_WANTED` such frames with the head
still and facing the camera (the dialog gates them by the head guide).

The reading is the camera's -- the room's light and the camera's white
balance are in it. That is what is wanted: the avatar must match what
the camera sees in the light the calls will have, not a skin colour in
the abstract. The same reader on the generated photograph (the
importer) gives the comparison, :func:`compare`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "FRAMES_WANTED",
    "SKIN_PLACEHOLDER",
    "SkinReading",
    "SkinSampler",
    "compare",
    "fill_prompt",
    "read_skin",
]

SKIN_PLACEHOLDER = "[your skin color]"
FRAMES_WANTED = 30

# MediaPipe points sampled: the cheeks (the importer's), the forehead's
# middle (151: between the brows' top and the hairline, flat skin).
_CHEEKS = (50, 280)
_FOREHEAD = 151
_FACE_POINTS = 468
_PATCH = 0.05  # the patch's half-size, of the face's width
_MIN_PATCH = 3  # pixels
_DARK = 0.55  # under this of the patch's median lightness: hair, shadow
_BRIGHT_QUANTILE = 0.95  # above it: a highlight
_MIN_CELLS = 16

# The ITA bands (degrees), highest first, and the words the prompt gets.
_ITA_BANDS: tuple[tuple[float, str], ...] = (
    (55.0, "very light"),
    (41.0, "light"),
    (28.0, "light-medium"),
    (10.0, "medium (tan)"),
    (-30.0, "brown"),
    (-math.inf, "deep brown"),
)
# The hue angle (degrees, CIELAB a*/b*) that separates the undertones:
# skin sits between about 35 and 75; the golden end is high, the pink
# end low. Rough by nature -- words for a prompt, not a measurement.
_WARM_HUE = 60.0
_COOL_HUE = 45.0

# The comparison's bands (CIE76 distance in CIELAB).
_CLOSE = 8.0
_FAR = 16.0


def _srgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """sRGB (0..255) to CIELAB (D65), the standard formulae."""

    def linear(c: float) -> float:
        c = min(max(c / 255.0, 0.0), 1.0)
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(v) for v in rgb)
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1.0 / 3.0) if t > 216.0 / 24389.0 else (841.0 / 108.0) * t + 4.0 / 29.0

    fx, fy, fz = f(x), f(y), f(z)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


@dataclass(frozen=True)
class SkinReading:
    """One skin colour -- from the camera or from a photograph -- as the
    BGR it was read in, with its words and its code."""

    bgr: tuple[float, float, float]
    frames: int = 1

    @property
    def rgb(self) -> tuple[float, float, float]:
        b, g, r = self.bgr
        return (r, g, b)

    @property
    def hex(self) -> str:
        r, g, b = (round(min(max(v, 0.0), 255.0)) for v in self.rgb)
        return f"#{r:02X}{g:02X}{b:02X}"

    @property
    def lab(self) -> tuple[float, float, float]:
        return _srgb_to_lab(self.rgb)

    @property
    def ita(self) -> float:
        """The Individual Typology Angle, degrees."""
        lum, _, b = self.lab
        return math.degrees(math.atan2(lum - 50.0, max(b, 1e-6)))

    @property
    def tone(self) -> str:
        angle = self.ita
        for floor, words in _ITA_BANDS:
            if angle > floor:
                return words
        return _ITA_BANDS[-1][1]

    @property
    def undertone(self) -> str:
        _, a, b = self.lab
        hue = math.degrees(math.atan2(b, a))
        if hue > _WARM_HUE:
            return "warm, golden"
        if hue < _COOL_HUE:
            return "cool, pinkish"
        return "neutral"

    def describe(self) -> str:
        """The words the prompt gets in place of :data:`SKIN_PLACEHOLDER`."""
        return f"{self.tone} skin with a {self.undertone} undertone, sRGB {self.hex}"

    def distance(self, other: SkinReading) -> float:
        """CIE76: the Euclidean distance in CIELAB."""
        return float(math.dist(self.lab, other.lab))

    def as_dict(self) -> dict[str, Any]:
        return {
            "bgr": [round(float(v), 1) for v in self.bgr],
            "hex": self.hex,
            "tone": self.tone,
            "undertone": self.undertone,
            "frames": self.frames,
        }


def read_skin(frame_bgr: np.ndarray, points_px: np.ndarray) -> tuple[float, float, float] | None:
    """One frame's skin colour (BGR) from the cheeks and the forehead, or
    ``None`` when the patches hold too little usable skin (the face at
    the picture's edge, the patches all hair or shadow)."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or points_px.shape[0] < _FACE_POINTS:
        return None
    pts = np.asarray(points_px, dtype=np.float32)[:_FACE_POINTS, :2]
    height, width = frame_bgr.shape[:2]
    face_w = float(pts[:, 0].max() - pts[:, 0].min())
    half = max(int(_PATCH * face_w), _MIN_PATCH)
    cells: list[np.ndarray] = []
    for index in (*_CHEEKS, _FOREHEAD):
        x, y = int(pts[index, 0]), int(pts[index, 1])
        x0, y0 = max(x - half, 0), max(y - half, 0)
        x1, y1 = min(x + half, width), min(y + half, height)
        if x1 > x0 and y1 > y0:
            cells.append(frame_bgr[y0:y1, x0:x1].reshape(-1, 3))
    if not cells:
        return None
    stacked = np.concatenate(cells).astype(np.float32)
    lum = stacked[:, 0] * 0.114 + stacked[:, 1] * 0.587 + stacked[:, 2] * 0.299
    median = float(np.median(lum))
    bright = float(np.quantile(lum, _BRIGHT_QUANTILE))
    keep = (lum >= _DARK * median) & (lum <= bright)
    if int(keep.sum()) < _MIN_CELLS:
        return None
    b, g, r = np.median(stacked[keep], axis=0)
    return (float(b), float(g), float(r))


class SkinSampler:
    """The frames' readings gathered until :data:`FRAMES_WANTED` of them,
    then their median -- one :class:`SkinReading`."""

    def __init__(self, wanted: int = FRAMES_WANTED) -> None:
        self._wanted = max(int(wanted), 1)
        self._readings: list[tuple[float, float, float]] = []

    @property
    def wanted(self) -> int:
        return self._wanted

    @property
    def count(self) -> int:
        return len(self._readings)

    @property
    def done(self) -> bool:
        return len(self._readings) >= self._wanted

    def add(self, frame_bgr: np.ndarray, points_px: np.ndarray) -> bool:
        """Read one frame; ``True`` when it gave a reading."""
        if self.done:
            return False
        reading = read_skin(frame_bgr, points_px)
        if reading is None:
            return False
        self._readings.append(reading)
        return True

    def reading(self) -> SkinReading | None:
        if not self._readings:
            return None
        b, g, r = np.median(np.asarray(self._readings, dtype=np.float64), axis=0)
        return SkinReading((float(b), float(g), float(r)), frames=len(self._readings))


def fill_prompt(text: str, reading: SkinReading | None) -> str:
    """``text`` with every :data:`SKIN_PLACEHOLDER` replaced by the
    reading's words; unchanged without a reading."""
    if reading is None:
        return text
    return text.replace(SKIN_PLACEHOLDER, reading.describe())


def compare(camera: SkinReading, photograph: SkinReading) -> str:
    """The photograph's skin against the camera's reading, in words: how
    much the live skin match (day 21) will have to do."""
    delta = camera.distance(photograph)
    head = (
        f"your skin on the camera {camera.hex} ({camera.tone}, {camera.undertone}); "
        f"the photograph's {photograph.hex} ({photograph.tone}, {photograph.undertone}); "
        f"distance {delta:.0f}"
    )
    if delta <= _CLOSE:
        return f"{head} -- close; the live skin match has little to do."
    if delta <= _FAR:
        return f"{head} -- noticeably different; the live skin match will carry it."
    return (
        f"{head} -- far; the live skin match is bounded and may not reach it. "
        "Generating the photograph again with the words from your reading is worth it."
    )
