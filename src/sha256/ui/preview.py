"""Live video preview.

Shows what the pipeline is doing, at the rate the pipeline is doing it. Two
views, because at this stage they answer different questions:

*   **Camera + tracking** — the camera image with the tracked landmarks drawn
    on it. This is the honest answer to "is it working": you can see the mesh
    follow your face, and see it fail when it fails.
*   **Avatar output** — the frame that will go to the virtual camera. Until a
    real engine exists this is the placeholder, which is worth knowing rather
    than being surprised by.

**The widget paints nothing but the video.** No background, no border, no
placeholder text. An opaque widget here would cover the window background
image, and it is sized to expand, so it would cover nearly all of it. Only the
video rectangle itself is drawn; everything around it is whatever the parent
painted.

The widget owns no timing and no pipeline state. It is handed a frame and
paints it; the window decides when.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from sha256.core.enums import PixelFormat
from sha256.core.types import FaceTrackingResult
from sha256.ui.head_guide import GuideFit, HeadGuide

__all__ = ["PreviewWidget"]

_LANDMARK_COLOUR = QColor(64, 224, 160)
_FRAME_BORDER = QColor(255, 255, 255, 40)
# Day 21: the head guide's oval -- green with the head in it, amber until.
_GUIDE_GOOD = QColor(64, 224, 96)
_GUIDE_WAIT = QColor(255, 190, 40)
_GUIDE_SHADOW = QColor(0, 0, 0, 190)

# Drawing all 478 points at 30 fps is visual noise as much as it is work. Every
# third point still reads as a mesh on a face and costs a third as much.
_LANDMARK_STRIDE = 3


class PreviewWidget(QWidget):
    """Paints pipeline frames, scaled to fit, over whatever is behind it."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, on=True)

        self._pixmap: QPixmap | None = None
        self._show_landmarks = True
        self._mirror = True

    # --- configuration -----------------------------------------------------

    @property
    def has_frame(self) -> bool:
        return self._pixmap is not None

    def set_show_landmarks(self, enabled: bool) -> None:
        self._show_landmarks = enabled

    def set_mirror(self, enabled: bool) -> None:
        """Mirror the preview.

        On by default because a preview that does not mirror feels wrong to
        everyone who has ever used a webcam — you move left and the image moves
        right. It affects **only** what is displayed here, never what tracking
        receives or what the virtual camera publishes.
        """
        self._mirror = enabled

    def clear_frame(self) -> None:
        self._pixmap = None
        self.update()

    # --- input -------------------------------------------------------------

    def show_frame(
        self,
        data: np.ndarray,
        pixel_format: PixelFormat,
        tracking: FaceTrackingResult | None = None,
        *,
        mirror: bool | None = None,
        guide: tuple[HeadGuide, GuideFit] | None = None,
    ) -> None:
        """Accept one frame, optionally with its tracking overlay.

        ``mirror`` overrides the widget's setting for this one frame: the
        black screen carries words, and words mirrored read backwards (his
        look of 2026-09-11), so the window shows that frame as the other
        side sees it whatever the mirror is set to.

        ``guide`` (day 21, see sha256.ui.head_guide) draws the head guide's
        oval and its one line over the frame, in the fit's colour.
        """
        image = _to_qimage(data, pixel_format)
        if image is None:
            return

        mirrored = self._mirror if mirror is None else mirror
        if mirrored:
            image = image.flipped(Qt.Orientation.Horizontal)

        pixmap = QPixmap.fromImage(image)

        if self._show_landmarks and tracking is not None and tracking.face_detected:
            self._draw_landmarks(pixmap, tracking, mirrored)
        if guide is not None:
            self._draw_guide(pixmap, guide[0], guide[1], mirrored)

        self._pixmap = pixmap
        self.update()

    def _draw_guide(self, pixmap: QPixmap, guide: HeadGuide, fit: GuideFit, mirrored: bool) -> None:
        """The oval where the head should be, and the fit's line under it.
        Everything is a fraction of the frame's height, so the oval is the
        same head on 720p and on 1080p; the centre's x is mirrored with
        the picture."""
        width = pixmap.width()
        height = pixmap.height()
        centre_x = (1.0 - guide.centre[0]) if mirrored else guide.centre[0]
        colour = _GUIDE_GOOD if fit.good else _GUIDE_WAIT
        radius_x = guide.oval_width * height / 2.0
        radius_y = guide.oval_height * height / 2.0
        centre = QPointF(centre_x * width, guide.centre[1] * height)

        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, on=True)
            painter.setPen(QPen(colour, max(3.0, height / 180.0)))
            painter.drawEllipse(centre, radius_x, radius_y)
            font = painter.font()
            font.setPointSizeF(max(10.0, height / 22.0))
            font.setBold(True)
            painter.setFont(font)
            top = centre.y() + radius_y + height * 0.015
            box = QRectF(0.0, top, float(width), height * 0.12)
            flags = int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            painter.setPen(QPen(_GUIDE_SHADOW, 1.0))
            painter.drawText(box.translated(2.0, 2.0), flags, fit.message)
            painter.setPen(QPen(colour, 1.0))
            painter.drawText(box, flags, fit.message)
        finally:
            painter.end()

    def _draw_landmarks(
        self, pixmap: QPixmap, tracking: FaceTrackingResult, mirrored: bool | None = None
    ) -> None:
        landmarks = tracking.landmarks
        if landmarks is None or landmarks.count == 0:
            return

        width = pixmap.width()
        height = pixmap.height()
        points = landmarks.points[::_LANDMARK_STRIDE]
        flip = self._mirror if mirrored is None else mirrored

        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, on=True)
            painter.setPen(QPen(_LANDMARK_COLOUR, 2.0))
            # Landmarks are normalised to the source frame, so they scale to
            # whatever the pixmap happens to be without knowing its resolution.
            for x, y, _z in points:
                screen_x = (1.0 - float(x)) * width if flip else float(x) * width
                painter.drawPoint(QPointF(screen_x, float(y) * height))
        finally:
            painter.end()

    # --- painting ----------------------------------------------------------

    def video_rect(self) -> QRect:
        """Where the video is drawn, centred and aspect-preserved."""
        if self._pixmap is None:
            return QRect()

        scaled = self._pixmap.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRect(
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2,
            scaled.width(),
            scaled.height(),
        )

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt naming
        del event
        if self._pixmap is None:
            return

        target = self.video_rect()
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, on=True)
            painter.drawPixmap(target, self._pixmap)
            # A hairline edge so the video reads as a panel sitting on the
            # background rather than a hole cut out of it.
            painter.setPen(QPen(_FRAME_BORDER, 1))
            painter.drawRect(target.adjusted(0, 0, -1, -1))
        finally:
            painter.end()


def _to_qimage(data: np.ndarray, pixel_format: PixelFormat) -> QImage | None:
    """Wrap a frame buffer as a QImage that owns its pixels.

    The copy is not optional. Frame buffers are borrowed — the capture backend
    may be writing the next frame into the same memory while Qt is painting —
    and a QImage built on a foreign buffer that disappears is a crash, not a
    glitch.
    """
    if data.ndim != 3 or data.shape[2] != 3:
        return None

    height, width = data.shape[:2]
    if pixel_format is PixelFormat.BGR24:
        image_format = QImage.Format.Format_BGR888
    elif pixel_format is PixelFormat.RGB24:
        image_format = QImage.Format.Format_RGB888
    else:
        return None

    contiguous = np.ascontiguousarray(data)
    return QImage(contiguous.tobytes(), width, height, width * 3, image_format)
