"""Live video preview.

Shows what the pipeline is doing, at the rate the pipeline is doing it. Two
views, because at this stage they answer different questions:

*   **Camera + tracking** — the camera image with the tracked landmarks drawn
    on it. This is the honest answer to "is it working": you can see the mesh
    follow your face, and see it fail when it fails.
*   **Avatar output** — the frame that will go to the virtual camera. Until a
    real engine exists this is the placeholder, which is worth knowing rather
    than being surprised by.

The widget owns no timing and no pipeline state. It is handed a frame and paints
it; the window decides when.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

from sha256.core.enums import PixelFormat
from sha256.core.types import FaceTrackingResult

__all__ = ["PreviewWidget"]

_LANDMARK_COLOUR = QColor(64, 224, 160)
_IDLE_BACKGROUND = QColor(16, 16, 20)
_IDLE_TEXT = "Press Start"

# Drawing all 478 points at 30 fps is visual noise as much as it is work. Every
# third point still reads as a mesh on a face and costs a third as much.
_LANDMARK_STRIDE = 3


class PreviewWidget(QLabel):
    """Displays pipeline frames, scaled to fit, with an optional overlay."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAutoFillBackground(True)
        self.setStyleSheet(
            f"background-color: {_IDLE_BACKGROUND.name()}; color: #8a8a95; font-size: 14px;"
        )
        self.setText(_IDLE_TEXT)

        self._show_landmarks = True
        self._mirror = True

    # --- configuration -----------------------------------------------------

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
        self.setPixmap(QPixmap())
        self.setText(_IDLE_TEXT)

    # --- painting ----------------------------------------------------------

    def show_frame(
        self,
        data: np.ndarray,
        pixel_format: PixelFormat,
        tracking: FaceTrackingResult | None = None,
    ) -> None:
        """Display one frame, optionally with its tracking overlay."""
        image = _to_qimage(data, pixel_format)
        if image is None:
            return

        if self._mirror:
            image = image.flipped(Qt.Orientation.Horizontal)

        pixmap = QPixmap.fromImage(image)

        if self._show_landmarks and tracking is not None and tracking.face_detected:
            self._draw_landmarks(pixmap, tracking)

        self.setPixmap(
            pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _draw_landmarks(self, pixmap: QPixmap, tracking: FaceTrackingResult) -> None:
        landmarks = tracking.landmarks
        if landmarks is None or landmarks.count == 0:
            return

        width = pixmap.width()
        height = pixmap.height()
        points = landmarks.points[::_LANDMARK_STRIDE]

        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, on=True)
            painter.setPen(QPen(_LANDMARK_COLOUR, 2.0))
            # Landmarks are normalised to the source frame, so they scale to
            # whatever the pixmap happens to be without knowing its resolution.
            for x, y, _z in points:
                screen_x = (1.0 - float(x)) * width if self._mirror else float(x) * width
                painter.drawPoint(QPointF(screen_x, float(y) * height))
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
