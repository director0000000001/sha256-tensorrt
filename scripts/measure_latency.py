"""How many milliseconds behind you is the picture? Measured with the screen itself.

**Why this exists.** Every number this project has measured about the mouth is a
frame-by-frame comparison: which photograph was chosen, how far it sits from the
mouth it was given, how often it changes. All of them are computed *in step* —
frame 100 against frame 100 — so not one of them can see a delay. And a delay is
exactly what "it does not react" means: the owner, 2026-09-03, after the grid,
the filter and the hold had each been measured, judged and left as they were:
*"it needs more speed, the application needs a much better reaction to the
face's changes."*

**The trick, and why no second camera is needed.** The window shows a large
millisecond counter on the left and the camera's own picture on the right. Point
the camera at the screen and the camera panel shows *the counter as it was when
that frame left the sensor*. Both numbers are in one image, so the difference
between them is the whole round trip that happens before the engine runs: the
display's own latency, the camera's exposure, and the transfer over USB. The
application's own work — 31 ms a frame, measured on this machine — sits on top.

The counter is drawn by this script, so "now" is exact rather than read off
another program's clock.

**Qt rather than `cv2.imshow`**: the OpenCV wheel installed here is the headless
build and has no window support at all. Qt is the application's own toolkit, so
this costs no new dependency.

Usage::

    uv run python scripts/measure_latency.py

Point the camera at the big number, wait, and it writes ten snapshots into
``recordings/latency/``. Nothing is uploaded and nothing is judged here — the
snapshots are read afterwards, one pair of numbers at a time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sha256.camera.opencv_capture import OpenCVCameraProvider  # noqa: E402
from sha256.config.loader import load_settings  # noqa: E402
from sha256.config.paths import AppPaths  # noqa: E402

PANEL = (640, 480)
SNAPSHOTS = 10
SNAPSHOT_EVERY = 1.0
SETTLE = 3.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "recordings" / "latency",
        help="Where the snapshots go. Gitignored.",
    )
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    import cv2
    import numpy as np
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import QApplication, QLabel

    paths = AppPaths.resolve()
    settings = load_settings(paths)
    camera = OpenCVCameraProvider(
        device_index=settings.camera.device_index,
        width=settings.camera.width,
        height=settings.camera.height,
        target_fps=settings.camera.target_fps,
        use_mjpeg=settings.camera.use_mjpeg,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    print("=" * 68)
    print("Measuring how far behind the camera's picture is")
    print("=" * 68)
    print("  1. Point the camera at THIS SCREEN, at the big number on the left.")
    print("  2. Move it until that number is readable inside the right-hand panel.")
    print("  3. Hold it there. Ten snapshots are saved and it closes on its own.")
    print()

    camera.start()
    app = QApplication(sys.argv[:1])
    label = QLabel()
    label.setWindowTitle("sha256 — latency")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.resize(PANEL[0] * 2, PANEL[1])
    label.show()

    state = {"saved": 0, "next": time.perf_counter() + SETTLE, "started": time.perf_counter()}

    def tick() -> None:
        frame = camera.read()
        now = time.perf_counter()
        elapsed_ms = int((now - state["started"]) * 1000)

        left = np.full((PANEL[1], PANEL[0], 3), 12, np.uint8)
        cv2.putText(
            left,
            f"{elapsed_ms % 100000:05d}",
            (18, PANEL[1] // 2 + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            4.2,
            (255, 255, 255),
            9,
            cv2.LINE_AA,
        )
        cv2.putText(
            left,
            "point the camera here",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (150, 150, 150),
            2,
            cv2.LINE_AA,
        )
        right = (
            cv2.resize(frame.data, PANEL)
            if frame is not None
            else np.full((PANEL[1], PANEL[0], 3), 40, np.uint8)
        )
        canvas = np.hstack([left, right])

        image = QImage(
            canvas.data,
            canvas.shape[1],
            canvas.shape[0],
            canvas.strides[0],
            QImage.Format.Format_BGR888,
        )
        label.setPixmap(QPixmap.fromImage(image.copy()))

        if frame is not None and now >= state["next"]:
            path = args.out / f"latency-{state['saved']:02d}-{elapsed_ms:05d}.png"
            cv2.imwrite(str(path), canvas)
            state["saved"] += 1
            state["next"] = now + SNAPSHOT_EVERY
            print(f"  saved {path.name}   (now = {elapsed_ms} ms)")

        if state["saved"] >= SNAPSHOTS or now - state["started"] > args.seconds:
            timer.stop()
            print()
            print(f"{state['saved']} snapshot(s) in {args.out}")
            print("In each: the number on the left is now; the number the camera sees is")
            print("when that frame left the sensor. The difference is the delay before")
            print("the engine has even started work.")
            app.quit()

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(16)
    app.exec()
    # Stopped here and only here: the timer callback must not close the camera
    # it is still reading from on the way out.
    camera.stop()
    return 0 if state["saved"] else 2


if __name__ == "__main__":
    sys.exit(main())
