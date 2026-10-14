"""The in-app instructions (day 16 night): every control on the screen is
named in them, in the order the sections promise."""

from __future__ import annotations

from sha256.app.instructions import INSTRUCTIONS_SECTIONS, instructions_text

CONTROLS = (
    "Start",
    "Stop",
    "Camera",
    "Avatar",
    "Virtual camera",
    "Camera battery died",
    "Ctrl+Alt+B",
    "Menu",
    "View -> Avatar",
    "Tracking (no avatar)",
    "Dots",
    "Mirror",
    "Avatar's hair",
    "Quality",
    "That looked wrong",
    "Re-centre",
    "Person",
    "Nobody",
    "Calibrate me",
    "Stop calibrating",
    "Add avatar",
    "Delete avatar",
    "Instructions",
)


def test_every_control_is_explained():
    text = instructions_text()
    for control in CONTROLS:
        assert control in text, control


def test_no_login_is_explained_any_more():
    """2026-10-09, his ask: Add avatar and Delete avatar are for everyone."""
    text = instructions_text()
    assert "Login" not in text and "passphrase" not in text


def test_the_sections_come_in_order_and_say_what_a_person_asks_first():
    text = instructions_text()
    order = [text.index(f"{n}. ") for n in range(1, 9)]
    assert order == sorted(order)
    assert len(INSTRUCTIONS_SECTIONS) == 8
    for needed in (
        "Windows 10",
        "NVIDIA",
        "720p 30 fps",
        "OBS Virtual Camera",
        "Google Meet",
        "Discord",
        "Zoom",
        "Teams",
        "recordings/",
        "captures/",
        "configs/drivers/",
    ):
        assert needed in text, needed


def test_the_status_lines_are_the_windows_own():
    text = instructions_text()
    for line in (
        "Sit still and look at the camera to centre the avatar",
        "Face not found",
        "Calibrating -- follow the instruction above",
        "BLACK SCREEN",
        "Profile: <name>",
    ):
        assert line in text, line
