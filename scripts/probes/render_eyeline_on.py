"""Render driving.mkv with the eye-line crop centre forced fully on at any
pitch (what the live fade does once his pitch is 8 degrees off his anchor's)
-- to see the toothy smile under it."""

import runpy
import sys

import sha256.avatar.engine_liveportrait as eng

eng._CROP_CENTRE_PITCH = (-1.0, -0.5)
sys.argv = [
    "render_liveportrait.py",
    "--clip",
    "recordings/driving.mkv",
    "--first",
    "0",
    "--last",
    "970",
    "--no-warp",
    "--out",
    "recordings/day40-smile-eyeline.mp4",
    "--stills",
    "925,968",
]
runpy.run_path("scripts/render_liveportrait.py", run_name="__main__")
