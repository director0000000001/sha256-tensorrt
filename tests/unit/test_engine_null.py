"""Tests for the placeholder avatar engine.

Just the anchor contract added 2026-09-05 (``reset_anchor``/``rest_status``
on :class:`AvatarEngine`, day 5 of the neural-engine work). The rest of this
engine is exercised indirectly through :mod:`test_avatar_manager`, where it
stands in as a working, inspectable fake for a real engine.
"""

from __future__ import annotations

from sha256.avatar.engine_null import NullAvatarEngine
from sha256.core.interfaces import AvatarEngine


def test_satisfies_the_avatar_engine_protocol():
    assert isinstance(NullAvatarEngine(), AvatarEngine)


def test_reset_anchor_and_rest_status_are_no_ops():
    """This engine synthesises from tracking alone; nothing is learned to
    re-anchor -- see the LivePortrait engine for the one that isn't."""
    engine = NullAvatarEngine()
    engine.reset_anchor("anything")  # must not raise
    assert engine.rest_status("anything") is None
