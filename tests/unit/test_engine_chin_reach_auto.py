"""``mask_chin_reach = "auto"`` (day 16, B3): the reach is the mask's own
clearance above the chin point -- the erosion's radius plus half the
feather, of the crop face width -- resolved at load like a number; on his
portrait it is the hand value 0.04 to within a tenth of a pixel."""

from __future__ import annotations

import pytest

from sha256.avatar.engine_liveportrait import (
    _CHIN_DROP_MAX,
    _OVAL_FEATHER,
    _OVAL_SHRINK,
    _auto_chin_reach,
    _chin_reach_is_auto,
    _mask_chin_reach,
    chin_room,
)
from tests.unit.test_portrait_preparation import synthetic_face


class TestAutoChinReach:
    def test_auto_reads_as_zero_as_a_fraction_and_is_recognised(self):
        assert _mask_chin_reach({"mask_chin_reach": "auto"}) == 0.0
        assert _chin_reach_is_auto({"mask_chin_reach": "auto"})
        assert not _chin_reach_is_auto({"mask_chin_reach": 0.04}) and not _chin_reach_is_auto({})
        with pytest.raises(ValueError, match="auto"):
            _mask_chin_reach({"mask_chin_reach": "later"})

    def test_the_clearance_is_the_masks_own(self):
        crop = synthetic_face((256.0, 256.0), 233.0)[:468, :2]
        width = float(crop[:, 0].max() - crop[:, 0].min())
        assert _auto_chin_reach(crop) == pytest.approx(
            width * (_OVAL_SHRINK / 2.0 + _OVAL_FEATHER / 2.0)
        )

    def test_chin_room_resolves_auto_and_the_drop_takes_what_is_left(self):
        pts = synthetic_face((512.0, 600.0), 400.0)
        auto = chin_room(pts, chin_reach_fraction="auto")
        plain = chin_room(pts)
        assert auto.headroom == plain.headroom
        assert 0.0 < auto.chin_reach < auto.headroom
        # About 0.04 of the face height, the hand value.
        assert auto.chin_reach / auto.crop_face_height == pytest.approx(0.04, abs=0.01)
        assert auto.chin_drop_max == pytest.approx(
            min(_CHIN_DROP_MAX * auto.crop_face_height, auto.headroom - auto.chin_reach)
        )
        tight = chin_room(pts, chin_reach_fraction="auto")
        assert tight.chin_reach <= tight.headroom
