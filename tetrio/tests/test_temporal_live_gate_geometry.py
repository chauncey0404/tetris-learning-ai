from __future__ import annotations

import unittest

from tetrio.tools.inspect_vision_temporal_live import (
    localize_candidate,
    tracking_region_for_candidate,
)
from tetrio.vision.layout import PlayfieldCandidate
from tetrio.vision.piece_reader import hold_roi, next_roi


class TemporalLiveGateGeometryTests(unittest.TestCase):
    def _candidate(self):
        return PlayfieldCandidate(
            x=794.0,
            y=229.0,
            w=330.0,
            h=660.0,
            cell_size=33.0,
            grid_score=0.4,
            matched_vertical_lines=11,
            confidence=0.95,
        )

    def test_tracking_region_contains_hold_board_and_next(self):
        candidate = self._candidate()
        region = tracking_region_for_candidate(candidate, 1920, 1080)
        local = localize_candidate(candidate, region)
        shape = (region.height, region.width, 3)

        hx, hy, hw, hh = hold_roi(local, shape)
        nx, ny, nw, nh = next_roi(local, shape)
        bx, by, bw, bh = local.bbox

        for x, y, w, h in (
            (hx, hy, hw, hh),
            (nx, ny, nw, nh),
            (bx, by, bw, bh),
        ):
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + w, region.width)
            self.assertLessEqual(y + h, region.height)

    def test_localize_preserves_board_geometry(self):
        candidate = self._candidate()
        region = tracking_region_for_candidate(candidate, 1920, 1080)
        local = localize_candidate(candidate, region)
        self.assertEqual(local.w, candidate.w)
        self.assertEqual(local.h, candidate.h)
        self.assertEqual(local.cell_size, candidate.cell_size)
        self.assertEqual(local.x + region.x, candidate.x)
        self.assertEqual(local.y + region.y, candidate.y)


if __name__ == "__main__":
    unittest.main()
