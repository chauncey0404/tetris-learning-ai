from __future__ import annotations

import unittest
import numpy as np

from tetrio.future.search_cache import (
    board_from_occupancy_key,
    clear_fast_search_cache,
    fast_landings,
    fast_search_cache_info,
    occupancy_key,
)


class FutureSearchCacheTests(unittest.TestCase):
    def test_board_key_round_trip_preserves_occupancy(self):
        rng = np.random.default_rng(20260919)
        board = (rng.random((40, 10)) < 0.25).astype(np.uint8)
        restored = board_from_occupancy_key(occupancy_key(board))
        np.testing.assert_array_equal(restored, board != 0)

    def test_cached_and_uncached_landings_match_empty_board(self):
        board = np.zeros((40, 10), dtype=np.uint8)
        clear_fast_search_cache()
        uncached = fast_landings(board, "T", max_states=10_000, use_cache=False)
        cached1 = fast_landings(board, "T", max_states=10_000, use_cache=True)
        cached2 = fast_landings(board, "T", max_states=10_000, use_cache=True)
        self.assertEqual([x.geometry_key() for x in uncached], [x.geometry_key() for x in cached1])
        self.assertEqual([x.geometry_key() for x in cached1], [x.geometry_key() for x in cached2])
        self.assertGreaterEqual(fast_search_cache_info().hits, 1)


if __name__ == "__main__":
    unittest.main()
