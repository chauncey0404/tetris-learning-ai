from __future__ import annotations

import unittest
import numpy as np

from tetrio.fast_reachability import enumerate_tetrio_reachable_geometries_fast
from tetrio.future.lookahead import FutureCandidateInput, FutureFeatureConfig, build_row_future_features
from tetrio.ruleset import TETRIO_MOVEMENT
from tetris_ai.core.movement import clear_lines, lock_piece


class FutureFeatureCacheParityTests(unittest.TestCase):
    def test_cached_and_uncached_features_match_on_empty_board(self):
        board = TETRIO_MOVEMENT.empty_board()
        landings = enumerate_tetrio_reachable_geometries_fast(board, "T", max_states=10_000)[:8]
        candidates = []
        for landing in landings:
            locked = lock_piece(board, landing, TETRIO_MOVEMENT)
            after, lines = clear_lines(locked, TETRIO_MOVEMENT)
            candidates.append(FutureCandidateInput(
                board_after=np.asarray(after, dtype=np.uint8),
                piece="T",
                rotation=int(landing.rotation) % 4,
                x=int(landing.x),
                y=int(landing.y),
                use_hold=False,
                lines=int(lines),
            ))
        kwargs = dict(
            board_before=board,
            active="T",
            hold="I",
            preview=("J", "L", "S", "Z", "O"),
            candidates=tuple(candidates),
        )
        uncached = build_row_future_features(**kwargs, config=FutureFeatureConfig(use_search_cache=False))
        cached = build_row_future_features(**kwargs, config=FutureFeatureConfig(use_search_cache=True))
        np.testing.assert_array_equal(cached, uncached)


if __name__ == "__main__":
    unittest.main()
