from __future__ import annotations

import unittest

import cv2
import numpy as np

from tetrio.vision.layout import (
    PlayfieldCandidate,
    _phase_lock_candidate,
    _vertical_line_clusters,
)


def synthetic_board(
    width=1400,
    height=900,
    x0=500,
    y0=100,
    cell=30,
):
    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    # Thin internal grid.
    for col in range(11):
        x = x0 + col * cell
        cv2.line(
            image,
            (x, y0),
            (x, y0 + 20 * cell),
            (90, 90, 90),
            1,
        )

    for row in range(21):
        y = y0 + row * cell
        cv2.line(
            image,
            (x0, y),
            (x0 + 10 * cell, y),
            (90, 90, 90),
            1,
        )

    # Strong continuous outer vertical borders.
    cv2.line(
        image,
        (x0, y0),
        (x0, y0 + 20 * cell),
        (255, 255, 255),
        3,
    )
    cv2.line(
        image,
        (x0 + 10 * cell, y0),
        (x0 + 10 * cell, y0 + 20 * cell),
        (255, 255, 255),
        3,
    )
    return image


class PhaseLockTests(unittest.TestCase):
    def test_two_cell_shift_is_reanchored(self):
        image = synthetic_board()
        edges, clusters = _vertical_line_clusters(
            image
        )

        # Same 10x20 size, but exactly two cells left/down.
        wrong = PlayfieldCandidate(
            x=440,
            y=160,
            w=300,
            h=600,
            cell_size=30,
            grid_score=0.2,
            matched_vertical_lines=10,
            confidence=0.5,
        )

        fixed = _phase_lock_candidate(
            image,
            edges,
            clusters,
            wrong,
        )

        self.assertLess(abs(fixed.x - 500), 4)
        self.assertLess(abs(fixed.y - 100), 4)

    def test_good_candidate_is_not_shifted(self):
        image = synthetic_board()
        edges, clusters = _vertical_line_clusters(
            image
        )

        good = PlayfieldCandidate(
            x=500,
            y=100,
            w=300,
            h=600,
            cell_size=30,
            grid_score=0.4,
            matched_vertical_lines=11,
            confidence=0.9,
        )

        fixed = _phase_lock_candidate(
            image,
            edges,
            clusters,
            good,
        )

        self.assertLess(abs(fixed.x - 500), 1)
        self.assertLess(abs(fixed.y - 100), 1)


if __name__ == "__main__":
    unittest.main()
