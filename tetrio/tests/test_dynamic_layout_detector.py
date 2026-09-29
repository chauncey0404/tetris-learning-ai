from __future__ import annotations

import unittest

import cv2
import numpy as np

from tetrio.vision.layout import (
    OcrObservation,
    PlayfieldCandidate,
    detect_playfields,
    normalize_username,
    resolve_playfield_roles,
    username_similarity,
)


def synthetic_grid(
    width: int,
    height: int,
    boards: list[tuple[int, int, int]],
) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for x0, y0, cell in boards:
        for c in range(11):
            x = x0 + c * cell
            cv2.line(
                image,
                (x, y0),
                (x, y0 + 20 * cell),
                (80, 80, 80),
                1,
            )
        for r in range(21):
            y = y0 + r * cell
            cv2.line(
                image,
                (x0, y),
                (x0 + 10 * cell, y),
                (80, 80, 80),
                1,
            )
        cv2.rectangle(
            image,
            (x0, y0),
            (x0 + 10 * cell, y0 + 20 * cell),
            (255, 255, 255),
            2,
        )
    return image


class DynamicLayoutTests(unittest.TestCase):
    def test_normalize_username(self):
        self.assertEqual(
            normalize_username(" MayShowGunMore77 "),
            "MAYSHOWGUNMORE77",
        )

    def test_username_similarity_tolerates_one_ocr_error(self):
        score = username_similarity(
            "MAYSHOWGUNM0RE77",
            "MAYSHOWGUNMORE77",
        )
        self.assertGreater(score, 0.90)

    def test_detect_single_grid(self):
        image = synthetic_grid(
            1280,
            720,
            [(520, 120, 24)],
        )
        boards = detect_playfields(image)
        self.assertEqual(len(boards), 1)
        x, y, w, h = boards[0].bbox
        self.assertLess(abs(x - 520), 8)
        self.assertLess(abs(w - 240), 8)
        self.assertLess(abs(h - 480), 10)

    def test_detect_two_grids(self):
        image = synthetic_grid(
            1600,
            900,
            [(260, 150, 26), (1010, 150, 26)],
        )
        boards = detect_playfields(image)
        self.assertEqual(len(boards), 2)

    def test_single_board_is_self_without_ocr(self):
        candidate = PlayfieldCandidate(
            x=100,
            y=100,
            w=300,
            h=600,
            cell_size=30,
            grid_score=0.4,
            matched_vertical_lines=11,
            confidence=1.0,
        )
        image = np.zeros((800, 800, 3), dtype=np.uint8)
        resolved = resolve_playfield_roles(
            image,
            [candidate],
            self_username="MAYSHOWGUNMORE77",
            ocr_reader=lambda *_: OcrObservation(None, 0.0),
        )
        self.assertEqual(resolved[0].role, "SELF")

    def test_two_board_username_resolver(self):
        left = PlayfieldCandidate(
            100, 100, 300, 600, 30, 0.4, 11, 1.0
        )
        right = PlayfieldCandidate(
            800, 100, 300, 600, 30, 0.4, 11, 1.0
        )
        image = np.zeros((800, 1300, 3), dtype=np.uint8)

        observations = {
            100: OcrObservation("MAYSHOWGUNM0RE77", 0.95),
            800: OcrObservation("LESYA", 0.99),
        }

        def reader(_image, candidate):
            return observations[int(candidate.x)]

        resolved = resolve_playfield_roles(
            image,
            [left, right],
            self_username="MAYSHOWGUNMORE77",
            ocr_reader=reader,
        )
        self.assertEqual(
            [x.role for x in resolved],
            ["SELF", "OPPONENT"],
        )


if __name__ == "__main__":
    unittest.main()
