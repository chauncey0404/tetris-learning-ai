from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from tetris_ai.core.board import normalize_binary_board


@dataclass(frozen=True, slots=True)
class GarbageInsertResult:
    board: np.ndarray
    lines_inserted: int
    top_out: bool


def insert_garbage_rows(board, holes: Iterable[int]) -> GarbageInsertResult:
    """Insert garbage rows at the bottom of a 10-column board.

    Each inserted row is full except for one explicit hole.  Existing rows are
    pushed upward.  ``top_out`` reports whether occupied cells were pushed off
    the top.  Hole generation is intentionally external so TETR.IO messiness
    can be replay-validated independently from board mechanics.
    """

    arr = normalize_binary_board(board)
    if arr.shape[1] != 10:
        raise ValueError(f"Expected width 10, got {arr.shape[1]}")

    hole_values = tuple(int(h) for h in holes)
    for h in hole_values:
        if not 0 <= h < 10:
            raise ValueError("hole must be in [0, 9]")

    count = len(hole_values)
    if count == 0:
        return GarbageInsertResult(arr.copy(), 0, False)

    height = arr.shape[0]
    if count >= height:
        top_out = bool(np.any(arr))
        kept = np.zeros((0, 10), dtype=np.uint8)
        hole_values = hole_values[-height:]
        count = height
    else:
        top_out = bool(np.any(arr[:count]))
        kept = arr[count:].copy()

    garbage = np.ones((count, 10), dtype=np.uint8)
    for row, hole in enumerate(hole_values):
        garbage[row, hole] = 0

    result = np.vstack((kept, garbage))
    if result.shape != arr.shape:
        raise AssertionError("garbage insertion changed board shape")
    return GarbageInsertResult(result, count, top_out)
