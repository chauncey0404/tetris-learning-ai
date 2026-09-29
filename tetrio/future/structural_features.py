from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoardStructure:
    holes: int
    max_height: int
    aggregate_height: int
    bumpiness: int
    max_well: int


@dataclass(frozen=True)
class BoardStructureBatch:
    holes: np.ndarray
    max_height: np.ndarray
    aggregate_height: np.ndarray
    bumpiness: np.ndarray
    max_well: np.ndarray


def column_heights(board: np.ndarray) -> np.ndarray:
    arr = np.asarray(board).reshape(40, 10) != 0
    heights = np.zeros(10, dtype=np.int16)
    for x in range(10):
        filled = np.flatnonzero(arr[:, x])
        if filled.size:
            heights[x] = 40 - int(filled[0])
    return heights


def hole_count(board: np.ndarray) -> int:
    arr = np.asarray(board).reshape(40, 10) != 0
    holes = 0
    for x in range(10):
        col = arr[:, x]
        filled = np.flatnonzero(col)
        if filled.size:
            holes += int(np.count_nonzero(~col[int(filled[0]):]))
    return holes


def max_well_depth(heights: np.ndarray) -> int:
    h = np.asarray(heights, dtype=np.int16)
    best = 0
    for x in range(10):
        left = int(h[x - 1]) if x > 0 else 40
        right = int(h[x + 1]) if x < 9 else 40
        depth = min(left, right) - int(h[x])
        best = max(best, depth)
    return max(0, int(best))


def board_structure(board: np.ndarray) -> BoardStructure:
    h = column_heights(board)
    return BoardStructure(
        holes=hole_count(board),
        max_height=int(h.max(initial=0)),
        aggregate_height=int(h.sum()),
        bumpiness=int(np.abs(np.diff(h)).sum()),
        max_well=max_well_depth(h),
    )


def board_structure_batch(boards: np.ndarray) -> BoardStructureBatch:
    """Exact vectorized structural metrics for [N,40,10] boards.

    The result is integer-equivalent to calling ``board_structure`` N times.
    This is used inside future lookahead, where dozens of legal next boards are
    evaluated for every shortlisted candidate.
    """
    arr = np.asarray(boards)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[1:] != (40, 10):
        raise ValueError(f"Expected [N,40,10], got {arr.shape}")

    n = int(arr.shape[0])
    if n == 0:
        empty = np.empty((0,), dtype=np.int16)
        return BoardStructureBatch(
            holes=empty.copy(),
            max_height=empty.copy(),
            aggregate_height=empty.copy(),
            bumpiness=empty.copy(),
            max_well=empty.copy(),
        )

    filled = arr != 0  # [N,40,10]

    any_filled = filled.any(axis=1)  # [N,10]
    first_filled = filled.argmax(axis=1).astype(np.int16, copy=False)
    heights = np.where(
        any_filled,
        40 - first_filled,
        0,
    ).astype(np.int16, copy=False)

    seen = np.maximum.accumulate(filled, axis=1)
    holes = ((~filled) & seen).sum(axis=(1, 2), dtype=np.int32)

    max_height = heights.max(axis=1)
    aggregate_height = heights.sum(axis=1, dtype=np.int32)
    bumpiness = np.abs(np.diff(heights, axis=1)).sum(axis=1, dtype=np.int32)

    left = np.empty_like(heights)
    right = np.empty_like(heights)
    left[:, 0] = 40
    left[:, 1:] = heights[:, :-1]
    right[:, -1] = 40
    right[:, :-1] = heights[:, 1:]
    well_depth = np.minimum(left, right) - heights
    max_well = np.maximum(well_depth, 0).max(axis=1)

    return BoardStructureBatch(
        holes=holes.astype(np.int16, copy=False),
        max_height=max_height.astype(np.int16, copy=False),
        aggregate_height=aggregate_height.astype(np.int32, copy=False),
        bumpiness=bumpiness.astype(np.int32, copy=False),
        max_well=max_well.astype(np.int16, copy=False),
    )
