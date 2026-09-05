from __future__ import annotations

from collections.abc import Iterable

import numpy as np

Cell = tuple[int, int]


def normalize_binary_board(board, *, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Return a uint8 0/1 board view/copy without changing board semantics."""
    arr = np.asarray(board, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"Board must be 2-dimensional; got shape {arr.shape}")
    if expected_shape is not None and arr.shape != tuple(expected_shape):
        raise ValueError(f"Expected board shape {tuple(expected_shape)}, got {arr.shape}")
    return (arr != 0).astype(np.uint8, copy=False)


def can_place_cells(
    board,
    cells: Iterable[Cell],
    *,
    allow_above: bool = True,
) -> bool:
    """Collision primitive shared by final-placement and path-aware engines."""
    arr = np.asarray(board)
    if arr.ndim != 2:
        raise ValueError("Board must be 2-dimensional")
    height, width = arr.shape
    for x, y in cells:
        x = int(x); y = int(y)
        if x < 0 or x >= width or y >= height:
            return False
        if y < 0:
            if allow_above:
                continue
            return False
        if arr[y, x] != 0:
            return False
    return True


def lock_cells(
    board,
    cells: Iterable[Cell],
    *,
    ignore_above: bool = True,
) -> tuple[np.ndarray, bool]:
    """Lock occupied cells and report whether any cell was above the field."""
    result = normalize_binary_board(board).copy()
    height, width = result.shape
    top_out = False
    for x, y in cells:
        x = int(x); y = int(y)
        if x < 0 or x >= width or y >= height:
            raise ValueError(f"Cell outside board: {(x, y)}")
        if y < 0:
            top_out = True
            if ignore_above:
                continue
            raise ValueError(f"Cell above board: {(x, y)}")
        result[y, x] = 1
    return result, top_out


def clear_full_rows(board) -> tuple[np.ndarray, int]:
    """Clear full rows while preserving row order and board shape."""
    arr = normalize_binary_board(board)
    full = np.all(arr != 0, axis=1)
    count = int(np.count_nonzero(full))
    if count == 0:
        return arr.copy(), 0
    kept = arr[~full]
    zeros = np.zeros((count, arr.shape[1]), dtype=np.uint8)
    return np.vstack((zeros, kept)), count
