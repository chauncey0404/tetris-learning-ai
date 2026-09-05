from __future__ import annotations

import numpy as np

from tetris_ai.battle.types import PieceState


PIECE_NAMES = ("I", "O", "T", "S", "Z", "J", "L")

# Preserve the exact V8.8/JAX native-matrix geometry supplied in v9_core_context.zip.
# This is intentionally separate from V8's trimmed placement shapes: V9 needs the
# native SRS rotation box so that wall-kick offsets have a stable rotation origin.
_BASE_MATRICES: dict[str, np.ndarray] = {
    "I": np.asarray(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    ),
    "O": np.asarray(
        [
            [1, 1],
            [1, 1],
        ],
        dtype=np.uint8,
    ),
    "T": np.asarray(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    ),
    "S": np.asarray(
        [
            [0, 1, 1],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    ),
    "Z": np.asarray(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    ),
    "J": np.asarray(
        [
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    ),
    "L": np.asarray(
        [
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    ),
}


def matrix(piece: str, rotation: int = 0) -> np.ndarray:
    """Return the native-box matrix for canonical geometric-CW rotation 0..3."""

    if piece not in _BASE_MATRICES:
        raise ValueError(f"Unknown piece: {piece!r}")
    r = int(rotation) % 4
    return np.rot90(_BASE_MATRICES[piece], k=-r).copy()


def box_size(piece: str) -> int:
    return int(_BASE_MATRICES[piece].shape[0])


def occupied_cells(state: PieceState) -> tuple[tuple[int, int], ...]:
    shape = matrix(state.piece, state.rotation)
    ys, xs = np.nonzero(shape)
    return tuple((state.x + int(x), state.y + int(y)) for y, x in zip(ys, xs))


def validate_piece(piece: str) -> None:
    if piece not in _BASE_MATRICES:
        raise ValueError(f"Unknown piece: {piece!r}; expected one of {PIECE_NAMES}")
