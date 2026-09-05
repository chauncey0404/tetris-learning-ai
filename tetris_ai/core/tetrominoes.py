from __future__ import annotations

import numpy as np

from tetris_ai.core.types import PieceState

PIECE_NAMES = ("I", "O", "T", "S", "Z", "J", "L")
PIECE_TO_ID = {piece: index for index, piece in enumerate(PIECE_NAMES)}

# Single authoritative native rotation boxes. These match the already-validated
# V8.8 JAX geometry and are also the boxes required by SRS/SRS+ kick semantics.
_NATIVE_BASE: dict[str, np.ndarray] = {
    "I": np.asarray([[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]], dtype=np.uint8),
    "O": np.asarray([[1,1],[1,1]], dtype=np.uint8),
    "T": np.asarray([[0,1,0],[1,1,1],[0,0,0]], dtype=np.uint8),
    "S": np.asarray([[0,1,1],[1,1,0],[0,0,0]], dtype=np.uint8),
    "Z": np.asarray([[1,1,0],[0,1,1],[0,0,0]], dtype=np.uint8),
    "J": np.asarray([[1,0,0],[1,1,1],[0,0,0]], dtype=np.uint8),
    "L": np.asarray([[0,0,1],[1,1,1],[0,0,0]], dtype=np.uint8),
}


def validate_piece(piece: str) -> None:
    if piece not in _NATIVE_BASE:
        raise ValueError(f"Unknown piece: {piece!r}; expected one of {PIECE_NAMES}")


def native_matrix(piece: str, rotation: int = 0) -> np.ndarray:
    """Native SRS-box matrix for canonical geometric-CW rotation 0..3."""
    validate_piece(piece)
    return np.rot90(_NATIVE_BASE[piece], k=-(int(rotation) % 4)).copy()

# Backward-friendly name for the V9 reference engine.
matrix = native_matrix


def box_size(piece: str) -> int:
    validate_piece(piece)
    return int(_NATIVE_BASE[piece].shape[0])


def trim_matrix(shape) -> np.ndarray:
    """Remove empty outer rows/columns without changing occupied geometry."""
    arr = np.asarray(shape, dtype=np.uint8)
    rows = np.any(arr != 0, axis=1)
    cols = np.any(arr != 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        raise ValueError("Tetromino shape cannot be empty")
    return arr[np.ix_(rows, cols)].copy()


def trimmed_matrix(piece: str, rotation: int = 0) -> np.ndarray:
    """Canonical occupied bounding box used by the legacy final-placement AI."""
    return trim_matrix(native_matrix(piece, rotation))


def unique_trimmed_rotations(piece: str) -> list[tuple[int, np.ndarray]]:
    validate_piece(piece)
    result: list[tuple[int, np.ndarray]] = []
    seen: set[tuple[tuple[int, ...], bytes]] = set()
    for rotation in range(4):
        shape = trimmed_matrix(piece, rotation)
        key = (shape.shape, shape.tobytes())
        if key in seen:
            continue
        seen.add(key)
        result.append((rotation, shape))
    return result


def occupied_cells(state: PieceState) -> tuple[tuple[int, int], ...]:
    shape = native_matrix(state.piece, state.rotation)
    ys, xs = np.nonzero(shape)
    return tuple((state.x + int(x), state.y + int(y)) for y, x in zip(ys, xs))


def padded_rotation_tensor(dtype=np.int8) -> np.ndarray:
    """Return [7,4,4,4] canonical rotation tensor for JAX/fast backends."""
    all_rots = np.zeros((len(PIECE_NAMES), 4, 4, 4), dtype=dtype)
    for p, piece in enumerate(PIECE_NAMES):
        for r in range(4):
            rot = native_matrix(piece, r).astype(dtype, copy=False)
            all_rots[p, r, :rot.shape[0], :rot.shape[1]] = rot
    return all_rots
