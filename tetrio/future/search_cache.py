from __future__ import annotations

from functools import lru_cache

import numpy as np

from tetrio.fast_reachability import enumerate_tetrio_reachable_geometries_fast
from tetris_ai.core.types import PieceState


_BOARD_BITS = 40 * 10
_BOARD_BYTES = (_BOARD_BITS + 7) // 8
FAST_LANDING_CACHE_SIZE = 1024


def occupancy_key(board: np.ndarray) -> bytes:
    arr = np.asarray(board).reshape(40, 10) != 0
    return np.packbits(arr.reshape(-1), bitorder="little").tobytes()


def board_from_occupancy_key(key: bytes) -> np.ndarray:
    if len(key) != _BOARD_BYTES:
        raise ValueError(
            f"Expected {_BOARD_BYTES} bytes for 40x10 board, got {len(key)}"
        )
    bits = np.unpackbits(
        np.frombuffer(key, dtype=np.uint8),
        bitorder="little",
        count=_BOARD_BITS,
    )
    return bits.reshape(40, 10).astype(np.uint8, copy=False)


@lru_cache(maxsize=FAST_LANDING_CACHE_SIZE)
def _fast_landings_from_key(
    board_key: bytes,
    piece: str,
    max_states: int,
) -> tuple[PieceState, ...]:
    board = board_from_occupancy_key(board_key)
    return tuple(
        enumerate_tetrio_reachable_geometries_fast(
            board,
            str(piece),
            max_states=int(max_states),
        )
    )


def fast_landings(
    board: np.ndarray,
    piece: str,
    *,
    max_states: int,
    use_cache: bool = True,
) -> tuple[PieceState, ...]:
    if not use_cache:
        return tuple(
            enumerate_tetrio_reachable_geometries_fast(
                board,
                str(piece),
                max_states=int(max_states),
            )
        )
    return _fast_landings_from_key(
        occupancy_key(board),
        str(piece),
        int(max_states),
    )


def clear_fast_search_cache() -> None:
    _fast_landings_from_key.cache_clear()


def fast_search_cache_info():
    return _fast_landings_from_key.cache_info()
