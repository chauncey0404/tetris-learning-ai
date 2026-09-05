from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from tetris_ai.core.board import can_place_cells, clear_full_rows, lock_cells
from tetris_ai.core.tetrominoes import occupied_cells
from tetris_ai.core.types import PieceState

BOARD_WIDTH = 10
BOARD_HEIGHT = 40
EMPTY_CODE = "N"

ROTATION_CW: Mapping[str, int] = {"N": 0, "E": 1, "S": 2, "W": 3}
ROTATION_CCW: Mapping[str, int] = {"N": 0, "E": 3, "S": 2, "W": 1}


@dataclass(frozen=True)
class CoordinateConvention:
    x_offset: int = -1
    y_base: int = 38
    rotation_map_name: str = "cw"

    @property
    def rotation_map(self) -> Mapping[str, int]:
        if self.rotation_map_name == "cw":
            return ROTATION_CW
        if self.rotation_map_name == "ccw":
            return ROTATION_CCW
        raise ValueError(f"Unknown rotation map: {self.rotation_map_name!r}")


DEFAULT_COORDINATE_CONVENTION = CoordinateConvention()


@dataclass(frozen=True)
class DatasetPieceMapping:
    canonical_piece: str
    x_offset: int
    y_base: int
    rotation_map_name: str = "cw"

    @property
    def rotation_map(self) -> Mapping[str, int]:
        if self.rotation_map_name == "cw":
            return ROTATION_CW
        if self.rotation_map_name == "ccw":
            return ROTATION_CCW
        raise ValueError(f"Unknown rotation map: {self.rotation_map_name!r}")


DATASET_PIECE_ORIENTATION_MAP: dict[tuple[str, str], DatasetPieceMapping] = {
    ("I", "N"): DatasetPieceMapping("I", -1, 38),
    ("I", "E"): DatasetPieceMapping("I", -2, 38),
    ("I", "S"): DatasetPieceMapping("I", -2, 37),
    ("I", "W"): DatasetPieceMapping("I", -1, 37),

    ("O", "N"): DatasetPieceMapping("O", 0, 38),

    ("T", "N"): DatasetPieceMapping("T", -1, 38),
    ("T", "E"): DatasetPieceMapping("T", -1, 38),
    ("T", "S"): DatasetPieceMapping("T", -1, 38),
    ("T", "W"): DatasetPieceMapping("T", -1, 38),

    ("S", "N"): DatasetPieceMapping("S", -1, 38),
    ("S", "E"): DatasetPieceMapping("S", -1, 38),
    ("S", "S"): DatasetPieceMapping("S", -1, 38),
    ("S", "W"): DatasetPieceMapping("S", -1, 38),

    ("Z", "N"): DatasetPieceMapping("Z", -1, 38),
    ("Z", "E"): DatasetPieceMapping("Z", -1, 38),
    ("Z", "S"): DatasetPieceMapping("Z", -1, 38),
    ("Z", "W"): DatasetPieceMapping("Z", -1, 38),

    ("J", "N"): DatasetPieceMapping("L", -1, 38),
    ("J", "E"): DatasetPieceMapping("L", -1, 38),
    ("J", "S"): DatasetPieceMapping("L", -1, 38),
    ("J", "W"): DatasetPieceMapping("L", -1, 38),

    ("L", "N"): DatasetPieceMapping("J", -1, 38),
    ("L", "E"): DatasetPieceMapping("J", -1, 38),
    ("L", "S"): DatasetPieceMapping("J", -1, 38),
    ("L", "W"): DatasetPieceMapping("J", -1, 38),
}


def decode_playfield(encoded: str | None, *, height: int = BOARD_HEIGHT, width: int = BOARD_WIDTH,
                     empty_code: str = EMPTY_CODE) -> np.ndarray:
    text = "" if encoded is None else str(encoded)
    if len(text) > height * width:
        raise ValueError("Encoded playfield exceeds configured board size")
    board = np.zeros((height, width), dtype=np.uint8)
    for index, code in enumerate(text):
        row_from_bottom, x = divmod(index, width)
        y = height - 1 - row_from_bottom
        if code != empty_code:
            board[y, x] = 1
    return board


def encode_binary_playfield(board: np.ndarray, *, empty_code: str = "N",
                            filled_code: str = "X") -> str:
    arr = np.asarray(board)
    if arr.ndim != 2 or arr.shape[1] != BOARD_WIDTH:
        raise ValueError("Expected a 2D board of width 10")
    chars: list[str] = []
    for row_from_bottom in range(arr.shape[0]):
        y = arr.shape[0] - 1 - row_from_bottom
        for x in range(arr.shape[1]):
            chars.append(filled_code if arr[y, x] != 0 else empty_code)
    while chars and chars[-1] == empty_code:
        chars.pop()
    return "".join(chars)


def row_to_piece_state(*, piece: str, x: int, y: int, rotation_code: str,
                       convention: CoordinateConvention = DEFAULT_COORDINATE_CONVENTION) -> PieceState:
    rotation = convention.rotation_map[str(rotation_code)]
    return PieceState(
        piece=str(piece),
        x=int(x) + convention.x_offset,
        y=convention.y_base - int(y),
        rotation=int(rotation),
    )


def resolve_dataset_mapping(piece: str, rotation_code: str) -> DatasetPieceMapping:
    key = (str(piece), str(rotation_code))
    try:
        return DATASET_PIECE_ORIENTATION_MAP[key]
    except KeyError as exc:
        raise ValueError(
            f"Unvalidated corpus mapping for piece={piece!r}, rotation={rotation_code!r}"
        ) from exc


def dataset_row_to_piece_state(*, piece: str, x: int, y: int, rotation_code: str) -> PieceState:
    mapping = resolve_dataset_mapping(piece, rotation_code)
    return PieceState(
        piece=mapping.canonical_piece,
        x=int(x) + mapping.x_offset,
        y=mapping.y_base - int(y),
        rotation=int(mapping.rotation_map[str(rotation_code)]),
    )


@dataclass(frozen=True)
class PlacementSimulation:
    legal: bool
    board_after_clear: np.ndarray | None
    lines_cleared: int | None
    state: PieceState


def _simulate_from_state(board_before: np.ndarray, state: PieceState) -> PlacementSimulation:
    cells = occupied_cells(state)
    if not can_place_cells(board_before, cells, allow_above=True):
        return PlacementSimulation(False, None, None, state)
    locked, _ = lock_cells(board_before, cells, ignore_above=True)
    cleared, count = clear_full_rows(locked)
    return PlacementSimulation(True, cleared, int(count), state)


def simulate_dataset_placement(board_before: np.ndarray, *, piece: str, x: int, y: int,
                               rotation_code: str,
                               convention: CoordinateConvention = DEFAULT_COORDINATE_CONVENTION
                               ) -> PlacementSimulation:
    return _simulate_from_state(
        board_before,
        row_to_piece_state(piece=piece, x=x, y=y, rotation_code=rotation_code, convention=convention),
    )


def simulate_mapped_dataset_placement(board_before: np.ndarray, *, piece: str, x: int, y: int,
                                      rotation_code: str) -> PlacementSimulation:
    return _simulate_from_state(
        board_before,
        dataset_row_to_piece_state(piece=piece, x=x, y=y, rotation_code=rotation_code),
    )


def binary_board_diff(a: np.ndarray, b: np.ndarray) -> int:
    aa = np.asarray(a) != 0
    bb = np.asarray(b) != 0
    if aa.shape != bb.shape:
        raise ValueError("Board shape mismatch")
    return int(np.count_nonzero(aa != bb))
