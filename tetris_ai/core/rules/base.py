from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tetris_ai.core.tetrominoes import box_size, validate_piece
from tetris_ai.core.rotation.base import RotationSystem
from tetris_ai.core.types import MoveAction, PieceState


@dataclass(frozen=True)
class MovementRuleset:
    """Game-specific movement configuration for the common V9 engine."""

    name: str
    rotation_system: RotationSystem
    width: int = 10
    height: int = 40
    visible_height: int = 20
    allow_180: bool = False

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Board dimensions must be positive")
        if not (0 < self.visible_height <= self.height):
            raise ValueError("visible_height must be in 1..height")
        if self.allow_180 and not self.rotation_system.supports_180:
            raise ValueError(
                f"{self.name} enables 180 but {self.rotation_system.name} does not support it"
            )

    @property
    def hidden_rows(self) -> int:
        return self.height - self.visible_height

    @property
    def movement_actions(self) -> tuple[MoveAction, ...]:
        actions = [
            MoveAction.LEFT,
            MoveAction.RIGHT,
            MoveAction.DOWN,
            MoveAction.CW,
            MoveAction.CCW,
        ]
        if self.allow_180:
            actions.append(MoveAction.ROTATE_180)
        return tuple(actions)

    def empty_board(self) -> np.ndarray:
        return np.zeros((self.height, self.width), dtype=np.uint8)

    def spawn_state(self, piece: str) -> PieceState:
        """Guideline-style modern spawn location inside a 40-row field.

        SRS documentation states that later games spawn one row lower than
        Tetris Worlds: JLSTZO occupy rows 21-22 and horizontal I occupies row
        21 in a 40-row field. With top-origin row-major coordinates and the
        supplied native matrices, a common matrix-anchor y of hidden_rows-2
        reproduces that geometry.
        """

        validate_piece(piece)
        size = box_size(piece)
        x = (self.width - size) // 2
        y = self.hidden_rows - 2
        return PieceState(piece=piece, x=x, y=y, rotation=0)

    def lift_visible_board(self, visible_board: np.ndarray) -> np.ndarray:
        """Embed a legacy 20x10 V8 board into the bottom of a V9 board."""

        arr = np.asarray(visible_board, dtype=np.uint8)
        expected = (self.visible_height, self.width)
        if arr.shape != expected:
            raise ValueError(f"Expected visible board shape {expected}, got {arr.shape}")
        result = self.empty_board()
        result[-self.visible_height :, :] = (arr != 0).astype(np.uint8)
        return result
