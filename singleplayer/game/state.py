from dataclasses import dataclass
from typing import Optional

import numpy as np



# ============================================================
# AI 永遠只認這套格式
# ============================================================

@dataclass(frozen=True)
class CanonicalState:
    # 固定堆積，不包含正在掉落的方塊
    # shape = (20, 10)
    # 0 = empty
    # 1 = occupied
    board: np.ndarray

    # "I", "O", "T", "S", "Z", "J", "L"
    current_piece: str

    # 例如 ("T", "L", "Z", "O")
    next_pieces: tuple[str, ...]

    # None 代表 Hold 為空
    hold_piece: Optional[str]

    # 這一顆是否還可以使用 Hold
    can_hold: bool


@dataclass(frozen=True)
class PlacementAction:
    """
    AI 未來真正輸出的動作。

    rotation:
        0 = spawn orientation
        1 = 90 degrees
        2 = 180 degrees
        3 = 270 degrees

    x:
        Canonical 10-column board 上的位置。

    use_hold:
        是否先 Hold 再決定 placement。
    """
    rotation: int
    x: int
    use_hold: bool = False


# ============================================================
# Debug display
# ============================================================

def print_board(board):

    print("+" + "-" * board.shape[1] + "+")

    for row in board:

        print(
            "|"
            + "".join(
                "#" if cell else "."
                for cell in row
            )
            + "|"
        )

    print("+" + "-" * board.shape[1] + "+")


def print_state(state):

    print()
    print("=" * 50)
    print("CANONICAL STATE")
    print("=" * 50)

    print("Current :", state.current_piece)
    print("Next    :", state.next_pieces)
    print("Hold    :", state.hold_piece)
    print("Can hold:", state.can_hold)

    print()
    print("Board:", state.board.shape)

    print_board(state.board)