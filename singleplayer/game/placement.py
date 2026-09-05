from dataclasses import dataclass

import numpy as np

from singleplayer.game.state import PlacementAction


# ============================================================
# 基礎 Tetromino
#
# 这里只定義 spawn orientation。
# 其他 rotation 由程式自動產生。
#
# Canonical x 的定義：
#     方塊實際佔用格子的「最左欄」
#
# 這樣不綁定任何特定遊戲的 rotation origin。
# ============================================================

from tetris_ai.core.board import can_place_cells, clear_full_rows, lock_cells
from tetris_ai.core.tetrominoes import PIECE_NAMES, trimmed_matrix, unique_trimmed_rotations

BASE_SHAPES = {piece: trimmed_matrix(piece, 0) for piece in PIECE_NAMES}

# ============================================================
# Placement 結果
# ============================================================

@dataclass(frozen=True)
class PlacementResult:

    action: PlacementAction

    # 方塊最後的左上角 Y。
    # 可能小於 0，代表部分方塊超過棋盤上方。
    landing_y: int

    # 放置並清行後的棋盤
    after_board: np.ndarray

    lines_cleared: int

    # 方塊鎖定時是否有格子仍在棋盤上方
    top_out: bool


# ============================================================
# Shape utility
# ============================================================

def trim_shape(shape):
    from tetris_ai.core.tetrominoes import trim_matrix
    return trim_matrix(shape)


def get_rotations(piece):
    return [(rotation, shape.copy()) for rotation, shape in unique_trimmed_rotations(piece)]

# ============================================================
# Collision
# ============================================================

def can_place(board, shape, x, y):
    """Legacy final-placement collision semantics backed by shared primitives."""
    arr = np.asarray(board)
    ys, xs = np.nonzero(np.asarray(shape, dtype=np.uint8))
    cells = tuple((int(x) + int(sx), int(y) + int(sy)) for sy, sx in zip(ys, xs))
    return can_place_cells(arr, cells, allow_above=True)


# ============================================================
# Hard drop
# ============================================================

def find_landing_y(
    board,
    shape,
    x,
):
    """
    從棋盤上方垂直下降，
    找到最終 landing y。
    """

    shape_height = shape.shape[0]

    # 整顆方塊先放在棋盤上方
    y = -shape_height

    if not can_place(
        board,
        shape,
        x,
        y,
    ):
        return None

    while can_place(
        board,
        shape,
        x,
        y + 1,
    ):
        y += 1

    return y


# ============================================================
# Lock piece
# ============================================================

def lock_piece(board, shape, x, y):
    ys, xs = np.nonzero(np.asarray(shape, dtype=np.uint8))
    cells = tuple((int(x) + int(sx), int(y) + int(sy)) for sy, sx in zip(ys, xs))
    return lock_cells(board, cells, ignore_above=True)


# ============================================================
# Line clear
# ============================================================

def clear_lines(board):
    return clear_full_rows(board)


# ============================================================
# 單一 Placement
# ============================================================

def simulate_placement(
    board,
    piece,
    rotation,
    x,
    use_hold=False,
):
    rotations = dict(
        get_rotations(piece)
    )

    if rotation not in rotations:
        return None

    shape = rotations[rotation]

    board_width = board.shape[1]

    shape_width = shape.shape[1]

    if x < 0:
        return None

    if x + shape_width > board_width:
        return None

    landing_y = find_landing_y(
        board,
        shape,
        x,
    )

    if landing_y is None:
        return None

    locked_board, top_out = lock_piece(
        board,
        shape,
        x,
        landing_y,
    )

    after_board, lines_cleared = (
        clear_lines(locked_board)
    )

    action = PlacementAction(
        rotation=rotation,
        x=x,
        use_hold=use_hold,
    )

    return PlacementResult(
        action=action,
        landing_y=landing_y,
        after_board=after_board,
        lines_cleared=lines_cleared,
        top_out=top_out,
    )


# ============================================================
# 列舉所有 Placement
# ============================================================

def enumerate_placements(
    board,
    piece,
    use_hold=False,
):
    """
    列出目前 piece 所有：

        unique rotation
        ×
        valid x

    的 Hard Drop Afterstate。

    注意：

    V2A 現在只處理：
        最終幾何落點

    還沒有處理：
        SRS wall kick
        移動路徑 reachability
        spin
        hold branch

    這些會在下一階段加入。
    """

    board = np.asarray(
        board,
        dtype=np.uint8,
    )

    if board.ndim != 2:
        raise ValueError(
            "Board must be 2-dimensional."
        )

    board_width = board.shape[1]

    results = []

    for rotation, shape in get_rotations(piece):

        shape_width = shape.shape[1]

        max_x = (
            board_width
            - shape_width
        )

        for x in range(
            0,
            max_x + 1,
        ):

            result = simulate_placement(
                board=board,
                piece=piece,
                rotation=rotation,
                x=x,
                use_hold=use_hold,
            )

            if result is not None:
                results.append(result)

    return results

# ============================================================
# State -> 實際要放置的 piece
# ============================================================

def piece_for_placement(
    state,
    action,
):
    """
    根據 CanonicalState + PlacementAction
    判斷真正被放下去的是哪一顆。

    不 Hold：
        current_piece

    Hold 已有方塊：
        hold_piece

    Hold 為空：
        next_pieces[0]
    """

    if not action.use_hold:
        return state.current_piece

    if not state.can_hold:
        raise ValueError(
            "Hold requested but can_hold is False."
        )

    if state.hold_piece is not None:
        return state.hold_piece

    if not state.next_pieces:
        raise ValueError(
            "Hold is empty but next queue is empty."
        )

    return state.next_pieces[0]


# ============================================================
# CanonicalState 的完整 Placement candidates
# ============================================================

def enumerate_state_placements(state):
    """
    列出目前狀態所有基本 placement。

    Branch A:
        不使用 Hold
        -> current_piece

    Branch B:
        使用 Hold

        holder 非空：
            -> hold_piece

        holder 為空：
            -> next_pieces[0]

    注意：
    現階段仍是：
        rotate at top
        horizontal move
        hard drop

    尚未加入 tuck / SRS path search。
    """

    results = []

    # --------------------------------------------------------
    # Branch A：不用 Hold
    # --------------------------------------------------------

    results.extend(
        enumerate_placements(
            board=state.board,
            piece=state.current_piece,
            use_hold=False,
        )
    )

    # --------------------------------------------------------
    # Branch B：使用 Hold
    # --------------------------------------------------------

    if state.can_hold:

        if state.hold_piece is not None:

            hold_branch_piece = (
                state.hold_piece
            )

        else:

            if not state.next_pieces:
                return results

            hold_branch_piece = (
                state.next_pieces[0]
            )

        results.extend(
            enumerate_placements(
                board=state.board,
                piece=hold_branch_piece,
                use_hold=True,
            )
        )

    return results