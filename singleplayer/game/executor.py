import numpy as np


# ============================================================
# Tetris-Gymnasium low-level actions
# ============================================================

MOVE_LEFT = 0
MOVE_RIGHT = 1
MOVE_DOWN = 2

# 注意：
# Gym 0.3.1 名稱和實際 np.rot90 幾何方向相反。
ROTATE_NAMED_CW = 3
ROTATE_NAMED_CCW = 4

HARD_DROP = 5
HOLD = 6
NO_OP = 7


# ============================================================
# Canonical rotation -> Gym actions
# ============================================================

def apply_rotation(adapter, rotation):
    """
    Canonical:

        0 = spawn
        1 = CW 90
        2 = 180
        3 = CW 270 / CCW 90
    """

    rotation %= 4

    if rotation == 0:
        actions = []

    elif rotation == 1:
        # Gym action 4 實際幾何是 CW
        actions = [
            ROTATE_NAMED_CCW,
        ]

    elif rotation == 2:
        actions = [
            ROTATE_NAMED_CCW,
            ROTATE_NAMED_CCW,
        ]

    else:
        # Gym action 3 實際幾何是 CCW
        actions = [
            ROTATE_NAMED_CW,
        ]

    for action in actions:

        state, reward, terminated, truncated, info = (
            adapter.gym_step(action)
        )

        if terminated or truncated:
            return False

    return True


# ============================================================
# 取得目前活動方塊最左邊的 Canonical X
# ============================================================

def active_left_x(adapter):
    """
    raw.x 是 padded board 上 tetromino matrix 的 x。

    tetromino.matrix 本身有可能有外圍 0，
    所以必須找真正最左 occupied column。
    """

    tetromino = adapter.raw.active_tetromino

    matrix = np.asarray(
        tetromino.matrix
    )

    occupied = np.argwhere(
        matrix != 0
    )

    if occupied.size == 0:
        raise RuntimeError(
            "Active tetromino matrix is empty."
        )

    left_inside_matrix = int(
        occupied[:, 1].min()
    )

    canonical_x = (
        int(adapter.raw.x)
        - int(adapter.raw.padding)
        + left_inside_matrix
    )

    return canonical_x


# ============================================================
# Horizontal
# ============================================================

def apply_horizontal_moves(
    adapter,
    delta,
):

    if delta < 0:
        action = MOVE_LEFT

    elif delta > 0:
        action = MOVE_RIGHT

    else:
        return True

    for _ in range(abs(delta)):

        state, reward, terminated, truncated, info = (
            adapter.gym_step(action)
        )

        if terminated or truncated:
            return False

    return True


# ============================================================
# PlacementAction -> 真正 Gym low-level actions
# ============================================================

def execute_placement(
    adapter,
    placement_action,
):
    """
    現階段 Executor：

        rotate at top
             ↓
        horizontal move
             ↓
        hard drop

    尚未支援：
        Hold
        tuck
        spin
        full SRS path search
    """

    # --------------------------------------------------------
    # Hold
    # --------------------------------------------------------

    if placement_action.use_hold:

        before_hold = adapter.state()

        if not before_hold.can_hold:
            raise RuntimeError(
                "Placement requested Hold, "
                "but can_hold is False."
            )

        state, reward, terminated, truncated, info = (
            adapter.gym_step(HOLD)
        )

        if terminated or truncated:
            raise RuntimeError(
                "Game ended while executing Hold."
            )

    # --------------------------------------------------------
    # Rotate
    # --------------------------------------------------------

    if not apply_rotation(
        adapter,
        placement_action.rotation,
    ):
        raise RuntimeError(
            "Game ended while rotating."
        )

    # --------------------------------------------------------
    # 看旋轉後實際最左 X
    # --------------------------------------------------------

    current_x = active_left_x(
        adapter
    )

    delta = (
        placement_action.x
        - current_x
    )

    # --------------------------------------------------------
    # Horizontal
    # --------------------------------------------------------

    if not apply_horizontal_moves(
        adapter,
        delta,
    ):
        raise RuntimeError(
            "Game ended while moving horizontally."
        )

    actual_x = active_left_x(
        adapter
    )

    if actual_x != placement_action.x:

        raise RuntimeError(
            "Could not reach requested X: "
            f"target={placement_action.x}, "
            f"actual={actual_x}"
        )

    # --------------------------------------------------------
    # Hard Drop
    # --------------------------------------------------------

    state_after, reward, terminated, truncated, info = (
        adapter.gym_step(
            HARD_DROP
        )
    )

    return {
        "state": state_after,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
        "horizontal_delta": delta,
    }

# ============================================================
# Non-destructive placement preview
# ============================================================

def preview_placement(
    adapter,
    placement_action,
):
    """
    在 Gym 中真的執行一次 placement，
    但執行完立即把整個環境恢復。

    用來判斷：

        幾何上存在的 placement

    是否真的可以透過目前的：

        Hold
        Rotate
        Horizontal Move
        Hard Drop

    執行出來。

    不會永久改變真正遊戲狀態。
    """

    snapshot = (
        adapter.raw.get_state()
    )

    try:

        result = execute_placement(
            adapter,
            placement_action,
        )

        return result

    except Exception:

        return None

    finally:

        adapter.raw.set_state(
            snapshot
        )


# ============================================================
# Teacher + Gym reachability
# ============================================================

def choose_reachable_teacher_decision(
    adapter,
    teacher,
    state,
):
    """
    Teacher 負責：

        哪個 placement 戰略上最好？

    Gym Adapter 負責：

        這個 placement 目前真的做得到嗎？

    從 Teacher 排名最高的候選開始，
    找第一個實際執行結果和 simulator
    完全一致的 placement。
    """

    decisions = teacher.rank(
        state
    )

    for decision in decisions:

        candidate = (
            decision.candidate
        )

        result = preview_placement(
            adapter,
            candidate.action,
        )

        if result is None:
            continue

        actual = result["state"]

        # ----------------------------------------------------
        # Board 必須完全一致
        # ----------------------------------------------------

        if not np.array_equal(
            candidate.after_board,
            actual.board,
        ):
            continue

        # ----------------------------------------------------
        # Line clear 也必須一致
        # ----------------------------------------------------

        actual_lines = int(
            result["info"].get(
                "lines_cleared",
                0,
            )
        )

        if (
            actual_lines
            != candidate.lines_cleared
        ):
            continue

        # 第一個真正可執行的最高分 decision
        return decision

    return None