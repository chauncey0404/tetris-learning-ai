import cv2
import numpy as np

from tetris_ai.game.core import (
    GymTetrisAdapter,
)

from tetris_ai.heuristic.teacher import (
    HeuristicTeacherV2,
)

from tetris_ai.game.executor import (
    MOVE_LEFT,
    MOVE_RIGHT,
    ROTATE_NAMED_CW,
    ROTATE_NAMED_CCW,
    HARD_DROP,
    HOLD,
    active_left_x,
)


# ============================================================
# Settings
# ============================================================

# seed 1 是 teacher_v1_10k.npz 的第一局
SEED = 6

# 先看 200 顆即可。
# 想一直看可以改成 2000。
MAX_PIECES = 700

# 每一個旋轉 / 左右移動停多久
STEP_DELAY_MS = 180

# 方塊 Hard Drop 落地後多停一下
LOCK_DELAY_MS = 250


# ============================================================
# UI
# ============================================================

def render_frame(
    adapter,
    delay_ms,
):
    """
    顯示一個 frame。

    Controls:
        P       pause/resume
        Q/ESC   exit
    """

    adapter.env.render()

    key = (
        cv2.waitKey(delay_ms)
        & 0xFF
    )

    if key in (
        27,
        ord("q"),
    ):
        raise KeyboardInterrupt

    # ----------------------------------------
    # Pause
    # ----------------------------------------

    if key == ord("p"):

        print()
        print(
            "[PAUSED] "
            "Press P or SPACE to continue."
        )

        while True:

            key = (
                cv2.waitKey(50)
                & 0xFF
            )

            if key in (
                ord("p"),
                ord(" "),
            ):
                break

            if key in (
                27,
                ord("q"),
            ):
                raise KeyboardInterrupt


# ============================================================
# 單一步 Gym action + 顯示
# ============================================================

def visual_step(
    adapter,
    action,
    delay_ms=STEP_DELAY_MS,
):

    result = adapter.gym_step(
        action
    )

    render_frame(
        adapter,
        delay_ms,
    )

    return result


# ============================================================
# Canonical rotation
# ============================================================

def visual_rotation(
    adapter,
    rotation,
):
    """
    Canonical:

        0 = Spawn
        1 = CW 90
        2 = 180
        3 = CW 270

    注意：
    Gym 0.3.1 的 rotation action 名稱和
    實際幾何方向相反。

    所以這裡維持我們已驗證過的 Adapter mapping。
    """

    rotation %= 4

    if rotation == 0:

        actions = []

    elif rotation == 1:

        # Gym named CCW
        # 實際 geometry = CW
        actions = [
            ROTATE_NAMED_CCW,
        ]

    elif rotation == 2:

        actions = [
            ROTATE_NAMED_CCW,
            ROTATE_NAMED_CCW,
        ]

    else:

        # Gym named CW
        # 實際 geometry = CCW
        actions = [
            ROTATE_NAMED_CW,
        ]

    for action in actions:

        (
            state,
            reward,
            terminated,
            truncated,
            info,
        ) = visual_step(
            adapter,
            action,
        )

        if terminated or truncated:

            raise RuntimeError(
                "Game ended during rotation."
            )


# ============================================================
# Horizontal movement
# ============================================================

def visual_horizontal_move(
    adapter,
    delta,
):

    if delta < 0:

        action = MOVE_LEFT

    elif delta > 0:

        action = MOVE_RIGHT

    else:

        return

    for _ in range(
        abs(delta)
    ):

        (
            state,
            reward,
            terminated,
            truncated,
            info,
        ) = visual_step(
            adapter,
            action,
        )

        if terminated or truncated:

            raise RuntimeError(
                "Game ended during "
                "horizontal movement."
            )


# ============================================================
# PlacementAction visual executor
# ============================================================

def execute_visual_placement(
    adapter,
    placement_action,
):

    # --------------------------------------------------------
    # HOLD
    # --------------------------------------------------------

    if placement_action.use_hold:

        (
            state,
            reward,
            terminated,
            truncated,
            info,
        ) = visual_step(
            adapter,
            HOLD,
        )

        if terminated or truncated:

            raise RuntimeError(
                "Game ended during Hold."
            )

    # --------------------------------------------------------
    # ROTATION
    # --------------------------------------------------------

    visual_rotation(
        adapter,
        placement_action.rotation,
    )

    # --------------------------------------------------------
    # 找出旋轉後目前的 Canonical X
    # --------------------------------------------------------

    current_x = active_left_x(
        adapter
    )

    target_x = (
        placement_action.x
    )

    delta = (
        target_x
        - current_x
    )

    # --------------------------------------------------------
    # MOVE
    # --------------------------------------------------------

    visual_horizontal_move(
        adapter,
        delta,
    )

    actual_x = active_left_x(
        adapter
    )

    if actual_x != target_x:

        raise RuntimeError(
            "Could not reach requested X: "
            f"target={target_x}, "
            f"actual={actual_x}"
        )

    # --------------------------------------------------------
    # HARD DROP
    # --------------------------------------------------------

    (
        state,
        reward,
        terminated,
        truncated,
        info,
    ) = visual_step(
        adapter,
        HARD_DROP,
        LOCK_DELAY_MS,
    )

    return {
        "state": state,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
        "horizontal_delta": delta,
    }


# ============================================================
# Main
# ============================================================

def main():

    adapter = GymTetrisAdapter(
        seed=SEED,
        render_mode="human",
    )

    teacher = (
        HeuristicTeacherV2()
    )

    state = adapter.reset(
        seed=SEED
    )

    # --------------------------------------------------------
    # 我們看的是策略，不測自然 gravity timing。
    # --------------------------------------------------------

    adapter.raw.gravity_enabled = False

    total_lines = 0
    hold_uses = 0

    print()
    print("=" * 76)
    print(
        "TETRIS AI - WATCH HEURISTIC TEACHER V1"
    )
    print("=" * 76)

    print()
    print(
        "Seed:",
        SEED,
    )

    print(
        "Max pieces:",
        MAX_PIECES,
    )

    print()
    print(
        "Controls:"
    )

    print(
        "  P      Pause / Resume"
    )

    print(
        "  Q/ESC  Stop"
    )

    print()

    # 顯示初始畫面
    render_frame(
        adapter,
        500,
    )

    try:

        for piece_number in range(
            1,
            MAX_PIECES + 1,
        ):

            decision = teacher.choose(
                state
            )

            if decision is None:

                print()
                print(
                    "Teacher has no "
                    "usable placement."
                )

                break

            candidate = (
                decision.candidate
            )

            action = (
                candidate.action
            )

            if action.use_hold:
                hold_uses += 1

            print(
                f"#{piece_number:04d} "
                f"current={state.current_piece} "
                f"hold={state.hold_piece or '-'} "
                f"play={decision.played_piece} "
                f"use_hold={action.use_hold} "
                f"rot={action.rotation} "
                f"x={action.x} "
                f"teacher_score="
                f"{decision.score:.1f}",
                flush=True,
            )

            # ------------------------------------------------
            # 真正在畫面中執行
            # ------------------------------------------------

            result = (
                execute_visual_placement(
                    adapter,
                    action,
                )
            )

            actual = (
                result["state"]
            )

            # ------------------------------------------------
            # 即使是視覺模式，
            # Simulator / Gym 仍必須完全一致
            # ------------------------------------------------

            if not np.array_equal(
                candidate.after_board,
                actual.board,
            ):

                raise RuntimeError(
                    "Simulator/Gym board mismatch "
                    f"at piece {piece_number}"
                )

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

                raise RuntimeError(
                    "Line clear mismatch "
                    f"at piece {piece_number}"
                )

            total_lines += (
                actual_lines
            )

            if actual_lines > 0:

                print(
                    f"       CLEAR="
                    f"{actual_lines} "
                    f"TOTAL_LINES="
                    f"{total_lines}"
                )

            state = actual

            if (
                result["terminated"]
                or result["truncated"]
            ):

                print()
                print(
                    "GAME OVER"
                )

                break

    except KeyboardInterrupt:

        print()
        print(
            "Stopped by user."
        )

    finally:

        print()
        print("=" * 76)

        print(
            "Pieces:",
            piece_number
            if "piece_number" in locals()
            else 0,
        )

        print(
            "Lines:",
            total_lines,
        )

        print(
            "Hold uses:",
            hold_uses,
        )

        print("=" * 76)

        adapter.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()