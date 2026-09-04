from statistics import mean, median

from tetris_core import (
    GymTetrisAdapter,
)

from gym_executor import (
    execute_placement,
    choose_reachable_teacher_decision,
)

from teacher import (
    HeuristicTeacherV2,
)


# ============================================================
# Benchmark settings
# ============================================================

# 第一輪先跑 20。
#
# 確認一切正常後可以改成：
#
# GAMES = 100
#
GAMES = 20

# 避免某一局玩超久導致 benchmark 永遠跑不完。
MAX_PIECES_PER_GAME = 2000


def play_one_game(
    adapter,
    teacher,
    seed,
):

    state = adapter.reset(
        seed=seed
    )

    # 我們現在評估的是策略，
    # 不測 timing。
    adapter.raw.gravity_enabled = False

    pieces = 0
    lines = 0

    single_clears = 0
    double_clears = 0
    triple_clears = 0
    tetris_clears = 0

    hold_uses = 0

    max_height_seen = 0
    max_holes_seen = 0

    ended = False

    for _ in range(
        MAX_PIECES_PER_GAME
    ):

        decision = (
            choose_reachable_teacher_decision(
                adapter=adapter,
                teacher=teacher,
                state=state,
            )
        )

        if decision is None:
            ended = True
            break

        candidate = (
            decision.candidate
        )

        action = (
            candidate.action
        )

        if action.use_hold:
            hold_uses += 1

        result = execute_placement(
            adapter,
            action,
        )

        actual_lines = int(
            result["info"].get(
                "lines_cleared",
                0,
            )
        )

        # 我們之前已驗證 simulator == Gym，
        # 這裡再保留 assertion 防止未來修改弄壞。
        assert (
            actual_lines
            == candidate.lines_cleared
        )

        pieces += 1
        lines += actual_lines

        if actual_lines == 1:
            single_clears += 1

        elif actual_lines == 2:
            double_clears += 1

        elif actual_lines == 3:
            triple_clears += 1

        elif actual_lines == 4:
            tetris_clears += 1

        max_height_seen = max(
            max_height_seen,
            decision.features.max_height,
        )

        max_holes_seen = max(
            max_holes_seen,
            decision.features.holes,
        )

        state = result["state"]

        if (
            result["terminated"]
            or result["truncated"]
        ):
            ended = True
            break

    return {
        "seed": seed,
        "pieces": pieces,
        "lines": lines,

        "single": single_clears,
        "double": double_clears,
        "triple": triple_clears,
        "tetris": tetris_clears,

        "hold_uses": hold_uses,

        "max_height": max_height_seen,
        "max_holes": max_holes_seen,

        "ended": ended,
        "hit_piece_limit": (
            pieces
            >= MAX_PIECES_PER_GAME
        ),
    }


def main():

    adapter = GymTetrisAdapter()

    teacher = HeuristicTeacherV2()

    print()
    print("=" * 76)
    print("TETRIS AI - HEURISTIC TEACHER BENCHMARK")
    print("=" * 76)

    print()
    print(
        "Teacher:",
        teacher.name,
    )

    print(
        "Games:",
        GAMES,
    )

    print(
        "Max pieces/game:",
        MAX_PIECES_PER_GAME,
    )

    results = []

    # seed 0 不使用。
    #
    # 之前已確認 Gym randomizer 對 seed=0
    # 有特殊問題。
    for seed in range(
        1,
        GAMES + 1,
    ):

        result = play_one_game(
            adapter=adapter,
            teacher=teacher,
            seed=seed,
        )

        results.append(
            result
        )

        status = (
            "LIMIT"
            if result["hit_piece_limit"]
            else "GAMEOVER"
        )

        print(
            f"seed={seed:>3} "
            f"pieces={result['pieces']:>5} "
            f"lines={result['lines']:>5} "
            f"hold={result['hold_uses']:>5} "
            f"maxH={result['max_height']:>2} "
            f"maxHole={result['max_holes']:>3} "
            f"{status}"
        )

    adapter.close()

    # ========================================================
    # Aggregate statistics
    # ========================================================

    pieces_values = [
        r["pieces"]
        for r in results
    ]

    line_values = [
        r["lines"]
        for r in results
    ]

    hold_values = [
        r["hold_uses"]
        for r in results
    ]

    total_single = sum(
        r["single"]
        for r in results
    )

    total_double = sum(
        r["double"]
        for r in results
    )

    total_triple = sum(
        r["triple"]
        for r in results
    )

    total_tetris = sum(
        r["tetris"]
        for r in results
    )

    survived_limit = sum(
        r["hit_piece_limit"]
        for r in results
    )

    print()
    print("=" * 76)
    print("SUMMARY")
    print("=" * 76)

    print()
    print(
        "Average pieces :",
        f"{mean(pieces_values):.1f}",
    )

    print(
        "Median pieces  :",
        f"{median(pieces_values):.1f}",
    )

    print(
        "Best pieces    :",
        max(pieces_values),
    )

    print(
        "Worst pieces   :",
        min(pieces_values),
    )

    print()

    print(
        "Average lines  :",
        f"{mean(line_values):.1f}",
    )

    print(
        "Median lines   :",
        f"{median(line_values):.1f}",
    )

    print(
        "Best lines     :",
        max(line_values),
    )

    print(
        "Worst lines    :",
        min(line_values),
    )

    print()

    print(
        "Single clears  :",
        total_single,
    )

    print(
        "Double clears  :",
        total_double,
    )

    print(
        "Triple clears  :",
        total_triple,
    )

    print(
        "Tetris clears  :",
        total_tetris,
    )

    print()

    print(
        "Average Hold uses:",
        f"{mean(hold_values):.1f}",
    )

    print()

    print(
        "Reached piece limit:",
        survived_limit,
        "/",
        GAMES,
    )

    print()
    print("=" * 76)
    print("BENCHMARK COMPLETE")
    print("=" * 76)


if __name__ == "__main__":
    main()