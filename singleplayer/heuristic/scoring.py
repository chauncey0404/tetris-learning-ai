from singleplayer.heuristic.features import extract_board_features_v2
from singleplayer.heuristic.i_dependency import analyze_i_dependency

LINE_CLEAR_REWARD = {
    0: 0.0,
    1: 900.0,
    2: 2000.0,
    3: 3300.0,
    4: 6000.0,
}


def score_candidate_v2(
    state,
    candidate,
):

    features = (
        extract_board_features_v2(
            candidate.after_board
        )
    )

    i_info = (
        analyze_i_dependency(
            state,
            candidate,
            features,
        )
    )

    score = 0.0

    # ========================================================
    # 1. Line Clear
    #
    # 非線性。
    #
    # 讓 Tetris 明顯比四次 single 有價值。
    # ========================================================

    score += LINE_CLEAR_REWARD.get(
        int(
            candidate.lines_cleared
        ),

        float(
            candidate.lines_cleared
            * 1000
        ),
    )

    # ========================================================
    # 2. Holes
    # ========================================================

    score -= (
        features.holes
        * 250.0
    )

    # ========================================================
    # 3. General board height
    # ========================================================

    score -= (
        features.aggregate_height
        * 4.0
    )

    # ========================================================
    # 4. Bumpiness
    # ========================================================

    score -= (
        features.bumpiness
        * 3.0
    )

    # ========================================================
    # 5. Max height
    # ========================================================

    score -= (
        features.max_height
        * 8.0
    )

    # ========================================================
    # 6. Danger height
    #
    # 10 以下：
    #     一般狀態
    #
    # >10：
    #     非線性增加風險
    # ========================================================

    if (
        features.max_height
        > 10
    ):

        danger = (
            features.max_height
            - 10
        )

        score -= (
            danger
            * danger
            * 30.0
        )

    # --------------------------------------------------------
    # Emergency zone
    # --------------------------------------------------------

    if (
        features.max_height
        >= 16
    ):

        score -= (
            (
                features.max_height
                - 15
            )
            * 500.0
        )

    # ========================================================
    # 7. Center Tower
    #
    # Seed 6 的主要 failure mode。
    #
    # 單側井不會被誤傷。
    # ========================================================

    if (
        features.center_tower_excess
        > 3
    ):

        excess = (
            features.center_tower_excess
            - 3
        )

        score -= (
            excess
            * excess
            * 160.0
        )

    # ========================================================
    # 8. Multiple Deep Wells
    #
    # 一個深井：
    #     可能是正常 Tetris strategy。
    #
    # 多個：
    #     同時欠很多 I，危險。
    # ========================================================

    if (
        features.deep_wells
        >= 2
    ):

        score -= (
            700.0
            * (
                features.deep_wells
                - 1
            )
        )

    # ========================================================
    # 9. Both-edge I dependency
    #
    # Seed 6 精準防護：
    #
    # .████████.
    # .████████.
    # .████████.
    # .████████.
    # ========================================================

    if (
        features.left_well_depth
        >= 4

        and

        features.right_well_depth
        >= 4
    ):

        score -= 1200.0

    # ========================================================
    # 10. Extremely deep single well
    #
    # depth 4~7：
    #     還可以視為一個 I strategy。
    #
    # >7：
    #     開始需要多支 I。
    # ========================================================

    if (
        features.max_well_depth
        > 7
    ):

        excess_depth = (
            features.max_well_depth
            - 7
        )

        score -= (
            excess_depth
            * excess_depth
            * 90.0
        )

    # ========================================================
    # 11. Uncovered I Debt
    #
    # 重點不是：
    #
    #     有井 = 壞
    #
    # 而是：
    #
    #     欠的 I
    #       >
    #     Hold + Next 已知供給
    #
    # 才真正危險。
    # ========================================================

    if (
        i_info.uncovered_i_debt
        > 0
    ):

        # ----------------------------------------------------
        # 健康單側 Tetris well
        #
        # 即使 preview 暫時沒有 I，
        # 也不要過度處罰。
        # ----------------------------------------------------

        if (
            i_info.clean_edge_tetris_well
        ):

            score -= (
                i_info.uncovered_i_debt
                * 120.0
            )

        else:

            score -= (
                i_info.uncovered_i_debt
                * 500.0
            )

            # ------------------------------------------------
            # 同時欠 >= 2 支 I：
            # 再增加非線性風險。
            # ------------------------------------------------

            if (
                i_info.uncovered_i_debt
                >= 2
            ):

                extra = (
                    i_info.uncovered_i_debt
                    - 1
                )

                score -= (
                    extra
                    * extra
                    * 350.0
                )

    # ========================================================
    # 12. High board + I debt
    #
    # 棋盤低時欠一支 I 還好。
    #
    # 高度已經危險，
    # 還欠 I 就要更保守。
    # ========================================================

    if (
        i_info.i_debt
        > 0

        and

        features.max_height
        > 12
    ):

        height_risk = (
            features.max_height
            - 12
        )

        score -= (
            i_info.i_debt
            * height_risk
            * 70.0
        )

    # ========================================================
    # 13. Tetris Well Potential - V2.1
    #
    # 這是本次最重要的修改。
    #
    # 不再等 depth=4 才突然承認：
    #
    #     「這是 Tetris well」
    #
    # 而是讓老師理解：
    #
    # depth 1 -> 正在建
    # depth 2 -> 更接近
    # depth 3 -> 很接近
    # depth 4~7 -> 已完成一個可用 Tetris well
    # ========================================================

    if (
        i_info.clean_edge_tetris_well
    ):

        well_depth = max(
            features.left_well_depth,
            features.right_well_depth,
        )

        # ----------------------------------------------------
        # 正在建立 Tetris well
        # ----------------------------------------------------

        if well_depth == 1:

            score += 120.0

        elif well_depth == 2:

            score += 300.0

        elif well_depth == 3:

            score += 600.0

        # ----------------------------------------------------
        # 已形成完整 Tetris-ready well
        #
        # depth 4~7：
        # 一支 I 可以至少把最主要的四行問題處理掉。
        # ----------------------------------------------------

        elif (
            4
            <= well_depth
            <= 7
        ):

            score += 1100.0

        # ----------------------------------------------------
        # 已知 Hold / Queue 中看得到 I
        #
        # 這個 setup 的實際風險更低。
        # ----------------------------------------------------

        if (
            i_info.visible_i_supply
            >= 1
        ):

            score += 450.0

    # ========================================================
    # 14. Top-out
    # ========================================================

    if candidate.top_out:

        score -= 100000.0

    return (
        score,
        features,
        i_info,
    )
