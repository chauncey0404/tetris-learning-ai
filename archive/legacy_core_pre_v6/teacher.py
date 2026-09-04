from dataclasses import dataclass

import numpy as np

from tetris_placement import (
    enumerate_state_placements,
    piece_for_placement,
)


# ============================================================
# HEURISTIC TEACHER V2.1
#
# 核心目標：
#
# 1. 避免 Seed 6：
#
#       .████████.
#       .████████.
#       .████████.
#       .████████.
#
#    左右兩側同時欠 I，
#    中間越堆越高。
#
# 2. 不要錯殺正常 Tetris well：
#
#       █████████.
#       █████████.
#       █████████.
#       █████████.
#
# 3. 能理解「正在建立 Tetris well」也有價值：
#
#       depth 1
#       depth 2
#       depth 3
#       depth 4
#
# 4. Hold / Next 已知有 I 時，
#    單側 Tetris well 的風險比較低。
#
# 5. 這些 heuristic 只是老師，
#    未來 Neural Network + RL 可以超越它。
# ============================================================


# ============================================================
# Line-clear preference
#
# 故意不是線性：
#
# 1 line  != 1000
# 4 lines != 4000
#
# Tetris 應該比四次 single 更有戰略價值。
# ============================================================

LINE_CLEAR_REWARD = {
    0: 0.0,
    1: 900.0,
    2: 2000.0,
    3: 3300.0,
    4: 6000.0,
}


# ============================================================
# Feature structures
# ============================================================

@dataclass(frozen=True)
class BoardFeaturesV2:
    aggregate_height: int
    max_height: int

    holes: int
    bumpiness: int

    column_heights: tuple[int, ...]
    well_depths: tuple[int, ...]

    total_well_depth: int
    max_well_depth: int

    deep_wells: int

    left_well_depth: int
    right_well_depth: int

    # 中間最高處相對於兩側最高支撐的超出高度。
    #
    # Seed 6：
    #
    # .████████.
    #
    # 中間很高、左右都低，
    # 這個值會變大。
    center_tower_excess: int


@dataclass(frozen=True)
class IDependencyAnalysis:
    # 估計目前深井需要多少支 I 才能解掉。
    i_debt: int

    # Hold / Queue 中目前看得到多少支 I。
    visible_i_supply: int

    # 欠的 I 扣除目前已知供給。
    uncovered_i_debt: int

    # 是否為健康的「單側」Tetris well。
    #
    # V2.1：
    # depth 1~7 都可以被辨識，
    # 不再只有 depth >= 4。
    clean_edge_tetris_well: bool


@dataclass(frozen=True)
class TeacherDecisionV2:
    candidate: object

    played_piece: str

    score: float

    features: BoardFeaturesV2

    i_dependency: IDependencyAnalysis


# ============================================================
# Column heights
# ============================================================

def get_column_heights(board):

    board = np.asarray(
        board,
        dtype=np.uint8,
    )

    height, width = board.shape

    heights = []

    for x in range(width):

        column = board[:, x]

        occupied = np.flatnonzero(
            column
        )

        if len(occupied) == 0:

            heights.append(0)

        else:

            first_occupied = int(
                occupied[0]
            )

            heights.append(
                height - first_occupied
            )

    return heights


# ============================================================
# Board feature extraction
# ============================================================

def extract_board_features_v2(board):

    board = np.asarray(
        board,
        dtype=np.uint8,
    )

    height, width = board.shape

    column_heights = (
        get_column_heights(
            board
        )
    )

    # --------------------------------------------------------
    # Holes
    #
    # 第一個 occupied cell 以下的空格，
    # 都視為 hole。
    # --------------------------------------------------------

    holes = 0

    for x in range(width):

        column = board[:, x]

        occupied = np.flatnonzero(
            column
        )

        if len(occupied) == 0:
            continue

        first_occupied = int(
            occupied[0]
        )

        holes += int(
            np.count_nonzero(
                column[
                    first_occupied:
                ] == 0
            )
        )

    # --------------------------------------------------------
    # Height
    # --------------------------------------------------------

    aggregate_height = int(
        sum(column_heights)
    )

    max_height = int(
        max(column_heights)
    )

    # --------------------------------------------------------
    # Bumpiness
    # --------------------------------------------------------

    bumpiness = int(
        sum(
            abs(
                column_heights[i]
                - column_heights[i + 1]
            )
            for i in range(
                width - 1
            )
        )
    )

    # --------------------------------------------------------
    # Wells
    #
    # Interior column:
    #
    #   depth =
    #       min(left_height, right_height)
    #       - current_height
    #
    # Edge column:
    #
    #   只和唯一的鄰居比較。
    #
    # 因此：
    #
    # █████████.
    #
    # 最右欄可以自然形成合法 Tetris well。
    # --------------------------------------------------------

    well_depths = []

    for x in range(width):

        current = column_heights[x]

        if x == 0:

            depth = max(
                0,
                column_heights[1]
                - current,
            )

        elif x == width - 1:

            depth = max(
                0,
                column_heights[
                    width - 2
                ]
                - current,
            )

        else:

            wall_height = min(
                column_heights[x - 1],
                column_heights[x + 1],
            )

            depth = max(
                0,
                wall_height
                - current,
            )

        well_depths.append(
            int(depth)
        )

    total_well_depth = int(
        sum(well_depths)
    )

    max_well_depth = int(
        max(well_depths)
    )

    deep_wells = int(
        sum(
            depth >= 4
            for depth
            in well_depths
        )
    )

    # --------------------------------------------------------
    # Center tower detection
    #
    # 正常單側 Tetris well：
    #
    # █████████.
    #
    # 中間高度 ≈ 左側高度，
    # 不會被當成 center tower。
    #
    #
    # Seed 6 型態：
    #
    # .████████.
    #
    # 左右都低，中間高，
    # center_tower_excess 會很大。
    # --------------------------------------------------------

    if width >= 3:

        center_max = max(
            column_heights[
                1:-1
            ]
        )

        edge_support = max(
            column_heights[0],
            column_heights[-1],
        )

        center_tower_excess = max(
            0,
            center_max
            - edge_support,
        )

    else:

        center_tower_excess = 0

    return BoardFeaturesV2(
        aggregate_height=aggregate_height,

        max_height=max_height,

        holes=holes,

        bumpiness=bumpiness,

        column_heights=tuple(
            int(value)
            for value
            in column_heights
        ),

        well_depths=tuple(
            int(value)
            for value
            in well_depths
        ),

        total_well_depth=(
            total_well_depth
        ),

        max_well_depth=(
            max_well_depth
        ),

        deep_wells=deep_wells,

        left_well_depth=int(
            well_depths[0]
        ),

        right_well_depth=int(
            well_depths[-1]
        ),

        center_tower_excess=int(
            center_tower_excess
        ),
    )


# ============================================================
# Known future pieces after candidate
# ============================================================

def known_future_pieces(
    state,
    candidate,
):
    """
    根據 placement 是否使用 Hold，
    估算完成這一手後仍然確定知道的 piece。

    ----------------------------------------------------------

    A. 不 Hold

        current 被放掉

        Known:
            hold
            + 原本 queue

    ----------------------------------------------------------

    B. Hold 已經有 piece

        current -> hold
        old hold -> 被放掉

        Known:
            原 current
            + 原 queue

    ----------------------------------------------------------

    C. Hold 原本是空的

        current -> hold
        queue[0] -> 被放掉

        Known:
            原 current
            + queue[1:]

    ----------------------------------------------------------
    """

    queue = list(
        state.next_pieces
    )

    # --------------------------------------------------------
    # No Hold
    # --------------------------------------------------------

    if not candidate.action.use_hold:

        pieces = []

        if state.hold_piece is not None:

            pieces.append(
                state.hold_piece
            )

        pieces.extend(
            queue
        )

        return tuple(
            pieces
        )

    # --------------------------------------------------------
    # Hold already contains piece
    # --------------------------------------------------------

    if state.hold_piece is not None:

        return tuple(
            [
                state.current_piece
            ]
            + queue
        )

    # --------------------------------------------------------
    # Empty Hold
    #
    # queue[0] 被拿出來並在本手放掉。
    # --------------------------------------------------------

    return tuple(
        [
            state.current_piece
        ]
        + queue[1:]
    )


# ============================================================
# I-piece dependency analysis
# ============================================================

def analyze_i_dependency(
    state,
    candidate,
    features,
):

    # --------------------------------------------------------
    # I debt
    #
    # depth 4~7:
    #     約欠 1 支 I
    #
    # depth 8~11:
    #     約欠 2 支 I
    #
    # depth 12~15:
    #     約欠 3 支 I
    #
    # 使用 floor(depth / 4)。
    # --------------------------------------------------------

    i_debt = int(
        sum(
            depth // 4

            for depth
            in features.well_depths

            if depth >= 4
        )
    )

    # --------------------------------------------------------
    # Visible I supply
    # --------------------------------------------------------

    future_pieces = (
        known_future_pieces(
            state,
            candidate,
        )
    )

    visible_i_supply = int(
        sum(
            piece == "I"

            for piece
            in future_pieces
        )
    )

    uncovered_i_debt = max(
        0,
        i_debt
        - visible_i_supply,
    )

    # ========================================================
    # Clean edge Tetris well - V2.1
    #
    # 不再要求 depth >= 4。
    #
    # 這樣：
    #
    # depth 1
    # depth 2
    # depth 3
    #
    # 的「建井階段」
    # 也能被老師理解。
    #
    # 條件：
    #
    # 1. 只有左或右其中一側有 edge well
    # 2. 沒有 holes
    # 3. 其他地方沒有另一個 deep well
    # 4. edge depth <= 7
    #
    # >7 開始有 multi-I dependency 風險。
    # ========================================================

    clean_edge_tetris_well = False

    edge_wells = []

    if (
        features.left_well_depth
        > 0
    ):

        edge_wells.append(
            (
                0,
                features.left_well_depth,
            )
        )

    if (
        features.right_well_depth
        > 0
    ):

        edge_wells.append(
            (
                len(
                    features.well_depths
                ) - 1,

                features.right_well_depth,
            )
        )

    # 只能有一側 edge well
    if (
        len(edge_wells) == 1
        and
        features.holes == 0
    ):

        (
            edge_position,
            edge_depth,
        ) = edge_wells[0]

        # 找其他 deep well
        other_deep_wells = [

            x

            for x, depth
            in enumerate(
                features.well_depths
            )

            if (
                x != edge_position
                and
                depth >= 4
            )
        ]

        if (
            not other_deep_wells
            and
            edge_depth <= 7
        ):

            clean_edge_tetris_well = True

    return IDependencyAnalysis(
        i_debt=i_debt,

        visible_i_supply=(
            visible_i_supply
        ),

        uncovered_i_debt=(
            uncovered_i_debt
        ),

        clean_edge_tetris_well=(
            clean_edge_tetris_well
        ),
    )


# ============================================================
# Candidate scoring
# ============================================================

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


# ============================================================
# Teacher
# ============================================================

class HeuristicTeacherV2:

    name = (
        "HEURISTIC_TEACHER_V2_1_"
        "TETRIS_POTENTIAL_I_DEPENDENCY"
    )

    def rank(
        self,
        state,
    ):
        """
        回傳所有 candidate，
        依 Teacher 分數由高到低排序。

        choose() 仍然只是拿第一名。

        Gym / 真正遊戲 Adapter 可以從 rank()
        中挑出「實際可到達」的最高分 placement。
        """

        candidates = (
            enumerate_state_placements(
                state
            )
        )

        if not candidates:
            return []

        scored = []

        for candidate in candidates:

            if candidate.top_out:
                continue

            (
                score,
                features,
                i_info,
            ) = score_candidate_v2(
                state,
                candidate,
            )

            played_piece = (
                piece_for_placement(
                    state,
                    candidate.action,
                )
            )

            scored.append(
                TeacherDecisionV2(
                    candidate=candidate,
                    played_piece=played_piece,
                    score=score,
                    features=features,
                    i_dependency=i_info,
                )
            )

        scored.sort(
            key=lambda decision: (
                decision.score,
                decision.candidate.lines_cleared,
                -int(
                    decision
                    .candidate
                    .action
                    .use_hold
                ),
                -decision
                .candidate
                .action
                .rotation,
                -decision
                .candidate
                .action
                .x,
            ),
            reverse=True,
        )

        return scored

    def choose(
        self,
        state,
    ):

        ranked = self.rank(
            state
        )

        if not ranked:
            return None

        return ranked[0]