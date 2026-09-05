from dataclasses import dataclass

from singleplayer.heuristic.features import BoardFeaturesV2

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
