from dataclasses import dataclass
import numpy as np

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
