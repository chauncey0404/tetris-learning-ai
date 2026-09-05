import numpy as np


PIECES = (
    "I",
    "O",
    "T",
    "S",
    "Z",
    "J",
    "L",
)

PIECE_TO_ID = {
    piece: index
    for index, piece in enumerate(PIECES)
}


def one_hot_piece(piece):
    """
    7-dimensional piece one-hot.

    None -> all zeros.
    """

    result = np.zeros(
        7,
        dtype=np.float32,
    )

    if piece is None:
        return result

    result[
        PIECE_TO_ID[piece]
    ] = 1.0

    return result


def encode_state(state):
    """
    CanonicalState -> neural-network state vector.

    Layout:

        board:
            20 x 10 = 200

        current:
            7

        hold:
            7

        next queue:
            4 x 7 = 28

        can_hold:
            1

    Total:
        243 floats
    """

    board = np.asarray(
        state.board,
        dtype=np.float32,
    ).reshape(-1)

    current = one_hot_piece(
        state.current_piece
    )

    hold = one_hot_piece(
        state.hold_piece
    )

    next_encoded = []

    for piece in state.next_pieces:

        next_encoded.append(
            one_hot_piece(piece)
        )

    while len(next_encoded) < 4:

        next_encoded.append(
            np.zeros(
                7,
                dtype=np.float32,
            )
        )

    next_encoded = np.concatenate(
        next_encoded[:4]
    )

    can_hold = np.asarray(
        [
            float(state.can_hold)
        ],
        dtype=np.float32,
    )

    encoded = np.concatenate(
        [
            board,
            current,
            hold,
            next_encoded,
            can_hold,
        ]
    )

    assert encoded.shape == (243,)

    return encoded


def encode_legacy_candidate_216(candidate):
    """
    Candidate 本身的特徵。

    我們不只給 AI rotation/x，
    也讓它直接看到 candidate 放完後的 board。

    Layout:

        after_board:
            200

        rotation one-hot:
            4

        x one-hot:
            10

        use_hold:
            1

        lines_cleared:
            1

    Total:
        216
    """

    board = np.asarray(
        candidate.after_board,
        dtype=np.float32,
    ).reshape(-1)

    rotation = np.zeros(
        4,
        dtype=np.float32,
    )

    rotation[
        candidate.action.rotation
    ] = 1.0

    x_position = np.zeros(
        10,
        dtype=np.float32,
    )

    x_position[
        candidate.action.x
    ] = 1.0

    use_hold = np.asarray(
        [
            float(
                candidate.action.use_hold
            )
        ],
        dtype=np.float32,
    )

    lines = np.asarray(
        [
            float(
                candidate.lines_cleared
            ) / 4.0
        ],
        dtype=np.float32,
    )

    encoded = np.concatenate(
        [
            board,
            rotation,
            x_position,
            use_hold,
            lines,
        ]
    )

    assert encoded.shape == (216,)

    return encoded

# Backward-compatible alias. Production V8.4+ Q input uses the observable 215-d
# representation from singleplayer.network.candidates, not this legacy 216-d encoder.
encode_candidate = encode_legacy_candidate_216
