from __future__ import annotations

import gymnasium as gym
import numpy as np

from singleplayer.game.state import CanonicalState
from tetris_ai.core.tetrominoes import PIECE_NAMES

class GymTetrisAdapter:

    PIECE_NAMES = PIECE_NAMES

    def __init__(
        self,
        seed=42,
        render_mode=None,
    ):

        self.seed = seed

        self.env = gym.make(
            "tetris_gymnasium/Tetris",
            render_mode=render_mode,
        )

        self.raw = self.env.unwrapped

    # --------------------------------------------------------
    # Gym Tetromino ID -> Canonical Piece Name
    # --------------------------------------------------------

    def piece_name_from_tetromino(self, tetromino):

        if tetromino is None:
            return None

        # Gym 會把原本 0~6 的 ID
        # offset 成 2~8。
        index = int(tetromino.id) - 2

        if not 0 <= index < len(self.PIECE_NAMES):
            raise ValueError(
                f"Unknown tetromino id: {tetromino.id}"
            )

        return self.PIECE_NAMES[index]

    # --------------------------------------------------------
    # Queue ID
    #
    # queue 裡保存的是 0~6 index，
    # 不是 offset 後的 2~8。
    # --------------------------------------------------------

    def piece_name_from_queue_id(self, piece_id):

        piece_id = int(piece_id)

        if not 0 <= piece_id < len(self.PIECE_NAMES):
            raise ValueError(
                f"Unknown queue piece id: {piece_id}"
            )

        return self.PIECE_NAMES[piece_id]

    # --------------------------------------------------------
    # Gym board -> Canonical 20×10 board
    # --------------------------------------------------------

    def canonical_board(self):

        raw_board = self.raw.board

        padding = self.raw.padding
        height = self.raw.height
        width = self.raw.width

        playfield = raw_board[
            0:height,
            padding:padding + width,
        ]

        # AI 不需要知道方塊顏色。
        #
        # Gym:
        #   0 = empty
        #   >=2 = 已鎖定 tetromino
        #
        # Canonical:
        #   0 = empty
        #   1 = occupied
        board = (playfield >= 2).astype(np.uint8)

        return board.copy()

    # --------------------------------------------------------
    # Current
    # --------------------------------------------------------

    def current_piece(self):

        return self.piece_name_from_tetromino(
            self.raw.active_tetromino
        )

    # --------------------------------------------------------
    # Queue
    # --------------------------------------------------------

    def next_pieces(self):

        ids = self.raw.queue.get_queue()

        return tuple(
            self.piece_name_from_queue_id(piece_id)
            for piece_id in ids
        )

    # --------------------------------------------------------
    # Hold
    # --------------------------------------------------------

    def hold_piece(self):

        pieces = self.raw.holder.get_tetrominoes()

        if len(pieces) == 0:
            return None

        return self.piece_name_from_tetromino(
            pieces[0]
        )

    # --------------------------------------------------------
    # Canonical State
    # --------------------------------------------------------

    def state(self):

        return CanonicalState(
            board=self.canonical_board(),

            current_piece=self.current_piece(),

            next_pieces=self.next_pieces(),

            hold_piece=self.hold_piece(),

            can_hold=not self.raw.has_swapped,
        )

    def reset(self, seed=None):

        if seed is None:
            seed = self.seed

        self.env.reset(seed=seed)

        return self.state()

    def gym_step(self, action):

        obs, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        return (
            self.state(),
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        self.env.close()


