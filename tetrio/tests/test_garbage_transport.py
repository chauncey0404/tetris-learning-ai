from __future__ import annotations

import unittest

import numpy as np

from tetrio.battle_state import BattleState
from tetrio.clear import ClearEvent
from tetrio.garbage_board import insert_garbage_rows
from tetrio.garbage_queue import GarbageQueue
from tetrio.versus import (
    TETRA_LEAGUE_TRANSPORT_REFERENCE,
    VersusPlayerState,
    VersusTransportConfig,
    resolve_clear_exchange,
    tank_active_garbage,
)


class TetrioGarbageTransportTests(unittest.TestCase):
    def test_passthrough_off_reference_and_20f_travel(self):
        self.assertFalse(TETRA_LEAGUE_TRANSPORT_REFERENCE.passthrough)
        self.assertEqual(TETRA_LEAGUE_TRANSPORT_REFERENCE.travel_frames, 20)

    def test_packet_is_cancelable_before_it_is_active(self):
        q = GarbageQueue().enqueue_packets((4,), sent_frame=100, travel_frames=20, holes=(3,))
        self.assertEqual(q.active_lines(119), 0)
        result = q.cancel_lines(2)
        self.assertEqual(result.cancelled, 2)
        self.assertEqual(sum(p.lines for p in result.packets_after), 2)

    def test_packet_activates_at_twenty_frames(self):
        q = GarbageQueue().enqueue_packets((4,), sent_frame=100, travel_frames=20, holes=(3,))
        self.assertEqual(q.active_lines(119), 0)
        self.assertEqual(q.active_lines(120), 4)

    def test_fifo_cancellation_preserves_packet_order(self):
        q = GarbageQueue().enqueue_packets((2, 4), sent_frame=0, holes=(1, 8))
        result = q.cancel_lines(3)
        self.assertEqual(result.cancelled, 3)
        self.assertEqual([(p.lines, p.hole) for p in result.packets_after], [(3, 8)])

    def test_active_pop_respects_profile_cap(self):
        q = GarbageQueue().enqueue_packets((6,), sent_frame=0, travel_frames=0, holes=(4,))
        popped = q.pop_active(0, cap=3)
        self.assertEqual(popped.lines, 3)
        self.assertEqual(popped.holes, (4, 4, 4))
        self.assertEqual(sum(p.lines for p in popped.packets_after), 3)

    def test_board_insertion_is_bottom_up_and_detects_topout(self):
        board = np.zeros((40, 10), dtype=np.uint8)
        board[0, 0] = 1
        result = insert_garbage_rows(board, (2, 7))
        self.assertTrue(result.top_out)
        self.assertEqual(result.lines_inserted, 2)
        self.assertEqual(result.board[-2, 2], 0)
        self.assertEqual(result.board[-1, 7], 0)
        self.assertEqual(int(result.board[-2].sum()), 9)
        self.assertEqual(int(result.board[-1].sum()), 9)

    def test_attack_is_delivered_to_opponent_queue(self):
        a = VersusPlayerState(board=np.zeros((40, 10), dtype=np.uint8), frame=50)
        b = VersusPlayerState(board=np.zeros((40, 10), dtype=np.uint8), frame=50)
        # Quad sends base 4 on a fresh combo/B2B state.
        result = resolve_clear_exchange(a, b, ClearEvent(piece="I", lines=4))
        self.assertEqual(sum(result.packets_delivered), 4)
        self.assertEqual(result.opponent_after.pending_garbage, 4)
        packet = result.opponent_after.incoming.packets[0]
        self.assertEqual(packet.sent_frame, 50)
        self.assertEqual(packet.active_frame, 70)

    def test_outgoing_attack_cancels_own_queue_before_sending_remainder(self):
        q = GarbageQueue().enqueue_packets((2,), sent_frame=0, travel_frames=20, holes=(5,))
        a = VersusPlayerState(
            board=np.zeros((40, 10), dtype=np.uint8),
            battle=BattleState(incoming_pending=2),
            incoming=q,
            frame=5,
        )
        b = VersusPlayerState(board=np.zeros((40, 10), dtype=np.uint8), frame=5)
        result = resolve_clear_exchange(a, b, ClearEvent(piece="I", lines=4))
        self.assertEqual(result.actor_after.pending_garbage, 0)
        self.assertEqual(sum(result.packets_delivered), 3)
        self.assertEqual(result.opponent_after.pending_garbage, 3)
        self.assertTrue(result.battle_step.cancellation.opener_double_cancel)

    def test_tank_requires_explicit_hole_when_not_yet_validated(self):
        q = GarbageQueue().enqueue_packets((2,), sent_frame=0, travel_frames=0)
        p = VersusPlayerState(
            board=np.zeros((40, 10), dtype=np.uint8),
            battle=BattleState(incoming_pending=2),
            incoming=q,
            frame=0,
        )
        with self.assertRaises(ValueError):
            tank_active_garbage(p)

    def test_tank_active_garbage_with_explicit_packet_hole(self):
        q = GarbageQueue().enqueue_packets((2,), sent_frame=0, travel_frames=0, holes=(6,))
        p = VersusPlayerState(
            board=np.zeros((40, 10), dtype=np.uint8),
            battle=BattleState(incoming_pending=2),
            incoming=q,
            frame=0,
        )
        after, inserted = tank_active_garbage(p)
        self.assertEqual(inserted.lines_inserted, 2)
        self.assertEqual(after.pending_garbage, 0)
        self.assertEqual(after.board[-1, 6], 0)

    def test_configurable_cap_not_hardcoded_as_ranked_fact(self):
        config = VersusTransportConfig(travel_frames=20, passthrough=False, garbage_cap_per_piece=2)
        q = GarbageQueue().enqueue_packets((5,), sent_frame=0, travel_frames=0, holes=(4,))
        p = VersusPlayerState(
            board=np.zeros((40, 10), dtype=np.uint8),
            battle=BattleState(incoming_pending=5),
            incoming=q,
            frame=0,
        )
        after, inserted = tank_active_garbage(p, transport=config)
        self.assertEqual(inserted.lines_inserted, 2)
        self.assertEqual(after.pending_garbage, 3)


if __name__ == "__main__":
    unittest.main()
