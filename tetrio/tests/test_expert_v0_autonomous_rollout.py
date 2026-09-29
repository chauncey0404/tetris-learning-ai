from __future__ import annotations

import unittest

from tetrio.tools.watch_expert_v0 import (
    Decision,
    BranchPlan,
    ScoredCandidate,
    SevenBagQueue,
    parse_seed_spec,
)


class ExpertV0AutonomousRolloutTests(unittest.TestCase):
    def test_seed_range_parser(self):
        self.assertEqual(parse_seed_spec("9001-9003", 1), [9001, 9002, 9003])
        self.assertEqual(parse_seed_spec("9001,9003", 1), [9001, 9003])
        self.assertEqual(parse_seed_spec("", 9001), [9001])

    def test_every_seven_bag_contains_all_pieces_once(self):
        stream = SevenBagQueue(9001)
        first = [stream.pop() for _ in range(7)]
        second = [stream.pop() for _ in range(7)]
        self.assertEqual(set(first), set("IOTSZJL"))
        self.assertEqual(set(second), set("IOTSZJL"))

    def test_peek_does_not_consume(self):
        stream = SevenBagQueue(9001)
        before = stream.peek(7)
        popped = tuple(stream.pop() for _ in range(7))
        self.assertEqual(before, popped)

    def test_avoidable_hole_flag(self):
        dummy_state = None
        # Avoid constructing full candidate objects here; the property depends
        # only on the stored hole diagnostics.
        decision = object.__new__(Decision)
        object.__setattr__(decision, "chosen_hole_delta", 1)
        object.__setattr__(decision, "min_candidate_holes", 2)
        object.__setattr__(decision, "holes_before", 2)
        self.assertTrue(decision.avoidable_hole)

        decision2 = object.__new__(Decision)
        object.__setattr__(decision2, "chosen_hole_delta", 1)
        object.__setattr__(decision2, "min_candidate_holes", 3)
        object.__setattr__(decision2, "holes_before", 2)
        self.assertFalse(decision2.avoidable_hole)


if __name__ == "__main__":
    unittest.main()
