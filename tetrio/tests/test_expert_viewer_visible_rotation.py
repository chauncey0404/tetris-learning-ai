from __future__ import annotations

import unittest

from tetrio.reachability import tetrio_spawn_state
from tetrio.tools.watch_expert_v0 import VisualDrop


class ExpertViewerVisibleRotationTests(unittest.TestCase):
    def test_r3_uses_visible_ccw_step(self):
        self.assertEqual(
            VisualDrop._rotation_steps(3),
            (0, 3),
        )

    def test_r2_has_two_visible_quarter_turns(self):
        self.assertEqual(
            VisualDrop._rotation_steps(2),
            (0, 1, 2),
        )

    def test_rotation_phase_is_visible_row(self):
        drop = VisualDrop(
            piece="T",
            target_rotation=3,
            target_x=6,
            landing_y=34,
            display_hold=None,
            display_preview=(),
            started_at=100.0,
            duration=1.0,
        )

        # Middle of rotation phase.
        pose = drop.pose(100.35)
        self.assertGreaterEqual(pose.y, 20)

    def test_t_turn_changes_orientation_during_rotation_phase(self):
        drop = VisualDrop(
            piece="T",
            target_rotation=3,
            target_x=6,
            landing_y=34,
            display_hold=None,
            display_preview=(),
            started_at=100.0,
            duration=1.0,
        )

        before_turn = drop.pose(100.22)
        after_turn = drop.pose(100.43)
        self.assertEqual(before_turn.rotation % 4, 0)
        self.assertEqual(after_turn.rotation % 4, 3)

    def test_animation_still_ends_at_exact_target(self):
        drop = VisualDrop(
            piece="T",
            target_rotation=2,
            target_x=1,
            landing_y=36,
            display_hold=None,
            display_preview=(),
            started_at=10.0,
            duration=2.0,
        )
        pose = drop.pose(12.0)
        self.assertEqual(pose.rotation % 4, 2)
        self.assertEqual(pose.x, 1)
        self.assertEqual(pose.y, 36)


if __name__ == "__main__":
    unittest.main()
