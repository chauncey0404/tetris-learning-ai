from __future__ import annotations

import unittest

from tetrio.reachability import tetrio_spawn_state
from tetrio.tools.watch_expert_v0 import VisualDrop


class ExpertRolloutViewerV34Tests(unittest.TestCase):
    def test_visual_drop_finishes_at_exact_target(self):
        drop = VisualDrop(
            piece="T",
            target_rotation=3,
            target_x=6,
            landing_y=31,
            display_hold="I",
            display_preview=("O", "S", "Z", "J", "L"),
            started_at=100.0,
            duration=0.5,
        )
        pose = drop.pose(100.5)
        self.assertEqual(pose.piece, "T")
        self.assertEqual(pose.rotation % 4, 3)
        self.assertEqual(pose.x, 6)
        self.assertEqual(pose.y, 31)

    def test_visual_drop_begins_at_tetrio_entry(self):
        drop = VisualDrop(
            piece="O",
            target_rotation=0,
            target_x=0,
            landing_y=30,
            display_hold=None,
            display_preview=("I", "T", "S", "Z", "J"),
            started_at=50.0,
            duration=1.0,
        )
        pose = drop.pose(50.0)
        spawn = tetrio_spawn_state("O")
        self.assertEqual(pose.x, spawn.x)
        self.assertEqual(pose.y, spawn.y)
        self.assertEqual(pose.rotation % 4, spawn.rotation % 4)

    def test_animation_progress_clamps(self):
        drop = VisualDrop(
            piece="I",
            target_rotation=1,
            target_x=3,
            landing_y=32,
            display_hold=None,
            display_preview=(),
            started_at=10.0,
            duration=2.0,
        )
        self.assertEqual(drop.progress(0.0), 0.0)
        self.assertEqual(drop.progress(20.0), 1.0)


if __name__ == "__main__":
    unittest.main()
