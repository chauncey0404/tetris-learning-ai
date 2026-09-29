from __future__ import annotations
import unittest
from tetrio.control.input_controller import WindowsInputController
from tetrio.control.keymap import Action, DEFAULT_KEYMAP

class KeyboardMappingTests(unittest.TestCase):
    def test_expected_mapping(self):
        expected = {
            Action.MOVE_LEFT: "A",
            Action.MOVE_RIGHT: "D",
            Action.SOFT_DROP: "W",
            Action.HARD_DROP: "S",
            Action.ROTATE_CCW: "LEFT",
            Action.ROTATE_CW: "RIGHT",
            Action.ROTATE_180: "UP",
            Action.HOLD: "LSHIFT",
        }
        self.assertEqual(
            {a: DEFAULT_KEYMAP[a].name for a in expected},
            expected,
        )

    def test_arrow_keys_are_extended(self):
        self.assertTrue(DEFAULT_KEYMAP[Action.ROTATE_CCW].extended)
        self.assertTrue(DEFAULT_KEYMAP[Action.ROTATE_CW].extended)
        self.assertTrue(DEFAULT_KEYMAP[Action.ROTATE_180].extended)

    def test_dry_run_accepts_all_actions(self):
        c = WindowsInputController(dry_run=True)
        for action in Action:
            c.tap(action, hold_seconds=0.0)

if __name__ == "__main__":
    unittest.main()
