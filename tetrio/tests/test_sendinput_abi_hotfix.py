from __future__ import annotations
import ctypes
import os
import unittest
from tetrio.control.input_controller import WindowsInputController, windows_input_abi
from tetrio.control.keymap import Action, DEFAULT_KEYMAP

class SendInputAbiHotfixTests(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(DEFAULT_KEYMAP[Action.MOVE_LEFT].name, "A")
        self.assertEqual(DEFAULT_KEYMAP[Action.HOLD].name, "LSHIFT")

    def test_dry_run(self):
        c = WindowsInputController(dry_run=True)
        c.tap(Action.MOVE_LEFT, hold_seconds=0.0)

    @unittest.skipUnless(os.name == "nt", "Windows ABI assertion")
    def test_windows_input_size(self):
        abi = windows_input_abi()
        expected = 40 if abi["is_64_bit"] else 28
        self.assertEqual(abi["input_size"], expected)

if __name__ == "__main__":
    unittest.main()
