from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tetrio.parity.replay_inventory import inspect_ttrm_object
from tetrio.parity.trace import GarbageParityEvent, load_parity_trace
from tetrio.parity.validator import validate_garbage_trace


class TetrioReplayParityTests(unittest.TestCase):
    def test_inventory_understands_round_replay_container_without_deeper_assumptions(self):
        raw = {
            "data": [
                {
                    "replays": [
                        {"frames": 123, "events": [{"type": "keydown", "frame": 10}]},
                        {"frames": 120, "events": [{"type": "garbage", "frame": 11}]},
                    ]
                }
            ],
            "endcontext": [],
        }
        inv = inspect_ttrm_object(raw)
        self.assertEqual(inv.round_count, 1)
        self.assertEqual(len(inv.player_replays), 2)
        self.assertEqual(inv.player_replays[0].frames, 123)
        self.assertTrue(any(x["path"].endswith(".type") for x in inv.matching_paths["garbage"]))
        self.assertEqual(inv.tagged_value_counts["type"]["garbage"], 1)

    def test_trace_loader_accepts_wrapped_json(self):
        payload = {
            "events": [
                {"kind": "send", "player": "A", "target": "B", "frame": 10, "packets": [4]},
                {"kind": "assert_queue", "player": "B", "frame": 10, "expected": {"pending": 4}},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            events = load_parity_trace(path)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].packets, (4,))

    def test_send_travel_activation_cancel_and_tank_exact_parity(self):
        events = (
            GarbageParityEvent(
                kind="send",
                player="A",
                target="B",
                frame=100,
                packets=(4,),
                holes=(3,),
                expected={"pending": 4, "active": 0, "active_frames": [120]},
            ),
            GarbageParityEvent(
                kind="cancel",
                player="B",
                frame=110,
                lines=2,
                expected={"cancelled": 2, "pending": 2, "active": 0},
            ),
            GarbageParityEvent(
                kind="advance",
                player="B",
                frame=120,
                expected={"pending": 2, "active": 2},
            ),
            GarbageParityEvent(
                kind="tank",
                player="B",
                frame=120,
                expected={"inserted": 2, "pending": 0, "bottom_garbage_holes": [3, 3]},
            ),
        )
        report = validate_garbage_trace(events)
        self.assertTrue(report.passed, report.mismatches)

    def test_parity_report_is_fail_closed_on_wrong_oracle_value(self):
        events = (
            GarbageParityEvent(
                kind="send",
                player="A",
                target="B",
                frame=0,
                packets=(4,),
                expected={"pending": 5},
            ),
        )
        report = validate_garbage_trace(events)
        self.assertFalse(report.passed)
        self.assertEqual(report.mismatches[0].field, "pending")
        self.assertEqual(report.mismatches[0].actual, 4)

    def test_tank_remains_fail_closed_without_observed_hole(self):
        events = (
            GarbageParityEvent(kind="send", player="A", target="B", frame=0, packets=(2,), travel_frames=0),
            GarbageParityEvent(kind="tank", player="B", frame=0),
        )
        with self.assertRaises(ValueError):
            validate_garbage_trace(events)


if __name__ == "__main__":
    unittest.main()
