from __future__ import annotations

import unittest

import numpy as np

from tetrio.vision.board import (
    EMPTY,
    MINO,
    TRANSIENT,
    CellEvidence,
    VisualBoard,
)
from tetrio.vision.piece_reader import PieceObservation, PreviewObservation
from tetrio.vision.state_tracker import TemporalStateTracker, TrackerConfig


def _visual_board(
    occupied: set[tuple[int, int]],
    *,
    transient: set[tuple[int, int]] | None = None,
) -> VisualBoard:
    transient = transient or set()
    rows = []
    conf = []
    cells = []
    for r in range(20):
        row = []
        crow = []
        for c in range(10):
            if (r, c) in transient:
                label = TRANSIENT
            elif (r, c) in occupied:
                label = MINO
            else:
                label = EMPTY
            row.append(label)
            crow.append(0.98)
            cells.append(
                CellEvidence(
                    row=r,
                    col=c,
                    label=label,
                    confidence=0.98,
                    value_p90=0.0,
                    chroma_p90=0.0,
                    edge_density=0.0,
                    colored_fill_ratio=0.0,
                )
            )
        rows.append(tuple(row))
        conf.append(tuple(crow))
    return VisualBoard(
        rows=tuple(rows),
        confidences=tuple(conf),
        cells=tuple(cells),
        mino_count=len(occupied),
        ghost_candidate_count=0,
        neutral_candidate_count=0,
        unknown_count=0,
    )


def _piece(piece: str | None, confidence: float = 0.98) -> PieceObservation:
    return PieceObservation(
        piece=piece,
        confidence=confidence,
        shape_score=1.0,
        matrix=None,
        bbox=None,
    )


def _preview(
    hold: str | None = None,
    queue: tuple[str, ...] = ("I", "O", "S", "Z", "L"),
) -> PreviewObservation:
    return PreviewObservation(
        hold=_piece(hold),
        next_queue=tuple(_piece(p) for p in queue),
        next_complete=True,
    )


def _shape_t(y: int, x: int) -> set[tuple[int, int]]:
    return {
        (y, x + 1),
        (y + 1, x),
        (y + 1, x + 1),
        (y + 1, x + 2),
    }


def _shape_i(y: int, x: int) -> set[tuple[int, int]]:
    return {(y, x + i) for i in range(4)}


class TemporalTrackerTests(unittest.TestCase):
    def new_tracker(self) -> TemporalStateTracker:
        return TemporalStateTracker(
            TrackerConfig(
                history_frames=5,
                min_history_frames=3,
                panel_stability_frames=2,
                active_stability_frames=2,
                locked_stability_frames=2,
                transition_cooldown_frames=0,
            )
        )

    def test_cold_start_recovers_falling_active_and_locked_board(self):
        tracker = self.new_tracker()
        locked = {(19, c) for c in range(5)}
        outputs = []
        for y in (2, 3, 4, 5, 6):
            outputs.append(
                tracker.update(
                    _visual_board(locked | _shape_t(y, 4)),
                    _preview(),
                    layout_bbox=(800, 180, 300, 600),
                )
            )

        last = outputs[-1]
        self.assertTrue(last.stable_pre_action)
        self.assertEqual(last.phase, "READY")
        self.assertIsNotNone(last.active)
        self.assertEqual(last.active.piece, "T")
        self.assertEqual(
            int(np.asarray(last.locked_board, dtype=np.uint8).sum()),
            len(locked),
        )

    def test_locked_board_commit_without_new_generation_is_not_a_lock_event(self):
        tracker = self.new_tracker()
        locked = {(19, c) for c in range(5)}
        for y in (2, 3, 4, 5, 6):
            tracker.update(_visual_board(locked | _shape_t(y, 4)), _preview())

        changed_locked = locked | {(18, 9)}
        events = []
        for y in (7, 8, 9, 10):
            events.append(
                tracker.update(
                    _visual_board(changed_locked | _shape_t(y, 4)),
                    _preview(),
                )
            )
        self.assertEqual(sum(int(x.lock_event) for x in events), 0)
        self.assertEqual(sum(int(x.spawn_event) for x in events), 0)

    def test_one_missing_active_cell_is_temporally_recovered(self):
        tracker = self.new_tracker()
        locked = {(19, c) for c in range(5)}
        for y in (2, 3, 4, 5, 6):
            tracker.update(_visual_board(locked | _shape_t(y, 4)), _preview())

        full = _shape_t(7, 4)
        missing = set(full)
        missing.remove((8, 6))
        out = tracker.update(_visual_board(locked | missing), _preview())
        self.assertIsNotNone(out.active)
        self.assertEqual(out.active.piece, "T")
        self.assertTrue(out.active.recovered)
        self.assertEqual(len(out.active.recovered_cells), 1)

    def test_new_locked_board_emits_lock_and_spawn(self):
        tracker = self.new_tracker()
        locked = {(19, c) for c in range(5)}
        for y in (2, 3, 4, 5, 6):
            tracker.update(_visual_board(locked | _shape_t(y, 4)), _preview())

        new_locked = locked | _shape_t(17, 5)
        shifted = ("O", "S", "Z", "L", "J")
        events = []
        for y in (2, 3, 4, 5, 6, 7):
            events.append(
                tracker.update(
                    _visual_board(new_locked | _shape_i(y, 3)),
                    _preview(queue=shifted),
                )
            )

        self.assertEqual(sum(int(x.lock_event) for x in events), 1)
        self.assertEqual(sum(int(x.spawn_event) for x in events), 1)
        last = events[-1]
        self.assertEqual(last.active.piece, "I")
        self.assertEqual(
            int(np.asarray(last.locked_board, dtype=np.uint8).sum()),
            len(new_locked),
        )

    def test_transient_overlay_fails_closed_when_outside_active_corridor(self):
        tracker = self.new_tracker()
        locked = {(19, c) for c in range(5)}
        for y in (2, 3, 4, 5, 6):
            tracker.update(_visual_board(locked | _shape_t(y, 4)), _preview())

        out = tracker.update(
            _visual_board(
                locked | _shape_t(7, 4),
                transient={(10, 0)},
            ),
            _preview(),
        )
        self.assertFalse(out.stable_pre_action)
        self.assertEqual(out.reason, "transient_overlay")

    def test_falling_piece_partial_cell_transient_is_benign(self):
        tracker = self.new_tracker()
        locked = {(19, c) for c in range(5)}
        for y in (2, 3, 4, 5, 6):
            tracker.update(_visual_board(locked | _shape_t(y, 4)), _preview())

        active = _shape_t(7, 4)
        # One active cell is only partially covered while the falling piece is
        # between grid rows. Static board reading calls it TRANSIENT; temporal
        # tracking may recover the one missing cell and still be model-ready.
        out = tracker.update(
            _visual_board(active | locked, transient={(8, 6)}),
            _preview(),
        )
        self.assertTrue(out.stable_pre_action)
        self.assertEqual(out.reason, "stable_pre_action")
        self.assertGreater(out.transient_count, 0)

    def test_hold_change_requires_confirmation(self):
        tracker = self.new_tracker()
        locked = {(19, c) for c in range(5)}
        for y in (2, 3, 4, 5, 6):
            out = tracker.update(_visual_board(locked | _shape_t(y, 4)), _preview())
        self.assertIsNone(out.hold_piece)

        one = tracker.update(_visual_board(locked | _shape_t(7, 4)), _preview(hold="Z"))
        self.assertIsNone(one.hold_piece)
        self.assertFalse(one.stable_pre_action)

        two = tracker.update(_visual_board(locked | _shape_t(8, 4)), _preview(hold="Z"))
        self.assertEqual(two.hold_piece, "Z")
        self.assertTrue(two.hold_changed)


    def test_populated_hold_does_not_clear_from_visual_none(self):
        tracker = self.new_tracker()
        locked = {(19, c) for c in range(5)}
        for y in (2, 3, 4, 5, 6):
            tracker.update(
                _visual_board(locked | _shape_t(y, 4)),
                _preview(hold="Z"),
            )

        one = tracker.update(
            _visual_board(locked | _shape_t(7, 4)),
            _preview(hold=None),
        )
        two = tracker.update(
            _visual_board(locked | _shape_t(8, 4)),
            _preview(hold=None),
        )
        self.assertEqual(one.hold_piece, "Z")
        self.assertEqual(two.hold_piece, "Z")
        self.assertFalse(one.hold_changed)
        self.assertFalse(two.hold_changed)
        self.assertFalse(two.stable_pre_action)
        self.assertEqual(two.reason, "hold_unstable")

    def test_preview_multi_piece_shift_is_recognized_after_confirmation(self):
        tracker = self.new_tracker()
        locked = {(19, c) for c in range(5)}
        old_queue = ("J", "O", "I", "T", "Z")
        # Manual live play advanced two pieces between two stable panel reads.
        new_queue = ("I", "T", "Z", "T", "S")
        for y in (2, 3, 4, 5, 6):
            tracker.update(
                _visual_board(locked | _shape_t(y, 4)),
                _preview(queue=old_queue),
            )

        one = tracker.update(
            _visual_board(locked | _shape_t(7, 4)),
            _preview(queue=new_queue),
        )
        self.assertFalse(one.preview_shift_event)
        two = tracker.update(
            _visual_board(locked | _shape_t(8, 4)),
            _preview(queue=new_queue),
        )
        self.assertTrue(two.preview_shift_event)

    def test_preview_shift_requires_confirmation(self):
        tracker = self.new_tracker()
        locked = {(19, c) for c in range(5)}
        old_queue = ("I", "O", "S", "Z", "L")
        new_queue = ("O", "S", "Z", "L", "J")
        for y in (2, 3, 4, 5, 6):
            tracker.update(
                _visual_board(locked | _shape_t(y, 4)),
                _preview(queue=old_queue),
            )

        one = tracker.update(
            _visual_board(locked | _shape_t(7, 4)),
            _preview(queue=new_queue),
        )
        self.assertEqual(one.preview_queue, old_queue)
        self.assertFalse(one.preview_shift_event)
        self.assertFalse(one.stable_pre_action)

        two = tracker.update(
            _visual_board(locked | _shape_t(8, 4)),
            _preview(queue=new_queue),
        )
        self.assertEqual(two.preview_queue, new_queue)
        self.assertTrue(two.preview_shift_event)

    def test_layout_shift_resets_temporal_state(self):
        tracker = self.new_tracker()
        locked = {(19, c) for c in range(5)}
        for y in (2, 3, 4, 5, 6):
            tracker.update(
                _visual_board(locked | _shape_t(y, 4)),
                _preview(),
                layout_bbox=(800, 180, 300, 600),
            )
        out = tracker.update(
            _visual_board(locked | _shape_t(7, 4)),
            _preview(),
            layout_bbox=(900, 180, 300, 600),
        )
        self.assertTrue(out.layout_reset)
        self.assertFalse(out.stable_pre_action)
        self.assertEqual(out.reason, "layout_reset")


if __name__ == "__main__":
    unittest.main()
