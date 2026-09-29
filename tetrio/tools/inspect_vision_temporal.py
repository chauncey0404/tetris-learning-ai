from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from tetrio.vision.board import draw_board_overlay, read_visual_board
from tetrio.vision.image_io import imread_unicode, imwrite_unicode
from tetrio.vision.layout import (
    OcrObservation,
    RapidUsernameReader,
    detect_playfields,
    draw_layout_overlay,
    resolve_playfield_roles,
)
from tetrio.vision.piece_reader import (
    draw_piece_preview_overlay,
    read_piece_previews,
)
from tetrio.vision.state_tracker import TemporalStateTracker, TrackerConfig


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the temporal TETR.IO visual tracker over a filename-sorted "
            "screenshot sequence. The sequence should come from one continuous "
            "simulator/game session."
        )
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--self-username", type=str, default=None)
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"artifacts\tetrio\vision\temporal_debug"),
    )
    return p.parse_args()


def image_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"Input not found: {path}")
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _draw_temporal_text(image, obs):
    out = image.copy()
    active = "-" if obs.active is None else (
        f"{obs.active.piece} c={obs.active.confidence:.2f}"
        + (" REC" if obs.active.recovered else "")
    )
    preview = "-" if obs.preview_queue is None else "".join(obs.preview_queue)
    lines = [
        f"TEMP {obs.phase} ready={int(obs.stable_pre_action)} reason={obs.reason}",
        f"active={active} hold={obs.hold_piece or '-'} next={preview}",
        f"spawn={int(obs.spawn_event)} lock={int(obs.lock_event)} "
        f"qshift={int(obs.preview_shift_event)}",
    ]
    y = 28
    for line in lines:
        cv2.putText(
            out,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        y += 25
    return out


def main() -> None:
    args = parse_args()
    files = image_files(args.input)
    if not files:
        raise SystemExit(f"No screenshots found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rapid = None if args.no_ocr else RapidUsernameReader()
    tracker = TemporalStateTracker(TrackerConfig())
    report = []

    print("=" * 112)
    print("TETR.IO VISUAL TEMPORAL TRACKER")
    print("=" * 112)
    print(f"Input frames : {len(files)}")
    print(f"OCR          : {'OFF' if args.no_ocr else 'ON'}")
    print()

    for path in files:
        image = imread_unicode(path)
        if image is None:
            print(f"[SKIP] cannot read: {path}")
            continue

        candidates = detect_playfields(image)
        if args.no_ocr:
            def reader(_image, _candidate):
                return OcrObservation(None, 0.0)
        else:
            assert rapid is not None
            reader = rapid.read

        resolved = resolve_playfield_roles(
            image,
            candidates,
            self_username=args.self_username,
            ocr_reader=reader,
        )
        self_boards = [item for item in resolved if item.role == "SELF"]
        if len(self_boards) != 1:
            tracker.reset()
            print(f"{path.name}: WAIT self-board count={len(self_boards)}")
            report.append({
                "input": str(path),
                "status": "SELF_BOARD_UNRESOLVED",
                "self_board_count": len(self_boards),
            })
            continue

        item = self_boards[0]
        board = read_visual_board(image, item.candidate)
        previews = read_piece_previews(image, item.candidate)
        tracked = tracker.update(
            board,
            previews,
            layout_bbox=item.candidate.bbox,
        )

        overlay = draw_layout_overlay(image, resolved)
        overlay = draw_board_overlay(overlay, item.candidate, board)
        overlay = draw_piece_preview_overlay(overlay, previews)
        overlay = _draw_temporal_text(overlay, tracked)
        out = args.output_dir / f"{path.stem}_temporal.png"
        imwrite_unicode(out, overlay)

        active = "-" if tracked.active is None else tracked.active.piece
        print(
            f"{path.name}: {tracked.phase:10s} ready={int(tracked.stable_pre_action)} "
            f"active={active} hold={tracked.hold_piece or '-'} "
            f"next={tracked.preview_queue} reason={tracked.reason}"
        )
        report.append({
            "input": str(path),
            "overlay": str(out),
            "layout_bbox": list(item.candidate.bbox),
            "raw_board": board.to_dict(),
            "raw_previews": previews.to_dict(),
            "tracked": tracked.to_dict(),
        })

    report_path = args.output_dir / "temporal_report.json"
    report_path.write_text(
        json.dumps(
            {
                "format": "tetrio_visual_temporal_tracker",
                "frames": report,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print()
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
