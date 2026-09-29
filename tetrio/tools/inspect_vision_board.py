from __future__ import annotations

import argparse
import json
from pathlib import Path

from tetrio.vision.board import (
    draw_board_overlay,
    read_visual_board,
)
from tetrio.vision.image_io import (
    imread_unicode,
    imwrite_unicode,
)
from tetrio.vision.layout import (
    OcrObservation,
    RapidUsernameReader,
    detect_playfields,
    draw_layout_overlay,
    resolve_playfield_roles,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read the visible 10x20 TETR.IO grid after dynamic playfield "
            "localization. V0 reports visible colored minos plus diagnostic "
            "ghost/neutral candidates; it does not yet separate active from "
            "locked pieces."
        )
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--self-username", type=str, default=None)
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"artifacts\tetrio\vision\board_debug"),
    )
    return p.parse_args()


def input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"Input not found: {path}")
    return sorted(
        p
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def main() -> None:
    args = parse_args()
    files = input_files(args.input)
    if not files:
        raise SystemExit(f"No screenshots found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rapid = None if args.no_ocr else RapidUsernameReader()
    report_images = []

    print("=" * 112)
    print("TETR.IO VISUAL BOARD READER V0")
    print("=" * 112)
    print(f"Input         : {args.input}")
    print(f"Images        : {len(files)}")
    print(f"Self username : {args.self_username or '(not supplied)'}")
    print(f"OCR           : {'OFF' if args.no_ocr else 'ON'}")
    print()

    for path in files:
        image = imread_unicode(path)
        if image is None:
            print(f"[SKIP] cannot read: {path}")
            continue

        candidates = detect_playfields(image)

        if args.no_ocr:
            def no_ocr(_image, _candidate):
                return OcrObservation(None, 0.0)
            reader = no_ocr
        else:
            assert rapid is not None
            reader = rapid.read

        resolved = resolve_playfield_roles(
            image,
            candidates,
            self_username=args.self_username,
            ocr_reader=reader,
        )

        overlay = draw_layout_overlay(image, resolved)
        board_reports = []

        for index, item in enumerate(resolved):
            board = read_visual_board(
                image,
                item.candidate,
            )
            overlay = draw_board_overlay(
                overlay,
                item.candidate,
                board,
            )

            board_reports.append(
                {
                    "index": index,
                    "role": item.role,
                    "username": item.username,
                    "bbox": list(item.candidate.bbox),
                    **board.to_dict(),
                }
            )

            print(
                f"{path.name} board#{index} {item.role}: "
                f"mino={board.mino_count} "
                f"ghost?={board.ghost_candidate_count} "
                f"neutral?={board.neutral_candidate_count} "
                f"unknown={board.unknown_count} "
                f"transient?={sum(1 for row in board.rows for cell in row if cell == 'TRANSIENT_OVERLAY_CANDIDATE')}"
            )
            for line in board.symbols():
                print(f"  {line}")

        out_path = args.output_dir / f"{path.stem}_board.png"
        imwrite_unicode(out_path, overlay)

        report_images.append(
            {
                "input": str(path),
                "overlay": str(out_path),
                "boards": board_reports,
            }
        )

    report_path = args.output_dir / "board_report.json"
    report_path.write_text(
        json.dumps(
            {
                "format": "tetrio_visual_board_reader_v0",
                "self_username": args.self_username,
                "images": report_images,
                "symbol_legend": {
                    ".": "EMPTY",
                    "#": "visible colored MINO",
                    "g": "GHOST_CANDIDATE",
                    "n": "NEUTRAL_CANDIDATE",
                    "?": "UNKNOWN",
                    "t": "TRANSIENT_OVERLAY_CANDIDATE",
                },
                "important": (
                    "This is static visual occupancy, not yet the final locked "
                    "board state. Falling pieces can appear as #."
                ),
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
