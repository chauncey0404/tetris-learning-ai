from __future__ import annotations
import argparse
import json
from pathlib import Path

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

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--self-username", type=str, default=None)
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"artifacts\tetrio\vision\piece_debug"),
    )
    return p.parse_args()

def image_files(path: Path):
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rapid = None if args.no_ocr else RapidUsernameReader()
    report = []

    print("=" * 100)
    print("TETR.IO VISUAL HOLD / NEXT READER")
    print("=" * 100)

    for path in image_files(args.input):
        image = imread_unicode(path)
        if image is None:
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

        overlay = draw_layout_overlay(image, resolved)
        rows = []

        for index, item in enumerate(resolved):
            obs = read_piece_previews(image, item.candidate)
            overlay = draw_piece_preview_overlay(overlay, obs)

            queue = [x.piece for x in obs.next_queue]
            print(
                f"{path.name} board#{index} {item.role}: "
                f"HOLD={obs.hold.piece or '-'} "
                f"NEXT={queue} complete={obs.next_complete}"
            )

            rows.append({
                "board_index": index,
                "role": item.role,
                "username": item.username,
                **obs.to_dict(),
            })

        out = args.output_dir / f"{path.stem}_pieces.png"
        imwrite_unicode(out, overlay)
        report.append({
            "input": str(path),
            "overlay": str(out),
            "boards": rows,
        })

    report_path = args.output_dir / "piece_report.json"
    report_path.write_text(
        json.dumps(
            {"format": "tetrio_visual_hold_next_reader", "images": report},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Report: {report_path}")

if __name__ == "__main__":
    main()
