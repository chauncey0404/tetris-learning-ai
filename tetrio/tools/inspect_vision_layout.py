from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tetrio.vision.layout import (
    RapidUsernameReader,
    detect_playfields,
    draw_layout_overlay,
    resolve_playfield_roles,
    username_strip_bbox,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}



def imread_unicode(path: Path):
    """Windows/OpenCV-safe image loader for Unicode file names."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image) -> bool:
    """Windows/OpenCV-safe image writer for Unicode file names."""
    suffix = path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        suffix = ".png"
        path = path.with_suffix(suffix)

    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded.tofile(str(path))
    except OSError:
        return False
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Detect full-size TETR.IO playfields and resolve SELF/OPPONENT "
            "using the username strip below each board."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Screenshot file or directory containing screenshots.",
    )
    p.add_argument(
        "--self-username",
        type=str,
        default=None,
        help=(
            "Your TETR.IO username. Required to resolve SELF in multi-board "
            "views. Single-board views are safely SELF without OCR."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"artifacts\tetrio\vision\layout_debug"),
    )
    p.add_argument(
        "--no-ocr",
        action="store_true",
        help="Detect boards only; multi-board roles remain UNKNOWN.",
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
    reports = []

    print("=" * 112)
    print("TETR.IO DYNAMIC LAYOUT DETECTOR")
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
                from tetrio.vision.layout import OcrObservation
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
        out_image = args.output_dir / f"{path.stem}_layout.png"
        if not imwrite_unicode(out_image, overlay):
            print(f"[WARN] cannot write overlay: {out_image}")

        # Also save exact username crops for diagnosing OCR failures.
        crop_paths = []
        for idx, item in enumerate(resolved):
            x, y, w, h = username_strip_bbox(item.candidate, image.shape)
            crop = image[y:y+h, x:x+w]
            crop_path = args.output_dir / f"{path.stem}_name_{idx}.png"
            if crop.size and imwrite_unicode(crop_path, crop):
                crop_paths.append(str(crop_path))

        result = {
            "input": str(path),
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "playfield_count": len(resolved),
            "playfields": [x.to_dict() for x in resolved],
            "overlay": str(out_image),
            "username_crops": crop_paths,
        }
        reports.append(result)

        roles = ", ".join(
            f"{x.role}:{x.username or '?'}"
            for x in resolved
        )
        print(
            f"{path.name}: boards={len(resolved)} "
            f"[{roles}]"
        )

    summary = {
        "format": "tetrio_dynamic_layout_detector",
        "self_username": args.self_username,
        "ocr_enabled": not args.no_ocr,
        "images": reports,
    }
    summary_path = args.output_dir / "layout_report.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print()
    print(f"Report: {summary_path}")


if __name__ == "__main__":
    main()
