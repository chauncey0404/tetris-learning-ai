from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: Path):
    """Windows/OpenCV-safe image reader for Unicode paths."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image) -> bool:
    """Windows/OpenCV-safe image writer for Unicode paths."""
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
