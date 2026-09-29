from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from tetrio.vision.layout import PlayfieldCandidate


_BASE_TEMPLATES: dict[str, np.ndarray] = {
    "I": np.array([[1, 1, 1, 1]], dtype=np.uint8),
    "O": np.array([[1, 1], [1, 1]], dtype=np.uint8),
    "T": np.array([[0, 1, 0], [1, 1, 1]], dtype=np.uint8),
    "S": np.array([[0, 1, 1], [1, 1, 0]], dtype=np.uint8),
    "Z": np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
    "J": np.array([[1, 0, 0], [1, 1, 1]], dtype=np.uint8),
    "L": np.array([[0, 0, 1], [1, 1, 1]], dtype=np.uint8),
}


def _trim(matrix: np.ndarray) -> np.ndarray:
    ys, xs = np.where(matrix > 0)
    if len(xs) == 0:
        return matrix[:0, :0]
    return matrix[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1,
    ]


def _rotations(matrix: np.ndarray) -> tuple[np.ndarray, ...]:
    out: list[np.ndarray] = []
    current = matrix.copy()
    for _ in range(4):
        current = _trim(current)
        if not any(np.array_equal(current, old) for old in out):
            out.append(current.copy())
        current = np.rot90(current)
    return tuple(out)


_TEMPLATES = {
    piece: _rotations(matrix)
    for piece, matrix in _BASE_TEMPLATES.items()
}


@dataclass(frozen=True)
class PieceObservation:
    piece: str | None
    confidence: float
    shape_score: float
    matrix: tuple[tuple[int, ...], ...] | None
    bbox: tuple[int, int, int, int] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreviewObservation:
    hold: PieceObservation
    next_queue: tuple[PieceObservation, ...]
    next_complete: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "hold": self.hold.to_dict(),
            "next_queue": [x.to_dict() for x in self.next_queue],
            "next_complete": self.next_complete,
        }


def hold_roi(
    candidate: PlayfieldCandidate,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    bx, by, bw, bh = candidate.bbox

    x1 = max(0, int(round(bx - 0.62 * bw)))
    x2 = min(width, int(round(bx - 0.03 * bw)))
    y1 = max(0, int(round(by + 0.05 * bh)))
    y2 = min(height, int(round(by + 0.20 * bh)))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def next_roi(
    candidate: PlayfieldCandidate,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    bx, by, bw, bh = candidate.bbox

    x1 = max(0, int(round(bx + bw + 0.05 * bw)))
    x2 = min(width, int(round(bx + bw + 0.55 * bw)))
    y1 = max(0, int(round(by + 0.04 * bh)))
    y2 = min(height, int(round(by + 0.82 * bh)))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def classify_piece_matrix(
    matrix: np.ndarray,
) -> tuple[str | None, float]:
    matrix = _trim(matrix)
    if matrix.size == 0 or int(matrix.sum()) != 4:
        return None, 0.0

    best_piece: str | None = None
    best_score = -1.0

    for piece, rotations in _TEMPLATES.items():
        for template in rotations:
            if template.shape != matrix.shape:
                continue
            score = float(np.mean(template == matrix))
            if score > best_score:
                best_piece = piece
                best_score = score

    if best_score < 0.0:
        return None, 0.0
    return best_piece, best_score


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for idx in range(1, count):
        x, y, w, h, area = map(int, stats[idx])
        out.append((area, x, y, w, h))
    return out


def _matrix_from_component(
    mask: np.ndarray,
    component: tuple[int, int, int, int, int],
    cell_size: float,
) -> tuple[np.ndarray, float]:
    _, x, y, w, h = component
    rows = max(1, min(4, int(round(h / cell_size))))
    cols = max(1, min(4, int(round(w / cell_size))))

    crop = mask[y:y + h, x:x + w]
    matrix = np.zeros((rows, cols), dtype=np.uint8)
    fills: list[float] = []

    for row in range(rows):
        y1 = round(row * h / rows)
        y2 = round((row + 1) * h / rows)
        for col in range(cols):
            x1 = round(col * w / cols)
            x2 = round((col + 1) * w / cols)
            patch = crop[y1:y2, x1:x2]
            fill = float(np.mean(patch > 0)) if patch.size else 0.0
            occupied = fill >= 0.30
            matrix[row, col] = 1 if occupied else 0
            if occupied:
                fills.append(fill)

    matrix = _trim(matrix)
    return matrix, float(np.mean(fills)) if fills else 0.0


def read_next_queue(
    image_bgr: np.ndarray,
    candidate: PlayfieldCandidate,
) -> tuple[PieceObservation, ...]:
    x, y, w, h = next_roi(candidate, image_bgr.shape)
    crop = image_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        return ()

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Color only isolates the preview objects. Piece identity is shape-based.
    mask = (
        (saturation >= 70)
        & (value >= 70)
    ).astype(np.uint8) * 255

    d = float(candidate.cell_size)
    selected: list[tuple[int, int, int, int, int]] = []

    for component in _components(mask):
        area, _, _, cw, ch = component

        if not (2.2 * d * d <= area <= 5.4 * d * d):
            continue
        if not (0.75 * d <= ch <= 2.55 * d):
            continue
        if not (1.70 * d <= cw <= 4.55 * d):
            continue

        matrix, _ = _matrix_from_component(mask, component, d)
        piece, score = classify_piece_matrix(matrix)
        if piece is None or score < 0.99:
            continue

        selected.append(component)

    selected.sort(key=lambda item: item[2])

    observations: list[PieceObservation] = []
    for component in selected[:5]:
        _, cx, cy, cw, ch = component
        matrix, mean_fill = _matrix_from_component(mask, component, d)
        piece, shape_score = classify_piece_matrix(matrix)

        confidence = min(
            1.0,
            0.75 * shape_score
            + 0.25 * min(1.0, mean_fill),
        )

        observations.append(
            PieceObservation(
                piece=piece,
                confidence=confidence,
                shape_score=shape_score,
                matrix=tuple(
                    tuple(int(v) for v in row)
                    for row in matrix
                ),
                bbox=(x + cx, y + cy, cw, ch),
            )
        )

    return tuple(observations)


def read_hold_piece(
    image_bgr: np.ndarray,
    candidate: PlayfieldCandidate,
) -> PieceObservation:
    x, y, w, h = hold_roi(candidate, image_bgr.shape)
    crop = image_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        return PieceObservation(None, 0.0, 0.0, None, None)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)

    # HOLD is gray in the supplied layouts. Use local contrast, not hue.
    local_background = float(np.percentile(value, 50))
    threshold = max(30.0, local_background + 8.0)
    mask = (value >= threshold).astype(np.uint8) * 255

    d = float(candidate.cell_size)
    expected_center = np.array([0.53 * w, 0.48 * h], dtype=np.float64)

    matches = []

    for component in _components(mask):
        area, cx, cy, cw, ch = component

        if not (1.45 * d * d <= area <= 5.6 * d * d):
            continue
        if not (0.72 * d <= ch <= 2.85 * d):
            continue
        if not (1.45 * d <= cw <= 4.55 * d):
            continue

        matrix, mean_fill = _matrix_from_component(mask, component, d)
        piece, shape_score = classify_piece_matrix(matrix)
        if piece is None or shape_score < 0.99:
            continue

        center = np.array(
            [cx + cw / 2.0, cy + ch / 2.0],
            dtype=np.float64,
        )
        distance = float(
            np.linalg.norm(center - expected_center)
            / max(d, 1.0)
        )

        rank = (
            3.0 * shape_score
            + 0.4 * min(1.0, mean_fill)
            - 0.18 * distance
        )
        matches.append(
            (
                rank,
                component,
                matrix,
                mean_fill,
                piece,
                shape_score,
            )
        )

    if not matches:
        return PieceObservation(
            piece=None,
            confidence=0.96,
            shape_score=1.0,
            matrix=None,
            bbox=None,
        )

    matches.sort(key=lambda item: item[0], reverse=True)
    _, component, matrix, mean_fill, piece, shape_score = matches[0]
    _, cx, cy, cw, ch = component

    confidence = min(
        1.0,
        0.78 * shape_score
        + 0.22 * min(1.0, mean_fill),
    )

    return PieceObservation(
        piece=piece,
        confidence=confidence,
        shape_score=shape_score,
        matrix=tuple(
            tuple(int(v) for v in row)
            for row in matrix
        ),
        bbox=(x + cx, y + cy, cw, ch),
    )


def read_piece_previews(
    image_bgr: np.ndarray,
    candidate: PlayfieldCandidate,
) -> PreviewObservation:
    hold = read_hold_piece(image_bgr, candidate)
    next_queue = read_next_queue(image_bgr, candidate)
    return PreviewObservation(
        hold=hold,
        next_queue=next_queue,
        next_complete=len(next_queue) == 5,
    )


def draw_piece_preview_overlay(
    image_bgr: np.ndarray,
    observation: PreviewObservation,
) -> np.ndarray:
    out = image_bgr.copy()

    if observation.hold.bbox is not None:
        x, y, w, h = observation.hold.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(
            out,
            f"HOLD={observation.hold.piece}",
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for index, piece in enumerate(observation.next_queue):
        if piece.bbox is None:
            continue
        x, y, w, h = piece.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(
            out,
            f"N{index + 1}={piece.piece}",
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return out
