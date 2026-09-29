from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from tetrio.vision.layout import PlayfieldCandidate


EMPTY = "EMPTY"
MINO = "MINO"
GHOST = "GHOST_CANDIDATE"
NEUTRAL = "NEUTRAL_CANDIDATE"
UNKNOWN = "UNKNOWN"
TRANSIENT = "TRANSIENT_OVERLAY_CANDIDATE"


@dataclass(frozen=True)
class CellEvidence:
    row: int
    col: int
    label: str
    confidence: float
    value_p90: float
    chroma_p90: float
    edge_density: float
    colored_fill_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualBoard:
    rows: tuple[tuple[str, ...], ...]
    confidences: tuple[tuple[float, ...], ...]
    cells: tuple[CellEvidence, ...]
    mino_count: int
    ghost_candidate_count: int
    neutral_candidate_count: int
    unknown_count: int

    def symbols(self) -> list[str]:
        mapping = {
            EMPTY: ".",
            MINO: "#",
            GHOST: "g",
            NEUTRAL: "n",
            UNKNOWN: "?",
            TRANSIENT: "t",
        }
        return [
            "".join(mapping[cell] for cell in row)
            for row in self.rows
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": [list(row) for row in self.rows],
            "symbols": self.symbols(),
            "confidences": [list(row) for row in self.confidences],
            "mino_count": self.mino_count,
            "ghost_candidate_count": self.ghost_candidate_count,
            "neutral_candidate_count": self.neutral_candidate_count,
            "unknown_count": self.unknown_count,
            "transient_candidate_count": sum(
                1
                for row in self.rows
                for cell in row
                if cell == TRANSIENT
            ),
            "cells": [cell.to_dict() for cell in self.cells],
            "warning": (
                "MINO means a visible colored mino in this static frame. "
                "It may be locked stack or the currently falling piece. "
                "GHOST/NEUTRAL are diagnostic candidates only."
            ),
        }


def _cell_patch(
    image_bgr: np.ndarray,
    candidate: PlayfieldCandidate,
    row: int,
    col: int,
    *,
    inner_margin: float = 0.18,
) -> np.ndarray:
    d = float(candidate.cell_size)
    x1 = int(round(candidate.x + (col + inner_margin) * d))
    x2 = int(round(candidate.x + (col + 1.0 - inner_margin) * d))
    y1 = int(round(candidate.y + (row + inner_margin) * d))
    y2 = int(round(candidate.y + (row + 1.0 - inner_margin) * d))

    h, w = image_bgr.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    return image_bgr[y1:y2, x1:x2]


def _features(
    patch: np.ndarray,
) -> tuple[float, float, float, float]:
    if patch.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)

    p = patch.astype(np.int16)
    chroma = p.max(axis=2) - p.min(axis=2)

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 90)

    value_p90 = float(np.percentile(value, 90))
    chroma_p90 = float(np.percentile(chroma, 90))
    edge_density = float(edges.mean() / 255.0)

    # Real TETR.IO minos fill almost the entire interior of a cell. Large
    # countdown/UI glyphs can have the same yellow/red hue, but only occupy a
    # thin fraction of the cell. This non-percentile coverage signal is the
    # important guard against treating "3/2/1" overlays as blocks.
    colored_fill_ratio = float(
        np.mean(
            (value >= 85.0)
            & (chroma >= 45.0)
        )
    )

    return (
        value_p90,
        chroma_p90,
        edge_density,
        colored_fill_ratio,
    )


def classify_cell(
    value_p90: float,
    chroma_p90: float,
    edge_density: float,
    colored_fill_ratio: float = 1.0,
) -> tuple[str, float]:
    """Conservative static-frame cell classifier.

    The supplied screenshots give a strong additional invariant:
      * genuine colored minos fill >= ~0.93 of the sampled cell interior;
      * the large pre-match countdown glyph that fooled V0 fills only
        ~0.10-0.43.

    Therefore color alone is never enough to promote a cell to MINO.
    """

    strongly_colored = (
        value_p90 >= 85.0
        and chroma_p90 >= 45.0
    )

    if strongly_colored and colored_fill_ratio >= 0.70:
        value_margin = min(
            1.0,
            max(0.0, (value_p90 - 85.0) / 80.0),
        )
        chroma_margin = min(
            1.0,
            max(0.0, (chroma_p90 - 45.0) / 80.0),
        )
        fill_margin = min(
            1.0,
            max(
                0.0,
                (colored_fill_ratio - 0.70) / 0.25,
            ),
        )
        confidence = min(
            1.0,
            0.68
            + 0.10 * value_margin
            + 0.10 * chroma_margin
            + 0.12 * fill_margin,
        )
        return MINO, confidence

    # Bright/saturated but partial cell coverage is characteristic of
    # countdown digits, attack text, flashes, and other transient UI drawn
    # over the board. Keep it explicit and never feed it as occupancy.
    if strongly_colored and colored_fill_ratio < 0.70:
        confidence = min(
            0.95,
            0.55
            + 0.40
            * min(1.0, (0.70 - colored_fill_ratio) / 0.60),
        )
        return TRANSIENT, confidence

    if (
        value_p90 >= 95.0
        and chroma_p90 < 45.0
        and edge_density >= 0.08
    ):
        confidence = min(
            0.85,
            0.45
            + 0.20
            * min(1.0, (value_p90 - 95.0) / 80.0)
            + 0.20
            * min(1.0, edge_density / 0.25),
        )
        return NEUTRAL, confidence

    if (
        38.0 <= value_p90 < 85.0
        and edge_density >= 0.16
        and colored_fill_ratio < 0.15
    ):
        confidence = min(
            0.80,
            0.40
            + 0.25
            * min(1.0, (value_p90 - 38.0) / 35.0)
            + 0.15
            * min(1.0, edge_density / 0.30),
        )
        return GHOST, confidence

    near_colored = (
        value_p90 >= 65.0
        and chroma_p90 >= 32.0
    )
    near_neutral = (
        value_p90 >= 75.0
        and edge_density >= 0.10
    )
    if near_colored or near_neutral:
        return UNKNOWN, 0.35

    brightness_safety = max(
        0.0,
        min(1.0, (75.0 - value_p90) / 50.0),
    )
    chroma_safety = max(
        0.0,
        min(1.0, (40.0 - chroma_p90) / 30.0),
    )
    confidence = min(
        0.98,
        0.72
        + 0.13 * brightness_safety
        + 0.13 * chroma_safety,
    )
    return EMPTY, confidence


def read_visual_board(
    image_bgr: np.ndarray,
    candidate: PlayfieldCandidate,
) -> VisualBoard:
    rows: list[list[str]] = []
    confidence_rows: list[list[float]] = []
    evidence: list[CellEvidence] = []

    for row in range(20):
        label_row: list[str] = []
        confidence_row: list[float] = []

        for col in range(10):
            patch = _cell_patch(
                image_bgr,
                candidate,
                row,
                col,
            )
            (
                value_p90,
                chroma_p90,
                edge_density,
                colored_fill_ratio,
            ) = _features(patch)
            label, confidence = classify_cell(
                value_p90,
                chroma_p90,
                edge_density,
                colored_fill_ratio,
            )

            label_row.append(label)
            confidence_row.append(confidence)
            evidence.append(
                CellEvidence(
                    row=row,
                    col=col,
                    label=label,
                    confidence=confidence,
                    value_p90=value_p90,
                    chroma_p90=chroma_p90,
                    edge_density=edge_density,
                    colored_fill_ratio=colored_fill_ratio,
                )
            )

        rows.append(label_row)
        confidence_rows.append(confidence_row)

    flat = [cell for row in rows for cell in row]
    return VisualBoard(
        rows=tuple(tuple(row) for row in rows),
        confidences=tuple(tuple(row) for row in confidence_rows),
        cells=tuple(evidence),
        mino_count=flat.count(MINO),
        ghost_candidate_count=flat.count(GHOST),
        neutral_candidate_count=flat.count(NEUTRAL),
        unknown_count=flat.count(UNKNOWN),
    )


def draw_board_overlay(
    image_bgr: np.ndarray,
    candidate: PlayfieldCandidate,
    board: VisualBoard,
) -> np.ndarray:
    out = image_bgr.copy()
    d = float(candidate.cell_size)

    colors = {
        MINO: (0, 255, 0),
        GHOST: (255, 255, 0),
        NEUTRAL: (255, 0, 255),
        UNKNOWN: (0, 0, 255),
        TRANSIENT: (0, 128, 255),
    }

    for cell in board.cells:
        if cell.label == EMPTY:
            continue

        x1 = int(round(candidate.x + cell.col * d))
        x2 = int(round(candidate.x + (cell.col + 1) * d))
        y1 = int(round(candidate.y + cell.row * d))
        y2 = int(round(candidate.y + (cell.row + 1) * d))

        color = colors[cell.label]
        thickness = 2 if cell.label == MINO else 1
        cv2.rectangle(
            out,
            (x1 + 2, y1 + 2),
            (x2 - 2, y2 - 2),
            color,
            thickness,
        )

    return out
