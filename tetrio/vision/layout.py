from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class PlayfieldCandidate:
    x: float
    y: float
    w: float
    h: float
    cell_size: float
    grid_score: float
    matched_vertical_lines: int
    confidence: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (
            int(round(self.x)),
            int(round(self.y)),
            int(round(self.w)),
            int(round(self.h)),
        )

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["bbox"] = list(self.bbox)
        return out


@dataclass(frozen=True)
class OcrObservation:
    text: str | None
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedPlayfield:
    candidate: PlayfieldCandidate
    role: str  # SELF | OPPONENT | UNKNOWN
    username: str | None
    username_ocr_confidence: float
    self_match_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.candidate.to_dict(),
            "role": self.role,
            "username": self.username,
            "username_ocr_confidence": self.username_ocr_confidence,
            "self_match_score": self.self_match_score,
        }


def _iou(a: PlayfieldCandidate, b: PlayfieldCandidate) -> float:
    ax1, ay1, aw, ah = a.x, a.y, a.w, a.h
    bx1, by1, bw, bh = b.x, b.y, b.w, b.h
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _vertical_line_clusters(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    h, _ = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 100)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=45,
        minLineLength=max(80, int(h * 0.08)),
        maxLineGap=12,
    )

    segments: list[tuple[int, int, int]] = []
    if lines is not None:
        for raw in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, raw)
            if abs(x2 - x1) <= 4 and abs(y2 - y1) >= max(80, h * 0.08):
                segments.append(
                    (
                        int(round((x1 + x2) / 2)),
                        min(y1, y2),
                        max(y1, y2),
                    )
                )

    segments.sort()
    grouped: list[tuple[list[int], list[tuple[int, int]]]] = []
    for x, y1, y2 in segments:
        if not grouped or x - grouped[-1][0][-1] > 4:
            grouped.append(([x], [(y1, y2)]))
        else:
            grouped[-1][0].append(x)
            grouped[-1][1].append((y1, y2))

    clusters: list[dict[str, Any]] = []
    for xs, ys in grouped:
        clusters.append(
            {
                "x": float(np.median(xs)),
                "segments": ys,
            }
        )
    return edges, clusters



def _cluster_vertical_extent(
    cluster: dict[str, Any] | None,
) -> tuple[float, float, float]:
    if cluster is None:
        return 0.0, math.nan, math.nan

    segments = cluster.get("segments") or []
    if not segments:
        return 0.0, math.nan, math.nan

    starts = [float(min(a, b)) for a, b in segments]
    ends = [float(max(a, b)) for a, b in segments]
    top = min(starts)
    bottom = max(ends)
    return max(0.0, bottom - top), top, bottom


def _nearest_vertical_cluster(
    clusters: list[dict[str, Any]],
    target_x: float,
    tolerance: float,
) -> dict[str, Any] | None:
    matches = [
        c
        for c in clusters
        if abs(float(c["x"]) - target_x) <= tolerance
    ]
    if not matches:
        return None
    return min(
        matches,
        key=lambda c: abs(float(c["x"]) - target_x),
    )


def _horizontal_grid_energy(
    edges: np.ndarray,
    x0: float,
    y0: float,
    cell_size: float,
) -> float:
    height, width = edges.shape[:2]
    x1 = int(round(x0))
    x2 = int(round(x0 + 10.0 * cell_size))
    if x1 < 0 or x2 > width or x2 <= x1:
        return 0.0

    scores: list[float] = []
    for row in range(21):
        yy = int(round(y0 + row * cell_size))
        if yy < 1 or yy >= height - 1:
            return 0.0
        strip = edges[
            yy - 1:yy + 2,
            x1:x2,
        ]
        scores.append(
            float(strip.mean() / 255.0)
            if strip.size
            else 0.0
        )
    return float(np.mean(scores)) if scores else 0.0


def _full_grid_energy(
    edges: np.ndarray,
    x0: float,
    y0: float,
    cell_size: float,
) -> float:
    height, width = edges.shape[:2]
    x1 = int(round(x0))
    x2 = int(round(x0 + 10.0 * cell_size))
    y1 = int(round(y0))
    y2 = int(round(y0 + 20.0 * cell_size))
    if (
        x1 < 0
        or y1 < 0
        or x2 > width
        or y2 > height
        or x2 <= x1
        or y2 <= y1
    ):
        return 0.0

    vertical_scores: list[float] = []
    for col in range(11):
        xx = int(round(x0 + col * cell_size))
        if xx < 1 or xx >= width - 1:
            return 0.0
        strip = edges[
            y1:y2,
            xx - 1:xx + 2,
        ]
        vertical_scores.append(
            float(strip.mean() / 255.0)
            if strip.size
            else 0.0
        )

    horizontal = _horizontal_grid_energy(
        edges,
        x0,
        y0,
        cell_size,
    )
    vertical = (
        float(np.mean(vertical_scores))
        if vertical_scores
        else 0.0
    )
    return 0.55 * vertical + 0.45 * horizontal


def _phase_lock_candidate(
    image_bgr: np.ndarray,
    edges: np.ndarray,
    clusters: list[dict[str, Any]],
    candidate: PlayfieldCandidate,
) -> PlayfieldCandidate:
    """Resolve whole-cell phase ambiguity using long outer playfield borders.

    A 10x20 grid is periodic, so a pure grid detector can occasionally return
    the correct size but shifted by one or two cells. TETR.IO's actual left and
    right playfield borders are long continuous vertical strokes, while
    internal grid lines and nearby HOLD/NEXT UI lines are much shorter.

    We only re-anchor when the current phase is clearly weak. This keeps
    already-good layouts unchanged.
    """
    d = float(candidate.cell_size)
    board_h = 20.0 * d
    board_w = 10.0 * d
    tolerance = max(3.0, 0.14 * d)

    def phase_metrics(x0: float):
        left = _nearest_vertical_cluster(
            clusters,
            x0,
            tolerance,
        )
        right = _nearest_vertical_cluster(
            clusters,
            x0 + board_w,
            tolerance,
        )

        left_span, left_top, left_bottom = (
            _cluster_vertical_extent(left)
        )
        right_span, right_top, right_bottom = (
            _cluster_vertical_extent(right)
        )

        if left_span <= 0.0 or right_span <= 0.0:
            min_coverage = 0.0
            avg_coverage = (
                left_span + right_span
            ) / max(1.0, 2.0 * board_h)
        else:
            min_coverage = min(
                left_span,
                right_span,
            ) / max(1.0, board_h)
            avg_coverage = (
                left_span + right_span
            ) / max(1.0, 2.0 * board_h)

        distance_penalty = 0.0
        if left is None:
            distance_penalty += 2.0
        else:
            distance_penalty += (
                abs(float(left["x"]) - x0)
                / tolerance
            )

        if right is None:
            distance_penalty += 2.0
        else:
            distance_penalty += (
                abs(
                    float(right["x"])
                    - (x0 + board_w)
                )
                / tolerance
            )

        score = (
            2.0 * min_coverage
            + 0.5 * avg_coverage
            - 0.08 * distance_penalty
        )
        return {
            "x0": x0,
            "score": score,
            "min_coverage": min_coverage,
            "avg_coverage": avg_coverage,
            "left": left,
            "right": right,
            "left_top": left_top,
            "right_top": right_top,
            "left_bottom": left_bottom,
            "right_bottom": right_bottom,
        }

    options = [
        phase_metrics(candidate.x + shift * d)
        for shift in range(-3, 4)
    ]
    current = options[3]
    best = max(options, key=lambda item: item["score"])

    needs_x_reanchor = (
        current["min_coverage"] < 0.85
        and best["min_coverage"] >= 0.90
        and (
            best["min_coverage"]
            - current["min_coverage"]
        ) >= 0.12
    )

    chosen = best if needs_x_reanchor else current
    x0 = float(candidate.x)

    if needs_x_reanchor:
        left = chosen["left"]
        right = chosen["right"]
        assert left is not None and right is not None

        # Average the origin implied by both long outer borders.
        x0 = (
            float(left["x"])
            + (
                float(right["x"])
                - board_w
            )
        ) / 2.0

    # If the chosen left/right borders are both almost full-height, their top
    # endpoints give a non-periodic y anchor. Correct y only when the existing
    # candidate is displaced by more than half a cell.
    y0 = float(candidate.y)
    if chosen["min_coverage"] >= 0.85:
        tops = [
            value
            for value in (
                chosen["left_top"],
                chosen["right_top"],
            )
            if math.isfinite(value)
        ]

        if tops:
            top_estimate = float(np.median(tops))

            if abs(y0 - top_estimate) > 0.55 * d:
                search_start = top_estimate - 0.35 * d
                search_end = top_estimate + 0.35 * d

                best_y = None
                yy = search_start
                while yy <= search_end + 1e-9:
                    periodic = _horizontal_grid_energy(
                        edges,
                        x0,
                        yy,
                        d,
                    )

                    # The border endpoint is only approximate because thick
                    # borders produce edges on both sides. Keep a tiny
                    # distance preference without overpowering grid evidence.
                    score = (
                        periodic
                        - 0.002
                        * abs(yy - top_estimate)
                        / max(d, 1e-9)
                    )

                    if (
                        best_y is None
                        or score > best_y[0]
                    ):
                        best_y = (score, yy)
                    yy += 0.5

                if best_y is not None:
                    y0 = float(best_y[1])

    if (
        abs(x0 - candidate.x) < 0.25
        and abs(y0 - candidate.y) < 0.25
    ):
        return candidate

    grid_score = _full_grid_energy(
        edges,
        x0,
        y0,
        d,
    )
    confidence = max(
        0.0,
        min(
            1.0,
            0.30
            * (
                candidate.matched_vertical_lines
                / 11.0
            )
            + 0.70
            * min(1.0, grid_score / 0.35),
        ),
    )

    return PlayfieldCandidate(
        x=x0,
        y=y0,
        w=board_w,
        h=board_h,
        cell_size=d,
        grid_score=grid_score,
        matched_vertical_lines=(
            candidate.matched_vertical_lines
        ),
        confidence=confidence,
    )



def detect_playfields(
    image_bgr: np.ndarray,
    *,
    min_matched_vertical_lines: int = 10,
    min_relative_grid_score: float = 0.55,
    min_absolute_grid_score: float = 0.10,
    max_candidates: int = 8,
) -> list[PlayfieldCandidate]:
    """Detect full-size TETR.IO 10x20 playfields.

    The detector is intentionally geometry-first:
      * it looks for 11 approximately equally spaced vertical grid lines,
      * verifies the 20-row geometry,
      * scores both vertical and horizontal grid energy,
      * suppresses shifted aliases.

    Tiny multiplayer spectator boards are intentionally ignored in V0. The
    active player's full-size board and full-size 1v1 opponent board are the
    targets of this stage.
    """
    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("image_bgr must be a BGR image")

    height, width = image_bgr.shape[:2]
    edges, clusters = _vertical_line_clusters(image_bgr)
    if len(clusters) < min_matched_vertical_lines:
        return []

    x_positions = np.array([c["x"] for c in clusters], dtype=np.float64)

    # Candidate cell sizes from observed vertical-line pair distances.
    cell_sizes: set[float] = set()
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            dx = x_positions[j] - x_positions[i]
            for cell_delta in range(1, 11):
                d = dx / cell_delta
                # Deliberately exclude very tiny spectator boards.
                if width * 0.007 <= d <= width * 0.045:
                    cell_sizes.add(round(d * 2.0) / 2.0)

    raw_candidates: list[PlayfieldCandidate] = []
    visited: set[tuple[int, float]] = set()

    for d in sorted(cell_sizes):
        tolerance = max(2.5, 0.12 * d)

        for cluster in clusters:
            for starting_grid_index in range(11):
                x0 = cluster["x"] - starting_grid_index * d
                if x0 < 0 or x0 + 10.0 * d >= width:
                    continue

                key = (int(round(x0 / 2.0) * 2), d)
                if key in visited:
                    continue
                visited.add(key)

                matched: list[dict[str, Any] | None] = []
                for k in range(11):
                    target_x = x0 + k * d
                    idx = int(np.argmin(np.abs(x_positions - target_x)))
                    if abs(x_positions[idx] - target_x) <= tolerance:
                        matched.append(clusters[idx])
                    else:
                        matched.append(None)

                matched_count = sum(x is not None for x in matched)
                if matched_count < min_matched_vertical_lines:
                    continue

                starts = [
                    y1
                    for item in matched
                    if item is not None
                    for y1, _ in item["segments"]
                ]
                if not starts:
                    continue

                y_seed = float(np.percentile(starts, 20))
                board_height = 20.0 * d

                search_lo = max(0.0, y_seed - d)
                search_hi = min(float(height) - board_height, y_seed + d)
                if search_hi < search_lo:
                    continue

                best: tuple[float, float, float, float] | None = None
                step = max(1.0, d / 8.0)

                yy = search_lo
                while yy <= search_hi + 1e-6:
                    y1 = int(round(yy))
                    y2 = int(round(yy + board_height))
                    if y2 <= y1:
                        yy += step
                        continue

                    vertical_scores: list[float] = []
                    for k in range(11):
                        xx = int(round(x0 + k * d))
                        strip = edges[
                            y1:y2,
                            max(0, xx - 1):min(width, xx + 2),
                        ]
                        vertical_scores.append(
                            float(strip.mean() / 255.0) if strip.size else 0.0
                        )

                    x1 = int(round(x0))
                    x2 = int(round(x0 + 10.0 * d))
                    horizontal_scores: list[float] = []
                    for r in range(21):
                        y = int(round(yy + r * d))
                        strip = edges[
                            max(0, y - 1):min(height, y + 2),
                            x1:x2,
                        ]
                        horizontal_scores.append(
                            float(strip.mean() / 255.0) if strip.size else 0.0
                        )

                    vertical = float(np.mean(vertical_scores))
                    horizontal = float(np.mean(horizontal_scores))
                    grid_score = 0.55 * vertical + 0.45 * horizontal

                    if best is None or grid_score > best[0]:
                        best = (grid_score, float(yy), vertical, horizontal)
                    yy += step

                if best is None:
                    continue

                grid_score, best_y, _, _ = best
                confidence = max(
                    0.0,
                    min(
                        1.0,
                        0.30 * (matched_count / 11.0)
                        + 0.70 * min(1.0, grid_score / 0.35),
                    ),
                )

                raw_candidates.append(
                    PlayfieldCandidate(
                        x=float(x0),
                        y=best_y,
                        w=float(10.0 * d),
                        h=float(20.0 * d),
                        cell_size=float(d),
                        grid_score=float(grid_score),
                        matched_vertical_lines=int(matched_count),
                        confidence=float(confidence),
                    )
                )

    # Prefer structurally complete candidates.
    raw_candidates.sort(
        key=lambda c: (c.matched_vertical_lines, c.grid_score),
        reverse=True,
    )

    # Remove the common one-cell-shift aliases produced by HOLD/NEXT borders.
    deduplicated: list[PlayfieldCandidate] = []
    for candidate in raw_candidates:
        is_duplicate = False
        for accepted in deduplicated:
            scale_delta = abs(candidate.cell_size - accepted.cell_size) / max(
                candidate.cell_size,
                accepted.cell_size,
            )
            same_scale = scale_delta < 0.12
            horizontal_center_delta = abs(
                candidate.center[0] - accepted.center[0]
            )
            nearby_same_board = (
                same_scale
                and horizontal_center_delta
                < 0.65 * max(candidate.w, accepted.w)
            )
            overlapping_same_board = (
                same_scale
                and _iou(candidate, accepted) > 0.25
            )
            if nearby_same_board or overlapping_same_board:
                is_duplicate = True
                break
        if not is_duplicate:
            deduplicated.append(candidate)

    if not deduplicated:
        return []

    max_grid_score = max(x.grid_score for x in deduplicated)
    score_floor = max(
        min_absolute_grid_score,
        max_grid_score * min_relative_grid_score,
    )

    filtered = [
        x
        for x in deduplicated
        if x.matched_vertical_lines >= min_matched_vertical_lines
        and x.grid_score >= score_floor
    ]

    # Resolve whole-cell phase ambiguity after coarse candidate selection.
    filtered = [
        _phase_lock_candidate(
            image_bgr,
            edges,
            clusters,
            candidate,
        )
        for candidate in filtered[:max_candidates]
    ]

    # Stable left-to-right ordering is convenient for diagnostics and OCR.
    filtered.sort(key=lambda x: x.x)
    return filtered


def username_strip_bbox(
    candidate: PlayfieldCandidate,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Region immediately around/below a playfield where TETR.IO shows name."""
    height, width = image_shape[:2]

    x1 = max(0, int(round(candidate.x - 0.08 * candidate.w)))
    x2 = min(
        width,
        int(round(candidate.x + 1.08 * candidate.w)),
    )
    y1 = max(
        0,
        int(round(candidate.y + candidate.h - 0.025 * candidate.h)),
    )
    y2 = min(
        height,
        int(round(candidate.y + candidate.h + 0.10 * candidate.h)),
    )
    return x1, y1, x2 - x1, y2 - y1


def normalize_username(value: str | None) -> str:
    if not value:
        return ""
    return "".join(
        ch
        for ch in value.upper().strip()
        if ch.isalnum() or ch in "_-"
    )


def username_similarity(observed: str | None, expected: str | None) -> float:
    a = normalize_username(observed)
    b = normalize_username(expected)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return min(len(a), len(b)) / max(len(a), len(b))
    return float(SequenceMatcher(None, a, b).ratio())


class RapidUsernameReader:
    """Lazy RapidOCR wrapper so OpenCV-only detection remains usable."""

    def __init__(self) -> None:
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Username OCR requires rapidocr-onnxruntime. Install with:\n"
                    r"  .venv\Scripts\python.exe -m pip install rapidocr-onnxruntime"
                ) from exc
            self._engine = RapidOCR()
        return self._engine

    def read(
        self,
        image_bgr: np.ndarray,
        candidate: PlayfieldCandidate,
    ) -> OcrObservation:
        x, y, w, h = username_strip_bbox(candidate, image_bgr.shape)
        crop = image_bgr[y:y+h, x:x+w]
        if crop.size == 0:
            return OcrObservation(None, 0.0)

        # Upscaling is useful because username strips can be quite small.
        scale = 3.0 if max(crop.shape[:2]) < 500 else 2.0
        enlarged = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

        engine = self._get_engine()
        result, _ = engine(enlarged)

        if not result:
            return OcrObservation(None, 0.0)

        best_text = None
        best_score = 0.0
        for item in result:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            text = str(item[1]).strip()
            try:
                score = float(item[2])
            except (TypeError, ValueError):
                score = 0.0

            normalized = normalize_username(text)
            if len(normalized) < 3:
                continue

            # Prefer a confident, username-like token.
            quality = score * min(1.0, len(normalized) / 8.0)
            if quality > best_score:
                best_text = text
                best_score = quality

        return OcrObservation(best_text, best_score)


def resolve_playfield_roles(
    image_bgr: np.ndarray,
    candidates: list[PlayfieldCandidate],
    *,
    self_username: str | None,
    ocr_reader: Callable[
        [np.ndarray, PlayfieldCandidate],
        OcrObservation,
    ] | None = None,
    self_match_threshold: float = 0.72,
    self_match_margin: float = 0.08,
) -> list[ResolvedPlayfield]:
    """Assign SELF / OPPONENT without guessing on ambiguous multi-board views."""
    if not candidates:
        return []

    if ocr_reader is None:
        rapid = RapidUsernameReader()
        ocr_reader = rapid.read

    observations: list[OcrObservation] = []
    for candidate in candidates:
        try:
            observations.append(ocr_reader(image_bgr, candidate))
        except RuntimeError:
            # OCR package missing: board detection still works. The single-board
            # case can still be resolved safely.
            observations.append(OcrObservation(None, 0.0))

    # One dominant full-size board: this is SELF in the supported V0 modes.
    # Do not OCR score/time/VS text below single-player boards as usernames.
    if len(candidates) == 1:
        configured = (
            self_username.strip()
            if isinstance(self_username, str) and self_username.strip()
            else None
        )
        return [
            ResolvedPlayfield(
                candidate=candidates[0],
                role="SELF",
                username=configured,
                username_ocr_confidence=0.0,
                self_match_score=1.0 if configured else 0.0,
            )
        ]

    expected = normalize_username(self_username)
    match_scores: list[float] = []
    for obs in observations:
        similarity = username_similarity(obs.text, expected)
        # Keep OCR confidence in the score, but do not destroy a very close
        # text match merely because OCR confidence is slightly conservative.
        score = similarity * (0.65 + 0.35 * max(0.0, min(1.0, obs.confidence)))
        match_scores.append(score)

    best_index = int(np.argmax(match_scores))
    sorted_scores = sorted(match_scores, reverse=True)
    best_score = sorted_scores[0]
    runner_up = sorted_scores[1] if len(sorted_scores) > 1 else 0.0

    self_is_resolved = (
        bool(expected)
        and best_score >= self_match_threshold
        and best_score - runner_up >= self_match_margin
    )

    resolved: list[ResolvedPlayfield] = []
    for idx, (candidate, obs, score) in enumerate(
        zip(candidates, observations, match_scores)
    ):
        if self_is_resolved and idx == best_index:
            role = "SELF"
        elif self_is_resolved:
            role = "OPPONENT"
        else:
            role = "UNKNOWN"

        resolved.append(
            ResolvedPlayfield(
                candidate=candidate,
                role=role,
                username=obs.text,
                username_ocr_confidence=obs.confidence,
                self_match_score=float(score),
            )
        )
    return resolved


def draw_layout_overlay(
    image_bgr: np.ndarray,
    resolved: list[ResolvedPlayfield],
) -> np.ndarray:
    out = image_bgr.copy()

    role_color = {
        "SELF": (0, 255, 0),
        "OPPONENT": (0, 165, 255),
        "UNKNOWN": (0, 0, 255),
    }

    for index, item in enumerate(resolved):
        x, y, w, h = item.candidate.bbox
        color = role_color.get(item.role, (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)

        name = item.username or "?"
        label = (
            f"{item.role} #{index}  {name}  "
            f"grid={item.candidate.grid_score:.3f} "
            f"conf={item.candidate.confidence:.3f} "
            f"self={item.self_match_score:.3f}"
        )
        label_y = max(25, y - 10)
        cv2.putText(
            out,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        sx, sy, sw, sh = username_strip_bbox(
            item.candidate,
            image_bgr.shape,
        )
        cv2.rectangle(
            out,
            (sx, sy),
            (sx + sw, sy + sh),
            color,
            1,
        )

    return out
