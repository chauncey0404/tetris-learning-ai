from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import json
import math
import itertools
import time
import pathlib
from pathlib import Path
import random
from typing import Iterable, Optional

import numpy as np
import torch

from tetrio.network.cache import batches_from_shard, load_shard, shard_paths
from tetrio.network.encoding import (
    EMPTY_PIECE_ID,
    PIECES,
    unpack_boards,
)
from tetrio.network.model import TetrioExpertV0Network


# Visual style follows the project's existing tools/watch_models.py V3.x viewer.
BG = (18, 20, 24)
PANEL_BG = (27, 30, 36)
PANEL_BORDER = (64, 69, 80)
GRID = (48, 53, 62)
EMPTY = (21, 24, 29)
TEXT = (232, 235, 240)
MUTED = (160, 166, 178)
GOOD = (105, 222, 143)
WARN = (255, 194, 88)
BAD = (255, 112, 112)

TETROMINO_COLORS = {
    "I": (0, 240, 240),
    "J": (0, 80, 240),
    "L": (240, 160, 0),
    "O": (240, 240, 0),
    "S": (0, 220, 0),
    "T": (160, 0, 240),
    "Z": (240, 0, 0),
}

# Match the old watch_models V3.4 visual IDs. 0=empty, 1=unknown/garbage.
VIS_PIECE_ID = {
    "I": 2,
    "O": 3,
    "T": 4,
    "S": 5,
    "Z": 6,
    "J": 7,
    "L": 8,
}
VIS_ID_TO_PIECE = {value: key for key, value in VIS_PIECE_ID.items()}

# 0 = empty
# 1 = occupied in the validated binary cache, but exact historical identity
#     could not be recovered from the source parquet.
# 9 = exact non-tetromino occupied cell from the historical source playfield;
#     in this versus corpus this is rendered as garbage.
UNKNOWN_BLOCK_ID = 1
GARBAGE_BLOCK_ID = 9
UNKNOWN_BLOCK_COLOR = (170, 176, 188)
GARBAGE_BLOCK_COLOR = (92, 98, 108)

# The historical corpus has the already-validated J/L extractor naming swap.
# Canonicalize source piece identity before choosing the display color.
SOURCE_PIECE_TO_CANONICAL = {
    "I": "I",
    "O": "O",
    "T": "T",
    "S": "S",
    "Z": "Z",
    "J": "L",
    "L": "J",
}

SPEED_PRESETS = {
    1: 0.5,
    2: 1.0,
    3: 2.0,
    4: 4.0,
    5: 6.0,
    6: 10.0,
    7: 15.0,
    8: 25.0,
    9: 60.0,
}

ID_TO_PIECE = {i: p for i, p in enumerate(PIECES)}
ID_TO_PIECE[EMPTY_PIECE_ID] = "-"


@dataclass(frozen=True)
class CandidateView:
    index: int
    piece: str
    rotation: int
    x: int
    y: int
    use_hold: bool
    lines: int
    score: float
    board_after: np.ndarray
    board_after_ids: Optional[np.ndarray] = None


@dataclass(frozen=True)
class CaseView:
    shard_name: str
    sample_index: int
    game_id: int
    subframe: int
    board_before: np.ndarray
    board_before_ids: Optional[np.ndarray]
    active: str
    hold: str
    preview: tuple[str, ...]
    expert_use_hold: bool
    hold_probability: float
    expert_index: int
    expert_rank: int
    top_candidates: tuple[CandidateView, ...]
    expert_candidate: CandidateView

    @property
    def top1_matches(self) -> bool:
        return self.top_candidates[0].index == self.expert_index

    @property
    def expert_in_top3(self) -> bool:
        return self.expert_rank <= 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Graphically inspect TETR.IO Expert-v0 held-out decisions. "
            "The UI mirrors the older watch_models V3.x visual style, but loads "
            "the Expert-v0 candidate-ranking architecture and test cache."
        )
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"models\tetrio_expert_v0_full.pt"),
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=Path(r"data\tetrio\expert_v0\test_full_fast_s8192"),
    )
    p.add_argument(
        "--source",
        type=Path,
        default=Path(r"data\tetrio\expert\top_players_s1_test.parquet"),
        help=(
            "Optional held-out source parquet used only to restore historical "
            "tetromino colors. If it contains playfield/piece IDs, the GUI "
            "reconstructs the old V3.4 per-piece palette. Model inference still "
            "uses only the validated binary cache."
        ),
    )
    p.add_argument(
        "--color-source-fallback",
        type=Path,
        default=Path(r"data\tetrio\processed\top_players_s1.parquet"),
        help=(
            "Fallback source for exact historical playfield colors. The "
            "processed 138 MiB corpus usually retains the original playfield."
        ),
    )
    p.add_argument(
        "--no-source-colors",
        action="store_true",
        help="Skip source-color enrichment; unknown locked cells stay gray.",
    )
    p.add_argument(
        "--filter",
        choices=("disagreement", "top3_miss", "all"),
        default="disagreement",
        help=(
            "disagreement: model Top-1 != expert; "
            "top3_miss: expert rank > 3; all: any test row."
        ),
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=300,
        help="Reservoir-sampled cases kept for GUI browsing.",
    )
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--width", type=int, default=1500)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--start", type=int, default=1)
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Autoplay cases per second; 1-9 and +/- can change this in the GUI.",
    )
    p.add_argument(
        "--scan-only",
        action="store_true",
        help="Scan/filter and print summary without opening pygame.",
    )
    p.add_argument(
        "--save-summary",
        type=Path,
        default=Path(r"artifacts\tetrio\expert_v0_gui_scan.json"),
    )
    return p.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> tuple[TetrioExpertV0Network, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    safe_path_globals = [
        pathlib.Path,
        pathlib.PurePath,
        pathlib.PurePosixPath,
        pathlib.PureWindowsPath,
        pathlib.PosixPath,
        pathlib.WindowsPath,
    ]
    with torch.serialization.safe_globals(safe_path_globals):
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

    if checkpoint.get("format") != "tetrio_expert_v0":
        raise RuntimeError(
            f"Unsupported checkpoint format: {checkpoint.get('format')!r}"
        )

    model = TetrioExpertV0Network().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def piece_name(value: int) -> str:
    return ID_TO_PIECE.get(int(value), "?")


def visible_board(board40: np.ndarray) -> np.ndarray:
    """TETR.IO cache stores 40 rows; render the normal bottom 20-row field."""
    board = np.asarray(board40).reshape(40, 10)
    return board[20:40]


def piece_color(piece: Optional[str]) -> tuple[int, int, int]:
    if piece is None:
        return UNKNOWN_BLOCK_COLOR
    return TETROMINO_COLORS.get(str(piece), UNKNOWN_BLOCK_COLOR)


def brighten(color: tuple[int, int, int], amount: int = 42) -> tuple[int, int, int]:
    return tuple(min(255, int(c) + amount) for c in color)


def darken(color: tuple[int, int, int], factor: float = 0.58) -> tuple[int, int, int]:
    return tuple(max(0, int(c * factor)) for c in color)


def occupancy_to_unknown_ids(board40: np.ndarray) -> np.ndarray:
    board = np.asarray(board40).reshape(40, 10)
    return np.where(board != 0, UNKNOWN_BLOCK_ID, 0).astype(np.uint8)


def _decode_piece_char_grid(values: list[str]) -> Optional[np.ndarray]:
    """
    Decode the historical TETR.IO playfield without dropping any serialized cell.

    The playfield is a floor-up flat string:
      index 0..9   = bottom row
      index 10..19 = row above it
      ...

    N is empty. I/O/T/S/Z/J/L are tetromino identities. Literal G is garbage. Any other unexpected non-empty source code is preserved positionally and rendered as Unknown identity.

    IMPORTANT: never filter unknown characters out of the string. Doing so
    collapses indices after a garbage cell and produces visually shifted colors.
    """
    chars = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        s = str(value)
        if len(s) == 1:
            chars.append(s.upper())
        else:
            chars.extend(list(s.upper()))

    if not chars:
        return None
    if len(chars) > 400:
        return None

    out = np.zeros((40, 10), dtype=np.uint8)
    for i, ch in enumerate(chars):
        floor_row, col = divmod(i, 10)
        if floor_row >= 40:
            break
        y = 39 - floor_row

        if ch == "N":
            visual_id = 0
        elif ch in SOURCE_PIECE_TO_CANONICAL:
            canonical = SOURCE_PIECE_TO_CANONICAL[ch]
            visual_id = int(VIS_PIECE_ID[canonical])
        elif ch == "G":
            # Historical corpus inspection confirms literal G is Garbage.
            visual_id = GARBAGE_BLOCK_ID
        else:
            # Preserve unexpected source codes without falsely labelling them
            # as garbage. The validated model still sees only occupancy.
            visual_id = UNKNOWN_BLOCK_ID

        out[y, col] = visual_id
    return out


def decode_playfield_ids(raw) -> Optional[np.ndarray]:
    """Exact, position-preserving visual decode of historical playfield IDs."""
    if raw is None:
        return None

    if isinstance(raw, np.ndarray):
        raw = raw.tolist()

    if isinstance(raw, (list, tuple)):
        arr = np.asarray(raw, dtype=object).reshape(-1)
        if arr.size == 0:
            return None

        if all(isinstance(x, (str, bytes, np.str_)) for x in arr):
            return _decode_piece_char_grid(list(arr))

        try:
            numeric = np.asarray(raw).reshape(-1)
            if numeric.size == 400 and np.issubdtype(numeric.dtype, np.number):
                numeric = numeric.astype(np.int16, copy=False)
                uniq = set(int(x) for x in np.unique(numeric))
                # Preserve old Gym/V3.4 visual IDs when available.
                allowed = set(range(0, 10))
                if uniq.issubset(allowed):
                    return numeric.reshape(40, 10).astype(np.uint8)
        except Exception:
            pass
        return None

    if isinstance(raw, bytes):
        text = raw.decode(errors="replace")
    else:
        text = str(raw)

    # DuckDB returns the parquet VARCHAR directly, so every character is one
    # serialized board cell. Keep all characters to preserve exact positions.
    if not text:
        return np.zeros((40, 10), dtype=np.uint8)
    if len(text) > 400:
        return None
    return _decode_piece_char_grid(list(text))


def _propagate_colors_after_clear(
    before_ids: np.ndarray,
    after_binary: np.ndarray,
    *,
    placed_piece: str,
    lines: int,
) -> np.ndarray:
    """
    Carry old per-piece colors through a hypothetical candidate board.

    Expert-v0 cache intentionally stores occupancy only. When the source board
    gives us historical colors, this function preserves those colors and paints
    cells introduced by the candidate with the candidate tetromino color-ID.
    For line clears we infer the cleared rows by matching the authoritative
    after-board. This is display-only and never affects model evaluation.
    """
    before_ids = np.asarray(before_ids, dtype=np.uint8).reshape(40, 10)
    before = before_ids != 0
    after = np.asarray(after_binary).reshape(40, 10) != 0
    piece_id = int(VIS_PIECE_ID.get(placed_piece, UNKNOWN_BLOCK_ID))
    line_count = max(0, min(4, int(lines)))

    if line_count == 0:
        out = np.zeros((40, 10), dtype=np.uint8)
        keep = before & after
        out[keep] = before_ids[keep]
        new_cells = after & (out == 0)
        out[new_cells] = piece_id
        return out

    candidate_rows = [
        y for y in range(40)
        if 6 <= int(np.count_nonzero(before[y])) <= 9
    ]
    if len(candidate_rows) < line_count:
        candidate_rows = list(range(40))

    best_score = -10**9
    best_out: Optional[np.ndarray] = None

    # In normal play this search set is small because a row needs at least six
    # existing cells to become full after adding a four-cell tetromino.
    for cleared in itertools.combinations(candidate_rows, line_count):
        cleared_set = set(cleared)
        out = np.zeros((40, 10), dtype=np.uint8)
        preserved = 0
        contradictions = 0

        for y in range(40):
            if y in cleared_set:
                continue
            shift = sum(1 for row in cleared if row > y)
            ny = y + shift
            if ny >= 40:
                continue
            for x in np.flatnonzero(before[y]):
                if after[ny, x]:
                    out[ny, x] = before_ids[y, x]
                    preserved += 1
                else:
                    contradictions += 1

        missing = int(np.count_nonzero(after & (out == 0)))
        score = preserved * 20 - contradictions * 50 - abs(missing - max(0, 4 - 10 * line_count))
        if score > best_score:
            best_score = score
            best_out = out

    if best_out is None:
        best_out = np.zeros((40, 10), dtype=np.uint8)

    best_out[~after] = 0
    new_cells = after & (best_out == 0)
    best_out[new_cells] = piece_id
    return best_out


def _source_columns(con, source: Path) -> set[str]:
    quoted = str(source).replace("'", "''")
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{quoted}')"
    ).fetchall()
    return {str(row[0]) for row in rows}


def enrich_case_colors(cases: list[CaseView], source: Path) -> tuple[list[CaseView], str]:
    """
    Restore exact historical block colors when the source parquet retained them.

    The neural network/cache contract remains binary occupancy. This extra read
    is strictly for rendering and is fail-soft: if the source does not expose
    piece identity, the viewer still works and only unknown locked cells stay gray.
    """
    if not cases:
        return cases, "none"
    if not source.is_file():
        print(f"Color source  : unavailable ({source}) -> occupancy fallback")
        return cases, "occupancy-fallback"

    try:
        import duckdb
    except Exception:
        print("Color source  : duckdb unavailable -> occupancy fallback")
        return cases, "occupancy-fallback"

    con = duckdb.connect(database=":memory:")
    try:
        columns = _source_columns(con, source)
        if not {"game_id", "subframe"}.issubset(columns):
            print("Color source  : no game_id/subframe -> occupancy fallback")
            return cases, "occupancy-fallback"

        preferred = None
        for name in ("playfield", "board_before_ids", "board_ids", "board_before"):
            if name in columns:
                preferred = name
                break
        if preferred is None:
            print("Color source  : no piece-identity board column -> occupancy fallback")
            return cases, "occupancy-fallback"

        wanted = sorted({(int(c.game_id), int(c.subframe)) for c in cases})
        values = ",".join(f"({gid},{sub})" for gid, sub in wanted)
        quoted = str(source).replace("'", "''")
        query = f"""
            WITH wanted(game_id, subframe) AS (VALUES {values})
            SELECT p.game_id, p.subframe, p.{preferred}
            FROM read_parquet('{quoted}') AS p
            INNER JOIN wanted AS w
              ON p.game_id = w.game_id AND p.subframe = w.subframe
        """
        rows = con.execute(query).fetchall()
        decoded: dict[tuple[int, int], np.ndarray] = {}
        for gid, sub, raw in rows:
            ids = decode_playfield_ids(raw)
            if ids is None and preferred == "board_before":
                # Some expert parquets store board_before as binary occupancy.
                # Do not invent colors from binary data.
                continue
            if ids is not None:
                decoded[(int(gid), int(sub))] = ids

        if not decoded:
            print(
                f"Color source  : {preferred} has occupancy only -> "
                "current-placement colors + gray historical stack"
            )
            return cases, "occupancy-fallback"

        enriched: list[CaseView] = []
        exact = 0
        for case in cases:
            ids = decoded.get((case.game_id, case.subframe))
            if ids is None:
                before_ids = occupancy_to_unknown_ids(case.board_before)
            else:
                # Authoritative binary cache wins on occupancy. Source is used
                # only for the visual identity of cells that really exist.
                occupancy = np.asarray(case.board_before).reshape(40, 10) != 0
                before_ids = np.asarray(ids, dtype=np.uint8).reshape(40, 10).copy()
                before_ids[~occupancy] = 0
                before_ids[occupancy & (before_ids == 0)] = UNKNOWN_BLOCK_ID
                exact += 1

            top = tuple(
                replace(
                    cand,
                    board_after_ids=_propagate_colors_after_clear(
                        before_ids,
                        cand.board_after,
                        placed_piece=cand.piece,
                        lines=cand.lines,
                    ),
                )
                for cand in case.top_candidates
            )
            expert = replace(
                case.expert_candidate,
                board_after_ids=_propagate_colors_after_clear(
                    before_ids,
                    case.expert_candidate.board_after,
                    placed_piece=case.expert_candidate.piece,
                    lines=case.expert_candidate.lines,
                ),
            )
            enriched.append(
                replace(
                    case,
                    board_before_ids=before_ids,
                    top_candidates=top,
                    expert_candidate=expert,
                )
            )

        garbage_cells = 0
        unknown_cells = 0
        colored_cells = 0
        for case in enriched:
            ids = case.board_before_ids
            if ids is None:
                continue
            garbage_cells += int(np.count_nonzero(ids == GARBAGE_BLOCK_ID))
            unknown_cells += int(np.count_nonzero(ids == UNKNOWN_BLOCK_ID))
            colored_cells += int(np.count_nonzero(
                (ids >= min(VIS_PIECE_ID.values()))
                & (ids <= max(VIS_PIECE_ID.values()))
            ))

        print(
            f"Color source  : {preferred} exact={exact}/{len(cases)} "
            f"piece_cells={colored_cells:,} garbage_cells={garbage_cells:,} "
            f"unknown_cells={unknown_cells:,}"
        )
        return enriched, f"{preferred}:{exact}/{len(cases)}"
    except Exception as exc:
        print(f"Color source  : failed ({type(exc).__name__}: {exc}) -> fallback")
        return cases, "occupancy-fallback"
    finally:
        con.close()


def candidate_from_raw(
    data: dict[str, np.ndarray],
    absolute_index: int,
    score: float,
    candidate_index: int,
) -> CandidateView:
    board = unpack_boards(
        data["candidate_board_packed"][absolute_index:absolute_index + 1]
    )[0].reshape(40, 10)
    return CandidateView(
        index=int(candidate_index),
        piece=piece_name(int(data["candidate_piece"][absolute_index])),
        rotation=int(data["candidate_rotation"][absolute_index]) % 4,
        x=int(data["candidate_x"][absolute_index]),
        y=int(data["candidate_y"][absolute_index]),
        use_hold=bool(data["candidate_use_hold"][absolute_index]),
        lines=int(data["candidate_lines"][absolute_index]),
        score=float(score),
        board_after=board,
    )


def matches_filter(kind: str, top1: int, expert_index: int, expert_rank: int) -> bool:
    if kind == "all":
        return True
    if kind == "disagreement":
        return top1 != expert_index
    if kind == "top3_miss":
        return expert_rank > 3
    raise ValueError(kind)


def reservoir_add(
    reservoir: list[CaseView],
    case: CaseView,
    *,
    matched_count: int,
    capacity: int,
    rng: random.Random,
) -> None:
    if len(reservoir) < capacity:
        reservoir.append(case)
        return
    slot = rng.randrange(matched_count)
    if slot < capacity:
        reservoir[slot] = case


def scan_cases(
    *,
    model: TetrioExpertV0Network,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    filter_kind: str,
    max_cases: int,
    top_k: int,
    seed: int,
) -> tuple[list[CaseView], dict]:
    rng = random.Random(seed)
    reservoir: list[CaseView] = []

    rows_total = 0
    top1_correct = 0
    top3_correct = 0
    matched_filter = 0
    hold_correct = 0

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif device.type == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    print("=" * 100)
    print("TETR.IO EXPERT V0 — GUI CASE SCAN")
    print("=" * 100)
    print(f"Filter       : {filter_kind}")
    print(f"Reservoir    : {max_cases}")
    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(device)}")
    print()

    with torch.inference_mode():
        for shard_path in shard_paths(cache_dir):
            data = load_shard(shard_path)
            n = int(data["expert_index"].shape[0])
            offsets = data["candidate_offsets"].astype(np.int64, copy=False)
            local_start = 0

            for batch in batches_from_shard(
                data,
                batch_size=batch_size,
                rng=None,
            ):
                state = torch.from_numpy(batch.state).to(
                    device=device,
                    non_blocking=device.type == "cuda",
                )
                candidates = torch.from_numpy(batch.candidates).to(
                    device=device,
                    non_blocking=device.type == "cuda",
                )
                mask = torch.from_numpy(batch.candidate_mask).to(
                    device=device,
                    non_blocking=device.type == "cuda",
                )
                target = torch.from_numpy(batch.expert_index).to(
                    device=device,
                    dtype=torch.long,
                    non_blocking=device.type == "cuda",
                )

                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=device.type == "cuda",
                ):
                    scores, hold_logit = model(
                        state=state,
                        candidates=candidates,
                    )

                masked_scores = scores.masked_fill(~mask, float("-inf"))
                order = torch.argsort(masked_scores, dim=1, descending=True)
                top1 = order[:, 0]
                ranks = (
                    (order == target[:, None])
                    .to(torch.int64)
                    .argmax(dim=1)
                    + 1
                )
                hold_pred = hold_logit >= 0
                hold_target = torch.from_numpy(batch.use_hold).to(
                    device=device,
                    dtype=torch.float32,
                ) >= 0.5

                b = int(target.shape[0])
                rows_total += b
                top1_correct += int((top1 == target).sum().item())
                top3_correct += int((ranks <= 3).sum().item())
                hold_correct += int((hold_pred == hold_target).sum().item())

                top1_cpu = top1.cpu().numpy()
                ranks_cpu = ranks.cpu().numpy()
                order_cpu = order[:, :max(1, top_k)].cpu().numpy()
                scores_cpu = masked_scores.float().cpu().numpy()
                hold_prob_cpu = torch.sigmoid(hold_logit.float()).cpu().numpy()
                target_cpu = target.cpu().numpy()

                for row_pos in range(b):
                    expert_index = int(target_cpu[row_pos])
                    top1_index = int(top1_cpu[row_pos])
                    expert_rank = int(ranks_cpu[row_pos])
                    if not matches_filter(
                        filter_kind,
                        top1_index,
                        expert_index,
                        expert_rank,
                    ):
                        continue

                    matched_filter += 1
                    sample_idx = local_start + row_pos
                    lo = int(offsets[sample_idx])
                    hi = int(offsets[sample_idx + 1])

                    before = unpack_boards(
                        data["state_board_packed"][sample_idx:sample_idx + 1]
                    )[0].reshape(40, 10)

                    top_views = []
                    seen = set()
                    for candidate_index in order_cpu[row_pos]:
                        candidate_index = int(candidate_index)
                        if candidate_index >= hi - lo:
                            continue
                        if candidate_index in seen:
                            continue
                        seen.add(candidate_index)
                        absolute = lo + candidate_index
                        top_views.append(
                            candidate_from_raw(
                                data,
                                absolute,
                                float(scores_cpu[row_pos, candidate_index]),
                                candidate_index,
                            )
                        )

                    expert_absolute = lo + expert_index
                    expert_view = candidate_from_raw(
                        data,
                        expert_absolute,
                        float(scores_cpu[row_pos, expert_index]),
                        expert_index,
                    )

                    preview = tuple(
                        piece_name(x)
                        for x in data["state_preview"][sample_idx]
                        if int(x) != EMPTY_PIECE_ID
                    )

                    case = CaseView(
                        shard_name=shard_path.name,
                        sample_index=sample_idx,
                        game_id=(
                            int(data["game_id"][sample_idx])
                            if "game_id" in data else -1
                        ),
                        subframe=(
                            int(data["subframe"][sample_idx])
                            if "subframe" in data else -1
                        ),
                        board_before=before.copy(),
                        board_before_ids=occupancy_to_unknown_ids(before),
                        active=piece_name(int(data["state_active"][sample_idx])),
                        hold=piece_name(int(data["state_hold"][sample_idx])),
                        preview=preview,
                        expert_use_hold=bool(data["use_hold"][sample_idx]),
                        hold_probability=float(hold_prob_cpu[row_pos]),
                        expert_index=expert_index,
                        expert_rank=expert_rank,
                        top_candidates=tuple(top_views),
                        expert_candidate=expert_view,
                    )
                    reservoir_add(
                        reservoir,
                        case,
                        matched_count=matched_filter,
                        capacity=max_cases,
                        rng=rng,
                    )

                local_start += b

            print(
                f"Scanned {shard_path.name}: "
                f"rows={rows_total:,} filter_hits={matched_filter:,}"
            )

    rng.shuffle(reservoir)
    summary = {
        "rows": rows_total,
        "top1": top1_correct / rows_total if rows_total else 0.0,
        "top3": top3_correct / rows_total if rows_total else 0.0,
        "hold_acc": hold_correct / rows_total if rows_total else 0.0,
        "filter": filter_kind,
        "filter_hits": matched_filter,
        "sampled_cases": len(reservoir),
        "seed": seed,
    }
    print()
    print(f"Rows         : {rows_total:,}")
    print(f"Top-1        : {summary['top1']:.4f}")
    print(f"Top-3        : {summary['top3']:.4f}")
    print(f"Hold acc     : {summary['hold_acc']:.4f}")
    print(f"Filter hits  : {matched_filter:,}")
    print(f"GUI cases    : {len(reservoir):,}")
    return reservoir, summary


class Inspector:
    def __init__(
        self,
        cases: list[CaseView],
        *,
        width: int,
        height: int,
        start: int,
        screenshot_dir: Path,
        speed: float,
    ) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame is required for the graphical inspector.\n"
                r"Install with: .venv\Scripts\python.exe -m pip install pygame"
            ) from exc

        self.pygame = pygame
        pygame.init()
        pygame.display.set_caption(
            "Tetris Learning AI - TETR.IO Expert v0 Inspector V3.4 Style / Exact Colors"
        )
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        self.font_med = pygame.font.Font(None, 27)
        self.font_big = pygame.font.Font(None, 30)

        self.cases = cases
        self.index = max(0, min(len(cases) - 1, start - 1))
        self.show_detail = True
        self.running = True
        self.autoplay = False
        self.speed = max(0.05, float(speed))
        self.last_auto = time.perf_counter()
        self.screenshot_dir = screenshot_dir
        self.control_rects: dict[str, object] = {}

    def _text(self, text: str, x: int, y: int, *, font=None, color=TEXT) -> int:
        font = font or self.font
        img = font.render(str(text), True, color)
        self.screen.blit(img, (x, y))
        return img.get_height()

    def _draw_block(self, rect, color, *, ghost: bool = False) -> None:
        pygame = self.pygame
        if ghost:
            ghost_fill = tuple(max(25, int(c * 0.24)) for c in color)
            pygame.draw.rect(self.screen, ghost_fill, rect.inflate(-4, -4))
            pygame.draw.rect(self.screen, color, rect.inflate(-4, -4), width=1)
            return

        inner = rect.inflate(-2, -2)
        pygame.draw.rect(self.screen, darken(color, 0.55), inner, border_radius=2)
        face = pygame.Rect(
            inner.x + 2,
            inner.y + 2,
            max(1, inner.width - 4),
            max(1, inner.height - 4),
        )
        pygame.draw.rect(self.screen, color, face, border_radius=2)
        if rect.width >= 8 and rect.height >= 8:
            pygame.draw.line(
                self.screen,
                brighten(color, 55),
                (face.left + 1, face.top + 1),
                (face.right - 1, face.top + 1),
                width=1,
            )
            pygame.draw.line(
                self.screen,
                brighten(color, 28),
                (face.left + 1, face.top + 1),
                (face.left + 1, face.bottom - 1),
                width=1,
            )

    def _draw_board(
        self,
        board40: np.ndarray,
        rect,
        *,
        title: str,
        border_color=PANEL_BORDER,
        ids40: Optional[np.ndarray] = None,
    ) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=8)
        pygame.draw.rect(self.screen, border_color, rect, 2, border_radius=8)

        self._text(title, rect.x + 8, rect.y + 7, font=self.font_small)
        inner = pygame.Rect(rect.x + 10, rect.y + 30, rect.width - 20, rect.height - 44)
        board = visible_board(board40)
        ids = visible_board(ids40) if ids40 is not None else None
        cell = max(2, min(inner.width // 10, inner.height // 20))
        bw = cell * 10
        bh = cell * 20
        ox = inner.x + (inner.width - bw) // 2
        oy = inner.y + (inner.height - bh) // 2

        for y in range(20):
            for x in range(10):
                r = pygame.Rect(ox + x * cell, oy + y * cell, cell, cell)
                if not board[y, x]:
                    pygame.draw.rect(self.screen, EMPTY, r.inflate(-1, -1))
                else:
                    piece_id = int(ids[y, x]) if ids is not None else UNKNOWN_BLOCK_ID
                    name = VIS_ID_TO_PIECE.get(piece_id)
                    if name is not None:
                        block_color = piece_color(name)
                    elif piece_id == GARBAGE_BLOCK_ID:
                        block_color = GARBAGE_BLOCK_COLOR
                    else:
                        block_color = UNKNOWN_BLOCK_COLOR
                    self._draw_block(r, block_color)

                    # Small center mark makes exact garbage visibly distinct
                    # from "identity unavailable" silver blocks even on dim screens.
                    if piece_id == GARBAGE_BLOCK_ID and r.width >= 10 and r.height >= 10:
                        mark = pygame.Rect(
                            r.centerx - max(1, r.width // 10),
                            r.centery - max(1, r.height // 10),
                            max(2, r.width // 5),
                            max(2, r.height // 5),
                        )
                        pygame.draw.rect(
                            self.screen,
                            darken(GARBAGE_BLOCK_COLOR, 0.42),
                            mark,
                            border_radius=1,
                        )
                pygame.draw.rect(self.screen, GRID, r, 1)

    def _draw_candidate_panel(
        self,
        candidate: CandidateView,
        rect,
        *,
        title: str,
        color,
    ) -> None:
        ids = candidate.board_after_ids
        if ids is None:
            before = self.cases[self.index].board_before_ids
            if before is None:
                before = occupancy_to_unknown_ids(self.cases[self.index].board_before)
            ids = _propagate_colors_after_clear(
                before,
                candidate.board_after,
                placed_piece=candidate.piece,
                lines=candidate.lines,
            )
        self._draw_board(
            candidate.board_after,
            rect,
            title=title,
            border_color=color,
            ids40=ids,
        )
        line_y = rect.bottom - 24
        self._text(
            f"{candidate.piece} r{candidate.rotation} x={candidate.x} y={candidate.y} "
            f"hold={int(candidate.use_hold)} lines={candidate.lines} score={candidate.score:.3f}",
            rect.x + 8,
            line_y,
            font=self.font_small,
            color=color,
        )

    def _draw_piece_chip(self, piece: str, x: int, y: int, label: str) -> int:
        pygame = self.pygame
        color = piece_color(None if piece == "-" else piece)
        text = f"{label} {piece}"
        img = self.font_small.render(text, True, color)
        w = img.get_width() + 18
        rect = pygame.Rect(x, y, w, 23)
        pygame.draw.rect(self.screen, (35, 39, 47), rect, border_radius=5)
        pygame.draw.rect(self.screen, color, rect, width=1, border_radius=5)
        self.screen.blit(img, (rect.x + 9, rect.y + 3))
        return rect.right

    def _save_screenshot(self) -> None:
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.screenshot_dir / f"expert_v0_case_{self.index+1:04d}_{stamp}.png"
        self.pygame.image.save(self.screen, str(path))
        print(f"Screenshot: {path}")

    def _set_index(self, value: int) -> None:
        self.index = max(0, min(len(self.cases) - 1, int(value)))
        self.last_auto = time.perf_counter()

    def _run_control(self, action: str) -> None:
        if action == "pause":
            self.autoplay = not self.autoplay
            self.last_auto = time.perf_counter()
        elif action == "prev":
            self._set_index(self.index - 1)
        elif action == "next":
            self._set_index(self.index + 1)
        elif action == "prev10":
            self._set_index(self.index - 10)
        elif action == "next10":
            self._set_index(self.index + 10)
        elif action == "first":
            self._set_index(0)
        elif action == "last":
            self._set_index(len(self.cases) - 1)
        elif action == "slower":
            self.speed = max(0.05, self.speed / 1.5)
        elif action == "faster":
            self.speed = min(240.0, self.speed * 1.5)
        elif action == "detail":
            self.show_detail = not self.show_detail
        elif action == "shot":
            self._save_screenshot()
        elif action == "quit":
            self.running = False

    def _handle(self) -> None:
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(
                    (event.w, event.h), pygame.RESIZABLE
                )
                continue
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for action, rect in self.control_rects.items():
                    if rect.collidepoint(event.pos):
                        self._run_control(action)
                        break
                continue
            if event.type != pygame.KEYDOWN:
                continue

            key = event.key
            if key in (pygame.K_ESCAPE, pygame.K_q):
                self._run_control("quit")
            elif key == pygame.K_SPACE:
                self._run_control("pause")
            elif key in (pygame.K_RIGHT, pygame.K_n):
                self._run_control("next")
            elif key in (pygame.K_LEFT, pygame.K_p):
                self._run_control("prev")
            elif key == pygame.K_PAGEDOWN:
                self._run_control("next10")
            elif key == pygame.K_PAGEUP:
                self._run_control("prev10")
            elif key == pygame.K_HOME:
                self._run_control("first")
            elif key == pygame.K_END:
                self._run_control("last")
            elif key == pygame.K_d:
                self._run_control("detail")
            elif key == pygame.K_s:
                self._run_control("shot")
            elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self._run_control("faster")
            elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self._run_control("slower")
            elif pygame.K_1 <= key <= pygame.K_9:
                self.speed = SPEED_PRESETS[int(key - pygame.K_0)]

    def _draw_controls(self, *, y: int, width: int, height: int) -> None:
        pygame = self.pygame
        bar = pygame.Rect(10, y, max(100, width - 20), max(70, height))
        pygame.draw.rect(self.screen, PANEL_BG, bar, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, bar, width=1, border_radius=8)

        controls = [
            ("pause", "Space Play/Pause"),
            ("prev", "← Prev"),
            ("next", "→ Next"),
            ("prev10", "PgUp -10"),
            ("next10", "PgDn +10"),
            ("first", "Home First"),
            ("last", "End Last"),
            ("slower", "- Slower"),
            ("faster", "+ Faster"),
            ("detail", "D Detail"),
            ("shot", "S Screenshot"),
            ("quit", "Esc Quit"),
        ]

        self.control_rects = {}
        x = bar.x + 8
        cy = bar.y + 7
        row_h = 28
        mouse_pos = pygame.mouse.get_pos()

        for action, label in controls:
            label_img = self.font_small.render(label, True, TEXT)
            chip_w = label_img.get_width() + 18
            if x + chip_w > bar.right - 8:
                x = bar.x + 8
                cy += row_h
            rect = pygame.Rect(x, cy, chip_w, 23)
            hovered = rect.collidepoint(mouse_pos)
            fill = (53, 59, 70) if hovered else (35, 39, 47)
            if action == "pause" and self.autoplay:
                fill = (69, 61, 38)
            pygame.draw.rect(self.screen, fill, rect, border_radius=5)
            pygame.draw.rect(self.screen, PANEL_BORDER, rect, width=1, border_radius=5)
            self.screen.blit(label_img, (rect.x + 9, rect.y + 3))
            self.control_rects[action] = rect
            x = rect.right + 6

        case = self.cases[self.index]
        state_name = "PLAYING" if self.autoplay else "PAUSED"
        state_text = (
            f"State: {state_name}   Speed: {self.speed:.1f} cases/s   "
            f"Case: {self.index+1}/{len(self.cases)}   Expert rank: #{case.expert_rank}   "
            "1-9 speed presets   Gray●=Garbage   Silver=Unknown identity"
        )
        self._text(
            state_text,
            bar.x + 10,
            bar.bottom - 23,
            font=self.font_small,
            color=GOOD if self.autoplay else WARN,
        )

    def _draw(self) -> None:
        pygame = self.pygame
        self.screen.fill(BG)
        w, h = self.screen.get_size()
        case = self.cases[self.index]

        status_color = BAD if not case.expert_in_top3 else WARN
        self._text(
            f"TETR.IO Expert v0 — Held-out Inspector   "
            f"Case {self.index+1}/{len(self.cases)}   "
            f"Expert rank #{case.expert_rank}",
            16, 12, font=self.font_big, color=status_color,
        )

        x = 16
        y = 45
        x = self._draw_piece_chip(case.active, x, y, "ACTIVE") + 7
        x = self._draw_piece_chip(case.hold, x, y, "HOLD") + 7
        for i, piece in enumerate(case.preview[:5], 1):
            x = self._draw_piece_chip(piece, x, y, f"N{i}") + 5

        hold_ok = ((case.hold_probability >= 0.5) == case.expert_use_hold)
        self._text(
            f"Game {case.game_id} / subframe {case.subframe}   "
            f"Shard {case.shard_name} / row {case.sample_index}   "
            f"Hold P={case.hold_probability:.3f} vs expert={int(case.expert_use_hold)}",
            16, 73, font=self.font_small,
            color=GOOD if hold_ok else BAD,
        )

        top_y = 100
        footer_h = 104
        gap = 10
        available_h = h - top_y - footer_h
        left_w = max(250, int(w * 0.23))
        right_x = 14 + left_w + gap
        right_w = w - right_x - 14

        before_rect = pygame.Rect(14, top_y, left_w, available_h)
        self._draw_board(
            case.board_before,
            before_rect,
            title="BOARD BEFORE",
            ids40=case.board_before_ids,
        )

        main_gap = 10
        main_w = max(250, (right_w - main_gap) // 2)
        model_rect = pygame.Rect(right_x, top_y, main_w, available_h)
        expert_rect = pygame.Rect(right_x + main_w + main_gap, top_y, main_w, available_h)

        self._draw_candidate_panel(
            case.top_candidates[0],
            model_rect,
            title="MODEL TOP-1",
            color=WARN if not case.top1_matches else GOOD,
        )
        self._draw_candidate_panel(
            case.expert_candidate,
            expert_rect,
            title=f"EXPERT  (model rank #{case.expert_rank})",
            color=GOOD,
        )

        if self.show_detail:
            detail_y = h - footer_h - 20
            tops = " | ".join(
                f"#{rank+1}:{c.piece} r{c.rotation} x{c.x} y{c.y} "
                f"h{int(c.use_hold)} s{c.score:.2f}"
                for rank, c in enumerate(case.top_candidates[:3])
            )
            self._text(tops, 14, detail_y, font=self.font_small, color=MUTED)

        self._draw_controls(
            y=h - footer_h + 4,
            width=w,
            height=footer_h - 8,
        )
        pygame.display.flip()

    def run(self) -> None:
        while self.running:
            self._handle()
            if self.autoplay:
                now = time.perf_counter()
                if now - self.last_auto >= 1.0 / max(self.speed, 0.05):
                    if self.index >= len(self.cases) - 1:
                        self.index = 0
                    else:
                        self.index += 1
                    self.last_auto = now
            self._draw()
            self.clock.tick(60)
        self.pygame.quit()

def main() -> None:
    args = parse_args()
    if args.max_cases <= 0:
        raise SystemExit("--max-cases must be positive")
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)

    cases, summary = scan_cases(
        model=model,
        cache_dir=args.cache,
        device=device,
        batch_size=args.batch_size,
        filter_kind=args.filter,
        max_cases=args.max_cases,
        top_k=max(3, args.top_k),
        seed=args.seed,
    )
    summary["checkpoint"] = str(args.checkpoint)
    summary["checkpoint_epoch"] = checkpoint.get("epoch")
    summary["cache"] = str(args.cache)

    if not args.no_source_colors:
        cases, color_source = enrich_case_colors(cases, args.source)
        if (
            color_source == "occupancy-fallback"
            and args.color_source_fallback != args.source
        ):
            cases, fallback_source = enrich_case_colors(
                cases, args.color_source_fallback
            )
            if fallback_source != "occupancy-fallback":
                color_source = fallback_source
    else:
        color_source = "disabled"
    summary["color_source"] = color_source

    args.save_summary.parent.mkdir(parents=True, exist_ok=True)
    args.save_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Summary      : {args.save_summary}")

    if args.scan_only:
        return
    if not cases:
        raise SystemExit("No cases matched the requested filter")

    viewer = Inspector(
        cases,
        width=args.width,
        height=args.height,
        start=args.start,
        screenshot_dir=Path(r"artifacts\tetrio\expert_v0_gui_screenshots"),
        speed=args.speed,
    )
    viewer.run()


if __name__ == "__main__":
    main()
