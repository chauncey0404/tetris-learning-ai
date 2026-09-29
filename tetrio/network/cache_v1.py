from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from tetrio.network.cache import load_shard


@dataclass(frozen=True)
class ExpertV1CompactBatch:
    state_board_packed: np.ndarray
    state_active: np.ndarray
    state_hold: np.ndarray
    state_preview: np.ndarray

    candidate_board_packed: np.ndarray
    candidate_piece: np.ndarray
    candidate_rotation: np.ndarray
    candidate_x: np.ndarray
    candidate_y: np.ndarray
    candidate_use_hold: np.ndarray
    candidate_lines: np.ndarray
    candidate_holes: np.ndarray
    candidate_owner: np.ndarray
    candidate_local: np.ndarray
    candidate_counts: np.ndarray
    max_candidates: int

    expert_index: np.ndarray
    expert_use_hold: np.ndarray
    holes_before: np.ndarray


def sidecar_paths(sidecar_dir: str | Path) -> list[Path]:
    paths = sorted(Path(sidecar_dir).glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No Expert-v1 counterfactual shards found in {sidecar_dir}"
        )
    return paths


def _gather(
    offsets: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts = offsets[indices]
    counts = offsets[indices + 1] - starts
    counts = counts.astype(np.int64, copy=False)
    total = int(counts.sum())
    owner = np.repeat(np.arange(len(indices), dtype=np.int32), counts)
    prefix = np.cumsum(counts, dtype=np.int64) - counts
    local = (
        np.arange(total, dtype=np.int64)
        - np.repeat(prefix, counts)
    )
    gather = np.repeat(starts, counts) + local
    return (
        gather.astype(np.int64, copy=False),
        owner,
        local.astype(np.int32, copy=False),
        counts.astype(np.int32, copy=False),
    )


def _cat(v0, cf, key, selected_gather, cf_gather):
    return np.ascontiguousarray(
        np.concatenate(
            (
                v0[key][selected_gather],
                cf[key][cf_gather],
            ),
            axis=0,
        )
    )


def compact_batches_from_pair(
    *,
    v0: dict[str, np.ndarray],
    cf: dict[str, np.ndarray],
    batch_size: int,
    rng: np.random.Generator | None,
) -> Iterator[ExpertV1CompactBatch]:
    n = int(cf["holes_before"].shape[0])
    if int(v0["expert_index"].shape[0]) < n:
        raise RuntimeError("V0 shard shorter than V1 sidecar shard")

    if not np.array_equal(
        np.asarray(v0["game_id"][:n], dtype=np.int64),
        np.asarray(cf["game_id"], dtype=np.int64),
    ) or not np.array_equal(
        np.asarray(v0["subframe"][:n], dtype=np.int64),
        np.asarray(cf["subframe"], dtype=np.int64),
    ):
        raise RuntimeError("V0/V1 sidecar row identity mismatch")

    order = np.arange(n, dtype=np.int64)
    if rng is not None:
        rng.shuffle(order)

    sel_offsets = v0["candidate_offsets"].astype(np.int64, copy=False)
    cf_offsets = cf["candidate_offsets"].astype(np.int64, copy=False)

    for start in range(0, n, int(batch_size)):
        indices = order[start:start + int(batch_size)]

        sg, so, sl, sc = _gather(sel_offsets, indices)
        cg, co, cl, cc = _gather(cf_offsets, indices)

        # The expert-selected branch always occupies local slots [0, sc).
        # Counterfactual candidates are appended at local slots [sc, sc+cc).
        cf_local = cl + sc[co]
        owner = np.concatenate((so, co)).astype(np.int32, copy=False)
        local = np.concatenate((sl, cf_local)).astype(np.int32, copy=False)
        counts = (sc + cc).astype(np.int32, copy=False)

        yield ExpertV1CompactBatch(
            state_board_packed=np.ascontiguousarray(
                v0["state_board_packed"][indices]
            ),
            state_active=np.ascontiguousarray(v0["state_active"][indices]),
            state_hold=np.ascontiguousarray(v0["state_hold"][indices]),
            state_preview=np.ascontiguousarray(v0["state_preview"][indices]),

            candidate_board_packed=_cat(
                v0, cf, "candidate_board_packed", sg, cg
            ),
            candidate_piece=_cat(v0, cf, "candidate_piece", sg, cg),
            candidate_rotation=_cat(
                v0, cf, "candidate_rotation", sg, cg
            ),
            candidate_x=_cat(v0, cf, "candidate_x", sg, cg),
            candidate_y=_cat(v0, cf, "candidate_y", sg, cg),
            candidate_use_hold=_cat(
                v0, cf, "candidate_use_hold", sg, cg
            ),
            candidate_lines=_cat(v0, cf, "candidate_lines", sg, cg),
            candidate_holes=np.ascontiguousarray(
                np.concatenate(
                    (
                        cf["selected_candidate_holes"][sg],
                        cf["candidate_holes"][cg],
                    )
                )
            ),
            candidate_owner=owner,
            candidate_local=local,
            candidate_counts=counts,
            max_candidates=int(counts.max()),

            # Expert target remains unchanged because its original branch is
            # deliberately placed first for every row.
            expert_index=np.ascontiguousarray(
                v0["expert_index"][indices].astype(np.int64, copy=False)
            ),
            expert_use_hold=np.ascontiguousarray(
                v0["use_hold"][indices].astype(np.uint8, copy=False)
            ),
            holes_before=np.ascontiguousarray(
                cf["holes_before"][indices].astype(np.uint8, copy=False)
            ),
        )


def load_pair(v0_cache_dir: str | Path, sidecar_path: str | Path):
    sidecar_path = Path(sidecar_path)
    v0_path = Path(v0_cache_dir) / sidecar_path.name
    if not v0_path.is_file():
        raise FileNotFoundError(
            f"Matching V0 shard not found for {sidecar_path.name}: {v0_path}"
        )
    return load_shard(v0_path), load_shard(sidecar_path)
