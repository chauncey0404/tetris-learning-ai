from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.compare_model_calibration_matched_states import (
    DEFAULT_CHALLENGER,
    DEFAULT_CHAMPION,
    HeuristicTeacherV2,
    choose_device,
    compact_candidate_arrays,
    deduplicate_same_state,
    load_policy,
    parse_float_list,
    parse_seed_spec,
    quality,
    replay_selected_state,
    run_branch,
    scan_source_trajectory,
    select_stratified_balanced,
)

PAIR_INDEX = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)
PROTECTED_FINAL_SEEDS = set(range(6, 21))


def branch_pair_targets(by_rank: dict) -> np.ndarray:
    targets = np.zeros(6, dtype=np.int8)
    for p, (i, j) in enumerate(PAIR_INDEX):
        ri = by_rank.get(i + 1)
        rj = by_rank.get(j + 1)
        if ri is None or rj is None:
            continue
        qi = quality(ri)
        qj = quality(rj)
        if qi > qj:
            targets[p] = 1
        elif qi < qj:
            targets[p] = -1
    return targets


def split_by_seed(
    seeds: list[int],
    *,
    validation_fraction: float,
    sample_seed: int,
) -> tuple[set[int], set[int]]:
    unique = sorted(set(int(s) for s in seeds))
    if len(unique) < 2:
        raise RuntimeError(
            "Need at least two distinct seeds for train/validation split."
        )

    rng = random.Random(int(sample_seed))
    shuffled = list(unique)
    rng.shuffle(shuffled)

    n_val = max(
        1,
        min(
            len(shuffled) - 1,
            int(round(len(shuffled) * float(validation_fraction))),
        ),
    )
    val = set(shuffled[:n_val])
    train = set(shuffled[n_val:])
    return train, val


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Build a small offline V8.8.7 pairwise-ranking corpus from "
            "already development-consumed matched states."
        )
    )
    p.add_argument("--champion", default=DEFAULT_CHAMPION)
    p.add_argument("--challenger", default=DEFAULT_CHALLENGER)
    p.add_argument("--seeds", default="4761-4775")
    p.add_argument("--max-pieces", type=int, default=5000)
    p.add_argument("--top-k", type=int, default=4)

    p.add_argument("--confidence-min", type=float, default=0.600)
    p.add_argument(
        "--confidence-bins",
        default="0.60,0.70,0.80,0.90,0.99,1.01",
    )
    p.add_argument(
        "--events-per-bin",
        type=int,
        default=8,
        help="Default 8 x 5 bins = ~40 pilot states.",
    )
    p.add_argument("--max-events-per-seed", type=int, default=5)
    p.add_argument("--sample-seed", type=int, default=20260831)

    p.add_argument(
        "--horizon",
        type=int,
        default=100,
        help=(
            "Pilot uses 100-piece fixed-policy counterfactual horizon. "
            "The established 250-piece scanner remains the evaluation tool."
        ),
    )
    p.add_argument(
        "--rollout-model",
        default=DEFAULT_CHAMPION,
        help=(
            "One fixed continuation policy for every branch. "
            "Default formal 31.2M Champion."
        ),
    )
    p.add_argument(
        "--rollout-label",
        default="V8.8.6 31.2M Fixed Ranking-Corpus Rollout",
    )
    p.add_argument(
        "--validation-fraction",
        type=float,
        default=0.25,
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    p.add_argument("--gate", type=float, default=0.600)
    p.add_argument(
        "--output",
        default="data/v8_8_7_ranking_corpus_4761_4775.npz",
    )
    p.add_argument(
        "--metadata-output",
        default="data/v8_8_7_ranking_corpus_4761_4775.json",
    )
    p.add_argument("--allow-protected-seeds", action="store_true")
    args = p.parse_args()

    if args.top_k != 4:
        raise SystemExit("V8.8.7 pilot corpus requires --top-k 4.")
    if args.events_per_bin < 1:
        raise SystemExit("--events-per-bin must be >= 1.")
    if args.max_events_per_seed < 1:
        raise SystemExit("--max-events-per-seed must be >= 1.")
    if args.horizon < 1:
        raise SystemExit("--horizon must be >= 1.")
    if not 0.0 < args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be in (0,1).")

    seeds = parse_seed_spec(args.seeds)
    protected = [s for s in seeds if s in PROTECTED_FINAL_SEEDS]
    if protected and not args.allow_protected_seeds:
        raise SystemExit(
            f"Protected final-report seeds blocked: {protected}"
        )

    edges = parse_float_list(args.confidence_bins)
    device = choose_device(args.device)
    teacher = HeuristicTeacherV2()

    champion = load_policy(
        args.champion,
        label="V8.8.6 31.2M Champion",
        device=device,
        gate_override=args.gate,
        semantics_override="normalized_q_margin",
    )
    challenger = load_policy(
        args.challenger,
        label="V8.8.6 41.2M Control",
        device=device,
        gate_override=args.gate,
        semantics_override="normalized_q_margin",
    )
    rollout = load_policy(
        args.rollout_model,
        label=args.rollout_label,
        device=device,
        gate_override=args.gate,
        semantics_override="normalized_q_margin",
    )

    print("=" * 108)
    print("V8.8.7 PILOT RANKING CORPUS BUILDER")
    print("=" * 108)
    print("Seeds         :", args.seeds, "(REUSED DEVELOPMENT-CONSUMED)")
    print("State sources : 31.2M + 41.2M")
    print("Events/bin    :", args.events_per_bin)
    print("Horizon       :", args.horizon)
    print("Fixed rollout :", rollout.label)
    print("Device        :", device)
    print()
    print(
        "Counterfactual VALUE is used only to derive pairwise ordering labels. "
        "It is NOT written into TD rewards."
    )

    started = time.perf_counter()

    all_events = []
    event_id = 0
    for source_name, source_policy in (
        ("champion", champion),
        ("challenger", challenger),
    ):
        rows, event_id = scan_source_trajectory(
            source_name=source_name,
            source_policy=source_policy,
            champion=champion,
            challenger=challenger,
            teacher=teacher,
            device=device,
            seeds=seeds,
            max_pieces=args.max_pieces,
            top_k=4,
            confidence_min=args.confidence_min,
            edges=edges,
            start_event_id=event_id,
        )
        all_events.extend(rows)

    all_events = deduplicate_same_state(all_events)
    selected = select_stratified_balanced(
        all_events,
        edges=edges,
        events_per_bin=args.events_per_bin,
        max_events_per_seed=args.max_events_per_seed,
        sample_seed=args.sample_seed,
    )
    if len(selected) < 10:
        raise RuntimeError(
            f"Only {len(selected)} states selected; pilot requires >= 10."
        )

    print()
    print("Eligible unique states:", len(all_events))
    print("Selected pilot states :", len(selected))

    source_lookup = {
        "champion": champion,
        "challenger": challenger,
    }

    states = []
    candidates_all = []
    rewards_all = []
    scores_all = []
    ranks_all = []
    masks_all = []
    pair_targets_all = []

    seed_values = []
    piece_values = []
    source_values = []
    hash_values = []

    total_jobs = len(selected) * 4
    job = 0

    for ei, event in enumerate(selected, 1):
        source_policy = source_lookup[event.source]
        session = replay_selected_state(
            event=event,
            source_policy=source_policy,
            teacher=teacher,
            device=device,
            top_k=4,
            max_horizon=args.horizon,
        )
        try:
            from tools.compare_model_calibration_matched_states import (
                inspect_successors,
            )

            successors, _, _ = inspect_successors(session)
            candidate_features, rewards, scores, ranks = (
                compact_candidate_arrays(successors)
            )

            k = len(successors)
            if k < 2:
                raise RuntimeError(
                    f"Event {event.event_id} has only {k} candidates."
                )

            state_arr = np.asarray(
                session.state_features,
                dtype=np.float32,
            ).copy()
            candidate_pad = np.zeros((4, 215), dtype=np.float32)
            reward_pad = np.zeros(4, dtype=np.float32)
            score_pad = np.zeros(4, dtype=np.float32)
            rank_pad = np.zeros(4, dtype=np.float32)
            mask_pad = np.zeros(4, dtype=np.bool_)

            candidate_pad[:k] = np.asarray(
                candidate_features,
                dtype=np.float32,
            )
            reward_pad[:k] = np.asarray(rewards, dtype=np.float32)
            score_pad[:k] = np.asarray(scores, dtype=np.float32)
            rank_pad[:k] = np.asarray(ranks, dtype=np.float32)
            mask_pad[:k] = True
        finally:
            session.close()

        by_rank = {}
        print(
            f"[state {ei:>2}/{len(selected)}] "
            f"{event.source} seed={event.seed} P{event.piece} "
            f"hash={event.state_hash}"
        )

        for branch_rank in range(1, k + 1):
            job += 1
            t0 = time.perf_counter()
            row = run_branch(
                event=event,
                branch_rank=branch_rank,
                source_policy=source_policy,
                rollout_policy=rollout,
                teacher=teacher,
                device=device,
                top_k=4,
                horizon=args.horizon,
            )
            by_rank[branch_rank] = row
            print(
                f"  branch#{branch_rank} "
                f"V={row.cumulative_value_from_branch:>7} "
                f"T={row.cumulative_tetrises_from_branch:>3} "
                f"maxH={row.max_height_from_branch:>2} "
                f"holes={row.max_holes_from_branch:>2} "
                f"survive={row.survived} "
                f"| {time.perf_counter()-t0:.1f}s "
                f"[{job}/{total_jobs}]"
            )

        targets = branch_pair_targets(by_rank)
        if not np.any(targets):
            print("  NOTE: all branch pairs tied; retained with zero pair targets.")

        states.append(state_arr)
        candidates_all.append(candidate_pad)
        rewards_all.append(reward_pad)
        scores_all.append(score_pad)
        ranks_all.append(rank_pad)
        masks_all.append(mask_pad)
        pair_targets_all.append(targets)

        seed_values.append(event.seed)
        piece_values.append(event.piece)
        source_values.append(
            0 if event.source == "champion" else 1
        )
        hash_values.append(event.state_hash)

    train_seeds, val_seeds = split_by_seed(
        seed_values,
        validation_fraction=args.validation_fraction,
        sample_seed=args.sample_seed + 1,
    )
    split = np.asarray(
        [
            0 if int(seed) in train_seeds else 1
            for seed in seed_values
        ],
        dtype=np.int8,
    )

    if not np.any(split == 0) or not np.any(split == 1):
        raise RuntimeError("Train/validation state split failed.")

    state_np = np.stack(states).astype(np.float32)
    candidates_np = np.stack(candidates_all).astype(np.float32)
    rewards_np = np.stack(rewards_all).astype(np.float32)
    scores_np = np.stack(scores_all).astype(np.float32)
    ranks_np = np.stack(ranks_all).astype(np.float32)
    masks_np = np.stack(masks_all).astype(np.bool_)
    pair_targets_np = np.stack(pair_targets_all).astype(np.int8)

    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        state=state_np,
        candidates=candidates_np,
        rewards=rewards_np,
        teacher_scores=scores_np,
        teacher_ranks=ranks_np,
        candidate_mask=masks_np,
        pair_targets=pair_targets_np,
        split=split,
        seed=np.asarray(seed_values, dtype=np.int64),
        piece=np.asarray(piece_values, dtype=np.int64),
        source=np.asarray(source_values, dtype=np.int8),
        state_hash=np.asarray(hash_values, dtype="U20"),
    )

    elapsed = time.perf_counter() - started
    valid_pairs = int(np.count_nonzero(pair_targets_np))

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "V8.8.7 PILOT OFFLINE RANKING CORPUS",
        "champion": str(champion.path or args.champion),
        "challenger": str(challenger.path or args.challenger),
        "seed_spec": args.seeds,
        "seeds": seeds,
        "state_sources": ["champion", "challenger"],
        "fixed_rollout": str(rollout.path or args.rollout_model),
        "horizon": int(args.horizon),
        "selected_states": int(len(selected)),
        "valid_pair_labels": valid_pairs,
        "train_states": int(np.count_nonzero(split == 0)),
        "validation_states": int(np.count_nonzero(split == 1)),
        "train_seeds": sorted(train_seeds),
        "validation_seeds": sorted(val_seeds),
        "confidence_min": args.confidence_min,
        "confidence_edges": edges,
        "events_per_bin": args.events_per_bin,
        "max_events_per_seed": args.max_events_per_seed,
        "elapsed_seconds": elapsed,
        "important": (
            "Counterfactual branch values are not TD rewards. "
            "They are converted only into -1/0/+1 pairwise ordering labels."
        ),
    }

    metadata_output = PROJECT_ROOT / args.metadata_output
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 108)
    print("V8.8.7 PILOT RANKING CORPUS COMPLETE")
    print("=" * 108)
    print("States        :", len(selected))
    print("Pair labels   :", valid_pairs)
    print("Train states  :", int(np.count_nonzero(split == 0)))
    print("Val states    :", int(np.count_nonzero(split == 1)))
    print("Train seeds   :", sorted(train_seeds))
    print("Val seeds     :", sorted(val_seeds))
    print("Elapsed       :", f"{elapsed:.2f}s")
    print("Corpus        :", output)
    print("Metadata      :", metadata_output)
    print("=" * 108)


if __name__ == "__main__":
    main()
