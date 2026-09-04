import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT_MARKERS = [
    "teacher.py",
    "tetris_core.py",
    "tetris_placement.py",
    "gym_executor.py",
]

ARCHIVE_ROOT = Path("_archive") / "pre_v8_5_champion_cleanup_20260822"

# Historical experiment scripts that are no longer part of the active V8.5 path.
ARCHIVE_ROOT_FILES = [
    "calibrate_v8_4_20k_gate_deep_lower_cached.py",
    "calibrate_v8_4_20k_gate_fast.py",
    "calibrate_v8_4_20k_gate_final_fine_adaptive.py",
    "calibrate_v8_4_20k_gate_lower_cached.py",
    "calibrate_v8_5_30k_gate_final_fine_screened.py",
    "calibrate_v8_5_30k_gate_screened_fast.py",
    "cleanup_v5_v7.cmd",
    "compare_v8_10k_vs_20k_fresh.py",
    "compare_v8_20k_vs_30k_fresh.py",
    "compare_v8_5k_vs_10k_fresh.py",
    "diagnose_v8_q_margins.py",
    "evaluate_v8_4_observable.py",
    "evaluate_v8_conservative.py",
    "evaluate_v8_policy.py",
    "qualify_v8_4_10k_vs_20k_fresh_2701_2720_fast.py",
    "qualify_v8_5_30k_champion_fresh_2801_2820_fast.py",
    "sweep_v8_4_gate_fast.py",
    "sweep_v8_4_gate_fine_cached.py",
    "sweep_v8_4_gate_smart_upper.py",
    "sweep_v8_4_gate_upper_cached.py",
    "train_v8_1_conservative_td.py",
    "train_v8_2_parallel_td.py",
    "train_v8_3_fresh_parallel_td.py",
    "train_v8_4_observable_parallel_td.py",
    "train_v8_4_observable_resume_20k_fast.py",
    "train_v8_td_smoke.py",
    "validate_v8_4_gate_0275_fresh_fast.py",
    "validate_v8_gate_fresh.py",
    "V8_4_README.txt",
    "v8_replay.py",
]

# Old network implementation used by pre-observable V8.x experiments.
ARCHIVE_AI_FILES = [
    "td_q_network.py",
]

# Historical caches/results. Keep V8.5 final calibration + fresh qualification active.
ARCHIVE_DATA_FILES = [
    "v8_4_10k_vs_20k_fresh_2701_2720.json",
    "v8_4_20k_gate_calibration_2501_2520.json",
    "v8_4_gate_0275_fresh_2601_2620.json",
    "v8_4_gate_cache_2501_2520.json",
    "v8_4_teacher_gap_trace_2501_2520.json",
]

# Historical checkpoints. The active baseline and Champion are NOT here.
ARCHIVE_MODEL_FILES = [
    "v8_4_observable_safe_td_20k.pt",
]

# These are obsolete enough to delete only when --purge-obsolete is explicitly given.
PURGE_DATA_FILES = [
    "dagger_v2_1_round1_20k.npz",
]

PURGE_MODEL_FILES = [
    "v8_td_smoke.pt",
    "v8_td_5k.pt",
    "v8_1_td_10k.pt",
    "v8_2_parallel_td_20k.pt",
    "v8_3_fresh_parallel_td_30k.pt",
]

# Expected active files. This is informational and used for a preflight check.
KEEP_ROOT_FILES = [
    "benchmark_teacher.py",
    "gym_executor.py",
    "teacher.py",
    "test_hold_placements.py",
    "test_state_encoder.py",
    "test_successor_candidates.py",
    "test_v8_4_observable_safety.py",
    "tetris_core.py",
    "tetris_placement.py",
    "train_v8_5_risk_aware_30k_autobatch.py",
    "v8_4_observable.py",
    "v8_4_replay.py",
    "v8_successor.py",
    # If downloaded later, this stays active automatically because it is not in archive lists:
    "benchmark_v8_5_final_permanent_6_20.py",
]

KEEP_AI_FILES = [
    "observable_q_network.py",
    "state_encoder.py",
]

KEEP_MODEL_FILES = [
    "v8_4_observable_safe_td_10k.pt",
    "v8_5_risk_aware_observable_safe_td_30k.pt",
]

KEEP_DATA_FILES = [
    "v8_5_30k_champion_qualification_2801_2820.json",
    "v8_5_30k_gate_calibration_2501_2520.json",
    # Final permanent result, if it exists later, is untouched.
    "v8_5_final_permanent_benchmark_6_20.json",
]


def fail(msg):
    print(f"ERROR: {msg}")
    raise SystemExit(2)


def ensure_project_root(root: Path):
    missing = [name for name in ROOT_MARKERS if not (root / name).exists()]
    if missing:
        fail(
            "Run this from F:\\tetris-learning-ai. "
            f"Missing root markers: {missing}"
        )


def unique_destination(dest: Path) -> Path:
    if not dest.exists():
        return dest

    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    i = 2

    while True:
        candidate = parent / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def archive_file(root: Path, rel: Path, apply: bool):
    src = root / rel
    if not src.exists():
        return False

    dest = root / ARCHIVE_ROOT / rel
    dest = unique_destination(dest)

    print(f"ARCHIVE  {rel}  ->  {dest.relative_to(root)}")

    if apply:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))

    return True


def delete_file(root: Path, rel: Path, apply: bool):
    path = root / rel
    if not path.exists():
        return False

    print(f"DELETE   {rel}")

    if apply:
        path.unlink()

    return True


def remove_generated_caches(root: Path, apply: bool):
    removed_dirs = 0
    removed_files = 0

    # Only project code caches; never descend into .venv or archive.
    excluded_top = {".venv", ARCHIVE_ROOT.parts[0]}

    for current, dirs, files in os.walk(root, topdown=True):
        current_path = Path(current)

        if current_path == root:
            dirs[:] = [d for d in dirs if d not in excluded_top]

        # Prune nested .venv/archive defensively.
        dirs[:] = [
            d for d in dirs
            if d not in {".venv", ARCHIVE_ROOT.parts[0]}
        ]

        if current_path.name == "__pycache__":
            print(f"DELETE   {current_path.relative_to(root)}\\")
            if apply:
                shutil.rmtree(current_path, ignore_errors=False)
            removed_dirs += 1
            dirs[:] = []
            continue

        for filename in files:
            if filename.endswith((".pyc", ".pyo")):
                p = current_path / filename
                print(f"DELETE   {p.relative_to(root)}")
                if apply:
                    p.unlink()
                removed_files += 1

    pytest_cache = root / ".pytest_cache"
    if pytest_cache.exists():
        print("DELETE   .pytest_cache\\")
        if apply:
            shutil.rmtree(pytest_cache)
        removed_dirs += 1

    return removed_dirs, removed_files


def print_keep_status(root: Path):
    print("\nACTIVE KEEP CHECK")
    print("-" * 80)

    checks = [
        ("root", KEEP_ROOT_FILES),
        ("ai", KEEP_AI_FILES),
        ("models", KEEP_MODEL_FILES),
        ("data", KEEP_DATA_FILES),
    ]

    for folder, names in checks:
        for name in names:
            rel = Path(name) if folder == "root" else Path(folder) / name
            state = "KEEP" if (root / rel).exists() else "not present"
            print(f"{state:<11} {rel}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Safe cleanup for F:\\tetris-learning-ai after V8.5-30K "
            "became the observable-safe Champion. Default is DRY RUN."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move/archive files and delete generated caches.",
    )
    parser.add_argument(
        "--purge-obsolete",
        action="store_true",
        help=(
            "Also permanently delete obsolete DAgger/leaky pre-V8.4 models. "
            "Requires --apply."
        ),
    )
    args = parser.parse_args()

    if args.purge_obsolete and not args.apply:
        fail("--purge-obsolete requires --apply")

    root = Path.cwd().resolve()
    ensure_project_root(root)

    print("=" * 80)
    print("TETRIS PROJECT CLEANUP")
    print("=" * 80)
    print("Root:", root)
    print("Mode:", "APPLY" if args.apply else "DRY RUN")
    print("Archive:", ARCHIVE_ROOT)
    print("Touch .venv:", "NO")
    print("Purge obsolete historical assets:", "YES" if args.purge_obsolete else "NO")
    print()

    archive_count = 0
    delete_count = 0

    for name in ARCHIVE_ROOT_FILES:
        archive_count += int(
            archive_file(root, Path(name), args.apply)
        )

    for name in ARCHIVE_AI_FILES:
        archive_count += int(
            archive_file(root, Path("ai") / name, args.apply)
        )

    for name in ARCHIVE_DATA_FILES:
        archive_count += int(
            archive_file(root, Path("data") / name, args.apply)
        )

    for name in ARCHIVE_MODEL_FILES:
        archive_count += int(
            archive_file(root, Path("models") / name, args.apply)
        )

    # The tree report has served its purpose. Delete only on APPLY.
    if (root / "project_tree.txt").exists():
        print("DELETE   project_tree.txt")
        if args.apply:
            (root / "project_tree.txt").unlink()
        delete_count += 1

    cache_dirs, cache_files = remove_generated_caches(root, args.apply)
    delete_count += cache_dirs + cache_files

    if args.purge_obsolete:
        for name in PURGE_DATA_FILES:
            delete_count += int(
                delete_file(root, Path("data") / name, args.apply)
            )

        for name in PURGE_MODEL_FILES:
            delete_count += int(
                delete_file(root, Path("models") / name, args.apply)
            )

    print_keep_status(root)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Archive actions:", archive_count)
    print("Delete/cache actions:", delete_count)

    if not args.apply:
        print("\nDRY RUN ONLY: nothing changed.")
        print("Apply safe cleanup with:")
        print("  python cleanup_tetris_project.py --apply")
        print()
        print("Optional later permanent purge of obsolete historical assets:")
        print("  python cleanup_tetris_project.py --apply --purge-obsolete")
    else:
        print("\nCleanup applied.")
        if not args.purge_obsolete:
            print(
                "Obsolete large models/dataset were intentionally left in place. "
                "Use --purge-obsolete only after you are satisfied with the archive."
            )


if __name__ == "__main__":
    main()
