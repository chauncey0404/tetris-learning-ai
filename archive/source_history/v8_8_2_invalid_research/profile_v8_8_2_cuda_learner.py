from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from ai.observable_q_network import ObservableSafeQNetwork
from v8_7_scale_invariant_policy import normalized_margin_actions_torch
from v8_8_1_packed_replay import V881PackedReplayBuffer


def load_model_state(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["model_state_dict"]


def build_synthetic_replay(*, capacity: int, device, seed: int):
    replay = V881PackedReplayBuffer(
        capacity=capacity,
        device=device,
        seed=seed,
    )
    with torch.no_grad():
        replay.data.normal_(0.0, 0.25)

        # Make the mask valid and terminal flag well behaved.
        lo, hi = replay.layout.offsets["next_mask"]
        replay.data[:, lo:hi].fill_(1.0)

        lo, hi = replay.layout.offsets["done"]
        replay.data[:, lo:hi].bernoulli_(0.01)

        # Keep reward scale realistic.
        lo, hi = replay.layout.offsets["reward"]
        replay.data[:, lo:hi].uniform_(-0.05, 1.0)

        lo, hi = replay.layout.offsets["next_rewards"]
        replay.data[:, lo:hi].uniform_(-0.05, 1.0)

        # Teacher scores are divided by 1000 by the network.
        lo, hi = replay.layout.offsets["teacher_score"]
        replay.data[:, lo:hi].normal_(0.0, 500.0)

        lo, hi = replay.layout.offsets["next_teacher_scores"]
        replay.data[:, lo:hi].normal_(0.0, 500.0)

        # Ranks 0..3.
        lo, hi = replay.layout.offsets["teacher_rank"]
        replay.data[:, lo:hi].uniform_(0.0, 3.0)
        lo, hi = replay.layout.offsets["next_teacher_ranks"]
        replay.data[:, lo:hi].uniform_(0.0, 3.0)

        replay.size = replay.capacity
        replay.position = 0
    return replay


class NvidiaSmiSampler:
    def __init__(self, interval_ms: int = 100):
        self.interval_ms = int(interval_ms)
        self.samples = []
        self.proc = None
        self.thread = None
        self.stop_flag = threading.Event()
        self.available = False
        self.error = None

    def start(self):
        exe = shutil.which("nvidia-smi")
        if exe is None:
            self.error = "nvidia-smi not found"
            return

        cmd = [
            exe,
            "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        ]

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.error = repr(exc)
            return

        self.available = True

        def reader():
            assert self.proc is not None
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                if self.stop_flag.is_set():
                    break
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) < 6:
                    continue
                try:
                    self.samples.append(
                        {
                            "timestamp": parts[0],
                            "gpu_util": float(parts[1]),
                            "mem_util": float(parts[2]),
                            "mem_used_mb": float(parts[3]),
                            "power_w": float(parts[4]),
                            "sm_clock_mhz": float(parts[5]),
                        }
                    )
                except ValueError:
                    continue

        self.thread = threading.Thread(target=reader, daemon=True)
        self.thread.start()

    def mark(self):
        return len(self.samples)

    def slice_from(self, mark):
        return list(self.samples[int(mark):])

    def stop(self):
        self.stop_flag.set()
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=1.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    @staticmethod
    def summarize(samples):
        if not samples:
            return None
        def stat(key):
            vals = np.asarray([x[key] for x in samples], dtype=np.float64)
            return {
                "mean": float(vals.mean()),
                "max": float(vals.max()),
                "min": float(vals.min()),
            }
        return {
            "count": len(samples),
            "gpu_util": stat("gpu_util"),
            "mem_util": stat("mem_util"),
            "mem_used_mb": stat("mem_used_mb"),
            "power_w": stat("power_w"),
            "sm_clock_mhz": stat("sm_clock_mhz"),
        }


def one_training_step(
    *,
    model,
    target_model,
    optimizer,
    replay,
    batch_size,
    gamma,
    target_gate,
    terminal_penalty,
):
    batch = replay.sample(batch_size)

    state = batch["state"]
    candidate = batch["candidate"]
    reward = batch["reward"]
    teacher_score = batch["teacher_score"]
    teacher_rank = batch["teacher_rank"]
    done = batch["done"]

    next_state = batch["next_state"]
    next_candidates = batch["next_candidates"]
    next_rewards = batch["next_rewards"]
    next_teacher_scores = batch["next_teacher_scores"]
    next_teacher_ranks = batch["next_teacher_ranks"]
    next_mask = batch["next_mask"]

    current_q = model(
        state=state,
        candidates=candidate.unsqueeze(1),
        rewards=reward.unsqueeze(1),
        teacher_scores=teacher_score.unsqueeze(1),
        teacher_ranks=teacher_rank.unsqueeze(1),
    )[:, 0]

    with torch.no_grad():
        online_next_q = model(
            state=next_state,
            candidates=next_candidates,
            rewards=next_rewards,
            teacher_scores=next_teacher_scores,
            teacher_ranks=next_teacher_ranks,
        )
        online_next_q = online_next_q.masked_fill(~next_mask, -1e9)

        next_action, _, _, _ = normalized_margin_actions_torch(
            online_next_q,
            next_mask,
            target_gate,
        )

        target_next_q_all = target_model(
            state=next_state,
            candidates=next_candidates,
            rewards=next_rewards,
            teacher_scores=next_teacher_scores,
            teacher_ranks=next_teacher_ranks,
        )

        rows = torch.arange(
            online_next_q.shape[0],
            device=online_next_q.device,
        )
        selected_next_q = target_next_q_all[rows, next_action]
        has_next = next_mask.any(dim=1)
        bootstrap = (1.0 - done) * has_next.float() * selected_next_q
        learning_reward = reward - terminal_penalty * done
        td_target = learning_reward + gamma * bootstrap

    loss = F.smooth_l1_loss(current_q, td_target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss


def profiled_step_events(
    *,
    model,
    target_model,
    optimizer,
    replay,
    batch_size,
    gamma,
    target_gate,
    terminal_penalty,
):
    names = [
        "replay_sample",
        "current_forward",
        "online_next_forward",
        "next_action_select",
        "target_next_forward",
        "target_build",
        "loss",
        "zero_grad",
        "backward",
        "grad_clip",
        "optimizer_step",
    ]

    boundaries = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(len(names) + 1)
    ]

    boundaries[0].record()
    batch = replay.sample(batch_size)
    boundaries[1].record()

    state = batch["state"]
    candidate = batch["candidate"]
    reward = batch["reward"]
    teacher_score = batch["teacher_score"]
    teacher_rank = batch["teacher_rank"]
    done = batch["done"]

    next_state = batch["next_state"]
    next_candidates = batch["next_candidates"]
    next_rewards = batch["next_rewards"]
    next_teacher_scores = batch["next_teacher_scores"]
    next_teacher_ranks = batch["next_teacher_ranks"]
    next_mask = batch["next_mask"]

    current_q = model(
        state=state,
        candidates=candidate.unsqueeze(1),
        rewards=reward.unsqueeze(1),
        teacher_scores=teacher_score.unsqueeze(1),
        teacher_ranks=teacher_rank.unsqueeze(1),
    )[:, 0]
    boundaries[2].record()

    with torch.no_grad():
        online_next_q = model(
            state=next_state,
            candidates=next_candidates,
            rewards=next_rewards,
            teacher_scores=next_teacher_scores,
            teacher_ranks=next_teacher_ranks,
        )
        online_next_q = online_next_q.masked_fill(~next_mask, -1e9)
    boundaries[3].record()

    with torch.no_grad():
        next_action, _, _, _ = normalized_margin_actions_torch(
            online_next_q,
            next_mask,
            target_gate,
        )
    boundaries[4].record()

    with torch.no_grad():
        target_next_q_all = target_model(
            state=next_state,
            candidates=next_candidates,
            rewards=next_rewards,
            teacher_scores=next_teacher_scores,
            teacher_ranks=next_teacher_ranks,
        )
    boundaries[5].record()

    with torch.no_grad():
        rows = torch.arange(
            online_next_q.shape[0],
            device=online_next_q.device,
        )
        selected_next_q = target_next_q_all[rows, next_action]
        has_next = next_mask.any(dim=1)
        bootstrap = (1.0 - done) * has_next.float() * selected_next_q
        learning_reward = reward - terminal_penalty * done
        td_target = learning_reward + gamma * bootstrap
    boundaries[6].record()

    loss = F.smooth_l1_loss(current_q, td_target)
    boundaries[7].record()

    optimizer.zero_grad(set_to_none=True)
    boundaries[8].record()

    loss.backward()
    boundaries[9].record()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    boundaries[10].record()

    optimizer.step()
    boundaries[11].record()

    return names, boundaries


def event_breakdown(
    *,
    model,
    target_model,
    optimizer,
    replay,
    batch_size,
    gamma,
    target_gate,
    terminal_penalty,
    iterations,
):
    all_rows = []

    torch.cuda.synchronize()
    wall_start = time.perf_counter()

    for _ in range(iterations):
        names, events = profiled_step_events(
            model=model,
            target_model=target_model,
            optimizer=optimizer,
            replay=replay,
            batch_size=batch_size,
            gamma=gamma,
            target_gate=target_gate,
            terminal_penalty=terminal_penalty,
        )
        all_rows.append((names, events))

    torch.cuda.synchronize()
    wall_elapsed = time.perf_counter() - wall_start

    stage = {name: [] for name in all_rows[0][0]}
    totals = []

    for names, events in all_rows:
        row_total = 0.0
        for i, name in enumerate(names):
            ms = float(events[i].elapsed_time(events[i + 1]))
            stage[name].append(ms)
            row_total += ms
        totals.append(row_total)

    result = {
        "iterations": iterations,
        "wall_ms_per_gradient": wall_elapsed / iterations * 1000.0,
        "wall_gradients_per_sec": iterations / max(wall_elapsed, 1e-12),
        "event_window_ms_per_gradient": float(np.mean(totals)),
        "stages": {},
    }

    for name, values in stage.items():
        arr = np.asarray(values, dtype=np.float64)
        result["stages"][name] = {
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
        }

    return result


def burst_benchmark(
    *,
    model_state,
    replay,
    device,
    batch_size,
    gamma,
    target_gate,
    terminal_penalty,
    burst_sizes,
    total_gradients,
    sampler,
):
    results = []

    for burst in burst_sizes:
        model = ObservableSafeQNetwork().to(device)
        model.load_state_dict(model_state)
        model.train()

        target = ObservableSafeQNetwork().to(device)
        target.load_state_dict(model_state)
        target.eval()

        # lr=0 keeps parameters fixed while preserving AdamW launch/work shape.
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0,
            weight_decay=1e-4,
        )

        for _ in range(8):
            one_training_step(
                model=model,
                target_model=target,
                optimizer=optimizer,
                replay=replay,
                batch_size=batch_size,
                gamma=gamma,
                target_gate=target_gate,
                terminal_penalty=terminal_penalty,
            )

        torch.cuda.synchronize()
        mark = sampler.mark() if sampler is not None else 0
        start = time.perf_counter()

        completed = 0
        while completed < total_gradients:
            n = min(burst, total_gradients - completed)
            for _ in range(n):
                one_training_step(
                    model=model,
                    target_model=target,
                    optimizer=optimizer,
                    replay=replay,
                    batch_size=batch_size,
                    gamma=gamma,
                    target_gate=target_gate,
                    terminal_penalty=terminal_penalty,
                )
            # Production V8.8.2 would synchronize only for low-frequency
            # diagnostics, not after each burst. Do not add a sync here.
            completed += n

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        time.sleep(0.15)

        samples = (
            sampler.slice_from(mark)
            if sampler is not None
            else []
        )
        nvml = NvidiaSmiSampler.summarize(samples)

        result = {
            "burst": int(burst),
            "gradients": int(total_gradients),
            "wall_seconds": float(elapsed),
            "gradients_per_sec": float(total_gradients / elapsed),
            "samples_per_sec": float(total_gradients * batch_size / elapsed),
            "nvidia_smi": nvml,
        }
        results.append(result)

        util_text = ""
        if nvml is not None:
            util_text = (
                f" nvsmi(mean/max)="
                f"{nvml['gpu_util']['mean']:.1f}/"
                f"{nvml['gpu_util']['max']:.1f}%"
            )

        print(
            f"burst={burst:>2} "
            f"grad/s={result['gradients_per_sec']:8.2f} "
            f"samples/s={result['samples_per_sec']:10.0f}"
            f"{util_text}"
        )

        del model, target, optimizer
        torch.cuda.empty_cache()

    return results


def main():
    ap = argparse.ArgumentParser(
        description=(
            "V8.8.2 diagnostic: exact CUDA learner profiling + burst workload "
            "without modifying or saving a training checkpoint."
        )
    )
    ap.add_argument(
        "--checkpoint",
        default="models/v8_8_1_longtraj_gpu_replay_td_200k.pt",
    )
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--replay-size", type=int, default=50000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--target-gate", type=float, default=0.600)
    ap.add_argument("--terminal-penalty", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--event-iters", type=int, default=40)
    ap.add_argument("--profiler-iters", type=int, default=12)
    ap.add_argument("--burst-sizes", default="1,2,4,8,16")
    ap.add_argument("--burst-total-gradients", type=int, default=256)
    ap.add_argument("--nvsmi-interval-ms", type=int, default=100)
    ap.add_argument(
        "--report",
        default="data/v8_8_2_cuda_profile_report.json",
    )
    ap.add_argument(
        "--trace",
        default="data/v8_8_2_cuda_trace.json",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for V8.8.2 profiler.")

    device = torch.device("cuda")
    print("=" * 80)
    print("V8.8.2 CUDA LEARNER PROFILER + BURST BENCHMARK")
    print("=" * 80)
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Checkpoint:", args.checkpoint)
    print("Batch:", args.batch_size)
    print("Synthetic packed replay:", args.replay_size)
    print()

    if not Path(args.checkpoint).exists():
        fallback = Path("models/v8_8_jax_vectorized_td_150k.pt")
        if fallback.exists():
            print(
                "Requested checkpoint missing; profiling formal 150K Champion instead:",
                str(fallback),
            )
            args.checkpoint = str(fallback)
        else:
            raise FileNotFoundError(args.checkpoint)

    model_state = load_model_state(args.checkpoint)

    replay = build_synthetic_replay(
        capacity=args.replay_size,
        device=device,
        seed=882001,
    )

    model = ObservableSafeQNetwork().to(device)
    model.load_state_dict(model_state)
    model.train()

    target = ObservableSafeQNetwork().to(device)
    target.load_state_dict(model_state)
    target.eval()

    # lr=0: benchmark is non-destructive and cannot drift weights.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,
        weight_decay=1e-4,
    )

    for _ in range(args.warmup):
        one_training_step(
            model=model,
            target_model=target,
            optimizer=optimizer,
            replay=replay,
            batch_size=args.batch_size,
            gamma=args.gamma,
            target_gate=args.target_gate,
            terminal_penalty=args.terminal_penalty,
        )
    torch.cuda.synchronize()

    sampler = NvidiaSmiSampler(args.nvsmi_interval_ms)
    sampler.start()
    time.sleep(0.2)

    print("=" * 80)
    print("CUDA EVENT BREAKDOWN")
    print("=" * 80)

    event_mark = sampler.mark()
    breakdown = event_breakdown(
        model=model,
        target_model=target,
        optimizer=optimizer,
        replay=replay,
        batch_size=args.batch_size,
        gamma=args.gamma,
        target_gate=args.target_gate,
        terminal_penalty=args.terminal_penalty,
        iterations=args.event_iters,
    )
    time.sleep(0.15)
    event_nvml = NvidiaSmiSampler.summarize(
        sampler.slice_from(event_mark)
    )

    print(
        f"wall        : {breakdown['wall_ms_per_gradient']:.3f} ms/grad "
        f"({breakdown['wall_gradients_per_sec']:.2f} grad/s)"
    )
    print(
        f"CUDA window : {breakdown['event_window_ms_per_gradient']:.3f} ms/grad"
    )
    if event_nvml is not None:
        print(
            "nvidia-smi  : "
            f"mean/max GPU util "
            f"{event_nvml['gpu_util']['mean']:.1f}%/"
            f"{event_nvml['gpu_util']['max']:.1f}% "
            f"power mean/max "
            f"{event_nvml['power_w']['mean']:.1f}/"
            f"{event_nvml['power_w']['max']:.1f} W"
        )
    else:
        print("nvidia-smi  : sampler unavailable:", sampler.error)

    print()
    for name, stat in breakdown["stages"].items():
        pct = (
            stat["mean_ms"]
            / max(breakdown["event_window_ms_per_gradient"], 1e-12)
            * 100.0
        )
        print(
            f"{name:<22} "
            f"mean={stat['mean_ms']:7.3f}ms "
            f"p95={stat['p95_ms']:7.3f}ms "
            f"share={pct:5.1f}%"
        )

    print()
    print("=" * 80)
    print("TORCH.PROFILER — ACTUAL CUDA KERNEL/OP TIME")
    print("=" * 80)

    trace_path = Path(args.trace)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(args.profiler_iters):
            one_training_step(
                model=model,
                target_model=target,
                optimizer=optimizer,
                replay=replay,
                batch_size=args.batch_size,
                gamma=args.gamma,
                target_gate=args.target_gate,
                terminal_penalty=args.terminal_penalty,
            )
    torch.cuda.synchronize()

    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=30,
        )
    )
    prof.export_chrome_trace(str(trace_path))
    print("Chrome trace:", trace_path)

    print()
    print("=" * 80)
    print("BURST WORKLOAD BENCHMARK")
    print("=" * 80)
    print(
        "Same 8192 replay batch and same optimizer step semantics; "
        "only host scheduling groups consecutive gradients before returning "
        "to rollout handling."
    )

    burst_sizes = [
        int(x.strip())
        for x in args.burst_sizes.split(",")
        if x.strip()
    ]

    # Free the profiler model before building each independent burst case.
    del model, target, optimizer
    torch.cuda.empty_cache()

    bursts = burst_benchmark(
        model_state=model_state,
        replay=replay,
        device=device,
        batch_size=args.batch_size,
        gamma=args.gamma,
        target_gate=args.target_gate,
        terminal_penalty=args.terminal_penalty,
        burst_sizes=burst_sizes,
        total_gradients=args.burst_total_gradients,
        sampler=sampler if sampler.available else None,
    )

    sampler.stop()

    best = max(bursts, key=lambda x: x["gradients_per_sec"])
    baseline = next(
        (x for x in bursts if x["burst"] == 1),
        bursts[0],
    )

    report = {
        "version": "V8_8_2_CUDA_PROFILE_AND_BURST_DIAGNOSTIC",
        "checkpoint": args.checkpoint,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "batch_size": args.batch_size,
        "replay_size": args.replay_size,
        "event_breakdown": breakdown,
        "event_nvidia_smi": event_nvml,
        "bursts": bursts,
        "best_burst": best["burst"],
        "best_gradients_per_sec": best["gradients_per_sec"],
        "burst_speedup_vs_1": (
            best["gradients_per_sec"]
            / max(baseline["gradients_per_sec"], 1e-12)
        ),
        "trace": str(trace_path),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("V8.8.2 DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print("Best burst:", best["burst"])
    print(
        "Best grad/s:",
        f"{best['gradients_per_sec']:.2f}",
    )
    print(
        "Speedup vs burst=1:",
        f"{report['burst_speedup_vs_1']:.3f}x",
    )
    print("Report:", report_path)
    print("Trace :", trace_path)
    print()
    print("V8.8.2 CUDA PROFILER: PASS")


if __name__ == "__main__":
    main()
