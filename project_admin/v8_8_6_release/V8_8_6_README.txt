V8.8.6 — 3 PRODUCERS + CPU AFFINITY + SHARED ACTOR WEIGHTS
==========================================================

Measured hardware probes:

3 producers x 256 envs:
  9468.09 transitions/s
  235.19 gradients/s
  weight lag mean/max 1449 / 3840

4 producers x 256 envs:
  10165.39 transitions/s
  252.51 gradients/s
  only +7.36% throughput
  weight lag mean/max 4025 / 38144

The fourth producer increased policy staleness far more than throughput.
Production concurrency is therefore fixed at 3 x 256 = 768 streams.

NEW IN V8.8.6
-------------
1) Windows topology-aware CPU affinity
   - detects physical cores + EfficiencyClass;
   - keeps SMT siblings together;
   - balances P/E cores across 3 JAX producers;
   - reserves low-efficiency logical CPUs for the learner/main process;
   - has a balanced logical-CPU fallback.

2) Shared-memory actor weight bank
   - one CPU model snapshot is shared by all producers;
   - eliminates repeated full state_dict queue serialization;
   - version + lock prevents partial actor-weight reads.

3) Large production run
   input:
     models/v8_8_3_dynamicmask_cuda_graph_td_1200k.pt

   new transitions:
     30,000,000

   target total:
     31,200,000

   producers:
     3

   envs / producer:
     256

   total streams:
     768

   nominal transitions / initial stream:
     39,062.5

   replay capacity:
     750,000 (~4.42 GiB raw packed replay)

   batch:
     8192

   gradients:
     745,200

   replay samples:
     6,104,678,400

   replay samples / new transition:
     203.48928 (unchanged)

   checkpoint every:
     3,000,000 new transitions

Expected total-transition checkpoints:
  4.2M, 7.2M, 10.2M, 13.2M, 16.2M,
  19.2M, 22.2M, 25.2M, 28.2M, 31.2M

RUN ORDER
---------
1)
python -m py_compile v8_8_3_cuda_graph_train_common.py v8_8_2_graph_safe_policy.py test_v8_8_6_affinity_sharedweight_preflight.py train_v8_8_6_affinity_sharedweight_1200k_to_31200k.py

2)
python test_v8_8_6_affinity_sharedweight_preflight.py

Must end:
  V8.8.6 PREFLIGHT: PASS

3)
python train_v8_8_6_affinity_sharedweight_1200k_to_31200k.py

SAFE STOP
---------
Ctrl+C saves an INTERRUPTED checkpoint before producer shutdown.

PERFORMANCE BASELINE
--------------------
Clean 3-producer probe:
  9468 transitions/s
  235 gradients/s

Affinity/shared weights should improve scheduling consistency, actor sync
overhead, policy freshness and/or throughput stability. A huge raw TPS increase
is not required; preserving ~9.5K+ TPS with better lag is already useful.

MODEL STATUS
------------
Formal Champion remains V8.8 150K normalized gate 0.600.
V8.8.3 1.2M is the clean challenger training base.
V8.8.2 frozen-next-mask checkpoints remain excluded.
Hardware probe checkpoints are not promoted automatically.
