V8.8.2 PRODUCTION CUDA GRAPH — 200K -> 300K CHALLENGER LINEAGE
================================================================

CURRENT MODEL STATUS
--------------------
Formal Champion remains:
  models/v8_8_jax_vectorized_td_150k.pt
  normalized gate = 0.600

V8.8.1 200K:
  models/v8_8_1_longtraj_gpu_replay_td_200k.pt
  fresh qualification 3201~3220 = FAIL
  It is NOT Champion.

Why 200K is still used as the training warm-start
--------------------------------------------------
A failed challenger may continue as a challenger training lineage. Promotion
failure means "do not deploy/promote it", not "its optimizer/model state may
never be trained further". We do NOT reuse qualification seeds 3201~3220.
The next 250K model will be a new challenger evaluated on new blocks.

The 200K failure was narrow:
  GO       : 0 vs 0
  pieces   : 2000 vs 2000
  R/1000   : 487265.0 vs Champion 488087.5
  value    : 974530 vs Champion 976175

V8.8.2 keeps the successful V8.8.1 data design and changes the learner
execution path to the validated CUDA Graph implementation.

V8.8.2 TRAINING DESIGN
----------------------
Input:
  models/v8_8_1_longtraj_gpu_replay_td_200k.pt

Training:
  +100,000 new transitions
  total 200K -> 300K
  training seed start 13001
  32 natural JAX vector environments
  ~20% risk streams
  no forced segmentation
  normalized behavior gate 0.600
  risk gate 0.400
  DDQN target gate 0.600
  exploration 5%
  packed GPU replay
  batch 8192
  effective 2484 gradients / 20,348,928 replay samples
  ~203.49 replay samples per new transition

CUDA:
  Dynamic no-replacement replay sampling remains outside the graph.
  Fixed-shape DDQN forward/target/loss/backward/clip/AdamW is CUDA Graph.
  Graph capture is non-destructive: warmup/capture state is restored before
  the first counted training gradient.

Checkpoints:
  models/v8_8_2_cuda_graph_longtraj_td_210k.pt
  models/v8_8_2_cuda_graph_longtraj_td_220k.pt
  models/v8_8_2_cuda_graph_longtraj_td_230k.pt
  models/v8_8_2_cuda_graph_longtraj_td_240k.pt
  models/v8_8_2_cuda_graph_longtraj_td_250k.pt
  models/v8_8_2_cuda_graph_longtraj_td_260k.pt
  models/v8_8_2_cuda_graph_longtraj_td_270k.pt
  models/v8_8_2_cuda_graph_longtraj_td_280k.pt
  models/v8_8_2_cuda_graph_longtraj_td_290k.pt
  models/v8_8_2_cuda_graph_longtraj_td_300k.pt

All remain CHALLENGERS until fresh qualification.

REQUIRED ORDER
--------------
1. Copy/extract this bundle into:
     F:\tetris-learning-ai

2. Syntax:
     python -m py_compile v8_8_2_graph_safe_policy.py v8_8_2_cuda_graph_train_common.py test_v8_8_2_production_cuda_graph_gate.py train_v8_8_2_cuda_graph_longtraj_200k_to_300k.py

3. REQUIRED production graph gate:
     python test_v8_8_2_production_cuda_graph_gate.py

   Must end with:
     V8.8.2 PRODUCTION CUDA GRAPH GATE: PASS

   It writes:
     data\v8_8_2_cuda_graph_production_gate_pass.json

   It checks:
     - exact normalized-policy parity on CUDA
     - graph capture does not alter model before counted training
     - resumed AdamW step count does not move during capture
     - one counted graph gradient increments AdamW exactly once
     - graph output metrics remain finite
     - model parameters actually update

4. Only after PASS:
     python train_v8_8_2_cuda_graph_longtraj_200k_to_300k.py

SEED PROTOCOL
-------------
Already consumed:
  4301~4320 development
  3101~3120 qualification
  4401~4420 development
  3201~3220 qualification

Protected:
  6~20 permanent final-report seeds

Planned after V8.8.2 training:
  4501~4520 = development checkpoint selection only
  3301~3320 = fresh formal qualification only

Do not reuse old consumed blocks.

IMPORTANT ABOUT GPU UTILIZATION
-------------------------------
CUDA Graph increased the isolated learner from roughly 169 eager gradients/s
to ~413 gradients/s in the user's RTX 5070 benchmark. Production end-to-end
training may NOT show high GPU utilization because the learner now has far more
capacity than the fixed 1242-gradient schedule requires. JAX rollout / data
generation can become the wall-clock limiter. Do not raise replay ratio merely
to make the GPU percentage larger.
