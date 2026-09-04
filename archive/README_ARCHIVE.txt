TETRIS LEARNING AI ARCHIVE
==========================

This archive is preservation-only.

source_history\
  Completed/obsolete training, benchmark, evaluation, qualification,
  migration, and diagnostic entry-point scripts.

models_history\
  Valid historical and intermediate checkpoints.

models_invalid\v8_8_2_frozen_next_mask\
  INVALID CUDA-Graph checkpoints from the frozen next_mask incident.
  Never use these as a training base or positive ground truth.

Important:
  Shared modules with old-looking version names remain in the project root
  because the current V8.8.6 stack still depends on them.

Current formal Champion stays outside archive:
  models\v8_8_6_affinity_sharedweight_cuda_graph_td_31200k.pt

Previous formal Champion also stays outside archive:
  models\v8_8_jax_vectorized_td_150k.pt
