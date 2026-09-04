V8.8.6 VALIDATION BUNDLE
========================

Purpose
-------
1. Development-only checkpoint selection on fresh seeds 4501..4520.
2. One-time formal qualification of ONLY that selected checkpoint
   against the current formal V8.8 150K Champion on fresh seeds 3301..3320.

Why the development script auto-discovers checkpoints
-----------------------------------------------------
The 30M V8.8.6 production run was safely interrupted and resumed.
Therefore the actual checkpoint sequence is not guaranteed to be exactly:
4.2M, 7.2M, ..., 31.2M.

The script reads:
  models\v8_8_6_affinity_sharedweight_cuda_graph_td*.pt

It loads checkpoint metadata, sorts by actual env_steps, de-duplicates equal
training points, and prefers a normal checkpoint over an INTERRUPTED duplicate.

The safe-interrupt checkpoint itself is allowed as a development candidate
because it is a valid saved training state.

Protocol
--------
Development seeds:
  4501..4520
  20 games/model
  max 2000 pieces/game

Development selection rule:
  1. lowest gameovers
  2. highest R/1000
  3. highest average value

The formal V8.8 150K Champion and Teacher are references only.
The Champion is NOT eligible to become the development-selected V8.8.6
checkpoint.

The development evaluator writes:
  data\v8_8_6_dev_selection_4501_4520.json

The qualification evaluator REFUSES to run without this handoff. This prevents
accidentally qualifying a checkpoint that was not the predeclared development
winner.

Formal qualification seeds:
  3301..3320
  20 games
  max 2000 pieces/game

Formal Champion:
  models\v8_8_jax_vectorized_td_150k.pt
  normalized gate = 0.600

Challenger:
  development-selected V8.8.6 checkpoint
  normalized gate = 0.600

Promotion gates — ALL four required:
  1. challenger.gameovers <= champion.gameovers
  2. challenger.pieces    >= champion.pieces
  3. challenger.R/1000    >  champion.R/1000
  4. challenger.avg value >  champion.avg value

Paired mean / 95% CI / W-T-L are diagnostics only.

Permanent seeds 6..20 stay protected and unopened.

Run order
---------

1) Copy both .py files to:
   F:\tetris-learning-ai

2) Syntax check:

python -m py_compile evaluate_v8_8_6_checkpoints_dev_4501_4520.py qualify_v8_8_6_winner_vs_champion_fresh_3301_3320.py

3) Development sweep:

python evaluate_v8_8_6_checkpoints_dev_4501_4520.py

Do NOT run qualification until development finishes and prints:
  DEVELOPMENT SELECTION

4) Formal qualification:

python qualify_v8_8_6_winner_vs_champion_fresh_3301_3320.py

Files produced
--------------
Development cache:
  data\v8_8_6_checkpoint_dev_4501_4520.json

Development winner handoff:
  data\v8_8_6_dev_selection_4501_4520.json

Qualification cache:
  data\v8_8_6_qualification_3301_3320.json

Formal result:
  data\v8_8_6_qualification_result_3301_3320.json

Important
---------
4501..4520 become DEVELOPMENT-CONSUMED as soon as this sweep is used.
3301..3320 become QUALIFICATION-CONSUMED when formal qualification is run.
Never use either block for subsequent tuning.

The current formal Champion remains V8.8 150K unless ALL four formal promotion
gates pass on 3301..3320.
