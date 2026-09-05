# V9.2 — TETR.IO Battle Rules Reference Engine

Scope: Tetra League Season 2 attack-state semantics.

Implemented:
- Clear-event classification inputs
- Combo Multiplier with DOWN rounding
- Season-2 base attack table
- B2B Charging (+1 continuing difficult clears)
- Surge charge from displayed B2B x4 and three-way release
- All Clear flat +5 attack and B2B eligibility
- Garbage Special flat +1 for Quads/Spins that clear garbage rows
- Opener Phase double cancellation for first 14 pieces
- Compact BattleState / BattleStepResult metrics
- Separate Tetra League All-Mini and optional All-Mini+ spin profiles

Not yet implemented:
- Physical garbage-row generation / hole randomization
- Garbage travel/activation timing
- Garbage cap/messiness
- Full two-player clocked simulator
- Clutch Clear
- Network/self-play training

Performance policy:
This is the semantic reference layer. Scalar state transitions are O(1) and
use slots dataclasses, but the movement reference BFS remains intentionally
unoptimized until parity is frozen. A later fast backend will target CPU
bitboards + parallel producers and batched RTX 5070 inference/training.
