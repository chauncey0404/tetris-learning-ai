# \# Tetris Learning AI — Current Project Status

# 

# Last updated: 2026-09-05

# 

# \## Current Formal Champion

# 

# \- Version: V8.8.6 31.2M

# \- Checkpoint:

# &#x20; `models/v8\_8\_6\_affinity\_sharedweight\_cuda\_graph\_td\_31200k.pt`

# \- Confidence semantics: normalized Q-margin

# \- Gate: `0.600`

# \- Formal qualification seeds: `3301-3320`

# \- Previous Champion: V8.8 150K

# \- Paired W/T/L vs previous Champion: `20/0/0`

# \- All four formal promotion gates: PASS

# 

# The formal Champion remains V8.8.6 31.2M.

# 

# V8.8.7 has passed research validation but has NOT been promoted.

# 

# \---

# 

# \## Previous Champion / Historical Baseline

# 

# Checkpoint:

# 

# `models/v8\_8\_jax\_vectorized\_td\_150k.pt`

# 

# This checkpoint is retained as an important historical baseline.

# 

# \---

# 

# \## Current Stable Code Layout

# 

# The active implementation uses the V6+ flat-layout package structure:

# 

# ```text

# tetris\_ai/

# ├─ game/

# │  ├─ core.py

# │  ├─ placement.py

# │  └─ executor.py

# ├─ heuristic/

# │  ├─ features.py

# │  ├─ i\_dependency.py

# │  ├─ scoring.py

# │  └─ teacher.py

# ├─ model/

# │  ├─ state\_encoder.py

# │  ├─ q\_network.py

# │  └─ candidates.py

# ├─ policy/

# │  ├─ successor.py

# │  ├─ confidence.py

# │  └─ legacy\_raw\_margin.py

# ├─ replay/

# │  ├─ array.py

# │  └─ packed.py

# ├─ learning/

# │  ├─ eager.py

# │  ├─ cuda\_graph.py

# │  └─ ranking.py

# ├─ backend/jax/

# │  ├─ vector\_env.py

# │  └─ teacher.py

# ├─ testing/

# │  └─ observable\_poison.py

# └─ schema.py

