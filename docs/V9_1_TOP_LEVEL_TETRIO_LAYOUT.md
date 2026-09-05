# V9.1 Top-Level TETR.IO Layout

- `tetris_ai/` = shared engine/library.
- `tetrio/` = concrete TETR.IO game package at repository root.
- TETR.IO-specific tests/tools/experiments belong under `tetrio/`.
- Existing V8 code is intentionally not moved in this patch.

After extracting, remove the old V9.0 game-specific files:

```bat
git rm tetris_ai\core\rotation\tetrio_srs_plus.py
git rm tetris_ai\core\rules\tetrio.py
git rm tests\test_v9_rotation_systems.py
git rm tests\test_v9_reachability.py
```

Then run:

```bat
.venv\Scripts\python.exe -m unittest discover -s tetrio\tests -t . -v
.venv\Scripts\python.exe -m unittest discover -s tetris_ai\core\tests -t . -v
.venv\Scripts\python.exe -m compileall -q tetris_ai\core tetrio
```

The V8 `tetris_ai/game`, `tetris_ai/model`, root `tools/`, root `tests/`, and root `experiments/` are deliberately left untouched. Migrate them later in a compatibility-preserving pass.
