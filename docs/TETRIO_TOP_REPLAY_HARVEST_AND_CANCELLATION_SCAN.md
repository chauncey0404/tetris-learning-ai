# Top-player replay harvester + garbage cancellation scanner

This stage avoids manually recording many matches.

The harvester uses the documented TETRA CHANNEL API for leaderboard/recent
TETRA LEAGUE record metadata. It does **not** call the TETR.IO main game API.
Replay bytes are fetched through Inoue's documented public replay forwarding
endpoint. Both sides are rate-limited conservatively; HTTP 429 backs off.

## Harvest

```bat
.venv\Scripts\python.exe -m tetrio.tools.harvest_top_player_replays ^
  --top-players 20 ^
  --recent-per-player 10 ^
  --max-replays 100 ^
  --output-dir data\tetrio\replays\top_players ^
  --manifest artifacts\tetrio\top_replay_harvest_manifest.json
```

For a first smoke test, use fewer downloads:

```bat
.venv\Scripts\python.exe -m tetrio.tools.harvest_top_player_replays ^
  --top-players 5 ^
  --recent-per-player 5 ^
  --max-replays 10
```

You can also target explicit users:

```bat
.venv\Scripts\python.exe -m tetrio.tools.harvest_top_player_replays ^
  --users osk czsmall fortissim2 ^
  --recent-per-player 10 ^
  --max-replays 30
```

Pipeline:

```text
current League leaderboard
→ recent League records
→ skip pruned/stub replays
→ deduplicate replay IDs
→ download .ttrm
→ validate replay.rounds
→ manifest
```

## Scan cancellation candidates

```bat
.venv\Scripts\python.exe -m tetrio.tools.scan_garbage_cancellation_candidates ^
  data\tetrio\replays\top_players ^
  --top 30 ^
  --output artifacts\tetrio\garbage_cancellation_candidates.json ^
  --csv artifacts\tetrio\garbage_cancellation_candidates.csv
```

Candidate filter:

```text
attack - sent > 0
own inbound interaction-garbage > 0
stats.garbage.sent == opponent inbound packet total
```

`attack-sent` remains a **candidate** cancellation amount, not frame-level
ground truth. The scanner prefers few inbound packets, small candidate
cancellation, exact sent↔packet accounting, and received↔inbound aggregate
consistency.

The next research gate is to reconstruct only the best natural replay
candidates at placement granularity. No model checkpoint or fresh rollout
seed is touched here.
