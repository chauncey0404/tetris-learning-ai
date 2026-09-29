# Current `.ttrm` schema probe

A real September 2026 multiplayer replay showed the current container:

```text
root["replay"]["rounds"][round][player]["replay"]
```

The original V9.3B inventory only recognized the older/community shape:

```text
root["data"][round]["replays"][player]
```

That caused misleading inventory output:

```text
Rounds        : None
Player replays: 0
```

even though recursive paths clearly contained
`$.replay.rounds[0][0].replay.events[...]`.

This patch keeps legacy support and adds the current schema.

It also adds a raw event probe. The probe does **not** claim that strings such
as `interaction`, `garbage`, or `interaction_confirm` already correspond to
the V9.3A normalized send/cancel/tank semantics. It simply captures complete
raw `ige` payloads so those semantics can be derived from evidence.

Run:

```bat
.venv\Scripts\python.exe -m tetrio.tools.extract_ttrm_garbage_events ^
  data\tetrio\replays\garbage_parity\match01.ttrm ^
  --output artifacts\tetrio\parity\match01_garbage_events.json ^
  --max-print 40
```

The next gate is to inspect exact payload keys and cross-player frame ordering
for:

```text
interaction
interaction_confirm
garbage
```

Only after that should a normalized parity trace be generated.
