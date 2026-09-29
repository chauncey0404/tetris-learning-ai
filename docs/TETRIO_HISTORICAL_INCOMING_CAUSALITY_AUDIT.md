# Historical incoming_garbage causal timing audit

This is the fallback after current public replay forwarding proved unreliable.

It uses the full 7.7M-row historical expert corpus itself to answer the only
question required for **historical training causality**:

```text
Is row[t].incoming_garbage state available before action t,
or is it a post-action quantity that leaks action t?
```

The key diagnostic is current `garbage_cleared[t]`. It remains an action
outcome and is never admitted as a model input.

If `incoming[t]` is pre-action, the cancellation amount should almost never
exceed `incoming[t]`:

```text
garbage_cleared[t] <= incoming[t]
```

The shifted `incoming[t-1]` should be a worse capacity bound when new packets
arrive between placements.

The audit also compares two queue-transition hypotheses:

```text
PRE:
next_incoming == max(0, incoming - garbage_cleared)

POST:
incoming == max(0, prev_incoming - garbage_cleared)
```

Neither is expected to be universally exact because opponent arrivals,
maturation and tanking can occur between rows. The comparison is therefore
directional, not a universal queue simulator.

Run:

```bat
.venv\Scripts\python.exe -m tetrio.tools.audit_historical_incoming_causality ^
  --input data\tetrio\processed\top_players_s1.parquet ^
  --threads 20 ^
  --output artifacts\tetrio\historical_incoming_causality_audit.json ^
  --candidate-output artifacts\tetrio\historical_clean_cancellation_candidates.json
```

A strong pass approves `incoming_garbage` only as a pre-action field in this
historical Season 1 corpus. It does not claim current Season 2 transport parity.

The audit also exports clean natural cancellation rows for manual/secondary
inspection.
