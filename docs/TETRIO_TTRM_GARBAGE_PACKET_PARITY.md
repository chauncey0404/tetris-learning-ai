# TTRM garbage packet parity

The raw current-replay probe showed repeated pairs:

```text
interaction(type=garbage, ...)
interaction_confirm(type=garbage, ...)
```

with identical packet payloads and a short confirm delay.

This audit tests two evidence claims without yet approving any model input:

1. every garbage interaction has exactly one identical confirmation;
2. in a two-player round, the sum of one player's interaction amounts is
   compared against the *other* player's final `stats.garbage.attack`.

The second test is deliberately described as a candidate cross-check rather
than an axiom. If it is exact across all streams, that is strong evidence that
the interaction stream is a receiver-side inbound packet oracle.

Run:

```bat
.venv\Scripts\python.exe -m tetrio.tools.audit_ttrm_garbage_packets ^
  data\tetrio\replays\garbage_parity\match01.ttrm ^
  --output artifacts\tetrio\parity\match01_packet_parity.json
```

Even a perfect packet oracle is not enough to approve historical
`incoming_garbage`. The next required gate is receiver-side queue
reconstruction:

```text
packet arrival
+ own cancellation
+ queue maturation/activation
+ tank/insertion
```

Only then can the reconstructed decision-time queue be compared to the
historical placement-level `incoming_garbage` semantics.
