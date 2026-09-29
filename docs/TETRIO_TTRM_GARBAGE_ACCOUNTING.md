# TTRM end-of-round garbage accounting audit

The packet-parity audit established:

```text
interaction(type=garbage)
interaction_confirm(type=garbage)
```

form exact 1:1 packet/confirmation pairs in the observed replay.

One round showed a mismatch between the sum of inbound interaction amounts and
the opponent's final `stats.garbage.attack`. This is useful evidence rather
than an automatic failure: TETR.IO exposes multiple final garbage counters.

This audit prints and cross-checks:

```text
attack
sent
sent_nomult
received
cleared
```

against the packet oracle.

Primary check:

```text
player.stats.garbage.sent
==
sum(interaction garbage amt in opponent replay stream)
```

If that is exact across all six streams, it strongly supports this accounting
model:

```text
attack = gross/generated attack candidate
sent   = net garbage actually transmitted candidate
attack - sent = cancelled-by-own-incoming candidate
```

Those names remain *candidates* until timing is established.

The audit also reports:

```text
inbound - (attack - sent) - received
```

as a candidate end-of-round pending amount. It is allowed to be positive,
because a round can terminate with garbage still queued.

Run:

```bat
.venv\Scripts\python.exe -m tetrio.tools.audit_ttrm_garbage_accounting ^
  data\tetrio\replays\garbage_parity\match01.ttrm ^
  --output artifacts\tetrio\parity\match01_garbage_accounting.json
```

Even a perfect aggregate accounting result does not yet approve
`incoming_garbage` as a decision-time feature. Exact cancellation and tank
timing still require either a controlled replay or deterministic engine replay.
