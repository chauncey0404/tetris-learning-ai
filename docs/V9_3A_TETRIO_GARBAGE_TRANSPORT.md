# V9.3A — TETR.IO Garbage Transport Reference

This milestone connects the V9.2 attack/cancellation engine to an explicit incoming garbage queue and 10-column board insertion primitive.

## Verified/default facts used
- TETR.IO Tetra League has passthrough disabled by default.
- With passthrough disabled, incoming garbage is visible/cancelable before it becomes tankable.
- TETR.IO documentation supports a 20-frame / 333 ms baseline garbage travel time for the ranked path; the later size-scaled travel experiment was explicitly documented as not applying to Tetra League at introduction.
- Standard multiplayer garbage is change-on-attack in structure.

## Deliberately NOT hard-coded yet
The exact current ranked Season-2 values for garbage cap per piece / activation cap, default change-on-attack messiness probability, in-attack messiness probability, and any packet-level timing nuance not exposed in current public documentation remain profile parameters.

Hole layout must be supplied explicitly before an active packet is inserted. This prevents V9 from silently inventing a TETR.IO rule and then training against the invented behavior.

## Next validation gate — V9.3B
Use current TETR.IO replays / controlled custom-room captures to establish exact parity for garbage hole generation, activation, cap, blocking and clutch/top-out ordering. Only after that parity gate should the full 1v1 simulator be frozen for mass self-play.
