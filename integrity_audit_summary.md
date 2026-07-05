# recover035_latent384_T80_h10000_p560 Integrity Audit

Generated: 2026-05-29T12:41:47.132834+00:00
Mode: DRY-RUN (config-level only)

## Overall result

**PASS** — 0 error(s) found.

## Config diff check (vs recover035_longpatience_T80_h10000_p560 base model)

Config errors: 0
Expected single diff: latent_dim 256 -> 384

## Schedule invariants (unchanged from longpatience)

- Cycle length: 10000/125 = 80 epochs
- lr_scheduler_T0: 80 (unchanged — phase-aligned)
- Patience: 560 = 7 × 80 cycles
- latent_dim: 384 (the controlled change)

## Latent cache check

Each fold's latent cache should have 384 mu columns (mu_0..mu_383).

## Read-only guarantee

This audit does NOT train, fit thresholds, modify tensors, metadata, ledger, configs, or model outputs.
