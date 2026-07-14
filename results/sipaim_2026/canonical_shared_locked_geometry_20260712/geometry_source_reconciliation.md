# Geometry source reconciliation

Read-only consolidation task. No manuscript file was read or edited (per
authoritative manuscript policy: the Overleaf manuscript remains sole
authoritative). No training, inference, or Procrustes alignment was
performed. This directory contains no data that was not already computed
in `results/sipaim_2026/final_blocker_resolution_20260712/`.

## Source used (as instructed)

`results/sipaim_2026/final_blocker_resolution_20260712/diagnostic_centroid_geometry_revised.csv`
and its companion `_paired_fold_deltas_vs_locked.csv`.

## Source explicitly NOT used (as instructed)

`results/sipaim_2026/final_evidence_package_20260712/diagnostic_axis_geometry.csv`
was not read, opened, or referenced by the consolidation script
(`scripts/sipaim_2026/build_canonical_shared_locked_geometry_20260712.py`
loads only the one path given above).

## Why `previous_adni_fitted_siemens_combat` is excluded

The source table's discriminant axis and ADNI reference centroid are built
once from the **locked model's** frozen ADNI train/dev latents
(`recover035_latent384_beta3p75_T80_h10000_p560_full5x5/classifier_only_readout/latent_cache/`)
and then used to project OASIS latents from all four arms. That projection
is only meaningful if an arm's OASIS latents live in the *same* coordinate
system as those ADNI latents:

- `locked_frozen_transfer`, `external_dataset_combat_adni_reference`, and
  `external_dataset_combat_no_reference` all encode (possibly
  input-harmonized) OASIS tensors through the **same frozen, never-retrained**
  ADNI VAE fold encoders used to produce the ADNI axis. They share coordinates
  by construction.
- `previous_adni_fitted_siemens_combat`'s OASIS latents come from a
  **separately retrained VAE**
  (`results/revision_bspc_2026/post_revision_exploratory_20260630/foldcombat_cleanrepro_final_audit_20260706/`,
  whose own `cleanrepro_training_reproducibility.csv` documents independent
  checkpoint/epoch selection per fold, distinct from the locked model's
  training run). Its latent space is not the same coordinate system as the
  locked-model ADNI axis it was projected onto in the source table.

The revised source table (`diagnostic_centroid_geometry_revised.csv`)
computed this arm's projection anyway, using the same ADNI axis as the
other three -- this is scientifically invalid for a shared-coordinate
comparison (the axis and centroid were never fit in that arm's own latent
space) and is why this consolidation excludes it rather than merely copying
it forward. It was, however, correctly transparent in the original source:
that table never claimed the four arms shared coordinates, and Task 2's
own markdown flagged its axis as fold-local/locked-model-derived
throughout. This consolidation makes the shared-coordinate restriction
explicit and structural rather than left to a careful reader.

**Validity note (verbatim, applied to every row of both output tables):**

> Dataset-ComBat arms and locked transfer share the frozen VAE coordinates;
> the separately retrained Siemens-ComBat VAE does not.

## Exact-reproduction verification

Every numeric column of every kept row in `canonical_shared_locked_geometry.csv`
and `canonical_geometry_paired_deltas.csv` (including the full `per_fold_deltas`
JSON arrays) was compared byte-for-byte against the corresponding row in the
source files.

**Result: EXACT_MATCH, 0 mismatches** across all 3 kept geometry rows (29
numeric columns each) and all 14 kept paired-delta rows (7 numeric columns
+ 1 array column each). Full machine-readable result: `_verification_result.json`.

## What changed vs. the source

Nothing numeric. The only additions are: (1) row filtering (3 of 4 geometry
rows kept, 14 of 21 paired-delta rows kept), (2) a `validity_note` column
appended to both tables, (3) this reconciliation document. No value in any
kept cell was recomputed, rescaled, or otherwise touched.

## Guardrail compliance

- No training, no inference, no Procrustes alignment.
- No manuscript file read or edited.
- `final_evidence_package_20260712/diagnostic_axis_geometry.csv` not used.
- Written to a new directory (`results/sipaim_2026/canonical_shared_locked_geometry_20260712/`);
  neither `final_blocker_resolution_20260712/` nor `final_evidence_package_20260712/`
  was overwritten or modified.
