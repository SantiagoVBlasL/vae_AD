#!/usr/bin/env python3
"""Consolidate the revised diagnostic-centroid geometry into the final
shared-coordinate table, restricted to arms that actually share the frozen
VAE's latent coordinate system.

Read-only consolidation. No training, no inference, no Procrustes alignment,
no manuscript file is read or edited. Pure filtering + exact-reproduction
verification of an already-computed source table.

Source (MUST use): results/sipaim_2026/final_blocker_resolution_20260712/
  diagnostic_centroid_geometry_revised.csv (+ its _paired_fold_deltas_vs_locked.csv)
Explicitly NOT used: results/sipaim_2026/final_evidence_package_20260712/
  diagnostic_axis_geometry.csv

Rationale for exclusion: locked_frozen_transfer and both external_dataset_combat_*
arms score OASIS subjects by encoding (possibly input-harmonized) OASIS tensors
through the SAME frozen, never-retrained ADNI VAE encoders
(recover035_latent384_beta3p75_T80_h10000_p560_full5x5/fold_{1..5}), so all
three project into the identical fold-local latent coordinate system as the
ADNI centroids used to build the discriminant axis. previous_adni_fitted_siemens_combat's
OASIS latents come from a SEPARATELY RETRAINED VAE
(foldcombat_cleanrepro_final_audit_20260706 -- see its own
cleanrepro_training_reproducibility.csv, which documents independent
checkpoint/epoch selection for this retrained model), so its latent
coordinates are not directly comparable to the locked-model ADNI axis they
were projected onto in the source table. Excluding it from the
shared-coordinate table corrects that mismatch.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/home/diego/proyectos/vae_AD")
SOURCE_DIR = PROJECT_ROOT / "results/sipaim_2026/final_blocker_resolution_20260712"
SOURCE_GEOMETRY = SOURCE_DIR / "diagnostic_centroid_geometry_revised.csv"
SOURCE_DELTAS = SOURCE_DIR / "_paired_fold_deltas_vs_locked.csv"
EXCLUDED_PACKAGE = PROJECT_ROOT / "results/sipaim_2026/final_evidence_package_20260712/diagnostic_axis_geometry.csv"

OUT_DIR = PROJECT_ROOT / "results/sipaim_2026/canonical_shared_locked_geometry_20260712"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEEP_ARMS = ["locked_frozen_transfer", "external_dataset_combat_adni_reference", "external_dataset_combat_no_reference"]
EXCLUDE_ARM = "previous_adni_fitted_siemens_combat"

VALIDITY_NOTE = (
    "Dataset-ComBat arms and locked transfer share the frozen VAE coordinates; "
    "the separately retrained Siemens-ComBat VAE does not."
)


def log(msg: str) -> None:
    print(msg, flush=True)


# ── 1. Load source (and confirm we are NOT touching the excluded package) ──
assert not str(EXCLUDED_PACKAGE) in str(SOURCE_GEOMETRY), "accidental use of excluded package path"
geom = pd.read_csv(SOURCE_GEOMETRY)
deltas = pd.read_csv(SOURCE_DELTAS)
log(f"Loaded source geometry: {geom.shape}, source paired deltas: {deltas.shape}")
log(f"Excluded package NOT read (per instruction): {EXCLUDED_PACKAGE}")

# ── 2. Filter to the three valid shared-coordinate arms ─────────────────────
geom_filtered = geom[geom["arm"].isin(KEEP_ARMS)].copy()
assert set(geom_filtered["arm"]) == set(KEEP_ARMS), f"unexpected arm set: {set(geom_filtered['arm'])}"
assert EXCLUDE_ARM not in set(geom_filtered["arm"])
geom_filtered["validity_note"] = VALIDITY_NOTE

deltas_filtered = deltas[deltas["arm"].isin(KEEP_ARMS)].copy()
assert EXCLUDE_ARM not in set(deltas_filtered["arm"])
deltas_filtered["validity_note"] = VALIDITY_NOTE

log(f"Filtered geometry: {geom_filtered.shape} ({sorted(geom_filtered['arm'].unique())})")
log(f"Filtered paired deltas: {deltas_filtered.shape} ({sorted(deltas_filtered['arm'].unique())})")

# ── 3. Exact-reproduction verification against the source (every kept row, every numeric column) ──
numeric_cols_geom = [c for c in geom.columns if c not in ("arm", "role")]
mismatches = []
for _, row in geom_filtered.iterrows():
    src_row = geom[geom["arm"] == row["arm"]].iloc[0]
    for c in numeric_cols_geom:
        if float(row[c]) != float(src_row[c]):
            mismatches.append((row["arm"], c, row[c], src_row[c]))

numeric_cols_deltas = ["n_folds", "mean_delta", "sd_delta", "min_delta", "max_delta",
                        "n_folds_arm_lower_than_locked", "n_folds_arm_higher_than_locked"]
for _, row in deltas_filtered.iterrows():
    src_row = deltas[(deltas["arm"] == row["arm"]) & (deltas["metric"] == row["metric"])].iloc[0]
    for c in numeric_cols_deltas:
        if float(row[c]) != float(src_row[c]):
            mismatches.append((f"{row['arm']}/{row['metric']}", c, row[c], src_row[c]))
    if row["per_fold_deltas"] != src_row["per_fold_deltas"]:
        mismatches.append((f"{row['arm']}/{row['metric']}", "per_fold_deltas", row["per_fold_deltas"], src_row["per_fold_deltas"]))

verification_status = "EXACT_MATCH" if not mismatches else "MISMATCH_FOUND"
log(f"Exact-reproduction verification: {verification_status} ({len(mismatches)} mismatches)")

# ── 4. Write outputs ──────────────────────────────────────────────────────
geom_filtered.to_csv(OUT_DIR / "canonical_shared_locked_geometry.csv", index=False)
deltas_filtered.to_csv(OUT_DIR / "canonical_geometry_paired_deltas.csv", index=False)
log(f"Wrote {OUT_DIR / 'canonical_shared_locked_geometry.csv'}")
log(f"Wrote {OUT_DIR / 'canonical_geometry_paired_deltas.csv'}")

with open(OUT_DIR / "_verification_result.json", "w") as f:
    json.dump(dict(status=verification_status, n_mismatches=len(mismatches),
                    mismatches=[dict(key=m[0], column=m[1], filtered_value=m[2], source_value=m[3]) for m in mismatches]),
              f, indent=2)

log("Consolidation done.")
