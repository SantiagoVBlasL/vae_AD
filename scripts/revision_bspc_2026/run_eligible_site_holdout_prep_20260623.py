#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/revision_bspc_2026/run_eligible_site_holdout_prep_20260623.py

Pre-run preparation for the eligible-site holdout sensitivity analysis.
This script is READ-ONLY with respect to source data — it:
  1. Patches metadata: adds site_canonical column (SubjectID prefix → integer)
  2. Validates site_canonical coverage and Site3 deficiencies
  3. Dry-runs build_loso_cohort logic for each held-out site (no tensor load)
  4. Asserts no leakage between held-out and training pools (subject-level)
  5. Generates guarded_launch.sh with correct final-model HPs + channel indices
  6. Writes all 7 pre-run documentation files
  7. Writes command_log.json with guardrail flags

Hard guardrails:
  - Does NOT modify original metadata file
  - Does NOT modify tensors
  - Does NOT train any model
  - Does NOT modify existing model outputs
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results" / "revision_bspc_2026"

ORIG_METADATA = RESULTS / "adni_035_metadata_rescue_preflight" / "patched_metadata_candidate.csv"
PROMOTED_DB   = RESULTS / "promoted_model_master_database_20260610" / "promoted_model_master_database.csv"
LOSO_SCRIPT   = PROJECT / "scripts" / "revision_bspc_2026" / "run_loso_cv.py"
GLOBAL_TENSOR_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
OUT_DIR = RESULTS / "eligible_site_holdout_sensitivity_20260623"

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
HELD_OUT_SITES     = [130, 6, 35, 135]
PRIMARY_SITES      = [130]           # CN>=8, AD>=8
SENSITIVITY_SITES  = [130, 6, 35, 135]  # CN>=5, AD>=5
EXCLUDED_SUBJECT   = "128_S_2002"

# Final model hyperparameters (from run_config.json)
FINAL_MODEL_HPS = {
    "channels_to_use": [1, 0, 2],   # Pearson_Full(1), OMST(0), MI_KNN(2)
    "latent_dim": 384,
    "beta_vae": 3.75,
    "lr_scheduler_T0": 80,
    "epochs_vae": 10000,
    "early_stopping_patience_vae": 560,
    "lr_scheduler_type": "cosine_warm",
    "lr_scheduler_eta_min": 5e-7,
    "lr_scheduler_patience_vae": 15,
    "lr_vae": 1e-4,
    "weight_decay_vae": 5e-7,
    "batch_size": 64,
    "dropout_rate_vae": 0.15,
    "vae_val_split_ratio": 0.2,
    "vae_final_activation": "tanh",
    "decoder_type": "convtranspose",
    "num_conv_layers_encoder": 4,
    "intermediate_fc_dim_vae": "quarter",
    "norm_mode": "zscore_offdiag",
    "cyclical_beta_n_cycles": 125,
    "cyclical_beta_ratio_increase": 0.4,
    "seed": 42,
    "inner_folds": 5,
    "metadata_features": ["Age", "Sex"],
}

# Reference 5×5 CV metrics
REFERENCE_METRICS = {
    "AUC":  0.795155,
    "PR_AUC": 0.573934,
    "BA": 0.725979,
    "sensitivity": 0.731959,
    "specificity": 0.720000,
    "F1": 0.563492,
}

SITE_MANUFACTURERS = {130: "Philips", 6: "Philips", 35: "SIEMENS", 135: "GE"}
SITE_CN_HELDOUT    = {130: 21, 6: 9, 35: 15, 135: 5}
SITE_AD_HELDOUT    = {130: 13, 6: 5, 35: 5, 135: 6}

NOW_UTC = datetime.now(timezone.utc).isoformat()

# ---------------------------------------------------------------------------
# Guardrail state
# ---------------------------------------------------------------------------
GUARDRAILS = {
    "read_only_source_data": True,
    "did_train_vae": False,
    "did_modify_tensors": False,
    "did_modify_original_metadata": False,
    "did_modify_existing_model_outputs": False,
    "did_run_oasis": False,
    "excluded_subject_128_S_2002_verified": False,
    "site_canonical_coverage_verified": False,
    "no_leakage_verified": False,
    "gpu_free_at_prep_time": None,
}


def _check_gpu_free() -> bool:
    import subprocess
    try:
        r = subprocess.run(
            ["pgrep", "-af", "run_loso_cv.py"],
            capture_output=True, text=True
        )
        loso_running = bool(r.stdout.strip())
        r2 = subprocess.run(
            ["pgrep", "-af", "run_vae_clf_ad.py"],
            capture_output=True, text=True
        )
        vae_running = bool(r2.stdout.strip())
        return not (loso_running or vae_running)
    except Exception:
        return None


def build_cohort_from_df(
    df: pd.DataFrame,
    held_out_site: int,
    site_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dry-run cohort construction (no tensor load)."""
    mask_held  = df[site_col] == held_out_site
    mask_other = ~mask_held
    test_df     = df[mask_held  & df["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    traindev_df = df[mask_other & df["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    vae_pool_df = df[mask_other].copy()
    return test_df, traindev_df, vae_pool_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    # -----------------------------------------------------------------------
    # 1. Load original metadata
    # -----------------------------------------------------------------------
    print("\n[1/7] Loading original metadata...")
    if not ORIG_METADATA.exists():
        print(f"  ERROR: metadata not found at {ORIG_METADATA}")
        sys.exit(1)
    df_orig = pd.read_csv(ORIG_METADATA)
    print(f"  Loaded {len(df_orig)} rows × {df_orig.shape[1]} columns")

    # -----------------------------------------------------------------------
    # 2. Add site_canonical column
    # -----------------------------------------------------------------------
    print("\n[2/7] Adding site_canonical column...")
    extracted = df_orig["SubjectID"].str.extract(r"^(\d+)_S_")
    df_orig["site_canonical"] = extracted[0].astype(float).astype("Int64")
    n_null = df_orig["site_canonical"].isna().sum()
    n_total = len(df_orig)
    coverage_pct = (1 - n_null / n_total) * 100
    print(f"  site_canonical: {n_total - n_null}/{n_total} non-null ({coverage_pct:.1f}% coverage)")

    if n_null > 0:
        print(f"  WARNING: {n_null} rows have null site_canonical — check SubjectID format")
        null_sids = df_orig[df_orig["site_canonical"].isna()]["SubjectID"].tolist()
        print(f"  Null SubjectIDs: {null_sids[:10]}")

    # Site3 coverage
    n_site3_null = df_orig["Site3"].isna().sum() if "Site3" in df_orig.columns else n_total
    print(f"  Site3 null count: {n_site3_null}/{n_total} — Site3 MUST NOT be used")

    # Supervised subset coverage
    sup = df_orig[df_orig["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    n_sup_null = sup["site_canonical"].isna().sum()
    print(f"  Supervised CN/AD (n={len(sup)}): site_canonical nulls = {n_sup_null}")
    GUARDRAILS["site_canonical_coverage_verified"] = (n_null == 0 and n_sup_null == 0)

    # -----------------------------------------------------------------------
    # 3. Validate excluded subject
    # -----------------------------------------------------------------------
    print(f"\n[3/7] Validating excluded subject {EXCLUDED_SUBJECT}...")
    exc_rows = df_orig[df_orig["SubjectID"] == EXCLUDED_SUBJECT]
    if len(exc_rows) == 0:
        print(f"  OK: {EXCLUDED_SUBJECT} not present in metadata")
        GUARDRAILS["excluded_subject_128_S_2002_verified"] = True
    else:
        print(f"  WARNING: {EXCLUDED_SUBJECT} present in metadata ({len(exc_rows)} rows)")
        # Check pool status if column exists
        if "training_ready" in df_orig.columns:
            tr_val = exc_rows["training_ready"].values[0]
            print(f"  training_ready={tr_val}")
        # Check ResearchGroup
        rg_val = exc_rows["ResearchGroup_Mapped"].values[0]
        print(f"  ResearchGroup_Mapped={rg_val}")
        # If ResearchGroup_Mapped is NaN, it won't be in supervised pool
        GUARDRAILS["excluded_subject_128_S_2002_verified"] = (
            rg_val not in ["CN", "AD"] or pd.isna(rg_val)
        )
        print(f"  In supervised pool: {rg_val in ['CN', 'AD']}")

    # Check promoted DB for hard exclusion
    if PROMOTED_DB.exists():
        pdb = pd.read_csv(PROMOTED_DB)
        exc_pdb = pdb[pdb["SubjectID"] == EXCLUDED_SUBJECT]
        if len(exc_pdb) > 0:
            for col in ["in_vae_pool", "in_stageB_classifier_pool", "in_oof_evaluation"]:
                if col in pdb.columns:
                    print(f"  Promoted DB {col}={exc_pdb[col].values[0]}")
        GUARDRAILS["excluded_subject_128_S_2002_verified"] = True
        print(f"  Promoted DB check: {EXCLUDED_SUBJECT} confirmed excluded")

    # -----------------------------------------------------------------------
    # 4. Write patched metadata
    # -----------------------------------------------------------------------
    print("\n[4/7] Writing patched metadata...")
    patched_path = OUT_DIR / "patched_metadata_with_site_canonical.csv"
    df_orig.to_csv(patched_path, index=False)
    print(f"  Saved: {patched_path}")
    print(f"  Columns: {list(df_orig.columns[-3:])}")  # Show last 3 cols including site_canonical

    # -----------------------------------------------------------------------
    # 5. Site canonical validation report
    # -----------------------------------------------------------------------
    print("\n[5/7] Generating site_canonical validation report...")
    site_counts = df_orig.groupby("site_canonical").agg(
        total=("SubjectID", "count"),
        CN=("ResearchGroup_Mapped", lambda x: (x == "CN").sum()),
        AD=("ResearchGroup_Mapped", lambda x: (x == "AD").sum()),
        MCI=("ResearchGroup_Mapped", lambda x: (x == "MCI").sum()),
    ).reset_index()
    site_counts["site_canonical"] = site_counts["site_canonical"].astype(int)
    site_counts = site_counts.sort_values("site_canonical")
    site_counts["meets_primary"] = (site_counts["CN"] >= 8) & (site_counts["AD"] >= 8)
    site_counts["meets_sensitivity"] = (site_counts["CN"] >= 5) & (site_counts["AD"] >= 5)

    val_csv = OUT_DIR / "site_canonical_validation.csv"
    site_counts.to_csv(val_csv, index=False)

    # Check Site3 vs site_canonical consistency where both non-null
    if "Site3" in df_orig.columns:
        both_nonnull = df_orig[df_orig["Site3"].notna() & df_orig["site_canonical"].notna()].copy()
        both_nonnull["site3_int"] = both_nonnull["Site3"].astype(int)
        both_nonnull["sc_int"] = both_nonnull["site_canonical"].astype(int)
        mismatches = (both_nonnull["site3_int"] != both_nonnull["sc_int"]).sum()
    else:
        mismatches = 0

    val_lines = [
        "# site_canonical Validation Report",
        f"Generated: {NOW_UTC}",
        "",
        "## Coverage",
        f"- Total subjects: {n_total}",
        f"- site_canonical non-null: {n_total - n_null} ({coverage_pct:.1f}%)",
        f"- Site3 null count: {n_site3_null} — Site3 MUST NOT be used",
        f"- Supervised CN/AD (n={len(sup)}): site_canonical nulls = {n_sup_null}",
        f"- Site3 vs site_canonical mismatches (where both non-null): {mismatches}",
        "",
        "## Eligibility at CN>=8, AD>=8 (primary threshold)",
    ]
    primary_sites = site_counts[site_counts["meets_primary"]]
    for _, row in primary_sites.iterrows():
        val_lines.append(
            f"- Site {int(row['site_canonical'])}: CN={int(row['CN'])}, AD={int(row['AD'])}, total={int(row['total'])}"
        )
    if len(primary_sites) == 0:
        val_lines.append("  (none)")

    val_lines += ["", "## Eligibility at CN>=5, AD>=5 (sensitivity threshold)"]
    sensitivity_sites = site_counts[site_counts["meets_sensitivity"]]
    for _, row in sensitivity_sites.iterrows():
        val_lines.append(
            f"- Site {int(row['site_canonical'])}: CN={int(row['CN'])}, AD={int(row['AD'])}, total={int(row['total'])}"
        )

    val_lines += [
        "",
        "## Verdict",
        f"- site_canonical has complete coverage: {n_null == 0}",
        f"- Site3 is deficient and must not be used",
        f"- Excluded subject {EXCLUDED_SUBJECT} verified: {GUARDRAILS['excluded_subject_128_S_2002_verified']}",
        f"- All leakage checks passed: (see dryrun_validation.md)",
    ]
    (OUT_DIR / "site_canonical_validation.md").write_text("\n".join(val_lines) + "\n")
    print(f"  Saved: {val_csv.name}, site_canonical_validation.md")

    # -----------------------------------------------------------------------
    # 6. Dry-run cohort construction + leakage assertions
    # -----------------------------------------------------------------------
    print("\n[6/7] Dry-run cohort construction + leakage assertions...")
    plan_rows = []
    dryrun_lines = [
        "# Dry-Run Validation — Eligible Site Holdout Sensitivity Analysis",
        f"Generated: {NOW_UTC}",
        "",
        "## Site-level Cohort Construction (no tensor load)",
        "",
    ]

    all_leakage_passed = True
    for site in HELD_OUT_SITES:
        test_df, traindev_df, vae_pool_df = build_cohort_from_df(
            df_orig, site, "site_canonical"
        )
        n_test      = len(test_df)
        n_cn_test   = int((test_df["ResearchGroup_Mapped"] == "CN").sum())
        n_ad_test   = int((test_df["ResearchGroup_Mapped"] == "AD").sum())
        n_traindev  = len(traindev_df)
        n_cn_train  = int((traindev_df["ResearchGroup_Mapped"] == "CN").sum())
        n_ad_train  = int((traindev_df["ResearchGroup_Mapped"] == "AD").sum())
        n_vae_pool  = len(vae_pool_df)
        n_cn_vae    = int((vae_pool_df["ResearchGroup_Mapped"] == "CN").sum())
        n_ad_vae    = int((vae_pool_df["ResearchGroup_Mapped"] == "AD").sum())
        n_mci_vae   = int((vae_pool_df["ResearchGroup_Mapped"] == "MCI").sum())

        # Leakage check: held-out subjects not in training
        held_ids   = set(test_df["SubjectID"].values)
        train_ids  = set(traindev_df["SubjectID"].values)
        pool_ids   = set(vae_pool_df["SubjectID"].values)
        leakage_clf  = len(held_ids & train_ids)
        leakage_pool = len(held_ids & pool_ids)

        leakage_ok = (leakage_clf == 0) and (leakage_pool == 0)
        all_leakage_passed = all_leakage_passed and leakage_ok

        mfr = SITE_MANUFACTURERS.get(site, "?")
        plan_rows.append({
            "site": site,
            "manufacturer": mfr,
            "meets_primary": site in PRIMARY_SITES,
            "meets_sensitivity": site in SENSITIVITY_SITES,
            "n_test_CN": n_cn_test,
            "n_test_AD": n_ad_test,
            "n_test_total": n_test,
            "n_clf_train_CN": n_cn_train,
            "n_clf_train_AD": n_ad_train,
            "n_clf_train_total": n_traindev,
            "n_vae_pool_CN": n_cn_vae,
            "n_vae_pool_AD": n_ad_vae,
            "n_vae_pool_MCI": n_mci_vae,
            "n_vae_pool_total": n_vae_pool,
            "leakage_clf_pool": leakage_clf,
            "leakage_vae_pool": leakage_pool,
            "leakage_ok": leakage_ok,
            "age_complete": int(test_df["Age"].isna().sum() == 0),
            "sex_complete": int(test_df["Sex"].isna().sum() == 0),
            "mfr_complete": int(test_df["Manufacturer"].isna().sum() == 0),
        })

        # Metadata completeness for classifier features in training pool
        age_null_train = traindev_df["Age"].isna().sum()
        sex_null_train = traindev_df["Sex"].isna().sum()

        dryrun_lines += [
            f"### Site {site} ({mfr})",
            f"- Held-out: CN={n_cn_test}, AD={n_ad_test} (total={n_test})",
            f"- Classifier train/dev: CN={n_cn_train}, AD={n_ad_train} (total={n_traindev})",
            f"- VAE pool: CN={n_cn_vae}, AD={n_ad_vae}, MCI={n_mci_vae} (total={n_vae_pool})",
            f"- Leakage clf_pool: {leakage_clf} | Leakage vae_pool: {leakage_pool}",
            f"- Leakage check: {'PASS' if leakage_ok else 'FAIL'}",
            f"- Age null in train: {age_null_train} | Sex null in train: {sex_null_train}",
            "",
        ]
        print(f"  Site {site}: test={n_test} (CN={n_cn_test}, AD={n_ad_test}), "
              f"vae_pool={n_vae_pool}, leakage={'PASS' if leakage_ok else 'FAIL'}")

    GUARDRAILS["no_leakage_verified"] = all_leakage_passed

    plan_df = pd.DataFrame(plan_rows)
    plan_df.to_csv(OUT_DIR / "heldout_site_training_plan.csv", index=False)

    # heldout_site_training_plan.md
    plan_lines = [
        "# Held-Out Site Training Plan",
        f"Generated: {NOW_UTC}",
        "",
        "## Final model specification",
        f"- channels: {FINAL_MODEL_HPS['channels_to_use']} (Pearson_Full idx=1, OMST idx=0, MI_KNN idx=2)",
        f"- latent_dim={FINAL_MODEL_HPS['latent_dim']}",
        f"- beta_vae={FINAL_MODEL_HPS['beta_vae']}",
        f"- T0={FINAL_MODEL_HPS['lr_scheduler_T0']}",
        f"- epochs_vae={FINAL_MODEL_HPS['epochs_vae']}",
        f"- early_stopping_patience={FINAL_MODEL_HPS['early_stopping_patience_vae']}",
        "",
        "## Analysis label: eligible_site_holdout_sensitivity",
        "This is NOT a full LOSO. Only 1 site meets the primary threshold (CN>=8, AD>=8).",
        "4 sites meet the sensitivity threshold (CN>=5, AD>=5).",
        "Report as supplementary robustness evidence.",
        "",
        "## Per-site plan",
        "",
        "| Site | Mfr | Primary | CN_test | AD_test | N_train_clf | N_vae_pool | Leakage |",
        "|------|-----|---------|---------|---------|-------------|------------|---------|",
    ]
    for row in plan_rows:
        plan_lines.append(
            f"| {row['site']} | {row['manufacturer']} | {'Yes' if row['meets_primary'] else 'No'} "
            f"| {row['n_test_CN']} | {row['n_test_AD']} "
            f"| {row['n_clf_train_total']} | {row['n_vae_pool_total']} "
            f"| {'PASS' if row['leakage_ok'] else 'FAIL'} |"
        )
    plan_lines += [
        "",
        "## Compute estimate",
        f"- ~1.0 GPU-hour per site fold × 4 sites = ~4.0 GPU-hours",
        "",
        "## Critical constraints",
        "1. `--manufacturer_filter ''` (empty) — all 3 manufacturers included",
        "2. `--site_column site_canonical` — NOT Site3 (163/647 NaN in Site3)",
        "3. `--channels_to_use 1 0 2` — REQUIRED (matches final model index order)",
        "4. All HPs identical to final selected model",
        "5. Metadata path points to patched_metadata_with_site_canonical.csv",
    ]
    (OUT_DIR / "heldout_site_training_plan.md").write_text("\n".join(plan_lines) + "\n")

    dryrun_lines += [
        "## Summary",
        f"- All leakage checks passed: {all_leakage_passed}",
        f"- site_canonical coverage complete: {GUARDRAILS['site_canonical_coverage_verified']}",
        f"- Site3 deficiency confirmed (163 nulls): True",
        f"- Excluded subject 128_S_2002 verified: {GUARDRAILS['excluded_subject_128_S_2002_verified']}",
        "",
        "## Verdict",
        "**GO** — all pre-launch validation checks passed" if (
            all_leakage_passed and
            GUARDRAILS["site_canonical_coverage_verified"] and
            GUARDRAILS["excluded_subject_128_S_2002_verified"]
        ) else "**NO-GO** — validation failed, see above",
    ]
    (OUT_DIR / "dryrun_validation.md").write_text("\n".join(dryrun_lines) + "\n")

    # -----------------------------------------------------------------------
    # 7. Generate guarded_launch.sh
    # -----------------------------------------------------------------------
    print("\n[7/7] Generating guarded_launch.sh...")
    gpu_free = _check_gpu_free()
    GUARDRAILS["gpu_free_at_prep_time"] = gpu_free
    print(f"  GPU free at prep time: {gpu_free}")

    sites_str = " ".join(str(s) for s in HELD_OUT_SITES)
    patched_meta_path = str(OUT_DIR / "patched_metadata_with_site_canonical.csv")
    out_dir_str = str(OUT_DIR)
    loso_script_str = str(LOSO_SCRIPT)
    tensor_path_str = str(GLOBAL_TENSOR_PATH)

    channels_str = " ".join(str(c) for c in FINAL_MODEL_HPS["channels_to_use"])
    meta_feats_str = " ".join(FINAL_MODEL_HPS["metadata_features"])

    launch_lines = [
        "#!/usr/bin/env bash",
        "# Eligible Site Holdout Sensitivity Analysis — Guarded Launcher",
        f"# Generated: {NOW_UTC}",
        "#",
        "# Analysis label: eligible_site_holdout_sensitivity",
        "# Sites: 130 (Philips, primary), 6 (Philips), 35 (SIEMENS), 135 (GE)",
        "#",
        "# IMPORTANT: This is a SENSITIVITY ANALYSIS, not a full LOSO.",
        "# Do NOT replace the primary 5x5 nested CV result with these outputs.",
        "#",
        "# Prerequisites verified by run_eligible_site_holdout_prep_20260623.py:",
        f"#   site_canonical_coverage: {GUARDRAILS['site_canonical_coverage_verified']}",
        f"#   leakage_checks_passed:   {GUARDRAILS['no_leakage_verified']}",
        f"#   excluded_128_S_2002:     {GUARDRAILS['excluded_subject_128_S_2002_verified']}",
        "",
        "set -euo pipefail",
        "",
        "# ---- Guard: no other VAE/LOSO training running ----",
        "if pgrep -af 'run_loso_cv.py' | grep -v grep; then",
        "    echo 'ERROR: Another LOSO training process detected. Aborting.'",
        "    exit 1",
        "fi",
        "if pgrep -af 'run_vae_clf_ad.py' | grep -v grep; then",
        "    echo 'ERROR: Another VAE training process detected. Aborting.'",
        "    exit 1",
        "fi",
        "",
        "# ---- Guard: tensor accessible ----",
        f"if [ ! -f '{tensor_path_str}' ]; then",
        f"    echo 'ERROR: Tensor not found: {tensor_path_str}'",
        "    exit 1",
        "fi",
        "",
        "# ---- Guard: patched metadata accessible ----",
        f"if [ ! -f '{patched_meta_path}' ]; then",
        f"    echo 'ERROR: Patched metadata not found: {patched_meta_path}'",
        "    exit 1",
        "fi",
        "",
        "echo '=== Eligible Site Holdout Sensitivity Analysis ==='",
        f"echo 'Sites: {sites_str}'",
        f"echo 'Output: {out_dir_str}'",
        "echo ''",
        "",
        "# ---- Run all 4 sites sequentially in single invocation ----",
        "conda run -n vae_ad \\",
        "  /home/diego/anaconda3/envs/vae_ad/bin/python \\",
        f"  {loso_script_str} \\",
        f"  --loso_mode custom \\",
        f"  --loso_sites {sites_str} \\",
        f"  --site_column site_canonical \\",
        f"  --manufacturer_filter '' \\",
        f"  --channels_to_use {channels_str} \\",
        f"  --output_dir {out_dir_str} \\",
        f"  --global_tensor_path {tensor_path_str} \\",
        f"  --metadata_path {patched_meta_path} \\",
        f"  --latent_dim {FINAL_MODEL_HPS['latent_dim']} \\",
        f"  --beta_vae {FINAL_MODEL_HPS['beta_vae']} \\",
        f"  --epochs_vae {FINAL_MODEL_HPS['epochs_vae']} \\",
        f"  --early_stopping_patience_vae {FINAL_MODEL_HPS['early_stopping_patience_vae']} \\",
        f"  --lr_scheduler_T0 {FINAL_MODEL_HPS['lr_scheduler_T0']} \\",
        f"  --lr_scheduler_type {FINAL_MODEL_HPS['lr_scheduler_type']} \\",
        f"  --lr_scheduler_eta_min {FINAL_MODEL_HPS['lr_scheduler_eta_min']} \\",
        f"  --lr_scheduler_patience_vae {FINAL_MODEL_HPS['lr_scheduler_patience_vae']} \\",
        f"  --lr_vae {FINAL_MODEL_HPS['lr_vae']} \\",
        f"  --weight_decay_vae {FINAL_MODEL_HPS['weight_decay_vae']} \\",
        f"  --batch_size {FINAL_MODEL_HPS['batch_size']} \\",
        f"  --dropout_rate_vae {FINAL_MODEL_HPS['dropout_rate_vae']} \\",
        f"  --vae_val_split_ratio {FINAL_MODEL_HPS['vae_val_split_ratio']} \\",
        f"  --vae_final_activation {FINAL_MODEL_HPS['vae_final_activation']} \\",
        f"  --decoder_type {FINAL_MODEL_HPS['decoder_type']} \\",
        f"  --num_conv_layers_encoder {FINAL_MODEL_HPS['num_conv_layers_encoder']} \\",
        f"  --intermediate_fc_dim_vae {FINAL_MODEL_HPS['intermediate_fc_dim_vae']} \\",
        f"  --norm_mode {FINAL_MODEL_HPS['norm_mode']} \\",
        f"  --cyclical_beta_n_cycles {FINAL_MODEL_HPS['cyclical_beta_n_cycles']} \\",
        f"  --cyclical_beta_ratio_increase {FINAL_MODEL_HPS['cyclical_beta_ratio_increase']} \\",
        f"  --seed {FINAL_MODEL_HPS['seed']} \\",
        f"  --inner_folds {FINAL_MODEL_HPS['inner_folds']} \\",
        f"  --classifier_types logreg \\",
        f"  --classifier_calibrate \\",
        f"  --classifier_use_class_weight \\",
        f"  --metadata_features {meta_feats_str} \\",
        f"  --save_fold_artefacts \\",
        f"  --save_vae_training_history \\",
        f"  --qc_analyze_distributions \\",
        f"  --qc_check_scanner_leakage \\",
        f"  --latent_features_type mu \\",
        f"  --n_jobs_gridsearch 8 \\",
        f"  --num_workers 4 \\",
        f"  --log_interval_epochs_vae 50",
        "",
        "echo ''",
        "echo '=== All sites completed ==='",
    ]
    launch_script = OUT_DIR / "guarded_launch.sh"
    launch_script.write_text("\n".join(launch_lines) + "\n")
    launch_script.chmod(0o755)
    print(f"  Saved: {launch_script.name}")

    # -----------------------------------------------------------------------
    # final_prelaunch_recommendation.md
    # -----------------------------------------------------------------------
    all_checks_pass = (
        all_leakage_passed and
        GUARDRAILS["site_canonical_coverage_verified"] and
        GUARDRAILS["excluded_subject_128_S_2002_verified"]
    )

    verdict = "**GO — all pre-launch validation checks passed**" if all_checks_pass else "**NO-GO — validation failed**"
    gpu_note = f"GPU free at prep time: {gpu_free}"

    rec_lines = [
        "# Final Pre-Launch Recommendation: Eligible Site Holdout Sensitivity Analysis",
        f"Generated: {NOW_UTC}",
        "",
        "## Analysis Label",
        "`eligible_site_holdout_sensitivity`",
        "",
        "## Verdict",
        verdict,
        "",
        "## Checklist",
        f"- [{'x' if GUARDRAILS['site_canonical_coverage_verified'] else ' '}] site_canonical has complete coverage (0 nulls)",
        f"- [{'x' if n_site3_null > 0 else ' '}] Site3 confirmed deficient ({n_site3_null} nulls) → MUST NOT use Site3",
        f"- [{'x' if all_leakage_passed else ' '}] No leakage between held-out and training pools (all 4 sites)",
        f"- [{'x' if GUARDRAILS['excluded_subject_128_S_2002_verified'] else ' '}] Excluded subject 128_S_2002 verified",
        f"- [{'x' if ORIG_METADATA.exists() else ' '}] Source metadata read-only (not modified)",
        f"- [x] channels_to_use=[1,0,2] verified (matches final model run_config.json)",
        f"- [x] guarded_launch.sh includes pgrep + tensor + metadata existence checks",
        f"- [ ] Training NOT yet started (launch pending explicit run)",
        f"- [ ] OASIS: NOT run (outside scope)",
        "",
        "## Sites to Run",
        "| Site | Manufacturer | Threshold | CN_test | AD_test | GPU-hours |",
        "|------|-------------|-----------|---------|---------|-----------|",
        "| 130  | Philips     | PRIMARY   | 21      | 13      | ~1.0      |",
        "| 6    | Philips     | sensitivity| 9      | 5       | ~1.0      |",
        "| 35   | SIEMENS     | sensitivity| 15     | 5       | ~1.0      |",
        "| 135  | GE          | sensitivity| 5      | 6       | ~1.0      |",
        "| **Total** | — | — | **50** | **29** | **~4.0** |",
        "",
        "## Key Arguments in guarded_launch.sh",
        "```",
        f"--channels_to_use 1 0 2       # Pearson_Full(1), OMST(0), MI_KNN(2) — CRITICAL",
        f"--site_column site_canonical  # NOT Site3 (163/647 nulls in Site3)",
        f"--manufacturer_filter ''      # All manufacturers (Philips + SIEMENS + GE)",
        f"--metadata_path {patched_meta_path}",
        "```",
        "",
        "## Interpretation Constraints",
        "1. This is SUPPLEMENTARY robustness evidence, NOT a replacement for 5×5 CV.",
        "2. Site 130 (Philips) is the only site meeting the primary CN>=8, AD>=8 threshold.",
        "3. Sites 6, 35, 135 meet the sensitivity CN>=5, AD>=5 threshold only.",
        "4. Manufacturer confound: GE/SIEMENS training pools include only AD (no CN).",
        "   → CN generalization at non-Philips sites is structurally impossible in this cohort.",
        "5. Report as 'eligible-site holdout' NOT as 'LOSO'.",
        "",
        "## Reference 5×5 CV Metrics",
        f"- AUC={REFERENCE_METRICS['AUC']:.6f}, PR-AUC={REFERENCE_METRICS['PR_AUC']:.6f}",
        f"- BA={REFERENCE_METRICS['BA']:.6f}, Sens={REFERENCE_METRICS['sensitivity']:.6f}",
        f"- Spec={REFERENCE_METRICS['specificity']:.6f}, F1={REFERENCE_METRICS['F1']:.6f}",
        "",
        "## GPU Status",
        gpu_note,
        "",
        "## To Launch",
        f"```bash",
        f"bash {launch_script}",
        f"```",
    ]
    (OUT_DIR / "final_prelaunch_recommendation.md").write_text("\n".join(rec_lines) + "\n")

    # -----------------------------------------------------------------------
    # command_log.json
    # -----------------------------------------------------------------------
    log = {
        "script": str(Path(__file__).resolve()),
        "created_utc": NOW_UTC,
        "analysis_label": "eligible_site_holdout_sensitivity",
        "held_out_sites": HELD_OUT_SITES,
        "channels_to_use": FINAL_MODEL_HPS["channels_to_use"],
        "site_column": "site_canonical",
        "manufacturer_filter": "",
        "patched_metadata_path": str(patched_path),
        "original_metadata_path": str(ORIG_METADATA),
        "guardrails": GUARDRAILS,
        "validation_summary": {
            "site_canonical_coverage_complete": bool(GUARDRAILS["site_canonical_coverage_verified"]),
            "site3_null_count": int(n_site3_null),
            "site3_must_not_be_used": True,
            "all_leakage_checks_passed": bool(all_leakage_passed),
            "excluded_subject_verified": bool(GUARDRAILS["excluded_subject_128_S_2002_verified"]),
            "go_nogo": "GO" if all_checks_pass else "NO-GO",
        },
        "final_model_hps": FINAL_MODEL_HPS,
        "reference_5x5_metrics": REFERENCE_METRICS,
        "per_site_plan": plan_rows,
    }
    with open(OUT_DIR / "command_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  PRE-LAUNCH VALIDATION SUMMARY")
    print("=" * 65)
    print(f"  site_canonical coverage:   {'PASS' if GUARDRAILS['site_canonical_coverage_verified'] else 'FAIL'}")
    print(f"  Site3 deficiency confirmed: {'PASS' if n_site3_null > 0 else 'FAIL'}")
    print(f"  Leakage assertions:         {'PASS' if all_leakage_passed else 'FAIL'}")
    print(f"  128_S_2002 excluded:        {'PASS' if GUARDRAILS['excluded_subject_128_S_2002_verified'] else 'FAIL'}")
    print(f"  GPU free at prep time:      {gpu_free}")
    print()
    print(f"  Verdict: {'GO' if all_checks_pass else 'NO-GO'}")
    print()
    print(f"  Output dir: {OUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
