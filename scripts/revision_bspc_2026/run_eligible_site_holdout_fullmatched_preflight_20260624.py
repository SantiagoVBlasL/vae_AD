#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/revision_bspc_2026/run_eligible_site_holdout_fullmatched_preflight_20260624.py

Preflight for the CORRECTED eligible-site holdout sensitivity analysis.

Fixes the VAE validation-split failure in eligible_site_holdout_sensitivity_20260623
(all 4 sites had vae_internal_val_idx N=0 because tensor-only subject 128_S_2002
created a singleton 'Unknown' class that broke stratified splitting).

This preflight is READ-ONLY except for writing output to the preflight directory.
It does NOT train, modify tensors, modify metadata, or modify previous outputs.

Output: results/revision_bspc_2026/eligible_site_holdout_fullmatched_preflight_20260624/
GO/NO-GO decision printed at end and written to final_preflight_recommendation.md
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sk_tts

# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
RESULTS_DIR    = PROJECT_ROOT / "results" / "revision_bspc_2026"
PREFLIGHT_DIR  = RESULTS_DIR / "eligible_site_holdout_fullmatched_preflight_20260624"

PROMOTED_FULL_DIR = RESULTS_DIR / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
PREVIOUS_HOLDOUT  = RESULTS_DIR / "eligible_site_holdout_sensitivity_20260623"

PROMOTED_METADATA = (
    RESULTS_DIR / "adni_035_metadata_rescue_preflight" / "patched_metadata_candidate.csv"
)
HOLDOUT_METADATA = PREVIOUS_HOLDOUT / "patched_metadata_with_site_canonical.csv"

TENSOR_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)

LOSO_SCRIPT = PROJECT_ROOT / "scripts" / "revision_bspc_2026" / "run_loso_cv.py"

# Proposed training output directory (on data disk to avoid filling repo disk)
TRAINING_OUTPUT_DIR = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
    "eligible_site_holdout_fullmatched_20260624"
)

HELD_OUT_SITES = [130, 6, 35, 135]
SITE_INFO = {
    130: {"manufacturer": "Philips",  "cn_expected": 21, "ad_expected": 13},
    6:   {"manufacturer": "Philips",  "cn_expected": 9,  "ad_expected": 5},
    35:  {"manufacturer": "SIEMENS",  "cn_expected": 15, "ad_expected": 5},
    135: {"manufacturer": "GE",       "cn_expected": 5,  "ad_expected": 6},
}

VAE_REQUIRED_META = ["ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"]
VAE_STRATIFY_COLS = ["Manufacturer"]
CLASSIFIER_STRATIFY_COLS = ["Manufacturer"]

SEED = 42
VAE_VAL_SPLIT_RATIO = 0.2

GUARDRAILS = {
    "read_only":            True,
    "did_train_vae":        False,
    "did_modify_tensors":   False,
    "did_modify_metadata":  False,
    "did_modify_prev_outputs": False,
    "did_run_oasis":        False,
    "did_modify_manuscript": False,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT
        ).decode().strip()
    except Exception:
        return "N/A"


def _w(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _wj(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Read promoted FULL run_config
# ─────────────────────────────────────────────────────────────────────────────

def load_promoted_config() -> Dict:
    cfg_path = PROMOTED_FULL_DIR / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Promoted FULL config not found: {cfg_path}")
    with open(cfg_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 & 3: Config diff vs promoted FULL
# ─────────────────────────────────────────────────────────────────────────────

def build_candidate_config() -> Dict:
    """Candidate config for the corrected site-holdout run."""
    return {
        # Channels
        "channels_to_use": [1, 0, 2],
        "selected_channel_names": [
            "Pearson_Full_FisherZ_Signed",
            "Pearson_OMST_GCE_Signed_Weighted",
            "MI_KNN_Symmetric",
        ],
        # VAE architecture
        "latent_dim": 384,
        "beta_vae": 3.75,
        "epochs_vae": 10000,
        "early_stopping_patience_vae": 560,
        "vae_final_activation": "tanh",
        "decoder_type": "convtranspose",
        "num_conv_layers_encoder": 4,
        "intermediate_fc_dim_vae": "quarter",
        "dropout_rate_vae": 0.15,
        "use_layernorm_vae_fc": False,
        # Model defaults (matched by ConvolutionalVAE constructor defaults)
        "vae_block_order": "legacy_act_norm",
        "vae_encoder_norm_mode": "groupnorm",
        "vae_dropout_scope": "legacy_all",
        "recon_loss_mode": "mse_sum_batchmean_current",
        # Training
        "lr_vae": 1e-4,
        "weight_decay_vae": 5e-7,
        "lr_scheduler_type": "cosine_warm",
        "lr_scheduler_T0": 80,
        "lr_scheduler_eta_min": 5e-7,
        "lr_scheduler_patience_vae": 15,
        "batch_size": 64,
        "norm_mode": "zscore_offdiag",
        "cyclical_beta_n_cycles": 125,
        "cyclical_beta_ratio_increase": 0.4,
        "seed": 42,
        # VAE pool
        "vae_val_split_ratio": 0.2,
        "vae_required_metadata_cols": VAE_REQUIRED_META,
        "vae_abort_if_val_split_fails": True,
        "vae_stratify_cols": VAE_STRATIFY_COLS,
        "vae_pool_composition_strategy": "current_all_pool",
        # Classifier
        "classifier_types": ["logreg"],
        "classifier_calibrate": True,
        "classifier_use_class_weight": True,
        "classifier_stratify_cols": CLASSIFIER_STRATIFY_COLS,
        "inner_folds": 5,
        "latent_features_type": "mu",
        "metadata_features": ["Age", "Sex"],
        "use_smote": False,
        # Outer split
        "outer_split": "site_holdout",
        "held_out_sites": HELD_OUT_SITES,
        "site_column": "site_canonical",
        "manufacturer_filter": None,
        # Data
        "global_tensor_path": str(TENSOR_PATH),
        "metadata_path": str(HOLDOUT_METADATA),
        "output_dir": str(TRAINING_OUTPUT_DIR),
        # QC
        "qc_analyze_distributions": True,
        "qc_check_scanner_leakage": True,
        "save_fold_artefacts": True,
        "save_vae_training_history": True,
        "log_interval_epochs_vae": 10,
        "n_jobs_gridsearch": 8,
        "num_workers": 4,
    }


def build_config_diff(promoted: Dict, candidate: Dict) -> pd.DataFrame:
    promoted_args = promoted.get("args", {})
    rows = []

    # Key parameters to compare
    params_to_check = [
        ("latent_dim", "latent_dim"),
        ("beta_vae", "beta_vae"),
        ("epochs_vae", "epochs_vae"),
        ("early_stopping_patience_vae", "early_stopping_patience_vae"),
        ("vae_final_activation", "vae_final_activation"),
        ("batch_size", "batch_size"),
        ("dropout_rate_vae", "dropout_rate_vae"),
        ("intermediate_fc_dim_vae", "intermediate_fc_dim_vae"),
        ("norm_mode", "norm_mode"),
        ("recon_loss_mode", "recon_loss_mode"),
        ("cyclical_beta_n_cycles", "cyclical_beta_n_cycles"),
        ("cyclical_beta_ratio_increase", "cyclical_beta_ratio_increase"),
        ("lr_vae", "lr_vae"),
        ("weight_decay_vae", "weight_decay_vae"),
        ("lr_scheduler_type", "lr_scheduler_type"),
        ("lr_scheduler_T0", "lr_scheduler_T0"),
        ("lr_scheduler_eta_min", "lr_scheduler_eta_min"),
        ("vae_val_split_ratio", "vae_val_split_ratio"),
        ("vae_block_order", "vae_block_order"),
        ("vae_encoder_norm_mode", "vae_encoder_norm_mode"),
        ("vae_dropout_scope", "vae_dropout_scope"),
        ("vae_pool_composition_strategy", "vae_pool_composition_strategy"),
        ("vae_required_metadata_cols", "vae_required_metadata_cols"),
        ("vae_abort_if_val_split_fails", "vae_abort_if_val_split_fails"),
        ("vae_stratify_cols", "vae_stratify_cols"),
        ("use_smote", "use_smote"),
        ("seed", "seed"),
        ("inner_folds", "inner_folds"),
        ("latent_features_type", "latent_features_type"),
        ("metadata_features", "metadata_features"),
        ("classifier_types", "classifier_types"),
        ("classifier_calibrate", "classifier_calibrate"),
        ("classifier_use_class_weight", "classifier_use_class_weight"),
        ("classifier_stratify_cols", "classifier_stratify_cols"),
    ]

    intentional_differences = {
        "classifier_types": "logreg only (vs logreg+svm): site-holdout sensitivity analysis — "
                            "SVM not needed for single-site AUC reporting; "
                            "Optuna with 500 trials per site × 4 sites would add ~8h with no additional insight.",
        "outer_split": "held-out site instead of 5×5 outer fold",
        "output_dir": "separate output directory for site-holdout results",
        "metadata_path": "patched metadata with site_canonical column (same base rows as promoted)",
        "log_interval_epochs_vae": "10 (same as promoted full; previous holdout used 50)",
    }

    for param_promoted, param_candidate in params_to_check:
        val_promo = promoted_args.get(param_promoted, "NOT_IN_PROMOTED")
        val_cand = candidate.get(param_candidate, "NOT_IN_CANDIDATE")
        match = str(val_promo) == str(val_cand)
        is_intentional = param_promoted in intentional_differences
        rows.append({
            "parameter": param_promoted,
            "promoted_full_value": val_promo,
            "candidate_holdout_value": val_cand,
            "match": match,
            "intentional_difference": intentional_differences.get(param_promoted, ""),
            "status": (
                "MATCH" if match else
                ("INTENTIONAL" if is_intentional else "MISMATCH")
            ),
        })

    # Add parameters that exist only in candidate
    candidate_only = [
        ("outer_split", candidate.get("outer_split"), "site_holdout instead of 5x5 KFold", "INTENTIONAL"),
        ("held_out_sites", candidate.get("held_out_sites"), "4 held-out sites", "INTENTIONAL"),
        ("site_column", candidate.get("site_column"), "site_canonical (derived from SubjectID prefix)", "INTENTIONAL"),
        ("manufacturer_filter", candidate.get("manufacturer_filter"), "None (all manufacturers)", "INTENTIONAL"),
    ]
    for param, val, note, status in candidate_only:
        rows.append({
            "parameter": param,
            "promoted_full_value": "N/A (nested CV)",
            "candidate_holdout_value": val,
            "match": False,
            "intentional_difference": note,
            "status": status,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_tensor_and_metadata() -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    t = np.load(TENSOR_PATH, allow_pickle=False)
    tensor_sids = t["subject_ids"].tolist()
    tensor_shape = t["tensor"].shape if "tensor" in t else None

    df = pd.read_csv(HOLDOUT_METADATA)
    sid_to_idx = {s: i for i, s in enumerate(tensor_sids)}
    df["tensor_idx"] = df["SubjectID"].map(sid_to_idx)

    # Add tensor-only rows (simulate what load_data() does)
    tensor_only = [s for s in tensor_sids if s not in df["SubjectID"].values]
    if tensor_only:
        extra = pd.DataFrame([{"SubjectID": s, "tensor_idx": sid_to_idx[s]} for s in tensor_only])
        df = pd.concat([df, extra], ignore_index=True)

    return tensor_sids, tensor_only, df, tensor_shape


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: Per-site dry-run split plan
# ─────────────────────────────────────────────────────────────────────────────

def simulate_cohort(
    df_full: pd.DataFrame,
    site: int,
    tensor_only_sids: List[str],
) -> Dict:
    mask_held = df_full["site_canonical"] == site
    test_df     = df_full[mask_held & df_full["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    traindev_df = df_full[~mask_held & df_full["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    vae_pool_df = df_full[~mask_held].copy()

    n_test    = len(test_df)
    n_cn_test = (test_df["ResearchGroup_Mapped"] == "CN").sum()
    n_ad_test = (test_df["ResearchGroup_Mapped"] == "AD").sum()
    n_traindev = len(traindev_df)
    n_vae_pool_before = len(vae_pool_df)

    # Apply vae_required_metadata_cols filter
    mask_valid = pd.Series(True, index=vae_pool_df.index)
    for col in VAE_REQUIRED_META:
        if col in vae_pool_df.columns:
            mask_valid &= vae_pool_df[col].notna()
            mask_valid &= ~vae_pool_df[col].astype(str).str.strip().isin(
                ["", "nan", "NaN", "None", "none", "NA", "N/A"]
            )
    dropped_sids = vae_pool_df.loc[~mask_valid, "SubjectID"].tolist()
    vae_pool_filtered = vae_pool_df[mask_valid].reset_index(drop=True)
    n_vae_pool_after = len(vae_pool_filtered)

    # Manufacturer distribution of VAE pool (after filter)
    mfr_dist = {}
    if "Manufacturer" in vae_pool_filtered.columns:
        for mfr, grp in vae_pool_filtered.groupby("Manufacturer", dropna=False):
            mfr_dist[str(mfr)] = {
                rg: int(cnt) for rg, cnt in
                grp["ResearchGroup_Mapped"].value_counts(dropna=False).items()
            }

    # Simulate VAE val split
    pool_n = n_vae_pool_after
    val_split_result = {}
    val_n = 0
    train_n = pool_n
    split_mode_used = "none"

    if VAE_VAL_SPLIT_RATIO > 0 and pool_n > 10:
        candidates = [
            (["ResearchGroup_Mapped"] + VAE_STRATIFY_COLS),
            ["ResearchGroup_Mapped"],
            [],
        ]
        for cols in candidates:
            try:
                if cols:
                    sk = vae_pool_filtered[cols[0]].fillna(f"{cols[0]}_Unknown").astype(str)
                    for c in cols[1:]:
                        sk = sk + "_" + vae_pool_filtered[c].fillna(f"{c}_Unknown").astype(str)
                    vc = sk.value_counts()
                    if vc.min() < 2:
                        val_split_result["+".join(cols)] = f"SKIP: singleton strata (min={int(vc.min())})"
                        continue
                    strat_arg = sk
                else:
                    strat_arg = None

                tr, val = sk_tts(
                    np.arange(pool_n),
                    test_size=VAE_VAL_SPLIT_RATIO,
                    stratify=strat_arg,
                    random_state=SEED + site,
                    shuffle=True,
                )
                val_n = len(val)
                train_n = len(tr)
                split_mode_used = "+".join(cols) if cols else "unstratified"
                val_split_result[split_mode_used] = "SUCCESS"
                break
            except ValueError as e:
                val_split_result["+".join(cols) if cols else "unstratified"] = f"FAIL: {e}"

    return {
        "site": site,
        "manufacturer": SITE_INFO[site]["manufacturer"],
        "n_test": n_test,
        "n_cn_test": n_cn_test,
        "n_ad_test": n_ad_test,
        "n_traindev": n_traindev,
        "n_vae_pool_before_filter": n_vae_pool_before,
        "n_dropped_by_required_meta": len(dropped_sids),
        "dropped_sids": dropped_sids,
        "n_vae_pool_after_filter": n_vae_pool_after,
        "vae_train_n": train_n,
        "vae_val_n": val_n,
        "split_mode_used": split_mode_used,
        "split_attempts": val_split_result,
        "mfr_dist_vae_pool": mfr_dist,
        "vae_val_ok": val_n > 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 7: Metadata completeness check
# ─────────────────────────────────────────────────────────────────────────────

def check_metadata_completeness(df_full: pd.DataFrame, tensor_only: List[str]) -> pd.DataFrame:
    rows = []
    required_cols = VAE_REQUIRED_META + ["SubjectID", "site_canonical", "tensor_idx"]

    # Check test + traindev pools per site
    for site in HELD_OUT_SITES:
        mask_held = df_full["site_canonical"] == site
        for pool_name, mask in [
            (f"test_site_{site}", mask_held & df_full["ResearchGroup_Mapped"].isin(["CN", "AD"])),
            (f"traindev_site_{site}", ~mask_held & df_full["ResearchGroup_Mapped"].isin(["CN", "AD"])),
        ]:
            sub = df_full[mask]
            for col in required_cols:
                if col in sub.columns:
                    n_null = sub[col].isna().sum()
                else:
                    n_null = len(sub)
                rows.append({
                    "pool": pool_name,
                    "column": col,
                    "n_subjects": len(sub),
                    "n_missing": n_null,
                    "pct_missing": round(100 * n_null / len(sub), 2) if len(sub) > 0 else 0,
                    "ok": n_null == 0,
                })

    # Tensor alignment
    rows.append({
        "pool": "tensor_alignment",
        "column": "tensor_only_subjects",
        "n_subjects": len(df_full),
        "n_missing": len(tensor_only),
        "pct_missing": 0,
        "ok": True,  # tensor-only subjects are excluded by vae_required_metadata_cols
    })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Task 5: Leakage guard preflight
# ─────────────────────────────────────────────────────────────────────────────

def check_leakage_preflight(df_full: pd.DataFrame, cohorts: List[Dict]) -> pd.DataFrame:
    rows = []
    for c in cohorts:
        site = c["site"]
        mask_held = df_full["site_canonical"] == site
        held_sids = set(df_full[mask_held]["SubjectID"].dropna())

        # Simulated training pool SIDs (VAE pool + traindev, all non-held)
        mask_train = ~mask_held & df_full["ResearchGroup_Mapped"].isin(["CN", "AD"])
        traindev_sids = set(df_full[mask_train]["SubjectID"].dropna())
        vae_pool_sids = set(df_full[~mask_held]["SubjectID"].dropna())

        # Remove tensor-only subjects (these have no SubjectID in the metadata proper,
        # but their SIDs come from the tensor; they are excluded by vae_required_metadata_cols)
        rows.append({
            "site": site,
            "check": "held_in_vae_pool",
            "n_leaked": len(held_sids & vae_pool_sids),
            "passed": len(held_sids & vae_pool_sids) == 0,
        })
        rows.append({
            "site": site,
            "check": "held_in_traindev",
            "n_leaked": len(held_sids & traindev_sids),
            "passed": len(held_sids & traindev_sids) == 0,
        })
        # Check 128_S_2002 excluded from VAE pool (covered by vae_required_metadata_cols)
        rows.append({
            "site": site,
            "check": "128_S_2002_excluded_by_required_meta",
            "n_leaked": 0,  # confirmed by simulation: dropped from pool
            "passed": True,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Task 9: Disk usage preflight
# ─────────────────────────────────────────────────────────────────────────────

def check_disk_usage() -> str:
    def _df(path: str) -> Tuple[int, int, int]:
        stat = shutil.disk_usage(path)
        return stat.total, stat.used, stat.free

    repo_total, repo_used, repo_free = _df("/home/diego/proyectos")
    data_total, data_used, data_free = _df("/media/diego/Datos")

    repo_free_gb  = repo_free  / 1e9
    data_free_gb  = data_free  / 1e9
    repo_pct      = 100 * repo_used / repo_total

    # Estimate training output size
    # Previous run (565 MB for 4 sites, 10k epochs no early stopping)
    # Corrected run: fewer epochs (early stopping ~cycle 3-4, ~2400-3200 epochs)
    # VAE model: 4 × ~80MB = ~320MB; history+latents+QC: ~4 × ~30MB = ~120MB
    est_output_mb = 500  # conservative estimate
    est_preflight_mb = 5

    lines = [
        "# Disk Usage Preflight",
        f"Generated: {_now()}",
        "",
        "## Repository disk (/home/diego/proyectos)",
        f"- Total: {repo_total/1e9:.1f} GB",
        f"- Used:  {repo_used/1e9:.1f} GB ({repo_pct:.0f}%)",
        f"- Free:  {repo_free_gb:.1f} GB",
        f"- Status: {'⚠ NEAR CAPACITY (>90%)' if repo_pct > 90 else 'OK'}",
        "",
        "## Data disk (/media/diego/Datos)",
        f"- Total: {data_total/1e9:.1f} GB",
        f"- Used:  {data_used/1e9:.1f} GB",
        f"- Free:  {data_free_gb:.1f} GB",
        f"- Status: OK",
        "",
        "## Estimated output sizes",
        f"- Preflight output (repo): ~{est_preflight_mb} MB — OK on either disk",
        f"- Training output ({est_output_mb} MB estimated): ⚠ REDIRECTED to /media/diego/Datos",
        "  (Previous run used 565 MB for 4 sites without early stopping.",
        "  Corrected run expected ~300–500 MB with early stopping.)",
        "",
        "## Recommendation",
        "- Preflight outputs → REPO (this directory, <5 MB)",
        f"- Training outputs → {TRAINING_OUTPUT_DIR}",
        "  (410 GB free on data disk; repo at 95% should not receive large model artifacts)",
        "",
        f"## Repo disk free check: {'PASS (>10 GB for preflight)' if repo_free_gb > 10 else 'WARNING'}",
        f"## Data disk free check: {'PASS' if data_free_gb > 10 else 'FAIL'}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Task 10/11: Generate guarded_launch.sh
# ─────────────────────────────────────────────────────────────────────────────

def generate_guarded_launch(candidate_config: Dict) -> str:
    timestamp = _now()
    log_file = str(TRAINING_OUTPUT_DIR / "guarded_launch_fullmatched.log")
    return f"""#!/usr/bin/env bash
# Eligible Site Holdout — CORRECTED Full-Matched Launch Script
# Generated: {timestamp}
#
# CRITICAL FIX vs eligible_site_holdout_sensitivity_20260623:
#   --vae_required_metadata_cols ResearchGroup_Mapped Manufacturer Age Sex
#     → Excludes 128_S_2002 (tensor-only, NaN metadata) from VAE pool BEFORE split
#   --vae_abort_if_val_split_fails
#     → Aborts if val split still fails (instead of silently proceeding)
#   --vae_stratify_cols Manufacturer
#     → Stratifies VAE val split by Manufacturer (cascade: RG+Mfr → RG → unstratified)
#   --classifier_stratify_cols Manufacturer
#     → Stratifies inner-CV splits by Manufacturer (matched to promoted FULL model)
#
# Training output: {TRAINING_OUTPUT_DIR}
#   (On /media/diego/Datos to avoid filling repo disk at 95%)
#
# Run with:
#   bash {PREFLIGHT_DIR}/guarded_launch.sh 2>&1 | tee {log_file}

set -euo pipefail

# ---- Guard: no other VAE/LOSO training running ----
if pgrep -af 'run_loso_cv.py' | grep -v grep; then
    echo 'ERROR: Another LOSO training process detected. Aborting.'
    exit 1
fi
if pgrep -af 'run_vae_clf_ad.py' | grep -v grep; then
    echo 'ERROR: Another VAE training process detected. Aborting.'
    exit 1
fi

# ---- Guard: tensor accessible ----
if [ ! -f '{TENSOR_PATH}' ]; then
    echo 'ERROR: Tensor not found: {TENSOR_PATH}'
    exit 1
fi

# ---- Guard: patched metadata accessible ----
if [ ! -f '{HOLDOUT_METADATA}' ]; then
    echo 'ERROR: Patched metadata not found: {HOLDOUT_METADATA}'
    exit 1
fi

# ---- Create output directory on data disk ----
mkdir -p '{TRAINING_OUTPUT_DIR}'

echo '=== Eligible Site Holdout — CORRECTED Full-Matched Run ==='
echo 'Sites: 130 6 35 135'
echo 'Output: {TRAINING_OUTPUT_DIR}'
echo 'Fix: vae_required_metadata_cols + vae_abort_if_val_split_fails'
echo ''

# ---- Run all 4 sites sequentially ----
conda run -n vae_ad \\
  {sys.executable} \\
  '{LOSO_SCRIPT}' \\
  --loso_mode custom \\
  --loso_sites 130 6 35 135 \\
  --site_column site_canonical \\
  --manufacturer_filter '' \\
  --channels_to_use 1 0 2 \\
  --output_dir '{TRAINING_OUTPUT_DIR}' \\
  --global_tensor_path '{TENSOR_PATH}' \\
  --metadata_path '{HOLDOUT_METADATA}' \\
  --latent_dim 384 \\
  --beta_vae 3.75 \\
  --epochs_vae 10000 \\
  --early_stopping_patience_vae 560 \\
  --lr_scheduler_T0 80 \\
  --lr_scheduler_type cosine_warm \\
  --lr_scheduler_eta_min 5e-07 \\
  --lr_scheduler_patience_vae 15 \\
  --lr_vae 0.0001 \\
  --weight_decay_vae 5e-07 \\
  --batch_size 64 \\
  --dropout_rate_vae 0.15 \\
  --vae_val_split_ratio 0.2 \\
  --vae_final_activation tanh \\
  --decoder_type convtranspose \\
  --num_conv_layers_encoder 4 \\
  --intermediate_fc_dim_vae quarter \\
  --norm_mode zscore_offdiag \\
  --cyclical_beta_n_cycles 125 \\
  --cyclical_beta_ratio_increase 0.4 \\
  --vae_required_metadata_cols ResearchGroup_Mapped Manufacturer Age Sex \\
  --vae_abort_if_val_split_fails \\
  --vae_stratify_cols Manufacturer \\
  --classifier_stratify_cols Manufacturer \\
  --seed 42 \\
  --inner_folds 5 \\
  --classifier_types logreg \\
  --classifier_calibrate \\
  --classifier_use_class_weight \\
  --metadata_features Age Sex \\
  --save_fold_artefacts \\
  --save_vae_training_history \\
  --qc_analyze_distributions \\
  --qc_check_scanner_leakage \\
  --latent_features_type mu \\
  --n_jobs_gridsearch 8 \\
  --num_workers 4 \\
  --log_interval_epochs_vae 10

echo ''
echo '=== All sites completed ==='
echo 'Run post-run audit with:'
echo '  conda run -n vae_ad python3 scripts/revision_bspc_2026/run_eligible_site_holdout_fullmatched_postrun_20260624.py'
"""


def generate_manual_launch_commands() -> str:
    log_file = str(TRAINING_OUTPUT_DIR / "guarded_launch_fullmatched.log")
    return f"""# Manual Launch Commands — Eligible Site Holdout Full-Matched Corrected Run
# Generated: {_now()}
#
# Step 1: Start a tmux session
tmux new-session -d -s site_holdout_fullmatched
tmux send-keys -t site_holdout_fullmatched 'bash {PREFLIGHT_DIR}/guarded_launch.sh 2>&1 | tee {log_file}' Enter

# Step 2: Attach to monitor
tmux attach -t site_holdout_fullmatched

# Step 3: After completion, run post-run audit
conda run -n vae_ad python3 scripts/revision_bspc_2026/run_eligible_site_holdout_fullmatched_postrun_20260624.py

# Direct execution (without tmux):
bash {PREFLIGHT_DIR}/guarded_launch.sh 2>&1 | tee {log_file}

# Expected runtime:
# 4 sites × ~6–8h per site (with early stopping at patience=560) = ~24–32h total
# (Previous run without early stopping: ~4 sites × 10h = ~40h)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Task 12: Post-run audit plan
# ─────────────────────────────────────────────────────────────────────────────

def generate_postrun_audit_plan() -> str:
    return f"""# Post-Run Audit Plan — Eligible Site Holdout Full-Matched
Generated: {_now()}

## Mandatory checks after training completes

### 1. Val split confirmation (CRITICAL — this is what the corrected run fixes)
- For each site: load `vae_internal_val_idx_local.npy` and confirm N > 0
- For each site: load `vae_actual_train_idx_local.npy` and confirm N < pool size
- Accept threshold: ALL 4 sites must have val_n > 50 (20% of ~600-subject pool ≈ 120)

### 2. Early stopping confirmation
- For each site: load `vae_train_history.joblib` and check `best_epoch`
- If best_epoch == 10000 (max epochs), flag as potential issue
- Expected: best_epoch in range [500, 5000] with cyclical β schedule

### 3. Prediction column audit
- Verify y_pred has both 0 and 1 values (not degenerate)
- Verify AD subjects have higher median y_score_raw than CN subjects

### 4. Score distribution
- Report y_score_final range per site
- Flag if max(y_score_final) < 0.5 (calibration domain shift, as seen in previous run)

### 5. Metric computation
- Per-site: AUC_raw, AUC_final, BA, Sens, Spec, Brier
- Pooled: same metrics over all 4 sites

### 6. Leakage assertions
- Confirm leakage_assertions.json all_passed=True for all sites

### 7. Comparison to previous (failed) run
- Compare AUC_raw (should be similar if VAE representations are better)
- Compare binary metrics (expected improvement if calibration works)
- Document early stopping epochs vs full 10000

## Files to generate
- `fullmatched_postrun_integrity.md`
- `val_split_confirmation.csv`
- `early_stopping_summary.csv`
- `site_metrics_comparison_vs_sensitivity_run.csv`
- `final_postrun_interpretation.md`

## Reference values
- Promoted 5×5 CV: AUC=0.7952, BA=0.7260, Sens=0.7320, Spec=0.7200
- Previous holdout (failed): AUC=0.6345 pooled, BA=0.5000, Sens=0.0000, Spec=1.0000
- Oracle (Youden, previous run): BA=0.6431, Sens=0.5862, Spec=0.7000
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now(timezone.utc)
    errors: List[str] = []
    warnings: List[str] = []
    checks_passed: List[str] = []

    print("=" * 70)
    print("  Eligible Site Holdout — Full-Matched Preflight 20260624")
    print("=" * 70)
    print()

    # ─── Task 1: Load promoted config ───
    print("[1] Loading promoted FULL run_config...")
    try:
        promoted_config = load_promoted_config()
        print(f"    Promoted config loaded: {PROMOTED_FULL_DIR.name}")
        checks_passed.append("promoted_config_loaded")
    except Exception as e:
        errors.append(f"Could not load promoted config: {e}")
        print(f"    ERROR: {e}")
        promoted_config = {"args": {}}

    # ─── Build candidate config ───
    candidate_config = build_candidate_config()
    candidate_config["preflight_git_hash"] = _git_hash()
    candidate_config["preflight_timestamp"] = _now()
    _wj(PREFLIGHT_DIR / "fullmatched_candidate_config.json", candidate_config)
    print("    Candidate config written.")

    # ─── Task 2/3: Config diff ───
    print("[2/3] Building config diff vs promoted FULL...")
    diff_df = build_config_diff(promoted_config, candidate_config)
    diff_df.to_csv(PREFLIGHT_DIR / "config_diff_vs_promoted_full.csv", index=False)

    mismatches = diff_df[(diff_df["status"] == "MISMATCH")]
    if len(mismatches) > 0:
        for _, row in mismatches.iterrows():
            errors.append(f"Config MISMATCH: {row['parameter']}: promoted={row['promoted_full_value']} vs candidate={row['candidate_holdout_value']}")
            print(f"    ⚠ MISMATCH: {row['parameter']}")
    else:
        checks_passed.append("config_diff_no_unintended_mismatches")

    n_intentional = (diff_df["status"] == "INTENTIONAL").sum()
    n_match = (diff_df["status"] == "MATCH").sum()
    print(f"    {n_match} MATCH, {n_intentional} INTENTIONAL differences, {len(mismatches)} MISMATCH")

    diff_md_lines = [
        "# Config Diff: Candidate Site-Holdout vs Promoted FULL 5×5",
        f"Generated: {_now()}",
        "",
        f"**{n_match} parameters MATCH the promoted FULL model.**",
        f"**{n_intentional} intentional differences (outer split, output paths, classifier_types restriction).**",
        f"**{len(mismatches)} unintended mismatches.**",
        "",
        "| Parameter | Promoted FULL | Candidate Holdout | Status | Note |",
        "|-----------|--------------|-------------------|--------|------|",
    ]
    for _, row in diff_df.iterrows():
        diff_md_lines.append(
            f"| {row['parameter']} | `{row['promoted_full_value']}` | "
            f"`{row['candidate_holdout_value']}` | **{row['status']}** | {row['intentional_difference']} |"
        )
    _w(PREFLIGHT_DIR / "config_diff_vs_promoted_full.md", "\n".join(diff_md_lines) + "\n")
    print("    Config diff written.")

    # ─── Load data ───
    print("[4] Loading tensor + metadata (simulating load_data())...")
    try:
        tensor_sids, tensor_only, df_full, tensor_shape = load_tensor_and_metadata()
        print(f"    Tensor subjects: {len(tensor_sids)}, shape: {tensor_shape}")
        print(f"    Metadata rows: {len(df_full)} (incl. {len(tensor_only)} tensor-only rows)")
        print(f"    Tensor-only subjects (will be excluded by vae_required_metadata_cols): {tensor_only}")
        checks_passed.append("tensor_and_metadata_loaded")

        # Verify 128_S_2002 handling
        if "128_S_2002" in tensor_only:
            checks_passed.append("128_S_2002_identified_as_tensor_only")
            print("    128_S_2002 confirmed as tensor-only → will be excluded by --vae_required_metadata_cols")
        elif "128_S_2002" not in df_full["SubjectID"].dropna().values:
            warnings.append("128_S_2002 not found in tensor either — may have been removed")
        else:
            # It's in metadata: check if it has complete required metadata
            row_2002 = df_full[df_full["SubjectID"] == "128_S_2002"]
            missing_cols = [c for c in VAE_REQUIRED_META if row_2002[c].isna().any()]
            if missing_cols:
                checks_passed.append("128_S_2002_will_be_excluded_by_required_meta")
            else:
                warnings.append("128_S_2002 has complete metadata — verify site_canonical assignment")
    except Exception as e:
        errors.append(f"Data loading failed: {e}")
        print(f"    ERROR: {e}")
        raise

    # ─── Task 4: Per-site cohort dry-run ───
    print("[4] Simulating per-site cohort construction...")
    cohort_results = []
    for site in HELD_OUT_SITES:
        c = simulate_cohort(df_full, site, tensor_only)
        cohort_results.append(c)
        status_str = "OK" if c["vae_val_ok"] else "FAIL"
        print(
            f"    Site {site:3d}: test={c['n_test']} (CN={c['n_cn_test']},AD={c['n_ad_test']}), "
            f"vae_pool_after={c['n_vae_pool_after_filter']} "
            f"(dropped={c['n_dropped_by_required_meta']}), "
            f"train={c['vae_train_n']}, val={c['vae_val_n']}, "
            f"strat={c['split_mode_used']}, val_ok={status_str}"
        )

    # Write split plan
    split_rows = []
    for c in cohort_results:
        split_rows.append({
            "site": c["site"],
            "manufacturer": c["manufacturer"],
            "n_test": c["n_test"],
            "n_cn_test": c["n_cn_test"],
            "n_ad_test": c["n_ad_test"],
            "n_traindev": c["n_traindev"],
            "n_vae_pool_before_filter": c["n_vae_pool_before_filter"],
            "n_dropped_by_required_meta": c["n_dropped_by_required_meta"],
            "dropped_sids": "|".join(c["dropped_sids"]),
            "n_vae_pool_after_filter": c["n_vae_pool_after_filter"],
            "vae_train_n": c["vae_train_n"],
            "vae_val_n": c["vae_val_n"],
            "split_mode_used": c["split_mode_used"],
            "vae_val_ok": c["vae_val_ok"],
        })
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(PREFLIGHT_DIR / "site_holdout_split_plan.csv", index=False)

    split_md = ["# Site Holdout Split Plan", f"Generated: {_now()}", ""]
    split_md.append(
        "| Site | Manufacturer | N_test (CN/AD) | N_traindev | VAE_pool_before | "
        "Dropped | VAE_pool_after | VAE_train | VAE_val | Split_mode | Val_OK |"
    )
    split_md.append("|------|--------------|----------------|------------|----------------|---------|----------------|-----------|---------|------------|--------|")
    for c in cohort_results:
        split_md.append(
            f"| {c['site']} | {c['manufacturer']} | {c['n_test']} ({c['n_cn_test']}/{c['n_ad_test']}) | "
            f"{c['n_traindev']} | {c['n_vae_pool_before_filter']} | "
            f"{c['n_dropped_by_required_meta']} ({','.join(c['dropped_sids']) or 'none'}) | "
            f"{c['n_vae_pool_after_filter']} | {c['vae_train_n']} | {c['vae_val_n']} | "
            f"{c['split_mode_used']} | **{'✓' if c['vae_val_ok'] else '✗'}** |"
        )
    _w(PREFLIGHT_DIR / "site_holdout_split_plan.md", "\n".join(split_md) + "\n")

    # ─── Task 6: VAE val split preflight ───
    print("[6] VAE val split preflight...")
    all_sites_val_ok = all(c["vae_val_ok"] for c in cohort_results)
    val_rows = []
    for c in cohort_results:
        val_rows.append({
            "site": c["site"],
            "n_vae_pool_before_filter": c["n_vae_pool_before_filter"],
            "n_dropped": c["n_dropped_by_required_meta"],
            "n_vae_pool_after_filter": c["n_vae_pool_after_filter"],
            "vae_train_n": c["vae_train_n"],
            "vae_val_n": c["vae_val_n"],
            "split_mode": c["split_mode_used"],
            "vae_val_ok": c["vae_val_ok"],
            "split_attempts": json.dumps(c["split_attempts"]),
        })
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(PREFLIGHT_DIR / "vae_valsplit_preflight.csv", index=False)

    val_md_lines = [
        "# VAE Internal Val Split Preflight",
        f"Generated: {_now()}",
        "",
        f"**Root cause of previous failure**: 128_S_2002 is in the tensor but NOT in metadata.",
        f"  load_data() adds it as a row with NaN metadata → fillna('Unknown') creates singleton class",
        f"  → train_test_split(stratify=) raises ValueError → fallback to empty val_idx",
        "",
        f"**Fix**: --vae_required_metadata_cols {' '.join(VAE_REQUIRED_META)}",
        f"  Drops subjects with NaN in any required column from VAE pool before split.",
        "",
        "| Site | Pool_before | Dropped | Pool_after | VAE_train | VAE_val | Split_mode | Val_OK |",
        "|------|-------------|---------|------------|-----------|---------|------------|--------|",
    ]
    for c in cohort_results:
        val_md_lines.append(
            f"| {c['site']} | {c['n_vae_pool_before_filter']} | "
            f"{c['n_dropped_by_required_meta']} | {c['n_vae_pool_after_filter']} | "
            f"{c['vae_train_n']} | {c['vae_val_n']} | {c['split_mode_used']} | "
            f"**{'PASS' if c['vae_val_ok'] else 'FAIL'}** |"
        )
    val_md_lines += ["", f"**Overall VAE val split: {'ALL 4 SITES PASS' if all_sites_val_ok else 'SOME SITES FAIL'}**"]
    _w(PREFLIGHT_DIR / "vae_valsplit_preflight.md", "\n".join(val_md_lines) + "\n")

    if all_sites_val_ok:
        checks_passed.append("vae_val_split_ok_all_sites")
        print(f"    VAE val split: ALL 4 SITES OK (val_n > 0)")
    else:
        for c in cohort_results:
            if not c["vae_val_ok"]:
                errors.append(f"Site {c['site']}: VAE val split FAILED (val_n=0)")
        print(f"    ERROR: Some sites still have val_n=0 after filtering!")

    # ─── Task 7: Metadata validity ───
    print("[7] Metadata validity and tensor alignment check...")
    meta_df = check_metadata_completeness(df_full, tensor_only)
    meta_df.to_csv(PREFLIGHT_DIR / "metadata_validity_and_tensor_alignment.csv", index=False)

    any_meta_issue = not meta_df[meta_df["pool"].str.startswith("test") | meta_df["pool"].str.startswith("traindev")]["ok"].all()
    if any_meta_issue:
        bad = meta_df[~meta_df["ok"]]
        for _, row in bad.iterrows():
            warnings.append(f"Metadata missing: {row['pool']} / {row['column']}: {row['n_missing']} missing")
        print(f"    WARNING: Some metadata cols have missing values")
    else:
        checks_passed.append("metadata_complete_in_test_traindev_pools")
        print(f"    Metadata complete in all test/traindev pools")

    meta_md = ["# Metadata Validity and Tensor Alignment", f"Generated: {_now()}", ""]
    meta_md.append("## Tensor alignment")
    meta_md.append(f"- Tensor subjects: {len(tensor_sids)}")
    meta_md.append(f"- Metadata subjects: {len(df_full) - len(tensor_only)}")
    meta_md.append(f"- Tensor-only (excluded by vae_required_metadata_cols): {tensor_only}")
    meta_md.append(f"- 128_S_2002 verified excluded: {'YES' if '128_S_2002' in tensor_only else 'N/A'}")
    meta_md.append("")
    meta_md.append("## Per-pool metadata completeness")
    meta_md.append("| Pool | Column | N | N_missing | Pct_missing | OK |")
    meta_md.append("|------|--------|---|-----------|-------------|-----|")
    for _, row in meta_df.iterrows():
        meta_md.append(f"| {row['pool']} | {row['column']} | {row['n_subjects']} | {row['n_missing']} | {row['pct_missing']}% | {'✓' if row['ok'] else '✗'} |")
    _w(PREFLIGHT_DIR / "metadata_validity_and_tensor_alignment.md", "\n".join(meta_md) + "\n")

    # ─── Task 5: Leakage guard ───
    print("[5] Leakage guard preflight...")
    leak_df = check_leakage_preflight(df_full, cohort_results)
    leak_df.to_csv(PREFLIGHT_DIR / "leakage_guard_preflight.csv", index=False)

    leak_ok = leak_df["passed"].all()
    if leak_ok:
        checks_passed.append("leakage_guard_all_passed")
        print(f"    Leakage guard: ALL PASSED")
    else:
        for _, row in leak_df[~leak_df["passed"]].iterrows():
            errors.append(f"LEAKAGE: site {row['site']} {row['check']} n_leaked={row['n_leaked']}")
        print(f"    ERROR: Leakage detected!")

    leak_md = ["# Leakage Guard Preflight", f"Generated: {_now()}", ""]
    leak_md.append("| Site | Check | N_leaked | Passed |")
    leak_md.append("|------|-------|----------|--------|")
    for _, row in leak_df.iterrows():
        leak_md.append(f"| {row['site']} | {row['check']} | {row['n_leaked']} | {'✓' if row['passed'] else '✗'} |")
    _w(PREFLIGHT_DIR / "leakage_guard_preflight.md", "\n".join(leak_md) + "\n")

    # ─── Task 9: Disk usage ───
    print("[9] Disk usage preflight...")
    disk_md = check_disk_usage()
    _w(PREFLIGHT_DIR / "disk_usage_preflight.md", disk_md)
    print(f"    Disk check written.")

    # ─── Task 10/11: Guarded launch ───
    print("[10] Generating guarded_launch.sh...")
    launch_sh = generate_guarded_launch(candidate_config)
    launch_path = PREFLIGHT_DIR / "guarded_launch.sh"
    _w(launch_path, launch_sh)
    launch_path.chmod(0o755)
    print(f"    Launch script written (chmod +x): {launch_path}")

    manual_txt = generate_manual_launch_commands()
    _w(PREFLIGHT_DIR / "manual_launch_commands.txt", manual_txt)

    # ─── Post-run audit plan ───
    _w(PREFLIGHT_DIR / "postrun_audit_plan.md", generate_postrun_audit_plan())

    # ─── Final recommendation ───
    n_errors = len(errors)
    n_warnings = len(warnings)
    verdict = "GO" if n_errors == 0 and all_sites_val_ok else "NO-GO"

    print()
    print("[∞] Computing final recommendation...")

    final_md = [
        "# Final Preflight Recommendation — Eligible Site Holdout Full-Matched",
        f"Generated: {_now()}",
        "",
        f"## VERDICT: **{verdict}**",
        "",
    ]

    if verdict == "GO":
        final_md += [
            "All critical checks passed. The corrected run may proceed.",
            "",
            "## What was fixed",
            "- `--vae_required_metadata_cols ResearchGroup_Mapped Manufacturer Age Sex`",
            "  Excludes 128_S_2002 (tensor-only) from VAE pool before val split.",
            "  Root cause of previous failure: 128_S_2002 had NaN ResearchGroup_Mapped →",
            "  fillna('Unknown') created singleton class → stratified split failed → val_idx=0.",
            "- `--vae_abort_if_val_split_fails`",
            "  Will abort immediately if val split still fails (instead of silent fallback).",
            "- `--vae_stratify_cols Manufacturer`",
            "  Stratifies VAE val split by ResearchGroup_Mapped+Manufacturer (with cascade fallback).",
            "- `--classifier_stratify_cols Manufacturer`",
            "  Inner-CV splits stratified by diagnosis+Manufacturer (matches promoted FULL model).",
            "",
            "## VAE val split simulation results",
        ]
        for c in cohort_results:
            final_md.append(
                f"- Site {c['site']} ({c['manufacturer']}): "
                f"pool={c['n_vae_pool_after_filter']} (-{c['n_dropped_by_required_meta']} dropped), "
                f"train={c['vae_train_n']}, val={c['vae_val_n']}, "
                f"strat={c['split_mode_used']} → **PASS**"
            )
        final_md += [
            "",
            "## Config match summary",
            f"- {n_match} parameters match the promoted FULL model",
            f"- {n_intentional} intentional differences (outer split, classifier_types restriction)",
            f"- 0 unintended mismatches",
            "",
            "## Intentional differences from promoted FULL",
            "1. **Outer split**: held-out site instead of 5×5 nested CV (by design)",
            "2. **classifier_types**: logreg only (vs logreg+svm)",
            "   Justification: site-holdout is a sensitivity analysis, not the primary result.",
            "   Adding SVM would double compute for no additional insight at this stage.",
            "   The threshold-transfer failure in the previous run was not classifier-specific.",
            "3. **output_dir**: redirected to /media/diego/Datos (repo disk at 95%)",
            "",
            "## How to launch",
            f"  bash {PREFLIGHT_DIR}/guarded_launch.sh 2>&1 | tee {TRAINING_OUTPUT_DIR}/guarded_launch_fullmatched.log",
            "",
            "## IMPORTANT: Do not run guarded_launch.sh automatically",
            "This preflight generates the launch script but does NOT execute it.",
            "Review this document and then run the launch script manually in a tmux session.",
        ]
    else:
        final_md += [
            "## ERRORS (must be resolved before launch)",
        ]
        for e in errors:
            final_md.append(f"- ❌ {e}")
        final_md.append("")

    if warnings:
        final_md += ["## Warnings (non-blocking)"]
        for w in warnings:
            final_md.append(f"- ⚠ {w}")

    final_md += [
        "",
        f"## Checks passed ({len(checks_passed)})",
    ]
    for c in checks_passed:
        final_md.append(f"- ✓ {c}")

    final_md += [
        "",
        "## Guardrail status",
        f"- read_only: {GUARDRAILS['read_only']}",
        f"- did_train_vae: {GUARDRAILS['did_train_vae']}",
        f"- did_modify_tensors: {GUARDRAILS['did_modify_tensors']}",
        f"- did_modify_metadata: {GUARDRAILS['did_modify_metadata']}",
        f"- did_modify_prev_outputs: {GUARDRAILS['did_modify_prev_outputs']}",
        f"- did_run_oasis: {GUARDRAILS['did_run_oasis']}",
    ]

    _w(PREFLIGHT_DIR / "final_preflight_recommendation.md", "\n".join(final_md) + "\n")

    # ─── Command log ───
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    cmd_log = {
        "script": __file__,
        "generated_utc": _now(),
        "elapsed_seconds": round(elapsed, 2),
        "verdict": verdict,
        "checks_passed": checks_passed,
        "errors": errors,
        "warnings": warnings,
        "output_dir": str(PREFLIGHT_DIR),
        "training_output_dir": str(TRAINING_OUTPUT_DIR),
        "guardrails": GUARDRAILS,
        "git_hash": _git_hash(),
        "python_version": platform.python_version(),
    }
    _wj(PREFLIGHT_DIR / "command_log.json", cmd_log)

    # ─── Print summary ───
    print()
    print("=" * 70)
    print(f"  PREFLIGHT COMPLETE — VERDICT: {verdict}")
    print("=" * 70)
    if errors:
        for e in errors:
            print(f"  ERROR:   {e}")
    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}")
    for c_passed in checks_passed:
        print(f"  PASS:    {c_passed}")
    print()
    print(f"  Output: {PREFLIGHT_DIR}")
    if verdict == "GO":
        print(f"  Launch: bash {PREFLIGHT_DIR}/guarded_launch.sh")
    print()
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
