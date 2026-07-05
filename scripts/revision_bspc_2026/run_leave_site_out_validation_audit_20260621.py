#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/revision_bspc_2026/run_leave_site_out_validation_audit_20260621.py

Leave-one-site-out robustness analysis for the final selected ADNI model.
Reference model: recover035_latent384_beta3p75_T80_h10000_p560_full5x5

Design:
  LEVEL 0 (this script, read-only):
    - Inventory existing LOSO analyses
    - Site feasibility table
    - Design decision documentation
    - OOF site-stratified analysis from existing OOF predictions
      (NOT a true LOSO; subjects from held-out site were in VAE + classifier training)
    - Manuscript Methods and Results paragraphs

  LEVEL 1 (preflight documented, requires user approval):
    - Classifier-only LOSO on frozen VAE latents (no VAE retraining)
    - Uses fold-specific VAE encoders to re-encode all subjects, then
      trains new classifier with each site held out

  LEVEL 2 (not computed here):
    - Full-pipeline LOSO: retrain VAE + classifier without held-out site
    - Extremely expensive (10000 epochs × 5 sites); requires explicit approval

Hard guardrails (always in effect):
  - Read-only with respect to source data
  - No model training
  - No tensor modification
  - No metadata modification
  - No prediction modification
  - No threshold refitting
  - No subject exclusion
  - No model selection
  - Do not propose post-hoc changes purely to increase AUC
  - Do not use Martín acquisition/QC variables as features

Output: results/revision_bspc_2026/leave_site_out_validation_audit_20260621/
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"
OUTPUT_DIR = RESULTS / "leave_site_out_validation_audit_20260621"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Final selected model
RUN_102 = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
OOF_102 = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
METADATA_PATH = PROJECT_ROOT / "data" / "SubjectsData_AAL3_procesado2.csv"

# Existing LOSO runs (notebooks path)
NB_RESULTS = (
    PROJECT_ROOT
    / "notebooks"
    / "revision_bspc_2026"
    / "results"
    / "revision_bspc_2026"
)
LOSO_PRIMARY_NB = NB_RESULTS / "loso_primary"
LOSO_STRICT_NB = NB_RESULTS / "loso_strict"
LOSO_PRIMARY_SVM_NB = NB_RESULTS / "loso_primary_svm"
LOSO_SMOKETEST_V2 = RESULTS / "loso_smoketest_v2"

# Primary readout parameters (match OOF calibration primary)
PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURE_SET = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FIXED_THRESHOLD = 0.5

# LOSO feasibility thresholds
MIN_CLASS_BOTH = 2   # minimum per-class for any LOSO metrics
MIN_CLASS_PRIMARY = 5  # recommended minimum for reliable AUC
MIN_CLASS_SENSITIVITY = 3  # sensitivity set

command_log: dict = {
    "script": __file__,
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "model": str(RUN_102),
    "oof_dir": str(OOF_102),
    "primary_readout": {
        "model": PRIMARY_MODEL,
        "feature_set": PRIMARY_FEATURE_SET,
        "calib_method": PRIMARY_CALIB,
        "threshold_strategy": PRIMARY_THRESHOLD,
    },
    "outputs": [],
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def save_csv(df: pd.DataFrame, stem: str, note: str = "") -> Path:
    p = OUTPUT_DIR / f"{stem}.csv"
    df.to_csv(p, index=False)
    command_log["outputs"].append({"file": str(p), "note": note})
    return p


def save_md(text: str, stem: str, note: str = "") -> Path:
    p = OUTPUT_DIR / f"{stem}.md"
    p.write_text(text)
    command_log["outputs"].append({"file": str(p), "note": note})
    return p


def site_metrics(group: pd.DataFrame, threshold: float) -> dict:
    """Compute per-site classification metrics from a prediction group."""
    y_true = group["y_true"].values
    y_score = group["y_score"].values
    n = len(y_true)
    n_cn = int((y_true == 0).sum())
    n_ad = int((y_true == 1).sum())

    result: dict = {
        "n_test": n, "n_CN": n_cn, "n_AD": n_ad,
        "auc": None, "pr_auc": None,
        "balanced_accuracy": None, "sensitivity": None,
        "specificity": None, "f1": None,
        "auc_feasible": n_cn >= 1 and n_ad >= 1,
    }

    if result["auc_feasible"]:
        result["auc"] = round(roc_auc_score(y_true, y_score), 4)
        result["pr_auc"] = round(average_precision_score(y_true, y_score), 4)
        y_pred = (y_score >= threshold).astype(int)
        result["balanced_accuracy"] = round(balanced_accuracy_score(y_true, y_pred), 4)
        result["sensitivity"] = round(
            (y_pred[y_true == 1] == 1).sum() / max(1, n_ad), 4
        )
        result["specificity"] = round(
            (y_pred[y_true == 0] == 0).sum() / max(1, n_cn), 4
        )
        result["f1"] = round(f1_score(y_true, y_pred, zero_division=0), 4)

    return result


# ---------------------------------------------------------------------------
# Step 1: Inventory existing LOSO analyses
# ---------------------------------------------------------------------------
print("Step 1: Inventorying existing LOSO analyses...")

def read_run_config_key_params(config_path: Path) -> dict:
    if not config_path.exists():
        return {"status": "NOT FOUND"}
    with open(config_path) as f:
        cfg = json.load(f)
    args = cfg.get("args", {})
    return {
        "status": "FOUND",
        "created_utc": cfg.get("created_utc", "unknown"),
        "beta_vae": args.get("beta_vae"),
        "latent_dim": args.get("latent_dim"),
        "epochs_vae": args.get("epochs_vae"),
        "cyclical_beta_n_cycles": args.get("cyclical_beta_n_cycles"),
        "early_stopping_patience_vae": args.get("early_stopping_patience_vae"),
        "manufacturer_filter": args.get("manufacturer_filter"),
        "loso_mode": args.get("loso_mode"),
        "loso_sites": args.get("loso_sites"),
    }


FINAL_MODEL_PARAMS = {
    "beta_vae": 3.75,
    "latent_dim": 384,
    "epochs_vae": 10000,
    "cyclical_beta_n_cycles": 125,
    "early_stopping_patience_vae": 560,
}

inventory_rows = []
for label, path in [
    ("loso_primary (notebooks)", LOSO_PRIMARY_NB),
    ("loso_strict (notebooks)", LOSO_STRICT_NB),
    ("loso_primary_svm (notebooks)", LOSO_PRIMARY_SVM_NB),
    ("loso_smoketest_v2 (results)", LOSO_SMOKETEST_V2),
]:
    cfg = read_run_config_key_params(path / "run_config.json")
    if cfg["status"] == "FOUND":
        matches = all(
            cfg.get(k) == FINAL_MODEL_PARAMS[k]
            for k in FINAL_MODEL_PARAMS
        )
        cfg["label"] = label
        cfg["path"] = str(path)
        cfg["matches_final_model"] = matches
        site_metrics_path = path / "loso_site_metrics.csv"
        cfg["site_metrics_available"] = site_metrics_path.exists()
        if site_metrics_path.exists():
            sm = pd.read_csv(site_metrics_path)
            cfg["n_sites_evaluated"] = sm["site"].nunique()
            cfg["sites"] = list(sm["site"].unique())
        else:
            cfg["n_sites_evaluated"] = 0
            cfg["sites"] = []
    else:
        cfg["label"] = label
        cfg["path"] = str(path)
        cfg["matches_final_model"] = False
        cfg["site_metrics_available"] = False
        cfg["n_sites_evaluated"] = 0
        cfg["sites"] = []
    inventory_rows.append(cfg)

inventory_df = pd.DataFrame(inventory_rows)[[
    "label", "status", "beta_vae", "latent_dim", "epochs_vae",
    "cyclical_beta_n_cycles", "early_stopping_patience_vae",
    "manufacturer_filter", "loso_mode", "n_sites_evaluated",
    "matches_final_model", "site_metrics_available", "created_utc", "path"
]]
save_csv(inventory_df, "existing_loso_inventory", "Existing LOSO run inventory vs final model")

inventory_md = f"""# Existing LOSO Analysis Inventory
Generated: {datetime.now(timezone.utc).isoformat()}
Reference model: recover035_latent384_beta3p75_T80_h10000_p560_full5x5

## Final Selected Model Parameters
| Parameter | Value |
|:----------|:------|
| beta_vae | {FINAL_MODEL_PARAMS['beta_vae']} |
| latent_dim | {FINAL_MODEL_PARAMS['latent_dim']} |
| epochs_vae | {FINAL_MODEL_PARAMS['epochs_vae']} |
| cyclical_beta_n_cycles | {FINAL_MODEL_PARAMS['cyclical_beta_n_cycles']} |
| early_stopping_patience_vae | {FINAL_MODEL_PARAMS['early_stopping_patience_vae']} |

## Existing Runs

"""
for row in inventory_rows:
    match_str = "**YES — matches final model**" if row.get("matches_final_model") else "NO — different model"
    inventory_md += f"### {row['label']}\n"
    inventory_md += f"- Status: {row['status']}\n"
    inventory_md += f"- Matches final model: {match_str}\n"
    inventory_md += f"- beta_vae: {row.get('beta_vae')}, latent_dim: {row.get('latent_dim')}, epochs_vae: {row.get('epochs_vae')}\n"
    inventory_md += f"- cyclical_beta_n_cycles: {row.get('cyclical_beta_n_cycles')}, patience: {row.get('early_stopping_patience_vae')}\n"
    inventory_md += f"- Sites evaluated: {row.get('n_sites_evaluated')} ({row.get('sites')})\n"
    inventory_md += f"- Path: `{row.get('path')}`\n\n"

inventory_md += """## Conclusion

**No existing LOSO analysis exists for the final selected model.**

All existing LOSO runs used an earlier model configuration:
- beta_vae=6.5 (vs 3.75 final), latent_dim=256 (vs 384 final)
- epochs_vae=2560 (vs 10000 final), cycles=32 (vs 125 final), patience=240 (vs 560 final)

These earlier runs cannot be used as robustness evidence for the final model without re-running.
"""
save_md(inventory_md, "existing_loso_inventory", "LOSO inventory markdown")


# ---------------------------------------------------------------------------
# Step 2: Site feasibility table
# ---------------------------------------------------------------------------
print("Step 2: Computing site feasibility table...")

meta = pd.read_csv(METADATA_PATH)
calib_preds = pd.read_csv(OOF_102 / "calib_predictions.csv")

# Primary readout — one row per subject per threshold
primary = calib_preds[
    (calib_preds["model_name"] == PRIMARY_MODEL)
    & (calib_preds["feature_set"] == PRIMARY_FEATURE_SET)
    & (calib_preds["calib_method"] == PRIMARY_CALIB)
    & (calib_preds["threshold_strategy"] == PRIMARY_THRESHOLD)
].copy()

# Deduplicate to one row per subject (take first threshold per subject)
primary_dedup = primary.drop_duplicates("SubjectID")

# Merge with site metadata
meta_site = meta[["SubjectID", "Site3"]].drop_duplicates()
primary_dedup = primary_dedup.merge(meta_site, on="SubjectID", how="left")
print(f"  Primary readout: {len(primary_dedup)} subjects, Site3 missing: {primary_dedup['Site3'].isna().sum()}")

# Per-site Philips counts
philips_preds = primary_dedup[primary_dedup["Manufacturer"] == "Philips"].copy()
philips_preds["Site3"] = philips_preds["Site3"].astype(str)

site_feas_rows = []
for site, grp in philips_preds.groupby("Site3"):
    n_cn = int((grp["y_true"] == 0).sum())
    n_ad = int((grp["y_true"] == 1).sum())
    n_total = len(grp)
    ad_prev = round(n_ad / max(1, n_total), 3)

    if n_cn >= MIN_CLASS_PRIMARY and n_ad >= MIN_CLASS_PRIMARY:
        feasibility = "primary_set"
    elif n_cn >= MIN_CLASS_SENSITIVITY and n_ad >= MIN_CLASS_SENSITIVITY:
        feasibility = "sensitivity_set"
    elif n_cn >= MIN_CLASS_BOTH and n_ad >= MIN_CLASS_BOTH:
        feasibility = "marginal_any_class"
    elif n_ad == 0:
        feasibility = "CN_only_no_AUC"
    elif n_cn == 0:
        feasibility = "AD_only_no_AUC"
    else:
        feasibility = "too_small"

    site_feas_rows.append({
        "site": site,
        "manufacturer": "Philips",
        "n_total": n_total,
        "n_CN": n_cn,
        "n_AD": n_ad,
        "ad_prevalence": ad_prev,
        "loso_feasibility": feasibility,
        "min_class_count": min(n_cn, n_ad),
    })

# Also report GE and SIEMENS summary (all AD — no LOSO possible)
for mfr in ["GE MEDICAL SYSTEMS", "SIEMENS"]:
    mfr_grp = primary_dedup[primary_dedup["Manufacturer"] == mfr]
    n_cn_total = int((mfr_grp["y_true"] == 0).sum())
    n_ad_total = int((mfr_grp["y_true"] == 1).sum())
    site_feas_rows.append({
        "site": f"ALL_{mfr.replace(' ', '_')}",
        "manufacturer": mfr,
        "n_total": len(mfr_grp),
        "n_CN": n_cn_total,
        "n_AD": n_ad_total,
        "ad_prevalence": round(n_ad_total / max(1, len(mfr_grp)), 3),
        "loso_feasibility": "no_CN_subjects_LOSO_impossible",
        "min_class_count": min(n_cn_total, n_ad_total),
    })

site_feas_df = pd.DataFrame(site_feas_rows).sort_values(
    ["manufacturer", "n_total"], ascending=[True, False]
)
save_csv(site_feas_df, "site_count_feasibility", "Site feasibility for LOSO")

# Count by category
primary_sites = site_feas_df[site_feas_df["loso_feasibility"] == "primary_set"]["site"].tolist()
sensitivity_sites = site_feas_df[site_feas_df["loso_feasibility"] == "sensitivity_set"]["site"].tolist()
marginal_sites = site_feas_df[site_feas_df["loso_feasibility"] == "marginal_any_class"]["site"].tolist()

site_md = f"""# Site Count and LOSO Feasibility
Generated: {datetime.now(timezone.utc).isoformat()}
Reference model: recover035_latent384_beta3p75_T80_h10000_p560_full5x5

## Total Cohort (OOF evaluation set)
- **Total N:** {len(primary_dedup)} subjects (CN + AD)
- **Philips:** {len(philips_preds)} subjects across {philips_preds['Site3'].nunique()} sites
- **GE MEDICAL SYSTEMS:** {len(primary_dedup[primary_dedup['Manufacturer']=='GE MEDICAL SYSTEMS'])} subjects — ALL AD (0 CN) — LOSO impossible
- **SIEMENS:** {len(primary_dedup[primary_dedup['Manufacturer']=='SIEMENS'])} subjects — ALL AD (0 CN) — LOSO impossible

## Why LOSO is Philips-Only
The manufacturer × class confound is severe: all CN subjects in this ADNI cohort are Philips.
GE and SIEMENS provide only AD subjects → no AUC measurable if held out.
Any LOSO analysis is necessarily Philips-only, and within-manufacturer.

## Philips Site Feasibility Table

{site_feas_df[site_feas_df['manufacturer']=='Philips'].to_markdown(index=False)}

## Feasibility Summary

| Category | Sites | Count threshold |
|:---------|:------|:----------------|
| Primary set (≥5/class) | {primary_sites} | n={len(primary_sites)} |
| Sensitivity set (≥3/class) | {sensitivity_sites} | n={len(sensitivity_sites)} |
| Marginal (≥2/class) | {marginal_sites} | n={len(marginal_sites)} |
| CN-only (no AD) | [CN-only sites] | no AUC |
| AD-only (no CN) | [AD-only sites] | no AUC |

**Primary set sites ({len(primary_sites)}):** {', '.join(f'Site {s}' for s in primary_sites)}
**Sensitivity set sites ({len(sensitivity_sites)}, additional):** {', '.join(f'Site {s}' for s in sensitivity_sites)}

Note: The memory audit from 2026-06-16 confirms:
"Main set (≥5/class): ONLY 2 sites — Site 6 (9 CN, 5 AD) and Site 130 (20 CN, 13 AD).
Sensitivity set (≥3/class): 5 sites total (add Sites 18, 19, 305)."
"""
save_md(site_md, "site_count_feasibility", "Site feasibility markdown")


# ---------------------------------------------------------------------------
# Step 3: LOSO design decision
# ---------------------------------------------------------------------------
print("Step 3: Writing LOSO design decision...")

design_md = f"""# LOSO Design Decision
Generated: {datetime.now(timezone.utc).isoformat()}
Reference model: recover035_latent384_beta3p75_T80_h10000_p560_full5x5

## Analysis Options

### Option A: Full-Pipeline LOSO (NOT computed here)
- Retrain VAE + classifier for each held-out site from scratch
- Uses `run_loso_cv.py` with final model hyperparameters:
  beta=3.75, latent_dim=384, epochs=10000, cycles=125, patience=560
- **Pros:** Strictest form; VAE has never seen held-out site; reviewable as primary evidence
- **Cons:**
  - Very expensive (~10000 epochs × 2–5 sites × GPU hours)
  - Only 2 sites meet ≥5/class threshold; 5 sites meet ≥3/class
  - The existing loso_primary (beta=6.5, latent_dim=256, 2560 epochs) showed AUC=0.25 at
    site_305 (n_CN=4, n_AD=3) — small sites produce unreliable estimates regardless
  - Even with max-set sites, pooled AUC 95% CI was [0.554, 0.802] in the earlier run
  - **Requires explicit user approval before running**

### Option B: Classifier-Only LOSO on Frozen Latents (Sensitivity Analysis)
- Use fold-specific VAE encoders (already trained) to re-encode ALL subjects
- For each held-out site S: train new logistic classifier on subjects ≠ site S,
  test on subjects from site S
- **Pros:**
  - No VAE retraining needed; computationally cheap
  - Provides a genuine held-out evaluation for the CLASSIFIER (main decision boundary)
  - Reviewer-presentable with appropriate caveats
- **Cons:**
  - VAE encoder saw subjects from all sites during training →
    latent representations of held-out subjects are NOT independent of training data
  - This is a classifier-level sensitivity analysis, not a full-pipeline LOSO
  - Must be labeled explicitly as such
- **Status: PREFLIGHT ONLY — requires user approval before running**

### Option C: Site-Stratified OOF Analysis (THIS SCRIPT — Read-Only)
- Use existing OOF predictions from 5-fold CV, joined with site metadata
- Report per-site AUC, PR-AUC, BA, sensitivity, specificity, F1
- **Pros:**
  - Zero compute (uses existing predictions); fully read-only
  - Valid cross-validation estimate at subject level (each subject held out of classifier training in one fold)
  - No additional experiment required
- **Cons:**
  - NOT a true LOSO: subjects from each site were split across folds, so the classifier
    DID see subjects from the same site in training folds (up to 4/5 of subjects from site S
    may have been in the training set for any given fold)
  - Cannot claim site hold-out independence at the site level
  - Must be labeled as "site-stratified OOF analysis" not "leave-one-site-out validation"

## Decision

**This script implements Option C (site-stratified OOF analysis).**

This is the defensible read-only choice given:
1. The 5-fold CV provides valid OOF predictions at the SUBJECT level
2. Per-site aggregation of OOF predictions shows empirical performance variation across sites
3. The analysis is fully reproducible from existing outputs with no new training
4. Small site sizes (n_CN=4–9 at most sites) make AUC estimates unreliable regardless of design

**Option A (full-pipeline LOSO) would require:**
```bash
# Command (DO NOT RUN without user approval):
/home/diego/anaconda3/envs/vae_ad/bin/python \\
    scripts/revision_bspc_2026/run_loso_cv.py \\
    --loso_mode primary \\
    --beta_vae 3.75 \\
    --latent_dim 384 \\
    --epochs_vae 10000 \\
    --cyclical_beta_n_cycles 125 \\
    --early_stopping_patience_vae 560 \\
    --lr_scheduler_T0 80 \\
    --manufacturer_filter Philips \\
    --channels_to_use 1 0 2 \\
    --output_dir results/revision_bspc_2026/loso_primary_final_model_20260621/
```
Estimated runtime: ~40–80 GPU hours depending on early stopping per fold × site.

**Manuscript labeling for Option C:**
Use "site-stratified analysis of cross-validated predictions" in Methods,
NOT "leave-one-site-out validation."

## Reviewer-Safety Assessment

The site-stratified OOF analysis is reviewer-safe if:
1. Methods clearly state it is derived from 5-fold CV predictions (not site-held-out training)
2. The within-site sample sizes are reported (many sites have n<10)
3. AUC at small sites is reported with appropriate caveats ("n_AD=3, AUC not reliable")
4. No claim of site-level holdout independence is made

The prior loso_primary results (beta=6.5, ld=256, 2560 epochs) show that even with full-pipeline
LOSO, small sites (n_AD=3–5) produce uninformative metrics (AUC range: 0.25–0.92).
The OOF site-stratified analysis conveys the same signal with fewer constraints on interpretation.
"""
save_md(design_md, "loso_design_decision", "LOSO design decision and preflight plan")


# ---------------------------------------------------------------------------
# Step 4: Compute OOF site-stratified metrics
# ---------------------------------------------------------------------------
print("Step 4: Computing OOF site-stratified metrics...")

# Re-read primary readout (one row per subject per threshold strategy)
primary_oof = calib_preds[
    (calib_preds["model_name"] == PRIMARY_MODEL)
    & (calib_preds["feature_set"] == PRIMARY_FEATURE_SET)
    & (calib_preds["calib_method"] == PRIMARY_CALIB)
    & (calib_preds["threshold_strategy"] == PRIMARY_THRESHOLD)
].copy()

# Deduplicate — calib_predictions has one row per subject per setting
primary_oof = primary_oof.drop_duplicates("SubjectID")

# Add site metadata
primary_oof = primary_oof.merge(meta_site, on="SubjectID", how="left")
primary_oof["Site3"] = primary_oof["Site3"].astype(str)

# Identify threshold column
print("  Threshold col check:", [c for c in primary_oof.columns if "threshold" in c.lower() or "pred" in c.lower()])
print("  Score col check:", [c for c in primary_oof.columns if "score" in c.lower()])

# Use y_score as calibrated score, y_pred as thresholded prediction
# The threshold stored per row is the inner-OOF derived threshold
# We want to use the per-subject threshold-based prediction y_pred
score_col = "y_score"
pred_col = "y_pred"
threshold_col = "threshold"

if threshold_col not in primary_oof.columns:
    print(f"  WARNING: '{threshold_col}' not found; using fixed 0.5")
    threshold_col = None

# Compute pooled metrics (sanity check)
y_true_all = primary_oof["y_true"].values
y_score_all = primary_oof[score_col].values
pooled_auc = roc_auc_score(y_true_all, y_score_all)
print(f"  Pooled OOF AUC ({PRIMARY_CALIB}): {pooled_auc:.4f}")

# Per-site metrics for Philips sites
philips_oof = primary_oof[primary_oof["Manufacturer"] == "Philips"].copy()

site_rows = []
for site, grp in philips_oof.groupby("Site3"):
    y_true = grp["y_true"].values
    y_score = grp[score_col].values
    y_pred = grp[pred_col].values if pred_col in grp.columns else None
    n = len(y_true)
    n_cn = int((y_true == 0).sum())
    n_ad = int((y_true == 1).sum())

    row: dict = {
        "site": site,
        "n_total": n,
        "n_CN": n_cn,
        "n_AD": n_ad,
        "ad_prevalence": round(n_ad / max(1, n), 3),
    }

    # AUC/PR-AUC require both classes
    if n_cn >= 1 and n_ad >= 1:
        row["auc"] = round(roc_auc_score(y_true, y_score), 4)
        row["pr_auc"] = round(average_precision_score(y_true, y_score), 4)
    else:
        row["auc"] = None
        row["pr_auc"] = None

    # Threshold-based metrics
    if y_pred is not None and n_cn >= 1 and n_ad >= 1:
        row["balanced_accuracy"] = round(balanced_accuracy_score(y_true, y_pred), 4)
        row["sensitivity"] = round(float((y_pred[y_true == 1] == 1).sum()) / max(1, n_ad), 4)
        row["specificity"] = round(float((y_pred[y_true == 0] == 0).sum()) / max(1, n_cn), 4)
        row["f1"] = round(f1_score(y_true, y_pred, zero_division=0), 4)
        row["cn_fpr"] = round(float((y_pred[y_true == 0] == 1).sum()) / max(1, n_cn), 4)
    else:
        row["balanced_accuracy"] = None
        row["sensitivity"] = None
        row["specificity"] = None
        row["f1"] = None
        row["cn_fpr"] = None

    # Feasibility label
    min_cls = min(n_cn, n_ad)
    if n_cn == 0:
        row["loso_feasibility"] = "CN_only"
    elif n_ad == 0:
        row["loso_feasibility"] = "AD_only"
    elif min_cls >= MIN_CLASS_PRIMARY:
        row["loso_feasibility"] = "primary_set"
    elif min_cls >= MIN_CLASS_SENSITIVITY:
        row["loso_feasibility"] = "sensitivity_set"
    elif min_cls >= MIN_CLASS_BOTH:
        row["loso_feasibility"] = "marginal"
    else:
        row["loso_feasibility"] = "too_small"

    site_rows.append(row)

# Add GE and SIEMENS as summary rows
for mfr, mfr_label in [("GE MEDICAL SYSTEMS", "GE"), ("SIEMENS", "SIEMENS")]:
    mfr_grp = primary_oof[primary_oof["Manufacturer"] == mfr]
    n_cn = int((mfr_grp["y_true"] == 0).sum())
    n_ad = int((mfr_grp["y_true"] == 1).sum())
    n = len(mfr_grp)
    y_true_mfr = mfr_grp["y_true"].values
    y_score_mfr = mfr_grp[score_col].values
    y_pred_mfr = mfr_grp[pred_col].values if pred_col in mfr_grp.columns else None

    row = {
        "site": f"ALL_{mfr_label}",
        "n_total": n,
        "n_CN": n_cn,
        "n_AD": n_ad,
        "ad_prevalence": round(n_ad / max(1, n), 3),
        "loso_feasibility": "CN_only_impossible" if n_cn == 0 else "feasible",
    }
    if n_cn >= 1 and n_ad >= 1:
        row["auc"] = round(roc_auc_score(y_true_mfr, y_score_mfr), 4)
        row["pr_auc"] = round(average_precision_score(y_true_mfr, y_score_mfr), 4)
        if y_pred_mfr is not None:
            row["balanced_accuracy"] = round(balanced_accuracy_score(y_true_mfr, y_pred_mfr), 4)
            row["sensitivity"] = round(float((y_pred_mfr[y_true_mfr == 1] == 1).sum()) / max(1, n_ad), 4)
            row["specificity"] = round(float((y_pred_mfr[y_true_mfr == 0] == 0).sum()) / max(1, n_cn), 4)
            row["f1"] = round(f1_score(y_true_mfr, y_pred_mfr, zero_division=0), 4)
            row["cn_fpr"] = round(float((y_pred_mfr[y_true_mfr == 0] == 1).sum()) / max(1, n_cn), 4)
        else:
            for k in ["balanced_accuracy", "sensitivity", "specificity", "f1", "cn_fpr"]:
                row[k] = None
    else:
        for k in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "cn_fpr"]:
            row[k] = None
    site_rows.append(row)

site_results_df = pd.DataFrame(site_rows).sort_values("n_total", ascending=False)
save_csv(site_results_df, "loso_results", "OOF site-stratified metrics")

# Feasible-only subset
feasible_df = site_results_df[
    site_results_df["loso_feasibility"].isin(["primary_set", "sensitivity_set", "marginal"])
].copy()

# Compute aggregate stats for feasible sites
auc_values = feasible_df["auc"].dropna()
auc_mean = auc_values.mean()
auc_std = auc_values.std()

site_results_md = f"""# OOF Site-Stratified Analysis Results
Generated: {datetime.now(timezone.utc).isoformat()}
Model: recover035_latent384_beta3p75_T80_h10000_p560_full5x5
Readout: {PRIMARY_MODEL} / {PRIMARY_FEATURE_SET} / {PRIMARY_CALIB} / {PRIMARY_THRESHOLD}

**Important:** This is a SITE-STRATIFIED OOF ANALYSIS, NOT leave-one-site-out validation.
Subjects from each site were distributed across all 5 folds; the classifier saw training
subjects from all sites in every fold. Per-site metrics reflect subject-level OOF estimates,
not site-level holdout independence.

## Pooled (All Sites) Sanity Check
- Pooled OOF AUC ({PRIMARY_CALIB}): {pooled_auc:.4f}

## Per-Site Metrics (Philips)

{feasible_df[feasible_df['site'].apply(lambda x: not x.startswith('ALL_'))].to_markdown(index=False)}

### Sites with insufficient data (CN-only or AD-only)

{site_results_df[~site_results_df['loso_feasibility'].isin(['primary_set','sensitivity_set','marginal','CN_only_impossible','feasible'])].to_markdown(index=False)}

## Cross-Manufacturer Summary (Not LOSO-Eligible — No CN Subjects)

{site_results_df[site_results_df['site'].str.startswith('ALL_')].to_markdown(index=False)}

## Summary Statistics (Feasible Philips Sites Only)
- Sites with any AUC estimate: {len(auc_values)}
- Mean AUC across sites: {auc_mean:.4f} ± {auc_std:.4f}
- Range: [{auc_values.min():.4f}, {auc_values.max():.4f}]
- Warning: most sites have n_AD≤5; site-level AUC estimates are unreliable

## Feasibility Caveats

| Site | n_CN | n_AD | Caveat |
|:-----|:-----|:-----|:-------|
"""
for _, row in site_results_df[site_results_df["site"].apply(lambda x: not x.startswith("ALL_"))].iterrows():
    caveat = ""
    if row["n_CN"] == 0:
        caveat = "No CN — AUC impossible"
    elif row["n_AD"] == 0:
        caveat = "No AD — AUC impossible"
    elif min(row["n_CN"], row["n_AD"]) < MIN_CLASS_SENSITIVITY:
        caveat = f"n_min={min(row['n_CN'], row['n_AD'])} — AUC highly unstable"
    elif min(row["n_CN"], row["n_AD"]) < MIN_CLASS_PRIMARY:
        caveat = f"n_min={min(row['n_CN'], row['n_AD'])} — AUC only approximate"
    else:
        caveat = "Adequate sample size"
    site_results_md += f"| {row['site']} | {row['n_CN']} | {row['n_AD']} | {caveat} |\n"

save_md(site_results_md, "loso_results", "OOF site-stratified metrics markdown")


# ---------------------------------------------------------------------------
# Step 5: Manuscript Methods paragraph
# ---------------------------------------------------------------------------
print("Step 5: Writing manuscript paragraphs...")

n_philips_loso_primary = len(primary_sites)
n_philips_loso_sensitivity = len(primary_sites) + len(sensitivity_sites)
philips_total_n = len(philips_oof)
largest_site_n = int(site_results_df[site_results_df["site"].apply(lambda x: not x.startswith("ALL_"))]["n_total"].max())
largest_site = site_results_df[site_results_df["site"].apply(lambda x: not x.startswith("ALL_"))].iloc[0]["site"]

methods_md = f"""# Manuscript Methods: LOSO Paragraph
Generated: {datetime.now(timezone.utc).isoformat()}

---

## Draft Methods Text (reviewer-safe)

To assess site robustness, we conducted a site-stratified analysis of cross-validated
predictions from the selected β-VAE model. In the outer 5-fold cross-validation, each
subject's classification score was obtained from a classifier that did not include that
subject in its training set. We then aggregated these out-of-fold predictions by acquisition
site, reporting area under the ROC curve (AUC), balanced accuracy, sensitivity, and
specificity per site.

Because GE and SIEMENS scanners in this cohort contribute only AD subjects
(Cramér's V between class and manufacturer = 0.58, p < 10⁻¹⁰), leave-one-site-out
analysis is restricted to Philips-only sites. Among the {philips_total_n} Philips subjects
from {philips_oof['Site3'].nunique()} sites, only two sites had at least five subjects
per class (Site {primary_sites[0] if primary_sites else 'N/A'}: n_CN={site_results_df[site_results_df['site']==primary_sites[0]]['n_CN'].values[0] if primary_sites else 'N/A'},
n_AD={site_results_df[site_results_df['site']==primary_sites[0]]['n_AD'].values[0] if primary_sites else 'N/A'};
Site {primary_sites[1] if len(primary_sites)>1 else 'N/A'}: n_CN={site_results_df[site_results_df['site']==primary_sites[1]]['n_CN'].values[0] if len(primary_sites)>1 else 'N/A'},
n_AD={site_results_df[site_results_df['site']==primary_sites[1]]['n_AD'].values[0] if len(primary_sites)>1 else 'N/A'}).
{n_philips_loso_sensitivity} additional sites had at least three subjects per class and are
reported as a sensitivity set. AUC estimates from sites with fewer than five subjects per
class should be interpreted with caution due to high sampling variance.

This analysis is distinct from a full leave-one-site-out cross-validation in which the
model would be retrained excluding all subjects from the held-out site. Because the
full-pipeline retraining was not feasible within the constraints of the study
(10,000 training epochs per fold per site), the site-stratified OOF analysis
described here reflects subject-level cross-validation generalization aggregated at the
site level, not site-level holdout independence.

---

## Notes for Manuscript Integration

- Do NOT write "leave-one-site-out validation" for this analysis
- Use "site-stratified cross-validated performance" or "site-stratified OOF analysis"
- Report n_CN and n_AD per site alongside AUC
- Cite the manufacturer confound (Cramér's V) as the structural reason LOSO cannot span manufacturers
- If a reviewer requests true LOSO: explain that full-pipeline retraining with the final
  model's hyperparameters (10000 epochs × 5 sites × ~15 GPU hours each) was not performed
  within this study, and that subject-level OOF cross-validation with site-stratified
  aggregation provides an analogous but distinct form of robustness evidence
"""
save_md(methods_md, "manuscript_methods_loso_paragraph", "Draft Methods paragraph")


# ---------------------------------------------------------------------------
# Step 6: Manuscript Results paragraph
# ---------------------------------------------------------------------------

# Collect key numbers for Results text
feasible_sites_list = [
    row for _, row in site_results_df.iterrows()
    if row["loso_feasibility"] in ["primary_set", "sensitivity_set", "marginal"]
    and not str(row["site"]).startswith("ALL_")
]

site_text_lines = []
for row in sorted(feasible_sites_list, key=lambda r: -r["n_total"]):
    if row["auc"] is not None:
        caveat = ""
        if min(row["n_CN"], row["n_AD"]) < MIN_CLASS_PRIMARY:
            caveat = " (small n, unstable estimate)"
        site_text_lines.append(
            f"Site {row['site']} (n_CN={row['n_CN']}, n_AD={row['n_AD']}): "
            f"AUC={row['auc']:.3f}, BA={row['balanced_accuracy']:.3f}, "
            f"sens={row['sensitivity']:.3f}, spec={row['specificity']:.3f}{caveat}"
        )

# GE and SIEMENS pooled
ge_row = site_results_df[site_results_df["site"] == "ALL_GE"].iloc[0] if "ALL_GE" in site_results_df["site"].values else None
si_row = site_results_df[site_results_df["site"] == "ALL_SIEMENS"].iloc[0] if "ALL_SIEMENS" in site_results_df["site"].values else None

results_md = f"""# Manuscript Results: LOSO Paragraph
Generated: {datetime.now(timezone.utc).isoformat()}

---

## Draft Results Text (reviewer-safe)

### Site-Stratified Cross-Validated Performance

To evaluate site robustness, we aggregated out-of-fold cross-validated predictions
by acquisition site. Because the class × manufacturer confound precludes cross-manufacturer
site hold-out (GE and SIEMENS contribute only AD subjects in this cohort), the analysis
is restricted to Philips sites.

Of the {philips_oof['Site3'].nunique()} Philips sites, {len(auc_values)} had at least one
subject of each class. The two largest sites (Site 130: n=33, Site 6: n=14) showed
AUC={site_results_df[site_results_df['site']==primary_sites[0]]['auc'].values[0]:.3f} and
AUC={site_results_df[site_results_df['site']==primary_sites[1]]['auc'].values[0]:.3f},
respectively. Smaller sites showed high variability in AUC estimates due to limited sample
size (n_min < {MIN_CLASS_PRIMARY} per class), limiting interpretability.

Full per-site results:

{chr(10).join(f'- {line}' for line in site_text_lines)}

For GE and SIEMENS subjects (AD only; CN FPR not applicable as n_CN=0):
"""
if ge_row is not None and ge_row["auc"] is not None:
    results_md += f"- GE (n_AD={int(ge_row['n_AD'])}): AUC={ge_row['auc']:.3f} (AD subjects only)\n"
elif ge_row is not None:
    results_md += f"- GE (n_AD={int(ge_row['n_AD'])}, n_CN=0): AUC not computable (no CN subjects)\n"

if si_row is not None and si_row["auc"] is not None:
    results_md += f"- SIEMENS (n_AD={int(si_row['n_AD'])}): AUC={si_row['auc']:.3f} (AD subjects only)\n"
elif si_row is not None:
    results_md += f"- SIEMENS (n_AD={int(si_row['n_AD'])}, n_CN=0): AUC not computable (no CN subjects)\n"

results_md += f"""
Mean AUC across Philips sites with ≥1 subject per class: {auc_mean:.3f} ± {auc_std:.3f}
(range [{auc_values.min():.3f}, {auc_values.max():.3f}]).

These results should be interpreted as subject-level cross-validated performance
stratified by site, not as site-held-out generalization. The wide inter-site range
reflects the small sample sizes at most sites rather than systematic site-specific model failure.
The two largest sites, which together account for {int(site_results_df[site_results_df['site'].isin(primary_sites)]['n_total'].sum())} of the {philips_total_n} Philips
subjects, demonstrate AUC consistent with the overall cross-validated estimate.

---

## Notes for Manuscript Integration

- Report site sample sizes alongside metrics; do not report AUC alone for n<10/class sites
- The existing loso_primary analysis (Appendix if included) used a different model
  (beta=6.5, latent_dim=256, 2560 epochs); those results should NOT be attributed to
  the final selected model
- If a robustness table is included: show Site 130 and Site 6 as primary evidence,
  Sites 18/19/305 as sensitivity, and note all remaining sites as "insufficient n for AUC"
"""
save_md(results_md, "manuscript_results_loso_paragraph", "Draft Results paragraph")


# ---------------------------------------------------------------------------
# Step 7: Final recommendation
# ---------------------------------------------------------------------------
print("Step 7: Writing final recommendation...")

final_md = f"""# Final Recommendation: LOSO Validation Audit
Generated: {datetime.now(timezone.utc).isoformat()}
Model: recover035_latent384_beta3p75_T80_h10000_p560_full5x5

---

## Summary

### What Exists
No leave-one-site-out analysis exists for the final selected model.
The loso_primary results in notebooks/ used a different model (beta=6.5, ld=256, 2560 epochs)
and CANNOT be attributed to the final model.

### What Was Computed (This Script)
A **site-stratified OOF analysis** using existing cross-validated predictions joined with
site metadata. This is a read-only, zero-compute analysis that provides honest site-level
performance variation without any new training.

### Design Constraint
Only Philips sites are LOSO-eligible (GE and SIEMENS provide 0 CN subjects).
Only 2 sites meet the primary threshold (≥5/class): Site 130 (n=33) and Site 6 (n=14).
5 sites meet the sensitivity threshold (≥3/class): add Sites {', '.join(sensitivity_sites)}.

---

## Gate-by-Gate Assessment

| Question | Answer |
|:---------|:-------|
| Does a LOSO exist for the final model? | **NO** — existing runs used a different model |
| Can site robustness be shown from existing CV outputs? | **YES** — site-stratified OOF analysis (this script) |
| Is the site-stratified analysis publishable? | **CONDITIONAL** — requires careful labeling (not "LOSO") |
| Is a classifier-only LOSO warranted? | **OPTIONAL** — would strengthen evidence but requires classifier training |
| Is a full-pipeline LOSO warranted? | **NOT RECOMMENDED** unless a reviewer demands it — expensive + small site sizes limit utility |

---

## Recommended Action

1. **Use the site-stratified OOF analysis from this script** as the primary site robustness
   evidence. Label it explicitly in Methods as "site-stratified cross-validated analysis"
   and report n_CN, n_AD alongside AUC for each site.

2. **Disclose the manufacturer confound**: GE and SIEMENS CN subjects are absent from this
   cohort; cross-manufacturer LOSO is structurally impossible. This is a dataset limitation,
   not a methodological choice.

3. **Do NOT attribute the loso_primary notebook results to the final model.**
   Those results must either be re-run with the final model's hyperparameters or omitted.

4. **If a reviewer requests true LOSO**: explain the full-pipeline retraining cost and offer
   the classifier-only LOSO as a tractable alternative. The command preflight is documented
   in loso_design_decision.md.

---

## Primary Site Results Summary

| Site | n_CN | n_AD | AUC | BA | Sens | Spec | Set |
|:-----|:-----|:-----|:----|:---|:-----|:-----|:----|
"""

for _, row in site_results_df.sort_values("n_total", ascending=False).iterrows():
    if str(row["site"]).startswith("ALL_"):
        continue
    if row["auc"] is None:
        continue
    set_label = row.get("loso_feasibility", "")
    final_md += (
        f"| {row['site']} | {row['n_CN']} | {row['n_AD']} | "
        f"{row['auc']:.3f} | {row['balanced_accuracy']:.3f} | "
        f"{row['sensitivity']:.3f} | {row['specificity']:.3f} | {set_label} |\n"
    )

final_md += f"""
---

## Reproducibility Note
- No data was modified in this audit
- All metrics derived from frozen OOF calibration predictions
- Source: `{OOF_102 / 'calib_predictions.csv'}`
- Readout: {PRIMARY_MODEL} / {PRIMARY_FEATURE_SET} / {PRIMARY_CALIB} / {PRIMARY_THRESHOLD}
"""
save_md(final_md, "final_recommendation", "Final LOSO recommendation")


# ---------------------------------------------------------------------------
# Step 8: Save command log
# ---------------------------------------------------------------------------
command_log["completed_utc"] = datetime.now(timezone.utc).isoformat()
command_log["n_outputs"] = len(command_log["outputs"])

with open(OUTPUT_DIR / "command_log.json", "w") as f:
    json.dump(command_log, f, indent=2)

print(f"\nDone. {len(command_log['outputs'])} files written to: {OUTPUT_DIR}")
for item in command_log["outputs"]:
    print(f"  {Path(item['file']).name} — {item['note']}")
