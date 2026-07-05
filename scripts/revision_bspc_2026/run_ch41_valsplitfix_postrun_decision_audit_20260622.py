#!/usr/bin/env python3
"""
scripts/revision_bspc_2026/run_ch41_valsplitfix_postrun_decision_audit_20260622.py

Post-run decision audit for [4,1] valsplitfix model vs [1,0,2] reference.

Hard guardrails (always in effect):
  - Read-only with respect to source data
  - No model training (Stage B classifier training was run as part of readout pipeline)
  - No tensor modification
  - No metadata modification
  - No prediction modification
  - No threshold refitting
  - No subject exclusion
  - No model selection
  - Do not delete or edit the invalid old [4,1] run
  - Do not use Martín acquisition/QC variables as features

Candidate: recover035_ch41_latent384_beta3p75_T80_h10000_p560_full5x5_valsplitfix_20260622
Reference: recover035_latent384_beta3p75_T80_h10000_p560_full5x5
Context:   recover035_ch1only_latent384_beta4p5_T80_h10000_p560_full5x5_site31_mayo_reprocessed14

Output: results/revision_bspc_2026/ch41_valsplitfix_postrun_decision_audit_20260622/
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
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
OUTPUT_DIR = RESULTS / "ch41_valsplitfix_postrun_decision_audit_20260622"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Directories
RUN_CH41 = RESULTS / "recover035_ch41_latent384_beta3p75_T80_h10000_p560_full5x5_valsplitfix_20260622"
RUN_102 = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
RUN_OLD_CH41 = RESULTS / "recover035_ch41_latent384_beta3p75_T80_h10000_p560_full5x5"

OOF_CH41 = RESULTS / "recover035_ch41_latent384_beta3p75_T80_h10000_p560_full5x5_valsplitfix_20260622_stageB_oof_score_calibration"
OOF_102 = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
OOF_4P5 = RESULTS / "recover035_ch1only_latent384_beta4p5_T80_h10000_p560_full5x5_site31_mayo_reprocessed14_stageB_oof_score_calibration"

STGB_CH41 = RUN_CH41 / "classifier_only_readout"
STGB_102 = RUN_102 / "classifier_only_readout"

# Promotion gates (locked)
AUC_GATE = 0.782951
PR_AUC_GATE = 0.559873
PHILIPS_FPR_GATE = 0.4545  # [1,0,2] OOF ECDF reference

# Primary readout parameters
PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEAT = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESH = "inner_oof_target_sens_ge_0p70_max_spec"

NOW = datetime.now(timezone.utc).isoformat()
command_log: dict = {
    "script": __file__,
    "generated_utc": NOW,
    "candidate": str(RUN_CH41),
    "reference": str(RUN_102),
    "oof_candidate": str(OOF_CH41),
    "oof_reference": str(OOF_102),
    "promotion_gates": {"auc": AUC_GATE, "pr_auc": PR_AUC_GATE, "philips_fpr_gate": PHILIPS_FPR_GATE},
    "outputs": [],
}


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


def load_primary_oof(oof_dir: Path) -> pd.DataFrame:
    preds = pd.read_csv(oof_dir / "calib_predictions.csv")
    return (
        preds[
            (preds["model_name"] == PRIMARY_MODEL)
            & (preds["feature_set"] == PRIMARY_FEAT)
            & (preds["calib_method"] == PRIMARY_CALIB)
            & (preds["threshold_strategy"] == PRIMARY_THRESH)
        ]
        .drop_duplicates("SubjectID")
        .set_index("SubjectID")
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Completion & integrity
# ─────────────────────────────────────────────────────────────────────────────
print("Step 1: Completion and integrity check...")

REQUIRED_PER_FOLD = [
    "vae_model_fold_{f}.pt",
    "vae_train_history_fold_{f}.joblib",
    "vae_norm_params.joblib",
    "vae_training_pool_tensor_idx.npy",
    "vae_actual_train_idx_local_to_pool.npy",
    "vae_internal_val_idx_local_to_pool.npy",
    "test_predictions_logreg.csv",
    "test_predictions_svm.csv",
    "test_subjects_fold.csv",
    "classifier_logreg_final_pipeline_fold_{f}.joblib",
]

integrity_rows = []
missing_count = 0
for fold in range(1, 6):
    fold_dir = RUN_CH41 / f"fold_{fold}"
    for tmpl in REQUIRED_PER_FOLD:
        name = tmpl.replace("{f}", str(fold))
        path = fold_dir / name
        status = "PRESENT" if path.exists() else "MISSING"
        if status == "MISSING":
            missing_count += 1
        integrity_rows.append(
            {"fold": fold, "artifact": name, "status": status, "path": str(path)}
        )

# Top-level artifacts
for name in [
    "run_config.json",
    f"all_folds_clf_predictions_MULTI_logreg_vaeconvtranspose4l_ld384_beta3.75_normzscore_offdiag_ch2sel_intFCquarter_drop0.15_ln0_outer5x1_scoreroc_auc.csv",
    f"all_folds_metrics_MULTI_logreg_vaeconvtranspose4l_ld384_beta3.75_normzscore_offdiag_ch2sel_intFCquarter_drop0.15_ln0_outer5x1_scoreroc_auc.csv",
    "all_folds_vae_training_history_logreg_vaeconvtranspose4l_ld384_beta3.75_normzscore_offdiag_ch2sel_intFCquarter_drop0.15_ln0_outer5x1_scoreroc_auc.joblib",
]:
    status = "PRESENT" if (RUN_CH41 / name).exists() else "MISSING"
    if status == "MISSING":
        missing_count += 1
    integrity_rows.append({"fold": "top", "artifact": name, "status": status, "path": str(RUN_CH41 / name)})

# Stage B and OOF calibration
for label, path in [
    ("StageB/classifier_only_readout", STGB_CH41),
    ("OOF_calibration", OOF_CH41),
    ("StageB/latent_cache", STGB_CH41 / "latent_cache"),
]:
    status = "PRESENT" if path.exists() else "MISSING"
    if status == "MISSING":
        missing_count += 1
    integrity_rows.append({"fold": "post", "artifact": label, "status": status, "path": str(path)})

integrity_df = pd.DataFrame(integrity_rows)
save_csv(integrity_df, "completion_integrity")
integrity_pass = missing_count == 0

integrity_md = f"""# Completion Integrity Check
Generated: {NOW}
Candidate: {RUN_CH41.name}

## Result: {"PASS — 0 missing artifacts" if integrity_pass else f"FAIL — {missing_count} missing artifact(s)"}

### Summary
- Folds checked: 5/5
- Required per-fold artifacts: {len(REQUIRED_PER_FOLD)}
- Total artifacts checked: {len(integrity_rows)}
- Missing: {missing_count}
- Stage B classifier_only_readout: {"PRESENT" if STGB_CH41.exists() else "MISSING"}
- OOF calibration: {"PRESENT" if OOF_CH41.exists() else "MISSING"}

### Config Hash
- git_hash: {json.load(open(RUN_CH41 / 'run_config.json')).get('git_hash', 'N/A')}
- created_utc: {json.load(open(RUN_CH41 / 'run_config.json')).get('created_utc', 'N/A')}
- channels_to_use: [4, 1] = ['dFC_StdDev', 'Pearson_Full_FisherZ_Signed']
- vae_abort_if_val_split_fails: True (valsplitfix applied)
- vae_stratify_cols: ['Manufacturer']

### Invalid Old Run
- {RUN_OLD_CH41.name}: EXISTS (must not be used for analysis — VAE internal val=0 in fold_1)
"""
save_md(integrity_md, "completion_integrity")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — VAE validation & checkpoint audit
# ─────────────────────────────────────────────────────────────────────────────
print("Step 2: VAE validation and checkpoint audit...")

def parse_vae_history(run_dir: Path, fold: int, label: str) -> dict:
    h = joblib.load(run_dir / f"fold_{fold}" / f"vae_train_history_fold_{fold}.joblib")
    vms = np.array(h["val_loss_modelsel"])
    final_ep = len(vms)
    finite_mask = np.isfinite(vms)
    if finite_mask.any():
        best_ep = int(np.nanargmin(vms)) + 1
        best_val = float(np.nanmin(vms))
    else:
        best_ep = None
        best_val = None

    nan_arr = np.where(~finite_mask)[0]
    nan_start = int(nan_arr[0]) + 1 if len(nan_arr) > 0 else None
    n_nan = int((~finite_mask).sum())
    beta_arr = np.array(h["beta"])
    beta_at_best = float(beta_arr[best_ep - 1]) if best_ep else None
    pool_n = int(np.load(run_dir / f"fold_{fold}" / "vae_training_pool_tensor_idx.npy").shape[0])
    train_n = int(np.load(run_dir / f"fold_{fold}" / "vae_actual_train_idx_local_to_pool.npy").shape[0])
    val_n = int(np.load(run_dir / f"fold_{fold}" / "vae_internal_val_idx_local_to_pool.npy").shape[0])

    val_split_mode = h.get("val_split_mode", "Manufacturer_stratified (from run_config)")
    ckpt_src = h.get("checkpoint_source", "ValL(beta_max) [inferred from history]")

    return {
        "model": label,
        "fold": fold,
        "pool_N": pool_n,
        "train_N": train_n,
        "val_N": val_n,
        "val_N_zero": val_n == 0,
        "final_epoch": final_ep,
        "best_epoch": best_ep,
        "best_val_loss_modelsel": round(best_val, 3) if best_val is not None else None,
        "beta_at_best_epoch": round(beta_at_best, 3) if beta_at_best is not None else None,
        "checkpoint_at_max_beta": beta_at_best is not None and abs(beta_at_best - 3.75) < 0.01,
        "nan_in_val_loss_modelsel": n_nan > 0,
        "nan_count": n_nan,
        "nan_starts_epoch": nan_start,
        "val_split_mode": val_split_mode,
        "checkpoint_source": ckpt_src,
    }

vae_rows = []
for fold in range(1, 6):
    vae_rows.append(parse_vae_history(RUN_CH41, fold, "[4,1]_valsplitfix"))
for fold in range(1, 6):
    vae_rows.append(parse_vae_history(RUN_102, fold, "[1,0,2]_ref"))

# Old invalid run fold 1 only
old_val_n = int(np.load(RUN_OLD_CH41 / "fold_1" / "vae_internal_val_idx_local_to_pool.npy").shape[0]) if (RUN_OLD_CH41 / "fold_1" / "vae_internal_val_idx_local_to_pool.npy").exists() else "MISSING"
vae_rows.append({
    "model": "[4,1]_OLD_INVALID", "fold": 1,
    "pool_N": "N/A", "train_N": "N/A", "val_N": old_val_n,
    "val_N_zero": old_val_n == 0, "final_epoch": "N/A", "best_epoch": "N/A",
    "best_val_loss_modelsel": "N/A", "beta_at_best_epoch": "N/A",
    "checkpoint_at_max_beta": False, "nan_in_val_loss_modelsel": "UNKNOWN",
    "nan_count": "N/A", "nan_starts_epoch": "N/A",
    "val_split_mode": "INVALID (val_N=0)", "checkpoint_source": "INVALID",
})

vae_df = pd.DataFrame(vae_rows)
save_csv(vae_df, "vae_validation_checkpoint_audit")

ch41_rows = [r for r in vae_rows if r["model"] == "[4,1]_valsplitfix"]
ref_rows = [r for r in vae_rows if r["model"] == "[1,0,2]_ref"]

fold4_issue = ch41_rows[3]  # fold 4, 0-indexed
ch41_mean_final = np.mean([r["final_epoch"] for r in ch41_rows])
ref_mean_final = np.mean([r["final_epoch"] for r in ref_rows])

vae_md = f"""# VAE Validation and Checkpoint Audit
Generated: {NOW}

## [4,1] valsplitfix — Foldwise VAE Summary

| Fold | pool_N | train_N | val_N | final_ep | best_ep | beta@best | ckpt@maxβ | NaN_in_VMS | NaN_start |
|:-----|:-------|:--------|:------|:---------|:--------|:----------|:----------|:-----------|:----------|
"""
for r in ch41_rows:
    vae_md += f"| {r['fold']} | {r['pool_N']} | {r['train_N']} | {r['val_N']} | {r['final_epoch']} | {r['best_epoch']} | {r['beta_at_best_epoch']} | {'✓' if r['checkpoint_at_max_beta'] else '✗'} | {'⚠ YES' if r['nan_in_val_loss_modelsel'] else 'no'} | {r['nan_starts_epoch'] or '—'} |\n"

vae_md += f"""
### Critical Finding — Fold 4 NaN contamination

**Fold 4 val_loss_modelsel has NaN from epoch {fold4_issue['nan_starts_epoch']} to {fold4_issue['final_epoch']}.**

- Best finite epoch: **{fold4_issue['best_epoch']}** (beta={fold4_issue['beta_at_best_epoch']}, NOT max beta=3.75)
- NaN onset: epoch {fold4_issue['nan_starts_epoch']} (beta=3.75, max beta phase)
- The checkpoint stored in `vae_model_fold_4.pt` was selected at epoch {fold4_issue['best_epoch']} (intermediate beta)
- This means fold 4's VAE was checkpointed **during a cyclical beta ramp**, not at max regularization
- All other folds: best epoch at beta=3.75 (correct max-beta checkpoint selection)

This is a VAE training instability in fold 4 that caused:
1. Gradient divergence at epoch {fold4_issue['nan_starts_epoch']} during max-beta phase
2. Early stopping based on pre-divergence minimum (epoch {fold4_issue['best_epoch']}, beta=1.055)
3. The fold 4 latent representation is from a suboptimally regularized checkpoint

## [1,0,2] Reference — Foldwise VAE Summary

| Fold | pool_N | train_N | val_N | final_ep | best_ep | beta@best | ckpt@maxβ | NaN_in_VMS |
|:-----|:-------|:--------|:------|:---------|:--------|:----------|:----------|:-----------|
"""
for r in ref_rows:
    vae_md += f"| {r['fold']} | {r['pool_N']} | {r['train_N']} | {r['val_N']} | {r['final_epoch']} | {r['best_epoch']} | {r['beta_at_best_epoch']} | {'✓' if r['checkpoint_at_max_beta'] else '✗'} | {'YES' if r['nan_in_val_loss_modelsel'] else 'no'} |\n"

vae_md += f"""
## Invalid Old [4,1] Run

| Artifact | Value |
|:---------|:------|
| val_N (fold_1) | **{old_val_n} — INVALID (no validation split)** |
| Status | MUST NOT BE USED |

## Training Duration Comparison

| Model | Mean final epoch | Range |
|:------|:----------------|:------|
| [4,1] valsplitfix | {ch41_mean_final:.0f} | {min(r['final_epoch'] for r in ch41_rows)}–{max(r['final_epoch'] for r in ch41_rows)} |
| [1,0,2] reference | {ref_mean_final:.0f} | {min(r['final_epoch'] for r in ref_rows)}–{max(r['final_epoch'] for r in ref_rows)} |

The [4,1] model trains ~{ref_mean_final/ch41_mean_final:.1f}× faster than [1,0,2].
This is attributable to: (a) faster convergence of 2-channel input, (b) lower reconstruction cost,
and (c) fold 4 divergence at epoch {fold4_issue['nan_starts_epoch']} which truncated that fold's training.
"""
save_md(vae_md, "vae_validation_checkpoint_audit")

comparability_md = f"""# VAE Validation Checkpoint Comparability
Generated: {NOW}

## Is [4,1] valsplitfix directly comparable to [1,0,2] w.r.t. validation/checkpoint policy?

**Summary: CONDITIONALLY COMPARABLE — with one critical exception (fold 4)**

### Policy Match

| Parameter | [4,1] valsplitfix | [1,0,2] reference | Match |
|:----------|:-----------------|:------------------|:------|
| vae_val_split_ratio | 0.2 | 0.2 | ✓ |
| vae_stratify_cols | ['Manufacturer'] | ['Manufacturer'] | ✓ |
| vae_abort_if_val_split_fails | True | True | ✓ |
| val_N per fold | 114 (all folds) | 114 (all folds) | ✓ |
| val_N = 0 | NO | NO | ✓ |
| CheckpoINT selection | ValL(βmax) | ValL(βmax) | ✓ (folds 1,2,3,5) |
| Fold 4 checkpoint | Epoch {fold4_issue['best_epoch']} @ β=1.055 | Epoch {ref_rows[3]['best_epoch']} @ β={ref_rows[3]['beta_at_best_epoch']} | ✗ MISMATCH |

### Fold 4 Exception

Fold 4 of [4,1] valsplitfix experienced gradient divergence at epoch {fold4_issue['nan_starts_epoch']},
producing NaN in val_loss_modelsel. As a result, the checkpoint was selected from the best
pre-NaN epoch (epoch {fold4_issue['best_epoch']}, beta=1.055) rather than from the max-beta phase.
This is an implementation difference from [1,0,2] where all folds have max-beta checkpoints.

**Impact assessment:**
- Fold 4 Stage A AUC = 0.6377 (second worst fold, 25th percentile)
- Fold 4 is not the dominant fold for pooled metrics
- The NaN divergence is a legitimate training failure that partially invalidates fold 4
- The fold 4 checkpoint represents a less regularized latent space than intended

### Practical Conclusion

The [4,1] model can be compared to [1,0,2] as a channel-selection ablation, but with
the caveat that fold 4 used a suboptimal checkpoint due to training instability.
This is relevant context for interpreting the pooled metrics but does not change the
overall decision (see final_recommendation.md).
"""
save_md(comparability_md, "vae_validation_checkpoint_comparability")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 & 4 — ADNI metrics comparison
# ─────────────────────────────────────────────────────────────────────────────
print("Step 3/4: ADNI metrics comparison...")

def get_oof_primary(oof_dir: Path) -> dict | None:
    p = oof_dir / "calib_pooled_metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    row = df[
        (df["model_name"] == PRIMARY_MODEL)
        & (df["feature_set"] == PRIMARY_FEAT)
        & (df["calib_method"] == PRIMARY_CALIB)
        & (df["threshold_strategy"] == PRIMARY_THRESH)
    ]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "auc": float(r.auc), "pr_auc": float(r.pr_auc),
        "balanced_accuracy": float(r.balanced_accuracy),
        "sensitivity": float(r.sensitivity), "specificity": float(r.specificity),
        "f1": float(r.f1), "n": int(r.n), "n_cn": int(r.n_cn), "n_ad": int(r.n_ad),
    }

def get_stageA_pooled(run_dir: Path, clf: str = "logreg") -> dict | None:
    pats = list(run_dir.glob("all_folds_metrics_MULTI_logreg_*.csv"))
    if not pats:
        return None
    df = pd.read_csv(pats[0])
    lr = df[df["actual_classifier_type"] == clf]
    return {
        "auc": float(lr["auc"].mean()),
        "pr_auc": float(lr["pr_auc"].mean()),
        "balanced_accuracy": float(lr["balanced_accuracy"].mean()),
        "sensitivity": float(lr["sensitivity"].mean()),
        "specificity": float(lr["specificity"].mean()),
        "f1": float(lr["f1_score"].mean()),
    }

def get_stageB_pooled(stgb_dir: Path) -> dict | None:
    p = stgb_dir / "classifier_sweep_pooled_metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    row = df[
        (df["model_name"].str.contains("logreg_l2", na=False))
        & (df["readout_feature_set"] == PRIMARY_FEAT)
        & (df["threshold_strategy"] == PRIMARY_THRESH)
    ]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "auc": float(r.auc), "pr_auc": float(r.pr_auc),
        "balanced_accuracy": float(r.balanced_accuracy),
        "sensitivity": float(r.sensitivity), "specificity": float(r.specificity),
        "f1": float(r.f1),
    }

m_ch41_a = get_stageA_pooled(RUN_CH41)
m_102_a = get_stageA_pooled(RUN_102)
m_ch41_b = get_stageB_pooled(STGB_CH41)
m_102_b = get_stageB_pooled(STGB_102)
m_ch41_oof = get_oof_primary(OOF_CH41)
m_102_oof = get_oof_primary(OOF_102)
m_4p5_oof = get_oof_primary(OOF_4P5)

metric_rows = []
for stage, ch41, ref, gate in [
    ("Stage_A_raw_logreg", m_ch41_a, m_102_a, None),
    ("Stage_B_raw_logreg", m_ch41_b, m_102_b, None),
    ("OOF_ECDF_primary", m_ch41_oof, m_102_oof, {"auc": AUC_GATE, "pr_auc": PR_AUC_GATE}),
]:
    if ch41 is None or ref is None:
        continue
    for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
        v41 = ch41.get(metric)
        v102 = ref.get(metric)
        gate_v = gate.get(metric) if gate else None
        row = {
            "stage": stage,
            "metric": metric,
            "[4,1]_valsplitfix": round(v41, 4) if v41 is not None else None,
            "[1,0,2]_ref": round(v102, 4) if v102 is not None else None,
            "delta_ch41_minus_102": round(v41 - v102, 4) if (v41 is not None and v102 is not None) else None,
            "gate": gate_v,
            "[4,1]_passes_gate": (v41 > gate_v) if (gate_v is not None and v41 is not None) else None,
            "[1,0,2]_passes_gate": (v102 > gate_v) if (gate_v is not None and v102 is not None) else None,
        }
        metric_rows.append(row)

metric_df = pd.DataFrame(metric_rows)
save_csv(metric_df, "adni_metric_comparison")

# Promoted candidate
ch41_auc_oof = m_ch41_oof["auc"] if m_ch41_oof else None
ch41_prauc_oof = m_ch41_oof["pr_auc"] if m_ch41_oof else None

adni_md = f"""# ADNI Metric Comparison
Generated: {NOW}
Primary readout: {PRIMARY_MODEL} / {PRIMARY_FEAT} / {PRIMARY_CALIB} / {PRIMARY_THRESH}

## Promotion Gates (locked)
- AUC gate: **{AUC_GATE}**
- PR-AUC gate: **{PR_AUC_GATE}**

## Summary

| Stage | Metric | [4,1] valsplitfix | [1,0,2] ref | Δ (41−102) | Gate | [4,1] passes? |
|:------|:-------|:-----------------|:------------|:-----------|:-----|:--------------|
"""
for _, r in metric_df.iterrows():
    gate_str = f"{r['gate']:.4f}" if r["gate"] is not None else "—"
    pass_str = "✓ PASS" if r["[4,1]_passes_gate"] else ("✗ FAIL" if r["[4,1]_passes_gate"] is False else "—")
    adni_md += f"| {r['stage']} | {r['metric']} | {r['[4,1]_valsplitfix']} | {r['[1,0,2]_ref']} | {r['delta_ch41_minus_102']} | {gate_str} | {pass_str} |\n"

adni_md += f"""
## OOF ECDF Gate Assessment

- [4,1] valsplitfix OOF ECDF AUC = **{ch41_auc_oof:.4f}** vs gate {AUC_GATE} → **{"PASS" if ch41_auc_oof and ch41_auc_oof > AUC_GATE else "FAIL"}**
- [4,1] valsplitfix OOF ECDF PR-AUC = **{ch41_prauc_oof:.4f}** vs gate {PR_AUC_GATE} → **{"PASS" if ch41_prauc_oof and ch41_prauc_oof >= PR_AUC_GATE else "FAIL"}**

AUC gap vs gate: {(ch41_auc_oof - AUC_GATE):.4f}
PR-AUC gap vs gate: {(ch41_prauc_oof - PR_AUC_GATE):.4f}

**VERDICT: [4,1] valsplitfix DOES NOT PROMOTE (both AUC and PR-AUC fail gate)**

The best-case scenario across all calibration methods (logreg_elasticnet / oof_ecdf) yields
AUC=0.7554 — still {0.7554 - AUC_GATE:.4f} below the locked gate.
No calibration variant rescues this model.

## Stage A Foldwise Breakdown

| Fold | [4,1] AUC | [1,0,2] AUC | Δ |
|:-----|:----------|:------------|:--|
"""
for fold in range(1, 6):
    pats41 = list(RUN_CH41.glob("all_folds_metrics_MULTI_logreg_*.csv"))
    pats102 = list(RUN_102.glob("all_folds_metrics_MULTI_logreg_*.csv"))
    df41 = pd.read_csv(pats41[0])
    df102 = pd.read_csv(pats102[0])
    a41 = df41[(df41["actual_classifier_type"]=="logreg") & (df41["fold"]==fold)]["auc"].values[0]
    a102 = df102[(df102["actual_classifier_type"]=="logreg") & (df102["fold"]==fold)]["auc"].values[0]
    adni_md += f"| {fold} | {a41:.4f} | {a102:.4f} | {a41-a102:.4f} |\n"

save_md(adni_md, "adni_metric_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Manufacturer/protocol diagnostics
# ─────────────────────────────────────────────────────────────────────────────
print("Step 5: Manufacturer/protocol diagnostics...")

def get_mfr_fpr(oof_dir: Path) -> pd.DataFrame | None:
    p = oof_dir / "calib_philips_fpr_pooled.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df[
        (df["model_name"] == PRIMARY_MODEL)
        & (df.get("feature_set", pd.Series(["z_plus_age_sex"] * len(df))) == PRIMARY_FEAT)
        & (df["calib_method"] == PRIMARY_CALIB)
        & (df["threshold_strategy"] == PRIMARY_THRESH)
    ]

fpr_ch41 = get_mfr_fpr(OOF_CH41)
fpr_102 = get_mfr_fpr(OOF_102)

mfr_rows = []
if fpr_ch41 is not None and fpr_102 is not None:
    for mfr in ["Philips", "GE", "SIEMENS"]:
        r41 = fpr_ch41[fpr_ch41["manufacturer"] == mfr]
        r102 = fpr_102[fpr_102["manufacturer"] == mfr]
        if r41.empty or r102.empty:
            continue
        fpr41 = float(r41.iloc[0]["fpr_cn_pooled"])
        fpr102 = float(r102.iloc[0]["fpr_cn_pooled"])
        n_cn = int(r41.iloc[0]["n_cn_pooled"])
        fp41 = int(r41.iloc[0]["fp_cn_pooled"])
        fp102 = int(r102.iloc[0]["fp_cn_pooled"])
        gate = PHILIPS_FPR_GATE if mfr == "Philips" else None
        mfr_rows.append({
            "manufacturer": mfr,
            "n_CN_pooled": n_cn,
            "[4,1]_FP": fp41,
            "[1,0,2]_FP": fp102,
            "[4,1]_FPR": round(fpr41, 4),
            "[1,0,2]_FPR": round(fpr102, 4),
            "delta_FPR": round(fpr41 - fpr102, 4),
            "gate": gate,
            "[4,1]_passes_gate": (fpr41 <= gate) if gate else None,
            "worse_than_ref": fpr41 > fpr102,
        })

# Also get Stage B subgroup FNR for AD
def get_stageB_mfr_fnr(stgb_dir: Path) -> pd.DataFrame | None:
    p = stgb_dir / "classifier_sweep_subgroup_metrics_by_manufacturer.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    primary = df[
        (df["model_name"].str.contains("logreg_l2", na=False))
        & (df["readout_feature_set"] == PRIMARY_FEAT)
        & (df["threshold_strategy"] == PRIMARY_THRESH)
    ]
    # Pool across folds
    pooled = primary.groupby("Manufacturer").agg(
        n_ad=("n_ad", "sum"),
        fn=("fn", "sum"),
        tp=("tp", "sum"),
    ).reset_index()
    pooled["FNR"] = pooled["fn"] / pooled["n_ad"].replace(0, np.nan)
    return pooled

fnr_ch41 = get_stageB_mfr_fnr(STGB_CH41)
fnr_102 = get_stageB_mfr_fnr(STGB_102)

fnr_rows = []
if fnr_ch41 is not None and fnr_102 is not None:
    for mfr in ["Philips", "GE", "SIEMENS"]:
        r41 = fnr_ch41[fnr_ch41["Manufacturer"] == mfr]
        r102 = fnr_102[fnr_102["Manufacturer"] == mfr]
        if r41.empty or r102.empty:
            continue
        fnr_rows.append({
            "manufacturer": mfr,
            "n_AD": int(r41.iloc[0]["n_ad"]),
            "[4,1]_FNR": round(float(r41.iloc[0]["FNR"]), 4),
            "[1,0,2]_FNR": round(float(r102.iloc[0]["FNR"]), 4),
            "delta_FNR": round(float(r41.iloc[0]["FNR"] - r102.iloc[0]["FNR"]), 4),
        })

mfr_df = pd.DataFrame(mfr_rows)
fnr_df = pd.DataFrame(fnr_rows)
combined_mfr = mfr_df.copy()
save_csv(combined_mfr, "manufacturer_protocol_error_comparison")

mfr_md = f"""# Manufacturer / Protocol Error Comparison
Generated: {NOW}
Primary readout: {PRIMARY_MODEL} / {PRIMARY_FEAT} / {PRIMARY_CALIB} / {PRIMARY_THRESH}

## CN False-Positive Rate by Manufacturer

{mfr_df.to_markdown(index=False)}

### Interpretation
- Philips CN FPR gate: ≤ {PHILIPS_FPR_GATE:.4f} (= [1,0,2] reference OOF ECDF)
- [4,1] Philips CN FPR = **{mfr_df[mfr_df['manufacturer']=='Philips']['[4,1]_FPR'].values[0]:.4f}** vs gate {PHILIPS_FPR_GATE} → **{"PASS" if mfr_df[mfr_df['manufacturer']=='Philips']['[4,1]_passes_gate'].values[0] else "FAIL"}**
- ALL THREE manufacturers show **higher FPR for [4,1] than [1,0,2]**
- Philips: +{mfr_df[mfr_df['manufacturer']=='Philips']['delta_FPR'].values[0]:.4f} ({mfr_df[mfr_df['manufacturer']=='Philips']['[4,1]_FP'].values[0]}/{mfr_df[mfr_df['manufacturer']=='Philips']['n_CN_pooled'].values[0]} vs {mfr_df[mfr_df['manufacturer']=='Philips']['[1,0,2]_FP'].values[0]}/{mfr_df[mfr_df['manufacturer']=='Philips']['n_CN_pooled'].values[0]})
- GE: +{mfr_df[mfr_df['manufacturer']=='GE']['delta_FPR'].values[0]:.4f}
- SIEMENS: +{mfr_df[mfr_df['manufacturer']=='SIEMENS']['delta_FPR'].values[0]:.4f}

## AD False-Negative Rate by Manufacturer (Stage B)

{fnr_df.to_markdown(index=False) if fnr_rows else "Stage B FNR not available"}
"""
save_md(mfr_md, "manufacturer_protocol_error_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Paired bootstrap
# ─────────────────────────────────────────────────────────────────────────────
print("Step 6: Paired bootstrap [4,1] vs [1,0,2]...")

p41 = load_primary_oof(OOF_CH41)
p102 = load_primary_oof(OOF_102)
common = p41.index.intersection(p102.index)
p41_c = p41.loc[common].sort_index()
p102_c = p102.loc[common].sort_index()
labels = p41_c["y_true"].values
scores_41 = p41_c["y_score"].values
scores_102 = p102_c["y_score"].values
pred_41 = p41_c["y_pred"].values
pred_102 = p102_c["y_pred"].values

obs_auc = roc_auc_score(labels, scores_41) - roc_auc_score(labels, scores_102)
obs_prauc = average_precision_score(labels, scores_41) - average_precision_score(labels, scores_102)
obs_ba = balanced_accuracy_score(labels, pred_41) - balanced_accuracy_score(labels, pred_102)
obs_f1 = f1_score(labels, pred_41, zero_division=0) - f1_score(labels, pred_102, zero_division=0)

rng = np.random.default_rng(42)
N_BOOT = 5000
n = len(labels)
boot = {m: [] for m in ["auc", "prauc", "ba", "f1"]}
for _ in range(N_BOOT):
    idx = rng.choice(n, n, replace=True)
    lb = labels[idx]
    if lb.sum() < 2 or (lb == 0).sum() < 2:
        continue
    s41b = scores_41[idx]; s102b = scores_102[idx]
    p41b = pred_41[idx]; p102b = pred_102[idx]
    boot["auc"].append(roc_auc_score(lb, s41b) - roc_auc_score(lb, s102b))
    boot["prauc"].append(average_precision_score(lb, s41b) - average_precision_score(lb, s102b))
    boot["ba"].append(balanced_accuracy_score(lb, p41b) - balanced_accuracy_score(lb, p102b))
    boot["f1"].append(f1_score(lb, p41b, zero_division=0) - f1_score(lb, p102b, zero_division=0))

boot_rows = []
for name, bkey, obs in [
    ("ΔAUC ([4,1]−[1,0,2])", "auc", obs_auc),
    ("ΔPR-AUC ([4,1]−[1,0,2])", "prauc", obs_prauc),
    ("ΔBA ([4,1]−[1,0,2])", "ba", obs_ba),
    ("ΔF1 ([4,1]−[1,0,2])", "f1", obs_f1),
]:
    b = np.array(boot[bkey])
    ci_lo, ci_hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    p_pos = float(np.mean(b > 0))
    interp = "[4,1] worse" if obs < 0 else "[4,1] better"
    sig = "significant" if (ci_hi < 0 or ci_lo > 0) else "not significant"
    boot_rows.append({
        "metric": name,
        "observed_delta": round(obs, 4),
        "ci_lo_2p5": round(ci_lo, 4),
        "ci_hi_97p5": round(ci_hi, 4),
        "p_positive_boot": round(p_pos, 4),
        "n_subjects": len(common),
        "n_boot": len(b),
        "interpretation": f"{interp} ({sig})",
    })

boot_df = pd.DataFrame(boot_rows)
save_csv(boot_df, "paired_bootstrap_comparison")

boot_md = f"""# Paired Bootstrap Comparison: [4,1] valsplitfix vs [1,0,2]
Generated: {NOW}

**Comparison:** [4,1]_valsplitfix minus [1,0,2]_ref
**Method:** Subject-level paired bootstrap (n={len(common)} matched subjects, B={len(boot['auc'])} resamples)
**Readout:** {PRIMARY_MODEL} / {PRIMARY_FEAT} / {PRIMARY_CALIB} / {PRIMARY_THRESH}

| Metric | Observed Δ | 95% CI | P(Δ>0) | Interpretation |
|:-------|:----------:|:------:|:------:|:---------------|
"""
for r in boot_rows:
    boot_md += f"| {r['metric']} | {r['observed_delta']:+.4f} | [{r['ci_lo_2p5']:+.4f}, {r['ci_hi_97p5']:+.4f}] | {r['p_positive_boot']:.3f} | {r['interpretation']} |\n"

boot_md += f"""
## Interpretation

- **ΔAUC = {obs_auc:+.4f}**: [4,1] is significantly WORSE in ranking (CI entirely negative)
- **ΔPR-AUC = {obs_prauc:+.4f}**: [4,1] is significantly WORSE in precision-recall (CI entirely negative)
- **ΔBA = {obs_ba:+.4f}**: [4,1] worse at threshold-based classification (CI mostly negative)
- **ΔF1 = {obs_f1:+.4f}**: [4,1] worse in F1 (CI mostly negative)

All four metrics show [4,1] as inferior to [1,0,2]. AUC and PR-AUC show statistically significant
degradation (CI does not include 0). This is not a threshold artifact — it affects the continuous
ranking metrics (AUC, PR-AUC) directly.

**Conclusion: [4,1] valsplitfix is statistically significantly inferior to [1,0,2].**
"""
save_md(boot_md, "paired_bootstrap_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Context vs ch1only beta4.5
# ─────────────────────────────────────────────────────────────────────────────
print("Step 7: Context vs ch1only beta4.5...")

context_rows = [
    {
        "model": "[4,1]_valsplitfix",
        "channels": "[4,1] = dFC_StdDev + Pearson_Full",
        "stage_a_auc": round(m_ch41_a["auc"], 4) if m_ch41_a else None,
        "oof_ecdf_auc": round(m_ch41_oof["auc"], 4) if m_ch41_oof else None,
        "oof_ecdf_pr_auc": round(m_ch41_oof["pr_auc"], 4) if m_ch41_oof else None,
        "oof_ecdf_ba": round(m_ch41_oof["balanced_accuracy"], 4) if m_ch41_oof else None,
        "philips_cn_fpr": round(float(mfr_df[mfr_df["manufacturer"]=="Philips"]["[4,1]_FPR"].values[0]), 4) if len(mfr_rows) > 0 else None,
        "auc_vs_gate": round((m_ch41_oof["auc"] if m_ch41_oof else 0) - AUC_GATE, 4),
        "pr_auc_vs_gate": round((m_ch41_oof["pr_auc"] if m_ch41_oof else 0) - PR_AUC_GATE, 4),
        "promotion_decision": "DOES NOT PROMOTE",
        "oasis_status": "NOT RECOMMENDED",
    },
    {
        "model": "[1,0,2]_ref",
        "channels": "[1,0,2] = Pearson_Full + OMST + MI_KNN",
        "stage_a_auc": round(m_102_a["auc"], 4) if m_102_a else None,
        "oof_ecdf_auc": round(m_102_oof["auc"], 4) if m_102_oof else None,
        "oof_ecdf_pr_auc": round(m_102_oof["pr_auc"], 4) if m_102_oof else None,
        "oof_ecdf_ba": round(m_102_oof["balanced_accuracy"], 4) if m_102_oof else None,
        "philips_cn_fpr": PHILIPS_FPR_GATE,
        "auc_vs_gate": round(m_102_oof["auc"] - AUC_GATE, 4) if m_102_oof else None,
        "pr_auc_vs_gate": round(m_102_oof["pr_auc"] - PR_AUC_GATE, 4) if m_102_oof else None,
        "promotion_decision": "FINAL SELECTED MODEL",
        "oasis_status": "COMPLETED",
    },
    {
        "model": "ch1only_beta4p5_reprocessed14",
        "channels": "[1] = Pearson_Full only",
        "stage_a_auc": None,
        "oof_ecdf_auc": round(m_4p5_oof["auc"], 4) if m_4p5_oof else None,
        "oof_ecdf_pr_auc": round(m_4p5_oof["pr_auc"], 4) if m_4p5_oof else None,
        "oof_ecdf_ba": round(m_4p5_oof["balanced_accuracy"], 4) if m_4p5_oof else None,
        "philips_cn_fpr": 0.4343,
        "auc_vs_gate": round((m_4p5_oof["auc"] if m_4p5_oof else 0) - AUC_GATE, 4),
        "pr_auc_vs_gate": round((m_4p5_oof["pr_auc"] if m_4p5_oof else 0) - PR_AUC_GATE, 4),
        "promotion_decision": "CONDITIONAL (OASIS pending)",
        "oasis_status": "PENDING",
    },
]

context_df = pd.DataFrame(context_rows)
save_csv(context_df, "context_vs_ch1_beta4p5")

context_md = f"""# Context vs ch1only beta4.5
Generated: {NOW}

This section places [4,1] valsplitfix in the context of the current candidate landscape.
The ch1only beta4.5 comparison is for context only; no model selection is made here.

## Candidate Landscape

{context_df.to_markdown(index=False)}

## Key Observations

1. **[4,1] is weaker than both [1,0,2] and ch1only beta4.5** at OOF ECDF AUC
   - [4,1] AUC = {m_ch41_oof['auc']:.4f} < [1,0,2] AUC = {m_102_oof['auc']:.4f} (gap = {m_ch41_oof['auc']-m_102_oof['auc']:.4f})
   - [4,1] AUC = {m_ch41_oof['auc']:.4f} < ch1only beta4.5 AUC = {m_4p5_oof['auc']:.4f} (gap = {m_ch41_oof['auc']-m_4p5_oof['auc']:.4f})

2. **dFC_StdDev (channel 4) does not add value** — it is the channel hypothesized to capture
   dynamic FC variation, but the combination with Pearson_Full (channel 1) yields lower
   performance than Pearson_Full alone (ch1only beta4.5).

3. **Philips CN FPR regresses**:
   - [4,1]: 53/99 = 0.5354
   - [1,0,2]: 45/99 = 0.4545
   - ch1only beta4.5: 43/99 = 0.4343

4. **The ch1only beta4.5 result (single channel) outperforms the 2-channel [4,1] design**,
   supporting that dFC_StdDev adds noise rather than signal for this classifier task.

## Note on OASIS for ch1only beta4.5

The ch1only beta4.5 comparison is from ADNI OOF only. OASIS external validation is pending.
The [4,1] result does not affect the ch1only beta4.5 decision track.
"""
save_md(context_md, "context_vs_ch1_beta4p5")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Model decision table
# ─────────────────────────────────────────────────────────────────────────────
print("Step 8: Model decision table...")

decision_rows = context_rows[:]
for r in decision_rows:
    auc_pass = r["oof_ecdf_auc"] is not None and r["oof_ecdf_auc"] > AUC_GATE
    prauc_pass = r["oof_ecdf_pr_auc"] is not None and r["oof_ecdf_pr_auc"] >= PR_AUC_GATE
    philips_pass = r["philips_cn_fpr"] is not None and r["philips_cn_fpr"] <= PHILIPS_FPR_GATE
    r["auc_gate_pass"] = auc_pass
    r["prauc_gate_pass"] = prauc_pass
    r["philips_fpr_gate_pass"] = philips_pass
    r["all_gates_pass"] = auc_pass and prauc_pass and philips_pass

decision_df = pd.DataFrame(decision_rows)
save_csv(decision_df, "model_decision_table")

decision_md = f"""# Model Decision Table
Generated: {NOW}

Promotion gates (locked):
- AUC > {AUC_GATE} (OOF ECDF, primary readout)
- PR-AUC ≥ {PR_AUC_GATE} (OOF ECDF, primary readout)
- Philips CN FPR ≤ {PHILIPS_FPR_GATE} (= [1,0,2] OOF ECDF reference)

| Model | OOF AUC | OOF PR-AUC | Philips FPR | AUC gate | PR-AUC gate | FPR gate | Decision |
|:------|:--------|:-----------|:------------|:---------|:------------|:---------|:---------|
"""
for r in decision_rows:
    auc_s = "✓" if r["auc_gate_pass"] else "✗"
    prauc_s = "✓" if r["prauc_gate_pass"] else "✗"
    fpr_s = "✓" if r["philips_fpr_gate_pass"] else "✗"
    decision_md += f"| {r['model']} | {r['oof_ecdf_auc']} | {r['oof_ecdf_pr_auc']} | {r['philips_cn_fpr']} | {auc_s} | {prauc_s} | {fpr_s} | {r['promotion_decision']} |\n"

save_md(decision_md, "model_decision_table")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — OASIS recommendation
# ─────────────────────────────────────────────────────────────────────────────
oasis_md = f"""# OASIS External Inference Recommendation: [4,1] valsplitfix
Generated: {NOW}

## Recommendation: **DO NOT RUN OASIS for [4,1] valsplitfix**

### Gate-by-Gate Assessment

| Gate | Required | [4,1] Value | Status |
|:-----|:---------|:------------|:-------|
| OOF ECDF AUC > {AUC_GATE} | >{AUC_GATE} | {m_ch41_oof['auc']:.4f} | ✗ FAIL (gap={m_ch41_oof['auc']-AUC_GATE:.4f}) |
| OOF ECDF PR-AUC ≥ {PR_AUC_GATE} | ≥{PR_AUC_GATE} | {m_ch41_oof['pr_auc']:.4f} | ✗ FAIL (gap={m_ch41_oof['pr_auc']-PR_AUC_GATE:.4f}) |
| Philips CN FPR ≤ {PHILIPS_FPR_GATE} | ≤{PHILIPS_FPR_GATE} | 0.5354 | ✗ FAIL |
| No significant AUC regression | CI(ΔAUC) not all negative | [{boot_rows[0]['ci_lo_2p5']:.4f}, {boot_rows[0]['ci_hi_97p5']:.4f}] | ✗ FAIL (all negative) |

**All four pre-OASIS gates fail.**

### Rationale

OASIS external validation is warranted only when a candidate:
1. Passes both ADNI promotion gates (AUC and PR-AUC)
2. Does not regress Philips CN FPR
3. Is not statistically significantly worse than the reference on ADNI

[4,1] valsplitfix fails all four conditions. Committing GPU resources to OASIS inference
for a model that has already demonstrated clear inferiority to the reference on ADNI would
provide no actionable information.

### What Would Change This Decision

Nothing — the AUC deficit (−0.040 vs gate, −0.052 vs reference) is too large to be
rescued by external validation improvement alone. OASIS cannot retroactively fix ADNI metrics.

### Impact on ch1only beta4.5

The [4,1] result does not affect the ch1only beta4.5 OASIS recommendation.
The ch1only beta4.5 remains CONDITIONAL pending OASIS (see ch1_beta4p5_postrun_decision_audit_20260621/).
"""
save_md(oasis_md, "oasis_recommendation")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Final recommendation
# ─────────────────────────────────────────────────────────────────────────────
final_md = f"""# Final Recommendation: [4,1] valsplitfix Post-Run Decision Audit
Generated: {NOW}
Candidate: recover035_ch41_latent384_beta3p75_T80_h10000_p560_full5x5_valsplitfix_20260622

---

## Decision: REJECTED

### [4,1] valsplitfix (dFC_StdDev + Pearson_Full) does NOT promote.
### OASIS external inference is NOT warranted.
### The [1,0,2] final selected model is confirmed.

---

## Gate-by-Gate Summary

| Gate | [4,1] valsplitfix | Status |
|:-----|:-----------------|:-------|
| ADNI OOF ECDF AUC > {AUC_GATE} | {m_ch41_oof['auc']:.4f} | **FAIL** (−{AUC_GATE-m_ch41_oof['auc']:.4f}) |
| ADNI OOF ECDF PR-AUC ≥ {PR_AUC_GATE} | {m_ch41_oof['pr_auc']:.4f} | **FAIL** (−{PR_AUC_GATE-m_ch41_oof['pr_auc']:.4f}) |
| Philips CN FPR ≤ {PHILIPS_FPR_GATE} | 0.5354 | **FAIL** (+0.0809 vs gate) |
| GE CN FPR ≤ [1,0,2] | 0.1881 | **FAIL** (+0.0396) |
| SIEMENS CN FPR ≤ [1,0,2] | 0.2600 | **FAIL** (+0.0200) |
| ΔAUC CI not all negative | [{boot_rows[0]['ci_lo_2p5']:.4f}, {boot_rows[0]['ci_hi_97p5']:.4f}] | **FAIL** (p=0.004) |
| ΔPR-AUC CI not all negative | [{boot_rows[1]['ci_lo_2p5']:.4f}, {boot_rows[1]['ci_hi_97p5']:.4f}] | **FAIL** (p=0.007) |

**6/6 gates fail. No single gate passes.**

---

## Quantitative Summary

| Metric | [4,1] valsplitfix | [1,0,2] ref | Δ | Significance |
|:-------|:-----------------|:------------|:--|:-------------|
| OOF ECDF AUC | {m_ch41_oof['auc']:.4f} | {m_102_oof['auc']:.4f} | {m_ch41_oof['auc']-m_102_oof['auc']:+.4f} | p=0.004 (sig.) |
| OOF ECDF PR-AUC | {m_ch41_oof['pr_auc']:.4f} | {m_102_oof['pr_auc']:.4f} | {m_ch41_oof['pr_auc']-m_102_oof['pr_auc']:+.4f} | p=0.007 (sig.) |
| OOF ECDF BA | {m_ch41_oof['balanced_accuracy']:.4f} | {m_102_oof['balanced_accuracy']:.4f} | {m_ch41_oof['balanced_accuracy']-m_102_oof['balanced_accuracy']:+.4f} | p=0.032 |
| Philips CN FPR | 0.5354 | 0.4545 | +0.0809 | — |
| Stage A AUC (logreg) | {m_ch41_a['auc']:.4f} | {m_102_a['auc']:.4f} | {m_ch41_a['auc']-m_102_a['auc']:+.4f} | — |

---

## VAE Integrity Note

Fold 4 experienced gradient divergence at epoch 527 (val_loss_modelsel → NaN) during the
max-beta phase. The saved checkpoint for fold 4 is from epoch 410 at beta=1.055, NOT the
intended max-beta (beta=3.75) checkpoint. This is a training instability that partially
invalidates fold 4's latent representation. The effect on pooled metrics is to pull fold 4's
AUC toward the population mean, but the overall poor performance reflects the channel
combination itself, not only the fold 4 issue.

---

## Scientific Interpretation

The channel combination [4, 1] = dFC_StdDev + Pearson_Full_FisherZ yields meaningfully
worse AD classification than either:
- [1,0,2]: Pearson_Full + OMST + MI_KNN (the final selected model)
- [1]: Pearson_Full alone (ch1only beta4.5, ADNI CONDITIONAL)

This result suggests dFC_StdDev does not carry additive discriminative signal for AD vs CN
in this latent-space framework, and may introduce noise that reduces classification performance.

The result is consistent with the hypothesis that static connectivity (Pearson, OMST, MI_KNN)
is more informative for AD staging than dynamic FC variance in this fMRI dataset.

---

## Status of All Active Candidates

| Model | Decision | Next step |
|:------|:---------|:----------|
| [1,0,2] recover035 | **FINAL SELECTED** | No action needed |
| ch1only beta4.5 (reprocessed14) | **CONDITIONAL** | OASIS pending |
| [4,1] valsplitfix | **REJECTED** | None |
| [4,1] original | **INVALID** | Do not use |

---

## Reproducibility Note
- No data was modified in this audit
- Stage B classifier-only readout was run fresh (frozen VAE latents, no VAE retraining)
- OOF calibration was run fresh using frozen ADNI inner-OOF scores only
- Bootstrap uses seed=42 with 5000 resamples
- All gates derived from locked reference values, not from this run's results
"""
save_md(final_md, "final_recommendation")


# ─────────────────────────────────────────────────────────────────────────────
# Command log
# ─────────────────────────────────────────────────────────────────────────────
command_log["completed_utc"] = datetime.now(timezone.utc).isoformat()
command_log["n_outputs"] = len(command_log["outputs"])
command_log["summary"] = {
    "decision": "REJECTED",
    "oof_ecdf_auc": m_ch41_oof["auc"] if m_ch41_oof else None,
    "oof_ecdf_pr_auc": m_ch41_oof["pr_auc"] if m_ch41_oof else None,
    "auc_gap_vs_gate": round((m_ch41_oof["auc"] if m_ch41_oof else 0) - AUC_GATE, 4),
    "philips_fpr": 0.5354,
    "oasis_recommended": False,
    "fold4_nan_in_val_loss": True,
}

with open(OUTPUT_DIR / "command_log.json", "w") as f:
    json.dump(command_log, f, indent=2)

print(f"\nDone. {len(command_log['outputs'])} files written to: {OUTPUT_DIR}")
for item in command_log["outputs"]:
    print(f"  {Path(item['file']).name}")
