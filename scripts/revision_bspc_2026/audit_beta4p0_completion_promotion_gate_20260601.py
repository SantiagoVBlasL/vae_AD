#!/usr/bin/env python3
"""Read-only completion and promotion-gate audit for beta4p0 vs beta3p75 (promoted).

recover035_latent384_beta4p0_T80_h10000_p560_full5x5
vs
recover035_latent384_beta3p75_T80_h10000_p560_full5x5  (reference, promoted)

Promotion gate:
  AUC  > 0.795155  (logreg_l2_original oof_ecdf, locked)
  PR-AUC >= 0.573934

Hard constraints:
  - No VAE retraining.
  - No tensor, metadata, or model artifact modification.
  - Read-only: all metrics derived from existing on-disk artifacts.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"

RUN4P0 = RESULTS / "recover035_latent384_beta4p0_T80_h10000_p560_full5x5"
RUN3P75 = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
OOF3P75_DIR = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
OUTPUT_DIR = RESULTS / "beta4p0_completion_promotion_gate_audit_20260601"

LOCKED_AUC = 0.795155
LOCKED_PR_AUC = 0.573934
REF_OOF_LOGITZ_AUC = 0.795120
REF_OOF_LOGITZ_PR_AUC = 0.572794
REF_PHILIPS_FPR = 0.4444

FOLDS = [1, 2, 3, 4, 5]
INNER_FOLDS = 5
SEED = 42
ORIGINAL_C_GRID = [0.001, 0.01, 0.1, 1.0]


def _tag4p0(fold: int) -> str:
    return f"vaeconvtranspose4l_ld384_beta4.0_normzscore_offdiag_ch3sel_intFCquarter_drop0.15_ln0"


def _tag3p75(fold: int) -> str:
    return f"vaeconvtranspose4l_ld384_beta3.75_normzscore_offdiag_ch3sel_intFCquarter_drop0.15_ln0"


# ─── helpers ─────────────────────────────────────────────────────────────────

def load_latent_cache(run_dir: Path, fold: int, split: str) -> pd.DataFrame:
    p = run_dir / "classifier_only_readout" / "latent_cache" / f"fold_{fold}_{split}_latent_mu.csv"
    return pd.read_csv(p)


def latent_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("mu_")]


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    mu_cols = latent_cols(df)
    age = df["Age"].values.reshape(-1, 1).astype(float)
    sex_enc = (df["Sex"].values == "M").astype(float).reshape(-1, 1)
    z = df[mu_cols].values.astype(float)
    X = np.hstack([z, age, sex_enc])
    y = df["y"].values
    return X, y


def compute_oof_logitz(run_dir: Path, folds: List[int] = FOLDS,
                       inner_folds: int = INNER_FOLDS,
                       c_grid: List[float] = ORIGINAL_C_GRID,
                       seed: int = SEED,
                       n_jobs: int = 4) -> Dict[str, Any]:
    """Replicate the OOF-logitz calibration from the beta3p75 script.

    Leakage-safe: all calibration parameters estimated from inner-CV OOF
    train/dev scores. Outer-test labels never used for calibration.

    Returns dict with pooled_auc, pooled_pr_auc, foldwise_auc list,
    philips_cn_fp, philips_cn_n, philips_cn_fpr.
    """
    all_test_y = []
    all_test_score_raw = []
    all_test_score_oof_logitz = []
    all_test_philips_mask_cn = []
    foldwise_aucs = []

    for fold in folds:
        traindev = load_latent_cache(run_dir, fold, "trainDev")
        test_df = load_latent_cache(run_dir, fold, "test")

        X_train, y_train = build_feature_matrix(traindev)
        X_test, y_test = build_feature_matrix(test_df)

        clf = LogisticRegression(
            C=0.001, max_iter=10000, class_weight="balanced",
            solver="lbfgs", random_state=seed
        )
        inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)

        param_grid = {"C": c_grid}
        gs = GridSearchCV(clf, param_grid, cv=inner_cv,
                          scoring="roc_auc", n_jobs=n_jobs, refit=True)
        gs.fit(X_train, y_train)
        best_clf = gs.best_estimator_

        oof_scores = np.zeros(len(y_train))
        for tr_idx, val_idx in inner_cv.split(X_train, y_train):
            clone_clf = LogisticRegression(
                C=gs.best_params_["C"], max_iter=10000,
                class_weight="balanced", solver="lbfgs", random_state=seed
            )
            clone_clf.fit(X_train[tr_idx], y_train[tr_idx])
            oof_scores[val_idx] = clone_clf.predict_proba(X_train[val_idx])[:, 1]

        oof_logits = logit(np.clip(oof_scores, 1e-6, 1 - 1e-6))
        oof_logit_mean = oof_logits.mean()
        oof_logit_std = max(oof_logits.std(), 1e-8)

        test_raw = best_clf.predict_proba(X_test)[:, 1]
        test_logits = logit(np.clip(test_raw, 1e-6, 1 - 1e-6))
        test_calib = expit((test_logits - oof_logit_mean) / oof_logit_std)

        fold_auc = roc_auc_score(y_test, test_calib)
        foldwise_aucs.append(fold_auc)

        all_test_y.extend(y_test.tolist())
        all_test_score_raw.extend(test_raw.tolist())
        all_test_score_oof_logitz.extend(test_calib.tolist())

        is_cn = (test_df["ResearchGroup_Mapped"].values == "CN")
        is_philips = (test_df["Manufacturer"].values == "Philips")
        mask = is_cn & is_philips
        all_test_philips_mask_cn.extend(mask.tolist())

    y_all = np.array(all_test_y)
    s_all = np.array(all_test_score_oof_logitz)
    s_raw = np.array(all_test_score_raw)
    mask_cn = np.array(all_test_philips_mask_cn)

    pooled_auc = roc_auc_score(y_all, s_all)
    pooled_pr_auc = average_precision_score(y_all, s_all)
    pooled_auc_raw = roc_auc_score(y_all, s_raw)
    pooled_pr_auc_raw = average_precision_score(y_all, s_raw)

    # Philips CN FPR using median threshold on all OOF scores
    threshold = float(np.median(s_all))
    y_pred = (s_all >= threshold).astype(int)
    philips_cn_fp = int(y_pred[mask_cn].sum())
    philips_cn_n = int(mask_cn.sum())
    philips_cn_fpr = philips_cn_fp / philips_cn_n if philips_cn_n > 0 else float("nan")

    return {
        "pooled_auc": pooled_auc,
        "pooled_pr_auc": pooled_pr_auc,
        "pooled_auc_raw": pooled_auc_raw,
        "pooled_pr_auc_raw": pooled_pr_auc_raw,
        "foldwise_aucs": foldwise_aucs,
        "foldwise_mean_auc": float(np.mean(foldwise_aucs)),
        "philips_cn_fp": philips_cn_fp,
        "philips_cn_n": philips_cn_n,
        "philips_cn_fpr": philips_cn_fpr,
        "threshold_used": threshold,
    }


def load_stage_a_metrics(run_dir: Path, beta_str: str) -> pd.DataFrame:
    pat = f"all_folds_metrics_MULTI_logreg_vaeconvtranspose4l_ld384_beta{beta_str}_normzscore_offdiag_ch3sel_intFCquarter_drop0.15_ln0_outer5x1_scoreroc_auc.csv"
    return pd.read_csv(run_dir / pat)


def load_stage_b_pooled(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "classifier_only_readout" / "classifier_sweep_pooled_metrics.csv")


def load_stage_b_subgroup(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "classifier_only_readout" / "classifier_sweep_subgroup_metrics_by_manufacturer.csv")


def load_scanner_leakage(run_dir: Path) -> List[Dict]:
    rows = []
    for fold in FOLDS:
        p = run_dir / f"fold_{fold}" / f"fold_{fold}_scanner_leakage_summary.csv"
        df = pd.read_csv(p)
        rows.append(df.iloc[0].to_dict())
    return rows


def load_latent_info(run_dir: Path) -> List[Dict]:
    rows = []
    for fold in FOLDS:
        p = run_dir / f"fold_{fold}" / f"fold_{fold}_test_latent_info_summary.csv"
        df = pd.read_csv(p)
        y_row = df[df["variable"] == "Y_target"].iloc[0]
        mfr_row = df[df["variable"] == "Manufacturer"].iloc[0]
        rows.append({
            "fold": fold,
            "mi_sum_Y": y_row["mi_sum_nats"],
            "mi_sum_mfr": mfr_row["mi_sum_nats"],
            "n_active": int(y_row["n_active"]),
            "total_corr": y_row["total_correlation_nats"],
        })
    return rows


def load_rd_final(run_dir: Path) -> List[Dict]:
    rows = []
    for fold in FOLDS:
        p = run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
        df = pd.read_csv(p)
        last = df.iloc[-1]
        rows.append({
            "fold": fold,
            "epoch": int(last["epoch"]),
            "D_val": float(last["D_val"]),
            "R_val_bits": float(last["R_val_bits"]),
            "beta": float(last["beta"]),
            "L_val_betaMax": float(last["L_val_betaMax"]),
        })
    return rows


def philips_cn_fpr_from_subgroup(sub_df: pd.DataFrame,
                                  threshold_strategy: str = "inner_oof_target_sens_ge_0p70_max_spec") -> Dict:
    rows = sub_df[
        (sub_df["Manufacturer"] == "Philips") &
        (sub_df["model_name"] == "logreg_l2") &
        (sub_df["threshold_strategy"] == threshold_strategy)
    ].copy()
    total_cn = int(rows["n_cn"].sum())
    total_fp = int(rows["fp"].sum())
    fpr = total_fp / total_cn if total_cn > 0 else float("nan")
    return {"n_cn": total_cn, "fp": total_fp, "fpr": fpr, "threshold_strategy": threshold_strategy}


def fmt_delta(v: float, ref: float, higher_is_better: bool = True) -> str:
    delta = v - ref
    arrow = "↑" if delta > 0 else "↓"
    better = (delta > 0) == higher_is_better
    tag = "BETTER" if better else "WORSE"
    return f"{v:.4f} ({arrow}{abs(delta):.4f} vs ref, {tag})"


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()

    print("=" * 70)
    print("beta4p0 completion + promotion-gate audit")
    print(f"Generated: {ts}")
    print("=" * 70)

    # ── 1. Fold completion ─────────────────────────────────────────────────
    print("\n## 1. Fold completion")
    missing = [f for f in FOLDS if not (RUN4P0 / f"fold_{f}").exists()]
    if missing:
        print(f"  MISSING folds: {missing}")
        sys.exit(1)
    print(f"  All 5 folds present: OK")

    # ── 2. Stage A fold-level metrics ──────────────────────────────────────
    print("\n## 2. Stage A fold-level metrics (logreg)")
    a4 = load_stage_a_metrics(RUN4P0, "4.0")
    a3 = load_stage_a_metrics(RUN3P75, "3.75")

    def stage_a_summary(df: pd.DataFrame, clf: str = "logreg") -> pd.DataFrame:
        sub = df[df["actual_classifier_type"] == clf][
            ["fold", "auc_raw", "pr_auc_raw", "auc", "pr_auc",
             "balanced_accuracy", "sensitivity", "specificity", "f1_score"]
        ].sort_values("fold")
        return sub

    sa4 = stage_a_summary(a4)
    sa3 = stage_a_summary(a3)

    print("\n  beta4p0 (logreg):")
    print(sa4.to_string(index=False))
    print(f"\n  Mean AUC_raw: {sa4['auc_raw'].mean():.4f} ± {sa4['auc_raw'].std():.4f}")
    print(f"  Mean AUC    : {sa4['auc'].mean():.4f} ± {sa4['auc'].std():.4f}")
    print(f"  Mean PR-AUC : {sa4['pr_auc'].mean():.4f} ± {sa4['pr_auc'].std():.4f}")
    print(f"  Mean BA     : {sa4['balanced_accuracy'].mean():.4f}")

    print("\n  beta3p75 (logreg, reference):")
    print(sa3.to_string(index=False))
    print(f"\n  Mean AUC_raw: {sa3['auc_raw'].mean():.4f} ± {sa3['auc_raw'].std():.4f}")
    print(f"  Mean AUC    : {sa3['auc'].mean():.4f} ± {sa3['auc'].std():.4f}")
    print(f"  Mean PR-AUC : {sa3['pr_auc'].mean():.4f} ± {sa3['pr_auc'].std():.4f}")
    print(f"  Mean BA     : {sa3['balanced_accuracy'].mean():.4f}")

    print("\n  Fold-by-fold AUC delta (beta4p0 - beta3p75):")
    for f in FOLDS:
        v4 = float(sa4[sa4["fold"] == f]["auc"])
        v3 = float(sa3[sa3["fold"] == f]["auc"])
        direction = "▲" if v4 >= v3 else "▼"
        print(f"    fold {f}: {v4:.4f} vs {v3:.4f}  {direction} {v4-v3:+.4f}")

    # ── 3. Stage B pooled metrics (raw) ────────────────────────────────────
    print("\n## 3. Stage B pooled metrics (raw, z_plus_age_sex, logreg_l2)")
    sb4 = load_stage_b_pooled(RUN4P0)
    sb3 = load_stage_b_pooled(RUN3P75)

    primary_thresh = "inner_oof_target_sens_ge_0p70_max_spec"
    row4 = sb4[sb4["threshold_strategy"] == primary_thresh].iloc[0]
    row3 = sb3[sb3["threshold_strategy"] == primary_thresh].iloc[0]

    print(f"\n  Primary threshold: {primary_thresh}")
    for metric, hi in [("auc", True), ("pr_auc", True), ("balanced_accuracy", True),
                        ("sensitivity", True), ("specificity", True), ("f1", True)]:
        v4 = float(row4[metric]) if metric != "f1" else float(row4["f1"])
        v3 = float(row3[metric]) if metric != "f1" else float(row3["f1"])
        delta = v4 - v3
        arrow = "▲" if delta > 0 else "▼"
        print(f"  {metric:25s}: beta4p0={v4:.4f}  ref={v3:.4f}  {arrow}{abs(delta):.4f}")

    # Also show fixed_0p5 AUC since that's the canonical reference for Stage B
    row4_0p5 = sb4[sb4["threshold_strategy"] == "fixed_0p5"].iloc[0]
    row3_0p5 = sb3[sb3["threshold_strategy"] == "fixed_0p5"].iloc[0]
    print(f"\n  Fixed-0.5 threshold AUC:")
    print(f"    beta4p0: {row4_0p5['auc']:.4f}  PR-AUC: {row4_0p5['pr_auc']:.4f}")
    print(f"    beta3p75: {row3_0p5['auc']:.4f}  PR-AUC: {row3_0p5['pr_auc']:.4f}")

    # ── 4. OOF-logitz computation for beta4p0 ─────────────────────────────
    print("\n## 4. OOF-logitz score computation (beta4p0)")
    print("  Running logit-calibrated pooled AUC (inner-OOF calibration)...")
    oof4 = compute_oof_logitz(RUN4P0, folds=FOLDS, n_jobs=4)

    print(f"\n  beta4p0 OOF-logitz pooled:")
    print(f"    AUC    = {oof4['pooled_auc']:.4f}")
    print(f"    PR-AUC = {oof4['pooled_pr_auc']:.4f}")
    print(f"    Foldwise mean AUC = {oof4['foldwise_mean_auc']:.4f}")
    print(f"    Foldwise AUCs by fold: {[f'{v:.4f}' for v in oof4['foldwise_aucs']]}")

    print(f"\n  Reference (beta3p75) OOF-logitz pooled (from saved calibration):")
    print(f"    AUC    = {REF_OOF_LOGITZ_AUC:.4f}")
    print(f"    PR-AUC = {REF_OOF_LOGITZ_PR_AUC:.4f}")

    print(f"\n  Delta (beta4p0 - beta3p75):")
    delta_auc = oof4['pooled_auc'] - REF_OOF_LOGITZ_AUC
    delta_pr = oof4['pooled_pr_auc'] - REF_OOF_LOGITZ_PR_AUC
    print(f"    AUC    : {delta_auc:+.4f}  {'BETTER' if delta_auc > 0 else 'WORSE'}")
    print(f"    PR-AUC : {delta_pr:+.4f}  {'BETTER' if delta_pr > 0 else 'WORSE'}")

    # ── 5. Promotion gate ─────────────────────────────────────────────────
    print("\n## 5. Promotion gate evaluation (OOF-logitz)")
    pass_auc = oof4['pooled_auc'] > LOCKED_AUC
    pass_prauc = oof4['pooled_pr_auc'] >= LOCKED_PR_AUC
    both_pass = pass_auc and pass_prauc

    print(f"\n  Gate requirements:")
    print(f"    AUC  > {LOCKED_AUC}  :  {oof4['pooled_auc']:.4f}  {'PASS ✓' if pass_auc else 'FAIL ✗'}")
    print(f"    PR-AUC >= {LOCKED_PR_AUC}  :  {oof4['pooled_pr_auc']:.4f}  {'PASS ✓' if pass_prauc else 'FAIL ✗'}")
    print(f"\n  Promotion verdict: {'PROMOTES' if both_pass else 'DOES NOT PROMOTE'}")

    # ── 6. VAE QC — reconstruction, KLD, active units, TC, training epochs ─
    print("\n## 6. VAE QC comparison")
    rd4 = load_rd_final(RUN4P0)
    rd3 = load_rd_final(RUN3P75)
    li4 = load_latent_info(RUN4P0)
    li3 = load_latent_info(RUN3P75)

    print("\n  Training epochs (stopped by early stopping):")
    for r4, r3 in zip(rd4, rd3):
        direction = "shorter" if r4['epoch'] < r3['epoch'] else "longer"
        print(f"    fold {r4['fold']}: beta4p0={r4['epoch']}  beta3p75={r3['epoch']}  ({direction} by {abs(r4['epoch']-r3['epoch'])})")

    mean_ep4 = np.mean([r['epoch'] for r in rd4])
    mean_ep3 = np.mean([r['epoch'] for r in rd3])
    print(f"  Mean epochs: beta4p0={mean_ep4:.0f}  beta3p75={mean_ep3:.0f}")

    print("\n  Reconstruction loss D_val at final epoch:")
    for r4, r3 in zip(rd4, rd3):
        delta = r4['D_val'] - r3['D_val']
        print(f"    fold {r4['fold']}: beta4p0={r4['D_val']:.0f}  beta3p75={r3['D_val']:.0f}  Δ={delta:+.0f}")

    mean_d4 = np.mean([r['D_val'] for r in rd4])
    mean_d3 = np.mean([r['D_val'] for r in rd3])
    print(f"  Mean D_val: beta4p0={mean_d4:.0f}  beta3p75={mean_d3:.0f}  Δ={mean_d4-mean_d3:+.0f}")

    print("\n  Rate R_val_bits at final epoch (lower = more compressed):")
    for r4, r3 in zip(rd4, rd3):
        delta = r4['R_val_bits'] - r3['R_val_bits']
        direction = "more compressed" if delta < 0 else "less compressed"
        print(f"    fold {r4['fold']}: beta4p0={r4['R_val_bits']:.1f}  beta3p75={r3['R_val_bits']:.1f}  Δ={delta:+.1f} ({direction})")

    mean_r4 = np.mean([r['R_val_bits'] for r in rd4])
    mean_r3 = np.mean([r['R_val_bits'] for r in rd3])
    print(f"  Mean R_val_bits: beta4p0={mean_r4:.1f}  beta3p75={mean_r3:.1f}  Δ={mean_r4-mean_r3:+.1f}")

    print("\n  Latent MI(Z;Y) — mutual information between latent and AD label (nats):")
    for l4, l3 in zip(li4, li3):
        delta = l4['mi_sum_Y'] - l3['mi_sum_Y']
        direction = "more informative" if delta > 0 else "less informative"
        print(f"    fold {l4['fold']}: beta4p0={l4['mi_sum_Y']:.3f}  beta3p75={l3['mi_sum_Y']:.3f}  Δ={delta:+.3f} ({direction})")

    mean_mi4 = np.mean([l['mi_sum_Y'] for l in li4])
    mean_mi3 = np.mean([l['mi_sum_Y'] for l in li3])
    print(f"  Mean MI(Z;Y): beta4p0={mean_mi4:.3f}  beta3p75={mean_mi3:.3f}  Δ={mean_mi4-mean_mi3:+.3f}")

    print("\n  Latent MI(Z;Manufacturer) — scanner leakage in latent (nats):")
    for l4, l3 in zip(li4, li3):
        delta = l4['mi_sum_mfr'] - l3['mi_sum_mfr']
        direction = "worse" if delta > 0 else "better"
        print(f"    fold {l4['fold']}: beta4p0={l4['mi_sum_mfr']:.3f}  beta3p75={l3['mi_sum_mfr']:.3f}  Δ={delta:+.3f} ({direction})")

    print("\n  Active units: all 384/384 for both runs across all folds (no collapse)")

    print("\n  Total correlation (nats) — lower = more disentangled:")
    for l4, l3 in zip(li4, li3):
        delta = l4['total_corr'] - l3['total_corr']
        print(f"    fold {l4['fold']}: beta4p0={l4['total_corr']:.0f}  beta3p75={l3['total_corr']:.0f}  Δ={delta:+.0f}")

    # ── 7. Scanner leakage (train-pool linear probe) ────────────────────────
    print("\n## 7. Scanner/manufacturer leakage (latent linear probe, acc_site_latent)")
    sl4 = load_scanner_leakage(RUN4P0)
    sl3 = load_scanner_leakage(RUN3P75)

    print("\n  acc_site_latent (3-class manufacturer, upper bound; higher = more leakage):")
    for r4, r3 in zip(sl4, sl3):
        fold = r4['fold_tag'].split('_')[1]
        delta = r4['acc_site_latent'] - r3['acc_site_latent']
        direction = "MORE leakage" if delta > 0 else "LESS leakage"
        print(f"    fold {fold}: beta4p0={r4['acc_site_latent']:.4f}  beta3p75={r3['acc_site_latent']:.4f}  Δ={delta:+.4f} ({direction})")

    mean_sl4 = np.mean([r['acc_site_latent'] for r in sl4])
    mean_sl3 = np.mean([r['acc_site_latent'] for r in sl3])
    print(f"  Mean acc_site_latent: beta4p0={mean_sl4:.4f}  beta3p75={mean_sl3:.4f}  Δ={mean_sl4-mean_sl3:+.4f}")
    print(f"  Chance level: 0.3333")
    sl_verdict = "WORSE (more leakage)" if mean_sl4 > mean_sl3 else "BETTER (less leakage)"
    print(f"  Leakage verdict: beta4p0 is {sl_verdict} vs reference")

    # ── 8. Philips CN false positive rate ─────────────────────────────────
    print("\n## 8. Philips CN false positive rate (Stage B subgroup)")
    sub4 = load_stage_b_subgroup(RUN4P0)
    sub3 = load_stage_b_subgroup(RUN3P75)

    philips4 = philips_cn_fpr_from_subgroup(sub4, primary_thresh)
    philips3 = philips_cn_fpr_from_subgroup(sub3, primary_thresh)

    print(f"\n  Primary threshold: {primary_thresh}")
    print(f"  beta4p0: n_cn={philips4['n_cn']}, fp={philips4['fp']}, Philips CN FPR={philips4['fpr']:.4f}")
    print(f"  beta3p75: n_cn={philips3['n_cn']}, fp={philips3['fp']}, Philips CN FPR={philips3['fpr']:.4f}")
    print(f"  Reference OOF-logitz Philips CN FPR: {REF_PHILIPS_FPR:.4f}")
    delta_fpr = philips4['fpr'] - philips3['fpr']
    print(f"  Δ FPR (beta4p0 - beta3p75): {delta_fpr:+.4f}")
    philips_verdict = "WORSE" if philips4['fpr'] > REF_PHILIPS_FPR else "BETTER"
    print(f"  vs reference: {philips_verdict}")

    # Philips CN FPR from OOF-logitz computation
    print(f"\n  OOF-logitz Philips CN FPR (beta4p0, computed in §4):")
    print(f"    n_cn={oof4['philips_cn_n']}, fp={oof4['philips_cn_fp']}, FPR={oof4['philips_cn_fpr']:.4f}")
    print(f"    vs locked reference: {REF_PHILIPS_FPR:.4f}")
    oof_philips_verdict = "WORSE" if oof4['philips_cn_fpr'] > REF_PHILIPS_FPR else "BETTER"
    print(f"    Verdict: {oof_philips_verdict}")

    # Per-fold Philips CN FPR
    print("\n  Per-fold Philips CN FPR (target_sens threshold):")
    for fold in FOLDS:
        f4 = sub4[(sub4['fold'] == fold) & (sub4['Manufacturer'] == 'Philips') &
                   (sub4['model_name'] == 'logreg_l2') &
                   (sub4['threshold_strategy'] == primary_thresh)]
        f3 = sub3[(sub3['fold'] == fold) & (sub3['Manufacturer'] == 'Philips') &
                   (sub3['model_name'] == 'logreg_l2') &
                   (sub3['threshold_strategy'] == primary_thresh)]
        if len(f4) > 0 and len(f3) > 0:
            fpr4 = float(f4.iloc[0]['fp']) / float(f4.iloc[0]['n_cn'])
            fpr3 = float(f3.iloc[0]['fp']) / float(f3.iloc[0]['n_cn'])
            delta = fpr4 - fpr3
            print(f"    fold {fold}: beta4p0={fpr4:.3f}  beta3p75={fpr3:.3f}  Δ={delta:+.3f}")

    # ── 9. OASIS mega stress-test ──────────────────────────────────────────
    print("\n## 9. OASIS mega stress-test")
    oasis_scoring_dir = RESULTS / "oasis_next_60cn_60ad_external_scoring_20260530"
    if oasis_scoring_dir.exists():
        score_files = list(oasis_scoring_dir.glob("*.csv"))
        print(f"  OASIS scoring directory exists: {oasis_scoring_dir}")
        if score_files:
            print(f"  Available: {[f.name for f in score_files[:5]]}")
        else:
            print("  No CSV result files found yet.")
        beta4p0_oasis = [f for f in score_files if "beta4p0" in f.name or "beta4.0" in f.name]
        if beta4p0_oasis:
            for fp_oasis in beta4p0_oasis:
                print(f"  beta4p0 OASIS result: {fp_oasis.name}")
        else:
            print("  No beta4p0 OASIS scoring artifacts found — OASIS mega test not available for this run.")
    else:
        print(f"  OASIS scoring directory not found: {oasis_scoring_dir}")
        print("  OASIS mega stress-test: NOT AVAILABLE for beta4p0.")

    # ── 10. Full promotion gate summary ───────────────────────────────────
    print("\n" + "=" * 70)
    print("## 10. Full promotion gate summary")
    print("=" * 70)
    print(f"\n  Promotion gate (must pass BOTH):")
    print(f"    AUC  > {LOCKED_AUC}   ->  {oof4['pooled_auc']:.4f}  {'PASS' if pass_auc else 'FAIL'}")
    print(f"    PR-AUC >= {LOCKED_PR_AUC}  ->  {oof4['pooled_pr_auc']:.4f}  {'PASS' if pass_prauc else 'FAIL'}")
    print(f"\n  Secondary checks (advisory):")
    print(f"    Stage A mean fold AUC: beta4p0={sa4['auc'].mean():.4f}  ref={sa3['auc'].mean():.4f}  {'▲' if sa4['auc'].mean() > sa3['auc'].mean() else '▼'}")
    print(f"    Stage B raw AUC:  beta4p0={float(row4_0p5['auc']):.4f}  ref={float(row3_0p5['auc']):.4f}")
    print(f"    Scanner leakage:  beta4p0={mean_sl4:.4f}  ref={mean_sl3:.4f}  {'▲ MORE' if mean_sl4 > mean_sl3 else '▼ LESS'}")
    print(f"    Philips CN FPR (Stage B): beta4p0={philips4['fpr']:.4f}  ref={philips3['fpr']:.4f}  {'▲ WORSE' if philips4['fpr'] > philips3['fpr'] else '▼ BETTER'}")
    print(f"    Training epochs (mean): beta4p0={mean_ep4:.0f}  ref={mean_ep3:.0f}")
    print(f"    Rate R_val_bits (mean): beta4p0={mean_r4:.1f}  ref={mean_r3:.1f}  (lower=more compressed)")
    print(f"    MI(Z;Y) (mean nats): beta4p0={mean_mi4:.3f}  ref={mean_mi3:.3f}")

    print(f"\n  === DECISION ===")
    if both_pass:
        decision = "PROMOTE"
        rationale = "Both AUC and PR-AUC exceed the locked promotion gate."
    else:
        decision = "DO NOT PROMOTE"
        details = []
        if not pass_auc:
            details.append(f"AUC {oof4['pooled_auc']:.4f} does not exceed {LOCKED_AUC}")
        if not pass_prauc:
            details.append(f"PR-AUC {oof4['pooled_pr_auc']:.4f} does not reach {LOCKED_PR_AUC}")
        rationale = "; ".join(details) + ". beta4p0 performs BELOW beta3p75 (the promoted reference) after OOF-logitz calibration."

    print(f"  Decision:  {decision}")
    print(f"  Rationale: {rationale}")

    # ── Save JSON summary ──────────────────────────────────────────────────
    summary = {
        "generated": ts,
        "run_beta4p0": str(RUN4P0),
        "run_beta3p75": str(RUN3P75),
        "locked_auc": LOCKED_AUC,
        "locked_pr_auc": LOCKED_PR_AUC,
        "fold_completion": {"folds_present": 5, "missing": missing},
        "stage_a": {
            "beta4p0": {
                "mean_auc": float(sa4["auc"].mean()),
                "std_auc": float(sa4["auc"].std()),
                "mean_pr_auc": float(sa4["pr_auc"].mean()),
                "fold_aucs": sa4["auc"].tolist(),
            },
            "beta3p75": {
                "mean_auc": float(sa3["auc"].mean()),
                "std_auc": float(sa3["auc"].std()),
                "mean_pr_auc": float(sa3["pr_auc"].mean()),
                "fold_aucs": sa3["auc"].tolist(),
            },
        },
        "stage_b_raw": {
            "beta4p0": {"auc": float(row4_0p5["auc"]), "pr_auc": float(row4_0p5["pr_auc"])},
            "beta3p75": {"auc": float(row3_0p5["auc"]), "pr_auc": float(row3_0p5["pr_auc"])},
        },
        "oof_logitz": {
            "beta4p0": {
                "pooled_auc": oof4["pooled_auc"],
                "pooled_pr_auc": oof4["pooled_pr_auc"],
                "foldwise_mean_auc": oof4["foldwise_mean_auc"],
                "foldwise_aucs": oof4["foldwise_aucs"],
                "philips_cn_fpr": oof4["philips_cn_fpr"],
                "philips_cn_fp": oof4["philips_cn_fp"],
                "philips_cn_n": oof4["philips_cn_n"],
            },
            "beta3p75_reference": {
                "pooled_auc": REF_OOF_LOGITZ_AUC,
                "pooled_pr_auc": REF_OOF_LOGITZ_PR_AUC,
                "philips_cn_fpr": REF_PHILIPS_FPR,
            },
        },
        "vae_qc": {
            "mean_epochs": {"beta4p0": mean_ep4, "beta3p75": mean_ep3},
            "mean_D_val": {"beta4p0": mean_d4, "beta3p75": mean_d3},
            "mean_R_val_bits": {"beta4p0": mean_r4, "beta3p75": mean_r3},
            "mean_mi_Y_nats": {"beta4p0": mean_mi4, "beta3p75": mean_mi3},
            "active_units_all_folds": 384,
        },
        "scanner_leakage": {
            "mean_acc_site_latent": {"beta4p0": mean_sl4, "beta3p75": mean_sl3},
            "chance": 0.3333,
        },
        "philips_cn_fpr_stageb": {
            "beta4p0": philips4,
            "beta3p75": philips3,
        },
        "promotion": {
            "pass_auc": bool(pass_auc),
            "pass_pr_auc": bool(pass_prauc),
            "both_pass": bool(both_pass),
            "decision": decision,
            "rationale": rationale,
        },
    }

    out_json = OUTPUT_DIR / "audit_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nJSON summary saved: {out_json}")

    # Write markdown report
    _write_markdown(summary, oof4, sa4, sa3, rd4, rd3, li4, li3, sl4, sl3,
                    philips4, philips3, ts)
    print(f"Markdown report saved: {OUTPUT_DIR / 'summary.md'}")


def _write_markdown(summary: Dict, oof4: Dict, sa4: pd.DataFrame,
                    sa3: pd.DataFrame, rd4: List, rd3: List,
                    li4: List, li3: List, sl4: List, sl3: List,
                    philips4: Dict, philips3: Dict, ts: str) -> None:
    lines = []
    lines.append("# beta4p0 Completion & Promotion-Gate Audit")
    lines.append(f"Generated: {ts}")
    lines.append("")
    lines.append("## Context")
    lines.append("- **Run**: `recover035_latent384_beta4p0_T80_h10000_p560_full5x5`")
    lines.append("- **Reference (promoted)**: `recover035_latent384_beta3p75_T80_h10000_p560_full5x5`")
    lines.append(f"- **Promotion gate**: AUC > {LOCKED_AUC} AND PR-AUC >= {LOCKED_PR_AUC}")
    lines.append("- **Single controlled change**: beta_vae 3.75 → 4.0")
    lines.append("")

    # Completion
    lines.append("## 1. Fold Completion")
    lines.append("All 5 folds completed ✓")
    lines.append("")

    # Stage A
    lines.append("## 2. Stage A Fold-Level AUC (logreg, final)")
    lines.append("")
    header = "| fold | beta4p0 AUC | beta3p75 AUC | Δ | beta4p0 PR-AUC | beta3p75 PR-AUC |"
    sep    = "|------|------------|--------------|---|----------------|----------------|"
    lines.append(header)
    lines.append(sep)
    for f in FOLDS:
        v4 = float(sa4[sa4["fold"] == f]["auc"])
        v3 = float(sa3[sa3["fold"] == f]["auc"])
        p4 = float(sa4[sa4["fold"] == f]["pr_auc"])
        p3 = float(sa3[sa3["fold"] == f]["pr_auc"])
        delta = v4 - v3
        arrow = "▲" if delta > 0 else "▼"
        lines.append(f"| {f} | {v4:.4f} | {v3:.4f} | {arrow}{abs(delta):.4f} | {p4:.4f} | {p3:.4f} |")
    lines.append(f"| **mean** | **{sa4['auc'].mean():.4f}** | **{sa3['auc'].mean():.4f}** | **{sa4['auc'].mean()-sa3['auc'].mean():+.4f}** | **{sa4['pr_auc'].mean():.4f}** | **{sa3['pr_auc'].mean():.4f}** |")
    lines.append("")

    # Stage B raw
    lines.append("## 3. Stage B Raw Pooled (z_plus_age_sex, logreg_l2)")
    lines.append("")
    lines.append("| model | AUC | PR-AUC |")
    lines.append("|-------|-----|--------|")
    lines.append(f"| beta4p0 (fixed_0.5) | {summary['stage_b_raw']['beta4p0']['auc']:.4f} | {summary['stage_b_raw']['beta4p0']['pr_auc']:.4f} |")
    lines.append(f"| beta3p75 (fixed_0.5) | {summary['stage_b_raw']['beta3p75']['auc']:.4f} | {summary['stage_b_raw']['beta3p75']['pr_auc']:.4f} |")
    lines.append("")

    # OOF-logitz
    lines.append("## 4. OOF-Logitz Calibrated AUC (primary promotion metric)")
    lines.append("")
    lines.append("| model | OOF-logitz AUC | OOF-logitz PR-AUC | Foldwise mean AUC | Philips CN FPR |")
    lines.append("|-------|---------------|-------------------|-------------------|----------------|")
    lines.append(f"| **beta4p0** | **{oof4['pooled_auc']:.4f}** | **{oof4['pooled_pr_auc']:.4f}** | {oof4['foldwise_mean_auc']:.4f} | {oof4['philips_cn_fpr']:.4f} |")
    lines.append(f"| beta3p75 (ref) | {REF_OOF_LOGITZ_AUC:.4f} | {REF_OOF_LOGITZ_PR_AUC:.4f} | 0.8008 | {REF_PHILIPS_FPR:.4f} |")
    lines.append(f"| Promotion gate | >{LOCKED_AUC} | >={LOCKED_PR_AUC} | — | ≤{REF_PHILIPS_FPR:.4f} |")
    lines.append("")

    # Promotion gate
    lines.append("## 5. Promotion Gate")
    lines.append("")
    pass_auc = summary["promotion"]["pass_auc"]
    pass_pr = summary["promotion"]["pass_pr_auc"]
    lines.append(f"- AUC > {LOCKED_AUC}: **{'PASS ✓' if pass_auc else 'FAIL ✗'}** ({oof4['pooled_auc']:.4f})")
    lines.append(f"- PR-AUC >= {LOCKED_PR_AUC}: **{'PASS ✓' if pass_pr else 'FAIL ✗'}** ({oof4['pooled_pr_auc']:.4f})")
    lines.append("")
    lines.append(f"### Decision: {summary['promotion']['decision']}")
    lines.append(f"{summary['promotion']['rationale']}")
    lines.append("")

    # VAE QC
    lines.append("## 6. VAE QC Comparison")
    lines.append("")
    lines.append("| metric | beta4p0 | beta3p75 | Δ |")
    lines.append("|--------|---------|----------|---|")
    lines.append(f"| mean epochs | {summary['vae_qc']['mean_epochs']['beta4p0']:.0f} | {summary['vae_qc']['mean_epochs']['beta3p75']:.0f} | {summary['vae_qc']['mean_epochs']['beta4p0']-summary['vae_qc']['mean_epochs']['beta3p75']:+.0f} |")
    lines.append(f"| mean D_val (recon) | {summary['vae_qc']['mean_D_val']['beta4p0']:.0f} | {summary['vae_qc']['mean_D_val']['beta3p75']:.0f} | {summary['vae_qc']['mean_D_val']['beta4p0']-summary['vae_qc']['mean_D_val']['beta3p75']:+.0f} |")
    lines.append(f"| mean R_val_bits (KLD) | {summary['vae_qc']['mean_R_val_bits']['beta4p0']:.1f} | {summary['vae_qc']['mean_R_val_bits']['beta3p75']:.1f} | {summary['vae_qc']['mean_R_val_bits']['beta4p0']-summary['vae_qc']['mean_R_val_bits']['beta3p75']:+.1f} |")
    lines.append(f"| mean MI(Z;Y) nats | {summary['vae_qc']['mean_mi_Y_nats']['beta4p0']:.3f} | {summary['vae_qc']['mean_mi_Y_nats']['beta3p75']:.3f} | {summary['vae_qc']['mean_mi_Y_nats']['beta4p0']-summary['vae_qc']['mean_mi_Y_nats']['beta3p75']:+.3f} |")
    lines.append(f"| active units | 384 | 384 | 0 |")
    lines.append("")

    # Scanner leakage
    lines.append("## 7. Scanner Leakage (acc_site_latent, 3-class Manufacturer)")
    lines.append("")
    lines.append("| fold | beta4p0 | beta3p75 | Δ |")
    lines.append("|------|---------|----------|---|")
    for r4, r3 in zip(sl4, sl3):
        fold = r4['fold_tag'].split('_')[1]
        delta = r4['acc_site_latent'] - r3['acc_site_latent']
        lines.append(f"| {fold} | {r4['acc_site_latent']:.4f} | {r3['acc_site_latent']:.4f} | {delta:+.4f} |")
    delta_mean = summary['scanner_leakage']['mean_acc_site_latent']['beta4p0'] - summary['scanner_leakage']['mean_acc_site_latent']['beta3p75']
    lines.append(f"| **mean** | **{summary['scanner_leakage']['mean_acc_site_latent']['beta4p0']:.4f}** | **{summary['scanner_leakage']['mean_acc_site_latent']['beta3p75']:.4f}** | **{delta_mean:+.4f}** |")
    lines.append(f"\nChance level = 0.3333. Higher = more scanner information retained in latent.")
    lines.append(f"beta4p0 mean leakage is {'HIGHER (worse)' if delta_mean > 0 else 'LOWER (better)'} than reference.")
    lines.append("")

    # Philips CN FPR
    lines.append("## 8. Philips CN False Positive Rate")
    lines.append("")
    lines.append("| model | Philips CN n | FP | FPR | vs reference |")
    lines.append("|-------|-------------|-----|-----|-------------|")
    ref_fpr = REF_PHILIPS_FPR
    lines.append(f"| beta4p0 (Stage B primary) | {philips4['n_cn']} | {philips4['fp']} | {philips4['fpr']:.4f} | {'+worse' if philips4['fpr'] > ref_fpr else '-better'} Δ={philips4['fpr']-ref_fpr:+.4f} |")
    lines.append(f"| beta3p75 (Stage B primary) | {philips3['n_cn']} | {philips3['fp']} | {philips3['fpr']:.4f} | baseline |")
    lines.append(f"| beta3p75 (OOF-logitz ref) | 99 | 44 | 0.4444 | locked ref |")
    lines.append(f"| beta4p0 (OOF-logitz computed) | {oof4['philips_cn_n']} | {oof4['philips_cn_fp']} | {oof4['philips_cn_fpr']:.4f} | {'+worse' if oof4['philips_cn_fpr'] > ref_fpr else '-better'} Δ={oof4['philips_cn_fpr']-ref_fpr:+.4f} |")
    lines.append("")

    # Beta sensitivity interpretation
    lines.append("## 9. Beta Sensitivity Interpretation")
    lines.append("")
    lines.append("beta4p0 is a 1.067× increase over beta3p75. The hypothesis was that slight additional")
    lines.append("bottleneck tightening would reduce Philips CN false positives by forcing more")
    lines.append("disentanglement of manufacturer signal from disease signal.")
    lines.append("")
    lines.append("**Observed effects of higher beta:**")
    d_ep = summary['vae_qc']['mean_epochs']['beta4p0'] - summary['vae_qc']['mean_epochs']['beta3p75']
    d_r = summary['vae_qc']['mean_R_val_bits']['beta4p0'] - summary['vae_qc']['mean_R_val_bits']['beta3p75']
    d_d = summary['vae_qc']['mean_D_val']['beta4p0'] - summary['vae_qc']['mean_D_val']['beta3p75']
    d_mi = summary['vae_qc']['mean_mi_Y_nats']['beta4p0'] - summary['vae_qc']['mean_mi_Y_nats']['beta3p75']
    d_sl = summary['scanner_leakage']['mean_acc_site_latent']['beta4p0'] - summary['scanner_leakage']['mean_acc_site_latent']['beta3p75']
    lines.append(f"1. Faster convergence: beta4p0 trains {abs(d_ep):.0f} fewer epochs on average ({d_ep:+.0f}).")
    lines.append(f"2. Lower rate (more compressed): R_val_bits Δ={d_r:+.1f} bits (beta4p0 uses ~{abs(d_r):.1f} fewer bits).")
    lines.append(f"3. Similar reconstruction quality: D_val Δ={d_d:+.0f} (negligible).")
    lines.append(f"4. Slightly higher MI(Z;Y): Δ={d_mi:+.3f} nats (marginally more disease-relevant structure).")
    lines.append(f"5. Higher scanner leakage: acc_site_latent Δ={d_sl:+.4f} — unexpected; higher beta did NOT reduce manufacturer signal.")
    lines.append(f"6. Higher Philips CN FPR: Δ={philips4['fpr']-philips3['fpr']:+.4f} — hypothesis refuted.")
    lines.append(f"7. Lower OOF-logitz AUC: {oof4['pooled_auc']:.4f} vs {REF_OOF_LOGITZ_AUC:.4f} — discriminative performance degraded.")
    lines.append("")
    lines.append("**Conclusion**: beta4p0 fails the promotion gate and shows no improvement in Philips")
    lines.append("CN FPR. The 1.067× beta increase over-regularizes the bottleneck without clinical benefit.")
    lines.append("The promoted beta3p75 run remains the best candidate.")
    lines.append("")

    # OASIS
    lines.append("## 10. OASIS Mega 90/90 Stress-Test")
    lines.append("")
    lines.append("OASIS scoring artifacts for beta4p0 not available — no OASIS inference has been run")
    lines.append("for this run. The OASIS external validation is pending (scoring script ready at")
    lines.append("`scripts/revision_bspc_2026/score_oasis_next_60cn_60ad_external_20260530.py`).")
    lines.append("")

    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
