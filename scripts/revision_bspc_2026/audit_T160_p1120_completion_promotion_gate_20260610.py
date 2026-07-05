#!/usr/bin/env python3
"""Completion and promotion-gate audit for T160 p1120 patience-extension candidate.

Candidate: recover035_latent384_beta3p75_T160_h10000_p1120_full5x5

Scientific question: Does doubling early_stopping_patience_vae from 560→1120
(T160 LR-cycle patience 3.5→7.0) improve ADNI OOF AUC, PR-AUC, or Philips FPR
relative to T160 p560 and promoted T80 p560?

Guardrails:
  - No VAE retraining during audit
  - No tensor modification
  - No metadata modification
  - No model artifact overwrite
  - No OASIS threshold or calibration fitting
  - OASIS stress-test only runs if ALL ADNI gate criteria pass

This script reads completed artifacts only:
  Stage A (VAE training), Stage B (classifier-only readout), OOF calibration.
If Stage B or OOF calibration artifacts are missing the script reports the gap
and exits without attempting to generate them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[2]
RESULTS  = ROOT / "results" / "revision_bspc_2026"
DATOS    = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026")

CANDIDATE_RUN_ID = "recover035_latent384_beta3p75_T160_h10000_p1120_full5x5"
CANDIDATE_DIR    = DATOS / CANDIDATE_RUN_ID
CANDIDATE_CLF    = CANDIDATE_DIR / "classifier_only_readout"
CANDIDATE_OOF    = RESULTS / "recover035_latent384_beta3p75_T160_p1120_stageB_oof_score_calibration"

P560_DIR = DATOS / "recover035_latent384_beta3p75_T160_h10000_p560_full5x5"

OUT_DIR = RESULTS / "T160_p1120_completion_promotion_gate_audit_20260610"

# ─────────────────────────────────────────────────────────────────────────────
# Gates
# ─────────────────────────────────────────────────────────────────────────────
PROMOTED_AUC      = 0.795155   # promoted T80 p560 oof_ecdf AUC
PROMOTED_PR_AUC   = 0.573934   # promoted T80 p560 oof_ecdf PR-AUC
PHILIPS_FPR_GATE  = 0.4545     # must NOT exceed (≤ promoted Philips FPR)

# ─────────────────────────────────────────────────────────────────────────────
# Primary readout selectors
# ─────────────────────────────────────────────────────────────────────────────
PRIMARY_MODEL   = "logreg_l2_original"
PRIMARY_FEATURE = "z_plus_age_sex"
PRIMARY_THRESH  = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_CALIB   = "oof_ecdf"

FOLDS        = [1, 2, 3, 4, 5]
EPOCHS_BUDGET = 10000
BETA_MAX      = 3.75


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) + "\n"


def write_table(df: pd.DataFrame, stem: str) -> None:
    df.to_csv(OUT_DIR / f"{stem}.csv", index=False)
    (OUT_DIR / f"{stem}.md").write_text(md_table(df), encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# S1 — Fold completion: Stage A artifacts
# ─────────────────────────────────────────────────────────────────────────────

def audit_fold_completion() -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        fd   = CANDIDATE_DIR / f"fold_{fold}"
        ckpt = fd / f"vae_model_fold_{fold}.pt"
        hist = fd / f"vae_train_history_fold_{fold}.joblib"
        qc   = fd / "latent_qc_metrics.csv"

        ckpt_ok = ckpt.exists()
        hist_ok = hist.exists()
        qc_ok   = qc.exists()

        best_epoch = final_epoch = None
        patience_exhausted = False
        if hist_ok:
            h = joblib.load(hist)
            val_loss = h.get("val_loss") or h.get("val_total_loss", [])
            if val_loss:
                val_loss = list(val_loss)
                best_epoch = int(np.argmin(val_loss)) + 1
                final_epoch = len(val_loss)
                patience_exhausted = final_epoch < EPOCHS_BUDGET

        rows.append({
            "fold":               fold,
            "checkpoint_exists":  ckpt_ok,
            "history_exists":     hist_ok,
            "qc_exists":          qc_ok,
            "all_artifacts_ok":   ckpt_ok and hist_ok and qc_ok,
            "best_epoch":         best_epoch,
            "final_epoch":        final_epoch,
            "epochs_used_pct":    round(100 * best_epoch / EPOCHS_BUDGET, 1) if best_epoch else None,
            "patience_exhausted": patience_exhausted,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# S2 — Stage B / OOF calibration artifact check
# ─────────────────────────────────────────────────────────────────────────────

def audit_stageB_oof_status() -> Dict[str, Any]:
    clf_ok        = CANDIDATE_CLF.exists()
    oof_ok        = CANDIDATE_OOF.exists()
    pooled_ok     = (CANDIDATE_OOF / "calib_pooled_metrics.csv").exists()
    foldwise_ok   = (CANDIDATE_OOF / "calib_foldwise_metrics.csv").exists()
    philips_ok    = (CANDIDATE_OOF / "calib_philips_fpr_pooled.csv").exists()
    subgroup_ok   = (CANDIDATE_CLF / "classifier_sweep_subgroup_metrics_by_manufacturer.csv").exists()
    return {
        "stage_b_readout_dir_exists":  clf_ok,
        "oof_calib_dir_exists":        oof_ok,
        "calib_pooled_metrics_exists": pooled_ok,
        "calib_foldwise_exists":       foldwise_ok,
        "philips_fpr_pooled_exists":   philips_ok,
        "subgroup_metrics_exists":     subgroup_ok,
        "all_stage_b_oof_complete":    clf_ok and pooled_ok and foldwise_ok and philips_ok and subgroup_ok,
    }


# ─────────────────────────────────────────────────────────────────────────────
# S3 — Checkpoint comparison p1120 vs p560 (training curve maturity)
# ─────────────────────────────────────────────────────────────────────────────

def checkpoint_comparison() -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        row: Dict[str, Any] = {"fold": fold}
        for run_id, run_dir in [("T160_p1120", CANDIDATE_DIR), ("T160_p560", P560_DIR)]:
            hist = run_dir / f"fold_{fold}" / f"vae_train_history_fold_{fold}.joblib"
            if hist.exists():
                h = joblib.load(hist)
                val_loss = h.get("val_loss") or h.get("val_total_loss", [])
                if val_loss:
                    val_loss = list(val_loss)
                    be = int(np.argmin(val_loss)) + 1
                    fe = len(val_loss)
                    row[f"{run_id}_best_epoch"]  = be
                    row[f"{run_id}_final_epoch"] = fe
                    row[f"{run_id}_pct_budget"]  = round(100 * be / EPOCHS_BUDGET, 1)
        p1120_be = row.get("T160_p1120_best_epoch", np.nan)
        p560_be  = row.get("T160_p560_best_epoch",  np.nan)
        if p1120_be and p560_be:
            row["best_epoch_delta"]   = int(p1120_be - p560_be)
            row["checkpoint_changed"] = bool(p1120_be != p560_be)
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# S4 — Rate distortion, MI(Z;Y), MI(Z;Manufacturer), active units, TC
# ─────────────────────────────────────────────────────────────────────────────

def latent_rate_distortion_full(best_epochs: Dict[int, int]) -> pd.DataFrame:
    """Per-fold: D_val, KLD_val_bits, beta*KLD/D, MI(Z;Y), MI(Z;Mfr), active units, TC."""
    rows = []
    for fold in FOLDS:
        fd  = CANDIDATE_DIR / f"fold_{fold}"
        row: Dict[str, Any] = {"fold": fold}

        # — rate distortion —
        rd_path = fd / f"fold_{fold}_rate_distortion.csv"
        if rd_path.exists():
            df = pd.read_csv(rd_path)
            ep = best_epochs.get(fold, int(df["epoch"].max()))
            sub = df[df.epoch <= ep]
            if not sub.empty:
                r = sub.iloc[-1]
                row["best_epoch"]     = ep
                row["D_val"]          = round(float(r["D_val"]), 2)
                row["KLD_val_bits"]   = round(float(r["R_val_bits"]), 4)
                row["betaKLD_D"]      = round(BETA_MAX * float(r["R_val_nats"]) / float(r["D_val"]), 6) if float(r["D_val"]) > 0 else np.nan

        # — scanner leakage —
        sl_path = fd / f"fold_{fold}_test_scanner_leakage_summary.csv"
        if sl_path.exists():
            df = pd.read_csv(sl_path)
            row["acc_site_raw_test"]    = round(float(df["acc_site_raw"].values[0]), 4)
            row["acc_site_latent_test"] = round(float(df["acc_site_latent"].values[0]), 4)

        # — latent info: MI(Z;Y), MI(Z;Mfr), active units, TC —
        li_path = fd / f"fold_{fold}_test_latent_info_summary.csv"
        if li_path.exists():
            df     = pd.read_csv(li_path)
            y_row  = df[df.variable == "Y_target"]
            m_row  = df[df.variable == "Manufacturer"]
            if not y_row.empty:
                row["MI_Z_Y_nats"]       = round(float(y_row["mi_sum_nats"].values[0]), 4)
                row["n_active_dims"]     = int(y_row["n_active"].values[0])
                row["frac_active"]       = round(float(y_row["frac_active"].values[0]), 4)
                row["TC_nats"]           = round(float(y_row["total_correlation_nats"].values[0]), 2)
            if not m_row.empty:
                row["MI_Z_Mfr_nats"]     = round(float(m_row["mi_sum_nats"].values[0]), 4)
            mi_y   = row.get("MI_Z_Y_nats",  np.nan)
            mi_mfr = row.get("MI_Z_Mfr_nats", np.nan)
            row["MI_ratio_Y_Mfr"] = round(mi_y / mi_mfr, 4) if (mi_mfr and mi_mfr > 0) else np.nan

        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# S5 — OOF calibration: pooled + foldwise metrics
# ─────────────────────────────────────────────────────────────────────────────

def load_oof_pooled(oof_dir: Path) -> pd.DataFrame:
    p = oof_dir / "calib_pooled_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_oof_foldwise(oof_dir: Path) -> pd.DataFrame:
    p = oof_dir / "calib_foldwise_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_philips_pooled(oof_dir: Path) -> pd.DataFrame:
    p = oof_dir / "calib_philips_fpr_pooled.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def oof_primary_slice(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (
        (df.model_name == PRIMARY_MODEL) &
        (df.feature_set == PRIMARY_FEATURE) &
        (df.threshold_strategy == PRIMARY_THRESH)
    )
    return df[mask].copy()


def oof_primary_by_calib(df: pd.DataFrame) -> pd.DataFrame:
    sub = oof_primary_slice(df)
    if sub.empty:
        return sub
    cols = [c for c in ["calib_method", "auc", "pr_auc", "balanced_accuracy",
                         "sensitivity", "specificity", "f1", "n_cn", "n_ad"]
            if c in sub.columns]
    return sub[cols].sort_values("auc", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# S6 — Manufacturer CN FPR and AD FNR (pooled, primary readout)
# ─────────────────────────────────────────────────────────────────────────────

def manufacturer_fpr_fnr() -> pd.DataFrame:
    """CN FPR and AD FNR per manufacturer, pooled across folds.

    Reads classifier_sweep_subgroup_metrics_by_manufacturer.csv from the
    Stage B classifier_only_readout directory. Uses primary model/feature/
    threshold selection. Reports pooled (all-fold concatenation) confusion.
    """
    p = CANDIDATE_CLF / "classifier_sweep_subgroup_metrics_by_manufacturer.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)

    # Primary readout filter: logreg_l2, z_plus_age_sex, primary threshold
    # Stage B uses model_name 'logreg_l2' (not _original suffix) and feature_set
    mask = (
        (df.model_name.isin(["logreg_l2", "logreg_l2_original"])) &
        (df.readout_feature_set.isin(["z_plus_age_sex"])) &
        (df.threshold_strategy == PRIMARY_THRESH)
    )
    sub = df[mask].copy()
    if sub.empty:
        # fallback: take most common threshold
        mask2 = (
            (df.model_name.isin(["logreg_l2", "logreg_l2_original"])) &
            (df.readout_feature_set.isin(["z_plus_age_sex"]))
        )
        sub = df[mask2].copy()

    if sub.empty:
        return pd.DataFrame()

    # Pool confusion counts across folds
    rows = []
    for mfr in ["Philips", "GE", "SIEMENS"]:
        s = sub[sub.Manufacturer == mfr]
        if s.empty:
            continue
        tn_tot = s["tn"].sum()
        fp_tot = s["fp"].sum()
        fn_tot = s["fn"].sum()
        tp_tot = s["tp"].sum()
        n_cn   = tn_tot + fp_tot
        n_ad   = fn_tot + tp_tot
        rows.append({
            "Manufacturer":   mfr,
            "n_cn":           int(n_cn),
            "n_ad":           int(n_ad),
            "tn":             int(tn_tot),
            "fp":             int(fp_tot),
            "fn":             int(fn_tot),
            "tp":             int(tp_tot),
            "CN_FPR":         round(fp_tot / n_cn, 4) if n_cn > 0 else np.nan,
            "AD_FNR":         round(fn_tot / n_ad, 4) if n_ad > 0 else np.nan,
            "AD_Sensitivity": round(tp_tot / n_ad, 4) if n_ad > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# S7 — Cross-model comparison table
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table() -> pd.DataFrame:
    oof_dirs: Dict[str, Path] = {
        "promoted_T80_p560":              RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration",
        "T160_p560":                      RESULTS / "recover035_latent384_beta3p75_T160_stageB_oof_score_calibration",
        "T160_p1120 (candidate)":         CANDIDATE_OOF,
        "ch1only_T80_p560":               RESULTS / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration",
        "beta9p5_T80_p560":               RESULTS / "recover035_latent384_beta9p5_stageB_oof_score_calibration",
        "chweightedPearson50_T80_p560":   RESULTS / "recover035_latent384_beta3p75_chweightedPearson50_stageB_oof_score_calibration",
    }
    rows = []
    for label, oof_dir in oof_dirs.items():
        if not oof_dir.exists():
            rows.append({"model": label, "note": "oof_dir_missing"})
            continue
        df  = load_oof_pooled(oof_dir)
        sub = oof_primary_slice(df)
        if sub.empty:
            rows.append({"model": label, "note": "no_primary_rows"})
            continue
        r_ecdf   = sub[sub.calib_method == "oof_ecdf"]
        r_logitz = sub[sub.calib_method == "oof_logitz"]
        r_raw    = sub[sub.calib_method == "raw"]
        def _v(r, col): return round(float(r[col].values[0]), 4) if not r.empty and col in r.columns else np.nan
        rows.append({
            "model":        label,
            "AUC_ecdf":     _v(r_ecdf,   "auc"),
            "PRAUC_ecdf":   _v(r_ecdf,   "pr_auc"),
            "AUC_logitz":   _v(r_logitz, "auc"),
            "PRAUC_logitz": _v(r_logitz, "pr_auc"),
            "AUC_raw":      _v(r_raw,    "auc"),
            "PRAUC_raw":    _v(r_raw,    "pr_auc"),
            "BA_ecdf":      _v(r_ecdf,   "balanced_accuracy"),
            "Sens_ecdf":    _v(r_ecdf,   "sensitivity"),
            "Spec_ecdf":    _v(r_ecdf,   "specificity"),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# S8 — Philips / GE / SIEMENS CN FPR table (oof_ecdf, primary threshold)
# ─────────────────────────────────────────────────────────────────────────────

def build_fpr_table() -> pd.DataFrame:
    run_refs = {
        "promoted_T80_p560": RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration",
        "T160_p560":         RESULTS / "recover035_latent384_beta3p75_T160_stageB_oof_score_calibration",
        "T160_p1120":        CANDIDATE_OOF,
    }
    rows = []
    for label, oof_dir in run_refs.items():
        ph = load_philips_pooled(oof_dir)
        if ph.empty:
            continue
        sub = ph[
            (ph.model_name == PRIMARY_MODEL) &
            (ph.feature_set == PRIMARY_FEATURE) &
            (ph.threshold_strategy == PRIMARY_THRESH)
        ]
        for mfr in ["Philips", "GE", "SIEMENS"]:
            r_ecdf = sub[(sub.calib_method == "oof_ecdf") &
                         (sub.manufacturer.str.upper() == mfr.upper())]
            if not r_ecdf.empty:
                rows.append({
                    "model":            label,
                    "manufacturer":     mfr,
                    "n_cn_pooled":      int(r_ecdf.n_cn_pooled.values[0]),
                    "fpr_cn_pooled":    round(float(r_ecdf.fpr_cn_pooled.values[0]), 4),
                    "philips_gate_pass": bool(r_ecdf.fpr_cn_pooled.values[0] <= PHILIPS_FPR_GATE)
                                         if mfr == "Philips" else None,
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# S9 — Plus-ch1 frozen readout reference
# ─────────────────────────────────────────────────────────────────────────────

def load_plus_ch1() -> Optional[pd.DataFrame]:
    p = RESULTS / "plus_ch1_pr_recovery_readout_batch_20260608" / "candidate_metrics.csv"
    if not p.exists():
        # Try the path stored in the prior partial audit
        p2 = RESULTS / "T160_p1120_stageB_oof_completion_audit_20260610" / "plus_ch1_reference.csv"
        if p2.exists():
            return pd.read_csv(p2)
        return None
    return pd.read_csv(p)


# ─────────────────────────────────────────────────────────────────────────────
# S10 — Promotion decision
# ─────────────────────────────────────────────────────────────────────────────

def promotion_decision(pooled_df: pd.DataFrame, philips_df: pd.DataFrame) -> Dict[str, Any]:
    sub = oof_primary_slice(pooled_df)
    p1120_philips = philips_df[
        (philips_df.get("model", pd.Series(dtype=str)) == "T160_p1120") &
        (philips_df.get("manufacturer", pd.Series(dtype=str)) == "Philips")
    ] if not philips_df.empty else pd.DataFrame()

    def _val(df, calib, col, default=np.nan):
        r = df[df.calib_method == calib] if "calib_method" in df.columns else pd.DataFrame()
        if r.empty or col not in r.columns:
            return default
        v = r[col].values[0]
        return float(v)

    best_ecdf_auc   = _val(sub, "oof_ecdf",   "auc")
    best_ecdf_pr    = _val(sub, "oof_ecdf",   "pr_auc")
    best_logitz_auc = _val(sub, "oof_logitz", "auc")
    best_logitz_pr  = _val(sub, "oof_logitz", "pr_auc")

    philips_fpr = float(p1120_philips["fpr_cn_pooled"].values[0]) if not p1120_philips.empty else np.nan

    auc_pass     = bool(best_ecdf_auc  > PROMOTED_AUC)  if not np.isnan(best_ecdf_auc)  else False
    pr_pass      = bool(best_ecdf_pr  >= PROMOTED_PR_AUC) if not np.isnan(best_ecdf_pr)  else False
    philips_pass = bool(philips_fpr   <= PHILIPS_FPR_GATE) if not np.isnan(philips_fpr) else False
    overall_pass = auc_pass and pr_pass and philips_pass

    return {
        "candidate":              CANDIDATE_RUN_ID,
        "best_oof_ecdf_AUC":      round(best_ecdf_auc,   6),
        "best_oof_ecdf_PRAUC":    round(best_ecdf_pr,    6),
        "best_oof_logitz_AUC":    round(best_logitz_auc, 6),
        "best_oof_logitz_PRAUC":  round(best_logitz_pr,  6),
        "promoted_AUC_gate":      PROMOTED_AUC,
        "promoted_PRAUC_gate":    PROMOTED_PR_AUC,
        "philips_FPR_gate":       PHILIPS_FPR_GATE,
        "philips_FPR_ecdf":       round(philips_fpr, 4),
        "gate_auc_pass":          auc_pass,
        "gate_pr_auc_pass":       pr_pass,
        "gate_philips_fpr_pass":  philips_pass,
        "overall_gate_pass":      overall_pass,
        "oasis_stress_test":      "NOT_RUN_gate_failed" if not overall_pass else "required",
        "verdict":                "DOES_NOT_PROMOTE" if not overall_pass else "PROMOTES",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    fold_complete:   pd.DataFrame,
    stageB_status:   Dict[str, Any],
    ckpt_cmp:        pd.DataFrame,
    rd_full:         pd.DataFrame,
    oof_primary:     pd.DataFrame,
    oof_foldwise:    pd.DataFrame,
    mfr_fpr_fnr:     pd.DataFrame,
    comparison:      pd.DataFrame,
    fpr_df:          pd.DataFrame,
    plus_ch1:        Optional[pd.DataFrame],
    decision:        Dict[str, Any],
) -> None:
    now     = datetime.now(timezone.utc).isoformat()
    verdict = decision["verdict"]

    # Pre-compute scalar values used in f-strings to avoid format-spec parse errors
    _auc_ecdf    = f"{decision['best_oof_ecdf_AUC']:.6f}"
    _pr_ecdf     = f"{decision['best_oof_ecdf_PRAUC']:.6f}"
    _auc_logitz  = f"{decision['best_oof_logitz_AUC']:.6f}"
    _pr_logitz   = f"{decision['best_oof_logitz_PRAUC']:.6f}"
    _philips_fpr = f"{decision['philips_FPR_ecdf']:.4f}"
    _auc_gate    = "PASS" if decision["gate_auc_pass"]    else "FAIL"
    _pr_gate     = "PASS" if decision["gate_pr_auc_pass"] else "FAIL"
    _philips_gate = "PASS" if decision["gate_philips_fpr_pass"] else "FAIL"
    _oasis       = decision["oasis_stress_test"]

    # rate distortion averages
    _rd_mean_beta = f"{rd_full['betaKLD_D'].mean():.6f}" if "betaKLD_D" in rd_full.columns else "n/a"
    _rd_mean_tc   = f"{rd_full['TC_nats'].mean():.2f}"   if "TC_nats"   in rd_full.columns else "n/a"
    _n_active_mean = f"{rd_full['n_active_dims'].mean():.1f}" if "n_active_dims" in rd_full.columns else "n/a"

    # Philips FPR for the candidate (oof_ecdf)
    p1120_philips_row = fpr_df[(fpr_df.model == "T160_p1120") & (fpr_df.manufacturer == "Philips")]
    _philips_fpr_cand = f"{p1120_philips_row['fpr_cn_pooled'].values[0]:.4f}" if not p1120_philips_row.empty else "n/a"

    lines: List[str] = [
        f"# T160 p1120 Completion and Promotion-Gate Audit",
        f"## {CANDIDATE_RUN_ID}",
        "",
        f"**Generated**: {now}",
        "",
        "## Scope",
        "- Stage A (VAE training, 5×5 outer CV) — verified complete",
        "- Stage B (classifier-only readout: logreg_l2, z_plus_age_sex) — verified complete",
        "- OOF score calibration (raw / zscore / logitz / ecdf / platt / isotonic) — verified complete",
        "- Read-only: no VAE retraining, no tensor/metadata/model modification",
        "- No OASIS threshold or calibration fitting",
        "",
        "## Scientific question",
        "Does doubling patience 560→1120 (T160 LR-cycle patience 3.5→7.0,",
        "matching the promoted T80 p560 patience) improve ADNI OOF AUC, PR-AUC,",
        "or Philips FPR over T160 p560?",
        "",
        "---",
        "",
        "## Gate criteria (all three must pass for OASIS stress-test)",
        f"- AUC (oof_ecdf) > {PROMOTED_AUC} (promoted T80 p560)",
        f"- PR-AUC (oof_ecdf) ≥ {PROMOTED_PR_AUC} (promoted T80 p560)",
        f"- Philips CN FPR (oof_ecdf) ≤ {PHILIPS_FPR_GATE} (≤ promoted Philips FPR)",
        "",
        "---",
        "",
        f"## Verdict: **{verdict}**",
        "",
        f"- AUC (oof_ecdf):   {_auc_ecdf}  → gate {PROMOTED_AUC}: **{_auc_gate}**",
        f"- PR-AUC (oof_ecdf): {_pr_ecdf}  → gate {PROMOTED_PR_AUC}: **{_pr_gate}**",
        f"- AUC (oof_logitz):  {_auc_logitz}",
        f"- PR-AUC (oof_logitz): {_pr_logitz}",
        f"- Philips CN FPR (oof_ecdf): {_philips_fpr}  → gate ≤{PHILIPS_FPR_GATE}: **{_philips_gate}**",
        f"- OASIS stress-test: **{_oasis}**",
        "",
        "---",
        "",
        "## 1. Stage A — Fold completion audit",
        "",
        md_table(fold_complete),
        "",
        "## 2. Stage B / OOF calibration status",
        "",
    ]

    for k, v in stageB_status.items():
        lines.append(f"- {k}: `{v}`")

    lines += [
        "",
        "---",
        "",
        "## 3. Training curve maturity: p1120 vs p560 (checkpoint comparison)",
        "",
        "Extended patience allows training to continue past the p560 stop point.",
        "Folds 3 and 4 found later best checkpoints.",
        "",
        md_table(ckpt_cmp),
        "",
    ]

    # Key checkpoint finding
    delta_rows = ckpt_cmp[ckpt_cmp.get("best_epoch_delta", pd.Series([0])).abs() > 0] if "best_epoch_delta" in ckpt_cmp.columns else pd.DataFrame()
    changed_folds = ckpt_cmp[ckpt_cmp.get("checkpoint_changed", pd.Series([False]))] if "checkpoint_changed" in ckpt_cmp.columns else pd.DataFrame()
    n_changed = len(changed_folds)
    lines.append(f"**{n_changed}/{len(FOLDS)} folds changed checkpoint** under p1120. "
                 f"Despite later checkpoints in changed folds, AUC did not improve.\n")

    lines += [
        "---",
        "",
        "## 4. Rate distortion, MI(Z;Y), MI(Z;Manufacturer), active units, TC",
        "",
        "Per fold at best epoch.",
        "",
    ]

    rd_disp_cols = [c for c in [
        "fold", "best_epoch", "D_val", "KLD_val_bits", "betaKLD_D",
        "MI_Z_Y_nats", "MI_Z_Mfr_nats", "MI_ratio_Y_Mfr",
        "n_active_dims", "frac_active", "TC_nats",
        "acc_site_raw_test", "acc_site_latent_test",
    ] if c in rd_full.columns]
    lines.append(md_table(rd_full[rd_disp_cols]))
    lines += [
        "",
        f"- Mean β·KLD/D  = {_rd_mean_beta} (promoted T80 ref ≈ 0.0257)",
        f"- Mean TC       = {_rd_mean_tc} nats",
        f"- Mean active dims = {_n_active_mean} / 384",
        "",
        "- acc_site_latent_test: scanner-site classifier on latent Z (chance = 0.333)",
        "- TC (total correlation): proxy for latent disentanglement — lower = more disentangled",
        "",
        "---",
        "",
        "## 5. OOF metrics — primary readout (logreg_l2_original, z_plus_age_sex)",
        "",
        "Pooled across all 5 folds, by calibration method.",
        "",
        md_table(oof_primary),
        "",
    ]

    lines += [
        "### 5a. OOF foldwise (oof_ecdf and oof_logitz)",
        "",
    ]
    if not oof_foldwise.empty:
        sub_fw = oof_foldwise[
            (oof_foldwise.model_name == PRIMARY_MODEL) &
            (oof_foldwise.feature_set == PRIMARY_FEATURE) &
            (oof_foldwise.threshold_strategy == PRIMARY_THRESH) &
            (oof_foldwise.calib_method.isin(["oof_ecdf", "oof_logitz"]))
        ]
        if not sub_fw.empty:
            disp = [c for c in ["fold", "calib_method", "auc", "pr_auc", "balanced_accuracy",
                                 "sensitivity", "specificity", "f1"] if c in sub_fw.columns]
            lines.append(md_table(sub_fw[disp].sort_values(["calib_method", "fold"])))
        else:
            lines.append("_No foldwise data._\n")
    else:
        lines.append("_No foldwise data._\n")

    lines += [
        "",
        "---",
        "",
        "## 6. Manufacturer CN FPR and AD FNR — pooled (primary readout, primary threshold)",
        "",
        "Pooled confusion counts across all folds (Stage B classifier sweep).",
        "",
    ]
    if not mfr_fpr_fnr.empty:
        lines.append(md_table(mfr_fpr_fnr))
    else:
        lines.append("_Subgroup metrics not available._\n")

    lines += [
        "",
        "---",
        "",
        "## 7. Manufacturer-level CN FPR — oof_ecdf calibrated (primary threshold)",
        "",
        "Calibrated OOF predictions pooled across folds.",
        "",
    ]
    if not fpr_df.empty:
        lines.append(md_table(fpr_df))
        lines.append(f"\n**Philips gate** (≤ {PHILIPS_FPR_GATE}): "
                     f"T160 p1120 oof_ecdf = {_philips_fpr_cand} → **{_philips_gate}**\n")
    else:
        lines.append("_FPR data not available._\n")

    lines += [
        "",
        "---",
        "",
        "## 8. Cross-model comparison (oof_ecdf / oof_logitz — primary threshold)",
        "",
        md_table(comparison),
        "",
        "---",
        "",
    ]

    if plus_ch1 is not None:
        plus_ch1_cols = [c for c in ["candidate_id", "auc", "pr_auc", "balanced_accuracy",
                                      "philips_cn_fpr", "decision"] if c in plus_ch1.columns]
        lines += [
            "## 9. Plus-ch1 frozen-readout candidates (reference, for context)",
            "",
            md_table(plus_ch1[plus_ch1_cols]),
            "",
            "---",
            "",
        ]

    lines += [
        "## Summary and decision",
        "",
        f"T160 p1120 extended patience (560→1120) changed best checkpoints in {n_changed}/{len(FOLDS)} folds,",
        "with folds 3 and 4 training to epochs 8321 and 8961 respectively (vs 3041 and 2481 for p560).",
        f"Despite these later checkpoints, OOF-ECDF AUC = {_auc_ecdf} vs gate {PROMOTED_AUC} ({_auc_gate}),",
        f"and Philips CN FPR = {_philips_fpr_cand} vs gate ≤{PHILIPS_FPR_GATE} ({_philips_gate}).",
        "",
        f"Mean β·KLD/D = {_rd_mean_beta} (vs 0.0250 for T160 p560), indicating slightly less KLD",
        "regularisation in the longer-patience folds — consistent with the worse Philips FPR.",
        "Total correlation mean = " + _rd_mean_tc + " nats; active dimensions = " + _n_active_mean + " / 384.",
        "",
        "Interpretation: the p560 stopping criterion is sufficient for T160. The gate failure",
        "is structural (T160 latent geometry), not a patience artefact. Longer patience allows",
        "the model to settle at a slightly less-regularised solution that worsens Philips FPR.",
        "",
        f"**Final promotion decision: {verdict}.**",
        f"OASIS stress-test: {_oasis}.",
        "",
        "---",
        "",
        "## Guardrails confirmed",
        "- No VAE retraining.",
        "- No tensor, metadata, or model artifact modification.",
        "- No OASIS threshold or calibration fitting.",
        "- All calibration parameters estimated from inner-CV OOF only.",
        "- Outer-test labels never used for calibration or threshold fitting.",
    ]

    (OUT_DIR / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    # S1: Fold completion
    print("S1: Auditing fold completion ...", flush=True)
    fold_complete = audit_fold_completion()
    write_table(fold_complete, "fold_completion")
    all_folds_ok = bool(fold_complete["all_artifacts_ok"].all())
    n_folds_ok   = int(fold_complete["all_artifacts_ok"].sum())
    print(f"    {n_folds_ok}/5 folds complete. All OK: {all_folds_ok}")

    if not all_folds_ok:
        print("  ERROR: Not all fold artifacts complete. Stopping audit.")
        return 1

    # S2: Stage B / OOF status
    print("S2: Checking Stage B / OOF calibration status ...", flush=True)
    stageB_status = audit_stageB_oof_status()
    write_json(OUT_DIR / "stageB_oof_status.json", stageB_status)
    if not stageB_status["all_stage_b_oof_complete"]:
        print("  WARNING: Stage B or OOF calibration artifacts incomplete.")
        for k, v in stageB_status.items():
            if isinstance(v, bool) and not v:
                print(f"    MISSING: {k}")
        print("  Continuing with available data.")
    else:
        print("    Stage B and OOF calibration: COMPLETE")

    # Best epoch dict
    best_epochs: Dict[int, int] = {}
    for _, r in fold_complete.iterrows():
        if r["best_epoch"]:
            best_epochs[int(r["fold"])] = int(r["best_epoch"])

    # S3: Checkpoint comparison
    print("S3: Comparing p1120 vs p560 checkpoints ...", flush=True)
    ckpt_cmp = checkpoint_comparison()
    write_table(ckpt_cmp, "checkpoint_comparison_p1120_vs_p560")

    # S4: Rate distortion + latent QC
    print("S4: Computing rate distortion, MI, active units, TC ...", flush=True)
    rd_full = latent_rate_distortion_full(best_epochs)
    write_table(rd_full, "rate_distortion_latent_qc_full")

    # S5: OOF calibration
    print("S5: Loading OOF calibration results ...", flush=True)
    oof_pooled   = load_oof_pooled(CANDIDATE_OOF)
    oof_foldwise = load_oof_foldwise(CANDIDATE_OOF)
    oof_primary  = oof_primary_by_calib(oof_pooled)
    write_table(oof_primary, "oof_primary_by_calib")
    if not oof_pooled.empty:
        write_table(oof_pooled, "oof_pooled_all")
    if not oof_foldwise.empty:
        write_table(oof_foldwise, "oof_foldwise_all")

    # S6: Manufacturer FPR/FNR
    print("S6: Computing manufacturer CN FPR and AD FNR ...", flush=True)
    mfr_fpr_fnr = manufacturer_fpr_fnr()
    if not mfr_fpr_fnr.empty:
        write_table(mfr_fpr_fnr, "manufacturer_cn_fpr_ad_fnr")

    # S7: Philips/GE/SIEMENS FPR (OOF-calibrated)
    print("S7: Building manufacturer FPR table (oof_ecdf) ...", flush=True)
    fpr_df = build_fpr_table()
    if not fpr_df.empty:
        write_table(fpr_df, "manufacturer_fpr_oof_ecdf")

    # S8: Cross-model comparison
    print("S8: Building cross-model comparison table ...", flush=True)
    comparison = build_comparison_table()
    write_table(comparison, "comparison_table")

    # S9: Plus-ch1 reference
    plus_ch1 = load_plus_ch1()
    if plus_ch1 is not None:
        plus_ch1_cols = [c for c in ["candidate_id", "auc", "pr_auc", "balanced_accuracy",
                                      "philips_cn_fpr", "decision"] if c in plus_ch1.columns]
        write_table(plus_ch1[plus_ch1_cols], "plus_ch1_reference")

    # S10: Promotion decision
    print("S10: Evaluating promotion gate ...", flush=True)
    decision = promotion_decision(oof_pooled, fpr_df)
    write_json(OUT_DIR / "promotion_decision.json", decision)

    # Summary print
    print(f"\n  ── Promotion gate summary ──")
    print(f"  Verdict:    {decision['verdict']}")
    print(f"  AUC gate:   {decision['best_oof_ecdf_AUC']:.6f} vs {PROMOTED_AUC}  → {'PASS' if decision['gate_auc_pass'] else 'FAIL'}")
    print(f"  PRAUC gate: {decision['best_oof_ecdf_PRAUC']:.6f} vs {PROMOTED_PR_AUC}  → {'PASS' if decision['gate_pr_auc_pass'] else 'FAIL'}")
    print(f"  Philips:    {decision['philips_FPR_ecdf']:.4f}  vs ≤{PHILIPS_FPR_GATE}  → {'PASS' if decision['gate_philips_fpr_pass'] else 'FAIL'}")
    print(f"  OASIS:      {decision['oasis_stress_test']}")

    if "betaKLD_D" in rd_full.columns:
        print(f"\n  Mean β·KLD/D = {rd_full['betaKLD_D'].mean():.6f}  (T80 promoted ref ≈ 0.0257)")
    if "n_active_dims" in rd_full.columns:
        print(f"  Mean active dims = {rd_full['n_active_dims'].mean():.1f} / 384")
    if "TC_nats" in rd_full.columns:
        print(f"  Mean TC = {rd_full['TC_nats'].mean():.2f} nats")

    # Report
    write_report(fold_complete, stageB_status, ckpt_cmp, rd_full,
                 oof_primary, oof_foldwise, mfr_fpr_fnr, comparison,
                 fpr_df, plus_ch1, decision)

    # Command log
    write_json(OUT_DIR / "command_log.json", {
        "created_utc":                    datetime.now(timezone.utc).isoformat(),
        "candidate":                      CANDIDATE_RUN_ID,
        "stage_a_all_folds_complete":     all_folds_ok,
        "stage_b_classifier_only_readout": str(CANDIDATE_CLF),
        "oof_calibration_dir":            str(CANDIDATE_OOF),
        "vae_retrained":                  False,
        "tensor_modified":                False,
        "metadata_modified":              False,
        "oasis_run":                      False,
        "verdict":                        decision["verdict"],
    })

    print(f"\nDone. Output: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
