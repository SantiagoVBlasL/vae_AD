#!/usr/bin/env python3
"""ch521 [5,2,1] Stage B completion + decision audit.

This script does NOT train any VAE or modify any source data.
It performs three phases:

  Phase 1 — Classifier-only sweep:
    Runs run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py on the
    already-trained ch521 FULL run to generate frozen latent cache + logreg_l2
    OOF predictions.

  Phase 2 — OOF ECDF calibration:
    Runs run_recover035_latent384_beta3p75_stageB_oof_score_calibration.py on
    the same run dir (reading the freshly generated latent cache) to apply
    all OOF calibration methods including oof_ecdf.

  Phase 3 — Decision analysis:
    Loads ch521 stageB primary results and compares against the locked reference
    [1,0,2] primary readout. Runs a paired bootstrap on matched OOF subjects.
    Generates the full decision package at the output directory.

Hard constraints (enforced by inspection, not by trust):
  - No VAE weights are modified.
  - No tensor or metadata file is modified.
  - Manufacturer/Site are used ONLY for error-profile stratification, never as
    classifier features.
  - Promotion gates are locked; no threshold refitting to increase AUC.
  - 128_S_2002 must be absent from classifier pool.
  - N must equal 397 (CN=300, AD=97) after filtering to CN+AD only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / "scripts" / "revision_bspc_2026"
RESULTS = PROJECT / "results" / "revision_bspc_2026"

PYTHON = Path("/home/diego/anaconda3/envs/vae_ad/bin/python")

CH521_RUN_DIR = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
    "recover035_ch521_latent384_beta3p75_T80_h10000_p560_full5x5_meta647_20260622"
)
CH521_CLF_READOUT = CH521_RUN_DIR / "classifier_only_readout"
CH521_STAGEB_DIR = RESULTS / "recover035_ch521_latent384_beta3p75_stageB_oof_score_calibration_20260623"

REF_RUN_DIR = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
REF_STAGEB_DIR = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"

OUT = RESULTS / "ch521_stageB_completion_decision_20260623"

# Primary readout convention (both candidate and reference)
PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEAT = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESH = "inner_oof_target_sens_ge_0p70_max_spec"

# Locked reference promotion gates
LOCKED_AUC = 0.782951
LOCKED_PR_AUC = 0.559873

# Reference primary readout (locked)
REF_AUC = 0.795155
REF_PR_AUC = 0.573934
REF_BA = 0.725979
REF_SENS = 0.731959
REF_SPEC = 0.720000
REF_F1 = 0.563492

EXPECTED_N = 397
EXPECTED_CN = 300
EXPECTED_AD = 97
EXCLUDED_SUBJECT = "128_S_2002"

BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 42

# ── Utilities ─────────────────────────────────────────────────────────────────

NOW = datetime.now(timezone.utc).isoformat()


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")
    return view.to_markdown(index=False) + "\n"


def write_table(stem: str, df: pd.DataFrame) -> None:
    df.to_csv(OUT / f"{stem}.csv", index=False)
    (OUT / f"{stem}.md").write_text(md_table(df), encoding="utf-8")


def write_md(stem: str, text: str) -> None:
    (OUT / f"{stem}.md").write_text(text, encoding="utf-8")


def compute_ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_score >= lo) & (y_score < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(float(y_true[mask].mean()) - float(y_score[mask].mean()))
    return float(ece)


def bootstrap_auc_prcauc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n: int = 5000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Return (auc_lo, auc_hi, prcauc_lo, prcauc_hi) at 2.5/97.5 percentiles."""
    rng = np.random.default_rng(seed)
    aucs, pr_aucs = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(float(roc_auc_score(yt, ys)))
        pr_aucs.append(float(average_precision_score(yt, ys)))
    return (
        float(np.percentile(aucs, 2.5)),
        float(np.percentile(aucs, 97.5)),
        float(np.percentile(pr_aucs, 2.5)),
        float(np.percentile(pr_aucs, 97.5)),
    )


def paired_bootstrap_delta(
    y_true: np.ndarray,
    score_cand: np.ndarray,
    score_ref: np.ndarray,
    n: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired bootstrap of AUC difference: candidate minus reference."""
    rng = np.random.default_rng(seed)
    delta_aucs, delta_prs = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            da = float(roc_auc_score(yt, score_cand[idx]) - roc_auc_score(yt, score_ref[idx]))
            dp = float(average_precision_score(yt, score_cand[idx]) - average_precision_score(yt, score_ref[idx]))
            delta_aucs.append(da)
            delta_prs.append(dp)
        except Exception:
            continue
    d_auc = np.array(delta_aucs)
    d_pr = np.array(delta_prs)
    obs_auc = float(roc_auc_score(y_true, score_cand) - roc_auc_score(y_true, score_ref))
    obs_pr = float(average_precision_score(y_true, score_cand) - average_precision_score(y_true, score_ref))
    p_auc = float((d_auc <= 0).mean())  # fraction of samples where cand ≤ ref
    p_pr = float((d_pr <= 0).mean())
    return {
        "obs_delta_auc": obs_auc,
        "obs_delta_pr_auc": obs_pr,
        "delta_auc_lo": float(np.percentile(d_auc, 2.5)),
        "delta_auc_hi": float(np.percentile(d_auc, 97.5)),
        "delta_pr_auc_lo": float(np.percentile(d_pr, 2.5)),
        "delta_pr_auc_hi": float(np.percentile(d_pr, 97.5)),
        "p_auc_cand_le_ref": p_auc,
        "p_pr_auc_cand_le_ref": p_pr,
        "n_bootstrap_valid": len(d_auc),
    }


# ── Phase 1: Classifier sweep ─────────────────────────────────────────────────

def run_phase1() -> int:
    """Run classifier-only sweep on ch521 run dir. Returns exit code."""
    cmd = [
        str(PYTHON),
        str(SCRIPTS / "run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"),
        "--run-dir", str(CH521_RUN_DIR),
        "--output-dir", str(CH521_CLF_READOUT),
        "--outer-folds", "5",
        "--inner-folds", "5",
        "--models", "logreg_l2",
        "--readout-feature-sets", "z_plus_age_sex",
        "--device", "cpu",
        "--n-jobs", "4",
        "--reuse-latent-cache",
    ]
    print("\n=== Phase 1: Classifier sweep ===")
    print("CMD:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode


# ── Phase 2: StageB OOF calibration ──────────────────────────────────────────

def run_phase2() -> int:
    """Run stageB OOF score calibration on ch521 run dir. Returns exit code."""
    cmd = [
        str(PYTHON),
        str(SCRIPTS / "run_recover035_latent384_beta3p75_stageB_oof_score_calibration.py"),
        "--run-dir", str(CH521_RUN_DIR),
        "--output-dir", str(CH521_STAGEB_DIR),
        "--n-jobs", "4",
        "--overwrite",
    ]
    print("\n=== Phase 2: StageB OOF calibration ===")
    print("CMD:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode


# ── Phase 3: Decision analysis ────────────────────────────────────────────────

def validate_pool(df: pd.DataFrame, label: str) -> List[str]:
    """Check N, CN/AD counts, excluded subject."""
    errors = []
    if EXCLUDED_SUBJECT in df["SubjectID"].values:
        errors.append(f"{label}: excluded subject {EXCLUDED_SUBJECT} present")
    cn = int((df["ResearchGroup_Mapped"].str.upper() == "CN").sum())
    ad = int((df["ResearchGroup_Mapped"].str.upper() == "AD").sum())
    n = len(df)
    if n != EXPECTED_N:
        errors.append(f"{label}: N={n} expected {EXPECTED_N}")
    if cn != EXPECTED_CN:
        errors.append(f"{label}: CN={cn} expected {EXPECTED_CN}")
    if ad != EXPECTED_AD:
        errors.append(f"{label}: AD={ad} expected {EXPECTED_AD}")
    dupes = df["SubjectID"].duplicated().sum()
    if dupes > 0:
        errors.append(f"{label}: {dupes} duplicated SubjectIDs")
    return errors


def load_primary_predictions(stageb_dir: Path, label: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load OOF predictions filtered to primary convention."""
    pred_csv = stageb_dir / "calib_predictions.csv"
    if not pred_csv.exists():
        return pd.DataFrame(), [f"{label}: {pred_csv} missing"]
    df = pd.read_csv(pred_csv)
    mask = (
        (df["model_name"] == PRIMARY_MODEL)
        & (df["feature_set"] == PRIMARY_FEAT)
        & (df["calib_method"] == PRIMARY_CALIB)
        & (df["threshold_strategy"] == PRIMARY_THRESH)
    )
    sub = df[mask].copy()
    errors = validate_pool(sub, label) if not sub.empty else [f"{label}: no rows match primary convention"]
    return sub, errors


def load_primary_pooled(stageb_dir: Path) -> Optional[pd.Series]:
    """Load pooled metrics for primary convention."""
    csv = stageb_dir / "calib_pooled_metrics.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    mask = (
        (df["model_name"] == PRIMARY_MODEL)
        & (df["feature_set"] == PRIMARY_FEAT)
        & (df["calib_method"] == PRIMARY_CALIB)
        & (df["threshold_strategy"] == PRIMARY_THRESH)
    )
    rows = df[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def load_foldwise_primary(stageb_dir: Path) -> pd.DataFrame:
    """Load foldwise metrics for primary convention."""
    csv = stageb_dir / "calib_foldwise_metrics.csv"
    if not csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv)
    mask = (
        (df["model_name"] == PRIMARY_MODEL)
        & (df["feature_set"] == PRIMARY_FEAT)
        & (df["calib_method"] == PRIMARY_CALIB)
        & (df["threshold_strategy"] == PRIMARY_THRESH)
    )
    return df[mask].copy()


def compute_primary_metrics_from_predictions(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute pooled metrics from OOF prediction rows."""
    y_true = df["y_true"].values.astype(int)
    y_score = df["y_score"].values.astype(float)
    # Threshold from predictions: use majority vote y_pred if present
    if "y_pred" not in df.columns:
        return {}
    y_pred = df["y_pred"].values.astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    ba = float(np.nanmean([sens, spec]))
    f1_denom = 2 * tp + fp + fn
    f1 = float(2 * tp / f1_denom) if f1_denom > 0 else float("nan")
    brier = float(brier_score_loss(y_true, y_score))
    ece = compute_ece(y_true, y_score)
    auc_lo, auc_hi, pr_lo, pr_hi = bootstrap_auc_prcauc(y_true, y_score, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    return {
        "n": int(len(y_true)),
        "n_cn": int((y_true == 0).sum()),
        "n_ad": int((y_true == 1).sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "auc": auc, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi,
        "pr_auc": pr_auc, "pr_auc_ci_lo": pr_lo, "pr_auc_ci_hi": pr_hi,
        "balanced_accuracy": ba,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": f1,
        "brier": brier,
        "ece": ece,
        "promotes": bool(auc > LOCKED_AUC and pr_auc >= LOCKED_PR_AUC),
    }


def manufacturer_error_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-manufacturer CN FPR and AD FNR (error profile only, not features)."""
    if "Manufacturer" not in df.columns or "y_pred" not in df.columns:
        return pd.DataFrame()
    rows = []
    for mfr in df["Manufacturer"].dropna().unique():
        sub = df[df["Manufacturer"] == mfr]
        cn = sub[sub["y_true"] == 0]
        ad = sub[sub["y_true"] == 1]
        n_cn = len(cn)
        n_ad = len(ad)
        fp = int((cn["y_pred"] == 1).sum()) if n_cn > 0 else 0
        fn = int((ad["y_pred"] == 0).sum()) if n_ad > 0 else 0
        fpr = fp / n_cn if n_cn > 0 else float("nan")
        fnr = fn / n_ad if n_ad > 0 else float("nan")
        rows.append({
            "manufacturer": mfr,
            "n_cn": n_cn,
            "n_ad": n_ad,
            "fp_cn": fp,
            "fn_ad": fn,
            "fpr_cn": round(fpr, 6),
            "fnr_ad": round(fnr, 6),
        })
    return pd.DataFrame(rows).sort_values("manufacturer")


def run_phase3() -> None:
    """Load results, compare, generate decision package."""
    print("\n=== Phase 3: Decision analysis ===")
    OUT.mkdir(parents=True, exist_ok=True)

    integrity_lines = [
        "# Stage B Completion Integrity",
        f"Generated: {NOW}",
        "",
        f"Candidate: ch521 [5,2,1] = DistanceCorr + MI_KNN + Pearson_Full",
        f"Reference: [1,0,2] = Pearson_Full + OMST + MI_KNN",
        "",
        "## Phase 1 — Classifier sweep",
    ]

    # ── Load ch521 stageB results ─────────────────────────────────────────────
    cand_preds, cand_errors = load_primary_predictions(CH521_STAGEB_DIR, "ch521")
    ref_preds, ref_errors = load_primary_predictions(REF_STAGEB_DIR, "reference")

    all_integrity_errors = cand_errors + ref_errors

    p1_ok = CH521_CLF_READOUT.exists() and (CH521_CLF_READOUT / "classifier_sweep_predictions.csv").exists()
    integrity_lines += [
        f"  Status: {'PASS' if p1_ok else 'FAIL'}",
        f"  Output: {CH521_CLF_READOUT}",
        "",
        "## Phase 2 — StageB OOF calibration",
    ]
    p2_ok = CH521_STAGEB_DIR.exists() and (CH521_STAGEB_DIR / "calib_predictions.csv").exists()
    integrity_lines += [
        f"  Status: {'PASS' if p2_ok else 'FAIL'}",
        f"  Output: {CH521_STAGEB_DIR}",
        "",
    ]

    if not cand_preds.empty:
        n_total = len(cand_preds)
        n_cn = int((cand_preds["ResearchGroup_Mapped"].str.upper() == "CN").sum())
        n_ad = int((cand_preds["ResearchGroup_Mapped"].str.upper() == "AD").sum())
        integrity_lines += [
            f"## Pool Validation (ch521 primary OOF predictions)",
            f"  N={n_total}, CN={n_cn}, AD={n_ad}",
            f"  {EXCLUDED_SUBJECT} absent: {EXCLUDED_SUBJECT not in cand_preds['SubjectID'].values}",
            f"  Duplicated SubjectIDs: {cand_preds['SubjectID'].duplicated().sum()}",
        ]
    if all_integrity_errors:
        integrity_lines += ["", "## INTEGRITY ERRORS"]
        integrity_lines += [f"  - {e}" for e in all_integrity_errors]
        integrity_lines += ["", "**FAIL** — integrity errors above must be resolved."]
    else:
        integrity_lines += ["", "**PASS** — all integrity checks passed."]

    write_md("stageb_completion_integrity", "\n".join(integrity_lines))

    if cand_preds.empty:
        print("ERROR: ch521 primary predictions not found; cannot generate comparison.")
        return

    # ── Compute primary metrics ───────────────────────────────────────────────
    cand_metrics = compute_primary_metrics_from_predictions(cand_preds)
    print(f"  ch521 primary AUC={cand_metrics.get('auc', 'N/A'):.6f}, PR-AUC={cand_metrics.get('pr_auc', 'N/A'):.6f}")

    cand_pooled_row = load_primary_pooled(CH521_STAGEB_DIR)
    ref_pooled_row = load_primary_pooled(REF_STAGEB_DIR)

    # Build stageb_primary_metrics table
    rows_pm = []
    for tag, m, pooled_row in [
        ("ch521 [5,2,1]", cand_metrics, cand_pooled_row),
        ("reference [1,0,2]", {
            "n": EXPECTED_N, "n_cn": EXPECTED_CN, "n_ad": EXPECTED_AD,
            "auc": REF_AUC, "auc_ci_lo": float("nan"), "auc_ci_hi": float("nan"),
            "pr_auc": REF_PR_AUC, "pr_auc_ci_lo": float("nan"), "pr_auc_ci_hi": float("nan"),
            "balanced_accuracy": REF_BA, "sensitivity": REF_SENS,
            "specificity": REF_SPEC, "f1": REF_F1,
            "promotes": True,
        }, ref_pooled_row),
    ]:
        row: Dict[str, Any] = {"model": tag}
        row["n"] = m.get("n", EXPECTED_N)
        row["auc"] = m.get("auc", float("nan"))
        row["auc_ci_lo"] = m.get("auc_ci_lo", float("nan"))
        row["auc_ci_hi"] = m.get("auc_ci_hi", float("nan"))
        row["pr_auc"] = m.get("pr_auc", float("nan"))
        row["pr_auc_ci_lo"] = m.get("pr_auc_ci_lo", float("nan"))
        row["pr_auc_ci_hi"] = m.get("pr_auc_ci_hi", float("nan"))
        row["balanced_accuracy"] = m.get("balanced_accuracy", float("nan"))
        row["sensitivity"] = m.get("sensitivity", float("nan"))
        row["specificity"] = m.get("specificity", float("nan"))
        row["f1"] = m.get("f1", float("nan"))
        row["brier"] = m.get("brier", float("nan"))
        row["ece"] = m.get("ece", float("nan"))
        row["promotes"] = m.get("promotes", False)
        row["delta_auc_vs_ref"] = row["auc"] - REF_AUC if tag != "reference [1,0,2]" else 0.0
        row["delta_pr_auc_vs_ref"] = row["pr_auc"] - REF_PR_AUC if tag != "reference [1,0,2]" else 0.0
        rows_pm.append(row)

    write_table("stageb_primary_metrics", pd.DataFrame(rows_pm))

    # ── Foldwise metrics ──────────────────────────────────────────────────────
    cand_fw = load_foldwise_primary(CH521_STAGEB_DIR)
    ref_fw = load_foldwise_primary(REF_STAGEB_DIR)

    fw_rows = []
    if not cand_fw.empty:
        for _, r in cand_fw.iterrows():
            fw_rows.append({
                "model": "ch521 [5,2,1]",
                "fold": r.get("fold", ""),
                "auc": r.get("auc", float("nan")),
                "pr_auc": r.get("pr_auc", float("nan")),
                "balanced_accuracy": r.get("balanced_accuracy", float("nan")),
                "sensitivity": r.get("sensitivity", float("nan")),
                "specificity": r.get("specificity", float("nan")),
                "f1": r.get("f1", float("nan")),
                "n": r.get("n", float("nan")),
                "n_ad": r.get("n_ad", float("nan")),
            })
    if not ref_fw.empty:
        for _, r in ref_fw.iterrows():
            fw_rows.append({
                "model": "reference [1,0,2]",
                "fold": r.get("fold", ""),
                "auc": r.get("auc", float("nan")),
                "pr_auc": r.get("pr_auc", float("nan")),
                "balanced_accuracy": r.get("balanced_accuracy", float("nan")),
                "sensitivity": r.get("sensitivity", float("nan")),
                "specificity": r.get("specificity", float("nan")),
                "f1": r.get("f1", float("nan")),
                "n": r.get("n", float("nan")),
                "n_ad": r.get("n_ad", float("nan")),
            })
    if fw_rows:
        write_table("stageb_foldwise_metrics", pd.DataFrame(fw_rows))

    # ── Copy calib_predictions.csv ────────────────────────────────────────────
    import shutil
    src = CH521_STAGEB_DIR / "calib_predictions.csv"
    if src.exists():
        shutil.copy2(src, OUT / "calib_predictions.csv")
        print(f"  Copied calib_predictions.csv ({src.stat().st_size} bytes)")

    # ── Manufacturer error profile ────────────────────────────────────────────
    mfr_cand = manufacturer_error_profile(cand_preds)
    mfr_rows = []
    if not mfr_cand.empty:
        for _, r in mfr_cand.iterrows():
            mfr_rows.append(dict(r, model="ch521 [5,2,1]"))
    if not ref_preds.empty:
        mfr_ref = manufacturer_error_profile(ref_preds)
        for _, r in mfr_ref.iterrows():
            mfr_rows.append(dict(r, model="reference [1,0,2]"))
    if mfr_rows:
        mfr_df = pd.DataFrame(mfr_rows)
        mfr_df = mfr_df[["model", "manufacturer", "n_cn", "n_ad", "fp_cn", "fn_ad", "fpr_cn", "fnr_ad"]]
        write_table("manufacturer_error_profile", mfr_df)

    # ── Paired bootstrap ──────────────────────────────────────────────────────
    boot_result: Optional[Dict[str, Any]] = None
    if not cand_preds.empty and not ref_preds.empty:
        common_subs = set(cand_preds["SubjectID"]) & set(ref_preds["SubjectID"])
        n_common = len(common_subs)
        print(f"  Paired bootstrap: {n_common} common subjects")
        if n_common >= 50:
            cand_m = cand_preds.set_index("SubjectID").loc[list(common_subs)]
            ref_m = ref_preds.set_index("SubjectID").loc[list(common_subs)]
            common_sorted = sorted(common_subs)
            cand_m = cand_preds.set_index("SubjectID").reindex(common_sorted)
            ref_m = ref_preds.set_index("SubjectID").reindex(common_sorted)
            yt = cand_m["y_true"].values.astype(int)
            sc = cand_m["y_score"].values.astype(float)
            sr = ref_m["y_score"].values.astype(float)
            boot_result = paired_bootstrap_delta(yt, sc, sr, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
            boot_result["n_common"] = n_common
            boot_result["n_bootstrap_valid"] = boot_result["n_bootstrap_valid"]
            boot_df = pd.DataFrame([{
                "comparison": "ch521 [5,2,1] vs reference [1,0,2]",
                "n_common_subjects": n_common,
                "obs_delta_auc": boot_result["obs_delta_auc"],
                "delta_auc_ci_lo": boot_result["delta_auc_lo"],
                "delta_auc_ci_hi": boot_result["delta_auc_hi"],
                "p_auc_cand_le_ref": boot_result["p_auc_cand_le_ref"],
                "obs_delta_pr_auc": boot_result["obs_delta_pr_auc"],
                "delta_pr_auc_ci_lo": boot_result["delta_pr_auc_lo"],
                "delta_pr_auc_ci_hi": boot_result["delta_pr_auc_hi"],
                "p_pr_auc_cand_le_ref": boot_result["p_pr_auc_cand_le_ref"],
                "n_bootstrap_samples": BOOTSTRAP_N,
                "n_bootstrap_valid": boot_result["n_bootstrap_valid"],
            }])
            write_table("paired_bootstrap_vs_reference", boot_df)
        else:
            print(f"  WARNING: only {n_common} common subjects; skipping paired bootstrap.")
            write_md("paired_bootstrap_vs_reference", f"# Paired Bootstrap\n\nInsufficient common subjects (n={n_common}); bootstrap not run.\n")

    # ── Model decision table ──────────────────────────────────────────────────
    cand_auc = cand_metrics.get("auc", float("nan"))
    cand_pr = cand_metrics.get("pr_auc", float("nan"))
    cand_ba = cand_metrics.get("balanced_accuracy", float("nan"))
    cand_f1 = cand_metrics.get("f1", float("nan"))
    cand_promotes = cand_metrics.get("promotes", False)

    delta_auc = cand_auc - REF_AUC if not np.isnan(cand_auc) else float("nan")
    delta_pr = cand_pr - REF_PR_AUC if not np.isnan(cand_pr) else float("nan")

    below_threshold = (
        (not np.isnan(delta_auc) and delta_auc < -0.01)
        or (not np.isnan(delta_pr) and delta_pr < -0.01)
    )
    improves_over_ref = (
        not np.isnan(delta_auc)
        and not np.isnan(delta_pr)
        and delta_auc > 0.0
        and delta_pr > 0.0
    )

    if not cand_promotes:
        decision = "REJECT — candidate does not pass ADNI promotion gate"
    elif below_threshold:
        decision = "REJECT — candidate below reference by > 0.01 on AUC or PR-AUC"
    elif improves_over_ref:
        decision = "FLAG FOR DISCUSSION — candidate exceeds reference; review required before replacement"
    else:
        decision = "NEUTRAL — candidate within tolerance of reference; no replacement recommended"

    dec_df = pd.DataFrame([{
        "model": "ch521 [5,2,1]",
        "auc": cand_auc,
        "pr_auc": cand_pr,
        "balanced_accuracy": cand_ba,
        "f1": cand_f1,
        "delta_auc_vs_ref": delta_auc,
        "delta_pr_auc_vs_ref": delta_pr,
        "promotes_adni_gate": cand_promotes,
        "decision": decision,
    }, {
        "model": "reference [1,0,2] (FINAL SELECTED)",
        "auc": REF_AUC,
        "pr_auc": REF_PR_AUC,
        "balanced_accuracy": REF_BA,
        "f1": REF_F1,
        "delta_auc_vs_ref": 0.0,
        "delta_pr_auc_vs_ref": 0.0,
        "promotes_adni_gate": True,
        "decision": "FINAL SELECTED — no replacement warranted",
    }])
    write_table("model_decision_table", dec_df)

    # ── Final recommendation ──────────────────────────────────────────────────
    boot_summary = ""
    if boot_result is not None:
        boot_summary = (
            f"\n## Paired Bootstrap (N={boot_result['n_common']} common subjects)\n\n"
            f"| Metric | Observed Δ | 95% CI | p(cand ≤ ref) |\n"
            f"|:-------|:----------:|:------:|:-------------:|\n"
            f"| AUC | {boot_result['obs_delta_auc']:+.6f} | "
            f"[{boot_result['delta_auc_lo']:+.6f}, {boot_result['delta_auc_hi']:+.6f}] | "
            f"{boot_result['p_auc_cand_le_ref']:.4f} |\n"
            f"| PR-AUC | {boot_result['obs_delta_pr_auc']:+.6f} | "
            f"[{boot_result['delta_pr_auc_lo']:+.6f}, {boot_result['delta_pr_auc_hi']:+.6f}] | "
            f"{boot_result['p_pr_auc_cand_le_ref']:.4f} |\n"
        )

    philips_note = ""
    if not mfr_cand.empty:
        ph_row = mfr_cand[mfr_cand["manufacturer"].str.upper() == "PHILIPS"]
        ref_ph = None
        if not ref_preds.empty:
            mfr_r = manufacturer_error_profile(ref_preds)
            ref_ph_row = mfr_r[mfr_r["manufacturer"].str.upper() == "PHILIPS"]
            if not ref_ph_row.empty:
                ref_ph = float(ref_ph_row.iloc[0]["fpr_cn"])
        if not ph_row.empty:
            ph_fpr = float(ph_row.iloc[0]["fpr_cn"])
            ref_str = f" (reference: {ref_ph:.4f})" if ref_ph is not None else ""
            philips_note = f"\n**Philips CN FPR:** {ph_fpr:.4f}{ref_str}\n"

    rec_lines = [
        "# Final Recommendation: ch521 [5,2,1] Stage B Completion",
        f"Generated: {NOW}",
        "",
        "## Candidate",
        "",
        "**ch521 [5,2,1]** = DistanceCorr + MI_KNN_Symmetric + Pearson_Full_FisherZ_Signed",
        "Architecture: ld=384, β=3.75, T₀=80, h=10000, p=560, 5×5 CV",
        "Run: recover035_ch521_latent384_beta3p75_T80_h10000_p560_full5x5_meta647_20260622",
        "",
        "## Stage B Primary Metrics (logreg_l2_original / z_plus_age_sex / oof_ecdf / primary threshold)",
        "",
        f"| Metric | ch521 [5,2,1] | Reference [1,0,2] | Δ |",
        f"|:-------|:---:|:---:|:---:|",
        f"| AUC (pooled) | {cand_auc:.6f} | {REF_AUC:.6f} | {delta_auc:+.6f} |",
        f"| PR-AUC (pooled) | {cand_pr:.6f} | {REF_PR_AUC:.6f} | {delta_pr:+.6f} |",
        f"| BA | {cand_ba:.6f} | {REF_BA:.6f} | {cand_ba - REF_BA:+.6f} |",
        f"| Sens | {cand_metrics.get('sensitivity', float('nan')):.6f} | {REF_SENS:.6f} | {cand_metrics.get('sensitivity', float('nan')) - REF_SENS:+.6f} |",
        f"| Spec | {cand_metrics.get('specificity', float('nan')):.6f} | {REF_SPEC:.6f} | {cand_metrics.get('specificity', float('nan')) - REF_SPEC:+.6f} |",
        f"| F1 | {cand_f1:.6f} | {REF_F1:.6f} | {cand_f1 - REF_F1:+.6f} |",
        f"| Brier | {cand_metrics.get('brier', float('nan')):.6f} | — | — |",
        f"| ECE | {cand_metrics.get('ece', float('nan')):.6f} | — | — |",
        f"| Promotes gate | {cand_promotes} | True | — |",
        "",
        philips_note,
        boot_summary,
        "## Decision",
        "",
        f"**{decision}**",
        "",
        "### Reasoning",
        "",
    ]

    if not cand_promotes:
        rec_lines += [
            f"The ch521 [5,2,1] candidate does not pass the locked promotion gate:",
            f"  AUC={cand_auc:.4f} (gate: >{LOCKED_AUC:.6f}), PR-AUC={cand_pr:.4f} (gate: ≥{LOCKED_PR_AUC:.6f}).",
            "",
            "No promotion to OASIS, no model replacement. The reference [1,0,2] remains final selected.",
        ]
    elif below_threshold:
        rec_lines += [
            f"The ch521 [5,2,1] candidate passes the promotion gate but is below the reference:",
            f"  ΔAUC={delta_auc:+.4f}, ΔPR-AUC={delta_pr:+.4f} (threshold: −0.01).",
            "",
            "No promotion to OASIS, no model replacement. The reference [1,0,2] remains final selected.",
        ]
    elif improves_over_ref:
        rec_lines += [
            f"The ch521 [5,2,1] candidate passes the promotion gate and exceeds the reference:",
            f"  ΔAUC={delta_auc:+.4f}, ΔPR-AUC={delta_pr:+.4f}.",
            "",
            "This warrants a separate discussion before any model replacement decision.",
            "Do NOT replace the reference model without explicit user approval.",
        ]
    else:
        rec_lines += [
            f"The ch521 [5,2,1] candidate passes the promotion gate but does not clearly exceed the reference:",
            f"  ΔAUC={delta_auc:+.4f}, ΔPR-AUC={delta_pr:+.4f}.",
            "",
            "No replacement recommended. The reference [1,0,2] remains final selected.",
        ]

    rec_lines += [
        "",
        "## Status of All Active Candidates",
        "",
        "| Model | Status | Next step |",
        "|:------|:-------|:----------|",
        "| [1,0,2] recover035 | **FINAL SELECTED** | No action needed |",
        "| ch1only β4.5 (reprocessed14) | **CONDITIONAL** | OASIS inference pending |",
        "| [4,1] valsplitfix | **REJECTED** | None |",
        f"| ch521 [5,2,1] | **{decision.split('—')[0].strip()}** | See decision above |",
    ]

    write_md("final_recommendation", "\n".join(rec_lines))

    # ── command_log.json ──────────────────────────────────────────────────────
    log = {
        "generated": NOW,
        "script": __file__,
        "read_only": True,
        "did_train_vae": False,
        "did_modify_tensors": False,
        "did_modify_metadata": False,
        "did_modify_predictions": False,
        "classifier_sweep_output": str(CH521_CLF_READOUT),
        "stageb_calibration_output": str(CH521_STAGEB_DIR),
        "reference_stageb": str(REF_STAGEB_DIR),
        "output_dir": str(OUT),
        "primary_convention": {
            "model": PRIMARY_MODEL,
            "feature_set": PRIMARY_FEAT,
            "calib_method": PRIMARY_CALIB,
            "threshold_strategy": PRIMARY_THRESH,
        },
        "locked_gates": {
            "auc": LOCKED_AUC,
            "pr_auc": LOCKED_PR_AUC,
        },
        "integrity_errors": all_integrity_errors,
        "decision": decision,
    }
    (OUT / "command_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== Decision package written to {OUT} ===")
    print(f"  Decision: {decision}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-phase1", action="store_true", help="Skip classifier sweep (if already done)")
    parser.add_argument("--skip-phase2", action="store_true", help="Skip stageB calibration (if already done)")
    parser.add_argument("--phase3-only", action="store_true", help="Skip both phases, only generate decision package")
    args = parser.parse_args()

    if args.phase3_only:
        args.skip_phase1 = True
        args.skip_phase2 = True

    rc1 = 0
    if not args.skip_phase1:
        if CH521_CLF_READOUT.exists() and (CH521_CLF_READOUT / "classifier_sweep_predictions.csv").exists():
            print(f"  Phase 1 output exists at {CH521_CLF_READOUT}; skipping (pass --skip-phase1 explicitly or delete to rerun).")
        else:
            rc1 = run_phase1()
            if rc1 != 0:
                print(f"ERROR: Phase 1 failed with exit code {rc1}")
                return rc1
    else:
        print("  Phase 1 skipped.")

    rc2 = 0
    if not args.skip_phase2:
        if CH521_STAGEB_DIR.exists() and (CH521_STAGEB_DIR / "calib_predictions.csv").exists():
            print(f"  Phase 2 output exists at {CH521_STAGEB_DIR}; skipping.")
        else:
            rc2 = run_phase2()
            if rc2 != 0:
                print(f"ERROR: Phase 2 failed with exit code {rc2}")
                return rc2
    else:
        print("  Phase 2 skipped.")

    run_phase3()
    return 0


if __name__ == "__main__":
    sys.exit(main())
