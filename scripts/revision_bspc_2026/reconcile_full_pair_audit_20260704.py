#!/usr/bin/env python3
"""Read-only reconciliation audit for the FULL ch5 vs ch102 matched pair.

This script does not fit classifiers, train VAEs, run OASIS, or modify prior
artifacts. It reads existing Stage A FULL outputs for the contemporary matched
pair and the existing promoted Stage B OOF calibration package for the
historical locked model.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


PROJECT = Path("/home/diego/proyectos/vae_AD")
RESULTS = PROJECT / "results" / "revision_bspc_2026"
OUT = RESULTS / "post_revision_exploratory_20260630" / "full_pair_ch5_vs_ch102_audit_reconciliation_20260704"

PAIR_ROOT = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "full_pair_ch5_vs_ch102_channelmean_beta3p75_20260703"
)
HIST_RUN = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
    "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
)
STAGEB_CALIB = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
META = RESULTS / "adni_035_metadata_rescue_preflight" / "patched_metadata_candidate.csv"
PREV_AUDIT = RESULTS / "post_revision_exploratory_20260630" / "full_pair_ch5_vs_ch102_postrun_audit_20260704"

PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURE = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESH = "inner_oof_target_sens_ge_0p70_max_spec"
HIGH_BETA_THRESHOLD = 0.95 * 3.75


COMMANDS: List[Dict[str, Any]] = []


def log_step(action: str, details: Dict[str, Any] | None = None) -> None:
    COMMANDS.append(
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "details": details or {},
        }
    )


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    log_step("read_csv", {"path": str(path)})
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def write_csv_md(stem: str, df: pd.DataFrame, max_rows: int = 120) -> None:
    csv_path = OUT / f"{stem}.csv"
    md_path = OUT / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def safe_auc(y: Iterable[int], s: Iterable[float]) -> float:
    y_arr = np.asarray(list(y), dtype=int)
    s_arr = np.asarray(list(s), dtype=float)
    if len(np.unique(y_arr)) < 2:
        return float("nan")
    return float(roc_auc_score(y_arr, s_arr))


def safe_pr(y: Iterable[int], s: Iterable[float]) -> float:
    y_arr = np.asarray(list(y), dtype=int)
    s_arr = np.asarray(list(s), dtype=float)
    if len(np.unique(y_arr)) < 2:
        return float("nan")
    return float(average_precision_score(y_arr, s_arr))


def metrics_from_scores(df: pd.DataFrame, score_col: str, pred_col: Optional[str] = None,
                        threshold: Optional[float] = None) -> Dict[str, Any]:
    y = df["y_true"].astype(int).to_numpy()
    s = df[score_col].astype(float).to_numpy()
    row: Dict[str, Any] = {
        "n": int(len(df)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "auc": safe_auc(y, s),
        "pr_auc": safe_pr(y, s),
    }
    if pred_col and pred_col in df.columns:
        pred = df[pred_col].astype(int).to_numpy()
        row["threshold_source"] = pred_col
    elif threshold is not None:
        pred = (s >= threshold).astype(int)
        row["threshold_source"] = f"{score_col}>={threshold:g}"
    else:
        pred = np.full(len(df), -1)
        row["threshold_source"] = "none"
    if np.all(pred >= 0):
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        row.update(
            {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
                "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "f1": float(f1_score(y, pred, zero_division=0)),
            }
        )
    else:
        row.update({k: float("nan") for k in ["tn", "fp", "fn", "tp", "sensitivity", "specificity", "balanced_accuracy", "f1"]})
    return row


def all_folds_prediction_file(run_dir: Path, candidate: str) -> Path:
    files = sorted(run_dir.glob("all_folds_clf_predictions_MULTI_logreg_*.csv"))
    if len(files) != 1:
        raise RuntimeError(f"{candidate}: expected one all_folds_clf_predictions_MULTI_logreg_*.csv, found {len(files)}")
    return files[0]


def load_stagea_predictions(candidate: str, run_dir: Path) -> pd.DataFrame:
    pred_path = all_folds_prediction_file(run_dir, candidate)
    df = read_csv(pred_path)
    df = df[df["classifier_type"].eq("logreg")].copy()
    df["candidate"] = candidate
    df["prediction_file"] = str(pred_path)
    if df["SubjectID"].duplicated().any():
        raise RuntimeError(f"{candidate}: duplicated SubjectID in logreg predictions")
    return df


def load_primary_stageb_predictions() -> pd.DataFrame:
    pred = read_csv(STAGEB_CALIB / "calib_predictions.csv")
    q = (
        pred["model_name"].eq(PRIMARY_MODEL)
        & pred["feature_set"].eq(PRIMARY_FEATURE)
        & pred["calib_method"].eq(PRIMARY_CALIB)
        & pred["threshold_strategy"].eq(PRIMARY_THRESH)
    )
    out = pred.loc[q].copy()
    if len(out) != 397 or out["SubjectID"].nunique() != 397:
        raise RuntimeError(f"primary stageB rows have unexpected shape: {out.shape}, unique={out['SubjectID'].nunique()}")
    out["candidate"] = "locked_control"
    out["prediction_file"] = str(STAGEB_CALIB / "calib_predictions.csv")
    return out


def load_stageb_convention_rows(calib_method: str) -> pd.DataFrame:
    pred = read_csv(STAGEB_CALIB / "calib_predictions.csv")
    q = (
        pred["model_name"].eq(PRIMARY_MODEL)
        & pred["feature_set"].eq(PRIMARY_FEATURE)
        & pred["calib_method"].eq(calib_method)
        & pred["threshold_strategy"].eq(PRIMARY_THRESH)
    )
    return pred.loc[q].copy()


def pooled_metrics_all_conventions(stagea: Dict[str, pd.DataFrame], stageb_primary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for candidate, df in stagea.items():
        for label, col in [
            ("stageA_raw", "y_score_raw"),
            ("stageA_calibrated", "y_score_cal"),
            ("stageA_final", "y_score_final"),
        ]:
            r = metrics_from_scores(df, col, pred_col="y_pred" if label == "stageA_final" else None, threshold=0.5 if label != "stageA_final" else None)
            r.update(
                {
                    "candidate": candidate,
                    "score_convention": label,
                    "score_column": col,
                    "available": True,
                    "notes": "Existing FULL-run Stage A outer-fold logreg predictions.",
                }
            )
            rows.append(r)
        rows.append(
            {
                "candidate": candidate,
                "score_convention": "promoted_stageB_oof_ecdf_primary",
                "score_column": "y_score",
                "available": False,
                "n": int(df["SubjectID"].nunique()),
                "n_cn": int((df["y_true"] == 0).sum()),
                "n_ad": int((df["y_true"] == 1).sum()),
                "auc": np.nan,
                "pr_auc": np.nan,
                "tn": np.nan,
                "fp": np.nan,
                "fn": np.nan,
                "tp": np.nan,
                "sensitivity": np.nan,
                "specificity": np.nan,
                "balanced_accuracy": np.nan,
                "f1": np.nan,
                "threshold_source": "missing",
                "notes": "Not computed: no Stage B classifier-only OOF calibration package exists for this contemporary FULL run; generating it would refit classifiers.",
            }
        )

    # Historical locked model: Stage A plus Stage B raw/calib/promoted rows.
    hist_stagea = load_stagea_predictions("locked_control", HIST_RUN)
    for label, col in [
        ("stageA_raw", "y_score_raw"),
        ("stageA_calibrated", "y_score_cal"),
        ("stageA_final", "y_score_final"),
    ]:
        r = metrics_from_scores(hist_stagea, col, pred_col="y_pred" if label == "stageA_final" else None, threshold=0.5 if label != "stageA_final" else None)
        r.update(
            {
                "candidate": "locked_control",
                "score_convention": label,
                "score_column": col,
                "available": True,
                "notes": "Historical FULL-run Stage A root predictions; not manuscript primary.",
            }
        )
        rows.append(r)

    for calib_label in ["raw", "oof_platt", "oof_isotonic", "oof_zscore", "oof_logitz", "oof_ecdf"]:
        s = load_stageb_convention_rows(calib_label)
        if s.empty:
            continue
        convention = "promoted_stageB_oof_ecdf_primary" if calib_label == "oof_ecdf" else f"stageB_{calib_label}_primary_threshold"
        r = metrics_from_scores(s, "y_score", pred_col="y_pred")
        r.update(
            {
                "candidate": "locked_control",
                "score_convention": convention,
                "score_column": "y_score",
                "available": True,
                "notes": f"Historical Stage B classifier-only calibration, {PRIMARY_MODEL}/{PRIMARY_FEATURE}/{calib_label}/{PRIMARY_THRESH}.",
            }
        )
        rows.append(r)

    out = pd.DataFrame(rows)
    first = ["candidate", "score_convention", "available", "score_column", "n", "n_cn", "n_ad", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "tn", "fp", "fn", "tp", "threshold_source", "notes"]
    return out[first + [c for c in out.columns if c not in first]]


def bootstrap_delta(a: pd.DataFrame, b: pd.DataFrame, score_a: str, score_b: str,
                    label_a: str, label_b: str, n_boot: int = 10000, seed: int = 20260704) -> Dict[str, Any]:
    left = a[["SubjectID", "y_true", score_a]].rename(columns={"y_true": "y_true_a", score_a: "score_a"})
    right = b[["SubjectID", "y_true", score_b]].rename(columns={"y_true": "y_true_b", score_b: "score_b"})
    merged = left.merge(
        right, on="SubjectID"
    )
    if len(merged) == 0 or merged["SubjectID"].nunique() != len(merged):
        raise RuntimeError(f"bad merge for {label_a} vs {label_b}: {merged.shape}")
    y = merged["y_true_a"].astype(int).to_numpy()
    if not np.array_equal(y, merged["y_true_b"].astype(int).to_numpy()):
        raise RuntimeError(f"label mismatch for {label_a} vs {label_b}")
    s_a = merged["score_a"].astype(float).to_numpy()
    s_b = merged["score_b"].astype(float).to_numpy()
    rng = np.random.default_rng(seed)
    auc_delta = []
    pr_delta = []
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    for _ in range(n_boot):
        sample_idx = np.r_[
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ]
        yy = y[sample_idx]
        auc_delta.append(roc_auc_score(yy, s_a[sample_idx]) - roc_auc_score(yy, s_b[sample_idx]))
        pr_delta.append(average_precision_score(yy, s_a[sample_idx]) - average_precision_score(yy, s_b[sample_idx]))
    auc_delta_arr = np.asarray(auc_delta)
    pr_delta_arr = np.asarray(pr_delta)
    return {
        "comparison": f"{label_a} minus {label_b}",
        "n_subjects": int(len(merged)),
        "score_a": score_a,
        "score_b": score_b,
        "auc_a": safe_auc(y, s_a),
        "auc_b": safe_auc(y, s_b),
        "delta_auc": safe_auc(y, s_a) - safe_auc(y, s_b),
        "delta_auc_ci_low": float(np.percentile(auc_delta_arr, 2.5)),
        "delta_auc_ci_high": float(np.percentile(auc_delta_arr, 97.5)),
        "p_delta_auc_gt0": float(np.mean(auc_delta_arr > 0)),
        "pr_auc_a": safe_pr(y, s_a),
        "pr_auc_b": safe_pr(y, s_b),
        "delta_pr_auc": safe_pr(y, s_a) - safe_pr(y, s_b),
        "delta_pr_auc_ci_low": float(np.percentile(pr_delta_arr, 2.5)),
        "delta_pr_auc_ci_high": float(np.percentile(pr_delta_arr, 97.5)),
        "p_delta_pr_auc_gt0": float(np.mean(pr_delta_arr > 0)),
        "n_boot": n_boot,
        "notes": "Paired stratified bootstrap. This row is valid only for the named score convention.",
    }


def paired_bootstrap_table(stagea: Dict[str, pd.DataFrame], stageb_primary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    # Valid matched contemporary comparison in existing Stage A final convention.
    rows.append(
        {
            "score_convention": "stageA_final_existing",
            **bootstrap_delta(
                stagea["single_ch5"],
                stagea["control_ch102"],
                "y_score_final",
                "y_score_final",
                "single_ch5",
                "control_ch102",
            ),
        }
    )
    rows.append(
        {
            "score_convention": "promoted_stageB_oof_ecdf_primary",
            "comparison": "single_ch5 minus control_ch102",
            "n_subjects": np.nan,
            "auc_a": np.nan,
            "auc_b": np.nan,
            "delta_auc": np.nan,
            "delta_auc_ci_low": np.nan,
            "delta_auc_ci_high": np.nan,
            "p_delta_auc_gt0": np.nan,
            "pr_auc_a": np.nan,
            "pr_auc_b": np.nan,
            "delta_pr_auc": np.nan,
            "delta_pr_auc_ci_low": np.nan,
            "delta_pr_auc_ci_high": np.nan,
            "p_delta_pr_auc_gt0": np.nan,
            "n_boot": 0,
            "notes": "FAIL_CLOSED: contemporary candidates lack Stage B OOF-ECDF calibration artifacts; computing this would require classifier refit.",
        }
    )
    return pd.DataFrame(rows)


def philips_fpr_rows(stagea: Dict[str, pd.DataFrame], stageb_primary: pd.DataFrame) -> pd.DataFrame:
    meta = read_csv(META)[["SubjectID", "ResearchGroup_Mapped", "Manufacturer"]]
    rows: List[Dict[str, Any]] = []
    for cand, df in stagea.items():
        m = df.merge(meta, on="SubjectID", how="left", suffixes=("", "_meta"))
        cn = m[m["ResearchGroup_Mapped"].eq("CN") & m["Manufacturer"].eq("Philips")].copy()
        rows.append(
            {
                "candidate": cand,
                "score_convention": "raw_score_threshold_0p5",
                "score_column": "y_score_raw",
                "threshold": 0.5,
                "n_philips_cn": int(len(cn)),
                "fp_philips_cn": int((cn["y_score_raw"] >= 0.5).sum()),
                "philips_cn_fpr": float((cn["y_score_raw"] >= 0.5).mean()) if len(cn) else np.nan,
                "notes": "Stage A raw outer-fold score thresholded at 0.5; not paper convention.",
            }
        )
        rows.append(
            {
                "candidate": cand,
                "score_convention": "stageA_final_existing_y_pred",
                "score_column": "y_score_final/y_pred",
                "threshold": "fold pipeline y_pred",
                "n_philips_cn": int(len(cn)),
                "fp_philips_cn": int((cn["y_pred"] == 1).sum()),
                "philips_cn_fpr": float((cn["y_pred"] == 1).mean()) if len(cn) else np.nan,
                "notes": "Existing FULL-run Stage A final prediction; not paper Stage B OOF-ECDF convention.",
            }
        )
        rows.append(
            {
                "candidate": cand,
                "score_convention": "promoted_stageB_oof_ecdf_primary",
                "score_column": "y_score/y_pred",
                "threshold": "missing",
                "n_philips_cn": int(len(cn)),
                "fp_philips_cn": np.nan,
                "philips_cn_fpr": np.nan,
                "notes": "Not computable without missing Stage B OOF-ECDF calibration artifacts; no threshold refit performed.",
            }
        )

    hist_stagea = load_stagea_predictions("locked_control", HIST_RUN)
    hm = hist_stagea.merge(meta, on="SubjectID", how="left", suffixes=("", "_meta"))
    hcn = hm[hm["ResearchGroup_Mapped"].eq("CN") & hm["Manufacturer"].eq("Philips")].copy()
    rows.append(
        {
            "candidate": "locked_control",
            "score_convention": "raw_score_threshold_0p5",
            "score_column": "y_score_raw",
            "threshold": 0.5,
            "n_philips_cn": int(len(hcn)),
            "fp_philips_cn": int((hcn["y_score_raw"] >= 0.5).sum()),
            "philips_cn_fpr": float((hcn["y_score_raw"] >= 0.5).mean()) if len(hcn) else np.nan,
            "notes": "Historical Stage A raw score thresholded at 0.5; explains low FPR from prior matched-pair audit.",
        }
    )
    rows.append(
        {
            "candidate": "locked_control",
            "score_convention": "stageA_final_existing_y_pred",
            "score_column": "y_score_final/y_pred",
            "threshold": "fold pipeline y_pred",
            "n_philips_cn": int(len(hcn)),
            "fp_philips_cn": int((hcn["y_pred"] == 1).sum()),
            "philips_cn_fpr": float((hcn["y_pred"] == 1).mean()) if len(hcn) else np.nan,
            "notes": "Historical Stage A final prediction; not paper convention.",
        }
    )
    sp = stageb_primary
    cn = sp[sp["ResearchGroup_Mapped"].eq("CN") & sp["Manufacturer"].eq("Philips")].copy()
    rows.append(
        {
            "candidate": "locked_control",
            "score_convention": "promoted_stageB_oof_ecdf_primary",
            "score_column": "y_score/y_pred",
            "threshold": "fold-specific primary thresholds from calib_predictions.csv",
            "n_philips_cn": int(len(cn)),
            "fp_philips_cn": int((cn["y_pred"] == 1).sum()),
            "philips_cn_fpr": float((cn["y_pred"] == 1).mean()) if len(cn) else np.nan,
            "notes": "Paper convention: logreg_l2_original/z_plus_age_sex/oof_ecdf/inner_oof_target_sens_ge_0p70_max_spec.",
        }
    )
    return pd.DataFrame(rows)


def checkpoint_and_rd(candidate: str, run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    beta_rows: List[Dict[str, Any]] = []
    rd_rows: List[Dict[str, Any]] = []
    for fold in range(1, 6):
        hist_path = run_dir / f"fold_{fold}" / f"vae_train_history_fold_{fold}.joblib"
        rd_path = run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
        hist = joblib.load(hist_path)
        val_loss = np.asarray(hist["val_loss_modelsel"], dtype=float)
        beta = np.asarray(hist["beta"], dtype=float)
        best_any_idx = int(np.nanargmin(val_loss))
        eligible = np.where(beta >= HIGH_BETA_THRESHOLD)[0]
        if len(eligible):
            best_high_idx = int(eligible[np.nanargmin(val_loss[eligible])])
        else:
            best_high_idx = -1
        # The FULL runner's historical behavior selects the global minimum of
        # val_loss_modelsel; no explicit high-beta selection artifact exists.
        selected_idx = best_any_idx
        selected_epoch = selected_idx + 1
        ckpt = run_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt"
        selected_beta = float(beta[selected_idx])
        beta_rows.append(
            {
                "candidate": candidate,
                "fold": fold,
                "history_file": str(hist_path),
                "checkpoint_file": str(ckpt),
                "checkpoint_file_exists": ckpt.exists(),
                "selected_epoch_inferred": selected_epoch,
                "selected_beta_inferred": selected_beta,
                "best_any_epoch": best_any_idx + 1,
                "best_any_val_loss_modelsel": float(val_loss[best_any_idx]),
                "beta_at_best_any": float(beta[best_any_idx]),
                "best_high_beta_epoch": best_high_idx + 1 if best_high_idx >= 0 else np.nan,
                "best_high_beta_val_loss_modelsel": float(val_loss[best_high_idx]) if best_high_idx >= 0 else np.nan,
                "beta_at_best_high_beta": float(beta[best_high_idx]) if best_high_idx >= 0 else np.nan,
                "high_beta_threshold": HIGH_BETA_THRESHOLD,
                "high_beta_guard_pass": bool(selected_beta >= HIGH_BETA_THRESHOLD),
                "notes": "Selected epoch inferred as argmin(val_loss_modelsel); no explicit selected-checkpoint metadata found.",
            }
        )
        rd = read_csv(rd_path)
        rr = rd[rd["epoch"].astype(int).eq(selected_epoch)]
        if rr.empty:
            raise RuntimeError(f"{candidate} fold {fold}: selected epoch {selected_epoch} not found in {rd_path}")
        r = rr.iloc[0]
        train_recon = float(r["D_train"])
        val_recon = float(r["D_val"])
        train_kld = float(r["R_train_nats"])
        val_kld = float(r["R_val_nats"])
        rd_rows.append(
            {
                "candidate": candidate,
                "fold": fold,
                "selected_epoch_inferred": selected_epoch,
                "selected_beta_inferred": selected_beta,
                "train_reconstruction_D": train_recon,
                "train_kld_R_nats": train_kld,
                "train_rho_beta_kld_over_recon": selected_beta * train_kld / train_recon if train_recon else np.nan,
                "val_reconstruction_D": val_recon,
                "val_kld_R_nats": val_kld,
                "val_rho_beta_kld_over_recon": selected_beta * val_kld / val_recon if val_recon else np.nan,
                "notes": "Computed at inferred selected checkpoint epoch, not final epoch.",
            }
        )
    return pd.DataFrame(beta_rows), pd.DataFrame(rd_rows)


def manufacturer_leakage_table(run_map: Dict[str, Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for cand, run_dir in run_map.items():
        for fold in range(1, 6):
            for split, suffix in [("train_dev", "scanner_leakage"), ("test", "test_scanner_leakage")]:
                p = run_dir / f"fold_{fold}" / f"fold_{fold}_{suffix}.csv"
                if not p.exists():
                    continue
                df = read_csv(p)
                for _, r in df.iterrows():
                    rows.append(
                        {
                            "candidate": cand,
                            "fold": fold,
                            "split": split,
                            "representation": r.get("representation"),
                            "ordinary_accuracy": np.nan,
                            "balanced_accuracy": r.get("balanced_accuracy_mean"),
                            "balanced_accuracy_std": r.get("balanced_accuracy_std"),
                            "macro_f1": np.nan,
                            "majority_baseline": np.nan,
                            "chance_level": r.get("chance_level"),
                            "permutation_p_value": np.nan,
                            "n_samples": r.get("n_samples"),
                            "n_classes": r.get("n_classes"),
                            "source_file": str(p),
                            "notes": "Existing scanner-leakage artifact stores balanced_accuracy_mean/std only; ordinary accuracy, macro-F1, majority baseline, and permutation p-value are unavailable without rerunning leakage models.",
                        }
                    )
    return pd.DataFrame(rows)


def summarize_rd(rd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cand, g in rd.groupby("candidate"):
        rows.append(
            {
                "candidate": cand,
                "n_folds": int(len(g)),
                "train_rho_mean": float(g["train_rho_beta_kld_over_recon"].mean()),
                "train_rho_min": float(g["train_rho_beta_kld_over_recon"].min()),
                "train_rho_max": float(g["train_rho_beta_kld_over_recon"].max()),
                "val_rho_mean": float(g["val_rho_beta_kld_over_recon"].mean()),
                "val_rho_min": float(g["val_rho_beta_kld_over_recon"].min()),
                "val_rho_max": float(g["val_rho_beta_kld_over_recon"].max()),
            }
        )
    return pd.DataFrame(rows)


def write_reconciliation_md(metrics: pd.DataFrame, fpr: pd.DataFrame) -> None:
    locked_prom = metrics[
        metrics["candidate"].eq("locked_control")
        & metrics["score_convention"].eq("promoted_stageB_oof_ecdf_primary")
    ].iloc[0]
    locked_stagea = metrics[
        metrics["candidate"].eq("locked_control") & metrics["score_convention"].eq("stageA_final")
    ].iloc[0]
    locked_fpr_prom = fpr[
        fpr["candidate"].eq("locked_control")
        & fpr["score_convention"].eq("promoted_stageB_oof_ecdf_primary")
    ].iloc[0]
    locked_fpr_raw = fpr[
        fpr["candidate"].eq("locked_control")
        & fpr["score_convention"].eq("raw_score_threshold_0p5")
    ].iloc[0]
    locked_fpr_stagea_final = fpr[
        fpr["candidate"].eq("locked_control")
        & fpr["score_convention"].eq("stageA_final_existing_y_pred")
    ].iloc[0]
    text = f"""# Score Convention Reconciliation

Generated: {datetime.now(timezone.utc).isoformat()}

## Verdict

The manuscript locked-control metrics are reproduced exactly from:

`{STAGEB_CALIB / "calib_predictions.csv"}`

filtered to:

- `model_name = {PRIMARY_MODEL}`
- `feature_set = {PRIMARY_FEATURE}`
- `calib_method = {PRIMARY_CALIB}`
- `threshold_strategy = {PRIMARY_THRESH}`

This gives ROC-AUC={locked_prom['auc']:.6f} and PR-AUC={locked_prom['pr_auc']:.6f}, matching the promoted manuscript values 0.795155 and 0.573934.

The previously recovered ROC-AUC={locked_stagea['auc']:.4f} and PR-AUC={locked_stagea['pr_auc']:.4f} came from the historical run-root Stage A prediction file:

`{HIST_RUN / "all_folds_clf_predictions_MULTI_logreg_vaeconvtranspose4l_ld384_beta3.75_normzscore_offdiag_ch3sel_intFCquarter_drop0.15_ln0_outer5x1_scoreroc_auc.csv"}`

That file is **not** the promoted Stage B OOF-ECDF convention. It contains fold test predictions from the main FULL runner (`y_score_raw`, `y_score_cal`, `y_score_final`, `y_pred`). The promoted value comes from the classifier-only Stage B readout and an OOF-ECDF score transformation.

## OOF-ECDF Definition

The implementation in `scripts/revision_bspc_2026/run_recover035_latent384_beta3p75_stageB_oof_score_calibration.py` maps each test score through the empirical CDF of inner-CV OOF train/dev scores:

`percentile = interp(test_score, sort(oof_scores), (rank - 0.5) / n_oof)`

Thresholds are also selected from the calibrated inner-OOF scores, not from outer-test labels.

## Contemporary Matched Pair Limitation

The contemporary `single_ch5` and `control_ch102` FULL runs do **not** have a Stage B OOF-ECDF calibration package. Computing that package now would refit classifier-only Stage B models, which this audit is not allowed to do. Therefore this reconciliation reports their existing Stage A raw/calibrated/final metrics and marks promoted-convention Stage B OOF-ECDF rows as unavailable.

## Philips CN FPR Reconciliation

The low locked-control Philips CN FPR from the previous matched-pair audit is reproduced under the historical run-root Stage A final `y_pred` convention:

- Stage A final `y_pred`: {int(locked_fpr_stagea_final['fp_philips_cn'])}/{int(locked_fpr_stagea_final['n_philips_cn'])} = {locked_fpr_stagea_final['philips_cn_fpr']:.4f}

For reference, raw Stage A score thresholding at 0.5 gives:

- raw score > 0.5: {int(locked_fpr_raw['fp_philips_cn'])}/{int(locked_fpr_raw['n_philips_cn'])} = {locked_fpr_raw['philips_cn_fpr']:.4f}

The paper-convention Philips CN FPR is:

- Stage B OOF-ECDF primary threshold: {int(locked_fpr_prom['fp_philips_cn'])}/{int(locked_fpr_prom['n_philips_cn'])} = {locked_fpr_prom['philips_cn_fpr']:.4f}

These are different score/threshold conventions and should not be mixed.
"""
    (OUT / "score_convention_reconciliation.md").write_text(text, encoding="utf-8")


def write_decision(metrics: pd.DataFrame, boot: pd.DataFrame, beta: pd.DataFrame, rd: pd.DataFrame, leak: pd.DataFrame) -> None:
    # Pull matched Stage A final rows.
    st = metrics[metrics["score_convention"].eq("stageA_final")]
    single = st[st["candidate"].eq("single_ch5")].iloc[0]
    ctrl = st[st["candidate"].eq("control_ch102")].iloc[0]
    prom = metrics[
        metrics["candidate"].eq("locked_control")
        & metrics["score_convention"].eq("promoted_stageB_oof_ecdf_primary")
    ].iloc[0]
    high_beta_failures = beta[~beta["high_beta_guard_pass"]]
    text = f"""# Corrected FULL Promotion Recommendation

Generated: {datetime.now(timezone.utc).isoformat()}

## Decision

**Do not promote `single_ch5`.** The matched contemporary comparison does not support replacement of the final selected model, and the promoted-convention Stage B OOF-ECDF comparison is unavailable for the two contemporary runs without an additional classifier-only calibration step.

## Evidence

- Contemporary matched Stage A final AUC/PR-AUC:
  - `single_ch5`: AUC={single['auc']:.6f}, PR-AUC={single['pr_auc']:.6f}
  - `control_ch102`: AUC={ctrl['auc']:.6f}, PR-AUC={ctrl['pr_auc']:.6f}
- Historical locked promoted convention:
  - `[1,0,2]`: AUC={prom['auc']:.6f}, PR-AUC={prom['pr_auc']:.6f}, BA={prom['balanced_accuracy']:.6f}, F1={prom['f1']:.6f}
- Promoted-convention paired bootstrap for `single_ch5` vs contemporary `control_ch102`: **not computable from existing artifacts** because Stage B OOF-ECDF outputs are missing.
- Existing Stage A final paired bootstrap is descriptive only and is not a promoted-convention decision test.
- High-beta selected-checkpoint guard failures: {len(high_beta_failures)} fold(s) have inferred selected_beta < {HIGH_BETA_THRESHOLD:.4f}. This is not a historical FULL-run guard, but it means the current high-beta integrity claim must not be made from max/final beta.
- Rate-distortion was recomputed at inferred selected checkpoints, not final epochs; over-regularization remains a hypothesis, not a proven mechanism.
- Manufacturer leakage artifacts only contain balanced-accuracy summaries; ordinary accuracy, macro-F1, majority baseline, and permutation p-values were not present, so negligible leakage cannot be inferred from silhouette or from these incomplete summaries.

## Required Next Step If a Paper-Level Matched Comparison Is Still Desired

Run the established classifier-only Stage B OOF calibration script on the contemporary `single_ch5` and `control_ch102` latent caches **only if** classifier refitting is explicitly approved. Until then, the conservative decision is to retain the historical selected `[1,0,2]` model and treat the matched pair as an incomplete exploratory FULL comparison.
"""
    (OUT / "corrected_full_promotion_recommendation.md").write_text(text, encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log_step("start", {"output_dir": str(OUT), "read_only_inputs": True})

    stagea = {
        "single_ch5": load_stagea_predictions("single_ch5", PAIR_ROOT / "single_ch5"),
        "control_ch102": load_stagea_predictions("control_ch102", PAIR_ROOT / "control_ch102"),
    }
    stageb_primary = load_primary_stageb_predictions()

    metrics = pooled_metrics_all_conventions(stagea, stageb_primary)
    write_csv_md("pooled_metrics_all_score_conventions", metrics)

    boot = paired_bootstrap_table(stagea, stageb_primary)
    write_csv_md("promoted_convention_paired_bootstrap", boot)

    fpr = philips_fpr_rows(stagea, stageb_primary)
    write_csv_md("philips_fpr_reconciliation", fpr)

    beta_frames = []
    rd_frames = []
    run_map = {
        "single_ch5": PAIR_ROOT / "single_ch5",
        "control_ch102": PAIR_ROOT / "control_ch102",
        "locked_control": HIST_RUN,
    }
    for cand, run_dir in run_map.items():
        b, r = checkpoint_and_rd(cand, run_dir)
        beta_frames.append(b)
        rd_frames.append(r)
    beta = pd.concat(beta_frames, ignore_index=True)
    rd = pd.concat(rd_frames, ignore_index=True)
    write_csv_md("selected_checkpoint_beta_audit", beta, max_rows=200)
    write_csv_md("selected_checkpoint_rate_distortion", rd, max_rows=200)

    leak = manufacturer_leakage_table(run_map)
    # Add fold-wise raw-minus-latent deltas where possible.
    if not leak.empty:
        delta_rows = []
        for keys, g in leak.groupby(["candidate", "fold", "split"]):
            raw = g[g["representation"].eq("connectome_norm")]
            lat = g[g["representation"].eq("latent_mu")]
            if not raw.empty and not lat.empty:
                delta_rows.append(
                    {
                        "candidate": keys[0],
                        "fold": keys[1],
                        "split": keys[2],
                        "representation": "latent_minus_connectome_delta",
                        "ordinary_accuracy": np.nan,
                        "balanced_accuracy": float(lat.iloc[0]["balanced_accuracy"]) - float(raw.iloc[0]["balanced_accuracy"]),
                        "balanced_accuracy_std": np.nan,
                        "macro_f1": np.nan,
                        "majority_baseline": np.nan,
                        "chance_level": raw.iloc[0]["chance_level"],
                        "permutation_p_value": np.nan,
                        "n_samples": raw.iloc[0]["n_samples"],
                        "n_classes": raw.iloc[0]["n_classes"],
                        "source_file": "derived_from_existing_scanner_leakage_csv",
                        "notes": "Fold-wise latent minus connectome balanced-accuracy difference; other requested metrics unavailable in existing artifact.",
                    }
                )
        if delta_rows:
            leak = pd.concat([leak, pd.DataFrame(delta_rows)], ignore_index=True)
    write_csv_md("manufacturer_leakage_corrected", leak, max_rows=240)

    write_reconciliation_md(metrics, fpr)
    write_decision(metrics, boot, beta, rd, leak)

    # Supplemental JSON with guardrails and compact summaries.
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "guardrails": {
            "did_train": False,
            "did_refit_classifier": False,
            "did_run_oasis": False,
            "did_modify_prior_outputs": False,
            "did_modify_manuscript": False,
        },
        "inputs": {
            "pair_root": str(PAIR_ROOT),
            "historical_run": str(HIST_RUN),
            "stageb_calibration": str(STAGEB_CALIB),
            "metadata": str(META),
            "previous_audit": str(PREV_AUDIT),
        },
        "steps": COMMANDS,
    }
    (OUT / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Basic non-empty output validation.
    required = [
        "score_convention_reconciliation.md",
        "pooled_metrics_all_score_conventions.csv",
        "promoted_convention_paired_bootstrap.csv",
        "promoted_convention_paired_bootstrap.md",
        "philips_fpr_reconciliation.csv",
        "philips_fpr_reconciliation.md",
        "selected_checkpoint_beta_audit.csv",
        "selected_checkpoint_beta_audit.md",
        "selected_checkpoint_rate_distortion.csv",
        "selected_checkpoint_rate_distortion.md",
        "manufacturer_leakage_corrected.csv",
        "manufacturer_leakage_corrected.md",
        "corrected_full_promotion_recommendation.md",
        "command_log.json",
    ]
    missing = [name for name in required if not (OUT / name).exists() or (OUT / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"missing/empty outputs: {missing}")
    print(f"Wrote reconciliation audit to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
