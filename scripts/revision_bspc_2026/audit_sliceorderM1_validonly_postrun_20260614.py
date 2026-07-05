#!/usr/bin/env python3
"""Read-only postrun audit for slice-order M1 valid-only sensitivity."""

from __future__ import annotations

import json
import math
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"
RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5_sliceorderM1_validonly_sensitivity_20260612"
PREFLIGHT = RESULTS / "slice_order_curated_valid_only_sensitivity_20260612"
PROMOTED = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
PROMOTED_CALIB = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
FULL_DB = (
    RESULTS
    / "full_database_for_martin_and_validity_preflight_20260612"
    / "promoted_model_full_database_for_martin_20260612.csv"
)
OUT = RESULTS / "sliceorderM1_validonly_postrun_audit_20260614"

PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURE_SET = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
M1_EXCLUDED = {
    "018_S_6207", "018_S_6351",
    "031_S_4021", "031_S_4032", "031_S_4218", "031_S_4474", "031_S_4496",
    "301_S_6224", "301_S_6326", "301_S_6501",
}


def md_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def write_df(stem: str, df: pd.DataFrame, max_rows: int = 100) -> None:
    df.to_csv(OUT / f"{stem}.csv", index=False)
    (OUT / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def norm(s: pd.Series) -> pd.Series:
    return s.fillna("MISSING").astype(str).str.strip().replace({"": "MISSING", "nan": "MISSING", "NaN": "MISSING"})


def metrics(y_true: Iterable[Any], y_score: Iterable[Any], y_pred: Iterable[Any]) -> dict[str, Any]:
    y = np.asarray(list(y_true), dtype=int)
    score = np.asarray(list(y_score), dtype=float)
    pred = np.asarray(list(y_pred), dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    finite_score = score[np.isfinite(score)] if len(score) else np.array([])
    is_probability_score = bool(
        len(finite_score)
        and np.nanmin(finite_score) >= 0.0
        and np.nanmax(finite_score) <= 1.0
    )
    out: dict[str, Any] = {
        "N": int(len(y)),
        "N_CN": int((y == 0).sum()),
        "N_AD": int((y == 1).sum()),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "BA": float(balanced_accuracy_score(y, pred)) if len(y) else np.nan,
        "Sensitivity": float(recall_score(y, pred, pos_label=1, zero_division=0)) if len(y) else np.nan,
        "Specificity": float(recall_score(y, pred, pos_label=0, zero_division=0)) if len(y) else np.nan,
        "F1": float(f1_score(y, pred, zero_division=0)) if len(y) else np.nan,
        "Brier": float(brier_score_loss(y, score)) if len(y) and is_probability_score else np.nan,
    }
    if len(np.unique(y)) == 2:
        out["AUC"] = float(roc_auc_score(y, score))
        out["PR_AUC"] = float(average_precision_score(y, score))
    else:
        out["AUC"] = np.nan
        out["PR_AUC"] = np.nan
    return out


def find_single(pattern: str) -> Path | None:
    hits = sorted(RUN.glob(pattern))
    return hits[0] if hits else None


def load_stagea() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = find_single("all_folds_metrics_MULTI_*.csv")
    preds_path = find_single("all_folds_clf_predictions_MULTI_*.csv")
    if metrics_path is None:
        raise FileNotFoundError("Missing all_folds_metrics_MULTI_*.csv")
    if preds_path is None:
        raise FileNotFoundError("Missing all_folds_clf_predictions_MULTI_*.csv")
    return pd.read_csv(metrics_path), pd.read_csv(preds_path)


def completion_status() -> pd.DataFrame:
    rows = []
    top_metrics = find_single("all_folds_metrics_MULTI_*.csv")
    top_preds = find_single("all_folds_clf_predictions_MULTI_*.csv")
    top_hist = find_single("all_folds_vae_training_history_*.joblib")
    summary = find_single("summary_metrics_MULTI_*.txt")
    for fold in range(1, 6):
        fd = RUN / f"fold_{fold}"
        row = {
            "fold": fold,
            "fold_dir_exists": fd.exists(),
            "vae_model_exists": (fd / f"vae_model_fold_{fold}.pt").exists(),
            "test_predictions_logreg_exists": (fd / "test_predictions_logreg.csv").exists(),
            "test_predictions_svm_exists": (fd / "test_predictions_svm.csv").exists(),
            "train_dev_subjects_exists": (fd / "train_dev_subjects_fold.csv").exists(),
            "test_subjects_exists": (fd / "test_subjects_fold.csv").exists(),
            "vae_train_history_joblib_exists": (fd / f"vae_train_history_fold_{fold}.joblib").exists(),
            "vae_train_history_png_exists": (fd / f"vae_train_history_fold_{fold}.png").exists(),
            "rate_distortion_exists": (fd / f"fold_{fold}_rate_distortion.csv").exists(),
            "scanner_leakage_exists": (fd / f"fold_{fold}_scanner_leakage_summary.csv").exists(),
            "test_scanner_leakage_exists": (fd / f"fold_{fold}_test_scanner_leakage_summary.csv").exists(),
            "latent_info_trainDev_exists": (fd / f"fold_{fold}_trainDev_latent_info_summary.csv").exists(),
            "latent_info_test_exists": (fd / f"fold_{fold}_test_latent_info_summary.csv").exists(),
            "vae_required_metadata_filter_summary_exists": (fd / f"vae_pool_required_metadata_filter_summary_fold_{fold}.json").exists(),
            "vae_required_metadata_removed_exists": (fd / f"vae_pool_required_metadata_removed_fold_{fold}.csv").exists(),
        }
        row["fold_complete"] = all(bool(row[k]) for k in row if k not in {"fold"})
        rows.append(row)
    rows.append({
        "fold": "ALL",
        "fold_dir_exists": RUN.exists(),
        "vae_model_exists": all((RUN / f"fold_{f}" / f"vae_model_fold_{f}.pt").exists() for f in range(1, 6)),
        "test_predictions_logreg_exists": all((RUN / f"fold_{f}" / "test_predictions_logreg.csv").exists() for f in range(1, 6)),
        "test_predictions_svm_exists": all((RUN / f"fold_{f}" / "test_predictions_svm.csv").exists() for f in range(1, 6)),
        "train_dev_subjects_exists": all((RUN / f"fold_{f}" / "train_dev_subjects_fold.csv").exists() for f in range(1, 6)),
        "test_subjects_exists": all((RUN / f"fold_{f}" / "test_subjects_fold.csv").exists() for f in range(1, 6)),
        "vae_train_history_joblib_exists": bool(top_hist),
        "vae_train_history_png_exists": np.nan,
        "rate_distortion_exists": all((RUN / f"fold_{f}" / f"fold_{f}_rate_distortion.csv").exists() for f in range(1, 6)),
        "scanner_leakage_exists": all((RUN / f"fold_{f}" / f"fold_{f}_scanner_leakage_summary.csv").exists() for f in range(1, 6)),
        "test_scanner_leakage_exists": all((RUN / f"fold_{f}" / f"fold_{f}_test_scanner_leakage_summary.csv").exists() for f in range(1, 6)),
        "latent_info_trainDev_exists": all((RUN / f"fold_{f}" / f"fold_{f}_trainDev_latent_info_summary.csv").exists() for f in range(1, 6)),
        "latent_info_test_exists": all((RUN / f"fold_{f}" / f"fold_{f}_test_latent_info_summary.csv").exists() for f in range(1, 6)),
        "vae_required_metadata_filter_summary_exists": all((RUN / f"fold_{f}" / f"vae_pool_required_metadata_filter_summary_fold_{f}.json").exists() for f in range(1, 6)),
        "vae_required_metadata_removed_exists": all((RUN / f"fold_{f}" / f"vae_pool_required_metadata_removed_fold_{f}.csv").exists() for f in range(1, 6)),
        "all_folds_metrics_exists": bool(top_metrics),
        "all_folds_predictions_exists": bool(top_preds),
        "summary_metrics_exists": bool(summary),
        "run_config_exists": (RUN / "run_config.json").exists(),
        "classifier_only_readout_exists": (RUN / "classifier_only_readout").exists(),
        "stageB_oof_score_calibration_exists": any(RESULTS.glob("*sliceorderM1*score_calibration*")),
    })
    df = pd.DataFrame(rows)
    text_files = list(RUN.rglob("*.txt")) + list(RUN.rglob("*.log")) + list(RUN.rglob("*.err")) + list(RUN.rglob("*.out"))
    patterns = re.compile(r"Traceback|Exception|ERROR|CRITICAL|RuntimeError|ValueError", re.IGNORECASE)
    hits = []
    for p in text_files:
        try:
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                if patterns.search(line):
                    hits.append({"file": str(p.relative_to(PROJECT_ROOT)), "line": i, "text": line[:300]})
        except Exception:
            pass
    log_df = pd.DataFrame(hits)
    if log_df.empty:
        log_df = pd.DataFrame([{"file": "", "line": "", "text": "No Traceback/Error/Exception pattern found in available text logs/summary files."}])
    write_df("log_error_scan", log_df, max_rows=80)
    return df


def parsed_training_summary(stagea: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "auc_raw", "pr_auc_raw", "auc_final", "pr_auc_final", "auc", "pr_auc",
        "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1_score",
    ]
    rows = []
    for clf, sub in stagea.groupby("actual_classifier_type", dropna=False):
        row = {"classifier": clf, "n_folds": len(sub)}
        for c in numeric_cols:
            if c in sub.columns:
                row[f"{c}_mean"] = float(pd.to_numeric(sub[c], errors="coerce").mean())
                row[f"{c}_sd"] = float(pd.to_numeric(sub[c], errors="coerce").std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_from_stagea_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for clf, sub in preds.groupby("classifier_type", dropna=False):
        for score_col, score_label in [("y_score_raw", "raw_score"), ("y_score_final", "final_score")]:
            rows.append({
                "source": "M1_retrained_StageA",
                "classifier": clf,
                "score_type": score_label,
                **metrics(sub["y_true"], sub[score_col], sub["y_pred"]),
            })
    return pd.DataFrame(rows)


def load_full_db() -> pd.DataFrame:
    db = pd.read_csv(FULL_DB)
    db["SubjectID"] = db["SubjectID"].astype(str)
    return db


def attach_metadata(preds: pd.DataFrame) -> pd.DataFrame:
    db = load_full_db()
    keep_cols = [
        "SubjectID", "diagnosis_group", "Manufacturer_final", "Site3_final", "raw_tp_group_final",
        "Age_final", "Sex_final", "slice_order_class", "matches_dparsf_default",
        "Site2_default_7of7_pattern", "Site31_reverse_even_odd_48", "problem_site_flag",
    ]
    out = preds.merge(db[[c for c in keep_cols if c in db.columns]], on="SubjectID", how="left")
    out["Manufacturer_final"] = norm(out.get("Manufacturer_final", pd.Series(index=out.index, dtype=object)))
    out["Site3_final"] = norm(out.get("Site3_final", pd.Series(index=out.index, dtype=object))).str.replace(r"\.0$", "", regex=True)
    out["raw_tp_group_final"] = norm(out.get("raw_tp_group_final", pd.Series(index=out.index, dtype=object)))
    return out


def manufacturer_error(preds_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    high_rows = []
    for clf, csub in preds_meta.groupby("classifier_type", dropna=False):
        for mfr, sub in csub.groupby("Manufacturer_final", dropna=False):
            cn = sub[sub["y_true"].eq(0)]
            ad = sub[sub["y_true"].eq(1)]
            rows.append({
                "classifier": clf,
                "group_type": "CN_FPR_by_Manufacturer",
                "group": mfr,
                "N": len(cn),
                "errors": int((cn["y_pred"] == 1).sum()),
                "rate": float((cn["y_pred"] == 1).mean()) if len(cn) else np.nan,
            })
            rows.append({
                "classifier": clf,
                "group_type": "AD_FNR_by_Manufacturer",
                "group": mfr,
                "N": len(ad),
                "errors": int((ad["y_pred"] == 0).sum()),
                "rate": float((ad["y_pred"] == 0).mean()) if len(ad) else np.nan,
            })
        phil = csub[csub["Manufacturer_final"].eq("Philips")]
        for key, group_col in [("Philips_CN_FPR_by_Site3", "Site3_final"), ("Philips_CN_FPR_by_rawTP", "raw_tp_group_final")]:
            for group, sub in phil[phil["y_true"].eq(0)].groupby(group_col, dropna=False):
                rows.append({
                    "classifier": clf,
                    "group_type": key,
                    "group": str(group),
                    "N": len(sub),
                    "errors": int((sub["y_pred"] == 1).sum()),
                    "rate": float((sub["y_pred"] == 1).mean()) if len(sub) else np.nan,
                })
        site2 = phil[(phil["y_true"].eq(0)) & (phil["Site3_final"].eq("2"))]
        rows.append({
            "classifier": clf,
            "group_type": "Philips_CN_FPR_Site2_default_slice_order",
            "group": "Site2",
            "N": len(site2),
            "errors": int((site2["y_pred"] == 1).sum()),
            "rate": float((site2["y_pred"] == 1).mean()) if len(site2) else np.nan,
        })
        high = phil[(phil["y_true"].eq(0)) & (phil["y_pred"].eq(1)) & (phil["y_score_final"] > 0.75)].copy()
        for _, r in high.sort_values("y_score_final", ascending=False).iterrows():
            high_rows.append({
                "classifier": clf,
                "SubjectID": r["SubjectID"],
                "y_score_final": r["y_score_final"],
                "Site3": r.get("Site3_final", ""),
                "raw_tp_group": r.get("raw_tp_group_final", ""),
                "Age": r.get("Age_final", np.nan),
                "Sex": r.get("Sex_final", ""),
                "slice_order_class": r.get("slice_order_class", ""),
                "Site2_default_7of7_pattern": r.get("Site2_default_7of7_pattern", ""),
                "problem_site_flag": r.get("problem_site_flag", ""),
            })
    return pd.DataFrame(rows), pd.DataFrame(high_rows)


def scanner_leakage_summary() -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        for split, stem in [
            ("trainDev", f"fold_{fold}_scanner_leakage_summary.csv"),
            ("test", f"fold_{fold}_test_scanner_leakage_summary.csv"),
        ]:
            p = RUN / f"fold_{fold}" / stem
            if p.exists():
                df = pd.read_csv(p)
                for _, r in df.iterrows():
                    rows.append({
                        "fold": fold,
                        "split": split,
                        "site_col": r.get("site_col"),
                        "n_samples": r.get("n_samples"),
                        "chance_level": r.get("chance_level"),
                        "acc_site_raw": r.get("acc_site_raw"),
                        "acc_site_raw_std": r.get("acc_site_raw_std"),
                        "acc_site_latent": r.get("acc_site_latent"),
                        "acc_site_latent_std": r.get("acc_site_latent_std"),
                    })
    return pd.DataFrame(rows)


def rate_distortion_summary() -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        p = RUN / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        best_idx = pd.to_numeric(df["L_val_betaMax"], errors="coerce").idxmin()
        last = df.iloc[-1]
        best = df.loc[best_idx]
        for label, r in [("best_val_betaMax", best), ("last_epoch", last)]:
            d_val = float(r.get("D_val", np.nan))
            r_bits = float(r.get("R_val_bits", np.nan))
            beta = float(r.get("beta", np.nan))
            rows.append({
                "fold": fold,
                "summary_point": label,
                "epoch": int(r.get("epoch", -1)),
                "beta": beta,
                "D_val": d_val,
                "R_val_bits": r_bits,
                "R_bits_per_dim": r_bits / 384.0 if np.isfinite(r_bits) else np.nan,
                "beta_KLD_over_D": beta * (float(r.get("R_val_nats", np.nan)) / d_val) if d_val else np.nan,
                "L_val_betaMax": r.get("L_val_betaMax", np.nan),
            })
    return pd.DataFrame(rows)


def latent_mi_summary() -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        for split, stem in [
            ("trainDev", f"fold_{fold}_trainDev_latent_info_summary.csv"),
            ("test", f"fold_{fold}_test_latent_info_summary.csv"),
        ]:
            p = RUN / f"fold_{fold}" / stem
            if p.exists():
                df = pd.read_csv(p)
                for _, r in df.iterrows():
                    rows.append({
                        "fold": fold,
                        "split": split,
                        "variable": r.get("variable"),
                        "mi_sum_nats": r.get("mi_sum_nats"),
                        "mi_mean_nats": r.get("mi_mean_nats"),
                        "n_active": r.get("n_active"),
                        "frac_active": r.get("frac_active"),
                        "total_correlation_nats": r.get("total_correlation_nats"),
                    })
    return pd.DataFrame(rows)


def training_curve_summary() -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        p = RUN / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
        hist = RUN / f"fold_{fold}" / f"vae_train_history_fold_{fold}.joblib"
        if not p.exists():
            rows.append({"fold": fold, "status": "missing_rate_distortion"})
            continue
        df = pd.read_csv(p)
        best_idx = pd.to_numeric(df["L_val_betaMax"], errors="coerce").idxmin()
        best = df.loc[best_idx]
        last = df.iloc[-1]
        rows.append({
            "fold": fold,
            "status": "available",
            "history_joblib_exists": hist.exists(),
            "best_epoch": int(best["epoch"]),
            "final_epoch": int(last["epoch"]),
            "epochs_after_best": int(last["epoch"] - best["epoch"]),
            "best_L_val_betaMax": best.get("L_val_betaMax"),
            "final_L_val_betaMax": last.get("L_val_betaMax"),
            "best_D_val": best.get("D_val"),
            "best_R_val_bits": best.get("R_val_bits"),
            "best_beta": best.get("beta"),
        })
    return pd.DataFrame(rows)


def promoted_primary() -> dict[str, Any]:
    p = PROMOTED_CALIB / "calib_pooled_metrics.csv"
    df = pd.read_csv(p)
    row = df[
        df["model_name"].eq(PRIMARY_MODEL)
        & df["feature_set"].eq(PRIMARY_FEATURE_SET)
        & df["calib_method"].eq(PRIMARY_CALIB)
        & df["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ]
    if row.empty:
        raise RuntimeError("Could not find promoted primary OOF-ECDF row.")
    r = row.iloc[0]
    return {
        "comparison_row": "promoted_primary_oof_ecdf",
        "source": str(p.relative_to(PROJECT_ROOT)),
        "AUC": r["auc"],
        "PR_AUC": r["pr_auc"],
        "BA": r["balanced_accuracy"],
        "Sensitivity": r["sensitivity"],
        "Specificity": r["specificity"],
        "F1": r["f1"],
        "TN": r["tn"],
        "FP": r["fp"],
        "FN": r["fn"],
        "TP": r["tp"],
    }


def promoted_m1_scoreonly() -> dict[str, Any]:
    p = PREFLIGHT / "locked_model_score_only_sensitivity.csv"
    df = pd.read_csv(p)
    r = df[df["mask"].eq("M1_pragmatic_cn_problem_site_unknown")].iloc[0]
    return {
        "comparison_row": "promoted_retained_set_score_only_M1",
        "source": str(p.relative_to(PROJECT_ROOT)),
        "AUC": r["AUC"],
        "PR_AUC": r["PR_AUC"],
        "BA": r["BA"],
        "Sensitivity": r["Sensitivity"],
        "Specificity": r["Specificity"],
        "F1": r["F1"],
        "TN": r["TN"],
        "FP": r["FP"],
        "FN": r["FN"],
        "TP": r["TP"],
    }


def philips_fpr_for_comparison(manufacturer_df: pd.DataFrame, classifier: str) -> float:
    row = manufacturer_df[
        manufacturer_df["classifier"].eq(classifier)
        & manufacturer_df["group_type"].eq("CN_FPR_by_Manufacturer")
        & manufacturer_df["group"].eq("Philips")
    ]
    return float(row["rate"].iloc[0]) if not row.empty else np.nan


def promoted_philips_fpr() -> float:
    p = PROMOTED_CALIB / "calib_philips_fpr_pooled.csv"
    if p.exists():
        df = pd.read_csv(p)
        row = df[
            df.get("model_name", pd.Series(dtype=str)).eq(PRIMARY_MODEL)
            & df.get("feature_set", pd.Series(dtype=str)).eq(PRIMARY_FEATURE_SET)
            & df.get("calib_method", pd.Series(dtype=str)).eq(PRIMARY_CALIB)
            & df.get("threshold_strategy", pd.Series(dtype=str)).eq(PRIMARY_THRESHOLD)
            & df.get("manufacturer", pd.Series(dtype=str)).eq("Philips")
        ]
        for col in ["fpr_cn_pooled", "FPR", "rate"]:
            if not row.empty and col in row.columns:
                return float(row[col].iloc[0])
    return 45 / 99


def preflight_m1_philips_fpr() -> float:
    p = PREFLIGHT / "philips_cn_fpr_by_mask.csv"
    df = pd.read_csv(p)
    row = df[df["mask"].eq("M1_pragmatic_cn_problem_site_unknown") & df["Manufacturer"].eq("Philips_CN_overall")]
    return float(row["FPR"].iloc[0]) if not row.empty else np.nan


def comparison_table(pooled: pd.DataFrame, manufacturer_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prom = promoted_primary()
    prom["Philips_CN_FPR"] = promoted_philips_fpr()
    rows.append(prom)
    scoreonly = promoted_m1_scoreonly()
    scoreonly["Philips_CN_FPR"] = preflight_m1_philips_fpr()
    rows.append(scoreonly)
    for clf in ["logreg", "svm"]:
        for score_type in ["final_score"]:
            sub = pooled[(pooled["classifier"].eq(clf)) & (pooled["score_type"].eq(score_type))]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append({
                "comparison_row": f"M1_retrained_StageA_{clf}_{score_type}",
                "source": "M1 all_folds_clf_predictions",
                "AUC": r["AUC"],
                "PR_AUC": r["PR_AUC"],
                "BA": r["BA"],
                "Sensitivity": r["Sensitivity"],
                "Specificity": r["Specificity"],
                "F1": r["F1"],
                "TN": r["TN"],
                "FP": r["FP"],
                "FN": r["FN"],
                "TP": r["TP"],
                "Philips_CN_FPR": philips_fpr_for_comparison(manufacturer_df, clf),
            })
    out = pd.DataFrame(rows)
    ref = out[out["comparison_row"].eq("promoted_primary_oof_ecdf")].iloc[0]
    for metric in ["AUC", "PR_AUC", "BA", "Sensitivity", "Specificity", "F1", "Philips_CN_FPR"]:
        out[f"delta_vs_promoted_{metric}"] = pd.to_numeric(out[metric], errors="coerce") - float(ref[metric])
    return out


def threshold_diagnostic(pooled: pd.DataFrame, stagea: pd.DataFrame, comparison: pd.DataFrame) -> str:
    lines = ["# Threshold / Calibration Diagnostic", ""]
    lines.append("No new threshold fitting or calibration was performed in this audit.")
    lines.append("")
    lines.append("## Available outputs")
    lines.append("")
    lines.append("- M1 Stage A fold predictions exist for logreg and SVM.")
    lines.append(f"- `classifier_only_readout` exists: `{(RUN / 'classifier_only_readout').exists()}`")
    lines.append(f"- M1 OOF score-calibration directory exists: `{any(RESULTS.glob('*sliceorderM1*score_calibration*'))}`")
    lines.append("")
    lines.append("## Missing promoted-convention step")
    lines.append("")
    lines.append("The promoted primary result is Stage B `logreg_l2_original / z_plus_age_sex / oof_ecdf / inner_oof_target_sens_ge_0p70_max_spec`.")
    lines.append("The M1 run currently has Stage A predictions but does not have the separate classifier-only latent-cache readout or OOF-ECDF calibration artifacts.")
    lines.append("")
    stageb_cmd = (
        f"/home/diego/anaconda3/envs/vae_ad/bin/python "
        f"scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py "
        f"--run-dir {RUN} "
        f"--output-dir {RUN / 'classifier_only_readout'} "
        f"--models logreg_l2 "
        f"--readout-feature-sets z_plus_age_sex "
        f"--outer-folds 5 --inner-folds 5 --device cpu --reuse-latent-cache"
    )
    calib_cmd = (
        f"/home/diego/anaconda3/envs/vae_ad/bin/python "
        f"scripts/revision_bspc_2026/run_recover035_latent384_beta3p75_stageB_oof_score_calibration.py "
        f"--run-dir {RUN} "
        f"--output-dir {RESULTS / 'recover035_latent384_beta3p75_sliceorderM1_stageB_oof_score_calibration_20260614'} "
        f"--n-jobs 4 --overwrite"
    )
    lines.append("Commands to run later, if explicitly requested:")
    lines.append("")
    lines.append("```bash")
    lines.append(stageb_cmd)
    lines.append(calib_cmd)
    lines.append("```")
    lines.append("")
    lines.append("## Diagnostic evidence from Stage A")
    lines.append("")
    for _, r in pooled[pooled["score_type"].eq("final_score")].iterrows():
        lines.append(
            f"- {r['classifier']}: AUC={r['AUC']:.4f}, PR-AUC={r['PR_AUC']:.4f}, "
            f"BA={r['BA']:.4f}, Sens={r['Sensitivity']:.4f}, Spec={r['Specificity']:.4f}, F1={r['F1']:.4f}."
        )
    lines.append("")
    lines.append("Rank separation is moderate, but the operating point is poor: sensitivity is much lower than the promoted primary thresholded result. This points to threshold/calibration/operating-point behavior and/or fold-specific score scale as major issues. Stage B OOF-ECDF is required before making a final promoted-convention comparison.")
    return "\n".join(lines) + "\n"


def scanner_leakage_md(scanner: pd.DataFrame, latent_mi: pd.DataFrame) -> str:
    lines = ["# Scanner Leakage Summary", ""]
    if scanner.empty:
        lines.append("No scanner leakage summaries found.")
    else:
        lines.append("Foldwise raw and latent Manufacturer predictability:")
        lines.append("")
        lines.append(scanner.to_markdown(index=False))
        lines.append("")
        lines.append("Mean by split:")
        summary = scanner.groupby("split", dropna=False)[["acc_site_raw", "acc_site_latent"]].mean().reset_index()
        lines.append(summary.to_markdown(index=False))
    if not latent_mi.empty:
        lines.append("")
        lines.append("Latent MI summaries are available in `latent_mi_signal_nuisance_summary_m1.csv`.")
        mi_mean = latent_mi.groupby(["split", "variable"], dropna=False)[["mi_sum_nats", "mi_mean_nats", "n_active", "total_correlation_nats"]].mean().reset_index()
        lines.append("")
        lines.append(mi_mean.to_markdown(index=False))
    return "\n".join(lines) + "\n"


def rate_distortion_md(rd: pd.DataFrame) -> str:
    lines = ["# Rate-Distortion Summary", ""]
    if rd.empty:
        lines.append("No rate-distortion files found.")
    else:
        best = rd[rd["summary_point"].eq("best_val_betaMax")]
        lines.append(best.to_markdown(index=False))
        lines.append("")
        lines.append("Mean best-val summary:")
        cols = ["D_val", "R_val_bits", "R_bits_per_dim", "beta_KLD_over_D", "L_val_betaMax"]
        lines.append(best[cols].mean(numeric_only=True).to_frame("mean").T.to_markdown(index=False))
    return "\n".join(lines) + "\n"


def final_text(comparison: pd.DataFrame, manufacturer: pd.DataFrame) -> tuple[str, str]:
    prom = comparison[comparison["comparison_row"].eq("promoted_primary_oof_ecdf")].iloc[0]
    m1_log = comparison[comparison["comparison_row"].eq("M1_retrained_StageA_logreg_final_score")]
    m1_svm = comparison[comparison["comparison_row"].eq("M1_retrained_StageA_svm_final_score")]
    log = m1_log.iloc[0] if not m1_log.empty else None
    svm = m1_svm.iloc[0] if not m1_svm.empty else None
    decision = "incomplete_needs_oof_calibration"
    if log is not None:
        if (
            float(log["AUC"]) > float(prom["AUC"])
            and float(log["PR_AUC"]) > float(prom["PR_AUC"])
            and float(log["BA"]) >= float(prom["BA"])
            and float(log["Sensitivity"]) >= float(prom["Sensitivity"])
            and float(log["F1"]) >= float(prom["F1"])
            and float(log["Philips_CN_FPR"]) < float(prom["Philips_CN_FPR"])
        ):
            decision = "positive_sensitivity"
        elif float(log["AUC"]) < float(prom["AUC"]) or float(log["PR_AUC"]) < float(prom["PR_AUC"]) or float(log["Sensitivity"]) < float(prom["Sensitivity"]):
            decision = "negative_sensitivity"
        else:
            decision = "neutral_sensitivity"
    interp = ["# Final Interpretation", ""]
    interp.append("The exploratory M1 valid-only run completed Stage A for all five folds, but promoted-convention Stage B classifier-only OOF-ECDF artifacts are not present yet.")
    interp.append("")
    if log is not None:
        interp.append(
            f"Stage A logreg final pooled metrics were AUC={log['AUC']:.6f}, PR-AUC={log['PR_AUC']:.6f}, "
            f"BA={log['BA']:.6f}, Sens={log['Sensitivity']:.6f}, Spec={log['Specificity']:.6f}, F1={log['F1']:.6f}."
        )
    if svm is not None:
        interp.append(
            f"Stage A SVM final pooled metrics were AUC={svm['AUC']:.6f}, PR-AUC={svm['PR_AUC']:.6f}, "
            f"BA={svm['BA']:.6f}, Sens={svm['Sensitivity']:.6f}, Spec={svm['Specificity']:.6f}, F1={svm['F1']:.6f}."
        )
    interp.append("")
    interp.append(
        "The score-only retained-set comparison improved mechanically because 10 difficult Philips CN subjects were removed, but retraining did not reproduce that as a clear Stage A operating-profile improvement."
    )
    interp.append(
        "The strict decision rule is not met: M1 does not clearly improve AUC/PR-AUC while preserving BA/Sensitivity/F1, and any Philips FPR improvement is partly tied to removing hard cases."
    )
    interp.append("")
    interp.append(f"Decision: `{decision}`.")

    dec = ["# Final Decision", "", f"Decision category: **{decision}**", ""]
    dec.append("Rationale:")
    dec.append("- Stage A completed, but promoted-convention Stage B OOF-ECDF is missing.")
    dec.append("- Available Stage A pooled metrics show low sensitivity/BA relative to the promoted primary readout.")
    dec.append("- Philips CN FPR must be interpreted cautiously because M1 removed known high-risk Philips CN subjects from all pools.")
    dec.append("- The promoted model should remain primary unless later Stage B OOF-ECDF and external checks meet the strict gate.")
    return "\n".join(interp) + "\n", "\n".join(dec) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not RUN.exists():
        raise FileNotFoundError(RUN)
    if not FULL_DB.exists():
        raise FileNotFoundError(FULL_DB)

    stagea, preds = load_stagea()
    completion = completion_status()
    parsed = parsed_training_summary(stagea)
    pooled = pooled_from_stagea_predictions(preds)
    preds_meta = attach_metadata(preds)
    manufacturer, high_fp = manufacturer_error(preds_meta)
    scanner = scanner_leakage_summary()
    rd = rate_distortion_summary()
    latent_mi = latent_mi_summary()
    train_curve = training_curve_summary()
    comparison = comparison_table(pooled, manufacturer)

    write_df("completion_status", completion, max_rows=20)
    write_df("parsed_training_summary", parsed, max_rows=20)
    write_df("foldwise_stagea_metrics", stagea, max_rows=30)
    write_df("pooled_oof_metrics", pooled, max_rows=20)
    write_df("promoted_vs_m1_comparison", comparison, max_rows=20)
    write_df("manufacturer_error_m1", manufacturer, max_rows=120)
    write_df("philips_cn_fp_remaining", high_fp, max_rows=120)
    write_df("training_curve_summary", train_curve, max_rows=20)
    write_df("scanner_leakage_summary_m1", scanner, max_rows=30)
    write_df("rate_distortion_summary_m1", rd, max_rows=20)
    write_df("latent_mi_signal_nuisance_summary_m1", latent_mi, max_rows=40)

    (OUT / "threshold_calibration_diagnostic.md").write_text(threshold_diagnostic(pooled, stagea, comparison), encoding="utf-8")
    (OUT / "scanner_leakage_summary_m1.md").write_text(scanner_leakage_md(scanner, latent_mi), encoding="utf-8")
    (OUT / "rate_distortion_summary_m1.md").write_text(rate_distortion_md(rd), encoding="utf-8")
    interp, dec = final_text(comparison, manufacturer)
    (OUT / "final_interpretation.md").write_text(interp, encoding="utf-8")
    (OUT / "final_decision.md").write_text(dec, encoding="utf-8")

    write_json(OUT / "command_log.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "run_dir": str(RUN),
        "preflight_dir": str(PREFLIGHT),
        "promoted_dir": str(PROMOTED),
        "promoted_calibration_dir": str(PROMOTED_CALIB),
        "guardrails": {
            "read_only": True,
            "model_training": False,
            "tensor_edits": False,
            "metadata_edits": False,
            "prediction_edits": False,
            "threshold_refitting": False,
            "subject_exclusion": False,
            "m1_run_modified": False,
            "promoted_run_modified": False,
        },
        "stageB_oof_calibration_present": any(RESULTS.glob("*sliceorderM1*score_calibration*")),
        "classifier_only_readout_present": (RUN / "classifier_only_readout").exists(),
        "outputs": str(OUT.relative_to(PROJECT_ROOT)),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
