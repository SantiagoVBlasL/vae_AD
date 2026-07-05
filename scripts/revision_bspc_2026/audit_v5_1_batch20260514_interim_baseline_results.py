#!/usr/bin/env python3
"""Audit ADNI v5.1 batch20260514 interim baseline ML results.

This script is read-only with respect to model outputs, tensors and ledgers. It
only writes derived audit tables under results/revision_bspc_2026.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_RUN_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514_ch1_0_2_interim_baseline"
)
BIG_RUN_DIR = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
    "adni_v5_1_batch20260514_ch1_0_2_interim_baseline"
)
METADATA_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514_no_pybandpass/"
    "training_ready_metadata_v5_1_batch20260514_no_pybandpass.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514_interim_baseline_audit"
)

PAPER_ORIGINAL = {
    "version": "paper_original",
    "N_training_ready": 431,
    "AD": 95,
    "CN": 89,
    "MCI": 247,
    "CN_GE": 0,
}

COMPARISON_RUNS = [
    {
        "version": "v5_1_batch20260514_ch1_0_2",
        "metrics_dir": BIG_RUN_DIR,
        "metadata_path": METADATA_PATH,
        "channels": "[1,0,2]",
    },
    {
        "version": "v5_ch1_0_2",
        "metrics_dir": Path(
            "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
            "adni_v5_dparsf10000_no_pybandpass_ch1_0_2_baseline"
        ),
        "metadata_path": Path(
            "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
            "adni_expanded_v5_dparsf10000_no_pybandpass/"
            "training_ready_metadata_v5_dparsf10000_no_pybandpass.csv"
        ),
        "channels": "[1,0,2]",
    },
    {
        "version": "v5_ch4_1_0",
        "metrics_dir": Path(
            "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
            "adni_v5_dparsf10000_no_pybandpass_ch4_1_0_baseline"
        ),
        "metadata_path": Path(
            "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
            "adni_expanded_v5_dparsf10000_no_pybandpass/"
            "training_ready_metadata_v5_dparsf10000_no_pybandpass.csv"
        ),
        "channels": "[4,1,0]",
    },
    {
        "version": "v4_static3_beta25",
        "metrics_dir": Path(
            "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
            "adni_expanded_v4_beta25_static3"
        ),
        "metadata_path": PROJECT_ROOT
        / "data"
        / "revision_bspc_2026"
        / "adni_expanded_v4_all_available"
        / "subject_metadata_adni_expanded_v4_all_available.csv",
        "channels": "static3",
    },
    {
        "version": "v4_ch4_1_0_beta25",
        "metrics_dir": Path(
            "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
            "adni_expanded_v4_beta25_ch4_1_0"
        ),
        "metadata_path": PROJECT_ROOT
        / "data"
        / "revision_bspc_2026"
        / "adni_expanded_v4_all_available"
        / "subject_metadata_adni_expanded_v4_all_available.csv",
        "channels": "[4,1,0]",
    },
]


def resolve_run_dir() -> Path:
    if BIG_RUN_DIR.exists():
        return BIG_RUN_DIR
    if LOCAL_RUN_DIR.exists():
        return LOCAL_RUN_DIR
    raise FileNotFoundError(f"Run directory not found: {BIG_RUN_DIR} or {LOCAL_RUN_DIR}")


def first_file(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} under {path}")
    return matches[0]


def read_json_optional(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def normalize_classifier_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "actual_classifier_type" in out.columns and "classifier" not in out.columns:
        out["classifier"] = out["actual_classifier_type"]
    if "classifier_type" in out.columns and "classifier" not in out.columns:
        out["classifier"] = out["classifier_type"]
    return out


def normalize_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    defaults = {
        "Manufacturer": "UNKNOWN",
        "ResearchGroup_Mapped": "UNKNOWN",
        "Sex": "UNKNOWN",
        "source_batch": "UNKNOWN",
        "source_label": "UNKNOWN",
        "tensor_source": "UNKNOWN",
        "included_in_dataset_version": "UNKNOWN",
    }
    for col, value in defaults.items():
        if col not in out.columns:
            out[col] = value
        out[col] = out[col].fillna(value).astype(str)
    if "Age" not in out.columns:
        out["Age"] = np.nan
    return out


def load_predictions(run_dir: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    combined_matches = sorted(run_dir.glob("all_folds_clf_predictions_MULTI*.csv"))
    if combined_matches:
        pred = pd.read_csv(combined_matches[0])
    else:
        chunks: List[pd.DataFrame] = []
        for fold_dir in sorted(run_dir.glob("fold_*")):
            if not fold_dir.is_dir():
                continue
            fold = int(fold_dir.name.split("_")[-1])
            for path in sorted(fold_dir.glob("test_predictions_*.csv")):
                classifier = path.stem.replace("test_predictions_", "")
                chunk = pd.read_csv(path)
                chunk["fold"] = fold
                chunk["classifier"] = classifier
                chunks.append(chunk)
        if not chunks:
            raise FileNotFoundError(f"No prediction CSV files found under {run_dir}")
        pred = pd.concat(chunks, ignore_index=True)

    pred = normalize_classifier_col(pred)
    if "classifier" not in pred.columns:
        raise RuntimeError("Prediction table has no classifier column.")
    if "y_score_final" not in pred.columns:
        if "y_score_cal" in pred.columns:
            pred["y_score_final"] = pred["y_score_cal"]
        elif "y_score_raw" in pred.columns:
            pred["y_score_final"] = pred["y_score_raw"]
        else:
            raise RuntimeError("Prediction table has no usable score column.")
    if "y_pred" not in pred.columns:
        pred["y_pred"] = (pred["y_score_final"] >= 0.5).astype(int)

    keep_meta = [
        col
        for col in [
            "SubjectID",
            "ResearchGroup_Mapped",
            "Diagnosis",
            "Age",
            "Sex",
            "Manufacturer",
            "Site3",
            "ImageID",
            "Visit",
            "tensor_source",
            "source_label",
            "source_batch",
            "included_in_dataset_version",
        ]
        if col in metadata.columns
    ]
    pred = pred.merge(metadata[keep_meta], on="SubjectID", how="left", suffixes=("", "_meta"))
    pred["diagnosis_label"] = pred["ResearchGroup_Mapped"].fillna(
        pred["y_true"].map({0: "CN", 1: "AD"}).fillna("UNKNOWN")
    )
    pred["manufacturer_label"] = pred["Manufacturer"].fillna("UNKNOWN")
    pred["source_batch"] = pred.get("source_batch", pd.Series(index=pred.index, dtype=object)).fillna("UNKNOWN")
    pred["source_label"] = pred.get("source_label", pd.Series(index=pred.index, dtype=object)).fillna("UNKNOWN")
    pred["tensor_source"] = pred.get("tensor_source", pd.Series(index=pred.index, dtype=object)).fillna("UNKNOWN")
    pred["score_bin_decile"] = pd.cut(
        pred["y_score_final"],
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
        duplicates="drop",
    ).astype(str)
    pred["error_type"] = np.select(
        [
            (pred["y_true"].eq(1) & pred["y_pred"].eq(0)),
            (pred["y_true"].eq(0) & pred["y_pred"].eq(1)),
            (pred["y_true"].eq(1) & pred["y_pred"].eq(1)),
            (pred["y_true"].eq(0) & pred["y_pred"].eq(0)),
        ],
        ["false_negative", "false_positive", "true_positive", "true_negative"],
        default="unknown",
    )
    return pred


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=labels).ravel()
    out: Dict[str, float] = {
        "n": int(len(y_true_arr)),
        "n_cn": int((y_true_arr == 0).sum()),
        "n_ad": int((y_true_arr == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": safe_div(tp + tn, len(y_true_arr)),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "predicted_ad_rate": float(y_pred_arr.mean()),
    }
    if out["n_ad"] and out["n_cn"]:
        out["balanced_accuracy"] = float((out["sensitivity"] + out["specificity"]) / 2.0)
    elif out["n_ad"]:
        out["balanced_accuracy"] = out["sensitivity"]
    elif out["n_cn"]:
        out["balanced_accuracy"] = out["specificity"]
    else:
        out["balanced_accuracy"] = float("nan")
    if len(np.unique(y_true_arr)) == 2:
        out["auc"] = float(roc_auc_score(y_true_arr, y_score_arr))
        out["pr_auc"] = float(average_precision_score(y_true_arr, y_score_arr))
    else:
        out["auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def calibration_metrics(y_true: Sequence[int], y_score: Sequence[float], n_bins: int = 10) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    brier = float(brier_score_loss(y_true_arr, y_score_arr))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_score_arr, bins[1:-1], right=True)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        frac = float(mask.mean())
        ece += frac * abs(float(y_true_arr[mask].mean()) - float(y_score_arr[mask].mean()))
    return {"brier": brier, "ece_10bin": float(ece)}


def reliability_rows(pred: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for (fold, clf), sub in pred.groupby(["fold", "classifier"], dropna=False):
        scores = sub["y_score_final"].to_numpy(float)
        y = sub["y_true"].to_numpy(int)
        bin_ids = np.digitize(scores, bins[1:-1], right=True)
        for bin_id in range(n_bins):
            mask = bin_ids == bin_id
            rows.append(
                {
                    "fold": int(fold),
                    "classifier": clf,
                    "bin": bin_id,
                    "bin_left": bins[bin_id],
                    "bin_right": bins[bin_id + 1],
                    "n": int(mask.sum()),
                    "mean_score": float(scores[mask].mean()) if mask.any() else np.nan,
                    "observed_ad_rate": float(y[mask].mean()) if mask.any() else np.nan,
                    "calibration_gap_observed_minus_score": (
                        float(y[mask].mean() - scores[mask].mean()) if mask.any() else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_foldwise_metrics(metrics: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    metrics = normalize_classifier_col(metrics)
    rows: List[Dict[str, Any]] = []
    for _, row in metrics.iterrows():
        fold = int(row["fold"])
        clf = row["classifier"]
        sub = pred[pred["fold"].eq(fold) & pred["classifier"].eq(clf)]
        cal = calibration_metrics(sub["y_true"], sub["y_score_final"]) if len(sub) else {}
        out = {
            "fold": fold,
            "classifier": clf,
            "auc_raw": safe_float(row.get("auc_raw")),
            "pr_auc_raw": safe_float(row.get("pr_auc_raw")),
            "auc_final": safe_float(row.get("auc_final", row.get("auc"))),
            "pr_auc_final": safe_float(row.get("pr_auc_final", row.get("pr_auc"))),
            "accuracy": safe_float(row.get("accuracy")),
            "balanced_accuracy": safe_float(row.get("balanced_accuracy")),
            "sensitivity": safe_float(row.get("sensitivity")),
            "specificity": safe_float(row.get("specificity")),
            "f1": safe_float(row.get("f1_score", row.get("f1"))),
            "n_test": int(len(sub)),
            "n_ad": int(sub["y_true"].eq(1).sum()) if len(sub) else 0,
            "n_cn": int(sub["y_true"].eq(0).sum()) if len(sub) else 0,
            "brier": cal.get("brier", np.nan),
            "ece_10bin": cal.get("ece_10bin", np.nan),
            "did_calibrate": row.get("did_calibrate", np.nan),
        }
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["classifier", "fold"])


def make_confusions(pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: List[Dict[str, Any]] = []
    pooled_rows: List[Dict[str, Any]] = []
    for (fold, clf), sub in pred.groupby(["fold", "classifier"], dropna=False):
        row = {"fold": int(fold), "classifier": clf, "threshold": 0.5}
        row.update(binary_metrics(sub["y_true"], sub["y_score_final"], sub["y_pred"]))
        fold_rows.append(row)
    for clf, sub in pred.groupby("classifier", dropna=False):
        row = {"classifier": clf, "threshold": 0.5}
        row.update(binary_metrics(sub["y_true"], sub["y_score_final"], sub["y_pred"]))
        pooled_rows.append(row)
    return pd.DataFrame(fold_rows).sort_values(["classifier", "fold"]), pd.DataFrame(pooled_rows)


def summarize_scores(pred: pd.DataFrame) -> pd.DataFrame:
    groupings = [
        ("diagnosis", ["diagnosis_label"]),
        ("manufacturer", ["manufacturer_label"]),
        ("diagnosis_x_manufacturer", ["diagnosis_label", "manufacturer_label"]),
        ("source_batch", ["source_batch"]),
        ("source_label", ["source_label"]),
        ("tensor_source", ["tensor_source"]),
        ("fold", ["fold"]),
    ]
    rows: List[Dict[str, Any]] = []
    for clf, clf_df in pred.groupby("classifier", dropna=False):
        for grouping_name, cols in groupings:
            if any(col not in clf_df.columns for col in cols):
                continue
            for keys, sub in clf_df.groupby(cols, dropna=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                group_value = " | ".join(str(x) for x in keys)
                score = sub["y_score_final"]
                rows.append(
                    {
                        "classifier": clf,
                        "grouping": grouping_name,
                        "group_value": group_value,
                        "n": int(len(sub)),
                        "n_ad": int(sub["y_true"].eq(1).sum()),
                        "n_cn": int(sub["y_true"].eq(0).sum()),
                        "score_mean": float(score.mean()),
                        "score_std": float(score.std(ddof=1)) if len(score) > 1 else np.nan,
                        "score_median": float(score.median()),
                        "score_q25": float(score.quantile(0.25)),
                        "score_q75": float(score.quantile(0.75)),
                        "score_min": float(score.min()),
                        "score_max": float(score.max()),
                        "predicted_ad_rate": float(sub["y_pred"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def subgroup_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("manufacturer", ["manufacturer_label"]),
        ("diagnosis_x_manufacturer", ["diagnosis_label", "manufacturer_label"]),
        ("cn_by_manufacturer", ["manufacturer_label"]),
        ("ad_by_manufacturer", ["manufacturer_label"]),
    ]
    rows: List[Dict[str, Any]] = []
    for clf, clf_df in pred.groupby("classifier", dropna=False):
        for group_name, cols in specs:
            base = clf_df
            if group_name == "cn_by_manufacturer":
                base = clf_df[clf_df["y_true"].eq(0)]
            elif group_name == "ad_by_manufacturer":
                base = clf_df[clf_df["y_true"].eq(1)]
            for keys, sub in base.groupby(cols, dropna=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                row = {
                    "classifier": clf,
                    "grouping": group_name,
                    "group_value": " | ".join(str(x) for x in keys),
                    "threshold": 0.5,
                }
                row.update(binary_metrics(sub["y_true"], sub["y_score_final"], sub["y_pred"]))
                row["score_mean"] = float(sub["y_score_final"].mean())
                row["score_median"] = float(sub["y_score_final"].median())
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["classifier", "grouping", "group_value"])


def threshold_analysis(pred: pd.DataFrame) -> pd.DataFrame:
    thresholds = np.round(np.linspace(0.0, 1.0, 101), 2)
    rows: List[Dict[str, Any]] = []
    for clf, sub in pred.groupby("classifier", dropna=False):
        y = sub["y_true"].to_numpy(int)
        score = sub["y_score_final"].to_numpy(float)
        for thr in thresholds:
            yhat = (score >= thr).astype(int)
            m = binary_metrics(y, score, yhat)
            rows.append(
                {
                    "classifier": clf,
                    "scope": "pooled_test_folds",
                    "threshold": float(thr),
                    "selection_context": "exploratory_test_only_do_not_report_as_final",
                    "youden_j": m["sensitivity"] + m["specificity"] - 1.0,
                    **m,
                }
            )
    out = pd.DataFrame(rows)
    out["is_threshold_0_5"] = out["threshold"].eq(0.5)
    for metric in ["youden_j", "f1", "balanced_accuracy"]:
        rank_col = f"rank_desc_{metric}"
        out[rank_col] = out.groupby("classifier")[metric].rank(method="min", ascending=False)
        out[f"is_best_{metric}"] = out[rank_col].eq(1)
    return out


def fold5_error_audit(pred: pd.DataFrame) -> pd.DataFrame:
    fold5 = pred[pred["fold"].eq(5)].copy()
    rows: List[Dict[str, Any]] = []
    summary_cols = ["manufacturer_label", "Sex", "diagnosis_label", "source_batch", "source_label", "tensor_source"]
    for clf, clf_df in fold5.groupby("classifier", dropna=False):
        for col in summary_cols:
            if col not in clf_df.columns:
                continue
            for value, sub in clf_df.groupby(col, dropna=False):
                rows.append(
                    {
                        "row_type": "fold5_composition",
                        "classifier": clf,
                        "grouping": col,
                        "group_value": value,
                        "n": int(len(sub)),
                        "n_ad": int(sub["y_true"].eq(1).sum()),
                        "n_cn": int(sub["y_true"].eq(0).sum()),
                        "age_mean": float(sub["Age"].mean()) if "Age" in sub.columns else np.nan,
                        "predicted_ad_rate": float(sub["y_pred"].mean()),
                        "false_negative_count": int(sub["error_type"].eq("false_negative").sum()),
                        "false_positive_count": int(sub["error_type"].eq("false_positive").sum()),
                    }
                )
        for error_type in ["false_negative", "false_positive"]:
            err = clf_df[clf_df["error_type"].eq(error_type)].copy()
            for _, r in err.sort_values("y_score_final", ascending=(error_type == "false_negative")).iterrows():
                rows.append(
                    {
                        "row_type": "fold5_error_subject",
                        "classifier": clf,
                        "error_type": error_type,
                        "SubjectID": r.get("SubjectID"),
                        "y_true": int(r.get("y_true")),
                        "y_pred": int(r.get("y_pred")),
                        "y_score_final": float(r.get("y_score_final")),
                        "diagnosis_label": r.get("diagnosis_label"),
                        "Manufacturer": r.get("manufacturer_label"),
                        "Age": r.get("Age"),
                        "Sex": r.get("Sex"),
                        "source_batch": r.get("source_batch"),
                        "source_label": r.get("source_label"),
                        "tensor_source": r.get("tensor_source"),
                    }
                )
    return pd.DataFrame(rows)


def load_scanner_leakage(run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        fold = int(fold_dir.name.split("_")[-1])
        test_path = fold_dir / f"fold_{fold}_test_scanner_leakage_summary.csv"
        train_path = fold_dir / f"fold_{fold}_scanner_leakage_summary.csv"
        latent_path = fold_dir / "latent_qc_metrics.csv"
        row: Dict[str, Any] = {"fold": fold}
        if test_path.exists():
            t = pd.read_csv(test_path).iloc[0].to_dict()
            row.update({f"test_{k}": v for k, v in t.items() if k != "fold_tag"})
        if train_path.exists():
            t = pd.read_csv(train_path).iloc[0].to_dict()
            row.update({f"train_dev_{k}": v for k, v in t.items() if k != "fold_tag"})
        if latent_path.exists():
            t = pd.read_csv(latent_path).iloc[0].to_dict()
            row.update({f"latent_qc_{k}": v for k, v in t.items() if k != "fold"})
        rows.append(row)
    return pd.DataFrame(rows)


def scanner_vs_auc(foldwise: pd.DataFrame, leakage: pd.DataFrame) -> pd.DataFrame:
    merged = foldwise.merge(leakage, on="fold", how="left")
    merged.insert(0, "row_type", "fold")
    rows: List[pd.DataFrame] = [merged]
    corr_rows: List[Dict[str, Any]] = []
    candidate_x = [
        "test_acc_site_raw",
        "test_acc_site_latent",
        "train_dev_acc_site_raw",
        "train_dev_acc_site_latent",
        "latent_qc_acc_site_raw",
        "latent_qc_acc_site_latent",
    ]
    for clf, sub in merged.groupby("classifier", dropna=False):
        for x_col in candidate_x:
            if x_col not in sub.columns:
                continue
            for y_col in ["auc_final", "sensitivity", "specificity", "balanced_accuracy"]:
                valid = sub[[x_col, y_col]].dropna()
                corr = valid[x_col].corr(valid[y_col]) if len(valid) >= 3 else np.nan
                corr_rows.append(
                    {
                        "row_type": "correlation",
                        "classifier": clf,
                        "x": x_col,
                        "y": y_col,
                        "n": int(len(valid)),
                        "pearson_r": corr,
                    }
                )
    rows.append(pd.DataFrame(corr_rows))
    return pd.concat(rows, ignore_index=True, sort=False)


def metadata_counts(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"metadata_available": False}
    meta = normalize_metadata(pd.read_csv(path))
    out: Dict[str, Any] = {"metadata_available": True, "N_training_ready": int(len(meta))}
    diag = meta["ResearchGroup_Mapped"].fillna("UNKNOWN")
    man = meta["Manufacturer"].fillna("UNKNOWN")
    for label in ["AD", "CN", "MCI"]:
        out[label] = int(diag.eq(label).sum())
    out["CN_GE"] = int((diag.eq("CN") & man.eq("GE")).sum())
    out["CN_Siemens"] = int((diag.eq("CN") & man.eq("Siemens")).sum())
    out["CN_Philips"] = int((diag.eq("CN") & man.eq("Philips")).sum())
    out["AD_GE"] = int((diag.eq("AD") & man.eq("GE")).sum())
    return out


def compare_runs() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [dict(PAPER_ORIGINAL, run_available=False, metrics_available=False)]
    for spec in COMPARISON_RUNS:
        metrics_dir = Path(spec["metrics_dir"])
        row: Dict[str, Any] = {
            "version": spec["version"],
            "channels": spec["channels"],
            "run_available": metrics_dir.exists(),
            "metrics_available": False,
            "metrics_dir": str(metrics_dir),
        }
        row.update(metadata_counts(Path(spec["metadata_path"])))
        matches = sorted(metrics_dir.glob("all_folds_metrics_MULTI*.csv")) if metrics_dir.exists() else []
        if matches:
            m = normalize_classifier_col(pd.read_csv(matches[0]))
            row["metrics_available"] = True
            row["metrics_path"] = str(matches[0])
            for clf, sub in m.groupby("classifier", dropna=False):
                for metric in ["auc_final", "pr_auc_final", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1_score"]:
                    if metric in sub.columns:
                        row[f"{clf}_{metric}_mean"] = float(sub[metric].mean())
                        row[f"{clf}_{metric}_std"] = float(sub[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def top_level_findings(
    foldwise: pd.DataFrame,
    subgroup: pd.DataFrame,
    threshold: pd.DataFrame,
    fold5: pd.DataFrame,
    comparison: pd.DataFrame,
) -> Dict[str, Any]:
    logreg = foldwise[foldwise["classifier"].eq("logreg")]
    svm = foldwise[foldwise["classifier"].eq("svm")]
    primary = logreg if not logreg.empty else foldwise
    best_auc = float(foldwise["auc_final"].max()) if "auc_final" in foldwise.columns else np.nan
    primary_auc = float(primary["auc_final"].mean()) if len(primary) else np.nan
    primary_sens = float(primary["sensitivity"].mean()) if len(primary) else np.nan
    primary_spec = float(primary["specificity"].mean()) if len(primary) else np.nan
    worst = primary.sort_values("auc_final").iloc[0].to_dict() if len(primary) else {}
    fold5_primary = primary[primary["fold"].eq(5)]
    fold5_auc = float(fold5_primary["auc_final"].iloc[0]) if len(fold5_primary) else np.nan
    fold5_is_worst = bool(worst.get("fold") == 5) if worst else False
    threshold_05 = threshold[threshold["is_threshold_0_5"]]
    best_ba = threshold[threshold["is_best_balanced_accuracy"]]

    cn_man = subgroup[subgroup["grouping"].eq("cn_by_manufacturer")].copy()
    cn_man["false_positive_rate"] = 1.0 - cn_man["specificity"]
    max_cn_fp = (
        cn_man.sort_values("false_positive_rate", ascending=False).iloc[0].to_dict()
        if not cn_man.empty
        else {}
    )
    v5_row = comparison[comparison["version"].eq("v5_ch1_0_2")]
    v5_logreg_auc = (
        float(v5_row["logreg_auc_final_mean"].iloc[0])
        if len(v5_row) and "logreg_auc_final_mean" in v5_row.columns and pd.notna(v5_row["logreg_auc_final_mean"].iloc[0])
        else np.nan
    )
    return {
        "best_auc_any_classifier_fold_mean": best_auc,
        "primary_classifier": "logreg" if not logreg.empty else str(primary["classifier"].iloc[0]) if len(primary) else "NA",
        "primary_auc_mean": primary_auc,
        "primary_sensitivity_mean": primary_sens,
        "primary_specificity_mean": primary_spec,
        "fold5_auc_primary": fold5_auc,
        "fold5_is_worst_primary_auc": fold5_is_worst,
        "worst_primary_fold": int(worst.get("fold")) if worst else None,
        "threshold_05_rows": threshold_05,
        "best_ba_rows": best_ba,
        "max_cn_fp": max_cn_fp,
        "v5_logreg_auc_mean": v5_logreg_auc,
    }


def write_readme(
    run_dir: Path,
    metrics_path: Path,
    metadata_path: Path,
    foldwise: pd.DataFrame,
    subgroup: pd.DataFrame,
    threshold: pd.DataFrame,
    fold5: pd.DataFrame,
    comparison: pd.DataFrame,
    leakage: pd.DataFrame,
    run_config: Optional[Dict[str, Any]],
    run_manifest: Optional[Dict[str, Any]],
) -> None:
    findings = top_level_findings(foldwise, subgroup, threshold, fold5, comparison)
    primary = findings["primary_classifier"]
    primary_rows = foldwise[foldwise["classifier"].eq(primary)]
    svm_rows = foldwise[foldwise["classifier"].eq("svm")]
    def cn_manufacturer_rows(label: str) -> pd.DataFrame:
        return subgroup[
            subgroup["grouping"].eq("cn_by_manufacturer")
            & subgroup["group_value"].astype(str).str.upper().eq(label.upper())
        ]

    cn_ge = cn_manufacturer_rows("GE")
    cn_siemens = cn_manufacturer_rows("SIEMENS")
    cn_philips = cn_manufacturer_rows("Philips")
    threshold_05 = findings["threshold_05_rows"]
    best_ba = findings["best_ba_rows"]

    def mean_col(df: pd.DataFrame, col: str) -> float:
        return float(df[col].mean()) if col in df.columns and len(df) else np.nan

    def subgroup_line(label: str, df: pd.DataFrame) -> str:
        if df.empty:
            return f"- {label}: no disponible."
        parts = []
        for _, r in df.iterrows():
            parts.append(
                f"{r['classifier']} n={int(r['n'])}, specificity={r['specificity']:.3f}, "
                f"FP={int(r['fp'])}"
            )
        return f"- {label}: " + "; ".join(parts)

    lines = [
        "# ADNI v5.1 batch20260514 interim baseline audit",
        "",
        "This audit reads completed baseline outputs only. It does not train, modify tensors, or modify the ledger.",
        "",
        "## Inputs",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Metrics: `{metrics_path}`",
        f"- Metadata: `{metadata_path}`",
        "- Dataset: v5.1_batch20260514 no-Python-bandpass",
        "- Channels: `[1,0,2]`",
        "- Python bandpass applied: `False`",
        "",
        "## Main readout",
        "",
        (
            f"- Useful signal: yes, but modest. Mean AUC is "
            f"{mean_col(primary_rows, 'auc_final'):.3f} for {primary}"
            + (
                f" and {mean_col(svm_rows, 'auc_final'):.3f} for svm."
                if len(svm_rows)
                else "."
            )
        ),
        (
            f"- Main problem: threshold/sensitivity, not complete lack of ranking signal. "
            f"{primary} mean sensitivity={mean_col(primary_rows, 'sensitivity'):.3f}, "
            f"specificity={mean_col(primary_rows, 'specificity'):.3f}, "
            f"balanced accuracy={mean_col(primary_rows, 'balanced_accuracy'):.3f}."
        ),
        (
            f"- Fold 5: {'is' if findings['fold5_is_worst_primary_auc'] else 'is not'} "
            f"the worst {primary} fold by AUC. Fold 5 {primary} AUC="
            f"{findings['fold5_auc_primary']:.3f}; worst fold="
            f"{findings['worst_primary_fold']}."
        ),
        "",
        "## Threshold audit",
        "",
        "Threshold exploration is test-set-only and must not be used as the final reporting threshold.",
    ]
    for _, r in threshold_05.sort_values("classifier").iterrows():
        lines.append(
            f"- Current threshold 0.5, {r['classifier']}: sensitivity={r['sensitivity']:.3f}, "
            f"specificity={r['specificity']:.3f}, balanced_accuracy={r['balanced_accuracy']:.3f}, "
            f"F1={r['f1']:.3f}."
        )
    for _, r in best_ba.sort_values("classifier").iterrows():
        lines.append(
            f"- Exploratory best balanced-accuracy threshold, {r['classifier']}: "
            f"threshold={r['threshold']:.2f}, sensitivity={r['sensitivity']:.3f}, "
            f"specificity={r['specificity']:.3f}, balanced_accuracy={r['balanced_accuracy']:.3f}."
        )

    lines += [
        "",
        "## Manufacturer/subgroup audit",
        "",
        subgroup_line("CN-GE", cn_ge),
        subgroup_line("CN-Siemens", cn_siemens),
        subgroup_line("CN-Philips", cn_philips),
        (
            "- Manufacturer bias: review `subgroup_metrics_by_manufacturer.csv`; "
            "the key risk is a manufacturer-specific AD-like score shift in CN controls, "
            "especially when CN-GE sample size is small."
        ),
        "",
        "## Scanner leakage",
        "",
    ]
    if leakage.empty:
        lines.append("- Scanner leakage summaries were not found.")
    else:
        test_lat = "test_acc_site_latent"
        test_raw = "test_acc_site_raw"
        if test_lat in leakage.columns:
            lines.append(f"- Mean test scanner accuracy from latent features: {leakage[test_lat].mean():.3f}.")
        if test_raw in leakage.columns:
            lines.append(f"- Mean test scanner accuracy from raw inputs: {leakage[test_raw].mean():.3f}.")
        lines.append("- Correlations with AUC/sensitivity are exploratory only because n=5 folds.")

    lines += [
        "",
        "## Comparison with prior versions",
        "",
    ]
    for _, r in comparison.iterrows():
        if not bool(r.get("metrics_available", False)):
            continue
        lines.append(
            f"- `{r['version']}`: logreg AUC={r.get('logreg_auc_final_mean', np.nan):.3f}, "
            f"logreg sensitivity={r.get('logreg_sensitivity_mean', np.nan):.3f}, "
            f"logreg specificity={r.get('logreg_specificity_mean', np.nan):.3f}, "
            f"CN-GE={r.get('CN_GE', np.nan)}."
        )

    lines += [
        "",
        "## Recommended next experiment",
        "",
        (
            "Run a prespecified threshold/calibration experiment inside the training folds only "
            "(nested threshold selection or validation-derived threshold), then report untouched "
            "outer-fold performance. In parallel, compare logreg vs svm and check whether "
            "manufacturer-balanced folds or scanner-aware sensitivity analysis changes the AD "
            "sensitivity/specificity tradeoff."
        ),
        "",
        "## Files written",
        "",
        "- `foldwise_metrics_table.csv`",
        "- `confusion_by_fold.csv`",
        "- `pooled_confusion.csv`",
        "- `threshold_sensitivity_analysis.csv`",
        "- `subgroup_metrics_by_manufacturer.csv`",
        "- `score_distribution_by_group.csv`",
        "- `fold5_error_audit.csv`",
        "- `scanner_leakage_vs_auc.csv`",
        "- `calibration_reliability_table.csv`",
        "- `comparison_with_v5_v4.csv`",
        "- `audit_manifest.json`",
    ]
    OUTPUT_DIR.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    run_dir = resolve_run_dir()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = first_file(run_dir, "all_folds_metrics_MULTI*.csv")
    summary_path = first_file(run_dir, "summary_metrics_MULTI*.txt")
    metrics = pd.read_csv(metrics_path)
    metadata = normalize_metadata(pd.read_csv(METADATA_PATH))
    pred = load_predictions(run_dir, metadata)
    run_config = read_json_optional(run_dir / "run_config.json")
    run_manifest = read_json_optional(run_dir / "run_manifest.json")

    if pred["SubjectID"].duplicated().sum() == 0:
        raise RuntimeError("Expected repeated SubjectID across classifiers/folds; prediction table looks malformed.")
    if sorted(pred["classifier"].dropna().unique().tolist()) != ["logreg", "svm"]:
        raise RuntimeError(f"Unexpected classifiers: {sorted(pred['classifier'].dropna().unique().tolist())}")
    if not set(pred["y_true"].dropna().unique()).issubset({0, 1}):
        raise RuntimeError("y_true must be binary 0/1.")

    foldwise = make_foldwise_metrics(metrics, pred)
    confusion_by_fold, pooled_confusion = make_confusions(pred)
    threshold = threshold_analysis(pred)
    subgroup = subgroup_metrics(pred)
    score_dist = summarize_scores(pred)
    fold5 = fold5_error_audit(pred)
    reliability = reliability_rows(pred)
    leakage = load_scanner_leakage(run_dir)
    scanner = scanner_vs_auc(foldwise, leakage)
    comparison = compare_runs()

    foldwise.to_csv(OUTPUT_DIR / "foldwise_metrics_table.csv", index=False)
    confusion_by_fold.to_csv(OUTPUT_DIR / "confusion_by_fold.csv", index=False)
    pooled_confusion.to_csv(OUTPUT_DIR / "pooled_confusion.csv", index=False)
    threshold.to_csv(OUTPUT_DIR / "threshold_sensitivity_analysis.csv", index=False)
    subgroup.to_csv(OUTPUT_DIR / "subgroup_metrics_by_manufacturer.csv", index=False)
    score_dist.to_csv(OUTPUT_DIR / "score_distribution_by_group.csv", index=False)
    fold5.to_csv(OUTPUT_DIR / "fold5_error_audit.csv", index=False)
    scanner.to_csv(OUTPUT_DIR / "scanner_leakage_vs_auc.csv", index=False)
    reliability.to_csv(OUTPUT_DIR / "calibration_reliability_table.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "comparison_with_v5_v4.csv", index=False)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "summary_metrics_path": str(summary_path),
        "metadata_path": str(METADATA_PATH),
        "output_dir": str(OUTPUT_DIR),
        "training_run": False,
        "tensor_modified": False,
        "ledger_modified": False,
        "python_bandpass_applied": False,
        "n_prediction_rows": int(len(pred)),
        "classifiers": sorted(pred["classifier"].dropna().unique().tolist()),
        "folds": sorted(int(x) for x in pred["fold"].dropna().unique().tolist()),
        "run_config_available": run_config is not None,
        "run_manifest_available": run_manifest is not None,
    }
    (OUTPUT_DIR / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(
        run_dir=run_dir,
        metrics_path=metrics_path,
        metadata_path=METADATA_PATH,
        foldwise=foldwise,
        subgroup=subgroup,
        threshold=threshold,
        fold5=fold5,
        comparison=comparison,
        leakage=leakage,
        run_config=run_config,
        run_manifest=run_manifest,
    )

    print(f"Run dir: {run_dir}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Prediction rows: {len(pred)}")
    print(
        "Mean AUC final: "
        + ", ".join(
            f"{clf}={sub['auc_final'].mean():.4f}"
            for clf, sub in foldwise.groupby("classifier", dropna=False)
        )
    )
    print("No training. No tensor modification. No ledger modification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
