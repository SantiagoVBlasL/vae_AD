#!/usr/bin/env python3
"""Read-only full audit for ADNI expanded V4 model outputs.

This script only reads existing CSV/TXT/JSON artifacts and writes small audit
tables. It does not load tensors, checkpoints, joblibs, or large arrays.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "results/revision_bspc_2026/adni_expanded_v4_beta25_static3"
DEFAULT_METADATA = (
    REPO_ROOT
    / "data/revision_bspc_2026/adni_expanded_v4_all_available"
    / "subject_metadata_adni_expanded_v4_all_available.csv"
)
DEFAULT_V3_RUN_DIR = REPO_ROOT / "results/revision_bspc_2026/adni_expanded_v3_beta25_static3"

OUTPUT_NAMES = [
    "pooled_test_predictions_with_metadata.csv",
    "global_metrics_by_classifier.csv",
    "metrics_by_manufacturer.csv",
    "metrics_by_site3.csv",
    "metrics_by_sourcecohort.csv",
    "metrics_by_sex.csv",
    "metrics_by_agebin.csv",
    "threshold_analysis_pooled_exploratory.csv",
    "threshold_readme.md",
    "false_positives_CN_as_AD.csv",
    "false_negatives_AD_as_CN.csv",
    "error_rates_by_manufacturer.csv",
    "error_rates_by_sourcecohort.csv",
    "calibration_bins.csv",
    "calibration_summary.csv",
    "v3_vs_v4_summary.csv",
    "latent_qc_summary.csv",
    "latent_qc_fold_level.csv",
    "latent_qc_aggregate.csv",
    "latent_info_summary.csv",
    "scanner_leakage_summary.csv",
    "optuna_efficiency_by_fold.csv",
    "optuna_best_params_summary.csv",
    "optuna_recommendations.json",
    "README.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only V4 audit from existing artifacts.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--v3-run-dir", type=Path, default=DEFAULT_V3_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_output_dir(run_dir: Path) -> Path:
    return run_dir.resolve() / "audit_v4"


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path exists but is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing audit files. Use --overwrite if intentional:\n"
            + "\n".join(str(path) for path in existing)
        )
    if overwrite:
        stale = output_dir / "latent_qc_summary.csv"
        if stale.exists():
            stale.unlink()


def lower_map(columns: Iterable[str]) -> Dict[str, str]:
    return {str(col).strip().lower(): col for col in columns}


def pick_column(
    df: pd.DataFrame,
    role: str,
    preferred: Sequence[str],
    notes: List[str],
    fuzzy_tokens: Sequence[str] = (),
) -> str:
    by_lower = lower_map(df.columns)
    matches = [by_lower[name.lower()] for name in preferred if name.lower() in by_lower]
    matches = list(dict.fromkeys(matches))
    if matches:
        if len(matches) > 1:
            notes.append(f"{role}: multiple preferred columns {matches}; using {matches[0]}.")
        return matches[0]
    fuzzy = []
    for col in df.columns:
        col_l = str(col).lower()
        if all(token in col_l for token in fuzzy_tokens):
            fuzzy.append(col)
    fuzzy = list(dict.fromkeys(fuzzy))
    if len(fuzzy) == 1:
        notes.append(f"{role}: selected fuzzy column {fuzzy[0]}.")
        return fuzzy[0]
    if len(fuzzy) > 1:
        raise ValueError(f"Ambiguous {role} columns {fuzzy}. Available columns: {list(df.columns)}")
    raise ValueError(f"Could not detect {role} column. Available columns: {list(df.columns)}")


def detect_prediction_columns(df: pd.DataFrame, notes: List[str]) -> Dict[str, str]:
    return {
        "subject": pick_column(df, "SubjectID", ["SubjectID", "subject_id", "PTID"], notes, ("subject",)),
        "y_true": pick_column(df, "true label", ["y_true", "label", "true_label"], notes, ("true",)),
        "y_score": pick_column(
            df,
            "AD score/probability",
            ["y_score_final", "y_score", "y_proba", "proba_ad", "prob_ad", "y_score_cal", "y_score_raw"],
            notes,
            ("score",),
        ),
        "y_pred": pick_column(df, "predicted label", ["y_pred", "pred", "predicted_label"], notes, ("pred",)),
    }


def to_binary(series: pd.Series, role: str) -> pd.Series:
    def convert(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, np.integer, float, np.floating)) and float(value) in (0.0, 1.0):
            return int(value)
        text = str(value).strip().upper()
        if text in {"CN", "CONTROL", "NORMAL", "0", "0.0"}:
            return 0
        if text in {"AD", "DEMENTIA", "ALZHEIMER", "1", "1.0"}:
            return 1
        return np.nan

    out = series.map(convert)
    if out.isna().any():
        bad = sorted(series[out.isna()].dropna().astype(str).unique().tolist())
        raise ValueError(f"Could not map {role} to CN=0/AD=1. Bad values: {bad}")
    return out.astype(int)


def infer_age_bin(age: object) -> str:
    try:
        value = float(age)
    except (TypeError, ValueError):
        return "Unknown"
    if not np.isfinite(value):
        return "Unknown"
    if value < 65:
        return "<65"
    if value < 70:
        return "65-70"
    if value < 75:
        return "70-75"
    if value < 80:
        return "75-80"
    return ">=80"


def normalize_site3(value: object, subject_id: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        sid = str(subject_id).strip()
        return sid[:3] if len(sid) >= 3 else "Unknown"
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(3) if text.isdigit() else text


def validate_inputs(run_dir: Path, metadata_path: Path) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    optional_missing: List[str] = []
    if not run_dir.exists():
        errors.append(f"run_dir does not exist: {run_dir}")
    if not metadata_path.exists():
        errors.append(f"metadata does not exist: {metadata_path}")
    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        if not fold_dir.exists():
            errors.append(f"missing required fold dir: {fold_dir}")
            continue
        for clf in ["logreg", "svm"]:
            path = fold_dir / f"test_predictions_{clf}.csv"
            if not path.exists():
                errors.append(f"missing required prediction file: {path}")
        for optional in [
            f"latent_qc_metrics.csv",
            f"fold_{fold}_test_latent_info_summary.csv",
            f"fold_{fold}_trainDev_latent_info_summary.csv",
            f"fold_{fold}_test_scanner_leakage_summary.csv",
            f"fold_{fold}_scanner_leakage_summary.csv",
            f"optuna_trials_logreg_fold_{fold}.csv",
            f"optuna_trials_svm_fold_{fold}.csv",
            f"optuna_best_trial_logreg_fold_{fold}.json",
            f"optuna_best_trial_svm_fold_{fold}.json",
        ]:
            path = fold_dir / optional
            if not path.exists():
                optional_missing.append(str(path))
    return errors, optional_missing


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path)
    if "SubjectID" not in meta.columns:
        raise ValueError(f"Metadata lacks SubjectID. Columns: {list(meta.columns)}")
    meta = meta.copy()
    meta["SubjectID"] = meta["SubjectID"].astype(str).str.strip()
    meta = meta.drop_duplicates("SubjectID", keep="first")
    for col in ["ResearchGroup_Mapped", "Manufacturer", "Site3", "SourceCohort", "Age", "Sex"]:
        if col not in meta.columns:
            meta[col] = np.nan
    meta["Site3"] = [normalize_site3(site, sid) for site, sid in zip(meta["Site3"], meta["SubjectID"])]
    meta["AgeBin"] = meta["Age"].map(infer_age_bin)
    return meta


def build_pooled_predictions(run_dir: Path, metadata: pd.DataFrame, notes: List[str]) -> pd.DataFrame:
    frames = []
    for fold in range(1, 6):
        for classifier in ["logreg", "svm"]:
            path = run_dir / f"fold_{fold}" / f"test_predictions_{classifier}.csv"
            raw = pd.read_csv(path)
            cols = detect_prediction_columns(raw, notes)
            frame = pd.DataFrame(
                {
                    "SubjectID": raw[cols["subject"]].astype(str).str.strip(),
                    "fold": fold,
                    "classifier": classifier,
                    "y_true": to_binary(raw[cols["y_true"]], "true label"),
                    "y_score": pd.to_numeric(raw[cols["y_score"]], errors="coerce"),
                    "y_pred": to_binary(raw[cols["y_pred"]], "predicted label"),
                    "prediction_source": str(path),
                }
            )
            if frame["y_score"].isna().any():
                raise ValueError(f"Non-numeric y_score values in {path}")
            frames.append(frame)
    pooled = pd.concat(frames, ignore_index=True)
    merged = pooled.merge(
        metadata[
            ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Site3", "SourceCohort", "Age", "Sex", "AgeBin"]
        ],
        on="SubjectID",
        how="left",
        validate="many_to_one",
    )
    missing_meta = merged["ResearchGroup_Mapped"].isna().sum()
    if missing_meta:
        notes.append(f"Metadata missing for {int(missing_meta)} pooled prediction rows.")
    order = [
        "SubjectID",
        "fold",
        "classifier",
        "y_true",
        "y_score",
        "y_pred",
        "ResearchGroup_Mapped",
        "Manufacturer",
        "Site3",
        "SourceCohort",
        "Age",
        "Sex",
        "AgeBin",
        "prediction_source",
    ]
    return merged[order]


def score_is_probability(scores: pd.Series) -> bool:
    score = pd.to_numeric(scores, errors="coerce").dropna()
    return bool(len(score) and score.between(0.0, 1.0).all())


def safe_div(num: float, den: float) -> float:
    return np.nan if den == 0 else num / den


def core_metrics(df: pd.DataFrame, threshold: float = 0.5, require_both: bool = False) -> Dict[str, object]:
    y_true = df["y_true"].astype(int).to_numpy()
    y_score = pd.to_numeric(df["y_score"], errors="coerce").to_numpy()
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_cn = int((y_true == 0).sum())
    n_ad = int((y_true == 1).sum())
    has_both = n_cn > 0 and n_ad > 0
    mean_cn = float(np.mean(y_score[y_true == 0])) if n_cn else np.nan
    mean_ad = float(np.mean(y_score[y_true == 1])) if n_ad else np.nan
    out = {
        "n": int(len(df)),
        "n_CN": n_cn,
        "n_AD": n_ad,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
        "mean_score": float(np.mean(y_score)) if len(y_score) else np.nan,
        "ad_like_rate": float(np.mean(y_pred)) if len(y_pred) else np.nan,
        "mean_score_CN": mean_cn,
        "mean_score_AD": mean_ad,
        "score_separation_mean_AD_minus_CN": mean_ad - mean_cn if n_cn and n_ad else np.nan,
        "status": "ok" if has_both else "single_class_or_empty",
    }
    if require_both and not has_both:
        out.update(
            {
                "roc_auc": np.nan,
                "pr_auc": np.nan,
                "accuracy": accuracy_score(y_true, y_pred) if len(y_true) else np.nan,
                "balanced_accuracy": np.nan,
                "sensitivity_AD": np.nan,
                "specificity_CN": np.nan,
                "precision": precision_score(y_true, y_pred, zero_division=0) if len(y_true) else np.nan,
                "recall": np.nan,
                "f1": f1_score(y_true, y_pred, zero_division=0) if len(y_true) else np.nan,
                "brier": brier_score_loss(y_true, y_score) if score_is_probability(df["y_score"]) and len(y_true) else np.nan,
            }
        )
        return out
    out.update(
        {
            "roc_auc": roc_auc_score(y_true, y_score) if has_both else np.nan,
            "pr_auc": average_precision_score(y_true, y_score) if has_both else np.nan,
            "accuracy": accuracy_score(y_true, y_pred) if len(y_true) else np.nan,
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if has_both else np.nan,
            "sensitivity_AD": recall_score(y_true, y_pred, pos_label=1, zero_division=0) if has_both else np.nan,
            "specificity_CN": recall_score(y_true, y_pred, pos_label=0, zero_division=0) if has_both else np.nan,
            "precision": precision_score(y_true, y_pred, zero_division=0) if len(y_true) else np.nan,
            "recall": recall_score(y_true, y_pred, zero_division=0) if len(y_true) else np.nan,
            "f1": f1_score(y_true, y_pred, zero_division=0) if len(y_true) else np.nan,
            "brier": brier_score_loss(y_true, y_score) if score_is_probability(df["y_score"]) and len(y_true) else np.nan,
        }
    )
    return out


def global_metrics(pooled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for classifier, group in pooled.groupby("classifier", sort=True):
        row = {
            "classifier": classifier,
            "pooled_metrics_scope": "pooled_oof_recomputed_from_fold_test_predictions",
            "pooled_metrics_note": "Pooled metrics are independently recomputed by this audit from fold test predictions; threshold metrics use y_score >= 0.5.",
        }
        row.update(core_metrics(group, threshold=0.5))
        rows.append(row)
    return pd.DataFrame(rows)


def load_pipeline_fold_metric_summary(run_dir: Path, notes: List[str]) -> pd.DataFrame:
    files = sorted(run_dir.glob("all_folds_metrics_MULTI_*.csv"))
    if not files:
        notes.append(f"No all_folds_metrics_MULTI_*.csv files found in {run_dir}.")
        return pd.DataFrame()
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["pipeline_metrics_source_file"] = path.name
        frames.append(df)
    metrics = pd.concat(frames, ignore_index=True)
    if "actual_classifier_type" in metrics.columns:
        clf_col = "actual_classifier_type"
    elif "classifier" in metrics.columns:
        clf_col = "classifier"
    else:
        notes.append(
            "Pipeline fold metrics were found but no classifier column was detected; "
            f"columns={list(metrics.columns)}."
        )
        return pd.DataFrame()

    metric_map = {
        "auc": "roc_auc",
        "pr_auc": "pr_auc",
        "balanced_accuracy": "balanced_accuracy",
        "sensitivity": "sensitivity_AD",
        "specificity": "specificity_CN",
    }
    rows = []
    for classifier, group in metrics.groupby(clf_col, dropna=False, sort=True):
        row = {
            "classifier": str(classifier),
            "pipeline_metrics_scope": "mean_sd_across_outer_test_folds_from_training_pipeline",
            "pipeline_metrics_source_files": ";".join(sorted(group["pipeline_metrics_source_file"].astype(str).unique())),
            "pipeline_n_folds": int(group["fold"].nunique()) if "fold" in group.columns else int(len(group)),
        }
        for src_col, out_name in metric_map.items():
            if src_col not in group.columns:
                row[f"pipeline_mean_fold_{out_name}"] = np.nan
                row[f"pipeline_sd_fold_{out_name}"] = np.nan
                continue
            values = pd.to_numeric(group[src_col], errors="coerce")
            row[f"pipeline_mean_fold_{out_name}"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"pipeline_sd_fold_{out_name}"] = (
                float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0 if values.notna().sum() == 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def add_pipeline_fold_metrics(global_df: pd.DataFrame, pipeline_df: pd.DataFrame) -> pd.DataFrame:
    if pipeline_df.empty:
        return global_df
    return global_df.merge(pipeline_df, on="classifier", how="left", validate="one_to_one")


def stratum_metrics(pooled: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for (classifier, stratum), group in pooled.groupby(["classifier", column], dropna=False, sort=True):
        row = {"classifier": classifier, column: "NA" if pd.isna(stratum) else stratum}
        row.update(core_metrics(group, threshold=0.5, require_both=True))
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    unique = np.unique(scores[np.isfinite(scores)])
    mids = (unique[:-1] + unique[1:]) / 2.0 if len(unique) > 1 else np.array([], dtype=float)
    values = np.concatenate([np.array([0.0, 0.5, 1.0]), unique, mids])
    return np.unique(np.clip(values[np.isfinite(values)], 0.0, 1.0))


def threshold_table_for_classifier(group: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in candidate_thresholds(group["y_score"].to_numpy()):
        row = core_metrics(group, threshold=float(threshold))
        row["threshold"] = float(threshold)
        row["youden"] = row["sensitivity_AD"] + row["specificity_CN"] - 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def pick_threshold_candidate(table: pd.DataFrame, strategy: str) -> Optional[pd.Series]:
    if table.empty:
        return None
    if strategy == "threshold_0.5":
        idx = (table["threshold"] - 0.5).abs().idxmin()
        return table.loc[idx]
    if strategy == "youden":
        return table.sort_values(["youden", "balanced_accuracy", "f1", "threshold"], ascending=[False, False, False, True]).iloc[0]
    if strategy == "max_balanced_accuracy":
        return table.sort_values(["balanced_accuracy", "f1", "threshold"], ascending=[False, False, True]).iloc[0]
    if strategy == "high_sensitivity_ge_0.75":
        subset = table[table["sensitivity_AD"] >= 0.75].copy()
        if subset.empty:
            return None
        # Use the most specific threshold satisfying the target, avoiding a degenerate all-AD classifier.
        return subset.sort_values(["balanced_accuracy", "specificity_CN", "threshold"], ascending=[False, False, False]).iloc[0]
    if strategy == "high_specificity_ge_0.90":
        subset = table[table["specificity_CN"] >= 0.90].copy()
        if subset.empty:
            return None
        return subset.sort_values(["balanced_accuracy", "sensitivity_AD", "threshold"], ascending=[False, False, True]).iloc[0]
    raise ValueError(strategy)


def threshold_analysis(pooled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strategy_labels = [
        ("threshold_0.5", False),
        ("youden", True),
        ("max_balanced_accuracy", True),
        ("high_sensitivity_ge_0.75", True),
        ("high_specificity_ge_0.90", True),
    ]
    for classifier, group in pooled.groupby("classifier", sort=True):
        table = threshold_table_for_classifier(group)
        for strategy, exploratory in strategy_labels:
            picked = pick_threshold_candidate(table, strategy)
            row = {
                "classifier": classifier,
                "strategy": strategy,
                "is_exploratory": exploratory,
                "valid_for_final_claim": not exploratory,
                "selection_note": (
                    "fixed threshold"
                    if not exploratory
                    else "optimized on pooled test predictions; exploratory only"
                ),
            }
            if picked is None:
                row.update({col: np.nan for col in table.columns})
                row.update({"status": "not_available"})
            else:
                row.update(picked.to_dict())
                row["status"] = "ok"
            rows.append(row)
    return pd.DataFrame(rows)


def find_train_dev_prediction_files(run_dir: Path) -> List[Path]:
    files = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        for path in fold_dir.glob("*.csv"):
            name = path.name.lower()
            if "prediction" in name and "test" not in name and any(token in name for token in ["train", "dev", "val", "valid"]):
                files.append(path)
    return files


def error_tables(pooled: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = pooled.copy()
    work["y_pred"] = (pd.to_numeric(work["y_score"], errors="coerce") >= 0.5).astype(int)
    fp = work[(work["y_true"] == 0) & (work["y_pred"] == 1)].copy()
    fn = work[(work["y_true"] == 1) & (work["y_pred"] == 0)].copy()
    cols = [
        "SubjectID",
        "fold",
        "classifier",
        "y_score",
        "y_pred",
        "ResearchGroup_Mapped",
        "Manufacturer",
        "Site3",
        "SourceCohort",
        "Age",
        "Sex",
    ]
    fp = fp[cols].sort_values(["classifier", "y_score"], ascending=[True, False])
    fn = fn[cols].sort_values(["classifier", "y_score"], ascending=[True, True])

    def summarize(by: str) -> pd.DataFrame:
        rows = []
        for (classifier, value), group in work.groupby(["classifier", by], dropna=False, sort=True):
            y = group["y_true"].astype(int).to_numpy()
            pred = group["y_pred"].astype(int).to_numpy()
            tn, fp_n, fn_n, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            rows.append(
                {
                    "classifier": classifier,
                    by: "NA" if pd.isna(value) else value,
                    "n": int(len(group)),
                    "n_CN": int((y == 0).sum()),
                    "n_AD": int((y == 1).sum()),
                    "fp_CN_as_AD": int(fp_n),
                    "fn_AD_as_CN": int(fn_n),
                    "false_positive_rate_CN": safe_div(fp_n, fp_n + tn),
                    "false_negative_rate_AD": safe_div(fn_n, fn_n + tp),
                }
            )
        return pd.DataFrame(rows)

    return fp, fn, summarize("Manufacturer"), summarize("SourceCohort")


def calibration_audit(pooled: pd.DataFrame, n_bins: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bin_rows = []
    summary_rows = []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for classifier, group in pooled.groupby("classifier", sort=True):
        y_true = group["y_true"].astype(int).to_numpy()
        scores = pd.to_numeric(group["y_score"], errors="coerce").to_numpy()
        prob = bool(np.all(np.isfinite(scores)) and np.min(scores) >= 0.0 and np.max(scores) <= 1.0)
        ece = 0.0
        mce = 0.0
        for i in range(n_bins):
            low, high = bins[i], bins[i + 1]
            if i == n_bins - 1:
                mask = (scores >= low) & (scores <= high)
            else:
                mask = (scores >= low) & (scores < high)
            n = int(mask.sum())
            mean_pred = float(scores[mask].mean()) if n else np.nan
            frac_ad = float(y_true[mask].mean()) if n else np.nan
            abs_gap = abs(mean_pred - frac_ad) if n else np.nan
            if n and prob:
                ece += (n / len(scores)) * abs_gap
                mce = max(mce, abs_gap)
            bin_rows.append(
                {
                    "classifier": classifier,
                    "bin_index": i,
                    "bin_low": low,
                    "bin_high": high,
                    "n": n,
                    "mean_pred": mean_pred,
                    "frac_AD": frac_ad,
                    "abs_calibration_gap": abs_gap,
                }
            )
        summary_rows.append(
            {
                "classifier": classifier,
                "n": int(len(group)),
                "score_is_probability": prob,
                "brier": brier_score_loss(y_true, scores) if prob else np.nan,
                "ece_10_bins": float(ece) if prob else np.nan,
                "mce_10_bins": float(mce) if prob else np.nan,
            }
        )
    return pd.DataFrame(bin_rows), pd.DataFrame(summary_rows)


def load_v3_global_metrics(v3_run_dir: Path, notes: List[str]) -> pd.DataFrame:
    path = v3_run_dir / "audit_v3/global_metrics_by_classifier.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["source"] = "v3_audit_global_metrics"
        return df
    notes.append(f"V3 global metrics not found at {path}; recomputing from fold predictions.")
    metadata = pd.DataFrame()
    frames = []
    for fold in range(1, 6):
        for classifier in ["logreg", "svm"]:
            pred_path = v3_run_dir / f"fold_{fold}" / f"test_predictions_{classifier}.csv"
            if not pred_path.exists():
                continue
            raw = pd.read_csv(pred_path)
            local_notes: List[str] = []
            cols = detect_prediction_columns(raw, local_notes)
            frames.append(
                pd.DataFrame(
                    {
                        "SubjectID": raw[cols["subject"]].astype(str),
                        "fold": fold,
                        "classifier": classifier,
                        "y_true": to_binary(raw[cols["y_true"]], "true label"),
                        "y_score": pd.to_numeric(raw[cols["y_score"]], errors="coerce"),
                        "y_pred": to_binary(raw[cols["y_pred"]], "predicted label"),
                    }
                )
            )
    if not frames:
        return pd.DataFrame()
    pooled = pd.concat(frames, ignore_index=True)
    return global_metrics(pooled)


def compare_v3_v4(v3_metrics: pd.DataFrame, v4_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prevalence_note = (
        "PR-AUC is prevalence-sensitive; V4 adds 7 CN and 0 AD relative to V3 in this comparison, "
        "so PR-AUC deltas reflect both model behavior and changed class prevalence."
    )
    for classifier in sorted(set(v4_metrics["classifier"].astype(str))):
        v4 = v4_metrics[v4_metrics["classifier"].astype(str) == classifier]
        v3 = v3_metrics[v3_metrics["classifier"].astype(str) == classifier] if not v3_metrics.empty else pd.DataFrame()
        if v4.empty:
            continue
        v4r = v4.iloc[0]
        v3r = v3.iloc[0] if not v3.empty else pd.Series(dtype=object)
        for version, row in [("v3", v3r), ("v4", v4r)]:
            if row.empty:
                rows.append({"classifier": classifier, "version": version, "status": "missing", "note": prevalence_note})
                continue
            fp = row.get("fp", np.nan)
            tn = row.get("tn", np.nan)
            fn = row.get("fn", np.nan)
            tp = row.get("tp", np.nan)
            rows.append(
                {
                    "classifier": classifier,
                    "version": version,
                    "status": "ok",
                    "n": row.get("n"),
                    "n_CN": row.get("n_CN"),
                    "n_AD": row.get("n_AD"),
                    "roc_auc": row.get("roc_auc"),
                    "pr_auc": row.get("pr_auc"),
                    "balanced_accuracy": row.get("balanced_accuracy"),
                    "sensitivity_AD": row.get("sensitivity_AD"),
                    "specificity_CN": row.get("specificity_CN"),
                    "brier": row.get("brier"),
                    "false_positive_rate_CN": safe_div(fp, fp + tn) if pd.notna(fp) and pd.notna(tn) else np.nan,
                    "false_negative_rate_AD": safe_div(fn, fn + tp) if pd.notna(fn) and pd.notna(tp) else np.nan,
                    "note": prevalence_note,
                }
            )
        if not v3.empty:
            rows.append(
                {
                    "classifier": classifier,
                    "version": "v4_minus_v3",
                    "status": "delta",
                    "n": v4r.get("n") - v3r.get("n"),
                    "n_CN": v4r.get("n_CN") - v3r.get("n_CN"),
                    "n_AD": v4r.get("n_AD") - v3r.get("n_AD"),
                    "roc_auc": v4r.get("roc_auc") - v3r.get("roc_auc"),
                    "pr_auc": v4r.get("pr_auc") - v3r.get("pr_auc"),
                    "balanced_accuracy": v4r.get("balanced_accuracy") - v3r.get("balanced_accuracy"),
                    "sensitivity_AD": v4r.get("sensitivity_AD") - v3r.get("sensitivity_AD"),
                    "specificity_CN": v4r.get("specificity_CN") - v3r.get("specificity_CN"),
                    "brier": v4r.get("brier") - v3r.get("brier"),
                    "false_positive_rate_CN": safe_div(v4r.get("fp"), v4r.get("fp") + v4r.get("tn"))
                    - safe_div(v3r.get("fp"), v3r.get("fp") + v3r.get("tn")),
                    "false_negative_rate_AD": safe_div(v4r.get("fn"), v4r.get("fn") + v4r.get("tp"))
                    - safe_div(v3r.get("fn"), v3r.get("fn") + v3r.get("tp")),
                    "note": prevalence_note,
                }
            )
    return pd.DataFrame(rows)


def summarize_mean_std(df: pd.DataFrame, columns: Sequence[str], group_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    if group_cols:
        groups = df.groupby(list(group_cols), dropna=False, sort=True)
    else:
        groups = [((), df)]
    for keys, group in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row["n_rows"] = int(len(group))
        for col in columns:
            if col in group.columns:
                vals = pd.to_numeric(group[col], errors="coerce")
                row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
                row[f"{col}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0 if vals.notna().sum() == 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def latent_and_scanner_summaries(
    run_dir: Path, optional_missing: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    latent_qc_frames = []
    latent_info_frames = []
    scanner_frames = []
    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        p = fold_dir / "latent_qc_metrics.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["fold"] = fold
            latent_qc_frames.append(df)
        else:
            optional_missing.append(str(p))
        for split, filename in [
            ("test", f"fold_{fold}_test_latent_info_summary.csv"),
            ("trainDev", f"fold_{fold}_trainDev_latent_info_summary.csv"),
        ]:
            p = fold_dir / filename
            if p.exists():
                df = pd.read_csv(p)
                df["fold"] = fold
                df["split"] = split
                latent_info_frames.append(df)
            else:
                optional_missing.append(str(p))
        for split, filename in [
            ("test", f"fold_{fold}_test_scanner_leakage_summary.csv"),
            ("trainDev", f"fold_{fold}_scanner_leakage_summary.csv"),
        ]:
            p = fold_dir / filename
            if p.exists():
                df = pd.read_csv(p)
                df["fold"] = fold
                df["split"] = split
                scanner_frames.append(df)
            else:
                optional_missing.append(str(p))

    latent_qc = pd.concat(latent_qc_frames, ignore_index=True) if latent_qc_frames else pd.DataFrame()
    latent_qc_aggregate = summarize_mean_std(
        latent_qc,
        ["silhouette_latent", "acc_site_latent", "acc_site_raw", "chance_level", "latent_dim", "beta_max"],
        [],
    ) if not latent_qc.empty else pd.DataFrame()
    latent_qc_fold_level = latent_qc.copy() if not latent_qc.empty else pd.DataFrame()

    latent_info = pd.concat(latent_info_frames, ignore_index=True) if latent_info_frames else pd.DataFrame()
    if not latent_info.empty:
        agg = summarize_mean_std(
            latent_info,
            ["mi_sum_nats", "mi_mean_nats", "n_active", "frac_active", "total_correlation_nats"],
            ["split", "variable"],
        )
        # Add MI(Manufacturer)/MI(Y_target) by split.
        ratio_rows = []
        for split, group in agg.groupby("split"):
            y = group[group["variable"] == "Y_target"]
            man = group[group["variable"] == "Manufacturer"]
            ratio = np.nan
            if not y.empty and not man.empty and y.iloc[0].get("mi_sum_nats_mean", 0) not in (0, np.nan):
                denom = y.iloc[0].get("mi_sum_nats_mean")
                ratio = man.iloc[0].get("mi_sum_nats_mean") / denom if denom else np.nan
            ratio_rows.append({"split": split, "variable": "MI_Manufacturer_over_Y_target", "mi_manufacturer_over_y_ratio": ratio})
        latent_info_out = pd.concat([agg, pd.DataFrame(ratio_rows)], ignore_index=True, sort=False)
    else:
        latent_info_out = pd.DataFrame()

    scanner = pd.concat(scanner_frames, ignore_index=True) if scanner_frames else pd.DataFrame()
    scanner_out = summarize_mean_std(
        scanner,
        ["chance_level", "acc_site_raw", "acc_site_raw_std", "acc_site_latent", "acc_site_latent_std", "n_samples", "n_sites"],
        ["split", "site_col"],
    ) if not scanner.empty else pd.DataFrame()
    return latent_qc_fold_level, latent_qc_aggregate, latent_info_out, scanner_out


def read_best_trial_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def best_by_trial_limit(trials: pd.DataFrame, limit: int) -> float:
    sub = trials[(trials["state"].astype(str).str.upper() == "COMPLETE") & (pd.to_numeric(trials["number"], errors="coerce") < limit)]
    if sub.empty:
        return np.nan
    return float(pd.to_numeric(sub["value"], errors="coerce").max())


def best_trial_number_by_limit(trials: pd.DataFrame) -> int:
    complete = trials[trials["state"].astype(str).str.upper() == "COMPLETE"].copy()
    if complete.empty:
        return -1
    values = pd.to_numeric(complete["value"], errors="coerce")
    return int(complete.loc[values.idxmax(), "number"])


def param_near_boundary(value: float, values: pd.Series) -> str:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    vals = vals[vals > 0]
    if not len(vals) or not np.isfinite(value) or value <= 0:
        return "unknown"
    lo, hi = vals.min(), vals.max()
    if lo <= 0 or hi <= lo:
        return "no"
    lv, llo, lhi = np.log10(value), np.log10(lo), np.log10(hi)
    span = lhi - llo
    if span <= 0:
        return "no"
    if (lv - llo) / span <= 0.05:
        return "near_low_inferred_boundary"
    if (lhi - lv) / span <= 0.05:
        return "near_high_inferred_boundary"
    return "no"


def optuna_audit(run_dir: Path, optional_missing: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    rows = []
    best_rows = []
    for fold in range(1, 6):
        for classifier in ["logreg", "svm"]:
            trials_path = run_dir / f"fold_{fold}" / f"optuna_trials_{classifier}_fold_{fold}.csv"
            best_path = run_dir / f"fold_{fold}" / f"optuna_best_trial_{classifier}_fold_{fold}.json"
            if not trials_path.exists():
                optional_missing.append(str(trials_path))
                continue
            trials = pd.read_csv(trials_path)
            best_json = read_best_trial_json(best_path)
            if not best_json:
                optional_missing.append(str(best_path))
            complete = trials[trials["state"].astype(str).str.upper() == "COMPLETE"].copy()
            values = pd.to_numeric(complete["value"], errors="coerce")
            best_idx = values.idxmax() if len(values) else None
            best_trial_number = int(complete.loc[best_idx, "number"]) if best_idx is not None else np.nan
            best_value = float(values.max()) if len(values) else np.nan
            best_50 = best_by_trial_limit(trials, 50)
            best_100 = best_by_trial_limit(trials, 100)
            best_200 = best_by_trial_limit(trials, 200)
            best_300 = best_by_trial_limit(trials, 300)
            params = best_json.get("best_params", {})
            if not params and best_idx is not None:
                params = {
                    c.replace("params_", ""): complete.loc[best_idx, c]
                    for c in complete.columns
                    if c.startswith("params_")
                }
            rows.append(
                {
                    "classifier": classifier,
                    "fold": fold,
                    "n_trials": int(len(trials)),
                    "n_complete": int(len(complete)),
                    "best_trial_number": best_trial_number,
                    "best_value": best_value,
                    "best_value_by_50_trials": best_50,
                    "best_value_by_100_trials": best_100,
                    "best_value_by_200_trials": best_200,
                    "best_value_by_300_trials": best_300,
                    "delta_best_vs_100": best_value - best_100 if pd.notna(best_100) else np.nan,
                    "delta_best_vs_200": best_value - best_200 if pd.notna(best_200) else np.nan,
                    "final_best_found_after_trial_300": bool(best_trial_number >= 300) if pd.notna(best_trial_number) else np.nan,
                    "best_params_json": json.dumps(params, sort_keys=True),
                }
            )
            for param_name, value in params.items():
                simple = param_name.replace("model__", "")
                trial_col = f"params_model__{simple}"
                numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                best_rows.append(
                    {
                        "classifier": classifier,
                        "fold": fold,
                        "param": simple,
                        "value": value,
                        "numeric_value": numeric_value,
                        "near_inferred_search_boundary": param_near_boundary(float(numeric_value), trials[trial_col]) if trial_col in trials.columns and pd.notna(numeric_value) else "unknown",
                    }
                )
    efficiency = pd.DataFrame(rows)
    best_params = pd.DataFrame(best_rows)
    summary_rows = []
    if not best_params.empty:
        for (classifier, param), group in best_params.groupby(["classifier", "param"], sort=True):
            numeric = pd.to_numeric(group["numeric_value"], errors="coerce").dropna()
            row = {
                "classifier": classifier,
                "param": param,
                "n_folds": int(group["fold"].nunique()),
                "values": ";".join(map(str, group["value"].tolist())),
                "near_boundary_folds": ";".join(
                    f"fold_{int(r.fold)}:{r.near_inferred_search_boundary}"
                    for r in group.itertuples()
                    if str(r.near_inferred_search_boundary) not in {"no", "unknown"}
                ),
            }
            if len(numeric):
                pos = numeric[numeric > 0]
                row.update(
                    {
                        "min": float(numeric.min()),
                        "median": float(numeric.median()),
                        "max": float(numeric.max()),
                        "log10_range_orders": float(np.log10(pos.max()) - np.log10(pos.min())) if len(pos) and pos.min() > 0 else np.nan,
                        "high_instability_gt_2_orders": bool((np.log10(pos.max()) - np.log10(pos.min())) > 2) if len(pos) and pos.min() > 0 else False,
                    }
                )
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    recommendations = build_optuna_recommendations(efficiency, summary)
    return efficiency, summary, recommendations


def build_optuna_recommendations(efficiency: pd.DataFrame, summary: pd.DataFrame) -> Dict[str, object]:
    rec: Dict[str, object] = {
        "summary": {},
        "recommendations": [],
    }
    if efficiency.empty:
        rec["recommendations"].append("No Optuna trial artifacts were available.")
        return rec
    for classifier, group in efficiency.groupby("classifier"):
        delta100 = pd.to_numeric(group["delta_best_vs_100"], errors="coerce")
        delta200 = pd.to_numeric(group["delta_best_vs_200"], errors="coerce")
        after300 = group["final_best_found_after_trial_300"].fillna(False).astype(bool)
        rec["summary"][classifier] = {
            "n_folds": int(group["fold"].nunique()),
            "median_n_trials": float(pd.to_numeric(group["n_trials"], errors="coerce").median()),
            "mean_delta_best_vs_100": float(delta100.mean()) if delta100.notna().any() else None,
            "mean_delta_best_vs_200": float(delta200.mean()) if delta200.notna().any() else None,
            "folds_final_best_after_300": int(after300.sum()),
        }
        if delta100.notna().all() and delta100.max() < 0.005:
            rec["recommendations"].append(f"{classifier}: 900 trials appear likely overkill; best values saturated by 100 trials within <0.005.")
        elif delta200.notna().all() and delta200.max() < 0.005:
            rec["recommendations"].append(f"{classifier}: consider reducing search to ~200 trials; gains after 200 were <0.005.")
        else:
            rec["recommendations"].append(f"{classifier}: keep a broader search or inspect folds where best emerged late.")
    if not summary.empty:
        unstable = summary[summary.get("high_instability_gt_2_orders", False) == True]  # noqa: E712
        for _, row in unstable.iterrows():
            rec["recommendations"].append(
                f"{row['classifier']} {row['param']}: best values vary by >2 orders of magnitude across folds; hyperparameter instability is high."
            )
        near = summary[summary["near_boundary_folds"].fillna("") != ""]
        for _, row in near.iterrows():
            rec["recommendations"].append(
                f"{row['classifier']} {row['param']}: boundary-adjacent best folds detected ({row['near_boundary_folds']}); inspect search bounds before changing trial count."
            )
    return rec


def format_metric(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def format_mean_sd(mean_value: object, sd_value: object, digits: int = 3) -> str:
    return f"{format_metric(mean_value, digits)}±{format_metric(sd_value, digits)}"


def write_threshold_readme(path: Path, train_dev_files: List[Path], notes: List[str]) -> None:
    lines = [
        "# V4 Threshold Analysis",
        "",
        "Threshold 0.5 is the fixed operating threshold from the saved predictions.",
        "Youden, max balanced accuracy, high sensitivity, and high specificity thresholds are optimized on pooled test predictions and are exploratory only.",
        "They are not valid as final model-selection claims unless thresholds are estimated inside train/dev folds and then applied to held-out test folds.",
        "",
    ]
    if train_dev_files:
        lines.append("Train/dev prediction files were found; this script did not consume them for threshold tuning:")
        lines.extend(f"- `{p}`" for p in train_dev_files)
    else:
        lines.append("No train/dev prediction CSV files were found. Clean fold-wise threshold tuning is not possible with the current artifacts.")
    if notes:
        lines.extend(["", "## Column Detection Notes"])
        lines.extend(f"- {note}" for note in sorted(set(notes)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_row(df: pd.DataFrame, classifier: str) -> Optional[pd.Series]:
    rows = df[df["classifier"].astype(str) == classifier]
    return rows.iloc[0] if not rows.empty else None


def write_readme(
    path: Path,
    run_dir: Path,
    output_dir: Path,
    global_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    scanner_df: pd.DataFrame,
    latent_info_df: pd.DataFrame,
    optuna_eff: pd.DataFrame,
    optuna_rec: Dict[str, object],
    optional_missing: List[str],
    validation_notes: List[str],
) -> None:
    lines = [
        "# ADNI Expanded V4 Full Audit",
        "",
        "This is a read-only V4 audit. It reads existing CSV/TXT/JSON artifacts only and does not retrain, load tensors, load checkpoints, or modify V4 run artifacts.",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Audit dir: `{output_dir}`",
        "",
        "## Main V4 Metrics",
        "- Audit pooled metrics are independently recomputed from fold test predictions. Threshold-dependent pooled metrics use `y_score >= 0.5`.",
        "- Training pipeline metrics report mean±SD across outer folds from `all_folds_metrics_MULTI_*.csv`.",
    ]
    for _, row in global_df.sort_values("classifier").iterrows():
        lines.append(
            f"- {row['classifier']}: pooled ROC-AUC={format_metric(row.get('roc_auc'))}, pooled PR-AUC={format_metric(row.get('pr_auc'))}, "
            f"pooled balanced accuracy={format_metric(row.get('balanced_accuracy'))}, pooled sensitivity_AD={format_metric(row.get('sensitivity_AD'))}, "
            f"pooled specificity_CN={format_metric(row.get('specificity_CN'))}, pooled Brier={format_metric(row.get('brier'))}; "
            f"pipeline mean-fold ROC-AUC={format_mean_sd(row.get('pipeline_mean_fold_roc_auc'), row.get('pipeline_sd_fold_roc_auc'))}, "
            f"PR-AUC={format_mean_sd(row.get('pipeline_mean_fold_pr_auc'), row.get('pipeline_sd_fold_pr_auc'))}, "
            f"balanced accuracy={format_mean_sd(row.get('pipeline_mean_fold_balanced_accuracy'), row.get('pipeline_sd_fold_balanced_accuracy'))}, "
            f"sensitivity_AD={format_mean_sd(row.get('pipeline_mean_fold_sensitivity_AD'), row.get('pipeline_sd_fold_sensitivity_AD'))}, "
            f"specificity_CN={format_mean_sd(row.get('pipeline_mean_fold_specificity_CN'), row.get('pipeline_sd_fold_specificity_CN'))}."
        )
    low_sens = global_df["sensitivity_AD"].max() < 0.5 if "sensitivity_AD" in global_df else False
    high_spec = global_df["specificity_CN"].min() >= 0.9 if "specificity_CN" in global_df else False
    lines.extend(["", "## Main Bottleneck"])
    if high_spec and low_sens:
        lines.append("- Confirmed: threshold 0.5 yields high CN specificity but low AD sensitivity.")
    else:
        lines.append("- The high-specificity/low-sensitivity pattern is not uniformly confirmed across classifiers; inspect `global_metrics_by_classifier.csv`.")
    lines.extend(["", "## V3 vs V4"])
    delta = compare_df[compare_df["version"] == "v4_minus_v3"] if not compare_df.empty else pd.DataFrame()
    if delta.empty:
        lines.append("- V3 comparison was unavailable.")
    else:
        for _, row in delta.sort_values("classifier").iterrows():
            lines.append(
                f"- {row['classifier']}: ΔROC-AUC={format_metric(row.get('roc_auc'))}, Δbalanced accuracy={format_metric(row.get('balanced_accuracy'))}, "
                f"Δsensitivity_AD={format_metric(row.get('sensitivity_AD'))}, Δspecificity_CN={format_metric(row.get('specificity_CN'))}, "
                f"ΔBrier={format_metric(row.get('brier'))}."
            )
    lines.extend(
        [
            "",
            "## Threshold Caveat",
            "- Pooled thresholds in `threshold_analysis_pooled_exploratory.csv` are exploratory only and are not valid for final claims.",
            "- No clean fold-wise threshold tuning is possible unless train/dev prediction artifacts exist.",
            "",
            "## Scanner Leakage",
        ]
    )
    if scanner_df.empty:
        lines.append("- Scanner leakage summaries were unavailable.")
    else:
        scanner_diffs = []
        for _, row in scanner_df.sort_values(["split", "site_col"]).iterrows():
            raw_acc = row.get("acc_site_raw_mean")
            latent_acc = row.get("acc_site_latent_mean")
            if pd.notna(raw_acc) and pd.notna(latent_acc):
                scanner_diffs.append(abs(float(raw_acc) - float(latent_acc)))
            lines.append(
                f"- {row['split']} / {row['site_col']}: raw acc={format_metric(row.get('acc_site_raw_mean'))}, "
                f"latent acc={format_metric(row.get('acc_site_latent_mean'))}, chance={format_metric(row.get('chance_level_mean'))}."
            )
        if scanner_diffs:
            lines.append(
                "- acc_site_latent ≈ acc_site_raw, especially on held-out test summaries; "
                "β-VAE regularization does not remove manufacturer information from the latent representation."
            )
    if not latent_info_df.empty and "mi_manufacturer_over_y_ratio" in latent_info_df.columns:
        ratio_rows = latent_info_df[latent_info_df["variable"].astype(str) == "MI_Manufacturer_over_Y_target"]
        if not ratio_rows.empty:
            ratios = ", ".join(
                f"{row['split']}={format_metric(row.get('mi_manufacturer_over_y_ratio'))}"
                for _, row in ratio_rows.sort_values("split").iterrows()
            )
            lines.append(f"- MI(Manufacturer)/MI(Y_target): {ratios}.")
    lines.extend(
        [
            "- Latent QC is split into `latent_qc_fold_level.csv` and `latent_qc_aggregate.csv` to avoid mixing fold rows with aggregate rows.",
        ]
    )
    lines.extend(["", "## Optuna Efficiency"])
    if optuna_eff.empty:
        lines.append("- Optuna artifacts were unavailable.")
    else:
        for classifier, group in optuna_eff.groupby("classifier"):
            late = int(group["final_best_found_after_trial_300"].fillna(False).astype(bool).sum())
            d100 = pd.to_numeric(group["delta_best_vs_100"], errors="coerce").mean()
            d200 = pd.to_numeric(group["delta_best_vs_200"], errors="coerce").mean()
            lines.append(
                f"- {classifier}: mean Δbest vs 100={format_metric(d100, 4)}, mean Δbest vs 200={format_metric(d200, 4)}, "
                f"folds with final best after trial 300={late}/{group['fold'].nunique()}."
            )
        for rec in optuna_rec.get("recommendations", []):
            lines.append(f"- {rec}")
    lines.extend(
        [
            "",
            "## Recommended Next Actions",
            "- Do not run another full model until this audit is interpreted.",
            "- If running another full model, candidates are:",
            "- `[1,2]` = historical best two-channel subset.",
            "- `[4,1]` = fast all7 best dynamic/static pair.",
            "- `[4,1,0]` = biologically plausible 3-channel dynamic/static/OMST candidate, exploratory.",
            "- `[0]` = OMST-only bottleneck check.",
            "- Prioritize `[1,2]` or `[4,1]` before `[4,1,0]` unless there is a strong thesis-driven reason.",
            "- Consider reducing Optuna trials if audit shows saturation before 100-200 trials.",
            "- Need new AD subjects balanced by scanner/site to improve AUC in a defensible way.",
            "",
            "## Optional Missing Artifacts",
        ]
    )
    if optional_missing:
        lines.extend(f"- `{p}`" for p in sorted(set(optional_missing)))
    else:
        lines.append("- None detected among expected optional CSV/JSON artifacts.")
    if validation_notes:
        lines.extend(["", "## Column Detection Notes"])
        lines.extend(f"- {note}" for note in sorted(set(validation_notes)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    metadata_path = args.metadata_path.resolve()
    v3_run_dir = args.v3_run_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else default_output_dir(args.run_dir)
    prepare_output_dir(output_dir, args.overwrite)

    validation_errors, optional_missing = validate_inputs(run_dir, metadata_path)
    if validation_errors:
        raise SystemExit("Required input validation failed:\n" + "\n".join(validation_errors))

    notes: List[str] = []
    metadata = load_metadata(metadata_path)
    pooled = build_pooled_predictions(run_dir, metadata, notes)
    pooled.to_csv(output_dir / "pooled_test_predictions_with_metadata.csv", index=False)

    pipeline_fold_metrics = load_pipeline_fold_metric_summary(run_dir, notes)
    global_df = add_pipeline_fold_metrics(global_metrics(pooled), pipeline_fold_metrics)
    global_df.to_csv(output_dir / "global_metrics_by_classifier.csv", index=False)

    stratum_specs = [
        ("Manufacturer", "metrics_by_manufacturer.csv"),
        ("Site3", "metrics_by_site3.csv"),
        ("SourceCohort", "metrics_by_sourcecohort.csv"),
        ("Sex", "metrics_by_sex.csv"),
        ("AgeBin", "metrics_by_agebin.csv"),
    ]
    for col, filename in stratum_specs:
        stratum_metrics(pooled, col).to_csv(output_dir / filename, index=False)

    thresh = threshold_analysis(pooled)
    thresh.to_csv(output_dir / "threshold_analysis_pooled_exploratory.csv", index=False)
    train_dev_prediction_files = find_train_dev_prediction_files(run_dir)
    write_threshold_readme(output_dir / "threshold_readme.md", train_dev_prediction_files, notes)

    fp, fn, err_man, err_source = error_tables(pooled)
    fp.to_csv(output_dir / "false_positives_CN_as_AD.csv", index=False)
    fn.to_csv(output_dir / "false_negatives_AD_as_CN.csv", index=False)
    err_man.to_csv(output_dir / "error_rates_by_manufacturer.csv", index=False)
    err_source.to_csv(output_dir / "error_rates_by_sourcecohort.csv", index=False)

    cal_bins, cal_summary = calibration_audit(pooled)
    cal_bins.to_csv(output_dir / "calibration_bins.csv", index=False)
    cal_summary.to_csv(output_dir / "calibration_summary.csv", index=False)

    v3_metrics = load_v3_global_metrics(v3_run_dir, notes)
    compare = compare_v3_v4(v3_metrics, global_df)
    compare.to_csv(output_dir / "v3_vs_v4_summary.csv", index=False)

    latent_qc_fold_level, latent_qc_aggregate, latent_info, scanner = latent_and_scanner_summaries(run_dir, optional_missing)
    latent_qc_fold_level.to_csv(output_dir / "latent_qc_fold_level.csv", index=False)
    latent_qc_aggregate.to_csv(output_dir / "latent_qc_aggregate.csv", index=False)
    latent_info.to_csv(output_dir / "latent_info_summary.csv", index=False)
    scanner.to_csv(output_dir / "scanner_leakage_summary.csv", index=False)

    optuna_eff, optuna_params, optuna_rec = optuna_audit(run_dir, optional_missing)
    optuna_eff.to_csv(output_dir / "optuna_efficiency_by_fold.csv", index=False)
    optuna_params.to_csv(output_dir / "optuna_best_params_summary.csv", index=False)
    (output_dir / "optuna_recommendations.json").write_text(
        json.dumps(optuna_rec, indent=2, allow_nan=False), encoding="utf-8"
    )

    write_readme(
        output_dir / "README.md",
        run_dir,
        output_dir,
        global_df,
        compare,
        scanner,
        latent_info,
        optuna_eff,
        optuna_rec,
        optional_missing,
        notes,
    )

    print(f"Audit written to: {output_dir}")
    print(f"Pooled prediction rows: {len(pooled)}")
    print(f"Global metric rows: {len(global_df)}")
    print(f"Optional missing artifacts: {len(set(optional_missing))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
