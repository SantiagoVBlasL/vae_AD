#!/usr/bin/env python3
"""Fold-wise threshold audit for ADNI expanded v3 predictions.

This script never retrains models. It reads existing per-fold prediction CSVs,
estimates pooled exploratory thresholds from test predictions, and, only when
available, estimates fold-specific thresholds from train/dev prediction files.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "results/revision_bspc_2026/adni_expanded_v3_beta25_static3"
DEFAULT_OUTDIR_NAME = "foldwise_threshold_audit"

OUTPUT_FILES = [
    "threshold_metrics_pooled.csv",
    "threshold_metrics_by_fold.csv",
    "threshold_readme.md",
]


class ColumnDetectionError(ValueError):
    pass


@dataclass
class FoldPredictionSet:
    fold: int
    classifier: str
    test_path: Path
    test_df: pd.DataFrame
    train_dev_paths: List[Path]
    train_dev_df: Optional[pd.DataFrame]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a threshold audit from existing ADNI expanded v3 fold predictions."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def lower_map(columns: Iterable[str]) -> Dict[str, str]:
    return {str(col).strip().lower(): col for col in columns}


def detect_column(
    df: pd.DataFrame,
    role: str,
    preferred_names: Sequence[str],
    fuzzy_tokens: Sequence[str] = (),
) -> str:
    by_lower = lower_map(df.columns)
    exact = [by_lower[name.lower()] for name in preferred_names if name.lower() in by_lower]
    exact = list(dict.fromkeys(exact))
    if exact:
        return exact[0]

    matches = []
    for col in df.columns:
        col_l = str(col).lower()
        if all(token in col_l for token in fuzzy_tokens):
            matches.append(col)
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ColumnDetectionError(
            f"Ambiguous {role} column candidates: {matches}\nAvailable columns: {list(df.columns)}"
        )
    raise ColumnDetectionError(f"Could not detect {role} column. Available columns: {list(df.columns)}")


def detect_score_column(df: pd.DataFrame) -> str:
    by_lower = lower_map(df.columns)
    preferred = [
        "y_score_final",
        "prob_ad",
        "probability_ad",
        "ad_probability",
        "p_ad",
        "pred_proba_ad",
        "y_score",
        "score_ad",
    ]
    for name in preferred:
        if name in by_lower:
            return by_lower[name]
    candidates = []
    for col in df.columns:
        col_l = str(col).lower()
        if ("score" in col_l or "prob" in col_l or "proba" in col_l) and "cn" not in col_l:
            candidates.append(col)
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ColumnDetectionError(
            f"Ambiguous y_score/probability column candidates: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
    raise ColumnDetectionError(f"Could not detect y_score/probability column. Columns: {list(df.columns)}")


def detect_prediction_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {
        "subject_id": detect_column(
            df, "SubjectID", ["SubjectID", "subject_id", "subject", "PTID"], ("subject",)
        ),
        "y_true": detect_column(df, "y_true", ["y_true", "true_label", "label", "target"], ("true",)),
        "y_score": detect_score_column(df),
        "y_pred": detect_column(df, "y_pred", ["y_pred", "pred", "prediction", "predicted_label"], ("pred",)),
    }


def to_binary(series: pd.Series, role: str) -> pd.Series:
    def convert(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, np.integer, float, np.floating)) and float(value) in (0.0, 1.0):
            return int(value)
        text = str(value).strip().upper()
        mapping = {
            "0": 0,
            "0.0": 0,
            "CN": 0,
            "CONTROL": 0,
            "NORMAL": 0,
            "1": 1,
            "1.0": 1,
            "AD": 1,
            "DEMENTIA": 1,
            "ALZHEIMER": 1,
        }
        return mapping.get(text, np.nan)

    converted = series.map(convert)
    if converted.isna().any():
        bad = sorted(series[converted.isna()].dropna().astype(str).unique().tolist())
        raise ValueError(f"Could not map {role} to CN=0/AD=1. Bad values: {bad}")
    return converted.astype(int)


def normalize_predictions(df: pd.DataFrame, classifier: str, fold: int, source_path: Path) -> pd.DataFrame:
    columns = detect_prediction_columns(df)
    out = pd.DataFrame(
        {
            "SubjectID": df[columns["subject_id"]].astype(str).str.strip(),
            "y_true": to_binary(df[columns["y_true"]], "y_true"),
            "y_score": pd.to_numeric(df[columns["y_score"]], errors="coerce"),
            "y_pred_original": to_binary(df[columns["y_pred"]], "y_pred"),
            "classifier": classifier,
            "fold": fold,
            "source_path": str(source_path),
        }
    )
    if out["y_score"].isna().any() or not np.isfinite(out["y_score"]).all():
        raise ValueError(f"Non-finite y_score values in {source_path}")
    return out


def classifier_from_filename(path: Path) -> str:
    match = re.search(r"test_predictions_(.+)\.csv$", path.name)
    if match:
        return match.group(1)
    return path.stem.replace("test_predictions_", "")


def fold_from_path(path: Path) -> int:
    match = re.search(r"fold_(\d+)", str(path.parent))
    if not match:
        raise ValueError(f"Could not infer fold number from {path}")
    return int(match.group(1))


def find_train_dev_prediction_files(fold_dir: Path, classifier: str, test_path: Path) -> List[Path]:
    files = []
    for path in sorted(fold_dir.glob("*.csv")):
        if path == test_path:
            continue
        stem = path.stem.lower()
        if "prediction" not in stem:
            continue
        if classifier.lower() not in stem:
            continue
        if "test" in stem:
            continue
        if any(token in stem for token in ["train", "dev", "val", "valid", "internal"]):
            files.append(path)
    return files


def load_fold_prediction_sets(run_dir: Path) -> List[FoldPredictionSet]:
    test_paths = sorted(run_dir.glob("fold_*/test_predictions_*.csv"))
    if not test_paths:
        raise FileNotFoundError(f"No fold_*/test_predictions_*.csv files found under {run_dir}")
    sets = []
    for test_path in test_paths:
        fold = fold_from_path(test_path)
        classifier = classifier_from_filename(test_path)
        test_raw = pd.read_csv(test_path)
        test_df = normalize_predictions(test_raw, classifier, fold, test_path)
        train_dev_paths = find_train_dev_prediction_files(test_path.parent, classifier, test_path)
        train_dev_df = None
        if train_dev_paths:
            frames = []
            for path in train_dev_paths:
                frames.append(normalize_predictions(pd.read_csv(path), classifier, fold, path))
            train_dev_df = pd.concat(frames, ignore_index=True)
        sets.append(
            FoldPredictionSet(
                fold=fold,
                classifier=classifier,
                test_path=test_path,
                test_df=test_df,
                train_dev_paths=train_dev_paths,
                train_dev_df=train_dev_df,
            )
        )
    return sets


def confusion_values(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def safe_div(num: float, den: float) -> float:
    return np.nan if den == 0 else num / den


def evaluate_threshold(df: pd.DataFrame, threshold: float) -> Dict[str, object]:
    y_true = df["y_true"].astype(int).to_numpy()
    y_score = df["y_score"].astype(float).to_numpy()
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_values(y_true, y_pred)
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    has_both_classes = (y_true == 0).any() and (y_true == 1).any()
    return {
        "n": int(len(df)),
        "n_CN": int((y_true == 0).sum()),
        "n_AD": int((y_true == 1).sum()),
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if has_both_classes else np.nan,
        "sensitivity_AD": sensitivity,
        "specificity_CN": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if has_both_classes else np.nan,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores.astype(float))
    mids = (unique[:-1] + unique[1:]) / 2.0 if len(unique) > 1 else np.array([], dtype=float)
    values = np.concatenate([np.array([0.0, 0.5, 1.0]), unique, mids])
    return np.unique(np.clip(values[np.isfinite(values)], 0.0, 1.0))


def optimize_threshold(df: pd.DataFrame, objective: str) -> float:
    rows = []
    for threshold in threshold_grid(df["y_score"].to_numpy()):
        row = evaluate_threshold(df, float(threshold))
        row["candidate_threshold"] = float(threshold)
        row["youden"] = row["sensitivity_AD"] + row["specificity_CN"] - 1.0
        rows.append(row)
    table = pd.DataFrame(rows)
    if objective == "youden":
        table = table.sort_values(
            ["youden", "balanced_accuracy", "f1", "candidate_threshold"],
            ascending=[False, False, False, True],
        )
    elif objective == "balanced_accuracy":
        table = table.sort_values(
            ["balanced_accuracy", "f1", "candidate_threshold"],
            ascending=[False, False, True],
        )
    else:
        raise ValueError(f"Unknown threshold objective: {objective}")
    return float(table.iloc[0]["candidate_threshold"])


def add_metric_row(
    rows: List[Dict[str, object]],
    df: pd.DataFrame,
    threshold: float,
    classifier: str,
    strategy: str,
    analysis_type: str,
    threshold_source: str,
    is_exploratory: bool,
    foldwise_valid: bool,
    fold: Optional[int] = None,
) -> None:
    row = {
        "classifier": classifier,
        "fold": fold,
        "strategy": strategy,
        "analysis_type": analysis_type,
        "threshold_source": threshold_source,
        "is_exploratory": bool(is_exploratory),
        "foldwise_valid": bool(foldwise_valid),
    }
    row.update(evaluate_threshold(df, threshold))
    rows.append(row)


def build_metrics(sets: List[FoldPredictionSet]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    pooled_rows: List[Dict[str, object]] = []
    by_fold_rows: List[Dict[str, object]] = []
    pooled_thresholds: Dict[str, float] = {}
    all_test = pd.concat([item.test_df for item in sets], ignore_index=True)

    for classifier, group in all_test.groupby("classifier", sort=True):
        add_metric_row(
            pooled_rows,
            group,
            0.5,
            classifier,
            "threshold_0.5",
            "pooled_test_fixed_threshold",
            "fixed_0.5",
            is_exploratory=False,
            foldwise_valid=False,
        )
        youden = optimize_threshold(group, "youden")
        max_ba = optimize_threshold(group, "balanced_accuracy")
        pooled_thresholds[f"{classifier}:pooled_youden"] = youden
        pooled_thresholds[f"{classifier}:pooled_max_balanced_accuracy"] = max_ba
        add_metric_row(
            pooled_rows,
            group,
            youden,
            classifier,
            "pooled_youden_exploratory",
            "pooled_test_threshold_optimized_on_same_predictions",
            "pooled_test_predictions",
            is_exploratory=True,
            foldwise_valid=False,
        )
        add_metric_row(
            pooled_rows,
            group,
            max_ba,
            classifier,
            "pooled_max_balanced_accuracy_exploratory",
            "pooled_test_threshold_optimized_on_same_predictions",
            "pooled_test_predictions",
            is_exploratory=True,
            foldwise_valid=False,
        )

    for item in sets:
        add_metric_row(
            by_fold_rows,
            item.test_df,
            0.5,
            item.classifier,
            "threshold_0.5",
            "fold_test_fixed_threshold",
            "fixed_0.5",
            is_exploratory=False,
            foldwise_valid=False,
            fold=item.fold,
        )
        for key_suffix, strategy in [
            ("pooled_youden", "pooled_youden_exploratory_applied_to_fold"),
            ("pooled_max_balanced_accuracy", "pooled_max_balanced_accuracy_exploratory_applied_to_fold"),
        ]:
            threshold = pooled_thresholds[f"{item.classifier}:{key_suffix}"]
            add_metric_row(
                by_fold_rows,
                item.test_df,
                threshold,
                item.classifier,
                strategy,
                "fold_test_using_pooled_test_threshold",
                "pooled_test_predictions",
                is_exploratory=True,
                foldwise_valid=False,
                fold=item.fold,
            )

        if item.train_dev_df is not None and not item.train_dev_df.empty:
            train_dev_youden = optimize_threshold(item.train_dev_df, "youden")
            train_dev_ba = optimize_threshold(item.train_dev_df, "balanced_accuracy")
            add_metric_row(
                by_fold_rows,
                item.test_df,
                train_dev_youden,
                item.classifier,
                "fold_train_dev_youden_applied_to_test",
                "fold_test_using_train_dev_threshold",
                ";".join(str(path) for path in item.train_dev_paths),
                is_exploratory=False,
                foldwise_valid=True,
                fold=item.fold,
            )
            add_metric_row(
                by_fold_rows,
                item.test_df,
                train_dev_ba,
                item.classifier,
                "fold_train_dev_max_balanced_accuracy_applied_to_test",
                "fold_test_using_train_dev_threshold",
                ";".join(str(path) for path in item.train_dev_paths),
                is_exploratory=False,
                foldwise_valid=True,
                fold=item.fold,
            )

    pooled = pd.DataFrame(pooled_rows)
    by_fold = pd.DataFrame(by_fold_rows)
    return pooled, by_fold, pooled_thresholds


def prepare_outdir(outdir: Path, overwrite: bool) -> None:
    if outdir.exists() and not outdir.is_dir():
        raise FileExistsError(f"Output path exists but is not a directory: {outdir}")
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=False)
        return
    existing = [outdir / name for name in OUTPUT_FILES if (outdir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing threshold audit files. Use --overwrite if intentional:\n"
            + "\n".join(str(path) for path in existing)
        )


def format_metric(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}"
    return str(value)


def best_improvement_lines(pooled: pd.DataFrame) -> List[str]:
    lines = []
    for classifier, group in pooled.groupby("classifier", sort=True):
        base = group[group["strategy"] == "threshold_0.5"].iloc[0]
        candidates = group[group["strategy"] != "threshold_0.5"].copy()
        if candidates.empty:
            continue
        candidates["delta_sensitivity"] = candidates["sensitivity_AD"] - base["sensitivity_AD"]
        candidates["delta_balanced_accuracy"] = candidates["balanced_accuracy"] - base["balanced_accuracy"]
        sens_best = candidates.sort_values(
            ["delta_sensitivity", "balanced_accuracy"], ascending=[False, False]
        ).iloc[0]
        ba_best = candidates.sort_values(
            ["delta_balanced_accuracy", "sensitivity_AD"], ascending=[False, False]
        ).iloc[0]
        lines.append(
            f"- {classifier}: sensibilidad mejora mas con `{sens_best['strategy']}` "
            f"(threshold={format_metric(sens_best['threshold'])}, "
            f"sensitivity_AD={format_metric(sens_best['sensitivity_AD'])}, "
            f"delta={format_metric(sens_best['delta_sensitivity'])})."
        )
        lines.append(
            f"- {classifier}: balanced accuracy mejora mas con `{ba_best['strategy']}` "
            f"(threshold={format_metric(ba_best['threshold'])}, "
            f"balanced_accuracy={format_metric(ba_best['balanced_accuracy'])}, "
            f"delta={format_metric(ba_best['delta_balanced_accuracy'])})."
        )
    return lines


def write_readme(
    outdir: Path,
    run_dir: Path,
    sets: List[FoldPredictionSet],
    pooled: pd.DataFrame,
    by_fold: pd.DataFrame,
) -> None:
    any_train_dev = any(item.train_dev_df is not None and not item.train_dev_df.empty for item in sets)
    train_dev_files = [
        str(path)
        for item in sets
        for path in item.train_dev_paths
    ]
    lines = [
        "# V3 Fold-wise Threshold Audit",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Scope",
        "- No model was retrained.",
        "- Test predictions were read from `fold_*/test_predictions_logreg.csv` and `fold_*/test_predictions_svm.csv`.",
        "- Threshold 0.5 is a fixed-threshold descriptive test evaluation.",
        "- Pooled Youden and pooled max balanced accuracy are exploratory because the threshold is optimized on pooled test predictions.",
        "",
        "## Fold-wise Validity",
    ]
    if any_train_dev:
        lines.extend(
            [
                "- Train/dev prediction files were found, so fold-specific train/dev thresholds were estimated and applied to each test fold.",
                "- Rows with `foldwise_valid=True` in `threshold_metrics_by_fold.csv` are the clean fold-wise threshold analysis.",
                "",
                "Train/dev prediction files used:",
            ]
        )
        lines.extend(f"- `{path}`" for path in train_dev_files)
    else:
        lines.extend(
            [
                "- No train/dev or internal validation prediction files were found in the current artifacts.",
                "- A clean fold-wise threshold tuning analysis is therefore not possible from this run alone.",
                "- `threshold_metrics_by_fold.csv` contains fixed 0.5 fold metrics and pooled-test exploratory thresholds applied per fold only as descriptive diagnostics.",
                "- Rows with `foldwise_valid=False` must not be reported as unbiased fold-wise threshold tuning.",
            ]
        )
    lines.extend(["", "## Pooled Exploratory Thresholds"])
    for _, row in pooled.sort_values(["classifier", "strategy"]).iterrows():
        lines.append(
            f"- {row['classifier']} / {row['strategy']}: threshold={format_metric(row['threshold'])}, "
            f"sensitivity_AD={format_metric(row['sensitivity_AD'])}, "
            f"specificity_CN={format_metric(row['specificity_CN'])}, "
            f"balanced_accuracy={format_metric(row['balanced_accuracy'])}, "
            f"F1={format_metric(row['f1'])}."
        )
    lines.extend(["", "## Thresholds That Improve Sensitivity Or Balanced Accuracy"])
    lines.extend(best_improvement_lines(pooled))
    if any_train_dev:
        valid = by_fold[by_fold["foldwise_valid"] == True]  # noqa: E712
        if not valid.empty:
            lines.extend(["", "## Valid Fold-wise Threshold Summary"])
            summary = (
                valid.groupby(["classifier", "strategy"], as_index=False)[
                    ["sensitivity_AD", "specificity_CN", "balanced_accuracy", "f1"]
                ]
                .mean(numeric_only=True)
                .sort_values(["classifier", "strategy"])
            )
            for _, row in summary.iterrows():
                lines.append(
                    f"- {row['classifier']} / {row['strategy']}: mean sensitivity_AD={format_metric(row['sensitivity_AD'])}, "
                    f"mean specificity_CN={format_metric(row['specificity_CN'])}, "
                    f"mean balanced_accuracy={format_metric(row['balanced_accuracy'])}, "
                    f"mean F1={format_metric(row['f1'])}."
                )
    lines.extend(
        [
            "",
            "## Output Files",
            "- `threshold_metrics_pooled.csv`",
            "- `threshold_metrics_by_fold.csv`",
            "- `threshold_readme.md`",
            "",
        ]
    )
    (outdir / "threshold_readme.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    outdir = args.outdir.resolve() if args.outdir else run_dir / DEFAULT_OUTDIR_NAME
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    prepare_outdir(outdir, args.overwrite)
    try:
        sets = load_fold_prediction_sets(run_dir)
    except ColumnDetectionError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    pooled, by_fold, _ = build_metrics(sets)
    pooled.to_csv(outdir / "threshold_metrics_pooled.csv", index=False)
    by_fold.to_csv(outdir / "threshold_metrics_by_fold.csv", index=False)
    write_readme(outdir, run_dir, sets, pooled, by_fold)

    print(f"Threshold audit written to: {outdir}")
    print(f"Fold/classifier test files: {len(sets)}")
    train_dev_count = sum(len(item.train_dev_paths) for item in sets)
    if train_dev_count:
        print(f"Train/dev prediction files found: {train_dev_count}")
    else:
        print("Train/dev prediction files found: 0")
        print("Clean fold-wise threshold tuning is not possible with current artifacts.")
    print("\nPooled threshold metrics:")
    print(pooled.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
