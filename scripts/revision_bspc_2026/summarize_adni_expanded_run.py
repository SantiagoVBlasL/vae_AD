#!/usr/bin/env python3
"""
Post-training result summarizer for ADNI expanded runs.

Reads fold_*/test_predictions_*.csv from a results directory and computes:
  metrics_by_fold.csv
  metrics_pooled.csv
  confusion_by_classifier.csv
  metrics_by_source_cohort.csv
  metrics_by_manufacturer.csv
  metrics_by_site.csv
  threshold_sweep.csv
  calibration_summary.csv
  README_results_summary.md

If no fold predictions are found, writes empty CSVs with headers and
explains in the README that the run has not completed yet.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        brier_score_loss,
        confusion_matrix,
    )
    from sklearn.calibration import calibration_curve
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize fold predictions from an ADNI expanded run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results-dir", type=Path, required=True,
        help="Path to the run results directory (contains fold_N/ subdirs).",
    )
    p.add_argument(
        "--subject-manifest", type=Path, default=None,
        help="Per-subject manifest CSV for group-level breakdown.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to save summary files. Defaults to results_dir/summary/.",
    )
    p.add_argument(
        "--positive-label", type=str, default="AD",
        help="ResearchGroup_Mapped label for the positive class.",
    )
    p.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Score thresholds for threshold sweep.",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────────────

def discover_fold_predictions(results_dir: Path) -> List[Tuple[int, str, Path]]:
    """
    Return list of (fold_idx, classifier_name, csv_path) for all found predictions.
    Searches for fold_N/test_predictions*.csv patterns.
    """
    found = []
    for fold_dir in sorted(results_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        try:
            fold_idx = int(fold_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        for csv_path in sorted(fold_dir.glob("test_predictions*.csv")):
            # Extract classifier name from filename, e.g. test_predictions_logreg.csv
            stem = csv_path.stem
            clf_name = stem.replace("test_predictions", "").strip("_") or "unknown"
            found.append((fold_idx, clf_name, csv_path))
    return found


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if not SKLEARN_OK:
        return None
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if not SKLEARN_OK:
        return None
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return None


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    positive_label: int = 1,
) -> Dict:
    y_pred = (y_score >= threshold).astype(int)
    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos

    metrics: Dict = {
        "N": n,
        "N_positive": n_pos,
        "N_negative": n_neg,
        "threshold": threshold,
        "roc_auc": safe_roc_auc(y_true, y_score),
        "pr_auc": safe_pr_auc(y_true, y_score),
    }

    if SKLEARN_OK and n > 0:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        if n_pos > 0:
            metrics["brier_score"] = float(brier_score_loss(y_true, y_score))
        else:
            metrics["brier_score"] = None
        if n_pos > 0 and n_neg > 0:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            metrics["sensitivity_AD"] = float(tp / (tp + fn)) if (tp + fn) > 0 else None
            metrics["specificity_CN"] = float(tn / (tn + fp)) if (tn + fp) > 0 else None
        else:
            metrics["sensitivity_AD"] = None
            metrics["specificity_CN"] = None
    return metrics


def load_fold_predictions(csv_path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"WARNING: Cannot read {csv_path}: {e}")
        return None
    return df


def extract_y_arrays(
    df: pd.DataFrame, positive_label: str
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract (y_true_binary, y_score) from a predictions dataframe."""
    # Determine true label column
    true_col = None
    for c in ["y_true", "true_label", "ResearchGroup_Mapped", "label"]:
        if c in df.columns:
            true_col = c
            break
    if true_col is None:
        print("WARNING: Cannot find true label column in predictions CSV.")
        return None

    # Determine score column
    score_col = None
    for c in ["y_score", "score", "y_score_ensemble", "y_score_final", "p_ad", "ad_score"]:
        if c in df.columns:
            score_col = c
            break
    if score_col is None:
        print("WARNING: Cannot find score column in predictions CSV.")
        return None

    y_true_raw = df[true_col].astype(str)
    y_true = (y_true_raw == positive_label).astype(int).to_numpy()
    y_score = pd.to_numeric(df[score_col], errors="coerce").to_numpy()

    valid = ~np.isnan(y_score)
    return y_true[valid], y_score[valid]


# ──────────────────────────────────────────────────────────────────────────────
# Empty-result scaffolding
# ──────────────────────────────────────────────────────────────────────────────

FOLD_METRICS_COLS = [
    "fold", "classifier", "N", "N_positive", "N_negative", "threshold",
    "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "f1",
    "brier_score", "sensitivity_AD", "specificity_CN",
]
GROUP_METRICS_COLS = [
    "group_col", "group_value", "classifier", "N", "N_positive", "N_negative",
    "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "f1",
    "sensitivity_AD", "specificity_CN",
]
THRESHOLD_COLS = [
    "classifier", "threshold", "N", "N_positive", "N_negative",
    "accuracy", "balanced_accuracy", "f1", "sensitivity_AD", "specificity_CN",
]


def empty_df(cols: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=cols)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (args.results_dir / "summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load subject manifest for group breakdowns ────────────────────────────
    manifest: Optional[pd.DataFrame] = None
    if args.subject_manifest and args.subject_manifest.exists():
        manifest = pd.read_csv(args.subject_manifest)
        manifest["SubjectID"] = manifest["SubjectID"].astype(str).str.strip()
        print(f"Manifest loaded: {len(manifest)} subjects")

    # ── Discover fold predictions ─────────────────────────────────────────────
    fold_preds = discover_fold_predictions(args.results_dir)
    print(f"Found {len(fold_preds)} prediction file(s) in {args.results_dir}")

    if not fold_preds:
        print(
            "No fold predictions found. Writing empty summary files. "
            "Run this script again after training completes."
        )
        for fname, cols in [
            ("metrics_by_fold.csv", FOLD_METRICS_COLS),
            ("metrics_pooled.csv", FOLD_METRICS_COLS),
            ("confusion_by_classifier.csv",
             ["classifier", "TN", "FP", "FN", "TP", "N", "N_positive"]),
            ("metrics_by_source_cohort.csv", GROUP_METRICS_COLS),
            ("metrics_by_manufacturer.csv", GROUP_METRICS_COLS),
            ("metrics_by_site.csv", GROUP_METRICS_COLS),
            ("threshold_sweep.csv", THRESHOLD_COLS),
            ("calibration_summary.csv", ["classifier", "fraction_of_positives", "mean_predicted_value"]),
        ]:
            empty_df(cols).to_csv(output_dir / fname, index=False)

        readme = "\n".join([
            "# ADNI Expanded Run Summary",
            "",
            f"Results directory: {args.results_dir}",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "**No fold predictions found.** This directory does not contain completed "
            "training results yet.",
            "Re-run this script after training completes.",
            "",
        ])
        (output_dir / "README_results_summary.md").write_text(readme, encoding="utf-8")
        return 0

    # ── Load and aggregate all fold predictions ───────────────────────────────
    fold_rows: List[Dict] = []
    all_dfs: List[pd.DataFrame] = []  # for pooled + group-level

    for fold_idx, clf_name, csv_path in fold_preds:
        df = load_fold_predictions(csv_path)
        if df is None:
            continue

        df["_fold"] = fold_idx
        df["_classifier"] = clf_name
        if "SubjectID" in df.columns:
            df["SubjectID"] = df["SubjectID"].astype(str).str.strip()

        result = extract_y_arrays(df, args.positive_label)
        if result is None:
            continue
        y_true, y_score = result

        metrics = compute_metrics(y_true, y_score, threshold=0.5)
        fold_rows.append({
            "fold": fold_idx,
            "classifier": clf_name,
            **metrics,
        })
        all_dfs.append(df)

    fold_metrics_df = pd.DataFrame(fold_rows)
    if FOLD_METRICS_COLS:
        existing = [c for c in FOLD_METRICS_COLS if c in fold_metrics_df.columns]
        fold_metrics_df = fold_metrics_df.reindex(columns=FOLD_METRICS_COLS)
    fold_metrics_df.to_csv(output_dir / "metrics_by_fold.csv", index=False)

    # ── Pooled metrics ────────────────────────────────────────────────────────
    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    pooled_rows: List[Dict] = []

    if not combined.empty:
        for clf_name, sub in combined.groupby("_classifier", dropna=False):
            result = extract_y_arrays(sub, args.positive_label)
            if result is None:
                continue
            y_true, y_score = result
            m = compute_metrics(y_true, y_score, threshold=0.5)
            pooled_rows.append({"fold": "pooled", "classifier": clf_name, **m})

    pooled_df = pd.DataFrame(pooled_rows).reindex(columns=FOLD_METRICS_COLS)
    pooled_df.to_csv(output_dir / "metrics_pooled.csv", index=False)

    # ── Confusion matrices ────────────────────────────────────────────────────
    conf_rows: List[Dict] = []
    if not combined.empty and SKLEARN_OK:
        for clf_name, sub in combined.groupby("_classifier", dropna=False):
            result = extract_y_arrays(sub, args.positive_label)
            if result is None:
                continue
            y_true, y_score = result
            y_pred = (y_score >= 0.5).astype(int)
            if len(np.unique(y_true)) >= 2:
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                conf_rows.append({
                    "classifier": clf_name,
                    "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                    "N": int(len(y_true)),
                    "N_positive": int(y_true.sum()),
                })
    pd.DataFrame(conf_rows).to_csv(output_dir / "confusion_by_classifier.csv", index=False)

    # ── Group-level metrics ───────────────────────────────────────────────────
    def group_metrics(combined: pd.DataFrame, manifest: Optional[pd.DataFrame],
                      group_col: str, output_fname: str) -> None:
        rows: List[Dict] = []
        if combined.empty or manifest is None:
            empty_df(GROUP_METRICS_COLS).to_csv(output_dir / output_fname, index=False)
            return
        if group_col not in manifest.columns:
            empty_df(GROUP_METRICS_COLS).to_csv(output_dir / output_fname, index=False)
            return

        merged = combined.merge(
            manifest[["SubjectID", group_col]].drop_duplicates("SubjectID"),
            on="SubjectID", how="left",
        ) if "SubjectID" in combined.columns else combined
        if group_col not in merged.columns:
            empty_df(GROUP_METRICS_COLS).to_csv(output_dir / output_fname, index=False)
            return

        for (clf, grp_val), sub in merged.groupby(["_classifier", group_col], dropna=False):
            result = extract_y_arrays(sub, args.positive_label)
            if result is None:
                continue
            y_true, y_score = result
            m = compute_metrics(y_true, y_score, threshold=0.5)
            rows.append({
                "group_col": group_col,
                "group_value": str(grp_val),
                "classifier": str(clf),
                **{k: v for k, v in m.items() if k in GROUP_METRICS_COLS},
            })
        pd.DataFrame(rows).reindex(columns=GROUP_METRICS_COLS).to_csv(
            output_dir / output_fname, index=False
        )

    group_metrics(combined, manifest, "SourceCohort", "metrics_by_source_cohort.csv")
    group_metrics(combined, manifest, "Manufacturer", "metrics_by_manufacturer.csv")
    group_metrics(combined, manifest, "Site3", "metrics_by_site.csv")

    # ── Threshold sweep ───────────────────────────────────────────────────────
    sweep_rows: List[Dict] = []
    if not combined.empty:
        for clf_name, sub in combined.groupby("_classifier", dropna=False):
            result = extract_y_arrays(sub, args.positive_label)
            if result is None:
                continue
            y_true, y_score = result
            for thr in args.thresholds:
                m = compute_metrics(y_true, y_score, threshold=thr)
                sweep_rows.append({
                    "classifier": clf_name,
                    "threshold": thr,
                    **{k: v for k, v in m.items() if k in THRESHOLD_COLS},
                })
    pd.DataFrame(sweep_rows).reindex(columns=THRESHOLD_COLS).to_csv(
        output_dir / "threshold_sweep.csv", index=False
    )

    # ── Calibration ───────────────────────────────────────────────────────────
    cal_rows: List[Dict] = []
    if not combined.empty and SKLEARN_OK:
        for clf_name, sub in combined.groupby("_classifier", dropna=False):
            result = extract_y_arrays(sub, args.positive_label)
            if result is None:
                continue
            y_true, y_score = result
            if len(np.unique(y_true)) < 2:
                continue
            try:
                frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10)
                for fp, mp in zip(frac_pos, mean_pred):
                    cal_rows.append({
                        "classifier": clf_name,
                        "fraction_of_positives": float(fp),
                        "mean_predicted_value": float(mp),
                    })
            except Exception:
                pass
    pd.DataFrame(cal_rows).to_csv(output_dir / "calibration_summary.csv", index=False)

    # ── README ────────────────────────────────────────────────────────────────
    n_folds = len(set(f for f, _, _ in fold_preds))
    classifiers = sorted(set(c for _, c, _ in fold_preds))
    roc_by_clf = (
        pooled_df.set_index("classifier")["roc_auc"].to_dict()
        if not pooled_df.empty else {}
    )
    readme_lines = [
        "# ADNI Expanded Run Summary",
        "",
        f"Results directory: {args.results_dir}",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Sklearn available: {SKLEARN_OK}",
        "",
        f"## Overview",
        "",
        f"- Folds found: {n_folds}",
        f"- Classifiers: {', '.join(classifiers)}",
        f"- Total prediction rows (pooled): {len(combined)}",
        "",
        "## Pooled ROC-AUC by classifier",
        "",
        *[f"- {c}: {roc_by_clf.get(c, 'N/A'):.4f}" if isinstance(roc_by_clf.get(c), float)
          else f"- {c}: N/A" for c in classifiers],
        "",
        "## Output files",
        "",
        "- metrics_by_fold.csv",
        "- metrics_pooled.csv",
        "- confusion_by_classifier.csv",
        "- metrics_by_source_cohort.csv",
        "- metrics_by_manufacturer.csv",
        "- metrics_by_site.csv",
        "- threshold_sweep.csv",
        "- calibration_summary.csv",
        "",
    ]
    (output_dir / "README_results_summary.md").write_text(
        "\n".join(readme_lines), encoding="utf-8"
    )

    print(f"Summary written to: {output_dir}")
    print(f"Folds: {n_folds}, classifiers: {classifiers}")
    if roc_by_clf:
        for c, v in roc_by_clf.items():
            print(f"  {c} pooled ROC-AUC: {v:.4f}" if isinstance(v, float) else f"  {c}: N/A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
