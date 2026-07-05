#!/usr/bin/env python
"""Read-only OASIS threshold-transfer and calibration audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORING_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_external_scoring"
)
DEFAULT_POSTMORTEM_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_external_scoring_postmortem"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_threshold_transfer_audit"
)
PRIMARY_LEVEL = "ensemble_mean_score_majority_vote"
PRIMARY_STRATEGY = "inner_oof_target_sens_ge_0p70_max_spec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring-dir", type=Path, default=DEFAULT_SCORING_DIR)
    parser.add_argument("--postmortem-dir", type=Path, default=DEFAULT_POSTMORTEM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def write_csv_md(df: pd.DataFrame, csv_path: Path, md_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if df.empty:
            f.write("_No rows._\n")
        else:
            f.write(df.to_markdown(index=False))
            f.write("\n")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    return {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "accuracy": safe_div(tp + tn, len(y)),
    }


def threshold_candidates(scores: Sequence[float]) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    return np.unique(np.round(np.clip(np.concatenate(([0.0, 0.5, 1.0], s)), 0.0, 1.0), 12))


def threshold_grid(y_true: Sequence[int], y_score: Sequence[float]) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    rows = []
    for thr in threshold_candidates(score):
        pred = (score >= thr).astype(int)
        row = {"threshold": float(thr)}
        row.update(binary_metrics(y, score, pred))
        row["youden_j"] = row["sensitivity"] + row["specificity"] - 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def fixed_threshold_summary(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (build_candidate, adni_model), sub in primary.groupby(["build_candidate", "adni_model"], dropna=False):
        row = {
            "build_candidate": build_candidate,
            "adni_model": adni_model,
            "prediction_level": PRIMARY_LEVEL,
            "threshold_strategy": PRIMARY_STRATEGY,
            "threshold_source": "ADNI_inner_CV_OOF_fold_thresholds_transferred_to_OASIS",
            "mean_adni_threshold": float(sub["adni_threshold"].mean()),
            "min_adni_threshold": float(sub["adni_threshold"].min()),
            "max_adni_threshold": float(sub["adni_threshold"].max()),
        }
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows)


def score_threshold_overlap(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (build_candidate, adni_model), sub in primary.groupby(["build_candidate", "adni_model"], dropna=False):
        thr = float(sub["adni_threshold"].mean())
        for diagnosis, grp in sub.groupby("diagnosis", dropna=False):
            q = grp["y_score"].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).to_dict()
            rows.append(
                {
                    "build_candidate": build_candidate,
                    "adni_model": adni_model,
                    "diagnosis": diagnosis,
                    "n": int(len(grp)),
                    "mean_adni_threshold": thr,
                    "score_min": float(q[0.0]),
                    "score_q10": float(q[0.1]),
                    "score_q25": float(q[0.25]),
                    "score_median": float(q[0.5]),
                    "score_q75": float(q[0.75]),
                    "score_q90": float(q[0.9]),
                    "score_max": float(q[1.0]),
                    "fraction_below_adni_threshold": float((grp["y_score"] < thr).mean()),
                    "fraction_above_or_equal_adni_threshold": float((grp["y_score"] >= thr).mean()),
                    "interpretation": "AD below threshold drives false negatives" if str(diagnosis) == "AD_DEMENTIA" else "CN above threshold drives false positives",
                }
            )
    return pd.DataFrame(rows)


def descriptive_thresholds(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (build_candidate, adni_model), sub in primary.groupby(["build_candidate", "adni_model"], dropna=False):
        grid = threshold_grid(sub["y_true"], sub["y_score"])
        picks = []
        picks.append(
            (
                "posthoc_oasis_youden",
                grid.sort_values(["youden_j", "balanced_accuracy", "threshold"], ascending=[False, False, False]).iloc[0],
            )
        )
        picks.append(
            (
                "posthoc_oasis_balanced_accuracy_max",
                grid.sort_values(["balanced_accuracy", "youden_j", "threshold"], ascending=[False, False, False]).iloc[0],
            )
        )
        for target in [0.70, 0.80]:
            eligible = grid[grid["sensitivity"] >= target]
            if eligible.empty:
                row = grid.sort_values(["sensitivity", "specificity", "threshold"], ascending=[False, False, False]).iloc[0]
                label = f"posthoc_oasis_target_sens_ge_{target:.2f}_not_reached_best_available"
            else:
                row = eligible.sort_values(["specificity", "sensitivity", "balanced_accuracy", "threshold"], ascending=[False, False, False, False]).iloc[0]
                label = f"posthoc_oasis_target_sens_ge_{target:.2f}_max_spec"
            picks.append((label, row))
        for label, row in picks:
            out = {
                "build_candidate": build_candidate,
                "adni_model": adni_model,
                "threshold_rule": label,
                "threshold_source": "OASIS_labels_posthoc_descriptive_only",
                "posthoc_descriptive_only": True,
                "not_primary_metric": True,
            }
            out.update(row.to_dict())
            rows.append(out)
    return pd.DataFrame(rows)


def write_recommendation(output_dir: Path) -> None:
    text = """# Calibration Design Recommendation

The OASIS scores show threshold-transfer failure rather than absence of ranking signal. The ADNI-derived thresholds sit high relative to the OASIS AD_DEMENTIA score distribution, so specificity remains high while sensitivity drops.

## Rigorous Designs

1. **Larger independent OASIS calibration/test split.**
   - Split future OASIS into calibration and locked test partitions at subject level.
   - Use the calibration partition only to estimate a threshold or calibration mapping.
   - Report final AUC/PR-AUC and calibrated threshold metrics once on the locked test partition.

2. **Current OASIS as pilot/calibration, future OASIS as locked test.**
   - Treat this 60-subject batch as a pilot showing external ranking and domain-shifted threshold scale.
   - Pre-register the calibration method, then apply it without modification to a future OASIS test batch.

3. **No calibration for primary current result.**
   - For this revision, retain ADNI-derived thresholds for primary operating-point metrics.
   - Present OASIS-derived thresholds only as post-hoc descriptive diagnostics.

## Not Allowed In Current External Validation

- Do not choose the operating threshold on these OASIS labels for primary reporting.
- Do not tune model architecture, classifier, or calibration on OASIS.
- Do not merge OASIS with ADNI training.
"""
    (output_dir / "calibration_design_recommendation.md").write_text(text, encoding="utf-8")


def write_readme(output_dir: Path, fixed: pd.DataFrame, overlap: pd.DataFrame, oracle: pd.DataFrame) -> None:
    text = f"""# OASIS Threshold-Transfer And Calibration Audit

This is a read-only audit. It does not train models, does not alter predictions, and does not select a new primary threshold.

## Fixed ADNI Threshold Performance

{fixed.to_markdown(index=False)}

## Score/Threshold Overlap

{overlap.to_markdown(index=False)}

## Descriptive-Only OASIS Thresholds

These thresholds use OASIS labels and are therefore post-hoc diagnostics only.

{oracle.to_markdown(index=False)}
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def write_final(output_dir: Path, fixed: pd.DataFrame, overlap: pd.DataFrame) -> None:
    ad_rows = overlap[overlap["diagnosis"].astype(str).eq("AD_DEMENTIA")][
        ["build_candidate", "fraction_below_adni_threshold", "score_median", "mean_adni_threshold"]
    ]
    text = f"""# Final Recommendation

Decision: `ranking_signal_present_threshold_requires_independent_calibration`

The OASIS external predictions support a real threshold-independent ranking signal, but the ADNI-derived threshold is too high for the OASIS score scale. This produces high specificity and low sensitivity.

## Fixed Threshold Summary

{fixed.to_markdown(index=False)}

## AD Below ADNI Threshold

{ad_rows.to_markdown(index=False)}

## Recommendation

Keep the current OASIS result as external validation/stress-test with ADNI-derived thresholds. Do not report OASIS-derived oracle thresholds as model performance. If calibrated operating-point performance is needed, collect or reserve an independent OASIS calibration/test split.
"""
    (output_dir / "final_recommendation.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    require(args.scoring_dir / "predictions.csv", "OASIS predictions")
    require(args.postmortem_dir / "primary_metrics_with_ci.csv", "postmortem primary metrics")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.scoring_dir / "predictions.csv")
    primary = pred[pred["prediction_level"].astype(str).eq(PRIMARY_LEVEL)].copy()
    if primary.empty:
        raise ValueError(f"No primary prediction rows found: {PRIMARY_LEVEL}")

    fixed = fixed_threshold_summary(primary)
    overlap = score_threshold_overlap(primary)
    oracle = descriptive_thresholds(primary)

    write_csv_md(fixed, args.output_dir / "fixed_threshold_summary.csv", args.output_dir / "fixed_threshold_summary.md", "Fixed Threshold Summary")
    write_csv_md(overlap, args.output_dir / "score_threshold_overlap.csv", args.output_dir / "score_threshold_overlap.md", "Score Threshold Overlap")
    write_csv_md(oracle, args.output_dir / "descriptive_oracle_thresholds.csv", args.output_dir / "descriptive_oracle_thresholds.md", "Descriptive Oracle Thresholds")
    write_recommendation(args.output_dir)
    write_readme(args.output_dir, fixed, overlap, oracle)
    write_final(args.output_dir, fixed, overlap)

    command_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "scoring_dir": str(args.scoring_dir),
        "postmortem_dir": str(args.postmortem_dir),
        "output_dir": str(args.output_dir),
        "read_only": True,
        "training": False,
        "model_selection": False,
        "prediction_modification": False,
        "primary_prediction_level": PRIMARY_LEVEL,
        "oasis_derived_thresholds_primary": False,
        "outputs": [
            "README.md",
            "fixed_threshold_summary.csv",
            "fixed_threshold_summary.md",
            "score_threshold_overlap.csv",
            "score_threshold_overlap.md",
            "descriptive_oracle_thresholds.csv",
            "descriptive_oracle_thresholds.md",
            "calibration_design_recommendation.md",
            "final_recommendation.md",
            "command_log.json",
        ],
    }
    (args.output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "primary_rows": int(len(primary))}, indent=2))


if __name__ == "__main__":
    main()
