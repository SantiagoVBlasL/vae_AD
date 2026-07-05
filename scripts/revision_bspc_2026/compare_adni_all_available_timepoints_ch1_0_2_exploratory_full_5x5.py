#!/usr/bin/env python3
"""Compare exploratory all-timepoints [1,0,2] FULL 5x5 against locked model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
DEFAULT_CANDIDATE_RUN = RESULTS / "adni_all_available_timepoints_ch1_0_2_exploratory_full_5x5"
DEFAULT_CANDIDATE_READOUT = DEFAULT_CANDIDATE_RUN / "classifier_only_readout"
DEFAULT_REFERENCE_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
DEFAULT_REFERENCE_READOUT = DEFAULT_REFERENCE_RUN / "classifier_only_readout"
DEFAULT_OUTPUT = RESULTS / "adni_all_available_timepoints_ch1_0_2_exploratory_full_5x5_comparison"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FIXED_THRESHOLD = "fixed_0p5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run-dir", type=Path, default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--candidate-readout-dir", type=Path, default=DEFAULT_CANDIDATE_READOUT)
    parser.add_argument("--reference-run-dir", type=Path, default=DEFAULT_REFERENCE_RUN)
    parser.add_argument("--reference-readout-dir", type=Path, default=DEFAULT_REFERENCE_READOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_table(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def required_readout_files(readout: Path) -> list[Path]:
    return [
        readout / "classifier_sweep_pooled_metrics.csv",
        readout / "classifier_sweep_foldwise_metrics.csv",
        readout / "classifier_sweep_predictions.csv",
        readout / "classifier_sweep_thresholds_by_fold.csv",
    ]


def missing_files(readout: Path) -> list[str]:
    return [str(path) for path in required_readout_files(readout) if not path.exists()]


def primary_pooled(readout: Path, run_id: str, label: str) -> dict[str, Any]:
    pooled = pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv")
    row = pooled[
        pooled["model_name"].astype(str).eq(PRIMARY_MODEL)
        & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].iloc[0]
    out = {"run_id": run_id, "label": label, "model_name": PRIMARY_MODEL, "threshold_strategy": PRIMARY_THRESHOLD}
    for col in ["n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "predicted_ad_rate"]:
        out[col] = row.get(col, np.nan)
    return out


def threshold_table(readout: Path, run_id: str, label: str) -> pd.DataFrame:
    pooled = pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv")
    df = pooled[
        pooled["model_name"].astype(str).eq(PRIMARY_MODEL)
        & pooled["threshold_strategy"].astype(str).isin([FIXED_THRESHOLD, PRIMARY_THRESHOLD])
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    return df


def foldwise(readout: Path, run_id: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_foldwise_metrics.csv")
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    return df


def predictions(readout: Path, run_id: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_predictions.csv")
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    return df


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_metrics(df: pd.DataFrame) -> dict[str, Any]:
    y = df["y_true"].astype(int).to_numpy()
    score = df["y_score"].astype(float).to_numpy()
    pred = df["y_pred"].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    return {
        "n": int(len(df)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "sensitivity": sens,
        "specificity": spec,
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "brier": float(brier_score_loss(y, np.clip(score, 0.0, 1.0))),
        "cn_fp_rate": safe_div(fp, fp + tn),
        "ad_fn_rate": safe_div(fn, fn + tp),
    }


def subgroup(pred: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in pred.columns:
        return pd.DataFrame()
    rows = []
    for keys, group in pred.groupby(["run_id", "label", group_col], dropna=False):
        run_id, label, value = keys
        row = {"run_id": run_id, "label": label, group_col: value}
        row.update(binary_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def add_sitecode(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    if "SiteCode" not in out.columns:
        out["SiteCode"] = out["SubjectID"].astype(str).str.slice(0, 3)
    return out


def recommendation(main: pd.DataFrame) -> str:
    ref = main[main["run_id"].eq("locked_v5_1b_140TR")]
    cand = main[main["run_id"].eq("all_available_timepoints")]
    if ref.empty or cand.empty:
        return "Comparison incomplete; candidate readout is missing."
    ref = ref.iloc[0]
    cand = cand.iloc[0]
    passes = (
        float(cand["auc"]) > float(ref["auc"])
        and float(cand["pr_auc"]) >= float(ref["pr_auc"])
        and float(cand["balanced_accuracy"]) >= float(ref["balanced_accuracy"]) - 0.01
        and float(cand["sensitivity"]) >= 0.70
    )
    if passes:
        return (
            "Internal metrics pass the minimum screen, but this remains exploratory: "
            "promotion still requires n_TR/site/manufacturer leakage checks and OASIS external robustness."
        )
    return (
        "Do not promote based on current internal comparison. The branch remains an exploratory "
        "confounding-stress-test sensitivity analysis."
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "candidate_readout": str(args.candidate_readout_dir),
        "reference_readout": str(args.reference_readout_dir),
        "dry_run": bool(args.dry_run),
        "training_launched": False,
    }
    ref_missing = missing_files(args.reference_readout_dir)
    cand_missing = missing_files(args.candidate_readout_dir)
    if args.dry_run:
        status["reference_missing_files"] = ref_missing
        status["candidate_missing_files"] = cand_missing
        status["dry_run_decision"] = "OK: comparison script validated; candidate readout may be absent before training."
        (args.output_dir / "command_log.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    if ref_missing or cand_missing:
        raise FileNotFoundError("Missing readout files: " + json.dumps({"reference": ref_missing, "candidate": cand_missing}, indent=2))

    main_df = pd.DataFrame(
        [
            primary_pooled(args.reference_readout_dir, "locked_v5_1b_140TR", "Locked v5.1b [1,0,2] 140TR"),
            primary_pooled(args.candidate_readout_dir, "all_available_timepoints", "Exploratory all-timepoints [1,0,2]"),
        ]
    )
    ref = main_df[main_df["run_id"].eq("locked_v5_1b_140TR")].iloc[0]
    for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
        main_df[f"delta_vs_locked_{metric}"] = pd.to_numeric(main_df[metric], errors="coerce") - float(ref[metric])
    write_table(main_df, args.output_dir, "main_model_comparison")
    write_table(pd.concat([foldwise(args.reference_readout_dir, "locked_v5_1b_140TR", "Locked"), foldwise(args.candidate_readout_dir, "all_available_timepoints", "AllTR")]), args.output_dir, "foldwise_comparison")
    write_table(pd.concat([threshold_table(args.reference_readout_dir, "locked_v5_1b_140TR", "Locked"), threshold_table(args.candidate_readout_dir, "all_available_timepoints", "AllTR")]), args.output_dir, "threshold_comparison")
    pred = pd.concat([predictions(args.reference_readout_dir, "locked_v5_1b_140TR", "Locked"), predictions(args.candidate_readout_dir, "all_available_timepoints", "AllTR")])
    pred = add_sitecode(pred)
    write_table(subgroup(pred, "Manufacturer"), args.output_dir, "manufacturer_subgroup_comparison")
    write_table(subgroup(pred, "SiteCode"), args.output_dir, "sitecode_subgroup_comparison")
    rec = recommendation(main_df)
    (args.output_dir / "final_recommendation.md").write_text(
        "# Final Recommendation\n\n"
        f"{rec}\n\n"
        "This all-timepoints branch is diagnosis/manufacturer/site n_TR-confounded and must not be promoted based on internal AUC alone.\n",
        encoding="utf-8",
    )
    status["decision"] = rec
    (args.output_dir / "command_log.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
