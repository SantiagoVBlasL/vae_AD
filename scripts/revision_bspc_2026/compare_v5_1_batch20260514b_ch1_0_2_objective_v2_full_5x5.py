#!/usr/bin/env python3
"""Compare objective-v2 FULL [1,0,2] against existing FULL candidates.

Read-only. Filters all readouts to logreg_l2 +
inner_oof_target_sens_ge_0p70_max_spec. The existing [1,0,2] reference is a
multi-model sweep, so this script explicitly filters it to logreg_l2.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"

DEFAULT_OBJECTIVE_RUN_DIR = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_objective_v2_full_5x5"
DEFAULT_OBJECTIVE_READOUT_DIR = DEFAULT_OBJECTIVE_RUN_DIR / "classifier_only_readout"
DEFAULT_CH102_RUN_DIR = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
DEFAULT_CH102_READOUT_DIR = RESULTS / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
DEFAULT_CH1_RUN_DIR = RESULTS / "adni_v5_1_batch20260514b_ch1_full_5x5_candidate"
DEFAULT_CH1_READOUT_DIR = DEFAULT_CH1_RUN_DIR / "classifier_only_readout"
DEFAULT_CH14_RUN_DIR = RESULTS / "adni_v5_1_batch20260514b_ch1_4_full_5x5_candidate"
DEFAULT_CH14_READOUT_DIR = DEFAULT_CH14_RUN_DIR / "classifier_only_readout"
DEFAULT_OUTPUT_DIR = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_objective_v2_full_5x5_comparison"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_METRICS = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--objective-run-dir", type=Path, default=DEFAULT_OBJECTIVE_RUN_DIR)
    parser.add_argument("--objective-readout-dir", type=Path, default=DEFAULT_OBJECTIVE_READOUT_DIR)
    parser.add_argument("--ch102-run-dir", type=Path, default=DEFAULT_CH102_RUN_DIR)
    parser.add_argument("--ch102-readout-dir", type=Path, default=DEFAULT_CH102_READOUT_DIR)
    parser.add_argument("--ch1-run-dir", type=Path, default=DEFAULT_CH1_RUN_DIR)
    parser.add_argument("--ch1-readout-dir", type=Path, default=DEFAULT_CH1_READOUT_DIR)
    parser.add_argument("--ch14-run-dir", type=Path, default=DEFAULT_CH14_RUN_DIR)
    parser.add_argument("--ch14-readout-dir", type=Path, default=DEFAULT_CH14_READOUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def available_readout(readout_dir: Path) -> bool:
    required = [
        "classifier_sweep_pooled_metrics.csv",
        "classifier_sweep_foldwise_metrics.csv",
        "classifier_sweep_predictions.csv",
        "classifier_sweep_thresholds_by_fold.csv",
        "command_log.json",
    ]
    return all((readout_dir / name).exists() for name in required)


def prepare_output_dir(path: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory exists and is not empty; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_No rows._\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals: List[str] = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append("" if pd.isna(val) else f"{float(val):.{digits}f}")
            else:
                vals.append("" if pd.isna(val) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def primary_pooled_row(readout_dir: Path, run_id: str, label: str, channels: str) -> Dict[str, Any]:
    pooled = pd.read_csv(readout_dir / "classifier_sweep_pooled_metrics.csv")
    row = pooled[(pooled["model_name"].eq(PRIMARY_MODEL)) & (pooled["threshold_strategy"].eq(PRIMARY_THRESHOLD))]
    if len(row) != 1:
        raise RuntimeError(f"Expected one primary pooled row for {run_id}, found {len(row)}")
    r = row.iloc[0].to_dict()
    out: Dict[str, Any] = {
        "run_id": run_id,
        "label": label,
        "channels": channels,
        "model_name": PRIMARY_MODEL,
        "threshold_strategy": PRIMARY_THRESHOLD,
    }
    for col in ["n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "accuracy", *PRIMARY_METRICS]:
        out[col] = r.get(col, np.nan)
    return out


def primary_foldwise(readout_dir: Path, run_id: str, channels: str) -> pd.DataFrame:
    df = pd.read_csv(readout_dir / "classifier_sweep_foldwise_metrics.csv")
    df = df[(df["model_name"].eq(PRIMARY_MODEL)) & (df["threshold_strategy"].eq(PRIMARY_THRESHOLD))].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "channels", channels)
    keep = [
        "run_id",
        "channels",
        "fold",
        "threshold",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
        "inner_oof_sensitivity",
        "inner_oof_specificity",
        "inner_oof_balanced_accuracy",
        "threshold_selection_context",
    ]
    return df[[c for c in keep if c in df.columns]].sort_values(["fold"]).reset_index(drop=True)


def subgroup_metrics(readout_dir: Path, run_id: str, channels: str, group_col: str) -> pd.DataFrame:
    pred = pd.read_csv(readout_dir / "classifier_sweep_predictions.csv")
    pred = pred[(pred["model_name"].eq(PRIMARY_MODEL)) & (pred["threshold_strategy"].eq(PRIMARY_THRESHOLD))].copy()
    if group_col not in pred.columns:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for group_value, sub in pred.groupby(group_col, dropna=False):
        y = sub["y_true"].astype(int).to_numpy()
        score = sub["y_score"].astype(float).to_numpy()
        y_pred = sub["y_pred"].astype(int).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        rows.append(
            {
                "run_id": run_id,
                "channels": channels,
                "grouping": group_col,
                "group_value": group_value,
                "n": int(len(sub)),
                "n_cn": int((y == 0).sum()),
                "n_ad": int((y == 1).sum()),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "sensitivity": sens,
                "specificity": spec,
                "balanced_accuracy": np.nanmean([sens, spec]),
                "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
                "auc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
                "pr_auc": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["group_value", "run_id"]).reset_index(drop=True)


def validate_readout(readout_dir: Path, run_id: str, require_single_model: bool) -> Dict[str, Any]:
    log_path = readout_dir / "command_log.json"
    log: Dict[str, Any] = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    requested = log.get("classifiers_requested")
    if require_single_model and requested not in (None, [PRIMARY_MODEL]):
        raise RuntimeError(f"{run_id} readout was not restricted to {PRIMARY_MODEL}: {requested}")
    if isinstance(requested, list) and PRIMARY_MODEL not in requested:
        raise RuntimeError(f"{run_id} readout does not include {PRIMARY_MODEL}: {requested}")
    pooled = pd.read_csv(readout_dir / "classifier_sweep_pooled_metrics.csv")
    primary = pooled[(pooled["model_name"].eq(PRIMARY_MODEL)) & (pooled["threshold_strategy"].eq(PRIMARY_THRESHOLD))]
    if len(primary) != 1:
        raise RuntimeError(f"{run_id}: expected one {PRIMARY_MODEL}/{PRIMARY_THRESHOLD} row, found {len(primary)}")
    for key in ["tensor_modified", "metadata_modified", "ledger_modified"]:
        if log.get(key) not in (False, None):
            raise RuntimeError(f"{run_id}: command log has {key}={log.get(key)}")
    return log


def write_interpretation(output_dir: Path, main: pd.DataFrame, availability: pd.DataFrame) -> None:
    lines = [
        "# Objective-v2 FULL 5x5 [1,0,2] Comparison",
        "",
        "Readout for all rows is `logreg_l2` with `inner_oof_target_sens_ge_0p70_max_spec`.",
        "The threshold is selected from true inner-CV out-of-fold train/dev predictions by the classifier-only readout.",
        "",
        "## Availability",
        "",
        md_table(availability, digits=0),
        "",
        "## Main Comparison",
        "",
        md_table(main, digits=4),
        "",
    ]
    if "objective_v2_ch1_0_2_full_5x5" in set(main["run_id"]):
        by_id = main.set_index("run_id")
        obj = by_id.loc["objective_v2_ch1_0_2_full_5x5"]
        for ref_id, ref_label in [
            ("current_ch1_0_2_full_5x5", "current FULL [1,0,2]"),
            ("ch1_full_5x5", "FULL [1]"),
            ("ch1_4_full_5x5", "FULL [1,4]"),
        ]:
            if ref_id in by_id.index:
                ref = by_id.loc[ref_id]
                lines.append(
                    f"- vs {ref_label}: delta AUC={float(obj['auc']) - float(ref['auc']):+.4f}, "
                    f"delta PR-AUC={float(obj['pr_auc']) - float(ref['pr_auc']):+.4f}, "
                    f"delta BA={float(obj['balanced_accuracy']) - float(ref['balanced_accuracy']):+.4f}."
                )
    else:
        lines.append("- Objective-v2 readout is not available yet; run Stage A/Stage B before interpreting performance.")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- This comparison is read-only.",
            "- It does not train, and does not modify tensors, metadata, or ledger files.",
            "- The existing current FULL [1,0,2] reference is filtered from a multi-model sweep to `logreg_l2` only.",
        ]
    )
    (output_dir / "interpretation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    runs = [
        {
            "run_id": "objective_v2_ch1_0_2_full_5x5",
            "label": "objective_v2 FULL [1,0,2]",
            "channels": "[1,0,2]",
            "run_dir": resolve(args.objective_run_dir),
            "readout_dir": resolve(args.objective_readout_dir),
            "require_single_model": True,
        },
        {
            "run_id": "current_ch1_0_2_full_5x5",
            "label": "current FULL [1,0,2]",
            "channels": "[1,0,2]",
            "run_dir": resolve(args.ch102_run_dir),
            "readout_dir": resolve(args.ch102_readout_dir),
            "require_single_model": False,
        },
        {
            "run_id": "ch1_full_5x5",
            "label": "FULL [1]",
            "channels": "[1]",
            "run_dir": resolve(args.ch1_run_dir),
            "readout_dir": resolve(args.ch1_readout_dir),
            "require_single_model": True,
        },
        {
            "run_id": "ch1_4_full_5x5",
            "label": "FULL [1,4]",
            "channels": "[1,4]",
            "run_dir": resolve(args.ch14_run_dir),
            "readout_dir": resolve(args.ch14_readout_dir),
            "require_single_model": True,
        },
    ]
    availability_rows: List[Dict[str, Any]] = []
    for run in runs:
        readout_dir = Path(run["readout_dir"])
        available = available_readout(readout_dir)
        availability_rows.append(
            {
                "run_id": run["run_id"],
                "label": run["label"],
                "readout_dir": str(readout_dir),
                "readout_available": bool(available),
            }
        )
    availability = pd.DataFrame(availability_rows)

    print("=== Objective-v2 FULL [1,0,2] comparison ===")
    print(availability.to_string(index=False))
    if args.dry_run:
        print("Dry-run complete. No outputs written.")
        return 0

    missing_refs = availability[(~availability["readout_available"]) & (~availability["run_id"].eq("objective_v2_ch1_0_2_full_5x5"))]
    if not missing_refs.empty:
        raise RuntimeError("Reference readouts are missing:\n" + missing_refs.to_string(index=False))
    if not bool(availability.loc[availability["run_id"].eq("objective_v2_ch1_0_2_full_5x5"), "readout_available"].iloc[0]):
        raise RuntimeError("Objective-v2 readout is missing; run Stage A/Stage B before comparison.")

    outdir = resolve(args.output_dir)
    prepare_output_dir(outdir, overwrite=args.overwrite, dry_run=False)

    main_rows: List[Dict[str, Any]] = []
    foldwise: List[pd.DataFrame] = []
    manufacturer: List[pd.DataFrame] = []
    sex: List[pd.DataFrame] = []
    logs: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        readout_dir = Path(run["readout_dir"])
        logs[str(run["run_id"])] = validate_readout(readout_dir, str(run["run_id"]), bool(run["require_single_model"]))
        main_rows.append(primary_pooled_row(readout_dir, str(run["run_id"]), str(run["label"]), str(run["channels"])))
        foldwise.append(primary_foldwise(readout_dir, str(run["run_id"]), str(run["channels"])))
        manufacturer.append(subgroup_metrics(readout_dir, str(run["run_id"]), str(run["channels"]), "Manufacturer"))
        sex.append(subgroup_metrics(readout_dir, str(run["run_id"]), str(run["channels"]), "Sex"))

    main = pd.DataFrame(main_rows)
    main = main.sort_values("auc", ascending=False).reset_index(drop=True)
    foldwise_df = pd.concat(foldwise, ignore_index=True)
    manufacturer_df = pd.concat([df for df in manufacturer if not df.empty], ignore_index=True)
    sex_df = pd.concat([df for df in sex if not df.empty], ignore_index=True)

    main.to_csv(outdir / "main_model_comparison.csv", index=False)
    (outdir / "main_model_comparison.md").write_text(md_table(main), encoding="utf-8")
    foldwise_df.to_csv(outdir / "foldwise_comparison.csv", index=False)
    (outdir / "foldwise_comparison.md").write_text(md_table(foldwise_df), encoding="utf-8")
    manufacturer_df.to_csv(outdir / "manufacturer_subgroup_comparison.csv", index=False)
    (outdir / "manufacturer_subgroup_comparison.md").write_text(md_table(manufacturer_df), encoding="utf-8")
    sex_df.to_csv(outdir / "sex_subgroup_comparison.csv", index=False)
    (outdir / "sex_subgroup_comparison.md").write_text(md_table(sex_df), encoding="utf-8")
    availability.to_csv(outdir / "readout_availability.csv", index=False)
    write_interpretation(outdir, main, availability)
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "model_filter": PRIMARY_MODEL,
        "threshold_filter": PRIMARY_THRESHOLD,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "readout_logs": logs,
    }
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote comparison outputs to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
