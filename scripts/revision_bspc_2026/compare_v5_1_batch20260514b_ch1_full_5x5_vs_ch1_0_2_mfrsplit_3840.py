#!/usr/bin/env python3
"""Compare the channel-[1] full 5x5 candidate readout against current [1,0,2] references.

This script is read-only. It expects classifier-only readout outputs for the
candidate once the full VAE run has completed; until then --dry-run reports
which inputs are still pending.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"

DEFAULT_CANDIDATE_RUN_DIR = LOCAL_RESULTS / "adni_v5_1_batch20260514b_ch1_full_5x5_candidate"
DEFAULT_CANDIDATE_READOUT_DIR = DEFAULT_CANDIDATE_RUN_DIR / "classifier_only_readout"
DEFAULT_REFERENCE_RUN_DIR = LOCAL_RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
DEFAULT_REFERENCE_READOUT_DIR = LOCAL_RESULTS / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
DEFAULT_THRESHOLD_AUDIT_DIR = LOCAL_RESULTS / "adni_v5_1_batch20260514b_threshold_final_audit"
DEFAULT_OUTPUT_DIR = LOCAL_RESULTS / "adni_v5_1_batch20260514b_ch1_full_5x5_candidate_comparison"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run-dir", type=Path, default=DEFAULT_CANDIDATE_RUN_DIR)
    parser.add_argument("--candidate-readout-dir", type=Path, default=DEFAULT_CANDIDATE_READOUT_DIR)
    parser.add_argument("--reference-run-dir", type=Path, default=DEFAULT_REFERENCE_RUN_DIR)
    parser.add_argument("--reference-readout-dir", type=Path, default=DEFAULT_REFERENCE_READOUT_DIR)
    parser.add_argument("--threshold-audit-dir", type=Path, default=DEFAULT_THRESHOLD_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json_optional(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def available_readout(readout_dir: Path) -> bool:
    required = [
        "classifier_sweep_pooled_metrics.csv",
        "classifier_sweep_foldwise_metrics.csv",
        "classifier_sweep_predictions.csv",
        "classifier_sweep_thresholds_by_fold.csv",
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


def load_primary_from_readout(readout_dir: Path, run_id: str, label: str, channels: List[int]) -> Dict[str, Any]:
    pooled = pd.read_csv(readout_dir / "classifier_sweep_pooled_metrics.csv")
    row = pooled[(pooled["model_name"].eq(PRIMARY_MODEL)) & (pooled["threshold_strategy"].eq(PRIMARY_THRESHOLD))]
    if row.empty:
        raise RuntimeError(f"Missing primary pooled row in {readout_dir}")
    r = row.iloc[0].to_dict()
    out = {
        "run_id": run_id,
        "label": label,
        "channels": str(channels),
        "readout_dir": str(readout_dir),
        "model_name": PRIMARY_MODEL,
        "threshold_strategy": PRIMARY_THRESHOLD,
    }
    for col in ["auc", "pr_auc", "accuracy", "sensitivity", "specificity", "balanced_accuracy", "f1", "tn", "fp", "fn", "tp"]:
        out[col] = r.get(col, np.nan)
    return out


def load_primary_from_threshold_audit(audit_dir: Path) -> Optional[Dict[str, Any]]:
    path = audit_dir / "logreg_l2_threshold_comparison.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    row = table[(table["model_name"].eq(PRIMARY_MODEL)) & (table["threshold_strategy"].eq(PRIMARY_THRESHOLD))]
    if row.empty:
        return None
    r = row.iloc[0].to_dict()
    out = {
        "run_id": "ch1_0_2_threshold_final_audit",
        "label": "current full [1,0,2] threshold final audit",
        "channels": "[1,0,2]",
        "readout_dir": str(audit_dir),
        "model_name": PRIMARY_MODEL,
        "threshold_strategy": PRIMARY_THRESHOLD,
    }
    for col in ["auc", "pr_auc", "accuracy", "sensitivity", "specificity", "balanced_accuracy", "f1", "tn", "fp", "fn", "tp"]:
        out[col] = r.get(col, np.nan)
    return out


def foldwise_primary(readout_dir: Path, run_id: str) -> pd.DataFrame:
    path = readout_dir / "classifier_sweep_foldwise_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[(df["model_name"].eq(PRIMARY_MODEL)) & (df["threshold_strategy"].eq(PRIMARY_THRESHOLD))].copy()
    df.insert(0, "run_id", run_id)
    return df


def subgroup_from_predictions(readout_dir: Path, run_id: str) -> pd.DataFrame:
    path = readout_dir / "classifier_sweep_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    pred = pd.read_csv(path)
    pred = pred[(pred["model_name"].eq(PRIMARY_MODEL)) & (pred["threshold_strategy"].eq(PRIMARY_THRESHOLD))].copy()
    rows: List[Dict[str, Any]] = []
    for manufacturer, sub in pred.groupby("Manufacturer", dropna=False):
        y = sub["y_true"].astype(int).to_numpy()
        score = sub["y_score"].astype(float).to_numpy()
        y_pred = sub["y_pred"].astype(int).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        row: Dict[str, Any] = {
            "run_id": run_id,
            "Manufacturer": manufacturer,
            "n": int(len(sub)),
            "n_cn": int((y == 0).sum()),
            "n_ad": int((y == 1).sum()),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
            "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
            "accuracy": float((tp + tn) / len(sub)) if len(sub) else np.nan,
            "f1": float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else np.nan,
        }
        row["balanced_accuracy"] = np.nanmean([row["sensitivity"], row["specificity"]])
        row["auc"] = roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan
        row["pr_auc"] = average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def weak_fold_summary(foldwise: pd.DataFrame) -> pd.DataFrame:
    if foldwise.empty:
        return pd.DataFrame()
    rows = []
    for run_id, sub in foldwise.groupby("run_id"):
        sub = sub.sort_values("auc", ascending=True)
        row = sub.iloc[0].to_dict()
        rows.append(
            {
                "run_id": run_id,
                "weak_fold": int(row["fold"]),
                "weak_fold_auc": row.get("auc", np.nan),
                "weak_fold_pr_auc": row.get("pr_auc", np.nan),
                "weak_fold_sensitivity": row.get("sensitivity", np.nan),
                "weak_fold_specificity": row.get("specificity", np.nan),
                "weak_fold_balanced_accuracy": row.get("balanced_accuracy", np.nan),
                "weak_fold_f1": row.get("f1", np.nan),
            }
        )
    return pd.DataFrame(rows)


def load_run_config_summary(run_dir: Path) -> Dict[str, Any]:
    cfg = read_json_optional(run_dir / "run_config.json") or read_json_optional(run_dir / "run_manifest.json") or {}
    args = cfg.get("args", {})
    return {
        "run_dir": str(run_dir),
        "channels_to_use": args.get("channels_to_use") or cfg.get("selected_channels") or cfg.get("channels_to_use_indices"),
        "outer_folds": args.get("outer_folds") or cfg.get("outer_folds"),
        "inner_folds": args.get("inner_folds") or cfg.get("inner_folds"),
        "epochs_vae": args.get("epochs_vae") or cfg.get("epochs_vae"),
        "cyclical_beta_n_cycles": args.get("cyclical_beta_n_cycles") or cfg.get("cyclical_beta_n_cycles"),
        "lr_scheduler_T0": args.get("lr_scheduler_T0") or cfg.get("lr_scheduler_T0"),
        "classifier_stratify_cols": args.get("classifier_stratify_cols"),
        "vae_stratify_cols": args.get("vae_stratify_cols"),
        "python_bandpass_applied": cfg.get("python_bandpass_applied", False),
    }


def write_readme(output_dir: Path, availability: pd.DataFrame, primary: pd.DataFrame, weak: pd.DataFrame) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    vals.append("" if pd.isna(val) else f"{val:.4f}")
                else:
                    vals.append("" if pd.isna(val) else str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# Channel [1] Full 5x5 Candidate Comparison",
        "",
        "This is a read-only comparison. It does not train, modify tensors, metadata, or ledger files.",
        "",
        f"Primary readout: `{PRIMARY_MODEL}` + `{PRIMARY_THRESHOLD}`.",
        "",
        "## Availability",
        "",
        markdown_table(availability),
        "",
        "## Primary Metrics",
        "",
    ]
    if primary.empty:
        lines.append("Candidate/readout metrics are not available yet.")
    else:
        cols = ["run_id", "channels", "auc", "pr_auc", "sensitivity", "specificity", "balanced_accuracy", "f1"]
        lines.append(markdown_table(primary[cols]))
    lines += ["", "## Weak Fold", ""]
    if weak.empty:
        lines.append("No weak-fold summary available yet.")
    else:
        lines.append(markdown_table(weak))
    lines += [
        "",
        "## Outputs",
        "",
        "- `primary_metric_comparison.csv`",
        "- `manufacturer_subgroup_comparison.csv`",
        "- `foldwise_primary_metrics.csv`",
        "- `weak_fold_summary.csv`",
        "- `run_config_summary.csv`",
        "- `command_log.json`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    candidate_run = resolve(args.candidate_run_dir)
    candidate_readout = resolve(args.candidate_readout_dir)
    reference_run = resolve(args.reference_run_dir)
    reference_readout = resolve(args.reference_readout_dir)
    threshold_audit = resolve(args.threshold_audit_dir)
    output_dir = resolve(args.output_dir)

    availability = pd.DataFrame(
        [
            {
                "run_id": "ch1_full_5x5_candidate",
                "run_dir": str(candidate_run),
                "readout_dir": str(candidate_readout),
                "readout_available": available_readout(candidate_readout),
                "role": "candidate",
            },
            {
                "run_id": "ch1_0_2_mfrsplit_3840",
                "run_dir": str(reference_run),
                "readout_dir": str(reference_readout),
                "readout_available": available_readout(reference_readout),
                "role": "reference",
            },
            {
                "run_id": "ch1_0_2_threshold_final_audit",
                "run_dir": str(reference_run),
                "readout_dir": str(threshold_audit),
                "readout_available": (threshold_audit / "logreg_l2_threshold_comparison.csv").exists(),
                "role": "reference_audit",
            },
        ]
    )

    print("=== ch1 full 5x5 candidate comparison ===")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'WRITE OUTPUTS'}")
    print(availability.to_string(index=False))
    if args.dry_run:
        print("Dry-run complete. No outputs written.")
        return 0

    prepare_output_dir(output_dir, args.overwrite, args.dry_run)

    primary_rows: List[Dict[str, Any]] = []
    foldwise_tables: List[pd.DataFrame] = []
    subgroup_tables: List[pd.DataFrame] = []
    if available_readout(candidate_readout):
        primary_rows.append(load_primary_from_readout(candidate_readout, "ch1_full_5x5_candidate", "candidate [1]", [1]))
        foldwise_tables.append(foldwise_primary(candidate_readout, "ch1_full_5x5_candidate"))
        subgroup_tables.append(subgroup_from_predictions(candidate_readout, "ch1_full_5x5_candidate"))
    if available_readout(reference_readout):
        primary_rows.append(load_primary_from_readout(reference_readout, "ch1_0_2_mfrsplit_3840_readout", "current full [1,0,2] classifier-only sweep", [1, 0, 2]))
        foldwise_tables.append(foldwise_primary(reference_readout, "ch1_0_2_mfrsplit_3840_readout"))
        subgroup_tables.append(subgroup_from_predictions(reference_readout, "ch1_0_2_mfrsplit_3840_readout"))
    audit_row = load_primary_from_threshold_audit(threshold_audit)
    if audit_row is not None:
        primary_rows.append(audit_row)

    primary = pd.DataFrame(primary_rows)
    if not primary.empty and "ch1_full_5x5_candidate" in primary["run_id"].values:
        candidate_auc = float(primary.loc[primary["run_id"].eq("ch1_full_5x5_candidate"), "auc"].iloc[0])
        primary["delta_auc_vs_ch1_candidate"] = primary["auc"].astype(float) - candidate_auc
    foldwise = pd.concat(foldwise_tables, ignore_index=True) if foldwise_tables else pd.DataFrame()
    subgroup = pd.concat(subgroup_tables, ignore_index=True) if subgroup_tables else pd.DataFrame()
    weak = weak_fold_summary(foldwise)
    configs = pd.DataFrame(
        [
            {"run_id": "ch1_full_5x5_candidate", **load_run_config_summary(candidate_run)},
            {"run_id": "ch1_0_2_mfrsplit_3840", **load_run_config_summary(reference_run)},
        ]
    )

    availability.to_csv(output_dir / "availability.csv", index=False)
    primary.to_csv(output_dir / "primary_metric_comparison.csv", index=False)
    subgroup.to_csv(output_dir / "manufacturer_subgroup_comparison.csv", index=False)
    foldwise.to_csv(output_dir / "foldwise_primary_metrics.csv", index=False)
    weak.to_csv(output_dir / "weak_fold_summary.csv", index=False)
    configs.to_csv(output_dir / "run_config_summary.csv", index=False)
    write_readme(output_dir, availability, primary, weak)
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "candidate_run_dir": str(candidate_run),
        "candidate_readout_dir": str(candidate_readout),
        "reference_run_dir": str(reference_run),
        "reference_readout_dir": str(reference_readout),
        "threshold_audit_dir": str(threshold_audit),
        "output_dir": str(output_dir),
        "primary_model": PRIMARY_MODEL,
        "primary_threshold_strategy": PRIMARY_THRESHOLD,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
    }
    (output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote comparison outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
