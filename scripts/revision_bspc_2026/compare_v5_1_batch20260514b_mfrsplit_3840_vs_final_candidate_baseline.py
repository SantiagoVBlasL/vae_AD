#!/usr/bin/env python3
"""Compare the manufacturer-aware 3840-epoch candidate against the current final-candidate baseline.

Read-only by default in --dry-run mode. It does not train models.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BIG_DISK = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026")
LOCAL_RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"

TARGET_RUN = "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
BASELINE_RUN = "adni_v5_1_batch20260514b_ch1_0_2_final_candidate_baseline"

RUNS: Dict[str, Dict[str, Any]] = {
    TARGET_RUN: {
        "label": "v5.1 batch20260514b no-Python-bandpass manufacturer-aware split 3840 epochs [1,0,2]",
        "big_disk_dir": BIG_DISK / TARGET_RUN,
        "local_dir": LOCAL_RESULTS / TARGET_RUN,
        "channels": [1, 0, 2],
        "reference": False,
    },
    BASELINE_RUN: {
        "label": "v5.1 batch20260514b no-Python-bandpass current final-candidate baseline [1,0,2]",
        "big_disk_dir": BIG_DISK / BASELINE_RUN,
        "local_dir": LOCAL_RESULTS / BASELINE_RUN,
        "channels": [1, 0, 2],
        "reference": True,
    },
}

DEFAULT_OUTPUT_DIR = LOCAL_RESULTS / "adni_v5_1_batch20260514b_mfrsplit_3840_vs_final_candidate_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare v5.1 batch20260514b manufacturer-aware 3840 run against current baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and print planned outputs only.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_run_dir(spec: Dict[str, Any]) -> Path:
    if Path(spec["big_disk_dir"]).exists():
        return Path(spec["big_disk_dir"])
    return Path(spec["local_dir"])


def find_metrics_file(run_dir: Path) -> Optional[Path]:
    matches = sorted(run_dir.glob("all_folds_metrics_MULTI*.csv"))
    return matches[0] if matches else None


def load_json_optional(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_run(run_id: str, spec: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    run_dir = resolve_run_dir(spec)
    metrics_path = find_metrics_file(run_dir)
    metrics = pd.read_csv(metrics_path) if metrics_path is not None else None
    run_config = load_json_optional(run_dir / "run_config.json")
    manifest = load_json_optional(run_dir / "run_manifest.json")
    info = {
        "run_id": run_id,
        "label": spec["label"],
        "run_dir": str(run_dir),
        "available": metrics is not None,
        "metrics_path": str(metrics_path) if metrics_path else "",
        "run_config_path": str(run_dir / "run_config.json") if (run_dir / "run_config.json").exists() else "",
        "manifest_path": str(run_dir / "run_manifest.json") if (run_dir / "run_manifest.json").exists() else "",
        "expected_channels": spec["channels"],
        "channels_indices": None,
        "channel_names": None,
        "tensor_shape": None,
        "global_tensor_path": None,
        "metadata_path": None,
        "python_bandpass_applied": False,
        "epochs_vae": None,
        "cyclical_beta_n_cycles": None,
        "classifier_stratify_cols": None,
        "vae_stratify_cols": None,
    }
    config_like = run_config or manifest or {}
    args = config_like.get("args", {})
    info["channels_indices"] = config_like.get("channels_to_use_indices") or config_like.get("selected_channels") or args.get("channels_to_use")
    info["channel_names"] = config_like.get("channel_names_selected") or config_like.get("selected_channel_names")
    info["tensor_shape"] = config_like.get("tensor_shape")
    info["global_tensor_path"] = config_like.get("global_tensor_path")
    info["metadata_path"] = config_like.get("metadata_path")
    info["python_bandpass_applied"] = config_like.get("python_bandpass_applied", False)
    info["epochs_vae"] = args.get("epochs_vae") or config_like.get("epochs_vae")
    info["cyclical_beta_n_cycles"] = args.get("cyclical_beta_n_cycles") or config_like.get("cyclical_beta_n_cycles")
    info["classifier_stratify_cols"] = args.get("classifier_stratify_cols") or (config_like.get("split_strategy") or {}).get("classifier_outer")
    info["vae_stratify_cols"] = args.get("vae_stratify_cols") or (config_like.get("split_strategy") or {}).get("vae_internal_val")
    return info, metrics


def normalize_classifier(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    if "actual_classifier_type" in out.columns and "classifier" not in out.columns:
        out["classifier"] = out["actual_classifier_type"]
    elif "classifier_type" in out.columns and "classifier" not in out.columns:
        out["classifier"] = out["classifier_type"]
    return out


def metric_summary(info: Dict[str, Any], metrics: Optional[pd.DataFrame]) -> Dict[str, Any]:
    row = dict(info)
    if metrics is None:
        return row
    metrics = normalize_classifier(metrics)
    for clf in sorted(metrics["classifier"].dropna().unique()):
        sub = metrics[metrics["classifier"].eq(clf)]
        for metric in ["auc_raw", "auc_final", "pr_auc_raw", "pr_auc_final", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1_score"]:
            if metric not in sub.columns:
                continue
            row[f"{clf}_{metric}_mean"] = round(float(sub[metric].mean()), 4)
            row[f"{clf}_{metric}_std"] = round(float(sub[metric].std(ddof=1)), 4)
    return row


def foldwise_rows(run_id: str, metrics: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if metrics is None:
        return []
    metrics = normalize_classifier(metrics)
    rows: List[Dict[str, Any]] = []
    for _, r in metrics.iterrows():
        rows.append(
            {
                "run_id": run_id,
                "fold": int(r["fold"]),
                "classifier": r["classifier"],
                "auc_raw": round(float(r.get("auc_raw", np.nan)), 4),
                "auc_final": round(float(r.get("auc_final", np.nan)), 4),
                "pr_auc_final": round(float(r.get("pr_auc_final", r.get("pr_auc", np.nan))), 4),
                "balanced_accuracy": round(float(r.get("balanced_accuracy", np.nan)), 4),
                "sensitivity": round(float(r.get("sensitivity", np.nan)), 4),
                "specificity": round(float(r.get("specificity", np.nan)), 4),
            }
        )
    return rows


def build_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    target_row = summary[summary["run_id"].eq(TARGET_RUN)]
    baseline_row = summary[summary["run_id"].eq(BASELINE_RUN)]
    if target_row.empty or baseline_row.empty:
        return pd.DataFrame()
    if not bool(target_row["available"].iloc[0]) or not bool(baseline_row["available"].iloc[0]):
        return pd.DataFrame()
    t = target_row.iloc[0]
    b = baseline_row.iloc[0]
    rows: List[Dict[str, Any]] = []
    for clf in ["logreg", "svm"]:
        for metric in ["auc_final", "pr_auc_final", "balanced_accuracy", "sensitivity", "specificity", "f1_score"]:
            col = f"{clf}_{metric}_mean"
            if col not in summary.columns or pd.isna(t.get(col)) or pd.isna(b.get(col)):
                continue
            rows.append(
                {
                    "target_run": TARGET_RUN,
                    "reference_run": BASELINE_RUN,
                    "classifier": clf,
                    "metric": metric,
                    "target_mean": float(t[col]),
                    "reference_mean": float(b[col]),
                    "delta_target_minus_reference": round(float(t[col]) - float(b[col]), 4),
                }
            )
    return pd.DataFrame(rows)


def prepare_output_dir(path: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory exists and is not empty; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_readme(output_dir: Path, summary: pd.DataFrame, deltas: pd.DataFrame, run_ts: str) -> None:
    lines = [
        "# v5.1 batch20260514b Manufacturer-Aware 3840 vs Current Baseline",
        "",
        f"Generated: {run_ts}",
        "",
        "No training is performed by this script. It only reads completed run outputs.",
        "",
        "## Availability",
        "",
        "| Run | Available | Channels | Epochs | Stratification | Metrics |",
        "|---|---:|---|---:|---|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row.run_id}` | `{bool(row.available)}` | `{row.channels_indices}` | "
            f"`{row.epochs_vae}` | `{row.classifier_stratify_cols}` | `{row.metrics_path}` |"
        )
    lines += ["", "## Deltas", ""]
    if deltas.empty:
        lines.append("No deltas available yet, most likely because the manufacturer-aware 3840 run has not been trained.")
    else:
        lines += ["| Classifier | Metric | Target mean | Baseline mean | Delta |", "|---|---|---:|---:|---:|"]
        for _, row in deltas.iterrows():
            lines.append(
                f"| `{row.classifier}` | `{row.metric}` | {row.target_mean:.4f} | "
                f"{row.reference_mean:.4f} | {row.delta_target_minus_reference:.4f} |"
            )
    lines += [
        "",
        "## Outputs",
        "",
        "- `run_comparison_table.csv`",
        "- `foldwise_metrics_table.csv`",
        "- `delta_vs_final_candidate_baseline.csv`",
        "- `command_log.json`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    infos: Dict[str, Dict[str, Any]] = {}
    metrics_by_run: Dict[str, Optional[pd.DataFrame]] = {}
    for run_id, spec in RUNS.items():
        info, metrics = load_run(run_id, spec)
        infos[run_id] = info
        metrics_by_run[run_id] = metrics
    summary = pd.DataFrame([metric_summary(infos[run_id], metrics_by_run[run_id]) for run_id in RUNS])
    foldwise = pd.DataFrame([row for run_id in RUNS for row in foldwise_rows(run_id, metrics_by_run[run_id])])
    deltas = build_deltas(summary)

    print("=== v5.1 batch20260514b manufacturer-aware 3840 vs current final-candidate baseline ===")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'WRITE OUTPUTS'}")
    print(f"Output: {args.output_dir}")
    for _, row in summary.iterrows():
        status = "OK" if row["available"] else "NOT AVAILABLE"
        print(f"  [{status}] {row['run_id']}: {row['run_dir']}")

    if args.dry_run:
        print("Dry-run complete. No files written. No training run.")
        return

    prepare_output_dir(args.output_dir, overwrite=args.overwrite, dry_run=False)
    summary.to_csv(args.output_dir / "run_comparison_table.csv", index=False)
    foldwise.to_csv(args.output_dir / "foldwise_metrics_table.csv", index=False)
    deltas.to_csv(args.output_dir / "delta_vs_final_candidate_baseline.csv", index=False)
    command = {
        "script": str(Path(__file__).resolve()),
        "created": run_ts,
        "output_dir": str(args.output_dir),
        "overwrite": bool(args.overwrite),
        "training_run": False,
        "runs": {run_id: infos[run_id]["run_dir"] for run_id in RUNS},
    }
    (args.output_dir / "command_log.json").write_text(
        json.dumps(command, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(args.output_dir, summary, deltas, run_ts)
    print(f"Wrote comparison outputs to {args.output_dir}")
    print("No training run.")


if __name__ == "__main__":
    main()
