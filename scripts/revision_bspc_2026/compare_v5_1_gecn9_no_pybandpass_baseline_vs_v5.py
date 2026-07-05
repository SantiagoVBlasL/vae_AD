#!/usr/bin/env python3
"""Compare v5.1 GECN9 [1,0,2] baseline against v5 [1,0,2] and v5 [4,1,0].

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

RUNS: Dict[str, Dict[str, Any]] = {
    "v5_1_gecn9_ch1_0_2_baseline": {
        "label": "v5.1 GECN9 no-Python-bandpass [1,0,2]",
        "big_disk_dir": BIG_DISK / "adni_v5_1_gecn9_no_pybandpass_ch1_0_2_baseline",
        "local_dir": LOCAL_RESULTS / "adni_v5_1_gecn9_no_pybandpass_ch1_0_2_baseline",
        "channels": [1, 0, 2],
        "reference": False,
    },
    "v5_ch1_0_2_baseline": {
        "label": "v5 DPARSF-10000 no-Python-bandpass [1,0,2]",
        "big_disk_dir": BIG_DISK / "adni_v5_dparsf10000_no_pybandpass_ch1_0_2_baseline",
        "local_dir": LOCAL_RESULTS / "adni_v5_dparsf10000_no_pybandpass_ch1_0_2_baseline",
        "channels": [1, 0, 2],
        "reference": True,
    },
    "v5_ch4_1_0_baseline": {
        "label": "v5 DPARSF-10000 no-Python-bandpass [4,1,0]",
        "big_disk_dir": BIG_DISK / "adni_v5_dparsf10000_no_pybandpass_ch4_1_0_baseline",
        "local_dir": LOCAL_RESULTS / "adni_v5_dparsf10000_no_pybandpass_ch4_1_0_baseline",
        "channels": [4, 1, 0],
        "reference": True,
    },
}

DEFAULT_OUTPUT_DIR = LOCAL_RESULTS / "v5_1_gecn9_no_pybandpass_baseline_vs_v5"
METRICS_SUFFIX = (
    "all_folds_metrics_MULTI_logreg_"
    "vaeconvtranspose4l_ld256_beta2.5_normzscore_offdiag_"
    "ch3sel_intFCquarter_drop0.15_ln0_outer5x1_scoreroc_auc.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare v5.1 GECN9 baseline metrics against v5 baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and print planned outputs only.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_run_dir(spec: Dict[str, Any]) -> Path:
    big_disk_dir = Path(spec["big_disk_dir"])
    local_dir = Path(spec["local_dir"])
    if big_disk_dir.exists():
        return big_disk_dir
    return local_dir


def find_metrics_file(run_dir: Path) -> Optional[Path]:
    candidate = run_dir / METRICS_SUFFIX
    if candidate.exists():
        return candidate
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
    }
    if run_config:
        info["channels_indices"] = run_config.get("channels_to_use_indices")
        info["channel_names"] = run_config.get("channel_names_selected")
        info["tensor_shape"] = run_config.get("tensor_shape")
        info["global_tensor_path"] = run_config.get("global_tensor_path")
        info["metadata_path"] = run_config.get("metadata_path")
    elif manifest:
        info["channels_indices"] = manifest.get("selected_channels")
        info["channel_names"] = manifest.get("selected_channel_names")
        info["metadata_path"] = manifest.get("metadata_path")
        info["python_bandpass_applied"] = manifest.get("python_bandpass_applied", False)
    return info, metrics


def metric_summary(info: Dict[str, Any], metrics: Optional[pd.DataFrame]) -> Dict[str, Any]:
    row = dict(info)
    if metrics is None:
        return row
    for clf in sorted(metrics["actual_classifier_type"].dropna().unique()):
        sub = metrics[metrics["actual_classifier_type"].eq(clf)]
        for metric in ["auc_raw", "auc_final", "pr_auc_raw", "balanced_accuracy"]:
            if metric not in sub.columns:
                continue
            row[f"{clf}_{metric}_mean"] = round(float(sub[metric].mean()), 4)
            row[f"{clf}_{metric}_std"] = round(float(sub[metric].std()), 4)
    return row


def foldwise_rows(run_id: str, metrics: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if metrics is None:
        return []
    rows: List[Dict[str, Any]] = []
    for _, r in metrics.iterrows():
        rows.append(
            {
                "run_id": run_id,
                "fold": int(r["fold"]),
                "classifier": r["actual_classifier_type"],
                "auc_raw": round(float(r.get("auc_raw", np.nan)), 4),
                "auc_final": round(float(r.get("auc_final", np.nan)), 4),
                "pr_auc_raw": round(float(r.get("pr_auc_raw", np.nan)), 4),
                "balanced_accuracy": round(float(r.get("balanced_accuracy", np.nan)), 4),
            }
        )
    return rows


def build_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    target = "v5_1_gecn9_ch1_0_2_baseline"
    refs = ["v5_ch1_0_2_baseline", "v5_ch4_1_0_baseline"]
    rows: List[Dict[str, Any]] = []
    target_row = summary[summary["run_id"].eq(target)]
    if target_row.empty or not bool(target_row["available"].iloc[0]):
        return pd.DataFrame()
    t = target_row.iloc[0]
    for ref in refs:
        ref_row = summary[summary["run_id"].eq(ref)]
        if ref_row.empty or not bool(ref_row["available"].iloc[0]):
            continue
        r = ref_row.iloc[0]
        for clf in ["logreg", "svm"]:
            for metric in ["auc_raw", "auc_final"]:
                col = f"{clf}_{metric}_mean"
                if col not in summary.columns or pd.isna(t.get(col)) or pd.isna(r.get(col)):
                    continue
                rows.append(
                    {
                        "target_run": target,
                        "reference_run": ref,
                        "classifier": clf,
                        "metric": metric,
                        "target_mean": float(t[col]),
                        "reference_mean": float(r[col]),
                        "delta_target_minus_reference": round(float(t[col]) - float(r[col]), 4),
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
        "# v5.1 GECN9 No-Python-Bandpass Baseline vs v5",
        "",
        f"Generated: {run_ts}",
        "",
        "No training is performed by this script. It only reads completed run outputs.",
        "",
        "## Availability",
        "",
        "| Run | Available | Channels | Metrics |",
        "|---|---:|---|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row.run_id}` | `{bool(row.available)}` | `{row.channels_indices}` | `{row.metrics_path}` |"
        )

    lines += [
        "",
        "## Mean AUC",
        "",
        "| Run | LogReg raw | LogReg final | SVM raw | SVM final |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in summary[summary["available"]].iterrows():
        lines.append(
            f"| `{row.run_id}` | "
            f"{row.get('logreg_auc_raw_mean', np.nan)} | "
            f"{row.get('logreg_auc_final_mean', np.nan)} | "
            f"{row.get('svm_auc_raw_mean', np.nan)} | "
            f"{row.get('svm_auc_final_mean', np.nan)} |"
        )

    lines += ["", "## Deltas", ""]
    if deltas.empty:
        lines.append("No deltas available yet, most likely because the v5.1 run has not been trained.")
    else:
        lines += [
            "| Target | Reference | Classifier | Metric | Delta |",
            "|---|---|---|---|---:|",
        ]
        for _, row in deltas.iterrows():
            lines.append(
                f"| `{row.target_run}` | `{row.reference_run}` | `{row.classifier}` | "
                f"`{row.metric}` | {row.delta_target_minus_reference:.4f} |"
            )

    lines += [
        "",
        "## Outputs",
        "",
        "- `run_comparison_table.csv`",
        "- `foldwise_auc_table.csv`",
        "- `delta_vs_v5_references.csv`",
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
    foldwise = pd.DataFrame(
        [row for run_id in RUNS for row in foldwise_rows(run_id, metrics_by_run[run_id])]
    )
    deltas = build_deltas(summary)

    print("=== v5.1 GECN9 baseline vs v5 comparison ===")
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
    foldwise.to_csv(args.output_dir / "foldwise_auc_table.csv", index=False)
    deltas.to_csv(args.output_dir / "delta_vs_v5_references.csv", index=False)
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
