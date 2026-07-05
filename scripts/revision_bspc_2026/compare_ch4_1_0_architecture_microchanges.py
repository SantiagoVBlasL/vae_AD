#!/usr/bin/env python3
"""Compare ADNI V4 [4,1,0] architecture microchange and sensitivity runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = PROJECT_ROOT / "scripts/revision_bspc_2026/compare_ch4_1_0_tanh_vs_linearout.py"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/ch4_1_0_architecture_microchanges"
)

RUN_SPECS = [
    {
        "run_key": "baseline_tanh_quarter",
        "label": "Baseline tanh quarter-FC",
        "role": "primary_reference",
        "decision": "main_candidate",
        "run_dir": PROJECT_ROOT / "results/revision_bspc_2026/adni_expanded_v4_beta25_ch4_1_0",
    },
    {
        "run_key": "linearout_quarter",
        "label": "Linear-output quarter-FC",
        "role": "decoder_activation_microchange",
        "decision": "negative_control",
        "run_dir": PROJECT_ROOT
        / "results/revision_bspc_2026/auc_sprint_adni_only/adni_expanded_v4_beta25_ch4_1_0_linearout",
    },
    {
        "run_key": "manufacturer_strat_quarter",
        "label": "Manufacturer-stratified quarter-FC",
        "role": "scanner_balanced_sensitivity",
        "decision": "sensitivity_analysis",
        "run_dir": PROJECT_ROOT
        / "results/revision_bspc_2026/auc_sprint_adni_only/adni_expanded_v4_beta25_ch4_1_0_mfrstrat",
    },
    {
        "run_key": "ckptselect_quarter",
        "label": "Checkpoint-cadence quarter-FC",
        "role": "checkpoint_selection_control",
        "decision": "negative_control",
        "run_dir": PROJECT_ROOT
        / "results/revision_bspc_2026/auc_sprint_adni_only/adni_expanded_v4_beta25_ch4_1_0_ckptselect",
    },
    {
        "run_key": "half_fc_tanh",
        "label": "Half-FC tanh",
        "role": "intermediate_fc_microchange",
        "decision": "candidate_if_available",
        "run_dir": PROJECT_ROOT
        / "results/revision_bspc_2026/auc_sprint_adni_only/adni_expanded_v4_beta25_ch4_1_0_half_fc",
    },
]

GENERATED_FILES = [
    "microchange_model_comparison_metrics.csv",
    "microchange_model_comparison_fold_metrics.csv",
    "microchange_latent_scanner_qc_summary.csv",
    "microchange_reconstruction_summary.csv",
    "microchange_reconstruction_channel_summaries.csv",
    "comparison_manifest.json",
    "README.md",
]


def load_helper_module() -> Any:
    spec = importlib.util.spec_from_file_location("model_compare_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import comparison helpers from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = load_helper_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = resolve(path)
    path.mkdir(parents=True, exist_ok=True)
    existing = [path / name for name in GENERATED_FILES if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{path} already contains comparison outputs; pass --overwrite")
    if overwrite:
        for file_path in existing:
            file_path.unlink()
    return path


def markdown_table(df: pd.DataFrame, columns: List[str], n: int = 30) -> str:
    if df.empty:
        return "_No rows available._"
    display = df.copy()
    for col in columns:
        if col not in display.columns:
            display[col] = np.nan
    display = display[columns].head(n).copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            display[col] = display[col].fillna("").astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in display.to_numpy()]
    return "\n".join([header, sep, *rows])


def write_readme(
    outdir: Path,
    metrics: pd.DataFrame,
    qc_summary: pd.DataFrame,
    recon_summary: pd.DataFrame,
    unavailable: List[Dict[str, Any]],
) -> None:
    ok = metrics[metrics["status"].eq("complete")].copy() if "status" in metrics.columns else pd.DataFrame()
    if not ok.empty:
        ok = ok.sort_values(["decision", "auc_mean_fold"], ascending=[True, False])
    metric_cols = [
        "run_key",
        "decision",
        "role",
        "classifier",
        "auc_mean_fold",
        "auc_sd_fold",
        "roc_auc_pooled",
        "pr_auc_mean_fold",
        "balanced_accuracy_mean_fold",
        "sensitivity_AD_mean_fold",
        "specificity_CN_mean_fold",
        "brier_mean_fold",
        "ece_10_bins_mean_fold",
        "artifact_time_span_sec",
        "MI_Manufacturer_over_Y_test_mean",
        "acc_site_latent_test_mean",
        "reconstruction_gap_val_minus_train_mean",
    ]
    lines = [
        "# ADNI V4 [4,1,0] Architecture Microchange Comparison",
        "",
        "Compares available ADNI-only runs around the current tanh quarter-FC baseline.",
        "",
        "## Interpretation Rules",
        "",
        "- `baseline_tanh_quarter` is the primary reference.",
        "- `linearout_quarter` is a negative-control decoder-output microchange.",
        "- `manufacturer_strat_quarter` is scanner-balanced sensitivity analysis, not the main model.",
        "- `ckptselect_quarter` is a checkpoint-selection/control context with the same architecture.",
        "- `half_fc_tanh` is the next controlled architecture microchange when its run becomes available.",
        "- Manufacturer/Site are reported only as diagnostics; they are not predictive features.",
        "- Runtime is approximated from artifact timestamp span when explicit runtime is unavailable.",
        "",
        "## Metrics",
        "",
        markdown_table(ok, metric_cols),
        "",
        "## Latent/Scanner QC",
        "",
        markdown_table(
            qc_summary,
            [
                "run_key",
                "n_qc_folds",
                "MI_Manufacturer_over_Y_test_mean",
                "acc_site_latent_test_mean",
                "acc_site_raw_test_mean",
                "active_units_test_mean",
                "TC_test_mean",
            ],
        ),
        "",
        "## Reconstruction Summary",
        "",
        markdown_table(
            recon_summary,
            [
                "run_key",
                "n_recon_folds",
                "best_epoch_mean",
                "early_stop_epoch_mean",
                "recon_train_at_best_mean",
                "recon_val_at_best_mean",
                "reconstruction_gap_val_minus_train_mean",
                "kl_val_at_best_mean",
            ],
        ),
    ]
    if unavailable:
        lines.extend(["", "## Unavailable Or Incomplete Runs", ""])
        for item in unavailable:
            lines.append(f"- `{item['run_key']}`: {item['note']} Path: `{item['run_dir']}`")
    lines.extend(
        [
            "",
            "## Decision Use",
            "",
            "After the half-FC run completes, compare `half_fc_tanh` only against `baseline_tanh_quarter` as the controlled intermediate-FC experiment. Keep the manufacturer-stratified row as sensitivity context.",
        ]
    )
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = prepare_output_dir(args.output_dir, args.overwrite)

    metric_frames: List[pd.DataFrame] = []
    fold_frames: List[pd.DataFrame] = []
    latent_frames: List[pd.DataFrame] = []
    scanner_frames: List[pd.DataFrame] = []
    recon_frames: List[pd.DataFrame] = []
    channel_frames: List[pd.DataFrame] = []
    unavailable: List[Dict[str, Any]] = []

    for spec in RUN_SPECS:
        run_key = spec["run_key"]
        run_dir = resolve(spec["run_dir"])
        metrics, folds = HELPERS.summarize_model_metrics(
            run_key,
            spec["label"],
            spec["role"],
            spec["decision"],
            run_dir,
        )
        metrics["artifact_time_span_sec"] = HELPERS.artifact_time_span_sec(run_dir)
        metric_frames.append(metrics)
        if not folds.empty:
            folds["artifact_time_span_sec"] = HELPERS.artifact_time_span_sec(run_dir)
            fold_frames.append(folds)
        if (metrics["status"] != "complete").all():
            unavailable.append({"run_key": run_key, "run_dir": str(run_dir), "note": str(metrics.iloc[0].get("note"))})
            continue
        latent_frames.append(HELPERS.parse_latent_info(run_key, run_dir))
        scanner_frames.append(HELPERS.parse_scanner_leakage(run_key, run_dir))
        recon, channels = HELPERS.parse_reconstruction(run_key, run_dir)
        recon_frames.append(recon)
        channel_frames.append(channels)

    metrics_df = pd.concat(metric_frames, ignore_index=True, sort=False)
    fold_df = pd.concat(fold_frames, ignore_index=True, sort=False) if fold_frames else pd.DataFrame()
    latent_df = pd.concat(latent_frames, ignore_index=True, sort=False) if latent_frames else pd.DataFrame()
    scanner_df = pd.concat(scanner_frames, ignore_index=True, sort=False) if scanner_frames else pd.DataFrame()
    recon_df = pd.concat(recon_frames, ignore_index=True, sort=False) if recon_frames else pd.DataFrame()
    channel_df = pd.concat(channel_frames, ignore_index=True, sort=False) if channel_frames else pd.DataFrame()

    qc_summary = HELPERS.summarize_qc(latent_df, scanner_df)
    recon_summary = HELPERS.summarize_reconstruction(recon_df)
    metrics_df = metrics_df.merge(qc_summary, on="run_key", how="left").merge(recon_summary, on="run_key", how="left")

    metrics_df.to_csv(outdir / "microchange_model_comparison_metrics.csv", index=False)
    fold_df.to_csv(outdir / "microchange_model_comparison_fold_metrics.csv", index=False)
    qc_summary.to_csv(outdir / "microchange_latent_scanner_qc_summary.csv", index=False)
    recon_summary.to_csv(outdir / "microchange_reconstruction_summary.csv", index=False)
    channel_df.to_csv(outdir / "microchange_reconstruction_channel_summaries.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(outdir),
        "run_specs": [
            {**{k: v for k, v in spec.items() if k != "run_dir"}, "run_dir": str(resolve(spec["run_dir"]))}
            for spec in RUN_SPECS
        ],
        "unavailable": unavailable,
        "manufacturer_or_site_as_predictive_feature": False,
    }
    (outdir / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(outdir, metrics_df, qc_summary, recon_summary, unavailable)

    print(f"Comparison written to: {outdir}")
    cols = [
        "run_key",
        "decision",
        "role",
        "classifier",
        "status",
        "auc_mean_fold",
        "auc_sd_fold",
        "roc_auc_pooled",
        "MI_Manufacturer_over_Y_test_mean",
        "acc_site_latent_test_mean",
    ]
    print(metrics_df[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
