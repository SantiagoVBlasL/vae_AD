#!/usr/bin/env python3
"""Summarize block-order FAST 3x3 Stage B readouts after checkpoint-load fix."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_block_order_fast3x3_ch1_0_2"
)
RUN_ROOT = OUT_ROOT / "runs"
RUNS = {
    "legacy_act_norm": RUN_ROOT / "legacy_act_norm",
    "norm_act": RUN_ROOT / "norm_act",
}
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6f}" if np.isfinite(v) else "NA"
    return str(v)


def md_table(df: pd.DataFrame, cols: Sequence[str]) -> str:
    if df.empty:
        return "No rows.\n"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def load_run_tables(name: str, run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    readout = run_dir / "classifier_only_readout"
    pooled_path = readout / "classifier_sweep_pooled_metrics.csv"
    foldwise_path = readout / "classifier_sweep_foldwise_metrics.csv"
    subgroup_path = readout / "classifier_sweep_subgroup_metrics_by_manufacturer.csv"
    manifest_path = readout / "latent_feature_manifest.json"
    require(pooled_path)
    require(foldwise_path)
    require(subgroup_path)
    require(manifest_path)
    pooled = pd.read_csv(pooled_path)
    foldwise = pd.read_csv(foldwise_path)
    subgroup = pd.read_csv(subgroup_path)
    manifest = read_json(manifest_path)
    for df in [pooled, foldwise, subgroup]:
        df.insert(0, "block_order", name)
    return pooled, foldwise, subgroup, manifest


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pooled_rows: List[pd.DataFrame] = []
    foldwise_rows: List[pd.DataFrame] = []
    subgroup_rows: List[pd.DataFrame] = []
    manifests: Dict[str, Any] = {}
    for name, run_dir in RUNS.items():
        pooled, foldwise, subgroup, manifest = load_run_tables(name, run_dir)
        pooled_rows.append(pooled)
        foldwise_rows.append(foldwise)
        subgroup_rows.append(subgroup)
        manifests[name] = manifest

    pooled_all = pd.concat(pooled_rows, ignore_index=True, sort=False)
    foldwise_all = pd.concat(foldwise_rows, ignore_index=True, sort=False)
    subgroup_all = pd.concat(subgroup_rows, ignore_index=True, sort=False)

    primary = pooled_all[
        pooled_all["model_name"].eq(PRIMARY_MODEL)
        & pooled_all["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ].copy()
    primary = primary.sort_values("auc", ascending=False)
    metrics = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]
    if set(primary["block_order"]) == set(RUNS):
        ref = primary.set_index("block_order").loc["legacy_act_norm"]
        for metric in metrics:
            primary[f"{metric}_delta_vs_legacy"] = primary[metric].astype(float) - float(ref[metric])
    primary_cols = [
        "block_order",
        "model_name",
        "threshold_strategy",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "auc_delta_vs_legacy",
        "pr_auc_delta_vs_legacy",
    ]
    primary.to_csv(OUT_ROOT / "primary_comparison.csv", index=False)
    (OUT_ROOT / "primary_comparison.md").write_text(
        "# Block-Order FAST 3x3 Primary Comparison\n\n"
        + md_table(primary, [c for c in primary_cols if c in primary.columns]),
        encoding="utf-8",
    )

    fold_primary = foldwise_all[
        foldwise_all["model_name"].eq(PRIMARY_MODEL)
        & foldwise_all["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ].copy()
    fold_primary = fold_primary.sort_values(["fold", "block_order"])
    if set(fold_primary["block_order"]) == set(RUNS):
        legacy_by_fold = fold_primary[fold_primary["block_order"].eq("legacy_act_norm")].set_index("fold")
        for metric in metrics:
            fold_primary[f"{metric}_delta_vs_legacy"] = [
                float(row[metric]) - float(legacy_by_fold.loc[int(row["fold"]), metric])
                for _, row in fold_primary.iterrows()
            ]
    fold_primary.to_csv(OUT_ROOT / "foldwise_comparison.csv", index=False)
    fold_cols = [
        "fold",
        "block_order",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "auc_delta_vs_legacy",
        "pr_auc_delta_vs_legacy",
    ]
    (OUT_ROOT / "foldwise_comparison.md").write_text(
        "# Block-Order FAST 3x3 Foldwise Comparison\n\n"
        + md_table(fold_primary, [c for c in fold_cols if c in fold_primary.columns]),
        encoding="utf-8",
    )

    subgroup_primary = subgroup_all[
        subgroup_all["model_name"].eq(PRIMARY_MODEL)
        & subgroup_all["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ].copy()
    subgroup_primary.to_csv(OUT_ROOT / "manufacturer_subgroup_comparison.csv", index=False)

    best = primary.iloc[0] if not primary.empty else None
    legacy = primary[primary["block_order"].eq("legacy_act_norm")].iloc[0] if "legacy_act_norm" in set(primary["block_order"]) else None
    norm = primary[primary["block_order"].eq("norm_act")].iloc[0] if "norm_act" in set(primary["block_order"]) else None
    if best is None:
        decision = "No primary readout rows were available; no recommendation can be made."
    elif best["block_order"] == "norm_act" and legacy is not None:
        auc_delta = float(norm["auc"] - legacy["auc"])
        pr_delta = float(norm["pr_auc"] - legacy["pr_auc"])
        decision = (
            f"`norm_act` ranks above legacy in FAST 3x3 by AUC "
            f"(delta AUC={auc_delta:.4f}, delta PR-AUC={pr_delta:.4f}). "
            "Treat this as screening-only evidence; a FULL 5x5 run would still be required before promotion."
        )
    else:
        decision = (
            "`norm_act` does not beat legacy_act_norm on the pre-specified FAST 3x3 primary AUC readout. "
            "Do not promote it."
        )

    report_lines = [
        "# Block-Order FAST 3x3 Stage B Fix Report",
        "",
        "## What Was Fixed",
        "",
        "`run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py` now reconstructs "
        "`ConvolutionalVAE` from the candidate `run_config.json`, including `vae_block_order`, "
        "`vae_dropout_scope`, `encoder_norm_mode`, `vae_final_activation`, `intermediate_fc_dim_vae`, "
        "`decoder_type`, `num_conv_layers_encoder`, `latent_dim`, and selected input channels.",
        "",
        "Default behavior remains `vae_block_order=legacy_act_norm` when the key is absent.",
        "",
        "## Safety",
        "",
        "- Stage A retraining: `False`.",
        "- Stage B rerun only: `norm_act`; legacy Stage B was already complete.",
        "- Tensor modification: `False`.",
        "- Metadata/ledger modification: `False`.",
        "",
        "## Checkpoint Load Smoke",
        "",
        "See `block_order_stageb_checkpoint_load_smoke.csv`.",
        "",
        "## Primary Result",
        "",
        md_table(primary, [c for c in primary_cols if c in primary.columns]),
    ]
    (OUT_ROOT / "block_order_fast3x3_stageb_fix_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    (OUT_ROOT / "recommendation.md").write_text(
        "# Block-Order FAST 3x3 Recommendation\n\n"
        + decision
        + "\n\n"
        "Primary ranking used only Stage B `logreg_l2` with "
        "`inner_oof_target_sens_ge_0p70_max_spec`. Stage A dummy canonical logreg metrics were ignored.\n",
        encoding="utf-8",
    )
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "output_root": str(OUT_ROOT),
        "runs": {k: str(v) for k, v in RUNS.items()},
        "primary_model": PRIMARY_MODEL,
        "primary_threshold": PRIMARY_THRESHOLD,
        "manifests": manifests,
        "stage_a_retrained": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
    }
    (OUT_ROOT / "command_log_stageb_fix.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(primary[[c for c in ["block_order", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"] if c in primary.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
