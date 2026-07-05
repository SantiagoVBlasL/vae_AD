#!/usr/bin/env python3
"""Compare objective-v2 FAST 3x3 [1,0,2] pilot arms.

Read-only. It expects Stage B classifier-only outputs from the two pilot arms.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_objective_v2_fast_3x3_ch1_0_2"
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_No rows._\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append("" if pd.isna(val) else f"{float(val):.{digits}f}")
            else:
                vals.append("" if pd.isna(val) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def primary_rows(planned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled_rows: List[Dict[str, Any]] = []
    fold_rows: List[pd.DataFrame] = []
    availability: List[Dict[str, Any]] = []
    for row in planned.itertuples(index=False):
        readout_dir = Path(str(row.readout_output_dir))
        pooled_path = readout_dir / "classifier_sweep_pooled_metrics.csv"
        fold_path = readout_dir / "classifier_sweep_foldwise_metrics.csv"
        available = pooled_path.exists() and fold_path.exists()
        availability.append(
            {
                "run_name": row.run_name,
                "readout_dir": str(readout_dir),
                "readout_available": bool(available),
            }
        )
        if not available:
            continue
        pooled = pd.read_csv(pooled_path)
        primary = pooled[
            pooled["model_name"].eq(PRIMARY_MODEL)
            & pooled["threshold_strategy"].eq(PRIMARY_THRESHOLD)
        ]
        if len(primary) != 1:
            raise RuntimeError(f"{row.run_name}: expected one primary pooled row, found {len(primary)}")
        out = primary.iloc[0].to_dict()
        out.update(
            {
                "run_name": row.run_name,
                "channels_to_use": row.channels_to_use,
                "recon_loss_mode": row.recon_loss_mode,
                "vae_final_activation": row.vae_final_activation,
            }
        )
        pooled_rows.append(out)
        fold = pd.read_csv(fold_path)
        fold = fold[
            fold["model_name"].eq(PRIMARY_MODEL)
            & fold["threshold_strategy"].eq(PRIMARY_THRESHOLD)
        ].copy()
        fold.insert(0, "run_name", row.run_name)
        fold.insert(1, "recon_loss_mode", row.recon_loss_mode)
        fold_rows.append(fold)
    pooled_df = pd.DataFrame(pooled_rows)
    fold_df = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()
    return pd.DataFrame(availability), pooled_df, fold_df


def main() -> int:
    args = parse_args()
    root = resolve(args.output_root)
    planned_path = root / "planned_runs.csv"
    if not planned_path.exists():
        raise FileNotFoundError(f"Missing planned_runs.csv: {planned_path}")
    planned = pd.read_csv(planned_path)
    availability, pooled, foldwise = primary_rows(planned)
    print("=== Objective-v2 FAST 3x3 comparison ===")
    print(availability.to_string(index=False))
    if args.dry_run:
        print("Dry-run complete. No outputs written.")
        return 0
    comparison_dir = root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    availability.to_csv(comparison_dir / "readout_availability.csv", index=False)
    pooled.to_csv(comparison_dir / "objective_v2_primary_comparison.csv", index=False)
    foldwise.to_csv(comparison_dir / "objective_v2_foldwise_comparison.csv", index=False)
    (comparison_dir / "objective_v2_primary_comparison.md").write_text(md_table(pooled), encoding="utf-8")
    (comparison_dir / "objective_v2_foldwise_comparison.md").write_text(md_table(foldwise), encoding="utf-8")
    lines = [
        "# Objective-v2 FAST 3x3 [1,0,2] Comparison",
        "",
        f"Primary readout: `{PRIMARY_MODEL}` + `{PRIMARY_THRESHOLD}`.",
        "",
        "## Availability",
        "",
        md_table(availability),
        "",
        "## Primary Metrics",
        "",
        md_table(pooled),
        "",
        "This comparison is read-only and does not train, modify tensors, metadata, or ledger files.",
    ]
    (comparison_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "primary_model": PRIMARY_MODEL,
        "primary_threshold": PRIMARY_THRESHOLD,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
    }
    (comparison_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote comparison outputs to: {comparison_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
