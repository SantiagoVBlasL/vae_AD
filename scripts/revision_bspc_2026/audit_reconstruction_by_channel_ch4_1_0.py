#!/usr/bin/env python3
"""Audit reconstruction distributions by channel from existing QC artifacts.

This script reads only per-fold QC CSVs (`fold_*_dist_raw/norm/recon.csv`). It
does not load the global tensor, VAE checkpoints, or reconstructed arrays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/auc_sprint_adni_only/reconstruction_by_channel_audit"
)
RUN_CANDIDATES = {
    "baseline_tanh": PROJECT_ROOT / "results/revision_bspc_2026/adni_expanded_v4_beta25_ch4_1_0",
    "linearout": PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/adni_expanded_v4_beta25_ch4_1_0_linearout",
    "ckptselect": PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/adni_expanded_v4_beta25_ch4_1_0_ckptselect",
    "mfrstrat": PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/adni_expanded_v4_beta25_ch4_1_0_mfrstrat",
}
GENERATED_FILES = [
    "reconstruction_by_channel_table.csv",
    "reconstruction_saturation_summary.csv",
    "README.md",
]


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
        raise FileExistsError(f"{path} already contains generated outputs; pass --overwrite")
    if overwrite:
        for file_path in existing:
            file_path.unlink()
    return path


def safe_read(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def fold_channel_rows(run_name: str, run_dir: Path, fold: int) -> List[Dict[str, Any]]:
    fold_dir = run_dir / f"fold_{fold}"
    raw = safe_read(fold_dir / f"fold_{fold}_dist_raw.csv")
    norm = safe_read(fold_dir / f"fold_{fold}_dist_norm.csv")
    recon = safe_read(fold_dir / f"fold_{fold}_dist_recon.csv")
    if raw is None and norm is None and recon is None:
        return []
    channels = set()
    for df in [raw, norm, recon]:
        if df is not None and "channel" in df.columns:
            channels.update(df["channel"].astype(str).tolist())
    rows: List[Dict[str, Any]] = []
    for channel in sorted(channels):
        row: Dict[str, Any] = {
            "run": run_name,
            "run_dir": str(run_dir),
            "fold": fold,
            "channel": channel,
            "recon_error_summary_available": False,
            "channel_dominance_in_reconstruction_loss_available": False,
        }
        for stage, df in [("raw", raw), ("norm", norm), ("recon", recon)]:
            if df is None or "channel" not in df.columns:
                continue
            sub = df[df["channel"].astype(str).eq(channel)]
            if sub.empty:
                continue
            for col in ["mean", "std", "min", "max", "p05", "p95"]:
                if col in sub.columns:
                    row[f"{stage}_{col}"] = float(pd.to_numeric(sub[col], errors="coerce").iloc[0])
        recon_min = row.get("recon_min", np.nan)
        recon_max = row.get("recon_max", np.nan)
        recon_p05 = row.get("recon_p05", np.nan)
        recon_p95 = row.get("recon_p95", np.nan)
        row["tanh_lower_boundary_hit"] = bool(np.isfinite(recon_min) and recon_min <= -0.999)
        row["tanh_upper_boundary_hit"] = bool(np.isfinite(recon_max) and recon_max >= 0.999)
        row["tanh_p05_near_boundary"] = bool(np.isfinite(recon_p05) and recon_p05 <= -0.95)
        row["tanh_p95_near_boundary"] = bool(np.isfinite(recon_p95) and recon_p95 >= 0.95)
        if np.isfinite(row.get("norm_std", np.nan)) and np.isfinite(row.get("recon_std", np.nan)):
            row["recon_std_over_norm_std"] = row["recon_std"] / row["norm_std"] if row["norm_std"] else np.nan
        else:
            row["recon_std_over_norm_std"] = np.nan
        rows.append(row)
    return rows


def saturation_summary(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    group_cols = ["run", "channel"]
    rows = []
    for (run, channel), group in table.groupby(group_cols):
        rows.append(
            {
                "run": run,
                "channel": channel,
                "n_folds": int(group["fold"].nunique()),
                "folds_lower_boundary_hit": int(group["tanh_lower_boundary_hit"].sum()),
                "folds_upper_boundary_hit": int(group["tanh_upper_boundary_hit"].sum()),
                "folds_p05_near_lower": int(group["tanh_p05_near_boundary"].sum()),
                "folds_p95_near_upper": int(group["tanh_p95_near_boundary"].sum()),
                "mean_recon_std_over_norm_std": float(pd.to_numeric(group["recon_std_over_norm_std"], errors="coerce").mean()),
                "min_recon_min": float(pd.to_numeric(group.get("recon_min"), errors="coerce").min()),
                "max_recon_max": float(pd.to_numeric(group.get("recon_max"), errors="coerce").max()),
            }
        )
    return pd.DataFrame(rows)


def write_readme(outdir: Path, table: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Reconstruction By Channel Audit",
        "",
        "Read-only audit from existing QC distribution CSVs. No tensors or checkpoints were loaded.",
        "",
        f"- Rows: {len(table)}",
        f"- Runs included: {', '.join(sorted(table['run'].dropna().unique())) if not table.empty else 'none'}",
        "",
        "## Availability",
        "",
        "- Raw/norm/recon distribution summaries are available from existing QC CSVs.",
        "- Per-subject or per-channel reconstruction error arrays were not present in these artifacts, so exact reconstruction error by channel is marked unavailable.",
        "- Channel dominance in reconstruction loss is also unavailable without per-channel reconstruction losses.",
        "",
        "## Tanh Saturation Signal",
        "",
    ]
    if not summary.empty:
        hits = summary[
            (summary["folds_lower_boundary_hit"] > 0)
            | (summary["folds_upper_boundary_hit"] > 0)
            | (summary["folds_p05_near_lower"] > 0)
            | (summary["folds_p95_near_upper"] > 0)
        ]
        if hits.empty:
            lines.append("- No recon channel shows strong tanh boundary hits in the QC summaries.")
        else:
            lines.append("- Some recon channels reach or approach tanh boundaries; see `reconstruction_saturation_summary.csv`.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `reconstruction_by_channel_table.csv`",
            "- `reconstruction_saturation_summary.csv`",
        ]
    )
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = prepare_output_dir(args.output_dir, args.overwrite)
    rows: List[Dict[str, Any]] = []
    for run_name, run_dir in RUN_CANDIDATES.items():
        if not run_dir.exists():
            continue
        for fold in range(1, 6):
            rows.extend(fold_channel_rows(run_name, run_dir, fold))
    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("No reconstruction QC distribution CSVs found.")
    summary = saturation_summary(table)
    table.to_csv(outdir / "reconstruction_by_channel_table.csv", index=False)
    summary.to_csv(outdir / "reconstruction_saturation_summary.csv", index=False)
    write_readme(outdir, table, summary)
    print(table[["run", "fold", "channel", "norm_min", "norm_max", "recon_min", "recon_max", "tanh_lower_boundary_hit", "tanh_upper_boundary_hit", "recon_std_over_norm_std"]].to_string(index=False))
    print("\nSaturation summary:")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
