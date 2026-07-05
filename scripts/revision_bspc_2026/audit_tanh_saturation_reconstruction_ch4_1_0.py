#!/usr/bin/env python3
"""Read-only tanh saturation/reconstruction audit using existing QC CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PROJECT_ROOT / "results/revision_bspc_2026/adni_expanded_v4_beta25_ch4_1_0"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/tanh_saturation_ch4_1_0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = resolve(path)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_dist_files(run_dir: Path, suffix: str) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for path in sorted(run_dir.glob(f"fold_*/fold_*_dist_{suffix}.csv")):
        fold = int(path.parent.name.split("_")[-1])
        df = pd.read_csv(path)
        df.insert(0, "fold", fold)
        df.insert(1, "source_file", str(path))
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def aggregate_dist(df: pd.DataFrame, value_prefix: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby("channel").agg(
        folds=("fold", "nunique"),
        mean_min=("min", "mean"),
        worst_min=("min", "min"),
        mean_max=("max", "mean"),
        worst_max=("max", "max"),
        mean_p05=("p05", "mean"),
        mean_p95=("p95", "mean"),
        mean_std=("std", "mean"),
    ).reset_index()
    return agg.rename(columns={c: f"{value_prefix}_{c}" for c in agg.columns if c != "channel"})


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    outdir = prepare_output_dir(args.output_dir, args.overwrite)
    norm = read_dist_files(run_dir, "norm")
    recon = read_dist_files(run_dir, "recon")
    raw = read_dist_files(run_dir, "raw")
    norm_agg = aggregate_dist(norm, "norm")
    recon_agg = aggregate_dist(recon, "recon")
    raw_agg = aggregate_dist(raw, "raw")
    audit = norm_agg.merge(recon_agg, on="channel", how="outer").merge(raw_agg, on="channel", how="outer")
    if not audit.empty:
        audit["norm_fraction_abs_gt_0p95_available"] = False
        audit["recon_saturation_fraction_available"] = False
        audit["recon_hits_tanh_boundary_any_fold"] = audit["channel"].map(
            lambda ch: bool(((recon[recon["channel"].eq(ch)]["min"] <= -0.999).any()) or ((recon[recon["channel"].eq(ch)]["max"] >= 0.999).any()))
        )
        audit["norm_exceeds_tanh_range_any_fold"] = audit["channel"].map(
            lambda ch: bool(((norm[norm["channel"].eq(ch)]["min"] < -1.0).any()) or ((norm[norm["channel"].eq(ch)]["max"] > 1.0).any()))
        )
        audit["p95_recon_minus_norm"] = audit.get("recon_mean_p95", np.nan) - audit.get("norm_mean_p95", np.nan)
        audit["p05_recon_minus_norm"] = audit.get("recon_mean_p05", np.nan) - audit.get("norm_mean_p05", np.nan)
    norm.to_csv(outdir / "fold_distribution_norm.csv", index=False)
    recon.to_csv(outdir / "fold_distribution_recon.csv", index=False)
    raw.to_csv(outdir / "fold_distribution_raw.csv", index=False)
    audit.to_csv(outdir / "tanh_saturation_reconstruction_summary.csv", index=False)
    test_linear = bool((not audit.empty) and (audit["recon_hits_tanh_boundary_any_fold"].any() or audit["norm_exceeds_tanh_range_any_fold"].any()))
    lines = [
        "# Tanh Saturation Reconstruction Audit",
        "",
        "Read-only audit using existing `fold_*_dist_norm.csv`, `fold_*_dist_recon.csv`, and `fold_*_dist_raw.csv` files.",
        "",
        f"- Norm distribution files found: {norm['fold'].nunique() if not norm.empty else 0}",
        f"- Recon distribution files found: {recon['fold'].nunique() if not recon.empty else 0}",
        "- Fraction `|x_norm| > 0.95`: unavailable from current summary CSVs; per-value histograms or tensors would be required.",
        "- Reconstruction saturation fraction near +/-1: unavailable from current summary CSVs; min/max boundary hits are reported instead.",
        f"- Any recon min/max reaches tanh boundary: {bool(not audit.empty and audit['recon_hits_tanh_boundary_any_fold'].any())}",
        f"- Any normalized input range exceeds [-1, 1]: {bool(not audit.empty and audit['norm_exceeds_tanh_range_any_fold'].any())}",
        f"- Recommendation: {'test final_activation=linear in a controlled run' if test_linear else 'tanh boundary evidence is weak from available summaries; linear can remain secondary'}",
    ]
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
