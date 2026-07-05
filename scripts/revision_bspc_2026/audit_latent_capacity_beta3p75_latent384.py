#!/usr/bin/env python3
"""Latent capacity audit: latent384_beta3p75 vs latent384_beta2p5 and latent256 references.

Five models compared:
  locked_v5_1b             — adni_v5.1b baseline, latent_dim=256, short schedule
  recover035_latent256     — recover035 with standard 256-dim schedule
  recover035_longpatience256 — recover035 with 10k/p560, latent_dim=256
  recover035_latent384_beta2p5 — beta2.5, latent_dim=384 (base)
  recover035_latent384_beta3p75 — beta3.75, latent_dim=384 [candidate]

Analyses:
  1. Latent cache verification (fold count, mu-column count, subject counts)
  2. Active units  : var(mu_j) > 1e-4 (primary), > 1e-2 (liberal)
  3. Participation ratio  : (Σvar_j)² / Σ(var_j²)
  4. Explained variance spectrum  : dims needed for 50/80/90/95%
  5. Per-dimension MI with Y_target, Manufacturer, Sex from latent_info_per_dim.csv
  6. Classifier pooled AUC/PR-AUC
  7. Key question: does stronger beta compress more useful information?
     Does MI(z;Y_target) increase proportionally more than MI(z;Manufacturer)?

Paths:
  Latent caches  : {run_dir}/classifier_only_readout/latent_cache/fold_{N}_{split}_latent_mu.csv
  Per-dim MI     : {run_dir}/fold_{N}/fold_{N}_{split}_latent_info_per_dim.csv
  Pooled metrics : {run_dir}/classifier_only_readout/classifier_sweep_pooled_metrics.csv

Read-only. Does NOT train, fit thresholds, modify tensors, metadata, ledger, or model outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
BIG_DISK = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026")

DEFAULT_OUTPUT = RESULTS / "latent_capacity_audit_beta3p75_latent384"

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "locked_v5_1b": {
        "label": "locked_v5.1b_latent256",
        "run_dir_name": "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5",
        "latent_dim": 256,
        "beta_vae": 2.5,
        "epochs_vae": 4480,
        "n_cycles": 56,
        "patience": 560,
    },
    "recover035_latent256": {
        "label": "recover035_latent256_baseline",
        "run_dir_name": "recover035_full5x5",
        "latent_dim": 256,
        "beta_vae": 2.5,
        "epochs_vae": None,
        "n_cycles": None,
        "patience": None,
    },
    "recover035_longpatience256": {
        "label": "recover035_longpatience256_T80_h10000_p560",
        "run_dir_name": "recover035_longpatience_T80_h10000_p560_full5x5",
        "latent_dim": 256,
        "beta_vae": 2.5,
        "epochs_vae": 10000,
        "n_cycles": 125,
        "patience": 560,
    },
    "recover035_latent384_beta2p5": {
        "label": "recover035_latent384_beta2.5_T80_h10000_p560",
        "run_dir_name": "recover035_latent384_T80_h10000_p560_full5x5",
        "latent_dim": 384,
        "beta_vae": 2.5,
        "epochs_vae": 10000,
        "n_cycles": 125,
        "patience": 560,
    },
    "recover035_latent384_beta3p75": {
        "label": "recover035_latent384_beta3.75_T80_h10000_p560",
        "run_dir_name": "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        "latent_dim": 384,
        "beta_vae": 3.75,
        "epochs_vae": 10000,
        "n_cycles": 125,
        "patience": 560,
    },
}

MODEL_KEYS = [
    "locked_v5_1b",
    "recover035_latent256",
    "recover035_longpatience256",
    "recover035_latent384_beta2p5",
    "recover035_latent384_beta3p75",
]

N_FOLDS = 5
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
LOCKED_AUC = 0.782951
LOCKED_PR_AUC = 0.559873
VAR_EPS_ACTIVE = 1e-4
VAR_EPS_LIBERAL = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--beta3p75-run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths without reading artifacts.")
    return parser.parse_args()


def resolve_run_dir(key: str, overrides: Dict[str, Optional[Path]]) -> Path:
    override = overrides.get(key)
    if override is not None:
        return override if override.is_absolute() else PROJECT_ROOT / override
    name = MODEL_REGISTRY[key]["run_dir_name"]
    local = RESULTS / name
    big = BIG_DISK / name
    if local.exists():
        return local
    if big.exists():
        return big
    return local


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def write_table(outdir: Path, stem: str, df: pd.DataFrame, max_rows: int = 40) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def load_all_latent_mu(run_dir: Path) -> Optional[pd.DataFrame]:
    cache_dir = run_dir / "classifier_only_readout" / "latent_cache"
    if not cache_dir.exists():
        return None
    dfs: List[pd.DataFrame] = []
    for fold in range(1, N_FOLDS + 1):
        for split in ["trainDev", "test"]:
            p = cache_dir / f"fold_{fold}_{split}_latent_mu.csv"
            if p.exists():
                df = pd.read_csv(p)
                df["fold"] = fold
                df["split"] = split
                dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def latent_capacity_metrics(df_all: pd.DataFrame, latent_dim: int) -> Dict[str, Any]:
    mu_cols = [c for c in df_all.columns if c.startswith("mu_")]
    if not mu_cols:
        return {"error": "no mu columns found"}
    mu_arr = df_all[mu_cols].values
    variances = np.var(mu_arr, axis=0)
    active_strict = int((variances > VAR_EPS_ACTIVE).sum())
    active_liberal = int((variances > VAR_EPS_LIBERAL).sum())
    pr = float((variances.sum() ** 2) / (variances ** 2).sum()) if (variances ** 2).sum() > 0 else 0.0
    pr_pct = float(pr / latent_dim) * 100
    sorted_var = np.sort(variances)[::-1]
    cumvar = np.cumsum(sorted_var) / sorted_var.sum()
    dims_50 = int(np.searchsorted(cumvar, 0.50) + 1)
    dims_80 = int(np.searchsorted(cumvar, 0.80) + 1)
    dims_90 = int(np.searchsorted(cumvar, 0.90) + 1)
    dims_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    return {
        "latent_dim": latent_dim,
        "n_mu_cols": len(mu_cols),
        "active_units_strict": active_strict,
        "active_units_liberal": active_liberal,
        "pct_active_strict": float(active_strict / latent_dim) * 100,
        "participation_ratio": pr,
        "pr_pct": pr_pct,
        "dims_at_50pct": dims_50,
        "dims_at_80pct": dims_80,
        "dims_at_90pct": dims_90,
        "dims_at_95pct": dims_95,
        "pct_horizon_at_80pct": float(dims_80 / latent_dim) * 100,
        "pct_horizon_at_90pct": float(dims_90 / latent_dim) * 100,
    }


def load_per_dim_mi(run_dir: Path) -> Optional[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    for fold in range(1, N_FOLDS + 1):
        for split in ["trainDev", "test"]:
            p = run_dir / f"fold_{fold}" / f"fold_{fold}_{split}_latent_info_per_dim.csv"
            if p.exists():
                df = pd.read_csv(p)
                df["fold"] = fold
                df["split"] = split
                dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def aggregate_mi(mi_df: pd.DataFrame) -> Dict[str, float]:
    result: Dict[str, float] = {}
    target_cols = [c for c in mi_df.columns if "y_target" in c.lower() or "diagnosis" in c.lower() or "researchgroup" in c.lower()]
    mfr_cols = [c for c in mi_df.columns if "manufacturer" in c.lower()]
    sex_cols = [c for c in mi_df.columns if c.lower() in {"sex", "mi_sex"}]
    for label, cols in [("Y_target", target_cols), ("Manufacturer", mfr_cols), ("Sex", sex_cols)]:
        for col in cols:
            vals = mi_df[col].dropna()
            if not vals.empty:
                result[f"total_MI_{label}"] = float(vals.sum())
                result[f"mean_dim_MI_{label}"] = float(vals.mean())
    return result


def load_classifier_pooled(run_dir: Path, model_key: str) -> Dict[str, Any]:
    reg = MODEL_REGISTRY[model_key]
    path = run_dir / "classifier_only_readout" / "classifier_sweep_pooled_metrics.csv"
    base = {"model_key": model_key, "label": reg["label"], "latent_dim": reg["latent_dim"], "beta_vae": reg["beta_vae"]}
    if not path.exists():
        return {**base, "auc": float("nan"), "pr_auc": float("nan"), "note": "not found"}
    df = pd.read_csv(path)
    mask = df["model_name"].astype(str).eq(PRIMARY_MODEL)
    mask &= df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    df = df[mask].copy()
    if df.empty:
        return {**base, "auc": float("nan"), "pr_auc": float("nan"), "note": "no matching row"}
    row = df.iloc[0]
    return {
        **base,
        "auc": float(row.get("auc", float("nan"))),
        "pr_auc": float(row.get("pr_auc", float("nan"))),
        "balanced_accuracy": float(row.get("balanced_accuracy", float("nan"))),
        "sensitivity": float(row.get("sensitivity", float("nan"))),
        "specificity": float(row.get("specificity", float("nan"))),
        "f1": float(row.get("f1", float("nan"))),
        "note": "",
    }


def main() -> int:
    args = parse_args()
    outdir = resolve(args.output_dir)
    overrides: Dict[str, Optional[Path]] = {"recover035_latent384_beta3p75": args.beta3p75_run_dir}
    run_dirs = {key: resolve_run_dir(key, overrides) for key in MODEL_KEYS}

    print("Latent capacity audit: beta3p75 vs beta2p5 latent384")
    for key, rd in run_dirs.items():
        reg = MODEL_REGISTRY[key]
        status = "exists" if rd.exists() else "NOT FOUND"
        print(f"  [{key}] latent_dim={reg['latent_dim']} beta={reg['beta_vae']}  {rd.name}  ({status})")

    if args.dry_run:
        print("\nDry-run complete. No artifact files read.")
        return 0

    outdir.mkdir(parents=True, exist_ok=True)

    capacity_rows: List[Dict[str, Any]] = []
    mi_rows: List[Dict[str, Any]] = []
    pooled_rows: List[Dict[str, Any]] = []

    for key in MODEL_KEYS:
        rd = run_dirs[key]
        reg = MODEL_REGISTRY[key]
        base = {"model_key": key, "label": reg["label"], "latent_dim": reg["latent_dim"], "beta_vae": reg["beta_vae"]}

        pooled_rows.append(load_classifier_pooled(rd, key) if rd.exists() else {**base, "auc": float("nan"), "pr_auc": float("nan"), "note": "run_dir not found"})

        if not rd.exists():
            capacity_rows.append({**base, "error": "run_dir not found"})
            mi_rows.append({**base, "error": "run_dir not found"})
            continue

        mu_all = load_all_latent_mu(rd)
        if mu_all is None:
            capacity_rows.append({**base, "error": "latent cache not found"})
        else:
            metrics = latent_capacity_metrics(mu_all, reg["latent_dim"])
            capacity_rows.append({**base, **metrics})

        mi_all = load_per_dim_mi(rd)
        if mi_all is None:
            mi_rows.append({**base, "error": "per_dim MI not found"})
        else:
            mi_agg = aggregate_mi(mi_all)
            mi_rows.append({**base, **mi_agg})

    capacity_df = pd.DataFrame(capacity_rows)
    mi_df = pd.DataFrame(mi_rows)
    pooled_df = pd.DataFrame(pooled_rows)

    write_table(outdir, "latent_capacity_metrics", capacity_df)
    write_table(outdir, "mi_aggregated", mi_df)
    write_table(outdir, "classifier_pooled_metrics", pooled_df)

    now = datetime.now(timezone.utc).isoformat()
    report_lines = [
        "# Latent Capacity Audit — beta3p75 vs beta2p5 latent384",
        "",
        f"Generated: {now}",
        "",
        "## Model Registry",
        "| model_key | latent_dim | beta_vae | epochs |",
        "|---|---|---|---|",
    ]
    for key in MODEL_KEYS:
        reg = MODEL_REGISTRY[key]
        report_lines.append(f"| {key} | {reg['latent_dim']} | {reg['beta_vae']} | {reg.get('epochs_vae', '?')} |")

    report_lines += ["", "## Capacity Metrics"]
    if not capacity_df.empty:
        show_cols = [c for c in ["model_key", "latent_dim", "beta_vae", "active_units_strict", "pct_active_strict", "participation_ratio", "pr_pct", "dims_at_80pct", "dims_at_90pct"] if c in capacity_df.columns]
        report_lines.append(capacity_df[show_cols].to_markdown(index=False) if show_cols else "_no data_")

    report_lines += ["", "## Classifier Performance"]
    if not pooled_df.empty:
        show_cols = [c for c in ["model_key", "latent_dim", "beta_vae", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"] if c in pooled_df.columns]
        report_lines.append(pooled_df[show_cols].to_markdown(index=False) if show_cols else "_no data_")

    b3 = pooled_df[pooled_df["model_key"] == "recover035_latent384_beta3p75"]
    b2 = pooled_df[pooled_df["model_key"] == "recover035_latent384_beta2p5"]
    if not b3.empty and not b2.empty:
        b3_auc = float(b3["auc"].iloc[0])
        b3_pr = float(b3["pr_auc"].iloc[0])
        b2_auc = float(b2["auc"].iloc[0])
        b2_pr = float(b2["pr_auc"].iloc[0])
        report_lines += [
            "",
            "## Beta3.75 vs Beta2.5 Delta",
            f"- AUC: {b3_auc:.4f} - {b2_auc:.4f} = {b3_auc - b2_auc:+.4f}",
            f"- PR-AUC: {b3_pr:.4f} - {b2_pr:.4f} = {b3_pr - b2_pr:+.4f}",
            f"- Promotes AUC (> {LOCKED_AUC}): {b3_auc > LOCKED_AUC}",
            f"- Promotes PR-AUC (>= {LOCKED_PR_AUC}): {b3_pr >= LOCKED_PR_AUC}",
            f"- Verdict: **{'PROMOTES' if (b3_auc > LOCKED_AUC and b3_pr >= LOCKED_PR_AUC) else 'DOES NOT PROMOTE'}**",
        ]

    report_lines += [
        "",
        "## Read-only guarantee",
        "Did not train, fit thresholds, modify tensors, metadata, ledger, or model outputs.",
    ]
    (outdir / "final_recommendation.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nOutput: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
