#!/usr/bin/env python3
"""Read-only Philips CN false-positive audit for recover035_latent384_beta3p75.

Context:
  In the BSPC revision dataset, all CN subjects are Philips-only.
  Any false positive in the CN pool is therefore always a Philips CN FP.
  This audit checks:
    1. Per-fold Philips CN specificity (1 - FPR) vs locked v5.1b reference
    2. Pooled Philips CN specificity
    3. Score distribution for Philips CN vs GE/Siemens CN (if available)
    4. Whether the Fold 1 score anomaly disproportionately affects Philips CN
    5. Manufacturer MI from latent_info files (scanner leakage check)

Key benchmark:
  Philips CN FP not worse than locked reference by more than an acceptable margin.
  Manufacturer leakage (MI(z;Manufacturer)) must not be worse than beta2.5 latent384.

Hard constraints:
  - No training.
  - No threshold fitting.
  - Do not modify tensor, metadata, ledger, configs, or model outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
BIG_DISK = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026")

DEFAULT_RUN_DIR = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
LOCKED_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
BETA2P5_RUN = RESULTS / "recover035_latent384_T80_h10000_p560_full5x5"
DEFAULT_OUTPUT = RESULTS / "philips_cn_fp_audit_beta3p75_latent384"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FIXED_THRESHOLD = "fixed_0p5"
FOLDS = [1, 2, 3, 4, 5]

LOCKED_AUC = 0.782951
LOCKED_PR_AUC = 0.559873

BETA2P5_POOLED_AUC = 0.7694
BETA2P5_POOLED_PR_AUC = 0.4994


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--locked-run-dir", type=Path, default=LOCKED_RUN)
    parser.add_argument("--beta2p5-run-dir", type=Path, default=BETA2P5_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths without reading artifacts.")
    return parser.parse_args()


def resolve_run(path: Path) -> Path:
    if path.is_absolute():
        return path
    local = RESULTS / path
    if local.exists():
        return local
    big = BIG_DISK / path.name
    if big.exists():
        return big
    return PROJECT_ROOT / path


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


def load_predictions(run_dir: Path, model: str, threshold: str) -> Optional[pd.DataFrame]:
    path = run_dir / "classifier_only_readout" / "classifier_sweep_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    mask = df["model_name"].astype(str).str.startswith(model)
    if "threshold_strategy" in df.columns:
        mask &= df["threshold_strategy"].astype(str).eq(threshold)
    return df[mask].copy()


def per_fold_philips_specificity(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        fsub = df[(df["fold"] == fold) & (df["y_true"] == 0)]
        if fsub.empty:
            continue
        row: Dict[str, Any] = {"fold": fold, "n_CN_total": len(fsub)}
        for mfr in ["Philips", "GE", "SIEMENS"]:
            mfr_sub = fsub[fsub["Manufacturer"].astype(str).str.upper() == mfr.upper()] if "Manufacturer" in fsub.columns else pd.DataFrame()
            n = len(mfr_sub)
            row[f"n_CN_{mfr}"] = n
            if n > 0:
                y_pred = mfr_sub["y_pred"].values if "y_pred" in mfr_sub.columns else (mfr_sub["y_score"].values > 0.5).astype(int)
                fp = int((y_pred == 1).sum())
                row[f"FP_CN_{mfr}"] = fp
                row[f"specificity_CN_{mfr}"] = float(1 - fp / n)
                row[f"scores_CN_{mfr}_median"] = float(np.median(mfr_sub["y_score"].values))
                row[f"scores_CN_{mfr}_max"] = float(np.max(mfr_sub["y_score"].values))
            else:
                row[f"FP_CN_{mfr}"] = None
                row[f"specificity_CN_{mfr}"] = None
                row[f"scores_CN_{mfr}_median"] = None
                row[f"scores_CN_{mfr}_max"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_philips_specificity(df: pd.DataFrame) -> Dict[str, Any]:
    cn = df[df["y_true"] == 0]
    result: Dict[str, Any] = {"n_CN_total": len(cn)}
    for mfr in ["Philips", "GE", "SIEMENS"]:
        mfr_sub = cn[cn["Manufacturer"].astype(str).str.upper() == mfr.upper()] if "Manufacturer" in cn.columns else pd.DataFrame()
        n = len(mfr_sub)
        result[f"n_CN_{mfr}"] = n
        if n > 0:
            y_pred = mfr_sub["y_pred"].values if "y_pred" in mfr_sub.columns else (mfr_sub["y_score"].values > 0.5).astype(int)
            fp = int((y_pred == 1).sum())
            result[f"FP_CN_{mfr}"] = fp
            result[f"FPR_CN_{mfr}"] = float(fp / n)
            result[f"specificity_CN_{mfr}"] = float(1 - fp / n)
        else:
            result[f"FP_CN_{mfr}"] = 0
            result[f"FPR_CN_{mfr}"] = float("nan")
            result[f"specificity_CN_{mfr}"] = float("nan")
    return result


def load_manufacturer_mi(run_dir: Path) -> Optional[float]:
    total_mi = 0.0
    found = False
    for fold in FOLDS:
        for split in ["trainDev", "test"]:
            p = run_dir / f"fold_{fold}" / f"fold_{fold}_{split}_latent_info_per_dim.csv"
            if p.exists():
                df = pd.read_csv(p)
                mfr_cols = [c for c in df.columns if "manufacturer" in c.lower()]
                for col in mfr_cols:
                    vals = df[col].dropna()
                    if not vals.empty:
                        total_mi += float(vals.sum())
                        found = True
    return total_mi if found else None


def build_score_distribution_by_manufacturer(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if "Manufacturer" not in df.columns:
        return pd.DataFrame()
    for dx_label, dx_code in [("CN", 0), ("AD", 1)]:
        dx_sub = df[df["y_true"] == dx_code]
        for mfr in dx_sub["Manufacturer"].unique():
            mfr_sub = dx_sub[dx_sub["Manufacturer"] == mfr]
            scores = mfr_sub["y_score"].values
            rows.append({
                "diagnosis": dx_label,
                "manufacturer": mfr,
                "n": len(scores),
                "min": float(np.min(scores)),
                "p25": float(np.percentile(scores, 25)),
                "median": float(np.median(scores)),
                "p75": float(np.percentile(scores, 75)),
                "max": float(np.max(scores)),
                "mean": float(np.mean(scores)),
            })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    run_dir = resolve_run(args.run_dir)
    locked_dir = resolve_run(args.locked_run_dir)
    beta2p5_dir = resolve_run(args.beta2p5_run_dir)

    print("Philips CN FP audit: beta3p75 latent384")
    for label, rd in [("beta3p75", run_dir), ("locked", locked_dir), ("beta2p5", beta2p5_dir)]:
        pred_path = rd / "classifier_only_readout" / "classifier_sweep_predictions.csv"
        print(f"  [{label}] predictions: {'exists' if pred_path.exists() else 'NOT FOUND'}  ({rd.name})")

    if args.dry_run:
        print("\nDry-run complete. No artifact files read.")
        return 0

    outdir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    if outdir.exists() and not args.overwrite:
        print(f"Output dir exists; pass --overwrite: {outdir}")
        return 1
    outdir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    for label, rd, model, thr in [
        ("beta3p75", run_dir, PRIMARY_MODEL, PRIMARY_THRESHOLD),
        ("locked", locked_dir, PRIMARY_MODEL, PRIMARY_THRESHOLD),
        ("beta2p5", beta2p5_dir, PRIMARY_MODEL, PRIMARY_THRESHOLD),
    ]:
        df = load_predictions(rd, model, thr)
        if df is None:
            print(f"  [{label}] predictions not found — skipping")
            results[label] = None
            continue
        results[label] = df

    for label, df in results.items():
        if df is not None:
            per_fold_df = per_fold_philips_specificity(df)
            write_table(outdir, f"per_fold_philips_specificity_{label}", per_fold_df)
            score_dist_df = build_score_distribution_by_manufacturer(df)
            write_table(outdir, f"score_distribution_by_mfr_{label}", score_dist_df)

    pooled_rows: List[Dict[str, Any]] = []
    for label, df in results.items():
        if df is not None:
            pooled = pooled_philips_specificity(df)
            pooled["model"] = label
            pooled_rows.append(pooled)
    if pooled_rows:
        pooled_df = pd.DataFrame(pooled_rows)
        write_table(outdir, "pooled_philips_specificity_comparison", pooled_df)

    mi_beta3 = load_manufacturer_mi(run_dir)
    mi_beta2 = load_manufacturer_mi(beta2p5_dir)

    now = datetime.now(timezone.utc).isoformat()
    report_lines = [
        "# Philips CN FP Audit — beta3p75 latent384",
        "",
        f"Generated: {now}",
        "",
        "## Context",
        "All CN subjects are Philips scanner. Any CN FP is a Philips CN FP.",
        "Key checks: FP rate not worse than locked; Manufacturer MI not worse than beta2p5.",
        "",
        "## Manufacturer MI (scanner leakage)",
        f"- beta3p75 total MI(z;Manufacturer): {mi_beta3}",
        f"- beta2p5 total MI(z;Manufacturer): {mi_beta2}",
    ]
    if mi_beta3 is not None and mi_beta2 is not None:
        delta = mi_beta3 - mi_beta2
        report_lines.append(f"- Delta: {delta:+.4f} ({'leakage increased' if delta > 0 else 'leakage decreased or unchanged'})")

    report_lines += [
        "",
        "## Pooled CN Specificity by Manufacturer",
    ]
    if pooled_rows:
        pooled_df = pd.DataFrame(pooled_rows)
        report_lines.append(pooled_df.to_markdown(index=False))
    else:
        report_lines.append("_No pooled data available._")

    report_lines += [
        "",
        "## Read-only guarantee",
        "Did not train, fit thresholds, modify tensors, metadata, ledger, or model outputs.",
    ]
    (outdir / "final_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    cmd_log = {
        "created_utc": now,
        "run_name": "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        "training_launched": False,
        "threshold_fitting_performed": False,
    }
    (outdir / "command_log.json").write_text(json.dumps(cmd_log, indent=2), encoding="utf-8")

    print(f"\nOutput: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
