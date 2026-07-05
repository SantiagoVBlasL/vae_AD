#!/usr/bin/env python3
"""Compare recover035_latent384_beta3p75_T80_h10000_p560 FULL 5x5 against four references.

References:
  1. locked v5.1b horizon4480/cycles56 [1,0,2]               — primary promotion gate
  2. recover035_full5x5 [1,0,2]                              — original latent256 short-horizon (context)
  3. recover035_longpatience256_T80_h10000_p560 [1,0,2]      — same schedule, latent_dim=256
  4. recover035_latent384_beta2p5_T80_h10000_p560 [1,0,2]    — direct base (same everything except beta_vae)

Isolation principle:
  beta3p75 vs latent384_beta2p5 isolates the beta effect:
  same latent_dim=384, metadata, scheduler, channels, loss — only beta_vae 2.5->3.75.

Promotion rule:
  AUC > 0.782951 AND PR-AUC >= 0.559873 (must beat BOTH locked reference values simultaneously)
  Note: exploratory because outer-fold splits shifted due to 035_S_6927 rescue.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"

DEFAULT_CANDIDATE_RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
DEFAULT_CANDIDATE_READOUT = DEFAULT_CANDIDATE_RUN / "classifier_only_readout"
DEFAULT_OUTPUT = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5_comparison"

REFERENCE_LOCKED_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
REFERENCE_LOCKED_READOUT = REFERENCE_LOCKED_RUN / "classifier_only_readout"

REFERENCE_RECOVER035_RUN = RESULTS / "recover035_full5x5"
REFERENCE_RECOVER035_READOUT = REFERENCE_RECOVER035_RUN / "classifier_only_readout"

REFERENCE_LONGPATIENCE256_RUN = RESULTS / "recover035_longpatience_T80_h10000_p560_full5x5"
REFERENCE_LONGPATIENCE256_READOUT = REFERENCE_LONGPATIENCE256_RUN / "classifier_only_readout"

REFERENCE_LATENT384_BETA2P5_RUN = RESULTS / "recover035_latent384_T80_h10000_p560_full5x5"
REFERENCE_LATENT384_BETA2P5_READOUT = REFERENCE_LATENT384_BETA2P5_RUN / "classifier_only_readout"

RECOVER_SUBJECT = "035_S_6927"
EXCLUDED_SUBJECT = "128_S_2002"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FIXED_THRESHOLD = "fixed_0p5"
METRICS = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "brier"]

LOCKED_AUC = 0.782951
LOCKED_PR_AUC = 0.559873

RUN_ID_LOCKED = "locked_horizon4480_cycles56"
RUN_ID_RECOVER035 = "recover035_full5x5"
RUN_ID_LONGPATIENCE256 = "recover035_longpatience256_T80_h10000_p560"
RUN_ID_LATENT384_BETA2P5 = "recover035_latent384_beta2p5_T80_h10000_p560"
RUN_ID_CAND = "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"

LABEL_LOCKED = "locked v5.1b horizon4480/cycles56 [1,0,2]"
LABEL_RECOVER035 = "recover035 FULL [1,0,2] (latent256, 4480/56, patience=320)"
LABEL_LONGPATIENCE256 = "recover035_longpatience FULL [1,0,2] (latent256, 10000/125, T0=80, p=560)"
LABEL_LATENT384_BETA2P5 = "recover035_latent384_beta2.5 FULL [1,0,2] (latent384, 10000/125, T0=80, p=560)"
LABEL_CAND = "recover035_latent384_beta3.75 FULL [1,0,2] (latent384, 10000/125, T0=80, p=560)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run-dir", type=Path, default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--candidate-readout-dir", type=Path, default=DEFAULT_CANDIDATE_READOUT)
    parser.add_argument("--reference-locked-run-dir", type=Path, default=REFERENCE_LOCKED_RUN)
    parser.add_argument("--reference-locked-readout-dir", type=Path, default=REFERENCE_LOCKED_READOUT)
    parser.add_argument("--reference-recover035-run-dir", type=Path, default=REFERENCE_RECOVER035_RUN)
    parser.add_argument("--reference-recover035-readout-dir", type=Path, default=REFERENCE_RECOVER035_READOUT)
    parser.add_argument("--reference-longpatience256-run-dir", type=Path, default=REFERENCE_LONGPATIENCE256_RUN)
    parser.add_argument("--reference-longpatience256-readout-dir", type=Path, default=REFERENCE_LONGPATIENCE256_READOUT)
    parser.add_argument("--reference-latent384-beta2p5-run-dir", type=Path, default=REFERENCE_LATENT384_BETA2P5_RUN)
    parser.add_argument("--reference-latent384-beta2p5-readout-dir", type=Path, default=REFERENCE_LATENT384_BETA2P5_READOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def load_pooled_metrics(readout_dir: Path, run_id: str, label: str, model: str, threshold: str) -> Dict[str, Any]:
    path = readout_dir / "classifier_sweep_pooled_metrics.csv"
    base = {"run_id": run_id, "label": label}
    if not path.exists():
        return {**base, **{m: float("nan") for m in METRICS}, "note": "pooled_metrics not found"}
    df = pd.read_csv(path)
    mask = df["model_name"].astype(str).eq(model)
    mask &= df["threshold_strategy"].astype(str).eq(threshold)
    df = df[mask].copy()
    if df.empty:
        return {**base, **{m: float("nan") for m in METRICS}, "note": f"no row for {model}/{threshold}"}
    row = df.iloc[0]
    result: Dict[str, Any] = {**base, "note": ""}
    for m in METRICS:
        val = row.get(m, float("nan"))
        try:
            result[m] = float(val)
        except (TypeError, ValueError):
            result[m] = float("nan")
    return result


def load_foldwise_metrics(readout_dir: Path, run_id: str, label: str, model: str, threshold: str) -> pd.DataFrame:
    path = readout_dir / "classifier_sweep_foldwise_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    mask = df["model_name"].astype(str).eq(model)
    mask &= df["threshold_strategy"].astype(str).eq(threshold)
    df = df[mask].copy()
    df["run_id"] = run_id
    df["label"] = label
    return df


def build_comparison_table(
    runs: List[Dict[str, Any]],
    threshold: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        row = load_pooled_metrics(run["readout_dir"], run["run_id"], run["label"], PRIMARY_MODEL, threshold)
        rows.append(row)
    df = pd.DataFrame(rows)
    if "auc" in df.columns:
        df["vs_locked_auc"] = df["auc"] - LOCKED_AUC
        df["vs_locked_pr_auc"] = df["pr_auc"] - LOCKED_PR_AUC
        df["promotes"] = (df["auc"] > LOCKED_AUC) & (df["pr_auc"] >= LOCKED_PR_AUC)
    return df


def write_final_report(
    outdir: Path,
    cand_pooled_youden: Dict[str, Any],
    cand_pooled_fixed: Dict[str, Any],
    comparison_youden: pd.DataFrame,
    comparison_fixed: pd.DataFrame,
    foldwise_summary: pd.DataFrame,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    cand_auc = cand_pooled_youden.get("auc", float("nan"))
    cand_pr = cand_pooled_youden.get("pr_auc", float("nan"))
    promotes = bool(cand_auc > LOCKED_AUC and cand_pr >= LOCKED_PR_AUC)
    verdict = "PROMOTES" if promotes else "DOES NOT PROMOTE"

    lines = [
        "# Stage B Comparison Report",
        "",
        f"Generated: {now}",
        "",
        f"Candidate: {RUN_ID_CAND}",
        f"Controlled change: beta_vae 2.5 -> 3.75 vs latent384_beta2p5 (single diff)",
        "",
        "## Promotion Rule",
        f"- AUC > {LOCKED_AUC} AND PR-AUC >= {LOCKED_PR_AUC} (must beat both simultaneously)",
        f"- Candidate AUC: {cand_auc:.4f}  ({'PASS' if cand_auc > LOCKED_AUC else 'FAIL'})",
        f"- Candidate PR-AUC: {cand_pr:.4f}  ({'PASS' if cand_pr >= LOCKED_PR_AUC else 'FAIL'})",
        f"- **Verdict: {verdict}**",
        "",
        "## Pooled Metrics (Youden threshold)",
        comparison_youden.to_markdown(index=False) if not comparison_youden.empty else "_no data_",
        "",
        "## Pooled Metrics (fixed_0p5 threshold)",
        comparison_fixed.to_markdown(index=False) if not comparison_fixed.empty else "_no data_",
        "",
        "## Foldwise AUC Summary (candidate)",
        foldwise_summary.to_markdown(index=False) if not foldwise_summary.empty else "_no data_",
        "",
        "## Key Questions",
        f"- Is Fold 1 score-range anomaly reduced vs beta2.5? (see score-scale audit)",
        f"- Does stronger beta improve PR-AUC (was 0.4994 in beta2.5)?",
        f"- Is Philips CN FP rate acceptable? (see Philips FP audit)",
        "",
        "## Read-only guarantee",
        "Did not train, fit thresholds, modify tensors, metadata, ledger, or model outputs.",
    ]
    (outdir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = resolve(args.output_dir)
    if outdir.exists() and not args.overwrite:
        print(f"Output dir exists; pass --overwrite to replace: {outdir}")
        return 1
    outdir.mkdir(parents=True, exist_ok=True)

    runs = [
        {"run_id": RUN_ID_LOCKED, "label": LABEL_LOCKED, "run_dir": resolve(args.reference_locked_run_dir), "readout_dir": resolve(args.reference_locked_readout_dir)},
        {"run_id": RUN_ID_RECOVER035, "label": LABEL_RECOVER035, "run_dir": resolve(args.reference_recover035_run_dir), "readout_dir": resolve(args.reference_recover035_readout_dir)},
        {"run_id": RUN_ID_LONGPATIENCE256, "label": LABEL_LONGPATIENCE256, "run_dir": resolve(args.reference_longpatience256_run_dir), "readout_dir": resolve(args.reference_longpatience256_readout_dir)},
        {"run_id": RUN_ID_LATENT384_BETA2P5, "label": LABEL_LATENT384_BETA2P5, "run_dir": resolve(args.reference_latent384_beta2p5_run_dir), "readout_dir": resolve(args.reference_latent384_beta2p5_readout_dir)},
        {"run_id": RUN_ID_CAND, "label": LABEL_CAND, "run_dir": resolve(args.candidate_run_dir), "readout_dir": resolve(args.candidate_readout_dir)},
    ]

    for run in runs:
        status = "exists" if run["readout_dir"].exists() else "NOT FOUND"
        print(f"  [{run['run_id']}] {status}")

    comparison_youden = build_comparison_table(runs, PRIMARY_THRESHOLD)
    comparison_fixed = build_comparison_table(runs, FIXED_THRESHOLD)

    write_table(outdir, "comparison_youden", comparison_youden)
    write_table(outdir, "comparison_fixed", comparison_fixed)

    cand_run = runs[-1]
    foldwise_dfs: List[pd.DataFrame] = []
    for thr in [PRIMARY_THRESHOLD, FIXED_THRESHOLD]:
        fw = load_foldwise_metrics(cand_run["readout_dir"], RUN_ID_CAND, LABEL_CAND, PRIMARY_MODEL, thr)
        if not fw.empty:
            foldwise_dfs.append(fw)
    foldwise_df = pd.concat(foldwise_dfs, ignore_index=True) if foldwise_dfs else pd.DataFrame()
    write_table(outdir, "candidate_foldwise", foldwise_df, max_rows=60)

    cand_youden_row = load_pooled_metrics(cand_run["readout_dir"], RUN_ID_CAND, LABEL_CAND, PRIMARY_MODEL, PRIMARY_THRESHOLD)
    cand_fixed_row = load_pooled_metrics(cand_run["readout_dir"], RUN_ID_CAND, LABEL_CAND, PRIMARY_MODEL, FIXED_THRESHOLD)

    fw_summary = pd.DataFrame()
    if not foldwise_df.empty:
        group_cols = [c for c in ["fold", "threshold_strategy", "auc", "pr_auc", "balanced_accuracy"] if c in foldwise_df.columns]
        fw_summary = foldwise_df[group_cols]

    write_final_report(outdir, cand_youden_row, cand_fixed_row, comparison_youden, comparison_fixed, fw_summary)

    cand_auc = cand_youden_row.get("auc", float("nan"))
    cand_pr = cand_youden_row.get("pr_auc", float("nan"))
    promotes = cand_auc > LOCKED_AUC and cand_pr >= LOCKED_PR_AUC
    print(f"\nCandidate: AUC={cand_auc:.4f}, PR-AUC={cand_pr:.4f}")
    print(f"Promotion: {'PROMOTES' if promotes else 'DOES NOT PROMOTE'}")
    print(f"Output: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
