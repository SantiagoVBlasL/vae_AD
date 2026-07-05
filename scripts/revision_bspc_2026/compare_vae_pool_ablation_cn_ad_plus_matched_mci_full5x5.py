#!/usr/bin/env python3
"""Compare exploratory VAE-pool FULL 5x5 run against locked v5.1b model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_RUN = PROJECT_ROOT / "results/revision_bspc_2026/vae_pool_ablation_cn_ad_plus_matched_mci_full5x5"
REFERENCE_RUN = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
DEFAULT_OUTPUT = CANDIDATE_RUN / "comparison_vs_locked"
PRIMARY_MODEL = "logreg_l2"
PRIMARY_READOUT = "z_plus_age_sex"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run-dir", type=Path, default=CANDIDATE_RUN)
    parser.add_argument("--reference-run-dir", type=Path, default=REFERENCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_No rows._\n"
    sub = df.head(max_rows)
    lines = [
        "| " + " | ".join(sub.columns) + " |",
        "| " + " | ".join(["---"] * len(sub.columns)) + " |",
    ]
    for _, row in sub.iterrows():
        vals: List[str] = []
        for col in sub.columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append(f"{value:.6f}" if np.isfinite(value) else "")
            elif pd.isna(value):
                vals.append("")
            else:
                vals.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_df_pair(root: Path, stem: str, df: pd.DataFrame, max_rows: int = 120) -> None:
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / f"{stem}.csv", index=False)
    (root / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_primary(readout_dir: Path, filename: str) -> pd.DataFrame:
    path = readout_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "readout_feature_set" not in df.columns:
        df["readout_feature_set"] = PRIMARY_READOUT
    mask = (
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
        & df["readout_feature_set"].astype(str).eq(PRIMARY_READOUT)
    )
    return df.loc[mask].copy()


def load_all(run_dir: Path, label: str) -> Dict[str, pd.DataFrame]:
    readout = run_dir / "classifier_only_readout"
    out = {
        "pooled": read_primary(readout, "classifier_sweep_pooled_metrics.csv"),
        "foldwise": read_primary(readout, "classifier_sweep_foldwise_metrics.csv"),
        "thresholds": read_primary(readout, "classifier_sweep_thresholds_by_fold.csv"),
    }
    conf_file = readout / "classifier_sweep_pooled_confusion.csv"
    if conf_file.exists():
        out["pooled_confusion"] = read_primary(readout, "classifier_sweep_pooled_confusion.csv")
    else:
        out["pooled_confusion"] = pd.DataFrame()
    mfr_file = readout / "classifier_sweep_subgroup_metrics_by_manufacturer.csv"
    if mfr_file.exists():
        mfr = pd.read_csv(mfr_file)
        if "readout_feature_set" not in mfr.columns:
            mfr["readout_feature_set"] = PRIMARY_READOUT
        out["manufacturer"] = mfr[
            mfr["model_name"].astype(str).eq(PRIMARY_MODEL)
            & mfr["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
            & mfr["readout_feature_set"].astype(str).eq(PRIMARY_READOUT)
        ].copy()
    else:
        out["manufacturer"] = pd.DataFrame()
    for df in out.values():
        if not df.empty:
            df.insert(0, "run_label", label)
    return out


def metric_summary(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, df in [("locked_v5_1b_ch1_0_2", reference), ("cn_ad_plus_matched_mci_pool", candidate)]:
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["run_label"] = label
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out) == 2:
        c = out[out["run_label"].eq("cn_ad_plus_matched_mci_pool")].iloc[0]
        r = out[out["run_label"].eq("locked_v5_1b_ch1_0_2")].iloc[0]
        delta = {"run_label": "delta_candidate_minus_locked"}
        for col in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "accuracy"]:
            if col in out.columns:
                delta[col] = float(c[col]) - float(r[col])
        for col in ["tn", "fp", "fn", "tp", "n", "n_cn", "n_ad"]:
            if col in out.columns:
                delta[col] = float(c[col]) - float(r[col])
        out = pd.concat([out, pd.DataFrame([delta])], ignore_index=True)
    return out


def make_recommendation(primary: pd.DataFrame) -> str:
    lines = [
        "# Final Recommendation",
        "",
        "This branch is exploratory because diagnosis labels influence VAE pool composition inside each outer-training fold.",
        "",
        f"Ranking uses Stage B classifier-only `{PRIMARY_MODEL}` on `{PRIMARY_READOUT}` at `{PRIMARY_THRESHOLD}`.",
        "",
    ]
    if primary.empty or "delta_candidate_minus_locked" not in set(primary["run_label"]):
        lines.append("Candidate results are not available yet. Do not make a promotion decision.")
        return "\n".join(lines) + "\n"
    cand = primary[primary["run_label"].eq("cn_ad_plus_matched_mci_pool")].iloc[0]
    ref = primary[primary["run_label"].eq("locked_v5_1b_ch1_0_2")].iloc[0]
    auc_ok = float(cand["auc"]) > float(ref["auc"])
    pr_ok = float(cand["pr_auc"]) >= float(ref["pr_auc"])
    ba_ok = float(cand["balanced_accuracy"]) >= float(ref["balanced_accuracy"]) - 0.005
    f1_ok = float(cand["f1"]) >= float(ref["f1"]) - 0.005
    sens_ok = float(cand["sensitivity"]) >= 0.70
    promoted = auc_ok and pr_ok and ba_ok and f1_ok and sens_ok
    lines.extend(
        [
            f"- AUC improves over locked: `{auc_ok}`.",
            f"- PR-AUC nondecreasing: `{pr_ok}`.",
            f"- BA not materially worse: `{ba_ok}`.",
            f"- F1 not materially worse: `{f1_ok}`.",
            f"- Sensitivity >= 0.70: `{sens_ok}`.",
            "",
            f"Decision: `{'promote_for_further_review' if promoted else 'do_not_promote_by_default'}`.",
        ]
    )
    if not promoted:
        lines.append("Keep the locked v5.1b `[1,0,2]` model as the manuscript model unless the integrity audit and external validation provide a stronger reason.")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    candidate_run = resolve(args.candidate_run_dir)
    reference_run = resolve(args.reference_run_dir)
    outdir = resolve(args.output_dir)
    candidate_readout = candidate_run / "classifier_only_readout"
    reference_readout = reference_run / "classifier_only_readout"
    print(f"Candidate readout: {candidate_readout}")
    print(f"Reference readout: {reference_readout}")
    if args.dry_run:
        if not reference_readout.exists():
            raise FileNotFoundError(reference_readout)
        print("Dry-run complete. Candidate outputs are not required yet.")
        return 0

    cand = load_all(candidate_run, "cn_ad_plus_matched_mci_pool")
    ref = load_all(reference_run, "locked_v5_1b_ch1_0_2")
    primary = metric_summary(cand["pooled"], ref["pooled"])
    foldwise = pd.concat([ref["foldwise"], cand["foldwise"]], ignore_index=True)
    thresholds = pd.concat([ref["thresholds"], cand["thresholds"]], ignore_index=True)
    mfr = pd.concat([ref["manufacturer"], cand["manufacturer"]], ignore_index=True)
    write_df_pair(outdir, "primary_comparison", primary)
    write_df_pair(outdir, "foldwise_comparison", foldwise, max_rows=200)
    write_df_pair(outdir, "threshold_comparison", thresholds, max_rows=200)
    write_df_pair(outdir, "manufacturer_subgroup_comparison", mfr, max_rows=240)
    (outdir / "final_recommendation.md").write_text(make_recommendation(primary), encoding="utf-8")
    write_json(
        outdir / "command_log.json",
        {
            "timestamp": now(),
            "candidate_run": str(candidate_run),
            "reference_run": str(reference_run),
            "dry_run": False,
            "wrote": [
                "primary_comparison.csv/.md",
                "foldwise_comparison.csv/.md",
                "threshold_comparison.csv/.md",
                "manufacturer_subgroup_comparison.csv/.md",
                "final_recommendation.md",
            ],
            "training_launched": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "ledger_modified": False,
        },
    )
    print(f"Wrote comparison: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
