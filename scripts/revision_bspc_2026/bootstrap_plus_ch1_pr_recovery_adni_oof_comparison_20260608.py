#!/usr/bin/env python3
"""Paired bootstrap comparison of plus-ch1 readouts against promoted ADNI OOF.

The script uses existing OOF predictions only. It does not train, refit,
calibrate, threshold-fit, or modify model/tensor artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


INPUT_DIR = Path("results/revision_bspc_2026/plus_ch1_pr_recovery_readout_batch_20260608")
OUTPUT_DIR = Path("results/revision_bspc_2026/plus_ch1_pr_recovery_bootstrap_comparison_20260608")
PROMOTED = "reference_promoted_ch102_latent384_beta3p75"
CANDIDATES = [
    "plus_ch1_meta_logreg_rank_features",
    "plus_ch1_meta_logreg_pr_auc_selected",
]


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score))


def philips_cn_fpr(df: pd.DataFrame) -> float:
    mask = df["Manufacturer"].eq("Philips") & df["y_true"].eq(0)
    denom = int(mask.sum())
    if denom == 0:
        return np.nan
    return float(df.loc[mask, "y_pred"].sum() / denom)


def metric_bundle(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["y_true"].to_numpy(dtype=int)
    y_score = df["y_score"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=int)
    return {
        "auc": safe_auc(y_true, y_score),
        "pr_auc": safe_ap(y_true, y_score),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "philips_cn_fpr": philips_cn_fpr(df),
    }


def status(y_true: int, y_pred: int) -> str:
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    if y_true == 1 and y_pred == 0:
        return "FN"
    if y_true == 1 and y_pred == 1:
        return "TP"
    raise ValueError((y_true, y_pred))


def compare_subjects(preds: pd.DataFrame, candidate: str) -> pd.DataFrame:
    base = preds[preds["candidate_id"].eq(PROMOTED)].copy()
    cand = preds[preds["candidate_id"].eq(candidate)].copy()
    keep = ["SubjectID", "y_score", "y_pred"]
    merged = base.merge(
        cand[keep],
        on="SubjectID",
        suffixes=("_promoted", "_candidate"),
        validate="one_to_one",
    )
    merged["candidate_id"] = candidate
    merged["promoted_status"] = [
        status(int(y), int(p)) for y, p in zip(merged["y_true"], merged["y_pred_promoted"])
    ]
    merged["candidate_status"] = [
        status(int(y), int(p)) for y, p in zip(merged["y_true"], merged["y_pred_candidate"])
    ]
    merged["promoted_correct"] = merged["y_true"].eq(merged["y_pred_promoted"])
    merged["candidate_correct"] = merged["y_true"].eq(merged["y_pred_candidate"])

    def change_type(row: pd.Series) -> str:
        if row["promoted_correct"] and row["candidate_correct"]:
            return "both_correct"
        if (not row["promoted_correct"]) and row["candidate_correct"]:
            return "fixed_by_candidate"
        if row["promoted_correct"] and (not row["candidate_correct"]):
            return "introduced_by_candidate"
        if row["promoted_status"] == row["candidate_status"]:
            return "both_wrong_same_error"
        return "both_wrong_changed_error"

    merged["change_type"] = merged.apply(change_type, axis=1)
    merged["score_delta_candidate_minus_promoted"] = (
        merged["y_score_candidate"] - merged["y_score_promoted"]
    )
    return merged[
        [
            "candidate_id",
            "SubjectID",
            "ResearchGroup_Mapped",
            "Manufacturer",
            "Age",
            "Sex",
            "fold",
            "y_true",
            "y_score_promoted",
            "y_pred_promoted",
            "promoted_status",
            "y_score_candidate",
            "y_pred_candidate",
            "candidate_status",
            "score_delta_candidate_minus_promoted",
            "change_type",
        ]
    ].copy()


def bootstrap_diffs(
    preds: pd.DataFrame,
    candidate: str,
    n_boot: int,
    seed: int,
    reference: str = PROMOTED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = preds[preds["candidate_id"].eq(reference)].copy()
    cand = preds[preds["candidate_id"].eq(candidate)].copy()
    subjects = sorted(set(base["SubjectID"]) & set(cand["SubjectID"]))
    if len(subjects) != len(base) or len(subjects) != len(cand):
        raise RuntimeError(
            f"Subject mismatch for {candidate} vs {reference}: reference={len(base)}, candidate={len(cand)}, shared={len(subjects)}"
        )

    base = base.set_index("SubjectID").loc[subjects].reset_index()
    cand = cand.set_index("SubjectID").loc[subjects].reset_index()
    observed_base = metric_bundle(base)
    observed_cand = metric_bundle(cand)
    observed = {
        metric: observed_cand[metric] - observed_base[metric]
        for metric in observed_base
    }

    rng = np.random.default_rng(seed)
    n = len(subjects)
    boot = {metric: [] for metric in observed}
    valid = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b_base = base.iloc[idx]
        b_cand = cand.iloc[idx]
        mb = metric_bundle(b_base)
        mc = metric_bundle(b_cand)
        if np.isnan(mb["auc"]) or np.isnan(mc["auc"]):
            continue
        for metric in boot:
            boot[metric].append(mc[metric] - mb[metric])
        valid += 1

    summary_rows = []
    sample_rows = []
    for metric, values in boot.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
        p_le_0 = float(np.mean(arr <= 0))
        p_ge_0 = float(np.mean(arr >= 0))
        summary_rows.append(
            {
                "candidate_id": candidate,
                "reference_id": reference,
                "metric": metric,
                "reference_value": observed_base[metric],
                "candidate_value": observed_cand[metric],
                "observed_diff_candidate_minus_reference": observed[metric],
                "observed_diff_candidate_minus_promoted": observed[metric] if reference == PROMOTED else np.nan,
                "bootstrap_mean_diff": float(np.mean(arr)),
                "ci_2p5": float(ci_low),
                "ci_97p5": float(ci_high),
                "n_boot_requested": n_boot,
                "n_boot_valid": valid,
                "bootstrap_p_diff_le_0": p_le_0,
                "bootstrap_p_diff_ge_0": p_ge_0,
            }
        )
        sample_rows.extend(
            {
                "candidate_id": candidate,
                "reference_id": reference,
                "metric": metric,
                "bootstrap_index": i,
                "diff_candidate_minus_reference": float(v),
            }
            for i, v in enumerate(arr)
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(sample_rows)


def write_csv_md(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    path.with_suffix(".md").write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(INPUT_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    preds = pd.read_csv(input_dir / "predictions.csv")
    required = {PROMOTED, *CANDIDATES}
    missing = required - set(preds["candidate_id"].unique())
    if missing:
        raise RuntimeError(f"Missing prediction candidates: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    boot_samples = []
    subject_tables = []
    for i, candidate in enumerate(CANDIDATES):
        summary, samples = bootstrap_diffs(
            preds,
            candidate,
            n_boot=args.n_boot,
            seed=args.seed + i,
        )
        summaries.append(summary)
        boot_samples.append(samples)
        subject_tables.append(compare_subjects(preds, candidate))

    bootstrap_summary = pd.concat(summaries, ignore_index=True)
    bootstrap_samples = pd.concat(boot_samples, ignore_index=True)

    pairwise_summaries = [bootstrap_summary.copy()]
    pairwise_samples = [bootstrap_samples.copy()]
    pairwise_pairs = [
        (PROMOTED, "plus_ch1_meta_logreg_rank_features"),
        (PROMOTED, "plus_ch1_meta_logreg_pr_auc_selected"),
        ("plus_ch1_meta_logreg_rank_features", "plus_ch1_meta_logreg_pr_auc_selected"),
    ]
    extra_pairwise_pairs = [
        ("plus_ch1_meta_logreg_rank_features", "plus_ch1_meta_logreg_pr_auc_selected"),
    ]
    for j, (reference, candidate) in enumerate(extra_pairwise_pairs):
        summary, samples = bootstrap_diffs(
            preds,
            candidate,
            n_boot=args.n_boot,
            seed=args.seed + 100 + j,
            reference=reference,
        )
        pairwise_summaries.append(summary)
        pairwise_samples.append(samples)
    pairwise_summary = pd.concat(pairwise_summaries, ignore_index=True)
    pairwise_bootstrap_samples = pd.concat(pairwise_samples, ignore_index=True)

    subject_changes = pd.concat(subject_tables, ignore_index=True)
    error_summary = (
        subject_changes.groupby(["candidate_id", "change_type", "ResearchGroup_Mapped", "Manufacturer"], dropna=False)
        .size()
        .reset_index(name="n_subjects")
        .sort_values(["candidate_id", "change_type", "ResearchGroup_Mapped", "Manufacturer"])
    )
    overall_error_summary = (
        subject_changes.groupby(["candidate_id", "change_type"], dropna=False)
        .size()
        .reset_index(name="n_subjects")
        .sort_values(["candidate_id", "change_type"])
    )

    write_csv_md(bootstrap_summary, output_dir / "paired_bootstrap_metric_differences.csv")
    write_csv_md(pairwise_summary, output_dir / "paired_bootstrap_all_pairwise_metric_differences.csv")
    write_csv_md(subject_changes, output_dir / "subject_level_error_changes.csv")
    write_csv_md(error_summary, output_dir / "subject_level_error_changes_by_dx_manufacturer.csv")
    write_csv_md(overall_error_summary, output_dir / "subject_level_error_change_summary.csv")
    bootstrap_samples.to_csv(output_dir / "paired_bootstrap_samples.csv", index=False)
    pairwise_bootstrap_samples.to_csv(output_dir / "paired_bootstrap_all_pairwise_samples.csv", index=False)

    rank_auc = bootstrap_summary[
        bootstrap_summary["candidate_id"].eq("plus_ch1_meta_logreg_rank_features")
        & bootstrap_summary["metric"].eq("auc")
    ].iloc[0]
    pr_auc = bootstrap_summary[
        bootstrap_summary["candidate_id"].eq("plus_ch1_meta_logreg_pr_auc_selected")
        & bootstrap_summary["metric"].eq("auc")
    ].iloc[0]

    readme = f"""# Plus-Ch1 PR-Recovery Paired Bootstrap Comparison

Input predictions: `{input_dir / 'predictions.csv'}`

Reference: `{PROMOTED}`

Candidates:

- `plus_ch1_meta_logreg_rank_features`
- `plus_ch1_meta_logreg_pr_auc_selected`

Bootstrap design:

- subject-level paired bootstrap over ADNI OOF predictions
- `{args.n_boot}` requested resamples per candidate
- differences are candidate minus promoted
- thresholds and predictions are the existing frozen-readout outputs; no refit or threshold selection was performed

Quick AUC summary:

- rank-features AUC difference: `{rank_auc['observed_diff_candidate_minus_promoted']:.6f}` with 95% CI `[{rank_auc['ci_2p5']:.6f}, {rank_auc['ci_97p5']:.6f}]`
- PR-AUC-selected AUC difference: `{pr_auc['observed_diff_candidate_minus_promoted']:.6f}` with 95% CI `[{pr_auc['ci_2p5']:.6f}, {pr_auc['ci_97p5']:.6f}]`
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    interpretation_lines = ["# Final Interpretation", ""]
    for candidate in CANDIDATES:
        sub = bootstrap_summary[bootstrap_summary["candidate_id"].eq(candidate)]
        interpretation_lines.append(f"## {candidate}")
        for metric in ["auc", "pr_auc", "balanced_accuracy", "f1", "philips_cn_fpr"]:
            row = sub[sub["metric"].eq(metric)].iloc[0]
            interpretation_lines.append(
                f"- {metric}: diff `{row['observed_diff_candidate_minus_reference']:.6f}`, "
                f"95% CI `[{row['ci_2p5']:.6f}, {row['ci_97p5']:.6f}]`."
            )
        changes = overall_error_summary[overall_error_summary["candidate_id"].eq(candidate)]
        change_text = ", ".join(f"{r.change_type}={int(r.n_subjects)}" for r in changes.itertuples())
        interpretation_lines.append(f"- Subject-level changes: {change_text}.")
        interpretation_lines.append("")
    interpretation_lines.extend(
        [
            "Guardrails: no VAE training, no tensor modification, no metadata modification,",
            "no model artifact overwrite, no threshold fitting, and no OASIS scoring were performed.",
        ]
    )
    (output_dir / "final_interpretation.md").write_text("\n".join(interpretation_lines) + "\n", encoding="utf-8")

    command_log = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__)),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "reference": PROMOTED,
        "candidates": CANDIDATES,
        "all_pairwise_pairs": pairwise_pairs,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "guardrails": [
            "no VAE training",
            "no OASIS scoring",
            "no threshold fitting",
            "no tensor modification",
            "no metadata modification",
            "no model artifact overwrite",
        ],
    }
    (output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "output_dir": str(output_dir),
        "bootstrap_summary_rows": int(bootstrap_summary.shape[0]),
        "pairwise_summary_rows": int(pairwise_summary.shape[0]),
        "bootstrap_sample_rows": int(bootstrap_samples.shape[0]),
        "pairwise_bootstrap_sample_rows": int(pairwise_bootstrap_samples.shape[0]),
        "subject_change_rows": int(subject_changes.shape[0]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
