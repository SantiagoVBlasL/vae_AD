#!/usr/bin/env python3
"""Aggregate VAE pool-composition FAST 3x3 results.

Read-only with respect to tensors, metadata, ledgers, and VAE/model outputs.
When not in dry-run mode, this script writes summary CSV/Markdown files inside
the VAE pool ablation package.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/vae_pool_ablation_fast3x3"
CANDIDATES = (
    "current_all_pool",
    "cn_ad_only_pool",
    "balanced_cn_ad_mci_pool",
    "cn_ad_plus_matched_mci_pool",
)
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_FEATURE_SET = "z_plus_age_sex"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Validate expected files and report completion only.")
    parser.add_argument("--allow-partial", action="store_true", help="Aggregate completed candidates even if others are missing.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_No rows._\n"
    sub = df.head(max_rows).copy()
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


def write_pair(root: Path, stem: str, df: pd.DataFrame, max_rows: int = 120) -> None:
    df.to_csv(root / f"{stem}.csv", index=False)
    (root / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(root: Path) -> pd.DataFrame:
    path = root / "run_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing run_manifest.csv: {path}")
    df = pd.read_csv(path)
    if tuple(df["candidate"].astype(str).tolist()) != CANDIDATES:
        raise RuntimeError("run_manifest.csv candidate order does not match the expected four-candidate plan.")
    return df


def stage_a_complete(run_dir: Path) -> bool:
    checkpoints = [run_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt" for fold in range(1, 4)]
    return all(p.exists() for p in checkpoints) and (run_dir / "run_config.json").exists()


def stage_b_complete(readout_dir: Path) -> bool:
    required = [
        "classifier_sweep_pooled_metrics.csv",
        "classifier_sweep_foldwise_metrics.csv",
        "classifier_sweep_predictions.csv",
        "classifier_sweep_thresholds_by_fold.csv",
    ]
    return all((readout_dir / name).exists() for name in required)


def completion_status(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in manifest.iterrows():
        run_dir = resolve(Path(str(row["run_dir"])))
        readout_dir = run_dir / "classifier_only_readout"
        rows.append(
            {
                "candidate": row["candidate"],
                "run_dir": str(run_dir),
                "readout_dir": str(readout_dir),
                "stage_a_complete": stage_a_complete(run_dir),
                "stage_b_complete": stage_b_complete(readout_dir),
                "pooled_metrics_exists": (readout_dir / "classifier_sweep_pooled_metrics.csv").exists(),
                "foldwise_metrics_exists": (readout_dir / "classifier_sweep_foldwise_metrics.csv").exists(),
                "thresholds_exists": (readout_dir / "classifier_sweep_thresholds_by_fold.csv").exists(),
                "manufacturer_subgroup_exists": (readout_dir / "classifier_sweep_subgroup_metrics_by_manufacturer.csv").exists(),
            }
        )
    return pd.DataFrame(rows)


def add_candidate(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "candidate", str(row["candidate"]))
    out.insert(1, "vae_pool_composition_strategy", str(row["vae_pool_composition_strategy"]))
    out.insert(2, "exploratory_uses_diagnosis_for_pool_composition", bool(row["exploratory_uses_diagnosis_for_pool_composition"]))
    return out


def primary_filter(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "readout_feature_set" not in out.columns:
        out["readout_feature_set"] = PRIMARY_FEATURE_SET
    mask = (
        out["model_name"].astype(str).eq(PRIMARY_MODEL)
        & out["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
        & out["readout_feature_set"].astype(str).eq(PRIMARY_FEATURE_SET)
    )
    return out.loc[mask].copy()


def collect_stage_b(manifest: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled_parts: List[pd.DataFrame] = []
    foldwise_parts: List[pd.DataFrame] = []
    threshold_parts: List[pd.DataFrame] = []
    confusion_parts: List[pd.DataFrame] = []
    subgroup_parts: List[pd.DataFrame] = []
    for _, row in manifest.iterrows():
        run_dir = resolve(Path(str(row["run_dir"])))
        readout = run_dir / "classifier_only_readout"
        if (readout / "classifier_sweep_pooled_metrics.csv").exists():
            df = primary_filter(pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv"))
            if not df.empty:
                pooled_parts.append(add_candidate(df, row))
        if (readout / "classifier_sweep_foldwise_metrics.csv").exists():
            df = primary_filter(pd.read_csv(readout / "classifier_sweep_foldwise_metrics.csv"))
            if not df.empty:
                foldwise_parts.append(add_candidate(df, row))
        if (readout / "classifier_sweep_thresholds_by_fold.csv").exists():
            df = primary_filter(pd.read_csv(readout / "classifier_sweep_thresholds_by_fold.csv"))
            if not df.empty:
                threshold_parts.append(add_candidate(df, row))
        if (readout / "classifier_sweep_confusion_by_fold.csv").exists():
            df = primary_filter(pd.read_csv(readout / "classifier_sweep_confusion_by_fold.csv"))
            if not df.empty:
                confusion_parts.append(add_candidate(df, row))
        if (readout / "classifier_sweep_subgroup_metrics_by_manufacturer.csv").exists():
            df = pd.read_csv(readout / "classifier_sweep_subgroup_metrics_by_manufacturer.csv")
            if "readout_feature_set" not in df.columns:
                df["readout_feature_set"] = PRIMARY_FEATURE_SET
            mask = (
                df["model_name"].astype(str).eq(PRIMARY_MODEL)
                & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
                & df["readout_feature_set"].astype(str).eq(PRIMARY_FEATURE_SET)
            )
            df = df.loc[mask].copy()
            if not df.empty:
                subgroup_parts.append(add_candidate(df, row))

    def concat(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    return concat(pooled_parts), concat(foldwise_parts), concat(threshold_parts), concat(confusion_parts), concat(subgroup_parts)


def collect_pool_summaries(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in manifest.iterrows():
        run_dir = resolve(Path(str(row["run_dir"])))
        candidate = str(row["candidate"])
        for fold in range(1, 4):
            path = run_dir / f"fold_{fold}" / f"vae_pool_composition_strategy_summary_fold_{fold}.csv"
            if path.exists():
                df = pd.read_csv(path)
                df.insert(0, "candidate", candidate)
                rows.extend(df.to_dict("records"))
            elif candidate == "current_all_pool" and (run_dir / f"fold_{fold}" / "vae_training_pool_tensor_idx.npy").exists():
                rows.append(
                    {
                        "candidate": candidate,
                        "fold": fold,
                        "vae_pool_composition_strategy": "current_all_pool",
                        "n_after": np.nan,
                        "selection_note": "default_current_pool_no_strategy_summary_written",
                    }
                )
    return pd.DataFrame(rows)


def collect_rate_distortion(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in manifest.iterrows():
        run_dir = resolve(Path(str(row["run_dir"])))
        for fold in range(1, 4):
            fold_dir = run_dir / f"fold_{fold}"
            candidates = sorted(fold_dir.glob("*rate*distortion*.csv")) + sorted(fold_dir.glob("*Rate*Distortion*.csv"))
            for path in candidates:
                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue
                summary: Dict[str, Any] = {
                    "candidate": row["candidate"],
                    "fold": fold,
                    "source_file": str(path),
                    "n_rows": len(df),
                }
                for col in ["KLD/R", "kld_over_recon", "beta_kld_over_recon", "val_beta_kld_over_recon"]:
                    if col in df.columns:
                        values = pd.to_numeric(df[col], errors="coerce").dropna()
                        if not values.empty:
                            summary[f"{col}_last"] = float(values.iloc[-1])
                            summary[f"{col}_min"] = float(values.min())
                            summary[f"{col}_max"] = float(values.max())
                rows.append(summary)
    return pd.DataFrame(rows)


def make_recommendation(primary: pd.DataFrame, status: pd.DataFrame) -> str:
    lines = [
        "# Final Recommendation",
        "",
        "Ranking uses Stage B classifier-only `logreg_l2` with `z_plus_age_sex` at `inner_oof_target_sens_ge_0p70_max_spec`.",
        "Stage A dummy logreg metrics are not used for ranking.",
        "",
    ]
    if primary.empty:
        lines.extend(
            [
                "No completed Stage B primary results were found yet.",
                "",
                "Run the launcher with `--confirm-training` after reviewing the dry-run manifest. Non-current VAE pool strategies remain exploratory because diagnosis labels are used to alter the VAE pool composition.",
            ]
        )
        return "\n".join(lines) + "\n"

    sort_cols = [c for c in ["auc", "pr_auc", "balanced_accuracy", "f1"] if c in primary.columns]
    ranked = primary.sort_values(sort_cols, ascending=False) if sort_cols else primary.copy()
    best = ranked.iloc[0]
    current = primary[primary["candidate"].eq("current_all_pool")]
    lines.append("Completed candidates ranked by AUC, then PR-AUC, BA, and F1:")
    lines.append("")
    lines.append(md_table(ranked[["candidate", *sort_cols]], max_rows=20))
    if not current.empty and "auc" in primary.columns and "pr_auc" in primary.columns:
        cur = current.iloc[0]
        delta_auc = float(best["auc"]) - float(cur["auc"]) if not math.isnan(float(best["auc"])) else np.nan
        delta_pr = float(best["pr_auc"]) - float(cur["pr_auc"]) if not math.isnan(float(best["pr_auc"])) else np.nan
        lines.extend(
            [
                "",
                f"Best candidate: `{best['candidate']}`.",
                f"Delta vs `current_all_pool`: AUC `{delta_auc:.6f}`, PR-AUC `{delta_pr:.6f}`.",
            ]
        )
    lines.extend(
        [
            "",
            "A FULL 5x5 follow-up is only defensible if the FAST winner improves AUC and PR-AUC against `current_all_pool`, preserves BA/F1 and sensitivity, and does not worsen scanner/manufacturer leakage.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    root = resolve(args.output_root)
    manifest = load_manifest(root)
    status = completion_status(manifest)

    print(f"Output root: {root}")
    print(status[["candidate", "stage_a_complete", "stage_b_complete"]].to_string(index=False))
    if args.dry_run:
        incomplete = status[~status["stage_b_complete"]]
        print(f"Dry-run complete. Completed Stage B candidates: {len(status) - len(incomplete)}/{len(status)}")
        return 0

    incomplete = status[~status["stage_b_complete"]]
    if not incomplete.empty and not args.allow_partial:
        raise SystemExit(
            "Not all candidates have complete Stage B outputs. Use --allow-partial to aggregate completed candidates.\n"
            + incomplete[["candidate", "stage_a_complete", "stage_b_complete"]].to_string(index=False)
        )

    primary, foldwise, thresholds, confusion, subgroup = collect_stage_b(manifest)
    pool_summaries = collect_pool_summaries(manifest)
    rate_distortion = collect_rate_distortion(manifest)

    if not primary.empty:
        sort_cols = [c for c in ["auc", "pr_auc", "balanced_accuracy", "f1"] if c in primary.columns]
        primary = primary.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    if not foldwise.empty and "fold" in foldwise.columns:
        foldwise = foldwise.sort_values(["candidate", "fold"]).reset_index(drop=True)
    if not thresholds.empty and "fold" in thresholds.columns:
        thresholds = thresholds.sort_values(["candidate", "fold"]).reset_index(drop=True)

    write_pair(root, "primary_results", primary)
    write_pair(root, "foldwise_metrics", foldwise, max_rows=200)
    write_pair(root, "threshold_by_fold", thresholds, max_rows=200)
    write_pair(root, "confusion_by_fold", confusion, max_rows=200)
    write_pair(root, "manufacturer_subgroup_metrics", subgroup, max_rows=200)
    write_pair(root, "stage_completion_status", status)
    write_pair(root, "pool_composition_strategy_summaries_from_runs", pool_summaries, max_rows=200)
    write_pair(root, "rate_distortion_by_candidate", rate_distortion, max_rows=200)
    (root / "final_recommendation.md").write_text(make_recommendation(primary, status), encoding="utf-8")
    write_json(
        root / "command_log_aggregation.json",
        {
            "timestamp": now(),
            "output_root": str(root),
            "dry_run": False,
            "allow_partial": bool(args.allow_partial),
            "completed_stage_b": int(status["stage_b_complete"].sum()),
            "total_candidates": int(len(status)),
            "wrote": [
                "primary_results.csv/.md",
                "foldwise_metrics.csv/.md",
                "threshold_by_fold.csv/.md",
                "confusion_by_fold.csv/.md",
                "manufacturer_subgroup_metrics.csv/.md",
                "stage_completion_status.csv/.md",
                "pool_composition_strategy_summaries_from_runs.csv/.md",
                "rate_distortion_by_candidate.csv/.md",
                "final_recommendation.md",
            ],
        },
    )
    print("Aggregation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
