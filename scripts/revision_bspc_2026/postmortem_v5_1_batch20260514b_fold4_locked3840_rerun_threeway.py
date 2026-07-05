#!/usr/bin/env python3
"""Read-only three-way postmortem for Fold-4 rerun diagnostics.

Compares:
1. original locked Fold 4
2. fold4_locked3840_rerun
3. fold4_horizon4160_cycles52
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOLD = 4
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

RUNS = {
    "original_locked_fold4": {
        "run_dir": PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate",
        "stage_b_dir": PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep",
    },
    "fold4_locked3840_rerun": {
        "run_dir": PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_ch1_0_2_fold4_locked3840_rerun",
        "stage_b_dir": PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_ch1_0_2_fold4_locked3840_rerun/classifier_only_readout",
    },
    "fold4_horizon4160_cycles52": {
        "run_dir": PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_ch1_0_2_fold4_horizon4160_cycles52",
        "stage_b_dir": PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_ch1_0_2_fold4_horizon4160_cycles52/classifier_only_readout",
    },
}
OUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_fold4_locked3840_rerun_postmortem"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def md_table(df: pd.DataFrame, path: Path) -> None:
    text = "_No rows._" if df.empty else df.to_markdown(index=False, floatfmt=".6g")
    path.write_text(text + "\n", encoding="utf-8")


def history_row(run_id: str, run_dir: Path) -> dict[str, Any]:
    path = run_dir / f"fold_{FOLD}" / f"vae_train_history_fold_{FOLD}.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    obj = joblib.load(path)
    hist = obj.copy() if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    hist = hist.reset_index(drop=True)
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    best_idx = int(hist["val_loss_modelsel"].astype(float).idxmin())
    best = hist.iloc[best_idx]
    train_recon = float(best.get("train_recon", np.nan))
    val_recon = float(best.get("val_recon", np.nan))
    train_kld = float(best.get("train_kld", np.nan))
    val_kld = float(best.get("val_kld", np.nan))
    return {
        "run_id": run_id,
        "fold": FOLD,
        "best_epoch": int(best["epoch"]),
        "stop_epoch": int(hist["epoch"].iloc[-1]),
        "epochs_after_best": int(hist["epoch"].iloc[-1] - best["epoch"]),
        "best_val_loss_beta_max": float(best["val_loss_modelsel"]),
        "train_recon_at_best": train_recon,
        "val_recon_at_best": val_recon,
        "val_minus_train_recon_at_best": val_recon - train_recon,
        "train_kld_at_best": train_kld,
        "val_kld_at_best": val_kld,
        "train_kld_over_recon_at_best": train_kld / train_recon if train_recon else np.nan,
        "val_kld_over_recon_at_best": val_kld / val_recon if val_recon else np.nan,
        "beta_at_best_epoch": float(best.get("beta", np.nan)),
    }


def stage_b_row(run_id: str, stage_b_dir: Path) -> dict[str, Any]:
    path = stage_b_dir / "classifier_sweep_foldwise_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    mask = (
        df["fold"].astype(int).eq(FOLD)
        & df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    )
    if not mask.any():
        raise RuntimeError(f"No primary Stage B row in {path}")
    row = df[mask].iloc[0].to_dict()
    return {
        "run_id": run_id,
        "stage_b_auc": row.get("auc", np.nan),
        "stage_b_pr_auc": row.get("pr_auc", np.nan),
        "stage_b_ba": row.get("balanced_accuracy", np.nan),
        "stage_b_sens": row.get("sensitivity", np.nan),
        "stage_b_spec": row.get("specificity", np.nan),
        "stage_b_f1": row.get("f1", np.nan),
        "stage_b_threshold": row.get("threshold", np.nan),
        "tn": row.get("tn", np.nan),
        "fp": row.get("fp", np.nan),
        "fn": row.get("fn", np.nan),
        "tp": row.get("tp", np.nan),
    }


def stage_a_rows(run_id: str, run_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(run_dir.glob("all_folds_metrics_MULTI_*.csv"))
    if not paths:
        return []
    df = pd.read_csv(paths[0])
    df = df[df["fold"].astype(int).eq(FOLD)].copy()
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "run_id": run_id,
                "fold": FOLD,
                "stage_a_classifier": row.get("actual_classifier_type", ""),
                "stage_a_auc": row.get("auc", np.nan),
                "stage_a_pr_auc": row.get("pr_auc", np.nan),
                "stage_a_ba": row.get("balanced_accuracy", np.nan),
                "stage_a_sens": row.get("sensitivity", np.nan),
                "stage_a_spec": row.get("specificity", np.nan),
                "stage_a_f1": row.get("f1_score", np.nan),
            }
        )
    return rows


def scanner_rows(run_id: str, run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for split, suffix in [("trainDev", ""), ("test", "_test")]:
        path = run_dir / f"fold_{FOLD}" / f"fold_{FOLD}{suffix}_scanner_leakage_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["run_id"] = run_id
        row["fold"] = FOLD
        row["split"] = split
        if pd.notna(row.get("acc_site_latent")) and pd.notna(row.get("acc_site_raw")):
            row["latent_minus_raw_site_acc"] = row["acc_site_latent"] - row["acc_site_raw"]
        rows.append(row)
    return rows


def latent_rows(run_id: str, run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for split in ["trainDev", "test"]:
        path = run_dir / f"fold_{FOLD}" / f"fold_{FOLD}_{split}_latent_info_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            item = row.to_dict()
            item["run_id"] = run_id
            item["fold"] = FOLD
            item["split"] = split
            rows.append(item)
    return rows


def confirm_scope(run_id: str, run_dir: Path, stage_b_dir: Path) -> dict[str, Any]:
    fold_dirs = sorted(p.name for p in run_dir.glob("fold_*") if p.is_dir())
    stage_a_metric_folds: list[int] = []
    metrics = sorted(run_dir.glob("all_folds_metrics_MULTI_*.csv"))
    if metrics:
        stage_a_metric_folds = sorted(pd.read_csv(metrics[0])["fold"].astype(int).unique().tolist())
    stage_b_metric_folds: list[int] = []
    p = stage_b_dir / "classifier_sweep_foldwise_metrics.csv"
    if p.exists():
        stage_b_metric_folds = sorted(pd.read_csv(p)["fold"].astype(int).unique().tolist())
    return {
        "run_id": run_id,
        "top_level_fold_dirs": ",".join(fold_dirs),
        "stage_a_metric_folds": ",".join(map(str, stage_a_metric_folds)),
        "stage_b_metric_folds": ",".join(map(str, stage_b_metric_folds)),
        "only_fold4_confirmed": fold_dirs == ["fold_4"] and stage_a_metric_folds == [4] and (not stage_b_metric_folds or stage_b_metric_folds == [4]),
    }


def required_paths() -> dict[str, list[Path]]:
    req: dict[str, list[Path]] = {}
    for run_id, info in RUNS.items():
        req[run_id] = [
            info["run_dir"] / f"fold_{FOLD}" / f"vae_train_history_fold_{FOLD}.joblib",
            info["stage_b_dir"] / "classifier_sweep_foldwise_metrics.csv",
        ]
    return req


def build_tables() -> dict[str, pd.DataFrame]:
    dynamics = []
    stage_b = []
    stage_a = []
    scanner = []
    latent = []
    scope = []
    for run_id, info in RUNS.items():
        dynamics.append(history_row(run_id, info["run_dir"]))
        stage_b.append(stage_b_row(run_id, info["stage_b_dir"]))
        stage_a.extend(stage_a_rows(run_id, info["run_dir"]))
        scanner.extend(scanner_rows(run_id, info["run_dir"]))
        latent.extend(latent_rows(run_id, info["run_dir"]))
        scope.append(confirm_scope(run_id, info["run_dir"], info["stage_b_dir"]))
    dynamics_df = pd.DataFrame(dynamics)
    stage_b_df = pd.DataFrame(stage_b)
    stage_a_df = pd.DataFrame(stage_a)
    scanner_df = pd.DataFrame(scanner)
    latent_df = pd.DataFrame(latent)
    scope_df = pd.DataFrame(scope)

    main = dynamics_df.merge(stage_b_df, on="run_id", how="left")
    if not scanner_df.empty:
        test_scanner = scanner_df[scanner_df["split"].astype(str).eq("test")][
            ["run_id", "acc_site_raw", "acc_site_latent", "latent_minus_raw_site_acc"]
        ].rename(
            columns={
                "acc_site_raw": "test_acc_site_raw",
                "acc_site_latent": "test_acc_site_latent",
                "latent_minus_raw_site_acc": "test_latent_minus_raw_site_acc",
            }
        )
        main = main.merge(test_scanner, on="run_id", how="left")
    if not latent_df.empty:
        y_latent = latent_df[(latent_df["split"].astype(str).eq("trainDev")) & (latent_df["variable"].astype(str).eq("Y_target"))]
        if not y_latent.empty:
            main = main.merge(
                y_latent[["run_id", "n_active", "total_correlation_nats", "mi_sum_nats"]].rename(
                    columns={
                        "n_active": "trainDev_active_units",
                        "total_correlation_nats": "trainDev_total_correlation_nats",
                        "mi_sum_nats": "trainDev_y_mi_sum_nats",
                    }
                ),
                on="run_id",
                how="left",
            )
    original = main[main["run_id"].eq("original_locked_fold4")].iloc[0]
    for idx, row in main.iterrows():
        if row["run_id"] == "original_locked_fold4":
            continue
        for col in [
            "best_epoch",
            "stop_epoch",
            "best_val_loss_beta_max",
            "stage_b_auc",
            "stage_b_pr_auc",
            "stage_b_ba",
            "stage_b_f1",
            "test_acc_site_latent",
            "test_latent_minus_raw_site_acc",
            "trainDev_total_correlation_nats",
            "val_kld_over_recon_at_best",
        ]:
            if col in main.columns and pd.notna(row.get(col)) and pd.notna(original.get(col)):
                main.loc[idx, f"delta_{col}_vs_original"] = row[col] - original[col]
    return {
        "fold4_threeway_comparison": main,
        "stage_b_fold4_threeway": stage_b_df,
        "training_dynamics_threeway": dynamics_df,
        "stage_a_fold4_threeway": stage_a_df,
        "manufacturer_leakage_threeway": scanner_df,
        "latent_info_threeway": latent_df,
        "run_scope_confirmation": scope_df,
    }


def write_recommendation(outdir: Path, tables: dict[str, pd.DataFrame]) -> None:
    main = tables["fold4_threeway_comparison"]
    original = main[main["run_id"].eq("original_locked_fold4")].iloc[0]
    rerun = main[main["run_id"].eq("fold4_locked3840_rerun")].iloc[0]
    horizon = main[main["run_id"].eq("fold4_horizon4160_cycles52")].iloc[0]
    def fmt(x: Any) -> str:
        return "NA" if pd.isna(x) else f"{float(x):.6f}"
    rerun_matches_horizon = (
        abs(float(rerun["stage_b_auc"]) - float(horizon["stage_b_auc"])) < 0.01
        and abs(float(rerun["stage_b_pr_auc"]) - float(horizon["stage_b_pr_auc"])) < 0.015
    )
    horizon_only_improves = (
        float(horizon["stage_b_auc"]) > float(original["stage_b_auc"])
        and float(horizon["stage_b_pr_auc"]) > float(original["stage_b_pr_auc"])
        and float(rerun["stage_b_auc"]) <= float(original["stage_b_auc"])
    )
    if rerun_matches_horizon:
        decision = "Improvement is consistent with stochastic/rerun variability; do not launch FULL 4160/52."
    elif horizon_only_improves:
        decision = "Only horizon4160 improves; consider FULL 4160/52 only after checking leakage/QC tradeoffs."
    else:
        decision = "No clean horizon-specific effect; do not launch FULL 4160/52."
    lines = [
        "# Fold-4 Locked3840 Rerun Three-Way Recommendation",
        "",
        "| Variant | Stage B AUC | PR-AUC | BA | F1 | best_epoch | stop_epoch |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in main.iterrows():
        lines.append(
            f"| {row['run_id']} | {fmt(row.get('stage_b_auc'))} | {fmt(row.get('stage_b_pr_auc'))} | "
            f"{fmt(row.get('stage_b_ba'))} | {fmt(row.get('stage_b_f1'))} | "
            f"{int(row.get('best_epoch'))} | {int(row.get('stop_epoch'))} |"
        )
    lines.extend(
        [
            "",
            f"**Decision: {decision}**",
            "",
            "Interpretation rule:",
            "- If `fold4_locked3840_rerun` matches `fold4_horizon4160_cycles52`, the earlier horizon improvement is stochastic/rerun variability rather than a horizon effect.",
            "- If only `fold4_horizon4160_cycles52` improves and does not worsen leakage/QC, then a controlled FULL horizon confirmation can be considered.",
            "",
            "Safety: this postmortem is read-only and does not train or modify tensors, metadata, ledgers, configs, or existing model outputs.",
        ]
    )
    (outdir / "recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    req = required_paths()
    missing = {run_id: [str(p) for p in paths if not p.exists()] for run_id, paths in req.items()}
    missing = {run_id: paths for run_id, paths in missing.items() if paths}
    if args.dry_run:
        print("Dry-run three-way postmortem preflight.")
        print(json.dumps({"missing": missing}, indent=2))
        return 0
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + json.dumps(missing, indent=2))
    outdir.mkdir(parents=True, exist_ok=True)
    tables = build_tables()
    for stem, df in tables.items():
        df.to_csv(outdir / f"{stem}.csv", index=False)
        md_table(df, outdir / f"{stem}.md")
    write_recommendation(outdir, tables)
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "runs": {k: {kk: str(vv) for kk, vv in info.items()} for k, info in RUNS.items()},
        "output_dir": str(outdir),
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "config_modified": False,
        "existing_model_output_modified": False,
    }
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote three-way Fold-4 postmortem to {outdir}")
    print(tables["fold4_threeway_comparison"][["run_id", "best_epoch", "stop_epoch", "best_val_loss_beta_max", "stage_b_auc", "stage_b_pr_auc", "stage_b_ba", "stage_b_f1"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
