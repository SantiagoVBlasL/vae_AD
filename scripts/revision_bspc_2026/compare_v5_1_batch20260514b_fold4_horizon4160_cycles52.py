#!/usr/bin/env python3
"""Read-only comparison for the Fold-4 horizon4160/cycles52 diagnostic.

The script compares candidate Fold 4 against the locked current Fold 4 once the
diagnostic has been run. With --dry-run it only validates paths and reports what
would be read.
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
LOCKED_RUN = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
LOCKED_STAGE_B = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
CANDIDATE_RUN = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_ch1_0_2_fold4_horizon4160_cycles52"
CANDIDATE_STAGE_B = CANDIDATE_RUN / "classifier_only_readout"
OUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_fold4_horizon4160_cycles52_comparison"
TARGET_FOLD = 4
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--locked-run", type=Path, default=LOCKED_RUN)
    parser.add_argument("--locked-stage-b", type=Path, default=LOCKED_STAGE_B)
    parser.add_argument("--candidate-run", type=Path, default=CANDIDATE_RUN)
    parser.add_argument("--candidate-stage-b", type=Path, default=CANDIDATE_STAGE_B)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def history_summary(run_dir: Path, label: str) -> dict[str, Any]:
    path = run_dir / f"fold_{TARGET_FOLD}" / f"vae_train_history_fold_{TARGET_FOLD}.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    obj = joblib.load(path)
    df = obj.copy() if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    df = df.reset_index(drop=True)
    df.insert(0, "epoch", np.arange(1, len(df) + 1))
    best_idx = int(df["val_loss_modelsel"].astype(float).idxmin())
    best_epoch = best_idx + 1
    row = df.iloc[best_idx]
    train_recon = float(row.get("train_recon", np.nan))
    val_recon = float(row.get("val_recon", np.nan))
    train_kld = float(row.get("train_kld", np.nan))
    val_kld = float(row.get("val_kld", np.nan))
    return {
        "run_id": label,
        "fold": TARGET_FOLD,
        "best_epoch": best_epoch,
        "final_epoch_or_early_stop_epoch": int(df["epoch"].iloc[-1]),
        "best_val_loss_beta_max": float(row["val_loss_modelsel"]),
        "train_recon_at_best": train_recon,
        "val_recon_at_best": val_recon,
        "train_kld_at_best": train_kld,
        "val_kld_at_best": val_kld,
        "train_kld_over_recon_at_best": train_kld / train_recon if train_recon else np.nan,
        "val_kld_over_recon_at_best": val_kld / val_recon if val_recon else np.nan,
        "beta_at_best_epoch": float(row.get("beta", np.nan)),
    }


def stage_a_summary(run_dir: Path, label: str) -> pd.DataFrame:
    paths = sorted(run_dir.glob("all_folds_metrics_MULTI_*.csv"))
    if not paths:
        return pd.DataFrame()
    df = pd.read_csv(paths[0])
    df = df[df["fold"].astype(int).eq(TARGET_FOLD)].copy()
    df.insert(0, "run_id", label)
    keep = [
        "run_id",
        "fold",
        "actual_classifier_type",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1_score",
    ]
    return df[[c for c in keep if c in df.columns]]


def stage_b_summary(stage_b_dir: Path, label: str) -> pd.DataFrame:
    path = stage_b_dir / "classifier_sweep_foldwise_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    mask = (
        df["fold"].astype(int).eq(TARGET_FOLD)
        & df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    )
    out = df[mask].copy()
    out.insert(0, "run_id", label)
    keep = [
        "run_id",
        "fold",
        "model_name",
        "threshold_strategy",
        "threshold",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    return out[[c for c in keep if c in out.columns]]


def scanner_summary(run_dir: Path, label: str) -> pd.DataFrame:
    rows = []
    for split_label, suffix in [("trainDev", ""), ("test", "_test")]:
        path = run_dir / f"fold_{TARGET_FOLD}" / f"fold_{TARGET_FOLD}{suffix}_scanner_leakage_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["run_id"] = label
        row["fold"] = TARGET_FOLD
        row["split"] = split_label
        if pd.notna(row.get("acc_site_latent")) and pd.notna(row.get("acc_site_raw")):
            row["latent_minus_raw_site_acc"] = row["acc_site_latent"] - row["acc_site_raw"]
        rows.append(row)
    return pd.DataFrame(rows)


def latent_info_summary(run_dir: Path, label: str) -> pd.DataFrame:
    rows = []
    for split in ["trainDev", "test"]:
        path = run_dir / f"fold_{TARGET_FOLD}" / f"fold_{TARGET_FOLD}_{split}_latent_info_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df.insert(0, "run_id", label)
        df.insert(1, "fold", TARGET_FOLD)
        df.insert(2, "split", split)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_md(df: pd.DataFrame, path: Path) -> None:
    text = df.to_markdown(index=False, floatfmt=".6g") if not df.empty else "_No rows available._"
    path.write_text(text + "\n", encoding="utf-8")


def require_for_real(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required candidate artifacts:\n" + "\n".join(missing))


def main() -> int:
    args = parse_args()
    locked_run = resolve(args.locked_run)
    locked_stage_b = resolve(args.locked_stage_b)
    candidate_run = resolve(args.candidate_run)
    candidate_stage_b = resolve(args.candidate_stage_b)
    outdir = resolve(args.output_dir)

    required_locked = [
        locked_run / f"fold_{TARGET_FOLD}" / f"vae_train_history_fold_{TARGET_FOLD}.joblib",
        locked_stage_b / "classifier_sweep_foldwise_metrics.csv",
    ]
    require_for_real(required_locked)
    required_candidate = [
        candidate_run / f"fold_{TARGET_FOLD}" / f"vae_train_history_fold_{TARGET_FOLD}.joblib",
        candidate_stage_b / "classifier_sweep_foldwise_metrics.csv",
    ]
    if args.dry_run:
        print("Dry-run comparison preflight.")
        print(f"Locked run: {locked_run}")
        print(f"Candidate run: {candidate_run}")
        print("Candidate artifacts present:", all(p.exists() for p in required_candidate))
        print("No files written.")
        return 0
    require_for_real(required_candidate)
    outdir.mkdir(parents=True, exist_ok=True)

    vae = pd.DataFrame(
        [
            history_summary(locked_run, "locked_current_fold4"),
            history_summary(candidate_run, "horizon4160_cycles52_fold4"),
        ]
    )
    stage_a = pd.concat(
        [
            stage_a_summary(locked_run, "locked_current_fold4"),
            stage_a_summary(candidate_run, "horizon4160_cycles52_fold4"),
        ],
        ignore_index=True,
    )
    stage_b = pd.concat(
        [
            stage_b_summary(locked_stage_b, "locked_current_fold4"),
            stage_b_summary(candidate_stage_b, "horizon4160_cycles52_fold4"),
        ],
        ignore_index=True,
    )
    scanner = pd.concat(
        [
            scanner_summary(locked_run, "locked_current_fold4"),
            scanner_summary(candidate_run, "horizon4160_cycles52_fold4"),
        ],
        ignore_index=True,
    )
    latent = pd.concat(
        [
            latent_info_summary(locked_run, "locked_current_fold4"),
            latent_info_summary(candidate_run, "horizon4160_cycles52_fold4"),
        ],
        ignore_index=True,
    )

    vae.to_csv(outdir / "fold4_vae_training_comparison.csv", index=False)
    stage_a.to_csv(outdir / "fold4_stage_a_classifier_comparison.csv", index=False)
    stage_b.to_csv(outdir / "fold4_stage_b_logreg_l2_comparison.csv", index=False)
    scanner.to_csv(outdir / "fold4_scanner_leakage_comparison.csv", index=False)
    latent.to_csv(outdir / "fold4_latent_information_comparison.csv", index=False)
    write_md(vae, outdir / "fold4_vae_training_comparison.md")
    write_md(stage_a, outdir / "fold4_stage_a_classifier_comparison.md")
    write_md(stage_b, outdir / "fold4_stage_b_logreg_l2_comparison.md")
    write_md(scanner, outdir / "fold4_scanner_leakage_comparison.md")
    write_md(latent, outdir / "fold4_latent_information_comparison.md")

    recommendation = [
        "# Fold-4 Horizon4160/Cycles52 Diagnostic Comparison",
        "",
        "This read-only comparison evaluates only outer fold 4 against the locked fold-4 baseline.",
        "",
        "Promotion or follow-up logic:",
        "- This is diagnostic only and cannot replace the locked manuscript model because it runs one fold.",
        "- Treat improvement as evidence only if Fold-4 Stage B AUC/PR-AUC improve together, VAE ValL(beta_max) improves, and scanner/manufacturer leakage does not worsen.",
        "- If threshold metrics improve but AUC/PR-AUC do not, the result should be interpreted as another threshold operating-point effect rather than improved ranking.",
    ]
    (outdir / "fold4_horizon_diagnostic_recommendation.md").write_text("\n".join(recommendation) + "\n", encoding="utf-8")
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "locked_run": str(locked_run),
        "candidate_run": str(candidate_run),
        "target_fold": TARGET_FOLD,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "existing_model_output_modified": False,
        "output_dir": str(outdir),
    }
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote comparison to {outdir}")
    if not stage_b.empty:
        print(stage_b.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
