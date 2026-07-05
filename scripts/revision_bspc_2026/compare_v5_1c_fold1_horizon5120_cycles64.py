#!/usr/bin/env python3
"""Read-only comparison for v5.1c Fold-1 horizon5120/cycles64 diagnostic.

Compares candidate Fold 1 against:
  1. v5.1c Fold 1 horizon4480/cycles56
  2. v5.1b Fold 1 horizon4480/cycles56, if available

With --dry-run this only validates source/reference paths and reports candidate
artifact availability. It never trains or modifies run outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_FOLD = 1
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

V51C_SOURCE_RUN = PROJECT_ROOT / (
    "results/revision_bspc_2026/"
    "adni_v5_1c_recover035_ch1_0_2_horizon4480_cycles56_full_5x5"
)
V51C_SOURCE_STAGE_B = V51C_SOURCE_RUN / "classifier_only_readout"
V51B_REFERENCE_RUN = PROJECT_ROOT / (
    "results/revision_bspc_2026/"
    "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
)
V51B_REFERENCE_STAGE_B = V51B_REFERENCE_RUN / "classifier_only_readout"
CANDIDATE_RUN = PROJECT_ROOT / (
    "results/revision_bspc_2026/"
    "adni_v5_1c_recover035_ch1_0_2_fold1_horizon5120_cycles64"
)
CANDIDATE_STAGE_B = CANDIDATE_RUN / "classifier_only_readout"
OUT_DIR = PROJECT_ROOT / (
    "results/revision_bspc_2026/"
    "adni_v5_1c_fold1_horizon5120_cycles64_comparison"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run", type=Path, default=CANDIDATE_RUN)
    parser.add_argument("--candidate-stage-b", type=Path, default=CANDIDATE_STAGE_B)
    parser.add_argument("--v51c-source-run", type=Path, default=V51C_SOURCE_RUN)
    parser.add_argument("--v51c-source-stage-b", type=Path, default=V51C_SOURCE_STAGE_B)
    parser.add_argument("--v51b-reference-run", type=Path, default=V51B_REFERENCE_RUN)
    parser.add_argument("--v51b-reference-stage-b", type=Path, default=V51B_REFERENCE_STAGE_B)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_history(run_dir: Path, run_id: str) -> dict[str, Any]:
    path = run_dir / f"fold_{TARGET_FOLD}" / f"vae_train_history_fold_{TARGET_FOLD}.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    hist = joblib.load(path)
    df = hist.copy() if isinstance(hist, pd.DataFrame) else pd.DataFrame(hist)
    df = df.reset_index(drop=True)
    df.insert(0, "epoch", np.arange(1, len(df) + 1))
    best_idx = int(df["val_loss_modelsel"].astype(float).idxmin())
    best = df.iloc[best_idx]
    final = df.iloc[-1]
    return {
        "run_id": run_id,
        "fold": TARGET_FOLD,
        "best_epoch": int(best["epoch"]),
        "final_epoch_or_stop_epoch": int(final["epoch"]),
        "epochs_after_best": int(final["epoch"] - best["epoch"]),
        "best_valL_beta_max": float(best["val_loss_modelsel"]),
        "last_valL_beta_max": float(final["val_loss_modelsel"]),
        "train_recon_at_best": float(best.get("train_recon", np.nan)),
        "val_recon_at_best": float(best.get("val_recon", np.nan)),
        "train_kld_at_best": float(best.get("train_kld", np.nan)),
        "val_kld_at_best": float(best.get("val_kld", np.nan)),
        "val_kld_over_recon_at_best": float(best.get("val_kld_over_recon", np.nan)),
        "val_beta_kld_over_recon_at_best": float(best.get("val_beta_kld_over_recon", np.nan)),
        "beta_at_best_epoch": float(best.get("beta", np.nan)),
    }


def read_stage_a(run_dir: Path, run_id: str) -> pd.DataFrame:
    paths = sorted(run_dir.glob("all_folds_metrics_MULTI_*.csv"))
    if not paths:
        return pd.DataFrame()
    df = pd.read_csv(paths[0])
    df = df[df["fold"].astype(int).eq(TARGET_FOLD)].copy()
    df.insert(0, "run_id", run_id)
    keep = [
        "run_id",
        "fold",
        "actual_classifier_type",
        "best_clf_params",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1_score",
    ]
    return df[[c for c in keep if c in df.columns]]


def read_stage_b(stage_b_dir: Path, run_id: str) -> pd.DataFrame:
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
    out.insert(0, "run_id", run_id)
    keep = [
        "run_id",
        "fold",
        "model_name",
        "threshold_strategy",
        "threshold",
        "best_params",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "n",
        "n_cn",
        "n_ad",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    return out[[c for c in keep if c in out.columns]]


def read_scanner(run_dir: Path, run_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_label, suffix in [("trainDev", ""), ("test", "_test")]:
        path = run_dir / f"fold_{TARGET_FOLD}" / f"fold_{TARGET_FOLD}{suffix}_scanner_leakage_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["run_id"] = run_id
        row["fold"] = TARGET_FOLD
        row["split"] = split_label
        if pd.notna(row.get("acc_site_latent")) and pd.notna(row.get("acc_site_raw")):
            row["latent_minus_raw_site_acc"] = row["acc_site_latent"] - row["acc_site_raw"]
        rows.append(row)
    return pd.DataFrame(rows)


def read_latent_info(run_dir: Path, run_id: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for split in ["trainDev", "test"]:
        path = run_dir / f"fold_{TARGET_FOLD}" / f"fold_{TARGET_FOLD}_{split}_latent_info_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df.insert(0, "run_id", run_id)
        df.insert(1, "fold", TARGET_FOLD)
        df.insert(2, "split", split)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_md(df: pd.DataFrame, path: Path, title: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if df.empty:
            f.write("_No rows available._\n")
        else:
            f.write(df.to_markdown(index=False, floatfmt=".6g"))
            f.write("\n")


def require(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def candidate_files(candidate_run: Path, candidate_stage_b: Path) -> list[Path]:
    return [
        candidate_run / f"fold_{TARGET_FOLD}" / f"vae_train_history_fold_{TARGET_FOLD}.joblib",
        candidate_stage_b / "classifier_sweep_foldwise_metrics.csv",
    ]


def recommendation_text(stage_b: pd.DataFrame, scanner: pd.DataFrame) -> str:
    source = stage_b[stage_b["run_id"] == "v5.1c_horizon4480_fold1"]
    cand = stage_b[stage_b["run_id"] == "v5.1c_horizon5120_fold1"]
    if source.empty or cand.empty:
        return "Candidate Stage B metrics are unavailable; no recommendation can be made."
    src = source.iloc[0]
    cnd = cand.iloc[0]
    auc_delta = float(cnd["auc"] - src["auc"])
    pr_delta = float(cnd["pr_auc"] - src["pr_auc"])
    ba_delta = float(cnd["balanced_accuracy"] - src["balanced_accuracy"])
    f1_delta = float(cnd["f1"] - src["f1"])

    leakage_note = "Scanner/manufacturer leakage table was available for inspection."
    if not scanner.empty and "latent_minus_raw_site_acc" in scanner.columns:
        leakage_note = "Scanner/manufacturer leakage deltas should be checked in `fold1_manufacturer_leakage.csv`."

    if auc_delta >= 0.03 or (pr_delta > 0 and ba_delta > 0 and f1_delta > 0):
        decision = "Fold 1 materially improved; a FULL v5.1c horizon5120/cycles64 run may be considered."
    else:
        decision = "Fold 1 did not meet the pre-specified improvement rule; do not launch a FULL v5.1c horizon5120/cycles64 run."

    return f"""Decision: **{decision}**

Fold 1 Stage B deltas versus v5.1c horizon4480:

- AUC delta: {auc_delta:.6f}
- PR-AUC delta: {pr_delta:.6f}
- BA delta: {ba_delta:.6f}
- F1 delta: {f1_delta:.6f}

Promotion rule: recommend FULL v5.1c horizon5120/cycles64 only if Fold 1 AUC improves by at least +0.03, or PR-AUC/BA/F1 improve clearly without worsening leakage/subgroups.

{leakage_note}
"""


def main() -> int:
    args = parse_args()
    candidate_run = resolve(args.candidate_run)
    candidate_stage_b = resolve(args.candidate_stage_b)
    v51c_run = resolve(args.v51c_source_run)
    v51c_stage_b = resolve(args.v51c_source_stage_b)
    v51b_run = resolve(args.v51b_reference_run)
    v51b_stage_b = resolve(args.v51b_reference_stage_b)
    outdir = resolve(args.output_dir)

    source_required = [
        v51c_run / f"fold_{TARGET_FOLD}" / f"vae_train_history_fold_{TARGET_FOLD}.joblib",
        v51c_stage_b / "classifier_sweep_foldwise_metrics.csv",
    ]
    require(source_required)
    v51b_available = all(
        p.exists()
        for p in [
            v51b_run / f"fold_{TARGET_FOLD}" / f"vae_train_history_fold_{TARGET_FOLD}.joblib",
            v51b_stage_b / "classifier_sweep_foldwise_metrics.csv",
        ]
    )
    cand_available = all(p.exists() for p in candidate_files(candidate_run, candidate_stage_b))

    if args.dry_run:
        print("Dry-run Fold-1 comparison preflight.")
        print(f"Candidate run: {candidate_run}")
        print(f"Candidate artifacts present: {cand_available}")
        print(f"v5.1c source run: {v51c_run}")
        print(f"v5.1b reference available: {v51b_available}")
        print("No files written.")
        return 0

    require(candidate_files(candidate_run, candidate_stage_b))
    outdir.mkdir(parents=True, exist_ok=True)

    run_specs = [
        ("v5.1c_horizon4480_fold1", v51c_run, v51c_stage_b),
        ("v5.1c_horizon5120_fold1", candidate_run, candidate_stage_b),
    ]
    if v51b_available:
        run_specs.append(("v5.1b_horizon4480_fold1", v51b_run, v51b_stage_b))

    training = pd.DataFrame([read_history(run_dir, label) for label, run_dir, _ in run_specs])
    stage_a = pd.concat([read_stage_a(run_dir, label) for label, run_dir, _ in run_specs], ignore_index=True)
    stage_b = pd.concat([read_stage_b(stage_b_dir, label) for label, _, stage_b_dir in run_specs], ignore_index=True)
    scanner = pd.concat([read_scanner(run_dir, label) for label, run_dir, _ in run_specs], ignore_index=True)
    latent = pd.concat([read_latent_info(run_dir, label) for label, run_dir, _ in run_specs], ignore_index=True)

    outputs = [
        ("fold1_training_comparison", training, "Fold 1 VAE Training Comparison"),
        ("fold1_stage_a_metrics", stage_a, "Fold 1 Stage A Metrics"),
        ("fold1_stage_b_metrics", stage_b, "Fold 1 Stage B Metrics"),
        ("fold1_threshold_confusion", stage_b, "Fold 1 Threshold / Confusion"),
        ("fold1_manufacturer_leakage", scanner, "Fold 1 Scanner / Manufacturer Leakage"),
        ("fold1_latent_information", latent, "Fold 1 Latent Information"),
    ]
    for stem, df, title in outputs:
        df.to_csv(outdir / f"{stem}.csv", index=False)
        write_md(df, outdir / f"{stem}.md", title)

    rec = recommendation_text(stage_b, scanner)
    (outdir / "fold1_horizon5120_recommendation.md").write_text(
        "# Fold 1 Horizon5120/Cycles64 Recommendation\n\n" + rec + "\n",
        encoding="utf-8",
    )

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "target_fold": TARGET_FOLD,
        "candidate_run": str(candidate_run),
        "candidate_stage_b": str(candidate_stage_b),
        "v51c_source_run": str(v51c_run),
        "v51c_source_stage_b": str(v51c_stage_b),
        "v51b_reference_run": str(v51b_run),
        "v51b_reference_stage_b": str(v51b_stage_b),
        "v51b_reference_available": v51b_available,
        "output_dir": str(outdir),
        "read_only": True,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "existing_model_output_modified": False,
    }
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote comparison to {outdir}")
    print(rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
