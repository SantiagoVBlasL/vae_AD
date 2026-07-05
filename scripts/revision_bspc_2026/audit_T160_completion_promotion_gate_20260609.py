#!/usr/bin/env python3
"""Read-only completion and promotion-gate audit for T160 scheduler candidate.

Candidate:
  recover035_latent384_beta3p75_T160_h10000_p560_full5x5

Scientific question:
  Does changing lr_scheduler_T0 from 80 to 160 improve latent geometry or
  ADNI OOF performance while keeping all other promoted settings fixed?

Comparison set:
  - promoted [1,0,2] beta3.75 T80 (primary reference)
  - ch1-only beta3.75 T80
  - beta9.5 T80
  - chweightedPearson50 T80
  - beta6.5 T80
  - chmeanloss T80

This script reads existing artifacts only. It does not train a VAE, run the
classifier-only sweep, fit OASIS thresholds/calibration, modify tensors,
metadata, ledgers, or overwrite model artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/revision_bspc_2026"
OUT_DEFAULT = RESULTS / "T160_completion_promotion_gate_audit_20260609"

PRIMARY_MODEL_RAW = "logreg_l2"
PRIMARY_MODEL_OOF = "logreg_l2_original"
PRIMARY_FEATURES = "z_plus_age_sex"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_CALIB = "oof_ecdf"

PROMOTED_AUC = 0.795155
PROMOTED_PR_AUC = 0.573934
CH1_AUC = 0.800378
CH1_PR_AUC = 0.585842
PHILIPS_FPR_GATE = 0.4545

# LR scheduler parameters from T160 run config
T0_CANDIDATE = 160
T0_PROMOTED = 80

# Cyclical beta schedule parameters (same for all runs)
BETA_N_CYCLES = 125
BETA_EPOCHS_TOTAL = 10000
BETA_RATIO_INCREASE = 0.4

PLUS_CH1_BATCH = RESULTS / "plus_ch1_pr_recovery_readout_batch_20260608"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    run_dir: Path
    oof_dir: Path | None
    beta: float
    latent_dim: int
    channel_set: str
    role: str
    lr_T0: int = 80


RUNS = [
    RunSpec(
        "candidate_latent384_beta3p75_T160",
        RESULTS / "recover035_latent384_beta3p75_T160_h10000_p560_full5x5",
        RESULTS / "recover035_latent384_beta3p75_T160_stageB_oof_score_calibration",
        3.75,
        384,
        "[1,0,2]",
        "candidate",
        lr_T0=160,
    ),
    RunSpec(
        "promoted_latent384_beta3p75_T80",
        RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration",
        3.75,
        384,
        "[1,0,2]",
        "promoted_reference",
        lr_T0=80,
    ),
    RunSpec(
        "ch1only_latent384_beta3p75_T80",
        RESULTS / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5",
        RESULTS / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration",
        3.75,
        384,
        "[1]",
        "ch1_reference",
        lr_T0=80,
    ),
    RunSpec(
        "candidate_latent384_beta9p5_T80",
        RESULTS / "recover035_latent384_beta9p5_T80_h10000_p560_full5x5",
        RESULTS / "recover035_latent384_beta9p5_stageB_oof_score_calibration",
        9.5,
        384,
        "[1,0,2]",
        "beta_reference",
        lr_T0=80,
    ),
    RunSpec(
        "chweighted_latent384_beta3p75_T80",
        RESULTS / "recover035_latent384_beta3p75_chweightedPearson50_T80_h10000_p560_full5x5",
        RESULTS / "recover035_latent384_beta3p75_chweightedPearson50_stageB_oof_score_calibration",
        3.75,
        384,
        "[1,0,2]",
        "loss_reference",
        lr_T0=80,
    ),
    RunSpec(
        "latent384_beta6p5_T80",
        RESULTS / "recover035_latent384_beta6p5_T80_h10000_p560_full5x5",
        RESULTS / "recover035_latent384_beta6p5_stageB_oof_score_calibration",
        6.5,
        384,
        "[1,0,2]",
        "beta_reference",
        lr_T0=80,
    ),
    RunSpec(
        "chmeanloss_latent384_beta3p75_T80",
        RESULTS / "recover035_latent384_beta3p75_chmeanloss_T80_h10000_p560_full5x5",
        RESULTS / "recover035_latent384_beta3p75_chmeanloss_stageB_oof_score_calibration",
        3.75,
        384,
        "[1,0,2]",
        "loss_reference",
        lr_T0=80,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def safe_div(num: float, den: float) -> float:
    if den == 0 or math.isnan(den):
        return float("nan")
    return num / den


def to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._\n"
    try:
        return df.to_markdown(index=False) + "\n"
    except Exception:
        return df.to_string(index=False) + "\n"


def write_table(df: pd.DataFrame, stem: str, out: Path) -> None:
    df.to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}.md").write_text(to_md(df), encoding="utf-8")


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join(str(part) for part in col if str(part) and str(part) != "nan").strip("_")
            for col in out.columns
        ]
    return out


def summarize(df: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = [c for c in metrics if c in df.columns]
    if not cols:
        return pd.DataFrame()
    return flatten_columns(df.groupby(by, dropna=False)[cols].agg(["mean", "std", "min", "max"]).reset_index())


def find_stagea_metrics(run_dir: Path) -> Path | None:
    matches = sorted(run_dir.glob("all_folds_metrics_MULTI*.csv"))
    return matches[0] if matches else None


def completion_status(run: RunSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    readout_dir = run.run_dir / "classifier_only_readout"
    for fold in range(1, 6):
        fold_dir = run.run_dir / f"fold_{fold}"
        vae_checks = {
            "fold_dir": fold_dir.exists(),
            "vae_model_saved": (fold_dir / f"vae_model_fold_{fold}.pt").exists(),
            "vae_history": (fold_dir / f"vae_train_history_fold_{fold}.joblib").exists(),
            "rate_distortion": (fold_dir / f"fold_{fold}_rate_distortion.csv").exists(),
            "trainDev_latent_info": (fold_dir / f"fold_{fold}_trainDev_latent_info_summary.csv").exists(),
            "test_latent_info": (fold_dir / f"fold_{fold}_test_latent_info_summary.csv").exists(),
            "trainDev_scanner_leakage": (fold_dir / f"fold_{fold}_scanner_leakage_summary.csv").exists(),
            "test_scanner_leakage": (fold_dir / f"fold_{fold}_test_scanner_leakage_summary.csv").exists(),
            "stageA_logreg_predictions": (fold_dir / "test_predictions_logreg.csv").exists(),
            "stageA_svm_predictions": (fold_dir / "test_predictions_svm.csv").exists(),
        }
        stageb_checks = {
            "latent_cache_trainDev": (readout_dir / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv").exists(),
            "latent_cache_test": (readout_dir / "latent_cache" / f"fold_{fold}_test_latent_mu.csv").exists(),
        }
        row: dict[str, Any] = {"run_id": run.run_id, "fold": fold}
        row.update(vae_checks)
        row.update(stageb_checks)
        row["stageA_classifier_artifacts_present"] = (
            (fold_dir / f"classifier_logreg_final_pipeline_fold_{fold}.joblib").exists()
            and (fold_dir / f"classifier_svm_final_pipeline_fold_{fold}.joblib").exists()
        )
        row["classifier_only_readout_present"] = readout_dir.exists()
        row["oof_score_calibration_present"] = bool(run.oof_dir and run.oof_dir.exists())
        row["vae_stageA_complete"] = all(vae_checks.values()) and bool(row["stageA_classifier_artifacts_present"])
        row["stageB_complete"] = all(stageb_checks.values()) and bool(row["classifier_only_readout_present"])
        rows.append(row)
    global_checks = {
        "stageA_metrics": bool(find_stagea_metrics(run.run_dir)),
        "classifier_only_readout": (readout_dir / "classifier_sweep_pooled_metrics.csv").exists(),
        "classifier_only_foldwise": (readout_dir / "classifier_sweep_foldwise_metrics.csv").exists(),
        "oof_pooled": bool(run.oof_dir and (run.oof_dir / "calib_pooled_metrics.csv").exists()),
        "oof_foldwise": bool(run.oof_dir and (run.oof_dir / "calib_foldwise_metrics.csv").exists()),
        "oof_predictions": bool(run.oof_dir and (run.oof_dir / "calib_predictions.csv").exists()),
    }
    rows.append(
        {
            "run_id": run.run_id,
            "fold": "all",
            **global_checks,
            "vae_stageA_complete": bool(global_checks["stageA_metrics"]),
            "stageB_complete": bool(global_checks["classifier_only_readout"] and global_checks["oof_pooled"]),
        }
    )
    return pd.DataFrame(rows)


def stagea_foldwise(run: RunSpec) -> pd.DataFrame:
    path = find_stagea_metrics(run.run_dir)
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(0, "run_id", run.run_id)
    df.insert(1, "source_file", rel(path))
    return df


def stagea_summary(stagea: pd.DataFrame) -> pd.DataFrame:
    if stagea.empty:
        return pd.DataFrame()
    model_col = None
    for col in ["actual_classifier_type", "classifier_model", "model", "classifier"]:
        if col in stagea.columns:
            model_col = col
            break
    if model_col is None:
        return pd.DataFrame()
    metric_cols = [
        c
        for c in [
            "auc_raw",
            "pr_auc_raw",
            "auc_final",
            "pr_auc_final",
            "auc",
            "pr_auc",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1_score",
        ]
        if c in stagea.columns
    ]
    summary = stagea.groupby(["run_id", model_col], dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    return flatten_columns(summary).rename(columns={model_col: "stageA_model"})


def raw_stageb_pooled(run: RunSpec) -> pd.DataFrame:
    path = run.run_dir / "classifier_only_readout/classifier_sweep_pooled_metrics.csv"
    df = read_csv(path)
    if df.empty:
        return df
    out = df.copy()
    out.insert(0, "run_id", run.run_id)
    out.insert(1, "source", "classifier_only_raw")
    out["calib_method"] = "raw"
    if "readout_feature_set" in out.columns:
        out = out.rename(columns={"readout_feature_set": "feature_set"})
    return out


def oof_stageb_pooled(run: RunSpec) -> pd.DataFrame:
    if run.oof_dir is None:
        return pd.DataFrame()
    path = run.oof_dir / "calib_pooled_metrics.csv"
    df = read_csv(path)
    if df.empty:
        return df
    out = df.copy()
    out.insert(0, "run_id", run.run_id)
    out.insert(1, "source", "oof_score_calibration")
    return out


def missing_stageb_row(run: RunSpec) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": run.run_id,
                "source": "missing",
                "model_name": PRIMARY_MODEL_OOF,
                "feature_set": PRIMARY_FEATURES,
                "calib_method": PRIMARY_CALIB,
                "threshold_strategy": PRIMARY_THRESHOLD,
                "status": "missing_classifier_only_or_oof_artifacts",
                "auc": np.nan,
                "pr_auc": np.nan,
                "balanced_accuracy": np.nan,
                "sensitivity": np.nan,
                "specificity": np.nan,
                "f1": np.nan,
            }
        ]
    )


def stageb_all_pooled(run: RunSpec) -> pd.DataFrame:
    frames = [raw_stageb_pooled(run), oof_stageb_pooled(run)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return missing_stageb_row(run)
    df = pd.concat(frames, ignore_index=True)
    keep = [
        "run_id",
        "source",
        "model_name",
        "feature_set",
        "calib_method",
        "threshold_strategy",
        "n",
        "n_cn",
        "n_ad",
        "tn",
        "fp",
        "fn",
        "tp",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "f1",
        "predicted_ad_rate",
        "auc",
        "pr_auc",
        "status",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan
    return df[keep]


def focused_stageb_rows(pooled: pd.DataFrame) -> pd.DataFrame:
    if pooled.empty:
        return pooled
    rows = pooled[
        pooled["feature_set"].astype(str).eq(PRIMARY_FEATURES)
        & pooled["calib_method"].astype(str).isin(["raw", "oof_zscore", "oof_logitz", "oof_ecdf", "oof_platt", "oof_isotonic"])
        & pooled["threshold_strategy"].astype(str).isin(
            ["fixed_0p5", "inner_oof_balanced_accuracy", PRIMARY_THRESHOLD, "inner_oof_youden_j"]
        )
        & pooled["model_name"].astype(str).isin([PRIMARY_MODEL_RAW, PRIMARY_MODEL_OOF])
    ].copy()
    status = rows.get("status", pd.Series(index=rows.index, dtype=object)).astype(str)
    rows = rows[~status.eq("missing_classifier_only_or_oof_artifacts")].copy()
    missing = pooled[
        pooled.get("status", pd.Series(index=pooled.index, dtype=object)).astype(str).eq("missing_classifier_only_or_oof_artifacts")
    ]
    rows = pd.concat([rows, missing], ignore_index=True)
    calib_order = {"raw": 0, "oof_zscore": 1, "oof_logitz": 2, "oof_ecdf": 3, "oof_platt": 4, "oof_isotonic": 5}
    rows["calib_order"] = rows["calib_method"].map(calib_order).fillna(99)
    return rows.sort_values(["run_id", "calib_order", "threshold_strategy"]).drop(columns=["calib_order"])


def primary_oof_rows(pooled: pd.DataFrame) -> pd.DataFrame:
    if pooled.empty:
        return pooled
    rows = pooled[
        pooled["model_name"].astype(str).eq(PRIMARY_MODEL_OOF)
        & pooled["feature_set"].astype(str).eq(PRIMARY_FEATURES)
        & pooled["calib_method"].astype(str).isin(["oof_logitz", PRIMARY_CALIB])
        & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    status = rows.get("status", pd.Series(index=rows.index, dtype=object)).astype(str)
    rows = rows[~status.eq("missing_classifier_only_or_oof_artifacts")].copy()
    missing = pooled[
        pooled.get("status", pd.Series(index=pooled.index, dtype=object)).astype(str).eq("missing_classifier_only_or_oof_artifacts")
    ]
    rows = pd.concat([rows, missing], ignore_index=True)
    return rows.sort_values(["run_id", "calib_method"])


def oof_foldwise(run: RunSpec) -> pd.DataFrame:
    if run.oof_dir is None:
        return pd.DataFrame()
    path = run.oof_dir / "calib_foldwise_metrics.csv"
    df = read_csv(path)
    if df.empty:
        return df
    rows = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL_OOF)
        & df["feature_set"].astype(str).eq(PRIMARY_FEATURES)
        & df["calib_method"].astype(str).isin(["oof_logitz", PRIMARY_CALIB])
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    rows.insert(0, "run_id", run.run_id)
    return rows


def manufacturer_errors_from_predictions(run: RunSpec) -> pd.DataFrame:
    if run.oof_dir is None:
        return pd.DataFrame()
    path = run.oof_dir / "calib_predictions.csv"
    df = read_csv(path)
    if df.empty:
        return pd.DataFrame()
    rows = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL_OOF)
        & df["feature_set"].astype(str).eq(PRIMARY_FEATURES)
        & df["calib_method"].astype(str).eq(PRIMARY_CALIB)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    out_rows: list[dict[str, Any]] = []
    for mfr, g in rows.groupby("Manufacturer", dropna=False):
        cn = g[g["y_true"] == 0]
        ad = g[g["y_true"] == 1]
        cn_fp = int((cn["y_pred"] == 1).sum()) if not cn.empty else 0
        ad_fn = int((ad["y_pred"] == 0).sum()) if not ad.empty else 0
        out_rows.append(
            {
                "run_id": run.run_id,
                "Manufacturer": mfr,
                "n_cn": int(len(cn)),
                "cn_fp": cn_fp,
                "cn_fpr": safe_div(float(cn_fp), float(len(cn))),
                "n_ad": int(len(ad)),
                "ad_fn": ad_fn,
                "ad_fnr": safe_div(float(ad_fn), float(len(ad))),
            }
        )
    return pd.DataFrame(out_rows)


def scanner_leakage(run: RunSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        for split, file_name in [
            ("trainDev", f"fold_{fold}_scanner_leakage_summary.csv"),
            ("test", f"fold_{fold}_test_scanner_leakage_summary.csv"),
        ]:
            path = run.run_dir / f"fold_{fold}" / file_name
            df = read_csv(path)
            if df.empty:
                rows.append({"run_id": run.run_id, "fold": fold, "split": split, "available": False})
                continue
            rec = df.iloc[0].to_dict()
            rec.update({"run_id": run.run_id, "fold": fold, "split": split, "available": True})
            rec["latent_minus_raw"] = safe_float(rec.get("acc_site_latent")) - safe_float(rec.get("acc_site_raw"))
            rows.append(rec)
    return pd.DataFrame(rows)


def rate_distortion(run: RunSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        path = run.run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
        df = read_csv(path)
        if df.empty:
            rows.append({"run_id": run.run_id, "fold": fold, "status": "missing"})
            continue
        best_idx = df["L_val_betaMax"].idxmin() if "L_val_betaMax" in df.columns else df.index[-1]
        best = df.loc[best_idx]
        final = df.iloc[-1]
        d_best = safe_float(best.get("D_val"))
        r_nats_best = safe_float(best.get("R_val_nats"))
        best_epoch_val = safe_float(best.get("epoch"))
        final_epoch_val = safe_float(final.get("epoch"))
        rows.append(
            {
                "run_id": run.run_id,
                "fold": fold,
                "beta_vae": run.beta,
                "latent_dim": run.latent_dim,
                "lr_T0": run.lr_T0,
                "best_epoch": best_epoch_val,
                "final_epoch": final_epoch_val,
                "D_val_best": d_best,
                "R_val_nats_best": r_nats_best,
                "R_val_bits_best": safe_float(best.get("R_val_bits")),
                "bits_per_dim_best": safe_div(safe_float(best.get("R_val_bits")), float(run.latent_dim)),
                "KLD_over_D_best": safe_div(r_nats_best, d_best),
                "beta_KLD_over_D_best": run.beta * safe_div(r_nats_best, d_best),
                "L_val_betaMax_best": safe_float(best.get("L_val_betaMax")),
                "D_val_final": safe_float(final.get("D_val")),
                "R_val_nats_final": safe_float(final.get("R_val_nats")),
                "R_val_bits_final": safe_float(final.get("R_val_bits")),
                "bits_per_dim_final": safe_div(safe_float(final.get("R_val_bits")), float(run.latent_dim)),
                "KLD_over_D_final": safe_div(safe_float(final.get("R_val_nats")), safe_float(final.get("D_val"))),
                "beta_KLD_over_D_final": run.beta
                * safe_div(safe_float(final.get("R_val_nats")), safe_float(final.get("D_val"))),
            }
        )
    return pd.DataFrame(rows)


def _beta_phase_at_epoch(epoch: float) -> str:
    """Return 'increase' or 'max' based on cyclical beta schedule."""
    if math.isnan(epoch):
        return "unknown"
    e = int(epoch) - 1  # 0-indexed
    cycle_length = BETA_EPOCHS_TOTAL // BETA_N_CYCLES  # 80 epochs per cycle
    pos = e % cycle_length
    increase_len = int(BETA_RATIO_INCREASE * cycle_length)  # 32 epochs
    return "increase" if pos < increase_len else "max"


def _lr_phase_info(epoch: float, T0: int) -> dict[str, Any]:
    """Return LR cycle position, fractional phase (0=max LR, 1=min LR), and distance to restart."""
    if math.isnan(epoch):
        return {"lr_cycle_number": float("nan"), "lr_position_in_cycle": float("nan"),
                "lr_frac_phase": float("nan"), "lr_distance_to_restart": float("nan")}
    e = int(epoch) - 1  # 0-indexed
    cycle_number = e // T0
    position = e % T0
    frac_phase = position / T0  # 0=start of cycle (max LR), 1=end (min LR)
    distance_to_restart = T0 - position - 1
    return {
        "lr_cycle_number": cycle_number,
        "lr_position_in_cycle": position,
        "lr_frac_phase": frac_phase,
        "lr_distance_to_restart": distance_to_restart,
    }


def scheduler_analysis(runs: list[RunSpec]) -> pd.DataFrame:
    """Fold-level scheduler analysis: beta phase, LR phase, cycles completed at best and final epoch."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        for fold in range(1, 6):
            path = run.run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
            df = read_csv(path)
            if df.empty:
                rows.append({"run_id": run.run_id, "fold": fold, "status": "missing"})
                continue
            best_idx = df["L_val_betaMax"].idxmin() if "L_val_betaMax" in df.columns else df.index[-1]
            best_epoch = safe_float(df.loc[best_idx, "epoch"])
            final_epoch = safe_float(df.iloc[-1]["epoch"])

            beta_cycle_len = BETA_EPOCHS_TOTAL // BETA_N_CYCLES
            n_beta_cycles_best = (int(best_epoch) - 1) // beta_cycle_len if not math.isnan(best_epoch) else float("nan")
            n_beta_cycles_final = (int(final_epoch) - 1) // beta_cycle_len if not math.isnan(final_epoch) else float("nan")
            n_lr_cycles_best = (int(best_epoch) - 1) // run.lr_T0 if not math.isnan(best_epoch) else float("nan")
            n_lr_cycles_final = (int(final_epoch) - 1) // run.lr_T0 if not math.isnan(final_epoch) else float("nan")

            lr_info_best = _lr_phase_info(best_epoch, run.lr_T0)
            lr_info_final = _lr_phase_info(final_epoch, run.lr_T0)

            row: dict[str, Any] = {
                "run_id": run.run_id,
                "fold": fold,
                "lr_T0": run.lr_T0,
                "best_epoch": best_epoch,
                "final_epoch": final_epoch,
                "beta_phase_at_best_epoch": _beta_phase_at_epoch(best_epoch),
                "n_beta_cycles_at_best": n_beta_cycles_best,
                "n_lr_cycles_at_best": n_lr_cycles_best,
                "lr_position_at_best": lr_info_best["lr_position_in_cycle"],
                "lr_frac_phase_at_best": lr_info_best["lr_frac_phase"],
                "lr_distance_to_restart_at_best": lr_info_best["lr_distance_to_restart"],
                "beta_phase_at_final_epoch": _beta_phase_at_epoch(final_epoch),
                "n_beta_cycles_at_final": n_beta_cycles_final,
                "n_lr_cycles_at_final": n_lr_cycles_final,
                "lr_position_at_final": lr_info_final["lr_position_in_cycle"],
                "lr_frac_phase_at_final": lr_info_final["lr_frac_phase"],
                "lr_distance_to_restart_at_final": lr_info_final["lr_distance_to_restart"],
                "patience_used": final_epoch - best_epoch if not (math.isnan(final_epoch) or math.isnan(best_epoch)) else float("nan"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def latent_info(run: RunSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        for split in ["trainDev", "test"]:
            path = run.run_dir / f"fold_{fold}" / f"fold_{fold}_{split}_latent_info_summary.csv"
            df = read_csv(path)
            if df.empty:
                rows.append({"run_id": run.run_id, "fold": fold, "split": split, "status": "missing"})
                continue
            rec: dict[str, Any] = {"run_id": run.run_id, "fold": fold, "split": split}
            for variable, prefix in [("Y_target", "Y"), ("Manufacturer", "Manufacturer")]:
                row = df[df["variable"].astype(str).eq(variable)]
                if row.empty:
                    continue
                r = row.iloc[0]
                rec[f"MI_Z_{prefix}_nats"] = safe_float(r.get("mi_sum_nats"))
                rec[f"MI_Z_{prefix}_mean_nats"] = safe_float(r.get("mi_mean_nats"))
                if variable == "Y_target":
                    rec["active_units"] = safe_float(r.get("n_active"))
                    rec["frac_active"] = safe_float(r.get("frac_active"))
                    rec["total_correlation_nats"] = safe_float(r.get("total_correlation_nats"))
            rec["MI_Manufacturer_over_MI_Y"] = safe_div(
                rec.get("MI_Z_Manufacturer_nats", float("nan")),
                rec.get("MI_Z_Y_nats", float("nan")),
            )
            rows.append(rec)
    return pd.DataFrame(rows)


def get_primary_metric(primary: pd.DataFrame, run_id: str, calib: str, metric: str) -> float:
    rows = primary[(primary["run_id"] == run_id) & (primary["calib_method"] == calib)]
    if rows.empty or metric not in rows.columns:
        return float("nan")
    return safe_float(rows.iloc[0][metric])


def get_manufacturer_error(mfr_errors: pd.DataFrame, run_id: str, manufacturer: str, metric: str) -> float:
    rows = mfr_errors[
        (mfr_errors["run_id"] == run_id)
        & (mfr_errors["Manufacturer"].astype(str).str.lower() == manufacturer.lower())
    ]
    if rows.empty or metric not in rows.columns:
        return float("nan")
    return safe_float(rows.iloc[0][metric])


def get_leakage_mean(scanner_summary: pd.DataFrame, run_id: str, split: str = "test") -> float:
    if scanner_summary.empty:
        return float("nan")
    rows = scanner_summary[(scanner_summary["run_id"] == run_id) & (scanner_summary["split"] == split)]
    if rows.empty:
        return float("nan")
    for col in ["acc_site_latent_mean", "acc_site_latent"]:
        if col in rows.columns:
            return safe_float(rows.iloc[0][col])
    return float("nan")


def promotion_gate(
    primary: pd.DataFrame,
    mfr_errors: pd.DataFrame,
    scanner_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    cand = "candidate_latent384_beta3p75_T160"
    ref = "promoted_latent384_beta3p75_T80"
    rows: list[dict[str, Any]] = []

    stageb_present = not primary[
        (primary["run_id"] == cand)
        & primary.get("status", pd.Series(index=primary.index, dtype=object))
        .astype(str)
        .eq("missing_classifier_only_or_oof_artifacts")
    ].empty
    rows.append(
        {
            "gate": "stageB_oof_available",
            "candidate_value": not stageb_present,
            "promoted_reference_value": True,
            "required_limit": True,
            "delta_vs_promoted": np.nan,
            "passes": bool(not stageb_present),
            "details": "Stage B classifier-only and OOF calibration artifacts must exist before promotion can be evaluated.",
        }
    )
    for metric, limit, op, details in [
        ("auc", PROMOTED_AUC, ">", f"AUC must exceed promoted {PROMOTED_AUC} and preferably ch1-only {CH1_AUC}."),
        ("pr_auc", PROMOTED_PR_AUC, ">=", f"PR-AUC must meet promoted {PROMOTED_PR_AUC} and preferably ch1-only {CH1_PR_AUC}."),
    ]:
        value = get_primary_metric(primary, cand, PRIMARY_CALIB, metric)
        ref_value = get_primary_metric(primary, ref, PRIMARY_CALIB, metric)
        passes = (value > limit) if op == ">" else (value >= limit)
        rows.append(
            {
                "gate": metric,
                "candidate_value": value,
                "promoted_reference_value": ref_value,
                "required_limit": limit,
                "delta_vs_promoted": value - ref_value,
                "passes": bool(passes),
                "details": details,
            }
        )
    for metric, tolerance in [("balanced_accuracy", 0.01), ("f1", 0.01), ("sensitivity", 0.01)]:
        value = get_primary_metric(primary, cand, PRIMARY_CALIB, metric)
        ref_value = get_primary_metric(primary, ref, PRIMARY_CALIB, metric)
        rows.append(
            {
                "gate": metric,
                "candidate_value": value,
                "promoted_reference_value": ref_value,
                "required_limit": ref_value - tolerance,
                "delta_vs_promoted": value - ref_value,
                "passes": bool(value >= ref_value - tolerance),
                "details": f"Must not be materially worse than promoted; tolerance {-tolerance:+.3f}.",
            }
        )
    cand_fpr = get_manufacturer_error(mfr_errors, cand, "Philips", "cn_fpr")
    ref_fpr = get_manufacturer_error(mfr_errors, ref, "Philips", "cn_fpr")
    rows.append(
        {
            "gate": "Philips_CN_FPR",
            "candidate_value": cand_fpr,
            "promoted_reference_value": ref_fpr,
            "required_limit": PHILIPS_FPR_GATE,
            "delta_vs_promoted": cand_fpr - ref_fpr,
            "passes": bool(cand_fpr <= PHILIPS_FPR_GATE),
            "details": f"Philips CN FPR must be <= {PHILIPS_FPR_GATE}.",
        }
    )
    cand_leak = get_leakage_mean(scanner_summary, cand, "test")
    ref_leak = get_leakage_mean(scanner_summary, ref, "test")
    rows.append(
        {
            "gate": "scanner_leakage_test_latent_acc",
            "candidate_value": cand_leak,
            "promoted_reference_value": ref_leak,
            "required_limit": ref_leak,
            "delta_vs_promoted": cand_leak - ref_leak,
            "passes": bool(cand_leak <= ref_leak) if not (math.isnan(cand_leak) or math.isnan(ref_leak)) else False,
            "details": "Latent scanner/manufacturer leakage must not worsen vs promoted.",
        }
    )
    gate = pd.DataFrame(rows)
    decision = "promote" if bool(gate["passes"].all()) else "reject_incomplete_or_gate_failed"
    return gate, decision


def scheduler_analysis_text(sched_summary: pd.DataFrame) -> str:
    """Narrative summary of T160 vs T80 scheduler behavior."""
    lines = ["# Scheduler Analysis: T160 vs T80\n"]
    lines.append(
        "Question: does T0=160 (longer cosine warm-restart cycles) change when early stopping"
        " triggers relative to the LR and beta schedules?\n"
    )
    lines.append(f"- Cyclical beta cycle length: {BETA_EPOCHS_TOTAL // BETA_N_CYCLES} epochs "
                 f"({BETA_N_CYCLES} cycles over {BETA_EPOCHS_TOTAL} epochs)\n")
    lines.append(f"- T80: LR cycle = 80 epochs (synchronized with beta cycle)\n")
    lines.append(f"- T160: LR cycle = 160 epochs = 2× beta cycles (desynchronized)\n\n")

    for run_id in ["candidate_latent384_beta3p75_T160", "promoted_latent384_beta3p75_T80"]:
        sub = sched_summary[sched_summary["run_id"] == run_id]
        if sub.empty:
            lines.append(f"- {run_id}: no data\n")
            continue
        best_mean = sub["best_epoch"].mean()
        final_mean = sub["final_epoch"].mean()
        patience_mean = sub["patience_used"].mean()
        lr_cycles_best_mean = sub["n_lr_cycles_at_best"].mean()
        beta_cycles_best_mean = sub["n_beta_cycles_at_best"].mean()
        lr_frac_best_mean = sub["lr_frac_phase_at_best"].mean()
        dist_restart_best_mean = sub["lr_distance_to_restart_at_best"].mean()
        beta_phases_at_best = sub["beta_phase_at_best_epoch"].value_counts().to_dict()
        lines.append(f"## {run_id}\n")
        lines.append(f"- Mean best epoch: {best_mean:.1f}\n")
        lines.append(f"- Mean final epoch: {final_mean:.1f}\n")
        lines.append(f"- Mean patience used after best: {patience_mean:.1f} epochs\n")
        lines.append(f"- Mean LR cycles completed at best epoch: {lr_cycles_best_mean:.1f}\n")
        lines.append(f"- Mean beta cycles completed at best epoch: {beta_cycles_best_mean:.1f}\n")
        lines.append(f"- Mean LR fractional phase at best epoch: {lr_frac_best_mean:.3f} (0=max LR, 1=min LR)\n")
        lines.append(f"- Mean distance to next LR restart at best epoch: {dist_restart_best_mean:.1f} epochs\n")
        lines.append(f"- Beta phase at best epoch (count across folds): {beta_phases_at_best}\n\n")
    return "".join(lines)


def effective_regularization_text(rd_summary: pd.DataFrame, primary: pd.DataFrame, mfr_errors: pd.DataFrame) -> str:
    def mean_for(run_id: str, col: str) -> float:
        row = rd_summary[rd_summary["run_id"] == run_id]
        if row.empty:
            return float("nan")
        mean_col = f"{col}_mean"
        if mean_col in row.columns:
            return safe_float(row.iloc[0][mean_col])
        return safe_float(row.iloc[0].get(col))

    cand_reg = mean_for("candidate_latent384_beta3p75_T160", "beta_KLD_over_D_best")
    prom_reg = mean_for("promoted_latent384_beta3p75_T80", "beta_KLD_over_D_best")
    ch1_reg = mean_for("ch1only_latent384_beta3p75_T80", "beta_KLD_over_D_best")
    beta6_reg = mean_for("latent384_beta6p5_T80", "beta_KLD_over_D_best")
    chmean_reg = mean_for("chmeanloss_latent384_beta3p75_T80", "beta_KLD_over_D_best")
    beta9p5_reg = mean_for("candidate_latent384_beta9p5_T80", "beta_KLD_over_D_best")
    chweighted_reg = mean_for("chweighted_latent384_beta3p75_T80", "beta_KLD_over_D_best")

    if not math.isnan(cand_reg) and not math.isnan(ch1_reg) and not math.isnan(prom_reg):
        fraction = safe_div(cand_reg - prom_reg, ch1_reg - prom_reg)
        regime = (
            f"T160 beta*KLD/D={cand_reg:.6f}; promoted T80={prom_reg:.6f}; "
            f"ch1-only T80={ch1_reg:.6f}. T160 closes {fraction:.1%} of the promoted-to-ch1 gap."
        )
        reached = bool(cand_reg >= 0.9 * ch1_reg)
    else:
        regime = "Rate-distortion files were insufficient to quantify the T160 effective regularization regime."
        reached = False

    cand_auc = get_primary_metric(primary, "candidate_latent384_beta3p75_T160", PRIMARY_CALIB, "auc")
    cand_pr = get_primary_metric(primary, "candidate_latent384_beta3p75_T160", PRIMARY_CALIB, "pr_auc")
    cand_fpr = get_manufacturer_error(mfr_errors, "candidate_latent384_beta3p75_T160", "Philips", "cn_fpr")

    return f"""# Effective-Regularization Hypothesis Test

Question: did T0=160 change the effective regularization regime (beta*KLD/D) vs T80?

{regime}

Reference comparators:
- beta6.5 T80 beta*KLD/D={beta6_reg:.6f}
- chmeanloss T80 beta*KLD/D={chmean_reg:.6f}
- chweighted T80 beta*KLD/D={chweighted_reg:.6f}
- beta9.5 T80 beta*KLD/D={beta9p5_reg:.6f}

Stage B availability:
- T160 OOF-ECDF AUC={cand_auc if not math.isnan(cand_auc) else 'missing'}
- T160 OOF-ECDF PR-AUC={cand_pr if not math.isnan(cand_pr) else 'missing'}
- T160 Philips CN FPR={cand_fpr if not math.isnan(cand_fpr) else 'missing'}

Reached ch1-only regime by beta*KLD/D: **{reached}**.

Interpretation:
T0=160 uses longer LR cosine-restart cycles (2 beta cycles per LR cycle vs 1:1 for T80).
This decouples the LR and beta schedules. Whether this changes the optimal operating point
or the effective regularization depends on where early stopping triggers in the LR/beta cycle.
"""


def final_decision_text(gate: pd.DataFrame, decision: str, primary: pd.DataFrame) -> str:
    cand_id = "candidate_latent384_beta3p75_T160"
    cand_rows = primary[primary["run_id"].eq(cand_id)]
    if cand_rows.empty:
        row_text = "- No T160 primary Stage B rows were available."
    else:
        lines = []
        for _, row in cand_rows.iterrows():
            if pd.isna(row.get("auc")):
                lines.append(f"- {row.get('calib_method', 'unknown')}: missing Stage B/OOF artifacts.")
            else:
                lines.append(
                    f"- {row['calib_method']}: AUC={row['auc']:.6f}, PR-AUC={row['pr_auc']:.6f}, "
                    f"BA={row['balanced_accuracy']:.6f}, Sens={row['sensitivity']:.6f}, "
                    f"Spec={row['specificity']:.6f}, F1={row['f1']:.6f}."
                )
        row_text = "\n".join(lines)
    failed = gate.loc[~gate["passes"].astype(bool), "gate"].tolist()
    oasis_line = (
        "ADNI promotion gate passed; frozen OASIS stress-test may be run next."
        if decision == "promote"
        else "ADNI promotion gate failed or is incomplete; OASIS stress-test was not run."
    )
    return f"""# Final Decision

Decision: **{decision}**.

Candidate primary score-harmonized Stage B rows:
{row_text}

Promotion-gate failures:
- {", ".join(failed) if failed else "none"}

Gate definition:
- Stage B classifier-only and OOF calibration artifacts must exist.
- AUC must exceed promoted {PROMOTED_AUC} and preferably ch1-only {CH1_AUC}.
- PR-AUC must meet promoted {PROMOTED_PR_AUC} and preferably ch1-only {CH1_PR_AUC}.
- BA/F1/Sensitivity must not be materially worse than promoted.
- Philips CN FPR must be <= {PHILIPS_FPR_GATE}.
- Scanner leakage must not worsen vs promoted.

OASIS:
{oasis_line}

Guardrails:
This package is read-only. It performs no VAE retraining, no OASIS threshold or calibration fitting,
no tensor modification, no metadata modification, and no model artifact overwrite.
"""


def stagea_to_stageb_optimism(stagea_summary_df: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if stagea_summary_df.empty or primary.empty:
        return pd.DataFrame()
    stagea_focus = stagea_summary_df[stagea_summary_df["stageA_model"].astype(str).isin(["logreg", "svm"])].copy()
    primary_ecdf = primary[primary["calib_method"].astype(str).eq(PRIMARY_CALIB)].copy()
    for _, st in stagea_focus.iterrows():
        run_id = st["run_id"]
        pr = primary_ecdf[primary_ecdf["run_id"].eq(run_id)]
        if pr.empty:
            rows.append(
                {
                    "run_id": run_id,
                    "stageA_model": st["stageA_model"],
                    "stageA_auc_final_mean": safe_float(st.get("auc_final_mean", st.get("auc_mean"))),
                    "stageA_pr_auc_final_mean": safe_float(st.get("pr_auc_final_mean", st.get("pr_auc_mean"))),
                    "stageB_primary_auc": np.nan,
                    "stageB_primary_pr_auc": np.nan,
                    "stageB_minus_stageA_auc": np.nan,
                    "stageB_minus_stageA_pr_auc": np.nan,
                    "note": "Stage B primary row missing.",
                }
            )
            continue
        pr_row = pr.iloc[0]
        rows.append(
            {
                "run_id": run_id,
                "stageA_model": st["stageA_model"],
                "stageA_auc_final_mean": safe_float(st.get("auc_final_mean", st.get("auc_mean"))),
                "stageA_pr_auc_final_mean": safe_float(st.get("pr_auc_final_mean", st.get("pr_auc_mean"))),
                "stageA_ba_mean": safe_float(st.get("balanced_accuracy_mean")),
                "stageA_f1_mean": safe_float(st.get("f1_score_mean")),
                "stageB_primary_auc": safe_float(pr_row.get("auc")),
                "stageB_primary_pr_auc": safe_float(pr_row.get("pr_auc")),
                "stageB_primary_ba": safe_float(pr_row.get("balanced_accuracy")),
                "stageB_primary_f1": safe_float(pr_row.get("f1")),
                "stageB_minus_stageA_auc": safe_float(pr_row.get("auc"))
                - safe_float(st.get("auc_final_mean", st.get("auc_mean"))),
                "stageB_minus_stageA_pr_auc": safe_float(pr_row.get("pr_auc"))
                - safe_float(st.get("pr_auc_final_mean", st.get("pr_auc_mean"))),
                "stageB_minus_stageA_ba": safe_float(pr_row.get("balanced_accuracy"))
                - safe_float(st.get("balanced_accuracy_mean")),
                "stageB_minus_stageA_f1": safe_float(pr_row.get("f1")) - safe_float(st.get("f1_score_mean")),
                "note": "Positive delta means Stage B primary OOF-ECDF exceeds Stage A mean fold metric.",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out = args.output_dir
    candidate = RUNS[0]
    if not candidate.run_dir.exists():
        raise FileNotFoundError(candidate.run_dir)

    planned = [
        "README.md",
        "completion_status.csv/.md",
        "stagea_summary_metrics.csv/.md",
        "stagea_foldwise_metrics.csv/.md",
        "stageb_oof_calibration_metrics.csv/.md",
        "stageb_primary_rows.csv/.md",
        "stageb_primary_foldwise_metrics.csv/.md",
        "stagea_to_stageb_optimism.csv/.md",
        "primary_promotion_gate_table.csv/.md",
        "manufacturer_cn_fpr_ad_fnr.csv/.md",
        "scanner_leakage_summary.csv/.md",
        "rate_distortion_summary.csv/.md",
        "latent_mi_signal_nuisance_summary.csv/.md",
        "scheduler_analysis_foldwise.csv/.md",
        "scheduler_analysis_summary.csv/.md",
        "scheduler_analysis_narrative.md",
        "effective_regularization_hypothesis_test.md",
        "final_decision.md",
        "command_log.json",
    ]
    if args.dry_run:
        print("Candidate run directory OK.")
        print("Output:", rel(out))
        for item in planned:
            print("-", item)
        return

    out.mkdir(parents=True, exist_ok=True)
    available_runs = [run for run in RUNS if run.run_dir.exists()]

    status = pd.concat([completion_status(run) for run in available_runs], ignore_index=True)
    write_table(status, "completion_status", out)

    stagea_all = pd.concat([stagea_foldwise(run) for run in available_runs], ignore_index=True)
    write_table(stagea_all, "stagea_foldwise_metrics", out)
    stagea_summary_df = stagea_summary(stagea_all)
    write_table(stagea_summary_df, "stagea_summary_metrics", out)

    pooled = pd.concat([stageb_all_pooled(run) for run in available_runs], ignore_index=True)
    focused = focused_stageb_rows(pooled)
    write_table(focused, "stageb_oof_calibration_metrics", out)
    primary = primary_oof_rows(pooled)
    write_table(primary, "stageb_primary_rows", out)
    foldwise = pd.concat([oof_foldwise(run) for run in available_runs], ignore_index=True)
    write_table(foldwise, "stageb_primary_foldwise_metrics", out)
    write_table(stagea_to_stageb_optimism(stagea_summary_df, primary), "stagea_to_stageb_optimism", out)

    mfr_errors = pd.concat([manufacturer_errors_from_predictions(run) for run in available_runs], ignore_index=True)
    write_table(mfr_errors, "manufacturer_cn_fpr_ad_fnr", out)

    scanner = pd.concat([scanner_leakage(run) for run in available_runs], ignore_index=True)
    scanner_summary = summarize(scanner, ["run_id", "split"], ["acc_site_raw", "acc_site_latent", "latent_minus_raw"])
    write_table(scanner, "scanner_leakage_foldwise", out)
    write_table(scanner_summary, "scanner_leakage_summary", out)

    rd = pd.concat([rate_distortion(run) for run in available_runs], ignore_index=True)
    rd_metrics = [
        "best_epoch",
        "final_epoch",
        "D_val_best",
        "R_val_bits_best",
        "bits_per_dim_best",
        "KLD_over_D_best",
        "beta_KLD_over_D_best",
        "D_val_final",
        "R_val_bits_final",
        "bits_per_dim_final",
        "KLD_over_D_final",
        "beta_KLD_over_D_final",
    ]
    write_table(rd, "rate_distortion_foldwise", out)
    rd_summary = summarize(rd, ["run_id"], rd_metrics)
    write_table(rd_summary, "rate_distortion_summary", out)

    latent = pd.concat([latent_info(run) for run in available_runs], ignore_index=True)
    latent_metrics = [
        "MI_Z_Y_nats",
        "MI_Z_Manufacturer_nats",
        "MI_Manufacturer_over_MI_Y",
        "active_units",
        "frac_active",
        "total_correlation_nats",
    ]
    write_table(latent, "latent_mi_signal_nuisance_foldwise", out)
    latent_summary = summarize(latent, ["run_id", "split"], latent_metrics)
    write_table(latent_summary, "latent_mi_signal_nuisance_summary", out)

    # Scheduler analysis (T160 vs T80 comparison)
    sched_runs = [run for run in available_runs if run.run_id in [
        "candidate_latent384_beta3p75_T160", "promoted_latent384_beta3p75_T80"
    ]]
    sched_df = scheduler_analysis(sched_runs)
    write_table(sched_df, "scheduler_analysis_foldwise", out)
    sched_metrics = [
        "best_epoch", "final_epoch", "patience_used",
        "n_beta_cycles_at_best", "n_lr_cycles_at_best",
        "lr_position_at_best", "lr_frac_phase_at_best", "lr_distance_to_restart_at_best",
        "n_lr_cycles_at_final", "lr_distance_to_restart_at_final",
    ]
    sched_summary = summarize(sched_df, ["run_id", "lr_T0"], sched_metrics)
    write_table(sched_summary, "scheduler_analysis_summary", out)
    (out / "scheduler_analysis_narrative.md").write_text(
        scheduler_analysis_text(sched_df), encoding="utf-8"
    )

    gate, decision = promotion_gate(primary, mfr_errors, scanner_summary)
    write_table(gate, "primary_promotion_gate_table", out)
    (out / "effective_regularization_hypothesis_test.md").write_text(
        effective_regularization_text(rd_summary, primary, mfr_errors),
        encoding="utf-8",
    )
    (out / "final_decision.md").write_text(final_decision_text(gate, decision, primary), encoding="utf-8")

    cand_global = status[
        (status["run_id"].eq("candidate_latent384_beta3p75_T160")) & (status["fold"].astype(str).eq("all"))
    ]
    stageb_complete = bool(cand_global["stageB_complete"].iloc[0]) if not cand_global.empty else False
    vae_complete = bool(
        status[
            (status["run_id"].eq("candidate_latent384_beta3p75_T160"))
            & (~status["fold"].astype(str).eq("all"))
        ]["vae_stageA_complete"].all()
    )
    readme = f"""# T160 Completion and Promotion-Gate Audit

Candidate: `recover035_latent384_beta3p75_T160_h10000_p560_full5x5`.

Scientific question: does T0=160 (longer cosine warm-restart cycles) improve latent geometry
or ADNI OOF performance vs promoted T80, keeping all other settings fixed?

This audit compares T160 against:
- promoted [1,0,2] beta3.75 T80 (primary reference)
- ch1-only beta3.75 T80
- beta9.5 T80
- chweightedPearson50 T80
- beta6.5 T80
- chmeanloss T80

Candidate status:
- VAE/Stage A complete: {vae_complete}
- Stage B classifier-only + OOF calibration complete: {stageb_complete}

Decision: {decision}

OASIS: {"Run if gate passes." if decision == "promote" else "Not run — gate failed or incomplete."}

Guardrails: no VAE retraining, no classifier-only readout generation, no OASIS threshold/calibration
fitting, no tensor modification, no metadata modification, and no model artifact overwrite.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    command_log = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": rel(Path(__file__)),
        "output_dir": rel(out),
        "candidate": "recover035_latent384_beta3p75_T160_h10000_p560_full5x5",
        "scientific_question": "Does T0=160 improve latent geometry or ADNI OOF performance vs T80?",
        "read_only_guardrails": {
            "no_vae_retraining": True,
            "no_classifier_only_generation": True,
            "no_oasis_threshold_or_calibration_fitting": True,
            "no_tensor_modification": True,
            "no_metadata_modification": True,
            "no_model_artifact_overwrite": True,
        },
        "available_runs_used": [
            {
                "run_id": run.run_id,
                "run_dir": rel(run.run_dir),
                "oof_dir": rel(run.oof_dir),
                "beta": run.beta,
                "latent_dim": run.latent_dim,
                "channel_set": run.channel_set,
                "role": run.role,
                "lr_T0": run.lr_T0,
            }
            for run in available_runs
        ],
        "primary_gate": {
            "model_name": PRIMARY_MODEL_OOF,
            "feature_set": PRIMARY_FEATURES,
            "calib_method": PRIMARY_CALIB,
            "threshold_strategy": PRIMARY_THRESHOLD,
        },
        "decision": decision,
        "oasis_stress_test_run": decision == "promote",
        "oasis_stress_test_reason": (
            "ADNI promotion gate passed" if decision == "promote"
            else "ADNI promotion gate failed or was incomplete"
        ),
        "generated_outputs": planned,
    }
    (out / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(f"Wrote audit package: {rel(out)}")
    print(f"Decision: {decision}")
    print(f"Stage B complete: {stageb_complete}")


if __name__ == "__main__":
    main()
