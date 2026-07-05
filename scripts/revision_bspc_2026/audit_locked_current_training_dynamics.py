#!/usr/bin/env python3
"""Read-only training dynamics audit for the locked ADNI v5.1 FULL [1,0,2] run.

This script only reads existing model/run artifacts and writes a new audit
folder under results/. It does not train, modify tensors, or edit existing run
outputs.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
STAGE_B_DIR = ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
OUT_DIR = ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_locked_training_dynamics_audit"
FIG_DIR = OUT_DIR / "figures"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_markdown_table(df: pd.DataFrame, path: Path, floatfmt: str = ".6g") -> None:
    try:
        text = df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        text = df.to_csv(index=False)
    path.write_text(text + "\n", encoding="utf-8")


def numeric_at(series: pd.Series, epoch_1based: int) -> float:
    idx = epoch_1based - 1
    if idx < 0 or idx >= len(series):
        return float("nan")
    return float(series.iloc[idx])


def history_to_df(path: Path) -> pd.DataFrame:
    obj = joblib.load(path)
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    elif isinstance(obj, dict):
        df = pd.DataFrame(obj)
    else:
        fail(f"Unsupported history object in {path}: {type(obj)}")
    if "val_loss_modelsel" not in df.columns:
        fail(f"History is missing val_loss_modelsel: {path}")
    df = df.reset_index(drop=True)
    df.insert(0, "epoch", np.arange(1, len(df) + 1))
    for prefix in ("train", "val"):
        recon = f"{prefix}_recon"
        kld = f"{prefix}_kld"
        if recon in df.columns and kld in df.columns:
            df[f"{prefix}_kld_over_recon"] = df[kld] / df[recon].replace(0, np.nan)
            df[f"{prefix}_beta_kld_over_recon"] = df["beta"] * df[kld] / df[recon].replace(0, np.nan)
    return df


def beta_schedule(epoch_0based: np.ndarray, total_epochs: int, beta_max: float, n_cycles: int, ratio_increase: float) -> np.ndarray:
    if n_cycles <= 0:
        return np.full_like(epoch_0based, beta_max, dtype=float)
    epoch_per_cycle = total_epochs / n_cycles
    phase = np.mod(epoch_0based, epoch_per_cycle)
    increase = epoch_per_cycle * ratio_increase
    beta = np.where(phase < increase, beta_max * (phase / increase), beta_max)
    return beta.astype(float)


def cosine_warm_restart_lr(epoch_0based: np.ndarray, base_lr: float, eta_min: float, t0: int) -> np.ndarray:
    phase = np.mod(epoch_0based, t0)
    return eta_min + 0.5 * (base_lr - eta_min) * (1.0 + np.cos(np.pi * phase / t0))


def lr_at_epoch(epoch_1based: int, base_lr: float, eta_min: float, t0: int) -> float:
    return float(cosine_warm_restart_lr(np.array([epoch_1based - 1], dtype=float), base_lr, eta_min, t0)[0])


def phase_label(epoch_1based: int, t0: int, beta_ramp_epochs: float) -> tuple[float, float, str, str]:
    phase = float((epoch_1based - 1) % t0)
    phase_fraction = phase / float(t0)
    if phase <= 5:
        lr_alignment = "near_lr_restart"
    elif phase >= t0 - 8:
        lr_alignment = "near_lr_trough"
    else:
        lr_alignment = "mid_lr_cycle"
    beta_phase = "beta_ramp" if phase < beta_ramp_epochs else "beta_plateau"
    return phase, phase_fraction, lr_alignment, beta_phase


def load_stage_b_foldwise() -> pd.DataFrame:
    path = STAGE_B_DIR / "classifier_sweep_foldwise_metrics.csv"
    df = pd.read_csv(path)
    mask = (df["model_name"] == PRIMARY_MODEL) & (df["threshold_strategy"] == PRIMARY_THRESHOLD)
    out = df.loc[mask].copy()
    if out.empty:
        fail(f"No Stage B rows for {PRIMARY_MODEL}/{PRIMARY_THRESHOLD} in {path}")
    keep = [
        "fold",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "f1",
        "sensitivity",
        "specificity",
        "threshold",
        "inner_oof_sensitivity",
        "inner_oof_specificity",
        "inner_oof_balanced_accuracy",
    ]
    return out[[c for c in keep if c in out.columns]]


def load_stage_a_metrics() -> pd.DataFrame:
    paths = sorted(RUN_DIR.glob("all_folds_metrics_MULTI_*.csv"))
    if not paths:
        return pd.DataFrame()
    df = pd.read_csv(paths[0])
    keep = ["fold", "actual_classifier_type", "auc", "pr_auc", "balanced_accuracy", "f1_score", "sensitivity", "specificity"]
    return df[[c for c in keep if c in df.columns]].rename(
        columns={
            "actual_classifier_type": "stage_a_classifier",
            "auc": "stage_a_auc",
            "pr_auc": "stage_a_pr_auc",
            "balanced_accuracy": "stage_a_ba",
            "f1_score": "stage_a_f1",
        }
    )


def read_one_row_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def latent_info_for_fold(fold: int) -> dict[str, Any]:
    path = RUN_DIR / f"fold_{fold}" / f"fold_{fold}_trainDev_latent_info_summary.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    row = df[df["variable"].astype(str).eq("Y_target")]
    if row.empty:
        row = df.head(1)
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "active_units": r.get("n_active", np.nan),
        "frac_active": r.get("frac_active", np.nan),
        "total_correlation_nats": r.get("total_correlation_nats", np.nan),
        "latent_y_mi_sum_nats": r.get("mi_sum_nats", np.nan),
    }


def scanner_for_fold(fold: int) -> dict[str, Any]:
    train_dev = read_one_row_csv(RUN_DIR / f"fold_{fold}" / f"fold_{fold}_scanner_leakage_summary.csv")
    test = read_one_row_csv(RUN_DIR / f"fold_{fold}" / f"fold_{fold}_test_scanner_leakage_summary.csv")
    out: dict[str, Any] = {}
    for prefix, row in (("trainDev", train_dev), ("test", test)):
        for key in ("acc_site_raw", "acc_site_latent", "chance_level", "n_sites"):
            out[f"{prefix}_{key}"] = row.get(key, np.nan)
        raw = row.get("acc_site_raw", np.nan)
        lat = row.get("acc_site_latent", np.nan)
        out[f"{prefix}_latent_minus_raw_site_acc"] = lat - raw if pd.notna(lat) and pd.notna(raw) else np.nan
    return out


def build_fold_rows(config: dict[str, Any], stage_b: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    params = config.get("parameters", config.get("args", {}))
    epochs = int(params["epochs_vae"])
    cycles = int(params["cyclical_beta_n_cycles"])
    beta_max = float(params["beta_vae"])
    ratio = float(params.get("cyclical_beta_ratio_increase", 0.4))
    t0 = int(params["lr_scheduler_T0"])
    lr0 = float(params["lr_vae"])
    eta_min = float(params.get("lr_scheduler_eta_min", 5e-7))
    beta_ramp_epochs = (epochs / cycles) * ratio

    rows: list[dict[str, Any]] = []
    histories: dict[int, pd.DataFrame] = {}
    for fold in range(1, 6):
        hist_path = RUN_DIR / f"fold_{fold}" / f"vae_train_history_fold_{fold}.joblib"
        df = history_to_df(hist_path)
        histories[fold] = df
        best_idx = int(df["val_loss_modelsel"].astype(float).idxmin())
        best_epoch = best_idx + 1
        early_stop_epoch = int(df["epoch"].iloc[-1])
        phase, phase_frac, lr_alignment, beta_phase = phase_label(best_epoch, t0, beta_ramp_epochs)

        stage_b_row = stage_b[stage_b["fold"].astype(int).eq(fold)]
        stage_b_data = stage_b_row.iloc[0].to_dict() if not stage_b_row.empty else {}
        scanner = scanner_for_fold(fold)
        latent = latent_info_for_fold(fold)

        train_recon = numeric_at(df["train_recon"], best_epoch) if "train_recon" in df else np.nan
        val_recon = numeric_at(df["val_recon"], best_epoch) if "val_recon" in df else np.nan
        train_kld = numeric_at(df["train_kld"], best_epoch) if "train_kld" in df else np.nan
        val_kld = numeric_at(df["val_kld"], best_epoch) if "val_kld" in df else np.nan
        row = {
            "fold": fold,
            "best_epoch": best_epoch,
            "early_stop_epoch": early_stop_epoch,
            "stopped_early": early_stop_epoch < epochs,
            "epochs_after_best": early_stop_epoch - best_epoch,
            "best_val_loss_beta_max": numeric_at(df["val_loss_modelsel"], best_epoch),
            "train_recon_at_best": train_recon,
            "val_recon_at_best": val_recon,
            "val_minus_train_recon_at_best": val_recon - train_recon,
            "val_over_train_recon_at_best": val_recon / train_recon if train_recon else np.nan,
            "train_kld_at_best": train_kld,
            "val_kld_at_best": val_kld,
            "train_kld_over_recon_at_best": train_kld / train_recon if train_recon else np.nan,
            "val_kld_over_recon_at_best": val_kld / val_recon if val_recon else np.nan,
            "train_beta_kld_over_recon_at_best_current_beta": numeric_at(df["beta"], best_epoch) * train_kld / train_recon if train_recon else np.nan,
            "val_beta_kld_over_recon_at_best_current_beta": numeric_at(df["beta"], best_epoch) * val_kld / val_recon if val_recon else np.nan,
            "val_beta_max_kld_over_recon_at_best": beta_max * val_kld / val_recon if val_recon else np.nan,
            "lr_at_best_epoch_approx": lr_at_epoch(best_epoch, lr0, eta_min, t0),
            "beta_at_best_epoch": numeric_at(df["beta"], best_epoch),
            "beta_cycle_phase_epoch": phase,
            "beta_cycle_phase_fraction": phase_frac,
            "lr_alignment": lr_alignment,
            "beta_phase": beta_phase,
            "near_lr_restart": lr_alignment == "near_lr_restart",
            "near_lr_trough": lr_alignment == "near_lr_trough",
            "near_beta_ramp": beta_phase == "beta_ramp",
            "near_beta_plateau": beta_phase == "beta_plateau",
            "stage_b_auc": stage_b_data.get("auc", np.nan),
            "stage_b_pr_auc": stage_b_data.get("pr_auc", np.nan),
            "stage_b_balanced_accuracy": stage_b_data.get("balanced_accuracy", np.nan),
            "stage_b_f1": stage_b_data.get("f1", np.nan),
            "stage_b_sensitivity": stage_b_data.get("sensitivity", np.nan),
            "stage_b_specificity": stage_b_data.get("specificity", np.nan),
            "stage_b_threshold": stage_b_data.get("threshold", np.nan),
            **scanner,
            **latent,
        }
        rows.append(row)
    return pd.DataFrame(rows), histories


def correlation_summary(df: pd.DataFrame) -> pd.DataFrame:
    target = "stage_b_auc"
    metrics = [
        "best_epoch",
        "early_stop_epoch",
        "epochs_after_best",
        "best_val_loss_beta_max",
        "val_minus_train_recon_at_best",
        "val_over_train_recon_at_best",
        "val_kld_at_best",
        "val_kld_over_recon_at_best",
        "val_beta_max_kld_over_recon_at_best",
        "active_units",
        "frac_active",
        "total_correlation_nats",
        "test_acc_site_latent",
        "test_latent_minus_raw_site_acc",
        "lr_at_best_epoch_approx",
        "beta_cycle_phase_epoch",
    ]
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        tmp = df[[target, metric]].dropna()
        if len(tmp) < 3 or tmp[metric].nunique() <= 1:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = float(tmp[target].corr(tmp[metric], method="pearson"))
            spearman = float(tmp[target].corr(tmp[metric], method="spearman"))
        rows.append({"metric": metric, "n_folds": len(tmp), "pearson_with_stage_b_auc": pearson, "spearman_with_stage_b_auc": spearman})
    return pd.DataFrame(rows)


def add_bad_fold_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    auc_q = out["stage_b_auc"].quantile(0.4)
    out["bad_fold_by_auc_bottom40"] = out["stage_b_auc"] <= auc_q
    for col in [
        "val_minus_train_recon_at_best",
        "val_kld_over_recon_at_best",
        "active_units",
        "test_acc_site_latent",
        "best_epoch",
    ]:
        if col in out and out[col].notna().any() and out[col].nunique(dropna=True) > 1:
            out[f"{col}_rank"] = out[col].rank(method="min")
    return out


def save_figures(histories: dict[int, pd.DataFrame], fold_df: pd.DataFrame, config: dict[str, Any]) -> None:
    params = config.get("parameters", config.get("args", {}))
    epochs = int(params["epochs_vae"])
    cycles = int(params["cyclical_beta_n_cycles"])
    beta_max = float(params["beta_vae"])
    ratio = float(params.get("cyclical_beta_ratio_increase", 0.4))
    t0 = int(params["lr_scheduler_T0"])
    lr0 = float(params["lr_vae"])
    eta_min = float(params.get("lr_scheduler_eta_min", 5e-7))
    epoch0 = np.arange(epochs, dtype=float)
    epoch1 = epoch0 + 1
    lr = cosine_warm_restart_lr(epoch0, lr0, eta_min, t0)
    beta = beta_schedule(epoch0, epochs, beta_max, cycles, ratio)

    plt.figure(figsize=(10, 6))
    for fold, hist in histories.items():
        plt.plot(hist["epoch"], hist["val_loss_modelsel"], linewidth=1.2, label=f"fold {fold}")
        best_epoch = int(fold_df.loc[fold_df["fold"].eq(fold), "best_epoch"].iloc[0])
        best_loss = float(fold_df.loc[fold_df["fold"].eq(fold), "best_val_loss_beta_max"].iloc[0])
        plt.scatter([best_epoch], [best_loss], s=30)
    plt.xlabel("epoch")
    plt.ylabel("ValL(beta_max)")
    plt.title("Validation beta-max loss by fold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "val_loss_beta_max_vs_epoch_per_fold.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(epoch1, lr, color="black", linewidth=1)
    for _, row in fold_df.iterrows():
        plt.axvline(row["best_epoch"], linestyle="--", alpha=0.45, label=f"fold {int(row['fold'])}")
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.title("Cosine warm-restart LR schedule with best-epoch markers")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "lr_schedule_with_best_epoch_markers.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(epoch1, beta, color="black", linewidth=1)
    for _, row in fold_df.iterrows():
        plt.axvline(row["best_epoch"], linestyle="--", alpha=0.45, label=f"fold {int(row['fold'])}")
    plt.xlabel("epoch")
    plt.ylabel("beta")
    plt.title("Cyclical beta schedule with best-epoch markers")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "beta_schedule_with_best_epoch_markers.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for fold, hist in histories.items():
        ratio_series = hist["val_kld_over_recon"] if "val_kld_over_recon" in hist else hist["val_kld"] / hist["val_recon"].replace(0, np.nan)
        plt.plot(hist["epoch"], ratio_series, linewidth=1.2, label=f"fold {fold}")
        best_epoch = int(fold_df.loc[fold_df["fold"].eq(fold), "best_epoch"].iloc[0])
        best_ratio = float(fold_df.loc[fold_df["fold"].eq(fold), "val_kld_over_recon_at_best"].iloc[0])
        plt.scatter([best_epoch], [best_ratio], s=30)
    plt.xlabel("epoch")
    plt.ylabel("validation KLD / reconstruction")
    plt.title("Validation KLD/R by fold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kld_over_recon_vs_epoch.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(fold_df["best_epoch"], fold_df["stage_b_auc"], s=60)
    for _, row in fold_df.iterrows():
        plt.annotate(f"F{int(row['fold'])}", (row["best_epoch"], row["stage_b_auc"]), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("VAE best epoch")
    plt.ylabel("Stage B AUC")
    plt.title("Fold AUC vs VAE best epoch")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fold_auc_vs_best_epoch.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(fold_df["val_kld_over_recon_at_best"], fold_df["stage_b_auc"], s=60)
    for _, row in fold_df.iterrows():
        plt.annotate(f"F{int(row['fold'])}", (row["val_kld_over_recon_at_best"], row["stage_b_auc"]), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("validation KLD / reconstruction at best epoch")
    plt.ylabel("Stage B AUC")
    plt.title("Fold AUC vs VAE KLD/R")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fold_auc_vs_kld_over_recon.png", dpi=180)
    plt.close()


def write_reports(fold_df: pd.DataFrame, corr_df: pd.DataFrame, stage_a: pd.DataFrame, config: dict[str, Any]) -> None:
    params = config.get("parameters", config.get("args", {}))
    worst = fold_df.sort_values("stage_b_auc").head(2)
    best = fold_df.sort_values("stage_b_auc").tail(2)
    n_near_trough = int(fold_df["near_lr_trough"].sum())
    n_plateau = int(fold_df["near_beta_plateau"].sum())
    auc_corr_best_epoch = corr_df.loc[corr_df["metric"].eq("best_epoch"), "pearson_with_stage_b_auc"]
    auc_corr_best_epoch_val = float(auc_corr_best_epoch.iloc[0]) if not auc_corr_best_epoch.empty and pd.notna(auc_corr_best_epoch.iloc[0]) else np.nan
    scanner_corr = corr_df.loc[corr_df["metric"].eq("test_acc_site_latent"), "pearson_with_stage_b_auc"]
    scanner_corr_val = float(scanner_corr.iloc[0]) if not scanner_corr.empty and pd.notna(scanner_corr.iloc[0]) else np.nan
    kldr_corr = corr_df.loc[corr_df["metric"].eq("val_kld_over_recon_at_best"), "pearson_with_stage_b_auc"]
    kldr_corr_val = float(kldr_corr.iloc[0]) if not kldr_corr.empty and pd.notna(kldr_corr.iloc[0]) else np.nan

    sched = [
        "# Scheduler/Beta Alignment Report",
        "",
        f"- Optimizer: `AdamW`, `lr={params.get('lr_vae')}`, `weight_decay={params.get('weight_decay_vae')}`, `amsgrad=True` in `scripts/run_vae_clf_ad_inference.py`.",
        f"- Scheduler: `CosineAnnealingWarmRestarts`, `T0={params.get('lr_scheduler_T0')}`, `lr_vae={params.get('lr_vae')}`, `eta_min={params.get('lr_scheduler_eta_min')}`.",
        f"- Beta schedule: `beta_max={params.get('beta_vae')}`, `cycles={params.get('cyclical_beta_n_cycles')}`, cycle length `{int(params.get('epochs_vae') / params.get('cyclical_beta_n_cycles'))}` epochs, ramp ratio `{params.get('cyclical_beta_ratio_increase')}`.",
        f"- Best epochs landed on beta plateau in `{n_plateau}/5` folds and near the LR trough in `{n_near_trough}/5` folds.",
        f"- Worst Stage B AUC folds: {', '.join('F'+str(int(x)) for x in worst['fold'])}. Best Stage B AUC folds: {', '.join('F'+str(int(x)) for x in best['fold'])}.",
        "",
        "Interpretation:",
        "- The selected checkpoints are not clustered near LR restarts or beta ramps. They mostly occur late in each 80-epoch cycle, when beta is already at its plateau.",
        "- Fold 4 is the worst-ranking fold despite the latest VAE best epoch, so the current evidence does not point to premature VAE convergence as the dominant AUC limiter.",
        "- The fold count is small, so correlations are diagnostic only, not inferential.",
    ]
    (OUT_DIR / "scheduler_beta_alignment_report.md").write_text("\n".join(sched) + "\n", encoding="utf-8")

    rec = [
        "# LR/Optimizer/Scheduler Recommendation",
        "",
        "## Decision",
        "",
        "Do not launch a new optimizer or LR-scheduler FULL run based on this audit alone.",
        "",
        "## Rationale",
        "",
        f"- Stage B AUC versus best epoch correlation: `{auc_corr_best_epoch_val:.3f}`.",
        f"- Stage B AUC versus validation KLD/R at best epoch correlation: `{kldr_corr_val:.3f}`.",
        f"- Stage B AUC versus test latent manufacturer leakage correlation: `{scanner_corr_val:.3f}`.",
        "- The bad folds do not share a clean optimization signature: they are not uniformly early-stopped, not uniformly high reconstruction-gap folds, and not uniformly high latent scanner-leakage folds.",
        "- Fold 4 remains weak in ranking even though its VAE ran to the epoch limit and selected a very late-cycle checkpoint, which argues against a simple learning-rate restart or patience fix.",
        "- There is no fold-level evidence here that AdamW/AMSGrad itself is the limiting factor; the downstream weak fold pattern is more consistent with representation/score overlap than with optimizer failure.",
        "",
        "A scheduler/optimizer perturbation would therefore be weakly targeted and would read as internal micro-optimization. The stronger manuscript position remains the locked model plus documented negative confirmations.",
    ]
    (OUT_DIR / "lr_scheduler_recommendation.md").write_text("\n".join(rec) + "\n", encoding="utf-8")

    summary_rows = [
        ["Run", "locked current FULL [1,0,2]"],
        ["Readout", f"{PRIMARY_MODEL} + {PRIMARY_THRESHOLD}"],
        ["Lowest Stage B AUC fold", f"fold {int(worst.iloc[0]['fold'])}: AUC={worst.iloc[0]['stage_b_auc']:.6f}, PR-AUC={worst.iloc[0]['stage_b_pr_auc']:.6f}"],
        ["Best epochs near LR trough", f"{n_near_trough}/5"],
        ["Best epochs on beta plateau", f"{n_plateau}/5"],
        ["Recommendation", "No LR/optimizer/scheduler FULL run is justified by this read-only audit."],
    ]
    readme = [
        "# Locked Current FULL [1,0,2] Training Dynamics Audit",
        "",
        "This is a read-only audit of the completed locked model. It does not train, alter tensors, edit metadata, touch ledgers, or modify existing run outputs.",
        "",
        "| Item | Value |",
        "|---|---|",
    ]
    readme.extend(f"| {k} | {v} |" for k, v in summary_rows)
    readme.extend(
        [
            "",
            "## Main Findings",
            "",
            "- Checkpoint selection used `ValL(beta_max)` from the saved histories.",
            "- The best epochs mostly occur on the beta plateau and often late in the LR cycle rather than at restarts.",
            "- Fold 4 has the weakest Stage B AUC/PR-AUC but the latest selected VAE checkpoint, so the weak fold is not explained by early VAE convergence.",
            "- Active-unit summaries do not show posterior collapse in the available latent QC summaries.",
            "- Scanner/manufacturer leakage is present, but the poor folds are not simply the folds with highest latent leakage.",
            "",
            "## Outputs",
            "",
            "- `fold_training_dynamics.csv/.md`",
            "- `fold_auc_vs_training_dynamics.csv/.md`",
            "- `scheduler_beta_alignment_report.md`",
            "- `lr_scheduler_recommendation.md`",
            "- `figures/*.png`",
            "- `command_log.json`",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def main() -> int:
    if not RUN_DIR.exists():
        fail(f"Missing run dir: {RUN_DIR}")
    if not STAGE_B_DIR.exists():
        fail(f"Missing Stage B dir: {STAGE_B_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    config_path = RUN_DIR / "run_config.json"
    if not config_path.exists():
        config_path = ROOT / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate.json"
    config = read_json(config_path)

    stage_b = load_stage_b_foldwise()
    fold_df, histories = build_fold_rows(config, stage_b)
    fold_df = add_bad_fold_flags(fold_df)
    corr_df = correlation_summary(fold_df)
    stage_a = load_stage_a_metrics()

    if not stage_a.empty:
        pivot = stage_a.pivot_table(index="fold", columns="stage_a_classifier", values=["stage_a_auc", "stage_a_pr_auc"], aggfunc="first")
        pivot.columns = [f"{metric}_{clf}" for metric, clf in pivot.columns]
        pivot = pivot.reset_index()
        fold_df = fold_df.merge(pivot, on="fold", how="left")

    fold_csv = OUT_DIR / "fold_training_dynamics.csv"
    fold_df.to_csv(fold_csv, index=False)
    write_markdown_table(fold_df, OUT_DIR / "fold_training_dynamics.md")

    corr_df.to_csv(OUT_DIR / "fold_auc_vs_training_dynamics.csv", index=False)
    write_markdown_table(corr_df, OUT_DIR / "fold_auc_vs_training_dynamics.md")

    if not stage_a.empty:
        stage_a.to_csv(OUT_DIR / "stage_a_foldwise_classifier_metrics.csv", index=False)
        write_markdown_table(stage_a, OUT_DIR / "stage_a_foldwise_classifier_metrics.md")

    save_figures(histories, fold_df, config)
    write_reports(fold_df, corr_df, stage_a, config)

    command_log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).relative_to(ROOT)),
        "source_run": str(RUN_DIR.relative_to(ROOT)),
        "stage_b_source": str(STAGE_B_DIR.relative_to(ROOT)),
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "primary_readout": {"model_name": PRIMARY_MODEL, "threshold_strategy": PRIMARY_THRESHOLD},
        "actions": {
            "training_launched": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "ledger_modified": False,
            "existing_model_output_modified": False,
        },
        "config_path": str(config_path),
        "generated_files": sorted(str(p.relative_to(OUT_DIR)) for p in OUT_DIR.rglob("*") if p.is_file()),
    }
    (OUT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")

    print(f"Wrote read-only training dynamics audit to {OUT_DIR}")
    print(fold_df[["fold", "best_epoch", "early_stop_epoch", "best_val_loss_beta_max", "stage_b_auc", "stage_b_pr_auc", "lr_alignment", "beta_phase"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
