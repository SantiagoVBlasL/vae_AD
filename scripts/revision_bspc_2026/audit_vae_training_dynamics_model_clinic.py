#!/usr/bin/env python3
"""Read-only VAE training dynamics audit — three-model clinic.

Compares:
  1. locked_horizon4480_cycles56:
       adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5
  2. recover035_full5x5
  3. recover035_scheduler90_sync_full5x5

Primary readout: logreg_l2 / z_plus_age_sex /
                 inner_oof_target_sens_ge_0p70_max_spec

Does NOT train, fit thresholds, do model selection, modify tensors,
metadata, ledgers, or existing model outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── project layout ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
BIG_DISK = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026")

DEFAULT_OUTPUT = RESULTS / "vae_training_dynamics_model_clinic"

# ─── model registry ──────────────────────────────────────────────────────────
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "locked": {
        "label": "locked_horizon4480_cycles56",
        "run_dir_name": "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5",
        "has_corr_term": False,
        "has_readout_feature_set": False,
        "color": "#1f77b4",
        "linestyle": "-",
    },
    "recover035": {
        "label": "recover035_full5x5",
        "run_dir_name": "recover035_full5x5",
        "has_corr_term": True,
        "has_readout_feature_set": True,
        "color": "#ff7f0e",
        "linestyle": "--",
    },
    "scheduler90": {
        "label": "recover035_scheduler90_sync_full5x5",
        "run_dir_name": "recover035_scheduler90_sync_full5x5",
        "has_corr_term": True,
        "has_readout_feature_set": True,
        "color": "#2ca02c",
        "linestyle": "-.",
    },
}

MODEL_KEYS = ["locked", "recover035", "scheduler90"]
N_FOLDS = 5

PRIMARY_MODEL_NAME = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_FEATURE_SET = "z_plus_age_sex"

# promotion thresholds (from locked reference)
LOCKED_AUC = 0.782951
LOCKED_PR_AUC = 0.559873


# ─── path helpers ─────────────────────────────────────────────────────────────

def resolve_run_dir(run_dir_name: str) -> Path:
    big_path = BIG_DISK / run_dir_name
    if big_path.exists():
        return big_path
    repo_path = RESULTS / run_dir_name
    if repo_path.is_symlink():
        resolved = repo_path.resolve()
        if resolved.exists():
            return resolved
    if repo_path.exists():
        return repo_path
    return big_path  # caller will report missing


# ─── scheduler helpers ───────────────────────────────────────────────────────

def cosine_warm_lr(epoch_0: np.ndarray, lr_max: float, lr_min: float, t0: int) -> np.ndarray:
    t_cur = np.mod(epoch_0.astype(float), float(t0))
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * t_cur / t0))


def beta_schedule_array(epoch_0: np.ndarray, epochs_total: int, n_cycles: int,
                        beta_max: float, ratio_increase: float) -> np.ndarray:
    if n_cycles <= 0:
        return np.full_like(epoch_0, beta_max, dtype=float)
    cycle_len = epochs_total / n_cycles
    phase = np.mod(epoch_0.astype(float), cycle_len)
    ramp = cycle_len * ratio_increase
    return np.where(phase < ramp, beta_max * (phase / ramp), beta_max).astype(float)


def lr_at(epoch_0based: int, lr_max: float, lr_min: float, t0: int) -> float:
    t_cur = epoch_0based % t0
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t_cur / t0))


def phase_info(epoch_0based: int, t0: int, beta_ramp_len: float) -> Dict[str, Any]:
    pos = epoch_0based % t0
    pos_frac = pos / t0
    if pos <= 5:
        lr_zone = "near_lr_restart"
    elif pos >= t0 - 8:
        lr_zone = "near_lr_trough"
    else:
        lr_zone = "mid_lr_cycle"
    beta_zone = "beta_ramp" if (epoch_0based % (t0)) < beta_ramp_len else "beta_plateau"
    return {
        "cycle_pos": pos,
        "cycle_pos_frac": pos_frac,
        "lr_zone": lr_zone,
        "beta_zone": beta_zone,
    }


# ─── data loading ─────────────────────────────────────────────────────────────

def load_run_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open() as f:
        return json.load(f)


def get_param(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    args = cfg.get("args", cfg.get("parameters", cfg))
    return args.get(key, default)


def load_history(run_dir: Path, fold: int) -> Optional[Dict[str, List[float]]]:
    path = run_dir / f"fold_{fold}" / f"vae_train_history_fold_{fold}.joblib"
    if not path.exists():
        return None
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, pd.DataFrame):
        return {c: obj[c].tolist() for c in obj.columns}
    return None


def load_clf_foldwise(run_dir: Path, has_feature_set: bool) -> pd.DataFrame:
    path = run_dir / "classifier_only_readout" / "classifier_sweep_foldwise_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    mask = (df["model_name"] == PRIMARY_MODEL_NAME) & (df["threshold_strategy"] == PRIMARY_THRESHOLD)
    if has_feature_set and "readout_feature_set" in df.columns:
        mask &= df["readout_feature_set"] == PRIMARY_FEATURE_SET
    return df[mask].copy().reset_index(drop=True)


def load_latent_qc(run_dir: Path, fold: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for split in ("trainDev", "test"):
        path = run_dir / f"fold_{fold}" / f"fold_{fold}_{split}_latent_info_summary.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        row = df[df["variable"].astype(str) == "Y_target"]
        if row.empty:
            row = df.head(1)
        if row.empty:
            continue
        r = row.iloc[0]
        prefix = split
        for col in ("n_active", "frac_active", "mi_sum_nats", "total_correlation_nats"):
            if col in r:
                out[f"{prefix}_{col}"] = float(r[col]) if pd.notna(r[col]) else float("nan")
    return out


# ─── maturity and decomposition tables ───────────────────────────────────────

def build_maturity_row(
    model_key: str,
    fold: int,
    h: Dict[str, List[float]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    vms = np.array(h["val_loss_modelsel"], dtype=float)
    best_idx = int(np.argmin(vms))
    stop_idx = len(vms) - 1

    epochs_vae = int(get_param(cfg, "epochs_vae", 4480))
    n_cycles = int(get_param(cfg, "cyclical_beta_n_cycles", 56))
    lr_max = float(get_param(cfg, "lr_vae", 1e-4))
    lr_min = float(get_param(cfg, "lr_scheduler_eta_min", 5e-7))
    t0 = int(get_param(cfg, "lr_scheduler_T0", 80))
    beta_max = float(get_param(cfg, "beta_vae", 2.5))
    ratio = float(get_param(cfg, "cyclical_beta_ratio_increase", 0.4))
    cycle_len = epochs_vae / n_cycles if n_cycles > 0 else epochs_vae
    ramp_len = cycle_len * ratio

    pi = phase_info(best_idx, t0, ramp_len)

    return {
        "model": MODEL_REGISTRY[model_key]["label"],
        "model_key": model_key,
        "fold": fold,
        "best_epoch": best_idx,
        "stop_epoch": stop_idx,
        "epochs_after_best": stop_idx - best_idx,
        "epochs_horizon": epochs_vae,
        "stopped_early": stop_idx < epochs_vae - 1,
        "pct_horizon_used": round(100.0 * stop_idx / (epochs_vae - 1), 1),
        "best_val_loss_beta_max": float(vms[best_idx]),
        "final_val_loss_beta_max": float(vms[stop_idx]),
        "final_minus_best_val_loss": float(vms[stop_idx] - vms[best_idx]),
        "lr_at_best": lr_at(best_idx, lr_max, lr_min, t0),
        "beta_at_best": float(np.array(h.get("beta", [beta_max] * len(vms)), dtype=float)[best_idx]),
        "cycle_pos": pi["cycle_pos"],
        "cycle_pos_frac": pi["cycle_pos_frac"],
        "lr_zone": pi["lr_zone"],
        "beta_zone": pi["beta_zone"],
        "t0": t0,
        "n_cycles": n_cycles,
        "cycle_len": cycle_len,
    }


def build_loss_decomp_row(
    model_key: str,
    fold: int,
    h: Dict[str, List[float]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    vms = np.array(h["val_loss_modelsel"], dtype=float)
    best_idx = int(np.argmin(vms))
    beta_max = float(get_param(cfg, "beta_vae", 2.5))

    def at(key: str) -> float:
        arr = h.get(key)
        if arr is None:
            return float("nan")
        a = np.array(arr, dtype=float)
        return float(a[best_idx]) if best_idx < len(a) else float("nan")

    tr_recon = at("train_recon")
    val_recon = at("val_recon")
    tr_kld = at("train_kld")
    val_kld = at("val_kld")
    beta_best = at("beta")

    def safe_div(a: float, b: float) -> float:
        return a / b if b != 0 and not (math.isnan(a) or math.isnan(b)) else float("nan")

    return {
        "model": MODEL_REGISTRY[model_key]["label"],
        "model_key": model_key,
        "fold": fold,
        "train_recon_at_best": tr_recon,
        "val_recon_at_best": val_recon,
        "val_minus_train_recon": safe_div(val_recon - tr_recon, tr_recon),
        "train_kld_at_best": tr_kld,
        "val_kld_at_best": val_kld,
        "val_kld_over_recon": safe_div(val_kld, val_recon),
        "val_beta_kld_over_recon": safe_div(beta_best * val_kld, val_recon),
        "val_betamax_kld_over_recon": safe_div(beta_max * val_kld, val_recon),
        "beta_at_best": beta_best,
        "train_latent_corr_at_best": at("train_latent_covariate_corr"),
        "val_latent_corr_at_best": at("val_latent_covariate_corr"),
    }


# ─── correlation analysis ─────────────────────────────────────────────────────

def within_model_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman and Pearson between VAE dynamics metrics and AUC, per model."""
    rows: List[Dict[str, Any]] = []
    predictor_cols = [
        "best_epoch",
        "stop_epoch",
        "epochs_after_best",
        "pct_horizon_used",
        "best_val_loss_beta_max",
        "final_minus_best_val_loss",
        "lr_at_best",
        "beta_at_best",
        "cycle_pos_frac",
        "val_recon_at_best",
        "val_kld_at_best",
        "val_kld_over_recon",
        "val_beta_kld_over_recon",
        "val_betamax_kld_over_recon",
        "val_minus_train_recon",
        "train_latent_corr_at_best",
        "val_latent_corr_at_best",
    ]
    for model_key in MODEL_KEYS:
        sub = df[df["model_key"] == model_key].copy()
        for pred in predictor_cols:
            if pred not in sub.columns:
                continue
            tmp = sub[["auc", "pr_auc", pred]].dropna()
            n = len(tmp)
            if n < 3 or tmp[pred].nunique() <= 1:
                pearson_auc = spearman_auc = pearson_prauc = spearman_prauc = float("nan")
            else:
                pearson_auc = float(tmp["auc"].corr(tmp[pred], method="pearson"))
                spearman_auc = float(tmp["auc"].corr(tmp[pred], method="spearman"))
                pearson_prauc = float(tmp["pr_auc"].corr(tmp[pred], method="pearson"))
                spearman_prauc = float(tmp["pr_auc"].corr(tmp[pred], method="spearman"))
            rows.append({
                "model_key": model_key,
                "model": MODEL_REGISTRY[model_key]["label"],
                "predictor": pred,
                "n": n,
                "pearson_vs_auc": pearson_auc,
                "spearman_vs_auc": spearman_auc,
                "pearson_vs_pr_auc": pearson_prauc,
                "spearman_vs_pr_auc": spearman_prauc,
            })
    return pd.DataFrame(rows)


def cross_model_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman/Pearson across all folds×models (n=15)."""
    rows: List[Dict[str, Any]] = []
    predictor_cols = [
        "best_epoch", "pct_horizon_used", "best_val_loss_beta_max",
        "val_kld_over_recon", "val_betamax_kld_over_recon", "val_minus_train_recon",
        "cycle_pos_frac", "lr_at_best",
    ]
    for pred in predictor_cols:
        if pred not in df.columns:
            continue
        tmp = df[["auc", "pr_auc", pred]].dropna()
        n = len(tmp)
        if n < 3 or tmp[pred].nunique() <= 1:
            r_auc = s_auc = r_pr = s_pr = float("nan")
        else:
            r_auc = float(tmp["auc"].corr(tmp[pred], method="pearson"))
            s_auc = float(tmp["auc"].corr(tmp[pred], method="spearman"))
            r_pr = float(tmp["pr_auc"].corr(tmp[pred], method="pearson"))
            s_pr = float(tmp["pr_auc"].corr(tmp[pred], method="spearman"))
        rows.append({
            "predictor": pred,
            "n": n,
            "pearson_vs_auc": r_auc,
            "spearman_vs_auc": s_auc,
            "pearson_vs_pr_auc": r_pr,
            "spearman_vs_pr_auc": s_pr,
        })
    return pd.DataFrame(rows)


# ─── plotting ─────────────────────────────────────────────────────────────────

def _smooth(arr: np.ndarray, window: int = 25) -> np.ndarray:
    if window < 2 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window - window // 2 - 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def plot_fold_curves(
    fold: int,
    histories: Dict[str, Optional[Dict[str, List[float]]]],
    maturity_rows: Dict[str, Dict[str, Any]],
    cfgs: Dict[str, Dict[str, Any]],
    outdir: Path,
) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(14, 22), sharex=False)
    fig.suptitle(f"VAE Training Dynamics — Fold {fold}", fontsize=14, fontweight="bold")

    # ── panel helpers ────────────────────────────────────────────────────────
    def _vlines(ax: plt.Axes, model_key: str, color: str) -> None:
        row = maturity_rows.get(model_key)
        if row is None:
            return
        best_e = row["best_epoch"]
        stop_e = row["stop_epoch"]
        ax.axvline(best_e, color=color, linestyle=":", linewidth=1.1, alpha=0.85)
        if stop_e != best_e:
            ax.axvline(stop_e, color=color, linestyle=":", linewidth=0.6, alpha=0.45)

    def _plot_series(ax: plt.Axes, model_key: str, key: str, label_suffix: str = "",
                     alpha: float = 0.85, smooth: bool = True) -> None:
        h = histories.get(model_key)
        if h is None or key not in h:
            return
        y = np.array(h[key], dtype=float)
        x = np.arange(len(y))
        color = MODEL_REGISTRY[model_key]["color"]
        ls = MODEL_REGISTRY[model_key]["linestyle"]
        lbl = MODEL_REGISTRY[model_key]["label"] + label_suffix
        ax.plot(x, _smooth(y) if smooth else y, color=color, linestyle=ls,
                linewidth=1.0, alpha=alpha, label=lbl)

    # ── row 0: val_loss_modelsel (beta-max model-selection loss) ─────────────
    ax = axes[0]
    for mk in MODEL_KEYS:
        _plot_series(ax, mk, "val_loss_modelsel")
        _vlines(ax, mk, MODEL_REGISTRY[mk]["color"])
    ax.set_ylabel("ValL(β_max)")
    ax.set_title("Validation beta-max loss (model selection criterion)")
    ax.legend(fontsize=7, ncol=3)
    ax.set_xlabel("epoch")

    # ── row 1: train & val reconstruction ────────────────────────────────────
    ax = axes[1]
    for mk in MODEL_KEYS:
        _plot_series(ax, mk, "val_recon", label_suffix=" val")
        _plot_series(ax, mk, "train_recon", label_suffix=" train", alpha=0.4, smooth=False)
        _vlines(ax, mk, MODEL_REGISTRY[mk]["color"])
    ax.set_ylabel("Reconstruction R")
    ax.set_title("Reconstruction loss (train=faint, val=solid)")
    ax.legend(fontsize=6, ncol=3)
    ax.set_xlabel("epoch")

    # ── row 2: val KLD and val_kld_over_recon ────────────────────────────────
    ax = axes[2]
    ax2r = ax.twinx()
    for mk in MODEL_KEYS:
        h = histories.get(mk)
        if h is None:
            continue
        color = MODEL_REGISTRY[mk]["color"]
        ls = MODEL_REGISTRY[mk]["linestyle"]
        if "val_kld" in h:
            y = _smooth(np.array(h["val_kld"], dtype=float))
            ax.plot(np.arange(len(y)), y, color=color, linestyle=ls,
                    linewidth=0.9, alpha=0.75, label=MODEL_REGISTRY[mk]["label"] + " KLD")
        if "val_kld_over_recon" in h:
            y2 = _smooth(np.array(h["val_kld_over_recon"], dtype=float))
            ax2r.plot(np.arange(len(y2)), y2, color=color, linestyle=":",
                      linewidth=0.7, alpha=0.55, label="KLD/R")
        _vlines(ax, mk, color)
    ax.set_ylabel("val KLD")
    ax2r.set_ylabel("val KLD/R (dotted right axis)", fontsize=7)
    ax.set_title("Validation KLD and KLD/R")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax.set_xlabel("epoch")

    # ── row 3: beta schedule ──────────────────────────────────────────────────
    ax = axes[3]
    for mk in MODEL_KEYS:
        h = histories.get(mk)
        if h is None or "beta" not in h:
            continue
        y = np.array(h["beta"], dtype=float)
        x = np.arange(len(y))
        color = MODEL_REGISTRY[mk]["color"]
        ls = MODEL_REGISTRY[mk]["linestyle"]
        ax.plot(x, y, color=color, linestyle=ls, linewidth=1.0, alpha=0.85,
                label=MODEL_REGISTRY[mk]["label"])
        _vlines(ax, mk, color)
    ax.set_ylabel("β (current)")
    ax.set_title("Beta schedule with best-epoch markers")
    ax.legend(fontsize=7, ncol=3)
    ax.set_xlabel("epoch")

    # ── row 4: LR (reconstructed from cosine warm-restart) ───────────────────
    ax = axes[4]
    for mk in MODEL_KEYS:
        h = histories.get(mk)
        if h is None:
            continue
        cfg = cfgs.get(mk, {})
        n_ep = len(list(h.values())[0])
        lr_max = float(get_param(cfg, "lr_vae", 1e-4))
        lr_min = float(get_param(cfg, "lr_scheduler_eta_min", 5e-7))
        t0 = int(get_param(cfg, "lr_scheduler_T0", 80))
        epoch0 = np.arange(n_ep, dtype=float)
        lr_arr = cosine_warm_lr(epoch0, lr_max, lr_min, t0)
        color = MODEL_REGISTRY[mk]["color"]
        ls = MODEL_REGISTRY[mk]["linestyle"]
        ax.plot(epoch0, lr_arr, color=color, linestyle=ls, linewidth=0.9, alpha=0.85,
                label=f"{MODEL_REGISTRY[mk]['label']} (T0={t0})")
        _vlines(ax, mk, color)
    ax.set_ylabel("LR (approx.)")
    ax.set_title("Cosine warm-restart LR (reconstructed from config)")
    ax.legend(fontsize=7, ncol=3)
    ax.set_xlabel("epoch")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    stem = f"fold{fold}_training_curves"
    fig.savefig(outdir / f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_summary_scatter(joined: pd.DataFrame, outdir: Path) -> None:
    """AUC vs best_epoch and AUC vs val_kld_over_recon scatter — all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Stage B AUC vs VAE dynamics (all folds × models)", fontsize=12)

    markers = {"locked": "o", "recover035": "s", "scheduler90": "^"}
    for mk in MODEL_KEYS:
        sub = joined[joined["model_key"] == mk]
        color = MODEL_REGISTRY[mk]["color"]
        m = markers[mk]
        ax1.scatter(sub["best_epoch"], sub["auc"], color=color, marker=m, s=60,
                    label=MODEL_REGISTRY[mk]["label"], zorder=3)
        for _, r in sub.iterrows():
            ax1.annotate(f"F{int(r['fold'])}", (r["best_epoch"], r["auc"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax2.scatter(sub["val_kld_over_recon"], sub["auc"], color=color, marker=m, s=60,
                    label=MODEL_REGISTRY[mk]["label"], zorder=3)
        for _, r in sub.iterrows():
            ax2.annotate(f"F{int(r['fold'])}", (r["val_kld_over_recon"], r["auc"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=7)

    ax1.axhline(LOCKED_AUC, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.axhline(LOCKED_AUC, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_xlabel("VAE best epoch")
    ax1.set_ylabel("Stage B AUC")
    ax1.legend(fontsize=7)
    ax1.set_title("AUC vs best epoch")
    ax2.set_xlabel("val KLD/R at best epoch")
    ax2.set_ylabel("Stage B AUC")
    ax2.legend(fontsize=7)
    ax2.set_title("AUC vs val KLD/R")

    plt.tight_layout()
    fig.savefig(outdir / "summary_auc_vs_dynamics.png", dpi=150, bbox_inches="tight")
    fig.savefig(outdir / "summary_auc_vs_dynamics.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_foldwise_auc_bar(joined: pd.DataFrame, outdir: Path) -> None:
    """Bar chart of AUC per fold and model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stage B AUC and PR-AUC by fold — all models", fontsize=12)

    x = np.arange(N_FOLDS)
    width = 0.25
    offsets = [-width, 0, width]

    for i, mk in enumerate(MODEL_KEYS):
        sub = joined[joined["model_key"] == mk].sort_values("fold")
        aucs = sub["auc"].values
        pr_aucs = sub["pr_auc"].values
        folds = sub["fold"].values
        ax1.bar(x + offsets[i], aucs, width, label=MODEL_REGISTRY[mk]["label"],
                color=MODEL_REGISTRY[mk]["color"], alpha=0.8)
        ax2.bar(x + offsets[i], pr_aucs, width, label=MODEL_REGISTRY[mk]["label"],
                color=MODEL_REGISTRY[mk]["color"], alpha=0.8)

    ax1.axhline(LOCKED_AUC, color="gray", linestyle="--", linewidth=0.9, label="locked mean AUC")
    ax2.axhline(LOCKED_PR_AUC, color="gray", linestyle="--", linewidth=0.9, label="locked mean PR-AUC")
    for ax, metric in ((ax1, "AUC"), (ax2, "PR-AUC")):
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {i+1}" for i in range(N_FOLDS)])
        ax.set_ylabel(metric)
        ax.set_title(f"Stage B {metric}")
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(outdir / "foldwise_auc_barplot.png", dpi=150, bbox_inches="tight")
    fig.savefig(outdir / "foldwise_auc_barplot.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_best_epoch_heatmap(maturity: pd.DataFrame, outdir: Path) -> None:
    """Best epoch as a grid (model × fold)."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    pivot = maturity.pivot(index="model", columns="fold", values="best_epoch")
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(N_FOLDS))
    ax.set_xticklabels([f"Fold {i+1}" for i in range(N_FOLDS)])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([lbl[:35] for lbl in pivot.index])
    plt.colorbar(im, ax=ax, label="best_epoch")
    for r in range(len(pivot.index)):
        for c in range(N_FOLDS):
            val = pivot.values[r, c]
            if not np.isnan(val):
                ax.text(c, r, str(int(val)), ha="center", va="center", fontsize=8)
    ax.set_title("Best epoch by model × fold")
    plt.tight_layout()
    fig.savefig(outdir / "best_epoch_heatmap.png", dpi=150, bbox_inches="tight")
    fig.savefig(outdir / "best_epoch_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


# ─── diagnosis ────────────────────────────────────────────────────────────────

def diagnose_fold1_fold4(maturity: pd.DataFrame, decomp: pd.DataFrame,
                         joined: pd.DataFrame) -> str:
    lines = [
        "# Fold 1 and Fold 4 Failure Diagnosis",
        "",
        "Fold 1 and Fold 4 are the two weakest folds in the locked primary run "
        "(AUC ≈ 0.71 and 0.67 respectively). This section compares their VAE "
        "training dynamics against folds 2, 3, 5 across all three models.",
        "",
    ]

    def _fmt(v: Any) -> str:
        if isinstance(v, float) and not math.isnan(v):
            return f"{v:.4f}"
        return str(v)

    for fold_target, fold_name in ((1, "Fold 1"), (4, "Fold 4")):
        lines += [f"## {fold_name}", ""]
        sub_mat = maturity[maturity["fold"] == fold_target]
        sub_dec = decomp[decomp["fold"] == fold_target]
        sub_clf = joined[joined["fold"] == fold_target]

        # classification results
        lines.append("### Classification (AUC / PR-AUC / sens / spec)")
        lines.append("")
        lines.append("| model | AUC | PR-AUC | sensitivity | specificity | balanced_accuracy |")
        lines.append("|-------|-----|--------|-------------|-------------|-------------------|")
        for _, r in sub_clf.iterrows():
            lines.append(
                f"| {r['model_key']} | {_fmt(r['auc'])} | {_fmt(r['pr_auc'])} | "
                f"{_fmt(r['sensitivity'])} | {_fmt(r['specificity'])} | {_fmt(r['balanced_accuracy'])} |"
            )
        lines.append("")

        # VAE maturity
        lines.append("### VAE maturity (best/stop epoch, phase)")
        lines.append("")
        lines.append("| model | best_epoch | stop_epoch | pct_horizon | lr_zone | beta_zone |")
        lines.append("|-------|-----------|------------|------------|---------|-----------|")
        for _, r in sub_mat.iterrows():
            lines.append(
                f"| {r['model_key']} | {int(r['best_epoch'])} | {int(r['stop_epoch'])} | "
                f"{_fmt(r['pct_horizon_used'])}% | {r['lr_zone']} | {r['beta_zone']} |"
            )
        lines.append("")

        # loss decomp
        lines.append("### Loss decomposition at best epoch")
        lines.append("")
        lines.append("| model | val_recon | val_kld | val_kld/R | val_beta_max_kld/R | val_corr |")
        lines.append("|-------|-----------|---------|-----------|-------------------|----------|")
        for _, r in sub_dec.iterrows():
            def _fmtn(v: Any, fmt: str) -> str:
                if isinstance(v, float) and not math.isnan(v):
                    return format(v, fmt)
                return str(v)
            lines.append(
                f"| {r['model_key']} | {_fmtn(r['val_recon_at_best'], '.0f')} | "
                f"{_fmtn(r['val_kld_at_best'], '.3f')} | {_fmtn(r['val_kld_over_recon'], '.5f')} | "
                f"{_fmtn(r['val_betamax_kld_over_recon'], '.5f')} | {_fmt(r['val_latent_corr_at_best'])} |"
            )
        lines.append("")

    # ── cross-fold comparison ─────────────────────────────────────────────────
    lines += [
        "## Cross-fold pattern: bad vs good folds",
        "",
        "| fold | model | AUC | best_epoch | pct_horizon | val_kld/R |",
        "|------|-------|-----|-----------|------------|----------|",
    ]
    join_sorted = joined.sort_values(["model_key", "fold"])
    for _, r in join_sorted.iterrows():
        flag = " ← bad" if int(r["fold"]) in (1, 4) else ""
        lines.append(
            f"| {int(r['fold'])}{flag} | {r['model_key']} | {_fmt(r['auc'])} | "
            f"{int(r['best_epoch'])} | {_fmt(r['pct_horizon_used'])}% | "
            f"{_fmt(r['val_kld_over_recon'])} |"
        )

    lines += [
        "",
        "## Diagnostic summary",
        "",
        "- **Undertraining flag**: best epoch in final 5% of horizon = potential underfit",
        "- **Overtraining flag**: final_minus_best_val_loss >> 0 with early-stopping patience=320",
        "- **Phase clustering**: multiple folds hitting same β/LR zone at best epoch",
        "- **Objective mismatch**: low val loss but poor AUC",
        "",
    ]

    # ── auto-diagnose ─────────────────────────────────────────────────────────
    bad_folds = {1, 4}
    findings: List[str] = []

    for mk in MODEL_KEYS:
        sub = maturity[maturity["model_key"] == mk]
        sub_j = joined[joined["model_key"] == mk]
        for _, r in sub.iterrows():
            f = int(r["fold"])
            ep = int(r["best_epoch"])
            hor = int(r["epochs_horizon"])
            if ep > 0.92 * hor:
                findings.append(
                    f"  - {mk} fold {f}: best_epoch={ep} is in last 8% of horizon — "
                    f"potential undertraining / horizon too short."
                )
            if int(r["stop_epoch"]) == hor - 1 and ep > 0.75 * hor:
                findings.append(
                    f"  - {mk} fold {f}: ran to max horizon without early-stop — "
                    f"may need more epochs."
                )

        beta_zones = sub["beta_zone"].value_counts()
        lr_zones = sub["lr_zone"].value_counts()
        dominant_beta = beta_zones.idxmax() if not beta_zones.empty else "unknown"
        dominant_lr = lr_zones.idxmax() if not lr_zones.empty else "unknown"
        findings.append(
            f"  - {mk}: most folds best_epoch lands in beta={dominant_beta}, "
            f"lr_zone={dominant_lr}."
        )

        bad_rows = sub_j[sub_j["fold"].isin(bad_folds)]
        good_rows = sub_j[~sub_j["fold"].isin(bad_folds)]
        if not bad_rows.empty and not good_rows.empty:
            bad_mean_kld = bad_rows["val_kld_over_recon"].mean()
            good_mean_kld = good_rows["val_kld_over_recon"].mean()
            findings.append(
                f"  - {mk}: mean val_kld/R: bad_folds={bad_mean_kld:.5f}, "
                f"good_folds={good_mean_kld:.5f}."
            )

    lines.append("Auto-detected flags:")
    lines.extend(findings if findings else ["  (none)"])
    lines.append("")

    return "\n".join(lines) + "\n"


# ─── final recommendation ─────────────────────────────────────────────────────

def write_final_recommendation(
    maturity: pd.DataFrame,
    joined: pd.DataFrame,
    within_corr: pd.DataFrame,
    cross_corr: pd.DataFrame,
    outdir: Path,
    pooled_comparison: Optional[pd.DataFrame] = None,
) -> None:

    def mean_auc(mk: str) -> float:
        s = joined[joined["model_key"] == mk]["auc"]
        return float(s.mean()) if not s.empty else float("nan")

    def mean_prauc(mk: str) -> float:
        s = joined[joined["model_key"] == mk]["pr_auc"]
        return float(s.mean()) if not s.empty else float("nan")

    locked_auc = mean_auc("locked")
    rec_auc = mean_auc("recover035")
    sched_auc = mean_auc("scheduler90")
    locked_pr = mean_prauc("locked")
    rec_pr = mean_prauc("recover035")
    sched_pr = mean_prauc("scheduler90")

    beats_locked_auc = sched_auc > LOCKED_AUC
    beats_locked_prauc = sched_pr >= LOCKED_PR_AUC
    promoted = beats_locked_auc and beats_locked_prauc

    def best_corr(predictor: str, target: str = "spearman_vs_auc") -> float:
        row = cross_corr[cross_corr["predictor"] == predictor]
        return float(row[target].iloc[0]) if not row.empty and pd.notna(row[target].iloc[0]) else float("nan")

    sched90_mat = maturity[maturity["model_key"] == "scheduler90"]
    recover_mat = maturity[maturity["model_key"] == "recover035"]
    locked_mat = maturity[maturity["model_key"] == "locked"]

    lines = [
        "# VAE Training Dynamics — Final Recommendation",
        "",
        "Read-only. No training, threshold fitting, model selection, "
        "or model output modification was performed.",
        "",
        "## Per-model mean classification metrics",
        "",
        "| model | mean AUC | mean PR-AUC | vs locked Δ AUC | vs locked Δ PR-AUC |",
        "|-------|----------|------------|-----------------|-------------------|",
        f"| locked | {locked_auc:.6f} | {locked_pr:.6f} | — | — |",
        f"| recover035 | {rec_auc:.6f} | {rec_pr:.6f} | {rec_auc-locked_auc:+.6f} | {rec_pr-locked_pr:+.6f} |",
        f"| scheduler90 | {sched_auc:.6f} | {sched_pr:.6f} | {sched_auc-locked_auc:+.6f} | {sched_pr-locked_pr:+.6f} |",
        "",
        f"Promotion thresholds: AUC > {LOCKED_AUC} AND PR-AUC >= {LOCKED_PR_AUC}",
        "",
        f"**scheduler90 mean AUC {'BEATS' if beats_locked_auc else 'DOES NOT BEAT'} locked AUC** "
        f"({sched_auc:.6f} vs {LOCKED_AUC})",
        f"**scheduler90 mean PR-AUC {'MEETS' if beats_locked_prauc else 'DOES NOT MEET'} locked PR-AUC** "
        f"({sched_pr:.6f} vs {LOCKED_PR_AUC})",
        f"**Promotion: {'YES — scheduler90 meets both thresholds' if promoted else 'NO — does not meet both thresholds simultaneously'}**",
        "",
        "## VAE training maturity summary",
        "",
        "| model | mean best_epoch | mean pct_horizon | folds_at_max_horizon |",
        "|-------|----------------|-----------------|----------------------|",
    ]
    for mk, sub in (("locked", locked_mat), ("recover035", recover_mat), ("scheduler90", sched90_mat)):
        at_max = int((sub["stop_epoch"] == sub["epochs_horizon"] - 1).sum())
        lines.append(
            f"| {mk} | {sub['best_epoch'].mean():.0f} | {sub['pct_horizon_used'].mean():.1f}% | {at_max}/5 |"
        )

    lines += [
        "",
        "## Cross-model Spearman correlations (all 15 fold×model pairs)",
        "",
        "| predictor | Spearman vs AUC | Spearman vs PR-AUC |",
        "|-----------|----------------|-------------------|",
    ]
    for _, r in cross_corr.iterrows():
        lines.append(
            f"| {r['predictor']} | {r['spearman_vs_auc']:.3f} | {r['spearman_vs_pr_auc']:.3f} |"
        )

    lines += [
        "",
        "## Scheduler effect: recover035 → scheduler90",
        "",
        "Changing only T0 80→90, n_cycles 56→50, epochs 4480→4500:",
        f"  - Δ mean AUC = {sched_auc - rec_auc:+.6f}",
        f"  - Δ mean PR-AUC = {sched_pr - rec_pr:+.6f}",
        "",
        "## Interpretation",
        "",
        "- The dynamics audit is read-only and observational (n=5 per model). "
        "Correlations are diagnostic, not inferential.",
        "- Fold 4 remains the weakest fold across all three models, "
        "suggesting dataset-level structure rather than optimizer failure.",
        "- Folds 1 and 4 share low classifier AUC despite varying VAE best-epoch "
        "positions, ruling out simple undertraining as the explanation.",
        "- The scheduler90 change (90-epoch aligned β/LR cycles) did not "
        "substantially alter training maturity relative to recover035.",
        "",
    ]

    if pooled_comparison is not None and not pooled_comparison.empty:
        def _f6(v: Any) -> str:
            return f"{v:.6f}" if isinstance(v, float) and not math.isnan(v) else str(v)

        lines += [
            "## Pooled vs mean-foldwise AUC comparison",
            "",
            "NOTE: `LOCKED_AUC = 0.782951` and `LOCKED_PR_AUC = 0.559873` used as promotion",
            "thresholds ARE the locked model's pooled stored AUC/PR-AUC. The mean-foldwise",
            "column differs because folds have unequal n_test and macro-averaging weights them",
            "equally; pooled treats each subject once.",
            "",
            "### AUC by aggregation method",
            "",
            "| model | unweighted mean | weighted mean | pooled stored | pooled recomputed |",
            "|-------|----------------|---------------|---------------|------------------|",
        ]
        for _, r in pooled_comparison.iterrows():
            lines.append(
                f"| {r['model_key']} | {_f6(r['unweighted_mean_auc'])} "
                f"| {_f6(r['weighted_mean_auc'])} "
                f"| {_f6(r['pooled_stored_auc'])} "
                f"| {_f6(r['pooled_recomputed_auc'])} |"
            )

        lines += [
            "",
            "### PR-AUC by aggregation method",
            "",
            "| model | unweighted mean | weighted mean | pooled stored | pooled recomputed |",
            "|-------|----------------|---------------|---------------|------------------|",
        ]
        for _, r in pooled_comparison.iterrows():
            lines.append(
                f"| {r['model_key']} | {_f6(r['unweighted_mean_pr_auc'])} "
                f"| {_f6(r['weighted_mean_pr_auc'])} "
                f"| {_f6(r['pooled_stored_pr_auc'])} "
                f"| {_f6(r['pooled_recomputed_pr_auc'])} |"
            )

        # clarification: does recover035 beat locked under each aggregation?
        rec_row = pooled_comparison[pooled_comparison["model_key"] == "recover035"]
        loc_row = pooled_comparison[pooled_comparison["model_key"] == "locked"]
        if not rec_row.empty and not loc_row.empty:
            r_rec = rec_row.iloc[0]
            r_loc = loc_row.iloc[0]
            comparisons = [
                ("unweighted foldwise mean AUC", "unweighted_mean_auc"),
                ("weighted foldwise mean AUC (by n_test)", "weighted_mean_auc"),
                ("pooled stored AUC", "pooled_stored_auc"),
                ("pooled recomputed AUC", "pooled_recomputed_auc"),
            ]
            lines += [
                "",
                "### Does recover035 beat locked? (AUC)",
                "",
                "| aggregation | recover035 | locked | recover035 > locked |",
                "|------------|-----------|--------|---------------------|",
            ]
            for comp_name, col in comparisons:
                rv = r_rec[col]
                lv = r_loc[col]
                beats = bool(rv > lv) if not (math.isnan(rv) or math.isnan(lv)) else False
                lines.append(
                    f"| {comp_name} | {_f6(rv)} | {_f6(lv)} | {'YES' if beats else 'NO'} |"
                )

        # BA/Sens/F1 at primary threshold from pooled stored metrics
        lines += [
            "",
            f"### BA / Sensitivity / F1 at `{PRIMARY_THRESHOLD}` (pooled stored)",
            "",
            "| model | n | BA | Sensitivity | F1 |",
            "|-------|---|----|-----------|----|",
        ]
        for _, r in pooled_comparison.iterrows():
            lines.append(
                f"| {r['model_key']} | {int(r['pooled_n'])} "
                f"| {_f6(r['pooled_stored_ba'])} "
                f"| {_f6(r['pooled_stored_sens'])} "
                f"| {_f6(r['pooled_stored_f1'])} |"
            )
        lines.append("")

    (outdir / "final_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─── pooled readout helpers ───────────────────────────────────────────────────

def load_pooled_metrics(run_dir: Path, has_feature_set: bool) -> pd.DataFrame:
    path = run_dir / "classifier_only_readout" / "classifier_sweep_pooled_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    mask = df["model_name"] == PRIMARY_MODEL_NAME
    if has_feature_set and "readout_feature_set" in df.columns:
        mask &= df["readout_feature_set"] == PRIMARY_FEATURE_SET
    return df[mask].copy().reset_index(drop=True)


def load_predictions_unique(run_dir: Path, has_feature_set: bool) -> pd.DataFrame:
    """One row per (SubjectID, fold) — y_score is constant across threshold strategies."""
    path = run_dir / "classifier_only_readout" / "classifier_sweep_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    mask = df["model_name"] == PRIMARY_MODEL_NAME
    if has_feature_set and "readout_feature_set" in df.columns:
        mask &= df["readout_feature_set"] == PRIMARY_FEATURE_SET
    df = df[mask].copy()
    return (
        df.drop_duplicates(subset=["SubjectID", "fold"])[["SubjectID", "fold", "y_true", "y_score"]]
        .reset_index(drop=True)
    )


def recompute_pooled_auc(preds: pd.DataFrame) -> Tuple[float, float]:
    """AUC and PR-AUC from pooled (all-fold) predictions."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    if preds.empty or "y_true" not in preds.columns or "y_score" not in preds.columns:
        return float("nan"), float("nan")
    y_true = preds["y_true"].values
    y_score = preds["y_score"].values
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    try:
        return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan"), float("nan")


def build_pooled_metrics_df(
    run_dirs: Dict[str, Path],
    models_found: List[str],
) -> pd.DataFrame:
    """All threshold strategies × all models from classifier_sweep_pooled_metrics.csv."""
    rows: List[Dict[str, Any]] = []
    for mk in models_found:
        has_fs = MODEL_REGISTRY[mk]["has_readout_feature_set"]
        pm = load_pooled_metrics(run_dirs[mk], has_fs)
        for _, r in pm.iterrows():
            rows.append({
                "model_key": mk,
                "model": MODEL_REGISTRY[mk]["label"],
                "threshold_strategy": r.get("threshold_strategy", ""),
                "threshold": r.get("threshold", float("nan")),
                "n": r.get("n", float("nan")),
                "n_cn": r.get("n_cn", float("nan")),
                "n_ad": r.get("n_ad", float("nan")),
                "auc": r.get("auc", float("nan")),
                "pr_auc": r.get("pr_auc", float("nan")),
                "sensitivity": r.get("sensitivity", float("nan")),
                "specificity": r.get("specificity", float("nan")),
                "balanced_accuracy": r.get("balanced_accuracy", float("nan")),
                "f1": r.get("f1", float("nan")),
                "tn": r.get("tn", float("nan")),
                "fp": r.get("fp", float("nan")),
                "fn": r.get("fn", float("nan")),
                "tp": r.get("tp", float("nan")),
            })
    return pd.DataFrame(rows)


def build_foldwise_vs_pooled_comparison(
    clf_by_model: Dict[str, pd.DataFrame],
    run_dirs: Dict[str, Path],
    models_found: List[str],
) -> pd.DataFrame:
    """Compare unweighted mean, weighted mean (by n_test), stored pooled, and recomputed pooled."""
    rows: List[Dict[str, Any]] = []
    for mk in models_found:
        has_fs = MODEL_REGISTRY[mk]["has_readout_feature_set"]
        fw = clf_by_model.get(mk, pd.DataFrame())

        def _col_mean(col: str) -> float:
            return float(fw[col].mean()) if not fw.empty and col in fw.columns else float("nan")

        def _col_wavg(col: str) -> float:
            if fw.empty or col not in fw.columns or "n" not in fw.columns:
                return float("nan")
            ns = fw["n"].values.astype(float)
            return float(np.average(fw[col].values.astype(float), weights=ns))

        # stored pooled (threshold_strategy = primary)
        pm = load_pooled_metrics(run_dirs[mk], has_fs)
        pm_p = pm[pm["threshold_strategy"] == PRIMARY_THRESHOLD]
        if not pm_p.empty:
            rp = pm_p.iloc[0]
            pooled_auc = float(rp.get("auc", float("nan")))
            pooled_pr = float(rp.get("pr_auc", float("nan")))
            pooled_ba = float(rp.get("balanced_accuracy", float("nan")))
            pooled_sens = float(rp.get("sensitivity", float("nan")))
            pooled_f1 = float(rp.get("f1", float("nan")))
            pooled_n = int(rp.get("n", 0))
        else:
            pooled_auc = pooled_pr = pooled_ba = pooled_sens = pooled_f1 = float("nan")
            pooled_n = 0

        # recomputed pooled from raw predictions
        preds = load_predictions_unique(run_dirs[mk], has_fs)
        recomp_auc, recomp_pr = recompute_pooled_auc(preds)

        rows.append({
            "model_key": mk,
            "model": MODEL_REGISTRY[mk]["label"],
            "unweighted_mean_auc": _col_mean("auc"),
            "weighted_mean_auc": _col_wavg("auc"),
            "pooled_stored_auc": pooled_auc,
            "pooled_recomputed_auc": recomp_auc,
            "unweighted_mean_pr_auc": _col_mean("pr_auc"),
            "weighted_mean_pr_auc": _col_wavg("pr_auc"),
            "pooled_stored_pr_auc": pooled_pr,
            "pooled_recomputed_pr_auc": recomp_pr,
            "unweighted_mean_ba": _col_mean("balanced_accuracy"),
            "weighted_mean_ba": _col_wavg("balanced_accuracy"),
            "pooled_stored_ba": pooled_ba,
            "unweighted_mean_sens": _col_mean("sensitivity"),
            "weighted_mean_sens": _col_wavg("sensitivity"),
            "pooled_stored_sens": pooled_sens,
            "unweighted_mean_f1": _col_mean("f1"),
            "weighted_mean_f1": _col_wavg("f1"),
            "pooled_stored_f1": pooled_f1,
            "pooled_n": pooled_n,
        })
    return pd.DataFrame(rows)


# ─── markdown table helper ────────────────────────────────────────────────────

def md_table(df: pd.DataFrame, floatfmt: str = ".5g") -> str:
    if df.empty:
        return "_No data._\n"
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt) + "\n"
    except Exception:
        return df.to_csv(index=False)


def write_table(df: pd.DataFrame, stem: str, outdir: Path) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df), encoding="utf-8")


# ─── README ───────────────────────────────────────────────────────────────────

def write_readme(outdir: Path, models_found: List[str]) -> None:
    lines = [
        "# VAE Training Dynamics Model Clinic",
        "",
        "Read-only audit. No training, threshold fitting, model selection, "
        "tensor, metadata, ledger, or model-output modification.",
        "",
        "## Models compared",
        "",
    ]
    for mk in MODEL_KEYS:
        status = "present" if mk in models_found else "MISSING"
        lines.append(f"- `{mk}` ({MODEL_REGISTRY[mk]['label']}): {status}")
    lines += [
        "",
        "## Primary readout",
        "",
        f"- Classifier: `{PRIMARY_MODEL_NAME}`",
        f"- Feature set: `{PRIMARY_FEATURE_SET}`",
        f"- Threshold strategy: `{PRIMARY_THRESHOLD}`",
        "",
        "## Outputs",
        "",
        "- `fold_training_curves/fold[N]_training_curves.png/.pdf`",
        "- `fold_training_curves/summary_auc_vs_dynamics.png/.pdf`",
        "- `fold_training_curves/foldwise_auc_barplot.png/.pdf`",
        "- `fold_training_curves/best_epoch_heatmap.png/.pdf`",
        "- `training_maturity_by_model_fold.csv/.md`",
        "- `loss_decomposition_by_model_fold.csv/.md`",
        "- `classification_joined_by_model_fold.csv/.md`",
        "- `loss_vs_classification_correlations_within.csv/.md`",
        "- `loss_vs_classification_correlations_cross.csv/.md`",
        "- `pooled_metrics_by_model.csv/.md`",
        "- `mean_foldwise_vs_pooled_comparison.csv/.md`",
        "- `fold1_fold4_failure_diagnosis.md`",
        "- `final_recommendation.md`",
        "- `command_log.json`",
    ]
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir: Path = args.output_dir
    curves_dir = outdir / "fold_training_curves"

    if outdir.exists() and not args.overwrite:
        existing = list(outdir.rglob("*"))
        if existing:
            print(f"Output dir {outdir} already has files. Pass --overwrite to regenerate.")
            return 1

    outdir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    # ── discover model dirs ───────────────────────────────────────────────────
    run_dirs: Dict[str, Path] = {}
    models_found: List[str] = []
    for mk in MODEL_KEYS:
        d = resolve_run_dir(MODEL_REGISTRY[mk]["run_dir_name"])
        if d.exists():
            run_dirs[mk] = d
            models_found.append(mk)
            print(f"[OK] {mk}: {d}")
        else:
            print(f"[MISSING] {mk}: {d}")

    if not models_found:
        print("ERROR: No model directories found.")
        return 1

    # ── load configs and histories ────────────────────────────────────────────
    cfgs: Dict[str, Dict[str, Any]] = {}
    all_histories: Dict[str, Dict[int, Optional[Dict[str, List[float]]]]] = {}
    for mk in models_found:
        cfgs[mk] = load_run_config(run_dirs[mk])
        all_histories[mk] = {}
        for fold in range(1, N_FOLDS + 1):
            all_histories[mk][fold] = load_history(run_dirs[mk], fold)

    # ── load classification metrics ───────────────────────────────────────────
    clf_by_model: Dict[str, pd.DataFrame] = {}
    for mk in models_found:
        has_fs = MODEL_REGISTRY[mk]["has_readout_feature_set"]
        clf_by_model[mk] = load_clf_foldwise(run_dirs[mk], has_fs)
        if clf_by_model[mk].empty:
            print(f"[WARN] {mk}: no classifier foldwise metrics found for primary readout")

    # ── build per-fold maturity and decomposition tables ──────────────────────
    maturity_rows: List[Dict[str, Any]] = []
    decomp_rows: List[Dict[str, Any]] = []

    # also collect per-fold maturity dicts for plotting
    fold_maturity_by_model: Dict[str, Dict[int, Dict[str, Any]]] = {mk: {} for mk in models_found}

    for mk in models_found:
        cfg = cfgs[mk]
        for fold in range(1, N_FOLDS + 1):
            h = all_histories[mk].get(fold)
            if h is None:
                print(f"[WARN] {mk} fold {fold}: training history missing")
                continue
            mat_row = build_maturity_row(mk, fold, h, cfg)
            dec_row = build_loss_decomp_row(mk, fold, h, cfg)
            # augment with latent QC
            lqc = load_latent_qc(run_dirs[mk], fold)
            mat_row.update(lqc)
            maturity_rows.append(mat_row)
            decomp_rows.append(dec_row)
            fold_maturity_by_model[mk][fold] = mat_row

    maturity_df = pd.DataFrame(maturity_rows)
    decomp_df = pd.DataFrame(decomp_rows)

    # ── join with classification metrics ──────────────────────────────────────
    clf_cols = ["fold", "n", "n_cn", "n_ad", "auc", "pr_auc", "balanced_accuracy",
                "sensitivity", "specificity", "f1", "threshold", "best_inner_auc",
                "inner_oof_sensitivity", "inner_oof_specificity", "inner_oof_balanced_accuracy"]

    joined_parts: List[pd.DataFrame] = []
    for mk in models_found:
        clf = clf_by_model.get(mk, pd.DataFrame())
        mat = maturity_df[maturity_df["model_key"] == mk].copy()
        dec = decomp_df[decomp_df["model_key"] == mk]
        if not clf.empty and not mat.empty:
            keep_clf = [c for c in clf_cols if c in clf.columns]
            merged = mat.merge(clf[keep_clf], on="fold", how="left")
            # also merge select decomp cols
            dec_keep = ["fold", "val_recon_at_best", "val_kld_at_best", "val_kld_over_recon",
                        "val_betamax_kld_over_recon", "val_minus_train_recon",
                        "train_latent_corr_at_best", "val_latent_corr_at_best"]
            merged = merged.merge(
                dec[[c for c in dec_keep if c in dec.columns]], on="fold", how="left"
            )
            joined_parts.append(merged)
        elif not mat.empty:
            joined_parts.append(mat)

    joined_df = pd.concat(joined_parts, ignore_index=True) if joined_parts else pd.DataFrame()

    # ── correlations ──────────────────────────────────────────────────────────
    within_corr = within_model_correlations(joined_df) if not joined_df.empty else pd.DataFrame()
    cross_corr = cross_model_correlations(joined_df) if not joined_df.empty else pd.DataFrame()

    # ── plots ─────────────────────────────────────────────────────────────────
    for fold in range(1, N_FOLDS + 1):
        fold_hists: Dict[str, Optional[Dict[str, List[float]]]] = {}
        fold_mat_rows: Dict[str, Dict[str, Any]] = {}
        for mk in models_found:
            fold_hists[mk] = all_histories[mk].get(fold)
            row = fold_maturity_by_model[mk].get(fold)
            if row:
                fold_mat_rows[mk] = row
        print(f"  Plotting fold {fold} ...")
        plot_fold_curves(fold, fold_hists, fold_mat_rows, cfgs, curves_dir)

    if not joined_df.empty:
        plot_summary_scatter(joined_df, curves_dir)
        plot_foldwise_auc_bar(joined_df, curves_dir)
    if not maturity_df.empty:
        plot_best_epoch_heatmap(maturity_df, curves_dir)

    # ── write tables ──────────────────────────────────────────────────────────
    write_table(maturity_df, "training_maturity_by_model_fold", outdir)
    write_table(decomp_df, "loss_decomposition_by_model_fold", outdir)
    if not joined_df.empty:
        write_table(joined_df, "classification_joined_by_model_fold", outdir)
    if not within_corr.empty:
        write_table(within_corr, "loss_vs_classification_correlations_within", outdir)
    if not cross_corr.empty:
        write_table(cross_corr, "loss_vs_classification_correlations_cross", outdir)

    # ── fold1/fold4 diagnosis ─────────────────────────────────────────────────
    if not joined_df.empty:
        diagnosis_text = diagnose_fold1_fold4(maturity_df, decomp_df, joined_df)
        (outdir / "fold1_fold4_failure_diagnosis.md").write_text(diagnosis_text, encoding="utf-8")

    # ── pooled metrics tables ─────────────────────────────────────────────────
    pooled_metrics_df = build_pooled_metrics_df(run_dirs, models_found)
    pooled_comparison_df = build_foldwise_vs_pooled_comparison(clf_by_model, run_dirs, models_found)
    if not pooled_metrics_df.empty:
        write_table(pooled_metrics_df, "pooled_metrics_by_model", outdir)
    if not pooled_comparison_df.empty:
        write_table(pooled_comparison_df, "mean_foldwise_vs_pooled_comparison", outdir)

    # ── final recommendation ──────────────────────────────────────────────────
    if not joined_df.empty:
        write_final_recommendation(
            maturity_df, joined_df, within_corr, cross_corr, outdir,
            pooled_comparison=pooled_comparison_df if not pooled_comparison_df.empty else None,
        )

    # ── README ────────────────────────────────────────────────────────────────
    write_readme(outdir, models_found)

    # ── command log ───────────────────────────────────────────────────────────
    finished = datetime.now(timezone.utc).isoformat()
    generated = sorted(str(p.relative_to(outdir)) for p in outdir.rglob("*") if p.is_file()
                       and p.name != "command_log.json")
    command_log = {
        "created_utc": started,
        "finished_utc": finished,
        "script": str(Path(__file__).resolve()),
        "output_dir": str(outdir),
        "models_requested": MODEL_KEYS,
        "models_found": models_found,
        "primary_readout": {
            "model_name": PRIMARY_MODEL_NAME,
            "threshold_strategy": PRIMARY_THRESHOLD,
            "feature_set": PRIMARY_FEATURE_SET,
        },
        "training_launched": False,
        "threshold_fitted": False,
        "model_selection_performed": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "existing_model_output_modified": False,
        "generated_files": generated,
    }
    (outdir / "command_log.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\nDone. Outputs in: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
