#!/usr/bin/env python3
"""Plot FAST meta647 greedy900 losses after an approved run completes."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_histories(run_root: Path) -> pd.DataFrame:
    rows = []
    for hist_path in sorted(run_root.glob("**/vae_train_history_fold_*.joblib")):
        run_tag = hist_path.parent.parent.name if hist_path.parent.name.startswith("fold_") else hist_path.parent.name
        fold = hist_path.stem.split("_")[-1]
        hist = joblib.load(hist_path)
        n = len(hist.get("train_loss", []))
        for i in range(n):
            beta = hist.get("beta", [np.nan] * n)[i]
            train_kld = hist.get("train_kld", [np.nan] * n)[i]
            val_kld = hist.get("val_kld", [np.nan] * n)[i] if i < len(hist.get("val_kld", [])) else np.nan
            rows.append({
                "run_tag": run_tag,
                "fold": int(fold),
                "epoch": i + 1,
                "train_total_loss": hist.get("train_loss", [np.nan] * n)[i],
                "train_recon_loss": hist.get("train_recon", [np.nan] * n)[i],
                "train_kl": train_kld,
                "val_total_loss": hist.get("val_loss", [np.nan] * n)[i] if i < len(hist.get("val_loss", [])) else np.nan,
                "val_modelsel_loss": hist.get("val_loss_modelsel", [np.nan] * n)[i] if i < len(hist.get("val_loss_modelsel", [])) else np.nan,
                "val_recon_loss": hist.get("val_recon", [np.nan] * n)[i] if i < len(hist.get("val_recon", [])) else np.nan,
                "val_kl": val_kld,
                "beta": beta,
                "train_beta_kl": beta * train_kld if pd.notna(train_kld) and pd.notna(beta) else np.nan,
                "val_beta_kl": beta * val_kld if pd.notna(val_kld) and pd.notna(beta) else np.nan,
                "lr": hist.get("lr", [np.nan] * n)[i] if "lr" in hist else np.nan,
                "active_units": hist.get("active_units", [np.nan] * n)[i] if "active_units" in hist else np.nan,
            })
    if not rows:
        raise SystemExit(f"No vae_train_history_fold_*.joblib files found under {run_root}")
    return pd.DataFrame(rows)


def mean_se(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = df.groupby(["run_tag", "epoch"], as_index=False)[metric].agg(["mean", "sem"]).reset_index()
    return g


def plot_metric(df: pd.DataFrame, metric: str, out: Path, ylabel: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    for run_tag, sub in df.groupby("run_tag"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for fold, fdf in sub.groupby("fold"):
            ax.plot(fdf["epoch"], fdf[metric], lw=1, alpha=0.45, label=f"fold {fold}")
            if metric == "val_modelsel_loss" and fdf[metric].notna().any():
                best_idx = fdf[metric].idxmin()
                ax.axvline(float(fdf.loc[best_idx, "epoch"]), color="k", lw=0.7, alpha=0.25)
        ax.set_title(f"{run_tag}: {ylabel}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out / f"{run_tag}_{metric}_per_fold.png", dpi=180)
        fig.savefig(out / f"{run_tag}_{metric}_per_fold.svg")
        plt.close(fig)

    m = mean_se(df, metric)
    fig, ax = plt.subplots(figsize=(10, 5))
    for run_tag, sub in m.groupby("run_tag"):
        x = sub["epoch"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        se = sub["sem"].fillna(0).to_numpy(dtype=float)
        ax.plot(x, y, lw=1.8, label=run_tag)
        ax.fill_between(x, y - se, y + se, alpha=0.12)
    ax.set_title(f"FAST greedy900: mean +/- SE {ylabel}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / f"{metric}_mean_se.png", dpi=180)
    fig.savefig(out / f"{metric}_mean_se.svg")
    plt.close(fig)


def plot_summary_auc(run_root: Path, out: Path) -> None:
    summary = run_root / "summary_ablation.csv"
    if not summary.exists():
        return
    df = pd.read_csv(summary)
    df.to_csv(out / "summary_ablation_plotted_values.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["step"], df["metric_mean"], marker="o", color="#1b9e77")
    ax.set_xlabel("Greedy step")
    ax.set_ylabel("Mean outer-fold ROC-AUC")
    ax.set_title("FAST greedy900 AUC by selected step")
    ax.grid(alpha=0.25)
    for _, row in df.iterrows():
        ax.annotate(str(row["channels_indices"]), (row["step"], row["metric_mean"]), xytext=(3, 4), textcoords="offset points", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "greedy_auc_curve.png", dpi=220)
    fig.savefig(out / "greedy_auc_curve.svg")
    plt.close(fig)

    if "delta_vs_prev" in df.columns:
        delta = pd.to_numeric(df["delta_vs_prev"], errors="coerce")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(df["step"], delta.fillna(0), color="#7570b3")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Greedy step")
        ax.set_ylabel("Delta AUC vs previous step")
        ax.set_title("FAST greedy900 stepwise delta AUC")
        fig.tight_layout()
        fig.savefig(out / "greedy_delta_auc.png", dpi=220)
        fig.savefig(out / "greedy_delta_auc.svg")
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--output-dir", default=None, type=Path)
    args = p.parse_args()
    out = args.output_dir or (args.run_root / "loss_plots")
    out.mkdir(parents=True, exist_ok=True)
    df = load_histories(args.run_root)
    df.to_csv(out / "loss_history_long.csv", index=False)
    for metric, label in [
        ("train_total_loss", "Train total loss"),
        ("val_total_loss", "Validation total loss at current beta"),
        ("val_modelsel_loss", "Validation model-selection loss at beta max"),
        ("train_recon_loss", "Train reconstruction loss"),
        ("val_recon_loss", "Validation reconstruction loss"),
        ("train_kl", "Train KL"),
        ("val_kl", "Validation KL"),
        ("train_beta_kl", "Train beta*KL"),
        ("val_beta_kl", "Validation beta*KL"),
        ("beta", "Beta schedule"),
    ]:
        plot_metric(df, metric, out, label)
    if df["lr"].notna().any():
        plot_metric(df, "lr", out, "Learning rate")
    plot_summary_auc(args.run_root, out)
    print(f"Wrote plots and plotted values to {out}")


if __name__ == "__main__":
    main()
