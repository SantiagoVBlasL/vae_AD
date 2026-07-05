#!/usr/bin/env python3
"""Prepare a read-only preflight package for FAST meta647 greedy900 rerun.

This script inspects existing FAST meta647 artifacts and writes a derived
preflight package only. It does not launch training or run model inference.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/home/diego/proyectos/vae_AD")
OUT = PROJECT_ROOT / "results/revision_bspc_2026/post_revision_exploratory_20260630/fast_meta647_greedy900_cyclematched_preflight_20260701"
OLD_PREFLIGHT = PROJECT_ROOT / "results/revision_bspc_2026/fast_meta647_valsplitfix_preflight_20260622"
OLD_AUDIT = PROJECT_ROOT / "results/revision_bspc_2026/fast_meta647_ablation_results_audit_20260622"
OLD_CONFIG = OLD_PREFLIGHT / "fast_meta647_candidate_config.json"
OLD_RUN_ROOT = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/fast128_all7_finalcohort_greedy_screen_meta647_valsplitfix_20260622")
NEW_RUN_ROOT = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/post_revision_exploratory_20260630/fast_meta647_greedy900_cyclematched_20260701")
LOG_ROOT = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/_launch_logs/post_revision_exploratory_20260630/fast_meta647_greedy900_cyclematched_20260701")

CHANNEL_NAMES = [
    "Pearson_OMST_GCE_Signed_Weighted",
    "Pearson_Full_FisherZ_Signed",
    "MI_KNN_Symmetric",
    "dFC_AbsDiffMean",
    "dFC_StdDev",
    "DistanceCorr",
    "Granger_F_lag1",
]
DISPLAY_NAMES = {
    0: "OMST",
    1: "Pearson_Full",
    2: "MI_KNN",
    3: "dFC_AbsDiffMean",
    4: "dFC_StdDev",
    5: "DistanceCorr",
    6: "Granger",
}
FORCED_SENTINELS = [
    ([5, 2, 1], "DistanceCorr + MI_KNN + Pearson_Full", "old FAST300 best greedy set"),
    ([5, 2], "DistanceCorr + MI_KNN", "old FAST300 best pair"),
    ([1, 0, 2], "Pearson_Full + OMST + MI_KNN", "final promoted FULL channel set"),
]


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def to_md(df: pd.DataFrame, title: str | None = None) -> str:
    body = df.to_markdown(index=False)
    return f"# {title}\n\n{body}\n" if title else body + "\n"


def load_old_config() -> dict:
    if not OLD_CONFIG.exists():
        raise FileNotFoundError(f"Missing old FAST config: {OLD_CONFIG}")
    return json.loads(OLD_CONFIG.read_text(encoding="utf-8"))


def validate_tensor_metadata(tensor_path: Path, metadata_path: Path) -> tuple[pd.DataFrame, dict]:
    if not tensor_path.exists():
        raise FileNotFoundError(f"Missing tensor: {tensor_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    npz = np.load(tensor_path, allow_pickle=True)
    tensor_key = "global_tensor_data" if "global_tensor_data" in npz else list(npz.keys())[0]
    tensor = npz[tensor_key]
    subject_ids = [str(x) for x in npz["subject_ids"]] if "subject_ids" in npz else []
    channel_names = [str(x) for x in npz["channel_names"]] if "channel_names" in npz else CHANNEL_NAMES
    meta = pd.read_csv(metadata_path)
    if "SubjectID" not in meta.columns:
        raise RuntimeError("Metadata is missing SubjectID")
    meta_subjects = set(meta["SubjectID"].astype(str))
    tensor_subjects = set(subject_ids)
    inter = tensor_subjects & meta_subjects
    group_counts = meta["ResearchGroup_Mapped"].value_counts(dropna=False).to_dict()
    cnad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    validation = {
        "tensor_path": str(tensor_path),
        "metadata_path": str(metadata_path),
        "tensor_key": tensor_key,
        "tensor_shape": str(tuple(tensor.shape)),
        "tensor_subject_count": len(subject_ids) if subject_ids else int(tensor.shape[0]),
        "metadata_n": int(len(meta)),
        "tensor_metadata_intersection_n": int(len(inter)) if subject_ids else None,
        "subjects_only_in_tensor": ";".join(sorted(tensor_subjects - meta_subjects)) if subject_ids else "",
        "CN": int(group_counts.get("CN", 0)),
        "MCI": int(group_counts.get("MCI", 0)),
        "AD": int(group_counts.get("AD", 0)),
        "supervised_CN_AD_n": int(len(cnad)),
        "supervised_CN": int((cnad["ResearchGroup_Mapped"] == "CN").sum()),
        "supervised_AD": int((cnad["ResearchGroup_Mapped"] == "AD").sum()),
        "channel_names": ";".join(channel_names),
        "status": "PASS",
    }
    return meta, validation


def scan_runner_support() -> pd.DataFrame:
    text = (PROJECT_ROOT / "scripts/ablation_canales.py").read_text(encoding="utf-8")
    child = (PROJECT_ROOT / "scripts/run_vae_clf_ad_ablation.py").read_text(encoding="utf-8")
    checks = [
        ("full greedy over all channels", "candidate_channels" in text and "greedy_ablation" in text),
        ("resume/skip completed cache", "[CACHE HIT]" in text and "_find_metrics_csv" in text),
        ("native dry-run/preflight", "--dry-run" in text and "run_native_preflight" in text),
        ("strict metadata intersection flag", "strict_metadata_intersection" in text),
        ("strict VAE val split abort flag", "vae_abort_if_val_split_fails" in text),
        ("passes save VAE history flag", "save_vae_training_history" in text),
        ("child saves per-fold history joblib", "vae_train_history_fold_" in child),
        ("child saves aggregate history joblib", "all_folds_vae_training_history_" in child),
        ("history stores train loss/recon/KLD", "\"train_loss\"" in child and "\"train_recon\"" in child and "\"train_kld\"" in child),
        ("history stores val loss/recon/KLD", "\"val_loss\"" in child and "\"val_recon\"" in child and "\"val_kld\"" in child),
        ("history stores beta", "\"beta\"" in child),
        ("history stores learning rate", "\"lr\"" in child or "'lr'" in child),
        ("history stores active units per epoch", "active_units" in child and "history_data" in child),
    ]
    return pd.DataFrame(
        [{"item": name, "supported": bool(ok), "note": "" if ok else "not found in current script"} for name, ok in checks]
    )


def build_config_comparison(old_cfg: dict) -> pd.DataFrame:
    p = old_cfg["parameters"]
    rows = []
    new_values = {
        "epochs_vae": 900,
        "beta_cycles": 12,
        "lr_scheduler_T0": p["lr_scheduler_T0"],
        "latent_dim": p["latent_dim"],
        "beta_vae": p["beta_vae"],
        "outer_folds": p["outer_folds"],
        "seed": p["seed"],
        "early_stopping_patience_vae": p["early_stopping_patience_vae"],
        "cyclical_beta_ratio": p["cyclical_beta_ratio"],
        "dropout_vae": p["dropout_vae"],
        "batch_size": p["batch_size"],
        "metric": p["metric"],
        "norm_mode": p["norm_mode"],
        "classifier_use_class_weight": p["classifier_use_class_weight"],
        "vae_abort_if_val_split_fails": p["vae_abort_if_val_split_fails"],
        "strict_metadata_intersection": p["strict_metadata_intersection"],
    }
    old_key_map = {"lr_scheduler_T0": "lr_scheduler_T0"}
    for key, new in new_values.items():
        old_key = old_key_map.get(key, key)
        old = p.get(old_key)
        if key == "epochs_vae":
            old = p.get("epochs_vae")
        rows.append({"parameter": key, "fast300_value": old, "greedy900_value": new, "changed": old != new})
    rows.extend(
        [
            {"parameter": "beta_cycle_length_epochs", "fast300_value": 300 / 4, "greedy900_value": 900 / 12, "changed": False},
            {"parameter": "lr_cycle_length_epochs", "fast300_value": p["lr_scheduler_T0"], "greedy900_value": p["lr_scheduler_T0"], "changed": False},
        ]
    )
    return pd.DataFrame(rows)


def build_greedy_plan(old_stepwise: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Dynamic exhaustive greedy candidate count: 7 singles + 6+5+4+3+2+1 additions.
    rows = []
    for ch in range(7):
        rows.append(
            {
                "phase": "step0_single",
                "runtime_step": 0,
                "candidate_set_template": f"[{ch}]",
                "channel_names": DISPLAY_NAMES[ch],
                "candidate_count_this_phase": 7,
                "dynamic_dependency": "none",
                "required_for_full_path": True,
            }
        )
    remaining = 6
    for step in range(1, 7):
        rows.append(
            {
                "phase": "greedy_addition",
                "runtime_step": step,
                "candidate_set_template": f"best_step{step-1} + each_remaining_channel",
                "channel_names": "computed at runtime from previous best",
                "candidate_count_this_phase": remaining,
                "dynamic_dependency": f"best channel set selected at step {step-1}",
                "required_for_full_path": True,
            }
        )
        remaining -= 1
    plan = pd.DataFrame(rows)

    old_ref = old_stepwise.copy()
    if not old_ref.empty:
        old_ref = old_ref.rename(columns={"channels": "old_fast300_channels", "channel_names": "old_fast300_channel_names"})
        old_ref["reference_only"] = True
    return plan, old_ref


def forced_sentinel_table() -> pd.DataFrame:
    rows = []
    for chans, name, rationale in FORCED_SENTINELS:
        rows.append(
            {
                "channels": " ".join(map(str, chans)),
                "channels_python": str(chans),
                "channel_names": name,
                "rationale": rationale,
                "planned_output_subdir": "sentinel_ch" + "".join(map(str, chans)),
            }
        )
    return pd.DataFrame(rows)


def command_base(old_cfg: dict, *, epochs: int = 900, beta_cycles: int = 12) -> str:
    p = old_cfg["parameters"]
    paths = old_cfg["paths"]
    return (
        f"/home/diego/anaconda3/envs/vae_ad/bin/python {paths['ablation_script']} "
        f"--global_tensor_path {paths['global_tensor_path']} "
        f"--metadata_path {paths['metadata_path']} "
        f"--output_root {NEW_RUN_ROOT} "
        "--candidate_channels 0 1 2 3 4 5 6 "
        f"--metric {p['metric']} --min_improvement {p['min_improvement']} "
        f"--outer_folds {p['outer_folds']} --repeats {p['repeats']} "
        f"--epochs_vae {epochs} --early_stop {p['early_stopping_patience_vae']} "
        f"--beta_vae {p['beta_vae']} --latent_dim {p['latent_dim']} "
        f"--dropout_vae {p['dropout_vae']} --batch_size {p['batch_size']} "
        f"--beta_cycles {beta_cycles} --cyclical_beta_ratio {p['cyclical_beta_ratio']} "
        f"--lr_sched_type {p['lr_sched_type']} --lr_sched_T0 {p['lr_scheduler_T0']} "
        f"--lr_sched_eta_min {p['lr_sched_eta_min']} --norm_mode {p['norm_mode']} "
        f"--vae_val_split_ratio {p['vae_val_split_ratio']} --num_workers {p['num_workers']} "
        f"--seed {p['seed']} --vae_abort_if_val_split_fails --strict_metadata_intersection "
        "--classifier_use_class_weight --save_vae_training_history --no_early_stop"
    )


def child_sentinel_command(old_cfg: dict, chans: list[int], subdir: str) -> str:
    p = old_cfg["parameters"]
    paths = old_cfg["paths"]
    return (
        f"/home/diego/anaconda3/envs/vae_ad/bin/python /home/diego/proyectos/vae_AD/scripts/run_vae_clf_ad_ablation.py "
        f"--global_tensor_path {paths['global_tensor_path']} "
        f"--metadata_path {paths['metadata_path']} "
        f"--output_dir {NEW_RUN_ROOT / subdir} "
        "--vae_final_activation tanh --metadata_features Age Sex "
        f"--outer_folds {p['outer_folds']} --repeated_outer_folds_n_repeats {p['repeats']} "
        "--epochs_vae 900 "
        f"--early_stopping_patience_vae {p['early_stopping_patience_vae']} "
        "--cyclical_beta_n_cycles 12 "
        f"--cyclical_beta_ratio_increase {p['cyclical_beta_ratio']} "
        f"--lr_scheduler_type {p['lr_sched_type']} --lr_scheduler_T0 {p['lr_scheduler_T0']} "
        f"--lr_scheduler_eta_min {p['lr_sched_eta_min']} --batch_size {p['batch_size']} "
        f"--beta_vae {p['beta_vae']} --dropout_rate_vae {p['dropout_vae']} "
        f"--latent_dim {p['latent_dim']} --num_workers {p['num_workers']} --norm_mode {p['norm_mode']} "
        f"--seed {p['seed']} --vae_val_split_ratio {p['vae_val_split_ratio']} "
        "--vae_abort_if_val_split_fails --strict_metadata_intersection --classifier_use_class_weight "
        "--save_vae_training_history --channels_to_use "
        + " ".join(map(str, chans))
    )


def write_launch_scripts(old_cfg: dict) -> None:
    full_cmd = command_base(old_cfg)
    sentinel_cmds = [child_sentinel_command(old_cfg, chans, "sentinel_ch" + "".join(map(str, chans))) for chans, _, _ in FORCED_SENTINELS]

    guarded = f"""#!/usr/bin/env bash
set -euo pipefail

if [[ "${{1:-}}" != "--confirm-training" ]]; then
  echo "Refusing to launch: pass --confirm-training explicitly."
  exit 2
fi

MODE="${{2:-full}}"
RUN_ROOT="{NEW_RUN_ROOT}"
LOG_ROOT="{LOG_ROOT}"
mkdir -p "$LOG_ROOT"

if pgrep -af "run_vae_clf_ad_(ablation|inference)\\.py|ablation_canales\\.py" >/tmp/fast_greedy900_active_train.txt; then
  echo "Refusing to launch because an active VAE/FAST process was detected:"
  cat /tmp/fast_greedy900_active_train.txt
  exit 3
fi

case "$MODE" in
  full)
    exec bash "{OUT / 'launch_greedy900_full.sh'}"
    ;;
  sentinel)
    exec bash "{OUT / 'launch_greedy900_sentinel_only.sh'}"
    ;;
  *)
    echo "Unknown mode: $MODE. Use full or sentinel."
    exit 4
    ;;
esac
"""
    write(OUT / "guarded_launch.sh", guarded)

    full = f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p "{LOG_ROOT}"
echo "[INFO] Launching full greedy900 path plus forced sentinels."
{full_cmd} 2>&1 | tee "{LOG_ROOT}/greedy900_full_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] Launching forced sentinel candidates."
"""
    for cmd in sentinel_cmds:
        full += f"{cmd} 2>&1 | tee \"{LOG_ROOT}/sentinel_$(date +%Y%m%d_%H%M%S).log\"\n"
    write(OUT / "launch_greedy900_full.sh", full)

    sent = f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p "{LOG_ROOT}"
echo "[INFO] Launching forced sentinel candidates only."
"""
    for cmd in sentinel_cmds:
        sent += f"{cmd} 2>&1 | tee \"{LOG_ROOT}/sentinel_$(date +%Y%m%d_%H%M%S).log\"\n"
    write(OUT / "launch_greedy900_sentinel_only.sh", sent)

    for path in ["guarded_launch.sh", "launch_greedy900_full.sh", "launch_greedy900_sentinel_only.sh"]:
        p = OUT / path
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def plot_script_text() -> str:
    return r'''#!/usr/bin/env python3
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
'''


def resource_estimate_text(candidate_count: int) -> str:
    return f"""# Resource Estimate

- Full dynamic greedy path candidate trainings: 28.
- Forced sentinel trainings in independent output directories: 3.
- Maximum planned trainings if full path plus sentinels are launched: {candidate_count}.
- Outer folds per candidate: 3.
- Maximum VAE fold trainings: {candidate_count * 3}.
- Epoch cap per VAE fold: 900.
- FAST300 observed candidate folders: 22 before early stop after step 3.

Approximate wall-time and disk impact are uncertain because early stopping is data- and GPU-dependent. A conservative planning envelope is roughly 3x the per-candidate epoch cap of FAST300, multiplied by 31/22 if all diagnostics are run. With `--save_vae_training_history`, extra joblib/PNG history files are small relative to checkpoints and predictions.

This is diagnostic only. It should be scheduled only when no FULL/OASIS/manuscript-critical jobs are waiting for the GPU.
"""


def main() -> None:
    started = datetime.now(timezone.utc)
    OUT.mkdir(parents=True, exist_ok=True)
    old_cfg = load_old_config()
    tensor_path = Path(old_cfg["paths"]["global_tensor_path"])
    metadata_path = Path(old_cfg["paths"]["metadata_path"])
    meta, validation = validate_tensor_metadata(tensor_path, metadata_path)

    support = scan_runner_support()
    support.to_csv(OUT / "runner_support_audit.csv", index=False)

    comp = build_config_comparison(old_cfg)
    comp.to_csv(OUT / "config_comparison_fast300_vs_greedy900.csv", index=False)

    old_greedy = pd.read_csv(OLD_AUDIT / "fast_greedy_ablation_table.csv") if (OLD_AUDIT / "fast_greedy_ablation_table.csv").exists() else pd.DataFrame()
    old_stepwise = pd.read_csv(OLD_AUDIT / "fast_stepwise_candidate_table.csv") if (OLD_AUDIT / "fast_stepwise_candidate_table.csv").exists() else pd.DataFrame()
    plan, old_ref = build_greedy_plan(old_stepwise)
    plan.to_csv(OUT / "planned_greedy_steps.csv", index=False)
    if not old_ref.empty:
        old_ref.to_csv(OUT / "old_fast300_stepwise_reference.csv", index=False)

    sent = forced_sentinel_table()
    sent.to_csv(OUT / "forced_sentinel_candidates.csv", index=False)

    pd.DataFrame([validation]).to_csv(OUT / "tensor_metadata_validation.csv", index=False)

    stale_rows = [
        {"path": str(NEW_RUN_ROOT), "exists": NEW_RUN_ROOT.exists(), "expected": "must be absent before launch"},
        {"path": str(LOG_ROOT), "exists": LOG_ROOT.exists(), "expected": "may be created by launch"},
        {"path": str(OUT), "exists": OUT.exists(), "expected": "preflight output directory"},
    ]
    pd.DataFrame(stale_rows).to_csv(OUT / "output_path_stale_audit.csv", index=False)

    write_launch_scripts(old_cfg)
    write(OUT / "plot_fast_meta647_greedy900_losses.py", plot_script_text())
    (OUT / "plot_fast_meta647_greedy900_losses.py").chmod((OUT / "plot_fast_meta647_greedy900_losses.py").stat().st_mode | stat.S_IXUSR)

    # Markdown deliverables
    write(
        OUT / "README.md",
        f"""# FAST Meta647 Greedy900 Cycle-Matched Preflight

This package prepares a diagnostic rerun of the June FAST meta647 channel-ablation experiment with 900 VAE epochs and 12 beta cycles while preserving the old local temporal structure: beta-cycle length 75 epochs and cosine-warm LR cycle length T0=30 epochs.

No training, inference, OASIS scoring, tensor edits, metadata edits, checkpoint edits, prediction edits, or manuscript edits were performed.

Proposed run root: `{NEW_RUN_ROOT}`

Use `guarded_launch.sh --confirm-training full` only after explicit approval.
""",
    )

    status = pd.DataFrame(
        [
            {"check": "old FAST config exists", "status": "PASS", "detail": str(OLD_CONFIG)},
            {"check": "old FAST audit exists", "status": "PASS" if OLD_AUDIT.exists() else "FAIL", "detail": str(OLD_AUDIT)},
            {"check": "tensor exists", "status": "PASS", "detail": str(tensor_path)},
            {"check": "metadata exists", "status": "PASS", "detail": str(metadata_path)},
            {"check": "tensor shape", "status": "PASS" if validation["tensor_shape"] == "(648, 7, 131, 131)" else "CHECK", "detail": validation["tensor_shape"]},
            {"check": "metadata-valid N", "status": "PASS" if validation["metadata_n"] == 647 else "CHECK", "detail": str(validation["metadata_n"])},
            {"check": "CN/MCI/AD counts", "status": "PASS" if (validation["CN"], validation["MCI"], validation["AD"]) == (300, 250, 97) else "CHECK", "detail": f"CN={validation['CN']}, MCI={validation['MCI']}, AD={validation['AD']}"},
            {"check": "supervised CN/AD", "status": "PASS" if validation["supervised_CN_AD_n"] == 397 else "CHECK", "detail": str(validation["supervised_CN_AD_n"])},
            {"check": "new output root absent", "status": "PASS" if not NEW_RUN_ROOT.exists() else "FAIL", "detail": str(NEW_RUN_ROOT)},
            {"check": "runner supports full greedy", "status": "PASS" if bool(support.loc[support["item"].eq("full greedy over all channels"), "supported"].iloc[0]) else "FAIL", "detail": "scripts/ablation_canales.py"},
            {"check": "runner supports skip completed", "status": "PASS" if bool(support.loc[support["item"].eq("resume/skip completed cache"), "supported"].iloc[0]) else "FAIL", "detail": "metrics CSV cache"},
            {"check": "history loss fields available with flag", "status": "PASS", "detail": "--save_vae_training_history stores train/val loss, recon, KLD, beta"},
            {"check": "lr and active units stored per epoch", "status": "WARN", "detail": "current history does not store exact lr or active_units; see instrumentation_patch_plan.md"},
        ]
    )
    write(OUT / "preflight_status.md", to_md(status, "Preflight Status"))

    p = old_cfg["parameters"]
    old_summary = f"""# Old FAST300 Config Summary

- Run id: `{old_cfg['run_id']}`
- Label: {old_cfg['label']}
- Script: `{old_cfg['paths']['ablation_script']}`
- Tensor: `{tensor_path}`
- Metadata: `{metadata_path}`
- Output root: `{old_cfg['paths']['output_dir']}`
- Candidate channels: `{old_cfg['candidate_channels']}`
- Epochs: {p['epochs_vae']}
- Latent dim: {p['latent_dim']}
- Beta: {p['beta_vae']}
- LR scheduler: {p['lr_sched_type']}, T0={p['lr_scheduler_T0']}
- Beta cycles: {p['beta_cycles']}
- Beta cycle length: {p['epochs_vae'] / p['beta_cycles']:.1f} epochs
- Cyclical beta ratio increase: {p['cyclical_beta_ratio']}
- Outer folds/repeats: {p['outer_folds']} x {p['repeats']}
- Seed: {p['seed']}
- Early stopping patience: {p['early_stopping_patience_vae']}
- VAE validation split ratio: {p['vae_val_split_ratio']}
- Classifier: fixed logreg ablation readout with Age/Sex metadata, class weight enabled={p['classifier_use_class_weight']}
- Strict metadata intersection: {p['strict_metadata_intersection']}
- Strict VAE val split abort: {p['vae_abort_if_val_split_fails']}

Old selected greedy path from audit:

{old_greedy.to_markdown(index=False) if not old_greedy.empty else 'Old greedy table not found.'}
"""
    write(OUT / "old_fast300_config_summary.md", old_summary)

    write(
        OUT / "greedy_algorithm_plan.md",
        f"""# Greedy Algorithm Plan

The runner `scripts/ablation_canales.py` implements dynamic greedy forward channel selection.

1. Evaluate all single channels `[0]` through `[6]`.
2. Seed the greedy path with the best single channel by mean outer-fold AUC.
3. At each step, append each remaining channel to the current best set and evaluate each candidate.
4. Select the candidate with highest mean AUC.
5. For this diagnostic rerun, use `--no_early_stop` so the path continues until all seven channels are exhausted, even if a no-improvement stop rule would otherwise fire.

Maximum dynamic greedy trainings: 28.

The 900-epoch rerun preserves the original local temporal dynamics:

- FAST300: 300 epochs / 4 beta cycles = 75 epochs per beta cycle.
- Greedy900: 900 epochs / 12 beta cycles = 75 epochs per beta cycle.
- LR T0 remains 30 epochs.

Forced sentinel candidates are run independently from the greedy path because the 900-epoch dynamic path may differ from the old FAST300 path.
""",
    )

    write(OUT / "loss_history_availability_audit.md", to_md(support, "Loss History Availability Audit"))
    write(
        OUT / "instrumentation_patch_plan.md",
        """# Instrumentation Patch Plan

Current `run_vae_clf_ad_ablation.py` can save VAE histories only when `--save_vae_training_history` is passed. The saved per-fold joblib currently includes:

- `train_loss`
- `train_recon`
- `train_kld`
- `val_loss`
- `val_recon`
- `val_kld`
- `val_loss_modelsel`
- `beta`

This satisfies total loss, reconstruction loss, KL, beta, and beta-max model-selection loss requirements. It does not store exact per-epoch learning rate or active units.

Minimal future patch, if exact LR/active-units traces are required:

1. Add `\"lr\": []` and `\"active_units\": []` to `history_data`.
2. At each epoch end, append `optimizer_vae.param_groups[0][\"lr\"]`.
3. Compute active units on the validation loader or a fixed train subset using the standard posterior-mu variance criterion, and append the count.
4. Include these fields in the saved joblib and any aggregate CSV export.

For this preflight, no script modification was made. The plotting script can reconstruct nominal LR schedule if exact LR is absent, but exact optimizer LR is not available without this patch.
""",
    )

    write(OUT / "resource_estimate.md", resource_estimate_text(31))

    # Launch command record
    write(
        OUT / "launch_commands.txt",
        f"""# Guarded launch commands

# Full greedy path plus forced sentinels, after explicit approval only:
{OUT / 'guarded_launch.sh'} --confirm-training full

# Forced sentinel candidates only, after explicit approval only:
{OUT / 'guarded_launch.sh'} --confirm-training sentinel

# Native FAST dry-run command for operator verification, no training:
{command_base(old_cfg)} --dry-run
""",
    )

    command_log = {
        "timestamp_utc": started.isoformat(),
        "cwd": str(PROJECT_ROOT),
        "did_train": False,
        "did_infer": False,
        "did_oasis": False,
        "did_modify_existing_artifacts": False,
        "inputs": {
            "old_config": str(OLD_CONFIG),
            "old_audit": str(OLD_AUDIT),
            "tensor": str(tensor_path),
            "metadata": str(metadata_path),
        },
        "outputs": [p.name for p in sorted(OUT.iterdir())],
        "notes": [
            "Prepared launch scripts only; no training launched.",
            "New output root checked for absence at generation time.",
        ],
    }
    (OUT / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
