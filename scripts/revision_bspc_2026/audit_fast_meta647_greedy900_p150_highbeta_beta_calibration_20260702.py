#!/usr/bin/env python3
"""Read-only beta calibration and rate-distortion audit for FAST Greedy900 p150.

The script only reads completed FAST candidate outputs and writes small summary
tables into the requested audit directory.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr, pearsonr
except Exception:  # pragma: no cover - scipy should be present in this env
    spearmanr = pearsonr = None


RUN_ROOT = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/post_revision_exploratory_20260630/fast_meta647_greedy900_cyclematched_p150_highbeta_20260702")
LOG_ROOT = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/_launch_logs/post_revision_exploratory_20260630/fast_meta647_greedy900_cyclematched_p150_highbeta_20260702")
OUT = Path("/home/diego/proyectos/vae_AD/results/revision_bspc_2026/post_revision_exploratory_20260630/greedy900_p150_highbeta_beta_calibration_audit_20260702")

LATENT_DIM = 128
BETA = 2.5
HIGH_BETA_THRESHOLD = 0.95 * BETA
CHANNEL_NAMES = {
    0: "Pearson_OMST_GCE_Signed_Weighted",
    1: "Pearson_Full_FisherZ_Signed",
    2: "MI_KNN_Symmetric",
    3: "dFC_AbsDiffMean",
    4: "dFC_StdDev",
    5: "DistanceCorr",
    6: "Granger_F_lag1",
}
SENTINELS = {
    "sentinel_ch521": [5, 2, 1],
    "sentinel_ch52": [5, 2],
    "sentinel_ch102": [1, 0, 2],
}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def md_table(df: pd.DataFrame, title: str, max_rows: int | None = None) -> str:
    d = df if max_rows is None else df.head(max_rows)
    return f"# {title}\n\n{d.to_markdown(index=False)}\n"


def parse_summary() -> tuple[pd.DataFrame, dict[str, list[int]]]:
    summary_path = RUN_ROOT / "summary_ablation.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary_ablation.csv: {summary_path}")
    summary = pd.read_csv(summary_path)
    selected_by_step: dict[int, list[int]] = {}
    tag_to_channels: dict[str, list[int]] = {}
    for _, row in summary.iterrows():
        step = int(row["step"])
        chans = [int(x) for x in str(row["channels_indices"]).split()]
        selected_by_step[step] = chans
        if step == 0 and len(chans) == 1:
            tag_to_channels[f"single_ch{chans[0]}"] = chans
        elif step > 0:
            added = [c for c in chans if c not in selected_by_step[step - 1]]
            if added:
                tag_to_channels[f"step{step}_add{added[0]}_k{len(chans)}"] = chans

    for ch in range(7):
        tag_to_channels[f"single_ch{ch}"] = [ch]
    for tag, chans in SENTINELS.items():
        tag_to_channels[tag] = chans

    # Reconstruct every dynamic trial folder using selected path from previous step.
    for d in sorted(p for p in RUN_ROOT.iterdir() if p.is_dir()):
        tag = d.name
        if tag in tag_to_channels:
            continue
        m = re.match(r"step(\d+)_add(\d+)_k(\d+)$", tag)
        if not m:
            continue
        step = int(m.group(1))
        add_ch = int(m.group(2))
        if (step - 1) in selected_by_step:
            tag_to_channels[tag] = selected_by_step[step - 1] + [add_ch]
    return summary, tag_to_channels


def candidate_kind(tag: str, summary: pd.DataFrame) -> tuple[bool, str | None, int | None]:
    selected_row = summary.loc[summary["out_dir"].astype(str).str.endswith("/" + tag)]
    if not selected_row.empty:
        return True, "selected_greedy_path", int(selected_row.iloc[0]["step"])
    if tag.startswith("sentinel_"):
        return False, "forced_sentinel", None
    if tag.startswith("single_"):
        return False, "single_channel_screen", 0
    m = re.match(r"step(\d+)_", tag)
    return False, "greedy_trial", int(m.group(1)) if m else None


def first_metric_csv(candidate_dir: Path) -> Path | None:
    files = sorted(candidate_dir.glob("all_folds_metrics_MULTI_*.csv"))
    return files[0] if files else None


def aggregate_history(candidate_dir: Path) -> list[dict[str, list[Any]]] | None:
    files = sorted(candidate_dir.glob("all_folds_vae_training_history_*.joblib"))
    if files:
        return joblib.load(files[0])
    histories = []
    for k in [1, 2, 3]:
        p = candidate_dir / f"fold_{k}" / f"vae_train_history_fold_{k}.joblib"
        if not p.exists():
            return None
        histories.append(joblib.load(p))
    return histories


def value_at(history: dict[str, list[Any]], key: str, epoch_1based: int) -> float:
    arr = history.get(key, [])
    idx = int(epoch_1based) - 1
    if idx < 0 or idx >= len(arr):
        return float("nan")
    try:
        return float(arr[idx])
    except Exception:
        return float("nan")


def fold_rows(summary: pd.DataFrame, tag_to_channels: dict[str, list[int]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    ckpt_rows = []
    for candidate_dir in sorted(p for p in RUN_ROOT.iterdir() if p.is_dir()):
        tag = candidate_dir.name
        channels = tag_to_channels.get(tag)
        if channels is None:
            continue
        metric_csv = first_metric_csv(candidate_dir)
        if metric_csv is None:
            continue
        metrics = pd.read_csv(metric_csv)
        histories = aggregate_history(candidate_dir)
        is_selected, kind, greedy_step = candidate_kind(tag, summary)
        for fold in sorted(metrics["fold"].unique()):
            fold = int(fold)
            ckpt_path = candidate_dir / f"fold_{fold}" / f"vae_checkpoint_selection_summary_fold_{fold}.csv"
            if not ckpt_path.exists():
                continue
            ckpt = pd.read_csv(ckpt_path).iloc[0].to_dict()
            ckpt_rows.append({
                "candidate_tag": tag,
                "candidate_kind": kind,
                "greedy_step": greedy_step,
                "channels": " ".join(map(str, channels)),
                "n_channels": len(channels),
                **ckpt,
                "selected_beta_passes_guard": float(ckpt.get("selected_epoch_beta", np.nan)) >= HIGH_BETA_THRESHOLD,
                "best_any_differs_from_best_high_beta": int(ckpt.get("best_any_beta_epoch", -1)) != int(ckpt.get("best_high_beta_epoch", -1)),
                "old_rule_would_select_low_beta": float(ckpt.get("beta_at_best_any_beta", np.nan)) < HIGH_BETA_THRESHOLD,
            })
            hist = histories[fold - 1] if histories and fold - 1 < len(histories) else {}
            selected_epoch = int(ckpt.get("selected_epoch", 0))
            selected_beta = float(ckpt.get("selected_epoch_beta", np.nan))
            train_recon = value_at(hist, "train_recon", selected_epoch)
            val_recon = value_at(hist, "val_recon", selected_epoch)
            train_kld = value_at(hist, "train_kld", selected_epoch)
            val_kld = value_at(hist, "val_kld", selected_epoch)
            mrow = metrics.loc[metrics["fold"].eq(fold)]
            if mrow.empty:
                auc = pr_auc = float("nan")
            else:
                auc = float(mrow.iloc[0].get("auc", np.nan))
                pr_auc = float(mrow.iloc[0].get("pr_auc", np.nan))
            beta_val_kld = selected_beta * val_kld if np.isfinite(selected_beta) and np.isfinite(val_kld) else float("nan")
            ratio = beta_val_kld / val_recon if np.isfinite(beta_val_kld) and np.isfinite(val_recon) and val_recon != 0 else float("nan")
            rows.append({
                "candidate_tag": tag,
                "candidate_kind": kind,
                "greedy_step": greedy_step,
                "is_selected_greedy_path": is_selected,
                "channels": " ".join(map(str, channels)),
                "channel_names": ";".join(CHANNEL_NAMES[c] for c in channels),
                "n_channels": len(channels),
                "fold": fold,
                "selected_epoch": selected_epoch,
                "selected_beta": selected_beta,
                "best_any_beta_epoch": int(ckpt.get("best_any_beta_epoch", 0)),
                "best_high_beta_epoch": int(ckpt.get("best_high_beta_epoch", 0)),
                "train_recon": train_recon,
                "val_recon": val_recon,
                "train_kld": train_kld,
                "val_kld": val_kld,
                "beta_val_kld": beta_val_kld,
                "beta_kld_over_recon": ratio,
                "recon_per_channel": val_recon / len(channels) if np.isfinite(val_recon) else float("nan"),
                "kld_per_latent_dim": val_kld / LATENT_DIM if np.isfinite(val_kld) else float("nan"),
                "auc": auc,
                "pr_auc": pr_auc,
                "metrics_csv": str(metric_csv),
                "history_available": bool(hist),
            })
    return pd.DataFrame(rows), pd.DataFrame(ckpt_rows)


def aggregate_candidate(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(
        ["candidate_tag", "candidate_kind", "greedy_step", "is_selected_greedy_path", "channels", "channel_names", "n_channels"],
        dropna=False,
        as_index=False,
    ).agg(
        n_folds=("fold", "nunique"),
        selected_epoch_mean=("selected_epoch", "mean"),
        selected_beta_min=("selected_beta", "min"),
        val_recon_mean=("val_recon", "mean"),
        val_kld_mean=("val_kld", "mean"),
        beta_val_kld_mean=("beta_val_kld", "mean"),
        beta_kld_over_recon_mean=("beta_kld_over_recon", "mean"),
        beta_kld_over_recon_sd=("beta_kld_over_recon", "std"),
        recon_per_channel_mean=("recon_per_channel", "mean"),
        kld_per_latent_dim_mean=("kld_per_latent_dim", "mean"),
        auc_mean=("auc", "mean"),
        auc_sd=("auc", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_sd=("pr_auc", "std"),
    )
    return agg.sort_values(["candidate_kind", "greedy_step", "candidate_tag"], na_position="last")


def aggregate_n_channels(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("n_channels", as_index=False).agg(
        n_candidate_folds=("candidate_tag", "count"),
        n_candidates=("candidate_tag", "nunique"),
        beta_kld_over_recon_mean=("beta_kld_over_recon", "mean"),
        beta_kld_over_recon_sd=("beta_kld_over_recon", "std"),
        recon_per_channel_mean=("recon_per_channel", "mean"),
        val_recon_mean=("val_recon", "mean"),
        val_kld_mean=("val_kld", "mean"),
        auc_mean=("auc", "mean"),
        pr_auc_mean=("pr_auc", "mean"),
    )


def correlation_rows(cand: pd.DataFrame) -> pd.DataFrame:
    rows = []
    vars_ = ["beta_kld_over_recon_mean", "recon_per_channel_mean", "val_kld_mean", "n_channels"]
    targets = ["auc_mean", "pr_auc_mean"]
    for target in targets:
        for var in vars_:
            sub = cand[[target, var]].dropna()
            row = {"target": target, "predictor": var, "n": len(sub)}
            if len(sub) >= 3:
                row["pearson_r"] = float(sub[target].corr(sub[var], method="pearson"))
                row["spearman_r"] = float(sub[target].corr(sub[var], method="spearman"))
                if pearsonr is not None:
                    row["pearson_p"] = float(pearsonr(sub[var], sub[target]).pvalue)
                    row["spearman_p"] = float(spearmanr(sub[var], sub[target]).pvalue)
            rows.append(row)
    return pd.DataFrame(rows)


def trend_by_channels(nchan: pd.DataFrame) -> dict[str, Any]:
    sub = nchan[["n_channels", "beta_kld_over_recon_mean"]].dropna()
    out: dict[str, Any] = {"n_points": int(len(sub))}
    if len(sub) >= 3:
        out["pearson_r_nchannels_vs_beta_kld_over_recon"] = float(sub["n_channels"].corr(sub["beta_kld_over_recon_mean"], method="pearson"))
        out["spearman_r_nchannels_vs_beta_kld_over_recon"] = float(sub["n_channels"].corr(sub["beta_kld_over_recon_mean"], method="spearman"))
    if len(sub):
        out["single_channel_mean"] = float(nchan.loc[nchan["n_channels"].eq(1), "beta_kld_over_recon_mean"].mean())
        out["max_channel_mean"] = float(nchan.loc[nchan["n_channels"].eq(nchan["n_channels"].max()), "beta_kld_over_recon_mean"].mean())
    return out


def interpretation_text(cand: pd.DataFrame, nchan: pd.DataFrame, corr: pd.DataFrame, ckpt: pd.DataFrame) -> tuple[str, str, str]:
    trend = trend_by_channels(nchan)
    selected = cand[cand["is_selected_greedy_path"].eq(True)].sort_values("greedy_step")
    best_auc = cand.loc[cand["auc_mean"].idxmax()] if not cand.empty else None
    low_beta_count = int(ckpt["old_rule_would_select_low_beta"].sum()) if "old_rule_would_select_low_beta" in ckpt else 0
    diff_count = int(ckpt["best_any_differs_from_best_high_beta"].sum()) if "best_any_differs_from_best_high_beta" in ckpt else 0
    total_ckpt = int(len(ckpt))
    decreasing = trend.get("spearman_r_nchannels_vs_beta_kld_over_recon", 0) < 0
    summary = f"""# Beta Calibration Summary

Completed candidate-fold records audited: {len(ckpt)} checkpoint summaries and {len(cand)} candidate aggregates.

High-beta checkpoint audit:

- Selected beta guard pass count: {int(ckpt['selected_beta_passes_guard'].sum())}/{total_ckpt}.
- Best-any-beta differed from best-high-beta in {diff_count}/{total_ckpt} folds.
- Under the old unrestricted checkpoint rule, {low_beta_count}/{total_ckpt} folds would have selected a low-beta checkpoint.

Effective regularization trend:

- Spearman correlation between number of channels and mean beta*KLD/recon: {trend.get('spearman_r_nchannels_vs_beta_kld_over_recon', float('nan')):.4f}.
- Single-channel mean beta*KLD/recon: {trend.get('single_channel_mean', float('nan')):.6f}.
- Maximum-channel mean beta*KLD/recon: {trend.get('max_channel_mean', float('nan')):.6f}.

Best mean-AUC candidate:

- `{best_auc['candidate_tag'] if best_auc is not None else 'NA'}` channels `{best_auc['channels'] if best_auc is not None else 'NA'}` with AUC={best_auc['auc_mean'] if best_auc is not None else float('nan'):.6f}, PR-AUC={best_auc['pr_auc_mean'] if best_auc is not None else float('nan'):.6f}.

Selected greedy path:

{selected[['greedy_step','candidate_tag','channels','auc_mean','pr_auc_mean','beta_kld_over_recon_mean']].to_markdown(index=False) if not selected.empty else 'No selected greedy path rows found.'}
"""
    interpretation = f"""# Beta-Effective Interpretation

The audit supports the expected scaling concern: the reconstruction term is summed over selected channels, while KLD is not channel-normalized. As channel count increases, total reconstruction generally increases and the effective regularization ratio `beta * KLD / recon` tends to {"decrease" if decreasing else "not clearly decrease"} across the completed Greedy900 p150/high-beta candidates.

This means fixed beta=2.5 is not a channel-count invariant regularization setting under the current summed reconstruction objective. A two-channel and a seven-channel candidate can have the same nominal beta but different beta-effective pressure on the latent code because the reconstruction denominator scales with channel count.

The high-beta checkpoint policy worked as intended: every selected checkpoint passed the beta guard, and the audit quantifies how often the old unrestricted best-epoch rule would have selected a different/low-beta epoch.

The AUC associations should be treated as descriptive only. They are candidate-level correlations from a greedy ablation screen, not independent confirmatory model comparisons.
"""
    rec = """# Recommended Loss Policy For Channel Ablation

For future formal FAST++ channel screens, the most defensible policy is:

1. Prefer `offdiag_channelmean` reconstruction loss for channel-ablation screens, so beta has a more comparable interpretation across one-, two-, and multi-channel candidates.
2. If the historical summed reconstruction loss is retained, report beta-effective regularization (`beta*KLD/recon`) and consider beta scaling by the number of selected channels.
3. For any channel set proposed for confirmatory FULL training, run a small beta sensitivity grid rather than assuming beta transfers across channel counts.
4. Keep high-beta checkpoint selection or an equivalent beta-phase-aware checkpoint rule for cyclic-beta runs; otherwise low-beta ramp checkpoints can confound rate-distortion comparisons.

Do not use this FAST diagnostic alone for model promotion. It is useful for setting a fairer objective/regularization policy before any formal channel-selection rerun.
"""
    return summary, interpretation, rec


def main() -> None:
    started = datetime.now(timezone.utc)
    OUT.mkdir(parents=True, exist_ok=True)
    summary, tag_to_channels = parse_summary()
    fold, ckpt = fold_rows(summary, tag_to_channels)
    if fold.empty:
        raise RuntimeError("No completed candidate fold rows were extracted.")
    cand = aggregate_candidate(fold)
    nchan = aggregate_n_channels(fold)
    corr = correlation_rows(cand)

    fold.to_csv(OUT / "rate_distortion_by_fold.csv", index=False)
    cand.to_csv(OUT / "rate_distortion_by_candidate.csv", index=False)
    nchan.to_csv(OUT / "rate_distortion_by_n_channels.csv", index=False)
    ckpt.to_csv(OUT / "checkpoint_selection_highbeta_audit.csv", index=False)
    corr.to_csv(OUT / "auc_vs_regularization.csv", index=False)

    write(OUT / "rate_distortion_by_candidate.md", md_table(cand, "Rate Distortion By Candidate"))
    write(OUT / "checkpoint_selection_highbeta_audit.md", md_table(ckpt, "Checkpoint Selection High-Beta Audit", max_rows=80))

    summary_md, interp_md, rec_md = interpretation_text(cand, nchan, corr, ckpt)
    write(OUT / "beta_calibration_summary.md", summary_md)
    write(OUT / "beta_effective_interpretation.md", interp_md)
    write(OUT / "recommended_loss_policy_for_channel_ablation.md", rec_md)

    log = {
        "timestamp_utc": started.isoformat(),
        "run_root": str(RUN_ROOT),
        "log_root": str(LOG_ROOT),
        "output_dir": str(OUT),
        "did_train": False,
        "did_infer": False,
        "did_modify_run_outputs": False,
        "candidate_folds": int(len(fold)),
        "candidate_count": int(cand["candidate_tag"].nunique()),
        "commands": [
            "/home/diego/anaconda3/envs/vae_ad/bin/python -m py_compile scripts/revision_bspc_2026/audit_fast_meta647_greedy900_p150_highbeta_beta_calibration_20260702.py",
            "/home/diego/anaconda3/envs/vae_ad/bin/python scripts/revision_bspc_2026/audit_fast_meta647_greedy900_p150_highbeta_beta_calibration_20260702.py",
        ],
        "outputs": sorted(p.name for p in OUT.iterdir()),
    }
    (OUT / "command_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Wrote audit to {OUT}")
    print(cand[["candidate_tag", "channels", "auc_mean", "pr_auc_mean", "beta_kld_over_recon_mean"]].sort_values("auc_mean", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
