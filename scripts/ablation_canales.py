#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablation_canales.py
-------------------
Greedy forward channel-selection orchestrator for run_vae_clf_ad_ablation.py.

Strategy
--------
1) Evaluate EVERY candidate channel individually (single-channel runs).
2) Seed with the best single channel.
3) Greedily add the candidate that most improves the target metric.
4) Stop when improvement < min_improvement (or when all channels are used).

Parity flags (always passed to the ablation script)
----------------------------------------------------
  --vae_final_activation tanh
  --metadata_features Age Sex       (parity with inference)
  (LogReg + no-tune + no-SMOTE are ablation-script invariants — not CLI flags)

Run tags (short, deterministic)
--------------------------------
  single:  single_ch{c}
  greedy:  step{n}_add{c}_k{K}      (K = total channels after adding c)

Caching
-------
A run is reused iff:
  - its output directory contains all_folds_metrics_MULTI_*.csv
  - AND that CSV has an 'auc' column
  - AND has at least one row where actual_classifier_type == 'logreg'
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Channel name registry (for pretty printing only)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CHANNEL_NAMES: List[str] = [
    "Pearson_OMST_GCE_Signed_Weighted",
    "Pearson_Full_FisherZ_Signed",
    "MI_KNN_Symmetric",
    "dFC_AbsDiffMean",
    "dFC_StdDev",
    "DistanceCorr",
    "Granger_F_lag1",
]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _pretty_channels(indices: List[int], channel_names: List[str]) -> str:
    parts = []
    for i in indices:
        name = channel_names[i] if i < len(channel_names) else f"Channel{i}"
        parts.append(f"{i}:{name}")
    return "[" + ", ".join(parts) + "]"


def _run_cmd(cmd: List[str], cwd: Optional[str] = None) -> int:
    """Execute *cmd* as a subprocess and return its exit code."""
    print("\n[RUN]", " ".join(map(str, cmd)))
    proc = subprocess.run(cmd, cwd=cwd)
    return proc.returncode


def _find_metrics_csv(out_dir: Path) -> Optional[Path]:
    """Return the most-recently modified all_folds_metrics_MULTI_*.csv in *out_dir*."""
    cands = sorted(
        out_dir.glob("all_folds_metrics_MULTI_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def _csv_is_valid(csv_path: Path, metric: str = "auc") -> bool:
    """
    Return True iff csv_path exists, has an 'auc' column and at least one row
    with actual_classifier_type == 'logreg'.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False
    if metric not in df.columns:
        return False
    if "actual_classifier_type" in df.columns:
        if not (df["actual_classifier_type"] == "logreg").any():
            return False
    return True


def _read_metric(
    csv_path: Path, metric: str = "auc"
) -> Tuple[float, float]:
    """
    Return (mean, std) of *metric* across all logreg rows in *csv_path*.
    Returns (nan, nan) on any failure.
    """
    try:
        df = pd.read_csv(csv_path)
        if "actual_classifier_type" in df.columns:
            df = df[df["actual_classifier_type"] == "logreg"]
        if df.empty or metric not in df.columns:
            return float("nan"), float("nan")
        return float(df[metric].mean()), float(df[metric].std())
    except Exception:
        return float("nan"), float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Single-run runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_ablation_once(
    channels: List[int],
    run_tag: str,
    run_root: Path,
    cfg: Dict,
    opt_metric: str,
) -> Dict:
    """
    Launch one ablation run for *channels*, save to run_root/run_tag/.

    Parameters
    ----------
    channels    : channel indices passed to --channels_to_use
    run_tag     : short unique identifier for this run
    run_root    : parent directory for all ablation runs
    cfg         : flat dict with all CLI parameters for the ablation script
    opt_metric  : CSV column to read as ranking metric (default "auc")

    Returns
    -------
    dict with keys: ok, out_dir, metrics_csv, metric_name, metric_mean,
                    metric_std, channels (list)
    """
    if not channels:
        raise ValueError("channels must not be empty.")

    out_dir = run_root / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Check cache ────────────────────────────────────────────────────────────
    cached_csv = _find_metrics_csv(out_dir)
    if cached_csv is not None and _csv_is_valid(cached_csv, opt_metric):
        mean_val, std_val = _read_metric(cached_csv, opt_metric)
        print(
            f"[CACHE HIT] {run_tag}: {opt_metric}={mean_val:.4f} ±{std_val:.4f} "
            f"— skipping re-run."
        )
        return {
            "ok":          True,
            "cached":      True,
            "out_dir":     str(out_dir),
            "metrics_csv": str(cached_csv),
            "metric_name": opt_metric,
            "metric_mean": mean_val,
            "metric_std":  std_val,
            "channels":    list(channels),
        }

    # ── Build CLI command ──────────────────────────────────────────────────────
    cmd: List[str] = [
        sys.executable, cfg["ablation_script_path"],
        "--global_tensor_path", cfg["global_tensor_path"],
        "--metadata_path",      cfg["metadata_path"],
        "--output_dir",         str(out_dir),

        # ── Parity flags (always) ────────────────────────────────────────────
        "--vae_final_activation", "tanh",
        "--metadata_features", "Age", "Sex",

        # ── VAE hyperparameters ──────────────────────────────────────────────
        "--outer_folds",                     str(cfg.get("outer_folds", 5)),
        "--repeated_outer_folds_n_repeats",  str(cfg.get("repeats", 1)),
        "--epochs_vae",                      str(cfg.get("epochs_vae", 300)),
        "--early_stopping_patience_vae",     str(cfg.get("early_stop", 20)),
        "--cyclical_beta_n_cycles",          str(cfg.get("beta_cycles", 4)),
        "--cyclical_beta_ratio_increase",    str(cfg.get("cyclical_beta_ratio", 0.4)),
        "--lr_scheduler_type",               str(cfg.get("lr_sched_type", "cosine_warm")),
        "--lr_scheduler_T0",                 str(cfg.get("lr_sched_T0", 30)),
        "--lr_scheduler_eta_min",            str(cfg.get("lr_sched_eta_min", 5e-7)),
        "--batch_size",                      str(cfg.get("batch_size", 64)),
        "--beta_vae",                        str(cfg.get("beta_vae", 4.6)),
        "--dropout_rate_vae",                str(cfg.get("dropout_vae", 0.2)),
        "--latent_dim",                      str(cfg.get("latent_dim", 128)),
        "--num_workers",                     str(cfg.get("num_workers", 4)),
        "--norm_mode",                       str(cfg.get("norm_mode", "zscore_offdiag")),
        "--seed",                            str(cfg.get("seed", 42)),
        "--vae_val_split_ratio",             str(cfg.get("vae_val_split_ratio", 0.2)),

        # ── Channel subset ────────────────────────────────────────────────────
        "--channels_to_use",
    ] + [str(i) for i in channels]

    # Optional boolean flags
    if cfg.get("classifier_use_class_weight", False):
        cmd.append("--classifier_use_class_weight")
    if cfg.get("save_fold_artefacts", False):
        cmd.append("--save_fold_artefacts")
    if cfg.get("save_vae_training_history", False):
        cmd.append("--save_vae_training_history")

    # ── Execute ────────────────────────────────────────────────────────────────
    rc = _run_cmd(cmd)
    if rc != 0:
        return {
            "ok":    False,
            "error": f"ablation script exited with code {rc}",
            "out_dir": str(out_dir),
        }

    metrics_csv = _find_metrics_csv(out_dir)
    if metrics_csv is None or not _csv_is_valid(metrics_csv, opt_metric):
        return {
            "ok":    False,
            "error": "No valid metrics CSV found after run.",
            "out_dir": str(out_dir),
        }

    mean_val, std_val = _read_metric(metrics_csv, opt_metric)
    return {
        "ok":          True,
        "cached":      False,
        "out_dir":     str(out_dir),
        "metrics_csv": str(metrics_csv),
        "metric_name": opt_metric,
        "metric_mean": mean_val,
        "metric_std":  std_val,
        "channels":    list(channels),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Greedy ablation algorithm
# ─────────────────────────────────────────────────────────────────────────────

def greedy_ablation(
    candidate_channels: List[int],
    run_root: Path,
    cfg: Dict,
    opt_metric: str,
    min_improvement: float,
    stop_on_no_improve: bool,
    channel_names: List[str],
) -> List[Dict]:
    """
    Greedy forward channel selection.

    Returns ordered list of selected run results (one entry per step).
    """
    results: List[Dict] = []

    # ── Step 0: evaluate all single-channel runs ──────────────────────────────
    print("\n========== PHASE 1: Single-channel evaluation ==========")
    singles: List[Dict] = []
    for ch in candidate_channels:
        tag = f"single_ch{ch}"
        res = _run_ablation_once([ch], tag, run_root, cfg, opt_metric)
        if res.get("ok"):
            m = res["metric_mean"]
            s = res["metric_std"]
            print(
                f"[SINGLE] ch={_pretty_channels([ch], channel_names)} → "
                f"{opt_metric}={m:.4f} ±{s:.4f}"
            )
            singles.append(res)
        else:
            print(f"[WARN] Single ch={ch} failed: {res.get('error')}")

    if not singles:
        print("[ERROR] No single-channel runs succeeded. Aborting.")
        return results

    # Best single: highest metric, ties broken by fewest channels
    best_single = max(singles, key=lambda r: (r["metric_mean"], -len(r["channels"])))
    picked = list(best_single["channels"])
    results.append(best_single)
    remaining = [c for c in candidate_channels if c not in picked]

    print(
        f"\n[SEED] Best single: {_pretty_channels(picked, channel_names)} → "
        f"{opt_metric}={best_single['metric_mean']:.4f}"
    )

    # ── Greedy steps ──────────────────────────────────────────────────────────
    print("\n========== PHASE 2: Greedy forward selection ==========")
    step = 1
    while remaining:
        print(f"\n[STEP {step}] Current set ({len(picked)} channels): "
              f"{_pretty_channels(picked, channel_names)}")
        trials: List[Dict] = []
        for cand in remaining:
            trial_ch = picked + [cand]
            K   = len(trial_ch)
            tag = f"step{step}_add{cand}_k{K}"
            res = _run_ablation_once(trial_ch, tag, run_root, cfg, opt_metric)
            if res.get("ok"):
                m = res["metric_mean"]
                print(
                    f"  + add ch={cand} ({channel_names[cand] if cand < len(channel_names) else '?'}) "
                    f"→ {opt_metric}={m:.4f}"
                )
                trials.append(res)
            else:
                print(f"  ! ch={cand} failed: {res.get('error')}")

        if not trials:
            print("[INFO] No valid trials in this step. Stopping.")
            break

        best_trial = max(trials, key=lambda r: r["metric_mean"])
        gain       = best_trial["metric_mean"] - results[-1]["metric_mean"]
        new_ch     = [c for c in best_trial["channels"] if c not in picked]
        print(
            f"[CHOICE] add {_pretty_channels(new_ch, channel_names)} → "
            f"{opt_metric}={best_trial['metric_mean']:.4f} (Δ={gain:+.4f})"
        )

        results.append(best_trial)
        picked    = list(best_trial["channels"])
        remaining = [c for c in candidate_channels if c not in picked]

        if stop_on_no_improve and gain < min_improvement:
            print(
                f"[STOP] Improvement {gain:.4f} < threshold {min_improvement:.4f}. "
                "Stopping early."
            )
            break

        step += 1

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Greedy channel-ablation orchestrator for run_vae_clf_ad_ablation.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Paths ──────────────────────────────────────────────────────────────────
    p.add_argument(
        "--ablation_script_path", type=str,
        default="scripts/run_vae_clf_ad_ablation.py",
        help="Path to run_vae_clf_ad_ablation.py (relative to CWD or absolute).",
    )
    p.add_argument("--global_tensor_path", type=str, required=True)
    p.add_argument("--metadata_path",      type=str, required=True)
    p.add_argument(
        "--output_root", type=str, default="./ablation_runs",
        help="Root directory for all ablation sub-runs.",
    )

    # ── Channel selection ──────────────────────────────────────────────────────
    p.add_argument(
        "--candidate_channels", type=int, nargs="*", default=None,
        help="Candidate channel indices. Defaults to all channels in the tensor.",
    )

    # ── Ablation control ───────────────────────────────────────────────────────
    p.add_argument(
        "--metric", type=str, default="auc",
        help="CSV column to optimise (e.g. auc, pr_auc, balanced_accuracy).",
    )
    p.add_argument(
        "--min_improvement", type=float, default=0.001,
        help="Minimum gain to continue greedy selection.",
    )
    p.add_argument(
        "--no_early_stop", action="store_true",
        help="Do not stop on no-improvement; run until all channels are added.",
    )

    # ── VAE hyper-parameters ───────────────────────────────────────────────────
    p.add_argument("--outer_folds",            type=int,   default=5)
    p.add_argument("--repeats",                type=int,   default=1)
    p.add_argument("--epochs_vae",             type=int,   default=300)
    p.add_argument("--early_stop",             type=int,   default=20)
    p.add_argument("--beta_cycles",            type=int,   default=4)
    p.add_argument("--cyclical_beta_ratio",    type=float, default=0.4)
    p.add_argument("--lr_sched_type",          type=str,   default="cosine_warm",
                   choices=["plateau", "cosine_warm"])
    p.add_argument("--lr_sched_T0",            type=int,   default=30)
    p.add_argument("--lr_sched_eta_min",       type=float, default=5e-7)
    p.add_argument("--batch_size",             type=int,   default=64)
    p.add_argument("--beta_vae",               type=float, default=4.6)
    p.add_argument("--dropout_vae",            type=float, default=0.2)
    p.add_argument("--latent_dim",             type=int,   default=128)
    p.add_argument("--norm_mode",              type=str,   default="zscore_offdiag",
                   choices=["zscore_offdiag", "minmax_offdiag"])
    p.add_argument("--vae_val_split_ratio",    type=float, default=0.2)
    p.add_argument("--num_workers",            type=int,   default=4)
    p.add_argument("--seed",                   type=int,   default=42)

    # ── Optional flags ─────────────────────────────────────────────────────────
    p.add_argument("--classifier_use_class_weight", action="store_true",
                   help="Pass --classifier_use_class_weight to ablation script.")
    p.add_argument("--save_fold_artefacts",    action="store_true",
                   help="Pass --save_fold_artefacts to ablation script.")
    p.add_argument("--save_vae_training_history", action="store_true",
                   help="Pass --save_vae_training_history to ablation script.")

    args = p.parse_args()

    run_root = Path(args.output_root)
    run_root.mkdir(parents=True, exist_ok=True)

    # ── Discover channel count from tensor if not provided ─────────────────────
    if args.candidate_channels is None:
        npz = np.load(args.global_tensor_path)
        key = "global_tensor_data" if "global_tensor_data" in npz else list(npz.keys())[0]
        _, C, _, _ = npz[key].shape
        candidate_channels = list(range(C))
        print(f"[INFO] Discovered {C} channels from tensor key '{key}'.")
    else:
        candidate_channels = list(args.candidate_channels)

    # Build channel names list (best-effort, for printing)
    npz_check = np.load(args.global_tensor_path)
    if "channel_names" in npz_check:
        channel_names = list(npz_check["channel_names"])
    else:
        channel_names = list(DEFAULT_CHANNEL_NAMES)
    # Pad if needed
    if len(channel_names) < max(candidate_channels, default=0) + 1:
        channel_names += [f"Channel{i}" for i in range(len(channel_names), max(candidate_channels) + 1)]

    # ── Build cfg dict ─────────────────────────────────────────────────────────
    cfg: Dict = {
        "ablation_script_path":        args.ablation_script_path,
        "global_tensor_path":          args.global_tensor_path,
        "metadata_path":               args.metadata_path,
        "outer_folds":                 args.outer_folds,
        "repeats":                     args.repeats,
        "epochs_vae":                  args.epochs_vae,
        "early_stop":                  args.early_stop,
        "beta_cycles":                 args.beta_cycles,
        "cyclical_beta_ratio":         args.cyclical_beta_ratio,
        "lr_sched_type":               args.lr_sched_type,
        "lr_sched_T0":                 args.lr_sched_T0,
        "lr_sched_eta_min":            args.lr_sched_eta_min,
        "batch_size":                  args.batch_size,
        "beta_vae":                    args.beta_vae,
        "dropout_vae":                 args.dropout_vae,
        "latent_dim":                  args.latent_dim,
        "norm_mode":                   args.norm_mode,
        "vae_val_split_ratio":         args.vae_val_split_ratio,
        "num_workers":                 args.num_workers,
        "seed":                        args.seed,
        "classifier_use_class_weight": args.classifier_use_class_weight,
        "save_fold_artefacts":         args.save_fold_artefacts,
        "save_vae_training_history":   args.save_vae_training_history,
    }

    print("\n========== Channel Ablation ==========")
    print(f"Ablation script: {args.ablation_script_path}")
    print(f"Candidate channels ({len(candidate_channels)}): "
          f"{_pretty_channels(candidate_channels, channel_names)}")
    print(f"Optimise metric:  {args.metric}")
    print(f"Min improvement:  {args.min_improvement}")
    print(f"Output root:      {run_root.resolve()}")
    print(f"[INVARIANT] Classifier: logreg (no HP tuning, no SMOTE)")
    print(f"[PARITY]    --vae_final_activation tanh  --metadata_features Age Sex")
    print("======================================\n")

    t0 = time.time()
    results = greedy_ablation(
        candidate_channels=candidate_channels,
        run_root=run_root,
        cfg=cfg,
        opt_metric=args.metric,
        min_improvement=args.min_improvement,
        stop_on_no_improve=not args.no_early_stop,
        channel_names=channel_names,
    )
    elapsed = time.time() - t0

    if not results:
        print("\n[FIN] No results to summarise.")
        return

    # ── Save summary CSV ───────────────────────────────────────────────────────
    summary_csv_path = run_root / "summary_ablation.csv"
    rows = []
    prev_metric = None
    for i, res in enumerate(results):
        delta = (
            "" if prev_metric is None
            else f"{(res['metric_mean'] - prev_metric):+.6f}"
        )
        rows.append({
            "step":             i,
            "n_channels":       len(res["channels"]),
            "channels_indices": " ".join(map(str, res["channels"])),
            "channels_pretty":  _pretty_channels(res["channels"], channel_names),
            "metric":           res["metric_name"],
            "metric_mean":      f"{res['metric_mean']:.6f}",
            "metric_std":       f"{res['metric_std']:.6f}",
            "delta_vs_prev":    delta,
            "cached":           res.get("cached", False),
            "out_dir":          res["out_dir"],
            "metrics_csv":      res.get("metrics_csv", ""),
        })
        prev_metric = res["metric_mean"]

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n[OK] Summary CSV: {summary_csv_path.resolve()}")

    # ── Save summary JSON (B5) ─────────────────────────────────────────────────
    summary_json_path = run_root / "summary_ablation.json"
    best_result = max(results, key=lambda r: r["metric_mean"])
    summary_json: Dict = {
        "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds":   round(elapsed, 1),
        "opt_metric":        args.metric,
        "min_improvement":   args.min_improvement,
        "n_candidate_channels": len(candidate_channels),
        "candidate_channels": candidate_channels,
        "config": {
            k: v for k, v in cfg.items()
            if k not in ("ablation_script_path", "global_tensor_path", "metadata_path")
        },
        "parity_flags": {
            "vae_final_activation": "tanh",
            "metadata_features":    ["Age", "Sex"],
            "classifier":           "logreg",
            "clf_no_tune":          True,
            "use_smote":            False,
        },
        "best": {
            "channels":     best_result["channels"],
            "channel_names": [
                channel_names[c] if c < len(channel_names) else f"Channel{c}"
                for c in best_result["channels"]
            ],
            "metric_mean":  best_result["metric_mean"],
            "metric_std":   best_result["metric_std"],
            "out_dir":      best_result["out_dir"],
        },
        "greedy_path": [
            {
                "step":        i,
                "channels":    r["channels"],
                "metric_mean": r["metric_mean"],
                "metric_std":  r["metric_std"],
                "delta":       (
                    None if i == 0
                    else round(r["metric_mean"] - results[i - 1]["metric_mean"], 6)
                ),
            }
            for i, r in enumerate(results)
        ],
    }
    try:
        with open(summary_json_path, "w") as fh:
            json.dump(summary_json, fh, indent=2, default=str)
        print(f"[OK] Summary JSON: {summary_json_path.resolve()}")
    except Exception as exc:
        print(f"[WARN] Could not save summary JSON: {exc}")

    # ── Print table ────────────────────────────────────────────────────────────
    print("\n--- Greedy Selection Path ---")
    with pd.option_context("display.max_colwidth", 80):
        print(
            summary_df[
                ["step", "n_channels", "channels_pretty", "metric", "metric_mean",
                 "metric_std", "delta_vs_prev"]
            ].to_string(index=False)
        )

    print(f"\n[BEST] {args.metric}={best_result['metric_mean']:.4f} ±{best_result['metric_std']:.4f}")
    print(f"       Channels: {_pretty_channels(best_result['channels'], channel_names)}")
    print(f"\nTotal wall time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
