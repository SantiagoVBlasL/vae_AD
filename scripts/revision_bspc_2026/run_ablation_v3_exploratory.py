#!/usr/bin/env python3
"""
Exploratory channel-ablation wrapper for ADNI_expanded_v3.

Calls the existing greedy-forward ablation engine (scripts/ablation_canales.py →
scripts/run_vae_clf_ad_ablation.py) with v3 paths and inference-matched
hyperparameters.  Does NOT modify the engine scripts.

IMPORTANT — READ BEFORE INTERPRETING RESULTS
---------------------------------------------
This is a SCREENING study, not a confirmatory one.

  * The greedy selection criterion is outer-fold test ROC-AUC.  Because the
    same test folds are used both to score candidate subsets and to estimate
    performance, the ablation AUC is optimistically biased.

  * The channel ranking is useful for hypothesis generation, not for claiming
    a final AUC in a paper.

  * Any claimed performance must come from an INDEPENDENT run of
    run_vae_clf_ad_inference.py (or run_adni_expanded_retraining.py) with the
    channel subset PRE-SPECIFIED before seeing v3 data.

  * The pre-specified confirmatory subset from the historical ablation is
    channels [1, 0, 2] (Pearson_FisherZ, Pearson_OMST, MI_KNN).

Usage
-----
# Validate inputs and print command (no run):
    python scripts/revision_bspc_2026/run_ablation_v3_exploratory.py --dry-run

# Smoke-test: only pre-specified channels [1, 0, 2]:
    python scripts/revision_bspc_2026/run_ablation_v3_exploratory.py --only-original --dry-run

# Launch full ablation (all 7 channels, long):
    python scripts/revision_bspc_2026/run_ablation_v3_exploratory.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Paths ─────────────────────────────────────────────────────────────────────
TENSOR_PATH = (
    PROJECT_ROOT
    / "data"
    / "revision_bspc_2026"
    / "adni_expanded_v3_all_available"
    / "GLOBAL_TENSOR_ADNI_expanded_v3_all_available.npz"
)
METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "revision_bspc_2026"
    / "adni_expanded_v3_all_available"
    / "subject_metadata_adni_expanded_v3_all_available.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "results" / "revision_bspc_2026" / "ablation_v3_exploratory"
ABLATION_SCRIPT = PROJECT_ROOT / "scripts" / "ablation_canales.py"

# ── Channel registry ──────────────────────────────────────────────────────────
CHANNEL_NAMES: Dict[int, str] = {
    0: "Pearson_OMST_GCE_Signed_Weighted",
    1: "Pearson_Full_FisherZ_Signed",
    2: "MI_KNN_Symmetric",
    3: "dFC_AbsDiffMean",
    4: "dFC_StdDev",
    5: "DistanceCorr",
    6: "Granger_F_lag1",
}
ALL_CHANNELS = list(CHANNEL_NAMES)          # [0, 1, 2, 3, 4, 5, 6]
CONFIRMATORY_CHANNELS = [1, 0, 2]           # pre-specified from historical ablation

README_TEXT = """\
# Ablation v3 — Exploratory Channel Screening

## Purpose

Greedy-forward channel selection on ADNI_expanded_v3 to screen whether the
historically selected channels [1, 0, 2] remain optimal under the expanded
cohort (508 subjects: historical + martin59 + santiago_siemens + santiago_ge).

## IMPORTANT: Exploratory, not confirmatory

| Property | This run | Confirmatory requirement |
|---|---|---|
| Channel selection criterion | Outer-fold test AUC | Pre-specified before data |
| Performance estimate | Optimistically biased | Independent run with fixed channels |
| Use in paper | Channel ranking only | Requires separate inference run |

The AUC numbers produced here should NOT be reported as the final model
performance.  They are used only to decide whether to re-confirm [1, 0, 2]
or to propose a revised subset for a pre-registered confirmatory run.

## Greedy selection bias

Each candidate channel subset is scored using the average ROC-AUC across the
outer test folds.  Because the same test observations are used for both
selection and scoring, the selected subset's ablation-AUC is an upper bound
on the true generalisation performance.

## Recommended workflow

1. Run this script (exploratory).
2. Inspect `summary_ablation.csv` and the ablation curve figure.
3. If [1, 0, 2] is still optimal or parsimonious (1-SE rule), confirm it with:
       python scripts/revision_bspc_2026/run_adni_expanded_retraining.py \\
         --config configs/runs/adni_expanded_v3_beta25_static3.json
4. If a different subset is suggested, pre-register it and run a fresh
   independent evaluation.

## Run settings

- outer_folds: 5 (matches v3 inference run)
- beta_vae: 2.5 (matches v3 inference run)
- latent_dim: 256 (matches v3 inference run)
- epochs_vae: 500, early_stop: 40 (reduced for compute; documents ablation tradeoff)
- Classifier: LogisticRegression, no HP tuning (ablation invariant)
- Metric: ROC-AUC

## Files

- ablation_runs/           : per-subset VAE run outputs
- summary_ablation.csv     : step-by-step greedy results
- command.txt              : exact command used to launch the ablation engine
- run_manifest.json        : all paths, parameters, and timestamps
- README.md                : this file
"""


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Exploratory channel-ablation for ADNI_expanded_v3. "
            "Calls ablation_canales.py without modifying it."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tensor-path", type=Path, default=TENSOR_PATH,
                   help="Global tensor NPZ.")
    p.add_argument("--metadata-path", type=Path, default=METADATA_PATH,
                   help="Subject metadata CSV.")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT,
                   help="Root output directory for all ablation sub-runs.")
    p.add_argument("--ablation-script", type=Path, default=ABLATION_SCRIPT,
                   help="Path to ablation_canales.py.")
    p.add_argument(
        "--candidate-channels", type=int, nargs="+", default=None,
        metavar="IDX",
        help="Override candidate channel indices (0-based). Default: all 7.",
    )
    p.add_argument(
        "--only-original", action="store_true",
        help=(
            "Smoke-test mode: restrict candidates to the historical subset "
            "[1, 0, 2] only.  Useful to verify the pipeline runs on v3 before "
            "committing to the full 7-channel ablation."
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs, print command, write manifest, then exit without running.",
    )
    p.add_argument(
        "--python-executable", type=str, default=None,
        help="Python interpreter to use. Defaults to the current interpreter.",
    )
    # Compute-control overrides (documented defaults match the run settings above)
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--epochs-vae", type=int, default=500)
    p.add_argument("--early-stop", type=int, default=40)
    p.add_argument("--beta-vae", type=float, default=2.5)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--dropout-vae", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr-sched-type", type=str, default="cosine_warm",
                   choices=["cosine_warm", "plateau"])
    p.add_argument("--lr-sched-t0", type=int, default=80)
    p.add_argument("--lr-sched-eta-min", type=float, default=5e-7)
    p.add_argument("--beta-cycles", type=int, default=32,
                   help="cyclical_beta_n_cycles passed to the ablation engine.")
    p.add_argument("--norm-mode", type=str, default="zscore_offdiag",
                   choices=["zscore_offdiag", "minmax_offdiag"])
    p.add_argument("--vae-val-split-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metric", type=str, default="auc",
                   help="Greedy selection metric (column in all_folds_metrics_*.csv).")
    p.add_argument("--min-improvement", type=float, default=0.001)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_git_hash() -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
            cwd=str(PROJECT_ROOT), timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def validate_inputs(tensor_path: Path, metadata_path: Path) -> None:
    errors: List[str] = []
    if not tensor_path.exists():
        errors.append(f"Tensor not found: {tensor_path}")
    if not metadata_path.exists():
        errors.append(f"Metadata not found: {metadata_path}")
    if not ABLATION_SCRIPT.exists():
        errors.append(f"ablation_canales.py not found: {ABLATION_SCRIPT}")
    if errors:
        raise FileNotFoundError("\n".join(errors))


def resolve_candidate_channels(args: argparse.Namespace) -> List[int]:
    if args.only_original:
        return list(CONFIRMATORY_CHANNELS)
    if args.candidate_channels is not None:
        return list(args.candidate_channels)
    return list(ALL_CHANNELS)


def build_command(args: argparse.Namespace, channels: List[int]) -> List[str]:
    python = args.python_executable or sys.executable
    cmd = [
        python,
        str(args.ablation_script),
        "--global_tensor_path", str(args.tensor_path),
        "--metadata_path",      str(args.metadata_path),
        "--output_root",        str(args.output_root / "ablation_runs"),
        # Channel selection
        "--candidate_channels",
    ] + [str(c) for c in channels] + [
        # Ablation control
        "--metric",             args.metric,
        "--min_improvement",    str(args.min_improvement),
        # Outer CV
        "--outer_folds",        str(args.outer_folds),
        "--repeats",            str(args.repeats),
        # VAE hyperparameters (match v3 inference run)
        "--epochs_vae",         str(args.epochs_vae),
        "--early_stop",         str(args.early_stop),
        "--beta_vae",           str(args.beta_vae),
        "--latent_dim",         str(args.latent_dim),
        "--dropout_vae",        str(args.dropout_vae),
        "--batch_size",         str(args.batch_size),
        "--beta_cycles",        str(args.beta_cycles),
        "--lr_sched_type",      args.lr_sched_type,
        "--lr_sched_T0",        str(args.lr_sched_t0),
        "--lr_sched_eta_min",   str(args.lr_sched_eta_min),
        "--norm_mode",          args.norm_mode,
        "--vae_val_split_ratio", str(args.vae_val_split_ratio),
        "--num_workers",        str(args.num_workers),
        "--seed",               str(args.seed),
        # Parity flags
        "--classifier_use_class_weight",
    ]
    return cmd


def save_manifest(
    output_root: Path,
    cmd: List[str],
    args: argparse.Namespace,
    channels: List[int],
    dry_run: bool,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "git_hash": get_git_hash(),
        "study_type": "exploratory_channel_ablation",
        "confirmatory": False,
        "confirmatory_note": (
            "Results are for channel ranking only. "
            "Final performance must be evaluated in an independent run "
            "with channels pre-specified before seeing v3 data."
        ),
        "only_original_mode": args.only_original,
        "paths": {
            "tensor": str(args.tensor_path.resolve()),
            "metadata": str(args.metadata_path.resolve()),
            "output_root": str(output_root.resolve()),
            "ablation_script": str(args.ablation_script.resolve()),
        },
        "candidate_channels": channels,
        "channel_names": {str(i): CHANNEL_NAMES.get(i, f"Unknown_{i}") for i in channels},
        "confirmatory_channels": CONFIRMATORY_CHANNELS,
        "parameters": {
            "metric": args.metric,
            "outer_folds": args.outer_folds,
            "repeats": args.repeats,
            "epochs_vae": args.epochs_vae,
            "early_stop": args.early_stop,
            "beta_vae": args.beta_vae,
            "latent_dim": args.latent_dim,
            "dropout_vae": args.dropout_vae,
            "batch_size": args.batch_size,
            "beta_cycles": args.beta_cycles,
            "lr_sched_type": args.lr_sched_type,
            "lr_sched_T0": args.lr_sched_t0,
            "lr_sched_eta_min": args.lr_sched_eta_min,
            "norm_mode": args.norm_mode,
            "vae_val_split_ratio": args.vae_val_split_ratio,
            "seed": args.seed,
            "classifier_use_class_weight": True,
            "min_improvement": args.min_improvement,
        },
        "command": cmd,
        "command_shell": " ".join(
            f'"{t}"' if " " in t else t for t in cmd
        ),
    }
    manifest_path = output_root / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest_path


def save_command_txt(output_root: Path, cmd: List[str]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    txt_path = output_root / "command.txt"
    txt_path.write_text(" \\\n  ".join(cmd) + "\n", encoding="utf-8")
    return txt_path


def write_readme(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "README.md").write_text(README_TEXT, encoding="utf-8")


def stream_subprocess(cmd: List[str], log_path: Path) -> int:
    """Run cmd with real-time stdout+stderr, also tee to log_path."""
    print(f"\n[stream] Log: {log_path}")
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return process.wait()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    channels = resolve_candidate_channels(args)

    # ── Validate inputs ────────────────────────────────────────────────────────
    print("Validating inputs …")
    validate_inputs(args.tensor_path, args.metadata_path)
    print(f"  Tensor:   {args.tensor_path}")
    print(f"  Metadata: {args.metadata_path}")

    # ── Build command ──────────────────────────────────────────────────────────
    cmd = build_command(args, channels)

    channel_labels = [
        f"  {i}: {CHANNEL_NAMES.get(i, '?')}" for i in channels
    ]
    mode = "only-original smoke-test" if args.only_original else "full 7-channel"
    print(f"\nMode: {mode}")
    print(f"Candidate channels ({len(channels)}):")
    print("\n".join(channel_labels))
    print(f"\nKey parameters:")
    print(f"  outer_folds={args.outer_folds}, beta_vae={args.beta_vae}, "
          f"latent_dim={args.latent_dim}, epochs_vae={args.epochs_vae}, "
          f"early_stop={args.early_stop}")
    print(f"\nCommand:")
    print("  " + " \\\n    ".join(cmd))

    # ── Write artefacts ────────────────────────────────────────────────────────
    output_root = args.output_root
    manifest_path = save_manifest(output_root, cmd, args, channels, args.dry_run)
    cmd_path = save_command_txt(output_root, cmd)
    write_readme(output_root)

    print(f"\nManifest: {manifest_path}")
    print(f"Command:  {cmd_path}")
    print(f"README:   {output_root / 'README.md'}")

    if args.dry_run:
        print("\nDry-run complete. Nothing was executed.")
        return 0

    # ── Launch ablation ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LAUNCHING EXPLORATORY ABLATION")
    print("=" * 60)
    print("WARNING: AUC figures produced here are SCREENING estimates only.")
    print("         Do not report them as final model performance.")
    print("=" * 60 + "\n")

    log_path = output_root / "ablation_stdout.log"
    rc = stream_subprocess(cmd, log_path)

    if rc == 0:
        summary_csv = output_root / "ablation_runs" / "summary_ablation.csv"
        print(f"\nAblation finished (rc=0).")
        if summary_csv.exists():
            print(f"Summary: {summary_csv}")
        else:
            # The summary is written into output_root by ablation_canales.py;
            # location may vary — search for it.
            candidates = list(output_root.rglob("summary_ablation.csv"))
            if candidates:
                print(f"Summary: {candidates[0]}")
    else:
        print(f"\nAblation FAILED with exit code {rc}. Check {log_path}.")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
