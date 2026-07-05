#!/usr/bin/env python3
"""
Downstream-only latent sweep for ADNI v5 DPARSF-10000 no-Python-bandpass [4,1,0].

PURPOSE
-------
The VAE models from the ch4_1_0_baseline run are already trained (5 folds).
This script encodes test subjects through the saved VAE models and sweeps over
downstream classifier configurations WITHOUT retraining the VAE. This isolates:
  - How much the AUC depends on the classifier hyperparameter space.
  - Whether removing Age/Sex metadata features recovers or worsens performance.
  - Whether raw connectivity features (no VAE) explain fold-level variance.

SWEEP CONFIGURATIONS
--------------------
Each sweep variant re-runs the downstream classifier step for each fold:

  cfg_a : logreg + svm, WITH Age+Sex (replicate baseline)
  cfg_b : logreg + svm, WITHOUT Age+Sex (latent-only)
  cfg_c : logreg + svm, WITH Age+Sex, increased trials (n=500)
  cfg_d : logreg only, WITH Age+Sex, exclude fold 4 from mean (diagnostic)

NOTE: cfg_d does not re-run fold 4; it computes mean AUC excluding fold 4 from
already-collected results to quantify fold 4's drag.

STATUS: DRY-RUN ONLY. Do not call with --execute until VAE encoding logic
        is reviewed and approved.

Usage:
    # Dry-run (default): show plan, check VAE model paths, no computation
    python run_adni_v5_dparsf10000_no_pybandpass_latent_sweep.py --dry-run

    # When approved for execution (requires --execute flag, not yet enabled):
    # python run_adni_v5_dparsf10000_no_pybandpass_latent_sweep.py --execute
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Source run to read VAE models and fold splits from
_SOURCE_RUN_DIR = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026"
    "/adni_v5_dparsf10000_no_pybandpass_ch4_1_0_baseline"
)
_TENSOR_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_dparsf10000_no_pybandpass/subject_tensors"
    "/GLOBAL_TENSOR_ADNI_expanded_v5_dparsf10000_no_pybandpass.npz"
)
_METADATA_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_dparsf10000_no_pybandpass"
    "/training_ready_metadata_v5_dparsf10000_no_pybandpass.csv"
)
_OUTPUT_ROOT_REPO = (
    PROJECT_ROOT / "results" / "revision_bspc_2026"
    / "adni_v5_dparsf10000_no_pybandpass_ch4_1_0_latent_sweep"
)
_OUTPUT_ROOT_BIG_DISK = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026"
    "/adni_v5_dparsf10000_no_pybandpass_ch4_1_0_latent_sweep"
)
_N_FOLDS = 5
_CHANNELS_TO_USE = [4, 1, 0]

# Sweep configuration definitions
SWEEP_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "cfg_a_baseline_replicate",
        "description": "Replicate baseline: logreg+svm with Age+Sex metadata",
        "classifiers": ["logreg", "svm"],
        "metadata_features": ["Age", "Sex"],
        "n_iter": 300,
        "note": "Should match baseline AUC if VAE encoding is consistent.",
    },
    {
        "name": "cfg_b_latent_only",
        "description": "Latent-only: logreg+svm WITHOUT Age+Sex metadata",
        "classifiers": ["logreg", "svm"],
        "metadata_features": [],
        "n_iter": 300,
        "note": "Isolates pure latent representation quality.",
    },
    {
        "name": "cfg_c_more_trials",
        "description": "Extended search: logreg+svm with Age+Sex, n_iter=500",
        "classifiers": ["logreg", "svm"],
        "metadata_features": ["Age", "Sex"],
        "n_iter": 500,
        "note": "Tests if 300 trials was the binding constraint.",
    },
    {
        "name": "cfg_d_fold4_analysis",
        "description": (
            "Fold 4 diagnostic: compute mean AUC excluding fold 4 from baseline results. "
            "No retraining — reads existing baseline metrics CSV."
        ),
        "classifiers": ["logreg", "svm"],
        "metadata_features": ["Age", "Sex"],
        "n_iter": 0,
        "note": (
            "Diagnostic only. Uses already-collected fold metrics. "
            "AUC_excl_fold4 shows whether fold 4 is driving the low mean."
        ),
        "analysis_only": True,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Show plan and check paths without running any computation (default: True).",
    )
    p.add_argument(
        "--execute", action="store_true",
        help="Actually run the sweep. NOT YET ENABLED — pending review.",
    )
    return p.parse_args()


def check_vae_models(n_folds: int = _N_FOLDS) -> List[Dict[str, Any]]:
    rows = []
    for fold in range(1, n_folds + 1):
        vae_path = _SOURCE_RUN_DIR / f"fold_{fold}" / f"vae_model_fold_{fold}.pt"
        norm_path = _SOURCE_RUN_DIR / f"fold_{fold}" / "vae_norm_params.joblib"
        test_idx_path = _SOURCE_RUN_DIR / f"fold_{fold}" / "test_tensor_idx.npy"
        train_idx_path = _SOURCE_RUN_DIR / f"fold_{fold}" / "train_dev_tensor_idx.npy"
        rows.append({
            "fold": fold,
            "vae_model": str(vae_path),
            "vae_model_exists": vae_path.exists(),
            "norm_params_exists": norm_path.exists(),
            "test_idx_exists": test_idx_path.exists(),
            "train_idx_exists": train_idx_path.exists(),
            "all_ok": all([
                vae_path.exists(), norm_path.exists(),
                test_idx_path.exists(), train_idx_path.exists()
            ]),
        })
    return rows


def cfg_d_analysis_only() -> Dict[str, Any]:
    """Read existing baseline metrics and compute AUC excluding fold 4."""
    metrics_glob = list(_SOURCE_RUN_DIR.glob("all_folds_metrics_MULTI*.csv"))
    if not metrics_glob:
        return {"error": "baseline metrics CSV not found"}
    mf = pd.read_csv(metrics_glob[0])
    results = {}
    for clf in mf["actual_classifier_type"].unique():
        sub = mf[mf["actual_classifier_type"] == clf]
        results[clf] = {
            "auc_raw_all_folds": round(float(sub["auc_raw"].mean()), 4),
            "auc_raw_excl_fold4": round(float(sub[sub["fold"] != 4]["auc_raw"].mean()), 4),
            "auc_raw_fold4_only": round(float(sub[sub["fold"] == 4]["auc_raw"].iloc[0]), 4),
            "fold4_drag": round(
                float(sub["auc_raw"].mean()) -
                float(sub[sub["fold"] != 4]["auc_raw"].mean()), 4
            ),
        }
    return results


def main() -> int:
    args = parse_args()

    if args.execute:
        print(
            "ERROR: --execute is not yet enabled for this sweep script.\n"
            "The VAE encoding and downstream classifier loop requires review before running.\n"
            "Use --dry-run to see the plan and verify VAE artifact availability.",
            file=sys.stderr,
        )
        return 1

    print("=== ADNI v5 [4,1,0] Downstream Latent Sweep — DRY-RUN ===")
    print()
    print(f"Source run    : {_SOURCE_RUN_DIR.name}")
    print(f"Tensor        : {_TENSOR_PATH}")
    print(f"Metadata      : {_METADATA_PATH}")
    print(f"Channels      : {_CHANNELS_TO_USE} = dFC_StdDev, Pearson_FisherZ, OMST")
    print(f"N folds       : {_N_FOLDS}")
    print()

    # --- Check source artifacts ---
    print("Checking VAE model artifacts:")
    model_checks = check_vae_models()
    all_models_ok = True
    for row in model_checks:
        status = "OK" if row["all_ok"] else "MISSING"
        print(f"  Fold {row['fold']}: [{status}]  VAE={row['vae_model_exists']}  "
              f"norm={row['norm_params_exists']}  test_idx={row['test_idx_exists']}")
        if not row["all_ok"]:
            all_models_ok = False
    print()

    # --- Check inputs ---
    print("Checking input paths:")
    for label, p in [("tensor", _TENSOR_PATH), ("metadata", _METADATA_PATH),
                      ("source_run_dir", _SOURCE_RUN_DIR)]:
        status = "OK" if p.exists() else "NOT FOUND"
        print(f"  [{status}] {label}: {p}")
    print()

    # --- cfg_d analysis (analysis-only, no VAE needed) ---
    print("cfg_d (fold-4 drag analysis) — runs immediately (reads existing results):")
    d_results = cfg_d_analysis_only()
    if "error" in d_results:
        print(f"  ERROR: {d_results['error']}")
    else:
        for clf, stats in d_results.items():
            print(f"  {clf}:")
            print(f"    AUC all folds       : {stats['auc_raw_all_folds']}")
            print(f"    AUC excluding fold 4: {stats['auc_raw_excl_fold4']}")
            print(f"    AUC fold 4 only     : {stats['auc_raw_fold4_only']}")
            print(f"    Fold 4 drag on mean : {stats['fold4_drag']:.4f}")
    print()

    # --- Sweep plan ---
    print("Planned sweep configurations (NOT executing):")
    for cfg in SWEEP_CONFIGS:
        tag = "[ANALYSIS ONLY]" if cfg.get("analysis_only") else "[VAE ENCODE + CLASSIFY]"
        print(f"  {cfg['name']} {tag}")
        print(f"    {cfg['description']}")
        print(f"    classifiers={cfg['classifiers']}, "
              f"metadata_features={cfg['metadata_features']}, n_iter={cfg['n_iter']}")
        print(f"    Note: {cfg['note']}")
        print()

    # --- Output plan ---
    print("Output directory (when executed):")
    print(f"  Repo symlink : {_OUTPUT_ROOT_REPO}")
    print(f"  Big disk     : {_OUTPUT_ROOT_BIG_DISK}")
    print()

    # --- Prerequisites for execution ---
    print("Prerequisites before running --execute:")
    print("  1. Review VAE encoding logic in run_vae_clf_ad_inference.py to extract")
    print("     latent mu vectors from saved vae_model_fold_N.pt.")
    print("  2. Confirm that fold normalization params (vae_norm_params.joblib) apply")
    print("     consistently to test subjects when using the saved VAE.")
    print("  3. Prepare output symlink:")
    print(f"     mkdir -p {_OUTPUT_ROOT_BIG_DISK}")
    print(f"     ln -s {_OUTPUT_ROOT_BIG_DISK} {_OUTPUT_ROOT_REPO}")
    print()

    verdict = "READY (all VAE models found)" if all_models_ok else "NOT READY (some models missing)"
    print(f"Sweep readiness: {verdict}")
    print()
    print("Dry-run complete. Use --execute when approved to launch the sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
