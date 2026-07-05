#!/usr/bin/env python3
"""
Full SHAP + IG interpretability run on the final promoted model.

Pipeline:
  Stage 1: Latent-space SHAP for all 5 folds (LinearExplainer, Age/Sex frozen)
  Stage 2: IG backprojection for all 5 folds (50 steps, cn_median_train baseline)
  Stage 3: Aggregation — fold consensus, sign-consistency, network summaries

Read-only with respect to model artifacts, tensors, and metadata.
All heavy outputs go to /media/diego/Datos/.../full/.

Prerequisites:
  1. Smoketest PASSED.
  2. GPU free (run_loso_cv.py not running).
  3. /media/diego/Datos has ≥ 1 GB free.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("full_shap_ig")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── Paths ─────────────────────────────────────────────────────────────────────
PROMOTED_RUN = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
)
TENSOR_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors"
    "/GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
METADATA_PATH = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv"
)
ROI_ANNOTATION_PATH = PROMOTED_RUN / "roi_info_from_tensor.csv"
INTERP_SCRIPT = PROJECT_ROOT / "scripts/run_interpretability.py"

HEAVY_OUT = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/final_model_shap_ig_20260624/full"
)
LIGHT_OUT = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/final_model_shap_ig_preflight_20260624/full_outputs"
)

# ── Model params ───────────────────────────────────────────────────────────────
CHANNELS = [1, 0, 2]
CHANNEL_NAMES = ["Pearson_Full_FisherZ_Signed", "Pearson_OMST_GCE_Signed_Weighted", "MI_KNN_Symmetric"]
LATENT_DIM = 384
DROPOUT = 0.15
NUM_CONV = 4
DECODER_TYPE = "convtranspose"
INTER_FC = "quarter"
FINAL_ACT = "tanh"
META_FEATS = ["Age", "Sex"]
SEED = 42

FOLDS = [1, 2, 3, 4, 5]
IG_N_STEPS = 50
BG_SAMPLE_SIZE = 100
TOP_K_LATENT = 50
SIGN_CONSISTENCY_THRESHOLD = 0.6

PYTHON = sys.executable


def _check_guards() -> None:
    """Abort if conflicting processes are running."""
    import shutil
    pgrep = shutil.which("pgrep")
    if pgrep:
        for proc_name in ["run_loso_cv.py", "run_vae_clf_ad.py", "run_vae_clf_ad_inference.py"]:
            result = subprocess.run(
                [pgrep, "-af", proc_name], capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                raise RuntimeError(
                    f"Conflicting process detected: {proc_name}\n"
                    f"  {result.stdout.strip()}\n"
                    "Abort. Wait for LOSO training to complete before running SHAP/IG."
                )

    # Disk guard
    import shutil as sh
    usage = sh.disk_usage("/media/diego/Datos")
    free_gb = usage.free / 1e9
    if free_gb < 1.0:
        raise RuntimeError(f"/media/diego/Datos has only {free_gb:.1f} GB free. Need ≥ 1 GB.")

    # Resume guard
    full_marker = HEAVY_OUT / ".full_run_complete"
    if full_marker.exists():
        log.warning(
            "Full run marker found: .full_run_complete. "
            "Pass --resume to overwrite (not yet implemented — delete the marker manually)."
        )
        raise RuntimeError(
            "Full run already completed. Delete .full_run_complete to re-run."
        )


def common_args(fold: int) -> list[str]:
    return [
        "--run_dir", str(PROMOTED_RUN),
        "--fold", str(fold),
        "--clf", "logreg",
        "--global_tensor_path", str(TENSOR_PATH),
        "--metadata_path", str(METADATA_PATH),
        "--channels_to_use", *[str(c) for c in CHANNELS],
        "--latent_dim", str(LATENT_DIM),
        "--latent_features_type", "mu",
        "--metadata_features", *META_FEATS,
        "--num_conv_layers_encoder", str(NUM_CONV),
        "--decoder_type", DECODER_TYPE,
        "--dropout_rate_vae", str(DROPOUT),
        "--intermediate_fc_dim_vae", INTER_FC,
        "--vae_final_activation", FINAL_ACT,
        "--seed", str(SEED),
    ]


def run_shap_fold(fold: int) -> None:
    cmd = [PYTHON, str(INTERP_SCRIPT), "shap"] + common_args(fold) + [
        "--bg_mode", "train",
        "--bg_sample_size", str(BG_SAMPLE_SIZE),
        "--bg_seed", str(SEED),
        "--freeze_meta", "Age", "Sex",
        "--freeze_strategy", "train_stats",
        "--shap_link", "identity",
        "--shap_tag", "final",
    ]
    log.info(f"[fold {fold}] Stage 1: SHAP")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    log.info(f"[fold {fold}] SHAP done in {elapsed:.1f}s")
    if result.returncode != 0:
        raise RuntimeError(f"SHAP fold {fold} FAILED (returncode={result.returncode})")


def run_ig_fold(fold: int) -> None:
    cmd = [PYTHON, str(INTERP_SCRIPT), "saliency"] + common_args(fold) + [
        "--roi_annotation_path", str(ROI_ANNOTATION_PATH),
        "--saliency_method", "integrated_gradients",
        "--ig_n_steps", str(IG_N_STEPS),
        "--ig_baseline", "cn_median_train",
        "--top_k", str(TOP_K_LATENT),
        "--shap_weight_mode", "ad_vs_cn_diff",
        "--shap_tag", "final",
    ]
    log.info(f"[fold {fold}] Stage 2: IG backprojection")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    log.info(f"[fold {fold}] IG done in {elapsed:.1f}s")
    if result.returncode != 0:
        raise RuntimeError(f"IG fold {fold} FAILED (returncode={result.returncode})")


def aggregate_shap(fold_dirs: list[Path]) -> dict:
    """Aggregate SHAP values across folds and return consensus latent dims."""
    import joblib as _joblib
    all_shap = []
    for fd in fold_dirs:
        # interpret_fold.py saves a joblib pack, not a raw npy
        pack_path = fd / "interpretability_shap" / "shap_pack_logreg_final.joblib"
        if not pack_path.exists():
            log.warning(f"No SHAP pack found: {pack_path}")
            continue
        pack = _joblib.load(pack_path)
        sv = pack.get("shap_values")
        if sv is None:
            log.warning(f"Key 'shap_values' missing in {pack_path}")
            continue
        all_shap.append(np.array(sv))
    if not all_shap:
        log.error("No SHAP arrays found for aggregation.")
        return {}

    shap_all = np.concatenate(all_shap, axis=0)  # (397, 386)
    log.info(f"  Aggregated SHAP shape: {shap_all.shape}")

    mean_abs = np.abs(shap_all).mean(axis=0)
    mean_signed = shap_all.mean(axis=0)
    latent_mean_abs = mean_abs[:LATENT_DIM]
    ranking = np.argsort(latent_mean_abs)[::-1]
    top_k_dims = ranking[:TOP_K_LATENT].tolist()

    # Fold-by-fold Spearman on top-50
    from scipy.stats import spearmanr
    fold_rankings = []
    for arr in all_shap:
        fa = np.abs(arr).mean(axis=0)[:LATENT_DIM]
        fold_rankings.append(np.argsort(fa)[::-1].tolist())

    spearmans = []
    for i in range(len(fold_rankings)):
        for j in range(i + 1, len(fold_rankings)):
            r, _ = spearmanr(fold_rankings[i][:TOP_K_LATENT], fold_rankings[j][:TOP_K_LATENT])
            spearmans.append(float(r))
    mean_spearman = float(np.mean(spearmans)) if spearmans else 0.0

    out = {
        "n_subjects": int(shap_all.shape[0]),
        "n_features": int(shap_all.shape[1]),
        "top_k_latent_dims": top_k_dims,
        "mean_fold_spearman_rank_r": round(mean_spearman, 4),
        "shap_all_shape": list(shap_all.shape),
    }
    log.info(f"  Top-5 latent dims: {top_k_dims[:5]}")
    log.info(f"  Mean fold Spearman r (top-{TOP_K_LATENT}): {mean_spearman:.4f}")

    agg_dir = HEAVY_OUT / "shap_aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    np.save(agg_dir / "shap_all_subjects.npy", shap_all)
    (agg_dir / "consensus_latent_dims.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    ranking_df = pd.DataFrame({
        "latent_dim": np.arange(LATENT_DIM),
        "mean_abs_shap": latent_mean_abs,
        "mean_signed_shap": mean_signed[:LATENT_DIM],
        "rank": np.argsort(np.argsort(-latent_mean_abs)),
    })
    ranking_df.to_csv(agg_dir / "shap_latent_ranking.csv", index=False)
    return out


def aggregate_ig(fold_dirs: list[Path], n_rois: int = 131, n_channels: int = 3) -> None:
    """Aggregate IG maps across folds and compute cross-fold sign consistency.

    interpret_fold.py saliency averages attributions over subjects within each fold,
    saving a per-fold mean map of shape (C, R, R).  We stack these fold-mean maps
    and compute cross-fold sign consistency.
    """
    # Expected filename from interpret_fold.py with method=integrated_gradients, top_k=TOP_K_LATENT
    map_fname = f"saliency_map_diff_signed_integrated_gradients_top{TOP_K_LATENT}.npy"
    fold_maps = []
    for fd in fold_dirs:
        map_path = fd / f"interpretability_logreg" / map_fname
        if map_path.exists():
            fold_maps.append(np.load(map_path))  # each (C, R, R)
        else:
            log.warning(f"No IG diff map found: {map_path}")

    if not fold_maps:
        log.warning("No IG maps found for aggregation.")
        return

    # Stack across folds: (n_folds, C, R, R)
    ig_stack = np.stack(fold_maps, axis=0)
    log.info(f"  IG stack shape (n_folds, C, R, R): {ig_stack.shape}")

    mean_ig = ig_stack.mean(axis=0)      # (C, R, R)
    abs_mean_ig = np.abs(ig_stack).mean(axis=0)  # (C, R, R)

    # Sign consistency: fraction of folds where sign matches the pooled mean sign
    sign_pool = np.sign(mean_ig)
    sign_agree = (np.sign(ig_stack) == sign_pool[np.newaxis]).astype(float)
    sign_consistency = sign_agree.mean(axis=0)  # (C, R, R)

    agg_dir = HEAVY_OUT / "ig_aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    np.save(agg_dir / "mean_ig_all_subjects.npy", mean_ig)
    np.save(agg_dir / "abs_mean_ig_all_subjects.npy", abs_mean_ig)
    np.save(agg_dir / "sign_consistency.npy", sign_consistency)

    # Consensus edges: top 10% by abs_mean AND sign_consistency >= threshold
    threshold_abs = np.percentile(abs_mean_ig, 90)
    consensus_mask = (abs_mean_ig >= threshold_abs) & (sign_consistency >= SIGN_CONSISTENCY_THRESHOLD)

    roi_df = pd.read_csv(ROI_ANNOTATION_PATH)
    rows = []
    for c_idx, ch_name in enumerate(CHANNEL_NAMES):
        for i in range(n_rois):
            for j in range(n_rois):
                if i >= j:
                    continue
                if not consensus_mask[c_idx, i, j]:
                    continue
                rows.append({
                    "channel": ch_name,
                    "roi_i": roi_df["roi_name_in_tensor"].iloc[i] if i < len(roi_df) else f"ROI_{i}",
                    "roi_j": roi_df["roi_name_in_tensor"].iloc[j] if j < len(roi_df) else f"ROI_{j}",
                    "network_i": roi_df["network_label_in_tensor"].iloc[i] if i < len(roi_df) else "",
                    "network_j": roi_df["network_label_in_tensor"].iloc[j] if j < len(roi_df) else "",
                    "mean_ig": float(mean_ig[c_idx, i, j]),
                    "abs_mean_ig": float(abs_mean_ig[c_idx, i, j]),
                    "sign_consistency": float(sign_consistency[c_idx, i, j]),
                })
    consensus_df = pd.DataFrame(rows).sort_values("abs_mean_ig", ascending=False)
    consensus_df.to_csv(agg_dir / "consensus_edges.csv", index=False)
    log.info(f"  Consensus edges: {len(consensus_df)}")

    # Network-pair summary
    if rows:
        nw_pairs = (
            consensus_df.groupby(["network_i", "network_j", "channel"])
            .agg(n_edges=("abs_mean_ig", "count"), mean_abs_ig=("abs_mean_ig", "mean"))
            .reset_index()
            .sort_values("mean_abs_ig", ascending=False)
        )
        nw_pairs.to_csv(agg_dir / "network_pair_summary.csv", index=False)


def main() -> None:
    t_start = time.time()
    log.info("=" * 70)
    log.info("FULL SHAP + IG Run — final model, 5 folds, 50 IG steps")
    log.info("=" * 70)

    _check_guards()

    if not TENSOR_PATH.exists():
        raise FileNotFoundError(f"Tensor not found: {TENSOR_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    HEAVY_OUT.mkdir(parents=True, exist_ok=True)
    LIGHT_OUT.mkdir(parents=True, exist_ok=True)

    # Stage 1: SHAP for all folds
    for fold in FOLDS:
        run_shap_fold(fold)

    # Stage 1 aggregation
    fold_dirs = [PROMOTED_RUN / f"fold_{f}" for f in FOLDS]
    shap_summary = aggregate_shap(fold_dirs)

    # Stage 2: IG for all folds
    for fold in FOLDS:
        run_ig_fold(fold)

    # Stage 2 aggregation
    aggregate_ig(fold_dirs)

    # Write completion marker
    elapsed = time.time() - t_start
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "folds": FOLDS,
        "ig_steps": IG_N_STEPS,
        "top_k_latent": TOP_K_LATENT,
        "shap_summary": shap_summary,
        "status": "COMPLETE",
    }
    out_json = HEAVY_OUT / "full_run_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (HEAVY_OUT / ".full_run_complete").touch()

    # Copy light outputs to repo
    for f in (HEAVY_OUT / "shap_aggregated").glob("*.csv"):
        (LIGHT_OUT / f.name).write_bytes(f.read_bytes())
    for f in (HEAVY_OUT / "ig_aggregated").glob("*.csv"):
        (LIGHT_OUT / f.name).write_bytes(f.read_bytes())
    out_json_light = LIGHT_OUT / "full_run_summary.json"
    out_json_light.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info(f"Full run COMPLETE in {elapsed/60:.1f} min.")
    log.info(f"Heavy outputs: {HEAVY_OUT}")
    log.info(f"Light outputs: {LIGHT_OUT}")


if __name__ == "__main__":
    main()
