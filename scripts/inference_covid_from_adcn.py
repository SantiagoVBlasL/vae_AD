#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/inference_covid_from_adcn.py  (v2 — paper-grade)

Transfer inference: apply trained AD/CN β-VAE + classifier pipeline(s) to
a new (COVID) cohort.

Paper-grade implementation with:
  - Optional metadata (graceful degradation when unavailable)
  - Robust classifier resolution (calibrated → final → raw)
  - OOD diagnostics: reconstruction error + latent Mahalanobis distance
  - Consensus-edge "AD signature" scoring (if interpretability outputs exist)
  - run_manifest.json capturing full provenance
  - Smoke-test mode (--smoke_test)

Workflow per fold:
  1. Load fold artifacts (VAE weights, norm params, classifier pipeline,
     feature_columns.json, metadata_imputation.json).
  2. Select same channels from the new tensor.
  3. Apply saved normalisation parameters (NO recomputation).
  4. Forward pass through VAE → extract μ (latent mean) AND reconstruction.
  5. Build feature DataFrame  =  [latent μ]  ⊕  [metadata imputed/encoded].
  6. Predict with each saved classifier pipeline.
  7. Compute per-subject reconstruction error (off-diagonal).
  8. Encode AD train-dev pool and fit Gaussian → Mahalanobis distance for COVID.

Ensemble across K folds → final predictions, recon errors, distances.

Usage:
    python scripts/inference_covid_from_adcn.py \
        --training_output_dir results/vae_3channels_beta65_pro \
        --classifier_types logreg

Smoke test:
    python scripts/inference_covid_from_adcn.py --smoke_test
"""
from __future__ import annotations

# ── Bootstrap: add src/ to sys.path ────────────────────────────────────
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
else:
    raise FileNotFoundError(f"'src/' not found at: {SRC_DIR}")
# ── End bootstrap ──────────────────────────────────────────────────────

import argparse
import hashlib
import json
import logging
import subprocess
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import mahalanobis

from betavae_xai.models import ConvolutionalVAE
from betavae_xai.utils.logging import setup_logging

logger = setup_logging(__name__)
warnings.filterwarnings("ignore")

SEED_GLOBAL = 42
SEED_SAMPLING = 42


# ═══════════════════════════════════════════════════════════════════════
#  JSON / I-O helpers
# ═══════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_json_dump(obj: Dict[str, Any], out_path: Path) -> None:
    """Dump JSON with robust numpy/Path serialisation."""
    def _default(o):
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True,
                  default=_default)


def _sha256_list(items) -> str:
    """Deterministic SHA-256 of a list of strings."""
    h = hashlib.sha256()
    for s in items:
        h.update(str(s).encode("utf-8"))
    return h.hexdigest()[:16]


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "N/A"


# ═══════════════════════════════════════════════════════════════════════
#  Normalisation (mirrors training exactly)
# ═══════════════════════════════════════════════════════════════════════

def apply_normalization_params(
    data_tensor_subset: np.ndarray,
    norm_params_per_channel_list: List[Dict[str, float]],
) -> np.ndarray:
    """Apply previously-computed normalisation parameters to a tensor."""
    num_subjects, num_channels, num_rois, _ = data_tensor_subset.shape
    normalized = data_tensor_subset.copy()
    off_diag_mask = ~np.eye(num_rois, dtype=bool)

    if len(norm_params_per_channel_list) != num_channels:
        raise ValueError(
            f"Channel count mismatch: data has {num_channels}, "
            f"params for {len(norm_params_per_channel_list)}"
        )

    for c_idx in range(num_channels):
        params = norm_params_per_channel_list[c_idx]
        mode = params.get("mode", "zscore_offdiag")
        if params.get("no_scale", False):
            continue
        ch = data_tensor_subset[:, c_idx, :, :]
        scaled = ch.copy()
        if off_diag_mask.any():
            if mode == "zscore_offdiag":
                if params["std"] > 1e-9:
                    scaled[:, off_diag_mask] = (
                        (ch[:, off_diag_mask] - params["mean"]) / params["std"]
                    )
            elif mode == "minmax_offdiag":
                range_val = params.get("max", 1.0) - params.get("min", 0.0)
                if range_val > 1e-9:
                    scaled[:, off_diag_mask] = (
                        (ch[:, off_diag_mask] - params["min"]) / range_val
                    )
                else:
                    scaled[:, off_diag_mask] = 0.0
        normalized[:, c_idx, :, :] = scaled
        normalized[:, c_idx, ~off_diag_mask] = 0.0
    return normalized


# ═══════════════════════════════════════════════════════════════════════
#  Metadata handling (graceful degradation)
# ═══════════════════════════════════════════════════════════════════════

def _prepare_metadata_features(
    covid_metadata_df: Optional[pd.DataFrame],
    metadata_features: List[str],
    imputation_info: Dict[str, Any],
    n_subjects: int,
) -> pd.DataFrame:
    """
    Prepare metadata columns for the classifier.

    If covid_metadata_df is None, creates a fully-imputed DataFrame
    using training-fold imputation values.
    """
    if covid_metadata_df is not None and all(
        c in covid_metadata_df.columns for c in metadata_features
    ):
        meta_df = covid_metadata_df[metadata_features].copy()
    else:
        if covid_metadata_df is not None:
            logger.warning(
                "Metadata provided but missing columns: "
                f"{set(metadata_features) - set(covid_metadata_df.columns)}. "
                "Falling back to full imputation."
            )
        meta_df = pd.DataFrame(index=range(n_subjects))
        for col in metadata_features:
            meta_df[col] = np.nan

    imputation_values = imputation_info.get("imputation_values", {})
    for col in metadata_features:
        if col not in meta_df.columns:
            meta_df[col] = np.nan
        if meta_df[col].isna().any():
            fill_val = imputation_values.get(col)
            if fill_val is not None:
                n_missing = meta_df[col].isna().sum()
                logger.info(
                    f"    Imputing {n_missing} NaN(s) in '{col}' "
                    f"with training value: {fill_val}"
                )
                meta_df[col] = meta_df[col].fillna(fill_val)
            else:
                logger.warning(
                    f"    No imputation value for '{col}'; filling with 0."
                )
                meta_df[col] = meta_df[col].fillna(0)
    return meta_df


# ═══════════════════════════════════════════════════════════════════════
#  Classifier helpers
# ═══════════════════════════════════════════════════════════════════════

def _resolve_classifier_path(
    fold_dir: Path, clf_type: str, fold_idx: int
) -> Optional[Path]:
    """
    Robust classifier resolution (priority order):
      1. classifier_{clf_type}_calibrated_pipeline_fold_{k}.joblib
      2. classifier_{clf_type}_final_pipeline_fold_{k}.joblib
      3. classifier_{clf_type}_raw_pipeline_fold_{k}.joblib
      4. classifier_{clf_type}_pipeline_fold_{k}.joblib  (legacy)
    """
    candidates = [
        fold_dir / f"classifier_{clf_type}_calibrated_pipeline_fold_{fold_idx}.joblib",
        fold_dir / f"classifier_{clf_type}_final_pipeline_fold_{fold_idx}.joblib",
        fold_dir / f"classifier_{clf_type}_raw_pipeline_fold_{fold_idx}.joblib",
        fold_dir / f"classifier_{clf_type}_pipeline_fold_{fold_idx}.joblib",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _get_classes(est):
    if hasattr(est, "classes_"):
        return est.classes_
    if hasattr(est, "named_steps"):
        last = list(est.named_steps.values())[-1]
        if hasattr(last, "classes_"):
            return last.classes_
    return None


def _get_score_1d(estimator, X) -> np.ndarray:
    """Return 1D scores (P(AD)) robustly."""
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)
        classes = _get_classes(estimator)
        if classes is not None:
            classes_list = list(classes)
            for pos_label in [1, "AD", 1.0]:
                if pos_label in classes_list:
                    return np.asarray(p)[:, classes_list.index(pos_label)].ravel()
            return np.asarray(p)[:, -1].ravel()
        return np.asarray(p)[:, 1].ravel() if p.ndim == 2 and p.shape[1] >= 2 \
            else np.asarray(p).ravel()
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X)).ravel()
    return np.asarray(estimator.predict(X)).astype(float).ravel()


# ═══════════════════════════════════════════════════════════════════════
#  Reconstruction error
# ═══════════════════════════════════════════════════════════════════════

def compute_recon_error_offdiag(
    X_true: np.ndarray, X_recon: np.ndarray
) -> np.ndarray:
    """
    Per-subject mean absolute off-diagonal reconstruction error,
    averaged across channels.  Shape: (N,C,R,R) → (N,).
    """
    n_rois = X_true.shape[-1]
    off_diag = ~np.eye(n_rois, dtype=bool)
    abs_diff = np.abs(X_true - X_recon)
    per_subj_per_ch = abs_diff[:, :, off_diag].mean(axis=-1)  # (N, C)
    return per_subj_per_ch.mean(axis=1)  # (N,)


# ═══════════════════════════════════════════════════════════════════════
#  Latent distance (Mahalanobis from train-dev Gaussian baseline)
# ═══════════════════════════════════════════════════════════════════════

def compute_mahalanobis_distances(
    covid_latent: np.ndarray,
    baseline_latent: np.ndarray,
    regularisation: float = 1e-6,
) -> np.ndarray:
    """
    Mahalanobis distance of each COVID subject from a Gaussian
    fitted to the baseline (AD train-dev pool) latent space.
    """
    mu_base = baseline_latent.mean(axis=0)
    cov_base = np.cov(baseline_latent, rowvar=False)
    cov_base += regularisation * np.eye(cov_base.shape[0])
    try:
        cov_inv = np.linalg.inv(cov_base)
    except np.linalg.LinAlgError:
        logger.warning("Covariance singular; using pseudo-inverse.")
        cov_inv = np.linalg.pinv(cov_base)

    distances = np.array([
        mahalanobis(z, mu_base, cov_inv) for z in covid_latent
    ])
    return distances


# ═══════════════════════════════════════════════════════════════════════
#  Tensor compatibility validation
# ═══════════════════════════════════════════════════════════════════════

def validate_tensor_compatibility(
    covid_npz, run_config: Dict[str, Any],
) -> None:
    """
    Strict compatibility: shape, ROI/channel hash checks.
    Raises ValueError on critical mismatch.
    """
    covid_tensor = covid_npz["global_tensor_data"]
    covid_ch_names = list(covid_npz["channel_names"])

    # Shape
    train_shape = run_config.get("tensor_shape", [])
    if len(train_shape) >= 3 and covid_tensor.shape[2] != train_shape[2]:
        raise ValueError(
            f"ROI dim mismatch: training={train_shape[2]}, COVID={covid_tensor.shape[2]}"
        )
    if covid_tensor.shape[2] != covid_tensor.shape[3]:
        raise ValueError(
            f"Non-square: {covid_tensor.shape[2]}x{covid_tensor.shape[3]}"
        )

    # Channel name hash
    master_ch = run_config.get("channel_names_master_in_tensor_order", [])
    if master_ch:
        h_train = _sha256_list(master_ch)
        h_covid = _sha256_list(covid_ch_names)
        if h_train != h_covid:
            raise ValueError(
                f"Channel name MISMATCH: train={h_train}, covid={h_covid}"
            )

    # Channel index bounds
    channels_to_use = run_config.get("channels_to_use_indices",
                                     run_config.get("args", {}).get("channels_to_use"))
    if channels_to_use and max(channels_to_use) >= covid_tensor.shape[1]:
        raise ValueError(
            f"Training used channel idx {max(channels_to_use)}, "
            f"COVID has {covid_tensor.shape[1]} channels."
        )

    logger.info(
        f"Tensor compatibility OK: shape={covid_tensor.shape}, "
        f"channels_to_use={channels_to_use}"
    )


# ═══════════════════════════════════════════════════════════════════════
#  VAE encoding (batched → μ + reconstruction)
# ═══════════════════════════════════════════════════════════════════════

def encode_and_reconstruct(
    vae: ConvolutionalVAE,
    tensor_norm: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
    latent_type: str = "mu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (latent, recon) both as numpy."""
    n = tensor_norm.shape[0]
    latent_parts, recon_parts = [], []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = torch.from_numpy(tensor_norm[start:end]).float().to(device)
            recon_x, mu, _, z = vae(batch)
            lat = mu if latent_type == "mu" else z
            latent_parts.append(lat.cpu().numpy())
            recon_parts.append(recon_x.cpu().numpy())
    return np.concatenate(latent_parts), np.concatenate(recon_parts)


# ═══════════════════════════════════════════════════════════════════════
#  Fold-level inference
# ═══════════════════════════════════════════════════════════════════════

def run_inference_single_fold(
    fold_idx: int,
    fold_dir: Path,
    run_config: Dict[str, Any],
    covid_tensor_selected: np.ndarray,
    covid_subject_ids: np.ndarray,
    covid_metadata_df: Optional[pd.DataFrame],
    classifier_types: List[str],
    device: torch.device,
    ad_tensor_selected: Optional[np.ndarray] = None,
    batch_size_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Inference for one fold.  Returns dict with:
      predictions, recon_error, covid_latent, baseline_latent, latent_distance
    """
    tag = f"Fold {fold_idx}"
    logger.info(f"--- {tag} ---")
    args_cfg = run_config["args"]
    empty = {"predictions": [], "recon_error": None,
             "covid_latent": None, "baseline_latent": None,
             "latent_distance": None}

    # ── 1. Load fold artifacts ──────────────────────────────────────
    norm_path = fold_dir / "vae_norm_params.joblib"
    vae_path  = fold_dir / f"vae_model_fold_{fold_idx}.pt"
    fc_path   = fold_dir / "feature_columns.json"
    imp_path  = fold_dir / "metadata_imputation.json"

    for p in [norm_path, vae_path, fc_path]:
        if not p.exists():
            logger.error(
                f"  {tag}: MISSING {p.name}. "
                f"Generate with: python scripts/run_vae_clf_ad_inference.py"
            )
            return empty

    norm_params = joblib.load(norm_path)
    fc_info = _load_json(fc_path)
    final_cols = fc_info["final_feature_columns"]
    latent_names = fc_info["latent_feature_names"]

    has_meta = fc_info.get("has_metadata_features", False)
    meta_feats = fc_info.get("metadata_features") or []
    imp_info: Dict[str, Any] = {}
    if has_meta and imp_path.exists():
        imp_info = _load_json(imp_path)

    # ── 2. VAE ──────────────────────────────────────────────────────
    n_ch = covid_tensor_selected.shape[1]
    img_sz = covid_tensor_selected.shape[-1]

    vae = ConvolutionalVAE(
        input_channels=n_ch,
        latent_dim=int(args_cfg["latent_dim"]),
        image_size=img_sz,
        final_activation=args_cfg["vae_final_activation"],
        intermediate_fc_dim_config=args_cfg["intermediate_fc_dim_vae"],
        dropout_rate=float(args_cfg["dropout_rate_vae"]),
        use_layernorm_fc=bool(args_cfg.get("use_layernorm_vae_fc", False)),
        num_conv_layers_encoder=int(args_cfg["num_conv_layers_encoder"]),
        decoder_type=args_cfg["decoder_type"],
    ).to(device)

    state = torch.load(vae_path, map_location=device, weights_only=True)
    vae.load_state_dict(state)
    vae.eval()
    logger.info(f"  {tag}: VAE loaded ({vae_path.name})")

    # ── 3. Normalise ────────────────────────────────────────────────
    covid_norm = apply_normalization_params(covid_tensor_selected, norm_params)
    logger.info(f"  {tag}: Norm applied (mode={norm_params[0].get('mode')})")

    # ── 4. Forward pass → μ + recon ─────────────────────────────────
    lat_type = args_cfg.get("latent_features_type", "mu")
    bs = batch_size_override or int(args_cfg.get("batch_size", 64))

    covid_latent, covid_recon = encode_and_reconstruct(
        vae, covid_norm, device, bs, lat_type
    )
    logger.info(f"  {tag}: Latent shape={covid_latent.shape}")

    lat_stds = covid_latent.std(axis=0)
    n_dead = (lat_stds < 1e-6).sum()
    if n_dead > 0:
        logger.warning(f"  {tag}: {n_dead}/{covid_latent.shape[1]} near-zero dims")

    # ── 5. Recon error ──────────────────────────────────────────────
    recon_err = compute_recon_error_offdiag(covid_norm, covid_recon)
    logger.info(
        f"  {tag}: Recon error mean={recon_err.mean():.4f} "
        f"std={recon_err.std():.4f}"
    )

    # ── 6. Latent distance (Mahalanobis) ────────────────────────────
    baseline_latent = None
    latent_dist = None
    td_csv = fold_dir / "train_dev_subjects_fold.csv"
    if ad_tensor_selected is not None and td_csv.exists():
        td_df = pd.read_csv(td_csv)
        if "tensor_idx" in td_df.columns:
            idx = td_df["tensor_idx"].values
            ad_sel = ad_tensor_selected[idx]
            ad_norm = apply_normalization_params(ad_sel, norm_params)
            baseline_latent, _ = encode_and_reconstruct(
                vae, ad_norm, device, bs, lat_type
            )
            latent_dist = compute_mahalanobis_distances(
                covid_latent, baseline_latent
            )
            logger.info(
                f"  {tag}: Mahalanobis mean={latent_dist.mean():.2f} "
                f"std={latent_dist.std():.2f} (baseline N={baseline_latent.shape[0]})"
            )

    # ── 7. Feature matrix ───────────────────────────────────────────
    X_lat = pd.DataFrame(covid_latent, columns=latent_names)

    if has_meta and meta_feats:
        logger.info(f"  {tag}: Metadata features: {meta_feats}")
        meta_df = _prepare_metadata_features(
            covid_metadata_df, meta_feats, imp_info,
            n_subjects=covid_latent.shape[0],
        )
        X_lat.reset_index(drop=True, inplace=True)
        meta_df.reset_index(drop=True, inplace=True)
        X_comb = pd.concat([X_lat, meta_df], axis=1)
    else:
        X_comb = X_lat

    missing = set(final_cols) - set(X_comb.columns)
    if missing:
        logger.warning(f"  {tag}: Missing cols: {missing}. Filling with 0.")
        for c in missing:
            X_comb[c] = 0.0
    X_final = X_comb[final_cols]

    # ── 8. Classify ─────────────────────────────────────────────────
    fold_preds: List[pd.DataFrame] = []
    for ct in classifier_types:
        cp = _resolve_classifier_path(fold_dir, ct, fold_idx)
        if cp is None:
            logger.warning(
                f"  {tag}: No {ct} classifier found "
                f"(checked calibrated/final/raw/legacy)."
            )
            continue
        pipeline = joblib.load(cp)
        variant = cp.stem.split(f"_{ct}_")[1].split("_pipeline")[0]
        logger.info(f"  {tag}: Loaded {ct} ({variant})")

        y_score = _get_score_1d(pipeline, X_final)
        y_pred = pipeline.predict(X_final)

        fold_preds.append(pd.DataFrame({
            "SubjectID": covid_subject_ids,
            "fold": fold_idx,
            "classifier": ct,
            "classifier_variant": variant,
            "y_score": y_score,
            "y_pred": y_pred,
        }))
        logger.info(
            f"  {tag} [{ct}]: score=[{y_score.min():.4f},{y_score.max():.4f}], "
            f"pred={{0:{(y_pred==0).sum()},1:{(y_pred==1).sum()}}}"
        )

    del vae
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "predictions": fold_preds,
        "recon_error": recon_err,
        "covid_latent": covid_latent,
        "baseline_latent": baseline_latent,
        "latent_distance": latent_dist,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Ensemble across folds
# ═══════════════════════════════════════════════════════════════════════

def ensemble_fold_predictions(
    all_fold_dfs: List[pd.DataFrame],
    method: str = "mean",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Aggregate per-fold predictions into ensemble."""
    if not all_fold_dfs:
        return pd.DataFrame()
    combined = pd.concat(all_fold_dfs, ignore_index=True)
    agg_fn = "mean" if method == "mean" else "median"

    ens = (
        combined.groupby(["SubjectID", "classifier"])
        .agg(
            y_score_ensemble=("y_score", agg_fn),
            y_score_std=("y_score", "std"),
            y_score_min=("y_score", "min"),
            y_score_max=("y_score", "max"),
            n_folds=("fold", "nunique"),
        )
        .reset_index()
    )
    ens["y_pred_ensemble"] = (ens["y_score_ensemble"] >= threshold).astype(int)

    vote = (
        combined.groupby(["SubjectID", "classifier"])["y_pred"]
        .agg(lambda x: int(x.mean() >= 0.5))
        .reset_index()
        .rename(columns={"y_pred": "y_pred_majority_vote"})
    )
    return ens.merge(vote, on=["SubjectID", "classifier"], how="left")


def ensemble_scalar_per_fold(
    per_fold_arrays: Dict[int, np.ndarray],
    subject_ids: np.ndarray,
    value_name: str,
    method: str = "mean",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ensemble a per-subject scalar across folds."""
    rows = []
    for fi, arr in per_fold_arrays.items():
        for i, sid in enumerate(subject_ids):
            rows.append({"SubjectID": sid, "fold": fi, value_name: arr[i]})
    pf = pd.DataFrame(rows)

    agg_fn = "mean" if method == "mean" else "median"
    ens = (
        pf.groupby("SubjectID")[value_name]
        .agg(**{
            f"{value_name}_ensemble": agg_fn,
            f"{value_name}_std": "std",
        })
        .reset_index()
    )
    return pf, ens


# ═══════════════════════════════════════════════════════════════════════
#  Consensus-edge "AD signature" scoring
# ═══════════════════════════════════════════════════════════════════════

def compute_signature_scores(
    consensus_edges_path: Path,
    covid_tensor_full: np.ndarray,
    covid_subject_ids: np.ndarray,
    ad_tensor: np.ndarray,
    ad_metadata_path: Path,
    roi_names: List[str],
    channels_to_use: Optional[List[int]] = None,
    channel_idx_for_signature: int = 0,
) -> Optional[pd.DataFrame]:
    """
    Per-COVID-subject "AD signature score":
        S_sig = sum_e (w_signed_e * z_e)
    where z_e standardized relative to AD CN baseline.
    """
    try:
        ce = pd.read_csv(consensus_edges_path)
    except Exception as e:
        logger.warning(f"Cannot load consensus edges: {e}")
        return None

    if "w_signed" not in ce.columns:
        logger.warning("consensus edges missing 'w_signed'. Skipping.")
        return None

    try:
        ad_meta = pd.read_csv(ad_metadata_path)
    except Exception as e:
        logger.warning(f"Cannot load AD metadata: {e}")
        return None

    cn_mask = ad_meta["ResearchGroup"].str.upper().isin(
        ["CN", "CONTROL", "HC"]
    )
    if cn_mask.sum() == 0:
        logger.warning("No CN subjects found in AD metadata.")
        return None
    cn_idx = ad_meta.index[cn_mask].values
    cn_idx = cn_idx[cn_idx < ad_tensor.shape[0]]
    logger.info(f"  Signature: {len(cn_idx)} CN subjects as baseline")

    actual_ch = channels_to_use[channel_idx_for_signature] \
        if channels_to_use else channel_idx_for_signature

    roi_map = {n: i for i, n in enumerate(roi_names)}

    edges = []
    for _, r in ce.iterrows():
        s, d = r.get("src_AAL3_Name"), r.get("dst_AAL3_Name")
        if s in roi_map and d in roi_map:
            edges.append((roi_map[s], roi_map[d], r["w_signed"]))

    if not edges:
        logger.warning("No consensus edges mapped to ROI indices.")
        return None
    logger.info(f"  Signature: {len(edges)} edges mapped")

    cn_ch = ad_tensor[cn_idx, actual_ch, :, :]
    cv_ch = covid_tensor_full[:, actual_ch, :, :]

    rows = []
    for si, sid in enumerate(covid_subject_ids):
        s_sig, n_val = 0.0, 0
        for i_s, i_d, w in edges:
            cn_vals = cn_ch[:, i_s, i_d]
            mu, sd = cn_vals.mean(), cn_vals.std()
            if sd < 1e-9:
                continue
            z_e = (cv_ch[si, i_s, i_d] - mu) / sd
            s_sig += w * z_e
            n_val += 1
        rows.append({"SubjectID": sid, "S_sig": s_sig, "n_edges_used": n_val})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_covid_tensor(path: Path):
    """Load COVID NPZ; returns dict-like npz object."""
    logger.info(f"Loading COVID tensor: {path}")
    if not path.exists():
        raise FileNotFoundError(f"COVID tensor not found: {path}")
    return np.load(path, allow_pickle=True)


def load_metadata_optional(
    meta_path: Optional[Path], subject_ids: np.ndarray
) -> Optional[pd.DataFrame]:
    """Load COVID metadata if exists; else return None."""
    if meta_path is None:
        logger.info("No COVID metadata path. Proceeding without.")
        return None
    meta_path = Path(meta_path)
    if not meta_path.exists():
        logger.warning(f"COVID metadata not found: {meta_path}. Skipping.")
        return None

    df = pd.read_csv(meta_path)
    df["SubjectID"] = df["SubjectID"].astype(str).str.strip()
    logger.info(f"COVID metadata loaded: {df.shape}")

    tdf = pd.DataFrame({
        "SubjectID": subject_ids.astype(str),
        "tensor_idx": np.arange(len(subject_ids)),
    })
    merged = pd.merge(tdf, df, on="SubjectID", how="left")
    merged = merged.sort_values("tensor_idx").reset_index(drop=True)
    return merged


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    np.random.seed(SEED_GLOBAL)
    torch.manual_seed(SEED_GLOBAL)

    training_dir = Path(args.training_output_dir)
    if not training_dir.is_absolute():
        training_dir = PROJECT_ROOT / training_dir

    output_dir = Path(args.output_dir) if args.output_dir else \
        training_dir / "inference_covid_paper_output"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    tables_dir  = output_dir / "Tables"
    figures_dir = output_dir / "Figures"
    logs_dir    = output_dir / "Logs"
    for d in [tables_dir, figures_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).isoformat()
    logger.info(f"{'='*70}")
    logger.info(f"COVID → AD Transfer Inference (paper-grade v2)")
    logger.info(f"Training dir : {training_dir}")
    logger.info(f"Output dir   : {output_dir}")
    logger.info(f"Timestamp    : {ts}")
    logger.info(f"{'='*70}")

    # ── Load run_config ─────────────────────────────────────────────
    rc_path = training_dir / "run_config.json"
    if not rc_path.exists():
        logger.critical(
            f"run_config.json not found at {rc_path}. "
            f"Run training first: python scripts/run_vae_clf_ad_inference.py"
        )
        return
    run_config = _load_json(rc_path)
    args_cfg = run_config["args"]
    logger.info(f"Config loaded. git_hash={run_config.get('git_hash', 'N/A')}")

    # ── COVID tensor ────────────────────────────────────────────────
    covid_tp = Path(args.covid_tensor_path)
    if not covid_tp.is_absolute():
        covid_tp = PROJECT_ROOT / covid_tp
    covid_npz = load_covid_tensor(covid_tp)
    covid_tensor = covid_npz["global_tensor_data"]
    covid_ids = covid_npz["subject_ids"].astype(str)
    covid_rois = list(covid_npz["roi_names_in_order"])
    covid_chs  = list(covid_npz["channel_names"])

    if args.smoke_test:
        n_sm = min(20, covid_tensor.shape[0])
        logger.info(f"SMOKE TEST: {n_sm} subjects")
        covid_tensor = covid_tensor[:n_sm]
        covid_ids = covid_ids[:n_sm]

    logger.info(f"COVID tensor: {covid_tensor.shape} {covid_tensor.dtype}")

    # ── Validate ────────────────────────────────────────────────────
    validate_tensor_compatibility(covid_npz, run_config)

    # ── Metadata (optional) ─────────────────────────────────────────
    meta_path = Path(args.covid_metadata_path) if args.covid_metadata_path else None
    if meta_path and not meta_path.is_absolute():
        meta_path = PROJECT_ROOT / meta_path
    covid_meta = load_metadata_optional(meta_path, covid_ids)

    # ── Channel selection ───────────────────────────────────────────
    ch_use = args_cfg.get("channels_to_use")
    ch_names_sel = args_cfg.get("selected_channel_names", [])
    if ch_use:
        covid_sel = covid_tensor[:, ch_use, :, :]
        logger.info(f"Channels: idx={ch_use} → {ch_names_sel}")
    else:
        covid_sel = covid_tensor

    # ── AD tensor (for OOD baselines) ──────────────────────────────
    ad_tensor = None
    ad_sel = None
    ad_path_str = args_cfg.get("global_tensor_path", "")
    if ad_path_str:
        ad_path = PROJECT_ROOT / ad_path_str
        if ad_path.exists():
            ad_npz = np.load(ad_path, allow_pickle=True)
            ad_tensor = ad_npz["global_tensor_data"]
            ad_sel = ad_tensor[:, ch_use, :, :] if ch_use else ad_tensor
            logger.info(f"AD tensor loaded: {ad_tensor.shape}")
        else:
            logger.warning(f"AD tensor not found ({ad_path}). OOD latent dist skipped.")

    # ── Discover folds ──────────────────────────────────────────────
    n_outer = int(args_cfg.get("outer_folds", 5))
    n_rep   = int(args_cfg.get("repeated_outer_folds_n_repeats", 1))
    total_f = n_outer * n_rep

    fold_dirs = [(k, training_dir / f"fold_{k}")
                 for k in range(1, total_f + 1)
                 if (training_dir / f"fold_{k}").is_dir()]

    if not fold_dirs:
        logger.critical(f"No fold dirs in {training_dir}.")
        return

    if args.smoke_test and len(fold_dirs) > 1:
        fold_dirs = fold_dirs[:1]
        logger.info(f"SMOKE TEST: fold {fold_dirs[0][0]} only")

    logger.info(f"Folds: {[k for k, _ in fold_dirs]}")

    # ── Classifiers ─────────────────────────────────────────────────
    clf_types = args.classifier_types or args_cfg.get("classifier_types", ["svm"])
    if args.smoke_test:
        clf_types = ["logreg"] if "logreg" in clf_types else clf_types[:1]
    logger.info(f"Classifiers: {clf_types}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ══════════════════════════════════════════════════════════════════
    #  Per-fold loop
    # ══════════════════════════════════════════════════════════════════
    all_preds: List[pd.DataFrame] = []
    recon_by_fold:  Dict[int, np.ndarray] = {}
    dist_by_fold:   Dict[int, np.ndarray] = {}
    latent_by_fold: Dict[int, np.ndarray] = {}

    for fi, fd in fold_dirs:
        res = run_inference_single_fold(
            fold_idx=fi, fold_dir=fd, run_config=run_config,
            covid_tensor_selected=covid_sel,
            covid_subject_ids=covid_ids,
            covid_metadata_df=covid_meta,
            classifier_types=clf_types,
            device=device,
            ad_tensor_selected=ad_sel,
            batch_size_override=args.batch_size,
        )
        all_preds.extend(res["predictions"])
        if res["recon_error"] is not None:
            recon_by_fold[fi] = res["recon_error"]
        if res["latent_distance"] is not None:
            dist_by_fold[fi] = res["latent_distance"]
        if res["covid_latent"] is not None:
            latent_by_fold[fi] = res["covid_latent"]

    if not all_preds:
        logger.critical("No predictions generated.")
        return

    # ══════════════════════════════════════════════════════════════════
    #  Save tables
    # ══════════════════════════════════════════════════════════════════
    raw_df = pd.concat(all_preds, ignore_index=True)
    raw_path = tables_dir / "covid_predictions_per_fold.csv"
    raw_df.to_csv(raw_path, index=False)
    logger.info(f"Per-fold preds → {raw_path}")

    ens_df = ensemble_fold_predictions(all_preds, args.ensemble_method, args.decision_threshold)
    ens_path = tables_dir / "covid_predictions_ensemble.csv"
    ens_df.to_csv(ens_path, index=False)
    logger.info(f"Ensemble preds → {ens_path}")

    # Recon error
    recon_pf_path = recon_ens_path = None
    if recon_by_fold:
        rpf, rens = ensemble_scalar_per_fold(recon_by_fold, covid_ids, "recon_error", args.ensemble_method)
        recon_pf_path  = tables_dir / "covid_recon_error_per_fold.csv"
        recon_ens_path = tables_dir / "covid_recon_error_ensemble.csv"
        rpf.to_csv(recon_pf_path, index=False)
        rens.to_csv(recon_ens_path, index=False)
        logger.info(f"Recon error tables → {tables_dir}")

    # Latent distance
    dist_pf_path = dist_ens_path = None
    if dist_by_fold:
        dpf, dens = ensemble_scalar_per_fold(dist_by_fold, covid_ids, "latent_distance", args.ensemble_method)
        dist_pf_path  = tables_dir / "covid_latent_distance_per_fold.csv"
        dist_ens_path = tables_dir / "covid_latent_distance_ensemble.csv"
        dpf.to_csv(dist_pf_path, index=False)
        dens.to_csv(dist_ens_path, index=False)
        logger.info(f"Latent distance tables → {tables_dir}")

    # Signature
    sig_path = None
    if not args.skip_signature:
        interp_t = training_dir / "interpretability_paper_output" / "tables"
        cands = list(interp_t.glob("consensus_edges_*.csv")) if interp_t.exists() else []
        if not cands:
            cands = list(training_dir.glob("consensus_edges_*.csv"))

        if cands:
            cpath = cands[0]
            logger.info(f"Consensus edges: {cpath.name}")
            ad_meta_p = PROJECT_ROOT / args_cfg.get(
                "metadata_path", "data/SubjectsData_AAL3_procesado2.csv")

            if ad_tensor is not None and ad_meta_p.exists():
                sig_df = compute_signature_scores(
                    cpath, covid_tensor, covid_ids, ad_tensor, ad_meta_p,
                    covid_rois, ch_use, 0,
                )
                if sig_df is not None:
                    sig_path = tables_dir / "covid_signature_scores.csv"
                    sig_df.to_csv(sig_path, index=False)
                    logger.info(f"Signature scores → {sig_path}")
        else:
            logger.info("No consensus_edges_*.csv found. Skipping signature scoring.")

    # ══════════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════════
    for ct in clf_types:
        ce = ens_df[ens_df["classifier"] == ct]
        if ce.empty:
            continue
        n = len(ce)
        npos = (ce["y_pred_ensemble"] == 1).sum()
        logger.info(
            f"\n{'='*60}\n{ct}: "
            f"{n} subj, {npos} AD ({100*npos/n:.1f}%), "
            f"score={ce['y_score_ensemble'].mean():.4f}±{ce['y_score_ensemble'].std():.4f}, "
            f"σ_fold={ce['y_score_std'].mean():.4f}\n{'='*60}"
        )

    # ══════════════════════════════════════════════════════════════════
    #  Manifest
    # ══════════════════════════════════════════════════════════════════
    outputs_written = {
        "predictions_per_fold": str(raw_path),
        "predictions_ensemble": str(ens_path),
    }
    if recon_pf_path:
        outputs_written["recon_error_per_fold"]  = str(recon_pf_path)
        outputs_written["recon_error_ensemble"]  = str(recon_ens_path)
    if dist_pf_path:
        outputs_written["latent_distance_per_fold"]  = str(dist_pf_path)
        outputs_written["latent_distance_ensemble"]  = str(dist_ens_path)
    if sig_path:
        outputs_written["signature_scores"] = str(sig_path)

    manifest = {
        "created_utc": ts,
        "git_commit": _git_hash(),
        "results_dir": str(training_dir.resolve()),
        "covid_tensor_path": str(covid_tp.resolve()),
        "covid_metadata_path": str(meta_path.resolve()) if meta_path else None,
        "ad_tensor_path": str((PROJECT_ROOT / ad_path_str).resolve()) if ad_tensor is not None else None,
        "covid_tensor_shape": list(covid_tensor.shape),
        "covid_n_subjects": int(len(covid_ids)),
        "channels_to_use": ch_use,
        "channel_names_selected": ch_names_sel,
        "latent_features_type": args_cfg.get("latent_features_type", "mu"),
        "classifier_types": clf_types,
        "ensemble_method": args.ensemble_method,
        "decision_threshold": args.decision_threshold,
        "n_folds_used": len(fold_dirs),
        "folds_used": [k for k, _ in fold_dirs],
        "smoke_test": args.smoke_test,
        "seed_global": SEED_GLOBAL,
        "seed_sampling": SEED_SAMPLING,
        "training_args_subset": {
            k: args_cfg.get(k)
            for k in [
                "latent_dim", "beta_vae", "norm_mode", "decoder_type",
                "vae_final_activation", "dropout_rate_vae",
                "num_conv_layers_encoder", "metadata_features",
                "outer_folds", "repeated_outer_folds_n_repeats",
            ]
        },
        "outputs_written": outputs_written,
    }
    _safe_json_dump(manifest, output_dir / "run_manifest.json")
    _safe_json_dump(manifest, output_dir / "inference_config.json")

    logger.info(f"\nDone. Outputs → {output_dir}")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Transfer inference: AD/CN pipeline → COVID (paper-grade v2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--training_output_dir", type=str,
                   default="results/vae_3channels_beta65_pro",
                   help="Training output dir (contains run_config.json, fold_*/)")
    p.add_argument("--covid_tensor_path", type=str,
                   default=(
                       "data/COVID_AAL3_Tensor_v1_AAL3_131ROIs_OMST_GCE_Signed_"
                       "GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned/"
                       "GLOBAL_TENSOR_from_COVID_AAL3_Tensor_v1_AAL3_131ROIs_OMST_GCE_"
                       "Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned.npz"),
                   help="Path to COVID .npz tensor")
    p.add_argument("--covid_metadata_path", type=str, default=None,
                   help="Optional COVID metadata CSV (must have 'SubjectID')")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output dir. Default: RESULTS_DIR/inference_covid_paper_output")
    p.add_argument("--classifier_types", nargs="+", default=None,
                   help="Classifiers. Default: from training config.")
    p.add_argument("--ensemble_method", type=str, default="mean",
                   choices=["mean", "median"])
    p.add_argument("--decision_threshold", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--skip_signature", action="store_true",
                   help="Skip consensus-edge signature scoring.")
    p.add_argument("--smoke_test", action="store_true",
                   help="Smoke: fold_1, 20 subjects, logreg only.")
    p.add_argument("--covid_label_column", type=str, default=None,
                   help="If metadata has labels, compute eval metrics.")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
