#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_vae_clf_ad_inference.py

Main pipeline: CNN β-VAE + classical classifiers for AD vs CN.

Usa:
- betavae_xai.models.ConvolutionalVAE
- betavae_xai.models.get_classifier_and_grid
- betavae_xai.analysis_qc (QC distribuciones y leakage de sitio)
"""
from __future__ import annotations

# --- Bootstrap: añadir src/ al sys.path (repo/{scripts,src}) ---
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
else:
    raise FileNotFoundError(f"No se encontró 'src/' en: {SRC_DIR}")
REVISION_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "revision_bspc_2026"
if REVISION_SCRIPT_DIR.is_dir():
    sys.path.insert(0, str(REVISION_SCRIPT_DIR))
# --- Fin bootstrap ---

import argparse
import copy
import gc
import json
import subprocess
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime, timezone
import platform
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler


# Scikit
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, f1_score,
    average_precision_score, balanced_accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold,
    train_test_split as sk_train_test_split
)

# Optuna
import optuna
from optuna.integration import OptunaSearchCV
from optuna.pruners import MedianPruner

# Torch Data
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# Plot (si lo usás)
import matplotlib.pyplot as plt

# Proyecto
from betavae_xai.models import (
    BLOCK_ORDER_CHOICES,
    CONDITIONING_MODE_CHOICES,
    ConvolutionalVAE,
    DROPOUT_SCOPE_CHOICES,
    build_vae_dropout_manifest,
    get_classifier_and_grid,
    get_available_classifiers,
    summarize_vae_dropout_manifest,
)
from betavae_xai.analysis_qc.fold_qc import (
    log_group_distributions,
    compute_latent_silhouette,
    #scanner_leakage_simple,
    summarize_distribution_stages,
    summarize_rate_distortion_history,
    evaluate_latent_information,
    evaluate_scanner_leakage,
    evaluate_scanner_leakage_per_channel,
)

from betavae_xai.utils.logging import setup_logging
from betavae_xai.utils.run_io import _safe_json_dump, _compute_file_sha256

# Data plumbing (carga + normalización) modularizada
from betavae_xai.data.preprocessing import (
    load_data,
    normalize_inter_channel_fold,
    apply_normalization_params,
)

try:
    from foldwise_combat_input_harmonization import (
        channel_shift_summary as combat_channel_shift_summary,
        dependency_status as combat_dependency_status,
        fit_tensor_combat_channelwise,
        manufacturer_centroid_separability_proxy,
        normalize_manufacturer as combat_normalize_manufacturer,
        transform_tensor_combat_channelwise,
    )
except Exception:  # pragma: no cover - only used when explicitly enabled
    combat_channel_shift_summary = None  # type: ignore[assignment]
    combat_dependency_status = None  # type: ignore[assignment]
    fit_tensor_combat_channelwise = None  # type: ignore[assignment]
    manufacturer_centroid_separability_proxy = None  # type: ignore[assignment]
    combat_normalize_manufacturer = None  # type: ignore[assignment]
    transform_tensor_combat_channelwise = None  # type: ignore[assignment]

# --- logging & warnings: una sola vez ---
logger = setup_logging(__name__)


# --- Constantes y Configuraciones Globales ---
DEFAULT_CHANNEL_NAMES = [
    'Pearson_OMST_GCE_Signed_Weighted', 'Pearson_Full_FisherZ_Signed', 'MI_KNN_Symmetric',
    'dFC_AbsDiffMean', 'dFC_StdDev', 'DistanceCorr', 'Granger_F_lag1' # <<< Lista actualizada para ser completa
]

CLASSIFIER_N_ITER_ARG_MAP = {
    "logreg": "n_iter_logreg",
    "svm": "n_iter_svm",
    "rf": "n_iter_rf",
    "gb": "n_iter_gb",
    "xgb": "n_iter_xgb",
    "mlp": "n_iter_mlp",
}

def _extend_channel_names_to_tensor(n_chan_tensor: int, base_names: List[str]) -> List[str]:
    """
    Asegura que la lista de nombres tenga longitud == n_chan_tensor.
    Si faltan nombres, completa con RawChan{i}.
    Si sobran, trunca.
    """
    base = list(base_names) if base_names is not None else []
    if len(base) < n_chan_tensor:
        base = base + [f"RawChan{i}" for i in range(len(base), n_chan_tensor)]
    elif len(base) > n_chan_tensor:
        base = base[:n_chan_tensor]
    return base

def _filter_existing_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Devuelve solo columnas que existen en df."""
    if df is None or df.empty:
        return []
    return [c for c in cols if c in df.columns]


def _normalize_mfr_for_harmonization(series: pd.Series) -> pd.Series:
    if combat_normalize_manufacturer is None:
        return series.fillna("UNKNOWN").astype(str)
    return series.map(combat_normalize_manufacturer).fillna("UNKNOWN").astype(str)


def _write_foldwise_input_harmonization_subjects(
    fold_output_dir: Path,
    *,
    fold: int,
    split_name: str,
    df: pd.DataFrame,
) -> None:
    cols = _filter_existing_cols(
        df,
        ["SubjectID", "tensor_idx", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"],
    )
    out = df[cols].copy()
    out.insert(0, "fold", int(fold))
    out.insert(1, "input_harmonization_split", str(split_name))
    out.to_csv(fold_output_dir / f"input_harmonization_{split_name}_subjects.csv", index=False)


def _validate_foldwise_input_harmonization_guards(
    *,
    fit_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_col: str,
    covariates: List[str],
    excluded_covariates: List[str],
    fold_number: int,
) -> Dict[str, Any]:
    required_cols = ["SubjectID", "tensor_idx", batch_col, *covariates]
    missing = [c for c in required_cols if c not in fit_df.columns or c not in test_df.columns]
    if missing:
        raise RuntimeError(f"Fold {fold_number} input harmonization missing required columns: {missing}")
    diagnosis_like = {"ResearchGroup_Mapped", "Diagnosis", "DX", "label", "y"}
    bad_covars = [c for c in covariates if c in diagnosis_like]
    if bad_covars:
        raise RuntimeError(f"Fold {fold_number} input harmonization covariates include diagnosis-like columns: {bad_covars}")
    if not set(excluded_covariates).issuperset({"ResearchGroup_Mapped"}):
        raise RuntimeError(
            f"Fold {fold_number} input harmonization must explicitly exclude ResearchGroup_Mapped; "
            f"got {excluded_covariates}"
        )
    fit_subjects = set(fit_df["SubjectID"].astype(str))
    test_subjects = set(test_df["SubjectID"].astype(str))
    overlap = sorted(fit_subjects.intersection(test_subjects))
    if overlap:
        raise RuntimeError(
            f"Fold {fold_number} leakage guard failed: fit/test SubjectID overlap detected: {overlap[:20]}"
        )
    fit_mfr = _normalize_mfr_for_harmonization(fit_df[batch_col])
    test_mfr = _normalize_mfr_for_harmonization(test_df[batch_col])
    required_mfr = {"GE", "Philips", "SIEMENS"}
    fit_levels = set(fit_mfr.unique())
    missing_levels = sorted(required_mfr.difference(fit_levels))
    if missing_levels:
        raise RuntimeError(
            f"Fold {fold_number} leakage guard failed: fit train/dev lacks Manufacturer levels {missing_levels}"
        )
    for covar in covariates:
        if covar == "Age":
            fit_age = pd.to_numeric(fit_df[covar], errors="coerce")
            test_age = pd.to_numeric(test_df[covar], errors="coerce")
            if fit_age.isna().any() or test_age.isna().any():
                raise RuntimeError(
                    f"Fold {fold_number} input harmonization has missing/non-numeric Age in fit/test."
                )
        elif covar == "Sex":
            bad_fit = fit_df[covar].isna() | fit_df[covar].astype(str).str.strip().isin(["", "nan", "NaN", "None", "NA"])
            bad_test = test_df[covar].isna() | test_df[covar].astype(str).str.strip().isin(["", "nan", "NaN", "None", "NA"])
            if bad_fit.any() or bad_test.any():
                raise RuntimeError(f"Fold {fold_number} input harmonization has missing Sex in fit/test.")
    return {
        "fold": int(fold_number),
        "status": "PASS",
        "fit_scope": "outer_train_dev_only",
        "batch_col": str(batch_col),
        "covariates_preserved": "+".join(covariates),
        "excluded_covariates": "+".join(excluded_covariates),
        "n_fit_subjects": int(len(fit_df)),
        "n_test_subjects": int(len(test_df)),
        "n_fit_test_subject_overlap": 0,
        "fit_manufacturer_levels": ";".join(sorted(fit_levels)),
        "test_manufacturer_levels": ";".join(sorted(set(test_mfr.unique()))),
        "diagnosis_used_in_harmonizer": False,
        "global_combat": False,
        "oasis_used": False,
    }


def _validate_n_iter_value(classifier_type: str, value: Any) -> int:
    """Validate optional per-classifier Optuna trial override."""
    if isinstance(value, bool):
        raise ValueError(f"n_iter for {classifier_type!r} must be a positive integer, got bool.")
    try:
        n_iter = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"n_iter for {classifier_type!r} must be a positive integer, got {value!r}.") from None
    if n_iter <= 0:
        raise ValueError(f"n_iter for {classifier_type!r} must be > 0, got {n_iter}.")
    return n_iter


def _parse_classifier_n_iter_overrides(args: argparse.Namespace) -> Dict[str, int]:
    """
    Parse optional per-classifier Optuna trial budgets.

    Backward compatibility: when all values are omitted, returns an empty dict and
    the classifier factory defaults remain in force exactly as before.
    """
    valid_classifiers = set(get_available_classifiers())
    overrides: Dict[str, int] = {}
    json_payload = getattr(args, "classifier_n_iter_json", None)
    if json_payload:
        try:
            parsed = json.loads(json_payload)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid --classifier_n_iter_json: {e}") from e
        if not isinstance(parsed, dict):
            raise ValueError("--classifier_n_iter_json must decode to an object, e.g. '{\"logreg\":100,\"svm\":300}'.")
        for key, value in parsed.items():
            classifier_type = str(key).lower()
            if classifier_type not in valid_classifiers:
                raise ValueError(
                    f"Unknown classifier in --classifier_n_iter_json: {classifier_type!r}. "
                    f"Valid classifiers: {sorted(valid_classifiers)}"
                )
            overrides[classifier_type] = _validate_n_iter_value(classifier_type, value)

    for classifier_type, arg_name in CLASSIFIER_N_ITER_ARG_MAP.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            if classifier_type not in valid_classifiers:
                raise ValueError(f"Unsupported n_iter flag for unavailable classifier: {classifier_type}")
            overrides[classifier_type] = _validate_n_iter_value(classifier_type, value)
    return overrides


def _resolve_classifier_n_iter(
    args: argparse.Namespace,
    classifier_type: str,
    factory_default_n_iter: int,
    max_trials: int = 1000,
) -> tuple[int, Optional[int], str]:
    """Return effective Optuna trials while preserving old behavior by default."""
    classifier_key = classifier_type.lower()
    overrides: Dict[str, int] = getattr(args, "classifier_n_iter_overrides", {}) or {}
    requested_override = overrides.get(classifier_key)
    if requested_override is None:
        requested_n_iter = int(factory_default_n_iter)
        source = "factory_default"
    else:
        requested_n_iter = int(requested_override)
        source = "cli_override"
    effective_n_trials = min(requested_n_iter, int(max_trials))
    return effective_n_trials, requested_override, source

def _get_score_1d(estimator, X):
    """
    Devuelve un score 1D para AUC/PR-AUC:
      - predict_proba[:,1] si existe
      - decision_function si existe
      - fallback: predict() como score (último recurso)
    """
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return np.asarray(p).ravel()
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
        return np.asarray(s).ravel()
    return np.asarray(estimator.predict(X)).astype(float).ravel()

RECON_LOSS_MODE_CURRENT = "mse_sum_batchmean_current"
RECON_LOSS_MODE_OFFDIAG_CHANNELMEAN = "offdiag_channelmean_sum"
RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_MEAN_SUM = "mse_offdiag_channel_mean_sum"
RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_WEIGHTED_SUM = "mse_offdiag_channel_weighted_sum"
RECON_LOSS_MODES = (
    RECON_LOSS_MODE_CURRENT,
    RECON_LOSS_MODE_OFFDIAG_CHANNELMEAN,
    RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_MEAN_SUM,
    RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_WEIGHTED_SUM,
)
VAE_TRAIN_SAMPLER_NONE = "none"
VAE_TRAIN_SAMPLER_MANUFACTURER = "manufacturer_balanced"
VAE_TRAIN_SAMPLER_DIAGNOSIS_MANUFACTURER = "diagnosis_manufacturer_balanced"
VAE_TRAIN_SAMPLER_STRATEGIES = (
    VAE_TRAIN_SAMPLER_NONE,
    VAE_TRAIN_SAMPLER_MANUFACTURER,
    VAE_TRAIN_SAMPLER_DIAGNOSIS_MANUFACTURER,
)
VAE_POOL_CURRENT_ALL = "current_all_pool"
VAE_POOL_CN_AD_ONLY = "cn_ad_only_pool"
VAE_POOL_BALANCED_CN_AD_MCI = "balanced_cn_ad_mci_pool"
VAE_POOL_CN_AD_PLUS_MATCHED_MCI = "cn_ad_plus_matched_mci_pool"
VAE_POOL_COMPOSITION_STRATEGIES = (
    VAE_POOL_CURRENT_ALL,
    VAE_POOL_CN_AD_ONLY,
    VAE_POOL_BALANCED_CN_AD_MCI,
    VAE_POOL_CN_AD_PLUS_MATCHED_MCI,
)
VAE_CONDITIONING_VAR_CHOICES = ("none", "Age", "age", "sex", "age_sex", "manufacturer")


def _offdiag_mask_for_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor [B,C,H,W], got shape={tuple(x.shape)}")
    if x.shape[-1] != x.shape[-2]:
        raise ValueError(f"Expected square matrices, got shape={tuple(x.shape)}")
    n_rois = int(x.shape[-1])
    return ~torch.eye(n_rois, dtype=torch.bool, device=x.device)


def _channel_weights_for_tensor(
    channel_weights: Optional[Sequence[float]],
    x: torch.Tensor,
) -> torch.Tensor:
    if channel_weights is None:
        raise ValueError(
            f"recon_loss_channel_weights must be provided for {RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_WEIGHTED_SUM}"
        )
    weights = torch.as_tensor(channel_weights, dtype=x.dtype, device=x.device)
    if weights.ndim != 1:
        raise ValueError(f"recon_loss_channel_weights must be 1D, got shape={tuple(weights.shape)}")
    if weights.numel() != int(x.shape[1]):
        raise ValueError(
            f"recon_loss_channel_weights length {weights.numel()} does not match input channels {int(x.shape[1])}"
        )
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("recon_loss_channel_weights must be finite")
    if bool((weights < 0).any()):
        raise ValueError("recon_loss_channel_weights must be non-negative")
    weight_sum = weights.sum()
    if not torch.isclose(weight_sum, weights.new_tensor(1.0), rtol=1e-5, atol=1e-6):
        raise ValueError(f"recon_loss_channel_weights must sum to 1.0, got {float(weight_sum.detach().cpu())}")
    return weights


def vae_reconstruction_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mode: str = RECON_LOSS_MODE_CURRENT,
    channel_weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Return reconstruction loss with explicit scale semantics.

    - mse_sum_batchmean_current: historical behavior, sum over all channels and
      pixels divided by batch size.
    - offdiag_channelmean_sum / mse_offdiag_channel_mean_sum: sum squared error
      over off-diagonal entries within each channel, then average across
      channels and batch. For identical per-channel errors this preserves the
      approximate single-channel scale and avoids linear growth with channel
      count.
    - mse_offdiag_channel_weighted_sum: same off-diagonal per-channel sums, but
      weighted across channels using explicit non-negative weights that sum to
      one.
    """
    if mode == RECON_LOSS_MODE_CURRENT:
        return nn.functional.mse_loss(recon_x, x, reduction='sum') / x.shape[0]
    if mode in {RECON_LOSS_MODE_OFFDIAG_CHANNELMEAN, RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_MEAN_SUM}:
        offdiag_mask = _offdiag_mask_for_tensor(x)
        diff2 = (recon_x - x).pow(2)
        per_subject_channel_sum = diff2[:, :, offdiag_mask].sum(dim=-1)
        return per_subject_channel_sum.mean(dim=1).mean()
    if mode == RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_WEIGHTED_SUM:
        offdiag_mask = _offdiag_mask_for_tensor(x)
        diff2 = (recon_x - x).pow(2)
        per_subject_channel_sum = diff2[:, :, offdiag_mask].sum(dim=-1)
        weights = _channel_weights_for_tensor(channel_weights, x)
        return (per_subject_channel_sum * weights.view(1, -1)).sum(dim=1).mean()
    raise ValueError(f"Unknown recon_loss_mode={mode!r}. Valid modes: {RECON_LOSS_MODES}")


def latent_covariate_corr_penalty(mu: torch.Tensor, condition: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean squared correlation between latent dimensions and conditioning variables.

    The penalty is differentiable w.r.t. ``mu``. It is intentionally batch-local
    and defaults to zero when conditioning is absent, preserving historical loss
    behavior when lambda is 0.
    """
    if condition is None or condition.numel() == 0 or mu.shape[0] < 2:
        return mu.new_tensor(0.0)
    cond = condition.to(device=mu.device, dtype=mu.dtype)
    if cond.ndim != 2:
        raise ValueError(f"condition must be 2D [B,C], got shape={tuple(cond.shape)}")
    mu_centered = mu - mu.mean(dim=0, keepdim=True)
    cond_centered = cond - cond.mean(dim=0, keepdim=True)
    mu_std = mu_centered.pow(2).mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
    cond_std = cond_centered.pow(2).mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
    mu_z = mu_centered / mu_std
    cond_z = cond_centered / cond_std
    corr = torch.matmul(mu_z.transpose(0, 1), cond_z) / float(mu.shape[0])
    return corr.pow(2).mean()


def describe_recon_loss_mode(
    mode: str,
    n_channels: int,
    n_rois: int,
    channel_weights: Optional[Sequence[float]] = None,
) -> str:
    offdiag_elements = int(n_rois * (n_rois - 1))
    all_elements = int(n_channels * n_rois * n_rois)
    if mode == RECON_LOSS_MODE_CURRENT:
        return (
            f"{mode}: sum over all channels and pixels / batch; "
            f"scale_terms_per_subject={all_elements}"
        )
    if mode in {RECON_LOSS_MODE_OFFDIAG_CHANNELMEAN, RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_MEAN_SUM}:
        return (
            f"{mode}: off-diagonal squared-error sum per channel, mean across channels and batch; "
            f"offdiag_elements_per_channel={offdiag_elements}, scale_terms_per_subject~={offdiag_elements}"
        )
    if mode == RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_WEIGHTED_SUM:
        return (
            f"{mode}: off-diagonal squared-error sum per channel, weighted across channels and mean over batch; "
            f"weights={list(channel_weights) if channel_weights is not None else None}, "
            f"offdiag_elements_per_channel={offdiag_elements}, scale_terms_per_subject~={offdiag_elements}"
        )
    return f"{mode}: unknown scale"


def vae_loss_function(
    recon_x,
    x,
    mu,
    logvar,
    beta=1.0,
    recon_loss_mode: str = RECON_LOSS_MODE_CURRENT,
    recon_loss_channel_weights: Optional[Sequence[float]] = None,
    condition: Optional[torch.Tensor] = None,
    latent_covariate_corr_lambda: float = 0.0,
):
    recon_x = recon_x.float()
    x       = x.float()
    mu      = mu.float()
    logvar  = logvar.float()

    recon_loss = vae_reconstruction_loss(
        recon_x,
        x,
        mode=recon_loss_mode,
        channel_weights=recon_loss_channel_weights,
    )
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    corr_penalty = latent_covariate_corr_penalty(mu, condition)
    total_loss = recon_loss + beta * kld_loss + float(latent_covariate_corr_lambda or 0.0) * corr_penalty
    return total_loss, recon_loss.detach(), kld_loss.detach(), corr_penalty.detach()


def apply_channel_dropout_train(x: torch.Tensor, p: float) -> torch.Tensor:
    """Drop complete input channels during VAE training only.

    This is a denoising-style augmentation: the model receives corrupted input
    but the reconstruction loss is still computed against the uncorrupted target.
    Defaults p=0 preserve historical behavior exactly.
    """
    p = float(p or 0.0)
    if p <= 0.0:
        return x
    if p >= 1.0:
        raise ValueError("vae_channel_dropout_p must be < 1.0")
    if x.ndim != 4 or x.shape[1] <= 1:
        return x
    keep = torch.rand((x.shape[0], x.shape[1], 1, 1), device=x.device, dtype=x.dtype) >= p
    all_dropped = keep.flatten(1).sum(dim=1) == 0
    if bool(all_dropped.any()):
        keep[all_dropped, 0, :, :] = True
    return x * keep / (1.0 - p)


def make_vae_weighted_sampler(
    train_rows: pd.DataFrame,
    strategy: str,
    seed: int,
) -> tuple[Optional[WeightedRandomSampler], pd.DataFrame]:
    """Build a fold-local VAE sampler without using outer test labels."""
    strategy = str(strategy or VAE_TRAIN_SAMPLER_NONE)
    if strategy == VAE_TRAIN_SAMPLER_NONE:
        return None, pd.DataFrame()
    if strategy == VAE_TRAIN_SAMPLER_MANUFACTURER:
        cols = ["Manufacturer"]
    elif strategy == VAE_TRAIN_SAMPLER_DIAGNOSIS_MANUFACTURER:
        cols = ["ResearchGroup_Mapped", "Manufacturer"]
    else:
        raise ValueError(f"Unknown VAE sampler strategy: {strategy}")
    missing = [col for col in cols if col not in train_rows.columns]
    if missing:
        raise ValueError(f"Cannot build VAE sampler; missing columns: {missing}")
    keys = train_rows[cols].fillna("UNKNOWN").astype(str).agg("::".join, axis=1)
    counts = keys.value_counts()
    weights = keys.map(lambda key: 1.0 / float(counts[key])).to_numpy(dtype=np.float64)
    weights = weights / np.mean(weights)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    summary = (
        train_rows.assign(_sampler_key=keys, _sampler_weight=weights)
        .groupby("_sampler_key", dropna=False)
        .agg(n=("_sampler_key", "size"), sampler_weight_mean=("_sampler_weight", "mean"))
        .reset_index()
        .rename(columns={"_sampler_key": "sampler_key"})
    )
    summary.insert(0, "vae_train_sampler_strategy", strategy)
    return sampler, summary


def apply_vae_pool_composition_strategy(
    vae_pool_df: pd.DataFrame,
    strategy: str,
    seed: int,
    fold_number: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a fold-local VAE pool composition.

    The default/current strategy returns the pool unchanged. Non-current
    strategies are exploratory because they use diagnosis labels to choose
    unsupervised VAE training subjects. Selection only sees the already
    fold-local VAE candidate pool, so outer classifier test subjects remain
    excluded.
    """
    strategy = str(strategy or VAE_POOL_CURRENT_ALL)
    if strategy not in VAE_POOL_COMPOSITION_STRATEGIES:
        raise ValueError(f"Unknown VAE pool composition strategy: {strategy}")
    if "ResearchGroup_Mapped" not in vae_pool_df.columns:
        raise ValueError("Cannot apply VAE pool composition strategy without ResearchGroup_Mapped")

    original = vae_pool_df.copy()
    original["_orig_order"] = np.arange(len(original), dtype=int)
    rng_seed = int(seed) + int(fold_number) * 1009
    groups = original["ResearchGroup_Mapped"].fillna("UNKNOWN").astype(str)

    if strategy == VAE_POOL_CURRENT_ALL:
        selected = original.copy()
        selection_note = "unchanged_current_pool"
    elif strategy == VAE_POOL_CN_AD_ONLY:
        selected = original[groups.isin(["CN", "AD"])].copy()
        selection_note = "kept_cn_ad_only_removed_mci"
    elif strategy == VAE_POOL_BALANCED_CN_AD_MCI:
        counts = groups[groups.isin(["CN", "AD", "MCI"])].value_counts()
        missing = [g for g in ["CN", "AD", "MCI"] if counts.get(g, 0) <= 0]
        if missing:
            raise ValueError(f"Cannot balance CN/AD/MCI VAE pool; missing groups: {missing}")
        target_n = int(counts[["CN", "AD", "MCI"]].min())
        parts = []
        for group in ["CN", "AD", "MCI"]:
            part = original[groups.eq(group)].sample(n=target_n, replace=False, random_state=rng_seed + len(parts))
            parts.append(part)
        selected = pd.concat(parts, ignore_index=False)
        selection_note = f"sampled_equal_cn_ad_mci_n_{target_n}"
    elif strategy == VAE_POOL_CN_AD_PLUS_MATCHED_MCI:
        cn_ad = original[groups.isin(["CN", "AD"])].copy()
        ad_n = int(groups.eq("AD").sum())
        mci = original[groups.eq("MCI")].copy()
        if ad_n <= 0:
            raise ValueError("Cannot sample MCI matched to AD count; no AD subjects in VAE pool")
        if len(mci) <= 0:
            raise ValueError("Cannot sample MCI matched to AD count; no MCI subjects in VAE pool")
        mci_n = min(ad_n, int(len(mci)))
        selected = pd.concat(
            [cn_ad, mci.sample(n=mci_n, replace=False, random_state=rng_seed)],
            ignore_index=False,
        )
        selection_note = f"kept_all_cn_ad_sampled_mci_n_{mci_n}_target_ad_n_{ad_n}"
    else:  # pragma: no cover - guarded above
        raise ValueError(strategy)

    selected = selected.sort_values("_orig_order").drop(columns=["_orig_order"]).reset_index(drop=True)
    before_counts = groups.value_counts(dropna=False).to_dict()
    after_counts = selected["ResearchGroup_Mapped"].fillna("UNKNOWN").astype(str).value_counts(dropna=False).to_dict()
    summary = pd.DataFrame(
        [
            {
                "fold": int(fold_number),
                "vae_pool_composition_strategy": strategy,
                "exploratory_uses_diagnosis_for_pool_composition": strategy != VAE_POOL_CURRENT_ALL,
                "n_before": int(len(original)),
                "n_after": int(len(selected)),
                "n_removed": int(len(original) - len(selected)),
                "counts_before": json.dumps({str(k): int(v) for k, v in before_counts.items()}, sort_keys=True),
                "counts_after": json.dumps({str(k): int(v) for k, v in after_counts.items()}, sort_keys=True),
                "selection_note": selection_note,
            }
        ]
    )
    return selected, summary


def conditioning_dim_from_args(args: argparse.Namespace) -> int:
    if str(getattr(args, "vae_conditioning_mode", "none")) == "none":
        return 0
    vars_mode = str(getattr(args, "vae_conditioning_vars", "none"))
    vars_mode_norm = "age" if vars_mode.lower() == "age" else vars_mode
    if vars_mode_norm in {"age", "sex"}:
        return 1
    if vars_mode_norm == "age_sex":
        return 2
    if vars_mode_norm == "manufacturer":
        return 3
    return 0


_CONDITIONING_MISSING_STRINGS = {"", "nan", "na", "n/a", "none", "null", "<na>", "missing"}
_SEX_NORMALIZE_MAP = {
    "f": "F",
    "female": "F",
    "0": "F",
    "0.0": "F",
    "m": "M",
    "male": "M",
    "1": "M",
    "1.0": "M",
}
_SEX_TO_FLOAT = {"M": 0.0, "F": 1.0}
_MANUFACTURER_NORMALIZE_MAP = {
    "ge": "GE",
    "g e": "GE",
    "general electric": "GE",
    "ge medical systems": "GE",
    "philips": "PHILIPS",
    "philips medical systems": "PHILIPS",
    "philips healthcare": "PHILIPS",
    "siemens": "SIEMENS",
    "siemens healthcare": "SIEMENS",
    "siemens healthineers": "SIEMENS",
    "siemens medical systems": "SIEMENS",
}


def _conditioning_missing_like(series: pd.Series) -> pd.Series:
    """Return a boolean mask for explicit missing-like covariate values."""
    raw = series.copy()
    text = raw.astype("string").str.strip().str.lower()
    return raw.isna() | text.isna() | text.isin(_CONDITIONING_MISSING_STRINGS)


def _normalize_conditioning_sex(series: pd.Series) -> pd.Series:
    """Normalize Sex to F/M while preserving missing-like values as NA."""
    missing = _conditioning_missing_like(series)
    normalized = pd.Series(pd.NA, index=series.index, dtype="object")
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map(_SEX_NORMALIZE_MAP)
    normalized.loc[~missing] = mapped.loc[~missing]
    unsupported = normalized.isna() & ~missing
    if unsupported.any():
        examples = series.loc[unsupported].astype(str).drop_duplicates().head(10).tolist()
        raise ValueError(f"VAE conditioning Sex has unsupported non-missing values: {examples}")
    return normalized


def _normalize_conditioning_manufacturer(series: pd.Series) -> pd.Series:
    """Normalize scanner Manufacturer to canonical GE/PHILIPS/SIEMENS labels.

    Missing-like values are returned as NA. Unknown non-missing values fail
    clearly because the current experiment intentionally does not define an
    explicit unknown category.
    """
    missing = _conditioning_missing_like(series)
    normalized = pd.Series(pd.NA, index=series.index, dtype="object")
    text = (
        series.astype("string")
        .str.strip()
        .str.replace(r"[_\\-]+", " ", regex=True)
        .str.replace(r"\\s+", " ", regex=True)
        .str.lower()
    )
    mapped = text.map(_MANUFACTURER_NORMALIZE_MAP)
    normalized.loc[~missing] = mapped.loc[~missing]
    unsupported = normalized.isna() & ~missing
    if unsupported.any():
        examples = series.loc[unsupported].astype(str).drop_duplicates().head(10).tolist()
        raise ValueError(f"VAE conditioning Manufacturer has unsupported non-missing values: {examples}")
    return normalized


def _numeric_conditioning_age(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Convert Age to numeric and return (age, missing_mask), failing on invalid non-missing values."""
    missing = _conditioning_missing_like(series)
    numeric = pd.to_numeric(series, errors="coerce")
    invalid_non_missing = numeric.isna() & ~missing
    if invalid_non_missing.any():
        examples = series.loc[invalid_non_missing].astype(str).drop_duplicates().head(10).tolist()
        raise ValueError(f"VAE conditioning Age has non-numeric non-missing values: {examples}")
    return numeric.astype(float), missing | numeric.isna()


def fit_vae_conditioning_transformer(train_rows: pd.DataFrame, vars_mode: str) -> Dict[str, Any]:
    """Fit fold-local conditioning transform on VAE train rows only."""
    vars_mode = str(vars_mode or "none")
    if vars_mode.lower() == "age":
        vars_mode = "age"
    if vars_mode == "none":
        return {"vars_mode": "none", "columns": [], "conditioning_dim": 0}
    if vars_mode == "age":
        required = ["Age"]
    elif vars_mode == "sex":
        required = ["Sex"]
    elif vars_mode == "age_sex":
        required = ["Age", "Sex"]
    elif vars_mode == "manufacturer":
        required = ["Manufacturer"]
    else:
        raise ValueError(f"Unsupported VAE conditioning vars_mode={vars_mode!r}")
    missing_cols = [col for col in required if col not in train_rows.columns]
    if missing_cols:
        raise ValueError(f"VAE conditioning requires missing metadata columns: {missing_cols}")
    transformer: Dict[str, Any] = {
        "vars_mode": vars_mode,
        "columns": required,
        "conditioning_dim": len(required),
        "sex_mapping": dict(_SEX_TO_FLOAT),
        "fit_scope": "vae_actual_train_only",
    }
    if "Age" in required:
        age, age_missing = _numeric_conditioning_age(train_rows["Age"])
        valid_age = age.loc[~age_missing].dropna()
        if valid_age.empty:
            raise ValueError("VAE conditioning Age has no valid train values for fold-local imputation.")
        age_impute = float(valid_age.median())
        age_imputed = age.mask(age_missing, age_impute)
        std = float(age_imputed.std(ddof=0))
        if not np.isfinite(std) or std <= 0.0:
            raise ValueError("VAE conditioning Age std is zero/non-finite in training fold.")
        transformer["age_impute_value"] = age_impute
        transformer["age_missing_train"] = int(age_missing.sum())
        transformer["age_mean"] = float(age_imputed.mean())
        transformer["age_std"] = std
    if "Sex" in required:
        sex = _normalize_conditioning_sex(train_rows["Sex"])
        sex_missing = sex.isna()
        sex_nonmissing = sex.loc[~sex_missing]
        if sex_nonmissing.empty:
            raise ValueError("VAE conditioning Sex has no valid train values for fold-local imputation.")
        sex_mode = str(sex_nonmissing.mode(dropna=True).iloc[0])
        transformer["sex_impute_value"] = sex_mode
        transformer["sex_missing_train"] = int(sex_missing.sum())
        transformer["sex_counts_train"] = {str(k): int(v) for k, v in sex_nonmissing.value_counts(dropna=False).to_dict().items()}
    if "Manufacturer" in required:
        manufacturer = _normalize_conditioning_manufacturer(train_rows["Manufacturer"])
        manufacturer_missing = manufacturer.isna()
        if manufacturer_missing.any():
            examples = train_rows.loc[manufacturer_missing, ["SubjectID", "Manufacturer"]].head(10).to_dict("records")
            raise ValueError(
                "VAE conditioning Manufacturer has missing train values and no unknown category is enabled: "
                f"{examples}"
            )
        categories = sorted(manufacturer.dropna().unique().tolist())
        if not categories:
            raise ValueError("VAE conditioning Manufacturer has no valid train categories.")
        transformer["manufacturer_categories"] = categories
        transformer["manufacturer_missing_train"] = int(manufacturer_missing.sum())
        transformer["manufacturer_counts_train"] = {
            str(k): int(v) for k, v in manufacturer.value_counts(dropna=False).to_dict().items()
        }
        transformer["manufacturer_mapping"] = {str(cat): int(i) for i, cat in enumerate(categories)}
        transformer["conditioning_dim"] = len(categories)
    return transformer


def transform_vae_conditioning_covariates(rows: pd.DataFrame, transformer: Dict[str, Any]) -> np.ndarray:
    """Apply fold-local VAE conditioning preprocessing to any split."""
    vars_mode = str(transformer.get("vars_mode", "none"))
    if vars_mode == "none":
        return np.zeros((len(rows), 0), dtype=np.float32)
    cols: List[np.ndarray] = []
    if str(vars_mode).lower() == "age":
        vars_mode = "age"
    if vars_mode in {"age", "age_sex"}:
        age, age_missing = _numeric_conditioning_age(rows["Age"])
        age = age.mask(age_missing, float(transformer["age_impute_value"]))
        cols.append(((age.astype(float) - float(transformer["age_mean"])) / float(transformer["age_std"])).to_numpy(dtype=np.float32))
    if vars_mode in {"sex", "age_sex"}:
        sex = _normalize_conditioning_sex(rows["Sex"]).fillna(str(transformer["sex_impute_value"]))
        encoded = sex.map(dict(transformer["sex_mapping"]))
        if encoded.isna().any():
            bad = rows.loc[encoded.isna(), ["SubjectID", "Sex"]].head(5).to_dict("records")
            raise ValueError(f"VAE conditioning Sex has unsupported/missing values: {bad}")
        cols.append(encoded.to_numpy(dtype=np.float32))
    if vars_mode == "manufacturer":
        manufacturer = _normalize_conditioning_manufacturer(rows["Manufacturer"])
        missing = manufacturer.isna()
        categories = list(transformer.get("manufacturer_categories", []))
        mapping = dict(transformer.get("manufacturer_mapping", {}))
        unseen = ~manufacturer.isin(categories)
        bad_mask = missing | unseen
        if bad_mask.any():
            bad = rows.loc[bad_mask, ["SubjectID", "Manufacturer"]].head(10).to_dict("records")
            raise ValueError(
                "VAE conditioning Manufacturer has missing/unseen values for fold-local categories "
                f"{categories}: {bad}"
            )
        onehot = np.zeros((len(rows), len(categories)), dtype=np.float32)
        codes = manufacturer.map(mapping).to_numpy(dtype=int)
        onehot[np.arange(len(rows)), codes] = 1.0
        cols.append(onehot)
    cols_2d = [c[:, None] if c.ndim == 1 else c for c in cols]
    return np.concatenate(cols_2d, axis=1).astype(np.float32)


def apply_vae_conditioning_transformer(rows: pd.DataFrame, transformer: Dict[str, Any]) -> np.ndarray:
    """Backward-compatible alias for the fold-safe conditioning transform."""
    return transform_vae_conditioning_covariates(rows, transformer)


def summarize_vae_conditioning_qc(
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    transformer: Dict[str, Any],
) -> Dict[str, Any]:
    """Summarize fold-local imputation and missingness without fitting on val/test."""
    def sex_missing_count(rows: pd.DataFrame) -> int:
        return int(_normalize_conditioning_sex(rows["Sex"]).isna().sum()) if "Sex" in rows.columns else np.nan

    def age_missing_count(rows: pd.DataFrame) -> int:
        if "Age" not in rows.columns:
            return np.nan
        _, missing = _numeric_conditioning_age(rows["Age"])
        return int(missing.sum())

    def manufacturer_missing_count(rows: pd.DataFrame) -> int:
        return int(_normalize_conditioning_manufacturer(rows["Manufacturer"]).isna().sum()) if "Manufacturer" in rows.columns else np.nan

    def manufacturer_counts(rows: pd.DataFrame) -> Dict[str, int]:
        if "Manufacturer" not in rows.columns:
            return {}
        vals = _normalize_conditioning_manufacturer(rows["Manufacturer"])
        return {str(k): int(v) for k, v in vals.value_counts(dropna=False).to_dict().items()}

    return {
        "conditioning_vars": transformer.get("vars_mode", "none"),
        "sex_missing_train": sex_missing_count(train_rows),
        "sex_missing_val": sex_missing_count(val_rows),
        "sex_missing_test": sex_missing_count(test_rows),
        "sex_impute_value": transformer.get("sex_impute_value"),
        "age_missing_train": age_missing_count(train_rows),
        "age_missing_val": age_missing_count(val_rows),
        "age_missing_test": age_missing_count(test_rows),
        "age_impute_value": transformer.get("age_impute_value"),
        "age_mean_train": transformer.get("age_mean"),
        "age_std_train": transformer.get("age_std"),
        "manufacturer_missing_train": manufacturer_missing_count(train_rows),
        "manufacturer_missing_val": manufacturer_missing_count(val_rows),
        "manufacturer_missing_test": manufacturer_missing_count(test_rows),
        "manufacturer_categories": json.dumps(transformer.get("manufacturer_categories", [])),
        "manufacturer_counts_train": json.dumps(transformer.get("manufacturer_counts_train", {}), sort_keys=True),
        "manufacturer_counts_val": json.dumps(manufacturer_counts(val_rows), sort_keys=True),
        "manufacturer_counts_test": json.dumps(manufacturer_counts(test_rows), sort_keys=True),
    }


def get_cyclical_beta_schedule(current_epoch: int, total_epochs: int, beta_max: float, n_cycles: int, ratio_increase: float = 0.5) -> float:
    if n_cycles <= 0: return beta_max
    epoch_per_cycle = total_epochs / n_cycles
    epoch_in_current_cycle = current_epoch % epoch_per_cycle
    increase_phase_duration = epoch_per_cycle * ratio_increase
    return beta_max * (epoch_in_current_cycle / increase_phase_duration) if epoch_in_current_cycle < increase_phase_duration else beta_max





def train_and_evaluate_pipeline(global_tensor_all_channels: np.ndarray, 
                                metadata_df_full: pd.DataFrame,
                                args: argparse.Namespace):
    
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    
    if hasattr(args, 'channels_to_use') and args.channels_to_use is not None:
        selected_channel_indices = args.channels_to_use
        
        # Preferimos la lista maestra generada desde el tensor (C1),
        # pero si no está, caemos a DEFAULT y completamos según n_canales.
        master_channel_list = getattr(args, 'all_original_channel_names', DEFAULT_CHANNEL_NAMES)
        master_channel_list = _extend_channel_names_to_tensor(
            int(global_tensor_all_channels.shape[1]),
            list(master_channel_list),
        )
        
        try:
            # Intenta mapear los índices a los nombres usando la lista maestra
            selected_channel_names_in_tensor = [master_channel_list[i] for i in selected_channel_indices]
        except IndexError:
            logger.error(f"Error de índice al mapear nombres de canal. Se esperaba una lista de nombres de longitud {global_tensor_all_channels.shape[1]} pero se recibió una de longitud {len(master_channel_list)}. Usando nombres genéricos.")
            # Lógica de fallback por si acaso
            selected_channel_names_in_tensor = [ f"RawChan{i}" for i in selected_channel_indices ]

        current_global_tensor = global_tensor_all_channels[:, selected_channel_indices, :, :]
        logger.info(f"Usando canales seleccionados (índices): {selected_channel_indices}")
        logger.info(f"Nombres de canales seleccionados: {selected_channel_names_in_tensor}")

    else:
        # Lógica original si no se especifica `channels_to_use`
        current_global_tensor = global_tensor_all_channels
        selected_channel_indices = list(range(int(current_global_tensor.shape[1])))
        master_channel_list = getattr(args, 'all_original_channel_names', DEFAULT_CHANNEL_NAMES)
        master_channel_list = _extend_channel_names_to_tensor(
            int(current_global_tensor.shape[1]),
            list(master_channel_list),
        )
        selected_channel_names_in_tensor = [master_channel_list[i] if i < len(master_channel_list) else f"RawChan{i}" for i in range(current_global_tensor.shape[1])]
        logger.info(f"Usando todos los {current_global_tensor.shape[1]} canales.")
    
    
    num_input_channels_for_vae = current_global_tensor.shape[1]

    # --- Validación temprana de columnas críticas ---
    required_cols = ['ResearchGroup_Mapped', 'tensor_idx', 'SubjectID']
    if not all(col in metadata_df_full.columns for col in required_cols):
        logger.error(f"Faltan columnas críticas en metadatos: {[c for c in required_cols if c not in metadata_df_full.columns]}. Abortando.")
        return
    cn_ad_df = metadata_df_full[metadata_df_full['ResearchGroup_Mapped'].isin(['CN', 'AD'])].copy()
    if cn_ad_df.empty or 'tensor_idx' not in cn_ad_df.columns:
        logger.error("No hay sujetos CN/AD o falta 'tensor_idx' en el DataFrame mergeado. Abortando.")
        return
    
    max_valid_idx_for_cn_ad = current_global_tensor.shape[0] - 1
    original_cn_ad_count = len(cn_ad_df)
    cn_ad_df = cn_ad_df[cn_ad_df['tensor_idx'] <= max_valid_idx_for_cn_ad].copy()
    if len(cn_ad_df) < original_cn_ad_count:
        logger.warning(f"Algunos sujetos CN/AD filtrados porque 'tensor_idx' excede las dimensiones del tensor. "
                       f"Original: {original_cn_ad_count}, Post-filtro: {len(cn_ad_df)}")

    classifier_exclude_subject_ids = [
        str(s).strip()
        for s in (getattr(args, "classifier_exclude_subject_ids", None) or [])
        if str(s).strip()
    ]
    if classifier_exclude_subject_ids:
        exclude_set = set(classifier_exclude_subject_ids)
        _sid_series = cn_ad_df["SubjectID"].astype(str)
        _exclude_mask = _sid_series.isin(exclude_set)
        excluded_from_classifier_df = cn_ad_df.loc[_exclude_mask].copy()
        missing_exclude_ids = sorted(exclude_set.difference(set(_sid_series.tolist())))
        cn_ad_df = cn_ad_df.loc[~_exclude_mask].copy()
        audit_rows = []
        for _sid in sorted(exclude_set):
            _rows = excluded_from_classifier_df[excluded_from_classifier_df["SubjectID"].astype(str).eq(_sid)]
            if _rows.empty:
                audit_rows.append({
                    "SubjectID": _sid,
                    "action": "not_present_in_classifier_pool",
                    "ResearchGroup_Mapped": "",
                    "tensor_idx": "",
                })
            else:
                for _, _row in _rows.iterrows():
                    audit_rows.append({
                        "SubjectID": _sid,
                        "action": "excluded_from_classifier_pool_only",
                        "ResearchGroup_Mapped": str(_row.get("ResearchGroup_Mapped", "")),
                        "tensor_idx": _row.get("tensor_idx", ""),
                    })
        if audit_rows:
            _audit_path = Path(args.output_dir) / "classifier_excluded_subjects.csv"
            _audit_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(audit_rows).to_csv(_audit_path, index=False)
        logger.warning(
            "classifier_exclude_subject_ids activo: %d sujetos removidos sólo del pool supervisado AD/CN; "
            "ids solicitados=%s; ids no presentes=%s; pool restante=%d.",
            int(_exclude_mask.sum()),
            classifier_exclude_subject_ids,
            missing_exclude_ids,
            len(cn_ad_df),
        )

    if cn_ad_df.empty:
        logger.error("No hay sujetos CN/AD válidos después de filtrar por tensor_idx. Abortando.")
        return

    cn_ad_df['label'] = cn_ad_df['ResearchGroup_Mapped'].map({'CN': 0, 'AD': 1})
    
    strat_cols = ['ResearchGroup_Mapped']
    if args.classifier_stratify_cols:
        for col in args.classifier_stratify_cols:
            if col in cn_ad_df.columns:
                # Asegurarse de que las columnas de estratificación no tengan NaNs
                cn_ad_df[col] = cn_ad_df[col].fillna(f"{col}_Unknown").astype(str)
                if col not in strat_cols:
                    strat_cols.append(col)
            else:
                logger.warning(f"Columna de estratificación para el clasificador '{col}' no encontrada.")

    cn_ad_df['stratify_key_clf'] = cn_ad_df[strat_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    logger.info(f"Estratificando folds del CLASIFICADOR por: {strat_cols}")

    X_classifier_subject_indices_in_cn_ad_df = np.arange(len(cn_ad_df))
    y_classifier_labels_cn_ad = cn_ad_df['label'].values
    stratify_key_for_clf_cv = cn_ad_df['stratify_key_clf']

    # --- Guard rail: estratos demasiado chicos en OUTER-CV -> fallback a label-only ---
    y_outer = stratify_key_for_clf_cv
    try:
        vc_outer = pd.Series(y_outer).value_counts()
        if (vc_outer < args.outer_folds).any():
            logger.warning(
                f"Outer-CV: estratos muy chicos con {strat_cols} "
                f"(min={int(vc_outer.min())} < n_splits={args.outer_folds}). "
                f"Fallback a estratificación por label."
            )
            y_outer = y_classifier_labels_cn_ad
    except Exception as e:
        logger.warning(f"Outer-CV: no se pudo validar estratos ({e}). Fallback a label.")
        y_outer = y_classifier_labels_cn_ad
    
    logger.info(f"Sujetos CN/AD para clasificación: {len(cn_ad_df)}. CN: {sum(y_classifier_labels_cn_ad == 0)}, AD: {sum(y_classifier_labels_cn_ad == 1)}")

    if args.repeated_outer_folds_n_repeats > 1:
        outer_cv_clf = RepeatedStratifiedKFold(n_splits=args.outer_folds, n_repeats=args.repeated_outer_folds_n_repeats, random_state=args.seed)
        total_outer_iterations = args.outer_folds * args.repeated_outer_folds_n_repeats
    else:
        outer_cv_clf = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
        total_outer_iterations = args.outer_folds
    logger.info(f"Usando CV externa: {type(outer_cv_clf).__name__} con {total_outer_iterations} iteraciones totales.")
    selected_outer_fold_indices = None
    if args.outer_fold_indices_to_run:
        selected_outer_fold_indices = sorted({int(x) for x in args.outer_fold_indices_to_run})
        invalid_outer = [x for x in selected_outer_fold_indices if x < 1 or x > total_outer_iterations]
        if invalid_outer:
            raise ValueError(
                f"outer_fold_indices_to_run contiene folds inválidos {invalid_outer}; "
                f"rango permitido: 1..{total_outer_iterations}"
            )
        logger.info(
            "Modo diagnóstico: se ejecutarán sólo las iteraciones externas 1-based: "
            f"{selected_outer_fold_indices}. Los demás folds se saltarán sin crear artefactos."
        )

    all_folds_metrics = []
    all_folds_vae_history = []
    all_folds_clf_predictions = []

    for fold_idx, (train_dev_clf_idx_in_cn_ad_df, test_clf_idx_in_cn_ad_df) in enumerate(outer_cv_clf.split(X_classifier_subject_indices_in_cn_ad_df, y_outer)):
    #for fold_idx, (train_dev_clf_idx_in_cn_ad_df, test_clf_idx_in_cn_ad_df) in enumerate(outer_cv_clf.split(X_classifier_subject_indices_in_cn_ad_df, stratify_key_for_clf_cv)):
        fold_start_time = time.time()
        fold_idx_str = f"Fold {fold_idx + 1}/{total_outer_iterations}"
        if selected_outer_fold_indices is not None and (fold_idx + 1) not in selected_outer_fold_indices:
            logger.info(f"--- Saltando {fold_idx_str} por outer_fold_indices_to_run={selected_outer_fold_indices} ---")
            continue
        logger.info(f"--- Iniciando {fold_idx_str} ---")
        
        
        fold_output_dir = Path(args.output_dir) / f"fold_{fold_idx + 1}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        # =========================
        # Split artefacts (mínimos para reproducibilidad / inferencia)
        # - índices "locales" (cn_ad_df) para train/dev y test
        # - tensor_idx "global" (índices del tensor) para train/dev y test
        # =========================
        np.save(fold_output_dir / "train_dev_indices.npy", train_dev_clf_idx_in_cn_ad_df)

        np.save(fold_output_dir / "test_indices.npy", test_clf_idx_in_cn_ad_df)


        # tensor_idx debe ser int para indexar el tensor
        cn_ad_df["tensor_idx"] = cn_ad_df["tensor_idx"].astype(int)
        train_dev_df_fold = cn_ad_df.iloc[train_dev_clf_idx_in_cn_ad_df][["SubjectID","tensor_idx","ResearchGroup_Mapped"]]
        #train_dev_df_fold = cn_ad_df.iloc[train_dev_clf_idx_in_cn_ad_df][["SubjectID","tensor_idx","ResearchGroup_Mapped"]]
        train_dev_df_fold.to_csv(fold_output_dir / "train_dev_subjects_fold.csv", index=False)
        np.save(fold_output_dir / "train_dev_tensor_idx.npy", train_dev_df_fold["tensor_idx"].values)


        test_df_fold = cn_ad_df.iloc[test_clf_idx_in_cn_ad_df][["SubjectID","tensor_idx","ResearchGroup_Mapped"]]
        test_df_fold.to_csv(fold_output_dir / "test_subjects_fold.csv", index=False)
        np.save(fold_output_dir / "test_tensor_idx.npy", test_df_fold["tensor_idx"].values)


        global_indices_clf_test_this_fold = cn_ad_df.iloc[test_clf_idx_in_cn_ad_df]['tensor_idx'].values
        log_group_distributions(cn_ad_df.iloc[test_clf_idx_in_cn_ad_df], strat_cols, "Test Set (Clasificador)", fold_idx_str)

        # Asegurar dtype int + unicidad antes de setdiff (evita bugs silenciosos)
        all_valid_subject_indices_from_metadata = (
            metadata_df_full.loc[metadata_df_full["tensor_idx"] <= max_valid_idx_for_cn_ad, "tensor_idx"]
            .astype(int)
            .to_numpy()
        )
        global_indices_clf_test_this_fold = np.asarray(global_indices_clf_test_this_fold, dtype=int)
        global_indices_vae_training_pool = np.setdiff1d(
            np.unique(all_valid_subject_indices_from_metadata),
            np.unique(global_indices_clf_test_this_fold),
            assume_unique=False
        )

        if len(global_indices_vae_training_pool) < 10: 
            logger.error(f"{fold_idx_str}: Muy pocos sujetos ({len(global_indices_vae_training_pool)}) para entrenamiento VAE. Saltando fold.")
            continue

        vae_train_pool_df = (
            metadata_df_full
            .set_index("tensor_idx")
            .loc[global_indices_vae_training_pool]
            .reset_index()
        )    
        cols_pool_log = _filter_existing_cols(vae_train_pool_df, ['ResearchGroup_Mapped', 'Sex', 'Age_Group'])
        log_group_distributions(vae_train_pool_df, cols_pool_log, "Pool Entrenamiento VAE", fold_idx_str)

        # --- vae_required_metadata_cols: optional fold-safe VAE pool filter ---
        _req_meta = getattr(args, "vae_required_metadata_cols", None) or []
        _req_meta = [c for c in _req_meta if c]
        if _req_meta:
            _missing_any = pd.Series(False, index=vae_train_pool_df.index)
            for _col in _req_meta:
                if _col in vae_train_pool_df.columns:
                    _missing_any |= (
                        vae_train_pool_df[_col].isna()
                        | vae_train_pool_df[_col].astype(str).str.strip().isin(["", "nan", "NaN", "None", "none", "NA", "N/A"])
                    )
                else:
                    logger.warning(f"  {fold_idx_str} vae_required_metadata_cols: column {_col!r} not found in metadata; skipping.")
            if _missing_any.any():
                _removed = vae_train_pool_df[_missing_any].copy()
                _n_before = len(vae_train_pool_df)
                # distribution before filter
                _dist_before = {
                    _c: vae_train_pool_df[_c].value_counts(dropna=False).to_dict()
                    for _c in ["ResearchGroup_Mapped", "Manufacturer"] if _c in vae_train_pool_df.columns
                }
                vae_train_pool_df = vae_train_pool_df[~_missing_any].reset_index(drop=True)
                global_indices_vae_training_pool = vae_train_pool_df["tensor_idx"].to_numpy(dtype=int)
                _n_after = len(vae_train_pool_df)
                _dist_after = {
                    _c: vae_train_pool_df[_c].value_counts(dropna=False).to_dict()
                    for _c in ["ResearchGroup_Mapped", "Manufacturer"] if _c in vae_train_pool_df.columns
                }
                # Per-subject QC rows
                _qc_rows = []
                for _, _r in _removed.iterrows():
                    _missing_cols_for_subject = [
                        _c for _c in _req_meta
                        if _c in _r.index and (
                            pd.isna(_r[_c])
                            or str(_r[_c]).strip() in ["", "nan", "NaN", "None", "none", "NA", "N/A"]
                        )
                    ]
                    _sid = str(_r.get("SubjectID", "?"))
                    _in_clf = bool(_sid in cn_ad_df["SubjectID"].values) if "SubjectID" in cn_ad_df.columns else None
                    _qc_rows.append({
                        "fold": fold_idx + 1,
                        "SubjectID": _sid,
                        "missing_cols": "|".join(_missing_cols_for_subject),
                        "ResearchGroup_Mapped": str(_r.get("ResearchGroup_Mapped", "")),
                        "in_classifier_pool": _in_clf,
                    })
                if _qc_rows:
                    _qc_df = pd.DataFrame(_qc_rows)
                    _qc_path = fold_output_dir / f"vae_pool_required_metadata_removed_fold_{fold_idx+1}.csv"
                    _qc_df.to_csv(_qc_path, index=False)
                _filter_summary = {
                    "fold": fold_idx + 1,
                    "required_cols": _req_meta,
                    "n_vae_pool_before_filter": _n_before,
                    "n_removed": _n_before - _n_after,
                    "n_vae_pool_after_filter": _n_after,
                    "distribution_before": {str(k): {str(kk): int(vv) for kk, vv in v.items()} for k, v in _dist_before.items()},
                    "distribution_after": {str(k): {str(kk): int(vv) for kk, vv in v.items()} for k, v in _dist_after.items()},
                }
                _summary_path = fold_output_dir / f"vae_pool_required_metadata_filter_summary_fold_{fold_idx+1}.json"
                with open(_summary_path, "w", encoding="utf-8") as _fj:
                    json.dump(_filter_summary, _fj, indent=2)
                logger.warning(
                    f"  {fold_idx_str} VAE pool filtered for required metadata {_req_meta}: "
                    f"{_n_before - _n_after} subjects removed, {_n_after} remaining."
                )
            else:
                logger.info(
                    f"  {fold_idx_str} vae_required_metadata_cols {_req_meta}: "
                    f"all {len(vae_train_pool_df)} VAE pool subjects have required metadata present."
                )
        # --- end vae_required_metadata_cols ---

        _pool_strategy = str(getattr(args, "vae_pool_composition_strategy", VAE_POOL_CURRENT_ALL) or VAE_POOL_CURRENT_ALL)
        if _pool_strategy != VAE_POOL_CURRENT_ALL:
            _n_before_pool_strategy = len(vae_train_pool_df)
            vae_train_pool_df, _pool_strategy_summary = apply_vae_pool_composition_strategy(
                vae_train_pool_df,
                strategy=_pool_strategy,
                seed=args.seed,
                fold_number=fold_idx + 1,
            )
            global_indices_vae_training_pool = vae_train_pool_df["tensor_idx"].to_numpy(dtype=int)
            _pool_strategy_summary.to_csv(
                fold_output_dir / f"vae_pool_composition_strategy_summary_fold_{fold_idx+1}.csv",
                index=False,
            )
            logger.warning(
                f"  {fold_idx_str} Exploratory VAE pool composition strategy {_pool_strategy!r}: "
                f"{_n_before_pool_strategy} -> {len(vae_train_pool_df)} subjects. "
                "This uses diagnosis labels for VAE pool composition and is not the historical default."
            )
            log_group_distributions(
                vae_train_pool_df,
                _filter_existing_cols(vae_train_pool_df, ["ResearchGroup_Mapped", "Manufacturer", "Sex", "Age_Group"]),
                "Pool Entrenamiento VAE tras estrategia composicional",
                fold_idx_str,
            )

        if len(global_indices_vae_training_pool) < 10:
            logger.error(
                f"{fold_idx_str}: Muy pocos sujetos ({len(global_indices_vae_training_pool)}) "
                f"para entrenamiento VAE tras estrategia de composición. Saltando fold."
            )
            continue

        vae_train_pool_tensor_original_scale = current_global_tensor[global_indices_vae_training_pool]
        harmonized_clf_train_dev_tensor_original_scale = None
        harmonized_clf_test_tensor_original_scale = None
        input_harmonization_mode = str(getattr(args, "input_harmonization_mode", "none") or "none")
        if input_harmonization_mode != "none":
            if input_harmonization_mode != "foldwise_combat":
                raise RuntimeError(f"Unsupported input_harmonization_mode={input_harmonization_mode!r}")
            if fit_tensor_combat_channelwise is None or transform_tensor_combat_channelwise is None:
                raise RuntimeError(
                    "input_harmonization_mode=foldwise_combat requested, but foldwise ComBat helper import failed."
                )
            if len(selected_channel_indices) != 3 or list(selected_channel_indices) != [1, 0, 2]:
                raise RuntimeError(
                    "Branch B foldwise ComBat is currently restricted to selected channels [1,0,2]; "
                    f"got {list(selected_channel_indices)}"
                )
            batch_col = str(getattr(args, "input_harmonization_batch_col", "Manufacturer") or "Manufacturer")
            covariates = list(getattr(args, "input_harmonization_covariates", None) or ["Age", "Sex"])
            excluded_covariates = list(
                getattr(args, "input_harmonization_excluded_covariates", None) or ["ResearchGroup_Mapped"]
            )
            if str(getattr(args, "input_harmonization_fit_scope", "outer_train_dev_only")) != "outer_train_dev_only":
                raise RuntimeError("Foldwise ComBat only supports input_harmonization_fit_scope=outer_train_dev_only")
            if str(getattr(args, "input_harmonization_vectorization", "upper_offdiag_by_channel")) != "upper_offdiag_by_channel":
                raise RuntimeError("Foldwise ComBat only supports vectorization=upper_offdiag_by_channel")

            clf_train_dev_df_for_harmonization = cn_ad_df.iloc[train_dev_clf_idx_in_cn_ad_df].copy()
            clf_test_df_for_harmonization = cn_ad_df.iloc[test_clf_idx_in_cn_ad_df].copy()
            guard_row = _validate_foldwise_input_harmonization_guards(
                fit_df=vae_train_pool_df,
                test_df=clf_test_df_for_harmonization,
                batch_col=batch_col,
                covariates=covariates,
                excluded_covariates=excluded_covariates,
                fold_number=fold_idx + 1,
            )
            _safe_json_dump(guard_row, fold_output_dir / "input_harmonization_leakage_guard.json")
            pd.DataFrame([guard_row]).to_csv(fold_output_dir / "input_harmonization_leakage_guard.csv", index=False)
            _write_foldwise_input_harmonization_subjects(
                fold_output_dir,
                fold=fold_idx + 1,
                split_name="fit_train_dev_vae_pool",
                df=vae_train_pool_df,
            )
            _write_foldwise_input_harmonization_subjects(
                fold_output_dir,
                fold=fold_idx + 1,
                split_name="classifier_train_dev_apply",
                df=clf_train_dev_df_for_harmonization,
            )
            _write_foldwise_input_harmonization_subjects(
                fold_output_dir,
                fold=fold_idx + 1,
                split_name="classifier_test_apply",
                df=clf_test_df_for_harmonization,
            )

            logger.info(
                f"  {fold_idx_str} Input harmonization activo: foldwise_combat; "
                f"fit_scope=outer_train_dev_only, batch={batch_col}, covariates={covariates}, "
                f"excluded={excluded_covariates}, channels={selected_channel_indices}"
            )
            fitted_combat = fit_tensor_combat_channelwise(
                vae_train_pool_tensor_original_scale,
                vae_train_pool_df,
                channel_indices=list(selected_channel_indices),
                channel_names=selected_channel_names_in_tensor,
            )
            pd.DataFrame(fitted_combat.audit_rows).to_csv(
                fold_output_dir / "input_harmonization_fit_audit.csv",
                index=False,
            )
            pre_sep_fit = manufacturer_centroid_separability_proxy(
                vae_train_pool_tensor_original_scale,
                vae_train_pool_df,
                channel_names=selected_channel_names_in_tensor,
            ) if manufacturer_centroid_separability_proxy is not None else []

            vae_train_pool_tensor_harmonized = transform_tensor_combat_channelwise(
                fitted_combat,
                vae_train_pool_tensor_original_scale,
                vae_train_pool_df,
                preserve_diagonal=True,
            )
            global_indices_clf_train_dev_for_harmonization = clf_train_dev_df_for_harmonization["tensor_idx"].to_numpy(dtype=int)
            global_indices_clf_test_for_harmonization = clf_test_df_for_harmonization["tensor_idx"].to_numpy(dtype=int)
            clf_train_dev_tensor_original = current_global_tensor[global_indices_clf_train_dev_for_harmonization]
            clf_test_tensor_original = current_global_tensor[global_indices_clf_test_for_harmonization]
            harmonized_clf_train_dev_tensor_original_scale = transform_tensor_combat_channelwise(
                fitted_combat,
                clf_train_dev_tensor_original,
                clf_train_dev_df_for_harmonization,
                preserve_diagonal=True,
            )
            harmonized_clf_test_tensor_original_scale = transform_tensor_combat_channelwise(
                fitted_combat,
                clf_test_tensor_original,
                clf_test_df_for_harmonization,
                preserve_diagonal=True,
            )
            post_sep_fit = manufacturer_centroid_separability_proxy(
                vae_train_pool_tensor_harmonized,
                vae_train_pool_df,
                channel_names=selected_channel_names_in_tensor,
            ) if manufacturer_centroid_separability_proxy is not None else []
            sep_rows = []
            pre_by_channel = {r["channel_position"]: r for r in pre_sep_fit}
            post_by_channel = {r["channel_position"]: r for r in post_sep_fit}
            for cpos in sorted(set(pre_by_channel) | set(post_by_channel)):
                pre = pre_by_channel.get(cpos, {})
                post = post_by_channel.get(cpos, {})
                sep_rows.append({
                    "fold": fold_idx + 1,
                    "channel_position": int(cpos),
                    "channel_name": pre.get("channel_name", post.get("channel_name", "")),
                    "pre_mean_pairwise_centroid_distance_per_edge": pre.get("mean_pairwise_centroid_distance_per_edge", np.nan),
                    "post_mean_pairwise_centroid_distance_per_edge": post.get("mean_pairwise_centroid_distance_per_edge", np.nan),
                    "delta_mean_pairwise_centroid_distance_per_edge": (
                        post.get("mean_pairwise_centroid_distance_per_edge", np.nan)
                        - pre.get("mean_pairwise_centroid_distance_per_edge", np.nan)
                    ),
                })
            pd.DataFrame(sep_rows).to_csv(
                fold_output_dir / "input_harmonization_manufacturer_separability_proxy.csv",
                index=False,
            )
            shift_rows = []
            if combat_channel_shift_summary is not None:
                for split_name, before_tensor, after_tensor in [
                    ("vae_pool_fit_train_dev", vae_train_pool_tensor_original_scale, vae_train_pool_tensor_harmonized),
                    ("classifier_train_dev_apply", clf_train_dev_tensor_original, harmonized_clf_train_dev_tensor_original_scale),
                    ("classifier_test_apply", clf_test_tensor_original, harmonized_clf_test_tensor_original_scale),
                ]:
                    for row in combat_channel_shift_summary(
                        before_tensor,
                        after_tensor,
                        split_name=split_name,
                        channel_names=selected_channel_names_in_tensor,
                    ):
                        row["fold"] = fold_idx + 1
                        shift_rows.append(row)
            pd.DataFrame(shift_rows).to_csv(
                fold_output_dir / "input_harmonization_channel_shift_summary.csv",
                index=False,
            )
            diag_before = np.diagonal(vae_train_pool_tensor_original_scale, axis1=2, axis2=3)
            diag_after = np.diagonal(vae_train_pool_tensor_harmonized, axis1=2, axis2=3)
            symmetry_error = np.max(np.abs(vae_train_pool_tensor_harmonized - np.swapaxes(vae_train_pool_tensor_harmonized, 2, 3)))
            integrity_row = {
                "fold": fold_idx + 1,
                "status": "PASS",
                "fit_scope": "outer_train_dev_only",
                "n_fit_subjects": int(len(vae_train_pool_df)),
                "n_classifier_train_dev_apply": int(len(clf_train_dev_df_for_harmonization)),
                "n_classifier_test_apply": int(len(clf_test_df_for_harmonization)),
                "finite_fraction_vae_pool_post": float(np.isfinite(vae_train_pool_tensor_harmonized).mean()),
                "max_abs_diagonal_delta_vae_pool": float(np.max(np.abs(diag_after - diag_before))),
                "max_abs_symmetry_error_vae_pool": float(symmetry_error),
                "diagnosis_used_in_harmonizer": False,
                "test_distribution_used_in_fit": False,
                "global_combat": False,
            }
            pd.DataFrame([integrity_row]).to_csv(
                fold_output_dir / "input_harmonization_integrity.csv",
                index=False,
            )
            if not np.isfinite(vae_train_pool_tensor_harmonized).all():
                raise RuntimeError(f"{fold_idx_str} foldwise ComBat produced non-finite VAE pool values.")
            if not np.isfinite(harmonized_clf_train_dev_tensor_original_scale).all():
                raise RuntimeError(f"{fold_idx_str} foldwise ComBat produced non-finite classifier train/dev values.")
            if not np.isfinite(harmonized_clf_test_tensor_original_scale).all():
                raise RuntimeError(f"{fold_idx_str} foldwise ComBat produced non-finite classifier test values.")
            if float(integrity_row["max_abs_diagonal_delta_vae_pool"]) > 1e-10:
                raise RuntimeError(f"{fold_idx_str} foldwise ComBat changed diagonal values.")
            if float(integrity_row["max_abs_symmetry_error_vae_pool"]) > 1e-8:
                raise RuntimeError(f"{fold_idx_str} foldwise ComBat broke symmetry.")
            vae_train_pool_tensor_original_scale = vae_train_pool_tensor_harmonized
        
        # DEFAULT SEGURO:
        # - si vae_val_split_ratio == 0 -> train = todo el pool, val = vacío
        # - si ratio > 0 -> intentamos split de validación con fallbacks explícitos
        #   y registramos el modo real usado. Las corridas confirmatorias deben
        #   pasar --vae_abort_if_val_split_fails para impedir train sin val interno.
        pool_n = int(len(global_indices_vae_training_pool))
        vae_actual_train_indices_local_to_pool = np.arange(pool_n, dtype=int)
        vae_internal_val_indices_local_to_pool = np.array([], dtype=int)

        def _make_vae_split_stratify_key(candidate_cols):
            available_cols = [c for c in candidate_cols if c in vae_train_pool_df.columns]
            if len(available_cols) != len(candidate_cols):
                missing_cols = sorted(set(candidate_cols) - set(available_cols))
                return None, available_cols, f"missing columns: {missing_cols}"
            if not available_cols:
                return None, [], "unstratified"

            temp_vae_strat_df = vae_train_pool_df[available_cols].copy()
            for col in available_cols:
                temp_vae_strat_df[col] = temp_vae_strat_df[col].fillna(f"{col}_Unknown").astype(str)
            stratify_key = temp_vae_strat_df.apply(lambda x: '_'.join(x.values.astype(str)), axis=1)
            counts = stratify_key.value_counts(dropna=False)
            min_count = int(counts.min()) if len(counts) else 0
            if min_count < 2:
                return None, available_cols, f"singleton strata present (min_count={min_count})"
            return stratify_key, available_cols, "ok"

        if args.vae_val_split_ratio > 0 and len(global_indices_vae_training_pool) > 10:
            vae_split_candidates = [
                ("ResearchGroup_Mapped+Manufacturer", ["ResearchGroup_Mapped", "Manufacturer"]),
                ("ResearchGroup_Mapped", ["ResearchGroup_Mapped"]),
                ("Manufacturer", ["Manufacturer"]),
                ("unstratified", []),
            ]
            split_errors = []
            split_success = False
            for split_mode, split_cols in vae_split_candidates:
                stratify_key_vae_split = None
                if split_cols:
                    stratify_key_vae_split, available_cols, reason = _make_vae_split_stratify_key(split_cols)
                    if stratify_key_vae_split is None:
                        msg = f"{split_mode} skipped ({reason})"
                        split_errors.append(msg)
                        logger.warning(f"  {fold_idx_str} VAE val split fallback: {msg}.")
                        continue
                try:
                    vae_actual_train_indices_local_to_pool, vae_internal_val_indices_local_to_pool = sk_train_test_split(
                        np.arange(len(global_indices_vae_training_pool)),
                        test_size=args.vae_val_split_ratio,
                        stratify=stratify_key_vae_split,
                        random_state=args.seed + fold_idx + 10, shuffle=True
                    )
                    vae_actual_train_indices_local_to_pool = np.asarray(vae_actual_train_indices_local_to_pool, dtype=int)
                    vae_internal_val_indices_local_to_pool = np.asarray(vae_internal_val_indices_local_to_pool, dtype=int)
                    logger.info(f"  {fold_idx_str} VAE val split mode used: {split_mode}")
                    split_success = True
                    break
                except ValueError as e:
                    msg = f"{split_mode} failed ({e})"
                    split_errors.append(msg)
                    logger.warning(f"  {fold_idx_str} VAE val split fallback: {msg}.")

            if not split_success:
                if getattr(args, "vae_abort_if_val_split_fails", False):
                    raise RuntimeError(
                        f"{fold_idx_str} VAE internal validation split failed and "
                        "--vae_abort_if_val_split_fails is set. "
                        f"Attempts: {' | '.join(split_errors)}. "
                        "Check for subjects with missing stratification metadata in the VAE pool "
                        "(e.g. tensor subjects absent from metadata). "
                        "Use --vae_required_metadata_cols to remove such subjects before splitting."
                    )
                logger.error(
                    f"  {fold_idx_str} Error al hacer el split de validación del VAE. "
                    f"Attempts: {' | '.join(split_errors)}. Usando todo el pool como train."
                )
                vae_actual_train_indices_local_to_pool = np.arange(len(global_indices_vae_training_pool), dtype=int)
                vae_internal_val_indices_local_to_pool = np.array([], dtype=int)

        # Guardar splits VAE (C2)
        try:
            np.save(fold_output_dir / "vae_training_pool_tensor_idx.npy", np.asarray(global_indices_vae_training_pool, dtype=int))
            np.save(fold_output_dir / "vae_actual_train_idx_local_to_pool.npy", np.asarray(vae_actual_train_indices_local_to_pool, dtype=int))
            np.save(fold_output_dir / "vae_internal_val_idx_local_to_pool.npy", np.asarray(vae_internal_val_indices_local_to_pool, dtype=int))
        except Exception as e:
            logger.warning(f"  {fold_idx_str} No se pudieron guardar splits VAE (pool/train/val): {e}")

        # Logs distribuciones (filtrando columnas existentes)
        cols_vae_log = _filter_existing_cols(vae_train_pool_df, ['ResearchGroup_Mapped', 'Sex', 'Age_Group'])
        log_group_distributions(vae_train_pool_df.iloc[vae_actual_train_indices_local_to_pool], cols_vae_log, "Actual Train Set (VAE)", fold_idx_str)
        if len(vae_internal_val_indices_local_to_pool) > 0:
            log_group_distributions(vae_train_pool_df.iloc[vae_internal_val_indices_local_to_pool], cols_vae_log, "Internal Val Set (VAE)", fold_idx_str)
        logger.info(f"  {fold_idx_str} Sujetos VAE actual train: {len(vae_actual_train_indices_local_to_pool)}, VAE internal val: {len(vae_internal_val_indices_local_to_pool)}")

        vae_pool_tensor_norm, norm_params_fold_list = normalize_inter_channel_fold(
            vae_train_pool_tensor_original_scale, vae_actual_train_indices_local_to_pool, 
            mode=args.norm_mode, selected_channel_original_names=selected_channel_names_in_tensor
        )

        joblib.dump(norm_params_fold_list, fold_output_dir / "vae_norm_params.joblib")
        vae_conditioning_enabled = str(args.vae_conditioning_mode) != "none"
        vae_conditioning_transformer = fit_vae_conditioning_transformer(
            vae_train_pool_df.iloc[vae_actual_train_indices_local_to_pool].copy(),
            str(args.vae_conditioning_vars) if vae_conditioning_enabled else "none",
        )
        joblib.dump(vae_conditioning_transformer, fold_output_dir / "vae_conditioning_transformer.joblib")
        vae_pool_conditioning = apply_vae_conditioning_transformer(vae_train_pool_df, vae_conditioning_transformer)
        vae_conditioning_dim = int(vae_pool_conditioning.shape[1])
        if vae_conditioning_enabled:
            conditioning_qc = summarize_vae_conditioning_qc(
                train_rows=vae_train_pool_df.iloc[vae_actual_train_indices_local_to_pool].copy(),
                val_rows=vae_train_pool_df.iloc[vae_internal_val_indices_local_to_pool].copy(),
                test_rows=cn_ad_df.iloc[test_clf_idx_in_cn_ad_df].copy(),
                transformer=vae_conditioning_transformer,
            )
            pd.DataFrame([{
                "fold": fold_idx + 1,
                "vae_conditioning_mode": args.vae_conditioning_mode,
                "vae_conditioning_vars": args.vae_conditioning_vars,
                "conditioning_dim": vae_conditioning_dim,
                "age_mean_train": vae_conditioning_transformer.get("age_mean"),
                "age_std_train": vae_conditioning_transformer.get("age_std"),
                "age_impute_value": vae_conditioning_transformer.get("age_impute_value"),
                "sex_impute_value": vae_conditioning_transformer.get("sex_impute_value"),
                "sex_counts_train": json.dumps(vae_conditioning_transformer.get("sex_counts_train", {}), sort_keys=True),
                "sex_mapping": json.dumps(vae_conditioning_transformer.get("sex_mapping", {}), sort_keys=True),
                "manufacturer_categories": json.dumps(vae_conditioning_transformer.get("manufacturer_categories", [])),
                "manufacturer_counts_train": json.dumps(vae_conditioning_transformer.get("manufacturer_counts_train", {}), sort_keys=True),
                "fit_scope": "vae_actual_train_only",
                **conditioning_qc,
            }]).to_csv(fold_output_dir / "vae_conditioning_summary.csv", index=False)
            pd.DataFrame([conditioning_qc]).to_csv(fold_output_dir / "vae_conditioning_missingness_qc.csv", index=False)
            logger.info(
                f"  {fold_idx_str} VAE conditioning activo: mode={args.vae_conditioning_mode}, "
                f"vars={args.vae_conditioning_vars}, dim={vae_conditioning_dim}, "
                f"corr_lambda={args.vae_latent_covariate_corr_lambda}, "
                f"sex_impute={vae_conditioning_transformer.get('sex_impute_value')}, "
                f"age_impute={vae_conditioning_transformer.get('age_impute_value')}, "
                f"manufacturer_categories={vae_conditioning_transformer.get('manufacturer_categories')}"
            )
        if vae_conditioning_enabled:
            vae_train_dataset = TensorDataset(
                torch.from_numpy(vae_pool_tensor_norm[vae_actual_train_indices_local_to_pool]).float(),
                torch.from_numpy(vae_pool_conditioning[vae_actual_train_indices_local_to_pool]).float(),
            )
        else:
            vae_train_dataset = TensorDataset(torch.from_numpy(vae_pool_tensor_norm[vae_actual_train_indices_local_to_pool]).float())
        vae_sampler = None
        vae_shuffle = True
        if args.vae_train_sampler_strategy != VAE_TRAIN_SAMPLER_NONE:
            try:
                sampler_rows = vae_train_pool_df.iloc[vae_actual_train_indices_local_to_pool].copy()
                vae_sampler, sampler_summary = make_vae_weighted_sampler(
                    sampler_rows,
                    strategy=args.vae_train_sampler_strategy,
                    seed=args.seed + fold_idx + 1000,
                )
                vae_shuffle = False
                if not sampler_summary.empty:
                    sampler_summary.to_csv(fold_output_dir / f"vae_train_sampler_summary_fold_{fold_idx+1}.csv", index=False)
                logger.info(
                    f"  {fold_idx_str} VAE train sampler activo: {args.vae_train_sampler_strategy}; "
                    f"replacement=True, n_samples={len(vae_train_dataset)}"
                )
            except Exception as e:
                logger.error(f"  {fold_idx_str} No se pudo construir VAE sampler {args.vae_train_sampler_strategy}: {e}")
                raise
        vae_train_loader = DataLoader(
            vae_train_dataset,
            batch_size=args.batch_size,
            shuffle=vae_shuffle,
            sampler=vae_sampler,
            num_workers=args.num_workers,
            pin_memory=bool(torch.cuda.is_available())
        )
        vae_internal_val_loader = None
        if len(vae_internal_val_indices_local_to_pool) > 0:
            if vae_conditioning_enabled:
                vae_internal_val_dataset = TensorDataset(
                    torch.from_numpy(vae_pool_tensor_norm[vae_internal_val_indices_local_to_pool]).float(),
                    torch.from_numpy(vae_pool_conditioning[vae_internal_val_indices_local_to_pool]).float(),
                )
            else:
                vae_internal_val_dataset = TensorDataset(torch.from_numpy(vae_pool_tensor_norm[vae_internal_val_indices_local_to_pool]).float())
            vae_internal_val_loader = DataLoader(
                vae_internal_val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=bool(torch.cuda.is_available())
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"  {fold_idx_str} Usando dispositivo: {device}")
        
        vae_fold_k = ConvolutionalVAE(
            input_channels=num_input_channels_for_vae, latent_dim=args.latent_dim, image_size=current_global_tensor.shape[-1],
            final_activation=args.vae_final_activation, intermediate_fc_dim_config=args.intermediate_fc_dim_vae,
            dropout_rate=args.dropout_rate_vae,
            encoder_dropout_rate=args.encoder_dropout_rate_vae,
            decoder_dropout_rate=args.decoder_dropout_rate_vae,
            use_layernorm_fc=args.use_layernorm_vae_fc,
            num_conv_layers_encoder=args.num_conv_layers_encoder, decoder_type=args.decoder_type,
            encoder_norm_mode=args.vae_encoder_norm_mode,
            dropout_scope=args.vae_dropout_scope,
            block_order=args.vae_block_order,
            conditioning_mode=args.vae_conditioning_mode,
            conditioning_dim=vae_conditioning_dim,
        ).to(device)
        dropout_manifest_rows = build_vae_dropout_manifest(vae_fold_k)
        dropout_summary_rows = summarize_vae_dropout_manifest(
            dropout_manifest_rows,
            dropout_scope=args.vae_dropout_scope,
            dropout_rate=args.dropout_rate_vae,
            num_conv_layers_encoder=args.num_conv_layers_encoder,
            has_intermediate_fc=bool(getattr(vae_fold_k, "intermediate_fc_dim", 0)),
            encoder_dropout_rate=args.encoder_dropout_rate_vae,
            decoder_dropout_rate=args.decoder_dropout_rate_vae,
        )
        dropout_manifest_path = fold_output_dir / "dropout_manifest.csv"
        dropout_manifest_json_path = fold_output_dir / "dropout_manifest.json"
        dropout_summary_path = fold_output_dir / "dropout_summary.csv"
        pd.DataFrame(dropout_manifest_rows).to_csv(dropout_manifest_path, index=False)
        pd.DataFrame(dropout_summary_rows).to_csv(dropout_summary_path, index=False)
        with open(dropout_manifest_json_path, "w", encoding="utf-8") as f_dropout_manifest:
            json.dump(
                {
                    "fold": fold_idx + 1,
                    "dropout_rate_vae": args.dropout_rate_vae,
                    "encoder_dropout_rate_vae": args.encoder_dropout_rate_vae,
                    "decoder_dropout_rate_vae": args.decoder_dropout_rate_vae,
                    "vae_dropout_scope": args.vae_dropout_scope,
                    "manifest": dropout_manifest_rows,
                    "summary": dropout_summary_rows,
                },
                f_dropout_manifest,
                indent=2,
            )
        logger.info(
            f"  {fold_idx_str} Dropout manifest guardado: "
            f"{dropout_manifest_path}, {dropout_manifest_json_path}, {dropout_summary_path}"
        )
        n_rois_for_loss = int(current_global_tensor.shape[-1])
        offdiag_elements_for_loss = int(n_rois_for_loss * (n_rois_for_loss - 1))
        logger.info(
            f"  {fold_idx_str} VAE objective: recon_loss_mode={args.recon_loss_mode}, "
            f"final_activation={args.vae_final_activation}, encoder_norm_mode={args.vae_encoder_norm_mode}, "
            f"dropout_scope={args.vae_dropout_scope}, dropout_rate={args.dropout_rate_vae}, "
            f"encoder_dropout_rate={args.encoder_dropout_rate_vae}, decoder_dropout_rate={args.decoder_dropout_rate_vae}, "
            f"block_order={args.vae_block_order}, "
            f"conditioning_mode={args.vae_conditioning_mode}, conditioning_vars={args.vae_conditioning_vars}, "
            f"conditioning_dim={vae_conditioning_dim}, corr_lambda={args.vae_latent_covariate_corr_lambda}, "
            f"channel_dropout_p={args.vae_channel_dropout_p}, "
            f"input_harmonization_mode={args.input_harmonization_mode}, "
            f"input_channels={num_input_channels_for_vae}, "
            f"n_rois={n_rois_for_loss}, offdiag_elements_per_channel={offdiag_elements_for_loss}, "
            f"recon_loss_channel_weights={args.recon_loss_channel_weights}, "
            f"reconstruction_loss_scale={describe_recon_loss_mode(args.recon_loss_mode, num_input_channels_for_vae, n_rois_for_loss, args.recon_loss_channel_weights)}"
        )
        
        optimizer_vae = optim.AdamW(vae_fold_k.parameters(), lr=args.lr_vae, weight_decay=args.weight_decay_vae, amsgrad=True)
        scheduler_vae = None

        if vae_internal_val_loader:
            if args.lr_scheduler_type == 'plateau':
                if args.lr_scheduler_patience_vae > 0:
                    logger.info(f"  {fold_idx_str} Usando scheduler: ReduceLROnPlateau")
                    scheduler_vae = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer_vae, 'min', 
                        patience=args.lr_scheduler_patience_vae, 
                        factor=0.1
                    )
            elif args.lr_scheduler_type == 'cosine_warm':
                logger.info(f"  {fold_idx_str} Usando scheduler: CosineAnnealingWarmRestarts (T_0={args.lr_scheduler_T0})")
                scheduler_vae = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer_vae, 
                    T_0=args.lr_scheduler_T0, 
                    eta_min=args.lr_scheduler_eta_min
                )
        # ... (código previo de inicialización del VAE) ...
        logger.info(f"  {fold_idx_str} Entrenando VAE (Decoder: {args.decoder_type}, Encoder Layers: {args.num_conv_layers_encoder})...")
        best_val_loss = float('inf')
        best_epoch = 0
        epochs_no_improve = 0
        best_model_state_dict = None
        periodic_vae_checkpoint_rows: List[Dict[str, Any]] = []
        periodic_vae_checkpoint_dir: Optional[Path] = None
        checkpoint_start_epoch = int(args.save_vae_checkpoints_start_epoch or 0)
        legacy_checkpoint_cadence_options = (
            args.save_vae_checkpoints_start_epoch is None
            and args.save_vae_checkpoints_keep_last_n is None
            and not bool(args.save_vae_checkpoints_always_keep_best)
            and not bool(args.save_vae_checkpoints_always_keep_final)
        )
        final_epoch_seen = 0
        if args.save_vae_checkpoints_every_n_epochs is not None:
            periodic_vae_checkpoint_dir = fold_output_dir / "vae_checkpoints"
            periodic_vae_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"  {fold_idx_str} Guardado periódico de checkpoints VAE activado: "
                f"cada {args.save_vae_checkpoints_every_n_epochs} épocas, "
                f"desde epoch {checkpoint_start_epoch} -> {periodic_vae_checkpoint_dir}"
            )

        def _history_value_at_epoch(key: str, epoch_num: int) -> Optional[float]:
            values = history_data.get(key, [])
            if epoch_num <= 0 or epoch_num > len(values):
                return None
            value = values[epoch_num - 1]
            try:
                if np.isnan(value):
                    return None
            except TypeError:
                pass
            return float(value)

        def _checkpoint_row(
            checkpoint_path: Path,
            checkpoint_epoch: int,
            checkpoint_type: str,
            current_beta_value: Optional[float],
            protected: bool = False,
        ) -> Dict[str, Any]:
            return {
                "fold": int(fold_idx + 1),
                "epoch": int(checkpoint_epoch),
                "path": str(checkpoint_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_type": str(checkpoint_type),
                "checkpoint_reason": str(checkpoint_type),
                "current_beta": float(current_beta_value) if current_beta_value is not None else None,
                "train_loss": _history_value_at_epoch("train_loss", checkpoint_epoch),
                "train_recon": _history_value_at_epoch("train_recon", checkpoint_epoch),
                "train_kld": _history_value_at_epoch("train_kld", checkpoint_epoch),
                "train_latent_covariate_corr": _history_value_at_epoch("train_latent_covariate_corr", checkpoint_epoch),
                "val_loss": _history_value_at_epoch("val_loss", checkpoint_epoch),
                "val_recon": _history_value_at_epoch("val_recon", checkpoint_epoch),
                "val_kld": _history_value_at_epoch("val_kld", checkpoint_epoch),
                "val_latent_covariate_corr": _history_value_at_epoch("val_latent_covariate_corr", checkpoint_epoch),
                "val_loss_beta_max": _history_value_at_epoch("val_loss_modelsel", checkpoint_epoch),
                "val_loss_modelsel": _history_value_at_epoch("val_loss_modelsel", checkpoint_epoch),
                "best_epoch_so_far": int(best_epoch),
                "best_val_loss_modelsel_so_far": float(best_val_loss) if np.isfinite(best_val_loss) else None,
                "protected_from_pruning": bool(protected),
                "status": "kept",
                "kept_deleted_status": "kept",
                "deleted_utc": None,
            }

        def _prune_periodic_vae_checkpoints() -> None:
            keep_last_n = args.save_vae_checkpoints_keep_last_n
            if keep_last_n is None:
                return
            keep_last_n = int(keep_last_n)
            kept_periodic_rows = [
                row for row in periodic_vae_checkpoint_rows
                if row.get("status") == "kept" and row.get("checkpoint_type") == "periodic"
            ]
            if len(kept_periodic_rows) <= keep_last_n:
                return
            kept_periodic_rows = sorted(kept_periodic_rows, key=lambda row: int(row.get("epoch") or 0))
            rows_to_delete = kept_periodic_rows[:-keep_last_n]
            for row in rows_to_delete:
                if row.get("protected_from_pruning"):
                    continue
                checkpoint_path = Path(str(row.get("path")))
                try:
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                    row["status"] = "deleted"
                    row["kept_deleted_status"] = "deleted"
                    row["deleted_utc"] = datetime.now(timezone.utc).isoformat()
                    logger.info(f"  {fold_idx_str} Checkpoint VAE periódico eliminado por keep_last_n: {checkpoint_path}")
                except Exception as e:
                    row["status"] = "delete_failed"
                    row["kept_deleted_status"] = "delete_failed"
                    row["delete_error"] = str(e)
                    logger.warning(f"  {fold_idx_str} No se pudo eliminar checkpoint periódico {checkpoint_path}: {e}")
        
        # Listas para guardar el historial completo
        history_data = {
            # train_* usan el beta actual de esa época
            "train_loss": [], "train_recon": [], "train_kld": [],
            "val_loss": [], "val_recon": [], "val_kld": [],
            "val_loss_modelsel": [],
            "train_latent_covariate_corr": [], "val_latent_covariate_corr": [],

            # para debug
            "beta": [],
            "train_kld_over_recon": [], "train_beta_kld_over_recon": [],
            "val_kld_over_recon": [], "val_beta_kld_over_recon": []
        }

        scaler = GradScaler(enabled=(device.type == 'cuda'))

        for epoch in range(args.epochs_vae):
            should_stop_vae = False
            vae_fold_k.train()
            # Acumuladores para la época de entrenamiento
            epoch_train_loss, epoch_train_recon, epoch_train_kld, epoch_train_corr = 0.0, 0.0, 0.0, 0.0
            current_beta = get_cyclical_beta_schedule(
                current_epoch=epoch,
                total_epochs=args.epochs_vae,
                beta_max=args.beta_vae,
                n_cycles=args.cyclical_beta_n_cycles,
                ratio_increase=args.cyclical_beta_ratio_increase
            )

            for i, batch in enumerate(vae_train_loader):
                data = batch[0].to(device)
                cond_batch = batch[1].to(device) if vae_conditioning_enabled else None
                optimizer_vae.zero_grad(set_to_none=True)

                with autocast(enabled=(device.type == 'cuda')):
                    vae_input = apply_channel_dropout_train(data, args.vae_channel_dropout_p)
                    recon_batch, mu, logvar, _ = vae_fold_k(vae_input, condition=cond_batch)
                    loss, recon, kld, corr_penalty = vae_loss_function(
                        recon_batch, data, mu, logvar,
                        beta=current_beta,
                        recon_loss_mode=args.recon_loss_mode,
                        recon_loss_channel_weights=args.recon_loss_channel_weights,
                        condition=cond_batch,
                        latent_covariate_corr_lambda=args.vae_latent_covariate_corr_lambda,
                    )
                
                scaler.scale(loss).backward()
                scaler.step(optimizer_vae)
                scaler.update()

                # Actualizar el scheduler de coseno en cada paso
                if scheduler_vae and args.lr_scheduler_type == 'cosine_warm':
                    scheduler_vae.step(epoch + i / len(vae_train_loader)) # Actualización por paso

                epoch_train_loss += loss.item() * data.size(0)
                epoch_train_recon += recon.item() * data.size(0)
                epoch_train_kld += kld.item() * data.size(0)
                epoch_train_corr += corr_penalty.item() * data.size(0)
            # ▲▲▲ FIN BUCLE DE ENTRENAMIENTO MODIFICADO ▲▲▲
            
            # Calculamos las medias y las guardamos en el historial
            history_data["train_loss"].append(epoch_train_loss / len(vae_train_loader.dataset))
            history_data["train_recon"].append(epoch_train_recon / len(vae_train_loader.dataset))
            history_data["train_kld"].append(epoch_train_kld / len(vae_train_loader.dataset))
            history_data["train_latent_covariate_corr"].append(epoch_train_corr / len(vae_train_loader.dataset))
            train_kld_over_recon = (
                history_data["train_kld"][-1] / history_data["train_recon"][-1]
                if history_data["train_recon"][-1] else np.nan
            )
            train_beta_kld_over_recon = (
                current_beta * history_data["train_kld"][-1] / history_data["train_recon"][-1]
                if history_data["train_recon"][-1] else np.nan
            )
            history_data["train_kld_over_recon"].append(train_kld_over_recon)
            history_data["train_beta_kld_over_recon"].append(train_beta_kld_over_recon)
            history_data["beta"].append(current_beta)
            
            log_msg = (f"  {fold_idx_str} FOLD{fold_idx+1}: E{epoch+1}/{args.epochs_vae}, "
                       f"TrL(curβ): {history_data['train_loss'][-1]:.2f} "
                       f"(R: {history_data['train_recon'][-1]:.2f}, "
                       f"KLD: {history_data['train_kld'][-1]:.2f}), "
                       f"Corr: {history_data['train_latent_covariate_corr'][-1]:.4f}, "
                       f"KLD/R={train_kld_over_recon:.4f}, βKLD/R={train_beta_kld_over_recon:.4f}, "
                       f"β={current_beta:.3f}, LR={optimizer_vae.param_groups[0]['lr']:.2e}")


            if vae_internal_val_loader:
                vae_fold_k.eval()
                # Acumuladores para la época de validación
                epoch_val_loss_curBeta, epoch_val_recon, epoch_val_kld, epoch_val_corr = 0.0, 0.0, 0.0, 0.0
                with torch.no_grad():
                    with autocast(enabled=(device.type == 'cuda')):
                        for val_batch in vae_internal_val_loader:
                            val_data = val_batch[0].to(device)
                            val_cond = val_batch[1].to(device) if vae_conditioning_enabled else None
                            recon_val, mu_val, logvar_val, _ = vae_fold_k(val_data, condition=val_cond)
                            # forward con el beta actual (lo que realmente entrenamos esta época)
                            v_loss_curBeta, v_recon, v_kld, v_corr = vae_loss_function(
                                recon_val, val_data, mu_val, logvar_val,
                                beta=current_beta,
                                recon_loss_mode=args.recon_loss_mode,
                                recon_loss_channel_weights=args.recon_loss_channel_weights,
                                condition=val_cond,
                                latent_covariate_corr_lambda=args.vae_latent_covariate_corr_lambda,
                            )

                            epoch_val_loss_curBeta += v_loss_curBeta.item() * val_data.size(0)
                            epoch_val_recon       += v_recon.item()       * val_data.size(0)
                            epoch_val_kld += v_kld.item() * val_data.size(0)
                            epoch_val_corr += v_corr.item() * val_data.size(0)

                # --- MÉTRICAS DE VALIDACIÓN ---
                N_val = len(vae_internal_val_loader.dataset)

                # promedio usando el beta ACTUAL de esta época (solo para log)
                avg_val_loss_curBeta = epoch_val_loss_curBeta / N_val

               # promedios puros de los componentes
                avg_val_recon = epoch_val_recon / N_val
                avg_val_kld   = epoch_val_kld   / N_val
                avg_val_corr = epoch_val_corr / N_val
                val_kld_over_recon = avg_val_kld / avg_val_recon if avg_val_recon else np.nan
                val_beta_kld_over_recon = current_beta * avg_val_kld / avg_val_recon if avg_val_recon else np.nan

                # métrica CONSISTENTE entre épocas:
                # simulamos "cómo le iría" al modelo si usáramos siempre beta_max (= args.beta_vae)
                avg_val_loss_betaMax = avg_val_recon + args.beta_vae * avg_val_kld

                # guardamos todo en history
                history_data["val_loss"].append(avg_val_loss_curBeta)
                history_data["val_recon"].append(avg_val_recon)
                history_data["val_kld"].append(avg_val_kld)
                history_data["val_loss_modelsel"].append(avg_val_loss_betaMax)
                history_data["val_latent_covariate_corr"].append(avg_val_corr)
                history_data["val_kld_over_recon"].append(val_kld_over_recon)
                history_data["val_beta_kld_over_recon"].append(val_beta_kld_over_recon)

                log_msg += (
                    f", ValL(curβ): {avg_val_loss_curBeta:.2f} "
                    f"(R: {avg_val_recon:.2f}, KLD: {avg_val_kld:.2f}, "
                    f"Corr: {avg_val_corr:.4f}, "
                    f"KLD/R={val_kld_over_recon:.4f}, βKLD/R={val_beta_kld_over_recon:.4f}) "
                    f"| ValL(βmax): {avg_val_loss_betaMax:.2f}"
                )

                score_for_scheduler = avg_val_loss_betaMax

                old_lr = optimizer_vae.param_groups[0]['lr']
                if scheduler_vae and args.lr_scheduler_type == 'plateau':
                    scheduler_vae.step(score_for_scheduler)

                new_lr = optimizer_vae.param_groups[0]['lr']
                if new_lr < old_lr:
                    logger.info(
                        f"  {fold_idx_str} LR reducido a {new_lr:.2e} -> reseteo paciencia early stopping."
                    )
                    epochs_no_improve = 0

                # --- early stopping / best checkpoint con beta_max constante ---
                if score_for_scheduler < best_val_loss and not np.isnan(score_for_scheduler):
                    best_val_loss = score_for_scheduler
                    best_epoch = epoch + 1
                    epochs_no_improve = 0
                    best_model_state_dict = copy.deepcopy(vae_fold_k.state_dict())
                else:
                    epochs_no_improve += 1

                if args.early_stopping_patience_vae > 0 and epochs_no_improve >= args.early_stopping_patience_vae:
                    logger.info(
                        f"  {fold_idx_str} Early stopping VAE en epoch {epoch+1}. "
                        f"Mejor ValL(βmax): {best_val_loss:.4f} (época {best_epoch})"
                    )
                    should_stop_vae = True
            else: 
                # Sin validación -> rellenamos NaN para mantener forma
                for key in ["val_loss", "val_recon", "val_kld", "val_loss_modelsel", "val_kld_over_recon", "val_beta_kld_over_recon"]:
                    history_data[key].append(np.nan)
                best_model_state_dict = copy.deepcopy(vae_fold_k.state_dict())

            if (epoch + 1) % args.log_interval_epochs_vae == 0 or epoch == args.epochs_vae - 1:
                logger.info(log_msg)

            final_epoch_seen = epoch + 1
            if periodic_vae_checkpoint_dir is not None:
                checkpoint_interval = int(args.save_vae_checkpoints_every_n_epochs)
                epoch_num = epoch + 1
                is_terminal_epoch = (epoch == args.epochs_vae - 1) or should_stop_vae
                is_interval_epoch = epoch_num >= checkpoint_start_epoch and (epoch_num % checkpoint_interval == 0)
                save_periodic = (
                    is_interval_epoch
                    or is_terminal_epoch
                )
                if save_periodic:
                    checkpoint_type = "early_stop" if should_stop_vae else ("final" if epoch == args.epochs_vae - 1 else "periodic")
                    protected = (
                        (checkpoint_type == "final" and bool(args.save_vae_checkpoints_always_keep_final))
                        or (checkpoint_type == "early_stop" and bool(args.save_vae_checkpoints_always_keep_final))
                    )
                    if legacy_checkpoint_cadence_options:
                        checkpoint_path = periodic_vae_checkpoint_dir / f"vae_checkpoint_fold_{fold_idx+1}_epoch_{epoch_num:04d}.pt"
                    else:
                        checkpoint_path = periodic_vae_checkpoint_dir / f"vae_checkpoint_fold_{fold_idx+1}_{checkpoint_type}_epoch_{epoch_num:04d}.pt"
                    checkpoint_payload = {
                        "model_state_dict": {k: v.detach().cpu() for k, v in vae_fold_k.state_dict().items()},
                        "fold": int(fold_idx + 1),
                        "epoch": int(epoch_num),
                        "current_beta": float(current_beta),
                        "train_loss": float(history_data["train_loss"][-1]),
                        "train_recon": float(history_data["train_recon"][-1]),
                        "train_kld": float(history_data["train_kld"][-1]),
                        "val_loss": float(history_data["val_loss"][-1]),
                        "val_recon": float(history_data["val_recon"][-1]),
                        "val_kld": float(history_data["val_kld"][-1]),
                        "val_loss_modelsel": float(history_data["val_loss_modelsel"][-1]),
                        "val_loss_beta_max": float(history_data["val_loss_modelsel"][-1]),
                        "best_epoch_so_far": int(best_epoch),
                        "best_val_loss_modelsel_so_far": float(best_val_loss) if np.isfinite(best_val_loss) else None,
                        "checkpoint_type": checkpoint_type,
                        "checkpoint_reason": checkpoint_type,
                    }
                    try:
                        torch.save(checkpoint_payload, checkpoint_path)
                        periodic_vae_checkpoint_rows.append(
                            _checkpoint_row(checkpoint_path, epoch_num, checkpoint_type, current_beta, protected=protected)
                        )
                        logger.info(f"  {fold_idx_str} Checkpoint VAE {checkpoint_type} guardado: {checkpoint_path}")
                        _prune_periodic_vae_checkpoints()
                    except Exception as e:
                        logger.warning(f"  {fold_idx_str} No se pudo guardar checkpoint VAE {checkpoint_path.name}: {e}")

            if should_stop_vae:
                break
        
        
        if best_model_state_dict:
            vae_fold_k.load_state_dict(best_model_state_dict)
            if vae_internal_val_loader and not np.isnan(best_val_loss):
                val_string = f"{best_val_loss:.4f} (βmax)"
            else:
                val_string = "N/A - Last Epoch"
            logger.info(
                f"  {fold_idx_str} VAE final model loaded "
                f"(best ValL(βmax): {val_string})."
            )

        vae_model_fname = f"vae_model_fold_{fold_idx+1}.pt"
        torch.save(vae_fold_k.state_dict(), fold_output_dir / vae_model_fname)
        logger.info(f"  {fold_idx_str} Modelo VAE guardado en: {fold_output_dir / vae_model_fname}")

        if args.save_vae_checkpoints_every_n_epochs is not None:
            if args.save_vae_checkpoints_always_keep_best and best_epoch > 0:
                periodic_vae_checkpoint_rows.append(
                    _checkpoint_row(
                        fold_output_dir / vae_model_fname,
                        best_epoch,
                        "best",
                        _history_value_at_epoch("beta", best_epoch),
                        protected=True,
                    )
                )
            if args.save_vae_checkpoints_always_keep_final and final_epoch_seen > 0:
                already_recorded_final = any(
                    row.get("epoch") == final_epoch_seen
                    and row.get("checkpoint_type") in {"final", "early_stop"}
                    and row.get("status") == "kept"
                    for row in periodic_vae_checkpoint_rows
                )
                if not already_recorded_final:
                    periodic_vae_checkpoint_rows.append(
                        _checkpoint_row(
                            fold_output_dir / vae_model_fname if final_epoch_seen == best_epoch else periodic_vae_checkpoint_dir / f"vae_checkpoint_fold_{fold_idx+1}_final_epoch_{final_epoch_seen:04d}.pt",
                            final_epoch_seen,
                            "final",
                            _history_value_at_epoch("beta", final_epoch_seen),
                            protected=True,
                        )
                    )
            manifest_csv = fold_output_dir / f"vae_checkpoint_manifest_fold_{fold_idx+1}.csv"
            manifest_json = fold_output_dir / f"vae_checkpoint_manifest_fold_{fold_idx+1}.json"
            manifest_payload = {
                "fold": int(fold_idx + 1),
                "checkpoint_interval_epochs": int(args.save_vae_checkpoints_every_n_epochs),
                "checkpoint_start_epoch": int(checkpoint_start_epoch),
                "checkpoint_keep_last_n": args.save_vae_checkpoints_keep_last_n,
                "checkpoint_always_keep_best": bool(args.save_vae_checkpoints_always_keep_best),
                "checkpoint_always_keep_final": bool(args.save_vae_checkpoints_always_keep_final),
                "checkpoint_dir": str(periodic_vae_checkpoint_dir),
                "final_best_checkpoint_path": str(fold_output_dir / vae_model_fname),
                "final_best_checkpoint_format": "raw_state_dict",
                "periodic_checkpoint_format": "dict_with_model_state_dict_and_epoch_metadata",
                "n_checkpoints_manifest_rows": len(periodic_vae_checkpoint_rows),
                "n_checkpoints_kept": int(sum(1 for row in periodic_vae_checkpoint_rows if row.get("status") == "kept")),
                "n_checkpoints_deleted": int(sum(1 for row in periodic_vae_checkpoint_rows if row.get("status") == "deleted")),
                "checkpoints": periodic_vae_checkpoint_rows,
            }
            try:
                pd.DataFrame(periodic_vae_checkpoint_rows).to_csv(manifest_csv, index=False)
                _safe_json_dump(manifest_payload, manifest_json)
                logger.info(f"  {fold_idx_str} Manifest de checkpoints VAE guardado: {manifest_csv}")
            except Exception as e:
                logger.warning(f"  {fold_idx_str} No se pudo guardar manifest de checkpoints VAE: {e}")


        if args.qc_analyze_distributions:
            try:
                logger.info(f"  {fold_idx_str} Ejecutando QC de distribuciones (raw/norm/recon)...")

                # Subconjuntos alineados con el set de entrenamiento real del VAE
                raw_subset  = vae_train_pool_tensor_original_scale[vae_actual_train_indices_local_to_pool]
                norm_subset = vae_pool_tensor_norm[vae_actual_train_indices_local_to_pool]

                vae_fold_k.eval()
                recon_list = []
                bs = 32
                with torch.no_grad():
                    for i in range(0, norm_subset.shape[0], bs):
                        batch = torch.from_numpy(norm_subset[i:i+bs]).float().to(device)
                        cond_batch = None
                        if vae_conditioning_enabled:
                            cond_np = vae_pool_conditioning[vae_actual_train_indices_local_to_pool][i:i+bs]
                            cond_batch = torch.from_numpy(cond_np).float().to(device)
                        recon_batch, _, _, _ = vae_fold_k(batch, condition=cond_batch)
                        recon_list.append(recon_batch.detach().cpu().numpy())
                recon_subset = np.concatenate(recon_list, axis=0)

                summarize_distribution_stages(
                    raw_tensor=raw_subset,
                    norm_tensor=norm_subset,
                    recon_tensor=recon_subset,
                    channel_names=selected_channel_names_in_tensor,
                    out_dir=fold_output_dir,
                    prefix=f"fold_{fold_idx+1}",
                    final_activation=args.vae_final_activation
                )
                logger.info(f"  {fold_idx_str} QC distribuciones guardado en {fold_output_dir}")
            except Exception as e:
                logger.warning(f"  {fold_idx_str} QC distribuciones falló: {e}")

        
        if args.save_vae_training_history:
            # El diccionario `history_data` ya está completo. Solo lo guardamos.
            joblib.dump(history_data, fold_output_dir / f"vae_train_history_fold_{fold_idx+1}.joblib")
            try:
                fig, ax1 = plt.subplots(figsize=(12, 6)) # Un poco más ancho
                
                # Graficar componentes del entrenamiento
                ax1.plot(history_data["train_loss"], label="Train Loss (Total)", color='blue', linewidth=2)
                ax1.plot(history_data["train_recon"], label="Train Recon Loss", color='cyan', linestyle=':')
                ax1.plot(history_data["train_kld"], label="Train KLD Loss", color='magenta', linestyle=':')
                
                # Graficar componentes de la validación si existen
                if vae_internal_val_loader and any(not np.isnan(x) for x in history_data["val_loss"]):
                    ax1.plot(history_data["val_loss"], label="Val Loss (curβ)", color='orange', linewidth=2)
                    ax1.plot(history_data["val_loss_modelsel"], label="Val Loss (βmax)", color='red', linestyle='-.', linewidth=2)
                    ax1.plot(history_data["val_recon"], label="Val Recon Loss", color='#ff9966', linestyle='--') # Naranja claro
                    ax1.plot(history_data["val_kld"], label="Val KLD Loss", color='#ff66b2', linestyle='--') # Rosa/Rojo claro
                
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Loss Value")
                ax1.set_title(f"Fold {fold_idx+1} VAE Training History (β warmup aware)")
                ax1.legend(loc='upper left')
                ax1.grid(True, linestyle='--', alpha=0.6)
                ax1.set_ylim(bottom=0) # La pérdida no debería ser negativa

                # Eje secundario para Beta
                ax2 = ax1.twinx()
                ax2.plot(history_data["beta"], label="Beta", color='green', linestyle='-.', alpha=0.8)
                ax2.set_ylabel("Beta Value", color='green')
                ax2.tick_params(axis='y', labelcolor='green')
                ax2.legend(loc='upper right')

                fig.tight_layout() # Ajusta el layout para que no se superpongan las etiquetas
                plt.savefig(fold_output_dir / f"vae_train_history_fold_{fold_idx+1}.png")
                plt.close(fig)
            except Exception as e:
                logger.warning(f"  {fold_idx_str} No se pudo guardar la gráfica de historial VAE: {e}")

        if args.qc_rate_distortion:
            try:
                logger.info(f"  {fold_idx_str} Generando tabla Rate–Distortion (D/R) desde history_data...")
                summarize_rate_distortion_history(
                    history_data=history_data,
                    out_dir=fold_output_dir,
                    prefix=f"fold_{fold_idx+1}",
                    beta_max=float(args.beta_vae),
                    log_base=float(args.qc_rd_log_base),
                )
                logger.info(f"  {fold_idx_str} Rate–Distortion guardado en: {fold_output_dir}")
            except Exception as e:
                logger.warning(f"  {fold_idx_str} QC Rate–Distortion falló: {e}")

        all_folds_vae_history.append(history_data if args.save_vae_training_history else None)
        
        clf_train_dev_df = cn_ad_df.iloc[train_dev_clf_idx_in_cn_ad_df].copy()
        global_indices_clf_train_dev_all = clf_train_dev_df['tensor_idx'].values
        y_clf_train_dev_all = clf_train_dev_df['label'].values
        log_group_distributions(clf_train_dev_df, strat_cols, "Pool Train/Dev (Clasificador)", fold_idx_str)
        clf_train_dev_conditioning = apply_vae_conditioning_transformer(clf_train_dev_df, vae_conditioning_transformer) if vae_conditioning_enabled else None


        vae_fold_k.eval()
        with torch.no_grad():
            clf_train_dev_tensor_for_latent = (
                harmonized_clf_train_dev_tensor_original_scale
                if harmonized_clf_train_dev_tensor_original_scale is not None
                else current_global_tensor[global_indices_clf_train_dev_all]
            )
            full_train_dev_tensor_norm = apply_normalization_params(
                clf_train_dev_tensor_for_latent,
                norm_params_fold_list
            )
            cond_train_dev_tensor = (
                torch.from_numpy(clf_train_dev_conditioning).float().to(device)
                if vae_conditioning_enabled else None
            )
            _, mu_train_dev, _, z_train_dev = vae_fold_k(
                torch.from_numpy(full_train_dev_tensor_norm).float().to(device),
                condition=cond_train_dev_tensor,
            )

            
            # --- GPU MEMORY OPTIMIZATION START ---
            # 1. Capture CPU numpy versions immediately for DataFrame and QC
            mu_train_dev_np = mu_train_dev.detach().cpu().numpy()
            z_train_dev_np = z_train_dev.detach().cpu().numpy()
            
            # 2. Release GPU tensors immediately
            del mu_train_dev, z_train_dev
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            X_np = mu_train_dev_np if args.latent_features_type == 'mu' else z_train_dev_np
            feature_names = [f"latent_{i}" for i in range(X_np.shape[1])]
            X_latent_train_dev = pd.DataFrame(X_np, columns=feature_names)

            if args.qc_latent_information:
                try:
                    logger.info(f"  {fold_idx_str} Calculando info latente (MI/ActiveUnits/TC) en TRAIN/DEV...")
                    _ = evaluate_latent_information(
                        metadata_df_full=metadata_df_full,
                        subject_global_indices=global_indices_clf_train_dev_all,
                        latent_mu_subjects=mu_train_dev_np,  # QC siempre sobre mu
                        y_target=y_clf_train_dev_all,
                        out_dir=fold_output_dir,
                        fold_tag=f"fold_{fold_idx+1}_trainDev",
                        nuisance_cols=args.qc_nuisance_cols if args.qc_nuisance_cols else None,
                        random_state=int(args.seed + fold_idx + 777),
                        n_neighbors=int(args.qc_mi_n_neighbors),
                        top_k=int(args.qc_mi_top_k),
                        var_eps_active=float(args.qc_var_eps_active),
                        tc_ridge=float(args.qc_tc_ridge),
                    )
                    logger.info(f"  {fold_idx_str} Info latente TRAIN/DEV guardada en {fold_output_dir}")
                except Exception as e:
                    logger.warning(f"  {fold_idx_str} QC info latente TRAIN/DEV falló: {e}")

            if args.qc_check_scanner_leakage:
                try:
                    logger.info(f"  {fold_idx_str} Evaluando leakage de escáner/sitio (espacio crudo vs latente)...")
                    _leak_df = evaluate_scanner_leakage(
                        metadata_df_full=metadata_df_full,
                        subject_global_indices=global_indices_clf_train_dev_all,
                        normalized_tensor_subjects=full_train_dev_tensor_norm,
                        latent_mu_subjects=mu_train_dev_np,  # Leakage siempre sobre mu
                        out_dir=fold_output_dir,
                        fold_tag=f"fold_{fold_idx+1}",
                        random_state=args.seed + fold_idx + 99,
                        vectorize_mode="auto",
                    )
                    if _leak_df is not None:
                        logger.info(f"  {fold_idx_str} Leakage escáner guardado en {fold_output_dir}")
                    else:
                        logger.info(f"  {fold_idx_str} Leakage escáner no aplicable (no hay columna de sitio/escáner o sólo 1 clase).")
                except Exception as e:
                    logger.warning(f"  {fold_idx_str} QC leakage falló: {e}")

            # Free large CPU tensor after QC usage
            del full_train_dev_tensor_norm
            gc.collect()


            clf_test_tensor_for_latent = (
                harmonized_clf_test_tensor_original_scale
                if harmonized_clf_test_tensor_original_scale is not None
                else current_global_tensor[global_indices_clf_test_this_fold]
            )
            X_test_final_tensor_norm = apply_normalization_params(
                clf_test_tensor_for_latent,
                norm_params_fold_list
            )
            mu_test_final_np, z_test_final_np, X_np_test = None, None, None
            if X_test_final_tensor_norm is not None and X_test_final_tensor_norm.shape[0] > 0:

                clf_test_conditioning = (
                    apply_vae_conditioning_transformer(cn_ad_df.iloc[test_clf_idx_in_cn_ad_df].copy(), vae_conditioning_transformer)
                    if vae_conditioning_enabled else None
                )
                cond_test_tensor = (
                    torch.from_numpy(clf_test_conditioning).float().to(device)
                    if vae_conditioning_enabled else None
                )
                _, mu_test_final, _, z_test_final = vae_fold_k(
                    torch.from_numpy(X_test_final_tensor_norm).float().to(device),
                    condition=cond_test_tensor,
                )
                #X_np_test = mu_test_final.cpu().numpy() if args.latent_features_type == 'mu' else z_test_final.cpu().numpy()
                # Capture CPU numpy versions
                mu_test_final_np = mu_test_final.detach().cpu().numpy()
                z_test_final_np = z_test_final.detach().cpu().numpy()
                
                del mu_test_final, z_test_final
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                X_np_test = mu_test_final_np if args.latent_features_type == 'mu' else z_test_final_np
                
                X_latent_test_final = pd.DataFrame(X_np_test, columns=feature_names)
            else:
                X_latent_test_final = pd.DataFrame(columns=feature_names)
            y_test_final = y_classifier_labels_cn_ad[test_clf_idx_in_cn_ad_df]


            if args.qc_latent_information and X_test_final_tensor_norm.shape[0] > 0:
                try:
                    logger.info(f"  {fold_idx_str} Calculando info latente (MI/ActiveUnits/TC) en TEST externo...")
                    _ = evaluate_latent_information(
                        metadata_df_full=metadata_df_full,
                        subject_global_indices=global_indices_clf_test_this_fold,
                        latent_mu_subjects=mu_test_final_np, # QC siempre sobre mu
                        y_target=y_test_final,
                        out_dir=fold_output_dir,
                        fold_tag=f"fold_{fold_idx+1}_test",
                        nuisance_cols=args.qc_nuisance_cols if args.qc_nuisance_cols else None,
                        random_state=int(args.seed + fold_idx + 888),
                        n_neighbors=int(args.qc_mi_n_neighbors),
                        top_k=int(args.qc_mi_top_k),
                        var_eps_active=float(args.qc_var_eps_active),
                        tc_ridge=float(args.qc_tc_ridge),
                    )
                    logger.info(f"  {fold_idx_str} Info latente TEST guardada en {fold_output_dir}")
                except Exception as e:
                    logger.warning(f"  {fold_idx_str} QC info latente TEST falló: {e}")

            if args.metadata_features:
                logger.info(f"  Añadiendo metadatos al clasificador: {args.metadata_features}")

                # =================================================================
                # IMPORTANTE (FIX AUC LEAKAGE):
                # - NO codificar / imputar / escalar aquí.
                # - Dejar que el Pipeline del clasificador (_AutoPreprocessor) lo haga
                #   *dentro* de cada split interno del CV.
                # - Si imputás/escaleás acá, filtrás información del fold interno
                #   de validación hacia el entrenamiento → AUC inflado artificialmente.
                #
                # EXCEPCIÓN: Guardamos los valores de imputación para inferencia
                # externa (COVID, etc.) donde no hay CV y necesitamos reproducir
                # el preprocesamiento.
                # =================================================================

                # --- TRAIN/DEV: pasar metadata SIN preprocesar ---
                metadata_train_dev = clf_train_dev_df[args.metadata_features].copy()

                # Concatenar: crucial resetear los índices para una alineación correcta
                X_latent_train_dev.reset_index(drop=True, inplace=True)
                metadata_train_dev.reset_index(drop=True, inplace=True)
                X_train_dev_combined = pd.concat([X_latent_train_dev, metadata_train_dev], axis=1)

                # Reemplazamos el DataFrame original
                X_latent_train_dev = X_train_dev_combined
                logger.info(f"  Forma final del set de entrenamiento del clasificador: {X_latent_train_dev.shape}")

                # --- TEST: pasar metadata SIN preprocesar ---
                if not X_latent_test_final.empty:
                    clf_test_df = cn_ad_df.iloc[test_clf_idx_in_cn_ad_df]
                    metadata_test = clf_test_df[args.metadata_features].copy()

                    # Concatenar
                    X_latent_test_final.reset_index(drop=True, inplace=True)
                    metadata_test.reset_index(drop=True, inplace=True)
                    X_test_final_combined = pd.concat([X_latent_test_final, metadata_test], axis=1)

                    # Reemplazar el DataFrame original
                    X_latent_test_final = X_test_final_combined

                # ---------------------------------------------------------
                # Calcular y GUARDAR imputation metadata (para inferencia externa)
                # Esto NO se aplica a los datos de train/test que van al clasificador,
                # solo se guarda para reproducir el preprocesamiento en COVID.
                # ---------------------------------------------------------
                imputation_values: Dict[str, Any] = {}
                imputation_strategies: Dict[str, str] = {}
                sex_mapping = None

                for col in metadata_train_dev.columns:
                    s = clf_train_dev_df[col]  # usar el DF original, no el preprocesado

                    if col == 'Sex':
                        sex_mapping = {'M': 0, 'm': 0, 'F': 1, 'f': 1}

                    # estrategia: mean para numéricas "continuas"; mode para binarias/categóricas
                    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 2:
                        val = float(s.mean(skipna=True)) if s.notna().any() else 0.0
                        strat = "mean"
                    else:
                        if s.notna().any():
                            try:
                                val = s.mode(dropna=True).iloc[0]
                            except Exception:
                                val = s.dropna().iloc[0]
                        else:
                            val = 0.0
                        strat = "mode"

                    imputation_values[col] = val
                    imputation_strategies[col] = strat

                # Guardar metadata_imputation.json (para inferencia externa)
                try:
                    _safe_json_dump(
                        {
                            "fold": int(fold_idx + 1),
                            "metadata_features": list(args.metadata_features),
                            "sex_mapping": sex_mapping,
                            "imputation_values": imputation_values,
                            "imputation_strategies": imputation_strategies,
                            "note": "⚠️ SOLO para inferencia externa (COVID). NO aplicado a train/test en este fold para evitar leakage."
                        },
                        fold_output_dir / "metadata_imputation.json"
                    )
                except Exception as e:
                    logger.warning(f"  {fold_idx_str} No se pudo guardar metadata_imputation.json: {e}")

            # ---------------------------------------------------------
            # Guardar el ORDEN FINAL de columnas de features (latentes + metadata)
            # para inferencia externa y consistencia (por fold).
            # - Se guarda SIEMPRE, con o sin metadata_features.
            # ---------------------------------------------------------
            try:
                _safe_json_dump(
                    {
                        "fold": int(fold_idx + 1),
                        "latent_features_type": str(args.latent_features_type),
                        "latent_feature_names": list(feature_names),
                        "final_feature_columns": list(X_latent_train_dev.columns),
                        "has_metadata_features": bool(args.metadata_features),
                        "metadata_features": list(args.metadata_features) if args.metadata_features else None,
                        "note": "Usar 'final_feature_columns' para reordenar columnas en inferencia externa antes de predict()."
                    },
                    fold_output_dir / "feature_columns.json"
                )
            except Exception as e:
                logger.warning(f"  {fold_idx_str} No se pudo guardar feature_columns.json: {e}")
 


            # ==========================
            # QC latente (fold externo): silhouette + scanner/site leakage
            # Opción B: UNA sola implementación → evaluate_scanner_leakage()
            # y usar su salida para poblar latent_qc_metrics.csv
            # ==========================
            if X_np_test is not None and len(y_test_final) == len(X_np_test) and len(y_test_final) > 2:
                silhouette_latent = compute_latent_silhouette(latent_feats=X_np_test, labels_binary=y_test_final)
            else:
                silhouette_latent = np.nan

            # Defaults (si QC leakage está apagado o no aplicable)
            acc_site_raw = np.nan
            acc_site_latent = np.nan
            n_sites = np.nan

            def _pick_first_col(df: pd.DataFrame, candidates: List[str]) -> str:
                for c in candidates:
                    if c in df.columns:
                        return c
                return ""

            if args.qc_check_scanner_leakage:
                try:
                    logger.info(
                        f"  {fold_idx_str} Evaluando leakage de escáner/sitio (TEST externo) para latent_qc_metrics.csv..."
                    )
                    leak_df_test = evaluate_scanner_leakage(
                        metadata_df_full=metadata_df_full,
                        subject_global_indices=global_indices_clf_test_this_fold,
                        normalized_tensor_subjects=X_test_final_tensor_norm,
                        latent_mu_subjects=mu_test_final_np,  # leakage siempre sobre mu
                        out_dir=fold_output_dir,
                        fold_tag=f"fold_{fold_idx+1}_test",
                        random_state=args.seed + fold_idx + 202,
                        vectorize_mode="auto",
                    )

                    # Extraer métricas de forma robusta (según el esquema de columnas del DF)
                    if leak_df_test is not None and isinstance(leak_df_test, pd.DataFrame) and not leak_df_test.empty:
                        # 1) n_sites / n_classes
                        ncol = _pick_first_col(leak_df_test, ["n_sites", "n_classes", "n_site", "n_scanners", "n_labels"])
                        if ncol:
                            try:
                                n_sites = float(leak_df_test[ncol].iloc[0])
                            except Exception:
                                pass

                        # 2) Si vienen columnas directas (lo más simple)
                        if "acc_site_raw" in leak_df_test.columns:
                            try:
                                acc_site_raw = float(leak_df_test["acc_site_raw"].iloc[0])
                            except Exception:
                                pass
                        if "acc_site_latent" in leak_df_test.columns:
                            try:
                                acc_site_latent = float(leak_df_test["acc_site_latent"].iloc[0])
                            except Exception:
                                pass

                        # 3) Si el DF viene en formato "largo" (filas raw vs latent)
                        if (np.isnan(acc_site_raw) or np.isnan(acc_site_latent)):
                            space_col = _pick_first_col(
                                leak_df_test,
                                ["space", "representation", "repr", "feature_space", "domain"],
                            )
                            score_col = _pick_first_col(
                                leak_df_test,
                                [
                                    "balanced_accuracy_mean",
                                    "balanced_accuracy",
                                    "bal_acc_mean",
                                    "bal_acc",
                                    "acc_mean",
                                    "acc",
                                    "score_mean",
                                    "score",
                                ],
                            )
                            if space_col and score_col:
                                s = leak_df_test[space_col].astype(str).str.lower()
                                raw_rows = leak_df_test[s.str.contains("raw")]
                                lat_rows = leak_df_test[s.str.contains("latent") | s.str.contains("mu") | s.str.contains("z")]
                                try:
                                    if np.isnan(acc_site_raw) and not raw_rows.empty:
                                        acc_site_raw = float(raw_rows[score_col].iloc[0])
                                except Exception:
                                    pass
                                try:
                                    if np.isnan(acc_site_latent) and not lat_rows.empty:
                                        acc_site_latent = float(lat_rows[score_col].iloc[0])
                                except Exception:
                                    pass

                        logger.info(
                            f"  {fold_idx_str} Leakage TEST extraído: "
                            f"acc_raw={acc_site_raw if not np.isnan(acc_site_raw) else 'NaN'}, "
                            f"acc_latent={acc_site_latent if not np.isnan(acc_site_latent) else 'NaN'}, "
                            f"n_sites={n_sites if not np.isnan(n_sites) else 'NaN'}"
                        )
                    else:
                        logger.info(
                            f"  {fold_idx_str} Leakage TEST no aplicable (sin columna de sitio/escáner o 1 clase)."
                        )
                except Exception as e:
                    logger.warning(f"  {fold_idx_str} QC leakage TEST falló: {e}")

            # Clean up large CPU tensor after final QC usage
            del X_test_final_tensor_norm
            gc.collect()

            chance_level = (1.0 / n_sites) if (isinstance(n_sites, (int, float)) and n_sites and n_sites > 0) else np.nan
 
            # 3) Guardar CSV de QC latente por fold (una sola fila)
            qc_latent_row = {
                "fold":               fold_idx + 1,
                "silhouette_latent":  silhouette_latent,
                "acc_site_latent":    acc_site_latent,
                "acc_site_raw":       acc_site_raw,
                "n_sites":            n_sites,
                "chance_level":       chance_level,
                "latent_dim":         args.latent_dim,
                "beta_max":           args.beta_vae,
                "decoder_type":       args.decoder_type,
                "num_conv_layers":    args.num_conv_layers_encoder,
                "norm_mode":          args.norm_mode,
                "recon_loss_mode":    args.recon_loss_mode,
                "vae_final_activation": args.vae_final_activation,
                "vae_dropout_scope":  args.vae_dropout_scope,
                "dropout_rate_vae":   args.dropout_rate_vae,
                "encoder_dropout_rate_vae": args.encoder_dropout_rate_vae,
                "decoder_dropout_rate_vae": args.decoder_dropout_rate_vae,
                "vae_block_order":    args.vae_block_order,
                "channels_used":      ",".join(map(str, selected_channel_names_in_tensor)),
            }
            qc_latent_df = pd.DataFrame([qc_latent_row])
            qc_latent_path = fold_output_dir / "latent_qc_metrics.csv"
            qc_latent_df.to_csv(qc_latent_path, index=False)
            logger.info(f"  {fold_idx_str} QC latente guardado en: {qc_latent_path}")
            logger.info(f"  {fold_idx_str} Características latentes obtenidas para clasificación.")




        # --- BUCLE DE CLASIFICADOR REFACTORIZADO ---
        for current_classifier_type in args.classifier_types:
            logger.info(f"    --- Entrenando Clasificador: {current_classifier_type} ---")
            
            try:
                # Obtenemos el pipeline completo y el grid de parámetros
                full_pipeline, param_distributions, n_iter_search = get_classifier_and_grid(
                    classifier_type=current_classifier_type,
                    seed=args.seed,
                    balance=args.classifier_use_class_weight,
                    use_smote=args.use_smote,
                    tune_sampler_params=args.tune_sampler_params,
                    mlp_hidden_layers=args.mlp_classifier_hidden_layers,
                    calibrate=False
                )
            except (ImportError, ValueError) as e:
                logger.error(f"Error al obtener pipeline para {current_classifier_type}: {e}. Saltando.")
                continue

            # ------------------------------------------------------------
            # CV interna: para selección de HPs, conviene preservar la misma
            # estratificación (p.ej., ResearchGroup + Sex) que en el outer split.
            # Esto suele estabilizar y subir AUC cuando hay covariables fuertes.
            # ------------------------------------------------------------
            inner_skf = StratifiedKFold(
                n_splits=args.inner_folds,
                shuffle=True,
                random_state=args.seed + fold_idx + 30
            )

            # Intentamos estratificar por la MISMA clave del outer (ResearchGroup_Mapped + cols extras).
            # Si algún estrato es demasiado chico para n_splits, fallback a label-only.
            try:
                inner_strat_df = clf_train_dev_df[strat_cols].copy()
                for col in strat_cols:
                    if col in inner_strat_df.columns:
                        inner_strat_df[col] = inner_strat_df[col].fillna(f"{col}_Unknown").astype(str)
                inner_key = inner_strat_df.apply(lambda r: "_".join(r.values.astype(str)), axis=1)

                vc = inner_key.value_counts()
                if (vc < args.inner_folds).any():
                    logger.warning(
                        f"      Inner-CV: estratos muy chicos con {strat_cols} (min={int(vc.min())}). "
                        f"Fallback a estratificación por label."
                    )
                    inner_key = y_clf_train_dev_all
            except Exception as e:
                logger.warning(f"      Inner-CV: no se pudo construir clave estratificada ({e}). Fallback a label.")
                inner_key = y_clf_train_dev_all

            inner_splits = list(inner_skf.split(np.zeros(len(inner_key)), inner_key))

            # 1. Crear el sampler que se quiere usar
            sampler = optuna.samplers.TPESampler(seed=args.seed)

            # 2. Opcionalmente, crear un pruner para detener trials no prometedores
            pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=1, interval_steps=1, n_min_trials=5) if args.use_optuna_pruner else None
            if pruner:
                logger.info("      Usando Optuna MedianPruner para acelerar la búsqueda.")

            # 3. Crear un 'study' de Optuna, especificando la dirección, sampler y pruner.
            #    La dirección debe ser "maximize" porque 'roc_auc' es mejor cuanto más alto.
            study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

            # 3. Instanciar OptunaSearchCV, pasando el 'study' en lugar del 'sampler'.
            for k, v in param_distributions.items():
                if not isinstance(v, optuna.distributions.BaseDistribution):
                    raise TypeError(f"{k} no es una distribución de Optuna ({type(v)})")
                
            # 1. Capar el número de trials para no exceder un límite razonable
            max_trials = 1000
            effective_n_trials, requested_n_iter_override, n_iter_source = _resolve_classifier_n_iter(
                args,
                current_classifier_type,
                n_iter_search,
                max_trials=max_trials,
            )
            logger.info(
                f"      Optuna trials for {current_classifier_type}: {effective_n_trials} "
                f"(source={n_iter_source}, factory_default={int(n_iter_search)}, "
                f"requested_override={requested_n_iter_override}, cap={max_trials})"
            )

            # 2. Definir un timeout en segundos (ej. 1800s = 30 minutos) por clasificador/fold
            timeout_seconds = 1800


            optuna_search = OptunaSearchCV(
                estimator=full_pipeline,
                param_distributions=param_distributions,
                study=study,
                cv=inner_splits,
                scoring=args.gridsearch_scoring,
                n_trials=effective_n_trials,  # <--- USAR EL VALOR CAPADO
                refit=True,
                n_jobs=args.n_jobs_gridsearch,
                timeout=timeout_seconds,       # <--- AÑADIR TIMEOUT
                random_state=args.seed
            )
            optuna_search.fit(X_latent_train_dev, y_clf_train_dev_all)
            n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
            n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
            logger.info(f"[Optuna] Trials COMPLETE={n_complete}, PRUNED={n_pruned}")

            # (S1) Guardar auditoría Optuna por fold/clf
            if args.save_fold_artefacts:
                try:
                    trials_df = study.trials_dataframe()
                    trials_df.to_csv(fold_output_dir / f"optuna_trials_{current_classifier_type}_fold_{fold_idx+1}.csv", index=False)
                except Exception as e:
                    logger.warning(f"  {fold_idx_str} No se pudo guardar optuna_trials CSV ({current_classifier_type}): {e}")
                try:
                    joblib.dump(study, fold_output_dir / f"optuna_study_{current_classifier_type}_fold_{fold_idx+1}.joblib")
                except Exception as e:
                    logger.warning(f"  {fold_idx_str} No se pudo guardar optuna_study joblib ({current_classifier_type}): {e}")
                try:
                    bt = study.best_trial
                    _safe_json_dump(
                        {
                            "fold": int(fold_idx + 1),
                            "classifier": str(current_classifier_type),
                            "best_value": float(bt.value) if bt.value is not None else None,
                            "best_params": dict(bt.params) if bt.params is not None else None,
                            "n_trials": int(len(study.trials)),
                            "n_complete": int(n_complete),
                            "n_pruned": int(n_pruned),
                            "n_iter_search_factory_default": int(n_iter_search),
                            "n_iter_search_requested_override": int(requested_n_iter_override) if requested_n_iter_override is not None else None,
                            "n_iter_search_source": str(n_iter_source),
                            "effective_n_trials": int(effective_n_trials),
                        },
                        fold_output_dir / f"optuna_best_trial_{current_classifier_type}_fold_{fold_idx+1}.json",
                    )
                except Exception as e:
                    logger.warning(f"  {fold_idx_str} No se pudo guardar optuna_best_trial JSON ({current_classifier_type}): {e}")


            best_params_clf = optuna_search.best_params_
            raw_clf_model = optuna_search.best_estimator_
            final_clf_model = raw_clf_model
            did_calibrate = False

            # Guardar SIEMPRE el RAW (para SHAP)
            if args.save_fold_artefacts:
                joblib.dump(
                    raw_clf_model,
                    fold_output_dir / f"classifier_{current_classifier_type}_raw_pipeline_fold_{fold_idx+1}.joblib"
                )

            # Calibración post-hoc (solo TRAIN/DEV del fold externo)
            if args.classifier_calibrate:
                logger.info("      Calibrando post-hoc (sigmoid, cv=3) el pipeline completo...")
                cal_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed + fold_idx + 123)
                cal = CalibratedClassifierCV(estimator=raw_clf_model, method="sigmoid", cv=cal_cv)
                cal.fit(X_latent_train_dev, y_clf_train_dev_all)
                final_clf_model = cal
                did_calibrate = True

                if args.save_fold_artefacts:
                    joblib.dump(
                        final_clf_model,
                        fold_output_dir / f"classifier_{current_classifier_type}_calibrated_pipeline_fold_{fold_idx+1}.joblib"
                    )

            logger.info(f"      Mejores HPs para {current_classifier_type}: {best_params_clf}")

            # --- evaluación ---
            fold_results_clf = {
                "fold": fold_idx + 1,
                "actual_classifier_type": current_classifier_type,
                "best_clf_params": best_params_clf,
                "did_calibrate": did_calibrate,
            }

            if X_latent_test_final.shape[0] > 0:
                y_score_raw = _get_score_1d(raw_clf_model, X_latent_test_final)

                if did_calibrate:
                    y_score_cal = _get_score_1d(final_clf_model, X_latent_test_final)  # calibrated proba
                    y_score_final = y_score_cal
                else:
                    y_score_cal = np.full_like(np.asarray(y_score_raw, dtype=float), np.nan, dtype=float)
                    y_score_final = y_score_raw

                # pred final
                y_pred = final_clf_model.predict(X_latent_test_final)

                test_df_fold = cn_ad_df.iloc[test_clf_idx_in_cn_ad_df]
                df_preds = pd.DataFrame({
                    "SubjectID": test_df_fold["SubjectID"].values,
                    "tensor_idx": test_df_fold["tensor_idx"].values,
                    "y_true": y_test_final,
                    "y_score_raw": y_score_raw,
                    "y_score_cal": y_score_cal,
                    "y_score_final": y_score_final,
                    "y_pred": y_pred,
                    "did_calibrate": did_calibrate,
                })
                pred_save_path = fold_output_dir / f"test_predictions_{current_classifier_type}.csv"
                df_preds.to_csv(pred_save_path, index=False)

                # métricas: yo guardaría AUC en raw y en final (si calibraste)
                auc_raw = roc_auc_score(y_test_final, y_score_raw)
                pr_auc_raw = average_precision_score(y_test_final, y_score_raw)
                auc_final = roc_auc_score(y_test_final, y_score_final)
                pr_auc_final = average_precision_score(y_test_final, y_score_final)
                fold_results_clf.update({
                    "auc_raw": auc_raw,
                    "pr_auc_raw": pr_auc_raw,
                    "auc_final": auc_final,
                    "pr_auc_final": pr_auc_final,
                    # Compatibilidad con summaries antiguos:
                    "auc": auc_final,
                    "pr_auc": pr_auc_final,
                    "accuracy": accuracy_score(y_test_final, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(y_test_final, y_pred),
                    "sensitivity": recall_score(y_test_final, y_pred, pos_label=1, zero_division=0),
                    "specificity": recall_score(y_test_final, y_pred, pos_label=0, zero_division=0),
                    "f1_score": f1_score(y_test_final, y_pred, pos_label=1, zero_division=0),
                })

# AGREGAR ESTO AQUÍ:
                logger.info(f"  >>> RESULTADOS {fold_idx_str} [{current_classifier_type}]: "
                            f"AUC={auc_raw:.4f}, Acc={fold_results_clf['accuracy']:.4f}")          

                # (S2) Acumular predicciones para guardado agregado
                try:
                    df_preds_extra = df_preds.copy()
                    df_preds_extra.insert(0, "fold", int(fold_idx + 1))
                    df_preds_extra.insert(1, "classifier_type", str(current_classifier_type))
                    all_folds_clf_predictions.append(df_preds_extra)
                except Exception as e:
                    logger.warning(f"  {fold_idx_str} No se pudo acumular df_preds para agregado: {e}")
  
            else:
                for m in ["auc_raw","pr_auc_raw","auc_final","pr_auc_final","accuracy","balanced_accuracy","sensitivity","specificity","f1_score"]:
                    fold_results_clf[m] = np.nan
                fold_results_clf["auc"] = np.nan
                fold_results_clf["pr_auc"] = np.nan

            # Guardar "pipeline final" (opcional; si calibraste duplica al calibrated)
            if args.save_fold_artefacts:
                joblib.dump(
                    final_clf_model,
                    fold_output_dir / f"classifier_{current_classifier_type}_final_pipeline_fold_{fold_idx+1}.joblib"
                )
            all_folds_metrics.append(fold_results_clf)
        
        del vae_fold_k, optimizer_vae, vae_train_loader, vae_internal_val_loader, scheduler_vae, best_model_state_dict
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
        logger.info(f"  {fold_idx_str} completado en {time.time() - fold_start_time:.2f} segundos.")

    if all_folds_metrics:
        metrics_df = pd.DataFrame(all_folds_metrics)
        
        for clf_type_iterated in args.classifier_types:
            metrics_df_clf = metrics_df[metrics_df['actual_classifier_type'] == clf_type_iterated]
            if not metrics_df_clf.empty:
                logger.info(f"\n--- Resumen de Rendimiento para Clasificador: {clf_type_iterated} (Promedio sobre Folds Externos) ---")
                # Reporte explícito RAW vs FINAL (si calibraste)
                report_metrics = [
                    "auc_raw", "pr_auc_raw",
                    "auc_final", "pr_auc_final",
                    "accuracy", "balanced_accuracy",
                    "sensitivity", "specificity", "f1_score",
                ]
                for metric in report_metrics:
                    if metric in metrics_df_clf.columns and metrics_df_clf[metric].notna().any():
                        mean_val = metrics_df_clf[metric].mean()
                        std_val = metrics_df_clf[metric].std()
                        logger.info(f"{metric:<20}: {mean_val:.4f} +/- {std_val:.4f}")

        
        main_clf_type_for_fname = args.classifier_types[0] if args.classifier_types else "genericclf"
        fname_suffix = (f"{main_clf_type_for_fname}_vae{args.decoder_type}{args.num_conv_layers_encoder}l_"
                        f"ld{args.latent_dim}_beta{args.beta_vae}_norm{args.norm_mode}_"
                        f"ch{num_input_channels_for_vae}{'sel' if args.channels_to_use else 'all'}_"
                        f"intFC{args.intermediate_fc_dim_vae}_drop{args.dropout_rate_vae}_"
                        f"ln{1 if args.use_layernorm_vae_fc else 0}_outer{args.outer_folds}x{args.repeated_outer_folds_n_repeats if args.repeated_outer_folds_n_repeats > 1 else 1}_"
                        f"score{args.gridsearch_scoring}")
        
        results_csv_path = output_base_dir / f"all_folds_metrics_MULTI_{fname_suffix}.csv"
        metrics_df.to_csv(results_csv_path, index=False)
        logger.info(f"Resultados detallados de todos los clasificadores guardados en: {results_csv_path}")

        summary_txt_path = output_base_dir / f"summary_metrics_MULTI_{fname_suffix}.txt"
        with open(summary_txt_path, 'w') as f:
            f.write(f"Run Arguments:\n{vars(args)}\n\n")
            f.write(f"Git Commit Hash: {args.git_hash}\n\n")
            for clf_type_iterated in args.classifier_types:
                metrics_df_clf = metrics_df[metrics_df['actual_classifier_type'] == clf_type_iterated]
                if not metrics_df_clf.empty:
                    f.write(f"--- Metrics Summary for Classifier: {clf_type_iterated} ---\n")
                    for metric in ['auc_raw','pr_auc_raw','auc_final','pr_auc_final','accuracy','balanced_accuracy','sensitivity','specificity','f1_score']:
                        if metric in metrics_df_clf.columns and metrics_df_clf[metric].notna().any():
                            f.write(f"{metric.capitalize():<20}: {metrics_df_clf[metric].mean():.4f} +/- {metrics_df_clf[metric].std():.4f}\n")
                    f.write("\nFull Metrics DataFrame Description:\n")
                    f.write(metrics_df_clf.describe().to_string())
                    f.write("\n\n")
        logger.info(f"Sumario estadístico de métricas (por clasificador) guardado en: {summary_txt_path}")

        if args.save_vae_training_history and all_folds_vae_history:
             joblib.dump(all_folds_vae_history, output_base_dir / f"all_folds_vae_training_history_{fname_suffix}.joblib")
        if all_folds_clf_predictions:
             try:
                 preds_all_df = pd.concat(all_folds_clf_predictions, axis=0, ignore_index=True)
                 preds_all_df.to_csv(output_base_dir / f"all_folds_clf_predictions_MULTI_{fname_suffix}.csv", index=False)
             except Exception as e:
                 logger.warning(f"No se pudo guardar CSV agregado de predicciones: {e}")
             joblib.dump(all_folds_clf_predictions, output_base_dir / f"all_folds_clf_predictions_MULTI_{fname_suffix}.joblib")
        return metrics_df
    else:
        logger.warning("No se pudieron calcular métricas para ningún fold.")
        return None

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline VAE+Clasificador para AD/CN (v1.8.0)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group_data = parser.add_argument_group('Data and Paths')
    group_data.add_argument("--global_tensor_path", type=str, required=True, help="Ruta al archivo .npz del tensor global.")
    group_data.add_argument("--metadata_path", type=str, required=True, help="Ruta al archivo CSV de metadatos.")
    group_data.add_argument(
                                "--output_dir",
                                type=str,
                                default="results/vae_clf_output_v1.8.0",
                                help="Directorio para guardar resultados."
                            )

    group_data.add_argument("--channels_to_use", type=int, nargs='*', default=None, help="Lista de índices de canales a usar (0-based, según DEFAULT_CHANNEL_NAMES).")
    group_cv = parser.add_argument_group('Cross-validation')
    group_cv.add_argument("--outer_folds", type=int, default=5, help="Número de folds para CV externa del clasificador.")
    group_cv.add_argument("--repeated_outer_folds_n_repeats", type=int, default=1, help="Número de repeticiones para RepeatedStratifiedKFold.")
    group_cv.add_argument("--inner_folds", type=int, default=5, help="Folds para CV interna (búsqueda de HP con OptunaSearchCV).") 
    group_cv.add_argument(
        "--outer_fold_indices_to_run",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional 1-based outer fold iteration indices to execute. "
            "Default None runs all folds. Intended for controlled diagnostics only."
        ),
    )
    group_cv.add_argument("--classifier_stratify_cols", type=str, nargs='*', default=['Sex'], help="Columnas adicionales para estratificación del clasificador.")
    group_cv.add_argument(
        "--classifier_exclude_subject_ids",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Default-off guard for branch-specific diagnostics: SubjectID values to "
            "exclude from the supervised AD/CN classifier pool only. The subjects "
            "remain eligible for the unsupervised VAE pool if otherwise valid."
        ),
    )
    #group_cv.add_argument("--classifier_hp_tune_ratio", type=float, default=0.25, help="Proporción de datos de train/dev para ajuste de HP.")
    group_vae = parser.add_argument_group('VAE Model and Training')
    group_vae.add_argument("--num_conv_layers_encoder", type=int, default=4, choices=[3, 4], help="Capas convolucionales en encoder VAE.") 
    group_vae.add_argument("--decoder_type", type=str, default="convtranspose", choices=["upsample_conv", "convtranspose"], help="Tipo de decoder para VAE.") 
    group_vae.add_argument("--latent_dim", type=int, default=128, help="Dimensión del espacio latente VAE. (Recomendado: 128-256)")
    group_vae.add_argument("--lr_vae", type=float, default=1e-4, help="Tasa de aprendizaje VAE.")
    group_vae.add_argument("--epochs_vae", type=int, default=800, help="Épocas máximas para VAE.")
    group_vae.add_argument("--batch_size", type=int, default=32, help="Tamaño del batch.")
    group_vae.add_argument("--beta_vae", type=float, default=1.0, help="Peso KLD (beta_max para annealing).")
    group_vae.add_argument(
        "--recon_loss_mode",
        type=str,
        default=RECON_LOSS_MODE_CURRENT,
        choices=list(RECON_LOSS_MODES),
        help=(
            "Modo de pérdida de reconstrucción VAE. "
            "mse_sum_batchmean_current preserva el comportamiento histórico. "
            "offdiag_channelmean_sum y mse_offdiag_channel_mean_sum usan sólo off-diagonal, "
            "suman por canal y promedian canales/batch."
        ),
    )
    group_vae.add_argument(
        "--recon_loss_channel_weights",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Pesos por canal seleccionado para mse_offdiag_channel_weighted_sum. "
            "Deben ser finitos, no negativos, tener la misma longitud que channels_to_use y sumar 1."
        ),
    )
    group_vae.add_argument("--cyclical_beta_n_cycles", type=int, default=4, help="Ciclos para annealing de beta.")
    group_vae.add_argument("--cyclical_beta_ratio_increase", type=float, default=0.4, help="Proporción de ciclo para aumentar beta. (Recomendado: 0.4)")
    group_vae.add_argument("--weight_decay_vae", type=float, default=1e-5, help="Decaimiento de peso (L2 reg) para VAE.")
    group_vae.add_argument("--vae_final_activation", type=str, default="tanh", choices=["sigmoid", "tanh", "linear", "none"], help="Activación final del decoder VAE.")
    group_vae.add_argument("--intermediate_fc_dim_vae", type=str, default="quarter", help="Dimensión FC intermedia en VAE ('0', 'half', 'quarter', o entero).")
    group_vae.add_argument("--dropout_rate_vae", type=float, default=0.2, help="Tasa de dropout en VAE.")
    group_vae.add_argument(
        "--encoder_dropout_rate_vae",
        type=float,
        default=None,
        help=(
            "Optional encoder hidden dropout rate. If omitted, encoder_conv and encoder_fc "
            "use --dropout_rate_vae exactly as in legacy configs."
        ),
    )
    group_vae.add_argument(
        "--decoder_dropout_rate_vae",
        type=float,
        default=None,
        help=(
            "Optional decoder hidden dropout rate. If omitted, decoder_fc and decoder_conv "
            "use --dropout_rate_vae exactly as in legacy configs."
        ),
    )
    group_vae.add_argument(
        "--vae_dropout_scope",
        type=str,
        default="legacy_all",
        choices=list(DROPOUT_SCOPE_CHOICES),
        help=(
            "Ámbito del dropout VAE. legacy_all preserva el comportamiento histórico; "
            "encoder_only/no_decoder_dropout quitan dropout del decoder; none desactiva dropout explícito."
        ),
    )
    group_vae.add_argument(
        "--vae_block_order",
        type=str,
        default="legacy_act_norm",
        choices=list(BLOCK_ORDER_CHOICES),
        help=(
            "Orden de bloques VAE. legacy_act_norm preserva el comportamiento histórico "
            "(Conv/Linear -> GELU -> Norm -> Dropout en conv; FC histórico). "
            "norm_act usa Conv/Linear -> Norm -> GELU -> Dropout."
        ),
    )
    group_vae.add_argument(
        "--vae_conditioning_mode",
        type=str,
        default="none",
        choices=list(CONDITIONING_MODE_CHOICES),
        help=(
            "Modo condicional del VAE. none preserva comportamiento histórico; "
            "decoder_only concatena covariables a z antes del decoder FC; "
            "encoder_decoder concatena Age al encoder FC antes de mu/logvar y a z antes del decoder."
        ),
    )
    group_vae.add_argument(
        "--vae_conditioning_vars",
        type=str,
        default="none",
        choices=list(VAE_CONDITIONING_VAR_CHOICES),
        help=(
            "Covariables para condicionar el VAE. Age se estandariza fold-localmente; "
            "Sex se codifica fold-localmente. encoder_decoder sólo permite Age."
        ),
    )
    group_vae.add_argument(
        "--vae_latent_covariate_corr_lambda",
        type=float,
        default=0.0,
        help="Peso opcional para penalización diferenciable de correlación cuadrada entre mu y covariables condicionantes.",
    )
    group_vae.add_argument("--use_layernorm_vae_fc", action='store_true', help="Usar LayerNorm en capas FC del VAE.")
    group_vae.add_argument(
        "--vae_encoder_norm_mode",
        type=str,
        default="groupnorm",
        choices=["groupnorm", "layernorm", "none"],
        help="Normalización de bloques convolucionales del encoder. groupnorm preserva el comportamiento histórico.",
    )
    group_vae.add_argument(
        "--vae_channel_dropout_p",
        type=float,
        default=0.0,
        help="Dropout de canales completos aplicado sólo a la entrada de entrenamiento VAE; target de reconstrucción sin corromper.",
    )
    group_vae.add_argument(
        "--vae_train_sampler_strategy",
        type=str,
        default=VAE_TRAIN_SAMPLER_NONE,
        choices=list(VAE_TRAIN_SAMPLER_STRATEGIES),
        help="Sampler VAE opcional y fold-local. none preserva comportamiento histórico.",
    )
    group_vae.add_argument(
        "--vae_pool_composition_strategy",
        type=str,
        default=VAE_POOL_CURRENT_ALL,
        choices=list(VAE_POOL_COMPOSITION_STRATEGIES),
        help=(
            "Estrategia fold-local para componer el pool VAE antes del split train/val. "
            "current_all_pool preserva el comportamiento histórico. Las otras opciones "
            "usan etiquetas diagnósticas para componer el pool y deben tratarse como exploratorias."
        ),
    )
    group_vae.add_argument(
        "--input_harmonization_mode",
        type=str,
        default="none",
        choices=["none", "foldwise_combat"],
        help=(
            "Default-off pre-VAE input harmonization. 'foldwise_combat' fits ComBat only on "
            "outer train/dev VAE-pool subjects and applies frozen transforms to classifier "
            "train/dev and outer test tensors."
        ),
    )
    group_vae.add_argument(
        "--input_harmonization_batch_col",
        type=str,
        default="Manufacturer",
        help="Batch column for foldwise input harmonization. Branch B uses Manufacturer.",
    )
    group_vae.add_argument(
        "--input_harmonization_covariates",
        type=str,
        nargs="*",
        default=["Age", "Sex"],
        help="Protected covariates for foldwise input harmonization. Diagnosis must not be included.",
    )
    group_vae.add_argument(
        "--input_harmonization_excluded_covariates",
        type=str,
        nargs="*",
        default=["ResearchGroup_Mapped"],
        help="Covariates explicitly excluded from foldwise input harmonization.",
    )
    group_vae.add_argument(
        "--input_harmonization_fit_scope",
        type=str,
        default="outer_train_dev_only",
        choices=["outer_train_dev_only"],
        help="Fit scope for foldwise input harmonization.",
    )
    group_vae.add_argument(
        "--input_harmonization_vectorization",
        type=str,
        default="upper_offdiag_by_channel",
        choices=["upper_offdiag_by_channel"],
        help="Vectorization used by foldwise input harmonization.",
    )
    group_vae.add_argument("--vae_val_split_ratio", type=float, default=0.2, help="Proporción para validación VAE.")
    group_vae.add_argument("--vae_stratify_cols", type=str, nargs='*', default=None, help="Columnas adicionales para estratificar el split interno train/val del VAE. Siempre se antepone ResearchGroup_Mapped. Default None preserva el comportamiento histórico ResearchGroup_Mapped+Sex+Age_Group.")
    group_vae.add_argument(
        "--vae_required_metadata_cols",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Si se pasa, los sujetos del pool VAE que tengan valores missing en CUALQUIERA "
            "de estas columnas son removidos del pool antes del split train/val, con QC CSV por fold. "
            "Default None preserva el comportamiento actual sin ningún filtro adicional. "
            "NO afecta el pool del clasificador AD/CN."
        ),
    )
    group_vae.add_argument(
        "--vae_abort_if_val_split_fails",
        action="store_true",
        default=False,
        help=(
            "If set, abort the entire run when the VAE internal validation split fails "
            "(e.g. due to singleton strata from tensor subjects absent from metadata). "
            "Default False preserves the existing silent fallback to full-pool train / empty val. "
            "Recommended for metadata-rescue runs to prevent silent disablement of early stopping."
        ),
    )
    group_vae.add_argument("--early_stopping_patience_vae", type=int, default=20, help="Paciencia early stopping VAE. (Recomendado: 15-20)")
    group_vae.add_argument("--lr_scheduler_patience_vae", type=int, default=15, help="Paciencia para el scheduler ReduceLROnPlateau del VAE.")
    # ▼▼▼ NUEVOS ARGUMENTOS ▼▼▼
    group_vae.add_argument("--lr_scheduler_type", type=str, default="plateau", choices=["plateau", "cosine_warm"], help="Tipo de scheduler para el VAE.")
    group_vae.add_argument("--lr_scheduler_T0", type=int, default=50, help="Épocas para el primer reinicio en CosineAnnealingWarmRestarts.")
    group_vae.add_argument("--lr_scheduler_eta_min", type=float, default=1e-7, help="Tasa de aprendizaje mínima para CosineAnnealingWarmRestarts.")
    group_vae.add_argument(
        "--save_vae_checkpoints_every_n_epochs",
        type=int,
        default=None,
        help="Si se define, guarda checkpoints VAE periódicos por fold cada N épocas. Default None preserva el comportamiento histórico.",
    )
    group_vae.add_argument(
        "--save_vae_checkpoints_start_epoch",
        type=int,
        default=None,
        help="Epoch inicial 1-based para empezar a guardar checkpoints periódicos. Default None/0 preserva el comportamiento histórico.",
    )
    group_vae.add_argument(
        "--save_vae_checkpoints_keep_last_n",
        type=int,
        default=None,
        help="Si se define, conserva solo los N checkpoints periódicos más recientes por fold. No afecta checkpoints best/final protegidos.",
    )
    group_vae.add_argument(
        "--save_vae_checkpoints_always_keep_best",
        action="store_true",
        help="Incluye el checkpoint best/final existente en el manifest como protegido contra pruning.",
    )
    group_vae.add_argument(
        "--save_vae_checkpoints_always_keep_final",
        action="store_true",
        help="Protege el checkpoint terminal/final de entrenamiento contra pruning cuando hay guardado periódico.",
    )
    # ▲▲▲ FIN NUEVOS ARGUMENTOS ▲▲▲


    group_clf = parser.add_argument_group('Classifier')
    
    clf_choices = get_available_classifiers()
    group_clf.add_argument(
        "--classifier_types", nargs="+", default=["rf", "svm", "gb"],
        choices=clf_choices,
        help=f"Tipos de clasificadores a entrenar. Disponibles: {', '.join(clf_choices)}"
    )

    group_clf.add_argument("--use_optuna_pruner", action="store_true", 
                        help="Usar MedianPruner de Optuna para acelerar la búsqueda de HPs.") # <-- AÑADE ESTA LÍNEA
    group_clf.add_argument("--latent_features_type", type=str, default="mu", choices=["mu", "z"], help="Usar 'mu' o 'z' como features latentes.")
    group_clf.add_argument("--gridsearch_scoring", type=str, default="roc_auc", help="Métrica para la búsqueda de HP (OptunaSearchCV).")
    
    group_clf.add_argument("--classifier_use_class_weight", action="store_true", help="Usar class_weight='balanced' en clasificadores que lo soporten.")
    group_clf.add_argument("--classifier_calibrate", action="store_true", help="Aplicar calibración de probabilidad a los clasificadores (CalibratedClassifierCV).")
    group_clf.add_argument("--use_smote", action="store_true", help="Usar SMOTE en el pipeline. (Recomendado activar)")
    group_clf.add_argument("--tune_sampler_params", action="store_true", help="Incluir hiperparámetros de SMOTE en la búsqueda de RandomizedSearch.")
    group_clf.add_argument("--mlp_classifier_hidden_layers", type=str, default="64,16", help="Capas ocultas para el clasificador MLP.")
    group_clf.add_argument(
        "--metadata_features", nargs="*", default=None,
        help="Lista de columnas de metadatos para añadir como features al clasificador (ej: Age Sex Years_of_Education)."
    )
    group_clf.add_argument("--n_iter_logreg", type=int, default=None, help="Override opcional de trials Optuna para logreg. Si se omite, conserva el default del factory.")
    group_clf.add_argument("--n_iter_svm", type=int, default=None, help="Override opcional de trials Optuna para svm. Si se omite, conserva el default del factory.")
    group_clf.add_argument("--n_iter_rf", type=int, default=None, help="Override opcional de trials Optuna para rf. Si se omite, conserva el default del factory.")
    group_clf.add_argument("--n_iter_gb", type=int, default=None, help="Override opcional de trials Optuna para gb. Si se omite, conserva el default del factory.")
    group_clf.add_argument("--n_iter_xgb", type=int, default=None, help="Override opcional de trials Optuna para xgb. Si se omite, conserva el default del factory.")
    group_clf.add_argument("--n_iter_mlp", type=int, default=None, help="Override opcional de trials Optuna para mlp. Si se omite, conserva el default del factory.")
    group_clf.add_argument(
        "--classifier_n_iter_json",
        type=str,
        default=None,
        help="Overrides opcionales de trials Optuna por clasificador, ej: '{\"logreg\":100,\"svm\":300}'. Los flags --n_iter_* tienen precedencia.",
    )
    
    group_general = parser.add_argument_group('General and Saving Settings')
    group_general.add_argument("--norm_mode", type=str, default="zscore_offdiag", choices=["zscore_offdiag", "minmax_offdiag"], help="Modo de normalización inter-canal.")
    group_general.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")
    group_general.add_argument("--num_workers", type=int, default=4, help="Workers para DataLoader.")
    group_general.add_argument("--n_jobs_gridsearch", type=int, default=4, help="Jobs para RandomizedSearchCV.")
    group_general.add_argument("--log_interval_epochs_vae", type=int, default=10, help="Intervalo de épocas para loguear VAE.")
    group_general.add_argument("--save_fold_artefacts", action='store_true', help="Guardar pipeline de clasificador de cada fold.")
    group_general.add_argument("--save_vae_training_history", action='store_true', help="Guardar historial de entrenamiento del VAE (loss, beta) por fold.")
    # === NUEVO GRUPO QC / RIGOR PUBLICACIÓN ===
    group_qc = parser.add_argument_group('QC / Reproducibility (Paper-Grade)')
    group_qc.add_argument(
        "--qc_analyze_distributions",
        action='store_true',
        help="Analiza y guarda stats de distribución raw/norm/recon por canal, "
             "más histogramas superpuestos, para justificar normalización y activación final del decoder."
    )
    group_qc.add_argument(
        "--qc_check_scanner_leakage",
        action='store_true',
        help="Evalúa cuán predecible es el sitio/escáner a partir del conectoma normalizado vs. la latente mu "
             "(balanced_accuracy en CV). Guarda CSV por fold."
    )    

    group_qc.add_argument(
        "--qc_rate_distortion",
        action="store_true",
        help="Genera tabla Rate–Distortion (D/R) desde history_data del VAE por fold (CSV)."
    )
    group_qc.add_argument(
        "--qc_rd_log_base",
        type=float,
        default=2.0,
        help="Base log para convertir R (nats) a otras unidades (ej 2.0 -> bits)."
    )
    group_qc.add_argument(
        "--qc_latent_information",
        action="store_true",
        help="Calcula métricas de teoría de la información en el espacio latente (MI por dim con Y y nuisances, active units y total correlation)."
    )
    group_qc.add_argument(
        "--qc_nuisance_cols",
        type=str,
        nargs="*",
        default=None,
        help="Columnas de confusores/nuisances para MI (ej: Site Manufacturer Sex Age Age_Group). Si se omite, auto-detecta site+Sex+Age_Group."
    )
    group_qc.add_argument(
        "--qc_mi_n_neighbors",
        type=int,
        default=3,
        help="n_neighbors para estimación kNN de mutual_info_* (sklearn)."
    )
    group_qc.add_argument(
        "--qc_mi_top_k",
        type=int,
        default=10,
        help="Top-K dimensiones latentes reportadas por MI."
    )
    group_qc.add_argument(
        "--qc_var_eps_active",
        type=float,
        default=1e-4,
        help="Umbral var_eps para 'active units' (Var(mu_i) > eps)."
    )
    group_qc.add_argument(
        "--qc_tc_ridge",
        type=float,
        default=1e-6,
        help="Ridge diagonal para estabilidad en total correlation gaussiana."
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        help="Valida argumentos y muestra configuración sin cargar datos ni entrenar.",
    )

    args = parser.parse_args()

    try:
        args.classifier_n_iter_overrides = _parse_classifier_n_iter_overrides(args)
    except ValueError as e:
        parser.error(str(e))
    if args.classifier_n_iter_overrides:
        logger.info(f"Optuna trial overrides requested: {args.classifier_n_iter_overrides}")
    else:
        logger.info("Optuna trial overrides requested: none; using classifier factory defaults.")

    if isinstance(args.intermediate_fc_dim_vae, str) and args.intermediate_fc_dim_vae.lower() not in ["0", "half", "quarter"]:
        try:
            args.intermediate_fc_dim_vae = int(args.intermediate_fc_dim_vae)
        except ValueError:
            logger.error(f"Valor inválido para intermediate_fc_dim_vae: {args.intermediate_fc_dim_vae}. Abortando.")
            exit(1)
    
    if not (0 <= args.vae_val_split_ratio < 1): 
        if args.vae_val_split_ratio != 0:
            logger.warning(f"vae_val_split_ratio ({args.vae_val_split_ratio}) inválido. Se usará 0.")
        args.vae_val_split_ratio = 0
    
    if args.vae_val_split_ratio == 0:
        logger.info("Sin validación VAE, early stopping y LR scheduler para VAE deshabilitados.")
        args.early_stopping_patience_vae = 0 
        args.lr_scheduler_patience_vae = 0
    if not (0.0 <= float(args.vae_channel_dropout_p) < 1.0):
        parser.error("--vae_channel_dropout_p must be >= 0 and < 1.")
    if args.vae_conditioning_mode == "none":
        if args.vae_conditioning_vars != "none":
            parser.error("--vae_conditioning_vars must be none when --vae_conditioning_mode=none.")
        if float(args.vae_latent_covariate_corr_lambda) != 0.0:
            parser.error("--vae_latent_covariate_corr_lambda must be 0 when --vae_conditioning_mode=none.")
    elif args.vae_conditioning_mode == "decoder_only":
        if args.vae_conditioning_vars == "none":
            parser.error("--vae_conditioning_vars must be Age, sex, age_sex, or manufacturer when --vae_conditioning_mode=decoder_only.")
        if float(args.vae_latent_covariate_corr_lambda) < 0.0:
            parser.error("--vae_latent_covariate_corr_lambda must be >= 0.")
    elif args.vae_conditioning_mode == "encoder_decoder":
        if str(args.vae_conditioning_vars).lower() != "age":
            parser.error("--vae_conditioning_vars must be Age when --vae_conditioning_mode=encoder_decoder.")
        if float(args.vae_latent_covariate_corr_lambda) < 0.0:
            parser.error("--vae_latent_covariate_corr_lambda must be >= 0.")
    else:
        parser.error(f"Unsupported --vae_conditioning_mode={args.vae_conditioning_mode!r}.")

    if args.save_vae_checkpoints_every_n_epochs is not None and args.save_vae_checkpoints_every_n_epochs <= 0:
        logger.warning(
            f"save_vae_checkpoints_every_n_epochs={args.save_vae_checkpoints_every_n_epochs} inválido. "
            "Se desactiva guardado periódico de checkpoints VAE."
        )
        args.save_vae_checkpoints_every_n_epochs = None
    if args.save_vae_checkpoints_start_epoch is not None and args.save_vae_checkpoints_start_epoch < 0:
        logger.warning(
            f"save_vae_checkpoints_start_epoch={args.save_vae_checkpoints_start_epoch} inválido. "
            "Se usará 0."
        )
        args.save_vae_checkpoints_start_epoch = 0
    if args.save_vae_checkpoints_keep_last_n is not None and args.save_vae_checkpoints_keep_last_n <= 0:
        logger.warning(
            f"save_vae_checkpoints_keep_last_n={args.save_vae_checkpoints_keep_last_n} inválido. "
            "Se desactiva pruning de checkpoints periódicos."
        )
        args.save_vae_checkpoints_keep_last_n = None
    if args.recon_loss_mode == RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_WEIGHTED_SUM:
        if args.recon_loss_channel_weights is None or len(args.recon_loss_channel_weights) == 0:
            parser.error("--recon_loss_channel_weights is required for mse_offdiag_channel_weighted_sum.")
        weights = np.asarray(args.recon_loss_channel_weights, dtype=float)
        if not np.isfinite(weights).all():
            parser.error("--recon_loss_channel_weights must be finite.")
        if (weights < 0).any():
            parser.error("--recon_loss_channel_weights must be non-negative.")
        if not np.isclose(float(weights.sum()), 1.0, rtol=1e-5, atol=1e-6):
            parser.error(f"--recon_loss_channel_weights must sum to 1.0, got {float(weights.sum())}.")
        if getattr(args, "channels_to_use", None) is not None and len(weights) != len(args.channels_to_use):
            parser.error(
                "--recon_loss_channel_weights length must match --channels_to_use length "
                f"({len(args.channels_to_use)}), got {len(weights)}."
            )
    elif args.recon_loss_channel_weights:
        logger.warning(
            "recon_loss_channel_weights=%s supplied but recon_loss_mode=%s does not use weights.",
            args.recon_loss_channel_weights,
            args.recon_loss_mode,
        )

    if args.dry_run:
        logger.info("DRY-RUN solicitado: no se cargarán datos, no se entrenará VAE/clasificador y no se escribirán resultados.")
        logger.info(
            "VAE objective preview: recon_loss_mode=%s, final_activation=%s, beta_vae=%s, "
            "encoder_norm_mode=%s, dropout_scope=%s, dropout_rate=%s, "
            "encoder_dropout_rate=%s, decoder_dropout_rate=%s, block_order=%s, "
            "conditioning_mode=%s, conditioning_vars=%s, corr_lambda=%s, "
            "channel_dropout_p=%s, train_sampler_strategy=%s, pool_composition_strategy=%s, "
            "input_harmonization_mode=%s, recon_loss_channel_weights=%s",
            args.recon_loss_mode,
            args.vae_final_activation,
            args.beta_vae,
            args.vae_encoder_norm_mode,
            args.vae_dropout_scope,
            args.dropout_rate_vae,
            args.encoder_dropout_rate_vae,
            args.decoder_dropout_rate_vae,
            args.vae_block_order,
            args.vae_conditioning_mode,
            args.vae_conditioning_vars,
            args.vae_latent_covariate_corr_lambda,
            args.vae_channel_dropout_p,
            args.vae_train_sampler_strategy,
            args.vae_pool_composition_strategy,
            args.input_harmonization_mode,
            args.recon_loss_channel_weights,
        )
        sys.exit(0)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        args.git_hash = git_hash
    except Exception:
        args.git_hash = "N/A"
    logger.info(f"Git commit hash: {args.git_hash}")

    args_dump = "\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items()))
    logger.info(f"Run arguments:\n{args_dump}\n------------------------------------")

    global_tensor_data, metadata_df_full, roi_names_in_order, network_labels_in_order = load_data(
        Path(args.global_tensor_path),
        Path(args.metadata_path)
    )

    if global_tensor_data is not None and metadata_df_full is not None:
        # === Consistencia interna tensor ↔ ROI names ===
        n_rois_tensor = global_tensor_data.shape[2]

        if roi_names_in_order is None:
            logger.critical(
                "El archivo .npz NO contiene 'roi_names_in_order'. "
                "Esta versión del pipeline asume que el tensor es autocontenido. Aborta."
            )
            exit(1)

        if len(roi_names_in_order) != n_rois_tensor:
            logger.critical(
                f"Dimensión de ROIs en el tensor ({n_rois_tensor}) "
                f"no coincide con roi_names_in_order ({len(roi_names_in_order)}). Aborta."
            )
            exit(1)

        if global_tensor_data.shape[2] != global_tensor_data.shape[3]:
            logger.critical(
                f"Las matrices de conectividad no son cuadradas: {global_tensor_data.shape[2]}x{global_tensor_data.shape[3]}. Aborta."
            )
            exit(1)

        if network_labels_in_order is not None and len(network_labels_in_order) != len(roi_names_in_order):
            logger.warning(
                "network_labels_in_order tiene longitud distinta a roi_names_in_order. "
                "Se ignorarán las etiquetas de red."
            )
            network_labels_in_order = None

        # === Guardar una copia humana de la info de ROIs en la carpeta de salida ===
        roi_info_df = pd.DataFrame({"roi_name_in_tensor": roi_names_in_order})
        if network_labels_in_order is not None:
            roi_info_df["network_label_in_tensor"] = network_labels_in_order

        roi_info_csv = Path(args.output_dir) / "roi_info_from_tensor.csv"
        roi_info_joblib = Path(args.output_dir) / "roi_info_from_tensor.joblib"
        roi_info_df.to_csv(roi_info_csv, index=False)
        joblib.dump(
            {
                "roi_names_in_order": roi_names_in_order,
                "network_labels_in_order": network_labels_in_order
            },
            roi_info_joblib
        )
        logger.info(f"Información de ROIs guardada en: {roi_info_csv} y {roi_info_joblib}")
        logger.info(f"Primeras 5 ROIs en el tensor: {roi_names_in_order[:5]}")

        # === Paso 1 (paper-grade): Guardar run_config.json para inferencia/reproducibilidad ===
        # Incluye vars(args) + DEFAULT_CHANNEL_NAMES + fingerprint del tensor + ROI/channel names.
        try:
            out_dir = Path(args.output_dir)
            tensor_path = Path(args.global_tensor_path)

            # 1) Fingerprint estable del .npz usado para entrenar
            train_tensor_fingerprint = None
            if tensor_path.exists():
                train_tensor_fingerprint = _compute_file_sha256(tensor_path)

            # 2) Resolver nombres de canales (master + selected) de forma estable
            master_channel_names = list(DEFAULT_CHANNEL_NAMES)
            # Si el tensor tiene más canales que DEFAULT_CHANNEL_NAMES, los completamos con nombres genéricos
            n_chan_tensor = int(global_tensor_data.shape[1])
            if len(master_channel_names) < n_chan_tensor:
                master_channel_names = master_channel_names + [
                    f"RawChan{i}" for i in range(len(master_channel_names), n_chan_tensor)
                ]

            if getattr(args, "channels_to_use", None):
                sel_idx = list(args.channels_to_use)
                try:
                    selected_channel_names = [master_channel_names[i] for i in sel_idx]
                except Exception:
                    selected_channel_names = [f"RawChan{i}" for i in sel_idx]
            else:
                sel_idx = None
                selected_channel_names = master_channel_names[:n_chan_tensor]

            # Inyectar en args para que train_and_evaluate_pipeline use exactamente lo mismo (C1)
            try:
                args.all_original_channel_names = master_channel_names
                args.selected_channel_names = selected_channel_names
            except Exception:
                pass

            run_config = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_hash": args.git_hash,
                "python_version": platform.python_version(),
                "torch_version": getattr(torch, "__version__", None),
                "cuda_available": bool(torch.cuda.is_available()),

                # Core: args + listas canónicas
                "args": vars(args),
                "DEFAULT_CHANNEL_NAMES": list(DEFAULT_CHANNEL_NAMES),
                "channel_names_master_in_tensor_order": master_channel_names,
                "channels_to_use_indices": sel_idx,
                "channel_names_selected": selected_channel_names,

                # Tensor / ROI metadata para compatibilidad en inferencia
                "global_tensor_path": str(tensor_path.resolve()),
                "metadata_path": str(Path(args.metadata_path).resolve()),
                "tensor_shape": [int(x) for x in global_tensor_data.shape],
                "train_tensor_fingerprint_sha256": train_tensor_fingerprint,
                "roi_names_in_order": roi_names_in_order,
                "network_labels_in_order": network_labels_in_order,
            }

            run_config_path = out_dir / "run_config.json"
            _safe_json_dump(run_config, run_config_path)
            logger.info(f"Run config guardado en: {run_config_path}")
        except Exception as e:
            logger.warning(f"No se pudo guardar run_config.json (continúo igual): {e}")
 

        pipeline_start_time = time.time()
        train_and_evaluate_pipeline(global_tensor_data, metadata_df_full, args)
        logger.info(f"Pipeline completo en {time.time() - pipeline_start_time:.2f} segundos.")
    else:
        logger.critical("No se pudieron cargar los datos. Abortando.")


    logger.info("--- Consideraciones Finales ---")
    logger.info(
        f"Normalización: '{args.norm_mode}'. Activación VAE: '{args.vae_final_activation}'. "
        f"Recon loss mode: '{args.recon_loss_mode}'. Dropout scope: '{args.vae_dropout_scope}'. "
        f"Block order: '{args.vae_block_order}'. "
        "Asegurar compatibilidad."
    )

    if args.qc_analyze_distributions:
        logger.info("QC distribuciones ACTIVADO: Se guardaron CSV e histogramas por fold para raw/norm/recon.")
    if args.qc_check_scanner_leakage:
        logger.info("QC leakage ACTIVADO: Se guardaron métricas de separabilidad de sitio/escáner por fold.")
