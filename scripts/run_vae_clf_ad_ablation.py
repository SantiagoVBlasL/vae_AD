#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_vae_clf_ad_ablation.py

Canal-ablation variant of the β-VAE + classifier pipeline for AD vs CN.

INTENTIONAL DIFFERENCE vs. run_vae_clf_ad_inference.py
-------------------------------------------------------
Always uses a simple LogisticRegression (logreg) with NO hyperparameter
tuning (constant ABLATION_CLF = "logreg").  This removes HP-tuning variance
from the scoring function so channel ranking is comparable across runs.

PARITY with run_vae_clf_ad_inference.py
----------------------------------------
- VAE training loop (cyclical β, AMP, history keys including val_loss_modelsel)
- β-max consistent model selection (avg_val_recon + β_max * avg_val_kld)
- Outer-CV stratification guardrail (fallback to label-only)
- VAE pool construction with max_valid_idx_for_cn_ad (no stale variable)
- Robust VAE val-split stratification (3-col composite → fallback)
- Normalization (normalize_inter_channel_fold / apply_normalization_params)
- Memory-safe latent extraction (detach → CPU → del GPU tensors)
- Split & norm artifact saves (train_dev_indices.npy, vae_norm_params.joblib …)
- feature_columns.json per fold
- Raw metadata concat (no manual encoding, no leakage)
- pin_memory conditional on CUDA
- Output file naming: all_folds_metrics_MULTI_*.csv with 'auc' column
- all_folds_clf_predictions_MULTI_*.joblib as list[DataFrame] (inference format)
"""
from __future__ import annotations

# ─── Bootstrap: add src/ to sys.path ─────────────────────────────────────────
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
else:
    raise FileNotFoundError(f"src/ not found at {SRC_DIR}")

# ─── Standard library ────────────────────────────────────────────────────────
import argparse
import copy
import gc
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

# ─── Third-party ──────────────────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    average_precision_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    train_test_split as sk_train_test_split,
)

# ─── Project ──────────────────────────────────────────────────────────────────
from betavae_xai.models import ConvolutionalVAE, get_classifier_and_grid
from betavae_xai.analysis_qc.fold_qc import (
    log_group_distributions,
    compute_latent_silhouette,
)
from betavae_xai.utils.logging import setup_logging
from betavae_xai.utils.run_io import _safe_json_dump
from betavae_xai.data.preprocessing import (
    load_data,
    normalize_inter_channel_fold,
    apply_normalization_params,
)

# ─── Logging (single setup, no duplication) ───────────────────────────────────
logger = setup_logging(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

#: Intentional ablation invariant — do NOT change this to tune HPs.
ABLATION_CLF: str = "logreg"

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
# Helper functions (parity with run_vae_clf_ad_inference.py)
# ─────────────────────────────────────────────────────────────────────────────

def _extend_channel_names_to_tensor(n_chan: int, base: List[str]) -> List[str]:
    """Pad or trim *base* so its length equals *n_chan*."""
    b = list(base) if base else []
    if len(b) < n_chan:
        b += [f"RawChan{i}" for i in range(len(b), n_chan)]
    return b[:n_chan]


def _filter_existing_cols(df: Optional[pd.DataFrame], cols: List[str]) -> List[str]:
    """Return subset of *cols* that actually exist in *df*."""
    if df is None or df.empty:
        return []
    return [c for c in cols if c in df.columns]


def _get_score_1d(estimator, X) -> np.ndarray:
    """Return a 1-D probability / decision score suitable for AUC."""
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)
        return p[:, 1] if (p.ndim == 2 and p.shape[1] >= 2) else np.asarray(p).ravel()
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X)).ravel()
    return np.asarray(estimator.predict(X)).astype(float).ravel()


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MSE reconstruction loss + β·KLD.  Returns (total, recon.detach(), kld.detach())."""
    recon_x = recon_x.float(); x = x.float()
    mu = mu.float();           logvar = logvar.float()
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum") / x.shape[0]
    kld_loss   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss.detach(), kld_loss.detach()


def get_cyclical_beta_schedule(
    current_epoch: int,
    total_epochs: int,
    beta_max: float,
    n_cycles: int,
    ratio_increase: float = 0.5,
) -> float:
    """Cyclical β annealing schedule (linear ramp then flat per cycle)."""
    if n_cycles <= 0:
        return beta_max
    epoch_per_cycle     = total_epochs / n_cycles
    epoch_in_cycle      = current_epoch % epoch_per_cycle
    increase_phase_dur  = epoch_per_cycle * ratio_increase
    if epoch_in_cycle < increase_phase_dur:
        return beta_max * (epoch_in_cycle / increase_phase_dur)
    return beta_max


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate_pipeline(
    global_tensor_all_channels: np.ndarray,
    metadata_df_full: pd.DataFrame,
    args: argparse.Namespace,
) -> Optional[pd.DataFrame]:
    """
    Run nested-CV β-VAE + LogReg ablation pipeline.

    One intentional difference from inference:
        Classifier is always ``ABLATION_CLF = "logreg"`` with NO HP tuning.
    """
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # ── Channel selection ─────────────────────────────────────────────────────
    master_names = list(
        getattr(args, "all_original_channel_names", DEFAULT_CHANNEL_NAMES)
    )
    master_names = _extend_channel_names_to_tensor(
        int(global_tensor_all_channels.shape[1]), master_names
    )

    if args.channels_to_use is not None:
        sel_idx: List[int] = list(args.channels_to_use)
        try:
            sel_names = [master_names[i] for i in sel_idx]
        except IndexError:
            logger.error("IndexError mapping channel indices to names; using generic names.")
            sel_names = [f"RawChan{i}" for i in sel_idx]
        current_tensor = global_tensor_all_channels[:, sel_idx, :, :]
        logger.info(f"Selected channel indices: {sel_idx}")
        logger.info(f"Selected channel names:   {sel_names}")
    else:
        current_tensor = global_tensor_all_channels
        sel_names      = master_names
        logger.info(f"Using all {current_tensor.shape[1]} channels.")

    n_ch = current_tensor.shape[1]

    # ── Validate required metadata columns ────────────────────────────────────
    required_cols = ["ResearchGroup_Mapped", "tensor_idx"]
    missing_req = [c for c in required_cols if c not in metadata_df_full.columns]
    if missing_req:
        logger.error(f"Missing required metadata columns: {missing_req}. Aborting.")
        return None

    cn_ad_df = metadata_df_full[
        metadata_df_full["ResearchGroup_Mapped"].isin(["CN", "AD"])
    ].copy()
    if cn_ad_df.empty:
        logger.error("No CN/AD subjects found. Aborting.")
        return None

    max_valid_idx_for_cn_ad = int(current_tensor.shape[0] - 1)
    n_orig = len(cn_ad_df)
    cn_ad_df = cn_ad_df[cn_ad_df["tensor_idx"] <= max_valid_idx_for_cn_ad].copy()
    if len(cn_ad_df) < n_orig:
        logger.warning(
            f"Filtered {n_orig - len(cn_ad_df)} subjects with tensor_idx > "
            f"{max_valid_idx_for_cn_ad}."
        )
    if cn_ad_df.empty:
        logger.error("No valid CN/AD subjects after tensor_idx filter. Aborting.")
        return None

    cn_ad_df["label"] = cn_ad_df["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1})

    # ── Build outer-CV stratification key ────────────────────────────────────
    strat_cols: List[str] = ["ResearchGroup_Mapped"]
    if args.classifier_stratify_cols:
        for col in args.classifier_stratify_cols:
            if col in cn_ad_df.columns:
                cn_ad_df[col] = cn_ad_df[col].fillna(f"{col}_Unknown").astype(str)
                if col not in strat_cols:
                    strat_cols.append(col)
            else:
                logger.warning(f"Stratification column '{col}' not in metadata.")

    cn_ad_df["stratify_key_clf"] = cn_ad_df[strat_cols].apply(
        lambda r: "_".join(r.astype(str)), axis=1
    )
    logger.info(f"Outer-CV stratification columns: {strat_cols}")

    X_idx    = np.arange(len(cn_ad_df))
    y_labels = cn_ad_df["label"].values

    # ── Guard rail: strata too small → fall back to label-only ───────────────
    y_outer = cn_ad_df["stratify_key_clf"]
    try:
        vc = pd.Series(y_outer).value_counts()
        if (vc < args.outer_folds).any():
            logger.warning(
                f"Outer-CV: some strata too small with {strat_cols} "
                f"(min={int(vc.min())} < n_splits={args.outer_folds}). "
                "Falling back to label-only stratification."
            )
            y_outer = y_labels
    except Exception as exc:
        logger.warning(f"Outer-CV: cannot validate strata ({exc}). Fallback to label.")
        y_outer = y_labels

    logger.info(
        f"CN/AD subjects: {len(cn_ad_df)}  |  CN={int((y_labels == 0).sum())}, "
        f"AD={int((y_labels == 1).sum())}"
    )

    # ── Outer CV splitter ─────────────────────────────────────────────────────
    if args.repeated_outer_folds_n_repeats > 1:
        outer_cv = RepeatedStratifiedKFold(
            n_splits=args.outer_folds,
            n_repeats=args.repeated_outer_folds_n_repeats,
            random_state=args.seed,
        )
        total_outer = args.outer_folds * args.repeated_outer_folds_n_repeats
    else:
        outer_cv = StratifiedKFold(
            n_splits=args.outer_folds, shuffle=True, random_state=args.seed
        )
        total_outer = args.outer_folds
    logger.info(f"Outer CV: {type(outer_cv).__name__} — {total_outer} iterations.")

    all_folds_metrics:         List[Dict[str, Any]]       = []
    all_folds_vae_history:     List[Optional[Dict]]        = []
    all_folds_clf_predictions: List[pd.DataFrame]          = []

    for fold_idx, (train_dev_local, test_local) in enumerate(
        outer_cv.split(X_idx, y_outer)
    ):
        fold_start = time.time()
        fold_tag   = f"Fold {fold_idx + 1}/{total_outer}"
        logger.info(f"--- Starting {fold_tag} ---")

        fold_out = output_base_dir / f"fold_{fold_idx + 1}"
        fold_out.mkdir(parents=True, exist_ok=True)

        # ── Save split indices ────────────────────────────────────────────────
        np.save(fold_out / "train_dev_indices.npy", train_dev_local)
        np.save(fold_out / "test_indices.npy",      test_local)

        cn_ad_df["tensor_idx"] = cn_ad_df["tensor_idx"].astype(int)
        global_idx_test = cn_ad_df.iloc[test_local]["tensor_idx"].values
        cols_log = _filter_existing_cols(cn_ad_df, strat_cols)
        log_group_distributions(
            cn_ad_df.iloc[test_local], cols_log, "Test Set (Clf)", fold_tag
        )

        # ── VAE training pool = all valid metadata subjects minus test set ────
        all_valid = (
            metadata_df_full
            .loc[metadata_df_full["tensor_idx"] <= max_valid_idx_for_cn_ad, "tensor_idx"]
            .astype(int)
            .to_numpy()
        )
        test_global = np.asarray(global_idx_test, dtype=int)
        vae_pool_global = np.setdiff1d(
            np.unique(all_valid), np.unique(test_global), assume_unique=False
        )

        if len(vae_pool_global) < 10:
            logger.error(
                f"{fold_tag}: VAE pool too small ({len(vae_pool_global)}). Skipping."
            )
            continue

        # Build pool DataFrame aligned to pool global indices
        vae_pool_df = (
            metadata_df_full
            .set_index("tensor_idx")
            .loc[vae_pool_global]
            .reset_index()
        )
        cols_pool = _filter_existing_cols(
            vae_pool_df, ["ResearchGroup_Mapped", "Sex", "Age_Group"]
        )
        log_group_distributions(vae_pool_df, cols_pool, "VAE Training Pool", fold_tag)
        vae_pool_tensor_orig = current_tensor[vae_pool_global]

        # ── VAE internal val split (stratified; robust fallback) ──────────────
        pool_n    = len(vae_pool_global)
        vae_tr_idx  = np.arange(pool_n, dtype=int)
        vae_val_idx = np.array([], dtype=int)

        if args.vae_val_split_ratio > 0 and pool_n > 10:
            vae_strat_cands = ["ResearchGroup_Mapped", "Sex", "Age_Group"]
            avail = [c for c in vae_strat_cands if c in vae_pool_df.columns]
            if not avail:
                avail = ["ResearchGroup_Mapped"]

            tmp_st = vae_pool_df[avail].copy()
            for col in avail:
                tmp_st[col] = tmp_st[col].fillna(f"{col}_Unknown").astype(str)

            try:
                strat_key = tmp_st.apply(
                    lambda r: "_".join(r.values.astype(str)), axis=1
                )
                if not all(strat_key.value_counts() >= 2):
                    logger.warning(
                        f"  {fold_tag} VAE val strata too small with {avail}. "
                        "Falling back to ResearchGroup_Mapped."
                    )
                    strat_key = (
                        vae_pool_df["ResearchGroup_Mapped"].fillna("RG_Unknown").astype(str)
                    )
                logger.info(f"  {fold_tag} VAE val split stratified by: {avail}")
            except Exception as exc:
                logger.error(
                    f"  {fold_tag} VAE val strat key failed ({exc}). "
                    "Using ResearchGroup_Mapped."
                )
                strat_key = (
                    vae_pool_df["ResearchGroup_Mapped"].fillna("RG_Unknown").astype(str)
                )

            try:
                vae_tr_idx, vae_val_idx = sk_train_test_split(
                    np.arange(pool_n),
                    test_size=args.vae_val_split_ratio,
                    stratify=strat_key,
                    random_state=args.seed + fold_idx + 10,
                    shuffle=True,
                )
                vae_tr_idx  = np.asarray(vae_tr_idx,  dtype=int)
                vae_val_idx = np.asarray(vae_val_idx, dtype=int)
            except ValueError as exc:
                logger.error(
                    f"  {fold_tag} VAE val split failed ({exc}). Using full pool as train."
                )
                vae_tr_idx  = np.arange(pool_n, dtype=int)
                vae_val_idx = np.array([], dtype=int)

        # Save VAE split artifacts
        try:
            np.save(fold_out / "vae_training_pool_tensor_idx.npy",
                    np.asarray(vae_pool_global, dtype=int))
            np.save(fold_out / "vae_actual_train_idx_local_to_pool.npy", vae_tr_idx)
            np.save(fold_out / "vae_internal_val_idx_local_to_pool.npy", vae_val_idx)
        except Exception as exc:
            logger.warning(f"  {fold_tag} Could not save VAE split artifacts: {exc}")

        cols_vae_log = _filter_existing_cols(
            vae_pool_df, ["ResearchGroup_Mapped", "Sex", "Age_Group"]
        )
        log_group_distributions(
            vae_pool_df.iloc[vae_tr_idx], cols_vae_log, "VAE actual train", fold_tag
        )
        if len(vae_val_idx) > 0:
            log_group_distributions(
                vae_pool_df.iloc[vae_val_idx], cols_vae_log, "VAE internal val", fold_tag
            )
        logger.info(
            f"  {fold_tag} VAE train: {len(vae_tr_idx)}, VAE val: {len(vae_val_idx)}"
        )

        # ── Normalization (parity with inference) ──────────────────────────────
        vae_pool_norm, norm_params = normalize_inter_channel_fold(
            vae_pool_tensor_orig,
            vae_tr_idx,
            mode=args.norm_mode,
            selected_channel_original_names=sel_names,
        )
        joblib.dump(norm_params, fold_out / "vae_norm_params.joblib")

        # ── DataLoaders ───────────────────────────────────────────────────────
        pin = bool(torch.cuda.is_available())
        vae_tr_ds  = TensorDataset(
            torch.from_numpy(vae_pool_norm[vae_tr_idx]).float()
        )
        vae_tr_loader = DataLoader(
            vae_tr_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin,
        )
        vae_val_loader: Optional[DataLoader] = None
        if len(vae_val_idx) > 0:
            vae_val_ds = TensorDataset(
                torch.from_numpy(vae_pool_norm[vae_val_idx]).float()
            )
            vae_val_loader = DataLoader(
                vae_val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin,
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"  {fold_tag} Device: {device}")

        # ── VAE model ─────────────────────────────────────────────────────────
        vae_fold_k = ConvolutionalVAE(
            input_channels=n_ch,
            latent_dim=args.latent_dim,
            image_size=current_tensor.shape[-1],
            final_activation=args.vae_final_activation,
            intermediate_fc_dim_config=args.intermediate_fc_dim_vae,
            dropout_rate=args.dropout_rate_vae,
            use_layernorm_fc=args.use_layernorm_vae_fc,
            num_conv_layers_encoder=args.num_conv_layers_encoder,
            decoder_type=args.decoder_type,
        ).to(device)

        optimizer_vae = optim.AdamW(
            vae_fold_k.parameters(),
            lr=args.lr_vae,
            weight_decay=args.weight_decay_vae,
            amsgrad=True,
        )
        scheduler_vae: Optional[Any] = None
        if vae_val_loader:
            if args.lr_scheduler_type == "plateau" and args.lr_scheduler_patience_vae > 0:
                logger.info(f"  {fold_tag} Scheduler: ReduceLROnPlateau")
                scheduler_vae = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_vae, "min",
                    patience=args.lr_scheduler_patience_vae,
                    factor=0.1,
                )
            elif args.lr_scheduler_type == "cosine_warm":
                logger.info(
                    f"  {fold_tag} Scheduler: CosineAnnealingWarmRestarts "
                    f"(T_0={args.lr_scheduler_T0})"
                )
                scheduler_vae = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer_vae,
                    T_0=args.lr_scheduler_T0,
                    eta_min=args.lr_scheduler_eta_min,
                )

        # ── VAE training loop ──────────────────────────────────────────────────
        logger.info(
            f"  {fold_tag} Training VAE "
            f"(decoder={args.decoder_type}, encoder_layers={args.num_conv_layers_encoder})..."
        )
        best_val_loss    = float("inf")
        best_epoch       = 0
        epochs_no_improv = 0
        best_state: Optional[Dict] = None

        history_data: Dict[str, List] = {
            "train_loss": [], "train_recon": [], "train_kld": [],
            "val_loss":   [], "val_recon":   [], "val_kld":   [],
            "val_loss_modelsel": [],  # β-max consistent (epoch-independent)
            "beta": [],
        }

        scaler = GradScaler(enabled=(device.type == "cuda"))

        for epoch in range(args.epochs_vae):
            vae_fold_k.train()
            ep_tr_loss, ep_tr_recon, ep_tr_kld = 0.0, 0.0, 0.0

            current_beta = get_cyclical_beta_schedule(
                epoch, args.epochs_vae, args.beta_vae,
                args.cyclical_beta_n_cycles, args.cyclical_beta_ratio_increase,
            )

            for i, (data,) in enumerate(vae_tr_loader):
                data = data.to(device)
                optimizer_vae.zero_grad(set_to_none=True)
                with autocast(enabled=(device.type == "cuda")):
                    recon_b, mu, logvar, _ = vae_fold_k(data)
                    loss, recon, kld = vae_loss_function(
                        recon_b, data, mu, logvar, beta=current_beta
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer_vae)
                scaler.update()

                if scheduler_vae and args.lr_scheduler_type == "cosine_warm":
                    scheduler_vae.step(epoch + i / len(vae_tr_loader))

                n_b = data.size(0)
                ep_tr_loss  += loss.item()  * n_b
                ep_tr_recon += recon.item() * n_b
                ep_tr_kld   += kld.item()   * n_b

            N_tr = len(vae_tr_loader.dataset)
            history_data["train_loss"].append(ep_tr_loss  / N_tr)
            history_data["train_recon"].append(ep_tr_recon / N_tr)
            history_data["train_kld"].append(ep_tr_kld   / N_tr)
            history_data["beta"].append(current_beta)

            log_msg = (
                f"  {fold_tag} E{epoch+1}/{args.epochs_vae}: "
                f"TrL={history_data['train_loss'][-1]:.2f} "
                f"(R={history_data['train_recon'][-1]:.2f},"
                f"KLD={history_data['train_kld'][-1]:.2f}), "
                f"β={current_beta:.3f}, LR={optimizer_vae.param_groups[0]['lr']:.2e}"
            )

            if vae_val_loader:
                vae_fold_k.eval()
                ep_val_cb, ep_val_recon, ep_val_kld = 0.0, 0.0, 0.0
                with torch.no_grad():
                    with autocast(enabled=(device.type == "cuda")):
                        for (val_data,) in vae_val_loader:
                            val_data = val_data.to(device)
                            rv, mv, lv, _ = vae_fold_k(val_data)
                            vl_cb, vr, vkld = vae_loss_function(
                                rv, val_data, mv, lv, beta=current_beta
                            )
                            n_vb = val_data.size(0)
                            ep_val_cb   += vl_cb.item() * n_vb
                            ep_val_recon += vr.item()   * n_vb
                            ep_val_kld   += vkld.item() * n_vb

                N_val           = len(vae_val_loader.dataset)
                avg_val_recon   = ep_val_recon / N_val
                avg_val_kld     = ep_val_kld   / N_val
                avg_val_loss_cb = ep_val_cb    / N_val

                # β-max consistent metric (parity with inference)
                avg_val_loss_bmax = avg_val_recon + args.beta_vae * avg_val_kld

                history_data["val_loss"].append(avg_val_loss_cb)
                history_data["val_recon"].append(avg_val_recon)
                history_data["val_kld"].append(avg_val_kld)
                history_data["val_loss_modelsel"].append(avg_val_loss_bmax)

                log_msg += (
                    f" | ValL(curβ)={avg_val_loss_cb:.2f} "
                    f"(R={avg_val_recon:.2f},KLD={avg_val_kld:.2f}) "
                    f"| ValL(βmax)={avg_val_loss_bmax:.2f}"
                )

                old_lr = optimizer_vae.param_groups[0]["lr"]
                if scheduler_vae and args.lr_scheduler_type == "plateau":
                    scheduler_vae.step(avg_val_loss_bmax)
                new_lr = optimizer_vae.param_groups[0]["lr"]
                if new_lr < old_lr:
                    logger.info(
                        f"  {fold_tag} LR reduced to {new_lr:.2e}; "
                        "resetting early-stop counter."
                    )
                    epochs_no_improv = 0

                # Checkpoint based on β-max consistent loss (parity with inference)
                if avg_val_loss_bmax < best_val_loss and not np.isnan(avg_val_loss_bmax):
                    best_val_loss    = avg_val_loss_bmax
                    best_epoch       = epoch + 1
                    epochs_no_improv = 0
                    best_state       = copy.deepcopy(vae_fold_k.state_dict())
                else:
                    epochs_no_improv += 1

                if (
                    args.early_stopping_patience_vae > 0
                    and epochs_no_improv >= args.early_stopping_patience_vae
                ):
                    logger.info(
                        f"  {fold_tag} Early stop at epoch {epoch + 1}. "
                        f"Best ValL(βmax)={best_val_loss:.4f} (epoch {best_epoch})."
                    )
                    break
            else:
                # No val loader: fill history with NaN, snapshot last epoch
                for key in ["val_loss", "val_recon", "val_kld", "val_loss_modelsel"]:
                    history_data[key].append(np.nan)
                best_state = copy.deepcopy(vae_fold_k.state_dict())

            if (epoch + 1) % args.log_interval_epochs_vae == 0 or epoch == args.epochs_vae - 1:
                logger.info(log_msg)

        # ── Load best VAE checkpoint ───────────────────────────────────────────
        if best_state:
            vae_fold_k.load_state_dict(best_state)
            val_str = (
                f"{best_val_loss:.4f} (βmax)"
                if vae_val_loader and not np.isnan(best_val_loss)
                else "N/A - Last Epoch"
            )
            logger.info(f"  {fold_tag} Loaded best VAE (ValL(βmax)={val_str}).")

        vae_fname = f"vae_model_fold_{fold_idx + 1}.pt"
        torch.save(vae_fold_k.state_dict(), fold_out / vae_fname)
        logger.info(f"  {fold_tag} VAE saved: {fold_out / vae_fname}")

        # ── Save VAE training history ──────────────────────────────────────────
        if args.save_vae_training_history:
            joblib.dump(
                history_data,
                fold_out / f"vae_train_history_fold_{fold_idx + 1}.joblib",
            )
            try:
                fig, ax1 = plt.subplots(figsize=(12, 6))
                ax1.plot(history_data["train_loss"],  label="Train Loss",       color="blue",    lw=2)
                ax1.plot(history_data["train_recon"], label="Train Recon",      color="cyan",    ls=":")
                ax1.plot(history_data["train_kld"],   label="Train KLD",        color="magenta", ls=":")
                if vae_val_loader and any(not np.isnan(x) for x in history_data["val_loss"]):
                    ax1.plot(history_data["val_loss"],        label="Val Loss (curβ)", color="orange", lw=2)
                    ax1.plot(history_data["val_loss_modelsel"], label="Val Loss (βmax)", color="red", ls="-.", lw=2)
                    ax1.plot(history_data["val_recon"],       label="Val Recon",       color="#ff9966", ls="--")
                    ax1.plot(history_data["val_kld"],         label="Val KLD",         color="#ff66b2", ls="--")
                ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
                ax1.set_title(f"Fold {fold_idx + 1} VAE Training History (ablation)")
                ax1.legend(loc="upper left"); ax1.grid(True, ls="--", alpha=0.6)
                ax1.set_ylim(bottom=0)
                ax2 = ax1.twinx()
                ax2.plot(history_data["beta"], label="Beta", color="green", ls="-.", alpha=0.8)
                ax2.set_ylabel("Beta", color="green")
                ax2.tick_params(axis="y", labelcolor="green")
                ax2.legend(loc="upper right")
                fig.tight_layout()
                plt.savefig(fold_out / f"vae_train_history_fold_{fold_idx + 1}.png")
                plt.close(fig)
            except Exception as exc:
                logger.warning(f"  {fold_tag} Could not save VAE history plot: {exc}")

        all_folds_vae_history.append(
            history_data if args.save_vae_training_history else None
        )

        # ── Latent extraction (memory-safe) ────────────────────────────────────
        clf_train_dev_df        = cn_ad_df.iloc[train_dev_local].copy()
        global_idx_train_dev    = clf_train_dev_df["tensor_idx"].values
        y_clf_train_dev         = clf_train_dev_df["label"].values

        log_group_distributions(
            clf_train_dev_df, strat_cols, "Clf Train/Dev Pool", fold_tag
        )

        vae_fold_k.eval()
        with torch.no_grad():
            # --- Train/Dev ---
            norm_td = apply_normalization_params(
                current_tensor[global_idx_train_dev], norm_params
            )
            _, mu_td, _, z_td = vae_fold_k(
                torch.from_numpy(norm_td).float().to(device)
            )
            mu_td_np = mu_td.detach().cpu().numpy()
            z_td_np  = z_td.detach().cpu().numpy()
            del mu_td, z_td
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del norm_td
            gc.collect()

            X_np_td      = mu_td_np if args.latent_features_type == "mu" else z_td_np
            feature_names = [f"latent_{i}" for i in range(X_np_td.shape[1])]
            X_train_dev   = pd.DataFrame(X_np_td, columns=feature_names)

            # --- Test ---
            norm_test = apply_normalization_params(
                current_tensor[global_idx_test], norm_params
            )
            mu_test_np: Optional[np.ndarray] = None
            if norm_test is not None and norm_test.shape[0] > 0:
                _, mu_test, _, z_test = vae_fold_k(
                    torch.from_numpy(norm_test).float().to(device)
                )
                mu_test_np = mu_test.detach().cpu().numpy()
                z_test_np  = z_test.detach().cpu().numpy()
                del mu_test, z_test
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                X_np_test    = mu_test_np if args.latent_features_type == "mu" else z_test_np
                X_test_final = pd.DataFrame(X_np_test, columns=feature_names)
            else:
                X_test_final = pd.DataFrame(columns=feature_names)
            del norm_test
            gc.collect()

        y_test = y_labels[test_local]

        # ── Minimal latent QC (silhouette) ─────────────────────────────────────
        sil = np.nan
        if mu_test_np is not None and len(y_test) > 2:
            try:
                sil = compute_latent_silhouette(
                    latent_feats=mu_test_np, labels_binary=y_test
                )
            except Exception as exc:
                logger.warning(f"  {fold_tag} Silhouette computation failed: {exc}")
        pd.DataFrame([{
            "fold":             fold_idx + 1,
            "silhouette_latent": sil,
            "latent_dim":       args.latent_dim,
            "beta_max":         args.beta_vae,
            "n_channels":       n_ch,
        }]).to_csv(fold_out / "latent_qc_metrics.csv", index=False)

        # ── Metadata features (raw concat — no leakage) ────────────────────────
        if args.metadata_features:
            avail_meta = _filter_existing_cols(clf_train_dev_df, args.metadata_features)
            if len(avail_meta) < len(args.metadata_features):
                missing_m = [c for c in args.metadata_features if c not in clf_train_dev_df.columns]
                logger.warning(f"  {fold_tag} Missing metadata cols: {missing_m}")
            if avail_meta:
                X_train_dev.reset_index(drop=True, inplace=True)
                meta_td = clf_train_dev_df[avail_meta].reset_index(drop=True)
                X_train_dev = pd.concat([X_train_dev, meta_td], axis=1)
                logger.info(f"  {fold_tag} Combined features shape: {X_train_dev.shape}")

                if not X_test_final.empty:
                    X_test_final.reset_index(drop=True, inplace=True)
                    meta_test = cn_ad_df.iloc[test_local][avail_meta].reset_index(drop=True)
                    X_test_final = pd.concat([X_test_final, meta_test], axis=1)

        # ── Save feature column order ──────────────────────────────────────────
        try:
            _safe_json_dump(
                {
                    "fold":                 int(fold_idx + 1),
                    "latent_features_type": str(args.latent_features_type),
                    "latent_feature_names": list(feature_names),
                    "final_feature_columns": list(X_train_dev.columns),
                    "has_metadata_features": bool(args.metadata_features),
                    "metadata_features":    list(args.metadata_features) if args.metadata_features else None,
                    "ablation_clf":         ABLATION_CLF,
                    "note":                 "Ablation: always logreg, no HP tuning.",
                },
                fold_out / "feature_columns.json",
            )
        except Exception as exc:
            logger.warning(f"  {fold_tag} Could not save feature_columns.json: {exc}")

        # ── Classifier: ALWAYS LogReg, NO HP tuning ────────────────────────────
        # ⚡ INTENTIONAL DIFFERENCE vs. run_vae_clf_ad_inference.py:
        #    No Optuna, no inner CV, no HP search.  Removes HP-tuning variance
        #    from channel ranking scores.
        logger.info(f"    Classifier: {ABLATION_CLF} — no HP tuning (ablation invariant)")
        try:
            clf_pipeline, _, _ = get_classifier_and_grid(
                classifier_type=ABLATION_CLF,
                seed=args.seed,
                balance=args.classifier_use_class_weight,
                use_smote=False,          # No SMOTE in ablation (invariant)
                tune_sampler_params=False,
                calibrate=False,
            )
        except (ImportError, ValueError) as exc:
            logger.error(f"    Could not build {ABLATION_CLF} pipeline: {exc}. Skipping.")
            continue

        clf_pipeline.fit(X_train_dev, y_clf_train_dev)
        logger.info(f"    {ABLATION_CLF} fitted on {len(y_clf_train_dev)} subjects.")

        # ── Evaluation ────────────────────────────────────────────────────────
        fold_results: Dict[str, Any] = {
            "fold":                  fold_idx + 1,
            "actual_classifier_type": ABLATION_CLF,
            "clf_no_tune":           True,
            "channels_used":         ",".join(sel_names),
            "n_channels":            n_ch,
        }

        if X_test_final.shape[0] > 0:
            y_score = _get_score_1d(clf_pipeline, X_test_final)
            y_pred  = clf_pipeline.predict(X_test_final)

            # Per-fold predictions CSV (inference format)
            has_sid = "SubjectID" in cn_ad_df.columns
            df_preds = pd.DataFrame({
                **({
                    "SubjectID": cn_ad_df.iloc[test_local]["SubjectID"].values
                } if has_sid else {}),
                "tensor_idx":    cn_ad_df.iloc[test_local]["tensor_idx"].values,
                "y_true":        y_test,
                "y_score_raw":   y_score,
                "y_score_final": y_score,
                "y_pred":        y_pred,
            })
            df_preds.to_csv(
                fold_out / f"test_predictions_{ABLATION_CLF}.csv", index=False
            )

            # Accumulate for global predictions joblib (inference format)
            df_preds_extra = df_preds.copy()
            df_preds_extra.insert(0, "fold",            int(fold_idx + 1))
            df_preds_extra.insert(1, "classifier_type", ABLATION_CLF)
            all_folds_clf_predictions.append(df_preds_extra)

            fold_results.update({
                "auc":              roc_auc_score(y_test, y_score),
                "pr_auc":           average_precision_score(y_test, y_score),
                "accuracy":         accuracy_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "sensitivity":      recall_score(y_test, y_pred, pos_label=1, zero_division=0),
                "specificity":      recall_score(y_test, y_pred, pos_label=0, zero_division=0),
                "f1_score":         f1_score(y_test, y_pred,     pos_label=1, zero_division=0),
            })
            logger.info(
                f"  >>> RESULTS {fold_tag}: "
                f"AUC={fold_results['auc']:.4f}, "
                f"BalAcc={fold_results['balanced_accuracy']:.4f}"
            )
        else:
            for m in [
                "auc", "pr_auc", "accuracy", "balanced_accuracy",
                "sensitivity", "specificity", "f1_score",
            ]:
                fold_results[m] = np.nan

        if args.save_fold_artefacts:
            joblib.dump(
                clf_pipeline,
                fold_out / f"classifier_{ABLATION_CLF}_pipeline_fold_{fold_idx + 1}.joblib",
            )

        all_folds_metrics.append(fold_results)

        # ── GPU cleanup ────────────────────────────────────────────────────────
        del vae_fold_k, optimizer_vae, vae_tr_loader, vae_val_loader
        del scheduler_vae, best_state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info(f"  {fold_tag} completed in {time.time() - fold_start:.2f}s.")

    # ── Aggregate and save results ────────────────────────────────────────────
    if not all_folds_metrics:
        logger.warning("No metrics computed for any fold.")
        return None

    metrics_df = pd.DataFrame(all_folds_metrics)
    n_folds    = len(metrics_df)

    logger.info(f"\n--- Performance Summary: {ABLATION_CLF} ({n_folds} folds) ---")
    for metric in [
        "auc", "pr_auc", "accuracy", "balanced_accuracy",
        "sensitivity", "specificity", "f1_score",
    ]:
        if metric in metrics_df.columns and metrics_df[metric].notna().any():
            logger.info(
                f"  {metric:<20}: "
                f"{metrics_df[metric].mean():.4f} ± {metrics_df[metric].std():.4f}"
            )

    # File naming matches inference convention (parseable by notebook_utils)
    fname_suffix = (
        f"logreg_notune_vae{args.decoder_type}{args.num_conv_layers_encoder}l_"
        f"ld{args.latent_dim}_beta{args.beta_vae}_norm{args.norm_mode}_"
        f"ch{n_ch}{'sel' if args.channels_to_use else 'all'}_"
        f"outer{args.outer_folds}x"
        f"{args.repeated_outer_folds_n_repeats if args.repeated_outer_folds_n_repeats > 1 else 1}"
    )

    csv_path = output_base_dir / f"all_folds_metrics_MULTI_{fname_suffix}.csv"
    metrics_df.to_csv(csv_path, index=False)
    logger.info(f"Metrics CSV: {csv_path}")

    summary_path = output_base_dir / f"summary_metrics_MULTI_{fname_suffix}.txt"
    with open(summary_path, "w") as fh:
        fh.write(f"Run Arguments:\n{vars(args)}\n\n")
        fh.write(f"Git Commit Hash: {args.git_hash}\n\n")
        fh.write(
            "ABLATION NOTE: always uses LogReg (no HP tuning) for channel ranking.\n\n"
        )
        fh.write(f"--- Metrics Summary: {ABLATION_CLF} ---\n")
        for metric in [
            "auc", "pr_auc", "accuracy", "balanced_accuracy",
            "sensitivity", "specificity", "f1_score",
        ]:
            if metric in metrics_df.columns and metrics_df[metric].notna().any():
                fh.write(
                    f"  {metric:<20}: "
                    f"{metrics_df[metric].mean():.4f} ± {metrics_df[metric].std():.4f}\n"
                )
        fh.write("\nFull Metrics DataFrame Description:\n")
        fh.write(metrics_df.describe().to_string())
        fh.write("\n")
    logger.info(f"Summary TXT: {summary_path}")

    if args.save_vae_training_history and all_folds_vae_history:
        joblib.dump(
            all_folds_vae_history,
            output_base_dir / f"all_folds_vae_training_history_{fname_suffix}.joblib",
        )

    if all_folds_clf_predictions:
        try:
            preds_df = pd.concat(all_folds_clf_predictions, axis=0, ignore_index=True)
            preds_df.to_csv(
                output_base_dir / f"all_folds_clf_predictions_MULTI_{fname_suffix}.csv",
                index=False,
            )
        except Exception as exc:
            logger.warning(f"Could not save aggregated predictions CSV: {exc}")
        joblib.dump(
            all_folds_clf_predictions,
            output_base_dir / f"all_folds_clf_predictions_MULTI_{fname_suffix}.joblib",
        )

    return metrics_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Ablation pipeline: β-VAE + LogReg (no HP tuning) for AD/CN channel ranking.\n"
            f"INVARIANT: classifier is always '{ABLATION_CLF}' — do NOT pass "
            "--classifier_types to this script."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = parser.add_argument_group("Data and Paths")
    g.add_argument("--global_tensor_path", type=str, required=True,
                   help="Path to global tensor .npz file.")
    g.add_argument("--metadata_path", type=str, required=True,
                   help="Path to metadata CSV file.")
    g.add_argument("--output_dir", type=str,
                   default="results/vae_clf_ablation_v2",
                   help="Output directory.")
    g.add_argument("--channels_to_use", type=int, nargs="*", default=None,
                   help="0-based channel indices to use (subset); omit to use all.")

    g = parser.add_argument_group("Cross-validation")
    g.add_argument("--outer_folds", type=int, default=5)
    g.add_argument("--repeated_outer_folds_n_repeats", type=int, default=1)
    g.add_argument("--classifier_stratify_cols", type=str, nargs="*", default=["Sex"],
                   help="Extra columns for outer-CV stratification (beyond ResearchGroup_Mapped).")

    g = parser.add_argument_group("VAE Model and Training")
    g.add_argument("--num_conv_layers_encoder", type=int, default=4, choices=[3, 4])
    g.add_argument("--decoder_type", type=str, default="convtranspose",
                   choices=["upsample_conv", "convtranspose"])
    g.add_argument("--latent_dim", type=int, default=128)
    g.add_argument("--lr_vae", type=float, default=1e-4)
    g.add_argument("--epochs_vae", type=int, default=300)
    g.add_argument("--batch_size", type=int, default=32)
    g.add_argument("--beta_vae", type=float, default=4.6)
    g.add_argument("--cyclical_beta_n_cycles", type=int, default=4)
    g.add_argument("--cyclical_beta_ratio_increase", type=float, default=0.4)
    g.add_argument("--weight_decay_vae", type=float, default=1e-5)
    g.add_argument("--vae_final_activation", type=str, default="tanh",
                   choices=["sigmoid", "tanh", "linear"])
    g.add_argument("--intermediate_fc_dim_vae", type=str, default="quarter",
                   help="Intermediate FC dim: '0', 'half', 'quarter', or integer.")
    g.add_argument("--dropout_rate_vae", type=float, default=0.2)
    g.add_argument("--use_layernorm_vae_fc", action="store_true")
    g.add_argument("--vae_val_split_ratio", type=float, default=0.2)
    g.add_argument("--early_stopping_patience_vae", type=int, default=20)
    g.add_argument("--lr_scheduler_patience_vae", type=int, default=15)
    g.add_argument("--lr_scheduler_type", type=str, default="plateau",
                   choices=["plateau", "cosine_warm"])
    g.add_argument("--lr_scheduler_T0", type=int, default=50)
    g.add_argument("--lr_scheduler_eta_min", type=float, default=1e-7)

    g = parser.add_argument_group(f"Classifier ({ABLATION_CLF} — ablation invariant)")
    g.add_argument("--latent_features_type", type=str, default="mu",
                   choices=["mu", "z"])
    g.add_argument("--classifier_use_class_weight", action="store_true",
                   help="Use class_weight='balanced' in LogReg.")
    g.add_argument(
        "--metadata_features", nargs="*", default=None,
        help=(
            "Metadata columns to append as features (e.g. Age Sex). "
            "Passed raw — no manual encoding, no imputation (leakage-safe)."
        ),
    )

    g = parser.add_argument_group("General Settings")
    g.add_argument("--norm_mode", type=str, default="zscore_offdiag",
                   choices=["zscore_offdiag", "minmax_offdiag"])
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--num_workers", type=int, default=4)
    g.add_argument("--log_interval_epochs_vae", type=int, default=10)
    g.add_argument("--save_fold_artefacts", action="store_true",
                   help="Save LogReg pipeline per fold.")
    g.add_argument("--save_vae_training_history", action="store_true",
                   help="Save VAE training history (loss, beta) per fold.")

    args = parser.parse_args()

    # ── Validate args ──────────────────────────────────────────────────────────
    if (
        isinstance(args.intermediate_fc_dim_vae, str)
        and args.intermediate_fc_dim_vae.lower() not in ("0", "half", "quarter")
    ):
        try:
            args.intermediate_fc_dim_vae = int(args.intermediate_fc_dim_vae)
        except ValueError:
            logger.error(
                f"Invalid --intermediate_fc_dim_vae: {args.intermediate_fc_dim_vae}. Aborting."
            )
            sys.exit(1)

    if not (0.0 <= args.vae_val_split_ratio < 1.0) and args.vae_val_split_ratio != 0.0:
        logger.warning(
            f"--vae_val_split_ratio={args.vae_val_split_ratio} is invalid; setting to 0."
        )
        args.vae_val_split_ratio = 0.0

    if args.vae_val_split_ratio == 0.0:
        logger.info("No VAE val split: early stopping and LR scheduler disabled.")
        args.early_stopping_patience_vae = 0
        args.lr_scheduler_patience_vae   = 0

    # ── Reproducibility ────────────────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        args.git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
    except Exception:
        args.git_hash = "N/A"
    logger.info(f"Git commit: {args.git_hash}")

    logger.info("--- Ablation Run Configuration ---")
    for k, v in sorted(vars(args).items()):
        logger.info(f"  {k}: {v}")
    logger.info(f"  [INVARIANT] classifier: {ABLATION_CLF} (no HP tuning)")
    logger.info("----------------------------------")

    # ── Load data ──────────────────────────────────────────────────────────────
    global_tensor, metadata_df, roi_names, network_labels = load_data(
        Path(args.global_tensor_path), Path(args.metadata_path)
    )

    if global_tensor is None or metadata_df is None:
        logger.critical("Data loading failed. Aborting.")
        sys.exit(1)

    # ✅ channel names (desde el .npz, o fallback)
    try:
        with np.load(Path(args.global_tensor_path), allow_pickle=True) as npz:
            ch = npz.get("channel_names", None)
        if ch is not None:
            channel_names = [str(x) for x in np.asarray(ch).ravel().tolist()]
        else:
            channel_names = list(DEFAULT_CHANNEL_NAMES)
    except Exception:
        channel_names = list(DEFAULT_CHANNEL_NAMES)

    args.all_original_channel_names = channel_names
    logger.info(f"Loaded channel names: {args.all_original_channel_names}")

    # ✅ roi order (desde roi_names)
    if roi_names is not None:
        roi_path = Path(args.output_dir) / f"roi_order_{global_tensor.shape[2]}.joblib"
        joblib.dump(list(roi_names), roi_path)
        logger.info(f"ROI order saved: {roi_path}")

    t0 = time.time()
    train_and_evaluate_pipeline(global_tensor, metadata_df, args)
    logger.info(f"Pipeline completed in {time.time() - t0:.2f}s.")
