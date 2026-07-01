#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/revision_bspc_2026/run_loso_cv.py

Leave-One-Site-Out (LOSO) evaluation for the BSPC 2026 major revision.

Design:
  - Philips-only cohort (default; configurable via --manufacturer_filter)
  - Outer loop: explicit site hold-out (no StratifiedKFold)
  - Inner loop: OptunaSearchCV for classifier HP tuning (unchanged from baseline)
  - VAE pool: all Philips subjects from NON-held-out sites (CN + AD + MCI)
  - Classifier train/dev: Philips CN/AD from NON-held-out sites
  - Classifier test: Philips CN/AD from held-out site ONLY
  - Hard leakage assertions: held-out site cannot appear in ANY training split

Primary LOSO set (--loso_mode primary):  sites [6, 18, 19, 130, 305]
Strict LOSO set  (--loso_mode strict):   sites [6, 130]

Outputs:
  results/revision_bspc_2026/loso_primary/   (or loso_strict/)
    run_config.json
    loso_site_metrics.csv
    loso_all_predictions.csv
    pooled_metrics.json
    pooled_calibration_curve.csv
    pooled_calibration_plot.png
    pooled_roc_pr_summary.md
    site_006/
      vae_model_site_006.pt
      vae_norm_params.joblib
      train_subjects.csv  / test_subjects.csv
      train_tensor_idx.npy / test_tensor_idx.npy
      vae_pool_tensor_idx.npy
      vae_actual_train_idx_local.npy
      vae_internal_val_idx_local.npy
      leakage_assertions.json
      test_predictions_<clf>.csv
      optuna_trials_<clf>.csv / .joblib
      optuna_best_trial_<clf>.json
      classifier_<clf>_raw_pipeline.joblib
      classifier_<clf>_calibrated_pipeline.joblib (if calibrated)
      classifier_<clf>_final_pipeline.joblib
      latent_qc_metrics.csv
      vae_train_history.joblib / .png
      ...
"""
from __future__ import annotations

# Bootstrap: add src/ to sys.path
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
else:
    raise FileNotFoundError(f"src/ not found at: {SRC_DIR}")

import argparse
import copy
import gc
import json
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import platform

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

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, f1_score,
    average_precision_score, balanced_accuracy_score,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold, train_test_split as sk_train_test_split

import optuna
from optuna.integration import OptunaSearchCV
from optuna.pruners import MedianPruner

# Project modules
from betavae_xai.models import ConvolutionalVAE, get_classifier_and_grid, get_available_classifiers
from betavae_xai.analysis_qc.fold_qc import (
    log_group_distributions,
    compute_latent_silhouette,
    summarize_distribution_stages,
    summarize_rate_distortion_history,
    evaluate_latent_information,
    evaluate_scanner_leakage,
)
from betavae_xai.utils.logging import setup_logging
from betavae_xai.utils.run_io import _safe_json_dump, _compute_file_sha256
from betavae_xai.data.preprocessing import (
    load_data,
    normalize_inter_channel_fold,
    apply_normalization_params,
)

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CHANNEL_NAMES = [
    "Pearson_OMST_GCE_Signed_Weighted",
    "Pearson_Full_FisherZ_Signed",
    "MI_KNN_Symmetric",
    "dFC_AbsDiffMean",
    "dFC_StdDev",
    "DistanceCorr",
    "Granger_F_lag1",
]

LOSO_PRIMARY_SITES = [6, 18, 19, 130, 305]
LOSO_STRICT_SITES  = [6, 130]


# ---------------------------------------------------------------------------
# Helpers (adapted from run_vae_clf_ad_inference.py)
# ---------------------------------------------------------------------------

def _extend_channel_names(n: int, base: List[str]) -> List[str]:
    base = list(base or [])
    if len(base) < n:
        base = base + [f"RawChan{i}" for i in range(len(base), n)]
    return base[:n]


def _filter_existing_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    if df is None or df.empty:
        return []
    return [c for c in cols if c in df.columns]

def _get_prob_1d(estimator, X):
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return np.asarray(p).ravel()
    return None


def _safe_brier(y_true, y_prob):
    if y_prob is None:
        return np.nan
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if not np.all(np.isfinite(y_prob)):
        return np.nan
    if y_prob.min() < 0.0 or y_prob.max() > 1.0:
        return np.nan
    return brier_score_loss(y_true, y_prob)


def _get_score_1d(estimator, X):
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return np.asarray(p).ravel()
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X)).ravel()
    return np.asarray(estimator.predict(X)).astype(float).ravel()


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    recon_x = recon_x.float(); x = x.float()
    mu = mu.float(); logvar = logvar.float()
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum") / x.shape[0]
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    return recon_loss + beta * kld_loss, recon_loss.detach(), kld_loss.detach()


def get_cyclical_beta(epoch, total_epochs, beta_max, n_cycles, ratio_increase=0.5):
    if n_cycles <= 0:
        return beta_max
    cycle_len = total_epochs / n_cycles
    pos = epoch % cycle_len
    return beta_max * (pos / (cycle_len * ratio_increase)) if pos < cycle_len * ratio_increase else beta_max


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------

def build_loso_cohort(
    metadata_df: pd.DataFrame,
    held_out_site: int,
    site_col: str,
    manufacturer_filter: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For a given held-out site, return:
      test_df     : CN/AD rows from held-out site (filtered manufacturer)
      traindev_df : CN/AD rows from all other sites (filtered manufacturer)
      vae_pool_df : ALL classes (CN/AD/MCI) from all other sites (filtered manufacturer)

    All DataFrames contain: SubjectID, tensor_idx, ResearchGroup_Mapped, Site3, Manufacturer, ...
    """
    df = metadata_df.copy()

    # Manufacturer filter
    if manufacturer_filter:
        df = df[df["Manufacturer"] == manufacturer_filter].copy()
        if df.empty:
            raise ValueError(f"No subjects after filtering to manufacturer={manufacturer_filter}")

    # Site column must exist
    if site_col not in df.columns:
        raise ValueError(f"Site column '{site_col}' not found in metadata")

    mask_held = df[site_col] == held_out_site
    mask_other = ~mask_held

    # Test: CN/AD only from held-out site
    test_df = df[mask_held & df["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    if test_df.empty:
        raise ValueError(
            f"Held-out site {held_out_site}: no CN/AD subjects after filtering to {manufacturer_filter}"
        )
    n_cn = (test_df["ResearchGroup_Mapped"] == "CN").sum()
    n_ad = (test_df["ResearchGroup_Mapped"] == "AD").sum()
    if n_cn == 0 or n_ad == 0:
        raise ValueError(
            f"Held-out site {held_out_site}: needs both CN and AD. "
            f"Got CN={n_cn}, AD={n_ad}"
        )

    # Classifier train/dev: CN/AD from other sites
    traindev_df = df[mask_other & df["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    if traindev_df.empty:
        raise ValueError(f"No CN/AD subjects for training after holding out site {held_out_site}")

    # VAE pool: all classes from other sites
    vae_pool_df = df[mask_other].copy()
    if vae_pool_df.empty:
        raise ValueError(f"No VAE pool subjects after holding out site {held_out_site}")

    return test_df, traindev_df, vae_pool_df


def assert_no_leakage(
    test_tensor_idx: np.ndarray,
    vae_pool_tensor_idx: np.ndarray,
    vae_actual_train_idx_local: np.ndarray,
    vae_internal_val_idx_local: np.ndarray,
    clf_traindev_tensor_idx: np.ndarray,
    site_tag: str,
) -> Dict[str, Any]:
    """
    Assert that no held-out subject appears in any training split.
    Raises AssertionError if any leakage is found.
    Returns a dict suitable for JSON serialization.
    """
    held = set(test_tensor_idx.tolist())
    pool = set(vae_pool_tensor_idx.tolist())
    vae_train = set(vae_pool_tensor_idx[vae_actual_train_idx_local].tolist())
    vae_val   = set(vae_pool_tensor_idx[vae_internal_val_idx_local].tolist()) if len(vae_internal_val_idx_local) > 0 else set()
    clf_train = set(clf_traindev_tensor_idx.tolist())

    checks = {
        "held_in_vae_pool":   len(held & pool),
        "held_in_vae_train":  len(held & vae_train),
        "held_in_vae_val":    len(held & vae_val),
        "held_in_clf_train":  len(held & clf_train),
    }

    for check_name, n_leaks in checks.items():
        if n_leaks > 0:
            raise AssertionError(
                f"LEAKAGE DETECTED [{site_tag}] in {check_name}: "
                f"{n_leaks} subject(s) from held-out site in training split. "
                f"Aborting to protect data integrity."
            )

    return {
        "site": site_tag,
        "n_test": len(held),
        "n_vae_pool": len(pool),
        "n_vae_train": len(vae_train),
        "n_vae_val": len(vae_val),
        "n_clf_train": len(clf_train),
        "leakage_checks": checks,
        "all_passed": True,
    }


# ---------------------------------------------------------------------------
# VAE training (identical logic to run_vae_clf_ad_inference.py)
# ---------------------------------------------------------------------------

def train_vae(
    pool_tensor_norm: np.ndarray,
    actual_train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
    n_channels: int,
    n_rois: int,
    device: torch.device,
    fold_seed_offset: int,
) -> Tuple[ConvolutionalVAE, Dict]:
    """Train β-VAE on pool_tensor_norm[actual_train_idx]. Returns (model, history)."""
    vae = ConvolutionalVAE(
        input_channels=n_channels,
        latent_dim=args.latent_dim,
        image_size=n_rois,
        final_activation=args.vae_final_activation,
        intermediate_fc_dim_config=args.intermediate_fc_dim_vae,
        dropout_rate=args.dropout_rate_vae,
        use_layernorm_fc=args.use_layernorm_vae_fc,
        num_conv_layers_encoder=args.num_conv_layers_encoder,
        decoder_type=args.decoder_type,
    ).to(device)

    optimizer = optim.AdamW(
        vae.parameters(),
        lr=args.lr_vae,
        weight_decay=args.weight_decay_vae,
        amsgrad=True,
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(pool_tensor_norm[actual_train_idx]).float()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=bool(torch.cuda.is_available()),
        drop_last=True,
    )
    val_loader = None
    if len(val_idx) > 0:
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(pool_tensor_norm[val_idx]).float()),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=bool(torch.cuda.is_available()),
        )

    # Scheduler
    scheduler = None
    if val_loader:
        if args.lr_scheduler_type == "cosine_warm":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=args.lr_scheduler_T0, eta_min=args.lr_scheduler_eta_min
            )
        elif args.lr_scheduler_type == "plateau" and args.lr_scheduler_patience_vae > 0:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", patience=args.lr_scheduler_patience_vae, factor=0.1
            )

    history = {
        "train_loss": [], "train_recon": [], "train_kld": [],
        "val_loss": [], "val_recon": [], "val_kld": [], "val_loss_modelsel": [],
        "beta": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_state = None
    scaler = GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(args.epochs_vae):
        vae.train()
        tr_loss = tr_recon = tr_kld = 0.0
        cur_beta = get_cyclical_beta(
            epoch, args.epochs_vae, args.beta_vae, args.cyclical_beta_n_cycles,
            args.cyclical_beta_ratio_increase,
        )

        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                recon, mu, logvar, _ = vae(batch)
                loss, r, k = vae_loss_function(recon, batch, mu, logvar, beta=cur_beta)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler and args.lr_scheduler_type == "cosine_warm":
                scheduler.step(epoch + 0.5)
            tr_loss += loss.item() * batch.size(0)
            tr_recon += r.item() * batch.size(0)
            tr_kld += k.item() * batch.size(0)

        n_tr = len(train_loader.dataset)
        history["train_loss"].append(tr_loss / n_tr)
        history["train_recon"].append(tr_recon / n_tr)
        history["train_kld"].append(tr_kld / n_tr)
        history["beta"].append(cur_beta)

        if val_loader:
            vae.eval()
            vl = vr = vk = 0.0
            with torch.no_grad():
                with autocast(enabled=(device.type == "cuda")):
                    for (vb,) in val_loader:
                        vb = vb.to(device)
                        rv, mv, lv, _ = vae(vb)
                        _, r_v, k_v = vae_loss_function(rv, vb, mv, lv, beta=cur_beta)
                        vl += (r_v.item() + cur_beta * k_v.item()) * vb.size(0)
                        vr += r_v.item() * vb.size(0)
                        vk += k_v.item() * vb.size(0)
            n_v = len(val_loader.dataset)
            avg_vl = vl / n_v
            avg_vr = vr / n_v
            avg_vk = vk / n_v
            avg_vl_bmax = avg_vr + args.beta_vae * avg_vk
            history["val_loss"].append(avg_vl)
            history["val_recon"].append(avg_vr)
            history["val_kld"].append(avg_vk)
            history["val_loss_modelsel"].append(avg_vl_bmax)

            if scheduler and args.lr_scheduler_type == "plateau":
                old_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(avg_vl_bmax)
                if optimizer.param_groups[0]["lr"] < old_lr:
                    epochs_no_improve = 0

            if avg_vl_bmax < best_val_loss and not np.isnan(avg_vl_bmax):
                best_val_loss = avg_vl_bmax
                best_epoch = epoch + 1
                epochs_no_improve = 0
                best_state = copy.deepcopy(vae.state_dict())
            else:
                epochs_no_improve += 1

            if args.early_stopping_patience_vae > 0 and epochs_no_improve >= args.early_stopping_patience_vae:
                logger.info(
                    f"  Early stopping at epoch {epoch+1}. "
                    f"Best val(βmax)={best_val_loss:.4f} (epoch {best_epoch})"
                )
                break
        else:
            for key in ["val_loss", "val_recon", "val_kld", "val_loss_modelsel"]:
                history[key].append(np.nan)
            best_state = copy.deepcopy(vae.state_dict())

        if (epoch + 1) % args.log_interval_epochs_vae == 0 or epoch == args.epochs_vae - 1:
            logger.info(
                f"  E{epoch+1}/{args.epochs_vae} TrL={history['train_loss'][-1]:.3f} β={cur_beta:.3f}"
            )

    if best_state:
        vae.load_state_dict(best_state)

    return vae, history


# ---------------------------------------------------------------------------
# Classifier training (adapted from run_vae_clf_ad_inference.py)
# ---------------------------------------------------------------------------

def train_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
    clf_type: str,
    args: argparse.Namespace,
    fold_output_dir: Path,
    site_tag: str,
    fold_seed_offset: int,
    traindev_stratify_key: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Tune, (optionally) calibrate, and evaluate one classifier on one LOSO fold.
    Returns dict of metrics + saves per-fold artifacts.
    """
    full_pipeline, param_distributions, n_iter_search = get_classifier_and_grid(
        classifier_type=clf_type,
        seed=args.seed,
        balance=args.classifier_use_class_weight,
        use_smote=False,
        tune_sampler_params=False,
        mlp_hidden_layers=args.mlp_classifier_hidden_layers,
        calibrate=False,
    )

    inner_skf = StratifiedKFold(
        n_splits=args.inner_folds, shuffle=True,
        random_state=args.seed + fold_seed_offset + 30,
    )
    _inner_strat = traindev_stratify_key if traindev_stratify_key is not None else pd.Series(y_train)
    if traindev_stratify_key is not None:
        _vc = _inner_strat.value_counts()
        if (_vc < args.inner_folds).any():
            logger.warning(
                f"  [{site_tag}/{clf_type}] classifier_stratify_cols: strata too small "
                f"(min={int(_vc.min())} < {args.inner_folds} folds). Falling back to label-only."
            )
            _inner_strat = pd.Series(y_train)
    inner_splits = list(inner_skf.split(np.zeros(len(y_train)), _inner_strat))

    sampler = optuna.samplers.TPESampler(seed=args.seed + fold_seed_offset)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna_search = OptunaSearchCV(
        estimator=full_pipeline,
        param_distributions=param_distributions,
        study=study,
        cv=inner_splits,
        scoring=args.gridsearch_scoring,
        n_trials=min(n_iter_search, 1000),
        refit=True,
        n_jobs=args.n_jobs_gridsearch,
        timeout=1800,
        random_state=args.seed,
    )
    optuna_search.fit(X_train, y_train)

    n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    logger.info(f"  [{site_tag}/{clf_type}] Optuna COMPLETE={n_complete} PRUNED={n_pruned}")

    # Save Optuna artifacts
    if args.save_fold_artefacts:
        try:
            study.trials_dataframe().to_csv(
                fold_output_dir / f"optuna_trials_{clf_type}.csv", index=False
            )
        except Exception as e:
            logger.warning(f"  Could not save optuna_trials CSV ({clf_type}): {e}")
        try:
            joblib.dump(study, fold_output_dir / f"optuna_study_{clf_type}.joblib")
        except Exception as e:
            logger.warning(f"  Could not save optuna_study joblib ({clf_type}): {e}")
        try:
            bt = study.best_trial
            _safe_json_dump(
                {
                    "site": site_tag, "classifier": clf_type,
                    "best_value": float(bt.value) if bt.value is not None else None,
                    "best_params": dict(bt.params) if bt.params else None,
                    "n_trials": len(study.trials), "n_complete": n_complete, "n_pruned": n_pruned,
                },
                fold_output_dir / f"optuna_best_trial_{clf_type}.json",
            )
        except Exception as e:
            logger.warning(f"  Could not save optuna_best_trial JSON ({clf_type}): {e}")

    raw_model = optuna_search.best_estimator_
    final_model = raw_model
    did_calibrate = False

    if args.save_fold_artefacts:
        joblib.dump(raw_model, fold_output_dir / f"classifier_{clf_type}_raw_pipeline.joblib")

    if args.classifier_calibrate:
        cal_cv = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=args.seed + fold_seed_offset + 123
        )
        cal = CalibratedClassifierCV(estimator=raw_model, method="sigmoid", cv=cal_cv)
        cal.fit(X_train, y_train)
        final_model = cal
        did_calibrate = True
        if args.save_fold_artefacts:
            joblib.dump(final_model, fold_output_dir / f"classifier_{clf_type}_calibrated_pipeline.joblib")

    if args.save_fold_artefacts:
        joblib.dump(final_model, fold_output_dir / f"classifier_{clf_type}_final_pipeline.joblib")

    # Evaluate on test set
    y_score_raw = _get_score_1d(raw_model, X_test)      # score para AUC/PR-AUC
    y_prob_raw = _get_prob_1d(raw_model, X_test)        # probabilidad cruda si existe

    if did_calibrate:
        y_score_cal = _get_prob_1d(final_model, X_test)  # prob calibrada
        y_score_final = y_score_cal
    else:
        y_score_cal = np.full_like(np.asarray(y_score_raw, dtype=float), np.nan)
        y_score_final = y_prob_raw if y_prob_raw is not None else y_score_raw

    y_pred = final_model.predict(X_test)

    # Save per-subject predictions
    df_preds = pd.DataFrame({
        "site": site_tag,
        "SubjectID": test_df["SubjectID"].values,
        "tensor_idx": test_df["tensor_idx"].values,
        "y_true": y_test,
        "y_score_raw": y_score_raw,
        "y_score_cal": y_score_cal,
        "y_score_final": y_score_final,
        "y_pred": y_pred,
        "did_calibrate": did_calibrate,
    })
    df_preds.to_csv(fold_output_dir / f"test_predictions_{clf_type}.csv", index=False)

    # Compute metrics
    auc_raw = roc_auc_score(y_test, y_score_raw)
    pr_auc_raw = average_precision_score(y_test, y_score_raw)
    auc_final = roc_auc_score(y_test, y_score_final)
    pr_auc_final = average_precision_score(y_test, y_score_final)
    brier_raw = _safe_brier(y_test, y_prob_raw)
    brier_final = _safe_brier(y_test, y_score_final)

    metrics = {
        "site": site_tag,
        "classifier_type": clf_type,
        "n_test": len(y_test),
        "n_CN_test": int((y_test == 0).sum()),
        "n_AD_test": int((y_test == 1).sum()),
        "auc_raw": auc_raw,
        "pr_auc_raw": pr_auc_raw,
        "auc_final": auc_final,
        "pr_auc_final": pr_auc_final,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "sensitivity": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y_test, y_pred, pos_label=0, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "brier_raw": brier_raw,
        "brier_final": brier_final,
        "did_calibrate": did_calibrate,
    }
    logger.info(
        f"  [{site_tag}/{clf_type}] AUC_raw={auc_raw:.4f} AUC_final={auc_final:.4f} "
        f"BalAcc={metrics['balanced_accuracy']:.4f}"
    )
    return metrics, df_preds


# ---------------------------------------------------------------------------
# Pooled metrics
# ---------------------------------------------------------------------------

def compute_pooled_metrics(
    all_preds: pd.DataFrame,
    clf_type: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """Compute metrics across all held-out test predictions pooled together."""
    sub = all_preds[all_preds["classifier_type"] == clf_type].copy()
    if sub.empty:
        return {}

    y_true = sub["y_true"].values
    y_score_raw = sub["y_score_raw"].values
    y_score_final = sub["y_score_final"].values
    y_pred = sub["y_pred"].values

    pooled = {
        "classifier_type": clf_type,
        "n_total": len(y_true),
        "n_CN": int((y_true == 0).sum()),
        "n_AD": int((y_true == 1).sum()),
        "pooled_auc_raw": roc_auc_score(y_true, y_score_raw),
        "pooled_pr_auc_raw": average_precision_score(y_true, y_score_raw),
        "pooled_auc_final": roc_auc_score(y_true, y_score_final),
        "pooled_pr_auc_final": average_precision_score(y_true, y_score_final),
        "pooled_accuracy": accuracy_score(y_true, y_pred),
        "pooled_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "pooled_sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "pooled_specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "pooled_brier_raw": _safe_brier(y_true, y_score_raw),
        "pooled_brier_final": _safe_brier(y_true, y_score_final),
    }

    # Calibration curve (final scores)
    frac_pos, mean_pred = calibration_curve(y_true, y_score_final, n_bins=5, strategy="quantile")
    cal_df = pd.DataFrame({"mean_pred_prob": mean_pred, "frac_pos": frac_pos})
    cal_df.to_csv(out_dir / "pooled_calibration_curve.csv", index=False)

    # Calibration plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "s-", color="#DD8452",
            label=f"{clf_type} (Brier={pooled['pooled_brier_final']:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Pooled calibration — LOSO ({clf_type})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pooled_calibration_plot.png", dpi=150)
    plt.close(fig)

    return pooled


def write_pooled_summary_md(
    site_metrics: pd.DataFrame,
    pooled_metrics: List[Dict],
    out_dir: Path,
    loso_mode: str,
    held_out_sites: List[int],
) -> None:
    """Write a human-readable pooled summary markdown file."""
    lines = [
        f"# LOSO Evaluation Summary — BSPC 2026 Revision",
        f"",
        f"**Mode**: {loso_mode}",
        f"**Held-out sites**: {held_out_sites}",
        f"",
        "---",
        "",
        "## Per-site metrics",
        "",
    ]
    for clf in site_metrics["classifier_type"].unique():
        sub = site_metrics[site_metrics["classifier_type"] == clf]
        lines += [
            f"### Classifier: {clf}",
            "",
            "| Site | n_test | n_CN | n_AD | AUC_raw | AUC_final | BalAcc | Sens | Spec | Brier_final |",
            "|------|--------|------|------|---------|-----------|--------|------|------|-------------|",
        ]
        for _, row in sub.iterrows():
            lines.append(
                f"| {row['site']} | {row['n_test']} | {row['n_CN_test']} | {row['n_AD_test']} "
                f"| {row['auc_raw']:.3f} | {row['auc_final']:.3f} "
                f"| {row['balanced_accuracy']:.3f} | {row['sensitivity']:.3f} "
                f"| {row['specificity']:.3f} | {row['brier_final']:.3f} |"
            )
        # Mean ± SD row
        num_cols = ["auc_raw", "auc_final", "balanced_accuracy", "sensitivity", "specificity", "brier_final"]
        means = sub[num_cols].mean()
        stds = sub[num_cols].std(ddof=1)
        n = len(sub)
        lines.append(
            f"| **mean±SD** (n={n}) | — | — | — "
            f"| {means['auc_raw']:.3f}±{stds['auc_raw']:.3f} "
            f"| {means['auc_final']:.3f}±{stds['auc_final']:.3f} "
            f"| {means['balanced_accuracy']:.3f}±{stds['balanced_accuracy']:.3f} "
            f"| {means['sensitivity']:.3f}±{stds['sensitivity']:.3f} "
            f"| {means['specificity']:.3f}±{stds['specificity']:.3f} "
            f"| {means['brier_final']:.3f}±{stds['brier_final']:.3f} |"
        )
        lines.append("")

    lines += ["---", "", "## Pooled metrics (all held-out subjects concatenated)", ""]
    for pm in pooled_metrics:
        lines += [
            f"### Classifier: {pm.get('classifier_type', '?')}",
            "",
            f"- n_total={pm.get('n_total')}, n_CN={pm.get('n_CN')}, n_AD={pm.get('n_AD')}",
            f"- Pooled AUC_raw={pm.get('pooled_auc_raw', float('nan')):.4f} / "
            f"AUC_final={pm.get('pooled_auc_final', float('nan')):.4f}",
            f"- Pooled balanced_acc={pm.get('pooled_balanced_accuracy', float('nan')):.4f}",
            f"- Pooled Brier_final={pm.get('pooled_brier_final', float('nan')):.4f}",
            "",
        ]

    (out_dir / "pooled_roc_pr_summary.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main LOSO loop
# ---------------------------------------------------------------------------

def run_loso(
    global_tensor: np.ndarray,
    metadata_df_full: pd.DataFrame,
    selected_channel_names: List[str],
    selected_channel_indices: List[int],
    held_out_sites: List[int],
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    """Run the full LOSO pipeline over all held-out sites."""
    out_dir.mkdir(parents=True, exist_ok=True)

    n_channels = len(selected_channel_indices)
    n_rois = global_tensor.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Select channels
    current_tensor = global_tensor[:, selected_channel_indices, :, :]

    # We use metadata_df_full for QC lookups (needs all subjects, all manufacturers)
    # but we derive cohorts from the Philips-filtered subset

    all_site_metrics: List[Dict] = []
    all_predictions: List[pd.DataFrame] = []

    for site_num in held_out_sites:
        site_tag = f"site_{site_num:03d}"
        fold_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"  LOSO FOLD: {site_tag}")
        logger.info(f"{'='*60}")

        fold_output_dir = out_dir / site_tag
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        # --- Build cohort ---
        try:
            test_df, traindev_df, vae_pool_df = build_loso_cohort(
                metadata_df=metadata_df_full,
                held_out_site=site_num,
                site_col=args.site_column,
                manufacturer_filter=args.manufacturer_filter,
            )
        except ValueError as e:
            logger.error(f"  [{site_tag}] Cohort construction failed: {e}. Skipping.")
            continue

        # Convert tensor_idx to int
        test_df = test_df.copy(); test_df["tensor_idx"] = test_df["tensor_idx"].astype(int)
        traindev_df = traindev_df.copy(); traindev_df["tensor_idx"] = traindev_df["tensor_idx"].astype(int)
        vae_pool_df = vae_pool_df.copy(); vae_pool_df["tensor_idx"] = vae_pool_df["tensor_idx"].astype(int)

        max_idx = current_tensor.shape[0] - 1
        test_df     = test_df[test_df["tensor_idx"] <= max_idx]
        traindev_df = traindev_df[traindev_df["tensor_idx"] <= max_idx]
        vae_pool_df = vae_pool_df[vae_pool_df["tensor_idx"] <= max_idx]

        test_tensor_idx   = test_df["tensor_idx"].values
        traindev_tensor_idx = traindev_df["tensor_idx"].values
        vae_pool_tensor_idx = vae_pool_df["tensor_idx"].values

        logger.info(
            f"  [{site_tag}] TEST: {len(test_df)} (CN={int((test_df['ResearchGroup_Mapped']=='CN').sum())}, "
            f"AD={int((test_df['ResearchGroup_Mapped']=='AD').sum())})"
        )
        logger.info(
            f"  [{site_tag}] TRAIN/DEV: {len(traindev_df)} CN/AD subjects"
        )
        logger.info(
            f"  [{site_tag}] VAE POOL: {len(vae_pool_df)} subjects (all classes)"
        )

        # Save subject lists (pre-filter; may be overwritten after vae_required_metadata_cols)
        test_df.to_csv(fold_output_dir / "test_subjects.csv", index=False)
        traindev_df.to_csv(fold_output_dir / "train_subjects.csv", index=False)
        vae_pool_df[["SubjectID", "tensor_idx", "ResearchGroup_Mapped"]].to_csv(
            fold_output_dir / "vae_pool_subjects.csv", index=False
        )
        np.save(fold_output_dir / "test_tensor_idx.npy", test_tensor_idx)
        np.save(fold_output_dir / "train_tensor_idx.npy", traindev_tensor_idx)
        np.save(fold_output_dir / "vae_pool_tensor_idx.npy", vae_pool_tensor_idx)

        # --- vae_required_metadata_cols: drop subjects with missing required columns ---
        _req_meta = getattr(args, "vae_required_metadata_cols", None) or []
        _req_meta = [c for c in _req_meta if c]
        if _req_meta:
            _missing_any = pd.Series(False, index=vae_pool_df.index)
            for _col in _req_meta:
                if _col in vae_pool_df.columns:
                    _missing_any |= (
                        vae_pool_df[_col].isna()
                        | vae_pool_df[_col].astype(str).str.strip().isin(["", "nan", "NaN", "None", "none", "NA", "N/A"])
                    )
                else:
                    logger.warning(f"  [{site_tag}] vae_required_metadata_cols: column {_col!r} not found; skipping.")
            if _missing_any.any():
                _n_before = len(vae_pool_df)
                _removed_sids = vae_pool_df.loc[_missing_any, "SubjectID"].tolist() if "SubjectID" in vae_pool_df.columns else []
                vae_pool_df = vae_pool_df[~_missing_any].reset_index(drop=True)
                vae_pool_tensor_idx = vae_pool_df["tensor_idx"].values
                _n_after = len(vae_pool_df)
                logger.warning(
                    f"  [{site_tag}] vae_required_metadata_cols: dropped {_n_before - _n_after} "
                    f"subjects ({_removed_sids}). Pool: {_n_before} → {_n_after}."
                )
                # Overwrite saved pool files with filtered set
                vae_pool_df[["SubjectID", "tensor_idx", "ResearchGroup_Mapped"]].to_csv(
                    fold_output_dir / "vae_pool_subjects.csv", index=False
                )
                np.save(fold_output_dir / "vae_pool_tensor_idx.npy", vae_pool_tensor_idx)
            else:
                logger.info(
                    f"  [{site_tag}] vae_required_metadata_cols: all {len(vae_pool_df)} "
                    "VAE pool subjects have required metadata."
                )

        # --- VAE internal train/val split (cascade: RG+Mfr → RG → unstratified) ---
        pool_n = len(vae_pool_tensor_idx)
        vae_actual_train_idx = np.arange(pool_n, dtype=int)
        vae_val_idx = np.array([], dtype=int)

        if args.vae_val_split_ratio > 0 and pool_n > 10:
            # Build stratification candidates from vae_stratify_cols (default: Manufacturer)
            _extra_strat = getattr(args, "vae_stratify_cols", None) or []
            _strat_candidates = []
            if _extra_strat:
                _strat_candidates.append(["ResearchGroup_Mapped"] + list(_extra_strat))
            _strat_candidates.append(["ResearchGroup_Mapped"])
            _strat_candidates.append([])  # unstratified

            _split_success = False
            _split_errors = []
            for _cols in _strat_candidates:
                try:
                    if _cols:
                        _sk = vae_pool_df[_cols[0]].fillna(f"{_cols[0]}_Unknown").astype(str)
                        for _c in _cols[1:]:
                            _sk = _sk + "_" + vae_pool_df[_c].fillna(f"{_c}_Unknown").astype(str)
                        _vc = _sk.value_counts()
                        if _vc.min() < 2:
                            _split_errors.append(f"{'+'.join(_cols)}: singleton strata (min={int(_vc.min())})")
                            continue
                        _strat_arg = _sk
                    else:
                        _strat_arg = None
                    vae_actual_train_idx, vae_val_idx = sk_train_test_split(
                        np.arange(pool_n),
                        test_size=args.vae_val_split_ratio,
                        stratify=_strat_arg,
                        random_state=args.seed + site_num,
                        shuffle=True,
                    )
                    vae_actual_train_idx = np.asarray(vae_actual_train_idx, dtype=int)
                    vae_val_idx = np.asarray(vae_val_idx, dtype=int)
                    logger.info(
                        f"  [{site_tag}] VAE val split: train={len(vae_actual_train_idx)}, "
                        f"val={len(vae_val_idx)}, strat={_cols or 'none'}"
                    )
                    _split_success = True
                    break
                except ValueError as _e:
                    _split_errors.append(f"{'+'.join(_cols) or 'unstratified'}: {_e}")

            if not _split_success:
                _err_msg = (
                    f"  [{site_tag}] VAE val split failed. Attempts: {' | '.join(_split_errors)}."
                )
                if getattr(args, "vae_abort_if_val_split_fails", False):
                    raise RuntimeError(
                        _err_msg + " --vae_abort_if_val_split_fails is set. "
                        "Use --vae_required_metadata_cols to remove problematic subjects."
                    )
                logger.warning(_err_msg + " Using full pool as train (no early stopping).")
                vae_actual_train_idx = np.arange(pool_n, dtype=int)
                vae_val_idx = np.array([], dtype=int)

        np.save(fold_output_dir / "vae_actual_train_idx_local.npy", vae_actual_train_idx)
        np.save(fold_output_dir / "vae_internal_val_idx_local.npy", vae_val_idx)

        # --- LEAKAGE ASSERTIONS ---
        try:
            assertion_result = assert_no_leakage(
                test_tensor_idx=test_tensor_idx,
                vae_pool_tensor_idx=vae_pool_tensor_idx,
                vae_actual_train_idx_local=vae_actual_train_idx,
                vae_internal_val_idx_local=vae_val_idx,
                clf_traindev_tensor_idx=traindev_tensor_idx,
                site_tag=site_tag,
            )
            _safe_json_dump(assertion_result, fold_output_dir / "leakage_assertions.json")
            logger.info(f"  [{site_tag}] All leakage assertions PASSED.")
        except AssertionError as e:
            logger.error(str(e))
            raise

        # --- Normalize tensor (fit on VAE actual train set, apply to all) ---
        vae_pool_raw = current_tensor[vae_pool_tensor_idx]
        vae_pool_norm, norm_params = normalize_inter_channel_fold(
            vae_pool_raw,
            vae_actual_train_idx,
            mode=args.norm_mode,
            selected_channel_original_names=selected_channel_names,
        )
        joblib.dump(norm_params, fold_output_dir / "vae_norm_params.joblib")

        # --- Train VAE ---
        logger.info(f"  [{site_tag}] Training VAE...")
        vae_model, history = train_vae(
            pool_tensor_norm=vae_pool_norm,
            actual_train_idx=vae_actual_train_idx,
            val_idx=vae_val_idx,
            args=args,
            n_channels=n_channels,
            n_rois=n_rois,
            device=device,
            fold_seed_offset=site_num,
        )
        vae_model_path = fold_output_dir / f"vae_model_{site_tag}.pt"
        torch.save(vae_model.state_dict(), vae_model_path)
        logger.info(f"  [{site_tag}] VAE saved: {vae_model_path}")

        if args.save_vae_training_history:
            joblib.dump(history, fold_output_dir / "vae_train_history.joblib")
            try:
                fig, ax1 = plt.subplots(figsize=(12, 5))
                ax1.plot(history["train_loss"], label="Train Loss", color="blue")
                if any(not np.isnan(v) for v in history["val_loss"]):
                    ax1.plot(history["val_loss_modelsel"], label="Val Loss (βmax)", color="red", linestyle="-.")
                ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
                ax1.set_title(f"{site_tag} VAE Training History")
                ax1.legend(); ax1.grid(True, alpha=0.4)
                ax2 = ax1.twinx()
                ax2.plot(history["beta"], color="green", linestyle="--", alpha=0.7, label="Beta")
                ax2.set_ylabel("Beta", color="green"); ax2.tick_params(axis="y", labelcolor="green")
                ax2.legend(loc="upper right")
                fig.tight_layout()
                plt.savefig(fold_output_dir / "vae_train_history.png")
                plt.close(fig)
            except Exception as e:
                logger.warning(f"  [{site_tag}] Could not save VAE history plot: {e}")

        if args.qc_analyze_distributions:
            try:
                raw_sub = vae_pool_raw[vae_actual_train_idx]
                norm_sub = vae_pool_norm[vae_actual_train_idx]
                vae_model.eval()
                with torch.no_grad():
                    recon_list = []
                    for i in range(0, norm_sub.shape[0], 32):
                        b = torch.from_numpy(norm_sub[i:i+32]).float().to(device)
                        r, _, _, _ = vae_model(b)
                        recon_list.append(r.detach().cpu().numpy())
                recon_sub = np.concatenate(recon_list, axis=0)
                summarize_distribution_stages(
                    raw_tensor=raw_sub, norm_tensor=norm_sub, recon_tensor=recon_sub,
                    channel_names=selected_channel_names, out_dir=fold_output_dir,
                    prefix=site_tag, final_activation=args.vae_final_activation,
                )
            except Exception as e:
                logger.warning(f"  [{site_tag}] QC distributions failed: {e}")

        # --- Encode train/dev and test sets ---
        vae_model.eval()
        with torch.no_grad():
            traindev_norm = apply_normalization_params(
                current_tensor[traindev_tensor_idx], norm_params
            )
            _, mu_train, _, z_train = vae_model(
                torch.from_numpy(traindev_norm).float().to(device)
            )
            mu_train_np = mu_train.detach().cpu().numpy()
            z_train_np  = z_train.detach().cpu().numpy()
            del mu_train, z_train
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            test_norm = apply_normalization_params(
                current_tensor[test_tensor_idx], norm_params
            )
            _, mu_test, _, z_test = vae_model(
                torch.from_numpy(test_norm).float().to(device)
            )
            mu_test_np = mu_test.detach().cpu().numpy()
            z_test_np  = z_test.detach().cpu().numpy()
            del mu_test, z_test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        feat_names = [f"latent_{i}" for i in range(mu_train_np.shape[1])]
        X_train_lat = mu_train_np if args.latent_features_type == "mu" else z_train_np
        X_test_lat  = mu_test_np  if args.latent_features_type == "mu" else z_test_np
        X_train_df  = pd.DataFrame(X_train_lat, columns=feat_names)
        X_test_df   = pd.DataFrame(X_test_lat, columns=feat_names)

        # Append metadata features if requested
        if args.metadata_features:
            for X_df, source_df in [(X_train_df, traindev_df.reset_index(drop=True)),
                                     (X_test_df, test_df.reset_index(drop=True))]:
                for col in args.metadata_features:
                    if col in source_df.columns:
                        X_df[col] = source_df[col].values

        y_train = traindev_df["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1}).values
        y_test  = test_df["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1}).values

        # QC: scanner leakage on train set
        if args.qc_check_scanner_leakage:
            try:
                evaluate_scanner_leakage(
                    metadata_df_full=metadata_df_full,
                    subject_global_indices=traindev_tensor_idx,
                    normalized_tensor_subjects=traindev_norm,
                    latent_mu_subjects=mu_train_np,
                    out_dir=fold_output_dir,
                    fold_tag=f"{site_tag}_train",
                    random_state=args.seed + site_num + 99,
                    vectorize_mode="auto",
                )
            except Exception as e:
                logger.warning(f"  [{site_tag}] QC scanner leakage (train) failed: {e}")

        # QC: silhouette
        silhouette = np.nan
        if len(y_test) > 2:
            silhouette = compute_latent_silhouette(
                latent_feats=X_test_lat, labels_binary=y_test
            )
        qc_row = {
            "site": site_tag,
            "silhouette_latent_test": silhouette,
            "latent_dim": args.latent_dim,
            "beta_max": args.beta_vae,
        }
        pd.DataFrame([qc_row]).to_csv(fold_output_dir / "latent_qc_metrics.csv", index=False)

        del traindev_norm, test_norm
        gc.collect()

        # --- Build classifier inner-CV stratification key ---
        _clf_strat_cols = getattr(args, "classifier_stratify_cols", None) or []
        _clf_strat_key = None
        if _clf_strat_cols:
            _traindev_meta = traindev_df.reset_index(drop=True)
            _base_key = pd.Series(y_train, dtype=str)
            _extra_keys = []
            for _sc in _clf_strat_cols:
                if _sc in _traindev_meta.columns:
                    _extra_keys.append(_traindev_meta[_sc].fillna(f"{_sc}_Unknown").astype(str))
                else:
                    logger.warning(f"  [{site_tag}] classifier_stratify_cols: '{_sc}' not found; skipping.")
            if _extra_keys:
                _clf_strat_key = _base_key.str.cat(_extra_keys, sep="_")
                logger.info(f"  [{site_tag}] Classifier inner-CV stratification: diagnosis + {_clf_strat_cols}")

        # --- Train classifiers ---
        for clf_type in args.classifier_types:
            logger.info(f"  [{site_tag}] Training classifier: {clf_type}")
            try:
                metrics, df_preds = train_classifier(
                    X_train=X_train_df, y_train=y_train,
                    X_test=X_test_df,  y_test=y_test,
                    test_df=test_df,
                    clf_type=clf_type,
                    args=args,
                    fold_output_dir=fold_output_dir,
                    site_tag=site_tag,
                    fold_seed_offset=site_num,
                    traindev_stratify_key=_clf_strat_key,
                )
                df_preds.insert(1, "classifier_type", clf_type)
                all_site_metrics.append(metrics)
                all_predictions.append(df_preds)
            except Exception as e:
                logger.error(f"  [{site_tag}/{clf_type}] Classifier failed: {e}", exc_info=True)

        del vae_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"  [{site_tag}] Completed in {time.time() - fold_start:.1f}s")

    if not all_site_metrics:
        logger.error("No site metrics collected. Check logs above.")
        return

    # --- Save site-level results ---
    site_metrics_df = pd.DataFrame(all_site_metrics)
    site_metrics_df.to_csv(out_dir / "loso_site_metrics.csv", index=False)
    logger.info(f"Saved: {out_dir / 'loso_site_metrics.csv'}")

    all_preds_df = pd.concat(all_predictions, ignore_index=True)
    all_preds_df.to_csv(out_dir / "loso_all_predictions.csv", index=False)
    logger.info(f"Saved: {out_dir / 'loso_all_predictions.csv'}")

    # --- Pooled metrics per classifier ---
    pooled_list = []
    for clf_type in args.classifier_types:
        pm = compute_pooled_metrics(all_preds_df, clf_type, out_dir)
        if pm:
            pooled_list.append(pm)

    _safe_json_dump(pooled_list, out_dir / "pooled_metrics.json")
    logger.info(f"Saved: {out_dir / 'pooled_metrics.json'}")

    write_pooled_summary_md(
        site_metrics=site_metrics_df,
        pooled_metrics=pooled_list,
        out_dir=out_dir,
        loso_mode=args.loso_mode,
        held_out_sites=held_out_sites,
    )
    logger.info(f"Saved: {out_dir / 'pooled_roc_pr_summary.md'}")

    # --- Terminal summary ---
    print()
    print("=" * 65)
    print(f"  LOSO RESULTS ({args.loso_mode.upper()}) — TERMINAL SUMMARY")
    print("=" * 65)
    for clf_type in args.classifier_types:
        sub = site_metrics_df[site_metrics_df["classifier_type"] == clf_type]
        if sub.empty:
            continue
        print(f"\n  Classifier: {clf_type}")
        print(f"  {'Site':<12} {'n_test':>6} {'AUC_raw':>8} {'AUC_fin':>8} {'BalAcc':>8}")
        for _, row in sub.iterrows():
            print(
                f"  {row['site']:<12} {row['n_test']:>6} "
                f"{row['auc_raw']:>8.3f} {row['auc_final']:>8.3f} "
                f"{row['balanced_accuracy']:>8.3f}"
            )
        if len(sub) > 1:
            print(
                f"  {'mean±SD':<12} {'-':>6} "
                f"{sub['auc_raw'].mean():>8.3f}±{sub['auc_raw'].std():.3f} "
                f"{sub['auc_final'].mean():>8.3f}±{sub['auc_final'].std():.3f} "
                f"{sub['balanced_accuracy'].mean():>8.3f}±{sub['balanced_accuracy'].std():.3f}"
            )
        pm = next((p for p in pooled_list if p.get("classifier_type") == clf_type), None)
        if pm:
            print(f"\n  Pooled AUC_final={pm['pooled_auc_final']:.4f} "
                  f"BalAcc={pm['pooled_balanced_accuracy']:.4f} "
                  f"Brier={pm['pooled_brier_final']:.4f}")
    print()
    print(f"  Outputs → {out_dir}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LOSO evaluation for BSPC 2026 revision (β-VAE + classifier)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--global_tensor_path", type=str, required=True)
    p.add_argument("--metadata_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="results/revision_bspc_2026/loso_primary")
    p.add_argument("--channels_to_use", type=int, nargs="*", default=None)

    # LOSO design
    p.add_argument("--site_column", type=str, default="Site3")
    p.add_argument("--manufacturer_filter", type=str, default="Philips",
                   help="Restrict cohort to this manufacturer. Set to '' to disable.")
    p.add_argument("--loso_mode", type=str, default="primary",
                   choices=["primary", "strict", "custom"],
                   help="primary=[6,18,19,130,305] / strict=[6,130] / custom=--loso_sites")
    p.add_argument("--loso_sites", type=int, nargs="+", default=None,
                   help="Explicit list of held-out sites (used when --loso_mode custom)")

    # VAE
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--epochs_vae", type=int, default=2560)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--beta_vae", type=float, default=6.5)
    p.add_argument("--lr_vae", type=float, default=1e-4)
    p.add_argument("--weight_decay_vae", type=float, default=5e-7)
    p.add_argument("--dropout_rate_vae", type=float, default=0.15)
    p.add_argument("--vae_val_split_ratio", type=float, default=0.2)
    p.add_argument("--early_stopping_patience_vae", type=int, default=240)
    p.add_argument("--cyclical_beta_n_cycles", type=int, default=32)
    p.add_argument("--cyclical_beta_ratio_increase", type=float, default=0.5)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine_warm",
                   choices=["plateau", "cosine_warm"])
    p.add_argument("--lr_scheduler_T0", type=int, default=80)
    p.add_argument("--lr_scheduler_eta_min", type=float, default=5e-7)
    p.add_argument("--lr_scheduler_patience_vae", type=int, default=15)
    p.add_argument("--vae_final_activation", type=str, default="tanh",
                   choices=["sigmoid", "tanh", "linear"])
    p.add_argument("--intermediate_fc_dim_vae", type=str, default="quarter")
    p.add_argument("--use_layernorm_vae_fc", action="store_true")
    p.add_argument("--num_conv_layers_encoder", type=int, default=4, choices=[3, 4])
    p.add_argument("--decoder_type", type=str, default="convtranspose",
                   choices=["upsample_conv", "convtranspose"])

    # Classifier
    clf_choices = get_available_classifiers()
    p.add_argument("--classifier_types", nargs="+", default=["logreg"], choices=clf_choices)
    p.add_argument("--classifier_calibrate", action="store_true")
    p.add_argument("--classifier_use_class_weight", action="store_true")
    p.add_argument("--inner_folds", type=int, default=5)
    p.add_argument("--gridsearch_scoring", type=str, default="roc_auc")
    p.add_argument("--latent_features_type", type=str, default="mu", choices=["mu", "z"])
    p.add_argument("--metadata_features", nargs="*", default=None)
    p.add_argument("--mlp_classifier_hidden_layers", type=str, default="64,16")
    p.add_argument("--classifier_stratify_cols", type=str, nargs="*", default=None,
                   help="Additional columns for inner-CV stratification (combined with diagnosis).")
    p.add_argument("--vae_required_metadata_cols", type=str, nargs="*", default=None,
                   help="Columns that must be non-null in the VAE pool. "
                        "Subjects missing any are excluded before the val split.")
    p.add_argument("--vae_abort_if_val_split_fails", action="store_true",
                   help="Abort if VAE internal val split fails instead of silently using full pool.")
    p.add_argument("--vae_stratify_cols", type=str, nargs="*", default=None,
                   help="Additional columns for VAE val split stratification (prepended by ResearchGroup_Mapped).")

    # Misc
    p.add_argument("--norm_mode", type=str, default="zscore_offdiag")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_jobs_gridsearch", type=int, default=8)
    p.add_argument("--log_interval_epochs_vae", type=int, default=50)
    p.add_argument("--save_fold_artefacts", action="store_true")
    p.add_argument("--save_vae_training_history", action="store_true")
    p.add_argument("--qc_analyze_distributions", action="store_true")
    p.add_argument("--qc_check_scanner_leakage", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # Resolve manufacturer_filter
    if args.manufacturer_filter == "":
        args.manufacturer_filter = None

    # Resolve held-out sites
    if args.loso_mode == "primary":
        held_out_sites = LOSO_PRIMARY_SITES
    elif args.loso_mode == "strict":
        held_out_sites = LOSO_STRICT_SITES
    elif args.loso_mode == "custom":
        if not args.loso_sites:
            raise ValueError("--loso_mode custom requires --loso_sites")
        held_out_sites = args.loso_sites
    else:
        raise ValueError(f"Unknown loso_mode: {args.loso_mode}")

    logger.info(f"LOSO mode: {args.loso_mode} | Held-out sites: {held_out_sites}")
    logger.info(f"Manufacturer filter: {args.manufacturer_filter}")

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Git hash
    try:
        args.git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        args.git_hash = "N/A"

    # Load data
    logger.info("Loading data...")
    global_tensor, metadata_df_full, roi_names, network_labels = load_data(
        Path(args.global_tensor_path), Path(args.metadata_path)
    )
    if global_tensor is None or metadata_df_full is None:
        logger.critical("Failed to load data. Aborting.")
        sys.exit(1)

    # Resolve channel names and indices
    n_chan_tensor = global_tensor.shape[1]
    master_channel_names = _extend_channel_names(n_chan_tensor, DEFAULT_CHANNEL_NAMES)
    if args.channels_to_use:
        selected_channel_indices = list(args.channels_to_use)
        try:
            selected_channel_names = [master_channel_names[i] for i in selected_channel_indices]
        except IndexError:
            selected_channel_names = [f"RawChan{i}" for i in selected_channel_indices]
    else:
        selected_channel_indices = list(range(n_chan_tensor))
        selected_channel_names = master_channel_names

    logger.info(f"Channels selected: {selected_channel_names}")

    # Validate tensor
    if roi_names is None or len(roi_names) != global_tensor.shape[2]:
        logger.critical("roi_names_in_order missing or mismatched. Aborting.")
        sys.exit(1)
    if global_tensor.shape[2] != global_tensor.shape[3]:
        logger.critical("Non-square connectivity matrices. Aborting.")
        sys.exit(1)

    # Save run config
    try:
        tensor_path = Path(args.global_tensor_path)
        fingerprint = _compute_file_sha256(tensor_path) if tensor_path.exists() else None
        run_config = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_hash": args.git_hash,
            "python_version": platform.python_version(),
            "torch_version": str(getattr(torch, "__version__", None)),
            "cuda_available": bool(torch.cuda.is_available()),
            "loso_mode": args.loso_mode,
            "held_out_sites": held_out_sites,
            "manufacturer_filter": args.manufacturer_filter,
            "site_column": args.site_column,
            "args": vars(args),
            "DEFAULT_CHANNEL_NAMES": DEFAULT_CHANNEL_NAMES,
            "channel_names_selected": selected_channel_names,
            "channels_to_use_indices": selected_channel_indices,
            "global_tensor_path": str(tensor_path.resolve()),
            "tensor_shape": list(global_tensor.shape),
            "train_tensor_fingerprint_sha256": fingerprint,
            "roi_names_in_order": roi_names,
        }
        _safe_json_dump(run_config, out_dir / "run_config.json")
        logger.info(f"run_config.json saved to {out_dir}")
    except Exception as e:
        logger.warning(f"Could not save run_config.json: {e}")

    # Run LOSO
    t0 = time.time()
    run_loso(
        global_tensor=global_tensor,
        metadata_df_full=metadata_df_full,
        selected_channel_names=selected_channel_names,
        selected_channel_indices=selected_channel_indices,
        held_out_sites=held_out_sites,
        args=args,
        out_dir=out_dir,
    )
    logger.info(f"Total LOSO runtime: {time.time() - t0:.1f}s")
