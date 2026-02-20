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
# --- Fin bootstrap ---

import argparse
import copy
import gc
import subprocess
import time
from typing import Any, Dict, List, Optional
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
from torch.utils.data import DataLoader, TensorDataset

# Plot (si lo usás)
import matplotlib.pyplot as plt

# Proyecto
from betavae_xai.models import ConvolutionalVAE, get_classifier_and_grid, get_available_classifiers
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

# --- logging & warnings: una sola vez ---
logger = setup_logging(__name__)


# --- Constantes y Configuraciones Globales ---
DEFAULT_CHANNEL_NAMES = [
    'Pearson_OMST_GCE_Signed_Weighted', 'Pearson_Full_FisherZ_Signed', 'MI_KNN_Symmetric',
    'dFC_AbsDiffMean', 'dFC_StdDev', 'DistanceCorr', 'Granger_F_lag1' # <<< Lista actualizada para ser completa
]

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

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    recon_x = recon_x.float()
    x       = x.float()
    mu      = mu.float()
    logvar  = logvar.float()

    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.shape[0]
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss.detach(), kld_loss.detach()


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

    all_folds_metrics = []
    all_folds_vae_history = []
    all_folds_clf_predictions = []

    for fold_idx, (train_dev_clf_idx_in_cn_ad_df, test_clf_idx_in_cn_ad_df) in enumerate(outer_cv_clf.split(X_classifier_subject_indices_in_cn_ad_df, y_outer)):
    #for fold_idx, (train_dev_clf_idx_in_cn_ad_df, test_clf_idx_in_cn_ad_df) in enumerate(outer_cv_clf.split(X_classifier_subject_indices_in_cn_ad_df, stratify_key_for_clf_cv)):
        fold_start_time = time.time()
        fold_idx_str = f"Fold {fold_idx + 1}/{total_outer_iterations}"
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
            
        vae_train_pool_tensor_original_scale = current_global_tensor[global_indices_vae_training_pool]
        
        # DEFAULT SEGURO:
        # - si vae_val_split_ratio == 0 -> train = todo el pool, val = vacío
        # - si ratio > 0 -> intentamos split estratificado; si falla -> train = todo
        pool_n = int(len(global_indices_vae_training_pool))
        vae_actual_train_indices_local_to_pool = np.arange(pool_n, dtype=int)
        vae_internal_val_indices_local_to_pool = np.array([], dtype=int)

        # columnas candidatas para balancear train/val del VAE
        vae_strat_candidates = ['ResearchGroup_Mapped', 'Sex', 'Age_Group']
        available_cols = [c for c in vae_strat_candidates if c in vae_train_pool_df.columns]

        if len(available_cols) == 0:
            # fallback duro: al menos ResearchGroup_Mapped debería existir por construcción arriba,
            # pero igual hagamos seguridad
            available_cols = ['ResearchGroup_Mapped']

        temp_vae_strat_df = vae_train_pool_df[available_cols].copy()

        # imputamos NaNs y los casteamos a str
        for col in available_cols:
            temp_vae_strat_df[col] = temp_vae_strat_df[col].fillna(f"{col}_Unknown").astype(str)

        try:
            stratify_key_vae_split = temp_vae_strat_df.apply(lambda x: '_'.join(x.values.astype(str)), axis=1)

            # ¿cada estrato tiene al menos 2 muestras? si no, bajamos a solo ResearchGroup_Mapped
            if not all(stratify_key_vae_split.value_counts() >= 2):
                logger.warning(f"  {fold_idx_str} Estratos muy chicos combinando {available_cols}. Uso solo 'ResearchGroup_Mapped'.")
                stratify_key_vae_split = vae_train_pool_df['ResearchGroup_Mapped'].fillna("RG_Unknown").astype(str)

            logger.info(f"  {fold_idx_str} VAE val split estratificado por columnas: {available_cols}")

        except Exception as e:
            logger.error(f"  {fold_idx_str} Error creando clave estratificación VAE ({e}). Uso solo 'ResearchGroup_Mapped'.")
            stratify_key_vae_split = vae_train_pool_df['ResearchGroup_Mapped'].fillna("RG_Unknown").astype(str)


        if args.vae_val_split_ratio > 0 and len(global_indices_vae_training_pool) > 10:
            try:
                vae_actual_train_indices_local_to_pool, vae_internal_val_indices_local_to_pool = sk_train_test_split(
                    np.arange(len(global_indices_vae_training_pool)),
                    test_size=args.vae_val_split_ratio,
                    stratify=stratify_key_vae_split, # Usamos la nueva clave de estratificación
                    random_state=args.seed + fold_idx + 10, shuffle=True
                )
                vae_actual_train_indices_local_to_pool = np.asarray(vae_actual_train_indices_local_to_pool, dtype=int)
                vae_internal_val_indices_local_to_pool = np.asarray(vae_internal_val_indices_local_to_pool, dtype=int)

            except ValueError as e:
                logger.error(f"  {fold_idx_str} Error al hacer el split de validación del VAE: {e}. Usando todo el pool como train.")
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
        vae_train_dataset = TensorDataset(torch.from_numpy(vae_pool_tensor_norm[vae_actual_train_indices_local_to_pool]).float())
        vae_train_loader = DataLoader(
            vae_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=bool(torch.cuda.is_available())
        )
        vae_internal_val_loader = None
        if len(vae_internal_val_indices_local_to_pool) > 0:
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
            dropout_rate=args.dropout_rate_vae, use_layernorm_fc=args.use_layernorm_vae_fc,
            num_conv_layers_encoder=args.num_conv_layers_encoder, decoder_type=args.decoder_type
        ).to(device)
        
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
        
        # Listas para guardar el historial completo
        history_data = {
            # train_* usan el beta actual de esa época
            "train_loss": [], "train_recon": [], "train_kld": [],
            "val_loss": [], "val_recon": [], "val_kld": [],
            "val_loss_modelsel": [],

            # para debug
            "beta": []
        }

        scaler = GradScaler(enabled=(device.type == 'cuda'))

        for epoch in range(args.epochs_vae):
            vae_fold_k.train()
            # Acumuladores para la época de entrenamiento
            epoch_train_loss, epoch_train_recon, epoch_train_kld = 0.0, 0.0, 0.0 # ⬅️ NUEVO
            current_beta = get_cyclical_beta_schedule(
                current_epoch=epoch,
                total_epochs=args.epochs_vae,
                beta_max=args.beta_vae,
                n_cycles=args.cyclical_beta_n_cycles,
                ratio_increase=args.cyclical_beta_ratio_increase
            )

            for i, (data,) in enumerate(vae_train_loader):
                data = data.to(device)
                optimizer_vae.zero_grad(set_to_none=True)

                with autocast(enabled=(device.type == 'cuda')):
                    recon_batch, mu, logvar, _ = vae_fold_k(data)
                    loss, recon, kld = vae_loss_function(recon_batch, data, mu, logvar, beta=current_beta)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer_vae)
                scaler.update()

                # Actualizar el scheduler de coseno en cada paso
                if scheduler_vae and args.lr_scheduler_type == 'cosine_warm':
                    scheduler_vae.step(epoch + i / len(vae_train_loader)) # Actualización por paso

                epoch_train_loss += loss.item() * data.size(0)
                epoch_train_recon += recon.item() * data.size(0)
                epoch_train_kld += kld.item() * data.size(0)
            # ▲▲▲ FIN BUCLE DE ENTRENAMIENTO MODIFICADO ▲▲▲
            
            # Calculamos las medias y las guardamos en el historial
            history_data["train_loss"].append(epoch_train_loss / len(vae_train_loader.dataset))
            history_data["train_recon"].append(epoch_train_recon / len(vae_train_loader.dataset))
            history_data["train_kld"].append(epoch_train_kld / len(vae_train_loader.dataset))
            history_data["beta"].append(current_beta)
            
            log_msg = (f"  {fold_idx_str} FOLD{fold_idx+1}: E{epoch+1}/{args.epochs_vae}, "
                       f"TrL(curβ): {history_data['train_loss'][-1]:.2f} "
                       f"(R: {history_data['train_recon'][-1]:.2f}, "
                       f"KLD: {history_data['train_kld'][-1]:.2f}), "
                       f"β={current_beta:.3f}, LR={optimizer_vae.param_groups[0]['lr']:.2e}")


            if vae_internal_val_loader:
                vae_fold_k.eval()
                # Acumuladores para la época de validación
                epoch_val_loss_curBeta, epoch_val_recon, epoch_val_kld = 0.0, 0.0, 0.0
                with torch.no_grad():
                    with autocast(enabled=(device.type == 'cuda')):
                        for (val_data,) in vae_internal_val_loader:
                            val_data = val_data.to(device)
                            recon_val, mu_val, logvar_val, _ = vae_fold_k(val_data)
                            # forward con el beta actual (lo que realmente entrenamos esta época)
                            v_loss_curBeta, v_recon, v_kld = vae_loss_function(
                                recon_val, val_data, mu_val, logvar_val, beta=current_beta
                            )

                            epoch_val_loss_curBeta += v_loss_curBeta.item() * val_data.size(0)
                            epoch_val_recon       += v_recon.item()       * val_data.size(0)
                            epoch_val_kld += v_kld.item() * val_data.size(0)

                # --- MÉTRICAS DE VALIDACIÓN ---
                N_val = len(vae_internal_val_loader.dataset)

                # promedio usando el beta ACTUAL de esta época (solo para log)
                avg_val_loss_curBeta = epoch_val_loss_curBeta / N_val

               # promedios puros de los componentes
                avg_val_recon = epoch_val_recon / N_val
                avg_val_kld   = epoch_val_kld   / N_val

                # métrica CONSISTENTE entre épocas:
                # simulamos "cómo le iría" al modelo si usáramos siempre beta_max (= args.beta_vae)
                avg_val_loss_betaMax = avg_val_recon + args.beta_vae * avg_val_kld

                # guardamos todo en history
                history_data["val_loss"].append(avg_val_loss_curBeta)
                history_data["val_recon"].append(avg_val_recon)
                history_data["val_kld"].append(avg_val_kld)
                history_data["val_loss_modelsel"].append(avg_val_loss_betaMax)

                log_msg += (
                    f", ValL(curβ): {avg_val_loss_curBeta:.2f} "
                    f"(R: {avg_val_recon:.2f}, KLD: {avg_val_kld:.2f}) "
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
                    break
            else: 
                # Sin validación -> rellenamos NaN para mantener forma
                for key in ["val_loss", "val_recon", "val_kld", "val_loss_modelsel"]:
                    history_data[key].append(np.nan)
                best_model_state_dict = copy.deepcopy(vae_fold_k.state_dict())

            if (epoch + 1) % args.log_interval_epochs_vae == 0 or epoch == args.epochs_vae - 1:
                logger.info(log_msg)
        
        
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
                        recon_batch, _, _, _ = vae_fold_k(batch)
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


        vae_fold_k.eval()
        with torch.no_grad():
            full_train_dev_tensor_norm = apply_normalization_params(
                current_global_tensor[global_indices_clf_train_dev_all],
                norm_params_fold_list
            )
            _, mu_train_dev, _, z_train_dev = vae_fold_k(
                torch.from_numpy(full_train_dev_tensor_norm).float().to(device)
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


            X_test_final_tensor_norm = apply_normalization_params(
                current_global_tensor[global_indices_clf_test_this_fold],
                norm_params_fold_list
            )
            mu_test_final_np, z_test_final_np, X_np_test = None, None, None
            if X_test_final_tensor_norm is not None and X_test_final_tensor_norm.shape[0] > 0:

                _, mu_test_final, _, z_test_final = vae_fold_k(
                    torch.from_numpy(X_test_final_tensor_norm).float().to(device)
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
            effective_n_trials = min(n_iter_search, max_trials)

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
    group_cv.add_argument("--classifier_stratify_cols", type=str, nargs='*', default=['Sex'], help="Columnas adicionales para estratificación del clasificador.")
    #group_cv.add_argument("--classifier_hp_tune_ratio", type=float, default=0.25, help="Proporción de datos de train/dev para ajuste de HP.")
    group_vae = parser.add_argument_group('VAE Model and Training')
    group_vae.add_argument("--num_conv_layers_encoder", type=int, default=4, choices=[3, 4], help="Capas convolucionales en encoder VAE.") 
    group_vae.add_argument("--decoder_type", type=str, default="convtranspose", choices=["upsample_conv", "convtranspose"], help="Tipo de decoder para VAE.") 
    group_vae.add_argument("--latent_dim", type=int, default=128, help="Dimensión del espacio latente VAE. (Recomendado: 128-256)")
    group_vae.add_argument("--lr_vae", type=float, default=1e-4, help="Tasa de aprendizaje VAE.")
    group_vae.add_argument("--epochs_vae", type=int, default=800, help="Épocas máximas para VAE.")
    group_vae.add_argument("--batch_size", type=int, default=32, help="Tamaño del batch.")
    group_vae.add_argument("--beta_vae", type=float, default=1.0, help="Peso KLD (beta_max para annealing).")
    group_vae.add_argument("--cyclical_beta_n_cycles", type=int, default=4, help="Ciclos para annealing de beta.")
    group_vae.add_argument("--cyclical_beta_ratio_increase", type=float, default=0.4, help="Proporción de ciclo para aumentar beta. (Recomendado: 0.4)")
    group_vae.add_argument("--weight_decay_vae", type=float, default=1e-5, help="Decaimiento de peso (L2 reg) para VAE.")
    group_vae.add_argument("--vae_final_activation", type=str, default="tanh", choices=["sigmoid", "tanh", "linear"], help="Activación final del decoder VAE.")
    group_vae.add_argument("--intermediate_fc_dim_vae", type=str, default="quarter", help="Dimensión FC intermedia en VAE ('0', 'half', 'quarter', o entero).")
    group_vae.add_argument("--dropout_rate_vae", type=float, default=0.2, help="Tasa de dropout en VAE.")
    group_vae.add_argument("--use_layernorm_vae_fc", action='store_true', help="Usar LayerNorm en capas FC del VAE.")
    group_vae.add_argument("--vae_val_split_ratio", type=float, default=0.2, help="Proporción para validación VAE.")
    group_vae.add_argument("--early_stopping_patience_vae", type=int, default=20, help="Paciencia early stopping VAE. (Recomendado: 15-20)")
    group_vae.add_argument("--lr_scheduler_patience_vae", type=int, default=15, help="Paciencia para el scheduler ReduceLROnPlateau del VAE.")
    # ▼▼▼ NUEVOS ARGUMENTOS ▼▼▼
    group_vae.add_argument("--lr_scheduler_type", type=str, default="plateau", choices=["plateau", "cosine_warm"], help="Tipo de scheduler para el VAE.")
    group_vae.add_argument("--lr_scheduler_T0", type=int, default=50, help="Épocas para el primer reinicio en CosineAnnealingWarmRestarts.")
    group_vae.add_argument("--lr_scheduler_eta_min", type=float, default=1e-7, help="Tasa de aprendizaje mínima para CosineAnnealingWarmRestarts.")
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

    args = parser.parse_args()


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
    logger.info(f"Normalización: '{args.norm_mode}'. Activación VAE: '{args.vae_final_activation}'. Asegurar compatibilidad.")

    if args.qc_analyze_distributions:
        logger.info("QC distribuciones ACTIVADO: Se guardaron CSV e histogramas por fold para raw/norm/recon.")
    if args.qc_check_scanner_leakage:
        logger.info("QC leakage ACTIVADO: Se guardaron métricas de separabilidad de sitio/escáner por fold.")