#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interpret_fold.py

Unified interpretability pipeline (SHAP + saliency/IG)
for one fold of a VAE+classifier experiment.
"""
from __future__ import annotations

import argparse
from typing import Optional
from sklearn.preprocessing import FunctionTransformer
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import joblib
import matplotlib
matplotlib.use("Agg")  # backend no interactivo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
from typing import Optional
import os
import random
from sklearn.preprocessing import FunctionTransformer

try:
    from captum.attr import IntegratedGradients
except ImportError:
    print("WARN: 'captum' no encontrado. 'integrated_gradients' no disponible. Instala: pip install captum")
    IntegratedGradients = None

# ---------------------------------------------------------------------------
# Imports de tu repo local
# ---------------------------------------------------------------------------
try:
    from betavae_xai.models.convolutional_vae import ConvolutionalVAE
except ImportError as e:  # ayuda si se ejecuta fuera del repo raíz
    raise ImportError("No se pudo importar ConvolutionalVAE desde models.convolutional_vae2.\n"
                      "Asegúrate de ejecutar desde la raíz del proyecto o de que PYTHONPATH esté configurado.") from e

from betavae_xai.interpretability.composite_edge_shap import (
    make_edge_index,
    make_edge_index_offdiag,
    make_edge_feature_names,
    make_edge_mapping_df,
    vectorize_tensor_to_edges,
    reconstruct_tensor_from_edges,
    select_top_edges,
    select_top_edges_per_channel,
    compute_train_edge_median,
    compute_frozen_meta_values,
    make_edge_predict_fn,
    validate_edge_roundtrip,
    extract_logreg_latent_weights,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger('interpret')
logging.getLogger('shap').setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Semillas
# ---------------------------------------------------------------------------
def _set_all_seeds(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    random.seed(seed); np.random.seed(seed)
 
# ---------------------------------------------------------------------------
# Utilidades básicas
# ---------------------------------------------------------------------------
import re

def _is_generic_x_names(names: Sequence[str]) -> bool:
    """True si los nombres son del tipo x0, x1, ... (sin semántica)."""
    if len(names) == 0:
        return False
    pat = re.compile(r'^x\d+$')
    return all(isinstance(n, str) and pat.match(n) for n in names)

def _safe_feature_names_after_preproc(
    preproc: Any,
    raw_feature_names: Sequence[str],
    selector: Optional[Any] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Devuelve (feat_names_after, support_mask_optional) de forma robusta:
    - Intenta usar get_feature_names_out con raw_feature_names.
    - Si el resultado parece genérico (x0, x1, ...), hace fallback a raw_feature_names.
    - Aplica el selector (si existe) tanto a los datos como a los nombres.
    """
    raw_feature_names = np.array(list(map(str, raw_feature_names)))
    feat_names_after = None
    try:
        # Muchos transformadores aceptan input_features opcionalmente
        if hasattr(preproc, "get_feature_names_out"):
            try:
                feat_names_after = preproc.get_feature_names_out(raw_feature_names)
            except TypeError:
                feat_names_after = preproc.get_feature_names_out()
    except Exception:
        feat_names_after = None

    if feat_names_after is None or _is_generic_x_names(feat_names_after):
        # Fallback seguro: preservamos el orden original
        feat_names_after = raw_feature_names

    feat_names_after = np.array(list(map(str, feat_names_after)))

    support = None
    if selector is not None and hasattr(selector, "get_support"):
        support = selector.get_support()
        if support is not None and support.dtype == bool and support.shape[0] == feat_names_after.shape[0]:
            feat_names_after = feat_names_after[support]
        else:
            # Si no coincide, mejor no aplicar para no desalinear
            support = None
    return feat_names_after, support

def _build_background_from_train(
    *, fold_dir: Path, cnad_df: pd.DataFrame, tensor_all: np.ndarray, channels: Sequence[int],
    norm_params: List[Dict[str, float]], meta_cols: List[str], vae: ConvolutionalVAE,
    device: torch.device, preproc: Any, selector: Optional[Any], feat_names_target: Sequence[str],
    sample_size: int = 100, seed: int = 42
) -> pd.DataFrame:
    """Genera un background de SHAP crudo (antes de procesar) desde los sujetos de entrenamiento del fold."""
    # 1) Obtener los índices de entrenamiento del clasificador
    test_idx_path = fold_dir / 'test_indices.npy'
    if not test_idx_path.exists():
        raise FileNotFoundError(f"No se encontró {test_idx_path} para reconstruir el background.")
    test_idx_in_cnad = np.load(test_idx_path)

    # El conjunto de entrenamiento son todos los sujetos CN/AD que no están en test
    all_cnad_indices = np.arange(len(cnad_df))
    train_idx_in_cnad = np.setdiff1d(all_cnad_indices, test_idx_in_cnad, assume_unique=True)

    if train_idx_in_cnad.size == 0:
        raise RuntimeError("No hay sujetos de entrenamiento para construir el background.")

    # 2) Tomar una muestra representativa (100-150 es un buen número para SHAP)
    rng = np.random.RandomState(seed)
    if train_idx_in_cnad.size > sample_size:
        train_idx_in_cnad_sample = np.sort(rng.choice(train_idx_in_cnad, size=sample_size, replace=False))
    else:
        train_idx_in_cnad_sample = train_idx_in_cnad
    
    train_df_sample = cnad_df.iloc[train_idx_in_cnad_sample]
    gidx_train_sample = train_df_sample['tensor_idx'].values

    # 3) Reconstruir las features crudas para esta muestra (Tensor -> Latentes -> +Metadatos)
    # Cargar y normalizar tensores
    tens_train_sample = tensor_all[gidx_train_sample][:, channels, :, :]
    tens_train_sample = apply_normalization_params(tens_train_sample, norm_params)
    tens_train_sample_t = torch.from_numpy(tens_train_sample).float().to(device)

    # Obtener latentes del VAE
    with torch.no_grad():
        _, mu, _, z = vae(tens_train_sample_t)
    lat_np = mu.detach().cpu().numpy()
    lat_cols = [f'latent_{i}' for i in range(lat_np.shape[1])]
    X_lat_train_sample = pd.DataFrame(lat_np, columns=lat_cols)
    
    # Replicar el mapeo de 'Sex' a numérico
    if 'Sex' in meta_cols:
        train_df_sample = train_df_sample.copy()
        train_df_sample.loc[:, 'Sex'] = train_df_sample['Sex'].map({'M': 0, 'F': 1, 'f': 1, 'm': 0})

    # Combinar latentes y metadatos
    X_bg_raw = pd.concat([X_lat_train_sample.reset_index(drop=True),
                          train_df_sample[meta_cols].reset_index(drop=True)],
                         axis=1)

    return X_bg_raw

def _pick_bg_indices(cnad_df: pd.DataFrame,
                     train_idx_in_cnad: np.ndarray,
                     mode: str, sample_size: int, seed: int) -> np.ndarray:
    if mode == 'train':
        pool = np.asarray(train_idx_in_cnad)
    elif mode == 'global':
        pool = np.arange(len(cnad_df))
    elif mode == 'global_cn':
        pool = cnad_df.index[cnad_df['ResearchGroup_Mapped'].astype(str).isin(['CN'])].to_numpy()
    else:
        raise ValueError(f"bg_mode desconocido: {mode}")
    rng = np.random.RandomState(seed)
    if pool.size > sample_size:
        return np.sort(rng.choice(pool, size=sample_size, replace=False))
    return np.sort(pool)

def _build_background_from_indices(
    *, idx_in_cnad: np.ndarray, cnad_df: pd.DataFrame, tensor_all: np.ndarray, channels: Sequence[int],
    norm_params: List[Dict[str, float]], meta_cols: List[str], vae: ConvolutionalVAE,
    device: torch.device
) -> pd.DataFrame:
    """Reconstruye background RAW (latentes + metadatos) usando índices absolutos del DF CN/AD."""
    if idx_in_cnad.size == 0:
        raise RuntimeError("Índices para background vacíos.")
    df = cnad_df.iloc[idx_in_cnad].copy()
    gidx = df['tensor_idx'].values
    tens = tensor_all[gidx][:, channels, :, :]
    tens = apply_normalization_params(tens, norm_params)
    with torch.no_grad():
        _, mu, _, z = vae(torch.from_numpy(tens).float().to(device))
    lat_np = mu.detach().cpu().numpy()
    lat_cols = [f'latent_{i}' for i in range(lat_np.shape[1])]
    X_lat = pd.DataFrame(lat_np, columns=lat_cols)
    if 'Sex' in meta_cols:
        df.loc[:, 'Sex'] = df['Sex'].map({'M':0,'F':1,'m':0,'f':1})
    return pd.concat([X_lat.reset_index(drop=True),
                      df[meta_cols].reset_index(drop=True)], axis=1)

def clean_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Elimina el prefijo "_orig_mod." que añade torch.compile."""
    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}


def build_vae(vae_kwargs: Dict[str, Any], state_dict_path: Path, device: torch.device) -> ConvolutionalVAE:
    """Construye e inicializa un VAE con pesos entrenados."""
    vae = ConvolutionalVAE(**vae_kwargs).to(device)
    sd = torch.load(state_dict_path, map_location=device)
    vae.load_state_dict(clean_state_dict(sd))
    vae.eval()
    return vae

def _project_to_H(sal_map: np.ndarray) -> np.ndarray:
    """Proyecta un mapa de saliencia al subespacio de matrices huecas y simétricas."""
    # sal_map: (C, R, R)
    sal_sym = 0.5 * (sal_map + sal_map.transpose(0, 2, 1))
    for c in range(sal_sym.shape[0]):
        np.fill_diagonal(sal_sym[c], 0.0)
    return sal_sym.astype(np.float32)


def unwrap_model_for_shap(model: Any, clf_type: str) -> Any:
    """Extrae el estimador base de un CalibratedClassifierCV cuando aplica (para cualquier clf)."""
    if hasattr(model, 'calibrated_classifiers_'):
        cc = model.calibrated_classifiers_[0]
        if hasattr(cc, 'estimator') and cc.estimator is not None:
            return cc.estimator
        if hasattr(cc, 'base_estimator') and cc.base_estimator is not None:
            return cc.base_estimator
    return model

def _grad_to_signed_and_abs(grad_batch: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recibe gradientes por-batch sobre la entrada (B,C,R,R) y devuelve:
      - signed: media (con signo) sobre el batch, proyectada a simétrica y hueca.
      - abs:    |signed|, también proyectada.
    """
    g = grad_batch.detach().mean(dim=0).cpu().numpy()  # (C,R,R)
    signed = _project_to_H(g)
    absmap = _project_to_H(np.abs(g))
    return signed.astype(np.float32), absmap.astype(np.float32)



def _to_sample_feature(sh_vals: Union[np.ndarray, List[np.ndarray]],
                       positive_idx: int,
                       n_samples: int,
                       n_features: int) -> np.ndarray:
    """Devuelve siempre array 2D (samples, features) para la clase positiva."""
    # TreeExplainer binario → array (n_samples, n_features)
    if isinstance(sh_vals, np.ndarray) and sh_vals.ndim == 2 and sh_vals.shape == (n_samples, n_features):
        return sh_vals
    # Tree/Kernal multiclase → list de arrays
    if isinstance(sh_vals, list):
        return sh_vals[positive_idx]
    # Formato 3D → (samples, features, classes)
    if isinstance(sh_vals, np.ndarray) and sh_vals.ndim == 3:
        return sh_vals[:, :, positive_idx]
    raise ValueError(f"Formato SHAP inesperado: type={type(sh_vals)} shape={getattr(sh_vals,'shape',None)}")


# ---------------------------------------------------------------------------
# Normalización (copiado de serentipia13.py para independencia)
# ---------------------------------------------------------------------------

def apply_normalization_params(data_tensor_subset: np.ndarray,
                               norm_params_per_channel_list: List[Dict[str, float]]) -> np.ndarray:
    """Aplica parámetros de normalización guardados por canal (off-diag)."""
    num_subjects, num_selected_channels, num_rois, _ = data_tensor_subset.shape
    normalized_tensor_subset = data_tensor_subset.copy()
    off_diag_mask = ~np.eye(num_rois, dtype=bool)
    if len(norm_params_per_channel_list) != num_selected_channels:
        raise ValueError("# canales en datos != # canales en parámetros de normalización")
    for c_idx, params in enumerate(norm_params_per_channel_list):
        mode = params.get('mode', 'zscore_offdiag')
        if params.get('no_scale', False):
            continue
        current_channel_data = data_tensor_subset[:, c_idx, :, :]
        scaled_channel_data = current_channel_data.copy()
        if off_diag_mask.any():
            if mode == 'zscore_offdiag':
                std = params.get('std', 1.0)
                mean = params.get('mean', 0.0)
                if std > 1e-9:
                    scaled_channel_data[:, off_diag_mask] = (current_channel_data[:, off_diag_mask] - mean) / std
            elif mode == 'minmax_offdiag':
                mn = params.get('min', 0.0)
                mx = params.get('max', 1.0)
                rng = mx - mn
                if rng > 1e-9:
                    scaled_channel_data[:, off_diag_mask] = (current_channel_data[:, off_diag_mask] - mn) / rng
                else:
                    scaled_channel_data[:, off_diag_mask] = 0.0
        normalized_tensor_subset[:, c_idx, :, :] = scaled_channel_data
    return normalized_tensor_subset


# ---------------------------------------------------------------------------
# Carga de datos / artefactos del fold
# ---------------------------------------------------------------------------

def _load_global_and_merge(global_tensor_path: Path, metadata_path: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Carga tensor global (.npz) + metadata CSV y los une en un DF con tensor_idx."""
    npz = np.load(global_tensor_path)
    tensor_all = npz['global_tensor_data']
    subj_all = npz['subject_ids'].astype(str)
    meta = pd.read_csv(metadata_path)
    meta['SubjectID'] = meta['SubjectID'].astype(str).str.strip()
    tensor_df = pd.DataFrame({'SubjectID': subj_all, 'tensor_idx': np.arange(len(subj_all))})
    merged = tensor_df.merge(meta, on='SubjectID', how='left')
    return tensor_all, merged


def _subset_cnad(merged_df: pd.DataFrame) -> pd.DataFrame:
    return merged_df[merged_df['ResearchGroup_Mapped'].isin(['CN', 'AD'])].reset_index(drop=True)


def _load_label_info(fold_dir: Path) -> Dict[str, Any]:
    p = fold_dir / 'label_mapping.json'
    if p.exists():
        with open(p) as f:
            return json.load(f)
    # fallback: asumir AD=1 CN=0
    log.warning("label_mapping.json no encontrado; se asume CN=0 / AD=1")
    return {'label_mapping': {'CN': 0, 'AD': 1}, 'positive_label_name': 'AD', 'positive_label_int': 1}


from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline

def _is_pipeline(obj: Any) -> bool:
    return isinstance(obj, (SkPipeline, ImbPipeline)) or hasattr(obj, "steps")

def _calibrator_inner_estimator(obj: Any) -> Any:
    if hasattr(obj, "calibrated_classifiers_"):
        cc = obj.calibrated_classifiers_[0]
        return getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
    return None

# ---------------------------------------------------------------------------
# SHAP (subcomando "shap")
# ---------------------------------------------------------------------------
def cmd_shap(args: argparse.Namespace) -> None:
    fold_dir = Path(args.run_dir) / f"fold_{args.fold}"
    out_dir = fold_dir / 'interpretability_shap'
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.shap_tag}" if getattr(args, 'shap_tag', None) else ""

    log.info(f"[SHAP] fold={args.fold} clf={args.clf}")

    # 1) Pipeline del clasificador (entrenado)
    pipe_path = fold_dir / f"classifier_{args.clf}_pipeline_fold_{args.fold}.joblib"
    if not pipe_path.exists():
        raise FileNotFoundError(f"No se encontró el pipeline del clasificador: {pipe_path}")
    pipe = joblib.load(pipe_path)

    # ------------------------------------------------------------
    # Soporte robusto: a veces el artefacto guardado NO es Pipeline,
    # sino CalibratedClassifierCV (por --classifier_calibrate).
    # En ese caso no existe .named_steps, y lo más seguro es explicar
    # en espacio RAW usando el predictor completo como black-box.
    # ------------------------------------------------------------
    is_pipeline = hasattr(pipe, "named_steps")
    if not is_pipeline:
        log.warning(f"[SHAP] Artefacto cargado no es Pipeline (type={type(pipe).__name__}). "
                    "Se explicará en espacio RAW (latentes+metadatos) con SHAP permutation.")

    model_step = pipe.named_steps["model"] if is_pipeline and "model" in pipe.named_steps else pipe


    # 2) Artefactos y datos que usaremos tanto para TEST como para background
    norm_params = joblib.load(fold_dir / 'vae_norm_params.joblib')
    label_info = _load_label_info(fold_dir)

    # ROI order no es estrictamente necesario para SHAP, pero lo conservamos por compatibilidad
    roi_order_path_joblib = Path(args.run_dir) / 'roi_order_131.joblib'
    if roi_order_path_joblib.exists():
        roi_names = joblib.load(roi_order_path_joblib)
    elif args.roi_order_path is not None:
        roi_names = _load_roi_names(Path(args.roi_order_path))
    else:
        roi_names = None  # OK para SHAP

    # Datos globales + merge con metadata
    tensor_all, merged = _load_global_and_merge(Path(args.global_tensor_path), Path(args.metadata_path))
    cnad = _subset_cnad(merged)

    # Índices de test en el DataFrame CN/AD (orden como en entrenamiento)
    test_idx_in_cnad = np.load(fold_dir / 'test_indices.npy')
    # Derivar índices de TRAIN para robustecer imputaciones de metadatos (sin fuga)
    all_cnad_idx = np.arange(len(cnad))
    train_idx_in_cnad = np.setdiff1d(all_cnad_idx, test_idx_in_cnad, assume_unique=True)
    test_df = cnad.iloc[test_idx_in_cnad].copy()
    gidx_test = test_df['tensor_idx'].values

    # 3) Tensores test normalizados
    tens_test = tensor_all[gidx_test][:, args.channels_to_use, :, :]
    tens_test = apply_normalization_params(tens_test, norm_params)
    tens_test_t = torch.from_numpy(tens_test).float()

    # 4) VAE y latentes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae_kwargs = _vae_kwargs_from_args(args, image_size=tensor_all.shape[-1])
    vae = build_vae(vae_kwargs, fold_dir / f"vae_model_fold_{args.fold}.pt", device)
    with torch.no_grad():
        recon, mu, logvar, z = vae(tens_test_t.to(device))
    lat_np = mu.cpu().numpy() if args.latent_features_type == 'mu' else z.cpu().numpy()
    lat_cols = [f'latent_{i}' for i in range(lat_np.shape[1])]
    X_lat = pd.DataFrame(lat_np, columns=lat_cols)

    # Combinar latentes + metadatos para tener X_raw (features crudas)
    meta_cols = args.metadata_features or []

    # --- Robustez metadatos ---
    test_df = test_df.copy()
    if 'Sex' in meta_cols:
        # map solo si es texto; si ya viene numérica, respetar
        if test_df['Sex'].dtype == object:
            test_df.loc[:, 'Sex'] = test_df['Sex'].map({'M': 0, 'F': 1, 'f': 1, 'm': 0})
        test_df.loc[:, 'Sex'] = pd.to_numeric(test_df['Sex'], errors='coerce')
        # imputación con la moda de TRAIN (0/1). Si no hay, caer al 0.
        sex_train = pd.to_numeric(cnad.iloc[train_idx_in_cnad]['Sex'], errors='coerce')
        sex_mode = sex_train.dropna().mode().iloc[0] if not sex_train.dropna().empty else 0.0
        test_df.loc[:, 'Sex'] = test_df['Sex'].fillna(sex_mode).astype(float)
    if 'Age' in meta_cols:
        test_df.loc[:, 'Age'] = pd.to_numeric(test_df['Age'], errors='coerce')
        age_train = pd.to_numeric(cnad.iloc[train_idx_in_cnad]['Age'], errors='coerce')
        if not age_train.dropna().empty:
            age_mean = float(age_train.dropna().mean())
        else:
            # fallback a la media del propio test_df si existe, si no usar un valor sensato
            if not test_df['Age'].dropna().empty:
                age_mean = float(test_df['Age'].dropna().mean())
            else:
                age_mean = 70.0
        test_df.loc[:, 'Age'] = test_df['Age'].fillna(age_mean).astype(float)

    # --- [NUEVO] Logica para "congelar" metadatos (Drop-in 1) ---
    if args.freeze_meta:
        test_df_freeze = test_df.copy()
        freeze_values = {}
        if args.freeze_strategy == 'train_stats':
            if 'Age' in args.freeze_meta and 'Age' in test_df_freeze.columns:
                age_train = pd.to_numeric(cnad.iloc[train_idx_in_cnad]['Age'], errors='coerce')
                age_med = float(age_train.dropna().median()) if not age_train.dropna().empty else float(test_df_freeze['Age'].dropna().median() if not test_df_freeze['Age'].dropna().empty else 70.0)
                test_df_freeze.loc[:, 'Age'] = age_med
                freeze_values['Age'] = age_med
            if 'Sex' in args.freeze_meta and 'Sex' in test_df_freeze.columns:
                # Re-usar la logica de conversion de string a numerico si es necesario
                sex_train_series = cnad.iloc[train_idx_in_cnad]['Sex']
                # map a {M:0, F:1} y numérico
                sex_train = (pd.to_numeric(
                    sex_train_series.map({'M':0,'F':1,'m':0,'f':1}) if sex_train_series.dtype==object else sex_train_series,
                    errors='coerce'))
                sex_mean = float(sex_train.dropna().mean()) if not sex_train.dropna().empty else 0.5
                test_df_freeze.loc[:, 'Sex'] = sex_mean
                freeze_values['Sex'] = sex_mean
        else:  # constants
            if 'Age' in args.freeze_meta and 'Age' in test_df_freeze.columns:
                test_df_freeze.loc[:, 'Age'] = 70.0
                freeze_values['Age'] = 70.0
            if 'Sex' in args.freeze_meta and 'Sex' in test_df_freeze.columns:
                test_df_freeze.loc[:, 'Sex'] = 0.0
                freeze_values['Sex'] = 0.0
        

        X_raw = pd.concat([X_lat.reset_index(drop=True),
                            test_df_freeze[meta_cols].reset_index(drop=True)], axis=1)
        log.info(f"[SHAP] META congelado para {args.freeze_meta} ({args.freeze_strategy}).")
    else:
        X_raw = pd.concat([X_lat.reset_index(drop=True),
                            test_df[meta_cols].reset_index(drop=True)], axis=1)
    # --- [FIN] Drop-in 1 ---
    log.info(f"[SHAP] X_raw (test) shape={X_raw.shape} (latentes + {len(meta_cols)} metadatos)")


    # 6) Preprocesamiento + nombres (si hay Pipeline). Si NO hay Pipeline,
    #    trabajamos directo en espacio RAW para ser consistente con el predictor.
    raw_names = np.array(list(map(str, X_raw.columns)))

    if is_pipeline:
        preproc = (pipe.named_steps.get('preproc')
                   or pipe.named_steps.get('transformer')
                   or pipe.named_steps.get('features')
                   or pipe.named_steps.get('prep')
                   or pipe.named_steps.get('scaler'))
        if preproc is None or preproc == 'passthrough':
            preproc = FunctionTransformer(validate=False)

        selector = (pipe.named_steps.get("feature_selector")
                    or pipe.named_steps.get("feature_select")
                    or pipe.named_steps.get("fs"))

        X_proc = preproc.transform(X_raw)
        if selector is not None:
            X_proc = selector.transform(X_proc)

        feat_names, support = _safe_feature_names_after_preproc(preproc, raw_names, selector)
        X_proc_df = pd.DataFrame(X_proc, columns=feat_names)
    else:
        preproc = FunctionTransformer(validate=False)
        selector = None
        feat_names = raw_names
        X_proc_df = X_raw.apply(pd.to_numeric, errors='coerce').astype(float)

    # Detectar latentes directamente en los NOMBRES FINALES (tras preproc/selector)
    latent_mask_proc, latent_idx_proc, _ = _find_latent_columns(feat_names)
    # Por si los nombres "finales" no conservan 'latent_*', generamos nombres latentes canónicos
    latent_cols_proc = np.array([f"latent_{i}" for i in latent_idx_proc], dtype=object)

 

    if latent_mask_proc.sum() == 0:
        log.error("[SHAP] No se detectaron columnas latentes en las features procesadas. "
                "Revisa el preprocesamiento/nombres de columnas.")
        # ayuda de depuración: mostrar algunos nombres
        log.error(f"[SHAP] Ejemplo de nombres: {list(X_proc_df.columns[:10])}")
    else:
        log.info(f"[SHAP] Latentes detectadas en procesado: {int(latent_mask_proc.sum())} / {len(feat_names)}")


    # 7) Background: cargar o construir AHORA (ya existen cnad, tensor_all, norm_params, vae, device, preproc, selector, feat_names)
    bg_path_raw = fold_dir / f"shap_background_raw_{args.clf}_{args.bg_mode}{tag}.joblib"
    bg_path_proc = fold_dir / f"shap_background_proc_{args.clf}_{args.bg_mode}{tag}.joblib"

    if bg_path_proc.exists() and not args.freeze_meta:
        log.info(f"[SHAP] Cargando background PROCESADO (cache): {bg_path_proc.name}")
        background_proc = joblib.load(bg_path_proc)
        if isinstance(background_proc, pd.DataFrame):
            background_proc = _coerce_numeric_df(background_proc, name="background_proc(cache)")

    else:
        # 🔁 Reconstrucción coherente desde RAW (si hay freeze_meta o no existe cache)
        if bg_path_raw.exists():
            log.info(f"[SHAP] Cargando background RAW: {bg_path_raw.name}")
            background_raw = joblib.load(bg_path_raw)
        else:
            log.info(f"[SHAP] No hay background RAW. Construyendo con bg_mode={args.bg_mode}…")
            all_cnad_idx = np.arange(len(cnad))
            train_idx_in_cnad = np.setdiff1d(all_cnad_idx, test_idx_in_cnad, assume_unique=True)
            idx_bg = _pick_bg_indices(cnad, train_idx_in_cnad,
                                      mode=args.bg_mode,
                                      sample_size=min(args.bg_sample_size, len(cnad)),
                                      seed=args.bg_seed)
            background_raw = _build_background_from_indices(
                idx_in_cnad=idx_bg, cnad_df=cnad, tensor_all=tensor_all,
                channels=args.channels_to_use, norm_params=norm_params,
                meta_cols=meta_cols, vae=vae, device=device
            )
        # 🔧 CRÍTICO: forzar numérico en RAW antes de freeze/proc (evita dtype object)
        if isinstance(background_raw, pd.DataFrame):
            background_raw = _coerce_numeric_df(background_raw, name="background_raw")


        # 🧊 Aplicar freeze SIEMPRE en RAW y luego transformar
        if args.freeze_meta:
            for k, v in freeze_values.items():
                if k in background_raw.columns:
                    background_raw.loc[:, k] = v
        joblib.dump(background_raw, bg_path_raw)
        background_proc = _ensure_background_processed(background_raw, preproc, feat_names, selector)
        if isinstance(background_proc, pd.DataFrame):
            background_proc = _coerce_numeric_df(background_proc, name="background_proc(built)")
  
        joblib.dump(background_proc, bg_path_proc)
        log.info(f"[SHAP] Background RAW → {bg_path_raw.name}; PROCESADO → {bg_path_proc.name}")

    # 8) Modelo a explicar y SHAP
    model = unwrap_model_for_shap(model_step, args.clf)
    # Determinar la clase positiva e índice ANTES de crear el explainer (evita NameError)
    classes_src = model_step
    classes_ = list(getattr(classes_src, 'classes_', getattr(model, 'classes_', [0,1])))
    pos_int = label_info['positive_label_int']
    pos_idx = classes_.index(pos_int) if pos_int in classes_ else (1 if len(classes_) > 1 else 0)

    def _predict_linked(X):
        model_step_local = model_step
        # 1) Si hay predict_proba, úsalo
        if hasattr(model_step_local, "predict_proba"):
            proba = model_step_local.predict_proba(X)[:, pos_idx]
            if args.shap_link == 'logit':
                eps = 1e-6
                p = np.clip(proba, eps, 1 - eps)
                return np.log(p / (1 - p))
            return proba
        # 2) Si no, intentar con decision_function (margen)
        if hasattr(model_step_local, "decision_function"):
            margin = model_step_local.decision_function(X)
            if isinstance(margin, np.ndarray) and margin.ndim == 2:
                margin = margin[:, pos_idx]
            # Para 'logit' devolvemos directamente el margen (escala similar a logit)
            if args.shap_link == 'logit':
                return margin
            # Para 'identity' devolvemos sigmoide(margen) como aproximación a prob
            return 1.0 / (1.0 + np.exp(-margin))
        # 3) Último recurso: predict() → float
        return model_step_local.predict(X).astype(float)



    # Bloque corregido para interpretar_fold_paper.py
    # Bloque para elegir el explicador SHAP (alrededor de la línea 660)

    base_val = None # Inicializamos por si acaso

    print(background_proc.dtypes.unique())
    print(X_proc_df.dtypes.unique())
# Deben terminar todos como float64

    # 🔧 doble seguro: convertir justo antes de checks (por caches viejos)
    if isinstance(X_proc_df, pd.DataFrame):
        X_proc_df = _coerce_numeric_df(X_proc_df, name="X_proc_df")
    if isinstance(background_proc, pd.DataFrame):
        background_proc = _coerce_numeric_df(background_proc, name="background_proc(final)")

    assert X_proc_df.shape[1] == background_proc.shape[1]
    assert np.isfinite(X_proc_df.to_numpy(dtype=float)).all()
    assert np.isfinite(background_proc.to_numpy(dtype=float)).all()



    is_calibrated_wrapper = hasattr(model_step, "calibrated_classifiers_")
    if args.clf in {'xgb', 'gb', 'rf', 'lgbm'} and not is_calibrated_wrapper:
        log.info(f"[SHAP] Usando TreeExplainer para {args.clf}.")
        explainer = shap.TreeExplainer(model, background_proc)
        shap_all = explainer.shap_values(X_proc_df)
        base_val = explainer.expected_value

    elif (args.clf == 'logreg' or (
        args.clf == 'svm' and getattr(model, 'kernel', 'linear') == 'linear' and hasattr(model, 'coef_'))
        ) and (not is_calibrated_wrapper) and (not is_pipeline):
        log.info("[SHAP] Usando LinearExplainer para modelo lineal (no pipeline, no calibrado).")
        explainer = shap.LinearExplainer(model, background_proc)
        shap_all = explainer.shap_values(X_proc_df)
        base_val = explainer.expected_value

    # --- reemplaza el bloque "else: Usando KernelExplainer ..." por:
    else:
        log.info("[SHAP] Usando shap.Explainer + Independent masker (permutation/interventional).")
        # El masker conoce la distribución marginal del background y evita artefactos en columnas constantes
        # Asegurar tipos numéricos puros para SHAP (evita Int64/Float64/NA/obj)
        if isinstance(background_proc, pd.DataFrame):
            background_for_masker = background_proc.to_numpy(dtype=float)
        else:
            background_for_masker = np.asarray(background_proc, dtype=float)
        # idem para X_proc_df
        X_proc_df = X_proc_df.apply(pd.to_numeric, errors='coerce').astype(float)
        masker = shap.maskers.Independent(background_for_masker)

        def _predict_pos(X_np: np.ndarray) -> np.ndarray:
            # preservamos nombres (muchos modelos esperan DataFrame, no solo array)
            X_df = pd.DataFrame(X_np, columns=feat_names)
            return _predict_linked(X_df)  # devuelve proba o logit de la clase positiva

        explainer = shap.Explainer(
            _predict_pos,
            masker,
            algorithm="permutation",          # robusto con colinealidad/constantes
            feature_names=feat_names
        )
        # Para Permutation, max_evals debe ser >= 2*F + 1
        F = X_proc_df.shape[1]
        min_required = 2 * F + 1
        # Reutilizamos --kernel_nsamples como presupuesto; si es menor, subimos al mínimo
        budget = getattr(args, "kernel_nsamples", 500)
        max_evals = int(max(min_required, budget))
        log.info(f"[SHAP] Permutation: F={F}, min_required={min_required}, usando max_evals={max_evals}.")
        sv = explainer(
            X_proc_df.to_numpy(dtype=float),
            max_evals=max_evals
        )     # sv.values -> (N,F), sv.base_values -> (N,)
        shap_all = sv.values
        base_val = sv.base_values



    # --- Reducir/normalizar base_value temprano (soporta lista, vector N, matriz N×C) ---
    base_val_raw = base_val
    def _reduce_base_value_to_scalar(bv, pos_idx: int) -> float:
        if isinstance(bv, (list, tuple)):
            bv = np.asarray(bv)
        if isinstance(bv, np.ndarray):
            if bv.ndim == 0:
                return float(bv)
            if bv.ndim == 1:                 # típico de permutation: (N,)
                return float(np.median(bv))  # robusto
            if bv.ndim == 2:                 # (N, clases)
                return float(np.mean(bv[:, pos_idx]))
            return float(np.ravel(bv)[0])
        return float(bv)
    base_val_scalar = _reduce_base_value_to_scalar(base_val, pos_idx)

    # 9) Clase positiva y empaquetado
    shap_pos = _to_sample_feature(shap_all, pos_idx, *X_proc_df.shape)

    # ----- ZEROS para columnas constantes en X_test y background -----
    try:
        const_in_x = X_proc_df.std(numeric_only=True) < 1e-12
        const_in_bg = background_proc.std(numeric_only=True) < 1e-12
        const_mask = (const_in_x & const_in_bg).reindex(X_proc_df.columns).fillna(False).to_numpy()
        if np.any(const_mask):
            shap_pos[:, const_mask] = 0.0
            # opcional: si shap_all es lista/3D (multiclase), replica el zeroing ahí también
            log.info(f"[SHAP] SHAP forzados a 0 por ser constantes: {list(X_proc_df.columns[const_mask])}")
    except Exception as e:
        log.warning(f"[SHAP] Guardia de constantes omitida: {e}")
        # ---------------------------------------------------------------



    # --- NORMALIZACIÓN POST-HOC ENTRE FOLDS (opcional) ---
    if args.shap_normalize != 'none':
        if args.shap_normalize == 'by_logit_median':
            # usar misma definición que el explainer para f(x)
            f = _predict_linked(X_proc_df)
            # Si base_value es vector por muestra, usalo; si no, usá el escalar
            if isinstance(base_val_raw, np.ndarray) and base_val_raw.ndim == 1 and base_val_raw.shape[0] == f.shape[0]:
                diff = np.abs(f - base_val_raw.astype(float))
            else:
                diff = np.abs(f - base_val_scalar)
            med = float(np.median(diff)) if np.isfinite(diff).all() else 0.0
            s = med if med > 1e-12 else 1.0
            shap_pos = shap_pos / s
            base_val_scalar = base_val_scalar / s
            log.info(f"[SHAP] Normalizado por mediana |f(x)-base| (escala ~logit): factor={s:.4g}")
        elif args.shap_normalize == 'by_l1_median':
            l1 = np.sum(np.abs(shap_pos), axis=1)
            s = float(np.median(l1)) if np.median(l1) > 1e-12 else 1.0
            shap_pos = shap_pos / s
            base_val_scalar = base_val_scalar / s
            log.info(f"[SHAP] Normalizado por mediana L1(SHAP): factor={s:.4g}")
        elif args.shap_normalize == 'per_feature_zscore':
            mu = shap_pos.mean(axis=0, keepdims=True)
            sd = shap_pos.std(axis=0, keepdims=True) + 1e-12
            shap_pos = (shap_pos - mu) / sd
            base_val_scalar = 0.0  # pierde aditividad; fijamos base en 0 para plots tipo beeswarm/bar
            log.info("[SHAP] Z-score per feature aplicado (comparabilidad de perfiles, no aditividad).")
    # A partir de acá usamos siempre el escalar reducido
    base_val = float(base_val_scalar)

    pack = {
        'shap_values': shap_pos.astype(np.float32),
        'base_value': float(base_val),
        'X_test': X_proc_df,
        'feature_names': feat_names.tolist(),
        'latent_feature_mask': latent_mask_proc.astype(bool),
        'latent_feature_indices': [int(i) for i in latent_idx_proc],  # índices latentes (enteros)
        'latent_feature_names': latent_cols_proc.tolist(),            # nombres de columnas latentes
        'test_subject_ids': test_df['SubjectID'].astype(str).tolist(),
        'test_labels': test_df['ResearchGroup_Mapped'].map({'CN': 0, 'AD': 1}).astype(int).tolist(),
        'latent_features_type': args.latent_features_type,
        'metadata_features': meta_cols,
        'seed_used': int(args.seed),
    }
    pack_path = out_dir / f'shap_pack_{args.clf}{tag}.joblib'
    joblib.dump(pack, pack_path)
    log.info(f"[SHAP] Pack guardado: {pack_path}")

    _plot_shap_summary(shap_pos, X_proc_df, out_dir, args.fold, args.clf, base_val)


def _find_latent_columns(feature_names: Sequence[str]) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """
    Devuelve:
      - mask booleano de columnas latentes,
      - lista con los índices latentes (enteros) detectados,
      - array con los nombres originales de columnas que contienen 'latent_<idx>'
    Acepta nombres tipo: 'latent_12', 'num__latent_12', 'scaler__latent_12', etc.
    """
    import re
    pat = re.compile(r'latent_(\d+)\b')
    mask = np.zeros(len(feature_names), dtype=bool)
    latent_indices: List[int] = []
    cols_matched: List[str] = []
    for i, name in enumerate(map(str, feature_names)):
        m = pat.search(name)
        if m:
            mask[i] = True
            latent_indices.append(int(m.group(1)))
            cols_matched.append(name)
    return mask, latent_indices, np.array(cols_matched, dtype=object)


def _coerce_numeric_df(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """
    Fuerza df a float puro (evita dtype=object que rompe np.isfinite/SHAP).
    Convierte no-numéricos a NaN y luego imputa con mediana por columna (fallback 0).
    """
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if out.isna().any().any():
        # mediana por columna (solo numéricas); luego fallback global a 0
        med = out.median(numeric_only=True)
        out = out.fillna(med)
        out = out.fillna(0.0)

    # cast final (evita Int64/Float64 pandas nullable dtypes)
    return out.astype(float)

def get_latent_weights_from_pack(pack: Dict[str, Any], mode: str, top_k: Optional[int]) -> pd.DataFrame:
    """Calcula pesos de las features latentes a partir de un shap_pack.

    mode:
        * mean_abs         → media de |SHAP| en todos los sujetos.
        * mean_signed      → media signed en todos los sujetos.
        * ad_vs_cn_diff    → (media SHAP AD) − (media SHAP CN)  (recomendado).
    """
    shap_values = pack['shap_values']            # (N, F)
    feature_names = pack['feature_names']        # list len=F
    labels = np.asarray(pack['test_labels'])     # (N,)

    # Preferir la máscara latente guardada en el pack (más robusta);
    # si no existe, fallback regex.
    if 'latent_feature_mask' in pack and isinstance(pack['latent_feature_mask'], (list, np.ndarray)):
        latent_mask = np.asarray(pack['latent_feature_mask'], dtype=bool)
        # Alinear por seguridad si la longitud difiere
        if latent_mask.shape[0] != len(feature_names):
            log.warning("[SALIENCY] Máscara latente del pack no alinea; se usará fallback regex.")
            import re
            latent_mask = np.array([bool(re.search(r'(?:^|__)latent_\d+$', n)) for n in feature_names])
    else:
        import re
        latent_mask = np.array([bool(re.search(r'(?:^|__)latent_\d+$', n)) for n in feature_names])

    latent_vals = shap_values[:, latent_mask]
    latent_names = np.array(feature_names)[latent_mask]

    if mode == 'mean_abs':
        importance = np.abs(latent_vals).mean(axis=0)
    elif mode == 'mean_signed':
        importance = latent_vals.mean(axis=0)
    elif mode == 'ad_vs_cn_diff':
        imp_ad = latent_vals[labels == 1].mean(axis=0)
        imp_cn = latent_vals[labels == 0].mean(axis=0)
        importance = imp_ad - imp_cn
    else:
        raise ValueError(f"Modo de pesos SHAP no reconocido: {mode}")

    df = pd.DataFrame({'feature': latent_names, 'importance': importance})
    df['latent_idx'] = (
        df['feature']
        .str.extract(r'(\d+)$', expand=False)
        .astype(int)
    )

    # ordenar por magnitud absoluta (para top_k)
    df = df.reindex(df['importance'].abs().sort_values(ascending=False).index)
    if top_k is not None and top_k > 0:
        df = df.head(min(top_k, len(df)))
    # pesos normalizados (usar magnitud absoluta para normalizar; conservar signo en importance si te interesa)
    denom = df['importance'].abs().sum()
    df['weight'] = 0.0 if denom == 0 else df['importance'] / denom
    return df[['latent_idx', 'weight', 'importance', 'feature']]


def _vae_kwargs_from_args(args: argparse.Namespace, image_size: int) -> Dict[str, Any]:
    return dict(
        input_channels=len(args.channels_to_use),
        latent_dim=args.latent_dim,
        image_size=image_size,
        dropout_rate=args.dropout_rate_vae,
        use_layernorm_fc=getattr(args, 'use_layernorm_vae_fc', False),
        num_conv_layers_encoder=args.num_conv_layers_encoder,
        decoder_type=args.decoder_type,
        intermediate_fc_dim_config=args.intermediate_fc_dim_vae,
        final_activation=args.vae_final_activation,
        num_groups=args.gn_num_groups,
    )


def _load_roi_names(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == '.joblib':
        return joblib.load(path)
    if path.suffix == '.npy':
        return np.load(path, allow_pickle=True).astype(str).tolist()
    # txt/csv
    return pd.read_csv(path, header=None).iloc[:, 0].astype(str).tolist()


def generate_saliency_vectorized(vae_model: ConvolutionalVAE,
                                 weights_df: pd.DataFrame,
                                 input_tensor: torch.Tensor,
                                 device: torch.device) -> np.ndarray:
    """Genera saliencia devolviendo (signed, abs), ambos (C,R,R)."""
    if input_tensor.numel() == 0:
        z = np.zeros((vae_model.input_channels, vae_model.image_size, vae_model.image_size), dtype=np.float32)
        return z, z

    vae_model.eval()
    x = input_tensor.clone().detach().to(device)
    x.requires_grad = True

    # vector de pesos latentes (1, latent_dim)
    w = torch.zeros((1, vae_model.latent_dim), device=device, dtype=torch.float32)
    idx = torch.as_tensor(weights_df['latent_idx'].values, device=device, dtype=torch.long)
    vals = torch.as_tensor(weights_df['weight'].values, device=device, dtype=torch.float32)
    w[0, idx] = vals
    w = w.repeat(x.shape[0], 1)  # expand a batch

    vae_model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        # asumiendo forward encode() devuelve (mu, logvar)
        mu, _ = vae_model.encode(x)  # shape (B, latent_dim)
        mu.backward(gradient=w)

    signed, absmap = _grad_to_signed_and_abs(x.grad)
    return signed, absmap


def generate_saliency_smoothgrad(vae_model: ConvolutionalVAE,
                                 weights_df: pd.DataFrame,
                                 input_tensor: torch.Tensor,
                                 device: torch.device,
                                 n_samples: int = 10,
                                 noise_std_perc: float = 0.15) -> np.ndarray:
    """Genera (signed, abs) con SmoothGrad."""
    if input_tensor.numel() == 0:
        z = np.zeros((vae_model.input_channels, vae_model.image_size, vae_model.image_size), dtype=np.float32)
        return z, z

    input_std = torch.std(input_tensor)
    noise_std = input_std * noise_std_perc
    
    total_signed = np.zeros((vae_model.input_channels, vae_model.image_size, vae_model.image_size), dtype=np.float32)
    total_abs    = np.zeros_like(total_signed)

    for _ in range(n_samples):
        noise = torch.randn_like(input_tensor) * noise_std
        noisy_input = input_tensor + noise
        s, a = generate_saliency_vectorized(vae_model, weights_df, noisy_input, device)
        total_signed += s
        total_abs    += a

    return total_signed / n_samples, total_abs / n_samples


def generate_saliency_integrated_gradients(vae_model: ConvolutionalVAE,
                                           weights_df: pd.DataFrame,
                                           input_tensor: torch.Tensor,
                                           device: torch.device,
                                           baseline: Optional[torch.Tensor] = None,
                                           n_steps: int = 50) -> np.ndarray:
    """Genera (signed, abs) con Integrated Gradients (Captum)."""
    if input_tensor.numel() == 0:
        z = np.zeros((vae_model.input_channels, vae_model.image_size, vae_model.image_size), dtype=np.float32)
        return z, z
    if IntegratedGradients is None:
        raise ImportError("Captum no está instalado. No se puede usar 'integrated_gradients'.")

    # 1. Definir el vector de pesos latentes (target para la atribución)
    w = torch.zeros(vae_model.latent_dim, device=device, dtype=torch.float32)
    idx = torch.as_tensor(weights_df['latent_idx'].values, device=device, dtype=torch.long)
    vals = torch.as_tensor(weights_df['weight'].values, device=device, dtype=torch.float32)
    w[idx] = vals

    # 2. Wrapper para Captum: calcula w^T * mu(x)
    def model_forward(x: torch.Tensor) -> torch.Tensor:
        mu, _ = vae_model.encode(x)
        return (mu * w).sum(dim=1)

    # 3. Calcular atribuciones
    ig = IntegratedGradients(model_forward)
    # Baseline:
    #  - si no se pasa, usar cero (comportamiento previo);
    #  - si se pasa (C,R,R) o (1,C,R,R), expandir al batch.
    if baseline is None:
        baselines = torch.zeros_like(input_tensor).to(device)
    else:
        b = baseline.to(device)
        if b.ndim == 3:              # (C,R,R) → (1,C,R,R)
            b = b.unsqueeze(0)
        # expandir a (N,C,R,R) igual que input
        if b.shape[0] == 1 and input_tensor.shape[0] > 1:
            b = b.expand(input_tensor.shape[0], -1, -1, -1)
        baselines = b

    attributions = ig.attribute(input_tensor.to(device), baselines=baselines, n_steps=n_steps)
    A = attributions.mean(dim=0).cpu().numpy()
    signed = _project_to_H(A)
    absmap = _project_to_H(np.abs(A))
    return signed.astype(np.float32), absmap.astype(np.float32)


def _ensure_background_processed(
        background_data: Any,
        preproc: Any,
        feat_names_target: Sequence[str],
        selector: Optional[Any] = None
) -> pd.DataFrame:

    if isinstance(background_data, pd.DataFrame):
        # 1️⃣  Caso ideal: ya vienen con los nombres correctos
        if list(background_data.columns) == list(feat_names_target):
            # 🔧 AUN ASÍ: garantizar float puro (si no, dtype object rompe np.isfinite/SHAP)
            df = background_data.apply(pd.to_numeric, errors='coerce')
            df = df.fillna(df.median(numeric_only=True)).fillna(0.0)
            return df.astype(float)

        # 2️⃣  Mismo nº de columnas ⇒ asumimos que YA está procesado,
        #     solo le ponemos los nombres que espera el clasificador
        if background_data.shape[1] == len(feat_names_target):
            log.info("[SHAP] Background ya parece procesado; renombrando columnas.")
            df = background_data.set_axis(feat_names_target, axis=1, copy=False)
            return df.apply(pd.to_numeric, errors='coerce').astype(float)


        # 3️⃣  Si no, intentamos procesarlo desde cero
        log.info("[SHAP] Background DataFrame detectado pero columnas no coinciden; transformando…")
        X_proc = preproc.transform(background_data) if hasattr(preproc, "transform") else background_data.values
        if selector is not None:
            X_proc = selector.transform(X_proc)
        df = pd.DataFrame(X_proc, columns=feat_names_target)
        return df.apply(pd.to_numeric, errors='coerce').astype(float)
    # -----------------------------------------------------------------
    # resto del cuerpo idéntico, pero añade selector en el branch ndarray
    if isinstance(background_data, np.ndarray):
        if background_data.shape[1] != len(feat_names_target):
            # Este caso es ambiguo: ¿es un array crudo o uno ya procesado con otro
            # preprocesador? El comportamiento más seguro es fallar o, como mínimo,
            # registrar una advertencia severa, ya que no podemos asumir cómo procesarlo.
            raise ValueError(
                f"Background ndarray tiene {background_data.shape[1]} columnas, "
                f"pero se esperaban {len(feat_names_target)}. No se puede continuar de forma segura."
            )
        # Si el número de columnas coincide, lo convertimos a DataFrame.
        return pd.DataFrame(
            np.asarray(background_data, dtype=float),
            columns=feat_names_target
        )
    # -----------------------------------------------------------------
    raise TypeError(f"Tipo de background desconocido: {type(background_data)}")

# --- NUEVO: helper para aplicar freeze en DF procesado (columnas finales) ---
def _apply_freeze_in_processed_df(df_proc: pd.DataFrame,
                                  freeze_values: Dict[str, float]) -> pd.DataFrame:
    """
    Fija columnas de metadatos ya transformadas/nombradas en el espacio procesado,
    buscando sufijos 'Age'/'__Age' y 'Sex'/'__Sex'.
    """
    cols = list(map(str, df_proc.columns))
    def _targets(key: str) -> List[str]:
        return [c for c in cols if c.endswith(key) or c.endswith(f"__{key}")]
    out = df_proc.copy()
    for k, v in freeze_values.items():
        for c in _targets(k):
            out.loc[:, c] = float(v)
    return out


def _compute_cn_median_baseline(
    *,
    cnad_df: pd.DataFrame,
    tensor_all: np.ndarray,
    channels: Sequence[int],
    norm_params: List[Dict[str, float]],
    test_idx_in_cnad: np.ndarray
) -> torch.Tensor:
    """
    Devuelve un baseline (C,R,R) como la mediana por elemento de los sujetos CN del *TRAIN* del fold,
    en el espacio ya normalizado (mismos params que el VAE del fold).
    """
    # TRAIN del clasificador en este fold (CN/AD): todo menos test
    all_cnad_idx = np.arange(len(cnad_df))
    train_idx_in_cnad = np.setdiff1d(all_cnad_idx, test_idx_in_cnad, assume_unique=True)
    train_df = cnad_df.iloc[train_idx_in_cnad]
    cn_train_df = train_df[train_df['ResearchGroup_Mapped'] == 'CN']
    if cn_train_df.empty:
        log.warning("[IG] No hay CN en TRAIN del fold; se intentará con CN del TEST como fallback.")
        cn_train_df = cnad_df.iloc[test_idx_in_cnad][cnad_df.iloc[test_idx_in_cnad]['ResearchGroup_Mapped']=='CN']
        if cn_train_df.empty:
            raise RuntimeError("[IG] No hay CN disponibles en TRAIN ni TEST para construir baseline.")

    gidx = cn_train_df['tensor_idx'].values
    tens = tensor_all[gidx][:, channels, :, :]                  # (N,C,R,R) en escala original
    tens = apply_normalization_params(tens, norm_params)        # normalizado como VAE

    # mediana por elemento → (C,R,R)
    median = np.median(tens, axis=0).astype(np.float32)
    # proyectar a simétrico y hueco, por robustez
    median = 0.5*(median + median.transpose(0,2,1))
    for c in range(median.shape[0]):
        np.fill_diagonal(median[c], 0.0)
    return torch.from_numpy(median).float()                     # (C,R,R)



# ==============================================================================
# FUNCIÓN DE PLOTEO CORREGIDA
# ==============================================================================
def _plot_shap_summary(shap_pos, X_proc_df, out_dir, fold, clf, base_val):
    # Asegurar tipos numéricos "puros"
    vals = np.asarray(shap_pos, dtype=np.float64)
    feats = np.asarray(X_proc_df.values, dtype=np.float64)
    names = list(X_proc_df.columns)
    base = np.full(vals.shape[0], float(base_val), dtype=np.float64)

    # Construimos un Explanation para usar la API moderna
    exp = shap.Explanation(
        values=vals,
        base_values=base,
        data=feats,
        feature_names=names
    )

    # BAR (global importance)
    plt.figure(figsize=(10, 8))
    shap.plots.bar(exp, max_display=20, show=False)
    plt.title(f'SHAP Importancia Global (bar) - Fold {fold} - {clf.upper()}')
    plt.tight_layout()
    suffix = ""  # dejamos el mismo nombre para compat; el pack ya lleva tag
    plt.savefig(out_dir / f'shap_global_importance_bar{suffix}.png', dpi=150)
    plt.close()

    # BEESWARM (impacto por feature)
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(exp, max_display=20, show=False)
    plt.title(f'SHAP Impacto Features (beeswarm) - Fold {fold} - {clf.upper()}')
    plt.tight_layout()
    plt.savefig(out_dir / f'shap_summary_beeswarm{suffix}.png', dpi=150)
    plt.close()

    # WATERFALL (primer sujeto)
    if vals.shape[0] > 0:
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(exp[0], show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(out_dir / f'shap_waterfall_subject_0{suffix}.png', dpi=150)
        plt.close()



def cmd_saliency(args: argparse.Namespace) -> None:
    # Semillas para reproducibilidad también en la etapa de saliencia
    _set_all_seeds(args.seed)
    fold_dir = Path(args.run_dir) / f"fold_{args.fold}"
    shap_dir = fold_dir / 'interpretability_shap'
    tag = f"_{args.shap_tag}" if getattr(args, 'shap_tag', None) else ""
    pack_path = shap_dir / f'shap_pack_{args.clf}{tag}.joblib'
    if not pack_path.exists():
        raise FileNotFoundError(f"No se encontró shap_pack para {args.clf} en {pack_path}. Corre primero el subcomando 'shap'.")

    pack = joblib.load(pack_path)
    log.info(f"[SALIENCY] fold={args.fold} clf={args.clf}  (pack cargado: {pack_path.name})")
    # ------------------------------------------------------------------
    roi_names: Optional[List[str]] = None
    roi_order_joblib = Path(args.run_dir) / 'roi_order_131.joblib'
    if roi_order_joblib.exists():
        roi_names = joblib.load(roi_order_joblib)
        log.info(f"Usando ROI order de {roi_order_joblib}.")
    elif getattr(args, "roi_order_path", None):
        p_roi = Path(args.roi_order_path)
        roi_names = _load_roi_names(p_roi)
        log.info(f"Usando ROI order de --roi_order_path={p_roi}.")

    # Intento con anotaciones si aún no tenemos roi_names
    roi_map_df: Optional[pd.DataFrame] = None
    annot_path = args.roi_annotation_path or args.roi_annotation_csv
    if annot_path is not None:
        try:
            roi_map_df = pd.read_csv(annot_path)
            log.info(f"Cargado fichero de anotaciones: {annot_path}")
            if roi_names is None:
                # ordenar por ROI_TensorIdx si existe; si no, por Original_Index_0_N
                if "ROI_TensorIdx" in roi_map_df.columns:
                    roi_map_df = roi_map_df.sort_values("ROI_TensorIdx").reset_index(drop=True)
                elif "Original_Index_0_N" in roi_map_df.columns:
                    roi_map_df = roi_map_df.sort_values("Original_Index_0_N").reset_index(drop=True)

                # --- NUEVO: detectar automáticamente la columna con los nombres de ROI ---
                name_col = None
                for cand in ["roi_name_in_tensor","AAL3_Name", "ROI_Name", "Name", "Label"]:
                    if cand in roi_map_df.columns:
                        name_col = cand
                        break
                if name_col is None:
                    # último recurso: la primera columna no-índice
                    name_col = roi_map_df.columns[0]
                    log.warning(
                        "[SALIENCY] Columna 'AAL3_Name' no encontrada; "
                        f"usando '{name_col}' como nombres de ROI."
                    )
                else:
                    log.info(f"[SALIENCY] Usando columna '{name_col}' como nombres de ROI.")

                roi_names = roi_map_df[name_col].astype(str).tolist()

                # Truco: renombrar internamente a 'AAL3_Name' para que _ranking_and_heatmap también funcione
                if name_col != "AAL3_Name":
                    roi_map_df = roi_map_df.rename(columns={name_col: "AAL3_Name"})

                log.info("Derivado orden de ROIs desde el CSV de anotaciones.")
        except FileNotFoundError:
            log.error(f"No se pudo leer el fichero de anotaciones: {annot_path}")

    if roi_names is None:
        raise FileNotFoundError(
            "No pude resolver el orden de ROIs. Proporciona --roi_order_path "
            "o asegúrate de tener roi_order_131.joblib o un CSV anotado con AAL3_Name."
        )


    # Pesos latentes ----------------------------------------------------------
    weights_df = get_latent_weights_from_pack(pack, args.shap_weight_mode, args.top_k)
    log.info(f"[SALIENCY] {len(weights_df)} latentes ponderadas. Ejemplo:\n{weights_df.head().to_string(index=False)}")

    # Datos test en espacio de entrada original --------------------------------
    tensor_all, merged = _load_global_and_merge(Path(args.global_tensor_path), Path(args.metadata_path))
    cnad = _subset_cnad(merged)
    test_idx_in_cnad = np.load(fold_dir / 'test_indices.npy')
    test_df = cnad.iloc[test_idx_in_cnad].copy()
    gidx_test = test_df['tensor_idx'].values

    norm_params = joblib.load(fold_dir / 'vae_norm_params.joblib')
    tens_test = tensor_all[gidx_test][:, args.channels_to_use, :, :]
    tens_test = apply_normalization_params(tens_test, norm_params)
    tens_test_t = torch.from_numpy(tens_test).float()

    n_rois_tensor = tens_test.shape[-1]
    if len(roi_names) != n_rois_tensor:
        log.error(f"Número de ROIs ({len(roi_names)}) != dimensión tensor ({n_rois_tensor}). "
                  "Verifica que el orden de ROIs corresponde al tensor usado en entrenamiento.")
        raise ValueError("Desajuste longitud roi_names vs tensor.")

    # VAE ----------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae_kwargs = _vae_kwargs_from_args(args, image_size=tens_test.shape[-1])
    vae = build_vae(vae_kwargs, fold_dir / f"vae_model_fold_{args.fold}.pt", device)

    labels = np.asarray(pack['test_labels'])  # CN=0 / AD=1 según pack
    x_ad = tens_test_t[labels == 1]
    x_cn = tens_test_t[labels == 0]

    log.info(f"[SALIENCY] Sujetos AD={x_ad.shape[0]}  CN={x_cn.shape[0]}")

    # ------- IG classifier-score mode (early path) -------
    saliency_mode = getattr(args, "saliency_mode", "latent")
    if saliency_mode == "ig_classifier_score":
        log.info("[SALIENCY] Mode: ig_classifier_score — using actual classifier weights.")
        # Need pipeline + feature_columns for this mode
        pipe_path = fold_dir / f"classifier_{args.clf}_pipeline_fold_{args.fold}.joblib"
        if not pipe_path.exists():
            raise FileNotFoundError(f"Pipeline not found: {pipe_path}")
        pipe = joblib.load(pipe_path)
        fc_path = fold_dir / "feature_columns.json"
        if fc_path.exists():
            with open(fc_path) as f:
                feature_columns = json.load(f)["final_feature_columns"]
        else:
            feature_columns = ([f"latent_{i}" for i in range(args.latent_dim)]
                               + (args.metadata_features or []))
        meta_cols = args.metadata_features or []

        # IG baseline
        ig_base = None
        if getattr(args, "ig_baseline", "cn_median_train") == "cn_median_train":
            ig_base = _compute_cn_median_baseline(
                cnad_df=cnad, tensor_all=tensor_all,
                channels=args.channels_to_use, norm_params=norm_params,
                test_idx_in_cnad=test_idx_in_cnad,
            )
        elif getattr(args, "ig_baseline", None) == "cn_median_test" and x_cn.shape[0] > 0:
            ig_base = torch.median(x_cn, dim=0).values.detach().cpu()

        sal_ad_signed, sal_ad_abs, st_ad = generate_saliency_ig_classifier_score(
            vae, pipe, x_ad, device, args.latent_dim,
            meta_cols, feature_columns, baseline=ig_base,
            n_steps=getattr(args, "ig_n_steps", 50),
        )
        sal_cn_signed, sal_cn_abs, st_cn = generate_saliency_ig_classifier_score(
            vae, pipe, x_cn, device, args.latent_dim,
            meta_cols, feature_columns, baseline=ig_base,
            n_steps=getattr(args, "ig_n_steps", 50),
        )
        if sal_ad_signed is None or sal_cn_signed is None:
            raise RuntimeError(
                f"[SALIENCY] ig_classifier_score failed: AD={st_ad} CN={st_cn}. "
                "This mode requires a linear model (logreg/linear SVM)."
            )
        sal_diff_signed = sal_ad_signed - sal_cn_signed
        sal_diff_abs = np.abs(sal_diff_signed)
        method_tag = "_ig_clf_score"
        file_suffix = f"{method_tag}_top{args.top_k}"
        out_dir = fold_dir / f"interpretability_{args.clf}"
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / f"saliency_map_ad_signed{file_suffix}.npy", sal_ad_signed)
        np.save(out_dir / f"saliency_map_ad_abs{file_suffix}.npy", sal_ad_abs)
        np.save(out_dir / f"saliency_map_cn_signed{file_suffix}.npy", sal_cn_signed)
        np.save(out_dir / f"saliency_map_cn_abs{file_suffix}.npy", sal_cn_abs)
        np.save(out_dir / f"saliency_map_diff_signed{file_suffix}.npy", sal_diff_signed)
        np.save(out_dir / f"saliency_map_diff_abs{file_suffix}.npy", sal_diff_abs)

        _ranking_and_heatmap(
            saliency_map_diff_signed=sal_diff_signed,
            saliency_map_diff_abs=sal_diff_abs,
            roi_map_df=roi_map_df,
            roi_names=roi_names,
            out_dir=out_dir, fold=args.fold, clf=args.clf,
            top_k=args.top_k, method_tag=method_tag,
        )
        with open(out_dir / f"run_args_saliency{file_suffix}.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        log.info(f"[SALIENCY] ig_classifier_score complete. Results in {out_dir}")
        return

    # ------- Standard latent-weighted saliency path -------
    # ------- Construir baseline para IG (si aplica) -------
    ig_baseline_tensor = None

    log.info(f"[SALIENCY] Usando método de saliencia: {args.saliency_method}")

    saliency_fn_map = {
        'vanilla': generate_saliency_vectorized,
        'smoothgrad': lambda v, w, x, d: generate_saliency_smoothgrad(
            v, w, x, d, n_samples=args.sg_n_samples, noise_std_perc=args.sg_noise_std
        ),
        'integrated_gradients': lambda v, w, x, d: generate_saliency_integrated_gradients(
            v, w, x, d, baseline=ig_baseline_tensor, n_steps=args.ig_n_steps
        ),
    }
    saliency_fn = saliency_fn_map[args.saliency_method]

    out_dir = fold_dir / f"interpretability_{args.clf}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # si el método es IG, resolver baseline según el flag
    if args.saliency_method == 'integrated_gradients':
        if args.ig_baseline == 'zeros':
            log.info("[IG] Baseline: tensor de ceros (mismo espacio normalizado).")
            ig_baseline_tensor = None
        elif args.ig_baseline == 'cn_median_test':
            log.info("[IG] Baseline: mediana CN del TEST del fold (mismo espacio normalizado).")
            if x_cn.shape[0] == 0:
                raise RuntimeError("[IG] No hay CN en el TEST para baseline 'cn_median_test'. Usa 'cn_median_train'.")
            cn_med = torch.median(x_cn, dim=0).values.detach().cpu()  # (C,R,R)
            ig_baseline_tensor = cn_med
        elif args.ig_baseline == 'cn_median_train':
            log.info("[IG] Baseline: mediana CN del TRAIN del fold (sin fuga de test).")
            ig_baseline_tensor = _compute_cn_median_baseline(
                cnad_df=cnad,
                tensor_all=tensor_all,
                channels=args.channels_to_use,
                norm_params=norm_params,
                test_idx_in_cnad=test_idx_in_cnad
            )
        np.save(out_dir / f"ig_baseline_{args.ig_baseline}.npy",
                (ig_baseline_tensor.cpu().numpy() if ig_baseline_tensor is not None else np.zeros_like(tens_test_t[0].cpu().numpy())))

    sal_ad_signed, sal_ad_abs = saliency_fn(vae, weights_df, x_ad, device)
    sal_cn_signed, sal_cn_abs = saliency_fn(vae, weights_df, x_cn, device)
    sal_diff_signed = sal_ad_signed - sal_cn_signed
    # diferencia en sentido signed; la variante abs se define como |diff_signed|
    sal_diff_abs = np.abs(sal_diff_signed)

    # Guardar mapas ------------------------------------------------------------
    out_dir = fold_dir / f"interpretability_{args.clf}"
    out_dir.mkdir(parents=True, exist_ok=True)
    method_tag = "" if args.saliency_method == "vanilla" else f"_{args.saliency_method}"
    file_suffix = f"{method_tag}_top{args.top_k}"

    # Grabar por grupo y diferencial (signed y abs)
    np.save(out_dir / f"saliency_map_ad_signed{file_suffix}.npy",  sal_ad_signed)
    np.save(out_dir / f"saliency_map_ad_abs{file_suffix}.npy",     sal_ad_abs)
    np.save(out_dir / f"saliency_map_cn_signed{file_suffix}.npy",  sal_cn_signed)
    np.save(out_dir / f"saliency_map_cn_abs{file_suffix}.npy",     sal_cn_abs)
    np.save(out_dir / f"saliency_map_diff_signed{file_suffix}.npy", sal_diff_signed)
    np.save(out_dir / f"saliency_map_diff_abs{file_suffix}.npy",    sal_diff_abs)
 

    _ranking_and_heatmap(
        saliency_map_diff_signed=sal_diff_signed,
        saliency_map_diff_abs=sal_diff_abs,
        roi_map_df=roi_map_df,
        roi_names=roi_names,
        out_dir=out_dir,
        fold=args.fold,
        clf=args.clf,
        top_k=args.top_k,
        method_tag=method_tag
    )

    # ===== NEW =====
    # (1) Contribución por canal desde el mapa diferencial
    l1_abs = sal_diff_abs.sum(axis=(1,2))  # L1 (abs)
    l1_sgn = np.sign(sal_diff_signed).sum(axis=(1,2))  # suma de signos (diagnóstico)
    frac_abs = l1_abs / (l1_abs.sum() + 1e-12)          # fracción relativa
    # nombres de canales (si el usuario los pasa, usarlos; si no, usar índices)
    ch_used = list(args.channels_to_use)
    if getattr(args, "channel_names", None) and len(args.channel_names) == len(ch_used):
        ch_names = list(args.channel_names)
    else:
        ch_names = [f"Ch{c}" for c in ch_used]
    # guardar con nombres de columnas amigables para el notebook
    chan_df = pd.DataFrame({
        'channel_index_used': ch_used,
        'channel_name': ch_names,
        'l1_norm_abs': l1_abs,
        'l1_norm_fraction_abs': frac_abs,
        'signed_sum': l1_sgn
    })
    # Guardar con y sin sufijo para que el notebook lo encuentre directo
    chan_csv_suff = out_dir / f'channel_contributions{file_suffix}.csv'
    chan_csv_nosuff = out_dir / 'channel_contributions.csv'
    chan_df.to_csv(chan_csv_suff, index=False)
    chan_df.to_csv(chan_csv_nosuff, index=False)

    plt.figure(figsize=(6,4)); plt.bar(np.arange(len(l1_abs)), frac_abs)
    plt.xlabel('Channel'); plt.ylabel('Fraction of total |ΔSal|')
    plt.title(f'Channel contributions – fold {args.fold}'); plt.tight_layout()
    plt.savefig(out_dir / f'channel_contributions{file_suffix}.png', dpi=150)
    plt.savefig(out_dir / 'channel_contributions.png', dpi=150)
    plt.close()

    # (2) Network-pair matrices and quick visuals from the annotated ranking
    edge_csv = out_dir / f"ranking_conexiones_ANOTADO{file_suffix}.csv"
    if edge_csv.exists():
        df_edges = pd.read_csv(edge_csv)
        # choose network columns available
        net_src_col = 'src_Refined_Network' if 'src_Refined_Network' in df_edges.columns else 'src_Yeo17_Network'
        net_dst_col = 'dst_Refined_Network' if 'dst_Refined_Network' in df_edges.columns else 'dst_Yeo17_Network'
        nets = sorted(set(df_edges[net_src_col].astype(str)) | set(df_edges[net_dst_col].astype(str)))
        pos = {n:i for i,n in enumerate(nets)}
        M_abs = np.zeros((len(nets),len(nets)), float)
        M_sgn = np.zeros_like(M_abs)
        for _, r in df_edges.iterrows():
            i = pos[str(r[net_src_col])]; j = pos[str(r[net_dst_col])]
            v = float(r['Saliency_Signed']); a = float(r['Saliency_Abs'])
            M_abs[i,j]+=a; M_abs[j,i]+=a; M_sgn[i,j]+=v; M_sgn[j,i]+=v
        pd.DataFrame(M_abs, index=nets, columns=nets).to_csv(out_dir / f'network_pairs_sumabs{file_suffix}.csv', index=True)
        pd.DataFrame(M_sgn, index=nets, columns=nets).to_csv(out_dir / f'network_pairs_signed{file_suffix}.csv', index=True)
        # visuals (default colormap)
        plt.figure(figsize=(8,7)); plt.imshow(M_abs)
        plt.xticks(range(len(nets)), nets, rotation=90); plt.yticks(range(len(nets)), nets)
        plt.title('Network-pair energy (sum |ΔSal|)'); plt.colorbar()
        plt.tight_layout(); plt.savefig(out_dir / f'heatmap_network_pairs_sumabs{file_suffix}.png', dpi=150); plt.close()
        plt.figure(figsize=(8,7)); plt.imshow(M_sgn)
        plt.xticks(range(len(nets)), nets, rotation=90); plt.yticks(range(len(nets)), nets)
        plt.title('Network-pair signed (ΔSal, AD>CN +)'); plt.colorbar()
        plt.tight_layout(); plt.savefig(out_dir / f'heatmap_network_pairs_signed{file_suffix}.png', dpi=150); plt.close()

    # (3) Robust hubs (degree-controlled node-strength) for K in {50,100,200}
    if edge_csv.exists():
        df_edges = pd.read_csv(edge_csv)
        for K in (50,100,200):
            sub = df_edges[df_edges['Rank']<=K].copy()
            nodes = sorted(set(sub['src_AAL3_Name'].astype(str)) | set(sub['dst_AAL3_Name'].astype(str)))
            deg = {n:0 for n in nodes}; strg = {n:0.0 for n in nodes}
            for _, r in sub.iterrows():
                a = str(r['src_AAL3_Name']); b = str(r['dst_AAL3_Name'])
                # usar Saliency_Abs si existe; si no, caer a |Saliency_Signed|
                col = 'Saliency_Abs' if 'Saliency_Abs' in r else 'Saliency_Signed'
                w = float(abs(r[col]))
                deg[a]+=1; deg[b]+=1; strg[a]+=w; strg[b]+=w
            # Construir tabla y residualizar fuerza respecto a grado
            tab = pd.DataFrame({
                'node': nodes,
                'degree': [deg[n] for n in nodes],
                'strength': [strg[n] for n in nodes]
            })
            if tab['degree'].max() > 0:
                x = tab['degree'].to_numpy(dtype=float)
                y = tab['strength'].to_numpy(dtype=float)
                slope, intercept = np.polyfit(x, y, 1)
                tab['residual_strength'] = y - (intercept + slope * x)
            else:
                tab['residual_strength'] = 0.0
            tab.sort_values('residual_strength', ascending=False).to_csv(out_dir / f'node_robust_hubs_top{K}{file_suffix}.csv', index=False)
    # ===== END NEW =====

 
    # Guardar args usados ------------------------------------------------------
    with open(out_dir / f"run_args_saliency{file_suffix}.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    log.info(f"[SALIENCY] Completo. Resultados en {out_dir}")


# ---------------------------------------------------------------------------
# Ranking + visualización
# ---------------------------------------------------------------------------

def _ranking_and_heatmap(saliency_map_diff_signed: np.ndarray,
                         saliency_map_diff_abs: np.ndarray,
                         roi_map_df: Optional[pd.DataFrame],
                         roi_names: Sequence[str],
                         out_dir: Path,
                         fold: int,
                         clf: str,
                         top_k: int,
                         annot_df: Optional[pd.DataFrame] = None,
                         method_tag: str = "") -> None:
    # Ambos (C,R,R) → promediamos sobre canales
    sal_m_sgn = saliency_map_diff_signed.mean(axis=0)
    sal_m_abs = saliency_map_diff_abs.mean(axis=0)
    n_rois = sal_m_sgn.shape[0]
    # Crear la tabla de conexiones con índices numéricos
    ut_indices = np.triu_indices(n_rois, k=1)
    df_edges = pd.DataFrame({
        'idx_i': ut_indices[0],
        'idx_j': ut_indices[1],
        'Saliency_Signed': sal_m_sgn[ut_indices],
        'Saliency_Abs':    sal_m_abs[ut_indices]
    })
    if roi_map_df is not None:
        # Alinear al orden del tensor
        idx_col = None
        if 'ROI_TensorIdx' in roi_map_df.columns:
            roi_map_df = roi_map_df.sort_values('ROI_TensorIdx').reset_index(drop=True)
            idx_col = 'ROI_TensorIdx'
        elif 'Original_Index_0_N' in roi_map_df.columns:
            roi_map_df = roi_map_df.sort_values('Original_Index_0_N').reset_index(drop=True)
            idx_col = 'Original_Index_0_N'
        elif len(roi_map_df) == n_rois:
            # Fallback: asumir que las filas ya están en el mismo orden que el tensor
            log.warning(
                "roi_map_df sin columna de índice explícita; "
                "se usará el índice 0..N-1 como ROI_TensorIdx (asumiendo orden coherente con el tensor)."
            )
            roi_map_df = roi_map_df.reset_index(drop=True).copy()
            roi_map_df['ROI_TensorIdx'] = np.arange(n_rois, dtype=int)
            idx_col = 'ROI_TensorIdx'
        else:
            log.warning(
                "roi_map_df no tiene columnas de índice reconocibles ni longitud compatible; "
                "se omitirá la anotación."
            )
            idx_col = None

        if idx_col is not None:
            idxed = roi_map_df.set_index(idx_col)

            # Nombres de ROI (AAL3 o equivalente abreviado)
            if 'AAL3_Name' in idxed.columns:
                name_map = idxed['AAL3_Name'].astype(str).to_dict()
            else:
                # Fallback: nombre = índice
                name_map = {i: str(i) for i in range(n_rois)}

            # Lóbulos (si existen en el CSV)
            lobe_map = idxed['Macro_Lobe'].astype(str).to_dict() if 'Macro_Lobe' in idxed.columns else {}

            # Redes "refinadas": si no hay Refined_Network, usamos la red Yeo como proxy
            if 'Refined_Network' in idxed.columns:
                net_map = idxed['Refined_Network'].astype(str).to_dict()
            elif 'network_label_in_tensor' in idxed.columns:
                net_map = idxed['network_label_in_tensor'].astype(str).to_dict()
            else:
                net_map = {}

            # Redes Yeo-17: admitir tanto Yeo17_Network como network_label_in_tensor
            if 'Yeo17_Network' in idxed.columns:
                y17_map = idxed['Yeo17_Network'].astype(str).to_dict()
            elif 'network_label_in_tensor' in idxed.columns:
                y17_map = idxed['network_label_in_tensor'].astype(str).to_dict()
            else:
                y17_map = {}

            # Añadir columnas al ranking
            df_edges['ROI_i_name'] = df_edges['idx_i'].map(name_map)
            df_edges['ROI_j_name'] = df_edges['idx_j'].map(name_map)
            df_edges['src_AAL3_Name'] = df_edges['ROI_i_name']
            df_edges['dst_AAL3_Name'] = df_edges['ROI_j_name']
            df_edges['src_Macro_Lobe'] = df_edges['idx_i'].map(lobe_map)
            df_edges['dst_Macro_Lobe'] = df_edges['idx_j'].map(lobe_map)
            df_edges['src_Refined_Network'] = df_edges['idx_i'].map(net_map)
            df_edges['dst_Refined_Network'] = df_edges['idx_j'].map(net_map)
            df_edges['src_Yeo17_Network'] = df_edges['idx_i'].map(y17_map)
            df_edges['dst_Yeo17_Network'] = df_edges['idx_j'].map(y17_map)
   


    df_edges = df_edges.sort_values('Saliency_Abs', ascending=False)
    df_edges.insert(0, 'Rank', range(1, len(df_edges) + 1))
    file_suffix = f"{method_tag}_top{top_k}"
    edge_csv_path = out_dir / f"ranking_conexiones_ANOTADO{file_suffix}.csv"
    df_edges.to_csv(edge_csv_path, index=False)
    log.info(f"[SALIENCY] Ranking de conexiones ANOTADO guardado: {edge_csv_path}")

    if annot_df is not None:
        meta = annot_df[['AAL3_Name','Macro_Lobe','Refined_Network']]
        df_edges = (df_edges
                    .merge(meta, left_on='ROI_i_name', right_on='AAL3_Name', how='left')
                    .rename(columns={'Macro_Lobe':'Lobe_i','Refined_Network':'Network_i'})
                    .drop(columns='AAL3_Name')
                    .merge(meta, left_on='ROI_j_name', right_on='AAL3_Name', how='left')
                    .rename(columns={'Macro_Lobe':'Lobe_j','Refined_Network':'Network_j'})
                    .drop(columns='AAL3_Name'))
    # preview top 20
    preview_cols = [
        'Rank', 'src_AAL3_Name', 'dst_AAL3_Name', 'Saliency_Signed', 'Saliency_Abs',
        'src_Refined_Network', 'dst_Refined_Network'
    ]
    preview_cols_exist = [c for c in preview_cols if c in df_edges.columns]
    log.info("Top 20 conexiones anotadas:\n" + df_edges.head(20)[preview_cols_exist].to_string())
 
 

    # heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(sal_m_sgn, cmap='coolwarm', center=0,
                xticklabels=list(roi_names)[:n_rois], yticklabels=list(roi_names)[:n_rois],
                cbar_kws={'label': 'Saliencia Diferencial (AD > CN)'} )
    plt.title(f'Mapa de Saliencia Diferencial (AD vs CN) - Fold {fold} - {clf.upper()}{method_tag.replace("_", " ").title()}')
    plt.tight_layout(); plt.savefig(out_dir / f"mapa_saliencia_diferencial{file_suffix}.png", dpi=150); plt.close()


# ---------------------------------------------------------------------------
# SHAP edges (subcomando "shap_edges")
# ---------------------------------------------------------------------------

def _load_roi_names_for_edges(run_dir: Path, args: argparse.Namespace,
                              R: int) -> List[str]:
    """
    Robust ROI name loading with multiple fallbacks.

    Priority:
      1. roi_order_131.joblib  (list[str] or dict with 'roi_names_in_order')
      2. roi_info_from_tensor.joblib  (dict with 'roi_names_in_order')
      3. --roi_order_path  (list or dict)
      4. --roi_annotation_path  (CSV with ROI name column)
      5. Generic names  ["ROI_0", ..., "ROI_{R-1}"]
    """
    roi_names = None
    source = None

    # --- 1) roi_order_131.joblib ---
    p1 = run_dir / "roi_order_131.joblib"
    if p1.exists():
        obj = joblib.load(p1)
        if isinstance(obj, list):
            roi_names, source = obj, str(p1)
        elif isinstance(obj, dict) and "roi_names_in_order" in obj:
            roi_names, source = obj["roi_names_in_order"], str(p1)
        else:
            log.warning(f"[ROI] roi_order_131.joblib has unexpected type "
                        f"{type(obj).__name__}; skipping.")

    # --- 2) roi_info_from_tensor.joblib ---
    if roi_names is None:
        p2 = run_dir / "roi_info_from_tensor.joblib"
        if p2.exists():
            obj = joblib.load(p2)
            if isinstance(obj, dict) and "roi_names_in_order" in obj:
                roi_names, source = obj["roi_names_in_order"], str(p2)
            elif isinstance(obj, list):
                roi_names, source = obj, str(p2)
            else:
                log.warning(f"[ROI] roi_info_from_tensor.joblib has unexpected "
                            f"type/keys; skipping.")

    # --- 3) --roi_order_path ---
    if roi_names is None and getattr(args, "roi_order_path", None):
        p3 = Path(args.roi_order_path)
        obj = _load_roi_names(p3)  # handles .joblib/.npy/.csv
        if isinstance(obj, dict) and "roi_names_in_order" in obj:
            roi_names, source = obj["roi_names_in_order"], str(p3)
        elif isinstance(obj, list):
            roi_names, source = obj, str(p3)

    # --- 4) --roi_annotation_path ---
    if roi_names is None and getattr(args, "roi_annotation_path", None):
        p4 = Path(args.roi_annotation_path)
        if p4.exists():
            try:
                df = pd.read_csv(p4)
                for cand in ("roi_name_in_tensor", "AAL3_Name",
                             "ROI_Name", "Name", "Label"):
                    if cand in df.columns:
                        roi_names = df[cand].astype(str).tolist()
                        source = f"{p4} (col={cand})"
                        break
            except Exception as e:
                log.warning(f"[ROI] Cannot read annotation CSV {p4}: {e}")

    # --- 5) Fallback: generic ---
    if roi_names is None:
        roi_names = [f"ROI_{i}" for i in range(R)]
        source = "generic_fallback"
        log.warning(f"[ROI] No ROI names found. Using generic names ROI_0..ROI_{R-1}.")

    # Validation
    assert len(roi_names) == R, (
        f"ROI names length mismatch: got {len(roi_names)}, expected {R} (from {source})"
    )
    log.info(f"[ROI] Loaded {len(roi_names)} ROI names from {source}")
    return roi_names


def cmd_shap_edges(args: argparse.Namespace) -> None:
    """Composite edge-level SHAP: attribute P(AD) directly to brain edges."""
    _set_all_seeds(args.seed)
    fold_dir = Path(args.run_dir) / f"fold_{args.fold}"
    out_dir = fold_dir / "interpretability_shap_edges"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.shap_tag}" if getattr(args, "shap_tag", None) else ""

    edge_vector_mode = getattr(args, "edge_vector_mode", "upper")
    use_per_channel = getattr(args, "edge_select_per_channel", False)
    bg_mode = getattr(args, "bg_mode", "train")
    n_test_shap = getattr(args, "n_test_shap", None)

    log.info(f"[SHAP_EDGES] fold={args.fold} clf={args.clf} K={args.edge_K} "
             f"vector_mode={edge_vector_mode} bg_mode={bg_mode} "
             f"per_channel={use_per_channel} n_test_shap={n_test_shap}")

    # 1) Load fold artifacts --------------------------------------------------
    pipe_path = fold_dir / f"classifier_{args.clf}_pipeline_fold_{args.fold}.joblib"
    if not pipe_path.exists():
        raise FileNotFoundError(f"Pipeline not found: {pipe_path}")
    pipe = joblib.load(pipe_path)

    norm_params = joblib.load(fold_dir / "vae_norm_params.joblib")
    label_info = _load_label_info(fold_dir)

    # Feature columns (exact order the pipeline expects)
    fc_path = fold_dir / "feature_columns.json"
    if fc_path.exists():
        with open(fc_path) as f:
            fc_info = json.load(f)
        feature_columns = fc_info["final_feature_columns"]
    else:
        feature_columns = ([f"latent_{i}" for i in range(args.latent_dim)]
                           + (args.metadata_features or []))
        log.warning("[SHAP_EDGES] feature_columns.json not found; using fallback column order.")

    # 2) Load data -------------------------------------------------------------
    tensor_all, merged = _load_global_and_merge(
        Path(args.global_tensor_path), Path(args.metadata_path)
    )
    cnad = _subset_cnad(merged)
    test_idx_in_cnad = np.load(fold_dir / "test_indices.npy")
    all_cnad_idx = np.arange(len(cnad))
    train_idx_in_cnad = np.setdiff1d(all_cnad_idx, test_idx_in_cnad, assume_unique=True)

    train_df = cnad.iloc[train_idx_in_cnad].copy()
    test_df = cnad.iloc[test_idx_in_cnad].copy()
    channels = list(args.channels_to_use)
    R = tensor_all.shape[-1]
    C = len(channels)

    # ROI names (robust loading, AFTER R is known)
    roi_names = _load_roi_names_for_edges(Path(args.run_dir), args, R)

    # 3) Normalize tensors -----------------------------------------------------
    gidx_train = train_df["tensor_idx"].values
    gidx_test = test_df["tensor_idx"].values

    tens_train = apply_normalization_params(
        tensor_all[gidx_train][:, channels, :, :], norm_params
    )
    tens_test = apply_normalization_params(
        tensor_all[gidx_test][:, channels, :, :], norm_params
    )

    # 4) Vectorize to edges ----------------------------------------------------
    mirror = True
    if edge_vector_mode == "full_offdiag":
        idx_rows, idx_cols, total_edges = make_edge_index_offdiag(R, C)
        mirror = False
        log.warning("[SHAP_EDGES] full_offdiag mode: matrices may not preserve "
                    "symmetry under SHAP perturbation. Use with caution.")
    else:
        idx_rows, idx_cols, total_edges = make_edge_index(R, C)
    edges_per_channel = len(idx_rows)
    edges_train = vectorize_tensor_to_edges(tens_train, idx_rows, idx_cols)
    edges_test = vectorize_tensor_to_edges(tens_test, idx_rows, idx_cols)
    log.info(f"[SHAP_EDGES] R={R} C={C} edges_per_channel={edges_per_channel} "
             f"total_edges={total_edges} mode={edge_vector_mode}")

    # 5) Feature selection (TRAIN only) ----------------------------------------
    per_ch_tag = "_perch" if use_per_channel else ""
    cache_path = (out_dir /
                  f"edge_selection_{args.edge_select_method}_K{args.edge_K}"
                  f"_{edge_vector_mode}{per_ch_tag}_seed{args.seed}.joblib")
    if cache_path.exists():
        log.info(f"[SHAP_EDGES] Loading cached edge selection: {cache_path.name}")
        cache = joblib.load(cache_path)
        selected_idx = cache["selected_indices"]
        scores = cache["scores"]
    else:
        y_train = (train_df["ResearchGroup_Mapped"]
                   .map({"CN": 0, "AD": 1}).values.astype(int))
        if use_per_channel:
            selected_idx, scores, sel_info = select_top_edges_per_channel(
                edges_train, y_train, K=args.edge_K, C=C,
                edges_per_channel=edges_per_channel,
                method=args.edge_select_method, seed=args.seed,
            )
        else:
            selected_idx, scores, sel_info = select_top_edges(
                edges_train, y_train, K=args.edge_K,
                method=args.edge_select_method, seed=args.seed,
            )
        joblib.dump(
            {"selected_indices": selected_idx, "scores": scores,
             "method": args.edge_select_method, "K": args.edge_K,
             "seed": args.seed, "per_channel": use_per_channel,
             "edge_vector_mode": edge_vector_mode},
            cache_path,
        )
        log.info(f"[SHAP_EDGES] Edge selection cached: {cache_path.name}")

    K = len(selected_idx)
    log.info(f"[SHAP_EDGES] Selected K={K} edges via {args.edge_select_method}"
             f"{' (per-channel)' if use_per_channel else ''}")

    # 6) Template (TRAIN median) & frozen metadata -----------------------------
    all_edges_template = compute_train_edge_median(edges_train)
    meta_cols = args.metadata_features or []
    meta_frozen = compute_frozen_meta_values(train_df, meta_cols)
    log.info(f"[SHAP_EDGES] Frozen metadata: {meta_frozen}")

    # 7) Build VAE --------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_kwargs = _vae_kwargs_from_args(args, image_size=R)
    vae = build_vae(
        vae_kwargs, fold_dir / f"vae_model_fold_{args.fold}.pt", device
    )

    # 8) Determine pos_idx -----------------------------------------------------
    is_pipeline = hasattr(pipe, "named_steps")
    model_step = (pipe.named_steps["model"]
                  if is_pipeline and "model" in pipe.named_steps
                  else pipe)
    model_for_classes = model_step
    classes_ = list(getattr(model_for_classes, "classes_",
                            getattr(unwrap_model_for_shap(model_step, args.clf),
                                    "classes_", [0, 1])))
    pos_int = label_info["positive_label_int"]
    pos_idx = (classes_.index(pos_int)
               if pos_int in classes_
               else (1 if len(classes_) > 1 else 0))

    # 9) Build predict_fn -------------------------------------------------------
    shap_link = getattr(args, "shap_link", "identity")
    predict_fn = make_edge_predict_fn(
        vae=vae, pipe=pipe,
        meta_cols=meta_cols, meta_frozen_values=meta_frozen,
        feature_columns=feature_columns,
        selected_edge_indices=selected_idx,
        all_edges_template=all_edges_template,
        R=R, C=C, triu_rows=idx_rows, triu_cols=idx_cols,
        device=device,
        latent_features_type=args.latent_features_type,
        pos_idx=pos_idx,
        shap_link=shap_link,
        mirror=mirror,
    )

    # 10) Validation check ------------------------------------------------------
    if not getattr(args, "skip_validation", False):
        passed, max_diff = validate_edge_roundtrip(
            vae=vae, pipe=pipe,
            tensor_test_norm=tens_test, test_df=test_df,
            meta_cols=meta_cols, feature_columns=feature_columns,
            triu_rows=idx_rows, triu_cols=idx_cols,
            device=device, latent_features_type=args.latent_features_type,
            pos_idx=pos_idx,
            n_check=min(10, len(test_df)),
            atol=getattr(args, "validation_atol", 1e-5),
            mirror=mirror,
        )
        if not passed:
            raise RuntimeError(
                f"[SHAP_EDGES] Validation FAILED: max|diff|={max_diff:.2e}. "
                "The edge vectorize/reconstruct roundtrip is not lossless. "
                "Check tensor symmetry and zero diagonal."
            )
    else:
        log.warning("[SHAP_EDGES] Validation skipped (--skip_validation).")

    # 11) SHAP background (leakage-safe) ---------------------------------------
    bg_size = getattr(args, "edge_bg_size", 50)
    if bg_mode == "train":
        bg_pool = train_idx_in_cnad
    elif bg_mode == "global":
        # All subjects (including test) — acceptable for SHAP background
        bg_pool = all_cnad_idx
    elif bg_mode == "global_cn":
        # CN from TRAIN only — leakage-safe: no test data, no label leak
        cn_mask = (cnad["ResearchGroup_Mapped"].astype(str) == "CN").values
        cn_train_mask = np.zeros(len(cnad), dtype=bool)
        cn_train_mask[train_idx_in_cnad] = True
        cn_train_pool = np.where(cn_mask & cn_train_mask)[0]
        if len(cn_train_pool) >= max(10, bg_size // 2):
            bg_pool = cn_train_pool
        else:
            log.warning(f"[SHAP_EDGES] global_cn: only {len(cn_train_pool)} CN in "
                        f"TRAIN (need {bg_size}). Falling back to train.")
            bg_pool = train_idx_in_cnad
    else:
        raise ValueError(f"Unknown bg_mode: {bg_mode}")

    rng_bg = np.random.RandomState(args.seed)
    if len(bg_pool) > bg_size:
        bg_idx = np.sort(rng_bg.choice(bg_pool, size=bg_size, replace=False))
    else:
        bg_idx = np.sort(bg_pool)
    log.info(f"[SHAP_EDGES] Background: mode={bg_mode}, n={len(bg_idx)}")

    tens_bg = apply_normalization_params(
        tensor_all[cnad.iloc[bg_idx]["tensor_idx"].values][:, channels, :, :],
        norm_params,
    )
    edges_bg_full = vectorize_tensor_to_edges(tens_bg, idx_rows, idx_cols)
    edges_bg_selected = edges_bg_full[:, selected_idx].astype(np.float64)
    edges_test_selected = edges_test[:, selected_idx].astype(np.float64)

    # 11b) Optional test subsampling (n_test_shap) ----------------------------
    test_subsample_idx = None
    if n_test_shap is not None and n_test_shap < edges_test_selected.shape[0]:
        rng_sub = np.random.RandomState(args.seed + 1)
        test_subsample_idx = np.sort(
            rng_sub.choice(edges_test_selected.shape[0],
                           size=n_test_shap, replace=False)
        )
        edges_test_for_shap = edges_test_selected[test_subsample_idx]
        log.info(f"[SHAP_EDGES] Subsampled test: {n_test_shap}/{edges_test_selected.shape[0]}")
    else:
        edges_test_for_shap = edges_test_selected

    # Feature names for selected edges
    all_edge_names = make_edge_feature_names(
        roi_names, channels,
        getattr(args, "channel_names", None),
        idx_rows=idx_rows, idx_cols=idx_cols,
    )
    selected_edge_names = [all_edge_names[i] for i in selected_idx]

    # 12) SHAP computation ------------------------------------------------------
    masker = shap.maskers.Independent(edges_bg_selected)
    explainer = shap.Explainer(
        predict_fn, masker,
        algorithm="permutation",
        feature_names=selected_edge_names,
    )
    min_evals = 2 * K + 1
    budget = getattr(args, "edge_shap_nsamples", 500)
    max_evals = int(max(min_evals, budget))
    log.info(f"[SHAP_EDGES] Permutation SHAP: K={K}, max_evals={max_evals}, "
             f"N_test_shap={edges_test_for_shap.shape[0]}")

    sv = explainer(edges_test_for_shap, max_evals=max_evals)
    shap_values = sv.values        # (N_shap, K)
    base_values = sv.base_values   # (N_shap,) or scalar

    # Reduce base_value
    if isinstance(base_values, np.ndarray) and base_values.ndim >= 1:
        base_val_scalar = float(np.median(base_values))
    else:
        base_val_scalar = float(base_values)

    # 13) Zero out constant columns ---------------------------------------------
    try:
        std_test = np.std(edges_test_for_shap, axis=0)
        std_bg = np.std(edges_bg_selected, axis=0)
        const_mask = (std_test < 1e-12) & (std_bg < 1e-12)
        if np.any(const_mask):
            shap_values[:, const_mask] = 0.0
            n_const = int(const_mask.sum())
            log.info(f"[SHAP_EDGES] Zeroed SHAP for {n_const} constant edges.")
    except Exception as e:
        log.warning(f"[SHAP_EDGES] Constant-column guard skipped: {e}")

    # 14) Edge mapping ---------------------------------------------------------
    mapping_df = make_edge_mapping_df(
        selected_idx, roi_names, channels,
        getattr(args, "channel_names", None),
        idx_rows, idx_cols, edges_per_channel, scores,
    )

    # 15) Save pack & CSV -------------------------------------------------------
    # Labels/ids: use full test set metadata, mark which were used for SHAP
    test_labels = (test_df["ResearchGroup_Mapped"]
                   .map({"CN": 0, "AD": 1}).astype(int).tolist())
    test_ids = test_df["SubjectID"].astype(str).tolist()

    pack = {
        "shap_values": shap_values.astype(np.float32),
        "base_value": base_val_scalar,
        "base_values_per_sample": (np.asarray(base_values, dtype=np.float32)
                                   if isinstance(base_values, np.ndarray)
                                   else np.full(shap_values.shape[0],
                                                base_val_scalar, dtype=np.float32)),
        "X_test_edges_selected": edges_test_for_shap.astype(np.float32),
        "feature_names_edges_selected": selected_edge_names,
        "selected_edge_indices": selected_idx,
        "edge_to_roi_mapping": mapping_df,
        "edge_K": K,
        "edge_select_method": args.edge_select_method,
        "edge_select_per_channel": use_per_channel,
        "edge_vector_mode": edge_vector_mode,
        "edge_selection_scores": scores,
        "test_subject_ids": test_ids,
        "test_labels": test_labels,
        "test_subsample_indices": test_subsample_idx,
        "n_test_shap": edges_test_for_shap.shape[0],
        "meta_frozen_values": meta_frozen,
        "bg_mode": bg_mode,
        "R": R, "C": C,
        "edges_per_channel": edges_per_channel,
        "total_edges": total_edges,
        "seed": int(args.seed),
        "shap_link": shap_link,
        "validation_passed": not getattr(args, "skip_validation", False),
    }
    pack_path = out_dir / f"shap_pack_edges_{args.clf}{tag}.joblib"
    joblib.dump(pack, pack_path)
    log.info(f"[SHAP_EDGES] Pack saved: {pack_path}")

    mapping_df.to_csv(
        out_dir / f"edge_to_roi_mapping_{args.clf}{tag}.csv", index=False,
    )

    # 16) Plots -----------------------------------------------------------------
    X_shap_df = pd.DataFrame(
        edges_test_for_shap.astype(np.float64), columns=selected_edge_names,
    )
    _plot_shap_summary(
        shap_values.astype(np.float64),
        X_shap_df, out_dir, args.fold, args.clf, base_val_scalar,
    )

    # 17) Save args -------------------------------------------------------------
    with open(out_dir / f"run_args_shap_edges{tag}.json", "w") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()}, f, indent=2)

    log.info(f"[SHAP_EDGES] Complete. Results in {out_dir}")


# ---------------------------------------------------------------------------
# IG with actual classifier weights
# ---------------------------------------------------------------------------

def generate_saliency_ig_classifier_score(
    vae_model: ConvolutionalVAE,
    pipe: Any,
    input_tensor: torch.Tensor,
    device: torch.device,
    latent_dim: int,
    meta_cols: List[str],
    feature_columns: List[str],
    baseline: Optional[torch.Tensor] = None,
    n_steps: int = 50,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    IG using actual classifier weights (logreg coef_) instead of
    SHAP-derived proxy weights.

    Returns (signed, abs, status) where status is "ok", "surrogate",
    or an error string. If status != "ok", signed/abs are None.
    """
    if IntegratedGradients is None:
        return None, None, "captum_not_installed"

    w_latent, bias, platt_a, platt_b, status = extract_logreg_latent_weights(
        pipe, latent_dim, meta_cols, feature_columns,
    )

    if status != "ok" or w_latent is None:
        log.warning(f"[IG_CLF] Cannot extract classifier weights: {status}. "
                    "IG with classifier score not available for this model.")
        return None, None, status

    # Build torch weight vector
    w_t = torch.from_numpy(w_latent).float().to(device)
    b_t = torch.tensor(bias or 0.0, dtype=torch.float32, device=device)

    def model_forward(x: torch.Tensor) -> torch.Tensor:
        mu, _ = vae_model.encode(x)
        score = (mu * w_t).sum(dim=1) + b_t
        if platt_a is not None and platt_b is not None:
            # Platt scaling: P = 1 / (1 + exp(a * score + b))
            # We differentiate through this for IG
            a_t = torch.tensor(platt_a, dtype=torch.float32, device=device)
            b_platt = torch.tensor(platt_b, dtype=torch.float32, device=device)
            score = torch.sigmoid(-(a_t * score + b_platt))
        return score

    ig = IntegratedGradients(model_forward)

    if baseline is None:
        baselines = torch.zeros_like(input_tensor).to(device)
    else:
        b = baseline.to(device)
        if b.ndim == 3:
            b = b.unsqueeze(0)
        if b.shape[0] == 1 and input_tensor.shape[0] > 1:
            b = b.expand(input_tensor.shape[0], -1, -1, -1)
        baselines = b

    attributions = ig.attribute(
        input_tensor.to(device), baselines=baselines, n_steps=n_steps,
    )
    A = attributions.mean(dim=0).cpu().numpy()
    signed = _project_to_H(A)
    absmap = _project_to_H(np.abs(A))
    return signed.astype(np.float32), absmap.astype(np.float32), "ok"


# ---------------------------------------------------------------------------
# Argumentos CLI
# ---------------------------------------------------------------------------

def _add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--roi_annotation_csv', default=None,
               help='CSV con anotaciones de ROI (Macro_Lobe, Redes, etc.).')
    p.add_argument('--run_dir', required=True, help='Directorio raíz del experimento (donde viven fold_*).')
    p.add_argument('--fold', type=int, required=True, help='Fold a analizar (1-indexed).')
    p.add_argument('--clf', required=True, help='Clasificador (xgb, svm, logreg, gb, rf, ...).')
    p.add_argument('--global_tensor_path', required=True, help='Ruta al GLOBAL_TENSOR .npz usado en entrenamiento.')
    p.add_argument('--metadata_path', required=True, help='Ruta al CSV de metadatos usado en entrenamiento.')
    p.add_argument('--channels_to_use', type=int, nargs='*', required=True, help='Índices de canales usados en entrenamiento.')
    p.add_argument('--latent_dim', type=int, required=True, help='Dimensión latente del VAE.')
    p.add_argument('--latent_features_type', choices=['mu','z'], default='mu', help='Usar mu o z como features latentes.')
    p.add_argument('--metadata_features', nargs='*', default=None, help='Columnas de metadatos añadidas al clasificador.')
    # Arquitectura VAE
    p.add_argument('--seed', type=int, default=42, help='Semilla global para numpy/torch/shap.')
    p.add_argument('--num_conv_layers_encoder', type=int, default=4)
    p.add_argument('--decoder_type', default='convtranspose', choices=['convtranspose','upsample_conv'])
    p.add_argument('--dropout_rate_vae', type=float, default=0.2)
    p.add_argument('--use_layernorm_vae_fc', action='store_true')
    p.add_argument('--intermediate_fc_dim_vae', default='quarter')
    p.add_argument('--vae_final_activation', default='tanh', choices=['tanh','sigmoid','linear'])
    p.add_argument('--gn_num_groups', type=int, default=16, help='n grupos para GroupNorm en VAE.')
    p.add_argument('--channel_names', nargs='*', default=None,
                   help='(Opcional) nombres legibles de los canales, longitud = len(channels_to_use).')



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Pipeline Unificado de Interpretabilidad (VAE+Clasificador).',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    # subcomando SHAP ---------------------------------------------------------
    p_shap = sub.add_parser('shap', help='Calcular y guardar valores SHAP para un fold+clf.')
    _add_shared_args(p_shap)

    p_shap.add_argument('--roi_order_path', default=None, help='(Opcional) ruta a ROI order si no está en run_dir.')
    p_shap.add_argument('--kernel_nsamples', type=int, default=100, help='nsamples para KernelExplainer (modelos no tree).')

    p_shap.add_argument('--shap_link', default='identity', choices=['identity','logit'],
                        help="Escala de explicación; 'logit' aplica log(p/(1-p)) a predict_proba[:,pos].")
    # --- ESTABILIZAR BACKGROUND ENTRE FOLDS ---
    p_shap.add_argument('--bg_mode', default='train',
                        choices=['train','global','global_cn'],
                        help="train: background del TRAIN del fold (actual); global: muestra fija CN+AD; global_cn: sólo CN.")
    p_shap.add_argument('--bg_sample_size', type=int, default=100, help='Tamaño del background canónico.')
    p_shap.add_argument('--bg_seed', type=int, default=42, help='Semilla para muestreo del background canónico.')

    # --- NORMALIZACIÓN POST-HOC DE SHAP PARA COMPARABILIDAD ---
    p_shap.add_argument('--shap_normalize', default='none',
                        choices=['none','by_logit_median','by_l1_median','per_feature_zscore'],
                        help="Escala común entre folds: 'by_logit_median' recomendado; alternativas: 'by_l1_median' o 'per_feature_zscore'.")
    p_shap.add_argument('--freeze_meta', nargs='*', default=None,
                        help="Lista opcional de metadatos a congelar (p.ej.: Age Sex).")
    p_shap.add_argument('--freeze_strategy', default='train_stats',
                        choices=['train_stats','constants'],
                        help="train_stats: Age=mediana(TRAIN), Sex=modo(TRAIN). constants: Age=70, Sex=0.")
    p_shap.add_argument('--shap_tag', default=None,
                        help="Sufijo para no pisar artefactos (p.ej., 'frozen' o 'unfrozen').")

    # subcomando SALIENCY -----------------------------------------------------
    p_sal = sub.add_parser('saliency', help='Generar mapas de saliencia a partir del shap_pack.')
    _add_shared_args(p_sal)
    p_sal.add_argument('--roi_order_path', default=None,
                    help='(Opcional) ruta a ROI order (.joblib/.npy/.txt) si no está en run_dir.')
    p_sal.add_argument('--roi_annotation_path', required=True, help='Ruta al fichero maestro de anotaciones de ROIs (roi_info_master.csv).')
    p_sal.add_argument('--top_k', type=int, default=50, help='Nº máx features latentes a usar.')
    p_sal.add_argument('--shap_weight_mode', default='ad_vs_cn_diff', choices=['mean_abs','mean_signed','ad_vs_cn_diff'],
                       help='Cómo convertir valores SHAP latentes en pesos para saliencia.')
    p_sal.add_argument('--saliency_mode', default='latent',
                       choices=['latent', 'ig_classifier_score'],
                       help="'latent': SHAP-weighted latent saliency (default). "
                            "'ig_classifier_score': IG using actual classifier weights (logreg/linear SVM).")
    p_sal.add_argument('--saliency_method', default='vanilla', choices=['vanilla', 'smoothgrad', 'integrated_gradients'],
                       help='Método para generar el mapa de saliencia (only used if saliency_mode=latent).')
    # Args para SmoothGrad
    p_sal.add_argument('--sg_n_samples', type=int, default=10, help='[SmoothGrad] Nº de muestras con ruido a promediar.')
    p_sal.add_argument('--sg_noise_std', type=float, default=0.15, help='[SmoothGrad] Desv. estándar del ruido como % de la desv. estándar de la entrada.')
    # Args para Integrated Gradients
    p_sal.add_argument('--ig_n_steps', type=int, default=50, help='[IG] Nº de pasos para la aproximación de la integral.')
    
    p_sal.add_argument('--ig_baseline',
                       default='cn_median_train',
                       choices=['zeros', 'cn_median_train', 'cn_median_test'],
                       help='Baseline para IG: ceros (antiguo), mediana CN del TRAIN (recomendado), o mediana CN del TEST.')
    p_sal.add_argument('--shap_tag', default=None,
                        help="Sufijo del pack SHAP a usar (debe coincidir con el usado en 'shap').")

    # subcomando SHAP_EDGES ---------------------------------------------------
    p_edges = sub.add_parser('shap_edges',
        help='Composite edge-level SHAP: attribute P(AD) directly to brain edges.')
    _add_shared_args(p_edges)
    p_edges.add_argument('--roi_order_path', default=None,
        help='(Optional) path to ROI order (.joblib/.npy/.txt) if not in run_dir.')
    p_edges.add_argument('--roi_annotation_path', default=None,
        help='CSV with ROI annotations (networks, lobes) for edge annotation.')
    # Edge vectorization mode
    p_edges.add_argument('--edge_vector_mode', default='upper',
        choices=['upper', 'full_offdiag'],
        help="'upper': triu k=1 (default, lossless for symmetric). "
             "'full_offdiag': all i!=j (for non-symmetric matrices).")
    # Edge selection
    p_edges.add_argument('--edge_K', type=int, required=True,
        help='Number of top edges to select for SHAP explanation.')
    p_edges.add_argument('--edge_select_method', default='f_classif',
        choices=['f_classif', 'mutual_info', 'l1_logreg'],
        help='Method for selecting top-K edges from TRAIN data.')
    p_edges.add_argument('--edge_select_per_channel', action='store_true',
        help='Select K/C edges per channel (balanced) instead of global top-K.')
    # SHAP computation
    p_edges.add_argument('--edge_shap_nsamples', type=int, default=500,
        help='max_evals budget for SHAP permutation explainer.')
    p_edges.add_argument('--edge_bg_size', type=int, default=50,
        help='Number of samples for SHAP background (in edge space).')
    p_edges.add_argument('--bg_mode', default='train',
        choices=['train', 'global', 'global_cn'],
        help="Background source: 'train' (leakage-safe, default), "
             "'global' (all subjects), 'global_cn' (CN from TRAIN only).")
    p_edges.add_argument('--n_test_shap', type=int, default=None,
        help='Subsample test set to N subjects for SHAP (reproducible). '
             'None = use all test subjects.')
    p_edges.add_argument('--shap_link', default='identity',
        choices=['identity', 'logit'],
        help="Output scale: 'identity' for proba, 'logit' for log-odds.")
    # Validation
    p_edges.add_argument('--skip_validation', action='store_true',
        help='Skip the pipeline-equivalence validation check.')
    p_edges.add_argument('--validation_atol', type=float, default=1e-5,
        help='Absolute tolerance for validation check.')
    # Tag
    p_edges.add_argument('--shap_tag', default=None,
        help='Suffix for output filenames (avoid overwriting).')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    # Semillas globales también aquí (por si algún camino llama antes)
    if hasattr(args, 'seed'):
        _set_all_seeds(int(args.seed))
    if args.cmd == 'shap':
        cmd_shap(args)
    elif args.cmd == 'saliency':
        cmd_saliency(args)
    elif args.cmd == 'shap_edges':
        cmd_shap_edges(args)
    else:
        raise ValueError(f"Subcomando desconocido: {args.cmd}")


if __name__ == '__main__':
    main()
