# src/betavae_xai/data/preprocessing.py
"""
betavae_xai.data.preprocessing

Carga y preprocesamiento del tensor de conectividad y metadatos.

Este módulo encapsula:
- load_data(): carga .npz + CSV, alinea por SubjectID y agrega tensor_idx
- normalize_inter_channel_fold(): estima stats SOLO en train_idx (anti-leakage) y normaliza
- apply_normalization_params(): aplica params ya estimados (útil para inferencia externa)
- _zero_diag_inplace(): asegura diagonal en 0 (evita valores espurios / autoconectividad)

Nota:
La lógica de normalización (off-diagonal) es crítica para estabilidad del VAE.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Parámetros fijos opcionales (por canal) para MinMax off-diagonal.
# Estructura esperada por canal:
#   FIXED_MINMAX_PARAMS_PER_CHANNEL["ChannelName"] = {"min": float, "max": float}
# ---------------------------------------------------------------------
FIXED_MINMAX_PARAMS_PER_CHANNEL: Dict[str, Dict[str, float]] = {}


def _zero_diag_inplace(x: np.ndarray) -> None:
    """
    Setea la diagonal a 0 de un array con shape (..., R, R).
    Implementación segura para tensores 4D/5D sin boolean indexing 2D sobre tensores mayores.
    """
    if x is None or not isinstance(x, np.ndarray):
        return
    if x.ndim < 2:
        return
    r = x.shape[-1]
    diag = np.arange(r)
    x[..., diag, diag] = 0.0


def load_data(
    tensor_path: Path,
    metadata_path: Path,
) -> Tuple[
    Optional[np.ndarray],
    Optional[pd.DataFrame],
    Optional[List[str]],
    Optional[List[str]],
]:
    """
    Carga el tensor global (.npz) y el CSV de metadatos y hace merge por SubjectID.

    Espera en el .npz:
      - global_tensor_data: np.ndarray [N, C, R, R]
      - subject_ids: array-like [N]
      - (opcional) roi_names_in_order: [R]
      - (opcional) network_labels_in_order: [R]

    Devuelve:
      global_tensor, merged_df, roi_names_in_order, network_labels_in_order

    merged_df siempre contiene:
      - SubjectID (str)
      - tensor_idx (int) índice del sujeto en el tensor
      - columnas del metadata (si están disponibles)
    """
    tensor_path = Path(tensor_path)
    metadata_path = Path(metadata_path)

    logger.info(f"Cargando tensor global desde: {tensor_path}")
    if not tensor_path.exists():
        logger.error(f"Archivo de tensor global NO encontrado: {tensor_path}")
        return None, None, None, None

    try:
        data_npz = np.load(tensor_path, allow_pickle=True)

        if "global_tensor_data" not in data_npz or "subject_ids" not in data_npz:
            logger.error(
                "El .npz no contiene claves requeridas: 'global_tensor_data' y/o 'subject_ids'."
            )
            return None, None, None, None

        global_tensor = data_npz["global_tensor_data"]
        subject_ids_tensor = np.asarray(data_npz["subject_ids"]).astype(str)

        roi_names_in_order = data_npz.get("roi_names_in_order", None)
        network_labels_in_order = data_npz.get("network_labels_in_order", None)

        if roi_names_in_order is not None:
            roi_names_in_order = np.asarray(roi_names_in_order).astype(str).tolist()
            logger.info(f"Se leyeron {len(roi_names_in_order)} ROIs de 'roi_names_in_order'.")
        else:
            logger.warning("Key 'roi_names_in_order' NO encontrada en el .npz.")

        if network_labels_in_order is not None:
            network_labels_in_order = np.asarray(network_labels_in_order).astype(str).tolist()
            logger.info(
                f"Se leyeron {len(network_labels_in_order)} etiquetas de red de 'network_labels_in_order'."
            )
        else:
            logger.warning("Key 'network_labels_in_order' NO encontrada en el .npz.")

        logger.info(f"Tensor global cargado. Forma: {getattr(global_tensor, 'shape', None)}")

    except Exception as e:
        logger.error(f"Error cargando tensor global: {e}")
        return None, None, None, None

    logger.info(f"Cargando metadatos desde: {metadata_path}")
    if not metadata_path.exists():
        logger.error(f"Archivo de metadatos NO encontrado: {metadata_path}")
        return None, None, None, None

    try:
        metadata_df = pd.read_csv(metadata_path)

        if "SubjectID" not in metadata_df.columns:
            logger.error("El CSV de metadatos NO contiene la columna requerida: 'SubjectID'.")
            return None, None, None, None

        # Limpieza estándar de IDs
        metadata_df["SubjectID"] = metadata_df["SubjectID"].astype(str).str.strip()

        # Seguridad: evitar explosión del merge por SubjectID duplicados (cartesian product).
        # Esto mantiene el mapeo 1:1 tensor_idx -> SubjectID.
        dup_mask = metadata_df["SubjectID"].duplicated(keep=False)
        if dup_mask.any():
            n_dup = int(dup_mask.sum())
            logger.warning(
                f"Metadatos con SubjectID duplicados (n={n_dup}). "
                f"Se conservará la primera ocurrencia por SubjectID para mantener alineación con el tensor."
            )
            metadata_df = metadata_df.drop_duplicates(subset=["SubjectID"], keep="first")

        logger.info(f"Metadatos cargados. Forma: {metadata_df.shape}")

    except Exception as e:
        logger.error(f"Error cargando metadatos: {e}")
        return None, None, None, None

    # Merge para alinear: preservamos el orden del tensor, asignando tensor_idx 0..N-1
    tensor_df = pd.DataFrame({"SubjectID": subject_ids_tensor})
    tensor_df["SubjectID"] = tensor_df["SubjectID"].astype(str).str.strip()
    tensor_df["tensor_idx"] = np.arange(len(subject_ids_tensor), dtype=int)

    merged_df = pd.merge(tensor_df, metadata_df, on="SubjectID", how="left", validate="one_to_one")

    # Warn por faltantes: buscamos filas donde TODAS las columnas de metadata estén NaN.
    meta_cols = [c for c in merged_df.columns if c not in ("SubjectID", "tensor_idx")]
    if meta_cols:
        n_missing = int(merged_df[meta_cols].isna().all(axis=1).sum())
        if n_missing > 0:
            logger.warning(
                f"Algunos SubjectIDs del tensor no se encontraron en metadatos: {n_missing}/{len(merged_df)}."
            )
    else:
        logger.warning("No se detectaron columnas adicionales en metadatos tras el merge.")

    return global_tensor, merged_df, roi_names_in_order, network_labels_in_order


def normalize_inter_channel_fold(
    data_tensor: np.ndarray,
    train_indices_in_fold: np.ndarray,
    mode: str = "zscore_offdiag",
    selected_channel_original_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Normalización inter-canal por fold, evitando leakage:
    - Calcula estadísticos (mean/std o min/max) SOLO sobre train_indices_in_fold
    - Aplica la transformación a TODOS los sujetos del tensor provisto
      (para que luego puedas slice-ar train/dev/test ya normalizados con coherencia)

    Args:
      data_tensor: np.ndarray [N, C, R, R]
      train_indices_in_fold: índices (locales al data_tensor) usados para estimar params
      mode: 'zscore_offdiag' o 'minmax_offdiag'
      selected_channel_original_names: nombres humanos de canales (len=C) para logs/params

    Returns:
      normalized_tensor_fold: np.ndarray [N, C, R, R]
      norm_params_per_channel_list: List[Dict] con params por canal (serializable)
    """
    if data_tensor is None or not isinstance(data_tensor, np.ndarray):
        raise ValueError("data_tensor debe ser un np.ndarray no nulo.")
    if data_tensor.ndim != 4:
        raise ValueError(f"data_tensor debe tener ndim=4 [N,C,R,R]. Recibido: ndim={data_tensor.ndim}")
    n_subj, n_chan, n_rois, n_rois2 = data_tensor.shape
    if n_rois != n_rois2:
        raise ValueError(f"Las matrices no son cuadradas: {n_rois}x{n_rois2}")

    train_idx = np.asarray(train_indices_in_fold, dtype=int)
    if train_idx.size == 0:
        logger.warning(
            "train_indices_in_fold está vacío. "
            "No se pueden estimar parámetros de normalización. Se marcará 'no_scale' para todos los canales."
        )

    logger.info(
        f"Aplicando normalización inter-canal (modo: {mode}) sobre {n_chan} canales. "
        f"Train subjects para params: {int(train_idx.size)}."
    )

    normalized_tensor_fold = data_tensor.copy()
    norm_params_per_channel_list: List[Dict[str, float]] = []
    off_diag_mask = ~np.eye(n_rois, dtype=bool)

    for c_idx in range(n_chan):
        ch_name = (
            selected_channel_original_names[c_idx]
            if selected_channel_original_names and c_idx < len(selected_channel_original_names)
            else f"Channel_{c_idx}"
        )

        params: Dict[str, Any] = {"mode": mode, "original_name": ch_name}
        use_fixed_params = False

        # -------------------------
        # 1) Resolver params (train-only)
        # -------------------------
        if mode == "minmax_offdiag" and ch_name in FIXED_MINMAX_PARAMS_PER_CHANNEL:
            fixed_p = FIXED_MINMAX_PARAMS_PER_CHANNEL[ch_name]
            params.update({"min": float(fixed_p["min"]), "max": float(fixed_p["max"])})
            use_fixed_params = True
            logger.info(
                f"Canal '{ch_name}': usando MinMax fijo "
                f"(min={params['min']:.4f}, max={params['max']:.4f})."
            )

        if not use_fixed_params:
            if train_idx.size == 0:
                params.update({"no_scale": True, "mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0})
            else:
                channel_data_train = data_tensor[train_idx, c_idx, :, :]
                all_off_diag_train_values = channel_data_train[:, off_diag_mask].reshape(-1)

                if all_off_diag_train_values.size == 0:
                    logger.warning(
                        f"Canal '{ch_name}': sin valores off-diagonal en train. No se escala."
                    )
                    params.update({"no_scale": True, "mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0})
                elif mode == "zscore_offdiag":
                    mean_val = float(np.mean(all_off_diag_train_values))
                    std_val = float(np.std(all_off_diag_train_values))
                    if std_val <= 1e-9:
                        logger.warning(
                            f"Canal '{ch_name}': STD muy bajo ({std_val:.2e}). "
                            "Se usará std=1.0. # TODO: monitor zero-variance channels"
                        )
                        std_val = 1.0
                    params.update({"mean": mean_val, "std": std_val})
                elif mode == "minmax_offdiag":
                    min_val = float(np.min(all_off_diag_train_values))
                    max_val = float(np.max(all_off_diag_train_values))
                    if (max_val - min_val) <= 1e-9:
                        logger.warning(
                            f"Canal '{ch_name}': rango (max-min) muy bajo ({(max_val - min_val):.2e})."
                        )
                    params.update({"min": min_val, "max": max_val})
                else:
                    raise ValueError(f"Modo de normalización desconocido: {mode}")

        norm_params_per_channel_list.append(params)

        # -------------------------
        # 2) Aplicar params a TODOS los sujetos
        # -------------------------
        if params.get("no_scale", False):
            _zero_diag_inplace(normalized_tensor_fold[:, c_idx, :, :])
            continue

        current_channel_all = data_tensor[:, c_idx, :, :]
        scaled_channel = current_channel_all.copy()

        if off_diag_mask.any():
            if mode == "zscore_offdiag":
                std = float(params.get("std", 1.0))
                mean = float(params.get("mean", 0.0))
                if std > 1e-9:
                    scaled_channel[:, off_diag_mask] = (current_channel_all[:, off_diag_mask] - mean) / std
            elif mode == "minmax_offdiag":
                mn = float(params.get("min", 0.0))
                mx = float(params.get("max", 1.0))
                rng = mx - mn
                if rng > 1e-9:
                    scaled_channel[:, off_diag_mask] = (current_channel_all[:, off_diag_mask] - mn) / rng
                else:
                    scaled_channel[:, off_diag_mask] = 0.0

        normalized_tensor_fold[:, c_idx, :, :] = scaled_channel
        _zero_diag_inplace(normalized_tensor_fold[:, c_idx, :, :])

        if not use_fixed_params and not params.get("no_scale", False):
            if mode == "zscore_offdiag":
                logger.info(
                    f"Canal '{ch_name}': off-diag zscore (train_params: "
                    f"mean={params.get('mean', 0.0):.3f}, std={params.get('std', 1.0):.3f})"
                )
            elif mode == "minmax_offdiag":
                logger.info(
                    f"Canal '{ch_name}': off-diag minmax (train_params: "
                    f"min={params.get('min', 0.0):.3f}, max={params.get('max', 1.0):.3f})"
                )

    return normalized_tensor_fold, norm_params_per_channel_list


def apply_normalization_params(
    data_tensor_subset: np.ndarray,
    norm_params_list: List[Dict[str, float]],
) -> np.ndarray:
    """
    Aplica parámetros de normalización pre-calculados a un subconjunto [N, C, R, R].

    Útil para:
      - normalizar train/dev/test del clasificador con params del fold VAE
      - inferencia externa (COVID) reproduciendo el preprocesamiento del fold

    Args:
      data_tensor_subset: np.ndarray [N, C, R, R]
      norm_params_list: lista de dicts (len=C), salida de normalize_inter_channel_fold()

    Returns:
      normalized_tensor_subset: np.ndarray [N, C, R, R]
    """
    if data_tensor_subset is None or not isinstance(data_tensor_subset, np.ndarray):
        raise ValueError("data_tensor_subset debe ser un np.ndarray no nulo.")
    if data_tensor_subset.ndim != 4:
        raise ValueError(
            f"data_tensor_subset debe tener ndim=4 [N,C,R,R]. Recibido: ndim={data_tensor_subset.ndim}"
        )

    n_subj, n_chan, n_rois, n_rois2 = data_tensor_subset.shape
    if n_rois != n_rois2:
        raise ValueError(f"Las matrices no son cuadradas: {n_rois}x{n_rois2}")

    if len(norm_params_list) != n_chan:
        raise ValueError(
            f"Mismatch en canales: data tiene {n_chan}, params provistos para {len(norm_params_list)}"
        )

    normalized_tensor_subset = data_tensor_subset.copy()
    off_diag_mask = ~np.eye(n_rois, dtype=bool)

    for c_idx in range(n_chan):
        params = norm_params_list[c_idx]
        mode = params.get("mode", "zscore_offdiag")

        if params.get("no_scale", False):
            _zero_diag_inplace(normalized_tensor_subset[:, c_idx, :, :])
            continue

        current = data_tensor_subset[:, c_idx, :, :]
        scaled = current.copy()

        if off_diag_mask.any():
            if mode == "zscore_offdiag":
                std = float(params.get("std", 1.0))
                mean = float(params.get("mean", 0.0))
                if std > 1e-9:
                    scaled[:, off_diag_mask] = (current[:, off_diag_mask] - mean) / std
            elif mode == "minmax_offdiag":
                mn = float(params.get("min", 0.0))
                mx = float(params.get("max", 1.0))
                rng = mx - mn
                if rng > 1e-9:
                    scaled[:, off_diag_mask] = (current[:, off_diag_mask] - mn) / rng
                else:
                    scaled[:, off_diag_mask] = 0.0
            else:
                raise ValueError(f"Modo de normalización desconocido en params: {mode}")

        normalized_tensor_subset[:, c_idx, :, :] = scaled
        _zero_diag_inplace(normalized_tensor_subset[:, c_idx, :, :])

    return normalized_tensor_subset


__all__ = [
    "FIXED_MINMAX_PARAMS_PER_CHANNEL",
    "_zero_diag_inplace",
    "load_data",
    "normalize_inter_channel_fold",
    "apply_normalization_params",
]
