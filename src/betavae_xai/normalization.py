# -*- coding: utf-8 -*-
"""
betavae_xai/normalization.py

Funciones para la normalización de tensores de conectividad (fMRI)
dentro de los folds de validación cruzada.
"""
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Configuración global para normalización fija si es necesaria
FIXED_MINMAX_PARAMS_PER_CHANNEL = {}

def normalize_inter_channel_fold(
    data_tensor: np.ndarray, 
    train_indices_in_fold: np.ndarray, 
    mode: str = 'zscore_offdiag',
    selected_channel_original_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Calcula parámetros de normalización (mean/std o min/max) basándose SOLO
    en los índices de entrenamiento (train_indices_in_fold) y aplica la transformación
    a todo el tensor.
    """
    _, num_selected_channels, num_rois, _ = data_tensor.shape
    logger.info(f"Aplicando normalización inter-canal (modo: {mode}) sobre {num_selected_channels} canales seleccionados.")
    logger.info(f"Parámetros de normalización se calcularán usando {len(train_indices_in_fold)} sujetos de entrenamiento.")
    
    normalized_tensor_fold = data_tensor.copy()
    norm_params_per_channel_list = []
    off_diag_mask = ~np.eye(num_rois, dtype=bool)

    for c_idx_selected in range(num_selected_channels):
        current_channel_original_name = selected_channel_original_names[c_idx_selected] if selected_channel_original_names and c_idx_selected < len(selected_channel_original_names) else f"Channel_{c_idx_selected}"
        params = {'mode': mode, 'original_name': current_channel_original_name}
        use_fixed_params = False

        if mode == 'minmax_offdiag' and current_channel_original_name in FIXED_MINMAX_PARAMS_PER_CHANNEL:
            fixed_p = FIXED_MINMAX_PARAMS_PER_CHANNEL[current_channel_original_name]
            params.update({'min': fixed_p['min'], 'max': fixed_p['max']})
            use_fixed_params = True
            logger.info(f"Canal '{current_channel_original_name}': Usando MinMax fijo (min={params['min']:.4f}, max={params['max']:.4f}).")

        if not use_fixed_params:
            channel_data_train_for_norm_params = data_tensor[train_indices_in_fold, c_idx_selected, :, :]
            all_off_diag_train_values = channel_data_train_for_norm_params[:, off_diag_mask].flatten()

            if all_off_diag_train_values.size == 0:
                logger.warning(f"Canal '{current_channel_original_name}': No hay elementos fuera de la diagonal en el training set. No se escala.")
                params.update({'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0, 'no_scale': True})
            elif mode == 'zscore_offdiag':
                mean_val = np.mean(all_off_diag_train_values)
                std_val = np.std(all_off_diag_train_values)
                params.update({'mean': mean_val, 'std': std_val if std_val > 1e-9 else 1.0})
                if std_val <= 1e-9: logger.warning(f"Canal '{current_channel_original_name}': STD muy bajo ({std_val:.2e}). Usando STD=1.")
            elif mode == 'minmax_offdiag':
                min_val = np.min(all_off_diag_train_values)
                max_val = np.max(all_off_diag_train_values)
                params.update({'min': min_val, 'max': max_val})
                if (max_val - min_val) <= 1e-9: logger.warning(f"Canal '{current_channel_original_name}': Rango (max-min) muy bajo ({(max_val - min_val):.2e}).")
            else:
                raise ValueError(f"Modo de normalización desconocido: {mode}")
        
        norm_params_per_channel_list.append(params)

        if not params.get('no_scale', False):
            current_channel_data_all_subjects = data_tensor[:, c_idx_selected, :, :]
            scaled_channel_data = current_channel_data_all_subjects.copy()
            if off_diag_mask.any():
                if mode == 'zscore_offdiag':
                    if params['std'] > 1e-9:
                        scaled_channel_data[:, off_diag_mask] = (current_channel_data_all_subjects[:, off_diag_mask] - params['mean']) / params['std']
                elif mode == 'minmax_offdiag':
                    range_val = params.get('max', 1.0) - params.get('min', 0.0)
                    if range_val > 1e-9: 
                        scaled_channel_data[:, off_diag_mask] = (current_channel_data_all_subjects[:, off_diag_mask] - params['min']) / range_val
                    else: 
                        scaled_channel_data[:, off_diag_mask] = 0.0 
            normalized_tensor_fold[:, c_idx_selected, :, :] = scaled_channel_data
            normalized_tensor_fold[:, c_idx_selected, ~off_diag_mask] = 0.0
            if not use_fixed_params:
                log_msg_params = f"Canal '{current_channel_original_name}': Off-diag {mode} (train_params: "
                if mode == 'zscore_offdiag': log_msg_params += f"mean={params['mean']:.3f}, std={params['std']:.3f})"
                elif mode == 'minmax_offdiag': log_msg_params += f"min={params['min']:.3f}, max={params['max']:.3f})"
                logger.info(log_msg_params)
    return normalized_tensor_fold, norm_params_per_channel_list


def apply_normalization_params(
    data_tensor_subset: np.ndarray, 
    norm_params_per_channel_list: List[Dict[str, float]]
) -> np.ndarray:
    """
    Aplica parámetros de normalización previamente calculados a un nuevo subconjunto de datos
    (ej. conjunto de test o validación).
    """
    _, num_selected_channels, num_rois, _ = data_tensor_subset.shape
    normalized_tensor_subset = data_tensor_subset.copy()
    off_diag_mask = ~np.eye(num_rois, dtype=bool)

    if len(norm_params_per_channel_list) != num_selected_channels:
        raise ValueError(f"Mismatch in number of channels for normalization: data has {num_selected_channels}, params provided for {len(norm_params_per_channel_list)}")

    for c_idx_selected in range(num_selected_channels):
        params = norm_params_per_channel_list[c_idx_selected]
        mode = params.get('mode', 'zscore_offdiag') 
        if params.get('no_scale', False):
            continue
        current_channel_data = data_tensor_subset[:, c_idx_selected, :, :]
        scaled_channel_data_subset = current_channel_data.copy()
        if off_diag_mask.any():
            if mode == 'zscore_offdiag':
                if params['std'] > 1e-9:
                    scaled_channel_data_subset[:, off_diag_mask] = (current_channel_data[:, off_diag_mask] - params['mean']) / params['std']
            elif mode == 'minmax_offdiag':
                range_val = params.get('max', 1.0) - params.get('min', 0.0)
                if range_val > 1e-9:
                    scaled_channel_data_subset[:, off_diag_mask] = (current_channel_data[:, off_diag_mask] - params['min']) / range_val
                else:
                    scaled_channel_data_subset[:, off_diag_mask] = 0.0 
        normalized_tensor_subset[:, c_idx_selected, :, :] = scaled_channel_data_subset
        normalized_tensor_subset[:, c_idx_selected, ~off_diag_mask] = 0.0
    return normalized_tensor_subset