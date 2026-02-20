"""
composite_edge_shap_v2.py

Edge-level SHAP (v2) for the composite predictor:
edges -> tensor -> VAE.encode -> latent+metadata -> classifier pipeline -> score.

This module is intentionally additive and does not modify v1 behavior/files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from betavae_xai.models.convolutional_vae import ConvolutionalVAE
from betavae_xai.interpretability.composite_edge_shap import (
    compute_frozen_meta_values as _compute_frozen_meta_values,
    compute_train_edge_median as _compute_train_edge_median,
    make_edge_feature_names as _make_edge_feature_names_upper,
    make_edge_index as _make_edge_index_upper,
    make_edge_mapping_df as _make_edge_mapping_df_upper,
    make_edge_predict_fn as _make_edge_predict_fn_upper,
    reconstruct_tensor_from_edges as _reconstruct_tensor_from_edges_upper,
    select_top_edges as _select_top_edges,
    vectorize_tensor_to_edges as _vectorize_tensor_to_edges_upper,
)


log = logging.getLogger("interpret_edges_v2")

_ROI_IDX_CANDIDATES = ["ROI_TensorIdx", "Original_Index_0_N"]


# ---------------------------------------------------------------------------
# Base helpers (kept local to avoid touching old modules)
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def build_vae(vae_kwargs: Dict[str, Any], state_dict_path: Path, device: torch.device) -> ConvolutionalVAE:
    if not state_dict_path.exists():
        raise FileNotFoundError(f"VAE state_dict not found: {state_dict_path}")
    vae = ConvolutionalVAE(**vae_kwargs).to(device)
    sd = torch.load(state_dict_path, map_location=device)
    vae.load_state_dict(clean_state_dict(sd))
    vae.eval()
    return vae


def vae_kwargs_from_args(args: argparse.Namespace, image_size: int) -> Dict[str, Any]:
    return dict(
        input_channels=len(args.channels_to_use),
        latent_dim=args.latent_dim,
        image_size=image_size,
        dropout_rate=args.dropout_rate_vae,
        use_layernorm_fc=getattr(args, "use_layernorm_vae_fc", False),
        num_conv_layers_encoder=args.num_conv_layers_encoder,
        decoder_type=args.decoder_type,
        intermediate_fc_dim_config=args.intermediate_fc_dim_vae,
        final_activation=args.vae_final_activation,
        num_groups=args.gn_num_groups,
    )


def apply_normalization_params(
    data_tensor_subset: np.ndarray,
    norm_params_per_channel_list: List[Dict[str, float]],
) -> np.ndarray:
    """Apply per-channel fold normalization (off-diagonal)."""
    num_subjects, num_selected_channels, num_rois, _ = data_tensor_subset.shape
    _ = num_subjects  # keep for readability
    normalized = data_tensor_subset.copy()
    off_diag_mask = ~np.eye(num_rois, dtype=bool)
    if len(norm_params_per_channel_list) != num_selected_channels:
        raise ValueError(
            "Mismatch: #channels in tensor != #channels in normalization params. "
            f"tensor={num_selected_channels} params={len(norm_params_per_channel_list)}"
        )
    for c_idx, params in enumerate(norm_params_per_channel_list):
        mode = params.get("mode", "zscore_offdiag")
        if params.get("no_scale", False):
            continue
        ch = data_tensor_subset[:, c_idx, :, :]
        scaled = ch.copy()
        if off_diag_mask.any():
            if mode == "zscore_offdiag":
                std = float(params.get("std", 1.0))
                mean = float(params.get("mean", 0.0))
                if std > 1e-9:
                    scaled[:, off_diag_mask] = (ch[:, off_diag_mask] - mean) / std
            elif mode == "minmax_offdiag":
                mn = float(params.get("min", 0.0))
                mx = float(params.get("max", 1.0))
                rng = mx - mn
                if rng > 1e-9:
                    scaled[:, off_diag_mask] = (ch[:, off_diag_mask] - mn) / rng
                else:
                    scaled[:, off_diag_mask] = 0.0
        normalized[:, c_idx, :, :] = scaled
    return normalized


def load_global_and_merge(global_tensor_path: Path, metadata_path: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    if not global_tensor_path.exists():
        raise FileNotFoundError(f"GLOBAL_TENSOR not found: {global_tensor_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    npz = np.load(global_tensor_path)
    if "global_tensor_data" not in npz or "subject_ids" not in npz:
        raise KeyError("NPZ must contain keys: 'global_tensor_data' and 'subject_ids'.")
    tensor_all = npz["global_tensor_data"]
    subject_ids = npz["subject_ids"].astype(str)
    meta = pd.read_csv(metadata_path)
    if "SubjectID" not in meta.columns:
        raise KeyError("metadata CSV must contain 'SubjectID'.")
    meta = meta.copy()
    meta["SubjectID"] = meta["SubjectID"].astype(str).str.strip()
    tensor_df = pd.DataFrame({"SubjectID": subject_ids, "tensor_idx": np.arange(len(subject_ids))})
    merged = tensor_df.merge(meta, on="SubjectID", how="left")
    return tensor_all, merged


def subset_cnad(merged_df: pd.DataFrame) -> pd.DataFrame:
    if "ResearchGroup_Mapped" not in merged_df.columns:
        raise KeyError("Merged dataframe must contain 'ResearchGroup_Mapped'.")
    return merged_df[merged_df["ResearchGroup_Mapped"].isin(["CN", "AD"])].reset_index(drop=True)


def load_label_info(fold_dir: Path) -> Dict[str, Any]:
    p = fold_dir / "label_mapping.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    log.warning("label_mapping.json not found. Falling back to CN=0 / AD=1.")
    return {"label_mapping": {"CN": 0, "AD": 1}, "positive_label_name": "AD", "positive_label_int": 1}


def load_feature_columns(
    fold_dir: Path,
    latent_dim: int,
    metadata_features: Optional[Sequence[str]],
) -> List[str]:
    fc_path = fold_dir / "feature_columns.json"
    if fc_path.exists():
        with open(fc_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        cols = info.get("final_feature_columns")
        if isinstance(cols, list) and len(cols) > 0:
            return [str(c) for c in cols]
        log.warning("feature_columns.json found but empty/invalid; using fallback order.")
    return [f"latent_{i}" for i in range(latent_dim)] + list(metadata_features or [])


def _coerce_roi_names(raw: Any, source: Path) -> Optional[List[str]]:
    value = raw
    if isinstance(raw, dict):
        if "roi_names_in_order" in raw:
            value = raw["roi_names_in_order"]
        else:
            return None
    if isinstance(value, np.ndarray):
        names = value.astype(str).tolist()
    elif isinstance(value, (list, tuple, pd.Series)):
        names = [str(x) for x in value]
    else:
        return None
    if len(names) == 0:
        raise ValueError(f"ROI names in {source} are empty.")
    return names


def load_roi_names_from_path(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".joblib":
        obj = joblib.load(path)
        names = _coerce_roi_names(obj, path)
        if names is None:
            raise ValueError(
                f"Unsupported ROI format in {path}. Expected list/array or dict['roi_names_in_order']."
            )
        return names
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True).astype(str).tolist()
    # txt/csv fallback: first column
    return pd.read_csv(path, header=None).iloc[:, 0].astype(str).tolist()


def load_roi_names_robust(run_dir: Path, roi_order_path: Optional[Union[str, Path]] = None) -> List[str]:
    """
    Robust ROI-order loading priority:
      1) run_dir/roi_order_131.joblib (list/array)
      2) run_dir/roi_info_from_tensor.joblib (dict with key 'roi_names_in_order')
      3) --roi_order_path (optional)
    """
    p_order = run_dir / "roi_order_131.joblib"
    if p_order.exists():
        obj = joblib.load(p_order)
        names = _coerce_roi_names(obj, p_order)
        if names is not None:
            log.info(f"[ROI] Using ROI order from {p_order}.")
            return names
        log.warning(f"[ROI] {p_order} exists but does not contain a valid ROI list.")

    p_info = run_dir / "roi_info_from_tensor.joblib"
    if p_info.exists():
        obj = joblib.load(p_info)
        names = _coerce_roi_names(obj, p_info)
        if names is not None:
            log.info(f"[ROI] Using ROI order from {p_info} (dict fallback).")
            return names
        log.warning(
            f"[ROI] {p_info} exists but key 'roi_names_in_order' is missing or invalid."
        )

    if roi_order_path is not None:
        p_ext = Path(roi_order_path)
        names = load_roi_names_from_path(p_ext)
        log.info(f"[ROI] Using ROI order from --roi_order_path={p_ext}.")
        return names

    raise FileNotFoundError(
        "Could not resolve ROI order. Expected one of: "
        f"{p_order}, {p_info} with key 'roi_names_in_order', or --roi_order_path."
    )


# ---------------------------------------------------------------------------
# Edge indexing / naming / reconstruction (v2)
# ---------------------------------------------------------------------------

def make_edge_index_v2(
    R: int,
    C: int,
    mode: str = "upper",
) -> Tuple[np.ndarray, np.ndarray, int, int, bool]:
    """
    Returns (rows, cols, edges_per_channel, total_edges, is_directed).
    """
    if mode == "upper":
        rows, cols, total_edges = _make_edge_index_upper(R, C)
        edges_per_channel = len(rows)
        return rows, cols, edges_per_channel, total_edges, False
    if mode == "full_offdiag":
        mask = ~np.eye(R, dtype=bool)
        rows, cols = np.where(mask)
        rows = rows.astype(np.int64)
        cols = cols.astype(np.int64)
        edges_per_channel = len(rows)
        total_edges = C * edges_per_channel
        return rows, cols, edges_per_channel, total_edges, True
    raise ValueError(f"Unknown edge_vector_mode: {mode}. Use 'upper' or 'full_offdiag'.")


def vectorize_tensor_to_edges_v2(
    tensor: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    mode: str = "upper",
) -> np.ndarray:
    if mode not in {"upper", "full_offdiag"}:
        raise ValueError(f"Unsupported edge_vector_mode: {mode}")
    # Wrapper: the v1 helper already vectorizes by explicit (rows, cols).
    return _vectorize_tensor_to_edges_upper(tensor, rows, cols)


def reconstruct_tensor_from_edges_v2(
    edge_vector: np.ndarray,
    R: int,
    C: int,
    rows: np.ndarray,
    cols: np.ndarray,
    mode: str = "upper",
) -> np.ndarray:
    if mode == "upper":
        return _reconstruct_tensor_from_edges_upper(edge_vector, R, C, rows, cols)
    if mode != "full_offdiag":
        raise ValueError(f"Unsupported edge_vector_mode: {mode}")

    x = np.asarray(edge_vector)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    N = x.shape[0]
    epc = len(rows)
    expected = C * epc
    if x.shape[1] != expected:
        raise ValueError(
            f"edge_vector has {x.shape[1]} features but expected {expected} (=C*E)."
        )
    tensor = np.zeros((N, C, R, R), dtype=np.float32)
    for c in range(C):
        vals = x[:, c * epc:(c + 1) * epc]
        tensor[:, c, rows, cols] = vals
        tensor[:, c, np.arange(R), np.arange(R)] = 0.0
    return tensor


def make_edge_feature_names_v2(
    roi_names: Sequence[str],
    channels_used: Sequence[int],
    channel_names: Optional[Sequence[str]],
    rows: np.ndarray,
    cols: np.ndarray,
    mode: str = "upper",
) -> List[str]:
    if mode == "upper":
        return _make_edge_feature_names_upper(
            list(roi_names),
            list(channels_used),
            list(channel_names) if channel_names is not None else None,
        )

    names: List[str] = []
    for ci, ch in enumerate(channels_used):
        ch_label = (
            str(channel_names[ci])
            if channel_names is not None and ci < len(channel_names)
            else f"Ch{ch}"
        )
        for ri, rj in zip(rows, cols):
            names.append(f"{ch_label}__{roi_names[int(ri)]}->{roi_names[int(rj)]}")
    return names


def make_edge_mapping_df_v2(
    selected_indices: np.ndarray,
    roi_names: Sequence[str],
    channels_used: Sequence[int],
    channel_names: Optional[Sequence[str]],
    rows: np.ndarray,
    cols: np.ndarray,
    edges_per_channel: int,
    scores: Optional[np.ndarray] = None,
    mode: str = "upper",
) -> pd.DataFrame:
    if mode == "upper":
        df = _make_edge_mapping_df_upper(
            selected_indices=np.asarray(selected_indices, dtype=int),
            roi_names=list(roi_names),
            channels_used=list(channels_used),
            channel_names=list(channel_names) if channel_names is not None else None,
            triu_rows=rows,
            triu_cols=cols,
            edges_per_channel=edges_per_channel,
            scores=scores,
        ).copy()
        df = df.rename(
            columns={
                "roi_i_idx": "i",
                "roi_j_idx": "j",
                "roi_i_name": "roi_i",
                "roi_j_name": "roi_j",
            }
        )
        return df

    all_names = make_edge_feature_names_v2(
        roi_names=roi_names,
        channels_used=channels_used,
        channel_names=channel_names,
        rows=rows,
        cols=cols,
        mode=mode,
    )
    out_rows: List[Dict[str, Any]] = []
    for rank, flat_idx in enumerate(np.asarray(selected_indices, dtype=int)):
        ch_local = int(flat_idx // edges_per_channel)
        edge_local = int(flat_idx % edges_per_channel)
        i = int(rows[edge_local])
        j = int(cols[edge_local])
        ch_orig = int(channels_used[ch_local]) if ch_local < len(channels_used) else ch_local
        ch_name = (
            str(channel_names[ch_local])
            if channel_names is not None and ch_local < len(channel_names)
            else f"Ch{ch_orig}"
        )
        row: Dict[str, Any] = {
            "edge_name": all_names[flat_idx],
            "flat_index": flat_idx,
            "channel": ch_orig,
            "channel_name": ch_name,
            "i": i,
            "j": j,
            "roi_i": str(roi_names[i]),
            "roi_j": str(roi_names[j]),
            "selection_rank": rank + 1,
        }
        if scores is not None:
            row["selection_score"] = float(scores[flat_idx])
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def maybe_annotate_mapping(mapping_df: pd.DataFrame, roi_annotation_csv: Optional[Union[str, Path]]) -> pd.DataFrame:
    if roi_annotation_csv is None:
        return mapping_df
    p = Path(roi_annotation_csv)
    if not p.exists():
        raise FileNotFoundError(f"--roi_annotation_csv does not exist: {p}")
    ann = pd.read_csv(p)
    idx_col = next((c for c in _ROI_IDX_CANDIDATES if c in ann.columns), None)
    if idx_col is None:
        log.warning(
            "[ROI_ANNOT] No ROI index column found in annotation CSV. "
            f"Expected one of {_ROI_IDX_CANDIDATES}. Skipping annotation merge."
        )
        return mapping_df
    ann = ann.copy()
    ann[idx_col] = pd.to_numeric(ann[idx_col], errors="coerce")
    ann = ann.dropna(subset=[idx_col]).drop_duplicates(subset=[idx_col])
    ann[idx_col] = ann[idx_col].astype(int)
    ann = ann.set_index(idx_col)
    cols = [c for c in ann.columns]
    if len(cols) == 0:
        return mapping_df
    out = mapping_df.copy()
    out = out.join(ann[cols].add_prefix("roi_i_"), on="i")
    out = out.join(ann[cols].add_prefix("roi_j_"), on="j")
    return out


# ---------------------------------------------------------------------------
# Predictor / validation
# ---------------------------------------------------------------------------

def _coerce_meta_series_to_numeric(series: pd.Series, col_name: str) -> np.ndarray:
    s = series.copy()
    if str(col_name).strip().lower() == "sex":
        if s.dtype == object:
            s = (
                s.astype(str)
                .str.strip()
                .str.lower()
                .map({"m": 0.0, "male": 0.0, "f": 1.0, "female": 1.0})
            )
    s = pd.to_numeric(s, errors="coerce")
    return s.to_numpy(dtype=np.float64)


def _encode_tensor_to_latent(
    vae: ConvolutionalVAE,
    tensor_norm: np.ndarray,
    device: torch.device,
    latent_features_type: str,
) -> np.ndarray:
    with torch.no_grad():
        t = torch.from_numpy(np.asarray(tensor_norm, dtype=np.float32)).to(device)
        mu, logvar = vae.encode(t)
        if latent_features_type == "mu":
            lat = mu
        elif latent_features_type == "z":
            lat = vae.reparameterize(mu, logvar)
        else:
            raise ValueError(f"latent_features_type must be 'mu' or 'z', got: {latent_features_type}")
    return lat.detach().cpu().numpy()


def _build_raw_feature_df(
    lat_np: np.ndarray,
    feature_columns: Sequence[str],
    meta_cols: Sequence[str],
    meta_df: Optional[pd.DataFrame],
    meta_frozen_values: Optional[Dict[str, float]],
) -> pd.DataFrame:
    lat_cols = [f"latent_{i}" for i in range(lat_np.shape[1])]
    X_raw = pd.DataFrame(lat_np, columns=lat_cols)

    frozen = meta_frozen_values or {}
    for col in meta_cols:
        if meta_df is not None and col in meta_df.columns:
            vals = _coerce_meta_series_to_numeric(meta_df[col], col)
            if np.isnan(vals).any():
                fill = frozen.get(col, 0.0)
                vals = np.where(np.isnan(vals), float(fill), vals)
            X_raw[col] = vals
        else:
            X_raw[col] = float(frozen.get(col, 0.0))

    missing = [c for c in feature_columns if c not in X_raw.columns]
    for c in missing:
        X_raw[c] = 0.0
    X_raw = X_raw[list(feature_columns)]
    return X_raw


def _predict_positive(pipe: Any, X_raw: pd.DataFrame, pos_idx: int) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        p = pipe.predict_proba(X_raw)
        p = np.asarray(p, dtype=np.float64)
        if p.ndim == 1:
            return p
        if pos_idx >= p.shape[1]:
            raise IndexError(f"pos_idx={pos_idx} out of bounds for predict_proba with shape={p.shape}.")
        return p[:, pos_idx]

    if hasattr(pipe, "decision_function"):
        margin = np.asarray(pipe.decision_function(X_raw), dtype=np.float64)
        if margin.ndim == 2:
            if pos_idx >= margin.shape[1]:
                raise IndexError(
                    f"pos_idx={pos_idx} out of bounds for decision_function with shape={margin.shape}."
                )
            margin = margin[:, pos_idx]
        # Fallback to calibrated-like probability from margin.
        return 1.0 / (1.0 + np.exp(-margin))

    return np.asarray(pipe.predict(X_raw), dtype=np.float64)


def _apply_link(score: np.ndarray, shap_link: str) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    if shap_link == "identity":
        return score
    if shap_link == "logit":
        eps = 1e-6
        p = np.clip(score, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))
    raise ValueError(f"Unknown shap_link: {shap_link}")


def resolve_positive_index(pipe: Any, label_info: Dict[str, Any]) -> int:
    pos_int = int(label_info.get("positive_label_int", 1))
    candidates: List[Any] = [pipe]

    if hasattr(pipe, "named_steps"):
        model_step = pipe.named_steps.get("model")
        if model_step is not None:
            candidates.append(model_step)

    if hasattr(pipe, "calibrated_classifiers_"):
        cc = pipe.calibrated_classifiers_[0]
        candidates.append(cc)
        inner = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if inner is not None:
            candidates.append(inner)
            if hasattr(inner, "named_steps"):
                inner_model = inner.named_steps.get("model")
                if inner_model is not None:
                    candidates.append(inner_model)

    classes: Optional[List[Any]] = None
    for obj in candidates:
        cls = getattr(obj, "classes_", None)
        if cls is not None:
            classes = list(cls)
            break

    if not classes:
        return 1

    if pos_int in classes:
        return int(classes.index(pos_int))
    pos_str = str(pos_int)
    classes_str = [str(c) for c in classes]
    if pos_str in classes_str:
        return int(classes_str.index(pos_str))
    return 1 if len(classes) > 1 else 0


def make_edge_predict_fn_v2(
    *,
    vae: ConvolutionalVAE,
    pipe: Any,
    meta_cols: List[str],
    meta_frozen_values: Dict[str, float],
    feature_columns: List[str],
    selected_edge_indices: np.ndarray,
    all_edges_template: np.ndarray,
    R: int,
    C: int,
    edge_rows: np.ndarray,
    edge_cols: np.ndarray,
    edge_vector_mode: str,
    device: torch.device,
    latent_features_type: str = "mu",
    pos_idx: int = 1,
    shap_link: str = "identity",
) -> Callable[[np.ndarray], np.ndarray]:
    """
    selected edges (N,K) -> model output score (proba or logit) for positive class.
    """
    if edge_vector_mode == "upper":
        return _make_edge_predict_fn_upper(
            vae=vae,
            pipe=pipe,
            meta_cols=meta_cols,
            meta_frozen_values=meta_frozen_values,
            feature_columns=feature_columns,
            selected_edge_indices=np.asarray(selected_edge_indices, dtype=int),
            all_edges_template=np.asarray(all_edges_template, dtype=np.float32),
            R=R,
            C=C,
            triu_rows=edge_rows,
            triu_cols=edge_cols,
            device=device,
            latent_features_type=latent_features_type,
            pos_idx=pos_idx,
            shap_link=shap_link,
        )

    if edge_vector_mode != "full_offdiag":
        raise ValueError(f"Unsupported edge_vector_mode: {edge_vector_mode}")

    template = np.asarray(all_edges_template, dtype=np.float32).copy()
    sel_idx = np.asarray(selected_edge_indices, dtype=np.int64)
    meta_cols_local = list(meta_cols)
    feature_cols_local = list(feature_columns)
    frozen_local = {k: float(v) for k, v in (meta_frozen_values or {}).items()}

    def predict_fn(X_selected: np.ndarray) -> np.ndarray:
        X_selected = np.asarray(X_selected, dtype=np.float64)
        if X_selected.ndim == 1:
            X_selected = X_selected[np.newaxis, :]
        N = X_selected.shape[0]
        if X_selected.shape[1] != len(sel_idx):
            raise ValueError(
                f"predict_fn expects K={len(sel_idx)} selected edges, got {X_selected.shape[1]}."
            )

        full_edges = np.tile(template, (N, 1))
        full_edges[:, sel_idx] = X_selected.astype(np.float32)
        tensor = reconstruct_tensor_from_edges_v2(
            edge_vector=full_edges,
            R=R,
            C=C,
            rows=edge_rows,
            cols=edge_cols,
            mode=edge_vector_mode,
        )
        lat_np = _encode_tensor_to_latent(
            vae=vae,
            tensor_norm=tensor,
            device=device,
            latent_features_type=latent_features_type,
        )
        X_raw = _build_raw_feature_df(
            lat_np=lat_np,
            feature_columns=feature_cols_local,
            meta_cols=meta_cols_local,
            meta_df=None,
            meta_frozen_values=frozen_local,
        )
        score = _predict_positive(pipe, X_raw, pos_idx)
        return _apply_link(score, shap_link)

    return predict_fn


def validate_prediction_identity_v2(
    *,
    vae: ConvolutionalVAE,
    pipe: Any,
    tensor_test_norm: np.ndarray,
    test_df: pd.DataFrame,
    meta_cols: List[str],
    feature_columns: List[str],
    edge_rows: np.ndarray,
    edge_cols: np.ndarray,
    edge_vector_mode: str,
    device: torch.device,
    latent_features_type: str,
    pos_idx: int,
    n_check: int = 10,
    atol_strict: float = 1e-6,
    atol_relaxed: float = 1e-5,
) -> Tuple[bool, float, float]:
    """
    Path A: tensor -> VAE -> X_raw -> pipe.predict(...)
    Path B: tensor -> edge vectorize -> reconstruct -> VAE -> X_raw -> pipe.predict(...)
    """
    n = int(min(n_check, tensor_test_norm.shape[0]))
    if n <= 0:
        raise ValueError("No test samples available for prediction identity check.")

    tens_sub = tensor_test_norm[:n]
    test_sub = test_df.iloc[:n].copy()
    R = tens_sub.shape[2]
    C = tens_sub.shape[1]

    lat_a = _encode_tensor_to_latent(
        vae=vae,
        tensor_norm=tens_sub,
        device=device,
        latent_features_type=latent_features_type,
    )
    X_a = _build_raw_feature_df(
        lat_np=lat_a,
        feature_columns=feature_columns,
        meta_cols=meta_cols,
        meta_df=test_sub,
        meta_frozen_values=None,
    )
    pred_a = _predict_positive(pipe, X_a, pos_idx)

    edges = vectorize_tensor_to_edges_v2(
        tensor=tens_sub,
        rows=edge_rows,
        cols=edge_cols,
        mode=edge_vector_mode,
    )
    tens_recon = reconstruct_tensor_from_edges_v2(
        edge_vector=edges,
        R=R,
        C=C,
        rows=edge_rows,
        cols=edge_cols,
        mode=edge_vector_mode,
    )
    lat_b = _encode_tensor_to_latent(
        vae=vae,
        tensor_norm=tens_recon,
        device=device,
        latent_features_type=latent_features_type,
    )
    X_b = _build_raw_feature_df(
        lat_np=lat_b,
        feature_columns=feature_columns,
        meta_cols=meta_cols,
        meta_df=test_sub,
        meta_frozen_values=None,
    )
    pred_b = _predict_positive(pipe, X_b, pos_idx)

    max_diff = float(np.max(np.abs(np.asarray(pred_a) - np.asarray(pred_b))))
    if max_diff <= atol_strict:
        log.info(
            "[VALIDATION_V2] Prediction identity PASSED (strict): "
            f"max|diff|={max_diff:.3e} <= {atol_strict:.1e}"
        )
        return True, max_diff, atol_strict
    if max_diff <= atol_relaxed:
        log.warning(
            "[VALIDATION_V2] Prediction identity PASSED (relaxed float tolerance): "
            f"max|diff|={max_diff:.3e} <= {atol_relaxed:.1e}"
        )
        return True, max_diff, atol_relaxed

    log.error(
        "[VALIDATION_V2] Prediction identity FAILED: "
        f"max|diff|={max_diff:.3e} > {atol_relaxed:.1e}"
    )
    return False, max_diff, atol_relaxed


# ---------------------------------------------------------------------------
# Selection / sampling / caching
# ---------------------------------------------------------------------------

def select_top_edges_v2(
    edges_train: np.ndarray,
    y_train: np.ndarray,
    *,
    K: int,
    method: str,
    seed: int,
    per_channel: bool,
    edges_per_channel: int,
    C: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    total_edges = edges_train.shape[1]
    if K <= 0:
        raise ValueError(f"edge_K must be >=1, got {K}.")

    if not per_channel:
        sel, scores, info = _select_top_edges(edges_train, y_train, K=K, method=method, seed=seed)
        meta = {"per_channel": False, "K_total": int(len(sel))}
        return np.asarray(sel, dtype=int), np.asarray(scores, dtype=np.float64), {"selector": info, **meta}

    # Per-channel semantics: K means K per channel (capped by edges_per_channel).
    k_per_ch = int(min(K, edges_per_channel))
    scores_full = np.zeros(total_edges, dtype=np.float64)
    selected_blocks: List[np.ndarray] = []
    selector_info: Dict[str, Any] = {}
    for c in range(C):
        start = c * edges_per_channel
        end = (c + 1) * edges_per_channel
        block = edges_train[:, start:end]
        sel_local, sc_local, info_local = _select_top_edges(
            block, y_train, K=k_per_ch, method=method, seed=seed
        )
        scores_full[start:end] = np.asarray(sc_local, dtype=np.float64)
        selected_blocks.append(np.asarray(sel_local, dtype=int) + start)
        selector_info[f"channel_{c}"] = info_local

    selected_idx = np.sort(np.concatenate(selected_blocks).astype(int))
    meta = {
        "per_channel": True,
        "K_per_channel": int(k_per_ch),
        "K_total": int(len(selected_idx)),
    }
    return selected_idx, scores_full, {"selector": selector_info, **meta}


def pick_bg_indices(
    cnad_df: pd.DataFrame,
    train_idx_in_cnad: np.ndarray,
    mode: str,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    if mode == "train":
        pool = np.asarray(train_idx_in_cnad, dtype=int)
    elif mode == "global":
        pool = np.arange(len(cnad_df), dtype=int)
    elif mode == "global_cn":
        pool = cnad_df.index[
            cnad_df["ResearchGroup_Mapped"].astype(str).isin(["CN"])
        ].to_numpy(dtype=int)
    else:
        raise ValueError(f"Unknown bg_mode: {mode}. Use train/global/global_cn.")

    if pool.size == 0:
        raise RuntimeError(f"Empty pool for bg_mode={mode}.")
    if sample_size is None or sample_size <= 0 or sample_size >= pool.size:
        return np.sort(pool)

    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(pool, size=int(sample_size), replace=False))


def sample_test_indices(n_total: int, n_keep: Optional[int], seed: int) -> np.ndarray:
    if n_total <= 0:
        raise ValueError("No test samples available.")
    if n_keep is None or n_keep >= n_total:
        return np.arange(n_total, dtype=int)
    if n_keep <= 0:
        raise ValueError(f"n_test_shap must be >=1 or None, got {n_keep}.")
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(np.arange(n_total), size=int(n_keep), replace=False))


def _channel_tag(channels: Sequence[int]) -> str:
    return "-".join(str(int(c)) for c in channels)


def _indices_hash(indices: np.ndarray) -> str:
    h = hashlib.sha1(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()
    return h[:12]


def _tag_suffix(tag: Optional[str]) -> str:
    if tag is None or str(tag).strip() == "":
        return ""
    return f"_{str(tag).strip()}"


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    return obj


def _labels_from_group(df: pd.DataFrame) -> np.ndarray:
    y = df["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1})
    if y.isna().any():
        bad = df.loc[y.isna(), "ResearchGroup_Mapped"].astype(str).unique().tolist()
        raise ValueError(f"Unexpected labels in ResearchGroup_Mapped: {bad}")
    return y.astype(int).to_numpy()


def _prepare_base_values_for_explanation(
    base_values_raw: Any,
    n_samples: int,
) -> Tuple[Union[float, np.ndarray], np.ndarray]:
    # pack representation + explanation vector representation
    if np.isscalar(base_values_raw):
        v = float(base_values_raw)
        return v, np.full(n_samples, v, dtype=np.float64)

    arr = np.asarray(base_values_raw, dtype=np.float64)
    if arr.ndim == 0:
        v = float(arr.item())
        return v, np.full(n_samples, v, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] == n_samples:
            return arr.copy(), arr.copy()
        # fallback: scalar median
        v = float(np.median(arr))
        return v, np.full(n_samples, v, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[0] == n_samples and arr.shape[1] >= 1:
            vec = arr[:, 0]
            return vec.copy(), vec.copy()
        v = float(np.median(arr))
        return v, np.full(n_samples, v, dtype=np.float64)

    v = float(np.median(arr))
    return v, np.full(n_samples, v, dtype=np.float64)


def _plot_shap_summary_v2(
    exp: shap.Explanation,
    out_dir: Path,
    fold: int,
    clf: str,
    tag: str,
) -> None:
    plt.figure(figsize=(10, 8))
    shap.plots.bar(exp, max_display=20, show=False)
    plt.title(f"SHAP Edges V2 - Global Importance (Fold {fold}, {clf.upper()})")
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_edges_v2_global_importance_bar{tag}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(exp, max_display=20, show=False)
    plt.title(f"SHAP Edges V2 - Beeswarm (Fold {fold}, {clf.upper()})")
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_edges_v2_summary_beeswarm{tag}.png", dpi=150)
    plt.close()

    if exp.values.shape[0] > 0:
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(exp[0], show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_edges_v2_waterfall_subject_0{tag}.png", dpi=150)
        plt.close()


# ---------------------------------------------------------------------------
# Main entrypoint used by scripts/interpret_edges_v2.py
# ---------------------------------------------------------------------------

def run_shap_edges_v2(args: argparse.Namespace) -> Dict[str, Any]:
    set_all_seeds(int(args.seed))

    fold_dir = Path(args.run_dir) / f"fold_{args.fold}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

    out_dir = fold_dir / str(args.out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _tag_suffix(getattr(args, "tag", None))

    if getattr(args, "cache_dir", None):
        cache_dir = Path(args.cache_dir)
        if not cache_dir.is_absolute():
            cache_dir = fold_dir / cache_dir
    else:
        cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        f"[SHAP_EDGES_V2] fold={args.fold} clf={args.clf} "
        f"vector_mode={args.edge_vector_mode} K={args.edge_K} "
        f"per_channel={bool(args.edge_select_per_channel)}"
    )

    pipe_path = fold_dir / f"classifier_{args.clf}_pipeline_fold_{args.fold}.joblib"
    if not pipe_path.exists():
        raise FileNotFoundError(f"Classifier pipeline not found: {pipe_path}")
    pipe = joblib.load(pipe_path)

    norm_params_path = fold_dir / "vae_norm_params.joblib"
    if not norm_params_path.exists():
        raise FileNotFoundError(f"Normalization params not found: {norm_params_path}")
    norm_params = joblib.load(norm_params_path)
    label_info = load_label_info(fold_dir)

    feature_columns = load_feature_columns(
        fold_dir=fold_dir,
        latent_dim=args.latent_dim,
        metadata_features=args.metadata_features,
    )
    roi_names = load_roi_names_robust(
        run_dir=Path(args.run_dir),
        roi_order_path=getattr(args, "roi_order_path", None),
    )

    tensor_all, merged = load_global_and_merge(
        global_tensor_path=Path(args.global_tensor_path),
        metadata_path=Path(args.metadata_path),
    )
    cnad = subset_cnad(merged)
    if len(cnad) == 0:
        raise RuntimeError("CN/AD subset is empty after metadata merge.")

    test_idx_path = fold_dir / "test_indices.npy"
    if not test_idx_path.exists():
        raise FileNotFoundError(f"Test indices not found: {test_idx_path}")
    test_idx_in_cnad = np.load(test_idx_path)
    all_cnad_idx = np.arange(len(cnad))
    train_idx_in_cnad = np.setdiff1d(all_cnad_idx, test_idx_in_cnad, assume_unique=True)
    if train_idx_in_cnad.size == 0:
        raise RuntimeError("No TRAIN subjects found (train_idx_in_cnad is empty).")

    train_df = cnad.iloc[train_idx_in_cnad].copy()
    test_df = cnad.iloc[test_idx_in_cnad].copy()

    channels = [int(c) for c in args.channels_to_use]
    if len(channels) == 0:
        raise ValueError("channels_to_use cannot be empty.")

    R = int(tensor_all.shape[-1])
    C = len(channels)
    if len(roi_names) != R:
        raise ValueError(
            f"ROI name count ({len(roi_names)}) does not match tensor ROI dimension ({R}). "
            "Check ROI order source."
        )

    gidx_train = train_df["tensor_idx"].to_numpy(dtype=int)
    gidx_test = test_df["tensor_idx"].to_numpy(dtype=int)
    tens_train = apply_normalization_params(
        tensor_all[gidx_train][:, channels, :, :],
        norm_params,
    )
    tens_test = apply_normalization_params(
        tensor_all[gidx_test][:, channels, :, :],
        norm_params,
    )

    edge_rows, edge_cols, edges_per_channel, total_edges, is_directed = make_edge_index_v2(
        R=R,
        C=C,
        mode=args.edge_vector_mode,
    )
    _ = is_directed
    edges_train = vectorize_tensor_to_edges_v2(
        tensor=tens_train,
        rows=edge_rows,
        cols=edge_cols,
        mode=args.edge_vector_mode,
    )
    edges_test = vectorize_tensor_to_edges_v2(
        tensor=tens_test,
        rows=edge_rows,
        cols=edge_cols,
        mode=args.edge_vector_mode,
    )
    log.info(
        f"[SHAP_EDGES_V2] R={R} C={C} edges_per_channel={edges_per_channel} total_edges={total_edges}"
    )

    ch_tag = _channel_tag(channels)
    sel_cache_path = (
        cache_dir
        / (
            "edge_selection_"
            f"{args.edge_select_method}_mode-{args.edge_vector_mode}_"
            f"K{args.edge_K}_{'perch' if args.edge_select_per_channel else 'global'}_"
            f"seed{args.seed}_ch{ch_tag}.joblib"
        )
    )
    if sel_cache_path.exists():
        cache_sel = joblib.load(sel_cache_path)
        selected_idx = np.asarray(cache_sel["selected_indices"], dtype=int)
        scores = np.asarray(cache_sel["scores"], dtype=np.float64)
        selection_info = cache_sel.get("selection_info", {})
        log.info(f"[CACHE] Loaded edge selection: {sel_cache_path.name}")
    else:
        y_train = _labels_from_group(train_df)
        selected_idx, scores, selection_info = select_top_edges_v2(
            edges_train=edges_train,
            y_train=y_train,
            K=int(args.edge_K),
            method=args.edge_select_method,
            seed=int(args.seed),
            per_channel=bool(args.edge_select_per_channel),
            edges_per_channel=edges_per_channel,
            C=C,
        )
        joblib.dump(
            {
                "selected_indices": selected_idx,
                "scores": scores,
                "selection_info": selection_info,
                "edge_select_method": args.edge_select_method,
                "edge_select_per_channel": bool(args.edge_select_per_channel),
                "edge_vector_mode": args.edge_vector_mode,
                "edge_K": int(args.edge_K),
                "seed": int(args.seed),
                "channels": channels,
            },
            sel_cache_path,
        )
        log.info(f"[CACHE] Saved edge selection: {sel_cache_path.name}")

    K = int(len(selected_idx))
    if K <= 0:
        raise RuntimeError("No edges selected. Check edge_K and selection method.")
    if bool(args.edge_select_per_channel):
        log.info(
            "[SHAP_EDGES_V2] edge_select_per_channel=True -> edge_K interpreted as K per channel. "
            f"K_total_selected={K}."
        )
    else:
        log.info(f"[SHAP_EDGES_V2] edge_select_per_channel=False -> global top-K. K_selected={K}.")

    template_cache_path = cache_dir / f"train_edge_median_mode-{args.edge_vector_mode}_ch{ch_tag}.joblib"
    if template_cache_path.exists():
        all_edges_template = np.asarray(joblib.load(template_cache_path), dtype=np.float32)
        log.info(f"[CACHE] Loaded train median edge template: {template_cache_path.name}")
    else:
        all_edges_template = np.asarray(_compute_train_edge_median(edges_train), dtype=np.float32)
        joblib.dump(all_edges_template, template_cache_path)
        log.info(f"[CACHE] Saved train median edge template: {template_cache_path.name}")

    meta_cols = list(args.metadata_features or [])
    meta_frozen = _compute_frozen_meta_values(train_df, meta_cols)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = build_vae(
        vae_kwargs=vae_kwargs_from_args(args, image_size=R),
        state_dict_path=fold_dir / f"vae_model_fold_{args.fold}.pt",
        device=device,
    )
    pos_idx = resolve_positive_index(pipe, label_info)

    passed, max_diff, used_tol = validate_prediction_identity_v2(
        vae=vae,
        pipe=pipe,
        tensor_test_norm=tens_test,
        test_df=test_df,
        meta_cols=meta_cols,
        feature_columns=feature_columns,
        edge_rows=edge_rows,
        edge_cols=edge_cols,
        edge_vector_mode=args.edge_vector_mode,
        device=device,
        latent_features_type=args.latent_features_type,
        pos_idx=pos_idx,
        n_check=min(10, len(test_df)),
        atol_strict=1e-6,
        atol_relaxed=1e-5,
    )
    if not passed:
        raise RuntimeError(
            "Prediction identity check FAILED before SHAP. "
            f"max|diff|={max_diff:.3e}. "
            "This suggests vectorization/reconstruction or normalization mismatch."
        )

    indices_test_used = sample_test_indices(
        n_total=len(test_df),
        n_keep=args.n_test_shap,
        seed=int(args.seed),
    )
    test_df_used = test_df.iloc[indices_test_used].copy()
    edges_test_used = edges_test[indices_test_used]
    indices_test_used_in_cnad = np.asarray(test_idx_in_cnad, dtype=int)[indices_test_used]

    bg_idx_in_cnad = pick_bg_indices(
        cnad_df=cnad,
        train_idx_in_cnad=train_idx_in_cnad,
        mode=args.bg_mode,
        sample_size=int(args.bg_sample_size),
        seed=int(args.bg_seed),
    )
    bg_cache_path = (
        cache_dir
        / (
            "background_edges_"
            f"mode-{args.edge_vector_mode}_pool-{args.bg_mode}_"
            f"n{args.bg_sample_size}_seed{args.bg_seed}_ch{ch_tag}.joblib"
        )
    )
    if bg_cache_path.exists():
        cache_bg = joblib.load(bg_cache_path)
        edges_bg_full = np.asarray(cache_bg["edges_bg_full"], dtype=np.float32)
        bg_idx_cached = np.asarray(cache_bg.get("bg_idx_in_cnad", []), dtype=int)
        if edges_bg_full.ndim != 2 or edges_bg_full.shape[1] != total_edges:
            raise RuntimeError(
                f"Cached background has invalid shape {edges_bg_full.shape}, expected (_, {total_edges})."
            )
        if bg_idx_cached.size > 0 and not np.array_equal(bg_idx_cached, bg_idx_in_cnad):
            log.warning(
                "[CACHE] Background cache indices differ from requested config; "
                "using cached content because key matched."
            )
        log.info(f"[CACHE] Loaded background edges: {bg_cache_path.name}")
    else:
        gidx_bg = cnad.iloc[bg_idx_in_cnad]["tensor_idx"].to_numpy(dtype=int)
        tens_bg = apply_normalization_params(
            tensor_all[gidx_bg][:, channels, :, :],
            norm_params,
        )
        edges_bg_full = vectorize_tensor_to_edges_v2(
            tensor=tens_bg,
            rows=edge_rows,
            cols=edge_cols,
            mode=args.edge_vector_mode,
        )
        joblib.dump(
            {
                "bg_idx_in_cnad": bg_idx_in_cnad,
                "edges_bg_full": edges_bg_full,
                "bg_mode": args.bg_mode,
                "bg_sample_size": int(args.bg_sample_size),
                "bg_seed": int(args.bg_seed),
            },
            bg_cache_path,
        )
        log.info(f"[CACHE] Saved background edges: {bg_cache_path.name}")

    all_edge_names = make_edge_feature_names_v2(
        roi_names=roi_names,
        channels_used=channels,
        channel_names=getattr(args, "channel_names", None),
        rows=edge_rows,
        cols=edge_cols,
        mode=args.edge_vector_mode,
    )
    if len(all_edge_names) != total_edges:
        raise RuntimeError(
            f"Edge-name count mismatch: names={len(all_edge_names)} total_edges={total_edges}."
        )
    selected_edge_names = [all_edge_names[int(i)] for i in selected_idx]

    mapping_hash = _indices_hash(selected_idx)
    if getattr(args, "roi_annotation_csv", None):
        ann_hash = hashlib.sha1(str(args.roi_annotation_csv).encode("utf-8")).hexdigest()[:8]
        mapping_cache_path = cache_dir / (
            f"mapping_edges_mode-{args.edge_vector_mode}_sel-{mapping_hash}_ann-{ann_hash}.joblib"
        )
    else:
        mapping_cache_path = cache_dir / (
            f"mapping_edges_mode-{args.edge_vector_mode}_sel-{mapping_hash}_plain.joblib"
        )
    if mapping_cache_path.exists():
        mapping_df = joblib.load(mapping_cache_path)
        log.info(f"[CACHE] Loaded edge mapping: {mapping_cache_path.name}")
    else:
        mapping_df = make_edge_mapping_df_v2(
            selected_indices=selected_idx,
            roi_names=roi_names,
            channels_used=channels,
            channel_names=getattr(args, "channel_names", None),
            rows=edge_rows,
            cols=edge_cols,
            edges_per_channel=edges_per_channel,
            scores=scores,
            mode=args.edge_vector_mode,
        )
        mapping_df = maybe_annotate_mapping(mapping_df, getattr(args, "roi_annotation_csv", None))
        joblib.dump(mapping_df, mapping_cache_path)
        log.info(f"[CACHE] Saved edge mapping: {mapping_cache_path.name}")

    predict_fn = make_edge_predict_fn_v2(
        vae=vae,
        pipe=pipe,
        meta_cols=meta_cols,
        meta_frozen_values=meta_frozen,
        feature_columns=feature_columns,
        selected_edge_indices=selected_idx,
        all_edges_template=all_edges_template,
        R=R,
        C=C,
        edge_rows=edge_rows,
        edge_cols=edge_cols,
        edge_vector_mode=args.edge_vector_mode,
        device=device,
        latent_features_type=args.latent_features_type,
        pos_idx=pos_idx,
        shap_link=args.shap_link,
    )

    edges_bg_selected = np.asarray(edges_bg_full[:, selected_idx], dtype=np.float64)
    edges_test_selected = np.asarray(edges_test_used[:, selected_idx], dtype=np.float64)
    if edges_bg_selected.shape[0] == 0:
        raise RuntimeError("Background in edge space is empty; increase --bg_sample_size or change --bg_mode.")
    if edges_test_selected.shape[0] == 0:
        raise RuntimeError("No test samples selected for SHAP; check --n_test_shap.")

    min_evals = 2 * K + 1
    max_evals = int(max(min_evals, int(args.max_evals)))
    if int(args.max_evals) < min_evals:
        log.warning(
            f"[SHAP_EDGES_V2] max_evals={args.max_evals} too small for K={K}; using {max_evals}."
        )
    log.info(f"[SHAP_EDGES_V2] Running permutation SHAP with K={K}, max_evals={max_evals}.")

    masker = shap.maskers.Independent(edges_bg_selected)
    explainer = shap.Explainer(
        predict_fn,
        masker,
        algorithm="permutation",
        feature_names=selected_edge_names,
    )
    sv = explainer(edges_test_selected, max_evals=max_evals)
    shap_values = np.asarray(sv.values, dtype=np.float64)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[:, :, 0]
    if shap_values.ndim != 2 or shap_values.shape[1] != K:
        raise RuntimeError(
            f"Unexpected SHAP values shape: {shap_values.shape}; expected (N_used, {K})."
        )

    # Constant-feature guard: if feature is constant in both TEST and BG, force SHAP=0.
    std_test = np.std(edges_test_selected, axis=0)
    std_bg = np.std(edges_bg_selected, axis=0)
    const_mask = (std_test < 1e-12) & (std_bg < 1e-12)
    if np.any(const_mask):
        shap_values[:, const_mask] = 0.0
        log.info(f"[SHAP_EDGES_V2] Zeroed SHAP on {int(const_mask.sum())} constant features.")

    base_values_pack, base_values_for_exp = _prepare_base_values_for_explanation(
        base_values_raw=sv.base_values,
        n_samples=shap_values.shape[0],
    )
    exp = shap.Explanation(
        values=shap_values,
        base_values=base_values_for_exp,
        data=edges_test_selected,
        feature_names=selected_edge_names,
    )

    X_test_edges_selected_df = pd.DataFrame(
        edges_test_selected,
        columns=selected_edge_names,
    )

    pack: Dict[str, Any] = {
        "shap_values": shap_values.astype(np.float64),
        "base_values": base_values_pack,
        "X_test_edges_selected": X_test_edges_selected_df,
        "feature_names_edges_selected": selected_edge_names,
        "mapping_edges_selected": mapping_df,
        "selected_edge_indices": selected_idx.astype(int),
        "edge_selection_scores": scores.astype(np.float64),
        "edge_selection_info": selection_info,
        "test_subject_ids": test_df_used["SubjectID"].astype(str).tolist(),
        "test_labels": _labels_from_group(test_df_used).tolist(),
        "indices_test_used": indices_test_used.astype(int).tolist(),
        "indices_test_used_in_cnad": indices_test_used_in_cnad.astype(int).tolist(),
        "bg_indices_in_cnad": bg_idx_in_cnad.astype(int).tolist(),
        "meta_frozen_values": {k: float(v) for k, v in meta_frozen.items()},
        "edge_vector_mode": str(args.edge_vector_mode),
        "edge_select_method": str(args.edge_select_method),
        "edge_select_per_channel": bool(args.edge_select_per_channel),
        "edge_K_requested": int(args.edge_K),
        "edge_K_selected": int(K),
        "bg_mode": str(args.bg_mode),
        "bg_sample_size": int(args.bg_sample_size),
        "max_evals": int(max_evals),
        "seed": int(args.seed),
        "bg_seed": int(args.bg_seed),
        "shap_link": str(args.shap_link),
        "validation_max_abs_diff": float(max_diff),
        "validation_tol_used": float(used_tol),
        "args": {k: _to_serializable(v) for k, v in vars(args).items()},
    }

    pack_path = out_dir / f"shap_pack_edges_{args.clf}{tag}.joblib"
    joblib.dump(pack, pack_path)
    mapping_csv = out_dir / f"edge_to_roi_mapping_{args.clf}{tag}.csv"
    mapping_df.to_csv(mapping_csv, index=False)

    _plot_shap_summary_v2(
        exp=exp,
        out_dir=out_dir,
        fold=int(args.fold),
        clf=str(args.clf),
        tag=tag,
    )

    args_json_path = out_dir / f"run_args_shap_edges_v2{tag}.json"
    with open(args_json_path, "w", encoding="utf-8") as f:
        json.dump({k: _to_serializable(v) for k, v in vars(args).items()}, f, indent=2)

    log.info(f"[SHAP_EDGES_V2] Completed. Pack: {pack_path}")
    return pack
