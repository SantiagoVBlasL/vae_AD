#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_qc.py

Módulo de Control de Calidad (QC) Post-Hoc para el pipeline VAE-Clasificador.

Funciones principales:
- summarize_distribution_stages:
    Compara las distribuciones de datos (crudo, normalizado, reconstruido)
    para validar la coherencia del preprocesamiento y la arquitectura del VAE.
- evaluate_scanner_leakage:
    Evalúa el "batch effect" (fuga de información del escáner/sitio de adquisición)
    en espacio conectómico normalizado y en el espacio latente.
"""
from __future__ import annotations
import sys

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import re
import warnings
import numpy.linalg as npl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def _impute_numeric(vals: np.ndarray) -> np.ndarray:
    """
    Convierte a float e imputa NaNs con la mediana (fallback 0.0 si todo es NaN).
    """
    s = pd.to_numeric(pd.Series(vals), errors="coerce")
    if s.isna().all():
        return np.zeros(len(s), dtype=float)
    med = float(s.median())
    return s.fillna(med).to_numpy(dtype=float)


def _offdiag_mask(n: int) -> np.ndarray:
    """
    Máscara booleana off-diagonal (True fuera de la diagonal).
    """
    m = np.ones((n, n), dtype=bool)
    np.fill_diagonal(m, False)
    return m

def _safe_make_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _factorize(values: np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Factoriza etiquetas (str/obj) a enteros. Devuelve (y_int, mapping_int_to_label).
    """
    y_cat, uniques = pd.factorize(values.astype(str), sort=True)
    mapping = {int(i): str(lbl) for i, lbl in enumerate(list(uniques))}
    return y_cat.astype(int), mapping


def _choose_n_splits(y: np.ndarray, max_splits: int = 5) -> int:
    """
    Elige n_splits robusto para StratifiedKFold dado y (puede ser str o int).
    """
    y_ser = pd.Series(y)
    counts = y_ser.value_counts()
    if counts.empty:
        return 0
    min_count = int(counts.min())
    return int(min(max_splits, min_count))


def _ensure_2d_float(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z)
    if Z.ndim != 2:
        raise ValueError(f"Latentes deben ser 2D [N, d]. Recibido shape={Z.shape}")
    Z = Z.astype(np.float64, copy=False)
    if not np.isfinite(Z).all():
        # limpieza defensiva: reemplazar inf/nan por 0
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z


def compute_active_units(mu: np.ndarray, var_eps: float = 1e-4) -> Dict[str, Any]:
    """
    'Active units' típico en VAEs: cuenta dims con Var(mu_i) > var_eps.
    """
    Z = _ensure_2d_float(mu)
    var = np.var(Z, axis=0)
    active_mask = var > float(var_eps)
    return {
        "latent_dim": int(Z.shape[1]),
        "n_active": int(np.sum(active_mask)),
        "frac_active": float(np.mean(active_mask)),
        "var_eps": float(var_eps),
        "var_per_dim": var,
        "active_mask": active_mask,
    }


def estimate_total_correlation_gaussian(mu: np.ndarray, ridge: float = 1e-6) -> float:
    """
    Aproximación barata de Total Correlation (multi-information) suponiendo Gauss:
    TC ≈ 0.5 * ( log|diag(Σ)| - log|Σ| )

    - Σ: covarianza de mu (centro en media)
    - ridge: regularización diagonal para estabilidad numérica
    Devuelve TC en nats.
    """
    Z = _ensure_2d_float(mu)
    N, d = Z.shape
    if N < 5 or d < 2:
        return float("nan")
    Zc = Z - np.mean(Z, axis=0, keepdims=True)
    # cov muestral
    Sigma = (Zc.T @ Zc) / max(N - 1, 1)
    Sigma = Sigma + np.eye(d) * float(ridge)
    diag = np.diag(np.diag(Sigma))
    try:
        signS, logdetS = npl.slogdet(Sigma)
        signD, logdetD = npl.slogdet(diag)
        if signS <= 0 or signD <= 0:
            return float("nan")
        tc = 0.5 * (logdetD - logdetS)
        return float(max(tc, 0.0))
    except Exception:
        return float("nan")


def _mi_classif_per_dim(Z: np.ndarray,
                        y: np.ndarray,
                        random_state: int = 0,
                        n_neighbors: int = 3) -> np.ndarray:
    """
    MI(Z_i; y) por dimensión usando mutual_info_classif.
    Devuelve array [d].
    """
    Z = _ensure_2d_float(Z)
    y = np.asarray(y).ravel()
    # mutual_info_* puede quejarse si y tiene un único valor
    if len(np.unique(y)) < 2:
        return np.full((Z.shape[1],), np.nan, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi = mutual_info_classif(
            Z, y,
            discrete_features=False,
            random_state=int(random_state),
            n_neighbors=int(n_neighbors),
        )
    return np.asarray(mi, dtype=float)


def _mi_regress_per_dim(Z: np.ndarray,
                        y: np.ndarray,
                        random_state: int = 0,
                        n_neighbors: int = 3) -> np.ndarray:
    """
    MI(Z_i; y_cont) por dimensión usando mutual_info_regression.
    Devuelve array [d].
    """
    Z = _ensure_2d_float(Z)
    y = np.asarray(y, dtype=float).ravel()
    if np.allclose(np.std(y), 0.0):
        return np.full((Z.shape[1],), np.nan, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi = mutual_info_regression(
            Z, y,
            discrete_features=False,
            random_state=int(random_state),
            n_neighbors=int(n_neighbors),
        )
    return np.asarray(mi, dtype=float)


def _sanitize_name(name: str) -> str:
    """
    Limpia nombres de canal para usarlos en nombres de archivo.
    """
    return re.sub(r"[^a-zA-Z0-9]+", "_", name)[:80]


def _compute_offdiag_values(
    tensor_4d: np.ndarray,
    channel_idx: int,
    offdiag_mask: np.ndarray
) -> np.ndarray:
    """
    Extrae TODOS los valores off-diagonal para un canal específico,
    apilando todos los sujetos.

    tensor_4d shape esperada: [N_subjects, N_channels, R, R]
    """
    vals = tensor_4d[:, channel_idx, :, :][:, offdiag_mask].ravel()
    return vals


def _stage_stats_df(
    tensor_4d: np.ndarray,
    channel_names: List[str],
    final_activation: Optional[str] = None
) -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas por canal (solo off-diagonal):
    - moments básicos (mean, std, skew, kurtosis)
    - percentiles
    - colas extremas
    - saturación si aplica (tanh / sigmoid / linear)
    """
    n_subj, n_ch, n_rois, _ = tensor_4d.shape
    offmask = _offdiag_mask(n_rois)
    stats_list: List[Dict] = []

    for c in range(n_ch):
        vals = _compute_offdiag_values(tensor_4d, c, offmask)
        if vals.size == 0:
            continue

        prc = np.percentile(vals, [1, 5, 25, 50, 75, 95, 99])

        row = {
            "channel": channel_names[c] if c < len(channel_names) else f"Chan{c}",
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "skew": float(skew(vals, bias=False)),
            "kurtosis": float(kurtosis(vals, fisher=False, bias=False)),
            "min": float(np.min(vals)),
            "p1": float(prc[0]),
            "p5": float(prc[1]),
            "p25": float(prc[2]),
            "p50": float(prc[3]),
            "p75": float(prc[4]),
            "p95": float(prc[5]),
            "p99": float(prc[6]),
            "max": float(np.max(vals)),
            "frac_abs_gt2": float(np.mean(np.abs(vals) > 2.0)),
            "frac_abs_gt3": float(np.mean(np.abs(vals) > 3.0)),
        }

        # saturación / clipping según activación final del decoder
        if final_activation is not None:
            if final_activation == "tanh":
                row["frac_saturation"] = float(np.mean(np.abs(vals) > 0.95))
            elif final_activation == "sigmoid":
                row["frac_saturation"] = float(np.mean((vals < 0.05) | (vals > 0.95)))
            elif final_activation == "linear":
                # sin saturación teórica; reportamos valores hiper-extremos
                row["frac_saturation"] = float(np.mean(np.abs(vals) > 5.0))

        stats_list.append(row)

    return pd.DataFrame(stats_list)


def _save_overlay_hist_per_channel(
    raw_tensor: np.ndarray,
    norm_tensor: np.ndarray,
    recon_tensor: np.ndarray,
    channel_names: List[str],
    out_dir: Path,
    prefix: str
) -> None:
    """
    Para cada canal: histograma raw vs norm vs recon.
    Guarda .png por canal en out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _, n_ch, n_rois, _ = raw_tensor.shape
    offmask = _offdiag_mask(n_rois)

    for c in range(n_ch):
        ch_name = channel_names[c] if c < len(channel_names) else f"Chan{c}"

        raw_vals = _compute_offdiag_values(raw_tensor, c, offmask)
        norm_vals = _compute_offdiag_values(norm_tensor, c, offmask)
        recon_vals = _compute_offdiag_values(recon_tensor, c, offmask)

        # downsample para que matplotlib no muera con decenas de millones de puntos
        if raw_vals.size > 2_000_000:
            raw_vals = np.random.choice(raw_vals, 2_000_000, replace=False)
        if norm_vals.size > 2_000_000:
            norm_vals = np.random.choice(norm_vals, 2_000_000, replace=False)
        if recon_vals.size > 2_000_000:
            recon_vals = np.random.choice(recon_vals, 2_000_000, replace=False)

        plt.figure(figsize=(10, 6))
        plt.hist(
            raw_vals,
            bins=100, density=True, alpha=0.5,
            label=f"Raw (Std={np.std(raw_vals):.2f})",
            color="gray"
        )
        plt.hist(
            norm_vals,
            bins=100, density=True, alpha=0.5,
            label=f"Norm (Std={np.std(norm_vals):.2f})",
            color="blue"
        )
        plt.hist(
            recon_vals,
            bins=100, density=True, alpha=0.5,
            label=f"Recon (Std={np.std(recon_vals):.2f})",
            color="red"
        )
        plt.xlabel("Valor de Conectividad (off-diag)")
        plt.ylabel("Densidad")
        plt.title(f"Superposición de Distribuciones: {ch_name} ({prefix})")
        plt.legend(loc="best")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        fname = out_dir / f"{prefix}_hist_{_sanitize_name(ch_name)}.png"
        plt.savefig(fname, dpi=150)
        plt.close()


def summarize_distribution_stages(
    raw_tensor: np.ndarray,
    norm_tensor: np.ndarray,
    recon_tensor: np.ndarray,
    channel_names: List[str],
    out_dir: Path,
    prefix: str,
    final_activation: str
) -> Dict[str, pd.DataFrame]:
    """
    Calcula stats canal-wise para raw/norm/recon y guarda:
    - CSVs con stats
    - PNGs con histogramas overlaid
    """
    _safe_make_out_dir(out_dir)

    df_raw = _stage_stats_df(
        raw_tensor,
        channel_names,
        final_activation=None
    )
    df_norm = _stage_stats_df(
        norm_tensor,
        channel_names,
        final_activation=None
    )
    df_rec = _stage_stats_df(
        recon_tensor,
        channel_names,
        final_activation=final_activation
    )

    df_raw.to_csv(out_dir / f"{prefix}_dist_raw.csv", index=False)
    df_norm.to_csv(out_dir / f"{prefix}_dist_norm.csv", index=False)
    df_rec.to_csv(out_dir / f"{prefix}_dist_recon.csv", index=False)

    _save_overlay_hist_per_channel(
        raw_tensor,
        norm_tensor,
        recon_tensor,
        channel_names,
        out_dir,
        prefix,
    )

    return {"raw": df_raw, "norm": df_norm, "recon": df_rec}


def summarize_rate_distortion_history(
    history_data: Dict[str, Any],
    out_dir: Path,
    prefix: str,
    beta_max: float,
    log_base: float = 2.0
) -> pd.DataFrame:
    """
    Convierte history_data (como el que guardás en run_vae_clf_ad.py) a una tabla Rate–Distortion.

    Espera keys típicas:
    - train_recon, train_kld, val_recon, val_kld, beta
    - (opcionales) val_loss_modelsel, train_loss, val_loss

    Define:
    D = recon (MSE / sample)
    R = KLD  (nats / sample)
    R_bits = R / ln(2)
    L_betaMax = D + beta_max * R
    """
    _safe_make_out_dir(out_dir)
    ln_base = np.log(float(log_base))
    ln2 = np.log(2.0)

    def _get_list(key: str) -> List[float]:
        v = history_data.get(key, [])
        return list(v) if isinstance(v, (list, tuple, np.ndarray)) else []

    train_recon = _get_list("train_recon")
    train_kld   = _get_list("train_kld")
    val_recon   = _get_list("val_recon")
    val_kld     = _get_list("val_kld")
    beta_list   = _get_list("beta")
    n_epochs = max(len(train_recon), len(train_kld), len(val_recon), len(val_kld), len(beta_list))
    if n_epochs == 0:
        df = pd.DataFrame()
        df.to_csv(out_dir / f"{prefix}_rate_distortion.csv", index=False)
        return df

    def _pad(x: List[float]) -> List[float]:
        if len(x) == n_epochs:
            return x
        if len(x) == 0:
            return [np.nan] * n_epochs
        # pad con el último valor
        return x + [x[-1]] * (n_epochs - len(x))

    train_recon = _pad(train_recon)
    train_kld   = _pad(train_kld)
    val_recon   = _pad(val_recon)
    val_kld     = _pad(val_kld)
    beta_list   = _pad(beta_list)

    df = pd.DataFrame({
        "epoch": np.arange(1, n_epochs + 1),
        "beta":  beta_list,
        "D_train": train_recon,
        "R_train_nats": train_kld,
        "R_train_bits": np.array(train_kld, dtype=float) / ln_base,
        "L_train_betaMax": np.array(train_recon, dtype=float) + float(beta_max) * np.array(train_kld, dtype=float),
        "D_val": val_recon,
        "R_val_nats": val_kld,
        "R_val_bits": np.array(val_kld, dtype=float) / ln2,
        "R_val_logbase": np.array(val_kld, dtype=float) / ln_base,
        "L_val_betaMax": np.array(val_recon, dtype=float) + float(beta_max) * np.array(val_kld, dtype=float),
        "log_base": float(log_base),
    })

    df.to_csv(out_dir / f"{prefix}_rate_distortion.csv", index=False)
    return df


def evaluate_latent_information(
    metadata_df_full: pd.DataFrame,
    subject_global_indices: np.ndarray,
    latent_mu_subjects: np.ndarray,
    y_target: np.ndarray,
    out_dir: Path,
    fold_tag: str,
    nuisance_cols: Optional[List[str]] = None,
    random_state: int = 0,
    n_neighbors: int = 3,
    top_k: int = 10,
    var_eps_active: float = 1e-4,
    tc_ridge: float = 1e-6,
) -> pd.DataFrame:
    """
    QC de teoría de la información en espacio latente (mu):
    - MI por dimensión: I(z_i; Y) en nats
    - MI total/mean con Y
    - MI con nuisances (por defecto intenta sitio/escáner, y opcionalmente otras cols)
    - Active units (Var(mu_i) > eps)
    - Total Correlation gaussiana (aprox)

    Guarda:
    - {fold_tag}_latent_info_summary.csv  (1 fila por variable: Y y nuisances)
    - {fold_tag}_latent_info_per_dim.csv  (MI por dimensión para Y y nuisances)
    """
    _safe_make_out_dir(out_dir)
    Z = _ensure_2d_float(latent_mu_subjects)
    y_target = np.asarray(y_target).ravel()
    # si Y es numérico y "casi entero", lo pasamos a int para mutual_info_classif
    if np.issubdtype(y_target.dtype, np.number):
        if np.all(np.isclose(y_target, np.round(y_target))):
            y_target = y_target.astype(int)

    # ------------ Active units + TC ------------
    au = compute_active_units(Z, var_eps=float(var_eps_active))
    tc = estimate_total_correlation_gaussian(Z, ridge=float(tc_ridge))

    # ------------ Construir diccionario tensor_idx -> filas metadata ------------
    if "tensor_idx" not in metadata_df_full.columns:
        raise ValueError("metadata_df_full debe contener columna 'tensor_idx'.")
    meta_by_idx = metadata_df_full.set_index("tensor_idx", drop=False)

    # ------------ Elegir nuisances por defecto ------------
    if nuisance_cols is None:
        # intentamos detectar una columna de sitio/escáner automáticamente
        site_cols = [
            c for c in metadata_df_full.columns
            if ("manufacturer" in c.lower()) or ("vendor" in c.lower()) or ("site" in c.lower())
        ]
        nuisance_cols = []
        if len(site_cols) > 0:
            nuisance_cols.append(site_cols[0])
        # si existen, las incluimos porque suelen ser confusores
        for c in ["Sex", "Age_Group"]:
            if c in metadata_df_full.columns and c not in nuisance_cols:
                nuisance_cols.append(c)

    # ------------ MI con Y (diagnóstico) ------------
    mi_y = _mi_classif_per_dim(Z, y_target, random_state=random_state, n_neighbors=n_neighbors)
    order_y = np.argsort(-np.nan_to_num(mi_y, nan=-1.0))
    top_idx_y = order_y[: int(min(top_k, Z.shape[1]))].tolist()

    per_dim_rows = []
    for i in range(Z.shape[1]):
        per_dim_rows.append({
            "fold_tag": fold_tag,
            "variable": "Y_target",
            "dim": int(i),
            "mi_nats": float(mi_y[i]) if np.isfinite(mi_y[i]) else np.nan,
        })

    summary_rows = [{
        "fold_tag": fold_tag,
        "variable": "Y_target",
        "mi_sum_nats": float(np.nansum(mi_y)),
        "mi_mean_nats": float(np.nanmean(mi_y)),
        "top_k": int(min(top_k, Z.shape[1])),
        "top_dims": ",".join(map(str, top_idx_y)),
        "n_samples": int(Z.shape[0]),
        "latent_dim": int(Z.shape[1]),
        "n_active": int(au["n_active"]),
        "frac_active": float(au["frac_active"]),
        "total_correlation_nats": float(tc),
    }]

    # ------------ MI con nuisances ------------
    for col in nuisance_cols:
        if col not in meta_by_idx.columns:
            continue
        try:
            vals = meta_by_idx.loc[subject_global_indices, col].values
        except Exception:
            # fallback: si hay índices faltantes
            vals = []
            for idx in subject_global_indices:
                if idx in meta_by_idx.index:
                    vals.append(meta_by_idx.loc[idx, col])
                else:
                    vals.append("Unknown")
            vals = np.asarray(vals)

        # heurística: si es numérica y tiene muchos valores -> regression, sino classification
        is_numeric = pd.api.types.is_numeric_dtype(pd.Series(vals))
        n_unique = int(pd.Series(vals).nunique(dropna=True))

        if is_numeric and n_unique > 10:
            y_cont = _impute_numeric(vals)
            mi = _mi_regress_per_dim(Z, y_cont, random_state=random_state, n_neighbors=n_neighbors)
        else:
            # imputación categórica
            if pd.isna(vals).any():
                vals = pd.Series(vals).fillna("Unknown").values
            y_int, _ = _factorize(vals)
            mi = _mi_classif_per_dim(Z, y_int, random_state=random_state, n_neighbors=n_neighbors)

        order = np.argsort(-np.nan_to_num(mi, nan=-1.0))
        top_idx = order[: int(min(top_k, Z.shape[1]))].tolist()

        for i in range(Z.shape[1]):
            per_dim_rows.append({
                "fold_tag": fold_tag,
                "variable": str(col),
                "dim": int(i),
                "mi_nats": float(mi[i]) if np.isfinite(mi[i]) else np.nan,
            })
        summary_rows.append({
            "fold_tag": fold_tag,
            "variable": str(col),
            "mi_sum_nats": float(np.nansum(mi)),
            "mi_mean_nats": float(np.nanmean(mi)),
            "top_k": int(min(top_k, Z.shape[1])),
            "top_dims": ",".join(map(str, top_idx)),
            "n_samples": int(Z.shape[0]),
            "latent_dim": int(Z.shape[1]),
            "n_active": int(au["n_active"]),
            "frac_active": float(au["frac_active"]),
            "total_correlation_nats": float(tc),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_per_dim = pd.DataFrame(per_dim_rows)

    df_summary.to_csv(out_dir / f"{fold_tag}_latent_info_summary.csv", index=False)
    df_per_dim.to_csv(out_dir / f"{fold_tag}_latent_info_per_dim.csv", index=False)
    return df_summary


def evaluate_scanner_leakage(
    metadata_df_full: pd.DataFrame,
    subject_global_indices: np.ndarray,
    normalized_tensor_subjects: np.ndarray,
    latent_mu_subjects: np.ndarray,
    out_dir: Path,
    fold_tag: str,
    random_state: int = 0,
) -> Optional[pd.DataFrame]:
    """
    Evalúa qué tan separable es el sitio/escáner (batch effect) con:
      A) conectomas normalizados aplanados
      B) espacio latente mu del VAE

    Usa LogisticRegression con CV (balanced_accuracy). Guarda CSV.
    """
    _safe_make_out_dir(out_dir)

    # buscar columna de sitio/escáner
    site_cols = [
        c for c in metadata_df_full.columns
        if ("manufacturer" in c.lower()) or ("vendor" in c.lower()) or ("site" in c.lower())
    ]

    if len(site_cols) == 0:
        print(f"[{fold_tag} QC] No se encontró columna de escáner/sitio. Saltando análisis de fuga.")
        return None
    site_col = site_cols[0]

    # mapear tensor_idx -> etiqueta de sitio
    idx_to_site = dict(
        zip(
            metadata_df_full["tensor_idx"].values,
            metadata_df_full[site_col].astype(str).values
        )
    )
    y_site = np.array([idx_to_site.get(i, "Unknown") for i in subject_global_indices])

    if len(np.unique(y_site)) < 2:
        print(f"[{fold_tag} QC] Solo se encontró 1 sitio/escáner. No se puede estimar fuga.")
        return None

    # features conectoma normalizado (flatten)
    N, C, R, _ = normalized_tensor_subjects.shape
    X_conn = normalized_tensor_subjects.reshape(N, C * R * R)

    # features latentes
    X_lat = _ensure_2d_float(latent_mu_subjects)

    clf_site = LogisticRegression(
        max_iter=1000,
        multi_class="auto",
        class_weight="balanced",
        solver="lbfgs"
    )
    n_splits = _choose_n_splits(y_site, max_splits=5)
    if n_splits < 2:
        print(f"[{fold_tag} QC] Muy pocas muestras por sitio para CV (n_splits={n_splits}). Saltando.")
        return None
    cv_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
  

    acc_conn = cross_val_score(
        clf_site, X_conn, y_site, cv=cv_inner, scoring="balanced_accuracy"
    )
    acc_lat = cross_val_score(
        clf_site, X_lat, y_site, cv=cv_inner, scoring="balanced_accuracy"
    )

    df_out = pd.DataFrame({
        "representation": ["connectome_norm", "latent_mu"],
        "balanced_accuracy_mean": [float(np.mean(acc_conn)), float(np.mean(acc_lat))],
        "balanced_accuracy_std":  [float(np.std(acc_conn)),  float(np.std(acc_lat))],
        "n_classes": [int(len(np.unique(y_site)))] * 2,
        "site_col": [site_col] * 2,
        "fold_tag": [fold_tag] * 2,
        "n_splits": [int(n_splits)] * 2,
    })

    df_out.to_csv(out_dir / f"{fold_tag}_scanner_leakage.csv", index=False)
    return df_out