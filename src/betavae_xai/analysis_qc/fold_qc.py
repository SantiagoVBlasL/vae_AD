# coding=utf-8
"""
betavae_xai.analysis_qc.fold_qc.py
Funciones de control de calidad (QC) específicas para folds de validación cruzada.
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Any
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

logger = logging.getLogger(__name__)
# En tu setting (canales simétricos, sin Granger) conviene forzar vectorización upper
# para evitar duplicar features y estabilizar leakage.
# OJO: en tu pipeline también existe Granger (direccional). Para evitar elegir "upper"
# cuando hay asimetría, lo más seguro es default "auto" y que la heurística decida.
DEFAULT_VECT_MODE_SYMMETRIC = "auto"

def _detect_site_column(metadata_df_full: pd.DataFrame, *, prefer_manufacturer: bool = True) -> Optional[str]:
    """Elige una columna de sitio/escáner de forma robusta.

    Preferimos Manufacturer/Vendor como proxy de batch effect (típicamente pocas clases),
    porque Site/Scanner suelen tener demasiadas clases y en subsets chicos se vuelve
    imposible hacer StratifiedKFold (min_count=1 -> n_splits=0).

    Si no existe Manufacturer/Vendor (o colapsa a 1 clase), caemos a Site/Scanner/Center.
    """
    if metadata_df_full is None or metadata_df_full.empty:
        return None
    pref: List[str] = []
    other: List[str] = []
    for c in metadata_df_full.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["manufacturer", "vendor"]):
            pref.append(c)
        elif any(k in cl for k in ["site", "scanner", "center", "centre"]):
            other.append(c)

    candidates = pref + other
    if not candidates:
        return None

    def _score(col: str) -> tuple[int, int]:
        s = metadata_df_full[col]
        nn = int(s.notna().sum())
        nunique = int(s.nunique(dropna=True))
        return nn, nunique

    def _best_from(cols: List[str]) -> Optional[str]:
        best: Optional[str] = None
        best_score: Optional[tuple[int, int]] = None
        for c in cols:
            nn, nunique = _score(c)
            # exigir al menos 2 clases y algo de cobertura
            if nn == 0 or nunique < 2:
                continue
            # preferimos cobertura alta y MENOS clases (más estable para CV)
            score = (nn, -nunique)
            if best is None or score > best_score:
                best, best_score = c, score
        return best

    # Opción 1: preferir Manufacturer/Vendor si es usable
    if prefer_manufacturer and pref:
        best = _best_from(pref)
        if best is not None:
            return best

    # Fallback: usar Site/Scanner/Center
    return _best_from(other if other else candidates)

def _is_approximately_symmetric(X: np.ndarray, tol: float = 1e-6, n_check: int = 3) -> bool:
    """
    Heurística barata: chequea asimetría en pocas muestras/canales.
    X: [N,C,R,R]
    """
    X = np.asarray(X)
    if X.ndim != 4:
        return True
    N, C, R, R2 = X.shape
    if R != R2 or N == 0 or C == 0:
        return True
    nN = min(int(N), int(n_check))
    #nC = min(int(C), int(n_check))
    # Para C chico (<=16), chequeamos todos los canales: evita perder Granger.
    if C <= 16:
        idxC = np.arange(C, dtype=int)
    else:
        nC = min(int(C), int(n_check))
        idxC = np.linspace(0, C - 1, nC, dtype=int) if C > 1 else np.array([0], dtype=int)
    # Importante: muestrear índices ESPACIADOS (primero/medio/último)
    # para no “perder” un canal direccional que esté al final (p.ej., Granger).
    idxN = np.linspace(0, N - 1, nN, dtype=int) if N > 1 else np.array([0], dtype=int)
    #idxC = np.linspace(0, C - 1, nC, dtype=int) if C > 1 else np.array([0], dtype=int)
    for i in idxN:
        for c in idxC:
            A = X[int(i), int(c)]
            # ignoramos diagonal
            d = A - A.T
            d = d[~np.eye(R, dtype=bool)]
            if np.nanmax(np.abs(d)) > tol:
                return False
    return True

def _resolve_vectorize_mode(X: np.ndarray, mode: str, sym_tol: float = 1e-6, n_check: int = 3) -> str:
    mode = str(mode).lower()
    if mode == "auto":
        return "upper" if _is_approximately_symmetric(X, tol=float(sym_tol), n_check=int(n_check)) else "full"
    if mode in ("upper", "full"):
        return mode
    raise ValueError(f"mode inválido: {mode}. Use 'auto', 'upper' o 'full'.")


def _vectorize_connectome_offdiag(
    X: np.ndarray,
    mode: str = "auto",
    sym_tol: float = 1e-6,
    n_check: int = 3,
) -> np.ndarray:
    """
    Vectoriza conectomas [N,C,R,R] por canal usando off-diagonal.
    - mode="upper": usa triángulo superior (k=1). OK si matrices simétricas.
    - mode="full":  usa TODA off-diagonal (ambos triángulos). Recomendado si hay direccionalidad (p.ej. Granger).
    - mode="auto":  detecta simetría y elige upper/full.
    """
    X = np.asarray(X)
    if X.ndim != 4:
        raise ValueError(f"Esperaba tensor 4D [N,C,R,R], recibí shape={X.shape}")
    N, C, R, R2 = X.shape
    if R != R2:
        raise ValueError(f"Matrices no cuadradas: R={R}, R2={R2}")
    mode = str(mode).lower()
    if mode == "auto":
        mode = "upper" if _is_approximately_symmetric(X, tol=float(sym_tol), n_check=int(n_check)) else "full"

    feats = []
    if mode == "upper":
        iu = np.triu_indices(R, k=1)
        for c in range(C):
            feats.append(X[:, c, iu[0], iu[1]])
    elif mode == "full":
        off_diag_mask = ~np.eye(R, dtype=bool)
        for c in range(C):
            feats.append(X[:, c, off_diag_mask])
    else:
        raise ValueError(f"mode inválido: {mode}. Use 'auto', 'upper' o 'full'.")
    return np.concatenate(feats, axis=1)

def log_group_distributions(df: pd.DataFrame, group_cols: List[str], dataset_name: str, fold_idx_str: str):
    """
    Loguea la distribución de grupos categóricos en un DataFrame dado.
    """
    if df.empty:
        logger.info(f"  {fold_idx_str} {dataset_name}: DataFrame vacío.")
        return
    log_msg = f"  {fold_idx_str} {dataset_name} (N={len(df)}):\n"
    for col in group_cols:
        if col in df.columns:
            counts = df[col].value_counts().sort_index()
            log_msg += f"    {col}:\n"
            for val, count in counts.items():
                log_msg += f"      {val}: {count} ({count/len(df)*100:.1f}%)\n"
        else:
            log_msg += f"    {col}: No encontrado en el DataFrame.\n"
    logger.info(log_msg.strip())


def compute_latent_silhouette(latent_feats: np.ndarray,
                              labels_binary: np.ndarray) -> float:
    """
    Devuelve silhouette_score AD vs CN en el espacio latente.
    latent_feats: [N_test, latent_dim] (mu o z del VAE)
    labels_binary: [N_test] con 0=CN,1=AD
    Si hay <2 clases en test, devuelve np.nan.
    """
    try:
        if len(np.unique(labels_binary)) < 2 or latent_feats.shape[0] < 5:
            return np.nan
        # silhouette_score necesita etiquetas en forma de cluster IDs (0/1 ya sirve)
        sil = silhouette_score(latent_feats, labels_binary, metric='euclidean')
        return float(sil)
    except Exception as e:
        logger.warning(f"[QC silhouette] fallo calculando silhouette: {e}")
        return np.nan


def scanner_leakage_simple(
    metadata_df_full: pd.DataFrame,
    subject_global_indices: np.ndarray,
    X_norm_subjects: np.ndarray,
    X_latent_subjects: np.ndarray,
    random_state: int = 0,
    vectorize_mode: str = DEFAULT_VECT_MODE_SYMMETRIC,
) -> dict:
    """
    Evalúa la 'fuga de sitio' en TEST SET o cualquier subconjunto dado.
    Devuelve dict con:
        acc_site_raw      (balanced_accuracy_mean conectoma normalizado)
        acc_site_latent   (balanced_accuracy_mean mu latente)
        n_sites           (#clases)
    Si no hay columna de sitio o sólo 1 sitio, devuelve NaNs.
    """
    # Wrapper legacy: usa evaluate_scanner_leakage (summary) y devuelve dict estable
    df = evaluate_scanner_leakage(
        metadata_df_full=metadata_df_full,
        subject_global_indices=subject_global_indices,
        normalized_tensor_subjects=X_norm_subjects,
        latent_mu_subjects=X_latent_subjects,
        out_dir=None,     # no guardamos
        fold_tag="scanner_leakage_simple",
        random_state=random_state,
        vectorize_mode=vectorize_mode,
    )
    if df is None or df.empty:
        return {"acc_site_raw": np.nan, "acc_site_latent": np.nan, "n_sites": 1}
    row = df.iloc[0]
    return {
        "acc_site_raw": float(row.get("acc_site_raw", np.nan)),
        "acc_site_latent": float(row.get("acc_site_latent", np.nan)),
        "n_sites": int(row.get("n_sites", 1)),
    }

def _safe_make_out_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

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
        "R_train_bits": np.array(train_kld, dtype=float) / ln2,
        "R_train_logbase": np.array(train_kld, dtype=float) / ln_base,
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
    out_dir: Optional[Path],
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
    #_safe_make_out_dir(out_dir)
    if out_dir is not None:
        _safe_make_out_dir(out_dir)
    Z = _ensure_2d_float(latent_mu_subjects)
    y_target = np.asarray(y_target).ravel()

    # ---- robustez: si Y es constante, MI no está definida ----
    if y_target.size == 0:
        return pd.DataFrame()
    if len(np.unique(y_target[~pd.isna(y_target)])) < 2:
        logger.warning(f"[{fold_tag} latent_info] y_target tiene <2 clases/valores. MI con Y será NaN.")

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
        nuisance_cols = []
        site_col = _detect_site_column(metadata_df_full,prefer_manufacturer=True)
        if site_col is not None:
            nuisance_cols.append(site_col)
        # si existen, las incluimos porque suelen ser confusores
        for c in ["Sex", "Age_Group"]:
            if c in metadata_df_full.columns and c not in nuisance_cols:
                nuisance_cols.append(c)

    else:
        nuisance_cols = list(nuisance_cols)
    # deduplicar manteniendo orden
    nuisance_cols = list(dict.fromkeys(nuisance_cols))

    # ------------ MI con Y (diagnóstico) ------------
    if len(np.unique(y_target[~pd.isna(y_target)])) < 2:
        mi_y = np.full(Z.shape[1], np.nan, dtype=float)
    else:
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
            # si la nuisance continua es (casi) constante, MI ~ 0
            if np.nanstd(y_cont) < 1e-12:
                mi = np.full(Z.shape[1], np.nan, dtype=float)
            else:
                mi = _mi_regress_per_dim(Z, y_cont, random_state=random_state, n_neighbors=n_neighbors)
        else:
            # imputación categórica
            if pd.isna(vals).any():
                vals = pd.Series(vals).fillna("Unknown").values
            y_int, _ = _factorize(vals)
            if len(np.unique(y_int)) < 2:
                mi = np.full(Z.shape[1], np.nan, dtype=float)
            else:
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

    if out_dir is not None:
        df_summary.to_csv(out_dir / f"{fold_tag}_latent_info_summary.csv", index=False)
        df_per_dim.to_csv(out_dir / f"{fold_tag}_latent_info_per_dim.csv", index=False)
    return df_summary


def evaluate_scanner_leakage(
    metadata_df_full: pd.DataFrame,
    subject_global_indices: np.ndarray,
    normalized_tensor_subjects: np.ndarray,
    latent_mu_subjects: np.ndarray,
    out_dir: Optional[Path],
    fold_tag: str,
    random_state: int = 0,
    vectorize_mode: str = DEFAULT_VECT_MODE_SYMMETRIC,
) -> Optional[pd.DataFrame]:
    """
    Evalúa qué tan separable es el sitio/escáner (batch effect) con:
      A) conectomas normalizados aplanados
      B) espacio latente mu del VAE

    Usa LogisticRegression con CV (balanced_accuracy). Guarda CSV.
    """
    if out_dir is not None:
        _safe_make_out_dir(out_dir)

    if "tensor_idx" not in metadata_df_full.columns:
        logger.info(f"[{fold_tag} QC] No se encontró 'tensor_idx' en metadata. Saltando fuga.")
        return None

    subject_global_indices = np.asarray(subject_global_indices).astype(int)

    # --- Selección robusta de columna batch PARA ESTE SUBSET ---
    # probamos candidates (manufacturer/vendor primero) y nos quedamos con el que permite CV (n_splits>=2)
    pref_cols: List[str] = []
    other_cols: List[str] = []
    for c in metadata_df_full.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["manufacturer", "vendor"]):
            pref_cols.append(c)
        elif any(k in cl for k in ["site", "scanner", "center", "centre"]):
            other_cols.append(c)
    candidates = pref_cols + other_cols
    if not candidates:
        logger.info(f"[{fold_tag} QC] No hay columnas candidatas (Manufacturer/Vendor/Site/Scanner/Center).")
        return None

    meta_by_idx = metadata_df_full.set_index("tensor_idx", drop=False)
    best_col = None
    best_score = None
    best_y = None
    best_keep = None

    for col in candidates:
        try:
            vals = meta_by_idx.loc[subject_global_indices, col].astype(str).values
        except Exception:
            vals = []
            for idx in subject_global_indices:
                if idx in meta_by_idx.index:
                    vals.append(str(meta_by_idx.loc[idx, col]))
                else:
                    vals.append("Unknown")
            vals = np.asarray(vals, dtype=str)

        y = pd.Series(vals).fillna("Unknown").astype(str).values
        keep = np.ones(len(y), dtype=bool)

        # filtrar Unknown sólo si eso deja >=2 clases reales
        if np.any(y == "Unknown"):
            keep_known = (y != "Unknown")
            y_known = y[keep_known]
            if len(np.unique(y_known)) >= 2:
                y = y_known
                keep = keep_known

        n_classes = int(len(np.unique(y)))
        if n_classes < 2:
            continue
        n_splits = _choose_n_splits(y, max_splits=5)
        if n_splits < 2:
            continue

        # score: preferimos más splits, más cobertura, menos clases
        score = (int(n_splits), int(np.sum(keep)), -int(n_classes))
        if best_col is None or score > best_score:
            best_col = col
            best_score = score
            best_y = y
            best_keep = keep

    if best_col is None:
        logger.info(f"[{fold_tag} QC] Ninguna columna batch permite CV (n_splits>=2).")
        return None

    site_col = best_col
    y_site = np.asarray(best_y)
    keep = np.asarray(best_keep, dtype=bool)

    # aplicar keep mask a features (si filtramos Unknown)
    if keep is not None and keep.size == normalized_tensor_subjects.shape[0] and not np.all(keep):
        n_drop = int(np.sum(~keep))
        logger.warning(f"[{fold_tag} QC] Filtrando {n_drop} muestras con {site_col}='Unknown'.")
        normalized_tensor_subjects = np.asarray(normalized_tensor_subjects)[keep]
        latent_mu_subjects = np.asarray(latent_mu_subjects)[keep]

    n_sites = int(len(np.unique(y_site)))
    if n_sites < 2:
        logger.info(f"[{fold_tag} QC] Sólo 1 clase en {site_col}. No se puede estimar fuga.")
        return None

    # features conectoma normalizado (flatten)
    N, C, R, _ = np.asarray(normalized_tensor_subjects).shape
    vect_mode_used = _resolve_vectorize_mode(np.asarray(normalized_tensor_subjects), str(vectorize_mode))
    X_conn = _vectorize_connectome_offdiag(normalized_tensor_subjects, mode=vect_mode_used)


    # features latentes
    X_lat = _ensure_2d_float(latent_mu_subjects)

    clf_site = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
    )
    n_splits = _choose_n_splits(y_site, max_splits=5)
    if n_splits < 2:
        logger.info(f"[{fold_tag} QC] Muy pocas muestras por clase para CV (n_splits={n_splits}). Saltando.")
        return None
    cv_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
  

    try:
        acc_conn = cross_val_score(clf_site, X_conn, y_site, cv=cv_inner, scoring="balanced_accuracy")
        acc_lat  = cross_val_score(clf_site, X_lat,  y_site, cv=cv_inner, scoring="balanced_accuracy")
    except Exception as e:
        logger.warning(f"[{fold_tag} QC] fallo CV scanner leakage: {e}")
        return None

    acc_site_raw_mean = float(np.mean(acc_conn))
    acc_site_raw_std  = float(np.std(acc_conn))
    acc_site_lat_mean = float(np.mean(acc_lat))
    acc_site_lat_std  = float(np.std(acc_lat))
    chance_level = float(1.0 / n_sites) if n_sites > 0 else np.nan

    # 1) CSV DETALLE (formato largo, útil para inspección humana)
    df_detail = pd.DataFrame({
        "representation": ["connectome_norm", "latent_mu"],
        "balanced_accuracy_mean": [acc_site_raw_mean, acc_site_lat_mean],
        "balanced_accuracy_std":  [acc_site_raw_std,  acc_site_lat_std],
        "n_classes": [int(n_sites)] * 2,
        "site_col": [site_col] * 2,
        "fold_tag": [fold_tag] * 2,
        "n_splits": [int(n_splits)] * 2,
        "vectorize_mode": [str(vect_mode_used)] * 2,
        "n_samples": [int(N)] * 2,
        "chance_level": [chance_level] * 2,
    })
    if out_dir is not None:
        df_detail.to_csv(out_dir / f"{fold_tag}_scanner_leakage.csv", index=False)

    # 2) CSV SUMMARY + retorno estable (1 fila, columnas estándar)
    df_summary = pd.DataFrame([{
        "fold_tag": str(fold_tag),
        "site_col": str(site_col),
        "n_sites": int(n_sites),
        "n_splits": int(n_splits),
        "n_samples": int(N),
        "vectorize_mode": str(vect_mode_used),
        "chance_level": float(chance_level),
        "acc_site_raw": float(acc_site_raw_mean),
        "acc_site_raw_std": float(acc_site_raw_std),
        "acc_site_latent": float(acc_site_lat_mean),
        "acc_site_latent_std": float(acc_site_lat_std),
    }])
    if out_dir is not None:
        df_summary.to_csv(out_dir / f"{fold_tag}_scanner_leakage_summary.csv", index=False)
    return df_summary

def evaluate_scanner_leakage_per_channel(
    metadata_df_full: pd.DataFrame,
    subject_global_indices: np.ndarray,
    normalized_tensor_subjects: np.ndarray,
    latent_mu_subjects: np.ndarray,
    out_dir: Path,
    fold_tag: str,
    *,
    channel_names: Optional[List[str]] = None,
    random_state: int = 0,
    vectorize_mode: str = DEFAULT_VECT_MODE_SYMMETRIC,
    include_all_channels_row: bool = True,
    include_latent_row: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Leakage (batch effect) por canal: estima qué tan predecible es Manufacturer/Site
    desde CADA canal del conectoma normalizado.

    Output:
      - {fold_tag}_scanner_leakage_per_channel.csv

    Filas típicas:
      - representation="connectome_norm", channel="<name>" (una por canal)
      - opcional: representation="connectome_norm_all", channel="__ALL__"
      - opcional: representation="latent_mu", channel="__LATENT__"

    Nota: en tu setting (canales simétricos, sin Granger) conviene vectorize_mode="upper".
    """
    _safe_make_out_dir(out_dir)

    # --- Sanidad tensor ---
    X = np.asarray(normalized_tensor_subjects)
    if X.ndim != 4:
        raise ValueError(f"normalized_tensor_subjects debe ser 4D [N,C,R,R], recibí shape={X.shape}")
    N, C, R, R2 = X.shape
    if R != R2:
        raise ValueError(f"Matrices no cuadradas: R={R}, R2={R2}")

    # --- columna batch (prefer Manufacturer/Vendor) ---
    site_col = _detect_site_column(metadata_df_full, prefer_manufacturer=True)
    if site_col is None:
        logger.info(f"[{fold_tag} QC] No se encontró columna Manufacturer/Vendor/Site. Saltando leakage per-channel.")
        return None
    if "tensor_idx" not in metadata_df_full.columns:
        logger.info(f"[{fold_tag} QC] No se encontró 'tensor_idx' en metadata. Saltando leakage per-channel.")
        return None

    # --- mapear tensor_idx -> etiqueta batch ---
    idx_to_site = dict(
        zip(
            metadata_df_full["tensor_idx"].values,
            metadata_df_full[site_col].astype(str).values
        )
    )
    y_site = np.array([idx_to_site.get(int(i), "Unknown") for i in subject_global_indices])

    # Filtrar Unknown si rompe estratificación (idéntico a evaluate_scanner_leakage)
    X_latent = np.asarray(latent_mu_subjects)
    if np.any(y_site == "Unknown"):
        keep = (y_site != "Unknown")
        y_site_known = y_site[keep]
        if len(np.unique(y_site_known)) >= 2:
            logger.warning(f"[{fold_tag} QC] Filtrando {int(np.sum(~keep))} muestras con sitio='Unknown' (per-channel).")
            y_site = y_site_known
            X = X[keep]
            X_latent = X_latent[keep]
            N = int(X.shape[0])

    n_classes = int(len(np.unique(y_site)))
    if n_classes < 2:
        logger.info(f"[{fold_tag} QC] Sólo 1 clase en {site_col}. No se puede estimar leakage per-channel.")
        return None

    n_splits = _choose_n_splits(y_site, max_splits=5)
    if n_splits < 2:
        logger.info(f"[{fold_tag} QC] Muy pocas muestras por clase para CV (n_splits={n_splits}). Saltando.")
        return None

    cv_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    clf_site = LogisticRegression(
        max_iter=1000,
        multi_class="auto",
        class_weight="balanced",
        solver="lbfgs"
    )

    # --- nombres de canal robustos ---
    def _ch_name(i: int) -> str:
        if channel_names and i < len(channel_names) and channel_names[i]:
            return str(channel_names[i])
        return f"ch{i}"

    rows: List[Dict[str, Any]] = []

    # --- leakage por canal (connectome_norm) ---
    for c in range(C):
        Xc = X[:, c:c+1, :, :]  # [N,1,R,R]
        X_conn_c = _vectorize_connectome_offdiag(Xc, mode=str(vectorize_mode))
        try:
            acc = cross_val_score(
                clf_site, X_conn_c, y_site, cv=cv_inner, scoring="balanced_accuracy"
            )
            rows.append({
                "representation": "connectome_norm",
                "channel": _ch_name(c),
                "channel_idx": int(c),
                "balanced_accuracy_mean": float(np.mean(acc)),
                "balanced_accuracy_std": float(np.std(acc)),
                "n_classes": int(n_classes),
                "site_col": str(site_col),
                "fold_tag": str(fold_tag),
                "n_splits": int(n_splits),
                "vectorize_mode": str(vectorize_mode),
                "n_samples": int(N),
            })
        except Exception as e:
            logger.warning(f"[{fold_tag} QC] fallo leakage canal {c} ({_ch_name(c)}): {e}")
            rows.append({
                "representation": "connectome_norm",
                "channel": _ch_name(c),
                "channel_idx": int(c),
                "balanced_accuracy_mean": np.nan,
                "balanced_accuracy_std": np.nan,
                "n_classes": int(n_classes),
                "site_col": str(site_col),
                "fold_tag": str(fold_tag),
                "n_splits": int(n_splits),
                "vectorize_mode": str(vectorize_mode),
                "n_samples": int(N),
            })

    # --- opcional: leakage usando TODOS los canales juntos (para comparar con evaluate_scanner_leakage) ---
    if include_all_channels_row:
        try:
            X_conn_all = _vectorize_connectome_offdiag(X, mode=str(vectorize_mode))
            acc_all = cross_val_score(
                clf_site, X_conn_all, y_site, cv=cv_inner, scoring="balanced_accuracy"
            )
            rows.append({
                "representation": "connectome_norm_all",
                "channel": "__ALL__",
                "channel_idx": -1,
                "balanced_accuracy_mean": float(np.mean(acc_all)),
                "balanced_accuracy_std": float(np.std(acc_all)),
                "n_classes": int(n_classes),
                "site_col": str(site_col),
                "fold_tag": str(fold_tag),
                "n_splits": int(n_splits),
                "vectorize_mode": str(vectorize_mode),
                "n_samples": int(N),
            })
        except Exception as e:
            logger.warning(f"[{fold_tag} QC] fallo leakage all-channels: {e}")

    # --- opcional: leakage en latente (mu) como referencia ---
    if include_latent_row:
        try:
            X_lat = _ensure_2d_float(X_latent)
            acc_lat = cross_val_score(
                clf_site, X_lat, y_site, cv=cv_inner, scoring="balanced_accuracy"
            )
            rows.append({
                "representation": "latent_mu",
                "channel": "__LATENT__",
                "channel_idx": -2,
                "balanced_accuracy_mean": float(np.mean(acc_lat)),
                "balanced_accuracy_std": float(np.std(acc_lat)),
                "n_classes": int(n_classes),
                "site_col": str(site_col),
                "fold_tag": str(fold_tag),
                "n_splits": int(n_splits),
                "vectorize_mode": "NA",
                "n_samples": int(N),
            })
        except Exception as e:
            logger.warning(f"[{fold_tag} QC] fallo leakage latent_mu (per-channel): {e}")

    df = pd.DataFrame(rows)
    # que el canal dominante quede arriba
    df = df.sort_values(
        by=["representation", "balanced_accuracy_mean"],
        ascending=[True, False],
        na_position="last",
    )
    df.to_csv(out_dir / f"{fold_tag}_scanner_leakage_per_channel.csv", index=False)
    return df



def _ensure_2d_float(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x

def _choose_n_splits(y, max_splits=5):
    """Elige n_splits para StratifiedKFold basado en la clase minoritaria."""
    _, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    # si min_count < 2, no se puede hacer StratifiedKFold válido
    if int(min_count) < 2:
        return 0
    return int(min(max_splits, int(min_count)))

def _impute_numeric(vals):
    vals = np.asarray(vals)
    # Reemplazar None/NaN con la media o 0
    vals = pd.to_numeric(vals, errors='coerce')
    mask = np.isnan(vals)
    if mask.any():
        mean_val = np.nanmean(vals) if not np.all(mask) else 0.0
        vals[mask] = mean_val
    return vals

def _factorize(vals):
    return pd.factorize(vals)

# --- Funciones de MI y Estadística ---

def compute_active_units(Z, var_eps=1e-4):
    # Varianza por dimensión
    vars_ = np.var(Z, axis=0)
    active_mask = vars_ > var_eps
    return {
        "n_active": int(np.sum(active_mask)),
        "frac_active": float(np.mean(active_mask)),
        "vars": vars_
    }

def estimate_total_correlation_gaussian(Z, ridge=1e-6):
    """Estima Total Correlation asumiendo normalidad: sum(log(var_i)) - log(det(Cov))."""
    C = np.cov(Z, rowvar=False)
    # Ridge para estabilidad
    C += np.eye(C.shape[0]) * ridge
    try:
        _, logdet = np.linalg.slogdet(C)
        vars_ = np.diag(C)
        sum_log_vars = np.sum(np.log(vars_))
        tc = 0.5 * (sum_log_vars - logdet)
        return max(0.0, tc)
    except:
        return np.nan

def _mi_classif_per_dim(Z, y, n_neighbors=3, random_state=0):
    # mutual_info_classif maneja matriz X, vector y
    # Lo hacemos todo de una vez para eficiencia
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y).ravel()
    N = int(Z.shape[0])
    if N < 3:
        return np.full(Z.shape[1], np.nan, dtype=float)
    # mutual_info_* requiere n_neighbors < N
    k = int(max(1, min(int(n_neighbors), N - 1)))
    try:
        return mutual_info_classif(Z, y, discrete_features=False, n_neighbors=k, random_state=random_state)
    except Exception as e:
        logger.warning(f"[MI classif] fallo mutual_info_classif (N={N}, k={k}): {e}")
        return np.full(Z.shape[1], np.nan, dtype=float)

def _mi_regress_per_dim(Z, y, n_neighbors=3, random_state=0):
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    N = int(Z.shape[0])
    if N < 3:
        return np.full(Z.shape[1], np.nan, dtype=float)
    k = int(max(1, min(int(n_neighbors), N - 1)))
    try:
        return mutual_info_regression(Z, y, discrete_features=False, n_neighbors=k, random_state=random_state)
    except Exception as e:
        logger.warning(f"[MI regress] fallo mutual_info_regression (N={N}, k={k}): {e}")
        return np.full(Z.shape[1], np.nan, dtype=float)


# --- Funciones de Distribución / Plots ---

def _stage_stats_df(tensor, channel_names, final_activation=None):
    """Calcula media/std/min/max por canal."""
    # tensor: [N, Channels, H, W]
    rows = []
    for c_idx, c_name in enumerate(channel_names):
        if c_idx >= tensor.shape[1]: break
        A = tensor[:, c_idx, :, :]               # [N,R,R]
        R = A.shape[-1]
        off = ~np.eye(R, dtype=bool)
        data_c = A[:, off].ravel()
        
        # Si hay activación final (ej. tanh), aplicarla para ver stats reales de salida
        if final_activation == 'tanh':
            # Nota: el tensor de recon ya suele venir activado del modelo, 
            # pero si fuera logits, aquí se aplicaría. Asumimos que ya viene procesado.
            pass
            
        rows.append({
            "channel": c_name,
            "mean": np.mean(data_c),
            "std": np.std(data_c),
            "min": np.min(data_c),
            "max": np.max(data_c),
            "p05": np.percentile(data_c, 5),
            "p95": np.percentile(data_c, 95)
        })
    return pd.DataFrame(rows)

def _save_overlay_hist_per_channel(raw, norm, rec, channel_names, out_dir, prefix):
    """Guarda histogramas superpuestos por canal."""
    # raw, norm, rec: [N, C, H, W]
    R = raw.shape[-1]
    off_diag_mask = ~np.eye(R, dtype=bool)    
    for c_idx, c_name in enumerate(channel_names):
        if c_idx >= raw.shape[1]: break
        
        # Fix: filtrar diagonal (zeros) para evitar sesgo en histograma
        r_vals = raw[:, c_idx][:, off_diag_mask].flatten()
        n_vals = norm[:, c_idx][:, off_diag_mask].flatten()
        rec_vals = rec[:, c_idx][:, off_diag_mask].flatten()
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Raw vs Recon (si tienen escalas compatibles, a veces no)
        # Por seguridad, ploteamos densidades separadas
        ax[0].hist(r_vals, bins=50, density=True, alpha=0.5, label='Raw', color='gray')
        ax[0].set_title(f"{c_name} (Raw)")
        
        ax[1].hist(n_vals, bins=50, density=True, alpha=0.5, label='Norm', color='blue')
        ax[1].hist(rec_vals, bins=50, density=True, alpha=0.5, label='Recon', color='red')
        ax[1].set_title(f"{c_name} (Norm vs Recon)")
        ax[1].legend()

        # Plot 3: Scatter plot simple de media por sujeto
        # IMPORTANTE: usar off-diagonal para no sesgar con la diagonal (0)
        mu_n = norm[:, c_idx][:, off_diag_mask].mean(axis=1)
        mu_r = rec[:, c_idx][:, off_diag_mask].mean(axis=1)
        ax[2].scatter(mu_n, mu_r, alpha=0.5, s=10)
        ax[2].set_xlabel("Mean Input (Norm)")
        ax[2].set_ylabel("Mean Recon")
        ax[2].set_title("Input vs Recon (Means)")
        
        plt.tight_layout()
        safe_name = "".join(x for x in c_name if x.isalnum() or x in "_-")
        plt.savefig(out_dir / f"{prefix}_hist_{safe_name}.png")
        plt.close(fig)