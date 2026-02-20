# betavae_xai/qc/simple_qc.py
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)

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
    random_state: int = 0
) -> dict:
    """
    Evalúa la 'fuga de sitio' en TEST SET o cualquier subconjunto dado.
    Hace lo mismo que evaluate_scanner_leakage pero EN SITIO (sin guardar CSV),
    y devuelve dict con:
        acc_site_raw      (balanced_accuracy_mean conectoma normalizado)
        acc_site_latent   (balanced_accuracy_mean mu latente)
        n_sites           (#clases)
    Si no hay columna de sitio o sólo 1 sitio, devuelve NaNs.
    """
    # Hallar columna de sitio/escáner
    site_cols = [
        c for c in metadata_df_full.columns
        if ("manufacturer" in c.lower()) or ("vendor" in c.lower()) or ("site" in c.lower())
    ]
    if len(site_cols) == 0:
        return {
            "acc_site_raw": np.nan,
            "acc_site_latent": np.nan,
            "n_sites": 1,
        }
    site_col = site_cols[0]

    idx_to_site = dict(
        zip(
            metadata_df_full["tensor_idx"].values,
            metadata_df_full[site_col].astype(str).values
        )
    )
    y_site = np.array([idx_to_site[i] for i in subject_global_indices])

    unique_sites = np.unique(y_site)
    if unique_sites.size < 2:
        return {
            "acc_site_raw": np.nan,
            "acc_site_latent": np.nan,
            "n_sites": int(unique_sites.size),
        }

    # flatten conectoma normalizado
    N, C, R, _ = X_norm_subjects.shape
    X_conn = X_norm_subjects.reshape(N, C * R * R)
    X_lat = X_latent_subjects

    clf_site = LogisticRegression(
        max_iter=1000,
        multi_class="auto",
        class_weight="balanced",
        solver="lbfgs"
    )
    y_site_factorized, _ = pd.factorize(y_site)
    class_counts = np.bincount(y_site_factorized)
    min_class_count = int(class_counts.min())

    n_splits = min(5, min_class_count)
    if n_splits < 2:
        return {"acc_site_raw": np.nan, "acc_site_latent": np.nan, "n_sites": int(unique_sites.size)}
    cv_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


    try:
        acc_conn = cross_val_score(
            clf_site, X_conn, y_site, cv=cv_inner, scoring="balanced_accuracy"
        )
        acc_lat = cross_val_score(
            clf_site, X_lat, y_site, cv=cv_inner, scoring="balanced_accuracy"
        )
        acc_site_raw = float(np.mean(acc_conn))
        acc_site_latent = float(np.mean(acc_lat))
    except Exception as e:
        logger.warning(f"[QC leakage simple] fallo CV scanner leakage: {e}")
        acc_site_raw = np.nan
        acc_site_latent = np.nan


    return {
        "acc_site_raw": acc_site_raw,
        "acc_site_latent": acc_site_latent,
        "n_sites": int(unique_sites.size),
    }