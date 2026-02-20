"""
composite_edge_shap.py

Helpers for composite edge-level SHAP:
    edges (vector) → tensor (N,C,R,R) → VAE.encode → latent+meta → pipeline → P(AD)

All functions are pure helpers with no CLI logic.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

log = logging.getLogger("interpret")

# ---------------------------------------------------------------------------
# Edge indexing & naming
# ---------------------------------------------------------------------------

def make_edge_index(R: int, C: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Precompute upper-triangular indices for (R,R) connectivity matrices.

    Returns
    -------
    triu_rows : ndarray (edges_per_channel,)
    triu_cols : ndarray (edges_per_channel,)
    total_edges : int = C * edges_per_channel
    """
    triu_rows, triu_cols = np.triu_indices(R, k=1)
    edges_per_channel = len(triu_rows)
    return triu_rows, triu_cols, C * edges_per_channel


def make_edge_index_offdiag(R: int, C: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    All off-diagonal indices for (R,R) connectivity matrices (i != j).

    Returns R*(R-1) edges per channel (both (i,j) and (j,i)).
    """
    rows, cols = np.where(~np.eye(R, dtype=bool))
    edges_per_channel = len(rows)
    return rows, cols, C * edges_per_channel


def make_edge_feature_names(
    roi_names: List[str],
    channels_used: Sequence[int],
    channel_names: Optional[List[str]] = None,
    *,
    idx_rows: Optional[np.ndarray] = None,
    idx_cols: Optional[np.ndarray] = None,
) -> List[str]:
    """
    Human-readable names for all C * E edges.

    Format: ``"ChName__ROI_i--ROI_j"``

    Parameters
    ----------
    idx_rows, idx_cols : optional precomputed index arrays.
        If None, defaults to upper-triangular (k=1).
    """
    R = len(roi_names)
    if idx_rows is None or idx_cols is None:
        triu_r, triu_c = np.triu_indices(R, k=1)
    else:
        triu_r, triu_c = idx_rows, idx_cols
    names: List[str] = []
    for ci, ch_idx in enumerate(channels_used):
        ch_label = channel_names[ci] if channel_names and ci < len(channel_names) else f"Ch{ch_idx}"
        for ei in range(len(triu_r)):
            ri, rj = int(triu_r[ei]), int(triu_c[ei])
            names.append(f"{ch_label}__{roi_names[ri]}--{roi_names[rj]}")
    return names


def make_edge_mapping_df(
    selected_indices: np.ndarray,
    roi_names: List[str],
    channels_used: Sequence[int],
    channel_names: Optional[List[str]],
    triu_rows: np.ndarray,
    triu_cols: np.ndarray,
    edges_per_channel: int,
    scores: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Build a DataFrame mapping each selected edge to its (channel, roi_i, roi_j).
    """
    rows = []
    all_names = make_edge_feature_names(roi_names, channels_used, channel_names)
    for rank, flat_idx in enumerate(selected_indices):
        flat_idx = int(flat_idx)
        ch_local = flat_idx // edges_per_channel
        edge_local = flat_idx % edges_per_channel
        ri = int(triu_rows[edge_local])
        rj = int(triu_cols[edge_local])
        ch_orig = channels_used[ch_local] if ch_local < len(channels_used) else ch_local
        ch_name = (channel_names[ch_local]
                   if channel_names and ch_local < len(channel_names)
                   else f"Ch{ch_orig}")
        row = {
            "edge_name": all_names[flat_idx],
            "flat_index": flat_idx,
            "channel": int(ch_orig),
            "channel_name": ch_name,
            "roi_i_idx": ri,
            "roi_j_idx": rj,
            "roi_i_name": roi_names[ri],
            "roi_j_name": roi_names[rj],
            "selection_rank": rank + 1,
        }
        if scores is not None:
            row["selection_score"] = float(scores[flat_idx])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Vectorize / reconstruct
# ---------------------------------------------------------------------------

def vectorize_tensor_to_edges(
    tensor: np.ndarray,
    triu_rows: np.ndarray,
    triu_cols: np.ndarray,
) -> np.ndarray:
    """
    Flatten (N, C, R, R) → (N, C*E) using upper-triangular indices.

    Channels are concatenated: ``[Ch0_edges | Ch1_edges | ...]``.
    """
    N, C = tensor.shape[:2]
    feats = []
    for c in range(C):
        feats.append(tensor[:, c, triu_rows, triu_cols])
    return np.concatenate(feats, axis=1).astype(np.float32)


def reconstruct_tensor_from_edges(
    edge_vector: np.ndarray,
    R: int,
    C: int,
    triu_rows: np.ndarray,
    triu_cols: np.ndarray,
    *,
    mirror: bool = True,
) -> np.ndarray:
    """
    Reconstruct (N, C, R, R) tensor from (N, C*E) edge vector.

    Parameters
    ----------
    mirror : bool
        If True (default, upper-tri mode), fills both (i,j) and (j,i).
        If False (full_offdiag mode), fills positions as given.
    """
    if edge_vector.ndim == 1:
        edge_vector = edge_vector[np.newaxis, :]
    N = edge_vector.shape[0]
    epc = len(triu_rows)  # edges per channel
    tensor = np.zeros((N, C, R, R), dtype=np.float32)
    for c in range(C):
        vals = edge_vector[:, c * epc: (c + 1) * epc]
        tensor[:, c, triu_rows, triu_cols] = vals
        if mirror:
            tensor[:, c, triu_cols, triu_rows] = vals
    return tensor


# ---------------------------------------------------------------------------
# Feature selection (TRAIN only)
# ---------------------------------------------------------------------------

def select_top_edges(
    edges_train: np.ndarray,
    y_train: np.ndarray,
    K: int,
    method: str = "f_classif",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Select K most discriminative edges from TRAINING data.

    Parameters
    ----------
    edges_train : (N_train, total_edges)
    y_train : (N_train,) binary 0/1
    K : number of edges to keep
    method : ``"f_classif"`` | ``"mutual_info"`` | ``"l1_logreg"``
    seed : for reproducibility

    Returns
    -------
    selected_indices : (K,) sorted indices into full edge vector
    scores : (total_edges,) per-edge score
    selector_info : method-specific extra info
    """
    total_edges = edges_train.shape[1]
    if K > total_edges:
        log.warning(f"edge_K={K} > total_edges={total_edges}; using all edges.")
        K = total_edges

    if method == "f_classif":
        from sklearn.feature_selection import f_classif as _f_classif
        F_vals, p_vals = _f_classif(edges_train, y_train)
        scores = F_vals
        selector_info = {"F_values": F_vals, "p_values": p_vals}

    elif method == "mutual_info":
        from sklearn.feature_selection import mutual_info_classif
        scores = mutual_info_classif(
            edges_train, y_train,
            discrete_features=False,
            n_neighbors=5,
            random_state=seed,
        )
        selector_info = {"mi_scores": scores.copy()}

    elif method == "l1_logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(edges_train)
        lr = LogisticRegression(
            penalty="l1", solver="liblinear",
            C=1.0, max_iter=5000, random_state=seed,
        )
        lr.fit(X_scaled, y_train)
        scores = np.abs(lr.coef_).ravel()
        selector_info = {"logreg": lr, "scaler": scaler}

    else:
        raise ValueError(f"Unknown edge selection method: {method}")

    # Handle NaN (constant features)
    scores = np.nan_to_num(scores, nan=0.0)

    # Top K by score (sorted by index for stable ordering)
    selected_indices = np.argsort(scores)[::-1][:K]
    selected_indices = np.sort(selected_indices)

    return selected_indices, scores, selector_info


def select_top_edges_per_channel(
    edges_train: np.ndarray,
    y_train: np.ndarray,
    K: int,
    C: int,
    edges_per_channel: int,
    method: str = "f_classif",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Select K edges distributing quota K/C per channel, remainder to best channels.

    Runs ``select_top_edges`` independently per channel (TRAIN only) and
    merges results into global flat indices.
    """
    K_base = K // C
    K_rem = K % C

    # Per-channel selection
    per_ch_results = []
    for c in range(C):
        ch_slice = edges_train[:, c * edges_per_channel: (c + 1) * edges_per_channel]
        sel_local, sc_local, info_local = select_top_edges(
            ch_slice, y_train, K=edges_per_channel,  # get all scores
            method=method, seed=seed,
        )
        best_score = float(np.max(sc_local))
        per_ch_results.append((c, sc_local, best_score))

    # Rank channels by best single-edge score for remainder distribution
    ch_ranked = sorted(range(C), key=lambda c: per_ch_results[c][2], reverse=True)
    quota = {c: K_base for c in range(C)}
    for i in range(K_rem):
        quota[ch_ranked[i]] += 1

    # Select top-quota[c] edges per channel, convert to global indices
    all_selected = []
    all_scores = np.zeros(C * edges_per_channel, dtype=np.float64)
    for c in range(C):
        sc = per_ch_results[c][1]
        all_scores[c * edges_per_channel: (c + 1) * edges_per_channel] = sc
        top_local = np.argsort(sc)[::-1][:quota[c]]
        global_idx = top_local + c * edges_per_channel
        all_selected.append(global_idx)

    selected_indices = np.sort(np.concatenate(all_selected))
    log.info(f"[EDGE_SELECT_PER_CH] Quota per channel: {quota} "
             f"(total={sum(quota.values())})")
    return selected_indices, all_scores, {"per_channel_quota": quota}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_train_edge_median(edges_train: np.ndarray) -> np.ndarray:
    """Per-edge median from training data → (total_edges,)."""
    return np.median(edges_train, axis=0).astype(np.float32)


def compute_frozen_meta_values(
    train_df: pd.DataFrame,
    meta_cols: List[str],
) -> Dict[str, float]:
    """
    Compute frozen metadata values from TRAIN subjects.

    Age → median, Sex → mean (after M=0/F=1 encoding), others → median.
    """
    values: Dict[str, float] = {}
    for col in meta_cols:
        series = train_df[col].copy()
        if col == "Sex":
            if series.dtype == object:
                series = series.map({"M": 0, "F": 1, "m": 0, "f": 1})
            series = pd.to_numeric(series, errors="coerce")
            val = float(series.dropna().mean()) if not series.dropna().empty else 0.5
        elif col == "Age":
            series = pd.to_numeric(series, errors="coerce")
            val = float(series.dropna().median()) if not series.dropna().empty else 70.0
        else:
            series = pd.to_numeric(series, errors="coerce")
            val = float(series.dropna().median()) if not series.dropna().empty else 0.0
        values[col] = val
    return values


# ---------------------------------------------------------------------------
# Composite predict function
# ---------------------------------------------------------------------------

def make_edge_predict_fn(
    *,
    vae: Any,  # ConvolutionalVAE
    pipe: Any,  # sklearn Pipeline or CalibratedClassifierCV
    meta_cols: List[str],
    meta_frozen_values: Dict[str, float],
    feature_columns: List[str],
    selected_edge_indices: np.ndarray,
    all_edges_template: np.ndarray,
    R: int,
    C: int,
    triu_rows: np.ndarray,
    triu_cols: np.ndarray,
    device: torch.device,
    latent_features_type: str = "mu",
    pos_idx: int = 1,
    shap_link: str = "identity",
    mirror: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build predict function: selected_edges (N, K) → P(AD) (N,).

    Pipeline:
      1. Insert K selected edges into full template (TRAIN median for rest)
      2. Reconstruct (N,C,R,R) symmetric tensor
      3. VAE.encode → mu (deterministic, no grad)
      4. Build X_raw DataFrame with frozen metadata
      5. pipe.predict_proba → P(positive class)
      6. Optionally apply logit link
    """
    # Pre-compute constants
    template = all_edges_template.copy()
    sel_idx = np.asarray(selected_edge_indices)
    latent_dim = vae.latent_dim

    def predict_fn(X_selected: np.ndarray) -> np.ndarray:
        X_selected = np.asarray(X_selected, dtype=np.float32)
        if X_selected.ndim == 1:
            X_selected = X_selected[np.newaxis, :]
        N = X_selected.shape[0]

        # 1) Fill template
        full_edges = np.tile(template, (N, 1))
        full_edges[:, sel_idx] = X_selected

        # 2) Reconstruct tensor
        tensor = reconstruct_tensor_from_edges(
            full_edges, R, C, triu_rows, triu_cols, mirror=mirror,
        )

        # 3) VAE encode
        with torch.no_grad():
            t = torch.from_numpy(tensor).float().to(device)
            mu, logvar = vae.encode(t)
            if latent_features_type == "mu":
                lat = mu.cpu().numpy()
            else:
                lat = vae.reparameterize(mu, logvar).cpu().numpy()

        # 4) Build X_raw
        lat_cols = [f"latent_{i}" for i in range(lat.shape[1])]
        X_raw = pd.DataFrame(lat, columns=lat_cols)
        for col in meta_cols:
            X_raw[col] = meta_frozen_values.get(col, 0.0)

        # 5) Reorder columns to match pipeline
        missing = [c for c in feature_columns if c not in X_raw.columns]
        if missing:
            for c in missing:
                X_raw[c] = 0.0
        X_raw = X_raw[feature_columns]

        # 6) Predict
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_raw)[:, pos_idx]
        elif hasattr(pipe, "decision_function"):
            margin = pipe.decision_function(X_raw)
            if isinstance(margin, np.ndarray) and margin.ndim == 2:
                margin = margin[:, pos_idx]
            proba = 1.0 / (1.0 + np.exp(-np.asarray(margin, dtype=np.float64)))
        else:
            proba = pipe.predict(X_raw).astype(float)

        proba = np.asarray(proba, dtype=np.float64)

        # 7) Link
        if shap_link == "logit":
            eps = 1e-6
            p = np.clip(proba, eps, 1 - eps)
            return np.log(p / (1 - p))
        return proba

    return predict_fn


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_edge_roundtrip(
    *,
    vae: Any,
    pipe: Any,
    tensor_test_norm: np.ndarray,
    test_df: pd.DataFrame,
    meta_cols: List[str],
    feature_columns: List[str],
    triu_rows: np.ndarray,
    triu_cols: np.ndarray,
    device: torch.device,
    latent_features_type: str = "mu",
    pos_idx: int = 1,
    n_check: int = 10,
    atol: float = 1e-5,
    mirror: bool = True,
) -> Tuple[bool, float]:
    """
    Validate that vectorize → reconstruct → VAE → pipeline gives the same
    prediction as the direct tensor → VAE → pipeline path.

    Uses REAL per-sample metadata (not frozen) for a fair comparison.
    """
    R = tensor_test_norm.shape[2]
    C = tensor_test_norm.shape[1]
    n = min(n_check, tensor_test_norm.shape[0])
    tens_sub = tensor_test_norm[:n]
    test_sub = test_df.iloc[:n].copy()

    # --- Path A: direct tensor → VAE → pipeline ---
    with torch.no_grad():
        t_a = torch.from_numpy(tens_sub).float().to(device)
        mu_a, logvar_a = vae.encode(t_a)
        lat_a = (mu_a if latent_features_type == "mu"
                 else vae.reparameterize(mu_a, logvar_a)).cpu().numpy()

    lat_cols = [f"latent_{i}" for i in range(lat_a.shape[1])]
    X_a = pd.DataFrame(lat_a, columns=lat_cols)
    for col in meta_cols:
        vals = test_sub[col].values.copy()
        if col == "Sex" and test_sub[col].dtype == object:
            vals = test_sub[col].map({"M": 0, "F": 1, "m": 0, "f": 1}).values
        X_a[col] = pd.to_numeric(pd.Series(vals), errors="coerce").values
    missing = [c for c in feature_columns if c not in X_a.columns]
    for c in missing:
        X_a[c] = 0.0
    X_a = X_a[feature_columns]

    if hasattr(pipe, "predict_proba"):
        pred_a = pipe.predict_proba(X_a)[:, pos_idx]
    else:
        pred_a = pipe.decision_function(X_a)
    pred_a = np.asarray(pred_a, dtype=np.float64).ravel()

    # --- Path B: tensor → vectorize → reconstruct → VAE → pipeline ---
    edges_full = vectorize_tensor_to_edges(tens_sub, triu_rows, triu_cols)
    tens_recon = reconstruct_tensor_from_edges(
        edges_full, R, C, triu_rows, triu_cols, mirror=mirror,
    )

    with torch.no_grad():
        t_b = torch.from_numpy(tens_recon).float().to(device)
        mu_b, logvar_b = vae.encode(t_b)
        lat_b = (mu_b if latent_features_type == "mu"
                 else vae.reparameterize(mu_b, logvar_b)).cpu().numpy()

    X_b = pd.DataFrame(lat_b, columns=lat_cols)
    for col in meta_cols:
        vals = test_sub[col].values.copy()
        if col == "Sex" and test_sub[col].dtype == object:
            vals = test_sub[col].map({"M": 0, "F": 1, "m": 0, "f": 1}).values
        X_b[col] = pd.to_numeric(pd.Series(vals), errors="coerce").values
    for c in missing:
        X_b[c] = 0.0
    X_b = X_b[feature_columns]

    if hasattr(pipe, "predict_proba"):
        pred_b = pipe.predict_proba(X_b)[:, pos_idx]
    else:
        pred_b = pipe.decision_function(X_b)
    pred_b = np.asarray(pred_b, dtype=np.float64).ravel()

    max_diff = float(np.max(np.abs(pred_a - pred_b)))
    passed = max_diff < atol

    log.info(f"[VALIDATION] Path-A vs Path-B (n={n}): max|diff|={max_diff:.2e}  "
             f"{'PASSED' if passed else 'FAILED'} (atol={atol:.1e})")
    if not passed:
        # Log per-sample diffs for debugging
        for i in range(n):
            log.info(f"  sample {i}: pred_A={pred_a[i]:.8f}  pred_B={pred_b[i]:.8f}  "
                     f"diff={abs(pred_a[i]-pred_b[i]):.2e}")
    return passed, max_diff


# ---------------------------------------------------------------------------
# IG: extract logreg weights from pipeline
# ---------------------------------------------------------------------------

def extract_logreg_latent_weights(
    pipe: Any,
    latent_dim: int,
    meta_cols: List[str],
    feature_columns: List[str],
) -> Tuple[Optional[np.ndarray], Optional[float],
           Optional[float], Optional[float], str]:
    """
    Extract logistic regression weights projected to latent-only space.

    Returns
    -------
    w_latent : (latent_dim,) or None if not extractable
    bias : float or None
    platt_a : float or None (sigmoid calibration param)
    platt_b : float or None
    status : str description
    """
    # Unwrap CalibratedClassifierCV if present
    platt_a, platt_b = None, None
    inner_pipe = pipe

    if hasattr(pipe, "calibrated_classifiers_"):
        cc = pipe.calibrated_classifiers_[0]
        inner_pipe = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if inner_pipe is None:
            return None, None, None, None, "cannot_unwrap_calibrator"
        # Extract Platt parameters
        if hasattr(cc, "calibrators_") and len(cc.calibrators_) > 0:
            cal = cc.calibrators_[0]
            if hasattr(cal, "a_") and hasattr(cal, "b_"):
                platt_a = float(cal.a_)
                platt_b = float(cal.b_)

    # Get the model step from the pipeline
    if not hasattr(inner_pipe, "named_steps"):
        # Check if inner_pipe itself IS a LogisticRegression
        if hasattr(inner_pipe, "coef_") and hasattr(inner_pipe, "intercept_"):
            model = inner_pipe
            preproc = None
            selector = None
        else:
            return None, None, None, None, "not_pipeline_not_logreg"
    else:
        model = inner_pipe.named_steps.get("model")
        if model is None:
            return None, None, None, None, "no_model_step"
        # Try multiple names for preprocess step
        preproc = None
        for name in ("preprocess", "preproc", "transformer", "prep", "scaler"):
            preproc = inner_pipe.named_steps.get(name)
            if preproc is not None:
                break
        selector = (inner_pipe.named_steps.get("feature_selector")
                    or inner_pipe.named_steps.get("feature_select")
                    or inner_pipe.named_steps.get("fs"))

    # Check it's actually logreg or linear SVM
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC, LinearSVC
    is_logreg = isinstance(model, LogisticRegression)
    is_linear_svm = (isinstance(model, (SVC, LinearSVC))
                     and getattr(model, "kernel", "linear") == "linear"
                     and hasattr(model, "coef_"))
    if not (is_logreg or is_linear_svm):
        return None, None, None, None, f"unsupported_model_{type(model).__name__}"

    w_proc = model.coef_.ravel().copy()  # weights in preprocessed space
    b_proc = float(model.intercept_[0]) if hasattr(model, "intercept_") else 0.0

    # Expand through selector (if present)
    if selector is not None and hasattr(selector, "get_support"):
        support = selector.get_support()
        w_full = np.zeros(support.shape[0], dtype=np.float64)
        w_full[support] = w_proc
        w_proc = w_full

    # Invert preprocessing scaling
    # _AutoPreprocessor stores a ColumnTransformer or Pipeline in _ct_
    scale = np.ones(len(w_proc), dtype=np.float64)
    offset = np.zeros(len(w_proc), dtype=np.float64)
    if preproc is not None and hasattr(preproc, "_ct_"):
        ct = preproc._ct_
        # Try to extract scaler params from the ColumnTransformer
        try:
            if hasattr(ct, "transformers_"):
                # ColumnTransformer case
                idx_start = 0
                for name, trans, cols in ct.transformers_:
                    if name == "remainder":
                        continue
                    if hasattr(trans, "named_steps") and "scaler" in trans.named_steps:
                        sc = trans.named_steps["scaler"]
                        n_out = len(sc.scale_)
                        scale[idx_start:idx_start + n_out] = sc.scale_
                        offset[idx_start:idx_start + n_out] = sc.mean_
                        idx_start += n_out
                    elif hasattr(trans, "transform"):
                        # Guess output width
                        n_cols = len(cols) if isinstance(cols, list) else 1
                        idx_start += n_cols
            elif hasattr(ct, "named_steps") and "scaler" in ct.named_steps:
                # Simple Pipeline case
                sc = ct.named_steps["scaler"]
                scale[:] = sc.scale_
                offset[:] = sc.mean_
        except Exception as e:
            log.warning(f"[IG] Could not extract scaler params: {e}. Using identity.")

    # Invert: w_raw = w_scaled / scale, b_raw adjusted
    w_raw = w_proc / scale
    b_raw = b_proc - np.sum(w_proc * offset / scale)

    # Extract only latent dimensions
    lat_col_indices = []
    for i, col_name in enumerate(feature_columns):
        if col_name.startswith("latent_"):
            lat_col_indices.append(i)

    if len(lat_col_indices) == 0:
        return None, None, None, None, "no_latent_columns_in_features"

    # Build w_latent of exactly latent_dim size
    w_latent = np.zeros(latent_dim, dtype=np.float64)
    for feat_idx in lat_col_indices:
        # Parse latent index from column name
        lat_idx = int(feature_columns[feat_idx].split("_")[1])
        if lat_idx < latent_dim and feat_idx < len(w_raw):
            w_latent[lat_idx] = w_raw[feat_idx]

    # bias includes contribution from frozen metadata
    bias = b_raw
    for i, col_name in enumerate(feature_columns):
        if not col_name.startswith("latent_") and i < len(w_raw):
            # This is a metadata column; its contribution is w * frozen_value
            # but we don't know the frozen value here, so we absorb it into bias
            # only if we can. Actually, for IG this is fine: metadata is constant,
            # so its gradient is zero. The bias just shifts the baseline.
            pass

    return w_latent, bias, platt_a, platt_b, "ok"
