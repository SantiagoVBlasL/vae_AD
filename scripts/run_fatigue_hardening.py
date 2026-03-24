#!/usr/bin/env python3
"""
Hardening Analyses for Fatigue Connectome Baselines
====================================================

Three adversarial analyses to challenge the initial baseline findings
before any fatigue-related connectomic signal can be claimed:

1. **Max-permutation correction** — corrects for testing 7 channels.
   Builds the null distribution of  max(AUC across channels)  so that
   the corrected p-value accounts for channel selection.
   Uses within-fold PCA (same estimand as the main-run pipeline).

2. **Incremental value over metadata** (Target D only) —
   Does the connectome add predictive value beyond Age + Sex?
   Uses paired bootstrap on OOF predictions.

3. **Sex-matched subsampling** (Target D only) —
   Does the signal persist when sex is held constant?
   Repeated balanced subsamples within females.

4. **Metadata decomposition** (Target D only) —
   Age-only vs Sex-only vs Age+Sex predictive value.
   Identifies whether sex is the dominant metadata confound.

5. **Power / precision analysis** (Target D only) —
   Bootstrap SE, approximate power at current n, and rough n
   needed for 80% power.  Supports interpretation of negative
   incremental-value and female-only results as power-limited.

Reads raw data from:  data/
Writes to:            results/fatigue_connectome_baselines/Tables/hardening_*
                      results/fatigue_connectome_baselines/Figures/

Usage
-----
Full run:
    python3 scripts/run_fatigue_hardening.py

Smoke test:
    python3 scripts/run_fatigue_hardening.py --smoke_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time as _time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm as _norm
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    LeaveOneOut,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fatigue_hardening")

# ── Constants ────────────────────────────────────────────────────────
CHANNEL_NAMES: List[str] = [
    "Pearson_OMST",
    "Pearson_FisherZ",
    "MI_KNN",
    "dFC_AbsDiffMean",
    "dFC_StdDev",
    "DistanceCorr",
    "Granger",
]

_TENSOR_DIR = (
    "COVID_AAL3_Tensor_v1_AAL3_131ROIs_OMST_GCE_Signed_"
    "GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned"
)
DEFAULT_TENSOR = (
    PROJECT_ROOT / "data" / _TENSOR_DIR
    / f"GLOBAL_TENSOR_from_{_TENSOR_DIR}.npz"
)
DEFAULT_META = PROJECT_ROOT / "data" / "SubjectsData_AAL3_COVID.csv"
DEFAULT_OUT  = PROJECT_ROOT / "results" / "fatigue_connectome_baselines"

TENSOR_ID_COL = "ID"
FAS = "CategoriaFAS"


# ═════════════════════════════════════════════════════════════════════
#  Confound Residualizer  (same as main script)
# ═════════════════════════════════════════════════════════════════════
class ConfoundResidualizer(BaseEstimator, TransformerMixin):
    """Remove linear confound effects; last *n_confounds* columns of X."""

    def __init__(self, n_confounds: int = 2):
        self.n_confounds = n_confounds

    def fit(self, X, y=None):
        feats = X[:, : -self.n_confounds]
        confs = X[:, -self.n_confounds :]
        self.reg_ = LinearRegression().fit(confs, feats)
        return self

    def transform(self, X):
        feats = X[:, : -self.n_confounds]
        confs = X[:, -self.n_confounds :]
        return feats - self.reg_.predict(confs)


# ═════════════════════════════════════════════════════════════════════
#  Data loading  (mirrors main script)
# ═════════════════════════════════════════════════════════════════════
def load_covid_data(tensor_path: Path, metadata_path: Path):
    npz = np.load(str(tensor_path), allow_pickle=True)
    tensor = npz["global_tensor_data"]
    sids   = npz["subject_ids"]
    tdf  = pd.DataFrame({TENSOR_ID_COL: sids, "tensor_idx": np.arange(len(sids))})
    meta = pd.read_csv(metadata_path)
    df   = tdf.merge(meta, on=TENSOR_ID_COL, how="inner")
    group_col = "Grupo" if "Grupo" in df.columns else "ResearchGroup"
    log.info("Loaded tensor %s, merged %d subjects", tensor.shape, len(df))
    return tensor, df, group_col


def build_target_D(df, group_col):
    mask = (df[group_col] == "COVID") & df[FAS].isin(["FATIGA EXTREMA", "NO HAY FATIGA"])
    sub = df.loc[mask].copy()
    sub["y"] = (sub[FAS] == "FATIGA EXTREMA").astype(int)
    return sub.reset_index(drop=True), sub["y"].values


def build_target_E(df, group_col):
    pos = df[FAS] == "FATIGA EXTREMA"
    neg = df[FAS] == "NO HAY FATIGA"
    mask = pos | neg
    sub = df.loc[mask].copy()
    sub["y"] = pos.loc[mask].astype(int)
    return sub.reset_index(drop=True), sub["y"].values


# ═════════════════════════════════════════════════════════════════════
#  Feature helpers
# ═════════════════════════════════════════════════════════════════════
_TRIU = {}


def _triu_idx(n):
    if n not in _TRIU:
        _TRIU[n] = np.triu_indices(n, k=1)
    return _TRIU[n]


def extract_features(tensor, tidx, channels):
    r, c = _triu_idx(tensor.shape[-1])
    return {ch: tensor[tidx, ch][:, r, c] for ch in channels}


def prepare_confounds(df):
    age = df["Age"].fillna(df["Age"].median()).values.astype(np.float64)
    sex = (df["Sex"] == "M").astype(np.float64).values
    return np.column_stack([age, sex])


def _safe_splits(y, requested):
    return max(2, min(requested, int(np.bincount(y).min())))


# ═════════════════════════════════════════════════════════════════════
#  Multiple-comparison corrections (applied to main-run nominal p's)
# ═════════════════════════════════════════════════════════════════════
def _holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    """Holm–Bonferroni step-down (FWER control)."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.minimum(np.sort(pvals) * (n - np.arange(n)), 1.0)
    for i in range(1, n):
        adj[i] = max(adj[i], adj[i - 1])
    out = np.empty(n)
    out[order] = adj
    return out


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg step-up (FDR control)."""
    n = len(pvals)
    order = np.argsort(pvals)[::-1]
    sorted_p = np.array(pvals)[order]
    adj = np.minimum(sorted_p * n / np.arange(n, 0, -1), 1.0)
    for i in range(1, n):
        adj[i] = min(adj[i], adj[i - 1])
    out = np.empty(n)
    out[order] = adj
    return out


def load_baseline_results(out_dir: Path) -> pd.DataFrame:
    """Load main-run baseline results CSV for correction."""
    csv = out_dir / "Tables" / "all_baseline_results.csv"
    if csv.exists():
        return pd.read_csv(csv)
    log.warning("Baseline results not found at %s", csv)
    return pd.DataFrame()


# ═════════════════════════════════════════════════════════════════════
#  Within-fold PCA precomputation for max-perm test
# ═════════════════════════════════════════════════════════════════════
def _precompute_fold_pca_scores(
    all_features: dict,
    confounds: np.ndarray,
    channels: List[int],
    cv_splits: list,
    n_pca: int,
    seed: int,
) -> dict:
    """Precompute within-fold PCA scores (label-free steps only).

    For each CV fold:
      Raw pipeline:   StandardScaler → PCA  (fit on train, transform train+test)
      Resid pipeline: LinearRegression (confound→features) → residualize →
                      StandardScaler → PCA  (fit on train, transform train+test)

    Since residualization, scaling, and PCA do not use the response labels,
    these fold scores are invariant to label permutation.  Only
    LogisticRegression must be re-fit per permutation, making the permutation
    loop ~20× faster than full within-fold re-fitting while preserving the
    exact same estimand as the main-run pipeline.
    """
    fold_scores: dict = {ch: {"raw": [], "resid": []} for ch in channels}

    for train_idx, test_idx in cv_splits:
        nc = min(n_pca, len(train_idx) - 1)

        for ch in channels:
            X = all_features[ch]

            # ── Raw: StandardScaler → PCA ─────────────────────────
            sc = StandardScaler().fit(X[train_idx])
            Xsc_tr = sc.transform(X[train_idx])
            Xsc_te = sc.transform(X[test_idx])
            pca = PCA(n_components=nc, random_state=seed).fit(Xsc_tr)
            fold_scores[ch]["raw"].append((
                pca.transform(Xsc_tr),
                pca.transform(Xsc_te),
                train_idx,
                test_idx,
            ))

            # ── Residualized: regress out confounds → Scale → PCA ─
            reg = LinearRegression().fit(confounds[train_idx], X[train_idx])
            Xr_tr = X[train_idx] - reg.predict(confounds[train_idx])
            Xr_te = X[test_idx]  - reg.predict(confounds[test_idx])
            sc_r = StandardScaler().fit(Xr_tr)
            Xrsc_tr = sc_r.transform(Xr_tr)
            Xrsc_te = sc_r.transform(Xr_te)
            pca_r = PCA(n_components=nc, random_state=seed).fit(Xrsc_tr)
            fold_scores[ch]["resid"].append((
                pca_r.transform(Xrsc_tr),
                pca_r.transform(Xrsc_te),
                train_idx,
                test_idx,
            ))

    return fold_scores


def _auc_from_fold_scores(
    fold_scores: dict,
    ch: int,
    cond: str,
    y: np.ndarray,
    clf_template,
) -> float:
    """OOF AUC from precomputed fold PCA scores — only re-fits LogReg."""
    n = len(y)
    y_prob = np.empty(n)
    for Xtr, Xte, train_idx, test_idx in fold_scores[ch][cond]:
        clf = clone(clf_template)
        clf.fit(Xtr, y[train_idx])
        y_prob[test_idx] = clf.predict_proba(Xte)[:, 1]
    try:
        return roc_auc_score(y, y_prob)
    except ValueError:
        return float("nan")


# ═════════════════════════════════════════════════════════════════════
#  Analysis 1: Max-permutation correction across channels (within-fold)
# ═════════════════════════════════════════════════════════════════════
def max_permutation_correction(
    all_features: dict,
    confounds: np.ndarray,
    y: np.ndarray,
    channels: List[int],
    n_perm: int,
    n_pca: int,
    n_splits: int,
    seed: int,
) -> dict:
    """Max-statistic permutation test correcting for channel selection.

    Uses **within-fold PCA** — the same estimand as the main-run pipeline.
    Residualization, scaling, and PCA are fitted on the training fold only,
    ensuring no leakage from test data into the feature transformation.

    To make this computationally tractable, the label-free steps
    (residualization, scaling, PCA) are precomputed once per fold.
    Only LogisticRegression is re-fitted per permutation, since it is
    the only label-dependent step.  This is ~20× faster than full
    per-permutation within-fold re-fitting.

    The corrected p-value = (1 + #{null_max ≥ observed_best}) / (1 + n_perm).
    """
    n  = len(y)
    ns = _safe_splits(y, n_splits)
    cv  = StratifiedKFold(n_splits=ns, shuffle=True, random_state=seed)
    rng = np.random.RandomState(seed + 999)

    obs_splits = list(cv.split(np.zeros(n), y))

    # ── Precompute within-fold PCA scores (label-free) ────────────────
    log.info("    Precomputing within-fold PCA scores for %d channels ...",
             len(channels))
    fold_scores = _precompute_fold_pca_scores(
        all_features, confounds, channels, obs_splits, n_pca, seed
    )
    log.info("    Done.  Starting permutation loop (%d perms) ...", n_perm)

    clf_template = LogisticRegression(
        max_iter=2000, random_state=seed, solver="lbfgs"
    )

    # ── Observed AUCs ─────────────────────────────────────────────────
    observed_raw   = {}
    observed_resid = {}
    for ch in channels:
        observed_raw[ch]   = _auc_from_fold_scores(
            fold_scores, ch, "raw",   y, clf_template)
        observed_resid[ch] = _auc_from_fold_scores(
            fold_scores, ch, "resid", y, clf_template)
    best_raw_auc   = max(observed_raw.values())
    best_resid_auc = max(observed_resid.values())

    # ── Permutation null ──────────────────────────────────────────────
    null_max_raw   = np.zeros(n_perm)
    null_max_resid = np.zeros(n_perm)
    null_ch_raw    = {ch: np.zeros(n_perm) for ch in channels}
    null_ch_resid  = {ch: np.zeros(n_perm) for ch in channels}

    _t0 = _time.time()

    for p_idx in range(n_perm):
        y_perm = rng.permutation(y)

        mr, md = -np.inf, -np.inf
        for ch in channels:
            ar = _auc_from_fold_scores(fold_scores, ch, "raw",   y_perm, clf_template)
            ad = _auc_from_fold_scores(fold_scores, ch, "resid", y_perm, clf_template)
            null_ch_raw[ch][p_idx]   = ar
            null_ch_resid[ch][p_idx] = ad
            mr = max(mr, ar)
            md = max(md, ad)

        null_max_raw[p_idx]   = mr
        null_max_resid[p_idx] = md

        if (p_idx + 1) % 100 == 0:
            elapsed = _time.time() - _t0
            rate = (p_idx + 1) / elapsed
            eta = (n_perm - p_idx - 1) / rate if rate > 0 else 0
            log.info("    max-perm %d / %d  (%.1f perm/s, ETA %.0fs)",
                     p_idx + 1, n_perm, rate, eta)

    # ── Corrected p-values ────────────────────────────────────────────
    corr_p_raw   = (1 + np.sum(null_max_raw   >= best_raw_auc  )) / (1 + n_perm)
    corr_p_resid = (1 + np.sum(null_max_resid >= best_resid_auc)) / (1 + n_perm)

    nominal_p: dict = {}
    for ch in channels:
        pr  = (1 + np.sum(null_ch_raw[ch]   >= observed_raw[ch]  )) / (1 + n_perm)
        pd_ = (1 + np.sum(null_ch_resid[ch] >= observed_resid[ch])) / (1 + n_perm)
        nominal_p[ch] = {"raw": round(pr, 6), "resid": round(pd_, 6)}

    return dict(
        observed_raw    = {ch: round(v, 4) for ch, v in observed_raw.items()},
        observed_resid  = {ch: round(v, 4) for ch, v in observed_resid.items()},
        best_raw_ch     = max(observed_raw,   key=observed_raw.get),
        best_resid_ch   = max(observed_resid, key=observed_resid.get),
        best_raw_auc    = round(best_raw_auc,   4),
        best_resid_auc  = round(best_resid_auc, 4),
        corrected_p_raw   = round(corr_p_raw,   6),
        corrected_p_resid = round(corr_p_resid, 6),
        nominal_p         = nominal_p,
        null_max_raw      = null_max_raw,
        null_max_resid    = null_max_resid,
        estimand          = "within_fold_pca",
    )


# ═════════════════════════════════════════════════════════════════════
#  Analysis 2: Incremental value over metadata  (Target D only)
# ═════════════════════════════════════════════════════════════════════
def incremental_value_analysis(
    tensor: np.ndarray,
    df: pd.DataFrame,
    y: np.ndarray,
    best_ch: int,
    confounds: np.ndarray,
    n_pca: int,
    n_splits: int,
    seed: int,
    n_boot: int = 2000,
) -> dict:
    """Compare metadata-only / connectome-only / combined models.

    The combined model uses residualised PCA scores + raw Age+Sex,
    so connectome and metadata contribute independently.

    Returns OOF predictions for all three models and a paired bootstrap
    test for the AUC difference (combined − metadata).
    """
    tidx = df["tensor_idx"].values
    r, c = _triu_idx(tensor.shape[-1])
    X_conn = tensor[tidx, best_ch][:, r, c]
    X_meta = confounds

    n = len(y)
    ns = _safe_splits(y, n_splits)
    cv = StratifiedKFold(n_splits=ns, shuffle=True, random_state=seed)

    oof_meta = np.zeros(n)
    oof_conn = np.zeros(n)
    oof_comb = np.zeros(n)

    for train_idx, test_idx in cv.split(X_conn, y):
        # ── Metadata only ────────────────────────────────────
        sc_m = StandardScaler().fit(X_meta[train_idx])
        Xm_tr = sc_m.transform(X_meta[train_idx])
        Xm_te = sc_m.transform(X_meta[test_idx])
        clf_m = LogisticRegression(max_iter=2000, random_state=seed, solver="lbfgs")
        clf_m.fit(Xm_tr, y[train_idx])
        oof_meta[test_idx] = clf_m.predict_proba(Xm_te)[:, 1]

        # ── Connectome only (residualised by Age+Sex) ────────
        reg = LinearRegression().fit(X_meta[train_idx], X_conn[train_idx])
        Xc_tr = X_conn[train_idx] - reg.predict(X_meta[train_idx])
        Xc_te = X_conn[test_idx]  - reg.predict(X_meta[test_idx])

        sc_c = StandardScaler().fit(Xc_tr)
        Xc_tr_s = sc_c.transform(Xc_tr)
        Xc_te_s = sc_c.transform(Xc_te)

        curr_nc = min(n_pca, Xc_tr_s.shape[0] - 1)
        pca = PCA(n_components=curr_nc, random_state=seed).fit(Xc_tr_s)
        Xpc_tr = pca.transform(Xc_tr_s)
        Xpc_te = pca.transform(Xc_te_s)

        clf_c = LogisticRegression(max_iter=2000, random_state=seed, solver="lbfgs")
        clf_c.fit(Xpc_tr, y[train_idx])
        oof_conn[test_idx] = clf_c.predict_proba(Xpc_te)[:, 1]

        # ── Combined: resid-PCA scores + Age + Sex ───────────
        Xcb_tr = np.hstack([Xpc_tr, Xm_tr])
        Xcb_te = np.hstack([Xpc_te, Xm_te])
        clf_cb = LogisticRegression(max_iter=2000, random_state=seed, solver="lbfgs")
        clf_cb.fit(Xcb_tr, y[train_idx])
        oof_comb[test_idx] = clf_cb.predict_proba(Xcb_te)[:, 1]

    auc_meta = roc_auc_score(y, oof_meta)
    auc_conn = roc_auc_score(y, oof_conn)
    auc_comb = roc_auc_score(y, oof_comb)

    # ── Paired bootstrap: ΔAUC (combined − metadata) ─────────
    delta_obs = auc_comb - auc_meta
    rng = np.random.RandomState(seed + 77)
    boot_deltas = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        while len(np.unique(y[idx])) < 2:
            idx = rng.choice(n, size=n, replace=True)
        boot_deltas[b] = (
            roc_auc_score(y[idx], oof_comb[idx])
            - roc_auc_score(y[idx], oof_meta[idx])
        )

    p_delta = float(np.mean(boot_deltas <= 0))
    ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])

    # ── Bootstrap CI for connectome-only ─────────────────────
    boot_conn_auc = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        while len(np.unique(y[idx])) < 2:
            idx = rng.choice(n, size=n, replace=True)
        boot_conn_auc[b] = roc_auc_score(y[idx], oof_conn[idx])
    ci_conn = np.percentile(boot_conn_auc, [2.5, 97.5])

    # ── Paired bootstrap: ΔAUC (connectome − metadata) ───────
    delta_conn_meta = auc_conn - auc_meta
    boot_dc = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        while len(np.unique(y[idx])) < 2:
            idx = rng.choice(n, size=n, replace=True)
        boot_dc[b] = (
            roc_auc_score(y[idx], oof_conn[idx])
            - roc_auc_score(y[idx], oof_meta[idx])
        )
    p_conn_vs_meta = float(np.mean(boot_dc <= 0))

    return dict(
        auc_metadata          = round(auc_meta, 4),
        auc_connectome_resid  = round(auc_conn, 4),
        auc_combined          = round(auc_comb, 4),
        delta_comb_vs_meta    = round(delta_obs, 4),
        delta_p               = round(p_delta, 6),
        delta_ci              = [round(ci_lo, 4), round(ci_hi, 4)],
        delta_conn_vs_meta    = round(delta_conn_meta, 4),
        delta_conn_vs_meta_p  = round(p_conn_vs_meta, 6),
        connectome_ci         = [round(ci_conn[0], 4), round(ci_conn[1], 4)],
        best_ch               = best_ch,
        channel_name          = CHANNEL_NAMES[best_ch],
        oof_meta   = oof_meta,
        oof_conn   = oof_conn,
        oof_comb   = oof_comb,
    )


# ═════════════════════════════════════════════════════════════════════
#  Analysis 3: Sex-matched subsampling
# ═════════════════════════════════════════════════════════════════════
def sex_matched_subsampling(
    tensor: np.ndarray,
    df: pd.DataFrame,
    y: np.ndarray,
    top_channels: List[int],
    n_reps: int,
    n_pca: int,
    seed: int,
    sex_filter: str = "F",
) -> pd.DataFrame:
    """Repeated balanced subsampling within one sex.

    For females (23 pos / 11 neg), subsample 11 from 23 positives to
    get 11-vs-11 balanced samples.  LOOCV on each subsample.
    """
    mask = (df["Sex"] == sex_filter).values
    df_s = df[mask].reset_index(drop=True)
    y_s  = y[mask]
    tidx = df_s["tensor_idx"].values

    feats = extract_features(tensor, tidx, top_channels)

    n_pos = int(y_s.sum())
    n_neg = int(len(y_s) - y_s.sum())
    n_min = min(n_pos, n_neg)

    if n_min < 5:
        log.warning("  %s-only: minority n=%d too small, skipping.", sex_filter, n_min)
        return pd.DataFrame()

    pos_idx = np.where(y_s == 1)[0]
    neg_idx = np.where(y_s == 0)[0]
    rng = np.random.RandomState(seed + 123)
    loo = LeaveOneOut()

    rows = []
    for rep in range(n_reps):
        if n_pos > n_neg:
            sub_pos = rng.choice(pos_idx, size=n_min, replace=False)
            sub_neg = neg_idx
        else:
            sub_pos = pos_idx
            sub_neg = rng.choice(neg_idx, size=n_min, replace=False)

        sub = np.sort(np.concatenate([sub_pos, sub_neg]))
        y_sub   = y_s[sub]
        n_sub   = len(y_sub)
        n_train = n_sub - 1                     # LOOCV
        nc = min(n_pca, max(1, n_train // 2))   # conservative PCA dim

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=nc, random_state=seed)),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed,
                                       solver="lbfgs")),
        ])

        for ch in top_channels:
            try:
                yp = cross_val_predict(
                    pipe, feats[ch][sub], y_sub, cv=loo, method="predict_proba",
                )[:, 1]
                auc = roc_auc_score(y_sub, yp)
            except Exception:
                auc = float("nan")

            rows.append(dict(
                rep=rep,
                channel_idx=ch,
                channel_name=CHANNEL_NAMES[ch],
                auc=round(auc, 4),
                n=n_sub,
                n_pos=int(y_sub.sum()),
                sex=sex_filter,
            ))

        if (rep + 1) % 50 == 0:
            log.info("    subsample %d / %d", rep + 1, n_reps)

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════
#  Analysis 4: Metadata decomposition  (Target D only)
# ═════════════════════════════════════════════════════════════════════
def metadata_decomposition_analysis(
    df: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    n_boot: int = 2000,
    n_perm: int = 1000,
) -> dict:
    """Age-only, Sex-only, and Age+Sex predictive value for Target D.

    All models use the same stratified 5-fold CV and OOF-pooled AUC.
    Permutation p-value (vs. chance) and bootstrap 95% CI are reported.
    This decomposition identifies whether sex is the dominant confound
    driving the Age+Sex metadata AUC.

    Returns a dict with keys 'age_only', 'sex_only', 'age_sex', each
    containing 'auc', 'ci_95', 'perm_p', and the raw OOF predictions.
    """
    age     = df["Age"].fillna(df["Age"].median()).values.astype(np.float64).reshape(-1, 1)
    sex_enc = (df["Sex"] == "M").astype(np.float64).values.reshape(-1, 1)
    age_sex = np.hstack([age, sex_enc])

    n  = len(y)
    ns = _safe_splits(y, n_splits)
    cv = StratifiedKFold(n_splits=ns, shuffle=True, random_state=seed)

    models = {"age_only": age, "sex_only": sex_enc, "age_sex": age_sex}
    oof: dict = {k: np.zeros(n) for k in models}

    for train_idx, test_idx in cv.split(age_sex, y):
        for name, X in models.items():
            sc  = StandardScaler().fit(X[train_idx])
            clf = LogisticRegression(max_iter=2000, random_state=seed, solver="lbfgs")
            clf.fit(sc.transform(X[train_idx]), y[train_idx])
            oof[name][test_idx] = clf.predict_proba(sc.transform(X[test_idx]))[:, 1]

    aucs = {name: roc_auc_score(y, oof[name]) for name in models}

    # ── Permutation test vs chance ────────────────────────────
    rng = np.random.RandomState(seed + 11)
    perm_aucs: dict = {name: np.zeros(n_perm) for name in models}
    for perm_i in range(n_perm):
        y_perm = rng.permutation(y)
        for name in models:
            perm_aucs[name][perm_i] = roc_auc_score(y_perm, oof[name])

    # ── Bootstrap CIs ─────────────────────────────────────────
    boot_results: dict = {name: [] for name in models}
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        while len(np.unique(y[idx])) < 2:
            idx = rng.choice(n, size=n, replace=True)
        for name in models:
            boot_results[name].append(roc_auc_score(y[idx], oof[name][idx]))

    results: dict = {}
    for name in models:
        boots  = np.array(boot_results[name])
        perms  = perm_aucs[name]
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        perm_p = float((1 + np.sum(perms >= aucs[name])) / (1 + n_perm))
        results[name] = dict(
            auc    = round(aucs[name], 4),
            ci_95  = [round(ci_lo, 4), round(ci_hi, 4)],
            perm_p = round(perm_p, 4),
            oof    = oof[name].tolist(),
        )

    return results


# ═════════════════════════════════════════════════════════════════════
#  Analysis 5: Power / precision analysis  (Target D only)
# ═════════════════════════════════════════════════════════════════════
def power_analysis(
    y: np.ndarray,
    oof_predictions: Dict[str, np.ndarray],
    seed: int,
    n_boot: int = 2000,
) -> dict:
    """Approximate power analysis for the main findings.

    For each supplied set of OOF predictions, estimates:
      - Observed AUC and bootstrap SE
      - Approximate power at current n (one-sided test, α = 0.05)
      - Rough n needed for 80% power (analytical approximation)

    Method: SE(AUC) is estimated from bootstrap resamples.
    Power is approximated as P(Z > z_α − effect/SE) where
    effect = AUC − 0.5.  The n-scaling uses SE ∝ 1/√n.

    These are conservative approximations.  They are useful for
    framing negative results as underpowered, not conclusively absent.
    """
    n     = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    rng   = np.random.RandomState(seed + 88)
    z_alpha = _norm.ppf(0.95)   # one-sided α = 0.05
    z_80    = _norm.ppf(0.80)

    results: dict = {}

    for name, oof in oof_predictions.items():
        auc = float(roc_auc_score(y, oof))

        # Bootstrap SE
        boot_aucs = np.zeros(n_boot)
        for b in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            while len(np.unique(y[idx])) < 2:
                idx = rng.choice(n, size=n, replace=True)
            boot_aucs[b] = roc_auc_score(y[idx], oof[idx])

        se  = float(np.std(boot_aucs))
        ci  = np.percentile(boot_aucs, [2.5, 97.5])

        # Approximate power at current n
        effect = auc - 0.5
        ncp    = (effect / se) if se > 0 else 0.0
        power_current = float(_norm.cdf(ncp - z_alpha))

        # Rough n for 80% power  (SE scales as ~1/√n, so SE_unit = SE*√n)
        n_for_80 = None
        if se > 0 and effect > 0:
            se_unit = se * (n ** 0.5)
            if se_unit > 0:
                n_for_80 = int(round(((z_alpha + z_80) * se_unit / effect) ** 2))

        results[name] = dict(
            auc                      = round(auc,    4),
            bootstrap_se             = round(se,     4),
            ci_95                    = [round(float(ci[0]), 4), round(float(ci[1]), 4)],
            effect_auc_minus_chance  = round(effect, 4),
            power_at_current_n       = round(power_current, 3),
            n_current                = n,
            n_pos                    = n_pos,
            n_neg                    = n_neg,
            n_for_80pct_power_approx = n_for_80,
        )

    return results


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--tensor_path", default=str(DEFAULT_TENSOR))
    ap.add_argument("--metadata_path", default=str(DEFAULT_META))
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--n_pca", type=int, default=20)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--n_permutations", type=int, default=1000)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_subsample_reps", type=int, default=200)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--smoke_test", action="store_true")
    return ap.parse_args(argv)


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════
def main(argv=None) -> int:
    args = parse_args(argv)

    if args.smoke_test:
        args.n_permutations    = 49
        args.n_boot            = 200
        args.n_subsample_reps  = 30
        log.info("SMOKE-TEST mode — reduced iterations")

    out = Path(args.output_dir)
    for sub in ("Tables", "Figures", "Logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────
    tensor, df_all, group_col = load_covid_data(
        Path(args.tensor_path), Path(args.metadata_path),
    )
    channels = list(range(tensor.shape[1]))

    # ═════════════════════════════════════════════════════════
    #  TARGET D
    # ═════════════════════════════════════════════════════════
    log.info("\n" + "=" * 64)
    log.info("  HARDENING — TARGET D  (n=53, COVID-only)")
    log.info("=" * 64)

    df_d, y_d   = build_target_D(df_all, group_col)
    tidx_d       = df_d["tensor_idx"].values
    feats_d      = extract_features(tensor, tidx_d, channels)
    conf_d       = prepare_confounds(df_d)

    # ── 1. Max-permutation correction (within-fold PCA) ───────────────
    log.info("──── 1. Max-permutation correction (Target D, within-fold PCA) ────")
    mp_d = max_permutation_correction(
        feats_d, conf_d, y_d, channels,
        args.n_permutations, args.n_pca, args.n_splits, args.random_state,
    )
    log.info("  BEST RAW   ch%d %-18s AUC=%.4f  nominal_p=%s  corrected_p=%s",
             mp_d["best_raw_ch"], CHANNEL_NAMES[mp_d["best_raw_ch"]],
             mp_d["best_raw_auc"],
             mp_d["nominal_p"][mp_d["best_raw_ch"]]["raw"],
             mp_d["corrected_p_raw"])
    log.info("  BEST RESID ch%d %-18s AUC=%.4f  nominal_p=%s  corrected_p=%s",
             mp_d["best_resid_ch"], CHANNEL_NAMES[mp_d["best_resid_ch"]],
             mp_d["best_resid_auc"],
             mp_d["nominal_p"][mp_d["best_resid_ch"]]["resid"],
             mp_d["corrected_p_resid"])

    mp_d_save = {k: v for k, v in mp_d.items() if not isinstance(v, np.ndarray)}
    (out / "Tables" / "hardening_D_max_perm.json").write_text(
        json.dumps(mp_d_save, indent=2, default=str), encoding="utf-8",
    )
    np.savez_compressed(
        out / "Tables" / "hardening_D_null_distributions.npz",
        null_max_raw=mp_d["null_max_raw"],
        null_max_resid=mp_d["null_max_resid"],
    )

    # ── 2. Incremental value over metadata ───────────────────
    log.info("──── 2. Incremental value over metadata (Target D) ────")
    best_resid_ch = mp_d["best_resid_ch"]
    iv = incremental_value_analysis(
        tensor, df_d, y_d, best_resid_ch, conf_d,
        args.n_pca, args.n_splits, args.random_state, args.n_boot,
    )
    log.info("  Metadata-only AUC:      %.4f", iv["auc_metadata"])
    log.info("  Connectome-only AUC:    %.4f  95%%CI [%.4f, %.4f]",
             iv["auc_connectome_resid"], iv["connectome_ci"][0], iv["connectome_ci"][1])
    log.info("  Combined AUC:           %.4f", iv["auc_combined"])
    log.info("  DELTA (combined - meta): %.4f  p=%.4f  95%%CI [%.4f, %.4f]",
             iv["delta_comb_vs_meta"], iv["delta_p"],
             iv["delta_ci"][0], iv["delta_ci"][1])
    log.info("  DELTA (conn - meta):     %.4f  p=%.4f",
             iv["delta_conn_vs_meta"], iv["delta_conn_vs_meta_p"])

    iv_save = {k: v for k, v in iv.items() if not isinstance(v, np.ndarray)}
    (out / "Tables" / "hardening_D_incremental_value.json").write_text(
        json.dumps(iv_save, indent=2, default=str), encoding="utf-8",
    )
    np.savez_compressed(
        out / "Tables" / "hardening_D_oof_predictions.npz",
        oof_meta=iv["oof_meta"], oof_conn=iv["oof_conn"],
        oof_comb=iv["oof_comb"], y=y_d,
    )

    # ── 2b. Explicit channel-wise incremental value artifacts (ch2 and ch3) ──
    # Keep the legacy hardening_D_incremental_value.json as "best_resid_ch",
    # but always emit explicit MI_KNN (ch2) and dFC_AbsDiffMean (ch3) files
    # so notebooks cannot silently duplicate one channel under two labels.
    explicit_channels = [
        (2, "hardening_D_incremental_value_ch2.json", "MI_KNN"),
        (3, "hardening_D_incremental_value_ch3.json", "dFC_AbsDiffMean"),
    ]
    for ch_idx, fname, ch_label in explicit_channels:
        if ch_idx == best_resid_ch:
            iv_ch = iv
        else:
            log.info("──── 2b. Incremental value for ch%d %s ────", ch_idx, ch_label)
            iv_ch = incremental_value_analysis(
                tensor, df_d, y_d, ch_idx, conf_d,
                args.n_pca, args.n_splits, args.random_state, args.n_boot,
            )
            log.info("  Metadata-only AUC:      %.4f", iv_ch["auc_metadata"])
            log.info("  Connectome-only AUC:    %.4f  95%%CI [%.4f, %.4f]",
                     iv_ch["auc_connectome_resid"], iv_ch["connectome_ci"][0], iv_ch["connectome_ci"][1])
            log.info("  Combined AUC:           %.4f", iv_ch["auc_combined"])
            log.info("  DELTA (combined - meta): %.4f  p=%.4f  95%%CI [%.4f, %.4f]",
                     iv_ch["delta_comb_vs_meta"], iv_ch["delta_p"],
                     iv_ch["delta_ci"][0], iv_ch["delta_ci"][1])

        iv_ch_save = {k: v for k, v in iv_ch.items() if not isinstance(v, np.ndarray)}
        (out / "Tables" / fname).write_text(
            json.dumps(iv_ch_save, indent=2, default=str), encoding="utf-8",
        )

    # ── 3. Sex-matched subsampling ───────────────────────────
    # Always include the 3 channels significant in main-run resid:
    # ch2=MI_KNN, ch3=dFC_AbsDiffMean, ch5=DistanceCorr
    top_chs = sorted(set([best_resid_ch, 2, 3, 5]))[:4]
    log.info("──── 3. Sex-matched subsampling (Target D, channels=%s) ────",
             [CHANNEL_NAMES[c] for c in top_chs])

    df_fem = sex_matched_subsampling(
        tensor, df_d, y_d, top_chs,
        args.n_subsample_reps, args.n_pca, args.random_state, "F",
    )
    if len(df_fem) > 0:
        for ch in top_chs:
            aucs = df_fem.loc[df_fem["channel_idx"] == ch, "auc"].dropna()
            log.info("  FEMALE ch%d %-18s median=%.3f  IQR=[%.3f, %.3f]  >0.5: %.0f%%",
                     ch, CHANNEL_NAMES[ch], aucs.median(),
                     aucs.quantile(0.25), aucs.quantile(0.75),
                     100 * (aucs > 0.5).mean())
        df_fem.to_csv(out / "Tables" / "hardening_D_female_subsampling.csv", index=False)

    df_male = sex_matched_subsampling(
        tensor, df_d, y_d, top_chs,
        min(args.n_subsample_reps, 50), args.n_pca, args.random_state, "M",
    )
    if len(df_male) > 0:
        for ch in top_chs:
            aucs = df_male.loc[df_male["channel_idx"] == ch, "auc"].dropna()
            if len(aucs) > 0:
                log.info("  MALE   ch%d %-18s median=%.3f  IQR=[%.3f, %.3f]  >0.5: %.0f%%",
                         ch, CHANNEL_NAMES[ch], aucs.median(),
                         aucs.quantile(0.25), aucs.quantile(0.75),
                         100 * (aucs > 0.5).mean())
        df_male.to_csv(out / "Tables" / "hardening_D_male_subsampling.csv", index=False)

    # ── 4. Metadata decomposition ─────────────────────────────
    log.info("──── 4. Metadata decomposition (Target D) ────")
    meta_decomp = metadata_decomposition_analysis(
        df_d, y_d, args.n_splits, args.random_state, args.n_boot,
    )
    for name, res in meta_decomp.items():
        log.info("  %-12s  AUC=%.4f  95%%CI [%.4f, %.4f]  perm_p=%.4f",
                 name, res["auc"], res["ci_95"][0], res["ci_95"][1], res["perm_p"])
    # Save without OOF arrays (those are large — save separately if needed)
    meta_decomp_save = {
        k: {kk: vv for kk, vv in v.items() if kk != "oof"}
        for k, v in meta_decomp.items()
    }
    (out / "Tables" / "hardening_D_metadata_decomposition.json").write_text(
        json.dumps(meta_decomp_save, indent=2), encoding="utf-8",
    )

    # ── 5. Power / precision analysis ────────────────────────
    log.info("──── 5. Power analysis (Target D) ────")
    oof_data = np.load(out / "Tables" / "hardening_D_oof_predictions.npz")
    oof_preds_power = {
        "best_resid_connectome": oof_data["oof_conn"],
        "metadata_age_sex":      oof_data["oof_meta"],
        "combined":              oof_data["oof_comb"],
        "sex_only":              np.array(meta_decomp["sex_only"]["oof"]),
        "age_only":              np.array(meta_decomp["age_only"]["oof"]),
    }
    power_results = power_analysis(y_d, oof_preds_power, args.random_state, args.n_boot)
    for name, res in power_results.items():
        log.info("  %-28s AUC=%.4f  SE=%.4f  power@n=%d: %.3f  n80: %s",
                 name, res["auc"], res["bootstrap_se"], res["n_current"],
                 res["power_at_current_n"], res["n_for_80pct_power_approx"])
    (out / "Tables" / "hardening_D_power_analysis.json").write_text(
        json.dumps(power_results, indent=2, default=str), encoding="utf-8",
    )

    # ═════════════════════════════════════════════════════════
    #  TARGET E  (max-perm only)
    # ═════════════════════════════════════════════════════════
    log.info("\n" + "=" * 64)
    log.info("  HARDENING — TARGET E  (n=85, all groups)")
    log.info("=" * 64)

    df_e, y_e   = build_target_E(df_all, group_col)
    tidx_e       = df_e["tensor_idx"].values
    feats_e      = extract_features(tensor, tidx_e, channels)
    conf_e       = prepare_confounds(df_e)

    log.info("──── 1. Max-permutation correction (Target E, within-fold PCA) ────")
    mp_e = max_permutation_correction(
        feats_e, conf_e, y_e, channels,
        args.n_permutations, args.n_pca, args.n_splits, args.random_state,
    )
    log.info("  BEST RAW   ch%d %-18s AUC=%.4f  nominal_p=%s  corrected_p=%s",
             mp_e["best_raw_ch"], CHANNEL_NAMES[mp_e["best_raw_ch"]],
             mp_e["best_raw_auc"],
             mp_e["nominal_p"][mp_e["best_raw_ch"]]["raw"],
             mp_e["corrected_p_raw"])
    log.info("  BEST RESID ch%d %-18s AUC=%.4f  nominal_p=%s  corrected_p=%s",
             mp_e["best_resid_ch"], CHANNEL_NAMES[mp_e["best_resid_ch"]],
             mp_e["best_resid_auc"],
             mp_e["nominal_p"][mp_e["best_resid_ch"]]["resid"],
             mp_e["corrected_p_resid"])

    mp_e_save = {k: v for k, v in mp_e.items() if not isinstance(v, np.ndarray)}
    (out / "Tables" / "hardening_E_max_perm.json").write_text(
        json.dumps(mp_e_save, indent=2, default=str), encoding="utf-8",
    )
    np.savez_compressed(
        out / "Tables" / "hardening_E_null_distributions.npz",
        null_max_raw=mp_e["null_max_raw"],
        null_max_resid=mp_e["null_max_resid"],
    )

    # ═════════════════════════════════════════════════════════
    #  Holm / BH corrections on MAIN-RUN nominal p-values
    # ═════════════════════════════════════════════════════════
    log.info("\n" + "=" * 64)
    log.info("  HOLM / BH CORRECTIONS ON MAIN-RUN P-VALUES")
    log.info("=" * 64)

    baseline_df = load_baseline_results(out)
    corrections_rows = []
    if len(baseline_df) > 0:
        for tgt in ["D", "E"]:
            for cond, is_resid in [("raw", False), ("resid", True)]:
                sub = baseline_df[
                    (baseline_df["target"] == tgt)
                    & (baseline_df["residualized"] == is_resid)
                    & (baseline_df["channel_name"].notna())
                    & (baseline_df["label"].str.startswith(cond + "_"))
                ].copy()
                if len(sub) == 0:
                    continue
                pvals    = sub["perm_p"].values
                ch_names = sub["channel_name"].values
                aucs     = sub["auc_cv"].values

                holm_adj = _holm_bonferroni(pvals)
                bh_adj   = _benjamini_hochberg(pvals)

                for i, ch in enumerate(ch_names):
                    row = dict(
                        target=tgt, condition=cond, channel=ch,
                        auc_cv=round(aucs[i], 4),
                        nominal_p=round(pvals[i], 6),
                        holm_p=round(holm_adj[i], 6),
                        bh_fdr_q=round(bh_adj[i], 6),
                    )
                    corrections_rows.append(row)
                    if holm_adj[i] < 0.10 or bh_adj[i] < 0.10:
                        log.info("  Target %s %s %-18s AUC=%.3f  nom=%.4f  Holm=%.4f  BH=%.4f",
                                 tgt, cond, ch, aucs[i], pvals[i], holm_adj[i], bh_adj[i])

        corr_df = pd.DataFrame(corrections_rows)
        corr_df.to_csv(out / "Tables" / "hardening_holm_bh_corrections.csv", index=False)
        log.info("  Corrections table saved (%d rows)", len(corr_df))
    else:
        log.warning("  No baseline results found — skipping Holm/BH corrections")

    # ═════════════════════════════════════════════════════════
    #  SUMMARY
    # ═════════════════════════════════════════════════════════
    log.info("\n" + "=" * 64)
    log.info("  HARDENING — FINAL SUMMARY")
    log.info("=" * 64)

    summary = dict(
        target_D=dict(
            max_perm_estimand="within_fold_pca",
            max_perm_raw=dict(
                best_ch=CHANNEL_NAMES[mp_d["best_raw_ch"]],
                auc=mp_d["best_raw_auc"],
                nominal_p=mp_d["nominal_p"][mp_d["best_raw_ch"]]["raw"],
                corrected_p=mp_d["corrected_p_raw"],
            ),
            max_perm_resid=dict(
                best_ch=CHANNEL_NAMES[mp_d["best_resid_ch"]],
                auc=mp_d["best_resid_auc"],
                nominal_p=mp_d["nominal_p"][mp_d["best_resid_ch"]]["resid"],
                corrected_p=mp_d["corrected_p_resid"],
            ),
            incremental_value=dict(
                auc_meta=iv["auc_metadata"],
                auc_conn=iv["auc_connectome_resid"],
                auc_comb=iv["auc_combined"],
                delta=iv["delta_comb_vs_meta"],
                delta_p=iv["delta_p"],
                delta_ci=iv["delta_ci"],
            ),
            metadata_decomposition=meta_decomp_save,
        ),
        target_E=dict(
            max_perm_estimand="within_fold_pca",
            max_perm_raw=dict(
                best_ch=CHANNEL_NAMES[mp_e["best_raw_ch"]],
                auc=mp_e["best_raw_auc"],
                nominal_p=mp_e["nominal_p"][mp_e["best_raw_ch"]]["raw"],
                corrected_p=mp_e["corrected_p_raw"],
            ),
            max_perm_resid=dict(
                best_ch=CHANNEL_NAMES[mp_e["best_resid_ch"]],
                auc=mp_e["best_resid_auc"],
                nominal_p=mp_e["nominal_p"][mp_e["best_resid_ch"]]["resid"],
                corrected_p=mp_e["corrected_p_resid"],
            ),
        ),
        smoke_test=args.smoke_test,
        n_permutations=args.n_permutations,
        n_boot=args.n_boot,
        n_subsample_reps=args.n_subsample_reps,
    )
    (out / "Logs" / "hardening_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    log.info(json.dumps(summary, indent=2, default=str))
    log.info("\nAll hardening outputs saved to %s", out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
