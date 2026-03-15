#!/usr/bin/env python3
"""
Fatigue Connectome Baselines — Adversarial Analysis
====================================================

Binary classification of post-COVID fatigue from resting-state
functional connectivity, with mandatory confound control.

Targets
-------
D : FATIGA EXTREMA  vs  NO HAY FATIGA  (COVID-only)
    Primary scientific question.
    Clean fatigue contrast within a single infection group.

E : FATIGA EXTREMA  vs  (NO HAY FATIGA ∪ healthy CONTROLS)
    Broader sensitivity / sanity-check analysis.
    CANNOT be used alone to claim fatigue biology.
    Susceptible to group-shortcut learning (COVID vs CONTROL).

Interpretation Rule
-------------------
If Target E >> Target D  →  default interpretation is **shortcut learning**.
Only Target D can support a fatigue-specific connectomic claim.

Usage
-----
Smoke test (< 2 min):
    python scripts/run_fatigue_connectome_baselines.py --smoke_test

Full run (≈ 20 min):
    python scripts/run_fatigue_connectome_baselines.py --target D E
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fatigue_baselines")

# ── Constants ────────────────────────────────────────────────────────────
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

# Merge key: tensor NPZ stores IDs as 'CP0001' which match the 'ID'
# column, **not** 'SubjectID', in SubjectsData_AAL3_COVID.csv.
TENSOR_ID_COL = "ID"


# ═════════════════════════════════════════════════════════════════════════
#  Confound Residualizer  (sklearn-compatible, fold-safe)
# ═════════════════════════════════════════════════════════════════════════
class ConfoundResidualizer(BaseEstimator, TransformerMixin):
    """Remove linear confound effects inside each CV fold.

    Expects the **last** ``n_confounds`` columns of *X* to be confound
    variables.  ``fit`` learns the OLS mapping confounds → features on the
    **training** fold; ``transform`` projects the confound contribution
    out of both train and test data.
    """

    def __init__(self, n_confounds: int = 2):
        self.n_confounds = n_confounds

    # noinspection PyPep8Naming
    def fit(self, X, y=None):
        feats = X[:, : -self.n_confounds]
        confs = X[:, -self.n_confounds :]
        self.reg_ = LinearRegression().fit(confs, feats)
        return self

    # noinspection PyPep8Naming
    def transform(self, X):
        feats = X[:, : -self.n_confounds]
        confs = X[:, -self.n_confounds :]
        return feats - self.reg_.predict(confs)


# ═════════════════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════════════════
def _resolve_group_col(df: pd.DataFrame) -> str:
    """Return the column name that holds COVID / CONTROL labels."""
    for col in ("Grupo", "ResearchGroup"):
        if col in df.columns:
            vals = set(df[col].dropna().unique())
            if {"COVID", "CONTROL"}.issubset(vals):
                return col
    raise KeyError("No column with COVID / CONTROL labels found in metadata")


def load_covid_data(
    tensor_path: Path,
    metadata_path: Path,
) -> Tuple[np.ndarray, pd.DataFrame, List[str], str]:
    """Load COVID tensor + metadata, merge on ``ID``.

    Returns
    -------
    tensor    : ndarray (N, C, R, R)
    df        : merged DataFrame with ``tensor_idx`` column
    ch_names  : channel names from NPZ
    group_col : column with COVID / CONTROL labels
    """
    npz = np.load(str(tensor_path), allow_pickle=True)
    tensor = npz["global_tensor_data"]               # (194, 7, 131, 131)
    sids   = npz["subject_ids"]                       # (194,)  e.g. 'CP0008'
    ch_names = [str(c) for c in npz["channel_names"]]

    tdf  = pd.DataFrame({TENSOR_ID_COL: sids, "tensor_idx": np.arange(len(sids))})
    meta = pd.read_csv(metadata_path)
    df   = tdf.merge(meta, on=TENSOR_ID_COL, how="inner")

    if len(df) == 0:
        raise RuntimeError(
            f"Merge on '{TENSOR_ID_COL}' produced 0 rows. "
            f"Tensor IDs sample: {sids[:3]}, CSV columns: {list(meta.columns[:6])}"
        )

    group_col = _resolve_group_col(df)
    log.info(
        "Tensor %s | metadata %d rows → merged %d | group_col='%s'",
        tensor.shape, len(meta), len(df), group_col,
    )
    return tensor, df, ch_names, group_col


# ═════════════════════════════════════════════════════════════════════════
#  Target definitions
# ═════════════════════════════════════════════════════════════════════════
FAS = "CategoriaFAS"   # column name — no accent


def build_target(
    df: pd.DataFrame, target: str, group_col: str,
) -> Tuple[pd.DataFrame, np.ndarray, dict]:
    """Filter *df* for the requested target.

    Returns (filtered_df_reset, y, info_dict).
    """
    if target == "D":
        # ── COVID-only: FATIGA EXTREMA vs NO HAY FATIGA ──
        mask = (
            (df[group_col] == "COVID")
            & df[FAS].isin(["FATIGA EXTREMA", "NO HAY FATIGA"])
        )
        sub = df.loc[mask].copy()
        sub["y"] = (sub[FAS] == "FATIGA EXTREMA").astype(int)
        info = dict(
            target="D",
            desc="FATIGA EXTREMA vs NO HAY FATIGA  (COVID-only)",
            pos_label="FATIGA EXTREMA",
            neg_label="NO HAY FATIGA",
            restriction="COVID-only",
        )

    elif target == "E":
        # ── FATIGA EXTREMA vs NO HAY FATIGA (both groups) ──────────
        # Positive: FATIGA EXTREMA  (COVID + CONTROL)
        # Negative: NO HAY FATIGA   (COVID + CONTROL)
        #
        # IMPORTANT: Controls with CategoriaFAS == "FATIGA" are EXCLUDED.
        # Including them would contaminate the negative class with
        # fatigued subjects, violating the intended contrast.
        pos_mask = df[FAS] == "FATIGA EXTREMA"
        neg_mask = df[FAS] == "NO HAY FATIGA"
        mask = pos_mask | neg_mask
        sub = df.loc[mask].copy()
        sub["y"] = pos_mask.loc[mask].astype(int)
        info = dict(
            target="E",
            desc="FATIGA EXTREMA vs NO HAY FATIGA  (all groups)",
            pos_label="FATIGA EXTREMA  (COVID + CONTROL)",
            neg_label="NO HAY FATIGA   (COVID + CONTROL)",
            restriction="All groups — only FATIGA EXTREMA and NO HAY FATIGA",
            WARNING=(
                "Target E is NOT a pure fatigue classifier. "
                "Positive class is ≈94 % COVID while Negative is ≈44 % COVID. "
                "Any classifier may exploit the COVID-vs-CONTROL shortcut. "
                "Controls with intermediate FATIGA are excluded."
            ),
        )
    else:
        raise ValueError(f"Unknown target '{target}'. Choose D or E.")

    y = sub["y"].values
    info.update(
        n_pos=int(y.sum()),
        n_neg=int(len(y) - y.sum()),
        n_total=int(len(y)),
        prevalence=round(float(y.mean()), 4),
    )
    log.info(
        "Target %s  →  %d pos / %d neg = %d  (prevalence %.2f)",
        target, info["n_pos"], info["n_neg"], info["n_total"], info["prevalence"],
    )
    return sub.reset_index(drop=True), y, info


# ═════════════════════════════════════════════════════════════════════════
#  Feature helpers
# ═════════════════════════════════════════════════════════════════════════
_TRIU: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


def _triu_idx(n: int):
    if n not in _TRIU:
        _TRIU[n] = np.triu_indices(n, k=1)
    return _TRIU[n]


def extract_upper_tri(
    tensor: np.ndarray, tidx: np.ndarray, ch: int,
) -> np.ndarray:
    """Vectorise off-diagonal upper triangle for one channel → (n, n_edges)."""
    mats = tensor[tidx, ch]            # (n, R, R)
    r, c = _triu_idx(mats.shape[1])
    return mats[:, r, c]


def prepare_confounds(df: pd.DataFrame) -> np.ndarray:
    """Return (n, 2) array  [Age, Sex_numeric(M=1)]."""
    age = df["Age"].fillna(df["Age"].median()).values.astype(np.float64)
    sex = (df["Sex"] == "M").astype(np.float64).values
    return np.column_stack([age, sex])


# ═════════════════════════════════════════════════════════════════════════
#  Confound audit
# ═════════════════════════════════════════════════════════════════════════
def confound_audit(
    df: pd.DataFrame,
    y: np.ndarray,
    target: str,
    group_col: str,
    out: Path,
) -> dict:
    """Write per-class distributions of every potential confound."""
    sub = df.copy()
    sub["class"] = np.where(y == 1, "Positive", "Negative")
    tables: dict = {}
    tdir = out / "Tables"

    # ── sex ──────────────────────────────────────────────────
    sex_ct  = pd.crosstab(sub["class"], sub["Sex"])
    sex_pct = sex_ct.div(sex_ct.sum(axis=1), axis=0).mul(100).round(1)
    tables["sex_counts"] = sex_ct
    tables["sex_pct"]    = sex_pct
    log.info("  Sex (counts):\n%s", sex_ct.to_string())
    if sex_ct.shape == (2, 2):
        chi2, p, *_ = stats.chi2_contingency(sex_ct)
        tables["sex_chi2"] = {"chi2": round(chi2, 4), "p": round(p, 6)}
        log.info("  Sex χ²=%.3f  p=%.4f%s", chi2, p,
                 "  ⚠️ SIGNIFICANT" if p < 0.05 else "")

    # ── age ──────────────────────────────────────────────────
    age_desc = sub.groupby("class")["Age"].describe()
    tables["age_desc"] = age_desc
    pos_a = sub.loc[sub["class"] == "Positive", "Age"].dropna()
    neg_a = sub.loc[sub["class"] == "Negative", "Age"].dropna()
    if len(pos_a) > 1 and len(neg_a) > 1:
        t, p = stats.ttest_ind(pos_a, neg_a, equal_var=False)
        tables["age_ttest"] = {"t": round(t, 4), "p": round(p, 6)}
        log.info("  Age Welch t=%.3f  p=%.4f", t, p)

    # ── group (COVID / CONTROL) — critical for Target E ──
    if sub[group_col].nunique() > 1:
        grp_ct  = pd.crosstab(sub["class"], sub[group_col])
        grp_pct = grp_ct.div(grp_ct.sum(axis=1), axis=0).mul(100).round(1)
        tables["group_counts"] = grp_ct
        tables["group_pct"]    = grp_pct
        log.info("  Group (counts):\n%s", grp_ct.to_string())
        if grp_ct.shape == (2, 2):
            chi2g, pg, *_ = stats.chi2_contingency(grp_ct)
            tables["group_chi2"] = {"chi2": round(chi2g, 4), "p": round(pg, 6)}
            log.info("  Group χ²=%.3f  p=%.4f%s", chi2g, pg,
                     "  ⚠️ SIGNIFICANT — shortcut threat!" if pg < 0.05 else "")

    # ── severity (CategoríaCOVID) ────────────────────────────
    sev_cols = [c for c in sub.columns if "COVID" in c and "Categor" in c]
    if sev_cols:
        sc = sev_cols[0]
        tmp = sub.dropna(subset=[sc])
        if len(tmp) > 0:
            tables["severity_counts"] = pd.crosstab(tmp["class"], tmp[sc])

    # ── CategoriaFAS breakdown (useful context for Target E) ──
    if FAS in sub.columns:
        fas_ct = pd.crosstab(sub["class"], sub[FAS])
        tables["fas_breakdown"] = fas_ct

    # ── Recuperado (recovery) ────────────────────────────────
    if "Recuperado" in sub.columns:
        rec_tmp = sub.dropna(subset=["Recuperado"])
        if len(rec_tmp) > 0 and rec_tmp["Recuperado"].nunique() > 1:
            tables["recuperado_counts"] = pd.crosstab(rec_tmp["class"], rec_tmp["Recuperado"])

    # ── persist ──────────────────────────────────────────────
    for name, val in tables.items():
        fpath = tdir / f"confound_{target}_{name}.csv"
        if isinstance(val, (pd.DataFrame, pd.Series)):
            val.to_csv(fpath)
        else:
            pd.Series(val).to_csv(fpath)

    return tables


# ═════════════════════════════════════════════════════════════════════════
#  Unified classification routine
# ═════════════════════════════════════════════════════════════════════════
def _safe_splits(y: np.ndarray, requested: int) -> int:
    mn = int(np.bincount(y).min())
    return max(2, min(requested, mn))


def classify(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_pca: int,
    n_perm: int,
    n_splits: int,
    seed: int,
    label: str,
    target: str,
    residualize: bool = False,
    n_confounds: int = 0,
    extra: Optional[dict] = None,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """PCA → LogReg with cross-val AUC + optional permutation test.

    When ``residualize=True``, the last ``n_confounds`` columns of *X*
    are treated as confound variables and removed within each CV fold.

    Returns
    -------
    result : dict   – summary metrics
    y_prob : (n,)   – cross-validated predicted probabilities
    perm_scores : (n_perm,) – permutation AUC distribution (empty if n_perm=0)
    """
    ns = _safe_splits(y, n_splits)

    # effective feature dim after possible residualization
    feat_dim = X.shape[1] - n_confounds if residualize else X.shape[1]
    n_train_approx = X.shape[0] * (ns - 1) // ns
    max_comp = max(1, min(n_train_approx, feat_dim) - 1)
    nc = max(1, min(n_pca, max_comp))

    steps: list = []
    if residualize and n_confounds > 0:
        steps.append(("residualizer", ConfoundResidualizer(n_confounds)))
    steps += [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=nc, random_state=seed)),
        ("clf", LogisticRegression(max_iter=2000, random_state=seed, solver="lbfgs")),
    ]
    pipe = Pipeline(steps)
    cv = StratifiedKFold(n_splits=ns, shuffle=True, random_state=seed)

    # ── cross-val predictions ────────────────────────────────
    try:
        y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, y_prob)
    except Exception as exc:
        log.warning("  classify '%s' failed: %s", label, exc)
        y_prob = np.full(len(y), np.nan)
        auc = float("nan")

    # ── permutation test ─────────────────────────────────────
    if n_perm > 0 and not np.isnan(auc):
        score, perm_sc, pval = permutation_test_score(
            pipe, X, y,
            scoring="roc_auc",
            cv=cv,
            n_permutations=n_perm,
            random_state=seed,
            n_jobs=-1,
        )
    else:
        score = float(auc)
        perm_sc = np.array([])
        pval = float("nan")

    res = dict(
        target=target,
        label=label,
        auc_cv=round(float(auc), 4),
        auc_perm=round(float(score), 4),
        perm_p=round(float(pval), 6) if not np.isnan(pval) else None,
        perm_mean=round(float(np.mean(perm_sc)), 4) if len(perm_sc) else None,
        perm_std=round(float(np.std(perm_sc)), 4)  if len(perm_sc) else None,
        n_perm=n_perm,
        n_pca=nc,
        n_splits=ns,
        n_samples=len(y),
        n_pos=int(y.sum()),
        n_neg=int(len(y) - y.sum()),
        residualized=residualize,
    )
    if extra:
        res.update(extra)

    log.info("    %s  AUC=%.3f  perm_p=%s", label, auc,
             f"{pval:.4f}" if not np.isnan(pval) else "n/a")
    return res, y_prob, perm_sc


# ═════════════════════════════════════════════════════════════════════════
#  Per-target experiment runner
# ═════════════════════════════════════════════════════════════════════════
def run_target(
    tensor: np.ndarray,
    df: pd.DataFrame,
    y: np.ndarray,
    target: str,
    group_col: str,
    channels: List[int],
    args: argparse.Namespace,
    out: Path,
) -> Tuple[List[dict], List[dict], pd.DataFrame]:
    """Run every baseline for one target. Returns (baselines, diagnostics, preds)."""
    baselines: List[dict] = []
    diagnostics: List[dict] = []
    pred_rows: List[dict] = []

    tidx = df["tensor_idx"].values
    C    = prepare_confounds(df)              # (n, 2)

    def _save_preds(lab: str, yp: np.ndarray):
        for i in range(len(df)):
            pred_rows.append(dict(
                ID=df.iloc[i][TENSOR_ID_COL],
                target=target,
                label=lab,
                y_true=int(y[i]),
                y_prob=float(yp[i]),
            ))

    # ─────────────────────────────────────────────────────────
    # 1. Metadata-only (Age + Sex)
    # ─────────────────────────────────────────────────────────
    log.info("─── [%s] Metadata-only (Age+Sex) ───", target)
    res_m, yp_m, _ = classify(
        C, y,
        n_pca=2, n_perm=args.n_permutations, n_splits=args.n_splits,
        seed=args.random_state, label="metadata_AgeSex", target=target,
    )
    baselines.append(res_m)
    _save_preds("metadata_AgeSex", yp_m)

    # ─────────────────────────────────────────────────────────
    # 2–3. Per-channel: raw + residualized connectome
    # ─────────────────────────────────────────────────────────
    for ch in channels:
        ch_name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"ch{ch}"
        tag_info = dict(channel_idx=ch, channel_name=ch_name)

        X_feat = extract_upper_tri(tensor, tidx, ch)

        # 2a ── raw ───────────────────────────────────────────
        log.info("─── [%s] ch%d %s — RAW ───", target, ch, ch_name)
        r_raw, yp_raw, _ = classify(
            X_feat, y,
            n_pca=args.n_pca, n_perm=args.n_permutations,
            n_splits=args.n_splits, seed=args.random_state,
            label=f"raw_ch{ch}_{ch_name}", target=target, extra=tag_info,
        )
        baselines.append(r_raw)
        _save_preds(f"raw_ch{ch}_{ch_name}", yp_raw)

        # 2b ── residualized (Age+Sex projected out within each fold) ──
        log.info("─── [%s] ch%d %s — RESIDUALIZED ───", target, ch, ch_name)
        X_comb = np.hstack([X_feat, C])
        r_res, yp_res, _ = classify(
            X_comb, y,
            n_pca=args.n_pca, n_perm=args.n_permutations,
            n_splits=args.n_splits, seed=args.random_state,
            label=f"resid_ch{ch}_{ch_name}", target=target,
            residualize=True, n_confounds=2, extra=tag_info,
        )
        baselines.append(r_res)
        _save_preds(f"resid_ch{ch}_{ch_name}", yp_res)

        # ── sex-prediction diagnostic ────────────────────────
        y_sex = (df["Sex"] == "M").astype(int).values
        if len(np.unique(y_sex)) == 2:
            log.info("    sex diagnostic ch%d", ch)
            r_sd, _, _ = classify(
                X_feat, y_sex,
                n_pca=args.n_pca, n_perm=0,
                n_splits=args.n_splits, seed=args.random_state,
                label=f"sex_diag_ch{ch}_{ch_name}", target=target,
                extra={**tag_info, "diagnostic": "sex_prediction"},
            )
            diagnostics.append(r_sd)

        # ── group-prediction diagnostic (Target E shortcut audit) ──
        if df[group_col].nunique() > 1:
            y_grp = (df[group_col] == "COVID").astype(int).values
            if len(np.unique(y_grp)) == 2:
                log.info("    group diagnostic ch%d", ch)
                r_gd, _, _ = classify(
                    X_feat, y_grp,
                    n_pca=args.n_pca, n_perm=0,
                    n_splits=args.n_splits, seed=args.random_state,
                    label=f"group_diag_ch{ch}_{ch_name}", target=target,
                    extra={**tag_info, "diagnostic": "group_prediction"},
                )
                diagnostics.append(r_gd)

        # ── female-only sensitivity ──────────────────────────
        f_mask = (df["Sex"] == "F").values
        y_f = y[f_mask]
        if len(np.unique(y_f)) == 2 and len(y_f) >= 10:
            log.info("    female-only ch%d  (n=%d: %d pos / %d neg)",
                     ch, len(y_f), y_f.sum(), len(y_f) - y_f.sum())
            tidx_f = df.loc[f_mask, "tensor_idx"].values
            X_f = extract_upper_tri(tensor, tidx_f, ch)
            r_fem, _, _ = classify(
                X_f, y_f,
                n_pca=args.n_pca,
                n_perm=args.n_permutations,   # full permutation test
                n_splits=args.n_splits, seed=args.random_state,
                label=f"female_ch{ch}_{ch_name}", target=target,
                extra={**tag_info, "subset": "female_only",
                       "n_female": int(f_mask.sum())},
            )
            baselines.append(r_fem)

    return baselines, diagnostics, pd.DataFrame(pred_rows)


# ═════════════════════════════════════════════════════════════════════════
#  Stop / Go decision table
# ═════════════════════════════════════════════════════════════════════════
def decision_table(
    results: List[dict], out: Path,
) -> Tuple[pd.DataFrame, str]:
    """Build an explicit stop / go table with cross-target interpretation."""
    df = pd.DataFrame(results)
    rows = []

    for tgt in sorted(df["target"].unique()):
        t = df[df["target"] == tgt]

        # metadata
        meta = t[t["label"].str.startswith("metadata")]
        meta_auc = float(meta["auc_cv"].iloc[0]) if len(meta) else float("nan")
        meta_p   = meta["perm_p"].iloc[0] if len(meta) else None

        # best raw channel (by cross-val AUC)
        raw = t[t["label"].str.startswith("raw_")]
        if len(raw):
            best_raw = raw.loc[raw["auc_cv"].idxmax()]
            raw_auc = float(best_raw["auc_cv"])
            raw_p   = best_raw["perm_p"]
            raw_ch  = best_raw.get("channel_name", "")
        else:
            raw_auc, raw_p, raw_ch = float("nan"), None, ""

        # best residualized channel
        resid = t[t["label"].str.startswith("resid_")]
        if len(resid):
            best_resid = resid.loc[resid["auc_cv"].idxmax()]
            resid_auc = float(best_resid["auc_cv"])
            resid_p   = best_resid["perm_p"]
            resid_ch  = best_resid.get("channel_name", "")
        else:
            resid_auc, resid_p, resid_ch = float("nan"), None, ""

        # best female-only
        fem = t[t["label"].str.startswith("female_")]
        if len(fem):
            best_fem = fem.loc[fem["auc_cv"].idxmax()]
            fem_auc = float(best_fem["auc_cv"])
            fem_p   = best_fem["perm_p"]
            fem_ch  = best_fem.get("channel_name", "")
        else:
            fem_auc, fem_p, fem_ch = float("nan"), None, ""

        _sig = lambda p: p is not None and not np.isnan(float(p)) and float(p) < 0.05
        raw_sig   = _sig(raw_p)
        resid_sig = _sig(resid_p)
        fem_sig   = _sig(fem_p)
        meta_sig  = _sig(meta_p)

        # Nominal layer: what the single-channel permutation test shows.
        # This is NEVER the authoritative verdict — it is always superseded
        # by the hardened assessment in §8 / run_fatigue_hardening.py.
        if resid_sig:
            nominal_layer = "nominal-pass (single ch, pre-correction)"
        elif raw_sig and not resid_sig:
            nominal_layer = "confound (raw passes, resid collapses)"
        else:
            nominal_layer = "nominal-null (does not beat perm null)"

        # The verdict column reports the NOMINAL status + mandatory caveats.
        # It deliberately leads with INDETERMINATE when hardening is needed,
        # so that stop_go_decision_table.csv cannot be misread as a go signal.
        if resid_sig and fem_sig and not meta_sig:
            verdict = (
                "INDETERMINATE (NOMINAL) — "
                + nominal_layer
                + " — requires multiplicity correction via hardening script"
            )
        elif resid_sig and not fem_sig:
            verdict = (
                "INDETERMINATE — "
                + nominal_layer
                + "; female-only NOT confirmed;"
                + " multiplicity correction and incremental-value test required"
            )
        elif resid_sig and meta_sig:
            verdict = (
                "INDETERMINATE — "
                + nominal_layer
                + "; metadata predicts target (incremental value untested);"
                + " multiplicity correction required"
            )
        else:
            verdict = "INDETERMINATE / NULL — " + nominal_layer

        rows.append(dict(
            target=tgt,
            meta_auc=meta_auc, meta_p=meta_p, meta_sig=meta_sig,
            best_raw_ch=raw_ch,   raw_auc=raw_auc,   raw_p=raw_p,   raw_sig=raw_sig,
            best_resid_ch=resid_ch, resid_auc=resid_auc, resid_p=resid_p, resid_sig=resid_sig,
            best_fem_ch=fem_ch,   fem_auc=fem_auc,   fem_p=fem_p,   fem_sig=fem_sig,
            verdict=verdict,
        ))

    dec = pd.DataFrame(rows)

    # ── cross-target interpretation ──────────────────────────
    d = dec[dec["target"] == "D"]
    e = dec[dec["target"] == "E"]
    d_sig = bool(d["resid_sig"].iloc[0]) if len(d) else False
    e_sig = bool(e["resid_sig"].iloc[0]) if len(e) else False

    if e_sig and not d_sig:
        cross = (
            "⚠️  SHORTCUT LEARNING / MIXED PHENOTYPE — "
            "Target E succeeds but Target D fails. "
            "The most parsimonious explanation is that the classifier exploits "
            "COVID-vs-CONTROL group differences rather than fatigue biology. "
            "Do NOT claim a fatigue biomarker."
        )
    elif d_sig and e_sig:
        d_fem = bool(d["fem_sig"].iloc[0]) if len(d) else False
        d_meta = bool(d["meta_sig"].iloc[0]) if len(d) else False
        if d_fem and not d_meta:
            cross = (
                "⚠️  INDETERMINATE (nominal pass, female-only confirms, "
                "metadata clean) — Both targets survive nominal test and "
                "female-only direction is consistent. "
                "STILL requires multiplicity correction via hardening script "
                "before any 'GO' verdict. Do NOT claim fatigue biomarker yet."
            )
        else:
            caveats = []
            if not d_fem:
                caveats.append("female-only NOT confirmed")
            if d_meta:
                caveats.append("metadata predicts target (incremental value untested)")
            cross = (
                "⚠️  INDETERMINATE (nominal signal, caveats unresolved) — "
                "Both targets show nominal signal at best single channel, "
                f"but: {'; '.join(caveats)}. "
                "Run hardening script for multiplicity correction. "
                "Do NOT claim fatigue biomarker without resolving caveats."
            )
    elif d_sig and not e_sig:
        cross = (
            "⚠️  INDETERMINATE (nominally fatigue-specific) — "
            "Target D passes nominal test but Target E fails. "
            "Pattern is consistent with a within-COVID fatigue signal, "
            "but requires multiplicity correction and incremental-value "
            "test before any fatigue-biomarker claim."
        )
    else:
        cross = (
            "❌  GLOBAL NULL — Neither target beats the permutation null "
            "after residualization. No evidence for a fatigue-related "
            "connectomic signal. Recommend publishing as a negative result."
        )

    tdir = out / "Tables"
    dec.to_csv(tdir / "stop_go_decision_table.csv", index=False)
    (tdir / "cross_target_interpretation.txt").write_text(
        cross + "\n", encoding="utf-8"
    )
    return dec, cross


# ═════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--target", nargs="+", default=["D", "E"],
                    choices=["D", "E"], help="Targets to evaluate")
    ap.add_argument("--tensor_path", type=str, default=str(DEFAULT_TENSOR))
    ap.add_argument("--metadata_path", type=str, default=str(DEFAULT_META))
    ap.add_argument("--output_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--n_pca", type=int, default=20,
                    help="Number of PCA components")
    ap.add_argument("--n_splits", type=int, default=5,
                    help="Stratified CV folds")
    ap.add_argument("--n_permutations", type=int, default=1000,
                    help="Permutation-test shuffles (full mode)")
    ap.add_argument("--channels", nargs="+", default=["all"],
                    help="Channel indices to test, or 'all'")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--smoke_test", action="store_true",
                    help="Quick run: 19 perms, 2 channels")
    return ap.parse_args(argv)


# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════
def main(argv=None) -> int:
    args = parse_args(argv)

    if args.smoke_test:
        args.n_permutations = 19
        log.info("🔧  SMOKE-TEST mode — 19 permutations, max 2 channels")

    out = Path(args.output_dir)
    for sub in ("Tables", "Figures", "Logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ── save config ──────────────────────────────────────────
    cfg = {k: str(v) if isinstance(v, Path) else v
           for k, v in vars(args).items()}
    cfg["script"] = str(Path(__file__).resolve())
    (out / "Logs" / "run_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8",
    )

    # ── load data ────────────────────────────────────────────
    tensor, df_all, ch_names, group_col = load_covid_data(
        Path(args.tensor_path), Path(args.metadata_path),
    )

    # ── channels ─────────────────────────────────────────────
    if "all" in args.channels:
        channels = list(range(tensor.shape[1]))
    else:
        channels = [int(c) for c in args.channels]
    if args.smoke_test:
        channels = channels[:2]
        log.info("  smoke-test channels: %s", channels)

    all_baselines:   List[dict] = []
    all_diagnostics: List[dict] = []
    all_preds:       List[pd.DataFrame] = []

    for tgt in args.target:
        log.info("\n" + "=" * 64)
        log.info("  TARGET  %s", tgt)
        log.info("=" * 64)

        df_tgt, y, tinfo = build_target(df_all, tgt, group_col)

        # persist definition
        (out / "Tables" / f"target_{tgt}_definition.json").write_text(
            json.dumps(tinfo, indent=2, ensure_ascii=False), encoding="utf-8",
        )

        # confound audit
        log.info("── confound audit ──")
        confound_audit(df_tgt, y, tgt, group_col, out)

        # baselines + diagnostics
        bl, dx, preds = run_target(
            tensor, df_tgt, y, tgt, group_col, channels, args, out,
        )
        all_baselines.extend(bl)
        all_diagnostics.extend(dx)
        all_preds.append(preds)

    # ── persist aggregated results ───────────────────────────
    tdir = out / "Tables"
    pd.DataFrame(all_baselines).to_csv(
        tdir / "all_baseline_results.csv", index=False,
    )
    pd.DataFrame(all_diagnostics).to_csv(
        tdir / "all_diagnostic_results.csv", index=False,
    )
    pd.concat(all_preds, ignore_index=True).to_csv(
        tdir / "all_subject_predictions.csv", index=False,
    )

    # ── decision table ───────────────────────────────────────
    log.info("\n" + "=" * 64)
    log.info("  STOP / GO  DECISION TABLE")
    log.info("=" * 64)
    dec, cross = decision_table(all_baselines, out)
    log.info("\n%s", dec.to_string(index=False))
    log.info("\n%s", cross)

    # ── summary ──────────────────────────────────────────────
    summary = dict(
        targets=args.target,
        channels=channels,
        n_permutations=args.n_permutations,
        smoke_test=args.smoke_test,
        cross_target_interpretation=cross,
        decision=dec.to_dict(orient="records"),
    )
    (out / "Logs" / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    log.info("\n✅  All outputs saved to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
