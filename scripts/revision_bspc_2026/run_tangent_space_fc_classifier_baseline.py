#!/usr/bin/env python3
"""
Fold-safe tangent-space FC classifier-only baseline.

Input:
  - Raw ROI time series (.mat files, 170 ROIs × 140 TPs) for v5.1b subjects
  - Outer-fold splits from adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate
  - Subject metadata: Age, Sex, Manufacturer

Pipeline (per outer fold):
  1. Load outer-train and outer-test ROI time series (140 × 170 → filter to 140 × 131)
  2. Fit ConnectivityMeasure(kind='tangent', cov_estimator=LedoitWolf()) on outer-train TS only
  3. Transform outer-train and outer-test → (n_subjects, 8515) tangent vectors
  4. Append Age (z-scored, train mean/std) + Sex (binary) → 8517 features
  5. Inner 5-fold CV for hyperparameter selection (C for logreg_l2, C+l1_ratio for elasticnet)
  6. Threshold selection via inner OOF: sensitivity ≥ 0.70, maximize specificity
  7. Train final classifier on full outer-train; evaluate outer-test once

ROI filter (170 → 131):
  - Remove 4 AAL3 systemically-missing columns (0-based: 34, 35, 80, 81) → 166
  - Remove 35 low-volume ROIs (< 100 vox) from those 166 → 131

Constraints:
  Read-only. No tensor, metadata, ledger, or VAE output modification.
  No VAE training. No global tangent fit across all subjects.
"""

from __future__ import annotations

import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Auto-detect DATA_ROOT: check both drive locations
_CANDIDATE_DATA_ROOTS = [
    Path("/media/diego/Datos/vae_AD_data"),
    Path("/media/diego/My_Book_Diego/vae_AD_data"),
]
_METADATA_REL = (
    "revision_bspc_2026"
    "/adni_expanded_v5_1_batch20260514b_no_pybandpass"
    "/training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)
DATA_ROOT = next(
    (p for p in _CANDIDATE_DATA_ROOTS if (p / _METADATA_REL).exists()),
    _CANDIDATE_DATA_ROOTS[0],  # fallback (will fail with clear error at load time)
)

METADATA_CSV = DATA_ROOT / _METADATA_REL

# Fold splits: use latent_cache CSVs (real dir, not symlink).
# These contain SubjectID, tensor_idx, ResearchGroup_Mapped, Age, Sex.
LATENT_CACHE_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
    / "latent_cache"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "tangent_space_fc_classifier_baseline"
)

# ROI signal directories (searched in order; first match wins)
ROISIG_DIRS: List[Path] = [
    Path("/home/diego/proyectos/vae_AD/data/OneDrive_1_13-5-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN"),
    Path("/home/diego/proyectos/vae_AD/data/OneDrive_1_14-5-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN"),
    Path("/home/diego/proyectos/vae_AD/data/OneDrive_2_14-5-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN"),
    Path("/home/diego/proyectos/vae_AD/data/OneDrive_1_27-4-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCF"),
    Path("/media/diego/Datos/desde_cero/ROISignalsAAL3"),
    Path("/media/diego/Datos/AAL3/ROISignalsAAL3"),
    Path("/media/diego/Datos/adni_expansion/MARTIN_20260429_PHILIPS10/OneDrive_2_29-4-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCF"),
    Path("/media/diego/Datos/adni_expansion/GE_batch7/ROISignals_AAL3_from_CovRegressed_GE_batch7"),
    Path("/media/diego/Datos/adni_expansion/GE_smoketest3/ROISignals_AAL3_from_CovRegressed_GE_smoketest3"),
    Path("/media/diego/My_Book_Diego/vae_AD_data/adni_passband_20260510/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN"),
]

# ---------------------------------------------------------------------------
# ROI filter: 170 → 131
# ---------------------------------------------------------------------------
# Step 1: drop 4 systemically-missing AAL3 columns (0-based in 170-column matrix)
_MISSING_0BASED = [34, 35, 80, 81]
_KEEP_FROM_170 = [i for i in range(170) if i not in _MISSING_0BASED]  # len=166

# Step 2: drop 35 small-volume ROIs (0-based positions in the 166-column result)
_SMALL_IN_166 = [
    108, 116, 117, 118, 119, 120, 121, 126, 127, 128, 129,
    132, 133, 134, 135, 136, 137, 138, 139, 142, 143, 144,
    145, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163,
    164, 165,
]
_KEEP_FROM_166 = [i for i in range(166) if i not in _SMALL_IN_166]  # len=131

# Final: which columns to take from the raw 170-column matrix
ROI_KEEP_170 = [_KEEP_FROM_170[j] for j in _KEEP_FROM_166]  # len=131
assert len(ROI_KEEP_170) == 131, f"Expected 131 ROI indices, got {len(ROI_KEEP_170)}"

N_ROIS = 131
N_TANGENT_FEATURES = N_ROIS * (N_ROIS - 1) // 2  # 8515
N_FEATURES_TOTAL = N_TANGENT_FEATURES + 2          # + Age + Sex

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
SEED = 42
TARGET_SENSITIVITY = 0.70

CLASSIFIERS = {
    "logreg_l2": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "logreg_elasticnet": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.5, 0.9],
    },
}

# Locked current model reference (adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate)
LOCKED_METRICS = {
    "AUC": 0.778785,
    "PR_AUC": 0.551832,
    "balanced_accuracy": 0.712917,
    "sensitivity": 0.729167,
    "specificity": 0.696667,
    "F1": 0.544747,
}


# ===========================================================================
# Helpers
# ===========================================================================

def df_to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(str(round(v, 6) if isinstance(v, float) else v) for v in row) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header, sep] + rows)


def build_mat_lookup_from_dirs(
    subjects: set,
    prepopulated: Optional[Dict[str, Path]] = None,
) -> Dict[str, Path]:
    """
    Return {SubjectID: mat_path} for the given subject set.
    prepopulated: paths already resolved (e.g. from metadata roisignals_path).
    Falls back to scanning ROISIG_DIRS for subjects not already found.
    """
    lookup: Dict[str, Path] = dict(prepopulated or {})
    missing = subjects - set(lookup.keys())

    for d in ROISIG_DIRS:
        if not d.exists() or not missing:
            continue
        for sid in list(missing):
            candidate = d / f"ROISignals_{sid}.mat"
            if candidate.exists():
                lookup[sid] = candidate
                missing.discard(sid)

    return lookup


def load_ts_131(mat_path: Path) -> np.ndarray:
    """Load .mat file → filter ROIs → return (140, 131) float64 array."""
    mat = sio.loadmat(str(mat_path))
    ts = mat["signals"].astype(np.float64)           # (n_tp, 170)
    assert ts.shape[1] >= max(ROI_KEEP_170) + 1, (
        f"Expected ≥{max(ROI_KEEP_170)+1} ROI cols in {mat_path}, got {ts.shape[1]}"
    )
    return ts[:, ROI_KEEP_170]                        # (n_tp, 131)


def select_threshold_inner_oof(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_sens: float = TARGET_SENSITIVITY,
) -> Tuple[float, str]:
    """
    Find threshold with maximum specificity where sensitivity ≥ target_sens.
    Falls back to Youden J if no threshold satisfies the constraint.
    Returns (threshold, strategy_used).
    """
    candidates = np.sort(np.unique(scores))

    best_thr, best_spec = None, -1.0
    for thr in candidates:
        y_pred = (scores >= thr).astype(int)
        sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        if sens >= target_sens and spec > best_spec:
            best_spec = spec
            best_thr = thr

    if best_thr is not None:
        return float(best_thr), "target_sens_ge_0p70_max_spec"

    # Fallback: Youden J
    best_j, best_thr = -1.0, float(np.median(scores))
    for thr in candidates:
        y_pred = (scores >= thr).astype(int)
        sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_thr = thr
    return float(best_thr), "youden_j_fallback"


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {
            "AUC": float("nan"), "PR_AUC": float("nan"),
            "balanced_accuracy": float("nan"), "sensitivity": float("nan"),
            "specificity": float("nan"), "F1": float("nan"),
            "mean_score_AD": float("nan"), "mean_score_CN": float("nan"),
            "score_separation": float("nan"),
        }
    auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    ba = balanced_accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    mean_ad = float(y_score[y_true == 1].mean()) if (y_true == 1).any() else float("nan")
    mean_cn = float(y_score[y_true == 0].mean()) if (y_true == 0).any() else float("nan")
    return {
        "AUC": auc, "PR_AUC": pr_auc,
        "balanced_accuracy": ba, "sensitivity": sens,
        "specificity": spec, "F1": f1,
        "mean_score_AD": mean_ad, "mean_score_CN": mean_cn,
        "score_separation": mean_ad - mean_cn if not (np.isnan(mean_ad) or np.isnan(mean_cn)) else float("nan"),
    }


def run_inner_cv(
    X: np.ndarray,
    y: np.ndarray,
    clf_name: str,
    param_grid: Dict,
    seed: int = SEED,
) -> Tuple[Dict, float, np.ndarray]:
    """
    Inner 5-fold CV for hyperparameter selection.
    Returns (best_params, best_mean_auc, oof_scores_with_best_params).
    """
    inner_skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed)

    # Build flat param combinations
    if clf_name == "logreg_l2":
        param_list = [{"C": c} for c in param_grid["C"]]
    elif clf_name == "logreg_elasticnet":
        param_list = [
            {"C": c, "l1_ratio": l}
            for c in param_grid["C"]
            for l in param_grid["l1_ratio"]
        ]
    else:
        raise ValueError(f"Unknown clf_name: {clf_name}")

    best_params = param_list[0]
    best_auc = -1.0

    for params in param_list:
        fold_aucs = []
        for train_idx, val_idx in inner_skf.split(X, y):
            clf = _make_clf(clf_name, params)
            clf.fit(X[train_idx], y[train_idx])
            scores_val = clf.predict_proba(X[val_idx])[:, 1]
            if len(np.unique(y[val_idx])) >= 2:
                fold_aucs.append(roc_auc_score(y[val_idx], scores_val))
        mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.0
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params

    # Collect OOF scores with best params for threshold selection
    oof_scores = np.zeros(len(y))
    for train_idx, val_idx in inner_skf.split(X, y):
        clf = _make_clf(clf_name, best_params)
        clf.fit(X[train_idx], y[train_idx])
        oof_scores[val_idx] = clf.predict_proba(X[val_idx])[:, 1]

    return best_params, best_auc, oof_scores


def _make_clf(clf_name: str, params: Dict) -> LogisticRegression:
    if clf_name == "logreg_l2":
        return LogisticRegression(
            penalty="l2",
            C=params["C"],
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        )
    elif clf_name == "logreg_elasticnet":
        return LogisticRegression(
            penalty="elasticnet",
            C=params["C"],
            l1_ratio=params["l1_ratio"],
            solver="saga",
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        )
    else:
        raise ValueError(f"Unknown clf_name: {clf_name}")


# ===========================================================================
# Per-fold runner
# ===========================================================================

def run_fold(
    fold_num: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ts_cache: Dict[str, np.ndarray],
) -> Dict:
    """
    Run one outer fold.
    train_df / test_df: DataFrames with columns SubjectID, ResearchGroup_Mapped, Age, Sex.
    Returns dict with metrics and hyperparameters.
    """
    train_sids = train_df["SubjectID"].tolist()
    test_sids  = test_df["SubjectID"].tolist()
    print(f"  Fold {fold_num}: train={len(train_sids)}, test={len(test_sids)}")

    # ---- 1. Prepare time-series lists ----------------------------------------
    train_ts = [ts_cache[s] for s in train_sids]
    test_ts  = [ts_cache[s] for s in test_sids]

    # ---- 2. Fit tangent reference on outer-train only ------------------------
    cm = ConnectivityMeasure(
        kind="tangent",
        cov_estimator=LedoitWolf(),
        vectorize=True,
        discard_diagonal=True,
    )
    X_train_tang = cm.fit_transform(train_ts)          # (n_train, 8515)
    X_test_tang  = cm.transform(test_ts)               # (n_test, 8515)

    assert X_train_tang.shape == (len(train_sids), N_TANGENT_FEATURES)
    assert X_test_tang.shape  == (len(test_sids),  N_TANGENT_FEATURES)

    # ---- 3. Age / Sex features (from latent cache CSV, no external metadata) --
    fold_meta = pd.concat([train_df, test_df]).set_index("SubjectID")

    def _age_sex(sids):
        ages  = np.array([float(fold_meta.loc[s, "Age"])  if s in fold_meta.index else np.nan for s in sids])
        sexes = np.array([1.0 if str(fold_meta.loc[s, "Sex"]).strip().upper() in ("M", "MALE") else 0.0
                          if s in fold_meta.index else np.nan for s in sids])
        return ages, sexes

    train_age, train_sex = _age_sex(train_sids)
    test_age,  test_sex  = _age_sex(test_sids)

    train_age_mean = float(np.nanmean(train_age))
    train_age[np.isnan(train_age)] = train_age_mean
    test_age[np.isnan(test_age)]   = train_age_mean
    train_sex[np.isnan(train_sex)] = 0.5
    test_sex[np.isnan(test_sex)]   = 0.5

    age_scaler = StandardScaler()
    train_age_sc = age_scaler.fit_transform(train_age.reshape(-1, 1)).ravel()
    test_age_sc  = age_scaler.transform(test_age.reshape(-1, 1)).ravel()

    X_train = np.column_stack([X_train_tang, train_age_sc, train_sex])
    X_test  = np.column_stack([X_test_tang,  test_age_sc,  test_sex])

    # ---- 4. Labels -----------------------------------------------------------
    def _labels(sids):
        return np.array([
            1 if fold_meta.loc[s, "ResearchGroup_Mapped"] == "AD" else 0
            for s in sids
        ], dtype=int)

    y_train = _labels(train_sids)
    y_test  = _labels(test_sids)

    # ---- 5. Inner CV + threshold selection per classifier --------------------
    fold_results: Dict = {
        "fold": fold_num,
        "n_train": len(train_sids),
        "n_test": len(test_sids),
        "n_train_AD": int(y_train.sum()),
        "n_test_AD": int(y_test.sum()),
    }
    clf_outputs = {}

    for clf_name, param_grid in CLASSIFIERS.items():
        print(f"    {clf_name}: inner CV ...", end="", flush=True)
        best_params, best_inner_auc, oof_scores = run_inner_cv(
            X_train, y_train, clf_name, param_grid
        )

        threshold, thr_strategy = select_threshold_inner_oof(y_train, oof_scores)

        # Train final model on full outer-train
        final_clf = _make_clf(clf_name, best_params)
        final_clf.fit(X_train, y_train)
        y_score_test = final_clf.predict_proba(X_test)[:, 1]
        y_pred_test  = (y_score_test >= threshold).astype(int)

        metrics = compute_metrics(y_test, y_score_test, y_pred_test)
        print(f" AUC={metrics['AUC']:.4f}  BA={metrics['balanced_accuracy']:.4f}")

        clf_outputs[clf_name] = {
            "best_params":      best_params,
            "best_inner_auc":   best_inner_auc,
            "threshold":        threshold,
            "threshold_strategy": thr_strategy,
            "metrics":          metrics,
            "y_true":           y_test,
            "y_score":          y_score_test,
            "test_sids":        test_sids,
        }

    fold_results["classifiers"] = clf_outputs
    return fold_results


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    t0 = datetime.now(timezone.utc)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TANGENT-SPACE FC CLASSIFIER BASELINE")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load fold splits from latent cache (real dir, always accessible)
    #    Each CSV has: SubjectID, tensor_idx, ResearchGroup_Mapped, Age, Sex
    # ------------------------------------------------------------------
    print("\n[1/6] Loading fold splits from latent cache ...")
    if not LATENT_CACHE_DIR.exists():
        sys.exit(f"LATENT_CACHE_DIR not found: {LATENT_CACHE_DIR}")

    fold_dfs: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    cv_subjects: set = set()
    cv_meta_parts: List[pd.DataFrame] = []

    for fold in range(1, N_OUTER_FOLDS + 1):
        train_df = pd.read_csv(LATENT_CACHE_DIR / f"fold_{fold}_trainDev_latent_mu.csv",
                               usecols=["SubjectID", "ResearchGroup_Mapped", "Age", "Sex"])
        test_df  = pd.read_csv(LATENT_CACHE_DIR / f"fold_{fold}_test_latent_mu.csv",
                               usecols=["SubjectID", "ResearchGroup_Mapped", "Age", "Sex"])
        fold_dfs.append((train_df, test_df))
        cv_subjects |= set(train_df["SubjectID"]) | set(test_df["SubjectID"])
        cv_meta_parts.append(train_df)
        cv_meta_parts.append(test_df)

    # Deduplicated subject metadata
    cv_meta = (
        pd.concat(cv_meta_parts)
        .drop_duplicates("SubjectID")
        .reset_index(drop=True)
    )
    print(f"  CV subjects: {len(cv_subjects)}  "
          f"(AD={int((cv_meta['ResearchGroup_Mapped']=='AD').sum())}, "
          f"CN={int((cv_meta['ResearchGroup_Mapped']=='CN').sum())})")

    # ------------------------------------------------------------------
    # 2. Data availability audit
    # ------------------------------------------------------------------
    print("\n[2/6] Data availability audit ...")

    # Try metadata CSV for Manufacturer info (optional — on Datos drive)
    mfr_lookup: Dict[str, str] = {}
    if METADATA_CSV.exists():
        meta_full = pd.read_csv(METADATA_CSV, usecols=["SubjectID", "Manufacturer",
                                                        "roisignals_path"])
        mfr_lookup = meta_full.set_index("SubjectID")["Manufacturer"].to_dict()
        # Also pre-populate mat_lookup from roisignals_path in metadata
        roisig_path_lookup = {
            row["SubjectID"]: Path(row["roisignals_path"])
            for _, row in meta_full.iterrows()
            if pd.notna(row.get("roisignals_path")) and Path(str(row["roisignals_path"])).exists()
        }
    else:
        roisig_path_lookup = {}
        print("  NOTE: metadata CSV not accessible (Datos drive not mounted). "
              "Manufacturer info unavailable; ROI signal paths from directory scan only.")

    mat_lookup = build_mat_lookup_from_dirs(cv_subjects, roisig_path_lookup)

    audit_rows = []
    for _, row in cv_meta.iterrows():
        sid = row["SubjectID"]
        audit_rows.append({
            "SubjectID":            sid,
            "Manufacturer":         mfr_lookup.get(sid, "unknown"),
            "ResearchGroup_Mapped": row["ResearchGroup_Mapped"],
            "mat_found":            sid in mat_lookup,
            "mat_path":             str(mat_lookup.get(sid, "")),
        })
    audit_df = pd.DataFrame(audit_rows)

    n_cv    = len(cv_subjects)
    n_found = int(audit_df["mat_found"].sum())
    n_miss  = n_cv - n_found

    print(f"  CV subjects:  {n_cv}")
    print(f"  .mat found:   {n_found}")
    print(f"  .mat missing: {n_miss}")

    if n_miss > 0:
        missing_sids = audit_df[~audit_df["mat_found"]]["SubjectID"].tolist()
        print(f"  MISSING examples: {missing_sids[:5]}")
        print()
        print("  *** The Datos drive (/dev/sda1) is not mounted. ***")
        print("  *** Mount it first, then re-run this script.    ***")
        print("  ***   sudo mount /dev/sda1 /media/diego/Datos   ***")
        audit_df.to_csv(OUTPUT_DIR / "data_availability_audit.csv", index=False)
        (OUTPUT_DIR / "data_availability_audit.md").write_text(
            f"# Data Availability Audit\n\nCV subjects: {n_cv}  "
            f"Found: {n_found}  Missing: {n_miss}\n\n"
            "**The Datos drive is not mounted. Mount /dev/sda1 first.**\n"
        )
        sys.exit(f"Aborted: {n_miss} CV subjects lack .mat files (Datos drive not mounted).")

    audit_df.to_csv(OUTPUT_DIR / "data_availability_audit.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Load time series for all CV subjects
    # ------------------------------------------------------------------
    print(f"\n[3/6] Loading {n_cv} time-series files ...")
    ts_cache: Dict[str, np.ndarray] = {}
    for sid in sorted(cv_subjects):
        ts_cache[sid] = load_ts_131(mat_lookup[sid])
    print(f"  Loaded. Shape example: {next(iter(ts_cache.values())).shape}")

    # ------------------------------------------------------------------
    # 4. Run 5 outer folds
    # ------------------------------------------------------------------
    print("\n[4/6] Running 5 outer folds ...")
    all_fold_results = []
    for fold_num, (train_df, test_df) in enumerate(fold_dfs, start=1):
        fold_res = run_fold(fold_num, train_df, test_df, ts_cache)
        all_fold_results.append(fold_res)

    # ------------------------------------------------------------------
    # 5. Aggregate results
    # ------------------------------------------------------------------
    print("\n[5/6] Aggregating results ...")

    metric_keys = ["AUC", "PR_AUC", "balanced_accuracy", "sensitivity", "specificity", "F1"]
    clf_names   = list(CLASSIFIERS.keys())

    # Foldwise metrics
    foldwise_rows = []
    for fr in all_fold_results:
        for clf in clf_names:
            out = fr["classifiers"][clf]
            row = {
                "fold":      fr["fold"],
                "classifier": clf,
                "threshold": round(out["threshold"], 6),
                "threshold_strategy": out["threshold_strategy"],
                "best_inner_auc": round(out["best_inner_auc"], 6),
            }
            row.update({k: round(out["metrics"][k], 6) for k in metric_keys})
            params = out["best_params"]
            row["best_C"] = params.get("C", "")
            row["best_l1_ratio"] = params.get("l1_ratio", "")
            foldwise_rows.append(row)
    foldwise_df = pd.DataFrame(foldwise_rows)
    foldwise_df.to_csv(OUTPUT_DIR / "foldwise_metrics.csv", index=False)

    # Pooled metrics (mean ± std across folds)
    primary_rows = []
    for clf in clf_names:
        sub = foldwise_df[foldwise_df["classifier"] == clf]
        row: Dict = {"classifier": clf}
        for k in metric_keys:
            row[f"{k}_mean"] = round(float(sub[k].mean()), 6)
            row[f"{k}_std"]  = round(float(sub[k].std(ddof=0)), 6)
        primary_rows.append(row)
    primary_df = pd.DataFrame(primary_rows)
    primary_df.to_csv(OUTPUT_DIR / "primary_results.csv", index=False)

    # Hyperparameters
    hp_rows = []
    for fr in all_fold_results:
        for clf in clf_names:
            out = fr["classifiers"][clf]
            params = out["best_params"]
            hp_rows.append({
                "fold": fr["fold"], "classifier": clf,
                "C": params.get("C", ""), "l1_ratio": params.get("l1_ratio", ""),
                "best_inner_auc": round(out["best_inner_auc"], 6),
            })
    hp_df = pd.DataFrame(hp_rows)
    hp_df.to_csv(OUTPUT_DIR / "selected_hyperparameters.csv", index=False)

    # ------------------------------------------------------------------
    # 6. OOF predictions
    # ------------------------------------------------------------------
    for clf in clf_names:
        pred_rows = []
        for fr in all_fold_results:
            out = fr["classifiers"][clf]
            for sid, yt, ys, yp in zip(
                out["test_sids"], out["y_true"], out["y_score"],
                (out["y_score"] >= out["threshold"]).astype(int),
            ):
                pred_rows.append({
                    "SubjectID": sid, "fold": fr["fold"],
                    "y_true": int(yt), "y_score": round(float(ys), 6),
                    "y_pred": int(yp), "threshold": round(out["threshold"], 6),
                })
        pd.DataFrame(pred_rows).to_csv(
            OUTPUT_DIR / f"oof_predictions_{clf}.csv", index=False
        )

    # ------------------------------------------------------------------
    # 7. Write text outputs
    # ------------------------------------------------------------------
    print("\n[6/6] Writing output files ...")
    _write_outputs(audit_df, foldwise_df, primary_df, hp_df, t0)

    # Console summary
    t1 = datetime.now(timezone.utc)
    print(f"\nDone in {round((t1-t0).total_seconds(), 1)}s")
    print("=" * 60)
    print("PRIMARY RESULTS")
    print("=" * 60)
    for _, row in primary_df.iterrows():
        clf = row["classifier"]
        print(f"  {clf}:")
        for k in metric_keys:
            locked = LOCKED_METRICS.get(k, float("nan"))
            diff = row[f"{k}_mean"] - locked
            arrow = "▲" if diff > 0 else "▼"
            print(f"    {k}: {row[f'{k}_mean']:.4f} ± {row[f'{k}_std']:.4f}  "
                  f"(locked={locked:.4f}, Δ={diff:+.4f} {arrow})")
    print(f"\n  Output: {OUTPUT_DIR}")
    print("=" * 60)

    # command_log.json
    log = {
        "script": Path(__file__).name,
        "started_utc": t0.isoformat(),
        "finished_utc": t1.isoformat(),
        "elapsed_s": round((t1 - t0).total_seconds(), 2),
        "n_cv_subjects": n_cv,
        "n_mat_found": n_found,
        "n_outer_folds": N_OUTER_FOLDS,
        "n_inner_folds": N_INNER_FOLDS,
        "n_rois": N_ROIS,
        "n_tangent_features": N_TANGENT_FEATURES,
        "n_features_total": N_FEATURES_TOTAL,
        "cov_estimator": "LedoitWolf",
        "tangent_reference": "fitted_on_outer_train_only",
        "classifiers": list(CLASSIFIERS.keys()),
        "threshold_strategy": "inner_oof_target_sens_ge_0p70_max_spec",
        "locked_reference_AUC": LOCKED_METRICS["AUC"],
        "locked_reference_PR_AUC": LOCKED_METRICS["PR_AUC"],
        "read_only": True,
        "python": sys.version,
    }
    (OUTPUT_DIR / "command_log.json").write_text(json.dumps(log, indent=2))


# ===========================================================================
# Output file writers
# ===========================================================================

def _write_outputs(
    audit_df: pd.DataFrame,
    foldwise_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    hp_df: pd.DataFrame,
    t0: datetime,
) -> None:
    metric_keys = ["AUC", "PR_AUC", "balanced_accuracy", "sensitivity", "specificity", "F1"]

    # ---- data_availability_audit.md ----------------------------------------
    mfr_counts = audit_df.groupby("Manufacturer")["mat_found"].agg(
        total="count", found=lambda x: x.sum()
    ).reset_index()
    (OUTPUT_DIR / "data_availability_audit.md").write_text(textwrap.dedent(f"""\
    # Data Availability Audit

    **CV subjects:** {len(audit_df)}
    **Mat files found:** {int(audit_df["mat_found"].sum())}
    **Mat files missing:** {int((~audit_df["mat_found"]).sum())}

    ## Coverage by Manufacturer

    {df_to_markdown(mfr_counts)}

    ## Input specification

    - Signal format: `.mat` (DPARSF output), variable `signals` shape `(140, 170)`
    - ROI filter: remove 4 systemically-missing AAL3 columns (0-based: 34, 35, 80, 81),
      then remove 35 small-volume ROIs (< 100 vox) → **131 ROIs** in AAL3 order
    - Covariance estimator: LedoitWolf (per subject, from own 140 × 131 time series)
    - Tangent reference: geometric mean fitted on **outer-train subjects only** per fold
    """))

    # ---- foldwise_metrics.md -----------------------------------------------
    display_cols = ["fold", "classifier", "AUC", "PR_AUC", "balanced_accuracy",
                    "sensitivity", "specificity", "F1", "threshold", "best_C", "best_l1_ratio"]
    (OUTPUT_DIR / "foldwise_metrics.md").write_text(
        "# Foldwise Metrics\n\n" + df_to_markdown(foldwise_df[display_cols])
    )

    # ---- primary_results.md ------------------------------------------------
    lines = ["# Primary Results (mean ± std across 5 folds)\n"]
    lines.append("## Comparison with locked current model")
    lines.append("| Metric | Locked VAE | " +
                 " | ".join(f"Tangent {c}" for c in primary_df["classifier"]) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in primary_df["classifier"]) + " |")
    for k in metric_keys:
        locked = LOCKED_METRICS.get(k, float("nan"))
        vals = [f"{row[f'{k}_mean']:.4f} ± {row[f'{k}_std']:.4f}" for _, row in primary_df.iterrows()]
        lines.append(f"| {k} | {locked:.4f} | " + " | ".join(vals) + " |")
    (OUTPUT_DIR / "primary_results.md").write_text("\n".join(lines) + "\n")

    # ---- selected_hyperparameters.md ---------------------------------------
    (OUTPUT_DIR / "selected_hyperparameters.md").write_text(
        "# Selected Hyperparameters\n\n" + df_to_markdown(hp_df)
    )

    # ---- leakage_risk_report.md --------------------------------------------
    (OUTPUT_DIR / "leakage_risk_report.md").write_text(textwrap.dedent("""\
    # Leakage Risk Report

    ## Tangent-space reference

    - The geometric mean (tangent reference) is fitted **only on outer-train subjects**
      in each of the 5 outer folds.
    - Outer-test subjects are never seen during `ConnectivityMeasure.fit()`.
    - Per-subject LedoitWolf covariance uses only that subject's 140 time points
      (no cross-subject covariance estimation).

    ## Feature scaling

    - Age is z-scored using mean and std computed on outer-train subjects only.
    - The same scaler is applied to outer-test subjects.

    ## Hyperparameter and threshold selection

    - Inner 5-fold CV is performed entirely within the outer-training set.
    - The threshold is selected from inner OOF predictions (not outer-test scores).
    - Outer-test set is evaluated exactly once per fold.

    ## ROI channel restriction

    - Only the Pearson covariance channel (via LedoitWolf from raw time series)
      is used for the tangent transform.
    - OMST, MI, dFC, DistanceCorr, and Granger channels are NOT used.

    ## Verdict: no leakage detected in design
    """))

    # ---- final_recommendation.md ------------------------------------------
    best_row = primary_df.sort_values("AUC_mean", ascending=False).iloc[0]
    best_clf = best_row["classifier"]
    best_auc = best_row["AUC_mean"]
    best_prauc = best_row["PR_AUC_mean"]
    locked_auc = LOCKED_METRICS["AUC"]
    locked_prauc = LOCKED_METRICS["PR_AUC"]
    promotes = best_auc > locked_auc and best_prauc >= locked_prauc
    verdict = "PROMOTE as supplemental baseline" if (best_auc > locked_auc * 0.97) else (
        "DO NOT promote — below VAE performance threshold"
    )
    (OUTPUT_DIR / "final_recommendation.md").write_text(textwrap.dedent(f"""\
    # Final Recommendation

    **Generated:** {t0.strftime('%Y-%m-%d %H:%M:%S UTC')}

    ## Best tangent-space classifier

    | Model | AUC | PR-AUC | BA |
    | --- | --- | --- | --- |
    | Locked VAE ({{}}) | {locked_auc:.4f} | {locked_prauc:.4f} | {LOCKED_METRICS['balanced_accuracy']:.4f} |
    | Tangent {best_clf} | {best_auc:.4f} | {best_prauc:.4f} | {best_row['balanced_accuracy_mean']:.4f} |

    ## Verdict

    **{verdict}**

    - Promotes to supplemental baseline if tangent AUC ≥ locked VAE AUC × 0.97
    - Promotes to main model only if AUC AND PR-AUC both exceed locked VAE metrics
    - Current: AUC diff = {best_auc - locked_auc:+.4f}, PR-AUC diff = {best_prauc - locked_prauc:+.4f}

    ## Notes

    - No global tangent fit; reference fitted per fold on outer-train subjects.
    - LedoitWolf shrinkage applied per subject from raw 140 × 131 time series.
    - 131 AAL3 ROIs (natural atlas order, not Yeo17 reordered).
    - Threshold strategy: inner_oof_target_sens_ge_0p70_max_spec.
    """).format("adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"))

    # ---- README.md ---------------------------------------------------------
    (OUTPUT_DIR / "README.md").write_text(textwrap.dedent(f"""\
    # Tangent-Space FC Classifier Baseline

    **Generated:** {t0.strftime('%Y-%m-%d %H:%M:%S UTC')}
    **Script:** `scripts/revision_bspc_2026/run_tangent_space_fc_classifier_baseline.py`

    ## Design

    Fold-safe tangent-space functional connectivity classifier using the same outer-fold
    subject splits as the locked v5.1b VAE model.

    | Parameter | Value |
    | --- | --- |
    | ROI time series | 140 TPs × 131 ROIs (AAL3, LedoitWolf) |
    | Tangent reference | Geometric mean, outer-train subjects only |
    | Vectorization | Upper triangle, 8515 features |
    | Extra features | Age (z-scored, train-fitted) + Sex binary |
    | Total features | 8517 |
    | Outer folds | 5 (same splits as VAE model) |
    | Inner folds | 5 (hyperparameter selection) |
    | Classifiers | logreg_l2, logreg_elasticnet |
    | Threshold | inner_oof_target_sens_ge_0p70_max_spec |

    ## Files

    | File | Contents |
    | --- | --- |
    | data_availability_audit.csv/.md | Subject coverage, .mat file lookup |
    | foldwise_metrics.csv/.md | Per-fold AUC, PR-AUC, BA, Sens, Spec, F1 |
    | primary_results.csv/.md | Pooled mean ± std vs locked VAE |
    | selected_hyperparameters.csv/.md | Best C, l1_ratio per fold |
    | oof_predictions_logreg_l2.csv | OOF scores per subject |
    | oof_predictions_logreg_elasticnet.csv | OOF scores per subject |
    | leakage_risk_report.md | Fold-safety confirmation |
    | final_recommendation.md | Promote/don't promote decision |
    | command_log.json | Execution metadata |
    """))


if __name__ == "__main__":
    main()
