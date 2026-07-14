#!/usr/bin/env python3
"""Leakage-safe Stage B score-calibration rescue audit — beta3p75 latent384.

Input:
  recover035_latent384_beta3p75_T80_h10000_p560_full5x5 latent cache

Goal:
  Test whether fold-specific score-scale mismatch can be corrected using
  inner-CV out-of-fold train/dev scores only.

Calibration methods (all parameters learned from inner OOF train/dev only):
  1. raw          — predict_proba, no calibration (baseline)
  2. oof_zscore   — standardise with OOF mean/std, then sigmoid
  3. oof_logitz   — logit-transform, standardise with OOF logit mean/std, sigmoid back
  4. oof_ecdf     — map test scores through empirical CDF of OOF scores
  5. oof_platt    — fit logistic sigmoid on (OOF scores, OOF labels), apply to test
  6. oof_isotonic — fit isotonic regression on (OOF scores, OOF labels),
                    only if n_pos_oof >= MIN_POSITIVES_ISOTONIC

Readouts:
  logreg_l2_original  (C grid: original 4-point [0.001, 0.01, 0.1, 1.0])
  logreg_elasticnet   (C + l1_ratio grid)

Feature sets:
  z_plus_age_sex, z_only

Leakage-safety guarantee:
  All calibration parameters estimated from inner-CV OOF train/dev scores.
  Outer-test labels are NEVER used for calibration or threshold fitting.
  Rank-normalisation using outer-test ranks is DESCRIPTIVE ONLY.

Promotion criteria (primary threshold, z_plus_age_sex):
  pooled AUC > 0.782951  AND  pooled PR-AUC >= 0.559873

Hard constraints:
  - No VAE retraining.
  - No threshold fitting on outer test data.
  - Do not modify tensors, metadata, ledger, or existing run outputs.
  - A subject flagged during metadata recovery must not appear in the pool
    (identifier withheld from this public copy; see the private run
    metadata for the specific ID).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"

DEFAULT_RUN_DIR = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
DEFAULT_OUTPUT_DIR = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"

LOCKED_AUC = 0.782951
LOCKED_PR_AUC = 0.559873
EXISTING_POOLED_AUC = 0.760
EXISTING_POOLED_PR_AUC = 0.50914

ORIGINAL_C_GRID = [0.001, 0.01, 0.1, 1.0]
ELASTICNET_C_GRID = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
ELASTICNET_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]

TARGET_SENSITIVITY = 0.70
FOLDS = [1, 2, 3, 4, 5]
INNER_FOLDS = 5
SEED = 42
MIN_POSITIVES_ISOTONIC = 10

MODELS = ["logreg_l2_original", "logreg_elasticnet"]
FEATURE_SETS = ["z_plus_age_sex", "z_only"]
CALIB_METHODS = ["raw", "oof_zscore", "oof_logitz", "oof_ecdf", "oof_platt", "oof_isotonic"]

PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_FEATURE_SET = "z_plus_age_sex"


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--folds", type=int, nargs="*", default=FOLDS)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate paths and print config without running classifiers.")
    return p.parse_args()


# ── Utilities ────────────────────────────────────────────────────────────────

def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def md_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def write_table(outdir: Path, stem: str, df: pd.DataFrame, max_rows: int = 100) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


# ── Metrics ──────────────────────────────────────────────────────────────────

def binary_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    y_pred: Sequence[int],
) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: Dict[str, Any] = {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "balanced_accuracy": float(np.nanmean([safe_div(tp, tp + fn), safe_div(tn, tn + fp)])),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "predicted_ad_rate": float(pred.mean()) if len(pred) else float("nan"),
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
    else:
        out["auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


# ── Threshold selection (from inner OOF only) ────────────────────────────────

def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    s = scores[np.isfinite(scores)]
    return np.unique(np.round(np.clip(np.concatenate(([0.0, 0.5, 1.0], s)), 0.0, 1.0), 12))


def select_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> List[Dict[str, Any]]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    rows = []
    for thr in threshold_candidates(s):
        pred = (s >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)
        rows.append({
            "threshold": float(thr),
            "sensitivity": sens,
            "specificity": spec,
            "balanced_accuracy": float(np.nanmean([sens, spec])),
            "youden_j": sens + spec - 1.0,
        })
    tbl = pd.DataFrame(rows)

    selections: List[Dict[str, Any]] = []
    for criterion, metric in [("inner_oof_youden_j", "youden_j"),
                               ("inner_oof_balanced_accuracy", "balanced_accuracy")]:
        r = tbl.sort_values(
            [metric, "sensitivity", "specificity", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
        selections.append({
            "threshold_strategy": criterion,
            "threshold": float(r["threshold"]),
            "inner_oof_sensitivity": float(r["sensitivity"]),
            "inner_oof_specificity": float(r["specificity"]),
            "inner_oof_balanced_accuracy": float(r["balanced_accuracy"]),
        })

    eligible = tbl[tbl["sensitivity"] >= TARGET_SENSITIVITY]
    if eligible.empty:
        r = tbl.sort_values(["sensitivity", "specificity", "threshold"],
                             ascending=[False, False, False]).iloc[0]
        status = "target_not_reached"
    else:
        r = eligible.sort_values(
            ["specificity", "sensitivity", "balanced_accuracy", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
        status = "selected"
    selections.append({
        "threshold_strategy": "inner_oof_target_sens_ge_0p70_max_spec",
        "threshold": float(r["threshold"]),
        "inner_oof_sensitivity": float(r["sensitivity"]),
        "inner_oof_specificity": float(r["specificity"]),
        "inner_oof_balanced_accuracy": float(r["balanced_accuracy"]),
        "threshold_selection_status": status,
    })
    selections.append({
        "threshold_strategy": "fixed_0p5",
        "threshold": 0.5,
        "inner_oof_sensitivity": float("nan"),
        "inner_oof_specificity": float("nan"),
        "inner_oof_balanced_accuracy": float("nan"),
    })
    return selections


# ── Preprocessing ────────────────────────────────────────────────────────────

def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(mu_cols: List[str], include_age_sex: bool) -> ColumnTransformer:
    transformers: list = [("latent", Pipeline([("scaler", StandardScaler())]), mu_cols)]
    if include_age_sex:
        transformers.append(("age",
            Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]),
            ["Age"]))
        transformers.append(("sex",
            Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", make_ohe())]),
            ["Sex"]))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


# ── Classifier specs ─────────────────────────────────────────────────────────

def classifier_specs(fold: int) -> Dict[str, Tuple[Pipeline, Dict[str, list]]]:
    seed = SEED + fold
    return {
        "logreg_l2_original": (
            Pipeline([
                ("pre", "passthrough"),
                ("model", LogisticRegression(
                    penalty="l2", solver="lbfgs", class_weight="balanced",
                    max_iter=5000, random_state=seed,
                )),
            ]),
            {"model__C": ORIGINAL_C_GRID},
        ),
        "logreg_elasticnet": (
            Pipeline([
                ("pre", "passthrough"),
                ("model", LogisticRegression(
                    penalty="elasticnet", solver="saga", class_weight="balanced",
                    max_iter=5000, random_state=seed, n_jobs=1,
                )),
            ]),
            {
                "model__C": ELASTICNET_C_GRID,
                "model__l1_ratio": ELASTICNET_L1_RATIOS,
            },
        ),
    }


def score_1d(estimator: Any, x: Any) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(x)[:, 1], dtype=float)
    raw = np.asarray(estimator.decision_function(x), dtype=float).ravel()
    return 1.0 / (1.0 + np.exp(-raw))


# ── Stratification key ────────────────────────────────────────────────────────

def inner_stratification_key(df: pd.DataFrame, n_splits: int) -> Tuple[Any, str]:
    cols = ["ResearchGroup_Mapped", "Manufacturer"]
    key_df = df[cols].copy()
    for col in cols:
        key_df[col] = key_df[col].fillna(f"{col}_UNKNOWN").astype(str)
    key = key_df.apply(lambda r: "_".join(r.values.astype(str)), axis=1)
    if int(key.value_counts().min()) < n_splits:
        return df["y"].astype(int), "label_only_fallback"
    return key, "ResearchGroup_Mapped+Manufacturer"


def load_latent_pair(cache_dir: Path, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(cache_dir / f"fold_{fold}_trainDev_latent_mu.csv")
    test = pd.read_csv(cache_dir / f"fold_{fold}_test_latent_mu.csv")
    return train, test


# ── Calibration ──────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def calibrate(
    oof_scores: np.ndarray,
    test_scores: np.ndarray,
    oof_labels: np.ndarray,
    method: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return (calibrated_oof, calibrated_test, meta_dict).

    Calibration parameters are learned exclusively from oof_scores (and
    oof_labels for Platt/isotonic). Outer-test labels are never used.
    """
    meta: Dict[str, Any] = {"method": method, "skipped": False}

    if method == "raw":
        return oof_scores.copy(), test_scores.copy(), meta

    if method == "oof_zscore":
        mu = float(np.mean(oof_scores))
        sigma = max(float(np.std(oof_scores, ddof=1)), 1e-8)
        meta.update({"oof_mean": round(mu, 6), "oof_std": round(sigma, 6)})
        return _sigmoid((oof_scores - mu) / sigma), _sigmoid((test_scores - mu) / sigma), meta

    if method == "oof_logitz":
        eps = 1e-6
        oof_l = np.log(np.clip(oof_scores, eps, 1 - eps) / (1 - np.clip(oof_scores, eps, 1 - eps)))
        test_l = np.log(np.clip(test_scores, eps, 1 - eps) / (1 - np.clip(test_scores, eps, 1 - eps)))
        mu = float(np.mean(oof_l))
        sigma = max(float(np.std(oof_l, ddof=1)), 1e-8)
        meta.update({"oof_logit_mean": round(mu, 6), "oof_logit_std": round(sigma, 6)})
        return _sigmoid((oof_l - mu) / sigma), _sigmoid((test_l - mu) / sigma), meta

    if method == "oof_ecdf":
        sorted_oof = np.sort(oof_scores)
        n = len(sorted_oof)
        pctiles = (np.arange(1, n + 1) - 0.5) / n   # avoids exact 0.0 and 1.0
        cal_oof = np.interp(oof_scores, sorted_oof, pctiles, left=0.0, right=1.0)
        cal_test = np.interp(test_scores, sorted_oof, pctiles, left=0.0, right=1.0)
        meta.update({
            "oof_n": n,
            "oof_score_min": round(float(sorted_oof[0]), 6),
            "oof_score_max": round(float(sorted_oof[-1]), 6),
        })
        return cal_oof, cal_test, meta

    if method == "oof_platt":
        # Classic Platt scaling: sigmoid fit on OOF scores → OOF labels.
        # C=1e10 ≈ unconstrained; only 1 feature, so no overfitting risk.
        platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=10000)
        platt.fit(oof_scores.reshape(-1, 1), oof_labels)
        cal_oof = platt.predict_proba(oof_scores.reshape(-1, 1))[:, 1]
        cal_test = platt.predict_proba(test_scores.reshape(-1, 1))[:, 1]
        meta.update({
            "platt_intercept": round(float(platt.intercept_[0]), 6),
            "platt_coef": round(float(platt.coef_[0, 0]), 6),
        })
        return cal_oof, cal_test, meta

    if method == "oof_isotonic":
        n_pos = int(oof_labels.sum())
        meta["n_pos_oof"] = n_pos
        if n_pos < MIN_POSITIVES_ISOTONIC:
            meta["skipped"] = True
            meta["reason"] = f"n_pos_oof={n_pos} < {MIN_POSITIVES_ISOTONIC}"
            return oof_scores.copy(), test_scores.copy(), meta
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_scores, oof_labels)
        return iso.predict(oof_scores), iso.predict(test_scores), meta

    raise ValueError(f"Unknown calibration method: {method!r}")


# ── Main sweep ───────────────────────────────────────────────────────────────

def run_calib_sweep(
    cache_dir: Path,
    folds: List[int],
    n_jobs: int,
) -> Dict[str, pd.DataFrame]:
    fold_metric_rows: List[Dict[str, Any]] = []
    pred_rows: List[pd.DataFrame] = []
    score_range_rows: List[Dict[str, Any]] = []
    calib_meta_rows: List[Dict[str, Any]] = []
    philips_rows: List[Dict[str, Any]] = []

    for fold in folds:
        print(f"  Fold {fold} ...", flush=True)
        train_df, test_df = load_latent_pair(cache_dir, fold)
        mu_cols = [c for c in train_df.columns if c.startswith("mu_")]
        y_train = train_df["y"].astype(int).to_numpy()
        y_test = test_df["y"].astype(int).to_numpy()

        inner_key, inner_context = inner_stratification_key(train_df, INNER_FOLDS)
        inner_cv = list(StratifiedKFold(
            n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + fold + 30,
        ).split(np.zeros(len(train_df)), inner_key))

        specs = classifier_specs(fold)

        for feature_set in FEATURE_SETS:
            include_age_sex = feature_set == "z_plus_age_sex"
            feature_cols = mu_cols + (["Age", "Sex"] if include_age_sex else [])
            x_train = train_df[feature_cols].copy()
            x_test = test_df[feature_cols].copy()
            pre = make_preprocessor(mu_cols, include_age_sex=include_age_sex)

            for model_name in MODELS:
                base_pipe, grid = specs[model_name]
                pipe = clone(base_pipe)
                pipe.steps[0] = ("pre", pre)

                # ── GridSearchCV: best hyperparams ─────────────────────────
                search = GridSearchCV(
                    estimator=pipe,
                    param_grid=grid,
                    scoring="roc_auc",
                    cv=inner_cv,
                    n_jobs=n_jobs,
                    refit=True,
                    error_score=np.nan,
                )
                search.fit(x_train, y_train)
                best = search.best_estimator_

                # ── Inner OOF scores (same splits as GridSearchCV) ─────────
                try:
                    oof_proba = cross_val_predict(
                        clone(best), x_train, y_train,
                        cv=inner_cv, method="predict_proba", n_jobs=n_jobs,
                    )
                    oof_scores_raw = oof_proba[:, 1].astype(float)
                except Exception as exc:
                    print(f"    WARNING: cross_val_predict failed "
                          f"fold={fold} {model_name}/{feature_set}: {exc}", flush=True)
                    continue

                # ── Outer test scores ──────────────────────────────────────
                test_scores_raw = score_1d(best, x_test)

                oof_range = float(np.ptp(oof_scores_raw))
                test_range = float(np.ptp(test_scores_raw))
                print(
                    f"    {model_name}/{feature_set}: "
                    f"params={search.best_params_}  "
                    f"inner_auc={search.best_score_:.4f}  "
                    f"oof_range={oof_range:.4f}  test_range={test_range:.4f}",
                    flush=True,
                )

                # ── Apply each calibration method ──────────────────────────
                for calib in CALIB_METHODS:
                    cal_oof, cal_test, meta = calibrate(
                        oof_scores_raw, test_scores_raw, y_train, calib,
                    )

                    calib_meta_rows.append({
                        "fold": fold,
                        "model_name": model_name,
                        "feature_set": feature_set,
                        "calib_method": calib,
                        "best_params": json.dumps(search.best_params_, sort_keys=True),
                        "inner_auc": round(float(search.best_score_), 6),
                        "oof_range_raw": round(oof_range, 6),
                        "test_range_raw": round(test_range, 6),
                        "cal_oof_range": round(float(np.ptp(cal_oof)), 6),
                        "cal_test_range": round(float(np.ptp(cal_test)), 6),
                        **{k: str(v) for k, v in meta.items()},
                    })

                    if meta.get("skipped"):
                        continue

                    # ── Threshold selection from calibrated OOF ────────────
                    thresholds = select_thresholds(y_train, cal_oof)

                    # ── Score range diagnostics ────────────────────────────
                    for dx_code, dx_name in [(0, "CN"), (1, "AD"), (-1, "ALL")]:
                        s = cal_test if dx_code == -1 else cal_test[y_test == dx_code]
                        if len(s) == 0:
                            continue
                        score_range_rows.append({
                            "fold": fold,
                            "model_name": model_name,
                            "feature_set": feature_set,
                            "calib_method": calib,
                            "diagnosis": dx_name,
                            "n": len(s),
                            "min": float(np.min(s)),
                            "p10": float(np.percentile(s, 10)),
                            "p25": float(np.percentile(s, 25)),
                            "median": float(np.median(s)),
                            "p75": float(np.percentile(s, 75)),
                            "p90": float(np.percentile(s, 90)),
                            "max": float(np.max(s)),
                            "score_range": float(np.max(s) - np.min(s)),
                        })

                    for sel in thresholds:
                        thr = float(sel["threshold"])
                        y_pred = (cal_test >= thr).astype(int)

                        row: Dict[str, Any] = {
                            "fold": fold,
                            "model_name": model_name,
                            "feature_set": feature_set,
                            "calib_method": calib,
                            "threshold_strategy": sel["threshold_strategy"],
                            "threshold": thr,
                            "best_inner_auc": float(search.best_score_),
                            "best_params": json.dumps(search.best_params_, sort_keys=True),
                            "inner_cv_context": inner_context,
                        }
                        row.update({k: v for k, v in sel.items() if k != "threshold_strategy"})
                        row.update(binary_metrics(y_test, cal_test, y_pred))
                        fold_metric_rows.append(row)

                        # ── Per-prediction rows ────────────────────────────
                        p = test_df[[
                            "SubjectID", "tensor_idx", "ResearchGroup_Mapped",
                            "Manufacturer", "Age", "Sex",
                        ]].copy()
                        p["fold"] = fold
                        p["model_name"] = model_name
                        p["feature_set"] = feature_set
                        p["calib_method"] = calib
                        p["threshold_strategy"] = sel["threshold_strategy"]
                        p["threshold"] = thr
                        p["y_true"] = y_test
                        p["y_score_raw"] = test_scores_raw
                        p["y_score"] = cal_test
                        p["y_pred"] = y_pred
                        pred_rows.append(p)

                        # ── Philips CN FP check ────────────────────────────
                        if "Manufacturer" in test_df.columns:
                            for mfr in ["Philips", "SIEMENS", "GE"]:
                                mfr_mask = test_df["Manufacturer"].astype(str).str.upper() == mfr.upper()
                                mfr_cn = mfr_mask & (y_test == 0)
                                mfr_ad = mfr_mask & (y_test == 1)
                                n_cn_mfr = int(mfr_cn.sum())
                                n_ad_mfr = int(mfr_ad.sum())
                                if n_cn_mfr > 0:
                                    fp_cn = int((y_pred[mfr_cn] == 1).sum())
                                    sens_ad = float("nan") if n_ad_mfr == 0 else float(
                                        (y_pred[mfr_ad] == 1).sum()) / n_ad_mfr
                                    philips_rows.append({
                                        "fold": fold,
                                        "model_name": model_name,
                                        "feature_set": feature_set,
                                        "calib_method": calib,
                                        "threshold_strategy": sel["threshold_strategy"],
                                        "manufacturer": mfr,
                                        "n_cn": n_cn_mfr,
                                        "fp_cn": fp_cn,
                                        "fpr_cn": round(fp_cn / n_cn_mfr, 6),
                                        "specificity_cn": round(1 - fp_cn / n_cn_mfr, 6),
                                        "n_ad": n_ad_mfr,
                                        "sensitivity_ad": round(sens_ad, 6) if not np.isnan(sens_ad) else float("nan"),
                                    })

    preds = pd.concat(pred_rows, ignore_index=True, sort=False) if pred_rows else pd.DataFrame()
    return {
        "foldwise_metrics": pd.DataFrame(fold_metric_rows),
        "predictions": preds,
        "score_range_by_fold": pd.DataFrame(score_range_rows),
        "calib_meta": pd.DataFrame(calib_meta_rows),
        "philips_cn_fpr": pd.DataFrame(philips_rows),
    }


# ── Aggregation ──────────────────────────────────────────────────────────────

def pooled_from_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["model_name", "feature_set", "calib_method", "threshold_strategy"]
    for keys, sub in pred.groupby(group_cols, dropna=False):
        model, fs, calib, strategy = keys
        row: Dict[str, Any] = {
            "model_name": model,
            "feature_set": fs,
            "calib_method": calib,
            "threshold_strategy": strategy,
        }
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["promotes"] = (df["auc"] > LOCKED_AUC) & (df["pr_auc"] >= LOCKED_PR_AUC)
    return df.sort_values(["calib_method", "threshold_strategy", "feature_set", "model_name"])


def pooled_vs_foldwise(pred: pd.DataFrame, foldwise: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["model_name", "feature_set", "calib_method", "threshold_strategy"]
    for keys, sub in pred.groupby(group_cols, dropna=False):
        model, fs, calib, strategy = keys
        uniq = np.unique(sub["y_true"])
        pooled_auc = float(roc_auc_score(sub["y_true"], sub["y_score"])) if len(uniq) == 2 else float("nan")
        pooled_pr = float(average_precision_score(sub["y_true"], sub["y_score"])) if len(uniq) == 2 else float("nan")

        mask = (
            (foldwise["model_name"] == model) &
            (foldwise["feature_set"] == fs) &
            (foldwise["calib_method"] == calib) &
            (foldwise["threshold_strategy"] == strategy)
        )
        sub_fw = foldwise[mask]
        fw_auc = float(sub_fw["auc"].mean()) if not sub_fw.empty else float("nan")
        fw_pr = float(sub_fw["pr_auc"].mean()) if not sub_fw.empty else float("nan")

        # Rank-normalise within each fold (DESCRIPTIVE ONLY)
        rn_parts: List[pd.DataFrame] = []
        for _, fsub in sub.groupby("fold"):
            fsub2 = fsub.copy()
            n = len(fsub2)
            fsub2["y_score_rn"] = fsub2["y_score"].rank(method="average").subtract(0.5).divide(n)
            rn_parts.append(fsub2)
        rn_all = pd.concat(rn_parts, ignore_index=True) if rn_parts else pd.DataFrame()
        if not rn_all.empty and len(np.unique(rn_all["y_true"])) == 2:
            rn_auc = float(roc_auc_score(rn_all["y_true"], rn_all["y_score_rn"]))
        else:
            rn_auc = float("nan")

        rows.append({
            "model_name": model,
            "feature_set": fs,
            "calib_method": calib,
            "threshold_strategy": strategy,
            "pooled_auc": pooled_auc,
            "pooled_pr_auc": pooled_pr,
            "foldwise_mean_auc": fw_auc,
            "foldwise_mean_pr_auc": fw_pr,
            "gap_auc_fw_minus_pooled": fw_auc - pooled_auc,
            "ranknorm_auc_DESCRIPTIVE_ONLY": rn_auc,
            "promotes": bool(pooled_auc > LOCKED_AUC and pooled_pr >= LOCKED_PR_AUC),
        })
    return pd.DataFrame(rows).sort_values(
        ["model_name", "feature_set", "calib_method", "threshold_strategy"]
    )


def philips_pooled(philips: pd.DataFrame) -> pd.DataFrame:
    if philips.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    group_cols = ["model_name", "feature_set", "calib_method", "threshold_strategy", "manufacturer"]
    for keys, sub in philips.groupby(group_cols, dropna=False):
        model, fs, calib, strategy, mfr = keys
        n_cn = int(sub["n_cn"].sum())
        fp_cn = int(sub["fp_cn"].sum())
        rows.append({
            "model_name": model,
            "feature_set": fs,
            "calib_method": calib,
            "threshold_strategy": strategy,
            "manufacturer": mfr,
            "n_cn_pooled": n_cn,
            "fp_cn_pooled": fp_cn,
            "fpr_cn_pooled": round(fp_cn / n_cn, 4) if n_cn > 0 else float("nan"),
            "specificity_cn_pooled": round(1 - fp_cn / n_cn, 4) if n_cn > 0 else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(
        ["manufacturer", "calib_method", "model_name", "feature_set", "threshold_strategy"]
    )


# ── Report ────────────────────────────────────────────────────────────────────

def write_final_report(
    outdir: Path,
    pooled: pd.DataFrame,
    pvf: pd.DataFrame,
    score_range: pd.DataFrame,
    philips_pool: pd.DataFrame,
) -> None:
    now = datetime.now(timezone.utc).isoformat()

    # primary-threshold, z_plus_age_sex slice
    def primary_slice(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        mask = (
            (df.get("threshold_strategy", pd.Series()) == PRIMARY_THRESHOLD) &
            (df.get("feature_set", pd.Series()) == PRIMARY_FEATURE_SET)
        )
        return df[mask].copy() if mask.any() else df.copy()

    sub_primary = primary_slice(pooled)
    best_row = (
        sub_primary.sort_values("auc", ascending=False).iloc[0]
        if not sub_primary.empty and "auc" in sub_primary.columns
        else None
    )

    best_auc = float(best_row["auc"]) if best_row is not None else float("nan")
    best_pr = float(best_row["pr_auc"]) if best_row is not None else float("nan")
    best_calib = str(best_row["calib_method"]) if best_row is not None else "n/a"
    best_model = str(best_row["model_name"]) if best_row is not None else "n/a"
    promotes = bool(best_auc > LOCKED_AUC and best_pr >= LOCKED_PR_AUC)
    verdict = "PROMOTES" if promotes else "DOES NOT PROMOTE"

    lines = [
        "# Stage B OOF Score Calibration Rescue Audit",
        "## recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        "",
        f"Generated: {now}",
        "",
        "## Scope",
        "Leakage-safe score calibration using inner-CV OOF scores only.",
        "No VAE retraining. No threshold fitting on outer test.",
        "No outer-test labels used for calibration at any stage.",
        "",
        "## Calibration Methods",
        "1. **raw** — predict_proba, no calibration (baseline)",
        "2. **oof_zscore** — standardise (OOF mean/std) → sigmoid",
        "3. **oof_logitz** — logit-transform, standardise (OOF logit mean/std) → sigmoid",
        "4. **oof_ecdf** — empirical CDF mapping: test score → percentile in OOF distribution",
        "5. **oof_platt** — logistic sigmoid fit on (OOF scores, OOF labels); C=1e10 ≈ unconstrained",
        "6. **oof_isotonic** — isotonic regression on (OOF scores, OOF labels); "
        f"only if n_pos_oof ≥ {MIN_POSITIVES_ISOTONIC}",
        "",
        "## Promotion Criteria",
        f"- pooled AUC > {LOCKED_AUC}  AND  pooled PR-AUC >= {LOCKED_PR_AUC}  (both must pass)",
        "",
        "## Primary Result (primary threshold, z_plus_age_sex)",
        f"- Best calibration:  {best_calib}  /  model: {best_model}",
        f"- Pooled AUC:   {best_auc:.4f}  "
        f"({'PASS' if best_auc > LOCKED_AUC else 'FAIL'} vs locked {LOCKED_AUC})",
        f"- Pooled PR-AUC: {best_pr:.4f}  "
        f"({'PASS' if best_pr >= LOCKED_PR_AUC else 'FAIL'} vs locked {LOCKED_PR_AUC})",
        f"- **Verdict: {verdict}**",
        "",
        "## Baseline Reference",
        f"- Existing raw readout (logreg_l2, z_plus_age_sex): "
        f"AUC={EXISTING_POOLED_AUC}  PR-AUC={EXISTING_POOLED_PR_AUC}",
        "",
        "## Pooled Metrics — Primary Threshold, z_plus_age_sex",
        "_(all calibration methods × all models, sorted by AUC descending)_",
    ]

    if not sub_primary.empty:
        disp_cols = [c for c in [
            "model_name", "calib_method", "auc", "pr_auc",
            "balanced_accuracy", "sensitivity", "specificity", "f1", "promotes",
        ] if c in sub_primary.columns]
        lines.append(sub_primary[disp_cols].sort_values("auc", ascending=False).to_markdown(index=False))
    else:
        lines.append("_No data._")

    lines += ["", "## Pooled vs Foldwise AUC Gap (primary threshold, z_plus_age_sex)",
              "_(gap reduction = calibration addressing scale mismatch)_",
              "_(ranknorm is DESCRIPTIVE ONLY — not promotable)_"]
    sub_pvf = primary_slice(pvf)
    if not sub_pvf.empty:
        pvf_cols = [c for c in [
            "model_name", "calib_method",
            "pooled_auc", "pooled_pr_auc",
            "foldwise_mean_auc", "gap_auc_fw_minus_pooled",
            "ranknorm_auc_DESCRIPTIVE_ONLY", "promotes",
        ] if c in sub_pvf.columns]
        lines.append(sub_pvf[pvf_cols].to_markdown(index=False))
    else:
        lines.append("_No data._")

    lines += ["",
              "## Score Range by Fold (ALL subjects, primary threshold, z_plus_age_sex)",
              "_Is the fold-1 pathological range reduced after calibration?_"]
    if not score_range.empty:
        sr_mask = (
            (score_range["diagnosis"] == "ALL") &
            (score_range["feature_set"] == PRIMARY_FEATURE_SET)
        )
        sr_sub = score_range[sr_mask]
        if not sr_sub.empty:
            sr_disp = sr_sub[[
                "fold", "model_name", "calib_method", "n",
                "min", "median", "max", "score_range",
            ]].sort_values(["model_name", "calib_method", "fold"])
            lines.append(sr_disp.to_markdown(index=False))
        else:
            lines.append("_No data._")
    else:
        lines.append("_No data._")

    lines += ["", "## Philips CN FP Rate — Pooled (primary threshold, z_plus_age_sex)"]
    if not philips_pool.empty:
        ph_mask = (
            (philips_pool["threshold_strategy"] == PRIMARY_THRESHOLD) &
            (philips_pool["feature_set"] == PRIMARY_FEATURE_SET) &
            (philips_pool["manufacturer"].str.upper() == "PHILIPS")
        )
        ph_sub = philips_pool[ph_mask]
        if not ph_sub.empty:
            ph_cols = [c for c in [
                "model_name", "calib_method",
                "n_cn_pooled", "fp_cn_pooled", "fpr_cn_pooled", "specificity_cn_pooled",
            ] if c in ph_sub.columns]
            lines.append(ph_sub[ph_cols].sort_values("fpr_cn_pooled").to_markdown(index=False))
        else:
            lines.append("_No Philips CN data._")
    else:
        lines.append("_No Philips data available._")

    lines += [
        "",
        "## Key Diagnostic Questions",
        "1. Does any calibration method close the pooled vs foldwise AUC gap?",
        "2. Is Fold 1 score range reduced from 0.995 (raw) toward other folds?",
        "3. Does any calibration push pooled AUC above 0.783 / PR-AUC above 0.560?",
        "4. Is Philips CN FPR comparable to or better than locked reference?",
        "",
        "## Read-only guarantee",
        "Did not retrain VAE. Did not fit calibration or thresholds on outer test data.",
        "Did not modify tensors, metadata, ledger, or existing run outputs.",
        "Rank-normalised metrics are DESCRIPTIVE ONLY — not used for promotion.",
    ]
    (outdir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    cache_dir = run_dir / "classifier_only_readout" / "latent_cache"
    outdir = resolve(args.output_dir)

    # ── Validate inputs ────────────────────────────────────────────────────
    print("Leakage-safe Stage B score calibration rescue audit")
    print(f"  run_dir:    {run_dir}")
    print(f"  cache_dir:  {cache_dir}")
    print(f"  output_dir: {outdir}")
    print(f"  folds:      {args.folds}")
    print(f"  n_jobs:     {args.n_jobs}")
    print(f"  models:     {MODELS}")
    print(f"  calib:      {CALIB_METHODS}")
    print(f"  feature sets: {FEATURE_SETS}")

    missing = []
    for fold in args.folds:
        for split in ["trainDev", "test"]:
            p = cache_dir / f"fold_{fold}_{split}_latent_mu.csv"
            if not p.exists():
                missing.append(str(p))
    if missing:
        print("MISSING cache files:")
        for m in missing:
            print(f"  {m}")
        return 1
    print(f"  All {len(args.folds) * 2} cache files found.")

    if args.dry_run:
        print("\nDry-run complete. No classifiers trained.")
        return 0

    if outdir.exists() and not args.overwrite:
        print(f"\nOutput dir exists; pass --overwrite to replace: {outdir}")
        return 1
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Run sweep ──────────────────────────────────────────────────────────
    print("\nRunning calibration sweep ...", flush=True)
    results = run_calib_sweep(cache_dir, folds=args.folds, n_jobs=args.n_jobs)

    foldwise = results["foldwise_metrics"]
    preds = results["predictions"]
    score_range = results["score_range_by_fold"]
    calib_meta = results["calib_meta"]
    philips = results["philips_cn_fpr"]

    # ── Compute aggregations ───────────────────────────────────────────────
    pooled = pooled_from_predictions(preds) if not preds.empty else pd.DataFrame()
    pvf = pooled_vs_foldwise(preds, foldwise) if not preds.empty and not foldwise.empty else pd.DataFrame()
    philips_pool = philips_pooled(philips) if not philips.empty else pd.DataFrame()

    # ── Write tables ───────────────────────────────────────────────────────
    write_table(outdir, "calib_foldwise_metrics", foldwise)
    write_table(outdir, "calib_pooled_metrics", pooled)
    write_table(outdir, "calib_pooled_vs_foldwise", pvf)
    write_table(outdir, "calib_score_range_by_fold", score_range)
    write_table(outdir, "calib_meta", calib_meta)
    write_table(outdir, "calib_philips_fpr_by_fold", philips)
    write_table(outdir, "calib_philips_fpr_pooled", philips_pool)
    if not preds.empty:
        preds.to_csv(outdir / "calib_predictions.csv", index=False)

    # ── Write report ───────────────────────────────────────────────────────
    write_final_report(outdir, pooled, pvf, score_range, philips_pool)

    # ── Write command log ──────────────────────────────────────────────────
    write_json(outdir / "command_log.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_dir.name,
        "folds": args.folds,
        "models": MODELS,
        "calib_methods": CALIB_METHODS,
        "feature_sets": FEATURE_SETS,
        "training_launched": False,
        "threshold_fitting_on_outer_test": False,
        "calibration_params_source": "inner_cv_oof_only",
    })

    # ── Console summary ────────────────────────────────────────────────────
    if not pooled.empty:
        print("\n=== Pooled AUC summary (primary threshold, z_plus_age_sex) ===")
        pri = pooled[
            (pooled["threshold_strategy"] == PRIMARY_THRESHOLD) &
            (pooled["feature_set"] == PRIMARY_FEATURE_SET)
        ].sort_values("auc", ascending=False)
        if not pri.empty:
            for _, row in pri.iterrows():
                marker = " *** PROMOTES ***" if row.get("promotes") else ""
                print(
                    f"  {row['model_name']:<22} {row['calib_method']:<16} "
                    f"AUC={row['auc']:.4f}  PR-AUC={row['pr_auc']:.4f}  "
                    f"BA={row['balanced_accuracy']:.4f}  F1={row['f1']:.4f}{marker}"
                )

    print(f"\nOutput: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
