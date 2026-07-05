#!/usr/bin/env python
"""Read-only Stage B latent harmonization audit for the promoted beta3p75 model.

This script uses saved fold-local latent mu caches only. It does not retrain the
VAE, modify tensors, modify metadata, or use outer-test labels for
harmonization, threshold selection, or score calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"
DEFAULT_LATENT_CACHE = (
    RESULTS
    / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
    / "classifier_only_readout"
    / "latent_cache"
)
REFERENCE_CALIB_DIR = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
DEFAULT_OUTPUT = (
    RESULTS
    / "promoted_beta3p75_stageB_latent_harmonization_by_manufacturer_20260602"
)

FOLDS = [1, 2, 3, 4, 5]
SEED = 42
ORIGINAL_C_GRID = [0.001, 0.01, 0.1, 1.0]
TARGET_SENSITIVITY = 0.70
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PROMOTED_AUC = 0.795155
PROMOTED_PR_AUC = 0.573934
PROMOTED_PHILIPS_CN_FPR = 0.444


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_cmd(cmd: Sequence[str]) -> Dict[str, Any]:
    proc = subprocess.run(
        list(cmd),
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "cmd": list(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def write_md_table(df: pd.DataFrame, path: Path, max_rows: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df if max_rows is None else df.head(max_rows)
    try:
        text = out.to_markdown(index=False)
    except Exception:
        text = out.to_string(index=False)
    path.write_text(text + "\n", encoding="utf-8")


def save_table(df: pd.DataFrame, csv_path: Path, md_path: Optional[Path] = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if md_path is not None:
        write_md_table(df, md_path)


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def normalize_dx(v: Any) -> str:
    s = str(v).strip().upper()
    if s in {"0", "CN", "CONTROL", "NORMAL"}:
        return "CN"
    if s in {"1", "AD", "AD_DEMENTIA", "DEMENTIA"}:
        return "AD"
    return str(v)


def ensure_y(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "y" in out.columns:
        out["y"] = pd.to_numeric(out["y"], errors="coerce")
    else:
        dx = out["ResearchGroup_Mapped"].map(normalize_dx)
        out["y"] = dx.map({"CN": 0, "AD": 1})
    out = out[out["y"].isin([0, 1])].copy()
    out["y"] = out["y"].astype(int)
    return out


def binary_metrics(y_true: Iterable[int], y_score: Iterable[float], y_pred: Iterable[int]) -> Dict[str, Any]:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_score_arr = np.asarray(list(y_score), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=int)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=labels).ravel()
    out = {
        "n": int(len(y_true_arr)),
        "n_cn": int((y_true_arr == 0).sum()),
        "n_ad": int((y_true_arr == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "balanced_accuracy": safe_div(safe_div(tp, tp + fn) + safe_div(tn, tn + fp), 2),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "predicted_ad_rate": float(np.mean(y_pred_arr)) if len(y_pred_arr) else float("nan"),
    }
    if len(np.unique(y_true_arr)) == 2:
        out["auc"] = float(roc_auc_score(y_true_arr, y_score_arr))
        out["pr_auc"] = float(average_precision_score(y_true_arr, y_score_arr))
    else:
        out["auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (scores >= threshold).astype(int)
    return binary_metrics(y_true, scores, pred)


def select_thresholds(y_true: np.ndarray, scores: np.ndarray) -> List[Dict[str, Any]]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    thresholds = sorted(set(float(x) for x in scores))
    if not thresholds:
        return [{"threshold_strategy": "fixed_0p5", "threshold": 0.5}]

    rows: List[Dict[str, Any]] = [{"threshold_strategy": "fixed_0p5", "threshold": 0.5}]

    # Youden from ROC thresholds, excluding infinities.
    fpr, tpr, roc_thr = roc_curve(y_true, scores)
    finite = np.isfinite(roc_thr)
    if finite.any():
        j = tpr[finite] - fpr[finite]
        idx = int(np.argmax(j))
        rows.append({"threshold_strategy": "inner_oof_youden_j", "threshold": float(roc_thr[finite][idx])})
    else:
        rows.append({"threshold_strategy": "inner_oof_youden_j", "threshold": 0.5})

    # Balanced accuracy threshold.
    best_ba = -np.inf
    best_ba_thr = 0.5
    best_target: Optional[Tuple[float, float, float]] = None
    for thr in thresholds:
        m = threshold_metrics(y_true, scores, thr)
        ba = float(m["balanced_accuracy"])
        sens = float(m["sensitivity"])
        spec = float(m["specificity"])
        if ba > best_ba:
            best_ba = ba
            best_ba_thr = thr
        if sens >= TARGET_SENSITIVITY:
            cand = (spec, ba, thr)
            if best_target is None or cand > best_target:
                best_target = cand
    rows.append({"threshold_strategy": "inner_oof_balanced_accuracy", "threshold": float(best_ba_thr)})
    if best_target is None:
        # Fallback to highest-sensitivity threshold if the target cannot be met.
        target_thr = min(thresholds)
    else:
        target_thr = best_target[2]
    rows.append({"threshold_strategy": PRIMARY_THRESHOLD, "threshold": float(target_thr)})
    return rows


def inner_stratification_key(df: pd.DataFrame, n_splits: int) -> Tuple[pd.Series, str, int]:
    key_df = pd.DataFrame(
        {
            "dx": df["ResearchGroup_Mapped"].fillna("DX_UNKNOWN").astype(str),
            "mfr": df["Manufacturer"].fillna("MFR_UNKNOWN").astype(str),
        }
    )
    key = key_df.apply(lambda r: "_".join(r.values.astype(str)), axis=1)
    min_count = int(key.value_counts().min())
    if min_count < n_splits:
        y_key = df["y"].astype(int)
        return y_key, "label_only_fallback", int(pd.Series(y_key).value_counts().min())
    return key, "ResearchGroup_Mapped+Manufacturer", min_count


def mu_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("mu_")]
    return sorted(cols, key=lambda c: int(c.split("_", 1)[1]))


def make_preprocessor(mu_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("latent", Pipeline([("scaler", StandardScaler())]), mu_cols),
            ("age", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), ["Age"]),
            ("sex", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_ohe())]), ["Sex"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def classifier_pipeline(fold: int, mu_cols: List[str]) -> Pipeline:
    return Pipeline(
        [
            ("pre", make_preprocessor(mu_cols)),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=SEED + fold,
                ),
            ),
        ]
    )


def score_estimator(est: Pipeline, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(est.predict_proba(x)[:, 1], dtype=float)


def logit(scores: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(s / (1 - s))


@dataclass
class ScoreCalibration:
    method: str
    threshold_scores: np.ndarray
    test_scores: np.ndarray
    note: str


def calibrate_scores(method: str, y_oof: np.ndarray, oof_scores: np.ndarray, test_scores: np.ndarray) -> ScoreCalibration:
    if method == "raw":
        return ScoreCalibration(method, oof_scores.copy(), test_scores.copy(), "raw scores")
    if method == "oof_logitz":
        x_oof = logit(oof_scores).reshape(-1, 1)
        x_test = logit(test_scores).reshape(-1, 1)
        cal = LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced", max_iter=2000)
        cal.fit(x_oof, y_oof)
        return ScoreCalibration(
            method,
            cal.predict_proba(x_oof)[:, 1],
            cal.predict_proba(x_test)[:, 1],
            "train/dev OOF logit-score logistic calibration",
        )
    if method == "oof_ecdf":
        sorted_oof = np.sort(np.asarray(oof_scores, dtype=float))

        def ecdf(vals: np.ndarray) -> np.ndarray:
            vals = np.asarray(vals, dtype=float)
            # Percentile location against train/dev OOF distribution only.
            return np.searchsorted(sorted_oof, vals, side="right") / max(len(sorted_oof), 1)

        # OOF self-mapping is rank-based and monotonic.
        ranks = pd.Series(oof_scores).rank(method="average").to_numpy()
        oof_ecdf = (ranks - 0.5) / len(oof_scores)
        return ScoreCalibration(method, oof_ecdf.astype(float), ecdf(test_scores).astype(float), "train/dev OOF ECDF mapping")
    raise ValueError(f"Unknown calibration method: {method}")


def residualize_manufacturer_preserve_age_sex(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mu_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    train = train_df.copy()
    test = test_df.copy()

    age_train = pd.to_numeric(train["Age"], errors="coerce")
    age_mean = float(age_train.mean())
    age_std = float(age_train.std(ddof=0) or 1.0)

    def design(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        age = pd.to_numeric(df["Age"], errors="coerce").fillna(age_mean)
        age_z = ((age - age_mean) / age_std).to_numpy(dtype=float).reshape(-1, 1)
        sex = df["Sex"].fillna("UNKNOWN").astype(str).str.upper().map({"M": 1.0, "MALE": 1.0, "F": 0.0, "FEMALE": 0.0})
        sex_arr = sex.fillna(float(np.nanmean(sex.dropna()) if sex.notna().any() else 0.0)).to_numpy(dtype=float).reshape(-1, 1)
        train_mfr_levels = sorted(train["Manufacturer"].fillna("UNKNOWN").astype(str).unique().tolist())
        # Drop the first level as reference. Unknown test levels are all-zero reference.
        dummy_cols = train_mfr_levels[1:]
        mfr = df["Manufacturer"].fillna("UNKNOWN").astype(str)
        dummies = np.zeros((len(df), len(dummy_cols)), dtype=float)
        for i, level in enumerate(dummy_cols):
            dummies[:, i] = (mfr == level).astype(float)
        x_cov = np.concatenate([np.ones((len(df), 1)), age_z, sex_arr, dummies], axis=1)
        return x_cov, dummies, dummy_cols

    x_train, mfr_train, dummy_cols = design(train)
    x_test, mfr_test, _ = design(test)
    coef_mfr = np.zeros((len(dummy_cols), len(mu_cols)), dtype=float)

    y_train = train[mu_cols].to_numpy(dtype=float)
    # OLS multi-output is equivalent to separate per-dimension regressions.
    reg = LinearRegression(fit_intercept=False)
    reg.fit(x_train, y_train)
    coef = np.asarray(reg.coef_, dtype=float)  # n_targets x n_features
    if len(dummy_cols) > 0:
        coef_mfr = coef[:, -len(dummy_cols) :].T
        train_adjust = mfr_train @ coef_mfr
        test_adjust = mfr_test @ coef_mfr
    else:
        train_adjust = 0.0
        test_adjust = 0.0
    train.loc[:, mu_cols] = y_train - train_adjust
    test.loc[:, mu_cols] = test[mu_cols].to_numpy(dtype=float) - test_adjust

    unknown_test = sorted(set(test["Manufacturer"].fillna("UNKNOWN").astype(str)) - set(train["Manufacturer"].fillna("UNKNOWN").astype(str)))
    meta = {
        "method": "residualize_mfr_preserve_age_sex",
        "train_manufacturer_levels": sorted(train["Manufacturer"].fillna("UNKNOWN").astype(str).unique().tolist()),
        "dummy_reference": sorted(train["Manufacturer"].fillna("UNKNOWN").astype(str).unique().tolist())[0],
        "dummy_cols": dummy_cols,
        "unknown_test_manufacturers": unknown_test,
        "age_mean_trainDev": age_mean,
        "age_std_trainDev": age_std,
        "preserved_covariates": "intercept, Age, Sex",
        "removed_terms": "Manufacturer dummy coefficients only",
    }
    return train, test, meta


def combat_train_transform_if_safe(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mu_cols: List[str],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any]]:
    try:
        from neuroCombat import neuroCombat, neuroCombatFromTraining  # type: ignore
    except Exception as exc:
        return None, None, {"method": "combat_mfr_age_sex", "status": "skipped", "reason": f"neuroCombat import failed: {exc}"}

    # The installed neuroCombatFromTraining API has no covariate argument. Using it
    # after a covariate-preserving training fit would not preserve per-subject
    # Age/Sex effects in held-out test subjects. Treat that as unsafe rather than
    # silently mixing train/test distributions.
    import inspect

    sig = str(inspect.signature(neuroCombatFromTraining))
    if "covars" not in sig and "covariates" not in sig:
        return None, None, {
            "method": "combat_mfr_age_sex",
            "status": "skipped",
            "reason": "installed neuroCombatFromTraining lacks held-out covariate support; skipped to avoid non-covariate-preserving outer-test transform",
            "neuroCombatFromTraining_signature": sig,
        }

    # Kept for future-proofing if a covariate-aware version is installed later.
    try:
        train = train_df.copy()
        test = test_df.copy()
        covars = pd.DataFrame(
            {
                "Manufacturer": train["Manufacturer"].astype(str).to_numpy(),
                "Age": pd.to_numeric(train["Age"], errors="coerce").to_numpy(),
                "Sex": train["Sex"].astype(str).to_numpy(),
            }
        )
        out_train = neuroCombat(
            dat=train[mu_cols].to_numpy(dtype=float).T,
            covars=covars,
            batch_col="Manufacturer",
            categorical_cols=["Sex"],
            continuous_cols=["Age"],
        )
        estimates = out_train["estimates"]
        test_covars = pd.DataFrame(
            {
                "Manufacturer": test["Manufacturer"].astype(str).to_numpy(),
                "Age": pd.to_numeric(test["Age"], errors="coerce").to_numpy(),
                "Sex": test["Sex"].astype(str).to_numpy(),
            }
        )
        out_test = neuroCombatFromTraining(
            dat=test[mu_cols].to_numpy(dtype=float).T,
            batch=test["Manufacturer"].astype(str).to_numpy(),
            estimates=estimates,
            covars=test_covars,
        )
        train.loc[:, mu_cols] = np.asarray(out_train["data"], dtype=float).T
        test.loc[:, mu_cols] = np.asarray(out_test["data"], dtype=float).T
        return train, test, {"method": "combat_mfr_age_sex", "status": "fit_ok"}
    except Exception as exc:
        return None, None, {"method": "combat_mfr_age_sex", "status": "skipped", "reason": f"safe ComBat transform failed: {exc}"}


def load_fold(cache_dir: Path, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = ensure_y(pd.read_csv(cache_dir / f"fold_{fold}_trainDev_latent_mu.csv"))
    test = ensure_y(pd.read_csv(cache_dir / f"fold_{fold}_test_latent_mu.csv"))
    return train, test


def fit_score_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold: int,
    harmonization_method: str,
    inner_folds: int,
    n_jobs: int,
) -> Dict[str, pd.DataFrame]:
    mu_cols = mu_columns(train_df)
    y_train = train_df["y"].astype(int).to_numpy()
    y_test = test_df["y"].astype(int).to_numpy()
    x_train = train_df[mu_cols + ["Age", "Sex"]].copy()
    x_test = test_df[mu_cols + ["Age", "Sex"]].copy()

    strat_key, strat_context, min_cell = inner_stratification_key(train_df, inner_folds)
    cv = list(
        StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=SEED + fold + 30).split(
            np.zeros(len(train_df)), strat_key
        )
    )

    pipe = classifier_pipeline(fold, mu_cols)
    search = GridSearchCV(
        estimator=pipe,
        param_grid={"model__C": ORIGINAL_C_GRID},
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        error_score=np.nan,
    )
    search.fit(x_train, y_train)
    best = search.best_estimator_
    oof_raw = cross_val_predict(clone(best), x_train, y_train, cv=cv, method="predict_proba", n_jobs=n_jobs)[:, 1]
    test_raw = score_estimator(best, x_test)

    pred_parts: List[pd.DataFrame] = []
    fold_metric_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    status_rows = [
        {
            "fold": fold,
            "harmonization_method": harmonization_method,
            "model_name": "logreg_l2_original",
            "feature_set": "z_plus_age_sex",
            "status": "fit_ok",
            "best_params": json.dumps(search.best_params_, sort_keys=True),
            "best_inner_auc": float(search.best_score_),
            "inner_cv_context": strat_context,
            "minimum_inner_stratum_count": min_cell,
        }
    ]

    for calib_method in ["raw", "oof_logitz", "oof_ecdf"]:
        cal = calibrate_scores(calib_method, y_train, oof_raw, test_raw)
        thresholds = select_thresholds(y_train, cal.threshold_scores)
        for sel in thresholds:
            thr = float(sel["threshold"])
            y_pred = (cal.test_scores >= thr).astype(int)
            metric_row: Dict[str, Any] = {
                "fold": fold,
                "harmonization_method": harmonization_method,
                "model_name": "logreg_l2_original",
                "feature_set": "z_plus_age_sex",
                "calib_method": calib_method,
                "threshold_strategy": sel["threshold_strategy"],
                "threshold": thr,
                "threshold_selection_context": (
                    "true_inner_cv_oof" if sel["threshold_strategy"] != "fixed_0p5" else "fixed_no_selection"
                ),
                "score_calibration_context": cal.note,
                "inner_cv_context": strat_context,
                "minimum_inner_stratum_count": min_cell,
                "best_inner_auc": float(search.best_score_),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            }
            metric_row.update(binary_metrics(y_test, cal.test_scores, y_pred))
            fold_metric_rows.append(metric_row)
            threshold_rows.append(
                {
                    "fold": fold,
                    "harmonization_method": harmonization_method,
                    "calib_method": calib_method,
                    "threshold_strategy": sel["threshold_strategy"],
                    "threshold": thr,
                    "inner_cv_context": strat_context,
                    "minimum_inner_stratum_count": min_cell,
                    "best_inner_auc": float(search.best_score_),
                }
            )

            pred = test_df[
                ["SubjectID", "tensor_idx", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "fold", "y"]
            ].copy()
            pred = pred.rename(columns={"y": "y_true"})
            pred["harmonization_method"] = harmonization_method
            pred["model_name"] = "logreg_l2_original"
            pred["feature_set"] = "z_plus_age_sex"
            pred["calib_method"] = calib_method
            pred["threshold_strategy"] = sel["threshold_strategy"]
            pred["threshold"] = thr
            pred["y_score_raw"] = test_raw
            pred["y_score"] = cal.test_scores
            pred["y_pred"] = y_pred
            pred_parts.append(pred)

    return {
        "predictions": pd.concat(pred_parts, ignore_index=True),
        "foldwise_metrics": pd.DataFrame(fold_metric_rows),
        "thresholds": pd.DataFrame(threshold_rows),
        "model_status": pd.DataFrame(status_rows),
    }


def pooled_metrics_from_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["harmonization_method", "model_name", "feature_set", "calib_method", "threshold_strategy"]
    for keys, sub in pred.groupby(group_cols, dropna=False):
        method, model, feature_set, calib, strategy = keys
        row: Dict[str, Any] = {
            "harmonization_method": method,
            "model_name": model,
            "feature_set": feature_set,
            "calib_method": calib,
            "threshold_strategy": strategy,
            "threshold": "fold_specific" if strategy != "fixed_0p5" else 0.5,
        }
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows)


def manufacturer_fpr(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["harmonization_method", "calib_method", "threshold_strategy", "Manufacturer"]
    for keys, sub in pred.groupby(group_cols, dropna=False):
        method, calib, strategy, mfr = keys
        cn = sub[sub["y_true"] == 0]
        ad = sub[sub["y_true"] == 1]
        rows.append(
            {
                "harmonization_method": method,
                "calib_method": calib,
                "threshold_strategy": strategy,
                "Manufacturer": mfr,
                "n_cn": int(len(cn)),
                "fp_cn": int((cn["y_pred"] == 1).sum()),
                "fpr_cn": safe_div((cn["y_pred"] == 1).sum(), len(cn)),
                "specificity_cn": safe_div((cn["y_pred"] == 0).sum(), len(cn)),
                "n_ad": int(len(ad)),
                "fn_ad": int((ad["y_pred"] == 0).sum()),
                "fnr_ad": safe_div((ad["y_pred"] == 0).sum(), len(ad)),
                "sensitivity_ad": safe_div((ad["y_pred"] == 1).sum(), len(ad)),
            }
        )
    return pd.DataFrame(rows)


def score_distribution_by_manufacturer(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["harmonization_method", "calib_method", "Manufacturer", "ResearchGroup_Mapped", "y_true"]
    for keys, sub in pred.groupby(group_cols, dropna=False):
        method, calib, mfr, dx, y_true = keys
        scores = sub["y_score"].to_numpy(dtype=float)
        rows.append(
            {
                "harmonization_method": method,
                "calib_method": calib,
                "Manufacturer": mfr,
                "diagnosis": dx,
                "y_true": int(y_true),
                "n": int(len(scores)),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=0)),
                "min": float(np.min(scores)),
                "p10": float(np.percentile(scores, 10)),
                "p25": float(np.percentile(scores, 25)),
                "median": float(np.median(scores)),
                "p75": float(np.percentile(scores, 75)),
                "p90": float(np.percentile(scores, 90)),
                "max": float(np.max(scores)),
            }
        )
    return pd.DataFrame(rows)


def manufacturer_leakage_for_method(
    fold_data: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]],
    harmonization_method: str,
    inner_folds: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold, (train, test) in fold_data.items():
        mu_cols = mu_columns(train)
        y_mfr_train = train["Manufacturer"].fillna("UNKNOWN").astype(str).to_numpy()
        y_mfr_test = test["Manufacturer"].fillna("UNKNOWN").astype(str).to_numpy()
        if len(set(y_mfr_train)) < 2 or len(set(y_mfr_test)) < 2:
            rows.append(
                {
                    "fold": fold,
                    "harmonization_method": harmonization_method,
                    "latent_manufacturer_balanced_accuracy": float("nan"),
                    "status": "skipped_single_class",
                }
            )
            continue
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=SEED + fold + 100,
                    ),
                ),
            ]
        )
        clf.fit(train[mu_cols].to_numpy(dtype=float), y_mfr_train)
        pred = clf.predict(test[mu_cols].to_numpy(dtype=float))
        rows.append(
            {
                "fold": fold,
                "harmonization_method": harmonization_method,
                "latent_manufacturer_balanced_accuracy": float(balanced_accuracy_score(y_mfr_test, pred)),
                "status": "fit_ok",
            }
        )
    return pd.DataFrame(rows)


def reference_predictions() -> pd.DataFrame:
    path = REFERENCE_CALIB_DIR / "calib_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    pred = pd.read_csv(path)
    sub = pred[
        pred["model_name"].astype(str).eq("logreg_l2_original")
        & pred["feature_set"].astype(str).eq("z_plus_age_sex")
        & pred["calib_method"].astype(str).isin(["oof_logitz", "oof_ecdf"])
    ].copy()
    if sub.empty:
        return sub
    sub["harmonization_method"] = "promoted_reference_no_harmonization"
    if "y_score_raw" not in sub.columns:
        sub["y_score_raw"] = sub["y_score"]
    return sub[
        [
            "SubjectID",
            "tensor_idx",
            "ResearchGroup_Mapped",
            "Manufacturer",
            "Age",
            "Sex",
            "fold",
            "harmonization_method",
            "model_name",
            "feature_set",
            "calib_method",
            "threshold_strategy",
            "threshold",
            "y_true",
            "y_score_raw",
            "y_score",
            "y_pred",
        ]
    ].copy()


def evaluate_method_set(cache_dir: Path, inner_folds: int, n_jobs: int) -> Dict[str, pd.DataFrame]:
    pred_parts: List[pd.DataFrame] = []
    fold_metric_parts: List[pd.DataFrame] = []
    threshold_parts: List[pd.DataFrame] = []
    status_parts: List[pd.DataFrame] = []
    method_status_rows: List[Dict[str, Any]] = []
    leakage_parts: List[pd.DataFrame] = []
    fold_data_by_method: Dict[str, Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]] = {}

    for method in ["baseline_no_harmonization", "residualize_mfr_preserve_age_sex", "combat_mfr_age_sex"]:
        fold_data_by_method[method] = {}

    for fold in FOLDS:
        print(f"Fold {fold}: loading latent cache", flush=True)
        train, test = load_fold(cache_dir, fold)
        mu_cols = mu_columns(train)

        # Baseline.
        fold_data_by_method["baseline_no_harmonization"][fold] = (train.copy(), test.copy())
        res = fit_score_fold(train.copy(), test.copy(), fold, "baseline_no_harmonization", inner_folds, n_jobs)
        pred_parts.append(res["predictions"])
        fold_metric_parts.append(res["foldwise_metrics"])
        threshold_parts.append(res["thresholds"])
        status_parts.append(res["model_status"])

        # Residualization.
        train_resid, test_resid, resid_meta = residualize_manufacturer_preserve_age_sex(train, test, mu_cols)
        method_status_rows.append({"fold": fold, "method": "residualize_mfr_preserve_age_sex", "status": "fit_ok", **resid_meta})
        fold_data_by_method["residualize_mfr_preserve_age_sex"][fold] = (train_resid.copy(), test_resid.copy())
        res = fit_score_fold(train_resid, test_resid, fold, "residualize_mfr_preserve_age_sex", inner_folds, n_jobs)
        pred_parts.append(res["predictions"])
        fold_metric_parts.append(res["foldwise_metrics"])
        threshold_parts.append(res["thresholds"])
        status_parts.append(res["model_status"])

        # ComBat if a safe held-out covariate-preserving transform exists.
        train_combat, test_combat, combat_meta = combat_train_transform_if_safe(train, test, mu_cols)
        method_status_rows.append({"fold": fold, **combat_meta})
        if train_combat is not None and test_combat is not None:
            fold_data_by_method["combat_mfr_age_sex"][fold] = (train_combat.copy(), test_combat.copy())
            res = fit_score_fold(train_combat, test_combat, fold, "combat_mfr_age_sex", inner_folds, n_jobs)
            pred_parts.append(res["predictions"])
            fold_metric_parts.append(res["foldwise_metrics"])
            threshold_parts.append(res["thresholds"])
            status_parts.append(res["model_status"])

    for method, data in fold_data_by_method.items():
        if data:
            leakage_parts.append(manufacturer_leakage_for_method(data, method, inner_folds))

    pred = pd.concat(pred_parts, ignore_index=True, sort=False)
    foldwise = pd.concat(fold_metric_parts, ignore_index=True, sort=False)
    thresholds = pd.concat(threshold_parts, ignore_index=True, sort=False)
    status = pd.concat(status_parts, ignore_index=True, sort=False)
    for fold in FOLDS:
        method_status_rows.append(
            {
                "fold": fold,
                "method": "covbat_pca_covbat",
                "status": "skipped",
                "reason": "no validated train-fold-only CovBat/PCA-CovBat implementation is available in this environment",
            }
        )
    method_status = pd.DataFrame(method_status_rows)
    leakage = pd.concat(leakage_parts, ignore_index=True, sort=False) if leakage_parts else pd.DataFrame()
    return {
        "predictions": pred,
        "foldwise_metrics": foldwise,
        "thresholds_by_fold": thresholds,
        "model_status": status,
        "method_status": method_status,
        "scanner_leakage": leakage,
    }


def add_gate_columns(pooled: pd.DataFrame, leakage: pd.DataFrame, mfr: pd.DataFrame) -> pd.DataFrame:
    out = pooled.copy()
    primary_leak = (
        leakage.groupby("harmonization_method", dropna=False)["latent_manufacturer_balanced_accuracy"]
        .mean()
        .rename("mean_latent_manufacturer_ba")
        .reset_index()
    )
    out = out.merge(primary_leak, on="harmonization_method", how="left")

    primary_fpr = mfr[
        (mfr["Manufacturer"].astype(str).str.lower() == "philips")
        & (mfr["threshold_strategy"].astype(str) == PRIMARY_THRESHOLD)
    ][["harmonization_method", "calib_method", "fpr_cn", "fp_cn", "n_cn"]].rename(
        columns={"fpr_cn": "philips_cn_fpr", "fp_cn": "philips_cn_fp", "n_cn": "philips_cn_n"}
    )
    out = out.merge(primary_fpr, on=["harmonization_method", "calib_method"], how="left")

    ref_leak = out[
        (out["harmonization_method"] == "baseline_no_harmonization")
        & (out["calib_method"] == "oof_ecdf")
        & (out["threshold_strategy"] == PRIMARY_THRESHOLD)
    ]["mean_latent_manufacturer_ba"]
    ref_leak_val = float(ref_leak.iloc[0]) if len(ref_leak) else float("nan")

    out["auc_gate_pass"] = out["auc"] >= (PROMOTED_AUC - 0.005)
    out["pr_auc_gate_pass"] = out["pr_auc"] >= (PROMOTED_PR_AUC - 0.005)
    out["philips_fpr_gate_pass"] = out["philips_cn_fpr"] < PROMOTED_PHILIPS_CN_FPR
    out["scanner_leakage_gate_pass"] = out["mean_latent_manufacturer_ba"] < ref_leak_val
    out["all_sensitivity_gates_pass"] = (
        out["auc_gate_pass"] & out["pr_auc_gate_pass"] & out["philips_fpr_gate_pass"] & out["scanner_leakage_gate_pass"]
    )
    out["reference_auc"] = PROMOTED_AUC
    out["reference_pr_auc"] = PROMOTED_PR_AUC
    out["reference_philips_cn_fpr"] = PROMOTED_PHILIPS_CN_FPR
    out["reference_mean_latent_manufacturer_ba_from_baseline_recompute"] = ref_leak_val
    return out


def build_primary_decision(pooled: pd.DataFrame, method_status: pd.DataFrame, output_dir: Path) -> None:
    primary = pooled[pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)].copy()
    primary = primary.sort_values(["all_sensitivity_gates_pass", "auc", "pr_auc"], ascending=[False, False, False])
    lines: List[str] = [
        "# Stage B Manufacturer Harmonization Decision",
        "",
        "This audit is classifier-only and uses saved train/dev and outer-test latent caches. No VAE was trained, no tensor or metadata was modified, and all harmonization parameters were fit inside the corresponding outer train/dev fold.",
        "",
        "## Promotion/Sensitivity Gate",
        "",
        f"- AUC must be at least `{PROMOTED_AUC:.6f}` or lose less than `0.005`.",
        f"- PR-AUC must be at least `{PROMOTED_PR_AUC:.6f}` or lose less than `0.005`.",
        f"- Philips CN FPR must be below `{PROMOTED_PHILIPS_CN_FPR:.3f}`.",
        "- Latent Manufacturer leakage must be lower than the no-harmonization recomputed reference.",
        "",
        "## Primary Rows",
        "",
    ]
    keep_cols = [
        "harmonization_method",
        "calib_method",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "philips_cn_fpr",
        "mean_latent_manufacturer_ba",
        "all_sensitivity_gates_pass",
    ]
    try:
        lines.append(primary[keep_cols].to_markdown(index=False))
    except Exception:
        lines.append(primary[keep_cols].to_string(index=False))

    gate_pass = primary[primary["all_sensitivity_gates_pass"].fillna(False)]
    if gate_pass.empty:
        decision = "do_not_promote; treat as sensitivity audit only"
    else:
        decision = "sensitivity_gate_passed; eligible for reviewer-facing sensitivity only, not primary promotion without external validation"
    lines.extend(
        [
            "",
            "## Decision",
            "",
            decision,
            "",
            "ComBat/CovBat note: ComBat was attempted only if the installed package exposed a held-out transform that preserves Age/Sex covariates. The local `neuroCombatFromTraining` API lacks held-out covariate arguments, so ComBat was skipped rather than using a leakage-prone train+test fit or a non-covariate-preserving transform.",
        ]
    )
    if not method_status.empty:
        lines.extend(["", "## Method Status", ""])
        try:
            lines.append(method_status[["fold", "method", "status", "reason"]].to_markdown(index=False))
        except Exception:
            lines.append(method_status.to_string(index=False))
    (output_dir / "primary_gate_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-cache", type=Path, default=DEFAULT_LATENT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    command_log: Dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "start_time": now_iso(),
        "latent_cache": str(args.latent_cache),
        "output_dir": str(output_dir),
        "inner_folds": args.inner_folds,
        "n_jobs": args.n_jobs,
        "guardrails": [
            "no VAE training",
            "no tensor modification",
            "no metadata modification",
            "no outer-test labels for harmonization, thresholding, or calibration",
            "no future conversion labels",
        ],
        "commands": [],
    }

    required = [args.latent_cache / f"fold_{fold}_{split}_latent_mu.csv" for fold in FOLDS for split in ["trainDev", "test"]]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing latent cache files:\n" + "\n".join(missing))

    if args.dry_run:
        report = [
            "# Dry Run",
            "",
            f"Latent cache: `{args.latent_cache}`",
            f"Output dir: `{output_dir}`",
            f"Found all {len(required)} required latent cache files.",
            "Planned methods: baseline_no_harmonization, residualize_mfr_preserve_age_sex, ComBat if safe held-out covariate transform is available.",
        ]
        (output_dir / "dry_run_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        command_log["dry_run"] = True
        command_log["end_time"] = now_iso()
        (output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
        return 0

    py_compile = run_cmd([sys.executable, "-m", "py_compile", str(Path(__file__).resolve())])
    command_log["commands"].append(py_compile)
    if py_compile["returncode"] != 0:
        raise RuntimeError("py_compile failed")

    result = evaluate_method_set(args.latent_cache, args.inner_folds, args.n_jobs)
    pred = result["predictions"]
    ref_pred = reference_predictions()
    if not ref_pred.empty:
        pred_with_ref = pd.concat([pred, ref_pred], ignore_index=True, sort=False)
    else:
        pred_with_ref = pred.copy()

    foldwise = result["foldwise_metrics"]
    pooled = pooled_metrics_from_predictions(pred)
    leakage = result["scanner_leakage"]
    mfr = manufacturer_fpr(pred)
    pooled_gate = add_gate_columns(pooled, leakage, mfr)
    score_dist = score_distribution_by_manufacturer(pred_with_ref)

    save_table(pred_with_ref, output_dir / "harmonized_stageb_predictions.csv")
    save_table(foldwise, output_dir / "foldwise_metrics.csv", output_dir / "foldwise_metrics.md")
    save_table(pooled_gate, output_dir / "pooled_metrics.csv", output_dir / "pooled_metrics.md")
    save_table(result["thresholds_by_fold"], output_dir / "thresholds_by_fold.csv", output_dir / "thresholds_by_fold.md")
    save_table(result["model_status"], output_dir / "selected_hyperparameters.csv", output_dir / "selected_hyperparameters.md")
    save_table(result["method_status"], output_dir / "method_status.csv", output_dir / "method_status.md")
    save_table(leakage, output_dir / "scanner_manufacturer_leakage.csv", output_dir / "scanner_manufacturer_leakage.md")
    save_table(mfr, output_dir / "manufacturer_fpr.csv", output_dir / "manufacturer_fpr.md")
    save_table(score_dist, output_dir / "score_distribution_by_manufacturer.csv", output_dir / "score_distribution_by_manufacturer.md")

    # Reference promoted rows summarized directly from fixed artifacts.
    primary_comparison_parts: List[pd.DataFrame] = [
        pooled_gate[pooled_gate["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)].copy()
    ]
    if not ref_pred.empty:
        ref_pooled = pooled_metrics_from_predictions(ref_pred)
        ref_mfr = manufacturer_fpr(ref_pred)
        save_table(ref_pooled, output_dir / "promoted_reference_pooled_metrics.csv", output_dir / "promoted_reference_pooled_metrics.md")
        save_table(ref_mfr, output_dir / "promoted_reference_manufacturer_fpr.csv", output_dir / "promoted_reference_manufacturer_fpr.md")
        ref_primary = ref_pooled[ref_pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)].copy()
        ref_philips = ref_mfr[
            (ref_mfr["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD))
            & (ref_mfr["Manufacturer"].astype(str).str.lower().eq("philips"))
        ][["harmonization_method", "calib_method", "fpr_cn", "fp_cn", "n_cn"]].rename(
            columns={"fpr_cn": "philips_cn_fpr", "fp_cn": "philips_cn_fp", "n_cn": "philips_cn_n"}
        )
        ref_primary = ref_primary.merge(ref_philips, on=["harmonization_method", "calib_method"], how="left")
        ref_primary["mean_latent_manufacturer_ba"] = float("nan")
        ref_primary["all_sensitivity_gates_pass"] = False
        primary_comparison_parts.append(ref_primary)
    primary_comparison = pd.concat(primary_comparison_parts, ignore_index=True, sort=False)
    display_cols = [
        "harmonization_method",
        "calib_method",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "philips_cn_fpr",
        "philips_cn_fp",
        "mean_latent_manufacturer_ba",
        "all_sensitivity_gates_pass",
    ]
    save_table(
        primary_comparison[display_cols],
        output_dir / "primary_comparison_with_reference.csv",
        output_dir / "primary_comparison_with_reference.md",
    )

    # Fold 4/5 focused table.
    focus = foldwise[foldwise["fold"].isin([4, 5])].copy()
    save_table(focus, output_dir / "fold4_fold5_focus_metrics.csv", output_dir / "fold4_fold5_focus_metrics.md")

    build_primary_decision(pooled_gate, result["method_status"], output_dir)

    readme = [
        "# Promoted beta3p75 Stage B Latent Harmonization Audit",
        "",
        "Classifier-only read-only audit using saved latent mu caches from the promoted beta3p75 model.",
        "",
        "## Methods",
        "",
        "- Baseline: no latent harmonization, L2 logistic regression on latent mu + Age + Sex.",
        "- Residualization: each latent dimension was adjusted by subtracting only train/dev-estimated Manufacturer dummy contributions from a linear model containing intercept, Age, Sex, and Manufacturer.",
        "- ComBat: skipped unless a safe held-out transform with Age/Sex covariate support is available. The installed `neuroCombatFromTraining` lacks covariate support for held-out subjects.",
        "- Thresholds and OOF score mappings are fit inside the train/dev fold only.",
        "",
        "## Primary Gate",
        "",
        "See `primary_gate_decision.md`.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    command_log["dry_run"] = False
    command_log["end_time"] = now_iso()
    (output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(f"Wrote audit to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
