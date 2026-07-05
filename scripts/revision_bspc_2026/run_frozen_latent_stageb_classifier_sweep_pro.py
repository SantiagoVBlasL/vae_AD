#!/usr/bin/env python3
"""Frozen-latent Stage B classifier sweep for final ADNI v5.1b VAE runs.

This script never retrains or modifies the VAE. It reads existing fold-specific
latent mu caches from completed runs and writes a new classifier-only audit.
Non-0.5 threshold selection uses inner-CV out-of-fold scores from train/dev
data only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/frozen_latent_stageb_classifier_sweep_pro"

RUNS = {
    "final_ch1_0_2_v5_1b_horizon4480": {
        "label": "final [1,0,2] v5.1b horizon4480",
        "run_dir": PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5",
        "reference_role": "main_model",
    },
    "simplified_ch1_offdiag_channelmean": {
        "label": "simplified [1] offdiag_channelmean",
        "run_dir": PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1b_ch1_only_offdiag_channelmean_horizon4480_cycles56_full_5x5",
        "reference_role": "secondary_simplified_auc_model",
    },
    "ch1_2_offdiag_channelmean": {
        "label": "[1,2] offdiag_channelmean",
        "run_dir": PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1b_ch1_2_offdiag_channelmean_horizon4480_cycles56_full_5x5",
        "reference_role": "rejected_pair_model",
    },
}

MODELS = [
    "logreg_l2",
    "logreg_elasticnet",
    "linear_svm",
    "rbf_svm",
    "lightgbm_very_regularized",
]
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
TARGET_SENSITIVITY = 0.70
REFERENCE_MAIN = {
    "auc": 0.782951,
    "pr_auc": 0.559873,
    "balanced_accuracy": 0.735417,
    "sensitivity": 0.770833,
    "specificity": 0.700000,
    "f1": 0.569231,
}


@dataclass
class ModelSpec:
    estimator: Pipeline
    params: Dict[str, Any]
    search_type: str
    status: str = "available"
    post_selection_calibration: str = "none"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-key", default="all", choices=["all", *RUNS.keys()])
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-iter-lightgbm", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and planned jobs only.")
    parser.add_argument("--confirm-run", action="store_true", help="Required to train classifier-only readouts.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory tables.")
    return parser.parse_args()


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(mu_cols: List[str]) -> ColumnTransformer:
    latent = Pipeline([("scaler", StandardScaler())])
    age = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    sex = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_ohe())])
    return ColumnTransformer(
        [
            ("latent", latent, mu_cols),
            ("age", age, ["Age"]),
            ("sex", sex, ["Sex"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def calibrated_cv(estimator: Any, cv: int = 3) -> CalibratedClassifierCV:
    try:
        return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method="sigmoid", cv=cv)


def classifier_specs(seed: int, y_train: np.ndarray, n_iter_lightgbm: int) -> Dict[str, ModelSpec]:
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = float(n_neg / n_pos) if n_pos else 1.0
    specs: Dict[str, ModelSpec] = {
        "logreg_l2": ModelSpec(
            estimator=Pipeline(
                [
                    ("pre", "passthrough"),
                    (
                        "model",
                        LogisticRegression(
                            penalty="l2",
                            solver="lbfgs",
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            params={"model__C": [0.001, 0.01, 0.1, 1.0]},
            search_type="grid",
        ),
        "logreg_elasticnet": ModelSpec(
            estimator=Pipeline(
                [
                    ("pre", "passthrough"),
                    (
                        "model",
                        LogisticRegression(
                            penalty="elasticnet",
                            solver="saga",
                            class_weight="balanced",
                            max_iter=8000,
                            random_state=seed,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            params={
                "model__C": [0.001, 0.003, 0.01, 0.03, 0.1],
                "model__l1_ratio": [0.05, 0.1, 0.25, 0.5],
            },
            search_type="grid",
        ),
        "linear_svm": ModelSpec(
            estimator=Pipeline(
                [
                    ("pre", "passthrough"),
                    (
                        "model",
                        SVC(
                            kernel="linear",
                            probability=True,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            params={"model__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
            search_type="grid",
            post_selection_calibration="svc_probability_platt",
        ),
        "rbf_svm": ModelSpec(
            estimator=Pipeline(
                [
                    ("pre", "passthrough"),
                    (
                        "model",
                        SVC(
                            kernel="rbf",
                            probability=True,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            params={"model__C": [0.01, 0.1, 1.0, 10.0], "model__gamma": ["scale", 0.0003, 0.001, 0.003]},
            search_type="grid",
            post_selection_calibration="svc_probability_platt",
        ),
    }
    try:
        from lightgbm import LGBMClassifier
        from scipy.stats import loguniform, randint, uniform

        specs["lightgbm_very_regularized"] = ModelSpec(
            estimator=Pipeline(
                [
                    ("pre", "passthrough"),
                    (
                        "model",
                        LGBMClassifier(
                            objective="binary",
                            random_state=seed,
                            n_jobs=1,
                            verbosity=-1,
                            force_col_wise=True,
                        ),
                    ),
                ]
            ),
            params={
                "model__num_leaves": randint(2, 9),
                "model__max_depth": randint(1, 4),
                "model__min_child_samples": randint(20, 81),
                "model__learning_rate": loguniform(0.005, 0.05),
                "model__n_estimators": randint(50, 401),
                "model__feature_fraction": uniform(0.3, 0.5),
                "model__bagging_fraction": uniform(0.5, 0.4),
                "model__bagging_freq": randint(1, 6),
                "model__lambda_l1": uniform(0.0, 10.0),
                "model__lambda_l2": loguniform(1.0, 100.0),
                "model__min_gain_to_split": uniform(0.0, 1.0),
                "model__scale_pos_weight": [0.75 * scale_pos_weight, scale_pos_weight, 1.25 * scale_pos_weight],
            },
            search_type=f"randomized_{int(n_iter_lightgbm)}",
            post_selection_calibration="sigmoid_cv3_after_hp_selection",
        )
    except Exception as exc:
        specs["lightgbm_very_regularized"] = ModelSpec(
            estimator=Pipeline([("pre", "passthrough"), ("model", LogisticRegression())]),
            params={},
            search_type="unavailable",
            status=f"unavailable: {exc}",
        )
    return specs


def run_search(spec: ModelSpec, pipe: Pipeline, inner_cv: List[Tuple[np.ndarray, np.ndarray]], x: pd.DataFrame, y: np.ndarray, seed: int, n_jobs: int, n_iter_lightgbm: int) -> Any:
    if spec.search_type.startswith("randomized"):
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=spec.params,
            n_iter=n_iter_lightgbm,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
            random_state=seed,
            error_score=np.nan,
        )
    else:
        search = GridSearchCV(
            estimator=pipe,
            param_grid=spec.params,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
            error_score=np.nan,
        )
    search.fit(x, y)
    return search


def score_1d(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(x)[:, 1], dtype=float)
    if hasattr(estimator, "decision_function"):
        s = np.asarray(estimator.decision_function(x), dtype=float)
        lo, hi = np.nanmin(s), np.nanmax(s)
        return (s - lo) / (hi - lo) if hi > lo else np.full_like(s, 0.5)
    return np.asarray(estimator.predict(x), dtype=float)


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=int)
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
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)) if len(np.unique(y_true_arr)) == 2 else np.nan,
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "predicted_ad_rate": float(y_pred_arr.mean()) if len(y_pred_arr) else np.nan,
        "auc": float(roc_auc_score(y_true_arr, y_score_arr)) if len(np.unique(y_true_arr)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y_true_arr, y_score_arr)) if len(np.unique(y_true_arr)) == 2 else np.nan,
        "brier": float(brier_score_loss(y_true_arr, np.clip(y_score_arr, 0.0, 1.0))) if len(y_true_arr) else np.nan,
    }
    return out


def threshold_candidates(scores: Sequence[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.asarray([0.5])
    qs = np.unique(np.quantile(arr, np.linspace(0, 1, min(301, max(11, arr.size)))))
    vals = np.unique(np.concatenate([qs, [0.5]]))
    return np.clip(vals, 0.0, 1.0)


def threshold_table(y_true: Sequence[int], y_score: Sequence[float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    for thr in threshold_candidates(s):
        pred = (s >= thr).astype(int)
        row = {"threshold": float(thr)}
        row.update(binary_metrics(y, s, pred))
        row["youden_j"] = row["sensitivity"] + row["specificity"] - 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def select_thresholds(y_true: Sequence[int], y_score: Sequence[float]) -> List[Dict[str, Any]]:
    tbl = threshold_table(y_true, y_score)
    out = [
        {
            "threshold_strategy": "fixed_0p5",
            "threshold": 0.5,
            "selection_metric": "fixed_no_selection",
            "inner_oof_sensitivity": np.nan,
            "inner_oof_specificity": np.nan,
            "inner_oof_balanced_accuracy": np.nan,
        }
    ]
    youden = tbl.sort_values(["youden_j", "balanced_accuracy", "specificity"], ascending=False).iloc[0]
    ba = tbl.sort_values(["balanced_accuracy", "youden_j", "specificity"], ascending=False).iloc[0]
    eligible = tbl[tbl["sensitivity"] >= TARGET_SENSITIVITY].copy()
    if eligible.empty:
        target = tbl.sort_values(["sensitivity", "specificity", "balanced_accuracy"], ascending=False).iloc[0]
    else:
        target = eligible.sort_values(["specificity", "balanced_accuracy", "sensitivity"], ascending=False).iloc[0]
    for strategy, row, metric in [
        ("inner_oof_youden_j", youden, "youden_j"),
        ("inner_oof_balanced_accuracy", ba, "balanced_accuracy"),
        ("inner_oof_target_sens_ge_0p70_max_spec", target, "selected_inner_oof"),
    ]:
        out.append(
            {
                "threshold_strategy": strategy,
                "threshold": float(row["threshold"]),
                "selection_metric": metric,
                "inner_oof_sensitivity": float(row["sensitivity"]),
                "inner_oof_specificity": float(row["specificity"]),
                "inner_oof_balanced_accuracy": float(row["balanced_accuracy"]),
            }
        )
    return out


def inner_stratification_key(df: pd.DataFrame, n_splits: int) -> Tuple[pd.Series, str, int]:
    key_df = df[["ResearchGroup_Mapped", "Manufacturer"]].copy()
    for col in key_df.columns:
        key_df[col] = key_df[col].fillna(f"{col}_UNKNOWN").astype(str)
    key = key_df.apply(lambda r: "_".join(r.values.astype(str)), axis=1)
    min_count = int(key.value_counts().min())
    if min_count < n_splits:
        y = df["y"].astype(int)
        return y, "label_only_fallback", int(pd.Series(y).value_counts().min())
    return key, "ResearchGroup_Mapped+Manufacturer", min_count


def load_latent_pair(latent_cache_dir: Path, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = latent_cache_dir / f"fold_{fold}_trainDev_latent_mu.csv"
    test_path = latent_cache_dir / f"fold_{fold}_test_latent_mu.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing latent cache for fold {fold}: {train_path}, {test_path}")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def site_code(subject_id: Any) -> str:
    s = str(subject_id)
    match = re.match(r"^(\d{3})[_-]", s)
    if match:
        return match.group(1)
    match = re.match(r"^(\d{3})", s)
    return match.group(1) if match else "UNKNOWN"


def ece_score(y_true: Sequence[int], y_score: Sequence[float], n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    if len(y) == 0:
        return np.nan
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (s >= lo) & (s < hi if hi < 1.0 else s <= hi)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(s[mask].mean()))
    return ece


def subgroup_metrics(pred: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, sub in pred.groupby(["run_key", "run_label", "model_name", "threshold_strategy", group_col], dropna=False):
        run_key, run_label, model_name, threshold_strategy, group = keys
        row = {
            "run_key": run_key,
            "run_label": run_label,
            "model_name": model_name,
            "threshold_strategy": threshold_strategy,
            group_col: group,
        }
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_metrics(pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    conf: List[Dict[str, Any]] = []
    for keys, sub in pred.groupby(["run_key", "run_label", "model_name", "threshold_strategy"], dropna=False):
        run_key, run_label, model_name, threshold_strategy = keys
        row = {
            "run_key": run_key,
            "run_label": run_label,
            "model_name": model_name,
            "threshold_strategy": threshold_strategy,
            "threshold": "fold_specific" if threshold_strategy != "fixed_0p5" else 0.5,
        }
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
        conf.append({k: row[k] for k in ["run_key", "run_label", "model_name", "threshold_strategy", "threshold", "n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "sensitivity", "specificity", "balanced_accuracy", "f1"]})
    return pd.DataFrame(rows), pd.DataFrame(conf)


def run_sweep(args: argparse.Namespace, selected_runs: Dict[str, Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    fold_metric_rows: List[Dict[str, Any]] = []
    pred_rows: List[pd.DataFrame] = []
    threshold_rows: List[Dict[str, Any]] = []
    hp_rows: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []
    for run_key, run_info in selected_runs.items():
        latent_cache = Path(run_info["run_dir"]) / "classifier_only_readout" / "latent_cache"
        for fold in range(1, int(args.outer_folds) + 1):
            train_df, test_df = load_latent_pair(latent_cache, fold)
            train_df = train_df.copy()
            test_df = test_df.copy()
            train_df["SiteCode"] = train_df["SubjectID"].map(site_code)
            test_df["SiteCode"] = test_df["SubjectID"].map(site_code)
            mu_cols = [c for c in train_df.columns if c.startswith("mu_")]
            feature_cols = mu_cols + ["Age", "Sex"]
            x_train = train_df[feature_cols].copy()
            x_test = test_df[feature_cols].copy()
            y_train = train_df["y"].astype(int).to_numpy()
            y_test = test_df["y"].astype(int).to_numpy()
            inner_key, inner_context, min_cell = inner_stratification_key(train_df, int(args.inner_folds))
            inner_cv = list(
                StratifiedKFold(
                    n_splits=int(args.inner_folds),
                    shuffle=True,
                    random_state=42 + fold + 30,
                ).split(np.zeros(len(train_df)), inner_key)
            )
            specs = classifier_specs(seed=42 + fold, y_train=y_train, n_iter_lightgbm=int(args.n_iter_lightgbm))
            pre = make_preprocessor(mu_cols)
            for model_name in args.models:
                spec = specs[model_name]
                if spec.status != "available":
                    status_rows.append(
                        {
                            "run_key": run_key,
                            "fold": fold,
                            "model_name": model_name,
                            "status": spec.status,
                        }
                    )
                    continue
                pipe = clone(spec.estimator)
                pipe.steps[0] = ("pre", pre)
                search = run_search(spec, pipe, inner_cv, x_train, y_train, seed=42 + fold, n_jobs=int(args.n_jobs), n_iter_lightgbm=int(args.n_iter_lightgbm))
                best_base = search.best_estimator_
                if spec.post_selection_calibration == "sigmoid_cv3_after_hp_selection":
                    best_for_scores = calibrated_cv(clone(best_base), cv=3)
                else:
                    best_for_scores = best_base
                oof_score = cross_val_predict(
                    clone(best_for_scores),
                    x_train,
                    y_train,
                    cv=inner_cv,
                    method="predict_proba",
                    n_jobs=int(args.n_jobs),
                )[:, 1]
                best_for_scores.fit(x_train, y_train)
                test_score = score_1d(best_for_scores, x_test)
                thresholds = select_thresholds(y_train, oof_score)
                hp_row = {
                    "run_key": run_key,
                    "run_label": run_info["label"],
                    "fold": fold,
                    "model_name": model_name,
                    "status": "fit_ok",
                    "search_type": spec.search_type,
                    "post_selection_calibration": spec.post_selection_calibration,
                    "best_params": json.dumps(search.best_params_, sort_keys=True),
                    "best_inner_auc": float(search.best_score_),
                    "inner_cv_context": inner_context,
                    "minimum_inner_stratum_count": int(min_cell),
                    "n_search_candidates": int(getattr(search, "n_iter", len(search.cv_results_.get("params", [])))),
                }
                hp_rows.append(hp_row)
                status_rows.append(hp_row)
                for sel in thresholds:
                    thr = float(sel["threshold"])
                    y_pred = (test_score >= thr).astype(int)
                    row = {
                        "run_key": run_key,
                        "run_label": run_info["label"],
                        "fold": fold,
                        "model_name": model_name,
                        "threshold_strategy": sel["threshold_strategy"],
                        "threshold": thr,
                        "threshold_selection_context": "true_inner_cv_oof" if sel["threshold_strategy"] != "fixed_0p5" else "fixed_no_selection",
                        "inner_cv_context": inner_context,
                        "minimum_inner_stratum_count": int(min_cell),
                        "best_inner_auc": float(search.best_score_),
                        "best_params": json.dumps(search.best_params_, sort_keys=True),
                        **sel,
                    }
                    row.update(binary_metrics(y_test, test_score, y_pred))
                    fold_metric_rows.append(row)
                    pred = test_df[
                        [
                            "SubjectID",
                            "tensor_idx",
                            "ResearchGroup_Mapped",
                            "Manufacturer",
                            "Age",
                            "Sex",
                            "source_batch",
                            "source_label",
                            "tensor_source",
                            "SiteCode",
                        ]
                    ].copy()
                    pred["run_key"] = run_key
                    pred["run_label"] = run_info["label"]
                    pred["fold"] = fold
                    pred["model_name"] = model_name
                    pred["threshold_strategy"] = sel["threshold_strategy"]
                    pred["threshold"] = thr
                    pred["y_true"] = y_test
                    pred["y_score"] = test_score
                    pred["y_pred"] = y_pred
                    pred_rows.append(pred)
                    trow = {
                        "run_key": run_key,
                        "run_label": run_info["label"],
                        "fold": fold,
                        "model_name": model_name,
                        "threshold_strategy": sel["threshold_strategy"],
                        "threshold": thr,
                        "threshold_selection_context": row["threshold_selection_context"],
                        "inner_cv_context": inner_context,
                        "minimum_inner_stratum_count": int(min_cell),
                        "best_inner_auc": float(search.best_score_),
                        "selection_metric": sel["selection_metric"],
                        "inner_oof_sensitivity": sel["inner_oof_sensitivity"],
                        "inner_oof_specificity": sel["inner_oof_specificity"],
                        "inner_oof_balanced_accuracy": sel["inner_oof_balanced_accuracy"],
                    }
                    threshold_rows.append(trow)
    pred_all = pd.concat(pred_rows, ignore_index=True, sort=False) if pred_rows else pd.DataFrame()
    pooled, pooled_confusion = pooled_metrics(pred_all) if not pred_all.empty else (pd.DataFrame(), pd.DataFrame())
    calibration = calibration_table(pred_all)
    return {
        "primary_results": pooled,
        "foldwise_metrics": pd.DataFrame(fold_metric_rows),
        "selected_hyperparameters": pd.DataFrame(hp_rows),
        "threshold_by_fold": pd.DataFrame(threshold_rows),
        "manufacturer_subgroup": subgroup_metrics(pred_all, "Manufacturer") if not pred_all.empty else pd.DataFrame(),
        "sitecode_subgroup": subgroup_metrics(pred_all, "SiteCode") if not pred_all.empty else pd.DataFrame(),
        "calibration_brier": calibration,
        "pooled_confusion": pooled_confusion,
        "predictions": pred_all,
        "model_status": pd.DataFrame(status_rows),
    }


def calibration_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if pred.empty:
        return pd.DataFrame()
    score_once = pred.drop_duplicates(["run_key", "model_name", "SubjectID", "fold"])
    for keys, sub in score_once.groupby(["run_key", "run_label", "model_name"], dropna=False):
        run_key, run_label, model_name = keys
        y = sub["y_true"].astype(int).to_numpy()
        score = sub["y_score"].astype(float).to_numpy()
        rows.append(
            {
                "run_key": run_key,
                "run_label": run_label,
                "model_name": model_name,
                "n": int(len(sub)),
                "brier": float(brier_score_loss(y, np.clip(score, 0.0, 1.0))),
                "ece_10bin": float(ece_score(y, score, n_bins=10)),
                "mean_score_cn": float(np.mean(score[y == 0])) if (y == 0).any() else np.nan,
                "mean_score_ad": float(np.mean(score[y == 1])) if (y == 1).any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "No rows.\n"
    sub = df.head(max_rows)
    lines = ["| " + " | ".join(sub.columns) + " |", "| " + " | ".join(["---"] * len(sub.columns)) + " |"]
    for _, row in sub.iterrows():
        vals = []
        for col in sub.columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{val:.6f}" if np.isfinite(val) else "NA")
            else:
                vals.append(str(val).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_pair(root: Path, name: str, df: pd.DataFrame, max_rows: int = 120) -> None:
    df.to_csv(root / f"{name}.csv", index=False)
    (root / f"{name}.md").write_text(markdown_table(df, max_rows=max_rows), encoding="utf-8")


def write_outputs(root: Path, results: Dict[str, pd.DataFrame]) -> None:
    write_pair(root, "primary_results", results["primary_results"].sort_values(["threshold_strategy", "auc", "pr_auc"], ascending=[True, False, False]), max_rows=200)
    write_pair(root, "foldwise_metrics", results["foldwise_metrics"], max_rows=300)
    write_pair(root, "selected_hyperparameters", results["selected_hyperparameters"], max_rows=300)
    write_pair(root, "threshold_by_fold", results["threshold_by_fold"], max_rows=300)
    write_pair(root, "manufacturer_subgroup", results["manufacturer_subgroup"], max_rows=300)
    write_pair(root, "sitecode_subgroup", results["sitecode_subgroup"], max_rows=300)
    write_pair(root, "calibration_brier", results["calibration_brier"], max_rows=120)
    results["pooled_confusion"].to_csv(root / "pooled_confusion.csv", index=False)
    results["predictions"].to_csv(root / "predictions.csv", index=False)
    results["model_status"].to_csv(root / "model_status.csv", index=False)


def write_recommendation(root: Path, primary: pd.DataFrame, manufacturer: pd.DataFrame) -> None:
    lines = [
        "# Final Recommendation",
        "",
        "This is a frozen-latent Stage B classifier-only sweep. No VAE was retrained.",
        "",
    ]
    if primary.empty:
        lines.append("No completed classifier results were available.")
    else:
        target = primary[primary["threshold_strategy"].eq(PRIMARY_THRESHOLD)].copy()
        target = target.sort_values(["auc", "pr_auc", "balanced_accuracy"], ascending=False)
        best = target.iloc[0]
        main = target[target["run_key"].eq("final_ch1_0_2_v5_1b_horizon4480")]
        lines += [
            "## Primary Operating Point",
            "",
            f"Primary threshold strategy: `{PRIMARY_THRESHOLD}`.",
            "",
            f"Best candidate by AUC: `{best['run_key']}` / `{best['model_name']}` with AUC `{best['auc']:.6f}`, PR-AUC `{best['pr_auc']:.6f}`, BA `{best['balanced_accuracy']:.6f}`, sensitivity `{best['sensitivity']:.6f}`, F1 `{best['f1']:.6f}`.",
            "",
        ]
        if not main.empty:
            main_best = main.sort_values(["auc", "pr_auc"], ascending=False).iloc[0]
            lines += [
                f"Best classifier on the current main VAE run: `{main_best['model_name']}`, AUC `{main_best['auc']:.6f}`, PR-AUC `{main_best['pr_auc']:.6f}`.",
                "",
                f"Reference final model target: AUC `{REFERENCE_MAIN['auc']:.6f}`, PR-AUC `{REFERENCE_MAIN['pr_auc']:.6f}`, BA `{REFERENCE_MAIN['balanced_accuracy']:.6f}`, sensitivity `{REFERENCE_MAIN['sensitivity']:.6f}`, F1 `{REFERENCE_MAIN['f1']:.6f}`.",
                "",
            ]
        passes = target[
            (target["auc"] > REFERENCE_MAIN["auc"])
            & (target["pr_auc"] >= REFERENCE_MAIN["pr_auc"])
            & (target["balanced_accuracy"] >= REFERENCE_MAIN["balanced_accuracy"] - 0.01)
            & (target["f1"] >= REFERENCE_MAIN["f1"] - 0.01)
            & (target["sensitivity"] >= REFERENCE_MAIN["sensitivity"] - 0.02)
        ].copy()
        if passes.empty:
            lines += [
                "## Decision",
                "",
                "No classifier-only strategy satisfies the promotion rule. Do not replace the current main model based on this sweep.",
            ]
        else:
            lines += [
                "## Decision",
                "",
                "At least one classifier-only strategy meets the numeric promotion screen. Review subgroup and calibration tables before any manuscript change:",
                "",
                markdown_table(passes[["run_key", "model_name", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]], max_rows=20),
            ]
        lines += [
            "",
            "## Top Primary Results",
            "",
            markdown_table(target[["run_key", "model_name", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "brier"]].head(15), max_rows=15),
        ]
    (root / "final_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(root: Path, args: argparse.Namespace, selected_runs: Dict[str, Dict[str, Any]]) -> None:
    lines = [
        "# Frozen-Latent Stage B Classifier Sweep Pro",
        "",
        "## Scope",
        "",
        "- VAE retraining: `False`.",
        "- Tensor/metadata/config/VAE output modification: `False`.",
        "- Features: saved latent `mu + Age + Sex`.",
        "- Outer folds: existing fold-specific latent caches.",
        "- Hyperparameter selection: inner CV on train/dev only.",
        "- Non-0.5 threshold selection: true inner-CV OOF predictions only.",
        "",
        "## Runs",
        "",
    ]
    for key, info in selected_runs.items():
        lines.append(f"- `{key}`: {info['label']} from `{info['run_dir']}`")
    lines += [
        "",
        "## Models",
        "",
        "- `logreg_l2` baseline",
        "- `logreg_elasticnet`",
        "- `linear_svm`",
        "- `rbf_svm`",
        "- `lightgbm_very_regularized`",
        "",
        f"LightGBM randomized search iterations per fold: `{args.n_iter_lightgbm}`.",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_inputs(selected_runs: Dict[str, Dict[str, Any]], outer_folds: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run_key, info in selected_runs.items():
        run_dir = Path(info["run_dir"])
        cache = run_dir / "classifier_only_readout" / "latent_cache"
        row = {
            "run_key": run_key,
            "run_label": info["label"],
            "run_dir": str(run_dir),
            "latent_cache": str(cache),
            "run_dir_exists": run_dir.exists(),
            "latent_cache_exists": cache.exists(),
        }
        for fold in range(1, outer_folds + 1):
            row[f"fold_{fold}_train_exists"] = (cache / f"fold_{fold}_trainDev_latent_mu.csv").exists()
            row[f"fold_{fold}_test_exists"] = (cache / f"fold_{fold}_test_latent_mu.csv").exists()
        rows.append(row)
    df = pd.DataFrame(rows)
    missing = df.filter(like="_exists").eq(False).any(axis=1)
    if missing.any():
        raise FileNotFoundError("Missing required latent caches:\n" + df[missing].to_string(index=False))
    return df


def main() -> int:
    args = parse_args()
    root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    if root.exists() and any(root.iterdir()) and not args.overwrite and args.confirm_run:
        raise RuntimeError(f"Output root is not empty: {root}. Use --overwrite to refresh classifier-sweep outputs.")
    root.mkdir(parents=True, exist_ok=True)
    selected_runs = RUNS if args.run_key == "all" else {args.run_key: RUNS[args.run_key]}
    input_status = validate_inputs(selected_runs, int(args.outer_folds))
    input_status.to_csv(root / "input_run_manifest.csv", index=False)
    write_readme(root, args, selected_runs)
    payload: Dict[str, Any] = {
        "updated_utc": now_utc(),
        "script": str(Path(__file__).resolve()),
        "output_root": str(root),
        "dry_run": bool(args.dry_run or not args.confirm_run),
        "confirm_run": bool(args.confirm_run),
        "run_keys": list(selected_runs),
        "models": list(args.models),
        "outer_folds": int(args.outer_folds),
        "inner_folds": int(args.inner_folds),
        "n_iter_lightgbm": int(args.n_iter_lightgbm),
        "vae_retrained": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "configs_modified": False,
        "vae_outputs_modified": False,
    }
    if args.dry_run or not args.confirm_run:
        (root / "dry_run_report.md").write_text(
            "# Dry-Run Report\n\n"
            f"Generated at `{now_utc()}`.\n\n"
            f"Runs: `{', '.join(selected_runs)}`.\n\n"
            f"Models: `{', '.join(args.models)}`.\n\n"
            "No classifier fitting was launched.\n",
            encoding="utf-8",
        )
        (root / "command_log.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(input_status.to_string(index=False))
        print("Dry-run complete. No classifier fitting launched.")
        return 0
    results = run_sweep(args, selected_runs)
    write_outputs(root, results)
    write_recommendation(root, results["primary_results"], results["manufacturer_subgroup"])
    payload["dry_run"] = False
    payload["classifier_fitting_launched"] = True
    payload["output_files"] = sorted(p.name for p in root.iterdir() if p.is_file())
    (root / "command_log.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
