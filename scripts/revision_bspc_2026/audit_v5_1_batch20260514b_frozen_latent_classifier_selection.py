#!/usr/bin/env python3
"""Frozen-latent classifier-selection audit for locked ADNI v5.1 FULL [1,0,2].

This script intentionally does not load or train the VAE. It reuses the
classifier-only latent mu cache produced from the locked FULL run and evaluates
alternative inner-CV hyperparameter selection rules for simple linear readouts.
All non-0.5 thresholds are selected from true train/dev inner-CV out-of-fold
scores only.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCKED_RUN_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
)
LATENT_CACHE_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
    / "latent_cache"
)
REFERENCE_SWEEP_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
)
OUTDIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_frozen_latent_classifier_selection_audit"
)

TARGET_SENSITIVITY = 0.70
OUTER_FOLDS = 5
INNER_FOLDS = 5
SEED = 42
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
MIN_SITE_N = 12
MIN_SITE_PER_CLASS = 3


@dataclass(frozen=True)
class Strategy:
    name: str
    model_type: str
    grid: Tuple[Dict[str, Any], ...]
    selection_rule: str
    description: str


def c_grid(values: Sequence[float]) -> Tuple[Dict[str, Any], ...]:
    return tuple({"C": float(v)} for v in values)


def strategy_specs() -> List[Strategy]:
    current_c = [0.001, 0.01, 0.1, 1.0]
    dense_c = np.round(np.logspace(-4, -1, 13), 10).tolist()
    elastic_c = np.round(np.logspace(-4, -1, 9), 10).tolist()
    svm_c = np.round(np.logspace(-4, 0, 9), 10).tolist()
    return [
        Strategy(
            "baseline_logreg_l2_current_selection",
            "logreg_l2",
            c_grid(current_c),
            "global_auc",
            "Matches the existing classifier-only logreg_l2 C grid and global inner-CV AUC selection.",
        ),
        Strategy(
            "logreg_l2_dense_C_global_auc",
            "logreg_l2",
            c_grid(dense_c),
            "global_auc",
            "Denser C grid from 1e-4 to 1e-1 selected by global inner-CV AUC.",
        ),
        Strategy(
            "logreg_l2_manufacturer_mean_auc",
            "logreg_l2",
            c_grid(dense_c),
            "manufacturer_mean_auc",
            "Dense C grid selected by mean inner-CV AUC across manufacturers.",
        ),
        Strategy(
            "logreg_l2_manufacturer_min_auc",
            "logreg_l2",
            c_grid(dense_c),
            "manufacturer_min_auc",
            "Dense C grid selected by minimum inner-CV AUC across manufacturers.",
        ),
        Strategy(
            "logreg_l2_manufacturer_auc_penalized",
            "logreg_l2",
            c_grid(dense_c),
            "global_auc_minus_0p5_manufacturer_std",
            "Dense C grid selected by global AUC minus 0.5 times manufacturer-AUC std.",
        ),
        Strategy(
            "elasticnet_manufacturer_mean_auc",
            "logreg_elasticnet",
            tuple({"C": float(c), "l1_ratio": float(l1)} for c in elastic_c for l1 in [0.05, 0.1, 0.25, 0.5, 0.75]),
            "manufacturer_mean_auc",
            "Elastic-net logistic regression selected by mean manufacturer inner-CV AUC.",
        ),
        Strategy(
            "linear_svm_calibrated_manufacturer_mean_auc",
            "linear_svm_calibrated",
            c_grid(svm_c),
            "manufacturer_mean_auc",
            "Linear SVM with sigmoid calibration selected by mean manufacturer inner-CV AUC.",
        ),
    ]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(mu_cols: Sequence[str]) -> ColumnTransformer:
    numeric_latent = Pipeline([("scaler", StandardScaler())])
    numeric_age = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_ohe())])
    return ColumnTransformer(
        [
            ("latent", numeric_latent, list(mu_cols)),
            ("age", numeric_age, ["Age"]),
            ("sex", categorical, ["Sex"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def make_calibrated_linear_svm(base: LinearSVC) -> CalibratedClassifierCV:
    try:
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)


def make_estimator(strategy: Strategy, params: Dict[str, Any], mu_cols: Sequence[str], seed: int) -> Pipeline:
    pre = make_preprocessor(mu_cols)
    if strategy.model_type == "logreg_l2":
        model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=float(params["C"]),
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )
    elif strategy.model_type == "logreg_elasticnet":
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=float(params["C"]),
            l1_ratio=float(params["l1_ratio"]),
            class_weight="balanced",
            max_iter=7000,
            random_state=seed,
            n_jobs=1,
        )
    elif strategy.model_type == "linear_svm_calibrated":
        base = LinearSVC(C=float(params["C"]), class_weight="balanced", max_iter=7000, random_state=seed)
        model = make_calibrated_linear_svm(base)
    else:
        raise ValueError(f"Unknown model_type: {strategy.model_type}")
    return Pipeline([("pre", pre), ("model", model)])


def score_1d(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(x)[:, 1], dtype=float)
    if hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(x), dtype=float).ravel()
        return 1.0 / (1.0 + np.exp(-raw))
    raise TypeError(f"Estimator has no probability-like scorer: {type(estimator)}")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def safe_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_pr_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    p = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    out: Dict[str, Any] = {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": safe_div(tp + tn, len(y)),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "balanced_accuracy": float(np.nanmean([safe_div(tp, tp + fn), safe_div(tn, tn + fp)])),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "predicted_ad_rate": float(p.mean()) if len(p) else float("nan"),
        "auc": safe_auc(y, s),
        "pr_auc": safe_pr_auc(y, s),
        "brier": float(brier_score_loss(y, np.clip(s, 0.0, 1.0))) if len(np.unique(y)) >= 1 else float("nan"),
    }
    return out


def threshold_candidates(scores: Sequence[float]) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    return np.unique(np.round(np.clip(np.concatenate(([0.0, 0.5, 1.0], s)), 0.0, 1.0), 12))


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
    selections: List[Dict[str, Any]] = []
    for strategy, metric in [("inner_oof_youden_j", "youden_j"), ("inner_oof_balanced_accuracy", "balanced_accuracy")]:
        r = tbl.sort_values([metric, "sensitivity", "specificity", "threshold"], ascending=[False, False, False, False]).iloc[0]
        selections.append(
            {
                "threshold_strategy": strategy,
                "threshold": float(r["threshold"]),
                "threshold_selection_context": "true_inner_cv_oof",
                "selection_metric": metric,
                "inner_oof_sensitivity": float(r["sensitivity"]),
                "inner_oof_specificity": float(r["specificity"]),
                "inner_oof_balanced_accuracy": float(r["balanced_accuracy"]),
            }
        )
    eligible = tbl[tbl["sensitivity"] >= TARGET_SENSITIVITY]
    if eligible.empty:
        r = tbl.sort_values(["sensitivity", "specificity", "threshold"], ascending=[False, False, False]).iloc[0]
        selection_metric = "target_not_reached_inner_oof"
    else:
        r = eligible.sort_values(["specificity", "sensitivity", "balanced_accuracy", "threshold"], ascending=[False, False, False, False]).iloc[0]
        selection_metric = "selected_inner_oof"
    selections.append(
        {
            "threshold_strategy": PRIMARY_THRESHOLD,
            "threshold": float(r["threshold"]),
            "threshold_selection_context": "true_inner_cv_oof",
            "selection_metric": selection_metric,
            "inner_oof_sensitivity": float(r["sensitivity"]),
            "inner_oof_specificity": float(r["specificity"]),
            "inner_oof_balanced_accuracy": float(r["balanced_accuracy"]),
        }
    )
    selections.append(
        {
            "threshold_strategy": "fixed_0p5",
            "threshold": 0.5,
            "threshold_selection_context": "fixed_no_selection",
            "selection_metric": "fixed_no_selection",
            "inner_oof_sensitivity": float("nan"),
            "inner_oof_specificity": float("nan"),
            "inner_oof_balanced_accuracy": float("nan"),
        }
    )
    return selections


def add_site_code(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sid = out["SubjectID"].fillna("").astype(str)
    out["SiteCode"] = sid.str.extract(r"^(\d{3})", expand=False).fillna("UNKNOWN")
    return out


def load_latents(fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = add_site_code(pd.read_csv(LATENT_CACHE_DIR / f"fold_{fold}_trainDev_latent_mu.csv"))
    test = add_site_code(pd.read_csv(LATENT_CACHE_DIR / f"fold_{fold}_test_latent_mu.csv"))
    return train, test


def inner_stratification_key(df: pd.DataFrame, n_splits: int) -> Tuple[pd.Series, str, int]:
    key_df = df[["ResearchGroup_Mapped", "Manufacturer"]].copy()
    for col in key_df.columns:
        key_df[col] = key_df[col].fillna(f"{col}_UNKNOWN").astype(str)
    key = key_df.apply(lambda r: "_".join(r.values.astype(str)), axis=1)
    min_count = int(key.value_counts().min())
    if min_count < n_splits:
        label_key = df["y"].astype(int)
        return label_key, "label_only_fallback", int(label_key.value_counts().min())
    return key, "ResearchGroup_Mapped+Manufacturer", min_count


def manufacturer_auc_stats(df: pd.DataFrame, score_col: str = "score") -> Dict[str, Any]:
    aucs: Dict[str, float] = {}
    for manufacturer, sub in df.groupby("Manufacturer", dropna=False):
        auc = safe_auc(sub["y"], sub[score_col])
        if np.isfinite(auc):
            aucs[str(manufacturer)] = float(auc)
    vals = np.asarray(list(aucs.values()), dtype=float)
    return {
        "manufacturer_auc_json": json.dumps(aucs, sort_keys=True),
        "manufacturer_auc_mean": float(np.mean(vals)) if len(vals) else float("nan"),
        "manufacturer_auc_min": float(np.min(vals)) if len(vals) else float("nan"),
        "manufacturer_auc_std": float(np.std(vals, ddof=0)) if len(vals) else float("nan"),
        "n_manufacturer_auc_valid": int(len(vals)),
    }


def selection_score(rule: str, global_auc: float, manufacturer_mean: float, manufacturer_min: float, manufacturer_std: float) -> float:
    if rule == "global_auc":
        return float(global_auc)
    if rule == "manufacturer_mean_auc":
        return float(manufacturer_mean)
    if rule == "manufacturer_min_auc":
        return float(manufacturer_min)
    if rule == "global_auc_minus_0p5_manufacturer_std":
        return float(global_auc - 0.5 * manufacturer_std)
    raise ValueError(f"Unknown selection rule: {rule}")


def oof_for_params(
    strategy: Strategy,
    params: Dict[str, Any],
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    inner_cv: Sequence[Tuple[np.ndarray, np.ndarray]],
    mu_cols: Sequence[str],
    seed: int,
) -> np.ndarray:
    y = train_df["y"].astype(int).to_numpy()
    x = train_df[list(feature_cols)].copy()
    scores = np.full(len(train_df), np.nan, dtype=float)
    for inner_idx, (tr_idx, va_idx) in enumerate(inner_cv, start=1):
        estimator = make_estimator(strategy, params, mu_cols, seed=seed + inner_idx)
        estimator.fit(x.iloc[tr_idx], y[tr_idx])
        scores[va_idx] = score_1d(estimator, x.iloc[va_idx])
    if not np.isfinite(scores).all():
        raise RuntimeError(f"Non-finite OOF scores for {strategy.name} params={params}")
    return scores


def evaluate_candidates(
    strategy: Strategy,
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    inner_cv: Sequence[Tuple[np.ndarray, np.ndarray]],
    mu_cols: Sequence[str],
    fold: int,
) -> Tuple[Dict[str, Any], np.ndarray, pd.DataFrame]:
    candidate_rows: List[Dict[str, Any]] = []
    oof_cache: Dict[int, np.ndarray] = {}
    y = train_df["y"].astype(int).to_numpy()
    for idx, params in enumerate(strategy.grid):
        oof = oof_for_params(strategy, params, train_df, feature_cols, inner_cv, mu_cols, seed=SEED + fold * 100 + idx)
        oof_cache[idx] = oof
        scored = train_df[["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "SiteCode", "y"]].copy()
        scored["score"] = oof
        man_stats = manufacturer_auc_stats(scored)
        global_auc = safe_auc(y, oof)
        global_pr_auc = safe_pr_auc(y, oof)
        sel_score = selection_score(
            strategy.selection_rule,
            global_auc,
            float(man_stats["manufacturer_auc_mean"]),
            float(man_stats["manufacturer_auc_min"]),
            float(man_stats["manufacturer_auc_std"]),
        )
        row = {
            "fold": fold,
            "strategy": strategy.name,
            "model_type": strategy.model_type,
            "selection_rule": strategy.selection_rule,
            "candidate_index": idx,
            "params_json": json.dumps(params, sort_keys=True),
            "inner_oof_global_auc": global_auc,
            "inner_oof_global_pr_auc": global_pr_auc,
            "selection_score": sel_score,
            **man_stats,
        }
        candidate_rows.append(row)
    candidates = pd.DataFrame(candidate_rows)
    candidates = candidates.sort_values(
        ["selection_score", "inner_oof_global_auc", "manufacturer_auc_min", "manufacturer_auc_mean"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    best = candidates.iloc[0].to_dict()
    best_idx = int(best["candidate_index"])
    best_params = json.loads(str(best["params_json"]))
    return best | {"best_params": best_params}, oof_cache[best_idx], pd.DataFrame(candidate_rows)


def fit_score_test(
    strategy: Strategy,
    params: Dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    mu_cols: Sequence[str],
    fold: int,
) -> np.ndarray:
    estimator = make_estimator(strategy, params, mu_cols, seed=SEED + fold * 1000)
    estimator.fit(train_df[list(feature_cols)].copy(), train_df["y"].astype(int).to_numpy())
    return score_1d(estimator, test_df[list(feature_cols)].copy())


def pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (strategy, threshold_strategy), sub in predictions.groupby(["strategy", "threshold_strategy"], dropna=False):
        row = {
            "strategy": strategy,
            "threshold_strategy": threshold_strategy,
            "threshold": "fold_specific" if threshold_strategy != "fixed_0p5" else 0.5,
        }
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["threshold_strategy", "strategy"]).reset_index(drop=True)


def foldwise_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (fold, strategy, threshold_strategy), sub in predictions.groupby(["fold", "strategy", "threshold_strategy"], dropna=False):
        row = {"fold": int(fold), "strategy": strategy, "threshold_strategy": threshold_strategy}
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["strategy", "threshold_strategy", "fold"]).reset_index(drop=True)


def subgroup_by_manufacturer(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (strategy, threshold_strategy, manufacturer), sub in predictions.groupby(["strategy", "threshold_strategy", "Manufacturer"], dropna=False):
        row = {"strategy": strategy, "threshold_strategy": threshold_strategy, "Manufacturer": manufacturer}
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["threshold_strategy", "strategy", "Manufacturer"]).reset_index(drop=True)


def sitecode_auc(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (strategy, threshold_strategy, site), sub in predictions.groupby(["strategy", "threshold_strategy", "SiteCode"], dropna=False):
        n_cn = int((sub["y_true"] == 0).sum())
        n_ad = int((sub["y_true"] == 1).sum())
        eligible = len(sub) >= MIN_SITE_N and n_cn >= MIN_SITE_PER_CLASS and n_ad >= MIN_SITE_PER_CLASS
        row = {
            "strategy": strategy,
            "threshold_strategy": threshold_strategy,
            "SiteCode": site,
            "n": int(len(sub)),
            "n_cn": n_cn,
            "n_ad": n_ad,
            "site_auc_eligible": bool(eligible),
            "auc": safe_auc(sub["y_true"], sub["y_score"]) if eligible else float("nan"),
            "pr_auc": safe_pr_auc(sub["y_true"], sub["y_score"]) if eligible else float("nan"),
            "brier": float(brier_score_loss(sub["y_true"], np.clip(sub["y_score"], 0.0, 1.0))) if len(sub) else float("nan"),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["threshold_strategy", "strategy", "site_auc_eligible", "SiteCode"], ascending=[True, True, False, True]).reset_index(drop=True)


def error_focus(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    primary = predictions[predictions["threshold_strategy"].eq(PRIMARY_THRESHOLD)].copy()
    for strategy, sub in primary.groupby("strategy", dropna=False):
        philips_cn = sub[(sub["Manufacturer"].eq("Philips")) & (sub["y_true"].eq(0))]
        ge_ad = sub[(sub["Manufacturer"].eq("GE")) & (sub["y_true"].eq(1))]
        rows.append(
            {
                "strategy": strategy,
                "threshold_strategy": PRIMARY_THRESHOLD,
                "philips_cn_n": int(len(philips_cn)),
                "philips_cn_fp": int((philips_cn["y_pred"] == 1).sum()),
                "philips_cn_fp_rate": safe_div(float((philips_cn["y_pred"] == 1).sum()), float(len(philips_cn))),
                "ge_ad_n": int(len(ge_ad)),
                "ge_ad_fn": int((ge_ad["y_pred"] == 0).sum()),
                "ge_ad_fn_rate": safe_div(float((ge_ad["y_pred"] == 0).sum()), float(len(ge_ad))),
            }
        )
    return pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True)


def fold_hyperparameters(selected_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in selected_rows:
        params = row.get("best_params", {})
        flat = {f"param_{k}": v for k, v in params.items()}
        keep = {
            "fold": row["fold"],
            "strategy": row["strategy"],
            "model_type": row["model_type"],
            "selection_rule": row["selection_rule"],
            "selection_score": row["selection_score"],
            "inner_oof_global_auc": row["inner_oof_global_auc"],
            "inner_oof_global_pr_auc": row["inner_oof_global_pr_auc"],
            "manufacturer_auc_mean": row["manufacturer_auc_mean"],
            "manufacturer_auc_min": row["manufacturer_auc_min"],
            "manufacturer_auc_std": row["manufacturer_auc_std"],
            "manufacturer_auc_json": row["manufacturer_auc_json"],
        }
        rows.append(keep | flat | {"params_json": json.dumps(params, sort_keys=True)})
    return pd.DataFrame(rows).sort_values(["strategy", "fold"]).reset_index(drop=True)


def markdown_table(df: pd.DataFrame, path: Path, cols: Sequence[str] | None = None, max_rows: int | None = None) -> None:
    view = df.copy()
    if cols is not None:
        view = view[list(cols)]
    if max_rows is not None:
        view = view.head(max_rows)
    path.write_text(view.to_markdown(index=False) + "\n", encoding="utf-8")


def prepare_outdir() -> None:
    if OUTDIR.exists():
        shutil.rmtree(OUTDIR)
    OUTDIR.mkdir(parents=True, exist_ok=True)


def run_audit() -> Dict[str, pd.DataFrame]:
    required = [LOCKED_RUN_DIR, LATENT_CACHE_DIR, REFERENCE_SWEEP_DIR / "classifier_sweep_pooled_metrics.csv"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    pred_frames: List[pd.DataFrame] = []
    selected_rows: List[Dict[str, Any]] = []
    candidate_rows: List[pd.DataFrame] = []
    threshold_rows: List[Dict[str, Any]] = []
    strategies = strategy_specs()

    for fold in range(1, OUTER_FOLDS + 1):
        train_df, test_df = load_latents(fold)
        mu_cols = [c for c in train_df.columns if c.startswith("mu_")]
        feature_cols = mu_cols + ["Age", "Sex"]
        y_train = train_df["y"].astype(int).to_numpy()
        inner_key, inner_context, min_inner_count = inner_stratification_key(train_df, INNER_FOLDS)
        inner_cv = list(
            StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + fold + 30).split(
                np.zeros(len(train_df)), inner_key
            )
        )
        for strategy in strategies:
            best, oof_score, cand = evaluate_candidates(strategy, train_df, feature_cols, inner_cv, mu_cols, fold)
            cand["inner_cv_context"] = inner_context
            cand["minimum_inner_stratum_count"] = int(min_inner_count)
            candidate_rows.append(cand)
            selected_rows.append(best | {"inner_cv_context": inner_context, "minimum_inner_stratum_count": int(min_inner_count)})
            test_score = fit_score_test(strategy, best["best_params"], train_df, test_df, feature_cols, mu_cols, fold)
            thresholds = select_thresholds(y_train, oof_score)
            for sel in thresholds:
                threshold_rows.append(
                    {
                        "fold": fold,
                        "strategy": strategy.name,
                        "model_type": strategy.model_type,
                        "selection_rule": strategy.selection_rule,
                        "params_json": json.dumps(best["best_params"], sort_keys=True),
                        "inner_cv_context": inner_context,
                        "minimum_inner_stratum_count": int(min_inner_count),
                        **sel,
                    }
                )
                pred = test_df[
                    [
                        "SubjectID",
                        "tensor_idx",
                        "ResearchGroup_Mapped",
                        "Manufacturer",
                        "SiteCode",
                        "Age",
                        "Sex",
                        "source_batch",
                        "source_label",
                        "tensor_source",
                    ]
                ].copy()
                pred["fold"] = fold
                pred["strategy"] = strategy.name
                pred["model_type"] = strategy.model_type
                pred["selection_rule"] = strategy.selection_rule
                pred["threshold_strategy"] = sel["threshold_strategy"]
                pred["threshold"] = float(sel["threshold"])
                pred["threshold_selection_context"] = sel["threshold_selection_context"]
                pred["y_true"] = test_df["y"].astype(int).to_numpy()
                pred["y_score"] = test_score
                pred["y_pred"] = (test_score >= float(sel["threshold"])).astype(int)
                pred_frames.append(pred)

    predictions = pd.concat(pred_frames, ignore_index=True, sort=False)
    selected = fold_hyperparameters(selected_rows)
    candidates = pd.concat(candidate_rows, ignore_index=True, sort=False)
    folds = foldwise_metrics(predictions)
    pooled = pooled_metrics(predictions)
    manufacturers = subgroup_by_manufacturer(predictions)
    sites = sitecode_auc(predictions)
    focus = error_focus(predictions)
    thresholds = pd.DataFrame(threshold_rows).sort_values(["strategy", "threshold_strategy", "fold"]).reset_index(drop=True)

    primary = pooled[pooled["threshold_strategy"].eq(PRIMARY_THRESHOLD)].copy()
    primary = primary.merge(focus, on=["strategy", "threshold_strategy"], how="left")
    baseline = primary[primary["strategy"].eq("baseline_logreg_l2_current_selection")]
    if not baseline.empty:
        ref = baseline.iloc[0]
        for metric in ["auc", "pr_auc", "balanced_accuracy", "f1", "brier", "philips_cn_fp_rate", "ge_ad_fn_rate"]:
            primary[f"delta_vs_baseline_{metric}"] = primary[metric].astype(float) - float(ref[metric])

    return {
        "predictions": predictions,
        "selected_hyperparameters_by_fold": selected,
        "candidate_inner_cv_scores": candidates,
        "foldwise_metrics": folds,
        "pooled_metrics": pooled,
        "primary_comparison": primary.sort_values(["auc", "pr_auc", "balanced_accuracy"], ascending=[False, False, False]).reset_index(drop=True),
        "manufacturer_metrics": manufacturers,
        "sitecode_auc": sites,
        "philips_ge_error_focus": focus,
        "thresholds_by_fold": thresholds,
    }


def make_recommendation(primary: pd.DataFrame, manufacturer_metrics: pd.DataFrame) -> str:
    baseline = primary[primary["strategy"].eq("baseline_logreg_l2_current_selection")]
    if baseline.empty:
        return "Baseline row was not found; no promotion decision can be made.\n"
    ref = baseline.iloc[0]
    rows = []
    for _, row in primary.iterrows():
        if row["strategy"] == "baseline_logreg_l2_current_selection":
            continue
        auc_ok = float(row["auc"]) > float(ref["auc"])
        pr_ok = float(row["pr_auc"]) > float(ref["pr_auc"])
        philips_ok = float(row["philips_cn_fp_rate"]) <= float(ref["philips_cn_fp_rate"])
        ge_ok = float(row["ge_ad_fn_rate"]) <= float(ref["ge_ad_fn_rate"])
        promoted = auc_ok and pr_ok and philips_ok and ge_ok
        rows.append(
            {
                "strategy": row["strategy"],
                "auc_delta": float(row["auc"] - ref["auc"]),
                "pr_auc_delta": float(row["pr_auc"] - ref["pr_auc"]),
                "philips_cn_fp_rate_delta": float(row["philips_cn_fp_rate"] - ref["philips_cn_fp_rate"]),
                "ge_ad_fn_rate_delta": float(row["ge_ad_fn_rate"] - ref["ge_ad_fn_rate"]),
                "promoted": promoted,
            }
        )
    decision_df = pd.DataFrame(rows)
    promoted = decision_df[decision_df["promoted"].eq(True)] if not decision_df.empty else pd.DataFrame()
    best = primary.iloc[0]
    lines = [
        "# Frozen-Latent Classifier-Selection Recommendation",
        "",
        "## Decision",
        "",
    ]
    if promoted.empty:
        lines += [
            "No frozen-latent classifier-selection strategy satisfies the promotion rule.",
            "",
            "The locked current FULL `[1,0,2]` readout remains the manuscript classifier readout.",
        ]
    else:
        names = ", ".join(promoted["strategy"].astype(str).tolist())
        lines += [
            f"The following strategy satisfies the strict promotion rule: `{names}`.",
            "Treat this as a readout-only candidate; it still does not justify VAE retraining.",
        ]
    lines += [
        "",
        "## Best Primary-Threshold Row",
        "",
        f"- Strategy: `{best['strategy']}`",
        f"- AUC: `{float(best['auc']):.4f}`",
        f"- PR-AUC: `{float(best['pr_auc']):.4f}`",
        f"- BA: `{float(best['balanced_accuracy']):.4f}`",
        f"- F1: `{float(best['f1']):.4f}`",
        f"- Brier: `{float(best['brier']):.4f}`",
        f"- Philips CN FP rate: `{float(best['philips_cn_fp_rate']):.4f}`",
        f"- GE AD FN rate: `{float(best['ge_ad_fn_rate']):.4f}`",
        "",
        "## Promotion Rule Used",
        "",
        "Promote only if AUC and PR-AUC both improve versus locked current FULL and Philips CN FP / GE AD FN rates do not worsen.",
        "",
        "## Notes",
        "",
        "- All strategies use frozen VAE latent `mu + Age + Sex`.",
        "- SiteCode is derived from the first three SubjectID digits and is used only for reporting.",
        "- Non-0.5 thresholds use true inner-CV OOF scores from train/dev only.",
        "- No VAE retraining, tensor edits, metadata edits, or ledger edits were performed.",
    ]
    if not decision_df.empty:
        lines += ["", "## Strategy Deltas", "", decision_df.to_markdown(index=False)]
    return "\n".join(lines) + "\n"


def make_readme(outputs: Dict[str, pd.DataFrame]) -> str:
    primary = outputs["primary_comparison"]
    lines = [
        "# ADNI v5.1 batch20260514b Frozen-Latent Classifier-Selection Audit",
        "",
        "## Scope",
        "",
        "- Locked VAE run: current FULL tanh `[1,0,2]`.",
        "- Features: saved latent `mu + Age + Sex`.",
        "- Outer folds: existing locked 5 folds.",
        "- Inner CV: 5-fold `ResearchGroup_Mapped + Manufacturer` when feasible.",
        "- SiteCode: first three digits of SubjectID, reporting only.",
        "- VAE retrained: `False`.",
        "- Tensor/metadata/ledger modified: `False`.",
        "",
        "## Primary Threshold Comparison",
        "",
        primary[
            [
                "strategy",
                "auc",
                "pr_auc",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "f1",
                "brier",
                "philips_cn_fp_rate",
                "ge_ad_fn_rate",
            ]
        ].to_markdown(index=False),
        "",
        "## Outputs",
        "",
        "- `primary_model_comparison.csv/.md`",
        "- `pooled_metrics_all_thresholds.csv/.md`",
        "- `foldwise_metrics.csv/.md`",
        "- `selected_hyperparameters_by_fold.csv/.md`",
        "- `candidate_inner_cv_scores.csv`",
        "- `thresholds_by_fold.csv/.md`",
        "- `manufacturer_subgroup_metrics.csv/.md`",
        "- `sitecode_auc_report.csv/.md`",
        "- `philips_ge_error_focus.csv/.md`",
        "- `classifier_selection_predictions.csv`",
        "- `recommendation.md`",
        "- `command_log.json`",
    ]
    return "\n".join(lines) + "\n"


def save_outputs(outputs: Dict[str, pd.DataFrame]) -> None:
    prepare_outdir()
    csv_names = {
        "primary_comparison": "primary_model_comparison.csv",
        "pooled_metrics": "pooled_metrics_all_thresholds.csv",
        "foldwise_metrics": "foldwise_metrics.csv",
        "selected_hyperparameters_by_fold": "selected_hyperparameters_by_fold.csv",
        "candidate_inner_cv_scores": "candidate_inner_cv_scores.csv",
        "thresholds_by_fold": "thresholds_by_fold.csv",
        "manufacturer_metrics": "manufacturer_subgroup_metrics.csv",
        "sitecode_auc": "sitecode_auc_report.csv",
        "philips_ge_error_focus": "philips_ge_error_focus.csv",
        "predictions": "classifier_selection_predictions.csv",
    }
    for key, name in csv_names.items():
        outputs[key].to_csv(OUTDIR / name, index=False)

    markdown_table(
        outputs["primary_comparison"],
        OUTDIR / "primary_model_comparison.md",
        cols=[
            "strategy",
            "auc",
            "pr_auc",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1",
            "brier",
            "philips_cn_fp_rate",
            "ge_ad_fn_rate",
        ],
    )
    markdown_table(
        outputs["pooled_metrics"],
        OUTDIR / "pooled_metrics_all_thresholds.md",
        cols=["strategy", "threshold_strategy", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "brier"],
    )
    markdown_table(
        outputs["foldwise_metrics"],
        OUTDIR / "foldwise_metrics.md",
        cols=["fold", "strategy", "threshold_strategy", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "brier"],
    )
    markdown_table(
        outputs["selected_hyperparameters_by_fold"],
        OUTDIR / "selected_hyperparameters_by_fold.md",
        cols=[
            "fold",
            "strategy",
            "selection_rule",
            "params_json",
            "selection_score",
            "inner_oof_global_auc",
            "manufacturer_auc_mean",
            "manufacturer_auc_min",
            "manufacturer_auc_std",
        ],
    )
    markdown_table(
        outputs["thresholds_by_fold"],
        OUTDIR / "thresholds_by_fold.md",
        cols=[
            "fold",
            "strategy",
            "threshold_strategy",
            "threshold",
            "threshold_selection_context",
            "inner_oof_sensitivity",
            "inner_oof_specificity",
            "inner_oof_balanced_accuracy",
        ],
    )
    markdown_table(
        outputs["manufacturer_metrics"],
        OUTDIR / "manufacturer_subgroup_metrics.md",
        cols=["strategy", "threshold_strategy", "Manufacturer", "n", "n_cn", "n_ad", "auc", "pr_auc", "sensitivity", "specificity", "brier"],
    )
    markdown_table(
        outputs["sitecode_auc"],
        OUTDIR / "sitecode_auc_report.md",
        cols=["strategy", "threshold_strategy", "SiteCode", "n", "n_cn", "n_ad", "site_auc_eligible", "auc", "pr_auc", "brier"],
    )
    markdown_table(
        outputs["philips_ge_error_focus"],
        OUTDIR / "philips_ge_error_focus.md",
        cols=["strategy", "philips_cn_n", "philips_cn_fp", "philips_cn_fp_rate", "ge_ad_n", "ge_ad_fn", "ge_ad_fn_rate"],
    )

    (OUTDIR / "README.md").write_text(make_readme(outputs), encoding="utf-8")
    (OUTDIR / "recommendation.md").write_text(make_recommendation(outputs["primary_comparison"], outputs["manufacturer_metrics"]), encoding="utf-8")
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "locked_run_dir": str(LOCKED_RUN_DIR),
        "latent_cache_dir": str(LATENT_CACHE_DIR),
        "output_dir": str(OUTDIR),
        "strategies": [s.__dict__ | {"grid_n": len(s.grid)} for s in strategy_specs()],
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "threshold_selection": "true_inner_cv_oof_for_non_0p5_thresholds",
        "sitecode_rule": "first_three_digits_of_SubjectID_reporting_only",
        "vae_retrained": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "existing_result_folders_modified": False,
    }
    write_json(OUTDIR / "command_log.json", command_log)


def main() -> int:
    outputs = run_audit()
    save_outputs(outputs)
    primary = outputs["primary_comparison"]
    print(f"output_dir={OUTDIR}")
    print("vae_retrained=False")
    print("tensor_modified=False")
    print("metadata_modified=False")
    print("ledger_modified=False")
    print(primary[["strategy", "auc", "pr_auc", "balanced_accuracy", "f1", "brier", "philips_cn_fp_rate", "ge_ad_fn_rate"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
