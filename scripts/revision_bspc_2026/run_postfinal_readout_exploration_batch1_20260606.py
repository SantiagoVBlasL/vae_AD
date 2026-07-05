#!/usr/bin/env python3
"""Postfinal classifier-only readout exploration batch 1.

This script consumes already-trained fold-wise latent caches. It does not load
or train VAEs, does not touch tensors/metadata/model artifacts, and does not
score OASIS. All preprocessing, PCA/PLS, classifier hyperparameter selection,
and threshold selection are fitted inside train/dev data only.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results/revision_bspc_2026"
PREFLIGHT_DIR = RESULTS_ROOT / "postfinal_readout_exploration_plan_20260606"
DEFAULT_OUTPUT = RESULTS_ROOT / "postfinal_readout_exploration_batch1_20260606"

FOLDS = [1, 2, 3, 4, 5]
INNER_FOLDS = 5
STACK_BASE_INNER_FOLDS = 3
SEED = 42
TARGET_SENS = 0.70

PROMOTED_REF = {
    "auc": 0.795155,
    "pr_auc": 0.573934,
    "balanced_accuracy": 0.725979,
    "f1": 0.563492,
    "philips_cn_fpr": 0.4545,
}
CH1_REF = {
    "auc": 0.800378,
    "pr_auc": 0.585842,
    "philips_cn_fpr": 0.4646,
}

SELECTED_CANDIDATES = [
    "promoted_ch102_latent384_beta3p75__logreg_l2_baseline",
    "promoted_ch102_latent384_beta3p75__pca_logreg_l2",
    "promoted_ch102_latent384_beta3p75__pls_logreg_l2",
    "ch1only_latent384_beta3p75__pca_logreg_l2",
    "ch1only_latent384_beta3p75__pls_logreg_l2",
    "promoted_plus_ch1only__two_source_score_stacking_logreg",
]

C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
PCA_COMPONENTS = [16, 32, 64, 96, 128, 192]
PLS_COMPONENTS = [2, 4, 8, 16, 32]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def to_md(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class PLSFeatureExtractor(BaseEstimator, TransformerMixin):
    """Supervised PLS transformer for latent columns inside an sklearn Pipeline."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSFeatureExtractor":
        n = min(int(self.n_components), X.shape[1], max(1, X.shape[0] - 1))
        if n < 1:
            raise ValueError("PLSFeatureExtractor requires at least one component")
        self.n_components_ = n
        self.model_ = PLSRegression(n_components=n, scale=False)
        self.model_.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.model_.transform(X)


def load_preflight() -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = pd.read_csv(PREFLIGHT_DIR / "experiment_matrix.csv")
    sources = pd.read_csv(PREFLIGHT_DIR / "latent_source_inventory.csv")
    return matrix, sources


def source_cache(source_id: str, sources: pd.DataFrame) -> Path:
    row = sources[sources["source_id"].astype(str).eq(source_id)]
    if row.empty:
        raise KeyError(f"Unknown latent source: {source_id}")
    return PROJECT_ROOT / str(row.iloc[0]["latent_cache_dir"])


def load_fold(cache_dir: Path, fold: int, split: str) -> pd.DataFrame:
    path = cache_dir / f"fold_{fold}_{split}_latent_mu.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "y" not in df.columns:
        raise ValueError(f"{path} missing y")
    return df


def mu_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("mu_")]
    if not cols:
        raise ValueError("No mu_* latent columns found")
    return cols


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    cols = mu_cols(df)
    required = cols + ["Age", "Sex"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Latent cache missing required columns: {missing}")
    X = df[required].copy()
    X["Age"] = pd.to_numeric(X["Age"], errors="coerce")
    X["Sex"] = X["Sex"].astype(str).replace({"nan": np.nan, "None": np.nan, "": np.nan})
    y = pd.to_numeric(df["y"], errors="raise").astype(int).to_numpy()
    return X, y, cols


def make_pipeline(readout_id: str, params: dict[str, Any], latent_cols: list[str]) -> Pipeline:
    latent_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer()),
        ("scale", StandardScaler()),
    ]
    if readout_id == "pca_logreg_l2":
        latent_steps.append(("pca", PCA(n_components=int(params["n_components"]), random_state=SEED)))
    elif readout_id == "pls_logreg_l2":
        latent_steps.append(("pls", PLSFeatureExtractor(n_components=int(params["n_components"]))))
    elif readout_id == "logreg_l2_baseline":
        pass
    else:
        raise ValueError(f"Unsupported single-source readout: {readout_id}")

    pre = ColumnTransformer(
        transformers=[
            ("latent", Pipeline(latent_steps), latent_cols),
            ("age", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), ["Age"]),
            (
                "sex",
                Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", make_one_hot_encoder())]),
                ["Sex"],
            ),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        C=float(params["C"]),
        class_weight="balanced",
        max_iter=10000,
        penalty="l2",
        solver="liblinear",
        random_state=SEED,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def param_grid(readout_id: str, n_latent: int, n_inner_train_min: int) -> list[dict[str, Any]]:
    if readout_id == "logreg_l2_baseline":
        return [{"C": c} for c in C_GRID]
    if readout_id == "pca_logreg_l2":
        comps = [k for k in PCA_COMPONENTS if k <= n_latent and k <= n_inner_train_min - 1]
        return [{"C": c, "n_components": k} for k in comps for c in C_GRID]
    if readout_id == "pls_logreg_l2":
        comps = [k for k in PLS_COMPONENTS if k <= n_latent and k <= n_inner_train_min - 1]
        return [{"C": c, "n_components": k} for k in comps for c in C_GRID]
    raise ValueError(readout_id)


def inner_splits(y: np.ndarray, n_splits: int, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y))


def select_threshold(scores: np.ndarray, y: np.ndarray, target_sens: float = TARGET_SENS) -> dict[str, Any]:
    thresholds = np.unique(scores)
    thresholds = np.r_[thresholds.min() - 1e-12, thresholds, thresholds.max() + 1e-12]
    rows = []
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        ba = 0.5 * (sens + spec)
        rows.append((thr, sens, spec, ba, tn, fp, fn, tp))
    feasible = [r for r in rows if r[1] >= target_sens]
    chosen = max(feasible, key=lambda r: (r[2], r[3], r[0])) if feasible else max(rows, key=lambda r: (r[3], r[1], r[2]))
    return {
        "threshold": float(chosen[0]),
        "inner_sensitivity": float(chosen[1]),
        "inner_specificity": float(chosen[2]),
        "inner_balanced_accuracy": float(chosen[3]),
        "inner_tn": int(chosen[4]),
        "inner_fp": int(chosen[5]),
        "inner_fn": int(chosen[6]),
        "inner_tp": int(chosen[7]),
    }


def oof_scores_for_params(df: pd.DataFrame, y: np.ndarray, readout_id: str, params: dict[str, Any], folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    scores = np.full(len(y), np.nan, dtype=float)
    for tr, va in folds:
        X_tr, _, latent_cols = prepare_xy(df.iloc[tr].reset_index(drop=True))
        X_va, _, _ = prepare_xy(df.iloc[va].reset_index(drop=True))
        pipe = make_pipeline(readout_id, params, latent_cols)
        pipe.fit(X_tr, y[tr])
        scores[va] = pipe.predict_proba(X_va)[:, 1]
    if np.isnan(scores).any():
        raise RuntimeError("OOF scores contain NaN")
    return scores


def select_single_source(df: pd.DataFrame, readout_id: str, n_splits: int = INNER_FOLDS) -> dict[str, Any]:
    X, y, latent_cols = prepare_xy(df)
    folds = inner_splits(y, n_splits)
    min_inner_train = min(len(tr) for tr, _ in folds)
    grid = param_grid(readout_id, len(latent_cols), min_inner_train)
    results = []
    best: dict[str, Any] | None = None
    for params in grid:
        scores = oof_scores_for_params(df, y, readout_id, params, folds)
        auc = roc_auc_score(y, scores)
        pr = average_precision_score(y, scores)
        rec = {"params": params, "auc": auc, "pr_auc": pr, "scores": scores}
        results.append(rec)
        if best is None or (auc, pr) > (best["auc"], best["pr_auc"]):
            best = rec
    assert best is not None
    threshold = select_threshold(best["scores"], y)
    return {
        "params": best["params"],
        "inner_oof_auc": float(best["auc"]),
        "inner_oof_pr_auc": float(best["pr_auc"]),
        "inner_oof_scores": best["scores"],
        "threshold_info": threshold,
        "grid_results": results,
    }


def fit_score_single_source(train: pd.DataFrame, test: pd.DataFrame, readout_id: str, selected: dict[str, Any]) -> np.ndarray:
    X_train, y_train, latent_cols = prepare_xy(train)
    X_test, _, _ = prepare_xy(test)
    pipe = make_pipeline(readout_id, selected["params"], latent_cols)
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_test)[:, 1]


def metric_row(candidate_id: str, display: str, y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "candidate_id": candidate_id,
        "display_name": display,
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "predicted_ad_rate": float(pred.mean()),
    }


def manufacturer_association(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for suffix, sub in [("all", df), ("cn", df[df["y"] == 0])]:
        groups = [g["score"].to_numpy(dtype=float) for _, g in sub.groupby("Manufacturer") if len(g) > 1]
        if len(groups) >= 2:
            try:
                stat, p = kruskal(*groups)
            except ValueError:
                stat, p = np.nan, np.nan
        else:
            stat, p = np.nan, np.nan
        means = sub.groupby("Manufacturer")["score"].mean()
        out[f"score_mfr_kruskal_stat_{suffix}"] = float(stat) if pd.notna(stat) else np.nan
        out[f"score_mfr_kruskal_p_{suffix}"] = float(p) if pd.notna(p) else np.nan
        out[f"score_mfr_max_mean_diff_{suffix}"] = float(means.max() - means.min()) if len(means) >= 2 else np.nan
    return out


def philips_fpr(df: pd.DataFrame) -> dict[str, Any]:
    cn = df[(df["y"] == 0) & (df["Manufacturer"].astype(str).eq("Philips"))]
    n = len(cn)
    fp = int(cn["y_pred"].sum()) if n else 0
    return {"philips_cn_n": int(n), "philips_cn_fp": fp, "philips_cn_fpr": fp / n if n else np.nan}


def source_fold_predictions(candidate: pd.Series, sources: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    candidate_id = str(candidate["candidate_id"])
    display = str(candidate["latent_source_display"]) + " / " + str(candidate["readout_id"])
    cache = source_cache(str(candidate["latent_source_id"]), sources)
    readout_id = str(candidate["readout_id"])
    fold_metrics: list[dict[str, Any]] = []
    hyper_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for fold in FOLDS:
        train = load_fold(cache, fold, "trainDev")
        test = load_fold(cache, fold, "test")
        selected = select_single_source(train, readout_id, INNER_FOLDS)
        scores = fit_score_single_source(train, test, readout_id, selected)
        y_test = pd.to_numeric(test["y"], errors="raise").astype(int).to_numpy()
        thr = selected["threshold_info"]["threshold"]
        pred = (scores >= thr).astype(int)
        fm = metric_row(candidate_id, display, y_test, scores, pred)
        fm["fold"] = fold
        fold_metrics.append(fm)
        hp = {
            "candidate_id": candidate_id,
            "fold": fold,
            "readout_id": readout_id,
            "selected_params_json": json.dumps(selected["params"], sort_keys=True),
            "inner_oof_auc": selected["inner_oof_auc"],
            "inner_oof_pr_auc": selected["inner_oof_pr_auc"],
            "n_grid_rows": len(selected["grid_results"]),
        }
        hyper_rows.append(hp)
        tr = {"candidate_id": candidate_id, "fold": fold, **selected["threshold_info"]}
        threshold_rows.append(tr)
        pred_df = test[["SubjectID", "ResearchGroup_Mapped", "Age", "Sex", "Manufacturer", "y", "fold"]].copy()
        pred_df["candidate_id"] = candidate_id
        pred_df["score"] = scores
        pred_df["threshold"] = thr
        pred_df["y_pred"] = pred
        predictions.append(pred_df)
    return fold_metrics, hyper_rows, threshold_rows, pd.concat(predictions, ignore_index=True)


def align_two_sources(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = a.copy()
    b = b.copy()
    a["SubjectID"] = a["SubjectID"].astype(str)
    b["SubjectID"] = b["SubjectID"].astype(str)
    common = sorted(set(a["SubjectID"]) & set(b["SubjectID"]))
    if len(common) != len(a) or len(common) != len(b):
        raise ValueError("Two-source stack subject sets do not match")
    a = a.set_index("SubjectID").loc[common].reset_index()
    b = b.set_index("SubjectID").loc[common].reset_index()
    for col in ["y", "Age", "Sex", "Manufacturer"]:
        if col in a.columns and col in b.columns:
            if not a[col].astype(str).reset_index(drop=True).equals(b[col].astype(str).reset_index(drop=True)):
                raise ValueError(f"Two-source stack mismatch in {col}")
    return a, b


def base_oof_and_target_scores(train: pd.DataFrame, target: pd.DataFrame, n_splits: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    selected = select_single_source(train, "logreg_l2_baseline", n_splits)
    target_scores = fit_score_single_source(train, target, "logreg_l2_baseline", selected)
    return selected["inner_oof_scores"], target_scores, selected["params"]


def stack_fold_predictions(candidate_id: str, sources: pd.DataFrame, fold: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame]:
    src_a = source_cache("promoted_ch102_latent384_beta3p75", sources)
    src_b = source_cache("ch1only_latent384_beta3p75", sources)
    train_a, train_b = align_two_sources(load_fold(src_a, fold, "trainDev"), load_fold(src_b, fold, "trainDev"))
    test_a, test_b = align_two_sources(load_fold(src_a, fold, "test"), load_fold(src_b, fold, "test"))
    y_train = pd.to_numeric(train_a["y"], errors="raise").astype(int).to_numpy()
    y_test = pd.to_numeric(test_a["y"], errors="raise").astype(int).to_numpy()

    outer_inner = inner_splits(y_train, INNER_FOLDS)
    # Precompute leakage-safe meta OOF features once per inner split.
    split_records = []
    for tr, va in outer_inner:
        a_tr = train_a.iloc[tr].reset_index(drop=True)
        b_tr = train_b.iloc[tr].reset_index(drop=True)
        a_va = train_a.iloc[va].reset_index(drop=True)
        b_va = train_b.iloc[va].reset_index(drop=True)
        a_oof_tr, a_va_score, a_params = base_oof_and_target_scores(a_tr, a_va, STACK_BASE_INNER_FOLDS)
        b_oof_tr, b_va_score, b_params = base_oof_and_target_scores(b_tr, b_va, STACK_BASE_INNER_FOLDS)
        split_records.append(
            {
                "tr": tr,
                "va": va,
                "X_meta_tr": np.column_stack([a_oof_tr, b_oof_tr]),
                "y_meta_tr": y_train[tr],
                "X_meta_va": np.column_stack([a_va_score, b_va_score]),
                "a_base_params": a_params,
                "b_base_params": b_params,
            }
        )

    meta_results = []
    for c in C_GRID:
        oof_meta = np.full(len(y_train), np.nan)
        for rec in split_records:
            meta = LogisticRegression(C=c, class_weight="balanced", max_iter=10000, solver="liblinear", random_state=SEED)
            meta.fit(rec["X_meta_tr"], rec["y_meta_tr"])
            oof_meta[rec["va"]] = meta.predict_proba(rec["X_meta_va"])[:, 1]
        auc = roc_auc_score(y_train, oof_meta)
        pr = average_precision_score(y_train, oof_meta)
        meta_results.append({"C": c, "auc": auc, "pr_auc": pr, "scores": oof_meta})
    best = max(meta_results, key=lambda r: (r["auc"], r["pr_auc"]))
    threshold = select_threshold(best["scores"], y_train)

    a_oof_full, a_test_score, a_full_params = base_oof_and_target_scores(train_a, test_a, INNER_FOLDS)
    b_oof_full, b_test_score, b_full_params = base_oof_and_target_scores(train_b, test_b, INNER_FOLDS)
    meta = LogisticRegression(C=float(best["C"]), class_weight="balanced", max_iter=10000, solver="liblinear", random_state=SEED)
    meta.fit(np.column_stack([a_oof_full, b_oof_full]), y_train)
    test_score = meta.predict_proba(np.column_stack([a_test_score, b_test_score]))[:, 1]
    pred = (test_score >= threshold["threshold"]).astype(int)

    display = "promoted [1,0,2] + ch1-only / two-source score stacking"
    fm = metric_row(candidate_id, display, y_test, test_score, pred)
    fm["fold"] = fold
    hp = {
        "candidate_id": candidate_id,
        "fold": fold,
        "readout_id": "two_source_score_stacking_logreg",
        "selected_params_json": json.dumps(
            {
                "meta_C": best["C"],
                "full_train_promoted_base_params": a_full_params,
                "full_train_ch1_base_params": b_full_params,
            },
            sort_keys=True,
        ),
        "inner_oof_auc": best["auc"],
        "inner_oof_pr_auc": best["pr_auc"],
        "n_grid_rows": len(meta_results),
    }
    tr = {"candidate_id": candidate_id, "fold": fold, **threshold}
    pred_df = test_a[["SubjectID", "ResearchGroup_Mapped", "Age", "Sex", "Manufacturer", "y", "fold"]].copy()
    pred_df["candidate_id"] = candidate_id
    pred_df["score"] = test_score
    pred_df["threshold"] = threshold["threshold"]
    pred_df["y_pred"] = pred
    pred_df["base_promoted_score"] = a_test_score
    pred_df["base_ch1_score"] = b_test_score
    return fm, hp, tr, pred_df


def stack_predictions(candidate_id: str, sources: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    fold_metrics, hyper_rows, threshold_rows, predictions = [], [], [], []
    for fold in FOLDS:
        fm, hp, tr, pred = stack_fold_predictions(candidate_id, sources, fold)
        fold_metrics.append(fm)
        hyper_rows.append(hp)
        threshold_rows.append(tr)
        predictions.append(pred)
    return fold_metrics, hyper_rows, threshold_rows, pd.concat(predictions, ignore_index=True)


def pooled_outputs(predictions: pd.DataFrame, candidate_display: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled_rows = []
    philips_rows = []
    dist_rows = []
    assoc_rows = []
    for cid, df in predictions.groupby("candidate_id"):
        y = df["y"].to_numpy(dtype=int)
        score = df["score"].to_numpy(dtype=float)
        pred = df["y_pred"].to_numpy(dtype=int)
        row = metric_row(cid, candidate_display.get(cid, cid), y, score, pred)
        row.update(manufacturer_association(df))
        row.update(philips_fpr(df))
        pooled_rows.append(row)
        pf = philips_fpr(df)
        philips_rows.append({"candidate_id": cid, "display_name": candidate_display.get(cid, cid), **pf})
        assoc_rows.append(
            {
                "candidate_id": cid,
                "display_name": candidate_display.get(cid, cid),
                **manufacturer_association(df),
            }
        )
        for (mfr, y_val), sub in df.groupby(["Manufacturer", "y"]):
            n = len(sub)
            dist_rows.append(
                {
                    "candidate_id": cid,
                    "display_name": candidate_display.get(cid, cid),
                    "Manufacturer": mfr,
                    "y": int(y_val),
                    "diagnosis": "AD" if int(y_val) == 1 else "CN",
                    "n": n,
                    "score_mean": float(sub["score"].mean()),
                    "score_sd": float(sub["score"].std(ddof=1)) if n > 1 else np.nan,
                    "score_median": float(sub["score"].median()),
                    "predicted_ad_rate": float(sub["y_pred"].mean()),
                }
            )
    return pd.DataFrame(pooled_rows), pd.DataFrame(philips_rows), pd.DataFrame(dist_rows), pd.DataFrame(assoc_rows)


def promotion_gate(pooled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    promoted_assoc = pooled.loc[
        pooled["candidate_id"].eq("promoted_ch102_latent384_beta3p75__logreg_l2_baseline"),
        "score_mfr_max_mean_diff_cn",
    ]
    assoc_ref = float(promoted_assoc.iloc[0]) if not promoted_assoc.empty and pd.notna(promoted_assoc.iloc[0]) else np.nan
    for _, r in pooled.iterrows():
        auc = r["auc"]
        pr = r["pr_auc"]
        gate_auc = bool((auc > CH1_REF["auc"]) or (auc > PROMOTED_REF["auc"] and pr > PROMOTED_REF["pr_auc"]))
        gate_pr_min = bool(pr >= PROMOTED_REF["pr_auc"])
        gate_pr_ideal = bool(pr >= CH1_REF["pr_auc"])
        gate_ba_f1 = bool(r["balanced_accuracy"] >= PROMOTED_REF["balanced_accuracy"] and r["f1"] >= PROMOTED_REF["f1"])
        gate_philips = bool(r["philips_cn_fpr"] <= PROMOTED_REF["philips_cn_fpr"])
        assoc = r.get("score_mfr_max_mean_diff_cn", np.nan)
        gate_assoc = bool(pd.notna(assoc) and (pd.isna(assoc_ref) or assoc <= assoc_ref + 1e-12))
        promotes = gate_auc and gate_pr_min and gate_ba_f1 and gate_philips and gate_assoc
        decision = "promote_candidate" if promotes else "do_not_promote"
        if (not promotes) and gate_auc and gate_pr_min:
            decision = "sensitivity_only"
        rows.append(
            {
                "candidate_id": r["candidate_id"],
                "display_name": r["display_name"],
                "auc": auc,
                "pr_auc": pr,
                "balanced_accuracy": r["balanced_accuracy"],
                "sensitivity": r["sensitivity"],
                "specificity": r["specificity"],
                "f1": r["f1"],
                "philips_cn_fpr": r["philips_cn_fpr"],
                "score_mfr_max_mean_diff_cn": assoc,
                "gate_auc": gate_auc,
                "gate_pr_min_promoted": gate_pr_min,
                "gate_pr_ideal_ch1": gate_pr_ideal,
                "gate_ba_f1": gate_ba_f1,
                "gate_philips_cn_fpr": gate_philips,
                "gate_score_mfr_association": gate_assoc,
                "decision": decision,
            }
        )
    return pd.DataFrame(rows)


def write_final_decision(out: Path, gate: pd.DataFrame) -> None:
    promoted = gate[gate["decision"].eq("promote_candidate")]
    lines = [
        "# Final Decision",
        "",
        "Scope: postfinal classifier-only readout exploration batch 1. No VAE retraining, no OASIS scoring, and no OASIS threshold/calibration fitting were performed.",
        "",
    ]
    if promoted.empty:
        lines.append("Decision: **no batch-1 readout candidate is promoted**.")
    else:
        lines.append("Decision: **candidate(s) passing all promotion gates found**.")
    lines.extend(
        [
            "",
            "Primary gates applied:",
            "- ADNI AUC > 0.800378 preferred, or AUC > 0.795155 with PR-AUC improvement.",
            "- PR-AUC >= 0.573934 minimum; PR-AUC >= 0.585842 ideal.",
            "- BA/F1 not worse than the promoted reference.",
            "- Philips CN FPR <= 0.4545.",
            "- No increase in score-Manufacturer association versus the promoted baseline in this batch.",
            "",
            "Candidate summary:",
        ]
    )
    for _, r in gate.sort_values(["decision", "auc", "pr_auc"], ascending=[True, False, False]).iterrows():
        lines.append(
            f"- `{r['candidate_id']}`: {r['decision']}; "
            f"AUC={r['auc']:.6f}, PR-AUC={r['pr_auc']:.6f}, "
            f"BA={r['balanced_accuracy']:.6f}, F1={r['f1']:.6f}, "
            f"Philips CN FPR={r['philips_cn_fpr']:.4f}."
        )
    (out / "final_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate", action="append", default=None, help="Candidate ID to run; repeatable.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    matrix, sources = load_preflight()
    requested = args.candidate or SELECTED_CANDIDATES
    matrix = matrix[matrix["candidate_id"].isin(requested)].copy()
    missing = sorted(set(requested) - set(matrix["candidate_id"]))
    if missing:
        raise ValueError(f"Requested candidates absent from preflight matrix: {missing}")

    command_log: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "script": rel(Path(__file__)),
        "output_dir": rel(out),
        "candidate_ids": requested,
        "dry_run": args.dry_run,
        "guardrails": [
            "no VAE retraining",
            "no tensor modification",
            "no metadata modification",
            "no OASIS scoring during candidate selection",
            "all transforms fit inside inner CV or outer train/dev only",
            "PLS only inside sklearn Pipeline",
            "thresholds selected from inner OOF train/dev scores",
            "two-source stacking meta-logreg uses inner-OOF base scores only",
        ],
    }

    if args.dry_run:
        dry = matrix.copy()
        dry["status"] = "planned_not_run"
        dry.to_csv(out / "dry_run_candidates.csv", index=False)
        to_md(dry, out / "dry_run_candidates.md")
        (out / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return

    fold_metrics: list[dict[str, Any]] = []
    hyper_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    candidate_display: dict[str, str] = {}

    for _, cand in matrix.iterrows():
        cid = str(cand["candidate_id"])
        if cid == "promoted_plus_ch1only__two_source_score_stacking_logreg":
            fm, hp, th, pred = stack_predictions(cid, sources)
        else:
            fm, hp, th, pred = source_fold_predictions(cand, sources)
        fold_metrics.extend(fm)
        hyper_rows.extend(hp)
        threshold_rows.extend(th)
        pred_frames.append(pred)
        candidate_display[cid] = str(cand["latent_source_display"]) + " / " + str(cand["readout_id"])

    predictions = pd.concat(pred_frames, ignore_index=True)
    pooled, philips, dist, assoc = pooled_outputs(predictions, candidate_display)
    gate = promotion_gate(pooled)

    pooled.to_csv(out / "candidate_readout_metrics.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(out / "foldwise_metrics.csv", index=False)
    pd.DataFrame(hyper_rows).to_csv(out / "selected_hyperparameters.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(out / "thresholds_by_fold.csv", index=False)
    philips.to_csv(out / "philips_cn_fpr.csv", index=False)
    dist.to_csv(out / "score_distribution_by_manufacturer.csv", index=False)
    assoc.to_csv(out / "score_manufacturer_association.csv", index=False)
    gate.to_csv(out / "stagewise_promotion_gate_table.csv", index=False)
    predictions.to_csv(out / "candidate_predictions.csv", index=False)

    to_md(pooled, out / "candidate_readout_metrics.md")
    to_md(pd.DataFrame(fold_metrics), out / "foldwise_metrics.md")
    to_md(pd.DataFrame(hyper_rows), out / "selected_hyperparameters.md")
    to_md(pd.DataFrame(threshold_rows), out / "thresholds_by_fold.md")
    to_md(philips, out / "philips_cn_fpr.md")
    to_md(dist, out / "score_distribution_by_manufacturer.md")
    to_md(assoc, out / "score_manufacturer_association.md")
    to_md(gate, out / "stagewise_promotion_gate_table.md")
    write_final_decision(out, gate)

    command_log["outputs"] = sorted(p.name for p in out.iterdir())
    (out / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
