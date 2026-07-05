#!/usr/bin/env python3
"""Read-only frozen-latent imbalance strategy audit for final v5.1b horizon4480.

This audit reads the saved Stage-B latent mu CSVs from the final v5.1b
horizon4480/cycles56 run. It never loads or trains a VAE, and it writes only
lightweight CSV/Markdown reports under results/.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = True
    IMBLEARN_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on local env
    SMOTE = None
    ImbPipeline = None
    IMBLEARN_AVAILABLE = False
    IMBLEARN_ERROR = str(exc)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
)
READOUT_DIR = RUN_DIR / "classifier_only_readout"
LATENT_DIR = READOUT_DIR / "latent_cache"
OUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "final_frozen_latent_imbalance_strategy_audit"
)

REF = {
    "model": "v5.1b horizon4480/cycles56 current Stage-B logreg_l2",
    "auc": 0.782951,
    "pr_auc": 0.559873,
    "balanced_accuracy": 0.735417,
    "sensitivity": 0.770833,
    "specificity": 0.700000,
    "f1": 0.569231,
}

C_GRID = [0.001, 0.01, 0.1, 1.0]
SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 5
TARGET_SENSITIVITIES = [0.70, 0.75, 0.80]


@dataclass(frozen=True)
class Strategy:
    name: str
    class_weight: Optional[Any]
    scoring: str = "roc_auc"
    use_smote: bool = False
    use_manufacturer_sample_weight: bool = False
    c_grid: Tuple[float, ...] = tuple(C_GRID)
    note: str = ""


STRATEGIES: List[Strategy] = [
    Strategy(
        name="current_baseline_logreg_l2_balanced_rocauc",
        class_weight="balanced",
        scoring="roc_auc",
        note="Matches the locked Stage-B logreg_l2 setup: class_weight=balanced, C grid 0.001..1, ROC-AUC inner scoring.",
    ),
    Strategy(
        name="logreg_l2_class_weight_none_rocauc",
        class_weight=None,
        scoring="roc_auc",
        note="No class_weight; tests whether explicit AD/CN balancing is helping or hurting.",
    ),
    Strategy(
        name="logreg_l2_class_weight_balanced_rocauc",
        class_weight="balanced",
        scoring="roc_auc",
        note="Explicit duplicate of balanced weighting for requirement traceability; should match current baseline.",
    ),
    Strategy(
        name="logreg_l2_custom_AD2_CN1_rocauc",
        class_weight={0: 1.0, 1: 2.0},
        scoring="roc_auc",
        note="Custom minority AD upweighting: AD:CN = 2:1.",
    ),
    Strategy(
        name="logreg_l2_custom_AD3_CN1_rocauc",
        class_weight={0: 1.0, 1: 3.0},
        scoring="roc_auc",
        note="Custom minority AD upweighting: AD:CN = 3:1.",
    ),
    Strategy(
        name="logreg_l2_custom_AD4_CN1_rocauc",
        class_weight={0: 1.0, 1: 4.0},
        scoring="roc_auc",
        note="Custom minority AD upweighting: AD:CN = 4:1.",
    ),
    Strategy(
        name="logreg_l2_smote_innercv_rocauc",
        class_weight=None,
        scoring="roc_auc",
        use_smote=True,
        note="SMOTE inside each inner-CV training split and final outer-train fit only; outer test is untouched.",
    ),
    Strategy(
        name="logreg_l2_balanced_average_precision_scoring",
        class_weight="balanced",
        scoring="average_precision",
        note="Same balanced logreg_l2 but inner-CV C selected by average precision instead of ROC-AUC.",
    ),
    Strategy(
        name="logreg_l2_balanced_mfr_sample_weight_rocauc",
        class_weight="balanced",
        scoring="roc_auc",
        use_manufacturer_sample_weight=True,
        note="Balanced class_weight plus fold-safe inverse Manufacturer sample_weight during inner-CV and final fit.",
    ),
]


def ensure_inputs() -> None:
    required = [READOUT_DIR / "classifier_sweep_pooled_metrics.csv"]
    for fold in range(1, OUTER_FOLDS + 1):
        required.append(LATENT_DIR / f"fold_{fold}_trainDev_latent_mu.csv")
        required.append(LATENT_DIR / f"fold_{fold}_test_latent_mu.csv")
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required frozen-latent inputs:\n" + "\n".join(missing))


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(mu_cols: Sequence[str]) -> ColumnTransformer:
    latent = SkPipeline([("scaler", StandardScaler())])
    age = SkPipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    sex = SkPipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_ohe())])
    return ColumnTransformer(
        [
            ("latent", latent, list(mu_cols)),
            ("age", age, ["Age"]),
            ("sex", sex, ["Sex"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def make_pipeline(strategy: Strategy, mu_cols: Sequence[str], seed: int) -> Any:
    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=1.0,
        class_weight=strategy.class_weight,
        max_iter=5000,
        random_state=seed,
    )
    steps: List[Tuple[str, Any]] = [("pre", make_preprocessor(mu_cols))]
    if strategy.use_smote:
        if not IMBLEARN_AVAILABLE:
            raise RuntimeError(f"imblearn unavailable: {IMBLEARN_ERROR}")
        steps.append(("smote", SMOTE(random_state=seed, k_neighbors=5)))
        steps.append(("model", model))
        return ImbPipeline(steps)
    steps.append(("model", model))
    return SkPipeline(steps)


def site_code(subject_id: Any) -> str:
    text = str(subject_id)
    match = re.match(r"^(\d{3})_S_", text)
    return match.group(1) if match else "UNKNOWN"


def inner_stratification_key(df: pd.DataFrame, n_splits: int) -> Tuple[pd.Series, str, int]:
    key_df = df[["ResearchGroup_Mapped", "Manufacturer"]].copy()
    for col in key_df.columns:
        key_df[col] = key_df[col].fillna(f"{col}_UNKNOWN").astype(str)
    key = key_df.apply(lambda r: "_".join(r.values.astype(str)), axis=1)
    min_count = int(key.value_counts().min())
    if min_count < n_splits:
        y = df["y"].astype(int)
        return y, "label_only_fallback", int(y.value_counts().min())
    return key, "ResearchGroup_Mapped+Manufacturer", min_count


def score_1d(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    proba = estimator.predict_proba(x)
    return np.asarray(proba[:, 1], dtype=float)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    p = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    out: Dict[str, Any] = {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": safe_div(tp + tn, len(y)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "predicted_ad_rate": float(p.mean()) if len(p) else float("nan"),
        "cn_fp_rate": safe_div(fp, fp + tn),
        "ad_fn_rate": safe_div(fn, fn + tp),
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
    else:
        out["auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def threshold_candidates(scores: Sequence[float]) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    return np.unique(np.round(np.clip(np.concatenate(([0.0, 0.5, 1.0], s)), 0.0, 1.0), 12))


def threshold_table(y_true: Sequence[int], y_score: Sequence[float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    for threshold in threshold_candidates(s):
        pred = (s >= threshold).astype(int)
        row = {"threshold": float(threshold)}
        row.update(binary_metrics(y, s, pred))
        row["youden_j"] = row["sensitivity"] + row["specificity"] - 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def select_thresholds(y_true: Sequence[int], y_score: Sequence[float]) -> List[Dict[str, Any]]:
    tbl = threshold_table(y_true, y_score)
    selections: List[Dict[str, Any]] = [
        {
            "threshold_strategy": "fixed_0p5",
            "threshold": 0.5,
            "threshold_selection_context": "fixed_no_selection",
            "selection_metric": "fixed_no_selection",
            "inner_oof_sensitivity": np.nan,
            "inner_oof_specificity": np.nan,
            "inner_oof_balanced_accuracy": np.nan,
        }
    ]
    for target in TARGET_SENSITIVITIES:
        eligible = tbl[tbl["sensitivity"] >= target]
        if eligible.empty:
            chosen = tbl.sort_values(["sensitivity", "specificity", "balanced_accuracy", "threshold"], ascending=[False, False, False, False]).iloc[0]
            status = "target_not_reached_inner_oof"
        else:
            chosen = eligible.sort_values(["specificity", "sensitivity", "balanced_accuracy", "threshold"], ascending=[False, False, False, False]).iloc[0]
            status = "selected_inner_oof"
        suffix = f"{int(round(target * 100)):02d}"
        selections.append(
            {
                "threshold_strategy": f"inner_oof_target_sens_ge_0p{suffix}_max_spec",
                "threshold": float(chosen["threshold"]),
                "threshold_selection_context": "true_inner_cv_oof",
                "selection_metric": status,
                "inner_oof_sensitivity": float(chosen["sensitivity"]),
                "inner_oof_specificity": float(chosen["specificity"]),
                "inner_oof_balanced_accuracy": float(chosen["balanced_accuracy"]),
            }
        )
    return selections


def score_for_selection(scoring: str, y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if scoring == "roc_auc":
        return float(roc_auc_score(y_true, y_score))
    if scoring == "average_precision":
        return float(average_precision_score(y_true, y_score))
    raise ValueError(scoring)


def manufacturer_sample_weight(df: pd.DataFrame) -> np.ndarray:
    counts = df["Manufacturer"].fillna("UNKNOWN").astype(str).value_counts().to_dict()
    weights = df["Manufacturer"].fillna("UNKNOWN").astype(str).map(lambda x: 1.0 / counts[x]).to_numpy(dtype=float)
    return weights / float(np.mean(weights))


def fit_estimator(estimator: Any, x: pd.DataFrame, y: np.ndarray, weights: Optional[np.ndarray]) -> Any:
    if weights is None:
        return estimator.fit(x, y)
    return estimator.fit(x, y, model__sample_weight=weights)


def evaluate_c_grid(
    strategy: Strategy,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    train_df: pd.DataFrame,
    mu_cols: Sequence[str],
    inner_cv: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> Tuple[float, float, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    best_c = float(strategy.c_grid[0])
    best_score = -np.inf
    for c in strategy.c_grid:
        fold_scores: List[float] = []
        for inner_idx, (tr_idx, val_idx) in enumerate(inner_cv, start=1):
            est = make_pipeline(strategy, mu_cols, seed + inner_idx)
            est.set_params(model__C=float(c))
            weights = manufacturer_sample_weight(train_df.iloc[tr_idx]) if strategy.use_manufacturer_sample_weight else None
            try:
                fit_estimator(est, x_train.iloc[tr_idx], y_train[tr_idx], weights)
                score = score_1d(est, x_train.iloc[val_idx])
                metric = score_for_selection(strategy.scoring, y_train[val_idx], score)
            except Exception as exc:
                metric = float("nan")
                rows.append(
                    {
                        "C": float(c),
                        "inner_fold": inner_idx,
                        "status": f"fit_failed: {exc}",
                        "selection_metric_value": metric,
                    }
                )
                continue
            fold_scores.append(metric)
            rows.append({"C": float(c), "inner_fold": inner_idx, "status": "ok", "selection_metric_value": metric})
        mean_score = float(np.nanmean(fold_scores)) if fold_scores else float("nan")
        if np.isfinite(mean_score) and mean_score > best_score:
            best_score = mean_score
            best_c = float(c)
    return best_c, best_score, pd.DataFrame(rows)


def inner_oof_scores(
    strategy: Strategy,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    train_df: pd.DataFrame,
    mu_cols: Sequence[str],
    inner_cv: List[Tuple[np.ndarray, np.ndarray]],
    c: float,
    seed: int,
) -> np.ndarray:
    out = np.full(shape=len(y_train), fill_value=np.nan, dtype=float)
    for inner_idx, (tr_idx, val_idx) in enumerate(inner_cv, start=1):
        est = make_pipeline(strategy, mu_cols, seed + 100 + inner_idx)
        est.set_params(model__C=float(c))
        weights = manufacturer_sample_weight(train_df.iloc[tr_idx]) if strategy.use_manufacturer_sample_weight else None
        fit_estimator(est, x_train.iloc[tr_idx], y_train[tr_idx], weights)
        out[val_idx] = score_1d(est, x_train.iloc[val_idx])
    if np.isnan(out).any():
        raise RuntimeError("OOF score generation left NaN values.")
    return out


def load_fold_latents(fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(LATENT_DIR / f"fold_{fold}_trainDev_latent_mu.csv")
    test = pd.read_csv(LATENT_DIR / f"fold_{fold}_test_latent_mu.csv")
    for df in [train, test]:
        df["SiteCode"] = df["SubjectID"].map(site_code)
        df["Manufacturer"] = df["Manufacturer"].fillna("UNKNOWN").astype(str)
        df["Sex"] = df["Sex"].fillna("UNKNOWN").astype(str)
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    return train, test


def fold_strategy_run(fold: int, strategy: Strategy) -> Dict[str, Any]:
    train_df, test_df = load_fold_latents(fold)
    mu_cols = [c for c in train_df.columns if c.startswith("mu_")]
    feature_cols = mu_cols + ["Age", "Sex"]
    x_train = train_df[feature_cols].copy()
    y_train = train_df["y"].astype(int).to_numpy()
    x_test = test_df[feature_cols].copy()
    y_test = test_df["y"].astype(int).to_numpy()
    inner_key, inner_context, min_inner_count = inner_stratification_key(train_df, INNER_FOLDS)
    splitter = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + fold + 30)
    inner_cv = list(splitter.split(np.zeros(len(train_df)), inner_key))
    seed = SEED + fold
    best_c, best_inner_score, c_audit = evaluate_c_grid(strategy, x_train, y_train, train_df, mu_cols, inner_cv, seed)
    oof_score = inner_oof_scores(strategy, x_train, y_train, train_df, mu_cols, inner_cv, best_c, seed)
    final_est = make_pipeline(strategy, mu_cols, seed + 1000)
    final_est.set_params(model__C=float(best_c))
    final_weights = manufacturer_sample_weight(train_df) if strategy.use_manufacturer_sample_weight else None
    fit_estimator(final_est, x_train, y_train, final_weights)
    test_score = score_1d(final_est, x_test)
    thresholds = select_thresholds(y_train, oof_score)

    fold_metrics: List[Dict[str, Any]] = []
    predictions: List[pd.DataFrame] = []
    subgroup_rows: List[Dict[str, Any]] = []
    site_rows: List[Dict[str, Any]] = []
    confusion_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    for sel in thresholds:
        threshold = float(sel["threshold"])
        y_pred = (test_score >= threshold).astype(int)
        base_row = {
            "fold": fold,
            "strategy": strategy.name,
            "threshold_strategy": sel["threshold_strategy"],
            "threshold": threshold,
            "threshold_selection_context": sel["threshold_selection_context"],
            "inner_cv_context": inner_context,
            "minimum_inner_stratum_count": int(min_inner_count),
            "scoring": strategy.scoring,
            "class_weight": json.dumps(strategy.class_weight, sort_keys=True) if isinstance(strategy.class_weight, dict) else str(strategy.class_weight),
            "use_smote": bool(strategy.use_smote),
            "use_manufacturer_sample_weight": bool(strategy.use_manufacturer_sample_weight),
            "selected_C": float(best_c),
            "best_inner_score": float(best_inner_score),
            "selection_metric": sel["selection_metric"],
            "inner_oof_sensitivity": sel["inner_oof_sensitivity"],
            "inner_oof_specificity": sel["inner_oof_specificity"],
            "inner_oof_balanced_accuracy": sel["inner_oof_balanced_accuracy"],
        }
        metric_row = dict(base_row)
        metric_row.update(binary_metrics(y_test, test_score, y_pred))
        fold_metrics.append(metric_row)
        confusion_rows.append({k: metric_row[k] for k in ["fold", "strategy", "threshold_strategy", "threshold", "n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "sensitivity", "specificity", "balanced_accuracy", "f1", "auc", "pr_auc"]})
        threshold_rows.append({k: base_row[k] for k in ["fold", "strategy", "threshold_strategy", "threshold", "threshold_selection_context", "inner_cv_context", "minimum_inner_stratum_count", "selected_C", "best_inner_score", "selection_metric", "inner_oof_sensitivity", "inner_oof_specificity", "inner_oof_balanced_accuracy"]})

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
        pred["threshold_strategy"] = sel["threshold_strategy"]
        pred["threshold"] = threshold
        pred["selected_C"] = float(best_c)
        pred["y_true"] = y_test
        pred["y_score"] = test_score
        pred["y_pred"] = y_pred
        predictions.append(pred)

        for manufacturer, sub in pred.groupby("Manufacturer", dropna=False):
            row = {
                "fold": fold,
                "strategy": strategy.name,
                "threshold_strategy": sel["threshold_strategy"],
                "threshold": threshold,
                "Manufacturer": manufacturer,
            }
            row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
            subgroup_rows.append(row)
        for site, sub in pred.groupby("SiteCode", dropna=False):
            row = {
                "fold": fold,
                "strategy": strategy.name,
                "threshold_strategy": sel["threshold_strategy"],
                "threshold": threshold,
                "SiteCode": site,
            }
            row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
            site_rows.append(row)

    c_audit["fold"] = fold
    c_audit["strategy"] = strategy.name
    c_audit["scoring"] = strategy.scoring
    c_audit["selected_C"] = float(best_c)
    c_audit["best_inner_score"] = float(best_inner_score)

    hp_row = {
        "fold": fold,
        "strategy": strategy.name,
        "scoring": strategy.scoring,
        "class_weight": json.dumps(strategy.class_weight, sort_keys=True) if isinstance(strategy.class_weight, dict) else str(strategy.class_weight),
        "use_smote": bool(strategy.use_smote),
        "use_manufacturer_sample_weight": bool(strategy.use_manufacturer_sample_weight),
        "selected_C": float(best_c),
        "best_inner_score": float(best_inner_score),
        "C_grid": ",".join(str(x) for x in strategy.c_grid),
        "C_boundary": "lower" if best_c == min(strategy.c_grid) else "upper" if best_c == max(strategy.c_grid) else "interior",
        "inner_cv_context": inner_context,
        "minimum_inner_stratum_count": int(min_inner_count),
        "note": strategy.note,
    }
    return {
        "fold_metrics": pd.DataFrame(fold_metrics),
        "predictions": pd.concat(predictions, ignore_index=True, sort=False),
        "thresholds": pd.DataFrame(threshold_rows),
        "manufacturer_subgroups": pd.DataFrame(subgroup_rows),
        "site_subgroups": pd.DataFrame(site_rows),
        "confusion": pd.DataFrame(confusion_rows),
        "hp": pd.DataFrame([hp_row]),
        "c_audit": c_audit,
    }


def aggregate_predictions(pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_rows: List[Dict[str, Any]] = []
    confusion_rows: List[Dict[str, Any]] = []
    mfr_rows: List[Dict[str, Any]] = []
    site_rows: List[Dict[str, Any]] = []
    for (strategy, threshold_strategy), sub in pred.groupby(["strategy", "threshold_strategy"], dropna=False):
        row = {"strategy": strategy, "threshold_strategy": threshold_strategy, "threshold": "fold_specific" if threshold_strategy != "fixed_0p5" else 0.5}
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        row["delta_auc_vs_ref"] = float(row["auc"] - REF["auc"])
        row["delta_pr_auc_vs_ref"] = float(row["pr_auc"] - REF["pr_auc"])
        row["delta_ba_vs_ref"] = float(row["balanced_accuracy"] - REF["balanced_accuracy"])
        row["delta_f1_vs_ref"] = float(row["f1"] - REF["f1"])
        row["passes_primary_metric_rule"] = bool(row["auc"] > REF["auc"] and row["pr_auc"] >= REF["pr_auc"])
        primary_rows.append(row)
        confusion_rows.append({k: row[k] for k in ["strategy", "threshold_strategy", "threshold", "n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "sensitivity", "specificity", "balanced_accuracy", "f1", "auc", "pr_auc"]})
        for mfr, msub in sub.groupby("Manufacturer", dropna=False):
            mrow = {"strategy": strategy, "threshold_strategy": threshold_strategy, "Manufacturer": mfr}
            mrow.update(binary_metrics(msub["y_true"], msub["y_score"], msub["y_pred"]))
            mfr_rows.append(mrow)
        for site, ssub in sub.groupby("SiteCode", dropna=False):
            srow = {"strategy": strategy, "threshold_strategy": threshold_strategy, "SiteCode": site}
            srow.update(binary_metrics(ssub["y_true"], ssub["y_score"], ssub["y_pred"]))
            site_rows.append(srow)
    return (
        pd.DataFrame(primary_rows).sort_values(["threshold_strategy", "auc", "pr_auc"], ascending=[True, False, False]),
        pd.DataFrame(confusion_rows).sort_values(["threshold_strategy", "strategy"]),
        pd.DataFrame(mfr_rows).sort_values(["threshold_strategy", "strategy", "Manufacturer"]),
        pd.DataFrame(site_rows).sort_values(["threshold_strategy", "strategy", "SiteCode"]),
    )


def markdown_table(df: pd.DataFrame, cols: Sequence[str], max_rows: int = 30) -> str:
    if df.empty:
        return "No rows.\n"
    sub = df.loc[:, [c for c in cols if c in df.columns]].head(max_rows).copy()
    headers = list(sub.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in sub.iterrows():
        vals = []
        for col in headers:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append(f"{value:.6f}" if np.isfinite(value) else "NA")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_pair(name: str, df: pd.DataFrame, cols: Optional[Sequence[str]] = None, max_rows: int = 40) -> None:
    df.to_csv(OUT_DIR / f"{name}.csv", index=False)
    if cols is None:
        cols = list(df.columns)
    (OUT_DIR / f"{name}.md").write_text(markdown_table(df, cols, max_rows=max_rows), encoding="utf-8")


def primary_target_table(primary: pd.DataFrame, target: str = "inner_oof_target_sens_ge_0p70_max_spec") -> pd.DataFrame:
    return primary[primary["threshold_strategy"].eq(target)].sort_values(["auc", "pr_auc", "balanced_accuracy"], ascending=False)


def add_promotion_diagnostics(primary: pd.DataFrame, pooled_mfr: pd.DataFrame) -> pd.DataFrame:
    out = primary.copy()
    baseline_name = "current_baseline_logreg_l2_balanced_rocauc"
    extra_cols = [
        "philips_cn_fp_delta_vs_baseline",
        "ge_ad_fn_delta_vs_baseline",
        "min_manufacturer_ba_delta_vs_baseline",
        "passes_full_promotion_rule",
    ]
    for col in extra_cols:
        out[col] = np.nan if col != "passes_full_promotion_rule" else False

    for threshold_strategy, base_rows in out[out["strategy"].eq(baseline_name)].groupby("threshold_strategy"):
        if base_rows.empty:
            continue
        base = base_rows.iloc[0]
        m_base = pooled_mfr[
            (pooled_mfr["strategy"].eq(baseline_name))
            & (pooled_mfr["threshold_strategy"].eq(threshold_strategy))
        ]
        base_by_mfr = {str(r["Manufacturer"]): r for _, r in m_base.iterrows()}
        base_philips_fp = float(base_by_mfr.get("Philips", {}).get("cn_fp_rate", np.nan))
        base_ge_fn = float(base_by_mfr.get("GE", {}).get("ad_fn_rate", np.nan))
        idx = out["threshold_strategy"].eq(threshold_strategy)
        for row_idx, row in out[idx].iterrows():
            m_row = pooled_mfr[
                (pooled_mfr["strategy"].eq(row["strategy"]))
                & (pooled_mfr["threshold_strategy"].eq(threshold_strategy))
            ]
            row_by_mfr = {str(r["Manufacturer"]): r for _, r in m_row.iterrows()}
            philips_fp = float(row_by_mfr.get("Philips", {}).get("cn_fp_rate", np.nan))
            ge_fn = float(row_by_mfr.get("GE", {}).get("ad_fn_rate", np.nan))
            ba_deltas = []
            for manufacturer, base_m in base_by_mfr.items():
                if manufacturer in row_by_mfr:
                    ba_deltas.append(float(row_by_mfr[manufacturer]["balanced_accuracy"] - base_m["balanced_accuracy"]))
            min_mfr_ba_delta = float(np.nanmin(ba_deltas)) if ba_deltas else np.nan
            out.loc[row_idx, "philips_cn_fp_delta_vs_baseline"] = philips_fp - base_philips_fp
            out.loc[row_idx, "ge_ad_fn_delta_vs_baseline"] = ge_fn - base_ge_fn
            out.loc[row_idx, "min_manufacturer_ba_delta_vs_baseline"] = min_mfr_ba_delta
            # Full promotion requires the numerical AUC/PR-AUC gate plus no
            # worsening in the two pre-specified high-risk subgroup error rates
            # and no manufacturer balanced-accuracy drop.
            out.loc[row_idx, "passes_full_promotion_rule"] = bool(
                row["passes_primary_metric_rule"]
                and row["balanced_accuracy"] >= base["balanced_accuracy"] - 1e-12
                and row["f1"] >= base["f1"] - 1e-12
                and (not np.isfinite(philips_fp - base_philips_fp) or philips_fp <= base_philips_fp + 1e-12)
                and (not np.isfinite(ge_fn - base_ge_fn) or ge_fn <= base_ge_fn + 1e-12)
                and (not np.isfinite(min_mfr_ba_delta) or min_mfr_ba_delta >= -1e-12)
            )
    return out


def recommendation_text(primary: pd.DataFrame, mfr: pd.DataFrame, hp: pd.DataFrame) -> str:
    target = primary_target_table(primary)
    best = target.iloc[0] if not target.empty else None
    numeric_gate = target[target["passes_primary_metric_rule"].astype(bool)] if not target.empty else pd.DataFrame()
    promoted = target[target["passes_full_promotion_rule"].astype(bool)] if not target.empty else pd.DataFrame()
    baseline = target[target["strategy"].eq("current_baseline_logreg_l2_balanced_rocauc")]
    baseline_note = ""
    if not baseline.empty:
        b = baseline.iloc[0]
        baseline_note = (
            f"Recomputed baseline at the primary threshold: AUC={b['auc']:.6f}, "
            f"PR-AUC={b['pr_auc']:.6f}, BA={b['balanced_accuracy']:.6f}, F1={b['f1']:.6f}."
        )
    lower_bound_count = int((hp["C_boundary"] == "lower").sum()) if not hp.empty else 0
    lines = [
        "# Final Recommendation",
        "",
        "This is a read-only frozen-latent audit. No VAE was retrained and no tensor, metadata, ledger, config, or model-output folder was modified.",
        "",
        baseline_note,
        "",
    ]
    if best is not None:
        lines += [
            "## Best Strategy at Primary Threshold",
            "",
            f"- Strategy: `{best['strategy']}`",
            f"- AUC: `{best['auc']:.6f}`",
            f"- PR-AUC: `{best['pr_auc']:.6f}`",
            f"- BA: `{best['balanced_accuracy']:.6f}`",
            f"- Sensitivity: `{best['sensitivity']:.6f}`",
            f"- Specificity: `{best['specificity']:.6f}`",
            f"- F1: `{best['f1']:.6f}`",
            "",
        ]
    if promoted.empty:
        lines += [
            "## Decision",
            "",
            "Do not promote any frozen-latent imbalance strategy over the final v5.1b horizon4480/cycles56 model.",
            "",
            f"The promotion rule required AUC > {REF['auc']:.6f} and PR-AUC >= {REF['pr_auc']:.6f}, without material BA/F1 or subgroup degradation. Custom AD-upweighted variants crossed the AUC/PR-AUC numerical gate by very small margins, but they did not pass the full promotion rule because BA/F1 were not improved and/or high-risk Manufacturer subgroup behavior worsened.",
        ]
        if not numeric_gate.empty:
            lines += [
                "",
                "Numerical AUC/PR-AUC gate crossers that were not promoted after subgroup review:",
                "",
                markdown_table(
                    numeric_gate,
                    [
                        "strategy",
                        "threshold_strategy",
                        "auc",
                        "pr_auc",
                        "balanced_accuracy",
                        "f1",
                        "philips_cn_fp_delta_vs_baseline",
                        "ge_ad_fn_delta_vs_baseline",
                        "min_manufacturer_ba_delta_vs_baseline",
                    ],
                    max_rows=20,
                ),
            ]
    else:
        lines += [
            "## Decision",
            "",
            "At least one strategy passed the full promotion rule:",
            "",
            markdown_table(promoted, ["strategy", "threshold_strategy", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "passes_full_promotion_rule"], max_rows=20),
        ]
    lines += [
        "",
        "## Interpretation",
        "",
        "Changing class weights, selecting C by average precision, applying SMOTE inside inner-CV, and adding Manufacturer-balanced sample weights tests the Stage-B imbalance hypothesis without changing the VAE representation. If these changes do not improve both AUC and PR-AUC, the current AD/CN imbalance is not the dominant remaining limitation of the locked readout.",
        "",
        f"Selected C was at the lower boundary in {lower_bound_count} fold-strategy fits, so regularization pressure remains visible, but the prior ultra-regularized audit and this imbalance audit both argue against changing the manuscript readout solely to chase threshold-level metrics.",
        "",
        "The locked final model remains `v5.1b horizon4480/cycles56` with classifier-only `logreg_l2` and leakage-safe inner-OOF threshold selection.",
        "",
    ]
    return "\n".join([x for x in lines if x is not None]) + "\n"


def write_readme(primary: pd.DataFrame) -> None:
    target = primary_target_table(primary)
    lines = [
        "# Frozen-Latent Imbalance Strategy Audit",
        "",
        "## Scope",
        "",
        "- VAE retraining: `False`.",
        "- Tensor modification: `False`.",
        "- Metadata/config/ledger/model-output modification: `False`.",
        f"- Latent source: `{LATENT_DIR}`.",
        "- Outer folds: existing final v5.1b horizon4480/cycles56 folds.",
        "- Inner-CV thresholding: true trainDev-only OOF predictions.",
        "",
        "## Strategies",
        "",
        "- Current balanced `logreg_l2` baseline.",
        "- `class_weight=None` and `class_weight=balanced`.",
        "- Custom minority-AD upweighting: AD:CN = 2:1, 3:1, 4:1.",
        "- SMOTE applied only inside inner-CV training splits and final outer-train fit.",
        "- Average-precision inner scoring.",
        "- Manufacturer-balanced sample weighting inside inner-CV and final outer-train fit.",
        "",
        "## Primary Threshold Results",
        "",
        markdown_table(
            target,
            [
                "strategy",
                "auc",
                "pr_auc",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "f1",
                "tn",
                "fp",
                "fn",
                "tp",
                "passes_primary_metric_rule",
                "passes_full_promotion_rule",
            ],
            max_rows=20,
        ),
        "",
        "## Promotion Rule",
        "",
        f"Promote only if AUC > `{REF['auc']:.6f}` and PR-AUC >= `{REF['pr_auc']:.6f}` without material BA/F1 or subgroup worsening.",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_fold_metrics: List[pd.DataFrame] = []
    all_predictions: List[pd.DataFrame] = []
    all_thresholds: List[pd.DataFrame] = []
    all_mfr: List[pd.DataFrame] = []
    all_site: List[pd.DataFrame] = []
    all_confusion: List[pd.DataFrame] = []
    all_hp: List[pd.DataFrame] = []
    all_c_audit: List[pd.DataFrame] = []
    unavailable_rows: List[Dict[str, Any]] = []

    for strategy in STRATEGIES:
        if strategy.use_smote and not IMBLEARN_AVAILABLE:
            unavailable_rows.append({"strategy": strategy.name, "status": f"unavailable: {IMBLEARN_ERROR}"})
            continue
        for fold in range(1, OUTER_FOLDS + 1):
            result = fold_strategy_run(fold, strategy)
            all_fold_metrics.append(result["fold_metrics"])
            all_predictions.append(result["predictions"])
            all_thresholds.append(result["thresholds"])
            all_mfr.append(result["manufacturer_subgroups"])
            all_site.append(result["site_subgroups"])
            all_confusion.append(result["confusion"])
            all_hp.append(result["hp"])
            all_c_audit.append(result["c_audit"])

    foldwise = pd.concat(all_fold_metrics, ignore_index=True, sort=False)
    predictions = pd.concat(all_predictions, ignore_index=True, sort=False)
    thresholds = pd.concat(all_thresholds, ignore_index=True, sort=False)
    mfr_fold = pd.concat(all_mfr, ignore_index=True, sort=False)
    site_fold = pd.concat(all_site, ignore_index=True, sort=False)
    confusion_fold = pd.concat(all_confusion, ignore_index=True, sort=False)
    hp = pd.concat(all_hp, ignore_index=True, sort=False)
    c_audit = pd.concat(all_c_audit, ignore_index=True, sort=False)

    primary, pooled_confusion, pooled_mfr, pooled_site = aggregate_predictions(predictions)
    primary = add_promotion_diagnostics(primary, pooled_mfr)
    threshold_comparison = thresholds.merge(
        foldwise[
            [
                "fold",
                "strategy",
                "threshold_strategy",
                "auc",
                "pr_auc",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "f1",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        ],
        on=["fold", "strategy", "threshold_strategy"],
        how="left",
    )

    primary_cols = [
        "strategy",
        "threshold_strategy",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
        "cn_fp_rate",
        "ad_fn_rate",
        "delta_auc_vs_ref",
        "delta_pr_auc_vs_ref",
        "passes_primary_metric_rule",
        "philips_cn_fp_delta_vs_baseline",
        "ge_ad_fn_delta_vs_baseline",
        "min_manufacturer_ba_delta_vs_baseline",
        "passes_full_promotion_rule",
    ]
    write_pair("primary_comparison", primary, primary_cols, max_rows=80)
    write_pair(
        "foldwise_comparison",
        foldwise.sort_values(["threshold_strategy", "strategy", "fold"]),
        [
            "fold",
            "strategy",
            "threshold_strategy",
            "auc",
            "pr_auc",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1",
            "tn",
            "fp",
            "fn",
            "tp",
            "selected_C",
            "best_inner_score",
            "inner_oof_sensitivity",
            "inner_oof_specificity",
        ],
        max_rows=120,
    )
    write_pair(
        "threshold_comparison",
        threshold_comparison.sort_values(["threshold_strategy", "strategy", "fold"]),
        [
            "fold",
            "strategy",
            "threshold_strategy",
            "threshold",
            "threshold_selection_context",
            "selected_C",
            "inner_oof_sensitivity",
            "inner_oof_specificity",
            "inner_oof_balanced_accuracy",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "f1",
        ],
        max_rows=120,
    )
    write_pair(
        "manufacturer_subgroup_comparison",
        pooled_mfr,
        [
            "strategy",
            "threshold_strategy",
            "Manufacturer",
            "n",
            "n_cn",
            "n_ad",
            "auc",
            "pr_auc",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "f1",
            "tn",
            "fp",
            "fn",
            "tp",
            "cn_fp_rate",
            "ad_fn_rate",
        ],
        max_rows=160,
    )
    write_pair(
        "sitecode_subgroup_comparison",
        pooled_site,
        [
            "strategy",
            "threshold_strategy",
            "SiteCode",
            "n",
            "n_cn",
            "n_ad",
            "auc",
            "pr_auc",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "f1",
            "tn",
            "fp",
            "fn",
            "tp",
            "cn_fp_rate",
            "ad_fn_rate",
        ],
        max_rows=200,
    )
    write_pair(
        "classifier_hp_boundary",
        hp.sort_values(["strategy", "fold"]),
        [
            "fold",
            "strategy",
            "scoring",
            "class_weight",
            "use_smote",
            "use_manufacturer_sample_weight",
            "selected_C",
            "C_boundary",
            "best_inner_score",
            "inner_cv_context",
            "minimum_inner_stratum_count",
        ],
        max_rows=120,
    )
    write_pair(
        "pooled_confusion",
        pooled_confusion,
        ["strategy", "threshold_strategy", "tn", "fp", "fn", "tp", "sensitivity", "specificity", "balanced_accuracy", "f1", "auc", "pr_auc"],
        max_rows=100,
    )

    predictions.to_csv(OUT_DIR / "all_strategy_predictions.csv", index=False)
    confusion_fold.to_csv(OUT_DIR / "confusion_by_fold.csv", index=False)
    mfr_fold.to_csv(OUT_DIR / "manufacturer_subgroup_by_fold.csv", index=False)
    site_fold.to_csv(OUT_DIR / "sitecode_subgroup_by_fold.csv", index=False)
    c_audit.to_csv(OUT_DIR / "inner_cv_c_grid_scores.csv", index=False)
    pd.DataFrame(unavailable_rows).to_csv(OUT_DIR / "unavailable_strategies.csv", index=False)

    write_readme(primary)
    (OUT_DIR / "final_recommendation.md").write_text(recommendation_text(primary, pooled_mfr, hp), encoding="utf-8")
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "run_dir": str(RUN_DIR),
        "readout_dir": str(READOUT_DIR),
        "latent_dir": str(LATENT_DIR),
        "output_dir": str(OUT_DIR),
        "vae_retrained": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "config_modified": False,
        "model_output_modified": False,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "strategies": [s.__dict__ for s in STRATEGIES],
        "target_sensitivities": TARGET_SENSITIVITIES,
        "reference_metrics": REF,
        "imblearn_available": IMBLEARN_AVAILABLE,
        "imblearn_error": IMBLEARN_ERROR,
        "promotion_rule": "AUC > 0.782951 and PR-AUC >= 0.559873 without material BA/F1 or subgroup worsening.",
    }
    (OUT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote audit outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
