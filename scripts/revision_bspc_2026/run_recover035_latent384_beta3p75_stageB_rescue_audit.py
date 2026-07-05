#!/usr/bin/env python3
"""Stage B rescue audit — recover035_latent384_beta3p75_T80_h10000_p560_full5x5.

Goal: determine whether pooled AUC=0.760 / PR-AUC=0.509 (vs locked 0.783/0.560) is caused by:
  1. Coarse Stage B logreg C grid (Fold 1 selects C=0.1, Folds 2-5 select C=0.001 → score-range mismatch)
  2. Score-scale mismatch (Fold 1 range=0.995 vs Folds 2-5 range≈0.40)
  3. Insufficient readout flexibility (feature set, classifier type)

Readout models:
  logreg_l2_original   C in [0.001, 0.01, 0.1, 1.0]                     (replica of existing readout)
  logreg_l2_dense      C in DENSE_C_GRID (18 values, 1e-5 ... 1e-1)      (fine sweep near C=0.001..0.1)
  svm_rbf              C/gamma GridSearch, Platt calibration             (extended RBF grid)
  logreg_elasticnet    C + l1_ratio GridSearch                           (mixed L1/L2)

Feature sets:
  z_only             latent mu only (no Age/Sex)
  z_plus_age_sex     latent mu + Age + Sex (current behaviour)

Threshold strategies: fixed_0p5, inner_oof_youden_j, inner_oof_balanced_accuracy,
                      inner_oof_target_sens_ge_0p70_max_spec

Extra diagnostics:
  - Score range by fold (min/p10/p25/med/p75/p90/max by diagnosis)
  - Pooled vs foldwise AUC/PR-AUC comparison table
  - Rank-normalized descriptive metrics (DESCRIPTIVE ONLY — labelled explicitly)
  - FP/FN by Manufacturer, Sex, Age quartile
  - Precision at top-k AD-risk subjects (k=5,10,15,20)
  - Selected hyperparameters by fold
  - Confusion matrices by fold

Promotion criteria (pooled, logreg_l2_dense, z_plus_age_sex, primary threshold):
  AUC > 0.782951 AND PR-AUC >= 0.559873

Hard constraints:
  - No VAE retraining.
  - No threshold fitting on outer test data.
  - Rank-normalised metrics are DESCRIPTIVE ONLY.
  - Do not modify tensors, metadata, ledger, or existing run outputs.
  - 128_S_2002 must not appear (already absent from latent cache).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"

DEFAULT_RUN_DIR = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
DEFAULT_OUTPUT_DIR = RESULTS / "recover035_latent384_beta3p75_stageB_rescue_audit"

LOCKED_AUC = 0.782951
LOCKED_PR_AUC = 0.559873
EXISTING_POOLED_AUC = 0.760
EXISTING_POOLED_PR_AUC = 0.50914
EXISTING_FOLD1_RANGE = 0.9952
EXISTING_FOLDS25_RANGE_MEAN = 0.40  # approx mean of folds 2-5

# Dense C grid: covers the gap between 0.001 and 0.1 with 18 values
DENSE_C_GRID = [
    1e-5, 2e-5, 3e-5, 5e-5, 7e-5,
    1e-4, 2e-4, 3e-4, 5e-4, 7e-4,
    1e-3, 2e-3, 3e-3, 5e-3, 7e-3,
    1e-2, 3e-2, 1e-1,
]

ORIGINAL_C_GRID = [0.001, 0.01, 0.1, 1.0]

TARGET_SENSITIVITY = 0.70
FOLDS = [1, 2, 3, 4, 5]
SEED = 42

TOP_K_LIST = [5, 10, 15, 20]

MODELS = ["logreg_l2_original", "logreg_l2_dense", "svm_rbf", "logreg_elasticnet"]
FEATURE_SETS = ["z_only", "z_plus_age_sex"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--folds", type=int, nargs="*", default=FOLDS)
    p.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=MODELS,
    )
    p.add_argument(
        "--feature-sets",
        nargs="+",
        choices=FEATURE_SETS,
        default=FEATURE_SETS,
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Validate paths without running classifiers.")
    return p.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def md_table(df: pd.DataFrame, max_rows: int = 60) -> str:
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


def write_table(outdir: Path, stem: str, df: pd.DataFrame, max_rows: int = 60) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    y_pred: Sequence[int],
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: Dict[str, float] = {
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
        "predicted_ad_rate": float(pred.mean()) if len(pred) else float("nan"),
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


def select_thresholds(y_true: Sequence[int], y_score: Sequence[float]) -> List[Dict[str, Any]]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    rows: List[Dict[str, Any]] = []
    for thr in threshold_candidates(s):
        pred = (s >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)
        ba = float(np.nanmean([sens, spec]))
        rows.append({
            "threshold": float(thr),
            "sensitivity": sens,
            "specificity": spec,
            "balanced_accuracy": ba,
            "youden_j": sens + spec - 1.0,
        })
    tbl = pd.DataFrame(rows)

    selections: List[Dict[str, Any]] = []
    for criterion, metric in [("inner_oof_youden_j", "youden_j"), ("inner_oof_balanced_accuracy", "balanced_accuracy")]:
        r = tbl.sort_values([metric, "sensitivity", "specificity", "threshold"], ascending=[False, False, False, False]).iloc[0]
        selections.append({
            "threshold_strategy": criterion,
            "threshold": float(r["threshold"]),
            "selection_metric": metric,
            "inner_oof_sensitivity": float(r["sensitivity"]),
            "inner_oof_specificity": float(r["specificity"]),
            "inner_oof_balanced_accuracy": float(r["balanced_accuracy"]),
        })

    eligible = tbl[tbl["sensitivity"] >= TARGET_SENSITIVITY]
    if eligible.empty:
        r = tbl.sort_values(["sensitivity", "specificity", "threshold"], ascending=[False, False, False]).iloc[0]
        status = "target_not_reached_inner_oof"
    else:
        r = eligible.sort_values(["specificity", "sensitivity", "balanced_accuracy", "threshold"], ascending=[False, False, False, False]).iloc[0]
        status = "selected_inner_oof"
    selections.append({
        "threshold_strategy": "inner_oof_target_sens_ge_0p70_max_spec",
        "threshold": float(r["threshold"]),
        "selection_metric": status,
        "inner_oof_sensitivity": float(r["sensitivity"]),
        "inner_oof_specificity": float(r["specificity"]),
        "inner_oof_balanced_accuracy": float(r["balanced_accuracy"]),
    })
    selections.append({
        "threshold_strategy": "fixed_0p5",
        "threshold": 0.5,
        "selection_metric": "fixed_no_selection",
        "inner_oof_sensitivity": float("nan"),
        "inner_oof_specificity": float("nan"),
        "inner_oof_balanced_accuracy": float("nan"),
    })
    return selections


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(mu_cols: List[str], include_age_sex: bool) -> ColumnTransformer:
    numeric_latent = Pipeline([("scaler", StandardScaler())])
    transformers: List[Tuple[str, Any, List[str]]] = [("latent", numeric_latent, mu_cols)]
    if include_age_sex:
        numeric_age = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_ohe())])
        transformers.extend([("age", numeric_age, ["Age"]), ("sex", categorical, ["Sex"])])
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def rescue_classifier_specs(fold: int) -> Dict[str, Tuple[Pipeline, Dict[str, List[Any]], str]]:
    seed = SEED + fold
    specs: Dict[str, Tuple[Pipeline, Dict[str, List[Any]], str]] = {
        "logreg_l2_original": (
            Pipeline([
                ("pre", "passthrough"),
                ("model", LogisticRegression(
                    penalty="l2", solver="lbfgs", class_weight="balanced",
                    max_iter=5000, random_state=seed,
                )),
            ]),
            {"model__C": ORIGINAL_C_GRID},
            "available",
        ),
        "logreg_l2_dense": (
            Pipeline([
                ("pre", "passthrough"),
                ("model", LogisticRegression(
                    penalty="l2", solver="lbfgs", class_weight="balanced",
                    max_iter=5000, random_state=seed,
                )),
            ]),
            {"model__C": DENSE_C_GRID},
            "available",
        ),
        "svm_rbf": (
            Pipeline([
                ("pre", "passthrough"),
                ("model", SVC(
                    kernel="rbf", probability=True, class_weight="balanced", random_state=seed,
                )),
            ]),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__gamma": ["scale", "auto", 1e-4, 1e-3, 1e-2],
            },
            "available",
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
                "model__C": [1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
            "available",
        ),
    }
    return specs


def score_1d(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(x)[:, 1], dtype=float)
    raw = np.asarray(estimator.decision_function(x), dtype=float).ravel()
    return 1.0 / (1.0 + np.exp(-raw))


def inner_stratification_key(df: pd.DataFrame, n_splits: int) -> Tuple[pd.Series, str, int]:
    cols = ["ResearchGroup_Mapped", "Manufacturer"]
    key_df = df[cols].copy()
    for col in cols:
        key_df[col] = key_df[col].fillna(f"{col}_UNKNOWN").astype(str)
    key = key_df.apply(lambda r: "_".join(r.values.astype(str)), axis=1)
    min_count = int(key.value_counts().min())
    if min_count < n_splits:
        key = df["y"].astype(int)
        return key, "label_only_fallback", int(pd.Series(key).value_counts().min())
    return key, "ResearchGroup_Mapped+Manufacturer", min_count


def load_latent_pair(cache_dir: Path, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(cache_dir / f"fold_{fold}_trainDev_latent_mu.csv")
    test = pd.read_csv(cache_dir / f"fold_{fold}_test_latent_mu.csv")
    return train, test


def run_rescue_sweep(
    cache_dir: Path,
    folds: List[int],
    models: List[str],
    feature_sets: List[str],
    n_jobs: int,
    inner_folds: int = 5,
) -> Dict[str, pd.DataFrame]:
    fold_metric_rows: List[Dict[str, Any]] = []
    pred_rows: List[pd.DataFrame] = []
    confusion_rows: List[Dict[str, Any]] = []
    subgroup_rows: List[Dict[str, Any]] = []
    model_status_rows: List[Dict[str, Any]] = []

    for fold in folds:
        print(f"  Fold {fold} ...", flush=True)
        train_df, test_df = load_latent_pair(cache_dir, fold)
        mu_cols = [c for c in train_df.columns if c.startswith("mu_")]
        y_train = train_df["y"].astype(int).to_numpy()
        y_test = test_df["y"].astype(int).to_numpy()

        inner_key, inner_context, min_inner_cell = inner_stratification_key(train_df, n_splits=inner_folds)
        inner_cv = list(StratifiedKFold(
            n_splits=inner_folds, shuffle=True, random_state=SEED + fold + 30,
        ).split(np.zeros(len(train_df)), inner_key))
        specs = rescue_classifier_specs(fold)

        for feature_set in feature_sets:
            include_age_sex = feature_set == "z_plus_age_sex"
            feature_cols = mu_cols + (["Age", "Sex"] if include_age_sex else [])
            x_train = train_df[feature_cols].copy()
            x_test = test_df[feature_cols].copy()
            pre = make_preprocessor(mu_cols, include_age_sex=include_age_sex)

            for model_name in models:
                base_pipe, grid, status = specs.get(model_name, (None, None, "not_requested"))
                if status != "available":
                    model_status_rows.append({
                        "fold": fold, "model_name": model_name,
                        "feature_set": feature_set, "status": status,
                    })
                    continue

                pipe = clone(base_pipe)
                pipe.steps[0] = ("pre", pre)
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
                oof_score = cross_val_predict(
                    clone(best), x_train, y_train,
                    cv=inner_cv, method="predict_proba", n_jobs=n_jobs,
                )[:, 1]
                test_score = score_1d(best, x_test)
                thresholds = select_thresholds(y_train, oof_score)

                model_status_rows.append({
                    "fold": fold,
                    "model_name": model_name,
                    "feature_set": feature_set,
                    "status": "fit_ok",
                    "best_params": json.dumps(search.best_params_, sort_keys=True),
                    "best_inner_auc": float(search.best_score_),
                    "inner_cv_context": inner_context,
                    "minimum_inner_stratum_count": int(min_inner_cell),
                })

                for sel in thresholds:
                    thr = float(sel["threshold"])
                    y_pred = (test_score >= thr).astype(int)
                    row: Dict[str, Any] = {
                        "fold": fold,
                        "model_name": model_name,
                        "feature_set": feature_set,
                        "threshold_strategy": sel["threshold_strategy"],
                        "threshold": thr,
                        "threshold_selection_context": (
                            "true_inner_cv_oof" if sel["threshold_strategy"] != "fixed_0p5" else "fixed_no_selection"
                        ),
                        "inner_cv_context": inner_context,
                        "minimum_inner_stratum_count": int(min_inner_cell),
                        "best_inner_auc": float(search.best_score_),
                        "best_params": json.dumps(search.best_params_, sort_keys=True),
                    }
                    row.update(sel)
                    row.update(binary_metrics(y_test, test_score, y_pred))
                    fold_metric_rows.append(row)
                    confusion_rows.append({
                        k: row[k]
                        for k in [
                            "fold", "model_name", "feature_set", "threshold_strategy",
                            "threshold", "n", "n_cn", "n_ad", "tn", "fp", "fn", "tp",
                            "sensitivity", "specificity", "balanced_accuracy", "f1",
                        ]
                    })

                    pred = test_df[[
                        "SubjectID", "tensor_idx", "ResearchGroup_Mapped",
                        "Manufacturer", "Age", "Sex",
                    ]].copy()
                    pred["fold"] = fold
                    pred["model_name"] = model_name
                    pred["feature_set"] = feature_set
                    pred["threshold_strategy"] = sel["threshold_strategy"]
                    pred["threshold"] = thr
                    pred["y_true"] = y_test
                    pred["y_score"] = test_score
                    pred["y_pred"] = y_pred
                    pred_rows.append(pred)

                    for mfr, sub_idx in pred.groupby("Manufacturer", dropna=False).groups.items():
                        sub_pred = pred.loc[list(sub_idx)]
                        sub_row: Dict[str, Any] = {
                            "fold": fold, "model_name": model_name,
                            "feature_set": feature_set,
                            "threshold_strategy": sel["threshold_strategy"],
                            "threshold": thr, "Manufacturer": mfr,
                        }
                        sub_row.update(binary_metrics(
                            sub_pred["y_true"], sub_pred["y_score"], sub_pred["y_pred"],
                        ))
                        subgroup_rows.append(sub_row)

    return {
        "foldwise_metrics": pd.DataFrame(fold_metric_rows),
        "predictions": pd.concat(pred_rows, ignore_index=True, sort=False),
        "confusion_by_fold": pd.DataFrame(confusion_rows),
        "subgroup_by_manufacturer": pd.DataFrame(subgroup_rows),
        "model_status": pd.DataFrame(model_status_rows),
    }


def pooled_from_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["model_name", "feature_set", "threshold_strategy"]
    for (model, fs, strategy), sub in pred.groupby(group_cols, dropna=False):
        row: Dict[str, Any] = {
            "model_name": model,
            "feature_set": fs,
            "threshold_strategy": strategy,
            "threshold": "fold_specific" if strategy != "fixed_0p5" else 0.5,
        }
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["threshold_strategy", "feature_set", "model_name"])


def compute_score_range_by_fold(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (model, fs, strategy, fold), sub in pred.groupby(
        ["model_name", "feature_set", "threshold_strategy", "fold"], dropna=False,
    ):
        scores = sub["y_score"].values
        for dx_code, dx_name in [(0, "CN"), (1, "AD"), (-1, "ALL")]:
            if dx_code == -1:
                s = scores
            else:
                s = sub[sub["y_true"] == dx_code]["y_score"].values
            if len(s) == 0:
                continue
            rows.append({
                "model_name": model,
                "feature_set": fs,
                "threshold_strategy": strategy,
                "fold": fold,
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
                "iqr": float(np.percentile(s, 75) - np.percentile(s, 25)),
            })
    return pd.DataFrame(rows)


def compute_pooled_vs_foldwise(pred: pd.DataFrame, foldwise: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["model_name", "feature_set", "threshold_strategy"]

    for (model, fs, strategy), sub_pred in pred.groupby(group_cols, dropna=False):
        pooled_auc = float(roc_auc_score(sub_pred["y_true"], sub_pred["y_score"])) if len(np.unique(sub_pred["y_true"])) == 2 else float("nan")
        pooled_pr = float(average_precision_score(sub_pred["y_true"], sub_pred["y_score"])) if len(np.unique(sub_pred["y_true"])) == 2 else float("nan")

        sub_fw = foldwise[
            (foldwise["model_name"] == model) &
            (foldwise["feature_set"] == fs) &
            (foldwise["threshold_strategy"] == strategy)
        ]
        fw_auc_mean = float(sub_fw["auc"].mean()) if not sub_fw.empty else float("nan")
        fw_pr_mean = float(sub_fw["pr_auc"].mean()) if not sub_fw.empty else float("nan")

        # Rank-normalised: within-fold percentile ranks (DESCRIPTIVE ONLY)
        rn_parts: List[pd.DataFrame] = []
        for fold, fsub in sub_pred.groupby("fold"):
            fsub2 = fsub.copy()
            n = len(fsub2)
            fsub2["y_score_ranknorm"] = fsub2["y_score"].rank(method="average").subtract(0.5).divide(n)
            rn_parts.append(fsub2)
        rn_all = pd.concat(rn_parts, ignore_index=True)
        rn_auc = float(roc_auc_score(rn_all["y_true"], rn_all["y_score_ranknorm"])) if len(np.unique(rn_all["y_true"])) == 2 else float("nan")
        rn_pr = float(average_precision_score(rn_all["y_true"], rn_all["y_score_ranknorm"])) if len(np.unique(rn_all["y_true"])) == 2 else float("nan")

        rows.append({
            "model_name": model,
            "feature_set": fs,
            "threshold_strategy": strategy,
            "pooled_auc": pooled_auc,
            "pooled_pr_auc": pooled_pr,
            "foldwise_mean_auc": fw_auc_mean,
            "foldwise_mean_pr_auc": fw_pr_mean,
            "gap_auc_fw_minus_pooled": fw_auc_mean - pooled_auc,
            "gap_pr_fw_minus_pooled": fw_pr_mean - pooled_pr,
            "ranknorm_auc_DESCRIPTIVE_ONLY": rn_auc,
            "ranknorm_pr_auc_DESCRIPTIVE_ONLY": rn_pr,
            "locked_auc_threshold": LOCKED_AUC,
            "locked_pr_auc_threshold": LOCKED_PR_AUC,
            "promotes_vs_locked": bool(pooled_auc > LOCKED_AUC and pooled_pr >= LOCKED_PR_AUC),
        })
    return pd.DataFrame(rows)


def compute_fp_fn_by_subgroup(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["model_name", "feature_set", "threshold_strategy"]
    for (model, fs, strategy), sub in pred.groupby(group_cols, dropna=False):
        for subgroup_col in ["Manufacturer", "Sex"]:
            if subgroup_col not in sub.columns:
                continue
            for group_val, gsub in sub.groupby(subgroup_col, dropna=False):
                cn = gsub[gsub["y_true"] == 0]
                ad = gsub[gsub["y_true"] == 1]
                rows.append({
                    "model_name": model, "feature_set": fs,
                    "threshold_strategy": strategy,
                    "subgroup_col": subgroup_col, "subgroup_val": str(group_val),
                    "n_cn": len(cn), "n_ad": len(ad),
                    "fp_cn": int((cn["y_pred"] == 1).sum()),
                    "fn_ad": int((ad["y_pred"] == 0).sum()),
                    "fpr_cn": safe_div((cn["y_pred"] == 1).sum(), len(cn)),
                    "fnr_ad": safe_div((ad["y_pred"] == 0).sum(), len(ad)),
                    "specificity_cn": safe_div((cn["y_pred"] == 0).sum(), len(cn)),
                    "sensitivity_ad": safe_div((ad["y_pred"] == 1).sum(), len(ad)),
                })

        # Age quartile subgroup
        if "Age" in sub.columns:
            ages = pd.to_numeric(sub["Age"], errors="coerce")
            q25, q50, q75 = np.nanpercentile(ages, [25, 50, 75])
            age_bins = pd.cut(
                ages,
                bins=[-np.inf, q25, q50, q75, np.inf],
                labels=["Q1_youngest", "Q2", "Q3", "Q4_oldest"],
            )
            for age_group, gsub in sub.groupby(age_bins, observed=True):
                cn = gsub[gsub["y_true"] == 0]
                ad = gsub[gsub["y_true"] == 1]
                rows.append({
                    "model_name": model, "feature_set": fs,
                    "threshold_strategy": strategy,
                    "subgroup_col": "Age_quartile", "subgroup_val": str(age_group),
                    "n_cn": len(cn), "n_ad": len(ad),
                    "fp_cn": int((cn["y_pred"] == 1).sum()),
                    "fn_ad": int((ad["y_pred"] == 0).sum()),
                    "fpr_cn": safe_div((cn["y_pred"] == 1).sum(), len(cn)),
                    "fnr_ad": safe_div((ad["y_pred"] == 0).sum(), len(ad)),
                    "specificity_cn": safe_div((cn["y_pred"] == 0).sum(), len(cn)),
                    "sensitivity_ad": safe_div((ad["y_pred"] == 1).sum(), len(ad)),
                })

    return pd.DataFrame(rows)


def compute_precision_at_topk(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["model_name", "feature_set", "threshold_strategy"]
    for (model, fs, strategy), sub in pred.groupby(group_cols, dropna=False):
        sub_sorted = sub.sort_values("y_score", ascending=False).reset_index(drop=True)
        for k in TOP_K_LIST:
            topk = sub_sorted.head(k)
            prec = safe_div(int((topk["y_true"] == 1).sum()), k)
            rows.append({
                "model_name": model, "feature_set": fs,
                "threshold_strategy": strategy,
                "k": k,
                "precision_at_k": prec,
                "n_ad_in_topk": int((topk["y_true"] == 1).sum()),
                "n_cn_in_topk": int((topk["y_true"] == 0).sum()),
            })
    return pd.DataFrame(rows)


def compute_fold1_diagnosis(
    pred: pd.DataFrame,
    foldwise: pd.DataFrame,
    score_range: pd.DataFrame,
) -> str:
    lines: List[str] = [
        "## Fold 1 Score-Range Anomaly Diagnosis",
        "",
        "### Baseline (existing logreg_l2_original, z_plus_age_sex, primary threshold)",
        f"- Fold 1 range: {EXISTING_FOLD1_RANGE:.4f}  (selected C=0.1)",
        f"- Folds 2-5 mean range: {EXISTING_FOLDS25_RANGE_MEAN:.4f}  (selected C=0.001)",
        f"- Existing pooled AUC: {EXISTING_POOLED_AUC:.4f}  (locked target: {LOCKED_AUC})",
        f"- Existing pooled PR-AUC: {EXISTING_POOLED_PR_AUC:.5f}  (locked target: {LOCKED_PR_AUC})",
        "",
        "### Rescue Results by Model",
    ]

    for model in ["logreg_l2_original", "logreg_l2_dense"]:
        thr = "inner_oof_target_sens_ge_0p70_max_spec"
        fs = "z_plus_age_sex"
        sr_all = score_range[
            (score_range["model_name"] == model) &
            (score_range["feature_set"] == fs) &
            (score_range["threshold_strategy"] == thr) &
            (score_range["diagnosis"] == "ALL")
        ]
        fw_sub = foldwise[
            (foldwise["model_name"] == model) &
            (foldwise["feature_set"] == fs) &
            (foldwise["threshold_strategy"] == thr)
        ]
        lines.append(f"\n#### {model} | {fs} | {thr}")
        if not fw_sub.empty:
            for fold in sorted(fw_sub["fold"].unique()):
                fr = fw_sub[fw_sub["fold"] == fold].iloc[0]
                sr_f = sr_all[sr_all["fold"] == fold]
                rng = float(sr_f["score_range"].iloc[0]) if not sr_f.empty else float("nan")
                params = fr.get("best_params", "?")
                lines.append(
                    f"  Fold {int(fold)}: AUC={fr['auc']:.4f} PR-AUC={fr['pr_auc']:.4f}"
                    f"  range={rng:.4f}  params={params}"
                )

    return "\n".join(lines)


def write_final_report(
    outdir: Path,
    pooled: pd.DataFrame,
    pooled_vs_fw: pd.DataFrame,
    fp_fn: pd.DataFrame,
    topk: pd.DataFrame,
    fold1_diagnosis: str,
    model_status: pd.DataFrame,
) -> None:
    now = datetime.now(timezone.utc).isoformat()

    # Find best result
    primary_mask = (
        (pooled["model_name"] == "logreg_l2_dense") &
        (pooled["feature_set"] == "z_plus_age_sex") &
        (pooled["threshold_strategy"] == "inner_oof_target_sens_ge_0p70_max_spec")
    )
    primary = pooled[primary_mask]
    if not primary.empty:
        best_auc = float(primary.iloc[0]["auc"])
        best_pr = float(primary.iloc[0]["pr_auc"])
        best_ba = float(primary.iloc[0]["balanced_accuracy"])
        best_f1 = float(primary.iloc[0]["f1"])
    else:
        best_auc = best_pr = best_ba = best_f1 = float("nan")

    promotes = best_auc > LOCKED_AUC and best_pr >= LOCKED_PR_AUC
    verdict = "PROMOTES" if promotes else "DOES NOT PROMOTE"

    # Best AUC across all models
    auc_col = "auc" if "auc" in pooled.columns else None
    best_overall_row = pooled.sort_values("auc", ascending=False).iloc[0] if auc_col and not pooled.empty else None

    lines = [
        "# Stage B Rescue Audit Report",
        "## recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        "",
        f"Generated: {now}",
        "",
        "## Scope",
        "Read-only classifier sweep on existing latent cache.",
        "No VAE retraining. No threshold fitting on outer test. Rank-norm metrics DESCRIPTIVE ONLY.",
        "",
        "## Promotion Criteria",
        f"- pooled AUC > {LOCKED_AUC} AND pooled PR-AUC >= {LOCKED_PR_AUC} (both must pass)",
        "",
        "## Primary Result (logreg_l2_dense, z_plus_age_sex, primary threshold)",
        f"- Pooled AUC:  {best_auc:.4f}  ({'PASS' if best_auc > LOCKED_AUC else 'FAIL'} vs locked {LOCKED_AUC})",
        f"- Pooled PR-AUC: {best_pr:.4f}  ({'PASS' if best_pr >= LOCKED_PR_AUC else 'FAIL'} vs locked {LOCKED_PR_AUC})",
        f"- BA: {best_ba:.4f}   F1: {best_f1:.4f}",
        f"- **Verdict: {verdict}**",
        "",
        "## Baseline Comparison",
        f"- Existing readout pooled AUC: {EXISTING_POOLED_AUC:.4f} / PR-AUC: {EXISTING_POOLED_PR_AUC:.5f}",
        f"- Delta (dense vs original): AUC = {best_auc - EXISTING_POOLED_AUC:+.4f}  PR-AUC = {best_pr - EXISTING_POOLED_PR_AUC:+.5f}",
    ]

    if best_overall_row is not None and best_overall_row["model_name"] != "logreg_l2_dense":
        lines += [
            "",
            f"- Best AUC across all models: {best_overall_row['model_name']} ({best_overall_row['feature_set']}, {best_overall_row['threshold_strategy']}) "
            f"AUC={best_overall_row['auc']:.4f} PR-AUC={best_overall_row['pr_auc']:.4f}",
        ]

    lines += [
        "",
        "## Pooled Metrics — All Models (primary threshold, z_plus_age_sex)",
    ]
    primary_thresh_mask = pooled["threshold_strategy"] == "inner_oof_target_sens_ge_0p70_max_spec"
    z_mask = pooled["feature_set"] == "z_plus_age_sex"
    sub = pooled[primary_thresh_mask & z_mask].sort_values("auc", ascending=False)
    if not sub.empty:
        show_cols = [c for c in ["model_name", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"] if c in sub.columns]
        lines.append(sub[show_cols].to_markdown(index=False))

    lines += [
        "",
        "## Pooled vs Foldwise AUC Gap (z_plus_age_sex, primary threshold)",
    ]
    pvf_mask = (
        (pooled_vs_fw["feature_set"] == "z_plus_age_sex") &
        (pooled_vs_fw["threshold_strategy"] == "inner_oof_target_sens_ge_0p70_max_spec")
    )
    pvf_sub = pooled_vs_fw[pvf_mask].sort_values("pooled_auc", ascending=False)
    if not pvf_sub.empty:
        show_pvf = [c for c in [
            "model_name", "pooled_auc", "pooled_pr_auc", "foldwise_mean_auc", "foldwise_mean_pr_auc",
            "gap_auc_fw_minus_pooled", "ranknorm_auc_DESCRIPTIVE_ONLY", "promotes_vs_locked",
        ] if c in pvf_sub.columns]
        lines.append(pvf_sub[show_pvf].to_markdown(index=False))
    lines += ["", "_(Rank-norm metrics are DESCRIPTIVE ONLY — not derived from inner-CV.)_"]

    lines += [
        "",
        fold1_diagnosis,
        "",
        "## Philips CN False-Positive Check (pooled, all models, primary threshold)",
    ]
    philips_cn_mask = (
        (fp_fn["subgroup_col"] == "Manufacturer") &
        (fp_fn["subgroup_val"].str.upper() == "PHILIPS") &
        (fp_fn["threshold_strategy"] == "inner_oof_target_sens_ge_0p70_max_spec") &
        (fp_fn["feature_set"] == "z_plus_age_sex")
    )
    philips_cn = fp_fn[philips_cn_mask]
    if not philips_cn.empty:
        show_fp = [c for c in ["model_name", "n_cn", "fp_cn", "fpr_cn", "specificity_cn", "n_ad", "sensitivity_ad"] if c in philips_cn.columns]
        lines.append(philips_cn[show_fp].to_markdown(index=False))

    lines += [
        "",
        "## Precision at Top-k AD-Risk (primary threshold, z_plus_age_sex)",
    ]
    topk_mask = (
        (topk["threshold_strategy"] == "inner_oof_target_sens_ge_0p70_max_spec") &
        (topk["feature_set"] == "z_plus_age_sex")
    )
    topk_sub = topk[topk_mask].pivot_table(
        index="model_name", columns="k", values="precision_at_k",
    ).reset_index()
    if not topk_sub.empty:
        lines.append(topk_sub.to_markdown(index=False))

    lines += [
        "",
        "## Read-only guarantee",
        "Did not retrain VAE. Did not fit thresholds on outer test data.",
        "Did not modify tensors, metadata, ledger, or existing run outputs.",
    ]

    (outdir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    outdir = resolve(args.output_dir)
    cache_dir = run_dir / "classifier_only_readout" / "latent_cache"

    print("Stage B rescue audit: beta3p75 latent384")
    print(f"  run_dir:   {run_dir.name}  ({'exists' if run_dir.exists() else 'NOT FOUND'})")
    print(f"  cache_dir: {'exists' if cache_dir.exists() else 'NOT FOUND'}  ({cache_dir})")
    for fold in args.folds:
        for split in ["trainDev", "test"]:
            p = cache_dir / f"fold_{fold}_{split}_latent_mu.csv"
            status = "ok" if p.exists() else "MISSING"
            print(f"    fold_{fold}_{split}: {status}")
    print(f"  models:    {args.models}")
    print(f"  feature_sets: {args.feature_sets}")
    print(f"  output:    {outdir}")

    if args.dry_run:
        missing_folds = [
            f"fold_{fold}_{split}_latent_mu.csv"
            for fold in args.folds
            for split in ["trainDev", "test"]
            if not (cache_dir / f"fold_{fold}_{split}_latent_mu.csv").exists()
        ]
        if missing_folds:
            print(f"  [DRY-RUN] MISSING cache files: {missing_folds}")
            return 1
        print("  [DRY-RUN] All cache files present. No classifiers run.")
        return 0

    if not cache_dir.exists():
        print(f"ERROR: latent cache not found: {cache_dir}")
        return 1

    if outdir.exists() and not args.overwrite:
        print(f"Output dir exists; pass --overwrite: {outdir}")
        return 1
    outdir.mkdir(parents=True, exist_ok=True)

    print("\nRunning rescue sweep ...")
    sweep = run_rescue_sweep(
        cache_dir=cache_dir,
        folds=sorted(args.folds),
        models=args.models,
        feature_sets=args.feature_sets,
        n_jobs=args.n_jobs,
        inner_folds=5,
    )

    foldwise = sweep["foldwise_metrics"].sort_values(["model_name", "feature_set", "threshold_strategy", "fold"])
    pred = sweep["predictions"]
    pooled = pooled_from_predictions(pred)
    score_range = compute_score_range_by_fold(pred)
    pooled_vs_fw = compute_pooled_vs_foldwise(pred, foldwise)
    fp_fn = compute_fp_fn_by_subgroup(pred)
    topk = compute_precision_at_topk(pred)
    fold1_diag = compute_fold1_diagnosis(pred, foldwise, score_range)

    write_table(outdir, "rescue_foldwise_metrics", foldwise, max_rows=200)
    write_table(outdir, "rescue_pooled_metrics", pooled, max_rows=100)
    write_table(outdir, "rescue_confusion_by_fold", sweep["confusion_by_fold"], max_rows=200)
    write_table(outdir, "rescue_score_range_by_fold", score_range, max_rows=300)
    write_table(outdir, "rescue_pooled_vs_foldwise", pooled_vs_fw, max_rows=100)
    write_table(outdir, "rescue_fp_fn_by_subgroup", fp_fn, max_rows=200)
    write_table(outdir, "rescue_precision_at_topk", topk, max_rows=100)
    write_table(outdir, "rescue_subgroup_by_manufacturer", sweep["subgroup_by_manufacturer"], max_rows=300)
    pred.to_csv(outdir / "rescue_predictions.csv", index=False)
    sweep["model_status"].to_csv(outdir / "rescue_model_status.csv", index=False)

    write_final_report(outdir, pooled, pooled_vs_fw, fp_fn, topk, fold1_diag, sweep["model_status"])

    now = datetime.now(timezone.utc).isoformat()
    command_log = {
        "created_utc": now,
        "script": str(Path(__file__).resolve()),
        "run_dir": str(run_dir),
        "cache_dir": str(cache_dir),
        "output_dir": str(outdir),
        "models": args.models,
        "feature_sets": args.feature_sets,
        "folds": sorted(args.folds),
        "n_jobs": args.n_jobs,
        "locked_auc": LOCKED_AUC,
        "locked_pr_auc": LOCKED_PR_AUC,
        "existing_pooled_auc": EXISTING_POOLED_AUC,
        "existing_pooled_pr_auc": EXISTING_POOLED_PR_AUC,
        "vae_retrained": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "threshold_selection": "true_inner_cv_oof_for_selected_hyperparameters",
        "ranknorm_metrics_are_descriptive_only": True,
    }
    write_json(outdir / "command_log.json", command_log)

    # Summary to stdout
    print(f"\n{'='*60}")
    print("RESCUE SWEEP SUMMARY")
    print(f"{'='*60}")
    primary = pooled[
        (pooled["model_name"] == "logreg_l2_dense") &
        (pooled["feature_set"] == "z_plus_age_sex") &
        (pooled["threshold_strategy"] == "inner_oof_target_sens_ge_0p70_max_spec")
    ]
    if not primary.empty:
        r = primary.iloc[0]
        print(f"  logreg_l2_dense z+age/sex primary: AUC={r['auc']:.4f} PR-AUC={r['pr_auc']:.4f} BA={r['balanced_accuracy']:.4f} F1={r['f1']:.4f}")
        promotes = r["auc"] > LOCKED_AUC and r["pr_auc"] >= LOCKED_PR_AUC
        print(f"  Promotion: {'PROMOTES' if promotes else 'DOES NOT PROMOTE'}")
    print(f"\nAll pooled results (primary threshold, z_plus_age_sex):")
    pth = pooled[
        (pooled["threshold_strategy"] == "inner_oof_target_sens_ge_0p70_max_spec") &
        (pooled["feature_set"] == "z_plus_age_sex")
    ].sort_values("auc", ascending=False)
    cols = [c for c in ["model_name", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"] if c in pth.columns]
    print(pth[cols].to_string(index=False))
    print(f"\nOutput: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
