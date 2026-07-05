#!/usr/bin/env python3
"""Rescue readout for recover035_latent384_T80_h10000_p560_full5x5.

Read-only: reuses existing latent caches from classifier_only_readout/latent_cache/.
No VAE retraining. No threshold leakage. No OASIS use.

Classifiers
-----------
logreg_l2_original : C ∈ {0.001, 0.01, 0.1, 1.0}   — reproduce original Stage B
logreg_l2_dense    : C ∈ 14-value dense grid [1e-5 … 1e-2]
svm_rbf            : C × gamma grid chosen by inner CV only

Feature set: z_plus_age_sex
Calibration: Platt-sigmoid; uncalibrated and calibrated reported separately
Thresholds: fixed_0p5 | inner_oof_youden_j | inner_oof_balanced_accuracy
            | inner_oof_target_sens_ge_0p70_max_spec

Output
------
results/revision_bspc_2026/
  recover035_latent384_T80_h10000_p560_full5x5_rescue_readout/
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/recover035_latent384_T80_h10000_p560_full5x5"
)
LATENT_CACHE_DIR = RUN_DIR / "classifier_only_readout" / "latent_cache"
RESCUE_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026"
    / "recover035_latent384_T80_h10000_p560_full5x5_rescue_readout"
)

# ── constants ──────────────────────────────────────────────────────────────────
FOLDS = [1, 2, 3, 4, 5]
SEED = 42
INNER_FOLDS = 5
TARGET_SENSITIVITY = 0.70

# Promotion rule (locked v5.1b reference)
LOCKED_AUC = 0.7829513888888889
LOCKED_PR_AUC = 0.5598729847183398

LOGREG_L2_ORIGINAL_GRID = [0.001, 0.01, 0.1, 1.0]
LOGREG_L2_DENSE_GRID = [
    1e-5, 2e-5, 3e-5, 5e-5, 7e-5,
    1e-4, 2e-4, 3e-4, 5e-4, 7e-4,
    1e-3, 2e-3, 3e-3, 1e-2,
]
SVM_C_GRID = [0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 50.0]
SVM_GAMMA_GRID = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 3e-3, "scale"]

CLASSIFIER_NAMES = ["logreg_l2_original", "logreg_l2_dense", "svm_rbf"]
CALIBRATION_LABELS = ["uncal", "cal"]
TOP_K_VALUES = [10, 20, 30, 50, 97]

# ── utilities ──────────────────────────────────────────────────────────────────

def safe_div(n: float, d: float) -> float:
    return float(n / d) if d else float("nan")


def binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    out: Dict[str, Any] = {
        "n": int(len(y_true)),
        "n_cn": int((y_true == 0).sum()),
        "n_ad": int((y_true == 1).sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": safe_div(tp + tn, len(y_true)),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "predicted_ad_rate": float(y_pred.mean()),
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["auc"] = out["pr_auc"] = float("nan")
    return out


def select_thresholds(
    y_true: np.ndarray, y_score: np.ndarray
) -> List[Dict[str, Any]]:
    """Inner-OOF threshold selection. y_true/y_score are trainDev OOF."""
    thresholds = np.unique(np.concatenate([[0.0, 0.5, 1.0], y_score]))
    rows = []
    for thr in thresholds:
        pred = (y_score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
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
    selections = []

    # youden_j
    r = tbl.sort_values(
        ["youden_j", "sensitivity", "specificity", "threshold"],
        ascending=[False, False, False, False]
    ).iloc[0]
    selections.append({
        "threshold_strategy": "inner_oof_youden_j",
        "threshold": float(r["threshold"]),
        "selection_metric": "youden_j",
        "inner_oof_sensitivity": float(r["sensitivity"]),
        "inner_oof_specificity": float(r["specificity"]),
        "inner_oof_balanced_accuracy": float(r["balanced_accuracy"]),
    })

    # balanced_accuracy
    r = tbl.sort_values(
        ["balanced_accuracy", "sensitivity", "specificity", "threshold"],
        ascending=[False, False, False, False]
    ).iloc[0]
    selections.append({
        "threshold_strategy": "inner_oof_balanced_accuracy",
        "threshold": float(r["threshold"]),
        "selection_metric": "balanced_accuracy",
        "inner_oof_sensitivity": float(r["sensitivity"]),
        "inner_oof_specificity": float(r["specificity"]),
        "inner_oof_balanced_accuracy": float(r["balanced_accuracy"]),
    })

    # target sens >= 0.70 max spec
    eligible = tbl[tbl["sensitivity"] >= TARGET_SENSITIVITY]
    if eligible.empty:
        r = tbl.sort_values(["sensitivity", "threshold"], ascending=[False, False]).iloc[0]
        status = "target_not_reached_inner_oof"
    else:
        r = eligible.sort_values(
            ["specificity", "sensitivity", "balanced_accuracy", "threshold"],
            ascending=[False, False, False, False]
        ).iloc[0]
        status = "selected_inner_oof"
    selections.append({
        "threshold_strategy": "inner_oof_target_sens_ge_0p70_max_spec",
        "threshold": float(r["threshold"]),
        "selection_metric": status,
        "inner_oof_sensitivity": float(r["sensitivity"]),
        "inner_oof_specificity": float(r["specificity"]),
        "inner_oof_balanced_accuracy": float(r["balanced_accuracy"]),
    })

    # fixed 0.5
    selections.append({
        "threshold_strategy": "fixed_0p5",
        "threshold": 0.5,
        "selection_metric": "fixed_no_selection",
        "inner_oof_sensitivity": float("nan"),
        "inner_oof_specificity": float("nan"),
        "inner_oof_balanced_accuracy": float("nan"),
    })
    return selections


# ── preprocessor ──────────────────────────────────────────────────────────────

def make_preprocessor(mu_cols: List[str]) -> ColumnTransformer:
    latent_pipe = Pipeline([("scaler", StandardScaler())])
    age_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    try:
        from sklearn.preprocessing import OneHotEncoder
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        from sklearn.preprocessing import OneHotEncoder
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    sex_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])
    return ColumnTransformer(
        [("latent", latent_pipe, mu_cols),
         ("age", age_pipe, ["Age"]),
         ("sex", sex_pipe, ["Sex"])],
        remainder="drop",
        sparse_threshold=0.0,
    )


# ── classifier specs ───────────────────────────────────────────────────────────

def build_classifier_specs(seed: int) -> Dict[str, Tuple[Pipeline, Dict]]:
    def logreg_pipe(s):
        return Pipeline([
            ("pre", "passthrough"),
            ("model", LogisticRegression(
                penalty="l2", solver="lbfgs", class_weight="balanced",
                max_iter=5000, random_state=s,
            )),
        ])

    svm_pipe = Pipeline([
        ("pre", "passthrough"),
        ("model", SVC(
            kernel="rbf", probability=True, class_weight="balanced",
            random_state=seed,
        )),
    ])

    return {
        "logreg_l2_original": (
            logreg_pipe(seed),
            {"model__C": LOGREG_L2_ORIGINAL_GRID},
        ),
        "logreg_l2_dense": (
            logreg_pipe(seed),
            {"model__C": LOGREG_L2_DENSE_GRID},
        ),
        "svm_rbf": (
            svm_pipe,
            {"model__C": SVM_C_GRID, "model__gamma": SVM_GAMMA_GRID},
        ),
    }


# ── Platt calibration ─────────────────────────────────────────────────────────

def fit_platt(oof_scores: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    cal = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    cal.fit(oof_scores.reshape(-1, 1), y_train)
    return cal


def apply_platt(cal: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return cal.predict_proba(scores.reshape(-1, 1))[:, 1]


# ── inner stratification ──────────────────────────────────────────────────────

def inner_stratification_key(
    df: pd.DataFrame, n_splits: int
) -> Tuple[pd.Series, str, int]:
    cols = ["ResearchGroup_Mapped", "Manufacturer"]
    key = df[cols].fillna("UNKNOWN").astype(str).apply(
        lambda r: "_".join(r.values), axis=1
    )
    min_count = int(key.value_counts().min())
    if min_count < n_splits:
        key = df["y"].astype(int)
        return key, "label_only_fallback", int(pd.Series(key).value_counts().min())
    return key, "ResearchGroup_Mapped+Manufacturer", min_count


# ── per-fold runner ────────────────────────────────────────────────────────────

def run_fold(
    fold: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_jobs: int,
) -> Tuple[List[Dict], List[pd.DataFrame], List[Dict]]:
    """
    Returns (metric_rows, pred_frames, hyperparam_rows).
    Processes all 3 classifiers × {uncal, cal}.
    """
    mu_cols = [c for c in train_df.columns if c.startswith("mu_")]
    feature_cols = mu_cols + ["Age", "Sex"]
    x_train = train_df[feature_cols].copy()
    x_test = test_df[feature_cols].copy()
    y_train = train_df["y"].astype(int).to_numpy()
    y_test = test_df["y"].astype(int).to_numpy()

    pre = make_preprocessor(mu_cols)
    inner_key, inner_context, min_cell = inner_stratification_key(train_df, INNER_FOLDS)
    inner_cv = list(
        StratifiedKFold(
            n_splits=INNER_FOLDS, shuffle=True,
            random_state=SEED + fold + 30,
        ).split(np.zeros(len(train_df)), inner_key)
    )
    specs = build_classifier_specs(seed=SEED + fold)

    metric_rows: List[Dict] = []
    pred_frames: List[pd.DataFrame] = []
    hyperparam_rows: List[Dict] = []

    for clf_name, (base_pipe, param_grid) in specs.items():
        pipe = clone(base_pipe)
        pipe.steps[0] = ("pre", pre)

        # inner CV grid search
        search = GridSearchCV(
            estimator=pipe, param_grid=param_grid,
            scoring="roc_auc", cv=inner_cv,
            n_jobs=n_jobs, refit=True, error_score=np.nan,
        )
        search.fit(x_train, y_train)
        best = search.best_estimator_
        best_inner_auc = float(search.best_score_)
        best_params = search.best_params_

        hyperparam_rows.append({
            "fold": fold,
            "classifier": clf_name,
            "best_params": json.dumps(best_params, sort_keys=True),
            "best_inner_auc": best_inner_auc,
            "inner_cv_context": inner_context,
            "min_inner_stratum_count": int(min_cell),
        })

        # inner-OOF uncalibrated scores
        oof_uncal = cross_val_predict(
            clone(best), x_train, y_train,
            cv=inner_cv, method="predict_proba", n_jobs=n_jobs,
        )[:, 1]
        test_uncal = best.predict_proba(x_test)[:, 1]

        # Platt calibration
        platt = fit_platt(oof_uncal, y_train)
        oof_cal = apply_platt(platt, oof_uncal)
        test_cal = apply_platt(platt, test_uncal)

        # metadata to attach to predictions
        meta_cols = [
            "SubjectID", "tensor_idx", "ResearchGroup_Mapped",
            "Manufacturer", "Age", "Sex", "source_batch",
            "source_label", "tensor_source",
        ]
        base_pred = test_df[[c for c in meta_cols if c in test_df.columns]].copy()
        base_pred["fold"] = fold
        base_pred["y_true"] = y_test

        for cal_label, oof_scores, test_scores in [
            ("uncal", oof_uncal, test_uncal),
            ("cal",   oof_cal,   test_cal),
        ]:
            model_key = f"{clf_name}_{cal_label}"
            thresholds = select_thresholds(y_train, oof_scores)

            for sel in thresholds:
                thr = float(sel["threshold"])
                y_pred = (test_scores >= thr).astype(int)
                m = binary_metrics(y_test, test_scores, y_pred)

                row = {
                    "fold": fold,
                    "model_name": model_key,
                    "classifier": clf_name,
                    "calibrated": cal_label == "cal",
                    "readout_feature_set": "z_plus_age_sex",
                    "threshold_strategy": sel["threshold_strategy"],
                    "threshold": thr,
                    "threshold_selection_context": (
                        "true_inner_cv_oof"
                        if sel["threshold_strategy"] != "fixed_0p5"
                        else "fixed_no_selection"
                    ),
                    "inner_cv_context": inner_context,
                    "min_inner_stratum_count": int(min_cell),
                    "best_inner_auc": best_inner_auc,
                    "best_params": json.dumps(best_params, sort_keys=True),
                    **{f"inner_oof_{k}": sel.get(f"inner_oof_{k}", float("nan"))
                       for k in ["sensitivity", "specificity", "balanced_accuracy"]},
                    **m,
                }
                metric_rows.append(row)

                pred_row = base_pred.copy()
                pred_row["model_name"] = model_key
                pred_row["classifier"] = clf_name
                pred_row["calibrated"] = cal_label == "cal"
                pred_row["readout_feature_set"] = "z_plus_age_sex"
                pred_row["threshold_strategy"] = sel["threshold_strategy"]
                pred_row["threshold"] = thr
                pred_row["y_score"] = test_scores
                pred_row["y_pred"] = y_pred
                pred_frames.append(pred_row)

    return metric_rows, pred_frames, hyperparam_rows


# ── aggregation ────────────────────────────────────────────────────────────────

def compute_pooled(predictions: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_name", "classifier", "calibrated",
                  "readout_feature_set", "threshold_strategy"]
    rows = []
    for keys, sub in predictions.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["threshold"] = "fold_specific" if keys[-1] != "fixed_0p5" else 0.5
        row.update(binary_metrics(
            sub["y_true"].to_numpy(),
            sub["y_score"].to_numpy(),
            sub["y_pred"].to_numpy(),
        ))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["classifier", "calibrated", "threshold_strategy"]
    )


def compute_foldwise_summary(foldwise: pd.DataFrame) -> pd.DataFrame:
    """Unweighted foldwise mean and std of AUC/PR-AUC."""
    group_cols = ["model_name", "classifier", "calibrated", "threshold_strategy"]
    rows = []
    for keys, sub in foldwise.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        for col in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
            vals = sub[col].dropna().to_numpy()
            row[f"{col}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["classifier", "calibrated", "threshold_strategy"])


# ── top-k precision ───────────────────────────────────────────────────────────

def compute_topk_precision(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per model/strategy: precision@k for top-k highest-scored subjects."""
    rows = []
    group_cols = ["model_name", "classifier", "calibrated", "threshold_strategy"]
    for keys, sub in predictions.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        sorted_sub = sub.sort_values("y_score", ascending=False)
        y_sorted = sorted_sub["y_true"].to_numpy()
        for k in TOP_K_VALUES:
            k_eff = min(k, len(y_sorted))
            row[f"precision_at_{k}"] = float(np.mean(y_sorted[:k_eff]))
            row[f"n_ad_in_top{k}"] = int(y_sorted[:k_eff].sum())
        row["total_ad"] = int((sub["y_true"] == 1).sum())
        row["total_n"] = int(len(sub))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["classifier", "calibrated", "threshold_strategy"])


# ── error analysis by subgroup ────────────────────────────────────────────────

def age_quartile(age: pd.Series) -> pd.Series:
    q25, q50, q75 = age.quantile([0.25, 0.50, 0.75]).values
    def label(a):
        if pd.isna(a):
            return "Age_unknown"
        if a < q25:
            return f"Age<{q25:.0f}"
        if a < q50:
            return f"Age{q25:.0f}-{q50:.0f}"
        if a < q75:
            return f"Age{q50:.0f}-{q75:.0f}"
        return f"Age≥{q75:.0f}"
    return age.map(label)


def compute_error_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    preds = predictions.copy()
    preds["site"] = preds["SubjectID"].astype(str).str.split("_").str[0]
    preds["age_group"] = age_quartile(pd.to_numeric(preds["Age"], errors="coerce"))

    group_cols = ["model_name", "classifier", "calibrated", "threshold_strategy"]
    subgroup_cols = {
        "Manufacturer": "Manufacturer",
        "Sex": "Sex",
        "age_group": "age_group",
    }

    rows = []
    for keys, sub in predictions.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        for subgroup_name, col in subgroup_cols.items():
            # merge derived col for age_group
            if col == "age_group":
                sub_local = sub.copy()
                sub_local["age_group"] = age_quartile(
                    pd.to_numeric(sub_local["Age"], errors="coerce")
                )
            else:
                sub_local = sub
            for val, grp in sub_local.groupby(col, dropna=False):
                y_t = grp["y_true"].to_numpy()
                y_s = grp["y_score"].to_numpy()
                y_p = grp["y_pred"].to_numpy()
                m = binary_metrics(y_t, y_s, y_p)
                row = {**base, "subgroup_type": subgroup_name, "subgroup_value": str(val)}
                row.update(m)
                # FP rate and FN rate for quick scan
                row["fp_rate"] = safe_div(m["fp"], m["n_cn"]) if m["n_cn"] else float("nan")
                row["fn_rate"] = safe_div(m["fn"], m["n_ad"]) if m["n_ad"] else float("nan")
                rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["classifier", "calibrated", "threshold_strategy", "subgroup_type", "subgroup_value"]
    )


# ── plotting ───────────────────────────────────────────────────────────────────

def _savefig(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_roc_curves(predictions: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # One figure per (classifier, calibration)
    for (clf, cal_label), group in predictions.groupby(
        ["classifier", "calibrated"], dropna=False
    ):
        if not any(group["threshold_strategy"] == "fixed_0p5"):
            continue
        sub_fixed = group[group["threshold_strategy"] == "fixed_0p5"]
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = cm.tab10(np.linspace(0, 0.5, len(FOLDS)))
        fold_aucs = []
        for i, fold in enumerate(FOLDS):
            fold_sub = sub_fixed[sub_fixed["fold"] == fold]
            if fold_sub.empty:
                continue
            fpr, tpr, _ = roc_curve(fold_sub["y_true"], fold_sub["y_score"])
            auc_v = roc_auc_score(fold_sub["y_true"], fold_sub["y_score"])
            fold_aucs.append(auc_v)
            ax.plot(fpr, tpr, color=colors[i], alpha=0.6,
                    label=f"Fold {fold} (AUC={auc_v:.3f})", lw=1.2)
        # pooled
        fpr_p, tpr_p, _ = roc_curve(sub_fixed["y_true"], sub_fixed["y_score"])
        auc_pooled = roc_auc_score(sub_fixed["y_true"], sub_fixed["y_score"])
        ax.plot(fpr_p, tpr_p, "k-", lw=2.0,
                label=f"Pooled (AUC={auc_pooled:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        cal_str = "calibrated" if cal_label else "uncalibrated"
        ax.set_title(f"ROC — {clf} ({cal_str})\n"
                     f"FW-mean AUC={np.mean(fold_aucs):.3f} ± {np.std(fold_aucs, ddof=1):.3f}")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fname = f"roc_{clf}_{cal_str}.png"
        _savefig(fig, fig_dir / fname)


def plot_pr_curves(predictions: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    for (clf, cal_label), group in predictions.groupby(
        ["classifier", "calibrated"], dropna=False
    ):
        if not any(group["threshold_strategy"] == "fixed_0p5"):
            continue
        sub_fixed = group[group["threshold_strategy"] == "fixed_0p5"]
        ad_prev = (sub_fixed["y_true"] == 1).mean()
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = cm.tab10(np.linspace(0, 0.5, len(FOLDS)))
        fold_praucs = []
        for i, fold in enumerate(FOLDS):
            fold_sub = sub_fixed[sub_fixed["fold"] == fold]
            if fold_sub.empty:
                continue
            prec, rec, _ = precision_recall_curve(
                fold_sub["y_true"], fold_sub["y_score"]
            )
            prauc = average_precision_score(fold_sub["y_true"], fold_sub["y_score"])
            fold_praucs.append(prauc)
            ax.plot(rec, prec, color=colors[i], alpha=0.6,
                    label=f"Fold {fold} (AP={prauc:.3f})", lw=1.2)
        prec_p, rec_p, _ = precision_recall_curve(
            sub_fixed["y_true"], sub_fixed["y_score"]
        )
        prauc_pooled = average_precision_score(
            sub_fixed["y_true"], sub_fixed["y_score"]
        )
        ax.plot(rec_p, prec_p, "k-", lw=2.0,
                label=f"Pooled (AP={prauc_pooled:.3f})")
        ax.axhline(ad_prev, color="gray", lw=0.8, ls="--",
                   label=f"Chance (AD prev={ad_prev:.2f})")
        ax.set_xlabel("Recall (Sensitivity)")
        ax.set_ylabel("Precision")
        cal_str = "calibrated" if cal_label else "uncalibrated"
        ax.set_title(f"PR — {clf} ({cal_str})\n"
                     f"FW-mean AP={np.mean(fold_praucs):.3f} ± {np.std(fold_praucs, ddof=1):.3f}")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fname = f"pr_{clf}_{cal_str}.png"
        _savefig(fig, fig_dir / fname)


def plot_confusion_heatmaps(pooled: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    strategies = [
        "inner_oof_youden_j",
        "inner_oof_target_sens_ge_0p70_max_spec",
        "fixed_0p5",
    ]
    # show calibrated only for each classifier
    for (clf, cal), grp in pooled.groupby(["classifier", "calibrated"]):
        cal_str = "cal" if cal else "uncal"
        subset = grp[grp["threshold_strategy"].isin(strategies)]
        if subset.empty:
            continue
        n_strats = len(subset["threshold_strategy"].unique())
        fig, axes = plt.subplots(1, n_strats, figsize=(4 * n_strats, 4))
        if n_strats == 1:
            axes = [axes]
        for ax, strat in zip(axes, strategies):
            row = subset[subset["threshold_strategy"] == strat]
            if row.empty:
                ax.axis("off")
                continue
            r = row.iloc[0]
            cm_data = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], dtype=int)
            im = ax.imshow(cm_data, cmap="Blues", vmin=0)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Pred CN", "Pred AD"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["True CN", "True AD"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm_data[i, j]),
                            ha="center", va="center", fontsize=12,
                            color="white" if cm_data[i, j] > cm_data.max() / 2 else "black")
            sens = r.get("sensitivity", float("nan"))
            spec = r.get("specificity", float("nan"))
            ax.set_title(f"{strat}\nSens={sens:.2f} Spec={spec:.2f}", fontsize=8)
        fig.suptitle(f"Confusion — {clf} ({cal_str})", fontsize=10)
        plt.tight_layout()
        fname = f"confusion_{clf}_{cal_str}.png"
        _savefig(fig, fig_dir / fname)


# ── recommendation ────────────────────────────────────────────────────────────

def build_recommendation(
    pooled: pd.DataFrame,
    fw_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Rescue Readout — Final Recommendation",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"\nRun: recover035_latent384_T80_h10000_p560_full5x5",
        "",
        "## Promotion Rule",
        "",
        f"- Locked reference (v5.1b): pooled AUC = {LOCKED_AUC:.4f}, pooled PR-AUC = {LOCKED_PR_AUC:.4f}",
        f"- Candidate must achieve: AUC > {LOCKED_AUC:.4f} **AND** PR-AUC ≥ {LOCKED_PR_AUC:.4f}",
        "",
        "## Best Pooled AUC by Model (fixed_0p5 threshold)",
        "",
    ]
    fixed = pooled[pooled["threshold_strategy"] == "fixed_0p5"].copy()
    fixed_sorted = fixed.sort_values("auc", ascending=False)
    cols = ["model_name", "auc", "pr_auc", "balanced_accuracy",
            "sensitivity", "specificity", "f1"]
    lines += ["| " + " | ".join(cols) + " |",
              "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in fixed_sorted.head(8).iterrows():
        vals = []
        for c in cols:
            v = r.get(c, "")
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Best Pooled AUC by Model (inner_oof_youden_j threshold)",
        "",
    ]
    youden = pooled[pooled["threshold_strategy"] == "inner_oof_youden_j"].copy()
    youden_sorted = youden.sort_values("auc", ascending=False)
    lines += ["| " + " | ".join(cols) + " |",
              "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in youden_sorted.head(8).iterrows():
        vals = [f"{r.get(c, ''):.4f}" if isinstance(r.get(c, ""), float) else str(r.get(c, "")) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")

    lines += ["", "## Promotion Check", ""]
    best_row = fixed_sorted.iloc[0] if not fixed_sorted.empty else None
    if best_row is not None:
        best_auc = float(best_row["auc"])
        best_prauc = float(best_row["pr_auc"])
        promotes_auc = best_auc > LOCKED_AUC
        promotes_prauc = best_prauc >= LOCKED_PR_AUC
        promotes = promotes_auc and promotes_prauc
        status = "**PROMOTES**" if promotes else "**DOES NOT PROMOTE**"
        lines += [
            f"Best model: `{best_row['model_name']}` (fixed_0p5)",
            f"- Pooled AUC = {best_auc:.4f}  vs locked {LOCKED_AUC:.4f}  → {'PASS' if promotes_auc else 'FAIL'}",
            f"- Pooled PR-AUC = {best_prauc:.4f}  vs locked {LOCKED_PR_AUC:.4f}  → {'PASS' if promotes_prauc else 'FAIL'}",
            f"- Verdict: {status}",
        ]
    else:
        lines.append("No pooled results available.")

    lines += [
        "",
        "## Foldwise Mean AUC Summary (unweighted, fixed_0p5)",
        "",
        "| model_name | auc_mean | auc_std | pr_auc_mean | pr_auc_std |",
        "|---|---|---|---|---|",
    ]
    fw_fixed = fw_summary[fw_summary["threshold_strategy"] == "fixed_0p5"].sort_values(
        "auc_mean", ascending=False
    )
    for _, r in fw_fixed.iterrows():
        lines.append(
            f"| {r['model_name']} | {r['auc_mean']:.4f} | {r['auc_std']:.4f} "
            f"| {r['pr_auc_mean']:.4f} | {r['pr_auc_std']:.4f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "The foldwise-mean AUC inflates vs pooled AUC because per-fold Platt calibration",
        "is fit independently; pooled scores are not on a common scale.",
        "The canonical promotion metric is **pooled AUC** (Stage B).",
        "",
        "No retraining. No threshold fitting outside inner CV. No OASIS data.",
    ]
    return "\n".join(lines)


# ── output writers ─────────────────────────────────────────────────────────────

def df_to_md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except Exception:
        return df.to_string(index=False)


def write_outputs(
    out_dir: Path,
    foldwise: pd.DataFrame,
    pooled: pd.DataFrame,
    fw_summary: pd.DataFrame,
    predictions: pd.DataFrame,
    hyperparams: pd.DataFrame,
    topk: pd.DataFrame,
    error_df: pd.DataFrame,
    rec_text: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    foldwise.to_csv(out_dir / "classifier_rescue_foldwise_metrics.csv", index=False)
    (out_dir / "classifier_rescue_foldwise_metrics.md").write_text(
        "# Rescue Foldwise Metrics\n\n" + df_to_md(foldwise), encoding="utf-8"
    )
    pooled.to_csv(out_dir / "classifier_rescue_pooled_metrics.csv", index=False)
    (out_dir / "classifier_rescue_pooled_metrics.md").write_text(
        "# Rescue Pooled Metrics\n\n" + df_to_md(pooled), encoding="utf-8"
    )
    fw_summary.to_csv(out_dir / "classifier_rescue_foldwise_summary.csv", index=False)
    predictions.to_csv(out_dir / "classifier_rescue_predictions.csv", index=False)
    hyperparams.to_csv(out_dir / "classifier_rescue_hyperparams_by_fold.csv", index=False)
    (out_dir / "classifier_rescue_hyperparams_by_fold.md").write_text(
        "# Hyperparameters by Fold\n\n" + df_to_md(hyperparams), encoding="utf-8"
    )
    topk.to_csv(out_dir / "topk_precision.csv", index=False)
    (out_dir / "topk_precision.md").write_text(
        "# Precision @ Top-k AD-Risk Subjects\n\n" + df_to_md(topk), encoding="utf-8"
    )
    error_df.to_csv(out_dir / "error_analysis_by_subgroup.csv", index=False)
    (out_dir / "error_analysis_by_subgroup.md").write_text(
        "# Error Analysis by Subgroup\n\n" + df_to_md(error_df), encoding="utf-8"
    )
    (out_dir / "final_recommendation.md").write_text(rec_text, encoding="utf-8")


def write_readme(out_dir: Path, pooled: pd.DataFrame) -> None:
    best_auc_row = pooled[pooled["threshold_strategy"] == "fixed_0p5"].sort_values(
        "auc", ascending=False
    )
    best_str = (
        f"`{best_auc_row.iloc[0]['model_name']}`"
        f" pooled AUC={best_auc_row.iloc[0]['auc']:.4f}"
        if not best_auc_row.empty else "N/A"
    )
    text = f"""\
# Rescue Readout — recover035_latent384_T80_h10000_p560_full5x5

Read-only: existing latent caches reused, no VAE retraining.

## Best model (fixed_0p5)
{best_str}

## Promotion rule
AUC > {LOCKED_AUC:.4f} AND PR-AUC ≥ {LOCKED_PR_AUC:.4f}

## Files
| File | Description |
|------|-------------|
| `classifier_rescue_pooled_metrics.csv/.md` | Pooled metrics, all models × thresholds |
| `classifier_rescue_foldwise_metrics.csv/.md` | Per-fold metrics |
| `classifier_rescue_foldwise_summary.csv` | Foldwise mean/std of AUC/PR-AUC |
| `classifier_rescue_predictions.csv` | Per-subject scores and predictions |
| `classifier_rescue_hyperparams_by_fold.csv/.md` | Best C/gamma per fold |
| `topk_precision.csv/.md` | Precision at top-k AD-risk subjects |
| `error_analysis_by_subgroup.csv/.md` | FP/FN by Manufacturer, Sex, Age |
| `figures/` | ROC, PR curves and confusion heatmaps |
| `final_recommendation.md` | Promotion decision |
| `command_log.json` | Provenance |

## Guarantees
- No VAE retraining
- No threshold leakage (all selection from inner-CV OOF)
- No OASIS data
- No tensor/metadata/ledger modification
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def write_command_log(out_dir: Path, args: argparse.Namespace) -> None:
    log = {
        "script": str(Path(__file__).resolve()),
        "run_id": "recover035_latent384_T80_h10000_p560_full5x5",
        "latent_cache_dir": str(LATENT_CACHE_DIR),
        "output_dir": str(out_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classifiers": CLASSIFIER_NAMES,
        "calibration": CALIBRATION_LABELS,
        "feature_set": "z_plus_age_sex",
        "inner_folds": INNER_FOLDS,
        "threshold_strategies": [
            "fixed_0p5", "inner_oof_youden_j",
            "inner_oof_balanced_accuracy",
            "inner_oof_target_sens_ge_0p70_max_spec",
        ],
        "logreg_l2_original_C_grid": LOGREG_L2_ORIGINAL_GRID,
        "logreg_l2_dense_C_grid": LOGREG_L2_DENSE_GRID,
        "svm_rbf_C_grid": SVM_C_GRID,
        "svm_rbf_gamma_grid": [str(g) for g in SVM_GAMMA_GRID],
        "n_jobs": args.n_jobs,
        "locked_reference_auc": LOCKED_AUC,
        "locked_reference_pr_auc": LOCKED_PR_AUC,
        "vae_retrained": False,
        "no_threshold_leakage": True,
        "no_oasis": True,
        "tensor_modified": False,
        "metadata_modified": False,
    }
    (out_dir / "command_log.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=RESCUE_DIR)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir

    print("=== Rescue Readout — recover035_latent384_T80_h10000_p560_full5x5 ===")
    print(f"Latent cache : {LATENT_CACHE_DIR}")
    print(f"Output dir   : {out_dir}")

    # ── input checks ──
    errors = []
    for fold in FOLDS:
        for split in ["trainDev", "test"]:
            p = LATENT_CACHE_DIR / f"fold_{fold}_{split}_latent_mu.csv"
            if p.exists():
                print(f"  [OK]     {p.name}")
            else:
                print(f"  [MISSING] {p}")
                errors.append(str(p))
    if errors:
        print(f"\nFATAL: {len(errors)} latent cache files missing.")
        return 1

    if args.dry_run:
        print("\n[DRY-RUN] All latent caches present. Would write:")
        for f in [
            "classifier_rescue_pooled_metrics.csv/.md",
            "classifier_rescue_foldwise_metrics.csv/.md",
            "classifier_rescue_foldwise_summary.csv",
            "classifier_rescue_predictions.csv",
            "classifier_rescue_hyperparams_by_fold.csv/.md",
            "topk_precision.csv/.md",
            "error_analysis_by_subgroup.csv/.md",
            "figures/roc_{clf}_{cal}.png (6 files)",
            "figures/pr_{clf}_{cal}.png (6 files)",
            "figures/confusion_{clf}_{cal}.png (6 files)",
            "final_recommendation.md",
            "README.md",
            "command_log.json",
        ]:
            print(f"  {out_dir}/{f}")
        print("[DRY-RUN] complete — no files written.")
        return 0

    if out_dir.exists() and not args.overwrite:
        existing = list(out_dir.glob("classifier_rescue_*.csv"))
        if existing:
            print(f"ERROR: {out_dir} already has rescue outputs. Pass --overwrite.")
            return 1

    # ── run folds ──
    all_metric_rows: List[Dict] = []
    all_pred_frames: List[pd.DataFrame] = []
    all_hyperparam_rows: List[Dict] = []

    for fold in FOLDS:
        print(f"\n[Fold {fold}] loading latent cache...")
        train_df = pd.read_csv(
            LATENT_CACHE_DIR / f"fold_{fold}_trainDev_latent_mu.csv"
        )
        test_df = pd.read_csv(
            LATENT_CACHE_DIR / f"fold_{fold}_test_latent_mu.csv"
        )
        print(f"  trainDev n={len(train_df)}, test n={len(test_df)}")
        print(f"  Running classifiers: {CLASSIFIER_NAMES}")
        metric_rows, pred_frames, hp_rows = run_fold(
            fold=fold,
            train_df=train_df,
            test_df=test_df,
            n_jobs=args.n_jobs,
        )
        all_metric_rows.extend(metric_rows)
        all_pred_frames.extend(pred_frames)
        all_hyperparam_rows.extend(hp_rows)
        for clf_name in CLASSIFIER_NAMES:
            fold_rows = [r for r in metric_rows
                         if r["classifier"] == clf_name
                         and r["threshold_strategy"] == "fixed_0p5"
                         and not r["calibrated"]]
            if fold_rows:
                r = fold_rows[0]
                print(f"    {clf_name}_uncal fixed_0p5: "
                      f"AUC={r['auc']:.4f} PR-AUC={r['pr_auc']:.4f}")

    # ── aggregate ──
    print("\nAggregating...")
    foldwise = pd.DataFrame(all_metric_rows).sort_values(
        ["classifier", "calibrated", "threshold_strategy", "fold"]
    )
    predictions = pd.concat(all_pred_frames, ignore_index=True, sort=False)
    hyperparams = pd.DataFrame(all_hyperparam_rows).sort_values(["fold", "classifier"])
    pooled = compute_pooled(predictions)
    fw_summary = compute_foldwise_summary(foldwise)

    # ── extras ──
    print("Computing top-k precision...")
    topk = compute_topk_precision(predictions)
    print("Computing error analysis by subgroup...")
    error_df = compute_error_analysis(predictions)

    # ── plots ──
    print("Plotting ROC and PR curves...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        plot_roc_curves(predictions, fig_dir)
        plot_pr_curves(predictions, fig_dir)
        plot_confusion_heatmaps(pooled, fig_dir)
        print(f"  Figures written to {fig_dir}")
    except Exception as exc:
        warnings.warn(f"Figure generation failed: {exc}")

    # ── write outputs ──
    print("Writing outputs...")
    rec_text = build_recommendation(pooled, fw_summary)
    write_outputs(
        out_dir=out_dir,
        foldwise=foldwise,
        pooled=pooled,
        fw_summary=fw_summary,
        predictions=predictions,
        hyperparams=hyperparams,
        topk=topk,
        error_df=error_df,
        rec_text=rec_text,
    )
    write_readme(out_dir, pooled)
    write_command_log(out_dir, args)

    # ── summary ──
    print(f"\n=== Done. Output: {out_dir} ===")
    print("\nPooled metrics (fixed_0p5, sorted by AUC):")
    show_cols = ["model_name", "auc", "pr_auc", "balanced_accuracy",
                 "sensitivity", "specificity", "f1"]
    subset = pooled[pooled["threshold_strategy"] == "fixed_0p5"][show_cols]
    print(subset.sort_values("auc", ascending=False).to_string(index=False))
    print(f"\nPromotion rule: AUC > {LOCKED_AUC:.4f} AND PR-AUC ≥ {LOCKED_PR_AUC:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
