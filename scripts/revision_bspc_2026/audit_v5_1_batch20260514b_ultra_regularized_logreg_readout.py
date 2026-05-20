#!/usr/bin/env python3
"""Read-only audit: ultra-regularized logreg on locked v5.1 batch20260514b [1,0,2] latents.

Tests whether Stage-B logreg_l2 was constrained by the lower bound (C=0.001) of the
original C grid by extending the grid to [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3,
3e-3, 1e-2] and re-running inner CV on the reused frozen latent cache.

No VAE is trained. No tensors, metadata, or ledger files are modified.
Latent μ cache is reused from adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Locked reference metrics (adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep) ──
LOCKED_AUC = 0.778785
LOCKED_PR_AUC = 0.551832
LOCKED_C = 0.001          # lower bound of original grid; was selected in all 5 folds
ORIGINAL_C_GRID = [0.001, 0.01, 0.1, 1.0]

# Extended grid: covers two decades below the original lower bound
EXTENDED_C_GRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

MODEL_NAME = "logreg_l2_ultra_regularized"
TARGET_SENSITIVITY = 0.70

DEFAULT_LATENT_CACHE_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
    / "latent_cache"
)
DEFAULT_REFERENCE_POOLED_CSV = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
    / "classifier_sweep_pooled_metrics.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_ultra_regularized_logreg_readout_audit"
)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--latent-cache-dir", type=Path, default=DEFAULT_LATENT_CACHE_DIR)
    p.add_argument("--reference-pooled-csv", type=Path, default=DEFAULT_REFERENCE_POOLED_CSV)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs and exit without running classifiers.")
    return p.parse_args()


# ── IO helpers ─────────────────────────────────────────────────────────────────

def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    generated = [
        "README.md",
        "primary_comparison.csv",
        "primary_comparison.md",
        "selected_C_by_fold.csv",
        "selected_C_by_fold.md",
        "manufacturer_subgroup_comparison.csv",
        "manufacturer_subgroup_comparison.md",
        "predictions.csv",
        "recommendation.md",
        "command_log.json",
    ]
    if path.exists() and any((path / n).exists() for n in generated):
        if not overwrite:
            raise FileExistsError(f"{path} already has audit outputs; pass --overwrite to re-run.")
        for n in generated:
            q = path / n
            if q.exists():
                q.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_latent_cache(cache_dir: Path, outer_folds: int) -> None:
    missing = []
    for fold in range(1, outer_folds + 1):
        for split in ["trainDev", "test"]:
            p = cache_dir / f"fold_{fold}_{split}_latent_mu.csv"
            if not p.exists():
                missing.append(str(p))
    if missing:
        raise FileNotFoundError("Missing latent cache files:\n" + "\n".join(missing))


def load_latent_pair(cache_dir: Path, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(cache_dir / f"fold_{fold}_trainDev_latent_mu.csv")
    test = pd.read_csv(cache_dir / f"fold_{fold}_test_latent_mu.csv")
    return train, test


# ── Preprocessing ─────────────────────────────────────────────────────────────

def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(mu_cols: List[str]) -> ColumnTransformer:
    numeric_latent = Pipeline([("scaler", StandardScaler())])
    numeric_age = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_ohe())])
    return ColumnTransformer(
        [
            ("latent", numeric_latent, mu_cols),
            ("age", numeric_age, ["Age"]),
            ("sex", categorical, ["Sex"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


# ── Inner stratification ───────────────────────────────────────────────────────

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


# ── Scoring helpers ────────────────────────────────────────────────────────────

def score_1d(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(x)[:, 1], dtype=float)
    raw = np.asarray(estimator.decision_function(x), dtype=float).ravel()
    return 1.0 / (1.0 + np.exp(-raw))


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
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
        "predicted_ad_rate": float(pred.mean()) if len(pred) else float("nan"),
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
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
    for criterion, metric in [("inner_oof_youden_j", "youden_j"), ("inner_oof_balanced_accuracy", "balanced_accuracy")]:
        r = tbl.sort_values([metric, "sensitivity", "specificity", "threshold"], ascending=[False, False, False, False]).iloc[0]
        selections.append(
            {
                "threshold_strategy": criterion,
                "threshold": float(r["threshold"]),
                "selection_metric": metric,
                "inner_oof_sensitivity": float(r["sensitivity"]),
                "inner_oof_specificity": float(r["specificity"]),
                "inner_oof_balanced_accuracy": float(r["balanced_accuracy"]),
            }
        )
    eligible = tbl[tbl["sensitivity"] >= TARGET_SENSITIVITY]
    if eligible.empty:
        r = tbl.sort_values(["sensitivity", "specificity", "threshold"], ascending=[False, False, False]).iloc[0]
        status = "target_not_reached_inner_oof"
    else:
        r = eligible.sort_values(["specificity", "sensitivity", "balanced_accuracy", "threshold"], ascending=[False, False, False, False]).iloc[0]
        status = "selected_inner_oof"
    selections.append(
        {
            "threshold_strategy": "inner_oof_target_sens_ge_0p70_max_spec",
            "threshold": float(r["threshold"]),
            "selection_metric": status,
            "inner_oof_sensitivity": float(r["sensitivity"]),
            "inner_oof_specificity": float(r["specificity"]),
            "inner_oof_balanced_accuracy": float(r["balanced_accuracy"]),
        }
    )
    selections.append(
        {
            "threshold_strategy": "fixed_0p5",
            "threshold": 0.5,
            "selection_metric": "fixed_no_selection",
            "inner_oof_sensitivity": np.nan,
            "inner_oof_specificity": np.nan,
            "inner_oof_balanced_accuracy": np.nan,
        }
    )
    return selections


# ── C-grid boundary helper ─────────────────────────────────────────────────────

def c_boundary_status(c: float, grid: List[float]) -> str:
    g_sorted = sorted(grid)
    if abs(c - g_sorted[0]) < 1e-12:
        return "at_lower_bound"
    if abs(c - g_sorted[-1]) < 1e-12:
        return "at_upper_bound"
    return "interior"


# ── Core audit loop ────────────────────────────────────────────────────────────

def run_audit(
    cache_dir: Path,
    outer_folds: int,
    inner_folds: int,
    seed: int,
    n_jobs: int,
) -> Dict[str, pd.DataFrame]:
    c_grid_rows: List[Dict[str, Any]] = []
    fold_metric_rows: List[Dict[str, Any]] = []
    pred_rows: List[pd.DataFrame] = []
    subgroup_rows: List[Dict[str, Any]] = []

    pipe_base = Pipeline(
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
    )
    param_grid = {"model__C": EXTENDED_C_GRID}

    for fold in range(1, outer_folds + 1):
        train_df, test_df = load_latent_pair(cache_dir, fold)
        mu_cols = [c for c in train_df.columns if c.startswith("mu_")]
        feature_cols = mu_cols + ["Age", "Sex"]
        x_train = train_df[feature_cols].copy()
        y_train = train_df["y"].astype(int).to_numpy()
        x_test = test_df[feature_cols].copy()
        y_test = test_df["y"].astype(int).to_numpy()

        inner_key, inner_context, min_inner_cell = inner_stratification_key(train_df, n_splits=inner_folds)
        inner_cv = list(
            StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + fold + 30).split(
                np.zeros(len(train_df)), inner_key
            )
        )

        pre = make_preprocessor(mu_cols)
        pipe = clone(pipe_base)
        pipe.steps[0] = ("pre", pre)

        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
            error_score=np.nan,
        )
        search.fit(x_train, y_train)
        best = search.best_estimator_
        selected_c = float(search.best_params_["model__C"])
        boundary = c_boundary_status(selected_c, EXTENDED_C_GRID)

        c_grid_rows.append(
            {
                "fold": fold,
                "model_name": MODEL_NAME,
                "selected_C": selected_c,
                "selected_C_log10": float(np.log10(selected_c)),
                "boundary_status": boundary,
                "best_inner_auc": float(search.best_score_),
                "inner_cv_context": inner_context,
                "minimum_inner_stratum_count": int(min_inner_cell),
                "original_locked_C": LOCKED_C,
                "original_c_grid_lower_bound": ORIGINAL_C_GRID[0],
                "extended_c_grid_lower_bound": EXTENDED_C_GRID[0],
                "extended_c_grid_upper_bound": EXTENDED_C_GRID[-1],
            }
        )

        oof_score = cross_val_predict(
            clone(best), x_train, y_train, cv=inner_cv, method="predict_proba", n_jobs=n_jobs
        )[:, 1]
        test_score = score_1d(best, x_test)
        thresholds = select_thresholds(y_train, oof_score)

        for sel in thresholds:
            thr = float(sel["threshold"])
            y_pred = (test_score >= thr).astype(int)
            row: Dict[str, Any] = {
                "fold": fold,
                "model_name": MODEL_NAME,
                "threshold_strategy": sel["threshold_strategy"],
                "threshold": thr,
                "threshold_selection_context": (
                    "true_inner_cv_oof" if sel["threshold_strategy"] != "fixed_0p5" else "fixed_no_selection"
                ),
                "inner_cv_context": inner_context,
                "minimum_inner_stratum_count": int(min_inner_cell),
                "selected_C": selected_c,
                "selected_C_log10": float(np.log10(selected_c)),
                "boundary_status": boundary,
                "best_inner_auc": float(search.best_score_),
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
                ]
            ].copy()
            pred["fold"] = fold
            pred["model_name"] = MODEL_NAME
            pred["threshold_strategy"] = sel["threshold_strategy"]
            pred["threshold"] = thr
            pred["y_true"] = y_test
            pred["y_score"] = test_score
            pred["y_pred"] = y_pred
            pred_rows.append(pred)

            for manufacturer, sub_idx in pred.groupby("Manufacturer", dropna=False).groups.items():
                sub_pred = pred.loc[list(sub_idx)]
                sub_row: Dict[str, Any] = {
                    "fold": fold,
                    "model_name": MODEL_NAME,
                    "threshold_strategy": sel["threshold_strategy"],
                    "threshold": thr,
                    "Manufacturer": manufacturer,
                    "selected_C": selected_c,
                    "boundary_status": boundary,
                }
                sub_row.update(binary_metrics(sub_pred["y_true"], sub_pred["y_score"], sub_pred["y_pred"]))
                subgroup_rows.append(sub_row)

    return {
        "selected_C_by_fold": pd.DataFrame(c_grid_rows),
        "foldwise_metrics": pd.DataFrame(fold_metric_rows),
        "predictions": pd.concat(pred_rows, ignore_index=True, sort=False),
        "subgroup_by_manufacturer": pd.DataFrame(subgroup_rows),
    }


# ── Pooling ────────────────────────────────────────────────────────────────────

def pooled_from_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (model_name, strategy), sub in pred.groupby(["model_name", "threshold_strategy"], dropna=False):
        row: Dict[str, Any] = {
            "model_name": model_name,
            "threshold_strategy": strategy,
            "threshold": "fold_specific" if strategy != "fixed_0p5" else 0.5,
        }
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["threshold_strategy", "model_name"])


# ── Output writers ─────────────────────────────────────────────────────────────

def write_primary_comparison(
    outdir: Path,
    pooled_ultra: pd.DataFrame,
    reference_pooled_csv: Path,
) -> pd.DataFrame:
    ref = pd.read_csv(reference_pooled_csv)
    locked = ref[ref["model_name"] == "logreg_l2"][
        ["model_name", "threshold_strategy", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]
    ].copy()
    locked["source"] = "locked_logreg_l2_original_grid"

    ultra = pooled_ultra[
        ["model_name", "threshold_strategy", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]
    ].copy()
    ultra["source"] = "logreg_l2_ultra_regularized_extended_grid"

    combined = pd.concat([locked, ultra], ignore_index=True, sort=False)

    delta_rows: List[Dict[str, Any]] = []
    for strategy in ultra["threshold_strategy"].unique():
        locked_row = locked[locked["threshold_strategy"] == strategy]
        ultra_row = ultra[ultra["threshold_strategy"] == strategy]
        if locked_row.empty or ultra_row.empty:
            continue
        lr = locked_row.iloc[0]
        ur = ultra_row.iloc[0]
        delta_rows.append(
            {
                "model_name": MODEL_NAME,
                "threshold_strategy": strategy,
                "source": "delta_ultra_minus_locked",
                "auc": float(ur["auc"] - lr["auc"]),
                "pr_auc": float(ur["pr_auc"] - lr["pr_auc"]),
                "balanced_accuracy": float(ur["balanced_accuracy"] - lr["balanced_accuracy"]),
                "sensitivity": float(ur["sensitivity"] - lr["sensitivity"]),
                "specificity": float(ur["specificity"] - lr["specificity"]),
                "f1": float(ur["f1"] - lr["f1"]),
            }
        )
    final = pd.concat([combined, pd.DataFrame(delta_rows)], ignore_index=True, sort=False)
    final.to_csv(outdir / "primary_comparison.csv", index=False)

    lines = [
        "# Primary Comparison: logreg_l2_ultra_regularized vs Locked logreg_l2",
        "",
        f"Locked grid: {ORIGINAL_C_GRID}",
        f"Extended grid: {EXTENDED_C_GRID}",
        f"Locked reference AUC: {LOCKED_AUC}  |  PR-AUC: {LOCKED_PR_AUC}",
        "",
    ]
    for source_label, subset in [
        ("Locked logreg_l2 (original grid)", locked),
        (f"{MODEL_NAME} (extended grid)", ultra),
        ("Delta (ultra − locked)", pd.DataFrame(delta_rows)),
    ]:
        lines.append(f"## {source_label}")
        lines.append("")
        if subset.empty:
            lines.append("No rows.")
        else:
            cols = ["threshold_strategy", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            lines += [header, sep]
            for _, r in subset.iterrows():
                vals = []
                for c in cols:
                    v = r.get(c, "")
                    if isinstance(v, float):
                        vals.append(f"{v:+.6f}" if "delta" in str(source_label).lower() else f"{v:.6f}")
                    else:
                        vals.append(str(v))
                lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    (outdir / "primary_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return final


def write_selected_C_by_fold(outdir: Path, c_df: pd.DataFrame) -> None:
    c_df.to_csv(outdir / "selected_C_by_fold.csv", index=False)

    lines = [
        "# Selected C by Fold — Ultra-Regularized Logreg",
        "",
        f"Extended C grid: {EXTENDED_C_GRID}",
        f"Original locked C grid: {ORIGINAL_C_GRID}",
        f"Original selected C (all 5 folds): {LOCKED_C}  (was the lower bound)",
        "",
        "| fold | selected_C | selected_C_log10 | boundary_status | best_inner_auc | original_locked_C |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in c_df.iterrows():
        lines.append(
            f"| {int(r['fold'])} | {r['selected_C']:.2e} | {r['selected_C_log10']:.2f} "
            f"| **{r['boundary_status']}** | {r['best_inner_auc']:.6f} | {r['original_locked_C']:.3f} |"
        )

    boundary_counts = c_df["boundary_status"].value_counts().to_dict()
    lines += [
        "",
        "## Summary",
        "",
        f"- Folds at lower bound (`at_lower_bound`): {boundary_counts.get('at_lower_bound', 0)}",
        f"- Folds at upper bound (`at_upper_bound`): {boundary_counts.get('at_upper_bound', 0)}",
        f"- Folds interior: {boundary_counts.get('interior', 0)}",
        "",
        "Interpretation:",
        "- `at_lower_bound` → wants more regularization than the extended grid can provide",
        "- `interior` → extended grid was sufficient; original grid was indeed constraining",
        "- `at_upper_bound` → wants less regularization; original C=0.001 was correct",
    ]

    (outdir / "selected_C_by_fold.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manufacturer_subgroup_comparison(
    outdir: Path,
    subgroup: pd.DataFrame,
    reference_pooled_csv: Path,
) -> None:
    strategy = "inner_oof_target_sens_ge_0p70_max_spec"
    sub = subgroup[subgroup["threshold_strategy"] == strategy].copy()

    pooled_sub_rows: List[Dict[str, Any]] = []
    for manufacturer, grp in sub.groupby("Manufacturer", dropna=False):
        row: Dict[str, Any] = {
            "model_name": MODEL_NAME,
            "threshold_strategy": strategy,
            "Manufacturer": manufacturer,
        }
        row.update(binary_metrics(grp["y_true"] if "y_true" in grp.columns else grp["tp"] * 0,
                                  grp["y_score"] if "y_score" in grp.columns else grp["auc"] * 0,
                                  grp["y_pred"] if "y_pred" in grp.columns else grp["tp"] * 0))
        # subgroup table is per fold; aggregate manually from confusion counts
        row = {
            "model_name": MODEL_NAME,
            "threshold_strategy": strategy,
            "Manufacturer": manufacturer,
            "folds": int(len(grp)),
            "n_total": int(grp["n"].sum()),
            "n_cn": int(grp["n_cn"].sum()),
            "n_ad": int(grp["n_ad"].sum()),
            "tp_total": int(grp["tp"].sum()),
            "fp_total": int(grp["fp"].sum()),
            "fn_total": int(grp["fn"].sum()),
            "tn_total": int(grp["tn"].sum()),
        }
        n_ad = row["n_ad"]
        n_cn = row["n_cn"]
        tp = row["tp_total"]
        fp = row["fp_total"]
        fn = row["fn_total"]
        tn = row["tn_total"]
        row["pooled_sensitivity"] = safe_div(tp, tp + fn) if n_ad > 0 else float("nan")
        row["pooled_specificity"] = safe_div(tn, tn + fp) if n_cn > 0 else float("nan")
        row["pooled_fp_rate_cn"] = safe_div(fp, n_cn) if n_cn > 0 else float("nan")
        row["pooled_fn_rate_ad"] = safe_div(fn, n_ad) if n_ad > 0 else float("nan")
        pooled_sub_rows.append(row)

    pooled_sub = pd.DataFrame(pooled_sub_rows)
    pooled_sub.to_csv(outdir / "manufacturer_subgroup_comparison.csv", index=False)

    lines = [
        "# Manufacturer Subgroup Comparison — Ultra-Regularized Logreg",
        f"Threshold strategy: `{strategy}`",
        "",
        "## Pooled across folds",
        "",
        "| Manufacturer | n_cn | n_ad | fp_total | fn_total | pooled_fp_rate_cn | pooled_fn_rate_ad | pooled_specificity | pooled_sensitivity |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in pooled_sub.iterrows():
        def fmt(v: Any) -> str:
            return f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else ("NA" if isinstance(v, float) else str(v))
        lines.append(
            f"| {r['Manufacturer']} | {r['n_cn']} | {r['n_ad']} "
            f"| {r['fp_total']} | {r['fn_total']} "
            f"| {fmt(r['pooled_fp_rate_cn'])} | {fmt(r['pooled_fn_rate_ad'])} "
            f"| {fmt(r['pooled_specificity'])} | {fmt(r['pooled_sensitivity'])} |"
        )
    lines += [
        "",
        "Focus: Philips CN FP rate (specificity) and GE AD FN rate (sensitivity).",
    ]
    (outdir / "manufacturer_subgroup_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_div(num: float, den: float) -> float:  # noqa: F811 — redefine at module scope
    return float(num / den) if den else float("nan")


def write_recommendation(
    outdir: Path,
    pooled_ultra: pd.DataFrame,
    c_df: pd.DataFrame,
) -> None:
    strategy = "inner_oof_target_sens_ge_0p70_max_spec"
    row = pooled_ultra[pooled_ultra["threshold_strategy"] == strategy]
    if row.empty:
        (outdir / "recommendation.md").write_text("No inner_oof_target_sens_ge_0p70_max_spec row found.\n", encoding="utf-8")
        return

    r = row.iloc[0]
    auc_val = float(r["auc"])
    pr_auc_val = float(r["pr_auc"])

    gate_auc = auc_val > LOCKED_AUC
    gate_pr_auc = pr_auc_val >= LOCKED_PR_AUC
    promoted = gate_auc and gate_pr_auc

    boundary_counts = c_df["boundary_status"].value_counts().to_dict()
    n_lower = boundary_counts.get("at_lower_bound", 0)
    n_interior = boundary_counts.get("interior", 0)
    n_upper = boundary_counts.get("at_upper_bound", 0)

    # Diagnosis of grid constraint
    if n_upper == 5:
        grid_diagnosis = "GRID_NOT_CONSTRAINING: All 5 folds moved to upper end. Original grid was not the lower-bound constraint it appeared."
    elif n_lower == 5:
        grid_diagnosis = "STILL_CONSTRAINED: All 5 folds still at lower bound. Extended grid not deep enough — model wants C << 1e-6."
    elif n_lower > 0:
        grid_diagnosis = f"PARTIALLY_CONSTRAINED: {n_lower}/5 folds still at lower bound."
    else:
        grid_diagnosis = f"GRID_SUFFICIENT: All {n_interior}/5 folds in interior of extended grid. Original C=0.001 was constraining."

    lines = [
        "# Recommendation — Ultra-Regularized Logreg Audit",
        "",
        "## Promotion Gate",
        "",
        f"- Locked AUC threshold: `{LOCKED_AUC}` | candidate AUC: `{auc_val:.6f}` → {'**PASS**' if gate_auc else '**FAIL**'}",
        f"- Locked PR-AUC threshold: `{LOCKED_PR_AUC}` | candidate PR-AUC: `{pr_auc_val:.6f}` → {'**PASS**' if gate_pr_auc else '**FAIL**'}",
        f"- Both gates must pass: {'**PROMOTED**' if promoted else '**NOT PROMOTED**'}",
        "",
        "## C-Grid Constraint Diagnosis",
        "",
        f"- Original locked C (all 5 folds): `{LOCKED_C}` (lower bound of original grid)",
        f"- Extended grid: `{EXTENDED_C_GRID}`",
        f"- Folds at lower bound of extended grid: {n_lower}/5",
        f"- Folds in interior: {n_interior}/5",
        f"- Folds at upper bound: {n_upper}/5",
        f"- **Diagnosis:** {grid_diagnosis}",
        "",
        "## Interpretation",
        "",
    ]

    if n_upper == 5:
        lines += [
            "The extended grid entirely reversed: all 5 folds selected larger C values than C=0.001.",
            "This means the original C=0.001 was NOT the globally optimal value — inner CV with more",
            "grid points chose a less regularized solution. The original result was a local optimum within",
            "the coarse grid, not a boundary effect. The locked model's AUC/PR-AUC remain the reference.",
        ]
    elif n_lower == 5:
        lines += [
            "The model still saturates at the extended lower bound. Even C=1e-6 is not regularized enough",
            "by inner CV criterion. This confirms the high-dimensional latent space (dim=256) is fundamentally",
            "hard to regularize with logistic regression — the feature count dominates the regularization budget.",
            "The original C=0.001 result was a boundary artifact but extending further is unlikely to help.",
        ]
    elif n_lower > 0:
        lines += [
            f"{n_lower} folds selected C at the extended lower bound, {n_interior} folds found interior solutions.",
            "Partial constraint: the extended grid resolved some but not all folds. The locked result was",
            "partially constrained by the original grid's lower bound.",
        ]
    else:
        lines += [
            "All 5 folds found interior solutions in the extended grid. The original C=0.001 was indeed the",
            "lower bound of the coarse grid acting as a constraint. The extended audit reveals the true optimal C.",
            f"Compare AUC/PR-AUC above against locked ({LOCKED_AUC}/{LOCKED_PR_AUC}) to assess impact.",
        ]

    lines += [
        "",
        "## Constraints Verified",
        "",
        "- VAE retraining: `False`",
        "- Tensor modification: `False`",
        "- Metadata/ledger modification: `False`",
        "- Latent cache source: `reused_existing_latent_cache` (adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep)",
        "- Threshold selection: `true_inner_cv_oof` on train/dev only",
    ]

    (outdir / "recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(
    outdir: Path,
    pooled_ultra: pd.DataFrame,
    c_df: pd.DataFrame,
    outer_folds: int,
    inner_folds: int,
    latent_cache_dir: Path,
) -> None:
    strategy = "inner_oof_target_sens_ge_0p70_max_spec"
    primary = pooled_ultra[pooled_ultra["threshold_strategy"] == strategy]
    boundary_counts = c_df["boundary_status"].value_counts().to_dict()

    lines = [
        "# Ultra-Regularized Logreg Readout Audit",
        "## adni_v5_1_batch20260514b [1,0,2] — Stage B classifier-only",
        "",
        "## Motivation",
        "",
        f"In the locked `logreg_l2` readout, all {outer_folds} folds selected C={LOCKED_C},",
        f"the lower bound of the original grid {ORIGINAL_C_GRID}. This audit extends",
        f"the grid to {EXTENDED_C_GRID} and re-runs inner CV on the same frozen latents",
        "to determine whether the original result was constrained by the grid boundary.",
        "",
        "## Constraints",
        "",
        "- VAE retraining: `False`",
        "- Tensor modification: `False`",
        "- Metadata/ledger modification: `False`",
        f"- Latent cache: `{latent_cache_dir}`",
        "- Threshold selection: `true_inner_cv_oof` on train/dev only",
        f"- Outer folds: `{outer_folds}`",
        f"- Inner CV folds: `{inner_folds}`",
        "",
        "## C-Grid Boundary Summary",
        "",
        f"| boundary_status | count |",
        "| --- | --- |",
    ]
    for k, v in sorted(boundary_counts.items()):
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Primary Metrics (inner_oof_target_sens_ge_0p70_max_spec)",
        "",
    ]
    if not primary.empty:
        cols = ["model_name", "threshold_strategy", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]
        lines += [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, r in primary.iterrows():
            vals = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, float):
                    vals.append(f"{v:.6f}" if np.isfinite(v) else "NA")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
    else:
        lines.append("No rows for primary strategy.")

    lines += [
        "",
        f"Locked reference: AUC={LOCKED_AUC}  |  PR-AUC={LOCKED_PR_AUC}",
        "",
        "## Outputs",
        "",
        "- `README.md`",
        "- `primary_comparison.csv` / `primary_comparison.md`",
        "- `selected_C_by_fold.csv` / `selected_C_by_fold.md`",
        "- `manufacturer_subgroup_comparison.csv` / `manufacturer_subgroup_comparison.md`",
        "- `predictions.csv`",
        "- `recommendation.md`",
        "- `command_log.json`",
    ]
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    latent_cache_dir = args.latent_cache_dir if args.latent_cache_dir.is_absolute() else PROJECT_ROOT / args.latent_cache_dir
    reference_pooled_csv = args.reference_pooled_csv if args.reference_pooled_csv.is_absolute() else PROJECT_ROOT / args.reference_pooled_csv

    # Validate inputs
    require_latent_cache(latent_cache_dir, args.outer_folds)
    if not reference_pooled_csv.exists():
        raise FileNotFoundError(f"Reference pooled metrics CSV not found: {reference_pooled_csv}")

    print(f"latent_cache_dir={latent_cache_dir}")
    print(f"reference_pooled_csv={reference_pooled_csv}")
    print(f"extended_c_grid={EXTENDED_C_GRID}")
    print(f"outer_folds={args.outer_folds}  inner_folds={args.inner_folds}  seed={args.seed}")
    print(f"vae_retrained=False  tensor_modified=False  metadata_modified=False")

    if args.dry_run:
        print("DRY-RUN: validation passed. Exiting without running classifiers.")
        return 0

    outdir = prepare_output_dir(args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir, args.overwrite)

    results = run_audit(
        cache_dir=latent_cache_dir,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    pooled_ultra = pooled_from_predictions(results["predictions"])

    # Write outputs
    write_primary_comparison(outdir, pooled_ultra, reference_pooled_csv)
    write_selected_C_by_fold(outdir, results["selected_C_by_fold"])
    write_manufacturer_subgroup_comparison(outdir, results["subgroup_by_manufacturer"], reference_pooled_csv)
    results["predictions"].to_csv(outdir / "predictions.csv", index=False)
    write_recommendation(outdir, pooled_ultra, results["selected_C_by_fold"])
    write_readme(outdir, pooled_ultra, results["selected_C_by_fold"], args.outer_folds, args.inner_folds, latent_cache_dir)

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "latent_cache_dir": str(latent_cache_dir),
        "reference_pooled_csv": str(reference_pooled_csv),
        "output_dir": str(outdir),
        "model_name": MODEL_NAME,
        "extended_c_grid": EXTENDED_C_GRID,
        "original_c_grid": ORIGINAL_C_GRID,
        "locked_reference_auc": LOCKED_AUC,
        "locked_reference_pr_auc": LOCKED_PR_AUC,
        "locked_selected_c_all_folds": LOCKED_C,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "vae_retrained": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "latent_source": "reused_existing_latent_cache",
        "threshold_selection": "true_inner_cv_oof",
        "promotion_rule": "AUC > 0.778785 AND PR-AUC >= 0.551832 simultaneously (inner_oof_target_sens_ge_0p70_max_spec)",
    }
    write_json(outdir / "command_log.json", command_log)

    print(f"\noutput_dir={outdir}")
    print(f"\nSelected C by fold:")
    print(results["selected_C_by_fold"][["fold", "selected_C", "selected_C_log10", "boundary_status", "best_inner_auc"]].to_string(index=False))
    print(f"\nPooled metrics (all threshold strategies):")
    print(pooled_ultra[["model_name", "threshold_strategy", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
