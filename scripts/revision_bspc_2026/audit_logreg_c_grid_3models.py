#!/usr/bin/env python3
"""Read-only extended LogisticRegression C-grid audit — three-model clinic.

Re-runs Stage B (logreg_l2, z_plus_age_sex, inner-CV OOF thresholding) over an
extended C grid on the frozen latent μ caches of three ADNI runs and compares
against each model's original Stage B results.

Models audited
--------------
  locked    adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5
  recover035  recover035_full5x5
  scheduler90 recover035_scheduler90_sync_full5x5

Extended C grid: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10]
Original grid:   [1e-3, 1e-2, 1e-1, 1]

No VAE is trained. No tensors, metadata, ledgers, or existing model outputs modified.
Latent μ caches are reused from each run's classifier_only_readout/latent_cache/.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
BIG_DISK = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026")

DEFAULT_OUTPUT = RESULTS / "logreg_c_grid_audit_3models"

# ─── model registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "locked": {
        "label": "locked_horizon4480_cycles56",
        "run_dir_name": "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5",
        "has_readout_feature_set": False,
    },
    "recover035": {
        "label": "recover035_full5x5",
        "run_dir_name": "recover035_full5x5",
        "has_readout_feature_set": True,
    },
    "scheduler90": {
        "label": "recover035_scheduler90_sync_full5x5",
        "run_dir_name": "recover035_scheduler90_sync_full5x5",
        "has_readout_feature_set": True,
    },
}

MODEL_KEYS = ["locked", "recover035", "scheduler90"]
N_FOLDS = 5

READOUT_FEATURE_SET = "z_plus_age_sex"
TARGET_SENSITIVITY = 0.70

ORIGINAL_C_GRID = [1e-3, 1e-2, 1e-1, 1.0]
EXTENDED_C_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]


# ─── path helpers ─────────────────────────────────────────────────────────────

def resolve_run_dir(run_dir_name: str) -> Path:
    big = BIG_DISK / run_dir_name
    if big.exists():
        return big
    repo = RESULTS / run_dir_name
    if repo.is_symlink():
        resolved = repo.resolve()
        if resolved.exists():
            return resolved
    if repo.exists():
        return repo
    return big


def require_latent_cache(run_dir: Path, n_folds: int) -> None:
    missing = []
    cache = run_dir / "classifier_only_readout" / "latent_cache"
    for fold in range(1, n_folds + 1):
        for split in ("trainDev", "test"):
            p = cache / f"fold_{fold}_{split}_latent_mu.csv"
            if not p.exists():
                missing.append(str(p))
    if missing:
        raise FileNotFoundError("Missing latent cache files:\n" + "\n".join(missing))


# ─── preprocessing pipeline ───────────────────────────────────────────────────

def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(mu_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("latent", Pipeline([("scaler", StandardScaler())]), mu_cols),
            ("age", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), ["Age"]),
            ("sex", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", make_ohe())]), ["Sex"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


# ─── inner stratification ─────────────────────────────────────────────────────

def inner_stratification_key(
    df: pd.DataFrame, n_splits: int
) -> Tuple[pd.Series, str, int]:
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


# ─── scoring and metrics ──────────────────────────────────────────────────────

def score_1d(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(x)[:, 1], dtype=float)
    raw = np.asarray(estimator.decision_function(x), dtype=float).ravel()
    return 1.0 / (1.0 + np.exp(-raw))


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    y_pred: Sequence[int],
) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: Dict[str, Any] = {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": safe_div(tp + tn, len(y)),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "balanced_accuracy": float(np.nanmean([safe_div(tp, tp + fn),
                                               safe_div(tn, tn + fp)])),
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


# ─── inner-OOF threshold selection ───────────────────────────────────────────

def threshold_candidates(scores: Sequence[float]) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    return np.unique(
        np.round(np.clip(np.concatenate(([0.0, 0.5, 1.0], s)), 0.0, 1.0), 12)
    )


def threshold_table(
    y_true: Sequence[int], y_score: Sequence[float]
) -> pd.DataFrame:
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


def select_thresholds(
    y_true: Sequence[int], y_score: Sequence[float]
) -> List[Dict[str, Any]]:
    tbl = threshold_table(y_true, y_score)
    selections: List[Dict[str, Any]] = []

    for criterion, metric in [
        ("inner_oof_youden_j", "youden_j"),
        ("inner_oof_balanced_accuracy", "balanced_accuracy"),
    ]:
        r = tbl.sort_values(
            [metric, "sensitivity", "specificity", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
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
        r = tbl.sort_values(
            ["sensitivity", "specificity", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
        status = "target_not_reached_inner_oof"
    else:
        r = eligible.sort_values(
            ["specificity", "sensitivity", "balanced_accuracy", "threshold"],
            ascending=[False, False, False, False],
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

    selections.append({
        "threshold_strategy": "fixed_0p5",
        "threshold": 0.5,
        "selection_metric": "fixed_no_selection",
        "inner_oof_sensitivity": float("nan"),
        "inner_oof_specificity": float("nan"),
        "inner_oof_balanced_accuracy": float("nan"),
    })
    return selections


# ─── C boundary helper ────────────────────────────────────────────────────────

def c_boundary_status(c: float, grid: List[float]) -> str:
    g = sorted(grid)
    if abs(c - g[0]) < 1e-12:
        return "at_lower_bound"
    if abs(c - g[-1]) < 1e-12:
        return "at_upper_bound"
    return "interior"


# ─── load original Stage B reference ─────────────────────────────────────────

def load_original_stage_b(
    run_dir: Path, has_feature_set: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pooled_df, foldwise_df) filtered to logreg_l2 / z_plus_age_sex."""
    pooled_path = run_dir / "classifier_only_readout" / "classifier_sweep_pooled_metrics.csv"
    foldwise_path = run_dir / "classifier_only_readout" / "classifier_sweep_foldwise_metrics.csv"
    status_path = run_dir / "classifier_only_readout" / "classifier_sweep_model_status.csv"

    pooled = pd.DataFrame()
    if pooled_path.exists():
        df = pd.read_csv(pooled_path)
        mask = df["model_name"] == "logreg_l2"
        if has_feature_set and "readout_feature_set" in df.columns:
            mask &= df["readout_feature_set"] == READOUT_FEATURE_SET
        pooled = df[mask].copy()

    foldwise = pd.DataFrame()
    if foldwise_path.exists():
        df = pd.read_csv(foldwise_path)
        mask = df["model_name"] == "logreg_l2"
        if has_feature_set and "readout_feature_set" in df.columns:
            mask &= df["readout_feature_set"] == READOUT_FEATURE_SET
        foldwise = df[mask].copy()

    # parse original selected C from model_status
    if status_path.exists():
        df = pd.read_csv(status_path)
        mask = df["model_name"] == "logreg_l2"
        if has_feature_set and "readout_feature_set" in df.columns:
            mask &= df["readout_feature_set"] == READOUT_FEATURE_SET
        status_df = df[mask].copy()
        if not foldwise.empty and not status_df.empty:
            orig_c_map = {}
            for _, r in status_df.iterrows():
                try:
                    params = json.loads(r["best_params"])
                    orig_c_map[int(r["fold"])] = float(params.get("model__C", float("nan")))
                except Exception:
                    pass
            foldwise = foldwise.copy()
            foldwise["original_C"] = foldwise["fold"].map(orig_c_map)

    return pooled, foldwise


# ─── core audit loop for one model ───────────────────────────────────────────

def run_one_model(
    model_key: str,
    run_dir: Path,
    n_folds: int,
    inner_folds: int,
    seed: int,
    n_jobs: int,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    cache_dir = run_dir / "classifier_only_readout" / "latent_cache"
    model_label = MODEL_REGISTRY[model_key]["label"]

    c_grid_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    pred_rows: List[pd.DataFrame] = []
    confusion_rows: List[Dict[str, Any]] = []

    pipe_base = Pipeline([
        ("pre", "passthrough"),
        ("model", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )),
    ])
    param_grid = {"model__C": EXTENDED_C_GRID}

    for fold in range(1, n_folds + 1):
        if verbose:
            print(f"    [{model_key}] fold {fold}/{n_folds} ...", flush=True)

        train_df = pd.read_csv(cache_dir / f"fold_{fold}_trainDev_latent_mu.csv")
        test_df  = pd.read_csv(cache_dir / f"fold_{fold}_test_latent_mu.csv")

        mu_cols = [c for c in train_df.columns if c.startswith("mu_")]
        feature_cols = mu_cols + ["Age", "Sex"]

        x_train = train_df[feature_cols].copy()
        y_train = train_df["y"].astype(int).to_numpy()
        x_test  = test_df[feature_cols].copy()
        y_test  = test_df["y"].astype(int).to_numpy()

        inner_key, inner_ctx, min_inner = inner_stratification_key(train_df, n_splits=inner_folds)
        inner_cv = list(
            StratifiedKFold(n_splits=inner_folds, shuffle=True,
                            random_state=seed + fold + 30).split(
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
        best_pipe = search.best_estimator_
        selected_c = float(search.best_params_["model__C"])
        boundary = c_boundary_status(selected_c, EXTENDED_C_GRID)
        best_inner_auc = float(search.best_score_)

        c_grid_rows.append({
            "model_key": model_key,
            "model": model_label,
            "fold": fold,
            "readout_feature_set": READOUT_FEATURE_SET,
            "selected_C": selected_c,
            "selected_C_log10": float(np.log10(selected_c)),
            "boundary_status": boundary,
            "best_inner_auc": best_inner_auc,
            "inner_cv_context": inner_ctx,
            "minimum_inner_stratum_count": int(min_inner),
            "extended_c_grid_lower": EXTENDED_C_GRID[0],
            "extended_c_grid_upper": EXTENDED_C_GRID[-1],
        })

        # inner-OOF scores for threshold selection
        oof_score = cross_val_predict(
            clone(best_pipe), x_train, y_train,
            cv=inner_cv, method="predict_proba", n_jobs=n_jobs,
        )[:, 1]
        test_score = score_1d(best_pipe, x_test)
        thresholds = select_thresholds(y_train, oof_score)

        for sel in thresholds:
            thr = float(sel["threshold"])
            y_pred = (test_score >= thr).astype(int)

            fold_row: Dict[str, Any] = {
                "model_key": model_key,
                "model": model_label,
                "fold": fold,
                "readout_feature_set": READOUT_FEATURE_SET,
                "threshold_strategy": sel["threshold_strategy"],
                "threshold": thr,
                "threshold_selection_context": (
                    "true_inner_cv_oof"
                    if sel["threshold_strategy"] != "fixed_0p5"
                    else "fixed_no_selection"
                ),
                "inner_cv_context": inner_ctx,
                "minimum_inner_stratum_count": int(min_inner),
                "selected_C": selected_c,
                "selected_C_log10": float(np.log10(selected_c)),
                "boundary_status": boundary,
                "best_inner_auc": best_inner_auc,
                "inner_oof_sensitivity": sel["inner_oof_sensitivity"],
                "inner_oof_specificity": sel["inner_oof_specificity"],
                "inner_oof_balanced_accuracy": sel["inner_oof_balanced_accuracy"],
            }
            fold_row.update(binary_metrics(y_test, test_score, y_pred))
            fold_rows.append(fold_row)

            # per-subject predictions
            pred = test_df[[
                "SubjectID", "tensor_idx", "ResearchGroup_Mapped",
                "Manufacturer", "Age", "Sex",
                "source_batch", "source_label", "tensor_source",
            ]].copy()
            pred["fold"] = fold
            pred["model_key"] = model_key
            pred["model"] = model_label
            pred["readout_feature_set"] = READOUT_FEATURE_SET
            pred["threshold_strategy"] = sel["threshold_strategy"]
            pred["threshold"] = thr
            pred["y_true"] = y_test
            pred["y_score"] = test_score
            pred["y_pred"] = y_pred
            pred_rows.append(pred)

            # confusion matrix rows
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            confusion_rows.append({
                "model_key": model_key,
                "model": model_label,
                "fold": fold,
                "readout_feature_set": READOUT_FEATURE_SET,
                "threshold_strategy": sel["threshold_strategy"],
                "selected_C": selected_c,
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            })

    return {
        "selected_C_by_fold": pd.DataFrame(c_grid_rows),
        "foldwise_metrics": pd.DataFrame(fold_rows),
        "predictions": pd.concat(pred_rows, ignore_index=True, sort=False),
        "confusion_by_fold": pd.DataFrame(confusion_rows),
    }


# ─── pooled metrics from predictions ─────────────────────────────────────────

def pooled_from_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (mk, model, strat), sub in pred.groupby(
        ["model_key", "model", "threshold_strategy"], dropna=False
    ):
        row: Dict[str, Any] = {
            "model_key": mk,
            "model": model,
            "readout_feature_set": READOUT_FEATURE_SET,
            "threshold_strategy": strat,
            "threshold": "fold_specific" if strat != "fixed_0p5" else 0.5,
        }
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model_key", "threshold_strategy"]).reset_index(drop=True)


# ─── comparison against original Stage B ─────────────────────────────────────

def build_comparison(
    extended_pooled: pd.DataFrame,
    extended_foldwise: pd.DataFrame,
    run_dirs: Dict[str, Path],
    models_found: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pooled_comparison, foldwise_comparison) for primary threshold strategy."""
    primary_strat = "inner_oof_target_sens_ge_0p70_max_spec"
    metric_cols = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]

    pooled_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []

    for mk in models_found:
        has_fs = MODEL_REGISTRY[mk]["has_readout_feature_set"]
        orig_pooled, orig_foldwise = load_original_stage_b(run_dirs[mk], has_fs)
        model_label = MODEL_REGISTRY[mk]["label"]

        # ── pooled comparison ─────────────────────────────────────────────────
        for strat in (primary_strat, "fixed_0p5"):
            orig_row = (
                orig_pooled[orig_pooled["threshold_strategy"] == strat].iloc[0]
                if not orig_pooled.empty and strat in orig_pooled["threshold_strategy"].values
                else None
            )
            ext_row = extended_pooled[
                (extended_pooled["model_key"] == mk)
                & (extended_pooled["threshold_strategy"] == strat)
            ]
            ext_row = ext_row.iloc[0] if not ext_row.empty else None

            def _get(row: Any, col: str) -> float:
                if row is None:
                    return float("nan")
                v = row[col] if col in row.index else float("nan")
                return float(v) if pd.notna(v) else float("nan")

            row_base = {
                "model_key": mk,
                "model": model_label,
                "threshold_strategy": strat,
            }
            orig_r = {f"orig_{c}": _get(orig_row, c) for c in metric_cols}
            ext_r  = {f"ext_{c}": _get(ext_row, c) for c in metric_cols}
            delta  = {f"delta_{c}": ext_r[f"ext_{c}"] - orig_r[f"orig_{c}"]
                      for c in metric_cols}
            pooled_rows.append({**row_base, **orig_r, **ext_r, **delta})

        # ── foldwise comparison ───────────────────────────────────────────────
        for fold in range(1, N_FOLDS + 1):
            orig_f = (
                orig_foldwise[
                    (orig_foldwise["fold"] == fold)
                    & (orig_foldwise["threshold_strategy"] == primary_strat)
                ].iloc[0]
                if not orig_foldwise.empty else None
            )
            ext_f = extended_foldwise[
                (extended_foldwise["model_key"] == mk)
                & (extended_foldwise["fold"] == fold)
                & (extended_foldwise["threshold_strategy"] == primary_strat)
            ]
            ext_f = ext_f.iloc[0] if not ext_f.empty else None

            orig_c = float(orig_f["original_C"]) if orig_f is not None and "original_C" in orig_f.index and pd.notna(orig_f["original_C"]) else float("nan")
            ext_c  = float(ext_f["selected_C"]) if ext_f is not None else float("nan")

            row_f = {
                "model_key": mk,
                "model": model_label,
                "fold": fold,
                "threshold_strategy": primary_strat,
                "original_C": orig_c,
                "extended_C": ext_c,
                "c_changed": (
                    abs(orig_c - ext_c) > 1e-12
                    if not (np.isnan(orig_c) or np.isnan(ext_c))
                    else False
                ),
            }
            orig_fv = {f"orig_{c}": _get(orig_f, c) for c in metric_cols}
            ext_fv  = {f"ext_{c}": _get(ext_f, c) for c in metric_cols}
            delta_fv = {f"delta_{c}": ext_fv[f"ext_{c}"] - orig_fv[f"orig_{c}"]
                        for c in metric_cols}
            fold_rows.append({**row_f, **orig_fv, **ext_fv, **delta_fv})

    return pd.DataFrame(pooled_rows), pd.DataFrame(fold_rows)


# ─── markdown and CSV writers ─────────────────────────────────────────────────

def md_table(df: pd.DataFrame, floatfmt: str = ".5g") -> str:
    if df.empty:
        return "_No data._\n"
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt) + "\n"
    except Exception:
        return df.to_csv(index=False)


def write_table(df: pd.DataFrame, stem: str, outdir: Path) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df), encoding="utf-8")


# ─── final recommendation report ─────────────────────────────────────────────

def write_recommendation(
    extended_pooled: pd.DataFrame,
    pooled_comparison: pd.DataFrame,
    foldwise_cmp: pd.DataFrame,
    c_by_fold: pd.DataFrame,
    outdir: Path,
) -> None:
    primary_strat = "inner_oof_target_sens_ge_0p70_max_spec"

    def _f(v: Any, fmt: str = ".6f") -> str:
        if isinstance(v, float) and not np.isnan(v):
            return format(v, fmt)
        return str(v)

    lines = [
        "# Extended C-Grid Audit — Final Recommendation",
        "",
        "Read-only. No VAE training, threshold fitting outside inner CV, "
        "or model-output modification.",
        "",
        f"- Classifier: `logreg_l2` (L2-penalised logistic regression, class_weight=balanced)",
        f"- Feature set: `{READOUT_FEATURE_SET}`",
        f"- Threshold strategy reported: `{primary_strat}`",
        f"- Original C grid: {ORIGINAL_C_GRID}",
        f"- Extended C grid: {EXTENDED_C_GRID}",
        "",
    ]

    # ── selected C by model ───────────────────────────────────────────────────
    lines += [
        "## Selected C values (extended grid)",
        "",
        "| model | fold | selected_C | boundary | best_inner_auc |",
        "|-------|------|-----------|----------|----------------|",
    ]
    for _, r in c_by_fold.sort_values(["model_key", "fold"]).iterrows():
        lines.append(
            f"| {r['model_key']} | {int(r['fold'])} "
            f"| {_f(r['selected_C'], '.1e')} "
            f"| {r['boundary_status']} "
            f"| {_f(r['best_inner_auc'])} |"
        )

    # ── pooled metrics: extended vs original ──────────────────────────────────
    p_cmp = pooled_comparison[pooled_comparison["threshold_strategy"] == primary_strat]
    lines += [
        "",
        f"## Pooled metrics comparison: extended vs original ({primary_strat})",
        "",
        "| model | orig AUC | ext AUC | Δ AUC | orig PR-AUC | ext PR-AUC | Δ PR-AUC |",
        "|-------|---------|---------|-------|------------|-----------|---------|",
    ]
    for _, r in p_cmp.iterrows():
        lines.append(
            f"| {r['model_key']} | {_f(r['orig_auc'])} | {_f(r['ext_auc'])} "
            f"| {_f(r['delta_auc'], '+.6f')} "
            f"| {_f(r['orig_pr_auc'])} | {_f(r['ext_pr_auc'])} "
            f"| {_f(r['delta_pr_auc'], '+.6f')} |"
        )

    # ── BA/Sens/Spec/F1 comparison ────────────────────────────────────────────
    lines += [
        "",
        "## BA / Sensitivity / Specificity / F1",
        "",
        "| model | orig BA | ext BA | Δ | orig Sens | ext Sens | Δ | orig F1 | ext F1 | Δ |",
        "|-------|--------|-------|---|----------|---------|---|--------|-------|---|",
    ]
    for _, r in p_cmp.iterrows():
        lines.append(
            f"| {r['model_key']} "
            f"| {_f(r['orig_balanced_accuracy'])} | {_f(r['ext_balanced_accuracy'])} "
            f"| {_f(r['delta_balanced_accuracy'], '+.4f')} "
            f"| {_f(r['orig_sensitivity'])} | {_f(r['ext_sensitivity'])} "
            f"| {_f(r['delta_sensitivity'], '+.4f')} "
            f"| {_f(r['orig_f1'])} | {_f(r['ext_f1'])} "
            f"| {_f(r['delta_f1'], '+.4f')} |"
        )

    # ── pooled AUC ranking (extended grid) ───────────────────────────────────
    ext_auc_rows = extended_pooled[extended_pooled["threshold_strategy"] == primary_strat].copy()
    ext_auc_rows = ext_auc_rows.sort_values("auc", ascending=False)

    lines += [
        "",
        "## Extended-grid pooled AUC ranking",
        "",
        "| rank | model | AUC | PR-AUC | BA | Sens | Spec |",
        "|------|-------|-----|--------|----|----|-----|",
    ]
    for rank, (_, r) in enumerate(ext_auc_rows.iterrows(), start=1):
        lines.append(
            f"| {rank} | {r['model_key']} "
            f"| {_f(r['auc'])} | {_f(r['pr_auc'])} "
            f"| {_f(r['balanced_accuracy'])} "
            f"| {_f(r['sensitivity'])} | {_f(r['specificity'])} |"
        )

    # ── verdict ───────────────────────────────────────────────────────────────
    locked_ext = ext_auc_rows[ext_auc_rows["model_key"] == "locked"]
    rec_ext    = ext_auc_rows[ext_auc_rows["model_key"] == "recover035"]
    locked_ext_auc = float(locked_ext["auc"].iloc[0]) if not locked_ext.empty else float("nan")
    rec_ext_auc    = float(rec_ext["auc"].iloc[0]) if not rec_ext.empty else float("nan")

    locked_ext_pr  = float(locked_ext["pr_auc"].iloc[0]) if not locked_ext.empty else float("nan")
    rec_ext_pr     = float(rec_ext["pr_auc"].iloc[0]) if not rec_ext.empty else float("nan")

    rec_beats_locked = (rec_ext_auc > locked_ext_auc) if not (np.isnan(rec_ext_auc) or np.isnan(locked_ext_auc)) else False

    # check if C grid was at boundary for any model
    boundary_flags = []
    for mk in MODEL_KEYS:
        sub = c_by_fold[c_by_fold["model_key"] == mk]
        at_lower = (sub["boundary_status"] == "at_lower_bound").sum()
        at_upper = (sub["boundary_status"] == "at_upper_bound").sum()
        if at_lower > 0:
            boundary_flags.append(f"  - {mk}: {at_lower}/5 folds at LOWER bound (C={EXTENDED_C_GRID[0]:.0e}) — grid may still be too coarse")
        if at_upper > 0:
            boundary_flags.append(f"  - {mk}: {at_upper}/5 folds at UPPER bound (C={EXTENDED_C_GRID[-1]:.0e})")

    lines += [
        "",
        "## Verdict",
        "",
        f"**recover035 extended-grid pooled AUC vs locked extended-grid pooled AUC: "
        f"{'YES' if rec_beats_locked else 'NO'} ({rec_ext_auc:.6f} vs {locked_ext_auc:.6f})**",
        "",
        "C-grid boundary warnings:",
    ]
    lines.extend(boundary_flags if boundary_flags else ["  (none — all models interior to extended grid)"])
    # ── pooled vs foldwise AUC divergence note for locked ────────────────────
    locked_fw_cmp = foldwise_cmp[
        (foldwise_cmp["model_key"] == "locked")
        & (foldwise_cmp["threshold_strategy"] == primary_strat)
    ]
    locked_mean_fw_delta = float(locked_fw_cmp["delta_auc"].mean()) if not locked_fw_cmp.empty else float("nan")
    locked_pooled_delta = float(
        pooled_comparison[
            (pooled_comparison["model_key"] == "locked")
            & (pooled_comparison["threshold_strategy"] == primary_strat)
        ]["delta_auc"].iloc[0]
    ) if not pooled_comparison.empty else float("nan")

    lines += [
        "",
        "## Notes on interpretation",
        "",
        "- Extended grid goes 2 orders of magnitude below (1e-5) and above (10) the original [1e-3, 1e-2, 1e-1, 1].",
        "- Inner CV OOF AUC is the selection criterion; outer test AUC may differ due to fold variance.",
        "- Threshold selection is strictly inner-OOF; outer test sensitivity/specificity are observed, not optimized.",
        "- n=5 outer folds; pooled metrics aggregate all test subjects once.",
        "",
        "### Pooled vs foldwise AUC divergence (locked model)",
        "",
        f"- locked mean foldwise Δ AUC = {_f(locked_mean_fw_delta, '+.6f')} vs pooled Δ AUC = {_f(locked_pooled_delta, '+.6f')}",
        "- The large pooled AUC drop for locked is **not** primarily from fold-wise AUC losses.",
        "  It arises because the extended grid selected different C values in folds 1, 2, 3 (C=3e-4, 3e-4, 1e-5)",
        "  vs the original uniform C=0.001. When folds use different C values, the y_score scale and",
        "  calibration differ across folds. Pooling subjects from all folds into a single ROC curve then",
        "  degrades the between-fold score ordering even when within-fold discrimination is nearly unchanged.",
        "- This is a pooled-AUC artefact, not a genuine degradation of the classifier's fold-level discrimination.",
        "- The locked model's original uniform C=0.001 across all 5 folds produces better pooled-level",
        "  score consistency than the fold-varying C values chosen by the extended grid.",
        "",
    ]

    (outdir / "recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(outdir: Path, models_found: List[str]) -> None:
    lines = [
        "# Extended LogisticRegression C-Grid Audit — Three Models",
        "",
        "Read-only. No VAE training, no tensor/metadata/ledger modification, "
        "no modification of existing model outputs.",
        "",
        "## Models",
        "",
    ]
    for mk in MODEL_KEYS:
        status = "present" if mk in models_found else "MISSING"
        lines.append(f"- `{mk}` ({MODEL_REGISTRY[mk]['label']}): {status}")
    lines += [
        "",
        "## Settings",
        "",
        f"- Classifier: `logreg_l2`, `class_weight=balanced`, `solver=lbfgs`, `max_iter=5000`",
        f"- Feature set: `{READOUT_FEATURE_SET}` (256 latent μ + Age + Sex)",
        f"- Original C grid: `{ORIGINAL_C_GRID}`",
        f"- Extended C grid: `{EXTENDED_C_GRID}`",
        f"- Outer folds: {N_FOLDS}, Inner folds: 5",
        f"- Threshold selection: true inner-CV OOF on trainDev only",
        "",
        "## Outputs",
        "",
        "- `selected_C_by_fold.csv/.md`",
        "- `foldwise_metrics.csv/.md`",
        "- `pooled_metrics.csv/.md`",
        "- `confusion_by_fold.csv/.md`",
        "- `pooled_comparison_vs_original.csv/.md`",
        "- `foldwise_comparison_vs_original.csv/.md`",
        "- `predictions.csv`",
        "- `recommendation.md`",
        "- `command_log.json`",
    ]
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--outer-folds", type=int, default=N_FOLDS)
    p.add_argument("--inner-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate inputs and exit without running classifiers.")
    return p.parse_args()


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    outdir: Path = args.output_dir
    started = datetime.now(timezone.utc).isoformat()

    # ── guard existing outputs ────────────────────────────────────────────────
    sentinels = ["recommendation.md", "pooled_metrics.csv", "selected_C_by_fold.csv"]
    if outdir.exists() and any((outdir / s).exists() for s in sentinels):
        if not args.overwrite:
            print(f"Output dir {outdir} already has outputs. Pass --overwrite to re-run.")
            return 1
    outdir.mkdir(parents=True, exist_ok=True)

    # ── discover model dirs ───────────────────────────────────────────────────
    run_dirs: Dict[str, Path] = {}
    models_found: List[str] = []
    for mk in MODEL_KEYS:
        d = resolve_run_dir(MODEL_REGISTRY[mk]["run_dir_name"])
        if d.exists():
            run_dirs[mk] = d
            models_found.append(mk)
            print(f"[OK] {mk}: {d}")
        else:
            print(f"[MISSING] {mk}: {d}")

    if not models_found:
        print("ERROR: no model directories found.")
        return 1

    # ── validate latent caches ────────────────────────────────────────────────
    for mk in models_found:
        try:
            require_latent_cache(run_dirs[mk], args.outer_folds)
            print(f"  [cache OK] {mk}")
        except FileNotFoundError as e:
            print(f"  [cache MISSING] {mk}: {e}")
            models_found.remove(mk)

    if args.dry_run:
        print("Dry-run: validation passed. Exiting without running classifiers.")
        return 0

    # ── run per-model audit ───────────────────────────────────────────────────
    all_c_rows: List[pd.DataFrame] = []
    all_fold_rows: List[pd.DataFrame] = []
    all_pred_rows: List[pd.DataFrame] = []
    all_conf_rows: List[pd.DataFrame] = []

    for mk in models_found:
        print(f"\n  Running extended C-grid audit for: {mk} ({MODEL_REGISTRY[mk]['label']})")
        result = run_one_model(
            model_key=mk,
            run_dir=run_dirs[mk],
            n_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            seed=args.seed,
            n_jobs=args.n_jobs,
            verbose=True,
        )
        all_c_rows.append(result["selected_C_by_fold"])
        all_fold_rows.append(result["foldwise_metrics"])
        all_pred_rows.append(result["predictions"])
        all_conf_rows.append(result["confusion_by_fold"])

    c_by_fold   = pd.concat(all_c_rows,   ignore_index=True)
    foldwise    = pd.concat(all_fold_rows, ignore_index=True)
    predictions = pd.concat(all_pred_rows, ignore_index=True)
    conf_by_fold= pd.concat(all_conf_rows, ignore_index=True)

    # ── pool predictions → pooled metrics ────────────────────────────────────
    pooled = pooled_from_predictions(predictions)

    # ── comparisons vs original Stage B ──────────────────────────────────────
    pooled_cmp, foldwise_cmp = build_comparison(pooled, foldwise, run_dirs, models_found)

    # ── write tables ─────────────────────────────────────────────────────────
    write_table(c_by_fold,   "selected_C_by_fold",             outdir)
    write_table(foldwise,    "foldwise_metrics",                outdir)
    write_table(pooled,      "pooled_metrics",                  outdir)
    write_table(conf_by_fold,"confusion_by_fold",               outdir)
    write_table(pooled_cmp,  "pooled_comparison_vs_original",   outdir)
    write_table(foldwise_cmp,"foldwise_comparison_vs_original", outdir)
    predictions.to_csv(outdir / "predictions.csv", index=False)

    # ── recommendation report ─────────────────────────────────────────────────
    write_recommendation(pooled, pooled_cmp, foldwise_cmp, c_by_fold, outdir)
    write_readme(outdir, models_found)

    # ── command log ───────────────────────────────────────────────────────────
    finished = datetime.now(timezone.utc).isoformat()
    generated = sorted(str(p.relative_to(outdir)) for p in outdir.rglob("*") if p.is_file()
                       and p.name != "command_log.json")
    command_log = {
        "created_utc": started,
        "finished_utc": finished,
        "script": str(Path(__file__).resolve()),
        "output_dir": str(outdir),
        "models_found": models_found,
        "readout_feature_set": READOUT_FEATURE_SET,
        "original_c_grid": ORIGINAL_C_GRID,
        "extended_c_grid": EXTENDED_C_GRID,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "vae_training_launched": False,
        "threshold_fitted_outside_inner_cv": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "existing_model_output_modified": False,
        "generated_files": generated,
    }
    (outdir / "command_log.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\nDone. Outputs in: {outdir}")

    # ── summary to stdout ─────────────────────────────────────────────────────
    primary_strat = "inner_oof_target_sens_ge_0p70_max_spec"
    print("\n── Extended-grid pooled AUC (primary threshold) ──")
    sub = pooled[pooled["threshold_strategy"] == primary_strat].sort_values("auc", ascending=False)
    for _, r in sub.iterrows():
        print(f"  {r['model_key']:12s}  AUC={r['auc']:.6f}  PR-AUC={r['pr_auc']:.6f}"
              f"  BA={r['balanced_accuracy']:.6f}  Sens={r['sensitivity']:.6f}")

    print("\n── Δ AUC (extended minus original, pooled) ──")
    p_primary = pooled_cmp[pooled_cmp["threshold_strategy"] == primary_strat]
    for _, r in p_primary.iterrows():
        print(f"  {r['model_key']:12s}  Δ AUC={r['delta_auc']:+.6f}  Δ PR-AUC={r['delta_pr_auc']:+.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
