#!/usr/bin/env python3
"""Minimal refined classifier-only sweep on saved mfrsplit_3840 latents.

This uses the existing outer folds and latent mu cache from the completed
mfrsplit_3840 run. It does not retrain the VAE and does not read or modify
tensor data. Non-0.5 thresholds are selected from train/dev inner-CV
out-of-fold scores only.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
)
PREVIOUS_SWEEP_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
)
PRIMARY_BASELINE_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/adni_v5_1_batch20260514b_threshold_final_audit"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/adni_v5_1_batch20260514b_refined_classifier_only_sweep"
)
TARGET_SENSITIVITY = 0.70
PRIMARY_MODELS = {"logreg_l2", "logreg_elasticnet", "svm_rbf"}


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    params: Dict[str, Any]
    model_role: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=SOURCE_RUN_DIR)
    parser.add_argument("--latent-cache-dir", type=Path, default=PREVIOUS_SWEEP_DIR / "latent_cache")
    parser.add_argument("--primary-baseline-dir", type=Path, default=PRIMARY_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = resolve(path)
    generated = [
        "README.md",
        "refined_sweep_model_status.csv",
        "refined_sweep_thresholds_by_fold.csv",
        "refined_sweep_foldwise_metrics.csv",
        "refined_sweep_pooled_metrics.csv",
        "refined_sweep_predictions.csv",
        "refined_sweep_subgroup_metrics_by_manufacturer.csv",
        "comparison_vs_primary_baseline.csv",
        "command_log.json",
    ]
    path.mkdir(parents=True, exist_ok=True)
    if any((path / name).exists() for name in generated) and not overwrite:
        raise FileExistsError(f"{path} already contains refined sweep outputs; pass --overwrite")
    for name in generated:
        p = path / name
        if p.exists():
            p.unlink()
    return path


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


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def safe_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(y_score, dtype=float)))


def safe_pr_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, np.asarray(y_score, dtype=float)))


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
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
        "auc": safe_auc(y, score),
        "pr_auc": safe_pr_auc(y, score),
    }


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
        row: Dict[str, Any] = {"threshold": float(thr)}
        row.update(binary_metrics(y, s, pred))
        row["youden_j"] = row["sensitivity"] + row["specificity"] - 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def select_thresholds(y_true: Sequence[int], y_score: Sequence[float]) -> List[Dict[str, Any]]:
    tbl = threshold_table(y_true, y_score)
    selections: List[Dict[str, Any]] = []
    r = tbl.sort_values(["youden_j", "sensitivity", "specificity", "threshold"], ascending=[False, False, False, False]).iloc[0]
    selections.append(
        {
            "threshold_strategy": "inner_oof_youden_j",
            "threshold": float(r["threshold"]),
            "selection_metric": "youden_j",
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


def score_1d(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(x)[:, 1], dtype=float)
    if hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(x), dtype=float).ravel()
        return 1.0 / (1.0 + np.exp(-raw))
    raise TypeError(f"Estimator has no scoring method: {type(estimator)}")


def c_values() -> List[float]:
    return [float(x) for x in np.logspace(-5, 2, 15)]


def build_model_configs() -> List[ModelConfig]:
    configs: List[ModelConfig] = []
    for class_weight in [None, "balanced"]:
        for calibration in ["none", "sigmoid"]:
            for c in c_values():
                configs.append(
                    ModelConfig(
                        "logreg_l2",
                        {"C": c, "class_weight": class_weight, "calibration": calibration},
                        "primary",
                    )
                )
    for class_weight in [None, "balanced"]:
        for c in c_values():
            for l1_ratio in [0.05, 0.10, 0.25, 0.50, 0.75]:
                configs.append(
                    ModelConfig(
                        "logreg_elasticnet",
                        {"C": c, "l1_ratio": l1_ratio, "class_weight": class_weight, "calibration": "none"},
                        "primary",
                    )
                )
    for class_weight in [None, "balanced"]:
        for c in [0.03, 0.1, 0.3, 1.0, 3.0]:
            for gamma in [1e-4, 3e-4, 1e-3, 3e-3, "scale"]:
                configs.append(
                    ModelConfig(
                        "svm_rbf",
                        {"C": c, "gamma": gamma, "class_weight": class_weight, "calibration": "sigmoid"},
                        "primary",
                    )
                )
    try:
        import lightgbm  # noqa: F401

        for class_weight in [None, "balanced"]:
            for n_estimators in [100, 250, 400]:
                for learning_rate in [0.03, 0.10]:
                    for num_leaves in [7, 15]:
                        configs.append(
                            ModelConfig(
                                "lightgbm",
                                {
                                    "n_estimators": n_estimators,
                                    "learning_rate": learning_rate,
                                    "num_leaves": num_leaves,
                                    "class_weight": class_weight,
                                    "calibration": "none",
                                },
                                "secondary_exploratory",
                            )
                        )
    except Exception:
        try:
            import xgboost  # noqa: F401

            for n_estimators in [100, 250, 500]:
                for learning_rate in [0.03, 0.10]:
                    for max_depth in [1, 2, 3]:
                        configs.append(
                            ModelConfig(
                                "xgboost",
                                {
                                    "n_estimators": n_estimators,
                                    "learning_rate": learning_rate,
                                    "max_depth": max_depth,
                                    "calibration": "none",
                                },
                                "secondary_exploratory",
                            )
                        )
        except Exception:
            pass
    return configs


def base_pipeline(config: ModelConfig, pre: ColumnTransformer, seed: int, y_train: np.ndarray) -> BaseEstimator:
    params = config.params
    if config.model_name == "logreg_l2":
        model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=float(params["C"]),
            class_weight=params["class_weight"],
            max_iter=10000,
            random_state=seed,
        )
    elif config.model_name == "logreg_elasticnet":
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=float(params["C"]),
            l1_ratio=float(params["l1_ratio"]),
            class_weight=params["class_weight"],
            max_iter=10000,
            tol=1e-3,
            random_state=seed,
            n_jobs=1,
        )
    elif config.model_name == "svm_rbf":
        model = SVC(
            kernel="rbf",
            C=float(params["C"]),
            gamma=params["gamma"],
            class_weight=params["class_weight"],
            probability=False,
            random_state=seed,
            cache_size=1000,
        )
    elif config.model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            objective="binary",
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            num_leaves=int(params["num_leaves"]),
            class_weight=params["class_weight"],
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            force_col_wise=True,
        )
    elif config.model_name == "xgboost":
        from xgboost import XGBClassifier

        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            scale_pos_weight=float(n_neg / n_pos) if n_pos else 1.0,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
    else:
        raise ValueError(f"Unsupported model: {config.model_name}")
    return Pipeline([("pre", clone(pre)), ("model", model)])


def calibrated_or_plain(config: ModelConfig, pre: ColumnTransformer, seed: int, y_train: np.ndarray) -> BaseEstimator:
    base = base_pipeline(config, pre, seed, y_train)
    if config.params.get("calibration") == "sigmoid":
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3, n_jobs=1)
    return base


def evaluate_config(
    config: ModelConfig,
    x: pd.DataFrame,
    y: np.ndarray,
    inner_cv: List[Tuple[np.ndarray, np.ndarray]],
    pre: ColumnTransformer,
    seed: int,
) -> Dict[str, Any]:
    oof = np.full(len(y), np.nan, dtype=float)
    for split_id, (train_idx, val_idx) in enumerate(inner_cv):
        est = calibrated_or_plain(config, pre, seed + split_id + 100, y[train_idx])
        est.fit(x.iloc[train_idx], y[train_idx])
        oof[val_idx] = score_1d(est, x.iloc[val_idx])
    if np.isnan(oof).any():
        raise RuntimeError(f"OOF scores incomplete for {config}")
    return {
        "model_name": config.model_name,
        "model_role": config.model_role,
        "params": config.params,
        "params_json": json.dumps(config.params, sort_keys=True),
        "inner_oof_auc": safe_auc(y, oof),
        "inner_oof_pr_auc": safe_pr_auc(y, oof),
        "inner_oof_scores": oof,
    }


def load_latent_pair(cache_dir: Path, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(require(cache_dir / f"fold_{fold}_trainDev_latent_mu.csv"))
    test = pd.read_csv(require(cache_dir / f"fold_{fold}_test_latent_mu.csv"))
    return train, test


def pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (model_name, strategy), group in predictions.groupby(["model_name", "threshold_strategy"], dropna=False):
        row = {
            "model_name": model_name,
            "threshold_strategy": strategy,
            "model_role": group["model_role"].iloc[0],
            "threshold": 0.5 if strategy == "fixed_0p5" else "fold_specific",
        }
        row.update(binary_metrics(group["y_true"], group["y_score"], group["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model_role", "model_name", "threshold_strategy"])


def subgroup_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (model_name, strategy, manufacturer), group in predictions.groupby(["model_name", "threshold_strategy", "Manufacturer"], dropna=False):
        row = {
            "model_name": model_name,
            "threshold_strategy": strategy,
            "model_role": group["model_role"].iloc[0],
            "Manufacturer": manufacturer,
            "threshold": 0.5 if strategy == "fixed_0p5" else "fold_specific",
        }
        row.update(binary_metrics(group["y_true"], group["y_score"], group["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model_role", "model_name", "threshold_strategy", "Manufacturer"])


def compare_to_baseline(outdir: Path, baseline_dir: Path, refined_pooled: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    baseline_path = baseline_dir / "logreg_l2_threshold_comparison.csv"
    if baseline_path.exists():
        old = pd.read_csv(baseline_path)
        old["run"] = "previous_classifier_only_logreg_l2"
        refined = refined_pooled[refined_pooled["model_name"] == "logreg_l2"].copy()
        refined["run"] = "refined_classifier_only"
        common_cols = [
            "run",
            "model_name",
            "threshold_strategy",
            "n",
            "n_cn",
            "n_ad",
            "tn",
            "fp",
            "fn",
            "tp",
            "accuracy",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "f1",
            "predicted_ad_rate",
            "auc",
            "pr_auc",
        ]
        rows_df = pd.concat([old[common_cols], refined[common_cols]], ignore_index=True)
        rows_df.to_csv(outdir / "comparison_vs_primary_baseline.csv", index=False)
        return rows_df
    empty = pd.DataFrame(rows)
    empty.to_csv(outdir / "comparison_vs_primary_baseline.csv", index=False)
    return empty


def write_readme(outdir: Path, pooled: pd.DataFrame, model_status: pd.DataFrame, comparison: pd.DataFrame) -> None:
    primary = pooled[pooled["model_role"] == "primary"].copy()
    best_auc = primary.sort_values(["auc", "pr_auc", "balanced_accuracy"], ascending=False).iloc[0]
    target = primary[primary["threshold_strategy"] == "inner_oof_target_sens_ge_0p70_max_spec"].copy()
    best_target = target.sort_values(["balanced_accuracy", "auc", "sensitivity"], ascending=False).iloc[0]
    status_l2 = model_status[model_status["model_name"] == "logreg_l2"]
    readme = f"""# ADNI v5.1 batch20260514b Refined Classifier-Only Sweep

## Scope

- VAE retrained: `False`.
- Tensor modified: `False`.
- Metadata/ledger modified: `False`.
- Latent source: saved mfrsplit_3840 fold-level `mu` cache.
- Threshold selection for non-0.5 operating points: `true_inner_cv_oof`.

## Best primary model by ranking

`{best_auc.model_name}` / `{best_auc.threshold_strategy}`: AUC={best_auc.auc:.3f}, PR-AUC={best_auc.pr_auc:.3f}, balanced accuracy={best_auc.balanced_accuracy:.3f}, sensitivity={best_auc.sensitivity:.3f}, specificity={best_auc.specificity:.3f}, F1={best_auc.f1:.3f}.

## Best primary target-sensitivity operating point

`{best_target.model_name}` / `{best_target.threshold_strategy}`: AUC={best_target.auc:.3f}, PR-AUC={best_target.pr_auc:.3f}, balanced accuracy={best_target.balanced_accuracy:.3f}, sensitivity={best_target.sensitivity:.3f}, specificity={best_target.specificity:.3f}, F1={best_target.f1:.3f}.

## LogReg L2 selected configurations

{status_l2[["fold", "best_params", "best_inner_auc", "best_inner_pr_auc"]].to_markdown(index=False)}

## Outputs

- `refined_sweep_model_status.csv`
- `refined_sweep_thresholds_by_fold.csv`
- `refined_sweep_foldwise_metrics.csv`
- `refined_sweep_pooled_metrics.csv`
- `refined_sweep_predictions.csv`
- `refined_sweep_subgroup_metrics_by_manufacturer.csv`
- `comparison_vs_primary_baseline.csv`
- `command_log.json`
"""
    outdir.joinpath("README.md").write_text(readme, encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = resolve(args.run_dir)
    cache_dir = resolve(args.latent_cache_dir)
    outdir = prepare_output_dir(args.output_dir, args.overwrite)
    require(run_dir / "run_config.json")
    require(cache_dir)
    require(args.primary_baseline_dir)

    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    seed = int(cfg.get("args", {}).get("seed", 42))
    configs = build_model_configs()

    fold_metric_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    pred_rows: List[pd.DataFrame] = []
    model_status_rows: List[Dict[str, Any]] = []

    for fold in range(1, args.folds + 1):
        train_df, test_df = load_latent_pair(cache_dir, fold)
        mu_cols = [c for c in train_df.columns if c.startswith("mu_")]
        feature_cols = mu_cols + ["Age", "Sex"]
        x_train = train_df[feature_cols].copy()
        y_train = train_df["y"].astype(int).to_numpy()
        x_test = test_df[feature_cols].copy()
        y_test = test_df["y"].astype(int).to_numpy()
        inner_key, inner_context, min_inner_cell = inner_stratification_key(train_df, n_splits=5)
        inner_cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + fold + 30).split(np.zeros(len(train_df)), inner_key))
        pre = make_preprocessor(mu_cols)

        print(f"[fold {fold}] evaluating {len(configs)} configs; inner={inner_context}, min_cell={min_inner_cell}", flush=True)
        evals = Parallel(n_jobs=args.n_jobs, verbose=0)(
            delayed(evaluate_config)(config, x_train, y_train, inner_cv, pre, seed + fold * 1000 + i)
            for i, config in enumerate(configs)
        )
        eval_df = pd.DataFrame(
            [
                {
                    "model_name": e["model_name"],
                    "model_role": e["model_role"],
                    "params_json": e["params_json"],
                    "inner_oof_auc": e["inner_oof_auc"],
                    "inner_oof_pr_auc": e["inner_oof_pr_auc"],
                }
                for e in evals
            ]
        )
        eval_df.to_csv(outdir / f"fold_{fold}_config_search_results.csv", index=False)

        for model_name, model_evals_df in eval_df.groupby("model_name"):
            best_idx = model_evals_df.sort_values(["inner_oof_auc", "inner_oof_pr_auc"], ascending=False).index[0]
            best_eval = evals[int(best_idx)]
            best_params = best_eval["params"]
            model_role = best_eval["model_role"]
            best_config = ModelConfig(model_name=model_name, params=best_params, model_role=model_role)
            best_oof = np.asarray(best_eval["inner_oof_scores"], dtype=float)
            final_est = calibrated_or_plain(best_config, pre, seed + fold * 2000, y_train)
            final_est.fit(x_train, y_train)
            test_score = score_1d(final_est, x_test)
            model_status_rows.append(
                {
                    "fold": fold,
                    "model_name": model_name,
                    "model_role": model_role,
                    "status": "fit_ok",
                    "best_params": json.dumps(best_params, sort_keys=True),
                    "best_inner_auc": best_eval["inner_oof_auc"],
                    "best_inner_pr_auc": best_eval["inner_oof_pr_auc"],
                    "inner_cv_context": inner_context,
                    "minimum_inner_stratum_count": int(min_inner_cell),
                    "n_configs_evaluated": int(len(model_evals_df)),
                }
            )
            for sel in select_thresholds(y_train, best_oof):
                thr = float(sel["threshold"])
                y_pred = (test_score >= thr).astype(int)
                metric = {
                    "fold": fold,
                    "model_name": model_name,
                    "model_role": model_role,
                    "threshold_strategy": sel["threshold_strategy"],
                    "threshold": thr,
                    "threshold_selection_context": "true_inner_cv_oof" if sel["threshold_strategy"] != "fixed_0p5" else "fixed_no_selection",
                    "inner_cv_context": inner_context,
                    "minimum_inner_stratum_count": int(min_inner_cell),
                    "best_params": json.dumps(best_params, sort_keys=True),
                    "best_inner_auc": best_eval["inner_oof_auc"],
                    "best_inner_pr_auc": best_eval["inner_oof_pr_auc"],
                    **sel,
                }
                metric.update(binary_metrics(y_test, test_score, y_pred))
                fold_metric_rows.append(metric)
                threshold_rows.append(
                    {
                        "fold": fold,
                        "model_name": model_name,
                        "model_role": model_role,
                        "threshold_strategy": sel["threshold_strategy"],
                        "threshold": thr,
                        "threshold_selection_context": metric["threshold_selection_context"],
                        "inner_cv_context": inner_context,
                        "minimum_inner_stratum_count": int(min_inner_cell),
                        "best_params": json.dumps(best_params, sort_keys=True),
                        "selection_metric": sel["selection_metric"],
                        "inner_oof_sensitivity": sel["inner_oof_sensitivity"],
                        "inner_oof_specificity": sel["inner_oof_specificity"],
                        "inner_oof_balanced_accuracy": sel["inner_oof_balanced_accuracy"],
                    }
                )
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
                pred["model_name"] = model_name
                pred["model_role"] = model_role
                pred["threshold_strategy"] = sel["threshold_strategy"]
                pred["threshold"] = thr
                pred["y_true"] = y_test
                pred["y_score"] = test_score
                pred["y_pred"] = y_pred
                pred_rows.append(pred)
        joblib.dump(model_status_rows, outdir / f"fold_{fold}_status_snapshot.joblib")

    foldwise = pd.DataFrame(fold_metric_rows).sort_values(["model_role", "model_name", "fold", "threshold_strategy"])
    thresholds = pd.DataFrame(threshold_rows).sort_values(["model_role", "model_name", "fold", "threshold_strategy"])
    predictions = pd.concat(pred_rows, ignore_index=True)
    status = pd.DataFrame(model_status_rows).sort_values(["model_role", "model_name", "fold"])
    pooled = pooled_metrics(predictions)
    subgroup = subgroup_metrics(predictions)

    status.to_csv(outdir / "refined_sweep_model_status.csv", index=False)
    thresholds.to_csv(outdir / "refined_sweep_thresholds_by_fold.csv", index=False)
    foldwise.to_csv(outdir / "refined_sweep_foldwise_metrics.csv", index=False)
    pooled.to_csv(outdir / "refined_sweep_pooled_metrics.csv", index=False)
    predictions.to_csv(outdir / "refined_sweep_predictions.csv", index=False)
    subgroup.to_csv(outdir / "refined_sweep_subgroup_metrics_by_manufacturer.csv", index=False)
    comparison = compare_to_baseline(outdir, resolve(args.primary_baseline_dir), pooled)
    write_readme(outdir, pooled, status, comparison)

    log = {
        "script": str(Path(__file__).resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_dir),
        "latent_cache_dir": str(cache_dir),
        "output_dir": str(outdir),
        "models": sorted(pooled["model_name"].unique().tolist()),
        "vae_retrained": False,
        "vae_training_run": False,
        "classifier_only_training_run": True,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "threshold_selection": "true_inner_cv_oof",
        "outer_test_threshold_leakage": False,
    }
    write_json(outdir / "command_log.json", log)
    return log


def main() -> None:
    args = parse_args()
    log = run(args)
    print(json.dumps(log, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
