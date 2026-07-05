#!/usr/bin/env python3
"""Downstream latent-dimension selection sweep for frozen tanh V4 [4,1,0] latents.

This script does not train or load the CVAE. It consumes fold-wise exported
baseline tanh `mu` latents, fits representation reducers and downstream
classifiers only inside each outer train/dev split, and evaluates each selected
model once on the held-out outer test split.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LATENT_EXPORT_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/auc_sprint_adni_only/latent_exports_ch4_1_0"
)
FALLBACK_MU_LOGVAR_EXPORT_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/auc_sprint_adni_only/latent_exports_mu_logvar_ch4_1_0"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/auc_sprint_adni_only/downstream_latent_selection_sweep"
)

K_VALUES = [16, 32, 64, 96, 128, 192, 256]
PCA_COMPONENTS = [16, 32, 64, 96, 128]
PLS_COMPONENTS = [2, 4, 8, 16, 32]
GENERATED_OUTPUTS = [
    "downstream_latent_selection_metrics.csv",
    "downstream_latent_selection_fold_metrics.csv",
    "downstream_latent_selection_predictions.csv",
    "downstream_latent_selection_failures.csv",
    "downstream_latent_selection_manifest.json",
    "README.md",
]

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", message="Variables are collinear.*")


class PLSFeatureExtractor(BaseEstimator, TransformerMixin):
    """Supervised PLS transformer returning latent scores."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSFeatureExtractor":
        n_components = min(int(self.n_components), X.shape[1], X.shape[0] - 1)
        if n_components < 1:
            raise ValueError("PLSFeatureExtractor requires at least one component")
        self.n_components_ = n_components
        self.model_ = PLSRegression(n_components=n_components, scale=False)
        self.model_.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.model_.transform(X)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--latent-export-dir", type=Path, default=DEFAULT_LATENT_EXPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--include-boosting",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Include LightGBM/XGBoost if installed; never required for the sweep.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = resolve(path)
    path.mkdir(parents=True, exist_ok=True)
    existing = [path / name for name in GENERATED_OUTPUTS if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{path} already contains generated outputs; pass --overwrite")
    if overwrite:
        for file_path in existing:
            file_path.unlink()
    return path


def split_path(latent_dir: Path, fold: int, split: str) -> Path:
    candidates = [
        latent_dir / f"fold_{fold}_{split}_latents_mu.csv",
        latent_dir / f"fold_{fold}_{split}_latents_mu_logvar.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing fold {fold} {split} latent export in {latent_dir}")


def validate_latent_files(latent_dir: Path) -> Path:
    latent_dir = resolve(latent_dir)
    if not latent_dir.exists() and FALLBACK_MU_LOGVAR_EXPORT_DIR.exists():
        latent_dir = FALLBACK_MU_LOGVAR_EXPORT_DIR
    missing: List[str] = []
    for fold in range(1, 6):
        for split in ["trainDev", "test"]:
            try:
                split_path(latent_dir, fold, split)
            except FileNotFoundError as exc:
                missing.append(str(exc))
    if missing:
        raise FileNotFoundError("Missing exported tanh mu latents:\n" + "\n".join(missing))
    return latent_dir


def natural_key(column: str) -> Tuple[str, int]:
    prefix, _, suffix = column.rpartition("_")
    try:
        return prefix, int(suffix)
    except ValueError:
        return prefix, 10**9


def mu_columns(df: pd.DataFrame) -> List[str]:
    cols = sorted([c for c in df.columns if c.startswith("mu_")], key=natural_key)
    if not cols:
        raise ValueError("No mu_* columns found in latent export")
    return cols


def load_fold_split(latent_dir: Path, fold: int, split: str) -> pd.DataFrame:
    df = pd.read_csv(split_path(latent_dir, fold, split))
    if "y" not in df.columns:
        df["y"] = df["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1})
    df["y_true"] = pd.to_numeric(df["y"], errors="coerce")
    df = df[df["y_true"].notna()].copy()
    df["y_true"] = df["y_true"].astype(int)
    return df


def make_feature_matrix(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    latent_cols = mu_columns(train)
    required = latent_cols + ["Age", "Sex"]
    missing_train = [c for c in required if c not in train.columns]
    missing_test = [c for c in required if c not in test.columns]
    if missing_train or missing_test:
        raise ValueError(f"Missing feature columns: train={missing_train}, test={missing_test}")

    forbidden = {"Manufacturer", "Site3", "SourceCohort", "InputCohort"}
    if any(c in forbidden or c.lower().startswith(("manufacturer", "site")) for c in required):
        raise ValueError("Forbidden scanner/site feature requested")

    train_x = train[required].copy()
    test_x = test[required].copy()
    for frame in [train_x, test_x]:
        for col in latent_cols + ["Age"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["Sex"] = frame["Sex"].astype(str).replace({"nan": np.nan, "None": np.nan})

    train_x = pd.get_dummies(train_x, columns=["Sex"], drop_first=False)
    test_x = pd.get_dummies(test_x, columns=["Sex"], drop_first=False)
    test_x = test_x.reindex(columns=train_x.columns, fill_value=0)

    meta_cols = [c for c in train_x.columns if c not in latent_cols]
    scanner_like_meta = [c for c in meta_cols if c.lower().startswith(("manufacturer", "site"))]
    if scanner_like_meta:
        raise ValueError(f"Forbidden scanner/site metadata columns present: {scanner_like_meta}")
    return train_x.astype(float), test_x.astype(float), latent_cols, meta_cols


def make_preprocessor(
    representation: str,
    latent_cols: Sequence[str],
    meta_cols: Sequence[str],
    seed: int,
) -> Tuple[ColumnTransformer, Dict[str, List[Any]], str]:
    latent_steps: List[Tuple[str, Any]] = [
        ("impute", SimpleImputer()),
        ("scale", StandardScaler()),
    ]
    grid: Dict[str, List[Any]] = {}
    selected_param = "all_256"

    if representation == "all_mu":
        selected_param = "all_256"
    elif representation == "select_f_classif":
        latent_steps.append(("select", SelectKBest(score_func=f_classif, k=64)))
        grid["pre__latent__select__k"] = K_VALUES
        selected_param = "k"
    elif representation == "select_mutual_info":
        score_func = partial(mutual_info_classif, random_state=seed)
        latent_steps.append(("select", SelectKBest(score_func=score_func, k=64)))
        grid["pre__latent__select__k"] = K_VALUES
        selected_param = "k"
    elif representation == "pca":
        latent_steps.append(("pca", PCA(n_components=64, random_state=seed)))
        grid["pre__latent__pca__n_components"] = PCA_COMPONENTS
        selected_param = "n_components"
    elif representation == "pls":
        latent_steps.append(("pls", PLSFeatureExtractor(n_components=8)))
        grid["pre__latent__pls__n_components"] = PLS_COMPONENTS
        selected_param = "n_components"
    else:
        raise ValueError(f"Unknown representation: {representation}")

    pre = ColumnTransformer(
        transformers=[
            ("latent", Pipeline(latent_steps), list(latent_cols)),
            ("metadata", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), list(meta_cols)),
        ],
        remainder="drop",
    )
    return pre, grid, selected_param


def base_classifier_specs(seed: int) -> Dict[str, Tuple[List[Tuple[str, Any]], Dict[str, List[Any]]]]:
    specs: Dict[str, Tuple[List[Tuple[str, Any]], Dict[str, List[Any]]]] = {
        "lda_shrinkage": (
            [("clf", LinearDiscriminantAnalysis(solver="lsqr"))],
            {"clf__shrinkage": ["auto", 0.5]},
        ),
        "logreg_l2": (
            [
                (
                    "clf",
                    LogisticRegression(
                        max_iter=6000,
                        class_weight="balanced",
                        penalty="l2",
                        solver="liblinear",
                        random_state=seed,
                    ),
                )
            ],
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "logreg_elasticnet": (
            [
                (
                    "clf",
                    LogisticRegression(
                        max_iter=8000,
                        class_weight="balanced",
                        penalty="elasticnet",
                        solver="saga",
                        random_state=seed,
                    ),
                )
            ],
            {"clf__C": [0.1, 1.0], "clf__l1_ratio": [0.25, 0.75]},
        ),
        "calibrated_ridge": (
            [("clf", RidgeClassifier(class_weight="balanced", random_state=seed))],
            {"clf__alpha": [1.0, 10.0, 100.0]},
        ),
        "linear_svm_calibrated": (
            [("clf", SVC(kernel="linear", class_weight="balanced", random_state=seed))],
            {"clf__C": [0.1, 1.0]},
        ),
        "rbf_svm": (
            [("clf", SVC(kernel="rbf", class_weight="balanced", random_state=seed))],
            {"clf__C": [1.0, 10.0], "clf__gamma": ["scale", 0.01]},
        ),
        "pls_logreg": (
            [
                ("post_pls", PLSFeatureExtractor(n_components=4)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=6000,
                        class_weight="balanced",
                        penalty="l2",
                        solver="liblinear",
                        random_state=seed,
                    ),
                ),
            ],
            {"post_pls__n_components": [2, 4, 8, 16], "clf__C": [0.1, 1.0]},
        ),
        "small_mlp": (
            [
                (
                    "clf",
                    MLPClassifier(
                        activation="relu",
                        solver="adam",
                        max_iter=350,
                        early_stopping=True,
                        validation_fraction=0.2,
                        n_iter_no_change=20,
                        random_state=seed,
                    ),
                )
            ],
            {"clf__hidden_layer_sizes": [(16,), (32,)], "clf__alpha": [0.03]},
        ),
        "extra_trees_constrained": (
            [
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=1,
                    ),
                )
            ],
            {"clf__max_depth": [3, 5], "clf__min_samples_leaf": [5]},
        ),
    }
    return specs


def boosting_specs(seed: int, include: str) -> Dict[str, Tuple[List[Tuple[str, Any]], Dict[str, List[Any]]]]:
    if include == "no":
        return {}
    specs: Dict[str, Tuple[List[Tuple[str, Any]], Dict[str, List[Any]]]] = {}
    if HAS_LIGHTGBM and LGBMClassifier is not None:
        specs["lightgbm_conservative"] = (
            [
                (
                    "clf",
                    LGBMClassifier(
                        objective="binary",
                        n_estimators=120,
                        learning_rate=0.04,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=1,
                        verbosity=-1,
                    ),
                )
            ],
            {"clf__num_leaves": [7], "clf__max_depth": [2], "clf__min_child_samples": [20]},
        )
    elif include == "yes":
        specs["lightgbm_conservative"] = ([], {"__unavailable__": ["lightgbm not installed"]})

    if HAS_XGBOOST and XGBClassifier is not None:
        specs["xgboost_conservative"] = (
            [
                (
                    "clf",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_estimators=120,
                        learning_rate=0.04,
                        subsample=0.85,
                        colsample_bytree=0.75,
                        reg_alpha=0.1,
                        reg_lambda=3.0,
                        random_state=seed,
                        n_jobs=1,
                    ),
                )
            ],
            {"clf__max_depth": [2], "clf__min_child_weight": [5]},
        )
    elif include == "yes":
        specs["xgboost_conservative"] = ([], {"__unavailable__": ["xgboost not installed"]})
    return specs


def classifier_specs(seed: int, include_boosting: str) -> Dict[str, Tuple[List[Tuple[str, Any]], Dict[str, List[Any]]]]:
    specs = base_classifier_specs(seed)
    specs.update(boosting_specs(seed, include_boosting))
    return specs


def make_pipeline_and_grid(
    representation: str,
    classifier: str,
    latent_cols: Sequence[str],
    meta_cols: Sequence[str],
    seed: int,
    include_boosting: str,
) -> Tuple[Pipeline, Dict[str, List[Any]], str]:
    pre, rep_grid, selected_param = make_preprocessor(representation, latent_cols, meta_cols, seed)
    specs = classifier_specs(seed, include_boosting)
    if classifier not in specs:
        raise KeyError(classifier)
    clf_steps, clf_grid = specs[classifier]
    if not clf_steps and "__unavailable__" in clf_grid:
        raise ImportError(str(clf_grid["__unavailable__"][0]))
    if representation == "pls" and classifier == "pls_logreg":
        raise ValueError("Skipping nested PLS representation plus PLS-logreg classifier")
    pipe = Pipeline([("pre", pre)] + clf_steps)
    grid = {}
    grid.update(rep_grid)
    grid.update(clf_grid)
    return pipe, grid, selected_param


def score_model(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        score = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    elif hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(x), dtype=float)
        score = 1.0 / (1.0 + np.exp(-np.clip(raw, -40, 40)))
    else:
        score = np.asarray(model.predict(x), dtype=float)
    return np.clip(score, 1e-6, 1.0 - 1e-6)


def ece_score(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (y_score >= low) & (y_score <= high) if i == n_bins - 1 else (y_score >= low) & (y_score < high)
        if mask.any():
            ece += (mask.sum() / n) * abs(float(y_score[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)


def metrics_from_scores(y: np.ndarray, score: np.ndarray) -> Dict[str, Any]:
    score = np.clip(np.asarray(score, dtype=float), 1e-6, 1.0 - 1e-6)
    pred = (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "n_CN": int((y == 0).sum()),
        "n_AD": int((y == 1).sum()),
        "roc_auc": roc_auc_score(y, score) if len(set(y)) == 2 else np.nan,
        "pr_auc": average_precision_score(y, score) if len(set(y)) == 2 else np.nan,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "brier": brier_score_loss(y, score),
        "ece": ece_score(y, score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def selected_latent_setting(representation: str, best_params: Dict[str, Any]) -> Tuple[Any, str]:
    key_candidates = [
        "pre__latent__select__k",
        "pre__latent__pca__n_components",
        "pre__latent__pls__n_components",
    ]
    for key in key_candidates:
        if key in best_params:
            return best_params[key], key
    if representation == "all_mu":
        return 256, "all_mu_dims"
    return np.nan, ""


def classifier_post_pls_setting(best_params: Dict[str, Any]) -> Any:
    return best_params.get("post_pls__n_components", np.nan)


def fold_metric_row(
    representation: str,
    classifier: str,
    fold: int,
    inner_cv_auc: float,
    y: np.ndarray,
    score: np.ndarray,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    model_id = f"{representation}__{classifier}"
    row: Dict[str, Any] = {
        "model_id": model_id,
        "representation": representation,
        "classifier": classifier,
        "fold": fold,
        "inner_cv_auc_trainDev": inner_cv_auc,
        "status": "ok",
    }
    row.update(metrics_from_scores(y, score))
    row.update(extra)
    return row


def aggregate_metrics(fold_metrics: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "brier",
        "ece",
    ]
    rows: List[Dict[str, Any]] = []
    for model_id, group in fold_metrics.groupby("model_id", sort=False):
        pred_group = preds[preds["model_id"] == model_id]
        if pred_group.empty:
            continue
        y = pred_group["y_true"].astype(int).to_numpy()
        score = pred_group["y_score"].astype(float).to_numpy()
        pooled = metrics_from_scores(y, score)
        row: Dict[str, Any] = {
            "model_id": model_id,
            "representation": str(group["representation"].iloc[0]),
            "classifier": str(group["classifier"].iloc[0]),
            "n_total": int(len(y)),
            "n_CN": int((y == 0).sum()),
            "n_AD": int((y == 1).sum()),
            "n_folds": int(group["fold"].nunique()),
            "status": "ok",
            "inner_cv_auc_mean_fold": float(group["inner_cv_auc_trainDev"].mean()),
            "inner_cv_auc_sd_fold": float(group["inner_cv_auc_trainDev"].std(ddof=1)),
            "selected_latent_setting_mode": mode_or_blank(group["selected_latent_setting"]),
            "selected_post_pls_components_mode": mode_or_blank(group["selected_post_pls_components"]),
            "best_params_by_fold": json.dumps(
                {str(int(r["fold"])): r.get("best_params", "") for _, r in group.iterrows()},
                sort_keys=True,
            ),
        }
        for col in metric_cols:
            row[f"{col}_mean_fold"] = float(group[col].mean())
            row[f"{col}_sd_fold"] = float(group[col].std(ddof=1))
            row[f"{col}_pooled"] = pooled[col]
        row["generalization_gap_inner_minus_outer_auc"] = row["inner_cv_auc_mean_fold"] - row["roc_auc_mean_fold"]
        row.update(
            {
                "threshold_0p5_TN": pooled["tn"],
                "threshold_0p5_FP": pooled["fp"],
                "threshold_0p5_FN": pooled["fn"],
                "threshold_0p5_TP": pooled["tp"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["roc_auc_mean_fold", "balanced_accuracy_mean_fold", "pr_auc_mean_fold"],
        ascending=False,
    )


def mode_or_blank(values: Iterable[Any]) -> Any:
    series = pd.Series(list(values)).dropna()
    if series.empty:
        return ""
    mode = series.astype(str).mode()
    return mode.iloc[0] if len(mode) else ""


def write_readme(outdir: Path, summary: pd.DataFrame, fold_metrics: pd.DataFrame, failures: pd.DataFrame) -> None:
    best = summary.iloc[0]
    all_mu_best_auc = float(summary.loc[summary["representation"] == "all_mu", "roc_auc_mean_fold"].max())
    best_auc = float(best["roc_auc_mean_fold"])
    reduced_rows = summary[summary["representation"] != "all_mu"]
    reduced_delta = float(reduced_rows.iloc[0]["roc_auc_mean_fold"] - all_mu_best_auc) if not reduced_rows.empty else np.nan

    lda_rows = summary[summary["classifier"] == "lda_shrinkage"]
    lda_best_auc = float(lda_rows["roc_auc_mean_fold"].max()) if not lda_rows.empty else np.nan
    lda_delta_from_best = best_auc - lda_best_auc if np.isfinite(lda_best_auc) else np.nan

    nonlinear = {"rbf_svm", "small_mlp", "extra_trees_constrained", "lightgbm_conservative", "xgboost_conservative"}
    nonlinear_rows = summary[summary["classifier"].isin(nonlinear)]
    nonlinear_best_auc = float(nonlinear_rows["roc_auc_mean_fold"].max()) if not nonlinear_rows.empty else np.nan
    nonlinear_gap = float(nonlinear_rows["generalization_gap_inner_minus_outer_auc"].median()) if not nonlinear_rows.empty else np.nan

    if best["representation"] != "all_mu" and best_auc > all_mu_best_auc + 0.005:
        latent_interpretation = (
            f"Reducing/projecting latent dimensions helped: the best reduced representation "
            f"improved mean-fold AUC by {best_auc - all_mu_best_auc:.4f} over the best all-mu model."
        )
    elif np.isfinite(reduced_delta) and reduced_delta > 0.005:
        latent_interpretation = (
            f"A reduced representation helped modestly, improving over all-mu by {reduced_delta:.4f}, "
            "but it was not the overall winner."
        )
    else:
        latent_interpretation = (
            "Latent-dimension reduction did not provide a material improvement over the full mu vector."
        )

    if np.isfinite(lda_delta_from_best) and lda_delta_from_best <= 0.005:
        lda_interpretation = "LDA shrinkage remains effectively tied with the best downstream result."
    elif str(best["classifier"]) == "lda_shrinkage":
        lda_interpretation = "LDA shrinkage remains the best downstream classifier in this sweep."
    else:
        lda_interpretation = (
            f"LDA shrinkage was not the top model here; its best mean-fold AUC trailed by {lda_delta_from_best:.4f}."
        )

    if np.isfinite(nonlinear_best_auc) and nonlinear_best_auc > best_auc - 0.005:
        nonlinear_interpretation = (
            "At least one nonlinear classifier was competitive, so it should be treated as a candidate rather than excluded."
        )
    elif np.isfinite(nonlinear_gap) and nonlinear_gap > 0.03:
        nonlinear_interpretation = (
            "Nonlinear classifiers show signs of overfitting: their train/dev CV AUC is higher than outer-fold AUC."
        )
    else:
        nonlinear_interpretation = (
            "Nonlinear classifiers did not outperform the best linear/shrinkage models."
        )

    checkpoint_interpretation = (
        "Proceed to leakage-safe checkpoint selection if this sweep does not create a clear, reproducible downstream-only "
        "AUC gain. That next step keeps the CVAE architecture fixed and changes only which saved epoch is selected."
    )

    display_cols = [
        "model_id",
        "roc_auc_mean_fold",
        "roc_auc_sd_fold",
        "roc_auc_pooled",
        "pr_auc_mean_fold",
        "balanced_accuracy_mean_fold",
        "sensitivity_mean_fold",
        "specificity_mean_fold",
        "brier_mean_fold",
        "ece_mean_fold",
        "selected_latent_setting_mode",
    ]
    top_table = summary[display_cols].head(15).to_markdown(index=False, floatfmt=".4f")

    lines = [
        "# Downstream Latent Selection Sweep",
        "",
        "No VAE retraining was run. The sweep uses exported tanh baseline `mu` latents from ADNI v4 [4,1,0].",
        "",
        "- Predictive features: selected/projected `mu` representation + Age + Sex.",
        "- Manufacturer/Site features were not used.",
        "- All scaling, feature selection, PCA, PLS, classifier tuning, calibration, and ensemble member selection were fitted inside each outer train/dev split.",
        "- Each selected model was evaluated once on the held-out outer test fold.",
        "",
        "## Best Result",
        f"- Model: `{best['model_id']}`",
        f"- Mean-fold ROC-AUC: {best['roc_auc_mean_fold']:.4f} +/- {best['roc_auc_sd_fold']:.4f}",
        f"- Pooled ROC-AUC: {best['roc_auc_pooled']:.4f}",
        f"- Mean-fold PR-AUC: {best['pr_auc_mean_fold']:.4f}",
        f"- Mean-fold balanced accuracy: {best['balanced_accuracy_mean_fold']:.4f}",
        f"- Selected latent setting mode: `{best['selected_latent_setting_mode']}`",
        "",
        "## Interpretation",
        f"- {latent_interpretation}",
        f"- {lda_interpretation}",
        f"- {nonlinear_interpretation}",
        f"- {checkpoint_interpretation}",
        "",
        "## Top Models",
        "",
        top_table,
        "",
        "## Outputs",
        "",
        "- `downstream_latent_selection_metrics.csv`: mean-fold and pooled metrics by model.",
        "- `downstream_latent_selection_fold_metrics.csv`: per-fold metrics and selected settings.",
        "- `downstream_latent_selection_predictions.csv`: held-out test predictions.",
        "- `downstream_latent_selection_failures.csv`: skipped or failed candidates.",
        "- `downstream_latent_selection_manifest.json`: run manifest.",
    ]
    if not failures.empty:
        lines.extend(["", f"Skipped/failed candidates recorded: {len(failures)}."])
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def model_candidates(include_boosting: str) -> List[Tuple[str, str]]:
    representations = ["all_mu", "select_f_classif", "select_mutual_info", "pls", "pca"]
    classifiers = list(classifier_specs(42, include_boosting).keys())
    return [(rep, clf) for rep in representations for clf in classifiers]


def restrict_grid_to_feasible_inner_cv(
    grid: Dict[str, List[Any]],
    cv: StratifiedKFold,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    n_latent_cols: int,
) -> Dict[str, List[Any]]:
    """Keep requested reducer grids inside sklearn's per-fold feasibility limits."""

    adjusted = dict(grid)
    if "pre__latent__pca__n_components" in adjusted:
        min_inner_train = min(len(train_idx) for train_idx, _dev_idx in cv.split(x_train, y_train))
        max_components = min(n_latent_cols, min_inner_train)
        adjusted["pre__latent__pca__n_components"] = [
            value for value in adjusted["pre__latent__pca__n_components"] if int(value) <= max_components
        ]
        if not adjusted["pre__latent__pca__n_components"]:
            raise ValueError(f"No feasible PCA components for inner train size {min_inner_train}")
    return adjusted


def main() -> int:
    args = parse_args()
    latent_dir = validate_latent_files(args.latent_export_dir)
    outdir = prepare_output_dir(args.output_dir, args.overwrite)

    fold_metrics_rows: List[Dict[str, Any]] = []
    pred_frames: List[pd.DataFrame] = []
    failures: List[Dict[str, Any]] = []
    candidates = model_candidates(args.include_boosting)
    started = datetime.now(timezone.utc)

    print(f"Latent export directory: {latent_dir}")
    print(f"Output directory: {outdir}")
    print(f"Candidate representation/classifier pairs: {len(candidates)}")

    for fold in range(1, 6):
        train = load_fold_split(latent_dir, fold, "trainDev")
        test = load_fold_split(latent_dir, fold, "test")
        train_x, test_x, latent_cols, meta_cols = make_feature_matrix(train, test)
        y_train = train["y_true"].astype(int).to_numpy()
        y_test = test["y_true"].astype(int).to_numpy()
        cv = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.random_seed)
        fold_best: List[Tuple[str, str, float, Any]] = []

        print(f"Fold {fold}: trainDev={len(train_x)} test={len(test_x)}")
        for candidate_index, (representation, classifier) in enumerate(candidates, start=1):
            model_id = f"{representation}__{classifier}"
            try:
                print(f"  [{fold}/5 {candidate_index:02d}/{len(candidates):02d}] {model_id}", flush=True)
                pipe, grid, _selected_param = make_pipeline_and_grid(
                    representation,
                    classifier,
                    latent_cols,
                    meta_cols,
                    args.random_seed,
                    args.include_boosting,
                )
                grid = restrict_grid_to_feasible_inner_cv(grid, cv, train_x, y_train, len(latent_cols))
                search = GridSearchCV(
                    pipe,
                    grid,
                    scoring="roc_auc",
                    cv=cv,
                    n_jobs=args.n_jobs,
                    error_score="raise",
                    refit=True,
                )
                search.fit(train_x, y_train)
                best = search.best_estimator_
                calibrated = CalibratedClassifierCV(best, method="sigmoid", cv=3)
                calibrated.fit(train_x, y_train)
                score = score_model(calibrated, test_x)
                selected_setting, selected_key = selected_latent_setting(representation, search.best_params_)
                post_pls_setting = classifier_post_pls_setting(search.best_params_)
                fold_metrics_rows.append(
                    fold_metric_row(
                        representation,
                        classifier,
                        fold,
                        float(search.best_score_),
                        y_test,
                        score,
                        {
                            "n_mu_dims_input": int(len(latent_cols)),
                            "metadata_features": ",".join(meta_cols),
                            "selected_latent_setting": selected_setting,
                            "selected_latent_param": selected_key,
                            "selected_post_pls_components": post_pls_setting,
                            "calibration_status": "sigmoid_cv3_trainDev",
                            "best_params": json.dumps(search.best_params_, sort_keys=True),
                        },
                    )
                )
                pred_frames.append(
                    pd.DataFrame(
                        {
                            "SubjectID": test["SubjectID"].values,
                            "fold": fold,
                            "model_id": model_id,
                            "representation": representation,
                            "classifier": classifier,
                            "y_true": y_test,
                            "y_score": score,
                            "y_pred": (score >= 0.5).astype(int),
                        }
                    )
                )
                fold_best.append((representation, classifier, float(search.best_score_), calibrated))
                print(
                    f"    ok inner_auc={search.best_score_:.4f} test_auc={roc_auc_score(y_test, score):.4f}",
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    {
                        "fold": fold,
                        "model_id": model_id,
                        "representation": representation,
                        "classifier": classifier,
                        "status": "failed_or_skipped",
                        "error": repr(exc),
                    }
                )
                print(f"    skipped/failed: {repr(exc)}", flush=True)

        top = sorted(fold_best, key=lambda item: item[2], reverse=True)[:3]
        if top:
            ensemble_score = np.mean([score_model(model, test_x) for _rep, _clf, _cv_auc, model in top], axis=0)
            ensemble_name = "soft_probability_ensemble_top3_trainDev_selected"
            fold_metrics_rows.append(
                fold_metric_row(
                    "ensemble",
                    "soft_ensemble",
                    fold,
                    float(np.mean([cv_auc for _rep, _clf, cv_auc, _model in top])),
                    y_test,
                    ensemble_score,
                    {
                        "n_mu_dims_input": int(len(latent_cols)),
                        "metadata_features": ",".join(meta_cols),
                        "selected_latent_setting": "",
                        "selected_latent_param": "ensemble_members",
                        "selected_post_pls_components": "",
                        "calibration_status": "member_models_calibrated",
                        "best_params": "",
                        "ensemble_members": ",".join(f"{rep}__{clf}" for rep, clf, _cv_auc, _model in top),
                    },
                )
            )
            pred_frames.append(
                pd.DataFrame(
                    {
                        "SubjectID": test["SubjectID"].values,
                        "fold": fold,
                        "model_id": "ensemble__soft_ensemble",
                        "representation": "ensemble",
                        "classifier": "soft_ensemble",
                        "y_true": y_test,
                        "y_score": ensemble_score,
                        "y_pred": (ensemble_score >= 0.5).astype(int),
                    }
                )
            )

    if not fold_metrics_rows:
        raise RuntimeError("No downstream latent-selection candidate completed successfully.")

    fold_metrics = pd.DataFrame(fold_metrics_rows)
    preds = pd.concat(pred_frames, ignore_index=True)
    summary = aggregate_metrics(fold_metrics, preds)
    failures_df = pd.DataFrame(failures)

    summary.to_csv(outdir / "downstream_latent_selection_metrics.csv", index=False)
    fold_metrics.to_csv(outdir / "downstream_latent_selection_fold_metrics.csv", index=False)
    preds.to_csv(outdir / "downstream_latent_selection_predictions.csv", index=False)
    failures_df.to_csv(outdir / "downstream_latent_selection_failures.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": started.isoformat(),
        "latent_export_dir": str(latent_dir),
        "latent_export_dir_realpath": str(latent_dir.resolve()),
        "output_dir": str(outdir),
        "output_dir_realpath": str(outdir.resolve()),
        "vae_retrained": False,
        "architecture_modified": False,
        "latent_features": "mu",
        "metadata_features": ["Age", "Sex"],
        "manufacturer_site_features_used": False,
        "outer_split": "existing ADNI v4 Sex-stratified folds",
        "model_selection_split": "outer-fold trainDev only",
        "inner_cv_folds": args.inner_folds,
        "representations": ["all_mu", "select_f_classif", "select_mutual_info", "pls", "pca"],
        "k_values": K_VALUES,
        "pca_components": PCA_COMPONENTS,
        "pls_components": PLS_COMPONENTS,
        "include_boosting": args.include_boosting,
        "has_lightgbm": HAS_LIGHTGBM,
        "has_xgboost": HAS_XGBOOST,
        "n_completed_fold_models": int(len(fold_metrics)),
        "n_failures_or_skips": int(len(failures_df)),
    }
    (outdir / "downstream_latent_selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(outdir, summary, fold_metrics, failures_df)

    display_cols = [
        "model_id",
        "roc_auc_mean_fold",
        "roc_auc_sd_fold",
        "roc_auc_pooled",
        "pr_auc_mean_fold",
        "balanced_accuracy_mean_fold",
        "sensitivity_mean_fold",
        "specificity_mean_fold",
        "brier_mean_fold",
        "ece_mean_fold",
        "selected_latent_setting_mode",
    ]
    print(summary[display_cols].head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
