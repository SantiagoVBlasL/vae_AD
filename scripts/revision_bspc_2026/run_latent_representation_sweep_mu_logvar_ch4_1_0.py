#!/usr/bin/env python3
"""ADNI-only latent representation sweep using exported mu/logvar features.

The sweep consumes fold-wise CSVs exported from saved VAE checkpoints. All
preprocessing, PCA, hyperparameter selection, ensembling, and calibration are
fit within each outer fold train/dev split. Held-out test folds are scored once.
Manufacturer is retained only as metadata in the export and is never used as a
predictive feature here.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LATENT_EXPORT_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/auc_sprint_adni_only/latent_exports_mu_logvar_ch4_1_0"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/auc_sprint_adni_only/latent_representation_sweep_mu_logvar_ch4_1_0"
)

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)


class PLSFeatureExtractor(BaseEstimator, TransformerMixin):
    """Pipeline-safe PLS transformer that returns only X scores."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSFeatureExtractor":
        self.model_ = PLSRegression(n_components=int(self.n_components), scale=False)
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
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = resolve(path)
    path.mkdir(parents=True, exist_ok=True)
    generated = [
        "latent_representation_sweep_metrics.csv",
        "latent_representation_sweep_fold_metrics.csv",
        "latent_representation_sweep_predictions.csv",
        "latent_representation_sweep_failures.csv",
        "latent_representation_sweep_skipped.csv",
        "latent_representation_sweep_manifest.json",
        "README.md",
    ]
    existing = [path / name for name in generated if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{path} already contains generated sweep outputs; pass --overwrite")
    if overwrite:
        for file_path in existing:
            file_path.unlink()
    return path


def required_latent_paths(latent_dir: Path) -> Dict[int, Dict[str, Path]]:
    return {
        fold: {
            "trainDev": latent_dir / f"fold_{fold}_trainDev_latents_mu_logvar.csv",
            "test": latent_dir / f"fold_{fold}_test_latents_mu_logvar.csv",
            "index": latent_dir / f"fold_{fold}_latent_subject_index.csv",
        }
        for fold in range(1, 6)
    }


def validate_latent_files(latent_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    report: Dict[str, Any] = {"latent_export_dir": str(latent_dir), "folds": []}
    ok = True
    for fold, fold_paths in required_latent_paths(latent_dir).items():
        row = {"fold": fold}
        for key, path in fold_paths.items():
            row[f"{key}_path"] = str(path)
            row[f"{key}_exists"] = path.exists()
            if key != "index" and not path.exists():
                ok = False
        report["folds"].append(row)
    return ok, report


def natural_suffix_key(column: str) -> Tuple[str, int]:
    prefix, _, suffix = column.rpartition("_")
    try:
        return prefix, int(suffix)
    except ValueError:
        return prefix, 10**9


def prefixed_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    token = f"{prefix}_"
    return sorted([c for c in df.columns if c.startswith(token)], key=natural_suffix_key)


def summary_uncertainty_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "posterior_entropy",
        "posterior_entropy_mean",
        "posterior_entropy_std",
        "posterior_std_mean",
        "posterior_std_std",
        "posterior_std_min",
        "posterior_std_max",
        "logvar_mean",
        "logvar_std",
        "logvar_min",
        "logvar_max",
    ]
    return [c for c in candidates if c in df.columns]


def load_fold_split(latent_dir: Path, fold: int, split: str) -> pd.DataFrame:
    path = latent_dir / f"fold_{fold}_{split}_latents_mu_logvar.csv"
    df = pd.read_csv(path)
    if "SubjectID" not in df.columns:
        raise ValueError(f"{path} is missing SubjectID")
    if "y" not in df.columns:
        df["y"] = df["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1})
    df["y_true"] = pd.to_numeric(df["y"], errors="coerce")
    df = df[df["y_true"].notna()].copy()
    df["y_true"] = df["y_true"].astype(int)
    return df


def representation_specs(train: pd.DataFrame) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    mu = prefixed_columns(train, "mu")
    logvar = prefixed_columns(train, "logvar")
    entropy_summary = summary_uncertainty_columns(train)
    skipped: List[Dict[str, Any]] = []
    if not mu:
        raise ValueError("No mu_* columns found in latent export")
    if not logvar:
        raise ValueError("No logvar_* columns found in latent export")
    if not entropy_summary:
        skipped.append(
            {
                "representation": "mu_plus_entropy_summary",
                "classifier": "all",
                "status": "skipped",
                "reason": "posterior entropy summary columns are absent from the export",
            }
        )

    specs = {
        "mu": {"feature_cols": mu, "use_pca": False},
        "logvar": {"feature_cols": logvar, "use_pca": False},
        "mu_plus_logvar": {"feature_cols": mu + logvar, "use_pca": False},
        "mu_plus_entropy_summary": {"feature_cols": mu + entropy_summary, "use_pca": False},
        "mu_plus_logvar_PCA": {"feature_cols": mu + logvar, "use_pca": True},
    }
    if not entropy_summary:
        specs.pop("mu_plus_entropy_summary")
    skipped.append(
        {
            "representation": "mc_z_sampling",
            "classifier": "all",
            "status": "skipped",
            "reason": (
                "A valid MC-z audit would need subject-grouped stochastic resampling inside each inner CV "
                "fold; this script keeps the minimal deterministic posterior-parameter sweep."
            ),
        }
    )
    return specs, skipped


def make_feature_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    representation_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    feature_cols = list(representation_cols) + ["Age", "Sex"]
    forbidden = {"Manufacturer", "Site3", "SourceCohort", "InputCohort"}
    used_forbidden = [c for c in feature_cols if c in forbidden or c.lower().startswith(("manufacturer", "site"))]
    if used_forbidden:
        raise ValueError(f"Forbidden scanner/site feature requested: {used_forbidden}")
    missing_train = [c for c in feature_cols if c not in train.columns]
    missing_test = [c for c in feature_cols if c not in test.columns]
    if missing_train or missing_test:
        raise ValueError(f"Missing feature columns: train={missing_train}, test={missing_test}")

    train_x = train[feature_cols].copy()
    test_x = test[feature_cols].copy()
    for frame in [train_x, test_x]:
        for col in frame.columns:
            if col != "Sex":
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["Sex"] = frame["Sex"].astype(str).replace({"nan": np.nan, "None": np.nan})
    train_x = pd.get_dummies(train_x, columns=["Sex"], drop_first=False, dummy_na=False)
    test_x = pd.get_dummies(test_x, columns=["Sex"], drop_first=False, dummy_na=False)
    test_x = test_x.reindex(columns=train_x.columns, fill_value=0)
    train_x = train_x.astype(float)
    test_x = test_x.astype(float)
    return train_x, test_x, feature_cols


def make_pipeline(clf: Any, use_pca: bool, scale: bool = True) -> Pipeline:
    steps: List[Tuple[str, Any]] = [("impute", SimpleImputer())]
    if scale or use_pca:
        steps.append(("scale", StandardScaler()))
    if use_pca:
        steps.append(("pca", PCA(svd_solver="full", random_state=42)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def add_pca_grid(grid: Dict[str, List[Any]], use_pca: bool) -> Dict[str, List[Any]]:
    out = dict(grid)
    if use_pca:
        out["pca__n_components"] = [0.80, 0.90, 0.95]
    return out


def classifier_grid(use_pca: bool, seed: int) -> Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]]:
    models: Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]] = {
        "logreg_l2": (
            make_pipeline(
                LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear", random_state=seed),
                use_pca=use_pca,
            ),
            add_pca_grid({"clf__C": [0.01, 0.1, 1.0, 10.0]}, use_pca),
        ),
        "logreg_elasticnet": (
            make_pipeline(
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="saga",
                    penalty="elasticnet",
                    random_state=seed,
                ),
                use_pca=use_pca,
            ),
            add_pca_grid({"clf__C": [0.01, 0.1, 1.0], "clf__l1_ratio": [0.1, 0.5, 0.9]}, use_pca),
        ),
        "linear_svm": (
            make_pipeline(
                LinearSVC(class_weight="balanced", dual="auto", max_iter=10000, random_state=seed),
                use_pca=use_pca,
            ),
            add_pca_grid({"clf__C": [0.01, 0.1, 1.0]}, use_pca),
        ),
        "rbf_svm": (
            make_pipeline(SVC(kernel="rbf", class_weight="balanced", random_state=seed), use_pca=use_pca),
            add_pca_grid({"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01]}, use_pca),
        ),
        "lda_shrinkage": (
            make_pipeline(
                LinearDiscriminantAnalysis(solver="lsqr"),
                use_pca=use_pca,
            ),
            add_pca_grid({"clf__shrinkage": ["auto", 0.1, 0.5, 0.9]}, use_pca),
        ),
        "pls_logreg": (
            Pipeline(
                [
                    ("impute", SimpleImputer()),
                    ("scale", StandardScaler()),
                    *([("pca", PCA(svd_solver="full", random_state=42))] if use_pca else []),
                    ("pls", PLSFeatureExtractor()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=5000,
                            class_weight="balanced",
                            solver="liblinear",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            add_pca_grid({"pls__n_components": [2, 4, 8], "clf__C": [0.1, 1.0, 10.0]}, use_pca),
        ),
        "extra_trees": (
            make_pipeline(
                ExtraTreesClassifier(class_weight="balanced", random_state=seed, n_jobs=1),
                use_pca=use_pca,
                scale=False,
            ),
            add_pca_grid(
                {
                    "clf__n_estimators": [200],
                    "clf__max_depth": [None, 4, 8],
                    "clf__min_samples_leaf": [1, 3, 5],
                },
                use_pca,
            ),
        ),
    }
    try:
        from lightgbm import LGBMClassifier  # type: ignore

        models["lightgbm"] = (
            make_pipeline(
                LGBMClassifier(
                    random_state=seed,
                    class_weight="balanced",
                    n_jobs=1,
                    verbosity=-1,
                ),
                use_pca=use_pca,
                scale=False,
            ),
            add_pca_grid(
                {
                    "clf__n_estimators": [100],
                    "clf__num_leaves": [7, 15],
                    "clf__learning_rate": [0.05],
                },
                use_pca,
            ),
        )
    except Exception:
        pass
    try:
        from xgboost import XGBClassifier  # type: ignore

        models["xgboost"] = (
            make_pipeline(
                XGBClassifier(
                    random_state=seed,
                    eval_metric="logloss",
                    n_jobs=1,
                    tree_method="hist",
                    verbosity=0,
                ),
                use_pca=use_pca,
                scale=False,
            ),
            add_pca_grid(
                {
                    "clf__n_estimators": [100],
                    "clf__max_depth": [2, 3],
                    "clf__learning_rate": [0.05],
                },
                use_pca,
            ),
        )
    except Exception:
        pass
    return models


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
    if n == 0:
        return np.nan
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


def fold_metric_row(
    representation: str,
    classifier: str,
    fold: int,
    inner_cv_auc: float,
    y: np.ndarray,
    score: np.ndarray,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "representation": representation,
        "classifier": classifier,
        "fold": fold,
        "inner_cv_auc_trainDev": inner_cv_auc,
        "status": "ok",
    }
    row.update(metrics_from_scores(y, score))
    if extra:
        row.update(extra)
    return row


def aggregate_metrics(fold_metrics: pd.DataFrame, preds: pd.DataFrame, skipped: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
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
    for (representation, classifier), grp in fold_metrics.groupby(["representation", "classifier"], sort=False):
        pred_grp = preds[(preds["representation"] == representation) & (preds["classifier"] == classifier)]
        y = pred_grp["y_true"].astype(int).to_numpy()
        score = pred_grp["y_score"].astype(float).to_numpy()
        pooled = metrics_from_scores(y, score)
        row: Dict[str, Any] = {
            "representation": representation,
            "classifier": classifier,
            "n_total": int(len(y)),
            "n_CN": int((y == 0).sum()),
            "n_AD": int((y == 1).sum()),
            "n_folds": int(grp["fold"].nunique()),
            "status": "ok",
            "inner_cv_auc_mean_fold": float(grp["inner_cv_auc_trainDev"].mean()),
            "inner_cv_auc_sd_fold": float(grp["inner_cv_auc_trainDev"].std(ddof=1)),
        }
        for col in metric_cols:
            row[f"{col}_mean_fold"] = float(grp[col].mean())
            row[f"{col}_sd_fold"] = float(grp[col].std(ddof=1))
            row[f"{col}_pooled"] = pooled[col]
        row.update(
            {
                "threshold_0p5_TN": pooled["tn"],
                "threshold_0p5_FP": pooled["fp"],
                "threshold_0p5_FN": pooled["fn"],
                "threshold_0p5_TP": pooled["tp"],
            }
        )
        rows.append(row)
    completed = pd.DataFrame(rows)
    if not completed.empty:
        completed = completed.sort_values(
            ["roc_auc_mean_fold", "balanced_accuracy_mean_fold", "brier_mean_fold"],
            ascending=[False, False, True],
        )
    skipped_rows = pd.DataFrame(skipped)
    if not skipped_rows.empty:
        for col in completed.columns if not completed.empty else ["representation", "classifier", "status"]:
            if col not in skipped_rows.columns:
                skipped_rows[col] = np.nan
        skipped_rows = skipped_rows[completed.columns] if not completed.empty else skipped_rows
    return pd.concat([completed, skipped_rows], ignore_index=True, sort=False)


def write_missing_report(outdir: Path, report: Dict[str, Any]) -> None:
    (outdir / "latent_representation_sweep_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Latent Representation Sweep Blocked",
        "",
        "No classifier sweep was run because exported mu/logvar CSVs are missing.",
        "",
        "Run:",
        "",
        "```bash",
        "/home/diego/anaconda3/envs/vae_ad/bin/python scripts/revision_bspc_2026/export_v4_ch4_1_0_fold_latents_mu_logvar.py",
        "```",
    ]
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    latent_dir = resolve(args.latent_export_dir)
    outdir = prepare_output_dir(args.output_dir, args.overwrite)
    ok, report = validate_latent_files(latent_dir)
    if not ok:
        write_missing_report(outdir, report)
        raise RuntimeError(f"Latent representation sweep blocked; exported latents are missing under {latent_dir}")

    all_fold_metrics: List[Dict[str, Any]] = []
    all_preds: List[pd.DataFrame] = []
    failures: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    feature_counts: Dict[str, int] = {}

    for fold in range(1, 6):
        train = load_fold_split(latent_dir, fold, "trainDev")
        test = load_fold_split(latent_dir, fold, "test")
        specs, fold_skipped = representation_specs(train)
        for item in fold_skipped:
            if item not in skipped:
                skipped.append(item)

        y_train = train["y_true"].astype(int).to_numpy()
        y_test = test["y_true"].astype(int).to_numpy()
        cv = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.random_seed)

        for representation, spec in specs.items():
            train_x, test_x, requested_cols = make_feature_matrix(train, test, spec["feature_cols"])
            feature_counts[representation] = len(requested_cols)
            models = classifier_grid(use_pca=bool(spec["use_pca"]), seed=args.random_seed)
            fold_best: List[Tuple[str, float, Any]] = []

            for name, (pipe, grid) in models.items():
                try:
                    search = GridSearchCV(
                        pipe,
                        grid,
                        scoring="roc_auc",
                        cv=cv,
                        n_jobs=args.n_jobs,
                        error_score=np.nan,
                        refit=True,
                    )
                    search.fit(train_x, y_train)
                    if pd.isna(search.best_score_):
                        raise RuntimeError("all inner-CV scores are NaN")
                    best = search.best_estimator_
                    calibrated: Any
                    calibration_status = "sigmoid_cv3"
                    try:
                        calibrated = CalibratedClassifierCV(best, method="sigmoid", cv=3)
                        calibrated.fit(train_x, y_train)
                    except Exception as exc:
                        calibrated = best
                        calibration_status = f"uncalibrated_fallback:{type(exc).__name__}"
                    score = score_model(calibrated, test_x)
                    all_fold_metrics.append(
                        fold_metric_row(
                            representation,
                            name,
                            fold,
                            float(search.best_score_),
                            y_test,
                            score,
                            {
                                "n_requested_features_with_age_sex": len(requested_cols),
                                "n_design_matrix_columns": int(train_x.shape[1]),
                                "calibration_status": calibration_status,
                                "best_params": json.dumps(search.best_params_, sort_keys=True),
                            },
                        )
                    )
                    all_preds.append(
                        pd.DataFrame(
                            {
                                "SubjectID": test["SubjectID"].values,
                                "fold": fold,
                                "representation": representation,
                                "classifier": name,
                                "y_true": y_test,
                                "y_score": score,
                                "y_pred": (score >= 0.5).astype(int),
                            }
                        )
                    )
                    fold_best.append((name, float(search.best_score_), calibrated))
                except Exception as exc:
                    failures.append(
                        {
                            "fold": fold,
                            "representation": representation,
                            "classifier": name,
                            "status": "failed",
                            "error": repr(exc),
                        }
                    )

            top = sorted(fold_best, key=lambda item: item[1], reverse=True)[:3]
            if top:
                ensemble_score = np.mean([score_model(model, test_x) for _name, _cv_auc, model in top], axis=0)
                ensemble_name = "soft_probability_ensemble_top3_trainDev_selected"
                all_fold_metrics.append(
                    fold_metric_row(
                        representation,
                        ensemble_name,
                        fold,
                        float(np.mean([cv_auc for _name, cv_auc, _model in top])),
                        y_test,
                        ensemble_score,
                        {
                            "n_requested_features_with_age_sex": len(requested_cols),
                            "n_design_matrix_columns": int(train_x.shape[1]),
                            "calibration_status": "member_models_calibrated_or_fallback",
                            "ensemble_members": ",".join(name for name, _cv_auc, _model in top),
                        },
                    )
                )
                all_preds.append(
                    pd.DataFrame(
                        {
                            "SubjectID": test["SubjectID"].values,
                            "fold": fold,
                            "representation": representation,
                            "classifier": ensemble_name,
                            "y_true": y_test,
                            "y_score": ensemble_score,
                            "y_pred": (ensemble_score >= 0.5).astype(int),
                        }
                    )
                )

    if not all_fold_metrics:
        raise RuntimeError("No representation/classifier completed successfully.")

    fold_metrics = pd.DataFrame(all_fold_metrics)
    preds = pd.concat(all_preds, ignore_index=True)
    summary = aggregate_metrics(fold_metrics, preds, skipped)

    summary.to_csv(outdir / "latent_representation_sweep_metrics.csv", index=False)
    fold_metrics.to_csv(outdir / "latent_representation_sweep_fold_metrics.csv", index=False)
    preds.to_csv(outdir / "latent_representation_sweep_predictions.csv", index=False)
    pd.DataFrame(failures).to_csv(outdir / "latent_representation_sweep_failures.csv", index=False)
    pd.DataFrame(skipped).to_csv(outdir / "latent_representation_sweep_skipped.csv", index=False)

    best = summary[summary["status"] == "ok"].iloc[0]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "latent_export_dir": str(latent_dir),
        "latent_export_dir_realpath": str(latent_dir.resolve()),
        "output_dir": str(outdir),
        "output_dir_realpath": str(outdir.resolve()),
        "representations_completed": sorted(fold_metrics["representation"].unique().tolist()),
        "feature_counts_with_age_sex": feature_counts,
        "metadata_features": ["Age", "Sex"],
        "manufacturer_site_features_used": False,
        "vae_retrained": False,
        "architecture_modified": False,
        "model_selection_split": "outer-fold trainDev only",
        "inner_folds": args.inner_folds,
        "calibration": "sigmoid calibration fit within trainDev only, with uncalibrated fallback recorded per fold",
        "skipped": skipped,
        "n_failures": len(failures),
    }
    (outdir / "latent_representation_sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Latent Representation Sweep: Mu/Logvar",
        "",
        "ADNI-only, inference-only classifier sweep over frozen V4 [4,1,0] posterior exports.",
        "",
        "- No VAE training was run.",
        "- No architecture was modified.",
        "- Age and Sex were included as covariates.",
        "- Manufacturer/Site were not used as predictive features.",
        "- PCA, scaling, imputation, classifier selection, calibration, and ensemble member selection were fit inside train/dev only.",
        "",
        "## Best Mean-Fold ROC-AUC",
        "",
        f"- Representation: `{best['representation']}`",
        f"- Classifier: `{best['classifier']}`",
        f"- Mean-fold ROC-AUC: {best['roc_auc_mean_fold']:.4f} +/- {best['roc_auc_sd_fold']:.4f}",
        f"- Pooled ROC-AUC: {best['roc_auc_pooled']:.4f}",
        f"- Mean-fold PR-AUC: {best['pr_auc_mean_fold']:.4f}",
        f"- Mean-fold balanced accuracy: {best['balanced_accuracy_mean_fold']:.4f}",
        f"- Mean-fold sensitivity: {best['sensitivity_mean_fold']:.4f}",
        f"- Mean-fold specificity: {best['specificity_mean_fold']:.4f}",
        f"- Mean-fold Brier: {best['brier_mean_fold']:.4f}",
        f"- Mean-fold ECE: {best['ece_mean_fold']:.4f}",
    ]
    if skipped:
        lines.extend(["", "## Skipped Representation Sets", ""])
        for item in skipped:
            lines.append(f"- `{item['representation']}`: {item['reason']}")
    if failures:
        lines.extend(["", "## Failures", "", f"{len(failures)} fold/model fits failed; see `latent_representation_sweep_failures.csv`."])
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    display_cols = [
        "representation",
        "classifier",
        "status",
        "roc_auc_mean_fold",
        "roc_auc_sd_fold",
        "roc_auc_pooled",
        "pr_auc_mean_fold",
        "balanced_accuracy_mean_fold",
        "sensitivity_mean_fold",
        "specificity_mean_fold",
        "brier_mean_fold",
        "ece_mean_fold",
    ]
    print(summary[display_cols].head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
