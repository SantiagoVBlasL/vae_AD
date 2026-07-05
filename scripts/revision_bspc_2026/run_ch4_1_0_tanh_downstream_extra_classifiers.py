#!/usr/bin/env python3
"""Downstream-only classifier sweep for frozen tanh V4 [4,1,0] mu latents.

This script never trains or loads the CVAE. It consumes exported fold-wise mu
CSV files from the completed tanh baseline, fits downstream classifiers inside
each outer fold train/dev split, and evaluates once on the held-out test fold.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LATENT_EXPORT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/latent_exports_ch4_1_0"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/auc_sprint_adni_only/ch4_1_0_tanh_downstream_extra_classifiers"
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


class PLSFeatureExtractor(BaseEstimator, TransformerMixin):
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
        "downstream_extra_classifier_metrics.csv",
        "downstream_extra_classifier_fold_metrics.csv",
        "downstream_extra_classifier_predictions.csv",
        "downstream_extra_classifier_failures.csv",
        "downstream_extra_classifier_manifest.json",
        "README.md",
    ]
    existing = [path / name for name in generated if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{path} already contains generated outputs; pass --overwrite")
    if overwrite:
        for file_path in existing:
            file_path.unlink()
    return path


def validate_latent_files(latent_dir: Path) -> None:
    missing: List[str] = []
    for fold in range(1, 6):
        for split in ["trainDev", "test"]:
            path = latent_dir / f"fold_{fold}_{split}_latents_mu.csv"
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing exported tanh mu latents:\n" + "\n".join(missing))


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
    path = latent_dir / f"fold_{fold}_{split}_latents_mu.csv"
    df = pd.read_csv(path)
    if "y" not in df.columns:
        df["y"] = df["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1})
    df["y_true"] = pd.to_numeric(df["y"], errors="coerce")
    df = df[df["y_true"].notna()].copy()
    df["y_true"] = df["y_true"].astype(int)
    return df


def make_feature_matrix(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    latent_cols = mu_columns(train)
    feature_cols = latent_cols + ["Age", "Sex"]
    forbidden = {"Manufacturer", "Site3", "SourceCohort", "InputCohort"}
    if any(c in forbidden or c.lower().startswith(("manufacturer", "site")) for c in feature_cols):
        raise ValueError("Forbidden scanner/site feature requested")
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
    train_x = pd.get_dummies(train_x, columns=["Sex"], drop_first=False)
    test_x = pd.get_dummies(test_x, columns=["Sex"], drop_first=False)
    test_x = test_x.reindex(columns=train_x.columns, fill_value=0)
    return train_x.astype(float), test_x.astype(float), feature_cols


def classifier_grid(seed: int) -> Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]]:
    return {
        "logreg_l2": (
            Pipeline(
                [
                    ("impute", SimpleImputer()),
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear", random_state=seed)),
                ]
            ),
            {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "svm_rbf": (
            Pipeline(
                [
                    ("impute", SimpleImputer()),
                    ("scale", StandardScaler()),
                    ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=seed)),
                ]
            ),
            {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.001, 0.01]},
        ),
        "lda_shrinkage": (
            Pipeline(
                [
                    ("impute", SimpleImputer()),
                    ("scale", StandardScaler()),
                    ("clf", LinearDiscriminantAnalysis(solver="lsqr")),
                ]
            ),
            {"clf__shrinkage": ["auto", 0.1, 0.5, 0.9]},
        ),
        "pls_logreg": (
            Pipeline(
                [
                    ("impute", SimpleImputer()),
                    ("scale", StandardScaler()),
                    ("pls", PLSFeatureExtractor()),
                    ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear", random_state=seed)),
                ]
            ),
            {"pls__n_components": [2, 4, 8, 16], "clf__C": [0.1, 1.0, 10.0]},
        ),
        "calibrated_ridge": (
            Pipeline(
                [
                    ("impute", SimpleImputer()),
                    ("scale", StandardScaler()),
                    ("clf", RidgeClassifier(class_weight="balanced", random_state=seed)),
                ]
            ),
            {"clf__alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
    }


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


def fold_metric_row(classifier: str, fold: int, inner_cv_auc: float, y: np.ndarray, score: np.ndarray, extra: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"classifier": classifier, "fold": fold, "inner_cv_auc_trainDev": inner_cv_auc, "status": "ok"}
    row.update(metrics_from_scores(y, score))
    row.update(extra)
    return row


def aggregate_metrics(fold_metrics: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "brier", "ece"]
    rows: List[Dict[str, Any]] = []
    for classifier, group in fold_metrics.groupby("classifier", sort=False):
        pred_group = preds[preds["classifier"] == classifier]
        y = pred_group["y_true"].astype(int).to_numpy()
        score = pred_group["y_score"].astype(float).to_numpy()
        pooled = metrics_from_scores(y, score)
        row: Dict[str, Any] = {
            "classifier": classifier,
            "n_total": int(len(y)),
            "n_CN": int((y == 0).sum()),
            "n_AD": int((y == 1).sum()),
            "n_folds": int(group["fold"].nunique()),
            "status": "ok",
            "inner_cv_auc_mean_fold": float(group["inner_cv_auc_trainDev"].mean()),
            "inner_cv_auc_sd_fold": float(group["inner_cv_auc_trainDev"].std(ddof=1)),
        }
        for col in metric_cols:
            row[f"{col}_mean_fold"] = float(group[col].mean())
            row[f"{col}_sd_fold"] = float(group[col].std(ddof=1))
            row[f"{col}_pooled"] = pooled[col]
        row.update({"threshold_0p5_TN": pooled["tn"], "threshold_0p5_FP": pooled["fp"], "threshold_0p5_FN": pooled["fn"], "threshold_0p5_TP": pooled["tp"]})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["roc_auc_mean_fold", "balanced_accuracy_mean_fold"], ascending=False)


def main() -> int:
    args = parse_args()
    latent_dir = resolve(args.latent_export_dir)
    outdir = prepare_output_dir(args.output_dir, args.overwrite)
    validate_latent_files(latent_dir)

    fold_metrics_rows: List[Dict[str, Any]] = []
    pred_frames: List[pd.DataFrame] = []
    failures: List[Dict[str, Any]] = []

    for fold in range(1, 6):
        train = load_fold_split(latent_dir, fold, "trainDev")
        test = load_fold_split(latent_dir, fold, "test")
        train_x, test_x, feature_cols = make_feature_matrix(train, test)
        y_train = train["y_true"].astype(int).to_numpy()
        y_test = test["y_true"].astype(int).to_numpy()
        cv = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.random_seed)
        fold_best: List[Tuple[str, float, Any]] = []

        for name, (pipe, grid) in classifier_grid(args.random_seed).items():
            try:
                search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=args.n_jobs, error_score="raise")
                search.fit(train_x, y_train)
                best = search.best_estimator_
                calibrated = CalibratedClassifierCV(best, method="sigmoid", cv=3)
                calibrated.fit(train_x, y_train)
                score = score_model(calibrated, test_x)
                fold_metrics_rows.append(
                    fold_metric_row(
                        name,
                        fold,
                        float(search.best_score_),
                        y_test,
                        score,
                        {
                            "n_requested_features_with_age_sex": len(feature_cols),
                            "n_design_matrix_columns": int(train_x.shape[1]),
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
                            "classifier": name,
                            "y_true": y_test,
                            "y_score": score,
                            "y_pred": (score >= 0.5).astype(int),
                        }
                    )
                )
                fold_best.append((name, float(search.best_score_), calibrated))
            except Exception as exc:
                failures.append({"fold": fold, "classifier": name, "status": "failed", "error": repr(exc)})

        top = sorted(fold_best, key=lambda item: item[1], reverse=True)[:3]
        if top:
            ensemble_score = np.mean([score_model(model, test_x) for _name, _cv_auc, model in top], axis=0)
            ensemble_name = "soft_probability_ensemble_top3_trainDev_selected"
            fold_metrics_rows.append(
                fold_metric_row(
                    ensemble_name,
                    fold,
                    float(np.mean([cv_auc for _name, cv_auc, _model in top])),
                    y_test,
                    ensemble_score,
                    {
                        "n_requested_features_with_age_sex": len(feature_cols),
                        "n_design_matrix_columns": int(train_x.shape[1]),
                        "calibration_status": "member_models_calibrated",
                        "ensemble_members": ",".join(name for name, _cv_auc, _model in top),
                    },
                )
            )
            pred_frames.append(
                pd.DataFrame(
                    {
                        "SubjectID": test["SubjectID"].values,
                        "fold": fold,
                        "classifier": ensemble_name,
                        "y_true": y_test,
                        "y_score": ensemble_score,
                        "y_pred": (ensemble_score >= 0.5).astype(int),
                    }
                )
            )

    if not fold_metrics_rows:
        raise RuntimeError("No downstream classifier completed successfully.")

    fold_metrics = pd.DataFrame(fold_metrics_rows)
    preds = pd.concat(pred_frames, ignore_index=True)
    summary = aggregate_metrics(fold_metrics, preds)

    summary.to_csv(outdir / "downstream_extra_classifier_metrics.csv", index=False)
    fold_metrics.to_csv(outdir / "downstream_extra_classifier_fold_metrics.csv", index=False)
    preds.to_csv(outdir / "downstream_extra_classifier_predictions.csv", index=False)
    pd.DataFrame(failures).to_csv(outdir / "downstream_extra_classifier_failures.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "latent_export_dir": str(latent_dir),
        "latent_export_dir_realpath": str(latent_dir.resolve()),
        "output_dir": str(outdir),
        "output_dir_realpath": str(outdir.resolve()),
        "vae_retrained": False,
        "architecture_modified": False,
        "latent_features": "mu",
        "metadata_features": ["Age", "Sex"],
        "manufacturer_site_features_used": False,
        "model_selection_split": "outer-fold trainDev only",
        "classifiers": sorted(summary["classifier"].tolist()),
        "n_failures": len(failures),
    }
    (outdir / "downstream_extra_classifier_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    best = summary.iloc[0]
    lines = [
        "# Tanh [4,1,0] Downstream Extra Classifiers",
        "",
        "No VAE retraining was run. This sweep uses exported `mu` latents from the completed tanh baseline.",
        "",
        "- Predictive features: `mu` + Age + Sex.",
        "- Manufacturer/Site features were not used.",
        "- All imputation, scaling, classifier selection, calibration, and ensemble member selection occurred inside each outer train/dev split.",
        "",
        "## Best Result",
        f"- Classifier: `{best['classifier']}`",
        f"- Mean-fold ROC-AUC: {best['roc_auc_mean_fold']:.4f} +/- {best['roc_auc_sd_fold']:.4f}",
        f"- Pooled ROC-AUC: {best['roc_auc_pooled']:.4f}",
        f"- Mean-fold PR-AUC: {best['pr_auc_mean_fold']:.4f}",
        f"- Mean-fold balanced accuracy: {best['balanced_accuracy_mean_fold']:.4f}",
    ]
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    display_cols = ["classifier", "roc_auc_mean_fold", "roc_auc_sd_fold", "roc_auc_pooled", "pr_auc_mean_fold", "balanced_accuracy_mean_fold", "sensitivity_mean_fold", "specificity_mean_fold", "brier_mean_fold", "ece_mean_fold"]
    print(summary[display_cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
