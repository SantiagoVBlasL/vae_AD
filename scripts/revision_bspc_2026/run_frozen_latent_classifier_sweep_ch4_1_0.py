#!/usr/bin/env python3
"""Frozen latent classifier sweep for V4 [4,1,0].

The sweep consumes pre-exported fold-wise VAE ``mu`` embeddings and never loads
VAE checkpoints, tensors, joblibs, or test folds for model selection. Models are
selected and calibrated inside each outer fold's train/dev split, then evaluated
once on the preserved held-out test fold.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PROJECT_ROOT / "results/revision_bspc_2026/adni_expanded_v4_beta25_ch4_1_0"
DEFAULT_METADATA = PROJECT_ROOT / "data/revision_bspc_2026/adni_expanded_v4_all_available/subject_metadata_adni_expanded_v4_all_available.csv"
DEFAULT_LATENT_EXPORT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/latent_exports_ch4_1_0"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/frozen_latent_classifier_sweep_ch4_1_0"

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--latent-export-dir", type=Path, default=DEFAULT_LATENT_EXPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = resolve(path)
    path.mkdir(parents=True, exist_ok=True)
    generated = [
        "frozen_latent_classifier_sweep_metrics.csv",
        "frozen_latent_classifier_sweep_fold_metrics.csv",
        "frozen_latent_classifier_sweep_predictions.csv",
        "frozen_latent_classifier_sweep_readme.md",
        "frozen_latent_classifier_sweep_manifest.json",
        "latent_artifact_check.json",
    ]
    existing = [path / name for name in generated if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{path} already contains sweep outputs; pass --overwrite")
    if overwrite:
        for file_path in existing:
            file_path.unlink()
    return path


def required_latent_paths(latent_dir: Path) -> Dict[int, Dict[str, Path]]:
    paths: Dict[int, Dict[str, Path]] = {}
    for fold in range(1, 6):
        paths[fold] = {
            "trainDev": latent_dir / f"fold_{fold}_trainDev_latents_mu.csv",
            "test": latent_dir / f"fold_{fold}_test_latents_mu.csv",
            "index": latent_dir / f"fold_{fold}_latent_subject_index.csv",
        }
    return paths


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


def write_missing_report(outdir: Path, report: Dict[str, Any]) -> None:
    (outdir / "frozen_latent_classifier_sweep_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Frozen Latent Classifier Sweep Blocked",
        "",
        "No classifier sweep was run.",
        "",
        "Direct exported fold-wise latent CSVs are missing. Run:",
        "",
        "```bash",
        "/home/diego/anaconda3/envs/vae_ad/bin/python scripts/revision_bspc_2026/export_v4_ch4_1_0_fold_latents_mu.py --overwrite",
        "```",
        "",
        "Required files per fold:",
        "- `fold_k_trainDev_latents_mu.csv`",
        "- `fold_k_test_latents_mu.csv`",
        "- `fold_k_latent_subject_index.csv`",
    ]
    (outdir / "frozen_latent_classifier_sweep_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(columns=["classifier", "status"]).to_csv(outdir / "frozen_latent_classifier_sweep_metrics.csv", index=False)
    pd.DataFrame(columns=["SubjectID", "fold", "classifier", "y_true", "y_score", "y_pred"]).to_csv(
        outdir / "frozen_latent_classifier_sweep_predictions.csv", index=False
    )


def latent_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.lower().startswith(("mu_", "z_", "latent_", "dim_"))]
    if not cols:
        raise ValueError("No latent feature columns found; expected columns like mu_000")
    return cols


def load_fold_split(latent_dir: Path, fold: int, split: str, metadata: pd.DataFrame) -> pd.DataFrame:
    path = latent_dir / f"fold_{fold}_{split}_latents_mu.csv"
    df = pd.read_csv(path)
    if "SubjectID" not in df.columns:
        raise ValueError(f"{path} is missing SubjectID")
    if "y" not in df.columns:
        df["y"] = df["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1})
    if "Age" not in df.columns or "Sex" not in df.columns:
        meta_cols = [c for c in ["SubjectID", "Age", "Sex"] if c in metadata.columns]
        df = df.merge(metadata[meta_cols].drop_duplicates("SubjectID"), on="SubjectID", how="left", suffixes=("", "_meta"))
        for col in ["Age", "Sex"]:
            meta_col = f"{col}_meta"
            if meta_col in df.columns:
                df[col] = df[col].fillna(df[meta_col]) if col in df.columns else df[meta_col]
                df = df.drop(columns=[meta_col])
    df["y_true"] = pd.to_numeric(df["y"], errors="coerce")
    df = df[df["y_true"].notna()].copy()
    df["y_true"] = df["y_true"].astype(int)
    return df


def make_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    latent_cols = latent_feature_columns(train)
    feature_cols = latent_cols + ["Age", "Sex"]
    missing_train = [c for c in feature_cols if c not in train.columns]
    missing_test = [c for c in feature_cols if c not in test.columns]
    if missing_train or missing_test:
        raise ValueError(f"Missing feature columns: train={missing_train}, test={missing_test}")
    train_x = pd.get_dummies(train[feature_cols], columns=["Sex"], drop_first=False)
    test_x = pd.get_dummies(test[feature_cols], columns=["Sex"], drop_first=False)
    test_x = test_x.reindex(columns=train_x.columns, fill_value=0)
    return train_x, test_x, latent_cols


def classifier_grid(n_jobs: int) -> Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]]:
    models: Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]] = {
        "logreg_l2": (
            Pipeline(
                [
                    ("impute", SimpleImputer()),
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")),
                ]
            ),
            {"clf__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "logreg_elasticnet": (
            Pipeline(
                [
                    ("impute", SimpleImputer()),
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=5000,
                            class_weight="balanced",
                            solver="saga",
                            penalty="elasticnet",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            {"clf__C": [0.01, 0.1, 1, 10], "clf__l1_ratio": [0.1, 0.5, 0.9]},
        ),
        "linear_svm": (
            Pipeline(
                [
                    ("impute", SimpleImputer()),
                    ("scale", StandardScaler()),
                    ("clf", LinearSVC(class_weight="balanced", dual="auto", max_iter=10000, random_state=42)),
                ]
            ),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "rbf_svm": (
            Pipeline(
                [
                    ("impute", SimpleImputer()),
                    ("scale", StandardScaler()),
                    ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=42)),
                ]
            ),
            {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.001, 0.01]},
        ),
        "extra_trees": (
            Pipeline(
                [
                    ("impute", SimpleImputer()),
                    ("clf", ExtraTreesClassifier(class_weight="balanced", random_state=42, n_jobs=n_jobs)),
                ]
            ),
            {"clf__n_estimators": [300], "clf__max_depth": [None, 4, 8], "clf__min_samples_leaf": [1, 3, 5]},
        ),
        "shallow_mlp": (
            Pipeline(
                [
                    ("impute", SimpleImputer()),
                    ("scale", StandardScaler()),
                    ("clf", MLPClassifier(random_state=42, max_iter=800, early_stopping=True)),
                ]
            ),
            {"clf__hidden_layer_sizes": [(64,), (64, 16)], "clf__alpha": [0.0001, 0.001]},
        ),
    }
    try:
        from lightgbm import LGBMClassifier  # type: ignore

        models["lightgbm"] = (
            Pipeline(
                [
                    ("impute", SimpleImputer()),
                    (
                        "clf",
                        LGBMClassifier(
                            random_state=42,
                            class_weight="balanced",
                            n_jobs=n_jobs,
                            verbosity=-1,
                        ),
                    ),
                ]
            ),
            {"clf__n_estimators": [100, 300], "clf__num_leaves": [7, 15], "clf__learning_rate": [0.03, 0.1]},
        )
    except Exception:
        pass
    try:
        from xgboost import XGBClassifier  # type: ignore

        models["xgboost"] = (
            Pipeline(
                [
                    ("impute", SimpleImputer()),
                    (
                        "clf",
                        XGBClassifier(
                            random_state=42,
                            eval_metric="logloss",
                            n_jobs=n_jobs,
                            tree_method="hist",
                        ),
                    ),
                ]
            ),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [2, 3], "clf__learning_rate": [0.03, 0.1]},
        )
    except Exception:
        pass
    return models


def score_model(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    return np.asarray(model.predict(x), dtype=float)


def ece_score(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return np.nan
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_score >= low) & (y_score <= high)
        else:
            mask = (y_score >= low) & (y_score < high)
        if mask.any():
            ece += (mask.sum() / n) * abs(float(y_score[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)


def metrics_from_scores(y: np.ndarray, score: np.ndarray) -> Dict[str, Any]:
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


def fold_metric_row(name: str, fold: int, inner_cv_auc: float, y: np.ndarray, score: np.ndarray) -> Dict[str, Any]:
    row = {"classifier": name, "fold": fold, "inner_cv_auc_trainDev": inner_cv_auc}
    row.update(metrics_from_scores(y, score))
    return row


def aggregate_metrics(fold_metrics: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metric_cols = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "brier", "ece"]
    for classifier, grp in fold_metrics.groupby("classifier"):
        pred_grp = preds[preds["classifier"] == classifier]
        y = pred_grp["y_true"].astype(int).to_numpy()
        score = pred_grp["y_score"].astype(float).to_numpy()
        pooled = metrics_from_scores(y, score)
        row: Dict[str, Any] = {
            "classifier": classifier,
            "n_total": int(len(y)),
            "n_CN": int((y == 0).sum()),
            "n_AD": int((y == 1).sum()),
            "status": "ok",
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
    return pd.DataFrame(rows).sort_values(["roc_auc_mean_fold", "balanced_accuracy_mean_fold"], ascending=False)


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    latent_dir = resolve(args.latent_export_dir)
    outdir = prepare_output_dir(args.output_dir, args.overwrite)
    metadata = pd.read_csv(resolve(args.metadata_path))

    ok, report = validate_latent_files(latent_dir)
    if not ok:
        write_missing_report(outdir, report)
        raise RuntimeError(f"Frozen latent sweep blocked; exported latents are missing. See {outdir / 'frozen_latent_classifier_sweep_readme.md'}")

    all_fold_metrics: List[Dict[str, Any]] = []
    all_preds: List[pd.DataFrame] = []
    failures: List[Dict[str, Any]] = []
    models = classifier_grid(args.n_jobs)

    for fold in range(1, 6):
        train = load_fold_split(latent_dir, fold, "trainDev", metadata)
        test = load_fold_split(latent_dir, fold, "test", metadata)
        train_x, test_x, latent_cols = make_features(train, test)
        y_train = train["y_true"].astype(int).to_numpy()
        y_test = test["y_true"].astype(int).to_numpy()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_best: List[Tuple[str, float, Any]] = []

        for name, (pipe, grid) in models.items():
            try:
                search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=args.n_jobs, error_score="raise")
                search.fit(train_x, y_train)
                best = search.best_estimator_
                calibrated = CalibratedClassifierCV(best, method="sigmoid", cv=3)
                calibrated.fit(train_x, y_train)
                score = score_model(calibrated, test_x)
                all_fold_metrics.append(fold_metric_row(name, fold, float(search.best_score_), y_test, score))
                all_preds.append(
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
            all_fold_metrics.append(
                fold_metric_row(
                    ensemble_name,
                    fold,
                    float(np.mean([cv_auc for _name, cv_auc, _model in top])),
                    y_test,
                    ensemble_score,
                )
            )
            all_preds.append(
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

    if not all_fold_metrics:
        raise RuntimeError("No classifier completed successfully.")

    fold_metrics = pd.DataFrame(all_fold_metrics)
    preds = pd.concat(all_preds, ignore_index=True)
    summary = aggregate_metrics(fold_metrics, preds)
    summary.to_csv(outdir / "frozen_latent_classifier_sweep_metrics.csv", index=False)
    fold_metrics.to_csv(outdir / "frozen_latent_classifier_sweep_fold_metrics.csv", index=False)
    preds.to_csv(outdir / "frozen_latent_classifier_sweep_predictions.csv", index=False)

    manifest = {
        "run_dir": str(run_dir),
        "run_dir_realpath": str(run_dir.resolve()),
        "latent_export_dir": str(latent_dir),
        "latent_export_dir_realpath": str(latent_dir.resolve()),
        "output_dir": str(outdir),
        "output_dir_realpath": str(outdir.resolve()),
        "n_latent_features": int(len(latent_cols)),
        "metadata_features": ["Age", "Sex"],
        "manufacturer_site_features_used": False,
        "vae_retrained": False,
        "model_selection_split": "outer-fold trainDev only",
        "failures": failures,
    }
    (outdir / "frozen_latent_classifier_sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    best = summary.iloc[0]
    lines = [
        "# Frozen Latent Classifier Sweep",
        "",
        "This sweep used exported VAE `mu` embeddings from the completed V4 [4,1,0] beta=2.5 run.",
        "",
        "- No VAE training was run.",
        "- No Manufacturer or Site features were used.",
        "- Age and Sex were included as metadata covariates.",
        "- Classifier hyperparameters and calibration were selected inside each outer fold train/dev split only.",
        "",
        "## Best Result",
        "",
        f"- Classifier: `{best['classifier']}`",
        f"- Mean-fold ROC-AUC: {best['roc_auc_mean_fold']:.4f} +/- {best['roc_auc_sd_fold']:.4f}",
        f"- Pooled ROC-AUC: {best['roc_auc_pooled']:.4f}",
        f"- Mean-fold balanced accuracy: {best['balanced_accuracy_mean_fold']:.4f}",
        f"- Mean-fold sensitivity AD: {best['sensitivity_mean_fold']:.4f}",
        f"- Mean-fold specificity CN: {best['specificity_mean_fold']:.4f}",
        "",
        "## Notes",
        "",
        "Mean-fold metrics are the primary comparison target for this nested-CV-style audit. Pooled metrics are reported as secondary summaries.",
    ]
    if failures:
        lines.extend(["", "## Classifier Failures", ""])
        for failure in failures:
            lines.append(f"- fold {failure['fold']} `{failure['classifier']}`: {failure['error']}")
    (outdir / "frozen_latent_classifier_sweep_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
