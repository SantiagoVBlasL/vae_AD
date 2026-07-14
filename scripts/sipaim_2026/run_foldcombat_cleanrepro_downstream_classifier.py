#!/usr/bin/env python3
"""Freeze the leakage-safe downstream diagnostic classifier for a clean run.

This consumes only posterior means emitted in-process by the foldwise-ComBat
trainer. It does not train a VAE, fit ComBat, or use outer-test labels for
model selection, score normalization, or threshold selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from run_recover035_latent384_beta3p75_stageB_oof_score_calibration import (
    INNER_FOLDS,
    ORIGINAL_C_GRID,
    PRIMARY_THRESHOLD,
    SEED,
    calibrate,
    classifier_specs,
    inner_stratification_key,
    make_preprocessor,
    score_1d,
    select_thresholds,
)


FOLDS = (1, 2, 3, 4, 5)
LATENT_DIM = 384
MU_COLUMNS = [f"mu_{index}" for index in range(LATENT_DIM)]
MODEL_NAME = "logreg_l2_original"
FEATURE_SET = "posterior_mu_plus_age_sex"
SCORE_NORMALIZATION = "oof_ecdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_subjects(values: pd.Series) -> str:
    return hashlib.sha256(
        "\n".join(values.astype(str).tolist()).encode("utf-8")
    ).hexdigest()


def load_cache(cache_dir: Path, fold: int, split: str) -> pd.DataFrame:
    path = cache_dir / f"fold_{fold}_{split}_latent_mu.csv"
    lineage_path = cache_dir / f"fold_{fold}_{split}_latent_mu_lineage.json"
    if not path.is_file() or not lineage_path.is_file():
        raise FileNotFoundError(f"Missing cache or lineage file: {path}")
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    required_lineage = {
        "input_lineage": "foldwise_combat_harmonized_then_fold_normalized",
        "posterior_statistic": "mu",
        "generated_in_training_process": True,
        "raw_input_cache_reused": False,
        "combat_fit_scope": "outer_train_dev_only",
    }
    for key, expected in required_lineage.items():
        if lineage.get(key) != expected:
            raise RuntimeError(
                f"Fold {fold} {split}: invalid lineage {key}="
                f"{lineage.get(key)!r}; expected {expected!r}"
            )
    if lineage.get("cache_sha256") != sha256_file(path):
        raise RuntimeError(f"Fold {fold} {split}: cache SHA256 mismatch")
    frame = pd.read_csv(path)
    required_columns = {
        "SubjectID",
        "tensor_idx",
        "ResearchGroup_Mapped",
        "Manufacturer",
        "Age",
        "Sex",
        "y",
        *MU_COLUMNS,
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing[:20]}")
    if len(frame) != frame["SubjectID"].nunique():
        raise ValueError(f"Fold {fold} {split}: duplicate SubjectID values")
    mu = frame[MU_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(mu).all():
        raise ValueError(f"Fold {fold} {split}: non-finite latent values")
    frame["y"] = pd.to_numeric(frame["y"], errors="raise").astype(int)
    frame["Age"] = pd.to_numeric(frame["Age"], errors="coerce")
    frame["Sex"] = frame["Sex"].astype(str)
    return frame


def binary_metrics(
    y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else np.nan
    specificity = float(tn / (tn + fp)) if tn + fp else np.nan
    return {
        "n": int(len(y_true)),
        "n_cn": int((y_true == 0).sum()),
        "n_ad": int((y_true == 1).sum()),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": float(2 * tp / (2 * tp + fp + fn))
        if 2 * tp + fp + fn
        else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run(run_dir: Path, n_jobs: int, dry_run: bool) -> None:
    run_dir = run_dir.resolve()
    output_dir = run_dir / "downstream_diagnostic_classifier"
    cache_dir = output_dir / "latent_cache"
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    cached: dict[tuple[int, str], pd.DataFrame] = {}
    for fold in FOLDS:
        for split in ("trainDev", "test"):
            cached[(fold, split)] = load_cache(cache_dir, fold, split)
        overlap = set(cached[(fold, "trainDev")]["SubjectID"]) & set(
            cached[(fold, "test")]["SubjectID"]
        )
        if overlap:
            raise RuntimeError(
                f"Fold {fold}: train/dev and outer-test overlap: {sorted(overlap)[:5]}"
            )

    result_paths = [
        output_dir / "frozen_classifiers",
        output_dir / "downstream_classifier_oof_predictions.csv",
        output_dir / "downstream_classifier_foldwise_metrics.csv",
    ]
    existing = [str(path) for path in result_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite downstream outputs:\n" + "\n".join(existing)
        )
    if dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_OK",
                    "run_dir": str(run_dir),
                    "cache_dir": str(cache_dir),
                    "folds": list(FOLDS),
                    "features": "posterior latent means + Age + Sex",
                    "classifier": "L2 logistic regression",
                    "c_grid": ORIGINAL_C_GRID,
                    "inner_folds": INNER_FOLDS,
                    "score_normalization": SCORE_NORMALIZATION,
                    "threshold_selection": PRIMARY_THRESHOLD,
                    "vae_training": False,
                    "combat_fitting": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    model_dir = output_dir / "frozen_classifiers"
    model_dir.mkdir(parents=False, exist_ok=False)
    fold_metrics: list[dict[str, Any]] = []
    outer_predictions: list[pd.DataFrame] = []
    inner_predictions: list[pd.DataFrame] = []
    model_manifest: list[dict[str, Any]] = []

    for fold in FOLDS:
        train = cached[(fold, "trainDev")]
        test = cached[(fold, "test")]
        inner_key, inner_context = inner_stratification_key(train, INNER_FOLDS)
        inner_random_state = SEED + fold + 30
        inner_cv = list(
            StratifiedKFold(
                n_splits=INNER_FOLDS,
                shuffle=True,
                random_state=inner_random_state,
            ).split(np.zeros(len(train)), inner_key)
        )
        feature_columns = MU_COLUMNS + ["Age", "Sex"]
        x_train = train[feature_columns].copy()
        x_test = test[feature_columns].copy()
        y_train = train["y"].to_numpy(dtype=int)
        y_test = test["y"].to_numpy(dtype=int)

        base_pipeline, parameter_grid = classifier_specs(fold)[MODEL_NAME]
        pipeline = clone(base_pipeline)
        pipeline.steps[0] = (
            "pre",
            make_preprocessor(MU_COLUMNS, include_age_sex=True),
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
        )
        search.fit(x_train, y_train)
        frozen_estimator = search.best_estimator_
        inner_raw = cross_val_predict(
            clone(frozen_estimator),
            x_train,
            y_train,
            cv=inner_cv,
            method="predict_proba",
            n_jobs=n_jobs,
        )[:, 1].astype(float)
        outer_raw = score_1d(frozen_estimator, x_test)
        inner_score, outer_score, normalization_metadata = calibrate(
            inner_raw,
            outer_raw,
            y_train,
            SCORE_NORMALIZATION,
        )
        threshold_candidates = select_thresholds(y_train, inner_score)
        selected = [
            item
            for item in threshold_candidates
            if item["threshold_strategy"] == PRIMARY_THRESHOLD
        ]
        if len(selected) != 1:
            raise RuntimeError(f"Fold {fold}: primary threshold selection failed")
        threshold_metadata = selected[0]
        threshold = float(threshold_metadata["threshold"])
        outer_pred = (outer_score >= threshold).astype(int)

        metrics = binary_metrics(y_test, outer_score, outer_pred)
        fold_metrics.append(
            {
                "fold": fold,
                "model_name": MODEL_NAME,
                "feature_set": FEATURE_SET,
                "score_normalization": SCORE_NORMALIZATION,
                "threshold_strategy": PRIMARY_THRESHOLD,
                "threshold": threshold,
                "best_inner_auc": float(search.best_score_),
                "best_parameters": json.dumps(
                    search.best_params_, sort_keys=True
                ),
                "inner_cv_context": inner_context,
                "inner_cv_random_state": inner_random_state,
                "classifier_random_state": SEED + fold,
                "outer_test_used_for_fit": False,
                "outer_test_used_for_normalization": False,
                "outer_test_used_for_threshold": False,
                **metrics,
            }
        )

        outer_frame = test[
            [
                "SubjectID",
                "tensor_idx",
                "ResearchGroup_Mapped",
                "Manufacturer",
                "Age",
                "Sex",
            ]
        ].copy()
        outer_frame["fold"] = fold
        outer_frame["y_true"] = y_test
        outer_frame["y_score_raw"] = outer_raw
        outer_frame["y_score"] = outer_score
        outer_frame["threshold"] = threshold
        outer_frame["y_pred"] = outer_pred
        outer_predictions.append(outer_frame)

        inner_frame = train[["SubjectID", "tensor_idx"]].copy()
        inner_frame["fold"] = fold
        inner_frame["y_true"] = y_train
        inner_frame["inner_oof_score_raw"] = inner_raw
        inner_frame["inner_oof_score"] = inner_score
        inner_frame["threshold"] = threshold
        inner_frame["inner_oof_pred"] = (inner_score >= threshold).astype(int)
        inner_predictions.append(inner_frame)

        sorted_inner_raw = np.sort(inner_raw)
        frozen_payload = {
            "fold": fold,
            "estimator": frozen_estimator,
            "feature_columns": feature_columns,
            "latent_columns": MU_COLUMNS,
            "classifier": "L2_regularized_logistic_regression",
            "classifier_random_state": SEED + fold,
            "best_parameters": search.best_params_,
            "best_inner_auc": float(search.best_score_),
            "inner_cv_random_state": inner_random_state,
            "inner_cv_context": inner_context,
            "score_normalization": SCORE_NORMALIZATION,
            "inner_oof_raw_scores_sorted": sorted_inner_raw,
            "inner_oof_ecdf_percentiles": (
                np.arange(1, len(sorted_inner_raw) + 1) - 0.5
            )
            / len(sorted_inner_raw),
            "normalization_metadata": normalization_metadata,
            "threshold": threshold,
            "threshold_strategy": PRIMARY_THRESHOLD,
            "threshold_metadata": threshold_metadata,
            "train_dev_subject_hash": sha256_subjects(train["SubjectID"]),
            "outer_test_subject_hash": sha256_subjects(test["SubjectID"]),
            "source_train_dev_cache": str(
                cache_dir / f"fold_{fold}_trainDev_latent_mu.csv"
            ),
            "source_outer_test_cache": str(
                cache_dir / f"fold_{fold}_test_latent_mu.csv"
            ),
            "source_lineage": "foldwise_combat_harmonized_then_fold_normalized",
            "raw_input_cache_reused": False,
            "outer_test_used_for_fit": False,
            "outer_test_used_for_normalization": False,
            "outer_test_used_for_threshold": False,
        }
        model_path = (
            model_dir / f"fold_{fold}_downstream_diagnostic_classifier.joblib"
        )
        joblib.dump(frozen_payload, model_path)
        model_manifest.append(
            {
                "fold": fold,
                "path": str(model_path),
                "sha256": sha256_file(model_path),
                "best_parameters": json.dumps(
                    search.best_params_, sort_keys=True
                ),
                "threshold": threshold,
                "train_dev_subject_hash": frozen_payload[
                    "train_dev_subject_hash"
                ],
                "outer_test_subject_hash": frozen_payload[
                    "outer_test_subject_hash"
                ],
            }
        )

    outer = pd.concat(outer_predictions, ignore_index=True)
    if len(outer) != 397 or outer["SubjectID"].nunique() != 397:
        raise RuntimeError(
            f"Expected 397 unique outer-fold predictions; got "
            f"{len(outer)} rows and {outer['SubjectID'].nunique()} subjects"
        )
    fold_metrics_frame = pd.DataFrame(fold_metrics)
    pooled = pd.DataFrame(
        [
            {
                "scope": "pooled_outer_fold_predictions",
                "model_name": MODEL_NAME,
                "feature_set": FEATURE_SET,
                "score_normalization": SCORE_NORMALIZATION,
                "threshold_strategy": PRIMARY_THRESHOLD,
                **binary_metrics(
                    outer["y_true"].to_numpy(dtype=int),
                    outer["y_score"].to_numpy(dtype=float),
                    outer["y_pred"].to_numpy(dtype=int),
                ),
            }
        ]
    )
    fold_metrics_frame.to_csv(
        output_dir / "downstream_classifier_foldwise_metrics.csv", index=False
    )
    pooled.to_csv(
        output_dir / "downstream_classifier_primary_metrics.csv", index=False
    )
    outer.to_csv(
        output_dir / "downstream_classifier_oof_predictions.csv", index=False
    )
    pd.concat(inner_predictions, ignore_index=True).to_csv(
        output_dir / "downstream_classifier_inner_oof_train_dev_predictions.csv",
        index=False,
    )
    pd.DataFrame(model_manifest).to_csv(
        output_dir / "frozen_classifier_manifest.csv", index=False
    )
    (output_dir / "downstream_classifier_command_log.json").write_text(
        json.dumps(
            {
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "script": str(Path(__file__).resolve()),
                "run_dir": str(run_dir),
                "status": "COMPLETE",
                "features": "posterior latent means + Age + Sex",
                "classifier": "L2-regularized logistic regression",
                "c_grid": ORIGINAL_C_GRID,
                "inner_folds": INNER_FOLDS,
                "base_seed": SEED,
                "score_normalization": SCORE_NORMALIZATION,
                "threshold_strategy": PRIMARY_THRESHOLD,
                "outer_test_predictions": int(len(outer)),
                "vae_training": False,
                "combat_fitting": False,
                "outer_test_fitting": False,
                "raw_input_cache_reused": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "output_dir": str(output_dir),
                "n_outer_predictions": int(len(outer)),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    args = parse_args()
    run(args.run_dir, int(args.n_jobs), bool(args.dry_run))


if __name__ == "__main__":
    main()
