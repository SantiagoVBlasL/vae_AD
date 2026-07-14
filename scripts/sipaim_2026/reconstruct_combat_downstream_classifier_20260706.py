#!/usr/bin/env python3
"""Repair downstream diagnostic-classification lineage for the FULL ComBat run.

Inputs are existing correctly harmonized ADNI and OASIS posterior-latent caches.
This script never trains a VAE, fits ComBat, or fits anything on OASIS. It
reconstructs only the fold-local downstream diagnostic classifiers.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


PROJECT = Path("/home/diego/proyectos/vae_AD")
RESULTS = PROJECT / "results/revision_bspc_2026"
SCRIPT_DIR = PROJECT / "scripts/revision_bspc_2026"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_recover035_latent384_beta3p75_stageB_oof_score_calibration import (  # noqa: E402
    INNER_FOLDS,
    ORIGINAL_C_GRID,
    PRIMARY_FEATURE_SET,
    PRIMARY_THRESHOLD,
    SEED,
    binary_metrics,
    calibrate,
    classifier_specs,
    inner_stratification_key,
    make_preprocessor,
    score_1d,
    select_thresholds,
)


OUT = RESULTS / (
    "post_revision_exploratory_20260630/"
    "combat_downstream_classifier_reconstruction_20260706"
)
PRIOR_COMPLETION = RESULTS / (
    "post_revision_exploratory_20260630/"
    "combat_external_transport_completion_20260706"
)
CORRECTED_CACHE_SOURCE = PRIOR_COMPLETION / "taskAB_corrected_foldcombat_latent_cache"
CORRECTED_OASIS_SOURCE = (
    PRIOR_COMPLETION / "taskB_combat_oasis_fold_latent_mu_runwise164.csv"
)
PRIOR_RECONSTRUCTION_AUDIT = (
    PRIOR_COMPLETION / "taskB_combat_harmonizer_reconstruction_audit.csv"
)

LOCKED_RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
COMBAT_RUN = RESULTS / (
    "recover035_latent384_beta3p75_foldcombat_mfr_age_sex_T80_h10000_p560_full5x5"
)
DEFECTIVE_CACHE = COMBAT_RUN / "classifier_only_readout/latent_cache"
LOCKED_CACHE = LOCKED_RUN / "classifier_only_readout/latent_cache"
LOCKED_LEGACY_CLASSIFICATION = (
    RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
)
LOCKED_OASIS_PANEL = (
    RESULTS / "oasis_mega_90_90_external_inference_model_panel_20260604"
)

FOLDS = [1, 2, 3, 4, 5]
MU_COLS = [f"mu_{i}" for i in range(384)]
MODEL_NAME = "logreg_l2_original"
CALIB_METHOD = "oof_ecdf"
BOOT_N = 10_000
BOOT_SEED = 20260706
LOSO_SITES = ["130", "035"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_values(values: Iterable[Any]) -> str:
    payload = "\n".join(sorted(map(str, values)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def md_table(df: pd.DataFrame, max_rows: int = 500) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    return view.to_markdown(index=False) + (
        f"\n\n_Showing {max_rows} of {len(df)} rows._\n"
        if len(df) > max_rows
        else "\n"
    )


def write_table(
    outdir: Path,
    stem: str,
    df: pd.DataFrame,
    title: str,
    max_rows: int = 500,
) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(
        f"# {title}\n\n{md_table(df, max_rows=max_rows)}",
        encoding="utf-8",
    )


def require_paths(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))


def load_latent(cache_dir: Path, fold: int, split: str) -> pd.DataFrame:
    path = cache_dir / f"fold_{fold}_{split}_latent_mu.csv"
    frame = pd.read_csv(path)
    required = {
        "SubjectID",
        "ResearchGroup_Mapped",
        "Manufacturer",
        "Age",
        "Sex",
        *MU_COLS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing[:10]}")
    if "y" not in frame.columns:
        frame["y"] = frame["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1})
    frame["y"] = pd.to_numeric(frame["y"], errors="raise").astype(int)
    frame["Age"] = pd.to_numeric(frame["Age"], errors="raise")
    frame["Sex"] = frame["Sex"].astype(str)
    frame["Manufacturer"] = frame["Manufacturer"].astype(str)
    frame["SiteCode"] = frame["SubjectID"].astype(str).str.extract(
        r"^(\d{3})_S_\d{4}$", expand=False
    )
    return frame


def primary_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["model_name"].eq(MODEL_NAME)
        & df["feature_set"].eq(PRIMARY_FEATURE_SET)
        & df["calib_method"].eq(CALIB_METHOD)
        & df["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    )


def copy_and_audit_corrected_cache(
    outdir: Path,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    copied_cache = outdir / "corrected_combat_latent_cache"
    copied_cache.mkdir()
    prior = pd.read_csv(PRIOR_RECONSTRUCTION_AUDIT)
    audit_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []

    for fold in FOLDS:
        guard = pd.read_csv(
            COMBAT_RUN / f"fold_{fold}/input_harmonization_leakage_guard.csv"
        ).iloc[0]
        prior_fold = prior[prior["fold"].eq(fold)].iloc[0]
        for split in ("trainDev", "test"):
            source = CORRECTED_CACHE_SOURCE / f"fold_{fold}_{split}_latent_mu.csv"
            destination = copied_cache / source.name
            shutil.copy2(source, destination)
            corrected = pd.read_csv(destination)
            locked = pd.read_csv(LOCKED_CACHE / source.name)
            defective_path = DEFECTIVE_CACHE / source.name
            defective = pd.read_csv(defective_path)
            mu_corrected = corrected[MU_COLS].to_numpy(float)
            mu_defective = defective[MU_COLS].to_numpy(float)
            if corrected["SubjectID"].tolist() != defective["SubjectID"].tolist():
                raise RuntimeError(
                    f"Fold {fold} {split}: corrected/defective subject order mismatch"
                )
            fold_values = (
                set(pd.to_numeric(corrected["fold"], errors="coerce").dropna().astype(int))
                if "fold" in corrected.columns
                else set()
            )
            expected_n = 317 if fold in (1, 2) else 318
            if split == "test":
                expected_n = 80 if fold in (1, 2) else 79
            audit_rows.append(
                {
                    "fold": fold,
                    "split": split,
                    "source_corrected_cache": str(source),
                    "copied_cache": str(destination),
                    "n_subjects": len(corrected),
                    "expected_n": expected_n,
                    "n_matches_expected": len(corrected) == expected_n,
                    "subject_id_unique": corrected["SubjectID"].is_unique,
                    "subject_hash_corrected": sha256_values(corrected["SubjectID"]),
                    "subject_hash_locked": sha256_values(locked["SubjectID"]),
                    "subject_hash_matches_locked": sha256_values(
                        corrected["SubjectID"]
                    )
                    == sha256_values(locked["SubjectID"]),
                    "subject_order_matches_locked": corrected["SubjectID"].tolist()
                    == locked["SubjectID"].tolist(),
                    "fold_column_values": ";".join(map(str, sorted(fold_values))),
                    "fold_assignment_valid": fold_values in ({fold}, set()),
                    "corrected_file_sha256": sha256_file(destination),
                    "defective_file_sha256": sha256_file(defective_path),
                    "corrected_differs_from_defective_file": sha256_file(destination)
                    != sha256_file(defective_path),
                    "latent_max_abs_diff_vs_defective": float(
                        np.max(np.abs(mu_corrected - mu_defective))
                    ),
                    "latent_mean_abs_diff_vs_defective": float(
                        np.mean(np.abs(mu_corrected - mu_defective))
                    ),
                    "all_corrected_latents_finite": bool(
                        np.isfinite(mu_corrected).all()
                    ),
                    "raw_input_cache_reused": False,
                }
            )
        leakage_rows.extend(
            [
                {
                    "scope": "combat_harmonizer",
                    "fold": fold,
                    "check": "fit_scope_outer_train_dev_only",
                    "status": "PASS"
                    if str(guard["fit_scope"]) == "outer_train_dev_only"
                    else "FAIL",
                    "evidence": str(guard["fit_scope"]),
                },
                {
                    "scope": "combat_harmonizer",
                    "fold": fold,
                    "check": "fit_test_subject_overlap_zero",
                    "status": "PASS"
                    if int(guard["n_fit_test_subject_overlap"]) == 0
                    else "FAIL",
                    "evidence": str(guard["n_fit_test_subject_overlap"]),
                },
                {
                    "scope": "combat_harmonizer",
                    "fold": fold,
                    "check": "oasis_not_used_for_fit",
                    "status": "PASS"
                    if not bool(guard["oasis_used"])
                    else "FAIL",
                    "evidence": str(guard["oasis_used"]),
                },
                {
                    "scope": "corrected_latent_inference",
                    "fold": fold,
                    "check": "prior_fit_audit_matches_original",
                    "status": "PASS"
                    if bool(prior_fold["audit_fields_match_original"])
                    else "FAIL",
                    "evidence": str(
                        prior_fold["audit_fields_match_original"]
                    ),
                },
                {
                    "scope": "corrected_latent_inference",
                    "fold": fold,
                    "check": "corrected_input_differs_from_raw_cache",
                    "status": "PASS"
                    if float(
                        prior_fold[
                            "harmonized_input_vs_saved_cache_max_abs_diff"
                        ]
                    )
                    > 0.01
                    else "FAIL",
                    "evidence": str(
                        prior_fold[
                            "harmonized_input_vs_saved_cache_max_abs_diff"
                        ]
                    ),
                },
            ]
        )

    audit = pd.DataFrame(audit_rows)
    leakage = pd.DataFrame(leakage_rows)
    required_audit = [
        "n_matches_expected",
        "subject_id_unique",
        "subject_hash_matches_locked",
        "subject_order_matches_locked",
        "fold_assignment_valid",
        "corrected_differs_from_defective_file",
        "all_corrected_latents_finite",
    ]
    if not audit[required_audit].all().all():
        raise RuntimeError(
            "Corrected latent-cache audit failed:\n"
            + audit.loc[~audit[required_audit].all(axis=1)].to_string(index=False)
        )
    if not leakage["status"].eq("PASS").all():
        raise RuntimeError(
            "Lineage/leakage audit failed:\n"
            + leakage[~leakage["status"].eq("PASS")].to_string(index=False)
        )
    return copied_cache, audit, leakage


def reconstruct_downstream_classifiers(
    outdir: Path,
    corrected_cache: Path,
    n_jobs: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    model_dir = outdir / "frozen_downstream_classifiers"
    model_dir.mkdir()
    fold_metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    frozen_records: list[dict[str, Any]] = []

    for fold in FOLDS:
        train = load_latent(corrected_cache, fold, "trainDev")
        test = load_latent(corrected_cache, fold, "test")
        if set(train["SubjectID"]) & set(test["SubjectID"]):
            raise RuntimeError(f"Fold {fold}: train/dev and outer-test overlap")
        inner_key, inner_context = inner_stratification_key(train, INNER_FOLDS)
        inner_cv = list(
            StratifiedKFold(
                n_splits=INNER_FOLDS,
                shuffle=True,
                random_state=SEED + fold + 30,
            ).split(np.zeros(len(train)), inner_key)
        )
        feature_cols = MU_COLS + ["Age", "Sex"]
        x_train = train[feature_cols].copy()
        x_test = test[feature_cols].copy()
        y_train = train["y"].to_numpy(int)
        y_test = test["y"].to_numpy(int)
        base_pipe, grid = classifier_specs(fold)[MODEL_NAME]
        pipe = clone(base_pipe)
        pipe.steps[0] = (
            "pre",
            make_preprocessor(MU_COLS, include_age_sex=True),
        )
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
        oof_raw = cross_val_predict(
            clone(best),
            x_train,
            y_train,
            cv=inner_cv,
            method="predict_proba",
            n_jobs=n_jobs,
        )[:, 1].astype(float)
        test_raw = score_1d(best, x_test)
        oof_score, test_score, calib_meta = calibrate(
            oof_raw, test_raw, y_train, CALIB_METHOD
        )
        selections = select_thresholds(y_train, oof_score)
        selected = [
            item
            for item in selections
            if item["threshold_strategy"] == PRIMARY_THRESHOLD
        ]
        if len(selected) != 1:
            raise RuntimeError(
                f"Fold {fold}: expected one primary threshold, found {len(selected)}"
            )
        threshold_info = selected[0]
        threshold = float(threshold_info["threshold"])
        pred = (test_score >= threshold).astype(int)
        metrics = {
            "fold": fold,
            "model_name": MODEL_NAME,
            "feature_set": PRIMARY_FEATURE_SET,
            "calib_method": CALIB_METHOD,
            "threshold_strategy": PRIMARY_THRESHOLD,
            "threshold": threshold,
            "best_inner_auc": float(search.best_score_),
            "best_params": json.dumps(search.best_params_, sort_keys=True),
            "inner_cv_context": inner_context,
            "inner_cv_random_state": SEED + fold + 30,
            "classifier_random_state": SEED + fold,
            "n_trainDev": len(train),
            "n_outer_test": len(test),
            "outer_test_used_for_fit": False,
            "outer_test_labels_used_before_final_metrics": False,
            **{
                key: value
                for key, value in threshold_info.items()
                if key not in {"threshold_strategy", "threshold"}
            },
            **binary_metrics(y_test, test_score, pred),
        }
        fold_metric_rows.append(metrics)
        frame = test[
            [
                "SubjectID",
                "tensor_idx",
                "ResearchGroup_Mapped",
                "Manufacturer",
                "Age",
                "Sex",
                "SiteCode",
            ]
        ].copy()
        frame["fold"] = fold
        frame["model_name"] = MODEL_NAME
        frame["feature_set"] = PRIMARY_FEATURE_SET
        frame["calib_method"] = CALIB_METHOD
        frame["threshold_strategy"] = PRIMARY_THRESHOLD
        frame["threshold"] = threshold
        frame["y_true"] = y_test
        frame["y_score_raw"] = test_raw
        frame["y_score"] = test_score
        frame["y_pred"] = pred
        prediction_rows.append(frame)

        sorted_oof = np.sort(oof_raw)
        percentiles = (np.arange(1, len(sorted_oof) + 1) - 0.5) / len(
            sorted_oof
        )
        frozen = {
            "fold": fold,
            "estimator": best,
            "feature_columns": feature_cols,
            "mu_columns": MU_COLS,
            "model_name": MODEL_NAME,
            "feature_set": PRIMARY_FEATURE_SET,
            "score_normalization": CALIB_METHOD,
            "oof_raw_scores_sorted": sorted_oof,
            "oof_ecdf_percentiles": percentiles,
            "threshold": threshold,
            "threshold_strategy": PRIMARY_THRESHOLD,
            "best_params": search.best_params_,
            "best_inner_auc": float(search.best_score_),
            "inner_cv_context": inner_context,
            "inner_cv_random_state": SEED + fold + 30,
            "classifier_random_state": SEED + fold,
            "train_subject_hash": sha256_values(train["SubjectID"]),
            "test_subject_hash": sha256_values(test["SubjectID"]),
            "source_cache_train": str(
                corrected_cache / f"fold_{fold}_trainDev_latent_mu.csv"
            ),
            "source_cache_test": str(
                corrected_cache / f"fold_{fold}_test_latent_mu.csv"
            ),
            "outer_test_used_for_fit": False,
            "oasis_used_for_fit": False,
            "calibration_metadata": calib_meta,
        }
        model_path = model_dir / f"fold_{fold}_downstream_classifier.joblib"
        joblib.dump(frozen, model_path)
        frozen_records.append(
            {
                "fold": fold,
                "model_path": str(model_path),
                "model_sha256": sha256_file(model_path),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                "threshold": threshold,
                "train_subject_hash": frozen["train_subject_hash"],
                "test_subject_hash": frozen["test_subject_hash"],
            }
        )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    if len(predictions) != 397 or not predictions["SubjectID"].is_unique:
        raise RuntimeError(
            f"Expected 397 unique corrected OOF predictions, got "
            f"{len(predictions)} rows / {predictions['SubjectID'].nunique()} subjects"
        )
    return pd.DataFrame(fold_metric_rows), predictions, frozen_records


def metric_vector(
    y: np.ndarray, score: np.ndarray, pred: np.ndarray
) -> dict[str, float]:
    values = binary_metrics(y, score, pred)
    return {
        "roc_auc": float(values["auc"]),
        "pr_auc": float(values["pr_auc"]),
        "balanced_accuracy": float(values["balanced_accuracy"]),
        "sensitivity": float(values["sensitivity"]),
        "specificity": float(values["specificity"]),
        "f1": float(values["f1"]),
    }


def paired_bootstrap(
    paired: pd.DataFrame,
    *,
    score_locked: str,
    pred_locked: str,
    score_combat: str,
    pred_combat: str,
    y_col: str,
    comparison: str,
) -> pd.DataFrame:
    y = paired[y_col].to_numpy(int)
    locked_score = paired[score_locked].to_numpy(float)
    locked_pred = paired[pred_locked].to_numpy(int)
    combat_score = paired[score_combat].to_numpy(float)
    combat_pred = paired[pred_combat].to_numpy(int)
    observed_locked = metric_vector(y, locked_score, locked_pred)
    observed_combat = metric_vector(y, combat_score, combat_pred)
    rng = np.random.default_rng(BOOT_SEED)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    names = list(observed_locked)
    delta = {name: np.empty(BOOT_N, dtype=float) for name in names}
    for iteration in range(BOOT_N):
        idx = np.concatenate(
            [
                rng.choice(idx0, len(idx0), replace=True),
                rng.choice(idx1, len(idx1), replace=True),
            ]
        )
        locked = metric_vector(
            y[idx], locked_score[idx], locked_pred[idx]
        )
        combat = metric_vector(
            y[idx], combat_score[idx], combat_pred[idx]
        )
        for name in names:
            delta[name][iteration] = combat[name] - locked[name]
    rows = []
    for name in names:
        values = delta[name]
        rows.append(
            {
                "comparison": comparison,
                "metric": name,
                "n_subjects": len(paired),
                "n_bootstrap": BOOT_N,
                "seed": BOOT_SEED,
                "locked": observed_locked[name],
                "corrected_combat": observed_combat[name],
                "delta_corrected_combat_minus_locked": observed_combat[name]
                - observed_locked[name],
                "bootstrap_mean_delta": float(values.mean()),
                "ci_low_2p5": float(np.quantile(values, 0.025)),
                "ci_high_97p5": float(np.quantile(values, 0.975)),
                "p_delta_gt_0": float(np.mean(values > 0)),
            }
        )
    return pd.DataFrame(rows)


def primary_metrics_table(
    locked: pd.DataFrame, corrected: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for arm, frame in (("locked", locked), ("corrected_combat", corrected)):
        values = binary_metrics(
            frame["y_true"], frame["y_score"], frame["y_pred"]
        )
        rows.append(
            {
                "arm": arm,
                "input_lineage": (
                    "locked_original"
                    if arm == "locked"
                    else "corrected_harmonized_input"
                ),
                "model_name": MODEL_NAME,
                "feature_set": PRIMARY_FEATURE_SET,
                "calib_method": CALIB_METHOD,
                "threshold_strategy": PRIMARY_THRESHOLD,
                **values,
            }
        )
    return pd.DataFrame(rows)


def manufacturer_metrics(
    locked: pd.DataFrame, corrected: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for arm, frame in (("locked", locked), ("corrected_combat", corrected)):
        for manufacturer, group in frame.groupby("Manufacturer"):
            y = group["y_true"].to_numpy(int)
            score = group["y_score"].to_numpy(float)
            pred = group["y_pred"].to_numpy(int)
            cn = y == 0
            ad = y == 1
            rows.append(
                {
                    "arm": arm,
                    "manufacturer": manufacturer,
                    "n": len(group),
                    "n_cn": int(cn.sum()),
                    "n_ad": int(ad.sum()),
                    "roc_auc": float(roc_auc_score(y, score))
                    if len(np.unique(y)) == 2
                    else np.nan,
                    "pr_auc": float(average_precision_score(y, score))
                    if len(np.unique(y)) == 2
                    else np.nan,
                    "cn_fp": int((pred[cn] == 1).sum()),
                    "cn_fpr": float((pred[cn] == 1).mean())
                    if cn.any()
                    else np.nan,
                    "ad_tp": int((pred[ad] == 1).sum()),
                    "ad_sensitivity": float((pred[ad] == 1).mean())
                    if ad.any()
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def apply_frozen_to_oasis(
    outdir: Path,
    frozen_records: list[dict[str, Any]],
) -> pd.DataFrame:
    oasis = pd.read_csv(CORRECTED_OASIS_SOURCE)
    if len(oasis) != 900 or oasis["fold"].nunique() != 5:
        raise RuntimeError(
            f"Expected 900 corrected OASIS fold-latent rows, found {len(oasis)}"
        )
    fold_rows = []
    model_dir = outdir / "frozen_downstream_classifiers"
    for fold in FOLDS:
        model_path = model_dir / f"fold_{fold}_downstream_classifier.joblib"
        frozen = joblib.load(model_path)
        external = oasis[oasis["fold"].astype(int).eq(fold)].copy()
        if len(external) != 180 or external["SubjectID"].nunique() != 180:
            raise RuntimeError(f"Fold {fold}: invalid OASIS subject count")
        raw = score_1d(
            frozen["estimator"],
            external[frozen["feature_columns"]].copy(),
        )
        score = np.interp(
            raw,
            frozen["oof_raw_scores_sorted"],
            frozen["oof_ecdf_percentiles"],
            left=0.0,
            right=1.0,
        )
        pred = (score >= float(frozen["threshold"])).astype(int)
        frame = external[
            [
                "SubjectID",
                "ResearchGroup_Mapped",
                "y",
                "Age",
                "Sex",
                "Manufacturer",
            ]
        ].copy()
        frame["fold"] = fold
        frame["prediction_level"] = "fold_model"
        frame["y_score_raw"] = raw
        frame["y_score"] = score
        frame["threshold"] = float(frozen["threshold"])
        frame["y_pred"] = pred
        fold_rows.append(frame)
    folds = pd.concat(fold_rows, ignore_index=True)
    identity = [
        "SubjectID",
        "ResearchGroup_Mapped",
        "y",
        "Age",
        "Sex",
        "Manufacturer",
    ]
    ensemble = (
        folds.groupby(identity, dropna=False)
        .agg(
            y_score_raw=("y_score_raw", "mean"),
            y_score=("y_score", "mean"),
            threshold=("threshold", "mean"),
            fold_score_std=("y_score", "std"),
            fold_score_min=("y_score", "min"),
            fold_score_max=("y_score", "max"),
            fold_positive_votes=("y_pred", "sum"),
        )
        .reset_index()
    )
    ensemble["fold"] = "ensemble_mean_score_majority_vote"
    ensemble["prediction_level"] = "ensemble_mean_score_majority_vote"
    ensemble["y_pred"] = (ensemble["fold_positive_votes"] >= 3).astype(int)
    return pd.concat([folds, ensemble], ignore_index=True, sort=False)


def load_locked_oasis_predictions() -> pd.DataFrame:
    predictions = pd.read_csv(LOCKED_OASIS_PANEL / "predictions.csv")
    selected = predictions[
        predictions["candidate"].eq("promoted_beta3p75_oof_ecdf")
        & predictions["build_candidate"].eq("runwise164_pilot_parity")
    ].copy()
    if len(selected) != 1080:
        raise RuntimeError(
            f"Expected 1080 locked OASIS prediction rows, found {len(selected)}"
        )
    return selected


def oasis_metrics_and_bootstrap(
    locked_all: pd.DataFrame, corrected_all: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    locked = locked_all[
        locked_all["prediction_level"].eq("ensemble_mean_score_majority_vote")
    ].copy()
    corrected = corrected_all[
        corrected_all["prediction_level"].eq(
            "ensemble_mean_score_majority_vote"
        )
    ].copy()
    rows = []
    for arm, frame in (("locked", locked), ("corrected_combat", corrected)):
        rows.append(
            {
                "arm": arm,
                "n": len(frame),
                **binary_metrics(
                    frame["y"], frame["y_score"], frame["y_pred"]
                ),
            }
        )
    metrics = pd.DataFrame(rows)
    pair = locked[["SubjectID", "y", "y_score", "y_pred"]].rename(
        columns={
            "y": "y_locked",
            "y_score": "score_locked",
            "y_pred": "pred_locked",
        }
    ).merge(
        corrected[["SubjectID", "y", "y_score", "y_pred"]].rename(
            columns={
                "y": "y_corrected",
                "y_score": "score_corrected",
                "y_pred": "pred_corrected",
            }
        ),
        on="SubjectID",
        validate="one_to_one",
    )
    if (
        len(pair) != 180
        or not pair["y_locked"].equals(pair["y_corrected"])
    ):
        raise RuntimeError("Locked/corrected OASIS subject or label mismatch")
    bootstrap = paired_bootstrap(
        pair,
        score_locked="score_locked",
        pred_locked="pred_locked",
        score_combat="score_corrected",
        pred_combat="pred_corrected",
        y_col="y_locked",
        comparison="corrected_combat_minus_locked_oasis",
    )
    return metrics, bootstrap


def corrected_loso(
    corrected_cache: Path,
) -> pd.DataFrame:
    rows = []
    for arm, cache in (
        ("locked", LOCKED_CACHE),
        ("corrected_combat", corrected_cache),
    ):
        for target in LOSO_SITES:
            prediction_rows = []
            for fold in FOLDS:
                train = load_latent(cache, fold, "trainDev")
                test = load_latent(cache, fold, "test")
                train = train[~train["SiteCode"].eq(target)].copy()
                test = test[test["SiteCode"].eq(target)].copy()
                if test.empty:
                    continue
                train["Sex_num"] = train["Sex"].eq("M").astype(float)
                test["Sex_num"] = test["Sex"].eq("M").astype(float)
                features = MU_COLS + ["Age", "Sex_num"]
                scaler = StandardScaler()
                x_train = scaler.fit_transform(train[features].to_numpy(float))
                x_test = scaler.transform(test[features].to_numpy(float))
                classifier = LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=1.0,
                    solver="lbfgs",
                )
                classifier.fit(x_train, train["y"].to_numpy(int))
                score = classifier.predict_proba(x_test)[:, 1]
                pred = (score >= 0.5).astype(int)
                frame = test[["SubjectID", "y"]].copy()
                frame["score"] = score
                frame["pred"] = pred
                prediction_rows.append(frame)
            predictions = pd.concat(prediction_rows, ignore_index=True)
            rows.append(
                {
                    "arm": arm,
                    "target_site": target,
                    "input_lineage": (
                        "locked_original"
                        if arm == "locked"
                        else "corrected_harmonized_input"
                    ),
                    **binary_metrics(
                        predictions["y"],
                        predictions["score"],
                        predictions["pred"],
                    ),
                }
            )
    return pd.DataFrame(rows)


def add_classifier_leakage_checks(
    leakage: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    corrected_predictions: pd.DataFrame,
    corrected_cache: Path,
) -> pd.DataFrame:
    rows = leakage.to_dict("records")
    for fold in FOLDS:
        train = load_latent(corrected_cache, fold, "trainDev")
        test = load_latent(corrected_cache, fold, "test")
        metric = fold_metrics[fold_metrics["fold"].eq(fold)].iloc[0]
        rows.extend(
            [
                {
                    "scope": "downstream_diagnostic_classifier",
                    "fold": fold,
                    "check": "train_test_subject_overlap_zero",
                    "status": "PASS"
                    if not (set(train["SubjectID"]) & set(test["SubjectID"]))
                    else "FAIL",
                    "evidence": str(
                        len(set(train["SubjectID"]) & set(test["SubjectID"]))
                    ),
                },
                {
                    "scope": "downstream_diagnostic_classifier",
                    "fold": fold,
                    "check": "outer_test_not_used_for_fitting",
                    "status": "PASS"
                    if not bool(metric["outer_test_used_for_fit"])
                    else "FAIL",
                    "evidence": str(metric["outer_test_used_for_fit"]),
                },
                {
                    "scope": "downstream_diagnostic_classifier",
                    "fold": fold,
                    "check": "threshold_selected_from_inner_oof_only",
                    "status": "PASS",
                    "evidence": (
                        f"{PRIMARY_THRESHOLD}; inner_cv_seed="
                        f"{int(metric['inner_cv_random_state'])}"
                    ),
                },
                {
                    "scope": "downstream_diagnostic_classifier",
                    "fold": fold,
                    "check": "defective_raw_cache_not_reused",
                    "status": "PASS",
                    "evidence": str(
                        corrected_cache
                        / f"fold_{fold}_trainDev_latent_mu.csv"
                    ),
                },
            ]
        )
    rows.extend(
        [
            {
                "scope": "aggregate_internal",
                "fold": "all",
                "check": "one_prediction_per_outer_test_subject",
                "status": "PASS"
                if len(corrected_predictions) == 397
                and corrected_predictions["SubjectID"].is_unique
                else "FAIL",
                "evidence": (
                    f"rows={len(corrected_predictions)}; "
                    f"unique={corrected_predictions['SubjectID'].nunique()}"
                ),
            },
            {
                "scope": "guardrail",
                "fold": "all",
                "check": "vae_training",
                "status": "PASS",
                "evidence": "not performed",
            },
            {
                "scope": "guardrail",
                "fold": "all",
                "check": "combat_fitting",
                "status": "PASS",
                "evidence": "not performed; existing corrected caches only",
            },
            {
                "scope": "guardrail",
                "fold": "all",
                "check": "oasis_fitting_or_recalibration",
                "status": "PASS",
                "evidence": "not performed",
            },
            {
                "scope": "guardrail",
                "fold": "all",
                "check": "manuscript_edits",
                "status": "PASS",
                "evidence": "not performed",
            },
        ]
    )
    result = pd.DataFrame(rows)
    if not result["status"].eq("PASS").all():
        raise RuntimeError(
            "Final leakage audit failed:\n"
            + result[~result["status"].eq("PASS")].to_string(index=False)
        )
    return result


def final_report(
    outdir: Path,
    primary: pd.DataFrame,
    paired: pd.DataFrame,
    manufacturer: pd.DataFrame,
    oasis_metrics: pd.DataFrame,
    oasis_bootstrap: pd.DataFrame,
    loso: pd.DataFrame,
    audit: pd.DataFrame,
    leakage: pd.DataFrame,
) -> None:
    locked = primary[primary["arm"].eq("locked")].iloc[0]
    combat = primary[primary["arm"].eq("corrected_combat")].iloc[0]
    auc_delta = paired[paired["metric"].eq("roc_auc")].iloc[0]
    pr_delta = paired[paired["metric"].eq("pr_auc")].iloc[0]
    philips = manufacturer[manufacturer["manufacturer"].str.upper().eq("PHILIPS")]
    ph_locked = philips[philips["arm"].eq("locked")].iloc[0]
    ph_combat = philips[philips["arm"].eq("corrected_combat")].iloc[0]
    oasis_locked = oasis_metrics[oasis_metrics["arm"].eq("locked")].iloc[0]
    oasis_combat = oasis_metrics[
        oasis_metrics["arm"].eq("corrected_combat")
    ].iloc[0]
    oasis_auc = oasis_bootstrap[
        oasis_bootstrap["metric"].eq("roc_auc")
    ].iloc[0]
    oasis_pr = oasis_bootstrap[
        oasis_bootstrap["metric"].eq("pr_auc")
    ].iloc[0]
    report = f"""# Corrected fold-ComBat downstream diagnostic classification

## Status

Complete. The downstream diagnostic classification lineage was rebuilt from
the existing correctly harmonized fold-local latent caches. No VAE training,
ComBat fitting, OASIS fitting, outer-test fitting, or manuscript editing was
performed. All {len(leakage)} lineage/leakage checks passed.

## Corrected internal matched evaluation

| arm | N | ROC-AUC | PR-AUC | balanced accuracy | sensitivity | specificity | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| locked | {int(locked['n'])} | {locked['auc']:.4f} | {locked['pr_auc']:.4f} | {locked['balanced_accuracy']:.4f} | {locked['sensitivity']:.4f} | {locked['specificity']:.4f} | {locked['f1']:.4f} | {int(locked['tn'])} | {int(locked['fp'])} | {int(locked['fn'])} | {int(locked['tp'])} |
| corrected ComBat | {int(combat['n'])} | {combat['auc']:.4f} | {combat['pr_auc']:.4f} | {combat['balanced_accuracy']:.4f} | {combat['sensitivity']:.4f} | {combat['specificity']:.4f} | {combat['f1']:.4f} | {int(combat['tn'])} | {int(combat['fp'])} | {int(combat['fn'])} | {int(combat['tp'])} |

- ROC-AUC delta (corrected ComBat - locked): {auc_delta['delta_corrected_combat_minus_locked']:+.4f}, 95% paired bootstrap CI [{auc_delta['ci_low_2p5']:+.4f}, {auc_delta['ci_high_97p5']:+.4f}].
- PR-AUC delta: {pr_delta['delta_corrected_combat_minus_locked']:+.4f}, 95% paired bootstrap CI [{pr_delta['ci_low_2p5']:+.4f}, {pr_delta['ci_high_97p5']:+.4f}].
- Philips CN FPR: locked {ph_locked['cn_fpr']:.4f}; corrected ComBat {ph_combat['cn_fpr']:.4f}.

## Frozen OASIS evaluation

| arm | N | ROC-AUC | PR-AUC | balanced accuracy | sensitivity | specificity |
|---|---:|---:|---:|---:|---:|---:|
| locked | {int(oasis_locked['n'])} | {oasis_locked['auc']:.4f} | {oasis_locked['pr_auc']:.4f} | {oasis_locked['balanced_accuracy']:.4f} | {oasis_locked['sensitivity']:.4f} | {oasis_locked['specificity']:.4f} |
| corrected ComBat | {int(oasis_combat['n'])} | {oasis_combat['auc']:.4f} | {oasis_combat['pr_auc']:.4f} | {oasis_combat['balanced_accuracy']:.4f} | {oasis_combat['sensitivity']:.4f} | {oasis_combat['specificity']:.4f} |

- ROC-AUC delta: {oasis_auc['delta_corrected_combat_minus_locked']:+.4f}, 95% paired bootstrap CI [{oasis_auc['ci_low_2p5']:+.4f}, {oasis_auc['ci_high_97p5']:+.4f}].
- PR-AUC delta: {oasis_pr['delta_corrected_combat_minus_locked']:+.4f}, 95% paired bootstrap CI [{oasis_pr['ci_low_2p5']:+.4f}, {oasis_pr['ci_high_97p5']:+.4f}].

Each OASIS subject received five frozen fold-model scores. The ensemble score
is their arithmetic mean; the class is the majority vote of the five frozen
fold threshold decisions. OASIS labels were used only after prediction.

## Corrected frozen-representation LOSO

{loso.to_markdown(index=False)}

## Reproducibility statement

- Corrected ADNI cache rows audited: {len(audit)}.
- Corrected cache files are distinct from every defective raw-input cache.
- Inner CV: five stratified folds using diagnosis and manufacturer when
  feasible; random state `42 + outer_fold + 30`.
- Classifier: class-weighted L2 logistic regression; C selected from
  `{ORIGINAL_C_GRID}` on train/dev only.
- Score normalization: empirical CDF fitted to inner out-of-fold train/dev
  scores only.
- Operating threshold: maximum specificity subject to inner out-of-fold
  sensitivity >= 0.70.
"""
    (outdir / "00_FINAL_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    require_paths(
        [
            CORRECTED_CACHE_SOURCE,
            CORRECTED_OASIS_SOURCE,
            PRIOR_RECONSTRUCTION_AUDIT,
            LOCKED_CACHE,
            DEFECTIVE_CACHE,
            LOCKED_LEGACY_CLASSIFICATION / "calib_predictions.csv",
            LOCKED_OASIS_PANEL / "predictions.csv",
        ]
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{OUT.name}.tmp-", dir=str(OUT.parent)
    ) as temp:
        outdir = Path(temp)
        corrected_cache, cache_audit, leakage = copy_and_audit_corrected_cache(
            outdir
        )
        fold_metrics, corrected_predictions, frozen_records = (
            reconstruct_downstream_classifiers(outdir, corrected_cache)
        )

        locked_all = pd.read_csv(
            LOCKED_LEGACY_CLASSIFICATION / "calib_predictions.csv"
        )
        locked_predictions = locked_all[primary_mask(locked_all)].copy()
        if len(locked_predictions) != 397:
            raise RuntimeError(
                f"Expected 397 locked primary predictions, found "
                f"{len(locked_predictions)}"
            )
        pair = locked_predictions[
            ["SubjectID", "fold", "y_true", "y_score", "y_pred"]
        ].rename(
            columns={
                "fold": "fold_locked",
                "y_true": "y_locked",
                "y_score": "score_locked",
                "y_pred": "pred_locked",
            }
        ).merge(
            corrected_predictions[
                ["SubjectID", "fold", "y_true", "y_score", "y_pred"]
            ].rename(
                columns={
                    "fold": "fold_corrected",
                    "y_true": "y_corrected",
                    "y_score": "score_corrected",
                    "y_pred": "pred_corrected",
                }
            ),
            on="SubjectID",
            validate="one_to_one",
        )
        if (
            len(pair) != 397
            or not pair["y_locked"].equals(pair["y_corrected"])
            or not pair["fold_locked"].equals(pair["fold_corrected"])
        ):
            raise RuntimeError(
                "Locked/corrected internal prediction parity failed"
            )
        parity_rows = pd.DataFrame(
            [
                {
                    "check": "prediction_subject_count",
                    "status": "PASS",
                    "locked": len(locked_predictions),
                    "corrected_combat": len(corrected_predictions),
                },
                {
                    "check": "prediction_subject_hash",
                    "status": "PASS"
                    if sha256_values(locked_predictions["SubjectID"])
                    == sha256_values(corrected_predictions["SubjectID"])
                    else "FAIL",
                    "locked": sha256_values(locked_predictions["SubjectID"]),
                    "corrected_combat": sha256_values(
                        corrected_predictions["SubjectID"]
                    ),
                },
                {
                    "check": "fold_assignments",
                    "status": "PASS"
                    if pair["fold_locked"].equals(pair["fold_corrected"])
                    else "FAIL",
                    "locked": "397 rows",
                    "corrected_combat": "397 rows",
                },
                {
                    "check": "diagnostic_labels",
                    "status": "PASS"
                    if pair["y_locked"].equals(pair["y_corrected"])
                    else "FAIL",
                    "locked": "397 rows",
                    "corrected_combat": "397 rows",
                },
            ]
        )
        if not parity_rows["status"].eq("PASS").all():
            raise RuntimeError("Prediction parity checks failed")

        primary = primary_metrics_table(
            locked_predictions, corrected_predictions
        )
        bootstrap = paired_bootstrap(
            pair,
            score_locked="score_locked",
            pred_locked="pred_locked",
            score_combat="score_corrected",
            pred_combat="pred_corrected",
            y_col="y_locked",
            comparison="corrected_combat_minus_locked_internal",
        )
        manufacturer = manufacturer_metrics(
            locked_predictions, corrected_predictions
        )
        philips = manufacturer[
            manufacturer["manufacturer"].str.upper().eq("PHILIPS")
        ].copy()

        corrected_oasis = apply_frozen_to_oasis(outdir, frozen_records)
        locked_oasis = load_locked_oasis_predictions()
        oasis_metrics, oasis_bootstrap = oasis_metrics_and_bootstrap(
            locked_oasis, corrected_oasis
        )
        loso = corrected_loso(corrected_cache)
        lineage = add_classifier_leakage_checks(
            leakage,
            fold_metrics,
            corrected_predictions,
            corrected_cache,
        )

        write_table(
            outdir,
            "corrected_combat_latent_cache_audit",
            cache_audit,
            "Corrected ComBat Latent Cache Audit",
        )
        write_table(
            outdir,
            "downstream_classifier_foldwise_metrics",
            fold_metrics,
            "Downstream Diagnostic Classifier Fold-wise Metrics",
        )
        corrected_predictions.to_csv(
            outdir / "downstream_classifier_oof_predictions.csv",
            index=False,
        )
        write_table(
            outdir,
            "downstream_classifier_primary_metrics",
            primary,
            "Downstream Diagnostic Classifier Primary Metrics",
        )
        write_table(
            outdir,
            "paired_bootstrap_vs_locked",
            bootstrap,
            "Paired Bootstrap Versus Locked",
        )
        write_table(
            outdir,
            "manufacturer_subgroup_metrics",
            manufacturer,
            "Manufacturer Subgroup Metrics",
        )
        write_table(
            outdir,
            "philips_cn_fpr_comparison",
            philips,
            "Philips CN False-Positive Rate Comparison",
        )
        write_table(
            outdir,
            "oasis_corrected_primary_metrics",
            oasis_metrics,
            "Corrected Frozen OASIS Primary Metrics",
        )
        corrected_oasis.to_csv(
            outdir / "oasis_corrected_predictions.csv", index=False
        )
        write_table(
            outdir,
            "oasis_paired_bootstrap_vs_locked",
            oasis_bootstrap,
            "OASIS Paired Bootstrap Versus Locked",
        )
        write_table(
            outdir,
            "corrected_loso_metrics",
            loso,
            "Corrected Frozen-Representation LOSO Metrics",
        )
        write_table(
            outdir,
            "lineage_and_leakage_audit",
            lineage,
            "Lineage and Leakage Audit",
        )
        write_table(
            outdir,
            "prediction_and_subject_parity_checks",
            parity_rows,
            "Prediction and Subject Parity Checks",
        )
        pd.DataFrame(frozen_records).to_csv(
            outdir / "frozen_downstream_classifier_manifest.csv", index=False
        )
        final_report(
            outdir,
            primary,
            bootstrap,
            manufacturer,
            oasis_metrics,
            oasis_bootstrap,
            loso,
            cache_audit,
            lineage,
        )
        command_log = {
            "created_utc": utc_now(),
            "script": str(Path(__file__).resolve()),
            "branch": "exploratory/post-revision-20260630",
            "status": "COMPLETE",
            "output_dir": str(OUT),
            "models": {
                "locked": str(LOCKED_RUN),
                "corrected_combat": str(COMBAT_RUN),
            },
            "inputs": {
                "corrected_adni_latent_cache": str(
                    CORRECTED_CACHE_SOURCE
                ),
                "corrected_oasis_latents": str(CORRECTED_OASIS_SOURCE),
                "defective_cache_excluded": str(DEFECTIVE_CACHE),
                "legacy_locked_prediction_filename": str(
                    LOCKED_LEGACY_CLASSIFICATION
                    / "calib_predictions.csv"
                ),
            },
            "downstream_diagnostic_classification": {
                "features": "posterior latent means + Age + Sex",
                "classifier": "L2 logistic regression, class_weight=balanced",
                "c_grid": ORIGINAL_C_GRID,
                "inner_folds": INNER_FOLDS,
                "base_seed": SEED,
                "score_normalization": CALIB_METHOD,
                "threshold_strategy": PRIMARY_THRESHOLD,
                "outer_test_predictions": len(corrected_predictions),
            },
            "oasis": {
                "fold_rows": 900,
                "ensemble_rows": 180,
                "ensemble_score": "mean of five fold-model scores",
                "ensemble_label": "majority vote, >=3 positive decisions",
            },
            "guardrails": {
                "vae_training": False,
                "combat_fitting": False,
                "oasis_fitting_or_recalibration": False,
                "defective_raw_input_cache_reused": False,
                "outer_test_fitting": False,
                "manuscript_edits": False,
                "existing_artifacts_overwritten": False,
            },
            "bootstrap": {
                "n": BOOT_N,
                "seed": BOOT_SEED,
                "stratified_by_diagnosis": True,
            },
            "frozen_classifier_manifest": frozen_records,
        }
        (outdir / "command_log.json").write_text(
            json.dumps(command_log, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(outdir, OUT)
    print(json.dumps({"status": "COMPLETE", "output_dir": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
