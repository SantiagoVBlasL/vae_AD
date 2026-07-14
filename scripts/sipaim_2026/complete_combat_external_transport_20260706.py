#!/usr/bin/env python3
"""Matched locked-vs-fold-ComBat external transport completion.

This script performs readout/diagnostic analyses and frozen VAE inference only.
It does not train a VAE, tune on OASIS, fit a harmonizer on OASIS, or edit a
manuscript.

The original fold-ComBat run did not serialize the fitted CombatModel objects.
For external application, each object is deterministically reconstructed from
the exact original ADNI VAE-pool tensor rows and metadata recorded by the run.
The reconstruction is validated against the saved ADNI test latent cache.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT = Path("/home/diego/proyectos/vae_AD")
RESULTS = PROJECT / "results/revision_bspc_2026"
SCRIPT_DIR = PROJECT / "scripts/revision_bspc_2026"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from foldwise_combat_input_harmonization import (  # noqa: E402
    fit_tensor_combat_channelwise,
    transform_tensor_combat_channelwise,
)
from score_oasis_mega_90_90_external_inference_model_panel_20260604 import (  # noqa: E402
    CandidateSpec,
    PRIMARY_CALIB,
    PRIMARY_FEATURE_SET,
    PRIMARY_MODEL,
    PRIMARY_THRESHOLD,
    binary_metrics,
    encode_external_fold,
    ensure_y,
    fit_adni_stageb_and_score_external,
    load_mega_tensor,
    load_threshold_from_oof,
    normalize_dx,
    normalize_mfr,
    normalize_sex,
    selected_channel_indices,
)
from run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep import load_config  # noqa: E402


OUT = RESULTS / (
    "post_revision_exploratory_20260630/"
    "combat_external_transport_completion_20260706"
)
LOCKED_RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
COMBAT_RUN = RESULTS / (
    "recover035_latent384_beta3p75_foldcombat_mfr_age_sex_T80_h10000_p560_full5x5"
)
LOCKED_STAGEB = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
COMBAT_STAGEB = RESULTS / "recover035_latent384_beta3p75_foldcombat_stageB_oof_score_calibration"
LOCKED_OASIS_PANEL = RESULTS / "oasis_mega_90_90_external_inference_model_panel_20260604"
LOCKED_OASIS_LATENT = RESULTS / (
    "promoted_latent384_oasis_vs_adni_latent_distance_audit_20260604/"
    "oasis_fold_latent_mu_runwise164.csv"
)
OASIS_DIR = RESULTS / "oasis_mega_90cn_90ad_pooled_external_validation_20260531"
OASIS_TENSOR = OASIS_DIR / "tensor_runwise164_pilot_parity.npz"
GLOBAL_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)

LOCKED_SPEC = CandidateSpec(
    label="locked",
    role="locked",
    run_dir=LOCKED_RUN,
    oof_dir=LOCKED_STAGEB,
)
COMBAT_SPEC = CandidateSpec(
    label="foldcombat",
    role="matched_full_foldcombat",
    run_dir=COMBAT_RUN,
    oof_dir=COMBAT_STAGEB,
)

FOLDS = [1, 2, 3, 4, 5]
MFR_SEED = 42
MFR_N_PERM = 1000
BOOT_SEED = 20260706
BOOT_N = 10_000
MU_COLS = [f"mu_{i}" for i in range(384)]
LOSO_SITES = ["130", "035"]


def write_table(stem: str, df: pd.DataFrame, title: str | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{stem}.csv", index=False)
    text = []
    if title:
        text.extend([f"# {title}", ""])
    text.append("_No rows._" if df.empty else df.to_markdown(index=False))
    (OUT / f"{stem}.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def sha256_subjects(values: Iterable[Any]) -> str:
    payload = "\n".join(sorted(map(str, values)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def latent_path(run: Path, fold: int, split: str) -> Path:
    return run / "classifier_only_readout/latent_cache" / f"fold_{fold}_{split}_latent_mu.csv"


def load_latent(run: Path, fold: int, split: str) -> pd.DataFrame:
    df = pd.read_csv(latent_path(run, fold, split))
    df["SiteCode"] = df["SubjectID"].astype(str).str.extract(r"^(\d{3})_S_\d{4}$", expand=False)
    return ensure_y(df)


def load_latent_cache(cache_dir: Path, fold: int, split: str) -> pd.DataFrame:
    df = pd.read_csv(cache_dir / f"fold_{fold}_{split}_latent_mu.csv")
    df["SiteCode"] = df["SubjectID"].astype(str).str.extract(r"^(\d{3})_S_\d{4}$", expand=False)
    return ensure_y(df)


def exact_nuisance_residuals(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Match the reconciled locked fold-local geometry protocol exactly."""
    scaler = StandardScaler()
    z_train = scaler.fit_transform(train[MU_COLS].to_numpy(float))
    z_test = scaler.transform(test[MU_COLS].to_numpy(float))
    age_mean = float(train["Age"].mean())
    age_sd = float(train["Age"].std())
    n_train = np.column_stack(
        [
            train["y"].to_numpy(float),
            (train["Age"].to_numpy(float) - age_mean) / (age_sd + 1e-8),
            train["Sex"].eq("M").to_numpy(float),
        ]
    )
    n_test = np.column_stack(
        [
            test["y"].to_numpy(float),
            (test["Age"].to_numpy(float) - age_mean) / (age_sd + 1e-8),
            test["Sex"].eq("M").to_numpy(float),
        ]
    )
    reg = LinearRegression(fit_intercept=True)
    reg.fit(n_train, z_train)
    return z_train - reg.predict(n_train), z_test - reg.predict(n_test)


def manufacturer_decoding(
    corrected_combat_cache: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parity_rows: list[dict[str, Any]] = []
    model_data: dict[tuple[str, int], dict[str, Any]] = {}
    for fold in FOLDS:
        locked_train = load_latent(LOCKED_RUN, fold, "trainDev")
        locked_test = load_latent(LOCKED_RUN, fold, "test")
        combat_train = load_latent_cache(corrected_combat_cache, fold, "trainDev")
        combat_test = load_latent_cache(corrected_combat_cache, fold, "test")
        for split, a, b in (
            ("trainDev", locked_train, combat_train),
            ("test", locked_test, combat_test),
        ):
            parity_rows.append(
                {
                    "fold": fold,
                    "split": split,
                    "locked_n": len(a),
                    "combat_n": len(b),
                    "locked_subject_sha256": sha256_subjects(a["SubjectID"]),
                    "combat_subject_sha256": sha256_subjects(b["SubjectID"]),
                    "same_subjects": set(a["SubjectID"]) == set(b["SubjectID"]),
                    "same_labels": a.set_index("SubjectID")["y"].sort_index().equals(
                        b.set_index("SubjectID")["y"].sort_index()
                    ),
                    "same_manufacturer": a.set_index("SubjectID")["Manufacturer"].sort_index().equals(
                        b.set_index("SubjectID")["Manufacturer"].sort_index()
                    ),
                }
            )
        if not all(r["same_subjects"] and r["same_labels"] and r["same_manufacturer"] for r in parity_rows[-2:]):
            raise RuntimeError(f"Locked/ComBat subject or label mismatch in fold {fold}")

        for arm, train, test in (
            ("locked", locked_train, locked_test),
            ("foldcombat", combat_train, combat_test),
        ):
            train = train.sort_values("SubjectID").reset_index(drop=True)
            test = test.sort_values("SubjectID").reset_index(drop=True)
            z_train, z_test = exact_nuisance_residuals(train, test)
            encoder = LabelEncoder().fit(train["Manufacturer"].to_numpy())
            y_train = encoder.transform(train["Manufacturer"])
            y_test = encoder.transform(test["Manufacturer"])
            clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
            clf.fit(z_train, y_train)
            pred = clf.predict(z_test)
            model_data[(arm, fold)] = {
                "test": test,
                "y_test": y_test,
                "pred": pred,
                "bacc": float(balanced_accuracy_score(y_test, pred)),
            }

    rng = np.random.default_rng(MFR_SEED)
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for fold in FOLDS:
        n_test = len(model_data[("locked", fold)]["y_test"])
        permutations = np.vstack([rng.permutation(n_test) for _ in range(MFR_N_PERM)])
        for arm in ("locked", "foldcombat"):
            item = model_data[(arm, fold)]
            y_test = item["y_test"]
            pred = item["pred"]
            null = np.asarray(
                [balanced_accuracy_score(y_test[idx], pred) for idx in permutations],
                dtype=float,
            )
            p = float((np.sum(null >= item["bacc"]) + 1) / (len(null) + 1))
            fold_rows.append(
                {
                    "arm": arm,
                    "fold": fold,
                    "input_lineage": (
                        "locked_original"
                        if arm == "locked"
                        else "corrected_harmonized_input_inference"
                    ),
                    "n_trainDev": len(
                        load_latent(LOCKED_RUN, fold, "trainDev")
                        if arm == "locked"
                        else load_latent_cache(corrected_combat_cache, fold, "trainDev")
                    ),
                    "n_test": n_test,
                    "balanced_accuracy": item["bacc"],
                    "permutation_null_mean": float(null.mean()),
                    "permutation_null_sd": float(null.std(ddof=1)),
                    "permutation_p_ge_observed": p,
                    "n_permutations": MFR_N_PERM,
                    "seed": MFR_SEED,
                }
            )
            pp = item["test"][["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"]].copy()
            pp.insert(0, "arm", arm)
            pp.insert(1, "fold", fold)
            pp["manufacturer_encoded"] = y_test
            pp["manufacturer_pred_encoded"] = pred
            pp["correct"] = y_test == pred
            prediction_rows.append(pp)

    fold_df = pd.DataFrame(fold_rows)
    locked = fold_df[fold_df["arm"].eq("locked")].set_index("fold")
    combat = fold_df[fold_df["arm"].eq("foldcombat")].set_index("fold")
    paired = pd.DataFrame(
        {
            "fold": FOLDS,
            "locked_bacc": locked.loc[FOLDS, "balanced_accuracy"].to_numpy(),
            "foldcombat_bacc": combat.loc[FOLDS, "balanced_accuracy"].to_numpy(),
        }
    )
    paired["delta_foldcombat_minus_locked"] = paired["foldcombat_bacc"] - paired["locked_bacc"]
    summary_rows = []
    for arm in ("locked", "foldcombat"):
        vals = fold_df.loc[fold_df["arm"].eq(arm), "balanced_accuracy"].to_numpy()
        summary_rows.append(
            {
                "arm": arm,
                "mean_bacc": float(vals.mean()),
                "sd_bacc_ddof1": float(vals.std(ddof=1)),
                "n_folds": len(vals),
            }
        )
    delta = paired["delta_foldcombat_minus_locked"].to_numpy()
    summary_rows.append(
        {
            "arm": "foldcombat_minus_locked",
            "mean_bacc": float(delta.mean()),
            "sd_bacc_ddof1": float(delta.std(ddof=1)),
            "n_folds": len(delta),
        }
    )
    summary = pd.DataFrame(summary_rows)
    write_table("taskA_subject_fold_parity", pd.DataFrame(parity_rows), "Task A Subject/Fold Parity")
    write_table("taskA_manufacturer_decoding_foldwise", fold_df, "Task A Manufacturer Decoding")
    write_table("taskA_manufacturer_decoding_paired_differences", paired, "Task A Paired Fold Differences")
    write_table("taskA_manufacturer_decoding_summary", summary, "Task A Manufacturer Decoding Summary")
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        OUT / "taskA_manufacturer_decoding_predictions.csv", index=False
    )
    return fold_df, paired, summary


def primary_locked_oasis_predictions() -> pd.DataFrame:
    df = pd.read_csv(LOCKED_OASIS_PANEL / "predictions.csv")
    mask = (
        df["candidate"].eq("promoted_beta3p75_oof_ecdf")
        & df["build_candidate"].eq("runwise164_pilot_parity")
    )
    out = df[mask].copy()
    if len(out) != 1080:
        raise RuntimeError(f"Expected 1080 locked runwise164 prediction rows, found {len(out)}")
    out["arm"] = "locked"
    return out


def load_global_tensor() -> tuple[np.ndarray, list[str]]:
    with np.load(GLOBAL_TENSOR, allow_pickle=True) as zf:
        tensor = np.asarray(zf["global_tensor_data"], dtype=np.float32)
        names = np.asarray(zf["channel_names"]).astype(str).tolist()
    return tensor, names


def reconstruct_combat_and_score_oasis(
    device: torch.device, batch_size: int = 64, n_jobs: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    global_tensor, global_names = load_global_tensor()
    oasis_tensor, oasis_meta, oasis_names = load_mega_tensor(OASIS_TENSOR)
    cfg = load_config(COMBAT_RUN)
    selected_names = list(cfg.get("selected_channel_names") or [])
    global_idx = selected_channel_indices(selected_names, global_names)
    oasis_idx = selected_channel_indices(selected_names, oasis_names)
    x_global = global_tensor[:, global_idx].astype(np.float32)
    x_oasis = oasis_tensor[:, oasis_idx].astype(np.float32)

    prediction_rows: list[pd.DataFrame] = []
    latent_rows: list[pd.DataFrame] = []
    reconstruction_rows: list[dict[str, Any]] = []
    corrected_cache = OUT / "taskAB_corrected_foldcombat_latent_cache"
    corrected_cache.mkdir(parents=True, exist_ok=True)

    for fold in FOLDS:
        fold_dir = COMBAT_RUN / f"fold_{fold}"
        fit_meta = pd.read_csv(
            fold_dir / "input_harmonization_fit_train_dev_vae_pool_subjects.csv"
        )
        fit_idx = fit_meta["tensor_idx"].to_numpy(dtype=int)
        fitted = fit_tensor_combat_channelwise(
            x_global[fit_idx],
            fit_meta,
            channel_indices=[1, 0, 2],
            channel_names=selected_names,
        )
        original_audit = pd.read_csv(fold_dir / "input_harmonization_fit_audit.csv")
        reconstructed_audit = pd.DataFrame(fitted.audit_rows)
        audit_cols = [
            "status",
            "n_fit",
            "n_features",
            "n_variable_features",
            "manufacturer_levels_fit",
            "protected_covariates",
            "batch",
            "excluded_covariates",
            "channel_position",
            "selected_channel_index",
            "channel_name",
        ]
        audit_match = original_audit[audit_cols].astype(str).reset_index(drop=True).equals(
            reconstructed_audit[audit_cols].astype(str).reset_index(drop=True)
        )

        # Lineage check: the saved post-hoc cache used raw (unharmonized) input.
        test_meta = pd.read_csv(fold_dir / "input_harmonization_classifier_test_apply_subjects.csv")
        check_meta = test_meta.head(8).copy()
        check_x = transform_tensor_combat_channelwise(
            fitted,
            x_global[check_meta["tensor_idx"].to_numpy(dtype=int)],
            check_meta,
        ).astype(np.float32)
        check_latent = encode_external_fold(
            COMBAT_SPEC, cfg, check_x, check_meta, fold, device, batch_size
        )
        cached = load_latent(COMBAT_RUN, fold, "test").set_index("SubjectID")
        check_latent = check_latent.set_index("SubjectID")
        cached_mu = cached.loc[check_latent.index, MU_COLS].to_numpy(float)
        rebuilt_mu = check_latent[MU_COLS].to_numpy(float)
        harmonized_vs_cache_max_abs = float(np.max(np.abs(cached_mu - rebuilt_mu)))
        harmonized_vs_cache_mean_abs = float(np.mean(np.abs(cached_mu - rebuilt_mu)))
        raw_check_latent = encode_external_fold(
            COMBAT_SPEC,
            cfg,
            x_global[check_meta["tensor_idx"].to_numpy(dtype=int)],
            check_meta,
            fold,
            device,
            batch_size,
        ).set_index("SubjectID")
        raw_mu = raw_check_latent[MU_COLS].to_numpy(float)
        raw_vs_cache_max_abs = float(np.max(np.abs(cached_mu - raw_mu)))
        raw_vs_cache_mean_abs = float(np.mean(np.abs(cached_mu - raw_mu)))
        if not audit_match or raw_vs_cache_max_abs > 1e-3:
            raise RuntimeError(
                f"Fold {fold} ComBat reconstruction failed: audit_match={audit_match}, "
                f"raw-cache latent max_abs={raw_vs_cache_max_abs}"
            )

        # Correct fold-ComBat latent inference for Task A and Wasserstein.
        train_dev_meta = pd.read_csv(
            fold_dir / "input_harmonization_classifier_train_dev_apply_subjects.csv"
        )
        harmonized_train_dev = transform_tensor_combat_channelwise(
            fitted,
            x_global[train_dev_meta["tensor_idx"].to_numpy(dtype=int)],
            train_dev_meta,
        ).astype(np.float32)
        harmonized_test = transform_tensor_combat_channelwise(
            fitted,
            x_global[test_meta["tensor_idx"].to_numpy(dtype=int)],
            test_meta,
        ).astype(np.float32)
        corrected_train_dev = encode_external_fold(
            COMBAT_SPEC,
            cfg,
            harmonized_train_dev,
            train_dev_meta,
            fold,
            device,
            batch_size,
        )
        corrected_test = encode_external_fold(
            COMBAT_SPEC,
            cfg,
            harmonized_test,
            test_meta,
            fold,
            device,
            batch_size,
        )
        corrected_train_dev.to_csv(
            corrected_cache / f"fold_{fold}_trainDev_latent_mu.csv", index=False
        )
        corrected_test.to_csv(
            corrected_cache / f"fold_{fold}_test_latent_mu.csv", index=False
        )

        harmonized_oasis = transform_tensor_combat_channelwise(
            fitted, x_oasis, oasis_meta
        ).astype(np.float32)
        ext = encode_external_fold(
            COMBAT_SPEC, cfg, harmonized_oasis, oasis_meta, fold, device, batch_size
        )
        ext["ResearchGroup_Mapped"] = ext["ResearchGroup_Mapped"].map(normalize_dx)
        ext["Sex"] = ext["Sex"].map(normalize_sex)
        ext["Manufacturer"] = ext["Manufacturer"].map(normalize_mfr)
        ext["Age"] = pd.to_numeric(ext["Age"], errors="coerce")
        train = load_latent(COMBAT_RUN, fold, "trainDev")
        raw_score, ecdf_score, readout_meta = fit_adni_stageb_and_score_external(
            train, ext, fold=fold, n_jobs=n_jobs, ecdf_mode="interp"
        )
        threshold = load_threshold_from_oof(COMBAT_SPEC, fold)
        y_pred = (ecdf_score >= threshold).astype(int)
        keep_cols = [
            "SubjectID",
            "subject_id",
            "session_id",
            "experiment_id",
            "source_batch",
            "protocol_subset",
            "diagnosis",
            "ResearchGroup_Mapped",
            "Age",
            "Sex",
            "Manufacturer",
            "ScannerModel",
            "selected_qc_runs",
            "selected_run_ids",
            "mean_fd_subject",
            "max_fd_subject",
            "y",
        ]
        pred = ext[[c for c in keep_cols if c in ext.columns]].copy()
        pred["arm"] = "foldcombat"
        pred["build_candidate"] = "runwise164_pilot_parity"
        pred["candidate"] = "foldcombat"
        pred["model_name"] = PRIMARY_MODEL
        pred["feature_set"] = PRIMARY_FEATURE_SET
        pred["calib_method"] = PRIMARY_CALIB
        pred["threshold_strategy"] = PRIMARY_THRESHOLD
        pred["fold"] = fold
        pred["prediction_level"] = "fold_model"
        pred["y_score_raw"] = raw_score
        pred["y_score"] = ecdf_score
        pred["threshold"] = threshold
        pred["y_pred"] = y_pred
        prediction_rows.append(pred)

        lat = ext[
            [
                "SubjectID",
                "ResearchGroup_Mapped",
                "y",
                "Age",
                "Sex",
                "Manufacturer",
                *MU_COLS,
            ]
        ].copy()
        lat.insert(0, "fold", fold)
        latent_rows.append(lat)
        reconstruction_rows.append(
            {
                "fold": fold,
                "fit_scope": "original_ADNI_VAE_pool_only",
                "n_fit": len(fit_meta),
                "oasis_rows_used_for_fit": 0,
                "audit_fields_match_original": audit_match,
                "adni_test_latent_check_n": len(check_meta),
                "saved_stageB_cache_input_lineage": "raw_unharmonized_input",
                "raw_input_vs_saved_cache_max_abs_diff": raw_vs_cache_max_abs,
                "raw_input_vs_saved_cache_mean_abs_diff": raw_vs_cache_mean_abs,
                "harmonized_input_vs_saved_cache_max_abs_diff": harmonized_vs_cache_max_abs,
                "harmonized_input_vs_saved_cache_mean_abs_diff": harmonized_vs_cache_mean_abs,
                "corrected_trainDev_latent_n": len(corrected_train_dev),
                "corrected_test_latent_n": len(corrected_test),
                "readout_best_inner_auc": readout_meta["best_inner_auc"],
                "readout_best_params": readout_meta["best_params"],
                "readout_inner_cv_context": readout_meta["inner_cv_context"],
                "readout_n_oof": readout_meta["n_oof"],
                "threshold": threshold,
            }
        )
        del (
            fitted,
            harmonized_oasis,
            harmonized_train_dev,
            harmonized_test,
            corrected_train_dev,
            corrected_test,
            ext,
            check_latent,
            raw_check_latent,
        )

    fold_preds = pd.concat(prediction_rows, ignore_index=True)
    identity = [
        "SubjectID",
        "subject_id",
        "session_id",
        "experiment_id",
        "source_batch",
        "protocol_subset",
        "diagnosis",
        "ResearchGroup_Mapped",
        "Age",
        "Sex",
        "Manufacturer",
        "ScannerModel",
        "selected_qc_runs",
        "selected_run_ids",
        "mean_fd_subject",
        "max_fd_subject",
        "y",
    ]
    identity = [c for c in identity if c in fold_preds.columns]
    ens = (
        fold_preds.groupby(identity, dropna=False)
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
    ens["arm"] = "foldcombat"
    ens["build_candidate"] = "runwise164_pilot_parity"
    ens["candidate"] = "foldcombat"
    ens["model_name"] = PRIMARY_MODEL
    ens["feature_set"] = PRIMARY_FEATURE_SET
    ens["calib_method"] = PRIMARY_CALIB
    ens["threshold_strategy"] = PRIMARY_THRESHOLD
    ens["fold"] = "ensemble_mean_score_majority_vote"
    ens["prediction_level"] = "ensemble_mean_score_majority_vote"
    ens["y_pred"] = (ens["fold_positive_votes"] >= 3).astype(int)
    all_preds = pd.concat([fold_preds, ens], ignore_index=True, sort=False)
    latents = pd.concat(latent_rows, ignore_index=True)
    reconstruction = pd.DataFrame(reconstruction_rows)
    write_table(
        "taskB_combat_harmonizer_reconstruction_audit",
        reconstruction,
        "Task B ComBat Harmonizer Reconstruction Audit",
    )
    all_preds.to_csv(OUT / "taskB_combat_oasis_predictions.csv", index=False)
    latents.to_csv(OUT / "taskB_combat_oasis_fold_latent_mu_runwise164.csv", index=False)
    return all_preds, latents, reconstruction, corrected_cache


def metric_dict(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    m = binary_metrics(y, score, pred)
    return {
        "roc_auc": float(m["auc"]),
        "pr_auc": float(m["pr_auc"]),
        "sensitivity": float(m["sensitivity"]),
        "specificity": float(m["specificity"]),
        "balanced_accuracy": float(m["balanced_accuracy"]),
        "mean_score": float(np.mean(score)),
        "cn_mean_score": float(np.mean(score[y == 0])),
        "ad_mean_score": float(np.mean(score[y == 1])),
    }


def paired_oasis_comparison(
    locked_all: pd.DataFrame, combat_all: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    locked = locked_all[
        locked_all["prediction_level"].eq("ensemble_mean_score_majority_vote")
    ].copy()
    combat = combat_all[
        combat_all["prediction_level"].eq("ensemble_mean_score_majority_vote")
    ].copy()
    a = locked[["SubjectID", "y", "y_score", "y_pred"]].rename(
        columns={"y": "y_locked", "y_score": "score_locked", "y_pred": "pred_locked"}
    )
    b = combat[["SubjectID", "y", "y_score", "y_pred"]].rename(
        columns={"y": "y_combat", "y_score": "score_combat", "y_pred": "pred_combat"}
    )
    paired = a.merge(b, on="SubjectID", how="inner", validate="one_to_one")
    if len(paired) != 180 or not paired["y_locked"].equals(paired["y_combat"]):
        raise RuntimeError("Locked/ComBat OASIS ensemble subject or label mismatch")
    y = paired["y_locked"].to_numpy(int)
    observed = {
        "locked": metric_dict(
            y, paired["score_locked"].to_numpy(float), paired["pred_locked"].to_numpy(int)
        ),
        "foldcombat": metric_dict(
            y, paired["score_combat"].to_numpy(float), paired["pred_combat"].to_numpy(int)
        ),
    }
    metric_rows = []
    for arm in ("locked", "foldcombat"):
        row = {"arm": arm, "n": len(y), "n_cn": int((y == 0).sum()), "n_ad": int((y == 1).sum())}
        row["validity_status"] = (
            "valid_locked_reference"
            if arm == "locked"
            else "artifact_faithful_not_valid_end_to_end_combat"
        )
        row.update(observed[arm])
        tn, fp, fn, tp = confusion_matrix(
            y,
            paired[f"pred_{'locked' if arm == 'locked' else 'combat'}"].to_numpy(int),
            labels=[0, 1],
        ).ravel()
        row.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)

    rng = np.random.default_rng(BOOT_SEED)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    names = list(observed["locked"])
    deltas = {name: np.empty(BOOT_N, dtype=float) for name in names}
    for i in range(BOOT_N):
        idx = np.concatenate(
            [
                rng.choice(idx0, size=len(idx0), replace=True),
                rng.choice(idx1, size=len(idx1), replace=True),
            ]
        )
        boot_y = y[idx]
        locked_m = metric_dict(
            boot_y,
            paired["score_locked"].to_numpy(float)[idx],
            paired["pred_locked"].to_numpy(int)[idx],
        )
        combat_m = metric_dict(
            boot_y,
            paired["score_combat"].to_numpy(float)[idx],
            paired["pred_combat"].to_numpy(int)[idx],
        )
        for name in names:
            deltas[name][i] = combat_m[name] - locked_m[name]
    boot_rows = []
    for name in names:
        values = deltas[name]
        observed_delta = observed["foldcombat"][name] - observed["locked"][name]
        boot_rows.append(
            {
                "metric": name,
                "comparison": "foldcombat_minus_locked",
                "validity_status": "artifact_faithful_not_valid_end_to_end_combat",
                "observed_delta": observed_delta,
                "bootstrap_mean_delta": float(values.mean()),
                "ci_low_2p5": float(np.quantile(values, 0.025)),
                "ci_high_97p5": float(np.quantile(values, 0.975)),
                "p_delta_gt_0": float(np.mean(values > 0)),
                "n_bootstrap": BOOT_N,
                "seed": BOOT_SEED,
            }
        )
    bootstrap = pd.DataFrame(boot_rows)

    dist_rows = []
    for arm, df in (("locked", locked), ("foldcombat", combat)):
        for diagnosis, sub in df.groupby("ResearchGroup_Mapped"):
            score = sub["y_score"].to_numpy(float)
            dist_rows.append(
                {
                    "arm": arm,
                    "diagnosis": diagnosis,
                    "validity_status": (
                        "valid_locked_reference"
                        if arm == "locked"
                        else "artifact_faithful_not_valid_end_to_end_combat"
                    ),
                    "n": len(score),
                    "mean": float(score.mean()),
                    "sd": float(score.std(ddof=1)),
                    "median": float(np.median(score)),
                    "q05": float(np.quantile(score, 0.05)),
                    "q25": float(np.quantile(score, 0.25)),
                    "q75": float(np.quantile(score, 0.75)),
                    "q95": float(np.quantile(score, 0.95)),
                    "min": float(score.min()),
                    "max": float(score.max()),
                }
            )
    distributions = pd.DataFrame(dist_rows)
    paired.to_csv(OUT / "taskB_oasis_paired_subject_predictions.csv", index=False)
    write_table("taskB_oasis_primary_metrics", metrics, "Task B OASIS Primary Metrics")
    write_table(
        "taskB_oasis_paired_subject_bootstrap",
        bootstrap,
        "Task B Paired Subject-Level Bootstrap",
    )
    write_table(
        "taskB_oasis_score_distributions",
        distributions,
        "Task B OASIS Score Distributions",
    )
    return metrics, bootstrap, distributions


def w1_per_dim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(
        [wasserstein_distance(a[:, j], b[:, j]) for j in range(a.shape[1])],
        dtype=float,
    )


def gaussian_w2(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    lwa = LedoitWolf(store_precision=False).fit(a)
    lwb = LedoitWolf(store_precision=False).fit(b)
    sa, sb = lwa.covariance_, lwb.covariance_
    sqrt_sa = sqrtm(sa).real
    sqrt_inner = sqrtm(sqrt_sa @ sb @ sqrt_sa).real
    mean_sq = float(np.sum((lwa.location_ - lwb.location_) ** 2))
    covariance_term = float(np.trace(sa) + np.trace(sb) - 2 * np.trace(sqrt_inner))
    return (
        float(np.sqrt(max(mean_sq + covariance_term, 0.0))),
        float(lwa.shrinkage_),
        float(lwb.shrinkage_),
    )


def wasserstein_comparison(
    combat_oasis_latents: pd.DataFrame, corrected_combat_cache: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    locked_oasis = pd.read_csv(LOCKED_OASIS_LATENT)
    rows = []
    per_pc_rows = []
    arm_sources = {
        "locked": ("run", LOCKED_RUN, locked_oasis),
        "foldcombat": ("cache", corrected_combat_cache, combat_oasis_latents),
    }
    for arm, (source_type, source, oasis_all) in arm_sources.items():
        for fold in FOLDS:
            adni = (
                load_latent(source, fold, "trainDev")
                if source_type == "run"
                else load_latent_cache(source, fold, "trainDev")
            )
            oasis = oasis_all[oasis_all["fold"].astype(int).eq(fold)].copy()
            if len(oasis) != 180:
                raise RuntimeError(f"{arm} fold {fold}: expected 180 OASIS latents")
            scaler = StandardScaler()
            z_adni = scaler.fit_transform(adni[MU_COLS].to_numpy(float))
            z_oasis = scaler.transform(oasis[MU_COLS].to_numpy(float))
            reg = LinearRegression(fit_intercept=True)
            reg.fit(adni[["y"]].to_numpy(float), z_adni)
            z_adni_r = z_adni - reg.predict(adni[["y"]].to_numpy(float))
            z_oasis_r = z_oasis - reg.predict(oasis[["y"]].to_numpy(float))
            pca = PCA(n_components=10, random_state=42).fit(z_adni_r)
            p_adni = pca.transform(z_adni_r)
            p_oasis = pca.transform(z_oasis_r)
            pca_w1 = np.asarray(
                [wasserstein_distance(p_adni[:, j], p_oasis[:, j]) for j in range(10)]
            )
            for pc, value in enumerate(pca_w1, start=1):
                per_pc_rows.append({"arm": arm, "fold": fold, "pc": pc, "w1_pc": value})
            per_dim = w1_per_dim(z_adni_r, z_oasis_r)
            gw2, shr_adni, shr_oasis = gaussian_w2(z_adni_r, z_oasis_r)
            rows.append(
                {
                    "arm": arm,
                    "fold": fold,
                    "input_lineage": (
                        "locked_original"
                        if arm == "locked"
                        else "corrected_harmonized_input_inference"
                    ),
                    "n_adni_trainDev": len(adni),
                    "n_oasis": len(oasis),
                    "residualization": "diagnosis_y_only_OLS_fit_ADNI_trainDev",
                    "w1_mean_all_dims": float(per_dim.mean()),
                    "w1_median_all_dims": float(np.median(per_dim)),
                    "w1_max_all_dims": float(per_dim.max()),
                    "pca10_w1_mean": float(pca_w1.mean()),
                    "gaussian_w2": gw2,
                    "lw_shrinkage_adni": shr_adni,
                    "lw_shrinkage_oasis": shr_oasis,
                }
            )
    distances = pd.DataFrame(rows)
    locked = distances[distances["arm"].eq("locked")].set_index("fold")
    combat = distances[distances["arm"].eq("foldcombat")].set_index("fold")
    delta_rows = []
    for fold in FOLDS:
        for metric in ("w1_mean_all_dims", "pca10_w1_mean", "gaussian_w2"):
            lv = float(locked.loc[fold, metric])
            cv = float(combat.loc[fold, metric])
            delta_rows.append(
                {
                    "fold": fold,
                    "metric": metric,
                    "locked": lv,
                    "foldcombat": cv,
                    "delta_foldcombat_minus_locked": cv - lv,
                    "oasis_closer_under_foldcombat": cv < lv,
                }
            )
    deltas = pd.DataFrame(delta_rows)
    summary_rows = []
    for metric, sub in deltas.groupby("metric"):
        d = sub["delta_foldcombat_minus_locked"].to_numpy(float)
        summary_rows.append(
            {
                "metric": metric,
                "locked_mean": float(sub["locked"].mean()),
                "locked_sd_ddof1": float(sub["locked"].std(ddof=1)),
                "foldcombat_mean": float(sub["foldcombat"].mean()),
                "foldcombat_sd_ddof1": float(sub["foldcombat"].std(ddof=1)),
                "mean_paired_delta": float(d.mean()),
                "sd_paired_delta_ddof1": float(d.std(ddof=1)),
                "folds_closer_n": int(sub["oasis_closer_under_foldcombat"].sum()),
                "folds_total": len(sub),
            }
        )
    summary = pd.DataFrame(summary_rows)
    write_table(
        "taskB_wasserstein_adni_vs_oasis_by_fold",
        distances,
        "Task B ADNI-vs-OASIS Wasserstein by Fold",
    )
    write_table(
        "taskB_wasserstein_paired_fold_differences",
        deltas,
        "Task B Wasserstein Paired Fold Differences",
    )
    write_table(
        "taskB_wasserstein_summary",
        summary,
        "Task B Wasserstein Summary",
    )
    pd.DataFrame(per_pc_rows).to_csv(
        OUT / "taskB_wasserstein_pca_by_pc_fold.csv", index=False
    )
    return distances, summary


def loso_readout() -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_rows = []
    metric_rows = []
    for arm, run in (("locked", LOCKED_RUN), ("foldcombat", COMBAT_RUN)):
        for target in LOSO_SITES:
            site_rows = []
            for fold in FOLDS:
                train = load_latent(run, fold, "trainDev")
                test = load_latent(run, fold, "test")
                test = test[test["SiteCode"].eq(target)].copy()
                if test.empty:
                    continue
                train = train[~train["SiteCode"].eq(target)].copy()
                train["Sex_num"] = train["Sex"].eq("M").astype(float)
                test["Sex_num"] = test["Sex"].eq("M").astype(float)
                cols = MU_COLS + ["Age", "Sex_num"]
                scaler = StandardScaler()
                x_train = scaler.fit_transform(train[cols].to_numpy(float))
                x_test = scaler.transform(test[cols].to_numpy(float))
                clf = LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=1.0,
                    solver="lbfgs",
                )
                clf.fit(x_train, train["y"].to_numpy(int))
                score = clf.predict_proba(x_test)[:, 1]
                pred = (score >= 0.5).astype(int)
                frame = test[
                    ["SubjectID", "SiteCode", "y", "ResearchGroup_Mapped", "Manufacturer"]
                ].copy()
                frame.insert(0, "arm", arm)
                frame.insert(1, "target_site", target)
                frame.insert(2, "fold", fold)
                frame["score"] = score
                frame["pred"] = pred
                site_rows.append(frame)
            pp = pd.concat(site_rows, ignore_index=True)
            pred_rows.append(pp)
            m = binary_metrics(pp["y"], pp["score"], pp["pred"])
            metric_rows.append(
                {
                    "arm": arm,
                    "target_site": target,
                    "lineage_status": (
                        "locked_original"
                        if arm == "locked"
                        else "existing_raw_input_cache_defect"
                    ),
                    **m,
                }
            )
    predictions = pd.concat(pred_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    locked = metrics[metrics["arm"].eq("locked")].set_index("target_site")
    combat = metrics[metrics["arm"].eq("foldcombat")].set_index("target_site")
    paired_rows = []
    for site in LOSO_SITES:
        for metric in (
            "auc",
            "pr_auc",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
        ):
            lv = float(locked.loc[site, metric])
            cv = float(combat.loc[site, metric])
            paired_rows.append(
                {
                    "target_site": site,
                    "metric": metric,
                    "locked": lv,
                    "foldcombat": cv,
                    "delta_foldcombat_minus_locked": cv - lv,
                }
            )
    paired = pd.DataFrame(paired_rows)
    predictions.to_csv(OUT / "taskC_loso_predictions.csv", index=False)
    write_table("taskC_loso_metrics", metrics, "Task C Frozen-Latent LOSO Metrics")
    write_table("taskC_loso_paired_differences", paired, "Task C LOSO Paired Differences")
    return metrics, paired


def write_final_report(
    task_a_summary: pd.DataFrame,
    oasis_metrics: pd.DataFrame,
    oasis_bootstrap: pd.DataFrame,
    wasserstein_summary: pd.DataFrame,
    loso_metrics: pd.DataFrame,
    reconstruction: pd.DataFrame,
) -> None:
    def row(df: pd.DataFrame, col: str, value: str) -> pd.Series:
        return df[df[col].eq(value)].iloc[0]

    locked_a = row(task_a_summary, "arm", "locked")
    combat_a = row(task_a_summary, "arm", "foldcombat")
    delta_a = row(task_a_summary, "arm", "foldcombat_minus_locked")
    locked_o = row(oasis_metrics, "arm", "locked")
    combat_o = row(oasis_metrics, "arm", "foldcombat")
    auc_boot = row(oasis_bootstrap, "metric", "roc_auc")
    pr_boot = row(oasis_bootstrap, "metric", "pr_auc")
    closer = "; ".join(
        f"{r.metric}: {int(r.folds_closer_n)}/{int(r.folds_total)} folds, "
        f"mean delta={r.mean_paired_delta:+.4f}"
        for r in wasserstein_summary.itertuples()
    )
    report = f"""# Locked vs fold-ComBat external transport completion

## Status

Partial completion with a pre-existing Stage-B lineage blocker. Task A, frozen
inference, Wasserstein, and Task C ran without VAE training; no Stage-B
component, threshold, calibration, or harmonizer was fit or tuned on OASIS; no
manuscript was edited. The requested OASIS operation was executed, but its
fold-ComBat performance estimate is not valid end to end for the reason below.

The original run did not serialize its fitted `CombatModel` objects. Each fold
was therefore reconstructed only from its exact recorded ADNI VAE-pool rows.
All five reconstructed fit audits matched the originals.

The saved post-hoc fold-ComBat Stage-B latent cache has an input-lineage defect:
it exactly matches raw-input inference (maximum sampled absolute difference
{reconstruction['raw_input_vs_saved_cache_max_abs_diff'].max():.3g}) and not
harmonized-input inference (maximum difference
{reconstruction['harmonized_input_vs_saved_cache_max_abs_diff'].max():.3g}).
Task A and Wasserstein therefore use newly inferred, correctly harmonized ADNI
latents. The requested OASIS scores use the existing Stage-B readout lineage and
are reported as artifact-faithful but not a valid end-to-end ComBat estimate.

## Task A — matched manufacturer decoding

- Locked: BACC {locked_a.mean_bacc:.4f} +/- {locked_a.sd_bacc_ddof1:.4f}.
- Fold-ComBat: BACC {combat_a.mean_bacc:.4f} +/- {combat_a.sd_bacc_ddof1:.4f}.
- Paired fold delta (ComBat - locked): {delta_a.mean_bacc:+.4f} +/- {delta_a.sd_bacc_ddof1:.4f}.
- Protocol: 5-NN Euclidean; trainDev-fitted StandardScaler and OLS nuisance
  residualization for diagnosis, age, and sex; outer-test evaluation; 1000
  shared label permutations per fold from seed 42.

## Task B — frozen runwise164 OASIS evaluation

| arm | ROC-AUC | PR-AUC | sensitivity | specificity | balanced accuracy |
|---|---:|---:|---:|---:|---:|
| locked | {locked_o.roc_auc:.4f} | {locked_o.pr_auc:.4f} | {locked_o.sensitivity:.4f} | {locked_o.specificity:.4f} | {locked_o.balanced_accuracy:.4f} |
| fold-ComBat | {combat_o.roc_auc:.4f} | {combat_o.pr_auc:.4f} | {combat_o.sensitivity:.4f} | {combat_o.specificity:.4f} | {combat_o.balanced_accuracy:.4f} |

- Paired ROC-AUC delta: {auc_boot.observed_delta:+.4f}, 95% bootstrap CI
  [{auc_boot.ci_low_2p5:+.4f}, {auc_boot.ci_high_97p5:+.4f}].
- Paired PR-AUC delta: {pr_boot.observed_delta:+.4f}, 95% bootstrap CI
  [{pr_boot.ci_low_2p5:+.4f}, {pr_boot.ci_high_97p5:+.4f}].
- Wasserstein movement toward the corresponding ADNI trainDev distribution:
  {closer}.

**Validity status:** the harmonizer-to-VAE part is correct and OASIS remains
fully frozen, but the existing Stage-B readout was built from the erroneous
raw-input ADNI cache. A valid corrected end-to-end estimate would require
refitting Stage B on the corrected harmonized ADNI latent cache, which is
outside the no-model-training guardrail.

## Task C — supported-site frozen-latent LOSO

{loso_metrics.to_markdown(index=False)}

## Interpretation guardrail

The LOSO analysis retrains only the specified logistic readout on the existing
fold-local latent caches after excluding the target site. It is not VAE-LOSO.
For fold-ComBat, those requested existing caches have the raw-input lineage
defect described above. Raw latent coordinates are never pooled across folds
before estimator fitting.
"""
    (OUT / "00_FINAL_REPORT.md").write_text(report, encoding="utf-8")


def preflight() -> pd.DataFrame:
    rows = []
    required = [
        LOCKED_RUN,
        COMBAT_RUN,
        LOCKED_STAGEB,
        COMBAT_STAGEB,
        LOCKED_OASIS_PANEL / "predictions.csv",
        LOCKED_OASIS_LATENT,
        OASIS_TENSOR,
        GLOBAL_TENSOR,
    ]
    for path in required:
        rows.append({"artifact": str(path), "exists": path.exists()})
    for run in (LOCKED_RUN, COMBAT_RUN):
        for fold in FOLDS:
            for split in ("trainDev", "test"):
                path = latent_path(run, fold, split)
                rows.append({"artifact": str(path), "exists": path.exists()})
    out = pd.DataFrame(rows)
    if not out["exists"].all():
        raise FileNotFoundError(out[~out["exists"]].to_string(index=False))
    write_table("00_preflight_artifacts", out, "Preflight Artifacts")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preflight()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        combat_predictions,
        combat_latents,
        reconstruction,
        corrected_combat_cache,
    ) = reconstruct_combat_and_score_oasis(device=device)
    task_a_fold, task_a_paired, task_a_summary = manufacturer_decoding(
        corrected_combat_cache
    )
    locked_predictions = primary_locked_oasis_predictions()
    oasis_metrics, oasis_bootstrap, oasis_distributions = paired_oasis_comparison(
        locked_predictions, combat_predictions
    )
    wasserstein_distances, wasserstein_summary = wasserstein_comparison(
        combat_latents, corrected_combat_cache
    )
    loso_metrics, loso_paired = loso_readout()
    write_final_report(
        task_a_summary,
        oasis_metrics,
        oasis_bootstrap,
        wasserstein_summary,
        loso_metrics,
        reconstruction,
    )
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "output_dir": str(OUT),
        "device": str(device),
        "models": {"locked": str(LOCKED_RUN), "foldcombat": str(COMBAT_RUN)},
        "oasis_tensor": str(OASIS_TENSOR),
        "oasis_build": "runwise164_pilot_parity",
        "status": "PARTIAL_COMPLETE_STAGEB_LINEAGE_BLOCKER",
        "guardrails": {
            "vae_training": False,
            "stageB_tuning_on_oasis": False,
            "oasis_harmonizer_fit": False,
            "threshold_recalibration": False,
            "raw_latent_pooling_across_folds": False,
            "manuscript_edits": False,
        },
        "combat_parameter_provenance": (
            "Original fitted objects were not serialized. Deterministically reconstructed "
            "per fold from exact original ADNI VAE-pool tensor_idx and metadata; verified "
            "against original fit audit. The post-hoc Stage-B cache was proven to come from "
            "raw unharmonized inference; corrected harmonized ADNI latents were generated "
            "for Task A and Wasserstein without model training."
        ),
        "stageB_validity": (
            "Artifact-faithful external scoring only: harmonized OASIS latents were scored "
            "by the existing Stage-B lineage, but that lineage was fitted from a raw-input "
            "ADNI latent cache. Not a valid corrected end-to-end ComBat performance estimate."
        ),
        "taskA": {
            "classifier": "KNeighborsClassifier(n_neighbors=5, metric='euclidean')",
            "nuisance": "diagnosis_y+Age_z+Sex_M",
            "n_permutations": MFR_N_PERM,
            "seed": MFR_SEED,
        },
        "taskB": {
            "ensemble_score": "arithmetic mean of five fold OOF-ECDF scores",
            "ensemble_label": "majority vote, >=3 positive fold decisions",
            "paired_bootstrap_n": BOOT_N,
            "paired_bootstrap_seed": BOOT_SEED,
            "wasserstein_residualization": "diagnosis_y_only; OLS fit on arm-specific ADNI trainDev",
        },
        "taskC": {
            "sites": LOSO_SITES,
            "scope": "existing fold-local latent caches only; logistic readout refit excluding target site",
        },
    }
    (OUT / "command_log.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PARTIAL_COMPLETE_STAGEB_LINEAGE_BLOCKER",
                "output_dir": str(OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
