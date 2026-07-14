#!/usr/bin/env python3
"""Post-hoc OASIS-as-external-batch ComBat analysis for SIPAIM transportability.

Read-only with respect to completed model artifacts. The script fits only
unsupervised external-domain ComBat transforms on ADNI train/dev plus OASIS
connectivity features using Dataset as the batch variable and Age/Sex as
preserved covariates. OASIS diagnosis labels are not passed to ComBat,
calibration, model selection, threshold selection, or feature selection.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from neuroCombat import neuroCombat
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

if not hasattr(np, "int"):
    # neuroCombat 0.x still references np.int in its ref_batch code path.
    # Keep this compatibility shim local to the post-hoc audit script rather
    # than modifying the installed package.
    np.int = int  # type: ignore[attr-defined]


PROJECT = Path("/home/diego/proyectos/vae_AD")
RESULTS = PROJECT / "results/revision_bspc_2026"
SCRIPT_DIR = PROJECT / "scripts/revision_bspc_2026"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep import load_config  # noqa: E402
from score_oasis_mega_90_90_external_inference_model_panel_20260604 import (  # noqa: E402
    CandidateSpec,
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


OUT = RESULTS / (
    "post_revision_exploratory_20260630/"
    "oasis_external_batch_combat_20260710"
)
LOCKED_RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
LOCKED_STAGEB = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
LOCKED_OASIS_PANEL = RESULTS / "oasis_mega_90_90_external_inference_model_panel_20260604"
CLEANREPRO_AUDIT = RESULTS / (
    "post_revision_exploratory_20260630/"
    "foldcombat_cleanrepro_final_audit_20260706"
)
GLOBAL_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
OASIS_TENSOR = RESULTS / (
    "oasis_mega_90cn_90ad_pooled_external_validation_20260531/"
    "tensor_runwise164_pilot_parity.npz"
)

FOLDS = [1, 2, 3, 4, 5]
MU_COLS = [f"mu_{i}" for i in range(384)]
PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURE_SET = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
BOOT_N = 10_000
BOOT_SEED = 20260710


@dataclass(frozen=True)
class Variant:
    arm: str
    label: str
    ref_batch: str | None
    methodological_status: str


VARIANTS = [
    Variant(
        arm="external_dataset_combat_no_reference",
        label="Dataset ComBat, no reference batch",
        ref_batch=None,
        methodological_status=(
            "exploratory_feature_space_adaptation; pooled ComBat target is not "
            "identical to the locked VAE training input distribution"
        ),
    ),
    Variant(
        arm="external_dataset_combat_adni_reference",
        label="Dataset ComBat, ADNI reference batch",
        ref_batch="ADNI",
        methodological_status=(
            "primary_posthoc_feature_space_adaptation; ADNI reference batch "
            "preserves the locked VAE training domain as the target"
        ),
    ),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_values(values: Iterable[Any], *, sort_values: bool = True) -> str:
    vals = [str(v) for v in values]
    if sort_values:
        vals = sorted(vals)
    return hashlib.sha256("\n".join(vals).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_table(stem: str, df: pd.DataFrame, title: str, max_rows: int = 200) -> None:
    csv_path = OUT / f"{stem}.csv"
    md_path = OUT / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    if df.empty:
        body = "_No rows._\n"
    else:
        body = df.head(max_rows).to_markdown(index=False)
        if len(df) > max_rows:
            body += f"\n\n_Showing {max_rows} of {len(df)} rows._"
        body += "\n"
    md_path.write_text(f"# {title}\n\n{body}", encoding="utf-8")


def normalize_sex_for_covars(value: Any) -> str:
    return normalize_sex(value)


def metric_dict(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = np.asarray(pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y, score)),
        "predicted_ad_rate": float(np.mean(pred)),
        "mean_score": float(np.mean(score)),
        "sd_score": float(np.std(score, ddof=1)),
        "cn_mean_score": float(np.mean(score[y == 0])),
        "ad_mean_score": float(np.mean(score[y == 1])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def upper_features(mats: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    tri = np.triu_indices(mats.shape[-1], k=1)
    return np.asarray(mats[:, tri[0], tri[1]], dtype=np.float64), tri


def reconstruct(features: np.ndarray, tri: tuple[np.ndarray, np.ndarray], n_rois: int, diag: np.ndarray) -> np.ndarray:
    out = np.zeros((features.shape[0], n_rois, n_rois), dtype=np.float64)
    out[:, tri[0], tri[1]] = features
    out[:, tri[1], tri[0]] = features
    idx = np.arange(n_rois)
    out[:, idx, idx] = diag
    return out


def covars_frame(adni_meta: pd.DataFrame, oasis_meta: pd.DataFrame) -> pd.DataFrame:
    a = pd.DataFrame(
        {
            "Dataset": "ADNI",
            "Age": pd.to_numeric(adni_meta["Age"], errors="raise"),
            "Sex": adni_meta["Sex"].map(normalize_sex_for_covars),
        }
    )
    o = pd.DataFrame(
        {
            "Dataset": "OASIS",
            "Age": pd.to_numeric(oasis_meta["Age"], errors="raise"),
            "Sex": oasis_meta["Sex"].map(normalize_sex_for_covars),
        }
    )
    cov = pd.concat([a, o], ignore_index=True)
    if cov["Sex"].isin(["UNKNOWN", "", "NAN"]).any():
        raise RuntimeError("Age/Sex covariates are not complete for ComBat.")
    return cov


def combat_channel(
    adni_channel: np.ndarray,
    oasis_channel: np.ndarray,
    adni_meta: pd.DataFrame,
    oasis_meta: pd.DataFrame,
    *,
    variant: Variant,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    adni_x, tri = upper_features(adni_channel)
    oasis_x, tri_o = upper_features(oasis_channel)
    if not (np.array_equal(tri[0], tri_o[0]) and np.array_equal(tri[1], tri_o[1])):
        raise RuntimeError("Upper-triangle ordering mismatch")
    combined = np.vstack([adni_x, oasis_x])
    finite = np.isfinite(combined).all(axis=0)
    variable = finite & (np.nanvar(combined, axis=0) > 1e-12)
    out = combined.copy()
    stdout = io.StringIO()
    if int(variable.sum()) > 0:
        covars = covars_frame(adni_meta, oasis_meta)
        with contextlib.redirect_stdout(stdout):
            result = neuroCombat(
                dat=combined[:, variable].T,
                covars=covars,
                batch_col="Dataset",
                categorical_cols=["Sex"],
                continuous_cols=["Age"],
                ref_batch=variant.ref_batch,
            )
        out[:, variable] = np.asarray(result["data"], dtype=np.float64).T
    n_adni = len(adni_meta)
    n_rois = adni_channel.shape[-1]
    diag_adni = np.diagonal(adni_channel, axis1=1, axis2=2)
    diag_oasis = np.diagonal(oasis_channel, axis1=1, axis2=2)
    adni_h = reconstruct(out[:n_adni], tri, n_rois, diag_adni)
    oasis_h = reconstruct(out[n_adni:], tri, n_rois, diag_oasis)
    audit = {
        "n_features": int(combined.shape[1]),
        "n_variable_features": int(variable.sum()),
        "n_carried_features": int((~variable).sum()),
        "max_abs_diagonal_delta_adni": float(np.max(np.abs(np.diagonal(adni_h, axis1=1, axis2=2) - diag_adni))),
        "max_abs_diagonal_delta_oasis": float(np.max(np.abs(np.diagonal(oasis_h, axis1=1, axis2=2) - diag_oasis))),
        "max_abs_symmetry_error_adni": float(np.max(np.abs(adni_h - np.swapaxes(adni_h, 1, 2)))),
        "max_abs_symmetry_error_oasis": float(np.max(np.abs(oasis_h - np.swapaxes(oasis_h, 1, 2)))),
        "all_finite": bool(np.isfinite(adni_h).all() and np.isfinite(oasis_h).all()),
        "neurocombat_stdout": stdout.getvalue().strip().replace("\n", " | ")[:500],
    }
    return adni_h, oasis_h, audit


def combat_tensor(
    adni_tensor: np.ndarray,
    oasis_tensor: np.ndarray,
    adni_meta: pd.DataFrame,
    oasis_meta: pd.DataFrame,
    *,
    variant: Variant,
    selected_channel_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    adni_out = np.empty_like(adni_tensor, dtype=np.float64)
    oasis_out = np.empty_like(oasis_tensor, dtype=np.float64)
    rows = []
    for pos, channel_name in enumerate(selected_channel_names):
        a, o, audit = combat_channel(
            adni_tensor[:, pos],
            oasis_tensor[:, pos],
            adni_meta,
            oasis_meta,
            variant=variant,
        )
        adni_out[:, pos] = a
        oasis_out[:, pos] = o
        audit.update({"channel_position": pos, "channel_name": channel_name})
        rows.append(audit)
    return adni_out, oasis_out, rows


def flatten_features(tensor: np.ndarray) -> np.ndarray:
    parts = []
    for pos in range(tensor.shape[1]):
        x, _ = upper_features(tensor[:, pos])
        parts.append(x)
    return np.concatenate(parts, axis=1)


def gaussian_w2_pca(a: np.ndarray, b: np.ndarray) -> float:
    lwa = LedoitWolf(store_precision=False).fit(a)
    lwb = LedoitWolf(store_precision=False).fit(b)
    sa, sb = lwa.covariance_, lwb.covariance_
    sqrt_sa = sqrtm(sa).real
    sqrt_inner = sqrtm(sqrt_sa @ sb @ sqrt_sa).real
    mean_sq = float(np.sum((lwa.location_ - lwb.location_) ** 2))
    cov_term = float(np.trace(sa) + np.trace(sb) - 2.0 * np.trace(sqrt_inner))
    return float(np.sqrt(max(mean_sq + cov_term, 0.0)))


def feature_distances(
    raw_adni: np.ndarray,
    raw_oasis: np.ndarray,
    after_adni: np.ndarray,
    after_oasis: np.ndarray,
    *,
    arm: str,
    fold: int,
) -> list[dict[str, Any]]:
    rows = []
    for state, a_tensor, o_tensor in [
        ("before_raw", raw_adni, raw_oasis),
        ("after_external_batch_combat", after_adni, after_oasis),
    ]:
        a = flatten_features(a_tensor)
        o = flatten_features(o_tensor)
        scaler = StandardScaler().fit(a)
        za = scaler.transform(a)
        zo = scaler.transform(o)
        w1 = np.asarray([wasserstein_distance(za[:, j], zo[:, j]) for j in range(za.shape[1])])
        pca = PCA(n_components=10, random_state=42).fit(za)
        pa = pca.transform(za)
        po = pca.transform(zo)
        pca_w1 = np.asarray([wasserstein_distance(pa[:, j], po[:, j]) for j in range(pa.shape[1])])
        rows.append(
            {
                "arm": arm,
                "fold": fold,
                "state": state,
                "space": "connectivity_upper_offdiag_selected_channels",
                "n_adni": int(a.shape[0]),
                "n_oasis": int(o.shape[0]),
                "n_features": int(a.shape[1]),
                "w1_mean_all_features": float(w1.mean()),
                "w1_median_all_features": float(np.median(w1)),
                "pca10_w1_mean": float(pca_w1.mean()),
                "pca10_gaussian_w2": gaussian_w2_pca(pa, po),
            }
        )
    return rows


def load_global_tensor_selected(selected_names: list[str]) -> tuple[np.ndarray, list[str]]:
    with np.load(GLOBAL_TENSOR, allow_pickle=True) as zf:
        tensor = np.asarray(zf["global_tensor_data"], dtype=np.float32)
        names = np.asarray(zf["channel_names"]).astype(str).tolist()
    idx = selected_channel_indices(selected_names, names)
    return tensor[:, idx].astype(np.float32), names


def load_locked_train_latent(fold: int) -> pd.DataFrame:
    path = LOCKED_RUN / "classifier_only_readout/latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv"
    return ensure_y(pd.read_csv(path))


def load_locked_oasis_baseline() -> pd.DataFrame:
    df = pd.read_csv(LOCKED_OASIS_PANEL / "predictions.csv")
    out = df[
        df["candidate"].eq("promoted_beta3p75_oof_ecdf")
        & df["build_candidate"].eq("runwise164_pilot_parity")
        & df["prediction_level"].eq("ensemble_mean_score_majority_vote")
    ].copy()
    out["arm"] = "locked_frozen_transfer"
    return out


def load_previous_adni_fitted_combat_baseline() -> pd.DataFrame:
    df = pd.read_csv(CLEANREPRO_AUDIT / "cleanrepro_oasis_predictions.csv")
    out = df[df["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
    out["arm"] = "previous_adni_fitted_siemens_combat"
    return out


def ensemble_predictions(fold_preds: pd.DataFrame, arm: str, variant: Variant) -> pd.DataFrame:
    identity = [
        c
        for c in [
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
        if c in fold_preds.columns
    ]
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
    ens["arm"] = arm
    ens["label"] = variant.label
    ens["ref_batch"] = variant.ref_batch or "none"
    ens["prediction_level"] = "ensemble_mean_score_majority_vote"
    ens["fold"] = "ensemble_mean_score_majority_vote"
    ens["model_name"] = PRIMARY_MODEL
    ens["feature_set"] = PRIMARY_FEATURE_SET
    ens["calib_method"] = PRIMARY_CALIB
    ens["threshold_strategy"] = PRIMARY_THRESHOLD
    ens["y_pred"] = (ens["fold_positive_votes"] >= 3).astype(int)
    return ens


def score_variant(
    variant: Variant,
    cfg: dict[str, Any],
    global_selected: np.ndarray,
    oasis_selected: np.ndarray,
    oasis_meta: pd.DataFrame,
    selected_names: list[str],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec = CandidateSpec(
        label=variant.arm,
        role=variant.arm,
        run_dir=LOCKED_RUN,
        oof_dir=LOCKED_STAGEB,
    )
    fold_pred_rows = []
    latent_rows = []
    audit_rows = []
    distance_rows = []
    for fold in FOLDS:
        train_latent = load_locked_train_latent(fold)
        adni_idx = train_latent["tensor_idx"].to_numpy(dtype=int)
        adni_meta = train_latent[["SubjectID", "tensor_idx", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "y"]].copy()
        adni_x = global_selected[adni_idx].astype(np.float32)
        adni_h, oasis_h, audit = combat_tensor(
            adni_x,
            oasis_selected,
            adni_meta,
            oasis_meta,
            variant=variant,
            selected_channel_names=selected_names,
        )
        for row in audit:
            row.update(
                {
                    "arm": variant.arm,
                    "fold": fold,
                    "batch_variable": "Dataset",
                    "batch_levels": "ADNI;OASIS",
                    "ref_batch": variant.ref_batch or "none",
                    "preserved_covariates": "Age+Sex",
                    "excluded_covariates": "Diagnosis;ResearchGroup_Mapped",
                    "oasis_labels_used_in_harmonizer": False,
                    "n_adni_train_dev": len(adni_meta),
                    "n_oasis": len(oasis_meta),
                }
            )
            audit_rows.append(row)
        distance_rows.extend(
            feature_distances(adni_x, oasis_selected, adni_h.astype(np.float32), oasis_h.astype(np.float32), arm=variant.arm, fold=fold)
        )
        ext = encode_external_fold(
            spec,
            cfg,
            oasis_h.astype(np.float32),
            oasis_meta,
            fold,
            device,
            batch_size=64,
        )
        ext["ResearchGroup_Mapped"] = ext["ResearchGroup_Mapped"].map(normalize_dx)
        ext["Sex"] = ext["Sex"].map(normalize_sex)
        ext["Manufacturer"] = ext["Manufacturer"].map(normalize_mfr)
        ext["Age"] = pd.to_numeric(ext["Age"], errors="coerce")
        raw_score, ecdf_score, readout_meta = fit_adni_stageb_and_score_external(
            train_latent,
            ext,
            fold=fold,
            n_jobs=4,
            ecdf_mode="interp",
        )
        threshold = load_threshold_from_oof(spec, fold)
        pred = (ecdf_score >= threshold).astype(int)
        keep_cols = [
            c
            for c in [
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
            if c in ext.columns
        ]
        pp = ext[keep_cols].copy()
        pp["arm"] = variant.arm
        pp["label"] = variant.label
        pp["ref_batch"] = variant.ref_batch or "none"
        pp["methodological_status"] = variant.methodological_status
        pp["fold"] = fold
        pp["prediction_level"] = "fold_model"
        pp["model_name"] = PRIMARY_MODEL
        pp["feature_set"] = PRIMARY_FEATURE_SET
        pp["calib_method"] = PRIMARY_CALIB
        pp["threshold_strategy"] = PRIMARY_THRESHOLD
        pp["y_score_raw"] = raw_score
        pp["y_score"] = ecdf_score
        pp["threshold"] = threshold
        pp["y_pred"] = pred
        pp["readout_best_inner_auc"] = readout_meta["best_inner_auc"]
        pp["readout_best_params"] = readout_meta["best_params"]
        pp["readout_inner_cv_context"] = readout_meta["inner_cv_context"]
        pp["oasis_labels_used_for_readout_fit_or_calibration"] = False
        fold_pred_rows.append(pp)
        lat = ext[["SubjectID", "ResearchGroup_Mapped", "y", "Age", "Sex", "Manufacturer", *MU_COLS]].copy()
        lat.insert(0, "fold", fold)
        lat.insert(0, "arm", variant.arm)
        latent_rows.append(lat)
    folds = pd.concat(fold_pred_rows, ignore_index=True)
    ens = ensemble_predictions(folds, variant.arm, variant)
    all_preds = pd.concat([folds, ens], ignore_index=True, sort=False)
    latents = pd.concat(latent_rows, ignore_index=True)
    return all_preds, latents, pd.DataFrame(audit_rows), pd.DataFrame(distance_rows)


def metrics_for_predictions(predictions: pd.DataFrame, baselines: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for baseline in baselines:
        rows.append(
            {
                "arm": baseline["arm"].iloc[0],
                "label": baseline["arm"].iloc[0],
                "prediction_level": "ensemble_mean_score_majority_vote",
                "methodological_status": "existing_reference_result",
                **metric_dict(baseline["y"], baseline["y_score"], baseline["y_pred"]),
            }
        )
    for arm, sub in predictions[predictions["prediction_level"].eq("ensemble_mean_score_majority_vote")].groupby("arm"):
        row = {
            "arm": arm,
            "label": sub["label"].iloc[0],
            "prediction_level": "ensemble_mean_score_majority_vote",
            "methodological_status": sub["methodological_status"].iloc[0],
            "mean_threshold": float(sub["threshold"].mean()),
            "mean_fold_score_std": float(sub["fold_score_std"].mean()),
        }
        row.update(metric_dict(sub["y"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(predictions: pd.DataFrame, locked: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baselines = {
        "locked_frozen_transfer": locked,
        "previous_adni_fitted_siemens_combat": previous,
    }
    metrics = ["roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "brier_score"]
    rng = np.random.default_rng(BOOT_SEED)
    for arm, sub in predictions[predictions["prediction_level"].eq("ensemble_mean_score_majority_vote")].groupby("arm"):
        cand = sub[["SubjectID", "y", "y_score", "y_pred"]].rename(
            columns={"y": "y_candidate", "y_score": "score_candidate", "y_pred": "pred_candidate"}
        )
        for base_name, base in baselines.items():
            base_s = base[["SubjectID", "y", "y_score", "y_pred"]].rename(
                columns={"y": "y_baseline", "y_score": "score_baseline", "y_pred": "pred_baseline"}
            )
            paired = base_s.merge(cand, on="SubjectID", how="inner", validate="one_to_one")
            if len(paired) != len(base_s) or not paired["y_baseline"].equals(paired["y_candidate"]):
                raise RuntimeError(f"Pairing failed for {arm} vs {base_name}")
            y = paired["y_baseline"].to_numpy(int)
            idx0 = np.where(y == 0)[0]
            idx1 = np.where(y == 1)[0]
            base_m = metric_dict(y, paired["score_baseline"], paired["pred_baseline"])
            cand_m = metric_dict(y, paired["score_candidate"], paired["pred_candidate"])
            boot = {m: np.empty(BOOT_N, dtype=float) for m in metrics}
            for i in range(BOOT_N):
                idx = np.concatenate([rng.choice(idx0, len(idx0), replace=True), rng.choice(idx1, len(idx1), replace=True)])
                b = metric_dict(y[idx], paired["score_baseline"].to_numpy(float)[idx], paired["pred_baseline"].to_numpy(int)[idx])
                c = metric_dict(y[idx], paired["score_candidate"].to_numpy(float)[idx], paired["pred_candidate"].to_numpy(int)[idx])
                for m in metrics:
                    boot[m][i] = c[m] - b[m]
            for m in metrics:
                vals = boot[m]
                rows.append(
                    {
                        "comparison": f"{arm}_minus_{base_name}",
                        "arm": arm,
                        "baseline": base_name,
                        "metric": m,
                        "baseline_value": base_m[m],
                        "external_batch_combat_value": cand_m[m],
                        "observed_delta": cand_m[m] - base_m[m],
                        "bootstrap_mean_delta": float(vals.mean()),
                        "ci_low_2p5": float(np.quantile(vals, 0.025)),
                        "ci_high_97p5": float(np.quantile(vals, 0.975)),
                        "p_delta_gt_0": float(np.mean(vals > 0)),
                        "n_bootstrap": BOOT_N,
                        "seed": BOOT_SEED,
                    }
                )
    return pd.DataFrame(rows)


def write_protocol(audit: pd.DataFrame) -> None:
    supports_ref = "ref_batch" in str(inspect.signature(neuroCombat))
    text = f"""# OASIS external Dataset-batch ComBat protocol audit

## Library support

- `neuroCombat.neuroCombat` is installed and has signature `{inspect.signature(neuroCombat)}`.
- Reference-batch harmonisation support: **{supports_ref}** via `ref_batch`.
- `neuroHarmonize` is not required for this run and was not used.
- `neurocombat_sklearn.CombatModel` was not used because it does not expose a reference-batch argument in this environment.

## Data and leakage controls

- Batch variable for the new harmoniser: `Dataset`, levels `ADNI` and `OASIS`.
- Preserved covariates: `Age`, `Sex`.
- Excluded covariates: `Diagnosis`, `ResearchGroup_Mapped`.
- OASIS labels were not included in the ComBat covariate frame and were used only for final metric evaluation.
- No VAE checkpoint was trained or updated.
- No OASIS calibration, threshold selection, feature selection, or diagnostic model selection was performed.
- The downstream readout follows the existing locked OASIS scoring convention: ADNI train/dev-only readout reconstruction and locked inner-OOF ECDF threshold transfer. This is documented as a limitation because the official locked downstream readout estimators were not serialized as frozen objects.

## Variants

1. Dataset-level ComBat without reference batch.
2. Dataset-level ComBat with `ADNI` as reference batch.

## Fit audit summary

{audit[['arm','fold','channel_name','ref_batch','n_adni_train_dev','n_oasis','n_variable_features','all_finite','max_abs_diagonal_delta_adni','max_abs_diagonal_delta_oasis']].head(40).to_markdown(index=False)}
"""
    (OUT / "oasis_external_batch_combat_protocol_audit.md").write_text(text, encoding="utf-8")


def write_limitations(metrics: pd.DataFrame, wasserstein: pd.DataFrame) -> None:
    text = """# OASIS external Dataset-batch ComBat limitations

This analysis is transductive and post-hoc: OASIS connectivity features and Age/Sex are used to estimate an unsupervised Dataset-batch ComBat mapping. OASIS diagnosis labels are not used for harmonisation or any scoring-model operation, but the external feature distribution is observed during harmonisation.

Feature-space ComBat is applied before a VAE that was trained on unadapted ADNI connectivity matrices. The ADNI-reference variant is the methodologically cleaner feature-space variant because it targets the original ADNI training domain. The no-reference variant targets a pooled ADNI/OASIS space and is therefore exploratory for a frozen ADNI VAE.

The official locked OASIS scoring convention did not serialize fold-specific downstream diagnostic classifier estimators. To remain comparable to the published locked OASIS reference, this script reconstructs the downstream readout from ADNI train/dev latents only and applies the locked ECDF/threshold convention to OASIS. No OASIS labels enter that reconstruction. If a stricter interpretation requires already-serialized diagnostic readouts only, the current repository does not support a feature-space external-batch ComBat primary analysis without first freezing and versioning those locked readouts.

Do not interpret improved operating-point sensitivity/specificity alone as evidence of improved transport unless ROC-AUC/PR-AUC paired bootstrap intervals support it.
"""
    (OUT / "oasis_external_batch_combat_limitations.md").write_text(text, encoding="utf-8")


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing output directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    command_log = [
        {
            "timestamp_utc": utc_now(),
            "argv": sys.argv,
            "cwd": str(PROJECT),
            "python": sys.executable,
            "platform": platform.platform(),
            "guardrails": {
                "no_vae_training": True,
                "no_oasis_label_use_for_harmonization": True,
                "no_oasis_calibration": True,
                "no_threshold_selection_on_oasis": True,
                "no_overwrite": True,
            },
        }
    ]

    cfg = load_config(LOCKED_RUN)
    selected_names = list(cfg.get("selected_channel_names") or [])
    if selected_names != [
        "Pearson_Full_FisherZ_Signed",
        "Pearson_OMST_GCE_Signed_Weighted",
        "MI_KNN_Symmetric",
    ]:
        raise RuntimeError(f"Unexpected selected channels: {selected_names}")
    global_selected, _ = load_global_tensor_selected(selected_names)
    oasis_tensor, oasis_meta, oasis_names = load_mega_tensor(OASIS_TENSOR)
    oasis_idx = selected_channel_indices(selected_names, oasis_names)
    oasis_selected = oasis_tensor[:, oasis_idx].astype(np.float32)
    oasis_meta = oasis_meta.copy()
    oasis_meta["ResearchGroup_Mapped"] = oasis_meta["ResearchGroup_Mapped"].map(normalize_dx)
    oasis_meta["Sex"] = oasis_meta["Sex"].map(normalize_sex)
    oasis_meta["Manufacturer"] = oasis_meta["Manufacturer"].map(normalize_mfr)
    oasis_meta["Age"] = pd.to_numeric(oasis_meta["Age"], errors="raise")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_parts = []
    latent_parts = []
    audit_parts = []
    distance_parts = []
    for variant in VARIANTS:
        preds, latents, audit, distances = score_variant(
            variant,
            cfg,
            global_selected,
            oasis_selected,
            oasis_meta,
            selected_names,
            device,
        )
        pred_parts.append(preds)
        latent_parts.append(latents)
        audit_parts.append(audit)
        distance_parts.append(distances)

    predictions = pd.concat(pred_parts, ignore_index=True)
    latents = pd.concat(latent_parts, ignore_index=True)
    audit = pd.concat(audit_parts, ignore_index=True)
    wasserstein = pd.concat(distance_parts, ignore_index=True)
    locked = load_locked_oasis_baseline()
    previous = load_previous_adni_fitted_combat_baseline()
    metrics = metrics_for_predictions(predictions, [locked, previous])
    boot = paired_bootstrap(predictions, locked, previous)

    predictions.to_csv(OUT / "oasis_external_batch_combat_predictions.csv", index=False)
    latents.to_csv(OUT / "oasis_external_batch_combat_latents.csv", index=False)
    audit.to_csv(OUT / "oasis_external_batch_combat_fit_audit.csv", index=False)
    write_table("oasis_external_batch_combat_metrics", metrics, "OASIS external Dataset-batch ComBat metrics")
    write_table("oasis_external_batch_combat_paired_bootstrap", boot, "Paired subject bootstrap")
    write_table("oasis_external_batch_combat_wasserstein", wasserstein, "ADNI-OASIS Wasserstein before/after external-batch ComBat")
    write_protocol(audit)
    write_limitations(metrics, wasserstein)

    command_log.append(
        {
            "timestamp_utc": utc_now(),
            "status": "COMPLETE",
            "output_dir": str(OUT),
            "script_sha256": sha256_file(Path(__file__)),
            "oasis_subject_hash": sha256_values(oasis_meta["SubjectID"]),
            "n_oasis": int(len(oasis_meta)),
        }
    )
    (OUT / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
