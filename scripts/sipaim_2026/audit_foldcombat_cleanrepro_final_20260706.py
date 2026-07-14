#!/usr/bin/env python3
"""Definitive read-only audit for the clean fold-wise input-ComBat FULL run.

This script writes only audit outputs. It does not train a VAE, refit ComBat,
fit/calibrate on OASIS, edit manuscript files, or mutate completed run
artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
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
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT = Path("/home/diego/proyectos/vae_AD")
RESULTS = PROJECT / "results/revision_bspc_2026"
SCRIPT_DIR = PROJECT / "scripts/revision_bspc_2026"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from foldwise_combat_input_harmonization import transform_tensor_combat_channelwise  # noqa: E402
from run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep import load_config  # noqa: E402
from score_oasis_mega_90_90_external_inference_model_panel_20260604 import (  # noqa: E402
    CandidateSpec,
    encode_external_fold,
    ensure_y,
    load_mega_tensor,
    normalize_dx,
    normalize_mfr,
    normalize_sex,
    selected_channel_indices,
)


OUT = RESULTS / (
    "post_revision_exploratory_20260630/"
    "foldcombat_cleanrepro_final_audit_20260706"
)
CLEAN_RUN = RESULTS / (
    "recover035_latent384_beta3p75_foldcombat_mfr_age_sex_cleanrepro_20260706"
)
LOCKED_RUN = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
    "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
)
LOCKED_STAGEB = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
LOCKED_OASIS_PANEL = RESULTS / "oasis_mega_90_90_external_inference_model_panel_20260604"
LOCKED_OASIS_LATENT = RESULTS / (
    "promoted_latent384_oasis_vs_adni_latent_distance_audit_20260604/"
    "oasis_fold_latent_mu_runwise164.csv"
)
HIST_COMBAT_RUN = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
    "recover035_latent384_beta3p75_foldcombat_mfr_age_sex_T80_h10000_p560_full5x5"
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
LOCKED_FEATURE = "z_plus_age_sex"
CLEAN_FEATURE = "posterior_mu_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
BOOT_N = 10_000
BOOT_SEED = 20260706
MFR_SEED = 42
MFR_N_PERM = 1000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_values(values: Iterable[Any], sort_values: bool = True) -> str:
    vals = list(map(str, values))
    if sort_values:
        vals = sorted(vals)
    return hashlib.sha256("\n".join(vals).encode("utf-8")).hexdigest()


def write_csv(name: str, df: pd.DataFrame) -> None:
    df.to_csv(OUT / name, index=False)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows)
    tail = "" if len(df) <= max_rows else f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return view.to_markdown(index=False) + tail


def as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "1", "yes", "pass"}


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = obj
    return out


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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_clean_oof() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_RUN / "downstream_diagnostic_classifier/downstream_classifier_oof_predictions.csv")
    df = df.rename(columns={"y_true": "y"})
    df["arm"] = "clean_foldwise_input_combat"
    return df


def load_locked_oof() -> pd.DataFrame:
    df = pd.read_csv(LOCKED_STAGEB / "calib_predictions.csv")
    mask = (
        df["model_name"].eq(PRIMARY_MODEL)
        & df["feature_set"].eq(LOCKED_FEATURE)
        & df["calib_method"].eq(PRIMARY_CALIB)
        & df["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    )
    out = df[mask].copy()
    if len(out) != 397:
        raise RuntimeError(f"Locked primary OOF row count is {len(out)}, expected 397")
    out["arm"] = "locked"
    out = out.rename(columns={"y_true": "y"})
    return out


def pooled_metrics_from_predictions(df: pd.DataFrame, arm: str) -> dict[str, Any]:
    row = {"arm": arm, "scope": "pooled_outer_fold_predictions"}
    row.update(metric_dict(df["y"], df["y_score"], df["y_pred"]))
    return row


def completion_and_lineage_audit() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check: str, status: bool, evidence: str, fold: int | str = "all") -> None:
        rows.append({"fold": fold, "check": check, "status": "PASS" if status else "FAIL", "evidence": evidence})

    exit_path = CLEAN_RUN / "training_exit_code.txt"
    add("training_exit_code_0", exit_path.exists() and exit_path.read_text().strip() == "0", str(exit_path))
    add("clean_completion_marker_exists", (CLEAN_RUN / "cleanrepro_completed_utc.txt").exists(), str(CLEAN_RUN / "cleanrepro_completed_utc.txt"))
    add("prepared_marker_exists", (CLEAN_RUN / ".cleanrepro_prepared").exists(), str(CLEAN_RUN / ".cleanrepro_prepared"))
    run_cfg = json.load(open(CLEAN_RUN / "run_config.json"))
    resolved_cfg = json.load(open(CLEAN_RUN / "provenance/resolved_config.json"))
    param_mismatch = []
    for key, value in resolved_cfg.get("parameters", {}).items():
        if run_cfg.get("args", {}).get(key) != value:
            param_mismatch.append(key)
    path_mismatch = []
    for key in ["global_tensor_path", "metadata_path", "output_dir"]:
        expected = resolved_cfg.get("resolved_paths", {}).get(key)
        observed = run_cfg.get("args", {}).get(key) or run_cfg.get(key)
        if expected and str(Path(observed).resolve()) != str(Path(expected).resolve()):
            path_mismatch.append(key)
    channel_match = (
        run_cfg.get("channel_names_selected") == resolved_cfg.get("selected_channel_names")
        and run_cfg.get("channel_names_master_in_tensor_order")
        == resolved_cfg.get("channel_names_master_in_tensor_order")
    )
    add(
        "run_config_matches_resolved_launch_config",
        not param_mismatch and not path_mismatch and channel_match,
        "compared resolved parameters, selected channels, and resolved paths; "
        f"param_mismatch={param_mismatch}; path_mismatch={path_mismatch}; "
        f"channel_match={channel_match}",
    )
    for rel in [
        "provenance/live_source_sha256.txt",
        "provenance/source_snapshot_sha256.txt",
        "provenance/git_status_porcelain.txt",
        "provenance/git_head.txt",
        "provenance/executed_training_command.txt",
        "provenance/executed_downstream_command.txt",
    ]:
        add(f"provenance_{Path(rel).name}_exists", (CLEAN_RUN / rel).exists(), str(CLEAN_RUN / rel))

    for fold in FOLDS:
        fdir = CLEAN_RUN / f"fold_{fold}"
        ckpt = fdir / f"vae_model_fold_{fold}.pt"
        try:
            loaded = isinstance(torch.load(ckpt, map_location="cpu"), dict)
        except Exception as e:  # noqa: BLE001
            loaded = False
            evidence = f"{ckpt}: {e}"
        else:
            evidence = str(ckpt)
        add("checkpoint_exists_and_loads", ckpt.exists() and loaded, evidence, fold)
        hist = fdir / f"vae_train_history_fold_{fold}.joblib"
        try:
            h = joblib.load(hist)
            complete = all(k in h for k in ["train_recon", "train_kld", "val_loss_modelsel", "beta", "learning_rate_start", "learning_rate_end"])
        except Exception as e:  # noqa: BLE001
            complete = False
            evidence = f"{hist}: {e}"
        else:
            evidence = f"{hist}; n_epochs={len(h.get('val_loss_modelsel', []))}"
        add("complete_training_history_exists", hist.exists() and complete, evidence, fold)
        for rel in [
            "vae_norm_params.joblib",
            "train_dev_subjects_fold.csv",
            "test_subjects_fold.csv",
            "input_harmonization_fit_audit.csv",
            "input_harmonization_leakage_guard.csv",
            "input_harmonization_leakage_guard.json",
            "input_harmonization_integrity.csv",
            "input_harmonization_tensor_sample_manifest.csv",
            "input_harmonization_fitted_objects/foldwise_combat_tensor.joblib",
            "input_harmonization_fitted_objects/fitted_combat_object_manifest.csv",
        ]:
            add(f"{Path(rel).name}_exists", (fdir / rel).exists(), str(fdir / rel), fold)
        manifest = pd.read_csv(fdir / "input_harmonization_fitted_objects/fitted_combat_object_manifest.csv")
        single = manifest[manifest["scope"].eq("single_channel")]
        add("fitted_combat_objects_for_three_selected_channels", len(single) == 3 and single["path"].map(lambda p: Path(p).exists()).all(), ";".join(single["path"].astype(str)), fold)
        for split in ["trainDev", "test"]:
            cache = CLEAN_RUN / "downstream_diagnostic_classifier/latent_cache" / f"fold_{fold}_{split}_latent_mu.csv"
            lineage = cache.with_name(cache.stem + "_lineage.json")
            try:
                lin = json.load(open(lineage))
                cache_ok = sha256_file(cache) == lin["cache_sha256"]
                lineage_ok = (
                    lin.get("input_lineage") == "foldwise_combat_harmonized_then_fold_normalized"
                    and lin.get("raw_input_cache_reused") is False
                    and lin.get("combat_fit_scope") == "outer_train_dev_only"
                )
            except Exception as e:  # noqa: BLE001
                cache_ok = lineage_ok = False
                evidence = f"{cache}: {e}"
            else:
                evidence = f"{cache}; {lineage}"
            add(f"{split}_harmonized_latent_cache_and_lineage", cache.exists() and lineage.exists() and cache_ok and lineage_ok, evidence, fold)
    return pd.DataFrame(rows)


def config_diff_and_subject_parity() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean = json.load(open(CLEAN_RUN / "run_config.json"))
    locked = json.load(open(LOCKED_RUN / "run_config.json"))
    cf = flatten(clean)
    lf = flatten(locked)
    path_tokens = ["output_dir", "big_disk_output_dir", "run_name", "description", "created_utc", "git_hash", "cuda_available", "python_version", "torch_version"]
    harm_tokens = [
        "input_harmonization_mode",
        "input_harmonization_batch_col",
        "input_harmonization_covariates",
        "input_harmonization_excluded_covariates",
        "input_harmonization_fit_scope",
        "input_harmonization_vectorization",
    ]
    rows = []
    for key in sorted(set(cf) | set(lf)):
        if cf.get(key) == lf.get(key):
            continue
        intended = any(tok in key for tok in harm_tokens)
        pathprov = any(tok in key for tok in path_tokens) or key.endswith("global_tensor_path") or key.endswith("metadata_path")
        rows.append(
            {
                "field": key,
                "locked_value": json.dumps(lf.get(key), sort_keys=True, default=str),
                "clean_value": json.dumps(cf.get(key), sort_keys=True, default=str),
                "difference_class": "intended_input_harmonization" if intended else ("output_or_provenance_path" if pathprov else "unexpected"),
            }
        )
    diff = pd.DataFrame(rows)

    args = clean["args"]
    protocol = []
    expected = {
        "channels_[1,0,2]": args.get("channels_to_use") == [1, 0, 2],
        "latent_dim_384": int(args.get("latent_dim")) == 384,
        "beta_3p75": math.isclose(float(args.get("beta_vae")), 3.75),
        "batch_size_64": int(args.get("batch_size")) == 64,
        "T0_80": int(args.get("lr_scheduler_T0")) == 80,
        "horizon_10000": int(args.get("epochs_vae")) == 10000,
        "patience_560": int(args.get("early_stopping_patience_vae")) == 560,
        "outer_folds_5": int(args.get("outer_folds")) == 5,
        "outer_repeats_1": int(args.get("repeated_outer_folds_n_repeats")) == 1,
        "inner_folds_5": int(args.get("inner_folds")) == 5,
        "seed_42": int(args.get("seed")) == 42,
        "classifier_features_age_sex": args.get("metadata_features") == ["Age", "Sex"],
        "architecture_and_loss_match": all(
            clean["args"].get(k) == locked["args"].get(k)
            for k in [
                "decoder_type",
                "num_conv_layers_encoder",
                "intermediate_fc_dim_vae",
                "recon_loss_mode",
                "vae_block_order",
                "vae_encoder_norm_mode",
                "vae_final_activation",
                "use_layernorm_vae_fc",
                "dropout_rate_vae",
                "vae_channel_dropout_p",
            ]
            if k in locked["args"] or k in clean["args"]
        ),
    }
    for check, ok in expected.items():
        protocol.append({"check": check, "status": "PASS" if ok else "FAIL", "clean_value": str(args)})
    protocol_df = pd.DataFrame(protocol)

    parity_rows = []
    for fold in FOLDS:
        c = pd.read_csv(CLEAN_RUN / f"fold_{fold}/test_subjects_fold.csv")
        l = pd.read_csv(LOCKED_RUN / f"fold_{fold}/test_subjects_fold.csv")
        parity_rows.append(
            {
                "fold": fold,
                "clean_n": len(c),
                "locked_n": len(l),
                "clean_subject_hash_sorted": sha256_values(c["SubjectID"]),
                "locked_subject_hash_sorted": sha256_values(l["SubjectID"]),
                "same_subject_set": set(c["SubjectID"]) == set(l["SubjectID"]),
                "same_subject_order": c["SubjectID"].astype(str).tolist() == l["SubjectID"].astype(str).tolist(),
            }
        )
    return diff, pd.DataFrame(parity_rows), protocol_df


def fold_safety() -> tuple[pd.DataFrame, str]:
    rows = []
    for fold in FOLDS:
        guard = pd.read_csv(CLEAN_RUN / f"fold_{fold}/input_harmonization_leakage_guard.csv").iloc[0]
        integrity = pd.read_csv(CLEAN_RUN / f"fold_{fold}/input_harmonization_integrity.csv").iloc[0]
        fit_audit = pd.read_csv(CLEAN_RUN / f"fold_{fold}/input_harmonization_fit_audit.csv")
        sample_manifest = pd.read_csv(CLEAN_RUN / f"fold_{fold}/input_harmonization_tensor_sample_manifest.csv")
        norm_mtime = os.path.getmtime(CLEAN_RUN / f"fold_{fold}/vae_norm_params.joblib")
        fit_mtime = os.path.getmtime(CLEAN_RUN / f"fold_{fold}/input_harmonization_fitted_objects/foldwise_combat_tensor.joblib")
        checks = {
            "fit_scope_outer_train_dev_only": guard["fit_scope"] == "outer_train_dev_only",
            "fit_test_subject_overlap_zero": int(guard["n_fit_test_subject_overlap"]) == 0,
            "batch_is_manufacturer": guard["batch_col"] == "Manufacturer" and set(fit_audit["batch"].astype(str)) == {"Manufacturer"},
            "age_sex_preserved": guard["covariates_preserved"] == "Age+Sex" and set(fit_audit["protected_covariates"].astype(str)) == {"Age+Sex"},
            "diagnosis_excluded": "ResearchGroup_Mapped" in str(guard["excluded_covariates"]) and as_bool(guard["diagnosis_used_in_harmonizer"]) is False,
            "outer_test_distribution_not_used": as_bool(integrity["test_distribution_used_in_fit"]) is False,
            "oasis_not_used_for_fit": as_bool(guard["oasis_used"]) is False,
            "finite_transformed_matrices": float(integrity["finite_fraction_vae_pool_post"]) == 1.0 and sample_manifest["all_values_finite"].map(as_bool).all(),
            "symmetric_transformed_matrices": float(integrity["max_abs_symmetry_error_vae_pool"]) == 0.0,
            "diagonal_values_preserved": float(integrity["max_abs_diagonal_delta_vae_pool"]) == 0.0,
            "normalization_after_harmonization": norm_mtime >= fit_mtime,
        }
        fold_status = "FOLD_SAFE" if all(checks.values()) else "LEAKY"
        row = {"fold": fold, "verdict": fold_status}
        row.update({k: "PASS" if v else "FAIL" for k, v in checks.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    verdict = "FOLD_SAFE" if df.drop(columns=["fold", "verdict"]).eq("PASS").all().all() else "LEAKY"
    return df, verdict


def summarize_history() -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        clean_h = joblib.load(CLEAN_RUN / f"fold_{fold}/vae_train_history_fold_{fold}.joblib")
        hist_h = joblib.load(HIST_COMBAT_RUN / f"fold_{fold}/vae_train_history_fold_{fold}.joblib") if (HIST_COMBAT_RUN / f"fold_{fold}/vae_train_history_fold_{fold}.joblib").exists() else {}
        clean_idx = int(clean_h.get("selected_checkpoint_epoch", np.argmin(clean_h["val_loss_modelsel"]) + 1)) - 1
        hist_idx = int(np.argmin(hist_h["val_loss_modelsel"])) if hist_h else -1
        train_mu = pd.read_csv(CLEAN_RUN / "downstream_diagnostic_classifier/latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv")
        active_units = int((train_mu[MU_COLS].var(axis=0).to_numpy() > 1e-4).sum())
        rows.append(
            {
                "fold": fold,
                "selected_checkpoint_epoch": clean_idx + 1,
                "terminal_epoch": int(clean_h.get("terminal_epoch", len(clean_h["val_loss_modelsel"]))),
                "selected_validation_objective": float(clean_h["val_loss_modelsel"][clean_idx]),
                "selected_train_reconstruction": float(clean_h["train_recon"][clean_idx]),
                "selected_train_kl": float(clean_h["train_kld"][clean_idx]),
                "selected_val_reconstruction": float(clean_h["val_recon"][clean_idx]),
                "selected_val_kl": float(clean_h["val_kld"][clean_idx]),
                "selected_val_beta_kld_over_recon": float(clean_h["val_beta_kld_over_recon"][clean_idx]),
                "active_latent_units_var_gt_1e_minus_4": active_units,
                "learning_rate_start_first": float(clean_h["learning_rate_start"][0]),
                "learning_rate_start_selected": float(clean_h["learning_rate_start"][clean_idx]),
                "learning_rate_end_last": float(clean_h["learning_rate_end"][-1]),
                "historical_selected_checkpoint_epoch": hist_idx + 1 if hist_h else np.nan,
                "historical_terminal_epoch": len(hist_h["val_loss_modelsel"]) if hist_h else np.nan,
                "historical_selected_validation_objective": float(hist_h["val_loss_modelsel"][hist_idx]) if hist_h else np.nan,
                "selected_epoch_reproduced": bool(hist_h and hist_idx == clean_idx),
                "terminal_epoch_reproduced": bool(hist_h and len(hist_h["val_loss_modelsel"]) == len(clean_h["val_loss_modelsel"])),
                "validation_loss_reproduced_allclose": bool(hist_h and np.allclose(hist_h["val_loss_modelsel"], clean_h["val_loss_modelsel"], rtol=0, atol=1e-9)),
            }
        )
    return pd.DataFrame(rows)


def downstream_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = load_clean_oof()
    pooled = pd.DataFrame([pooled_metrics_from_predictions(clean, "clean_foldwise_input_combat")])
    foldwise = pd.read_csv(CLEAN_RUN / "downstream_diagnostic_classifier/downstream_classifier_foldwise_metrics.csv")
    return foldwise, pooled


def paired_bootstrap(clean: pd.DataFrame, locked: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    a = locked[["SubjectID", "y", "y_score", "y_pred"]].rename(columns={"y": "y_locked", "y_score": "score_locked", "y_pred": "pred_locked"})
    b = clean[["SubjectID", "y", "y_score", "y_pred"]].rename(columns={"y": "y_clean", "y_score": "score_clean", "y_pred": "pred_clean"})
    paired = a.merge(b, on="SubjectID", how="inner", validate="one_to_one")
    if len(paired) != len(a) or not paired["y_locked"].equals(paired["y_clean"]):
        raise RuntimeError(f"{prefix} paired subjects/labels mismatch: {len(paired)} vs {len(a)}")
    y = paired["y_locked"].to_numpy(int)
    observed_locked = metric_dict(y, paired["score_locked"], paired["pred_locked"])
    observed_clean = metric_dict(y, paired["score_clean"], paired["pred_clean"])
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng = np.random.default_rng(BOOT_SEED)
    metrics = ["roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "brier_score"]
    values = {m: np.empty(BOOT_N) for m in metrics}
    for i in range(BOOT_N):
        idx = np.concatenate([rng.choice(idx0, len(idx0), replace=True), rng.choice(idx1, len(idx1), replace=True)])
        yl = y[idx]
        lm = metric_dict(yl, paired["score_locked"].to_numpy(float)[idx], paired["pred_locked"].to_numpy(int)[idx])
        cm = metric_dict(yl, paired["score_clean"].to_numpy(float)[idx], paired["pred_clean"].to_numpy(int)[idx])
        for m in metrics:
            values[m][i] = cm[m] - lm[m]
    return pd.DataFrame(
        [
            {
                "metric": m,
                "comparison": "clean_foldwise_input_combat_minus_locked",
                "locked": observed_locked[m],
                "clean_foldwise_input_combat": observed_clean[m],
                "observed_delta": observed_clean[m] - observed_locked[m],
                "bootstrap_mean_delta": float(values[m].mean()),
                "ci_low_2p5": float(np.quantile(values[m], 0.025)),
                "ci_high_97p5": float(np.quantile(values[m], 0.975)),
                "p_delta_gt_0": float(np.mean(values[m] > 0)),
                "n_bootstrap": BOOT_N,
                "seed": BOOT_SEED,
            }
            for m in metrics
        ]
    )


def manufacturer_transport(clean: pd.DataFrame, locked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, df in [("locked", locked), ("clean_foldwise_input_combat", clean)]:
        for mfr, sub in df.groupby("Manufacturer"):
            y = sub["y"].to_numpy(int)
            pred = sub["y_pred"].to_numpy(int)
            score = sub["y_score"].to_numpy(float)
            tn = int(((y == 0) & (pred == 0)).sum())
            fp = int(((y == 0) & (pred == 1)).sum())
            fn = int(((y == 1) & (pred == 0)).sum())
            tp = int(((y == 1) & (pred == 1)).sum())
            rows.append(
                {
                    "arm": arm,
                    "manufacturer": mfr,
                    "n": int(len(sub)),
                    "n_cn": int((y == 0).sum()),
                    "n_ad": int((y == 1).sum()),
                    "cn_false_positive_rate": float(fp / (fp + tn)) if fp + tn else np.nan,
                    "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
                    "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
                    "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
                    "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                }
            )
    return pd.DataFrame(rows)


def load_latent(run_or_cache: Path, fold: int, split: str, clean: bool = False) -> pd.DataFrame:
    if clean:
        path = run_or_cache / "downstream_diagnostic_classifier/latent_cache" / f"fold_{fold}_{split}_latent_mu.csv"
    else:
        path = run_or_cache / "classifier_only_readout/latent_cache" / f"fold_{fold}_{split}_latent_mu.csv"
    df = pd.read_csv(path)
    df["SiteCode"] = df["SubjectID"].astype(str).str.extract(r"^(\d{3})_S_\d{4}$", expand=False)
    return ensure_y(df)


def residualized_mu(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    z_train = scaler.fit_transform(train[MU_COLS].to_numpy(float))
    z_test = scaler.transform(test[MU_COLS].to_numpy(float))
    age_mean = float(train["Age"].mean())
    age_sd = float(train["Age"].std())
    n_train = np.column_stack([train["y"], (train["Age"] - age_mean) / (age_sd + 1e-8), train["Sex"].eq("M").astype(float)])
    n_test = np.column_stack([test["y"], (test["Age"] - age_mean) / (age_sd + 1e-8), test["Sex"].eq("M").astype(float)])
    reg = LinearRegression().fit(n_train, z_train)
    return z_train - reg.predict(n_train), z_test - reg.predict(n_test)


def manufacturer_decoding() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(MFR_SEED)
    for fold in FOLDS:
        for arm, run, clean_flag in [("locked", LOCKED_RUN, False), ("clean_foldwise_input_combat", CLEAN_RUN, True)]:
            train = load_latent(run, fold, "trainDev", clean_flag).sort_values("SubjectID").reset_index(drop=True)
            test = load_latent(run, fold, "test", clean_flag).sort_values("SubjectID").reset_index(drop=True)
            xtr, xte = residualized_mu(train, test)
            enc = LabelEncoder().fit(train["Manufacturer"])
            ytr = enc.transform(train["Manufacturer"])
            yte = enc.transform(test["Manufacturer"])
            clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean").fit(xtr, ytr)
            pred = clf.predict(xte)
            bacc = float(balanced_accuracy_score(yte, pred))
            macro_f1 = float(f1_score(yte, pred, average="macro", zero_division=0))
            null = np.asarray([balanced_accuracy_score(yte[rng.permutation(len(yte))], pred) for _ in range(MFR_N_PERM)])
            rows.append(
                {
                    "arm": arm,
                    "fold": fold,
                    "manufacturer_balanced_accuracy": bacc,
                    "macro_f1": macro_f1,
                    "permutation_null_mean_bacc": float(null.mean()),
                    "permutation_null_sd_bacc": float(null.std(ddof=1)),
                    "permutation_p_ge_observed": float((np.sum(null >= bacc) + 1) / (MFR_N_PERM + 1)),
                    "n_permutations": MFR_N_PERM,
                    "seed": MFR_SEED,
                }
            )
    df = pd.DataFrame(rows)
    locked = df[df.arm.eq("locked")].set_index("fold")
    clean = df[df.arm.eq("clean_foldwise_input_combat")].set_index("fold")
    for fold in FOLDS:
        rows.append(
            {
                "arm": "clean_minus_locked",
                "fold": fold,
                "manufacturer_balanced_accuracy": float(clean.loc[fold, "manufacturer_balanced_accuracy"] - locked.loc[fold, "manufacturer_balanced_accuracy"]),
                "macro_f1": float(clean.loc[fold, "macro_f1"] - locked.loc[fold, "macro_f1"]),
                "permutation_null_mean_bacc": np.nan,
                "permutation_null_sd_bacc": np.nan,
                "permutation_p_ge_observed": np.nan,
                "n_permutations": MFR_N_PERM,
                "seed": MFR_SEED,
            }
        )
    return pd.DataFrame(rows)


def ecdf_score_from_frozen(model_obj: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = frame[model_obj["feature_columns"]].copy()
    raw = model_obj["estimator"].predict_proba(x)[:, 1]
    sorted_oof = np.asarray(model_obj["inner_oof_raw_scores_sorted"], dtype=float)
    pct = np.asarray(model_obj["inner_oof_ecdf_percentiles"], dtype=float)
    score = np.interp(raw, sorted_oof, pct, left=0.0, right=1.0)
    return raw, score


def clean_oasis_inference() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = load_config(CLEAN_RUN)
    with np.load(GLOBAL_TENSOR, allow_pickle=True) as zf:
        global_names = np.asarray(zf["channel_names"]).astype(str).tolist()
    oasis_tensor, oasis_meta, oasis_names = load_mega_tensor(OASIS_TENSOR)
    selected_names = list(cfg.get("selected_channel_names") or [])
    idx = selected_channel_indices(selected_names, oasis_names)
    x_oasis = oasis_tensor[:, idx].astype(np.float32)
    spec = CandidateSpec("clean_foldwise_input_combat", "clean_foldwise_input_combat", CLEAN_RUN, CLEAN_RUN / "downstream_diagnostic_classifier")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_pred_rows = []
    latent_rows = []
    for fold in FOLDS:
        fitted = joblib.load(CLEAN_RUN / f"fold_{fold}/input_harmonization_fitted_objects/foldwise_combat_tensor.joblib")
        harmonized = transform_tensor_combat_channelwise(fitted, x_oasis, oasis_meta).astype(np.float32)
        ext = encode_external_fold(spec, cfg, harmonized, oasis_meta, fold, device, batch_size=64)
        ext["ResearchGroup_Mapped"] = ext["ResearchGroup_Mapped"].map(normalize_dx)
        ext["Sex"] = ext["Sex"].map(normalize_sex)
        ext["Manufacturer"] = ext["Manufacturer"].map(normalize_mfr)
        ext["Age"] = pd.to_numeric(ext["Age"], errors="coerce")
        model_obj = joblib.load(CLEAN_RUN / "downstream_diagnostic_classifier/frozen_classifiers" / f"fold_{fold}_downstream_diagnostic_classifier.joblib")
        raw, score = ecdf_score_from_frozen(model_obj, ext)
        pred = (score >= float(model_obj["threshold"])).astype(int)
        keep = [c for c in ["SubjectID", "subject_id", "session_id", "experiment_id", "source_batch", "protocol_subset", "diagnosis", "ResearchGroup_Mapped", "Age", "Sex", "Manufacturer", "ScannerModel", "selected_qc_runs", "selected_run_ids", "mean_fd_subject", "max_fd_subject", "y"] if c in ext.columns]
        pp = ext[keep].copy()
        pp["arm"] = "clean_foldwise_input_combat"
        pp["fold"] = fold
        pp["prediction_level"] = "fold_model"
        pp["model_name"] = PRIMARY_MODEL
        pp["feature_set"] = CLEAN_FEATURE
        pp["calib_method"] = PRIMARY_CALIB
        pp["threshold_strategy"] = PRIMARY_THRESHOLD
        pp["y_score_raw"] = raw
        pp["y_score"] = score
        pp["threshold"] = float(model_obj["threshold"])
        pp["y_pred"] = pred
        fold_pred_rows.append(pp)
        lat = ext[["SubjectID", "ResearchGroup_Mapped", "y", "Age", "Sex", "Manufacturer", *MU_COLS]].copy()
        lat.insert(0, "fold", fold)
        latent_rows.append(lat)

    folds = pd.concat(fold_pred_rows, ignore_index=True)
    identity = [c for c in ["SubjectID", "subject_id", "session_id", "experiment_id", "source_batch", "protocol_subset", "diagnosis", "ResearchGroup_Mapped", "Age", "Sex", "Manufacturer", "ScannerModel", "selected_qc_runs", "selected_run_ids", "mean_fd_subject", "max_fd_subject", "y"] if c in folds.columns]
    ens = folds.groupby(identity, dropna=False).agg(
        y_score_raw=("y_score_raw", "mean"),
        y_score=("y_score", "mean"),
        threshold=("threshold", "mean"),
        fold_score_std=("y_score", "std"),
        fold_score_min=("y_score", "min"),
        fold_score_max=("y_score", "max"),
        fold_positive_votes=("y_pred", "sum"),
    ).reset_index()
    ens["arm"] = "clean_foldwise_input_combat"
    ens["fold"] = "ensemble_mean_score_majority_vote"
    ens["prediction_level"] = "ensemble_mean_score_majority_vote"
    ens["model_name"] = PRIMARY_MODEL
    ens["feature_set"] = CLEAN_FEATURE
    ens["calib_method"] = PRIMARY_CALIB
    ens["threshold_strategy"] = PRIMARY_THRESHOLD
    ens["y_pred"] = (ens["fold_positive_votes"] >= 3).astype(int)
    all_preds = pd.concat([folds, ens], ignore_index=True, sort=False)
    latents = pd.concat(latent_rows, ignore_index=True)
    locked_all = pd.read_csv(LOCKED_OASIS_PANEL / "predictions.csv")
    locked = locked_all[
        locked_all["candidate"].eq("promoted_beta3p75_oof_ecdf")
        & locked_all["build_candidate"].eq("runwise164_pilot_parity")
        & locked_all["prediction_level"].eq("ensemble_mean_score_majority_vote")
    ].copy()
    clean = ens.copy()
    metrics = pd.DataFrame([
        {"arm": "locked", **metric_dict(locked["y"], locked["y_score"], locked["y_pred"])},
        {"arm": "clean_foldwise_input_combat", **metric_dict(clean["y"], clean["y_score"], clean["y_pred"])},
    ])
    return all_preds, latents, metrics


def oasis_bootstrap(clean_all: pd.DataFrame) -> pd.DataFrame:
    locked_all = pd.read_csv(LOCKED_OASIS_PANEL / "predictions.csv")
    locked = locked_all[
        locked_all["candidate"].eq("promoted_beta3p75_oof_ecdf")
        & locked_all["build_candidate"].eq("runwise164_pilot_parity")
        & locked_all["prediction_level"].eq("ensemble_mean_score_majority_vote")
    ].copy()
    clean = clean_all[clean_all["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
    return paired_bootstrap(clean.rename(columns={"y": "y"}), locked.rename(columns={"y": "y"}), prefix="oasis")


def w1_per_dim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray([wasserstein_distance(a[:, j], b[:, j]) for j in range(a.shape[1])], dtype=float)


def gaussian_w2(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    lwa = LedoitWolf(store_precision=False).fit(a)
    lwb = LedoitWolf(store_precision=False).fit(b)
    sa, sb = lwa.covariance_, lwb.covariance_
    sqrt_sa = sqrtm(sa).real
    sqrt_inner = sqrtm(sqrt_sa @ sb @ sqrt_sa).real
    mean_sq = float(np.sum((lwa.location_ - lwb.location_) ** 2))
    cov_term = float(np.trace(sa) + np.trace(sb) - 2 * np.trace(sqrt_inner))
    return float(np.sqrt(max(mean_sq + cov_term, 0.0))), float(lwa.shrinkage_), float(lwb.shrinkage_)


def wasserstein(clean_oasis_latents: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    locked_oasis = pd.read_csv(LOCKED_OASIS_LATENT)
    rows = []
    for arm, source, oasis, clean_flag in [
        ("locked", LOCKED_RUN, locked_oasis, False),
        ("clean_foldwise_input_combat", CLEAN_RUN, clean_oasis_latents, True),
    ]:
        for fold in FOLDS:
            adni = load_latent(source, fold, "trainDev", clean_flag)
            oo = oasis[oasis["fold"].astype(int).eq(fold)].copy()
            scaler = StandardScaler()
            z_adni = scaler.fit_transform(adni[MU_COLS].to_numpy(float))
            z_oasis = scaler.transform(oo[MU_COLS].to_numpy(float))
            reg = LinearRegression().fit(adni[["y"]].to_numpy(float), z_adni)
            za = z_adni - reg.predict(adni[["y"]].to_numpy(float))
            zo = z_oasis - reg.predict(oo[["y"]].to_numpy(float))
            pca = PCA(n_components=10, random_state=42).fit(za)
            pa, po = pca.transform(za), pca.transform(zo)
            pca_w1 = np.asarray([wasserstein_distance(pa[:, j], po[:, j]) for j in range(10)])
            gw2, sa, so = gaussian_w2(za, zo)
            per_dim = w1_per_dim(za, zo)
            rows.append(
                {
                    "arm": arm,
                    "fold": fold,
                    "n_adni_trainDev": len(adni),
                    "n_oasis": len(oo),
                    "residualization": "diagnosis_y_only_OLS_fit_ADNI_trainDev",
                    "w1_mean_all_dims": float(per_dim.mean()),
                    "pca10_w1_mean": float(pca_w1.mean()),
                    "gaussian_w2": gw2,
                    "lw_shrinkage_adni": sa,
                    "lw_shrinkage_oasis": so,
                }
            )
    by_fold = pd.DataFrame(rows)
    locked = by_fold[by_fold.arm.eq("locked")].set_index("fold")
    clean = by_fold[by_fold.arm.eq("clean_foldwise_input_combat")].set_index("fold")
    summary_rows = []
    for metric in ["w1_mean_all_dims", "pca10_w1_mean", "gaussian_w2"]:
        deltas = []
        for fold in FOLDS:
            deltas.append(float(clean.loc[fold, metric] - locked.loc[fold, metric]))
        summary_rows.append(
            {
                "metric": metric,
                "locked_mean": float(locked[metric].mean()),
                "clean_mean": float(clean[metric].mean()),
                "mean_delta_clean_minus_locked": float(np.mean(deltas)),
                "sd_delta": float(np.std(deltas, ddof=1)),
                "folds_closer_under_clean": int(np.sum(np.asarray(deltas) < 0)),
                "folds_total": len(FOLDS),
                "consistent_improvement_all_folds": bool(np.all(np.asarray(deltas) < 0)),
            }
        )
    return by_fold, pd.DataFrame(summary_rows)


def write_final_report(
    completion: pd.DataFrame,
    protocol: pd.DataFrame,
    safety_verdict: str,
    train_repro: pd.DataFrame,
    pooled: pd.DataFrame,
    boot: pd.DataFrame,
    mfr_dec: pd.DataFrame,
    transport: pd.DataFrame,
    oasis_metrics: pd.DataFrame,
    oasis_boot: pd.DataFrame,
    wass_summary: pd.DataFrame,
) -> None:
    cp = pooled.iloc[0]
    locked_oof = load_locked_oof()
    lp = pooled_metrics_from_predictions(locked_oof, "locked")
    auc_delta = boot[boot.metric.eq("roc_auc")].iloc[0]
    pr_delta = boot[boot.metric.eq("pr_auc")].iloc[0]
    locked_m = mfr_dec[mfr_dec.arm.eq("locked")]["manufacturer_balanced_accuracy"]
    clean_m = mfr_dec[mfr_dec.arm.eq("clean_foldwise_input_combat")]["manufacturer_balanced_accuracy"]
    philips = transport[transport.manufacturer.astype(str).str.lower().eq("philips")]
    om_locked = oasis_metrics[oasis_metrics.arm.eq("locked")].iloc[0]
    om_clean = oasis_metrics[oasis_metrics.arm.eq("clean_foldwise_input_combat")].iloc[0]
    closer_text = "; ".join(
        f"{r.metric}: {int(r.folds_closer_under_clean)}/{int(r.folds_total)} folds closer, mean delta {r.mean_delta_clean_minus_locked:+.4f}"
        for r in wass_summary.itertuples()
    )
    all_pass = completion["status"].eq("PASS").all() and protocol["status"].eq("PASS").all() and safety_verdict == "FOLD_SAFE"
    report = f"""# Final clean fold-wise input-ComBat audit

## Verdict

{'PASS_MANUSCRIPT_GRADE_AUDIT' if all_pass else 'PARTIAL_OR_FAILED_AUDIT'}

Fold-safety verdict: **{safety_verdict}**. The clean run loads the original global tensor and applies ComBat fold-locally after splitting; this is expected and was not treated as a global pre-harmonized tensor requirement.

## Internal downstream diagnostic classifier

Primary clean ComBat ADNI OOF result, using posterior latent means + Age + Sex, class-weighted L2 logistic regression, five inner folds, inner-OOF ECDF normalization, and the prespecified inner-OOF threshold:

- ROC-AUC {cp.roc_auc:.4f}; PR-AUC {cp.pr_auc:.4f}
- balanced accuracy {cp.balanced_accuracy:.4f}; sensitivity {cp.sensitivity:.4f}; specificity {cp.specificity:.4f}; F1 {cp.f1:.4f}; Brier {cp.brier_score:.4f}
- confusion matrix: TN={int(cp.tn)}, FP={int(cp.fp)}, FN={int(cp.fn)}, TP={int(cp.tp)}

Locked reference under the same convention: ROC-AUC {lp['roc_auc']:.4f}; PR-AUC {lp['pr_auc']:.4f}; balanced accuracy {lp['balanced_accuracy']:.4f}.

Paired 10,000 diagnosis-stratified subject bootstrap, clean minus locked:

- delta ROC-AUC {auc_delta.observed_delta:+.4f} [{auc_delta.ci_low_2p5:+.4f}, {auc_delta.ci_high_97p5:+.4f}]
- delta PR-AUC {pr_delta.observed_delta:+.4f} [{pr_delta.ci_low_2p5:+.4f}, {pr_delta.ci_high_97p5:+.4f}]

## Manufacturer information and transport

Manufacturer decoding mean BACC: locked {locked_m.mean():.4f} ± {locked_m.std(ddof=1):.4f}; clean ComBat {clean_m.mean():.4f} ± {clean_m.std(ddof=1):.4f}; paired mean delta {(clean_m.to_numpy() - locked_m.to_numpy()).mean():+.4f}. Lower values indicate less manufacturer-decodable latent information.

Philips CN FPR rows are included in `cleanrepro_manufacturer_transport.csv`; primary subgroup quantity is Philips CN false-positive rate.

{md_table(philips, max_rows=10)}

## Frozen OASIS inference

Clean OASIS used only ADNI-fitted preprocessing, serialized fold ComBat objects, VAE checkpoints, frozen downstream diagnostic classifiers, ECDF mappings and thresholds. No OASIS fitting, calibration, or threshold selection was performed.

- Locked OASIS: ROC-AUC {om_locked.roc_auc:.4f}; PR-AUC {om_locked.pr_auc:.4f}; balanced accuracy {om_locked.balanced_accuracy:.4f}; sensitivity {om_locked.sensitivity:.4f}; specificity {om_locked.specificity:.4f}; Brier {om_locked.brier_score:.4f}
- Clean ComBat OASIS: ROC-AUC {om_clean.roc_auc:.4f}; PR-AUC {om_clean.pr_auc:.4f}; balanced accuracy {om_clean.balanced_accuracy:.4f}; sensitivity {om_clean.sensitivity:.4f}; specificity {om_clean.specificity:.4f}; Brier {om_clean.brier_score:.4f}

## Distribution shift

ADNI-vs-OASIS paired locked-vs-clean latent distances: {closer_text}. External alignment is considered consistently improved only when all folds move closer for all distance definitions.

## Training reproducibility

Compared to the historical fold-ComBat run, selected/terminal epochs and validation objectives are summarized in `cleanrepro_training_reproducibility.csv`.

{md_table(train_repro[['fold','selected_checkpoint_epoch','terminal_epoch','selected_validation_objective','historical_selected_checkpoint_epoch','historical_terminal_epoch','selected_epoch_reproduced','terminal_epoch_reproduced','validation_loss_reproduced_allclose']], max_rows=10)}

## Deliverables

All requested CSV deliverables and this report were written in this directory. The audit command log is `command_log.json`.
"""
    (OUT / "00_FINAL_CLEANREPRO_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Output directory already contains files: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    start = utc_now()
    commands: list[dict[str, Any]] = [
        {
            "timestamp_utc": start,
            "command": " ".join(sys.argv),
            "cwd": str(PROJECT),
            "python": sys.executable,
            "platform": platform.platform(),
            "read_only_training_artifacts": True,
            "no_vae_training": True,
            "no_combat_refit": True,
            "no_oasis_fitting_or_recalibration": True,
        }
    ]

    completion = completion_and_lineage_audit()
    diff, parity, protocol = config_diff_and_subject_parity()
    safety_df, safety_verdict = fold_safety()
    train_repro = summarize_history()
    foldwise, pooled = downstream_metrics()
    clean = load_clean_oof()
    locked = load_locked_oof()
    boot = paired_bootstrap(clean, locked)
    mfr_dec = manufacturer_decoding()
    transport = manufacturer_transport(clean, locked)
    oasis_preds, oasis_latents, oasis_metrics = clean_oasis_inference()
    oasis_boot = oasis_bootstrap(oasis_preds)
    wass_by_fold, wass_summary = wasserstein(oasis_latents)

    # Requested filenames.
    write_csv("cleanrepro_completion_and_lineage_audit.csv", pd.concat([completion, safety_df.assign(check="fold_safety_summary", status=safety_df["verdict"])], ignore_index=True, sort=False))
    write_csv("cleanrepro_config_diff_vs_locked.csv", diff)
    write_csv("cleanrepro_fold_subject_parity.csv", parity)
    write_csv("cleanrepro_training_reproducibility.csv", train_repro)
    write_csv("cleanrepro_downstream_foldwise_metrics.csv", foldwise)
    write_csv("cleanrepro_downstream_pooled_metrics.csv", pooled)
    write_csv("cleanrepro_paired_bootstrap_vs_locked.csv", boot)
    write_csv("cleanrepro_manufacturer_decoding.csv", mfr_dec)
    write_csv("cleanrepro_manufacturer_transport.csv", transport)
    write_csv("cleanrepro_oasis_metrics.csv", oasis_metrics)
    write_csv("cleanrepro_oasis_paired_bootstrap.csv", oasis_boot)
    write_csv("cleanrepro_wasserstein_by_fold.csv", wass_by_fold)
    write_csv("cleanrepro_wasserstein_summary.csv", wass_summary)
    manuscript = pd.DataFrame(
        [
            {"domain": "ADNI_internal", "arm": "locked", **pooled_metrics_from_predictions(locked, "locked")},
            {"domain": "ADNI_internal", **pooled.iloc[0].to_dict()},
            {"domain": "OASIS_external", **oasis_metrics[oasis_metrics.arm.eq("locked")].iloc[0].to_dict()},
            {"domain": "OASIS_external", **oasis_metrics[oasis_metrics.arm.eq("clean_foldwise_input_combat")].iloc[0].to_dict()},
        ]
    )
    write_csv("manuscript_ready_results_table.csv", manuscript)
    oasis_preds.to_csv(OUT / "cleanrepro_oasis_predictions.csv", index=False)
    oasis_latents.to_csv(OUT / "cleanrepro_oasis_fold_latent_mu.csv", index=False)
    protocol.to_csv(OUT / "cleanrepro_protocol_checks.csv", index=False)

    write_final_report(completion, protocol, safety_verdict, train_repro, pooled, boot, mfr_dec, transport, oasis_metrics, oasis_boot, wass_summary)
    commands.append({"timestamp_utc": utc_now(), "status": "COMPLETE", "output_dir": str(OUT)})
    (OUT / "command_log.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
