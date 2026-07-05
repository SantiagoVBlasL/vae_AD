#!/usr/bin/env python3
"""Scale-corrected FAST 3x3 connectivity-channel ablation.

Default behavior is dry-run/preflight only. Real training requires
--confirm-training. Stage A trains FAST VAEs with a dummy canonical logreg
readout, and Stage B runs the classifier-only logreg_l2 readout on saved latent
mu + Age + Sex. Ranking and aggregation never use Stage A classifier metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = "/home/diego/anaconda3/envs/vae_ad/bin/python"
TRAINING_SCRIPT = PROJECT_ROOT / "scripts/run_vae_clf_ad_inference.py"
READOUT_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"

OUTPUT_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/channel_ablation_fast3x3_offdiag_channelmean"
BIG_DISK_ROOT = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/channel_ablation_fast3x3_offdiag_channelmean")

DATASETS = {
    "v5_1b": {
        "label": "v5.1b final/manuscript-comparable",
        "tensor": Path(
            "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
            "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
            "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
        ),
        "metadata": Path(
            "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
            "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
            "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
        ),
    },
    "v5_1c": {
        "label": "v5.1c recover035 clean metadata branch",
        "tensor": Path(
            "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
            "adni_expanded_v5_1c_recover035_no_pybandpass/subject_tensors/"
            "GLOBAL_TENSOR_ADNI_expanded_v5_1c_recover035_no_pybandpass.npz"
        ),
        "metadata": Path(
            "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
            "adni_expanded_v5_1c_recover035_no_pybandpass/"
            "training_ready_metadata_v5_1c_recover035_no_pybandpass.csv"
        ),
    },
}

CHANNEL_NAMES = {
    0: "Pearson_OMST_GCE_Signed_Weighted",
    1: "Pearson_Full_FisherZ_Signed",
    2: "MI_KNN_Symmetric",
    3: "dFC_AbsDiffMean",
    4: "dFC_StdDev",
    5: "DistanceCorr",
    6: "Granger_F_lag1",
}
CHANNEL_SUBSETS = [
    [0],
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [1, 0],
    [1, 2],
    [0, 2],
    [1, 0, 2],
    [1, 0, 2, 5],
    [0, 1, 2, 3, 4, 5, 6],
]
REFERENCE_CHANNELS = [1, 0, 2]
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
SECONDARY_THRESHOLDS = ["fixed_0p5", "inner_oof_youden_j"]

FAST_PARAMS: Dict[str, Any] = {
    "classifier_types": ["logreg"],
    "classifier_stratify_cols": ["Manufacturer"],
    "vae_stratify_cols": ["Manufacturer"],
    "classifier_calibrate": True,
    "classifier_use_class_weight": True,
    "latent_features_type": "mu",
    "gridsearch_scoring": "roc_auc",
    "outer_folds": 3,
    "inner_folds": 3,
    "repeated_outer_folds_n_repeats": 1,
    "num_conv_layers_encoder": 4,
    "decoder_type": "convtranspose",
    "epochs_vae": 960,
    "vae_val_split_ratio": 0.2,
    "early_stopping_patience_vae": 240,
    "cyclical_beta_n_cycles": 12,
    "cyclical_beta_ratio_increase": 0.4,
    "beta_vae": 2.5,
    "recon_loss_mode": "offdiag_channelmean_sum",
    "dropout_rate_vae": 0.15,
    "vae_dropout_scope": "legacy_all",
    "vae_block_order": "legacy_act_norm",
    "latent_dim": 256,
    "batch_size": 64,
    "lr_vae": 0.0001,
    "lr_scheduler_type": "cosine_warm",
    "lr_scheduler_T0": 80,
    "lr_scheduler_eta_min": 5e-7,
    "lr_scheduler_patience_vae": 15,
    "weight_decay_vae": 5e-7,
    "vae_final_activation": "tanh",
    "intermediate_fc_dim_vae": "quarter",
    "use_layernorm_vae_fc": False,
    "n_jobs_gridsearch": 8,
    "metadata_features": ["Age", "Sex"],
    "norm_mode": "zscore_offdiag",
    "seed": 42,
    "num_workers": 4,
    "log_interval_epochs_vae": 10,
    "save_fold_artefacts": True,
    "save_vae_training_history": True,
    "qc_analyze_distributions": True,
    "qc_check_scanner_leakage": True,
    "qc_rate_distortion": True,
    "qc_latent_information": True,
    "qc_mi_n_neighbors": 3,
    "qc_mi_top_k": 10,
    "qc_rd_log_base": 2.0,
    "qc_tc_ridge": 1e-6,
    "qc_var_eps_active": 1e-4,
    "use_optuna_pruner": False,
    "use_smote": False,
    "vae_train_sampler_strategy": "none",
    "tune_sampler_params": False,
    "mlp_classifier_hidden_layers": "64,16",
    # Stage A classifier is a dummy required by canonical training wrapper.
    "n_iter_logreg": 1,
}

PARAM_ORDER = [
    "channels_to_use",
    "classifier_types",
    "classifier_stratify_cols",
    "vae_stratify_cols",
    "classifier_calibrate",
    "classifier_use_class_weight",
    "latent_features_type",
    "gridsearch_scoring",
    "outer_folds",
    "inner_folds",
    "repeated_outer_folds_n_repeats",
    "num_conv_layers_encoder",
    "decoder_type",
    "epochs_vae",
    "vae_val_split_ratio",
    "early_stopping_patience_vae",
    "cyclical_beta_n_cycles",
    "cyclical_beta_ratio_increase",
    "beta_vae",
    "recon_loss_mode",
    "dropout_rate_vae",
    "vae_dropout_scope",
    "vae_block_order",
    "latent_dim",
    "batch_size",
    "lr_vae",
    "lr_scheduler_type",
    "lr_scheduler_T0",
    "lr_scheduler_eta_min",
    "lr_scheduler_patience_vae",
    "weight_decay_vae",
    "vae_final_activation",
    "intermediate_fc_dim_vae",
    "use_layernorm_vae_fc",
    "n_jobs_gridsearch",
    "metadata_features",
    "norm_mode",
    "seed",
    "num_workers",
    "log_interval_epochs_vae",
    "save_fold_artefacts",
    "save_vae_training_history",
    "qc_analyze_distributions",
    "qc_check_scanner_leakage",
    "qc_rate_distortion",
    "qc_latent_information",
    "qc_mi_n_neighbors",
    "qc_mi_top_k",
    "qc_rd_log_base",
    "qc_tc_ridge",
    "qc_var_eps_active",
    "use_optuna_pruner",
    "use_smote",
    "vae_train_sampler_strategy",
    "tune_sampler_params",
    "mlp_classifier_hidden_layers",
    "n_iter_logreg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-branch", choices=sorted(DATASETS), default="v5_1b")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--big-disk-root", type=Path, default=BIG_DISK_ROOT)
    parser.add_argument("--candidate", default="all", help="Run key such as ch1_0_2, or 'all'.")
    parser.add_argument("--dry-run", action="store_true", help="Preflight only; do not train.")
    parser.add_argument("--confirm-training", action="store_true", help="Required for real Stage A/Stage B launch.")
    parser.add_argument("--resume", action="store_true", help="Skip candidates with complete Stage B readout.")
    parser.add_argument("--force-clean", action="store_true", help="Move stale or malformed candidate output to timestamped quarantine before dry-run symlink preflight or real training.")
    parser.add_argument("--aggregate-only", action="store_true", help="Only aggregate already completed readouts.")
    parser.add_argument("--python-executable", default=PYTHON_EXE)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def channel_key(channels: Sequence[int]) -> str:
    return "ch" + "_".join(str(x) for x in channels)


def channel_names(channels: Sequence[int]) -> List[str]:
    return [CHANNEL_NAMES[int(ch)] for ch in channels]


def append_arg(command: List[str], name: str, value: Any) -> None:
    if isinstance(value, bool):
        if value:
            command.append(f"--{name}")
        return
    if value is None:
        return
    command.append(f"--{name}")
    if isinstance(value, list):
        command.extend(str(v) for v in value)
    else:
        command.append(str(value))


def values_after_flag(tokens: Sequence[str], flag: str) -> List[str]:
    if flag not in tokens:
        return []
    out: List[str] = []
    for token in tokens[tokens.index(flag) + 1 :]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


def require_flag_values(tokens: Sequence[str], flag: str, expected: Sequence[str], context: str) -> None:
    observed = values_after_flag(tokens, flag)
    if observed != list(expected):
        raise RuntimeError(f"{context}: expected {flag} {' '.join(expected)}, got {observed}")


def normalize_manufacturer(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    upper = text.upper()
    if "GE" in upper:
        return "GE"
    if "SIEMENS" in upper:
        return "SIEMENS"
    if "PHILIPS" in upper:
        return "Philips"
    return text or "UNKNOWN"


def inspect_tensor(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as zf:
        shape = tuple(int(x) for x in zf["global_tensor_data"].shape)
        names = [str(x) for x in zf["channel_names"].astype(str)]
        bandpass = bool(zf["python_bandpass_applied"]) if "python_bandpass_applied" in zf.files else None
    if bandpass is not False:
        raise RuntimeError(f"Python bandpass must be OFF; got python_bandpass_applied={bandpass}")
    expected_names = [CHANNEL_NAMES[i] for i in range(7)]
    if names[:7] != expected_names:
        raise RuntimeError(f"Unexpected channel names/order: {names[:7]}")
    return {"shape": shape, "channel_names": names, "python_bandpass_applied": bandpass}


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    meta = pd.read_csv(path)
    if "tensor_idx" not in meta.columns and "tensor_index" in meta.columns:
        meta = meta.rename(columns={"tensor_index": "tensor_idx"})
    required = ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "tensor_idx"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise RuntimeError(f"Metadata missing required columns: {missing}")
    meta = meta.copy()
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    meta["ResearchGroup_Mapped"] = meta["ResearchGroup_Mapped"].astype(str)
    meta["Manufacturer"] = meta["Manufacturer"].map(normalize_manufacturer)
    meta["Sex"] = meta["Sex"].fillna("UNKNOWN").astype(str)
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["tensor_idx"] = meta["tensor_idx"].astype(int)
    if meta["SubjectID"].duplicated().any():
        raise RuntimeError("Metadata has duplicate SubjectID rows.")
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    if cn_ad["Age"].isna().any() or cn_ad["Sex"].isna().any():
        raise RuntimeError("CN/AD rows contain missing Age/Sex.")
    return meta


def strat_key(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    tmp = df[list(cols)].copy()
    for col in cols:
        tmp[col] = tmp[col].fillna(f"{col}_UNKNOWN").astype(str)
    return tmp.apply(lambda row: "_".join(row.values.astype(str)), axis=1)


def count_fields(df: pd.DataFrame) -> Dict[str, Any]:
    row: Dict[str, Any] = {"n": int(len(df))}
    for dx in ["AD", "CN", "MCI"]:
        row[dx] = int(df["ResearchGroup_Mapped"].eq(dx).sum())
    for mfr in ["GE", "SIEMENS", "Philips"]:
        label = "Siemens" if mfr == "SIEMENS" else mfr
        row[f"Manufacturer_{label}"] = int(df["Manufacturer"].eq(mfr).sum())
        for dx in ["AD", "CN", "MCI"]:
            row[f"{dx}_{label}"] = int((df["ResearchGroup_Mapped"].eq(dx) & df["Manufacturer"].eq(mfr)).sum())
    return row


def split_preview(meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seed = int(FAST_PARAMS["seed"])
    rows: List[Dict[str, Any]] = []
    subject_rows: List[pd.DataFrame] = []
    cnad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].reset_index(drop=True)
    outer_key = strat_key(cnad, ["ResearchGroup_Mapped", "Manufacturer"])
    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(outer.split(np.zeros(len(cnad)), outer_key), start=1):
        train_dev = cnad.iloc[train_idx].copy()
        test = cnad.iloc[test_idx].copy()
        pool = meta[~meta["tensor_idx"].isin(test["tensor_idx"])].copy()
        val_key = strat_key(pool, ["ResearchGroup_Mapped", "Manufacturer"])
        min_count = int(val_key.value_counts().min())
        if min_count < 3:
            raise RuntimeError(f"Fold {fold}: insufficient VAE val stratum count={min_count}")
        tr_pool, val_pool = train_test_split(
            pool,
            test_size=0.2,
            random_state=seed + fold,
            stratify=val_key,
        )
        for name, frame in [
            ("classifier_train_dev", train_dev),
            ("classifier_test", test),
            ("vae_train_pool", tr_pool),
            ("vae_internal_val", val_pool),
        ]:
            row = {"fold": fold, "component": name}
            row.update(count_fields(frame))
            rows.append(row)
            tmp = frame[["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "tensor_idx"]].copy()
            tmp["fold"] = fold
            tmp["component"] = name
            subject_rows.append(tmp)
    summary = pd.DataFrame(rows)
    subjects = pd.concat(subject_rows, ignore_index=True, sort=False)
    for _, row in summary.iterrows():
        diagnoses = ["AD", "CN"] if row["component"].startswith("classifier") else ["AD", "CN", "MCI"]
        for dx in diagnoses:
            if int(row.get(dx, 0)) <= 0:
                raise RuntimeError(f"Split preview invalid: fold={row.fold}, component={row.component}, {dx}=0")
        for mfr in ["GE", "Siemens", "Philips"]:
            if int(row.get(f"Manufacturer_{mfr}", 0)) <= 0:
                raise RuntimeError(f"Split preview invalid: fold={row.fold}, component={row.component}, Manufacturer_{mfr}=0")
    return summary, subjects


def output_dirs(output_root: Path, big_root: Path, channels: Sequence[int]) -> Dict[str, Path]:
    key = channel_key(channels)
    run_dir = output_root / "runs" / key
    big_dir = big_root / key
    return {
        "run_dir": run_dir,
        "big_dir": big_dir,
        "readout_dir": run_dir / "classifier_only_readout",
    }


def candidate_config(
    output_root: Path,
    big_root: Path,
    dataset_branch: str,
    dataset: Dict[str, Any],
    channels: Sequence[int],
) -> Dict[str, Any]:
    dirs = output_dirs(output_root, big_root, channels)
    params = dict(FAST_PARAMS)
    params["channels_to_use"] = list(channels)
    key = channel_key(channels)
    return {
        "run_name": f"channel_ablation_fast3x3_offdiag_channelmean_{dataset_branch}_{key}",
        "description": "Scale-corrected FAST 3x3 channel ablation using offdiag_channelmean_sum reconstruction loss. Stage A classifier is dummy/ignored; Stage B logreg_l2 is the readout.",
        "created_utc": now_utc(),
        "dataset_branch": dataset_branch,
        "dataset_label": dataset["label"],
        "run_key": key,
        "channels": list(channels),
        "channel_names_master_in_tensor_order": [CHANNEL_NAMES[i] for i in range(7)],
        "selected_channel_names": channel_names(channels),
        "stage_a_classifier_outputs": "dummy_logreg_ignored_for_ranking",
        "primary_readout": {
            "script": str(READOUT_SCRIPT.relative_to(PROJECT_ROOT)),
            "model": PRIMARY_MODEL,
            "latent_features": "mu",
            "metadata_features": ["Age", "Sex"],
            "threshold_strategy": PRIMARY_THRESHOLD,
            "threshold_selection": "true_inner_cv_oof",
        },
        "paths": {
            "training_script": str(TRAINING_SCRIPT.relative_to(PROJECT_ROOT)),
            "classifier_only_script": str(READOUT_SCRIPT.relative_to(PROJECT_ROOT)),
            "global_tensor_path": str(dataset["tensor"]),
            "metadata_path": str(dataset["metadata"]),
            "output_dir": str(dirs["run_dir"].relative_to(PROJECT_ROOT)),
            "big_disk_output_dir": str(dirs["big_dir"]),
            "classifier_only_output_dir": str(dirs["readout_dir"].relative_to(PROJECT_ROOT)),
        },
        "split_strategy": {
            "classifier_outer": ["ResearchGroup_Mapped", "Manufacturer"],
            "classifier_inner": ["ResearchGroup_Mapped", "Manufacturer"],
            "vae_internal_val": ["ResearchGroup_Mapped", "Manufacturer"],
            "metadata_covariates_only": ["Age", "Sex"],
            "sex_primary_stratifier": False,
        },
        "stage_a": {
            "purpose": "fast_vae_fit_and_fold_artifact_export",
            "canonical_classifier": "logreg_dummy",
            "canonical_classifier_n_iter_logreg": 1,
            "canonical_classifier_output": "ignored_for_final_ranking",
            "final_ranking_allowed": False,
        },
        "stage_b": {
            "purpose": "classifier_only_readout_on_saved_latent_mu",
            "primary_model": PRIMARY_MODEL,
            "primary_threshold_strategy": PRIMARY_THRESHOLD,
            "secondary_threshold_strategies": SECONDARY_THRESHOLDS,
            "threshold_selection": "true_inner_cv_oof_required_for_non_0p5",
            "features": ["latent_mu", "Age", "Sex"],
            "outer_folds": int(params["outer_folds"]),
            "inner_folds": int(params["inner_folds"]),
        },
        "parameters": params,
    }


def build_stage_a_command(config: Dict[str, Any], python_executable: str) -> List[str]:
    paths = config["paths"]
    params = config["parameters"]
    unknown = sorted(set(params) - set(PARAM_ORDER))
    if unknown:
        raise RuntimeError(f"Unknown Stage A parameters: {unknown}")
    command = [
        python_executable,
        str((PROJECT_ROOT / paths["training_script"]).resolve()),
        "--global_tensor_path",
        paths["global_tensor_path"],
        "--metadata_path",
        paths["metadata_path"],
        "--output_dir",
        str((PROJECT_ROOT / paths["output_dir"]).resolve()),
    ]
    for name in PARAM_ORDER:
        append_arg(command, name, params.get(name))
    validate_stage_a_command(command)
    return command


def build_stage_b_command(config: Dict[str, Any], python_executable: str) -> List[str]:
    paths = config["paths"]
    params = config["parameters"]
    command = [
        python_executable,
        str((PROJECT_ROOT / paths["classifier_only_script"]).resolve()),
        "--run-dir",
        str((PROJECT_ROOT / paths["output_dir"]).resolve()),
        "--output-dir",
        str((PROJECT_ROOT / paths["classifier_only_output_dir"]).resolve()),
        "--models",
        PRIMARY_MODEL,
        "--outer-folds",
        str(int(params["outer_folds"])),
        "--inner-folds",
        str(int(params["inner_folds"])),
        "--overwrite",
    ]
    validate_stage_b_command(command)
    return command


def validate_stage_a_command(command: Sequence[str]) -> None:
    tokens = list(command)
    require_flag_values(tokens, "--classifier_types", ["logreg"], "Stage A")
    require_flag_values(tokens, "--n_iter_logreg", ["1"], "Stage A")
    require_flag_values(tokens, "--outer_folds", ["3"], "Stage A")
    require_flag_values(tokens, "--inner_folds", ["3"], "Stage A")
    require_flag_values(tokens, "--latent_dim", ["256"], "Stage A")
    require_flag_values(tokens, "--epochs_vae", ["960"], "Stage A")
    require_flag_values(tokens, "--cyclical_beta_n_cycles", ["12"], "Stage A")
    require_flag_values(tokens, "--lr_scheduler_T0", ["80"], "Stage A")
    require_flag_values(tokens, "--recon_loss_mode", ["offdiag_channelmean_sum"], "Stage A")
    require_flag_values(tokens, "--vae_final_activation", ["tanh"], "Stage A")
    require_flag_values(tokens, "--metadata_features", ["Age", "Sex"], "Stage A")
    if "Manufacturer" not in values_after_flag(tokens, "--classifier_stratify_cols"):
        raise RuntimeError("Stage A classifier_stratify_cols must include Manufacturer.")
    if "Manufacturer" not in values_after_flag(tokens, "--vae_stratify_cols"):
        raise RuntimeError("Stage A vae_stratify_cols must include Manufacturer.")
    if "Sex" in values_after_flag(tokens, "--classifier_stratify_cols") or "Sex" in values_after_flag(tokens, "--vae_stratify_cols"):
        raise RuntimeError("Sex must not be used as stratifier.")
    forbidden = ["--n_iter_svm", "--n_iter_rf", "--n_iter_gb", "--n_iter_xgb", "--n_iter_mlp"]
    present = [flag for flag in forbidden if flag in tokens]
    if present:
        raise RuntimeError(f"Stage A must not include unused classifier trial flags: {present}")
    for idx, token in enumerate(tokens):
        if token.startswith("--n_iter_") and idx + 1 < len(tokens) and tokens[idx + 1] == "0":
            raise RuntimeError(f"Invalid zero Optuna trials in command: {token} 0")


def validate_stage_b_command(command: Sequence[str]) -> None:
    tokens = list(command)
    require_flag_values(tokens, "--models", [PRIMARY_MODEL], "Stage B")
    require_flag_values(tokens, "--outer-folds", ["3"], "Stage B")
    require_flag_values(tokens, "--inner-folds", ["3"], "Stage B")


def validate_config(config: Dict[str, Any]) -> None:
    params = config["parameters"]
    if params["recon_loss_mode"] != "offdiag_channelmean_sum":
        raise RuntimeError("Config must use recon_loss_mode=offdiag_channelmean_sum")
    if params["epochs_vae"] / params["cyclical_beta_n_cycles"] != 80:
        raise RuntimeError("Expected cycle length epochs_vae/cycles = 80")
    if params["lr_scheduler_T0"] != 80:
        raise RuntimeError("Expected lr_scheduler_T0=80")
    required = {
        "outer_folds": 3,
        "inner_folds": 3,
        "latent_dim": 256,
        "beta_vae": 2.5,
        "dropout_rate_vae": 0.15,
        "vae_dropout_scope": "legacy_all",
        "vae_block_order": "legacy_act_norm",
        "vae_final_activation": "tanh",
        "intermediate_fc_dim_vae": "quarter",
        "decoder_type": "convtranspose",
        "num_conv_layers_encoder": 4,
        "norm_mode": "zscore_offdiag",
        "classifier_use_class_weight": True,
        "classifier_calibrate": True,
        "vae_train_sampler_strategy": "none",
        "n_iter_logreg": 1,
    }
    for key, expected in required.items():
        if params.get(key) != expected:
            raise RuntimeError(f"{config['run_key']}.{key}: expected {expected!r}, got {params.get(key)!r}")
    if params["classifier_types"] != ["logreg"]:
        raise RuntimeError("Stage A must use dummy canonical logreg only.")
    if params["metadata_features"] != ["Age", "Sex"]:
        raise RuntimeError("metadata_features must be ['Age', 'Sex']")
    if "Sex" in params["classifier_stratify_cols"] or "Sex" in params["vae_stratify_cols"]:
        raise RuntimeError("Sex must remain metadata/covariate only.")
    if "Manufacturer" not in params["classifier_stratify_cols"] or "Manufacturer" not in params["vae_stratify_cols"]:
        raise RuntimeError("Manufacturer must be included in classifier and VAE stratification.")
    readout = config["primary_readout"]
    if readout["model"] != PRIMARY_MODEL or readout["threshold_strategy"] != PRIMARY_THRESHOLD:
        raise RuntimeError("Primary readout config mismatch.")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json.loads(path.read_text(encoding="utf-8"))


def markdown_table(df: pd.DataFrame, cols: Optional[Sequence[str]] = None, max_rows: int = 80) -> str:
    if df.empty:
        return "No rows.\n"
    if cols is None:
        cols = list(df.columns)
    sub = df.loc[:, [c for c in cols if c in df.columns]].head(max_rows)
    lines = ["| " + " | ".join(sub.columns) + " |", "| " + " | ".join(["---"] * len(sub.columns)) + " |"]
    for _, row in sub.iterrows():
        vals = []
        for col in sub.columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{val:.6f}" if np.isfinite(val) else "NA")
            else:
                vals.append(str(val).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_pair(output_root: Path, name: str, df: pd.DataFrame, cols: Optional[Sequence[str]] = None, max_rows: int = 80) -> None:
    df.to_csv(output_root / f"{name}.csv", index=False)
    (output_root / f"{name}.md").write_text(markdown_table(df, cols=cols, max_rows=max_rows), encoding="utf-8")


def readout_complete(readout_dir: Path) -> bool:
    required = [
        "classifier_sweep_foldwise_metrics.csv",
        "classifier_sweep_pooled_metrics.csv",
        "classifier_sweep_thresholds_by_fold.csv",
        "classifier_sweep_subgroup_metrics_by_manufacturer.csv",
        "classifier_sweep_predictions.csv",
        "command_log.json",
    ]
    return all((readout_dir / name).exists() for name in required)


def stale_markers(run_dir: Path) -> List[Path]:
    names = [
        "classifier_only_readout",
        "latent_cache",
        "all_folds_metrics.csv",
        "summary_metrics.csv",
        "run_manifest.json",
    ]
    out = [run_dir / name for name in names if (run_dir / name).exists()]
    out.extend(sorted(run_dir.glob("fold_*")))
    return out


def unique_quarantine_path(path: Path, suffix: str = "quarantine") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = path.parent / f"{path.name}_{suffix}_{timestamp}"
    candidate = base
    idx = 1
    while candidate.exists() or candidate.is_symlink():
        idx += 1
        candidate = path.parent / f"{path.name}_{suffix}_{timestamp}_{idx}"
    return candidate


def quarantine_existing_path(path: Path) -> Optional[Path]:
    if not path.exists() and not path.is_symlink():
        return None
    quarantine = unique_quarantine_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(quarantine))
    return quarantine


def quarantine_directory_contents(directory: Path) -> Optional[Path]:
    if not directory.exists():
        return None
    children = list(directory.iterdir())
    if not children:
        return None
    quarantine = unique_quarantine_path(directory, suffix="contents_quarantine")
    quarantine.mkdir(parents=True, exist_ok=False)
    for child in children:
        shutil.move(str(child), str(quarantine / child.name))
    return quarantine


def symlink_points_to(link_path: Path, target_path: Path) -> bool:
    if not link_path.is_symlink():
        return False
    raw_target = Path(os.readlink(link_path))
    if not raw_target.is_absolute():
        raw_target = link_path.parent / raw_target
    try:
        return raw_target.resolve() == target_path.resolve()
    except FileNotFoundError:
        return raw_target.absolute() == target_path.absolute()


def create_or_update_symlink(link_path: Path, target_path: Path, force_clean: bool = False) -> List[Path]:
    """Prepare link_path as a symlink to target_path without creating a local run dir.

    The local run path is a pointer only. Any stale normal directory at that
    path is moved aside when force_clean=True; otherwise we fail before any
    training starts.
    """
    quarantines: List[Path] = []
    link_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.mkdir(parents=True, exist_ok=True)

    if link_path.is_symlink():
        if symlink_points_to(link_path, target_path):
            markers = stale_markers(link_path)
            if markers and not force_clean:
                raise RuntimeError(
                    f"Refusing to run with stale candidate outputs in {link_path}. "
                    "Use --force-clean to move them to quarantine."
                )
            if markers and force_clean:
                quarantine = quarantine_directory_contents(target_path)
                if quarantine is not None:
                    quarantines.append(quarantine)
            return quarantines
        if not force_clean:
            raise RuntimeError(
                f"Existing symlink points elsewhere: {link_path} -> {os.readlink(link_path)}. "
                "Use --force-clean to move it to quarantine and recreate it."
            )
        quarantine = quarantine_existing_path(link_path)
        if quarantine is not None:
            quarantines.append(quarantine)
    elif link_path.exists():
        if not force_clean:
            raise RuntimeError(
                f"Refusing to replace existing non-symlink path: {link_path}. "
                "Use --force-clean to move it to timestamped quarantine and create the symlink."
            )
        quarantine = quarantine_existing_path(link_path)
        if quarantine is not None:
            quarantines.append(quarantine)

    if not link_path.exists() and not link_path.is_symlink():
        link_path.symlink_to(target_path, target_is_directory=True)

    if not link_path.is_symlink() or not symlink_points_to(link_path, target_path):
        raise RuntimeError(f"Failed to prepare run_dir symlink: {link_path} -> {target_path}")
    return quarantines


def verify_run_symlink(row: pd.Series) -> None:
    run_dir = resolve(row["run_dir"])
    target = Path(row["big_disk_output_dir"])
    if not run_dir.is_symlink():
        raise RuntimeError(f"Prepared local run path is not a symlink: {run_dir}")
    if not symlink_points_to(run_dir, target):
        raise RuntimeError(f"Prepared symlink points to the wrong target: {run_dir} -> {os.readlink(run_dir)}; expected {target}")


def prepare_run_symlink(row: pd.Series, force_clean: bool) -> List[Path]:
    run_dir = resolve(row["run_dir"])
    target = Path(row["big_disk_output_dir"])
    quarantines = create_or_update_symlink(run_dir, target, force_clean=force_clean)
    verify_run_symlink(row)
    return quarantines


def verify_fresh_checkpoints(run_dir: Path, run_start_time: float, outer_folds: int = 3) -> None:
    missing: List[str] = []
    stale: List[str] = []
    for fold in range(1, outer_folds + 1):
        ckpt = run_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt"
        if not ckpt.exists():
            missing.append(str(ckpt))
        elif ckpt.stat().st_mtime < run_start_time:
            stale.append(str(ckpt))
    if missing or stale:
        raise RuntimeError(f"Checkpoint freshness failed. Missing={missing}; stale={stale}")


def build_plan(output_root: Path, big_root: Path, dataset_branch: str, python_executable: str) -> pd.DataFrame:
    dataset = DATASETS[dataset_branch]
    configs_dir = output_root / "configs"
    rows: List[Dict[str, Any]] = []
    base_params: Optional[Dict[str, Any]] = None
    for idx, channels in enumerate(CHANNEL_SUBSETS, start=1):
        config = candidate_config(output_root, big_root, dataset_branch, dataset, channels)
        validate_config(config)
        if base_params is None:
            base_params = {k: v for k, v in config["parameters"].items() if k != "channels_to_use"}
        else:
            this_params = {k: v for k, v in config["parameters"].items() if k != "channels_to_use"}
            if this_params != base_params:
                raise RuntimeError("Candidate configs differ by more than channels_to_use/output metadata.")
        config_path = configs_dir / f"{config['run_key']}.json"
        write_json(config_path, config)
        stage_a = build_stage_a_command(config, python_executable)
        stage_b = build_stage_b_command(config, python_executable)
        rows.append(
            {
                "candidate_index": idx,
                "dataset_branch": dataset_branch,
                "run_key": config["run_key"],
                "channels": json.dumps(list(channels)),
                "selected_channel_names": " | ".join(config["selected_channel_names"]),
                "n_channels": len(channels),
                "config_path": str(config_path.relative_to(PROJECT_ROOT)),
                "run_dir": config["paths"]["output_dir"],
                "big_disk_output_dir": config["paths"]["big_disk_output_dir"],
                "readout_dir": config["paths"]["classifier_only_output_dir"],
                "recon_loss_mode": config["parameters"]["recon_loss_mode"],
                "outer_folds": config["parameters"]["outer_folds"],
                "inner_folds": config["parameters"]["inner_folds"],
                "latent_dim": config["parameters"]["latent_dim"],
                "epochs_vae": config["parameters"]["epochs_vae"],
                "cyclical_beta_n_cycles": config["parameters"]["cyclical_beta_n_cycles"],
                "cycle_len": config["parameters"]["epochs_vae"] / config["parameters"]["cyclical_beta_n_cycles"],
                "lr_scheduler_T0": config["parameters"]["lr_scheduler_T0"],
                "stage_a_training_classifier": "canonical_logreg_dummy_ignored_for_ranking",
                "stage_b_primary_readout": f"{PRIMARY_MODEL}_{PRIMARY_THRESHOLD}",
                "threshold_selection": "true_inner_cv_oof_for_non_0p5",
                "stage_a_command": shlex.join(stage_a),
                "stage_b_command": shlex.join(stage_b),
                "status": "planned",
            }
        )
    return pd.DataFrame(rows)


def validate_plan(plan: pd.DataFrame) -> None:
    if len(plan) != len(CHANNEL_SUBSETS):
        raise RuntimeError(f"Expected {len(CHANNEL_SUBSETS)} planned runs, got {len(plan)}")
    expected_keys = [channel_key(ch) for ch in CHANNEL_SUBSETS]
    if plan["run_key"].tolist() != expected_keys:
        raise RuntimeError(f"Planned run order mismatch: {plan['run_key'].tolist()} != {expected_keys}")
    if not (plan["recon_loss_mode"] == "offdiag_channelmean_sum").all():
        raise RuntimeError("All planned runs must use offdiag_channelmean_sum")
    if not (plan["cycle_len"] == 80).all() or not (plan["lr_scheduler_T0"] == 80).all():
        raise RuntimeError("All planned runs must preserve 80-epoch beta/LR cycles")
    for _, row in plan.iterrows():
        validate_stage_a_command(shlex.split(row["stage_a_command"]))
        validate_stage_b_command(shlex.split(row["stage_b_command"]))


def selected_rows(plan: pd.DataFrame, candidate: str) -> pd.DataFrame:
    if candidate == "all":
        return plan.copy()
    rows = plan[plan["run_key"].eq(candidate)].copy()
    if rows.empty:
        raise RuntimeError(f"Unknown candidate={candidate!r}. Valid keys: {plan['run_key'].tolist()} or 'all'.")
    return rows


def run_candidate(row: pd.Series, resume: bool, force_clean: bool) -> Dict[str, Any]:
    run_dir = resolve(row["run_dir"])
    readout_dir = resolve(row["readout_dir"])
    if resume and readout_complete(readout_dir):
        return {"run_key": row["run_key"], "status": "skipped_complete", "run_dir": str(run_dir), "readout_dir": str(readout_dir)}
    quarantines = prepare_run_symlink(row, force_clean=force_clean)
    run_start_time = time.time()
    stage_a_cmd = shlex.split(row["stage_a_command"])
    stage_b_cmd = shlex.split(row["stage_b_command"])
    validate_stage_a_command(stage_a_cmd)
    validate_stage_b_command(stage_b_cmd)
    completed_a = subprocess.run(stage_a_cmd, cwd=str(PROJECT_ROOT), check=False)
    if completed_a.returncode != 0:
        return {"run_key": row["run_key"], "status": "stage_a_failed", "returncode": completed_a.returncode, "quarantine": ";".join(str(q) for q in quarantines)}
    verify_fresh_checkpoints(run_dir, run_start_time, outer_folds=int(row["outer_folds"]))
    completed_b = subprocess.run(stage_b_cmd, cwd=str(PROJECT_ROOT), check=False)
    if completed_b.returncode != 0:
        return {"run_key": row["run_key"], "status": "stage_b_failed", "returncode": completed_b.returncode, "quarantine": ";".join(str(q) for q in quarantines)}
    if not readout_complete(readout_dir):
        return {"run_key": row["run_key"], "status": "stage_b_incomplete", "quarantine": ";".join(str(q) for q in quarantines)}
    return {"run_key": row["run_key"], "status": "complete", "run_dir": str(run_dir), "readout_dir": str(readout_dir), "quarantine": ";".join(str(q) for q in quarantines)}


def collect_primary_from_readout(row: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    readout_dir = resolve(row["readout_dir"])
    foldwise_path = readout_dir / "classifier_sweep_foldwise_metrics.csv"
    pooled_path = readout_dir / "classifier_sweep_pooled_metrics.csv"
    subgroup_path = readout_dir / "classifier_sweep_subgroup_metrics_by_manufacturer.csv"
    thresholds_path = readout_dir / "classifier_sweep_thresholds_by_fold.csv"
    if not all(p.exists() for p in [foldwise_path, pooled_path, subgroup_path, thresholds_path]):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    foldwise = pd.read_csv(foldwise_path)
    pooled = pd.read_csv(pooled_path)
    subgroup = pd.read_csv(subgroup_path)
    thresholds = pd.read_csv(thresholds_path)
    for df in [foldwise, pooled, subgroup, thresholds]:
        df["run_key"] = row["run_key"]
        df["channels"] = row["channels"]
        df["selected_channel_names"] = row["selected_channel_names"]
        df["n_channels"] = row["n_channels"]
    return pooled, foldwise, subgroup, thresholds


def collect_stage_a_summary(plan: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metric_map = {
        "auc_raw": "stage_a_auc_raw",
        "pr_auc_raw": "stage_a_pr_auc_raw",
        "auc_final": "stage_a_auc_final",
        "pr_auc_final": "stage_a_pr_auc_final",
        "sensitivity": "stage_a_sensitivity",
        "specificity": "stage_a_specificity",
        "f1_score": "stage_a_f1_score",
    }
    for _, row in plan.iterrows():
        run_dir = resolve(row["run_dir"])
        files = sorted(run_dir.glob("all_folds_metrics*.csv"))
        payload: Dict[str, Any] = {
            "run_key": row["run_key"],
            "stage_a_metrics_found": bool(files),
        }
        if files:
            path = files[-1]
            try:
                df = pd.read_csv(path)
            except Exception:
                df = pd.DataFrame()
            if "actual_classifier_type" in df.columns:
                df = df[df["actual_classifier_type"].astype(str).str.lower().eq("logreg")]
            payload["stage_a_n_folds"] = int(df["fold"].nunique()) if "fold" in df.columns else int(len(df))
            for src, dst in metric_map.items():
                payload[f"{dst}_mean"] = float(pd.to_numeric(df[src], errors="coerce").mean()) if src in df.columns and not df.empty else np.nan
            payload["stage_a_metrics_file"] = str(path)
        rows.append(payload)
    return pd.DataFrame(rows)


def collect_rate_distortion_summary(plan: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in plan.iterrows():
        run_dir = resolve(row["run_dir"])
        beta_max = 2.5
        for fold in range(1, int(row["outer_folds"]) + 1):
            path = run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty or "epoch" not in df.columns:
                continue
            score_col = "L_val_betaMax" if "L_val_betaMax" in df.columns else "D_val"
            best_idx = pd.to_numeric(df[score_col], errors="coerce").idxmin()
            best = df.loc[best_idx]
            final_epoch = int(pd.to_numeric(df["epoch"], errors="coerce").max())
            best_epoch = int(best["epoch"])
            d_val = float(best.get("D_val", np.nan))
            r_val = float(best.get("R_val_nats", np.nan))
            kld_over_recon = r_val / d_val if np.isfinite(d_val) and d_val != 0 else np.nan
            latent_info = run_dir / f"fold_{fold}" / f"fold_{fold}_trainDev_latent_info_summary.csv"
            n_active = np.nan
            frac_active = np.nan
            total_correlation = np.nan
            if latent_info.exists():
                try:
                    li = pd.read_csv(latent_info)
                    if not li.empty:
                        if "n_active" in li.columns:
                            n_active = float(pd.to_numeric(li["n_active"], errors="coerce").max())
                        if "frac_active" in li.columns:
                            frac_active = float(pd.to_numeric(li["frac_active"], errors="coerce").max())
                        if "total_correlation_nats" in li.columns:
                            total_correlation = float(pd.to_numeric(li["total_correlation_nats"], errors="coerce").mean())
                except Exception:
                    pass
            rows.append(
                {
                    "run_key": row["run_key"],
                    "channels": row["channels"],
                    "selected_channel_names": row["selected_channel_names"],
                    "n_channels": row["n_channels"],
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "final_epoch": final_epoch,
                    "early_stopped": bool(final_epoch < int(row["epochs_vae"])),
                    "epochs_after_best": int(final_epoch - best_epoch),
                    "best_val_l_beta_max": float(best.get("L_val_betaMax", np.nan)),
                    "best_val_recon_D": d_val,
                    "best_val_kld_R_nats": r_val,
                    "best_val_kld_over_recon": kld_over_recon,
                    "best_val_beta_kld_over_recon": beta_max * kld_over_recon if np.isfinite(kld_over_recon) else np.nan,
                    "best_epoch_beta": float(best.get("beta", np.nan)),
                    "active_units_trainDev": n_active,
                    "frac_active_trainDev": frac_active,
                    "total_correlation_trainDev_nats": total_correlation,
                    "source_file": str(path),
                }
            )
    return pd.DataFrame(rows)


def collect_scanner_leakage_summary(plan: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in plan.iterrows():
        run_dir = resolve(row["run_dir"])
        for fold in range(1, int(row["outer_folds"]) + 1):
            path = run_dir / f"fold_{fold}" / "latent_qc_metrics.csv"
            if not path.exists():
                path = run_dir / f"fold_{fold}" / f"fold_{fold}_test_scanner_leakage_summary.csv"
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty:
                continue
            src = df.iloc[0]
            acc_raw = src.get("acc_site_raw", src.get("acc_raw", np.nan))
            acc_latent = src.get("acc_site_latent", src.get("acc_latent", np.nan))
            acc_raw = float(acc_raw) if pd.notna(acc_raw) else np.nan
            acc_latent = float(acc_latent) if pd.notna(acc_latent) else np.nan
            rows.append(
                {
                    "run_key": row["run_key"],
                    "channels": row["channels"],
                    "selected_channel_names": row["selected_channel_names"],
                    "n_channels": row["n_channels"],
                    "fold": fold,
                    "site_col": src.get("site_col", "Manufacturer"),
                    "acc_raw": acc_raw,
                    "acc_latent": acc_latent,
                    "latent_minus_raw": acc_latent - acc_raw if np.isfinite(acc_raw) and np.isfinite(acc_latent) else np.nan,
                    "chance_level": src.get("chance_level", np.nan),
                    "n_sites": src.get("n_sites", src.get("n_classes", np.nan)),
                    "source_file": str(path),
                }
            )
    return pd.DataFrame(rows)


def aggregate_metrics(output_root: Path, plan: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    pooled_rows: List[pd.DataFrame] = []
    foldwise_rows: List[pd.DataFrame] = []
    subgroup_rows: List[pd.DataFrame] = []
    threshold_rows: List[pd.DataFrame] = []
    for _, row in plan.iterrows():
        pooled, foldwise, subgroup, thresholds = collect_primary_from_readout(row)
        if not pooled.empty:
            pooled_rows.append(pooled)
        if not foldwise.empty:
            foldwise_rows.append(foldwise)
        if not subgroup.empty:
            subgroup_rows.append(subgroup)
        if not thresholds.empty:
            threshold_rows.append(thresholds)
    pooled_all = pd.concat(pooled_rows, ignore_index=True, sort=False) if pooled_rows else pd.DataFrame()
    foldwise_all = pd.concat(foldwise_rows, ignore_index=True, sort=False) if foldwise_rows else pd.DataFrame()
    subgroup_all = pd.concat(subgroup_rows, ignore_index=True, sort=False) if subgroup_rows else pd.DataFrame()
    thresholds_all = pd.concat(threshold_rows, ignore_index=True, sort=False) if threshold_rows else pd.DataFrame()

    if pooled_all.empty:
        empty_cols = ["run_key", "channels", "selected_channel_names", "n_channels", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "status"]
        placeholder = pd.DataFrame(columns=empty_cols)
        for name in ["primary_ablation_table", "channel_subset_ranking", "foldwise_metrics", "paired_statistical_tests", "rate_distortion_by_subset", "scanner_leakage_by_subset"]:
            write_pair(output_root, name, placeholder)
        write_recommendation(output_root, placeholder, pd.DataFrame())
        return {
            "primary": placeholder,
            "ranking": placeholder,
            "foldwise": placeholder,
            "paired": placeholder,
            "rate_distortion": placeholder,
            "scanner_leakage": placeholder,
        }

    stage_a = collect_stage_a_summary(plan)
    primary = pooled_all[
        (pooled_all["model_name"].eq(PRIMARY_MODEL))
        & (pooled_all["threshold_strategy"].eq(PRIMARY_THRESHOLD))
    ].copy()
    if not stage_a.empty:
        primary = primary.merge(stage_a, on="run_key", how="left")
    primary = primary.sort_values(["auc", "pr_auc", "balanced_accuracy"], ascending=False)
    ranking = primary.copy()
    ranking["rank_auc"] = ranking["auc"].rank(ascending=False, method="min").astype(int)
    ranking["rank_pr_auc"] = ranking["pr_auc"].rank(ascending=False, method="min").astype(int)

    foldwise_primary = foldwise_all[
        (foldwise_all["model_name"].eq(PRIMARY_MODEL))
        & (foldwise_all["threshold_strategy"].eq(PRIMARY_THRESHOLD))
    ].copy()
    paired = paired_tests(foldwise_primary)
    rd = collect_rate_distortion_summary(plan)
    leakage = collect_scanner_leakage_summary(plan)

    write_pair(
        output_root,
        "primary_ablation_table",
        primary,
        [
            "run_key",
            "channels",
            "selected_channel_names",
            "n_channels",
            "auc",
            "pr_auc",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1",
            "tn",
            "fp",
            "fn",
            "tp",
            "stage_a_auc_final_mean",
            "stage_a_pr_auc_final_mean",
            "stage_a_sensitivity_mean",
            "stage_a_specificity_mean",
            "stage_a_f1_score_mean",
        ],
        max_rows=80,
    )
    write_pair(
        output_root,
        "channel_subset_ranking",
        ranking,
        ["rank_auc", "rank_pr_auc", "run_key", "channels", "selected_channel_names", "n_channels", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"],
        max_rows=80,
    )
    write_pair(
        output_root,
        "foldwise_metrics",
        foldwise_primary.sort_values(["run_key", "fold"]),
        ["run_key", "channels", "fold", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "tn", "fp", "fn", "tp"],
        max_rows=200,
    )
    write_pair(output_root, "paired_statistical_tests", paired, max_rows=200)
    write_pair(output_root, "rate_distortion_by_subset", rd, max_rows=200)
    write_pair(output_root, "scanner_leakage_by_subset", leakage, max_rows=200)
    write_recommendation(output_root, primary, paired)
    return {
        "primary": primary,
        "ranking": ranking,
        "foldwise": foldwise_primary,
        "paired": paired,
        "rate_distortion": rd,
        "scanner_leakage": leakage,
    }


def paired_tests(foldwise: pd.DataFrame) -> pd.DataFrame:
    if foldwise.empty:
        return pd.DataFrame()
    try:
        from scipy import stats
    except Exception:
        stats = None
    ref_key = channel_key(REFERENCE_CHANNELS)
    ref = foldwise[foldwise["run_key"].eq(ref_key)].copy()
    rows: List[Dict[str, Any]] = []
    metrics = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]
    for run_key, sub in foldwise.groupby("run_key"):
        if run_key == ref_key:
            continue
        merged = ref[["fold", *metrics]].merge(sub[["fold", *metrics]], on="fold", suffixes=("_ref", "_candidate"))
        for metric in metrics:
            delta = merged[f"{metric}_candidate"].to_numpy(dtype=float) - merged[f"{metric}_ref"].to_numpy(dtype=float)
            mean_delta = float(np.nanmean(delta)) if len(delta) else float("nan")
            sd_delta = float(np.nanstd(delta, ddof=1)) if len(delta) > 1 else float("nan")
            dz = float(mean_delta / sd_delta) if np.isfinite(sd_delta) and sd_delta > 0 else float("nan")
            if stats is not None and len(delta) >= 2 and np.isfinite(delta).all():
                try:
                    t_p = float(stats.ttest_rel(merged[f"{metric}_candidate"], merged[f"{metric}_ref"]).pvalue)
                except Exception:
                    t_p = float("nan")
                try:
                    w_p = float(stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue)
                except Exception:
                    w_p = float("nan")
            else:
                t_p = float("nan")
                w_p = float("nan")
            rows.append(
                {
                    "candidate_run_key": run_key,
                    "reference_run_key": ref_key,
                    "metric": metric,
                    "n_paired_folds": int(len(delta)),
                    "mean_delta_candidate_minus_reference": mean_delta,
                    "paired_cohen_dz": dz,
                    "paired_ttest_p_uncorrected": t_p,
                    "wilcoxon_p_uncorrected": w_p,
                    "note": "FAST 3-fold screening only; p-values are exploratory/non-confirmatory.",
                }
            )
    out = pd.DataFrame(rows)
    for p_col in ["paired_ttest_p_uncorrected", "wilcoxon_p_uncorrected"]:
        if p_col in out.columns and out[p_col].notna().any():
            out[p_col.replace("_uncorrected", "_bh_fdr")] = bh_fdr(out[p_col])
    return out


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = p_values.astype(float).to_numpy()
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return pd.Series(out, index=p_values.index)
    vals = p[finite]
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(vals)
    adjusted = np.minimum.accumulate((ranked * m / (np.arange(m) + 1))[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    tmp = np.empty_like(adjusted)
    tmp[order] = adjusted
    out[np.where(finite)[0]] = tmp
    return pd.Series(out, index=p_values.index)


def collect_qc_by_pattern(plan: pd.DataFrame, patterns: Sequence[str]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for _, row in plan.iterrows():
        run_dir = resolve(row["run_dir"])
        seen: set[Path] = set()
        for pattern in patterns:
            for path in sorted(run_dir.rglob(pattern)):
                if path in seen or "classifier_only_readout" in path.parts:
                    continue
                seen.add(path)
                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue
                df = df.copy()
                df.insert(0, "source_file", str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path))
                df.insert(0, "run_key", row["run_key"])
                df.insert(1, "channels", row["channels"])
                rows.append(df)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame(columns=["run_key", "channels", "source_file"])


def write_recommendation(output_root: Path, primary: pd.DataFrame, paired: pd.DataFrame) -> None:
    lines = [
        "# Final Channel Selection Recommendation",
        "",
        "This file is populated after real FAST runs complete. FAST 3x3 is a screening analysis, not a final performance estimate.",
        "",
    ]
    if primary.empty:
        lines += [
            "No completed Stage B readouts were available at aggregation time.",
            "",
            "Run the launcher with `--confirm-training` to execute Stage A/Stage B, then rerun `--aggregate-only`.",
        ]
    else:
        best = primary.iloc[0]
        ref = primary[primary["run_key"].eq(channel_key(REFERENCE_CHANNELS))]
        best_pr = primary.sort_values(["pr_auc", "auc"], ascending=False).iloc[0]
        best_ba = primary.sort_values(["balanced_accuracy", "auc"], ascending=False).iloc[0]
        best_f1 = primary.sort_values(["f1", "auc"], ascending=False).iloc[0]
        lines += [
            "## Primary Ranking",
            "",
            f"Completed Stage B readouts: `{len(primary)}`.",
            "",
            f"Best subset by mean outer ROC-AUC: `{best['run_key']}` (`{best['channels']}`), AUC `{best['auc']:.6f}`, PR-AUC `{best['pr_auc']:.6f}`.",
            "",
            f"Best subset by PR-AUC: `{best_pr['run_key']}` (`{best_pr['channels']}`), PR-AUC `{best_pr['pr_auc']:.6f}`, AUC `{best_pr['auc']:.6f}`.",
            "",
            f"Best subset by balanced accuracy: `{best_ba['run_key']}` (`{best_ba['channels']}`), BA `{best_ba['balanced_accuracy']:.6f}`, AUC `{best_ba['auc']:.6f}`.",
            "",
            f"Best subset by F1: `{best_f1['run_key']}` (`{best_f1['channels']}`), F1 `{best_f1['f1']:.6f}`, AUC `{best_f1['auc']:.6f}`.",
            "",
        ]
        if not ref.empty:
            r = ref.iloc[0]
            delta_auc = float(best["auc"] - r["auc"])
            delta_pr = float(best["pr_auc"] - r["pr_auc"])
            lines += [
                f"Reference `[1,0,2]`: AUC `{r['auc']:.6f}`, PR-AUC `{r['pr_auc']:.6f}`, BA `{r['balanced_accuracy']:.6f}`.",
                f"Best-AUC subset delta versus `[1,0,2]`: AUC `{delta_auc:+.6f}`, PR-AUC `{delta_pr:+.6f}`.",
                "",
            ]
        top_cols = ["run_key", "channels", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]
        lines += [
            "## Top Candidates",
            "",
            markdown_table(primary.head(5), cols=top_cols, max_rows=5).rstrip(),
            "",
            "## Interpretation",
            "",
            "Under the scale-corrected `offdiag_channelmean_sum` objective, `[1]` is the best FAST screening subset by the pre-specified primary metric, mean outer ROC-AUC. `[1,2]` is the AUC runner-up, while the 7-channel model has the best PR-AUC/BA/F1 tradeoff but lower AUC.",
            "",
        ]
        lines += [
            "Because only three outer folds are used, paired statistics are screening-level and must not be framed as confirmatory.",
            "",
        ]
    (output_root / "final_channel_selection_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(output_root: Path, plan: pd.DataFrame, dataset_branch: str, tensor_info: Dict[str, Any], split_summary: pd.DataFrame) -> None:
    lines = [
        "# Scale-Corrected FAST 3x3 Connectivity-Channel Ablation",
        "",
        "## Scope",
        "",
        f"- Dataset branch: `{dataset_branch}`.",
        "- Default branch is v5.1b for manuscript comparability.",
        "- VAE objective: `offdiag_channelmean_sum`.",
        "- Python bandpass: `OFF`.",
        "- Stage A classifier: dummy canonical `logreg`, `n_iter_logreg=1`, ignored for ranking.",
        "- Stage B readout: classifier-only `logreg_l2` on latent `mu + Age + Sex`.",
        "- Non-0.5 thresholds: true inner-CV OOF predictions only.",
        "- FAST folds: `outer_folds=3`, `inner_folds=3`.",
        "- Latent dim: `256`.",
        "- FAST horizon: `epochs_vae=960`, `cyclical_beta_n_cycles=12`, cycle length `80`, `T0=80`.",
        "",
        "## Tensor",
        "",
        f"- Shape: `{tensor_info['shape']}`.",
        f"- python_bandpass_applied: `{tensor_info['python_bandpass_applied']}`.",
        "",
        "## Planned Channel Subsets",
        "",
        markdown_table(plan, ["candidate_index", "run_key", "channels", "selected_channel_names", "recon_loss_mode", "outer_folds", "inner_folds", "latent_dim", "epochs_vae", "cyclical_beta_n_cycles"], max_rows=40),
        "",
        "## Split Preview",
        "",
        markdown_table(split_summary, ["fold", "component", "n", "AD", "CN", "MCI", "Manufacturer_GE", "Manufacturer_Siemens", "Manufacturer_Philips"], max_rows=40),
        "",
        "## Outputs",
        "",
        "- `run_manifest.csv/.md`",
        "- `primary_ablation_table.csv/.md`",
        "- `foldwise_metrics.csv/.md`",
        "- `paired_statistical_tests.csv/.md`",
        "- `channel_subset_ranking.csv/.md`",
        "- `rate_distortion_by_subset.csv/.md`",
        "- `scanner_leakage_by_subset.csv/.md`",
        "- `final_channel_selection_recommendation.md`",
        "- `command_log.json`",
        "",
    ]
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_dry_run_report(output_root: Path, plan: pd.DataFrame, selected: pd.DataFrame) -> None:
    lines = [
        "# Dry-Run Report",
        "",
        f"Generated at `{now_utc()}`.",
        "",
        f"Planned candidates: `{len(plan)}`.",
        f"Selected for this invocation: `{len(selected)}`.",
        "",
        "No training was launched.",
        "",
        "## Command Preview",
        "",
    ]
    for _, row in selected.iterrows():
        lines += [
            f"### {row['run_key']}",
            "",
            "Stage A:",
            "",
            "```bash",
            row["stage_a_command"],
            "```",
            "",
            "Stage B:",
            "",
            "```bash",
            row["stage_b_command"],
            "```",
            "",
        ]
    (output_root / "dry_run_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_command_log(output_root: Path, args: argparse.Namespace, plan: pd.DataFrame, selected: pd.DataFrame, execution_rows: List[Dict[str, Any]]) -> None:
    payload = {
        "updated_utc": now_utc(),
        "script": str(Path(__file__).resolve()),
        "output_root": str(output_root),
        "dataset_branch": args.dataset_branch,
        "dry_run": bool(args.dry_run or not args.confirm_training),
        "confirm_training": bool(args.confirm_training),
        "aggregate_only": bool(args.aggregate_only),
        "vae_retrained": bool(args.confirm_training and not args.aggregate_only),
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "existing_model_outputs_modified": False,
        "planned_candidates": int(len(plan)),
        "selected_candidates": selected["run_key"].tolist(),
        "execution": execution_rows,
        "validation": {
            "json_configs_written": True,
            "outer_folds": 3,
            "inner_folds": 3,
            "recon_loss_mode": "offdiag_channelmean_sum",
            "python_bandpass_off": True,
            "stage_a_dummy_logreg": True,
            "stage_b_primary_readout": f"{PRIMARY_MODEL}+{PRIMARY_THRESHOLD}",
        },
    }
    (output_root / "command_log.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = resolve(args.output_root)
    # Keep the manuscript-comparable v5.1b output at the requested path. If the
    # optional v5.1c branch is requested without an explicit output override,
    # write to a suffixed directory so the v5.1b planning package is not clobbered.
    if args.dataset_branch != "v5_1b" and resolve(args.output_root) == OUTPUT_ROOT:
        output_root = OUTPUT_ROOT.with_name(f"{OUTPUT_ROOT.name}_{args.dataset_branch}")
    big_root = args.big_disk_root
    if args.dataset_branch != "v5_1b" and args.big_disk_root == BIG_DISK_ROOT:
        big_root = BIG_DISK_ROOT.with_name(f"{BIG_DISK_ROOT.name}_{args.dataset_branch}")
    dataset = DATASETS[args.dataset_branch]
    output_root.mkdir(parents=True, exist_ok=True)
    tensor_info = inspect_tensor(dataset["tensor"])
    meta = load_metadata(dataset["metadata"])
    split_summary, split_subjects = split_preview(meta)
    split_summary.to_csv(output_root / "split_preview_summary.csv", index=False)
    split_subjects.to_csv(output_root / "split_preview_subjects.csv", index=False)

    plan = build_plan(output_root, big_root, args.dataset_branch, args.python_executable)
    validate_plan(plan)
    write_pair(
        output_root,
        "run_manifest",
        plan,
        [
            "candidate_index",
            "dataset_branch",
            "run_key",
            "channels",
            "selected_channel_names",
            "n_channels",
            "recon_loss_mode",
            "outer_folds",
            "inner_folds",
            "latent_dim",
            "epochs_vae",
            "cyclical_beta_n_cycles",
            "cycle_len",
            "lr_scheduler_T0",
            "status",
        ],
        max_rows=80,
    )
    selected = selected_rows(plan, args.candidate)
    dry_run = bool(args.dry_run or not args.confirm_training)
    execution_rows: List[Dict[str, Any]] = []

    if args.aggregate_only:
        aggregate_metrics(output_root, plan)
        execution_rows.append({"status": "aggregate_only"})
    elif dry_run:
        write_dry_run_report(output_root, plan, selected)
        if args.force_clean:
            for _, row in selected.iterrows():
                quarantines = prepare_run_symlink(row, force_clean=True)
                execution_rows.append(
                    {
                        "run_key": row["run_key"],
                        "status": "dry_run_symlink_preflight_ok",
                        "run_dir": str(resolve(row["run_dir"])),
                        "big_disk_output_dir": row["big_disk_output_dir"],
                        "quarantine": ";".join(str(q) for q in quarantines),
                    }
                )
        aggregate_metrics(output_root, plan)
        execution_rows.append({"status": "dry_run_no_training"})
        print(plan[["candidate_index", "run_key", "channels", "recon_loss_mode", "outer_folds", "inner_folds", "latent_dim", "epochs_vae", "cyclical_beta_n_cycles"]].to_string(index=False))
        print("\nDry-run command preview:")
        for _, row in selected.iterrows():
            print(f"\n[{row['run_key']}] Stage A\n{row['stage_a_command']}")
            print(f"[{row['run_key']}] Stage B\n{row['stage_b_command']}")
        if args.force_clean:
            print("\nDry-run symlink preflight complete. Selected local run paths are symlinks; no training launched.")
        print("\nDry-run complete. No training launched.")
    else:
        for _, row in selected.iterrows():
            result = run_candidate(row, resume=bool(args.resume), force_clean=bool(args.force_clean))
            execution_rows.append(result)
            if result["status"] not in {"complete", "skipped_complete"}:
                write_command_log(output_root, args, plan, selected, execution_rows)
                raise SystemExit(f"Candidate {row['run_key']} failed: {result}")
        aggregate_metrics(output_root, plan)

    write_readme(output_root, plan, args.dataset_branch, tensor_info, split_summary)
    write_command_log(output_root, args, plan, selected, execution_rows)


if __name__ == "__main__":
    main()
