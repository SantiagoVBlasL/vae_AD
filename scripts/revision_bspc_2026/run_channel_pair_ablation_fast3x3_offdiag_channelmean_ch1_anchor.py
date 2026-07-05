#!/usr/bin/env python3
"""Targeted ch1-anchored pair ablation under offdiag_channelmean FAST 3x3.

Default mode is dry-run / preflight only. Real training requires
--confirm-training. Existing compatible runs from the broader scale-corrected
FAST ablation are reused for [1], [1,0], and [1,2].
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = "/home/diego/anaconda3/envs/vae_ad/bin/python"
TRAINING_SCRIPT = PROJECT_ROOT / "scripts/run_vae_clf_ad_inference.py"
READOUT_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"

OUTPUT_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/channel_pair_ablation_fast3x3_offdiag_channelmean_ch1_anchor"
BIG_DISK_ROOT = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/channel_pair_ablation_fast3x3_offdiag_channelmean_ch1_anchor")
PREVIOUS_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/channel_ablation_fast3x3_offdiag_channelmean"

TENSOR_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
METADATA_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
    "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)

CHANNEL_NAMES = {
    0: "Pearson_OMST_GCE_Signed_Weighted",
    1: "Pearson_Full_FisherZ_Signed",
    2: "MI_KNN_Symmetric",
    3: "dFC_AbsDiffMean",
    4: "dFC_StdDev",
    5: "DistanceCorr",
    6: "Granger_F_lag1",
}

BASELINE_SUBSET = [1]
PAIR_SUBSETS = [[1, 0], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]]
REUSE_KEYS = {"ch1", "ch1_0", "ch1_2"}
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

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
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--big-disk-root", type=Path, default=BIG_DISK_ROOT)
    parser.add_argument("--candidate", default="all", help="Run key such as ch1_4, or 'all'.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-training", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--python-executable", default=PYTHON_EXE)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def channel_key(channels: Sequence[int]) -> str:
    return "ch" + "_".join(str(int(x)) for x in channels)


def channel_names(channels: Sequence[int]) -> list[str]:
    return [CHANNEL_NAMES[int(x)] for x in channels]


def append_arg(command: list[str], name: str, value: Any) -> None:
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


def values_after_flag(tokens: Sequence[str], flag: str) -> list[str]:
    if flag not in tokens:
        return []
    out: list[str] = []
    for token in tokens[tokens.index(flag) + 1 :]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


def require_flag_values(tokens: Sequence[str], flag: str, expected: Sequence[str], context: str) -> None:
    observed = values_after_flag(tokens, flag)
    if observed != list(expected):
        raise RuntimeError(f"{context}: expected {flag} {' '.join(expected)}, got {observed}")


def inspect_tensor(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as zf:
        shape = tuple(int(x) for x in zf["global_tensor_data"].shape)
        names = [str(x) for x in zf["channel_names"].astype(str)]
        bandpass = bool(zf["python_bandpass_applied"]) if "python_bandpass_applied" in zf.files else None
    if bandpass is not False:
        raise RuntimeError(f"Python bandpass must be OFF; got {bandpass}")
    expected = [CHANNEL_NAMES[i] for i in range(7)]
    if names[:7] != expected:
        raise RuntimeError(f"Unexpected channel order: {names[:7]}")
    return {"shape": shape, "channel_names": names, "python_bandpass_applied": bandpass}


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
    meta["Manufacturer"] = meta["Manufacturer"].map(normalize_manufacturer)
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["tensor_idx"] = meta["tensor_idx"].astype(int)
    return meta


def strat_key(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    tmp = df[list(cols)].copy()
    for col in cols:
        tmp[col] = tmp[col].fillna(f"{col}_UNKNOWN").astype(str)
    return tmp.apply(lambda row: "_".join(row.values.astype(str)), axis=1)


def count_fields(df: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {"n": int(len(df))}
    for dx in ["AD", "CN", "MCI"]:
        row[dx] = int(df["ResearchGroup_Mapped"].eq(dx).sum())
    for mfr in ["GE", "SIEMENS", "Philips"]:
        label = "Siemens" if mfr == "SIEMENS" else mfr
        row[f"Manufacturer_{label}"] = int(df["Manufacturer"].eq(mfr).sum())
    return row


def split_preview(meta: pd.DataFrame) -> pd.DataFrame:
    seed = int(FAST_PARAMS["seed"])
    rows: list[dict[str, Any]] = []
    cnad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].reset_index(drop=True)
    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    outer_key = strat_key(cnad, ["ResearchGroup_Mapped", "Manufacturer"])
    for fold, (train_idx, test_idx) in enumerate(outer.split(np.zeros(len(cnad)), outer_key), start=1):
        train_dev = cnad.iloc[train_idx].copy()
        test = cnad.iloc[test_idx].copy()
        pool = meta[~meta["tensor_idx"].isin(test["tensor_idx"])].copy()
        val_key = strat_key(pool, ["ResearchGroup_Mapped", "Manufacturer"])
        if int(val_key.value_counts().min()) < 3:
            raise RuntimeError(f"Fold {fold}: insufficient VAE val stratum count.")
        tr_pool, val_pool = train_test_split(pool, test_size=0.2, random_state=seed + fold, stratify=val_key)
        for component, frame in [
            ("classifier_train_dev", train_dev),
            ("classifier_test", test),
            ("vae_train_pool", tr_pool),
            ("vae_internal_val", val_pool),
        ]:
            row = {"fold": fold, "component": component}
            row.update(count_fields(frame))
            rows.append(row)
    return pd.DataFrame(rows)


def output_dirs(output_root: Path, big_root: Path, channels: Sequence[int]) -> dict[str, Path]:
    key = channel_key(channels)
    return {
        "run_dir": output_root / "runs" / key,
        "big_dir": big_root / key,
        "readout_dir": output_root / "runs" / key / "classifier_only_readout",
    }


def previous_dirs(channels: Sequence[int]) -> dict[str, Path]:
    key = channel_key(channels)
    return {
        "run_dir": PREVIOUS_ROOT / "runs" / key,
        "readout_dir": PREVIOUS_ROOT / "runs" / key / "classifier_only_readout",
        "config_path": PREVIOUS_ROOT / "configs" / f"{key}.json",
    }


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


def candidate_config(output_root: Path, big_root: Path, channels: Sequence[int]) -> dict[str, Any]:
    dirs = output_dirs(output_root, big_root, channels)
    params = dict(FAST_PARAMS)
    params["channels_to_use"] = list(channels)
    key = channel_key(channels)
    return {
        "run_name": f"channel_pair_ablation_fast3x3_offdiag_ch1_anchor_{key}",
        "created_utc": now_utc(),
        "run_key": key,
        "channels": list(channels),
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
            "global_tensor_path": str(TENSOR_PATH),
            "metadata_path": str(METADATA_PATH),
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
        "parameters": params,
    }


def validate_config(config: dict[str, Any]) -> None:
    params = config["parameters"]
    required = {
        "recon_loss_mode": "offdiag_channelmean_sum",
        "outer_folds": 3,
        "inner_folds": 3,
        "epochs_vae": 960,
        "cyclical_beta_n_cycles": 12,
        "lr_scheduler_T0": 80,
        "beta_vae": 2.5,
        "latent_dim": 256,
        "batch_size": 64,
        "dropout_rate_vae": 0.15,
        "vae_dropout_scope": "legacy_all",
        "vae_block_order": "legacy_act_norm",
        "vae_final_activation": "tanh",
        "intermediate_fc_dim_vae": "quarter",
        "decoder_type": "convtranspose",
        "norm_mode": "zscore_offdiag",
        "n_iter_logreg": 1,
    }
    for key, expected in required.items():
        if params.get(key) != expected:
            raise RuntimeError(f"{config['run_key']}.{key}: expected {expected!r}, got {params.get(key)!r}")
    if params["epochs_vae"] / params["cyclical_beta_n_cycles"] != 80:
        raise RuntimeError("Expected 80-epoch beta/LR cycle length.")
    if params["metadata_features"] != ["Age", "Sex"]:
        raise RuntimeError("metadata_features must be Age Sex.")
    if "Sex" in params["classifier_stratify_cols"] or "Sex" in params["vae_stratify_cols"]:
        raise RuntimeError("Sex must remain metadata/covariate only.")
    if "Manufacturer" not in params["classifier_stratify_cols"] or "Manufacturer" not in params["vae_stratify_cols"]:
        raise RuntimeError("Manufacturer-aware splits are required.")


def append_stage_arg(command: list[str], name: str, value: Any) -> None:
    append_arg(command, name, value)


def build_stage_a_command(config: dict[str, Any], python_executable: str) -> list[str]:
    paths = config["paths"]
    params = config["parameters"]
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
        append_stage_arg(command, name, params.get(name))
    validate_stage_a_command(command)
    return command


def build_stage_b_command(config: dict[str, Any], python_executable: str) -> list[str]:
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
    require_flag_values(tokens, "--epochs_vae", ["960"], "Stage A")
    require_flag_values(tokens, "--cyclical_beta_n_cycles", ["12"], "Stage A")
    require_flag_values(tokens, "--lr_scheduler_T0", ["80"], "Stage A")
    require_flag_values(tokens, "--latent_dim", ["256"], "Stage A")
    require_flag_values(tokens, "--recon_loss_mode", ["offdiag_channelmean_sum"], "Stage A")
    require_flag_values(tokens, "--metadata_features", ["Age", "Sex"], "Stage A")
    if "Manufacturer" not in values_after_flag(tokens, "--classifier_stratify_cols"):
        raise RuntimeError("Stage A classifier_stratify_cols must include Manufacturer.")
    if "Manufacturer" not in values_after_flag(tokens, "--vae_stratify_cols"):
        raise RuntimeError("Stage A vae_stratify_cols must include Manufacturer.")
    if "Sex" in values_after_flag(tokens, "--classifier_stratify_cols") or "Sex" in values_after_flag(tokens, "--vae_stratify_cols"):
        raise RuntimeError("Sex must not be a stratification column.")
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json.loads(path.read_text(encoding="utf-8"))


def build_plan(output_root: Path, big_root: Path, python_executable: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    configs_dir = output_root / "configs"
    analysis_subsets = [BASELINE_SUBSET] + PAIR_SUBSETS
    for idx, channels in enumerate(analysis_subsets, start=1):
        key = channel_key(channels)
        is_baseline = channels == BASELINE_SUBSET
        source = "reuse_previous" if key in REUSE_KEYS and readout_complete(previous_dirs(channels)["readout_dir"]) else "new_targeted_run"
        if source == "reuse_previous":
            prev = previous_dirs(channels)
            run_dir = prev["run_dir"]
            readout_dir = prev["readout_dir"]
            config_path = prev["config_path"]
            stage_a = ""
            stage_b = ""
            status = "reuse_previous_completed"
            big_dir = ""
        else:
            config = candidate_config(output_root, big_root, channels)
            validate_config(config)
            config_path = configs_dir / f"{key}.json"
            write_json(config_path, config)
            run_dir = output_dirs(output_root, big_root, channels)["run_dir"]
            readout_dir = output_dirs(output_root, big_root, channels)["readout_dir"]
            big_dir = str(output_dirs(output_root, big_root, channels)["big_dir"])
            stage_a = shlex.join(build_stage_a_command(config, python_executable))
            stage_b = shlex.join(build_stage_b_command(config, python_executable))
            status = "planned"
        rows.append(
            {
                "candidate_index": idx,
                "analysis_role": "baseline_ch1" if is_baseline else "candidate_pair",
                "run_key": key,
                "channels": json.dumps(list(channels)),
                "selected_channel_names": " | ".join(channel_names(channels)),
                "n_channels": len(channels),
                "source": source,
                "config_path": str(config_path.relative_to(PROJECT_ROOT)) if config_path.exists() or config_path.is_absolute() else str(config_path),
                "run_dir": str(run_dir.relative_to(PROJECT_ROOT)) if run_dir.is_relative_to(PROJECT_ROOT) else str(run_dir),
                "big_disk_output_dir": big_dir,
                "readout_dir": str(readout_dir.relative_to(PROJECT_ROOT)) if readout_dir.is_relative_to(PROJECT_ROOT) else str(readout_dir),
                "recon_loss_mode": FAST_PARAMS["recon_loss_mode"],
                "outer_folds": FAST_PARAMS["outer_folds"],
                "inner_folds": FAST_PARAMS["inner_folds"],
                "latent_dim": FAST_PARAMS["latent_dim"],
                "epochs_vae": FAST_PARAMS["epochs_vae"],
                "cyclical_beta_n_cycles": FAST_PARAMS["cyclical_beta_n_cycles"],
                "cycle_len": FAST_PARAMS["epochs_vae"] / FAST_PARAMS["cyclical_beta_n_cycles"],
                "lr_scheduler_T0": FAST_PARAMS["lr_scheduler_T0"],
                "stage_a_training_classifier": "canonical_logreg_dummy_ignored_for_ranking",
                "stage_b_primary_readout": f"{PRIMARY_MODEL}_{PRIMARY_THRESHOLD}",
                "threshold_selection": "true_inner_cv_oof_for_non_0p5",
                "stage_a_command": stage_a,
                "stage_b_command": stage_b,
                "status": status,
            }
        )
    plan = pd.DataFrame(rows)
    expected = ["ch1", "ch1_0", "ch1_2", "ch1_3", "ch1_4", "ch1_5", "ch1_6"]
    if plan["run_key"].tolist() != expected:
        raise RuntimeError(f"Plan order mismatch: {plan['run_key'].tolist()} != {expected}")
    return plan


def markdown_table(df: pd.DataFrame, cols: Optional[Sequence[str]] = None, max_rows: int = 80) -> str:
    if df.empty:
        return "No rows.\n"
    if cols is None:
        cols = list(df.columns)
    sub = df.loc[:, [c for c in cols if c in df.columns]].head(max_rows).copy()
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


def write_pair(output_root: Path, name: str, df: pd.DataFrame, cols: Optional[Sequence[str]] = None, max_rows: int = 120) -> None:
    df.to_csv(output_root / f"{name}.csv", index=False)
    (output_root / f"{name}.md").write_text(markdown_table(df, cols=cols, max_rows=max_rows), encoding="utf-8")


def selected_rows(plan: pd.DataFrame, candidate: str) -> pd.DataFrame:
    if candidate == "all":
        return plan.copy()
    rows = plan[plan["run_key"].eq(candidate)].copy()
    if rows.empty:
        raise RuntimeError(f"Unknown candidate={candidate!r}. Valid keys: {plan['run_key'].tolist()} or 'all'.")
    return rows


def stale_markers(run_dir: Path) -> list[Path]:
    names = ["classifier_only_readout", "latent_cache", "all_folds_metrics.csv", "summary_metrics.csv", "run_manifest.json"]
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


def create_or_update_symlink(link_path: Path, target_path: Path, force_clean: bool = False) -> list[Path]:
    quarantines: list[Path] = []
    link_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        if symlink_points_to(link_path, target_path):
            markers = stale_markers(link_path)
            if markers and not force_clean:
                raise RuntimeError(f"Refusing stale candidate outputs in {link_path}; use --force-clean.")
            if markers and force_clean:
                q = quarantine_directory_contents(target_path)
                if q is not None:
                    quarantines.append(q)
            return quarantines
        if not force_clean:
            raise RuntimeError(f"Existing symlink points elsewhere: {link_path} -> {os.readlink(link_path)}")
        q = quarantine_existing_path(link_path)
        if q is not None:
            quarantines.append(q)
    elif link_path.exists():
        if not force_clean:
            raise RuntimeError(f"Refusing to replace non-symlink path: {link_path}")
        q = quarantine_existing_path(link_path)
        if q is not None:
            quarantines.append(q)
    if not link_path.exists() and not link_path.is_symlink():
        link_path.symlink_to(target_path, target_is_directory=True)
    if not link_path.is_symlink() or not symlink_points_to(link_path, target_path):
        raise RuntimeError(f"Failed to prepare symlink: {link_path} -> {target_path}")
    return quarantines


def prepare_run_symlink(row: pd.Series, force_clean: bool) -> list[Path]:
    if row["source"] == "reuse_previous":
        return []
    run_dir = resolve(row["run_dir"])
    target = Path(row["big_disk_output_dir"])
    return create_or_update_symlink(run_dir, target, force_clean=force_clean)


def verify_fresh_checkpoints(run_dir: Path, run_start_time: float, outer_folds: int = 3) -> None:
    missing: list[str] = []
    stale: list[str] = []
    for fold in range(1, outer_folds + 1):
        ckpt = run_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt"
        if not ckpt.exists():
            missing.append(str(ckpt))
        elif ckpt.stat().st_mtime < run_start_time:
            stale.append(str(ckpt))
    if missing or stale:
        raise RuntimeError(f"Checkpoint freshness failed. Missing={missing}; stale={stale}")


def run_candidate(row: pd.Series, resume: bool, force_clean: bool) -> dict[str, Any]:
    if row["source"] == "reuse_previous":
        return {"run_key": row["run_key"], "status": "reused_previous_completed", "readout_dir": row["readout_dir"]}
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
        return {"run_key": row["run_key"], "status": "stage_a_failed", "returncode": completed_a.returncode}
    verify_fresh_checkpoints(run_dir, run_start_time, outer_folds=int(row["outer_folds"]))
    completed_b = subprocess.run(stage_b_cmd, cwd=str(PROJECT_ROOT), check=False)
    if completed_b.returncode != 0:
        return {"run_key": row["run_key"], "status": "stage_b_failed", "returncode": completed_b.returncode}
    if not readout_complete(readout_dir):
        return {"run_key": row["run_key"], "status": "stage_b_incomplete"}
    return {"run_key": row["run_key"], "status": "complete", "run_dir": str(run_dir), "readout_dir": str(readout_dir), "quarantine": ";".join(str(q) for q in quarantines)}


def read_primary(row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    readout = resolve(row["readout_dir"])
    pooled_path = readout / "classifier_sweep_pooled_metrics.csv"
    foldwise_path = readout / "classifier_sweep_foldwise_metrics.csv"
    subgroup_path = readout / "classifier_sweep_subgroup_metrics_by_manufacturer.csv"
    if not pooled_path.exists() or not foldwise_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    pooled = pd.read_csv(pooled_path)
    foldwise = pd.read_csv(foldwise_path)
    subgroup = pd.read_csv(subgroup_path) if subgroup_path.exists() else pd.DataFrame()
    for df in [pooled, foldwise, subgroup]:
        if not df.empty:
            df["run_key"] = row["run_key"]
            df["channels"] = row["channels"]
            df["selected_channel_names"] = row["selected_channel_names"]
            df["n_channels"] = row["n_channels"]
            df["source"] = row["source"]
    return pooled, foldwise, subgroup


def collect_rate_distortion(plan: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in plan.iterrows():
        run_dir = resolve(row["run_dir"])
        for fold in range(1, int(row["outer_folds"]) + 1):
            path = run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if df.empty or "epoch" not in df.columns:
                continue
            best_idx = pd.to_numeric(df["L_val_betaMax"], errors="coerce").idxmin() if "L_val_betaMax" in df.columns else pd.to_numeric(df["D_val"], errors="coerce").idxmin()
            best = df.loc[best_idx]
            d_val = float(best.get("D_val", np.nan))
            r_val = float(best.get("R_val_nats", np.nan))
            rows.append(
                {
                    "run_key": row["run_key"],
                    "channels": row["channels"],
                    "selected_channel_names": row["selected_channel_names"],
                    "source": row["source"],
                    "fold": fold,
                    "best_epoch": int(best["epoch"]),
                    "final_epoch": int(pd.to_numeric(df["epoch"], errors="coerce").max()),
                    "best_val_l_beta_max": float(best.get("L_val_betaMax", np.nan)),
                    "best_val_recon_D": d_val,
                    "best_val_kld_R_nats": r_val,
                    "best_val_kld_over_recon": r_val / d_val if np.isfinite(d_val) and d_val else np.nan,
                    "best_val_beta_kld_over_recon": 2.5 * r_val / d_val if np.isfinite(d_val) and d_val else np.nan,
                }
            )
    return pd.DataFrame(rows)


def collect_scanner_leakage(plan: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in plan.iterrows():
        run_dir = resolve(row["run_dir"])
        for fold in range(1, int(row["outer_folds"]) + 1):
            path = run_dir / f"fold_{fold}" / "latent_qc_metrics.csv"
            if not path.exists():
                path = run_dir / f"fold_{fold}" / f"fold_{fold}_test_scanner_leakage_summary.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            rec = df.iloc[0]
            raw = rec.get("acc_site_raw", rec.get("acc_raw", np.nan))
            lat = rec.get("acc_site_latent", rec.get("acc_latent", np.nan))
            raw = float(raw) if pd.notna(raw) else np.nan
            lat = float(lat) if pd.notna(lat) else np.nan
            rows.append(
                {
                    "run_key": row["run_key"],
                    "channels": row["channels"],
                    "selected_channel_names": row["selected_channel_names"],
                    "source": row["source"],
                    "fold": fold,
                    "acc_raw": raw,
                    "acc_latent": lat,
                    "latent_minus_raw": lat - raw if np.isfinite(raw) and np.isfinite(lat) else np.nan,
                    "chance_level": rec.get("chance_level", np.nan),
                }
            )
    return pd.DataFrame(rows)


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


def paired_tests_vs_ch1(foldwise: pd.DataFrame) -> pd.DataFrame:
    if foldwise.empty:
        return pd.DataFrame()
    try:
        from scipy import stats
    except Exception:
        stats = None
    ref = foldwise[foldwise["run_key"].eq("ch1")]
    rows: list[dict[str, Any]] = []
    metrics = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]
    for run_key, sub in foldwise.groupby("run_key"):
        if run_key == "ch1":
            continue
        merged = ref[["fold", *metrics]].merge(sub[["fold", *metrics]], on="fold", suffixes=("_ch1", "_pair"))
        for metric in metrics:
            delta = merged[f"{metric}_pair"].to_numpy(dtype=float) - merged[f"{metric}_ch1"].to_numpy(dtype=float)
            mean_delta = float(np.nanmean(delta)) if len(delta) else np.nan
            sd_delta = float(np.nanstd(delta, ddof=1)) if len(delta) > 1 else np.nan
            dz = mean_delta / sd_delta if np.isfinite(sd_delta) and sd_delta > 0 else np.nan
            if stats is not None and len(delta) >= 2 and np.isfinite(delta).all():
                try:
                    t_p = float(stats.ttest_rel(merged[f"{metric}_pair"], merged[f"{metric}_ch1"]).pvalue)
                except Exception:
                    t_p = np.nan
                try:
                    w_p = float(stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue)
                except Exception:
                    w_p = np.nan
            else:
                t_p = np.nan
                w_p = np.nan
            rows.append(
                {
                    "candidate_pair": run_key,
                    "reference": "ch1",
                    "metric": metric,
                    "n_paired_folds": int(len(delta)),
                    "mean_delta_pair_minus_ch1": mean_delta,
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


def aggregate(output_root: Path, plan: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pooled_rows: list[pd.DataFrame] = []
    foldwise_rows: list[pd.DataFrame] = []
    subgroup_rows: list[pd.DataFrame] = []
    for _, row in plan.iterrows():
        pooled, foldwise, subgroup = read_primary(row)
        if not pooled.empty:
            pooled_rows.append(pooled)
        if not foldwise.empty:
            foldwise_rows.append(foldwise)
        if not subgroup.empty:
            subgroup_rows.append(subgroup)
    pooled = pd.concat(pooled_rows, ignore_index=True, sort=False) if pooled_rows else pd.DataFrame()
    foldwise = pd.concat(foldwise_rows, ignore_index=True, sort=False) if foldwise_rows else pd.DataFrame()
    subgroup = pd.concat(subgroup_rows, ignore_index=True, sort=False) if subgroup_rows else pd.DataFrame()
    primary = pooled[(pooled["model_name"].eq(PRIMARY_MODEL)) & (pooled["threshold_strategy"].eq(PRIMARY_THRESHOLD))].copy() if not pooled.empty else pd.DataFrame()
    foldwise_primary = foldwise[(foldwise["model_name"].eq(PRIMARY_MODEL)) & (foldwise["threshold_strategy"].eq(PRIMARY_THRESHOLD))].copy() if not foldwise.empty else pd.DataFrame()
    primary = primary.sort_values(["auc", "pr_auc", "balanced_accuracy"], ascending=False) if not primary.empty else primary
    if not primary.empty:
        primary["rank_auc"] = primary["auc"].rank(ascending=False, method="min").astype(int)
        primary["rank_pr_auc"] = primary["pr_auc"].rank(ascending=False, method="min").astype(int)
    ranking = primary.copy()
    paired = paired_tests_vs_ch1(foldwise_primary)
    leakage = collect_scanner_leakage(plan)
    rd = collect_rate_distortion(plan)
    write_pair(output_root, "primary_pair_table", primary, ["run_key", "channels", "selected_channel_names", "source", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "tn", "fp", "fn", "tp"], 120)
    write_pair(output_root, "pair_ranking", ranking, ["rank_auc", "rank_pr_auc", "run_key", "channels", "selected_channel_names", "source", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"], 120)
    write_pair(output_root, "foldwise_pair_metrics", foldwise_primary.sort_values(["run_key", "fold"]) if not foldwise_primary.empty else foldwise_primary, ["run_key", "channels", "source", "fold", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "tn", "fp", "fn", "tp"], 200)
    write_pair(output_root, "manufacturer_subgroup_by_pair", subgroup, max_rows=200)
    write_pair(output_root, "paired_tests_vs_ch1", paired, max_rows=200)
    write_pair(output_root, "scanner_leakage_by_pair", leakage, max_rows=200)
    write_pair(output_root, "rate_distortion_by_pair", rd, max_rows=200)
    write_recommendation(output_root, primary, leakage)
    return {"primary": primary, "foldwise": foldwise_primary, "paired": paired, "scanner": leakage, "rate_distortion": rd}


def write_recommendation(output_root: Path, primary: pd.DataFrame, leakage: pd.DataFrame) -> None:
    lines = [
        "# Final Pair Selection Recommendation",
        "",
        "This targeted FAST 3x3 analysis is a screening step, not a final performance estimate.",
        "",
    ]
    if primary.empty:
        lines += ["No completed Stage B readouts were available.", ""]
    else:
        completed = set(primary["run_key"].astype(str))
        missing = [channel_key(ch) for ch in PAIR_SUBSETS if channel_key(ch) not in completed]
        baseline = primary[primary["run_key"].eq("ch1")]
        pairs = primary[primary["run_key"].ne("ch1")].copy()
        if baseline.empty:
            lines += ["Baseline `[1]` readout was not available, so pair promotion cannot be evaluated.", ""]
        else:
            ch1 = baseline.iloc[0]
            best_pair = pairs.sort_values(["auc", "pr_auc", "balanced_accuracy"], ascending=False).iloc[0] if not pairs.empty else None
            lines += [
                f"Completed readouts: `{len(primary)}`. Missing pair readouts: `{', '.join(missing) if missing else 'none'}`.",
                "",
                f"Baseline `[1]`: AUC `{ch1['auc']:.6f}`, PR-AUC `{ch1['pr_auc']:.6f}`, BA `{ch1['balanced_accuracy']:.6f}`, F1 `{ch1['f1']:.6f}`.",
                "",
            ]
            if best_pair is not None:
                d_auc = float(best_pair["auc"] - ch1["auc"])
                d_pr = float(best_pair["pr_auc"] - ch1["pr_auc"])
                d_ba = float(best_pair["balanced_accuracy"] - ch1["balanced_accuracy"])
                d_f1 = float(best_pair["f1"] - ch1["f1"])
                pass_rule = d_auc >= 0.015 and d_pr >= 0.0 and d_ba >= -0.005 and d_f1 >= -0.005 and not missing
                lines += [
                    f"Best completed pair by AUC: `{best_pair['run_key']}` (`{best_pair['channels']}`), AUC `{best_pair['auc']:.6f}`, PR-AUC `{best_pair['pr_auc']:.6f}`.",
                    f"Delta versus `[1]`: AUC `{d_auc:+.6f}`, PR-AUC `{d_pr:+.6f}`, BA `{d_ba:+.6f}`, F1 `{d_f1:+.6f}`.",
                    "",
                ]
                if missing:
                    lines += [
                        "Recommendation: **do not recommend a FULL confirmation yet** because not all ch1-anchored pairs have completed.",
                        "",
                    ]
                elif pass_rule:
                    lines += [
                        f"Recommendation: `{best_pair['run_key']}` passes the FAST promotion rule and may justify one controlled FULL confirmation, pending scanner/manufacturer leakage review.",
                        "",
                    ]
                else:
                    lines += [
                        "Recommendation: no pair passes the pre-specified FAST promotion rule versus `[1]`; do not launch a FULL pair confirmation.",
                        "",
                    ]
        if not leakage.empty:
            leak_summary = leakage.groupby("run_key", as_index=False)[["acc_raw", "acc_latent", "latent_minus_raw"]].mean(numeric_only=True)
            lines += ["## Mean Scanner/Manufacturer Leakage", "", markdown_table(leak_summary, max_rows=20), ""]
    (output_root / "final_pair_selection_recommendation.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme(output_root: Path, plan: pd.DataFrame, tensor_info: dict[str, Any], split_summary: pd.DataFrame) -> None:
    lines = [
        "# Targeted Ch1-Anchored FAST Pair Ablation",
        "",
        "## Scope",
        "",
        "- Dataset: v5.1b final/manuscript-comparable.",
        "- Candidate pairs: `[1,0]`, `[1,2]`, `[1,3]`, `[1,4]`, `[1,5]`, `[1,6]`.",
        "- Baseline for decision: `[1]` Pearson Full.",
        "- Existing compatible `[1]`, `[1,0]`, and `[1,2]` readouts are reused from `channel_ablation_fast3x3_offdiag_channelmean`.",
        "- New candidates are default-safe and require `--confirm-training`.",
        "- Stage A canonical logreg is dummy/ignored; Stage B `logreg_l2` with true inner-CV OOF thresholding is the only ranking readout.",
        "- Python bandpass: OFF.",
        "",
        "## Tensor",
        "",
        f"- Shape: `{tensor_info['shape']}`.",
        f"- python_bandpass_applied: `{tensor_info['python_bandpass_applied']}`.",
        "",
        "## Planned/Reused Runs",
        "",
        markdown_table(plan, ["candidate_index", "analysis_role", "run_key", "channels", "selected_channel_names", "source", "status", "outer_folds", "inner_folds", "epochs_vae", "cyclical_beta_n_cycles"], 20),
        "",
        "## Split Preview",
        "",
        markdown_table(split_summary, ["fold", "component", "n", "AD", "CN", "MCI", "Manufacturer_GE", "Manufacturer_Siemens", "Manufacturer_Philips"], 40),
        "",
    ]
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_dry_run_report(output_root: Path, selected: pd.DataFrame) -> None:
    lines = ["# Dry-Run Report", "", f"Generated at `{now_utc()}`.", "", "No training was launched.", ""]
    for _, row in selected.iterrows():
        lines += [f"## {row['run_key']}", "", f"Source: `{row['source']}`.", ""]
        if row["source"] == "reuse_previous":
            lines += [f"Reusing readout: `{row['readout_dir']}`.", ""]
        else:
            lines += ["Stage A:", "", "```bash", row["stage_a_command"], "```", "", "Stage B:", "", "```bash", row["stage_b_command"], "```", ""]
    (output_root / "dry_run_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_command_log(output_root: Path, args: argparse.Namespace, plan: pd.DataFrame, selected: pd.DataFrame, execution_rows: list[dict[str, Any]]) -> None:
    payload = {
        "updated_utc": now_utc(),
        "script": str(Path(__file__).resolve()),
        "output_root": str(output_root),
        "dry_run": bool(args.dry_run or not args.confirm_training),
        "confirm_training": bool(args.confirm_training),
        "aggregate_only": bool(args.aggregate_only),
        "vae_retrained": bool(args.confirm_training and not args.aggregate_only),
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "existing_model_outputs_modified": False,
        "planned_rows": int(len(plan)),
        "candidate_pairs": [channel_key(ch) for ch in PAIR_SUBSETS],
        "baseline": "ch1",
        "selected": selected["run_key"].tolist(),
        "execution": execution_rows,
        "validation": {
            "outer_folds": 3,
            "inner_folds": 3,
            "epochs_vae": 960,
            "cyclical_beta_n_cycles": 12,
            "cycle_len": 80,
            "lr_scheduler_T0": 80,
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
    output_root.mkdir(parents=True, exist_ok=True)
    tensor_info = inspect_tensor(TENSOR_PATH)
    split_summary = split_preview(load_metadata(METADATA_PATH))
    split_summary.to_csv(output_root / "split_preview_summary.csv", index=False)
    plan = build_plan(output_root, args.big_disk_root, args.python_executable)
    write_pair(output_root, "run_manifest", plan, ["candidate_index", "analysis_role", "run_key", "channels", "selected_channel_names", "source", "status", "recon_loss_mode", "outer_folds", "inner_folds", "latent_dim", "epochs_vae", "cyclical_beta_n_cycles", "cycle_len", "lr_scheduler_T0"], 30)
    selected = selected_rows(plan, args.candidate)
    dry_run = bool(args.dry_run or not args.confirm_training)
    execution_rows: list[dict[str, Any]] = []

    if args.aggregate_only:
        aggregate(output_root, plan)
        execution_rows.append({"status": "aggregate_only"})
    elif dry_run:
        write_dry_run_report(output_root, selected)
        aggregate(output_root, plan)
        execution_rows.append({"status": "dry_run_no_training"})
        print(plan[["candidate_index", "analysis_role", "run_key", "channels", "source", "status", "outer_folds", "inner_folds", "epochs_vae", "cyclical_beta_n_cycles"]].to_string(index=False))
        print("\nDry-run command preview:")
        for _, row in selected.iterrows():
            print(f"\n[{row['run_key']}] source={row['source']}")
            if row["source"] == "reuse_previous":
                print(f"reuse readout: {row['readout_dir']}")
            else:
                print(f"Stage A\n{row['stage_a_command']}")
                print(f"Stage B\n{row['stage_b_command']}")
        print("\nDry-run complete. No training launched.")
    else:
        for _, row in selected.iterrows():
            result = run_candidate(row, resume=bool(args.resume), force_clean=bool(args.force_clean))
            execution_rows.append(result)
            if result["status"] not in {"complete", "skipped_complete", "reused_previous_completed"}:
                write_command_log(output_root, args, plan, selected, execution_rows)
                raise SystemExit(f"Candidate {row['run_key']} failed: {result}")
        aggregate(output_root, plan)

    write_readme(output_root, plan, tensor_info, split_summary)
    write_command_log(output_root, args, plan, selected, execution_rows)


if __name__ == "__main__":
    main()
